import copy
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


def softplus(x):
    if x > 20:
        return x
    if x < -20:
        return 0.0
    return math.log1p(math.exp(x))


def xielu_ref_scalar(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    """Scalar reference implementation in Python."""
    s_ap = softplus(alpha_p)
    s_an = softplus(alpha_n)
    if x > 0:
        return x * (s_ap * x + beta)
    else:
        return (beta + s_an) * (math.exp(min(x, eps)) - 1) - s_an * x


def xielu_ref(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    """Tensor reference implementation using PyTorch ops."""
    s_ap = F.softplus(alpha_p)
    s_an = F.softplus(alpha_n)
    pos = x * (s_ap * x + beta)
    neg = (beta + s_an) * torch.expm1(x.clamp(max=eps)) - s_an * x
    return torch.where(x > 0, pos, neg)


# =====================================================================
# Forward correctness
# =====================================================================


class TestXieluForward(TestCase):
    """Forward-pass correctness on CPU."""

    def _check_elementwise(self, x_vals, ap_val, an_val, beta=0.5, eps=-1e-6):
        x = torch.tensor(x_vals)
        ap = torch.tensor(ap_val)
        an = torch.tensor(an_val)
        out = F.xielu(x, ap, an, beta, eps)
        for i, xi in enumerate(x_vals):
            expected = xielu_ref_scalar(xi, ap_val, an_val, beta, eps)
            self.assertAlmostEqual(out[i].item(), expected, places=5)

    def test_positive_inputs(self):
        self._check_elementwise([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 0.5, -0.3)

    def test_negative_inputs(self):
        self._check_elementwise([-0.1, -0.5, -1.0, -2.0, -5.0, -10.0], 0.5, -0.3)

    def test_zero(self):
        self._check_elementwise([0.0], 0.5, -0.3)

    def test_mixed(self):
        self._check_elementwise([-3.0, -1.0, -0.01, 0.0, 0.01, 1.0, 3.0], 0.5, -0.3)

    def test_default_alphas_zero(self):
        # alpha_p = alpha_n = 0  =>  softplus(0) = ln(2)
        self._check_elementwise([-2.0, -0.5, 0.0, 0.5, 2.0], 0.0, 0.0)

    def test_large_positive_alphas(self):
        self._check_elementwise([-1.0, 0.0, 1.0], 5.0, 5.0)

    def test_large_negative_alphas(self):
        self._check_elementwise([-1.0, 0.0, 1.0], -5.0, -5.0)

    def test_custom_beta(self):
        self._check_elementwise([-1.0, 0.0, 1.0], 0.5, -0.3, beta=1.0)
        self._check_elementwise([-1.0, 0.0, 1.0], 0.5, -0.3, beta=0.0)

    def test_custom_eps(self):
        self._check_elementwise([-1.0, -0.5, 0.0, 0.5], 0.5, -0.3, eps=-0.5)

    def test_very_negative_input_linear_region(self):
        # For x << eps, exp(eps) dominates and the function is approximately linear:
        # f(x) ≈ (beta + s_an) * (exp(eps) - 1) - s_an * x
        self._check_elementwise([-50.0, -100.0], 0.0, 0.0)

    def test_matches_tensor_reference(self):
        torch.manual_seed(42)
        x = torch.randn(256)
        ap = torch.tensor(0.7)
        an = torch.tensor(-0.4)
        out = F.xielu(x, ap, an, 0.5, -1e-6)
        ref = xielu_ref(x, ap, an, 0.5, -1e-6)
        self.assertEqual(out, ref, atol=1e-6, rtol=1e-6)

    def test_output_shape_preserved(self):
        for shape in [(), (1,), (4,), (2, 3), (2, 3, 4), (1, 1, 1, 1)]:
            x = torch.randn(shape)
            ap = torch.tensor(0.0)
            an = torch.tensor(0.0)
            out = F.xielu(x, ap, an)
            self.assertEqual(out.shape, x.shape)

    def test_empty_tensor(self):
        x = torch.empty(0)
        ap = torch.tensor(0.0)
        an = torch.tensor(0.0)
        out = F.xielu(x, ap, an)
        self.assertEqual(out.shape, torch.Size([0]))
        self.assertEqual(out.numel(), 0)

    def test_float64_precision(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float64)
        ap = torch.tensor(0.5, dtype=torch.float64)
        an = torch.tensor(-0.3, dtype=torch.float64)
        out = F.xielu(x, ap, an)
        ref = xielu_ref(x, ap, an)
        self.assertEqual(out, ref, atol=1e-12, rtol=1e-12)


# =====================================================================
# Backward / gradient correctness
# =====================================================================


class TestXieluBackward(TestCase):
    """Gradient correctness on CPU using autograd.gradcheck."""

    def test_gradcheck_float64(self):
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an, 0.5, -1e-6),
            (x, ap, an),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_positive_only(self):
        x = torch.rand(8, dtype=torch.float64, requires_grad=True) + 0.1
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an),
            (x, ap, an),
        )

    def test_gradcheck_negative_only(self):
        x = -torch.rand(8, dtype=torch.float64, requires_grad=True) - 0.1
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an),
            (x, ap, an),
        )

    def test_gradcheck_custom_beta_eps(self):
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an, beta=1.0, eps=-0.5),
            (x, ap, an),
        )

    @unittest.expectedFailure
    def test_gradgradcheck(self):
        # Second-order gradients not yet implemented (no derivative for xielu_backward)
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            lambda x, ap, an: F.xielu(x, ap, an, 0.5, -1e-6),
            (x, ap, an),
            eps=1e-6,
            atol=1e-3,
        )

    def test_grad_shapes(self):
        x = torch.randn(4, 8, requires_grad=True)
        ap = torch.tensor(0.5, requires_grad=True)
        an = torch.tensor(-0.3, requires_grad=True)
        out = F.xielu(x, ap, an)
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(ap.grad.shape, ap.shape)
        self.assertEqual(an.grad.shape, an.shape)

    def test_grad_when_alpha_not_requires_grad(self):
        x = torch.randn(8, requires_grad=True)
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(ap.grad)
        self.assertIsNone(an.grad)

    def test_grad_when_input_not_requires_grad(self):
        x = torch.randn(8)
        ap = torch.tensor(0.5, requires_grad=True)
        an = torch.tensor(-0.3, requires_grad=True)
        out = F.xielu(x, ap, an)
        out.sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNotNone(ap.grad)
        self.assertIsNotNone(an.grad)

    def test_backward_matches_reference(self):
        torch.manual_seed(123)
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)

        x_ref = x.detach().clone().requires_grad_(True)
        ap_ref = ap.detach().clone().requires_grad_(True)
        an_ref = an.detach().clone().requires_grad_(True)

        F.xielu(x, ap, an).sum().backward()
        xielu_ref(x_ref, ap_ref, an_ref).sum().backward()

        self.assertEqual(x.grad, x_ref.grad, atol=1e-10, rtol=1e-10)
        self.assertEqual(ap.grad, ap_ref.grad, atol=1e-10, rtol=1e-10)
        self.assertEqual(an.grad, an_ref.grad, atol=1e-10, rtol=1e-10)

    def test_alpha_p_grad_positive_only(self):
        # alpha_p gradient should only come from positive inputs
        x = torch.tensor([-2.0, -1.0], requires_grad=False)
        ap = torch.tensor(0.5, requires_grad=True)
        an = torch.tensor(-0.3, requires_grad=True)
        F.xielu(x, ap, an).sum().backward()
        self.assertEqual(ap.grad.item(), 0.0)

    def test_alpha_n_grad_negative_only(self):
        # alpha_n gradient should only come from non-positive inputs
        x = torch.tensor([1.0, 2.0], requires_grad=False)
        ap = torch.tensor(0.5, requires_grad=True)
        an = torch.tensor(-0.3, requires_grad=True)
        F.xielu(x, ap, an).sum().backward()
        self.assertEqual(an.grad.item(), 0.0)


# =====================================================================
# nn.Module tests
# =====================================================================


class TestXieluModule(TestCase):
    """Tests for the nn.xIELU module."""

    def test_parameters(self):
        m = nn.xIELU()
        params = list(m.parameters())
        self.assertEqual(len(params), 2)
        self.assertEqual(m.alpha_p.shape, torch.Size([]))
        self.assertEqual(m.alpha_n.shape, torch.Size([]))

    def test_init_zeros(self):
        m = nn.xIELU()
        self.assertEqual(m.alpha_p.item(), 0.0)
        self.assertEqual(m.alpha_n.item(), 0.0)

    def test_forward_matches_functional(self):
        torch.manual_seed(42)
        m = nn.xIELU(beta=0.7, eps=-0.001)
        x = torch.randn(4, 8)
        out_module = m(x)
        out_fn = F.xielu(x, m.alpha_p, m.alpha_n, 0.7, -0.001)
        self.assertEqual(out_module, out_fn)

    def test_backward(self):
        m = nn.xIELU()
        x = torch.randn(4, 8, requires_grad=True)
        out = m(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(m.alpha_p.grad)
        self.assertIsNotNone(m.alpha_n.grad)

    def test_extra_repr(self):
        m = nn.xIELU(beta=0.7, eps=-2e-6)
        r = m.extra_repr()
        self.assertIn("beta=0.7", r)
        self.assertIn("eps=-2e-06", r)

    def test_extra_repr_defaults(self):
        m = nn.xIELU()
        r = m.extra_repr()
        self.assertIn("beta=0.5", r)
        self.assertIn("eps=-1e-06", r)

    def test_dtype_kwarg(self):
        for dtype in [torch.float32, torch.float64]:
            m = nn.xIELU(dtype=dtype)
            self.assertEqual(m.alpha_p.dtype, dtype)
            self.assertEqual(m.alpha_n.dtype, dtype)

    def test_state_dict_roundtrip(self):
        m1 = nn.xIELU(beta=0.7, eps=-0.001)
        with torch.no_grad():
            m1.alpha_p.fill_(1.5)
            m1.alpha_n.fill_(-0.8)
        sd = m1.state_dict()
        m2 = nn.xIELU(beta=0.7, eps=-0.001)
        m2.load_state_dict(sd)
        self.assertEqual(m2.alpha_p.item(), 1.5)
        self.assertEqual(m2.alpha_n.item(), -0.8)

    def test_deepcopy(self):
        m1 = nn.xIELU()
        with torch.no_grad():
            m1.alpha_p.fill_(2.0)
        m2 = copy.deepcopy(m1)
        self.assertEqual(m2.alpha_p.item(), 2.0)
        # Verify they are independent
        with torch.no_grad():
            m1.alpha_p.fill_(0.0)
        self.assertEqual(m2.alpha_p.item(), 2.0)

    def test_repr(self):
        m = nn.xIELU()
        r = repr(m)
        self.assertIn("xIELU", r)

    def test_training_mode(self):
        m = nn.xIELU()
        x = torch.randn(4, 8)
        m.train()
        out_train = m(x)
        m.eval()
        out_eval = m(x)
        # xIELU has no train/eval distinction
        self.assertEqual(out_train, out_eval)

    def test_in_sequential(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.xIELU(), nn.Linear(16, 4))
        x = torch.randn(2, 8)
        out = model(x)
        self.assertEqual(out.shape, (2, 4))
        out.sum().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)


# =====================================================================
# Non-contiguous and special tensor tests
# =====================================================================


class TestXieluTensorLayout(TestCase):
    """Tests for non-contiguous tensors and special layouts."""

    def test_non_contiguous_input(self):
        x_full = torch.randn(8, 8)
        x = x_full[::2, ::2]  # stride != 1
        self.assertFalse(x.is_contiguous())
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        ref = F.xielu(x.contiguous(), ap, an)
        self.assertEqual(out, ref)

    def test_non_contiguous_backward(self):
        x_full = torch.randn(8, 8, dtype=torch.float64)
        x = x_full[::2, ::2].requires_grad_(True)
        ap = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an),
            (x, ap, an),
        )

    def test_transposed_input(self):
        x = torch.randn(4, 8).t()
        self.assertFalse(x.is_contiguous())
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        ref = F.xielu(x.contiguous(), ap, an)
        self.assertEqual(out, ref)

    def test_scalar_tensor_input(self):
        x = torch.tensor(1.5)
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        expected = xielu_ref_scalar(1.5, 0.5, -0.3)
        self.assertAlmostEqual(out.item(), expected, places=5)

    def test_large_tensor(self):
        x = torch.randn(1024, 1024)
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        ref = xielu_ref(x, ap, an)
        self.assertEqual(out, ref, atol=1e-5, rtol=1e-5)


# =====================================================================
# Error handling
# =====================================================================


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestXieluErrors(TestCase):
    """Tests for proper error handling (CUDA path validates scalar alpha)."""

    def test_alpha_p_must_be_scalar_cuda(self):
        x = torch.randn(4, device="cuda")
        ap = torch.randn(4, device="cuda")
        an = torch.tensor(0.0, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "alpha_p must be a scalar"):
            F.xielu(x, ap, an)

    def test_alpha_n_must_be_scalar_cuda(self):
        x = torch.randn(4, device="cuda")
        ap = torch.tensor(0.0, device="cuda")
        an = torch.randn(4, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "alpha_n must be a scalar"):
            F.xielu(x, ap, an)


# =====================================================================
# CUDA tests
# =====================================================================


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestXieluCUDA(TestCase):
    """CUDA forward/backward tests."""

    def test_cuda_float32_forward(self):
        torch.manual_seed(42)
        x = torch.randn(128, device="cuda")
        ap = torch.tensor(0.5, device="cuda")
        an = torch.tensor(-0.3, device="cuda")
        out = F.xielu(x, ap, an)
        ref = xielu_ref(x, ap, an)
        self.assertEqual(out, ref, atol=1e-5, rtol=1e-5)

    def test_cuda_float32_backward(self):
        x = torch.randn(64, device="cuda", requires_grad=True)
        ap = torch.tensor(0.5, device="cuda", requires_grad=True)
        an = torch.tensor(-0.3, device="cuda", requires_grad=True)
        out = F.xielu(x, ap, an)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ap.grad)
        self.assertIsNotNone(an.grad)

    def test_cuda_float32_matches_cpu(self):
        torch.manual_seed(42)
        x = torch.randn(256)
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out_cpu = F.xielu(x, ap, an)
        out_cuda = F.xielu(x.cuda(), ap.cuda(), an.cuda())
        self.assertEqual(out_cpu, out_cuda.cpu(), atol=1e-5, rtol=1e-5)

    def test_cuda_float32_backward_matches_cpu(self):
        torch.manual_seed(42)
        x_cpu = torch.randn(64, requires_grad=True)
        ap_cpu = torch.tensor(0.5, requires_grad=True)
        an_cpu = torch.tensor(-0.3, requires_grad=True)

        x_cuda = x_cpu.detach().clone().cuda().requires_grad_(True)
        ap_cuda = ap_cpu.detach().clone().cuda().requires_grad_(True)
        an_cuda = an_cpu.detach().clone().cuda().requires_grad_(True)

        F.xielu(x_cpu, ap_cpu, an_cpu).sum().backward()
        F.xielu(x_cuda, ap_cuda, an_cuda).sum().backward()

        self.assertEqual(x_cpu.grad, x_cuda.grad.cpu(), atol=1e-5, rtol=1e-5)
        self.assertEqual(ap_cpu.grad, ap_cuda.grad.cpu(), atol=1e-4, rtol=1e-4)
        self.assertEqual(an_cpu.grad, an_cuda.grad.cpu(), atol=1e-4, rtol=1e-4)

    def test_cuda_float64(self):
        x = torch.randn(64, device="cuda", dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, device="cuda", dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, device="cuda", dtype=torch.float64, requires_grad=True)
        out = F.xielu(x, ap, an)
        ref = xielu_ref(x, ap, an)
        self.assertEqual(out, ref, atol=1e-12, rtol=1e-12)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_cuda_gradcheck_float64(self):
        x = torch.randn(4, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, device="cuda", dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, device="cuda", dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, ap, an: F.xielu(x, ap, an),
            (x, ap, an),
        )

    @unittest.expectedFailure
    def test_cuda_gradgradcheck_float64(self):
        # Second-order gradients not yet implemented (no derivative for xielu_backward)
        x = torch.randn(4, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        ap = torch.tensor(0.5, device="cuda", dtype=torch.float64, requires_grad=True)
        an = torch.tensor(-0.3, device="cuda", dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            lambda x, ap, an: F.xielu(x, ap, an),
            (x, ap, an),
            atol=1e-3,
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestXieluCUDAHalfPrecision(TestCase):
    """Tests for float16 and bfloat16 CUDA kernels."""

    def _test_forward_dtype(self, dtype):
        torch.manual_seed(42)
        x = torch.randn(128, device="cuda", dtype=dtype)
        ap = torch.tensor(0.5, device="cuda", dtype=dtype)
        an = torch.tensor(-0.3, device="cuda", dtype=dtype)
        out = F.xielu(x, ap, an)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.shape, x.shape)

    def _test_backward_dtype(self, dtype):
        x = torch.randn(64, device="cuda", dtype=dtype, requires_grad=True)
        ap = torch.tensor(0.5, device="cuda", dtype=dtype, requires_grad=True)
        an = torch.tensor(-0.3, device="cuda", dtype=dtype, requires_grad=True)
        out = F.xielu(x, ap, an)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ap.grad)
        self.assertIsNotNone(an.grad)
        self.assertEqual(x.grad.dtype, dtype)
        self.assertEqual(ap.grad.dtype, dtype)
        self.assertEqual(an.grad.dtype, dtype)

    def _test_matches_float32(self, dtype, atol=2e-2, rtol=2e-2):
        """Half-precision kernel should match float32 kernel within tolerance."""
        torch.manual_seed(42)
        x_f32 = torch.randn(128, device="cuda")
        ap_f32 = torch.tensor(0.5, device="cuda")
        an_f32 = torch.tensor(-0.3, device="cuda")

        x_hp = x_f32.to(dtype)
        ap_hp = ap_f32.to(dtype)
        an_hp = an_f32.to(dtype)

        out_f32 = F.xielu(x_hp.float(), ap_hp.float(), an_hp.float())
        out_hp = F.xielu(x_hp, ap_hp, an_hp)
        self.assertEqual(out_f32.to(dtype), out_hp, atol=atol, rtol=rtol)

    def test_bf16_forward(self):
        self._test_forward_dtype(torch.bfloat16)

    def test_bf16_backward(self):
        self._test_backward_dtype(torch.bfloat16)

    def test_bf16_matches_float32(self):
        self._test_matches_float32(torch.bfloat16)

    def test_fp16_forward(self):
        self._test_forward_dtype(torch.float16)

    def test_fp16_backward(self):
        self._test_backward_dtype(torch.float16)

    def test_fp16_matches_float32(self):
        self._test_matches_float32(torch.float16)

    def test_tail_elements(self):
        """Test sizes not divisible by vector width (VEC=8 for half types)."""
        for dtype in [torch.bfloat16, torch.float16]:
            for size in [1, 3, 7, 8, 9, 15, 16, 17, 31, 33]:
                x = torch.randn(size, device="cuda", dtype=dtype, requires_grad=True)
                ap = torch.tensor(0.5, device="cuda", dtype=dtype, requires_grad=True)
                an = torch.tensor(-0.3, device="cuda", dtype=dtype, requires_grad=True)
                out = F.xielu(x, ap, an)
                self.assertEqual(out.shape, x.shape)
                out.sum().backward()
                self.assertEqual(x.grad.shape, x.shape)

    def test_tail_elements_float32(self):
        """Test sizes not divisible by vector width (VEC=4 for float32)."""
        for size in [1, 2, 3, 4, 5, 7, 9]:
            x = torch.randn(size, device="cuda", requires_grad=True)
            ap = torch.tensor(0.5, device="cuda", requires_grad=True)
            an = torch.tensor(-0.3, device="cuda", requires_grad=True)
            out = F.xielu(x, ap, an)
            self.assertEqual(out.shape, x.shape)
            out.sum().backward()
            self.assertEqual(x.grad.shape, x.shape)

    def test_large_tensor_cuda(self):
        """Test with large tensor to exercise multi-block kernel paths."""
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            x = torch.randn(100000, device="cuda", dtype=dtype)
            ap = torch.tensor(0.5, device="cuda", dtype=dtype)
            an = torch.tensor(-0.3, device="cuda", dtype=dtype)
            out = F.xielu(x, ap, an)
            self.assertEqual(out.shape, x.shape)
            self.assertFalse(out.isnan().any())

    def test_non_contiguous_cuda(self):
        x = torch.randn(16, 16, device="cuda")[::2, ::2]
        self.assertFalse(x.is_contiguous())
        ap = torch.tensor(0.5, device="cuda")
        an = torch.tensor(-0.3, device="cuda")
        out = F.xielu(x, ap, an)
        ref = F.xielu(x.contiguous(), ap, an)
        self.assertEqual(out, ref)

    def test_multidim_cuda(self):
        for shape in [(4, 8), (2, 3, 4), (2, 3, 4, 5)]:
            x = torch.randn(shape, device="cuda", requires_grad=True)
            ap = torch.tensor(0.5, device="cuda", requires_grad=True)
            an = torch.tensor(-0.3, device="cuda", requires_grad=True)
            out = F.xielu(x, ap, an)
            self.assertEqual(out.shape, x.shape)
            out.sum().backward()
            self.assertEqual(x.grad.shape, x.shape)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestXieluModuleCUDA(TestCase):
    """nn.xIELU module on CUDA."""

    def test_module_cuda_float32(self):
        m = nn.xIELU().cuda()
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        out = m(x)
        self.assertEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertIsNotNone(m.alpha_p.grad)
        self.assertIsNotNone(m.alpha_n.grad)

    def test_module_cuda_bf16(self):
        m = nn.xIELU().cuda().bfloat16()
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        out.sum().backward()
        self.assertIsNotNone(m.alpha_p.grad)

    def test_module_cuda_fp16(self):
        m = nn.xIELU().cuda().half()
        x = torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        out = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float16)
        out.sum().backward()
        self.assertIsNotNone(m.alpha_p.grad)

    def test_module_cpu_to_cuda(self):
        m = nn.xIELU()
        m = m.cuda()
        self.assertEqual(m.alpha_p.device.type, "cuda")
        self.assertEqual(m.alpha_n.device.type, "cuda")

    def test_module_state_dict_cuda(self):
        m1 = nn.xIELU().cuda()
        with torch.no_grad():
            m1.alpha_p.fill_(1.0)
            m1.alpha_n.fill_(-1.0)
        sd = m1.state_dict()
        m2 = nn.xIELU().cuda()
        m2.load_state_dict(sd)
        self.assertEqual(m2.alpha_p.item(), 1.0)
        self.assertEqual(m2.alpha_n.item(), -1.0)


# =====================================================================
# Numerical edge cases
# =====================================================================


class TestXieluNumerical(TestCase):
    """Numerical stability and edge case tests."""

    def test_near_eps_boundary(self):
        # Values right around the eps transition point
        eps = -1e-6
        x = torch.tensor([eps - 1e-7, eps, eps + 1e-7], dtype=torch.float64)
        ap = torch.tensor(0.5, dtype=torch.float64)
        an = torch.tensor(-0.3, dtype=torch.float64)
        out = F.xielu(x, ap, an, eps=eps)
        ref = xielu_ref(x, ap, an, eps=eps)
        self.assertEqual(out, ref, atol=1e-10, rtol=1e-10)

    def test_no_nan_on_large_negative(self):
        x = torch.tensor([-100.0, -500.0, -1000.0])
        ap = torch.tensor(0.0)
        an = torch.tensor(0.0)
        out = F.xielu(x, ap, an)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    def test_large_positive_input(self):
        x = torch.tensor([50.0, 100.0])
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        # For large positive x, output grows quadratically
        self.assertTrue((out > 0).all())
        self.assertFalse(out.isnan().any())

    def test_monotonic_positive_branch(self):
        # For x > 0 and default parameters, the function should be monotonically increasing
        x = torch.linspace(0.01, 10.0, 100)
        ap = torch.tensor(0.5)
        an = torch.tensor(-0.3)
        out = F.xielu(x, ap, an)
        diffs = out[1:] - out[:-1]
        self.assertTrue((diffs > 0).all())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_no_nan_large_negative_cuda(self):
        x = torch.tensor([-100.0, -500.0], device="cuda")
        ap = torch.tensor(0.0, device="cuda")
        an = torch.tensor(0.0, device="cuda")
        out = F.xielu(x, ap, an)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_no_nan_large_negative_bf16(self):
        x = torch.tensor([-100.0, -500.0], device="cuda", dtype=torch.bfloat16)
        ap = torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)
        an = torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)
        out = F.xielu(x, ap, an)
        self.assertFalse(out.isnan().any())
        self.assertFalse(out.isinf().any())


if __name__ == "__main__":
    run_tests()
