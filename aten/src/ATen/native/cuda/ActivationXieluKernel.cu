#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/where.h>
#include <ATen/ops/xielu_backward_native.h>
#include <ATen/ops/xielu_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

constexpr int XIELU_N_THREADS = 256;

static int xielu_max_blocks() {
  static int max_blocks = -1;
  if (max_blocks == -1) {
    int device, nSMs;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, device);
    max_blocks = nSMs * 8;
  }
  return max_blocks;
}

static inline int xielu_num_blocks(int64_t n) {
  int nb = static_cast<int>((n + XIELU_N_THREADS - 1) / XIELU_N_THREADS);
  return std::max(1, std::min(nb, xielu_max_blocks()));
}

__device__ __forceinline__ float dev_softplus(float x) {
  return (x > 20.0f) ? x : ((x < -20.0f) ? 0.0f : log1pf(expf(x)));
}

__device__ __forceinline__ float dev_sigmoid(float x) {
  return (x > 20.0f) ? 1.0f : ((x < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-x)));
}

// ---- Type traits: maps ATen scalar types to CUDA types and intrinsics ----

template <typename scalar_t>
struct XieluTraits;

template <>
struct XieluTraits<at::BFloat16> {
  using cuda_t = __nv_bfloat16;
  static constexpr int VEC = 8; // 16 bytes / 2
  __device__ __forceinline__ static float load(cuda_t v) { return __bfloat162float(v); }
  __device__ __forceinline__ static cuda_t store(float v) { return __float2bfloat16_rn(v); }
  __device__ __forceinline__ static float xielu_exp(float x) { return __expf(x); }
};

template <>
struct XieluTraits<at::Half> {
  using cuda_t = __half;
  static constexpr int VEC = 8; // 16 bytes / 2
  __device__ __forceinline__ static float load(cuda_t v) { return __half2float(v); }
  __device__ __forceinline__ static cuda_t store(float v) { return __float2half_rn(v); }
  __device__ __forceinline__ static float xielu_exp(float x) { return __expf(x); }
};

template <>
struct XieluTraits<float> {
  using cuda_t = float;
  static constexpr int VEC = 4; // 16 bytes / 4
  __device__ __forceinline__ static float load(cuda_t v) { return v; }
  __device__ __forceinline__ static cuda_t store(float v) { return v; }
  __device__ __forceinline__ static float xielu_exp(float x) { return expf(x); }
};

// ---- Forward kernels ----

template <typename scalar_t>
__global__ void xielu_fwd_vec_kernel(
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ x,
    typename XieluTraits<scalar_t>::cuda_t* __restrict__ y,
    const float* __restrict__ alpha_p_ptr,
    const float* __restrict__ alpha_n_ptr,
    const float beta,
    const float eps,
    const int64_t num_vectors) {
  using T = XieluTraits<scalar_t>;
  using cuda_t = typename T::cuda_t;
  constexpr int VEC = T::VEC;
  const float s_ap = dev_softplus(__ldg(alpha_p_ptr));
  const float s_an = dev_softplus(__ldg(alpha_n_ptr));
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;
  const float alpha_n_val = beta + s_an;
  const float neg_s_an = -s_an;
  for (int64_t vi = tid; vi < num_vectors; vi += stride) {
    uint4 xd = __ldg(reinterpret_cast<const uint4*>(x + vi * VEC));
    const cuda_t* xl = reinterpret_cast<const cuda_t*>(&xd);
    uint4 yd;
    cuda_t* yl = reinterpret_cast<cuda_t*>(&yd);
#pragma unroll
    for (int j = 0; j < VEC; j++) {
      float xf = T::load(xl[j]);
      float yf = (xf > 0.0f)
          ? xf * fmaf(s_ap, xf, beta)
          : fmaf(alpha_n_val, T::xielu_exp(fminf(xf, eps)) - 1.0f, neg_s_an * xf);
      yl[j] = T::store(yf);
    }
    *reinterpret_cast<uint4*>(y + vi * VEC) = yd;
  }
}

template <typename scalar_t>
__global__ void xielu_fwd_scalar_kernel(
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ x,
    typename XieluTraits<scalar_t>::cuda_t* __restrict__ y,
    const float* __restrict__ alpha_p_ptr,
    const float* __restrict__ alpha_n_ptr,
    const float beta,
    const float eps,
    const int64_t offset,
    const int64_t n) {
  using T = XieluTraits<scalar_t>;
  const float s_ap = dev_softplus(__ldg(alpha_p_ptr));
  const float s_an = dev_softplus(__ldg(alpha_n_ptr));
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float xf = T::load(x[offset + idx]);
  float yf = (xf > 0.0f)
      ? xf * fmaf(s_ap, xf, beta)
      : fmaf(s_an + beta, T::xielu_exp(fminf(xf, eps)) - 1.0f, -s_an * xf);
  y[offset + idx] = T::store(yf);
}

// ---- Backward kernels ----

template <typename scalar_t>
__global__ void xielu_bwd_vec_kernel(
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ x,
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ go,
    typename XieluTraits<scalar_t>::cuda_t* __restrict__ gi,
    float* __restrict__ galpha_p,
    float* __restrict__ galpha_n,
    const float* __restrict__ alpha_p_ptr,
    const float* __restrict__ alpha_n_ptr,
    const float beta,
    const float eps,
    const int64_t num_vectors) {
  using T = XieluTraits<scalar_t>;
  using cuda_t = typename T::cuda_t;
  constexpr int VEC = T::VEC;
  const float raw_ap = __ldg(alpha_p_ptr);
  const float raw_an = __ldg(alpha_n_ptr);
  const float s_ap = dev_softplus(raw_ap);
  const float s_an = dev_softplus(raw_an);
  const float ds_ap = dev_sigmoid(raw_ap);
  const float ds_an = dev_sigmoid(raw_an);
  const float two_s_ap = 2.0f * s_ap;
  const float alpha_n_val = beta + s_an;
  float gap = 0.0f, gan = 0.0f;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t vi = tid; vi < num_vectors; vi += stride) {
    uint4 xd = __ldg(reinterpret_cast<const uint4*>(x + vi * VEC));
    uint4 god = __ldg(reinterpret_cast<const uint4*>(go + vi * VEC));
    const cuda_t* xl = reinterpret_cast<const cuda_t*>(&xd);
    const cuda_t* gol = reinterpret_cast<const cuda_t*>(&god);
    uint4 gid;
    cuda_t* gil = reinterpret_cast<cuda_t*>(&gid);
#pragma unroll
    for (int j = 0; j < VEC; j++) {
      float xf = T::load(xl[j]);
      float gof = T::load(gol[j]);
      float dx, cp = 0.0f, cn = 0.0f;
      if (xf > 0.0f) {
        dx = gof * fmaf(two_s_ap, xf, beta);
        cp = gof * ds_ap * xf * xf;
      } else {
        float e = T::xielu_exp(fminf(xf, eps));
        float be = (xf <= eps) ? 1.0f : 0.0f;
        dx = gof * fmaf(alpha_n_val, e * be - 1.0f, beta);
        cn = gof * ds_an * (e - 1.0f - xf);
      }
      gil[j] = T::store(dx);
      gap += cp;
      gan += cn;
    }
    *reinterpret_cast<uint4*>(gi + vi * VEC) = gid;
  }
#pragma unroll
  for (int i = 16; i > 0; i >>= 1) {
    gap += __shfl_down_sync(0xffffffff, gap, i);
    gan += __shfl_down_sync(0xffffffff, gan, i);
  }
  if (threadIdx.x % 32 == 0) {
    atomicAdd(galpha_p, gap);
    atomicAdd(galpha_n, gan);
  }
}

template <typename scalar_t>
__global__ void xielu_bwd_scalar_kernel(
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ x,
    const typename XieluTraits<scalar_t>::cuda_t* __restrict__ go,
    typename XieluTraits<scalar_t>::cuda_t* __restrict__ gi,
    float* __restrict__ galpha_p,
    float* __restrict__ galpha_n,
    const float* __restrict__ alpha_p_ptr,
    const float* __restrict__ alpha_n_ptr,
    const float beta,
    const float eps,
    const int64_t offset,
    const int64_t n) {
  using T = XieluTraits<scalar_t>;
  const float raw_ap = __ldg(alpha_p_ptr);
  const float raw_an = __ldg(alpha_n_ptr);
  const float s_ap = dev_softplus(raw_ap);
  const float s_an = dev_softplus(raw_an);
  const float ds_ap = dev_sigmoid(raw_ap);
  const float ds_an = dev_sigmoid(raw_an);
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float xf = T::load(x[offset + idx]);
  float gof = T::load(go[offset + idx]);
  float dx, cp = 0.0f, cn = 0.0f;
  if (xf > 0.0f) {
    dx = gof * fmaf(2.0f * s_ap, xf, beta);
    cp = gof * ds_ap * xf * xf;
  } else {
    float e = T::xielu_exp(fminf(xf, eps));
    float be = (xf <= eps) ? 1.0f : 0.0f;
    dx = gof * fmaf(s_an + beta, e * be - 1.0f, beta);
    cn = gof * ds_an * (e - 1.0f - xf);
  }
  gi[offset + idx] = T::store(dx);
  atomicAdd(galpha_p, cp);
  atomicAdd(galpha_n, cn);
}

// ---- Launch helpers ----

template <typename scalar_t>
Tensor xielu_fwd_launch(
    const Tensor& self,
    const float* alpha_p_ptr,
    const float* alpha_n_ptr,
    float beta_f, float eps_f) {
  using T = XieluTraits<scalar_t>;
  using cuda_t = typename T::cuda_t;
  constexpr int VEC = T::VEC;
  auto self_c = self.contiguous();
  const auto stream = c10::cuda::getCurrentCUDAStream(self_c.device().index());
  const c10::cuda::CUDAStreamGuard guard(stream);
  auto y = at::empty_like(self_c);
  auto x_ptr = reinterpret_cast<const cuda_t*>(self_c.data_ptr<scalar_t>());
  auto y_ptr = reinterpret_cast<cuda_t*>(y.data_ptr<scalar_t>());
  int64_t N = self_c.numel();
  int64_t nv = N / VEC;
  int64_t tail = N % VEC;

  if (nv > 0) {
    xielu_fwd_vec_kernel<scalar_t><<<xielu_num_blocks(nv), XIELU_N_THREADS, 0, stream>>>(
        x_ptr, y_ptr, alpha_p_ptr, alpha_n_ptr, beta_f, eps_f, nv);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  if (tail > 0) {
    xielu_fwd_scalar_kernel<scalar_t><<<xielu_num_blocks(tail), XIELU_N_THREADS, 0, stream>>>(
        x_ptr, y_ptr, alpha_p_ptr, alpha_n_ptr, beta_f, eps_f, nv * VEC, tail);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  return y;
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> xielu_bwd_launch(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    const float* alpha_p_ptr,
    const float* alpha_n_ptr,
    float beta_f, float eps_f) {
  using T = XieluTraits<scalar_t>;
  using cuda_t = typename T::cuda_t;
  constexpr int VEC = T::VEC;
  auto self_c = self.contiguous();
  auto go_c = grad_output.contiguous();
  const auto stream = c10::cuda::getCurrentCUDAStream(self_c.device().index());
  const c10::cuda::CUDAStreamGuard guard(stream);
  auto gi = at::empty_like(self_c);
  auto gap_f = at::zeros({1}, self_c.options().dtype(kFloat));
  auto gan_f = at::zeros({1}, self_c.options().dtype(kFloat));
  auto x_ptr = reinterpret_cast<const cuda_t*>(self_c.data_ptr<scalar_t>());
  auto go_ptr = reinterpret_cast<const cuda_t*>(go_c.data_ptr<scalar_t>());
  auto gi_ptr = reinterpret_cast<cuda_t*>(gi.data_ptr<scalar_t>());
  float* gap_ptr = gap_f.data_ptr<float>();
  float* gan_ptr = gan_f.data_ptr<float>();
  int64_t N = self_c.numel();
  int64_t nv = N / VEC;
  int64_t tail = N % VEC;

  if (nv > 0) {
    xielu_bwd_vec_kernel<scalar_t><<<xielu_num_blocks(nv), XIELU_N_THREADS, 0, stream>>>(
        x_ptr, go_ptr, gi_ptr, gap_ptr, gan_ptr,
        alpha_p_ptr, alpha_n_ptr, beta_f, eps_f, nv);
    C10_CUDA_CHECK(cudaGetLastError());
  }
  if (tail > 0) {
    xielu_bwd_scalar_kernel<scalar_t><<<xielu_num_blocks(tail), XIELU_N_THREADS, 0, stream>>>(
        x_ptr, go_ptr, gi_ptr, gap_ptr, gan_ptr,
        alpha_p_ptr, alpha_n_ptr, beta_f, eps_f, nv * VEC, tail);
    C10_CUDA_CHECK(cudaGetLastError());
  }

  return {
      gi,
      gap_f.to(alpha_p.scalar_type()).reshape_as(alpha_p),
      gan_f.to(alpha_n.scalar_type()).reshape_as(alpha_n)};
}

} // anonymous namespace

// ---- Public entry points ----

Tensor xielu_cuda(
    const Tensor& self,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    const Scalar& beta,
    const Scalar& eps) {
  TORCH_CHECK(alpha_p.numel() == 1, "xielu_cuda: alpha_p must be a scalar tensor");
  TORCH_CHECK(alpha_n.numel() == 1, "xielu_cuda: alpha_n must be a scalar tensor");

  auto dtype = self.scalar_type();

  if (dtype == kBFloat16 || dtype == kHalf || dtype == kFloat) {
    auto ap_f = alpha_p.to(kFloat).contiguous();
    auto an_f = alpha_n.to(kFloat).contiguous();
    float beta_f = beta.to<float>();
    float eps_f = eps.to<float>();

    if (dtype == kBFloat16) {
      return xielu_fwd_launch<at::BFloat16>(
          self, ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    } else if (dtype == kHalf) {
      return xielu_fwd_launch<at::Half>(
          self, ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    } else {
      return xielu_fwd_launch<float>(
          self, ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    }
  }

  double eps_d = eps.to<double>();
  auto s_ap = at::softplus(alpha_p);
  auto s_an = at::softplus(alpha_n);
  auto x_clamped = at::clamp_max(self, eps_d);
  auto pos = self.mul(s_ap.mul(self).add(beta));
  auto neg = s_an.add(beta).mul(at::exp(x_clamped).sub(1.0)).sub(s_an.mul(self));
  return at::where(self.gt(0), pos, neg);
}

std::tuple<Tensor, Tensor, Tensor> xielu_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& alpha_p,
    const Tensor& alpha_n,
    const Scalar& beta,
    const Scalar& eps) {
  TORCH_CHECK(alpha_p.numel() == 1, "xielu_backward_cuda: alpha_p must be a scalar tensor");
  TORCH_CHECK(alpha_n.numel() == 1, "xielu_backward_cuda: alpha_n must be a scalar tensor");

  auto dtype = self.scalar_type();

  if (dtype == kBFloat16 || dtype == kHalf || dtype == kFloat) {
    auto ap_f = alpha_p.to(kFloat).contiguous();
    auto an_f = alpha_n.to(kFloat).contiguous();
    float beta_f = beta.to<float>();
    float eps_f = eps.to<float>();

    if (dtype == kBFloat16) {
      return xielu_bwd_launch<at::BFloat16>(
          grad_output, self, alpha_p, alpha_n,
          ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    } else if (dtype == kHalf) {
      return xielu_bwd_launch<at::Half>(
          grad_output, self, alpha_p, alpha_n,
          ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    } else {
      return xielu_bwd_launch<float>(
          grad_output, self, alpha_p, alpha_n,
          ap_f.data_ptr<float>(), an_f.data_ptr<float>(), beta_f, eps_f);
    }
  }

  double eps_d = eps.to<double>();
  auto s_ap = at::softplus(alpha_p);
  auto s_an = at::softplus(alpha_n);
  auto ds_ap = at::sigmoid(alpha_p);
  auto ds_an = at::sigmoid(alpha_n);
  auto x_clamped = at::clamp_max(self, eps_d);
  auto e = at::exp(x_clamped);
  auto below_eps = self.le(eps_d).to(self.dtype());
  auto grad_x_pos = grad_output.mul(s_ap.mul(2.0).mul(self).add(beta));
  auto grad_x_neg = grad_output.mul(s_an.add(beta).mul(e).mul(below_eps).sub(s_an));
  auto grad_x = at::where(self.gt(0), grad_x_pos, grad_x_neg);
  auto contrib_ap =
      grad_output.mul(ds_ap).mul(self).mul(self).masked_fill(self.le(0), 0).to(kFloat).sum();
  auto contrib_an =
      grad_output.mul(ds_an).mul(e.sub(1.0).sub(self)).masked_fill(self.gt(0), 0).to(kFloat).sum();
  return {
      grad_x,
      contrib_ap.reshape_as(alpha_p),
      contrib_an.reshape_as(alpha_n)};
}

} // namespace at::native
