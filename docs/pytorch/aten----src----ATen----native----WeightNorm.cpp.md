# `.\pytorch\aten\src\ATen\native\WeightNorm.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/cpu/WeightNormKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_differentiable_backward_native.h>
#include <ATen/ops/_weight_norm_interface.h>
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/_weight_norm_native.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/norm_except_dim.h>
#include <ATen/ops/norm_except_dim_native.h>
#endif

#include <vector>

namespace at::native {

// Define dispatch stubs for weight normalization operations
DEFINE_DISPATCH(weight_norm_stub);
DEFINE_DISPATCH(weight_norm_backward_stub);

// Function to compute norm except along a specified dimension
// Staying faithful to the Python for now for clarity, look for optimizations later
// (e.g., single return statement for RVO)
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim)
{
  // Handle special cases for computing norm along different dimensions
  if (dim == -1) {
    return v.norm(pow); // Compute norm of the tensor v
  } else if (dim == 0) {
    // Compute norm along dimension 0 and reshape accordingly
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[0] = v.size(0);
    return v.contiguous().view({v.size(0), -1}).norm(pow, 1).view(output_size);
  } else if (dim == v.dim() - 1) {
    // Compute norm along the last dimension and reshape accordingly
    std::vector<int64_t> output_size(v.dim(), 1);
    output_size[v.dim() - 1] = v.size(v.dim() - 1);
    return v.contiguous().view({-1, v.size(v.dim() - 1)}).norm(pow, 0).view(output_size);
  } else {
    // General case: transpose the tensor and compute norm except along the specified dimension
    return at::norm_except_dim(v.transpose(0, dim), pow, 0).transpose(0, dim);
  }
}

// Function to compute weight normalization on CPU
std::tuple<Tensor,Tensor> weight_norm_cpu(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  auto w = at::empty_like(v, at::MemoryFormat::Contiguous); // Allocate tensor w like v

  // Determine dtype for norm based on g's scalar type
  const auto dtype = g.scalar_type() == at::ScalarType::BFloat16 ?
      at::ScalarType::Float : g.scalar_type();
  auto norm = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(dtype)); // Allocate tensor norm
  weight_norm_stub(kCPU, w, norm, v, g, dim); // Call the CPU dispatch stub for weight normalization

  return std::tuple<Tensor, Tensor>{w, norm}; // Return tensors w and norm as a tuple
}

// Function to compute backward pass of weight normalization on CPU
std::tuple<Tensor, Tensor> weight_norm_backward_cpu(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norm,
    int64_t dim) {
  
  // Compute gradient of weight with respect to norm
  auto grad_norm = at::native::norm_except_dim(saved_v, -1, dim);

  // Compute gradient of weight with respect to input tensor v
  auto grad_v = at::empty_like(saved_v);
  weight_norm_backward_stub(kCPU, grad_v, grad_w, saved_v, saved_g, saved_norm, grad_norm, dim);

  return std::tuple<Tensor, Tensor>{grad_v, grad_norm}; // Return gradients as a tuple
}

} // namespace at::native
  // 检查 saved_v 张量是否是连续的
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  // 检查 saved_g 张量是否是连续的
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  // 检查 saved_norm 张量是否是连续的
  TORCH_CHECK(saved_norm.is_contiguous(), "saved_norm must be contiguous");

  // 根据 saved_v 张量创建一个与其同样大小和数据类型的新张量 grad_v，并确保其是连续的
  auto grad_v = at::empty_like(saved_v, at::MemoryFormat::Contiguous);
  // 根据 saved_g 张量创建一个与其同样大小和数据类型的新张量 grad_g，并确保其是连续的
  auto grad_g = at::empty_like(saved_g, at::MemoryFormat::Contiguous);

  // 调用 weight_norm_backward_stub 函数，用于计算权重归一化操作的反向传播
  // 将结果保存到 grad_v 和 grad_g 张量中
  weight_norm_backward_stub(kCPU, grad_v, grad_g, grad_w, saved_v, saved_g, saved_norm, dim);

  // 返回包含 grad_v 和 grad_g 张量的 std::tuple
  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
// 定义权重归一化函数，计算给定维度上的权重归一化结果
Tensor _weight_norm
  (const Tensor & v_in,    // 输入张量 v_in
   const Tensor & g_in,    // 输入张量 g_in
   int64_t dim)            // 归一化操作的维度

{

  TORCH_CHECK(
    v_in.device() == g_in.device(),  // 检查输入张量在相同设备上
    "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
    "on ", v_in.device(), " and g_in is on ", g_in.device());

  auto v = v_in.contiguous();  // 确保输入张量 v 连续存储
  auto g = g_in.contiguous();  // 确保输入张量 g 连续存储

  auto has_half_dtype = v.scalar_type() == at::ScalarType::Half
    || g.scalar_type() == at::ScalarType::Half;

  bool can_use_fused = !has_half_dtype && ((dim == 0) || (dim == v.dim() - 1));

  if (can_use_fused) {
    // 对于可以使用融合的情况，返回调用 _weight_norm_interface 接口的结果
    // 这个调用会在自动求导图中构建 WeightNormFusedBackward 对象
    return std::get<0>(at::_weight_norm_interface(v, g, dim));
  } else {
    // 对于不可融合的情况，使用数值稳定的原始操作进行权重归一化计算
    // 这里使用了张量运算来实现权重归一化的反向传播
    return v*(g/at::norm_except_dim(v, 2, dim));
  }
}

// 可微分的反向传播路径，用作 weight_norm_backward 的替代方案，用于在 backward 本身创建图形时使用
// 在 Functions.cpp 中，必须执行 GradMode::is_enabled() 检查，这就是为什么我们在这里定义了一个单独的函数而不是在 weight_norm_cuda_backward 中内联它
std::tuple<Tensor, Tensor> _weight_norm_differentiable_backward
  (const Tensor & grad_w,       // 梯度张量 grad_w
   const Tensor & saved_v,      // 保存的输入张量 v
   const Tensor & saved_g,      // 保存的输入张量 g
   const Tensor & saved_norms,  // 保存的归一化结果 norms
   int64_t dim)                 // 归一化操作的维度

{
  // 在 Functions.cpp 中，HardshrinkBackward 对象将 "grad.contiguous()" 作为第一个参数提供，因此这里 grad_w 应该是连续的
  // 所有这些检查应该都成功通过：
  TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

  int64_t last_dim = saved_v.dim() - 1;
  int64_t last_size = saved_v.size(last_dim);

  // 类似于 weight_norm_fused_backward，weight_norm_differentiable_backward 应该只能通过 WeightNormFusedBackward 对象调用，
  // 因此我们期望 dim == 0 || dim == saved_v.size(-1)
  TORCH_CHECK(dim == 0 || dim == last_dim, "Expected dim to be the first or last dimension");

  // saved_g 和 saved_norms 已经被形状化，以在正确的维度上广播

  // ...但是当 saved_g 和 saved_v 是半精度时，saved_norms 可能是 Float
  // 考虑：saved_norms.to(..., True /*non_blocking*/);
  auto norms = saved_norms.to(saved_g.scalar_type());

  std::vector<int64_t> bcast_size(saved_v.dim(), 1);

  // 使用可微分的原始操作进行分析的反向传播路径
  if (dim == 0) {
    bcast_size[0] = saved_v.size(0);
    auto per_dim_sums = (grad_w*saved_v).view({saved_v.size(0), -1}).sum(1).view(bcast_size);
    auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
    // 如果 dim 不等于 last_dim，则执行以下代码块
    auto grad_g = per_dim_sums/norms;
    // 返回包含梯度向量 grad_v 和归一化梯度 grad_g 的元组
    return std::tuple<Tensor, Tensor>{grad_v, grad_g};
  } else { // dim == last_dim
    // 将 bcast_size 数组中的最后一个维度设为 last_size
    bcast_size[last_dim] = last_size;
    // 计算每个维度上的和，得到 per_dim_sums，将其形状调整为 bcast_size
    auto per_dim_sums = (grad_w*saved_v).view({-1, last_size}).sum(0).view(bcast_size);
    // 计算 grad_v，利用 saved_g 和 norms 归一化 grad_w 和 per_dim_sums 的组合
    auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
    // 计算 grad_g，每个维度的和除以 norms 得到归一化梯度
    auto grad_g = per_dim_sums/norms;
    // 返回包含梯度向量 grad_v 和归一化梯度 grad_g 的元组
    return std::tuple<Tensor, Tensor>{grad_v, grad_g};
  }
}

} // namespace at::native
```