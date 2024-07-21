# `.\pytorch\aten\src\ATen\native\Distance.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Distance.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_backward_native.h>
#include <ATen/ops/_cdist_forward.h>
#include <ATen/ops/_cdist_forward_native.h>
#include <ATen/ops/_euclidean_dist.h>
#include <ATen/ops/_euclidean_dist_native.h>
#include <ATen/ops/_pdist_backward_native.h>
#include <ATen/ops/_pdist_forward.h>
#include <ATen/ops/_pdist_forward_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cdist_native.h>
#include <ATen/ops/cosine_similarity_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/pairwise_distance_native.h>
#include <ATen/ops/pdist_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

#include <utility>
#endif

namespace at::native {

// 定义向前传播计算的分派器，用于计算点对距离
DEFINE_DISPATCH(pdist_forward_stub);
// 定义向后传播计算的分派器，用于计算点对距离的梯度
DEFINE_DISPATCH(pdist_backward_stub);
// 定义距离计算的分派器
DEFINE_DISPATCH(cdist_stub);
// 定义距离计算的梯度计算的分派器
DEFINE_DISPATCH(cdist_backward_stub);

// 计算两个张量之间的成对距离
Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  // 确定输出张量的维度为两个输入张量的最大维度
  auto x1_dim = x1.dim();
  auto x2_dim = x2.dim();
  auto output_dim = x1_dim > x2_dim ? x1_dim : x2_dim;
  auto innermost_dim = output_dim - 1;
  // 返回在内部维度上计算的范数，用于成对距离计算
  return at::norm(x1 - x2 + eps, p, innermost_dim, keepdim);
}

// 保证在向后传播时传递连续的内存
Tensor pdist(const Tensor& self, const double p) {
  // 检查张量的维度是否为2，因为 pdist 只支持二维张量
  TORCH_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  // 检查张量的数据类型是否为浮点数类型，因为 pdist 只支持浮点数类型
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
  // 检查 p 值是否为非负数，因为 pdist 只支持非负的 p 值
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
  // 调用 _pdist_forward 函数计算点对距离的前向传播
  return at::_pdist_forward(self.contiguous(), p);
}

// 计算欧几里得距离的第一部分，将其分为两个步骤以简化在反向传播阶段处理子梯度的复杂性
Tensor _euclidean_dist(const Tensor& x1, const Tensor& x2) {
  /** This function does the fist part of the euclidean distance calculation
   * We divide it in two steps to simplify dealing with subgradients in the
   * backward step */
  // 计算 x1 的范数平方
  Tensor x1_norm = x1.pow(2).sum(-1, true);
  // 创建一个与 x1_norm 形状相同的张量，并填充为 1
  Tensor x1_pad = at::ones_like(x1_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 计算 x2 的范数平方
  Tensor x2_norm = x2.pow(2).sum(-1, true);
  // 创建一个与 x2_norm 形状相同的张量，并填充为 1
  Tensor x2_pad = at::ones_like(x2_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 将 x1 的处理结果与填充张量连接在一起，形成 x1_ 张量
  Tensor x1_ = at::cat({x1.mul(-2), std::move(x1_norm), std::move(x1_pad)}, -1);
  // 将 x2 的处理结果与填充张量连接在一起，形成 x2_ 张量
  Tensor x2_ = at::cat({x2, std::move(x2_pad), std::move(x2_norm)}, -1);
  // 计算 x1_ 和 x2_ 的矩阵乘积，得到最终的欧几里得距离结果
  Tensor result = x1_.matmul(x2_.mT());
  // 将结果取非负下界为 0，并进行开方操作
  result.clamp_min_(0).sqrt_();
  return result;
}

} // namespace at::native
// 检查 x1 的数据类型是否为浮点类型，否则抛出错误信息
TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
// 获取 x1 的设备类型
auto device1 = x1.device().type();
// 检查 x2 的数据类型是否为浮点类型，否则抛出错误信息
TORCH_CHECK(at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
// 获取 x2 的设备类型
auto device2 = x2.device().type();
// 检查 p 值是否为非负数，否则抛出错误信息
TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
// 检查 x1 和 x2 的设备类型是否相同，否则抛出错误信息
TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);

// TODO: This is bad; this test should apply universally
// 如果 x1 在 CUDA 设备上，确保 x1 和 x2 在同一设备上，否则抛出错误信息
TORCH_CHECK(!x1.is_cuda() || x1.get_device() == x2.get_device(), "device of X1 (", x1.get_device(), ") must match device of X2 (", x2.get_device(), ")");

// 获取 x1 和 x2 的最后一个维度的符号化大小
SymInt c1 = x1.sym_size(-1);
SymInt c2 = x2.sym_size(-1);

// 0 - 默认值。如果 p = 2 且 r1 > 25 或 r2 > 25（基于性能指标），则尝试使用矩阵乘法计算距离
// 1 - 强制对 p = 2 使用矩阵乘法计算
// 2 - 对 p = 2 不使用矩阵乘法计算
// 根据提供的计算模式（如果未提供，默认为 0），设置 mode 变量
int64_t mode = compute_mode.value_or(0);
// 检查 mode 是否在有效范围内，否则抛出错误信息
TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

// 获取 x1 和 x2 的倒数第二个维度的符号化大小
SymInt r1 = x1.sym_size(-2);
SymInt r2 = x2.sym_size(-2);

// 查看注释 [cdist relies on cdist_impl redispatching]
// 如果条件不满足，则抛出错误信息
if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25))))) {
    // 检查 x1 的设备类型是否为 CPU 或 CUDA，否则抛出错误信息
    TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "cdist only supports CPU and CUDA devices, X1 got: ", device1);
}
    // 检查设备是否为 CPU 或 CUDA，否则抛出异常信息
    TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "cdist only supports CPU and CUDA devices, X2 got: ", device2);

  }

  // 获取张量 x1 和 x2 的维度
  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  // 对于批量计算，将除了最后两个维度以外的所有维度扩展为一个维度，大小为它们的乘积
  // 最后两个维度保持不变
  SymIntArrayRef batch_tensor1(x1.sym_sizes().data(), dim1 - 2);
  SymIntArrayRef batch_tensor2(x2.sym_sizes().data(), dim2 - 2);

  // 推断扩展后的尺寸，使用 infer_size_symint 函数计算
  std::vector<SymInt> expand_batch_portion = infer_size_symint(batch_tensor1, batch_tensor2);
  std::vector<SymInt> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<SymInt> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  // 计算扩展后批量维度的乘积
  const SymInt expand_batch_product = c10::multiply_integers(expand_batch_portion);

  // 定义视图的尺寸
  std::vector<SymInt> tensor1_view{expand_batch_product, r1, c1};
  std::vector<SymInt> tensor2_view{expand_batch_product, r2, c2};

  // 扩展张量 x1 和 x2，并确保内存连续性，然后按照指定的视图尺寸重新调整形状
  Tensor tensor1_expanded = x1.expand_symint(tensor1_expand_size).contiguous().view_symint(tensor1_view);
  Tensor tensor2_expanded = x2.expand_symint(tensor2_expand_size).contiguous().view_symint(tensor2_view);

  // 初始化输出的形状
  std::vector<SymInt> output_shape(std::move(expand_batch_portion));
  output_shape.insert(output_shape.end(), {r1, r2});

  Tensor result;

  // 根据不同情况生成不同类型的张量 result
  if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
    // 如果 r1 或 r2 或扩展批量乘积为 0，则生成一个空的符号整数张量
    result = at::empty_symint(output_shape, x1.options());
  } else if (c1 == 0) {
    // 如果 c1 为 0，则生成一个元素为零的符号整数张量
    result = at::zeros_symint(output_shape, x1.options());
  } else if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
    // 如果 p 为 2 并且满足特定的条件，计算欧几里得距离，并生成相应的符号整数张量
    // 查看注释 [cdist relies on cdist_impl redispatching]
    Tensor dist = (expand_batch_product == 1) ? at::_euclidean_dist(x1, x2) :
                  at::_euclidean_dist(tensor1_expanded, tensor2_expanded);
    result = dist.view_symint(output_shape);
  } else {
    // 否则生成一个空的符号整数张量，并调用 cdist_stub 函数进行计算
    result = at::empty_symint(output_shape, x1.options());
    cdist_stub(device1, result, tensor1_expanded, tensor2_expanded, p);
  }

  // 返回最终的结果张量
  return result;
}

Tensor cdist(const Tensor& x1, const Tensor& x2, const double p, std::optional<int64_t> compute_mode) {
  // 检查输入张量 x1 和 x2 的维度是否至少为 2
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  // 检查 x1 和 x2 的最后一个维度（列数）是否相等
  TORCH_CHECK(x1.sym_size(-1) == x2.sym_size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.sym_size(-1), " X2: ", x2.sym_size(-1));
  // 计算输出张量的名称
  auto maybe_outnames = namedinference::compute_cdist_outnames(x1, x2);
  // 使用 lambda 表达式计算结果张量
  auto result = [&]() {
    // 创建一个 NoNamesGuard 对象，确保没有名称传播
    NoNamesGuard guard;
    // 获取 x1 和 x2 的行数
    SymInt r1 = x1.sym_size(-2);
    SymInt r2 = x2.sym_size(-2);
    // 对空输入的特殊处理：确保图正确连接
    if (x1.sym_numel() == 0 || x2.sym_numel() == 0) {
        // 调用 _cdist_forward 函数处理空输入情况
        return at::_cdist_forward(x1, x2, p, compute_mode);
    }
    // 根据 p 和 mode 的值判断调用哪个计算函数
    int64_t mode = compute_mode.value_or(0);
    // 注意 [cdist relies on cdist_impl redispatching]
    // 如果 p=2 且 mode=1，或者 p=2 且 mode=0 且 r1 或 r2 大于 25，则调用 cdist_impl 函数
    if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
        return cdist_impl(x1, x2, p, compute_mode);
    } else {
        // 否则调用 _cdist_forward 函数
        return at::_cdist_forward(x1, x2, p, compute_mode);
    }
  }();
  // 如果输出张量非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回结果张量
  return result;
}

Tensor _cdist_forward(const Tensor& x1, const Tensor& x2, const double p, std::optional<int64_t> compute_mode) {
  // 检查输入张量 x1 和 x2 的维度是否至少为 2
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  // 检查 x1 和 x2 的最后一个维度（列数）是否相等
  TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
  // 计算输出张量的名称
  auto maybe_outnames = namedinference::compute_cdist_outnames(x1, x2);
  // 使用 lambda 表达式计算结果张量
  auto result = [&]() {
    // 创建一个 NoNamesGuard 对象，确保没有名称传播
    NoNamesGuard guard;
    // 调用 cdist_impl 函数进行计算
    return cdist_impl(x1, x2, p, compute_mode);
  }();
  // 如果输出张量非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回结果张量
  return result;
}
Tensor _cdist_backward(const Tensor& _grad, const Tensor& _x1, const Tensor& _x2, const double p, const Tensor& _cdist) {
  // 处理广播后可能生成的非连续张量，需在进行检查之前处理
  int64_t c1 = _x1.size(-1);  // 获取张量 _x1 的最后一个维度大小
  int64_t c2 = _x2.size(-1);  // 获取张量 _x2 的最后一个维度大小
  int64_t r1 = _x1.size(-2);  // 获取张量 _x1 的倒数第二个维度大小
  int64_t r2 = _x2.size(-2);  // 获取张量 _x2 的倒数第二个维度大小
  auto dim1 = _x1.dim();      // 获取张量 _x1 的维度数
  auto dim2 = _x2.dim();      // 获取张量 _x2 的维度数
  IntArrayRef batch_tensor1(_x1.sizes().data(), dim1 - 2);  // 获取 _x1 的批处理部分尺寸
  IntArrayRef batch_tensor2(_x2.sizes().data(), dim2 - 2);  // 获取 _x2 的批处理部分尺寸
  std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);  // 推断扩展的批处理部分尺寸
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);  // 创建扩展后的 _x1 尺寸向量
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});  // 在尺寸向量末尾插入倒数第二维度和最后一维度大小
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);  // 创建扩展后的 _x2 尺寸向量
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});  // 在尺寸向量末尾插入倒数第二维度和最后一维度大小

  // 计算线性化批处理大小
  const int64_t batch_product = c10::multiply_integers(expand_batch_portion);

  // 优雅处理空张量情况
  if (r1 == 0 || r2 == 0 || c1 == 0 || batch_product == 0) {
    return at::zeros_like(_x1, _x1.options());  // 返回和 _x1 相同大小和选项的零张量
  }

  Tensor x1 = _x1;
  if (tensor1_expand_size != x1.sizes()) {
    x1 = x1.expand(tensor1_expand_size);  // 如果扩展尺寸不等于当前尺寸，则扩展 _x1
  }
  Tensor x2 = _x2;
  if (tensor2_expand_size != x2.sizes()) {
    x2 = x2.expand(tensor2_expand_size);  // 如果扩展尺寸不等于当前尺寸，则扩展 _x2
  }

  x1 = x1.contiguous();  // 保证 _x1 是连续的
  x2 = x2.contiguous();  // 保证 _x2 是连续的
  auto cdist = _cdist.contiguous();  // 确保 _cdist 是连续的
  auto grad = _grad.contiguous();    // 确保 _grad 是连续的
  int64_t n = x1.size(-2);  // 获取 _x1 倒数第二维度大小
  int64_t m = x1.size(-1);  // 获取 _x1 最后一维度大小
  auto device1 = x1.device().type();  // 获取 _x1 的设备类型
  TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "_cdist_backward 只支持 CPU 和 CUDA 设备，X1 设备为: ", device1);
  auto device2 = x2.device().type();  // 获取 _x2 的设备类型
  TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "_cdist_backward 只支持 CPU 和 CUDA 设备，X2 设备为: ", device2);

  // 创建梯度张量 grad_x1
  Tensor grad_x1 =
      at::empty({batch_product, n, m}, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 cdist_backward_stub 计算梯度
  cdist_backward_stub(device1, grad_x1, grad, x1, x2, p, cdist);

  // 返回梯度张量，需使用 x1 的尺寸，而非原始 _x1 的尺寸，因为该梯度未考虑广播
  // 广播将由自动求导引擎自动处理
  return grad_x1.view(x1.sizes());
}

Tensor _pdist_forward(const Tensor& self, const double p) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward 需要连续的输入");
  auto device = self.device().type();  // 获取 self 的设备类型
  TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_forward 只支持 CPU 和 CUDA 设备，当前设备为: ", device);
  Tensor result = at::empty({0}, self.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // 创建空张量 result
  if (self.size(0) <= 1) {
    result.resize_({0});  // 如果 self 的第一个维度小于等于 1，则将 result 大小设为 0
  } else {
    int64_t n = self.size(0);  // 获取 self 的第一个维度大小
    int64_t c = n * (n - 1) / 2;  // 计算结果张量 result 的大小
    result.resize_({c});  // 设置 result 的大小为 c
    if (self.size(1) == 0) {
      result.fill_(0);  // 如果 self 的第二个维度为 0，则将 result 填充为 0
    } else {
      pdist_forward_stub(device, result, self, p);  // 调用 pdist_forward_stub 计算前向传播
    }
  }
  return result;  // 返回结果张量 result
}
Tensor _pdist_backward(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  // 检查 self 张量是否是连续的
  TORCH_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  // 检查 pdist 张量是否是连续的
  TORCH_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  // 获取 self 张量的设备类型
  auto device = self.device().type();
  // 检查设备类型是否为 CPU 或 CUDA
  TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_backward only supports CPU and CUDA devices, got: ", device);
  // 创建一个与 self 张量相同形状的空张量，使用旧式连续内存格式
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 pdist_backward_stub 函数进行后向传播计算
  pdist_backward_stub(device, result, grad, self, p, pdist);
  // 返回计算结果张量
  return result;
}

Tensor cosine_similarity(const Tensor& x1_, const Tensor& x2_, int64_t dim, double eps) {
    /*
   * cosine_similarity(x1, x2) = <x1, x2> / (||x1|| * ||x2||)
   *
   * 当前的实现改进了之前的版本。
   *
   * 之前的实现：
   * 1. 计算 num = <x1, x2>,
   * 2. 计算 denom = ||x1|| * ||x2||,
   * 3. 计算 denom = max(denom, eps) 避免除以零，
   * 4. 返回 num / denom。
   *
   * 之前的实现存在以下问题：
   * 1. 当 ||x1|| 和 ||x2|| 很大时，<x1, x2> 可能丢失精度。
   * 2. 当 ||x1|| 和 ||x2|| 很大时，||x1|| * ||x2|| 可能丢失精度。
   * 3. 精度丢失可能导致 |cosing_similarity(x1, x2)| > 1.0。
   *
   * 当前的实现：
   * 1. 计算 x1_normalized = x1 / max(||x1||, eps),
   *            x2_normalized = x2 / max(||x2||, eps),
   * 2. 返回 <x1_normalized, x2_normalized>.
   *
   * 当前的实现通过以下方式改进了之前的版本：
   * 1. 确保不直接计算 <x1, x2> 和 ||x1|| * ||x2||，从而避免浮点溢出。
   * 2. 两种方法在计算 ||x1|| 和 ||x2|| 时都可能存在问题，但对于当前方法，这是唯一的浮点数不精确的来源。
   * 3. 确保 |cosing_similarity(x1, x2)| <= 1.0。
   *
   */

  // 确定 x1_ 和 x2_ 的公共数据类型
  auto commonDtype = at::result_type(x1_, x2_);
  // 检查公共数据类型是否为浮点类型
  TORCH_CHECK(at::isFloatingType(commonDtype), "expected common dtype to be floating point, yet common dtype is ", commonDtype);

  // 我们接受整数类型（以及布尔类型），但 vector_norm 不接受
  auto x1_is_int = c10::isIntegralType(x1_.scalar_type(), /*încludeBool=*/true);
  auto x2_is_int = c10::isIntegralType(x2_.scalar_type(), /*încludeBool=*/true);
  // 如果 x1_ 是整数类型，则转换为公共数据类型
  auto x1_t = x1_is_int ? x1_.to(commonDtype) : x1_;
  // 如果 x2_ 是整数类型，则转换为公共数据类型
  auto x2_t = x2_is_int ? x2_.to(commonDtype) : x2_;
  // 扩展 x1_t 和 x2_t，以匹配维度
  auto [x1, x2] = expand_outplace(x1_t, x2_t);


  // 首先，我们希望将每个张量除以其范数，这样更稳定。
  // 这保证结果在 -1.0 到 1.0 之间
  // 我们克隆它们，因为我们将会就地修改它们
  // 这允许梯度正确传播到 x1 和 x2
  // 计算 x1 的 L2 范数，并克隆结果张量
  auto x1_norm = at::linalg_vector_norm(*x1, 2, /*dim=*/dim, /*keepdim=*/true).clone();
  // 计算 x2 的 L2 范数，并克隆结果张量
  auto x2_norm = at::linalg_vector_norm(*x2, 2, /*dim=*/dim, /*keepdim=*/true).clone();

  {
    # 创建一个禁止梯度计算的上下文管理器，该 guard 对象确保在其作用域内的操作不会被记录梯度
    at::NoGradGuard guard;
    
    # 对 x1_norm 进行截断操作，将所有小于 eps 的值设为 eps
    x1_norm.clamp_min_(eps);
    
    # 对 x2_norm 进行截断操作，将所有小于 eps 的值设为 eps
    x2_norm.clamp_min_(eps);
}

}  // namespace at::native
```