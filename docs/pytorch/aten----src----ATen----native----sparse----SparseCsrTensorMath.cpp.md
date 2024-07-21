# `.\pytorch\aten\src\ATen\native\sparse\SparseCsrTensorMath.cpp`

```py
// 定义预处理器宏，用于在 Torch 中仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 的分发机制头文件
#include <ATen/Dispatch.h>

// 包含 ATen 的扩展工具头文件
#include <ATen/ExpandUtils.h>

// 包含 ATen 的并行处理头文件
#include <ATen/Parallel.h>

// 包含 ATen 的稀疏 CSR 张量工具头文件
#include <ATen/SparseCsrTensorUtils.h>

// 包含 ATen 的张量核心定义头文件
#include <ATen/core/Tensor.h>

// 包含 ATen 的梯度模式控制头文件
#include <ATen/core/grad_mode.h>

// 包含 ATen 的 MKL 稀疏工具头文件
#include <ATen/mkl/Sparse.h>

// 包含 ATen 的二元操作头文件
#include <ATen/native/BinaryOps.h>

// 包含 ATen 的 CPU Blas 头文件
#include <ATen/native/CPUBlas.h>

// 包含 ATen 的大小调整头文件
#include <ATen/native/Resize.h>

// 包含 ATen 的稀疏张量工具头文件
#include <ATen/native/SparseTensorUtils.h>

// 包含 ATen 的张量转换头文件
#include <ATen/native/TensorConversions.h>

// 包含 ATen 的 MKL 稀疏 Blas 实现头文件
#include <ATen/native/mkl/SparseBlasImpl.h>

// 包含 ATen 的稀疏 Blas 实现头文件
#include <ATen/native/sparse/SparseBlasImpl.h>

// 包含 ATen 的稀疏 CSR 张量数学运算头文件
#include <ATen/native/sparse/SparseCsrTensorMath.h>

// 包含 C10 的宏定义头文件
#include <c10/macros/Macros.h>

// 包含 C10 的范围迭代工具头文件
#include <c10/util/irange.h>

// 包含 ATen 的累加类型头文件
#include <ATen/AccumulateType.h>

// 如果未定义每个操作符的单独头文件
#ifndef AT_PER_OPERATOR_HEADERS

// 包含 ATen 的函数头文件
#include <ATen/Functions.h>

// 包含 ATen 的原生函数头文件
#include <ATen/NativeFunctions.h>

// 包含 ATen 的操作符头文件
#include <ATen/Operators.h>

// 否则，包含 ATen 的特定操作的物理共轭原生头文件
#else
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_prod_native.h>
#include <ATen/ops/_sparse_csr_sum_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_mm_reduce_impl_backward_native.h>
#include <ATen/ops/_sparse_mm_reduce_impl_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/angle_native.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/mm_native.h>
#endif
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sparse_mask.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

// 包含所需的 ATen 头文件和标准库头文件
#include <algorithm>

// ATen 命名空间
namespace at {
namespace meta {

// 定义 TORCH_META_FUNC 函数 _convert_indices_from_coo_to_csr
TORCH_META_FUNC(_convert_indices_from_coo_to_csr)
(const Tensor& self, const int64_t size, const bool out_int32) {
  // 检查输入张量 self 的维度，确保其为一维向量
  TORCH_CHECK(self.dim() <= 1, "Input is supposed to be a vector, but got ",
              self.dim(), " dimensional tensor.");
  // 根据 out_int32 参数选择目标标量类型
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  // 设置输出张量的选项，包括设备和数据类型
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  // 调用 ATen 的设置输出原始步进函数，设置输出张量的大小和选项
  set_output_raw_strided(0, size + 1, {}, options);
}

// 定义 TORCH_META_FUNC 函数 _convert_indices_from_csr_to_coo
TORCH_META_FUNC(_convert_indices_from_csr_to_coo)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose) {
  // 检查 crow_indices 和 col_indices 张量的维度，确保它们维度相同
  TORCH_CHECK(
    crow_indices.dim() == col_indices.dim(), "crow_indices and col_indices are supposed to have"
    " the same dimensionality, but got ", crow_indices.dim(), " and ",
    crow_indices.dim(), " dimensional tensors, respectively.");
  // 根据 out_int32 参数选择目标标量类型
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  // 设置输出张量的选项，使用 crow_indices 的选项和选择的标量类型
  c10::TensorOptions options = crow_indices.options().dtype(scalar_type);
  // 调用 ATen 的设置输出原始步进函数，设置输出张量的大小、步进和选项
  set_output_raw_strided(0, {col_indices.dim() + 1, col_indices.numel()}, {}, options, {});
}

} // namespace meta

// 匿名命名空间内定义的模板函数 unary_op_out
namespace {

template <typename F>
Tensor& unary_op_out(F op_out, const Tensor& self, Tensor& result) {
  // 内部断言，确保输入张量 self 和输出张量 result 均为稀疏 CSR 格式
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(result.is_sparse_csr());

  if (!result.is_same(self)) {
    // 对于 result 张量为空（0x0）的情况，手动调整 result 张量大小与 self 张量相同
    if (result.numel() == 0) {
      // 调用 ATen 的 resize_as_sparse_compressed_ 函数，将 result 调整为与 self 相同的大小
      at::native::resize_as_sparse_compressed_(result, self);
  }
  // copy_sparse_compressed_ 内部检查 result 和 self 张量的大小
  // 因此不需要外部大小检查
  at::native::copy_sparse_compressed_(result, self);
}

// 获取 self 张量的值
auto self_values = self.values();
// 获取 result 张量的值
auto result_values = result.values();

// 使用 op_out 函数对 self_values 和 result_values 进行操作
op_out(self_values, result_values);

// 返回处理后的 result 张量
return result;
} // 结束匿名命名空间

namespace native {

using namespace at::sparse_csr;
// 可以从稀疏 COO 中使用某些实用函数
using namespace at::sparse;

// 在稀疏 CSR 格式上执行乘法操作，并将结果存储在给定的张量中
Tensor& mul_out_sparse_csr(const Tensor& t_, const Tensor& src_, Tensor& r) {
  // 如果 t_ 是稀疏 CSR 格式，并且 src_ 是分块的布局
  if (t_.is_sparse_csr() && src_.layout() == kStrided) {
    return mul_out_sparse_csr(t_, src_.sparse_mask(t_), r);
  }
  // 如果 t_ 是分块的布局，并且 src_ 是稀疏 CSR 格式
  if (t_.layout() == kStrided && src_.is_sparse_csr()) {
    return mul_out_sparse_csr(t_.sparse_mask(src_), src_, r);
  }
  // 确保结果张量 r 是稀疏 CSR 格式
  TORCH_CHECK(r.is_sparse_csr(), "Expected result Tensor to be of format CSR");
  // 将 t_ 和 src_ 转换为稀疏格式
  Tensor t = t_.to_sparse();
  Tensor src = src_.to_sparse();
  // 对稀疏张量 t 和 src 执行乘法操作
  Tensor tmp_result = t.mul(src);
  // 将结果转换为稀疏 CSR 格式，并存储在 r 中
  auto r_sparse_csr = tmp_result.to_sparse_csr();
  r.resize_as_sparse_(r_sparse_csr);
  r.copy_(r_sparse_csr);
  return r;
}

// 对于包装标量的交集二元操作，返回操作结果的稀疏张量
template <typename op_t>
Tensor intersection_binary_op_with_wrapped_scalar(const Tensor& sparse, const Tensor& scalar, const op_t& op) {
  // 注意: intersection_binary_op_with_wrapped_scalar 假设 scalar.numel() == 1
  // 执行操作 op 在 sparse.values() 和标量 scalar.squeeze() 上，并根据结果类型确定返回值
  const auto result_values = op(sparse.values(), scalar.squeeze()).to(at::result_type(sparse, scalar));
  // 推断结果张量的大小
  const auto result_sizes = infer_size(sparse.sizes(), scalar.sizes());
  // 获取压缩的索引和普通索引
  auto [compressed_indices, plain_indices] = getCompressedPlainIndices(sparse);
  // 构造稀疏压缩张量并返回
  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices.clone(),
      plain_indices.clone(),
      result_values,
      result_sizes,
      sparse.options().dtype(result_values.scalar_type()));
}

// 对于包装标量的交集二元操作，就地修改稀疏张量 sparse
template <typename op_t>
Tensor& intersection_binary_op_with_wrapped_scalar_(Tensor& sparse, const Tensor& scalar, const string& op_name, const op_t& op) {
  // 注意: intersection_binary_op_with_wrapped_scalar_ 假设 scalar.numel() == 1
  // 推断广播后的形状
  const auto broadcasted_shape = infer_size(sparse.sizes(), scalar.sizes());
  // 如果稀疏张量 sparse 的大小不等于广播后的形状，抛出错误
  if (sparse.sizes() != broadcasted_shape) {
    TORCH_CHECK(false, op_name, "(): output with shape ", sparse.sizes(), " does not match ",
        "the broadcast shape ", broadcasted_shape);
  }
  // 获取稀疏张量的值
  auto values = sparse.values();
  // 在这里可以安全地使用 squeeze，因为已知标量 scalar 可以安全广播
  op(values, scalar.squeeze());
  return sparse;
}

// 在稀疏 CSR 格式上执行乘法操作，并返回结果张量
Tensor mul_sparse_csr(const Tensor& self, const Tensor& other) {
  // 检查是否有一个参数是包装的标量
  if (self.layout() == kStrided && self.dim() == 0) {
    // 使用包装的标量在 other 和 self 上执行交集二元操作
    return intersection_binary_op_with_wrapped_scalar(other, self, [](const Tensor& a, const Tensor& b) -> Tensor {
        return a.mul(b);
    });
  }
  if (other.layout() == kStrided && other.dim() == 0) {
    // 调用二元操作的交集函数，使用标量包装，返回结果
    return intersection_binary_op_with_wrapped_scalar(self, other, [](const Tensor& a, const Tensor& b) -> Tensor {
        // 返回张量 a 和 b 的逐元素乘积
        return a.mul(b);
    });
  }

  // 如果 self 是稀疏 CSR 格式，并且 other 的布局是 kStrided
  if (self.is_sparse_csr() && other.layout() == kStrided) {
    // 返回 self 与使用 self 作为稀疏掩码的 other 的乘积结果
    return mul_sparse_csr(self, other.sparse_mask(self));
  }
  // 如果 self 的布局是 kStrided，并且 other 是稀疏 CSR 格式
  if (self.layout() == kStrided && other.is_sparse_csr()) {
    // 返回使用 other 作为稀疏掩码的 self 与 other 的乘积结果
    return mul_sparse_csr(self.sparse_mask(other), other);
  }

  // 计算 self 和 other 的公共数据类型
  auto commonDtype = at::result_type(self, other);
  // 根据公共数据类型设置结果张量的选项
  auto result_options = self.options().dtype(commonDtype);
  // 创建一个空的张量 result，形状为 {0, 0}，选项为 result_options
  // 这里创建的空张量可能用于存储乘积结果
  Tensor result = at::empty({0, 0}, result_options);
  // 使用 at::mul_out 函数计算 self 和 other 的乘积，并将结果存储到 result 中
  // 这里的 at::mul_out 可能会重派发操作到特定的乘法实现
  return at::mul_out(result, self, other); // redispatch!
}

Tensor& mul_sparse_csr_(Tensor& self, const Tensor& other) {
  // 检查 `other` 是否为标准布局且维度为 0
  if (other.layout() == kStrided && other.dim() == 0) {
    // 调用具有封装标量的二元操作 `intersection_binary_op_with_wrapped_scalar_`，传递操作 "mul_"
    return intersection_binary_op_with_wrapped_scalar_(self, other, "mul_", [](Tensor& a, const Tensor& b) -> Tensor& {
        // 执行张量 `a` 与张量 `b` 的乘法操作，并返回结果
        return a.mul_(b);
    });
  }
  // 否则，重新分发到 `at::mul_out`，执行乘法操作
  return at::mul_out(self, self, other); // redispatch!
}


namespace {

template <typename F>
// 为一元操作 `op` 获取结果张量
inline Tensor get_result_tensor_for_unary_op(F op, const Tensor& input) {
  // 获取输入张量的值
  auto values = input.values();

  // 处理一元操作的类型提升，获取结果值，并使用结果值创建稀疏压缩张量
  auto result_values = op(values);

  // 根据输入布局调度压缩和纯粹的索引
  auto compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(input.layout(),
                                                                      "get_result_tensor_for_unary_op",
                                                                      [&]{ return input.crow_indices(); },
                                                                      [&]{ return input.ccol_indices(); });
  auto plain_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(input.layout(),
                                                                 "get_result_tensor_for_unary_op",
                                                                 [&]{ return input.col_indices(); },
                                                                 [&]{ return input.row_indices(); });

  // 使用 `_sparse_compressed_tensor_unsafe` 创建稀疏压缩张量
  auto result = at::_sparse_compressed_tensor_unsafe(
      compressed_indices.clone(),
      plain_indices.clone(),
      result_values,
      input.sizes(),
      input.options().dtype(result_values.scalar_type()));

  // 返回结果张量
  return result;
}
} // namespace

// 使用 `unary_op_inplace` 执行正态分布的稀疏压缩张量的操作
Tensor& normal_sparse_csr_(
    Tensor& self,
    double mean,
    double std,
    std::optional<Generator> gen) {
  return unary_op_inplace(self, &Tensor::normal_, mean, std, gen);
}

// 使用 `unary_op_inplace` 执行填充稀疏压缩张量的操作
Tensor& fill_sparse_csr_(Tensor& self, const Scalar& value) {
  return unary_op_inplace(self, &TensorBase::fill_, value);
}

// 使用稀疏压缩张量 `self` 和 `mask` 执行稀疏掩码操作
Tensor sparse_mask_sparse_compressed(
    const Tensor& self,
    const Tensor& mask) {
  // 检查 `mask` 是否具有稀疏压缩布局
  TORCH_CHECK(at::sparse_csr::is_sparse_compressed(mask),
              "sparse_mask_sparse_compressed expects mask to have sparse compressed layout, got ", mask.layout());
  // 检查 `self` 和 `mask` 的尺寸是否兼容
  TORCH_CHECK(
      mask.sizes().equals(self.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      self.sizes(),
      " but mask has size ",
      mask.sizes());

  // 如果 `self` 与 `mask` 是同一个张量，则直接返回 `self`
  if (self.is_same(mask)) {
    return self;
  }

  // 如果 `mask` 张量为空或者没有非零元素，则克隆 `mask` 并按需移动到 `self` 的设备和标量类型
  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(self.device(), self.scalar_type());
  }

  // 如果 `self` 具有标准布局
  if (self.layout() == kStrided) {
    // 获取稀疏压缩的纯粹索引和压缩索引
    auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(mask);
    auto mask_values = mask.values();


这段代码涉及了稀疏压缩张量的操作，包括稀疏掩码、一元操作和二元操作，注释详细解释了每行代码的功能和目的。
    // 使用稀疏张量的压缩索引、普通索引和展开的布尔掩码值，创建稠密掩码
    auto dense_mask = at::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        at::ones({1}, self.options().dtype(kBool)).expand_as(mask_values),
        self.sizes(),
        self.options().dtype(kBool).layout(mask.layout())).to_dense();
    // 根据掩码的布局分发密集到稀疏的转换，返回稀疏张量
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          // 调用native中的函数，将self稠密张量应用dense_mask作为掩码转换为稀疏张量
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), {}, mask.dense_dim());
        },
        [&] {
          // 获取块大小，调用native中的函数，将self稠密张量应用dense_mask作为掩码转换为稀疏张量
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), blocksize, mask.dense_dim());
        });
  } else if (self.layout() == mask.layout()) {
    // TODO: 保留此处是为了向后兼容，但此处使用的方法可能导致索引不正确。
    // 返回self与mask逐元素相乘后的结果，转换为self的标量类型
    return self.mul(at::ones_like(mask)).to(self.scalar_type());
  } else {
    // TODO: 保留此处是为了向后兼容，但此处使用的方法不能支持批次维度，因为稀疏COO张量无法识别批次维度。
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          // 将self应用mask的稀疏版本作为掩码，转换为与mask布局相同的稀疏张量
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout());
        },
        [&] {
          // 获取块大小，将self应用mask的稀疏版本作为掩码，转换为与mask布局相同的稀疏张量
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout(), blocksize);
        });
  }
}

// 定义一个函数，对稀疏 CSR 格式的 Tensor 进行标量乘法操作
Tensor mul_scalar_sparse_csr(const Tensor& self, const Scalar& other) {
  // 计算乘法后的结果值
  auto result_values = self.values().mul(other);
  // 调用底层的不安全方法创建稀疏 CSR Tensor
  return at::native::_sparse_csr_tensor_unsafe(
      self.crow_indices().clone(),  // 克隆行索引数据
      self.col_indices().clone(),   // 克隆列索引数据
      result_values,                // 乘法结果值
      self.sizes(),                 // Tensor 的尺寸
      result_values.scalar_type(),  // 乘法结果的数据类型
      self.layout(),                // Tensor 的布局
      result_values.device());      // 乘法结果的设备
}

// 对稀疏 CSR 格式的 Tensor 进行清零操作
Tensor& zero_sparse_csr_(Tensor& self) {
  /*
    csr.zero_() resets nnz to 0.

    If the original sparsity pattern needs to be preserved, use
    `csr.values().zero_()` instead.

    The above behavior also implies that torch.zeros_like(csr) returns
    a new tensor with nnz == 0. If one needs a zeros_like semantics
    where the result has the same sparsity pattern as input, then use
    `result = csr.clone(); result.values.zero_();`
  */
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "zero_sparse_csr_", [](){});
  // 调用稀疏 CSR 实现的方法，重置并清空 Tensor
  get_sparse_csr_impl(self)->resize_and_clear_(self.sparse_dim(), self.dense_dim(), self.sizes());
  return self;  // 返回操作后的 Tensor 引用
}

/* 实现一元 Ufuncs，仅支持稀疏 CSR 布局的简单函数，其结果与 0->0 对应 */

// 定义宏以创建一元 Ufunc 函数，支持输出参数的形式
#define CREATE_UNARY_UFUNC_OUT(op_name)                                  \
  Tensor& op_name##_sparse_csr_out(const Tensor& self, Tensor& result) { \
    return unary_op_out(&at::op_name##_outf, self, result);              \
  }

// 定义宏以创建一元 Ufunc 函数，返回新的 Tensor 作为结果
#define CREATE_UNARY_UFUNC_FUNCTIONAL(op_name)                 \
  Tensor op_name##_sparse_csr(const Tensor& self) {            \
    return get_result_tensor_for_unary_op(&at::op_name, self); \
  }

// 定义宏以创建一元 Ufunc 函数，支持就地操作
#define CREATE_UNARY_UFUNC_INPLACE(op_name)             \
  Tensor& op_name##_sparse_csr_(Tensor& self) {         \
    return unary_op_inplace(self, &Tensor::op_name##_); \
  }

// 定义宏以创建一元 Ufunc 函数，包含所有支持的操作
#define CREATE_UNARY_UFUNC(op_name)       \
  CREATE_UNARY_UFUNC_OUT(op_name);        \
  CREATE_UNARY_UFUNC_FUNCTIONAL(op_name); \
  CREATE_UNARY_UFUNC_INPLACE(op_name);

// 定义宏以创建不支持就地操作的一元 Ufunc 函数
#define CREATE_UNARY_UFUNC_NO_INPLACE(op_name) \
  CREATE_UNARY_UFUNC_OUT(op_name);             \
  CREATE_UNARY_UFUNC_FUNCTIONAL(op_name);

// 枚举所有稀疏压缩格式支持的一元 Ufuncs
CREATE_UNARY_UFUNC(abs);
CREATE_UNARY_UFUNC(asin);
CREATE_UNARY_UFUNC(asinh);
CREATE_UNARY_UFUNC(atan);
CREATE_UNARY_UFUNC(atanh);
CREATE_UNARY_UFUNC(ceil);
CREATE_UNARY_UFUNC(deg2rad);
CREATE_UNARY_UFUNC(erf);
CREATE_UNARY_UFUNC(erfinv);
CREATE_UNARY_UFUNC(expm1);
CREATE_UNARY_UFUNC(floor);
CREATE_UNARY_UFUNC(frac);
CREATE_UNARY_UFUNC(log1p);
CREATE_UNARY_UFUNC(neg);
CREATE_UNARY_UFUNC(rad2deg);
CREATE_UNARY_UFUNC(sign);
CREATE_UNARY_UFUNC(sin);
CREATE_UNARY_UFUNC(sinh);
CREATE_UNARY_UFUNC(sgn);
CREATE_UNARY_UFUNC(sqrt);
CREATE_UNARY_UFUNC(tan);
CREATE_UNARY_UFUNC(tanh);
CREATE_UNARY_UFUNC(trunc);
CREATE_UNARY_UFUNC(conj_physical);

// 用于支持的特定 Ufunc，启用了警告忽略
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")
static CREATE_UNARY_UFUNC(relu);
C10_DIAGNOSTIC_POP()
// 使用新的 `round.decimals` 重载后，使用 CREATE_UNARY_UFUNC 导致无法解析重载。
Tensor& round_sparse_csr_out(const Tensor& self, Tensor& result) {
  // 调用 unary_op_out 函数，对稀疏 CSR 张量进行舍入操作，并将结果存入 result 中
  return unary_op_out(&at::_ops::round_out::call, self, result);
}

// 对稀疏 CSR 张量进行舍入操作，返回结果张量
Tensor round_sparse_csr(const Tensor& self) {
  // 调用 get_result_tensor_for_unary_op 函数，返回执行舍入操作后的结果张量
  return get_result_tensor_for_unary_op(&at::_ops::round::call, self);
}

// 在原地对稀疏 CSR 张量进行舍入操作
Tensor& round_sparse_csr_(Tensor& self) {
  // 内部断言，确保张量是稀疏 CSR 类型
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  // 对稀疏 CSR 张量的值部分进行舍入操作
  self.values().round_();
  // 返回原地操作后的张量自身
  return self;
}

// 在稀疏压缩格式张量上执行阈值反向传播
Tensor threshold_backward_sparse_compressed(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  // 调用 get_result_tensor_for_unary_op 函数，使用 lambda 函数执行阈值反向传播操作
  return get_result_tensor_for_unary_op(
      [&](const Tensor& t) {
        return at::threshold_backward(t, self.values(), threshold);
      },
      grad_output);
}

// 在稀疏压缩格式张量上执行带输出张量的阈值反向传播
Tensor& threshold_backward_sparse_compressed_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  // 调用 unary_op_out 函数，使用 lambda 函数执行带输出张量的阈值反向传播操作
  return unary_op_out(
      [&](const Tensor& t, Tensor& out) {
        return at::threshold_backward_outf(t, self.values(), threshold, out);
      },
      grad_output,
      grad_input);
}

// `angle`、`isneginf`、`isposinf` 和 `signbit` 目前没有原地操作的变体
CREATE_UNARY_UFUNC_NO_INPLACE(angle);

CREATE_UNARY_UFUNC_NO_INPLACE(isneginf);

CREATE_UNARY_UFUNC_NO_INPLACE(isposinf);

CREATE_UNARY_UFUNC_NO_INPLACE(signbit);

// `isnan` 和 `isinf` 没有输出张量的变体
CREATE_UNARY_UFUNC_FUNCTIONAL(isnan);

CREATE_UNARY_UFUNC_FUNCTIONAL(isinf);

template <typename scalar_t>
void addmm_out_sparse_csr_native_cpu(
    const Tensor& sparse,
    const Tensor& dense,
    const Tensor& r,
    Scalar alpha,
    Scalar beta) {
  // 获取稀疏矩阵的行数和稠密矩阵的列数
  auto dim_i = sparse.size(0);
  auto dim_k = dense.size(1);

  // 获取稀疏矩阵的CSR格式的行偏移数组
  auto csr = sparse.crow_indices();
  // 获取稀疏矩阵的列索引数组
  auto col_indices = sparse.col_indices();
  // 获取稀疏矩阵的值数组
  auto values = sparse.values();

  // 将 alpha 转换为与值类型相同的标量类型
  scalar_t cast_alpha = alpha.to<scalar_t>();
  // 将结果矩阵 r 乘以 beta
  r.mul_(beta);
  // 根据列索引的数据类型分发任务，并命名任务为 "csr_mm_crow_indices"
  AT_DISPATCH_INDEX_TYPES(
      col_indices.scalar_type(), "csr_mm_crow_indices", [&]() {
        // 获取CSR格式行偏移数组的访问器
        auto csr_accessor = csr.accessor<index_t, 1>();
        // 获取列索引数组的访问器
        auto col_indices_accessor = col_indices.accessor<index_t, 1>();

        // 获取稀疏矩阵的值数组的访问器
        auto values_accessor = values.accessor<scalar_t, 1>();
        // 获取稠密矩阵的数据指针
        scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
        // 获取结果矩阵 r 的数据指针
        scalar_t* r_ptr = r.data_ptr<scalar_t>();

        // 获取稠密矩阵的步长
        int64_t dense_stride0 = dense.stride(0);
        int64_t dense_stride1 = dense.stride(1);
        // 获取结果矩阵 r 的步长
        int64_t r_stride0 = r.stride(0);
        int64_t r_stride1 = r.stride(1);

        // 使用并行化方法处理任务
        at::parallel_for(
            0,
            dim_i,
            internal::GRAIN_SIZE,
            [&](int64_t irow_start, int64_t irow_end) {
              // 遍历每行稀疏矩阵的行偏移范围
              for (index_t h = irow_start; h < irow_end; ++h) {
                // 获取当前行的起始和结束索引
                index_t i_start = csr_accessor[h];
                index_t i_end = csr_accessor[h + 1];
                // 遍历当前行中的每个元素
                for (index_t i = i_start; i < i_end; i++) {
                  // 获取稀疏矩阵中的值和对应的列索引
                  scalar_t val = values_accessor[i];
                  index_t col = col_indices_accessor[i];
                  // 调用AXPY操作，更新结果矩阵 r
                  at::native::cpublas::axpy<scalar_t>(
                      dim_k,
                      cast_alpha * val,
                      dense_ptr + col * dense_stride0,
                      dense_stride1,
                      r_ptr + h * r_stride0,
                      r_stride1);
                }
              }
            });
      });
}

// 矩阵乘法的函数。
// result = beta * self + alpha (mat1 @ mat2)
Tensor& addmm_out_sparse_compressed_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // TODO: remove this, there are no codegenerated checks for devices yet
  // 检查 self 张量是否在 CPU 上
  sparse::impl::_check_is_cpu(self, "self");
  // 检查 mat1 张量是否在 CPU 上
  sparse::impl::_check_is_cpu(mat1, "mat1");
  // 检查 mat2 张量是否在 CPU 上
  sparse::impl::_check_is_cpu(mat2, "mat2");
  // 检查 result 张量是否在 CPU 上
  sparse::impl::_check_is_cpu(result, "result");

  // 所有的检查来自于 addmm_out_cuda_impl (ATen/native/cuda/Blas.cpp) 和
  // TORCH_META_FUNC(addmm) (ATen/native/LinearAlgebra.cpp)
  // TODO: 移除重复的代码，并统一代码
  // 检查 mat1 张量是否为二维
  sparse::impl::_check_dim(mat1, 2, "mat1");
  // 检查 mat2 张量是否为二维
  sparse::impl::_check_dim(mat2, 2, "mat2");

  // 检查 mat1 和 mat2 的形状是否可以相乘
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1.size(0), "x", mat1.size(1), " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  c10::MaybeOwned<at::Tensor> self_;
  // 如果这是一个原地操作，则不扩展 self
  if (&result == &self) {
     self_ = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
     // 根据 mat1 和 mat2 的形状扩展 self
     self_ = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
  }

  // 检查输入的 self 张量是否符合要求的形状
  TORCH_CHECK(((self_->dim() == 2) &&
               (self_->size(0) == mat1.size(0)) &&
               (self_->size(1) == mat2.size(1))),
              "The input tensor must be a matrix with size ",
              mat1.size(0),
              "x",
              mat2.size(1),
              ", but got a ",
              self_->dim(),
              "-D tensor with size ",
              self_->size(0),
              "x",
              self_->size(1));

  // 如果 result 不是 self，则调整 result 的大小以匹配 self，并复制 self 的内容到 result
  if (&result != &self) {
    if (result.layout() == kStrided) {
      at::native::resize_output(result, self_->sizes());
    } else {
      result.resize_as_sparse_(*self_);
    }
    result.copy_(*self_);
  }

  // 如果 result 张量的元素数为 0，则将其置零并返回
  if (result.numel() == 0) {
    // 如果 result 被调整大小并且是稀疏压缩的，
    // 其压缩索引张量将包含垃圾值，
    // 因此整个张量不是有效的压缩张量。
    // 为了解决这个问题，需要将 result 置零。
    if (at::sparse_csr::is_sparse_compressed(result)) {
      result.zero_();
    }
    return result;
  }

  // 如果 mat1 或 mat2 是稀疏且为零，则根据 beta 的值执行不同操作
  if (sparse::impl::_is_sparse_and_zero(mat1) || sparse::impl::_is_sparse_and_zero(mat2)) {
    // 根据文档，当 beta==0 时，应忽略 self 中的值。
    // NaN 和 Inf 不应传播。
    if (beta.toComplexDouble() == 0.) {
      result.zero_();
    } else {
      result.mul_(beta);
    }
    return result;
  }

#if !AT_USE_MKL_SPARSE()
  // 自定义实现 addmm_out_sparse_csr_native_cpu 仅支持 CSR @ strided -> strided
  // 如果 mat1 的布局是 strided
  if (mat1.layout() == kStrided) {
    # 如果 mat2 的布局是稀疏的 CSR 格式
    if (mat2.layout() == kSparseCsr) {
      # 如果结果 result 的布局是步进的
      if (result.layout() == kStrided) {
        # 针对 result 中浮点数和复数类型，执行以下操作
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              # 调用本地 CPU 函数 addmm_out_sparse_csr_native_cpu，处理稀疏 CSR 格式
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat2.transpose(-2, -1).to_sparse_csr(),  # 将 mat2 转置为 CSR 格式
                  mat1.transpose(-2, -1),  # mat1 转置
                  result.transpose(-2, -1),  # result 转置
                  alpha,  # 参数 alpha
                  beta);  # 参数 beta
            });
        # 返回处理后的 result
        return result;
      }
    }
    # 如果 mat2 的布局是稀疏的 CSC 格式
    if (mat2.layout() == kSparseCsc) {
      # 如果结果 result 的布局是步进的
      if (result.layout() == kStrided) {
        # 针对 result 中浮点数和复数类型，执行以下操作
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              # 调用本地 CPU 函数 addmm_out_sparse_csr_native_cpu，处理稀疏 CSR 格式
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat2.transpose(-2, -1),  # 将 mat2 转置
                  mat1.transpose(-2, -1),  # mat1 转置
                  result.transpose(-2, -1),  # result 转置
                  alpha,  # 参数 alpha
                  beta);  # 参数 beta
            });
        # 返回处理后的 result
        return result;
      }
    }
  } else if (mat1.layout() == kSparseCsr) {
    # 如果 mat1 的布局是稀疏的 CSR 格式
    if (mat2.layout() == kStrided) {
      # 如果 mat2 的布局是步进的
      if (result.layout() == kStrided) {
        # 针对 result 中浮点数和复数类型，执行以下操作
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              # 调用本地 CPU 函数 addmm_out_sparse_csr_native_cpu，处理稀疏 CSR 格式
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat1,  # mat1
                  mat2,  # mat2
                  result,  # result
                  alpha,  # 参数 alpha
                  beta);  # 参数 beta
            });
        # 返回处理后的 result
        return result;
      }
    }
  } else if (mat1.layout() == kSparseCsc) {
    # 如果 mat1 的布局是稀疏的 CSC 格式
    if (mat2.layout() == kStrided) {
      # 如果 mat2 的布局是步进的
      if (result.layout() == kStrided) {
        # 针对 result 中浮点数和复数类型，执行以下操作
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              # 调用本地 CPU 函数 addmm_out_sparse_csr_native_cpu，处理稀疏 CSR 格式
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat1.to_sparse_csr(),  # 将 mat1 转换为 CSR 格式
                  mat2,  # mat2
                  result,  # result
                  alpha,  # 参数 alpha
                  beta);  # 参数 beta
            });
        # 返回处理后的 result
        return result;
      }
    }
  }
  # 如果以上条件都不满足，抛出错误
  TORCH_CHECK(
      false,
      "addmm: computation on CPU is not implemented for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout(),
      " without MKL. PyTorch built with MKL has better support for addmm with sparse CPU tensors.");
#else
  sparse::impl::mkl::addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
#endif
  return result;
}


// 如果未定义条件编译宏#else，则调用 MKL 库中的稀疏矩阵乘法函数 sparse::impl::mkl::addmm_out_sparse_csr，
// 传入 mat1、mat2、beta、alpha 和 result，并返回 result。



Tensor addmm_sparse_compressed_dense(
    const Tensor& self,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor r = at::empty({0, 0}, self.options());
  at::addmm_out(r, self, sparse, dense, beta, alpha);
  return r;
}


// 使用稀疏 CSR 张量和稠密张量进行矩阵乘法，返回结果张量 r。
Tensor addmm_sparse_compressed_dense(
    const Tensor& self,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  // 创建一个空的张量 r，形状为 {0, 0}，使用 self 的选项。
  Tensor r = at::empty({0, 0}, self.options());
  // 调用 addmm_out 函数，计算 self + (sparse * dense * alpha) + (r * beta) 的结果，并存储到 r 中。
  at::addmm_out(r, self, sparse, dense, beta, alpha);
  // 返回结果张量 r。
  return r;
}



Tensor& _sparse_csr_mm_out(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result) {
  auto zero = at::zeros_like(result);
  return at::addmm_out(result, zero, mat1, mat2, 0.0, 1.0);
}


// 计算稀疏 CSR 张量 mat1 与 mat2 的矩阵乘法，并将结果存储到 result 中。
Tensor& _sparse_csr_mm_out(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result) {
  // 创建一个与 result 相同形状的零张量 zero。
  auto zero = at::zeros_like(result);
  // 调用 addmm_out 函数，计算 (zero * 0.0) + (mat1 * mat2 * 1.0)，将结果存储到 result 中。
  return at::addmm_out(result, zero, mat1, mat2, 0.0, 1.0);
}



Tensor _sparse_csr_mm(const Tensor& mat1, const Tensor& mat2) {
  if (mat1.is_sparse_csr() && mat2.is_sparse_csr()) {
    // Return sparse
    return at::addmm(
        at::zeros({mat1.size(0), mat2.size(1)}, mat2.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  if ((mat1.layout() == kSparseCsc || mat1.layout() == kSparseCsr) &&
      (mat2.layout() == kSparseCsc || mat2.layout() == kSparseCsr)) {
    // TODO: Expensive conversion to CSR. Should add native support for CSC.
    // Covers CSC @ CSR
    // Covers CSR @ CSC
    // Covers CSC @ CSC
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2.to_sparse_csr());
  }
  if (mat1.layout() == kSparseCsc && mat2.layout() == c10::kStrided) {
    // TODO: This is a costly conversion. We should have
    // native support for CSC.
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2);
  }
  // Default to taking options from mat1
  auto result_options = mat1.options();
  if (mat2.layout() == kStrided) {
    // if either  arg is strided we return strided, so update the options if
    // mat2 is strided.
    result_options = result_options.layout(kStrided);
  }
  return at::addmm(
      at::zeros({mat1.size(0), mat2.size(1)}, result_options),
      mat1,
      mat2,
      0.0,
      1.0);
}


// 计算稀疏矩阵 mat1 与 mat2 的乘法操作，根据输入张量的布局进行不同的处理并返回结果张量。
Tensor _sparse_csr_mm(const Tensor& mat1, const Tensor& mat2) {
  if (mat1.is_sparse_csr() && mat2.is_sparse_csr()) {
    // 如果 mat1 和 mat2 都是稀疏 CSR 张量，则返回稀疏结果。
    return at::addmm(
        at::zeros({mat1.size(0), mat2.size(1)}, mat2.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  if ((mat1.layout() == kSparseCsc || mat1.layout() == kSparseCsr) &&
      (mat2.layout() == kSparseCsc || mat2.layout() == kSparseCsr)) {
    // 如果 mat1 和 mat2 的布局是 CSR 或 CSC，则调用 _sparse_csr_mm 转换为 CSR 后进行计算。
    // 包括 CSR @ CSR、CSC @ CSR 和 CSC @ CSC。
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2.to_sparse_csr());
  }
  if (mat1.layout() == kSparseCsc && mat2.layout() == c10::kStrided) {
    // 如果 mat1 是 CSC 而 mat2 是 Strided，则调用 _sparse_csr_mm 转换为 CSR 后进行计算。
    // 这是一个高代价的转换。
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2);
  }
  // 默认情况下，采用来自 mat1 的选项。
  auto result_options = mat1.options();
  if (mat2.layout() == kStrided) {
    // 如果 mat2 是 Strided，则结果张量的布局也是 Strided。
    result_options = result_options.layout(kStrided);
  }
  // 返回 mat1 与 mat2 的乘法结果，结果张量形状为 {mat1.size(0), mat2.size(1)}。
  return at::addmm(
      at::zeros({mat1.size(0), mat2.size(1)}, result_options),
      mat1,
      mat2,
      0.0,
      1.0);
}



// Functions for element-wise addition.
Tensor add_sparse_csr(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result;
  if (self.layout() != kStrided && other.layout() == kStrided) {
    // add(sparse, dense) -> dense
    // 如果 self 是稀疏布局而 other 是稠密布局，则返回一个与 other 形状相同、dtype 与 self 与 other 公共的类型一致的张量。
    result = at::empty_like(
        other,
        other.options()
            .dtype(commonDtype)
            .memory_format(at::MemoryFormat::Contiguous));
  } else {
    // add(dense, sparse) -> dense AND add(sparse, sparse) -> sparse
    // 其它情况下，返回一个与 self 形状相同、dtype 与 self 与 other 公共的类型一致的张量。
    result = at::empty_like(
        self,
        self.options()
            .dtype(commonDtype)
            .memory_format(at::MemoryFormat::Contiguous));
  }
  // 调用 add_out 函数，计算 self + other * alpha 的结果，并返回结果张量。
  return at::add_out(result, self, other, alpha); // redispatch!
}


// 执行元素级的稀疏张量加法操作。
Tensor add_sparse_csr(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  // 确定 self 和 other 的公共数据类型。
  auto commonDtype = at::result_type(self, other);
  // 检查 alpha 是否符合数据类型要求。
  alpha_check(commonDtype, alpha);
  Tensor result;
  if (self.layout() != kStrided && other.layout() == kStrided) {
    // 如果 self 的布局不是 Strided 而 other 的布局是 Strided，则返回一个与 other 形状相
    // 确保 dense 张量的布局为 kStrided
    TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
    // 确保 src 张量的布局为 kSparseCsr 或 kSparseCsc
    TORCH_INTERNAL_ASSERT(
        src.layout() == kSparseCsr || src.layout() == kSparseCsc);
    // 确保 dense 张量位于 CPU 设备
    TORCH_INTERNAL_ASSERT(dense.device() == kCPU);

    // 检查输出张量 out 必须是连续的
    TORCH_CHECK(
        out.is_contiguous(),
        "out argument must be contiguous, but got: ",
        out.suggest_memory_format());
    // 检查输出张量 out 必须位于 CPU 设备
    TORCH_CHECK(
        out.device() == kCPU,
        "add: expected 'out' to be CPU tensor, but got tensor on device: ",
        out.device());
    // 检查输入张量 src 必须位于 CPU 设备
    TORCH_CHECK(
        src.device() == kCPU,
        "add: expected 'other' to be a CPU tensor, but got tensor on device: ",
        src.device());

    // 检查 dense 和 src 张量必须具有相同的尺寸
    TORCH_CHECK(
        dense.sizes().equals(src.sizes()),
        "add: expected 'self' and 'other' to have same size, but self has size ",
        dense.sizes(),
        " while other has size ",
        src.sizes(),
        " (FYI: op2-sparse addition does not currently support broadcasting)");

    // 推广 dense 和 src 张量的数据类型，并检查是否能够转换到输出张量的数据类型
    auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());
    TORCH_CHECK(
        canCast(commonDtype, out.scalar_type()),
        "Can't convert result type ",
        commonDtype,
        " to output ",
        out.scalar_type(),
        " in add operation");

    // 获取 src 张量的值
    auto src_values = src.values();

    // 调整输出张量 out 的尺寸以匹配 dense 张量
    resize_output(out, dense.sizes());

    // 将结果缓冲设置为 out 张量
    Tensor resultBuffer = out;

    // 如果 out 张量的数据类型与 commonDtype 不同，将 dense 张量转换为 commonDtype
    if (out.scalar_type() != commonDtype) {
        resultBuffer = dense.to(commonDtype);
    }
    // 如果 out 张量与 dense 张量不是同一个张量，则复制 dense 张量的值到 resultBuffer
    else if (!is_same_tensor(out, dense)) {
        resultBuffer.copy_(dense);
    }

    // 如果 src 张量的非零元素数量为 0，则直接将 resultBuffer 的值复制到 out 张量
    if (src._nnz() == 0) {
        out.copy_(resultBuffer);
    }
// 定义一个函数，对稀疏 CSR 张量在第一维度上进行降维操作，使用指定的规约运算符 ReductionOp
template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim0_cpu_template(const Tensor& sparse, ReductionOp rop) {
  /*
    考虑以下稀疏张量：

    1 * * * *
    * * * 2 *
    * * 3 * *
    * * * * *
    4 * 5 * *

    它的 CSR 表示为

      crow_indices = [0, 1, 2, 3, 3, 5]
      col_indices = [0, 3, 2, 0, 2]
      values = [1, 2, 3, 4, 5]

    对于 dim=0 的降维结果为：

    rop(1, 4) * rop(3, 5) 2 *

    其 CSR 表示为

      new_crow_indices = [0, 3]
      new_col_indices = [0, 2, 3]
      new_values = [rop(1, 4], rop(3, 5), 2]
  */
    In general, the CSR representation data can be computed as follows:
  
      new_col_indices, col_map = col_indices.unique(sorted=True, return_inverse=True)
      nnz = new_col_indices.numel()
      new_crow_indices = [0, nnz]
      new_values.resize(nnz); new_values.fill_(identity)
      for i in range(col_indices.numel()):
          new_values[col_map[i]] = rop(new_values[col_map[i], values[i])
   */

  // 获取稀疏张量的列索引和数值
  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto numel = values.numel();

  /*
    Calling at::_unique constitutes the main bottleneck of this
    function. However, it is still about 5x faster than using the
    invariant:
      csr.sum(dim=0) == csr.transpose(0, 1).sum(dim=1)
  */
  // 使用 at::_unique 函数对列索引进行去重，返回新的列索引和映射关系
  auto [new_col_indices, columns_map] = at::_unique(col_indices, true, true);
  auto nnz = new_col_indices.numel();

  // 创建新的行索引数组，初始值为 [0, nnz]
  Tensor new_crow_indices = at::empty({2}, col_indices.options());
  new_crow_indices[0] = 0;
  new_crow_indices[1] = nnz;

  // 在 CPU 后端中，将累加类型的 is_cuda 设置为 true。因为当前场景下，float 的累加类型应为 float。
  // 在 CUDA 中，float 是 float 的累加类型，而在 CPU 中，double 是 float 的累加类型。
  using acc_t = at::acc_type<scalar_t, true>;
  // 创建用于累加的缓冲区
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  // 将累加缓冲区的值初始化为 rop 的 identity
  new_values_acc.fill_(rop.identity());

  // 获取列映射指针和数值指针
  int64_t* columns_map_ptr = columns_map.data_ptr<int64_t>();
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr =
      new_values_acc.data_ptr<acc_t>();

  // 下面的 for 循环没有并行化的必要，因为99.3%的计算时间已经花费在上面的 at::_unique 调用中。
  // 对稀疏张量的每个元素进行处理，更新新值的累加缓冲区
  for (const auto i : c10::irange(numel)) {
    int64_t col = columns_map_ptr[i];
    scalar_t val = values_ptr[i];
    new_values_acc_ptr[col] = rop(new_values_acc_ptr[col], static_cast<acc_t>(val));
  }
  // 从累加缓冲区复制值到新的值数组
  copy_from_acc_buffer(new_values, new_values_acc);

  // 构造并返回新的稀疏 CSR 张量
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                              {1, sparse.size(1)},
                                              new_values.scalar_type(),
                                              sparse.layout(),
                                              new_values.device());
  // 模板函数 reduce_sparse_csr_dim1_cpu_template 接受稀疏张量 sparse 和归约操作 rop
template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim1_cpu_template(const Tensor& sparse, ReductionOp rop) {
  /*
    考虑以下稀疏张量：

    1 * * * *
    * * * 2 *
    * * 3 * *
    * * * * *
    4 * 5 * *

    其CSR表示为

      crow_indices = [0, 1, 2, 3, 3, 5]
      col_indices = [0, 3, 2, 0, 2]
      values = [1, 2, 3, 4, 5]

    维度 dim=1 的归约结果为：

    1
    2
    3
    *
    rop(4, 5)

    其CSR表示为

      new_crow_indices = [0, 1, 2, 3, 3, 4]
      new_col_indices = [0, 0, 0, 0]
      new_values = [1, 2, 3, rop(4, 5)]

    通常，结果CSR数据可以按以下方式计算：

      new_crow_indices = [0]
      for i in range(1, nrows+1):
          new_crow_indices[i] = new_crow_indices[i-1] + (crow_indices[i] == crow_indices[i-1])
      nnz = new_crow_indices[-1]
      new_col_indices = zeros(nnz)
      new_values.resize(nnz)
      j = -1
      for i in range(1, nrows+1):
          if crow_indices[i] == crow_indices[i-1]:
              continue
          j += 1
          new_values[j] = rop(values[crow_indices[i] : crow_indices[i-1]])
  */

  // 获取稀疏张量的 crow_indices、values 和行数 nrows
  Tensor crow_indices = sparse.crow_indices();
  auto ioptions = crow_indices.options();
  Tensor values = sparse.values();
  auto nrows = sparse.size(0);

  // 创建新的 crow_indices、col_indices 和 row_map 张量
  Tensor new_crow_indices = at::empty({crow_indices.numel()}, ioptions);
  Tensor new_col_indices = at::empty({}, ioptions);
  Tensor row_map = at::empty({nrows}, ioptions);

  // 在 CPU 后端中，设置 acc_type 的 is_cuda = true。因为在当前情况下，float 的累积类型应该是 float。
  // 在 CUDA 中，float 是 float 的累积类型，而在 CPU 中，double 是 float 的累积类型。
  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type());
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);

  // 根据 crow_indices 的类型分发索引类型，处理稀疏张量的行压缩索引
  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "reduce_sparse_csr_dim1_cpu_indices",
                          [&]() {
    index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
    index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
    index_t* row_map_ptr = row_map.data_ptr<index_t>();
    int64_t nnz = 0;
    new_crow_indices_ptr[0] = 0;
    for(int64_t i=0; i<nrows; i++) {
      if (crow_indices_ptr[i] != crow_indices_ptr[i + 1]) {
        row_map_ptr[i] = nnz;
        nnz++;
      }
      new_crow_indices_ptr[i + 1] = nnz;
    }
    new_col_indices.resize_(nnz);
    new_col_indices.fill_(index_t(0));
    new_values.resize_(nnz);
    new_values_acc.resize_(nnz);

    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
    # 使用 ATen 的 parallel_for 函数进行并行处理
    at::parallel_for(
        0,  # 循环的起始索引 irow_start
        nrows,  # 循环的结束索引 irow_end
        internal::GRAIN_SIZE,  # 并行任务的粒度
        [&](int64_t irow_start, int64_t irow_end) {  # Lambda 函数，接受起始和结束索引作为参数
            index_t i_end = crow_indices_ptr[irow_start];  # 初始化当前行结束的索引
            for (index_t h = irow_start; h < irow_end; ++h) {  # 外部循环遍历行索引范围
                index_t i_start = i_end;  # 记录当前行开始的索引
                i_end = crow_indices_ptr[h + 1];  # 更新当前行结束的索引
                if (i_start != i_end) {  # 如果当前行非空
                    acc_t res = static_cast<acc_t>(values_ptr[i_start]);  # 初始化结果为当前行第一个值
                    for (index_t i = i_start + 1; i < i_end; i++) {  # 内部循环遍历当前行的列索引范围
                        res = rop(res, static_cast<acc_t>(values_ptr[i]));  # 使用 rop 函数更新结果
                    }
                    new_values_acc_ptr[row_map_ptr[h]] = res;  # 将结果存入新值的累加指针中
                }
            }
        });

  copy_from_acc_buffer(new_values, new_values_acc);  # 将累加的新值复制到新值缓冲区中

  # 创建并返回一个新的稀疏 CSR 张量
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                                {sparse.size(0), 1},  # 稀疏张量的大小
                                                new_values.scalar_type(),  # 稀疏张量的数据类型
                                                sparse.layout(),  # 原始稀疏张量的布局
                                                new_values.device());  # 稀疏张量的设备类型
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_cpu_template(const Tensor& sparse, ReductionOp rop) {

  auto ioptions = sparse.col_indices().options();  // 获取稀疏张量的列索引选项
  Tensor values = sparse.values();  // 获取稀疏张量的值
  auto numel = values.numel();  // 获取稀疏张量的元素数量
  auto nnz = std::min<int64_t>(1, numel);  // 计算非零元素的数量，最少为1个

  /* TODO: we can likely do about 3x better than parallel_reduce:

In [2]: t=torch.randn(5000, 5000).to_sparse_csr()

In [3]: %timeit torch._sparse_csr_sum(t, dim=(0, 1), keepdim=True)
3.39 ms ± 898 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit torch.sum(t.values())
1.07 ms ± 291 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
  */

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;  // 使用带有CUDA的CPU后端的累加类型
  scalar_t* values_ptr = values.data_ptr<scalar_t>();  // 获取稀疏张量值的指针
  acc_t value = at::parallel_reduce(
                                       0,
                                       numel,
                                       internal::GRAIN_SIZE,
                                       rop.identity(),  // 获取归约操作的身份元素
                                       [&](int64_t i_start, int64_t i_end, scalar_t identity) {
                                         acc_t res = acc_t(identity);  // 初始化归约结果
                                         for (int64_t i=i_start; i<i_end; i++) {
                                           acc_t val = acc_t(values_ptr[i]);  // 获取当前值
                                           res = rop(res, val);  // 执行归约操作
                                         }
                                         return res;  // 返回最终结果
                                       }, rop
                                       );

  Tensor new_col_indices = at::zeros({nnz}, ioptions);  // 创建新的列索引张量
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);  // 创建新的行索引张量
  Tensor new_values;
  auto result_dtype = at::isIntegralType(values.scalar_type(), /*includeBool=*/true) ? ScalarType::Long : values.scalar_type();  // 确定结果张量的数据类型
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));  // 创建新的值张量
    new_values.fill_(value);  // 填充新的值张量
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));  // 创建空的标量值张量
  }
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, std::min<int64_t>(1, sparse.size(1))},  // 创建新的稀疏张量
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_cpu_template(const Tensor& sparse, std::vector<int64_t> dims, ReductionOp rop) {
  if (dims.size() == 1) {  // 如果维度大小为1
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_cpu_template<scalar_t>(sparse, rop);  // 调用维度0的稀疏张量归约模板
  } else {
    // 如果稀疏张量的维度大小为1，进行断言检查
    TORCH_INTERNAL_ASSERT(dims[0] == 1);
    // 调用模板函数，对稀疏张量进行维度1的CPU模板降维操作
    return reduce_sparse_csr_dim1_cpu_template<scalar_t>(sparse, rop);
  } else if (dims.size() == 2) {
    // 如果稀疏张量的维度大小为2，进行断言检查，要求维度组合为(0,1)或(1,0)
    TORCH_INTERNAL_ASSERT(((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    // 调用模板函数，对稀疏张量进行维度01的CPU模板降维操作
    return reduce_sparse_csr_dim01_cpu_template<scalar_t>(sparse, rop);
  }
  // 如果稀疏张量的维度为空，进行断言检查
  TORCH_INTERNAL_ASSERT(dims.empty());
  // 返回稀疏张量的克隆，这是在解决 gh-29137 问题之后的有效操作
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
// 定义了一个模板函数 reduce_sparse_csr_cpu_template，接受稀疏张量 sparse、要求和的维度 dims_to_sum、是否保持维度 keepdim、以及一个 ReductionOp 操作符 rop
Tensor reduce_sparse_csr_cpu_template(const Tensor& sparse, IntArrayRef dims_to_sum, bool keepdim, ReductionOp rop) {
  // 内部断言，确保输入的 sparse 张量是 CSR 稀疏张量
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  // 检查 keepdim 是否为 true，因为对 CSR 张量进行减少操作时不支持 keepdim 为 false
  TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
  // 内部断言，确保稀疏张量位于 CPU 设备上
  TORCH_INTERNAL_ASSERT(sparse.device() == kCPU);

  // 获取稀疏张量的维度数
  const int64_t input_dim = sparse.dim();
  // 内部断言，确保稀疏张量维度为 2
  TORCH_INTERNAL_ASSERT(input_dim == 2);
  // 将 dims_to_sum 转换为标准的维度数组，确保其与输入维度匹配
  auto dims = dims_to_sum.vec();
  maybe_wrap_dims(dims, input_dim);
  if (dims.empty()) {
    // 当 dims 为空时，在解决了 gh-29137 后，删除这个 if 块
    dims.emplace_back(0);
    dims.emplace_back(1);
  }
  // 调用模板函数自身，执行稀疏张量的减少操作并返回结果
  return reduce_sparse_csr_cpu_template<scalar_t>(sparse, dims, rop);
}

template <typename scalar_t>
// 定义了一个结构 ReductionAddOp，实现了加法操作符和身份元素
struct ReductionAddOp {
  inline scalar_t operator()(const scalar_t& a, const scalar_t& b) const {
    return a + b;
  }
  inline scalar_t identity() const { return 0; }
};

template <typename scalar_t>
// 定义了一个结构 ReductionMulOp，实现了乘法操作符和身份元素
struct ReductionMulOp {
  inline scalar_t operator()(const scalar_t& a, const scalar_t& b) const {
    return a * b;
  }
  inline scalar_t identity() const { return 1; }
};

}  // namespace

// 定义了一个函数 _sparse_csr_sum_cpu，用于计算 CSR 格式稀疏张量的和
Tensor _sparse_csr_sum_cpu(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
  // 获取或指定输入张量的数据类型
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  // 将输入张量转换为指定数据类型的 CSR 格式稀疏张量
  Tensor input_ = at::sparse_csr::to_type(input, dtype_);
  Tensor result;
  // 根据输入张量的数据类型分发到不同的类型处理器，并执行稀疏 CSR 张量的减少和操作，结果保存在 result 中
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_cpu", [&] {
        // 在 CPU 后端中，设置 acc_t 的 is_cuda 为 true。因为在当前场景下，float 的累积类型应为 float。
        // 在 CUDA 中，float 是 float 的累积类型，而在 CPU 中，double 是 float 的累积类型。
        using acc_t = at::acc_type<scalar_t, true>;
        result = reduce_sparse_csr_cpu_template<scalar_t>(
            input_, dims_to_sum, keepdim, ReductionAddOp<acc_t>());
      });
  // 返回结果张量
  return result;
}

// 定义了一个函数 _sparse_csr_prod_cpu，用于计算 CSR 格式稀疏张量的乘积
Tensor _sparse_csr_prod_cpu(const Tensor& input, IntArrayRef dims_to_reduce, bool keepdim, std::optional<ScalarType> dtype) {
  // 获取或指定输入张量的数据类型
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  // 将输入张量转换为指定数据类型
  Tensor input_ = input.to(dtype_);
  Tensor result;
  // 根据输入张量的数据类型分发到不同的类型处理器，并执行稀疏 CSR 张量的减少乘法操作，结果保存在 result 中
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_cpu",
    [&] {
      result = reduce_sparse_csr_cpu_template<scalar_t>(input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
    });
  // 返回结果张量
  return result;
}

// 定义了一个函数 _sparse_mm_reduce_impl_sparse_csr_cpu，用于执行 CSR 格式稀疏张量的稀疏矩阵乘法后的减少操作
std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& other,
    // 获取 self 张量的布局信息
    auto layout = self.layout();
    // 检查 self 是否为 SparseCsr 布局
    TORCH_CHECK(layout == kSparseCsr,
        "sparse_mm_reduce: expect self to be SparseCsr, got ", layout);
    // 检查 self 是否为纯稀疏张量（dense_dim 为 0）
    TORCH_CHECK(self.dense_dim() == 0,
        "sparse_mm_reduce: expected non-hybrid self tensor.");
    // 检查 self 是否为二维张量
    TORCH_CHECK(self.dim() == 2,
        "sparse_mm_reduce: expected self to be a 2-D tensor, got ", self.dim(), "-D tensor.");
    
    // 检查稀疏乘法 reduce 的输入条件
    sparse::impl::check_sparse_mm_reduce_impl_inputs</*train*/false>(
        self, Tensor(), other);
    
    // 获取 reduce 操作类型的枚举值
    auto op = get_reduction_enum(reduce);
    // 检查是否支持 REDUCTION_TYPE::PROD 的 reduce 类型
    TORCH_CHECK(op != ReductionType::PROD, "sparse_mm_reduce: reduce type of prod has not been enabled.")
    
    // 获取 self 的 crow_indices、col_indices 和 values
    auto crow = self.crow_indices();
    auto col = self.col_indices();
    auto val = self.values();
    
    // 初始化输出张量 out，所有元素初始化为零
    // 对于没有非零元素的 `rows`，输出中对应的行将是零。
    auto out = at::zeros({self.size(0), other.size(1)}, other.options());
    // 初始化 arg_out，长度为 0，使用与 col 相同的数据类型
    auto arg_out = at::empty({0}, col.options());
    
    // 获取 self 的非零元素个数
    int64_t nnz = self._nnz();
    // 如果 self 没有非零元素，直接返回零初始化的 out 和空的 arg_out
    if (nnz == 0) {
      return std::make_tuple(out, arg_out);
    }
    
    // 对于训练过程中需要计算 "amax" 和 "amin" reduce 类型的情况，
    // 需要计算 arg_out
    bool need_arg_out = at::GradMode::is_enabled()
        && (self.requires_grad() || other.requires_grad())
        && (op == ReductionType::MAX || op == ReductionType::MIN);
    
    // 如果不需要计算 arg_out，则调用 spmm_reduce_stub 计算输出 out
    if (!need_arg_out) {
      spmm_reduce_stub(kCPU, out, crow, col, val, other, op);
    } else {
      // 分配内存并用无效索引初始化 arg_out
      arg_out.resize_(out.sizes());
      arg_out.fill_(nnz);
      // 调用 spmm_reduce_arg_stub 计算输出 out 和 arg_out
      spmm_reduce_arg_stub(kCPU, out, arg_out, crow, col, val, other, op);
    }
    
    // 返回最终的输出，移动 out 和 arg_out 的所有权
    return std::make_tuple(std::move(out), std::move(arg_out));
}

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_backward_sparse_csr_cpu(
    const Tensor& self,  // 输入参数：稀疏矩阵自身
    const Tensor& grad_out,  // 输入参数：梯度输出
    const Tensor& other,  // 输入参数：另一个矩阵
    const c10::string_view reduce,  // 输入参数：指定的减少操作类型
    const Tensor& arg_out,  // 输入参数：输出参数
    std::array<bool, 2> output_mask) {  // 输入参数：输出掩码

  auto layout = self.layout();  // 获取输入稀疏矩阵的布局类型
  TORCH_CHECK(layout == kSparseCsr,
      "sparse_mm_reduce: expect self to be SparseCsr, got ", layout);  // 检查输入矩阵的布局类型是否为 SparseCsr

  sparse::impl::check_sparse_mm_reduce_impl_inputs</*train*/true>(
      self, grad_out, other);  // 检查稀疏矩阵乘法减少操作的输入参数是否合法

  auto op = get_reduction_enum(reduce);  // 获取减少操作的枚举值

  auto crow = self.crow_indices();  // 获取稀疏矩阵的行偏移索引
  auto col = self.col_indices();  // 获取稀疏矩阵的列索引
  auto val = self.values();  // 获取稀疏矩阵的值

  // `row`: COO 格式的行索引
  // `ccol`: CSC 格式的列索引（通过排列）
  // `permute`: 从CSR到CSC的排列模式
  //
  // TODO: 优化以下部分，
  // 当前 `argsort` 是顺序执行的。
  Tensor row, ccol, permute;
  {
    bool out_int32 = crow.scalar_type() == ScalarType::Int;  // 判断 crow 是否为 Int 类型
    Tensor coo_indices = at::_convert_indices_from_csr_to_coo(
        crow,
        col,
        out_int32,
        /*transpose*/false);  // 将CSR格式转换为COO格式的索引
    row = coo_indices.select(0, 0);  // 获取COO格式的行索引

    // 计算CSC格式的全局索引
    // 并获取转换的排列模式
    Tensor index = col.mul(self.size(0)).add_(row);  // 计算全局索引
    permute = index.argsort();  // 对索引进行排序，获取排列模式

    ccol = at::_convert_indices_from_coo_to_csr(
        /*column indices*/col.index_select(0, permute),  // 将列索引按排列模式选择
        /*column count*/self.size(1),
        out_int32);  // 将COO格式转换为CSR格式的列索引
  }

  Tensor grad_self, grad_other;
  if (output_mask[0]) {
    // grad_input 与输入具有相同的索引和非零值
    grad_self = at::empty_like(self);  // 创建一个与 self 类型和形状相同的空张量 grad_self
    grad_self.values().zero_();  // 将 grad_self 的值部分清零
    if (op == ReductionType::MAX || op == ReductionType::MIN) {
      spmm_reduce_backward_input_arg_stub(kCPU, grad_self, grad_out, col, other, arg_out, op);  // 调用稀疏矩阵乘法减少操作的反向输入函数
    } else {
      spmm_reduce_backward_input_stub(kCPU, grad_self, grad_out, crow, col, other, row, op);  // 调用稀疏矩阵乘法减少操作的反向输入函数
    }
  }
  if (output_mask[1]) {
    grad_other = at::zeros(other.sizes(), other.options());  // 创建一个与 other 具有相同形状和类型的零张量 grad_other
    if (op == ReductionType::MAX || op == ReductionType::MIN) {
      spmm_reduce_backward_other_arg_stub(kCPU, grad_other, grad_out, col, val, arg_out, op);  // 调用稀疏矩阵乘法减少操作的反向其他函数
    } else {
      spmm_reduce_backward_other_stub(kCPU, grad_other, grad_out, crow, val, row, ccol, permute, op);  // 调用稀疏矩阵乘法减少操作的反向其他函数
    }
  }

  return std::make_tuple(std::move(grad_self), std::move(grad_other));  // 返回 grad_self 和 grad_other 的元组
}

DEFINE_DISPATCH(spmm_reduce_stub);  // 定义稀疏矩阵乘法减少操作的分派函数
DEFINE_DISPATCH(spmm_reduce_arg_stub);  // 定义稀疏矩阵乘法减少操作的参数分派函数
DEFINE_DISPATCH(spmm_reduce_backward_input_stub);  // 定义稀疏矩阵乘法减少操作的反向输入分派函数
DEFINE_DISPATCH(spmm_reduce_backward_input_arg_stub);  // 定义稀疏矩阵乘法减少操作的反向输入参数分派函数
DEFINE_DISPATCH(spmm_reduce_backward_other_stub);  // 定义稀疏矩阵乘法减少操作的反向其他分派函数
DEFINE_DISPATCH(spmm_reduce_backward_other_arg_stub);  // 定义稀疏矩阵乘法减少操作的反向其他参数分派函数

} // namespace native
} // namespace at
```