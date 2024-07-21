# `.\pytorch\aten\src\ATen\native\sparse\SparseTensorMath.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIndexing.h>
#include <ATen/native/sparse/SparseTensorMath.h>

#include <c10/util/irange.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_addmm.h>
#include <ATen/ops/_sparse_addmm_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_mm_native.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_backward_native.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/_sparse_sparse_matmul.h>
#include <ATen/ops/_sparse_mm_reduce_impl.h>
#include <ATen/ops/_sparse_mm_reduce_impl_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/any.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/hspmm_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/native_norm_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/smm_native.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/sspaddmm_native.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/index.h>
#endif

#include <algorithm>

// 在 at::native 命名空间中定义代码
namespace at::native {

// 使用 at::sparse 命名空间
using namespace at::sparse;

// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// zero_sparse_ 函数的实现
// 参数为 SparseTensor 引用 self
SparseTensor& zero_sparse_(SparseTensor& self) {
  // 断言 self 是稀疏张量
  AT_ASSERT(self.is_sparse());
  // 调用 self 的 sparse_resize_and_clear_ 方法，设置稀疏张量的大小并清空
  self.sparse_resize_and_clear_(self.sizes(), self.sparse_dim(), self.dense_dim());
  // 返回自身的稀疏张量，并标记为 coalesced（合并）
  return self._coalesced_(true);
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// 注释结束
// --------------------------------------------------------------------
// mul_out_sparse_zerodim(SparseTensor&, const SparseTensor&, const Tensor&)
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  // 断言 r 和 t 都是稀疏张量
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  // 如果 value 是稀疏张量，将其值取出作为稠密张量处理
  Tensor value_;
  if (value.is_sparse()) {
    // 如果稀疏张量没有非零元素，则将 r 调整大小为 t 并返回零值稀疏张量
    if (value._nnz() == 0) {
      r.resize_as_(t);
      return r.zero_();
    }
    value_ = value.values();
  } else {
    value_ = value;
  }
  // 如果广播操作生效，value_ 可能是形状为 (1,) 的一维张量
  AT_ASSERT(value_.numel() == 1);

  // 如果 r 和 t 是同一个张量，则直接在 r 的值上进行乘法操作
  if (is_same_tensor(r, t)) {
    r._values().mul_(value_);
  } else {
    // 否则，将 r 调整大小为 t，并复制 t 的索引，然后进行乘法运算
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // 因为 mul_out 需要一个 Tensor&
    at::mul_out(r_values, t._values(), value_);
    // 更新稀疏张量的非零元素数量并紧缩存储
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

// --------------------------------------------------------------------
// mul_out_sparse_scalar(SparseTensor&, const SparseTensor&, const Scalar&)
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const Scalar& value) {
  return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// --------------------------------------------------------------------
// neg_out_sparse(const SparseTensor&, SparseTensor&)
// --------------------------------------------------------------------

SparseTensor& neg_out_sparse(const SparseTensor& t, SparseTensor& r) {
  // 断言 r 和 t 都是稀疏张量
  TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
  TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");

  // 如果 r 和 t 不是同一个张量，则复制 t 到 r
  copy_sparse_to_sparse_(r, t);
  // 对 r 的值取负
  r._values().neg_();
  return r;
}

// --------------------------------------------------------------------
// neg_sparse(const SparseTensor&)
// --------------------------------------------------------------------

SparseTensor neg_sparse(const SparseTensor& t) {
  // 创建一个与 t 相同形状的空张量 r
  SparseTensor r = at::empty_like(t);
  // 对 t 取负并将结果存入 r
  neg_out_sparse(t, r);
  return r;
}

// --------------------------------------------------------------------
// neg_sparse_(SparseTensor&)
// --------------------------------------------------------------------

SparseTensor& neg_sparse_(SparseTensor& t) {
  // 对 t 取负并将结果存入 t
  return neg_out_sparse(t, t);
}

// --------------------------------------------------------------------
// pow_out_sparse_scalar(const SparseTensor&, const Scalar&, SparseTensor&)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& pow_out_sparse_scalar(const SparseTensor& t_, const Scalar& value, SparseTensor& r) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t_.is_sparse());
  // 断言 value 不为零，否则会使结果张量变为稠密张量
  TORCH_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  // 对 t_ 进行紧缩操作以减少稀疏性
  SparseTensor t = t_.coalesce();

  // 将 r 调整为与 t 相同大小，并复制 t 的索引
  r.resize_as_(t);
  auto indices = r._indices();
  indices.resize_as_(t._indices());
  indices.copy_(t._indices());
  Tensor r_values = r._values(); // 因为 pow_out 需要一个 Tensor&
  at::pow_out(r_values, t._values(), value);
  // 更新稀疏张量的非零元素数量
  get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
  return r._coalesced_(t.is_coalesced());
}
SparseTensor pow_sparse_scalar(const SparseTensor& t, const Scalar& value) {
  // 创建一个空的稀疏张量 r，与输入张量 t 具有相同的选项
  SparseTensor r = at::empty({0}, t.options());
  // 调用 pow_out_sparse_scalar 函数计算 t 的 value 次幂并将结果存入 r
  pow_out_sparse_scalar(t, value, r);
  // 返回结果张量 r
  return r;
}

// --------------------------------------------------------------------
// coalesce(SparseTensor)
// --------------------------------------------------------------------

static SparseTensor& coalesce_(SparseTensor& tensor) {
  // 如果张量 tensor 已经是 coalesced（稠密化）的，则直接返回
  if (tensor.is_coalesced()) {
    return tensor;
  }

  // 对 tensor 进行 coalesce 操作，得到 coalesced 张量
  SparseTensor coalesced = tensor.coalesce();
  // 调整 tensor 的值张量和索引张量的大小以匹配 coalesced 张量
  tensor._values().resize_as_(coalesced._values());
  tensor._indices().resize_as_(coalesced._indices());
  // 将 coalesced 张量的值和索引复制到 tensor 的相应位置
  tensor._values().copy_(coalesced._values());
  tensor._indices().copy_(coalesced._indices());
  // 标记 tensor 已经被 coalesced
  tensor._coalesced_(true);
  // 返回处理后的 tensor
  return tensor;
}

// Note [Sparse Floor Division]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 未稠密化的稀疏张量无法正确进行 floor 除法。在此注释中，整数除法被视为 floor 除法的特例。
// 例如，数值为 [3, 3] 的整数张量除以 2 将产生数值 [1, 1]，而不是 3 (=6/2)。
// 数值为 [3., 3.] 的浮点张量除以 2 将产生数值 [1., 1.]（经过截断），而不是 3.f。
// 进行 floor 除法时，稀疏张量必须首先进行 coalesce 操作。
// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, std::optional<c10::string_view> rounding_mode, SparseTensor& r) {
  // 检查 value 的维度是否为 0，即确保其为标量或零维稠密张量
  TORCH_CHECK(value.dim() == 0, "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", value.sizes(), " for divisor)");
  // 检查 value 是否为稀疏张量，要求其为标量或零维稠密张量
  TORCH_CHECK(!value.is_sparse(), "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  // 断言输出张量 r 必须为稀疏张量
  AT_ASSERT(r.is_sparse());
  // 断言输入张量 t 必须为稀疏张量
  AT_ASSERT(t.is_sparse());

  // 查看是否需要进行 coalesce 操作，条件是 rounding_mode 存在且 t 尚未被 coalesced
  const bool should_coalesce = rounding_mode.has_value() && !t.is_coalesced();
  // 如果 r 和 t 是同一个张量，并且需要进行 coalesce，则执行 coalesce 操作
  if (is_same_tensor(r, t)) {
    if (should_coalesce) {
      coalesce_(r);
    }
    // 在 r 的值张量上执行除法操作，修改原地，使用指定的 rounding_mode
    r._values().div_(value, rounding_mode);
  } else {
    // 否则，创建 t_tmp 作为 t 的副本
    Tensor t_tmp = t;
    // 如果需要进行 coalesce 操作，则对 t_tmp 进行 coalesce
    if (should_coalesce) {
      t_tmp = t.coalesce();
    }
    // 调整 r 的大小以匹配 t_tmp
    r.resize_as_(t_tmp);
    // 复制 t_tmp 的索引到 r 的索引
    auto indices = r._indices();
    indices.resize_as_(t_tmp._indices());
    indices.copy_(t_tmp._indices());
    // 获取 r 的值张量，并在不改变其位置的情况下在 t_tmp 的值上执行除法
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    at::div_out(r_values, t_tmp._values(), value, rounding_mode);
    // 更新 r 的非零元素数量并收紧张量
    get_sparse_impl(r)->set_nnz_and_narrow(t_tmp._nnz());
    // 标记 r 已经被 coalesce（如果 t_tmp 为 coalesced）
    r._coalesced_(t_tmp.is_coalesced());
  }
  // 返回结果张量 r
  return r;
}

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, SparseTensor& r) {
  // 调用带有默认 rounding_mode 的 div_out_sparse_zerodim 函数
  return div_out_sparse_zerodim(t, value, /*rounding_mode=*/c10::nullopt, r);
}
// 定义一个函数，用于稀疏张量与标量或张量之间的除法操作
Tensor div_sparse(const Tensor& self, const Tensor& value) {
  // 确定输出张量的数据类型与输入张量和值张量的公共数据类型
  auto commonDtype = at::result_type(self, value);
  // 如果公共数据类型是整数类型（包括布尔型），则将其转换为默认的浮点数据类型
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true)) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  // 创建一个空的张量作为结果，使用公共数据类型，并且形状为空
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  // 调用 div_out_sparse_zerodim 函数，将结果存储在 result 中并返回
  return div_out_sparse_zerodim(self, value, result);
}

// 定义一个原地操作函数，用于稀疏张量与标量或张量之间的除法
Tensor& div_sparse_(Tensor& self, const Tensor& value) {
  // 调用 div_out_sparse_zerodim 函数，在 self 上进行原地操作，并返回 self 本身
  return div_out_sparse_zerodim(self, value, self);
}

// 定义一个函数，用于稀疏张量与标量或张量之间的除法操作，并考虑舍入模式
Tensor div_sparse(const Tensor& self, const Tensor& value, std::optional<c10::string_view> rounding_mode) {
  // 确定输出张量的数据类型与输入张量和值张量的公共数据类型
  auto commonDtype = at::result_type(self, value);
  // 如果公共数据类型是整数类型（包括布尔型），并且未指定舍入模式，则转换为默认的浮点数据类型
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true) && !rounding_mode.has_value()) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  // 创建一个空的张量作为结果，使用公共数据类型，并且形状为空
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  // 调用 div_out_sparse_zerodim 函数，将结果存储在 result 中并返回
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), result);
}

// 定义一个原地操作函数，用于稀疏张量与标量或张量之间的除法操作，并考虑舍入模式
Tensor& div_sparse_(Tensor& self, const Tensor& value, std::optional<c10::string_view> rounding_mode) {
  // 调用 div_out_sparse_zerodim 函数，在 self 上进行原地操作，并返回 self 本身
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), self);
}

// --------------------------------------------------------------------
// floor_divide(SparseTensor, Scalar)
// --------------------------------------------------------------------

// 对稀疏张量与标量或零维张量进行 floor_divide 操作，结果存储在稀疏张量 result 中
SparseTensor& floor_divide_out_sparse_zerodim(const SparseTensor& dividend,
  const Tensor& divisor,
  SparseTensor& result) {
  // 检查除数的维度是否为零，确保它是标量或零维密集张量
  TORCH_CHECK(divisor.dim() == 0, "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", divisor.sizes(), " for divisor)");
  // 检查除数是否为稀疏张量，确保它是标量或零维密集张量
  TORCH_CHECK(!divisor.is_sparse(), "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  // 断言结果张量为稀疏张量
  AT_ASSERT(result.is_sparse());
  // 断言被除数为稀疏张量
  AT_ASSERT(dividend.is_sparse());

  // 情况1：结果和被除数是同一个张量
  // 执行原地 floor_divide 操作
  if (is_same_tensor(result, dividend)) {

    // 参见注释 "Sparse Floor Division"
    // 如果结果张量未合并，则进行合并操作
    if (!result.is_coalesced()) {
      coalesce_(result);
    }

    // 在结果的值张量上执行 floor_divide_ 操作
    result._values().floor_divide_(divisor);
    return result;
  }

  // 情况2：结果和被除数是不同的张量
  // 复制被除数到临时张量 dividend_tmp
  Tensor dividend_tmp = dividend;

  // 确保 dividend_tmp 是合并的（参见上述注释）
  if (!dividend.is_coalesced()) {
    dividend_tmp = dividend.coalesce();
  }

  // 调整结果的大小，并按照 dividend_tmp 的索引进行索引
  result.resize_as_(dividend_tmp);
  result._indices().resize_as_(dividend_tmp._indices());
  result._indices().copy_(dividend_tmp._indices());

  // 计算结果的值张量
  Tensor result_values = result._values();
  at::floor_divide_out(result_values, dividend_tmp._values(), divisor);
  // 设置稀疏张量的非零元素数量，并缩小到 dividend_tmp 的非零元素数量
  get_sparse_impl(result)->set_nnz_and_narrow(dividend_tmp._nnz());
  // 设置结果的合并状态与 dividend_tmp 的合并状态相同
  result._coalesced_(dividend_tmp.is_coalesced());
  return result;
}
// 计算稀疏张量与稀疏张量之间的元素级除法，返回稀疏张量结果
Tensor floor_divide_sparse(const Tensor& self, const Tensor& value) {
  // 确定输入张量的公共数据类型
  auto commonDtype = at::result_type(self, value);
  // 创建一个空张量作为结果，与输入张量使用相同的数据类型
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  // 调用函数将稀疏张量除以另一张量，结果存储在预先分配的结果张量中
  return floor_divide_out_sparse_zerodim(self, value, result);
}

// 在原地对稀疏张量进行元素级除法运算，将结果保存在自身
Tensor& floor_divide_sparse_(Tensor& self, const Tensor& value) {
  // 调用函数实现在原地对稀疏张量进行元素级除法运算
  return floor_divide_out_sparse_zerodim(self, value, self);
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

// 仅支持浮点型输入张量
Tensor norm_sparse(const SparseTensor& self, const Scalar& p) {
  // 断言输入张量为稀疏张量
  AT_ASSERT(self.is_sparse());
  // 调用具体的 norm_sparse 函数，使用默认参数进行计算
  return norm_sparse(self, p, IntArrayRef{}, false, c10::nullopt);
}

// 计算稀疏张量的范数，支持指定维度和数据类型
Tensor norm_sparse(const SparseTensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  // 断言输入张量为稀疏张量
  AT_ASSERT(self.is_sparse());
  // 如果指定了维度，检查是否支持全维度的约简操作
  if (!dim.empty()) {
    int64_t ndim = self.dim();
    bool passed_full_reduction_check = static_cast<size_t>(ndim) == dim.size();
    if (passed_full_reduction_check) {
      auto dim_ = dim.vec();
      maybe_wrap_dims(dim_, ndim);
      std::vector<bool> dims_check(ndim, false);
      // 检查维度是否有重复，如果有则不通过
      for (auto dim_ind : dim_) {
        if (dims_check[dim_ind]) {
          passed_full_reduction_check = false;
          break;
        }
        dims_check[dim_ind] = true;
      }
    }
    // 断言通过全维度检查，即维度不能有重复且必须包含所有输入张量的维度
    TORCH_CHECK(passed_full_reduction_check,
      "norm_sparse currently only supports full reductions, so 'dim' must either be empty or contain all dimensions of the input");
  }
  // 断言 keepdim 参数为 false，因为当前不支持保持维度信息
  TORCH_CHECK(keepdim == false, "norm_sparse currently does not support keepdim=True");
  // 断言不支持 dtype 参数
  TORCH_CHECK(!dtype.has_value(), "norm_sparse currently does not support 'dtype' argument");
  constexpr auto TWO = 2.0;
  auto p_ = p.value_or(TWO);
  // 对稀疏张量进行共凝操作后，计算其 p 范数
  return self.coalesce()._values().norm(p_);
}

// --------------------------------------------------------------------
// mv(SparseTensor, Tensor)
// --------------------------------------------------------------------

// 计算稀疏张量与一维张量的矩阵-向量乘积
Tensor mv_sparse(const SparseTensor& self, const Tensor& vec)
{
  // 断言输入张量维度正确
  TORCH_CHECK(self.ndimension() == 2 &&
              vec.ndimension() == 1,
              "mv: two tensor dim should be 2 and 1, but got ",
              "SparseTensor Dim: ", self.ndimension(), "Tensor Dim: ", vec.ndimension());

  // 断言向量与稀疏张量的最后一个维度大小匹配
  TORCH_CHECK(vec.size(-1) == self.size(-1),
              "mv: expected self.size(-1) == vec.size(-1)");

  // 计算稀疏张量与向量的乘积，并去除额外的维度
  auto result = self.matmul(vec.unsqueeze(-1));

  return result.squeeze(-1);
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------
Tensor add_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 检查稀疏性和稠密性的组合，如果第一个参数是稀疏且第二个参数不是，则抛出异常
  TORCH_CHECK(!(self.is_sparse() && !other.is_sparse()),
              "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  // 确定结果张量的数据类型与输入张量的公共数据类型
  auto commonDtype = at::result_type(self, other);
  // 检查 alpha 是否与公共数据类型兼容
  alpha_check(commonDtype, alpha);
  // 创建一个空的张量作为结果，数据类型与 self 相同
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  // 调用底层的 add_out 函数进行张量加法操作，并返回结果张量
  return at::add_out(result, self, other, alpha);  // redispatch!
}

Tensor& add_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 调用底层的 add_out 函数进行 in-place 张量加法操作，并返回结果张量的引用
  return at::add_out(self, self, other, alpha);  // redispatch!
}

// 这些实现与稀疏张量无关

Tensor sub_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 检查张量的尺寸和数据类型是否匹配
  sub_check(self, other);
  // 调用 native 命名空间下的 add_sparse 函数实现张量的减法操作
  return native::add_sparse(self, other, -alpha);
}

Tensor& sub_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  // 检查张量的尺寸和数据类型是否匹配
  sub_check(self, other);
  // 调用 native 命名空间下的 add_sparse_ 函数实现 in-place 张量的减法操作
  return native::add_sparse_(self, other, -alpha);
}

Tensor& sub_out_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& r) {
  // 检查张量的尺寸和数据类型是否匹配
  sub_check(self, other);
  // 调用底层的 add_out 函数进行张量减法操作，并返回结果张量的引用
  return at::add_out(r, self, other, -alpha);  // redispatch!
}

static SparseTensor& add_out_sparse_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    // 保存 t 和 src 的非零元素数量，以及它们是否是合并的
    int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
    bool coalesced = t.is_coalesced() && src.is_coalesced();
    int64_t sparse_dim = src.sparse_dim();

    // 创建一个与 src 稀疏张量相同大小的索引张量 r_indices
    Tensor r_indices = at::empty({src.sparse_dim(), max_nnz}, t._indices().options());

    // 将 t 和 src 的值张量转换为公共数据类型 commonDtype
    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    // 创建一个新的值张量 r_values，大小与 s_values 相同，并初始化为零
    Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();

    // 获取 r_values 的步长和初始化索引变量
    int64_t blockSize = r_values.stride(0);
    int64_t r_i = 0, t_i = 0, s_i = 0;
    auto t_indices = t._indices();
    auto src_indices = src._indices();

    // 根据 nnz 的测试依赖性，使用张量访问器访问 t_indices、r_indices 和 src_indices
    auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
    auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
    auto src_indices_accessor = src_indices.accessor<int64_t, 2>();
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16,
        commonDtype, "cadd_sparse", [&] {
          // 获取 t_values 的数据指针，并将其强制转换为对应的 scalar_t 类型指针
          scalar_t* t_values_ptr = t_values.data_ptr<scalar_t>();
          // 获取 s_values 的数据指针，并将其强制转换为对应的 scalar_t 类型指针
          scalar_t* s_values_ptr = s_values.data_ptr<scalar_t>();
          // 获取 r_values 的数据指针，并将其强制转换为对应的 scalar_t 类型指针
          scalar_t* r_values_ptr = r_values.data_ptr<scalar_t>();
          // 将 value 强制转换为 scalar_t 类型，并赋给 cast_value
          scalar_t cast_value = value.to<scalar_t>();
          // 当 t_i 小于 t_nnz 或者 s_i 小于 s_nnz 时循环执行以下代码块
          while (t_i < t_nnz || s_i < s_nnz) {
            // 用于比较 t_indices_accessor[d][t_i] 和 src_indices_accessor[d][s_i] 的大小关系
            int64_t cmp;
            // 如果 t_i 大于等于 t_nnz，则设 cmp 为 -1
            if (t_i >= t_nnz) {
              cmp = -1;
            // 如果 s_i 大于等于 s_nnz，则设 cmp 为 1
            } else if (s_i >= s_nnz) {
              cmp = 1;
            // 否则依次比较 sparse_dim 维度上的索引值
            } else {
              cmp = 0;
              for (auto d: c10::irange(sparse_dim)) {
                if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                  cmp = 1;
                  break;
                }
                if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                  cmp = -1;
                  break;
                }
              }
            }
            // 如果 cmp 大于等于 0，则将 t_indices_accessor[d][t_i] 复制到 r_indices_accessor[d][r_i] 中，并将对应的 t_values 添加到 r_values
            if (cmp >= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
              }
              // 只有当 t_values 不是空张量时，才将 t_values 的元素添加到 r_values 中
              if (t_values.numel() > 0) {
                at::native::cpublas::axpy<scalar_t>(blockSize, 1,
                  t_values_ptr + t_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              t_i++;
            }
            // 如果 cmp 小于等于 0，则将 src_indices_accessor[d][s_i] 复制到 r_indices_accessor[d][r_i] 中，并将对应的 s_values 添加到 r_values
            if (cmp <= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
              }
              // 只有当 s_values 不是空张量时，才将 s_values 的元素添加到 r_values 中
              if (s_values.numel() > 0) {
                at::native::cpublas::axpy<scalar_t>(blockSize, cast_value,
                  s_values_ptr + s_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              s_i++;
            }
            // 将 r_i 自增，准备处理下一个位置的 r_indices 和 r_values
            r_i++;
          }
        }
    );

    // 如果 r 的数据类型不等于 commonDtype，则将 r_values 转换为 commonDtype 的数据类型
    if (r.scalar_type() != commonDtype) {
      r_values = r_values.to(r.scalar_type());
    }
    // 设置稀疏张量 r 的索引和值，采用不安全的方式设置
    get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
    // 设置稀疏张量 r 的非零元素数量，并进行窄化操作
    get_sparse_impl(r)->set_nnz_and_narrow(r_i);

    // TODO: 我认为可以在循环内部跟踪，并检测何时未协同（例如，通过观察索引是否倒退），这可能比在此处使用协同标志更精确。但这样更简单。
    // 将 _coalesced_ 方法应用于 r，使用给定的 coalesced 参数，并返回结果
    return r._coalesced_(coalesced);
// 添加稀疏张量到稀疏张量的原位操作，非连续情况下的实现
static SparseTensor& add_out_sparse_non_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    // 将稀疏张量 t 和 src 的值转换为指定的公共数据类型
    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    // 如果 t 或 src 包含非连续的值，则无法使用 at::native::cpublas::axpy，需拼接索引和值张量
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      commonDtype, "add_out_sparse_cpu", [&] {
          // 若 value 不等于 1，则对 s_values 进行标量乘法
          if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
            s_values = s_values.mul(value);
          }
        });

    // 拼接 t 和 src 的索引张量
    Tensor r_indices = at::cat({t._indices(), src._indices()}, 1);
    // 拼接 t_values 和 s_values 的值张量，并转换为 r 的标量类型
    Tensor r_values = at::cat({t_values, s_values}, 0).to(r.scalar_type());
    // 将结果别名到稀疏张量 r 中
    alias_into_sparse(r, r_indices, r_values);

    // 防止 nnz（非零元素数量）无限增长
    // TODO: 改进启发式方法来决定何时进行 coalesce 或者不需要进行 coalesce
    if (r._nnz() > r.numel()) {
      // 对 r 进行 coalesce 操作，以减少稀疏张量的碎片化
      auto c = r.coalesce();
      // 将 coalesce 后的结果别名到稀疏张量 r 中
      alias_into_sparse(r, c._indices(), c._values());
    }

    // 返回修改后的稀疏张量 r
    return r;
}

// 添加稀疏张量到稀疏张量的 CPU 实现
SparseTensor& add_out_sparse_cpu(const SparseTensor& t, const SparseTensor& src, const Scalar& value, SparseTensor& r) {
  // 如果 t 不是稀疏张量，则调用 add_dense_sparse_cpu 处理
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cpu(r, t, src, value);
  }
  // TODO: 这个测试看起来有些奇怪
  // 检查 src 是否为稀疏张量，否则报错提示
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  // 断言 t 不在 GPU 上，作为调度参数
  AT_ASSERT(!t.is_cuda());
  // 检查输出张量 r 不在 GPU 上
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  // 检查 src 不在 GPU 上
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  // 检查 t 和 src 的大小是否相等，否则报错提示
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

  // 提升 t 和 src 的标量类型为公共数据类型
  auto commonDtype = promoteTypes(t.scalar_type(), src.scalar_type());

  // 检查是否能将公共数据类型转换为输出张量 r 的标量类型，否则报错提示
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  // 如果 src 的非零元素数量为 0，则将稀疏张量 t 复制到稀疏张量 r
  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r, t);
  }
  // 如果 t 的非零元素数量为 0，则将稀疏张量 src 乘以标量值添加到稀疏张量 r
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r, src, value);
  }

  // 检查 t 和 src 的稀疏程度是否相同，否则报错提示
  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  // 将稀疏张量 r 的大小调整为与 src 相同
  r.resize_as_(src);
  // 如果 r 是元数据，则直接返回 r
  if (r.is_meta()) {
    return r;
  } else if (src._values().is_contiguous() && t._values().is_contiguous()) {
    // 如果 src 和 t 的值张量是连续的，则调用连续的稀疏张量添加函数
    return add_out_sparse_contiguous(r, t, src, value, commonDtype);
  } else {
    // 否则，调用非连续的稀疏张量添加函数
    return add_out_sparse_non_contiguous(r, t, src, value, commonDtype);
  }
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    以前称为 spcadd
// --------------------------------------------------------------------
// 为非混合模式下的稠密-稀疏加法实现函数（CPU端）
template <typename scalar_t>
void add_dense_sparse_worker_non_hybrid_cpu(Tensor& r, const Scalar& value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  // 获取索引和值的访问器
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  // 获取结果张量的指针和值的类型转换
  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  scalar_t cast_value = value.to<scalar_t>();

  // 获取稀疏张量的稀疏维度
  const int64_t sparse_dim = sparse.sparse_dim();

  // 计算结果张量的步长
  std::vector<int64_t> result_stride(sparse_dim);
  for (const auto d: c10::irange(sparse_dim)) {
    result_stride[d] = r.stride(d);
  }

  // 并行处理稀疏张量的非零元素
  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (const auto k: c10::irange(start, end)) {
      int64_t index = r.storage_offset();
      for (auto d: c10::irange(sparse_dim)) {
        index += result_stride[d] * indices_accessor[d][k];
      }
      // 执行稠密-稀疏加法操作
      r_ptr[index] += cast_value * values_accessor[k];
    }
  });
}

// --------------------------------------------------------------------
// 为混合模式下的稠密-稀疏加法实现函数（CPU端）
template <typename scalar_t>
inline void add_dense_sparse_worker_hybrid_cpu(Tensor& r, const Scalar& value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  // 获取值张量的稠密维度元素数量并检查连续性
  int64_t values_dense_size = values.stride(0);
  TORCH_CHECK(values.is_contiguous());
  scalar_t* v_ptr = values.data_ptr<scalar_t>();

  // 获取结果张量的指针并检查非空
  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  TORCH_CHECK(r_ptr != nullptr);

  // 获取索引张量的访问器和值的类型转换
  auto indices_accessor = indices.accessor<int64_t, 2>();
  scalar_t cast_value = value.to<scalar_t>();

  // 获取稀疏张量的稀疏维度和结果张量的步长
  auto sparse_dim = sparse.sparse_dim();
  std::vector<int64_t> result_stride(sparse_dim);
  for (auto d : c10::irange(sparse_dim)) {
    result_stride[d] = r.stride(d);
  }

  // 并行处理稀疏张量的非零元素
  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (auto k: c10::irange(start, end)) {
      auto r_index = r_ptr;
      for (auto d: c10::irange(sparse_dim)) {
        r_index += result_stride[d] * indices_accessor[d][k];
      }
      auto v_index = v_ptr + k * values_dense_size;
      // 调用cpublas库中的axpy函数执行稠密-稀疏加法操作
      at::native::cpublas::axpy<scalar_t>(values_dense_size, cast_value, v_index, 1, r_index, 1);
    }
  });
}

// --------------------------------------------------------------------
// 为非连续存储模式下的稠密-稀疏加法实现函数（CPU端）
template <typename scalar_t>
inline void add_dense_sparse_worker_non_coalesced_cpu(Tensor& r, const Scalar& value,
    const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  // 获取值张量的稠密维度元素数量并检查连续性
  auto values_dense_size = values.stride(0);
  TORCH_CHECK(values.is_contiguous());
  scalar_t* v_ptr = values.data_ptr<scalar_t>();
  TORCH_CHECK(v_ptr != nullptr);

  // 获取结果张量的指针并检查非空
  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  TORCH_CHECK(r_ptr != nullptr);

  // 获取值的类型转换和稀疏张量的稀疏维度
  scalar_t cast_value = value.to<scalar_t>();
  auto sparse_dim = sparse.sparse_dim();

  // 获取索引张量的访问器和结果张量的长度
  auto indices_accessor = indices.accessor<int64_t, 2>();
  int64_t result_length = r.size(0);

  // 计算结果张量的步长
  std::vector<int64_t> result_stride(sparse_dim);
  for (auto d : c10::irange(sparse_dim)) {
    // 将结果的步长设为 r 对应维度的步长
    result_stride[d] = r.stride(d);
  }

  // 获取稀疏张量 sparse 的非零元素个数
  auto sparse_nnz = sparse._nnz();
  // 获取当前系统最大的线程数
  int max_threads = at::get_num_threads();
  // 将最大线程数限制在结果长度和系统最大线程数中的较小值
  max_threads = (result_length < max_threads) ? result_length : max_threads;
  // 计算每个线程处理的平均 chunk 大小
  int64_t avg_chunk_down = result_length / max_threads;
  // 创建存储每个线程 chunk 大小的 vector
  std::vector<int64_t> chuck_size(max_threads);
  // 填充每个线程 chunk 大小为平均值
  for (const auto i : c10::irange(max_threads)) {
    chuck_size[i] = avg_chunk_down;
  }
  // 使 chunk 在线程之间平衡，将余下的部分分配给前几个线程
  for (auto i = 0 ; i < result_length % max_threads ; i++) {
    chuck_size[i] += 1;
  }
  // 创建存储累计 chunk 大小的 vector
  std::vector<int64_t> chuck_sum_size(max_threads + 1);
  // 初始化第一个元素为 0
  chuck_sum_size[0] = 0;
  // 计算累计 chunk 大小
  for (const auto i : c10::irange(1, max_threads)) {
    chuck_sum_size[i] = chuck_sum_size[i - 1] + chuck_size[i - 1];
  }
  // 最后一个元素为结果长度
  chuck_sum_size[max_threads] = result_length;
  // 使用并行化方法遍历每个线程处理的 chunk
  at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
    // 遍历每个 chunk 区间
    for (auto k: c10::irange(start, end)) {
      // 计算当前 chunk 的起始和结束位置
      int64_t chunk_begin = chuck_sum_size[k];
      int64_t chunk_end = chuck_sum_size[k + 1];
      // 遍历稀疏张量的非零元素
      for (const auto n: c10::irange(sparse_nnz)) {
        // 获取当前非零元素在稀疏张量中的偏移
        int64_t chunk_offset = indices_accessor[0][n];
        // 如果偏移在当前 chunk 区间内
        if (chunk_offset >= chunk_begin && chunk_offset < chunk_end) {
          // 计算结果张量中的偏移
          int64_t r_offset = result_stride[0] * chunk_offset;
          // 根据每个维度的步长计算偏移量
          for (const auto d : c10::irange(1, sparse_dim)) {
            r_offset += result_stride[d] * indices_accessor[d][n];
          }
          // 计算对应的值索引和结果索引
          scalar_t* v_index = v_ptr + n * values_dense_size;
          auto r_index = r_ptr + r_offset;
          // 调用 CPU BLAS 库中的 axpy 函数执行向量运算
          at::native::cpublas::axpy<scalar_t>(values_dense_size, cast_value, v_index, 1, r_index, 1);
        }
      }
    }
  });
  // 检查输出张量 `r` 不是稀疏张量
  TORCH_CHECK(!r.is_sparse());

  // 检查输入张量 `dense` 不是稀疏张量
  TORCH_CHECK(!dense.is_sparse());

  // 检查输入稀疏张量 `sparse_` 是稀疏张量
  TORCH_CHECK(sparse_.is_sparse());

  // 检查输入张量 `dense` 不在 CUDA 上
  TORCH_CHECK(!dense.is_cuda());

  // 检查输出张量 `r` 不在 CUDA 上，如果不是则返回错误信息
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");

  // 检查输入稀疏张量 `sparse_` 不在 CUDA 上，如果不是则返回错误信息
  TORCH_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  // 检查 `dense` 和 `sparse_` 的尺寸相同，否则返回错误信息，显示两者的尺寸信息
  TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  // 根据 `dense` 和 `sparse_` 的数据类型推断出一个共同的数据类型
  auto commonDtype = promoteTypes(dense.scalar_type(), sparse_.scalar_type());

  // 检查是否可以将推断的共同数据类型转换为输出张量 `r` 的数据类型，否则返回错误信息
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  // 调整输出张量 `r` 的大小与 `dense` 相同
  r.resize_as_(dense);

  // 获取稀疏张量 `sparse_` 的非零元素个数
  auto sparse_nnz = sparse_._nnz();

  // 如果稀疏张量 `sparse_` 的非零元素个数为零，且 `r` 与 `dense` 不是同一张量，则将 `dense` 复制到 `r` 并返回
  if (sparse_nnz == 0) {
    if (!is_same_tensor(r, dense)) r.copy_(dense);
    return r;
  }

  // 获取 `dense` 的维度数和 `sparse_` 的稀疏维度数
  int64_t dense_dim = dense.dim();
  int64_t sparse_dim = sparse_.sparse_dim();

  // 将输出结果缓冲设置为 `r`，若 `r` 的数据类型不是共同数据类型，则将 `dense` 转换为共同数据类型
  Tensor resultBuffer = r;
  if (r.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(r, dense)) {
    resultBuffer.copy_(dense);
  }

  // 获取稀疏张量 `sparse_` 的值和索引
  Tensor values = sparse_._values();
  Tensor indices = sparse_._indices();

  // 检查稀疏张量 `sparse_` 是否已经紧凑，或者其非零元素个数为 1
  bool sparse_is_coalesced = (sparse_.is_coalesced() || sparse_nnz == 1);

  // 检查输出结果 `r` 和值张量 `values` 是否连续存储
  bool result_is_contiguous = ((r.storage().data() != nullptr) && resultBuffer.is_contiguous());
  bool value_is_contiguous = values.is_contiguous();

  // 判断结果是否连续存储
  bool is_contiguous = (result_is_contiguous && value_is_contiguous);

  // 如果结果连续存储且稀疏张量 `sparse_` 已紧凑，则根据维度调用相应的稀疏-密集张量相加函数
  if (is_contiguous && sparse_is_coalesced) {
    // TODO: 对非混合模式进行优化，可以不使用缓冲区
    if (sparse_dim == dense_dim) {
      // 根据数据类型调度稀疏-密集非混合 CPU 工作器
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
          commonDtype, "add_dense_sparse_non_hybrid", [&] {
            add_dense_sparse_worker_non_hybrid_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
          });
    } else {
      // 根据数据类型调度混合稀疏-密集 CPU 工作器
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
          commonDtype, "add_dense_sparse_hybrid", [&] {
            add_dense_sparse_worker_hybrid_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
          });
    }
  } else if (is_contiguous && (sparse_dim > 0)) {
    // 处理稀疏张量不是紧凑的情况
    // TODO: 为非混合模式进行处理

    // 留待以后优化
    // 如果稀疏张量 `sparse_` 的稀疏维度大于 0，则处理稀疏张量不是紧凑的情况
    // TODO: Handle sparse is not coalesced
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
        commonDtype, "add_dense_sparse_worker_non_coalesced", [&] {
          // 使用宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4处理多种数据类型，其中包括ComplexHalf、Bool、BFloat16、Half，调用lambda表达式
          add_dense_sparse_worker_non_coalesced_cpu<scalar_t>(resultBuffer, value, sparse_, indices, valuesBuffer);
        });
  } else {
    // 对于非连续值和输出的慢速处理路径
    // TODO: 可以进一步改进coalesce()的性能
    // 稀疏张量进行压缩操作，使其具有连续的索引和数值
    sparse = sparse_.coalesce();
    // 获取稀疏张量的索引
    indices = sparse._indices();
    // 获取稀疏张量的数值
    values = sparse._values();
    // 将稀疏张量的数值转换为指定的公共数据类型
    valuesBuffer = values.to(commonDtype);
    // 获取稀疏张量索引的访问器
    auto indices_accessor = indices.accessor<int64_t, 2>();
    // 获取稀疏张量的非零元素数量
    auto sparse_nnz = sparse._nnz();
    // 并行处理稀疏张量的非零元素
    at::parallel_for(0, sparse_nnz, 100, [&](int64_t start, int64_t end) {
      for (auto k: c10::irange(start, end)) {
        // 对于每个非零元素，根据索引访问相应的结果缓冲区
        Tensor dstBuffer = resultBuffer;
        for (auto d: c10::irange(sparse_dim)) {
          dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
        }
        // 获取对应的数值缓冲区
        Tensor srcBuffer = valuesBuffer.select(0, k);
        // 将数值缓冲区的值加到结果缓冲区中
        dstBuffer.add_(srcBuffer, value);
      }
    });
  }
  // 如果结果张量的标量类型与公共数据类型不匹配，则进行复制操作
  if (r.scalar_type() != commonDtype) {
    r.copy_(resultBuffer);
  }
  // 返回结果张量
  return r;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

// 定义一个函数，用于稀疏张量与张量之间的乘法操作
Tensor mul_sparse(const Tensor& self, const Tensor& other) {
  // 确定结果的数据类型与输入张量的公共数据类型相同
  auto commonDtype = at::result_type(self, other);

  // 如果self是稀疏张量，则使用self的选项创建结果张量；否则使用other的选项
  auto result_options = self.is_sparse() ?
    self.options().dtype(commonDtype) : other.options().dtype(commonDtype);

  // 创建一个空张量作为结果
  Tensor result = at::empty({0}, result_options);

  // 调用at::mul_out函数进行乘法操作，结果保存在result中
  return at::mul_out(result, self, other);  // redispatch!
}

// 对稀疏张量进行原地乘法操作
Tensor& mul_sparse_(Tensor& self, const Tensor& other) {
  if (self.is_sparse()) {
    // 如果self是稀疏张量，则调用at::mul_out进行乘法操作，结果覆盖self
    return at::mul_out(self, self, other);  // redispatch!
  }
  else {
    // 如果self不是稀疏张量，则首先计算self与other的乘法结果，然后清空self并加上结果
    const auto res = at::mul(self, other);
    self.zero_();
    self.add_(res);
    return self;
  }
}

// 用于实现点对点操作的通用函数，处理稠密张量与稀疏COO张量之间的索引交集
template <typename binary_func_t>
Tensor& intersection_binary_op_sparse_dense_out(
    const Tensor& d,
    const SparseTensor& s_,
    Tensor& res,
    const char* const op_name,
    const binary_func_t& op,
    const bool coalesce = false) {
  // 计算广播后的形状
  const auto res_shape = infer_size(d.sizes(), s_.sizes());

  // 如果s_或d为空，则直接返回空稀疏张量
  if (!s_._nnz() || !s_.numel() || !d.numel()) {
    const int64_t dense_dim = s_.dense_dim();
    const int64_t sparse_dim = static_cast<int64_t>(res_shape.size()) - dense_dim;
    const int64_t nnz = 0;
    const auto indices = at::empty({sparse_dim, nnz}, s_._indices().options());
    auto res_values_shape = s_._values().sizes().vec();
    res_values_shape[0] = nnz;
    const auto values = at::empty(res_values_shape, s_._values().options().dtype(res.scalar_type()));
    auto* res_impl = get_sparse_impl(res);
    res_impl->raw_resize_(sparse_dim, dense_dim, /*size=*/res_shape);
    res_impl->set_indices_and_values_unsafe(indices, values);
    res_impl->set_nnz_and_narrow(nnz);
    return res._coalesced_(true);
  }

  const auto d_dim = d.dim();
  const auto s_dim = s_.dim();

  // 当稀疏张量广播到稠密张量时，总是进行coalesce操作，以消除重复索引
  const auto s = (coalesce || d_dim > s_dim) ? s_.coalesce() : s_;

  const auto sparse_dim = s.sparse_dim();
  const auto dense_dim = s.dense_dim();

  const auto s_indices = s._indices();
  const auto s_values = s._values();

  // 定义应用操作的lambda函数
  const auto apply_op = [&](const Tensor& d_filtered) -> Tensor& {
    // 克隆稀疏张量的索引
    const auto res_indices = s_indices.clone();
    // 当 d 和 s 都是0维时，才执行 to(res.scalar_type)
    // 这确保了类型提升遵循以下规则：
    // op(0维, 0维).dtype == <通用数据类型>
    // op(0维, >=1维).dtype == <>=1维>.dtype，
    // 其中 >=1维 是至少具有1维的张量。
    // 如果操作是原地执行，则不进行类型转换。
    // 如果 s 是非联合的0维张量且 d 是0维时，需要进行类型转换。
    // 这是因为 s.values 至少是1维的，因此
    // op(s.values, d).dtype == s.values.dtype，但我们希望
    // op(s.values, d).dtype == <通用数据类型>。
    const auto values = op(d_filtered, s_values);
    // 如果 s_ 和 res 是相同的张量，则直接使用 values，否则将 values 转换为 res.scalar_type() 类型。
    const auto res_values = is_same_tensor(s_, res) ? values : values.to(res.scalar_type());
    // 获取 res 的稀疏表示实现
    auto* res_impl = get_sparse_impl(res);
    // 调整 res 的稀疏维度、密集维度和形状
    res_impl->raw_resize_(sparse_dim, dense_dim, res_shape);
    // 设置 res 的索引和数值（不安全操作）
    res_impl->set_indices_and_values_unsafe(res_indices, res_values);
    // 设置 res 的非零元素数量，并紧缩（如果适用）
    res_impl->set_nnz_and_narrow(s._nnz());
    // 返回调整后的 res，并确保它是联合的（如果 s 是联合的则 res 也应该是联合的）
    return res._coalesced_(s.is_coalesced());
  };

  // 最简单的情况：只有密集维度相交。
  // 这意味着只有值张量会相互作用。
  if (d_dim <= dense_dim) {
    return apply_op(d);
  }

  // 现在我们有稀疏维度和密集维度之间的交集。
  const auto sparse_dim_intersec = std::min(sparse_dim, d_dim - dense_dim);
  const auto d_start_dim_intersec = std::max<int64_t>(0, d_dim - s_dim);
  const auto s_start_dim_intersec = std::max<int64_t>(0, s_dim - d_dim);

  // 使用 s_indices 索引 d，找到与 s_values 交互的值
  const auto d_filtered = [&]() -> Tensor {
    using at::indexing::Slice;
    using at::indexing::Ellipsis;
    using at::indexing::TensorIndex;

    std::vector<TensorIndex> intersec_indices;
    intersec_indices.reserve(d_dim);

    if (d_start_dim_intersec) {
      intersec_indices.emplace_back(Ellipsis);
    }
    // 构建交集索引，将 s_indices[s_start_dim_intersec + i] 添加到 intersec_indices 中
    for (const auto i : c10::irange(sparse_dim_intersec)) {
      const auto s_idx = s_start_dim_intersec + i;
      intersec_indices.emplace_back(s_indices[s_idx]);
    }
    // 添加 Slice() 来扩展 d 的维度，避免越界索引
    for (auto i = d_start_dim_intersec + sparse_dim_intersec; i < d_dim; ++i) {
      intersec_indices.emplace_back(Slice());
    }
    // 在被索引的维度上扩展 d，避免越界索引，并返回索引后的张量
    const auto d_expanded_shape = std::vector<int64_t>(
        res_shape.end() - d_dim, res_shape.end());
    return d.expand(d_expanded_shape).index(intersec_indices);
  }();

  // 当维度匹配或者稀疏维度 "更大" 时，结果的非零元素数量是相同的，
  // 因此只有值会被修改。
  if (s_dim >= d_dim) {
    return apply_op(d_filtered);
  }

  // 否则非零元素数量会增加，索引和值都需要更新。
  const auto d_batch_shape = d.sizes().slice(0, d_start_dim_intersec);
  const auto d_batch_len = static_cast<int64_t>(d_batch_shape.size());
  // 获取批处理维度的计数和最大维度
  int64_t batch_count = 1;
  int64_t max_batch_dim = 0;
  // 使用 lambda 表达式计算批处理维度的计数和最大维度
  std::tie(batch_count, max_batch_dim) = [d_batch_shape]() -> std::tuple<int64_t, int64_t> {
    int64_t batch_count = 1;
    // 计算批处理维度的数量和最大维度
    // 返回结果作为 tuple
    // 初始化最大批次维度为0
    int64_t max_batch_dim = 0;
    // 遍历输入的批次形状，计算总批次数并更新最大批次维度
    for (const auto& b : d_batch_shape) {
      batch_count *= b;
      max_batch_dim = std::max(b, max_batch_dim);
    }
    // 返回总批次数和最大批次维度的元组
    return std::make_tuple(batch_count, max_batch_dim);
  }();

  // 计算结果稀疏维度
  const auto res_sparse_dim = static_cast<int64_t>(d_batch_shape.size()) + sparse_dim;
  // 结果稠密维度等于输入的稠密维度
  const auto res_dense_dim = dense_dim;
  // 计算稀疏张量的非零元素数量
  const auto s_nnz = s._nnz();
  // 计算结果张量的非零元素数量
  const auto res_nnz = batch_count * s_nnz;
  // 初始化结果值张量的形状
  auto res_values_shape = s_values.sizes().vec();
  res_values_shape[0] = res_nnz;
  // 计算结果值张量
  const auto res_values = op(d_filtered, s_values).reshape(res_values_shape);
  // 初始化结果索引张量
  const auto res_indices = [&]() -> Tensor {
    // 创建索引缓冲区
    const auto index_buffer = at::arange(max_batch_dim, s_indices.options());
    // 初始化索引张量
    auto indices = at::empty({res_sparse_dim, res_nnz}, s_indices.options());
    // 填充对应于 d 的“批次”维度的索引
    int64_t n_repeat_interleave = res_nnz;
    int64_t n_repeat = 1;
    for (const auto dim : c10::irange(d_batch_len)) {
      const auto dim_size = d_batch_shape[dim];
      n_repeat_interleave /= dim_size;
      // 填充对应于维度 dim 的“批次”维度的索引
      // 相当于 indices[dim].copy_(repeat_interleave(dim_index, n_repeat_interleave).repeat(n_repeat))
      const std::initializer_list<int64_t> dim_index_expanded_shape = {n_repeat, dim_size, n_repeat_interleave};
      const auto dim_index = index_buffer.slice(-1, 0, dim_size);
      const auto dim_index_expanded = dim_index.unsqueeze(0).unsqueeze_(-1).expand(dim_index_expanded_shape);
      // 注意：indices 是连续的，因此视图操作是安全的
      indices[dim].view(dim_index_expanded_shape).copy_(dim_index_expanded);
      n_repeat *= dim_size;
    }
    // 填充对应于 s_indices 的索引
    // 相当于 indices_sparse.copy(s_indices.repeat({1, n_repeat})
    n_repeat = res_nnz / s_nnz;
    auto indices_sparse = indices.narrow(0, d_batch_len, res_sparse_dim - d_batch_len);
    const std::initializer_list<int64_t> s_indices_expanded_shape = {-1, n_repeat, s_nnz};
    const auto s_indices_expanded = s_indices.unsqueeze(1).expand(s_indices_expanded_shape);
    indices_sparse.view(s_indices_expanded_shape).copy_(s_indices_expanded);

    return indices;
  }();
  // 获取结果稀疏张量的实现
  auto* res_impl = get_sparse_impl(res);
  // 调整结果稀疏张量的大小和维度
  res_impl->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
  // 设置结果稀疏张量的索引和值（不安全地）
  res_impl->set_indices_and_values_unsafe(res_indices, res_values);
  // 设置结果稀疏张量的非零元素数量和压缩
  res_impl->set_nnz_and_narrow(res_nnz);
  // 根据索引扩展设计和 s 是紧凑的，结果也是紧凑的
  // 返回设置后的结果张量，保证它是紧凑的
  return res._coalesced_(true);
}

// 定义一个函数 _mul_dense_sparse_out，接受两个稀疏张量 d 和 s，以及结果张量 res，并调用 intersection_binary_op_sparse_dense_out 函数进行稀疏-稠密张量相乘操作
Tensor& _mul_dense_sparse_out(const Tensor& d, const Tensor& s, Tensor& res) {
  // 返回 intersection_binary_op_sparse_dense_out 函数的结果，使用 "mul" 作为操作名称，并定义一个 lambda 函数，用于执行元素级乘法操作 at::mul(a, b)
  return intersection_binary_op_sparse_dense_out(d, s, res, "mul", [](const Tensor& a, const Tensor& b) -> Tensor {
      return at::mul(a, b);
  });
}

// 定义一个函数 _mul_sparse_sparse_zero_dim_out，接受一个零维稀疏张量 zero_dim、另一个稀疏张量 other，以及结果张量 r
Tensor& _mul_sparse_sparse_zero_dim_out(const Tensor& zero_dim, const Tensor& other, Tensor& r) {
  // 定义一个 lambda 函数 is_wrapped_scalar，用于检查是否为包装的标量
  const auto is_wrapped_scalar = [](const Tensor& s) -> bool {
    return !s.dim() && s.is_coalesced();
  };

  // 定义一个 lambda 函数 extract_vals_from_wrapped_scalar，从包装的标量中提取值
  const auto extract_vals_from_wrapped_scalar = [](const Tensor& s) -> Tensor {
    auto vals = s._values().squeeze(0);
    // 如果 squeeze 操作未消除维度，则返回一个空的零维张量，避免在 intersection_binary_op_sparse_dense_out 函数中的广播问题
    if (vals.dim()) {
      return at::empty({}, vals.options());
    }
    return vals;
  };

  // 代码分派到 _mul_dense_sparse_out 函数，当至少一个输入是零维且已合并时，可以延迟调用 coalesce 转换为稠密张量
  if (zero_dim.is_coalesced()) {
    const auto scalar_val = extract_vals_from_wrapped_scalar(zero_dim);
    return _mul_dense_sparse_out(scalar_val, other, r);
  }
  
  // 在这里，zero_dim 不是包装的标量，因此我们测试 other
  if (is_wrapped_scalar(other)) {
    const auto scalar_val = extract_vals_from_wrapped_scalar(other);
    return _mul_dense_sparse_out(scalar_val, zero_dim, r);
  }
  
  // 如果都不是包装的标量，但 zero_dim 至少是零维，则我们将其合并为一个标量
  const auto scalar_val = extract_vals_from_wrapped_scalar(zero_dim.coalesce());
  return _mul_dense_sparse_out(scalar_val, other, r);
}

// 定义一个函数 _mul_sparse_sparse_out，接受两个稀疏张量 x 和 y，以及结果张量 res，并调用 mul_sparse_sparse_out_stub 进行稀疏-稀疏张量相乘操作
Tensor& _mul_sparse_sparse_out(const Tensor& x, const Tensor& y, Tensor& res) {
  // 调用 mul_sparse_sparse_out_stub 进行稀疏-稀疏张量相乘操作
  mul_sparse_sparse_out_stub(res.device().type(), res, x, y);
  return res;
}

// 定义一个函数 mul_out_sparse_cpu，接受两个张量 t_ 和 src_，以及结果张量 r，用于处理在 CPU 上进行稀疏张量乘法操作
SparseTensor& mul_out_sparse_cpu(const Tensor& t_, const Tensor& src_, Tensor& r) {
  // 断言 t_ 不在 CUDA 上，是一个分发参数
  AT_ASSERT(!t_.is_cuda());
  // 检查 r 不在 CUDA 上，确保 'out' 是 CPU 张量
  TORCH_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
  // 检查 src_ 不在 CUDA 上，确保 'other' 是 CPU 张量
  TORCH_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");
  
  // 如果 src_ 不是稀疏的，则调用 _mul_dense_sparse_out 进行稀疏-稠密张量乘法操作
  if (!src_.is_sparse()) {
    return _mul_dense_sparse_out(src_, t_, r);
  }
  
  // 如果 t_ 不是稀疏的，则调用 _mul_dense_sparse_out 进行稀疏-稠密张量乘法操作
  if (!t_.is_sparse()) {
    return _mul_dense_sparse_out(t_, src_, r);
  }
  
  // 如果 src_ 是零维，则调用 _mul_sparse_sparse_zero_dim_out 进行稀疏-稀疏张量乘法操作
  if (!src_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(src_, t_, r);
  }
  
  // 如果 t_ 是零维，则调用 _mul_sparse_sparse_zero_dim_out 进行稀疏-稀疏张量乘法操作
  if (!t_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(t_, src_, r);
  }
  
  // 如果 t_ 和 src_ 的大小不相等，则进行稀疏-稀疏张量乘法操作，仅在稠密维度上广播
  const auto is_equal_size_inputs = t_.sizes().equals(src_.sizes());
  
  // mul(sparse, sparse) 操作，输入在稠密维度上进行广播
  if (!is_equal_size_inputs) {
  // 调用函数 _mul_sparse_sparse_out，计算稀疏张量 t_ 和 src_ 的乘积，结果保存在 r 中
  _mul_sparse_sparse_out(t_, src_, r);
  // 返回计算结果 r
  return r;
}

// 检查输入的稀疏张量 t_ 和 src_ 是否具有相同的大小，如果不同则抛出错误信息
TORCH_CHECK(is_equal_size_inputs, "mul: expected 'self' and 'other' to have same sizes when both are sparse"
    ", but ", t_.sizes(), " != ", src_.sizes());

// 当 t_ 或 src_ 的非零元素数量为零时进行短路处理
// 这不是严格必要的，但有些测试检查 mul 函数在处理来自 .data/.detach 的张量时是否会失败。
if (!t_._nnz() || !src_._nnz()) {
  // 将结果张量 r 调整为和 t_ 相同的大小，并置零
  r.resize_as_(t_);
  return r.zero_();
}

// 当 t_ 或 src_ 未压缩时，_mul_sparse_sparse_out 函数更快
if (!t_.is_coalesced() || !src_.is_coalesced()) {
  // 调用函数 _mul_sparse_sparse_out，计算稀疏张量 t_ 和 src_ 的乘积，结果保存在 r 中
  _mul_sparse_sparse_out(t_, src_, r);
  // 返回计算结果 r
  return r;
}

// 否则，_mul_sparse_sparse_out 可能比以下暴力解法慢
SparseTensor t = t_.coalesce();
SparseTensor src = src_.coalesce();

// 保存因为可能在原地操作时被覆写的变量
int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
// 计算最大非零元素数量，为两个输入稀疏张量的最小值
int64_t max_nnz = std::min(t_nnz, s_nnz);
// 获取稀疏维度
int64_t sparse_dim = src.sparse_dim();
// 创建结果张量的索引，大小为 [sparse_dim, max_nnz]，使用 t_indices 的选项
Tensor t_indices = t._indices();
Tensor src_indices = src._indices();
Tensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());

// 初始化索引和计数器
int64_t r_i = 0, t_i = 0, s_i = 0;

// 获取公共数据类型，用于计算
auto commonDtype = promoteTypes(t_.scalar_type(), src_.scalar_type());
// 检查结果类型是否可以转换为输出张量 r 的类型，否则抛出错误
TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in mul operation");

// 将稀疏张量的值转换为公共数据类型
Tensor t_values = t._values().to(commonDtype);
Tensor s_values = src._values().to(commonDtype);

// 创建一个大小与 t_values 相同的全零张量 r_buffer
Tensor r_buffer = new_values_with_size_of(t_values, max_nnz).zero_();

// 注意事项：依赖于上面的 nnz 测试
// 获取张量的访问器，用于高效访问张量的元素
auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

// 检查是否可以找到匹配的索引，如果可以则写入结果索引向量，并返回 true
auto index_preamble = [&]() {
  for (auto d: c10::irange(sparse_dim)) {
    if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
      t_i++;
      return false;
    }
    if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
      s_i++;
      return false;
    }
  }
  for (auto d: c10::irange(sparse_dim)) {
    r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
  }
  return true;
};

// 如果 t_values 的维度大于 1，则执行以下循环
while (t_i < t_nnz && s_i < s_nnz) {
  if (!index_preamble()) continue;
  // 使用 addcmul_ 函数计算乘积并加到 r_buffer 的对应位置
  r_buffer.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
  r_i++;
  t_i++;
  s_i++;
} else {
    // 使用宏展开并遍历多个数据类型，包括复数半精度、BFloat16和半精度
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::ComplexHalf, at::ScalarType::BFloat16, at::ScalarType::Half,
        commonDtype, "mul_out_sparse", [&] {
          // 获取结果张量的访问器
          auto r_accessor = r_buffer.accessor<scalar_t, 1>();
          // 获取稀疏张量 t_values 的访问器
          auto t_accessor = t_values.accessor<scalar_t, 1>();
          // 获取稀疏张量 s_values 的访问器
          auto s_accessor = s_values.accessor<scalar_t, 1>();

          // 在 t_values 和 s_values 的非零元素上进行循环计算
          while (t_i < t_nnz && s_i < s_nnz) {
            // 如果索引前言不满足，继续下一次循环
            if (!index_preamble()) continue;
            // 计算乘法并存储到结果张量 r_accessor 中
            r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
            r_i++;
            t_i++;
            s_i++;
          }
        }
    );
  }

  // 调整结果张量 r 的大小以匹配源张量 src
  r.resize_as_(src);
  // 将 r_buffer 转换为与 r 相同的数据类型的张量 r_values
  Tensor r_values = r_buffer.to(r.scalar_type());
  // 设置稀疏张量 r 的索引和值（不安全操作）
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
  // 设置稀疏张量 r 的非零元素数量并进行狭窄化处理
  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  // 返回经过合并的 r 张量
  return r._coalesced_(true);
}

// --------------------------------------------------------------------
// addmm(D1, S, D2, beta, alpha) -> D  [broadcasts]
//
// D = beta * D1 + alpha * mm(S, D2)
// --------------------------------------------------------------------

template <typename scalar_t>
void s_addmm_out_sparse_dense_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& indices, const Tensor& values, const Tensor& dense) {

  // r_ = alpha * sparse * dense
  // 将 alpha 和 beta 转换为对应的 scalar_t 类型
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();

  // 如果 beta == 0，则将 r 清零
  if (cast_beta == static_cast<scalar_t>(0)) {
    r.zero_();
  }
  // 如果 beta == 1 且 r 不是 t，则将 t 复制到 r
  else if (cast_beta == static_cast<scalar_t>(1)) {
    if (!is_same_tensor(r, t)) {
      r.copy_(t);
    }
  }
  // 否则，计算 r = beta * r + alpha * mm(S, D2)
  else {
    at::mul_out(r, t, scalar_to_tensor(beta));
  }

  // 获取 indices 和 values 的访问器
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  // 获取 dense 和 r 的指针
  scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
  scalar_t* r_ptr = r.data_ptr<scalar_t>();

  // 获取 dense 和 r 的步长信息
  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t r_stride0 = r.stride(0);
  int64_t r_stride1 = r.stride(1);

  // 遍历稀疏矩阵的非零元素
  for (auto i: c10::irange(nnz)) {
    scalar_t val = values_accessor[i];  // 获取当前非零元素的值
    int64_t row = indices_accessor[0][i];  // 获取当前非零元素所在的行索引
    int64_t col = indices_accessor[1][i];  // 获取当前非零元素所在的列索引

    // 检查索引是否有效，执行稀疏矩阵和稠密矩阵的乘法
    if (col >= 0 && col < dim_j && row >= 0 && row < dim_i) {
      // 如果 dim_k == 0，则跳过当前乘法操作
      if (dim_k == 0) {
        continue;
      }
      // 调用 AXPY 函数执行乘法操作
      at::native::cpublas::axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + row * r_stride0, r_stride1);
    } else {
      // 如果索引超出范围，抛出对应的错误信息
      if (col < 0 || col >= dim_j) {
        AT_ERROR("addmm: index out of column bound: ", col, " not between 1 and ", dim_j);
      } else {
        AT_ERROR("addmm: index out of row bound: ", row, " not between 1 and ", dim_i);
      }
    }
  }
};

static Tensor& s_addmm_out_sparse_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Scalar& beta,
    // 检查输入张量 `t` 是否在 CPU 上，并输出相应错误消息，指出预期的设备位置与实际张量的设备位置不匹配
    // `addmm` 函数期望 `t` 是一个 CPU 张量，但是它在 `t.device()` 上有一个张量
    TORCH_CHECK(
        t.is_cpu(),
        "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got tensor on ",
        t.device());
    
    // 检查输出张量 `r` 是否在 CPU 上，并输出相应错误消息，指出预期的设备位置与实际张量的设备位置不匹配
    // `addmm` 函数期望输出 `r` 是一个 CPU 张量，但是它在 `r.device()` 上有一个张量
    TORCH_CHECK(
        r.is_cpu(),
        "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got tensor on ",
        r.device());
    
    // 检查稀疏输入张量 `sparse_` 是否在 CPU 上，并输出相应错误消息，指出预期的设备位置与实际张量的设备位置不匹配
    // `addmm` 函数期望稀疏输入 `sparse_` 是一个 CPU 张量，但是它在 `sparse_.device()` 上有一个张量
    TORCH_CHECK(
        sparse_.is_cpu(),
        "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got tensor on ",
        sparse_.device());
    
    // 检查稠密输入张量 `dense` 是否在 CPU 上，并输出相应错误消息，指出预期的设备位置与实际张量的设备位置不匹配
    // `addmm` 函数期望稠密输入 `dense` 是一个 CPU 张量，但是它在 `dense.device()` 上有一个张量
    TORCH_CHECK(
        dense.is_cpu(),
        "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got tensor on ",
        dense.device());
    
    // 检查输出张量 `r` 的布局是否为 strided，并输出相应错误消息，指出预期的布局与实际张量的布局不匹配
    // `addmm_sparse_dense` 函数期望输出张量 `r` 是 strided 布局的，但是它具有布局 `r.layout()`
    TORCH_CHECK(
        r.layout() == kStrided,
        "addmm_sparse_dense: expected strided result tensor, got tensor with layout ",
        r.layout());
    
    // 检查输入张量 `t` 的布局是否为 strided，并输出相应错误消息，指出预期的布局与实际张量的布局不匹配
    // `addmm_sparse_dense` 函数期望输入张量 `t` 是 strided 布局的，但是它具有布局 `t.layout()`
    TORCH_CHECK(
        t.layout() == kStrided,
        "addmm_sparse_dense: expected 't' to have strided layout, got tensor with layout ",
        t.layout());
    
    // 检查稀疏输入张量 `sparse_` 的布局是否为 kSparse，并且稠密输入张量 `dense` 的布局是否为 kStrided
    // 输出相应错误消息，指出预期的布局与实际张量的布局不匹配
    // `addmm_sparse_dense` 函数期望稀疏输入 `sparse_` 是 kSparse 布局的，稠密输入 `dense` 是 kStrided 布局的
    TORCH_CHECK(
        sparse_.layout() == kSparse && dense.layout() == kStrided,
        "addmm_sparse_dense: expected either 'mat1' to have sparse layout and 'mat2' to have strided layout, got 'mat1' with layout ",
        sparse_.layout(),
        " and 'mat2' with layout ",
        dense.layout());
    
    // 检查稀疏输入张量 `sparse_` 的稀疏维度是否为 2，并输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望稀疏输入 `sparse_` 是一个二维张量，但是它有 `sparse_.sparse_dim()` 维
    TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
    
    // 检查稀疏输入张量 `sparse_` 的稠密维度是否为 0，并输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望稀疏输入 `sparse_` 是一个没有稠密维度的张量，但是它有 `sparse_.dense_dim()` 维
    TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
    
    // 检查稠密输入张量 `dense` 的维度是否为 2，并输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望稠密输入 `dense` 是一个二维张量，但是它有 `dense.dim()` 维
    TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");
    
    // 计算矩阵乘法的维度：ixj * jxk = ixk
    int64_t dim_i = sparse_.size(0);
    int64_t dim_j = sparse_.size(1);
    int64_t dim_k = dense.size(1);
    
    // 检查稠密输入张量 `dense` 的第 0 维大小是否等于 `dim_j`，输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望稠密输入 `dense` 的第 0 维大小是 `dim_j`，但是它的大小是 `dense.size(0)`
    TORCH_CHECK(dense.size(0) == dim_j,
        "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
    
    // 检查输入张量 `t` 的第 0 维大小是否等于 `dim_i`，输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望输入张量 `t` 的第 0 维大小是 `dim_i`，但是它的大小是 `t.size(0)`
    TORCH_CHECK(t.size(0) == dim_i,
        "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
    
    // 检查输入张量 `t` 的第 1 维大小是否等于 `dim_k`，输出相应错误消息，指出预期的维度与实际张量的维度不匹配
    // `addmm` 函数期望输入张量 `t` 的第 1 维大小是 `dim_k`，但是它的大小是 `t.size(1)`
    TORCH_CHECK(t.size(1) == dim_k,
        "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));
    
    // 调整输出张量 `r` 的大小为 {dim_i, dim_k}
    r.resize_({dim_i, dim_k});
    
    // 获取稀疏输入张量 `sparse_` 的非零元素数量 `nnz`
    int64_t nnz = sparse_._nnz();
    
    // 如果 `nnz` 为 0，则执行逐元素乘法 `at::mul_out` 操作，并返回结果张量 `r`
    if (nnz == 0) {
      at::mul_out(r, t, at::scalar_tensor(beta, r.options()));
      return r;
    }
    
    // 获取稀疏输入张量 `sparse_` 的索引张量 `indices` 和值张量 `values`
    Tensor indices = sparse_._indices();
    Tensor values = sparse_._values();
    
    // 根据值张量 `values` 的数据类型，分发执行稀疏矩阵乘法操作 `s_addmm_out_sparse_dense_worker`
    // 通过
}

// 定义函数：稀疏张量与稠密张量的乘法，将结果写入给定的输出张量
Tensor& addmm_out_sparse_dense_cpu(
    // 输入参数：自身张量
    const Tensor& self,
    // 输入参数：稀疏张量
    const SparseTensor& mat1,
    // 输入参数：稠密张量
    const Tensor& mat2,
    // 输入参数：beta 标量
    const Scalar& beta,
    // 输入参数：alpha 标量
    const Scalar& alpha,
    // 输出参数：结果张量
    Tensor& result) {
  // 将自身张量扩展到指定大小
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  // 调用底层的稀疏-稠密乘法函数，并将结果写入输出张量
  return s_addmm_out_sparse_dense_cpu(result, *b_self, mat1, mat2, beta, alpha);
}

// 定义静态函数：执行稀疏张量与稠密张量的乘法，并返回结果张量
static Tensor s_addmm_sparse_dense_cpu(
    // 输入参数：张量 t
    const Tensor& t,
    // 输入参数：稀疏张量 sparse
    const SparseTensor& sparse,
    // 输入参数：稠密张量 dense
    const Tensor& dense,
    // 输入参数：beta 标量
    const Scalar& beta,
    // 输入参数：alpha 标量
    const Scalar& alpha
) {
  // 创建一个空张量 r，使用与 t 相同的选项
  Tensor r = at::empty({0}, t.options());
  // 调用底层的稀疏-稠密乘法函数，并将结果写入 r
  s_addmm_out_sparse_dense_cpu(r, t, sparse, dense, beta, alpha);
  // 返回结果张量 r
  return r;
}

// 定义函数：执行稀疏张量与稠密张量的乘法，并返回结果张量
Tensor addmm_sparse_dense_cpu(
    // 输入参数：自身张量
    const Tensor& self,
    // 输入参数：稀疏张量
    const SparseTensor& mat1,
    // 输入参数：稠密张量
    const Tensor& mat2,
    // 输入参数：beta 标量
    const Scalar& beta,
    // 输入参数：alpha 标量
    const Scalar& alpha
) {
  // 将自身张量扩展到指定大小
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  // 调用静态函数 s_addmm_sparse_dense_cpu，并返回结果张量
  return s_addmm_sparse_dense_cpu(*b_self, mat1, mat2, beta, alpha);
}

// 定义函数：执行稀疏张量与稠密张量的乘法（就地操作）
Tensor& s_addmm_sparse_dense_cpu_(
    // 输入输出参数：张量 t
    Tensor& t,
    // 输入参数：稀疏张量 sparse
    const SparseTensor& sparse,
    // 输入参数：稠密张量 dense
    const Tensor& dense,
    // 输入参数：beta 标量
    const Scalar& beta,
    // 输入参数：alpha 标量
    const Scalar& alpha
) {
  // 调用底层的稀疏-稠密乘法函数，并将结果写入 t，进行就地操作
  return s_addmm_out_sparse_dense_cpu(t, t, sparse, dense, beta, alpha);
}

// 注意事项：addmm 的就地操作版本没有广播功能

// 定义函数：执行稀疏张量与稠密张量的乘法
Tensor _sparse_addmm(
  // 输入参数：张量 t
  const Tensor& t,
  // 输入参数：稀疏张量 sparse
  const SparseTensor& sparse,
  // 输入参数：稠密张量 dense
  const Tensor& dense,
  // 输入参数：beta 标量
  const Scalar& beta,
  // 输入参数：alpha 标量
  const Scalar& alpha
) {
  // _sparse_addmm 的前向操作与 addmm 等效；只有反向操作有所不同。
  // 这里的 redispatch 实际上是不必要的，但我懒得去掉它
  return at::addmm(t, sparse, dense, beta, alpha);
}

// 定义函数：执行稀疏张量与张量的乘法
Tensor _sparse_mm(
  // 输入参数：张量 mat1
  const Tensor& mat1,
  // 输入参数：张量 mat2
  const Tensor& mat2
) {
  // 如果 mat1 和 mat2 都是稀疏张量，则调用 _sparse_sparse_matmul 函数
  if (mat1.is_sparse() && mat2.is_sparse()) {
    return at::_sparse_sparse_matmul(mat1, mat2);
  }
  // 如果 mat1 是稀疏张量或 CSR 压缩稀疏张量，则创建一个全零张量 t
  if (mat1.is_sparse() || at::sparse_csr::is_sparse_compressed(mat1)) {
    Tensor t = at::zeros({mat1.size(-2), mat2.size(-1)}, mat2.options());
    // 调用 _sparse_addmm 函数，并返回结果
    return at::_sparse_addmm(t, mat1, mat2, 0, 1);
  }
  // 创建一个全零张量 t，与 mat1 和 mat2 的转置进行稀疏-稠密乘法，并返回结果的转置
  Tensor t = at::zeros({mat1.size(-2), mat2.size(-1)}, mat1.options());
  return at::_sparse_addmm(t.transpose(-2, -1), mat2.transpose(-2, -1), mat1.transpose(-2, -1), 0, 1).transpose(-2, -1);
}

// 注意事项：尽管名称暗示着它是 mm 的稀疏掩码版本，实际上它只是重新分派到 addmm_out 的操作
// 这不是 mm 的稀疏掩码版本的实现
SparseTensor& _sparse_mm_out(const SparseTensor& sparse,
  // 输入参数：张量 dense
  const Tensor& dense,
  // 输出参数：结果稀疏张量
  SparseTensor& result) {
  // 创建一个空张量 t，与 dense 具有相同选项
  Tensor t = at::zeros({}, dense.options());
  // 调用 addmm_out 函数，执行 redispatch 操作
  return at::addmm_out(result, t, sparse, dense, 0, 1);  // redispatch!
}

// 定义函数：执行稀疏张量与张量的乘法，可选降维方式 reduce
Tensor _sparse_mm(const Tensor& mat1, const Tensor& mat2, const c10::string_view reduce) {
  // 调用 _sparse_mm_reduce_impl 函数，并返回第一个返回值（结果张量）
  auto result = at::_sparse_mm_reduce_impl(mat1, mat2, reduce);
  return std::get<0>(result);
}

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

// 在给定的稀疏张量和稠密张量之间执行 HSPMM 操作，并将结果写入稀疏张量 `r` 中
SparseTensor& hspmm_out_sparse_cpu(const SparseTensor& sparse_, const Tensor& dense, SparseTensor& r) {
  // TODO: Make this a real argument
  // 定义缩放因子 alpha，默认为 1
  Scalar alpha = 1;

  // 确保稀疏张量 `sparse_` 不在 GPU 上
  AT_ASSERT(!sparse_.is_cuda()); // dispatch argument
  // 确保输出张量 `r` 在 CPU 上
  TORCH_CHECK(!r.is_cuda(), "hspmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  // 确保稠密张量 `dense` 在 CPU 上
  TORCH_CHECK(!dense.is_cuda(), "hspmm: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  // 检查稀疏张量 `sparse_` 的维度为 2
  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  // 检查稀疏张量 `sparse_` 的稠密维度为 0
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  // 检查稠密张量 `dense` 的维度为 2
  TORCH_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");

  // 获取稀疏张量 `sparse_` 的大小信息
  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  // 获取稠密张量 `dense` 的列数
  int64_t n = dense.size(1);

  // 检查稠密张量 `dense` 的行数与稀疏张量 `sparse_` 的列数是否匹配
  TORCH_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  // 调整输出稀疏张量 `r` 的大小为 (m, n)
  get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

  // 对稀疏张量 `sparse_` 进行合并，以便处理重复索引
  SparseTensor sparse = sparse_.coalesce();

  // 获取稀疏张量 `sparse` 的非零元素数量
  int64_t nnz = sparse._nnz();

  // 如果稀疏张量 `sparse` 中没有非零元素，则将输出张量 `r` 置零并返回
  if (nnz == 0) {
    r.zero_();
    return r;
  }

  // 创建用于存储稀疏矩阵的索引的张量
  Tensor indices = at::empty({1, nnz}, at::initialTensorOptions().dtype(kLong));

  // 初始化新的稀疏矩阵 `newSparse`，其索引与 `sparse` 相同，但值需要重新计算
  SparseTensor newSparse = sparse.clone();
  Tensor spIndices = newSparse._indices();
  Tensor valueIndices = spIndices.select(0, 0);

  // 计算输出索引
  auto valueIndices_accessor = valueIndices.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();

  int64_t i = -1, prevIdx = -1;
  for (const auto j : c10::irange(nnz)) {
    int64_t currIdx = valueIndices_accessor[j];
    if (currIdx != prevIdx) {
      indices_accessor[0][++i] = currIdx;
      prevIdx = currIdx;
    }
    valueIndices_accessor[j] = i;
  }
  int64_t outNnz = i + 1;
  indices.resize_({1, outNnz});
  // 根据稠密张量 `dense` 的选项创建用于存储值的张量
  Tensor values = at::empty({outNnz, n}, dense.options());

  // 调整 `newSparse` 的大小，确保其与更新后的索引和值匹配
  std::vector<int64_t> new_size = get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = outNnz;
  get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

  // 使用稀疏 * 稠密的乘法计算输出值张量 `values`
  s_addmm_out_sparse_dense_cpu(values, values, newSparse, dense, 0, alpha);
  // 将计算得到的索引和值设置为输出稀疏张量 `r` 的内容
  get_sparse_impl(r)->set_indices_and_values_unsafe(indices, values);

  // 返回输出稀疏张量 `r`
  return r;
}

// 创建并返回执行 HSPMM 操作的结果稀疏张量
SparseTensor hspmm_sparse_cpu(const SparseTensor& sparse, const Tensor& dense) {
  // 创建一个新的稀疏张量 `r`，形状与 `sparse` 相同
  SparseTensor r = at::empty({0}, sparse.options());
  // 调用 `hspmm_out_sparse_cpu` 函数，将结果写入稀疏张量 `r` 中
  hspmm_out_sparse_cpu(sparse, dense, r);
  // 返回输出稀疏张量 `r`
  return r;
}

// --------------------------------------------------------------------
// sspaddmm(S1, S2, D, beta, alpha) -> S
//
// S = beta * S1 + alpha * mm(S2, D)
// --------------------------------------------------------------------

// 在 CPU 上执行稀疏张量乘法和加法操作，并将结果存储在给定的稀疏张量 r 中
SparseTensor& _sspaddmm_out_cpu(
    // 输入稀疏张量 t，稀疏张量 sparse_ 和密集张量 dense
    const SparseTensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    // 加法的标量权重 beta 和乘法的标量权重 alpha
    const Scalar& beta,
    const Scalar& alpha,
    // 存储结果的稀疏张量 r
    SparseTensor& r) {

  // 断言输入张量 t 不在 CUDA 上（dispatch 参数）
  AT_ASSERT(!t.is_cuda());

  // 检查结果张量 r 不在 CUDA 上
  TORCH_CHECK(!r.is_cuda(), "sspaddmm: expected 'out' to be CPU tensor, but got CUDA tensor");

  // 检查输入稀疏张量 sparse_ 不在 CUDA 上
  TORCH_CHECK(!sparse_.is_cuda(), "sspaddmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");

  // 检查输入密集张量 dense 不在 CUDA 上
  TORCH_CHECK(!dense.is_cuda(), "sspaddmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  // 检查稀疏张量 sparse_ 的稀疏维度为 2
  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");

  // 检查稀疏张量 sparse_ 的密集维度为 0
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "sspaddmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");

  // 检查密集张量 dense 的维度为 2
  TORCH_CHECK(dense.dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  // 对输入的稀疏张量 sparse_ 进行合并操作，得到 sparse
  SparseTensor sparse = sparse_.coalesce();

  // 计算稀疏张量 sparse 的维度信息
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  // 调整结果张量 r 的大小，为后续操作准备空间，这一步在检查之前执行，因为 r 可能与 t 重叠
  // 参见 test_saddmm
  get_sparse_impl(r)->raw_resize_(2, 0, {dim_i, dim_k});

  // 检查密集张量 dense 的第 0 维是否与稀疏张量 sparse 的第 1 维大小相等
  TORCH_CHECK(dense.size(0) == dim_j,
      "sspaddmm: Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));

  // 检查稀疏张量 t 的第 0 维是否与稀疏张量 sparse 的第 0 维大小相等
  TORCH_CHECK(t.size(0) == dim_i,
      "sspaddmm: Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));

  // 检查稀疏张量 t 的第 1 维是否与密集张量 dense 的第 1 维大小相等
  TORCH_CHECK(t.size(1) == dim_k,
      "sspaddmm: Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

  // 计算稀疏张量 sparse 的非零元素数量
  int64_t nnz = sparse._nnz();

  // 由于后续操作需要使用 indices.data_ptr，在这里需要确保 indices 是连续的
  Tensor indices = sparse._indices().contiguous();
  Tensor values = sparse._values();

  // 将 COO 格式的 indices 转换为 CSR 格式
  Tensor csr = coo_to_csr(indices.data_ptr<int64_t>(), dim_i, nnz);

  // 计算稀疏张量 t 的非零元素数量
  int64_t t_nnz = t._nnz();

  // 计算结果张量 r 的预期非零元素数量
  int64_t r_nnz = nnz * dim_k + t_nnz;

  // 创建新的索引张量 newi 和值张量 newv，用于存储最终结果
  Tensor newi = at::empty({2, r_nnz}, kLong);
  Tensor newv = at::zeros(
      {r_nnz},
      optTypeMetaToScalarType(values.options().dtype_opt()),
      values.options().layout_opt(),
      values.options().device_opt(),
      values.options().pinned_memory_opt());

  // 如果稀疏张量 t 的非零元素数量不为 0，则将 t 的索引和值复制到 newi 和 newv 中的相应位置
  if (t_nnz != 0) {
    Tensor narrowi = newi.narrow(1, 0, t_nnz);
    Tensor narrowv = newv.narrow(0, 0, t_nnz);

    narrowi.copy_(t._indices());
    narrowv.copy_(t._values());
    newv.mul_(beta);
  }

  // sparse = sparse * dense
  // 初始化稀疏矩阵行数计数器
  int64_t p = t_nnz;

  // 获取稀疏张量的 CSR 格式访问器和索引张量的访问器
  auto csr_accessor = csr.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto newi_accessor = newi.accessor<int64_t, 2>();

  // 获取 dense 张量的步长信息
  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t newv_stride0 = newv.stride(0);

  // 根据 values 的数据类型进行分发，执行稀疏矩阵乘法操作
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      values.scalar_type(), "sspmm", [&] {
        auto values_accessor = values.accessor<scalar_t, 1>();
        scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
        scalar_t* newv_ptr = newv.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();

        // 遍历稀疏张量的维度 dim_i
        for (const auto h : c10::irange(dim_i)) {
          int64_t i_start = csr_accessor[h];
          int64_t i_end = csr_accessor[h+1];

          // 遍历每行中的非零元素
          for (const auto i : c10::irange(i_start, i_end)) {
            scalar_t val = values_accessor[i];
            int64_t col = indices_accessor[1][i];

            // 如果列索引在有效范围内，执行稀疏矩阵乘法操作
            if (col >= 0 && col < dim_j) {
              at::native::cpublas::axpy<scalar_t>(dim_k,
                  cast_alpha * val,
                  dense_ptr + col * dense_stride0, dense_stride1,
                  newv_ptr + p * newv_stride0, 1);
            } else {
              // 若列索引越界，抛出错误
              AT_ERROR("index out of bound. sspmm: ", col, " not between 1 and ", dim_j);
            }
          }

          // 填充新索引值到 newi 张量中
          if (i_start != i_end) {
            for (const auto i : c10::irange(dim_k)) {
              newi_accessor[0][p+i] = h;
              newi_accessor[1][p+i] = i;
            }
            p += dim_k;
          }
        }
      }
  );

  // 避免克隆操作，直接设置稀疏张量的索引和数值
  get_sparse_impl(r)->set_indices_and_values_unsafe(newi, newv);
  // 设置稀疏张量的非零元素数量
  get_sparse_impl(r)->set_nnz_and_narrow(p);

  // 返回结果稀疏张量
  return r;
}

// 这里是一个 C++ 的代码块，包含了几个函数定义，下面是对每个函数的注释解释：

// 当前函数在只有稀疏张量的情况下执行 sspaddmm 操作，因此对于非稀疏张量会抛出错误信息
Tensor& _sspaddmm_out_only_sparse(const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
}

// 对稀疏张量执行矩阵乘法操作，返回结果稀疏张量
Tensor smm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  // 使用 sspaddmm_out 函数进行稀疏张量与密集张量的乘法运算
  at::sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
  return result;
}

// 对稀疏张量执行 sspaddmm 运算，返回结果稀疏张量
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha) {
  auto result = at::empty({0}, self.options());
  // 使用 sspaddmm_out 函数进行稀疏张量之间及稀疏与密集张量之间的乘法运算
  at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

// --------------------------------------------------------------------
// sparse.sum()
//
// 该实现调用 coalesce() 函数在稀疏维度上进行求和操作。
// 理想情况下，未来应该有一个统一的求和函数，用于诸如求和、最大值和最小值等操作。
// --------------------------------------------------------------------
// 对稀疏张量进行求和操作，返回结果稠密张量
Tensor _sparse_sum(const SparseTensor& input) {
  return input.coalesce().values().sum();
}

// 对稀疏张量按指定数据类型进行求和操作，返回结果张量
Tensor _sparse_sum(const SparseTensor& input, ScalarType dtype) {
  // 不需要首先进行数据类型转换，只需正确设置累加器即可
  return input.coalesce().values().sum(dtype);
}

// 对稀疏张量按指定维度进行求和操作，返回结果张量
Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum, ScalarType dtype) {
  return at::_sparse_sum(input.to(dtype), dims_to_sum);
}

// 对稀疏张量按指定维度进行求和操作，返回结果张量
Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum) {
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  Tensor indices = input._indices();
  Tensor values = input._values();
  IntArrayRef sizes = input.sizes();
  const int64_t sparse_dim = input.sparse_dim();

  auto dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (const auto d : c10::irange(input_dim)) {
    if (dims_to_sum_b[d]) {
      if (d >= sparse_dim) dense_dims_to_sum_v.emplace_back(d + 1 - sparse_dim);
    }
    else {
      dims_to_keep_v.emplace_back(d);
    }
  }
  const int64_t sparse_dims_to_sum_size = dims_to_sum_v.size() - dense_dims_to_sum_v.size();
  const bool sum_all_sparse_dim = (sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (!dense_dims_to_sum_v.empty());

  // 计算新的值张量
  Tensor new_values;
  if (sum_dense_dim) {
    new_values = values.sum(dense_dims_to_sum_v);
  }
  else {
    new_values = values.clone(at::MemoryFormat::Contiguous);
  }

  if (sum_all_sparse_dim) {
    // 如果对所有稀疏维度求和，则返回一个稠密张量
    new_values = new_values.sum(0);
    return new_values;
  }
  else { // !sum_all_sparse_dim
    // 计算新的索引张量
    // 声明一个 Tensor 类型的变量 new_indices
    Tensor new_indices;
    // 如果 sparse_dims_to_sum_size 等于 0，则直接克隆 indices，并保证内存格式为 Contiguous
    if (sparse_dims_to_sum_size == 0) {
      new_indices = indices.clone(at::MemoryFormat::Contiguous);
    }
    // 否则，创建一个新的索引张量 new_indices，形状为 {sparse_dim - sparse_dims_to_sum_size, input._nnz()}
    else {
      new_indices = at::empty({sparse_dim - sparse_dims_to_sum_size, input._nnz()}, indices.options());
      // 遍历 dims_to_keep_v 中的每一个索引
      for (auto i: c10::irange(dims_to_keep_v.size())) {
        int64_t d = dims_to_keep_v[i];
        // 如果 d 小于 sparse_dim，则将 indices[d] 的内容复制到 new_indices[i] 中
        if (d < sparse_dim) new_indices[i].copy_(indices[d]);
        // 否则，跳出循环
        else break;
      }
    }

    // 计算新的稀疏维度 new_sparse_dim
    int64_t new_sparse_dim = new_indices.size(0);
    // 计算新的稠密维度 new_dense_dim，排除 nnz 维度
    int64_t new_dense_dim = new_values.dim() - 1;
    // 创建一个新的大小向量 new_sizes，并预留空间
    std::vector<int64_t> new_sizes;
    new_sizes.reserve(dims_to_keep_v.size());
    // 根据 dims_to_keep_v 中的每一个维度 d，将 sizes[d] 添加到 new_sizes 中
    for (auto d : dims_to_keep_v) new_sizes.emplace_back(sizes[d]);
    // 如果 sum_all_sparse_dim 为真，则在 new_sizes 的开头插入维度 1
    if (sum_all_sparse_dim) new_sizes.emplace(new_sizes.begin(), 1);

    // 初始化是否已经进行了 coalesce() 操作的标志为 false
    bool is_coalesced = false;
    // 使用 _sparse_coo_tensor_with_dims_and_tensors 创建一个新的稀疏张量 new_sparse，
    // 参数包括 new_sparse_dim、new_dense_dim、new_sizes、new_indices、new_values 等
    SparseTensor new_sparse = at::_sparse_coo_tensor_with_dims_and_tensors(new_sparse_dim, new_dense_dim, new_sizes, new_indices, new_values, input.options(), is_coalesced);
    // 对新的稀疏张量 new_sparse 进行 coalesce() 操作，以执行求和归约
    new_sparse = new_sparse.coalesce();
    // 返回新的稀疏张量 new_sparse
    return new_sparse;
// --------------------------------------------------------------------
// NOTE [ sparse.sum() backward ]
//
// When summing over sparse_dim during backward pass:
// This function scatters gradients from 'grad_' tensor to 'input_' tensor.
// The indices in 'grad_' and 'input_' tensors are aligned over sparse_dim
// (where input_.sparse_dim >= grad_.sparse_dim). This function compares each pair
// of indices between grad_ and input_. When a matching indices pair (input_i, grad_i) is found,
// it copies grad_.values[grad_i] to input_grad_.values[input_i].
//
// Example:
//
//  input_.sparse_dim = [5, 5]
//  input_.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                    [1, 4, 4, 0, 1, 3, 2, 4]]
//  input_.values =   [0, 1, 2, 3, 4, 5, 6, 7]
//  ...
//  sparse.sum(input_, [0])
//  _sparse_sum_backward_cpu(...)
//  ...
//  grad_.indices = [[0, 1, 2, 3]]
//  grad_.values =   [1, 2, 0, 4]
//
// # after indices matching
//         input_         grad_
//        [[0, 1],   ->  [1]
//         [0, 4],   ->  [ ]
//         [1, 4],   ->  [ ]
//         [2, 0],   ->  [0]
//         [2, 1],   ->  [1]
//         [3, 3],   ->  [3]
//         [4, 2],   ->  [2]
//         [4, 4]])  ->  [ ]
//
// input_grad_.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                       [1, 4, 4, 0, 1, 3, 2, 4]]
// input_grad_.values =   [2, 0, 0, 1, 2, 4, 0, 0]
//
// Note: In the forward pass, input_ may be uncoalesced. Therefore, we must coalesce input_grad_
// in the backward pass. If input_ is not coalesced, coalescing input_grad_ may incorrectly sum
// grad values for duplicate indices, resulting in incorrect gradients for input_.
//
// Other edge cases handled:
// - Assign zero values to input gradients if matching indices are not found in grad_
// - grad_.values might have zeros
// --------------------------------------------------------------------
Tensor _sparse_sum_backward_cpu(const Tensor& grad_, const SparseTensor& input_, IntArrayRef dims_to_sum) {
  TORCH_CHECK(!grad_.is_cuda(), "_sparse_sum_backward_cpu: expected 'grad_' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!input_.is_cuda(), "_sparse_sum_backward_cpu: expected 'input_' to be CPU tensor, but got CUDA tensor");

  // Short circuit if grad_ is either zero or empty.
  if (((grad_.is_sparse() || at::sparse_csr::is_sparse_compressed(grad_)) && !grad_._nnz()) || !grad_.numel()) {
  // 如果输入是稀疏张量，则返回一个与输入形状相同的稀疏张量，其中所有稀疏维度都被求和，其他维度保持不变
  return at::zeros_like(input_);
}

// 将输入张量稀疏化，以便进一步处理
auto input = input_.coalesce();

// 获取输入张量的维度
const int64_t input_dim = input.dim();

// 将要求和的维度列表转换为位集，用于后续处理
auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);

// 获取要求和的维度列表
auto dims_to_sum_v = dims_to_sum.vec();

// 确保要求和的维度列表在输入维度范围内
maybe_wrap_dims(dims_to_sum_v, input_dim);

// 获取输入稀疏张量的索引
Tensor input_indices = input._indices();

// 获取输入稀疏张量的值
Tensor input_values = input._values();

// 获取输入张量的大小
IntArrayRef input_sizes = input.sizes();

// 获取输入稀疏张量的稀疏维度数
const int64_t input_sparse_dim = input.sparse_dim();

// 获取输入稀疏张量的稠密维度数
const int64_t input_dense_dim = input.dense_dim();

// 获取输入稀疏张量的非零元素数
const int64_t input_nnz = input._nnz();

// 初始化稀疏维度求和的大小
int64_t sparse_dims_to_sum_size = 0;

// 初始化要保留的稀疏维度列表
auto sparse_dims_to_keep_v = std::vector<int64_t>();

// 初始化要求和的稠密维度列表
auto dense_dims_to_sum_v = std::vector<int64_t>();

// 遍历输入张量的每个维度
for (auto d: c10::irange(input_dim)) {
  // 如果当前维度标记为要求和
  if (dims_to_sum_b[d]) {
    // 如果当前维度在稀疏维度内，则增加稀疏维度求和的计数
    if (d < input_sparse_dim) sparse_dims_to_sum_size++;
    // 否则，将当前维度添加到稠密维度求和的列表中
    else dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
  }
  // 如果当前维度不要求和且在稀疏维度内，则将其添加到保留的稀疏维度列表中
  else {
    if (d < input_sparse_dim) sparse_dims_to_keep_v.emplace_back(d);
  }
}

// 检查是否要对所有稀疏维度进行求和
const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);

// 检查是否要对稠密维度进行求和
const bool sum_dense_dim = (!dense_dims_to_sum_v.empty());

// 检查是否要对稀疏维度进行求和
const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

// 如果要对所有稀疏维度进行求和
if (sum_all_sparse_dim) {
  // 检查梯度张量是否为稠密张量
  TORCH_CHECK(!grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be dense since all sparse dims are summed");

  // 将梯度张量赋值给grad_input_values
  auto grad_input_values = grad_;

  // 获取扩展大小
  auto expand_size = input_values.sizes().vec();

  // 如果同时要求和稠密维度
  if (sum_dense_dim) {
    // 创建稠密扩展大小列表
    auto dense_expand_size = std::vector<int64_t>(expand_size);
    // 移除第一个维度（非零元素维度）
    dense_expand_size.erase(dense_expand_size.begin());
    // 断言稠密扩展大小列表的维度应与输入值张量的维度减一相匹配
    AT_ASSERT(dense_expand_size.size() == static_cast<size_t>(input_values.dim() - 1));
    // 根据稠密维度列表对grad_input_values进行扩展
    for (auto d : dense_dims_to_sum_v) grad_input_values = grad_input_values.unsqueeze(d - 1);  // -1 since grad has no nnz dim
    grad_input_values = grad_input_values.expand(dense_expand_size);
  }

  // 将grad_input_values根据expand_size扩展，并克隆为连续内存格式
  grad_input_values = grad_input_values.expand(expand_size).clone(at::MemoryFormat::Contiguous);

  // 检查输入张量是否已经联合
  bool grad_is_coalesced = input.is_coalesced();

  // 返回一个新的稀疏COO张量，其形状和张量类型与输入相同，值为grad_input_values，保持输入张量的稀疏特性
  return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, input.options().dtype(grad_.dtype()), grad_is_coalesced); // convert to grad dtype
}

// 如果不是对所有稀疏维度进行求和
else {
  // 检查梯度张量是否为稀疏张量
  TORCH_CHECK(grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be sparse, but got dense");

  // 对梯度张量进行稀疏化处理
  auto grad = grad_.coalesce();

  // 获取稀疏梯度张量的索引
  Tensor grad_indices = grad._indices();

  // 获取稀疏梯度张量的值
  Tensor grad_values = grad._values();

  // 获取稀疏梯度张量的稀疏维度数
  const int64_t grad_sparse_dim = grad.sparse_dim();

  // 获取稀疏梯度张量的非零元素数
  const int64_t grad_nnz = grad._nnz();

  // 将grad_values扩展为grad_values_expand
  Tensor grad_values_expand = grad_values;
    // 如果 sum_dense_dim 不为零，则进行以下操作
    if (sum_dense_dim) {
      // 获取输入张量的尺寸信息，并转换为向量形式
      auto expand_size = input_values.sizes().vec();
      // 如果 sum_sparse_dim 不为零，则将扩展尺寸的第一个维度设置为 grad_values 的大小
      if (sum_sparse_dim) expand_size[0] = grad_values.size(0);
      // 针对需要求和的密集维度进行循环，将 grad_values_expand 在这些维度上进行unsqueeze操作
      for (auto d : dense_dims_to_sum_v) grad_values_expand = grad_values_expand.unsqueeze(d);
      // 将 grad_values_expand 按照 expand_size 扩展，并且使用 Contiguous 内存格式进行克隆
      grad_values_expand = grad_values_expand.expand(expand_size).clone(at::MemoryFormat::Contiguous);
    }

    // 初始化 grad_input_values 张量
    Tensor grad_input_values;
    // 如果 sum_sparse_dim 不为零，则执行以下操作
    if (sum_sparse_dim) {
      // 创建一个和 input_values 相同大小和类型的全零张量，使用 LEGACY_CONTIGUOUS_MEMORY_FORMAT 内存格式
      grad_input_values = at::zeros_like(input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // 获取在梯度和输入张量上的扁平化索引
      auto grad_sparse_dim_to_keep_v = std::vector<int64_t>(grad_sparse_dim);
      std::iota(grad_sparse_dim_to_keep_v.begin(), grad_sparse_dim_to_keep_v.end(), 0);

      auto grad_indices_1D = flatten_indices_by_dims(grad_indices, grad.sizes(), grad_sparse_dim_to_keep_v); // 将梯度张量的索引在所有稀疏维度上扁平化，输出的索引是合并并排序的
      auto grad_indices_1D_accessor = grad_indices_1D.accessor<int64_t, 1>();
      auto input_indices_1D = flatten_indices_by_dims(input_indices, input_sizes, sparse_dims_to_keep_v);
      auto input_indices_1D_accessor = input_indices_1D.accessor<int64_t, 1>();

      // 配置张量迭代器，配置输出为 grad_input_values，输入为 grad_values_expand
      const auto copy_iter = TensorIteratorConfig()
        .add_output(grad_input_values)
        .add_input(grad_values_expand)
        .resize_outputs(false)
        .declare_static_shape(grad_values_expand.sizes(), /*squash_dims=*/0)
        .build();
      const auto device_type = kCPU;

      // 获取 grad_input_values 和 grad_values_expand 的指针数据
      const auto gIv_data = reinterpret_cast<char*>(grad_input_values.data_ptr());
      const auto gOv_data = reinterpret_cast<char*>(grad_values_expand.data_ptr());
      // 获取 grad_input_values 和 grad_values_expand 的步长
      const auto gIv_stride = (grad_input_values.strides()[0] *
                               grad_input_values.element_size());
      const auto gOv_stride = (grad_values_expand.strides()[0] *
                               grad_values_expand.element_size());

      // 使用二分查找来匹配索引
      at::parallel_for(0, input_nnz, 0, [&](int64_t start, int64_t end) {
        TensorIterator copy_iter_local(copy_iter);

        for (auto i: c10::irange(start, end)) {
          int64_t input_idx = input_indices_1D_accessor[i];
          int64_t l = 0, r = grad_nnz - 1;
          while (l <= r) {
            int64_t m = l + (r - l) / 2;
            if (grad_indices_1D_accessor[m] == input_idx) {
              // 将 grad_values_expand[m] 的值拷贝到 grad_input_values[i] 中
              copy_iter_local.unsafe_replace_operand(0, gIv_data + i * gIv_stride);
              copy_iter_local.unsafe_replace_operand(1, gOv_data + m * gOv_stride);
              copy_stub(device_type, copy_iter_local, /*non_blocking=*/false);
              break;
            }
            if (grad_indices_1D_accessor[m] < input_idx) {
              l = m + 1;
            }
            else {
              r = m - 1;
            }
          }
        }
      });
    }
    else {
      grad_input_values = grad_values_expand;
    }


    // 如果条件不满足，则使用扩展后的梯度值
    grad_input_values = grad_values_expand;



    bool grad_is_coalesced = input.is_coalesced();


    // 检查输入张量是否已经是紧凑格式（coalesced）
    bool grad_is_coalesced = input.is_coalesced();



    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, grad.options(), grad_is_coalesced);
  }


    // 返回稀疏 COO 张量，使用指定的维度和张量
    return at::_sparse_coo_tensor_with_dims_and_tensors(
        input_sparse_dim,                   // 稀疏张量的稀疏维度
        input_dense_dim,                    // 稀疏张量的密集维度
        input_sizes,                        // 稀疏张量的尺寸
        input_indices.clone(at::MemoryFormat::Contiguous),  // 克隆输入索引张量，保证内存布局为连续的
        grad_input_values,                  // 梯度输入值
        grad.options(),                     // 梯度的选项（如数据类型和设备）
        grad_is_coalesced                   // 梯度是否已紧凑
    );
  }
}

// 稀疏张量的任意元素是否为真值，要求张量本身必须是稀疏的
Tensor any_sparse(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_sparse());

  // 调用底层的 ATen 函数，返回稀疏张量 self 的非零元素构成的新张量
  return at::any(self._values());
}

// 计算稀疏张量 self 与密集张量 mat2 的矩阵乘积，结果保存在新建的 result 张量中
Tensor bmm_sparse_cpu(const SparseTensor& self, const Tensor& mat2) {
  // 根据 mat2 的选项创建一个空的张量 result
  Tensor result = at::empty({}, mat2.options());
  // 调用 bmm_out_sparse_cpu 函数进行计算并返回结果
  return bmm_out_sparse_cpu(self, mat2, result);
}

// 在已排序的步幅数组中搜索指定值的最右边实例
// 数组必须从最低到最高排序
// 返回找到元素的索引
// 通过引用返回 found，true 表示找到搜索值，false 表示未找到
template<typename scalar_t>
scalar_t binary_search_strided_rightmost(scalar_t search_val, TensorAccessor<scalar_t, 1>& sorted_arr_accessor, int64_t sorted_arr_begin_idx, int64_t length, bool* found) {
  if (length == 0) {
    *found = false;
    return -1;
  }

  int64_t left_ind = 0;
  int64_t right_ind = length - 1;
  // 初始中间索引值使用极限值，以确保在循环中被覆盖
  int64_t mid_ind = std::numeric_limits<int64_t>::max();
  bool done_searching = false;

  while (!done_searching) {
    mid_ind = left_ind + (right_ind - left_ind) / 2;
    scalar_t mid_val = sorted_arr_accessor[sorted_arr_begin_idx + mid_ind];

    if (mid_val > search_val) {
      right_ind = mid_ind - 1;
    } else if ((mid_val == search_val) && (
      (mid_ind == length - 1) || (sorted_arr_accessor[sorted_arr_begin_idx + mid_ind + 1] != search_val)
    )) {
      done_searching = true;
      *found = true;
    } else {
      left_ind = mid_ind + 1;
    }

    if (left_ind > right_ind) {
      done_searching = true;
      *found = false;
      mid_ind = -1;
    }
  }

  return mid_ind;
}

// 计算稀疏张量 self 与密集张量 mat2 的矩阵乘积，结果保存在 result 张量中并返回 result 的引用
Tensor& bmm_out_sparse_cpu(const SparseTensor& self, const Tensor& mat2, Tensor& result) {
  // 检查 mat2 是否为密集张量，若是则抛出异常
  TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");

  // 检查 self 的稠密维度必须为 0
  TORCH_CHECK(self.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self.dense_dim());
  // 检查 self 的稀疏维度必须为 3
  TORCH_CHECK(self.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self.sparse_dim());
  // 检查 mat2 的维度必须为 3
  TORCH_CHECK(mat2.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2.dim());

  // 检查 self 和 mat2 在第一维和第三维上的大小是否匹配
  TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
  TORCH_CHECK(self.size(2) == mat2.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");

  // 调整 result 张量的大小以匹配矩阵乘积的结果
  result.resize_({self.size(0), self.size(1), mat2.size(2)});

  // 如果 self 的非零元素个数为 0，则将 result 张量所有元素置为 0
  if (self._nnz() == 0) {
    result.zero_();
    // 返回结果
    return result;
  }

  // 首先需要进行合并操作，以按顺序获取第一维度的所有索引，
  // 因为我们将把每个矩阵发送到矩阵乘法操作中去
  SparseTensor self_coalesced = self.coalesce();

  // 获取稀疏张量的非零元素数量
  int64_t nnz = self_coalesced._nnz();
  // 获取稀疏张量的索引
  Tensor indices = self_coalesced._indices();
  // 获取稀疏张量的值
  Tensor values = self_coalesced._values();

  // 获取第一维度的索引
  Tensor indices_dim0 = indices[0];
  // 获取第一维度索引的访问器
  auto indices_dim0_accessor = indices_dim0.accessor<int64_t, 1>();
  // 在第0到2的位置上对索引进行切片
  Tensor indices_dim1_dim2 = indices.slice(0, 1, 3);

  // 获取第一维度的大小
  int64_t dim_i = self_coalesced.size(1);
  // 获取第二维度的大小
  int64_t dim_j = self_coalesced.size(2);
  // 获取矩阵mat2的第2维度的大小
  int64_t dim_k = mat2.size(2);

  // 初始化beta为标量0
  Scalar beta = 0;
  // 初始化一个临时张量t_dummy
  Tensor t_dummy;
  // 初始化alpha为标量1
  Scalar alpha = 1;

  // 初始化矩阵元素开始索引为0
  int64_t mat_el_begin_idx = 0;

  // 获取3D张量self_coalesced的矩阵数量
  int64_t num_matrices = self_coalesced.size(0);

  // 遍历每组3D张量输入中的2D矩阵，对每一个执行矩阵乘法操作
  // 这里开始矩阵乘法的调度和分发
  int64_t start_mat_num = indices_dim0_accessor[0];
  // 根据所有类型和复杂类型分派操作
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
    values.scalar_type(), "bmm_sparse_dense", [&] {
      // 遍历所有的矩阵
      for (int64_t cur_mat_num = 0;
        (cur_mat_num < num_matrices);
        cur_mat_num++
      ) {
        // 如果在开始或结束位置存在全零稀疏矩阵，则将结果矩阵置零
        if ((cur_mat_num < start_mat_num) || (mat_el_begin_idx >= nnz)) {
          result[cur_mat_num].zero_();
          continue;
        }

        // 查找对应当前矩阵编号的稀疏张量元素范围。已知当前矩阵开始位置，需要找到结束位置。
        bool mat_end_found;
        // 使用二分搜索确定右侧最后一个匹配的元素索引
        int64_t mat_el_end_idx = binary_search_strided_rightmost(
          cur_mat_num,
          indices_dim0_accessor,
          mat_el_begin_idx,
          nnz-mat_el_begin_idx,
          &mat_end_found
        ) + mat_el_begin_idx;

        if (mat_end_found) {
          mat_el_end_idx++;

          // 创建张量视图，仅查看当前矩阵集合的部分
          const Tensor dense_matrix = mat2[cur_mat_num];
          Tensor result_matrix = result[cur_mat_num];
          Tensor sparse_indices = indices_dim1_dim2.slice(1, mat_el_begin_idx, mat_el_end_idx);
          Tensor sparse_values = values.slice(0, mat_el_begin_idx, mat_el_end_idx);
          int64_t sparse_nnz = mat_el_end_idx - mat_el_begin_idx;

          // 执行稀疏矩阵乘法与稠密矩阵的加权和操作
          s_addmm_out_sparse_dense_worker<scalar_t>(
            sparse_nnz,
            dim_i, dim_j, dim_k,
            result_matrix,
            beta, t_dummy, alpha,
            sparse_indices, sparse_values,
            dense_matrix
          );
          mat_el_begin_idx = mat_el_end_idx;

        // 如果没有找到这个稀疏矩阵的元素，则它是零矩阵，需要将结果矩阵置零
        } else {
          result[cur_mat_num].zero_();
        }
      }
    }
  );
  // 返回计算结果
  return result;
} // 结束当前函数 conj_physical_out_sparse

Tensor& conj_physical_out_sparse(const Tensor& input, Tensor& result) {
    // 使用 TORCH_INTERNAL_ASSERT 断言输入张量 input 是稀疏张量
    TORCH_INTERNAL_ASSERT(input.is_sparse());
    
    // 如果 result 不是同一个张量对象作为 input，则将 input 的稀疏值复制到 result
    if (!is_same_tensor(result, input)) {
        copy_sparse_to_sparse_(result, input);
    }
    
    // 如果 input 不是复数张量，则直接返回 result
    if (!input.is_complex()) {
        return result;
    }
    
    // 否则，获取 result 的值张量，并对 input 的值张量执行共轭操作，结果保存在 result_values 中
    Tensor result_values = result._values();
    at::conj_physical_out(result_values, input._values());
    
    // 返回更新后的 result 张量
    return result;
}

} // 结束命名空间 at::native
```