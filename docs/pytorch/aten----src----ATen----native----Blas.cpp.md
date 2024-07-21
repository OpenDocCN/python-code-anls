# `.\pytorch\aten\src\ATen\native\Blas.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于条件编译时限制仅仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量操作的头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Config.h>

// 包含 MKL-DNN 特定的矩阵乘法头文件
#include <ATen/native/mkldnn/Matmul.h>

// 根据 AT_PER_OPERATOR_HEADERS 宏的定义，选择性地包含不同的 CPU 操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

// 定义命名空间 at::meta，其中包含了元函数 addmv 的实现
namespace at::meta {
// TORCH_META_FUNC 宏用于声明元函数 addmv，接受一些输入张量和标量作为参数
TORCH_META_FUNC(addmv)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha) {
  // 对输入张量的维度进行检查，确保符合向量加矩阵乘以向量的预期格式
  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
    "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(), ", ", vec.dim());

  // 检查矩阵和向量的尺寸匹配，以及输出张量的尺寸是否正确
  TORCH_CHECK(mat.size(1) == vec.size(0) && (mat.size(0) == self.numel() || self.numel() == 1),
    "size mismatch, got input (", self.size(0), "), mat (", mat.size(0), "x", mat.size(1), "), vec (", vec.size(0), ")");
  
  // 推断命名张量的名称传播规则，并设置输出张量的尺寸和命名
  auto names = at::namedinference::propagate_names_for_addmv(mat, vec, self);
  set_output_raw_strided(0, IntArrayRef(mat.sizes().data(), 1), {}, vec.options(), names);
}
} // namespace at::meta

// 定义命名空间 at::native，包含了 addmv_out_cpu 函数的实现
namespace at::native {

// 模板函数 gemv 用于执行通用矩阵向量乘法的计算
template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

// 模板函数 dot_impl 用于执行通用的向量点积计算
template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

// 模板函数 vdot_impl 用于执行通用的向量共轭点积计算
template<typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

// constexpr 内联函数 lda_cond 用于检查矩阵乘法中的参数条件是否满足
constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda >= std::max<int64_t>(1L, m);
}

// TORCH_IMPL_FUNC 宏定义了 addmv_out_cpu 函数的实现，用于在 CPU 上执行向量加矩阵乘以向量操作
TORCH_IMPL_FUNC(addmv_out_cpu)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  // 使用 expand_size 函数调整 self 的尺寸，以匹配矩阵 mat 的行数
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  
  // 将 beta_ 和 alpha_ 转换为复数双精度数值
  auto betaval = beta_.toComplexDouble();

  // 处理空矩阵的情况下的快捷方式
  if (mat.numel() == 0) {
    // 当 beta == 0 时，结果张量 result 应该被清零
    if (betaval == 0.0) {
      result.zero_();
    } else {
      // 否则，通过乘法运算填充 result 张量
      at::cpu::mul_out(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), c10::nullopt /* layout */, at::kCPU, c10::nullopt /* pin_memory */));
    }
  } else {
    if (!result.is_same(*self_) && betaval != 0.0) { // 如果 result 不同于 self，并且 betaval 不等于 0.0
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      // 强制类型转换以调用 ATen 的 copy_ 函数，将 self_ 的内容复制到 result 中
      at::native::copy_(const_cast<Tensor&>(result), *self_);
    }
    if (result.numel() != 0) {
      // 使用 NoNamesGuard 来确保操作不受命名影响
      NoNamesGuard guard;
      // 如果使用 MKLDNN 来进行矩阵乘法操作，并且指定 result 为空 Tensor
      if (use_mkldnn_matmul(mat, vec, /*result=*/Tensor())){
        // 调用 MKLDNN 加速的矩阵乘法操作，并返回结果
        mkldnn_matmul(mat, vec, result, beta_.to<float>(), alpha_.to<float>());
        return;
      }

      auto r_stride = result.stride(0);
      // 根据 result 的数据类型进行分发处理，包括所有类型和复数类型
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, mat.scalar_type(), "addmv_impl_cpu", [&] {
        auto beta = beta_.to<scalar_t>();
        auto alpha = alpha_.to<scalar_t>();
        // 如果 mat 的第一维度步幅为 1，并且满足 lda 条件
        if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
          // 调用 gemv 函数执行矩阵向量乘法（不转置）
          gemv<scalar_t>('n', mat.size(0), mat.size(1), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(1),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        // 如果 mat 的第二维度步幅为 1，并且满足 lda 条件
        else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
          // 调用 gemv 函数执行矩阵向量乘法（转置）
          gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(0),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        // 如果不满足上述条件，需要先将 mat 转换为连续存储的 Tensor
        else {
          Tensor cmat = mat.contiguous();
          // 调用 gemv 函数执行矩阵向量乘法（转置）
          gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, cmat.const_data_ptr<scalar_t>(), cmat.stride(0),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
      });
    }
}

// 函数定义：计算矩阵向量乘法的结果，并将结果保存在用户提供的 result 张量中
Tensor &mv_out(const Tensor &self, const Tensor &vec, Tensor& result) {
  // 检查 result 张量的维度是否大于1，或者其元素数量与 self 的大小不匹配
  // 若不匹配，则创建一个正确大小的临时张量 self_addmv，以确保 addmv_out 函数调用时参数 self 符合要求
  if (result.dim() > 1 || (result.numel() != self.size(0) || result.numel() != 1)) {
    Tensor self_addmv = at::empty({self.size(0)}, vec.options());
    return at::addmv_out(result, self_addmv, self, vec, 0, 1);
  }
  // 若 result 张量符合要求，则直接调用 addmv_out 函数
  return at::addmv_out(result, result, self, vec, 0, 1);
}

// 函数定义：计算矩阵向量乘法的结果，并返回该结果张量
Tensor mv(const Tensor &self, const Tensor &vec) {
  // 创建一个空张量 result，用于存储计算结果
  Tensor result = at::empty({self.size(0)}, vec.options());
  // 使用 inplace 版本的 addmv_ 函数计算结果，并返回该结果张量
  return at::addmv_(result, self, vec, 0, 1);
}

// 函数定义：检查两个张量是否符合进行点积运算的条件
inline void dot_check(const Tensor& self, const Tensor& other) {
  // 检查 self 和 other 张量是否都是1维张量
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");

  // 检查 self 和 other 张量的数据类型是否相同
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());

  // 检查 self 和 other 张量的元素数量是否相同
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
}

// 函数定义：计算两个向量的点积并返回结果张量
Tensor dot(const Tensor &self, const Tensor &other){
  // 如果 self 是复数张量
  if (self.is_complex()) {
    // 如果 self 是共轭张量
    if (self.is_conj()) {
      // 如果 other 也是共轭张量，则计算共轭向量的点积，并返回其共轭结果
      if (other.is_conj()) {
        return (at::native::dot(self.conj(), other.conj())).conj();
      } else {
        // 如果 other 不是共轭张量，则计算共轭向量的点积，并返回其结果
        return at::native::vdot(self.conj(), other);
      }
    } else if (other.is_conj()) {
      // 如果 self 不是共轭张量，而 other 是共轭张量，则计算共轭向量的点积，并返回其结果
      return at::native::vdot(other.conj(), self);
    }
  }

  // 使用 NoNamesGuard 对象，执行 dot_check 函数检查 self 和 other 张量是否符合点积条件
  at::NoNamesGuard guard;
  dot_check(self, other);

  // 如果 self 或 other 张量是零张量，则返回一个有效的零张量
  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  // 如果可以使用 MKL-DNN 进行矩阵乘法运算
  if (use_mkldnn_matmul(self, other, /*result=*/Tensor())){
    // 创建一个大小为 [1, 1] 的空张量 r
    auto r =  at::empty({1, 1}, self.options());
    // 调用 mkldnn_matmul 函数计算矩阵乘法，并将结果保存在 r 中，使用参数 beta=0
    mkldnn_matmul(self, other, r, /*beta=*/0);
    // 返回计算结果张量 r
    return r;
  }

  // 对所有数值类型及复数类型执行 dot_impl 函数，计算两个向量的点积，并将结果保存在 result 张量中
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(), "dot", [&] {
    Tensor result = at::empty({}, self.options());
    result.fill_(dot_impl<scalar_t>(self.numel(), const_cast<scalar_t*>(self.const_data_ptr<scalar_t>()), self.stride(0), const_cast<scalar_t*>(other.const_data_ptr<scalar_t>()), other.stride(0)));
    return result;
  });
}
// 虚函数 `vdot` 的实现，计算两个张量的向量点积
Tensor vdot(const Tensor &self, const Tensor &other){
  // 如果 `self` 不是复数类型，则调用 `dot` 函数进行计算
  if (!self.is_complex()){
    return at::dot(self, other);
  }

  // 如果 `self` 是共轭的
  if (self.is_conj()) {
    // 如果 `other` 也是共轭的，则计算 `other` 和 `self` 的共轭向量点积
    if (other.is_conj()) {
      return at::native::vdot(other.conj(), self.conj());
    } else {
      // 如果 `other` 不是共轭的，则计算 `self` 的共轭和 `other` 的向量点积
      return at::native::dot(self.conj(), other);
    }
  } else if (other.is_conj()) {
    // 如果 `other` 是共轭的，则计算 `self` 和 `other` 的共轭向量点积，并对结果取共轭
    return (at::native::dot(self, other.conj())).conj();
  }

  // 禁用张量命名保护
  at::NoNamesGuard guard;
  // 复数类型的特定处理
  dot_check(self, other);

  // 如果 `self` 或 `other` 是零张量，则返回零张量
  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  // 调度到特定复数类型的 `vdot_impl` 函数执行向量点积计算
  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    // 创建一个空张量 `result`
    Tensor result = at::empty({}, self.options());
    // 使用 `vdot_impl` 计算复数类型的向量点积并填充到 `result` 中
    result.fill_(vdot_impl<scalar_t>(self.numel(), const_cast<scalar_t*>(self.const_data_ptr<scalar_t>()), self.stride(0), const_cast<scalar_t *>(other.const_data_ptr<scalar_t>()), other.stride(0)));
    return result;
  });

}

}  // namespace at::native
```