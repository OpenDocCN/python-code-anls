# `.\pytorch\aten\src\ATen\native\cpu\LerpKernel.cpp`

```
// 定义宏以禁用Torch操作符，包含相关头文件
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/util/irange.h>

namespace at {
namespace native {
namespace {

// 模板函数：判断给定的向量是否满足线性插值的权重小于0.5的条件
template <typename scalar_t>
Vectorized<scalar_t> is_lerp_weight_small(Vectorized<scalar_t> weight) {
  // 静态断言：确保scalar_t不是复数类型
  static_assert(!c10::is_complex<scalar_t>::value, "");
  return weight.abs() < Vectorized<scalar_t>(0.5);
}

// 如果是复数类型，is_lerp_weight_small不能工作，因为z.abs()返回一个无法比较的复数向量。
// 要么使用z.abs_2_()实现，要么回退到标量函数。
#if !(defined(CPU_CAPABILITY_DEFAULT) || defined(_MSC_VER))
template <typename value_t>
Vectorized<c10::complex<value_t>> is_lerp_weight_small(Vectorized<c10::complex<value_t>> weight) {
  using vec_reg_t = decltype(weight.abs_2_());
  // 使用z.abs_2_()实现的复数向量版本，比较向量值是否小于0.25
  vec_reg_t mask = Vectorized<value_t>(weight.abs_2_()) < Vectorized<value_t>(0.25);
  return Vectorized<c10::complex<value_t>>(mask);
}
#else
// 如果不是复数类型，在标量情况下执行线性插值的向量化计算
template <typename scalar_t>
Vectorized<scalar_t> lerp_vec_map(Vectorized<scalar_t> start, Vectorized<scalar_t> end, Vectorized<scalar_t> weight) {
  using vec_t = Vectorized<scalar_t>;
  // 存储开始、结束和权重向量的数组
  __at_align__ scalar_t start_arr[vec_t::size()];
  __at_align__ scalar_t end_arr[vec_t::size()];
  __at_align__ scalar_t weight_arr[vec_t::size()];
  __at_align__ scalar_t result_arr[vec_t::size()];

  // 将向量内容存储到数组中
  start.store(start_arr);
  end.store(end_arr);
  weight.store(weight_arr);

  // 使用标量函数进行每个元素的线性插值计算
  for (auto i : c10::irange(vec_t::size())) {
    result_arr[i] = lerp(start_arr[i], end_arr[i], weight_arr[i]);
  }
  // 从数组加载结果并返回向量
  return vec_t::loadu(result_arr);
}

// 复数类型的向量化线性插值计算
template <typename value_t>
Vectorized<c10::complex<value_t>> lerp_vec(Vectorized<c10::complex<value_t>> start, Vectorized<c10::complex<value_t>> end, Vectorized<c10::complex<value_t>> weight) {
  return lerp_vec_map(start, end, weight);
}
#endif

// 标量版本的线性插值计算内核
template <typename scalar_t>
Vectorized<scalar_t> lerp_vec(Vectorized<scalar_t> start, Vectorized<scalar_t> end, Vectorized<scalar_t> weight) {
  using vec_t = Vectorized<scalar_t>;
  // 判断权重是否小于0.5的条件
  auto mask = is_lerp_weight_small(weight);
  // 根据条件进行插值系数和基础值的选择
  auto coeff = vec_t::blendv(weight - vec_t(1), weight, mask);
  auto base = vec_t::blendv(end, start, mask);
  // 使用FMA指令进行线性插值计算并返回结果
  return vec::fmadd(coeff, end - start, base);
}

// 标量版本的线性插值计算内核函数
void lerp_scalar_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  if (iter.common_dtype() == kBFloat16) {
    using bVec = Vectorized<BFloat16>;
    using fVec = Vectorized<float>;
    // 将标量权重值转换为float类型
    float weight_val = weight.to<float>();
    auto weight_vec = fVec(weight_val);
    // 如果迭代器的公共数据类型是 BFloat16
    if (iter.common_dtype() == kBFloat16) {
        // 定义 BFloat16 的向量类型
        using bVec = Vectorized<BFloat16>;
        // 将权重 weight 转换为 float 类型
        float weight_val = weight.to<float>();
        // 使用权重值初始化 float 向量
        auto weight_vec = fVec(weight_val);
        // 调用 ATen 的 CPU 向量化函数，处理 BFloat16 类型数据
        at::native::cpu_kernel_vec(
          iter,
          // 内核函数，对每对 BFloat16 值进行线性插值
          [weight_val](BFloat16 self_val, BFloat16 end_val) -> BFloat16 {
            return lerp(self_val, end_val, weight_val);
          },
          // 向量化函数，将 BFloat16 向量插值为 BFloat16 向量
          [=](bVec self_vec, bVec end_vec) -> bVec {
              // 将 BFloat16 向量转换为 float 向量
              auto [self_vec0, self_vec1] = convert_bfloat16_float(self_vec);
              auto [end_vec0, end_vec1] = convert_bfloat16_float(end_vec);
              // 对每个分量进行向量化线性插值
              auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec);
              auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec);
              // 将 float 向量转换为 BFloat16 向量
              return convert_float_bfloat16(result0, result1);
          });
    } else if (iter.common_dtype() == kHalf) {
        // 如果迭代器的公共数据类型是 Half
        using hVec = Vectorized<Half>;
        using fVec = Vectorized<float>;
        // 将权重 weight 转换为 float 类型
        float weight_val = weight.to<float>();
        // 使用权重值初始化 float 向量
        auto weight_vec = fVec(weight_val);
        // 调用 ATen 的 CPU 向量化函数，处理 Half 类型数据
        at::native::cpu_kernel_vec(
          iter,
          // 内核函数，对每对 Half 值进行线性插值
          [weight_val](Half self_val, Half end_val) -> Half {
            return lerp(self_val, end_val, weight_val);
          },
          // 向量化函数，将 Half 向量插值为 Half 向量
          [=](hVec self_vec, hVec end_vec) -> hVec {
              // 将 Half 向量转换为 float 向量
              auto [self_vec0, self_vec1] = convert_half_float(self_vec);
              auto [end_vec0, end_vec1] = convert_half_float(end_vec);
              // 对每个分量进行向量化线性插值
              auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec);
              auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec);
              // 将 float 向量转换为 Half 向量
              return convert_float_half(result0, result1);
          });
    } else {
        // 如果迭代器的公共数据类型不是 BFloat16 也不是 Half
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_scalar", [&] {
          // 获取当前数据类型的标量权重值
          auto weight_val = weight.to<scalar_t>();
          // 调用 ATen 的 CPU 向量化函数，处理当前数据类型的标量数据
          at::native::cpu_kernel_vec(
              iter,
              // 内核函数，对每对标量值进行线性插值
              [weight_val](scalar_t self_val, scalar_t end_val) {
                return lerp(self_val, end_val, weight_val);
              },
              // 向量化函数，对标量向量进行插值
              [weight_val](Vectorized<scalar_t> self, Vectorized<scalar_t> end) {
                // 使用标量权重值初始化向量
                const Vectorized<scalar_t> weight(weight_val);
                // 调用向量化线性插值函数
                return lerp_vec(self, end, weight);
              });
        });
    }
} // 匿名命名空间结束

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  // 如果迭代器中的通用数据类型为 kBFloat16
  if (iter.common_dtype() == kBFloat16) {
    // 使用 BFloat16 向量化类型
    using bVec = Vectorized<BFloat16>;
    // 调用 CPU 内核向量化函数
    at::native::cpu_kernel_vec(
      iter,
      // Lambda 函数，计算 BFloat16 类型的线性插值
      [=](BFloat16 self_val, BFloat16 end_val, BFloat16 weight_val) -> BFloat16 {
        return lerp(self_val, end_val, weight_val);
      },
      // Lambda 函数，处理 BFloat16 向量的线性插值
      [=](bVec self_vec, bVec end_vec, bVec weight_vec) -> bVec {
          // 将 BFloat16 向量转换为 float 向量
          auto [self_vec0, self_vec1] = convert_bfloat16_float(self_vec);
          auto [end_vec0, end_vec1] = convert_bfloat16_float(end_vec);
          auto [weight_vec0, weight_vec1] = convert_bfloat16_float(weight_vec);
          // 对 float 向量执行线性插值
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec0);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec1);
          // 将结果转换回 BFloat16 向量
          return convert_float_bfloat16(result0, result1);
      });
  } else if (iter.common_dtype() == kHalf) {
    // 使用 Half 向量化类型
    using hVec = Vectorized<Half>;
    // 调用 CPU 内核向量化函数
    at::native::cpu_kernel_vec(
      iter,
      // Lambda 函数，计算 Half 类型的线性插值
      [=](Half self_val, Half end_val, Half weight_val) -> Half {
        return lerp(self_val, end_val, weight_val);
      },
      // Lambda 函数，处理 Half 向量的线性插值
      [=](hVec self_vec, hVec end_vec, hVec weight_vec) -> hVec {
          // 将 Half 向量转换为 float 向量
          auto [self_vec0, self_vec1] = convert_half_float(self_vec);
          auto [end_vec0, end_vec1] = convert_half_float(end_vec);
          auto [weight_vec0, weight_vec1] = convert_half_float(weight_vec);
          // 对 float 向量执行线性插值
          auto result0 = lerp_vec(self_vec0, end_vec0, weight_vec0);
          auto result1 = lerp_vec(self_vec1, end_vec1, weight_vec1);
          // 将结果转换回 Half 向量
          return convert_float_half(result0, result1);
      });
  } else {
    // 对于其他类型的迭代器通用数据类型，使用泛型分发
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "lerp_kernel_tensor", [&] {
      // 调用 CPU 内核向量化函数，处理泛型类型的线性插值
      at::native::cpu_kernel_vec(
          iter,
          // Lambda 函数，计算标量类型的线性插值
          [](scalar_t self_val, scalar_t end_val, scalar_t weight_val) {
            return lerp(self_val, end_val, weight_val);
          },
          // Lambda 函数，处理泛型向量的线性插值
          [](Vectorized<scalar_t> self_val, Vectorized<scalar_t> end_val, Vectorized<scalar_t> weight_val) {
            return lerp_vec(self_val, end_val, weight_val);
          });
    });
  }
}

} // 匿名命名空间结束

// 注册标量权重的线性插值分发函数
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);
// 注册张量权重的线性插值分发函数
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);

} // namespace native 结束
} // namespace at 结束
```