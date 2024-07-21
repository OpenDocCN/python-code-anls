# `.\pytorch\aten\src\ATen\native\cpu\BinaryOpsKernel.cpp`

```
// 定义宏 TORCH_ASSERT_NO_OPERATORS，用于指定不包含操作符
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库中的二元操作函数头文件
#include <ATen/native/BinaryOps.h>

// 包含数学函数库的头文件
#include <cmath>

// 包含 ATen 库中的分发机制相关头文件
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
// 包含 ATen 库中的数学类型定义头文件
#include <ATen/OpMathType.h>
// 包含 ATen 库中的并行处理相关头文件
#include <ATen/Parallel.h>
// 包含 ATen 库中 CPU 矢量化功能相关头文件
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
// 包含 ATen 库中的数学计算函数头文件
#include <ATen/native/Math.h>
// 包含 ATen 库中的张量迭代器功能头文件
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库中 CPU 版本的对数加指数运算功能头文件
#include <ATen/native/cpu/LogAddExp.h>
// 包含 ATen 库中 CPU 版本的循环执行功能头文件
#include <ATen/native/cpu/Loops.h>
// 包含 C10 库中的宏定义头文件
#include <c10/macros/Macros.h>
// 包含 C10 库中的类型安全符号数学计算头文件
#include <c10/util/TypeSafeSignMath.h>
// 包含 C10 库中的通用数学计算功能头文件
#include <c10/util/generic_math.h>

// ATen 库的命名空间 at::native
namespace at::native {

// 匿名命名空间，用于定义私有函数或变量
namespace {

// 使用 vec 命名空间，用于简化矢量操作的使用
using namespace vec;

// 模板函数，执行标量与矢量间的二元操作
template <
    typename scalar_t,                                // 标量数据类型
    typename Op,                                      // 操作类型
    typename opmath_t = at::opmath_type<scalar_t>,     // 操作数的数学类型
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> binary_op_scalar(
    const Vectorized<scalar_t>& a,                     // 输入矢量 a
    opmath_t b,                                        // 标量 b
    const Op& op) {                                    // 操作符 op
  Vectorized<opmath_t> vec_b(b);                       // 将标量 b 转为矢量 vec_b
  auto [a0, a1] = convert_to_float<scalar_t>(a);       // 将输入矢量 a 转为浮点数
  return convert_from_float<scalar_t>(                 // 返回转回标量类型的矢量结果
      op(a0, vec_b), op(a1, vec_b));
}

// 执行 add_clamp 操作的内核函数
void add_clamp_kernel(
    TensorIterator& iter,                              // 张量迭代器
    const Scalar& alpha_scalar,                        // 标量 alpha
    const Scalar& min_val,                             // 最小值标量
    const Scalar& max_val) {                           // 最大值标量
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();          // 获取 alpha 的值
    auto alpha_vec = Vectorized<scalar_t>(alpha);      // 将 alpha 转为矢量 alpha_vec
    auto min_scalar = min_val.to<scalar_t>();          // 获取最小值的标量
    auto min_vec = Vectorized<scalar_t>(min_scalar);   // 将最小值标量转为矢量
    auto max_scalar = max_val.to<scalar_t>();          // 获取最大值的标量
    auto max_vec = Vectorized<scalar_t>(max_scalar);   // 将最大值标量转为矢量
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t {
          return std::min(                             // 返回 a+alpha*b 的结果
              max_scalar,                              // 与最大值标量比较
              std::max(min_scalar, static_cast<scalar_t>(a + alpha * b))); // 与最小值标量比较
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) // 对矢量执行 add_clamp 操作
            __ubsan_ignore_undefined__ {
              auto add_clamp_res = vec::fmadd(b, alpha_vec, a);  // 计算 a + alpha*b
              add_clamp_res = vec::clamp_min(add_clamp_res, min_vec); // 对结果应用最小值限制
              add_clamp_res = vec::clamp_max(add_clamp_res, max_vec); // 对结果应用最大值限制
              return add_clamp_res;                   // 返回结果
            });
  });
}

// 执行 atan2 操作的内核函数
void atan2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "atan2_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a, scalar_t b) -> scalar_t {  // 标量 atan2 操作
              return std::atan2(a, b);                // 返回 atan2(a, b)
            },
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { // 矢量 atan2 操作
              return a.atan2(b);                      // 返回矢量 atan2(a, b)
            });
      });
}

// 如果未定义 C10_MOBILE 宏，则定义以下内容
#if !defined(C10_MOBILE)
// 定义宏_AT_DISPATCH_ALL_TYPES_AND_BOOL，用于分发和处理不同数据类型和布尔类型的操作
#define _AT_DISPATCH_ALL_TYPES_AND_BOOL(TYPE, NAME, ...) \
  AT_DISPATCH_V2(                                       \
      TYPE,                                             \
      NAME,                                             \
      AT_WRAP(__VA_ARGS__),                             \
      kComplexHalf,                                      \
      kHalf,                                             \
      kBool,                                             \
      kBFloat16,                                         \
      AT_EXPAND(AT_FLOAT8_TYPES),                        \
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),               \
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))

// 定义宏_AT_DISPATCH_ALL_TYPES_NO_BOOL，用于分发和处理不同数据类型但不包括布尔类型的操作
#define _AT_DISPATCH_ALL_TYPES_NO_BOOL(TYPE, NAME, ...) \
  AT_DISPATCH_V2(                                       \
      TYPE,                                             \
      NAME,                                             \
      AT_WRAP(__VA_ARGS__),                             \
      kComplexHalf,                                      \
      kHalf,                                             \
      kBFloat16,                                         \
      AT_EXPAND(AT_FLOAT8_TYPES),                        \
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),               \
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))

// 定义宏_AT_DISPATCH_MUL_TYPES，用于分发和处理特定的乘法操作类型
#define _AT_DISPATCH_MUL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_V2(TYPE, NAME, AT_WRAP(__VA_ARGS__),    \
      kHalf, kBFloat16,                                \
      AT_EXPAND(AT_FLOAT8_TYPES),                      \
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),             \
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))
#else
// 如果未定义AT_DISPATCH_V2，则使用更通用的分发宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4
#define _AT_DISPATCH_ALL_TYPES_AND_BOOL(TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                \
      kComplexHalf, kHalf, kBool, kBFloat16, TYPE, NAME, __VA_ARGS__)

// 如果未定义AT_DISPATCH_V2，则使用更通用的分发宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3
#define _AT_DISPATCH_ALL_TYPES_NO_BOOL(TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(               \
      kComplexHalf, kHalf, kBFloat16, TYPE, NAME, __VA_ARGS__)

// 如果未定义AT_DISPATCH_V2，则使用更通用的分发宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2
#define _AT_DISPATCH_MUL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(       \
      kHalf, kBFloat16, TYPE, NAME, __VA_ARGS__)
#endif

// 定义函数mul_kernel，接受一个TensorIteratorBase类型的引用参数iter
void mul_kernel(TensorIteratorBase& iter) {
  // 获取迭代器中的通用数据类型
  auto dtype = iter.common_dtype();

  // 如果数据类型是布尔型
  if (dtype == ScalarType::Bool) {
    // 调用cpu_kernel函数处理布尔型数据的逻辑运算
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } 
  // 如果数据类型是kComplexHalf（复数类型）
  else if (dtype == kComplexHalf) {
    // 调用cpu_kernel函数处理复数类型数据的乘法运算
    cpu_kernel(
        iter,
        [=](c10::complex<at::Half> a,
            c10::complex<at::Half> b) -> c10::complex<at::Half> {
          using comp_t = c10::complex<float>;
          return comp_t{a} * comp_t{b};
        });
  } 
  // 如果迭代器中只有两个标量且第二个标量不为空且数据类型是缩减的浮点类型
  else if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "mul_cpu_reduced_float", [&]() {
      // 根据指定的数据类型分发处理器函数，针对减少的浮点数类型执行乘法操作
      using opmath_t = at::opmath_type<scalar_t>;
      // 从迭代器中获取第二个操作数的原始标量值，并转换为 opmath_t 类型
      opmath_t b = iter.original_scalar_value<opmath_t>(2);
      // 从迭代器中移除第二个操作数
      iter.remove_operand(2);
      // 调用 CPU 内核向量化函数，对每个标量执行乘法操作
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) __ubsan_ignore_undefined__ -> scalar_t {
            // 使用标量 a 乘以标量 b，并转换为 opmath_t 类型返回
            return static_cast<opmath_t>(a) * b;
          },
          [=](Vectorized<scalar_t> a) __ubsan_ignore_undefined__ {
            // 使用标量 a 和标量 b 执行向量化乘法操作
            return binary_op_scalar(
                a,
                b,
                [](const Vectorized<opmath_t>& x,
                   const Vectorized<opmath_t>& y) { return x * y; });
          });
    });
  } else {
    _AT_DISPATCH_MUL_TYPES(dtype, "mul_cpu", [&]() {
      // 根据指定的数据类型分发处理器函数，执行乘法操作
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b)
              __ubsan_ignore_undefined__ -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              __ubsan_ignore_undefined__ { return a * b; });
    });
  }
}

void div_true_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();  // 获取迭代器中的数据类型
  if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "div_cpu_reduced_float", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;  // 定义操作数类型
      opmath_t b = iter.original_scalar_value<opmath_t>(2);  // 获取第二个操作数的标量值
      iter.remove_operand(2);  // 移除第二个操作数
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {  // 定义标量运算函数，处理浮点除零
            return static_cast<opmath_t>(a) / b;
          },
          [=](Vectorized<scalar_t> a) {  // 定义向量化运算函数
            return binary_op_scalar(
                a,
                b,
                [](const Vectorized<opmath_t>& x,
                   const Vectorized<opmath_t>& y) { return x / y; });
          });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kBFloat16, kHalf, dtype, "div_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {  // 定义标量运算函数，处理浮点除零
                    return a / b;
                  },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {  // 定义向量化运算函数
                return a / b;
              });
        });
  }
}

void div_trunc_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();  // 获取迭代器中的数据类型
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    // 整数类型没有 SIMD 整数除法，因此不尝试向量化
    // TODO: 如果除数是标量，可以重写为乘以一个常数。
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {  // 定义整数除法的标量运算函数
        TORCH_CHECK(b != 0, "ZeroDivisionError");  // 检查除数不为零
        return a / b;
      });
    });
  } else if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        dtype, "div_trunc_cpu_reduced_float", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;  // 定义操作数类型
          opmath_t b = iter.original_scalar_value<opmath_t>(2);  // 获取第二个操作数的标量值
          iter.remove_operand(2);  // 移除第二个操作数
          cpu_kernel_vec(
              iter,
              [=](scalar_t a)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {  // 定义标量运算函数，处理浮点除零
                    return std::trunc(static_cast<opmath_t>(a) / b);
                  },
              [=](Vectorized<scalar_t> a) {  // 定义向量化运算函数
                return binary_op_scalar(
                    a,
                    b,
                    [](const Vectorized<opmath_t>& x,
                       const Vectorized<opmath_t>& y) {
                      return (x / y).trunc();  // 对结果向量进行截断操作
                    });
              });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, dtype, "div_trunc_cpu", [&]() {
          // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏来根据数据类型分发处理
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
                    // 对每对输入标量 a 和 b 执行截断除法操作，返回结果
                    return std::trunc(a / b);
                  },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                // 对每对向量化输入 a 和 b 执行截断除法操作，返回结果
                return (a / b).trunc();
              });
        });
  }
}

template <typename scalar_t>
inline Vectorized<scalar_t> div_floor_floating_vec(
    const Vectorized<scalar_t>& a,
    const Vectorized<scalar_t>& b) {
  using vec_t = Vectorized<scalar_t>;
  // 计算基本的浮点数除法
  const auto basic_div = a / b;
  // 创建一个无穷大的向量
  vec_t inf(std::numeric_limits<scalar_t>::infinity());
  // 计算 a 对 b 取模的结果
  auto mod = a.fmod(b);
  // 修正 Sleef_fmod 处理不正确的情况
  // 如果 basic_div 的绝对值为无穷大，并且 a 的绝对值不为无穷大，则修正为 a - mod，否则保持不变
  auto floor = vec_t::blendv(a - mod, a, (basic_div.abs() == inf) & (a.abs() != inf));
  // 对修正后的 floor 进行除法操作
  auto div = floor / b;
  const auto zero = vec_t(0);
  // 计算 mod 是否不等于零，并且 (b < 0) 异或 (mod < 0) 的掩码
  auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
  const auto one = vec_t(1);
  // 使用掩码进行修正，如果条件成立，则 div = div - 1
  div = vec_t::blendv(div, div - one, mask);
  // 对 div 取 floor 操作
  auto floordiv = div.floor();
  // 计算是否应该向上取整
  mask = (div - floordiv) > vec_t(0.5);
  floordiv = vec_t::blendv(floordiv, floordiv + one, mask);
  // 根据条件进行最终的修正
  floordiv = vec_t::blendv(floordiv, zero.copysign(basic_div), div == zero);
  floordiv = vec_t::blendv(floordiv, basic_div, b == zero);
  // 返回最终的结果 floordiv
  return floordiv;
};

void div_floor_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // 在无符号整数除法的特殊情况下，向下取整等同于截断除法（因为除数和被除数的符号总是相同的）
    return div_trunc_kernel(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    // 没有 SIMD 整数除法，因此不尝试进行向量化
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        return c10::div_floor_integer(a, b);
      });
    });
  } else {
    // 参见注释: [Python 中的向下取整除法]
    if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
          dtype, "div_floor_cpu_reduced_float", [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            // 获取操作数 b 的原始标量值
            opmath_t b = iter.original_scalar_value<opmath_t>(2);
            // 移除操作数 2
            iter.remove_operand(2);
            using vec_t = Vectorized<opmath_t>;
            // 使用 CPU 内核进行向量化计算
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t {
                  return c10::div_floor_floating(static_cast<opmath_t>(a), b);
                },
                [=](Vectorized<scalar_t> a) {
                  return binary_op_scalar(
                      a, b, [](const vec_t& x, const vec_t& y) {
                        return div_floor_floating_vec(x, y);
                      });
                });
          });
    // 如果数据类型是浮点类型（包括BFloat16和Half），则执行以下操作
    } else {
      // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏，处理浮点类型和两个特定类型（kBFloat16和kHalf）
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16, kHalf, dtype, "div_floor_cpu", [&]() {
            // 定义向量化类型vec_t为Vectorized<scalar_t>
            using vec_t = Vectorized<scalar_t>;
            // 调用cpu_kernel_vec函数处理迭代器iter
            cpu_kernel_vec(
                iter,
                // 匿名函数，对标量a和b执行向下取整除法
                [](scalar_t a, scalar_t b) -> scalar_t {
                  return c10::div_floor_floating(a, b);
                },
                // 匿名函数，对向量a和b执行向下取整除法
                [](vec_t a, vec_t b) -> vec_t {
                  return div_floor_floating_vec(a, b);
                });
          });
    }
  }
}

void remainder_kernel(TensorIteratorBase& iter) {
  // 检查数据类型是否为整数类型（不包括布尔型）
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    // 在整数类型下进行分发，命名为"remainder_cpu"
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_cpu", [&]() {
      // 调用 CPU 内核处理器，处理每个元素的取余操作
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        // 检查除数是否为零，若是则抛出异常
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        // 计算 a 对 b 的取余结果
        scalar_t r = a % b;
        // 处理余数与除数符号不一致的情况，使得余数保持正确性
        if ((r != 0) && (c10::is_negative(r) != c10::is_negative(b))) {
          r += b;
        }
        return r;
      });
    });
  } else if (iter.common_dtype() == kBFloat16) {
    // 若数据类型为 BFloat16，则使用特定的 CPU 内核向量化处理
    cpu_kernel_vec(
        iter,
        // lambda 表达式处理 BFloat16 类型的取余操作
        [=](BFloat16 a, BFloat16 b)
            __ubsan_ignore_float_divide_by_zero__ -> BFloat16 {
              // 将 BFloat16 转换为 float 进行取余运算
              float a0 = static_cast<float>(a);
              float b0 = static_cast<float>(b);
              float mod0 = std::fmod(a0, b0);
              // 处理余数与除数符号不一致的情况，保持余数正确性
              if ((mod0 != 0) && ((b0 < 0) != (mod0 < 0))) {
                mod0 += b0;
              }
              return mod0;
            },
        // lambda 表达式处理向量化的 BFloat16 类型取余操作
        [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          auto [a0, a1] = convert_bfloat16_float(a);
          auto [b0, b1] = convert_bfloat16_float(b);
          auto mod0 = a0.fmod(b0);
          auto mod1 = a1.fmod(b1);
          const auto zero = Vectorized<float>(0);
          auto mask0 = (mod0 != zero) & ((b0 < zero) ^ (mod0 < zero));
          auto mask1 = (mod1 != zero) & ((b1 < zero) ^ (mod1 < zero));
          // 根据条件合并余数与除数符号不一致的结果
          a0 = Vectorized<float>::blendv(mod0, mod0 + b0, mask0);
          a1 = Vectorized<float>::blendv(mod1, mod1 + b1, mask1);
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    // 对于其他浮点类型（包括半精度浮点），使用 CPU 内核向量化处理
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.common_dtype(), "remainder_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              // lambda 表达式处理浮点类型的取余操作
              [=](scalar_t a, scalar_t b)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
                    // 计算浮点数 a 对 b 的取余
                    scalar_t mod = std::fmod(a, b);
                    // 处理余数与除数符号不一致的情况，保持余数正确性
                    if ((mod != 0) && ((b < 0) != (mod < 0)))
                      mod += b;
                    return mod;
                  },
              // lambda 表达式处理向量化的浮点类型取余操作
              [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                auto mod = a.fmod(b);
                const auto zero = Vectorized<scalar_t>(0);
                auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
                // 根据条件合并余数与除数符号不一致的结果
                return Vectorized<scalar_t>::blendv(mod, mod + b, mask);
              });
        });
  }
}

void bitwise_and_kernel(TensorIteratorBase& iter) {
  // 若数据类型为布尔型，则使用 CPU 内核处理按位与操作
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [](bool a, bool b) { return a && b; });
  } else {
    // 对于整数类型，使用 CPU 内核向量化处理按位与操作
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          // lambda 表达式处理整数类型的按位与操作
          [](scalar_t a, scalar_t b) -> scalar_t { return a & b; },
          // lambda 表达式处理向量化的整数类型按位与操作
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a & b; });
    });
  }
}
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_or_cpu", [&]() {
      // 根据迭代器的数据类型分发处理函数，此处处理按位或操作
      cpu_kernel_vec(
          iter,
          // 对每对标量执行按位或操作并返回结果
          [](scalar_t a, scalar_t b) -> scalar_t { return a | b; },
          // 对每对向量化标量执行按位或操作并返回结果
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a | b; });
    });
  }
}

// 位异或操作的CPU内核函数
void bitwise_xor_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // 如果数据类型是布尔型，C++中^（位异或）操作不适用。bitwise_xor 函数包装了这一操作，适用于布尔和整型数据。
    cpu_kernel(iter, [](bool a, bool b) { return a != b; });
  } else {
    // 对于整型数据类型，使用模板函数 AT_DISPATCH_INTEGRAL_TYPES 进行分发，名称为 "bitwise_xor_cpu"
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_xor_cpu", [&]() {
      // 调用向量化的CPU内核函数，对每对元素执行按位异或操作
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return a ^ b; },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a ^ b; });
    });
  }
}

// 左移操作的CPU内核函数
void lshift_kernel(TensorIteratorBase& iter) {
  // 使用模板函数 AT_DISPATCH_INTEGRAL_TYPES 进行分发，名称为 "lshift_cpu"
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cpu", [&]() {
    // 调用向量化的CPU内核函数，对每对元素执行左移操作
    cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
          // 如果移位数为负数或超出类型最大位数限制，返回0
          if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
              (b >= max_shift)) {
            return 0;
          }
          return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a << b; });
  });
}

// 逻辑与操作的CPU内核函数
void logical_and_kernel(TensorIterator& iter) {
  // 如果数据类型是布尔型，使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3 进行分发，名称为 "logical_and_cpu"
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
          // 调用CPU内核函数，对每对元素执行逻辑与操作
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a && b; });
        });
  } else {
    // 对于其他数据类型，使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 进行分发，名称为 "logical_and_cpu"
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
          // 调用CPU内核函数，将逻辑与的结果转换为相应的数据类型
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return static_cast<scalar_t>(a && b);
          });
        });
  }
}

// 逻辑或操作的CPU内核函数
void logical_or_kernel(TensorIterator& iter) {
  // 如果数据类型是布尔型，使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3 进行分发，名称为 "logical_or_cpu"
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
          // 调用CPU内核函数，对每对元素执行逻辑或操作
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a || b; });
        });
  } else {
    // 对于其他数据类型，使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3 进行分发，名称为 "logical_or_cpu"
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
          // 调用CPU内核函数，将逻辑或的结果转换为相应的数据类型
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return static_cast<scalar_t>(a || b);
          });
        });
  }
}

// 逻辑异或操作的CPU内核函数
void logical_xor_kernel(TensorIterator& iter) {
  // 如果数据类型是布尔型，使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3 进行分发，名称为 "logical_xor_cpu"
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
          // 调用CPU内核函数，对每对元素执行逻辑异或操作
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
            return bool(a) != bool(b);
          });
        });
  } else {
    # 使用宏 `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2` 处理所有数据类型和复数类型，包括 `kBFloat16` 和 `kHalf`，
    # 并使用迭代器的常见数据类型，执行 CPU 端的逻辑异或操作。
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
          # 在 CPU 内核中执行函数，将每个元素 a 和 b 转换为布尔值后进行逻辑异或操作，
          # 返回结果的数据类型为 scalar_t。
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return static_cast<scalar_t>(bool(a) != bool(b));
          });
        });
    }
void rshift_kernel(TensorIteratorBase& iter) {
  // 使用 AT_DISPATCH_INTEGRAL_TYPES 宏，根据数据类型分发不同的 CPU 内核函数
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cpu", [&]() {
    // 调用 cpu_kernel_vec 函数，处理迭代器 iter
    cpu_kernel_vec(
        iter,
        // Lambda 函数，右移操作以保留符号位（有符号数）或不保留（无符号数）
        [](scalar_t a, scalar_t b) -> scalar_t {
          // 计算最大可右移的位数，根据是否有符号类型决定
          constexpr scalar_t max_shift =
              sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
          // 如果右移位数为负数或超过最大可移动位数，则返回右移最大位数的结果
          if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
              (b >= max_shift)) {
            return a >> max_shift;
          }
          // 否则返回右移 b 位后的结果
          return a >> b;
        },
        // Lambda 函数，处理向量化的右移操作
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a >> b; });
  });
}

void lt_kernel(TensorIteratorBase& iter) {
  // 查看特殊情况，输出为布尔类型
  if (iter.dtype() == ScalarType::Bool) {
    // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
          // 调用 cpu_kernel 函数，处理迭代器 iter
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a < b; });
        });
  } else {
    // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
          // 调用 cpu_kernel_vec 函数，处理迭代器 iter
          cpu_kernel_vec(
              iter,
              // Lambda 函数，小于比较操作
              [](scalar_t a, scalar_t b) -> scalar_t { return a < b; },
              // Lambda 函数，处理向量化的小于比较操作
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                  -> Vectorized<scalar_t> { return a.lt(b); });
        });
  }
}

void le_kernel(TensorIteratorBase& iter) {
  // 查看特殊情况，输出为布尔类型
  if (iter.dtype() == ScalarType::Bool) {
    // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
          // 调用 cpu_kernel 函数，处理迭代器 iter
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a <= b; });
        });
  } else {
    // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
          // 调用 cpu_kernel_vec 函数，处理迭代器 iter
          cpu_kernel_vec(
              iter,
              // Lambda 函数，小于等于比较操作
              [](scalar_t a, scalar_t b) -> scalar_t { return a <= b; },
              // Lambda 函数，处理向量化的小于等于比较操作
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                  -> Vectorized<scalar_t> { return a.le(b); });
        });
  }
}

void gt_kernel(TensorIteratorBase& iter) {
  // 查看特殊情况，输出为布尔类型
  if (iter.dtype() == ScalarType::Bool) {
    // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
          // 调用 cpu_kernel 函数，处理迭代器 iter
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a > b; });
        });
  } else {
    // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
          // 调用 cpu_kernel_vec 函数，处理迭代器 iter
          cpu_kernel_vec(
              iter,
              // Lambda 函数，大于比较操作
              [](scalar_t a, scalar_t b) -> scalar_t { return a > b; },
              // Lambda 函数，处理向量化的大于比较操作
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                  -> Vectorized<scalar_t> { return a.gt(b); });
        });
  }
}

void ge_kernel(TensorIteratorBase& iter) {
  // 查看特殊情况，输出为布尔类型
  if (iter.dtype() == ScalarType::Bool) {
    // 使用 AT_DISPATCH_ALL_TYPES_AND3 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          // 调用 cpu_kernel 函数，处理迭代器 iter
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a >= b; });
        });
  } else {
    // 使用 AT_DISPATCH_ALL_TYPES_AND2 宏，根据数据类型分发 CPU 内核函数
    AT_DISPATCH_ALL_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          // 调用 cpu_kernel_vec 函数，处理迭代器 iter
          cpu_kernel_vec(
              iter,
              // Lambda 函数，大于等于比较操作
              [](scalar_t a, scalar_t b) -> scalar_t { return a >= b; },
              // Lambda 函数，处理向量化的大于等于比较操作
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                  -> Vectorized<scalar_t> { return a.ge(b); });
        });
  }
}
    # 如果迭代器的常见数据类型包括所有类型和3个特定类型（kBool, kBFloat16, kHalf），
    # 并且函数名称为 "ge_cpu"，则执行以下代码块
    AT_DISPATCH_ALL_TYPES_AND3(
        kBool, kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          # 在 CPU 上执行核函数，比较两个标量值是否满足大于等于关系，返回布尔值
          cpu_kernel(
              iter, [](scalar_t a, scalar_t b) -> bool { return a >= b; });
        });
  } else {
    # 如果迭代器的常见数据类型包括所有类型和2个特定类型（kBFloat16, kHalf），
    # 并且函数名称为 "ge_cpu"，则执行以下代码块
    AT_DISPATCH_ALL_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
          # 在 CPU 上执行向量化核函数，比较两个向量化标量是否满足大于等于关系，返回相同类型的向量
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b) -> scalar_t { return a >= b; },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                  -> Vectorized<scalar_t> { return a.ge(b); });
        });
  }
void minimum_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中的数据类型是否为布尔型
  if (iter.dtype() == ScalarType::Bool) {
    // 如果是布尔型，调用 CPU 核函数处理，执行逻辑运算 a && b
    cpu_kernel(iter, [](bool a, bool b) -> bool { return a && b; });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    // 如果数据类型是整数类型（不包括布尔型）
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
      // 调用 CPU 向量化核函数处理
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return at::vec::minimum(a, b);
          });
    });
  } else {
    // 处理浮点数类型和其他特殊类型（Half 和 BFloat16）
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "minimum_cpu",
        [&]() {
          // 调用 CPU 向量化核函数处理
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b) -> scalar_t {
                // 处理 NaN 的情况，返回 quiet NaN
                if (a != a || b != b) {
                  return std::numeric_limits<scalar_t>::quiet_NaN();
                } else {
                  return std::min(a, b);  // 返回较小值
                }
              },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return at::vec::minimum(a, b);  // 返回向量中每个元素的最小值
              });
        });
  }
}
    # 根据数据类型分发不同的处理方式，处理整数类型
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
      # 调用 CPU 向量化函数，执行每个元素的最小值计算
      cpu_kernel_vec(
          iter,
          # Lambda 函数：计算两个标量的最小值
          [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
          # Lambda 函数：向量化计算两个向量的最小值
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return at::vec::minimum(a, b);
          });
    });
  } else {
    # 处理浮点数类型和特定类型（Half 和 BFloat16）
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "minimum_cpu",
        [&]() {
          cpu_kernel_vec(
              iter,
              # Lambda 函数：计算两个标量的最小值，处理 NaN 情况
              [](scalar_t a, scalar_t b) -> scalar_t {
                if (a != a || b != b) {  # 检查是否有 NaN 出现
                  return std::numeric_limits<scalar_t>::quiet_NaN();  # 返回 NaN
                } else {
                  return std::min(a, b);  # 返回两个数中的最小值
                }
              },
              # Lambda 函数：向量化计算两个向量的最小值
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return at::vec::minimum(a, b);
              });
        });
  }
void fmax_kernel(TensorIteratorBase& iter) {
  // 检查迭代器指向的数据类型是否为浮点类型
  if (isFloatingType(iter.common_dtype())) {
    // 根据迭代器的通用数据类型，调度浮点类型和Half、BFloat16类型的操作
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmax_cpu",
        [&]() {
          // 在CPU上执行核函数，比较两个标量的最大值并返回
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return std::fmax(a, b);
          });
        });
  } else {
    // 如果不是浮点类型，则调用最大值核函数
    maximum_kernel(iter);
  }
}

void fmin_kernel(TensorIteratorBase& iter) {
  // 检查迭代器指向的数据类型是否为浮点类型
  if (isFloatingType(iter.common_dtype())) {
    // 根据迭代器的通用数据类型，调度浮点类型和Half、BFloat16类型的操作
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmin_cpu",
        [&]() {
          // 在CPU上执行核函数，比较两个标量的最小值并返回
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return std::fmin(a, b);
          });
        });
  } else {
    // 如果不是浮点类型，则调用最小值核函数
    minimum_kernel(iter);
  }
}

void smooth_l1_kernel(TensorIteratorBase& iter, double beta) {
  // 如果迭代器指向的数据类型为BFloat16
  if (iter.dtype() == kBFloat16) {
    // 将beta值转换为float类型并进行向量化处理
    const float beta_val(beta);
    const Vectorized<float> beta_val_vec(beta_val);
    const Vectorized<float> point_five_vec(static_cast<float>(0.5));
    // 在CPU上执行向量化核函数，处理BFloat16类型数据的smooth L1损失
    cpu_kernel_vec(
        iter,
        // 定义BFloat16类型的smooth L1损失函数
        [&beta_val](BFloat16 a, BFloat16 b) -> BFloat16 {
          auto z = std::abs(float(a) - float(b));
          return z < beta_val ? static_cast<float>(0.5) * z * z / beta_val
                              : z - static_cast<float>(0.5) * beta_val;
        },
        // 处理向量化BFloat16数据的smooth L1损失函数
        [&beta_val_vec, &point_five_vec](
            Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          auto [a0, a1] = convert_bfloat16_float(a);
          auto [b0, b1] = convert_bfloat16_float(b);
          auto z = (a0 - b0).abs();
          a0 = Vectorized<float>::blendv(
              point_five_vec * z * z / beta_val_vec,
              z - point_five_vec * beta_val_vec,
              z >= beta_val_vec);
          z = (a1 - b1).abs();
          a1 = Vectorized<float>::blendv(
              point_five_vec * z * z / beta_val_vec,
              z - point_five_vec * beta_val_vec,
              z >= beta_val_vec);
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    // 如果迭代器指向的数据类型不是BFloat16，根据具体数据类型执行smooth L1损失计算
    AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "smooth_l1_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      const scalar_t beta_val(beta);
      const Vec beta_val_vec(beta_val);
      const Vec point_five_vec(static_cast<scalar_t>(0.5));
      // 在CPU上执行向量化核函数，处理其他浮点数类型数据的smooth L1损失
      cpu_kernel_vec(
          iter,
          // 定义浮点数类型的smooth L1损失函数
          [&beta_val](scalar_t a, scalar_t b) -> scalar_t {
            auto z = std::abs(a - b);
            return z < beta_val ? static_cast<scalar_t>(0.5) * z * z / beta_val
                                : z - static_cast<scalar_t>(0.5) * beta_val;
          },
          // 处理向量化浮点数数据的smooth L1损失函数
          [&beta_val_vec, &point_five_vec](Vec a, Vec b) {
            auto z = (a - b).abs();
            return Vec::blendv(
                point_five_vec * z * z / beta_val_vec,
                z - point_five_vec * beta_val_vec,
                z >= beta_val_vec);
          });
    });
  }
}
void huber_kernel(TensorIterator& iter, double delta) {
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的两种类型（kBFloat16, kHalf），生成对应的 CPU 函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "huber_cpu", [&]() {
        // 使用 Vectorized 类，为当前数据类型（scalar_t）创建矢量化操作
        using Vec = Vectorized<scalar_t>;
        // 将 delta 转换为 scalar_t 类型，并使用 Vec 封装为矢量化对象
        const scalar_t delta_val(delta);
        const Vec delta_val_vec(delta_val);
        const Vec point_five_vec(static_cast<scalar_t>(0.5));
        // 调用 cpu_kernel_vec 函数，使用 lambda 表达式实现 Huber 损失函数
        cpu_kernel_vec(
            iter,
            // 比较两个标量 a 和 b 的差的绝对值与 delta_val，计算 Huber 损失函数
            [&delta_val](scalar_t a, scalar_t b) -> scalar_t {
              auto z = std::abs(a - b);
              return z < delta_val
                  ? static_cast<scalar_t>(0.5) * z * z
                  : delta_val * (z - static_cast<scalar_t>(0.5) * delta_val);
            },
            // 使用矢量化操作处理 Vec 类型的 a 和 b，实现相同的 Huber 损失函数计算
            [&delta_val_vec, &point_five_vec](Vec a, Vec b) {
              auto z = (a - b).abs();
              return Vec::blendv(
                  point_five_vec * z * z,
                  delta_val_vec * (z - point_five_vec * delta_val_vec),
                  z >= delta_val_vec);
            });
      });
}

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  // 如果数据类型是复数类型，则调用对应的处理函数
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "sigmoid_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      // 使用 cpu_kernel_vec 函数，使用 lambda 表达式实现复数类型下的 sigmoid 反向传播函数
      cpu_kernel_vec(
          iter,
          // 计算复数 a 和 b 的 sigmoid 反向传播
          [=](scalar_t a, scalar_t b) -> scalar_t {
            return a * std::conj((scalar_t(1) - b) * b);
          },
          // 使用矢量化操作处理复数类型的 a 和 b，实现相同的 sigmoid 反向传播函数计算
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a * ((one_vec - b) * b).conj();
          });
    });
  } else if (iter.dtype() == kBFloat16) {
    auto one_vec = Vectorized<float>((float)(1));
    // 使用 cpu_kernel_vec 函数，使用 lambda 表达式实现 BFloat16 类型下的 sigmoid 反向传播函数
    cpu_kernel_vec(
        iter,
        // 计算 BFloat16 类型的 a 和 b 的 sigmoid 反向传播
        [=](BFloat16 a, BFloat16 b) -> BFloat16 {
          float a0 = static_cast<float>(a);
          float b0 = static_cast<float>(b);
          return a0 * (float(1) - b0) * b0;
        },
        // 使用矢量化操作处理 BFloat16 类型的 a 和 b，实现相同的 sigmoid 反向传播函数计算
        [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          auto [a0, a1] = convert_bfloat16_float(a);
          auto [b0, b1] = convert_bfloat16_float(b);
          a0 = a0 * (one_vec - b0) * b0;
          a1 = a1 * (one_vec - b1) * b1;
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        kHalf, iter.dtype(), "sigmoid_backward_cpu", [&]() {
          auto one_vec = Vectorized<scalar_t>((scalar_t)(1));
          // 使用 cpu_kernel_vec 函数，使用 lambda 表达式实现浮点类型和半精度类型下的 sigmoid 反向传播函数
          cpu_kernel_vec(
              iter,
              // 计算浮点类型和半精度类型的 a 和 b 的 sigmoid 反向传播
              [=](scalar_t a, scalar_t b) -> scalar_t {
                return a * (scalar_t(1) - b) * b;
              },
              // 使用矢量化操作处理浮点类型和半精度类型的 a 和 b，实现相同的 sigmoid 反向传播函数计算
              [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a * (one_vec - b) * b;
              });
        });
  }
}
// 反向传播 logit 函数的 CPU 实现，处理浮点数类型以及半精度浮点数类型（kBFloat16），使用迭代器进行操作
void logit_backward_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  // 根据迭代器的数据类型分发处理不同的浮点数类型
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "logit_backward_cpu", [&]() {
        // 将 Scalar 对象 eps_scalar 转换为当前浮点数类型的标量 eps
        const scalar_t eps = eps_scalar.to<scalar_t>();
        // 创建常量向量，表示零和一
        const Vectorized<scalar_t> kZeroVec(scalar_t(0));
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        // 如果 eps 小于零，处理条件为 NaN 的情况
        if (eps < scalar_t(0)) {
          // 创建 NaN 的向量 kNanVec
          const Vectorized<scalar_t> kNanVec(
              std::numeric_limits<scalar_t>::quiet_NaN());
          // CPU 内核函数，处理向量化操作
          cpu_kernel_vec(
              iter,
              [](scalar_t dy, scalar_t x) {
                // 根据 x 的值返回不同的结果：NaN 或者 dy / (x * (1 - x))
                return (x < scalar_t(0) || x > scalar_t(1))
                    ? std::numeric_limits<scalar_t>::quiet_NaN()
                    : ((x == scalar_t(0) || x == scalar_t(1))
                           ? (dy * std::numeric_limits<scalar_t>::infinity())
                           : (dy / (x * (scalar_t(1) - x))));
              },
              [kZeroVec, kOneVec, kNanVec](
                  Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
                // 使用向量化的方式处理，根据条件混合返回结果向量
                return Vectorized<scalar_t>::blendv(
                    kNanVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= kZeroVec) & (x_vec <= kOneVec));
              });
        } else {
          // 如果 eps 大于等于零，处理正常范围内的情况
          const scalar_t lo = eps;
          const scalar_t hi = scalar_t(1) - eps;
          const Vectorized<scalar_t> lo_vec(lo);
          const Vectorized<scalar_t> hi_vec(hi);
          // CPU 内核函数，处理向量化操作
          cpu_kernel_vec(
              iter,
              [lo, hi](scalar_t dy, scalar_t x) {
                // 根据 x 的值返回不同的结果：0 或者 dy / (x * (1 - x))
                return (x < lo || x > hi)
                    ? scalar_t(0)
                    : ((x == scalar_t(0) || x == scalar_t(1))
                           ? dy * std::numeric_limits<scalar_t>::infinity()
                           : dy / (x * (scalar_t(1) - x)));
              },
              [kZeroVec, kOneVec, lo_vec, hi_vec](
                  Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
                // 使用向量化的方式处理，根据条件混合返回结果向量
                return Vectorized<scalar_t>::blendv(
                    kZeroVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= lo_vec) & (x_vec <= hi_vec));
              });
        }
      });
}

// 反向传播 tanh 函数的 CPU 实现，处理复数类型的情况
void tanh_backward_kernel(TensorIteratorBase& iter) {
  // 如果数据类型为复数类型
  if (isComplexType(iter.dtype())) {
    // 根据数据类型分发处理复数类型
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
      // 创建标量向量，表示常数 1
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      // CPU 内核函数，处理向量化操作
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b) -> scalar_t {
            // 计算复数情况下的反向传播
            return a * std::conj(scalar_t{1} - b * b);
          },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            // 使用向量化的方式处理复数情况下的反向传播
            return a * (one_vec - b * b).conj();
          });
    });
  } else if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        iter.dtype(), "tanh_backward_cpu", [&]() {
          // 定义一个单元素的浮点向量，值为1
          auto one_vec = Vectorized<float>(float{1});
          // 调用 CPU 内核函数，对每对元素执行特定操作
          cpu_kernel_vec(
              iter,
              // 定义 CPU 内核函数中的操作：计算梯度反向传播的具体计算
              [=](scalar_t a, scalar_t b) -> scalar_t {
                float a0 = float(a);
                float b0 = float(b);
                // 计算 tanh 函数的反向传播中的局部梯度
                return a0 * (float{1} - b0 * b0);
              },
              // 定义 CPU 内核函数中的操作：向量化版本的操作
              [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                auto [a0, a1] = convert_to_float<scalar_t>(a);
                auto [b0, b1] = convert_to_float<scalar_t>(b);
                // 使用向量化的方式计算 tanh 函数的反向传播的局部梯度
                a0 = a0 * (one_vec - b0 * b0);
                a1 = a1 * (one_vec - b1 * b1);
                return convert_from_float<scalar_t>(a0, a1);
              });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
      // 定义一个单元素的标量类型向量，值为1
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      // 调用 CPU 内核函数，对每对元素执行特定操作
      cpu_kernel_vec(
          iter,
          // 定义 CPU 内核函数中的操作：计算梯度反向传播的具体计算
          [=](scalar_t a, scalar_t b) -> scalar_t {
            // 计算 tanh 函数的反向传播中的局部梯度
            return a * (scalar_t{1} - b * b);
          },
          // 定义 CPU 内核函数中的操作：向量化版本的操作
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            // 使用向量化的方式计算 tanh 函数的反向传播的局部梯度
            return a * (one_vec - b * b);
          });
    });
  }
}

void mse_kernel(TensorIteratorBase& iter) {
  // 如果数据类型是半精度浮点数，发出警告，因为使用CPU上的mse核可能比使用float或double类型的张量慢
  if (iter.dtype() == ScalarType::Half) {
    TORCH_WARN_ONCE(
        "Applying the CPU mse kernel on half-type tensors. "
        "This may be slower than using float or double-type tensors.");
  }

  // 根据张量迭代器的数据类型分发对应的CPU内核函数
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "mse_cpu", [&]() {
    // 调用cpu_kernel_vec函数处理迭代器，计算均方误差
    cpu_kernel_vec(
        iter,
        // 标量版本：计算两个标量之间的平方差
        [=](scalar_t a, scalar_t b) -> scalar_t {
          auto diff = a - b;
          return diff * diff;
        },
        // 向量化版本：计算两个向量的每个元素之间的平方差
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          auto diff = a - b;
          return diff * diff;
        });
  });
}

void fmod_kernel(TensorIteratorBase& iter) {
  // 如果迭代器的通用数据类型为整数类型（不包括布尔类型），则执行以下逻辑
  if (isIntegralType(iter.common_dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_cpu", [&]() {
      // 调用cpu_kernel函数处理迭代器，执行取模运算
      cpu_kernel(iter, [=](scalar_t x, scalar_t d) -> scalar_t {
        // 如果除数d为0，则抛出异常
        TORCH_CHECK(d != 0, "ZeroDivisionError");
        return x % d;
      });
    });
  } else {
    // 如果迭代器的通用数据类型为浮点数类型（包括BFloat16和Half）
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, iter.common_dtype(), "fmod_cpu", [&]() {
          // 调用cpu_kernel_vec函数处理迭代器，执行浮点数取模运算
          cpu_kernel_vec(
              iter,
              // 标量版本：计算x除以d的余数
              [](scalar_t x, scalar_t d) -> scalar_t {
                return std::fmod(x, d);
              },
              // 向量化版本：计算向量x和向量d的每个元素的余数
              [](Vectorized<scalar_t> x, Vectorized<scalar_t> d) {
                return x.fmod(d);
              });
        });
  }
}

void logaddexp_kernel(TensorIteratorBase& iter) {
  // 如果数据类型是缩减的浮点数类型，则执行以下逻辑
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      // 调用cpu_kernel_vec函数处理迭代器，执行logaddexp运算
      cpu_kernel_vec(
          iter,
          // 标量版本：计算两个标量a和b的logaddexp结果
          [=](scalar_t a, scalar_t b) -> scalar_t {
            float a0 = static_cast<float>(a);
            float b0 = static_cast<float>(b);
            if (std::isinf(a0) && a0 == b0) {
              return a0;
            } else {
              float m0 = std::max(a0, b0);
              return m0 + std::log1p(std::exp(-std::abs(a0 - b0)));
            }
          },
          // 向量化版本：计算两个向量a和b的每个元素的logaddexp结果
          [=](Vec a, Vec b) -> Vec {
            auto [a0, a1] = convert_to_float<scalar_t>(a);
            auto [b0, b1] = convert_to_float<scalar_t>(b);
            Vectorized<float> inf(std::numeric_limits<float>::infinity());
            Vectorized<float> m0 = maximum(a0, b0);
            Vectorized<float> m1 = maximum(a1, b1);
            a0 = Vectorized<float>::blendv(
                m0 + (a0 - b0).abs().neg().exp().log1p(),
                a0,
                (a0 == b0) & (a0.abs() == inf));
            a1 = Vectorized<float>::blendv(
                m1 + (a1 - b1).abs().neg().exp().log1p(),
                a1,
                (a1 == b1) & (a1.abs() == inf));
            return convert_from_float<scalar_t>(a0, a1);
          });
    });
  } else if (isComplexType(iter.dtype())) {
    // 如果数据类型是复数类型，则执行以下逻辑
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
      // 调用cpu_kernel函数处理迭代器，执行logaddexp辅助函数的计算
      cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
        return _log_add_exp_helper(a, b);
      });
    });
  } else {
    // 对输入的迭代器的浮点类型进行调度，用于 logaddexp_cpu 函数
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
      // 调用 CPU 内核向量化计算
      cpu_kernel_vec(
          iter,
          // 定义 Lambda 函数，计算两个标量的 logaddexp 操作结果
          [=](scalar_t a, scalar_t b) -> scalar_t {
            // 如果 a 是无穷大且等于 b，则返回 a
            if (std::isinf(a) && a == b) {
              return a;
            } else {
              // 否则计算较大的值 m
              scalar_t m = std::max(a, b);
              // 返回 m 加上 log1p(exp(-|a - b|)) 的结果
              return m + std::log1p(std::exp(-std::abs(a - b)));
            }
          },
          // 定义向量化 Lambda 函数，处理 Vectorized<scalar_t> 类型的 a 和 b
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            // 创建无穷大的向量
            Vectorized<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
            // 计算 a 和 b 中的较大值 m
            Vectorized<scalar_t> m = maximum(a, b);
            // 使用 blendv 函数根据条件混合返回向量化结果
            return Vectorized<scalar_t>::blendv(
                m + (a - b).abs().neg().exp().log1p(),  // m + abs(a - b).neg().exp().log1p() 的结果
                a,                                      // 如果条件不满足，则返回 a 向量
                (a == b) & (a.abs() == inf));           // 条件：a 等于 b 且 a 的绝对值等于无穷大
          });
    });
// 计算 logaddexp2_kernel 的核心逻辑，根据迭代器的数据类型调度不同的处理方式
void logaddexp2_kernel(TensorIteratorBase& iter) {
  // 检查迭代器的数据类型是否为降低的浮点类型
  if (at::isReducedFloatingType(iter.dtype())) {
    // 使用 AT_DISPATCH_REDUCED_FLOATING_TYPES 宏，根据具体的浮点类型执行以下操作
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "logaddexp2_cpu", [&]() {
      // 使用 Vectorized 包装浮点类型 scalar_t
      using Vec = Vectorized<scalar_t>;
      // 计算常量 inv_log_2，表示 ln(2) 的倒数
      constexpr auto inv_log_2 = static_cast<float>(1.0 / c10::ln_2<double>);
      // 使用 CPU 向量化操作处理迭代器中的每一对数据
      cpu_kernel_vec(
          iter,
          // 对单个标量值执行的操作：计算 logaddexp2
          [=](scalar_t a, scalar_t b) -> scalar_t {
            float a0 = static_cast<float>(a);
            float b0 = static_cast<float>(b);
            // 如果 a0 是正无穷并且 a0 等于 b0，则返回 a0
            if (std::isinf(a0) && a0 == b0) {
              return a0;
            } else {
              // 否则，计算最大值 m0，并返回 logaddexp2 的结果
              float m0 = std::max(a0, b0);
              return m0 + std::log1p(std::exp2(-std::abs(a0 - b0))) * inv_log_2;
            }
          },
          // 对向量化数据执行的操作：计算 logaddexp2
          [=](Vec a, Vec b) -> Vec {
            // 将向量化的数据转换为 float 类型
            auto [a0, a1] = convert_to_float<scalar_t>(a);
            auto [b0, b1] = convert_to_float<scalar_t>(b);
            // 创建包含正无穷大的向量
            Vectorized<float> inf(std::numeric_limits<float>::infinity());
            // 创建包含 inv_log_2 的向量
            Vectorized<float> inv_log_2_vec(inv_log_2);
            // 计算向量中的最大值 m0 和 m1
            Vectorized<float> m0 = maximum(a0, b0);
            Vectorized<float> m1 = maximum(a1, b1);
            // 根据条件混合向量中的值，执行 logaddexp2 计算
            a0 = Vectorized<float>::blendv(
                m0 + (a0 - b0).abs().neg().exp2().log1p() * inv_log_2_vec,
                a0,
                (a0 == b0) & (a0.abs() == inf));
            a1 = Vectorized<float>::blendv(
                m1 + (a1 - b1).abs().neg().exp2().log1p() * inv_log_2_vec,
                a1,
                (a1 == b1) & (a1.abs() == inf));
            return convert_from_float<scalar_t>(a0, a1);
          });
    });
  } else {
    // 如果数据类型不是降低的浮点类型，则执行以下操作
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cpu", [&]() {
      // 计算常量 inv_log_2，表示 ln(2) 的倒数
      constexpr auto inv_log_2 = static_cast<scalar_t>(1.0 / c10::ln_2<double>);
      // 使用 CPU 向量化操作处理迭代器中的每一对数据
      cpu_kernel_vec(
          iter,
          // 对单个标量值执行的操作：计算 logaddexp2
          [=](scalar_t a, scalar_t b) -> scalar_t {
            // 如果 a 是正无穷并且 a 等于 b，则返回 a
            if (std::isinf(a) && a == b) {
              return a;
            } else {
              // 否则，计算最大值 m，并返回 logaddexp2 的结果
              scalar_t m = std::max(a, b);
              return m + std::log1p(std::exp2(-std::abs(a - b))) * inv_log_2;
            }
          },
          // 对向量化数据执行的操作：计算 logaddexp2
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            // 创建包含正无穷大的向量
            Vectorized<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
            // 创建包含 inv_log_2 的向量
            Vectorized<scalar_t> inv_log_2_vec(inv_log_2);
            // 计算向量中的最大值 m
            Vectorized<scalar_t> m = maximum(a, b);
            // 根据条件混合向量中的值，执行 logaddexp2 计算
            return Vectorized<scalar_t>::blendv(
                m + (a - b).abs().neg().exp2().log1p() * inv_log_2_vec,
                a,
                (a == b) & (a.abs() == inf));
          });
    });
  }
}
    # 对每个 (a, b) 进行并行计算 CPU 内核
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
      # 计算 a 和 b 的最大公约数
      scalar_t g = calc_gcd(a, b);
      # 如果最大公约数为 0，返回 0；否则返回 a / g * b 的绝对值
      return (g == 0) ? 0 : std::abs(a / g * b);
    });
}

// 函数：在给定迭代器上执行 hypot_cpu 操作
void hypot_kernel(TensorIteratorBase& iter) {
  // 分派浮点类型和半精度浮点类型，针对 hypot_cpu 函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "hypot_cpu", [&]() {
        // 在 CPU 上执行向量化计算
        cpu_kernel_vec(
            iter,
            // 标量操作：计算两数的平方和的平方根
            [=](scalar_t a, scalar_t b) -> scalar_t {
              return std::hypot(a, b);
            },
            // 向量化操作：计算向量中每对元素的平方和的平方根
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
              return a.hypot(b);
            });
      });
}

// 函数：在给定迭代器上执行 igamma_cpu 操作
void igamma_kernel(TensorIteratorBase& iter) {
  // 分派半精度浮点类型和 bfloat16 类型，针对 igamma_cpu 函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "igamma_cpu", [&]() {
        // 在 CPU 上执行向量化计算
        cpu_kernel_vec(
            iter,
            // 标量操作：计算不完全伽马函数
            [=](scalar_t a, scalar_t b) -> scalar_t {
              return calc_igamma(a, b);
            },
            // 向量化操作：计算向量中每对元素的不完全伽马函数
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
              return a.igamma(b);
            });
      });
}

// 函数：在给定迭代器上执行 igammac_cpu 操作
void igammac_kernel(TensorIteratorBase& iter) {
  // 分派半精度浮点类型和 bfloat16 类型，针对 igammac_cpu 函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "igammac_cpu", [&]() {
        // 在 CPU 上执行向量化计算
        cpu_kernel_vec(
            iter,
            // 标量操作：计算补完全伽马函数
            [=](scalar_t a, scalar_t b) -> scalar_t {
              return calc_igammac(a, b);
            },
            // 向量化操作：计算向量中每对元素的补完全伽马函数
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
              return a.igammac(b);
            });
      });
}

// 函数：在给定迭代器上执行 nextafter_cpu 操作
void nextafter_kernel(TensorIteratorBase& iter) {
  // 检查是否为简化浮点类型
  if (at::isReducedFloatingType(iter.common_dtype())) {
    // 分派简化浮点类型，针对 nextafter_cpu 函数
    AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "nextafter_cpu", [&]() {
      // 在 CPU 上执行标量操作：计算两个数之间的下一个浮点数
      cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
        return std::nextafter(a, b);
      });
    });
  } else {
    // 分派非简化浮点类型，针对 nextafter_cpu 函数
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "nextafter_cpu", [&]() {
      // 在 CPU 上执行向量化计算
      cpu_kernel_vec(
          iter,
          // 标量操作：计算两个数之间的下一个浮点数
          [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::nextafter(a, b);
          },
          // 向量化操作：计算向量中每对元素之间的下一个浮点数
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.nextafter(b);
          });
    });
  }
}

// 函数：在给定迭代器上执行 heaviside_cpu 操作
void heaviside_kernel(TensorIteratorBase& iter) {
  // 分派所有类型以及半精度、布尔和 bfloat16 类型，针对 heaviside_cpu 函数
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cpu", [&]() {
        // 在 CPU 上执行标量操作：计算海维赛德阶跃函数
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
          return a == 0 ? b : static_cast<scalar_t>(a > 0);
        });
      });
}

// 函数：在给定迭代器上执行 copysign_cpu 操作
void copysign_kernel(TensorIteratorBase& iter) {
  // 分派半精度浮点类型和 bfloat16 类型，与迭代器公共数据类型一起，针对 copysign_cpu 函数
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "copysign_cpu", [&]() {
        // 在 CPU 上执行向量化计算
        cpu_kernel_vec(
            iter,
            // 标量操作：复制一个数的值并使用另一个数的符号
            [](scalar_t a, scalar_t b) -> scalar_t {
              return c10::copysign(a, b);
            },
            // 向量化操作：向量中每对元素执行符号复制
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
                -> Vectorized<scalar_t> { return a.copysign(b); });
      });
}
void xlogy_kernel(TensorIteratorBase& iter) {
  // 使用宏展开，处理浮点类型（包括特殊类型kBFloat16和kHalf），指定CPU函数名称"xlogy_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "xlogy_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          // 如果y为NaN，则返回NaN
          if (at::_isnan(y)) {
            return NAN;
          }
          // 如果x为0，则返回0
          if (x == 0) {
            return 0;
          }
          // 否则返回x乘以y的自然对数
          return x * std::log(y);
        });
      });
}

void xlog1py_kernel(TensorIteratorBase& iter) {
  // 使用宏展开，处理浮点类型（包括特殊类型kBFloat16和kHalf），指定CPU函数名称"xlog1py_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "xlog1py_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          // 如果y为NaN，则返回NaN
          if (at::_isnan(y)) {
            return NAN;
          }
          // 如果x为0，则返回0
          if (x == 0) {
            return 0;
          }
          // 否则返回x乘以log(1+y)
          return x * std::log1p(y);
        });
      });
}

void zeta_kernel(TensorIteratorBase& iter) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"zeta_cpu"
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_cpu", [&]() {
    // 调用CPU核函数，对每个元素执行操作
    cpu_kernel(
        iter, [](scalar_t x, scalar_t q) -> scalar_t { return zeta(x, q); });
  });
}

void chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"chebyshev_polynomial_t_cpu"
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "chebyshev_polynomial_t_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          // 返回第一类Chebyshev多项式T_n(x)
          return chebyshev_polynomial_t_forward(x, n);
        });
      });
} // chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"chebyshev_polynomial_u_cpu"
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "chebyshev_polynomial_u_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          // 返回第二类Chebyshev多项式U_n(x)
          return chebyshev_polynomial_u_forward(x, n);
        });
      });
} // chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"chebyshev_polynomial_v_cpu"
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "chebyshev_polynomial_v_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          // 返回第三类Chebyshev多项式V_n(x)
          return chebyshev_polynomial_v_forward(x, n);
        });
      });
} // chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"chebyshev_polynomial_w_cpu"
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "chebyshev_polynomial_w_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          // 返回第四类Chebyshev多项式W_n(x)
          return chebyshev_polynomial_w_forward(x, n);
        });
      });
} // chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator)

void hermite_polynomial_h_kernel(TensorIteratorBase& iterator) {
  // 使用宏展开，处理所有浮点类型，指定CPU函数名称"hermite_polynomial_h_cpu"
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "hermite_polynomial_h_cpu", [&]() {
        // 调用CPU核函数，对每个元素执行操作
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          // 返回Hermite多项式H_n(x)
          return hermite_polynomial_h_forward(x, n);
        });
      });
} // hermite_polynomial_h_kernel(TensorIteratorBase& iterator)
// 定义一个函数 hermite_polynomial_he_kernel，接受一个 Tensor 迭代器作为参数
void hermite_polynomial_he_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "hermite_polynomial_he_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Hermite 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return hermite_polynomial_he_forward(x, n);
        });
      });
} // hermite_polynomial_he_kernel(TensorIteratorBase& iterator)

// 定义一个函数 laguerre_polynomial_l_kernel，接受一个 Tensor 迭代器作为参数
void laguerre_polynomial_l_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "laguerre_polynomial_l_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Laguerre 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return laguerre_polynomial_l_forward(x, n);
        });
      });
} // laguerre_polynomial_l_kernel(TensorIteratorBase& iterator)

// 定义一个函数 legendre_polynomial_p_kernel，接受一个 Tensor 迭代器作为参数
void legendre_polynomial_p_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "legendre_polynomial_p_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Legendre 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return legendre_polynomial_p_forward(x, n);
        });
      });
} // legendre_polynomial_p_kernel(TensorIteratorBase& iterator)

// 定义一个函数 shifted_chebyshev_polynomial_t_kernel，接受一个 Tensor 迭代器作为参数
void shifted_chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Shifted Chebyshev T 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return shifted_chebyshev_polynomial_t_forward(x, n);
        });
      });
} // shifted_chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator)

// 定义一个函数 shifted_chebyshev_polynomial_u_kernel，接受一个 Tensor 迭代器作为参数
void shifted_chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Shifted Chebyshev U 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return shifted_chebyshev_polynomial_u_forward(x, n);
        });
      });
} // shifted_chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator)

// 定义一个函数 shifted_chebyshev_polynomial_v_kernel，接受一个 Tensor 迭代器作为参数
void shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Shifted Chebyshev V 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return shifted_chebyshev_polynomial_v_forward(x, n);
        });
      });
} // shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator)

// 定义一个函数 shifted_chebyshev_polynomial_w_kernel，接受一个 Tensor 迭代器作为参数
void shifted_chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator) {
  // 使用宏 AT_DISPATCH_FLOATING_TYPES，根据迭代器的公共数据类型分派到对应的函数模板
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cpu", [&]() {
        // 调用 cpu_kernel 函数，传入一个 lambda 表达式，用于计算 Shifted Chebyshev W 多项式的前向计算
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
          return shifted_chebyshev_polynomial_w_forward(x, n);
        });
      });
} // shifted_chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator)

// 结束命名空间
} // namespace

// 使用宏 REGISTER_DISPATCH 注册 add_clamp_stub，关联其对应的函数指针 &add_clamp_kernel
REGISTER_DISPATCH(add_clamp_stub, &add_clamp_kernel);
// 使用宏 REGISTER_DISPATCH 注册 mul_stub，关联其对应的函数指针 &mul_kernel
REGISTER_DISPATCH(mul_stub, &mul_kernel);
// 使用宏 REGISTER_DISPATCH 注册 div_true_stub，关联其对应的函数指针 &div_true_kernel
REGISTER_DISPATCH(div_true_stub, &div_true_kernel);
// 使用宏 REGISTER_DISPATCH 注册 div_trunc_stub，关联其对应的函数指针 &div_trunc_kernel
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel);
# 注册 div_floor_stub 到 div_floor_kernel 的分发器
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel);
# 注册 bitwise_and_stub 到 bitwise_and_kernel 的分发器
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);
# 注册 bitwise_or_stub 到 bitwise_or_kernel 的分发器
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel);
# 注册 bitwise_xor_stub 到 bitwise_xor_kernel 的分发器
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel);
# 注册 lshift_stub 到 lshift_kernel 的分发器
REGISTER_DISPATCH(lshift_stub, &lshift_kernel);
# 注册 rshift_stub 到 rshift_kernel 的分发器
REGISTER_DISPATCH(rshift_stub, &rshift_kernel);
# 注册 logical_xor_stub 到 logical_xor_kernel 的分发器
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel);
# 注册 logical_and_stub 到 logical_and_kernel 的分发器
REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel);
# 注册 logical_or_stub 到 logical_or_kernel 的分发器
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel);
# 注册 lt_stub 到 lt_kernel 的分发器
REGISTER_DISPATCH(lt_stub, &lt_kernel);
# 注册 le_stub 到 le_kernel 的分发器
REGISTER_DISPATCH(le_stub, &le_kernel);
# 注册 gt_stub 到 gt_kernel 的分发器
REGISTER_DISPATCH(gt_stub, &gt_kernel);
# 注册 ge_stub 到 ge_kernel 的分发器
REGISTER_DISPATCH(ge_stub, &ge_kernel);
# 注册 eq_stub 到 eq_kernel 的分发器
REGISTER_DISPATCH(eq_stub, &eq_kernel);
# 注册 ne_stub 到 ne_kernel 的分发器
REGISTER_DISPATCH(ne_stub, &ne_kernel);
# 注册 maximum_stub 到 maximum_kernel 的分发器
REGISTER_DISPATCH(maximum_stub, &maximum_kernel);
# 注册 minimum_stub 到 minimum_kernel 的分发器
REGISTER_DISPATCH(minimum_stub, &minimum_kernel);
# 注册 fmax_stub 到 fmax_kernel 的分发器
REGISTER_DISPATCH(fmax_stub, &fmax_kernel);
# 注册 fmin_stub 到 fmin_kernel 的分发器
REGISTER_DISPATCH(fmin_stub, &fmin_kernel);
# 注册 copysign_stub 到 copysign_kernel 的分发器
REGISTER_DISPATCH(copysign_stub, &copysign_kernel);
# 注册 remainder_stub 到 remainder_kernel 的分发器
REGISTER_DISPATCH(remainder_stub, &remainder_kernel);
# 注册 fmod_stub 到 fmod_kernel 的分发器
REGISTER_DISPATCH(fmod_stub, &fmod_kernel);
# 注册 gcd_stub 到 gcd_kernel 的分发器
REGISTER_DISPATCH(gcd_stub, &gcd_kernel);
# 注册 lcm_stub 到 lcm_kernel 的分发器
REGISTER_DISPATCH(lcm_stub, &lcm_kernel);
# 注册 xlogy_stub 到 xlogy_kernel 的分发器
REGISTER_DISPATCH(xlogy_stub, &xlogy_kernel);
# 注册 xlog1py_stub 到 xlog1py_kernel 的分发器
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_kernel);
# 注册 zeta_stub 到 zeta_kernel 的分发器
REGISTER_DISPATCH(zeta_stub, &zeta_kernel);
# 注册 nextafter_stub 到 nextafter_kernel 的分发器
REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel);
# 注册 heaviside_stub 到 heaviside_kernel 的分发器
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel);
# 注册 chebyshev_polynomial_t_stub 到 chebyshev_polynomial_t_kernel 的分发器
REGISTER_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_kernel);
# 注册 chebyshev_polynomial_v_stub 到 chebyshev_polynomial_v_kernel 的分发器
REGISTER_DISPATCH(chebyshev_polynomial_v_stub, &chebyshev_polynomial_v_kernel);
# 注册 chebyshev_polynomial_w_stub 到 chebyshev_polynomial_w_kernel 的分发器
REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_kernel);
# 注册 laguerre_polynomial_l_stub 到 laguerre_polynomial_l_kernel 的分发器
REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel);
# 注册 legendre_polynomial_p_stub 到 legendre_polynomial_p_kernel 的分发器
REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel);
# 注册 shifted_chebyshev_polynomial_t_stub 到 shifted_chebyshev_polynomial_t_kernel 的分发器
REGISTER_DISPATCH(shifted_chebyshev_polynomial_t_stub, &shifted_chebyshev_polynomial_t_kernel);
# 注册 shifted_chebyshev_polynomial_u_stub 到 shifted_chebyshev_polynomial_u_kernel 的分发器
REGISTER_DISPATCH(shifted_chebyshev_polynomial_u_stub, &shifted_chebyshev_polynomial_u_kernel);
# 注册 shifted_chebyshev_polynomial_v_stub 到 shifted_chebyshev_polynomial_v_kernel 的分发器
REGISTER_DISPATCH(shifted_chebyshev_polynomial_v_stub, &shifted_chebyshev_polynomial_v_kernel);
# 注册 shifted_chebyshev_polynomial_w_stub 到 shifted_chebyshev_polynomial_w_kernel 的分发器
REGISTER_DISPATCH(shifted_chebyshev_polynomial_w_stub, &shifted_chebyshev_polynomial_w_kernel);
# 注册 chebyshev_polynomial_u_stub 到 chebyshev_polynomial_u_kernel 的分发器，可能在启用了显式向量化后启用 AVX512 分发
REGISTER_DISPATCH(chebyshev_polynomial_u_stub, &chebyshev_polynomial_u_kernel);
# 注册 hermite_polynomial_h_stub 到 hermite_polynomial_h_kernel 的分发器
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_kernel);
# 注册 hermite_polynomial_he_stub 到 hermite_polynomial_he_kernel 的分发器
REGISTER_DISPATCH(hermite_polynomial_he_stub, &hermite_polynomial_he_kernel);

# ALSO_REGISTER_AVX512_DISPATCH 为以下函数注册 AVX512 分发器
ALSO_REGISTER_AVX512_DISPATCH(atan2_stub, &atan2_kernel);
ALSO_REGISTER_AVX512_DISPATCH(smooth_l1_stub, &smooth_l1_kernel);
ALSO_REGISTER_AVX512_DISPATCH(huber_stub, &huber_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel);
ALSO_REGISTER_AVX512_DISPATCH(logit_backward_stub, &logit_backward_kernel);
ALSO_REGISTER_AVX512_DISPATCH(tanh_backward_stub, &tanh_backward_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(mse_stub, &mse_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(logaddexp_stub, &logaddexp_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(logaddexp2_stub, &logaddexp2_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(hypot_stub, &hypot_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(igamma_stub, &igamma_kernel);
# 使用宏ALSO_REGISTER_AVX512_DISPATCH注册对应的函数和内核
ALSO_REGISTER_AVX512_DISPATCH(igammac_stub, &igammac_kernel);

} // namespace at::native
```