# `.\pytorch\aten\src\ATen\native\cpu\UnaryOpsKernel.cpp`

```
// 定义宏以禁用 Torch 的运算符断言
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库的一元操作头文件
#include <ATen/native/UnaryOps.h>

// 包含数学库、限制和类型特性的标准头文件
#include <cmath>
#include <limits>
#include <type_traits>

// 包含 ATen 库的配置、上下文、调度、并行、向量化功能等头文件
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/CopyKernel.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/OpMathType.h>

// 包含常数和标量的 C10 实用工具头文件
#include <c10/util/MathConstants.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/irange.h>

// 如果 MKL 可用，包含 MKL 头文件
#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

// ATen 库的命名空间
namespace at::native {

// 内联命名空间，与 CPU 功能相关
inline namespace CPU_CAPABILITY {

// 使用 vec 命名空间
using namespace vec;

// sigmoid_kernel 函数定义
static void sigmoid_kernel(TensorIteratorBase& iter) {
  // 获取迭代器中的公共数据类型
  const auto dtype = iter.common_dtype();
  // 如果是降低浮点类型
  if (at::isReducedFloatingType(dtype)) {
    // 使用降低的浮点类型分发函数
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "sigmoid_cpu_reduced_float", [&]() {
      // 使用 CPU 内核向量化操作
      cpu_kernel_vec(
          iter,
          // 标量函数：计算 sigmoid 函数
          [=](scalar_t a) -> scalar_t {
            float a0 = static_cast<float>(a);
            return static_cast<float>(1) / (static_cast<float>(1) + std::exp((-a0)));
          },
          // 向量化函数：使用 SIMD 指令加速计算
          [=](Vectorized<scalar_t> a) {
            auto [a0, a1] = convert_to_float<scalar_t>(a);
            a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
            a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
            return convert_from_float<scalar_t>(a0, a1);
          });
    });
  } else {
    // 如果是浮点或复数类型
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(dtype, "sigmoid_cpu", [&]() {
      // 使用 CPU 内核向量化操作
      cpu_kernel_vec(
          iter,
          // 标量函数：计算 sigmoid 函数
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          // 向量化函数：使用 SIMD 指令加速计算
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });
    });
  }
}

// 如果 MKL 可用，定义模板函数 VmlLog
#if AT_MKL_ENABLED()

// 模板函数：计算对数操作，使用 MKL 库
template <typename T>
void VmlLog(int64_t N, const T* X, T* Y) {
  constexpr int64_t K = Vectorized<T>::size();
  // 并行计算
  at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
    using VT = at::opmath_type<T>;
    // 使用向量化函数映射到 Y
    vec::map(
        [](Vectorized<VT> x_vec) { return x_vec.log(); },
        Y + begin,
        X + begin,
        end - begin);
  });
}

// 模板特化：float 类型的 VmlLog 函数，使用 MKL 库
template <>
void VmlLog<float>(int64_t N, const float* X, float* Y) {
  vsLn(N, X, Y);
}

// 模板特化：double 类型的 VmlLog 函数，使用 MKL 库
template <>
void VmlLog<double>(int64_t N, const double* X, double* Y) {
  vdLn(N, X, Y);
}

// 模板函数：LogitMKLKernel，使用 MKL 库计算 logit
template <typename T>
void LogitMKLKernel(T eps, TensorIteratorBase* it) {
  // 如果不能使用 32 位索引
  if (!it->can_use_32bit_indexing()) {
    // 对每个子迭代器使用 32 位索引
    for (auto& sub_it : it->with_32bit_indexing()) {
      LogitMKLKernel<T>(eps, &sub_it);
    }
    // 如果 eps 小于 0，则进行特殊处理：将 X_data 中值为 1 的元素映射为正无穷大，其它元素按公式进行计算
    return;
  }

  // 使用模板类型 T 的向量化大小作为常量 K
  constexpr int64_t K = Vectorized<T>::size();
  // 获取输入张量的元素数量 N
  const int64_t N = it->numel();
  // 获取输入张量的数据指针 X_data 和输出张量的数据指针 Y_data
  const T* X_data = static_cast<T*>(it->data_ptr(1));
  T* Y_data = static_cast<T*>(it->data_ptr(0));
  // 如果 eps 小于 T(0)
  if (eps < T(0)) {
    // 使用并行计算对区间 [0, N) 进行遍历
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      // 对每个区间内的元素进行处理
      for (const auto i : c10::irange(begin, end)) {
        // 如果 X_data[i] 等于 T(1)，则 Y_data[i] 被赋值为正无穷大，否则按公式计算
        Y_data[i] = X_data[i] == T(1) ? std::numeric_limits<T>::infinity()
                                      : X_data[i] / (T(1) - X_data[i]);
      }
      // 对处理后的区间进行日志记录
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  } else {
    // 否则，设置边界值 lo 和 hi
    const T lo = eps;
    const T hi = T(1) - eps;
    // 使用并行计算对区间 [0, N) 进行遍历
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      // 对每个区间内的元素进行处理
      for (const auto i : c10::irange(begin, end)) {
        // 将 X_data[i] 限制在 lo 和 hi 之间，并赋值给 x
        const T x = X_data[i] < lo ? lo : (X_data[i] > hi ? hi : X_data[i]);
        // 如果 x 等于 T(1)，则 Y_data[i] 被赋值为正无穷大，否则按公式计算
        Y_data[i] =
            x == T(1) ? std::numeric_limits<T>::infinity() : (x / (T(1) - x));
      }
      // 对处理后的区间进行日志记录
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  }
#else

// 如果未定义 AT_MKL_ENABLED，则定义 LogitMKLKernel 模板函数，接受 eps 和 TensorIteratorBase* 参数
template <typename T>
void LogitMKLKernel(T eps, TensorIteratorBase* it) {
  // 抛出错误，表示 ATen 未使用 MKL 编译
  TORCH_CHECK(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED

// 定义 logit_kernel 函数，接受 TensorIteratorBase& 和 eps_scalar 作为参数
static void logit_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和额外的 kBFloat16、kHalf 类型
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "logit_cpu", [&]() {
        // 获取 eps_scalar 的值，并转换为 scalar_t 类型
        const scalar_t eps = eps_scalar.to<scalar_t>();
        // 检查是否支持 MKL 并且迭代器是连续的
        if (at::hasMKL() && iter.is_contiguous()) {
          // 调用 LogitMKLKernel 模板函数，传入 eps 和迭代器地址，并转换输出
          LogitMKLKernel<scalar_t>(eps, &iter);
          iter.cast_outputs();  // 转换输出张量类型
        } else if (eps < scalar_t(0)) {
          // 如果 eps 小于 0，执行以下逻辑
          const Vectorized<scalar_t> kOneVec(scalar_t(1));
          // 使用 cpu_kernel_vec 处理迭代器，使用 lambda 表达式计算 logit
          cpu_kernel_vec(
              iter,
              [](scalar_t x) {
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()  // 如果 x 等于 1 返回无穷大
                    : std::log(x / (scalar_t(1) - x));  // 否则计算 logit
              },
              [kOneVec](Vectorized<scalar_t> x_vec) {
                return (x_vec / (kOneVec - x_vec)).log();  // 使用向量化计算 logit
              });
        } else {
          // 如果 eps 不小于 0，执行以下逻辑
          const scalar_t lo = eps;  // 设置下界为 eps
          const scalar_t hi = scalar_t(1) - eps;  // 设置上界为 1 - eps
          const Vectorized<scalar_t> kOneVec(scalar_t(1));
          const Vectorized<scalar_t> lo_vec(lo);  // 向量化 lo
          const Vectorized<scalar_t> hi_vec(hi);  // 向量化 hi
          // 使用 cpu_kernel_vec 处理迭代器，使用 lambda 表达式计算 clamped logit
          cpu_kernel_vec(
              iter,
              [lo, hi](scalar_t x) {
                // 对每个元素 x 进行 clamping 到 [lo, hi] 区间
                x = x < lo ? lo : (x > hi ? hi : x);
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()  // 如果 x 等于 1 返回无穷大
                    : std::log(x / (scalar_t(1) - x));  // 否则计算 clamped logit
              },
              [kOneVec, lo_vec, hi_vec](Vectorized<scalar_t> x_vec) {
                // 使用向量化计算 clamped logit
                x_vec = vec::clamp(x_vec, lo_vec, hi_vec);
                return (x_vec / (kOneVec - x_vec)).log();
              });
        }
      });
}

#if !defined(C10_MOBILE)

// 如果未定义 C10_MOBILE，则定义 _AT_DISPATCH_ABS_TYPES 宏，接受 TYPE, NAME 和可选参数
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)                                                 \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else

// 如果定义了 C10_MOBILE，则定义 _AT_DISPATCH_ABS_TYPES 宏，接受 TYPE, NAME 和可选参数
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)          \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(          \
            kHalf, kBFloat16,                            \
            TYPE, NAME, __VA_ARGS__)
#endif

// 定义 abs_kernel 函数，接受 TensorIteratorBase& 作为参数
static void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();  // 获取迭代器的数据类型
  // 如果数据类型为 kComplexHalf，则执行以下逻辑
  if (dtype == kComplexHalf) {
    using scalar_t = c10::complex<Half>;  // 定义 scalar_t 类型为 Half 的复数类型
    using opmath_t = at::opmath_type<scalar_t>;  // 使用 opmath_type 获取 scalar_t 对应的操作类型
    // 使用 cpu_kernel 处理迭代器，使用 lambda 表达式计算复数的绝对值
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t { return abs_impl(opmath_t{a}); });
  } else {
    // 否则，根据数据类型使用 _AT_DISPATCH_ABS_TYPES 处理迭代器
    _AT_DISPATCH_ABS_TYPES(iter.dtype(), "abs_cpu", [&]() {
      // 使用 cpu_kernel_vec 处理迭代器，使用 lambda 表达式计算绝对值
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return abs_impl(a); },
          [=](Vectorized<scalar_t> a) { return a.abs(); });  // 使用向量化计算绝对值
    });
  }
}
// 计算角度的核函数，处理浮点数和复数类型（包括 bfloat16 和 half）
static void angle_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "angle_cpu", [&]() {
    // 调用 CPU 内核处理向量化操作
    cpu_kernel_vec(
        iter,
        // 标量操作：返回标量的角度
        [=](scalar_t a) -> scalar_t { return angle_impl(a); },
        // 向量操作：返回向量的角度
        [=](Vectorized<scalar_t> a) { return a.angle(); });
  });
}

// 忽略张量上的负数位
void conj_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "conj_cpu",
    AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
      // 对于非复数类型，conj 操作无效，直接进行数据拷贝
      direct_copy_kernel(iter);
    })
    AT_DISPATCH_CASE_COMPLEX_TYPES_AND(kComplexHalf, [&] {
      // 调用 CPU 内核处理向量化操作
      cpu_kernel_vec(
          iter,
          // 标量操作：返回标量的共轭
          [=](scalar_t a) -> scalar_t { return conj_impl(a); },
          // 向量操作：返回向量的共轭
          [=](Vectorized<scalar_t> a) { return a.conj(); });
    })
  );
}

// 按位取反的核函数
static void bitwise_not_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // 布尔类型无法使用 ~（按位取反），bitwise_not 函数用于布尔和整数类型的按位取反操作
    cpu_kernel(
          iter,
          // 对布尔值取反
          [](bool a) {
            return !a;
          });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cpu", [&]() {
      // 调用 CPU 内核处理向量化操作
      cpu_kernel_vec(
          iter,
          // 标量操作：按位取反
          [](scalar_t a) -> scalar_t {
            return ~a;
          },
          // 向量操作：按位取反
          [](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
            return ~a;
          });
    });
  }
}

// 分数部分的核函数
static void frac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "frac_cpu", [&]() {
    // 调用 CPU 内核处理向量化操作
    cpu_kernel_vec(
        iter,
        // 标量操作：返回浮点数的小数部分
        [=](scalar_t a) -> scalar_t { return a - std::trunc(a); },
        // 向量操作：返回向量的小数部分
        [=](Vectorized<scalar_t> a) { return a.frac(); });
  });
}

// 逻辑非的核函数
static void logical_not_kernel(TensorIteratorBase& iter) {
  // 注意：此实现与 CUDA 实现不同，因为 CPU 内核不处理动态类型转换（见 needs_dynamic_casting）
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_cpu", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_cpu", [&]() {
      // 调用 CPU 内核处理逻辑非操作
      cpu_kernel(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

// 倒数的核函数
void reciprocal_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "reciprocal_cpu", [&]() {
    // 调用 CPU 内核处理向量化操作，忽略浮点数除以零的错误
    cpu_kernel_vec(
        iter,
        // 标量操作：返回标量的倒数
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t { return static_cast<scalar_t>(1.0) / a; },
        // 向量操作：返回向量的倒数
        [=](Vectorized<scalar_t> a) { return a.reciprocal(); });
  });
}
// 对输入的张量迭代器应用负数运算，支持所有数值类型和复数类型（包括复数的半精度、BFloat16和半精度）
void neg_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
    // 根据类型分发执行相应的 CPU 内核函数，对每个元素执行取负操作或者调用向量化函数取负
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -a; },  // 对每个标量执行取负操作
        [=](Vectorized<scalar_t> a) { return a.neg(); });  // 对向量化数据执行取负操作
  });
}

// 实现符号函数的内核，根据数据类型分发执行不同的 CPU 内核函数
static void sign_kernel(TensorIteratorBase& iter){
  if(iter.dtype() == ScalarType::Bool){
      cpu_kernel(iter, [=](bool x) -> bool { return x; });  // 对布尔类型直接返回原值
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.dtype(), "sign_cpu", [&]() {
        auto zero_vec = Vectorized<scalar_t>(static_cast<scalar_t>(0));
        auto one_vec = Vectorized<scalar_t>(static_cast<scalar_t>(1));

        cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return (0 < a) - c10::is_negative(a); },  // 计算每个标量的符号
          [=](Vectorized<scalar_t> self_vec){
              // 比较运算符返回位掩码
              auto left = Vectorized<scalar_t>::blendv(zero_vec, one_vec, zero_vec < self_vec);  // 左侧标量是否大于零的掩码
              auto right = Vectorized<scalar_t>::blendv(zero_vec, one_vec, self_vec < zero_vec);  // 右侧标量是否小于零的掩码

              return left - right;  // 计算符号函数值
          });
    });
  }
}

// 实现符号位函数的内核，根据输入数据类型选择不同的处理方式
static void signbit_kernel(TensorIteratorBase& iter){
  // 注意：signbit 函数不总是支持整数参数
  AT_DISPATCH_SWITCH(iter.input_dtype(), "signbit_cpu",
      AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
        cpu_kernel(iter, [](scalar_t a) -> bool { return c10::is_negative(a); });  // 对整数类型执行符号位判断
      })
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        cpu_kernel(iter, [](scalar_t a) -> bool { return std::signbit(opmath_t{a}); });  // 对浮点类型执行符号位判断
      })
    );
}

// 实现符号函数的内核，根据数据类型分发执行不同的 CPU 内核函数
static void sgn_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (dtype == kComplexHalf) {
    using scalar_t = c10::complex<Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    cpu_kernel(
        iter, [=](scalar_t a) -> scalar_t { return sgn_impl(opmath_t{a}); });  // 对半精度复数执行符号函数
  } else {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "sgn_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return sgn_impl(a); },  // 对每个复数执行符号函数
        [=](Vectorized<scalar_t> a) { return a.sgn(); });  // 对向量化数据执行符号函数
    });
  }
}

// 实现 sinc 函数的内核，处理浮点数和复数类型（包括 BFloat16 和半精度）
static void sinc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "sinc_cpu", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t {
          if (a == scalar_t(0)) {  // 若输入值为零
            return scalar_t(1);  // 返回 1
          } else {
            using opmath_t = at::opmath_type<scalar_t>;
            opmath_t product = c10::pi<opmath_t> * opmath_t{a};  // 计算输入值与 π 的乘积
            return static_cast<scalar_t>(std::sin(product) / product);  // 返回 sinc 函数值
          }
        });
  });
}

// 实现双曲正弦函数的内核，处理浮点数和复数类型（包括 BFloat16 和半精度）
static void sinh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "sinh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::sinh(a); },
        [=](Vectorized<scalar_t> self_vec){return self_vec.sinh();});


    // 调用cpu_kernel_vec函数，传入以下三个参数：
    // 1. iter：表示迭代器或迭代范围，可能用于指示操作的数据范围
    // 2. lambda函数 [=](scalar_t a) -> scalar_t { return std::sinh(a); }：
    //    用于处理标量a的双精度双精度双精度双精度双精度双精度双精度双精度数学函数，返回双精度
    // 3. lambda函数 [=](Vectorized<scalar_t> self_vec){return self_vec.sinh();}：
    //    用于处理Vectorized<scalar_t> self_vec的双精度数学函数, 返回双精度
表示
}

// 定义静态函数 cosh_kernel，处理张量迭代器 iter
static void cosh_kernel(TensorIteratorBase& iter) {
  // 根据数据类型 scalar_t 分发函数，处理浮点数和复数类型，以及额外的类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "cosh_cpu", [&]() {
    // 在 CPU 上执行向量化操作，对每个元素执行双目标函数
    cpu_kernel_vec(
        iter,
        // Lambda 函数，对单个元素应用双曲余弦函数 std::cosh
        [=](scalar_t a) -> scalar_t { return std::cosh(a); },
        // Lambda 函数，对向量化的元素应用向量化双曲余弦函数 self_vec.cosh()
        [=](Vectorized<scalar_t> self_vec){return self_vec.cosh();});
  });
}

// 定义静态函数 acosh_kernel，处理张量迭代器 iter
static void acosh_kernel(TensorIteratorBase& iter) {
    // 根据数据类型 scalar_t 分发函数，处理浮点数和复数类型，以及额外的类型 kBFloat16 和 kHalf
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "acosh_cpu", [&]() {
      // 在 CPU 上执行单目标函数，对每个元素执行反双曲余弦函数 std::acosh
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::acosh(a); });
    });
}

// 定义静态函数 asinh_kernel，处理张量迭代器 iter
static void asinh_kernel(TensorIteratorBase& iter) {
    // 根据数据类型 scalar_t 分发函数，处理浮点数和复数类型，以及额外的类型 kBFloat16 和 kHalf
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "asinh_cpu", [&]() {
      // 在 CPU 上执行单目标函数，对每个元素执行反双曲正弦函数 std::asinh
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::asinh(a); });
    });
}

// 定义静态函数 atanh_kernel，处理张量迭代器 iter
static void atanh_kernel(TensorIteratorBase& iter) {
    // 根据数据类型 scalar_t 分发函数，处理浮点数和复数类型，以及额外的类型 kBFloat16 和 kHalf
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "atanh_cpu", [&]() {
      // 在 CPU 上执行向量化操作，对每个元素执行单目标函数 std::atanh
      cpu_kernel_vec(
        iter,
        // Lambda 函数，对单个元素应用反双曲正切函数 std::atanh
        [=](scalar_t a) -> scalar_t { return std::atanh(a); },
        // Lambda 函数，对向量化的元素应用向量化反双曲正切函数 self_vec.atanh()
        [=](Vectorized<scalar_t> self_vec){return self_vec.atanh();});
    });
}

// 定义静态函数 digamma_kernel，处理张量迭代器 iter
static void digamma_kernel(TensorIteratorBase& iter) {
  // 根据数据类型 scalar_t 分发函数，处理浮点数类型，以及额外的类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "digamma", [&]() {
    // 在 CPU 上执行向量化操作，对每个元素执行单目标函数 calc_digamma
    cpu_kernel_vec(
        iter,
        // Lambda 函数，对单个元素应用 digamma 函数 calc_digamma
        [=](scalar_t a) -> scalar_t { return calc_digamma(a); },
        // Lambda 函数，对向量化的元素应用向量化 digamma 函数 x.digamma()
        [=](Vectorized<scalar_t> x) { return x.digamma(); });
  });
}

// 定义静态函数 trigamma_kernel，处理张量迭代器 iter
static void trigamma_kernel(TensorIteratorBase& iter) {
  // 根据数据类型 scalar_t 分发函数，处理浮点数类型，以及额外的类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "trigamma", [&]() {
    // 在 CPU 上执行单目标函数，对每个元素执行三角 gamma 函数 trigamma
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return trigamma(a); });
  });
}

// 定义静态函数 exp2_kernel，处理张量迭代器 iter
static void exp2_kernel(TensorIteratorBase& iter) {
  // 根据数据类型 scalar_t 分发函数，处理浮点数和复数类型，以及额外的类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "exp2", [&] {
    // 在 CPU 上执行向量化操作，对每个元素执行双目标函数 exp2_impl
    cpu_kernel_vec(
        iter,
        // Lambda 函数，对单个元素应用 exp2_impl 函数
        [](scalar_t a) -> scalar_t { return exp2_impl(a); },
        // Lambda 函数，对向量化的元素应用向量化 exp2 函数 a.exp2()
        [](Vectorized<scalar_t> a) { return a.exp2(); });
  });
}

// 定义静态函数 polygamma_kernel，处理张量迭代器 iter 和整数 n
static void polygamma_kernel(TensorIteratorBase& iter, int64_t n) {
  // 如果 n 等于 0，则调用 digamma_kernel 处理迭代器 iter
  if (n == 0) {
    digamma_kernel(iter);
  // 如果 n 等于 1，则调用 trigamma_kernel 处理迭代器 iter
  } else if (n == 1) {
    trigamma_kernel(iter);
  // 否则，根据数据类型 scalar_t 分发函数，处理浮点数类型，以及额外的类型 kBFloat16 和 kHalf
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "polygamma", [&]() {
      // 在 CPU 上执行单目标函数，对每个元素执行多项 gamma 函数 calc_polygamma
      cpu_kernel(
          iter, [=](scalar_t a) -> scalar_t { return calc_polygamma(a, n); });
    });
  }
}

// 模板函数，处理泛型 scalar_t 类型，替换 NaN 和无限值
template <typename scalar_t>
inline scalar_t _nan_to_num_replace(
    scalar_t a, scalar_t nan_replacement, scalar_t pos_inf_replacement, scalar_t neg_inf_replacement) {
  // 如果 a 是 NaN，则返回 nan_replacement
  if (at::_isnan(a)) {
    return nan_replacement;
  // 如果 a 是正无穷大，则返回 pos_inf_replacement
  } else if (a == std::numeric_limits<scalar_t>::infinity()) {
    return pos_inf_replacement;
  // 如果 a 是负无穷大，则返回 neg_inf_replacement
  } else if (a == -std::numeric_limits<scalar_t>::infinity()) {
    return neg_inf_replacement;
  // 否则，返回原始值 a
  } else {
    return a;
  }
}

// 模板函数，处理泛型 scalar_t 类型，替换复数值中的 NaN 和无限值
template <typename scalar_t>
inline c10::complex<scalar_t> _nan_to_num_replace(
    c10::complex<scalar_t> a, c10::complex<scalar_t> nan_replacement, c10::complex<scalar_t> pos_inf_replacement, c10::complex<scalar_t> neg_inf_replacement) {
  // 在复数情况下，分别替换实部和虚部
  return {
    _nan_to_num_replace(a.real(), nan_replacement.real(), pos_inf_replacement.real(), neg_inf_replacement.real()),
    _nan_to_num_replace(a.imag(), nan_replacement.imag(), pos_inf_replacement.imag(), neg_inf_replacement.imag())
  };
}
    // 定义一个名为c10::complex的复数对象a，同时定义三个标量参数nan、posinf、neginf
    // 返回一个新的复数对象，其实部和虚部分别通过_nan_to_num_replace函数处理得到
    return c10::complex<scalar_t>(
        // 调用_nan_to_num_replace函数，将a的实部、nan、posinf和neginf作为参数传入
        _nan_to_num_replace(a.real(), nan, posinf, neginf),
        // 调用_nan_to_num_replace函数，将a的虚部、nan、posinf和neginf作为参数传入
        _nan_to_num_replace(a.imag(), nan, posinf, neginf)
    );
}

template <typename scalar_t>
// 定义内联函数，用于将 NaN 替换为指定值，并根据条件将正无穷和负无穷替换为指定值
inline Vectorized<scalar_t> _nan_to_num_replace(
    Vectorized<scalar_t> a, scalar_t nan, scalar_t posinf, scalar_t neginf) {
  using vec_t = Vectorized<scalar_t>;
  // 创建一个无穷向量，其值为标量类型的无穷大
  vec_t inf(std::numeric_limits<scalar_t>::infinity());
  vec_t result;
  // 将 NaN 替换为指定值 nan
  result = vec_t::blendv(a, vec_t(nan), a.isnan());
  // 将正无穷替换为指定值 posinf
  result = vec_t::blendv(result, vec_t(posinf), a == inf);
  // 将负无穷替换为指定值 neginf
  return vec_t::blendv(result, vec_t(neginf), a == inf.neg());
}

template <typename scalar_t>
// 定义内联函数，用于复数类型，根据条件替换 NaN 和无穷值
inline Vectorized<c10::complex<scalar_t>> _nan_to_num_replace(
    Vectorized<c10::complex<scalar_t>> a, scalar_t nan, scalar_t posinf, scalar_t neginf) {
// 如果不是在 Microsoft Visual Studio 中，并且支持 AVX2 或 AVX512
#if !defined(_MSC_VER) && (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512))
  // 返回对标量类型转换为向量化类型后进行的 NaN 和无穷替换操作
  return {_nan_to_num_replace(Vectorized<scalar_t>(a), nan, posinf, neginf)};
#else
  // 在非 AVX2 或 AVX512 的环境中，创建复数类型的缓冲区
  __at_align__ c10::complex<scalar_t> buffer[a.size()];
  // 将向量化类型的数据存储到缓冲区
  a.store(buffer);
  // 加载缓冲区数据为实数向量
  auto asreal = Vectorized<scalar_t>::loadu(buffer);
  // 对实数向量执行 NaN 和无穷替换操作，并存储到缓冲区
  _nan_to_num_replace(asreal, nan, posinf, neginf).store(buffer);
  // 返回加载的复数向量
  return Vectorized<c10::complex<scalar_t>>::loadu(buffer);
#endif
}

static void nan_to_num_kernel(
    // 核函数，处理迭代器及 NaN、正无穷、负无穷的可选值
    TensorIteratorBase& iter,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "nan_to_num", [&]() {
    // 使用标量值类型进行类型别名
    using value_t = c10::scalar_value_type<scalar_t>::type;
    // 获取 NaN 替换值，如果未提供，默认为 0.0
    value_t nan_replacement = static_cast<value_t>(nan.value_or(0.));
    // 获取正无穷替换值，如果未提供，默认为标量类型的最大值
    value_t pos_inf_replacement = pos_inf.has_value()
        ? static_cast<value_t>(pos_inf.value())
        : std::numeric_limits<value_t>::max();
    // 获取负无穷替换值，如果未提供，默认为标量类型的最小值
    value_t neg_inf_replacement = neg_inf.has_value()
        ? static_cast<value_t>(neg_inf.value())
        : std::numeric_limits<value_t>::lowest();
    using vec_t = Vectorized<scalar_t>;

    // 对 CPU 中的向量执行核函数
    cpu_kernel_vec(iter, [=](scalar_t a) -> scalar_t {
      return _nan_to_num_replace(a, nan_replacement, pos_inf_replacement, neg_inf_replacement);
    }, [=](vec_t a) -> vec_t {
      return _nan_to_num_replace(a, nan_replacement, pos_inf_replacement, neg_inf_replacement);
    });
  });
}

static void kaiser_window_kernel(TensorIteratorBase& iter, int64_t window_length, double beta){
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "kaiser_window_cpu", [&](){
    // 定义计算函数 I0
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t alpha = static_cast<opmath_t>((window_length - 1) / 2.0);
    const opmath_t beta_ = static_cast<opmath_t>(beta);
    // 对 CPU 执行核函数，计算 Kaisor 窗口
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
        return calc_i0(beta_ * std::sqrt(std::abs(1 - std::pow((static_cast<opmath_t>(a) - alpha) / alpha, static_cast<opmath_t>(2.0))))) / calc_i0(beta_);
    });
  });
}

void rsqrt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "rsqrt_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          // lambda 函数，计算浮点数 a 的平方根的倒数，忽略除零错误
          return (static_cast<scalar_t>(1)) / std::sqrt(a);
        },
        [=](Vectorized<scalar_t> a) { return a.rsqrt(); });
  });
}

static void entr_kernel(TensorIteratorBase& iter) {
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和特定类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "entr_cpu", [&] {
        // 调用 cpu_kernel 函数，对迭代器进行操作
        cpu_kernel(iter, [](scalar_t x) -> scalar_t {
          // 检查 x 是否为 NaN
          if (at::_isnan(x)) {
            return x;  // 若是 NaN，则返回 x
          } else if (x > 0) {
            // 若 x 大于 0，则返回 -x * log(x)
            return -x * std::log(x);
          } else if (x == 0) {
            // 若 x 等于 0，则返回 0（静态类型转换为 scalar_t）
            return static_cast<scalar_t>(0);
          }
          // 若 x 小于等于 0，返回 -INFINITY（静态类型转换为 scalar_t）
          return static_cast<scalar_t>(-INFINITY);
        });
      });
}

static void frexp_kernel(TensorIteratorBase& iter) {
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和特定类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
    // iter.dtype() 表示输出的尾数的数据类型
    // 必须是浮点类型，并且与输入的 dtype 相同
    iter.dtype(),
    "frexp_cpu", [&]() {
      // 调用 cpu_kernel_multiple_outputs 函数，对迭代器进行操作
      cpu_kernel_multiple_outputs(
        iter,
        // 对每个输入值 a 进行 frexp 操作，返回尾数 mantissa 和指数 exponent
        [](scalar_t a) -> std::tuple<scalar_t, int32_t> {
          int32_t exponent;
          scalar_t mantissa = std::frexp(a, &exponent);
          return std::tuple<scalar_t, int32_t>(mantissa, exponent);
        }
      );
  });
}

static void ndtri_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中张量的数量是否为 2
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，处理浮点类型
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ndtri_cpu", [&]() {
        // 调用 cpu_kernel 函数，对迭代器进行操作，使用 calc_ndtri 函数处理每个 x
        cpu_kernel(iter, [](scalar_t x) { return calc_ndtri(x); });
      });
}

static void log_ndtr_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中张量的数量是否为 2
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，处理浮点类型
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "log_ndtr_cpu", [&]() {
        // 调用 cpu_kernel 函数，对迭代器进行操作，使用 calc_log_ndtr 函数处理每个 x
        cpu_kernel(iter, [](scalar_t x) { return calc_log_ndtr(x); });
      });
}

static void i0e_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中张量的数量是否为 2
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  // 使用 AT_DISPATCH_FLOATING_TYPES_AND2 宏，处理浮点类型和特定类型 kBFloat16 和 kHalf
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "i0e_cpu", [&]() {
        // 调用 cpu_kernel_vec 函数，对迭代器进行操作
        cpu_kernel_vec(
            iter,
            // 对每个输入值 x 进行 calc_i0e 函数处理
            [](scalar_t x) { return calc_i0e(x); },
            // 对每个向量化的输入值 x 进行 x.i0e() 处理
            [](Vectorized<scalar_t> x) { return x.i0e(); });
      });
}

static void i1_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中张量的数量是否为 2
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，处理浮点类型
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cpu", [&]() {
    // 调用 cpu_kernel 函数，对迭代器进行操作，使用 calc_i1 函数处理每个 x
    cpu_kernel(iter, [](scalar_t x) { return calc_i1(x); });
  });
}

static void i1e_kernel(TensorIteratorBase& iter) {
  // 检查迭代器中张量的数量是否为 2
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，处理浮点类型
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cpu", [&]() {
    // 调用 cpu_kernel 函数，对迭代器进行操作，使用 calc_i1e 函数处理每个 x
    cpu_kernel(iter, [](scalar_t x) { return calc_i1e(x); });
  });
}

static void erfcx_kernel(TensorIteratorBase& iter){
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏，处理浮点类型
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "erfcx_cpu", [&]() {
    // 调用 cpu_kernel 函数，对迭代器进行操作，使用 calc_erfcx 函数处理每个 a
    cpu_kernel(
      iter,
      [](scalar_t a) -> scalar_t { return calc_erfcx(a); });
  });
}
// 定义一个静态函数，用于将浮点数张量迭代器中的每个元素四舍五入到指定小数位数
static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  // 根据张量的数据类型分发函数，这里包括浮点数类型和一些特定类型（kBFloat16和kHalf），命名为"round_cpu"
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "round_cpu", [&]() {
        // 定义用于数学操作的类型
        using opmath_t = at::opmath_type<scalar_t>;
        // 标志是否为负数的标志位
        bool neg_flag = false;
        // 用于存储10的decimals次幂的操作数
        opmath_t ten_pow_decimals;
        // 如果decimals为负数，将其转换为正数并设置标志位
        if (decimals < 0) {
          decimals = -decimals;
          neg_flag = true;
        }
        // 计算10的decimals次幂并转换为opmath_t类型
        ten_pow_decimals = static_cast<opmath_t>(std::pow(10, decimals));
        // 调用CPU内核函数，对每个元素执行四舍五入操作
        cpu_kernel(iter, [ten_pow_decimals, neg_flag](scalar_t a) -> scalar_t {
          // 如果neg_flag为真，则将a除以ten_pow_decimals进行近似取整后再乘以ten_pow_decimals返回结果，
          // 否则将a乘以ten_pow_decimals进行近似取整后再除以ten_pow_decimals返回结果
          return neg_flag ? std::nearbyint(static_cast<opmath_t>(a) / ten_pow_decimals) * ten_pow_decimals
                          : std::nearbyint(static_cast<opmath_t>(a) * ten_pow_decimals) / ten_pow_decimals;
        });
      });
}

// 定义一个静态函数，用于计算第一类贝塞尔函数 J0 的内核操作
static void bessel_j0_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"bessel_j0_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行第一类贝塞尔函数 J0 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_j0_forward(x);
        });
    });
} // bessel_j0_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算第一类贝塞尔函数 J1 的内核操作
static void bessel_j1_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"bessel_j1_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j1_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行第一类贝塞尔函数 J1 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_j1_forward(x);
        });
    });
} // bessel_j1_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算第二类贝塞尔函数 Y0 的内核操作
static void bessel_y0_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"bessel_y0_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y0_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行第二类贝塞尔函数 Y0 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_y0_forward(x);
        });
    });
} // bessel_y0_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算第二类贝塞尔函数 Y1 的内核操作
static void bessel_y1_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"bessel_y1_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y1_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行第二类贝塞尔函数 Y1 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_y1_forward(x);
        });
    });
} // bessel_y1_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算修正第一类贝塞尔函数 I0 的内核操作
static void modified_bessel_i0_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"modified_bessel_i0_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i0_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行修正第一类贝塞尔函数 I0 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_i0_forward(x);
        });
    });
} // modified_bessel_i0_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算修正第一类贝塞尔函数 I1 的内核操作
static void modified_bessel_i1_kernel(TensorIteratorBase& iterator) {
    // 断言张量迭代器中张量的数量为2
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据张量的数据类型分发函数，命名为"modified_bessel_i1_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i1_cpu", [&]() {
        // 调用CPU内核函数，对每个元素执行修正第一类贝塞尔函数 I1 的计算
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_i1_forward(x);
        });
    });
} // modified_bessel_i1_kernel(TensorIteratorBase& iterator)
// 定义一个静态函数，用于计算修正贝塞尔函数 K0 的核心操作。这个函数接受一个迭代器对象作为参数。
static void modified_bessel_k0_kernel(TensorIteratorBase& iterator) {
    // 断言迭代器对象中张量的数量为2，确保输入参数正确。
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据迭代器中张量的数据类型，分发到相应的浮点数类型上，并命名为 "modified_bessel_k0_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k0_cpu", [&]() {
        // 在CPU上执行核函数，对每个标量进行 modified_bessel_k0_forward 函数的计算，并返回结果。
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_k0_forward(x);
        });
    });
} // modified_bessel_k0_kernel(TensorIteratorBase& iterator)

// 定义一个静态函数，用于计算修正贝塞尔函数 K1 的核心操作。这个函数接受一个迭代器对象作为参数。
static void modified_bessel_k1_kernel(TensorIteratorBase& iterator) {
    // 断言迭代器对象中张量的数量为2，确保输入参数正确。
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    // 根据迭代器中张量的数据类型，分发到相应的浮点数类型上，并命名为 "modified_bessel_k1_cpu"
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cpu", [&]() {
        // 在CPU上执行核函数，对每个标量进行 modified_bessel_k1_forward 函数的计算，并返回结果。
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_k1_forward(x);
        });
    });
} // modified_bessel_k1_kernel(TensorIteratorBase& iterator)

// TODO: Disable cont. branch to test more risky code
// TODO：禁用持续分支以测试更多风险的代码
#define IMPLEMENT_ITERATOR_LAMBDA(op)                                              \
          [&](char** data_, const int64_t* strides, int64_t n) {                   \
            // 将输出数据指针转换为目标类型的指针
            scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);            \
            // 将输入数据指针转换为目标类型的指针
            scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);             \
            // 计算输出数据的步长
            int64_t out_stride = strides[0] / sizeof(scalar_t);                    \
            // 计算输入数据的步长
            int64_t in_stride = strides[1] / sizeof(scalar_t);                     \
            // 如果输出和输入数据都是连续存储的
            if (out_stride == 1 && in_stride == 1) {                               \
              // 调用矢量数学库中的操作函数
              vml::v##op(out_data, in_data, n);                                    \
              return;                                                              \
            }                                                                      \
            // 设定缓冲区的宽度，确保能够进行矢量化计算
            static constexpr int64_t WIDTH = (8*1024) / sizeof(scalar_t);          \
            // 对数据进行分块处理，每块的大小为WIDTH
            for (int64_t i = 0; i < n; i += WIDTH) {                               \
              // 声明一个缓冲区
              scalar_t buffer[WIDTH];                                              \
              // 计算当前块的实际宽度，不超过剩余数据的长度
              const int64_t width = std::min(WIDTH, n - i);                        \
              /* 如果任一张量是连续的，则直接使用它，否则复制到一个连续缓冲区中 */ \
              scalar_t * in_buffer = in_stride == 1 ? &in_data[i] : &buffer[0];    \
              scalar_t * out_buffer = out_stride == 1 ? &out_data[i] : &buffer[0]; \
              // 如果输入数据不是连续的，则将数据复制到缓冲区中
              if (in_stride != 1)                                                  \
                for (const auto j : c10::irange(width))                            \
                  in_buffer[j] = in_data[in_stride * (i + j)];                     \
              // 调用矢量数学库中的操作函数处理缓冲区数据
              vml::v##op(out_buffer, in_buffer, width);                            \
              // 如果输出数据不是连续的，则将计算结果复制回输出数据中
              if (out_stride != 1)                                                 \
                for (const auto j : c10::irange(width))                            \
                    out_data[out_stride * (i + j)] = out_buffer[j];                \
            }                                                                      \
          }

#define IMPLEMENT_FLOAT_KERNEL(op)                                                  \
  inline namespace CPU_CAPABILITY {                                                 \
  // 实现特定浮点类型的操作核心函数，使用迭代器处理数据
  static void op##_kernel(TensorIteratorBase& iter) {                               \
    // 断言迭代器中张量数量为2
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                    \
    // 根据数据类型调度操作函数，处理浮点数类型和kBFloat16、kHalf类型
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
      // 设定处理的粒度大小
      constexpr int64_t grain_size = 2048;                                          \
      // 对迭代器中的每个元素应用实现迭代器Lambda函数，以指定的粒度处理
      iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                     \
    });                                                                             \
    iter.cast_outputs();                                                            \
  }                                                                                 \
  }


注释：


    // 调用iter对象的cast_outputs方法，可能用于转换迭代器的输出类型或执行其他操作
    iter.cast_outputs();
    // 结束当前的循环或函数体，对应于前面的条件或循环结构的结束
  }                                                                                 \
  // 结束当前的函数定义或其他代码块
  }
#define IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(op)                                   \
  // 定义实现不使用 AVX512 指令集的浮点数操作内核，调用宏 IMPLEMENT_FLOAT_KERNEL(op) \
  IMPLEMENT_FLOAT_KERNEL(op)                                                        \
  // 使用宏 REGISTER_DISPATCH 注册 op##_stub，并绑定到 CPU_CAPABILITY 结构体的 op##_kernel 函数

#define IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(op)                                      \
  // 定义实现使用 AVX512 指令集的浮点数操作内核，调用宏 IMPLEMENT_FLOAT_KERNEL(op) \
  IMPLEMENT_FLOAT_KERNEL(op)                                                        \
  // 使用宏 ALSO_REGISTER_AVX512_DISPATCH 注册 op##_stub，并绑定到 CPU_CAPABILITY 结构体的 op##_kernel 函数

#define IMPLEMENT_COMPLEX_KERNEL(op)                                                             \
  // 在 CPU_CAPABILITY 命名空间中定义复杂数操作 op 的内核函数 \
  inline namespace CPU_CAPABILITY {                                                              \
  // 实现 op##_kernel 函数，接收 TensorIteratorBase 类的迭代器参数 \
  void op##_kernel(TensorIteratorBase& iter) {                                                   \
    // 断言迭代器中张量数量为 2 \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 宏根据迭代器的数据类型和 op 的名称分发函数调用 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
        constexpr int64_t grain_size = 2048;                                                     \
        // 对迭代器执行 for_each 操作，使用 op 的实现迭代器 lambda 函数，并指定 grain_size \
        iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                                \
    });                                                                                          \
    // 将迭代器的输出转换为合适的数据类型 \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  }

#define IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(op)                            \
  // 定义实现不使用 AVX512 指令集的复杂数操作内核，调用宏 IMPLEMENT_COMPLEX_KERNEL(op) \
  IMPLEMENT_COMPLEX_KERNEL(op)                                                 \
  // 使用宏 REGISTER_DISPATCH 注册 op##_stub，并绑定到 CPU_CAPABILITY 结构体的 op##_kernel 函数

#define IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(op)                               \
  // 定义实现使用 AVX512 指令集的复杂数操作内核，调用宏 IMPLEMENT_COMPLEX_KERNEL(op) \
  IMPLEMENT_COMPLEX_KERNEL(op)                                                 \
  // 使用宏 ALSO_REGISTER_AVX512_DISPATCH 注册 op##_stub，并绑定到 CPU_CAPABILITY 结构体的 op##_kernel 函数

#define STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                                      \
  // 在 CPU_CAPABILITY 命名空间中定义静态复杂数操作 op 的内核函数 \
  inline namespace CPU_CAPABILITY {                                                              \
  // 定义静态的 op##_kernel 函数，接收 TensorIteratorBase 类的迭代器参数 \
  static void op##_kernel(TensorIteratorBase& iter) {                                            \
    // 断言迭代器中张量数量为 2 \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 宏根据迭代器的数据类型和 op 的名称分发函数调用 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
        constexpr int64_t grain_size = 2048;                                                     \
        // 对迭代器执行 for_each 操作，使用 op 的实现迭代器 lambda 函数，并指定 grain_size \
        iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                                \
    });                                                                                          \
    // 将迭代器的输出转换为合适的数据类型 \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  }
# 定义了一个宏，用于实现没有 AVX512 支持的复杂内核操作，并注册调度函数
#define STATIC_IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(op)                     \
  STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                          \
  REGISTER_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

# 定义了一个宏，用于实现有 AVX512 支持的复杂内核操作，并注册调度函数
#define STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(op)                        \
  STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                          \
  ALSO_REGISTER_AVX512_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

} // CPU_CAPABILITY namespace

# 下面的内核函数在使用 AVX512 时速度较慢，因此没有注册 AVX512 调度
REGISTER_DISPATCH(round_decimals_stub, &CPU_CAPABILITY::round_decimals_kernel);
REGISTER_DISPATCH(abs_stub, &CPU_CAPABILITY::abs_kernel);
REGISTER_DISPATCH(angle_stub, &CPU_CAPABILITY::angle_kernel);
REGISTER_DISPATCH(neg_stub, &CPU_CAPABILITY::neg_kernel);
REGISTER_DISPATCH(signbit_stub, &CPU_CAPABILITY::signbit_kernel);
REGISTER_DISPATCH(sinc_stub, &CPU_CAPABILITY::sinc_kernel);
REGISTER_DISPATCH(bitwise_not_stub, &CPU_CAPABILITY::bitwise_not_kernel);
REGISTER_DISPATCH(logical_not_stub, &CPU_CAPABILITY::logical_not_kernel);
REGISTER_DISPATCH(nan_to_num_stub, &CPU_CAPABILITY::nan_to_num_kernel);
REGISTER_DISPATCH(conj_physical_stub, &CPU_CAPABILITY::conj_kernel);
REGISTER_DISPATCH(rsqrt_stub, &CPU_CAPABILITY::rsqrt_kernel);
REGISTER_DISPATCH(frac_stub, &CPU_CAPABILITY::frac_kernel);
REGISTER_DISPATCH(special_entr_stub, &CPU_CAPABILITY::entr_kernel);
REGISTER_DISPATCH(special_i0e_stub, &CPU_CAPABILITY::i0e_kernel);
REGISTER_DISPATCH(special_ndtri_stub, &CPU_CAPABILITY::ndtri_kernel);
REGISTER_DISPATCH(special_modified_bessel_k0_stub, &CPU_CAPABILITY::modified_bessel_k0_kernel);
REGISTER_DISPATCH(special_modified_bessel_k1_stub, &CPU_CAPABILITY::modified_bessel_k1_kernel);

# 宏实现了没有 AVX512 支持的浮点内核操作（以 ceil 为例）
IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(ceil);
IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(floor);
IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(round);

# 宏实现了没有 AVX512 支持的复杂数内核操作（以 sqrt 为例）
IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(sqrt);

# 宏实现了没有 AVX512 支持的浮点内核操作（以 trunc 为例）
IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(trunc);
IMPLEMENT_FLOAT_KERNEL_WITHOUT_AVX512(i0);

# 宏实现了没有 AVX512 支持的复杂数内核操作（以 sin, cos, tan 为例）
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(sin);
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(cos);
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITHOUT_AVX512(tan);

# 下面的内核函数是计算密集型的，并且已经同时使用了 AVX512 和 AVX2 进行编译
ALSO_REGISTER_AVX512_DISPATCH(sign_stub, &CPU_CAPABILITY::sign_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sgn_stub, &CPU_CAPABILITY::sgn_kernel);
ALSO_REGISTER_AVX512_DISPATCH(reciprocal_stub, &CPU_CAPABILITY::reciprocal_kernel);
ALSO_REGISTER_AVX512_DISPATCH(exp2_stub, &CPU_CAPABILITY::exp2_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_stub, &CPU_CAPABILITY::sigmoid_kernel);
ALSO_REGISTER_AVX512_DISPATCH(logit_stub, &CPU_CAPABILITY::logit_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sinh_stub, &CPU_CAPABILITY::sinh_kernel);
ALSO_REGISTER_AVX512_DISPATCH(cosh_stub, &CPU_CAPABILITY::cosh_kernel);
ALSO_REGISTER_AVX512_DISPATCH(atanh_stub, &CPU_CAPABILITY::atanh_kernel);

# 可能在为这些函数启用显式向量化后启用 AVX512 调度
// 注册 acosh_stub 函数到 CPU_CAPABILITY 的 acosh_kernel 分发器
REGISTER_DISPATCH(acosh_stub, &CPU_CAPABILITY::acosh_kernel);
// 注册 asinh_stub 函数到 CPU_CAPABILITY 的 asinh_kernel 分发器
REGISTER_DISPATCH(asinh_stub, &CPU_CAPABILITY::asinh_kernel);
// 注册 digamma_stub 函数到 CPU_CAPABILITY 的 digamma_kernel 分发器
REGISTER_DISPATCH(digamma_stub, &CPU_CAPABILITY::digamma_kernel);
// 注册 trigamma_stub 函数到 CPU_CAPABILITY 的 trigamma_kernel 分发器
REGISTER_DISPATCH(trigamma_stub, &CPU_CAPABILITY::trigamma_kernel);
// 注册 polygamma_stub 函数到 CPU_CAPABILITY 的 polygamma_kernel 分发器
REGISTER_DISPATCH(polygamma_stub, &CPU_CAPABILITY::polygamma_kernel);
// 注册 kaiser_window_stub 函数到 CPU_CAPABILITY 的 kaiser_window_kernel 分发器
REGISTER_DISPATCH(kaiser_window_stub, &CPU_CAPABILITY::kaiser_window_kernel);
// 注册 frexp_stub 函数到 CPU_CAPABILITY 的 frexp_kernel 分发器
REGISTER_DISPATCH(frexp_stub, &CPU_CAPABILITY::frexp_kernel);
// 注册 special_log_ndtr_stub 函数到 CPU_CAPABILITY 的 log_ndtr_kernel 分发器
REGISTER_DISPATCH(special_log_ndtr_stub, &CPU_CAPABILITY::log_ndtr_kernel);
// 注册 special_i1_stub 函数到 CPU_CAPABILITY 的 i1_kernel 分发器
REGISTER_DISPATCH(special_i1_stub, &CPU_CAPABILITY::i1_kernel);
// 注册 special_i1e_stub 函数到 CPU_CAPABILITY 的 i1e_kernel 分发器
REGISTER_DISPATCH(special_i1e_stub, &CPU_CAPABILITY::i1e_kernel);
// 注册 special_erfcx_stub 函数到 CPU_CAPABILITY 的 erfcx_kernel 分发器
REGISTER_DISPATCH(special_erfcx_stub, &CPU_CAPABILITY::erfcx_kernel);
// 注册 special_bessel_j0_stub 函数到 CPU_CAPABILITY 的 bessel_j0_kernel 分发器
REGISTER_DISPATCH(special_bessel_j0_stub, &CPU_CAPABILITY::bessel_j0_kernel);
// 注册 special_bessel_j1_stub 函数到 CPU_CAPABILITY 的 bessel_j1_kernel 分发器
REGISTER_DISPATCH(special_bessel_j1_stub, &CPU_CAPABILITY::bessel_j1_kernel);
// 注册 special_bessel_y0_stub 函数到 CPU_CAPABILITY 的 bessel_y0_kernel 分发器
REGISTER_DISPATCH(special_bessel_y0_stub, &CPU_CAPABILITY::bessel_y0_kernel);
// 注册 special_bessel_y1_stub 函数到 CPU_CAPABILITY 的 bessel_y1_kernel 分发器
REGISTER_DISPATCH(special_bessel_y1_stub, &CPU_CAPABILITY::bessel_y1_kernel);
// 注册 special_modified_bessel_i0_stub 函数到 CPU_CAPABILITY 的 modified_bessel_i0_kernel 分发器
REGISTER_DISPATCH(special_modified_bessel_i0_stub, &CPU_CAPABILITY::modified_bessel_i0_kernel);
// 注册 special_modified_bessel_i1_stub 函数到 CPU_CAPABILITY 的 modified_bessel_i1_kernel 分发器
REGISTER_DISPATCH(special_modified_bessel_i1_stub, &CPU_CAPABILITY::modified_bessel_i1_kernel);

// 使用 AVX512 实现静态复杂数函数 acos 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(acos);
// 使用 AVX512 实现静态复杂数函数 asin 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(asin);
// 使用 AVX512 实现静态复杂数函数 atan 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(atan);
// 使用 AVX512 实现浮点数函数 erf 的内核
IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(erf);
// 使用 AVX512 实现浮点数函数 erfc 的内核
IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(erfc);
// 使用 AVX512 实现浮点数函数 erfinv 的内核
IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(erfinv);
// 使用 AVX512 实现静态复杂数函数 exp 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(exp);
// 使用 AVX512 实现静态复杂数函数 expm1 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(expm1);
// 使用 AVX512 实现静态复杂数函数 log 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log);
// 使用 AVX512 实现静态复杂数函数 log10 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log10);
// 使用 AVX512 实现静态复杂数函数 log1p 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log1p);
// 使用 AVX512 实现静态复杂数函数 log2 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log2);
// 使用 AVX512 实现静态复杂数函数 tanh 的内核
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(tanh);
// 使用 AVX512 实现浮点数函数 lgamma 的内核
IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(lgamma);

} // namespace at::native
```