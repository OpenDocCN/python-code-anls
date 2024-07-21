# `.\pytorch\aten\src\ATen\native\SharedReduceOps.h`

```py
#pragma once
// 这个指令确保头文件只被包含一次，避免重复定义

// Please note that this file is
// used across both CPU and GPU.
// 请注意，这个文件被同时用于 CPU 和 GPU 环境

#include <type_traits>
// 包含类型特性的标准库头文件
#include <complex>
// 包含复数运算的标准库头文件
#include <c10/macros/Macros.h>
// 包含 C10 库中的宏定义
#include <ATen/detail/FunctionTraits.h>
// 包含 ATen 库中的函数特性细节
#include <ATen/NumericUtils.h>
// 包含 ATen 库中的数值计算工具函数
#if defined(__CUDACC__)
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
// 如果是 CUDA 编译器，包含 CUDA 设备工具和平方根计算的头文件
#elif defined(__HIPCC__)
#include <ATen/hip/DeviceUtils.cuh>
#include <ATen/native/hip/DeviceSqrt.cuh>
// 如果是 HIP 编译器，包含 HIP 设备工具和平方根计算的头文件
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/pair.h>
// 如果是 CUDA 或 HIP 编译器，包含 Thrust 库的 pair 头文件
#else
#include <cmath>
#define device_sqrt std::sqrt
// 如果不是 CUDA 或 HIP 编译器，定义 device_sqrt 为标准库的 sqrt 函数
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename scalar_t>
inline C10_DEVICE scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  // 在 CUDA 或 HIP 编译器下，定义处理 NaN 传播的最大值函数
#if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
  scalar_t max = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::max(a, b));
#else
  scalar_t max = at::_isnan(b) ? b : std::max(a, b);
#endif
  return max;
}
template <typename scalar_t>
inline C10_DEVICE scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  // 在 CUDA 或 HIP 编译器下，定义处理 NaN 传播的最小值函数
#if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
  scalar_t min = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::min(a, b));
#else
  scalar_t min = at::_isnan(b) ? b : std::min(a, b);
#endif
  return min;
}
#define MAX(X, Y) max_propagate_nan(X,Y)
#define MIN(X, Y) min_propagate_nan(X,Y)
// 如果是 CUDA 或 HIP 编译器，定义 MAX 和 MIN 宏分别为处理 NaN 传播的最大和最小值函数
#else
#include <ATen/native/cpu/zmath.h>
#define MAX(X, Y) max_impl(X,Y)
#define MIN(X, Y) min_impl(X,Y)
// 如果不是 CUDA 或 HIP 编译器，定义 MAX 和 MIN 宏分别为 CPU 下的最大和最小值函数
#endif

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_pow c10::cuda::compat::pow
// 如果是 CUDA 架构，使用 C10 库中 CUDA 兼容的 pow 函数
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define compat_pow c10::hip::compat::pow
// 如果是 HIP 架构，使用 C10 库中 HIP 兼容的 pow 函数
#else
#define compat_pow std::pow
// 如果是其他架构，使用标准库的 pow 函数
#endif

namespace at { namespace native {

namespace detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T1, typename T2> using pair = thrust::pair<T1, T2>;
// 如果是 CUDA 或 HIP 编译器，使用 Thrust 库中的 pair 类型
#else
template <typename T1, typename T2> using pair = std::pair<T1, T2>;
// 如果不是 CUDA 或 HIP 编译器，使用标准库中的 pair 类型
#endif

} // namespace detail

template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}
  // 定义 WelfordData 结构体的构造函数，初始化成员变量为零

  C10_HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
  // 定义 WelfordData 结构体的构造函数，初始化成员变量为给定值
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOps {
  acc_scalar_t correction;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // 定义 WelfordOps 结构体的 reduce 成员函数，用于计算 WelfordData 结构体的累积值
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // 需要 nf 以在 combine 中使用，因为 int32 可能会溢出。
    index_t new_n = acc.n + 1;  // 计算新的数据点数量
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);  // 将新的数据点数量转换为累积标量类型
    acc_scalar_t delta = data - acc.mean;  // 计算当前数据点与平均值的差
    acc_scalar_t new_mean = acc.mean + delta / new_nf;  // 计算更新后的平均值
    acc_scalar_t new_delta = data - new_mean;  // 计算新的数据点与更新后的平均值的差
    return {
      new_mean,  // 返回更新后的平均值
      acc.m2 + delta * new_delta,  // 返回更新后的 m2（方差的第二累积量）
      new_n,  // 返回更新后的数据点数量
      new_nf,  // 返回更新后的数据点数量转换后的标量类型
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;  // 如果 a 的数据点数量为 0，则返回 b
    }
    if (b.nf == 0) {
      return a;  // 如果 b 的数据点数量为 0，则返回 a
    }
    acc_scalar_t delta = b.mean - a.mean;  // 计算两个累积对象的平均值之差
    acc_scalar_t new_count = a.nf + b.nf;  // 计算合并后的数据点总数
    acc_scalar_t nb_over_n = b.nf / new_count;  // 计算 b 中数据点数量与总数的比例
    return {
      a.mean + delta * nb_over_n,  // 返回合并后的平均值
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,  // 返回合并后的 m2（方差的第二累积量）
      // 将 acc.n 设置为 -1，因为 acc.n 可能无法正确表示计数在其范围内，
      // 将其设置为 -1 可以避免混淆
      -1,
      new_count  // 返回合并后的数据点总数
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);  // 将累积对象的平均值转换为指定类型的标量
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;  // 计算方差的除数，根据是否大于校正值进行调整
    const auto var = acc.m2 / divisor;  // 计算方差
    res_t results(take_sqrt ? device_sqrt(var) : var, mean);  // 构造结果对象，可能包含方差的平方根（如果需要）
    return results;  // 返回结果对象
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;  // 返回原始的累积对象，不进行任何操作
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果编译器为 CUDA 或 HIP，则定义以下函数为内联设备函数，实现对向下扩散的数据传递
inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)  // 使用 warp shuffle 指令将均值向下偏移
      , WARP_SHFL_DOWN(acc.m2, offset)  // 使用 warp shuffle 指令将 m2 向下偏移
      , WARP_SHFL_DOWN(acc.n, offset)   // 使用 warp shuffle 指令将 n 向下偏移
      , WARP_SHFL_DOWN(acc.nf, offset)  // 使用 warp shuffle 指令将 nf 向下偏移
    };
}
#endif

// 以下是一个模板结构 MeanOps，用于实现均值操作
template <typename scalar_t, typename acc_t=scalar_t, typename factor_t=acc_t, typename out_t = acc_t>
struct MeanOps {
  factor_t factor;  // 定义因子 factor

  // 在设备上执行的函数，将输入数据 a 和标量 b 进行合并操作
  inline C10_DEVICE acc_t reduce(acc_t a, scalar_t b, int64_t /*idx*/) const {
    return combine(a, static_cast<acc_t>(b));  // 调用 combine 函数进行合并
  }

  // 在设备上执行的函数，将两个 acc_t 类型的数据 a 和 b 进行加法操作
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;  // 返回 a 和 b 的和
  }

  // 在设备上执行的函数，将 acc_t 类型的数据 a 乘以因子 factor 后返回
  inline C10_DEVICE out_t project(acc_t a) const {
    return a * factor;  // 返回 a 乘以 factor 的结果
  }

  // 静态函数，将 acc_t 类型的数据 acc 和 base_idx 直接返回
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;  // 直接返回 acc
}

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译器为 CUDA 或 HIP，则定义以下函数为内联设备函数，实现对向下扩散的数据传递
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);  // 使用 warp shuffle 指令将 data 向下偏移
  }
#endif

  // 构造函数，初始化 MeanOps 结构体的因子 factor
  MeanOps(factor_t factor): factor(factor) {
  }
};

// This accumulator template is used to calculate the minimum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
// 以下是一个模板结构 AbsMinOps，用于计算一组数的最小绝对值
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct AbsMinOps {

  // 在设备上执行的函数，将 acc_t 类型的数据 acc 和 scalar_t 类型的数据 data 以及索引 idx 进行最小绝对值的计算
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MIN(acc, static_cast<acc_t>(std::abs(data)));  // 返回 acc 和 data 的绝对值的最小值
  }

  // 在设备上执行的函数，将两个 acc_t 类型的数据 a 和 b 进行最小值的计算
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MIN(a, b);  // 返回 a 和 b 的最小值
  }

  // 在设备上执行的函数，将 acc_t 类型的数据 a 直接返回
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;  // 直接返回 a
  }

  // 静态函数，将 acc_t 类型的数据 acc 和 base_idx 直接返回
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;  // 直接返回 acc
}

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译器为 CUDA 或 HIP，则定义以下函数为内联设备函数，实现对向下扩散的数据传递
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);  // 使用 warp shuffle 指令将 acc 向下偏移
  }
#endif
};

// This accumulator template is used to calculate the maximum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
// 以下是一个模板结构 AbsMaxOps，用于计算一组数的最大绝对值
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct AbsMaxOps {
  // 在设备上执行的函数，将 acc_t 类型的数据 acc 和 scalar_t 类型的数据 data 以及索引 idx 进行最大绝对值的计算
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MAX(acc, static_cast<acc_t>(std::abs(data)));  // 返回 acc 和 data 的绝对值的最大值
  }

  // 在设备上执行的函数，将两个 acc_t 类型的数据 a 和 b 进行最大值的计算
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MAX(a, b);  // 返回 a 和 b 的最大值
  }

  // 在设备上执行的函数，将 acc_t 类型的数据 a 直接返回
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;  // 直接返回 a
  }

  // 静态函数，将 acc_t 类型的数据 acc 和 base_idx 直接返回
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;  // 直接返回 acc
}

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 如果编译器为 CUDA 或 HIP，则定义以下函数为内联设备函数，实现对向下扩散的数据传递
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);  // 使用 warp shuffle 指令将 acc 向下偏移
  }
#endif
};
    // 使用 WARP_SHFL_DOWN 函数将 acc 变量的值向下位移 offset 位后返回
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// This accumulator template is used to calculate the norm of the absolute value
// of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormOps {
  acc_t norm_;

  // Reduce function accumulates the norm of the absolute value of `data`.
  // `acc` is the current accumulated value, `data` is the current data element,
  // and `idx` is the index (not used here).
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + compat_pow(static_cast<acc_t>(std::abs(data)), norm_);
  }

  // Combine function combines two accumulated values `a` and `b`.
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  // Project function applies the final projection on the accumulated value `a`.
  inline C10_DEVICE out_t project(acc_t a) const {
    return compat_pow(a, static_cast<acc_t>(1.0) / norm_);
  }

  // Static function to translate accumulated value `acc` given `base_idx`.
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // CUDA/HIP specific function using warp shuffle operation to exchange data
  // `acc` is the current accumulated value and `offset` is the shuffle offset.
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif

  // Constructor initializes with the norm value.
  NormOps(acc_t norm_): norm_(norm_) {
  }
};

// This accumulator template is used to calculate the order zero norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormZeroOps {
  // Reduce function accumulates the order zero norm of the absolute value of `data`.
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + (data == static_cast<scalar_t>(0) ? static_cast<acc_t>(0) : static_cast<acc_t>(1));
  }

  // Combine function combines two accumulated values `a` and `b`.
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  // Project function applies the final projection on the accumulated value `a`.
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }

  // Static function to translate accumulated value `acc` given `base_idx`.
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // CUDA/HIP specific function using warp shuffle operation to exchange data
  // `acc` is the current accumulated value and `offset` is the shuffle offset.
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// This accumulator template is used to calculate the order one norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormOneOps {
  // Reduce function accumulates the order one norm of the absolute value of `data`.
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + static_cast<acc_t>(std::abs(data));
  }

  // Combine function combines two accumulated values `a` and `b`.
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  // Project function applies the final projection on the accumulated value `a`.
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }

  // Static function to translate accumulated value `acc` given `base_idx`.
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // CUDA/HIP specific function using warp shuffle operation to exchange data
  // `acc` is the current accumulated value and `offset` is the shuffle offset.
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

template<typename acc_t>
struct AbsSwitch {};
// 当输入为非复数类型时，直接返回数据的绝对值，类型转换为累加类型 acc_t
template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(scalar_t data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(data);
}

// 当输入为 std::complex<scalar_t> 类型时，返回其绝对值的累加类型 acc_t，使用 std::abs 计算复数的模
template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(std::complex<scalar_t> data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(std::abs(data));
}

// 当输入为 c10::complex<scalar_t> 类型时，返回其绝对值的累加类型 acc_t，使用 std::abs 计算复数的模
template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(c10::complex<scalar_t> data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(std::abs(data));
}

// 此累加器模板用于计算一组数字的绝对值的二次范数。
// `scalar_t` 是输入的数据类型，`acc_t` 是累加值的类型，`out_t` 是输出的类型。
// 这些类型在支持复数输入时有所不同。
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormTwoOps {
  // 在累加器中减少操作：对于给定的数据，计算其绝对值后的平方，并累加到当前累加值 acc 上
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    acc_t data_ = abs_if_complex(data, AbsSwitch<acc_t>());
    return acc + data_ * data_;
  }

  // 将两个累加值合并的操作，这里简单地进行累加操作
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  // 投影操作：根据累加值计算并返回输出值 out_t
  inline C10_DEVICE out_t project(acc_t a) const {
    return device_sqrt(a);
  }

  // 将累加值与基础索引进行翻译的静态方法，这里不进行任何处理，直接返回累加值 acc
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 在 CUDA 或 HIP 环境下，使用 warp level shuffle 操作将累加值向下偏移 offset
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// 累加器模板，用于处理包含 NaN 值的累加操作
template <typename acc_t, typename data_t>
struct NanSumOps {
  // 在累加器中减少操作：如果数据为 NaN，则加 0，否则加上数据本身
  inline C10_DEVICE acc_t reduce(acc_t a, data_t b, int64_t /*idx*/) const {
    return a + (at::_isnan(b) ? acc_t{0.} : acc_t{b});
  }

  // 将两个累加值合并的操作，这里简单地进行累加操作
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return  a + b;
  }

  // 投影操作：将累加值转换为数据类型 data_t，并返回
  inline C10_DEVICE data_t project(acc_t a) const {
    return data_t{a};
  }

  // 将累加值与基础索引进行翻译的静态方法，这里不进行任何处理，直接返回累加值 acc
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 在 CUDA 或 HIP 环境下，使用 warp level shuffle 操作将数据向下偏移 offset
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

namespace detail {

// 比较器结构体，用于处理包含 NaN 值的比较操作，判断两个数的大小或者索引
template <typename scalar_t>
struct LessOrNan {
  // 比较操作符：如果其中一个数为 NaN，则将 NaN 视为最小值，否则比较它们的大小
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b; // 如果两个数都是 NaN，则按索引比较
      }
      return true; // 如果 a 是 NaN，而 b 不是，则认为 a 较小
    }
    return (a == b) ? idx_a < idx_b : (a < b); // 否则比较 a 和 b 的大小
  }
};

// 比较器结构体，用于处理包含 NaN 值的比较操作，判断两个数的大小或者索引
template <typename scalar_t>
struct GreaterOrNan {
  // 比较操作符：如果其中一个数为 NaN，则将 NaN 视为最大值，否则比较它们的大小
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b; // 如果两个数都是 NaN，则按索引比较
      }
      return true; // 如果 a 是 NaN，而 b 不是，则认为 a 较大
    }
    return (a == b) ? idx_a < idx_b : (a > b); // 否则比较 a 和 b 的大小
  }
};

template <typename comp_t>
struct MinMaxReductionOps {
  // scalar_t 定义为 comp_t 二元函数特征的第一个参数类型
  using scalar_t = typename binary_function_traits<comp_t>::arg1_t;
  // index_t 定义为 int64_t 类型
  using index_t = int64_t;
  // arg_t 定义为 detail::pair<scalar_t, index_t> 类型
  using arg_t = detail::pair<scalar_t, index_t>;

  // 静态函数，返回输入参数 arg
  static C10_DEVICE arg_t project(arg_t arg) {
    return arg;
  }

  // 静态函数，根据比较函数 comp_t 对 arg 和 (val, idx) 进行规约
  static C10_DEVICE arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    // 如果 comp_t{}(arg.first, val, arg.second, idx) 返回 true，则返回 arg，否则返回 arg_t(val, idx)
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  // 静态函数，根据比较函数 comp_t 对 a 和 b 进行合并
  static C10_DEVICE arg_t combine(arg_t a, arg_t b) {
    // 如果 comp_t{}(a.first, b.first, a.second, b.second) 返回 true，则返回 a，否则返回 b
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  // 静态函数，将参数 a 的索引值增加 base_idx 后返回
  static C10_DEVICE arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 静态函数，使用 WARP_SHFL_DOWN 将参数 arg 的成员向下移动 offset 位后返回
  static C10_DEVICE arg_t warp_shfl_down(arg_t arg, int offset) {
    return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                 WARP_SHFL_DOWN(arg.second, offset));
  }
#endif
};

template <typename comp_t>
// 继承 MinMaxReductionOps<comp_t>，定义 ArgReductionOps 类
struct ArgReductionOps : public MinMaxReductionOps<comp_t> {
  // scalar_t 和 index_t 从 MinMaxReductionOps<comp_t> 继承
  using typename MinMaxReductionOps<comp_t>::scalar_t;
  using typename MinMaxReductionOps<comp_t>::index_t;
  using typename MinMaxReductionOps<comp_t>::arg_t;

  // 静态函数，返回参数 arg 的索引值
  static C10_DEVICE index_t project(arg_t arg) {
    return arg.second;
  }
};

} // namespace detail

template <typename scalar_t>
// 定义 ArgMaxOps 类，继承于 detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>>
struct ArgMaxOps :
  public detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template <typename scalar_t>
// 定义 ArgMinOps 类，继承于 detail::ArgReductionOps<detail::LessOrNan<scalar_t>>
struct ArgMinOps :
  public detail::ArgReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
// 定义 MinOps 类，继承于 detail::MinMaxReductionOps<detail::LessOrNan<scalar_t>>
struct MinOps :
  public detail::MinMaxReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
// 定义 MaxOps 类，继承于 detail::MinMaxReductionOps<detail::GreaterOrNan<scalar_t>>
struct MaxOps :
  public detail::MinMaxReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template <typename scalar_t, typename acc_scalar_t, typename index_t>
// 定义 MinMaxOps 类
struct MinMaxOps {
  // 定义 acc_t 类型为 detail::pair<acc_scalar_t, acc_scalar_t>
  using acc_t = detail::pair<acc_scalar_t, acc_scalar_t>;

  // 内联函数，对 acc 进行 reduce 操作，将 data 和 /*idx*/ 组合后返回
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    return combine(acc, {data, data});
  }

  // 内联函数，对 acc 进行 combine 操作，返回更新后的 acc
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    auto min_val = (at::_isnan(a.first) || a.first < b.first) ? a.first : b.first;
    auto max_val = (at::_isnan(a.second) || a.second > b.second) ? a.second : b.second;

    return {min_val, max_val};
  }

  // 内联函数，直接返回参数 acc
  inline C10_DEVICE acc_t project(acc_t acc) const {
    return acc;
  }

  // 静态函数，对 acc 执行 translate_idx 操作，直接返回 acc
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  // 内联函数，使用 WARP_SHFL_DOWN 将 acc 的成员向下移动 offset 位后返回
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.first, offset), WARP_SHFL_DOWN(acc.second, offset)
    };
  }
#endif
};

}} // namespace at::native

#undef MAX
#undef MIN
```