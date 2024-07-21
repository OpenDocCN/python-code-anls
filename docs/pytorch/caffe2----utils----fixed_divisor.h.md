# `.\pytorch\caffe2\utils\fixed_divisor.h`

```
// 如果未定义 CAFFE2_UTILS_FIXED_DIVISOR_H_，则定义该头文件
#ifndef CAFFE2_UTILS_FIXED_DIVISOR_H_
#define CAFFE2_UTILS_FIXED_DIVISOR_H_

// 包含必要的 C++ 标准库头文件
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 根据条件定义 FIXED_DIVISOR_DECL 宏
// 如果代码运行在 CUDA 或 HIP 设备上，声明为内联函数并设定为主机和设备可调用
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__) || defined(__HIP__) || \
    (defined(__clang__) && defined(__CUDA__))
#define FIXED_DIVISOR_DECL inline __host__ __device__
#else
// 否则只声明为内联函数
#define FIXED_DIVISOR_DECL inline
#endif

// 定义 caffe2 命名空间
namespace caffe2 {

// Utility class for quickly calculating quotients and remainders for
// a known integer divisor
// 用于快速计算已知整数除数的商和余数的实用类模板
template <typename T>
class FixedDivisor {};

// Works for any positive divisor, 1 to INT_MAX. One 64-bit
// multiplication and one 64-bit shift is used to calculate the
// result.
// 特化模板 FixedDivisor<std::int32_t>，适用于任何正除数，范围为 1 到 INT_MAX
// 使用一次 64 位乘法和一次 64 位移位来计算结果
template <>
class FixedDivisor<std::int32_t> {
 public:
  // 默认构造函数
  FixedDivisor() = default;

  // 显式构造函数，根据给定的整数 d 初始化 FixedDivisor 对象
  explicit FixedDivisor(const std::int32_t d) : d_(d) {
    // 如果未定义 USE_ROCM，调用 CalcSignedMagic() 函数
#if !defined(USE_ROCM)
    CalcSignedMagic();
#endif // USE_ROCM
  }

  // 返回当前对象的整数除数 d_
  FIXED_DIVISOR_DECL std::int32_t d() const {
    return d_;
  }

  // 如果未定义 USE_ROCM，返回魔数 magic_
  FIXED_DIVISOR_DECL std::uint64_t magic() const {
    return magic_;
  }

  // 如果未定义 USE_ROCM，返回移位数 shift_
  FIXED_DIVISOR_DECL int shift() const {
    return shift_;
  }

  /// Calculates `q = n / d`.
  // 计算 `q = n / d` 的函数，如果定义了 USE_ROCM 直接返回结果，否则进行魔数计算
  FIXED_DIVISOR_DECL std::int32_t Div(const std::int32_t n) const {
#if defined(USE_ROCM)
    return n / d_;
#else // USE_ROCM
    // 如果没有 mulhi 指令可用，通过 uint64 进行计算
    return (int32_t)((magic_ * (uint64_t)n) >> shift_);
#endif // USE_ROCM
  }

  /// Calculates `r = n % d`.
  // 计算 `r = n % d` 的函数
  FIXED_DIVISOR_DECL std::int32_t Mod(const std::int32_t n) const {
    return n - d_ * Div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  // 同时计算 `q = n / d` 和 `r = n % d` 的函数
  FIXED_DIVISOR_DECL void
  DivMod(const std::int32_t n, std::int32_t* q, int32_t* r) const {
    *q = Div(n);
    *r = n - d_ * *q;
  }

 private:
  // 如果未定义 USE_ROCM，计算用于计算有符号 32 位整数除法的魔数和移位量
  // 实现来自 Hacker's Delight 第 10 节
  void CalcSignedMagic() {
    // 如果 d_ 等于 1，则设置 magic_ 为 2^32，shift_ 为 32，并返回
    if (d_ == 1) {
      magic_ = UINT64_C(0x1) << 32;
      shift_ = 32;
      return;
    }

    const std::uint32_t two31 = UINT32_C(0x80000000);
    const std::uint32_t ad = std::abs(d_);
    const std::uint32_t t = two31 + ((uint32_t)d_ >> 31);
    const std::uint32_t anc = t - 1 - t % ad; // Absolute value of nc.
    std::uint32_t p = 31; // Init. p.
    std::uint32_t q1 = two31 / anc; // Init. q1 = 2**p/|nc|.
    std::uint32_t r1 = two31 - q1 * anc; // Init. r1 = rem(2**p, |nc|).
    std::uint32_t q2 = two31 / ad; // Init. q2 = 2**p/|d|.
    std::uint32_t r2 = two31 - q2 * ad; // Init. r2 = rem(2**p, |d|).
    std::uint32_t delta = 0;


这段代码定义了一个用于快速计算整数商和余数的实用类模板 FixedDivisor，特化为处理 std::int32_t 类型的除数。
    // 循环计算魔数 magic 和位移量 shift_
    do {
      ++p;  // 增加 p 的值
      q1 <<= 1; // 更新 q1 = 2**p/|nc|
      r1 <<= 1; // 更新 r1 = rem(2**p, |nc|)
      if (r1 >= anc) { // 如果 r1 大于等于 anc（需要进行无符号比较）
        ++q1; // 增加 q1 的值
        r1 -= anc; // 减去 anc
      }
      q2 <<= 1; // 更新 q2 = 2**p/|d|
      r2 <<= 1; // 更新 r2 = rem(2**p, |d|)
      if (r2 >= ad) { // 如果 r2 大于等于 ad（需要进行无符号比较）
        ++q2; // 增加 q2 的值
        r2 -= ad; // 减去 ad
      }
      delta = ad - r2; // 计算 delta = ad - r2
    } while (q1 < delta || (q1 == delta && r1 == 0)); // 当 q1 小于 delta 或者 q1 等于 delta 且 r1 等于 0 时继续循环

    std::int32_t magic = q2 + 1; // 计算魔数 magic = q2 + 1
    if (d_ < 0) { // 如果 d_ 小于 0
      magic = -magic; // 取反魔数 magic
    }
    shift_ = p; // 设置 shift_ 为 p
    magic_ = (std::uint64_t)(std::uint32_t)magic; // 将 magic 转换为 std::uint64_t 类型并赋值给 magic_
  }
#endif // USE_ROCM

这行代码处于预处理指令 `#endif` 的起始位置，用于结束之前的条件编译区块，该区块在未定义 `USE_ROCM` 宏时生效。


  std::int32_t d_ = 1;

声明并初始化一个 `std::int32_t` 类型的变量 `d_`，赋值为 1。


#if !defined(USE_ROCM)
  std::uint64_t magic_;
  int shift_;
#endif // USE_ROCM

条件编译指令 `#if !defined(USE_ROCM)`，用于在未定义 `USE_ROCM` 宏时编译以下代码块：
- 声明一个未初始化的 `std::uint64_t` 类型变量 `magic_`。
- 声明一个 `int` 类型变量 `shift_`，未初始化。


};

} // namespace caffe2

#endif // CAFFE2_UTILS_FIXED_DIVISOR_H_

结束命名空间 `caffe2` 和头文件 `fixed_divisor.h` 的条件编译区块。
```