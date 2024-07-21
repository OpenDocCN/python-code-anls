# `.\pytorch\aten\src\ATen\core\PhiloxRNGEngine.h`

```py
#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#ifdef __CUDACC__
#include <cuda.h>
#endif

#include <ATen/core/Array.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>

namespace at {

// typedefs for holding vector data
namespace detail {

// 定义用于存储向量数据的类型别名
typedef at::detail::Array<uint32_t, 4> UINT4;   // 4 个 uint32_t 元素的数组
typedef at::detail::Array<uint32_t, 2> UINT2;   // 2 个 uint32_t 元素的数组
typedef at::detail::Array<double, 2> DOUBLE2;   // 2 个 double 元素的数组
typedef at::detail::Array<float, 2> FLOAT2;     // 2 个 float 元素的数组

} // namespace detail

/**
 * Note [Philox Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in PyTorch's fusion compiler
 * Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 * for details regarding the engine.
 *
 * Note that currently this implementation of the philox engine is not used
 * anywhere except for tests in cpu_generator_test.cpp. However, this engine
 * will replace curandStatePhilox4_32_10_t in the future.
 *
 * The philox engine takes a seed value, a subsequeunce
 * for starting the generation and an offset for the subsequence.
 * Think of this engine as an algorithm producing a huge array. We are
 * parallelizing this array by partitioning the huge array and assigning
 * a thread index to each partition. In other words, each seed value
 * (there are 2^64 possible seed values) gives a sub array of size
 * 2^128 (each element in that array is a 128 bit number). Reasoning
 * behind the array being of size 2^128 is, there are 2^64 possible
 * thread index value and there is an array of size 2^64 for each of
 * those thread index. Hence 2^64 * 2^64 = 2^128 for each seed value.
 *
 * In short, this generator can produce 2^64 (seed values) * 2^128 (number
 * of elements in an array given by a seed value) = 2^192 values.
 *
 * Arguments:
 * seed:        Seed values could be any number from 0 to 2^64-1.
 * subsequence: Subsequence is just the cuda thread indexing with:
 *              - blockIdx.x * blockDim.x + threadIdx.x
 * offset:      The offset variable in PhiloxEngine  decides how many 128-bit
 *              random numbers to skip (i.e. how many groups of 4, 32-bit numbers to skip)
 *              and hence really decides the total number of randoms that can be achieved
 *              for the given subsequence.
 */

// 定义 Philox 随机数生成引擎类
class philox_engine {
public:

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，初始化 Philox 引擎的状态
  C10_HOST_DEVICE inline explicit philox_engine(uint64_t seed = 67280421310721,
                                 uint64_t subsequence = 0,
                                 uint64_t offset = 0) {

    // 使用给定的 seed 和 subsequence 重置引擎状态
    reset_state(seed, subsequence);
    // 根据 offset 值增加状态，跳过一定数量的随机数
    incr_n(offset);
  }

  // 重置引擎状态的方法
  C10_HOST_DEVICE inline void reset_state(uint64_t seed = 67280421310721,
                                 uint64_t subsequence = 0) {
    // 将 seed 分配给引擎的关键字数组
    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> 32);

    // 将 seed 的低 32 位赋值给 key_ 数组的第一个元素，高 32 位赋值给第二个元素
  }

private:
  uint32_t key_[2];  // 用于存储引擎状态的关键字数组，每个元素为 32 位无符号整数
};

} // namespace at
    // 初始化计数器为零，detail::UINT4是一个自定义的数据结构，用于处理32位无符号整数的细节
    counter_ = detail::UINT4(0);
    // 将64位的subsequence分别存入计数器的第二和第三个位置
    counter_[2] = static_cast<uint32_t>(subsequence);
    counter_[3] = static_cast<uint32_t>(subsequence >> 32);
    // 状态设为0
    STATE = 0;
  }

  /**
   * 设置Philox生成器的偏移字段到指定的偏移量。
   */
  C10_HOST_DEVICE inline void set_offset(uint64_t offset) {
    // 将64位偏移量分别存入计数器的第一个和第二个位置
    counter_[0] = static_cast<uint32_t>(offset);
    counter_[1] = static_cast<uint32_t>(offset >> 32);
  }

  /**
   * 获取当前Philox生成器的偏移量。
   */
  C10_HOST_DEVICE uint64_t get_offset() const {
    // 将计数器的低32位转换为64位，将计数器的高32位左移32位，再转换为64位，然后合并返回
    uint64_t lo = static_cast<uint64_t>(counter_[0]);
    uint64_t hi = static_cast<uint64_t>(counter_[1]) << 32;
    return lo | hi;
  }

  /**
   * 每次调用产生一个唯一的32位伪随机数。保持状态以避免浪费。
   */
  C10_HOST_DEVICE inline uint32_t operator()(int32_t n_rounds = 10) { // 10 here to preserve back-compat behavior
    // 如果状态为0，则重新生成随机数，并增加状态
    if(STATE == 0) {
      detail::UINT4 counter = counter_;
      detail::UINT2 key = key_;
      output_ = rand(counter, key, n_rounds);
      incr();
    }
    // 返回当前状态位置的输出，并更新状态
    uint32_t ret = output_[static_cast<int>(STATE)];
    STATE = (STATE + 1) & 3;
    return ret;
  }

  // 在CUDA环境下，防止调用randn
  inline float randn(uint32_t n_rounds) {
    #ifdef __CUDA_ARCH__
    AT_ASSERT(false, "Unsupported invocation of randn on CUDA");
    #endif
    // 如果状态为0，则重新生成随机数，并增加状态
    if(STATE == 0) {
      detail::UINT4 counter = counter_;
      detail::UINT2 key = key_;
      output_ = rand(counter, key, n_rounds);
      incr();
    }
    // TODO(min-jean-cho) change to Polar method, a more efficient version of Box-Muller method
    // TODO(voz) We use std:: below, and thus need a separate impl for CUDA.
    // 生成标准正态分布的随机数，使用Box-Muller方法，这里还有一些待完善的地方
    float u1 = 1 - uint32_to_uniform_float(output_[0]); // uint32_to_uniform_float返回[0,1)，这里需要(0,1]避免传递0给log
    float u2 = 1 - uint32_to_uniform_float(output_[1]);
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
  }

  /**
   * 跳过子序列中的N个128位数的函数。
   */
  C10_HOST_DEVICE inline void incr_n(uint64_t n) {
    uint32_t nlo = static_cast<uint32_t>(n);
    uint32_t nhi = static_cast<uint32_t>(n >> 32);
    // 将低32位加到计数器的第一个位置，如果有溢出则进位到nhi
    counter_[0] += nlo;
    if (counter_[0] < nlo) {
      nhi++;
      // 如果进位过程中nhi有溢出，则继续向后进位
      counter_[1] += nhi;
      if(nhi != 0) {
        if (nhi <= counter_[1]) {
          return;
        }
      }
    } else {
      // 如果加法中y有溢出，则直接进位到nhi
      counter_[1] += nhi;
      if (nhi <= counter_[1]) {
        return;
      }
    }
    // 如果128位数计数器的第二和第三位置都加到最大值，则进位到第四和第五位
    if (++counter_[2])
      return;
    ++counter_[3];
  }

  /**
   * 跳过子序列中的一个128位数的函数。
   */
  C10_HOST_DEVICE inline void incr() {
    // 如果计数器第一个位置加一后没有溢出，则直接返回
    if (++counter_[0])
      return;
    // 如果计数器第一个位置加一后溢出了，则进位到第二个位置
    if (++counter_[1])
      return;
    // 如果计数器第二个位置加一后没有溢出，则直接返回
    if (++counter_[2])
      return;
    // 否则进位到第四个位置
    ++counter_[3];
  }
    // 如果 counter_ 数组的第三个元素自增后不为零，则执行以下代码块
    if (++counter_[2]) {
        // 如果条件满足，直接返回，中断当前函数执行
        return;
    }
    // 否则，对 counter_ 数组的第四个元素执行自增操作
    ++counter_[3];
}
private:
  // 计数器，用于存储 Philox 算法的计数器值
  detail::UINT4 counter_;
  // 输出值，用于存储 Philox 算法的输出值
  detail::UINT4 output_;
  // 密钥，用于存储 Philox 算法的密钥
  detail::UINT2 key_;
  // 状态值，用于存储 Philox 算法的状态

  // CUDA 可执行环境下的 32 位乘法高低位计算
  C10_HOST_DEVICE inline uint32_t mulhilo32(uint32_t a, uint32_t b,
                                    uint32_t *result_high) {
    #ifdef __CUDA_ARCH__
      // 使用 CUDA 内置函数计算乘法的高位和返回乘积的低位
      *result_high = __umulhi(a, b);
      return a * b;
    #else
      // 在非 CUDA 环境下，使用标准 C++ 实现乘法的高位和返回乘积的低位
      const uint64_t product = static_cast<uint64_t>(a) * b;
      *result_high = static_cast<uint32_t>(product >> 32);
      return static_cast<uint32_t>(product);
    #endif
  }

  // 单轮 Philox 算法的实现
  C10_HOST_DEVICE inline detail::UINT4 single_round(detail::UINT4 ctr, detail::UINT2 in_key) {
    // 定义变量用于存储乘法的高低位结果
    uint32_t hi0 = 0;
    uint32_t hi1 = 0;
    // 使用 mulhilo32 函数计算 PhiloxSA 和 PhiloxSB 与计数器的乘积的高低位
    uint32_t lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
    uint32_t lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
    // 创建返回的 UINT4 对象，按照 Philox 算法的要求设置每个元素的值
    detail::UINT4 ret;
    ret[0] = hi1 ^ ctr[1] ^ in_key[0];
    ret[1] = lo1;
    ret[2] = hi0 ^ ctr[3] ^ in_key[1];
    ret[3] = lo0;
    return ret;
  }

  // 将 32 位无符号整数转换为均匀分布的浮点数
  C10_HOST_DEVICE constexpr float uint32_to_uniform_float(uint32_t value) {
      // 最大值，使得 `MAX_INT * scale < 1.0` （考虑浮点数舍入）
      constexpr float scale = 4.6566127342e-10;
      return static_cast<float>(value & 0x7FFFFFFF) * scale;
  }

  // Philox 算法的主体，执行多轮的 Philox 算法
  C10_HOST_DEVICE inline detail::UINT4 rand(detail::UINT4& counter, detail::UINT2& key, uint32_t n_rounds) {
    // 执行指定轮数的 Philox 算法循环
    for (uint32_t round = 0; round < (n_rounds - 1); round++) {
        // 执行单轮 Philox 算法，并更新计数器值
        counter = single_round(counter, key);
        // 更新密钥的值，按照 Philox 算法的规则更新
        key[0] += (kPhilox10A); key[1] += (kPhilox10B);
      }
    // 执行最后一轮 Philox 算法，并返回结果
    return single_round(counter, key);
  }

  // Philox 算法常量定义
  static const uint32_t kPhilox10A = 0x9E3779B9;
  static const uint32_t kPhilox10B = 0xBB67AE85;
  static const uint32_t kPhiloxSA = 0xD2511F53;
  static const uint32_t kPhiloxSB = 0xCD9E8D57;
};

// 定义 Philox4_32 类型作为 philox_engine 的别名
typedef philox_engine Philox4_32;

} // namespace at
```