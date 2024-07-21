# `.\pytorch\aten\src\ATen\core\MT19937RNGEngine.h`

```py
#pragma once

#include <c10/util/irange.h>

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>

namespace at {

// 定义 Mersenne Twister 算法的状态空间大小和参数
constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

/**
 * mt19937_data_pod is used to get POD data in and out
 * of mt19937_engine. Used in torch.get_rng_state and
 * torch.set_rng_state functions.
 */
// 定义结构体 mt19937_data_pod，用于在 mt19937_engine 中传输 POD 数据
struct mt19937_data_pod {
  uint64_t seed_;         // 种子
  int left_;              // 剩余生成数目
  bool seeded_;           // 是否已种子化
  uint32_t next_;         // 下一个生成数的索引
  std::array<uint32_t, MERSENNE_STATE_N> state_;  // 状态数组
};

// 定义 Mersenne Twister 生成器类 mt19937_engine
class mt19937_engine {
public:

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，根据种子初始化生成器
  inline explicit mt19937_engine(uint64_t seed = 5489) {
    init_with_uint32(seed);
  }

  // 返回当前生成器的数据结构
  inline mt19937_data_pod data() const {
    return data_;
  }

  // 设置生成器的数据结构
  inline void set_data(const mt19937_data_pod& data) {
    data_ = data;
  }

  // 返回生成器的种子
  inline uint64_t seed() const {
    return data_.seed_;
  }

  // 检查生成器状态是否有效
  inline bool is_valid() {
    if ((data_.seeded_ == true)
      && (data_.left_ > 0 && data_.left_ <= MERSENNE_STATE_N)
      && (data_.next_ <= MERSENNE_STATE_N)) {
      return true;
    }
    return false;
  }

  // 生成下一个随机数
  inline uint32_t operator()() {
    if (--(data_.left_) == 0) {
        next_state();
    }
    uint32_t y = *(data_.state_.data() + data_.next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
  }

private:
  mt19937_data_pod data_;

  // 使用 uint32_t 类型的种子初始化生成器
  inline void init_with_uint32(uint64_t seed) {
    data_.seed_ = seed;
    data_.seeded_ = true;
    data_.state_[0] = seed & 0xffffffff;
    for (const auto j : c10::irange(1, MERSENNE_STATE_N)) {
      data_.state_[j] = (1812433253 * (data_.state_[j-1] ^ (data_.state_[j-1] >> 30)) + j);
    }
    data_.left_ = 1;
    data_.next_ = 0;
  }

  // 混合两个位
  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  // 扭曲函数
  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  // 生成下一个状态
  inline void next_state() {
    uint32_t* p = data_.state_.data();
    data_.left_ = MERSENNE_STATE_N;
    data_.next_ = 0;

    for(int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for(int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], data_.state_[0]);
  }

};

// mt19937_engine 的别名为 mt19937
typedef mt19937_engine mt19937;

} // namespace at
```