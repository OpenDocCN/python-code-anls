# `.\numpy\numpy\random\src\mt19937\mt19937.h`

```
#pragma once
// 防止头文件重复包含

#include <math.h>
// 包含数学库函数

#include <stdint.h>
// 包含标准整数类型定义

#if defined(_WIN32) && !defined (__MINGW32__)
#define inline __forceinline
#endif
// 如果编译环境为 Windows，并且非 MinGW，定义 inline 为 __forceinline

#define RK_STATE_LEN 624
// 定义 Mersenne Twister 状态数组的长度为 624

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL
// 定义 Mersenne Twister 算法中使用的常量

typedef struct s_mt19937_state {
  uint32_t key[RK_STATE_LEN];
  // 用于存储 Mersenne Twister 状态的数组
  int pos;
  // 当前状态数组中的位置指针
} mt19937_state;
// 定义 Mersenne Twister 状态结构体

extern void mt19937_seed(mt19937_state *state, uint32_t seed);
// 外部函数声明：初始化 Mersenne Twister 状态数组

extern void mt19937_gen(mt19937_state *state);
// 外部函数声明：生成 Mersenne Twister 的下一个状态

/* Slightly optimized reference implementation of the Mersenne Twister */
// 稍微优化的 Mersenne Twister 参考实现

static inline uint32_t mt19937_next(mt19937_state *state) {
  uint32_t y;

  if (state->pos == RK_STATE_LEN) {
    // 如果当前位置指针达到状态数组长度
    mt19937_gen(state);
    // 生成下一个 Mersenne Twister 状态数组
  }
  y = state->key[state->pos++];

  /* Tempering */
  // 状态数据的调整
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
  // 返回生成的随机数
}

extern void mt19937_init_by_array(mt19937_state *state, uint32_t *init_key,
                                  int key_length);
// 外部函数声明：通过数组初始化 Mersenne Twister 状态数组

static inline uint64_t mt19937_next64(mt19937_state *state) {
  return (uint64_t)mt19937_next(state) << 32 | mt19937_next(state);
  // 返回生成的 64 位随机数
}

static inline uint32_t mt19937_next32(mt19937_state *state) {
  return mt19937_next(state);
  // 返回生成的 32 位随机数
}

static inline double mt19937_next_double(mt19937_state *state) {
  int32_t a = mt19937_next(state) >> 5, b = mt19937_next(state) >> 6;
  // 使用生成的随机数生成双精度浮点数
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

void mt19937_jump(mt19937_state *state);
// 外部函数声明：进行 Mersenne Twister 的跳跃操作
```