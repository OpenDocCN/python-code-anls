# `.\numpy\numpy\random\src\sfc64\sfc64.h`

```py
#ifndef _RANDOMDGEN__SFC64_H_
#define _RANDOMDGEN__SFC64_H_

#include "numpy/npy_common.h"  // 包含 numpy 的通用头文件
#include <inttypes.h>          // 包含整数类型定义的头文件
#ifdef _WIN32
#include <stdlib.h>            // 在 Windows 下包含标准库头文件
#endif

typedef struct s_sfc64_state {
  uint64_t s[4];        // 64 位无符号整数数组，用于保存状态
  int has_uint32;       // 标志位，指示是否有未使用的 32 位随机数
  uint32_t uinteger;    // 保存未使用的 32 位随机数
} sfc64_state;          // 定义结构体 sfc64_state

// 定义一个左旋函数，根据操作系统选择具体实现方式
static inline uint64_t rotl(const uint64_t value, unsigned int rot) {
#ifdef _WIN32
  return _rotl64(value, rot);   // 在 Windows 下使用内置函数 _rotl64 实现左旋
#else
  return (value << rot) | (value >> ((-rot) & 63));  // 在其他系统下使用位操作实现左旋
#endif
}

// SFC64 算法的主要随机数生成函数，更新状态并返回生成的随机数
static inline uint64_t sfc64_next(uint64_t *s) {
  const uint64_t tmp = s[0] + s[1] + s[3]++;  // 计算临时变量 tmp

  s[0] = s[1] ^ (s[1] >> 11);   // 更新状态数组 s 的第一个元素
  s[1] = s[2] + (s[2] << 3);    // 更新状态数组 s 的第二个元素
  s[2] = rotl(s[2], 24) + tmp;  // 更新状态数组 s 的第三个元素

  return tmp;   // 返回生成的随机数
}

// 使用 sfc64_next 函数生成一个 64 位的随机数
static inline uint64_t sfc64_next64(sfc64_state *state) {
  return sfc64_next(&state->s[0]);  // 调用 sfc64_next 函数生成随机数并返回
}

// 使用 sfc64_next 函数生成一个 32 位的随机数
static inline uint32_t sfc64_next32(sfc64_state *state) {
  uint64_t next;
  if (state->has_uint32) {   // 如果标志位表明有未使用的 32 位随机数
    state->has_uint32 = 0;   // 清除标志位
    return state->uinteger;  // 返回未使用的 32 位随机数
  }
  next = sfc64_next(&state->s[0]);   // 否则调用 sfc64_next 生成一个随机数
  state->has_uint32 = 1;             // 设置标志位为有未使用的 32 位随机数
  state->uinteger = (uint32_t)(next >> 32);  // 将生成的 64 位随机数分成高 32 位和低 32 位
  return (uint32_t)(next & 0xffffffff);     // 返回低 32 位作为随机数
}

// 设置 SFC64 算法的种子
void sfc64_set_seed(sfc64_state *state, uint64_t *seed);

// 获取 SFC64 算法的状态
void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32,
                     uint32_t *uinteger);

// 设置 SFC64 算法的状态
void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32,
                     uint32_t uinteger);

#endif  // _RANDOMDGEN__SFC64_H_
```