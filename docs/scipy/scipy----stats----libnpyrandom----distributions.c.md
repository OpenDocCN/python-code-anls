# `D:\src\scipysrc\scipy\scipy\stats\libnpyrandom\distributions.c`

```
/*
Copyright (c) 2005-2022, NumPy Developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.
    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <stdint.h>

#include "distributions.h"
#include "ziggurat_constants.h"

/* 内联函数：生成下一个 32 位无符号整数 */
static inline uint32_t next_uint32(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint32(bitgen_state->state);
}

/* 内联函数：生成下一个 64 位无符号整数 */
static inline uint64_t next_uint64(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint64(bitgen_state->state);
}

/* 内联函数：生成下一个双精度浮点数 */
static inline double next_double(bitgen_t *bitgen_state) {
    return bitgen_state->next_double(bitgen_state->state);
}

/* 
   生成标准正态分布随机数
   使用 Ziggurat 方法
*/
double random_standard_normal(bitgen_t *bitgen_state) {
  uint64_t r;
  int sign;
  uint64_t rabs;
  int idx;
  double x, xx, yy;
  for (;;) {
    /* 生成一个 64 位随机整数 r */
    r = next_uint64(bitgen_state);
    /* 取低 8 位作为索引 */
    idx = r & 0xff;
    /* 右移 8 位 */
    r >>= 8;
    /* 取第 0 位作为符号 */
    sign = r & 0x1;
    /* 右移 1 位后取低 51 位作为绝对值部分 */
    rabs = (r >> 1) & 0x000fffffffffffff;
    /* 使用预先计算的常数和 rabs 计算 x */
    x = rabs * wi_double[idx];
    /* 如果符号位为 1，则取相反数 */
    if (sign & 0x1)
      x = -x;
    /* 如果 rabs 小于 ki_double[idx]，返回 x */
    if (rabs < ki_double[idx])
      return x; /* 99.3% 的概率会在这里返回 */
    /* 处理特殊情况，当 idx 为 0 时 */
    if (idx == 0) {
      for (;;) {
        /* 使用 1.0 - U 来避免 log(0.0)，参见 GH 13361 */
        xx = -ziggurat_nor_inv_r * log1p(-next_double(bitgen_state));
        yy = -log1p(-next_double(bitgen_state));
        /* 如果满足条件，则返回计算结果 */
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                     : ziggurat_nor_r + xx;
      }
    }
  }
}
    } else {
      // 如果前一个数减去当前数乘以下一个随机双精度数再加上当前数的结果小于 exp(-0.5 * x * x)，则返回 x
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(bitgen_state) +
           fi_double[idx]) < exp(-0.5 * x * x))
        return x;
    }
  }
}

/* 
   生成一个服从正态分布的随机数
   使用给定的均值和标准差生成随机数
*/
double random_normal(bitgen_t *bitgen_state, double loc, double scale) {
  return loc + scale * random_standard_normal(bitgen_state);
}


/*
   生成一个指定范围内的随机整数
   使用给定的最大值生成 [0, max] 范围内的随机整数
*/
uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) {
  uint64_t mask, value;
  if (max == 0) {
    return 0;
  }

  mask = max;

  /* 找到大于等于 max 的最小位掩码 */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  /* 在 [0..mask] 范围内搜索一个随机值，使其 <= max */
  if (max <= 0xffffffffUL) {
    while ((value = (next_uint32(bitgen_state) & mask)) > max)
      ;
  } else {
    while ((value = (next_uint64(bitgen_state) & mask)) > max)
      ;
  }
  return value;
}


/*
   生成一个标准均匀分布的随机数
   直接返回下一个双精度浮点数
*/
double random_standard_uniform(bitgen_t *bitgen_state) {
    return next_double(bitgen_state);
}
```