# `.\numpy\numpy\random\src\pcg64\pcg64.h`

```
/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     https://www.pcg-random.org
 *
 * Relicensed MIT in May 2019
 *
 * The MIT License
 *
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PCG64_H_INCLUDED
#define PCG64_H_INCLUDED 1

#include <inttypes.h>   // 包含整数类型定义

#ifdef _WIN32
#include <stdlib.h>     // 包含标准库定义
#endif

#if defined(_WIN32) && !defined (__MINGW32__)
#define inline __forceinline   // 定义内联函数修饰符为 __forceinline
#endif

#if defined(__GNUC_GNU_INLINE__) && !defined(__cplusplus)
#error Nonstandard GNU inlining semantics. Compile with -std=c99 or better.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SIZEOF_INT128__) && !defined(PCG_FORCE_EMULATED_128BIT_MATH)
typedef __uint128_t pcg128_t;   // 定义 128 位无符号整数类型 pcg128_t
#define PCG_128BIT_CONSTANT(high, low) (((pcg128_t)(high) << 64) + low)   // 定义生成 128 位常量的宏
#else
typedef struct {
  uint64_t high;    // 128 位结构体高位 64 位
  uint64_t low;     // 128 位结构体低位 64 位
} pcg128_t;

static inline pcg128_t PCG_128BIT_CONSTANT(uint64_t high, uint64_t low) {
  pcg128_t result;
  result.high = high;   // 设置高位值
  result.low = low;     // 设置低位值
  return result;        // 返回设置好的结构体
}

#define PCG_EMULATED_128BIT_MATH 1   // 定义为使用模拟的 128 位数学运算
#endif
typedef struct { pcg128_t state; } pcg_state_128;

定义了一个结构体 `pcg_state_128`，包含一个名为 `state` 的成员变量，类型为 `pcg128_t`。


typedef struct {
  pcg128_t state;
  pcg128_t inc;
} pcg_state_setseq_128;

定义了一个结构体 `pcg_state_setseq_128`，包含两个成员变量 `state` 和 `inc`，都是类型为 `pcg128_t` 的变量。


#define PCG_DEFAULT_MULTIPLIER_HIGH 2549297995355413924ULL
#define PCG_DEFAULT_MULTIPLIER_LOW 4865540595714422341ULL

定义了两个宏 `PCG_DEFAULT_MULTIPLIER_HIGH` 和 `PCG_DEFAULT_MULTIPLIER_LOW`，分别表示一个128位乘法的高位和低位默认乘数。


#define PCG_DEFAULT_MULTIPLIER_128                                             \
  PCG_128BIT_CONSTANT(PCG_DEFAULT_MULTIPLIER_HIGH, PCG_DEFAULT_MULTIPLIER_LOW)
#define PCG_DEFAULT_INCREMENT_128                                              \
  PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)

定义了两个宏 `PCG_DEFAULT_MULTIPLIER_128` 和 `PCG_DEFAULT_INCREMENT_128`，分别表示默认的128位乘数和增量。`PCG_128BIT_CONSTANT` 是一个宏，用于将高位和低位常量合并成一个 `pcg128_t` 类型的常量。


#define PCG_STATE_SETSEQ_128_INITIALIZER                                       \
  {                                                                            \
    PCG_128BIT_CONSTANT(0x979c9a98d8462005ULL, 0x7d3e9cb6cfe0549bULL)          \
    , PCG_128BIT_CONSTANT(0x0000000000000001ULL, 0xda3e39cb94b95bdbULL)        \
  }

定义了一个宏 `PCG_STATE_SETSEQ_128_INITIALIZER`，用于初始化 `pcg_state_setseq_128` 结构体的实例，其中包括 `state` 和 `inc` 的初始值。


#define PCG_CHEAP_MULTIPLIER_128 (0xda942042e4dd58b5ULL)

定义了一个宏 `PCG_CHEAP_MULTIPLIER_128`，表示一个较为简单的128位乘数。


static inline uint64_t pcg_rotr_64(uint64_t value, unsigned int rot) {
#ifdef _WIN32
  return _rotr64(value, rot);
#else
  return (value >> rot) | (value << ((-rot) & 63));
#endif
}

定义了一个静态内联函数 `pcg_rotr_64`，实现对 `value` 进行64位的循环右移操作。


#ifdef PCG_EMULATED_128BIT_MATH

static inline pcg128_t pcg128_add(pcg128_t a, pcg128_t b) {
  pcg128_t result;

  result.low = a.low + b.low;
  result.high = a.high + b.high + (result.low < b.low);
  return result;
}

如果定义了 `PCG_EMULATED_128BIT_MATH` 宏，则定义了一个静态内联函数 `pcg128_add`，实现对两个 `pcg128_t` 类型的数进行加法操作，并处理溢出。


static inline void _pcg_mult64(uint64_t x, uint64_t y, uint64_t *z1,
                               uint64_t *z0) {
#if defined _WIN32 && _M_AMD64
  z0[0] = _umul128(x, y, z1);
#else
  uint64_t x0, x1, y0, y1;
  uint64_t w0, w1, w2, t;
  *z0 = x * y;

  x0 = x & 0xFFFFFFFFULL;
  x1 = x >> 32;
  y0 = y & 0xFFFFFFFFULL;
  y1 = y >> 32;
  w0 = x0 * y0;
  t = x1 * y0 + (w0 >> 32);
  w1 = t & 0xFFFFFFFFULL;
  w2 = t >> 32;
  w1 += x0 * y1;
  *z1 = x1 * y1 + w2 + (w1 >> 32);
#endif
}

定义了一个静态内联函数 `_pcg_mult64`，用于对两个64位数 `x` 和 `y` 进行乘法操作，并将结果存储在 `z1` 和 `z0` 中。根据平台不同选择使用不同的乘法实现。


static inline pcg128_t pcg128_mult(pcg128_t a, pcg128_t b) {
  uint64_t h1;
  pcg128_t result;

  h1 = a.high * b.low + a.low * b.high;
  _pcg_mult64(a.low, b.low, &(result.high), &(result.low));
  result.high += h1;
  return result;
}

定义了一个静态内联函数 `pcg128_mult`，用于对两个 `pcg128_t` 类型的数进行乘法操作，返回乘积结果。


static inline void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng) {
  rng->state = pcg128_add(pcg128_mult(rng->state, PCG_DEFAULT_MULTIPLIER_128),
                           rng->inc);
}

定义了一个静态内联函数 `pcg_setseq_128_step_r`，实现了一步 PCG 128位序列生成器的状态更新。


static inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state) {
  return pcg_rotr_64(state.high ^ state.low, state.high >> 58u);
}

定义了一个静态内联函数 `pcg_output_xsl_rr_128_64`，实现了 PCG 128位序列生成器的输出函数。


static inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq) {
  rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
  rng->inc.high = initseq.high << 1u;
  rng->inc.high |= initseq.low >> 63u;
  rng->inc.low = (initseq.low << 1u) | 1u;
  pcg_setseq_128_step_r(rng);
  rng->state = pcg128_add(rng->state, initstate);
  pcg_setseq_128_step_r(rng);
}

定义了一个静态内联函数 `pcg_setseq_128_srandom_r`，用于初始化 PCG 128位序列生成器的状态和增量，以及执行初始步骤。


static inline uint64_t

定义了一个静态内联函数的开始部分，但截至当前行没有提供具体实现。
static inline pcg128_t pcg128_mult_64(pcg128_t a, uint64_t b) {
  // 定义变量 h1 存储高位乘积结果
  uint64_t h1;
  // 定义结果变量 result
  pcg128_t result;

  // 计算高位乘积
  h1 = a.high * b;
  // 调用 _pcg_mult64 函数计算低位乘积
  _pcg_mult64(a.low, b, &(result.high), &(result.low));
  // 将高位乘积加到 result 的高位
  result.high += h1;
  // 返回乘积结果
  return result;
}

static inline void pcg_cm_step_r(pcg_state_setseq_128 *rng) {
#if defined _WIN32 && _M_AMD64
  // 定义变量 h1 存储高位乘积结果
  uint64_t h1;
  // 定义结果变量 product
  pcg128_t product;

  // 手动内联使用内部函数进行乘法和加法运算
  // 计算高位乘积
  h1 = rng->state.high * PCG_CHEAP_MULTIPLIER_128;
  // 调用 _umul128 函数计算低位乘积
  product.low =
      _umul128(rng->state.low, PCG_CHEAP_MULTIPLIER_128, &(product.high));
  // 将高位乘积加到 product 的高位
  product.high += h1;
  // 使用 _addcarry_u64 函数更新 rng->state 中的值
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
#else
  // 在非 Windows 下，调用 pcg128_mult_64 和 pcg128_add 函数更新 rng->state
  rng->state = pcg128_add(pcg128_mult_64(rng->state, PCG_CHEAP_MULTIPLIER_128),
                           rng->inc);
#endif
}

static inline void pcg_cm_srandom_r(pcg_state_setseq_128 *rng, pcg128_t initstate, pcg128_t initseq) {
  // 初始化 rng->state 和 rng->inc
  rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
  rng->inc.high = initseq.high << 1u;
  rng->inc.high |= initseq.low >> 63u;
  rng->inc.low = (initseq.low << 1u) | 1u;
  // 执行一次 pcg_cm_step_r
  pcg_cm_step_r(rng);
  // 更新 rng->state
  rng->state = pcg128_add(rng->state, initstate);
  // 再次执行 pcg_cm_step_r
  pcg_cm_step_r(rng);
}

static inline uint64_t pcg_cm_random_r(pcg_state_setseq_128* rng)
{
  // 高位和低位初始化为 rng->state 的值
  uint64_t hi = rng->state.high;
  uint64_t lo = rng->state.low;

  // 执行 DXSM 输出函数在预迭代状态上
  // 将低位设置为 1
  lo |= 1;
  // 高位按位右移 32 位后与自身按位异或
  hi ^= hi >> 32;
  // 高位乘以常数
  hi *= 0xda942042e4dd58b5ULL;
  // 再次高位按位右移 48 位后与自身按位异或
  hi ^= hi >> 48;
  // 高位乘以低位
  hi *= lo;

  // 执行 CM 步骤
#if defined _WIN32 && _M_AMD64
  // 定义变量 h1 存储高位乘积结果
  uint64_t h1;
  // 定义结果变量 product
  pcg128_t product;

  // 手动内联使用内部函数进行乘法和加法运算
  // 计算高位乘积
  h1 = rng->state.high * PCG_CHEAP_MULTIPLIER_128;
  // 调用 _umul128 函数计算低位乘积
  product.low =
      _umul128(rng->state.low, PCG_CHEAP_MULTIPLIER_128, &(product.high));
  // 将高位乘积加到 product 的高位
  product.high += h1;
  // 使用 _addcarry_u64 函数更新 rng->state 中的值
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
#else
  // 在非 Windows 下，调用 pcg128_mult_64 和 pcg128_add 函数更新 rng->state
  rng->state = pcg128_add(pcg128_mult_64(rng->state, PCG_CHEAP_MULTIPLIER_128),
                           rng->inc);
#endif
  // 返回高位
  return hi;
}
#else /* PCG_EMULATED_128BIT_MATH */
代码块开始，说明在不支持128位整数运算的情况下使用的条件编译。

static inline void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng) {
  rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_128 + rng->inc;
}
定义函数pcg_setseq_128_step_r，用于PCG状态结构体的状态更新。

static inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state) {
  return pcg_rotr_64(((uint64_t)(state >> 64u)) ^ (uint64_t)state,
                     state >> 122u);
}
定义函数pcg_output_xsl_rr_128_64，用于生成64位随机数的输出函数，通过XSL-RR算法实现。

static inline void pcg_cm_step_r(pcg_state_setseq_128 *rng) {
  rng-> state = rng->state * PCG_CHEAP_MULTIPLIER_128 + rng->inc;
}
定义函数pcg_cm_step_r，用于PCG状态结构体的状态更新，采用便宜乘法器。

static inline uint64_t pcg_output_cm_128_64(pcg128_t state) {
  uint64_t hi = state >> 64;
  uint64_t lo = state;

  lo |= 1;
  hi ^= hi >> 32;
  hi *= 0xda942042e4dd58b5ULL;
  hi ^= hi >> 48;
  hi *= lo;
  return hi;
}
定义函数pcg_output_cm_128_64，用于生成64位随机数的输出函数，通过CM算法实现。

static inline void pcg_cm_srandom_r(pcg_state_setseq_128 *rng, pcg128_t initstate, pcg128_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg_cm_step_r(rng);
  rng->state += initstate;
  pcg_cm_step_r(rng);
}
定义函数pcg_cm_srandom_r，用于初始化PCG状态结构体，采用CM算法。

static inline uint64_t pcg_cm_random_r(pcg_state_setseq_128* rng)
{
    uint64_t ret = pcg_output_cm_128_64(rng->state);
    pcg_cm_step_r(rng);
    return ret;
}
定义函数pcg_cm_random_r，用于生成64位随机数，基于已初始化的PCG状态结构体，采用CM算法。

static inline uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128* rng)
{
    pcg_setseq_128_step_r(rng);
    return pcg_output_xsl_rr_128_64(rng->state);
}
定义函数pcg_setseq_128_xsl_rr_64_random_r，用于生成64位随机数，基于已初始化的PCG状态结构体，采用XSL-RR算法。

static inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg_setseq_128_step_r(rng);
  rng->state += initstate;
  pcg_setseq_128_step_r(rng);
}
定义函数pcg_setseq_128_srandom_r，用于初始化PCG状态结构体，采用setseq（序列设置）算法。

#endif /* PCG_EMULATED_128BIT_MATH */
代码块结束，结束条件编译段，对于不支持128位整数运算的情况。

static inline uint64_t
pcg_setseq_128_xsl_rr_64_boundedrand_r(pcg_state_setseq_128 *rng,
                                       uint64_t bound) {
  uint64_t threshold = -bound % bound;
  for (;;) {
    uint64_t r = pcg_setseq_128_xsl_rr_64_random_r(rng);
    if (r >= threshold)
      return r % bound;
  }
}
定义函数pcg_setseq_128_xsl_rr_64_boundedrand_r，生成一个不超过bound的随机数，基于已初始化的PCG状态结构体，采用XSL-RR算法。

extern pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta,
                                    pcg128_t cur_mult, pcg128_t cur_plus);
声明函数pcg_advance_lcg_128，用于LCG（线性同余生成器）的128位状态推进。

static inline void pcg_setseq_128_advance_r(pcg_state_setseq_128 *rng,
                                            pcg128_t delta) {
  rng->state = pcg_advance_lcg_128(rng->state, delta,
                                   PCG_DEFAULT_MULTIPLIER_128, rng->inc);
}
定义函数pcg_setseq_128_advance_r，用于推进PCG状态结构体，采用setseq算法。

static inline void pcg_cm_advance_r(pcg_state_setseq_128 *rng, pcg128_t delta) {
    rng->state = pcg_advance_lcg_128(rng->state, delta,
                                     PCG_128BIT_CONSTANT(0, PCG_CHEAP_MULTIPLIER_128),
                                     rng->inc);
}
定义函数pcg_cm_advance_r，用于推进PCG状态结构体，采用CM算法。

typedef pcg_state_setseq_128 pcg64_random_t;
声明pcg64_random_t类型，作为pcg_state_setseq_128的别名，用于64位随机数生成。

#define pcg64_random_r pcg_setseq_128_xsl_rr_64_random_r
#define pcg64_boundedrand_r pcg_setseq_128_xsl_rr_64_boundedrand_r
#define pcg64_srandom_r pcg_setseq_128_srandom_r
#define pcg64_advance_r pcg_setseq_128_advance_r
#define PCG64_INITIALIZER PCG_STATE_SETSEQ_128_INITIALIZER
定义宏，为pcg64_random_r、pcg64_boundedrand_r、pcg64_srandom_r、pcg64_advance_r提供别名，以及PCG64_INITIALIZER的定义。

#ifdef __cplusplus
}
#endif
结束C++的extern "C"声明，如果是C++环境则关闭extern "C"。
# 定义一个结构体 `s_pcg64_state`，用于封装 PCG64 随机数生成器的状态信息
typedef struct s_pcg64_state {
  pcg64_random_t *pcg_state;  // 指向 PCG64 随机数生成器状态的指针
  int has_uint32;              // 标志位，指示是否有未使用的 uint32_t 类型的随机数
  uint32_t uinteger;           // 存储未使用的 uint32_t 类型的随机数
} pcg64_state;

# 定义一个静态内联函数 `pcg64_next64`，用于生成下一个 64 位随机数
static inline uint64_t pcg64_next64(pcg64_state *state) {
  return pcg64_random_r(state->pcg_state);  // 调用 PCG64 随机数生成器生成下一个 64 位随机数
}

# 定义一个静态内联函数 `pcg64_next32`，用于生成下一个 32 位随机数
static inline uint32_t pcg64_next32(pcg64_state *state) {
  uint64_t next;
  if (state->has_uint32) {  // 如果有未使用的 uint32_t 类型的随机数
    state->has_uint32 = 0;  // 标志位复位
    return state->uinteger; // 返回未使用的 uint32_t 类型的随机数
  }
  next = pcg64_random_r(state->pcg_state);  // 生成一个新的随机数
  state->has_uint32 = 1;                    // 设置标志位，表示有未使用的 uint32_t 类型的随机数
  state->uinteger = (uint32_t)(next >> 32); // 提取新生成的随机数的高 32 位作为 uint32_t 类型的随机数
  return (uint32_t)(next & 0xffffffff);     // 返回新生成的随机数的低 32 位
}

# 定义一个静态内联函数 `pcg64_cm_next64`，用于生成下一个 64 位随机数（带状态变更）
static inline uint64_t pcg64_cm_next64(pcg64_state *state) {
  return pcg_cm_random_r(state->pcg_state);  // 调用 PCG64-CM 随机数生成器生成下一个 64 位随机数
}

# 定义一个静态内联函数 `pcg64_cm_next32`，用于生成下一个 32 位随机数（带状态变更）
static inline uint32_t pcg64_cm_next32(pcg64_state *state) {
  uint64_t next;
  if (state->has_uint32) {  // 如果有未使用的 uint32_t 类型的随机数
    state->has_uint32 = 0;  // 标志位复位
    return state->uinteger; // 返回未使用的 uint32_t 类型的随机数
  }
  next = pcg_cm_random_r(state->pcg_state);  // 生成一个新的随机数
  state->has_uint32 = 1;                     // 设置标志位，表示有未使用的 uint32_t 类型的随机数
  state->uinteger = (uint32_t)(next >> 32);  // 提取新生成的随机数的高 32 位作为 uint32_t 类型的随机数
  return (uint32_t)(next & 0xffffffff);      // 返回新生成的随机数的低 32 位
}

# 声明函数 `pcg64_advance`，用于向前推进 PCG64 随机数生成器状态
void pcg64_advance(pcg64_state *state, uint64_t *step);

# 声明函数 `pcg64_cm_advance`，用于向前推进 PCG64-CM 随机数生成器状态
void pcg64_cm_advance(pcg64_state *state, uint64_t *step);

# 声明函数 `pcg64_set_seed`，用于设置 PCG64 随机数生成器的种子和增量
void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc);

# 声明函数 `pcg64_get_state`，用于获取 PCG64 随机数生成器的状态信息
void pcg64_get_state(pcg64_state *state, uint64_t *state_arr, int *has_uint32,
                     uint32_t *uinteger);

# 声明函数 `pcg64_set_state`，用于设置 PCG64 随机数生成器的状态信息
void pcg64_set_state(pcg64_state *state, uint64_t *state_arr, int has_uint32,
                     uint32_t uinteger);

# 结束条件编译指令，结束头文件 `pcg64.h` 的定义
#endif /* PCG64_H_INCLUDED */
```