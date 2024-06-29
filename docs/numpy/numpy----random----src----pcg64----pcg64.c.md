# `.\numpy\numpy\random\src\pcg64\pcg64.c`

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

// 包含 PCG64 随机数生成器的头文件
#include "pcg64.h"

// 定义外部链接的内联函数，设置序列 128 步骤
extern inline void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng);
// 定义外部链接的内联函数，输出 XSL-RR 128 到 64 位
extern inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state);
// 定义外部链接的内联函数，初始化序列 128 的状态和序列
extern inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq);
// 定义外部链接的内联函数，生成 XSL-RR 64 位随机数
extern inline uint64_t pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128 *rng);
// 定义外部链接的内联函数，生成 XSL-RR 64 位有界随机数
extern inline uint64_t pcg_setseq_128_xsl_rr_64_boundedrand_r(pcg_state_setseq_128 *rng,
                                                              uint64_t bound);
// 定义外部链接的内联函数，推进序列 128 的状态
extern inline void pcg_setseq_128_advance_r(pcg_state_setseq_128 *rng,
                                            pcg128_t delta);
// 定义外部链接的内联函数，生成 CM 方法的随机数
extern inline uint64_t pcg_cm_random_r(pcg_state_setseq_128 *rng);
// 定义外部链接的内联函数，CM 方法的步骤
extern inline void pcg_cm_step_r(pcg_state_setseq_128 *rng);
/*
 * 外部声明：内联函数，用于从 pcg128_t 状态中输出 uint64_t 类型数据
 */
extern inline uint64_t pcg_output_cm_128_64(pcg128_t state);

/*
 * 外部声明：内联函数，用于初始化 pcg_state_setseq_128 结构中的状态
 */
extern inline void pcg_cm_srandom_r(pcg_state_setseq_128 *rng, pcg128_t initstate, pcg128_t initseq);

/*
 * 多步进函数（跳跃前进、跳跃后退）
 *
 * 该方法基于 Brown 的论文 "Random Number Generation with Arbitrary Stride,"
 * 算法类似于快速指数运算。
 *
 * 即使 delta 是无符号整数，我们也可以传递有符号整数以实现向后跳跃，只不过会"绕远路"。
 */
#ifndef PCG_EMULATED_128BIT_MATH

/*
 * 函数：pcg_advance_lcg_128
 * 功能：根据指定的 delta 值，以及当前的乘法和加法常数，对状态进行跳跃操作
 */
pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                             pcg128_t cur_plus) {
  pcg128_t acc_mult = 1u;   // 累积乘法常数
  pcg128_t acc_plus = 0u;   // 累积加法常数
  while (delta > 0) {       // 当 delta 大于 0 时执行循环
    if (delta & 1) {        // 如果 delta 的最低位为1
      acc_mult *= cur_mult; // 更新累积乘法常数
      acc_plus = acc_plus * cur_mult + cur_plus; // 更新累积加法常数
    }
    cur_plus = (cur_mult + 1) * cur_plus; // 更新当前加法常数
    cur_mult *= cur_mult;   // 更新当前乘法常数
    delta /= 2;             // delta 右移一位（相当于除以2）
  }
  return acc_mult * state + acc_plus; // 返回跳跃后的状态值
}

#else

/*
 * 函数：pcg_advance_lcg_128
 * 功能：根据指定的 delta 值，以及当前的乘法和加法常数，对状态进行跳跃操作
 * 注：使用了 PCG 128位常量来处理数值运算
 */
pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                             pcg128_t cur_plus) {
  pcg128_t acc_mult = PCG_128BIT_CONSTANT(0u, 1u); // 累积乘法常数初始化为 1
  pcg128_t acc_plus = PCG_128BIT_CONSTANT(0u, 0u); // 累积加法常数初始化为 0
  while ((delta.high > 0) || (delta.low > 0)) {    // 当 delta 的高位或低位大于 0 时执行循环
    if (delta.low & 1) {                          // 如果 delta 的最低位为1
      acc_mult = pcg128_mult(acc_mult, cur_mult); // 更新累积乘法常数
      acc_plus = pcg128_add(pcg128_mult(acc_plus, cur_mult), cur_plus); // 更新累积加法常数
    }
    cur_plus = pcg128_mult(pcg128_add(cur_mult, PCG_128BIT_CONSTANT(0u, 1u)),
                            cur_plus);           // 更新当前加法常数
    cur_mult = pcg128_mult(cur_mult, cur_mult);   // 更新当前乘法常数
    delta.low = (delta.low >> 1) | (delta.high << 63); // delta 右移一位，高位低位交替更新
    delta.high >>= 1;                             // delta 高位右移一位
  }
  return pcg128_add(pcg128_mult(acc_mult, state), acc_plus); // 返回跳跃后的状态值
}

#endif

/*
 * 外部声明：内联函数，用于从 pcg64_state 结构中获取下一个 uint64_t 类型数据
 */
extern inline uint64_t pcg64_next64(pcg64_state *state);

/*
 * 外部声明：内联函数，用于从 pcg64_state 结构中获取下一个 uint32_t 类型数据
 */
extern inline uint32_t pcg64_next32(pcg64_state *state);

/*
 * 外部声明：内联函数，用于从 pcg64_state 结构中获取下一个 uint64_t 类型数据
 */
extern inline uint64_t pcg64_cm_next64(pcg64_state *state);

/*
 * 外部声明：内联函数，用于从 pcg64_state 结构中获取下一个 uint32_t 类型数据
 */
extern inline uint32_t pcg64_cm_next32(pcg64_state *state);

/*
 * 函数：pcg64_advance
 * 功能：根据给定的步长，以 pcg128_t 形式传递 delta 值，并调用 pcg64_advance_r 函数
 */
extern void pcg64_advance(pcg64_state *state, uint64_t *step) {
  pcg128_t delta; // 定义 pcg128_t 类型的 delta 变量
#ifndef PCG_EMULATED_128BIT_MATH
  delta = (((pcg128_t)step[0]) << 64) | step[1]; // 构造 delta 值（未模拟128位数学时）
#else
  delta.high = step[0];  // 设置 delta 高位
  delta.low = step[1];   // 设置 delta 低位
#endif
  pcg64_advance_r(state->pcg_state, delta); // 调用 pcg64_advance_r 函数
}

/*
 * 函数：pcg64_cm_advance
 * 功能：根据给定的步长，以 pcg128_t 形式传递 delta 值，并调用 pcg_cm_advance_r 函数
 */
extern void pcg64_cm_advance(pcg64_state *state, uint64_t *step) {
  pcg128_t delta; // 定义 pcg128_t 类型的 delta 变量
#ifndef PCG_EMULATED_128BIT_MATH
  delta = (((pcg128_t)step[0]) << 64) | step[1]; // 构造 delta 值（未模拟128位数学时）
#else
  delta.high = step[0];  // 设置 delta 高位
  delta.low = step[1];   // 设置 delta 低位
#endif
  pcg_cm_advance_r(state->pcg_state, delta); // 调用 pcg_cm_advance_r 函数
}

/*
 * 函数：pcg64_set_seed
 * 功能：根据给定的种子和增量，以 pcg128_t 形式传递 s 和 i 值，并调用 pcg64_srandom_r 函数
 */
extern void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc) {
  pcg128_t s, i; // 定义 pcg128_t 类型的 s 和 i 变量
#ifndef PCG_EMULATED_128BIT_MATH
  s = (((pcg128_t)seed[0]) << 64) | seed[1]; // 构造种子 s 值（未模拟128位数学时）
  i = (((pcg128_t)inc[0]) << 64) | inc[1];   // 构造增量 i 值（未模拟128位数学时）
#else
  s.high = seed[0]; // 设置种子 s 的高位
  s.low = seed[1];  // 设置种子 s 的低位
  i.high = inc[0];  // 设置增量 i 的高位
  i.low = inc[1];   // 设置增量 i 的低位
#endif
  pcg64_srandom_r(state->pcg_state, s, i); // 调用 pcg64_srandom_r 函数
}
extern void pcg64_get_state(pcg64_state *state, uint64_t *state_arr,
                            int *has_uint32, uint32_t *uinteger) {
    /*
     * state_arr contains state.high, state.low, inc.high, inc.low
     *    which are interpreted as the upper 64 bits (high) or lower
     *    64 bits of a uint128_t variable
     *
     */

#ifndef PCG_EMULATED_128BIT_MATH
    // 将 PCG 状态的高 64 位存入 state_arr[0]
    state_arr[0] = (uint64_t)(state->pcg_state->state >> 64);
    // 将 PCG 状态的低 64 位存入 state_arr[1]
    state_arr[1] = (uint64_t)(state->pcg_state->state & 0xFFFFFFFFFFFFFFFFULL);
    // 将 PCG 增量的高 64 位存入 state_arr[2]
    state_arr[2] = (uint64_t)(state->pcg_state->inc >> 64);
    // 将 PCG 增量的低 64 位存入 state_arr[3]
    state_arr[3] = (uint64_t)(state->pcg_state->inc & 0xFFFFFFFFFFFFFFFFULL);
#else
    // 使用 emulated 128 位整数模式时，直接将状态和增量的高低位赋给 state_arr
    state_arr[0] = (uint64_t)state->pcg_state->state.high;
    state_arr[1] = (uint64_t)state->pcg_state->state.low;
    state_arr[2] = (uint64_t)state->pcg_state->inc.high;
    state_arr[3] = (uint64_t)state->pcg_state->inc.low;
#endif

    // 将 state 的 has_uint32 值赋给 has_uint32 数组
    has_uint32[0] = state->has_uint32;
    // 将 state 的 uinteger 值赋给 uinteger 数组
    uinteger[0] = state->uinteger;
}

extern void pcg64_set_state(pcg64_state *state, uint64_t *state_arr,
                            int has_uint32, uint32_t uinteger) {
    /*
     * state_arr contains state.high, state.low, inc.high, inc.low
     *    which are interpreted as the upper 64 bits (high) or lower
     *    64 bits of a uint128_t variable
     *
     */

#ifndef PCG_EMULATED_128BIT_MATH
    // 根据 state_arr 的高低位设置 PCG 状态的整体值
    state->pcg_state->state = (((pcg128_t)state_arr[0]) << 64) | state_arr[1];
    // 根据 state_arr 的高低位设置 PCG 增量的整体值
    state->pcg_state->inc = (((pcg128_t)state_arr[2]) << 64) | state_arr[3];
#else
    // 使用 emulated 128 位整数模式时，直接赋值给状态和增量的高低位
    state->pcg_state->state.high = state_arr[0];
    state->pcg_state->state.low = state_arr[1];
    state->pcg_state->inc.high = state_arr[2];
    state->pcg_state->inc.low = state_arr[3];
#endif

    // 将 has_uint32 的值赋给 state 的 has_uint32
    state->has_uint32 = has_uint32;
    // 将 uinteger 的值赋给 state 的 uinteger
    state->uinteger = uinteger;
}
```