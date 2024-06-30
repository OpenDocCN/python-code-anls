# `D:\src\scipysrc\scipy\scipy\stats\libnpyrandom\distributions.h`

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

#ifndef _DISTRIBUTIONS_H_
#define _DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// 定义用于生成随机数的状态结构体
typedef struct {
  void *state;  // 状态指针，指向保存随机数生成器状态的数据结构
  uint64_t (*next_uint64)(void *st);  // 函数指针，生成下一个 uint64_t 类型的随机数
  uint32_t (*next_uint32)(void *st);  // 函数指针，生成下一个 uint32_t 类型的随机数
  double (*next_double)(void *st);    // 函数指针，生成下一个 double 类型的随机数
  uint64_t (*next_raw)(void *st);     // 函数指针，生成下一个原始数据的随机数
} bitgen_t;

// 声明正态分布的随机数生成函数
extern double random_normal(bitgen_t *bitgen_state, double loc, double scale);
// 声明指定区间的随机数生成函数
extern uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max);
// 声明标准均匀分布的随机数生成函数
extern double random_standard_uniform(bitgen_t *bitgen_state);

#ifdef __cplusplus
}
#endif

#endif  /* _DISTRIBUTIONS_H_ */
```