# `D:\src\scipysrc\scipy\scipy\optimize\rectangular_lsap\rectangular_lsap.h`

```
/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
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

/*
定义 RECTANGULAR_LSAP_H 预处理器标识符，防止重复包含
*/
#ifndef RECTANGULAR_LSAP_H
#define RECTANGULAR_LSAP_H

/*
定义矩形线性和分配问题无法解决的返回值常量
*/
#define RECTANGULAR_LSAP_INFEASIBLE -1
/*
定义矩形线性和分配问题无效输入的返回值常量
*/
#define RECTANGULAR_LSAP_INVALID -2

#ifdef __cplusplus
extern "C" {
#endif

/*
引入标准整数类型 intptr_t 和布尔类型 bool 的头文件
*/
#include <stdint.h>
#include <stdbool.h>

/*
声明解决矩形线性和分配问题的函数，接受行数、列数、成本数组、最大化标志、结果数组 a 和 b 作为参数
*/
int solve_rectangular_linear_sum_assignment(intptr_t nr, intptr_t nc,
                                            double* input_cost, bool maximize,
                                            int64_t* a, int64_t* b);

#ifdef __cplusplus
}
#endif

/*
结束 RECTANGULAR_LSAP_H 的预处理器条件编译
*/
#endif
```