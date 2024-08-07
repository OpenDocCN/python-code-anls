# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib\trlib_types.h`

```
/* MIT License
 *
 * Copyright (c) 2016--2017 Felix Lenders
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef TRLIB_TYPES_H
#define TRLIB_TYPES_H

// 定义自定义整数类型 trlib_int_t 为长整型
typedef long trlib_int_t;
// 定义自定义浮点类型 trlib_flt_t 为双精度浮点型
typedef double trlib_flt_t;

// 定义机器精度下的常量
#define TRLIB_EPS            ((trlib_flt_t)2.2204460492503131e-16)
#define TRLIB_EPS_POW_4      ((trlib_flt_t)5.4774205922939014e-07)
#define TRLIB_EPS_POW_5      ((trlib_flt_t)1.4901161193847656e-08)
#define TRLIB_EPS_POW_75     ((trlib_flt_t)1.8189894035458565e-12)

#endif
```