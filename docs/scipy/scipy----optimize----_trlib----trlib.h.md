# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib.h`

```
/*
 * MIT License
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

#ifndef TRLIB_H
#define TRLIB_H

#include <stdio.h>                     // 引入标准输入输出库，提供基本输入输出功能
#include "trlib/trlib_types.h"         // 引入自定义库 trlib_types.h，定义了 TRlib 的类型
#include "trlib/trlib_eigen_inverse.h" // 引入自定义库 trlib_eigen_inverse.h，提供 TRlib 的特定功能
#include "trlib/trlib_krylov.h"        // 引入自定义库 trlib_krylov.h，提供 TRlib 的特定功能
#include "trlib/trlib_leftmost.h"      // 引入自定义库 trlib_leftmost.h，提供 TRlib 的特定功能
#include "trlib/trlib_quadratic_zero.h"// 引入自定义库 trlib_quadratic_zero.h，提供 TRlib 的特定功能
#include "trlib/trlib_tri_factor.h"    // 引入自定义库 trlib_tri_factor.h，提供 TRlib 的特定功能

#endif
```