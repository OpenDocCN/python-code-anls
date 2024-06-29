# `.\numpy\numpy\_core\src\multiarray\dragon4.h`

```py
/*
 * Copyright (c) 2014 Ryan Juckett
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 * This file contains a modified version of Ryan Juckett's Dragon4
 * implementation, obtained from https://www.ryanjuckett.com,
 * which has been ported from C++ to C and which has
 * modifications specific to printing floats in numpy.
 *
 * Ryan Juckett's original code was under the Zlib license; he gave numpy
 * permission to include it under the MIT license instead.
 */
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DRAGON4_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DRAGON4_H_

#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "npy_config.h"

#include "numpy/arrayscalars.h"

/* Half binary format */
#define NPY_HALF_BINFMT_NAME IEEE_binary16

/* Float binary format */
#if NPY_BITSOF_FLOAT == 32
    #define NPY_FLOAT_BINFMT_NAME IEEE_binary32
#elif NPY_BITSOF_FLOAT == 64
    #define NPY_FLOAT_BINFMT_NAME IEEE_binary64
#else
    #error No float representation defined
#endif

/* Double binary format */
#if NPY_BITSOF_DOUBLE == 32
    #define NPY_DOUBLE_BINFMT_NAME IEEE_binary32
#elif NPY_BITSOF_DOUBLE == 64
    #define NPY_DOUBLE_BINFMT_NAME IEEE_binary64
#else
    #error No double representation defined
#endif

/* LongDouble binary format */
#if defined(HAVE_LDOUBLE_IEEE_QUAD_BE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary128_be
#elif defined(HAVE_LDOUBLE_IEEE_QUAD_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary128_le
#elif (defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
       defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE))
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary64
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Intel_extended96
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Intel_extended128
#elif defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Motorola_extended96
#else
    #error No long double representation defined
#endif
#endif  // NUMPY_CORE_SRC_MULTIARRAY_DRAGON4_H_
    // 定义一个名为 NPY_LONGDOUBLE_BINFMT_NAME 的宏，并设置其值为 Motorola_extended96
    #define NPY_LONGDOUBLE_BINFMT_NAME Motorola_extended96
#elif (defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE) || \
       defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE))
    // 如果定义了具有 IBM 双倍精度浮点的小数类型，设置 NPY_LONGDOUBLE_BINFMT_NAME 为 IBM_double_double
    #define NPY_LONGDOUBLE_BINFMT_NAME IBM_double_double
#else
    // 如果没有定义任何长双精度表示，抛出错误
    #error No long double representation defined
#endif

typedef enum DigitMode
{
    // 将数字四舍五入以打印最短且唯一可识别的数字。
    DigitMode_Unique,
    // 输出数字的位数，就好像具有无限精度一样。
    DigitMode_Exact,
} DigitMode;

typedef enum CutoffMode
{
    // 打印最多 cutoffNumber 个有效数字。
    CutoffMode_TotalLength,
    // 打印小数点后最多 cutoffNumber 个有效数字。
    CutoffMode_FractionLength,
} CutoffMode;

typedef enum TrimMode
{
    TrimMode_None,         // 不修剪零，始终保留小数点。
    TrimMode_LeaveOneZero, // 修剪除小数点前的零以外的所有零。
    TrimMode_Zeros,        // 修剪所有尾随零，保留小数点。
    TrimMode_DptZeros,     // 修剪尾随零和小数点。
} TrimMode;

#define make_dragon4_typedecl(Type, npy_type) \
    PyObject *\
    Dragon4_Positional_##Type(npy_type *val, DigitMode digit_mode,\
                              CutoffMode cutoff_mode, int precision,\
                              int min_digits, int sign, TrimMode trim, \
                              int pad_left, int pad_right);\
    PyObject *\
    Dragon4_Scientific_##Type(npy_type *val, DigitMode digit_mode,\
                              int precision, int min_digits, int sign, \
                              TrimMode trim, int pad_left, int exp_digits);

// 为 Half、Float、Double、LongDouble 分别声明 Dragon4_Positional 和 Dragon4_Scientific 函数
make_dragon4_typedecl(Half, npy_half)
make_dragon4_typedecl(Float, npy_float)
make_dragon4_typedecl(Double, npy_double)
make_dragon4_typedecl(LongDouble, npy_longdouble)

#undef make_dragon4_typedecl

// 声明 Dragon4_Positional 和 Dragon4_Scientific 函数，参数为通用 PyObject 类型
PyObject *
Dragon4_Positional(PyObject *obj, DigitMode digit_mode, CutoffMode cutoff_mode,
                   int precision, int min_digits, int sign, TrimMode trim,
                   int pad_left, int pad_right);

PyObject *
Dragon4_Scientific(PyObject *obj, DigitMode digit_mode, int precision,
                   int min_digits, int sign, TrimMode trim, int pad_left,
                   int exp_digits);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DRAGON4_H_ */
```