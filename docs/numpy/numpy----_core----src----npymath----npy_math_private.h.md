# `.\numpy\numpy\_core\src\npymath\npy_math_private.h`

```py
/*
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * from: @(#)fdlibm.h 5.1 93/09/24
 * $FreeBSD$
 */

#ifndef _NPY_MATH_PRIVATE_H_
#define _NPY_MATH_PRIVATE_H_

#include <Python.h>
#ifdef __cplusplus
#include <cmath>
using std::isgreater;
using std::isless;
#else
#include <math.h>
#endif

#include "npy_config.h"
#include "npy_fpmath.h"

#include "numpy/npy_math.h"
#include "numpy/npy_endian.h"
#include "numpy/npy_common.h"

/*
 * The original fdlibm code used statements like:
 *      n0 = ((*(int*)&one)>>29)^1;             * index of high word *
 *      ix0 = *(n0+(int*)&x);                   * high word of x *
 *      ix1 = *((1-n0)+(int*)&x);               * low word of x *
 * to dig two 32 bit words out of the 64 bit IEEE floating point
 * value.  That is non-ANSI, and, moreover, the gcc instruction
 * scheduler gets it wrong.  We instead use the following macros.
 * Unlike the original code, we determine the endianness at compile
 * time, not at run time; I don't see much benefit to selecting
 * endianness at run time.
 */

/*
 * A union which permits us to convert between a double and two 32 bit
 * ints.
 */

/* XXX: not really, but we already make this assumption elsewhere. Will have to
 * fix this at some point */
#define IEEE_WORD_ORDER NPY_BYTE_ORDER

#if IEEE_WORD_ORDER == NPY_BIG_ENDIAN

typedef union
{
  double value;
  struct
  {
    npy_uint32 msw;
    npy_uint32 lsw;
  } parts;
} ieee_double_shape_type;

#endif

#if IEEE_WORD_ORDER == NPY_LITTLE_ENDIAN

typedef union
{
  double value;
  struct
  {
    npy_uint32 lsw;
    npy_uint32 msw;
  } parts;
} ieee_double_shape_type;

#endif

/* Get two 32 bit ints from a double.  */

#define EXTRACT_WORDS(ix0,ix1,d)                                \
do {                                                            \
  ieee_double_shape_type ew_u;                                  \
  ew_u.value = (d);                                             \
  (ix0) = ew_u.parts.msw;                                       \
  (ix1) = ew_u.parts.lsw;                                       \
} while (0)

/* Get the more significant 32 bit int from a double.  */

#define GET_HIGH_WORD(i,d)                                      \
do {                                                            \
  ieee_double_shape_type gh_u;                                  \
  gh_u.value = (d);                                             \
  (i) = gh_u.parts.msw;                                         \
} while (0)

/* Get the less significant 32 bit int from a double.  */

#define GET_LOW_WORD(i,d)                                       \
do {                                                            \
  ieee_double_shape_type gl_u;                                  \
  gl_u.value = (d);                                             \
  (i) = gl_u.parts.lsw;                                         \
} while (0)

#endif /* _NPY_MATH_PRIVATE_H_ */
/*
 * Set the low 32 bits of a double from an int value.
 */
#define SET_LOW_WORD(d,v)                                       \
do {                                                            \
  ieee_double_shape_type sl_u;                                  \
  sl_u.value = (d);                                             \
  sl_u.parts.lsw = (v);                                         \
  (d) = sl_u.value;                                             \
} while (0)

/*
 * Set the high 32 bits of a double from an int value.
 */
#define SET_HIGH_WORD(d,v)                                      \
do {                                                            \
  ieee_double_shape_type sh_u;                                  \
  sh_u.value = (d);                                             \
  sh_u.parts.msw = (v);                                         \
  (d) = sh_u.value;                                             \
} while (0)

/*
 * Set a double value from two 32-bit int values.
 */
#define INSERT_WORDS(d,ix0,ix1)                                 \
do {                                                            \
  ieee_double_shape_type iw_u;                                  \
  iw_u.parts.msw = (ix0);                                       \
  iw_u.parts.lsw = (ix1);                                       \
  (d) = iw_u.value;                                             \
} while (0)

/*
 * A union allowing conversion between a float and a 32-bit int.
 */
typedef union
{
  float value;
  /* FIXME: Assumes 32 bit int.  */
  npy_uint32 word;
} ieee_float_shape_type;

/*
 * Get a 32-bit int representation of a float value.
 */
#define GET_FLOAT_WORD(i,d)                                     \
do {                                                            \
  ieee_float_shape_type gf_u;                                   \
  gf_u.value = (d);                                             \
  (i) = gf_u.word;                                              \
} while (0)

/*
 * Set a float value from a 32-bit int representation.
 */
#define SET_FLOAT_WORD(d,i)                                     \
do {                                                            \
  ieee_float_shape_type sf_u;                                   \
  sf_u.word = (i);                                              \
  (d) = sf_u.value;                                             \
} while (0)

/*
 * Long double support
 */
#if defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE)
    /*
     * 定义一个类型别名 `IEEEl2bitsrep_part`，用于表示 Intel 扩展的 80 位浮点数的部分位表示
     * Bit representation is
     *          |  junk  |     s  |eeeeeeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          | 16 bits|  1 bit |    15 bits    |            64 bits            |
     *          |             a[2]                |     a[1]     |    a[0]        |
     *
     * 16 位 a[2] 中的更强的位是垃圾位
     */
    typedef npy_uint32 IEEEl2bitsrep_part;
    
    
    union IEEEl2bitsrep {
        npy_longdouble     e;       // 用长双精度浮点数表示的 IEEE 80 位浮点数
        IEEEl2bitsrep_part a[3];    // 用于访问 80 位浮点数各个部分的数组表示
    };
    
    #define LDBL_MANL_INDEX     0    // 低位部分在 a 数组中的索引
    #define LDBL_MANL_MASK      0xFFFFFFFF    // 低位部分的掩码
    #define LDBL_MANL_SHIFT     0    // 低位部分的位移
    
    #define LDBL_MANH_INDEX     1    // 高位部分在 a 数组中的索引
    #define LDBL_MANH_MASK      0xFFFFFFFF    // 高位部分的掩码
    #define LDBL_MANH_SHIFT     0    // 高位部分的位移
    
    #define LDBL_EXP_INDEX      2    // 指数部分在 a 数组中的索引
    #define LDBL_EXP_MASK       0x7FFF    // 指数部分的掩码
    #define LDBL_EXP_SHIFT      0    // 指数部分的位移
    
    #define LDBL_SIGN_INDEX     2    // 符号位在 a 数组中的索引
    #define LDBL_SIGN_MASK      0x8000    // 符号位的掩码
    #define LDBL_SIGN_SHIFT     15    // 符号位的位移
    
    #define LDBL_NBIT           0x80000000    // 用于检测 NaN 的位掩码
    
    typedef npy_uint32 ldouble_man_t;    // 用于表示长双精度浮点数低位部分的类型别名
    typedef npy_uint32 ldouble_exp_t;    // 用于表示长双精度浮点数指数部分的类型别名
    typedef npy_uint32 ldouble_sign_t;   // 用于表示长双精度浮点数符号部分的类型别名
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE)
    /*
     * Intel extended 80 bits precision, 16 bytes alignment.. Bit representation is
     *          |  junk  |     s  |eeeeeeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          | 16 bits|  1 bit |    15 bits    |            64 bits            |
     *          |             a[2]                |     a[1]     |    a[0]        |
     *
     * a[3] and 16 stronger bits of a[2] are junk
     */
    typedef npy_uint32 IEEEl2bitsrep_part;  // 定义IEEE长双精度类型的部分位表示

    union IEEEl2bitsrep {
        npy_longdouble     e;  // 使用长双精度类型的联合表示
        IEEEl2bitsrep_part a[4];  // 使用四个32位整数数组表示长双精度数的位
    };

    #define LDBL_MANL_INDEX     0  // 长双精度数低位的索引
    #define LDBL_MANL_MASK      0xFFFFFFFF  // 长双精度数低位的掩码
    #define LDBL_MANL_SHIFT     0  // 长双精度数低位的位移量

    #define LDBL_MANH_INDEX     1  // 长双精度数高位的索引
    #define LDBL_MANH_MASK      0xFFFFFFFF  // 长双精度数高位的掩码
    #define LDBL_MANH_SHIFT     0  // 长双精度数高位的位移量

    #define LDBL_EXP_INDEX      2  // 长双精度数指数部分的索引
    #define LDBL_EXP_MASK       0x7FFF  // 长双精度数指数部分的掩码
    #define LDBL_EXP_SHIFT      0  // 长双精度数指数部分的位移量

    #define LDBL_SIGN_INDEX     2  // 长双精度数符号位的索引
    #define LDBL_SIGN_MASK      0x8000  // 长双精度数符号位的掩码
    #define LDBL_SIGN_SHIFT     15  // 长双精度数符号位的位移量

    #define LDBL_NBIT           0x800000000  // 长双精度数N位的掩码

    typedef npy_uint32 ldouble_man_t;  // 长双精度数尾数的类型定义
    typedef npy_uint32 ldouble_exp_t;  // 长双精度数指数的类型定义
    typedef npy_uint32 ldouble_sign_t;  // 长双精度数符号的类型定义
#elif defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)
    /*
     * Motorola extended 80 bits precision. Bit representation is
     *          |     s  |eeeeeeeeeeeeeee|  junk  |mmmmmmmm................mmmmmmm|
     *          |  1 bit |    15 bits    | 16 bits|            64 bits            |
     *          |             a[0]                |     a[1]     |    a[2]        |
     *
     * 16 low bits of a[0] are junk
     */
    typedef npy_uint32 IEEEl2bitsrep_part;  // 定义IEEE长双精度类型的部分位表示

    union IEEEl2bitsrep {
        npy_longdouble     e;  // 使用长双精度类型的联合表示
        IEEEl2bitsrep_part a[3];  // 使用三个32位整数数组表示长双精度数的位
    };

    #define LDBL_MANL_INDEX     2  // 长双精度数低位的索引
    #define LDBL_MANL_MASK      0xFFFFFFFF  // 长双精度数低位的掩码
    #define LDBL_MANL_SHIFT     0  // 长双精度数低位的位移量

    #define LDBL_MANH_INDEX     1  // 长双精度数高位的索引
    #define LDBL_MANH_MASK      0xFFFFFFFF  // 长双精度数高位的掩码
    #define LDBL_MANH_SHIFT     0  // 长双精度数高位的位移量

    #define LDBL_EXP_INDEX      0  // 长双精度数指数部分的索引
    #define LDBL_EXP_MASK       0x7FFF0000  // 长双精度数指数部分的掩码
    #define LDBL_EXP_SHIFT      16  // 长双精度数指数部分的位移量

    #define LDBL_SIGN_INDEX     0  // 长双精度数符号位的索引
    #define LDBL_SIGN_MASK      0x80000000  // 长双精度数符号位的掩码
    #define LDBL_SIGN_SHIFT     31  // 长双精度数符号位的位移量

    #define LDBL_NBIT           0x80000000  // 长双精度数N位的掩码

    typedef npy_uint32 ldouble_man_t;  // 长双精度数尾数的类型定义
    typedef npy_uint32 ldouble_exp_t;  // 长双精度数指数的类型定义
    typedef npy_uint32 ldouble_sign_t;  // 长双精度数符号的类型定义
#elif defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE)
    /* 64 bits IEEE double precision aligned on 16 bytes: used by ppc arch on
     * Mac OS X */

    /*
     * IEEE double precision. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  11 bits  |            52 bits            |
     *          |          a[0]         |         a[1]            |
     */
    typedef npy_uint32 IEEEl2bitsrep_part;  // 定义IEEE长双精度类型的部分位表示

    union IEEEl2bitsrep {
        npy_longdouble     e;  // 使用长双精度类型的联合表示
        IEEEl2bitsrep_part a[2];  // 使用两个32位整数数组表示长双精度数的位
    };

    #define LDBL_MANL_INDEX     1  // 长双精度数低位的索引
    # 定义长双精度浮点数的尾数掩码和位移量
    # 尾数掩码，用于提取尾数部分
    # 尾数位移量，用于右移操作提取尾数

    # 定义长双精度浮点数的指数索引、指数掩码和位移量
    # 指数索引，指数掩码用于提取指数部分
    # 指数掩码，用于提取指数部分
    # 指数位移量，用于右移操作提取指数部分

    # 定义长双精度浮点数的符号索引、符号掩码和位移量
    # 符号索引，符号掩码用于提取符号部分
    # 符号掩码，用于提取符号部分
    # 符号位移量，用于右移操作提取符号部分

    # 定义长双精度浮点数的最低有效位
    typedef npy_uint32 ldouble_man_t;  // 定义长双精度浮点数的尾数类型
    typedef npy_uint32 ldouble_exp_t;  // 定义长双精度浮点数的指数类型
    typedef npy_uint32 ldouble_sign_t; // 定义长双精度浮点数的符号类型
#elif defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE)
    /*
     * 64 bits IEEE double precision, Little Endian.
     * 
     * IEEE double precision. Bit representation is
     *          |  s  |eeeeeeeeeee|mmmmmmmm................mmmmmmm|
     *          |1 bit|  11 bits  |            52 bits            |
     *          |          a[1]         |         a[0]            |
     */
    // 定义结构体，用于存储 IEEE 双精度浮点数的位表示
    typedef npy_uint32 IEEEl2bitsrep_part;

    // 联合体，允许以多种方式访问同一存储位置的数据
    union IEEEl2bitsrep {
        npy_longdouble     e;    // 长双精度浮点数
        IEEEl2bitsrep_part a[2]; // 用两个 32 位整数数组表示的位表示
    };

    // 定义尾数低位索引、掩码和位移量
    #define LDBL_MANL_INDEX     0
    #define LDBL_MANL_MASK      0xFFFFFFFF
    #define LDBL_MANL_SHIFT     0

    // 定义尾数高位索引、掩码和位移量
    #define LDBL_MANH_INDEX     1
    #define LDBL_MANH_MASK      0x000FFFFF
    #define LDBL_MANH_SHIFT     0

    // 定义指数索引、掩码和位移量
    #define LDBL_EXP_INDEX      1
    #define LDBL_EXP_MASK       0x7FF00000
    #define LDBL_EXP_SHIFT      20

    // 定义符号索引、掩码和位移量
    #define LDBL_SIGN_INDEX     1
    #define LDBL_SIGN_MASK      0x80000000
    #define LDBL_SIGN_SHIFT     31

    // 定义 NBIT 常量
    #define LDBL_NBIT           0x00000080

    // 定义长双精度浮点数的尾数、指数和符号类型
    typedef npy_uint32 ldouble_man_t;
    typedef npy_uint32 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
    #define LDBL_EXP_MASK       0x7FFF000000000000
    #define LDBL_EXP_SHIFT      48
    
    # 定义长双精度浮点数的指数掩码和指数位移量
    # LDBL_EXP_MASK: 用于屏蔽长双精度浮点数的指数部分的掩码
    # LDBL_EXP_SHIFT: 指数部分在长双精度浮点数中的位移量
    
    
    #define LDBL_SIGN_INDEX     1
    #define LDBL_SIGN_MASK      0x8000000000000000
    #define LDBL_SIGN_SHIFT     63
    
    # 定义长双精度浮点数的符号索引、符号掩码和符号位移量
    # LDBL_SIGN_INDEX: 符号位在长双精度浮点数中的索引位置
    # LDBL_SIGN_MASK: 用于屏蔽长双精度浮点数的符号位的掩码
    # LDBL_SIGN_SHIFT: 符号位在长双精度浮点数中的位移量
    
    
    #define LDBL_NBIT           0
    
    # 定义长双精度浮点数的第一位（最高位）
    # LDBL_NBIT: 长双精度浮点数的最高位（最高有效位）
    
    typedef npy_uint64 ldouble_man_t;
    typedef npy_uint64 ldouble_exp_t;
    typedef npy_uint32 ldouble_sign_t;
    
    # 声明长双精度浮点数的尾数、指数和符号的数据类型
    # ldouble_man_t: 长双精度浮点数的尾数类型，为 64 位无符号整数
    # ldouble_exp_t: 长双精度浮点数的指数类型，为 64 位无符号整数
    # ldouble_sign_t: 长双精度浮点数的符号类型，为 32 位无符号整数
#if !defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE) && \
    !defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE)
/* Get the sign bit of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_SIGN(x) \
    (((x).a[LDBL_SIGN_INDEX] & LDBL_SIGN_MASK) >> LDBL_SIGN_SHIFT)

/* Set the sign bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_SIGN(x, v) \
  ((x).a[LDBL_SIGN_INDEX] =                                             \
   ((x).a[LDBL_SIGN_INDEX] & ~LDBL_SIGN_MASK) |                         \
   (((IEEEl2bitsrep_part)(v) << LDBL_SIGN_SHIFT) & LDBL_SIGN_MASK))

/* Get the exp bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_EXP(x) \
    (((x).a[LDBL_EXP_INDEX] & LDBL_EXP_MASK) >> LDBL_EXP_SHIFT)

/* Set the exp bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_EXP(x, v) \
  ((x).a[LDBL_EXP_INDEX] =                                              \
   ((x).a[LDBL_EXP_INDEX] & ~LDBL_EXP_MASK) |                           \
   (((IEEEl2bitsrep_part)(v) << LDBL_EXP_SHIFT) & LDBL_EXP_MASK))

/* Get the manl bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_MANL(x) \
    (((x).a[LDBL_MANL_INDEX] & LDBL_MANL_MASK) >> LDBL_MANL_SHIFT)

/* Set the manl bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_MANL(x, v) \
  ((x).a[LDBL_MANL_INDEX] =                                             \
   ((x).a[LDBL_MANL_INDEX] & ~LDBL_MANL_MASK) |                         \
   (((IEEEl2bitsrep_part)(v) << LDBL_MANL_SHIFT) & LDBL_MANL_MASK))

/* Get the manh bits of x. x should be of type IEEEl2bitsrep */
#define GET_LDOUBLE_MANH(x) \
    (((x).a[LDBL_MANH_INDEX] & LDBL_MANH_MASK) >> LDBL_MANH_SHIFT)

/* Set the manh bit of x to v. x should be of type IEEEl2bitsrep */
#define SET_LDOUBLE_MANH(x, v) \
    ((x).a[LDBL_MANH_INDEX] = \
     ((x).a[LDBL_MANH_INDEX] & ~LDBL_MANH_MASK) |                       \
     (((IEEEl2bitsrep_part)(v) << LDBL_MANH_SHIFT) & LDBL_MANH_MASK))
#endif /* !HAVE_LDOUBLE_DOUBLE_DOUBLE_* */

#endif /* !_NPY_MATH_PRIVATE_H_ */



/* 
   These macros define operations to manipulate specific bits within a structure
   representing a long double's internal bit layout. They are used when certain
   configurations for IBM's double-double long double format are not defined.
*/

/* 
   GET_LDOUBLE_SIGN(x): Retrieves the sign bit from the IEEEl2bitsrep structure x.

   SET_LDOUBLE_SIGN(x, v): Sets the sign bit of x to the value v (0 or 1).

   GET_LDOUBLE_EXP(x): Retrieves the exponent bits from the IEEEl2bitsrep structure x.

   SET_LDOUBLE_EXP(x, v): Sets the exponent bits of x to the value v.

   GET_LDOUBLE_MANL(x): Retrieves the lower part of the mantissa from the IEEEl2bitsrep structure x.

   SET_LDOUBLE_MANL(x, v): Sets the lower part of the mantissa of x to the value v.

   GET_LDOUBLE_MANH(x): Retrieves the higher part of the mantissa from the IEEEl2bitsrep structure x.

   SET_LDOUBLE_MANH(x, v): Sets the higher part of the mantissa of x to the value v.
*/

/*
   The #endif directives close conditional compilation blocks, checking for specific configurations
   of IBM's double-double long double format and preventing redefinition of these macros when those
   configurations are defined. The final #endif closes the conditional block started by #if.
*/

/*
   The final #endif closes a conditional block based on whether the header file _NPY_MATH_PRIVATE_H_ 
   is defined, ensuring proper inclusion of content based on preprocessor directives.
*/
```