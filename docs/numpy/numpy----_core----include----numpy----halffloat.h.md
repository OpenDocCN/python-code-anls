# `.\numpy\numpy\_core\include\numpy\halffloat.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_

#include <Python.h>
#include <numpy/npy_math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Half-precision routines
 */

/* 将半精度浮点数转换为单精度浮点数 */
float npy_half_to_float(npy_half h);
/* 将半精度浮点数转换为双精度浮点数 */
double npy_half_to_double(npy_half h);
/* 将单精度浮点数转换为半精度浮点数 */
npy_half npy_float_to_half(float f);
/* 将双精度浮点数转换为半精度浮点数 */
npy_half npy_double_to_half(double d);

/* 比较操作 */
int npy_half_eq(npy_half h1, npy_half h2);      /* 比较两个半精度浮点数是否相等 */
int npy_half_ne(npy_half h1, npy_half h2);      /* 比较两个半精度浮点数是否不相等 */
int npy_half_le(npy_half h1, npy_half h2);      /* 判断第一个半精度浮点数是否小于等于第二个 */
int npy_half_lt(npy_half h1, npy_half h2);      /* 判断第一个半精度浮点数是否小于第二个 */
int npy_half_ge(npy_half h1, npy_half h2);      /* 判断第一个半精度浮点数是否大于等于第二个 */
int npy_half_gt(npy_half h1, npy_half h2);      /* 判断第一个半精度浮点数是否大于第二个 */

/* 更快的非NaN变体，当已知 h1 和 h2 都不是 NaN 时使用 */
int npy_half_eq_nonan(npy_half h1, npy_half h2);    /* 快速比较两个非NaN半精度浮点数是否相等 */
int npy_half_lt_nonan(npy_half h1, npy_half h2);    /* 快速比较第一个非NaN半精度浮点数是否小于第二个 */
int npy_half_le_nonan(npy_half h1, npy_half h2);    /* 快速比较第一个非NaN半精度浮点数是否小于等于第二个 */

/* 杂项函数 */
int npy_half_iszero(npy_half h);            /* 判断半精度浮点数是否为零 */
int npy_half_isnan(npy_half h);             /* 判断半精度浮点数是否为 NaN */
int npy_half_isinf(npy_half h);             /* 判断半精度浮点数是否为 无穷 */
int npy_half_isfinite(npy_half h);          /* 判断半精度浮点数是否有限 */
int npy_half_signbit(npy_half h);           /* 判断半精度浮点数的符号位 */
npy_half npy_half_copysign(npy_half x, npy_half y);   /* 将 x 的符号位设置为 y 的符号位 */
npy_half npy_half_spacing(npy_half h);      /* 返回相邻两个半精度浮点数之间的距离 */
npy_half npy_half_nextafter(npy_half x, npy_half y); /* 返回在 x 和 y 之间且与 x 最接近的半精度浮点数 */
npy_half npy_half_divmod(npy_half x, npy_half y, npy_half *modulus); /* 返回 x/y 的商和余数 */

/*
 * Half-precision constants
 */

#define NPY_HALF_ZERO   (0x0000u)   /* 半精度零 */
#define NPY_HALF_PZERO  (0x0000u)   /* 正半精度零 */
#define NPY_HALF_NZERO  (0x8000u)   /* 负半精度零 */
#define NPY_HALF_ONE    (0x3c00u)   /* 半精度一 */
#define NPY_HALF_NEGONE (0xbc00u)   /* 半精度负一 */
#define NPY_HALF_PINF   (0x7c00u)   /* 正无穷 */
#define NPY_HALF_NINF   (0xfc00u)   /* 负无穷 */
#define NPY_HALF_NAN    (0x7e00u)   /* NaN */

#define NPY_MAX_HALF    (0x7bffu)   /* 最大半精度浮点数 */

/*
 * Bit-level conversions
 */

npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f);   /* 将单精度浮点数位表示转换为半精度浮点数位表示 */
npy_uint16 npy_doublebits_to_halfbits(npy_uint64 d);  /* 将双精度浮点数位表示转换为半精度浮点数位表示 */
npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h);   /* 将半精度浮点数位表示转换为单精度浮点数位表示 */
npy_uint64 npy_halfbits_to_doublebits(npy_uint16 h);  /* 将半精度浮点数位表示转换为双精度浮点数位表示 */

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_ */
```