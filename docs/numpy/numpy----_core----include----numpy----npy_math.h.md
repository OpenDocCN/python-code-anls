# `.\numpy\numpy\_core\include\numpy\npy_math.h`

```
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_MATH_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_MATH_H_

#include <numpy/npy_common.h>

#include <math.h>

/* 通过在适当时添加 static inline 修饰符到 npy_math 函数定义中，
   编译器有机会进行优化 */
#if NPY_INLINE_MATH
#define NPY_INPLACE static inline
#else
#define NPY_INPLACE
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 定义一个宏，返回两个数中较大的数
#define PyArray_MAX(a,b) (((a)>(b))?(a):(b))
// 定义一个宏，返回两个数中较小的数
#define PyArray_MIN(a,b) (((a)<(b))?(a):(b))

/*
 * NAN 和 INFINITY 的宏定义（NAN 的行为与 glibc 一致，INFINITY 的行为与 C99 一致）
 *
 * XXX: 应测试平台上是否可用 INFINITY 和 NAN
 */
// 返回正无穷大的浮点数
static inline float __npy_inff(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7f800000UL};
    return __bint.__f;
}

// 返回 NaN 的浮点数
static inline float __npy_nanf(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7fc00000UL};
    return __bint.__f;
}

// 返回正零的浮点数
static inline float __npy_pzerof(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x00000000UL};
    return __bint.__f;
}

// 返回负零的浮点数
static inline float __npy_nzerof(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x80000000UL};
    return __bint.__f;
}

// 定义浮点数的正无穷大宏
#define NPY_INFINITYF __npy_inff()
// 定义浮点数的 NaN 宏
#define NPY_NANF __npy_nanf()
// 定义浮点数的正零宏
#define NPY_PZEROF __npy_pzerof()
// 定义浮点数的负零宏
#define NPY_NZEROF __npy_nzerof()

// 定义双精度浮点数的正无穷大宏
#define NPY_INFINITY ((npy_double)NPY_INFINITYF)
// 定义双精度浮点数的 NaN 宏
#define NPY_NAN ((npy_double)NPY_NANF)
// 定义双精度浮点数的正零宏
#define NPY_PZERO ((npy_double)NPY_PZEROF)
// 定义双精度浮点数的负零宏
#define NPY_NZERO ((npy_double)NPY_NZEROF)

// 定义长双精度浮点数的正无穷大宏
#define NPY_INFINITYL ((npy_longdouble)NPY_INFINITYF)
// 定义长双精度浮点数的 NaN 宏
#define NPY_NANL ((npy_longdouble)NPY_NANF)
// 定义长双精度浮点数的正零宏
#define NPY_PZEROL ((npy_longdouble)NPY_PZEROF)
// 定义长双精度浮点数的负零宏
#define NPY_NZEROL ((npy_longdouble)NPY_NZEROF)

/*
 * 一些有用的常量
 */
// 自然常数 e
#define NPY_E         2.718281828459045235360287471352662498  /* e */
// 以 2 为底 e 的对数
#define NPY_LOG2E     1.442695040888963407359924681001892137  /* log_2 e */
// 以 10 为底 e 的对数
#define NPY_LOG10E    0.434294481903251827651128918916605082  /* log_10 e */
// 自然对数 e 的底数
#define NPY_LOGE2     0.693147180559945309417232121458176568  /* log_e 2 */
// 自然对数 e 的底数
#define NPY_LOGE10    2.302585092994045684017991454684364208  /* log_e 10 */
// 圆周率 π
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */
// π 的一半
#define NPY_PI_2      1.570796326794896619231321691639751442  /* pi/2 */
// π 的四分之一
#define NPY_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
// 1/pi
#define NPY_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
// 2/pi
#define NPY_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
// 欧拉常数 γ
#define NPY_EULER     0.577215664901532860606512090082402431  /* Euler constant */
// 开平方根的 2 的值
#define NPY_SQRT2     1.414213562373095048801688724209698079  /* sqrt(2) */
// 1/sqrt(2)
#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

// 单精度浮点数的自然常数 e
#define NPY_Ef        2.718281828459045235360287471352662498F /* e */
// 单精度浮点数以 2 为底 e 的对数
#define NPY_LOG2Ef    1.442695040888963407359924681001892137F /* log_2 e */
// 单精度浮点数以 10 为底 e 的对数
#define NPY_LOG10Ef   0.434294481903251827651128918916605082F /* log_10 e */
/*
 * 定义常量：浮点数的对数值和数学常数
 */
#define NPY_LOGE2f    0.693147180559945309417232121458176568F /* log_e 2 */
#define NPY_LOGE10f   2.302585092994045684017991454684364208F /* log_e 10 */
#define NPY_PIf       3.141592653589793238462643383279502884F /* pi */
#define NPY_PI_2f     1.570796326794896619231321691639751442F /* pi/2 */
#define NPY_PI_4f     0.785398163397448309615660845819875721F /* pi/4 */
#define NPY_1_PIf     0.318309886183790671537767526745028724F /* 1/pi */
#define NPY_2_PIf     0.636619772367581343075535053490057448F /* 2/pi */
#define NPY_EULERf    0.577215664901532860606512090082402431F /* Euler constant */
#define NPY_SQRT2f    1.414213562373095048801688724209698079F /* sqrt(2) */
#define NPY_SQRT1_2f  0.707106781186547524400844362104849039F /* 1/sqrt(2) */

#define NPY_El        2.718281828459045235360287471352662498L /* e */
#define NPY_LOG2El    1.442695040888963407359924681001892137L /* log_2 e */
#define NPY_LOG10El   0.434294481903251827651128918916605082L /* log_10 e */
#define NPY_LOGE2l    0.693147180559945309417232121458176568L /* log_e 2 */
#define NPY_LOGE10l   2.302585092994045684017991454684364208L /* log_e 10 */
#define NPY_PIl       3.141592653589793238462643383279502884L /* pi */
#define NPY_PI_2l     1.570796326794896619231321691639751442L /* pi/2 */
#define NPY_PI_4l     0.785398163397448309615660845819875721L /* pi/4 */
#define NPY_1_PIl     0.318309886183790671537767526745028724L /* 1/pi */
#define NPY_2_PIl     0.636619772367581343075535053490057448L /* 2/pi */
#define NPY_EULERl    0.577215664901532860606512090082402431L /* Euler constant */
#define NPY_SQRT2l    1.414213562373095048801688724209698079L /* sqrt(2) */
#define NPY_SQRT1_2l  0.707106781186547524400844362104849039L /* 1/sqrt(2) */

/*
 * 整数函数声明
 */
NPY_INPLACE npy_uint npy_gcdu(npy_uint a, npy_uint b);  // 无符号整数最大公约数
NPY_INPLACE npy_uint npy_lcmu(npy_uint a, npy_uint b);  // 无符号整数最小公倍数
NPY_INPLACE npy_ulong npy_gcdul(npy_ulong a, npy_ulong b);  // 无符号长整型最大公约数
NPY_INPLACE npy_ulong npy_lcmul(npy_ulong a, npy_ulong b);  // 无符号长整型最小公倍数
NPY_INPLACE npy_ulonglong npy_gcdull(npy_ulonglong a, npy_ulonglong b);  // 无符号长长整型最大公约数
NPY_INPLACE npy_ulonglong npy_lcmull(npy_ulonglong a, npy_ulonglong b);  // 无符号长长整型最小公倍数

NPY_INPLACE npy_int npy_gcd(npy_int a, npy_int b);  // 整型最大公约数
NPY_INPLACE npy_int npy_lcm(npy_int a, npy_int b);  // 整型最小公倍数
NPY_INPLACE npy_long npy_gcdl(npy_long a, npy_long b);  // 长整型最大公约数
NPY_INPLACE npy_long npy_lcml(npy_long a, npy_long b);  // 长整型最小公倍数
NPY_INPLACE npy_longlong npy_gcdll(npy_longlong a, npy_longlong b);  // 长长整型最大公约数
NPY_INPLACE npy_longlong npy_lcmll(npy_longlong a, npy_longlong b);  // 长长整型最小公倍数

NPY_INPLACE npy_ubyte npy_rshiftuhh(npy_ubyte a, npy_ubyte b);  // 无符号字节右移
NPY_INPLACE npy_ubyte npy_lshiftuhh(npy_ubyte a, npy_ubyte b);  // 无符号字节左移
NPY_INPLACE npy_ushort npy_rshiftuh(npy_ushort a, npy_ushort b);  // 无符号短整型右移
NPY_INPLACE npy_ushort npy_lshiftuh(npy_ushort a, npy_ushort b);  // 无符号短整型左移
NPY_INPLACE npy_uint npy_rshiftu(npy_uint a, npy_uint b);  // 无符号整型右移
NPY_INPLACE npy_uint npy_lshiftu(npy_uint a, npy_uint b);  // 无符号整型左移
NPY_INPLACE npy_ulong npy_rshiftul(npy_ulong a, npy_ulong b);  // 无符号长整型右移
NPY_INPLACE npy_ulong npy_lshiftul(npy_ulong a, npy_ulong b);  // 无符号长整型左移
/*
 * NPY_INPLACE 宏定义的函数声明
 * 这些函数执行无符号整数的位移操作
 */
NPY_INPLACE npy_ulonglong npy_rshiftull(npy_ulonglong a, npy_ulonglong b);
NPY_INPLACE npy_ulonglong npy_lshiftull(npy_ulonglong a, npy_ulonglong b);

NPY_INPLACE npy_byte npy_rshifthh(npy_byte a, npy_byte b);
NPY_INPLACE npy_byte npy_lshifthh(npy_byte a, npy_byte b);
NPY_INPLACE npy_short npy_rshifth(npy_short a, npy_short b);
NPY_INPLACE npy_short npy_lshifth(npy_short a, npy_short b);
NPY_INPLACE npy_int npy_rshift(npy_int a, npy_int b);
NPY_INPLACE npy_int npy_lshift(npy_int a, npy_int b);
NPY_INPLACE npy_long npy_rshiftl(npy_long a, npy_long b);
NPY_INPLACE npy_long npy_lshiftl(npy_long a, npy_long b);
NPY_INPLACE npy_longlong npy_rshiftll(npy_longlong a, npy_longlong b);
NPY_INPLACE npy_longlong npy_lshiftll(npy_longlong a, npy_longlong b);

/*
 * NPY_INPLACE 宏定义的函数声明
 * 这些函数执行无符号整数和字节的位操作
 */
NPY_INPLACE uint8_t npy_popcountuhh(npy_ubyte a);
NPY_INPLACE uint8_t npy_popcountuh(npy_ushort a);
NPY_INPLACE uint8_t npy_popcountu(npy_uint a);
NPY_INPLACE uint8_t npy_popcountul(npy_ulong a);
NPY_INPLACE uint8_t npy_popcountull(npy_ulonglong a);
NPY_INPLACE uint8_t npy_popcounthh(npy_byte a);
NPY_INPLACE uint8_t npy_popcounth(npy_short a);
NPY_INPLACE uint8_t npy_popcount(npy_int a);
NPY_INPLACE uint8_t npy_popcountl(npy_long a);
NPY_INPLACE uint8_t npy_popcountll(npy_longlong a);

/*
 * C99 标准下的双精度数学函数声明，部分可能需要修复或者应该列入阻止列表
 */
NPY_INPLACE double npy_sin(double x);
NPY_INPLACE double npy_cos(double x);
NPY_INPLACE double npy_tan(double x);
NPY_INPLACE double npy_hypot(double x, double y);
NPY_INPLACE double npy_log2(double x);
NPY_INPLACE double npy_atan2(double x, double y);

/* 
 * 强制要求使用的 C99 标准下的双精度数学函数声明，不应列入阻止列表或需要修复
 * 这些函数为了向后兼容而定义，但应考虑在某些时候废弃它们
 */
#define npy_sinh sinh
#define npy_cosh cosh
#define npy_tanh tanh
#define npy_asin asin
#define npy_acos acos
#define npy_atan atan
#define npy_log log
#define npy_log10 log10
#define npy_cbrt cbrt
#define npy_fabs fabs
#define npy_ceil ceil
#define npy_fmod fmod
#define npy_floor floor
#define npy_expm1 expm1
#define npy_log1p log1p
#define npy_acosh acosh
#define npy_asinh asinh
#define npy_atanh atanh
#define npy_rint rint
#define npy_trunc trunc
#define npy_exp2 exp2
#define npy_frexp frexp
#define npy_ldexp ldexp
#define npy_copysign copysign
#define npy_exp exp
#define npy_sqrt sqrt
#define npy_pow pow
#define npy_modf modf
#define npy_nextafter nextafter

/*
 * 计算双精度数 x 的间距
 */
double npy_spacing(double x);

/*
 * IEEE 754 浮点数处理
 */

/* 在紧凑循环中使用内建函数以避免函数调用
 * 仅当 npy_config.h 可用时（即 NumPy 的自建版本）才可用
 */
#ifdef HAVE___BUILTIN_ISNAN
    #define npy_isnan(x) __builtin_isnan(x)
#else
    #define npy_isnan(x) isnan(x)
#endif

/* 仅当 npy_config.h 可用时（即 NumPy 的自建版本）才可用 */
#ifdef HAVE___BUILTIN_ISFINITE
    #define npy_isfinite(x) __builtin_isfinite(x)
#else
    #define npy_isfinite(x) isfinite((x))
#endif

/* 仅当 npy_config.h 可用时（即 NumPy 的自建版本）才可用 */
#ifdef HAVE___BUILTIN_ISINF
    # 定义一个宏 `npy_isinf(x)`，用于检查 x 是否为无穷大，基于编译器内置的 `__builtin_isinf(x)` 函数。
    #define npy_isinf(x) __builtin_isinf(x)
#else
    #define npy_isinf(x) isinf((x))
#endif

#define npy_signbit(x) signbit((x))

/*
 * float C99 math funcs that need fixups or are blocklist-able
 */
# 定义了一系列用于单精度浮点数的 C99 标准数学函数，需要修复或者可能会被阻止的函数
NPY_INPLACE float npy_sinf(float x);
NPY_INPLACE float npy_cosf(float x);
NPY_INPLACE float npy_tanf(float x);
NPY_INPLACE float npy_expf(float x);
NPY_INPLACE float npy_sqrtf(float x);
NPY_INPLACE float npy_hypotf(float x, float y);
NPY_INPLACE float npy_log2f(float x);
NPY_INPLACE float npy_atan2f(float x, float y);
NPY_INPLACE float npy_powf(float x, float y);
NPY_INPLACE float npy_modff(float x, float* y);

/* Mandatory C99 float math funcs, no blocklisting or fixups */
/* defined for legacy reasons, should be deprecated at some point */
# C99 标准要求的必需单精度浮点数数学函数，无需阻止或修复
#define npy_sinhf sinhf
#define npy_coshf coshf
#define npy_tanhf tanhf
#define npy_asinf asinf
#define npy_acosf acosf
#define npy_atanf atanf
#define npy_logf logf
#define npy_log10f log10f
#define npy_cbrtf cbrtf
#define npy_fabsf fabsf
#define npy_ceilf ceilf
#define npy_fmodf fmodf
#define npy_floorf floorf
#define npy_expm1f expm1f
#define npy_log1pf log1pf
#define npy_asinhf asinhf
#define npy_acoshf acoshf
#define npy_atanhf atanhf
#define npy_rintf rintf
#define npy_truncf truncf
#define npy_exp2f exp2f
#define npy_frexpf frexpf
#define npy_ldexpf ldexpf
#define npy_copysignf copysignf
#define npy_nextafterf nextafterf

float npy_spacingf(float x);

/*
 * long double C99 double math funcs that need fixups or are blocklist-able
 */
# 定义了一系列用于双精度长双精度浮点数的 C99 标准数学函数，需要修复或者可能会被阻止的函数
NPY_INPLACE npy_longdouble npy_sinl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_cosl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_tanl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_expl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_sqrtl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_hypotl(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_log2l(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_atan2l(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_powl(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_modfl(npy_longdouble x, npy_longdouble* y);

/* Mandatory C99 double math funcs, no blocklisting or fixups */
/* defined for legacy reasons, should be deprecated at some point */
# C99 标准要求的必需双精度浮点数数学函数，无需阻止或修复
#define npy_sinhl sinhl
#define npy_coshl coshl
#define npy_tanhl tanhl
#define npy_fabsl fabsl
#define npy_floorl floorl
#define npy_ceill ceill
#define npy_rintl rintl
#define npy_truncl truncl
#define npy_cbrtl cbrtl
#define npy_log10l log10l
#define npy_logl logl
#define npy_expm1l expm1l
#define npy_asinl asinl
#define npy_acosl acosl
#define npy_atanl atanl
#define npy_asinhl asinhl
#define npy_acoshl acoshl
#define npy_atanhl atanhl
#define npy_log1pl log1pl
#define npy_exp2l exp2l
#define npy_fmodl fmodl
#define npy_frexpl frexpl
#define npy_ldexpl ldexpl
#define npy_copysignl copysignl
#define npy_nextafterl nextafterl

npy_longdouble npy_spacingl(npy_longdouble x);

/*
 * Non standard functions
 */
# 非标准函数声明
NPY_INPLACE double npy_deg2rad(double x);
/*
 * Function declarations for mathematical operations on various data types
 */

NPY_INPLACE double npy_rad2deg(double x);
// Converts radians to degrees

NPY_INPLACE double npy_logaddexp(double x, double y);
// Computes log(exp(x) + exp(y)) without overflow

NPY_INPLACE double npy_logaddexp2(double x, double y);
// Computes log2(2^x + 2^y) without overflow

NPY_INPLACE double npy_divmod(double x, double y, double *modulus);
// Computes division x / y and stores remainder in 'modulus'

NPY_INPLACE double npy_heaviside(double x, double h0);
// Heaviside step function: 0 if x < 0, h0 if x == 0, 1 if x > 0

NPY_INPLACE float npy_deg2radf(float x);
// Converts degrees to radians (single precision)

NPY_INPLACE float npy_rad2degf(float x);
// Converts radians to degrees (single precision)

NPY_INPLACE float npy_logaddexpf(float x, float y);
// Computes log(exp(x) + exp(y)) in single precision without overflow

NPY_INPLACE float npy_logaddexp2f(float x, float y);
// Computes log2(2^x + 2^y) in single precision without overflow

NPY_INPLACE float npy_divmodf(float x, float y, float *modulus);
// Computes division x / y in single precision and stores remainder in 'modulus'

NPY_INPLACE float npy_heavisidef(float x, float h0);
// Heaviside step function in single precision: 0 if x < 0, h0 if x == 0, 1 if x > 0

NPY_INPLACE npy_longdouble npy_deg2radl(npy_longdouble x);
// Converts degrees to radians (extended precision)

NPY_INPLACE npy_longdouble npy_rad2degl(npy_longdouble x);
// Converts radians to degrees (extended precision)

NPY_INPLACE npy_longdouble npy_logaddexpl(npy_longdouble x, npy_longdouble y);
// Computes log(exp(x) + exp(y)) in extended precision without overflow

NPY_INPLACE npy_longdouble npy_logaddexp2l(npy_longdouble x, npy_longdouble y);
// Computes log2(2^x + 2^y) in extended precision without overflow

NPY_INPLACE npy_longdouble npy_divmodl(npy_longdouble x, npy_longdouble y,
                           npy_longdouble *modulus);
// Computes division x / y in extended precision and stores remainder in 'modulus'

NPY_INPLACE npy_longdouble npy_heavisidel(npy_longdouble x, npy_longdouble h0);
// Heaviside step function in extended precision: 0 if x < 0, h0 if x == 0, 1 if x > 0

#define npy_degrees npy_rad2deg
// Macro: alias for npy_rad2deg function

#define npy_degreesf npy_rad2degf
// Macro: alias for npy_rad2degf function

#define npy_degreesl npy_rad2degl
// Macro: alias for npy_rad2degl function

#define npy_radians npy_deg2rad
// Macro: alias for npy_deg2rad function

#define npy_radiansf npy_deg2radf
// Macro: alias for npy_deg2radf function

#define npy_radiansl npy_deg2radl
// Macro: alias for npy_deg2radl function

/*
 * Complex number operations
 */

static inline double npy_creal(const npy_cdouble z)
{
    return ((double *) &z)[0];
}
// Returns the real part of a complex double number

static inline void npy_csetreal(npy_cdouble *z, const double r)
{
    ((double *) z)[0] = r;
}
// Sets the real part of a complex double number to 'r'

static inline double npy_cimag(const npy_cdouble z)
{
    return ((double *) &z)[1];
}
// Returns the imaginary part of a complex double number

static inline void npy_csetimag(npy_cdouble *z, const double i)
{
    ((double *) z)[1] = i;
}
// Sets the imaginary part of a complex double number to 'i'

static inline float npy_crealf(const npy_cfloat z)
{
    return ((float *) &z)[0];
}
// Returns the real part of a complex float number

static inline void npy_csetrealf(npy_cfloat *z, const float r)
{
    ((float *) z)[0] = r;
}
// Sets the real part of a complex float number to 'r'

static inline float npy_cimagf(const npy_cfloat z)
{
    return ((float *) &z)[1];
}
// Returns the imaginary part of a complex float number

static inline void npy_csetimagf(npy_cfloat *z, const float i)
{
    ((float *) z)[1] = i;
}
// Sets the imaginary part of a complex float number to 'i'

static inline npy_longdouble npy_creall(const npy_clongdouble z)
{
    return ((longdouble_t *) &z)[0];
}
// Returns the real part of a complex long double number

static inline void npy_csetreall(npy_clongdouble *z, const longdouble_t r)
{
    ((longdouble_t *) z)[0] = r;
}
// Sets the real part of a complex long double number to 'r'

static inline npy_longdouble npy_cimagl(const npy_clongdouble z)
{
    return ((longdouble_t *) &z)[1];
}
// Returns the imaginary part of a complex long double number

static inline void npy_csetimagl(npy_clongdouble *z, const longdouble_t i)
{
    ((longdouble_t *) z)[1] = i;
}
// Sets the imaginary part of a complex long double number to 'i'

#define NPY_CSETREAL(z, r) npy_csetreal(z, r)
// Macro: sets the real part of a complex number

#define NPY_CSETIMAG(z, i) npy_csetimag(z, i)
// Macro: sets the imaginary part of a complex number

#define NPY_CSETREALF(z, r) npy_csetrealf(z, r)
// Macro: sets the real part of a complex float number

#define NPY_CSETIMAGF(z, i) npy_csetimagf(z, i)
// Macro: sets the imaginary part of a complex float number

#define NPY_CSETREALL(z, r) npy_csetreall(z, r)
// Macro: sets the real part of a complex long double number

#define NPY_CSETIMAGL(z, i) npy_csetimagl(z, i)
// Macro: sets the imaginary part of a complex long double number

static inline npy_cdouble npy_cpack(double x, double y)
{
    npy_cdouble z;
    npy_csetreal(&z, x);
    npy_csetimag(&z, y);
    return z;
}
// Packs real and imaginary parts into a complex double number

static inline npy_cfloat npy_cpackf(float x, float y)
{
    npy_cfloat z;
    npy_csetrealf(&z, x);
    # 调用 npy_csetimagf 函数，将变量 y 的值设置为 z 的虚部
    npy_csetimagf(&z, y);
    # 返回变量 z 的值作为函数的结果
    return z;
/*
 * Single precision complex number representation functions
 * using floating point types.
 */

/*
 * Packs two long double values into a complex long double value.
 */
static inline npy_clongdouble npy_cpackl(npy_longdouble x, npy_longdouble y)
{
    npy_clongdouble z;
    npy_csetreall(&z, x);   // 设置 z 的实部为 x
    npy_csetimagl(&z, y);   // 设置 z 的虚部为 y
    return z;               // 返回复数 z
}

/*
 * Double precision complex functions
 */
double npy_cabs(npy_cdouble z);      // 返回复数 z 的模
double npy_carg(npy_cdouble z);      // 返回复数 z 的幅角

npy_cdouble npy_cexp(npy_cdouble z); // 返回 e^z
npy_cdouble npy_clog(npy_cdouble z); // 返回 ln(z)
npy_cdouble npy_cpow(npy_cdouble x, npy_cdouble y); // 返回 x^y

npy_cdouble npy_csqrt(npy_cdouble z);    // 返回 sqrt(z)

npy_cdouble npy_ccos(npy_cdouble z);     // 返回 cos(z)
npy_cdouble npy_csin(npy_cdouble z);     // 返回 sin(z)
npy_cdouble npy_ctan(npy_cdouble z);     // 返回 tan(z)

npy_cdouble npy_ccosh(npy_cdouble z);    // 返回 cosh(z)
npy_cdouble npy_csinh(npy_cdouble z);    // 返回 sinh(z)
npy_cdouble npy_ctanh(npy_cdouble z);    // 返回 tanh(z)

npy_cdouble npy_cacos(npy_cdouble z);    // 返回 arccos(z)
npy_cdouble npy_casin(npy_cdouble z);    // 返回 arcsin(z)
npy_cdouble npy_catan(npy_cdouble z);    // 返回 arctan(z)

npy_cdouble npy_cacosh(npy_cdouble z);   // 返回 arccosh(z)
npy_cdouble npy_casinh(npy_cdouble z);   // 返回 arcsinh(z)
npy_cdouble npy_catanh(npy_cdouble z);   // 返回 arctanh(z)

/*
 * Single precision complex functions
 */
float npy_cabsf(npy_cfloat z);      // 返回复数 z 的模
float npy_cargf(npy_cfloat z);      // 返回复数 z 的幅角

npy_cfloat npy_cexpf(npy_cfloat z); // 返回 e^z
npy_cfloat npy_clogf(npy_cfloat z); // 返回 ln(z)
npy_cfloat npy_cpowf(npy_cfloat x, npy_cfloat y); // 返回 x^y

npy_cfloat npy_csqrtf(npy_cfloat z);    // 返回 sqrt(z)

npy_cfloat npy_ccosf(npy_cfloat z);     // 返回 cos(z)
npy_cfloat npy_csinf(npy_cfloat z);     // 返回 sin(z)
npy_cfloat npy_ctanf(npy_cfloat z);     // 返回 tan(z)

npy_cfloat npy_ccoshf(npy_cfloat z);    // 返回 cosh(z)
npy_cfloat npy_csinhf(npy_cfloat z);    // 返回 sinh(z)
npy_cfloat npy_ctanhf(npy_cfloat z);    // 返回 tanh(z)

npy_cfloat npy_cacosf(npy_cfloat z);    // 返回 arccos(z)
npy_cfloat npy_casinf(npy_cfloat z);    // 返回 arcsin(z)
npy_cfloat npy_catanf(npy_cfloat z);    // 返回 arctan(z)

npy_cfloat npy_cacoshf(npy_cfloat z);   // 返回 arccosh(z)
npy_cfloat npy_casinhf(npy_cfloat z);   // 返回 arcsinh(z)
npy_cfloat npy_catanhf(npy_cfloat z);   // 返回 arctanh(z)

/*
 * Extended precision complex functions
 */
npy_longdouble npy_cabsl(npy_clongdouble z);      // 返回复数 z 的模
npy_longdouble npy_cargl(npy_clongdouble z);      // 返回复数 z 的幅角

npy_clongdouble npy_cexpl(npy_clongdouble z);     // 返回 e^z
npy_clongdouble npy_clogl(npy_clongdouble z);     // 返回 ln(z)
npy_clongdouble npy_cpowl(npy_clongdouble x, npy_clongdouble y); // 返回 x^y

npy_clongdouble npy_csqrtl(npy_clongdouble z);    // 返回 sqrt(z)

npy_clongdouble npy_ccosl(npy_clongdouble z);     // 返回 cos(z)
npy_clongdouble npy_csinl(npy_clongdouble z);     // 返回 sin(z)
npy_clongdouble npy_ctanl(npy_clongdouble z);     // 返回 tan(z)

npy_clongdouble npy_ccoshl(npy_clongdouble z);    // 返回 cosh(z)
npy_clongdouble npy_csinhl(npy_clongdouble z);    // 返回 sinh(z)
npy_clongdouble npy_ctanhl(npy_clongdouble z);    // 返回 tanh(z)

npy_clongdouble npy_cacosl(npy_clongdouble z);    // 返回 arccos(z)
npy_clongdouble npy_casinl(npy_clongdouble z);    // 返回 arcsin(z)
npy_clongdouble npy_catanl(npy_clongdouble z);    // 返回 arctan(z)

npy_clongdouble npy_cacoshl(npy_clongdouble z);   // 返回 arccosh(z)
npy_clongdouble npy_casinhl(npy_clongdouble z);   // 返回 arcsinh(z)
npy_clongdouble npy_catanhl(npy_clongdouble z);   // 返回 arctanh(z)

/*
 * Functions that set the floating point error
 * status word.
 */

/*
 * platform-dependent code translates floating point
 * status to an integer sum of these values
 */
#define NPY_FPE_DIVIDEBYZERO  1   // 浮点数除以零错误
#define NPY_FPE_OVERFLOW      2   // 浮点数溢出
#define NPY_FPE_UNDERFLOW     4   // 浮点数下溢
#define NPY_FPE_INVALID       8   // 无效浮点数操作

int npy_clear_floatstatus_barrier(char*);    // 清除浮点数错误状态屏障
int npy_get_floatstatus_barrier(char*);      // 获取浮点数错误状态屏障
/*
 * use caution with these - clang and gcc8.1 are known to reorder calls
 * to this form of the function which can defeat the check. The _barrier
 * form of the call is preferable, where the argument is
 * (char*)&local_variable
 */
/* 
 * 清除浮点状态的函数声明，调用时需要谨慎，因为clang和gcc8.1已知会重新排序这种函数调用，
 * 这可能会破坏检查。更可取的是使用带_barrier后缀的调用形式，其中参数为
 * (char*)&local_variable
 */
int npy_clear_floatstatus(void);
/* 获取当前浮点状态的函数声明 */
int npy_get_floatstatus(void);

/* 设置浮点状态为除以零错误 */
void npy_set_floatstatus_divbyzero(void);
/* 设置浮点状态为溢出 */
void npy_set_floatstatus_overflow(void);
/* 设置浮点状态为下溢 */
void npy_set_floatstatus_underflow(void);
/* 设置浮点状态为无效操作 */
void npy_set_floatstatus_invalid(void);

#ifdef __cplusplus
}
#endif

/* 如果启用了内联数学操作，包含内联数学头文件 */
#if NPY_INLINE_MATH
#include "npy_math_internal.h"
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_MATH_H_ */
```