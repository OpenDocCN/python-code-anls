# `.\numpy\numpy\_core\src\common\npy_config.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_

#include "config.h"
#include "npy_cpu_dispatch.h" // 包含了定义了 NPY_HAVE_[CPU features] 的头文件
#include "numpy/numpyconfig.h"
#include "numpy/utils.h"
#include "numpy/npy_os.h"

/* blocklist */

/* 在 z/OS 上禁用已知有问题的函数 */
#if defined (__MVS__)

#define NPY_BLOCK_POWF
#define NPY_BLOCK_EXPF
#undef HAVE___THREAD

#endif

/* 在 MinGW 下禁用已知有问题的 MS 数学函数 */
#if defined(__MINGW32_VERSION)

#define NPY_BLOCK_ATAN2
#define NPY_BLOCK_ATAN2F
#define NPY_BLOCK_ATAN2L

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif

/* 在 MSVC 下禁用已知有问题的数学函数 */
#if defined(_MSC_VER)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CSQRT
#undef HAVE_CSQRTF
#undef HAVE_CSQRTL
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif

/* MSVC _hypot 在 32 位模式下影响浮点精度模式，参见 gh-9567 */
#if defined(_MSC_VER) && !defined(_WIN64)

#undef HAVE_CABS
#undef HAVE_CABSF
#undef HAVE_CABSL

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif

/* Intel C 编译器在 Windows 上对 64 位 longdouble 使用 POW */
#if defined(_MSC_VER) && defined(__INTEL_COMPILER)
#if NPY_SIZEOF_LONGDOUBLE == 8
#define NPY_BLOCK_POWL
#endif
#endif /* defined(_MSC_VER) && defined(__INTEL_COMPILER) */

/* powl 在 OS X 上会产生零除警告，参见 gh-8307 */
#if defined(NPY_OS_DARWIN)
#define NPY_BLOCK_POWL
#endif

#ifdef __CYGWIN__
/* 由于精度丢失，禁用一些函数 */
#undef HAVE_CASINHL
#undef HAVE_CASINH
#undef HAVE_CASINHF

/* 由于精度丢失，禁用一些函数 */
#undef HAVE_CATANHL
#undef HAVE_CATANH
#undef HAVE_CATANHF

/* 由于分支切割，禁用一些函数 */
#undef HAVE_CATANL
#undef HAVE_CATAN
#undef HAVE_CATANF

/* 由于分支切割，禁用一些函数 */
#undef HAVE_CACOSHF
#undef HAVE_CACOSH

/* 由于分支切割，禁用一些函数 */
#undef HAVE_CSQRTF
#undef HAVE_CSQRT

/* 由于分支切割和精度丢失，禁用一些函数 */
#undef HAVE_CASINF
#undef HAVE_CASIN
#undef HAVE_CASINL

/* 由于分支切割，禁用一些函数 */
#undef HAVE_CACOSF
#undef HAVE_CACOS

/* log2(exp2(i)) 会有几个 eps 的偏差 */
#define NPY_BLOCK_LOG2

/* np.power(..., dtype=np.complex256) 不会报告溢出 */
#undef HAVE_CPOWL
#undef HAVE_CEXPL

/*
 * Cygwin 使用 newlib，其复数对数函数实现比较简单。
 */
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL

#include <cygwin/version.h>
#if CYGWIN_VERSION_DLL_MAJOR < 3003
// 不支持低于 3.3 版本的 Cygwin，提示用户更新
#error cygwin < 3.3 not supported, please update
#endif
#endif

/* 禁用有问题的 GNU 三角函数 */
#if defined(HAVE_FEATURES_H)
#include <features.h>

#if defined(__GLIBC__)
#if !__GLIBC_PREREQ(2, 18)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
/*
 * 定义部分宏以确保在 GLIBC 2.18 以下的版本中不使用一些复杂数学函数，
 * 因为这些函数可能在旧版本中不存在或者有不兼容的实现。
 * 另外，针对 musl libc 进行类似的宏定义，这是一个独立的 C 库。
 */

#undef HAVE_CASINHF  // 取消定义复数反双曲正弦函数的高精度浮点数版本
#undef HAVE_CASINHL  // 取消定义复数反双曲正弦函数的高精度长双浮点数版本
#undef HAVE_CATAN    // 取消定义复数反正切函数
#undef HAVE_CATANF   // 取消定义复数反正切函数的浮点数版本
#undef HAVE_CATANL   // 取消定义复数反正切函数的长双浮点数版本
#undef HAVE_CATANH   // 取消定义复数反双曲正切函数
#undef HAVE_CATANHF  // 取消定义复数反双曲正切函数的浮点数版本
#undef HAVE_CATANHL  // 取消定义复数反双曲正切函数的长双浮点数版本
#undef HAVE_CACOS    // 取消定义复数反余弦函数
#undef HAVE_CACOSF   // 取消定义复数反余弦函数的浮点数版本
#undef HAVE_CACOSL   // 取消定义复数反余弦函数的长双浮点数版本
#undef HAVE_CACOSH   // 取消定义复数反双曲余弦函数
#undef HAVE_CACOSHF  // 取消定义复数反双曲余弦函数的浮点数版本
#undef HAVE_CACOSHL  // 取消定义复数反双曲余弦函数的长双浮点数版本

#endif  /* __GLIBC_PREREQ(2, 18) */

#else   /* defined(__GLIBC) */

/* 
 * 如果不是使用 GLIBC 标准 C 库，可能是使用 musl libc，这是另一个独立的 C 库。
 * 在这种情况下，取消定义一些复杂数学函数，以避免潜在的兼容性问题。
 */

#undef HAVE_CASIN    // 取消定义复数反正弦函数
#undef HAVE_CASINF   // 取消定义复数反正弦函数的浮点数版本
#undef HAVE_CASINL   // 取消定义复数反正弦函数的长双浮点数版本
#undef HAVE_CASINH   // 取消定义复数反双曲正弦函数
#undef HAVE_CASINHF  // 取消定义复数反双曲正弦函数的浮点数版本
#undef HAVE_CASINHL  // 取消定义复数反双曲正弦函数的长双浮点数版本
#undef HAVE_CATAN    // 取消定义复数反正切函数
#undef HAVE_CATANF   // 取消定义复数反正切函数的浮点数版本
#undef HAVE_CATANL   // 取消定义复数反正切函数的长双浮点数版本
#undef HAVE_CATANH   // 取消定义复数反双曲正切函数
#undef HAVE_CATANHF  // 取消定义复数反双曲正切函数的浮点数版本
#undef HAVE_CATANHL  // 取消定义复数反双曲正切函数的长双浮点数版本
#undef HAVE_CACOS    // 取消定义复数反余弦函数
#undef HAVE_CACOSF   // 取消定义复数反余弦函数的浮点数版本
#undef HAVE_CACOSL   // 取消定义复数反余弦函数的长双浮点数版本
#undef HAVE_CACOSH   // 取消定义复数反双曲余弦函数
#undef HAVE_CACOSHF  // 取消定义复数反双曲余弦函数的浮点数版本
#undef HAVE_CACOSHL  // 取消定义复数反双曲余弦函数的长双浮点数版本

/*
 * musl libc 中的 clog 函数对某些输入具有低精度。从 MUSL 1.2.5 版本开始，
 * clog.c 中的第一个注释是 "// FIXME"。
 * 参考 https://github.com/numpy/numpy/pull/24416#issuecomment-1678208628
 * 和 https://github.com/numpy/numpy/pull/24448
 * 这里取消定义复数对数函数及其浮点数版本，可能是为了避免精度问题。
 */
#undef HAVE_CLOG    // 取消定义复数对数函数
#undef HAVE_CLOGF   // 取消定义复数对数函数的浮点数版本
#undef HAVE_CLOGL   // 取消定义复数对数函数的长双浮点数版本

#endif  /* defined(__GLIBC) */

#endif  /* defined(HAVE_FEATURES_H) */

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_ */
```