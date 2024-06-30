# `D:\src\scipysrc\scipy\scipy\_build_utils\src\npy_2_complexcompat.h`

```
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_

// 如果未定义 NPY_CSETREALF 宏，则定义该宏，设置复数结构体 c 的实部为 r
#ifndef NPY_CSETREALF
#define NPY_CSETREALF(c, r) (c)->real = (r)
#endif

// 如果未定义 NPY_CSETIMAGF 宏，则定义该宏，设置复数结构体 c 的虚部为 i
#ifndef NPY_CSETIMAGF
#define NPY_CSETIMAGF(c, i) (c)->imag = (i)
#endif

// 如果未定义 NPY_CSETREAL 宏，则定义该宏，设置复数结构体 c 的实部为 r
#ifndef NPY_CSETREAL
#define NPY_CSETREAL(c, r)  (c)->real = (r)
#endif

// 如果未定义 NPY_CSETIMAG 宏，则定义该宏，设置复数结构体 c 的虚部为 i
#ifndef NPY_CSETIMAG
#define NPY_CSETIMAG(c, i)  (c)->imag = (i)
#endif

// 如果未定义 NPY_CSETREALL 宏，则定义该宏，设置复数结构体 c 的实部为 r
#ifndef NPY_CSETREALL
#define NPY_CSETREALL(c, r) (c)->real = (r)
#endif

// 如果未定义 NPY_CSETIMAGL 宏，则定义该宏，设置复数结构体 c 的虚部为 i
#ifndef NPY_CSETIMAGL
#define NPY_CSETIMAGL(c, i) (c)->imag = (i)
#endif

#endif
```