# `.\numpy\numpy\_core\include\numpy\npy_2_complexcompat.h`

```
/* This header is designed to be copy-pasted into downstream packages, since it provides
   a compatibility layer between the old C struct complex types and the new native C99
   complex types. The new macros are in numpy/npy_math.h, which is why it is included here. */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPLEXCOMPAT_H_

#include <numpy/npy_math.h>  // 包含 numpy/npy_math.h 头文件，这里定义了新的宏

#ifndef NPY_CSETREALF
#define NPY_CSETREALF(c, r) (c)->real = (r)  // 定义宏 NPY_CSETREALF，设置复数结构体 c 的实部为 r
#endif
#ifndef NPY_CSETIMAGF
#define NPY_CSETIMAGF(c, i) (c)->imag = (i)  // 定义宏 NPY_CSETIMAGF，设置复数结构体 c 的虚部为 i
#endif
#ifndef NPY_CSETREAL
#define NPY_CSETREAL(c, r)  (c)->real = (r)  // 定义宏 NPY_CSETREAL，设置复数结构体 c 的实部为 r
#endif
#ifndef NPY_CSETIMAG
#define NPY_CSETIMAG(c, i)  (c)->imag = (i)  // 定义宏 NPY_CSETIMAG，设置复数结构体 c 的虚部为 i
#endif
#ifndef NPY_CSETREALL
#define NPY_CSETREALL(c, r) (c)->real = (r)  // 定义宏 NPY_CSETREALL，设置复数结构体 c 的实部为 r
#endif
#ifndef NPY_CSETIMAGL
#define NPY_CSETIMAGL(c, i) (c)->imag = (i)  // 定义宏 NPY_CSETIMAGL，设置复数结构体 c 的虚部为 i
#endif

#endif  // 结束条件编译指令
```