# `.\numpy\numpy\_core\src\umath\string_ufuncs.h`

```
#ifndef _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_
#define _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_

# 如果宏 _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_ 未定义，则开始条件编译，防止重复包含


#ifdef __cplusplus
extern "C" {
#endif

# 如果是 C++ 环境，则按 C 语言的方式进行 extern "C" 处理，以便正确链接 C++ 代码


NPY_NO_EXPORT int
init_string_ufuncs(PyObject *umath);

# 定义一个非导出的函数原型 init_string_ufuncs，接受一个 PyObject 指针参数 umath


NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);

# 定义一个非导出的函数原型 _umath_strings_richcompare，接受两个 PyArrayObject 指针参数 self 和 other，以及两个整型参数 cmp_op 和 rstrip


#ifdef __cplusplus
}
#endif

# 如果是 C++ 环境，则结束 C 的 extern "C" 声明


#endif  /* _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_ */

# 结束条件编译指令，并注明结束宏 _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_
```