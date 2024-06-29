# `.\numpy\numpy\_core\src\umath\stringdtype_ufuncs.h`

```
#ifndef _NPY_CORE_SRC_UMATH_STRINGDTYPE_UFUNCS_H_
#define _NPY_CORE_SRC_UMATH_STRINGDTYPE_UFUNCS_H_

#ifdef __cplusplus
extern "C" {
#endif

// 声明一个不导出的函数，用于初始化字符串数据类型的通用函数
NPY_NO_EXPORT int
init_stringdtype_ufuncs(PyObject* umath);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_UMATH_STRINGDTYPE_UFUNCS_H_ */
```