# `.\numpy\numpy\_core\src\common\gil_utils.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_
#define NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

// 声明一个不导出的函数，用于处理 GIL 错误，接受一个异常类型和格式化字符串参数
NPY_NO_EXPORT void
npy_gil_error(PyObject *type, const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif /* NUMPY_CORE_SRC_COMMON_GIL_UTILS_H_ */
```