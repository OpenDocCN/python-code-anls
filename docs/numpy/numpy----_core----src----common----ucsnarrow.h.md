# `.\numpy\numpy\_core\src\common\ucsnarrow.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_
// 如果未定义 NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_ 宏，则执行以下内容
#define NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_
// 定义 NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_ 宏

// 声明一个不导出的函数 PyUnicode_FromUCS4，返回一个 PyUnicodeObject 对象指针
NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char const *src, Py_ssize_t size, int swap, int align);

// 结束条件，关闭 NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_ 宏定义
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_UCSNARROW_H_ */
// 结束条件，关闭条件编译指令
```