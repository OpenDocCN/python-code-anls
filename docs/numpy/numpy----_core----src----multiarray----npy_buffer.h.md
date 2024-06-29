# `.\numpy\numpy\_core\src\multiarray\npy_buffer.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_

// 声明了一个名为 array_as_buffer 的外部变量，类型为 PyBufferProcs 结构体
extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

// 声明了一个非导出函数 _buffer_info_free，接受一个 void 指针和一个 PyObject 指针作为参数，返回一个整型值
NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj);

// 声明了一个非导出函数 _descriptor_from_pep3118_format，接受一个 const char* 参数，返回一个 PyArray_Descr* 指针
NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s);

// 声明了一个非导出函数 void_getbuffer，接受一个 PyObject 指针和一个 Py_buffer 指针以及一个整型标志 flags 作为参数，返回一个整型值
NPY_NO_EXPORT int
void_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_ */
```