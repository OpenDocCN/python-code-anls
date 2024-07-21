# `.\pytorch\torch\csrc\cuda\Module.h`

```py
#ifndef THCP_CUDA_MODULE_INC
// 如果未定义 THCP_CUDA_MODULE_INC 宏，则包含以下内容，以避免重复定义
#define THCP_CUDA_MODULE_INC

// 声明函数 THCPModule_getDevice_wrap，返回一个 PyObject 指针，无参数
PyObject* THCPModule_getDevice_wrap(PyObject* self);

// 声明函数 THCPModule_setDevice_wrap，返回一个 PyObject 指针，接受一个 PyObject 指针作为参数
PyObject* THCPModule_setDevice_wrap(PyObject* self, PyObject* arg);

// 声明函数 THCPModule_getDeviceName_wrap，返回一个 PyObject 指针，接受一个 PyObject 指针作为参数
PyObject* THCPModule_getDeviceName_wrap(PyObject* self, PyObject* arg);

// 声明函数 THCPModule_getDriverVersion，返回一个 PyObject 指针，无参数
PyObject* THCPModule_getDriverVersion(PyObject* self);

// 声明函数 THCPModule_isDriverSufficient，返回一个 PyObject 指针，无参数
PyObject* THCPModule_isDriverSufficient(PyObject* self);

// 声明函数 THCPModule_getCurrentBlasHandle_wrap，返回一个 PyObject 指针，无参数
PyObject* THCPModule_getCurrentBlasHandle_wrap(PyObject* self);

// 结束条件编译指令
#endif
```