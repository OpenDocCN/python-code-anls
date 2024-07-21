# `.\pytorch\torch\csrc\cuda\Event.h`

```py
#ifndef THCP_EVENT_INC
#define THCP_EVENT_INC

# 如果 THCP_EVENT_INC 宏未定义，则定义它，避免重复包含头文件


#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/python_headers.h>

# 包含必要的头文件：CUDAEvent.h 和 python_headers.h


struct THCPEvent {
  PyObject_HEAD at::cuda::CUDAEvent cuda_event;
};

# 定义一个结构体 THCPEvent，它包含一个名为 cuda_event 的 at::cuda::CUDAEvent 对象，PyObject_HEAD 是宏，可能用于支持 Python 对象的基本结构


extern PyObject* THCPEventClass;

# 声明一个名为 THCPEventClass 的 PyObject 指针，作为 THCPEvent 结构体的 Python 类


void THCPEvent_init(PyObject* module);

# 声明函数 THCPEvent_init，用于初始化与 THCPEvent 相关的 Python 模块


inline bool THCPEvent_Check(PyObject* obj) {
  return THCPEventClass && PyObject_IsInstance(obj, THCPEventClass);
}

# 定义一个内联函数 THCPEvent_Check，用于检查给定的 Python 对象是否是 THCPEvent 类型的实例


#endif // THCP_EVENT_INC

# 结束 THCP_EVENT_INC 宏的定义
```