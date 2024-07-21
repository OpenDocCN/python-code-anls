# `.\pytorch\torch\csrc\xpu\Event.h`

```
#pragma once

# 预处理指令：指示编译器只包含当前头文件一次，避免重复包含。


#include <ATen/xpu/XPUEvent.h>
#include <torch/csrc/python_headers.h>

# 包含头文件：引入所需的 ATen/XPUEvent.h 和 torch/csrc/python_headers.h 头文件。


struct THXPEvent {
  PyObject_HEAD at::xpu::XPUEvent xpu_event;
};

# 结构体定义：定义名为 THXPEvent 的结构体，包含 PyObject_HEAD 和 at::xpu::XPUEvent 类型的成员变量 xpu_event。


extern PyObject* THXPEventClass;

# 外部声明：声明一个名为 THXPEventClass 的 PyObject 指针，用于表示 THXPEvent 类的 Python 类对象。


void THXPEvent_init(PyObject* module);

# 函数声明：声明一个函数 THXPEvent_init，接受一个 PyObject 指针作为参数，用于初始化 THXPEvent 结构体。


inline bool THXPEvent_Check(PyObject* obj) {
  return THXPEventClass && PyObject_IsInstance(obj, THXPEventClass);
}

# 内联函数定义：定义了一个内联函数 THXPEvent_Check，接受一个 PyObject 指针作为参数，用于检查该对象是否是 THXPEvent 类型的实例。
```