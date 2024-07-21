# `.\pytorch\torch\csrc\xpu\Stream.h`

```
#pragma once


// 使用 pragma once 防止头文件的多重包含问题

#include <c10/xpu/XPUStream.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>


// 引入必要的头文件，分别是 c10/xpu/XPUStream.h，torch/csrc/Stream.h 和 torch/csrc/python_headers.h

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)


// 禁止下一行 lint 提示，忽略 cppcoreguidelines-pro-type-member-init 规则

struct THXPStream : THPStream {
  // 定义 THXPStream 结构体，继承自 THPStream 结构体
  at::xpu::XPUStream xpu_stream;
  // 在 THXPStream 结构体中定义了一个 at::xpu::XPUStream 类型的成员变量 xpu_stream
};
extern PyObject* THXPStreamClass;


// 声明 THXPStreamClass，表示外部有一个 PyObject 类型的变量 THXPStreamClass

void THXPStream_init(PyObject* module);


// 声明 THXPStream_init 函数，该函数接受一个 PyObject 指针作为参数，用于初始化相关操作

inline bool THXPStream_Check(PyObject* obj) {
  // 定义 THXPStream_Check 内联函数，用于检查给定的 PyObject 对象是否是 THXPStream 类型
  return THXPStreamClass && PyObject_IsInstance(obj, THXPStreamClass);
  // 返回 THXPStreamClass 不为空且 obj 是 THXPStreamClass 的实例的布尔值
}
```