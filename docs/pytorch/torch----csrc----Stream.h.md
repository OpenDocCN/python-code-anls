# `.\pytorch\torch\csrc\Stream.h`

```
#ifndef THP_STREAM_INC
#define THP_STREAM_INC

#include <c10/core/Stream.h>      // 引入 C10 库中的流定义
#include <c10/macros/Export.h>    // 引入 C10 库中的导出宏定义
#include <torch/csrc/python_headers.h>  // 引入 Torch Python 头文件

// 定义 THPStream 结构体，表示 Torch 的流对象
struct THPStream {
  PyObject_HEAD int64_t stream_id;   // 流的唯一标识符
  int64_t device_type;               // 设备类型
  int64_t device_index;              // 设备索引
};
extern TORCH_API PyTypeObject* THPStreamClass;  // 外部声明 Torch API 的流类对象

void THPStream_init(PyObject* module);  // 初始化 THPStream 结构体的函数声明

// 在行内定义 THPStream_Check 函数，用于检查给定对象是否是 THPStream 类型
inline bool THPStream_Check(PyObject* obj) {
  return THPStreamClass && PyObject_IsInstance(obj, (PyObject*)THPStreamClass);
}

// 将 C10 的流对象包装为 Python 的 THPStream 对象，并返回 PyObject 指针
PyObject* THPStream_Wrap(const c10::Stream& stream);

#endif // THP_STREAM_INC
```