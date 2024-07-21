# `.\pytorch\torch\csrc\Event.h`

```
#ifndef THP_EVENT_INC
#define THP_EVENT_INC

// 包含头文件 Event.h 和 python_headers.h，用于定义事件和 Python 相关的接口
#include <c10/core/Event.h>
#include <torch/csrc/python_headers.h>

// 定义 THPEvent 结构体，包含一个 c10::Event 类型的成员变量
struct TORCH_API THPEvent {
  PyObject_HEAD     // 定义 Python 对象的头部
  c10::Event event;  // c10::Event 类型的事件对象
};

// 声明全局变量 THPEventClass 和 THPEventType，分别表示 THPEvent 类的 Python 类和类型对象
extern PyObject* THPEventClass;
TORCH_API extern PyTypeObject THPEventType;

// 初始化 THPEvent 模块
TORCH_API void THPEvent_init(PyObject* module);

// 创建一个新的 THPEvent 对象，参数为设备类型和事件标志
TORCH_API PyObject* THPEvent_new(
    c10::DeviceType device_type,
    c10::EventFlag flag);

// 内联函数，检查给定的 Python 对象是否是 THPEvent 类的实例
inline bool THPEvent_Check(PyObject* obj) {
  return THPEventClass && PyObject_IsInstance(obj, THPEventClass);
}

#endif // THP_EVENT_INC
```