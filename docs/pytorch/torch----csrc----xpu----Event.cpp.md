# `.\pytorch\torch\csrc\xpu\Event.cpp`

```
// 引入 pybind11 库，用于创建 Python 扩展模块
#include <pybind11/pybind11.h>
// 引入 Torch 的设备相关头文件
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
// 引入 Torch 的 XPU 相关头文件
#include <torch/csrc/xpu/Event.h>
#include <torch/csrc/xpu/Module.h>
#include <torch/csrc/xpu/Stream.h>

#include <structmember.h>

// 定义 THXPEventClass 作为全局变量，默认为 nullptr
PyObject* THXPEventClass = nullptr;

// 定义 THXPEvent_pynew 函数，用于创建新的 THXPEvent 对象
static PyObject* THXPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // 初始化 enable_timing 变量，默认为 0
  unsigned char enable_timing = 0;

  // 定义关键字参数列表
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"enable_timing", nullptr};

  // 解析传入的 Python 参数
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|b",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &enable_timing)) {
    return nullptr;
  }

  // 分配内存给 THXPEvent 对象
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  // 将 ptr 转换为 THXPEvent 指针
  THXPEvent* self = (THXPEvent*)ptr.get();

  // 使用 placement new 构造 XPUEvent 对象
  new (&self->xpu_event) at::xpu::XPUEvent(enable_timing);

  // 返回构造好的 Python 对象
  return (PyObject*)ptr.release();

  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_dealloc 函数，用于释放 THXPEvent 对象的内存
static void THXPEvent_dealloc(THXPEvent* self) {
  {
    // 释放 GIL，执行 xpu_event 的析构函数
    pybind11::gil_scoped_release no_gil{};
    self->xpu_event.~XPUEvent();
  }
  // 释放 self 对象内存
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// 定义 THXPEvent_get_sycl_event 函数，返回 THXPEvent 对象的 sycl_event
static PyObject* THXPEvent_get_sycl_event(THXPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(&self->xpu_event.event());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_get_device 函数，返回 THXPEvent 对象的 device
static PyObject* THXPEvent_get_device(THXPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  // 获取 XPUEvent 对象的设备信息
  at::optional<at::Device> device = self->xpu_event.device();
  if (!device) {
    // 如果设备信息不存在，则返回 None
    Py_RETURN_NONE;
  }
  // 否则返回设备信息的 Python 对象
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_record 函数，记录 THXPEvent 对象在指定 stream 上的事件
static PyObject* THXPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  // 将 _self 和 _stream 转换为 THXPEvent 和 THXPStream 对象
  auto* self = (THXPEvent*)_self;
  auto* stream = (THXPStream*)_stream;
  // 调用 XPUEvent 对象的记录函数
  self->xpu_event.record(stream->xpu_stream);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_wait 函数，等待 THXPEvent 对象在指定 stream 上的事件
static PyObject* THXPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  // 将 _self 和 _stream 转换为 THXPEvent 和 THXPStream 对象
  auto* self = (THXPEvent*)_self;
  auto* stream = (THXPStream*)_stream;
  // 调用 XPUEvent 对象的阻塞函数
  self->xpu_event.block(stream->xpu_stream);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_query 函数，查询 THXPEvent 对象的状态
static PyObject* THXPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THXPEvent 对象
  auto* self = (THXPEvent*)_self;
  // 返回 XPUEvent 对象的查询状态
  return PyBool_FromLong(self->xpu_event.query());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_elapsed_time 函数，计算 THXPEvent 对象与另一事件之间的时间差
static PyObject* THXPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  // 将 _self 和 _other 转换为 THXPEvent 对象
  auto* self = (THXPEvent*)_self;
  auto* other = (THXPEvent*)_other;
  // 返回 XPUEvent 对象的经过时间
  return PyFloat_FromDouble(self->xpu_event.elapsed_time(other->xpu_event));
  END_HANDLE_TH_ERRORS
}

// 定义 THXPEvent_synchronize 函数，同步 THXPEvent 对象
static PyObject* THXPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    // 释放 GIL，同步 XPUEvent 对象
    pybind11::gil_scoped_release no_gil;
    auto* self = (THXPEvent*)_self;
    // 执行 XPUEvent 的同步操作
    self->xpu_event.synchronize();
    // 返回 None
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}
    // 同步 CUDA 事件，等待事件完成
    self->xpu_event.synchronize();
  }
  // 返回 None 对象给 Python，表示成功执行完毕
  Py_RETURN_NONE;
  // 处理 PyTorch 错误的宏结束标记
  END_HANDLE_TH_ERRORS
// NOLINTNEXTLINE(*c-arrays*, *global-variables)
// 定义静态结构体数组，用于定义 THXPEvent 类型的属性
static struct PyGetSetDef THXPEvent_properties[] = {
    // 属性名为 "device"，getter 函数为 THXPEvent_get_device，其它参数为 nullptr
    {"device", (getter)THXPEvent_get_device, nullptr, nullptr, nullptr},
    // 属性名为 "sycl_event"，getter 函数为 THXPEvent_get_sycl_event，其它参数为 nullptr
    {"sycl_event", (getter)THXPEvent_get_sycl_event, nullptr, nullptr, nullptr},
    // 数组末尾标志，nullptr 表示结束
    {nullptr}};

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
// 定义静态方法数组，用于定义 THXPEvent 类型的方法
static PyMethodDef THXPEvent_methods[] = {
    // 方法名为 "record"，对应的 C 函数为 THXPEvent_record，参数类型为 METH_O（接受一个对象作为参数）
    {(char*)"record", THXPEvent_record, METH_O, nullptr},
    // 方法名为 "wait"，对应的 C 函数为 THXPEvent_wait，参数类型为 METH_O
    {(char*)"wait", THXPEvent_wait, METH_O, nullptr},
    // 方法名为 "query"，对应的 C 函数为 THXPEvent_query，参数类型为 METH_NOARGS（无参数）
    {(char*)"query", THXPEvent_query, METH_NOARGS, nullptr},
    // 方法名为 "elapsed_time"，对应的 C 函数为 THXPEvent_elapsed_time，参数类型为 METH_O
    {(char*)"elapsed_time", THXPEvent_elapsed_time, METH_O, nullptr},
    // 方法名为 "synchronize"，对应的 C 函数为 THXPEvent_synchronize，参数类型为 METH_NOARGS
    {(char*)"synchronize", THXPEvent_synchronize, METH_NOARGS, nullptr},
    // 数组末尾标志，nullptr 表示结束
    {nullptr}};

// 定义 THXPEventType 结构体，描述了 THXPEvent 类型对象的结构和行为
PyTypeObject THXPEventType = {
    // 对象头初始化，nullptr 表示无基类，0 表示不包含额外数据
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._XpuEventBase", /* tp_name */
    // 对象的基本大小为 sizeof(THXPEvent)
    sizeof(THXPEvent), /* tp_basicsize */
    // 对象的每个元素大小为 0，表示不支持可变长度对象
    0, /* tp_itemsize */
    // 对象销毁时调用的函数指针，指向 THXPEvent_dealloc 函数
    (destructor)THXPEvent_dealloc, /* tp_dealloc */
    // tp_vectorcall_offset 为 0，表示不使用 vectorcall 协议
    0, /* tp_vectorcall_offset */
    // tp_getattr 指针为 nullptr，表示不定义 getattr 操作
    nullptr, /* tp_getattr */
    // tp_setattr 指针为 nullptr，表示不定义 setattr 操作
    nullptr, /* tp_setattr */
    // tp_reserved 指针为 nullptr，保留字段
    nullptr, /* tp_reserved */
    // tp_repr 指针为 nullptr，表示不定义对象的字符串表示形式
    nullptr, /* tp_repr */
    // tp_as_number 指针为 nullptr，表示不支持数学运算
    nullptr, /* tp_as_number */
    // tp_as_sequence 指针为 nullptr，表示不支持序列操作
    nullptr, /* tp_as_sequence */
    // tp_as_mapping 指针为 nullptr，表示不支持映射操作
    nullptr, /* tp_as_mapping */
    // tp_hash 指针为 nullptr，表示不定义对象的哈希操作
    nullptr, /* tp_hash  */
    // tp_call 指针为 nullptr，表示不支持调用操作
    nullptr, /* tp_call */
    // tp_str 指针为 nullptr，表示不定义对象的字符串转换操作
    nullptr, /* tp_str */
    // tp_getattro 指针为 nullptr，表示不定义获取属性操作
    nullptr, /* tp_getattro */
    // tp_setattro 指针为 nullptr，表示不定义设置属性操作
    nullptr, /* tp_setattro */
    // tp_as_buffer 指针为 nullptr，表示不支持缓冲区操作
    nullptr, /* tp_as_buffer */
    // tp_flags 表示对象的特性，包括默认特性和基类特性
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    // tp_doc 指针为 nullptr，表示不定义对象的文档字符串
    nullptr, /* tp_doc */
    // tp_traverse 指针为 nullptr，表示不定义对象的遍历操作
    nullptr, /* tp_traverse */
    // tp_clear 指针为 nullptr，表示不定义对象的清理操作
    nullptr, /* tp_clear */
    // tp_richcompare 指针为 nullptr，表示不定义对象的比较操作
    nullptr, /* tp_richcompare */
    // tp_weaklistoffset 为 0，表示不支持弱引用
    0, /* tp_weaklistoffset */
    // tp_iter 指针为 nullptr，表示不定义对象的迭代器操作
    nullptr, /* tp_iter */
    // tp_iternext 指针为 nullptr，表示不定义对象的迭代器获取下一个元素操作
    nullptr, /* tp_iternext */
    // tp_methods 指针指向 THXPEvent_methods，定义了对象支持的方法
    THXPEvent_methods, /* tp_methods */
    // tp_members 指针为 nullptr，表示不定义对象的成员
    nullptr, /* tp_members */
    // tp_getset 指针指向 THXPEvent_properties，定义了对象支持的属性
    THXPEvent_properties, /* tp_getset */
    // tp_base 指针为 nullptr，表示不继承自其它类型
    nullptr, /* tp_base */
    // tp_dict 指针为 nullptr，表示不定义对象的字典
    nullptr, /* tp_dict */
    // tp_descr_get 指针为 nullptr，表示不定义描述符的获取操作
    nullptr, /* tp_descr_get */
    // tp_descr_set 指针为 nullptr，表示不定义描述符的设置操作
    nullptr, /* tp_descr_set */
    // tp_dictoffset 为 0，表示不定义对象的字典偏移量
    0, /* tp_dictoffset */
    // tp_init 指针为 nullptr，表示不定义对象的初始化操作
    nullptr, /* tp_init */
    // tp_alloc 指针为 nullptr，表示不定义对象的内存分配操作
    nullptr, /* tp_alloc */
    // tp_new 指针指向 THXPEvent_pynew，表示对象的创建操作
    THXPEvent_pynew, /* tp_new */
};

// 初始化 THXPEvent 类型对象，将其注册到指定的 Python 模块中
void THXPEvent_init(PyObject* module) {
  // 将 THXPEventType 设为 THXPEventClass，用于后续对象的创建
  THXPEventClass = (PyObject*)&THXPEventType;
  // 如果 PyType_Ready 函数返回错误，则抛出 python_error 异常
  if (PyType_Ready(&THXPEventType) < 0) {
    throw python_error();
  }
  // 增加 THXPEventType 的引用计数
  Py_INCREF(&THXPEventType);
  // 将 THXPEventType 对象添加到指定的 Python 模块中
  if (PyModule_AddObject(module, "_XpuEventBase", (PyObject*)&THXPEventType) <
      0) {
    throw python_error();
  }
}
```