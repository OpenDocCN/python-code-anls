# `.\pytorch\torch\csrc\xpu\Stream.cpp`

```
// 定义全局变量，表示 THXPStreamClass 类型的 Python 对象
PyObject* THXPStreamClass = nullptr;

// 定义 THXPStream_pynew 函数，用于创建新的 THXPStream 对象
static PyObject* THXPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // 获取当前 XPU 设备
  const auto current_device = c10::xpu::current_device();

  // 初始化优先级、流 ID、设备索引、设备类型
  int32_t priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // 定义关键字参数列表
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority", "stream_id", "device_index", "device_type", nullptr};

  // 解析传入的 Python 参数并填充到相应变量中
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type)) {
    return nullptr;
  }

  // 分配内存以创建新的 THPObjectPtr 对象
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  // 根据参数创建新的 XPU 流对象
  at::xpu::XPUStream stream = (stream_id || device_index || device_type)
      ? at::xpu::XPUStream::unpack3(
            stream_id,
            static_cast<c10::DeviceIndex>(device_index),
            static_cast<c10::DeviceType>(device_type))
      : at::xpu::getStreamFromPool(priority, current_device);

  // 获取指向 THXPStream 结构体的指针
  THXPStream* self = (THXPStream*)ptr.get();
  // 设置 stream_id、device_index 和 device_type 成员变量
  self->stream_id = static_cast<int64_t>(stream.id());
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  // 在 self->xpu_stream 上构造新的 at::xpu::XPUStream 对象
  new (&self->xpu_stream) at::xpu::XPUStream(stream);

  // 返回创建的 Python 对象指针
  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

// 定义 THXPStream_dealloc 函数，用于释放 THXPStream 对象
static void THXPStream_dealloc(THXPStream* self) {
  // 调用 XPUStream 对象的析构函数
  self->xpu_stream.~XPUStream();
  // 释放 self 对象内存
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// 定义 THXPStream_get_device 函数，返回 THXPStream 对象的设备信息
static PyObject* THXPStream_get_device(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  // 返回 self->xpu_stream 的设备信息
  return THPDevice_New(self->xpu_stream.device());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPStream_get_sycl_queue 函数，返回 THXPStream 对象的 SYCL 队列信息
static PyObject* THXPStream_get_sycl_queue(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  // 返回指向 self->xpu_stream 队列的指针作为 Python 的长整型对象
  return PyLong_FromVoidPtr(&self->xpu_stream.queue());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPStream_get_priority 函数，返回 THXPStream 对象的优先级信息
static PyObject* THXPStream_get_priority(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  // 返回 self->xpu_stream 的优先级信息
  return THPUtils_packInt64(self->xpu_stream.priority());
  END_HANDLE_TH_ERRORS
}

// 定义 THXPStream_priority_range 函数，返回 XPU 流的优先级范围
static PyObject* THXPStream_priority_range(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取 XPU 流的优先级范围并返回 Python 元组
  auto [least_priority, greatest_priority] =
      at::xpu::XPUStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}
// 定义一个静态函数 THXPStream_query，用于查询当前对象的 xpu_stream 是否有效
static PyObject* THXPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THXPStream 类型的指针
  auto* self = (THXPStream*)_self;
  // 调用 xpu_stream 对象的 query 方法，并将返回值转换为 Python 的布尔类型
  return PyBool_FromLong(self->xpu_stream.query());
  END_HANDLE_TH_ERRORS
}

// 定义一个静态函数 THXPStream_synchronize，用于同步当前对象的 xpu_stream
static PyObject* THXPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    // 释放 GIL，允许线程调度
    pybind11::gil_scoped_release no_gil;
    // 将 _self 转换为 THXPStream 类型的指针
    auto* self = (THXPStream*)_self;
    // 调用 xpu_stream 对象的 synchronize 方法，同步操作
    self->xpu_stream.synchronize();
  }
  // 返回 None 表示函数执行完毕
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 定义一个静态函数 THXPStream_eq，用于比较两个 THXPStream 对象的 xpu_stream 是否相等
static PyObject* THXPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  // 将 _self 和 _other 分别转换为 THXPStream 类型的指针
  auto* self = (THXPStream*)_self;
  auto* other = (THXPStream*)_other;
  // 比较两个 xpu_stream 是否相等，并将结果转换为 Python 的布尔类型
  return PyBool_FromLong(self->xpu_stream == other->xpu_stream);
  END_HANDLE_TH_ERRORS
}

// 定义一个空的成员变量列表 THXPStream_members
// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THXPStream_members[] = {{nullptr}};

// 定义一个属性列表 THXPStream_properties，包含两个属性 "sycl_queue" 和 "priority"
// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THXPStream_properties[] = {
    {"sycl_queue",
     (getter)THXPStream_get_sycl_queue,
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THXPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// 定义一个方法列表 THXPStream_methods，包含四个方法 "query", "synchronize", "priority_range", "__eq__"
// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static PyMethodDef THXPStream_methods[] = {
    {"query", THXPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THXPStream_synchronize, METH_NOARGS, nullptr},
    {"priority_range",
     THXPStream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {"__eq__", THXPStream_eq, METH_O, nullptr},
    {nullptr}};

// 定义一个 PyTypeObject 结构体 THXPStreamType，表示 Python 中的 THXPStream 类型
PyTypeObject THXPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._XpuStreamBase", /* tp_name */
    sizeof(THXPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THXPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THXPStream_methods, /* tp_methods */
    THXPStream_members, /* tp_members */
    THXPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THXPStream_pynew, /* tp_new */
};
void THXPStream_init(PyObject* module) {
  // 增加对 THPStreamClass 的引用计数，确保不会在使用期间被释放
  Py_INCREF(THPStreamClass);
  
  // 将 THXPStreamType 的基类设置为 THPStreamClass
  THXPStreamType.tp_base = THPStreamClass;
  
  // 将 THXPStreamClass 转换为 PyObject 指针，并赋给 THXPStreamType
  THXPStreamClass = (PyObject*)&THXPStreamType;
  
  // 准备 THXPStreamType 类型，如果失败则抛出 Python 错误
  if (PyType_Ready(&THXPStreamType) < 0) {
    throw python_error();
  }
  
  // 增加对 THXPStreamType 的引用计数，确保不会在使用期间被释放
  Py_INCREF(&THXPStreamType);
  
  // 将 THXPStreamType 添加到 module 模块中名为 "_XpuStreamBase" 的对象
  if (PyModule_AddObject(module, "_XpuStreamBase", (PyObject*)&THXPStreamType) <
      0) {
    throw python_error();
  }
}
```