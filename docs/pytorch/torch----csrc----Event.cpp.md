# `.\pytorch\torch\csrc\Event.cpp`

```
# 包含头文件 pybind11/pybind11.h，用于 PyTorch C++ 扩展的 Python 绑定
#include <pybind11/pybind11.h>

# 包含 Torch 的设备相关头文件
#include <torch/csrc/Device.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

# 包含 C10 核心的事件和流相关头文件
#include <c10/core/Event.h>
#include <c10/core/Stream.h>

# 包含 C10 核心的设备类型和设备保护实现接口的头文件
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

# 包含 Python 的 structmember.h 头文件，用于处理结构体成员
#include <structmember.h>

# 包含 C++ 标准库中的 string 头文件
#include <string>

# 初始化 THPEventClass 为 nullptr
PyObject* THPEventClass = nullptr;

# 定义 THPEvent_pynew 函数，用于创建新的 THPEvent 对象
static PyObject* THPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  # 初始化 enable_timing、blocking、interprocess 为 0
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  # 定义静态的 PythonArgParser 对象 parser，解析事件对象的参数
  static torch::PythonArgParser parser({
      "Event(Device device=None, *, bool enable_timing=True, bool blocking=False, bool interprocess=False)",
  });

  # 解析传入的参数 args 和 kwargs 到 parsed_args 中
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  # 获取设备信息
  auto device = r.deviceOptional(0);

  # 如果没有指定设备，默认使用 CPU 设备
  if (!device.has_value()) {
    device = at::Device(at::getAccelerator(false).value_or(at::kCPU));
  }

  # 解析布尔类型参数，并设置默认值
  enable_timing = r.toBoolWithDefault(1, true);
  blocking = r.toBoolWithDefault(2, false);
  interprocess = r.toBoolWithDefault(3, false);

  # 分配类型为 type 的新对象内存，并检查分配是否成功
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    TORCH_CHECK(ptr, "Failed to allocate memory for Event");
  }

  # 获取指向 THPEvent 类型的指针 self，并创建新的 c10::Event 对象
  THPEvent* self = (THPEvent*)ptr.get();

  # TODO: blocking 和 interprocess 目前不支持，需要重构 c10::Event 的标志系统，
  #       并提供一个通用构造函数来支持阻塞和跨进程事件。
  (void)blocking;
  (void)interprocess;

  # 使用设备类型和标志位创建 c10::Event 对象，根据 enable_timing 设置不同的行为标志
  new (&self->event) c10::Event(
      device->type(),
      (enable_timing ? c10::EventFlag::BACKEND_DEFAULT
                     : c10::EventFlag::PYTORCH_DEFAULT));

  # 释放 ptr 持有的 PyObject 对象的所有权，并返回该对象的 PyObject 指针
  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

# 创建并返回新的 THPEvent 对象，指定设备类型和事件标志
PyObject* THPEvent_new(c10::DeviceType device_type, c10::EventFlag flag) {
  auto type = (PyTypeObject*)&THPEventType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  TORCH_CHECK(self, "Failed to allocate memory for Event");
  auto self_ = reinterpret_cast<THPEvent*>(self.get());
  new (&self_->event) c10::Event(device_type, flag);
  return self.release();
}

# 定义 THPEvent_dealloc 函数，用于释放 THPEvent 对象的内存
static void THPEvent_dealloc(THPEvent* self) {
  {
    # 释放全局解释器锁，用于调用 c10::Event 对象的析构函数
    pybind11::gil_scoped_release no_gil{};
    # 调用 c10::Event 对象的析构函数
    self->event.~Event();
  }
  # 释放 THPEvent 对象的内存
  Py_TYPE(self)->tp_free((PyObject*)self);
}

# 定义 THPEvent_get_device 函数，用于获取事件关联的设备信息
static PyObject* THPEvent_get_device(THPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  # 获取事件对象的设备信息
  at::optional<at::Device> device = self->event.device();
  # 如果设备信息不存在，返回 None
  if (!device) {
    Py_RETURN_NONE;
  }
  # 返回包含设备信息的新的 THPDevice 对象
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

# 定义 THPEvent_record 函数，用于记录事件的发生
static PyObject* THPEvent_record(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
      // 处理异常，保护错误处理过程
      HANDLE_TH_ERRORS
      // 将_self转换为THPEvent类型的对象
      auto self = (THPEvent*)_self;
      // 初始化_stream为Py_None
      PyObject* _stream = Py_None;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      // 定义可接受的参数列表
      constexpr const char* accepted_args[] = {"stream", nullptr};
      // 解析传入的args和kwargs，可选参数为'O'
      if (!PyArg_ParseTupleAndKeywords(
              args,
              kwargs,
              "|O",
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              const_cast<char**>(accepted_args),
              &_stream)) {
        // 解析参数失败时发出警告
        TORCH_WARN("Parsing THPEvent_record arg fails");
        // 返回空指针
        return nullptr;
      }
      // 如果_stream不是Py_None
      if (_stream != Py_None) {
        // 将_stream转换为THPStream类型的对象
        auto stream = (THPStream*)_stream;
        // 记录事件到指定的流
        self->event.record(c10::Stream::unpack3(
            stream->stream_id,
            stream->device_index,
            static_cast<c10::DeviceType>(stream->device_type)));
      } else {
        // 创建虚拟设备防护实现
        c10::impl::VirtualGuardImpl impl{
            static_cast<c10::DeviceType>(self->event.device_type())};
        // 记录事件到虚拟设备的默认流
        self->event.record(impl.getStream(impl.getDevice()));
      }
      // 返回Py_None对象
      Py_RETURN_NONE;
      // 结束异常处理
      END_HANDLE_TH_ERRORS
    }
static PyObject* THPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs) {
  // 处理 C++ 异常，将错误信息传递给 Python 异常处理机制
  HANDLE_TH_ERRORS
  // 将 _type 转换为 PyTypeObject 类型，表示事件对象的类型
  auto type = (PyTypeObject*)_type;

  // 定义静态的 PythonArgParser 对象，指定函数的参数类型和名称
  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  // 解析 Python 参数，返回解析结果
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 获取设备类型和 IPC 句柄字符串
  at::Device device = r.device(0);
  std::string handle_string = r.string(1);
  // 抛出未实现错误，提示 IPC 功能暂不支持
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "torch.Event ipc is not supported yet, please open an issue if you need this!");
  
  // 分配给事件对象的内存空间，如果分配失败则返回空指针
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  // 将分配的指针转换为 THPEvent 类型的指针
  THPEvent* self = (THPEvent*)ptr.get();

  // TODO: 需要更通用的构造函数来从 IPC 句柄构造事件对象 c10::Event
  // 使用设备类型和默认事件标志来构造 c10::Event 对象
  new (&self->event) c10::Event(device.type(), c10::EventFlag::PYTORCH_DEFAULT);

  // 返回事件对象的 Python 对象表示
  return (PyObject*)ptr.release();
  // 处理 C++ 异常的结束标记
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_ipc_handle(PyObject* _self, PyObject* noargs) {
  // 处理 C++ 异常，将错误信息传递给 Python 异常处理机制
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THPEvent 类型的指针
  auto self = (THPEvent*)_self;
  // 抛出未实现错误，提示 IPC 功能暂不支持
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "torch.Event ipc is not supported yet, please open an issue if you need this!");
  // 返回默认的 IPC 句柄字符串，这里为了示例返回 "0"
  std::string handle = "0";
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  // 处理 C++ 异常的结束标记
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_wait(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  // 处理 C++ 异常，将错误信息传递给 Python 异常处理机制
  HANDLE_TH_ERRORS {
    // 将 _self 转换为 THPEvent 类型的指针
    auto self = (THPEvent*)_self;
    // 初始化流对象为 Py_None
    PyObject* _stream = Py_None;
    // 定义可接受的参数数组，目前只接受 "stream" 参数
    constexpr const char* accepted_args[] = {"stream", nullptr};
    // 解析传入的 Python 参数，可以接受可选的流对象
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|O",
            // 可接受的参数列表
            const_cast<char**>(accepted_args),
            &_stream)) {
      // 解析失败时发出警告并返回空指针
      TORCH_WARN("Parsing THPEvent_wait arg fails");
      return nullptr;
    }
    // 如果传入的流对象不是 Py_None
    if (_stream != Py_None) {
      // 将 _stream 转换为 THPStream 类型的指针
      auto stream = (THPStream*)_stream;
      // 使用流对象的信息阻塞事件对象
      self->event.block(c10::Stream::unpack3(
          stream->stream_id,
          stream->device_index,
          static_cast<c10::DeviceType>(stream->device_type)));
    } else {
      // 否则使用默认设备的虚拟保护对象阻塞事件对象
      c10::impl::VirtualGuardImpl impl{
          static_cast<c10::DeviceType>(self->event.device_type())};
      self->event.block(impl.getStream(impl.getDevice()));
    }
  }
  // 返回 None 表示操作成功完成
  Py_RETURN_NONE;
  // 处理 C++ 异常的结束标记
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_query(PyObject* _self, PyObject* noargs) {
  // 处理 C++ 异常，将错误信息传递给 Python 异常处理机制
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THPEvent 类型的指针
  auto self = (THPEvent*)_self;
  // 返回事件对象的状态，转换为 Python 的布尔值
  return PyBool_FromLong(self->event.query());
  // 处理 C++ 异常的结束标记
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  // 处理 C++ 异常，将错误信息传递给 Python 异常处理机制
  HANDLE_TH_ERRORS
  // 将 _self 和 _other 转换为 THPEvent 类型的指针
  auto self = (THPEvent*)_self;
  auto other = (THPEvent*)_other;
  // 返回从 self 到 other 的时间间隔，转换为 Python 的浮点数
  return PyFloat_FromDouble(self->event.elapsedTime(other->event));
  // 处理 C++ 异常的结束标记
  END_HANDLE_TH_ERRORS
}
// 同步事件对象，确保事件在当前线程完成
static PyObject* THPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    // 释放全局解释器锁，允许多线程操作
    pybind11::gil_scoped_release no_gil{};
    // 将 _self 转换为 THPEvent 类型
    auto self = (THPEvent*)_self;
    // 同步事件对象的状态
    self->event.synchronize();
  }
  // 返回 None 表示成功
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 获取事件对象的事件 ID
static PyObject* THPEvent_evend_id(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THPEvent 类型
  auto self = (THPEvent*)_self;
  // 返回事件对象的事件 ID，作为长整型指针
  return PyLong_FromVoidPtr(self->event.eventId());
  END_HANDLE_TH_ERRORS
}

// 返回事件对象的字符串表示形式
static PyObject* THPEvent_repr(THPEvent* self) {
  HANDLE_TH_ERRORS
  // 构造事件对象的详细字符串表示
  return THPUtils_packString(
      "torch.Event device_type=" +
      c10::DeviceTypeName(
          static_cast<c10::DeviceType>(self->event.device_type()), true) +
      ", device_index=" + std::to_string(self->event.device_index()) +
      ", event_flag=" +
      std::to_string(static_cast<int64_t>(self->event.flag())) + ", event_id=" +
      std::to_string(reinterpret_cast<int64_t>(self->event.eventId())));
  END_HANDLE_TH_ERRORS
}

// 定义 THPEvent 类型的属性
// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static struct PyGetSetDef THPEvent_properties[] = {
    {"device", (getter)THPEvent_get_device, nullptr, nullptr, nullptr}, // 获取事件的设备属性
    {"event_id", (getter)THPEvent_evend_id, nullptr, nullptr, nullptr}, // 获取事件的事件 ID 属性
    {nullptr}};

// 定义 THPEvent 类型的方法
// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static PyMethodDef THPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS, // 接受类和关键字参数的 IPC 句柄方法
     nullptr},
    {(char*)"record",
     castPyCFunctionWithKeywords(THPEvent_record),
     METH_VARARGS | METH_KEYWORDS, // 接受参数和关键字参数的记录方法
     nullptr},
    {(char*)"wait",
     castPyCFunctionWithKeywords(THPEvent_wait),
     METH_VARARGS | METH_KEYWORDS, // 接受参数和关键字参数的等待方法
     nullptr},
    {(char*)"query", THPEvent_query, METH_NOARGS, nullptr}, // 无参数的查询方法
    {(char*)"elapsed_time", THPEvent_elapsed_time, METH_O, nullptr}, // 接受对象参数的经过时间方法
    {(char*)"synchronize", THPEvent_synchronize, METH_NOARGS, nullptr}, // 无参数的同步方法
    {(char*)"ipc_handle", THPEvent_ipc_handle, METH_NOARGS, nullptr}, // 无参数的 IPC 句柄方法
    {nullptr}};

// 定义 THPEventType 类型对象
PyTypeObject THPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.Event", /* tp_name */ // 类型名称
    sizeof(THPEvent), /* tp_basicsize */ // 类型的基本大小
    0, /* tp_itemsize */ // 不使用额外数据
    (destructor)THPEvent_dealloc, /* tp_dealloc */ // 析构函数指针
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPEvent_repr, /* tp_repr */ // 字符串表示形式函数指针
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */ // 类型标志
    nullptr, /* tp_doc */ // 类型文档字符串
    nullptr, /* tp_traverse */ // 遍历函数指针
    nullptr, /* tp_clear */ // 清除函数指针
    nullptr, /* tp_richcompare */ // 富比较函数指针
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPEvent_methods, /* tp_methods */ // 类型的方法列表
    nullptr, /* tp_members */ // 类型的成员变量列表
    THPEvent_properties, /* tp_getset */
    // 设置类型的属性的 getset 描述器集合
    nullptr, /* tp_base */
    // 指向此类型的基类型，本例中为nullptr表示没有基类型
    nullptr, /* tp_dict */
    // 指向类型的字典，本例中为nullptr表示没有自定义的类型字典
    nullptr, /* tp_descr_get */
    // 获取描述符的方法，本例中为nullptr表示没有描述符获取方法
    nullptr, /* tp_descr_set */
    // 设置描述符的方法，本例中为nullptr表示没有描述符设置方法
    0, /* tp_dictoffset */
    // 类型字典偏移量，本例中为0表示字典存储在类型结构中的偏移量
    nullptr, /* tp_init */
    // 初始化对象的方法，本例中为nullptr表示没有对象初始化方法
    nullptr, /* tp_alloc */
    // 分配对象内存的方法，本例中为nullptr表示没有对象分配方法
    THPEvent_pynew, /* tp_new */
    // 创建新对象的方法，指向 THPEvent_pynew 函数
};

void THPEvent_init(PyObject* module) {
  // 将 THPEventType 强制转换为 PyObject*，作为 THPEventClass 的初始值
  THPEventClass = (PyObject*)&THPEventType;
  // 如果初始化 THPEventType 失败，抛出 Python 异常
  if (PyType_Ready(&THPEventType) < 0) {
    throw python_error();
  }
  // 增加对 THPEventType 的引用计数，确保对象不会在使用期间被销毁
  Py_INCREF(&THPEventType);
  // 将 THPEventType 添加到指定的 Python 模块中，名为 "Event"
  if (PyModule_AddObject(module, "Event", (PyObject*)&THPEventType) < 0) {
    throw python_error();
  }
}
```