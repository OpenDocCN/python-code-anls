# `.\pytorch\torch\csrc\Stream.cpp`

```
// 导入必要的头文件，包括 pybind11 和 Torch 相关的头文件
#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <structmember.h>
#include <cstdint>

// 定义全局变量 THPStreamClass，用于表示 THPStream 类型对象
PyTypeObject* THPStreamClass = nullptr;

// 定义 THPStream_pynew 函数，用于创建新的 THPStream 对象
static PyObject* THPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误

  // 初始化参数变量
  int64_t stream_id = -1;
  int64_t device_type = 0;
  int64_t device_index = 0;
  int64_t priority = 0;

  // 定义 PythonArgParser 对象，用于解析传入的参数
  static torch::PythonArgParser parser({
      "Steram(Device device=None, *, int64_t priority=0)",  // 构造函数签名1
      "Stream(int64_t stream_id, int64_t device_index, int64_t device_type, *, int64_t priority=0)",  // 构造函数签名2
  });

  // 解析参数并存储到 parsed_args 中
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 定义设备保护的指针，用于管理设备上下文
  std::unique_ptr<c10::DeviceGuard> device_guard_ptr;

  // 根据参数解析的结果进行不同的处理分支
  if (r.idx == 0) {
    // 获取默认加速器，并根据传入的 device 参数初始化设备类型和索引
    auto default_accelerator = at::getAccelerator(false);
    auto device = r.deviceOptional(0);
    if (device.has_value()) {
      device_type = static_cast<int64_t>(device->type());
      device_index = static_cast<int64_t>(device->index());
      // 如果设备不为空，初始化设备保护对象
      device_guard_ptr = std::make_unique<c10::DeviceGuard>(device.value());
    } else {
      // 如果设备为空，使用当前加速器和索引，若加速器未设置，默认使用 CPU
      device_type = static_cast<int64_t>(
          default_accelerator.value_or(c10::DeviceType::CPU));
      c10::impl::VirtualGuardImpl impl{
          static_cast<c10::DeviceType>(device_type)};
      const auto current_device = impl.getDevice();
      device_index = current_device.index();
    }
    priority = r.toInt64WithDefault(1, 0);  // 获取优先级，默认为 0
  } else if (r.idx == 1) {
    // 使用传入的参数初始化流对象的 ID、设备索引、设备类型和优先级
    stream_id = r.toInt64WithDefault(0, -1);
    device_index = r.toInt64WithDefault(1, 0);
    device_type =
        r.toInt64WithDefault(2, static_cast<int64_t>(c10::DeviceType::CPU));
    priority = r.toInt64WithDefault(3, 0);
  } else {
    // 如果解析失败，抛出错误并显示使用方法
    TORCH_CHECK(
        false,
        "parse stream arg fails please check the usage: ",
        parser.get_signatures());
  }

  // 分配内存创建 THPStream 对象
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  // 将分配的指针转换为 THPStream 指针
  THPStream* self = (THPStream*)ptr.get();

  // 如果不是从现有流创建 torch.Stream，则创建新的流对象
  // 各后端需重写 getNewStream 方法来实现具体的流对象创建
  std::optional<c10::Stream> stream_opt;
  if (r.idx == 0) {
    c10::impl::VirtualGuardImpl impl{static_cast<c10::DeviceType>(device_type)};
    // 省略了创建新流的细节，需要后端实现
  }
  // 如果是通过实现对象获取新的流
  stream_opt = impl.getNewStream(
      c10::Device(static_cast<c10::DeviceType>(device_type), device_index),
      static_cast<int>(priority));
} else {
  // 否则，解包现有流的信息
  stream_opt = c10::Stream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));
}

// 检查是否成功创建了流对象，否则抛出错误
TORCH_CHECK(stream_opt.has_value(), "Failed to create stream");
// 将流的 ID 转换为 int64_t 类型，并存储在 self 对象中
self->stream_id = static_cast<int64_t>(stream_opt->id());
// 将流的设备索引转换为 int64_t 类型，并存储在 self 对象中
self->device_index = static_cast<int64_t>(stream_opt->device_index());
// 将流的设备类型转换为 int64_t 类型，并存储在 self 对象中
self->device_type = static_cast<int64_t>(stream_opt->device_type());

// 返回 Python 中的 PyObject 指针，释放所有权
return (PyObject*)ptr.release();
END_HANDLE_TH_ERRORS
}

// 将 C10 流对象包装成 Python 对象
PyObject* THPStream_Wrap(const c10::Stream& stream) {
  HANDLE_TH_ERRORS
  // 获取 THPStream 类型对象
  auto type = (PyTypeObject*)THPStreamClass;
  // 分配内存并初始化为 THPStream 类型对象
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    // 如果分配失败，抛出异常
    throw python_error();
  }

  // 获取 self 指针
  THPStream* self = (THPStream*)ptr.get();
  // 设置流的 ID
  self->stream_id = stream.id();
  // 设置设备索引
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream.device_index());
  // 设置设备类型
  self->device_type = static_cast<int64_t>(stream.device_type());
  // 返回包装后的流对象
  return ptr.release();
  END_HANDLE_TH_ERRORS
}

// 释放 THPStream 对象
static void THPStream_dealloc(THPStream* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// 获取流对象的设备信息
static PyObject* THPStream_get_device(THPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  // 创建新的设备对象
  return THPDevice_New(c10::Device(
      static_cast<c10::DeviceType>(self->device_type),
      static_cast<c10::DeviceIndex>(self->device_index)));
  END_HANDLE_TH_ERRORS
}

// 查询流对象的状态
static PyObject* THPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;

  // 查询流对象的状态
  return PyBool_FromLong(c10::Stream::unpack3(
                             self->stream_id,
                             self->device_index,
                             static_cast<c10::DeviceType>(self->device_type))
                             .query());

  END_HANDLE_TH_ERRORS
}

// 同步流对象
static PyObject* THPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    // 释放 GIL
    pybind11::gil_scoped_release no_gil;
    auto self = (THPStream*)_self;

    // 同步流对象
    c10::Stream::unpack3(
        self->stream_id,
        self->device_index,
        static_cast<c10::DeviceType>(self->device_type))
        .synchronize();
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 等待事件
static PyObject* THPStream_wait_event(PyObject* _self, PyObject* _event) {
  HANDLE_TH_ERRORS {
    auto self = (THPStream*)_self;
    auto event = (THPEvent*)_event;
    // 等待事件
    c10::Stream::unpack3(
        self->stream_id,
        self->device_index,
        static_cast<c10::DeviceType>(self->device_type))
        .wait(event->event);
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 等待流对象
static PyObject* THPStream_wait_stream(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS {
    auto self = (THPStream*)_self;
    auto other_stream = (THPStream*)_other;
    // 创建新事件
    c10::Event new_event(
        static_cast<c10::DeviceType>(other_stream->device_type),
        c10::EventFlag::PYTORCH_DEFAULT);
    // 记录事件
    new_event.record(c10::Stream::unpack3(
        other_stream->stream_id,
        other_stream->device_index,
        static_cast<c10::DeviceType>(other_stream->device_type)));
    // 等待事件
    c10::Stream::unpack3(
        self->stream_id,
        self->device_index,
        static_cast<c10::DeviceType>(self->device_type))
        .wait(new_event);
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 记录事件
static PyObject* THPStream_record_event(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
      HANDLE_TH_ERRORS
      auto self = (THPStream*)_self;  // 将传入的 self 指针转换为 THPStream 类型
      PyObject* _new_event;  // 定义 PyObject 指针变量 _new_event
      PyObject* _event = Py_None;  // 初始化 _event 为 Py_None
    
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      constexpr const char* accepted_args[] = {"event", nullptr};  // 定义接受的参数列表，仅包含 "event" 一个参数
      if (!PyArg_ParseTupleAndKeywords(
              args,
              kwargs,
              "|O",
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              const_cast<char**>(accepted_args),
              &_event)) {  // 解析传入的参数 args 和 kwargs，接受一个可选参数 "event"
        TORCH_CHECK(false, "parse record_event arg fails");  // 若解析失败，则抛出错误信息
      }
      if (_event != Py_None) {  // 如果传入的参数不是 Py_None
        // Increase the refcount of the event to avoid it being destroyed.
        Py_INCREF(_event);  // 增加参数 _event 的引用计数，避免其被销毁
        _new_event = _event;  // 将 _new_event 指向 _event
      } else {
        _new_event = THPEvent_new(
            static_cast<c10::DeviceType>(self->device_type),  // 创建一个新的 THPEvent 对象，使用 self 的设备类型
            c10::EventFlag::PYTORCH_DEFAULT);  // 设置事件标志为 PYTORCH_DEFAULT
      }
      auto new_event = (THPEvent*)_new_event;  // 将 _new_event 转换为 THPEvent 类型的指针
      TORCH_CHECK(new_event, "event must not be null");  // 检查 new_event 不为空
      new_event->event.record(c10::Stream::unpack3(
          self->stream_id,  // 使用 self 的 stream_id
          self->device_index,  // 使用 self 的 device_index
          static_cast<c10::DeviceType>(self->device_type)));  // 使用 self 的设备类型，记录事件
      return (PyObject*)new_event;  // 返回 new_event 的 PyObject 指针形式
      END_HANDLE_TH_ERRORS  // 结束错误处理
    }
}

// 定义 THPStream_repr 函数，返回 THPStream 对象的字符串表示
static PyObject* THPStream_repr(THPStream* self) {
  HANDLE_TH_ERRORS
  // 构造表示 THPStream 对象的字符串，包括设备类型、设备索引和流ID
  return THPUtils_packString(
      "torch.Stream device_type=" +
      c10::DeviceTypeName(
          static_cast<c10::DeviceType>(self->device_type), true) +
      ", device_index=" + std::to_string(self->device_index) +
      ", stream_id=" + std::to_string(self->stream_id));
  END_HANDLE_TH_ERRORS
}

// 定义 THPStream_hash 函数，计算 THPStream 对象的哈希值
static Py_hash_t THPStream_hash(THPStream* self) {
  // 使用 at::hash_combine 计算结合了设备类型、流ID和设备索引的哈希值
  return static_cast<long>(at::hash_combine(
      self->device_type,
      (at::hash_combine(self->stream_id, self->device_index))));
}

// 定义 THPStream_eq 函数，判断两个 THPStream 对象是否相等
static PyObject* THPStream_eq(THPStream* self, THPStream* other) {
  HANDLE_TH_ERRORS
  // 比较两个 THPStream 对象的设备类型、设备索引和流ID是否完全相同
  return PyBool_FromLong(
      (self->stream_id == other->stream_id) &&
      (self->device_index == other->device_index) &&
      (self->device_type == other->device_type));
  END_HANDLE_TH_ERRORS
}

// 定义 THPStream_ne 函数，判断两个 THPStream 对象是否不相等
static PyObject* THPStream_ne(THPStream* self, THPStream* other) {
  HANDLE_TH_ERRORS
  // 比较两个 THPStream 对象的设备类型、设备索引和流ID是否有任意不同
  return PyBool_FromLong(
      (self->stream_id != other->stream_id) ||
      (self->device_index != other->device_index) ||
      (self->device_type != other->device_type));
  END_HANDLE_TH_ERRORS
}

// 定义 THPStream_richcompare 函数，实现 THPStream 对象的比较操作
static PyObject* THPStream_richcompare(
    PyObject* self,
    PyObject* other,
    int op) {
  PyObject* result = NULL;
  // 如果与比较对象是空，则返回 False
  if (other == Py_None) {
    result = Py_False;
  } else {
    // 根据比较操作符选择执行相等或不等比较函数
    switch (op) {
      case Py_EQ:
        result = THPStream_eq((THPStream*)self, (THPStream*)other);
        break;
      case Py_NE:
        result = THPStream_ne((THPStream*)self, (THPStream*)other);
        break;
      default:
        // 默认情况下返回 False
        result = Py_False;
        break;
    }
  }
  // 增加结果的引用计数并返回
  Py_XINCREF(result);
  return result;
}

// 定义 THPStream_members 数组，包含 THPStream 对象的成员变量定义
// 每个成员变量包括名称、类型、偏移量和属性标记
static struct PyMemberDef THPStream_members[] = {
    {"stream_id",
     T_LONGLONG,
     offsetof(THPStream, stream_id),
     READONLY,
     nullptr},
    {"device_index",
     T_LONGLONG,
     offsetof(THPStream, device_index),
     READONLY,
     nullptr},
    {"device_type",
     T_LONGLONG,
     offsetof(THPStream, device_type),
     READONLY,
     nullptr},
    {nullptr}};

// 定义 THPStream_properties 数组，包含 THPStream 对象的属性定义
// 每个属性包括名称、getter 函数指针以及 setter 函数指针，此处没有 setter 所以为 nullptr
static struct PyGetSetDef THPStream_properties[] = {
    {"device", (getter)THPStream_get_device, nullptr, nullptr, nullptr},
    {nullptr}};

// 定义 THPStream_methods 数组，包含 THPStream 对象的方法定义
// 每个方法包括名称、对应的 C 函数指针、调用方式标记和文档字符串（此处为 nullptr 表示没有文档）
static PyMethodDef THPStream_methods[] = {
    {"query", THPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THPStream_synchronize, METH_NOARGS, nullptr},
    {"wait_event", THPStream_wait_event, METH_O, nullptr},
    {"wait_stream", THPStream_wait_stream, METH_O, nullptr},
    {"record_event",
     castPyCFunctionWithKeywords(THPStream_record_event),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
    {nullptr}};



    // 定义一个静态数组，包含对象方法名 "__eq__" 和对应的函数指针 THPStream_eq，
    // 以及参数传递方式 METH_O（单个对象作为参数），最后一个元素是 nullptr 表示数组结束
    {"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
    // 最后一个元素是 nullptr，表示数组的结尾
    {nullptr}};
PyTypeObject THPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.Stream", /* tp_name */
    sizeof(THPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPStream_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    (hashfunc)THPStream_hash, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    THPStream_richcompare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPStream_methods, /* tp_methods */
    THPStream_members, /* tp_members */
    THPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPStream_pynew, /* tp_new */
};

void THPStream_init(PyObject* module) {
    // 将 THPStreamType 指向定义好的类型对象
    THPStreamClass = &THPStreamType;
    // 设置类型对象的基类为 PyType_Type
    Py_SET_TYPE(&THPStreamType, &PyType_Type);
    // 如果类型对象的准备过程失败，抛出 Python 错误
    if (PyType_Ready(&THPStreamType) < 0) {
        throw python_error();
    }
    // 增加类型对象的引用计数
    Py_INCREF(&THPStreamType);
    // 将类型对象添加到给定的模块中，对象名为 "Stream"
    if (PyModule_AddObject(module, "Stream", (PyObject*)&THPStreamType) < 0) {
        throw python_error();
    }
}
```