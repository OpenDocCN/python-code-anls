# `.\pytorch\torch\csrc\Device.cpp`

```
// 引入 Torch 的设备相关头文件

#include <torch/csrc/Device.h>

// 引入 Torch 的异常处理相关头文件
#include <torch/csrc/Exceptions.h>

// 引入 Torch 的对象指针工具头文件
#include <torch/csrc/utils/object_ptr.h>

// 引入 Torch 的 Python 绑定工具头文件
#include <torch/csrc/utils/pybind.h>

// 引入 Torch 的 Python 参数解析工具头文件
#include <torch/csrc/utils/python_arg_parser.h>

// 引入 Torch 的 Python 数字处理工具头文件
#include <torch/csrc/utils/python_numbers.h>

// 引入 Torch 的 Python 字符串处理工具头文件
#include <torch/csrc/utils/python_strings.h>

// 引入 ATen 的设备相关头文件
#include <ATen/Device.h>

// 引入 C10 的异常处理工具头文件
#include <c10/util/Exception.h>

// 引入 C 语言结构体成员相关头文件
#include <structmember.h>

// 引入 C++ 标准库中的数值极限相关头文件
#include <limits>

// 引入 C++ 标准库中的字符串流处理相关头文件
#include <sstream>

// 定义全局变量指针 PyObject* THPUpperModuleOfDevice，并禁止进行 NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) 类型的检查
PyObject* THPUpperModuleOfDevice = nullptr;

// 根据给定的 ATen 设备对象创建新的 THPDevice 对象并返回
PyObject* THPDevice_New(const at::Device& device) {
  auto type = (PyTypeObject*)&THPDeviceType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDevice*>(self.get());
  self_->device = device;
  return self.release();
}

// 返回给定 THPDevice 对象的字符串表示形式
PyObject* THPDevice_repr(THPDevice* self) {
  std::ostringstream oss;
  oss << "device(type=\'" << self->device.type() << "\'";
  if (self->device.has_index()) {
    // 将 uint8_t 类型的 self->device.index() 强制转换为 uint16_t，在打印时以 ASCII 形式处理
    // 参考：https://stackoverflow.com/questions/19562103/uint8-t-cant-be-printed-with-cout
    oss << ", index=" << static_cast<uint16_t>(self->device.index());
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}

// 返回给定 THPDevice 对象的字符串形式
PyObject* THPDevice_str(THPDevice* self) {
  std::ostringstream oss;
  oss << self->device;
  return THPUtils_packString(oss.str().c_str());
}

// 在 Python 中创建 THPDevice 对象的新实例
PyObject* THPDevice_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // 定义静态的 PythonArgParser 对象 parser，支持两种函数签名的解析
  static torch::PythonArgParser parser(
      {"device(Device device)",
       "device(c10::string_view type, int64_t? index=-1)"});

  // 解析参数并将结果存储在 parsed_args 中
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 如果解析结果包含 Torch 函数，则调用 handle_torch_function 处理
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPUpperModuleOfDevice, "torch");
  }

  // 根据解析索引选择相应的处理分支
  if (r.idx == 0) {
    // 解析第一个参数为 Device 对象，并返回对应的 THPDevice_New 实例
    auto device = r.device(0);
    return THPDevice_New(device);
  } else if (r.idx == 1) {
    // 解析第一个参数为字符串类型的设备类型，第二个参数为可选的设备索引
    auto as_device = r.device(0); // this works, because device can take strings

    // 如果设备类型包含索引信息，则抛出异常
    if (as_device.has_index()) {
      auto device_type = r.string(0);
      throw std::runtime_error(
          "type (string) must not include an index because index "
          "was passed explicitly: " +
          device_type);
    }

    // 解析设备索引，确保不为负数
    int64_t device_index = -1;
    if (!r.isNone(1)) {
      device_index = r.toInt64(1);
      // 在 ATen/C++ 中允许 -1 表示默认设备，但在 Python 中不允许
      TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    }

    // 创建并返回新的 ATen 设备对象
    at::Device device(
        as_device.type(), static_cast<c10::DeviceIndex>(device_index));
    return THPDevice_New(device);
  }

  // 默认情况下返回 None
  Py_RETURN_NONE;

  END_HANDLE_TH_ERRORS
}
PyObject* THPDevice_type(THPDevice* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 创建一个字符串流对象
  std::ostringstream oss;
  // 将 self->device.type() 的值写入 oss
  oss << self->device.type();
  // 将 oss 中的内容封装成 Python 字符串对象并返回
  return THPUtils_packString(oss.str().c_str());
  // 返回 None 对象（仅在出现异常时执行）
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPDevice_index(THPDevice* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 如果 self->device 有索引值
  if (self->device.has_index()) {
    // 将 self->device.index() 封装成 Python 整数对象并返回
    return THPUtils_packInt64(self->device.index());
  } else {
    // 返回 None 对象
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPDevice_hash(THPDevice* self) {
  HANDLE_TH_ERRORS
  // 返回 self->device 的哈希值对 std::numeric_limits<Py_ssize_t>::max() 取模的结果
  return static_cast<Py_ssize_t>(
      std::hash<at::Device>{}(self->device) %
      std::numeric_limits<Py_ssize_t>::max());
  // 返回 -1（仅在出现异常时执行）
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPDevice_rc(PyObject* a, PyObject* b, int op) {
  HANDLE_TH_ERRORS
  // 如果 a 或 b 不是 THPDevice 对象
  if (!THPDevice_Check(a) || !THPDevice_Check(b)) {
    // 返回 NotImplemented 对象（仅在 Python 2 中存在）
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }
  // 将 a 和 b 转换为 THPDevice 指针
  THPDevice* da = reinterpret_cast<THPDevice*>(a);
  THPDevice* db = reinterpret_cast<THPDevice*>(b);

  // 根据 op 进行不同的比较操作
  switch (op) {
    case Py_EQ:
      // 如果 da 和 db 的 device 相等，则返回 True 对象
      if (da->device == db->device) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      // 如果 da 和 db 的 device 不相等，则返回 True 对象
      if (da->device == db->device) {
        Py_RETURN_FALSE;
      } else {
        Py_RETURN_TRUE;
      }
    case Py_LT:
    case Py_LE:
    case Py_GT:
    case Py_GE:
      // 抛出异常，因为这些比较操作未实现
      throw torch::TypeError("comparison not implemented");
    default:
      // 抛出异常，因为 op 是意外的比较操作
      throw torch::TypeError("unexpected comparison op");
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPDevice_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THPDevice 指针
  auto self = (THPDevice*)_self;
  // 创建一个 Python 元组对象，包含两个元素
  auto ret = THPObjectPtr{PyTuple_New(2)};
  if (!ret)
    throw python_error();

  // 导入 torch 模块并获取 device 对象
  py::object torch_module = py::module::import("torch");
  py::object torch_device = torch_module.attr("device");
  // 将 torch_device 作为第一个元组元素
  PyTuple_SET_ITEM(ret.get(), 0, torch_device.release().ptr());

  // 创建参数元组 args，包含设备类型和索引（如果有）
  THPObjectPtr args;
  std::ostringstream oss;
  oss << self->device.type();
  if (self->device.has_index()) {
    args = THPObjectPtr{Py_BuildValue(
        "(si)", oss.str().c_str(), static_cast<int>(self->device.index()))};
  } else {
    args = THPObjectPtr{Py_BuildValue("(s)", oss.str().c_str())};
  }
  if (!args)
    throw python_error();
  // 将 args 作为第二个元组元素
  PyTuple_SET_ITEM(ret.get(), 1, args.release());

  // 返回元组对象 ret
  return ret.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDevice_enter(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 导入 torch.utils._device 模块并创建 DeviceContext 对象
  py::object mode = py::module::import("torch.utils._device")
                        .attr("DeviceContext")(py::handle(self));
  // 将 mode 对象推入 PythonTorchFunctionTLS 的堆栈
  at::impl::PythonTorchFunctionTLS::push_onto_stack(
      std::make_shared<c10::SafePyObject>(
          mode.release().ptr(), getPyInterpreter()));
  // 增加 self 的引用计数并返回 self 对象
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}
PyObject* THPDevice_exit(PyObject* self, PyObject* unused) {
  HANDLE_TH_ERRORS
  // 弹出 Python Torch 函数 TLS 栈中的顶部元素
  at::impl::PythonTorchFunctionTLS::pop_stack();
  // 返回 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPDevice_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 导入 torch.utils._device 模块，并获取 device_decorator 属性
  py::object deco =
      py::module::import("torch.utils._device").attr("device_decorator");
  // 调用获取的 decorator，传递 self, args 和 kwargs，并释放其所有权
  return deco(py::handle(self), *py::handle(args), **py::handle(kwargs))
      .release()
      .ptr();
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

// NB: If you edit these properties/methods, update torch/_C/__init__.pyi.in

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef THPDevice_properties[] = {
    // 定义 torch.device 类的属性列表，包含 type 和 index
    {"type", (getter)THPDevice_type, nullptr, nullptr, nullptr},
    {"index", (getter)THPDevice_index, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static PyMethodDef THPDevice_methods[] = {
    // 定义 torch.device 类的方法列表，包含 __reduce__, __enter__, __exit__
    {"__reduce__", THPDevice_reduce, METH_NOARGS, nullptr},
    {"__enter__", THPDevice_enter, METH_NOARGS, nullptr},
    {"__exit__", THPDevice_exit, METH_VARARGS, nullptr},
    {nullptr} /* Sentinel */
};

PyTypeObject THPDeviceType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.device", /* tp_name */
    sizeof(THPDevice), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPDevice_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    (hashfunc)THPDevice_hash, /* tp_hash  */
    // TODO: We're not sure if this is a good idea or not, because making
    // torch.device callable means that it will start returning true
    // for callable() queries, and that is unexpected.  We can always add
    // this later, so for now, don't actually implement this
    // THPDevice_call, /* tp_call */
    nullptr, /* tp_call */
    (reprfunc)THPDevice_str, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    (richcmpfunc)THPDevice_rc, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPDevice_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPDevice_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPDevice_pynew, /* tp_new */
};
void THPDevice_init(PyObject* module) {
    // 初始化 THPDeviceType 类型对象，如果失败则抛出异常
    if (PyType_Ready(&THPDeviceType) < 0) {
        throw python_error();
    }
    // 增加 THPDeviceType 的引用计数，防止被释放
    Py_INCREF(&THPDeviceType);
    // 将当前模块设为 THPUpperModuleOfDevice
    THPUpperModuleOfDevice = module;
    // 将 THPDeviceType 对象添加到模块中，如果失败则抛出异常
    if (PyModule_AddObject(module, "device", (PyObject*)&THPDeviceType) != 0) {
        throw python_error();
    }
}
```