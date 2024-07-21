# `.\pytorch\torch\csrc\Generator.cpp`

```
#include <torch/csrc/Generator.h>

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <structmember.h>

#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_types.h>

#include <utility>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#ifdef USE_MPS
#include <ATen/mps/MPSGeneratorImpl.h>
#endif

using namespace at;
using namespace torch;

// 全局变量，用于存储 THPGenerator 类的 Python 对象
PyObject* THPGeneratorClass = nullptr;

// 初始化默认生成器的 Python 对象
PyObject* THPGenerator_initDefaultGenerator(at::Generator cdata) {
  auto type = (PyTypeObject*)THPGeneratorClass;
  // 分配内存并创建 THPGenerator 对象
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  // 将传入的 Generator 对象移动到 self->cdata 中
  self_->cdata = std::move(cdata);
  return self.release();  // 返回 Python 对象的指针
}

// THPGenerator 对象的析构函数
static void THPGenerator_dealloc(PyObject* _self) {
  auto self = reinterpret_cast<THPGenerator*>(_self);
  // 如果 Generator 对象已定义，则置空 Python 对象并销毁 Generator
  if (self->cdata.defined()) {
    self->cdata.set_pyobj(nullptr);
    self->cdata.~Generator();
  }
  Py_TYPE(_self)->tp_free(_self);  // 释放 Python 对象的内存
}

// 创建新的 THPGenerator 对象的 Python 工厂函数
static PyObject* THPGenerator_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 错误处理开始
  static torch::PythonArgParser parser({"Generator(Device device=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));

  THPGeneratorPtr self((THPGenerator*)type->tp_alloc(type, 0));
  // 根据设备类型创建不同类型的 Generator 对象
  if (device.type() == at::kCPU) {
    self->cdata = make_generator<CPUGeneratorImpl>();
  }
#ifdef USE_CUDA
  else if (device.type() == at::kCUDA) {
    self->cdata = make_generator<CUDAGeneratorImpl>(device.index());
  }
#endif
#ifdef USE_MPS
  else if (device.type() == at::kMPS) {
    self->cdata = make_generator<MPSGeneratorImpl>();
  }
#endif
  else if (device.type() == at::kXPU) {
    self->cdata = at::detail::getXPUHooks().getXPUGenerator(device.index());
  } else if (device.type() == at::kIPU) {
    self->cdata = at::detail::getIPUHooks().newIPUGenerator(device.index());
  } else if (device.type() == at::kPrivateUse1) {
    self->cdata = at::GetGeneratorForPrivateuse1(device.index());
  } else {
    // 抛出错误，指示不支持的设备类型
    AT_ERROR(
        "Device type ",
        c10::DeviceTypeName(device.type()),
        " is not supported for torch.Generator() api.");
  }
  return (PyObject*)self.release();  // 返回创建的 THPGenerator 对象
  END_HANDLE_TH_ERRORS  // 错误处理结束
}
// 获取生成器状态的函数，返回一个包含生成器状态的 Torch 张量的 Python 对象
static PyObject* THPGenerator_getState(PyObject* _self, PyObject* noargs) {
  // 使用 torch::autograd 命名空间
  using namespace torch::autograd;
  HANDLE_TH_ERRORS

  // 获取 C++ 中的生成器对象
  auto& gen = ((THPGenerator*)_self)->cdata;

  // 查看“使用随机生成器时获取锁”的说明
  // 创建一个互斥锁，以确保在使用随机生成器时线程安全
  std::scoped_lock<std::mutex> lock(gen.mutex());

  // 调用生成器对象的 get_state 方法获取其状态
  auto state_tensor = gen.get_state();

  // 将获取的状态张量包装为一个 Torch 张量的 Python 对象并返回
  return THPVariable_Wrap(std::move(state_tensor));

  END_HANDLE_TH_ERRORS
}

// 设置生成器状态的函数，接受一个表示新状态的 Torch 张量的 Python 对象
static PyObject* THPGenerator_setState(PyObject* _self, PyObject* _new_state) {
  // 使用 torch::autograd 命名空间
  using namespace torch::autograd;

  HANDLE_TH_ERRORS

  // 检查传入的 _new_state 是否为 torch.ByteTensor 类型，否则抛出类型错误
  if (!THPVariable_Check(_new_state)) {
    throw torch::TypeError(
        "expected a torch.ByteTensor, but got %s",
        Py_TYPE(_new_state)->tp_name);
  }

  // 将 _self 转换为 THPGenerator 指针
  auto self = (THPGenerator*)_self;
  // 获取 C++ 中的生成器对象
  auto& gen = self->cdata;
  // 解包 _new_state，获取其内部的 Torch 张量对象
  const auto& new_state_tensor = THPVariable_Unpack(_new_state);

  // 查看“使用随机生成器时获取锁”的说明
  // 创建一个互斥锁，以确保在使用随机生成器时线程安全
  std::scoped_lock<std::mutex> lock(gen.mutex());

  // 调用生成器对象的 set_state 方法设置新状态
  gen.set_state(new_state_tensor);

  // 增加 self 的 Python 引用计数并返回 self 对象
  Py_INCREF(self);
  return (PyObject*)self;

  END_HANDLE_TH_ERRORS
}

// 将 Python 对象解包为 uint64_t 类型的函数
uint64_t unpack_uint64(PyObject* pyobj) {
  uint64_t unsigned_obj = 0;
  try {
    // 首先尝试将 pyobj 解包为 unsigned long 类型
    unsigned_obj = THPUtils_unpackUInt64(pyobj);
  } catch (...) {
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      // 如果发生溢出，尝试将 pyobj 解包为 signed long 类型
      PyErr_Clear();
      int64_t obj = THPUtils_unpackLong(pyobj);
      unsigned_obj = *(reinterpret_cast<uint64_t*>(&obj));
    } else {
      // 如果发生其他类型的异常，重新抛出该异常
      throw;
    }
  }
  // 返回解包后的 uint64_t 对象
  return unsigned_obj;
}

// 获取生成器状态的图安全版本函数，返回一个包含生成器状态的 Python 对象
static PyObject* THPGenerator_graphSafeGetState(
    PyObject* _self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取 C++ 中的生成器对象
  auto& gen = ((THPGenerator*)_self)->cdata;

  // 查看“使用随机生成器时获取锁”的说明
  // 创建一个互斥锁，以确保在使用随机生成器时线程安全
  std::scoped_lock<std::mutex> lock(gen.mutex());

  // 调用生成器对象的 graphsafe_get_state 方法获取其状态
  return THPGenerator_Wrap(gen.graphsafe_get_state());

  END_HANDLE_TH_ERRORS
}

// 设置生成器状态的图安全版本函数，接受一个表示新状态的 Python 对象
static PyObject* THPGenerator_graphSafeSetState(
    PyObject* _self,
    PyObject* _state) {
  HANDLE_TH_ERRORS
  // 将 _self 转换为 THPGenerator 指针
  auto self = (THPGenerator*)_self;
  // 获取 C++ 中的生成器对象
  auto& gen = self->cdata;

  // 查看“使用随机生成器时获取锁”的说明
  // 创建一个互斥锁，以确保在使用随机生成器时线程安全
  std::scoped_lock<std::mutex> lock(gen.mutex());

  // 解包 _state 获取其中的生成器状态，并调用 gen 的 graphsafe_set_state 方法设置新状态
  gen.graphsafe_set_state(THPGenerator_Unwrap(_state));

  // 增加 self 的 Python 引用计数并返回 self 对象
  Py_INCREF(self);
  return (PyObject*)self;

  END_HANDLE_TH_ERRORS
}

// 克隆生成器状态的函数，返回一个新生成器对象的 Python 对象
static PyObject* THPGenerator_cloneState(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取 C++ 中的生成器对象
  auto& gen = ((THPGenerator*)_self)->cdata;

  // 查看“使用随机生成器时获取锁”的说明
  // 创建一个互斥锁，以确保在使用随机生成器时线程安全
  std::scoped_lock<std::mutex> lock(gen.mutex());

  // 调用生成器对象的 clone 方法创建一个新的生成器对象
  auto new_generator = gen.clone();

  // 将新生成器对象包装为 Python 对象并返回
  return THPGenerator_Wrap(new_generator);

  END_HANDLE_TH_ERRORS
}
// 设置手动种子的函数，用于 Python 的包装器
static PyObject* THPGenerator_manualSeed(PyObject* _self, PyObject* seed) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  auto generator = self->cdata;  // 获取 THGenerator 实例
  TORCH_CHECK(
      THPUtils_checkLong(seed),  // 检查 seed 是否为长整型
      "manual_seed expected a long, "
      "but got ",
      THPUtils_typename(seed));  // 输出错误信息，显示实际输入的类型
  uint64_t unsigned_seed = unpack_uint64(seed);  // 将 Python 对象 seed 解包成无符号 64 位整数
  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(generator.mutex());  // 获取 generator 的互斥锁
  generator.set_current_seed(unsigned_seed);  // 设置 generator 的当前种子值
  Py_INCREF(self);  // 增加 self 的引用计数，避免 Python 对象被回收
  return (PyObject*)self;  // 返回 self 对象
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 设置偏移量的函数，用于 Python 的包装器
static PyObject* THPGenerator_setOffset(PyObject* _self, PyObject* offset) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  auto generator = self->cdata;  // 获取 THGenerator 实例
  TORCH_CHECK(
      THPUtils_checkLong(offset),  // 检查 offset 是否为长整型
      "manual_offset expected a long, "
      "but got ",
      THPUtils_typename(offset));  // 输出错误信息，显示实际输入的类型
  uint64_t unsigned_offset = unpack_uint64(offset);  // 将 Python 对象 offset 解包成无符号 64 位整数
  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(generator.mutex());  // 获取 generator 的互斥锁
  generator.set_offset(unsigned_offset);  // 设置 generator 的偏移量
  Py_INCREF(self);  // 增加 self 的引用计数，避免 Python 对象被回收
  return (PyObject*)self;  // 返回 self 对象
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 获取种子值的函数，用于 Python 的包装器
static PyObject* THPGenerator_seed(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // See Note [Acquire lock when using random generators]
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  std::scoped_lock<std::mutex> lock(self->cdata.mutex());  // 获取 generator 的互斥锁
  uint64_t seed_val = self->cdata.seed();  // 获取 generator 的当前种子值
  return THPUtils_packUInt64(seed_val);  // 将种子值打包成 Python 的整数对象并返回
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 获取初始种子值的函数，用于 Python 的包装器
static PyObject* THPGenerator_initialSeed(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  return THPUtils_packUInt64(self->cdata.current_seed());  // 获取 generator 的当前种子值并打包返回
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 获取偏移量的函数，用于 Python 的包装器
static PyObject* THPGenerator_getOffset(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  return THPUtils_packUInt64(self->cdata.get_offset());  // 获取 generator 的偏移量并打包返回
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 获取设备信息的函数，用于 Python 的包装器
static PyObject* THPGenerator_get_device(THPGenerator* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata.device());  // 创建包含 generator 设备信息的 Python 对象并返回
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常
}

// 函数用于序列化 generator 的状态，用于 Python 的包装器
PyObject* THPGenerator_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;  // 将输入的 self 转换为 THPGenerator 类型
  auto& gen = self->cdata;  // 获取 generator 实例的引用

  auto ret = THPObjectPtr{PyTuple_New(3)};  // 创建一个包含三个元素的 Python 元组
  if (!ret)
    throw python_error();  // 如果创建失败则抛出 Python 异常

  py::object torch_module = py::module::import("torch");  // 导入 torch 模块
  py::object torch_generator = torch_module.attr("Generator");  // 获取 Generator 类对象
  PyTuple_SET_ITEM(ret.get(), 0, torch_generator.release().ptr());  // 将 Generator 类对象放入元组中

  auto args = THPObjectPtr{PyTuple_New(1)};  // 创建一个包含一个元素的 Python 元组
  if (!args)
    throw python_error();  // 如果创建失败则抛出 Python 异常

  PyTuple_SET_ITEM(args.get(), 0, THPGenerator_get_device(self, nullptr));  // 将 generator 的设备信息放入元组
  PyTuple_SET_ITEM(ret.get(), 1, args.release());  // 将 args 元组放入 ret 元组的第二个位置

  auto state = THPObjectPtr{PyTuple_New(3)};  // 创建一个包含三个元素的 Python 元组
  if (!state)
    // 抛出一个 Python 错误
    throw python_error();

  // 获取生成器的设备类型
  c10::DeviceType device_type = gen.device().type();
  // 将初始种子放入状态元组的第一个位置
  PyTuple_SET_ITEM(state.get(), 0, THPGenerator_initialSeed(_self, nullptr));
  // 根据设备类型确定是否需要获取偏移量，放入状态元组的第二个位置
  PyTuple_SET_ITEM(
      state.get(),
      1,
      device_type != at::kCPU ? THPGenerator_getOffset(_self, nullptr)
                              : Py_None);
  // 获取生成器的完整状态，放入状态元组的第三个位置
  PyTuple_SET_ITEM(state.get(), 2, THPGenerator_getState(_self, nullptr));
  // 将状态元组作为第三个元素放入返回值元组
  PyTuple_SET_ITEM(ret.get(), 2, state.release());

  // 返回最终的返回值元组
  return ret.release();
  END_HANDLE_TH_ERRORS
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
// 定义 THPGenerator_properties 数组，包含一个 PyGetSetDef 结构体，用于定义 Python 对象属性的 getter 和 setter
static struct PyGetSetDef THPGenerator_properties[] = {
    {"device", (getter)THPGenerator_get_device, nullptr, nullptr, nullptr}, // 定义属性 "device"，使用 THPGenerator_get_device 作为 getter
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
// 定义 THPGenerator_methods 数组，包含一系列 PyMethodDef 结构体，定义了 THPGenerator 类型的方法及其行为描述
static PyMethodDef THPGenerator_methods[] = {
    {"__reduce__", THPGenerator_reduce, METH_NOARGS, nullptr}, // 方法 "__reduce__"，无参数，用于对象序列化
    {"__setstate__", THPGenerator_pickleSetState, METH_O, nullptr}, // 方法 "__setstate__"，带一个参数 state，用于反序列化对象状态
    {"get_state", THPGenerator_getState, METH_NOARGS, nullptr}, // 方法 "get_state"，无参数，获取对象状态
    {"set_state", THPGenerator_setState, METH_O, nullptr}, // 方法 "set_state"，带一个参数 state，设置对象状态
    {"clone_state", THPGenerator_cloneState, METH_NOARGS, nullptr}, // 方法 "clone_state"，无参数，克隆对象状态
    {"graphsafe_get_state",
     THPGenerator_graphSafeGetState,
     METH_NOARGS,
     nullptr}, // 方法 "graphsafe_get_state"，无参数，获取对象状态（图安全）
    {"graphsafe_set_state", THPGenerator_graphSafeSetState, METH_O, nullptr}, // 方法 "graphsafe_set_state"，带一个参数 state，设置对象状态（图安全）
    {"set_offset", THPGenerator_setOffset, METH_O, nullptr}, // 方法 "set_offset"，带一个参数 offset，设置随机数生成器的偏移量
    {"manual_seed", THPGenerator_manualSeed, METH_O, nullptr}, // 方法 "manual_seed"，带一个参数 seed，手动设置随机数种子
    {"seed", THPGenerator_seed, METH_NOARGS, nullptr}, // 方法 "seed"，无参数，获取当前随机数种子
    {"initial_seed", THPGenerator_initialSeed, METH_NOARGS, nullptr}, // 方法 "initial_seed"，无参数，获取初始随机数种子
    {"get_offset", THPGenerator_getOffset, METH_NOARGS, nullptr}, // 方法 "get_offset"，无参数，获取当前随机数生成器的偏移量
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
// 定义 THPGenerator_members 数组，包含一个 PyMemberDef 结构体，定义了 THPGenerator 类型的成员变量及其描述
static struct PyMemberDef THPGenerator_members[] = {
    {"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, nullptr}, // 成员变量 "_cdata"，类型为 T_ULONGLONG，位于 THPGenerator 结构体中的偏移量，只读
    {nullptr}};

// 定义 PyTypeObject 结构体 THPGeneratorType，表示 Python 类型对象 torch._C.Generator 的结构
PyTypeObject THPGeneratorType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C.Generator", // Python 类型对象的头部初始化
    sizeof(THPGenerator), // 对象的基本大小
    0, // 对象的项目大小
    THPGenerator_dealloc, // 对象的销毁函数
    0, // 向量调用偏移量
    nullptr, // 获取对象属性的函数
    nullptr, // 设置对象属性的函数
    nullptr, // 保留字段
    nullptr, // 对象的字符串表示形式
    nullptr, // 数值运算
    nullptr, // 序列操作
    nullptr, // 映射操作
    nullptr, // 哈希计算
    nullptr, // 对象的调用操作
    nullptr, // 对象的字符串表示形式
    nullptr, // 获取对象属性
    nullptr, // 设置对象属性
    nullptr, // 缓冲区接口
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // 对象的标志
    nullptr, // 对象的文档字符串
    nullptr, // 遍历对象的函数
    nullptr, // 清除对象的函数
    nullptr, // 对象的比较函数
    0, // 弱引用列表偏移量
    nullptr, // 对象的迭代器
    nullptr, // 迭代器的下一个元素
    THPGenerator_methods, // 对象的方法集合
    THPGenerator_members, // 对象的成员变量集合
    THPGenerator_properties, /* tp_getset */
    # THPGenerator_properties 是一个结构体或变量，用于处理属性的获取和设置
    nullptr, /* tp_base */
    # tp_base 通常用于指定此类型的基类，这里为 nullptr 表示没有基类
    nullptr, /* tp_dict */
    # tp_dict 通常用于存储类型的属性字典，这里为 nullptr 表示没有额外的属性字典
    nullptr, /* tp_descr_get */
    # tp_descr_get 用于描述获取属性的方法，这里为 nullptr 表示没有定义
    nullptr, /* tp_descr_set */
    # tp_descr_set 用于描述设置属性的方法，这里为 nullptr 表示没有定义
    0, /* tp_dictoffset */
    # tp_dictoffset 是一个整数，通常指定属性字典的偏移量
    nullptr, /* tp_init */
    # tp_init 用于对象初始化的方法，这里为 nullptr 表示没有定义
    nullptr, /* tp_alloc */
    # tp_alloc 用于对象分配内存的方法，这里为 nullptr 表示没有定义
    THPGenerator_pynew, /* tp_new */
    # THPGenerator_pynew 是一个函数或方法，用于创建新对象的方法
};

`
// End of code block
};

// Initialize the THPGeneratorClass as a PyObject pointer to THPGeneratorType
// If PyType_Ready(&THPGeneratorType) succeeds, increment its reference count and add it to the module
bool THPGenerator_init(PyObject* module) {
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject*)&THPGeneratorType);
  return true;
}

// Set the PyObject associated with the Generator object 'self' to 'pyobj'
void set_pyobj(const Generator& self, PyObject* pyobj) {
  TORCH_CHECK(self.defined(), "cannot call set_pyobj() on undefined generator");
  self.set_pyobj(pyobj);
}

// Return the PyObject associated with the Generator object 'self'
PyObject* pyobj(const Generator& self) {
  TORCH_CHECK(self.defined(), "cannot call pyobj() on undefined generator");
  return self.pyobj();
}

// Wrap a Generator 'gen' into a Python object; return Py_None if 'gen' is undefined
PyObject* THPGenerator_Wrap(Generator gen) {
  if (!gen.defined()) {
    Py_RETURN_NONE;
  }

  // If a PyObject already exists for 'gen', increment its reference count and return it
  if (auto obj = pyobj(gen)) {
    Py_INCREF(obj);
    return obj;
  }

  // Otherwise, create a new Python object for 'gen' using THPGenerator_NewWithVar
  return THPGenerator_NewWithVar((PyTypeObject*)THPGeneratorClass, std::move(gen));
}

// Unwrap a PyObject 'state' to obtain the Generator it represents
at::Generator THPGenerator_Unwrap(PyObject* state) {
  // Check if 'state' is of type THPGeneratorType; if not, raise a TypeError
  if (!Py_IS_TYPE(state, &THPGeneratorType)) {
    throw torch::TypeError(
        "expected a Generator, but got %s", Py_TYPE(state)->tp_name);
  }
  // Return the Generator object encapsulated in the THPGenerator 'state'
  return reinterpret_cast<THPGenerator*>(state)->cdata;
}

// Create a new Python object for a Generator 'gen'; associate 'gen' with 'obj'
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, Generator gen) {
  // Allocate memory for a new Python object of type 'type'
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto g = (THPGenerator*)obj;
    // Placement new to construct a Generator 'gen' within 'g->cdata'
    new (&g->cdata) Generator(std::move(gen));
    // Associate 'obj' with 'g->cdata' using set_pyobj
    set_pyobj(g->cdata, obj);
  }
  return obj;
}


What specific aspect of programming or software development are you currently focused on or interested in?
```