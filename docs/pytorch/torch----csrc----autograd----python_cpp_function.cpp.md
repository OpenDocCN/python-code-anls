# `.\pytorch\torch\csrc\autograd\python_cpp_function.cpp`

```py
// 包含头文件：c10/util/irange.h，提供了范围迭代的工具
#include <c10/util/irange.h>
// 包含头文件：torch/csrc/autograd/python_cpp_function.h，定义了与 Python 交互的 C++ 函数的相关功能
#include <torch/csrc/autograd/python_cpp_function.h>

// 包含标准库头文件
#include <torch/csrc/python_headers.h>
#include <cstdio>
#include <memory>
#include <typeindex>
#include <unordered_map>

// 包含 pybind11 库的头文件，用于 Python 和 C++ 之间的交互
#include <pybind11/pybind11.h>
// 包含头文件：torch/csrc/DynamicTypes.h，定义了动态类型相关的功能
#include <torch/csrc/DynamicTypes.h>
// 包含头文件：torch/csrc/Exceptions.h，定义了异常处理相关的功能
#include <torch/csrc/Exceptions.h>
// 包含头文件：torch/csrc/autograd/python_anomaly_mode.h，定义了与异常模式相关的功能
#include <torch/csrc/autograd/python_anomaly_mode.h>
// 包含头文件：torch/csrc/autograd/python_function.h，定义了与 Python 函数相关的功能
#include <torch/csrc/autograd/python_function.h>
// 包含头文件：torch/csrc/autograd/python_hook.h，定义了与钩子函数相关的功能
#include <torch/csrc/autograd/python_hook.h>
// 包含头文件：torch/csrc/autograd/python_variable.h，定义了与 Python 变量相关的功能
#include <torch/csrc/autograd/python_variable.h>
// 包含头文件：torch/csrc/utils/pybind.h，定义了 pybind11 库的辅助功能
#include <torch/csrc/utils/pybind.h>
// 包含头文件：torch/csrc/utils/python_numbers.h，定义了与 Python 数字相关的功能
#include <torch/csrc/utils/python_numbers.h>
// 包含头文件：torch/csrc/utils/python_strings.h，定义了与 Python 字符串相关的功能
#include <torch/csrc/utils/python_strings.h>

// 使用 torch::autograd 命名空间
using namespace torch::autograd;

// torch::autograd 命名空间内部
namespace torch::autograd {

// 匿名命名空间，用于定义私有函数或变量
namespace {

// THPCppFunction_call 函数实现，用于调用 C++ 函数对象
PyObject* THPCppFunction_call(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  // 检查 kwargs 是否不为空，如果不为空，抛出错误，不支持关键字参数
  if (kwargs && PyDict_Size(kwargs) != 0) {
    return PyErr_Format(PyExc_TypeError, "keyword arguments are not supported");
  }

  // 获取 args 中的参数数量
  auto num_inputs = PyTuple_GET_SIZE(args);
  // 获取 self 对象的输入参数数量
  auto num_inputs_required = ((THPCppFunction*)self)->cdata->num_inputs();
  // 如果传入参数数量与要求的数量不一致，抛出错误
  if (num_inputs != num_inputs_required) {
    return PyErr_Format(
        PyExc_TypeError,
        "expected %d arguments, got %d instead",
        num_inputs_required,
        num_inputs);
  }
  // 创建变量列表 vars，用于存储参数变量
  variable_list vars(num_inputs);
  // 遍历 args 中的参数
  for (int i = 0; i != num_inputs; ++i) {
    // 获取第 i 个参数
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    // 如果参数为 None，则跳过
    if (arg == Py_None) {
      continue;
    }
    // 检查参数是否为 THPVariable 类型，如果不是，抛出类型错误
    if (!THPVariable_Check(arg)) {
      return PyErr_Format(PyExc_TypeError, "argument %d is not a Variable", i);
    }
    // 将 THPVariable 转换为 C++ 的 Variable 类型，并存储到 vars 中
    vars[i] = THPVariable_Unpack(arg);
  }

  // 创建 output 变量列表
  variable_list output;

  // 处理 PyTorch 异常
  HANDLE_TH_ERRORS {
    // 释放全局解释器锁 GIL，允许在多线程环境下执行
    pybind11::gil_scoped_release nogil;
    // 调用 self 对象的函数调用操作符，传入 vars，并将结果存储到 output 中
    output = (*((THPCppFunction*)self)->cdata)(std::move(vars));
  }
  END_HANDLE_TH_ERRORS

  // 获取输出的变量数量
  auto num_outputs = output.size();
  // 如果输出变量数量为 1，则假设要解压缩一个元素元组，返回第一个元素的包装变量
  if (num_outputs == 1) {
    return THPVariable_Wrap(output[0]);
  }

  // 创建一个 Python 元组对象 tuple，用于存储多个输出变量
  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(num_outputs)));
  // 遍历 output 列表，将每个变量存储到 tuple 中
  for (size_t i = 0; i != num_outputs; ++i) {
    PyTuple_SET_ITEM(tuple.get(), i, THPVariable_Wrap(output[i]));
  }
  // 返回 tuple 对象，并释放其所有权
  return tuple.release();
}

// THPCppFunction_traverse 函数实现，用于遍历对象
int THPCppFunction_traverse(PyObject* self, visitproc visit, void* arg) {
  // 如果 self 对象的 cdata 引用计数为 1，说明仅有当前对象持有 cdata 引用
  if ((((THPCppFunction*)self)->cdata).use_count() == 1) {
    // 下面遍历的字段由 cpp grad_fn 拥有，我们持有其引用。
    // 只有当我们是 grad_fn 的唯一所有者时，才应该遍历它们，否则可能会导致过早释放 grad_fn。
    auto& fn = *((THPCppFunction*)self)->cdata;
    // 遍历 fn 中的 tensor_pre_hooks，并访问每个 hook 的字典对象
    for (const auto& hook : fn.tensor_pre_hooks()) {
      // 如果 hook 是 PyFunctionTensorPreHook 类型的实例，则访问其 dict 对象
      if (auto pyhook = dynamic_cast<PyFunctionTensorPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    // 注意事项 [retains_grad_hook PyObject traversal]
    // 理论上这不应该是必要的，因为 retains_grad_hooks 应该
    // 在实际使用中解释和弄清楚。
    // 遍历函数对象的所有保留梯度钩子，确保没有包含任何 PyFunctionTensorPreHooks。另一种方法是确保这一点的检查。
    for (const auto& pair : fn.retains_grad_hooks()) {
      // 如果钩子是 PyFunctionTensorPreHook 类型的，则访问其字典对象
      if (auto pyhook = dynamic_cast<PyFunctionTensorPreHook*>(pair.second.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    // 遍历函数对象的所有前置钩子
    for (const auto& hook : fn.pre_hooks()) {
      // 如果钩子是 PyFunctionPreHook 类型的，则访问其字典对象
      if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
    // 遍历函数对象的所有后置钩子
    for (const auto& hook : fn.post_hooks()) {
      // 如果钩子是 PyFunctionPostHook 类型的，则访问其字典对象
      if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
  }
  // 返回 0 表示正常退出
  return 0;
}

// 清除 THPCppFunction 对象的资源，包括清除可能存在的 C++ 对象的弱引用
int THPCppFunction_clear(PyObject* self) {
  auto f = (THPCppFunction*)self;
  // 如果存在 C++ 对象，则移除其弱引用
  if (f->cdata) {
    f->cdata->set_pyobj(nullptr);
  }
  // 重置 C++ 对象的智能指针
  f->cdata.reset();
  return 0;
}

// 释放 THPCppFunction 对象的内存
void THPCppFunction_dealloc(PyObject* self) {
  // 取消 Python 的垃圾回收跟踪
  PyObject_GC_UnTrack(self);
  // 清除 THPCppFunction 对象的资源
  THPCppFunction_clear(self);
  // 显式调用 C++ 对象的析构函数
  ((THPCppFunction*)self)->cdata.~shared_ptr();
  // 释放对象的内存
  Py_TYPE(self)->tp_free(self);
}

} // namespace

// 获取 THPCppFunction 对象的下一级函数（Python 元组）
PyObject* THPCppFunction_next_functions(PyObject* self, void* _unused) {
  // 获取 C++ 对象的指针
  auto cdata = reinterpret_cast<const THPCppFunction*>(self)->cdata;
  // 获取下一级函数的数量
  const auto num_next = cdata->num_outputs();
  // 创建 Python 元组来存储下一级函数
  THPObjectPtr py_functions(PyTuple_New(num_next));
  if (!py_functions)
    return nullptr;
  // 遍历下一级函数并创建 Python 元组
  for (const auto i : c10::irange(num_next)) {
    // 获取当前边缘的信息
    auto& c_tuple = cdata->next_edge(i);
    // 创建一个包含函数和索引的 Python 元组
    THPObjectPtr tuple(PyTuple_New(2));
    if (!tuple)
      return nullptr;
    // 将函数对象转换为 Python 对象并存储在元组中
    PyObject* py_fn = functionToPyObject(c_tuple.function);
    if (!py_fn)
      return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 0, py_fn);
    // 将索引转换为 Python 对象并存储在元组中
    PyObject* py_idx = THPUtils_packUInt32(c_tuple.input_nr);
    if (!py_idx)
      return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 1, py_idx);
    // 将当前元组添加到下一级函数元组中
    PyTuple_SET_ITEM(py_functions.get(), i, tuple.release());
  }
  return py_functions.release();
}

// 获取 THPCppFunction 对象的元数据
PyObject* THPCppFunction_metadata(PyObject* self, void* _unused) {
  // 获取元数据字典指针
  auto* metadata =
      static_cast<PyAnomalyMetadata*>(
          reinterpret_cast<THPCppFunction*>(self)->cdata->metadata())
          ->dict();
  // 增加元数据字典的引用计数并返回
  Py_XINCREF(metadata);
  return metadata;
}

// 返回 THPCppFunction 对象是否需要梯度跟踪
PyObject* THPCppFunction_requires_grad(PyObject* self, void* unused) {
  // 返回 Python 中的 True
  Py_RETURN_TRUE;
}

// 将变量注册为 THPCppFunction 对象的钩子字典
PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var) {
  // 检查 _var 是否为 THPVariable 对象
  if (!THPVariable_Check(_var)) {
    // 抛出类型错误，期望一个变量
    return PyErr_Format(
        PyExc_TypeError, "_register_hook_dict expected a variable");
  }
  // 转换 _var 为 THPVariable 指针
  auto var = (THPVariable*)_var;
  // 获取 THPCppFunction 对象的 C++ 数据对象并添加钩子
  auto& fn = *((THPCppFunction*)self)->cdata;
  fn.add_tensor_pre_hook(std::make_unique<PyFunctionTensorPreHook>(
      var->backward_hooks, THPVariable_Unpack(var).output_nr()));
  // 返回 None
  Py_RETURN_NONE;
}

// 注册钩子函数到 THPCppFunction 对象
PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook) {
  // 获取 THPCppFunction 对象的 C++ 数据对象并注册钩子
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionHook(fn, hook);
}

// 注册预钩子函数到 THPCppFunction 对象
PyObject* THPCppFunction_register_prehook(PyObject* self, PyObject* hook) {
  // 获取 THPCppFunction 对象的 C++ 数据对象并注册预钩子函数
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionPreHook(fn, hook);
}

// 获取 THPCppFunction 对象的名称
PyObject* THPCppFunction_name(PyObject* self, PyObject* noargs) {
  // 获取 THPCppFunction 对象的 C++ 数据对象并返回其名称
  auto& fn = *((THPCppFunction*)self)->cdata;
  return THPUtils_packString(fn.name());
}

// 获取 THPCppFunction 对象的序列号
PyObject* THPCppFunction_sequence_nr(PyObject* self, PyObject* noargs) {
  // 获取 THPCppFunction 对象的 C++ 数据对象并返回其序列号
  auto& fn = *((THPCppFunction*)self)->cdata;
  return THPUtils_packUInt64(fn.sequence_nr());
}

// 设置 THPCppFunction 对象的序列号
PyObject* THPCppFunction_set_sequence_nr(
    PyObject* self,
    PyObject* sequence_nr) {
  HANDLE_TH_ERRORS
  // 获取 THPCppFunction 对象的 C++ 数据对象并设置其序列号
  auto& fn = *((THPCppFunction*)self)->cdata;
  fn.set_sequence_nr(THPUtils_unpackUInt64(sequence_nr));
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
// 定义静态数组，存储默认的方法集合，以及 NOLINT 标记用于禁止特定的 Linter 检查
static struct PyMethodDef default_methods[] = {
    THP_FUNCTION_DEFAULT_METHODS,  // 插入预定义的方法
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
// 定义静态数组，存储默认的属性集合，以及 NOLINT 标记用于禁止特定的 Linter 检查
static struct PyGetSetDef default_properties[] = {
    THP_FUNCTION_DEFAULT_PROPERTIES,  // 插入预定义的属性
    {nullptr}};

// 初始化给定的 PyTypeObject 类型，并返回其指针
PyTypeObject* _initFunctionPyTypeObject(
    PyTypeObject& type,  // 要初始化的 PyTypeObject 对象的引用
    const char* name,    // 类型的名称
    PyGetSetDef* function_properties,  // 属性定义数组，可为空
    PyMethodDef* function_methods      // 方法定义数组，可为空
) {
  type.ob_base = {PyObject_HEAD_INIT(nullptr) 0};  // 初始化基础对象头部信息
  // NOLINTNEXTLINE(misc-redundant-expression)
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;  // 设置类型标志
  type.tp_name = name;  // 设置类型的名称
  type.tp_basicsize = sizeof(THPCppFunction);  // 设置类型的基本大小
  type.tp_call = THPCppFunction_call;  // 设置类型的调用方法
  // 设置类型的方法集合，若给定的方法数组为空则使用默认的方法集合
  type.tp_methods = function_methods ? function_methods : default_methods;
  // 设置类型的属性集合，若给定的属性数组为空则使用默认的属性集合
  type.tp_getset =
      function_properties ? function_properties : default_properties;
  type.tp_dealloc = THPCppFunction_dealloc;  // 设置类型的销毁方法
  type.tp_traverse = THPCppFunction_traverse;  // 设置类型的遍历方法
  type.tp_clear = THPCppFunction_clear;  // 设置类型的清除方法
  if (PyType_Ready(&type) < 0) {  // 准备类型对象，如果失败则抛出运行时错误
    auto msg = std::string("Unable to instantiate PyTypeObject for ") + name;
    throw std::runtime_error(msg);
  }
  return &type;  // 返回初始化后的类型对象指针
}

// 存储已注册的 C++ 函数类型的映射和集合
static std::unordered_map<std::type_index, THPObjectPtr> cpp_function_types_map;
static std::unordered_set<PyTypeObject*> cpp_function_types_set;

// 默认函数类型的结构体
struct DefaultFunctionType {
  DefaultFunctionType() : type() {
    _initFunctionPyTypeObject(type, "CppFunction", nullptr, nullptr);  // 初始化默认函数类型对象
  }

  PyTypeObject type;  // 默认函数类型的 PyTypeObject 对象
};

// 获取默认函数类型对象的指针
PyTypeObject* get_default_type() {
  static DefaultFunctionType default_type;  // 静态变量存储默认函数类型对象
  return &(default_type.type);  // 返回默认函数类型对象的指针
}

// 将 C++ 节点转换为 Python 对象
PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata) {
  if (!cdata) {  // 如果节点为空，则返回 None
    Py_RETURN_NONE;
  }

  if (auto pfw = dynamic_cast<PyNode*>(cdata.get())) {  // 如果节点是 PyNode 类型
    PyObject* obj = pfw->obj;  // 获取 PyNode 对象的 Python 对象
    Py_INCREF(obj);  // 增加 Python 对象的引用计数
    return obj;  // 返回 Python 对象
  }

  if (cdata->pyobj()) {  // 如果节点已有 Python 对象
    Py_INCREF(cdata->pyobj());  // 增加 Python 对象的引用计数
  } else {
    auto& fn = *cdata;  // 获取节点的引用
    auto it = cpp_function_types_map.find(std::type_index(typeid(fn)));  // 查找节点类型在映射中的位置
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyTypeObject* type;  // 声明 Python 类型对象指针
    if (it == cpp_function_types_map.end()) {  // 如果映射中没有找到节点类型
      type = get_default_type();  // 获取默认函数类型对象的指针
    } else {
      type = (PyTypeObject*)it->second.get();  // 获取映射中存储的 Python 类型对象指针
    }

    THPObjectPtr obj(type->tp_alloc(type, 0));  // 分配 Python 类型对象的内存
    if (!obj)  // 如果分配失败则返回空指针
      return nullptr;
    THPCppFunction* f = (THPCppFunction*)obj.get();  // 获取 THPCppFunction 对象指针
    new (&f->cdata) std::shared_ptr<Node>(cdata);  // 在 THPCppFunction 对象中存储节点的共享指针

    // 因为只有弱引用，所以这里不增加引用计数
    cdata->set_pyobj(obj.release());  // 设置节点的 Python 对象
  }

  return cdata->pyobj();  // 返回节点的 Python 对象
}

// 注册 C++ 函数类型及其对应的 Python 类型对象
void registerCppFunction(const std::type_info& type, PyTypeObject* pytype) {
  Py_INCREF((PyObject*)pytype);  // 增加 Python 类型对象的引用计数
  cpp_function_types_map[std::type_index(type)] =
      THPObjectPtr((PyObject*)pytype);  // 将 C++ 函数类型及其 Python 类型对象存储到映射中
  cpp_function_types_set.insert(pytype);  // 将 Python 类型对象插入到集合中
}
// 检查给定对象是否为 THPCppFunction 类型
bool THPCppFunction_Check(PyObject* obj) {
  // 获取对象的类型
  THPObjectPtr type = THPObjectPtr(PyObject_Type(obj));
  // 如果对象类型与默认类型相同，则返回 true
  if ((PyTypeObject*)type.get() == get_default_type()) {
    return true;
  }
  // 如果对象类型不在 cpp_function_types_set 集合中，则返回 false，否则返回 true
  if (cpp_function_types_set.find((PyTypeObject*)type.get()) ==
      cpp_function_types_set.end()) {
    return false;
  } else {
    return true;
  }
}

// 调用 THPFunctionClass 类的 _register_hook 方法
PyObject* callRegisterFn(PyObject* dict, PyObject* hook) {
  // 获取 _register_hook 方法对象
  THPObjectPtr register_fn(
      PyObject_GetAttrString(THPFunctionClass, "_register_hook"));
  // 如果获取失败，则返回空指针
  if (!register_fn) {
    return nullptr;
  }
  // 调用 _register_hook 方法，并传递 dict 和 hook 参数，返回调用结果
  THPObjectPtr res(
      PyObject_CallFunctionObjArgs(register_fn.get(), dict, hook, nullptr));
  // 如果调用失败，则返回空指针
  if (!res) {
    return nullptr;
  }
  // 返回调用结果，释放 res 对象的所有权
  return res.release();
}

// 注册函数的后置钩子
PyObject* registerFunctionHook(Node& fn, PyObject* hook) {
  // 初始化 dict 为 Py_None
  PyObject* dict = Py_None;
  // 遍历 fn 的后置钩子列表
  for (const auto& hook : fn.post_hooks()) {
    // 如果 hook 是 PyFunctionPostHook 类型的指针
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      // 设置 dict 为 pyhook 的 dict 成员
      dict = pyhook->dict;
      break;
    }
  }
  // 调用 callRegisterFn 函数，传递 dict 和 hook 参数
  THPObjectPtr res{callRegisterFn(dict, hook)};
  // 如果调用失败，则返回空指针
  if (!res) {
    return nullptr;
  }
  // 如果 dict 仍为 Py_None，则从 res 结果中获取第一个元素，并将其作为 dict
  if (dict == Py_None) {
    dict = PyTuple_GET_ITEM(res.get(), 0);
    // 将创建的 PyFunctionPostHook 对象添加到 fn 的后置钩子列表中
    fn.add_post_hook(std::make_unique<PyFunctionPostHook>(dict));
  }

  // 获取 res 结果中的第二个元素，并增加其引用计数，然后返回
  PyObject* handle = PyTuple_GET_ITEM(res.get(), 1);
  Py_INCREF(handle);
  return handle;
}

// 注册函数的前置钩子，与 registerFunctionHook 函数基本相同，只是操作 fn 的前置钩子列表
// This is almost a copy of the function above except post -> pre
PyObject* registerFunctionPreHook(Node& fn, PyObject* hook) {
  PyObject* dict = Py_None;
  for (const auto& hook : fn.pre_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
      dict = pyhook->dict;
      break;
    }
  }
  THPObjectPtr res{callRegisterFn(dict, hook)};
  if (!res) {
    return nullptr;
  }
  if (dict == Py_None) {
    dict = PyTuple_GET_ITEM(res.get(), 0);
    fn.add_pre_hook(std::make_unique<PyFunctionPreHook>(dict));
  }

  PyObject* handle = PyTuple_GET_ITEM(res.get(), 1);
  Py_INCREF(handle);
  return handle;
}

} // namespace torch::autograd
```