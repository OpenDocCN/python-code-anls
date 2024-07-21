# `.\pytorch\torch\csrc\autograd\python_hook.cpp`

```
// 包含 Torch 自动求导的 Python 钩子相关头文件
#include <torch/csrc/autograd/python_hook.h>

// 包含 C10 库的范围工具
#include <c10/util/irange.h>
// 包含 pybind11 库，用于 Python 和 C++ 之间的交互
#include <pybind11/pybind11.h>
// 包含 Torch 异常处理相关头文件
#include <torch/csrc/Exceptions.h>
// 包含 Torch Python 解释器相关头文件
#include <torch/csrc/PyInterpreter.h>
// 包含 Torch THP 相关头文件
#include <torch/csrc/THP.h>
// 包含 Torch 自动求导的 Python 变量相关头文件
#include <torch/csrc/autograd/python_variable.h>
// 包含 Torch 动态图编译自动求导相关头文件
#include <torch/csrc/dynamo/compiled_autograd.h>
// 包含 Torch 实用工具中的对象指针相关头文件
#include <torch/csrc/utils/object_ptr.h>
// 包含 Torch 实用工具中的 Python 绑定相关头文件
#include <torch/csrc/utils/pybind.h>
// 包含 Torch 实用工具中的 Python 字符串处理相关头文件
#include <torch/csrc/utils/python_strings.h>

// 包含标准输入输出流库
#include <iostream>
// 包含字符串流库，用于字符串操作
#include <sstream>

// 使用 torch::autograd 命名空间中的 Variable 类
using torch::autograd::Variable;
// 使用 torch::autograd 命名空间中的 variable_list 类型
using torch::autograd::variable_list;

// 声明静态函数 wrap_variables，用于将变量列表包装成 PyObject 对象
static PyObject* wrap_variables(const variable_list& c_variables);
// 声明静态函数 unwrap_variables，用于从 PyObject 对象中解包变量列表
static variable_list unwrap_variables(PyObject* py_variables);
// 声明静态函数 hook_name，用于获取钩子函数的名称
static std::string hook_name(PyObject* hook);
// 声明静态函数 check_result，用于检查钩子函数的返回结果
static void check_result(PyObject* original, PyObject* result, PyObject* hook);
// 声明静态函数 check_single_result，用于检查单个钩子函数的返回结果
static void check_single_result(
    PyObject* original,
    PyObject* result,
    PyObject* hook);

// torch::autograd 命名空间内部的匿名命名空间
namespace {

// 这个函数在四种不同情况下被调用：
//   1) TensorPreHook
//   2) PreHook
//   3) PostHook
//   4) TensorPostAccGradHook
//
// 根据情况，args 和 res 可以持有不同类型的对象：
//
// args:
// TensorPreHook         (Tensor,)
// PreHook               ((Tensor, ...),)                (grad_outputs,)
// PostHook              ((Tensor, ...), (Tensor, ...))  (grad_inputs, grad_outputs)
// TensorPostAccGradHook ((Tensor), ())                  (tensor,)
//
// res:
// TensorPreHook         Tensor
// PreHook               ((Tensor, ...),)                (grad_outputs,)
// PostHook              ((Tensor, ...),)                (grad_inputs,)
// TensorPostAccGradHook None
//
// 如果任何钩子函数返回非空值，则该函数返回 True，否则返回 False。
bool _call_hooks(PyObject* dict, PyObject* args) {
  // 注意: [Extend Hook Lifetime]
  // 在迭代钩子函数期间保持对其的引用。
  // 这是为了处理钩子函数在其中调用 `handle.remove` 且其引用计数降至 `0` 的情况。
  // 在这种情况下，Python 可能会对其进行垃圾回收。
  // 我们保持对一个过时的指针的持有，并且后续调用 `check_single_result`，它尝试获取 `hook` 的名称，会导致段错误。
  // 因此，我们使用 `PyDict_Values` 返回对值的新引用，即我们在迭代钩子函数期间保持对钩子函数的引用。
  // 参考: https://github.com/pytorch/pytorch/issues/58354
  auto hooks = THPObjectPtr{PyDict_Values(dict)};
  bool is_modified = false;
  const auto len = PyList_Size(hooks);
  for (Py_ssize_t idx = 0; idx < len; ++idx) {
    const auto hook = PyList_GetItem(hooks, idx);

    // 调用钩子函数，并获取其返回结果
    THPObjectPtr res(PyObject_CallObject(hook, args));
    if (!res)
      throw python_error();
    if (res == Py_None)
      continue;

    // 检查结果是否与输入参数相同，如果是则继续
    PyObject* args0 = PyTuple_GetItem(args, 0);
    if (res == args0)
      continue;

    // 根据输入参数类型进行结果检查
    if (PyTuple_CheckExact(args0)) {
      check_result(args0, res, hook);
    } else {
      check_single_result(args0, res, hook);
    }
    PyTuple_SetItem(args, 0, res.release());
    # 检查一个变量 is_modified 是否为 true
    is_modified = true;
    # 返回变量 is_modified 的值作为函数的结果
    return is_modified;
}

} // namespace

// 使用给定的字典和索引初始化 PyFunctionTensorPreHook 对象
PyFunctionTensorPreHook::PyFunctionTensorPreHook(
    PyObject* dict,       // 给定的字典对象
    size_t value_idx)     // 索引值
    : dict(dict), value_idx(value_idx) {
  Py_INCREF(dict);        // 增加字典对象的引用计数
}

// 析构函数，释放 PyFunctionTensorPreHook 对象占用的资源
// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionTensorPreHook::~PyFunctionTensorPreHook() {
  // 如果 Python 已经终止，释放包装的 Python 对象
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
    Py_DECREF(dict);    // 减少字典对象的引用计数
  }
}

// 运算符重载函数，用于处理变量列表 values
auto PyFunctionTensorPreHook::operator()(const variable_list& values)
    -> variable_list {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
  THPObjectPtr value(THPVariable_Wrap(values.at(value_idx)));  // 将变量包装为 Python 对象
  if (!value)
    throw python_error();  // 抛出 Python 异常
  THPObjectPtr tup(PyTuple_New(1));  // 创建包含一个元素的元组
  PyTuple_SET_ITEM(tup.get(), 0, value.release());  // 设置元组的第一个元素
  bool is_tup_modified = _call_hooks(dict, tup.get());  // 调用钩子函数处理元组
  variable_list results(values);  // 复制变量列表
  if (is_tup_modified) {
    results[value_idx] = THPVariable_Unpack(PyTuple_GetItem(tup.get(), 0));  // 解包元组中的结果
  }
  return results;  // 返回处理后的变量列表
}

// 使用给定的字典初始化 PyFunctionPreHook 对象
PyFunctionPreHook::PyFunctionPreHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);  // 增加字典对象的引用计数
}

// 析构函数，释放 PyFunctionPreHook 对象占用的资源
// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionPreHook::~PyFunctionPreHook() {
  // 如果 Python 已经终止，释放包装的 Python 对象
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
    Py_DECREF(dict);  // 减少字典对象的引用计数
  }
}

// 运算符重载函数，用于处理梯度输出变量列表
auto PyFunctionPreHook::operator()(const variable_list& grad_outputs_)
    -> variable_list {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
  THPObjectPtr grad_outputs(wrap_variables(grad_outputs_));  // 包装梯度输出变量列表为 Python 对象
  THPObjectPtr tup(PyTuple_New(1));  // 创建包含一个元素的元组
  PyTuple_SET_ITEM(tup.get(), 0, grad_outputs.release());  // 设置元组的第一个元素
  _call_hooks(dict, tup.get());  // 调用钩子函数处理元组
  return unwrap_variables(PyTuple_GetItem(tup.get(), 0));  // 解包处理后的结果
}

// 使用给定的字典初始化 PyFunctionPostHook 对象
PyFunctionPostHook::PyFunctionPostHook(PyObject* dict) : dict(dict) {
  Py_INCREF(dict);  // 增加字典对象的引用计数
}

// 析构函数，释放 PyFunctionPostHook 对象占用的资源
// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionPostHook::~PyFunctionPostHook() {
  // 如果 Python 已经终止，释放包装的 Python 对象
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
    Py_DECREF(dict);  // 减少字典对象的引用计数
  }
}

// 运算符重载函数，用于处理梯度输入和输出变量列表
auto PyFunctionPostHook::operator()(
    const variable_list& _outputs, /* grad_inputs */
    const variable_list& _inputs /* grad_outputs */) -> variable_list {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
  THPObjectPtr grad_inputs(wrap_variables(_outputs));  // 包装梯度输入变量列表为 Python 对象
  THPObjectPtr grad_outputs(wrap_variables(_inputs));  // 包装梯度输出变量列表为 Python 对象
  THPObjectPtr tup(PyTuple_New(2));  // 创建包含两个元素的元组
  PyTuple_SET_ITEM(tup.get(), 0, grad_inputs.release());  // 设置元组的第一个元素
  PyTuple_SET_ITEM(tup.get(), 1, grad_outputs.release());  // 设置元组的第二个元素
  _call_hooks(dict, tup.get());  // 调用钩子函数处理元组
  return unwrap_variables(PyTuple_GetItem(tup.get(), 0));  // 解包处理后的结果
}

// 将 PyFunctionTensorPreHook 对象编译为 CompiledNodeArgs 对象的方法
void PyFunctionTensorPreHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    Py_INCREF(value);  // 增加值对象的引用计数
    args.add_tensor_pre_hook(
        c10::SafePyObject(value, getPyInterpreter()),  // 安全地获取 Python 对象
        static_cast<int>(value_idx));  // 转换为整数索引并添加到 CompiledNodeArgs 对象中
  }
}
void PyFunctionPreHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  // 遍历字典中的每一对键值对
  while (PyDict_Next(dict, &pos, &key, &value)) {
    // 增加值的引用计数，以确保在使用时不会被释放
    Py_INCREF(value);
    // 将值包装成安全的 PyObject 并添加到预钩子中
    args.add_pre_hook(c10::SafePyObject(value, getPyInterpreter()));
  }
}

void PyFunctionPostHook::compiled_args(CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  // 遍历字典中的每一对键值对
  while (PyDict_Next(dict, &pos, &key, &value)) {
    // 增加值的引用计数，以确保在使用时不会被释放
    Py_INCREF(value);
    // 将值包装成安全的 PyObject 并添加到后钩子中
    args.add_post_hook(c10::SafePyObject(value, getPyInterpreter()));
  }
}

PyFunctionTensorPostAccGradHooks::PyFunctionTensorPostAccGradHooks(
    PyObject* dict)
    : dict(dict) {
  // 增加字典对象的引用计数，确保对象在使用期间不会被释放
  Py_INCREF(dict);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyFunctionTensorPostAccGradHooks::~PyFunctionTensorPostAccGradHooks() {
  // 如果 Python 解释器仍然活跃，释放字典对象的引用
  if (Py_IsInitialized()) {
    // 获取全局解释器锁，确保线程安全地操作 Python 对象
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict);
  }
}

auto PyFunctionTensorPostAccGradHooks::operator()(const Variable& tensor)
    -> void {
  pybind11::gil_scoped_acquire gil;
  // 创建一个包含单个变量的 Python 元组
  THPObjectPtr tup(PyTuple_New(1));
  PyTuple_SET_ITEM(tup.get(), 0, THPVariable_Wrap(tensor));
  // 调用钩子函数并检查返回值是否为 None
  bool returned_none = !_call_hooks(dict, tup.get());
  // 断言钩子函数返回值不为 None
  TORCH_CHECK(
      returned_none, "Tensor post accumulate grad hooks should return None.");
}

void PyFunctionTensorPostAccGradHooks::compiled_args(
    torch::dynamo::autograd::CompiledNodeArgs& args) {
  PyObject *key = nullptr, *value = nullptr;
  Py_ssize_t pos = 0;
  // 遍历字典中的每一对键值对
  while (PyDict_Next(dict, &pos, &key, &value)) {
    // 增加值的引用计数，以确保在使用时不会被释放
    Py_INCREF(value);
    // 将值包装成安全的 PyObject 并添加到后累积梯度钩子中
    c10::SafePyObject hook_obj(value, getPyInterpreter());
    args.add_post_acc_grad_hook(std::move(hook_obj));
  }
}

void PyFunctionTensorPostAccGradHooks::apply_with_saved(
    Variable& tensor,
    torch::dynamo::autograd::SwapSavedVariables& saved) {
  // 遍历保存的当前节点调用中的后累积梯度钩子
  for (const auto hook : saved.get_curr_node_call().post_acc_grad_hooks) {
    // 将变量 tensor 封装为 Python 对象
    THPObjectPtr py_var(THPVariable_Wrap(tensor));
    // 调用 Python 对象的方法来执行后累积梯度钩子
    PyObject_CallMethod(
        saved.get_py_compiler(),
        "post_acc_grad_hook",
        "Oi",
        py_var.get(),
        hook);
  }
}

} // namespace autograd
} // namespace torch

static PyObject* wrap_variables(const variable_list& c_variables) {
  // 创建一个包含指定数量变量的 Python 元组
  size_t num_vars = c_variables.size();
  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(num_vars)));
  if (!tuple)
    throw python_error();
  // 遍历 C++ 变量列表，并将每个变量包装成 Python 对象放入元组中
  for (const auto i : c10::irange(num_vars)) {
    THPObjectPtr var(THPVariable_Wrap(c_variables[i]));
    if (!var)
      throw python_error();
    PyTuple_SET_ITEM(tuple.get(), i, var.release());
  }
  // 返回创建的 Python 元组对象
  return tuple.release();
}

static variable_list unwrap_variables(PyObject* py_variables) {
  // 从 Python 元组中解析变量列表
  variable_list results(PyTuple_GET_SIZE(py_variables));
  for (const auto i : c10::irange(results.size())) {
    PyObject* item = PyTuple_GET_ITEM(py_variables, i);
    // 跳过空对象
    if (item == Py_None) {
      continue;
    } else if (THPVariable_Check(item)) {
      // 解包 THPVariable 对象并存入结果列表
      results[i] = THPVariable_Unpack(item);
      // PyErr_Occurred() 检查错误
      if (!results[i]) {
        throw python_error();
      }
    }
  }
    } else {
      // 如果程序执行到这里，通常表示出现了意料之外的情况
      // 构造一个字符串流对象
      std::stringstream ss;
      // 将错误信息添加到字符串流中，包括预期变量类型和实际获取的类型
      ss << "expected variable but got " << Py_TYPE(item)->tp_name;
      // 抛出运行时异常，并将字符串流中的内容作为错误信息
      throw std::runtime_error(ss.str());
    }
  }
  // 返回处理结果
  return results;
static void check_result(PyObject* prev, PyObject* result, PyObject* hook) {
    // 检查返回结果是否为元组类型
    if (!PyTuple_Check(result)) {
        // 抛出类型错误异常，指示钩子返回了非元组类型
        PyErr_Format(
            PyExc_TypeError,
            "expected tuple, but hook returned '%s'",
            THPUtils_typename(result));
        throw python_error();
    }

    // 获取前一个元组和当前结果元组的大小
    auto prev_size = PyTuple_GET_SIZE(prev);
    auto result_size = PyTuple_GET_SIZE(result);

    // 检查前一个元组和结果元组的大小是否相同
    if (prev_size != result_size) {
        // 构建错误信息，指示钩子返回的值数量不正确
        std::stringstream ss;
        auto name = hook_name(hook);
        ss << "hook '" << name << "' has returned an incorrect number ";
        ss << "of values (got " << result_size << ", but expected ";
        ss << prev_size << ")";
        throw std::runtime_error(ss.str());
    }

    // 逐个检查前一个元组和结果元组中的每个元素
    for (const auto i : c10::irange(prev_size)) {
        check_single_result(
            PyTuple_GET_ITEM(prev, i), PyTuple_GET_ITEM(result, i), hook);
    }
}

static void check_single_result(
    PyObject* _original,
    PyObject* _result,
    PyObject* hook) {
    // 如果结果为None，则直接返回，无需进一步处理
    if (_result == Py_None)
        return;

    // 如果原始值为None，则抛出运行时错误，不允许用非None值替换None梯度
    if (_original == Py_None) {
        throw std::runtime_error(
            "can't replace a None gradient with a non-None value");
    }

    // 检查结果是否为THPVariableClass类型
    if (!PyObject_IsInstance(_result, THPVariableClass)) {
        // 抛出类型错误异常，指示钩子返回了非Variable类型
        PyErr_Format(
            PyExc_TypeError,
            "expected Variable, but hook returned '%s'",
            THPUtils_typename(_result));
        throw python_error();
    }

    // 解包原始值和结果值，调用torch::autograd::check_variable_result检查
    const auto& original = THPVariable_Unpack(_original);
    const auto& result = THPVariable_Unpack(_result);

    // 检查变量结果
    torch::autograd::check_variable_result(original, result, hook_name(hook));
}

static std::string hook_name(PyObject* hook) {
    // 如果hook对象有__name__属性，则获取其值作为钩子的名称
    if (PyObject_HasAttrString(hook, "__name__")) {
        THPObjectPtr name(PyObject_GetAttrString(hook, "__name__"));
        if (!name)
            throw python_error();

        // 如果名称有效且为字符串类型，则返回其解包后的值
        if (name && THPUtils_checkString(name.get())) {
            return THPUtils_unpackString(name.get());
        }
    }
    // 如果未找到有效名称，则返回"<unknown>"
    return "<unknown>";
}
```