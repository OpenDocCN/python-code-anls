# `.\pytorch\torch\csrc\autograd\python_function.cpp`

```py
// 引入Torch的Python函数头文件
#include <torch/csrc/autograd/python_function.h>

// 引入ATen库
#include <ATen/ATen.h>
// 引入ATen的序列号
#include <ATen/SequenceNumber.h>
// 引入C10的工具函数：irange
#include <c10/util/irange.h>
// 引入pybind11库
#include <pybind11/pybind11.h>
// 引入structmember头文件
#include <structmember.h>
// 引入Torch的Python解释器头文件
#include <torch/csrc/PyInterpreter.h>
// 引入Torch的Python头文件
#include <torch/csrc/python_headers.h>
// 引入Torch的pybind工具函数
#include <torch/csrc/utils/pybind.h>

// 引入ATen的TLS函数
#include <ATen/FuncTorchTLS.h>
// 引入Functorch的动态层
#include <ATen/functorch/DynamicLayer.h>
// 引入Torch的动态类型
#include <torch/csrc/DynamicTypes.h>
// 引入Torch的异常处理
#include <torch/csrc/Exceptions.h>
// 引入Torch的THP
#include <torch/csrc/THP.h>
// 引入自动求导相关的函数：accumulate_grad
#include <torch/csrc/autograd/functions/accumulate_grad.h>
// 引入自动求导基本操作相关的函数
#include <torch/csrc/autograd/functions/basic_ops.h>
// 引入自动求导工具函数
#include <torch/csrc/autograd/functions/utils.h>
// 引入自动求导梯度模式
#include <torch/csrc/autograd/grad_mode.h>
// 引入自动求导图任务
#include <torch/csrc/autograd/graph_task.h>
// 引入自动求导异常模式
#include <torch/csrc/autograd/python_anomaly_mode.h>
// 引入自动求导C++函数
#include <torch/csrc/autograd/python_cpp_function.h>
// 引入自动求导钩子函数
#include <torch/csrc/autograd/python_hook.h>
// 引入自动求导保存变量
#include <torch/csrc/autograd/saved_variable.h>
// 引入自动求导包装输出工具函数
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 引入Torch的Dynamo编译自动求导
#include <torch/csrc/dynamo/compiled_autograd.h>
// 引入Torch的前端跟踪器
#include <torch/csrc/jit/frontend/tracer.h>
// 引入Torch的IR（Intermediate Representation）
#include <torch/csrc/jit/ir/ir.h>
// 引入Torch的pybind工具函数
#include <torch/csrc/jit/python/pybind_utils.h>
// 引入Torch的Python跟踪器
#include <torch/csrc/jit/python/python_tracer.h>
// 引入Torch的分析器API
#include <torch/csrc/profiler/api.h>
// 引入Torch的Python字符串工具函数
#include <torch/csrc/utils/python_strings.h>
// 引入Torch的张量数据类型
#include <torch/csrc/utils/tensor_dtypes.h>

// 引入C++标准库的功能函数
#include <functional>
// 引入内存管理的头文件
#include <memory>
// 引入标准异常处理的头文件
#include <stdexcept>
// 引入C++标准字符串处理的头文件
#include <string>
// 引入C++无序映射的头文件
#include <unordered_map>
// 引入C++无序集合的头文件
#include <unordered_set>
// 引入C++实用工具的头文件
#include <utility>
// 引入C++标准向量的头文件
#include <vector>

// 使用torch命名空间
using namespace torch;
// 使用torch的自动求导命名空间
using namespace torch::autograd;
// 使用ATen张量的别名
using at::Tensor;

// 定义THPFunctionClass和THPGradientEdgeClass为nullptr
PyObject* THPFunctionClass = nullptr;
PyObject* THPGradientEdgeClass = nullptr;

// 定义THPFunction_assert宏，用于检查条件并抛出Python错误
#define THPFunction_assert(condition, ...) \
  if (!(condition)) {                      \
    THPUtils_setError(__VA_ARGS__);        \
    throw python_error();                  \
  }

// 匿名命名空间，用于此文件中使用的辅助函数
namespace {

// TODO: 我们不应该需要调用这个函数，因为引擎已经可以为我们持久化错误。
// 然而，DistEngine仍然似乎需要它。
//
// python test/distributed/rpc/test_tensorpipe_agent.py -k
// test_backward_autograd_engine_error
//
// 参见Note [ Persisting PyErr state across autograd engine threads ]
// 抛出Python错误的静态函数
void throw_python_error() {
  python_error err;
  err.persist();
  throw std::move(err);
}

// 解包保存的变量的静态函数
static PyObject* unpack_saved_variables(
    THPFunction* self,
    const std::function<PyObject*(const Variable&)>& unpack_fn) {
  HANDLE_TH_ERRORS
  // 检查self是否已释放缓冲区
  TORCH_CHECK(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  // 获取保存的变量的引用
  auto& saved_variables = self->saved_variables;
  // 如果保存的变量为空，则返回空元组
  if (saved_variables.empty())
    return PyTuple_New(0);

  // 获取保存变量的数量
  auto num_saved = saved_variables.size();
  // 创建一个PyTuple对象，大小为num_saved
  THPObjectPtr saved(PyTuple_New(static_cast<Py_ssize_t>(num_saved)));
  // 如果创建失败，则返回nullptr
  if (!saved)
    return nullptr;
    // 返回空指针
    return nullptr;
  auto saved_for = self->cdata.lock();
  // 这其实是一个真正的断言，因为我们已经在函数开始处测试了
  // self->has_freed_buffers 的情况：当 PyNode 死亡时缓冲区被释放；
  // 如果缓冲区没有被释放，PyNode 必须是存活的。
  // （注意，即使 PyNode 存活，缓冲区也可能被释放，但这对本行代码无影响--
  // 并且在任何情况下 saved_for 都将是非空的。）
  TORCH_INTERNAL_ASSERT(saved_for);
  for (const auto i : c10::irange(num_saved)) {
    auto unpacked_var = saved_variables[i].unpack(saved_for);
    THPObjectPtr value;
    if (!unpacked_var.defined()) {
      // 增加 Python 对象 Py_None 的引用计数
      Py_INCREF(Py_None);
      value = Py_None;
    } else {
      // 对 unpacked_var 应用解包函数 unpack_fn，得到 Python 对象 value
      value = unpack_fn(unpacked_var);
    }
    // 将 value 的所有权释放给 PyTuple 对象 saved 的第 i 个元素
    PyTuple_SET_ITEM(saved.get(), i, value.release());
  }
  // 释放 saved 对象的所有权，并返回其指针
  return saved.release();
  END_HANDLE_TH_ERRORS
} // 结束前一个命名空间的定义

PyObject* to_py_size(const std::vector<c10::SymInt>& size) {
  // 将 std::vector<c10::SymInt> 转换为 c10::SymIntArrayRef
  c10::SymIntArrayRef sym_sizes(size);

  // 分配一个 THPSizeType 类型的 Python 对象，大小为 sym_sizes 的大小
  auto ret = THPObjectPtr(THPSizeType.tp_alloc(
      &THPSizeType, static_cast<Py_ssize_t>(sym_sizes.size())));
  if (!ret)
    throw python_error();

  // 遍历 sym_sizes，将每个 c10::SymInt 转换为 Python 对象并放入元组 ret 中
  for (auto i : c10::irange(sym_sizes.size())) {
    auto symint = sym_sizes[i];
    // 如果 symint 可能转换为整数，则打包成 int64_t 类型放入元组中
    if (auto maybe_int = symint.maybe_as_int(); maybe_int.has_value()) {
      PyTuple_SET_ITEM(ret.get(), i, THPUtils_packInt64(*maybe_int));
    } else {
      // 否则将 symint 转换为 Python 对象放入元组中
      auto py_symint = py::cast(symint).release().ptr();
      PyTuple_SET_ITEM(ret.get(), i, py_symint);
    }
  }
  // 返回元组 ret 的所有权，释放 ret
  return ret.release();
}

} // namespace 结束前一个命名空间的定义

namespace torch::autograd {

// 注意：此函数假设仅在反向传播调用，由 engine.cpp 使用。
// 负责将 C++ 的 Node::apply 调用转发到 Python 方法 "apply"。
auto PyNode::apply(variable_list&& inputs) -> variable_list {
  pybind11::gil_scoped_acquire gil;
  at::OptionalDeviceGuard _device_guard;
  THPFunction* py_fn = (THPFunction*)obj;

  // 将 C++ 的 variable_list 转换为 Python 的参数元组
  THPObjectPtr pyInputs(to_py_args(inputs, &_device_guard));

  // 获取 Python 对象的 "apply" 属性，并调用其对应的 Python 方法
  THPObjectPtr apply_fn(PyObject_GetAttrString(obj, "apply"));
  if (!apply_fn)
    throw_python_error();
  THPObjectPtr r(PyObject_CallObject(apply_fn, pyInputs.get()));
  if (!r)
    throw_python_error();
  ensure_tuple(r);

  auto& is_variable_input = py_fn->is_variable_input;
  auto num_outputs = PyTuple_GET_SIZE(r.get());
  auto num_forward_inputs = static_cast<Py_ssize_t>(is_variable_input.size());

  // 如果返回的输出数大于期望的输入数，则检查是否全部为 None，并进行必要的修剪
  if (num_outputs > num_forward_inputs) {
    bool all_none = true;
    for (const auto i : c10::irange(num_forward_inputs, num_outputs)) {
      all_none &= PyTuple_GET_ITEM(r.get(), i) == Py_None;
    }
    if (all_none) {
      num_outputs = num_forward_inputs;
      // 在这种情况下，截取结果元组，确保结果元组长度为 num_forward_inputs
      r = PyTuple_GetSlice(r.get(), 0, num_forward_inputs);
      if (!r)
        throw_python_error();
    }
  }

  // 现在应该确保梯度的数量与前向输入的数量相匹配
  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got ";
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  // 将 Python 结果元组重新转换为 C++ 的 variable_list
  return to_variable_list(r.get(), is_variable_input);
}

auto PyNode::defer_to_dynamo(
    variable_list&& inputs,
    std::optional<PyObject*> compiler) -> variable_list {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 保存当前设备，并设置设备为可选设备
  at::OptionalDeviceGuard _device_guard;
  // 将传入的 PyObject 对象强制转换为 THPFunction 指针
  THPFunction* py_fn = (THPFunction*)obj;

  // 将 C++ 的 inputs 转换为 Python 的 arguments 元组
  THPObjectPtr pyInputs(to_py_args(inputs, &_device_guard));

  // 获取是否为可变输入的标志和输入信息
  const auto& is_variable_input = py_fn->is_variable_input;
  const auto& input_infos = py_fn->input_info;
  // input_info 只包含可变输入的信息，应为 is_variable_input 的子集
  TORCH_INTERNAL_ASSERT(is_variable_input.size() >= input_infos.size());

  // 反向传播返回的梯度应与前向传播的输入数目匹配，以及它们的元数据，因此传递 fwdInputs
  THPObjectPtr fwdInputMetadatas(
      PyTuple_New(static_cast<Py_ssize_t>(is_variable_input.size())));
  if (!fwdInputMetadatas)
    throw python_error();

  int offset = 0;
  for (const auto i : c10::irange(is_variable_input.size())) {
    if (!is_variable_input[i]) {
      // 如果第 i 个输入不是变量，跳过该索引
      PyTuple_SET_ITEM(fwdInputMetadatas.get(), i, Py_None);
      offset++;
      continue;
    }

    const auto& input_info = input_infos[i - offset];

    // 创建代表元数据的 Python 元组，包含 4 个元素：(布局, 设备, 数据类型, 尺寸)
    PyObject* device(THPDevice_New(input_info.device));
    if (!device)
      throw_python_error();
    PyObject* fwdInputMetadata = PyTuple_Pack(
        4,
        autograd::utils::wrap(input_info.layout),
        device,
        autograd::utils::wrap(input_info.scalar_type),
        to_py_size(input_info.size));
    if (!fwdInputMetadata)
      throw python_error();

    PyTuple_SET_ITEM(fwdInputMetadatas.get(), i, fwdInputMetadata);
  }

  // 解包保存的变量，返回保存的 Tensor 的元组
  THPObjectPtr saved_tensors(unpack_saved_variables(
      py_fn, [](const Variable& var) { return THPVariable_Wrap(var); }));
  // 断言 _backward_idx 已设置，由 compiled_args 调用，apply_with_saved 之前已调用
  TORCH_INTERNAL_ASSERT(
      _backward_idx.has_value(),
      "indices should already be set by compiled_args, called before apply_with_saved");
  TORCH_INTERNAL_ASSERT(!_backward_state_idx.has_value());

  // 调用 Python 的 proxy_call_backward 方法执行反向传播
  THPObjectPtr r(PyObject_CallMethod(
      *compiler,
      "proxy_call_backward",
      "OOOi",
      pyInputs.get(),
      fwdInputMetadatas.get(),
      saved_tensors.get(),
      *_backward_idx));

  if (!r)
    throw_python_error();
  // 确保返回的结果是元组
  ensure_tuple(r);

  // 将 Python 的结果元组转换为 C++ 的 variable_list
  return to_variable_list(r.get(), is_variable_input);
auto PyNode::is_traceable() -> bool {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 获取对象的 "_forward_cls" 属性
  THPObjectPtr forward_class{PyObject_GetAttrString(obj, "_forward_cls")};
  // 如果获取失败，则抛出 Python 异常
  if (!forward_class)
    throw_python_error();
  // 获取 "is_traceable" 属性
  THPObjectPtr traceable_py_bool{
      PyObject_GetAttrString(forward_class, "is_traceable")};
  // 如果获取失败，则抛出 Python 异常
  if (!traceable_py_bool)
    throw_python_error();
  // 判断 "is_traceable" 是否为 True，返回结果
  return traceable_py_bool == Py_True;
}

auto PyNode::release_variables() -> void {
  // 此函数作为 Node 析构函数的一部分被调用！
  // 由于该对象可能被 C++ 保持存活，因此 Python 解释器可能已经死亡。
  // 在这种情况下，我们只能泄漏保存的对象。
  if (Py_IsInitialized()) {
    // 获取全局解释器锁，确保线程安全
    pybind11::gil_scoped_acquire gil;
    // 将 obj 转换为 THPFunction 指针
    auto f = (THPFunction*)obj;
    // 清空保存的变量
    f->saved_variables.clear();
    // 标记已释放缓冲区
    f->has_freed_buffers = 1;
  }
}

auto PyNode::name() const -> std::string {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 将 obj 转换为 THPFunction 指针
  auto f = (THPFunction*)obj;
  // 获取函数名
  auto name = std::string(Py_TYPE(f)->tp_name);
  // 返回函数名
  return name;
}

auto PyNode::compiled_autograd_should_lift() const -> bool {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 定义静态属性名称 "_compiled_autograd_should_lift"
  static PyObject* attr_name =
      PyUnicode_InternFromString("_compiled_autograd_should_lift");
  // 获取属性值
  THPObjectPtr should_lift(PyObject_GetAttr(obj, attr_name));
  // 返回属性值是否为 True
  return PyObject_IsTrue(should_lift.get()) == 1;
}

void PyNode::compiled_args(CompiledNodeArgs& args) {
  // 定义静态方法名称 "_compiled_autograd_key"
  static PyObject* method_name =
      PyUnicode_InternFromString("_compiled_autograd_key");
  // 调用对象的方法 "_compiled_autograd_key"
  THPObjectPtr pykey(PyObject_CallMethodNoArgs(obj, method_name));
  // 如果调用失败，则抛出 Python 异常
  if (!pykey)
    throw_python_error();
  // 检查返回值类型是否为元组
  TORCH_CHECK(
      PyTuple_CheckExact(pykey.get()),
      "_compiled_autograd_key should return tuple of ints");
  // 获取元组大小
  auto size = PyTuple_GET_SIZE(pykey.get());
  // 断言元组大小大于 0
  TORCH_INTERNAL_ASSERT(size > 0);
  // 获取第一个值作为唯一的 ID，由 AUTOGRAD_FUNCTION_COUNTER 管理
  auto key = PyLong_AsSsize_t(PyTuple_GET_ITEM(pykey.get(), 0));
  // 如果 key 小于 0，则抛出 Python 异常
  if (C10_UNLIKELY(key < 0)) {
    TORCH_CHECK(PyErr_Occurred(), "key must be positive");
    throw_python_error();
  }
  // 收集大小信息
  args.collect_size(static_cast<size_t>(key));
  args.collect_size(static_cast<size_t>(size));

  // 将 obj 转换为 THPFunction 指针
  auto f = (THPFunction*)obj;
  // 清空编译自动微分符号整数的缓存
  f->compiled_autograd_symints.clear();
  f->compiled_autograd_symints.reserve(size - 1);
  // 遍历元组中的值
  for (const auto i : c10::irange(1, size)) {
    // 将元组中的值转换为整数
    auto val = PyLong_AsSsize_t(PyTuple_GET_ITEM(pykey.get(), i));
    // 如果转换失败，则抛出 Python 异常
    if (C10_UNLIKELY(val == -1 && PyErr_Occurred()))
      throw_python_error();
    // 添加到编译自动微分符号整数的缓存中
    f->compiled_autograd_symints.push_back(static_cast<int64_t>(val));
  }
}
  // 将给定的值添加到 f 对象的 compiled_autograd_symints 后面
  f->compiled_autograd_symints.emplace_back(val);
}

// 设置所有 AotAutograd symints 为动态类型
auto prior =
    args.set_default_dyn_type(torch::dynamo::autograd::SizeInput::DYNAMIC);
// 收集 f 对象中 compiled_autograd_symints 的内容到 args 中
args.collect(f->compiled_autograd_symints);
// 恢复先前的动态类型设置
args.set_default_dyn_type(prior);

// 收集 f 对象中 saved_variables 的内容到 args 中
args.collect(f->saved_variables);
// 收集 f 对象中 materialize_grads 的内容到 args 中
args.collect(f->materialize_grads);
// 收集 f 对象中 is_variable_input 的内容到 args 中
args.collect(f->is_variable_input);
// 收集 f 对象中 needs_input_grad 的内容到 args 中
args.collect(f->needs_input_grad);
// 收集 f 对象中 materialize_non_diff_grads 的内容到 args 中
args.collect(f->materialize_non_diff_grads);
// 收集 f 对象中 output_info 的内容到 args 中
args.collect(f->output_info);
// 收集 f 对象中 input_info 的内容到 args 中
args.collect(f->input_info);

// 如果需要进行编译自动求导提升操作
if (compiled_autograd_should_lift()) {
  // 增加 obj 的 Python 引用计数
  Py_INCREF(obj);
  // 将 obj 添加为反向传播的一个参数，并返回其索引
  _backward_idx =
      args.add_backward(c10::SafePyObject(obj, getPyInterpreter()));
}

// 获取 f 对象的 compiled_autograd_backward_state
PyObject* bw_state = f->compiled_autograd_backward_state;
// 如果 compiled_autograd_backward_state 不为空
if (args.cond(bw_state != nullptr)) {
  // 增加 bw_state 的 Python 引用计数
  Py_INCREF(bw_state);
  // 将 bw_state 添加为反向传播状态的一个参数，并返回其索引
  _backward_state_idx = args.add_backward_state(
      c10::SafePyObject(bw_state, getPyInterpreter()));
}
}

variable_list PyNode::apply_with_saved(
    const variable_list& inputs,
    SwapSavedVariables& saved) {
  auto f = (THPFunction*)obj;  // 将对象转换为 THPFunction 指针
  TORCH_INTERNAL_ASSERT(!f->compiled_autograd_tracing);  // 断言：确保编译自动求导跟踪未启用
  saved.before(f->compiled_autograd_symints);  // 在 saved 中保存编译自动求导符号整数
  saved.before(f->saved_variables);  // 在 saved 中保存函数的保存变量
  saved.before(f->needs_input_grad);  // 在 saved 中保存函数的输入梯度需求
  saved.before(f->materialize_non_diff_grads);  // 在 saved 中保存函数的非可微梯度材料化
  saved.before(f->output_info);  // 在 saved 中保存函数的输出信息
  saved.before(f->input_info);  // 在 saved 中保存函数的输入信息
  f->compiled_autograd_tracing = true;  // 启用编译自动求导跟踪
  variable_list result;  // 声明变量列表 result
  if (!compiled_autograd_should_lift()) {  // 如果不应该提升编译自动求导
    if (_backward_state_idx.has_value()) {  // 如果存在反向状态索引值
      PyObject* r = PyObject_CallMethod(
          saved.get_py_compiler(),  // 调用 saved 的 Python 编译器对象
          "bind_backward_state",  // 调用方法 bind_backward_state
          "i",
          *_backward_state_idx);  // 使用反向状态索引值作为参数
      if (r == nullptr) {  // 如果调用返回空指针
        throw python_error();  // 抛出 Python 错误异常
      }
      THPObjectPtr prior(f->compiled_autograd_backward_state);  // 创建 THPObjectPtr 对象，保存先前的编译自动求导反向状态
      f->compiled_autograd_backward_state = r;  // 更新编译自动求导反向状态为新值
      result = apply(variable_list(inputs));  // 应用函数并保存结果
      Py_CLEAR(f->compiled_autograd_backward_state);  // 清除编译自动求导反向状态
      f->compiled_autograd_backward_state = prior.release();  // 恢复先前的编译自动求导反向状态
    } else {
      result = apply(variable_list(inputs));  // 应用函数并保存结果
    }
  } else {
    result = defer_to_dynamo(variable_list(inputs), saved.get_py_compiler());  // 延迟到 Dynamo，保存结果
  }
  f->compiled_autograd_tracing = false;  // 关闭编译自动求导跟踪
  saved.after(f->compiled_autograd_symints);  // 在 saved 中恢复编译自动求导符号整数
  saved.after(f->saved_variables);  // 在 saved 中恢复函数的保存变量
  saved.after(f->needs_input_grad);  // 在 saved 中恢复函数的输入梯度需求
  saved.after(f->materialize_non_diff_grads);  // 在 saved 中恢复函数的非可微梯度材料化
  saved.after(f->output_info);  // 在 saved 中恢复函数的输出信息
  saved.after(f->input_info);  // 在 saved 中恢复函数的输入信息
  return result;  // 返回结果列表
}

PyObject* PyNode::to_py_args(
    const variable_list& inputs,
    at::OptionalDeviceGuard* device_guard) {
  THPFunction* py_fn = (THPFunction*)obj;  // 将对象转换为 THPFunction 指针

  auto zeros_without_gil = [](const VariableInfo& variable,
                              at::OptionalDeviceGuard& dg) {
    pybind11::gil_scoped_release gil;  // 在 PyBind11 中释放全局解释器锁
    return variable.zeros(dg);  // 返回变量的零值
  };

  auto num_inputs = inputs.size();  // 获取输入变量列表的大小
  PyObject* pyInputs = PyTuple_New(static_cast<Py_ssize_t>(num_inputs));  // 创建一个新的元组对象 pyInputs
  if (!pyInputs)  // 如果创建元组失败
    throw_python_error();  // 抛出 Python 错误异常
  auto& output_info = py_fn->output_info;  // 获取函数的输出信息引用
  for (const auto i : c10::irange(num_inputs)) {  // 循环遍历输入变量列表
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject* input;  // 声明输入对象
    if (inputs[i].defined() || !py_fn->materialize_grads ||
        (input_metadata(i).was_default_constructed() &&
         !py_fn->materialize_non_diff_grads)) {
      input = THPVariable_Wrap(inputs[i]);  // 封装输入变量为 THPVariable 对象
    } else {
      input =
          THPVariable_Wrap(zeros_without_gil(output_info[i], *device_guard));  // 使用无全局解释器锁的零值封装输入变量
    }
    if (!input)  // 如果封装输入失败
      throw_python_error();  // 抛出 Python 错误异常
    PyTuple_SET_ITEM(pyInputs, i, input);  // 设置元组 pyInputs 的第 i 个元素为 input
  }

  return pyInputs;  // 返回 Python 元组 pyInputs
}

variable_list PyNode::to_variable_list(
    const PyObject* outputs,
    const std::vector<bool>& is_variable_input) {
  auto num_outputs = PyTuple_GET_SIZE(outputs);  // 获取元组 outputs 的大小
  variable_list results;  // 声明结果变量列表
  results.reserve(num_outputs);  // 预留结果变量列表的空间
  for (int i = 0; i != num_outputs; ++i) {  // 循环遍历输出元组
    PyObject* output = PyTuple_GET_ITEM(outputs, i);  // 获取输出元组中第 i 个元素
    bool was_variable = is_variable_input[i];  // 获取是否为变量输入的标志
    // 如果之前不是 Variable 类型
    if (!was_variable) {
        // 如果输出不是 Python 中的 None 对象
        if (output != Py_None) {
            // 构建错误消息，指示函数返回了与 None 不同的梯度，但相应的前向输入不是 Variable 类型
            std::string msg("function ");
            msg += name() + " returned a gradient different than None at position ";
            msg += std::to_string(i + 1) +
                ", but the corresponding forward input was not a Variable";
            // 抛出运行时错误，包含错误消息
            throw std::runtime_error(msg);
        }
        // 继续下一个循环迭代
        continue;
    }
    // 如果输出是 Python 中的 None 对象
    if (output == Py_None) {
        // 在结果列表中添加一个默认构造的对象
        results.emplace_back();
    } else {
        // 如果输出不是 THPVariable 类型
        if (!THPVariable_Check(output)) {
            // 构建错误消息，表明期望是 Variable 或 None，但得到了其他类型
            std::string msg("expected Variable or None (got ");
            msg += THPUtils_typename(output);
            msg += ")";
            // 抛出运行时错误，包含错误消息
            throw std::runtime_error(msg);
        }
        // 将 THPVariable 对象解包并添加到结果列表中
        results.emplace_back(THPVariable_Unpack(output));
    }
}

// 返回结果列表
return results;
}

} // namespace torch::autograd

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction* self, visitproc visit, void* arg) {
  // NB: We should not traverse PyObbject stored on PyNode, since we only hold
  // as weak reference to the PyNode.
  Py_VISIT(self->to_save);  // Visit the 'to_save' attribute for Python GC
  Py_VISIT(self->non_differentiable);  // Visit the 'non_differentiable' attribute for Python GC
  Py_VISIT(self->dirty_tensors);  // Visit the 'dirty_tensors' attribute for Python GC
  Py_VISIT(self->compiled_autograd_backward_state);  // Visit the 'compiled_autograd_backward_state' attribute for Python GC
  Py_VISIT(self->saved_for_forward);  // Visit the 'saved_for_forward' attribute for Python GC
  return 0;  // Return success
}

static int THPFunction_clear(THPFunction* self) {
  // Note that the cdata might not be expired yet in the case where this
  // object is part of a cycle and the GC happens to tp_clear this PyObject
  // before the other ones that trigger the de-allocation of the cdata

  Py_CLEAR(self->needs_input_grad);  // Clear the 'needs_input_grad' attribute

  Py_CLEAR(self->to_save);  // Clear the 'to_save' attribute
  Py_CLEAR(self->non_differentiable);  // Clear the 'non_differentiable' attribute
  Py_CLEAR(self->dirty_tensors);  // Clear the 'dirty_tensors' attribute
  Py_CLEAR(self->compiled_autograd_backward_state);  // Clear the 'compiled_autograd_backward_state' attribute
  Py_CLEAR(self->saved_for_forward);  // Clear the 'saved_for_forward' attribute

  self->output_info.clear();  // Clear the 'output_info' vector
  self->input_info.clear();  // Clear the 'input_info' vector
  self->saved_variables.clear();  // Clear the 'saved_variables' vector
  self->is_variable_input.clear();  // Clear the 'is_variable_input' vector

  return 0;  // Return success
}

static void THPFunction_dealloc(THPFunction* self) {
  // Why is this guaranteed to be true?  Suppose that self->cdata is non-null
  // (otherwise the condition is trivially true).  Then there is a PyNode
  // which contains an owning reference to this object.  But we are only
  // allowed to clear if all owning references are gone!  Contradiction.
  //
  // However, note that THPFunction_clear is typically called in the shared_ptr
  // destructor of PyNode; in that case, per
  // https://cplusplus.github.io/LWG/lwg-active.html#2751 it's not currently
  // specified in the standard that this is guaranteed.  If you see this
  // assert triggering in the wild, feel free to comment it out.  They're
  // likely to standardize that you ARE guaranteed to see the weak pointers
  // as expired in the destructor in the future, so we'll keep this for now.
  TORCH_INTERNAL_ASSERT(self->cdata.expired());  // Assert that 'cdata' is expired

  PyObject_GC_UnTrack(self);  // Untrack the object for Python GC
  THPFunction_clear(self);  // Clear all attributes and vectors
  self->cdata.~weak_ptr<PyNode>();  // Destruct 'cdata' weak pointer
  self->output_info.~vector();  // Destruct 'output_info' vector
  self->input_info.~vector();  // Destruct 'input_info' vector
  self->saved_variables.~vector();  // Destruct 'saved_variables' vector
  self->is_variable_input.~vector();  // Destruct 'is_variable_input' vector
  Py_TYPE(self)->tp_free((PyObject*)self);  // Free memory allocated for 'self'
}

PyObject* THPFunction_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);  // Allocate memory for a new object of type 'type'
  if (!obj)
    // 返回空指针，表示函数执行失败或无效
        return nullptr;
      // Python 会对对象内存进行零初始化，因此大多数字段无需手动初始化
      // 用 obj 强制转换为 THPFunction 指针类型的 self
      THPFunction* self = (THPFunction*)obj;
      // 后续设置 PyNode，此处不能保持其生存
      // 在 self->cdata 上构造一个 std::weak_ptr<PyNode> 对象
      new (&self->cdata) std::weak_ptr<PyNode>();
      // 在 self->output_info 上构造一个空的 std::vector<VariableInfo> 对象
      new (&self->output_info) std::vector<VariableInfo>();
      // 在 self->input_info 上构造一个空的 std::vector<VariableInfo> 对象
      new (&self->input_info) std::vector<VariableInfo>();
      // 在 self->saved_variables 上构造一个空的 std::vector<SavedVariable> 对象
      new (&self->saved_variables) std::vector<SavedVariable>();
      // 在 self->is_variable_input 上构造一个空的 std::vector<bool> 对象
      new (&self->is_variable_input) std::vector<bool>();
      // 设置 self->materialize_grads 为 true，表示需要计算梯度
      self->materialize_grads = true;
      // 设置 self->materialize_non_diff_grads 为 true，表示需要计算非差分梯度
      self->materialize_non_diff_grads = true;
      // 设置 self->compiled_autograd_tracing 为 false，表示未编译自动求导跟踪
      self->compiled_autograd_tracing = false;
      // 返回初始化后的对象指针 obj
      return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

// Bump the counters of all recorded dirty input tensors, adding each of them
// into dirty_inputs.  Also does some sanity checking.
// 标记所有记录的脏输入张量的计数器，并将它们添加到dirty_inputs中。还进行一些健全性检查。
static std::unordered_set<at::TensorImpl*> _mark_dirty(THPFunction* self) {
  // Increase versions of modified tensors
  // 增加修改张量的版本号
  std::unordered_set<at::TensorImpl*> dirty_inputs;
  if (!self->dirty_tensors)
    return dirty_inputs;

  THPFunction_assert(
      PyTuple_Check(self->dirty_tensors),
      "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is ",
      THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  dirty_inputs.reserve(num_dirty);
  for (const auto i : c10::irange(num_dirty)) {
    PyObject* obj = PyTuple_GET_ITEM(self->dirty_tensors, i);
    THPFunction_assert(
        THPVariable_Check(obj),
        "mark_dirty can "
        "only accept variables, but argument ",
        i,
        " is of type ",
        THPUtils_typename(obj));

    const auto& tensor = THPVariable_Unpack(obj);
    dirty_inputs.insert(tensor.unsafeGetTensorImpl());
    torch::autograd::impl::bump_version(tensor);
  }
  // We're not going to ever need this so let's remove references now
  // 现在我们不会再需要它们，所以现在移除引用
  Py_CLEAR(self->dirty_tensors);
  return dirty_inputs;
}

static std::unordered_set<at::TensorImpl*> _parse_non_differentiable(
    THPFunction* self);

// Given a Python tuple of raw output tensors (raw_output), set each of
// the corresponding entries in a different Python tuple (outputs) with
// these tensors wrapped with variables.  We save the gradient function (self)
// to the variable if the output requires grad.
//
// There is a considerable amount of complexity to handle if the operation
// that produced these output tensors is inplace.  A mapping of *input*
// tensors to variables (t2var) is used to test if this occurred, and
// the set of dirty tensors (dirty_inputs) is used to figure out what to
// do in this case.  After this method is run, t2var is extended with
// mappings for output tensors as well.
// 给定一个原始输出张量的Python元组（raw_output），将这些张量包装为变量，并将它们分别设置到另一个Python元组（outputs）的相应条目中。
// 如果输出需要梯度，则将梯度函数（self）保存到变量中。
//
// 如果生成这些输出张量的操作是原地操作，处理起来会非常复杂。
// 一个将*输入*张量映射到变量（t2var）的映射用于测试是否发生了这种情况，
// 并且脏张量集合（dirty_inputs）用于确定在这种情况下该做什么。
// 在运行此方法之后，t2var还会扩展到输出张量的映射。
static void _wrap_outputs(
    const std::shared_ptr<PyNode>& cdata,
    THPFunction* self,
    const variable_list& input_vars,
    PyObject* raw_output,
    PyObject* outputs,
    bool is_executable,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context) {
  auto cdata_if_executable = is_executable ? cdata : nullptr;
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  if (is_executable) {
    self->output_info.clear();
    self->output_info.reserve(num_outputs);
  }

  auto non_differentiable = _parse_non_differentiable(self);
  auto dirty_inputs = _mark_dirty(self);

  std::vector<std::optional<Variable>> raw_output_vars;
  raw_output_vars.reserve(num_outputs);
  for (const auto i : c10::irange(num_outputs)) {
    // 从元组 `raw_output` 中获取第 `i` 个元素（PyObject 对象）
    PyObject* obj = PyTuple_GET_ITEM(raw_output, i);
    
    // 只处理作为自动求导输出的张量
    if (THPVariable_Check(obj)) {
      // 如果 `obj` 是 THPVariable 类型，则解包为 THPVariable，并加入 raw_output_vars
      raw_output_vars.emplace_back(THPVariable_Unpack(obj));
    } else {
      // 如果 `obj` 不是 THPVariable 类型，加入一个空的元素到 raw_output_vars
      raw_output_vars.emplace_back();
    }
  }

  // 定义 jvp_user_function 匿名函数
  _jvp_fn_t jvp_user_function = [self](
                                    variable_list inputs,
                                    variable_list grad_inputs) {
    pybind11::gil_scoped_acquire gil;

    // 将 C++ 的 variable_list 转换为 Python 的参数元组
    // 确保非张量输入对应的位置插入 None
    auto num_inputs = self->is_variable_input.size();
    THPObjectPtr pyInputs(PyTuple_New(static_cast<Py_ssize_t>(num_inputs)));
    if (!pyInputs)
      throw_python_error();
    int64_t variable_idx = 0;
    for (const auto i : c10::irange(num_inputs)) {
      PyObject* input = nullptr;
      if (self->is_variable_input[i]) {
        // 如果输入是变量且梯度已定义，或者不需要生成梯度或者不是可微类型，则包装为 THPVariable
        if (grad_inputs[variable_idx].defined() || !self->materialize_grads ||
            !isDifferentiableType(inputs[variable_idx].scalar_type())) {
          input = THPVariable_Wrap(grad_inputs[variable_idx]);
        } else {
          // 否则生成一个与输入张量相同大小的零张量，并包装为 THPVariable
          input = THPVariable_Wrap(at::zeros_like(inputs[variable_idx]));
        }
        if (!input) {
          throw_python_error();
        }
        variable_idx++;
      } else {
        // 非变量输入位置插入 None
        Py_INCREF(Py_None);
        input = Py_None;
      }
      PyTuple_SET_ITEM(pyInputs.get(), i, input);
    }

    // 获取 self 对象的 apply_jvp 方法
    THPObjectPtr apply_jvp_fn(
        PyObject_GetAttrString((PyObject*)self, "apply_jvp"));
    if (!apply_jvp_fn)
      throw_python_error();
    
    // 调用 apply_jvp 方法，并传入参数元组 pyInputs
    THPObjectPtr r(PyObject_CallObject(apply_jvp_fn, pyInputs.get()));
    if (!r)
      throw_python_error();
    ensure_tuple(r);

    // 将 Python 返回的结果元组转换为 C++ 的 variable_list
    // 不对结果数量进行检查，由调用方处理
    const int num_outputs = PyTuple_GET_SIZE(r.get());
    variable_list results;
    results.reserve(num_outputs);
    for (const auto i : c10::irange(num_outputs)) {
      PyObject* output = PyTuple_GET_ITEM(r.get(), i);
      if (output == Py_None) {
        results.emplace_back();
      } else {
        // 检查输出是否为 THPVariable 类型，并解包到 results 中
        TORCH_CHECK(
            THPVariable_Check(output),
            "expected Variable or None (got ",
            THPUtils_typename(output),
            ") for grad output ",
            i,
            ".")
        results.emplace_back(THPVariable_Unpack(output));
      }
    }

    return results;
  };

  // 定义 view_as_self_fn 匿名函数
  auto view_as_self_fn = [](const at::Tensor& x) -> at::Tensor {
    pybind11::gil_scoped_acquire gil;
    THPObjectPtr py_x(THPVariable_Wrap(x));
    
    // 获取 py_x 对象的 view_as 方法
    THPObjectPtr py_view_as_method(PyObject_GetAttrString(py_x, "view_as"));
    if (!py_view_as_method)
      throw python_error();
      
    // 调用 view_as 方法，并传入 py_x 作为参数
    THPObjectPtr args(PyTuple_Pack(1, py_x.get()));
    if (!args)
      throw python_error();
    THPObjectPtr result(PyObject_CallObject(py_view_as_method, args));
    // 如果 result 为空指针，则抛出 Python 错误异常
    if (!result)
      throw python_error();
    // 返回解包后的 THPVariable 对象
    return THPVariable_Unpack(result);
    };
    
    // 仅包装张量类型的输出。
    auto wrapped_outputs = _wrap_outputs(
        input_vars,                     // 输入变量
        non_differentiable,             // 不可微分的标志
        dirty_inputs,                   // 脏输入标志
        raw_output_vars,                // 原始输出变量
        cdata_if_executable,            // 如果是可执行的，则是 CData
        jvp_user_function,              // JVP 用户函数
        to_save_if_setup_context,       // 如果在设置上下文中保存
        view_as_self_fn);               // 视为自身函数
    
    for (const auto i : c10::irange(num_outputs)) {
      PyObject* obj = PyTuple_GetItem(raw_output, i);
      // 保持非张量类型的输出不变。
      if (!THPVariable_Check(obj)) {
        if (is_executable) {
          self->output_info.emplace_back();  // 将空的输出信息添加到 self->output_info
        }
        Py_INCREF(obj);                     // 增加 Python 对象的引用计数
        PyTuple_SetItem(outputs, i, obj);   // 将 obj 设置为 PyTuple 的第 i 个元素
      } else {
        if (is_executable) {
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          self->output_info.emplace_back(*wrapped_outputs[i]);  // 将 wrapped_outputs[i] 添加到 self->output_info
        }
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        PyTuple_SetItem(outputs, i, THPVariable_Wrap(*wrapped_outputs[i]));  // 将 wrapped_outputs[i] 包装为 THPVariable 并设置到 outputs 的第 i 个位置
      }
    }
}

// 定义一个静态函数 `_get_tensors_to_save`，用于从函数对象中获取要保存的张量信息
static void _get_tensors_to_save(
    THPFunction* self,  // 函数对象自身的指针
    std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,  // 要保存的张量实现对象的集合
    std::vector<std::optional<at::Tensor>>& tensors_to_save,  // 要保存的张量的可选容器
    bool overridden_setup_context,  // 是否覆盖了设置上下文
    bool is_executable) {  // 是否可执行

  // 如果 `saved_for_forward` 存在且设置了 `overridden_setup_context`
  if (self->saved_for_forward && overridden_setup_context) {
    // 检查 `saved_for_forward` 是否为元组类型
    THPFunction_assert(
        PyTuple_Check(self->saved_for_forward),
        "autograd internal "
        "error: saved_for_forward attribute is expected to be a tuple but is ",
        THPUtils_typename(self->saved_for_forward));
    Py_ssize_t num_saved_for_forward =
        PyTuple_GET_SIZE(self->saved_for_forward);

    // 遍历 `saved_for_forward` 中的元素
    for (const auto i : c10::irange(num_saved_for_forward)) {
      PyObject* obj = PyTuple_GET_ITEM(self->saved_for_forward, i);
      // 如果元素是 THPVariable 类型
      if (THPVariable_Check(obj)) {
        // 解包为 Tensor 对象
        const auto& tensor = THPVariable_Unpack(obj);
        // 将张量的实现对象插入到 `to_save_if_setup_context` 集合中
        to_save_if_setup_context.insert(tensor.unsafeGetTensorImpl());
      }
    }
  }

  // 如果 `to_save` 属性存在
  if (self->to_save) {
    // 检查 `to_save` 是否为元组类型
    THPFunction_assert(
        PyTuple_Check(self->to_save),
        "autograd internal "
        "error: to_save attribute is expected to be a tuple but is ",
        THPUtils_typename(self->to_save));

    Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);

    // 遍历 `to_save` 中的元素
    for (const auto i : c10::irange(num_saved)) {
      PyObject* obj = PyTuple_GET_ITEM(self->to_save, i);

      // 如果元素是 None
      if (obj == Py_None) {
        // 在 `tensors_to_save` 中添加一个空的 optional
        tensors_to_save.emplace_back(c10::nullopt);
        continue;
      }
      // 如果元素是 THPVariable 类型
      else if (THPVariable_Check(obj)) {
        // 解包为 Tensor 对象
        const auto& tensor = THPVariable_Unpack(obj);
        // 如果覆盖了设置上下文，将张量的实现对象插入到 `to_save_if_setup_context` 集合中
        if (overridden_setup_context) {
          to_save_if_setup_context.insert(tensor.unsafeGetTensorImpl());
        }
        // 如果可执行，将张量添加到 `tensors_to_save` 中
        if (is_executable) {
          tensors_to_save.emplace_back(tensor);
        }
      }
      // 如果元素类型不是 THPVariable
      else {
        // 如果可执行，抛出类型错误
        if (is_executable) {
          // TODO: 我们真的应该在这里始终抛出错误，但这样做会破坏一些内部测试，我们应该修复这些。
          throw torch::TypeError(
              "save_for_backward can only save variables, but argument %ld is of "
              "type %s",
              i,
              Py_TYPE(obj)->tp_name);
        }
      }
    }
  }
}

// 保存由 `to_save` 请求的任何变量
static void _save_variables(
    const std::vector<std::optional<at::Tensor>>& tensors_to_save,  // 要保存的张量的可选容器
    const std::shared_ptr<PyNode>& cdata_ptr,  // 共享指针的 PyNode 对象
    THPFunction* self) {  // 函数对象自身的指针
  // 如果 `to_save` 为空，直接返回
  if (!self->to_save)
    return;

  // 获取要保存的张量数量
  size_t num_saved = tensors_to_save.size();

  // 清空 `saved_variables` 向量，并预留空间
  self->saved_variables.clear();
  self->saved_variables.reserve(num_saved);

  // 遍历 `tensors_to_save` 中的每个可选张量
  for (const auto& opt_tensor : tensors_to_save) {
    // 如果张量为空值，将空值添加到 `saved_variables`
    if (!opt_tensor.has_value()) {
      self->saved_variables.emplace_back();
    } else {
      // 否则，判断张量是否是输出，将其与是否与 `cdata_ptr` 相同进行比较
      bool is_output = opt_tensor.value().grad_fn().get() == cdata_ptr.get();
      self->saved_variables.emplace_back(opt_tensor.value(), is_output);
    }
  }
  // 释放 self->to_save 对象的引用计数，确保内存被正确释放
  Py_CLEAR(self->to_save);
}

// 在不可微变量上标记 requires_grad = 0（根据 non_differentiable）
static std::unordered_set<at::TensorImpl*> _parse_non_differentiable(
    THPFunction* self) {
  // 创建一个空的无序集合用于存储不可微变量的 TensorImpl 指针
  std::unordered_set<at::TensorImpl*> set;
  // 如果 self->non_differentiable 为空，则直接返回空集合
  if (!self->non_differentiable)
    return set;

  // 断言 self->non_differentiable 应该是一个元组
  THPFunction_assert(
      PyTuple_Check(self->non_differentiable),
      "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is ",
      THPUtils_typename(self->non_differentiable));
  // 获取非可微变量的数量
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  // 预留空间以存储非可微变量的 TensorImpl 指针
  set.reserve(num_nondiff);
  // 遍历非可微变量元组
  for (const auto i : c10::irange(num_nondiff)) {
    // 获取第 i 个元素
    PyObject* t = PyTuple_GET_ITEM(self->non_differentiable, i);
    // 断言 t 是一个 THPVariable
    THPFunction_assert(
        THPVariable_Check(t),
        "mark_non_differentiable "
        "only accepts variable arguments, but got ",
        THPUtils_typename(t));
    // 将对应的 TensorImpl 指针插入集合中
    set.insert(THPVariable_Unpack(t).unsafeGetTensorImpl());
  }
  // 清空 self->non_differentiable 引用的 Python 对象，防止内存泄漏
  Py_CLEAR(self->non_differentiable);
  // 返回存储了非可微变量 TensorImpl 指针的集合
  return set;
}

// UnpackedInput 结构体定义，用于存储解包后的输入数据
struct UnpackedInput {
  THPObjectPtr input_tuple;  // 输入数据的 Python 元组对象
  variable_list input_vars;  // 输入的变量列表
  // record_function_inputs 仅用于 RECORD_FUNCTION
  std::vector<c10::IValue> record_function_inputs;  // 记录函数输入的 IValue
};

// InputFlags 结构体定义，用于记录输入的标志信息
struct InputFlags {
  bool is_executable = false;  // 是否可执行
  edge_list next_edges;        // 下一个边缘列表
  THPObjectPtr needs_input_grad;  // 是否需要输入梯度的 Python 对象
  std::vector<bool> is_variable_input;  // 输入是否为变量的标志列表
};

// unpack_input 模板函数定义，用于解包输入参数
template <bool enforce_variables>
std::pair<UnpackedInput, InputFlags> unpack_input(PyObject* args) {
  UnpackedInput unpacked;  // 创建 UnpackedInput 结构体实例
  InputFlags flags;        // 创建 InputFlags 结构体实例

  auto num_args = PyTuple_GET_SIZE(args);  // 获取参数元组的大小
  unpacked.input_tuple = PyTuple_New(num_args);  // 创建一个新的 Python 元组对象
  flags.needs_input_grad = PyTuple_New(num_args);  // 创建一个新的 Python 元组对象
  bool profiler_need_input = torch::autograd::profiler::profilerEnabled() &&
      torch::autograd::profiler::getProfilerConfig().report_input_shapes;

  // 遍历参数元组中的每一个参数
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);  // 获取第 i 个参数

    bool is_variable = THPVariable_Check(arg);  // 检查参数是否为 THPVariable
    flags.is_variable_input.push_back(is_variable);  // 将是否为变量的标志添加到列表中
    if (!is_variable) {
      // TODO: 当 Variable 和 Tensor 在 Python 中合并后，移除此代码路径
      if (enforce_variables) {
        // 如果需要强制变量类型，抛出错误
        THPUtils_setError(
            "expected a Tensor argument, but got ", THPUtils_typename(arg));
        throw python_error();
      }
      Py_INCREF(Py_False);
      // 设置不需要输入梯度的标志为 False
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, Py_False);

      if (profiler_need_input) {
        // 如果启用了分析器并且需要输入形状信息，则将 PyObject 转换为 IValue
        auto match = torch::jit::tryToInferPrimitiveType(arg);
        if (match.success()) {
          unpacked.record_function_inputs.push_back(
              torch::jit::toIValue(arg, match.type()));
        }
      }
    }
    } else {
      // 如果参数不是 Variable 对象，则将其解包成 Tensor 对象
      const auto& tensor = THPVariable_Unpack(arg);
      // 将解包后的 Tensor 对象添加到解包后的输入变量列表中
      unpacked.input_vars.push_back(tensor);
      // 检查当前 Tensor 是否需要梯度计算，设置相应的 Python 布尔值
      PyObject* needs_grad = tensor.requires_grad() ? Py_True : Py_False;
      // 增加 Python 对象的引用计数
      Py_INCREF(needs_grad);
      // 将需要梯度计算的信息设置到输入梯度标记元组的第 i 个位置
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, needs_grad);
      // 将 Tensor 对象记录到记录函数输入的列表中
      unpacked.record_function_inputs.emplace_back(tensor);
    }
    // 增加 Python 对象的引用计数
    Py_INCREF(arg);
    // 将处理后的参数对象设置到输入元组的第 i 个位置
    PyTuple_SET_ITEM(unpacked.input_tuple.get(), i, arg);
  }

  // 设置执行标志，如果梯度模式开启且任何输入变量需要梯度计算
  flags.is_executable =
      GradMode::is_enabled() && any_variable_requires_grad(unpacked.input_vars);
  // 设置下一个边缘列表，如果可执行则收集输入变量的梯度边缘，否则为空列表
  flags.next_edges =
      (flags.is_executable ? collect_next_edges(unpacked.input_vars)
                           : edge_list());
  // 返回解包后的结果和标志作为 pair 对象
  return std::make_pair(std::move(unpacked), std::move(flags));
}

// 给定一个 prim::PythonOp 节点，_append_subgraph 函数创建一个子图，满足以下条件：
// (1) 子图具有与 prim::PythonOp 节点相同的输入
// (2) 用于 PythonOp 的中间节点被克隆并存储在子图中
// (3) trace_outputs 存储 Value* 对象，在 prim::PythonOp 节点分配新的跟踪值之前，帮助正确路由子图的输出
// 这个新创建的子图然后作为一个子图属性被添加到 prim::PythonOp 节点上
static void _append_subgraph(
    torch::jit::Node* node,                   // 要添加子图的 prim::PythonOp 节点
    torch::jit::Graph* graph,                 // 主图的当前作用域的 graph
    std::vector<torch::jit::Value*> trace_outputs,  // 跟踪输出的 Value* 对象
    bool unpack_output) {                     // 是否解包输出的标志
  using Value = torch::jit::Value;
  // 将 subgraph 属性设置为一个新的 torch::jit::Graph，当前作用域与主图相同
  node->g_(torch::jit::attr::Subgraph, std::make_shared<torch::jit::Graph>(graph->current_scope()));
  auto subgraph = node->g(torch::jit::attr::Subgraph);

  // 值映射，用于将主图节点的值映射到子图节点的值
  std::unordered_map<Value*, Value*> value_map;
  auto value_map_func = [&](Value* v) { return value_map.at(v); };

  // 复制主图节点的输入到子图中，并设置相同的元数据
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    auto subgraph_input = subgraph->addInput();
    subgraph_input->copyMetadata(node->inputs().at(i));
    value_map[node->inputs().at(i)] = subgraph_input;
  }

  // 查找节点在所属块中的位置，之后的所有节点都被添加到子图中
  auto owning_block = node->owningBlock();
  auto it = std::find(owning_block->nodes().begin(), owning_block->nodes().end(), node);

  // 如果不需要解包输出，则跳过 TupleUnpack 节点
  if (!unpack_output) {
    it++;
  }

  // 将节点及其后续节点克隆到子图中
  for (it++; it != owning_block->nodes().end(); ++it) {
    torch::jit::Node* node = *it;
    auto* clone_node = subgraph->insertNode(subgraph->createClone(node, value_map_func));
    // 更新值映射，将主图节点的输出值映射到子图节点的输出值
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = clone_node->outputs()[i];
      // 如果该输出值需要被跟踪，则在子图中注册该输出值
      auto trace_it = std::find(trace_outputs.begin(), trace_outputs.end(), node->outputs()[i]);
      if (trace_it != trace_outputs.end()) {
        subgraph->registerOutput(clone_node->outputs()[i]);
      }
    }
  }
}

// 在执行 Python 操作之前进行跟踪记录
static torch::jit::Node* _trace_pre_record(
    PyObject* op_obj,                       // Python 操作对象
    PyObject* input_objects,                // 输入对象元组
    const variable_list& input_vars) {      // 输入变量列表
  if (!jit::tracer::isTracing()) {
    return nullptr;  // 如果不在跟踪状态，则返回空指针
  }

  // 保存标量参数和调用约定
  auto num_args = PyTuple_GET_SIZE(input_objects);
  pyobj_list scalar_args;
  std::string arg_types;
  arg_types.reserve(num_args);
  scalar_args.reserve(num_args);

  // 遍历输入对象元组，确定每个参数的类型，并将标量参数保存到 scalar_args 中
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg_object = PyTuple_GET_ITEM(input_objects, i);
    if (THPVariable_Check(arg_object)) {
      arg_types.push_back('d');  // 双精度数值类型
    } else {
      arg_types.push_back('c');  // 复杂对象类型
      Py_INCREF(arg_object);
      scalar_args.emplace_back(arg_object);
    }
  }

  Py_INCREF(op_obj);
  auto pyobj = THPObjectPtr(op_obj);
  // 执行 Python 跟踪记录，并返回记录下来的 prim::PythonOp 节点
  return jit::tracer::preRecordPythonTrace(
      std::move(pyobj), arg_types, input_vars, std::move(scalar_args));
}

// 在执行 Python 操作之后进行跟踪记录
static void _trace_post_record(
    // 设置节点的 inplace 属性，表示是否为原地操作
    node->i_(jit::attr::inplace, is_inplace);
    
    // 从操作对象中获取模块名，并设置到节点的 module 属性中
    if (PyObject* module_name = PyDict_GetItemString(
            ((PyTypeObject*)op_obj)->tp_dict, "__module__")) {
        if (auto ptr = PyUnicode_AsUTF8(module_name)) {
            node->s_(jit::attr::module, std::string(ptr));
        }
    }
    
    // 如果不是拆分输出，则为节点添加一个输出
    int num_outputs = PyTuple_GET_SIZE(output_objects);
    auto graph = node->owningGraph();
    node->addOutput();
    auto old_node = node;
    if (!unpack_output) {
        // 创建一个元组类型，元素类型为张量
        std::vector<at::TypePtr> tuple_values(num_outputs, at::TensorType::get());
        auto tuple_type = at::TupleType::create(std::move(tuple_values));
        // 将节点的输出类型设置为这个元组类型
        node->output()->setType(std::move(tuple_type));
        // 在节点后创建元组拆分操作
        auto unpacked = graph->createTupleUnpack(node->output())->insertAfter(node);
        node = unpacked;
    }
    
    // 对于每个输出对象，将其转换为张量并设置跟踪信息
    std::vector<torch::jit::Value*> trace_outputs;
    for (const auto i : c10::irange(num_outputs)) {
        PyObject* obj = PyTuple_GET_ITEM(output_objects, i);
        if (THPVariable_Check(obj)) {
            auto value = node->outputs()[i];
            const auto& tensor = THPVariable_Unpack(obj);
            if (tensor.defined()) {
                // 推断张量的类型，并更新跟踪信息
                value->inferTypeFrom(tensor);
                trace_outputs.push_back(jit::tracer::getValueTrace(tensor));
                jit::tracer::setValueTrace(tensor, value);
            }
        }
    }
    
    // 导入需要使用的 Python 模块并获取全局变量
    py::object onnx_globals = py::module::import("torch.onnx._globals");
    py::bool_ is_in_onnx_export =
        py::module::import("torch.onnx.__init__").attr("is_in_onnx_export");
    py::bool_ is_autograd_inlining_enabled =
        py::cast<bool>(onnx_globals.attr("GLOBALS").attr("autograd_inlining"));
    
    // 如果在导出到 ONNX 时和自动求导内联都启用，则将子图附加到原始节点上
    if (py::cast<bool>(is_in_onnx_export) &&
        py::cast<bool>(is_autograd_inlining_enabled)) {
        _append_subgraph(old_node, graph, std::move(trace_outputs), unpack_output);
    }
    
    // 如果创建了 TupleUnpack 操作符，将其输出类型复制回原始元组类型
    if (!unpack_output) {
        std::vector<at::TypePtr> new_tuple_values;
        for (const auto i : c10::irange(num_outputs)) {
            auto ptr = node->outputs()[i]->type();
            new_tuple_values.push_back(ptr);
        }
        auto tuple_type = at::TupleType::create(std::move(new_tuple_values));
        old_node->output()->setType(std::move(tuple_type));
    }
}

// 处理函数的输出，返回一个 PyObject 指针
PyObject* process_outputs(
    PyObject* op_obj,  // 操作对象的 PyObject 指针
    const std::shared_ptr<PyNode>& cdata,  // 共享指针，指向 PyNode 对象的常量引用
    THPFunction* grad_fn,  // THPFunction 对象的指针，用于梯度函数
    const UnpackedInput& unpacked,  // UnpackedInput 对象的常量引用，表示解包后的输入
    PyObject* inputs,  // 输入的 PyObject 指针
    THPObjectPtr&& raw_output,  // 右值引用，指向 THPObjectPtr 对象的 PyObject 指针
    bool is_executable,  // 布尔值，表示是否可执行
    torch::jit::Node* node,  // torch::jit::Node 对象的指针，表示节点
    bool overridden_setup_context) {  // 布尔值，表示是否覆盖了设置上下文

  // 确保 raw_output 是元组形式
  bool unpack_output = ensure_tuple(raw_output);

  // 获取元组中的输出数量
  auto num_outputs = PyTuple_GET_SIZE(raw_output.get());

  // 创建一个长度为 num_outputs 的元组对象，用于存放输出
  THPObjectPtr outputs(PyTuple_New(num_outputs));
  if (!outputs)
    throw python_error();

  // 清除输入的元数据
  cdata->clear_input_metadata();

  // 如果可执行，记录关于输入的类型、设备和大小信息
  if (is_executable) {
    grad_fn->input_info.clear();
    grad_fn->input_info.reserve(unpacked.input_vars.size());
    for (auto& var : unpacked.input_vars) {
      grad_fn->input_info.emplace_back(var);
    }
  }

  // 获取需要保存的张量列表和相关设置
  std::unordered_set<at::TensorImpl*> to_save_if_setup_context{};
  std::vector<std::optional<at::Tensor>> tensors_to_save{};
  _get_tensors_to_save(
      grad_fn,
      to_save_if_setup_context,
      tensors_to_save,
      overridden_setup_context,
      is_executable);

  // 检查是否是 inplace 操作
  bool is_inplace = static_cast<bool>(grad_fn->dirty_tensors);

  // 包装输出
  _wrap_outputs(
      cdata,
      grad_fn,
      unpacked.input_vars,
      raw_output,
      outputs,
      is_executable,
      to_save_if_setup_context);

  // 进行记录后跟踪
  _trace_post_record(
      node, op_obj, unpacked.input_vars, outputs, is_inplace, unpack_output);

  // 在保存变量之前创建 SavedVariables 是很重要的，因为输出在保存之前必须设置好 grad_fn/fw_grad
  if (is_executable) {
    _save_variables(tensors_to_save, cdata, grad_fn);
  } else {
    // 移除不必要的属性
    Py_XDECREF(grad_fn->to_save);
    grad_fn->to_save = nullptr;
    Py_XDECREF(grad_fn->non_differentiable);
    grad_fn->non_differentiable = nullptr;
  }

  // 清空 saved_for_forward 属性
  Py_XDECREF(grad_fn->saved_for_forward);
  grad_fn->saved_for_forward = nullptr;

  // 如果需要解包输出（非元组形式），只返回第一个元素
  if (unpack_output) {
    PyObject* output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output);
    return output;
  }

  // 返回包装后的输出元组
  return outputs.release();
}

// 函数 THPFunction_name 的实现，返回一个 PyObject 指针
PyObject* THPFunction_name(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 获取 self 对应的 cdata 弱引用
  auto cdata = ((THPFunction*)self)->cdata.lock();
  // 检查 cdata 是否为空，否则抛出错误信息
  TORCH_CHECK(
      cdata,
      "Attribute 'name' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  // 封装 cdata 的 name 属性，并返回 PyObject 指针
  return THPUtils_packString(cdata->name());
  END_HANDLE_TH_ERRORS
}

// 函数 THPFunction_sequence_nr 的实现，返回一个 PyObject 指针
PyObject* THPFunction_sequence_nr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS;
  // 获取 self 对应的 cdata 弱引用
  auto cdata = ((THPFunction*)self)->cdata.lock();
  // 封装 cdata 的 sequence_nr 属性，并返回 PyObject 指针
  return THPUtils_packUInt64(cdata->sequence_nr());
  END_HANDLE_TH_ERRORS
}
PyObject* THPFunction_set_sequence_nr(PyObject* self, PyObject* sequence_nr) {
  HANDLE_TH_ERRORS;
  // 获取 self 对应的 THPFunction 对象的弱引用指针，并锁定其对应的 cdata 对象
  auto cdata = ((THPFunction*)self)->cdata.lock();
  // 设置 cdata 对象的序列号为从 sequence_nr 中解包得到的无符号 64 位整数
  cdata->set_sequence_nr(THPUtils_unpackUInt64(sequence_nr));
  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_maybe_clear_saved_tensors(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS;
  // 获取 self 对应的 THPFunction 对象的弱引用指针，并锁定其对应的 cdata 对象
  auto cdata = ((THPFunction*)self)->cdata.lock();
  // 如果当前没有保持图形状态的图任务存在
  if (!get_current_graph_task_keep_graph()) {
    // 释放 cdata 对象中保存的变量
    cdata->release_variables();
  }
  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

namespace {

THPObjectPtr make_ctx_input_tuple(
    THPFunction* ctx,
    const UnpackedInput& unpacked_input,
    int64_t num_args) {
  // 创建一个包含 num_args + 1 个元素的 Python 元组对象 ctx_input_tuple
  THPObjectPtr ctx_input_tuple(PyTuple_New(num_args + 1));
  // 如果创建失败，则返回空对象
  if (!ctx_input_tuple)
    return {};
  // 增加对 ctx 对象的引用计数，并将其设置为元组的第一个元素
  Py_INCREF(ctx);
  PyTuple_SET_ITEM(ctx_input_tuple.get(), 0, (PyObject*)ctx);
  // 遍历 unpacked_input.input_tuple 中的元素，增加每个元素的引用计数，并添加到 ctx_input_tuple 中
  for (const auto i : c10::irange(num_args)) {
    PyObject* arg = PyTuple_GET_ITEM(unpacked_input.input_tuple.get(), i);
    Py_INCREF(arg);
    PyTuple_SET_ITEM(ctx_input_tuple.get(), i + 1, arg);
  }
  // 返回创建的元组对象
  return ctx_input_tuple;
}

THPObjectPtr make_ctx_input_output_tuple(
    THPFunction* ctx,
    UnpackedInput& unpacked_input,
    PyObject* output) {
  // 创建一个包含 3 个元素的 Python 元组对象 result
  THPObjectPtr result(PyTuple_New(3));
  // 如果创建失败，则返回空对象
  if (!result)
    return {};
  // 增加对 ctx、unpacked_input.input_tuple 和 output 的引用计数，并设置为 result 的元素
  Py_INCREF(ctx);
  Py_INCREF(unpacked_input.input_tuple.get());
  Py_INCREF(output);
  PyTuple_SET_ITEM(result.get(), 0, (PyObject*)ctx);
  PyTuple_SET_ITEM(result.get(), 1, unpacked_input.input_tuple.get());
  PyTuple_SET_ITEM(result.get(), 2, output);
  // 返回创建的元组对象
  return result;
}

} // namespace

// 静态全局变量 THPFunction_setup_context 的定义与初始化
static PyObject* THPFunction_setup_context = nullptr;

// 获取基础的设置上下文对象
static PyObject* get_base_setup_context() {
  // 如果 THPFunction_setup_context 不为空，则直接返回它
  if (THPFunction_setup_context != nullptr) {
    return THPFunction_setup_context;
  }

  // 尝试导入 torch.autograd.function 模块
  auto module = THPObjectPtr(PyImport_ImportModule("torch.autograd.function"));
  if (!module)
    return nullptr;

  // 获取模块中名为 _SingleLevelFunction 的属性对象
  auto function =
      THPObjectPtr(PyObject_GetAttrString(module, "_SingleLevelFunction"));
  if (!function)
    return nullptr;

  // 获取 function 对象中名为 setup_context 的属性对象
  // setup_context 对象的引用计数将会被增加，并且被 THPFunction_setup_context 持有
  auto setup_context = PyObject_GetAttrString(function, "setup_context");
  if (!setup_context)
    return nullptr;
  THPFunction_setup_context = setup_context;
  // 返回 setup_context 对象
  return THPFunction_setup_context;
}

PyObject* THPFunction_apply(PyObject* cls, PyObject* inputs) {
  HANDLE_TH_ERRORS

  // 获取当前的序列号，并保存一个副本在其被增加之前
  auto seq_id = at::sequence_number::peek();
  // 解包输入，并获得解包后的输入信息和输入标记
  auto info_pair = unpack_input<false>(inputs);
  UnpackedInput& unpacked_input = info_pair.first;
  InputFlags& input_info = info_pair.second;

  // 在所有输入已解码但未分配上下文之前调用记录函数
  RECORD_FUNCTION(
      ((PyTypeObject*)cls)->tp_name,
      unpacked_input.record_function_inputs,
      seq_id);

  // 获取 functorchTLS 访问器返回的 functorch_tls
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    // autograd.Function support for functorch is handled in Python.
    // If we have gotten here, then either we are dealing with a
    // torch.autograd.function._SingleLevelFunction, or something in
    // the implementation went wrong.
    // The following code is useful for debugging when something goes wrong
    // because it'll raise a loud error (instead of being silently incorrect).
    functorch_tls->checkSupportsSingleLevelAutogradFunction();
  }

  // 获取类属性 _backward_cls
  THPObjectPtr backward_cls(PyObject_GetAttrString(cls, "_backward_cls"));
  if (!backward_cls)
    return nullptr;
  // 调用 _backward_cls 构造函数，创建 ctx_obj
  THPObjectPtr ctx_obj(PyObject_CallFunctionObjArgs(backward_cls, nullptr));
  if (!ctx_obj)
    return nullptr;
  // 将 ctx_obj 转换为 THPFunction 类型的指针 ctx
  THPFunction* ctx = (THPFunction*)ctx_obj.get();

  // 创建共享指针 cdata，关联到新创建的 PyNode 对象，使用 deleteNode 作为删除器
  auto cdata =
      std::shared_ptr<PyNode>(new PyNode(std::move(ctx_obj)), deleteNode);
  // 将 cdata 分配给 ctx 的 cdata 成员变量
  ctx->cdata = cdata;

  // 如果正在追踪，则记录输入节点
  auto* node = _trace_pre_record(cls, inputs, unpacked_input.input_vars);

  // 初始化反向函数及其上下文 ctx
  bool is_executable = input_info.is_executable;
  // 设置 ctx 的下一条边
  cdata->set_next_edges(std::move(input_info.next_edges));
  // 释放 input_info.needs_input_grad 的所有权给 ctx->needs_input_grad
  ctx->needs_input_grad = input_info.needs_input_grad.release();
  // 将 input_info.is_variable_input 转移给 ctx->is_variable_input
  ctx->is_variable_input = std::move(input_info.is_variable_input);

  // autograd.Function 可能选择重写 setup_context 静态方法
  // 确定是否重写了 setup_context
  auto cls_setup_context =
      THPObjectPtr(PyObject_GetAttrString(cls, "setup_context"));
  if (!cls_setup_context) {
    return nullptr;
  }
  // 获取基类的 setup_context
  auto orig_setup_context = get_base_setup_context();
  if (!orig_setup_context) {
    return nullptr;
  }
  // 检查是否 cls_setup_context 与 orig_setup_context 不同
  auto overridden_setup_context = cls_setup_context.get() != orig_setup_context;

  // 获取输入的参数个数
  auto num_args = PyTuple_GET_SIZE(inputs);

  // 调用 forward 方法
  THPObjectPtr output;
  {
    // 禁用梯度追踪
    AutoGradMode grad_mode(false);
    // 禁用前向梯度模式
    at::AutoFwGradMode fw_grad_mode(false);
    // 获取 forward 方法
    THPObjectPtr forward_fn(PyObject_GetAttrString(cls, "forward"));
    if (!forward_fn)
      return nullptr;
    if (overridden_setup_context) {
      // 调用 forward 后紧接着调用 setup_context
      output = PyObject_CallObject(forward_fn, unpacked_input.input_tuple);
      if (!output) {
        return nullptr;
      }
      // setup_context 的签名是 setup_context(ctx, inputs, output)
      // 创建 ctx, inputs, output 的元组
      auto ctx_input_output_tuple =
          make_ctx_input_output_tuple(ctx, unpacked_input, output);
      if (!ctx_input_output_tuple) {
        return nullptr;
      }
      // 调用 setup_context 方法
      THPObjectPtr setup_context_fn(
          PyObject_GetAttrString(cls, "setup_context"));
      auto result =
          PyObject_CallObject(setup_context_fn, ctx_input_output_tuple);
      if (!result) {
        return nullptr;
      }
    } else {
      // 直接调用 forward 方法
      auto ctx_input_tuple =
          make_ctx_input_tuple(ctx, unpacked_input, num_args);
      if (!ctx_input_tuple) {
        return nullptr;
      }
      output = PyObject_CallObject(forward_fn, ctx_input_tuple);
    }
    if (!output)
      return nullptr;
  }



    // 如果 output 为空指针，则返回 nullptr
    if (!output)
      return nullptr;
  }


注释：
这段代码检查变量 `output` 是否为空指针。如果为空，函数返回 `nullptr`，即空指针，表示处理无法继续或无效。这种情况下可能需要进行错误处理或异常处理。


  return process_outputs(
      cls,
      cdata,
      ctx,
      unpacked_input,
      inputs,
      std::move(output),
      is_executable,
      node,
      overridden_setup_context);
  END_HANDLE_TH_ERRORS



  // 调用 process_outputs 函数，处理输出数据，并返回结果
  return process_outputs(
      cls,  // 类型信息
      cdata,  // C 数据
      ctx,  // 上下文
      unpacked_input,  // 解包后的输入
      inputs,  // 输入数据
      std::move(output),  // 移动输出数据
      is_executable,  // 是否可执行
      node,  // 节点信息
      overridden_setup_context);  // 覆盖设置上下文
  END_HANDLE_TH_ERRORS  // 结束错误处理


注释：
这段代码调用了 `process_outputs` 函数，将多个参数传递给它，然后返回它的结果。这里使用了 C++ 的移动语义 `std::move(output)` 来传递 `output` 变量的所有权，以便在函数内部进行有效的资源管理。`END_HANDLE_TH_ERRORS` 可能是一个宏，用于处理可能发生的错误或异常情况。
////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

// 注册钩子字典的函数，用于向 autograd.Function 实例中注册钩子
PyObject* THPFunction__register_hook_dict(PyObject* _self, PyObject* _var) {
  HANDLE_TH_ERRORS
  // 检查传入的对象是否为 Tensor
  TORCH_CHECK(THPVariable_Check(_var), "_register_hook_dict expected a Tensor");
  // 将 PyObject 转换为 THPVariable 指针
  THPVariable* var = reinterpret_cast<THPVariable*>(_var);
  // 获取 Tensor 对象的引用
  const auto& tensor = THPVariable_Unpack(var);
  // 创建一个 PyFunctionTensorPreHook 实例，注册到 var 的 backward_hooks 中
  std::unique_ptr<FunctionPreHook> hook(
      new PyFunctionTensorPreHook(var->backward_hooks, tensor.output_nr()));
  // 获取当前实例的 self 指针
  auto self = (THPFunction*)_self;
  // 获取当前实例的 cdata，转换为 shared_ptr
  auto cdata = self->cdata.lock();
  // 检查 cdata 是否有效
  TORCH_CHECK(
      cdata,
      "Attribute '_register_hook_dict' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  // 将 hook 添加到 cdata 的 tensor_pre_hooks 中
  cdata->add_tensor_pre_hook(std::move(hook));
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 注册钩子的函数，用于向 autograd.Function 实例中注册钩子
PyObject* THPFunction_register_hook(PyObject* _self, PyObject* hook) {
  HANDLE_TH_ERRORS
  // 获取当前实例的 self 指针
  auto self = (THPFunction*)_self;
  // 获取当前实例的 cdata，转换为 shared_ptr
  auto cdata = self->cdata.lock();
  // 检查 cdata 是否有效
  TORCH_CHECK(
      cdata,
      "Attribute 'register_hook' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  // 调用 torch::autograd::registerFunctionHook 注册 hook 到 cdata 中
  return torch::autograd::registerFunctionHook(*cdata, hook);
  END_HANDLE_TH_ERRORS
}

// 注册预钩子的函数，用于向 autograd.Function 实例中注册预钩子
PyObject* THPFunction_register_prehook(PyObject* _self, PyObject* hook) {
  HANDLE_TH_ERRORS
  // 获取当前实例的 self 指针
  auto self = (THPFunction*)_self;
  // 获取当前实例的 cdata，转换为 shared_ptr
  auto cdata = self->cdata.lock();
  // 检查 cdata 是否有效
  TORCH_CHECK(
      cdata,
      "Attribute 'register_prehook' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  // 调用 torch::autograd::registerFunctionPreHook 注册 hook 到 cdata 中
  return torch::autograd::registerFunctionPreHook(*cdata, hook);
  END_HANDLE_TH_ERRORS
}

// 设置 materialize_grads 属性的函数
int THPFunction_set_materialize_grads(
    THPFunction* self,
    PyObject* value,
    void* unused) {
  HANDLE_TH_ERRORS
  // 检查传入的 value 是否为布尔类型
  if (!PyBool_Check(value)) {
    // 如果不是布尔类型，抛出异常
    THPUtils_invalidArguments(
        value, nullptr, "set_materialize_grads", 1, "(bool)");
    return -1;
  }
  // 将布尔值转换为对应的 self->materialize_grads 属性
  self->materialize_grads = (value == Py_True);
  // 返回 0 表示成功
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 获取 materialize_non_diff_grads 属性的函数
PyObject* THPFunction_get_materialize_non_diff_grads(
    THPFunction* self,
    ```
    // 开始错误处理区块
    HANDLE_TH_ERRORS
        // 如果 self 对象的 materialize_non_diff_grads 属性为真，则返回 Python 中的 True
        if (self->materialize_non_diff_grads) {
            Py_RETURN_TRUE;
        } else {
            // 否则返回 Python 中的 False
            Py_RETURN_FALSE;
        }
    // 结束错误处理区块
    END_HANDLE_TH_ERRORS
}

// 设置函数的非导数梯度材料化选项
int THPFunction_set_materialize_non_diff_grads(
    THPFunction* self,
    PyObject* value,
    void* unused) {
  HANDLE_TH_ERRORS
  // 检查传入的值是否为布尔类型
  if (!PyBool_Check(value)) {
    // 如果不是布尔类型，抛出参数错误异常
    THPUtils_invalidArguments(
        value, nullptr, "set_materialize_non_diff_grads", 1, "(bool)");
    return -1;
  }
  // 根据传入的布尔值设置非导数梯度材料化选项
  self->materialize_non_diff_grads = (value == Py_True);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 返回函数保存的张量
PyObject* THPFunction_saved_tensors(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // 如果在前向过程中保存了张量
  if (self->saved_for_forward) {
    // 增加引用计数并返回保存的张量
    Py_INCREF(self->saved_for_forward);
    return self->saved_for_forward;
  } else {
    // 否则通过 unpack_saved_variables 返回保存的变量
    return unpack_saved_variables(
        self, [](const Variable& var) { return THPVariable_Wrap(var); });
  }
  END_HANDLE_TH_ERRORS
}

// 返回函数保存的变量（已弃用）
PyObject* THPFunction_saved_variables(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // 发出已弃用警告
  auto r = PyErr_WarnEx(
      PyExc_DeprecationWarning,
      "'saved_variables' is deprecated; use 'saved_tensors'",
      0);
  if (r != 0)
    throw python_error();
  // 返回保存的变量，通过 unpack_saved_variables 转换为 Python 对象
  return unpack_saved_variables(
      self, [](const Variable& var) { return THPVariable_Wrap(var); });
  END_HANDLE_TH_ERRORS
}

// 返回是否编译了自动求导追踪
PyObject* THPFunction_is_compiled_autograd_tracing(
    PyObject* self,
    PyObject* _unused) {
  HANDLE_TH_ERRORS
  // 如果编译了自动求导追踪，则返回 True
  if (((THPFunction*)self)->compiled_autograd_tracing) {
    Py_RETURN_TRUE;
  } else {
    // 否则返回 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// 返回编译的自动求导符号整数
PyObject* THPFunction_get_compiled_autograd_symints(
    PyObject* _self,
    PyObject* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  auto size = self->compiled_autograd_symints.size();
  // 创建一个元组用于保存编译的自动求导符号整数
  PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(size));
  if (!result) {
    throw python_error();
  }
  // 将编译的自动求导符号整数逐个添加到元组中
  for (const auto i : c10::irange(size)) {
    PyTuple_SET_ITEM(
        result,
        i,
        py::cast(self->compiled_autograd_symints[i]).release().ptr());
  }
  return result;
  END_HANDLE_TH_ERRORS
}

// 返回编译的自动求导反向状态
PyObject* THPFunction_get_compiled_autograd_backward_state(
    PyObject* _self,
    void* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  // 获取编译的自动求导反向状态
  PyObject* bw_state = self->compiled_autograd_backward_state;
  if (bw_state == nullptr) {
    bw_state = Py_None;
  }
  // 增加引用计数并返回反向状态
  Py_INCREF(bw_state);
  return bw_state;
  END_HANDLE_TH_ERRORS
}

// 设置编译的自动求导反向状态
int THPFunction_set_compiled_autograd_backward_state(
    PyObject* _self,
    PyObject* bw_state,
    void* _unused) {
  HANDLE_TH_ERRORS
  auto self = (THPFunction*)_self;
  // 断言自动求导反向状态为 nullptr
  TORCH_INTERNAL_ASSERT(self->compiled_autograd_backward_state == nullptr);
  // 增加引用计数并设置编译的自动求导反向状态
  Py_INCREF(bw_state);
  self->compiled_autograd_backward_state = bw_state;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 返回原始保存的张量
PyObject* THPFunction_raw_saved_tensors(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // 用户尝试在已释放缓冲区之后访问保存的变量
  TORCH_CHECK(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  // 返回保存的变量的引用
  const auto& saved_variables = self->saved_variables;
  if (saved_variables.empty())
    // 返回一个空元组
    return PyTuple_New(0);
  // 获取保存变量的数量
  size_t num_saved = saved_variables.size();
  // 创建一个 Python 元组，用于存储保存的变量
  THPObjectPtr saved(PyTuple_New(static_cast<Py_ssize_t>(num_saved)));
  // 如果创建元组失败，则返回空指针
  if (!saved) {
    return nullptr;
  }
  // 遍历保存的变量列表
  for (const auto i : c10::irange(num_saved)) {
    // 将 C++ 对象转换为 Python 对象，并指定返回值策略为引用
    py::object obj =
        py::cast(saved_variables[i], py::return_value_policy::reference);
    // 将 Python 对象添加到元组中，并释放所有权
    PyTuple_SET_ITEM(saved.get(), i, obj.release().ptr());
  }
  // 返回创建的 Python 元组，并释放所有权
  return saved.release();
  // 结束处理 Python 异常
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_next_functions(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // 锁定自身的 cdata 成员，以防止其被释放
  auto cdata = self->cdata.lock();
  // 检查 cdata 是否有效，否则抛出错误信息
  TORCH_CHECK(
      cdata,
      "Attribute 'next_functions' is invalid for this instance of _C._FunctionBase. "
      "Accessing this attribute directly on an instance of autograd.Function is a legacy "
      "access pattern that is no longer supported. For examples on how to use new-style "
      "autograd functions, see "
      "https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function ");
  // 获取自身输出数量
  const auto num_outputs = cdata->num_outputs();
  // 创建一个 Python 元组对象，用于存储结果
  THPObjectPtr result(PyTuple_New(num_outputs));
  if (!result)
    return nullptr;
  // 遍历输出数量，为每个输出创建一个元组
  for (const auto i : c10::irange(num_outputs)) {
    // 创建包含两个元素的元组，用于存储下一个函数和输入编号
    THPObjectPtr fn_tuple(PyTuple_New(2));
    if (!fn_tuple)
      return nullptr;
    // 获取当前输出的边缘信息
    const auto& edge = cdata->next_edge(i);
    // 将边缘函数转换为 Python 对象
    PyObject* fn = functionToPyObject(edge.function);
    if (!fn)
      return nullptr;
    // 设置元组的第一个元素为函数对象，第二个元素为输入编号
    PyTuple_SET_ITEM(fn_tuple.get(), 0, fn);
    PyTuple_SET_ITEM(fn_tuple.get(), 1, THPUtils_packInt64(edge.input_nr));
    // 将当前元组添加到结果元组中
    PyTuple_SET_ITEM(result.get(), i, fn_tuple.release());
  }
  // 返回结果元组对象
  return result.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFunction_metadata(THPFunction* self, void* _unused) {
  HANDLE_TH_ERRORS
  // 锁定自身的 cdata 成员，以防止其被释放
  auto cdata = self->cdata.lock();
  // 检查 cdata 是否有效，否则抛出错误信息
  // 注释提到 PyFunctions 的 grad_fn 应使用 THPCppFunction 而不是 THPFunction，但未更改以保持 BC
  TORCH_CHECK(
      cdata,
      "You attempted to access the anomaly metadata of a custom autograd function "
      "but the underlying PyNode has already been deallocated.  The most likely "
      "reason this occurred is because you assigned x.grad_fn to a local variable "
      "and then let the original variable get deallocated.  Don't do that!  If "
      "you really have no way of restructuring your code so this is the case, "
      "please file an issue reporting that you are affected by this.");
  // 获取元数据字典
  auto metadata = static_cast<PyAnomalyMetadata*>(cdata->metadata())->dict();
  // 增加元数据字典的引用计数并返回
  Py_INCREF(metadata);
  return metadata;
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);
typedef int (*setter)(PyObject*, PyObject*, void*);

namespace {

template <PyObject* THPFunction::*ptr>
PyObject* getObject(PyObject* obj, void* _unused) {
  // 将 PyObject 转换为 THPFunction 对象
  auto self = (THPFunction*)obj;
  // 获取指定成员变量的值
  PyObject* value = self->*ptr;
  // 如果值为 NULL，则返回 Python 中的 None
  if (!value) {
    Py_RETURN_NONE;
  }
  // 增加值的引用计数并返回
  Py_INCREF(value);
  return value;
}

template <PyObject* THPFunction::*ptr>
int setObject(PyObject* obj, PyObject* value, void* _unused) {
  // 将 PyObject 转换为 THPFunction 对象
  auto self = (THPFunction*)obj;
  // 如果传入的值为 Py_None，则设置成员变量为 NULL
  if (value == Py_None) {
    value = nullptr;
  }
  // 释放原来的值的引用，增加新值的引用，并设置成员变量
  Py_XDECREF((self->*ptr));
  Py_XINCREF(value);
  self->*ptr = value;
  return 0;
}
// 定义模板函数，用于获取对象的成员变量，根据给定的成员指针和转换函数进行操作
template <typename M, M THPFunction::*ptr, PyObject* (*Convert)(long)>
PyObject* getMember(PyObject* obj, void* _unused) {
  // 将传入的 PyObject 转换为 THPFunction 对象
  auto self = (THPFunction*)obj;
  // 使用成员指针 ptr 访问 self 对象中的成员变量，并通过 Convert 函数转换成 PyObject*
  return Convert(self->*ptr);
}

// 定义模板函数，用于获取 autograd::Node 对象的成员变量，类似于 getMember，但是操作 autograd::Node 对象
template <typename M, M autograd::Node::*ptr, PyObject* (*Convert)(long)>
PyObject* getImplMember(PyObject* obj, void* _unused) {
  // 将传入的 PyObject 转换为 THPFunction 对象
  auto self = (THPFunction*)obj;
  // 使用成员指针 ptr 访问 self 对象中 autograd::Node 对象的成员变量，并通过 Convert 函数转换成 PyObject*
  return Convert(self->cdata.*ptr);
}

// 返回一个 Python 对象，表示对象需要梯度计算
PyObject* getRequiresGrad(PyObject* obj, void* _unused) {
  // 使用 Py_RETURN_TRUE 宏返回 Python 的 True 对象，表示需要梯度计算
  Py_RETURN_TRUE;
}

} // namespace

// 定义静态数组 PyGetSetDef，用于描述 THPFunction 的属性，每个条目包含属性名和相应的 getter 和 setter 函数
static struct PyGetSetDef THPFunction_properties[] = {
    {"saved_tensors",
     (getter)THPFunction_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"saved_variables",
     (getter)THPFunction_saved_variables,
     nullptr,
     nullptr,
     nullptr},
    {"_raw_saved_tensors",
     (getter)THPFunction_raw_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"next_functions",
     (getter)THPFunction_next_functions,
     nullptr,
     nullptr,
     nullptr},
    {"to_save",
     &getObject<&THPFunction::to_save>,
     &setObject<&THPFunction::to_save>,
     nullptr,
     nullptr},
    {"non_differentiable",
     &getObject<&THPFunction::non_differentiable>,
     &setObject<&THPFunction::non_differentiable>,
     nullptr,
     nullptr},
    {"dirty_tensors",
     &getObject<&THPFunction::dirty_tensors>,
     &setObject<&THPFunction::dirty_tensors>,
     nullptr,
     nullptr},
    {"saved_for_forward",
     &getObject<&THPFunction::saved_for_forward>,
     &setObject<&THPFunction::saved_for_forward>,
     nullptr,
     nullptr},
    {"needs_input_grad",
     &getObject<&THPFunction::needs_input_grad>,
     &setObject<&THPFunction::needs_input_grad>,
     nullptr,
     nullptr},
    {"requires_grad", getRequiresGrad, nullptr, nullptr, nullptr},
    {"metadata", (getter)THPFunction_metadata, nullptr, nullptr, nullptr},
    {"materialize_grads",
     nullptr,
     (setter)THPFunction_set_materialize_grads,
     nullptr,
     nullptr},
    {"_materialize_non_diff_grads",
     (getter)THPFunction_get_materialize_non_diff_grads,
     (setter)THPFunction_set_materialize_non_diff_grads,
     nullptr,
     nullptr},
    {"_compiled_autograd_backward_state",
     (getter)THPFunction_get_compiled_autograd_backward_state,
     (setter)THPFunction_set_compiled_autograd_backward_state,
     nullptr,
     nullptr},
    {nullptr}};

// 定义静态数组 PyMethodDef，用于描述 THPFunction 的方法，每个条目包含方法名、函数指针和调用方式
static struct PyMethodDef THPFunction_methods[] = {
    {(char*)"name", THPFunction_name, METH_NOARGS, nullptr},
    {(char*)"_sequence_nr", THPFunction_sequence_nr, METH_NOARGS, nullptr},
    {(char*)"_set_sequence_nr", THPFunction_set_sequence_nr, METH_O, nullptr},
    // 更多方法定义可以继续添加
};
    // 创建一个静态的方法映射数组，每个条目包含方法名字符串、对应的C函数指针、方法标志位和空指针
    {
        (char*)"maybe_clear_saved_tensors",  // 方法名字符串 "maybe_clear_saved_tensors"
        THPFunction_maybe_clear_saved_tensors,  // 对应的C函数指针 THPFunction_maybe_clear_saved_tensors
        METH_NOARGS,  // 方法标志位，表示此方法不接受任何参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"apply",  // 方法名字符串 "apply"
        THPFunction_apply,  // 对应的C函数指针 THPFunction_apply
        METH_CLASS | METH_VARARGS,  // 方法标志位，表示此方法可以作用于类和接受任意数量参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"_register_hook_dict",  // 方法名字符串 "_register_hook_dict"
        THPFunction__register_hook_dict,  // 对应的C函数指针 THPFunction__register_hook_dict
        METH_O,  // 方法标志位，表示此方法接受一个Python对象作为参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"register_hook",  // 方法名字符串 "register_hook"
        THPFunction_register_hook,  // 对应的C函数指针 THPFunction_register_hook
        METH_O,  // 方法标志位，表示此方法接受一个Python对象作为参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"register_prehook",  // 方法名字符串 "register_prehook"
        THPFunction_register_prehook,  // 对应的C函数指针 THPFunction_register_prehook
        METH_O,  // 方法标志位，表示此方法接受一个Python对象作为参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"_is_compiled_autograd_tracing",  // 方法名字符串 "_is_compiled_autograd_tracing"
        THPFunction_is_compiled_autograd_tracing,  // 对应的C函数指针 THPFunction_is_compiled_autograd_tracing
        METH_NOARGS,  // 方法标志位，表示此方法不接受任何参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {
        (char*)"_get_compiled_autograd_symints",  // 方法名字符串 "_get_compiled_autograd_symints"
        THPFunction_get_compiled_autograd_symints,  // 对应的C函数指针 THPFunction_get_compiled_autograd_symints
        METH_NOARGS,  // 方法标志位，表示此方法不接受任何参数
        nullptr  // 空指针，用于结束数组的标记
    },
    {nullptr}  // 最后一个条目，空指针，用于结束数组的标记
    };
PyTypeObject THPFunctionType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._FunctionBase", /* tp_name */
    sizeof(THPFunction), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPFunction_dealloc, /* tp_dealloc */
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
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)THPFunction_traverse, /* tp_traverse */
    (inquiry)THPFunction_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPFunction_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPFunction_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPFunction_new /* tp_new */
};

bool THPFunction_initModule(PyObject* module) {
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);  // 增加 THPFunctionType 的引用计数，避免被意外销毁
  PyModule_AddObject(module, "_FunctionBase", (PyObject*)&THPFunctionType);  // 将 THPFunctionType 添加到指定的 Python 模块中
  return true;  // 初始化模块成功
}
```