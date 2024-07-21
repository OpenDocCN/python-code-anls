# `.\pytorch\tools\autograd\templates\python_variable_methods.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于仅使用方法操作符

// 包含 Python.h 头文件，提供对 Python C API 的访问
#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// 取消定义 copysign 宏，以便 at::copysign 在 MSVC 中按预期工作
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

// 包含 Torch 的动态类型、异常处理、尺寸、变量类型等头文件
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/jit/frontend/tracer.h"

// 如果使用 CUDA，包含 CUDA 相关的事件头文件
#ifdef USE_CUDA
#include "torch/csrc/cuda/Event.h"
#endif

// 包含 Torch 的设备懒初始化工具
#include "torch/csrc/utils/device_lazy_init.h"

// 包含 Torch 的 NumPy 兼容性接口
#include <torch/csrc/utils/numpy_stub.h>

// 包含 Torch 的对象智能指针、Python C 函数助手等工具
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_tuples.h"

// 包含 Torch 的张量操作工具和类型定义
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"

// 包含 Torch 的结构序列工具
#include "torch/csrc/utils/structseq.h"

// 包含 Torch 自动求导生成的 Python 返回类型定义
#include "torch/csrc/autograd/generated/python_return_types.h"

// 包含 ATen 张量和功能接口
#include <ATen/core/Tensor.h>

// 包含 ATen 的 Tensor 对象管理工具和流对象
#include <ATen/FuncTorchTLS.h>
#include "c10/util/Optional.h"
#include "c10/core/Stream.h"

// 包含标准异常处理头文件
#include <stdexcept>

// 如果未定义每个操作符单独的头文件包含，则统一包含 ATen 的基础功能函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#include <ATen/ops/_local_scalar_dense.h>
#endif

// 使用 at 命名空间中定义的关键字和类型
using at::DeviceGuard;
using at::device_of;
using at::OptionalDeviceGuard;
using at::Backend;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using c10::Stream;
using namespace torch::autograd::utils;

// Torch 自动求导命名空间
namespace torch::autograd {

// 静态函数，用于检查 Python 对象是否为视图
static PyObject * THPVariable__is_view(PyObject *self, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 如果对象有 Torch 函数支持，则调用 Torch 函数处理
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "_is_view", args);
  }
  // 否则，解包 Python 对象为 THPVariable，并检查是否为视图
  auto& self_ = THPVariable_Unpack(self);
  if (self_.is_view()) {
    // 返回 Python 中的 True
    Py_RETURN_TRUE;
  } else {
    // 返回 Python 中的 False
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// Python 对象上实现的函数，因为原生函数中不支持一级函数，详见 ATen/native/README.md
static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  // 如果对象有 Torch 函数支持，则构建参数元组并调用 Torch 函数处理
  if (check_has_torch_function(self)) {
    auto args = py::make_tuple(py::handle(arg));
    return handle_torch_function(self, "apply_", args.ptr());
  }
  // 否则，解包 Python 对象为 THPVariable，并检查是否需要梯度
  auto& self_ = THPVariable_Unpack(self);
  if (self_.requires_grad()) {
    // 继续处理梯度计算
    // (此处代码未完，应继续阅读下文以了解其完整作用)
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");

抛出一个 `std::runtime_error` 异常，指示无法在需要梯度的变量上调用 `apply_()` 方法，建议使用 `var.detach().apply_()` 替代。


  }

结束当前的函数或代码块的条件语句。


  return THPVariable_Wrap(torch::utils::apply_(self_, arg));

返回一个经过 `torch::utils::apply_(self_, arg)` 处理后的 `THPVariable_Wrap` 对象。


  END_HANDLE_TH_ERRORS

结束对 Torch 引发的错误进行处理的宏定义。
// 处理 THPVariable_size 函数，用于获取张量的尺寸信息
static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持两种参数格式
  static PythonArgParser parser({
    "size(int64_t? dim=None)",
    "size(Dimname dim)",
  });
  // 解包 self 引用，获取 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析函数参数
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 根据解析结果进行相应操作
  if (r.idx == 0) {
    // 如果未提供维度参数，则返回整体尺寸信息
    if (!r.toInt64Optional(0).has_value()) {
      return THPSize_NewFromSymSizes(self_);
    }
    // 如果正在进行追踪，则调用追踪器获取尺寸
    if (jit::tracer::isTracing()) {
      // 如果张量包含符号整数，则会引发错误
      return wrap(jit::tracer::getSizeOf(self_, r.toInt64(0)));
    } else {
      // 否则，返回指定维度的尺寸
      return torch::toPyObject(self_.sym_size(r.toInt64(0)));
    }
  } else if (r.idx == 1) {
    // 如果使用了维度名称，则返回对应名称的尺寸
    if (jit::tracer::isTracing()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: Named tensors w/ JIT");
    }
    return wrap(self_.size(r.dimname(0)));
  }
  // 默认情况下返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 处理 THPVariable_stride 函数，用于获取张量的步长信息
static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持两种参数格式
  static PythonArgParser parser({
    "stride(int64_t? dim=None)",
    "stride(Dimname dim)",
  });
  // 解包 self 引用，获取 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析函数参数
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 根据解析结果进行相应操作
  if (r.idx == 0) {
    // 如果提供了维度参数，则返回该维度的步长信息
    if (r.toInt64Optional(0).has_value()) {
      return torch::toPyObject(self_.sym_stride(r.toInt64(0)));
    }
    // 否则，返回所有维度的步长信息
    // 注意：在 ATen 中称为 strides
    at::SymIntArrayRef strides = self_.sym_strides();
    // 由于 IntArrayRef 同时映射到 torch.Size 和元组(tuple)，因此需要特殊处理
    // TODO: 考虑将此部分代码提取为函数
    THPObjectPtr tuple(PyTuple_New(strides.size()));
    if (!tuple) throw python_error();
    for (size_t i = 0; i != strides.size(); i++) {
      PyObject* s = torch::toPyObject(strides[i]);
      if (!s) throw python_error();
      PyTuple_SET_ITEM(tuple.get(), i, s);
    }
    return tuple.release();
  } else if (r.idx == 1) {
    // 如果使用了维度名称，则返回对应名称的步长信息
    return wrap(self_.stride(r.dimname(0)));
  }
  // 默认情况下返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 在 Python 对象上实现，以避免调度开销
// 获取张量所在的设备信息
static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "get_device", args, nullptr);
  }
  // 解包 self 引用，获取 THPVariable 对象
  auto& self = THPVariable_Unpack(self_);
  // 返回张量所在的设备信息
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

// 检查张量是否具有名称
static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "has_names", args);
  }
  // 解包 self 引用，获取 THPVariable 对象
  auto& self = THPVariable_Unpack(self_);
  // 返回张量是否具有名称的布尔值
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}
// 实现在 Python 对象上以避免调度开销
static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 检查是否存在 torch 函数，若存在则调用该函数处理
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "data_ptr", args);
  }
  // 解包 Python 对象并获取其对应的 C++ Tensor 对象
  auto& self = THPVariable_Unpack(self_);
  // 调用 Tensor 对象的 data_ptr 方法并包装为 Python 对象返回
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

// 实现在 Python 对象上以避免调度开销
static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 检查是否存在 torch 函数，若存在则调用该函数处理
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "storage_offset");
  }
  // 解包 Python 对象并获取其对应的 C++ Tensor 对象
  auto& self = THPVariable_Unpack(self_);
  // 调用 Tensor 对象的 sym_storage_offset 方法并将其转换为 Python 对象返回
  return py::cast(self.sym_storage_offset()).release().ptr();
  END_HANDLE_TH_ERRORS
}

// 实现在 Python 对象上以避免调度开销
static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   // 检查是否存在 torch 函数，若存在则调用该函数处理
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "dim", args);
   }
   // 解包 Python 对象并获取其对应的 C++ Tensor 对象
   auto& self_ = THPVariable_Unpack(self);
   // 调用 Tensor 对象的 dim 方法并包装为 Python 整数对象返回
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

// 实现在 Python 对象上以避免调度开销
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   // 检查是否存在 torch 函数，若存在则调用该函数处理
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "numel", args);
   }
   // 解包 Python 对象并获取其对应的 C++ Tensor 对象
   auto& self_ = THPVariable_Unpack(self);
   // 如果处于追踪状态，则返回 Tensor 对象的 numel 数量
   if (jit::tracer::isTracing()) {
     return wrap(jit::tracer::getNumelOf(self_));
   } else {
     // 否则调用 Tensor 对象的 sym_numel 方法，并将其转换为 Python 对象返回
     return py::cast(self_.sym_numel()).release().ptr();
   }
   END_HANDLE_TH_ERRORS
}

// 分发函数，确保 Tensor 对象连续性，支持指定内存格式
static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  pybind11::gil_scoped_release no_gil;  // 释放 GIL，以避免线程锁
  OptionalDeviceGuard device_guard(device_of(self));  // 设置当前设备
  // 调用 Tensor 对象的 contiguous 方法以确保连续性，并返回结果
  return self.contiguous(memory_format);
}

// 实现在 Python 对象上以避免调度开销
static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持指定内存格式的 contiguous 方法
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  // 解析参数
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果存在 torch 函数，则调用处理该函数的逻辑
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 解包 Python 对象并获取其对应的 C++ Tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  auto memory_format = r.memoryformat(0);

  // 如果 Tensor 对象已经是指定内存格式的连续状态，则直接返回
  if (self_.is_contiguous(memory_format)) {
    // 注意：此逻辑与 VariableType.cpp 中的重复。由于我们需要在追踪中记录调用 contiguous() 的信息，
    // 因此我们需要手动记录此信息。
    // 返回空值，表示没有修改状态
    // 检查当前是否处于跟踪模式，若是则进行以下操作
    if (jit::tracer::isTracing()) {
      // 获取跟踪状态对象
      auto tracer_state = jit::tracer::getTracingState();
      // 创建操作符节点，操作符名称为"aten::contiguous"，没有输出
      auto op_name = c10::Symbol::fromQualString("aten::contiguous");
      auto node = tracer_state->createNode(op_name, /*num_outputs=*/0);
      // 记录源码位置到跟踪节点
      jit::tracer::recordSourceLocation(node);
      // 添加输入参数到跟踪节点，分别是"self"和memory_format
      jit::tracer::addInputs(node, "self", self_);
      jit::tracer::addInputs(node, "memory_format", memory_format);
      // 将节点插入跟踪状态中
      tracer_state->insertNode(node);
      // 将self_标记为操作节点的输出
      jit::tracer::addOutput(node, self_);
    }
    // 增加Python对象self的引用计数
    Py_INCREF(self);
    // 返回Python对象self
    return self;
  }
  // 处理异常，调用dispatch_contiguous函数返回THPVariable对象包装
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
  // 结束处理TH_ERRORS
  END_HANDLE_TH_ERRORS
}

// 静态函数，用于分派调用 self 的 copy_ 方法到 C++ 线程中执行，可能不等待
static Tensor dispatch_copy_(const Tensor & self, const Tensor & other, bool non_blocking) {
  // 释放全局解释器锁，允许其他线程运行
  pybind11::gil_scoped_release no_gil;
  // 设备守护，确保操作发生在 self 的设备上
  OptionalDeviceGuard device_guard(device_of(self));
  // 调用 self 的 copy_ 方法进行复制操作
  return self.copy_(other, non_blocking);
}

// THPVariable 类的 copy_ 方法的 Python 包装函数
static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 静态参数解析器，定义支持的参数类型
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)", // 复制另一张张量到当前张量，支持非阻塞模式
    "copy_(Tensor other, bool async=False)|deprecated" // 异步复制模式（已废弃）
  });
  // 解包 Python 对象 self
  auto& self_ = THPVariable_Unpack(self);
  // 解析输入的参数并返回解析结果
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果解析结果包含 torch 函数的调用，则处理并返回
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 否则，调用 dispatch_copy_ 执行复制操作，并包装成 Python 对象返回
  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

// 模板函数，将张量 self 转换为 Python 标量 T 类型
template<typename T>
static T dispatch_to(const Tensor & self) {
  // 释放全局解释器锁，允许其他线程运行
  pybind11::gil_scoped_release no_gil;
  // 设备守护，确保操作发生在 self 的设备上
  OptionalDeviceGuard device_guard(device_of(self));
  // 检查张量是否只有一个元素，否则抛出错误
  TORCH_CHECK_VALUE(self.sym_numel() == 1, "only one element tensors can be converted to Python scalars");
  // 调用 item 方法将张量转换为 T 类型的标量并返回
  return self.template item<T>();
}

// THPVariable 类的 float 标量转换函数
static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 如果存在 torch 函数的重载，则处理并返回
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__float__", args);
  }
  // 发出警告，将张量转换为 Python 浮点数
  jit::tracer::warn("Converting a tensor to a Python float", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 解包 Python 对象 self
  auto& self_ = THPVariable_Unpack(self);
  // 调用 dispatch_to<double> 将张量转换为双精度浮点数并返回
  return wrap(dispatch_to<double>(self_));
  END_HANDLE_TH_ERRORS
}

// THPVariable 类的 complex 标量转换函数
static PyObject * THPVariable_complex_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 如果存在 torch 函数的重载，则处理并返回
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__complex__", args);
  }
  // 发出警告，将张量转换为 Python 复数
  jit::tracer::warn("Converting a tensor to a Python complex", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 解包 Python 对象 self
  auto& self_ = THPVariable_Unpack(self);
  // 调用 dispatch_to<c10::complex<double>> 将张量转换为双精度复数并返回
  return wrap(dispatch_to<c10::complex<double>>(self_));
  END_HANDLE_TH_ERRORS
}

// THPVariable 类的 integral 标量转换函数
static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 如果存在 torch 函数的重载，则处理并返回
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__int__", args);
  }
  // 发出警告，将张量转换为 Python 整数
  jit::tracer::warn("Converting a tensor to a Python integer", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 解包 Python 对象 self
  auto& self_ = THPVariable_Unpack(self);
  // 如果张量是浮点类型，则避免 ATen 溢出检查，直接返回双精度浮点数转换为整数后的包装对象
  if (isFloatingType(self_.scalar_type())) {
    return THPUtils_packDoubleAsInt(dispatch_to<double>(self_));
  } else {
    // 否则，调用 dispatch_to<int64_t> 将张量转换为 64 位整数并返回包装对象
    return wrap(dispatch_to<int64_t>(self_));
  }
  END_HANDLE_TH_ERRORS
}

// Python 的 __index__ 函数的实现，类似于 __int__，但在作为切片时调用
static PyObject * THPVariable_index_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 如果存在 torch 函数的重载，则处理并返回
  if (check_has_torch_function(self)) {
    // 处理 torch 函数的调用
    return handle_torch_function(self, "__index__", args);
  }
  // 其他情况，继续处理，但该部分代码未完整给出
    return handle_torch_function(self, "__index__", args);
  }


// 调用 Torch 函数处理 "__index__" 操作，并返回结果
auto& self_ = THPVariable_Unpack(self);
// 获取 self 的引用，解包为 Torch 张量对象

// TODO: change the condition to `self_.dim() != 0` once we expose scalars
// in PyTorch.
// TODO: 一旦在 PyTorch 中公开标量（scalar），将条件更改为 `self_.dim() != 0`

if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true) || self_.sym_numel() != 1) {
// 如果 self_ 不是整数类型（包括布尔型），或者元素数量不为1，则抛出类型错误异常
  throw TypeError("only integer tensors of a single element can be converted to an index");
}

return wrap(dispatch_to<int64_t>(self_));
// 调用 dispatch_to 函数，将 self_ 转换为 int64_t 类型，并用 wrap 函数封装返回值
END_HANDLE_TH_ERRORS
// 结束 Torch 错误处理
}

// 定义静态函数，用于处理按位取反操作
static Tensor dispatch_invert(const Tensor & self) {
  // 释放全局解释器锁，允许其他线程运行
  pybind11::gil_scoped_release no_gil;
  // 临时设备保护，保证操作在正确的设备上进行
  OptionalDeviceGuard device_guard(device_of(self));
  // 调用张量的按位取反方法，并返回结果
  return self.bitwise_not();
}

// 定义 THPVariable 类的 __invert__ 方法的 Python C-API 实现
static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 如果对象有 torch function 覆盖，调用相应的 torch function
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__invert__", args);
  }
  // 拆包 Python 对象为 C++ 对象
  auto& self_ = THPVariable_Unpack(self);
  // 如果张量不是整数或布尔类型，抛出类型错误异常
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true)) {
    throw TypeError("~ (operator.invert) is only implemented on integer and Boolean-type tensors");
  }
  // 调用 dispatch_invert 处理按位取反操作，并包装成 Python 对象返回
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

// 根据目标设备、非阻塞标志、拷贝标志和内存格式，将张量转换到目标设备
static Tensor dispatch_to(const Tensor & self, Device device, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // 在跟踪期间记录 aten::to 操作的位置，以确保所有张量选项都被完全指定
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

// 根据非阻塞标志、拷贝标志和内存格式，将张量转换到目标设备
static Tensor dispatch_to(const Tensor & self, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // 将张量转换到指定内存格式，并返回结果
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

// 根据数据类型、非阻塞标志、拷贝标志和内存格式，将张量转换到目标设备
static Tensor dispatch_to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: 可能将此调用更新为 TensorOptions 版本
  // 根据指定选项将张量转换到目标设备，并返回结果
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

// 根据目标设备、数据类型、非阻塞标志、拷贝标志和内存格式，将张量转换到目标设备
static Tensor dispatch_to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: 可能将此调用更新为 TensorOptions 版本
  // 根据指定选项将张量转换到目标设备，并返回结果
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

// 定义 THPVariable 类的 cpu 方法的 Python C-API 实现
static PyObject * THPVariable_cpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
   HANDLE_TH_ERRORS
   // 解析参数，支持 MemoryFormat 可选参数
   static PythonArgParser parser({
     "cpu(*, MemoryFormat? memory_format=None)"
   });
   auto& self_ = THPVariable_Unpack(self);
   // 解析参数并检查是否有 torch function 覆盖
   ParsedArgs<1> parsed_args;
   auto r = parser.parse(self, args, kwargs, parsed_args);

   if(r.has_torch_function()){
    // 如果有 torch function 覆盖，调用相应的 torch function
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
    }

   auto opt_memory_format = r.memoryformatOptional(0);
   // 从返回值 r 中获取内存格式（可选），索引为 0
   return THPVariable_Wrap(dispatch_to(self_, at::Device(at::DeviceType::CPU), false, false, opt_memory_format));
   // 调用 dispatch_to 函数，将 self_ 对象分派到 CPU 设备上执行，
   // 参数依次为：目标设备（CPU），不进行异步处理，不保留计算图，使用 opt_memory_format 指定的内存格式
   END_HANDLE_TH_ERRORS
// 静态函数，接收一个 Tensor 参数并返回其非零元素的索引 Tensor
static Tensor dispatch_nonzero(const Tensor & self) {
  // 释放全局解释器锁，允许其他线程执行
  pybind11::gil_scoped_release no_gil;
  // 可选设备保护，确保在函数执行期间设备不变
  OptionalDeviceGuard device_guard(device_of(self));
  // 调用 Tensor 对象的非零索引方法并返回结果
  return self.nonzero();
}

// 静态函数，接收一个 Tensor 参数并返回其非零元素的索引列表（用于 NumPy）
static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  // 释放全局解释器锁，允许其他线程执行
  pybind11::gil_scoped_release no_gil;
  // 可选设备保护，确保在函数执行期间设备不变
  OptionalDeviceGuard device_guard(device_of(self));
  // 调用 Tensor 对象的 NumPy 非零索引方法并返回结果
  return self.nonzero_numpy();
}

// THPVariable_nonzero 函数实现，接收 Python 对象的参数并返回 PyObject 指针
static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 静态 PythonArgParser 对象定义和初始化，解析输入的 Python 参数
  static PythonArgParser parser({
    "nonzero()",
    "nonzero(*, bool as_tuple)",
  });
  // 解包 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析输入参数，获取解析结果
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数的重载
  if(r.has_torch_function()){
    // 处理 Torch 函数的重载调用
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 根据参数选择调用 dispatch_nonzero 或 dispatch_nonzero_numpy
  if (r.idx == 0 || (r.idx == 1 && !r.toBool(0))) {
    // 调用 dispatch_nonzero 函数并包装返回结果为 PyObject 指针
    return wrap(dispatch_nonzero(self_));
  } else {
    // 调用 dispatch_nonzero_numpy 函数并包装返回结果为 PyObject 指针
    return wrap(dispatch_nonzero_numpy(self_));
  }
  END_HANDLE_TH_ERRORS
}

// THPVariable_cuda 函数实现，接收 Python 对象的参数并返回 PyObject 指针
static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 静态 PythonArgParser 对象定义和初始化，解析输入的 Python 参数
  static PythonArgParser parser({
    "cuda(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "cuda(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  // 解包 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析输入参数，获取解析结果
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数的重载
  if(r.has_torch_function()){
    // 处理 Torch 函数的重载调用
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 解析设备参数，若未指定则使用 CUDA 设备
  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  // 检查设备类型，必须为 CUDA 设备
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  // 初始化 CUDA 设备
  torch::utils::device_lazy_init(at::kCUDA);
  // 调用 dispatch_to 函数，处理 Tensor 对象和设备参数，并包装返回结果为 PyObject 指针
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// THPVariable_xpu 函数实现，接收 Python 对象的参数并返回 PyObject 指针
static PyObject * THPVariable_xpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 静态 PythonArgParser 对象定义和初始化，解析输入的 Python 参数
  static PythonArgParser parser({
    "xpu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "xpu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  // 解包 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析输入参数，获取解析结果
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数的重载
  if (r.has_torch_function()) {
    // 处理 Torch 函数的重载调用
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 解析设备参数，若未指定则使用 XPU 设备
  auto device = r.isNone(0) ? at::Device(at::DeviceType::XPU) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  // 检查设备类型，必须为 XPU 设备
  TORCH_CHECK(device.is_xpu(), "Invalid device, must be xpu device");
  // 初始化 XPU 设备
  torch::utils::device_lazy_init(at::kXPU);
  // 调用 dispatch_to 函数，处理 Tensor 对象和设备参数，并包装返回结果为 PyObject 指针
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}
// 定义 THPVariable_ipu 函数，接受 Python 的 self, args, kwargs 作为参数
static PyObject * THPVariable_ipu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理 C++ 异常
  HANDLE_TH_ERRORS
  // 定义 PythonArgParser 对象，解析两种不同的参数格式
  static PythonArgParser parser({
    "ipu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "ipu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  // 解包 Python 的 self 对象
  auto& self_ = THPVariable_Unpack(self);
  // 定义 ParsedArgs<3> 对象，并解析参数
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果参数 r 包含 torch function，则调用处理 torch function 的函数并返回结果
  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取设备信息，若未提供则默认为 IPU 设备
  auto device = r.isNone(0) ? at::Device(at::DeviceType::IPU) : r.device(0);
  // 获取内存格式信息，可选参数
  auto opt_memory_format = r.memoryformatOptional(2);
  // 检查设备类型是否为 IPU，否则抛出错误
  TORCH_CHECK(device.is_ipu(), "Invalid device, must be ipu device");
  // 调用 dispatch_to 函数，根据参数调度对应操作，并返回结果
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  // 处理 C++ 异常结束
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_to_type 函数，接受 Python 的 self, scalarType, optional_memory_format 作为参数
static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType, std::optional<c10::MemoryFormat> optional_memory_format) {
  // 处理 C++ 异常
  HANDLE_TH_ERRORS
  // 解包 Python 的 self 对象
  auto& self_ = THPVariable_Unpack(self);
  // 调用 dispatch_to 函数，根据参数调度对应操作，并返回结果
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false, optional_memory_format));
  // 处理 C++ 异常结束
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_byte 函数，接受 Python 的 self, args, kwargs 作为参数
static PyObject * THPVariable_byte(PyObject* self, PyObject* args, PyObject* kwargs)  {
  // 处理 C++ 异常
  HANDLE_TH_ERRORS
  // 定义 PythonArgParser 对象，解析参数格式
  static PythonArgParser parser({
    "byte(*, MemoryFormat? memory_format=None)"
  });
  // 定义 ParsedArgs<1> 对象，并解析参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果参数 r 包含 torch function，则调用处理 torch function 的函数并返回结果
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取内存格式信息，可选参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 调用 THPVariable_to_type 函数，将 self 转换为 Byte 类型，并返回结果
  return THPVariable_to_type(self, ScalarType::Byte, opt_memory_format);
  // 处理 C++ 异常结束
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_char 函数，接受 Python 的 self, args, kwargs 作为参数
static PyObject * THPVariable_char(PyObject* self, PyObject* args, PyObject* kwargs)  {
  // 处理 C++ 异常
  HANDLE_TH_ERRORS
  // 定义 PythonArgParser 对象，解析参数格式
  static PythonArgParser parser({
    "char(*, MemoryFormat? memory_format=None)"
  });
  // 定义 ParsedArgs<1> 对象，并解析参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果参数 r 包含 torch function，则调用处理 torch function 的函数并返回结果
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取内存格式信息，可选参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 调用 THPVariable_to_type 函数，将 self 转换为 Char 类型，并返回结果
  return THPVariable_to_type(self, ScalarType::Char, opt_memory_format);
  // 处理 C++ 异常结束
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_double 函数，接受 Python 的 self, args, kwargs 作为参数
static PyObject * THPVariable_double(PyObject* self, PyObject* args, PyObject* kwargs) {
  // 处理 C++ 异常
  HANDLE_TH_ERRORS
  // 定义 PythonArgParser 对象，解析参数格式
  static PythonArgParser parser({
    "double(*, MemoryFormat? memory_format=None)"
  });
  // 定义 ParsedArgs<1> 对象，并解析参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果参数 r 包含 torch function，则调用处理 torch function 的函数并返回结果
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取内存格式信息，可选参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 调用 THPVariable_to_type 函数，将 self 转换为 Double 类型，并返回结果
  return THPVariable_to_type(self, ScalarType::Double, opt_memory_format);
  // 处理 C++ 异常结束
  END_HANDLE_TH_ERRORS
}
```cpp`
# 定义一个名为 THPVariable_float 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_float(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "float(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 Float 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::Float, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 定义一个名为 THPVariable_cdouble 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_cdouble(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "cdouble(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 ComplexDouble 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::ComplexDouble, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 定义一个名为 THPVariable_cfloat 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_cfloat(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "cfloat(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 ComplexFloat 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::ComplexFloat, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 定义一个名为 THPVariable_half 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_half(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "half(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 Half 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::Half, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 定义一个名为 THPVariable_int 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_int(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "int(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 Int 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::Int, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}

// 定义一个名为 THPVariable_long 的函数，接受三个参数：self, args, kwargs
static PyObject * THPVariable_long(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS  // 处理 Torch 错误的宏

  // 静态的 PythonArgParser 对象，解析参数列表
  static PythonArgParser parser({
    "long(*, MemoryFormat? memory_format=None)"  // 接受一个可选的 MemoryFormat 参数
  });

  ParsedArgs<1> parsed_args;  // 解析后的参数对象
  auto r = parser.parse(self, args, kwargs, parsed_args);  // 解析参数并存储在 r 中

  // 如果存在 Torch 函数，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);  // 获取可选的内存格式参数
  // 将 self 转换为 Long 类型，并根据可选的内存格式参数进行转换
  return THPVariable_to_type(self, ScalarType::Long, opt_memory_format);

  END_HANDLE_TH_ERRORS  // 结束 Torch 错误处理
}
  // 定义一个字符串字面量，描述一个函数签名 "long(*, MemoryFormat? memory_format=None)"
  "long(*, MemoryFormat? memory_format=None)"
});

// 创建一个 ParsedArgs 类型的对象 parsed_args，并初始化为空
ParsedArgs<1> parsed_args;

// 调用 parser 对象的 parse 方法，解析 self、args、kwargs，将结果存储在 parsed_args 中
auto r = parser.parse(self, args, kwargs, parsed_args);

// 如果返回的 r 对象具有 torch function，则调用 handle_torch_function 处理函数重载
if (r.has_torch_function()) {
  return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
}

// 获取第一个参数的内存格式，如果不存在则为可选的内存格式对象 opt_memory_format
auto opt_memory_format = r.memoryformatOptional(0);

// 将 self 转换为类型 ScalarType::Long 的 THPVariable，并使用可选的内存格式进行转换
return THPVariable_to_type(self, ScalarType::Long, opt_memory_format);
END_HANDLE_TH_ERRORS
static PyObject * THPVariable_short(PyObject* self, PyObject* args, PyObject* kwargs) {
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 定义静态的 Python 参数解析器，指定支持的参数格式
  static PythonArgParser parser({
    "short(*, MemoryFormat? memory_format=None)"
  });
  // 解析传入的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if(r.has_torch_function()){
    // 处理 Torch 函数重载
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取可选的内存格式参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 将 self 转换为 Short 类型的 THPVariable 对象，并返回
  return THPVariable_to_type(self, ScalarType::Short, opt_memory_format);
  // 处理 Torch 异常和错误
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool(PyObject* self, PyObject* args, PyObject* kwargs) {
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 定义静态的 Python 参数解析器，指定支持的参数格式
  static PythonArgParser parser({
    "bool(*, MemoryFormat? memory_format=None)"
  });
  // 解析传入的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if(r.has_torch_function()){
    // 处理 Torch 函数重载
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取可选的内存格式参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 将 self 转换为 Bool 类型的 THPVariable 对象，并返回
  return THPVariable_to_type(self, ScalarType::Bool, opt_memory_format);
  // 处理 Torch 异常和错误
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bfloat16(PyObject* self, PyObject* args, PyObject* kwargs) {
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 定义静态的 Python 参数解析器，指定支持的参数格式
  static PythonArgParser parser({
    "bfloat16(*, MemoryFormat? memory_format=None)"
  });
  // 解析传入的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if(r.has_torch_function()){
    // 处理 Torch 函数重载
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取可选的内存格式参数
  auto opt_memory_format = r.memoryformatOptional(0);
  // 将 self 转换为 BFloat16 类型的 THPVariable 对象，并返回
  return THPVariable_to_type(self, ScalarType::BFloat16, opt_memory_format);
  // 处理 Torch 异常和错误
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 如果检测到 Torch 函数重载，则处理 Torch 函数调用
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "element_size", args);
  }
  // 解包 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回封装后的 self_ 的元素大小
  return THPUtils_packInt64(self_.element_size());
  // 处理 Torch 异常和错误
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc PyObjects not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_numpy(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 定义静态的 Python 参数解析器，指定支持的参数格式
  static PythonArgParser parser({
    "numpy(*, bool force=False)"
  });
  // 解包 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);
  // 解析传入的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载
  if (r.has_torch_function()) {
    // 处理 Torch 函数重载
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 发出警告，将 Tensor 转换为 NumPy 数组
  jit::tracer::warn("Converting a tensor to a NumPy array", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 将 self_ 转换为 NumPy 数组并返回
  return torch::utils::tensor_to_numpy(self_, r.toBool(0));
  // 处理 Torch 异常和错误
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理异常和错误
  HANDLE_TH_ERRORS
  // 定义静态的 Python 参数解析器，指定支持的参数格式
  static PythonArgParser parser({
  "requires_grad_(bool requires_grad=True)",
  });
  // 解包传入的 Python 对象 self，获取其对应的 C++ Tensor 对象 self_
  auto& self_ = THPVariable_Unpack(self);
  // 解析函数参数，允许最多1个参数，并将结果存储在 parsed_args 中
  ParsedArgs<1> parsed_args;
  // 调用解析器来解析传入的参数，并返回解析结果
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果返回结果 r 有 torch function，则调用处理 torch function 的函数
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 临时的 Hack，用于改进 functorch 的用户体验
  // 获取 functorch 的 TLS（线程本地存储），如果存在则执行相应的检查函数
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    functorch_tls->checkSupportsInplaceRequiresGrad();
  }

  // 从解析结果 r 中获取 requires_grad 的布尔值
  auto requires_grad = r.toBool(0);
  // 如果 self_ 不是叶子节点且 requires_grad 为 false，则抛出异常
  if (!self_.is_leaf() && !requires_grad) {
    throw std::runtime_error(autograd::utils::requires_grad_leaf_error(requires_grad));
  }
  // 如果 requires_grad 为 true 且 self_ 的数据类型不可微分，则抛出异常
  if (requires_grad && ! isDifferentiableType(at::typeMetaToScalarType(self_.dtype()))) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  // 设置 self_ 的 requires_grad 属性为 requires_grad
  self_.set_requires_grad(requires_grad);
  // 包装修改后的 self_ 并返回给 Python
  return THPVariable_Wrap(self_);
  END_HANDLE_TH_ERRORS
// 结束之前的代码块
}

// 检查给定的张量是否在指定的内存格式下是连续的
inline bool dispatch_is_contiguous(const Tensor & self, MemoryFormat memory_format) {
  return self.is_contiguous(memory_format);
}

// Python 对象上的实现，以避免调度开销
static PyObject * THPVariable_is_contiguous(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义参数解析器，支持 "is_contiguous(*, MemoryFormat memory_format=contiguous_format)" 形式的参数
  static PythonArgParser parser({
    "is_contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  // 解析参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self_, args, kwargs, parsed_args);

  // 如果具有 torch 函数功能，则处理并返回对应的 torch 函数处理结果
  if(r.has_torch_function()){
    return handle_torch_function(r, self_, args, kwargs, PyObject_Type(self_), "torch.Tensor");
  }

  // 获取内存格式参数
  auto memory_format = r.memoryformat(0);
  // 解包 self_，获取对应的 Tensor 引用
  auto& self = THPVariable_Unpack(self_);
  // 调用内部的 dispatch_is_contiguous 函数，检查张量是否在指定内存格式下连续
  return wrap(dispatch_is_contiguous(self, memory_format));
  END_HANDLE_TH_ERRORS
}

// Python 对象上的实现，以避免调度开销
static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 如果检查到有 torch 函数，则处理并返回对应的 torch 函数处理结果
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "item", args);
  }
  // 在跟踪器中发出警告
  jit::tracer::warn("Converting a tensor to a Python number", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 解包 self，获取对应的 Tensor 引用
  auto& self_ = THPVariable_Unpack(self);
  // 定义调度函数 dispatch_item_，返回张量的标量值
  auto dispatch_item_ = [](const Tensor& self) -> at::Scalar {
    pybind11::gil_scoped_release no_gil; // 释放全局解释器锁
    return self.item(); // 调用 Tensor 对象的 item 方法，获取标量值
  };
  // 将 dispatch_item_ 函数的结果转换为 Python 对象并返回
  return py::cast(dispatch_item_(self_)).release().ptr();
  END_HANDLE_TH_ERRORS
}

// Python 对象上的实现，因为原生函数中没有对第一类函数的支持
// 查看：ATen/native/README.md 以获取更多上下文信息
static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义参数解析器，支持 "map_(Tensor other, PyObject* callable)" 形式的参数
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  // 解包 self，获取对应的 Tensor 引用
  auto& self_ = THPVariable_Unpack(self);
  // 解析参数
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果具有 torch 函数功能，则处理并返回对应的 torch 函数处理结果
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 获取第一个参数 other 的 Tensor 对象
  Variable other = r.tensor(0);
  // 如果 self 或 other 需要梯度，则抛出运行时错误
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  // 检查是否是 Python 调度，不支持 map_ 对于张量子类
  TORCH_CHECK(
      !self_.unsafeGetTensorImpl()->is_python_dispatch() && !other.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map_ is not supported for tensor subclasses.");

  // 调用 torch::utils::map_ 方法，并将结果进行包装后返回
  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

// Python 对象上的实现，因为原生函数中没有对第一类函数的支持
// 查看：ATen/native/README.md 以获取更多上下文信息
static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理 Torch 异常，确保异常处理范围安全
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定函数签名为 map2_(Tensor x, Tensor y, PyObject* callable)
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  // 解包 self 对象为 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 解析输入参数 args 和 kwargs
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 检查是否有 Torch 函数重载，若有则调用对应处理函数
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 提取解析后的 Tensor 对象 x 和 y
  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  // 检查是否有任何一个对象需要梯度，如果是则抛出运行时错误
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  // 检查 x 和 y 是否属于 Python 的分发类型，不支持对这些类型执行 .map2_ 操作
  TORCH_CHECK(
      !x.unsafeGetTensorImpl()->is_python_dispatch() && !y.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map2_ is not supported for tensor subclasses.");
  // 调用 Torch C++ 库中的 map2_ 函数，将结果封装为 Python 对象返回
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  // 结束 Torch 异常处理
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_new 函数，处理 Python 对象的新建操作
static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理 Torch 异常，确保异常处理范围安全
  HANDLE_TH_ERRORS
  // 检查是否存在 Torch 函数重载，若有则调用对应处理函数
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new", args, kwargs);
  }
  // 解包 self 对象为 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 根据 self_ 的设备类型创建设备保护器
  OptionalDeviceGuard device_guard(device_of(self_));
  // 调用 Torch C++ 库中的 legacy_tensor_new 函数，创建新的 Tensor 对象并封装返回
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  // 结束 Torch 异常处理
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_new_tensor 函数，处理 Python 对象的新建 Tensor 操作
static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理 Torch 异常，确保异常处理范围安全
  HANDLE_TH_ERRORS
  // 检查是否存在 Torch 函数重载，若有则调用对应处理函数
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new_tensor", args, kwargs);
  }
  // 解包 self 对象为 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 根据 self_ 的设备类型创建设备保护器
  OptionalDeviceGuard device_guard(device_of(self_));
  // 调用 Torch C++ 库中的 new_tensor 函数，创建新的 Tensor 对象并封装返回
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  // 结束 Torch 异常处理
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_storage 函数，返回 Python 对象的存储器
static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  // 处理 Torch 异常，确保异常处理范围安全
  HANDLE_TH_ERRORS
  // 检查是否存在 Torch 函数重载，若有则调用对应处理函数
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "untyped_storage");
  }
  // 解包 self 对象为 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 创建 Python 对象，表示 self_ 对象的存储器，并封装返回
  return createPyObject(self_.storage());
  // 结束 Torch 异常处理
  END_HANDLE_TH_ERRORS
}

// 定义 THPVariable_to 函数，处理 Python 对象的转换操作
static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  // 处理 Torch 异常，确保异常处理范围安全
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定多种 to() 函数的函数签名
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  // 解析输入参数 args 和 kwargs
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(self, args, kwargs);
  // 如果有 Torch 函数重载，调用对应处理函数
  if (r.has_torch_function()) {

    return handle_torch_function(self, "to", args, kwargs);
  }
  // 解包 self 对象为 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 执行转换操作，根据参数选择目标设备、数据类型等，并封装返回
  return THPVariable_Wrap(torch::utils::to_dispatch(self_, r));
  // 结束 Torch 异常处理
  END_HANDLE_TH_ERRORS
}
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }


  // 调用一个特定的 Torch 函数处理 Torch 功能的情况，返回处理结果
  return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");



  auto parsed = parse_to_conversion(r, /*allow_copy*/ true);


  // 调用解析函数将参数转换为适当类型，允许进行复制
  auto parsed = parse_to_conversion(r, /*allow_copy*/ true);



  auto& device = std::get<0>(parsed);


  // 获取解析后的结果中的设备引用
  auto& device = std::get<0>(parsed);



  auto& scalarType = std::get<1>(parsed);


  // 获取解析后的结果中的标量类型引用
  auto& scalarType = std::get<1>(parsed);



  auto non_blocking = std::get<2>(parsed);


  // 获取解析后的结果中的非阻塞标志
  auto non_blocking = std::get<2>(parsed);



  auto copy = std::get<3>(parsed);


  // 获取解析后的结果中的复制标志
  auto copy = std::get<3>(parsed);



  auto opt_memory_format = std::get<4>(parsed);


  // 获取解析后的结果中的可选内存格式
  auto opt_memory_format = std::get<4>(parsed);



  auto& self_ = THPVariable_Unpack(self);


  // 解包 self，获取其内部的 THPVariable 对象
  auto& self_ = THPVariable_Unpack(self);



  torch::utils::maybe_initialize_device(device);


  // 如果设备存在，可能需要初始化该设备
  torch::utils::maybe_initialize_device(device);



  if (device && device->is_privateuseone()) {
    at::globalContext().lazyInitPrivateUse1();
  }


  // 如果设备存在且为私有使用类型1，则延迟初始化全局上下文的私有使用类型1
  if (device && device->is_privateuseone()) {
    at::globalContext().lazyInitPrivateUse1();
  }



  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return self;
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }


  // 根据设备、标量类型、复制标志和内存格式分派到相应的处理函数
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    // 如果没有设备、标量类型、复制和内存格式，增加 Python 对象的引用计数并返回其本身
    Py_INCREF(self);
    return self;
  } else if (!device && !scalarType) {
    // 如果没有设备和标量类型，使用给定的非阻塞、复制和内存格式分派到处理函数，并封装返回结果
    return THPVariable_Wrap(dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    // 如果没有设备，使用给定的标量类型、非阻塞、复制和内存格式分派到处理函数，并封装返回结果
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    // 如果没有标量类型，使用给定的设备、非阻塞、复制和内存格式分派到处理函数，并封装返回结果
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    // 使用给定的设备、标量类型、非阻塞、复制和内存格式分派到处理函数，并封装返回结果
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }



  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS


  // 返回 None 对象的 Python 引用，同时结束处理 Torch 错误的区块
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
// 实现在 Python 对象上的方法，因为原生函数中不支持任意嵌套列表的声明
// 参考: ATen/native/README.md 获取更多上下文信息
static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  // 检查是否有 torch function，如果有则调用它
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "tolist", args);
  }
  // 发出警告，表示正在将张量转换为 Python 列表
  jit::tracer::warn("Converting a tensor to a Python list", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 解包 self 对象成 THPVariable
  auto self_ = THPVariable_Unpack(self);
  // 调用 torch::utils::tensor_to_list 函数，将张量转换为 Python 列表
  return torch::utils::tensor_to_list(self_);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  // 定义 PythonArgParser 对象 parser，用于解析参数
  static PythonArgParser parser({
    "type(PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  // 解包 self 对象成 THPVariable
  auto& self_ = THPVariable_Unpack(self);
  // 定义 ParsedArgs<3> 对象 parsed_args，用于存储解析后的参数
  ParsedArgs<3> parsed_args;
  // 调用 parser.parse 方法解析参数，并返回解析结果 r
  auto r = parser.parse(self, args, kwargs, parsed_args);

  // 如果 r 中有 torch function，则调用 handle_torch_function 处理
  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // 如果第一个参数是 None，则返回张量的选项字符串
  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::options_to_string(self_.options()));
  }
  // 否则，获取第一个参数对象
  auto obj = r.pyobject(0);
  // 获取第三个参数作为可选的 memory_format
  auto opt_memory_format = r.memoryformatOptional(2);
  // 定义 type_name 作为类型名称的字符串
  std::string type_name;
  // 初始化 is_dtype 标志为 false
  bool is_dtype = false;

  // 如果 obj 是 Python 类型对象
  if (PyType_Check(obj)) {
    // 如果 obj 是 THPVariableClass 类型，则 type_name 设为 "torch.Tensor"
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      // 否则，获取类型对象的名称
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    // 如果 obj 是字符串类型，则解包成字符串作为 type_name
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    // 如果 obj 是 dtype 类型对象，则设置 is_dtype 标志为 true
    is_dtype = true;
  } else {
    // 否则，抛出 TypeError 异常，提示 dtype 必须是类型、字符串或 dtype 对象
    throw TypeError("dtype must be a type, str, or dtype object");
  }

  // 定义 ScalarType 变量 scalar_type 和 Device 变量 device，获取张量的设备
  ScalarType scalar_type;
  Device device = self_.device();
  
  // 如果 is_dtype 为 true，则获取第一个参数作为标量类型，并调用 dispatch_to 函数
  if (is_dtype) {
    scalar_type = r.scalartype(0);
    return THPVariable_Wrap(dispatch_to(self_, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  }

  // 解析类型名称为张量选项 options
  at::TensorOptions options = torch::utils::options_from_string(type_name);
  // 获取选项中的标量类型
  scalar_type = at::typeMetaToScalarType(options.dtype());
  // 获取选项中的设备类型
  auto device_type = options.device().type();

  // 如果选项中的设备类型不同于当前设备的类型，则更新设备为选项中的设备类型
  if (device_type != device.type()) {
    device = at::Device(device_type);
  }

  // 可能会初始化设备 device
  torch::utils::maybe_initialize_device(device);

  // 如果设备类型是私有使用类型，则初始化私有使用上下文
  if (device.is_privateuseone()) {
    at::globalContext().lazyInitPrivateUse1();
  }

  // 调用 dispatch_to 函数，根据设备和标量类型处理张量
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// 生成的方法从这里开始

${py_methods}

// 实现在 Python 对象上的方法，用于处理 bool 标量转换
static PyObject * THPVariable_bool_scalar(PyObject* self, PyObject* args) {
  // 如果有 torch function，则调用 handle_torch_function 处理
  if (check_has_torch_function(self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function(self, "__bool__", args);
    END_HANDLE_TH_ERRORS
  }
  // 发出警告，表示正在将张量转换为 Python 布尔值
  jit::tracer::warn("Converting a tensor to a Python boolean", jit::tracer::WARN_PYTHON_DATAFLOW);
  // 调用 THPVariable_is_nonzero 函数，判断张量是否非零
  return THPVariable_is_nonzero(self, args);
}
// 实现 torch.Tensor 对象的 __eq__ 方法，用于比较相等性
static PyObject * THPVariable___eq__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
#ifdef USE_NUMPY
  // 检查是否支持 NumPy
  if (torch::utils::is_numpy_available()) {
    // 定义 Python 参数解析器，允许跟踪调用
    static PythonArgParser parser({
      "__eq__(PyObject* other)",
    }, /*traceable=*/true);

    ParsedArgs<1> parsed_args;
    auto _r = parser.parse(self_, args, kwargs, parsed_args);
    // 检查是否有 Torch 函数重载
    if(_r.has_torch_function()) {
      // 处理 Torch 函数重载调用
      return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
    }
    switch (_r.idx) {
      case 0: {
        auto other = _r.pyobject(0);
        // 检查参数是否为 NumPy 数组
        if (PyArray_Check(other)) {
          // 将 NumPy 数组转换为 Torch 张量
          auto other_tensor = torch::utils::tensor_from_numpy(other);
          // 定义相等比较的分发函数
          auto dispatch_eq = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
            // 释放全局解释器锁
            pybind11::gil_scoped_release no_gil;
            // 执行张量的相等性比较操作
            return self.eq(other);
          };
          // 获取当前 THPVariable 对象的 Tensor 引用
          const Tensor& self = THPVariable_Unpack(self_);
          // 将比较结果进行包装并返回
          return wrap(dispatch_eq(self, other_tensor));
        }
      }
    }
  }
#endif
  // 若不支持 NumPy 或未匹配到特定条件，则调用默认的 THPVariable_eq 方法
  return THPVariable_eq(self_, args, kwargs);
  // 返回 None
  Py_RETURN_NONE;
  // 处理 Torch 错误结束
  END_HANDLE_TH_ERRORS
}

// 将引发的 TypeError 转换为返回 NotImplemented
// 用于实现二元算术运算符
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {

  PyObject* ret = Func(self, args, kwargs);
  // 如果没有返回值并且引发了 TypeError，则清除异常并返回 NotImplemented
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  // 返回处理后的结果
  return ret;
}

// set_ 必须在模板中定义，因为 c10::Storage 对象没有具体类型，
// 我们需要确保 Python 存储对象的类型与张量的类型匹配
static PyObject* THPVariable_set_(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 获取当前 THPVariable 对象的 Tensor 引用
  const Tensor& self = THPVariable_Unpack(self_);
  // 定义 Python 参数解析器，不允许跟踪调用
  static PythonArgParser parser(
      {
          "set_()",
          "set_(Storage source)",
          "set_(Storage source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
          "set_(Tensor source)",
          "set_(Tensor source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
      },
      /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::set_(Tensor(a!) self) -> Tensor(a!)
      // 定义 set_ 操作的分发函数
      auto dispatch_set_ = [](const Tensor& self) -> Tensor {
        // 释放全局解释器锁
        pybind11::gil_scoped_release no_gil;
        // 执行张量的 set_ 操作
        return self.set_();
      };
      // 将结果进行包装并返回
      return wrap(dispatch_set_(self));
    }
    case 1: {
      // 定义存储标量类型变量
      at::ScalarType storage_scalar_type;
      // 是否为有类型的存储
      bool is_typed_storage = true;
      // 从返回值中获取存储对象
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      // 检查存储对象的标量类型与当前张量的数据类型是否匹配，或者存储对象是否为无类型存储
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      // 定义 lambda 函数 dispatch_set_，用于执行 set_ 操作
      auto dispatch_set_ = [](const Tensor& self, Storage source) -> Tensor {
        // 释放全局解释器锁，允许多线程操作
        pybind11::gil_scoped_release no_gil;
        // 调用 self 的 set_ 方法设置数据源为 source
        return self.set_(source);
      };
      // 使用 wrap 函数包装并返回 dispatch_set_ 函数的结果
      return wrap(dispatch_set_(self, storage));
    }
    case 2: {
      // 定义存储标量类型变量
      at::ScalarType storage_scalar_type;
      // 是否为有类型的存储
      bool is_typed_storage = true;
      // 从返回值中获取存储对象
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      // 检查存储对象的标量类型与当前张量的数据类型是否匹配，或者存储对象是否为无类型存储
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      // 定义 lambda 函数 dispatch_set_，用于执行 set_ 操作
      auto dispatch_set_ = [](const Tensor& self,
                              Storage source,
                              c10::SymInt storage_offset,
                              c10::SymIntArrayRef size,
                              c10::SymIntArrayRef stride) -> Tensor {
        // 释放全局解释器锁，允许多线程操作
        pybind11::gil_scoped_release no_gil;
        // 调用 self 的 set__symint 方法设置数据源为 source，并传入偏移量、大小和步幅
        return self.set__symint(source, storage_offset, size, stride);
      };
      // 使用 wrap 函数包装并返回 dispatch_set_ 函数的结果，传入相应参数
      return wrap(dispatch_set_(
          self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
    }
    case 3: {
      // 定义 lambda 函数 dispatch_set_，用于执行 set_ 操作
      auto dispatch_set_ = [](const Tensor& self, const Tensor& source) -> Tensor {
        // 检查源张量的数据类型是否与当前张量的数据类型匹配
        TORCH_CHECK(source.dtype() == self.dtype(), "Could not set tensor of type ", source.dtype(), " to a tensor of type ", self.dtype());
        // 释放全局解释器锁，允许多线程操作
        pybind11::gil_scoped_release no_gil;
        // 调用 self 的 set_ 方法设置数据源为 source
        return self.set_(source);
      };
      // 使用 wrap 函数包装并返回 dispatch_set_ 函数的结果，传入第一个参数为当前张量，第二个参数为 _r.tensor(0) 的返回值
      return wrap(dispatch_set_(self, _r.tensor(0)));
    }
    // case 4: 分支处理第四种情况

    // 获取第一个参数作为 Tensor 对象，并命名为 storage
    at::Tensor storage = _r.tensor(0);

    // 定义一个 lambda 函数 dispatch_set_，接受 Tensor 类型的参数和一些符号化的整数数组作为参数，并返回一个 Tensor 对象
    auto dispatch_set_ = [](const Tensor& self,
                            const Tensor& source,
                            c10::SymInt storage_offset,
                            c10::SymIntArrayRef size,
                            c10::SymIntArrayRef stride) -> Tensor {
        // 释放 GIL（全局解释器锁），允许线程并发执行
        pybind11::gil_scoped_release no_gil;
        // 调用 self 的 set__symint 方法，传入 source、storage_offset、size 和 stride，返回结果
        return self.set__symint(source, storage_offset, size, stride);
    };

    // 返回调用 dispatch_set_ lambda 函数的结果，传入相应参数包装后的结果
    return wrap(dispatch_set_(
        self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
};

} // namespace torch::autograd
```