# `.\pytorch\torch\csrc\autograd\python_variable.cpp`

```
// 包含 ATen 库中的命名张量工具函数
#include <ATen/NamedTensorUtils.h>
// 包含 C10 库中的设备类型定义
#include <c10/core/DeviceType.h>
// 包含 C10 库中的 GPU 跟踪实现
#include <c10/core/impl/GPUTrace.h>
// 包含 C10 库中的 Python 线程局部存储实现
#include <c10/core/impl/HermeticPyObjectTLS.h>
// 包含 C10 库中的 Python 调度器线程局部存储实现
#include <c10/core/impl/PythonDispatcherTLS.h>
// 包含 C10 实用工具中的整数范围函数
#include <c10/util/irange.h>
// 包含 pybind11 中的 Python 类型支持
#include <pybind11/pytypes.h>
// 包含 Torch 库中的设备类型定义
#include <torch/csrc/Device.h>
// 包含 Torch 库中的动态类型定义
#include <torch/csrc/DynamicTypes.h>
// 包含 Torch 库中的异常处理定义
#include <torch/csrc/Exceptions.h>
// 包含 Torch 库中的 Python 解释器接口定义
#include <torch/csrc/PyInterpreter.h>
// 包含 Torch 库中的张量尺寸定义
#include <torch/csrc/Size.h>
// 包含 Torch 库中的 THP 定义
#include <torch/csrc/THP.h>
// 包含 Torch 库中的类型定义
#include <torch/csrc/Types.h>
// 包含 Torch 自动求导模块中的定义
#include <torch/csrc/autograd/autograd.h>
// 包含 Torch 自动求导模块中的边缘处理定义
#include <torch/csrc/autograd/edge.h>
// 包含 Torch 自动求导模块中的函数定义
#include <torch/csrc/autograd/function.h>
// 包含 Torch 自动求导模块中的 Python-C++ 函数定义
#include <torch/csrc/autograd/python_cpp_function.h>
// 包含 Torch 自动求导模块中的 Python 钩子定义
#include <torch/csrc/autograd/python_hook.h>
// 包含 Torch 自动求导模块中的 Python 变量索引定义
#include <torch/csrc/autograd/python_variable_indexing.h>
// 包含 Torch 自动求导模块中的错误信息定义
#include <torch/csrc/autograd/utils/error_messages.h>
// 包含 Torch 自动求导模块中的输出包装定义
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 包含 Torch 自动求导模块中的变量定义
#include <torch/csrc/autograd/variable.h>
// 包含 Torch JIT 前端追踪器定义
#include <torch/csrc/jit/frontend/tracer.h>
// 包含 Torch JIT Python 绑定工具函数
#include <torch/csrc/jit/python/pybind_utils.h>
// 包含 Torch 张量 Python 接口定义
#include <torch/csrc/tensor/python_tensor.h>
// 包含 Torch 实用工具的 Python 绑定支持
#include <torch/csrc/utils/pybind.h>
// 包含 Torch 实用工具中的 Python C 函数助手函数
#include <torch/csrc/utils/pycfunction_helpers.h>
// 包含 Torch 实用工具中的 Python 对象保留实现
#include <torch/csrc/utils/pyobject_preservation.h>
// 包含 Torch 实用工具中的 Python 参数解析器定义
#include <torch/csrc/utils/python_arg_parser.h>
// 包含 Torch 实用工具中的 Python 分派支持
#include <torch/csrc/utils/python_dispatch.h>
// 包含 Torch 实用工具中的 Python 字符串处理支持
#include <torch/csrc/utils/python_strings.h>
// 包含 Torch 实用工具中的张量创建支持
#include <torch/csrc/utils/tensor_new.h>
// 包含 Torch 实用工具中的张量 NumPy 支持
#include <torch/csrc/utils/tensor_numpy.h>

// 包含 Torch 实用工具中的调度模式定义
#include <torch/csrc/utils/torch_dispatch_mode.h>

// 包含 ATen 库中的 ATen 张量实现
#include <ATen/ATen.h>

// 包含 C10 库中的符号整数数组引用定义
#include <c10/core/SymIntArrayRef.h>

// 包含 C 标准库中的结构成员定义
#include <structmember.h>
// 包含 C 标准库中的整数类型定义
#include <cstdint>
// 包含 C++ 标准库中的智能指针支持
#include <memory>
// 包含 C++ 标准库中的实用工具支持
#include <utility>
// 包含 C++ 标准库中的向量容器支持
#include <vector>

// 使用 at 命名空间
using namespace at;
// 使用 torch 命名空间
using namespace torch;
// 使用 torch::autograd 命名空间
using namespace torch::autograd;

// 定义一个函数，将 OperatorHandle 和 IValue 列表转换为 Python 参数和关键字参数
std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments) {
  // 检查 Python 全局解释器锁是否被持有
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");
  // 获取操作符的函数签名信息
  const auto& schema = op.schema();
  // 创建 Python 字典对象，用于存储关键字参数
  py::dict kwargs;

  // 关于指针的说明:
  //
  // f(int x, int y = 0, *, int z = 0)
  //                                  ^- arguments.size()
  //                        ^- kwarg_only_start
  //          ^- positional_default_start
  //   ^- 0

  // 查找关键字参数和常规参数之间的分隔点。由于大多数函数没有关键字参数，从右向左扫描更有效。
  // （理想情况下，这应该在 FunctionSchema 中预先计算，但这里是动态计算的）
  int64_t kwarg_only_start = static_cast<int64_t>(arguments.size());
  for (; kwarg_only_start > 0; kwarg_only_start--) {
    // 获取参数的定义信息
    const auto& arg = schema.arguments()[kwarg_only_start - 1];
    // 如果参数不是关键字参数，则停止
    if (!arg.kwarg_only()) {
      break;
    }
  }

  // 查找第一个非默认的位置参数
  auto is_default = [&](size_t idx) -> bool {
    // 获取参数列表中索引为 idx 的参数
    const auto& arg = schema.arguments()[idx];
    // 如果参数没有默认值，则返回 false
    if (!arg.default_value().has_value()) {
      return false;
    }
    // 获取参数的默认值
    const auto& default_ivalue = *arg.default_value();
    // 获取实际传入的参数值
    const auto& ivalue = arguments[idx];
    // 如果默认值和传入值不相等，则返回 false
    if (default_ivalue != ivalue) {
      return false;
    }
    // 否则返回 true
    return true;
  };

  // 确定起始位置为只有位置参数的默认值
  int64_t positional_default_start = kwarg_only_start;
  for (; positional_default_start > 0; positional_default_start--) {
    // 如果不是默认值，则退出循环
    if (!is_default(positional_default_start - 1)) {
      break;
    }
  }

  // 创建一个 Python 元组对象，用于存储位置参数的默认值
  auto args =
      py::reinterpret_steal<py::object>(PyTuple_New(positional_default_start));

  // 将参数类型转换为 Python 对象的 Lambda 函数
  auto schemaAwareToPyObject = [&](size_t idx) -> py::object {
    // 获取参数列表中索引为 idx 的参数
    const auto& arg = schema.arguments()[idx];
    // 匹配函数，用于判断参数类型是否匹配
    auto match = [&](c10::TypeKind kind) {
      // 获取参数的真实类型
      const auto& t = arg.real_type();
      // 如果类型匹配，则返回 true
      if (t->kind() == kind)
        return true;
      // 如果是可选类型，则判断元素类型是否匹配
      if (auto opt_t = t->cast<c10::OptionalType>()) {
        if (opt_t->getElementType()->kind() == kind)
          return true;
      }
      // 否则返回 false
      return false;
    };
    // 如果参数是 None 类型，则返回 Python 的 None 对象
    if (arguments[idx].isNone()) {
      return py::none();
    } else if (match(c10::ScalarTypeType::Kind)) {
      // 如果参数是 ScalarType 类型，则返回对应的 Python 对象
      auto* obj =
          getTHPDtype(static_cast<c10::ScalarType>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::LayoutType::Kind)) {
      // 如果参数是 LayoutType 类型，则返回对应的 Python 对象
      auto* obj =
          getTHPLayout(static_cast<c10::Layout>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::MemoryFormatType::Kind)) {
      // 如果参数是 MemoryFormatType 类型，则返回对应的 Python 对象
      return py::cast(static_cast<c10::MemoryFormat>(arguments[idx].toInt()));
    } else {
      // 否则调用 Torch 的函数将参数转换为 Python 对象并返回
      return torch::jit::toPyObject(arguments[idx]);
    }
  };

  // 填充位置参数
  for (const auto idx : c10::irange(positional_default_start)) {
    // 将 schemaAwareToPyObject 函数处理后的对象设置到 Python 元组中
    PyTuple_SET_ITEM(
        args.ptr(), idx, schemaAwareToPyObject(idx).release().ptr());
  }

  // 填充关键字参数
  for (const auto idx : c10::irange(kwarg_only_start, arguments.size())) {
    // 跳过默认的关键字参数
    if (is_default(idx))
      continue;
    // 获取参数列表中索引为 idx 的参数
    const auto& arg = schema.arguments()[idx];
    // 将 schemaAwareToPyObject 函数处理后的对象存储到 kwargs 中
    kwargs[py::cast(arg.name())] = schemaAwareToPyObject(idx);
  }
  // 返回填充好的 args 和 kwargs 对象
  return std::make_pair(std::move(args), std::move(kwargs));
}

// 将 Python 输出推送到堆栈中
void pushPyOutToStack(
    const c10::OperatorHandle& op,  // 操作符句柄
    torch::jit::Stack* stack,       // Torch JIT 的堆栈指针
    py::object out,                 // Python 对象作为输出
    const char* msg) {              // 错误消息字符串指针
  TORCH_CHECK(
      PyGILState_Check(), "GIL must be held before you call pushPyOutToStack");  // 检查全局解释器锁状态
  auto schema_returns = op.schema().returns();  // 获取操作符的返回值模式
  const auto num_returns = schema_returns.size();  // 返回值的数量
  if (num_returns == 0) {
    // 检查 Python 是否返回 None；其他返回值会导致错误
    TORCH_CHECK(
        out.is_none(),
        "Expected ",
        msg,
        " for ",
        op.operator_name(),
        " to return None but it returned something else instead.");
  } else if (num_returns == 1) {
    // 将单个返回值转换为 Torch 的 IValue 并推送到堆栈中
    torch::jit::push(
        stack, torch::jit::toIValue(out.ptr(), schema_returns[0].real_type()));
  } else {
    // 处理多个返回值的情况
    auto outs = py::cast<py::sequence>(out);  // 将 Python 对象转换为 Python 序列
    for (const auto idx : c10::irange(outs.size())) {
      // 将每个返回值转换为 Torch 的 IValue 并推送到堆栈中
      torch::jit::push(
          stack,
          torch::jit::toIValue(
              outs[idx].ptr(), schema_returns[idx].real_type()));
    }
  }
}

namespace {

// 解析 sizes_strides_policy 参数
c10::TensorImpl::SizesStridesPolicy parseSizesStridesPolicyArgument(
    c10::string_view arg) {
  if (arg == "strides") {
    return c10::TensorImpl::SizesStridesPolicy::CustomStrides;  // 返回自定义步幅策略
  }

  if (arg == "sizes") {
    return c10::TensorImpl::SizesStridesPolicy::CustomSizes;  // 返回自定义尺寸策略
  }

  TORCH_CHECK_VALUE(
      false,
      "Unknown sizes_strides_policy: ",
      arg,
      "; expected 'strides' or 'sizes'");  // 未知的参数值，抛出错误
}
} // 匿名命名空间结束

PyObject* THPVariableClass = nullptr;  // THPVariableClass 指针初始化为空

PyObject* ParameterClass = nullptr;  // ParameterClass 指针初始化为空

// 使用给定变量创建新的 THPVariable 对象
static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status,
    bool allow_preexisting_pyobj = false);

// 静态常量字符串，用于指示已弃用的 volatile 功能
static const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

// 检查对象是否具有 torch_dispatch 属性
static bool check_has_torch_dispatch(PyObject* obj) {
  PyTypeObject* tp = Py_TYPE(obj);  // 获取对象的类型
  if (THPVariable_CheckTypeExact(tp)) {  // 检查是否为 THPVariable 类型
    return false;  // 如果是 THPVariable 类型，返回 false
  }
  py::object attr = PyObject_FastGetAttrString(obj, "__torch_dispatch__");  // 快速获取对象的 "__torch_dispatch__" 属性
  return (
      attr.ptr() != nullptr &&
      attr.ptr() != torch::disabled_torch_dispatch_impl());  // 返回是否存在 "__torch_dispatch__" 属性
}

// 静态数组，用于存储每种设备类型的 Python 类
// NOLINTNEXTLINE
static PyObject* device_to_py_class_[static_cast<size_t>(
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

// 注册 Python Tensor 类
void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class) {
  c10::Device dev(device);  // 创建设备对象

  TORCH_CHECK(
      dev.type() == kXLA, "Only the python class for XLA can be overriden");  // 检查设备类型是否为 XLA
  if (device_to_py_class_[static_cast<size_t>(dev.type())] != nullptr) {
    TORCH_WARN(
        "Overriding a previously registered python class for ", dev.str());  // 警告：覆盖先前注册的 Python 类
  }

  device_to_py_class_[static_cast<size_t>(dev.type())] = python_tensor_class;  // 存储 Python Tensor 类
}

// 获取指定设备类型的 Python Tensor 类
static PyObject* getPythonTensorClass(c10::Device d) {
  return device_to_py_class_[static_cast<size_t>(d.type())];  // 返回对应设备类型的 Python Tensor 类
}
void activateGPUTrace() {
  // 激活 GPU 跟踪，设置跟踪器使用当前 Python 解释器
  c10::impl::GPUTrace::set_trace(getPyInterpreter());
}

// TODO: Make this take Variable by const reference
// 将 TensorBase 包装为 PyObject 指针
PyObject* THPVariable_Wrap(at::TensorBase var) {
  // 如果 var 未定义，则返回 None
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  // 如果处于隔离状态，则使用 THPVariable_NewWithVar 创建新的 THPVariable
  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    return THPVariable_NewWithVar(
        (PyTypeObject*)THPVariableClass,
        std::move(var),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  }

  // 检查是否有与 Python 对象相关联的 Python 对象
  std::optional<PyObject*> mb_obj =
      var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  c10::impl::PyInterpreterStatus status{};
  if (mb_obj.has_value()) {
    auto obj = *mb_obj;
    if (obj) {
      // 如果 C++ 拥有 Python 对象，则将所有权转移回 Python
      if (var.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()) {
        var.unsafeGetTensorImpl()->pyobj_slot()->set_owns_pyobj(false);
        reinterpret_cast<THPVariable*>(obj)->cdata =
            MaybeOwned<Variable>::owned(std::move(var));
        return obj;
      }
      // 增加引用计数并返回对象
      Py_INCREF(obj);
      return obj;
    }
    // 如果标记过但没有有效 PyObject，设置状态为 TAGGED_BY_US
    status = c10::impl::PyInterpreterStatus::TAGGED_BY_US;
  } else {
    // 假设：如果一个 Tensor 被跨线程共享，则增加引用计数
    if (var.use_count() <= 1) {
      status = c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED;
    } else {
      status = c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED;
    }
  }

  // 根据 Tensor 的设备类型选择创建 THPVariable
  if (C10_LIKELY(var.device().type() != c10::kXLA)) {
    return THPVariable_NewWithVar(
        (PyTypeObject*)THPVariableClass, std::move(var), status);
  }

  // 根据设备类型获取对应的 Python Tensor 类，创建 THPVariable
  if (auto clazz = getPythonTensorClass(var.device())) {
    return THPVariable_NewWithVar((PyTypeObject*)clazz, std::move(var), status);
  }

  // 默认情况下创建 THPVariable
  return THPVariable_NewWithVar(
      (PyTypeObject*)THPVariableClass, std::move(var), status);
}
// 判断是否可以复活该 THPVariable 对象
bool isResurrectable(THPVariable* self) {
  // 我们将此检查分为两种情况。

  // 1. C++ 拥有 PyObject（在这种情况下，self->cdata.unsafeIsBorrowed() 为 true）。
  // 在这种情况下，tp_clear 不会被调用：因为 C++ 引用的 PyObject 会保持它存活。
  // 实际上，当 C++ 拥有 PyObject 时，我们有一个不变量，即 PyObject 的引用计数应该恰好为一（因为如果您引用 PyObject 的另一个引用，我们应该将所有权指针翻转回来）。
  // 但实际上，通过弱引用，您可以暂时违反这个不变量，因此我们在断言中不测试它。

  // 2. PyObject 拥有 C++ 对象（在这种情况下，self->cdata.unsafeIsBorrowed() 为 false）。
  // 在这种情况下，如果 PyObject 被一个死循环引用，tp_clear 可能会被调用。
  // 但是如果没有复活发生，那么从 PyObject 到 C++ 对象的引用必须是唯一的。
  if (self->cdata.unsafeIsBorrowed()) {
    return false;
  }

  // 获取 THPVariable 对应的 tensor 对象
  auto const& tensor = THPVariable_Unpack(self);

  // 如果 tensor 未定义或者引用计数小于等于 1，则不可复活
  if (!tensor.defined() || tensor.use_count() <= 1) {
    return false;
  }

  // 检查是否是封闭的。如果是，不进行复活。
  if (tensor.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          getPyInterpreter(), /*ignore_hermetic_tls=*/false) !=
      c10::make_optional((PyObject*)self)) {
    return false;
  }

  // 可以复活
  return true;
}

// 尝试复活 THPVariable 对象，如果成功复活则取消后续的释放操作
static bool THPVariable_tryResurrect(THPVariable* self) {
  // 获取 THPVariable 对应的 tensor 对象
  const auto& tensor = THPVariable_Unpack(self);

  // 如果不可复活，则返回 false
  if (!isResurrectable(self)) {
    return false;
  }

  // 现在确定要复活 tensor 对象。因此，tensor 必须已定义
  TORCH_INTERNAL_ASSERT(tensor.defined());

  // 存在其他的 C++ 拥有者，翻转所有权，以便 C++ 拥有此 Python 对象，并取消释放操作
  TORCH_INTERNAL_ASSERT(
      !tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj());

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

  // 检查是否存在 PyObject，如果没有则报错
  auto maybe_pyobj = tensor_impl->pyobj_slot()->check_pyobj(
      getPyInterpreter(),
      /*ignore_hermetic_tls=*/false);

  TORCH_INTERNAL_ASSERT(
      maybe_pyobj.has_value(),
      "Trying to preserve a Python tensor whose PyObjectSlot does not have a PyObject");

  // 设置 tensor_impl 拥有 PyObject 的所有权
  tensor_impl->pyobj_slot()->set_owns_pyobj(true);

  // 复活 Python 对象。这是 CPython 偶尔在内部执行的操作，参见链接中的代码段。
  // 我们不必担心保存和恢复引用计数（如引用代码所示），因为我们确实需要在这里将引用计数重置为一，
  // 我们不能假设其他代码已经处理了这个。
// NB: this will overreport _Py_RefTotal but based on inspection of object.c
// there is no way to avoid this
#ifdef Py_TRACE_REFS
  // 如果启用了 Python 的引用追踪，将 self 添加到全局对象列表中
  _Py_AddToAllObjects(reinterpret_cast<PyObject*>(self), 1);
#endif

// 增加 self 的引用计数
Py_INCREF(self);

// Flip THPVariable to be non-owning
// (near use-after-free miss here: fresh MaybeOwned is created breaking
// reference on Tensor in struct BEFORE we overwrite the old one)
// 确保在此处 THPVariable 变为非所有权状态
TORCH_INTERNAL_ASSERT(!c10::impl::HermeticPyObjectTLS::get_state());
self->cdata = MaybeOwned<Variable>::borrowed(tensor);

// NB: At this point, tensor *could* be dead (e.g., some other C++ thread
// decrefed it.)  At this point, it is probably waiting on the GIL to
// deallocate the Python object and will kill self, BUT NOT YET.

// 在这一点上，tensor 可能已经被销毁（例如，某些其他的 C++ 线程已经减少了它的引用计数）。
// 现在它可能正在等待 GIL（全局解释器锁）来释放 Python 对象并销毁 self，但还没有发生。

return true;
}

static int THPVariable_clear(THPVariable* self) {
// Is it OK for an object to still be live after running
// tp_clear? Yes. When Python is breaking reference cycles, it can't assume
// that an object will dealloc after it's cleared.  The source code explicitly
// handles this case:
// https://github.com/python/cpython/blob/4e661cd69164318c1f871faa476c68a04092ddc4/Modules/gcmodule.c#L1010-L1025

// Note that we don't need to actually resurrect here. There are 2 cases:
// 1. The PyObject is not part of a reference cycle. In this case, we don't
// need to do anything. The GC will move on to try and break the reference
// cycle on another object, which will eventually trigger tp_dealloc (and thus
// resurrection).

// 2. The PyObject is part of a reference cycle. This case should not actually
// be possible, due to the logic in our tp_traverse
// (THPVariable_subclass_traverse).

// In fact, resurrecting here breaks the invariant that "C++ owns Python only
// when PyObject's refcount would otherwise be 0". Most immediately, as we're
// merely breaking reference cycles here, there can be other references to the
// PyObject. *However*, if other objects in the refcycle resurrect, then we
// will be in a state where the PyObject has multiple Python references, yet
// C++ owns the PyObject.

// See https://github.com/pytorch/pytorch/pull/75933 for more discussion.
// 如果对象可以在运行 tp_clear 后仍然存活，是否可以？是的。当 Python 打破引用循环时，
// 不能假设对象在清除后将被释放。源代码明确处理了这种情况：
// https://github.com/python/cpython/blob/4e661cd69164318c1f871faa476c68a04092ddc4/Modules/gcmodule.c#L1010-L1025

// 注意，我们不需要在这里实际上复活对象。有两种情况：
// 1. PyObject 不是引用循环的一部分。在这种情况下，我们不需要做任何操作。
//    垃圾收集器将继续尝试破坏另一个对象的引用循环，最终会触发 tp_dealloc（从而可能复活）。
// 2. PyObject 是引用循环的一部分。由于我们在 tp_traverse 中的逻辑，实际上不应该发生这种情况
//    （THPVariable_subclass_traverse）。

// 实际上，在这里复活对象会破坏“C++ 仅在 PyObject 的引用计数本应为 0 时拥有它”的不变性。
// 由于我们仅在此处打破引用循环，可能有其他对 PyObject 的引用。但是，如果引用循环中的其他对象
// 复活，则我们将处于一种状态，即 PyObject 具有多个 Python 引用，但 C++ 拥有该 PyObject。

// 有关更多讨论，请参见 https://github.com/pytorch/pytorch/pull/75933 。

if (isResurrectable((THPVariable*)self)) {
return 0;
}
Py_CLEAR(self->backward_hooks);
Py_CLEAR(self->post_accumulate_grad_hooks);
const auto& tensor = THPVariable_Unpack(self);
if (tensor.defined()) {
// Two situations to consider:
//    PyObject -owns-> Tensor
//        unsafeIsBorrowed() is FALSE.  We're obligated to look through
//        Tensor to break references.  Clearing cdata must induce the
//        destruction of the C++ Tensor.  If there were other references
//        to C++ tensor, the Python object would have been resurrected
//        by flipping the ownership.
//    Tensor -owns-> PyObject
//        unsafeIsBorrowed() is TRUE.  We're deallocating the PyObject
//        because Tensor asked us to (it's already destructing).

// 要考虑的两种情况：
//    PyObject 拥有 Tensor
//        unsafeIsBorrowed() 为 FALSE。我们有责任查看 Tensor 以打破引用。清除 cdata 必须导致
//        销毁 C++ Tensor。如果有其他对 C++ tensor 的引用，Python 对象将通过翻转所有权而复活。
//    Tensor 拥有 PyObject
//        unsafeIsBorrowed() 为 TRUE。我们正在释放 PyObject，因为 Tensor 要求我们这样做（它已经在析构）。
    // 如果张量不是借用状态并且其底层实现的 Python 对象拥有权检查通过
    // 注意：这段代码可能与 macOS 下的某些测试不一致，
    // 在 test_py_tensors_multi_async_call - ProcessGroupRpcTestWithSpawn 中出现
    // 如果出现异常，可能是因为不满足 !tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj() 的断言
    if (!self->cdata.unsafeIsBorrowed() &&
        tensor.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
            getPyInterpreter(), /*ignore_hermetic_tls=*/false) ==
            c10::make_optional((PyObject*)self)) {
      
      // 如果能获取到梯度累加器对象
      if (auto grad_acc =
              torch::autograd::impl::try_get_grad_accumulator(tensor)) {
        // 清空梯度累加器的预处理钩子
        grad_acc->pre_hooks().clear();
        grad_acc->tensor_pre_hooks().clear();
        grad_acc->retains_grad_hooks().clear();
      }
    }
  }
  
  // 断言：确保 self 不可复活（resurrectable）
  TORCH_INTERNAL_ASSERT(!isResurrectable((THPVariable*)self));
  
  {
    // MapAllocator 可能花费大量时间来释放大张量的内存；
    // 在此处释放 GIL（全局解释器锁），以避免影响主线程性能。
    pybind11::gil_scoped_release no_gil;
    // 将 self 的数据成员 cdata 设置为空的 MaybeOwned<Variable> 对象
    self->cdata = MaybeOwned<Variable>();
  }
  
  // 返回值为 0，表示正常结束
  return 0;
}

// THPFunction_traverse函数的实现，用于遍历THPFunction对象
int THPFunction_traverse(THPFunction* self, visitproc visit, void* arg) {
  // 断言，如果条件为假，则输出错误信息
  TORCH_INTERNAL_ASSERT(
      false, "Tensor tp_traverse function was not overriden properly");
  // 返回0，表示遍历完成
  return 0;
}

// THPVariable_pynew函数的声明
PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs);

// THPVariable_fix_weakref函数的实现，修复弱引用
static PyObject* THPVariable_fix_weakref(PyObject* self, PyObject* noargs) {
  // 获取THPVariable对象
  const auto& var = THPVariable_Unpack(self);
  // 减少THPVariable对象的引用计数
  Py_DECREF(THPVariable_Wrap(var));
  // 返回None对象
  Py_RETURN_NONE;
}

// map_py_func模板函数，将给定的Python可调用对象映射到项目的向量上
// 返回相同类型的项目向量
template <typename T>
static std::vector<T> map_py_func(
    const py::function& func,
    const std::vector<T>& items) {
  std::vector<T> new_items;
  new_items.reserve(items.size());
  // 遍历项目向量，应用给定的Python函数，并将结果存入新的向量中
  for (auto& item : items) {
    new_items.emplace_back(py::cast<T>(func(item)));
  }
  return new_items;
}

// 特化模板，针对at::Tensor类型的项目向量
std::vector<at::Tensor> map_py_func(
    const py::function& func,
    const std::vector<at::Tensor>& items) {
  std::vector<at::Tensor> new_items;
  new_items.reserve(items.size());
  // 遍历at::Tensor类型的项目向量，应用给定的Python函数，并将结果存入新的向量中
  for (auto& item : items) {
    auto output = func(item);
    // 如果输出为None对象，则将未定义的Tensor放入新向量中
    if (output.is(py::none())) {
      new_items.emplace_back();
    } else {
      new_items.emplace_back(py::cast<at::Tensor>(output));
    }
  }
  return new_items;
}

// view_func_impl函数的实现，用于处理视图操作的具体实现
static PyObject* view_func_impl(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs,
    bool check_has_same_meta) {
  HANDLE_TH_ERRORS
  // 解包THPVariable对象
  const auto& self = THPVariable_Unpack(_self);

  // 静态Python参数解析器，解析view函数的参数
  static PythonArgParser parser({
      "_view_func(Tensor new_base, PyObject* symint_visitor_fn=None, PyObject* tensor_visitor_fn=None)",
  });
  ParsedArgs<3> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  auto new_base = r.tensor(0);
  PyObject* symint_visitor_fn = r.pyobject(1);
  PyObject* tensor_visitor_fn = r.pyobject(2);

  // 确保self确实是可回溯的视图，如果不是，则返回未定义的Tensor (None)
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // 确保新提供的base与原始base类似
    // 如果不需要检查元数据是否相同，或者新的基本张量具有相同的元数据
    if (!check_has_same_meta ||
        torch::autograd::utils::has_same_meta(new_base, view_info.base_)) {
      // 进行实际的视图重放操作
    
      // 如果视图信息包含视图函数
      if (view_info.has_view_fn()) {
        auto& view_func = view_info.view_fn();
    
        // 根据需要确定新的 SymInt / 张量状态
        std::optional<std::vector<c10::SymInt>> new_symints = c10::nullopt;
        if (symint_visitor_fn != Py_None) {
          new_symints = map_py_func(
              py::cast<py::function>(symint_visitor_fn),
              view_func.get_symints());
        }
    
        std::optional<std::vector<at::Tensor>> new_tensors = c10::nullopt;
        if (tensor_visitor_fn != Py_None) {
          new_tensors = map_py_func(
              py::cast<py::function>(tensor_visitor_fn),
              view_func.get_tensors());
        }
    
        // 调用视图函数
        if (new_symints.has_value() || new_tensors.has_value()) {
          out = (*view_func.clone_and_set(new_symints, new_tensors))(new_base);
        } else {
          out = view_func(new_base);
        }
      } else {
        // 如果没有视图函数，进行基于给定参数的 stride 进行视图操作
        out = new_base.as_strided(
            self.sizes(), self.strides(), self.storage_offset());
      }
    }
    // 返回封装后的 THPVariable
    return THPVariable_Wrap(std::move(out));
    END_HANDLE_TH_ERRORS
// 返回视图函数的实现结果，基于传入的参数和关键字参数，选择是否检查是否具有相同的元数据
static PyObject* THPVariable_view_func(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/true);
}

// 返回不安全的视图函数的实现结果，基于传入的参数和关键字参数，选择是否检查是否具有相同的元数据
static PyObject* THPVariable_view_func_unsafe(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/false);
}

// 反向视图函数的实现，解析并处理传入的自变量和参数
static PyObject* rev_view_func_impl(PyObject* self_, PyObject* arg) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(self_);
  TORCH_CHECK(
      THPVariable_Check(arg),
      "_rev_view_func expect a single argument that is a Tensor");
  const auto& new_view = THPVariable_Unpack(arg);

  // 确保 self 确实是一个可反向微分的视图
  // 如果不是，返回未定义的 Tensor（None），由用户处理
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // 执行视图重播操作
    TORCH_CHECK(view_info.has_view_fn(), "No _rev_view_func() found");
    out = view_info.rev_view_fn()(new_view);
  }
  // 将计算结果封装为 Python 对象返回
  return THPVariable_Wrap(std::move(out));
  END_HANDLE_TH_ERRORS
}

// 返回不安全的反向视图函数的实现结果，基于传入的参数
static PyObject* THPVariable_rev_view_func_unsafe(
    PyObject* self_,
    PyObject* arg) {
  return rev_view_func_impl(self_, arg);
}

// 实例化一个子类，其数据与原对象相同
static PyObject* THPVariable_as_subclass(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);
  static PythonArgParser parser({
      "as_subclass(PyObject* cls)",
  });
  ParsedArgs<1> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");
  // 使用给定的类型实例化一个新的子类对象，并返回
  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      self.alias(),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}
    // 定义一个函数，接受多个参数（包括位置参数和关键字参数）
    PyObject* kwargs) {
    // 处理可能发生的异常
      HANDLE_TH_ERRORS
      // 定义静态变量，表示 Python 参数解析器
      static PythonArgParser parser({
          "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False, *, c10::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, Device? device_for_backend_keys=None)",
      });
      // 定义一个用于存储解析后参数的对象，最多包含 7 个参数
      ParsedArgs<7> parsed_args{};
      // 使用解析器解析传入的位置参数和关键字参数，并存储在 parsed_args 中
      auto r = parser.parse(args, kwargs, parsed_args);
      // 获取第一个位置参数作为 Python 对象
      PyObject* cls = r.pyobject(0);
      // 检查 cls 是否为 Python 类型，若不是则抛出异常
      TORCH_CHECK_TYPE(
          PyType_Check(cls),
          "cls must be a type (got ",
          Py_TYPE(cls)->tp_name,
          ")");
      // 使用 guard 关闭 torch 的分发模式，而不只是从栈中弹出
      torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
      // 禁用 Python 调度程序
      c10::impl::DisablePythonDispatcher dpd_g;
      // 获取第二个位置参数作为 Tensor 对象，并将其分离（detach）以创建一个新的 Tensor
      auto data =
          r.tensor(1).detach(); // creates a fresh Tensor (DEFINITELY_UNINITIALIZED)
      // 设置 `data` 的 `allow_tensor_metadata_change` 为 true，以支持后续的元数据更改
      data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
      // 设置 `data` 是否需要梯度，根据第三个位置参数
      data.set_requires_grad(r.toBool(2));
      // 获取第四个位置参数作为可选的字符串视图
      const auto sizes_strides_policy = r.stringViewOptional(3);
      // 如果存在 sizes_strides_policy 参数，则设置 Tensor 的自定义大小和步长策略
      if (sizes_strides_policy.has_value()) {
        data.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
            parseSizesStridesPolicyArgument(*sizes_strides_policy));
      }
      // 如果第五个位置参数为 true，则设置 Tensor 的自定义设备标志
      if (r.toBool(4)) {
        data.unsafeGetTensorImpl()->set_python_custom_device(true);
      }
      // 如果第六个位置参数为 true，则设置 Tensor 的自定义布局标志
      if (r.toBool(5)) {
        data.unsafeGetTensorImpl()->set_python_custom_layout(true);
      }
      // 如果第七个位置参数不为空，则根据该参数改变 Tensor 的后端组件键
      if (!r.isNone(6)) {
        data.unsafeGetTensorImpl()->_change_backend_component_keys(r.device(6));
      }
    
      // 使用解析后的参数创建一个新的 THPVariable 对象，并返回
      return THPVariable_NewWithVar(
          (PyTypeObject*)cls,
          data,
          c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
      // 结束异常处理块
      END_HANDLE_TH_ERRORS
    }
static PyObject* THPVariable_make_wrapper_subclass(
    PyObject*,  // 第一个参数：不使用，应为nullptr
    PyObject* args,  // 第二个参数：Python 中传入的参数元组

    // AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
    // tracer::impl::NoTracerDispatchMode tracer_guard{};
    // 自动调度保护器和追踪器模式，用于管理自动分发和追踪模式

    auto sym_sizes = r.symintlist(1);  // 获取符号整数列表参数
    auto sym_strides_own = r.symintlistOptional(2);  // 获取可选的符号整数列表参数
    auto sym_strides =
        static_cast<std::optional<c10::SymIntArrayRef>>(sym_strides_own);  // 转换符号步长为可选类型
    auto sym_storage_offset = r.toSymIntOptional(3);  // 获取可选的符号存储偏移量

    c10::SymInt size_bytes;  // 符号整数类型，用于存储字节大小
    auto dtype_itemsize = static_cast<int64_t>(options.dtype().itemsize());  // 获取数据类型的每个元素大小
    auto storage_size = r.toSymIntOptional(14);  // 获取可选的符号存储大小

    if (storage_size.has_value()) {  // 如果存在存储大小值
      size_bytes = storage_size.value();  // 使用存储大小的值
    } else if (sym_strides.has_value()) {  // 否则，如果存在符号步长值
      size_bytes = at::detail::computeStorageNbytes(
          sym_sizes,
          sym_strides.value(),
          dtype_itemsize,
          sym_storage_offset.value_or(0));  // 计算存储字节大小（非连续）
    } else {
      size_bytes = at::detail::computeStorageNbytesContiguous(
          sym_sizes, dtype_itemsize, sym_storage_offset.value_or(0));  // 计算连续存储字节大小
    }

    // We use storages **only** to track aliasing of subclasses during tracing.
    // The actual data pointers are not valid.
    // 仅用于跟踪子类的别名，实际数据指针无效
    Storage storage{
        Storage::use_byte_size_t{},  // 使用字节大小
        size_bytes,  // 存储大小
        /*allocator=*/c10::GetAllocator(c10::kMeta),  // 分配器为元数据
        /*resizable=*/true};  // 可调整大小的存储对象

    // TODO: constructor should probably accept data pointer
    // 构造函数可能应该接受数据指针

    storage.set_data_ptr_noswap(at::DataPtr{nullptr, r.device(7)});  // 设置数据指针，设备为第七个设备

    auto keys = c10::DispatchKeySet({options.computeDispatchKey()});  // 获取分发键集合
    if (auto mb_extra_keys = r.toDispatchKeySetOptional(13)) {  // 如果存在额外的分发键
      keys = keys | *mb_extra_keys;  // 合并分发键集合
    }
    tensor = at::detail::make_tensor<TensorImpl>(  // 创建张量
        std::move(storage),  // 使用移动语义的存储对象
        keys,  // 分发键集合
        options.dtype());  // 数据类型选项

    TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();  // 获取张量实现指针

    if (sym_strides.has_value()) {  // 如果存在符号步长值
      tensor_impl->set_sizes_and_strides(  // 设置大小和步长
          sym_sizes, sym_strides.value(), sym_storage_offset);
    } else {
      TORCH_CHECK(
          !sym_storage_offset.has_value(),
          "setting storage offset without stride not supported");  // 异常检查：不支持在没有步长的情况下设置存储偏移量
      tensor_impl->generic_set_sizes_contiguous(sym_sizes);  // 设置连续的大小
    }

    const auto sizes_strides_policy = r.stringViewOptional(10);  // 获取可选的大小步长策略
    if (sizes_strides_policy.has_value()) {  // 如果存在大小步长策略
      tensor.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
          parseSizesStridesPolicyArgument(*sizes_strides_policy));  // 设置 Python 自定义大小步长
    }

    tensor.set_requires_grad(r.toBool(9));  // 设置是否需要梯度

    if (r.toBool(11)) {  // 如果需要第 11 个布尔值
      tensor.unsafeGetTensorImpl()->set_python_custom_device(true);  // 设置 Python 自定义设备
    }
    if (r.toBool(12)) {  // 如果需要第 12 个布尔值
      tensor.unsafeGetTensorImpl()->set_python_custom_layout(true);  // 设置 Python 自定义布局
    }

    return THPVariable_NewWithVar(  // 返回新的 THP 变量对象
        (PyTypeObject*)cls,  // Python 类型对象指针
        tensor,  // 张量对象
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);  // Python 解释器状态为“DEFINITELY_UNINITIALIZED”
    END_HANDLE_TH_ERRORS  // 结束处理 Torch 错误
}
PyObject* THPVariable_get_python_dispatch(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 解包 THPVariable 对象，获取其对应的 Tensor
  const auto& var = THPVariable_Unpack(self);
  // 返回一个包装了 Tensor 是否具有 Python 分发的 Python 对象
  return torch::autograd::utils::wrap(
      var.unsafeGetTensorImpl()->is_python_dispatch());
  END_HANDLE_TH_ERRORS
}

// CRTP base class to implement the python bindings for a Tensor property in
// PyTorch A class that implements a property is expected to have:
// - static constexpr const char* name;
//   - This variable should hold the Python name of the property
// - static Tensor fn(const Tensor&);
//   - This function calls the relevant ATen on the tensor
template <typename T>
struct GetterBase {
  static PyObject* getter(THPVariable* self, void* /*unused*/) {
    HANDLE_TH_ERRORS
    // 如果对象具有 Torch 函数，调用处理 Torch 函数的 getter 函数
    if (check_has_torch_function((PyObject*)self)) {
      return handle_torch_function_getter(self, T::name);
    }
    // 否则，使用传入的 Tensor 的 fn 函数进行封装，返回一个 THPVariable 对象
    return THPVariable_Wrap(T::fn(THPVariable_Unpack(self)));
    END_HANDLE_TH_ERRORS
  }
};

struct PropertyT : GetterBase<PropertyT> {
  static constexpr const char* name = "T";
  // 返回 Tensor 对象的 numpy_T() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.numpy_T();
  }
};

struct PropertyH : GetterBase<PropertyH> {
  static constexpr const char* name = "H";
  // 返回 Tensor 对象的 matrix_H() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.matrix_H();
  }
};

struct PropertymT : GetterBase<PropertymT> {
  static constexpr const char* name = "mT";
  // 返回 Tensor 对象的 mT() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.mT();
  }
};

struct PropertymH : GetterBase<PropertymH> {
  static constexpr const char* name = "mH";
  // 返回 Tensor 对象的 mH() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.mH();
  }
};

struct PropertyData : GetterBase<PropertyData> {
  static constexpr const char* name = "data";
  // 返回 Tensor 对象的 variable_data() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.variable_data();
  }
};

struct PropertyGrad : GetterBase<PropertyGrad> {
  static constexpr const char* name = "grad";
  // 返回 Tensor 对象的 grad() 方法的结果
  static Tensor fn(const Tensor& t) {
    return t.grad();
  }
};

struct PropertyReal : GetterBase<PropertyReal> {
  static constexpr const char* name = "real";
  // 返回 Tensor 对象的 real() 方法的结果
  static Tensor fn(const Tensor& t) {
    return at::real(t);
  }
};

struct PropertyImag : GetterBase<PropertyImag> {
  static constexpr const char* name = "imag";
  // 返回 Tensor 对象的 imag() 方法的结果
  static Tensor fn(const Tensor& t) {
    return at::imag(t);
  }
};

PyObject* THPVariable_get_cdata(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象具有 Torch 函数，调用处理 Torch 函数的 getter 函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_cdata");
  }
  // 否则，获取 THPVariable 对象对应的 Tensor 实现的指针，并将其转换为 Python 中的长整型对象
  const auto& var = THPVariable_Unpack(self);
  return PyLong_FromVoidPtr(var.unsafeGetTensorImpl());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_version(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象具有 Torch 函数，调用处理 Torch 函数的 getter 函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_version");
  }
  // 否则，获取 THPVariable 对象对应的版本号并返回作为 Python 中的整型对象
  const auto& var = THPVariable_Unpack(self);
  return PyInt_FromLong(var._version());
  END_HANDLE_TH_ERRORS
}
PyObject* THPVariable_get_grad_fn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象有torch函数，则调用处理torch函数的getter方法
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "grad_fn");
  }
  // 解包THPVariable对象，获取其对应的C++变量引用
  const auto& var = THPVariable_Unpack(self);
  // 如果变量没有梯度函数，返回Python中的None对象
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  // 将梯度函数转换成Python对象并返回
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_fn(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象有torch函数，则调用处理torch函数的setter方法
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "_grad_fn", obj);
  }
  // 检查设置的对象不为空，否则抛出错误信息
  TORCH_CHECK(obj, "Deletion of _grad_fn not allowed. Detach tensor instead!");
  // 检查设置的对象必须为None，否则抛出错误信息
  TORCH_CHECK(obj == Py_None, "_grad_fn can be only set to None");
  // 解包THPVariable对象并执行detach操作
  THPVariable_Unpack(self).detach_();
  // 返回成功状态码
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_is_leaf(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象有torch函数，则调用处理torch函数的getter方法
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_leaf");
  }
  // 返回一个布尔值表示变量是否为叶子节点（无梯度函数）
  return PyBool_FromLong(!THPVariable_Unpack(self).grad_fn());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_data(THPVariable* self, PyObject* data, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象有torch函数，则调用处理torch函数的setter方法
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "data", data);
  }
  // 检查数据对象不为空，否则抛出错误信息
  TORCH_CHECK(
      data, "Deleting tensor data is not allowed. Delete tensor instead!");
  // 检查数据对象必须为THPVariable类型，否则抛出类型错误信息
  TORCH_CHECK_TYPE(
      THPVariable_Check(data),
      "Variable data has to be a tensor, but got ",
      Py_TYPE(data)->tp_name);

  // 解包THPVariable对象，设置其数据为传入的数据对象
  THPVariable_Unpack(self).set_data(THPVariable_Unpack(data));
  // 返回成功状态码
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

int THPVariable_set_grad(THPVariable* self, PyObject* py_grad, void* unused) {
  HANDLE_TH_ERRORS
  // 如果对象有torch函数，则调用处理torch函数的setter方法
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "grad", py_grad);
  }
  // 解包THPVariable对象，获取其对应的C++变量引用
  const auto& var = THPVariable_Unpack(self);
  // 如果梯度为None或者未传入梯度对象，重置变量的可变梯度对象并返回成功状态码
  if (!py_grad || py_grad == Py_None) {
    var.mutable_grad().reset();
    return 0;
  }

  // 检查传入的梯度对象必须为THPVariable类型，否则抛出类型错误信息
  TORCH_CHECK_TYPE(
      THPVariable_Check(py_grad),
      "assigned grad expected to be a Tensor or None but got grad of type ",
      THPUtils_typename(py_grad));
  // 检查不能将变量自身作为其梯度对象
  TORCH_CHECK(
      self != (THPVariable*)py_grad, "can't assign Variable as its own grad");

  // 获取传入梯度对象的C++变量引用
  const auto& grad = THPVariable_Unpack(py_grad);
  // 检查梯度对象和变量的数据类型必须相同，否则抛出数据类型不匹配错误信息
  TORCH_CHECK(
      var.dtype() == grad.dtype(),
      "attempting to assign a gradient with dtype '",
      grad.dtype(),
      "' to a tensor with dtype '",
      var.dtype(),
      "'. Please ensure that the gradient and the tensor have the same dtype");
  // 检查梯度对象和变量必须在相同的设备上，否则抛出设备类型不匹配错误信息
  TORCH_CHECK(
      var.device().type() == grad.device().type(),
      "attempting to assign a gradient with device type '",
      grad.device().type(),
      "' to a tensor with device type '",
      var.device().type(),
      "'. Please ensure that the gradient and the tensor are on the same device");
  // 如果梯度对象的布局不是稀疏布局，则继续执行后续操作
  if (grad.layout() != kSparse) {
    # 检查梯度张量和变量张量的数据类型是否相同
    TORCH_CHECK(
        grad.options().type_equal(var.options()),
        "attempting to assign a gradient to a tensor that has data of a different type");
  }
  # 检查梯度张量和变量张量是否位于相同的设备上
  TORCH_CHECK(
      grad.get_device() == var.get_device(),
      "attempting to assign a gradient located on device with index '",
      grad.get_device(),
      "' to a tensor located on device with index '",
      var.get_device(),
      "'. Please ensure that the gradient and the tensor are on the same device");
  # 检查梯度张量和变量张量的符号尺寸是否相等
  TORCH_CHECK(
      grad.sym_sizes().equals(var.sym_sizes()),
      "attempting to assign a gradient of size '",
      grad.sym_sizes(),
      "' to a tensor of size '",
      var.sym_sizes(),
      "'. Please ensure that the gradient and the tensor are the same size");

  # 将梯度张量赋值给变量张量的可变梯度
  var.mutable_grad() = grad;
  # 返回操作成功的标志，表示没有错误发生
  return 0;
  # 处理 Torch 错误并返回 -1，结束当前函数
  END_HANDLE_TH_ERRORS_RET(-1)
PyObject* THPVariable_get_volatile(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "volatile");
  }
  // 若没有 Torch 函数重载，则发出用户警告，说明 volatile 已移除且始终为 False
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  auto r = PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  if (r != 0)
    throw python_error();
  // 返回 Python 中的 False
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_volatile(THPVariable* self, PyObject* obj, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "volatile", obj);
  }
  // 若没有 Torch 函数重载，则发出用户警告，使用预定义的 VOLATILE_WARNING
  auto r = PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
  if (r != 0)
    throw python_error();
  // 返回 0 表示成功
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable_get_output_nr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "output_nr");
  }
  // 否则获取输出编号并转换为 Python 的长整型返回
  const auto output_nr =
      static_cast<long>(THPVariable_Unpack(self).output_nr());
  return PyInt_FromLong(output_nr);
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_requires_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "requires_grad");
  }
  // 否则检查 Tensor 是否需要梯度，返回相应的 Python 布尔值
  if (THPVariable_Unpack(self).requires_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_retains_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "retains_grad");
  }
  // 否则检查 Tensor 是否保留梯度，返回相应的 Python 布尔值
  if (THPVariable_Unpack(self).retains_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_ndim(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "ndim");
  }
  // 否则获取 Tensor 的维度并转换为 Python 的长整型返回
  return PyInt_FromLong(THPVariable_Unpack(self).dim());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_names(PyObject* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用对应的处理函数
  if (check_has_torch_function(self)) {
    return handle_torch_function_getter((THPVariable*)self, "names");
  }
  // 否则返回 Tensor 的维度名称列表（目前是字符串列表的临时实现）
  const auto& tensor = THPVariable_Unpack(self);
  auto size = tensor.dim();
  THPObjectPtr tuple(PyTuple_New(size));
  if (!tuple)
    throw python_error();

  const auto dimnames = tensor.names();
  for (const auto i : c10::irange(size)) {
    // 逐个创建 Python 字符串对象并添加到元组中
    PyObject* str = nullptr;
    ```
    str = PyUnicode_FromString(dimnames[i].c_str());
    ```
    PyTuple_SET_ITEM(tuple.get(), i, str);
  }

  return tuple.release();
  END_HANDLE_TH_ERRORS
}
    if (dimnames[i].type() == at::NameType::WILDCARD) {
      // 检查当前维度名称是否为通配符类型（WILDCARD）
      // PyTuple_SET_ITEM 会“偷走”对象的引用。当元组被释放时，
      // 它会减少对 Py_None 的引用计数，这可能导致问题。
      // 为了避免这种情况，我们通过增加引用计数来“创建”对 Py_None 的新引用。
      // 参考文档:
      // - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
      // - https://stackoverflow.com/questions/16400600/how-to-return-a-tuple-containing-a-none-value-from-the-c-api
      Py_INCREF(Py_None);
      // 将 Py_None 赋值给 str
      str = Py_None;
    } else {
      // 如果维度名称不是通配符类型，则将其转换为非限定字符串形式并打包
      str = THPUtils_packString(dimnames[i].symbol().toUnqualString());
      // 如果打包失败则抛出异常
      if (!str)
        throw python_error();
    }
    // 将 str 设置为元组 tuple 的第 i 个元素
    PyTuple_SET_ITEM(tuple.get(), i, str);
  }
  // 返回元组并释放其所有权
  return tuple.release();
  // 处理 Torch 引发的异常并结束处理错误
  END_HANDLE_TH_ERRORS
}

// 设置变量的名称，对应 Python 中的 names 属性
int THPVariable_set_names(PyObject* self, PyObject* names, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用处理函数
  if (check_has_torch_function(self)) {
    return handle_torch_function_setter((THPVariable*)self, "names", names);
  }
  // 解包 Python 对象为 C++ Tensor 变量
  const auto& var = THPVariable_Unpack(self);
  // 如果 names 是 None，则清除变量的命名信息
  if (names == Py_None) {
    at::internal_set_names_inplace(var, at::nullopt);
  } else {
    // 否则检查 names 是否符合 Dimname 列表的要求
    TORCH_CHECK(
        THPUtils_checkDimnameList(names),
        "names must either be None or a tuple of dim names");
    // 解析 names 并设置变量的命名信息
    at::internal_set_names_inplace(var, torch::parseDimnameList(names));
  }
  // 返回操作成功的标志
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 设置变量的 requires_grad 属性
int THPVariable_set_requires_grad(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否有 Torch 函数重载，如果有则调用处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "requires_grad", obj);
  }
  // 检查 obj 是否为布尔值
  TORCH_CHECK(obj && PyBool_Check(obj), "requires_grad must be a bool");
  // 解包 Python 对象为 C++ Tensor 变量
  const auto& var = THPVariable_Unpack(self);
  // 根据 obj 设置 requires_grad 属性
  auto requires_grad = (obj == Py_True);
  // 如果变量不是叶子节点，则报错
  if (!var.is_leaf()) {
    THPUtils_setError(
        autograd::utils::requires_grad_leaf_error(obj == Py_True).c_str());
    return -1;
  }
  // 如果需要设置 requires_grad，并且变量类型不支持梯度，则报错
  if (requires_grad &&
      !isDifferentiableType(at::typeMetaToScalarType((var.dtype())))) {
    THPUtils_setError(
        "only Tensors of floating point and complex dtype can require gradients");
    return -1;
  }
  // 设置 requires_grad 属性
  var.set_requires_grad(requires_grad);
  // 返回操作成功的标志
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 获取变量的名称属性
PyObject* THPVariable_get_name(THPVariable* self, void* unused) {
  // 如果有 Torch 函数重载，则调用处理函数
  if (check_has_torch_function((PyObject*)self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function_getter(self, "name");
    END_HANDLE_TH_ERRORS
  }
  // 解包 Python 对象为 C++ Tensor 变量
  const auto& tensor = THPVariable_Unpack(self);
  // 如果变量没有命名，则返回 None
  if (tensor.name().empty())
    Py_RETURN_NONE;
  // 否则返回变量的名称
  return THPUtils_packString(tensor.name().c_str());
}

// 获取变量的 _backward_hooks 属性
PyObject* THPVariable_get_backwards_hooks(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 如果有 Torch 函数重载，则调用处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_backward_hooks");
  }
  // 如果 self->backward_hooks 不为空，则增加其引用计数并返回
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  // 否则返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 设置变量的 _backward_hooks 属性
int THPVariable_set_backwards_hooks(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  // 如果有 Torch 函数重载，则调用处理函数
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "_backward_hooks", obj);
  }
  // 检查 obj 是否为空，如果是则报错
  TORCH_CHECK(obj, "Deletion of _backwards_hooks not allowed!");
  // 如果 obj 是 None，则设置 self->backward_hooks 为空
  if (obj == Py_None) {
    obj = nullptr;
  }
  // 增加 obj 的引用计数，减少 self->backward_hooks 的引用计数
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  // 解包 Python 对象为 C++ Tensor 变量
  const auto& tensor = THPVariable_Unpack(self);
  // 清除变量的 hooks
  torch::autograd::impl::clear_hooks(tensor);
  // 如果 obj 不为空，则添加 hook
  if (obj) {
    torch::autograd::impl::add_hook(
        tensor, std::make_unique<PyFunctionTensorPreHook>(obj, 0));
  }
  // 返回操作成功的标志
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 获取变量的 post_accumulate_grad_hooks 属性
PyObject* THPVariable_get_post_accumulate_grad_hooks(
    THPVariable* self,
    void* unused) {
```  
# 接受一个名为unused的void指针作为参数

  HANDLE_TH_ERRORS
```  
# 宏：处理异常，可能是一个错误处理宏

  if (check_has_torch_function((PyObject*)self)) {
```  
# 如果self对象有torch函数，检查是否存在torch函数处理

    return handle_torch_function_getter(self, "_post_accumulate_grad_hooks");
```  
# 调用处理torch函数的getter函数，返回"_post_accumulate_grad_hooks"

  }
```  
# 如果没有torch函数，则执行以下语句

  if (self->post_accumulate_grad_hooks) {
```  
# 如果self对象中的post_accumulate_grad_hooks不为空（即已经分配了对象）

    Py_INCREF(self->post_accumulate_grad_hooks);
```  
# 增加post_accumulate_grad_hooks的引用计数

    return self->post_accumulate_grad_hooks;
```  
# 返回post_accumulate_grad_hooks对象

  }
```  
# 如果post_accumulate_grad_hooks为空，执行以下语句

  Py_RETURN_NONE;
```  
# 返回None对象（Python的None）

  END_HANDLE_TH_ERRORS
```  
# 结束异常处理宏
}

// 设置变量的 _post_accumulate_grad_hooks 属性的后处理挂钩
int THPVariable_set_post_accumulate_grad_hooks(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数设置器
    return handle_torch_function_setter(
        self, "_post_accumulate_grad_hooks", obj);
  }
  // 检查 _post_accumulate_grad_hooks 是否为空
  TORCH_CHECK(obj, "Deletion of _post_accumulate_grad_hooks not allowed!");
  // 如果 obj 是 Py_None，则设置为 nullptr
  if (obj == Py_None) {
    obj = nullptr;
  }
  // 增加 obj 的引用计数
  Py_XINCREF(obj);
  // 清除 self->post_accumulate_grad_hooks 的引用并设置为 obj
  Py_CLEAR(self->post_accumulate_grad_hooks);
  self->post_accumulate_grad_hooks = obj;
  // 获取 tensor 对象
  const auto& tensor = THPVariable_Unpack(self);
  // 如果 obj 不为空，则设置后处理累积梯度挂钩
  if (obj) {
    torch::autograd::impl::set_post_acc_grad_hooks(
        tensor, std::make_unique<PyFunctionTensorPostAccGradHooks>(obj));
  }
  // 返回成功状态
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 获取变量的 _base 属性
PyObject* THPVariable_get_base(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "_base");
  }
  // 获取 tensor 对象
  const auto& tensor = THPVariable_Unpack(self);
  // 如果 tensor 是视图，则返回其基本 tensor
  if (tensor.is_view()) {
    return THPVariable_Wrap(tensor._base());
  }
  // 否则返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 获取变量的 shape 属性
PyObject* THPVariable_get_shape(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "shape");
  }
  // 返回变量的尺寸信息
  return THPSize_NewFromSymSizes(THPVariable_Unpack(self));
  END_HANDLE_TH_ERRORS
}

// 获取变量的 is_cpu 属性
PyObject* THPVariable_is_cpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "is_cpu");
  }
  // 获取 tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回 CPU 是否为 True 的包装结果
  return torch::autograd::utils::wrap(self_.is_cpu());
  END_HANDLE_TH_ERRORS
}

// 获取变量的 is_cuda 属性
PyObject* THPVariable_is_cuda(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "is_cuda");
  }
  // 获取 tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回 CUDA 是否为 True 的包装结果
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

// 获取变量的 is_mtia 属性
PyObject* THPVariable_is_mtia(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "is_mtia");
  }
  // 获取 tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回 MTIA 是否为 True 的包装结果
  return torch::autograd::utils::wrap(self_.is_mtia());
  END_HANDLE_TH_ERRORS
}

// 获取变量的 is_xla 属性
PyObject* THPVariable_is_xla(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "is_xla");
  }
  // 获取 tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回 XLA 是否为 True 的包装结果
  return torch::autograd::utils::wrap(self_.is_xla());
  END_HANDLE_TH_ERRORS
}

// 获取变量的 is_ipu 属性
PyObject* THPVariable_is_ipu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  // 检查是否具有 torch 函数
  if (check_has_torch_function((PyObject*)self)) {
    // 调用处理 torch 函数获取器
    return handle_torch_function_getter(self, "is_ipu");
  }
  // 获取 tensor 对象
  auto& self_ = THPVariable_Unpack(self);
  // 返回 IPU 是否为 True 的包装结果
  return torch::autograd::utils::wrap(self_.is_ipu());
  END_HANDLE_TH_ERRORS
}
// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_xpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_xpu");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_xpu() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_xpu());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_sparse(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_sparse() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_sparse_csr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse_csr");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_sparse_csr() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_sparse_csr());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_mkldnn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mkldnn");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_mkldnn() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_mkldnn());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_mps(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mps");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_mps() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_mps());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_maia(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_maia");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_maia() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_maia());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_vulkan(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_vulkan");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_vulkan() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_vulkan());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_quantized(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_quantized");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_quantized() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_quantized());
  END_HANDLE_TH_ERRORS
}

// 检查是否有 Torch 函数重载，如果有，则调用 Torch 函数处理
PyObject* THPVariable_is_meta(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_meta");
  }
  // 解包 THPVariable 对象为 self_
  auto& self_ = THPVariable_Unpack(self);
  // 返回 self_.is_meta() 的结果，使用 torch::autograd::utils::wrap 进行封装
  return torch::autograd::utils::wrap(self_.is_meta());
  END_HANDLE_TH_ERRORS
}
// 检查是否具有 Torch 函数的自定义实现，如果有，则返回该函数的结果
PyObject* THPVariable_is_complex(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_complex");
  }
  // 解包 THPVariable 对象，获得底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 返回 Tensor 是否为复数的包装结果
  return torch::autograd::utils::wrap(self_.is_complex());
  END_HANDLE_TH_ERRORS
}

// 检查是否具有 Torch 函数的自定义实现，如果有，则返回该函数的结果
PyObject* THPVariable_is_nested(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_nested");
  }
  // 解包 THPVariable 对象，获得底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 返回 Tensor 是否为嵌套形式的包装结果
  return torch::autograd::utils::wrap(self_.is_nested());
  END_HANDLE_TH_ERRORS
}

// 检查 Tensor 是否具有符号大小和步幅，返回结果的包装
PyObject* THPVariable_has_symbolic_sizes_strides(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  // 解包 THPVariable 对象，获得底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 返回底层 Tensor 实现是否具有符号大小和步幅的包装结果
  return torch::autograd::utils::wrap(
      self_.unsafeGetTensorImpl()->has_symbolic_sizes_strides());
  END_HANDLE_TH_ERRORS
}

// 返回 Tensor 的数据类型（dtype），如果有 Torch 函数的自定义实现，则返回该函数的结果
static PyObject* THPVariable_dtype(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "dtype");
  }
  // 解包 THPVariable 对象，获得底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 返回 Tensor 的数据类型（dtype）的包装结果
  return torch::autograd::utils::wrap(self_.scalar_type());
  END_HANDLE_TH_ERRORS
}

// 返回 Tensor 的布局（layout），如果有 Torch 函数的自定义实现，则返回该函数的结果
static PyObject* THPVariable_layout(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "layout");
  }
  // 解包 THPVariable 对象，获得底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 返回 Tensor 的布局（layout）的包装结果
  return torch::autograd::utils::wrap(self_.layout());
  END_HANDLE_TH_ERRORS
}

// 返回 Tensor 的设备（device），如果有 Torch 函数的自定义实现，则返回该函数的结果
static PyObject* THPVariable_device(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "device");
  }
  // 返回 Tensor 的设备（device）的包装结果
  return THPDevice_New(THPVariable_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

// 返回 Tensor 的字节大小（nbytes），如果有 Torch 函数的自定义实现，则返回该函数的结果
static PyObject* THPVariable_get_nbytes(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "nbytes");
  }
  // 返回 Tensor 的字节大小（nbytes）的包装结果
  return PyLong_FromSize_t(THPVariable_Unpack(self).nbytes());
  END_HANDLE_TH_ERRORS
}

// 返回 Tensor 的元素大小（itemsize），如果有 Torch 函数的自定义实现，则返回该函数的结果
static PyObject* THPVariable_get_itemsize(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "itemsize");
  }
  // 返回 Tensor 的元素大小（itemsize）的包装结果
  return PyLong_FromSize_t(THPVariable_Unpack(self).itemsize());
  END_HANDLE_TH_ERRORS
}

// 设置 Tensor 的实部，如果有异常则处理并返回 -1
int THPVariable_set_real(PyObject* self, PyObject* real, void* unused) {
  HANDLE_TH_ERRORS
  // 解包 Python 对象 self，并获取其底层 Tensor 对象的引用
  auto& self_ = THPVariable_Unpack(self);
  // 获取 self 的实部作为 Tensor
  auto self_real = at::real(self_);
  // 将输入的 real 转换为与 self_real 兼容的 Tensor
  auto real_ = valueToTensor(self_real.options(), real, self_real.device());
  {
    // 在 GIL 范围外释放 GIL，执行实际的张量复制操作
    pybind11::gil_scoped_release no_gil;
    // 将 real_ 的内容复制到 self_real 中
    self_real.copy_(real_);
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}
int THPVariable_set_imag(PyObject* self, PyObject* imag, void* unused) {
  HANDLE_TH_ERRORS
  // 解包 Python 对象 self，获取对应的 C++ Tensor 对象 self_
  auto& self_ = THPVariable_Unpack(self);
  // 获取 self_ 的虚部作为新的 Tensor 对象 self_imag
  auto self_imag = at::imag(self_);
  // 将传入的 imag 转换为 Tensor 对象 imag_
  auto imag_ = valueToTensor(self_imag.options(), imag, self_imag.device());
  {
    // 释放 GIL，允许 Python 多线程执行
    pybind11::gil_scoped_release no_gil;
    // 将 imag_ 的值复制给 self_imag
    self_imag.copy_(imag_);
    // 返回成功标志
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable__use_count(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // 解包 Python 对象 self，获取对应的 C++ Tensor 对象 t
  const auto& t = THPVariable_Unpack(self);
  // 将 t 的引用计数转换为 Python 的 unsigned long 对象，并打包返回
  return THPUtils_packUInt64(t.use_count());
  END_HANDLE_TH_ERRORS
}

// properties are registered here because we are currently only able to bind
// them manually. TODO: make declarable in native_functions
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPVariable_properties[] = {
    {"_python_dispatch",
     (getter)THPVariable_get_python_dispatch,
     nullptr,
     nullptr,
     nullptr},
    {"T", (getter)PropertyT::getter, nullptr, nullptr, nullptr},
    {"H", (getter)PropertyH::getter, nullptr, nullptr, nullptr},
    {"mT", (getter)PropertymT::getter, nullptr, nullptr, nullptr},
    {"mH", (getter)PropertymH::getter, nullptr, nullptr, nullptr},
    {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
    {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
    {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
    {"_grad_fn",
     (getter)THPVariable_get_grad_fn,
     (setter)THPVariable_set_grad_fn,
     nullptr,
     nullptr},
    {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
    {"retains_grad",
     (getter)THPVariable_retains_grad,
     nullptr,
     nullptr,
     nullptr},
    {"data",
     (getter)PropertyData::getter,
     (setter)THPVariable_set_data,
     nullptr,
     nullptr},
    {"_grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr}, // 允许 Python 类覆盖 .grad
    {"grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr},
    {"_base", (getter)THPVariable_get_base, nullptr, nullptr, nullptr},
    {"volatile",
     (getter)THPVariable_get_volatile,
     (setter)THPVariable_set_volatile,
     nullptr,
     nullptr},
    {"output_nr", (getter)THPVariable_get_output_nr, nullptr, nullptr, nullptr},
    {"requires_grad",
     (getter)THPVariable_get_requires_grad,
     (setter)THPVariable_set_requires_grad,
     nullptr,
     nullptr},
    {"_backward_hooks",
     (getter)THPVariable_get_backwards_hooks,
     (setter)THPVariable_set_backwards_hooks,
     nullptr,
     nullptr},
    {"_post_accumulate_grad_hooks",
     (getter)THPVariable_get_post_accumulate_grad_hooks,
     (setter)THPVariable_set_post_accumulate_grad_hooks,
     nullptr,
     nullptr},
    {"name", (getter)THPVariable_get_name, nullptr, nullptr, nullptr},
};
    {"shape", (getter)THPVariable_get_shape, nullptr, nullptr, nullptr},
    // 键为 "shape"，值为 THPVariable_get_shape 函数的指针，无 setter 或 deleter
    {"is_cuda", (getter)THPVariable_is_cuda, nullptr, nullptr, nullptr},
    // 键为 "is_cuda"，值为 THPVariable_is_cuda 函数的指针，无 setter 或 deleter
    {"is_mtia", (getter)THPVariable_is_mtia, nullptr, nullptr, nullptr},
    // 键为 "is_mtia"，值为 THPVariable_is_mtia 函数的指针，无 setter 或 deleter
    {"is_cpu", (getter)THPVariable_is_cpu, nullptr, nullptr, nullptr},
    // 键为 "is_cpu"，值为 THPVariable_is_cpu 函数的指针，无 setter 或 deleter
    {"is_xla", (getter)THPVariable_is_xla, nullptr, nullptr, nullptr},
    // 键为 "is_xla"，值为 THPVariable_is_xla 函数的指针，无 setter 或 deleter
    {"is_xpu", (getter)THPVariable_is_xpu, nullptr, nullptr, nullptr},
    // 键为 "is_xpu"，值为 THPVariable_is_xpu 函数的指针，无 setter 或 deleter
    {"is_ipu", (getter)THPVariable_is_ipu, nullptr, nullptr, nullptr},
    // 键为 "is_ipu"，值为 THPVariable_is_ipu 函数的指针，无 setter 或 deleter
    {"is_sparse", (getter)THPVariable_is_sparse, nullptr, nullptr, nullptr},
    // 键为 "is_sparse"，值为 THPVariable_is_sparse 函数的指针，无 setter 或 deleter
    {"is_sparse_csr",
     (getter)THPVariable_is_sparse_csr,
     nullptr,
     nullptr,
     nullptr},
    // 键为 "is_sparse_csr"，值为 THPVariable_is_sparse_csr 函数的指针，无 setter 或 deleter
    {"is_mkldnn", (getter)THPVariable_is_mkldnn, nullptr, nullptr, nullptr},
    // 键为 "is_mkldnn"，值为 THPVariable_is_mkldnn 函数的指针，无 setter 或 deleter
    {"is_mps", (getter)THPVariable_is_mps, nullptr, nullptr, nullptr},
    // 键为 "is_mps"，值为 THPVariable_is_mps 函数的指针，无 setter 或 deleter
    {"is_maia", (getter)THPVariable_is_maia, nullptr, nullptr, nullptr},
    // 键为 "is_maia"，值为 THPVariable_is_maia 函数的指针，无 setter 或 deleter
    {"is_vulkan", (getter)THPVariable_is_vulkan, nullptr, nullptr, nullptr},
    // 键为 "is_vulkan"，值为 THPVariable_is_vulkan 函数的指针，无 setter 或 deleter
    {"is_complex", (getter)THPVariable_is_complex, nullptr, nullptr, nullptr},
    // 键为 "is_complex"，值为 THPVariable_is_complex 函数的指针，无 setter 或 deleter
    {"is_quantized",
     (getter)THPVariable_is_quantized,
     nullptr,
     nullptr,
     nullptr},
    // 键为 "is_quantized"，值为 THPVariable_is_quantized 函数的指针，无 setter 或 deleter
    {"is_meta", (getter)THPVariable_is_meta, nullptr, nullptr, nullptr},
    // 键为 "is_meta"，值为 THPVariable_is_meta 函数的指针，无 setter 或 deleter
    {"is_nested", (getter)THPVariable_is_nested, nullptr, nullptr, nullptr},
    // 键为 "is_nested"，值为 THPVariable_is_nested 函数的指针，无 setter 或 deleter
    {"_has_symbolic_sizes_strides",
     (getter)THPVariable_has_symbolic_sizes_strides,
     nullptr,
     nullptr,
     nullptr},
    // 键为 "_has_symbolic_sizes_strides"，值为 THPVariable_has_symbolic_sizes_strides 函数的指针，无 setter 或 deleter
    {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
    // 键为 "dtype"，值为 THPVariable_dtype 函数的指针，无 setter 或 deleter
    {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
    // 键为 "layout"，值为 THPVariable_layout 函数的指针，无 setter 或 deleter
    {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
    // 键为 "device"，值为 THPVariable_device 函数的指针，无 setter 或 deleter
    {"ndim", (getter)THPVariable_get_ndim, nullptr, nullptr, nullptr},
    // 键为 "ndim"，值为 THPVariable_get_ndim 函数的指针，无 setter 或 deleter
    {"nbytes", (getter)THPVariable_get_nbytes, nullptr, nullptr, nullptr},
    // 键为 "nbytes"，值为 THPVariable_get_nbytes 函数的指针，无 setter 或 deleter
    {"itemsize", (getter)THPVariable_get_itemsize, nullptr, nullptr, nullptr},
    // 键为 "itemsize"，值为 THPVariable_get_itemsize 函数的指针，无 setter 或 deleter
    {"names",
     (getter)THPVariable_get_names,
     (setter)THPVariable_set_names,
     nullptr,
     nullptr},
    // 键为 "names"，getter 为 THPVariable_get_names 函数的指针，setter 为 THPVariable_set_names 函数的指针，无 deleter
    {"real",
     (getter)PropertyReal::getter,
     (setter)THPVariable_set_real,
     nullptr,
     nullptr},
    // 键为 "real"，getter 为 PropertyReal::getter 函数的指针，setter 为 THPVariable_set_real 函数的指针，无 deleter
    {"imag",
     (getter)PropertyImag::getter,
     (setter)THPVariable_set_imag,
     nullptr,
     nullptr},
    // 键为 "imag"，getter 为 PropertyImag::getter 函数的指针，setter 为 THPVariable_set_imag 函数的指针，无 deleter
    {nullptr}};
    // 最后一个条目，以 nullptr 结尾，表示字典结束
static PyMappingMethods THPVariable_as_mapping = {
    THPVariable_length,            // 指向长度函数的指针
    THPVariable_getitem,           // 指向获取元素函数的指针
    THPVariable_setitem,           // 指向设置元素函数的指针
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef extra_methods[] = {
    {"as_subclass",
     castPyCFunctionWithKeywords(THPVariable_as_subclass),  // 转换为子类的函数指针
     METH_VARARGS | METH_KEYWORDS,  // 方法接受可变位置参数和关键字参数
     nullptr},                      // 方法的文档字符串为空

    {"_make_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_subclass),  // 创建子类的静态方法指针
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,  // 静态方法接受可变位置参数和关键字参数
     nullptr},                      // 方法的文档字符串为空

    {"_make_wrapper_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_wrapper_subclass),  // 创建包装子类的静态方法指针
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,  // 静态方法接受可变位置参数和关键字参数
     nullptr},                      // 方法的文档字符串为空

    {"_fix_weakref", THPVariable_fix_weakref, METH_NOARGS, nullptr},  // 修复弱引用的方法指针，不接受参数

    {"_view_func",
     castPyCFunctionWithKeywords(THPVariable_view_func),  // 视图函数的函数指针
     METH_VARARGS | METH_KEYWORDS,  // 方法接受可变位置参数和关键字参数
     nullptr},                      // 方法的文档字符串为空

    {"_view_func_unsafe",
     castPyCFunctionWithKeywords(THPVariable_view_func_unsafe),  // 不安全视图函数的函数指针
     METH_VARARGS | METH_KEYWORDS,  // 方法接受可变位置参数和关键字参数
     nullptr},                      // 方法的文档字符串为空

    {"_rev_view_func_unsafe",
     THPVariable_rev_view_func_unsafe,  // 不安全反向视图函数的函数指针
     METH_O,                         // 方法接受一个对象参数
     nullptr},                      // 方法的文档字符串为空

    {"_use_count", THPVariable__use_count, METH_NOARGS, nullptr},  // 获取使用计数的方法指针，不接受参数

    {nullptr}  // 结束方法定义的标志
};

struct THPVariableMeta {
  PyHeapTypeObject base;
};

int THPVariableMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs);  // 初始化类型的函数声明

PyTypeObject THPVariableMetaType = {
    PyVarObject_HEAD_INIT(
        DEFERRED_ADDRESS(&PyType_Type),
        0) "torch._C._TensorMeta", /* tp_name */
    sizeof(THPVariableMeta),  // 类型对象的基本大小
    0,  // 不需要额外分配的内存大小
    nullptr,  // 释放对象的函数指针为空
    0,  // 没有矢量调用偏移
    nullptr,  // 获取属性的函数指针为空
    nullptr,  // 设置属性的函数指针为空
    nullptr,  // 保留字段为空
    nullptr,  // 对象的字符串表示为空
    nullptr,  // 数值运算方法为空
    nullptr,  // 序列方法为空
    nullptr,  // 映射方法为空
    nullptr,  // 哈希方法为空
    nullptr,  // 调用对象为空
    nullptr,  // 转换为字符串的方法为空
    nullptr,  // 获取属性为空
    nullptr,  // 设置属性为空
    nullptr,  // 缓冲区接口为空
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // 类型对象的标志
    nullptr,  // 类型对象的文档字符串为空
    nullptr,  // 遍历对象的函数为空
    nullptr,  // 清除对象的函数为空
    nullptr,  // 对象的丰富比较为空
    0,  // 弱引用列表偏移为0
    nullptr,  // 迭代器为空
    nullptr,  // 下一个迭代器为空
    nullptr,  // 方法为空
    nullptr,  // 成员为空
    nullptr,  // 获取和设置的方法为空
    DEFERRED_ADDRESS(&PyType_Type),  // 基类为延迟地址的 Python 类型对象
    nullptr,  // 类型字典为空
    nullptr,  // 获取描述符为空
    nullptr,  // 设置描述符为空
    0,  // 字典偏移为0
    THPVariableMetaType_init,  // 初始化函数指针
    nullptr,  // 分配对象的函数指针为空
    nullptr,  // 新建对象的函数指针为空
};

PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C.TensorBase", /* tp_name */
    sizeof(THPVariable),  // 类型对象的基本大小
    0,  // 不需要额外分配的内存大小
    // 创建 THPVariableType 类型对象时，不应直接分配 tp_dealloc，而应由元类适当设置子类的 tp_dealloc
    nullptr, /* tp_dealloc */
    
    // 指定 tp_vectorcall_offset 为 0，表示未启用向量调用
    0, /* tp_vectorcall_offset */
    
    // 指定 tp_getattr 为 nullptr，表示对象没有自定义的 getattr 方法
    nullptr, /* tp_getattr */
    
    // 指定 tp_setattr 为 nullptr，表示对象没有自定义的 setattr 方法
    nullptr, /* tp_setattr */
    
    // 指定 tp_reserved 为 nullptr，保留字段，当前未使用
    nullptr, /* tp_reserved */
    
    // 指定 tp_repr 为 nullptr，表示对象没有自定义的 repr 方法
    nullptr, /* tp_repr */
    
    // 指定 tp_as_number 为 nullptr，表示对象没有数值类型方法集合
    nullptr, /* tp_as_number */
    
    // 指定 tp_as_sequence 为 nullptr，表示对象没有序列类型方法集合
    nullptr, /* tp_as_sequence */
    
    // 指定 tp_as_mapping 为 THPVariable_as_mapping，表示对象使用 THPVariable_as_mapping 作为映射类型方法集合
    &THPVariable_as_mapping, /* tp_as_mapping */
    
    // 指定 tp_hash 为 nullptr，表示对象没有自定义的哈希方法
    nullptr, /* tp_hash */
    
    // 指定 tp_call 为 nullptr，表示对象不可调用
    nullptr, /* tp_call */
    
    // 指定 tp_str 为 nullptr，表示对象没有自定义的字符串表示方法
    nullptr, /* tp_str */
    
    // 指定 tp_getattro 为 nullptr，表示对象没有自定义的 getattr 方法（通过名称获取属性）
    nullptr, /* tp_getattro */
    
    // 指定 tp_setattro 为 nullptr，表示对象没有自定义的 setattr 方法（通过名称设置属性）
    nullptr, /* tp_setattro */
    
    // 指定 tp_as_buffer 为 nullptr，表示对象不支持缓冲区接口
    nullptr, /* tp_as_buffer */
    
    // 设置 tp_flags 为 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC，指定对象类型的标志
    // Py_TPFLAGS_DEFAULT 表示默认类型
    // Py_TPFLAGS_BASETYPE 表示基本类型
    // Py_TPFLAGS_HAVE_GC 表示对象使用垃圾回收
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    
    // 指定 tp_doc 为 nullptr，表示对象没有文档字符串
    nullptr, /* tp_doc */
    
    // 指定 tp_traverse 为 (traverseproc)THPFunction_traverse，表示对象使用 THPFunction_traverse 进行遍历
    (traverseproc)THPFunction_traverse, /* tp_traverse */
    
    // 指定 tp_clear 为 (inquiry)THPVariable_clear，表示对象使用 THPVariable_clear 进行清理
    (inquiry)THPVariable_clear, /* tp_clear */
    
    // 指定 tp_richcompare 为 nullptr，表示对象没有自定义的富比较方法
    nullptr, /* tp_richcompare */
    
    // 指定 tp_weaklistoffset 为 0，表示对象没有弱引用列表
    0, /* tp_weaklistoffset */
    
    // 指定 tp_iter 为 nullptr，表示对象不可迭代
    nullptr, /* tp_iter */
    
    // 指定 tp_iternext 为 nullptr，表示对象不支持迭代器的 next 方法
    nullptr, /* tp_iternext */
    
    // 指定 tp_methods 为 nullptr，表示对象没有自定义的方法集合
    nullptr, /* tp_methods */
    
    // 指定 tp_members 为 nullptr，表示对象没有成员（数据）集合
    nullptr, /* tp_members */
    
    // 指定 tp_getset 为 THPVariable_properties，表示对象的属性（get/set 方法）使用 THPVariable_properties
    THPVariable_properties, /* tp_getset */
    
    // 指定 tp_base 为 nullptr，表示对象没有基类
    nullptr, /* tp_base */
    
    // 指定 tp_dict 为 nullptr，表示对象没有自定义的字典属性
    nullptr, /* tp_dict */
    
    // 指定 tp_descr_get 为 nullptr，表示对象没有自定义的描述符获取方法
    nullptr, /* tp_descr_get */
    
    // 指定 tp_descr_set 为 nullptr，表示对象没有自定义的描述符设置方法
    nullptr, /* tp_descr_set */
    
    // 指定 tp_dictoffset 为 0，表示对象没有字典偏移量
    0, /* tp_dictoffset */
    
    // 指定 tp_init 为 nullptr，表示对象没有自定义的初始化方法
    nullptr, /* tp_init */
    
    // 指定 tp_alloc 为 nullptr，表示对象使用默认的内存分配方法
    nullptr, /* tp_alloc */
    
    // 指定 tp_new 为 THPVariable_pynew，表示对象使用 THPVariable_pynew 作为新对象创建方法
    // 注意：虽然在这里提供了 tp_new，但不能直接使用 cls == THPVariableMeta 调用它，而应先创建其子类，然后构造对象
    THPVariable_pynew, /* tp_new */
};

// 定义 THPVariable_pynew 函数，用于创建 Python 对象的新实例
PyObject* THPVariable_pynew(
    PyTypeObject* type,  // Python 类型对象指针
    PyObject* args,      // 参数元组
    PyObject* kwargs) {  // 关键字参数字典
  HANDLE_TH_ERRORS      // 错误处理开始
  TORCH_CHECK(
      type != &THPVariableType,
      "Cannot directly construct TensorBase; subclass it and then construct that");  // 检查类型是否为 THPVariableType 的子类
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);  // 发出跟踪警告
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);  // 调用基础张量构造函数
  // 警告：tensor 不保证是新的张量；例如，如果给定一个原始指针，它可能会增加引用计数
  // 注意：base_tensor_ctor 可能会调用分派的 ATen 函数（如 alias()、lift_fresh()），这些函数可能返回张量的子类，我们允许直接传递它们
  return THPVariable_NewWithVar(
      type,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED,
      /*allow_preexisting_pyobj=*/true);  // 使用给定参数创建新的 THPVariable 对象并返回
  END_HANDLE_TH_ERRORS  // 错误处理结束
}

// 注意：这不是 THPVariable 的 tp_dealloc，而是子类的 dealloc。因为无法直接构造 THPVariable，所以不需要实现它的 dealloc
void THPVariable_subclass_dealloc(PyObject* self) {
  if (THPVariable_tryResurrect((THPVariable*)self))  // 尝试复活 THPVariable 对象
    return;

  // 这类似于 subtype_dealloc 的简化版本。
  // 不幸的是，我们不能直接委托给 subtype_dealloc，因为它将从 self 的类型开始遍历父类链，这将导致我们回到自定义的 dealloc。
  //
  // 我们必须复制 subtype_dealloc 的逻辑，以确保正确处理 finalizer
  PyTypeObject* type = Py_TYPE(self);  // 获取对象的类型
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);  // 断言对象类型为堆对象
  TORCH_INTERNAL_ASSERT(PyType_IS_GC(type), "GC types not implemented");  // 断言对象类型支持垃圾回收

  PyObject_GC_UnTrack(self);  // 停止追踪该对象

  // TODO: 考虑使用垃圾桶

  bool has_finalizer = type->tp_finalize || type->tp_del;  // 检查是否有 finalizer 或者 del 方法

  if (type->tp_finalize) {
    PyObject_GC_Track(self);  // 开始追踪对象
    if (PyObject_CallFinalizerFromDealloc(self) < 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);  // 停止追踪对象
  }

  // 无需基类测试，因为 THPVariable 没有设置此项
  if (type->tp_weaklistoffset) {
    PyObject_ClearWeakRefs(self);  // 清除对象的弱引用
  }

  if (type->tp_del) {
    PyObject_GC_Track(self);  // 开始追踪对象
    type->tp_del(self);       // 调用类型的 del 方法
    if (Py_REFCNT(self) > 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);  // 停止追踪对象
  }

  if (has_finalizer) {
    /* 在 finalizer 调用期间可能会创建新的弱引用。
       如果发生这种情况，请在不调用它们的 finalizer 的情况下清除它们，因为它们可能依赖已经被销毁的对象的部分。 */
    if (type->tp_weaklistoffset) {
      /* 模仿 GET_WEAKREFS_LISTPTR() */
      PyWeakReference** list =
          (PyWeakReference**)PyObject_GET_WEAKREFS_LISTPTR(self);  // 获取对象的弱引用列表指针
      while (*list)
        _PyWeakref_ClearRef(*list);  // 清除弱引用
    }
  }

  // 清除所有插槽，直到到达基类 THPVariableType
  {
    // 将 type 指针赋值给 base，用于迭代查找基类直到 THPVariableType
    PyTypeObject* base = type;
    // 当 base 不是 THPVariableType 时进行循环
    while (base != &THPVariableType) {
      // 如果 base 的大小不为 0，则调用 clear_slots 清理其槽位信息
      if (Py_SIZE(base)) {
        clear_slots(base, self);
      }
      // 将 base 指向其基类
      base = base->tp_base;
      // 断言 base 不为空，确保在迭代过程中 base 指针有效
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // 所有由 Python 定义的类都有 __dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    // 获取 self 对象的 __dict__ 指针
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    // 如果 dictptr 不为 nullptr
    if (dictptr != nullptr) {
      // 获取 __dict__ 指向的对象
      PyObject* dict = *dictptr;
      // 如果 dict 不为 nullptr
      if (dict != nullptr) {
        // 减少 dict 的引用计数，释放其内存
        Py_DECREF(dict);
        // 将 dictptr 指向 nullptr，表示清空 __dict__ 引用
        *dictptr = nullptr;
      }
    }
  }

  // subtype_dealloc 允许这样做，但我们不需要
  // 断言 self 的类型与 type 相同，确保类型正确性
  TORCH_INTERNAL_ASSERT(Py_TYPE(self) == type);

  // 最后清理 base THPVariable
  // 调用 THPVariable_clear 清理 self 对象中的 Variable 数据
  THPVariable_clear((THPVariable*)self);
  // 调用 Variable 对象的析构函数
  ((THPVariable*)self)->cdata.~MaybeOwned<Variable>();
  // 释放 self 对象所占用的内存，调用 tp_free 方法
  Py_TYPE(self)->tp_free(self);

  // Python 定义的子类应始终位于堆上
  // 断言 type 的 tp_flags 包含 Py_TPFLAGS_HEAPTYPE，确保类型是堆类型
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  // 减少 type 的引用计数，释放其内存
  Py_DECREF(type);
// 结束上一个代码块的定义

// 创建一个新的Python对象用于表示一个变量。status参数指定对象在解释器标签状态上的情况；
// 例如，如果运行了check_pyobj，此对象的可选返回值告诉您张量是否已经被标记，因此可以传递TAGGED_BY_US或MAYBE_UNINITIALIZED；
// 在其他情况下，您知道var的来源，并可以直接断言它是DEFINITELY_UNINITIALIZED。始终可以安全地（尽管较慢）使用MAYBE_UNINITIALIZED调用此函数。
static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status,
  // 确保重新解释为 THPVariable* 的操作是有效的
  TORCH_CHECK(
      PyType_IsSubtype(type, &THPVariableType),
      "Creating a Tensor subclass from a class ",
      "that does not inherit from Tensor is not possible. Make sure your class inherits from Tensor.");

  // 这个函数会覆盖 Tensor 的 pyobj 字段而不进行额外检查
  // 确保 pyobj 字段未设置，否则可能会导致内存泄漏
  auto mb_obj = _var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
      getPyInterpreter(), /*ignore_hermetic_tls=*/false);

  // 在某些情况下，我们可能会尝试为已经有 Python 对象的变量创建一个新的 Python 对象
  // 这种情况最常见的是在有 TorchDispatchMode 激活的情况下，lift_fresh 返回一个子类
  // 这是无法组合处理的情况。比如用户调用 ATensor([1, 2, 3])，当激活了一个转换所有操作的模式，
  // lift_fresh 调用会将 [1, 2, 3] 转换为 torch.tensor([1., 2., 3.])，返回的是 BTensor。
  // 通常情况下，用户只是希望得到一个 FakeTensor，如同 torch.tensor([1, 2, 3]) 或 torch.arange(3) 会返回一个 fake tensor 一样。
  // 这种情况是可以组合处理的，因为 FakeTensor 是 Tensor 的子类。
  // 因此，我们这里要做的是返回对应的子类对象。

  if (mb_obj.has_value() && mb_obj.value()) {
    TORCH_CHECK(
        allow_preexisting_pyobj,
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name);
    // 即使允许存在的 PyObject，也不能完全忽略请求的类型。检查是否满足子类型关系。
    // 在常见情况下，请求的类型是 Tensor，这总是成功的。
    PyObject* obj = *mb_obj;
    // 检查是否可以直接返回 Python 对象而不分配新变量。
    // 我们检查现有的 Python 对象是否是请求类型的子类。
    PyTypeObject* obj_type = Py_TYPE(obj);

    // 使用 TORCH_CHECK 进行断言，确保 obj_type 是 type 或者其子类
    TORCH_CHECK(
        obj_type == type || PyType_IsSubtype(obj_type, type),
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name,
        " which is not a subclass of the "
        "requested type");

    // 在必要时需要复活这个对象
    return THPVariable_Wrap(std::move(_var));
  }

  // 为 type 分配一个新的对象
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*)obj;

    // TODO: 使用命名构造函数来避免默认初始化
    // 使用新的 MaybeOwned<Variable> 来初始化 v->cdata
    new (&v->cdata) MaybeOwned<Variable>();

    // 如果启用了 HermeticPyObjectTLS
    if (c10::impl::HermeticPyObjectTLS::get_state()) {
      // 不要在张量上初始化 pyobj 字段，C++ 拥有所有权
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));

      // 断言不应该具有 torch_dispatch
      TORCH_INTERNAL_ASSERT(
          !check_has_torch_dispatch(obj),
          "While HermeticPyObject was enabled, we attempted to create a tensor "
          "subclass with __torch_dispatch__.  This violates the invariant that "
          "operations in HermeticPyObject have equivalent C++ implementations. "
          "If your operator registered from Python operator registration isn't "
          "doing anything strange, there may be an internal PyTorch bug involving "
          "not appropriately disabling TorchDispatchMode before executing "
          "Python op registration.");
    } else {
      // 正常的代码路径
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));

      // 获取变量并初始化与 Python 对象关联的状态
      const auto& var = THPVariable_Unpack(v);
      var.unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(
          getPyInterpreter(), obj, status);

      // 检查是否具有 torch_dispatch
      if (check_has_torch_dispatch(obj)) {
        var.unsafeGetTensorImpl()->set_python_dispatch(true);
      }
    }
  }
  // 返回分配的对象
  return obj;
/// NOTE [ PyObject Traversal ]
///
/// PyObjects that are wrapping c++ objects can lead to non-trivial traverse
/// logic and it can be tricky to know what to traverse and when. This note
/// tries to clarify what is the danger here and a simple algorithm to choose
/// how to write the tp_traverse and tp_clear functions. If you're not already
/// familiar with how the CPython GC works, you should read this in-depth
/// description: https://devguide.python.org/garbage_collector/
///
/// The complexity for us comes from the fact that some c++ shared_ptr objects
/// own references to python objects and are also owned both by other python
/// objects and c++ objects. This means that to allow the GC to collect all
/// cycles, we need to properly implement the traverse/clear methods that take
/// into account these C++ ownership links.
///
/// The main danger here comes from the fact that, while all python-related code
/// is thread safe wrt the GC execution (thanks to the GIL), other threads might
/// be using our C++ objects arbitrarily which can lead to shared_ptr ref count
/// going up or down in between the different traverse/clear invocations. The
/// one constraint we add here that is not explicitly mentioned in the GC
/// description above is that for a given GC run (meaning while the GIL is
/// held), the traverse/clear pair should never report different ownership
/// relations: if traverse visited a given PyObject, then the clear within that
/// same GC run must still be the sole owner and clear that PyObject.
///
/// A more mechanical algorithm to know what to traverse/clear is as follows:
///   - Any field on this PyObject that contains a strong reference to another
///     PyObject must be visited and cleared. An example of that is the
///     "backward_hooks" field of the THPVariable.
///   - Any field that contains a C++ object that is uniquely owned by this
///     PyObject (either a unique_ptr or a shared_ptr with use_count==1) should
///     have all the PyObjects it owns visited and cleared. An example would be
///     here the tensor hooks.
///   - If that uniquely owned C++ object also uniquely owns other C++ objects,
///     these should be visited and cleared as well if they contain any PyObject.
///
/// Caveat: to avoid slow runtime, we limit the depth of this exploration of C++
/// objects in practice and we do not, for example, go through the whole
/// autograd graph, even if it is uniquely owned. This is a known place where
/// users can create noncollectable cycles as described in:
/// https://github.com/pytorch/pytorch/issues/7343
///

static int traverse_slots(
    PyTypeObject* type,
    PyObject* self,
    visitproc visit,
    void* arg) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Py_ssize_t i, n;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyMemberDef* mp;

  n = Py_SIZE(type);   // 获取给定 PyTypeObject 的大小
  mp = type->tp_members;   // 获取 PyTypeObject 的成员定义数组指针
  for (i = 0; i < n; i++, mp++) {   // 遍历 PyTypeObject 的成员定义数组
    // 如果 mp 指向的类型是 T_OBJECT_EX
    if (mp->type == T_OBJECT_EX) {
      // 计算对象地址并转换为 char 指针
      char* addr = (char*)self + mp->offset;
      // 获取指针 addr 处的 PyObject 对象
      PyObject* obj = *(PyObject**)addr;
      // 如果 obj 非空指针
      if (obj != nullptr) {
        // 访问 obj，并传递 arg 参数，将返回值保存在 err 中
        int err = visit(obj, arg);
        // 如果访问过程中出现错误，则立即返回 err
        if (err)
          return err;
      }
    }
  }
  // 如果没有错误发生，返回 0 表示成功
  return 0;
static int THPVariable_subclass_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  // 如果张量可以被复活，则不遍历它；相反，将其所有引用视为根（因为这些引用在引入的C++引用中会被视为根拥有者）。
  //
  // 这种方法有效是因为与传统的垃圾收集器不同，Python的GC操作分为两个阶段：首先使用traverse发现根，然后使用traverse进行可达性分析。
  // 在根发现阶段跳过traverse会强制Python将self视为其引用的一切的根。有关算法的完整解释请参见
  // https://devguide.python.org/garbage_collector/
  //
  // 注意：如果我们没有对底层张量持有的引用，底层张量可能已经失效。在这种情况下，访问它是不安全的。但是进行遍历是安全的，
  // 因为如果底层张量仍然有效，则根发现将确定self是活跃的，任何对象都不会被GC（如果C++对象拥有PyObject，则不会发生复活）。
  THPVariable* var = reinterpret_cast<THPVariable*>(self);
  if (isResurrectable(var)) {
    return 0;
  }

  // 类似THPVariable_subclass_dealloc的简化版本；与其相同

  PyTypeObject* type = Py_TYPE(self);
  // 遍历槽，直到找到基类THPVariableType
  {
    PyTypeObject* base = type;
    while (base != &THPVariableType) {
      if (Py_SIZE(base)) {
        int err = traverse_slots(base, self, visit, arg);
        if (err)
          return err;
      }
      base = base->tp_base;
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // 所有Python定义的类都有__dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr && *dictptr)
      Py_VISIT(*dictptr);
  }

  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  Py_VISIT(type);

  // 最后遍历THPVariable的特殊内容
  Py_VISIT(var->backward_hooks);
  Py_VISIT(var->post_accumulate_grad_hooks);
  if (!var->cdata.unsafeIsBorrowed()) {
    const auto& tensor = THPVariable_Unpack(var);
    // 检查张量是否已经定义
    if (tensor.defined()) {
      // 警告：grad_fn 遍历逻辑非常微妙，如果更改此处，请非常小心，避免重新引入以下 bug：
      // https://gist.github.com/zou3519/7ac92b84dd7d206dcc6eae55fee8372c

      // 确保遵循 NOTE [ PyObject Traversal ] 中的说明，通过检查此 Python 对象是否是其底层张量的唯一所有者，
      // 并且此张量是否是其 grad_fn 的唯一所有者。在这种情况下，获取 grad_fn 的新引用的唯一方式是使用此 Python 对象，
      // 这需要访问 GIL（全局解释器锁）。请注意，只有当用户不在不同线程之间共享非所有权引用时（这是不合理的，并且不应该这样做），
      // 才能有效执行此操作。
      auto autograd_meta = torch::autograd::impl::get_autograd_meta(tensor);
      // 检查张量的引用计数是否为 1
      if (tensor.use_count() == 1) {
        // 如果存在 autograd_meta 元数据
        if (autograd_meta) {
          // 不要在此处调用 grad_fn()，因为这可能会触发重新计算
          const auto& grad_fn = autograd_meta->grad_fn_;
          // 检查 grad_fn 的引用计数是否为 1
          if (grad_fn && grad_fn.use_count() == 1) {
            // 所有节点都可以具有 pyobj（存储在 "pyobj_" 中）
            Py_VISIT(grad_fn->pyobj());
            // PyNode 是特殊的，因为它们还具有 "obj" 字段
            if (auto py_node_fn = dynamic_cast<PyNode*>(grad_fn.get())) {
              Py_VISIT(py_node_fn->obj);
            }
          }
        }
      }
      // 如果存在 autograd_meta 元数据
      if (autograd_meta) {
        // 遍历张量的 autograd hook
        for (const auto& hook : torch::autograd::impl::hooks(tensor)) {
          // 如果 hook 是 PyFunctionTensorPreHook 类型的实例
          if (auto pyhook =
                  dynamic_cast<PyFunctionTensorPreHook*>(hook.get())) {
            Py_VISIT(pyhook->dict);
          }
        }
      }
    }
  }

  // 返回值为 0
  return 0;
}

// 初始化 THPVariableMetaType 对象的方法
// cls: 要初始化的类对象
// args: 初始化参数
// kwargs: 初始化关键字参数
int THPVariableMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs) {
    // 调用 PyType_Type.tp_init 函数进行基类的初始化，如果失败返回 -1
    if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
        return -1;
    }
    // 设置 cls 对象的析构函数为 THPVariable_subclass_dealloc
    ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPVariable_subclass_dealloc;
    // 设置 cls 对象的遍历函数为 THPVariable_subclass_traverse
    ((PyTypeObject*)cls)->tp_traverse = (traverseproc)THPVariable_subclass_traverse;

    // 如果 THPVariableClass 未定义，则不进行任何操作，直接返回 0
    if (!THPVariableClass) {
        return 0;
    }

    // 检查是否直接从 THPVariableClass 继承，如果不是则抛出 RuntimeError 异常
    py::tuple mro = py::reinterpret_borrow<py::tuple>(((PyTypeObject*)cls)->tp_mro);
    bool is_subclass_of_thpvariable = false;
    for (py::handle h : mro) {
        if (h.ptr() == THPVariableClass) {
            is_subclass_of_thpvariable = true;
            break;
        }
    }
    if (!is_subclass_of_thpvariable) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot subclass _TensorBase directly");
        return -1;
    }

    // 如果用户提供了 torch_dispatch 实现，则禁用 torch_function
    py::object torch_dispatch_impl = py::reinterpret_steal<py::object>(
        PyObject_GetAttrString(cls, "__torch_dispatch__"));
    py::object torch_dispatch_default = py::reinterpret_steal<py::object>(
        PyObject_GetAttrString(THPVariableClass, "__torch_dispatch__"));
    if (torch_dispatch_impl.ptr() != torch_dispatch_default.ptr()) {
        // 获取用户定义的 torch_function
        py::object torch_function_impl = py::reinterpret_steal<py::object>(
            PyObject_GetAttrString(cls, "__torch_function__"));
        // 获取默认的 torch_function
        py::object torch_function_default_bound = py::reinterpret_steal<py::object>(
            PyObject_GetAttrString(THPVariableClass, "__torch_function__"));
        
        // 解绑默认的 torch_function 类方法，获取原始函数
        py::object torch_function_default = py::reinterpret_steal<py::object>(
            PyObject_GetAttrString(torch_function_default_bound.ptr(), "__func__"));

        // 如果用户定义的 torch_function 是类方法，则需要解绑以获取原始函数
        if (PyObject_HasAttrString(torch_function_impl.ptr(), "__func__")) {
            torch_function_impl = py::reinterpret_steal<py::object>(
                PyObject_GetAttrString(torch_function_impl.ptr(), "__func__"));
        }
        // 如果用户定义的 torch_function 与默认值相同，则设置为禁用 torch_function
        if (torch_function_impl.ptr() == torch_function_default.ptr()) {
            PyObject_SetAttrString(
                cls, "__torch_function__", torch::disabled_torch_function_impl());
        }
    }

    return 0;
}

namespace torch::autograd {

// 声明 variable_methods 数组的外部变量
extern PyMethodDef variable_methods[];
// 声明初始化 Torch 函数的函数
extern void initTorchFunctions(PyObject* module);

// 初始化 Tensor 实现转换
void initTensorImplConversion(PyObject* module) {
    // 将 module 转换为 py::module 类型
    auto m = py::handle(module).cast<py::module>();
    // 定义 _wrap_tensor_impl 函数，用于包装 Tensor 实现
    m.def("_wrap_tensor_impl", [](void* ptr) {
        // 从指针 ptr 中重新获取 TensorImpl 对象
        auto p = c10::intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>::
            unsafe_reclaim_from_nonowning(static_cast<c10::TensorImpl*>(ptr));
        // 检查 TensorImpl 是否已定义，如果未定义则抛出异常
        TORCH_CHECK(p.defined(), "Can't wrap undefined tensor");
  // 使用 std::move 将 p 封装成 at::Tensor 对象，然后包装为 py::object 返回
  auto tensor = at::Tensor::wrap_tensor_impl(std::move(p));
  // 将 tensor 转换为 Python 对象，并返回
  return py::cast(std::move(tensor));
});

// 在模块级别设置此函数，以避免混合使用 pybind 和纯 CPython 扩展
m.def("_tensor_impl_raw_handle", [](torch::autograd::Variable* t) -> void* {
  // 返回一个原始的非拥有指针，依赖外部代码来保持原始张量的存活状态
  return t->getIntrusivePtr().get();
});
} // namespace torch::autograd
```