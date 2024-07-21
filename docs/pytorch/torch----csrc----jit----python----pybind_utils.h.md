# `.\pytorch\torch\csrc\jit\python\pybind_utils.h`

```
// 预处理指令，指定只包含一次此头文件
#pragma once

// 包含 ATen 库的头文件
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/stack.h>

// 包含 pybind11 库的复数支持和主头文件
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

// 包含 Torch 库的设备、数据类型、导出、布局、量化方案和流的头文件
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>

// 包含 Torch JIT 的模块、模式匹配、追踪、Python 模块和自定义类的头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/runtime/operator.h>

// 包含 Torch 的 Python 工具和参数解析器的头文件
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/six.h>

// 如果定义了 USE_DISTRIBUTED，包含分布式相关的头文件
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#endif

// 包含 ATen 函数模式的头文件
#include <ATen/core/function_schema.h>

// 包含 C10 核心的流和异常处理的头文件
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

// 包含标准库的算法、大小、字符串、实用工具和向量的头文件
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// 处理在 Windows 平台下的可见性属性
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

// Torch JIT 命名空间开始
namespace torch::jit {

// 定义一个解析回调函数类型，接受字符串参数并返回 Python 对象
using ResolutionCallback = std::function<py::object(std::string)>;

// 函数声明：清除注册的实例，接受一个指向 void 的指针参数
void clear_registered_instances(void* ptr);

// Torch Python API：将 Python 对象转换为 IValue，接受 Python 句柄、类型指针和可选的整数参数
TORCH_PYTHON_API IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    std::optional<int32_t> N = c10::nullopt);

// Torch Python API：将 IValue 转换为 Python 对象
TORCH_PYTHON_API py::object toPyObject(IValue ivalue);

// 类：ToIValueAllowNumbersAsTensors，用于允许在期望张量的地方接受 Python 数字
// 控制标志：保存旧的状态
class ToIValueAllowNumbersAsTensors {
  bool old_;

 public:
  // 构造函数，根据传入的布尔值设置状态
  ToIValueAllowNumbersAsTensors(bool enable);

  // 析构函数，恢复旧的状态
  ~ToIValueAllowNumbersAsTensors();
};

// 结构体：PythonFunctionGuard，用于封装 Python 函数并管理其生命周期
struct VISIBILITY_HIDDEN PythonFunctionGuard {
  // 构造函数，接受一个 py::function 对象并初始化成员变量 func_
  explicit PythonFunctionGuard(py::function func) : func_(std::move(func)) {}

  // 析构函数，释放 Python 全局解释器锁，减少 func_ 的引用计数，并显式将 func_ 的指针设为 nullptr
  ~PythonFunctionGuard() {
    pybind11::gil_scoped_acquire ag;
    func_.dec_ref();
    func_.ptr() = nullptr;
  }

  // 成员变量：py::function 对象用于保存封装的 Python 函数
  py::function func_;
};

} // namespace torch::jit
// PythonFutureWrapper 类定义，用于封装 ivalue::Future 的 Python 包装器
//
// NB: VISIBILITY_HIDDEN 用于消除编译错误，
// "error: 'torch::jit::PythonFutureWrapper' declared with greater visibility
// than the type of its field 'torch::jit::PythonFutureWrapper::unwrap_func'
// [-Werror=attributes]"
//
// NB: 继承自 enable_shared_from_this 是因为 then(py::function) 需要从此指针获取 shared_ptr。
struct VISIBILITY_HIDDEN PythonFutureWrapper
    : std::enable_shared_from_this<PythonFutureWrapper> {
  
  // 定义 UnwrapFunc 类型为接受 py::object 参数的 std::function<void> 函数
  using UnwrapFunc = std::function<void(py::object)>;

  // 构造函数，接受一个 ivalue::Future 指针和一个可选的 unwrap_func 函数对象
  explicit PythonFutureWrapper(
      c10::intrusive_ptr<c10::ivalue::Future> fut,
      std::optional<UnwrapFunc> unwrap_func = c10::nullopt)
      : fut(std::move(fut)), unwrap_func(std::move(unwrap_func)) {}

  // 禁用拷贝构造函数和赋值操作符
  explicit PythonFutureWrapper(const PythonFutureWrapper&) = delete;
  PythonFutureWrapper& operator=(const PythonFutureWrapper&) = delete;

  // 检查 Future 是否完成的方法
  bool done() {
    return fut->completed();
  }

  // 获取 Future 的值并返回一个 py::object 对象
  py::object value() {
    // 获取 GIL，因为 toPyObject 创建新的 py::object 而不会获取 GIL
    py::gil_scoped_acquire acquire;
    py::object py_obj = toPyObject(fut->value());
    // 如果存在 unwrap_func，则调用它来执行一些 Python 函数
    if (unwrap_func) {
      (*unwrap_func)(py_obj);
    }
    return py_obj;
  }

  // 等待 Future 完成，并在跟踪状态下插入对应的操作节点
  py::object wait() {
    fut->wait();
    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;

      // 获取 Future 的值的跟踪信息
      Value* fut_val = jit::tracer::getValueTrace(fut);
      // 在图中插入 aten::wait 操作节点，并更新跟踪信息
      auto output = graph->insert(aten::wait, {fut_val});
      jit::tracer::setValueTrace(fut->value(), output);
    }
    return value();
  }

  // then 方法，接受一个 py::function 参数 cb，返回一个 shared_ptr<PythonFutureWrapper>
  // cb 参数必须接受一个 std::shared_ptr<PythonFutureWrapper> 作为唯一参数
  std::shared_ptr<PythonFutureWrapper> then(py::function cb) {
    // 创建 PythonFunctionGuard 对象，用于保护 py::function 对象的析构
    auto pf = std::make_shared<PythonFunctionGuard>(std::move(cb));
    // 返回 pf 对象的 shared_ptr
    return pf;
  }

  // 成员变量：持有的 ivalue::Future 对象和可选的 unwrap_func 函数对象
  c10::intrusive_ptr<c10::ivalue::Future> fut;
  std::optional<UnwrapFunc> unwrap_func;
};
  return std::make_shared<jit::PythonFutureWrapper>(fut->then(
      // 捕获 ivalue::Future 的副本，而不是 this 指针，
      // 因为 PythonFutureWrapper 对象在回调函数执行时可能已被删除。
      // 例如，RPC 只捕获 ivalue::Future，而不是 PythonFutureWrapper
      // 在 JitFuture 的回调函数中。因此，如果用户代码没有持有对
      // PythonFutureWrapper 对象的引用，不能保证在运行回调时
      // PythonFutureWrapper 仍然有效。
      [pyFut(this->getPtr()),
       pf(std::move(pf))](c10::ivalue::Future& /* unused */) -> IValue {
        try {
          pybind11::gil_scoped_acquire ag;
          // 调用 pf->func_，将 PythonFuture 对象传递给函数，并将结果转换为 IValue
          return toIValue(pf->func_(pyFut), PyObjectType::get());
        } catch (py::error_already_set& e) {
          auto err = std::runtime_error(c10::str(
              "运行回调时发生以下错误：",
              e.what()));
          {
            pybind11::gil_scoped_acquire ag;
            // 释放对 py::objects 的所有权，并恢复 Python 错误指示器。
            e.restore();
            // 清除 Python 错误指示器，因为已将异常记录在响应消息中。
            PyErr_Clear();
          }

          throw err;  // 抛出运行时错误
        }
      },
      PyObjectType::get()));
}

void add_done_callback(py::function cb) {
  auto pf = std::make_shared<PythonFunctionGuard>(std::move(cb));
  // NOLINTNEXTLINE(modernize-avoid-bind)
  fut->addCallback(std::bind(
      // 注册回调函数，捕获 pyFut 的指针
      [pyFut(this->getPtr())](std::shared_ptr<PythonFunctionGuard> pf) {
        try {
          pybind11::gil_scoped_acquire ag;
          // 调用 pf->func_，将 PythonFuture 对象传递给函数
          pf->func_(pyFut);
        } catch (py::error_already_set& e) {
          {
            pybind11::gil_scoped_acquire ag;
            // 释放对 py::objects 的所有权，并恢复 Python 错误指示器。
            e.restore();
            // 清除 Python 错误指示器，因为已将异常记录在响应消息中。
            PyErr_Clear();
          }
          // 记录并忽略通过回调引发的异常
          LOG(ERROR) << "运行回调时发生以下错误：" << e.what();

        } catch (const std::exception& e) {
          // 记录并忽略通过回调引发的异常
          LOG(ERROR) << "运行回调时发生以下错误：" << e.what();
        }
      },
      std::move(pf)));
}

void markCompleted(const py::object& pyValue) {
  DCHECK(PyGILState_Check());
  // 将 Python 对象转换为 IValue 类型
  IValue value = toIValue(pyValue, PyObjectType::get());

  py::gil_scoped_release release;
    // 调用 markCompleted() 方法，将移动语义传递的 value 标记为完成状态
    fut->markCompleted(std::move(value));
  }

  c10::intrusive_ptr<c10::ivalue::Future> fut;
  // unwrap_func 类似于 PythonFutureWrapper::wait() 返回值的回调函数
  std::optional<UnwrapFunc> unwrap_func;

 private:
  // 返回指向当前对象的 shared_ptr
  std::shared_ptr<PythonFutureWrapper> getPtr() {
    return shared_from_this();
  }
};

// PythonAwaitWrapper 结构体定义，用于封装 ivalue::Await 的 Python 包装器
// 以 Lazy 语义表达延迟函数执行。例如，在 eager 模式下，Await[W] 可以被用作 W。
// 当请求 W 类型的属性时，Await[W] 将返回 W 的属性，并在此之前透明地调用 wait()。
// 对于脚本，没有 Lazy 语义，必须调用显式的 wait(Await[W]) -> W 来转换为类型 W。
//
// Await 对象以共享所有权的方式获取指定函数和参数。在首次调用 wait() 后，它拥有结果。
// 故意没有为 eager 模式推断类型。
struct VISIBILITY_HIDDEN PythonAwaitWrapper
    : std::enable_shared_from_this<PythonAwaitWrapper> {
  explicit PythonAwaitWrapper(c10::intrusive_ptr<c10::ivalue::Await> aw)
      : aw_(std::move(aw)) {}
  explicit PythonAwaitWrapper(py::handle input) {
    // 创建包含单个输入的 Python 元组
    args_ = py::tuple(1u);
    args_[0] = input;
    // 获取 PyObjectType 类型
    auto type = PyObjectType::get();
    // 创建 ivalue::Await 对象，并标记为已完成
    aw_ = c10::make_intrusive<c10::ivalue::Await>(type);
    aw_->markCompleted(toIValue(input, type));
  }

  explicit PythonAwaitWrapper(py::function pf, py::tuple args) {
    // 创建 PythonFunctionGuard 对象，管理 Python 函数 pf
    pyfg_ = std::make_shared<torch::jit::PythonFunctionGuard>(std::move(pf));
    args_ = std::move(args);
    // 创建 lambda 函数 f，用于执行 Python 函数调用并转换为 IValue
    std::function<IValue()> f = [fg(pyfg_), &args(args_)]() {
      pybind11::gil_scoped_acquire ag;
      return toIValue(fg->func_(*args), PyObjectType::get());
    };
    // 创建 ivalue::Await 对象，关联上述 lambda 函数
    aw_ = c10::make_intrusive<c10::ivalue::Await>(
        PyObjectType::get(), std::move(f));
  }

  // 删除拷贝构造函数和赋值运算符
  explicit PythonAwaitWrapper(const PythonAwaitWrapper&) = delete;
  PythonAwaitWrapper& operator=(const PythonAwaitWrapper&) = delete;

  // 等待操作，获取异步结果并转换为 Python 对象
  py::object wait() {
    py::gil_scoped_acquire acquire;
    return toPyObject(aw_->wait());
  }

  // 检查是否为 nowait 语义，即是否构造自可等待的现在等待
  bool is_nowait() {
    return pyfg_ == nullptr;
  }

  // 获取 Python 函数对象 fn
  const py::function fn() {
    TORCH_CHECK(
        pyfg_, "Await constructed as awaitable_nowait does not have fn");
    return pyfg_->func_;
  }

  // 获取参数元组 args
  const py::tuple args() {
    return args_;
  }

  // 获取类型指针
  TypePtr type() {
    return aw_->type();
  }

  // ivalue::Await 对象
  c10::intrusive_ptr<c10::ivalue::Await> aw_;

  // PythonFunctionGuard 共享指针，管理 Python 函数保护
  std::shared_ptr<torch::jit::PythonFunctionGuard> pyfg_;

  // 参数元组
  py::tuple args_;

 private:
  // 获取当前对象的 shared_ptr
  std::shared_ptr<PythonAwaitWrapper> getPtr() {
    return shared_from_this();
  }
};

// 错误报告：当报告用户引起的错误时，这些函数不应使用 AT_ERROR 宏
// 因为这些宏会添加堆栈跟踪信息，对最终用户显示混乱，因为它总是报告在 libtorch 代码中的位置，而不是用户代码中的位置。
//
// 返回 Python 编译单元的 shared_ptr
inline std::shared_ptr<CompilationUnit> get_python_cu() {
  return py::module::import("torch.jit._state")
      .attr("_python_cu")
      .cast<std::shared_ptr<CompilationUnit>>();
}

// TypedIValue 结构体继承自 std::pair<IValue, TypePtr>
// 提供对内部成员的访问方法
struct TypedIValue : public std::pair<IValue, TypePtr> {
  using pair::pair;

  // 获取 IValue 引用
  IValue& ivalue() {
    return this->first;
  }

  // 获取 TypePtr 引用
  TypePtr& type() {
    return this->second;
  }
};
// 将 Python 对象转换为 TypedIValue 类型，用作字典的键
inline TypedIValue toDictKeyIValue(py::handle key) {
  // 如果键是字符串，则创建 ConstantString 对象，并使用 StringType 类型
  if (py::isinstance<py::str>(key)) {
    return TypedIValue(
        ConstantString::create(py::cast<std::string>(key)), StringType::get());
  } 
  // 如果键是整数，则直接使用 IntType 类型
  else if (py::isinstance<py::int_>(key)) {
    return TypedIValue(py::cast<int64_t>(key), IntType::get());
  } 
  // 如果键是浮点数，则直接使用 FloatType 类型
  else if (py::isinstance<py::float_>(key)) {
    return TypedIValue(py::cast<double>(key), FloatType::get());
  } 
  // 如果键不是字符串、整数或浮点数，则抛出错误
  else {
    AT_ERROR("Dictionary inputs may only have string, int, or float keys");
  }
}

// 统一或初始化类型，根据当前类型和待统一类型决定返回值
inline std::optional<TypePtr> unifyOrInitializeType(
    const TypePtr& accum,
    const TypePtr& unify) {
  // 如果当前类型为空，则直接返回待统一类型
  if (!accum) {
    return unify;
  }
  // 否则调用 unifyTypes 函数，尝试统一两种类型
  return unifyTypes(accum, unify);
}

// 使用别名 InferredType 表示类型推断结果
using InferredType = c10::InferredType;

// 尝试推断输入 Python 对象的容器类型
// 当容器为空（如列表、字典）、列表中的元素类型无法统一、字典中的键或值类型无法统一时，无法推断类型
inline InferredType tryToInferContainerType(py::handle input, bool primitiveTypeOnly);

// 尝试推断 Python 对象的类型
// 包括特定的基本类型推断及特定对象类型的判断
inline InferredType tryToInferType(py::handle input) {
  // 尝试推断为张量类型
  if (THPVariable_Check(input.ptr())) {
    return InferredType(TensorType::get());
  }

  // 如果是 None 类型
  if (input.is_none()) {
    return InferredType(NoneType::get());
  }

  // 如果是 StrongFunctionPtr 类型
  if (py::isinstance<StrongFunctionPtr>(input)) {
    auto fn = py::cast<StrongFunctionPtr>(input).function_;
    return InferredType(FunctionType::create(fn));
  }

  // 尝试基本类型推断
  if (py::isinstance<py::bool_>(input)) {
    return InferredType(BoolType::get());
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (py::isinstance<py::int_>(input)) {
    return InferredType(IntType::get());
  } else if (py::isinstance<py::float_>(input)) {
    return InferredType(FloatType::get());
  } else if (PyComplex_CheckExact(input.ptr())) {
    return InferredType(ComplexType::get());
  } else if (py::isinstance<py::str>(input)) {
    return InferredType(StringType::get());
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPDevice_Check(input.ptr())) {
    return InferredType(DeviceObjType::get());
  } else if (THPGenerator_Check(input.ptr())) {
    return InferredType(GeneratorType::get());
  } else if (THPStream_Check(input.ptr())) {
    return InferredType(StreamObjType::get());
  } else if (THPDtype_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPQScheme_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  }

  // 尝试推断为枚举类型
  auto enum_type = py::module::import("enum").attr("Enum");
  py::bool_ isEnumValue = py::isinstance(input, enum_type);
  if (py::cast<bool>(isEnumValue)) {
    auto enum_class = input.attr("__class__");
    auto enum_type = py::cast<TypePtr>(
        py::module::import("torch.jit.annotations")
            .attr("try_ann_to_type")(enum_class, SourceRange()));
  // 返回推断类型，使用 std::move 来移动 enum_type 到返回值中
  return InferredType(std::move(enum_type));
}

// 检查输入是否为类对象
py::bool_ isClass =
    py::module::import("inspect").attr("isclass")(input.get_type());
if (py::cast<bool>(isClass)) {
  // 假设类已经编译完成或将在稍后编译，如果需要，稍后可以使这个假设失效
  bool class_compiled = true;

  // 检查类型是否已经编译
  py::object existing_ty = py::module::import("torch.jit._state")
                               .attr("_get_script_class")(input.get_type());

  if (existing_ty.is_none()) {
    // 如果没有编译，尝试编译它
    py::bool_ can_compile = py::module::import("torch._jit_internal")
                                .attr("can_compile_class")(input.get_type());

    if (py::cast<bool>(can_compile)) {
      // 尝试编译类。这里使用 try-catch 包裹是因为类类型的编译可能会引发异常，
      // 在这种情况下，我们希望在失败编译时继续尝试下面的类型推断，而不是完全失败。
      try {
        py::module::import("torch.jit._script")
            .attr("_recursive_compile_class")(
                input.get_type(), SourceRange());
      } catch (...) {
        // 如果类编译失败，使 class_compiled 失效，以便我们不会查找并返回其 JIT 类型作为输入的类型。
        class_compiled = false;
      }
    }
  }

  // 如果类成功编译，查找已存在的 JIT 类型并返回它
  if (class_compiled) {
    auto script_class = py::module::import("torch.jit._state")
                            .attr("_get_script_class")(input.get_type());

    if (!script_class.is_none()) {
      auto class_type = py::cast<ClassTypePtr>(script_class);

      // 如果 class_type 不为空且不是模块类型，则返回推断类型
      if (class_type && !class_type->is_module()) {
        return InferredType(std::move(class_type));
      }
    }
  }
}

// 如果输入是 Object 类型的实例
if (py::isinstance<Object>(input)) {
  auto object = py::cast<Object>(input);
  // 返回推断类型，使用 object.type() 来获取类型信息
  return InferredType(object.type());
#ifdef USE_RPC
  } else if (py::isinstance<torch::distributed::rpc::PyRRef>(input)) {
    auto rref_ivalue = input.cast<torch::distributed::rpc::PyRRef>().toIValue();
    return InferredType(rref_ivalue.type());
#endif
  }



// 如果定义了 USE_RPC，且输入对象是 torch 分布式 RPC 的 PyRRef 类型
// 则将其转换为 IValue 类型并返回其推断类型
} else if (py::isinstance<torch::distributed::rpc::PyRRef>(input)) {
  auto rref_ivalue = input.cast<torch::distributed::rpc::PyRRef>().toIValue();
  return InferredType(rref_ivalue.type());
#ifdef USE_RPC
#endif
}

auto await_type = py::module::import("torch._awaits").attr("_Await");
py::bool_ is_await = py::isinstance(input, await_type);
if (py::cast<bool>(is_await)) {
  auto awptr = input.cast<std::shared_ptr<PythonAwaitWrapper>>();
  return InferredType(AwaitType::create(awptr->aw_->elementType()));
}



// 导入并检查是否为 torch._awaits 模块的 _Await 类型
auto await_type = py::module::import("torch._awaits").attr("_Await");
// 检查输入对象是否为 _Await 类型
py::bool_ is_await = py::isinstance(input, await_type);
if (py::cast<bool>(is_await)) {
  // 如果是 _Await 类型，则从输入对象中获取 PythonAwaitWrapper 的共享指针
  auto awptr = input.cast<std::shared_ptr<PythonAwaitWrapper>>();
  // 返回该对象包装的元素类型的推断类型
  return InferredType(AwaitType::create(awptr->aw_->elementType()));
}



// 检查是否为 torch 脚本模块（ScriptModule）
if (as_module(py::cast<py::object>(input))) {
  return InferredType("Cannot infer type of ScriptModule");
}



// 导入并检查是否为 torch.nn 模块的 Module 类型
auto module_type = py::module::import("torch.nn").attr("Module");
py::bool_ is_module = py::isinstance(input, module_type);
if (py::cast<bool>(is_module)) {
  // 如果是 torch.nn.Module 类型，则返回无法推断具体类型的消息
  return InferredType("Cannot infer concrete type of torch.nn.Module");
}



// 尝试推断容器类型
// 调用 tryToInferContainerType 函数来进行推断
return tryToInferContainerType(input, false);
}

// This function is similar to tryToInferType, but it only tries to infer
// primitive types (int, float, bool, complex) or nested container of primitive
// types.
inline InferredType tryToInferPrimitiveType(py::handle input) {
if (input.is_none()) {
return InferredType(NoneType::get());
}

// Only primitive data type
if (py::isinstance<py::bool_>(input)) {
return InferredType(BoolType::get());
// NOLINTNEXTLINE(bugprone-branch-clone)
} else if (py::isinstance<py::int_>(input)) {
return InferredType(IntType::get());
} else if (py::isinstance<py::float_>(input)) {
return InferredType(FloatType::get());
} else if (PyComplex_CheckExact(input.ptr())) {
return InferredType(ComplexType::get());
}

// Try container types
return tryToInferContainerType(input, true);
}

inline InferredType tryToInferContainerType(
py::handle input,
bool primitiveTypeOnly = false) {
if (six::isTuple(input)) {
py::tuple tuple = py::cast<py::tuple>(input);
std::vector<TypePtr> element_types;
element_types.reserve(tuple.size());

for (py::handle elem : tuple) {
auto type_match = primitiveTypeOnly ? tryToInferPrimitiveType(elem)
: tryToInferType(elem);
if (type_match.success()) {
element_types.push_back(type_match.type());
} else {
// Forward error message along
return type_match.reason();
}
}
return InferredType(TupleType::create(std::move(element_types)));
} else if (PyDict_Check(input.ptr())) {
// Check to make sure we can generate useful input/output types
auto dict = py::cast<py::dict>(input);
size_t len = py::len(dict);
if (!len) {
return InferredType("Dictionary inputs must have entries");
}

TypePtr key_type = nullptr;
TypePtr value_type = nullptr;



// 检查输入是否为元组类型
if (six::isTuple(input)) {
py::tuple tuple = py::cast<py::tuple>(input);
std::vector<TypePtr> element_types;
element_types.reserve(tuple.size());

// 遍历元组中的每个元素
for (py::handle elem : tuple) {
// 根据需要是否仅推断基本类型，调用 tryToInferPrimitiveType 或 tryToInferType 进行类型推断
auto type_match = primitiveTypeOnly ? tryToInferPrimitiveType(elem)
: tryToInferType(elem);
if (type_match.success()) {
element_types.push_back(type_match.type());
} else {
// 如果推断失败，将错误消息继续传递
return type_match.reason();
}
}
// 返回推断出的元组类型
return InferredType(TupleType::create(std::move(element_types)));
} else if (PyDict_Check(input.ptr())) {
// 检查是否为字典类型
// 检查以确保我们可以生成有用的输入/输出类型
auto dict = py::cast<py::dict>(input);
size_t len = py::len(dict);
if (!len) {
return InferredType("Dictionary inputs must have entries");
}

TypePtr key_type = nullptr;
TypePtr value_type = nullptr;
    for (auto entry : dict) {
      // 对字典中的每个条目进行迭代

      // 尝试推断键的类型并与现有类型统一
      auto entry_key_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(entry.first)
          : tryToInferType(entry.first);
      if (!entry_key_type_match.success()) {
        return entry_key_type_match.reason();
      }

      // 统一或初始化键的类型
      auto unified_key =
          unifyOrInitializeType(key_type, entry_key_type_match.type());
      if (!unified_key) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            key_type->repr_str(),
            " and ",
            (entry_key_type_match.type())->repr_str()));
      }

      // 尝试推断值的类型并与现有类型统一
      auto entry_value_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(entry.second)
          : tryToInferType(entry.second);
      if (!entry_value_type_match.success()) {
        return entry_value_type_match.reason();
      }

      // 统一或初始化值的类型
      auto unified_value =
          unifyOrInitializeType(value_type, entry_value_type_match.type());
      if (!unified_value) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            value_type->repr_str(),
            " and ",
            (entry_value_type_match.type())->repr_str()));
      }

      // 更新键和值的类型
      key_type = *unified_key;
      value_type = *unified_value;
    }
    return InferredType(
        DictType::create(std::move(key_type), std::move(value_type)));
  } else if (PyList_Check(input.ptr())) {
    auto list = py::cast<py::list>(input);
    size_t len = py::len(list);
    if (!len) {
      return InferredType("List trace inputs must have elements");
    }

    TypePtr element_type = nullptr;

    // 对列表中的每个元素进行迭代
    for (auto elem : list) {
      // 尝试推断元素的类型并与现有类型统一
      auto element_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(elem)
          : tryToInferType(elem);
      if (!element_type_match.success()) {
        return InferredType(c10::str(
            "Could not infer type of list element: ",
            element_type_match.reason()));
      }

      // 统一或初始化元素的类型
      auto unified_type =
          unifyOrInitializeType(element_type, element_type_match.type());
      if (!unified_type) {
        return InferredType(c10::str(
            "List inputs to traced functions must have consistent element type. Found ",
            element_type->repr_str(),
            " and ",
            (element_type_match.type())->repr_str()));
      }

      // 更新元素的类型
      element_type = *unified_type;
    }

    // 返回推断出的列表类型
    return InferredType(ListType::create(element_type));
  } else {
    // 如果只支持基本类型，则返回对应的推断类型
    if (primitiveTypeOnly) {
      return InferredType(c10::str(
          // 构造错误消息，说明只支持基本类型的元组、列表或字典（可能是嵌套的），如 bool、float、int、complex
          "Only tuple, list, or dict (possibly nested) of primitive types (bool, float, int, complex)",
          "are supported ",
          // 提示输入或输出必须是追踪函数的类型，报告实际输入类型
          "as inputs or outputs of traced functions",
          ", but instead got value of type ",
          // 获取输入值的类型名称
          py::str(input.get_type().attr("__name__")),
          "."));
    } else {
      // 否则，返回对应的推断类型
      // TODO: 此消息不再正确，因为这个推断类型现在用于许多与追踪无关的情况。我们可以重复使用这个消息，而不是在 concreteType 中使用 attribute_failure 的内容。
      return InferredType(c10::str(
          // 构造错误消息，说明只支持张量和（可能嵌套的）张量的元组、列表或字典
          "Only tensors and (possibly nested) tuples of tensors, lists, or dicts",
          "are supported ",
          // 提示输入或输出必须是追踪函数的类型，报告实际输入类型
          "as inputs or outputs of traced functions",
          ", but instead got value of type ",
          // 获取输入值的类型名称
          py::str(input.get_type().attr("__name__")),
          "."));
    }
  }


这段代码的作用是根据条件判断生成相应的错误消息，用于描述输入或输出类型不符合预期的情况。
}

// 检查给定类型是否可以被追踪
inline bool isTraceableType(const TypePtr& type) {
  // 如果类型是 TensorType 的子类型，则可以被追踪
  if (type->isSubtypeOf(*TensorType::get())) {
    return true;
  }

  // 如果类型是 ListType 的实例，则检查列表元素类型是否可追踪
  if (auto list_type = type->cast<ListType>()) {
    return isTraceableType(list_type->getElementType());
  }

  // 如果类型是 TupleType 的实例，则检查所有元素类型是否可追踪
  if (auto tuple_type = type->cast<TupleType>()) {
    return std::all_of(
        tuple_type->elements().begin(),
        tuple_type->elements().end(),
        [](const TypePtr& element_type) {
          return isTraceableType(element_type);
        });
  }

  // 如果类型是 DictType 的实例，则检查字典值类型是否可追踪
  if (auto dict_type = type->cast<DictType>()) {
    return isTraceableType(dict_type->getValueType());
  }

  // 默认情况下，类型不可追踪
  return false;
}

// 将 Python 对象转换为推断类型的 IValue
inline IValue toTypeInferredIValue(py::handle input) {
  auto match = tryToInferType(input);
  if (!match.success()) {
    auto object = py::cast<py::object>(input);
    if (auto mod = as_module(object)) {
      // 如果对象已经是 ScriptModule，则直接返回其 ivalue
      auto ptr = mod.value()._ivalue();
      // 显式复制语义，确保资源的强引用
      return c10::intrusive_ptr<c10::ivalue::Object>::reclaim_copy(
          ptr.release());
    }

    // 检查对象是否是 ScriptObject
    if (auto script_obj = as_object(object)) {
      auto ptr = script_obj.value()._ivalue();
      return c10::intrusive_ptr<c10::ivalue::Object>::reclaim_copy(
          ptr.release());
    }
    // 报错，跟踪器无法推断对象的类型
    AT_ERROR(
        "Tracer cannot infer type of ", py::str(input), "\n:", match.reason());
  }
  // 返回推断出的 IValue
  return toIValue(input, match.type());
}

// 将 Python 元组转换为可追踪的堆栈
inline Stack toTraceableStack(const py::tuple& inputs) {
  auto info = toTypeInferredIValue(inputs);
  // 检查元组中的类型是否可追踪
  TORCH_CHECK(
      isTraceableType(info.type()),
      "Type '",
      info.type()->repr_str(),
      "' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and"
      " Tuples of Tensors can be traced");
  // 返回元组的元素作为堆栈
  return info.toTupleRef().elements().vec();
}

// 将 Python 字典序列化为可追踪的堆栈
inline Stack toTraceableStack(const py::dict& inputs) {
  Stack res;
  // 遍历字典中的每个项
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    // 如果值是 Tensor，将其转换为 IValue 并添加到堆栈中
    if (THPVariable_Check(it->second.ptr())) {
      res.push_back(toIValue(it->second, tryToInferType(it->second).type()));
    }
  }
  // 返回堆栈
  return res;
}

// 创建通用的 Python 列表的 IValue
inline IValue createGenericList(py::handle obj, const TypePtr& elem_type) {
  auto elems = c10::impl::GenericList(elem_type);
  // 遍历对象中的每个元素，将其转换为 IValue 并添加到列表中
  for (auto elem : obj) {
    elems.push_back(toIValue(elem, elem_type));
  }
  // 返回包含列表的 IValue
  return IValue(elems);
}

// 创建通用的 Python 字典的 IValue
inline IValue createGenericDict(
    const py::dict& obj,
    const TypePtr& key_type,
    const TypePtr& value_type) {
  // 使用指定的键和值类型创建通用字典
  c10::impl::GenericDict elems(key_type, value_type);
  elems.reserve(py::len(obj));
  // 遍历字典中的每个项，将键和值转换为对应的 IValue，并插入字典中
  for (auto& entry : obj) {
    elems.insert(
        toIValue(entry.first, key_type), toIValue(entry.second, value_type));
  }
  // 返回包含字典的 IValue
  return IValue(elems);
}

// 模板类定义的开头
template <class T>
// 检查变量是否具有命名张量，并抛出错误信息，指出在TorchScript中不支持命名张量，提供删除名称的替代方法。
inline void guardAgainstNamedTensor(const T& var) {
  TORCH_CHECK(
      !var.has_names(),
      "NYI: Named tensors are currently unsupported in TorchScript. As a  "
      "workaround please drop names via `tensor = tensor.rename(None)`.");
}

// 提取通过torchbind注册的自定义类的定制类
template <typename T>
c10::intrusive_ptr<T> toCustomClass(py::handle obj) {
  static_assert(
      std::is_base_of<CustomClassHolder, T>::value, "T is not a CustomClass");
  const auto& type = c10::getCustomClassType<c10::intrusive_ptr<T>>();
  // 将Python对象转换为IValue，使用指定的自定义类类型
  c10::IValue ivalue = toIValue(obj, type);
  // 将IValue转换为指定的自定义类类型并返回
  return std::move(ivalue).toCustomClass<T>();
}

// 用于获取Python类型名称字符串的小包装器，以便更容易解释类型，例如为NamedTuple提供结构类型
inline std::string friendlyTypeName(py::handle obj) {
  if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    // 如果对象是Python元组且具有"_fields"属性，表示它是一个NamedTuple
    auto field_names =
        py::cast<std::vector<std::string>>(py::getattr(obj, "_fields"));
    std::stringstream ss;
    ss << py::str(obj.get_type().attr("__name__"));
    ss << " (aka NamedTuple(";
    bool first = true;
    for (auto& field_name : field_names) {
      if (!first) {
        ss << ", ";
      }
      ss << field_name;
      first = false;
    }
    ss << "))";
    return ss.str();
  } else {
    // 否则返回对象的Python类型名称字符串
    return py::str(obj.get_type().attr("__name__"));
  }
}

// 当尝试创建不能转换的Python参数列表的模式时抛出，可以由调用者捕获以尝试使用其他模式，当存在重载运算符时
struct schema_match_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// 将Python对象转换为IValue，根据函数模式中的参数信息进行类型匹配
inline IValue argumentToIValue(
    const FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object) {
  const auto& argument = schema.arguments().at(argumentPosition);
  try {
    // 将Python对象转换为IValue，使用函数模式中指定的参数的真实类型和数量信息
    return toIValue(object, argument.real_type(), argument.N());
  } catch (const py::cast_error& error) {
    // 捕获类型转换错误，抛出带有详细错误信息的schema_match_error异常
    throw schema_match_error(c10::str(
        schema.formatTypeMismatchMsg(
            argument,
            friendlyTypeName(object),
            argumentPosition,
            py::repr(object)),
        "\nCast error details: ",
        error.what()));
  } catch (const py::error_already_set& error) {
    // 捕获Python异常，抛出带有详细错误信息的schema_match_error异常
    throw schema_match_error(c10::str(
        schema.formatTypeMismatchMsg(
            argument,
            friendlyTypeName(object),
            argumentPosition,
            py::repr(object)),
        "\n Python error details: ",
        error.what()));
  }
}

// 将Python对象转换为IValue，根据指定的类型
inline IValue returnToIValue(const TypePtr& type, py::handle object) {
  try {
    // 将Python对象转换为IValue，使用指定的类型
    return toIValue(object, type);
  } catch (const py::cast_error& error) {
    throw std::runtime_error(c10::str(
        " expected value of type ",
        type->str(),
        " for return value but instead got value of type ",
        py::str(object.get_type().attr("__name__")),
        ".",  // 抛出异常：期望返回类型为...，但实际得到类型为...
        "\nValue: ",
        py::repr(object),
        "\nCast error details: ",
        error.what()));  // 异常信息包括值的表示和转换错误的详细信息
}

// 获取脚本化类或报错信息的内联函数
inline py::object getScriptedClassOrError(const c10::NamedTypePtr& classType) {
  // 导入 torch.jit._state 模块并获取指定类名的 Python 类对象
  auto py_class =
      py::module::import("torch.jit._state")
          .attr("_get_python_class")(classType->name()->qualifiedName());
  // 如果未找到类对象，抛出运行时错误
  if (py_class.is_none()) {
    std::stringstream err;
    err << "Unknown reference to ScriptClass ";
    err << classType->name()->qualifiedName();
    err << ". (Did you forget to import it?)";
    throw std::runtime_error(err.str());
  }
  return py_class;
}

// 定义 tuple_slice 结构体，用于处理 Python 元组的切片操作
struct VISIBILITY_HIDDEN tuple_slice {
  /*implicit*/ tuple_slice(py::tuple tup_)
      : tup(std::move(tup_)), b(0), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_)
      : tup(std::move(tup_)), b(b_), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_, int64_t e_)
      : tup(std::move(tup_)), b(b_), e(e_) {}
  
  // 返回切片的起始迭代器
  py::detail::tuple_iterator begin() const {
    return {tup, static_cast<pybind11::ssize_t>(b)};
  }
  
  // 返回切片的结束迭代器
  py::detail::tuple_iterator end() const {
    return {tup, static_cast<pybind11::ssize_t>(e)};
  }
  
  // 返回切片的大小
  size_t size() const {
    return e - b;
  }
  
  // 返回指定索引处的元素访问器
  py::detail::tuple_accessor operator[](size_t index) const {
    return {tup, static_cast<size_t>(b + index)};
  }

 private:
  py::tuple tup;  // 存储元组对象
  int64_t b;      // 切片起始位置
  int64_t e;      // 切片结束位置
};

// 验证 FakeScriptObject 的对象结构是否符合指定的函数 schema
inline bool validateFakeScriptObjectSchema(
    const c10::FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object) {
  // 获取参数对应的类类型
  auto argument = schema.arguments().at(argumentPosition);
  auto class_type = argument.real_type()->expect<c10::ClassType>();
  
  // 导入 torch._library.fake_class_registry 模块并查找指定类名的 FakeScriptObject 类型
  auto fake_class_registry =
      py::module::import("torch._library.fake_class_registry");
  auto fake_class = fake_class_registry.attr("find_fake_class")(
      class_type->name().value().qualifiedName());
  
  // 如果对象不是指定 FakeScriptObject 类型，抛出 schema_match_error 错误
  if (!py::isinstance(object.attr("wrapped_obj"), fake_class)) {
    throw schema_match_error(c10::str(
        schema.formatTypeMismatchMsg(
            argument,
            friendlyTypeName(object),
            argumentPosition,
            py::repr(object.attr("wrapped_obj"))),
        "\nCast error details: ",
        argument.name(),
        " is expected to be a FakeScriptObject of ",
        class_type->name().value().qualifiedName()));
  }
  return true;
}

// 匹配 schema 并允许 FakeScriptObject 的函数匹配
inline bool matchSchemaAllowFakeScriptObject(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs) {
  size_t all_arguments = args.size() + kwargs.size();
  
  // 如果提供的参数数量超过 schema 规定的数量，抛出 schema_match_error 错误
  if (all_arguments > schema.arguments().size()) {
    throw schema_match_error(c10::str(
        schema.name(),
        "() expected at most ",
        schema.arguments().size(),
        " argument(s) but received ",
        all_arguments,
        " argument(s). Declaration: ",
        schema));
  }

  int64_t arg_idx = 0;
  auto fake_class_registry =
      py::module::import("torch._library.fake_class_registry");

  // 首先处理所有的位置参数
  for (const auto& arg : args) {
    // 如果 schema 要求参数只能作为关键字参数，则拒绝处理
    // ...but refuse to do it if the schema says that this was supposed
    // to be keyword only
    // 检查当前参数是否只能作为关键字参数传递，如果是则抛出异常
    if (schema.arguments()[arg_idx].kwarg_only()) {
      throw schema_match_error(c10::str(
          schema.name(),
          "() takes ",
          arg_idx,
          " positional argument(s) but ",
          args.size(),
          " was/were given.  Declaration: ",
          schema));
    }
    
    // 使用 schema 中的类型信息将 PyObject 转换为合适的类型
    const auto& argument = schema.arguments().at(arg_idx);
    
    // 如果参数的实际类型是 ClassType 并且参数是 FakeScriptObject 类型的实例，则验证其模式
    if (argument.real_type()->kind() == TypeKind::ClassType &&
        py::isinstance(arg, fake_class_registry.attr("FakeScriptObject"))) {
      validateFakeScriptObjectSchema(schema, arg_idx, arg);
    } else {
      // 否则将参数转换为对应的 IValue
      argumentToIValue(schema, arg_idx, arg);
    }

    // 增加参数索引，准备处理下一个参数
    arg_idx++;
  }

  // 现在处理 schema 中剩余的每个非位置参数，在 kwargs 字典中查找并推入参数，否则使用其默认值（如果有）
  size_t consumed_kwargs = 0;
  for (size_t i = arg_idx; i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    if (kwargs.contains(arg.name().c_str())) {
      auto cur_kwarg = kwargs[arg.name().c_str()];
      // 如果参数的实际类型是 ClassType 并且参数是 FakeScriptObject 类型的实例，则验证其模式
      if (arg.real_type()->kind() == TypeKind::ClassType &&
          py::isinstance(
              cur_kwarg, fake_class_registry.attr("FakeScriptObject"))) {
        validateFakeScriptObjectSchema(schema, i, cur_kwarg);
      } else {
        // 否则将参数转换为对应的 IValue
        argumentToIValue(schema, i, cur_kwarg);
      }
      consumed_kwargs += 1;
    } else if (arg.default_value()) {
      // 如果参数有默认值，则继续下一个参数
      continue;
    } else {
      // 否则抛出异常，指出缺少必要的参数值
      throw schema_match_error(c10::str(
          schema.name(),
          "() is missing value for argument '",
          arg.name(),
          "'. Declaration: ",
          schema));
    }
  }

  // 检查是否所有的 kwargs 都已经被消耗，否则抛出异常
  if (consumed_kwargs != kwargs.size()) {
    // 将未消耗的 kwargs 名称收集到 names 中，并抛出对应的异常
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    throw schema_match_error(schema.findErrorInKwargs(names));
  }

  // 如果执行到这里，说明参数匹配成功，返回 true
  return true;
}

// 创建用于函数模式的堆栈
inline Stack createStackForSchema(
    const FunctionSchema& schema,       // 函数的模式定义
    const tuple_slice& args,            // 位置参数的元组切片
    const py::kwargs& kwargs,           // 关键字参数的 Python 字典
    std::optional<IValue> self) {       // self 参数的可选值

  // 计算所有参数的总数，包括 self 参数（如果存在）、位置参数和关键字参数
  size_t all_arguments = (self ? 1 : 0) + args.size() + kwargs.size();

  // 如果传入的参数总数超过函数模式中定义的参数个数，则抛出异常
  if (all_arguments > schema.arguments().size()) {
    throw schema_match_error(c10::str(
        schema.name(),
        "() expected at most ",
        schema.arguments().size(),
        " argument(s) but received ",
        all_arguments,
        " argument(s). Declaration: ",
        schema));
  }

  // 创建一个堆栈，预留足够的空间以容纳函数模式中定义的所有参数
  Stack stack;
  stack.reserve(schema.arguments().size());

  int64_t arg_idx = 0;

  // 如果存在 self 参数，则将其推入堆栈中
  if (self) {
    push(stack, std::move(*self));
    arg_idx++;
  }

  // 首先将所有位置参数推入堆栈
  for (const auto& arg : args) {
    // 如果函数模式中指定此参数应为关键字参数，则抛出异常
    if (schema.arguments()[arg_idx].kwarg_only()) {
      throw schema_match_error(c10::str(
          schema.name(),
          "() takes ",
          arg_idx,
          " positional argument(s) but ",
          self ? 1 + args.size() : args.size(),
          " was/were given.  Declaration: ",
          schema));
    }
    // 使用函数模式中的类型信息将 PyObject 转换为 IValue，并推入堆栈
    push(stack, argumentToIValue(schema, stack.size(), arg));
    arg_idx++;
  }

  // 对于函数模式中剩余的非位置参数，从关键字参数字典中查找并推入堆栈，如果找不到则使用默认值
  size_t consumed_kwargs = 0;
  for (size_t i = stack.size(); i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    if (kwargs.contains(arg.name().c_str())) {
      push(stack, argumentToIValue(schema, i, kwargs[arg.name().c_str()]));
      consumed_kwargs += 1;
    } else if (arg.default_value()) {
      push(stack, *arg.default_value());
    } else {
      throw schema_match_error(c10::str(
          schema.name(),
          "() is missing value for argument '",
          arg.name(),
          "'. Declaration: ",
          schema));
    }
  }

  // 如果关键字参数字典中的参数数量与实际推入堆栈的参数数量不符，则抛出异常
  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    throw schema_match_error(schema.findErrorInKwargs(names));
  }

  // 返回填充好的堆栈
  return stack;
}

// 根据堆栈创建 Python 对象
inline py::object createPyObjectForStack(Stack&& stack) {
  // 如果堆栈为空，则返回 Python 的 None 对象
  if (stack.empty()) {
    return py::none();
  }

  // 如果堆栈中只有一个返回值，则直接返回其对应的 Python 对象
  if (stack.size() == 1) {
    return toPyObject(std::move(stack[0]));
  }

  // 如果堆栈中有多个返回值，则将它们放入一个 py::tuple 中并返回
  py::tuple return_values(stack.size());
  for (const auto ret : c10::irange(return_values.size())) {
    return_values[ret] = toPyObject(std::move(stack[ret]));
  }

  return std::move(return_values);
}

// TODO: Remove once we clean up the GraphExecutor usage.
// 创建一个被废弃且不建议使用的栈，这个栈与Python的tuple、输入值集合和可选的额外空间大小相关联
inline Stack evilDeprecatedBadCreateStackDoNotUse(
    const py::tuple& tuple,                    // Python元组作为输入
    at::ArrayRef<Value*> inputs,               // 输入值的集合
    size_t reserve_extra_space = 0) {          // 可选参数，用于预留额外空间大小
  // 如果输入的tuple大小与inputs的大小不一致，则抛出错误
  if (tuple.size() != inputs.size()) {
    AT_ERROR(
        "expected " + std::to_string(inputs.size()) + " inputs, but got " +
        std::to_string(tuple.size()));
  }
  // 创建一个空的Stack对象
  Stack result;
  // 预留足够的空间以容纳tuple的大小加上额外预留空间大小的元素
  result.reserve(tuple.size() + reserve_extra_space);
  // 遍历输入的inputs集合，将每个tuple元素转换为IValue并推送到Stack中
  for (const auto i : c10::irange(inputs.size())) {
    result.push_back(toIValue(std::move(tuple[i]), inputs[i]->type()));
  }
  // 返回填充好的Stack对象
  return result;
}

// 运行`callee`函数，并可能在追踪图中插入CallFunction/CallMethod节点
inline py::object runAndInsertCall(
    Function& callee,                          // 要运行的函数对象
    const tuple_slice& args,                    // 参数元组的切片
    const py::kwargs& kwargs,                   // 关键字参数
    std::optional<IValue> self,                 // 可选的self参数
    // 一个lambda函数，用于告诉该函数如何将`callee`插入到图中（如果正在追踪）
    const std::function<Value*(Graph&, const MatchedSchema& match)>&
        callInserter) {
  // 创建用于callee函数调用的栈
  auto stack =
      createStackForSchema(callee.getSchema(), args, kwargs, std::move(self));
  // 获取追踪状态
  const auto& tracing_state = tracer::getTracingState();
  // 如果没有正在追踪的状态
  if (!tracing_state) {
    // 释放全局解释器锁
    pybind11::gil_scoped_release no_gil_guard;
    // 如果不在追踪中，直接运行callee函数
    callee.run(stack);
  } else {
    // 如果正在追踪中，插入适当的CallFunction或CallMethod节点，然后在禁用追踪的情况下运行callee函数

    // 获取表示输入IValues的图形值
    auto inputs = last(stack, callee.num_inputs());
    auto input_values =
        fmap(inputs, [](const IValue& v) { return tracer::getValueTrace(v); });
    // 断言callee函数返回的值只有一个
    TORCH_INTERNAL_ASSERT(callee.getSchema().returns().size() == 1)
    auto return_type = callee.getSchema().returns().at(0).type();
    auto graph = tracing_state->graph;
    std::vector<NamedValue> named_values;
    named_values.reserve(input_values.size());
    for (Value* v : input_values) {
      named_values.emplace_back(v);
    }

    // 添加一个调用节点
    MatchedSchema match = matchSchema(
        callee.getSchema(),
        tracer::getPythonInterpreterSourceRange(),
        *graph,
        named_values,
        {});
    auto output_value = callInserter(*graph, match);

    // 实际运行callee函数。暂停追踪，以免重复添加callee节点
    {
      pybind11::gil_scoped_release no_gil_guard;
      ResourceGuard guard(tracer::pauseTracing());
      callee.run(stack);
    }

    // 将输出的IValues与图中的输出Values关联
    tracer::setValueTrace(stack.back(), output_value);
  }

  // 检查执行后栈中是否有值，如果没有则抛出错误
  TORCH_CHECK(
      !stack.empty(),
      "Expected values in the stack after execution but found none");
  // 将栈顶的值转换为Python对象并返回
  return toPyObject(std::move(stack.back()));
}

// 可能的torch函数分发，用于调用torch函数对象或方法
inline std::optional<py::object> maybeTorchFunctionDispatch(
    const py::object& callee,                   // 被调用的torch函数对象或方法
    const tuple_slice& args_no_self,            // 不包括self参数的参数元组切片
    const py::kwargs& kwargs,
    const c10::QualifiedName qualname) {
  // 创建一个空的 Python 对象列表
  std::vector<py::handle> args_vec;
  // 遍历参数列表 args_no_self，将每个参数加入到 args_vec 中
  for (const auto& arg : args_no_self) {
    args_vec.push_back(arg);
  }
  // 将 args_vec 转换为 Python 的元组对象
  py::tuple args = py::cast(args_vec);

  // 处理 __torch_function__ 分发
  // 创建一个空的 PyObject 指针列表，用于存放重载的参数
  std::vector<PyObject*> overloaded_args;
  // 计算参数总数，包括 args 和 kwargs
  size_t total_arg_num = args.size() + kwargs.size();
  // 遍历 args 中的每个参数，检查是否为 tensor 类型并将其加入 overloaded_args
  for (const auto& arg : args) {
    is_tensor_and_append_overloaded(arg.ptr(), &overloaded_args);
    // 检查是否为 tensor 列表类型并将其加入 overloaded_args，传入总参数数和错误抛出标志
    is_tensor_list_and_append_overloaded(
        arg.ptr(),
        &overloaded_args,
        static_cast<int>(total_arg_num),
        false /* throw_error */);
  }
  // 对于 kwargs，不能保证添加顺序与操作符模式中参数顺序一致
  // 这不是最优的做法，但应该是可行的。当我们有更好的模式匹配和参数解析时，
  // 可以首先匹配操作符中的 `operations`，然后顺序将会得到保证。
  for (auto item : kwargs) {
    // 检查是否为 tensor 类型并将其加入 overloaded_args
    is_tensor_and_append_overloaded(item.second.ptr(), &overloaded_args);
    // 检查是否为 tensor 列表类型并将其加入 overloaded_args，传入总参数数和错误抛出标志
    is_tensor_list_and_append_overloaded(
        item.second.ptr(),
        &overloaded_args,
        total_arg_num,
        false /* throw_error */);
  }
  // 如果 overloaded_args 不为空，则执行 Torch 函数分发
  if (!overloaded_args.empty()) {
    return pybind11::reinterpret_steal<py::object>(
        // 调用处理没有 Python 参数解析器的 Torch 函数
        handle_torch_function_no_python_arg_parser(
            /*overloaded_args=*/overloaded_args,
            /*args=*/args.ptr(),
            /*kwargs=*/kwargs.ptr(),
            /*func_name=*/qualname.name().c_str(),
            /*torch_api_function=*/callee.ptr(),
            /*module_name=*/qualname.prefix().c_str()));
  }

  // 如果没有重载的参数，返回空值
  return c10::nullopt;
}
}

// 定义一个内联函数，用于从 Python 调用脚本函数
inline py::object invokeScriptFunctionFromPython(
    Function& callee,                  // 被调用的函数对象的引用
    const tuple_slice& args,           // 位置参数的元组切片
    const py::kwargs& kwargs) {        // 关键字参数的字典

  // TODO: 我们可以在这里添加 __torch_function__ 分发，但我不确定这样做的影响
  // 暂时未实现的功能，可以添加 __torch_function__ 分发以支持 Torch 脚本函数的扩展

  // 调用 runAndInsertCall 函数来执行函数调用并插入调用的结果到图中
  return runAndInsertCall(
      callee,                         // 被调用的函数对象
      args,                           // 位置参数
      kwargs,                         // 关键字参数
      /*self=*/c10::nullopt,          // self 参数为空
      [&](Graph& graph, const MatchedSchema& match) {
        return graph.insertFunctionCall(&callee, match);  // 在图中插入函数调用
      });
}

// 定义一个内联函数，用于从 Python 调用脚本方法
inline py::object invokeScriptMethodFromPython(
    Method& callee,                    // 被调用的方法对象的引用
    const tuple_slice& args,           // 位置参数的元组切片
    const py::kwargs& kwargs) {        // 关键字参数的字典
  auto self = callee.owner()._ivalue();  // 获取方法的所有者对象的 IValue

  // 尝试通过 maybeTorchFunctionDispatch 进行 Torch 函数的分发
  if (auto torch_fn_result = maybeTorchFunctionDispatch(
          py::cast(callee), args, kwargs, callee.name())) {
    return *torch_fn_result;          // 返回 Torch 函数的结果
  }

  // 否则，调用 runAndInsertCall 函数执行方法调用并将结果插入到图中
  return runAndInsertCall(
      callee.function(),              // 被调用的方法对应的函数对象
      args,                           // 位置参数
      kwargs,                         // 关键字参数
      self,                           // self 参数
      [&](Graph& graph, const MatchedSchema& match) {
        return graph.insertMethodCall(callee.name(), match);  // 在图中插入方法调用
      });
}

// 获取带有堆栈的操作符对象
TORCH_PYTHON_API std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    const std::vector<std::shared_ptr<Operator>>& operations,  // 操作符对象的集合
    py::args args,                  // 位置参数
    const py::kwargs& kwargs);      // 关键字参数

// 从 Python 调用操作符
TORCH_PYTHON_API py::object invokeOperatorFromPython(
    const std::vector<std::shared_ptr<Operator>>& operations,  // 操作符对象的集合
    py::args args,                  // 位置参数
    const py::kwargs& kwargs,       // 关键字参数
    std::optional<c10::DispatchKey> dk = c10::nullopt);        // 分发键的可选值

// 处理 Torch 函数的可能性
TORCH_PYTHON_API std::optional<py::object> _maybe_handle_torch_function(
    const std::string& ns,          // 命名空间
    const std::string& method_name, // 方法名
    const std::string& overload_name,  // 重载名称
    bool is_overload,               // 是否是重载
    py::args args,                  // 位置参数
    const py::kwargs& kwargs);      // 关键字参数

// 检查允许虚假脚本对象的函数模式
TORCH_PYTHON_API bool checkSchemaAllowFakeScriptObject(
    const FunctionSchema& schema,   // 函数模式对象
    py::args args,                  // 位置参数
    const py::kwargs& kwargs);      // 关键字参数

// 获取用于重载或数据包的操作符对象
TORCH_PYTHON_API py::object _get_operation_for_overload_or_packet(
    const std::vector<std::shared_ptr<Operator>>& operations,  // 操作符对象的集合
    Symbol symbol,                  // 符号
    py::args args,                  // 位置参数
    const py::kwargs& kwargs,       // 关键字参数
    bool is_overload,               // 是否是重载
    std::optional<c10::DispatchKey> dk = c10::nullopt);        // 分发键的可选值

} // namespace torch::jit
```