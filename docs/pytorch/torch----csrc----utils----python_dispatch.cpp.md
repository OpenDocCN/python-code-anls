# `.\pytorch\torch\csrc\utils\python_dispatch.cpp`

```
// 引入 Torch 库中的函数和头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

// 引入 ATen 库中的头文件和函数
#include <ATen/ATen.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/NestedIntSymNodeImpl.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 引入 Functorch 中的 BatchedTensorImpl 头文件
#include <ATen/functorch/BatchedTensorImpl.h>

// 引入 Torch 的库
#include <torch/library.h>

// 引入 C10 中的 SafePyObject 类
#include <c10/core/SafePyObject.h>

// 引入 Torch 的 PyInterpreter 和 autograd 中的 python_variable 头文件
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>

// 引入 Torch JIT 的 pybind_utils 头文件
#include <torch/csrc/jit/python/pybind_utils.h>

// 引入 Torch 的 tensor_new.h 头文件
#include <torch/csrc/utils/tensor_new.h>

// 引入 C10 的 flat_hash_map 类
#include <c10/util/flat_hash_map.h>

// 引入 pybind11 的操作符和 stl 头文件
#include <pybind11/operators.h>
#include <pybind11/stl.h>

// 引入 Torch 的 aoti_eager 相关头文件
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>

// 引入 Torch 的 pybind 相关头文件
#include <torch/csrc/utils/pybind.h>

// 引入 Torch 的 python_raii 相关头文件
#include <torch/csrc/utils/python_raii.h>

// 引入标准库中的输入输出流和实用程序
#include <iostream>
#include <utility>

// 定义命名空间别名为 py，用于 pybind11 的命名空间
namespace py = pybind11;

// 定义 torch::impl::dispatch 命名空间
namespace torch::impl::dispatch {

// 声明一个静态的哈希映射，用于存储 Python 注册函数的信息，键为 OperatorName，值为哈希映射，键为 DispatchKey，值为 SafePyObject 指针的共享指针
static ska::flat_hash_map<
    c10::OperatorName,
    ska::flat_hash_map<c10::DispatchKey, std::shared_ptr<c10::SafePyObject>>>
    python_registrations_;

// 解析 Torch 库中的 Library::Kind 枚举类型，根据给定字符串 k 返回相应的枚举值
static torch::Library::Kind parseKind(const std::string& k) {
  // 静态的映射表，将字符串映射为 Library::Kind 枚举值
  static std::unordered_map<std::string, torch::Library::Kind> kind_map = {
      {"DEF", torch::Library::DEF},
      {"IMPL", torch::Library::IMPL},
      {"FRAGMENT", torch::Library::FRAGMENT},
  };
  // 在映射表中查找字符串 k 对应的枚举值
  auto it = kind_map.find(k);
  // 如果找不到对应的枚举值，则抛出异常
  TORCH_CHECK(it != kind_map.end(), "could not parse ", k);
  // 返回找到的枚举值
  return it->second;
}

// 解析 C10 库中的 AliasAnalysisKind 枚举类型，根据给定字符串 k 返回相应的枚举值
static c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
  // 静态的映射表，将字符串映射为 AliasAnalysisKind 枚举值
  static std::unordered_map<std::string, c10::AliasAnalysisKind> key_map = {
      {"CONSERVATIVE", c10::AliasAnalysisKind::CONSERVATIVE},
      {"FROM_SCHEMA", c10::AliasAnalysisKind::FROM_SCHEMA},
      {"PURE_FUNCTION", c10::AliasAnalysisKind::PURE_FUNCTION},
      {"", c10::AliasAnalysisKind::FROM_SCHEMA}, // 默认值
  };
  // 在映射表中查找字符串 k 对应的枚举值
  auto it = key_map.find(k);
  // 如果找不到对应的枚举值，则抛出异常
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  // 返回找到的枚举值
  return it->second;
}

// 模板函数，根据给定的 key 字符串调用 torch::dispatch 函数进行分发
template <typename Func>
inline torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  // 将 key 字符串转换为 c10::optional 对象 mb_key
  auto mb_key = std::string(key).empty()
      ? c10::nullopt
      : c10::make_optional(c10::parseDispatchKey(key));
  // 如果 mb_key 有值，则使用 torch::dispatch 函数进行分发
  if (mb_key) {
    return torch::dispatch(*mb_key, std::forward<Func>(raw_f));
  } else {
    // 否则直接调用传入的函数对象 raw_f
    torch::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}

} // namespace torch::impl::dispatch
// 定义一个结构体 EnableHermeticPyObject，用于管理 Python 对象的隔离状态
struct EnableHermeticPyObject {
  // 构造函数，在初始化时保存当前的 HermeticPyObject 状态和 Dispatch Key 状态
  EnableHermeticPyObject()
      : old_(c10::impl::HermeticPyObjectTLS::get_state()), // 保存当前的 HermeticPyObject 状态
        old_excluded_python_(
            c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Python)), // 保存 Python Dispatch Key 的排除状态
        old_python_(
            c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Python)), // 保存 Python Dispatch Key 的包含状态
        old_python_snapshot_(c10::impl::tls_is_dispatch_key_included(
            at::DispatchKey::PythonTLSSnapshot)) { // 保存 PythonTLSSnapshot Dispatch Key 的包含状态
    // 设置当前的 HermeticPyObject 状态为 true
    c10::impl::HermeticPyObjectTLS::set_state(true);
    // 将 Python Dispatch Key 设置为排除状态
    c10::impl::tls_set_dispatch_key_excluded(at::DispatchKey::Python, true);
    // 将 Python Dispatch Key 设置为不包含状态
    c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Python, false);
    // 将 PythonTLSSnapshot Dispatch Key 设置为不包含状态
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::PythonTLSSnapshot, false);
  }

  // 析构函数，在对象销毁时恢复保存的状态
  ~EnableHermeticPyObject() {
    // 恢复 HermeticPyObject 的状态
    c10::impl::HermeticPyObjectTLS::set_state(old_);
    // 恢复 Python Dispatch Key 的排除状态
    c10::impl::tls_set_dispatch_key_excluded(
        at::DispatchKey::Python, old_excluded_python_);
    // 恢复 Python Dispatch Key 的包含状态
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::Python, old_python_);
    // 恢复 PythonTLSSnapshot Dispatch Key 的包含状态
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::PythonTLSSnapshot, old_python_snapshot_);
  }

  // 保存原始的 HermeticPyObject 状态
  bool old_;
  // 保存原始的 Python Dispatch Key 的排除状态
  bool old_excluded_python_;
  // 保存原始的 Python Dispatch Key 的包含状态
  bool old_python_;
  // 保存原始的 PythonTLSSnapshot Dispatch Key 的包含状态
  bool old_python_snapshot_;
};

// 定义一个 PythonKernelHolder 类，继承自 OperatorKernel 类
class PythonKernelHolder : public c10::OperatorKernel {
  c10::SafePyObject func_; // 存储安全的 Python 对象
  c10::DispatchKey dispatch_key_; // 存储 Dispatch Key
  bool with_keyset_; // 是否包含 Keyset

 public:
  // 构造函数，初始化 PythonKernelHolder 对象
  PythonKernelHolder(
      py::object func, // Python 函数对象
      c10::DispatchKey dispatch_key, // Dispatch Key
      bool with_keyset = false) // 是否包含 Keyset，默认为 false
      : func_(func.release().ptr(), getPyInterpreter()), // 初始化 func_ 成员变量
        dispatch_key_(dispatch_key), // 初始化 dispatch_key_ 成员变量
        with_keyset_(with_keyset) {} // 初始化 with_keyset_ 成员变量

  // 运算符重载函数，处理运算符操作
  void operator()(
      const c10::OperatorHandle& op, // 运算符句柄
      c10::DispatchKeySet keyset, // Dispatch Key 集合
      torch::jit::Stack* stack) { // Torch 的堆栈指针
    // 判断是否在 Torch Dispatch 模式下，若是，则使用其 PyInterpreter 进行分发
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    if (mode_stack_len > 0) {
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      // 调用 Python 操作的注册桥接函数
      cur_torch_dispatch_mode_state->pyinterpreter()
          ->python_op_registration_trampoline(
              op, dispatch_key_, keyset, stack, with_keyset_);
      return;
    }

    const auto& schema = op.schema(); // 获取操作符的 schema
    const auto num_arguments = schema.arguments().size(); // 获取参数个数

    // 否则，在 Tensor 上查找 PyInterpreter，如果有 Python key（表示是一个非平凡的张量子类）
    // 对于 stack 中最近的 num_arguments 个 ivalue 进行迭代
    for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
      // 如果 ivalue 是 Tensor 类型
      if (ivalue.isTensor()) {
        // 获取与该 Tensor 相关联的 Python 解释器
        auto* interpreter =
            ivalue.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
        // 如果 interpreter 存在，并且 Tensor 的 key_set 中包含 Python DispatchKey
        if (interpreter &&
            ivalue.unsafeToTensorImpl()->key_set().has(
                at::DispatchKey::Python)) {
          // 调用 Python 操作的注册转发函数
          (*interpreter)
              ->python_op_registration_trampoline(
                  op, dispatch_key_, keyset, stack, with_keyset_);
          // 函数执行完毕，直接返回
          return;
        }
      } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
        // 如果 ivalue 是 TensorList 或 OptionalTensorList 类型
        // 注意：使用 toListRef 而不是 toTensorListRef，因为它不会增加引用计数
        for (const auto& nv : ivalue.toListRef()) {
          // 如果 nv 是 None，跳过当前循环
          if (nv.isNone()) {
            continue;
          }
          // 获取与当前 Tensor 相关联的 Python 解释器
          auto* interpreter =
              nv.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
          // 如果 interpreter 存在，并且 Tensor 的 key_set 中包含 Python DispatchKey
          if (interpreter &&
              nv.unsafeToTensorImpl()->key_set().has(at::DispatchKey::Python)) {
            // 调用 Python 操作的注册转发函数
            (*interpreter)
                ->python_op_registration_trampoline(
                    op, dispatch_key_, keyset, stack, with_keyset_);
            // 函数执行完毕，直接返回
            return;
          }
        }
      }
    }

    // 如果没有特定的 interpreter 要求操作符，就在当前的 interpreter 上执行
    auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
    // 获取全局解释器锁
    py::gil_scoped_acquire g;
    // 2024年1月：我们计划停止使用 multipy，因此停止在所有情况下强制使用 hermetic 模式。
    // 最终可以完全删除这段代码。（请注意，使用该方式可能会破坏需要关闭 hermetic 模式的调度器注册函数。）
#if defined(USE_DEPLOY)
    EnableHermeticPyObject g2;
#endif
    // 将操作和参数解析为 Python 的参数和关键字参数
    auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
    // 获取包装后的 Python 函数对象
    auto func =
        py::reinterpret_borrow<py::object>(func_.ptr(getPyInterpreter()));
    // 根据是否使用 keyset 调用 Python 函数，并获取返回的对象
    auto obj = with_keyset_
        ? func(keyset, *args_kwargs.first, **args_kwargs.second)
        : func(*args_kwargs.first, **args_kwargs.second);
    // 如果返回对象为空，则抛出 Python 错误
    if (!obj) {
      throw python_error();
    }
    // 将 Python 对象推送到堆栈中，用于后续处理
    pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
  }
};

// 静态函数，用于注册或验证 Torch 操作
static torch::_RegisterOrVerify register_or_verify() {
  if (isMainPyInterpreter()) {
    return torch::_RegisterOrVerify::REGISTER;
  } else {
    return torch::_RegisterOrVerify::VERIFY;
  }
}

// Python 调用的函数，处理 Torch 操作的运行
static py::object ophandle_call_boxed(
    const c10::OperatorHandle& handle,
    py::args args,
    const py::kwargs& kwargs) {
  // 创建用于 Torch 操作的堆栈
  auto stack = torch::jit::createStackForSchema(
      handle.schema(),
      std::move(args),
      kwargs,
      /*self=*/c10::nullopt);
  {
    // 释放 GIL，允许 Python 外部并行执行
    pybind11::gil_scoped_release no_gil_guard;
    // 调用 Torch 操作并传入堆栈
    handle.callBoxed(stack);
  }
  // 创建 Python 对象以便返回堆栈中的结果
  return torch::jit::createPyObjectForStack(std::move(stack));
}

// RAII 守卫类，允许显式地从 TLS 排除集中移除一个键
class SetExcludeDispatchKeyGuard {
 public:
  SetExcludeDispatchKeyGuard(at::DispatchKey k, bool set_excluded)
      : k(k), old(c10::impl::tls_is_dispatch_key_excluded(k)) {
    // 设置或取消 TLS 中的调度键排除状态
    c10::impl::tls_set_dispatch_key_excluded(k, set_excluded);
  }
  // 析构函数，恢复 TLS 中的调度键排除状态
  ~SetExcludeDispatchKeyGuard() {
    c10::impl::tls_set_dispatch_key_excluded(k, old);
  }
  // 禁用复制和移动构造函数和赋值运算符
  SetExcludeDispatchKeyGuard(const SetExcludeDispatchKeyGuard&) = delete;
  SetExcludeDispatchKeyGuard operator=(const SetExcludeDispatchKeyGuard&) =
      delete;
  SetExcludeDispatchKeyGuard(SetExcludeDispatchKeyGuard&&) = delete;
  SetExcludeDispatchKeyGuard operator=(SetExcludeDispatchKeyGuard&&) = delete;

 private:
  at::DispatchKey k; // 调度键
  bool old; // 旧的调度键状态
};

// 返回给定操作名称的状态信息，以字符串形式
auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
if (!op) {
  return "";
} else {
  return op->dumpState();
}
});

// 返回给定操作名称的计算表状态信息，以字符串形式
m.def("_dispatch_dump_table", [](const char* name) -> std::string {
  auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
  if (!op) {
    return "";
  } else {
    return op->dumpComputedTable();
  }
});

// 检查给定操作名称的不变式
m.def("_dispatch_check_invariants", [](const char* name) {
  auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
  if (!op) {
  } else {
    return op->checkInvariants();
  }
});

// 检查所有操作的不变式
m.def("_dispatch_check_all_invariants", []() {
  c10::Dispatcher::singleton().checkInvariants();
});

// 检查给定操作名称是否有内核实现
m.def("_dispatch_has_kernel", [](const char* name) -> bool {
  auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));

    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    // 如果找不到操作，则返回 false
    if (!op) {
      return false;
    } else {
      // 返回该操作是否有内核实现
      return op->hasKernel();
    }
  });
  return static_cast<bool>(op);
});

m.def(
    // 检查给定的 <op_name, dispatch_key> 是否有直接的内核注册
    "_dispatch_has_kernel_for_dispatch_key",
    [](const char* name, c10::DispatchKey dispatch) -> bool {
      // 查找操作符 op_name 对应的操作符对象
      auto op =
          c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
      // 如果操作符不存在，则抛出异常
      TORCH_CHECK(op, "operator ", name, " does not exist");
      // 返回给定 dispatch key 是否有内核注册
      return op->hasKernelForDispatchKey(dispatch);
    });

m.def(
    // 检查给定的 <op_name, dispatch_key> 是否是 fallthrough 内核
    "_dispatch_kernel_for_dispatch_key_is_fallthrough",
    [](const char* name, c10::DispatchKey dispatch) -> bool {
      // 查找操作符 op_name 对应的操作符对象
      auto op =
          c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
      // 返回给定 dispatch key 是否是 fallthrough 内核
      return op->isKernelFallthroughKernel(dispatch);
    });

m.def(
    "_dispatch_has_kernel_for_any_dispatch_key",
    [](const char* name, c10::DispatchKeySet ks) -> bool {
      // 查找操作符 op_name 对应的操作符对象
      auto op =
          c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
      // 如果操作符不存在，则抛出异常
      TORCH_CHECK(op, "operator ", name, " does not exist");
      // 返回是否有任何 dispatch key 的内核注册
      return op->hasKernelForAnyDispatchKey(ks);
    });

m.def(
    // 检查给定的 <op_name, dispatch> 是否有计算内核在运行时分发表中的条目
    "_dispatch_has_computed_kernel_for_dispatch_key",
    [](const char* name, const char* dispatch) -> bool {
      // 查找操作符 op_name 对应的操作符对象
      auto op =
          c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
      // 如果操作符不存在，则抛出异常
      TORCH_CHECK(op, "operator ", name, " does not exist");
      // 返回是否有给定 dispatch key 的计算内核
      return op->hasComputedKernelForDispatchKey(
          c10::parseDispatchKey(dispatch));
    });

m.def("_dispatch_find_dangling_impls", []() -> std::vector<std::string> {
  // 查找所有悬空实现的操作符
  auto danglingImpls = c10::Dispatcher::singleton().findDanglingImpls();

  std::vector<std::string> states;
  states.reserve(danglingImpls.size());
  // 获取所有悬空实现的状态信息
  for (auto& danglingImpl : danglingImpls) {
    states.emplace_back(danglingImpl.dumpState());
  }

  return states;
});

m.def("_dispatch_get_all_op_names", []() -> std::vector<std::string> {
  // 获取所有已注册操作符的名称
  auto op_names = c10::Dispatcher::singleton().getAllOpNames();

  std::vector<std::string> names;
  names.reserve(op_names.size());
  // 构造操作符名称列表，包括重载名称
  for (auto& op : op_names) {
    std::stringstream ss;
    ss << op.name;
    if (!op.overload_name.empty()) {
      ss << "." << op.overload_name;
    }
    names.emplace_back(ss.str());
  }

  return names;
});
  // 返回 names 数组
  return names;
});

m.def(
    "_dispatch_tls_set_dispatch_key_excluded",
    [](c10::DispatchKey dispatch_key, bool desired_state) {
      // 调用 C10 库函数设置指定的 dispatch key 是否排除
      c10::impl::tls_set_dispatch_key_excluded(dispatch_key, desired_state);
    });

m.def(
    "_dispatch_tls_is_dispatch_key_excluded",
    [](c10::DispatchKey dispatch_key) {
      // 调用 C10 库函数检查指定的 dispatch key 是否被排除
      return c10::impl::tls_is_dispatch_key_excluded(dispatch_key);
    });

m.def(
    "_dispatch_tls_set_dispatch_key_included",
    [](c10::DispatchKey dispatch_key, bool desired_state) {
      // 调用 C10 库函数设置指定的 dispatch key 是否包含
      c10::impl::tls_set_dispatch_key_included(dispatch_key, desired_state);
    });

m.def(
    "_dispatch_tls_is_dispatch_key_included",
    [](c10::DispatchKey dispatch_key) {
      // 调用 C10 库函数检查指定的 dispatch key 是否被包含
      return c10::impl::tls_is_dispatch_key_included(dispatch_key);
    });

m.def("_dispatch_isTensorSubclassLike", [](const at::Tensor& tensor) {
  // 调用 ATen 库函数检查 tensor 是否为子类张量
  return at::isTensorSubclassLike(tensor);
});

m.def("_dispatch_key_name", [](c10::DispatchKey k) {
  // 调用 C10 库函数将 DispatchKey 转换为其字符串表示
  return c10::toString(k);
});
m.def("_dispatch_key_parse", [](c10::DispatchKey k) { return k; });
m.def("_to_functionality_key", [](c10::DispatchKey k) {
  // 调用 C10 库函数将 DispatchKey 转换为功能键
  return c10::toFunctionalityKey(k);
});
// 返回与功能键相关的后端 dispatch key 集合
// 例如给定 `DispatchKey::AutogradFunctionality`，返回包括 AutogradCPU、AutogradCUDA 等在内的键集合
m.def("_functionality_to_backend_keys", [](c10::DispatchKey key) {
  std::vector<c10::DispatchKey> keys;
  if (c10::isPerBackendFunctionalityKey(key)) {
    // 使用 C10 库函数生成包含指定后端的 dispatch key 集合
    auto ks = c10::DispatchKeySet(key) |
        c10::DispatchKeySet(c10::DispatchKeySet::RAW, c10::full_backend_mask);
    for (auto k : ks) {
      keys.push_back(k);
    }
  } else {
    keys.push_back(key);
  }
  return keys;
});
m.def("_dispatch_num_backends", []() { return c10::num_backends; });
#define DEF_ONE(n) .value(#n, c10::DispatchKey::n)

// 定义一个宏，用于生成枚举值到对应的 `c10::DispatchKey` 枚举成员的映射


py::enum_<c10::DispatchKey>(m, "DispatchKey")
    // clang-format off
    DEF_ONE(Undefined)
    DEF_ONE(CompositeExplicitAutogradNonFunctional)
    DEF_ONE(CompositeExplicitAutograd)
    DEF_ONE(CompositeImplicitAutogradNestedTensor)
    DEF_ONE(CompositeImplicitAutograd)
    // NestedTensor is not a backend key
    DEF_ONE(AutogradNestedTensor)
    DEF_ONE(AutogradOther)
    DEF_ONE(Autograd)
    DEF_ONE(Conjugate)
    DEF_ONE(ZeroTensor)
    DEF_ONE(Negative)
    DEF_ONE(BackendSelect)
    DEF_ONE(ADInplaceOrView)
    DEF_ONE(PythonTLSSnapshot)
    DEF_ONE(Python)
    DEF_ONE(FuncTorchDynamicLayerFrontMode)
    DEF_ONE(FuncTorchDynamicLayerBackMode)
    DEF_ONE(FuncTorchBatchedDecomposition)
    DEF_ONE(FuncTorchBatched)
    DEF_ONE(FuncTorchVmapMode)
    DEF_ONE(FuncTorchGradWrapper)
    DEF_ONE(PythonDispatcher)
    DEF_ONE(PreDispatch)
    DEF_ONE(Functionalize)
    DEF_ONE(AutocastCPU)
    DEF_ONE(AutocastXPU)
    DEF_ONE(AutocastHPU)
    DEF_ONE(AutocastIPU)
    DEF_ONE(AutocastCUDA)
    DEF_ONE(AutocastPrivateUse1)
// clang-format on

// 创建一个 Python 绑定枚举 `DispatchKey`，将每个枚举成员与其相应的 `c10::DispatchKey` 值关联起来


#define DEF_SINGLE(n, prefix) .value(#prefix #n, c10::DispatchKey::prefix##n)
#define DEF_MULTIPLE(fullname, prefix)              \
  DEF_SINGLE(, fullname)                            \
  DEF_SINGLE(, StartOf##fullname##Backends)         \
  C10_FORALL_BACKEND_COMPONENTS(DEF_SINGLE, prefix) \
  DEF_SINGLE(, EndOf##fullname##Backends)

// 定义两个宏，用于生成包含多个后端组件的枚举值到 `c10::DispatchKey` 枚举成员的映射


    // clang-format off
C10_FORALL_FUNCTIONALITY_KEYS(DEF_MULTIPLE)
    // clang-format on

// 根据宏 `DEF_MULTIPLE` 批量生成各种功能键对应的枚举值，与 `c10::DispatchKey` 枚举成员关联起来


#undef DEF_MULTIPLE
#undef DEF_SINGLE
        ;

// 清除之前定义的宏，结束枚举类的定义


py::class_<c10::DispatchKeySet>(m, "DispatchKeySet")
    .def(py::init<c10::DispatchKey>())
    .def("__or__", &c10::DispatchKeySet::operator|)
    .def("__sub__", &c10::DispatchKeySet::operator-)
    .def("__and__", &c10::DispatchKeySet::operator&)
    .def("highestPriorityTypeId", &c10::DispatchKeySet::highestPriorityTypeId)
    .def(
        "remove",
        [](c10::DispatchKeySet self, c10::DispatchKey k) {
          return self.remove(k);
        })
    .def(
        "add",
        [](c10::DispatchKeySet self, c10::DispatchKey k) {
          return self.add(k);
        })
    .def("has", &c10::DispatchKeySet::has)
    .def("__repr__", [](c10::DispatchKeySet d) { return c10::toString(d); });

// 创建 Python 绑定类 `DispatchKeySet`，将其方法映射到 `c10::DispatchKeySet` 类的相应成员函数


m.attr("_dispatch_autogradother_backends") =
    py::cast(c10::autogradother_backends);

// 将 C++ 变量 `c10::autogradother_backends` 转换为 Python 对象，并将其赋值给模块 `m` 的属性 `_dispatch_autogradother_backends`


m.attr("_additional_keys_to_prop_for_wrapper_tensors") =
    py::cast(at::functorch::kKeysToPropagateToWrapper);

// 将 C++ 变量 `at::functorch::kKeysToPropagateToWrapper` 转换为 Python 对象，并将其赋值给模块 `m` 的属性 `_additional_keys_to_prop_for_wrapper_tensors`


m.attr("_after_autograd_keyset") = py::cast(c10::after_autograd_keyset);

// 将 C++ 变量 `c10::after_autograd_keyset` 转换为 Python 对象，并将其赋值给模块 `m` 的属性 `_after_autograd_keyset`


m.attr("_after_ADInplaceOrView_keyset") =
    py::cast(c10::after_ADInplaceOrView_keyset);

// 将 C++ 变量 `c10::after_ADInplaceOrView_keyset` 转换为 Python 对象，并将其赋值给模块 `m` 的属性 `_after_ADInplaceOrView_keyset`


m.def("_dispatch_has_backend_fallback", [](c10::DispatchKey t) {

// 定义 Python 绑定函数 `_dispatch_has_backend_fallback`，接受一个 `c10::DispatchKey` 类型的参数 `t`


    [](c10::DispatchKeySet self, c10::DispatchKey k) {
      return self.remove(k);
    })

// Python 绑定方法，用于从 `DispatchKeySet` 中移除指定的 `DispatchKey` 对象 `k`


    [](c10::DispatchKeySet self, c10::DispatchKey k) {
      return self.add(k);
    })

// Python 绑定方法，用于向 `DispatchKeySet` 中添加指定的 `DispatchKey` 对象 `k`


    .def("has", &c10::DispatchKeySet::has)

// Python 绑定方法，检查 `DispatchKeySet` 是否包含指定的 `DispatchKey`


    .def("__repr__", [](c10::DispatchKeySet d) { return c10::toString(d); });

// Python 绑定方法，返回 `DispatchKeySet` 的字符串表示形式
  m.def("_dispatch_keyset_full_after", [](c10::DispatchKey t) {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, t);
  });


  // 定义函数 "_dispatch_keyset_full_after"，接受一个调度键并返回一个包含该键的完整后置调度键集
  m.def("_dispatch_keyset_full_after", [](c10::DispatchKey t) {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, t);
  });



  m.def("_dispatch_keyset_full", []() {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  });


  // 定义函数 "_dispatch_keyset_full"，返回一个包含所有调度键的完整调度键集
  m.def("_dispatch_keyset_full", []() {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  });



  m.def("_dispatch_is_alias_key", c10::isAliasDispatchKey);


  // 定义函数 "_dispatch_is_alias_key"，用于检查给定的调度键是否是别名调度键
  m.def("_dispatch_is_alias_key", c10::isAliasDispatchKey);



  m.def("_dispatch_keyset_to_string", [](c10::DispatchKeySet keyset) {
    return c10::toString(keyset);
  });


  // 定义函数 "_dispatch_keyset_to_string"，将调度键集转换为字符串表示
  m.def("_dispatch_keyset_to_string", [](c10::DispatchKeySet keyset) {
    return c10::toString(keyset);
  });



  m.def("_dispatch_get_backend_keyset_from_autograd", [](c10::DispatchKey k) {
    return c10::getBackendKeySetFromAutograd(k);
  });


  // 定义函数 "_dispatch_get_backend_keyset_from_autograd"，从 Autograd 获取后端调度键集
  m.def("_dispatch_get_backend_keyset_from_autograd", [](c10::DispatchKey k) {
    return c10::getBackendKeySetFromAutograd(k);
  });



  m.def("_dispatch_keys", [](const at::Tensor& tensor) {
    auto* impl = tensor.unsafeGetTensorImpl();
    return impl->key_set();
  });


  // 定义函数 "_dispatch_keys"，返回给定张量的调度键集
  m.def("_dispatch_keys", [](const at::Tensor& tensor) {
    auto* impl = tensor.unsafeGetTensorImpl();
    return impl->key_set();
  });



  m.def("_dispatch_tls_local_include_set", []() {
    return c10::impl::tls_local_dispatch_key_set().included_;
  });


  // 定义函数 "_dispatch_tls_local_include_set"，返回当前线程局部的包含调度键集
  m.def("_dispatch_tls_local_include_set", []() {
    return c10::impl::tls_local_dispatch_key_set().included_;
  });



  m.def("_dispatch_tls_local_exclude_set", []() {
    return c10::impl::tls_local_dispatch_key_set().excluded_;
  });


  // 定义函数 "_dispatch_tls_local_exclude_set"，返回当前线程局部的排除调度键集
  m.def("_dispatch_tls_local_exclude_set", []() {
    return c10::impl::tls_local_dispatch_key_set().excluded_;
  });



  m.def("_functionalization_reapply_views_tls", []() {
    return c10::Dispatcher::singleton().getPyStub(
        c10::OperatorName(name, overload));
  });


  // 定义函数 "_functionalization_reapply_views_tls"，返回用于重新应用视图的功能化线程局部存根
  m.def("_functionalization_reapply_views_tls", []() {
    return c10::Dispatcher::singleton().getPyStub(
        c10::OperatorName(name, overload));
  });



  m.def("_replace_", [](const at::Tensor& a, const at::Tensor& b) {
    return at::functionalization::impl::replace_(a, b);
  });


  // 定义函数 "_replace_"，用于在功能化中替换张量
  m.def("_replace_", [](const at::Tensor& a, const at::Tensor& b) {
    return at::functionalization::impl::replace_(a, b);
  });



  m.def("_propagate_xla_data", [](const at::Tensor& a, const at::Tensor& b) {
    at::functionalization::impl::propagate_xla_data(a, b);
  });


  // 定义函数 "_propagate_xla_data"，用于在功能化中传播 XLA 数据
  m.def("_propagate_xla_data", [](const at::Tensor& a, const at::Tensor& b) {
    at::functionalization::impl::propagate_xla_data(a, b);
  });



  m.def("_commit_update", [](const at::Tensor& a) {
    return at::functionalization::impl::commit_update(a);
  });


  // 定义函数 "_commit_update"，用于在功能化中提交更新
  m.def("_commit_update", [](const at::Tensor& a) {
    return at::functionalization::impl::commit_update(a);
  });



  m.def("_unsafe_reset_storage", [](const at::Tensor& a) {
    return at::functionalization::impl::unsafe_reset_storage(a);
  });


  // 定义函数 "_unsafe_reset_storage"，用于在功能化中不安全地重置存储
  m.def("_unsafe_reset_storage", [](const at::Tensor& a) {
    return at::functionalization::impl::unsafe_reset_storage(a);
  });



  m.def("_dispatch_key_for_device", [](const std::string& device_type) {
    auto device = c10::Device(device_type);
    TORCH_CHECK(
        !device.has_index(),
        "Expected device_type string to not have a device index; got ",
        device_type);
    return c10::toString(
        c10::computeDispatchKey(c10::nullopt, c10::nullopt, device));
  });


  // 定义函数 "_dispatch_key_for_device"，根据设备类型字符串返回对应的调度键
  m.def("_dispatch_key_for_device", [](const std::string& device_type) {
    auto device = c10::Device(device_type);
    TORCH_CHECK(
        !device.has_index(),
        "Expected device_type string to not have a device index; got ",
        device_type);
    return c10::toString(
        c10::computeDispatchKey(c10::nullopt, c10::nullopt, device));
  });



  m.def("_are_functorch_transforms_active", []() {
    auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
    return (
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode));
  });


  // 定义函数 "_are_functorch_transforms_active"，检查功能化转换是否处于活动状态
  m.def("_are_functorch_transforms_active", []() {
    auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
    return (
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode));
  });



  m.def("_get_nested_int", [](int64_t data, int64_t coeff) {
    return c10::SymInt(c10::SymNode(
        c10::make_intrusive<c10::NestedIntSymNodeImpl>(data, coeff)));
  });


  // 定义函数 "_get_nested_int"，返回一个嵌套整数符号节点
  m.def("_get_nested_int", [](int64_t data, int64_t coeff) {
    return c10::SymInt(c10::SymNode(
        c10::make_intrusive<c10::NestedIntSymNodeImpl>(data, coeff)));
  });



  m.def("_get_constant_bool_symnode", [](int64_t data) {
    return c10::SymNode(
        c10::make_intrusive<c10::ConstantSymNodeImpl<bool>>(data));
  });


  // 定义函数 "_get_constant_bool_symnode"，返回一个常量布尔符号节点
  m.def("_get_constant_bool_symnode", [](int64_t data) {
    return c10::SymNode(
        c10::make_intrusive<c10::ConstantSymNodeImpl<bool>>(data));
  });



  m.def("_non
    // 如果张量没有存储空间，直接返回，因为访问 .data_ptr() 将会引发错误。
    if (!t.unsafeGetTensorImpl()->has_storage()) {
        return;
    }
    // 否则，设置存储实现 (StorageImpl) 的可变 data_ptr 访问时抛出异常。
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_throw_on_mutable_data_ptr();
});

// 不变条件：你只能使用 FakeTensors 调用这个函数。
m.def("_set_warn_deprecated_on_mutable_data_ptr", [](const at::Tensor& t) {
    // 如果张量没有存储空间，直接返回，因为访问 .data_ptr() 将会引发错误。
    if (!t.unsafeGetTensorImpl()->has_storage()) {
        return;
    }
    // 设置存储实现 (StorageImpl) 的可变 data_ptr 访问时发出弃用警告。
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_warn_deprecated_on_mutable_data_ptr();
});

// 将 torch::utils::only_lift_cpu_tensors 绑定为 Python 模块方法 _only_lift_cpu_tensors
m.def("_only_lift_cpu_tensors", &torch::utils::only_lift_cpu_tensors);

// 将 torch::utils::set_only_lift_cpu_tensors 绑定为 Python 模块方法 _set_only_lift_cpu_tensors
m.def("_set_only_lift_cpu_tensors", &torch::utils::set_only_lift_cpu_tensors);

// 使用枚举类型 TorchDispatchModeKey 定义枚举 _TorchDispatchModeKey，并添加三个枚举值
py::enum_<TorchDispatchModeKey>(m, "_TorchDispatchModeKey")
    .value("FUNCTIONAL", TorchDispatchModeKey::FUNCTIONAL)
    .value("PROXY", TorchDispatchModeKey::PROXY)
    .value("FAKE", TorchDispatchModeKey::FAKE);
// 结束函数 python_op_registration_trampoline_impl 的定义

// 用于将 C++ 中的操作注册到 Python 中，通过给定的操作句柄 op，
// 分发键 key，分发键集合 keyset，堆栈 stack，以及是否包含 keyset 来执行操作
void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,     // 操作的句柄
    c10::DispatchKey key,              // 分发键
    c10::DispatchKeySet keyset,        // 分发键集合
    torch::jit::Stack* stack,          // Torch 的堆栈指针
    bool with_keyset) {                // 是否包含分发键集合

  // 从堆栈中弹出操作所需的参数
  auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());

  // 获取全局解释器锁，以确保线程安全
  py::gil_scoped_acquire g;

  // 将传入的参数解析为 Python 中的 *args 和 **kwargs
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);

  // 获取注册的 Python 函数指针
  const auto& func = python_registrations_[op.operator_name()][key];

  // 断言函数指针不为空
  TORCH_INTERNAL_ASSERT(func != nullptr);

  // 获取 Python 函数对象的指针
  auto* pyobj = func->ptr(getPyInterpreter());

  // 断言 Python 对象指针不为空
  TORCH_INTERNAL_ASSERT(pyobj != nullptr);

  // 将 Python 函数对象指针转换为 py::object 类型的 callable
  auto callable = py::reinterpret_borrow<py::object>(pyobj);

  // 根据是否包含 keyset 调用 callable 函数，并将结果赋给 obj
  auto obj = with_keyset
      ? callable(keyset, *args_kwargs.first, **args_kwargs.second)
      : callable(*args_kwargs.first, **args_kwargs.second);

  // 如果调用结果为空，则抛出 python_error 异常
  if (!obj) {
    throw python_error();
  }

  // 将 Python 的输出 obj 推送到 Torch 的堆栈中，标记为 "PythonKernelHolder"
  pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
}

// 结束命名空间 torch::impl::dispatch
} // namespace torch::impl::dispatch
```