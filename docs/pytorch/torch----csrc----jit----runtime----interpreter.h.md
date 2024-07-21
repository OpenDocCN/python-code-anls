# `.\pytorch\torch\csrc\jit\runtime\interpreter.h`

```
#pragma once
// 用于防止头文件被多次包含的预处理指令

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional 类头文件

#include <memory>
// 包含 C++ 标准库中的内存管理相关头文件

#include <vector>
// 包含 C++ 标准库中的向量容器头文件

#include <ATen/ThreadLocalState.h>
// 包含 ATen 库中的 ThreadLocalState 头文件

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 ivalue 头文件

#include <ATen/core/jit_type.h>
// 包含 ATen 库中的 jit_type 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 库中的 Export 头文件

#include <torch/csrc/jit/frontend/source_range.h>
// 包含 Torch 中 JIT 前端的 source_range 头文件

C10_DECLARE_bool(torch_jit_disable_warning_prints);
// 在 C10 命名空间声明一个名为 torch_jit_disable_warning_prints 的布尔变量

C10_DECLARE_bool(torch_jit_enable_rethrow_caught_exception);
// 在 C10 命名空间声明一个名为 torch_jit_enable_rethrow_caught_exception 的布尔变量

namespace at {
class Tensor;
// 在 at 命名空间中声明 Tensor 类

TORCH_API void launch(std::function<void()> func);
// 在 at 命名空间中声明一个函数 launch，接受一个无返回值的函数对象作为参数
} // namespace at

namespace c10 {
struct IValue;
// 在 c10 命名空间中声明 IValue 结构体

struct OperatorName;
// 在 c10 命名空间中声明 OperatorName 结构体
} // namespace c10

namespace torch::jit {

// JIT 的代码解释器部分，用于运行图（Graphs）并处理张量输入和张量输出
namespace interpreter {
struct CodeImpl;
} // namespace interpreter

struct Node;
// 声明 Node 结构体

struct GraphExecutor;
// 声明 GraphExecutor 结构体

struct InterpreterStateImpl;
// 声明 InterpreterStateImpl 结构体

struct Graph;
// 声明 Graph 结构体

struct Node;
// 声明 Node 结构体（已声明）

struct Instruction;
// 声明 Instruction 结构体

using Stack = std::vector<c10::IValue>;
// 定义 Stack 类型为存储 c10::IValue 的向量

using c10::ivalue::Future;
// 使用 c10::ivalue 命名空间中的 Future 类

using TaskLauncher = std::function<void(std::function<void()>)>;
// 定义 TaskLauncher 类型为接受一个无返回值函数对象作为参数的函数类型

struct TORCH_API Code {
  Code() = default;
  // 默认构造函数

  explicit Code(interpreter::CodeImpl* pImpl);
  // 显式构造函数，接受 interpreter::CodeImpl 指针参数

  explicit Code(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      size_t remaining_bailout_depth = 0);
  // 显式构造函数，接受图（Graph）、函数名和可选的 remaining_bailout_depth 参数

  const std::vector<GraphExecutor*>& grad_executors();
  // 返回保存了梯度执行器指针的向量

  const std::vector<GraphExecutor*>& diff_graph_op_executors();
  // 返回保存了不同图操作执行器指针的向量

  explicit operator bool() const {
    return pImpl != nullptr;
  }
  // 转换操作符重载，判断 Code 对象是否有效

  size_t num_inputs() const;
  // 返回输入的数量

  size_t num_outputs() const;
  // 返回输出的数量

  size_t num_bailouts() const;
  // 返回 bailout 的数量

  const std::vector<c10::IValue>& constant_table() const;
  // 返回常量表

  const std::vector<c10::TypePtr>& type_table() const;
  // 返回类型表

  const std::vector<Instruction>& instructions() const;
  // 返回指令集

  const std::unordered_map<std::string, size_t>& op_to_num_specified_args()
      const;
  // 返回操作到指定参数数量的映射

  const std::vector<Node*>& instructions_source() const;
  // 返回指令来源的节点向量

  void request_bailout(size_t index);
  // 请求 bailout

  size_t register_size() const;
  // 返回寄存器的大小

  std::shared_ptr<Graph> graph() const;
  // 返回关联的图对象

 private:
  std::shared_ptr<interpreter::CodeImpl> pImpl;
  // 私有成员变量，指向 interpreter::CodeImpl 对象的智能指针

  friend struct InterpreterStateImpl;
  // 将 InterpreterStateImpl 声明为友元类

  friend std::ostream& operator<<(std::ostream& out, const Code& code);
  // 将操作符 << 的重载声明为友元函数
};

struct TORCH_API MobileCode : Code {
  explicit MobileCode(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      bool emit_default_input_instructions = true,
      bool support_default_args_before_out = true,
      bool emit_promoted_ops = true,
      size_t remaining_bailout_depth = 0);
  // 显式构造函数，衍生自 Code 类，接受图（Graph）、函数名和多个可选参数
};

} // namespace jit
// 定义一个结构体 InterpreterState，表示解释器状态
struct InterpreterState {
  // 构造函数，接受一个 Code 对象和一个 TaskLauncher 对象，默认为 at::launch
  TORCH_API InterpreterState(
      const Code& code,
      TaskLauncher taskLauncher = at::launch);
  // 运行函数，接受一个 Stack 引用作为参数，无返回值
  TORCH_API void run(Stack& stack);
  // 异步运行函数，接受一个 Stack 引用作为参数，返回一个 Future 对象指针
  TORCH_API c10::intrusive_ptr<Future> runAsync(Stack& stack);
  // 获取 Future 对象指针的函数
  c10::intrusive_ptr<Future> getFuture();

 private:
  // 私有构造函数，接受一个 intrusive_ptr_target 对象指针作为参数
  InterpreterState(c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl);
  // pImpl 对象指针，使用 intrusive_ptr 包装，用于隐藏 InterpreterStateImpl 的完整定义
  // 注意，这里应该使用 c10::intrusive_ptr<InterpreterStateImpl>，但 intrusive_ptr 要求完整定义
  c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl;
  // 声明 InterpreterStateImpl 为友元结构体，以便访问私有成员
  friend struct InterpreterStateImpl;
};

// Suspend 结构体继承自 std::exception，用于表示暂停异常
// Created by wait()
struct Suspend : public std::exception {
  // 返回异常信息字符串 "Suspend"
  const char* what() const noexcept override {
    return "Suspend";
  }

  // 构造函数，接受一个 Future 对象指针作为参数，并初始化 future 成员
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit Suspend(c10::intrusive_ptr<Future> future_)
      : future(std::move(future_)) {}

  // Future 对象指针成员
  c10::intrusive_ptr<Future> future;
};

// InterpreterContinuation 结构体用于在前向传播期间手动传播 dist_autograd_context_id
// 只能通过 ThreadLocalState 传播其他线程本地设置
struct InterpreterContinuation {
  // 构造函数，接受一个 InterpreterState 对象、一个 Stack 对象、一个 dist_autograd_context_id 和一个 tls_state
  InterpreterContinuation(
      InterpreterState state_,
      Stack stack_,
      int64_t dist_autograd_context_id = 0,
      std::optional<at::ThreadLocalState> tls_state = c10::nullopt)
      : state(std::move(state_)),
        stack(std::move(stack_)),
        tls_state_(std::move(tls_state))
#ifdef USE_DISTRIBUTED
        ,
        dist_autograd_context_id_(dist_autograd_context_id)
#endif
  {
  }

  // 函数调用运算符重载，没有参数和返回值
  void operator()();

 private:
  // InterpreterState 对象成员
  InterpreterState state;
  // Stack 对象成员
  Stack stack;
  // 可选的 ThreadLocalState 对象成员
  std::optional<at::ThreadLocalState> tls_state_ = c10::nullopt;
#ifdef USE_DISTRIBUTED
  // 分布式自动求导上下文 ID
  int64_t dist_autograd_context_id_;
#endif
};

// tensorTypeInCurrentExecutionContext 函数用于获取当前执行上下文中的张量类型
// 包括修改张量行为的当前执行上下文状态
// 例如，如果启用了 no_grad，TensorType 将具有 requires_grad=False
TORCH_API at::TensorTypePtr tensorTypeInCurrentExecutionContext(
    const at::Tensor& t);

// currentCallstack 函数返回当前 TorchScript 解释器的调用堆栈
TORCH_API std::vector<StackEntry> currentCallstack();

// currentModuleHierarchy 函数返回当前模块的层次结构路径
TORCH_API std::vector<std::string> currentModuleHierarchy();

// 结束 torch::jit 命名空间
} // namespace torch::jit
```