# `.\pytorch\torch\csrc\distributed\rpc\script_call.h`

```
#pragma once
// 防止头文件被多次引用

#include <c10/util/Optional.h>
// 包含使用 Optional 类型的头文件
#include <torch/csrc/distributed/rpc/message.h>
// 包含 RPC 消息的头文件
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 RPC 命令基类的头文件
#include <torch/csrc/jit/runtime/operator.h>
// 包含 JIT 运行时操作符的头文件
#include <torch/csrc/jit/serialization/pickler.h>
// 包含 JIT 序列化 Pickler 的头文件
#include <vector>
// 包含使用向量的头文件

namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::Operator;
// 使用 JIT 操作符命名空间

// ScriptCall 类表示 TorchScript 函数的内置操作调用。如果是内置操作，它包含一个指向 Operator 的共享指针和参数列表。
// 如果是 TorchScript 函数，它包含一个非空的 qualifiedName 字符串以及参数列表。
class TORCH_API ScriptCall : public RpcCommandBase {
 public:
  // 构造函数，用于内置操作调用。
  ScriptCall(std::shared_ptr<Operator> op, std::vector<at::IValue>&& stack);
  // 构造函数，用于 TorchScript 函数调用。
  ScriptCall(
      const c10::QualifiedName& qualifiedName,
      std::vector<at::IValue>&& stack,
      const bool isAsyncExecution = false);

  bool hasOp() const;
  // 返回是否有 Operator
  std::shared_ptr<Operator> op() const;
  // 返回 Operator 的共享指针
  bool hasQualifiedName() const;
  // 返回是否有 qualifiedName
  const c10::QualifiedName& qualifiedName() const;
  // 返回 qualifiedName 的常量引用
  // 返回此内置操作的参数堆栈
  const std::vector<at::IValue>& stack() const;
  // 返回参数堆栈的常量引用
  std::vector<at::IValue>& stackRef();
  // 返回参数堆栈的引用
  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }
  // 返回是否为异步执行

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  // 移动语义转换为消息实现
  static std::unique_ptr<ScriptCall> fromMessage(const Message& message);
  // 从消息创建 ScriptCall 的唯一指针

  ~ScriptCall() override = default;

 protected:
  virtual void toIValues(std::vector<at::IValue>& ivalues) const;
  // 转换为 IValue 的保护方法
  static std::unique_ptr<ScriptCall> fromIValues(
      std::vector<at::IValue>& ivalues);
  // 从 IValue 创建 ScriptCall 的唯一指针

 private:
  // 给定操作符符号和字符串模式，返回匹配的操作符。
  static std::shared_ptr<Operator> matchOperator(const std::string& str_schema);

  static const std::string BUILTIN_OP_NAMESPACE_;
  // 内置操作命名空间的常量字符串
  static const std::string ATEN_PREFIX_;
  // ATen 前缀的常量字符串

  // 如果此 ScriptCall 表示内置操作的调用，则此字段具有值。
  std::optional<std::shared_ptr<Operator>> op_;
  // 如果此 ScriptCall 表示用户定义的 TorchScript 函数调用，则此字段具有非空字符串。
  std::optional<const c10::QualifiedName> qualifiedName_;
  // 参数堆栈
  std::vector<at::IValue> stack_;
  // 是否为异步执行
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```