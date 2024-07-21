# `.\pytorch\aten\src\ATen\functorch\ADInterpreters.h`

```
#pragma once
#include <ATen/functorch/Interpreter.h>

namespace at::functorch {

// 这些结构体定义了我们的自动微分转换的解释器
// （grad、vjp 和 jvp）。
// 更多详情请参见 NOTE: [functorch interpreter stack]。

struct TORCH_API GradInterpreterPtr {
  // 构造函数，接受一个基础解释器指针，初始化 GradInterpreterPtr
  explicit GradInterpreterPtr(const Interpreter* base): base_(base) { TORCH_INTERNAL_ASSERT(base->key() == TransformType::Grad); }
  // 返回此解释器的转换类型
  TransformType key() const { return base_->key(); }
  // 返回解释器的级别
  int64_t level() const { return base_->level(); }
  // 处理操作的实现，修改堆栈
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  // 将操作传递给下一个解释器的实现，修改堆栈，并处理梯度特殊情况
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);
  // 返回前一个梯度模式的状态
  bool prevGradMode() const {
    return std::get<GradInterpreterMeta>(base_->meta()).prevGradMode_;
  }
  // 提升张量到当前解释器的梯度解释器中
  Tensor lift(const Tensor& tensor) const;
 private:
  // 基础解释器的指针
  const Interpreter* base_;
};

struct TORCH_API JvpInterpreterPtr {
  // 构造函数，接受一个基础解释器指针，初始化 JvpInterpreterPtr
  explicit JvpInterpreterPtr(const Interpreter* base): base_(base) { TORCH_INTERNAL_ASSERT(base->key() == TransformType::Jvp); }
  // 返回此解释器的转换类型
  TransformType key() const { return base_->key(); }
  // 返回解释器的级别
  int64_t level() const { return base_->level(); }
  // 处理操作的实现，修改堆栈
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  // 将操作传递给下一个解释器的实现，修改堆栈，并处理前向梯度模式特殊情况
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);
  // 返回前一个前向梯度模式的状态
  bool prevFwdGradMode() const {
    return std::get<JvpInterpreterMeta>(base_->meta()).prevFwdGradMode_;
  }
  // 提升张量到当前解释器的 JVP 解释器中
  Tensor lift(const Tensor& tensor) const;
 private:
  // 基础解释器的指针
  const Interpreter* base_;
};

} // namespace at::functorch
```