# `.\pytorch\aten\src\ATen\functorch\FunctionalizeInterpreter.h`

```py
#pragma once
// 使用预处理指令#pragma once确保头文件只被包含一次

#include <ATen/functorch/Interpreter.h>
// 包含ATen库中functorch模块的Interpreter头文件

namespace at::functorch {

// 在functorch命名空间下定义结构体FunctionalizeInterpreterPtr
// 该结构体用于处理functionalize()转换的解释器
// 更多细节参见 NOTE: [functorch interpreter stack]

struct FunctionalizeInterpreterPtr {
  explicit FunctionalizeInterpreterPtr(const Interpreter* base): base_(base) {
    // 使用显式构造函数初始化，要求base指针的转换类型为TransformType::Functionalize
    TORCH_INTERNAL_ASSERT(base->key() == TransformType::Functionalize);
  }

  // 返回转换类型的key
  TransformType key() const {
    return base_->key();
  }

  // 返回解释器的级别
  int64_t level() const {
    return base_->level();
  }

  // 处理操作的具体实现，传入操作句柄op和JIT栈stack
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);

  // 将操作发送到下一个解释器的具体实现，传入操作句柄op、JIT栈stack和是否是梯度特殊情况grad_special_case
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);

  // 返回functionalizeAddBackViews_属性，用于确定是否添加背景视图
  bool functionalizeAddBackViews() const {
    return std::get<FunctionalizeInterpreterMeta>(base_->meta()).functionalizeAddBackViews_;
  }

 private:
  const Interpreter* base_; // 指向基础Interpreter对象的常量指针
};

} // namespace at::functorch
// 结束functorch命名空间
```