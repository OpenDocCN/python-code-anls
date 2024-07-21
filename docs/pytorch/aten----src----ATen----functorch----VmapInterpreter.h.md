# `.\pytorch\aten\src\ATen\functorch\VmapInterpreter.h`

```py
#pragma once
// 包含头文件 Interpreter.h，位于 ATen/functorch 目录下
#include <ATen/functorch/Interpreter.h>

// 定义 at::functorch 命名空间
namespace at::functorch {

// 这是处理 functionalize() 转换的解释器。
// 更多详情请参见 NOTE: [functorch interpreter stack]。

// 定义结构体 VmapInterpreterPtr
struct VmapInterpreterPtr {
  // 构造函数，接受一个 const Interpreter* 类型的指针参数
  explicit VmapInterpreterPtr(const Interpreter* base): base_(base) { TORCH_INTERNAL_ASSERT(base->key() == TransformType::Vmap); }
  // 返回当前解释器的转换类型（key）
  TransformType key() const { return base_->key(); }
  // 返回当前解释器的层级
  int64_t level() const { return base_->level(); }
  // 处理操作的实现函数，接受操作句柄 op 和堆栈指针 stack 作为参数
  void processImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack);
  // 将操作传递给下一个解释器的实现函数，接受操作句柄 op、堆栈指针 stack 和梯度特例标志 grad_special_case 作为参数
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case);
  // 返回批处理大小（batch size）
  c10::SymInt batchSize() const {
    return std::get<VmapInterpreterMeta>(base_->meta()).batchSize_;
  }
  // 返回随机性类型（randomness type）
  RandomnessType randomness() const {
    return std::get<VmapInterpreterMeta>(base_->meta()).randomness_;
  }
 private:
  const Interpreter* base_; // 指向基础 Interpreter 对象的指针
};

} // namespace at::functorch
```