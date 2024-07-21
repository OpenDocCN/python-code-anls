# `.\pytorch\aten\src\ATen\core\Reduction.h`

```py
#pragma once

#pragma once 指令，用于确保头文件只被编译一次，防止重复包含。


namespace at::Reduction {

定义了命名空间 `at::Reduction`，用于将下面的内容组织在一个特定的作用域中。


// NB: Keep this in sync with Reduction class in torch/nn/_reduction.py

注释：提醒需要保持此处与 `torch/nn/_reduction.py` 中的 `Reduction` 类同步。


// These constants control the reduction behavior of loss functions.

注释：这些常量控制损失函数的减少行为。


// Ideally, this would be a scoped enum, but jit doesn't support that

注释：理想情况下，这应该是一个作用域枚举，但是 JIT 不支持。


enum Reduction {
  None, // Do not reduce
  Mean, // (Possibly weighted) mean of losses
  Sum,  // Sum losses
  END   // End marker
};

定义了枚举类型 `Reduction`，它包含以下常量：
- `None`：不进行减少
- `Mean`：（可能带权重的）损失均值
- `Sum`：损失求和
- `END`：枚举的结束标记


} // namespace at::Reduction

命名空间 `at::Reduction` 的结束标记。
```