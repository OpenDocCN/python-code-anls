# `.\pytorch\aten\src\ATen\templates\DispatchKeyFunctions.h`

```py
#include <ATen/core/TensorBody.h>

// TODO Undo all logic introduced for Note [Avoiding Include Cycles In Static Dispatch]
// 用于撤销为避免静态调度中的循环依赖而引入的所有逻辑
// 因为静态调度逻辑已经从 TensorBody.h 移动到 Operators.cpp，用于支持多个后端和多个内核，因此不再需要这些逻辑。
// 
// Note [Avoiding Include Cycles In Static Dispatch]
// 为了在静态调度构建中避免 #include 循环依赖，我们小心地将静态函数定义文件分割成 {DispatchKey}Functions.h 和 {DispatchKey}Functions_inl.h。
// 
// 没有这种分割，包含循环将是 TensorBody.h -> CPUFunctions.h -> TensorBody.h。
// - 在静态调度构建中，TensorBody.h #include CPUFunctions.h，因为张量方法都需要调用在 CPUFunctions.h 中定义的快速路径 C++ API。这些方法也都直接内联在 TensorBody.h 中。
// - CPUFunctions.h #include TensorBody.h，因为它包含整个 C++ API 的函数声明，其中包括具有可选 Tensor 参数的函数的默认值。
//   这要求了解完整的 Tensor 类定义。
// 
// 我们通过以下方式打破循环依赖：
// - 将 CPUFunctions.h 拆分为两个文件：CPUFunctions.h 和 CPUFunctions_inl.h
// - CPUFunctions.h 是一个虚拟文件，仅包含 Tensor 类，并包含 CPUFunctions_inl.h。
// - CPUFunctions_inl.h 包含其他所有内容。
// - （仅在静态调度构建中）TensorBody.h 确保完成定义 Tensor 类，然后包含 CPUFunctions_inl.h。
// - 所有其他希望使用 cpu 快速路径函数的文件可以直接包含 CPUFunctions.h。
// - 这也意味着在静态调度构建中，CPUFunctions.h 只需要 #include TensorBody.h，它将自动引入 CPUFunctions_inl.h。
${inline_headers}
```