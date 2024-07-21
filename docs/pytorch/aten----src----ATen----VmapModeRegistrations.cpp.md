# `.\pytorch\aten\src\ATen\VmapModeRegistrations.cpp`

```
// 引入 Torch 库的相关头文件
#include <torch/library.h>
// 引入 ATen 核心库的 boxing 模块中的 KernelFunction 头文件
#include <ATen/core/boxing/KernelFunction.h>

// 使用 torch 命名空间中的 CppFunction
using torch::CppFunction;

// 定义在 at 命名空间内
namespace at {

// [DispatchKey::VmapMode usage]
// 在 vmap 内部时，所有的 Tensor 都会使用这个 dispatch key。
// 目前，这个 key 主要用于禁用 vmap 内部的随机操作。
// 如果你正在寻找批处理规则，那些是使用 DispatchKey::Batched 注册的。
//
// [Ambiguity of random operations inside vmap]
// 随机操作存在一个歧义，即不清楚它们是否应该应用相同的随机性或不同的随机性。
// 例如：
// >>> vmap(lambda t: torch.rand(1))(torch.zeros(5))
// 上面的例子应该返回相同的随机数 5 次，还是不同的随机数？
//
// 我们还没有就此作出决定，因此我们暂时禁止在 vmap 内部执行随机操作，同时收集用户反馈。
template <typename... Args> Tensor unsupportedRandomOp(Args... args) {
  // 抛出错误信息，指示不支持在 vmap 内部调用随机操作
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

// 同上，但是支持对 Tensor 进行原位操作
template <typename... Args> Tensor& unsupportedRandomOp_(Args... args) {
  // 抛出错误信息，指示不支持在 vmap 内部调用随机操作
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

// 实现 TORCH 库中的 VmapMode，使用 torch::CppFunction::makeFallthrough() 作为默认实现
TORCH_LIBRARY_IMPL(_, VmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// 实现 ATen 库中的 VmapMode
TORCH_LIBRARY_IMPL(aten, VmapMode, m) {
  // 注意：我确实希望注册一个特殊的内核，例如 CppFunction::makeNamedNotSupported()，
  // 以避免列出所有类型的细节。然而，例如注册 CppFunction::makeNamedNotSupported() 作为一个
  // 实现只适用于支持 boxing 的运算符。
#undef TENSOROPTIONS
}

} // namespace at
```