# `.\pytorch\aten\src\ATen\functorch\VmapModeRegistrations.cpp`

```py
// 引入 Torch 库中所需的头文件
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/core/dispatch/Dispatcher.h>

// functorch 的 vmap 实现有两个分发键：FuncTorchBatched 和 FuncTorchVmapMode。
// 该文件包含了对 FuncTorchVmapMode 的注册，用于在我们不支持的操作上报错。
namespace at::functorch {

// 当在 vmap 内部调用不支持的随机操作变体时，使用此函数抛出错误信息
static void unsupportedRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: We do not support calling out variants of random operations inside of vmap. ",
              "Please use non-out variants as a workaround");
}

// 注册一个 fallback，以便在无法匹配的操作上执行默认处理
TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// 当在 vmap 内部调用尚未实现的随机操作时，使用此函数抛出错误信息
static void nyiRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: we do not yet support ", op.schema().operator_name(),
              ". Please file an issue");
}

// 定义宏，用于注册不支持的随机操作变体，并指定对应的处理函数
#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define NYI_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

#define NYI_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

// 注册 ATen 库中的随机操作变体，在 vmap 模式下不支持的操作将通过上述宏注册并进行处理
TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  UNSUPPORTED_RANDOM2(bernoulli, out);
  UNSUPPORTED_RANDOM2(rand, generator_out);
  UNSUPPORTED_RANDOM2(rand, out);
  UNSUPPORTED_RANDOM2(randint, generator_out);
  UNSUPPORTED_RANDOM2(randint, out);
  UNSUPPORTED_RANDOM2(randn, generator_out);
  UNSUPPORTED_RANDOM2(randn, out);
  UNSUPPORTED_RANDOM2(randperm, generator_out);
  UNSUPPORTED_RANDOM2(randperm, out);
  UNSUPPORTED_RANDOM2(multinomial, out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_float_out);
  UNSUPPORTED_RANDOM2(rrelu_with_noise, out);

  // 注册尚未实现的随机操作，将使用 nyiRandomOp 函数处理
  NYI_RANDOM(rrelu_with_noise);
  NYI_RANDOM(rrelu_with_noise_);
  NYI_RANDOM(rrelu_);
  NYI_RANDOM(rrelu);
}

} // namespace at::functorch
```