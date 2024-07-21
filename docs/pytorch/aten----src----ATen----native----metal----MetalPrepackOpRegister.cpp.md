# `.\pytorch\aten\src\ATen\native\metal\MetalPrepackOpRegister.cpp`

```
// 引入 ATen 库的头文件
#include <ATen/ATen.h>
// 引入 ATen 核心操作注册的头文件
#include <ATen/core/op_registration/op_registration.h>
// 引入 Metal 前置打包操作上下文的头文件
#include <ATen/native/metal/MetalPrepackOpContext.h>
// 引入 C10 库的累加工具函数头文件
#include <c10/util/accumulate.h>

// 在 at::native::metal 命名空间中定义静态函数 unpack，用于创建 Conv2dOpContext 上下文对象
namespace at::native::metal {

static c10::intrusive_ptr<Conv2dOpContext> unpack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 将 weight 张量按照 ChannelsLast 内存格式进行连续化
  auto packedWeight = weight.contiguous(MemoryFormat::ChannelsLast);
  // 创建并返回 Conv2dOpContext 对象，包含打包后的权重、偏置、步幅、填充、扩展、组数以及输出范围的信息
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(packedWeight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

// 在 at::native::metal 命名空间中定义静态函数 unpack，用于创建 LinearOpContext 上下文对象
static c10::intrusive_ptr<LinearOpContext> unpack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 检查权重张量的维度是否为 2
  TORCH_CHECK(weight.dim() == 2);
  // 不需要对权重张量进行转置操作，直接按照 ChannelsLast 内存格式进行视图重塑和连续化
  auto packedWeight = weight.view({weight.size(0), weight.size(1), 1, 1})
                          .contiguous(MemoryFormat::ChannelsLast);
  // 创建并返回 LinearOpContext 对象，包含打包后的权重、偏置以及输出范围的信息
  return c10::make_intrusive<LinearOpContext>(
      std::move(packedWeight), std::move(bias), output_min, output_max);
}

// 在 at::native::metal 命名空间中定义 TORCH_LIBRARY，注册 Metal 库的操作和上下文信息
TORCH_LIBRARY(metal, m) {
  // 注册 Conv2dOpContext 类，定义序列化和反序列化操作
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
  // 注册 LinearOpContext 类，定义序列化和反序列化操作
  m.class_<LinearOpContext>("LinearOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<LinearOpContext>& op_context)
              -> SerializationTypeLinearPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeLinearPrePack state)
              -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::get<2>(state),
                std::get<3>(state));
          });
  // 定义 Metal 库的 copy_to_host 函数签名
  m.def("copy_to_host(Tensor X) -> Tensor Y");
}

} // namespace at::native::metal
TORCH_LIBRARY(metal_prepack, m) {
  // 定义 metal_prepack 模块的函数 conv2d_prepack，用于预打包 Conv2d 操作的权重和偏置
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal.Conv2dOpContext"));

  // 定义 metal_prepack 模块的函数 conv2d_run，用于运行预打包的 Conv2d 操作
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y"));

  // 定义 metal_prepack 模块的函数 linear_prepack，用于预打包 Linear 操作的权重和偏置
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_prepack(Tensor W, Tensor? B, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.metal.LinearOpContext"));

  // 定义 metal_prepack 模块的函数 linear_run，用于运行预打包的 Linear 操作
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_run(Tensor X, __torch__.torch.classes.metal.LinearOpContext W_prepack) -> Tensor Y"));
}

static c10::intrusive_ptr<Conv2dOpContext> conv2d_prepack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 检查权重张量的维度是否为 4
  TORCH_CHECK(weight.dim() == 4);
  // 创建并返回一个 Conv2dOpContext 实例，封装了给定的参数
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

static c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  // 创建并返回一个 LinearOpContext 实例，封装了给定的权重、偏置和输出范围
  return c10::make_intrusive<LinearOpContext>(
      std::move(weight), std::move(bias), output_min, output_max);
}

TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
  // 实现 metal_prepack 模块中 conv2d_prepack 函数的具体逻辑
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::conv2d_prepack"), TORCH_FN(conv2d_prepack));
  // 实现 metal_prepack 模块中 linear_prepack 函数的具体逻辑
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::linear_prepack"), TORCH_FN(linear_prepack));
}

} // namespace at::native::metal
```