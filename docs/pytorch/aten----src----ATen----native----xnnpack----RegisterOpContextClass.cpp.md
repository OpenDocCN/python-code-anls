# `.\pytorch\aten\src\ATen\native\xnnpack\RegisterOpContextClass.cpp`

```
#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则编译以下代码块

#include <torch/library.h>
// 引入 Torch 库的头文件

#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
// 引入 XNNPACK 相关的头文件，用于实现卷积和线性操作的优化

#include <torch/custom_class.h>
// 引入 Torch 自定义类的头文件

namespace at::native::xnnpack {
// 进入 at::native::xnnpack 命名空间

using internal::linear::createLinearClampPrePackOpContext;
using internal::convolution2d::createConv2dClampPrePackOpContext;
using internal::convolution2d::createConv2dTransposeClampPrePackOpContext;
// 使用 XNNPACK 内部的函数来创建线性、卷积和转置卷积的预打包操作上下文

TORCH_LIBRARY(xnnpack, m) {
  // 在 Torch 的 xnnpack 库注册新类

  m.class_<LinearOpContext>(TORCH_SELECTIVE_CLASS("LinearOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<LinearOpContext>& op_context)
            -> SerializationTypeLinearPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeLinearPrePack state)
            -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
          return createLinearClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              std::move(std::get<2>(state)),
              std::move(std::get<3>(state)));
        });
  // 定义 LinearOpContext 类的序列化方法，包括获取状态和设置状态

  m.class_<Conv2dOpContext>(TORCH_SELECTIVE_CLASS("Conv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
            -> SerializationTypeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeConv2dPrePack state)
            -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
          return createConv2dClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              std::move(std::get<2>(state)),
              std::move(std::get<3>(state)),
              std::move(std::get<4>(state)),
              std::move(std::get<5>(state)),
              std::move(std::get<6>(state)),
              std::move(std::get<7>(state)));
        });
  // 定义 Conv2dOpContext 类的序列化方法，包括获取状态和设置状态

  m.class_<TransposeConv2dOpContext>(TORCH_SELECTIVE_CLASS("TransposeConv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<TransposeConv2dOpContext>& op_context)
            -> SerializationTypeTransposeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeTransposeConv2dPrePack state)
            -> c10::intrusive_ptr<TransposeConv2dOpContext> { // __setstate__
          return createConv2dTransposeClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              std::move(std::get<2>(state)),
              std::move(std::get<3>(state)),
              std::move(std::get<4>(state)),
              std::move(std::get<5>(state)),
              std::move(std::get<6>(state)),
              std::move(std::get<7>(state)),
              std::move(std::get<8>(state)));
        });
  // 定义 TransposeConv2dOpContext 类的序列化方法，包括获取状态和设置状态

}

// Registration using the TORCH_LIBRARY def gives dispatching errors when there is no tensor input
// 使用 TORCH_LIBRARY 来注册会在没有张量输入时产生调度错误
#endif
// 结束 USE_XNNPACK 宏的条件编译
TORCH_LIBRARY(prepacked, m) {
  // 定义 prepacked::unpack_prepacked_sizes_conv2d 的 Torch 模块接口，调用 convolution2d 命名空间的函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_conv2d(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::convolution2d::unpack_prepacked_sizes_conv2d(inp);});
  // 定义 prepacked::unpack_prepacked_sizes_linear 的 Torch 模块接口，调用 linear 命名空间的函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_linear(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::linear::unpack_prepacked_sizes_linear(inp);});
  // 定义 prepacked::linear_clamp_prepack 的 Torch 实现，使用 createLinearClampPrePackOpContext 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.LinearOpContext"));
  // 定义 prepacked::linear_clamp_run 的 Torch 实现，调用 linear 命名空间的 linear_clamp_run 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y"));
  // 定义 prepacked::conv2d_clamp_prepack 的 Torch 实现，使用 createConv2dClampPrePackOpContext 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.Conv2dOpContext"));
  // 定义 prepacked::conv2d_transpose_clamp_prepack 的 Torch 实现，使用 createConv2dTransposeClampPrePackOpContext 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.TransposeConv2dOpContext"));
  // 定义 prepacked::conv2d_clamp_run 的 Torch 实现，调用 convolution2d 命名空间的 conv2d_clamp_run 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y"));
  // 定义 prepacked::conv2d_transpose_clamp_run 的 Torch 实现，调用 convolution2d 命名空间的 conv2d_transpose_clamp_run 函数
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(prepacked, CPU, m) {
  // 实现 prepacked::linear_clamp_prepack 在 CPU 上的具体逻辑，使用 createLinearClampPrePackOpContext 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_prepack"), TORCH_FN(createLinearClampPrePackOpContext));
  // 实现 prepacked::linear_clamp_run 在 CPU 上的具体逻辑，调用 linear 命名空间的 linear_clamp_run 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_run"), TORCH_FN(internal::linear::linear_clamp_run));
  // 实现 prepacked::conv2d_clamp_prepack 在 CPU 上的具体逻辑，使用 createConv2dClampPrePackOpContext 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_prepack"), TORCH_FN(createConv2dClampPrePackOpContext));
  // 实现 prepacked::conv2d_transpose_clamp_prepack 在 CPU 上的具体逻辑，使用 createConv2dTransposeClampPrePackOpContext 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_prepack"), TORCH_FN(createConv2dTransposeClampPrePackOpContext));
  // 实现 prepacked::conv2d_clamp_run 在 CPU 上的具体逻辑，调用 convolution2d 命名空间的 conv2d_clamp_run 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_clamp_run));
  // 实现 prepacked::conv2d_transpose_clamp_run 在 CPU 上的具体逻辑，调用 convolution2d 命名空间的 conv2d_transpose_clamp_run 函数
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_transpose_clamp_run));
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
```