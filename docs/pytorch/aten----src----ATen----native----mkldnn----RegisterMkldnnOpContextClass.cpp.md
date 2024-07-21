# `.\pytorch\aten\src\ATen\native\mkldnn\RegisterMkldnnOpContextClass.cpp`

```py
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()  // 检查是否启用了 MKLDNN

#include <ATen/Tensor.h>  // 包含 ATen 库中的 Tensor 类定义
#include <ATen/native/mkldnn/ConvPrepack.h>  // 包含 MKLDNN 相关的 ConvPrepack 函数定义
#include <ATen/native/mkldnn/OpContext.h>  // 包含 MKLDNN 相关的 OpContext 定义
#include <ATen/native/mkldnn/Utils.h>  // 包含 MKLDNN 相关的实用函数定义
#include <torch/custom_class.h>  // 包含 Torch 自定义类的相关定义
#include <torch/library.h>  // 包含 Torch 库相关定义

namespace at {
namespace native {
namespace mkldnn {

using namespace internal::convolution;  // 使用命名空间 internal::convolution

// 检查当前设备是否支持 MKLDNN 的 BF16 数据类型
static bool is_mkldnn_bf16_supported() {
#if defined(__aarch64__)
  return mkldnn_bf16_device_check_arm();  // 在 ARM 架构上检查 BF16 支持情况
#else
  return mkldnn_bf16_device_check();  // 在其他架构上检查 BF16 支持情况
#endif
}

// 检查当前设备是否支持 MKLDNN 的 FP16 数据类型
static bool is_mkldnn_fp16_supported() {
  return mkldnn_fp16_device_check();  // 检查 FP16 支持情况
}

// 检查当前设备是否支持 MKLDNN 的 ACL 扩展
constexpr bool is_mkldnn_acl_supported() {
  return AT_MKLDNN_ACL_ENABLED();  // 返回 ACL 是否启用的状态
}

}

// 注册 MKLDNN 相关的预打包操作库
TORCH_LIBRARY(mkldnn_prepacked, m) {
  // 定义 conv2d_prepack 操作的 Torch 架构
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, int[4] input_size, str attr) -> __torch__.torch.classes.mkldnn.ConvOpContext"));

  // 定义 conv2d_run 操作的 Torch 架构
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> Tensor Y"));
}

// 实现 MKLDNN 相关预打包操作库在 CPU 上的具体实现
TORCH_LIBRARY_IMPL(mkldnn_prepacked, CPU, m) {
  // 实现 conv2d_prepack 操作的具体函数
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_prepack"),
      TORCH_FN(createConvPrePackOpContext));

  // 实现 conv2d_run 操作的具体函数
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_run"), TORCH_FN(conv_run));
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()

#if AT_MKL_ENABLED() && AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkl {

// 注册 MKL 相关的操作库
TORCH_LIBRARY(mkl, m) {
  // 定义 _mkl_reorder_linear_weight 操作的 Torch 架构
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkl::_mkl_reorder_linear_weight(Tensor X, int batch_size) -> Tensor"));
  // 定义 _mkl_linear 操作的 Torch 架构
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkl::_mkl_linear(Tensor X, Tensor MKL_W, Tensor ORI_W, Tensor? B, int batch_size) -> Tensor"));
}

} // namespace mkl
} // namespace native
} // namespace at

#endif // AT_MKL_ENABLED && AT_MKLDNN_ENABLED
```