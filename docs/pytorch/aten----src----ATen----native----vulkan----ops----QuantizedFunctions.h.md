# `.\pytorch\aten\src\ATen\native\vulkan\ops\QuantizedFunctions.h`

```py
// 引入 ATen 库中 Vulkan 实现的公共头文件
#include <ATen/native/vulkan/ops/Common.h>

// ATen 命名空间
namespace at {
// ATen 下的 native 命名空间
namespace native {
// Vulkan 实现下的 ops 命名空间
namespace vulkan {
// Vulkan 实现下 ops 命名空间内的函数定义

// 对输入张量进行整体量化，返回量化后的张量
Tensor quantize_per_tensor(
    const at::Tensor& input_arg, // 输入张量
    const double scale, // 量化的比例因子
    const int64_t zero_point, // 量化的零点
    const c10::ScalarType dtype); // 数据类型

// 对输入张量根据给定的量化参数进行整体量化，返回量化后的张量
Tensor quantize_per_tensor_tensor_qparams(
    const at::Tensor& input_arg, // 输入张量
    const at::Tensor& scale, // 量化的比例因子张量
    const at::Tensor& zero_point, // 量化的零点张量
    const c10::ScalarType dtype); // 数据类型

// 对输入张量进行反量化，返回反量化后的张量
Tensor dequantize_helper(
    const at::Tensor& input_arg, // 输入张量
    const double scale, // 反量化的比例因子
    const int64_t zero_point, // 反量化的零点
    const c10::ScalarType dtype); // 数据类型

// 对量化张量进行反量化，返回反量化后的张量
Tensor dequantize(const Tensor& self); // 输入量化张量

// 量化加法，返回两个量化张量相加后的结果张量
Tensor quantized_add(
    const Tensor& self_arg, // 第一个输入量化张量
    const Tensor& other_arg, // 第二个输入量化张量
    const double scale, // 量化的比例因子
    const int64_t zero_point); // 量化的零点

// 量化减法，返回两个量化张量相减后的结果张量
Tensor quantized_sub(
    const Tensor& self_arg, // 第一个输入量化张量
    const Tensor& other_arg, // 第二个输入量化张量
    const double scale, // 量化的比例因子
    const int64_t zero_point); // 量化的零点

// 量化乘法，返回两个量化张量相乘后的结果张量
Tensor quantized_mul(
    const Tensor& self_arg, // 第一个输入量化张量
    const Tensor& other_arg, // 第二个输入量化张量
    const double scale, // 量化的比例因子
    const int64_t zero_point); // 量化的零点

// 量化除法，返回两个量化张量相除后的结果张量
Tensor quantized_div(
    const Tensor& self_arg, // 第一个输入量化张量
    const Tensor& other_arg, // 第二个输入量化张量
    const double scale, // 量化的比例因子
    const int64_t zero_point); // 量化的零点

// 二维量化卷积操作，返回卷积操作后的量化结果张量
Tensor quantized_conv2d(
    const Tensor& input_, // 输入张量
    const Tensor& weight, // 卷积核张量
    const std::optional<Tensor>& bias_opt, // 可选的偏置张量
    IntArrayRef stride, // 步幅
    IntArrayRef padding, // 填充
    IntArrayRef dilation, // 空洞卷积扩张率
    int64_t groups, // 分组卷积的组数
    double out_scale, // 输出量化的比例因子
    int64_t out_zero_point); // 输出量化的零点

// 二维最近邻上采样操作，返回上采样后的量化结果张量
Tensor quantized_upsample_nearest2d(
    const Tensor& input_arg, // 输入张量
    const IntArrayRef output_sizes, // 输出大小数组
    const std::optional<double> scales_h, // 可选的高度缩放因子
    const std::optional<double> scales_w); // 可选的宽度缩放因子

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```