# `.\pytorch\aten\src\ATen\native\DilatedConvolutionUtils.h`

```
/*
#pragma once

#include <algorithm>
#include <vector>

#include <ATen/div_rtn.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

// 定义宏 TORCH_CHECK_DIM_SIZE，用于检查张量 T 的维度和指定维度的大小是否符合预期
#define TORCH_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE) \
  TORCH_CHECK(                                       \
      T.dim() == DIM && T.size(DIM_SIZE) == SIZE,    \
      "Need " #T " of dimension ",                   \
      DIM,                                           \
      " and " #T ".size[",                           \
      DIM_SIZE,                                      \
      "] == ",                                       \
      SIZE,                                          \
      " but got input to be of shape ",              \
      T.sizes())

// 定义在 at::native::internal 命名空间下的匿名命名空间，用于包含辅助函数
namespace at::native::internal {
namespace {

// 内联函数 all_positive，用于检查 IntArrayRef 中的所有元素是否均为正数
inline bool all_positive(IntArrayRef& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

// 内联函数 all_nonnegative，用于检查 std::vector<int64_t> 中的所有元素是否均为非负数
inline bool all_nonnegative(std::vector<int64_t>& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item >= 0; });
}

} // namespace

// 模板函数 get_output_size，计算输出张量的尺寸
// 参数：input 输入张量, kernel_size 卷积核大小, stride_size 步幅大小, pad_size 填充大小, dilation_size 膨胀大小
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  std::vector<int64_t> sizes;
  for (const auto index : c10::irange(dim)) {
    // 计算输出张量在当前维度上的尺寸
    sizes.push_back(
        div_rtn<int64_t>(
            input.size(index + input.dim() - dim) + 2 * pad_size[index] -
                (dilation_size[index] * (kernel_size[index] - 1) + 1),
            stride_size[index]) +
        1);
  }
  return sizes;
}

// 模板函数 get_output_size，计算输出张量的尺寸（重载版本，包含 weight 参数）
// 参数：input 输入张量, weight 卷积核张量, kernel_size 卷积核大小, stride_size 步幅大小, pad_size 填充大小, dilation_size 膨胀大小
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // 调用 get_output_size<dim> 获取输出尺寸
  auto output_size = get_output_size<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);
  // 将 weight 的第一维度大小插入到输出尺寸的最前面
  output_size.insert(output_size.begin(), weight.size(0));
  // 如果输入张量的维度为 dim + 2，则将输入张量的第一维度大小也插入到输出尺寸的最前面
  if (input.dim() == dim + 2) {
    output_size.insert(output_size.begin(), input.size(0));
  }
  return output_size;
}

/*
  slow_conv_dilated_shape_check - 检查用户输入的扩展卷积前向和反向函数的形状。
*/
// 模板函数 slow_conv_dilated_shape_check，用于检查扩展卷积的形状
// 参数：input 输入张量, weight 卷积核张量, bias 偏置张量, grad_output 梯度输出张量, kernel_size 卷积核大小, stride_size 步幅大小, pad_size 填充大小, dilation_size 膨胀大小
template <int64_t dim>
void slow_conv_dilated_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  /*
    当以下张量被定义时：

    bias, grad_weight, grad_output

    假设这些张量是连续的，不需要检查，
    因为这些张量在前向/反向函数中调用 .contiguous() 方法时或者通过
    调整零大小张量的大小来使它们连续。

    当 grad_weight 被定义时，则假设其连续而无需

*/
    // 检查 kernel_size 的长度是否与维度 dim 匹配
    TORCH_CHECK(
        kernel_size.size() == dim,
        "kernel sizes length should be ",
        dim,
        ", but got ",
        kernel_size.size());
    
    // 检查 stride_size 的长度是否与维度 dim 匹配
    TORCH_CHECK(
        stride_size.size() == dim,
        "strides length should be ",
        dim,
        ", but got ",
        stride_size.size());
    
    // 检查 dilation_size 的长度是否与维度 dim 匹配
    TORCH_CHECK(
        dilation_size.size() == dim,
        "dilations length should be ",
        dim,
        ", but got ",
        dilation_size.size());
    
    // 检查 pad_size 的长度是否与维度 dim 匹配
    TORCH_CHECK(
        pad_size.size() == dim,
        "pads length should be ",
        dim,
        ", but got ",
        pad_size.size());
    
    // 检查 kernel_size 的所有元素是否大于零
    TORCH_CHECK(
        all_positive(kernel_size),
        "kernel size should be greater than zero, but got ",
        kernel_size);
    
    // 检查 stride_size 的所有元素是否大于零
    TORCH_CHECK(
        all_positive(stride_size),
        "stride should be greater than zero, but got ",
        stride_size);
    
    // 检查 dilation_size 的所有元素是否大于零
    TORCH_CHECK(
        all_positive(dilation_size),
        "dilation should be greater than zero, but got ",
        dilation_size);
    
    // 检查输入 input 是否已定义
    TORCH_CHECK(input.defined(), "input must be defined");
    
    // 检查是否为批处理，即 input 的维度是否为 dim + 2
    bool is_batch = input.dim() == dim + 2;
    int64_t n = (is_batch ? 2 : 1); // 如果是批处理则 n=2，否则 n=1
    int64_t ndim = n + dim; // 计算总维度
    
    // 如果不是批处理，则 input 的维度必须为 dim + 1
    if (!is_batch) {
        TORCH_CHECK(
            input.dim() == dim + 1,
            "input must be 4D or 5D tensor but got ",
            input.dim(),
            "D tensor");
    }
    
    // 获取计算后的输出尺寸
    auto output_size = get_output_size<dim>(
        input, kernel_size, stride_size, pad_size, dilation_size);
    
    // 检查输出尺寸是否全部为非负数
    TORCH_CHECK(
        all_nonnegative(output_size),
        "calculated output size ",
        output_size,
        " is too small (all sizes must be non-negative)");
    
    // 检查权重 weight 是否已定义
    TORCH_CHECK(weight.defined(), "weight must be defined");
    
    // 检查权重 weight 的维度是否为 dim + 2
    TORCH_CHECK(
        weight.dim() == dim + 2,
        "weight must be ",
        dim + 2,
        "D tensor but got ",
        weight.dim(),
        "D tensor dim=",
        dim);
    
    // 检查权重 weight 的后两维度是否与 kernel_size 相匹配
    TORCH_CHECK(
        weight.sizes().slice(2) == kernel_size,
        "weight[2:] shape ",
        weight.sizes().slice(2),
        " must be equal to kernel_size ",
        kernel_size);
    
    // 检查 input 的第 (is_batch ? 1 : 0) 维度是否与 weight 的第二维度大小相匹配
    TORCH_CHECK_DIM_SIZE(input, input.dim(), (is_batch ? 1 : 0), weight.size(1));
    
    // 当存在偏置 bias 时进行检查
    if (bias.defined()) {
        TORCH_CHECK(
            bias.dim() == 1,
            "bias must be 1D tensor but got ",
            bias.dim(),
            "D tensor");
    
        // 检查偏置 bias 的大小是否与 weight 的第一维度大小相匹配
        TORCH_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
    }
    
    // 当存在梯度输出 grad_output 时进行检查
    if (grad_output.defined()) {
        // 检查 grad_output 的维度是否为 ndim
        TORCH_CHECK(
            grad_output.dim() == ndim,
            "grad_output must be ",
            ndim,
            "D tensor but got ",
            grad_output.dim(),
            "D tensor");
    
        // 如果是批处理，则检查 grad_output 的第一维度大小是否与 input 的第一维度大小相等
        if (is_batch) {
            TORCH_CHECK(
                grad_output.size(0) == input.size(0),
                "grad_output.size(0)=",
                grad_output.size(0),
                " must be input.size(0)=",
                input.size(0));
        }
    }
    # 检查梯度输出的维度是否与权重的大小匹配
    TORCH_CHECK(
        grad_output.size(n - 1) == weight.size(0),  # 检查 grad_output 在第 (n-1) 维的大小是否等于 weight 的第一个维度大小
        "grad_output.size(",                      # 错误信息的开头，指示问题出现在 grad_output 的大小
        n - 1,                                    # 当前检查的维度索引
        ")=",                                     
        grad_output.size(n - 1),                  # 实际的 grad_output 在当前维度的大小
        " must be weight.size(0)=",               # 错误信息的一部分，指示期望 grad_output 在当前维度与 weight 第一个维度大小相等
        weight.size(0));                          # 实际的 weight 的第一个维度大小
    
    # 检查梯度输出的切片是否与指定的输出大小匹配
    TORCH_CHECK(
        grad_output.sizes().slice(n) == output_size,  # 检查 grad_output 在从第 n 维到结束的切片大小是否等于指定的 output_size
        "grad_output[",                             # 错误信息的开头，指示问题出现在 grad_output 的切片大小
        n,                                          # 当前检查的起始维度索引
        ":] shape",                                 # 错误信息的一部分，指示问题出现在 grad_output 的切片
        grad_output.sizes().slice(n),               # 实际的 grad_output 在切片维度的大小
        " must be equal to output size ",           # 错误信息的一部分，指示期望 grad_output 在切片维度与 output_size 相等
        output_size);                               # 指定的输出大小
    }
}

} // namespace at::native::internal
```