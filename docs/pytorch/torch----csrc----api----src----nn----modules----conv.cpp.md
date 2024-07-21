# `.\pytorch\torch\csrc\api\src\nn\modules\conv.cpp`

```
// 包含 Torch 中用于卷积操作的函数和模块的头文件
#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/modules/conv.h>

// 包含 Torch 中的基础数据类型和实用函数的头文件
#include <c10/util/irange.h>
#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

// 包含标准库头文件
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

// 将 torch::nn::functional 命名空间别名为 F
namespace F = torch::nn::functional;

// 定义一个静态函数 _get_pad_mode_from_conv_padding_mode，将 Conv2d 的 padding mode 转换为 PadFuncOptions 的 mode_t
static F::PadFuncOptions::mode_t _get_pad_mode_from_conv_padding_mode(
    torch::nn::detail::conv_padding_mode_t conv_padding_mode) {
  F::PadFuncOptions::mode_t pad_mode;
  // 根据不同的 conv_padding_mode 类型分配对应的 pad_mode
  if (std::holds_alternative<torch::enumtype::kReflect>(conv_padding_mode)) {
    pad_mode = torch::kReflect;
  } else if (std::holds_alternative<torch::enumtype::kReplicate>(conv_padding_mode)) {
    pad_mode = torch::kReplicate;
  } else if (std::holds_alternative<torch::enumtype::kCircular>(conv_padding_mode)) {
    pad_mode = torch::kCircular;
  } else {
    // 如果 conv_padding_mode 不是上述支持的类型，则抛出错误
    TORCH_CHECK(
        false,
        "Unsupported conv padding mode: ",
        torch::enumtype::get_enum_name(conv_padding_mode));
  }
  return pad_mode;
}

// 定义 torch 命名空间下的 nn 命名空间
namespace torch {
namespace nn {

// Conv1dImpl 类的构造函数，初始化 Conv1d 的参数
Conv1dImpl::Conv1dImpl(Conv1dOptions options_)
    : ConvNdImpl(detail::ConvNdOptions<1>(
                     /*in_channels=*/options_.in_channels(),
                     /*out_channels=*/options_.out_channels(),
                     /*kernel_size=*/options_.kernel_size())
                     .stride(options_.stride())
                     .padding(options_.padding())
                     .dilation(options_.dilation())
                     .transposed(false)
                     .output_padding(0)
                     .groups(options_.groups())
                     .bias(options_.bias())
                     .padding_mode(options_.padding_mode())) {}

// Conv1dImpl 类的前向传播函数，根据 padding_mode 不同选择合适的 padding 方式进行卷积
Tensor Conv1dImpl::forward(const Tensor& input) {
  // 如果 padding_mode 不是 kZeros 类型，则使用 PadFuncOptions 配置的 padding
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    return F::detail::conv1d(
        F::pad(
            input,
            // 使用 PadFuncOptions 配置的 padding mode
            F::PadFuncOptions(_reversed_padding_repeated_twice)
                .mode(_get_pad_mode_from_conv_padding_mode(
                    options.padding_mode()))),
        weight,
        bias,
        options.stride(),
        /*padding=*/0,
        options.dilation(),
        options.groups());
  }
  // 否则，使用 options 中的 padding 进行卷积操作
  return F::detail::conv1d(
      input,
      weight,
      bias,
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

// Conv2dImpl 类的构造函数声明
Conv2dImpl::Conv2dImpl(Conv2dOptions options_)
    # 使用 ConvNdOptions<2> 构造函数创建 ConvNdImpl 对象，参数为选项对象的属性
    : ConvNdImpl(detail::ConvNdOptions<2>(
                     # 设置输入通道数为 options_ 对象的 in_channels 属性
                     /*in_channels=*/options_.in_channels(),
                     # 设置输出通道数为 options_ 对象的 out_channels 属性
                     /*out_channels=*/options_.out_channels(),
                     # 设置核大小为 options_ 对象的 kernel_size 属性
                     /*kernel_size=*/options_.kernel_size())
                     # 设置步幅为 options_ 对象的 stride 属性
                     .stride(options_.stride())
                     # 设置填充为 options_ 对象的 padding 属性
                     .padding(options_.padding())
                     # 设置膨胀率为 options_ 对象的 dilation 属性
                     .dilation(options_.dilation())
                     # 设置是否转置为 false
                     .transposed(false)
                     # 设置输出填充为 0
                     .output_padding(0)
                     # 设置分组数为 options_ 对象的 groups 属性
                     .groups(options_.groups())
                     # 设置是否使用偏置为 options_ 对象的 bias 属性
                     .bias(options_.bias())
                     # 设置填充模式为 options_ 对象的 padding_mode 属性
                     .padding_mode(options_.padding_mode())) {}
// 返回类型为 Tensor 的 Conv2dImpl 类的 _conv_forward 方法，接受 input 和 weight 两个 Tensor 参数
Tensor Conv2dImpl::_conv_forward(const Tensor& input, const Tensor& weight) {
  // 检查 padding_mode 是否为非 kZeros，如果是则执行以下代码块
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    // 返回 F::detail::conv2d 函数的结果，该函数接受经过 F::pad 处理后的 input Tensor，pad 的参数通过 _reversed_padding_repeated_twice 指定
    return F::detail::conv2d(
        F::pad(
            input,
            F::PadFuncOptions(_reversed_padding_repeated_twice)
                .mode(_get_pad_mode_from_conv_padding_mode(
                    options.padding_mode()))),
        weight,
        bias,  // 使用当前对象的 bias 成员变量
        options.stride(),  // 使用 options 对象的 stride 成员变量
        /*padding=*/0,  // 明确指定 padding 为 0
        options.dilation(),  // 使用 options 对象的 dilation 成员变量
        options.groups());  // 使用 options 对象的 groups 成员变量
  }
  // 如果 padding_mode 为 kZeros，则执行以下代码块
  return F::detail::conv2d(
      input,  // 直接使用 input Tensor
      weight,  // 使用传入的 weight Tensor
      bias,  // 使用当前对象的 bias 成员变量
      options.stride(),  // 使用 options 对象的 stride 成员变量
      options.padding(),  // 使用 options 对象的 padding 成员变量
      options.dilation(),  // 使用 options 对象的 dilation 成员变量
      options.groups());  // 使用 options 对象的 groups 成员变量
}

// 返回类型为 Tensor 的 Conv2dImpl 类的 forward 方法，接受 input Tensor 参数
Tensor Conv2dImpl::forward(const Tensor& input) {
  // 调用当前对象的 _conv_forward 方法，传入 input 和 weight 成员变量作为参数
  return _conv_forward(input, weight);
}

// Conv3dImpl 类的构造函数，接受 Conv3dOptions 类型的参数 options_
Conv3dImpl::Conv3dImpl(Conv3dOptions options_)
    : ConvNdImpl(detail::ConvNdOptions<3>(
                     /*in_channels=*/options_.in_channels(),  // 从 options_ 参数中获取 in_channels
                     /*out_channels=*/options_.out_channels(),  // 从 options_ 参数中获取 out_channels
                     /*kernel_size=*/options_.kernel_size())  // 从 options_ 参数中获取 kernel_size
                     .stride(options_.stride())  // 使用 options_ 参数中的 stride 设置到 ConvNdOptions 对象中
                     .padding(options_.padding())  // 使用 options_ 参数中的 padding 设置到 ConvNdOptions 对象中
                     .dilation(options_.dilation())  // 使用 options_ 参数中的 dilation 设置到 ConvNdOptions 对象中
                     .transposed(false)  // 设置 transposed 属性为 false
                     .output_padding(0)  // 设置 output_padding 为 0
                     .groups(options_.groups())  // 使用 options_ 参数中的 groups 设置到 ConvNdOptions 对象中
                     .bias(options_.bias())  // 使用 options_ 参数中的 bias 设置到 ConvNdOptions 对象中
                     .padding_mode(options_.padding_mode())) {}  // 使用 options_ 参数中的 padding_mode 设置到 ConvNdOptions 对象中

// 返回类型为 Tensor 的 Conv3dImpl 类的 forward 方法，接受 input Tensor 参数
Tensor Conv3dImpl::forward(const Tensor& input) {
  // 检查 padding_mode 是否为非 kZeros，如果是则执行以下代码块
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    // 返回 F::detail::conv3d 函数的结果，该函数接受经过 F::pad 处理后的 input Tensor，pad 的参数通过 _reversed_padding_repeated_twice 指定
    return F::detail::conv3d(
        F::pad(
            input,
            F::PadFuncOptions(_reversed_padding_repeated_twice)
                .mode(_get_pad_mode_from_conv_padding_mode(
                    options.padding_mode()))),
        weight,  // 使用当前对象的 weight 成员变量
        bias,  // 使用当前对象的 bias 成员变量
        options.stride(),  // 使用 options 对象的 stride 成员变量
        /*padding=*/0,  // 明确指定 padding 为 0
        options.dilation(),  // 使用 options 对象的 dilation 成员变量
        options.groups());  // 使用 options 对象的 groups 成员变量
  }
  // 如果 padding_mode 为 kZeros，则执行以下代码块
  return F::detail::conv3d(
      input,  // 直接使用 input Tensor
      weight,  // 使用当前对象的 weight 成员变量
      bias,  // 使用当前对象的 bias 成员变量
      options.stride(),  // 使用 options 对象的 stride 成员变量
      options.padding(),  // 使用 options 对象的 padding 成员变量
      options.dilation(),  // 使用 options 对象的 dilation 成员变量
      options.groups());  // 使用 options 对象的 groups 成员变量
}

// 显式实例化 ConvNdImpl 模板类的 <1, Conv1dImpl> 版本
template class ConvNdImpl<1, Conv1dImpl>;
// 显式实例化 ConvNdImpl 模板类的 <2, Conv2dImpl> 版本
template class ConvNdImpl<2, Conv2dImpl>;
// 显式实例化 ConvNdImpl 模板类的 <3, Conv3dImpl> 版本
template class ConvNdImpl<3, Conv3dImpl>;

// ============================================================================

// ConvTransposeNdImpl<D, Derived> 类的 _output_padding 方法，接受 input Tensor 参数，以及几个 ExpandingArray<D> 类型的参数
template <size_t D, typename Derived>
std::vector<int64_t> ConvTransposeNdImpl<D, Derived>::_output_padding(
    const Tensor& input,
    const std::optional<at::IntArrayRef>& output_size,
    const ExpandingArray<D>& stride,
    const ExpandingArray<D>& padding,
    const ExpandingArray<D>& kernel_size) {
  // 定义返回的整型向量 ret
  std::vector<int64_t> ret;
  // 定义可选的 output_size_，初始化为传入的 output_size
  std::optional<at::IntArrayRef> output_size_ = output_size;

  // 如果 output_size_ 为 nullptr，则执行以下代码块
  if (output_size_ == c10::nullopt) {
    // 将当前对象的 options.output_padding() 转换为 IntArrayRef 后存入 ret
    ret = at::IntArrayRef(this->options.output_padding()).vec();
  } else {
    // 否则，计算 input 的维度数减去 2，存入 k 变量
    auto k = input.dim() - 2;
    // 如果 output_size_ 的维度与 k+2 相同，则取其从第二维开始的切片，并将结果重新赋值给 output_size_
    if (output_size_.value().size() == static_cast<size_t>(k + 2)) {
      output_size_ = output_size_.value().slice(2);
    }
    // 检查输出大小是否符合预期，若不符合则抛出错误信息
    if (output_size_.value().size() != static_cast<size_t>(k)) {
      TORCH_CHECK(
          false,
          "output_size must have ",
          k,
          " or ",
          k + 2,
          " elements (got ",
          output_size_.value().size(),
          ")");
    }

    // 初始化存放最小尺寸和最大尺寸的向量
    std::vector<int64_t> min_sizes;
    std::vector<int64_t> max_sizes;
    // 计算每个维度的最小尺寸和最大尺寸
    for (const auto d : c10::irange(k)) {
      int64_t dim_size =
          ((input.sizes()[d + 2] - 1) * (*stride)[d] - 2 * (*padding)[d] +
           (*kernel_size)[d]);
      min_sizes.push_back(dim_size);
      max_sizes.push_back(min_sizes[d] + (*stride)[d] - 1);
    }

    // 检查每个输出尺寸是否在有效范围内，若不在则抛出错误信息
    for (const auto i : c10::irange(output_size_.value().size())) {
      int64_t size = output_size_.value()[i];
      int64_t min_size = min_sizes[i];
      int64_t max_size = max_sizes[i];
      if (size < min_size || size > max_size) {
        TORCH_CHECK(
            false,
            "requested an output size of ",
            output_size_.value(),
            ", but valid sizes range "
            "from ",
            min_sizes,
            " to ",
            max_sizes,
            " (for an input of ",
            input.sizes().slice(2),
            ")");
      }
    }

    // 计算每个维度的输出尺寸并存储到返回结果中
    for (const auto d : c10::irange(k)) {
      ret.push_back(output_size_.value()[d] - min_sizes[d]);
    }
}

// ConvTranspose1dImpl 类的构造函数，初始化 ConvTransposeNdImpl 基类
ConvTranspose1dImpl::ConvTranspose1dImpl(ConvTranspose1dOptions options_)
    : ConvTransposeNdImpl(detail::ConvNdOptions<1>(
                              /*in_channels=*/options_.in_channels(),
                              /*out_channels=*/options_.out_channels(),
                              /*kernel_size=*/options_.kernel_size())
                              .stride(options_.stride())
                              .padding(options_.padding())
                              .dilation(options_.dilation())
                              .transposed(true)
                              .output_padding(options_.output_padding())
                              .groups(options_.groups())
                              .bias(options_.bias())
                              .padding_mode(options_.padding_mode())) {}

// ConvTranspose1dImpl 类的前向传播函数
Tensor ConvTranspose1dImpl::forward(
    const Tensor& input,
    const std::optional<at::IntArrayRef>& output_size) {
  // 检查 padding_mode 是否为 enumtype::kZeros，否则抛出错误
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(
        false, "Only `zeros` padding mode is supported for ConvTranspose1d");
  }

  // 获取 padding，并计算 output_padding
  const auto& pad = padding();
  std::vector<int64_t> output_padding = _output_padding(
      input, output_size, options.stride(), pad, options.kernel_size());

  // 调用 F::detail::conv_transpose1d 进行卷积转置操作
  return F::detail::conv_transpose1d(
      input,
      weight,
      bias,
      options.stride(),
      pad,
      output_padding,
      options.groups(),
      options.dilation());
}

// ConvTranspose2dImpl 类的构造函数，初始化 ConvTransposeNdImpl 基类
ConvTranspose2dImpl::ConvTranspose2dImpl(ConvTranspose2dOptions options_)
    : ConvTransposeNdImpl(detail::ConvNdOptions<2>(
                              /*in_channels=*/options_.in_channels(),
                              /*out_channels=*/options_.out_channels(),
                              /*kernel_size=*/options_.kernel_size())
                              .stride(options_.stride())
                              .padding(options_.padding())
                              .dilation(options_.dilation())
                              .transposed(true)
                              .output_padding(options_.output_padding())
                              .groups(options_.groups())
                              .bias(options_.bias())
                              .padding_mode(options_.padding_mode())) {}

// ConvTranspose2dImpl 类的前向传播函数
Tensor ConvTranspose2dImpl::forward(
    const Tensor& input,
    const std::optional<at::IntArrayRef>& output_size) {
  // 检查 padding_mode 是否为 enumtype::kZeros，否则抛出错误
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(
        false, "Only `zeros` padding mode is supported for ConvTranspose2d");
  }

  // 获取 padding，并计算 output_padding
  const auto& pad = padding();
  std::vector<int64_t> output_padding = _output_padding(
      input, output_size, options.stride(), pad, options.kernel_size());

  // 调用 F::detail::conv_transpose2d 进行卷积转置操作
  return F::detail::conv_transpose2d(
      input,
      weight,
      bias,
      options.stride(),
      pad,
      output_padding,
      options.groups(),
      options.dilation());
}

ConvTranspose3dImpl::ConvTranspose3dImpl(ConvTranspose3dOptions options_)
    # 定义 ConvTransposeNdImpl 构造函数，用于创建一个转置卷积层实现对象
    : ConvTransposeNdImpl(detail::ConvNdOptions<3>(
                              # 设置输入通道数为 options_ 对象中的设置值
                              /*in_channels=*/options_.in_channels(),
                              # 设置输出通道数为 options_ 对象中的设置值
                              /*out_channels=*/options_.out_channels(),
                              # 设置卷积核大小为 options_ 对象中的设置值
                              /*kernel_size=*/options_.kernel_size())
                              # 设置步长为 options_ 对象中的设置值
                              .stride(options_.stride())
                              # 设置填充大小为 options_ 对象中的设置值
                              .padding(options_.padding())
                              # 设置扩张大小为 options_ 对象中的设置值
                              .dilation(options_.dilation())
                              # 设置为转置卷积操作
                              .transposed(true)
                              # 设置输出填充大小为 options_ 对象中的设置值
                              .output_padding(options_.output_padding())
                              # 设置组数为 options_ 对象中的设置值
                              .groups(options_.groups())
                              # 设置是否使用偏置项为 options_ 对象中的设置值
                              .bias(options_.bias())
                              # 设置填充模式为 options_ 对象中的设置值
                              .padding_mode(options_.padding_mode())) {}
// 实现 ConvTranspose3dImpl 类的 forward 方法，接收输入张量 input 和可选的输出尺寸 output_size
Tensor ConvTranspose3dImpl::forward(
    const Tensor& input,
    const std::optional<at::IntArrayRef>& output_size) {
  // 检查填充模式是否为 `zeros`，如果不是，则抛出错误
  if (!std::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(
        false, "Only `zeros` padding mode is supported for ConvTranspose3d");
  }

  // 获取当前层的填充信息
  const auto& pad = padding();
  // 计算输出填充信息
  std::vector<int64_t> output_padding = _output_padding(
      input, output_size, options.stride(), pad, options.kernel_size());

  // 调用 F 命名空间下的 conv_transpose3d 函数进行反卷积操作
  return F::detail::conv_transpose3d(
      input,
      weight,
      bias,
      options.stride(),
      pad,
      output_padding,
      options.groups(),
      options.dilation());
}

// 实例化 ConvTransposeNdImpl 模板类，指定维度和具体的 ConvTransposeNdImpl 类型
template class ConvTransposeNdImpl<1, ConvTranspose1dImpl>;
template class ConvTransposeNdImpl<2, ConvTranspose2dImpl>;
template class ConvTransposeNdImpl<3, ConvTranspose3dImpl>;

} // namespace nn
} // namespace torch


这段代码的主要作用是实现了 ConvTranspose3dImpl 类的前向传播方法 `forward`，以及对 ConvTransposeNdImpl 模板类的实例化。
```