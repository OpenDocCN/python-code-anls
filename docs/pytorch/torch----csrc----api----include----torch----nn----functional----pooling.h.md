# `.\pytorch\torch\csrc\api\include\torch\nn\functional\pooling.h`

```
#pragma once

#include <c10/util/irange.h>  // 包含用于处理范围的头文件
#include <torch/nn/functional/activation.h>  // 包含神经网络相关的激活函数头文件
#include <torch/nn/modules/utils.h>  // 包含神经网络模块相关的实用函数头文件
#include <torch/nn/options/pooling.h>  // 包含池化操作选项相关的头文件

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool1d(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    bool ceil_mode,
    bool count_include_pad) {
  return torch::avg_pool1d(
      input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.avg_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AvgPool1dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));
/// ```
inline Tensor avg_pool1d(
    const Tensor& input,
    const AvgPool1dFuncOptions& options) {
  return avg_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  return torch::avg_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.avg_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AvgPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));
/// ```
inline Tensor avg_pool2d(
    const Tensor& input,
    const AvgPool2dFuncOptions& options) {
  return detail::avg_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool3d(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  // 调用 PyTorch 提供的 avg_pool3d 函数进行三维平均池化操作
  return torch::avg_pool3d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
    // 使用 Torch 库中的 avg_pool3d 函数进行三维平均池化操作
    // input: 输入的张量数据
    // kernel_size: 池化核的大小，用于指定池化窗口的尺寸
    // stride: 池化操作的步长，控制每次池化窗口移动的距离
    // padding: 对输入张量进行填充以适应池化操作的大小
    // ceil_mode: 指定是否使用 ceil 函数来计算输出大小
    // count_include_pad: 是否在计算平均值时包括填充部分的值
    // divisor_override: 可选参数，允许覆盖默认的除数值
  return torch::avg_pool3d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义内部函数 max_pool1d，用于执行一维最大池化操作
inline Tensor max_pool1d(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    bool ceil_mode) {
  // 调用 PyTorch 提供的 max_pool1d 函数进行一维最大池化操作
  return torch::max_pool1d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此函数的详细行为，访问以下链接：
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool1d
///
/// 若要了解此功能支持的可选参数，请参阅 `torch::nn::functional::MaxPool1dFuncOptions` 类的文档。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));
/// ```
inline Tensor max_pool1d(
    const Tensor& input,
    const MaxPool1dFuncOptions& options) {
  // 调用内部 detail 命名空间的 max_pool1d 函数，传递选项参数
  return detail::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
// 定义内部函数 max_pool1d_with_indices，执行带索引的一维最大池化操作
inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    bool ceil_mode) {
  // 调用 PyTorch 提供的 max_pool1d_with_indices 函数执行带索引的一维最大池化操作
  return torch::max_pool1d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 若要了解 `torch::nn::functional::MaxPool1dFuncOptions` 类支持的可选参数，请参阅其文档。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool1d_with_indices(x, F::MaxPool1dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& input,
    const MaxPool1dFuncOptions& options) {
    // 调用 max_pool1d_with_indices 函数，执行一维最大池化操作并返回结果
    return detail::max_pool1d_with_indices(
        input,                            // 输入张量
        options.kernel_size(),            // 池化核大小选项
        options.stride(),                 // 步幅选项
        options.padding(),                // 填充选项
        options.dilation(),               // 膨胀选项
        options.ceil_mode());             // 是否使用向上取整模式的选项
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    bool ceil_mode) {
  // 调用 PyTorch 提供的 max_pool2d 函数进行二维最大池化操作
  return torch::max_pool2d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));
/// ```
inline Tensor max_pool2d(
    const Tensor& input,
    const MaxPool2dFuncOptions& options) {
  // 调用 detail 命名空间中的 max_pool2d 函数，执行二维最大池化操作
  return detail::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    bool ceil_mode) {
  // 调用 PyTorch 提供的 max_pool2d_with_indices 函数，执行带索引的二维最大池化操作
  return torch::max_pool2d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for `torch::nn::functional::MaxPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool2d_with_indices(x, F::MaxPool2dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    const MaxPool2dFuncOptions& options) {
  // 调用 detail 命名空间中的 max_pool2d_with_indices 函数，执行带索引的二维最大池化操作
  return detail::max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_pool3d(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    bool ceil_mode) {
  // 调用 PyTorch 提供的 max_pool3d 函数，执行三维最大池化操作
  return torch::max_pool3d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxPool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// 调用torch库中的max_pool3d函数，使用给定的选项对输入进行三维最大池化操作
inline Tensor max_pool3d(
    const Tensor& input,
    const MaxPool3dFuncOptions& options) {
  return detail::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 在细节命名空间中定义函数，执行带索引的三维最大池化操作
inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    bool ceil_mode) {
  return torch::max_pool3d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 `torch::nn::functional::MaxPool3dFuncOptions` 类的文档，了解此函数的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool3d_with_indices(x, F::MaxPool3dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(
    const Tensor& input,
    const MaxPool3dFuncOptions& options) {
  return detail::max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 在细节命名空间中定义函数，执行自适应一维最大池化操作，返回输出张量及其索引
inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  return torch::adaptive_max_pool1d(input, output_size);
}
} // namespace detail

/// 查看 `torch::nn::functional::AdaptiveMaxPool1dFuncOptions` 类的文档，了解此函数的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool1d_with_indices(x, F::AdaptiveMaxPool1dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
    const Tensor& input,
    const AdaptiveMaxPool1dFuncOptions& options) {
  return detail::adaptive_max_pool1d_with_indices(input, options.output_size());
}

namespace detail {
/// 在细节命名空间中定义函数，执行自适应一维最大池化操作，仅返回输出张量
inline Tensor adaptive_max_pool1d(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  return std::get<0>(adaptive_max_pool1d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看以下链接了解此函数的具体行为：https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool1d
///
/// 查看 `torch::nn::functional::AdaptiveMaxPool1dFuncOptions` 类的文档，了解此函数的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// ```
/// 调用 adaptive_max_pool1d 函数，对输入张量进行自适应最大池化操作，输出大小为3
/// ```
inline Tensor adaptive_max_pool1d(
    const Tensor& input,
    const AdaptiveMaxPool1dFuncOptions& options) {
  return detail::adaptive_max_pool1d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现自适应最大池化操作，返回池化后的张量和对应的索引张量
inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_max_pool2d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 torch::nn::functional::AdaptiveMaxPool2dFuncOptions 类的文档，了解此功能支持的可选参数
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool2d_with_indices(x, F::AdaptiveMaxPool2dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
    const Tensor& input,
    const AdaptiveMaxPool2dFuncOptions& options) {
  return detail::adaptive_max_pool2d_with_indices(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现自适应最大池化操作，返回池化后的张量
inline Tensor adaptive_max_pool2d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  return std::get<0>(adaptive_max_pool2d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看此功能的确切行为，请参阅 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool2d
///
/// 查看 torch::nn::functional::AdaptiveMaxPool2dFuncOptions 类的文档，了解此功能支持的可选参数
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));
/// ```
inline Tensor adaptive_max_pool2d(
    const Tensor& input,
    const AdaptiveMaxPool2dFuncOptions& options) {
  return detail::adaptive_max_pool2d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现自适应最大池化操作，返回池化后的张量和对应的索引张量
inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_max_pool3d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 torch::nn::functional::AdaptiveMaxPool3dFuncOptions 类的文档，了解此功能支持的可选参数
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool3d_with_indices(x, F::AdaptiveMaxPool3dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
    // 定义一个函数，执行自适应三维最大池化操作，接受输入张量和自适应最大池化的选项参数
    const Tensor& input,
    // options 参数包含自适应最大池化的配置选项
    const AdaptiveMaxPool3dFuncOptions& options) {
  // 调用细节函数 detail::adaptive_max_pool3d_with_indices，执行具体的自适应最大池化操作，
  // 并传入输入张量 input 和输出大小参数 options.output_size()
  return detail::adaptive_max_pool3d_with_indices(input, options.output_size());
}
#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 如果没有定义 DOXYGEN_SHOULD_SKIP_THIS 宏，则包含以下代码块

namespace detail {
// 定义了一个命名空间 detail

inline Tensor adaptive_max_pool3d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  // 定义了一个内联函数 adaptive_max_pool3d，接受一个 Tensor 类型的 input 和一个 ExpandingArrayWithOptionalElem 类型的 output_size 参数
  // 调用 detail 命名空间下的 adaptive_max_pool3d_with_indices 函数，并返回其结果的第一个元素
  return std::get<0>(adaptive_max_pool3d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
// 结束命名空间 detail 并结束条件编译块

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));
/// ```
// 解释了 adaptive_max_pool3d 函数的功能和用法，包括了链接到 PyTorch 文档和示例代码

inline Tensor adaptive_max_pool3d(
    const Tensor& input,
    const AdaptiveMaxPool3dFuncOptions& options) {
  // 定义了一个内联函数 adaptive_max_pool3d，接受一个 Tensor 类型的 input 和一个 AdaptiveMaxPool3dFuncOptions 类型的 options 参数
  // 调用 detail 命名空间下的 adaptive_max_pool3d 函数，并返回其结果
  return detail::adaptive_max_pool3d(input, options.output_size());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 如果没有定义 DOXYGEN_SHOULD_SKIP_THIS 宏，则包含以下代码块

namespace detail {
// 定义了一个命名空间 detail

inline Tensor adaptive_avg_pool1d(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  // 定义了一个内联函数 adaptive_avg_pool1d，接受一个 Tensor 类型的 input 和一个 ExpandingArray<1> 类型的 output_size 参数
  // 调用 torch 命名空间下的 adaptive_avg_pool1d 函数，并返回其结果
  return torch::adaptive_avg_pool1d(input, output_size);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
// 结束命名空间 detail 并结束条件编译块

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveAvgPool1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));
/// ```
// 解释了 adaptive_avg_pool1d 函数的功能和用法，包括了链接到 PyTorch 文档和示例代码

inline Tensor adaptive_avg_pool1d(
    const Tensor& input,
    const AdaptiveAvgPool1dFuncOptions& options) {
  // 定义了一个内联函数 adaptive_avg_pool1d，接受一个 Tensor 类型的 input 和一个 AdaptiveAvgPool1dFuncOptions 类型的 options 参数
  // 调用 detail 命名空间下的 adaptive_avg_pool1d 函数，并返回其结果
  return detail::adaptive_avg_pool1d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// 如果没有定义 DOXYGEN_SHOULD_SKIP_THIS 宏，则包含以下代码块

namespace detail {
// 定义了一个命名空间 detail

inline Tensor adaptive_avg_pool2d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  // 定义了一个内联函数 adaptive_avg_pool2d，接受一个 Tensor 类型的 input 和一个 ExpandingArrayWithOptionalElem<2> 类型的 output_size 参数
  // 调用 torch 命名空间下的 adaptive_avg_pool2d 函数，并返回其结果
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_avg_pool2d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
// 结束命名空间 detail 并结束条件编译块

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveAvgPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));
/// ```
// 解释了 adaptive_avg_pool2d 函数的功能和用法，包括了链接到 PyTorch 文档和示例代码

inline Tensor adaptive_avg_pool2d(
    const Tensor& input,
    const AdaptiveAvgPool2dFuncOptions& options) {
  // 定义了一个内联函数 adaptive_avg_pool2d，接受一个 Tensor 类型的 input 和一个 AdaptiveAvgPool2dFuncOptions 类型的 options 参数
  // 调用 detail 命名空间下的 adaptive_avg_pool2d 函数，并返回其结果
  return detail::adaptive_avg_pool2d(input, options.output_size());
}
/// 这段代码定义了一个内部命名空间 detail，并提供了 adaptive_avg_pool3d 函数的实现。
/// 此函数用于执行自适应三维平均池化操作。
///
/// @param input 输入张量，即需要执行池化操作的数据
/// @param output_size 自适应平均池化操作的输出尺寸，可以是一个整数或者整数数组
/// @return 返回执行自适应三维平均池化操作后的张量
inline Tensor adaptive_avg_pool3d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  // 调用 torch::nn::modules::utils::_list_with_default 函数来处理 output_size
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  // 调用 torch::adaptive_avg_pool3d 函数执行池化操作并返回结果
  return torch::adaptive_avg_pool3d(input, output_size_);
}

/// 该命名空间提供了对 adaptive_avg_pool3d 函数的封装，以及相关文档链接和示例。
/// 详细了解此函数的行为，请参阅 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool3d
///
/// 要了解支持此功能的可选参数，请查看 `torch::nn::functional::AdaptiveAvgPool3dFuncOptions` 类的文档。
///
/// 示例用法：
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool3d(x, F::AdaptiveAvgPool3dFuncOptions(3));
/// ```
inline Tensor adaptive_avg_pool3d(
    const Tensor& input,
    const AdaptiveAvgPool3dFuncOptions& options) {
  // 调用内部命名空间 detail 的 adaptive_avg_pool3d 函数，传递 options 中的输出尺寸
  return detail::adaptive_avg_pool3d(input, options.output_size());
}

// ============================================================================

/// 根据输入张量的大小和池化参数计算解池化操作的输出尺寸。
///
/// @param input 输入张量，即需要执行解池化操作的数据
/// @param kernel_size 解池化核大小，一个整数数组
/// @param stride 解池化步长，一个整数数组
/// @param padding 解池化填充，一个整数数组
/// @param output_size 可选参数，指定输出尺寸的整数数组
/// @return 返回解池化操作的输出尺寸
inline std::vector<int64_t> _unpool_output_size(
    const Tensor& input,
    const IntArrayRef& kernel_size,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  // 获取输入张量的尺寸
  auto input_size = input.sizes();
  // 计算默认的输出尺寸
  std::vector<int64_t> default_size;
  for (const auto d : c10::irange(kernel_size.size())) {
    default_size.push_back(
        (input_size[input_size.size() - kernel_size.size() + d] - 1) *
            stride[d] +
        kernel_size[d] - 2 * padding[d]);
  }
  // 如果没有提供自定义输出尺寸，则返回默认尺寸
  if (!output_size) {
    return default_size;
  } else {
    // 如果提供了自定义输出尺寸，则进行进一步的验证和处理
    std::vector<int64_t> output_size_;
    if (output_size->size() == kernel_size.size() + 2) {
      output_size_ = IntArrayRef(*output_size).slice(2).vec();
    }
    // 检查自定义输出尺寸的有效性
    if (output_size_.size() != kernel_size.size()) {
      TORCH_CHECK(
          false,
          "output_size should be a sequence containing ",
          kernel_size.size(),
          " or ",
          kernel_size.size() + 2,
          " elements, but it has a length of '",
          output_size_.size(),
          "'");
    }
    for (const auto d : c10::irange(kernel_size.size())) {
      const auto min_size = default_size[d] - stride[d];
      const auto max_size = default_size[d] + stride[d];
      // 检查每个维度的输出尺寸是否在合理范围内
      if (!(min_size <= output_size_[d] && output_size_[d] <= max_size)) {
        TORCH_CHECK(
            false,
            "invalid output_size ",
            output_size_,
            " (dim ",
            d,
            " must be between ",
            min_size,
            " and ",
            max_size,
            ")");
      }
    }
    return output_size_;
  }
}
    // 定义函数 max_unpool，接收输入张量、池化核大小、步长、填充、输出尺寸的可选引用
    const std::optional<std::vector<int64_t>>& output_size) {
  // 计算未池化输出的大小，存储在 output_size_ 中
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);
  // 将尺寸维度扩展到 1
  output_size_.push_back(1);
  // 调用 PyTorch 的最大反池化函数 max_unpool2d，传入扩展维度后的输入、索引和输出尺寸
  return torch::max_unpool2d(
             input.unsqueeze(-1), indices.unsqueeze(-1), output_size_)
      // 压缩最后的维度
      .squeeze(-1);
/// 结束了前面的代码块
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 这个函数用于执行 1D 最大值反池化操作。
///
/// 详细行为请参考 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool1d
///
/// 要了解此功能支持的可选参数，请查看 `torch::nn::functional::MaxUnpool1dFuncOptions` 类的文档。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool1d(x, indices,
///                 F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool1d(
    const Tensor& input,                           // 输入张量
    const Tensor& indices,                         // 池化时记录的索引张量
    const MaxUnpool1dFuncOptions& options) {       // 反池化操作的选项
  return detail::max_unpool1d(
      input,                                      // 输入张量
      indices,                                    // 池化时记录的索引张量
      options.kernel_size(),                      // 池化核的大小
      options.stride(),                           // 步幅大小
      options.padding(),                          // 填充大小
      options.output_size());                     // 输出大小
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 这个函数执行 2D 最大值反池化操作。
///
/// 详细行为请参考 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool2d
///
/// 要了解此功能支持的可选参数，请查看 `torch::nn::functional::MaxUnpool2dFuncOptions` 类的文档。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool2d(x, indices,
///                 F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool2d(
    const Tensor& input,                           // 输入张量
    const Tensor& indices,                         // 池化时记录的索引张量
    ExpandingArray<2> kernel_size,                 // 池化核的大小
    ExpandingArray<2> stride,                      // 步幅大小
    ExpandingArray<2> padding,                     // 填充大小
    const std::optional<std::vector<int64_t>>& output_size) {  // 输出大小的可选参数
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);

  return torch::max_unpool2d(input, indices, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxUnpool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool2d(x, indices,
///                 F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool2d(
    const Tensor& input,                           // 输入张量
    const Tensor& indices,                         // 池化时记录的索引张量
    const MaxUnpool2dFuncOptions& options) {       // 反池化操作的选项
  return detail::max_unpool2d(
      input,                                      // 输入张量
      indices,                                    // 池化时记录的索引张量
      options.kernel_size(),                      // 池化核的大小
      options.stride(),                           // 步幅大小
      options.padding(),                          // 填充大小
      options.output_size());                     // 输出大小
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 这个函数执行 3D 最大值反池化操作。
///
/// 详细行为请参考 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool3d
///
/// 要了解此功能支持的可选参数，请查看 `torch::nn::functional::MaxUnpool3dFuncOptions` 类的文档。
///
/// 示例:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool3d(x, indices,
///                 F::MaxUnpool3dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool3d(
    const Tensor& input,                           // 输入张量
    const Tensor& indices,                         // 池化时记录的索引张量
    ExpandingArray<3> kernel_size,                 // 池化核的大小
    ExpandingArray<3> stride,                      // 步幅大小
    ExpandingArray<3> padding,                     // 填充大小
    const std::optional<std::vector<int64_t>>& output_size) {  // 输出大小的可选参数
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);

  return torch::max_unpool3d(input, indices, output_size_, stride, padding);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxUnpool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// 调用细节函数 `detail::max_unpool3d` 来执行三维最大解池操作。
inline Tensor max_unpool3d(
    const Tensor& input,
    const Tensor& indices,
    const MaxUnpool3dFuncOptions& options) {
  return detail::max_unpool3d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.output_size());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现带索引的二维分数最大池化操作。
inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
    const Tensor& input,
    const ExpandingArray<2>& kernel_size,
    const std::optional<ExpandingArray<2>>& output_size,
    const std::optional<ExpandingArray<2, double>>& output_ratio,
    const Tensor& _random_samples) {
  // 如果未指定 output_size 或 output_ratio，则抛出错误
  if (output_size == c10::nullopt && output_ratio == c10::nullopt) {
    TORCH_CHECK(
        false,
        "fractional_max_pool2d requires specifying either ",
        "an output_size or an output_ratio");
  }
  
  // 根据情况计算 output_size_
  std::optional<ExpandingArray<2>> output_size_ = output_size;
  if (output_size_ == c10::nullopt) {
    TORCH_INTERNAL_ASSERT(output_ratio != c10::nullopt);
    output_size_ = {
        (int64_t)(static_cast<double>(input.size(-2)) *
                  (*output_ratio.value())[0]),
        (int64_t)(static_cast<double>(input.size(-1)) *
                  (*output_ratio.value())[1])};
  }

  // 如果未定义随机样本，则生成一个符合要求的随机样本
  Tensor _random_samples_ = _random_samples;
  if (!_random_samples_.defined()) {
    auto n_batch = input.dim() == 3 ? 1 : input.size(0);
    _random_samples_ = torch::rand(
        {n_batch, input.size(-3), 2},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  }

  // 调用 Torch 中的分数最大池化函数
  return torch::fractional_max_pool2d(
      input, kernel_size, *output_size_, _random_samples_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看 `torch::nn::functional::FractionalMaxPool2dFuncOptions` 类的文档以了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d_with_indices(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
    const Tensor& input,
    const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 实现分数最大池化操作。
inline Tensor fractional_max_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    std::optional<ExpandingArray<2>> output_size,
    std::optional<ExpandingArray<2, double>> output_ratio,
    const Tensor& _random_samples) {
  // 调用带索引的分数最大池化函数，并返回其结果的第一个元素（即池化后的结果）
  return std::get<0>(fractional_max_pool2d_with_indices(
      input, kernel_size, output_size, output_ratio, _random_samples));
}
} // namespace detail
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// 在文档中查看 `torch::nn::functional::FractionalMaxPool2dFuncOptions` 类，了解这个功能支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 使用输入张量和选项执行二维分数最大池化操作。
inline Tensor fractional_max_pool2d(
    const Tensor& input,
    const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 在三维张量上执行带索引的分数最大池化操作。
inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(
    const Tensor& input,
    const ExpandingArray<3>& kernel_size,
    const std::optional<ExpandingArray<3>>& output_size,
    const std::optional<ExpandingArray<3, double>>& output_ratio,
    const Tensor& _random_samples) {
  if (output_size == c10::nullopt && output_ratio == c10::nullopt) {
    TORCH_CHECK(
        false,
        "fractional_max_pool3d requires specifying either ",
        "an output_size or an output_ratio");
  }

  std::optional<ExpandingArray<3>> output_size_ = output_size;
  if (output_size_ == c10::nullopt) {
    TORCH_INTERNAL_ASSERT(output_ratio != c10::nullopt);
    output_size_ = {
        (int64_t)(static_cast<double>(input.size(-3)) *
                  (*output_ratio.value())[0]),
        (int64_t)(static_cast<double>(input.size(-2)) *
                  (*output_ratio.value())[1]),
        (int64_t)(static_cast<double>(input.size(-1)) *
                  (*output_ratio.value())[2])};
  }

  Tensor _random_samples_ = _random_samples;
  if (!_random_samples_.defined()) {
    auto n_batch = input.dim() == 4 ? 1 : input.size(0);
    _random_samples_ = torch::rand(
        {n_batch, input.size(-4), 3},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  }
  return torch::fractional_max_pool3d(
      input, kernel_size, *output_size_, _random_samples_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看文档以了解 `torch::nn::functional::FractionalMaxPool3dFuncOptions` 类支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d_with_indices(x,
/// F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(
    const Tensor& input,
    const FractionalMaxPool3dFuncOptions& options) {
  return detail::fractional_max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
/// 在三维张量上执行分数最大池化操作。
inline Tensor fractional_max_pool3d(
    const Tensor& input,
    // 调用 fractional_max_pool3d_with_indices 函数进行三维分数最大池化操作，并返回其结果的第一个元素
    return std::get<0>(fractional_max_pool3d_with_indices(
        // 输入张量，用于进行三维分数最大池化操作
        input,
        // 池化核大小，指定池化操作的窗口大小
        kernel_size,
        // 输出尺寸的可选参数，指定池化后的输出尺寸
        output_size,
        // 输出比例的可选参数，指定池化后的输出比例
        output_ratio,
        // 随机采样的张量，用于池化过程中的随机采样
        _random_samples));
/// 结束 `detail` 命名空间，这里定义了一些内部函数，用于实现具体的池化操作
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 此函数实现了 3D 分数最大池化操作，参考 `torch::nn::functional::FractionalMaxPool3dFuncOptions` 类的文档了解可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d(x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
inline Tensor fractional_max_pool3d(
    const Tensor& input,
    const FractionalMaxPool3dFuncOptions& options) {
  // 调用内部函数 `detail::fractional_max_pool3d` 来执行实际的池化操作
  return detail::fractional_max_pool3d(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// 开始 `detail` 命名空间，这里定义了一些内部函数，用于实现具体的池化操作
namespace detail {
/// 实现 1D LP 池化操作，内部函数，用于 `torch::nn::functional::lp_pool1d` 函数。
inline Tensor lp_pool1d(
    const Tensor& input,
    double norm_type,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    bool ceil_mode) {
  // 对输入进行幂运算，然后调用 `detail::avg_pool1d` 函数执行平均池化
  Tensor out = detail::avg_pool1d(
      input.pow(norm_type),
      kernel_size,
      stride,
      /*padding=*/0,
      ceil_mode,
      /*count_include_pad=*/true);

  // 应用 relu 函数，计算绝对值并乘以池化核大小，再进行幂运算
  return (torch::sign(out) * relu(torch::abs(out)))
      .mul((*kernel_size)[0])
      .pow(1. / norm_type);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 此函数实现了 1D LP 池化操作，参考 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.lp_pool1d 获取详细信息。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool1d(x, F::LPPool1dFuncOptions(2, 3).stride(2));
/// ```
inline Tensor lp_pool1d(
    const Tensor& input,
    const LPPool1dFuncOptions& options) {
  // 调用内部函数 `detail::lp_pool1d` 来执行实际的 1D LP 池化操作
  return detail::lp_pool1d(
      input,
      options.norm_type(),
      options.kernel_size(),
      options.stride(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// 开始 `detail` 命名空间，这里定义了一些内部函数，用于实现具体的池化操作
namespace detail {
/// 实现 2D LP 池化操作，内部函数，用于 `torch::nn::functional::lp_pool2d` 函数。
inline Tensor lp_pool2d(
    const Tensor& input,
    double norm_type,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    bool ceil_mode) {
  int kw = (*kernel_size)[0];
  int kh = (*kernel_size)[1];
  // 对输入进行幂运算，然后调用 `detail::avg_pool2d` 函数执行平均池化
  Tensor out = detail::avg_pool2d(
      input.pow(norm_type),
      kernel_size,
      stride,
      /*padding=*/0,
      ceil_mode,
      /*count_include_pad=*/true,
      /*divisor_override=*/c10::nullopt);

  // 应用 relu 函数，计算绝对值并乘以池化核大小乘积，再进行幂运算
  return (torch::sign(out) * relu(torch::abs(out)))
      .mul(kw * kh)
      .pow(1. / norm_type);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 此函数实现了 2D LP 池化操作，参考 https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.lp_pool2d 获取详细信息。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool2d(x, F::LPPool2dFuncOptions(2, 3).stride(2));
/// ```
inline Tensor lp_pool2d(
    const Tensor& input,
    const LPPool2dFuncOptions& options) {
  // 调用内部函数 `detail::lp_pool2d` 来执行实际的 2D LP 池化操作
  return detail::lp_pool2d(
      input,
      options.norm_type(),
      options.kernel_size(),
      options.stride(),
      options.ceil_mode());
}
/// 命名空间别名，简化命名空间访问
namespace F = torch::nn::functional;

/// 调用 torch::nn::functional 命名空间下的 lp_pool2d 函数
/// 使用给定的 LPPool2dFuncOptions 参数进行 2D Lp 池化操作
/// 参数 x 是输入张量，使用 {2, 3} 作为参数创建 LPPool2dFuncOptions 对象，并设定步长为 2
F::lp_pool2d(x, F::LPPool2dFuncOptions(2, {2, 3}).stride(2));
```