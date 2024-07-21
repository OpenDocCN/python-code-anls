# `.\pytorch\torch\csrc\api\include\torch\nn\functional\upsampling.h`

```
// 防止头文件被多次包含
#pragma once

// 包含必要的头文件
#include <c10/util/irange.h>  // 用于迭代范围的实用工具
#include <torch/nn/functional/pooling.h>  // 池化相关功能
#include <torch/nn/options/upsampling.h>  // 上采样选项

#include <cmath>  // 数学函数库
#include <utility>  // 实用工具库

// 命名空间定义：torch -> nn -> functional
namespace torch {
namespace nn {
namespace functional {

// 内联函数定义：_interp_output_size
inline std::vector<int64_t> _interp_output_size(
    int64_t dim,  // 输入张量的维度数
    std::tuple<
        Tensor,  // 输入张量
        std::optional<std::vector<int64_t>>,  // 可选的输出尺寸
        std::optional<std::vector<double>>,  // 可选的缩放因子
        std::optional<bool>> closed_over_args) {  // 闭包参数

  // 解构元组参数
  auto [input, size, scale_factor, recompute_scale_factor] = closed_over_args;

  // 如果既未定义尺寸也未定义缩放因子，则抛出错误
  if (size == c10::nullopt && scale_factor == c10::nullopt) {
    TORCH_CHECK(false, "either size or scale_factor should be defined");
  }

  // 如果尺寸和缩放因子同时定义，则抛出错误
  if (size != c10::nullopt && scale_factor != c10::nullopt) {
    TORCH_CHECK(false, "only one of size or scale_factor should be defined");
  }

  // 如果缩放因子定义了，则检查其形状是否匹配输入形状
  if (scale_factor != c10::nullopt) {
    if (static_cast<int64_t>(scale_factor.value().size()) != dim) {
      TORCH_CHECK(
          false,
          "scale_factor shape must match input shape. ",
          "Input is ",
          dim,
          "D, scale_factor size is ",
          torch::ArrayRef<double>(*scale_factor));
    }
  }

  // 如果定义了尺寸，则直接返回该尺寸
  if (size != c10::nullopt) {
    return *size;
  }

  // 确保缩放因子已定义
  TORCH_INTERNAL_ASSERT(scale_factor != c10::nullopt);
  auto scale_factors = *scale_factor;

  // 当未定义是否重新计算缩放因子时，对浮点缩放因子进行警告
  if (recompute_scale_factor == c10::nullopt) {
    bool is_float_scale_factor = false;
    for (double scale : scale_factors) {
      is_float_scale_factor = floor(scale) != scale;
      if (is_float_scale_factor) {
        break;
      }
    }
    if (is_float_scale_factor) {
      TORCH_WARN(
          "The default behavior for interpolate/upsample with float scale_factor changed "
          "in 1.6.0 to align with other frameworks/libraries, and uses scale_factor directly, "
          "instead of relying on the computed output size. "
          "If you wish to keep the old behavior, please set recompute_scale_factor=True. "
          "See the documentation of nn.Upsample for details. ");
    }
  }

  // 计算并返回插值后的输出尺寸
  std::vector<int64_t> ret;
  for (const auto i : c10::irange(dim)) {
    ret.emplace_back(static_cast<int64_t>(
        floor(static_cast<double>(input.size(i + 2)) * scale_factors[i])));
  }
  return ret;
}

// 以下代码未完整展示，无法提供进一步注释
    if (align_corners != c10::nullopt) {
      // 如果 align_corners 参数不为空，抛出错误，说明 align_corners 只能在特定的插值模式下使用
      TORCH_CHECK(
          false,
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    // 如果 align_corners 参数为空
    if (align_corners == c10::nullopt) {
      // 发出警告，说明从版本 0.4.0 开始，默认的上采样行为已更改为 align_corners=False
      TORCH_WARN(
          "Default upsampling behavior when mode=",
          enumtype::get_enum_name(mode),
          " is changed "
          "to align_corners=False since 0.4.0. Please specify "
          "align_corners=True if the old behavior is desired. "
          "See the documentation of nn.Upsample for details.");
      // 将 align_corners 设置为 False
      align_corners = false;
    }
  }

  // 检查输入张量的维度是否在支持的范围内（3D、4D、5D），并且检查是否与插值模式相匹配
  TORCH_CHECK(
      input.dim() >= 3 && input.dim() <= 5,
      "Input Error: Only 3D, 4D and 5D input Tensors supported "
      "(got ",
      input.dim(),
      "D) for the modes: nearest | linear | bilinear | bicubic | trilinear "
      "(got ",
      enumtype::get_enum_name(mode),
      ")");

  // 计算输入张量的 scale_factor 的长度
  auto scale_factor_len = input.dim() - 2;
  // 创建一个存储 scale_factor 的可选型向量，初始值为 c10::nullopt
  std::vector<std::optional<double>> scale_factor_list(
      scale_factor_len, c10::nullopt);
  // 如果 scale_factor 不为空，并且 recompute_scale_factor 不为 true，则重新计算 scale_factor_list
  if (scale_factor != c10::nullopt && !recompute_scale_factor.value_or(false)) {
    auto _scale_factor_repeated = *scale_factor;
    scale_factor_list = {};
    // 遍历 _scale_factor_repeated 中的元素，将其添加到 scale_factor_list 中
    for (const auto& elem : _scale_factor_repeated) {
      scale_factor_list.emplace_back(elem);
    }
  }

  // 如果 antialias 为 true，并且输入张量的维度为 4，并且模式是 bilinear 或 bicubic，则抛出错误
  if (antialias &&
      !(input.dim() == 4 &&
        (std::get_if<enumtype::kBilinear>(&mode) ||
         std::get_if<enumtype::kBicubic>(&mode)))) {
    TORCH_CHECK(
        false,
        "Anti-alias option is only supported for bilinear and bicubic modes");
  }

  // 创建一个元组 closed_over_args，包含 input、size、scale_factor 和 recompute_scale_factor
  auto closed_over_args =
      std::make_tuple(input, size, scale_factor, recompute_scale_factor);

  // 根据输入张量的维度和插值模式选择对应的上采样函数调用
  if (input.dim() == 3 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),
        scale_factor_list.at(0));
  } else if (input.dim() == 4 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else if (input.dim() == 5 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else if (input.dim() == 3 && std::get_if<enumtype::kNearestExact>(&mode)) {
    return torch::_upsample_nearest_exact1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),
        scale_factor_list.at(0));
  } else if (input.dim() == 4 && std::get_if<enumtype::kNearestExact>(&mode)) {
    // 继续上述模式的处理

        input,
        _interp_output_size(2, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else if (input.dim() == 5 && std::get_if<enumtype::kNearestExact>(&mode)) {
    return torch::_upsample_nearest_exact3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else {
    TORCH_CHECK(
        false,
        "Unsupported input dimension and/or interpolation mode: ",
        input.dim(),
        "D for ",
        enumtype::get_enum_name(mode));
  }


注释：
    // 如果输入张量是3维且插值模式为最近邻准确（nearest exact），则调用3维最近邻准确上采样函数
    return torch::_upsample_nearest_exact2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),  // 计算输出大小的插值
        scale_factor_list.at(0),  // 获取缩放因子列表的第一个元素
        scale_factor_list.at(1));  // 获取缩放因子列表的第二个元素
  } else if (input.dim() == 5 && std::get_if<enumtype::kNearestExact>(&mode)) {
    // 如果输入张量是5维且插值模式为最近邻准确（nearest exact），则调用5维最近邻准确上采样函数
    return torch::_upsample_nearest_exact3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),  // 计算输出大小的插值
        scale_factor_list.at(0),  // 获取缩放因子列表的第一个元素
        scale_factor_list.at(1),  // 获取缩放因子列表的第二个元素
        scale_factor_list.at(2));  // 获取缩放因子列表的第三个元素
  } else if (input.dim() == 3 && std::get_if<enumtype::kArea>(&mode)) {
    // 如果输入张量是3维且插值模式为区域平均（area），则调用1维自适应平均池化函数
    return detail::adaptive_avg_pool1d(
        input, _interp_output_size(1, std::move(closed_over_args)));  // 计算输出大小的插值
  } else if (input.dim() == 4 && std::get_if<enumtype::kArea>(&mode)) {
    // 如果输入张量是4维且插值模式为区域平均（area），则调用2维自适应平均池化函数
    return detail::adaptive_avg_pool2d(
        input, _interp_output_size(2, std::move(closed_over_args)));  // 计算输出大小的插值
  } else if (input.dim() == 5 && std::get_if<enumtype::kArea>(&mode)) {
    // 如果输入张量是5维且插值模式为区域平均（area），则调用3维自适应平均池化函数
    return detail::adaptive_avg_pool3d(
        input, _interp_output_size(3, std::move(closed_over_args)));  // 计算输出大小的插值
  } else if (input.dim() == 3 && std::get_if<enumtype::kLinear>(&mode)) {
    // 如果输入张量是3维且插值模式为线性插值（linear），则调用1维线性插值函数
    TORCH_CHECK(
        align_corners != c10::nullopt, "align_corners should be specified.");  // 检查是否指定了align_corners参数
    return torch::upsample_linear1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),  // 计算输出大小的插值
        *align_corners,  // 获取align_corners参数的值
        scale_factor_list.at(0));  // 获取缩放因子列表的第一个元素
  } else if (input.dim() == 3 && std::get_if<enumtype::kBilinear>(&mode)) {
    // 如果输入张量是3维且插值模式为双线性插值（bilinear），则报错，因为需要4维输入
    TORCH_CHECK(false, "Got 3D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 3 && std::get_if<enumtype::kTrilinear>(&mode)) {
    // 如果输入张量是3维且插值模式为三线性插值（trilinear），则报错，因为需要5维输入
    TORCH_CHECK(false, "Got 3D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 4 && std::get_if<enumtype::kLinear>(&mode)) {
    // 如果输入张量是4维且插值模式为线性插值（linear），则报错，因为需要3维输入
    TORCH_CHECK(false, "Got 4D input, but linear mode needs 3D input");
  } else if (input.dim() == 4 && std::get_if<enumtype::kBilinear>(&mode)) {
    // 如果输入张量是4维且插值模式为双线性插值（bilinear），则根据是否开启抗锯齿决定调用哪种函数
    TORCH_CHECK(
        align_corners != c10::nullopt, "align_corners should be specified.");  // 检查是否指定了align_corners参数
    if (antialias) {
      return torch::_upsample_bilinear2d_aa(
          input,
          _interp_output_size(2, std::move(closed_over_args)),  // 计算输出大小的插值
          *align_corners,  // 获取align_corners参数的值
          scale_factor_list.at(0),  // 获取缩放因子列表的第一个元素
          scale_factor_list.at(1));  // 获取缩放因子列表的第二个元素
    }
    return torch::upsample_bilinear2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),  // 计算输出大小的插值
        *align_corners,  // 获取align_corners参数的值
        scale_factor_list.at(0),  // 获取缩放因子列表的第一个元素
        scale_factor_list.at(1));  // 获取缩放因子列表的第二个元素
  } else if (input.dim() == 4 && std::get_if<enumtype::kTrilinear>(&mode)) {
    // 如果输入张量是4维且插值模式为三线性插值（trilinear），则报错，因为需要5维输入
    TORCH_CHECK(false, "Got 4D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kLinear>(&mode)) {
    // 如果输入张量是5维且插值模式为线性插值（linear），则报错，因为需要3维输入
    TORCH_CHECK(false, "Got 5D input, but linear mode needs 3D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kBilinear>(&mode)) {
    // 如果输入张量是5维且插值模式为双线性插值（bilinear），则报错，因为需要4维输入
    TORCH_CHECK(false, "Got 5D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kTrilinear>(&mode)) {
    # 检查是否指定了 align_corners 参数，如果未指定则报错
    TORCH_CHECK(
        align_corners != c10::nullopt, "align_corners should be specified.");
    # 如果输入张量的维度为 3 并且插值模式为 trilinear
    return torch::upsample_trilinear3d(
        input,
        # 根据模式和参数计算插值后的输出大小
        _interp_output_size(3, std::move(closed_over_args)),
        *align_corners,  # 使用指定的 align_corners 参数
        scale_factor_list.at(0),  # 沿各个轴的缩放因子
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else if (input.dim() == 4 && std::get_if<enumtype::kBicubic>(&mode)) {
    # 检查是否指定了 align_corners 参数，如果未指定则报错
    TORCH_CHECK(
        align_corners != c10::nullopt, "align_corners should be specified.");
    # 如果启用了抗锯齿
    if (antialias) {
      # 使用带抗锯齿的双三次插值进行二维上采样
      return torch::_upsample_bicubic2d_aa(
          input,
          # 根据模式和参数计算插值后的输出大小
          _interp_output_size(2, std::move(closed_over_args)),
          *align_corners,  # 使用指定的 align_corners 参数
          scale_factor_list.at(0),  # 沿各个轴的缩放因子
          scale_factor_list.at(1));
    }
    # 使用双三次插值进行二维上采样
    return torch::upsample_bicubic2d(
        input,
        # 根据模式和参数计算插值后的输出大小
        _interp_output_size(2, std::move(closed_over_args)),
        *align_corners,  # 使用指定的 align_corners 参数
        scale_factor_list.at(0),  # 沿各个轴的缩放因子
        scale_factor_list.at(1));
  } else {
    # 如果以上条件都不满足，则抛出错误，说明输入张量维度不支持当前的插值模式
    TORCH_CHECK(
        false,
        "Input Error: Only 3D, 4D and 5D input Tensors supported "
        "(got ",
        input.dim(),
        "D) for the modes: nearest | linear | bilinear | bicubic | trilinear "
        "(got ",
        enumtype::get_enum_name(mode),
        ")");
  }
/// 结束 detail 命名空间
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// 查看
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.interpolate
/// 以了解此函数的确切行为。
///
/// 查看 `torch::nn::functional::InterpolateFuncOptions` 的文档，
/// 了解此函数支持的可选参数。
///
/// 示例：
/// ```
/// namespace F = torch::nn::functional;
/// F::interpolate(input,
/// F::InterpolateFuncOptions().size({4}).mode(torch::kNearest));
/// ```
inline Tensor interpolate(
    const Tensor& input,
    const InterpolateFuncOptions& options = {}) {
  return detail::interpolate(
      input,
      options.size(),                // 使用 options 提供的尺寸参数
      options.scale_factor(),        // 使用 options 提供的缩放因子参数
      options.mode(),                // 使用 options 提供的插值模式参数
      options.align_corners(),       // 使用 options 提供的对齐角参数
      options.recompute_scale_factor(),  // 使用 options 提供的重新计算缩放因子参数
      options.antialias());         // 使用 options 提供的抗锯齿参数
}

/// 结束 functional 命名空间
} // namespace functional
/// 结束 nn 命名空间
} // namespace nn
/// 结束 torch 命名空间
} // namespace torch
```