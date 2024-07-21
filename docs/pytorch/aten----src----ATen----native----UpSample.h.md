# `.\pytorch\aten\src\ATen\native\UpSample.h`

```py
#pragma once

#include <math.h>  // 包含数学函数库

#include <ATen/OpMathType.h>  // ATen 数学操作类型
#include <ATen/TensorUtils.h>  // ATen 张量工具
#include <ATen/OpMathType.h>  // ATen 数学操作类型（重复引用）
#include <ATen/core/Tensor.h>  // ATen 核心张量类
#include <ATen/cpu/vec/functional.h>  // ATen CPU 向量功能
#include <ATen/cpu/vec/vec.h>  // ATen CPU 向量化支持
#include <ATen/native/DispatchStub.h>  // ATen 原生调度存根
#include <ATen/native/cpu/utils.h>  // ATen 原生 CPU 工具

/**
 * Note [compute_scales_value]
 * Note [area_pixel_compute_scale]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 使用 scale_factor 进行插值时，根据 recompute_scale_factor 的值可能有不同行为：
 *
 * - 当 recompute_scale_factor = True 时（当前默认行为）：
 * 用户提供的 scale_factor 用于计算输出尺寸。然后使用输入尺寸和计算出的输出尺寸
 * 推断出在插值中使用的新的 scales 值。由于浮点数运算不精确，这可能与用户提供的
 * scales 值不同。
 *
 * - 当 recompute_scale_factor = False 时（从 1.5.0 开始将成为默认行为）：
 * 行为遵循 opencv 的逻辑，使用用户提供的 scales 值进行插值计算。
 *
 * 如果未提供 scales 或者提供了 scales 但 recompute_scale_factor 设置为 True（默认行为），
 * 则从输入和输出尺寸计算 scales。
 *
 * 当从输入和输出尺寸推断出 scales 时，我们将每个像素视为一个区域，idx + 0.5 视为其中心索引。
 * 以下是在 1D 情况下的示例公式。
 * 若 align_corners = True：保留两个角像素区域的中心，
 *     (0.5, 0.5) -> (0.5, 0.5),
 *     (input_size - 0.5, 0.5) -> (output_size - 0.5)
 *     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
 *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
 * 若 align_corners = False：整个范围按比例缩放
 *     scale = input_size / output_size
 *     src_idx + 0.5 = scale * (dst_index + 0.5)
 */

namespace at::native {

namespace upsample {

// 计算输出尺寸的函数声明
TORCH_API c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,  // 输入张量的完整尺寸
    at::OptionalIntArrayRef output_size,  // 输出尺寸（可选）
    std::optional<c10::ArrayRef<double>> scale_factors);  // 尺度因子（可选）

// 获取指定索引处的尺度值（可选）
inline std::optional<double> get_scale_value(std::optional<c10::ArrayRef<double>> scales, int idx) {
  if (!scales) {
    return c10::nullopt;  // 如果未提供尺度值，则返回空值
  }
  return scales->at(idx);  // 返回指定索引处的尺度值
}

} // namespace upsample

// 使用 scale_t 定义尺度类型
using scale_t = std::optional<double>;

// 定义不同维度下的最近邻插值函数指针类型
using upsampling_nearest1d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_w);
using _upsampling_nearest_exact1d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_w);
using upsampling_nearest2d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_h, scale_t scales_w);
using _upsampling_nearest_exact2d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_h, scale_t scales_w);

} // namespace at::native
// 定义指向不同三维最近邻上采样函数的指针类型
using upsampling_nearest3d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_d, scale_t scales_h, scale_t scales_w);
// 定义指向精确三维最近邻上采样函数的指针类型
using _upsampling_nearest_exact3d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_d, scale_t scales_h, scale_t scales_w);
// 定义指向一维线性插值上采样函数的指针类型
using upsampling_linear1d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_w);
// 定义指向二维双线性插值上采样函数的指针类型
using upsampling_bilinear2d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
// 定义指向带有反锯齿效果的二维双线性插值上采样函数的指针类型
using _upsampling_bilinear2d_aa = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
// 定义指向三维三线性插值上采样函数的指针类型
using upsampling_trilinear3d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_d, scale_t scales_h, scale_t scales_w);
// 定义指向二维双三次插值上采样函数的指针类型
using upsampling_bicubic2d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
// 定义指向带有反锯齿效果的二维双三次插值上采样函数的指针类型
using _upsampling_bicubic2d_aa = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);

// 声明一维最近邻上采样函数的调度器
DECLARE_DISPATCH(upsampling_nearest1d, upsample_nearest1d_kernel);
// 声明精确一维最近邻上采样函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact1d, _upsample_nearest_exact1d_kernel);
// 声明二维最近邻上采样函数的调度器
DECLARE_DISPATCH(upsampling_nearest2d, upsample_nearest2d_kernel);
// 声明精确二维最近邻上采样函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact2d, _upsample_nearest_exact2d_kernel);
// 声明三维最近邻上采样函数的调度器
DECLARE_DISPATCH(upsampling_nearest3d, upsample_nearest3d_kernel);
// 声明精确三维最近邻上采样函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact3d, _upsample_nearest_exact3d_kernel);
// 声明一维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(upsampling_nearest1d, upsample_nearest1d_backward_kernel);
// 声明精确一维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact1d, _upsample_nearest_exact1d_backward_kernel);
// 声明二维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(upsampling_nearest2d, upsample_nearest2d_backward_kernel);
// 声明精确二维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact2d, _upsample_nearest_exact2d_backward_kernel);
// 声明三维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(upsampling_nearest3d, upsample_nearest3d_backward_kernel);
// 声明精确三维最近邻上采样反向函数的调度器
DECLARE_DISPATCH(_upsampling_nearest_exact3d, _upsample_nearest_exact3d_backward_kernel);
// 声明一维线性插值上采样函数的调度器
DECLARE_DISPATCH(upsampling_linear1d, upsample_linear1d_kernel);
// 声明二维双线性插值上采样函数的调度器
DECLARE_DISPATCH(upsampling_bilinear2d, upsample_bilinear2d_kernel);
// 声明带有反锯齿效果的二维双线性插值上采样函数的调度器
DECLARE_DISPATCH(_upsampling_bilinear2d_aa, _upsample_bilinear2d_aa_kernel);
// 声明三维三线性插值上采样函数的调度器
DECLARE_DISPATCH(upsampling_trilinear3d, upsample_trilinear3d_kernel);
// 声明一维线性插值反向上采样函数的调度器
DECLARE_DISPATCH(upsampling_linear1d, upsample_linear1d_backward_kernel);
// 声明二维双线性插值反向上采样函数的调度器
DECLARE_DISPATCH(upsampling_bilinear2d, upsample_bilinear2d_backward_kernel);
// 声明带有反锯齿效果的二维双线性插值反向上采样函数的调度器
DECLARE_DISPATCH(_upsampling_bilinear2d_aa, _upsample_bilinear2d_aa_backward_kernel);
// 声明三维三线性插值反向上采样函数的调度器
DECLARE_DISPATCH(upsampling_trilinear3d, upsample_trilinear3d_backward_kernel);
// 声明二维双三次插值上采样函数的调度器
DECLARE_DISPATCH(upsampling_bicubic2d, upsample_bicubic2d_kernel);
// 声明带有反锯齿效果的二维双三次插值上采样函数的调度器
DECLARE_DISPATCH(_upsampling_bicubic2d_aa, _upsample_bicubic2d_aa_kernel);
// 声明带有反锯齿效果的二维双三次插值反向上采样函数的调度器
DECLARE_DISPATCH(_upsampling_bicubic2d_aa, _upsample_bicubic2d_aa_backward_kernel);
// 检查输入输出尺寸的有效性，返回一个长度为3的整数数组
inline C10_UNUSED std::array<int64_t, 3> upsample_1d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  // 检查输出尺寸是否为1
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  // 检查输入尺寸是否为3
  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  // 获取输出宽度
  int64_t output_width = output_size[0];

  // 获取输入尺寸中的批次数、通道数和输入宽度
  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  // 检查输入和输出尺寸是否大于0
  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  // 返回由批次数、通道数和输出宽度构成的数组
  return {nbatch, channels, output_width};
}

// 检查输入输出尺寸的有效性，返回一个长度为4的整数数组
inline C10_UNUSED std::array<int64_t, 4> upsample_2d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  // 检查输出尺寸是否为2
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  // 检查输入尺寸是否为4
  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  // 获取输出高度和宽度
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // 获取输入尺寸中的批次数、通道数、输入高度和宽度
  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  // 检查输入和输出尺寸是否大于0
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  // 返回由批次数、通道数、输出高度和宽度构成的数组
  return {nbatch, channels, output_height, output_width};
}

// 表示函数未使用的宏定义
inline C10_UNUSED
// 检查输入和输出大小，确保输出大小为3
std::array<int64_t, 5> upsample_3d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  // 检查输出大小是否为3
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  // 检查输入大小是否为5
  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  // 提取输出尺寸的深度、高度、宽度
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // 提取输入尺寸的批次数、通道数、深度、高度、宽度
  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

  // 检查输入和输出尺寸都大于0
  TORCH_CHECK(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
          output_depth > 0 && output_height > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (D: ",
      input_depth,
      ", H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (D: ",
      output_depth,
      ", H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  // 返回包含批次数、通道数、输出深度、输出高度、输出宽度的数组
  return {nbatch, channels, output_depth, output_height, output_width};
}

// 检查2D上采样的输入和输出尺寸
inline void upsample_2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t nbatch,
    int64_t nchannels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  // 检查输入和输出尺寸都大于0
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  // 如果输入张量已定义，检查其为非空4D张量
  if (input.defined()) {
    // 允许批次大小为空但不允许其他维度为空
    TORCH_CHECK(
                (input.numel() != 0 ||
                 (input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0)
                 ) &&
                input.dim() == 4,
                "Non-empty 4D data tensor expected but got a tensor with sizes ",
                input.sizes());
  } else if (grad_output.defined()) {
    // 如果输入张量未定义但梯度输出张量已定义，检查其维度和尺寸
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, nchannels);
    check_dim_size(grad_output, 4, 2, output_height);
    check_dim_size(grad_output, 4, 3, output_width);
  }
}

// 计算缩放比例的函数模板
template <typename scalar_t>
inline scalar_t compute_scales_value(
    const std::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
      // 见注释[compute_scales_value]，根据给定的缩放因子或输入输出大小计算比例值
      // FIXME: 在确保没有使用-1默认值的模型序列化后，移除魔数大于0的检查
      return (scale.has_value() && scale.value() > 0.)
          ? static_cast<scalar_t>(1.0 / scale.value())
          : (static_cast<scalar_t>(input_size) / output_size);
}
    // 根据注释 [area_pixel_compute_scale]，此处根据 align_corners 参数的不同情况计算缩放比例
    if(align_corners) {
        // 如果 align_corners 为 true
        if(output_size > 1) {
            // 如果输出尺寸大于1，则计算并返回靠角对齐的缩放比例
            return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
        } else {
            // 如果输出尺寸不大于1，则返回0（避免除以零的情况）
            return static_cast<scalar_t>(0);
        }
    } else {
        // 如果 align_corners 为 false，则调用 compute_scales_value 函数计算非靠角对齐的缩放比例
        return compute_scales_value<scalar_t>(scale, input_size, output_size);
    }
// 在模板函数中，计算像素区域插值时，根据缩放比例、目标索引、对齐角落、是否使用立方插值来计算源索引
template <typename scalar_t>
inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,         // 缩放比例
    int64_t dst_index,      // 目标索引
    bool align_corners,     // 是否对齐角落
    bool cubic              // 是否使用立方插值
) {
  if (align_corners) {
    return scale * dst_index;   // 如果对齐角落，则简单计算源索引
  } else {
    scalar_t src_idx = scale * (dst_index + static_cast<scalar_t>(0.5)) -
        static_cast<scalar_t>(0.5);
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO: Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    // 如果不对齐角落，则根据 Opencv 的调整逻辑计算源索引
    return (!cubic && src_idx < static_cast<scalar_t>(0)) ? scalar_t(0)
                                                          : src_idx;
    // 如果不使用立方插值且源索引小于零，则返回零；否则返回计算出的源索引
  }
}

// 计算最近邻插值时的源索引，以匹配 OpenCV 的 INTER_NEAREST 方法
inline int64_t nearest_neighbor_compute_source_index(
    const float scale,      // 缩放比例
    int64_t dst_index,      // 目标索引
    int64_t input_size      // 输入尺寸
) {
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  // 计算最近邻插值时的源索引，确保不超过输入尺寸的边界
  return src_index;
}

// 计算精确的最近邻插值时的源索引，与 Pillow 和 Scikit-Image/Scipy ndi.zoom 相匹配
inline int64_t nearest_neighbor_exact_compute_source_index(
    const float scale,      // 缩放比例
    int64_t dst_index,      // 目标索引
    int64_t input_size      // 输入尺寸
) {
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf((dst_index + 0.5) * scale)), input_size - 1);
  // 计算精确的最近邻插值时的源索引，确保不超过输入尺寸的边界
  return src_index;
}

// 计算最近邻索引，考虑到特定情况下的尺寸匹配，如果匹配则直接返回目标索引
inline int64_t nearest_idx(
    int64_t output_index,       // 输出索引
    int64_t input_size,         // 输入尺寸
    int64_t output_size,        // 输出尺寸
    std::optional<double> scales // 可选的缩放因子
) {
  if (output_size == input_size) {
    // 如果输出尺寸等于输入尺寸，则直接返回输出索引
    return output_index;
  } else if (output_size == 2 * input_size) {
    // 如果输出尺寸是输入尺寸的两倍，则右移输出索引一位
    return output_index >> 1;
  } else {
    // 否则，根据缩放因子计算最近邻插值时的源索引
    float scale = compute_scales_value<float>(scales, input_size, output_size);
    return nearest_neighbor_compute_source_index(scale, output_index, input_size);
  }
}

// 计算精确最近邻索引，作为 nearest_idx 方法的替代品，处理更精确的插值计算需求
inline int64_t nearest_exact_idx(
    int64_t output_index,       // 输出索引
    int64_t input_size,         // 输入尺寸
    int64_t output_size,        // 输出尺寸
    std::optional<double> scales // 可选的缩放因子
) {
  float scale = compute_scales_value<float>(scales, input_size, output_size);
  // 计算精确最近邻索引时的源索引，确保不超过输入尺寸的边界
    # 调用名为 nearest_neighbor_exact_compute_source_index 的函数，返回其计算结果
    return nearest_neighbor_exact_compute_source_index(scale, output_index, input_size);
}

// 定义一个 typedef，用于调度 nearest_idx 或 nearest_exact_idx 函数
typedef int64_t (*nearest_idx_fn_t)(int64_t, int64_t, int64_t, std::optional<double>);

// 根据给定的 scalar_t 类型，返回受限制的数据值
template <typename scalar_t>
scalar_t upsample_get_value_bounded(
    scalar_t* data,          // 数据数组指针
    int64_t width,           // 数据数组的宽度
    int64_t height,          // 数据数组的高度
    int64_t x,               // 访问的 x 坐标
    int64_t y) {             // 访问的 y 坐标
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));   // 获取受限制的 x 坐标
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));  // 获取受限制的 y 坐标
  return data[access_y * width + access_x];  // 返回受限制位置的数据值
}

// 根据给定的 scalar_t 类型，增加受限制的数据值
template <typename scalar_t>
void upsample_increment_value_bounded(
    scalar_t* data,          // 数据数组指针
    int64_t width,           // 数据数组的宽度
    int64_t height,          // 数据数组的高度
    int64_t x,               // 访问的 x 坐标
    int64_t y,               // 访问的 y 坐标
    scalar_t value) {        // 增加的值
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));   // 获取受限制的 x 坐标
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));  // 获取受限制的 y 坐标
  data[access_y * width + access_x] += value;  // 增加受限制位置的数据值
}

// 基于 https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
// 根据给定的 scalar_t 类型和参数 A，计算 cubic convolution 算法的第一个插值
template <typename scalar_t>
scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;  // 返回 cubic convolution 算法的计算结果
}

// 根据给定的 scalar_t 类型和参数 A，计算 cubic convolution 算法的第二个插值
template <typename scalar_t>
scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;  // 返回 cubic convolution 算法的计算结果
}

// 根据给定的 scalar_t 类型，获取 cubic interpolation 算法的上采样系数
template <typename scalar_t>
void get_cubic_upsample_coefficients(
    scalar_t coeffs[4],   // 存储上采样系数的数组
    scalar_t t) {         // 插值参数 t
  scalar_t A = -0.75;     // 设定常数 A

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);  // 计算第一个系数
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);        // 计算第二个系数

  // 相反的系数
  scalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);        // 计算第三个系数
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);  // 计算第四个系数
}

// 根据给定的 scalar_t 类型，进行一维 cubic interpolation
template <typename scalar_t>
inline scalar_t cubic_interp1d(
    scalar_t x0,    // 输入数据点
    scalar_t x1,    // 输入数据点
    scalar_t x2,    // 输入数据点
    scalar_t x3,    // 输入数据点
    scalar_t t) {   // 插值参数 t
  scalar_t coeffs[4];                         // 存储上采样系数的数组
  get_cubic_upsample_coefficients<scalar_t>(coeffs, t);  // 获取 cubic interpolation 的系数

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];  // 返回插值结果
}

// 当 real_input_index 超过浮点类型可以精确表示的范围时，类型转换为 int64_t 可能导致超出 input_size，造成溢出。因此我们用 std::min 进行保护。
template<typename scalar_t, typename opmath_t>
inline void guard_index_and_lambda(const opmath_t& real_input_index, const int64_t& input_size, int64_t& input_index, scalar_t& lambda) {
  input_index = std::min(static_cast<int64_t>(floorf(real_input_index)), input_size - 1);  // 保护输入索引，避免溢出
  lambda = std::min(
      std::max(real_input_index - input_index, static_cast<opmath_t>(0)),  // 计算 lambda 值，保护其在 [0, 1] 范围内
      static_cast<opmath_t>(1)
    );
}

// 根据指定的参数和条件，计算源索引和 lambda 值
template<typename scalar_t, typename opmath_t>
inline void compute_source_index_and_lambda(
    int64_t& input_index0,     // 输出：第一个输入索引
    int64_t& input_index1,     // 输出：第二个输入索引
    scalar_t& lambda0,         // 输出：第一个 lambda 值
    scalar_t& lambda1,         // 输出：第二个 lambda 值
    opmath_t ratio,            // 比例因子
    int64_t output_index,      // 输出索引
    int64_t input_size,        // 输入尺寸
    int64_t output_size,       // 输出尺寸
    bool align_corners) {      // 是否对齐角落的标志
  if (output_size == input_size) {  // 如果输出尺寸与输入尺寸相同
    // 如果 scale_factor = 1，表示简单复制
    // 将输出索引赋值给输入索引0和输入索引1
    input_index0 = output_index;
    input_index1 = output_index;
    // 设置 lambda0 和 lambda1 的值
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    // 如果 scale_factor 不等于 1，则需要计算真实的输入索引
    const auto real_input_index =
        area_pixel_compute_source_index<opmath_t>(
            ratio, output_index, align_corners, /*cubic=*/false);
    // 根据计算得到的真实输入索引，保护索引并确定 lambda1
    guard_index_and_lambda(real_input_index, input_size, input_index0, lambda1);
    // 计算输入索引1相对于输入索引0的偏移量
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    // 根据偏移量计算输入索引1的值
    input_index1 = input_index0 + offset;
    // 计算 lambda0 的值
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}  // 结束命名空间 at::native

// 当数据类型不是 BFloat16 或 Half 时，此函数不会被使用
template <typename scalar_in, typename scalar_out,
          typename std::enable_if_t<!is_reduced_floating_point_v<scalar_out> || !std::is_same<scalar_in, float>::value, int> = 0>
void inline apply_grad_input(scalar_in* buffer_ptr, scalar_out* gin, int64_t size) {
  // 检查 scalar_out 是否为降低精度的浮点类型
  TORCH_CHECK((is_reduced_floating_point_v<scalar_out>),
              "Upsample backward only support BFloat16 and Half in the lower precision data types on CPU.")
  // 检查 scalar_in 是否为 float 类型
  TORCH_CHECK((std::is_same<scalar_in, float>::value),
              "Upsample backward should use float as acc buffer for BFloat16 and Half grad input on CPU.")
  return;
}

// 当数据类型为 BFloat16 和 Half 时，应用梯度输入
template <typename scalar_in, typename scalar_out,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_out> && std::is_same<scalar_in, float>::value, int> = 0>
void inline apply_grad_input(scalar_in* buffer_ptr, scalar_out* gin, int64_t size) {
  // 使用 Vectorized 类型别名
  using bVec = Vectorized<scalar_out>;
  using fVec = Vectorized<float>;
  int64_t d = 0;
  // 对于每个 bVec 大小的数据块进行循环
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    // 加载 gin 中的数据到 bVec
    bVec gin_bvec = bVec::loadu(gin + d);
    // 定义两个 fVec 类型变量
    fVec gin_fvec0, gin_fvec1;
    // 将 gin_bvec 转换为 float 类型并分配给 gin_fvec0 和 gin_fvec1
    std::tie(gin_fvec0, gin_fvec1) = convert_to_float<scalar_out>(gin_bvec);
    // 将 buffer_ptr 中的数据加载到 gin_fvec0 和 gin_fvec1 中
    gin_fvec0 += fVec::loadu(buffer_ptr + d);
    gin_fvec1 += fVec::loadu(buffer_ptr + d + fVec::size());
    // 存储零值到 buffer_ptr 中的 d 和 d + fVec::size() 处
    fVec(0).store(buffer_ptr + d);
    fVec(0).store(buffer_ptr + d + fVec::size());
    // 将 gin_fvec0 和 gin_fvec1 转换为 scalar_out 并存储到 gin 中的 d 处
    convert_from_float<scalar_out>(gin_fvec0, gin_fvec1).store(gin + d);
  }
  // 处理剩余的数据
  for (; d < size; d++) {
    // 将 gin 中的数据加上 buffer_ptr 中对应位置的数据
    gin[d] += buffer_ptr[d];
    // 将 buffer_ptr 中的数据设为零
    buffer_ptr[d] = 0;
  }
}
```