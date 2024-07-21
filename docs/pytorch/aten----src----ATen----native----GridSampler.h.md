# `.\pytorch\aten\src\ATen\native\GridSampler.h`

```
// 预处理指令，指示编译器只包含此头文件一次
#pragma once

// 包含算法、数学运算、整数类型等标准库头文件
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

// 包含GridSamplerUtils.h头文件中的定义
#include <ATen/native/GridSamplerUtils.h>

// 命名空间at::native的开始
namespace at::native {

// 使用GridSamplerInterpolation和GridSamplerPadding的别名
using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

// 将从-1到+1范围的坐标反标准化为像素索引值的模板函数
// 当align_corners为true时，将-1和+1映射到角落像素的中心点
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// 当align_corners为false时，将-1和+1映射到图像边缘
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize(scalar_t coord, int64_t size,
                                                bool align_corners) {
  if (align_corners) {
    // 将坐标从[-1, 1]反标准化为[0, size - 1]
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // 将坐标从[-1, 1]反标准化为[-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
  }
}

// grid_sampler_unnormalize_set_grad与grid_sampler_unnormalize功能相同，
// 但还通过指针参数grad_in返回`d output / d input`。
// 这在grid_sampler的反向传播中很有用。
template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int64_t size,
                                                         bool align_corners, scalar_t *grad_in) {
  if (align_corners) {
    // 将坐标从[-1, 1]反标准化为[0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // 将坐标从[-1, 1]反标准化为[-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1) * size - 1) / 2;
  }
}

// 将坐标限制在0和clip_limit - 1之间的模板函数
template<typename scalar_t>
static inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
}

// clip_coordinates_set_grad与clip_coordinates类似，
// 但通过指针参数grad_in返回`d output / d input`。
// 这在grid_sampler的反向传播中很有用。
template<typename scalar_t>
static inline scalar_t clip_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                 scalar_t *grad_in) {
  // 注意，对于梯度计算来说，将边界视为超出范围是很重要的。
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}


这段代码定义了一些用于图像处理中的坐标处理和梯度计算的函数模板。
    }
  }
// 反射坐标，使其落在指定的范围内（包括边界）。
// 边界通过其两倍值传递，以便可以将半整数值表示为整数。
template<typename scalar_t>
static inline scalar_t reflect_coordinates(scalar_t in, int64_t twice_low,
                                           int64_t twice_high) {
  // 如果范围相同，返回零
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  // 计算最小值和范围
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  // 计算相对位置，并取绝对值
  in = std::fabs(in - min);
  // `fmod` 返回与 `in` 同号的余数，此处 `fabs` 之后 `in` 为正
  scalar_t extra = std::fmod(in, span);
  // 计算翻转次数
  int flips = static_cast<int>(std::floor(in / span));
  // 根据翻转次数决定返回值
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// reflect_coordinates_set_grad 类似于 reflect_coordinates，
// 但还通过指针参数 `grad_in` 返回 `d output / d input`。
// 这在 grid_sampler 的反向传播中很有用。
template<typename scalar_t>
static inline scalar_t reflect_coordinates_set_grad(scalar_t in, int64_t twice_low,
                                                    int64_t twice_high, scalar_t *grad_in) {
  // 如果范围相同，设置 grad_in 为零并返回零
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  int grad_in_mult_;
  // 计算最小值和范围
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  // 计算相对位置并处理负数情况
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  // `fmod` 返回与 `in` 同号的余数，此处 `if` 后 `in` 为正
  scalar_t extra = std::fmod(in, span);
  // 计算翻转次数
  int flips = static_cast<int>(std::floor(in / span));
  // 根据翻转次数决定返回值并设置 grad_in
  if (flips % 2 == 0) {
    *grad_in = static_cast<scalar_t>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
  }
}

// 将超出边界的点映射回边界
// 仅影响 padding_mode=border 或 reflection 的情况
template<typename scalar_t>
static inline scalar_t compute_coordinates(scalar_t coord, int64_t size,
                                           GridSamplerPadding padding_mode,
                                           bool align_corners) {
  // 根据 padding_mode 进行不同的处理
  if (padding_mode == GridSamplerPadding::Border) {
    // 将坐标剪裁到图像边界
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // 根据 align_corners 来反射坐标
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
    }
    // 将坐标剪裁到图像边界
    coord = clip_coordinates(coord, size);
  }
  return coord;
}
// 计算规范化后的坐标，使其适应于输入大小和对齐设置
template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  // 反规范化坐标，以适应输入大小和对齐设置
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  // 计算坐标在给定填充模式下的最终位置
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;  // 返回计算得到的坐标
}

// grid_sampler_compute_source_index_set_grad 与 grid_sampler_compute_source_index 类似，
// 但它还通过指针参数 `grad_in` 返回 `d output / d input`。
// 在 grid_sampler 的反向传播过程中很有用。
template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t *grad_in) {
  scalar_t grad_clip, grad_refl;
  // 使用梯度信息反规范化坐标
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    // 将坐标限制在图像边界内
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;  // 更新输入梯度
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // 根据图像边界反射坐标
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
    }
    // 将坐标限制在图像边界内
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;  // 更新输入梯度
  }
  return coord;  // 返回计算得到的坐标
}

// 检查二维坐标是否在指定的边界内
static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

// 检查三维坐标是否在指定的边界内
static inline bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

// 获取在边界内的像素值，如果超出边界则返回0
template<typename scalar_t>
static inline scalar_t get_value_bounded(
    const scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    GridSamplerPadding padding_mode,
    bool align_corners) {

  x = compute_coordinates(x, W, padding_mode, align_corners);  // 计算 x 坐标
  y = compute_coordinates(y, H, padding_mode, align_corners);  // 计算 y 坐标

  int64_t ix = static_cast<int64_t>(x);  // 转换为整数索引
  int64_t iy = static_cast<int64_t>(y);  // 转换为整数索引

  if (within_bounds_2d(iy, ix, H, W)) {  // 检查坐标是否在边界内
    return data[iy * sH + ix * sW];  // 返回边界内的像素值
  }
  return static_cast<scalar_t>(0);  // 超出边界返回0
}

// 安全地对二维数据进行加法操作，仅在边界内有效
template<typename scalar_t>
static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                               int64_t sH, int64_t sW, int64_t H, int64_t W,
                               scalar_t delta) {
  if (within_bounds_2d(h, w, H, W)) {  // 检查坐标是否在边界内
    data[h * sH + w * sW] += delta;  // 执行安全的加法操作
  }
}

template<typename scalar_t>
// 在三维数据数组中安全地添加增量，只有当给定的坐标 (d, h, w) 在指定的边界 (D, H, W) 内时才执行
static inline void safe_add_3d(scalar_t *data, int64_t d, int64_t h, int64_t w,
                               int64_t sD, int64_t sH, int64_t sW,
                               int64_t D, int64_t H, int64_t W,
                               scalar_t delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    // 根据步长和坐标 (d, h, w) 计算线性索引，并在数据数组中增加增量 delta
    data[d * sD + h * sH + w * sW] += delta;
  }
}

// 向二维数据数组中的特定位置安全地添加增量
template<typename scalar_t>
static inline void add_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    scalar_t delta,
    GridSamplerPadding padding_mode,
    bool align_corners) {

  // 根据 padding_mode 和 align_corners 计算坐标 x 和 y 的有效位置
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  // 将浮点坐标 x 和 y 转换为整数索引 ix 和 iy
  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  // 在数据数组中安全地添加增量 delta 到 (iy, ix) 处
  safe_add_2d(data, iy, ix, sH, sW, H, W, delta);
}

// 计算三次插值的微分，即 `d coeff / d x`
template<typename scalar_t>
static inline void get_cubic_coefficients_grad(
    scalar_t coeffs[4],
    scalar_t t) {

  // 必须与 forward 计算中的 aten/src/ATen/native/UpSample.h:get_cubic_upsample_coefficients 相同
  scalar_t A = -0.75;

  scalar_t x;
  // 计算四个插值系数的微分值
  x = -1 - t; // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A ) * x - 8 * A;
  x = -t;     // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;  // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;  // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

}  // namespace at::native


这段代码主要包括了几个静态内联函数，用于在多维数据数组中进行安全的增量操作，以及计算三次插值的微分系数。注释解释了每个函数的目的和关键步骤，确保了代码的易读性和可理解性。
```