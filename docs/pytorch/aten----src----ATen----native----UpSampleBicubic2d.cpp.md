# `.\pytorch\aten\src\ATen\native\UpSampleBicubic2d.cpp`

```py
// 定义预处理宏，仅在编译时包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的核心头文件
#include <ATen/core/Tensor.h>
// 包含分派机制的头文件
#include <ATen/Dispatch.h>
// 包含张量元数据的头文件
#include <ATen/TensorMeta.h>
// 包含上采样相关功能的头文件
#include <ATen/native/UpSample.h>
// 包含范围迭代器的头文件
#include <c10/util/irange.h>
// 包含并行计算的头文件
#include <ATen/Parallel.h>

// 如果未定义每个操作符的头文件，则包含操作函数的通用头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，分别包含特定的上采样双三次插值头文件
#else
#include <ATen/ops/_upsample_bicubic2d_aa.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/upsample_bicubic2d.h>
#include <ATen/ops/upsample_bicubic2d_backward.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#include <ATen/ops/upsample_bicubic2d_native.h>
#endif

// 在 AT 命名空间下定义元数据函数 upsample_bicubic2d
namespace at::meta {

// 定义 upsample_bicubic2d 元数据函数，用于前向操作
TORCH_META_FUNC(upsample_bicubic2d) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 根据输入和输出尺寸计算完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入张量是否是非空的 4 维数据张量
  // 如果空，则报错，并附带当前张量的尺寸信息
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长分布，根据输入张量的内存格式推荐设置
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

// 定义 upsample_bicubic2d 元数据函数，用于反向操作
TORCH_META_FUNC(upsample_bicubic2d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 根据输入和输出尺寸计算完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  // 检查梯度输出张量是否为 4 维
  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  // 检查每个维度的梯度输出张量大小是否与预期的输出尺寸相同
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输出张量的原始步长分布，根据梯度输出张量的选项设置
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

// 定义 _upsample_bicubic2d_aa 元数据函数，用于前向操作（带抗锯齿的双三次插值）
TORCH_META_FUNC(_upsample_bicubic2d_aa) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  // 根据输入和输出尺寸计算完整的输出尺寸
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // 检查输入张量是否是非空的 4 维数据张量
  // 如果空，则报错，并附带当前张量的尺寸信息
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  // 设置输出张量的原始步长分布，根据输入张量的内存格式推荐设置
  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}


这些注释解释了每个代码块中的每一行的作用和功能，保留了原始代码的结构和缩进。
// 定义 TORCH_META_FUNC 宏来声明 _upsample_bicubic2d_aa_backward 函数
TORCH_META_FUNC(_upsample_bicubic2d_aa_backward) (
  // grad_output: 梯度输出张量
  const Tensor& grad_output,
  // output_size: 输出尺寸数组
  IntArrayRef output_size,
  // input_size: 输入尺寸数组
  IntArrayRef input_size,
  // align_corners: 对齐角点标志
  bool align_corners,
  // scales_h, scales_w: 可选的高度和宽度缩放因子
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  // 检查输入的输出张量维度是否为4
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  // 检查梯度输出张量的每个维度是否与输出尺寸相匹配
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  // 设置输出张量的原始步幅，用于分配空间
  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace at::meta

// 进入 at::native 命名空间
namespace at::native {
// 进入匿名命名空间
namespace {

// 定义模板函数 upsample_bicubic2d_backward_out_frame
template <typename scalar_t>
static void upsample_bicubic2d_backward_out_frame(
    // odata: 输出数据指针
    const scalar_t* odata,
    // idata: 输入数据指针
    scalar_t* idata,
    // input_height, input_width: 输入高度和宽度
    int64_t input_height,
    int64_t input_width,
    // output_height, output_width: 输出高度和宽度
    int64_t output_height,
    int64_t output_width,
    // nbatch: 批次大小
    int64_t nbatch,
    // channels: 通道数
    int64_t channels,
    // align_corners: 对齐角点标志
    bool align_corners,
    // scales_h, scales_w: 可选的高度和宽度缩放因子
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  
  // 计算扩展后的通道数
  channels = channels * nbatch;
  // 计算输入和输出每个片段的大小
  auto input_slice_size = input_height * input_width;
  auto output_slice_size = output_height * output_width;

  // 定义 opmath_t 类型为 scalar_t 的数学操作类型
  using opmath_t = at::opmath_type<scalar_t>;
  // 计算高度和宽度的缩放比例
  const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
      input_height, output_height, align_corners, scales_h);
  const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
      input_width, output_width, align_corners, scales_w);
  
  // 并行处理每个通道的数据
  at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, [&](int64_t start, int64_t end) {
    // 初始化累加数据指针和缓冲区数据
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    // 如果 scalar_t 类型与 opmath_t 类型不同
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
      // 使用缓冲区数据进行初始化
      buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
      acc_data_ptr = buffer_data.get();
      // 初始化累加数据为0
      memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    }
    // 对于输入范围 [start, end) 的每个索引 i 进行循环处理
    for (const auto i : c10::irange(start, end)) {
      // 计算当前输入片的起始位置
      scalar_t* in = idata + i * input_slice_size;
      // 计算当前输出片的起始位置
      const scalar_t* out = odata + i * output_slice_size;
      // 对输出高度进行循环
      for (const auto output_y : c10::irange(output_height)) {
        // 对输出宽度进行循环
        for (const auto output_x : c10::irange(output_width)) {

          // 计算实际的输入 x 索引位置，并进行边界保护和插值计算
          const opmath_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
          int64_t input_x;
          opmath_t t_x;
          guard_index_and_lambda(real_x, input_width, input_x, t_x);

          // 计算实际的输入 y 索引位置，并进行边界保护和插值计算
          const opmath_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
          int64_t input_y;
          opmath_t t_y;
          guard_index_and_lambda(real_y, input_height, input_y, t_y);

          // 定义用于存储插值系数的数组，并获取插值系数
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
          opmath_t x_coeffs[4];
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
          opmath_t y_coeffs[4];

          get_cubic_upsample_coefficients<opmath_t>(x_coeffs, t_x);
          get_cubic_upsample_coefficients<opmath_t>(y_coeffs, t_y);

          // 获取当前输出位置处的值，并对周围四个像素进行插值
          opmath_t out_value = out[output_y * output_width + output_x];
          for (const auto ii : c10::irange(4)) {
            for (const auto jj : c10::irange(4)) {
              upsample_increment_value_bounded<opmath_t>(
                  acc_data_ptr == nullptr ? reinterpret_cast<opmath_t*>(in) : acc_data_ptr,
                  input_width,
                  input_height,
                  input_x - 1 + ii,
                  input_y - 1 + jj,
                  out_value * y_coeffs[jj] * x_coeffs[ii]);
            }
          }
        }
      }
      // 如果累加数据指针不为空，则应用梯度输入
      if (acc_data_ptr != nullptr) {
        apply_grad_input(acc_data_ptr, in, input_slice_size);
      }
    }
// 定义静态函数，实现双线性插值二维上采样的反向传播
static void upsample_bicubic2d_backward_kernel(
    const Tensor& grad_input,  // 梯度输入张量
    const Tensor& grad_output_,  // 梯度输出张量（已进行连续化）
    IntArrayRef output_size,  // 输出大小数组
    IntArrayRef input_size,  // 输入大小数组
    bool align_corners,  // 是否对齐角点
    std::optional<double> scales_h,  // 高度尺度因子（可选）
    std::optional<double> scales_w) {  // 宽度尺度因子（可选）

  int64_t output_height = output_size[0];  // 输出高度
  int64_t output_width = output_size[1];  // 输出宽度

  int64_t nbatch = input_size[0];  // 批次大小
  int64_t channels = input_size[1];  // 通道数
  int64_t input_height = input_size[2];  // 输入高度
  int64_t input_width = input_size[3];  // 输入宽度

  auto grad_output = grad_output_.contiguous();  // 连续化梯度输出张量

  // 特殊情况：输入输出大小相同，直接复制梯度输出到梯度输入
  if (input_height == output_height && input_width == output_width) {
    grad_input.copy_(grad_output);  // 复制梯度输出到梯度输入
    return;  // 返回
  }

  // 根据梯度输入类型进行派发处理，支持浮点类型、半精度浮点类型、BFloat16 类型
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_bicubic2d_backward", [&] {
        scalar_t* idata = grad_input.mutable_data_ptr<scalar_t>();  // 可变指向梯度输入的数据指针
        const scalar_t* odata = grad_output.const_data_ptr<scalar_t>();  // 梯度输出的常量数据指针

        // 调用双线性插值二维反向传播输出帧的函数
        upsample_bicubic2d_backward_out_frame<scalar_t>(
            odata,
            idata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            align_corners,
            scales_h,
            scales_w);
      });
}
    # 使用可选参数 output_size 来确定输出大小的数组引用，如果未提供则为空
    at::OptionalIntArrayRef output_size,
    # 指定是否在对齐角落时使用插值
    bool align_corners,
    # 可选参数，包含双精度数组引用的可选项，表示高度和宽度的缩放因子
    std::optional<ArrayRef<double>> scale_factors) {
  # 计算输出大小，根据输入大小、output_size 和 scale_factors 确定
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  # 获取高度的缩放因子
  auto scale_h = get_scale_value(scale_factors, 0);
  # 获取宽度的缩放因子
  auto scale_w = get_scale_value(scale_factors, 1);
  # 使用双三次插值对二维输入进行上采样，使用给定的输出大小、对齐角落、以及高度和宽度的缩放因子
  return at::upsample_bicubic2d(input, osize, align_corners, scale_h, scale_w);
}

Tensor _upsample_bicubic2d_aa(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  // 计算输出尺寸
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  // 获取垂直方向的缩放因子
  auto scale_h = get_scale_value(scale_factors, 0);
  // 获取水平方向的缩放因子
  auto scale_w = get_scale_value(scale_factors, 1);
  // 调用 ATen 库中的双三次插值上采样函数
  return at::_upsample_bicubic2d_aa(input, osize, align_corners, scale_h, scale_w);
}

// 定义 ATen 分发的函数，用于双三次插值上采样的核心算法
DEFINE_DISPATCH(upsample_bicubic2d_kernel);

// 定义 ATen 分发的函数，用于带抗锯齿的双三次插值上采样的核心算法
DEFINE_DISPATCH(_upsample_bicubic2d_aa_kernel);

// 定义 ATen 分发的函数，用于带抗锯齿的双三次插值上采样的反向传播核心算法
DEFINE_DISPATCH(_upsample_bicubic2d_aa_backward_kernel);

} // namespace at::native
```