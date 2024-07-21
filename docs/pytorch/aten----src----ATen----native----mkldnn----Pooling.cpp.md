# `.\pytorch\aten\src\ATen\native\mkldnn\Pooling.cpp`

```py
// 定义编译时使用的宏，指定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含头文件：张量、配置、梯度模式、大小调整、参数工具等
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/Resize.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>
#include <tuple>

// 根据条件选择是否包含操作符头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_native.h>
#include <ATen/ops/mkldnn_max_pool2d_backward_native.h>
#include <ATen/ops/mkldnn_max_pool2d_native.h>
#include <ATen/ops/mkldnn_max_pool3d_backward_native.h>
#include <ATen/ops/mkldnn_max_pool3d_native.h>
#endif

// 如果未启用 MKLDNN 支持，则定义空间at::native命名空间
#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

// 定义未启用 MKLDNN 支持时的各种池化函数，抛出错误信息
Tensor mkldnn_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(false, "mkldnn_max_pool2d: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor mkldnn_max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(false, "mkldnn_max_pool3d: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor mkldnn_avg_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(false, "mkldnn_avg_pool2d: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor& mkldnn_avg_pool2d_out(const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "mkldnn_avg_pool2d_out: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor mkldnn_avg_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(false, "mkldnn_avg_pool3d: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor& mkldnn_avg_pool3d_out(const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& output) {
  TORCH_CHECK(false, "mkldnn_avg_pool3d_out: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor mkldnn_adaptive_avg_pool2d(Tensor const& input, IntArrayRef output_size) {
  TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d: ATen not compiled with MKLDNN support");
  // 函数没有返回值，因为抛出了异常
}

Tensor& mkldnn_adaptive_avg_pool2d_out_stub(const Tensor& input,
    // 检查条件是否为假，如果是，则抛出错误信息并终止程序，说明 ATen 没有使用 MKLDNN 支持编译
    TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_out_stub: ATen not compiled with MKLDNN support");
#else // AT_MKLDNN_ENABLED


// 如果 AT_MKLDNN_ENABLED 宏未定义，则编译这部分代码
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

// 定义静态函数 _mkldnn_pooling，用于 MKLDNN 池化操作
static Tensor _mkldnn_pooling(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,


这段代码是C++中的预处理指令和声明语句。
  // 计算输入张量的维度数减去2，即空间维度
  const int64_t dims = input.dim() - 2;
  // 根据需要扩展核大小参数到对应维度
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  // 如果步长参数为空，则将其设为核大小
  if (stride.empty()) stride = kernel_size;
  // 根据需要扩展步长参数到对应维度
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  // 根据需要扩展填充参数到对应维度
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  // 复制填充向量以备份
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  // 根据需要扩展膨胀参数到对应维度
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  // 将输入张量转换为MKLDNN格式的张量
  const ideep::tensor& x = itensor_from_mkldnn(input);
  // 存储输出尺寸的向量
  std::vector<int64_t> output_sizes;

  // 如果使用ceil模式
  if (ceil_mode) {
    // MKLDNN不支持ceil模式，因此调整右侧填充以匹配行为，并相应地调整输出大小
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // 调整填充直到输出大小一致
    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /* ceil_mode */);

      all_equal = true;
      // 检查每个空间维度上的输出尺寸是否满足ceil模式下的尺寸
      for (const auto i : c10::irange(2, input.sizes().size())) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  } else {
    // 否则，使用标准模式计算输出尺寸
    output_sizes = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        false /* ceil_mode */);
  }

  // 设定算法的推理方式为前向
  auto aprop_kind = ideep::prop_kind::forward;
  // 对于最大池化，如果不需要梯度或处于推理模式，则设定为前向推理
  if (ideep::algorithm::pooling_max == algo
      && !((input.requires_grad() && at::GradMode::is_enabled()) || input._fw_grad(/* level */ 0).defined())) {
    aprop_kind = ideep::prop_kind::forward_inference;
  }

  // 定义输出张量y
  ideep::tensor y;
  // 执行池化操作的前向计算
  ideep::pooling_forward::compute(
      x,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride_vec.cbegin(), stride_vec.cend()},
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algo,
      aprop_kind);

  // 使用MKLDNN张量创建新的PyTorch张量，并返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()), input.options().device_opt());
}

// 定义一个静态函数，用于计算 MKLDNN 最大池化操作的反向传播
static Tensor _mkldnn_pooling_backward(
    const Tensor& grad_output,  // 梯度输出张量
    const Tensor& output,       // 输出张量
    const Tensor& input,        // 输入张量
    IntArrayRef kernel_size,    // 池化核大小
    IntArrayRef stride,         // 步幅
    IntArrayRef padding,        // 填充
    IntArrayRef dilation,       // 空洞
    bool ceil_mode,             // 是否启用 ceil 模式
    ideep::algorithm algo) {    // 算法选择

  // 计算输入张量的维度，排除批量和通道维度
  const int64_t dims = input.dim() - 2;
  
  // 根据需要扩展池化核大小、步幅、填充和空洞参数
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  auto padding_vec_l = padding_vec;  // 左填充向量
  auto padding_vec_r = padding_vec;  // 右填充向量
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  // 如果启用 ceil 模式，调整填充向量以匹配输出尺寸
  if (ceil_mode) {
    // 计算 ceil 模式下的输出尺寸
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // 调整填充直到输出尺寸一致
    bool all_equal = false;
    std::vector<int64_t> output_sizes;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /* ceil_mode */);

      all_equal = true;
      for (const auto i : c10::irange(2, input.sizes().size())) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  }

  // 从 PyTorch 张量转换为 MKLDNN 张量
  const ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  const ideep::tensor& y = itensor_from_mkldnn(output);
  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor gradx;
  
  // 执行池化操作的反向传播计算
  ideep::pooling_backward::compute(
      grady,
      y,
      x,
      gradx,
      {stride_vec.cbegin(), stride_vec.cend()},          // 步幅
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},// 池化核大小
      {padding_vec_l.cbegin(), padding_vec_l.cend()},    // 左填充
      {padding_vec_r.cbegin(), padding_vec_r.cend()},    // 右填充
      algo);                                             // 算法选择

  // 创建一个新的 PyTorch 张量并返回，基于 MKLDNN 张量
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

// 执行 MKLDNN 最大池化操作
Tensor mkldnn_max_pool2d(
    const Tensor& input,        // 输入张量
    IntArrayRef kernel_size,    // 池化核大小
    IntArrayRef stride,         // 步幅
    IntArrayRef padding,        // 填充
    IntArrayRef dilation,       // 空洞
    bool ceil_mode) {           // 是否启用 ceil 模式

  // 检查是否支持膨胀情况（不支持）
  TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "mkldnn_max_pool2d does not support dilation case");

  // 如果输入张量为 BFloat16 类型
  if (input.scalar_type() == ScalarType::BFloat16) {
    // 使用 TORCH_CHECK 宏来检查是否满足 MKL-DNN 的 BF16 设备要求
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_max_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 调用 _mkldnn_pooling 函数执行池化操作
  return _mkldnn_pooling(
      input,                // 输入张量
      kernel_size,          // 池化核大小
      stride,               // 步幅大小
      padding,              // 填充大小
      dilation,             // 膨胀大小
      ceil_mode,            // 是否使用向上取整模式
      ideep::algorithm::pooling_max);  // 指定使用的池化算法为最大池化
}

// 使用 MKLDNN 库实现 3D 最大池化操作
Tensor mkldnn_max_pool3d(
    const Tensor& input,                        // 输入张量
    IntArrayRef kernel_size,                     // 池化核大小
    IntArrayRef stride,                          // 步幅大小
    IntArrayRef padding,                         // 填充大小
    IntArrayRef dilation,                        // 膨胀大小
    bool ceil_mode) {                            // 是否使用 ceil 模式

  // 检查所有膨胀值是否为 1
  TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "mkldnn_max_pool3d does not support dilation case");

  // 如果输入张量是 BFloat16 类型，检查是否支持相应的 CPU 特性
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_max_pool3d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 调用 _mkldnn_pooling 函数进行池化操作，选择最大池化算法
  return _mkldnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

// 使用 MKLDNN 库实现 2D 平均池化操作
Tensor mkldnn_avg_pool2d(
    const Tensor& input,                        // 输入张量
    IntArrayRef kernel_size,                     // 池化核大小
    IntArrayRef stride,                          // 步幅大小
    IntArrayRef padding,                         // 填充大小
    bool ceil_mode,                              // 是否使用 ceil 模式
    bool count_include_pad,                      // 是否包括填充部分计数
    std::optional<int64_t> divisor_override) {   // 除数覆盖选项

  // 检查是否存在除数覆盖选项，该操作不支持除数
  TORCH_CHECK(!divisor_override.has_value(),
      "mkldnn_avg_pool2d operator does not support divisor");

  // 如果输入张量是 BFloat16 类型，检查是否支持相应的 CPU 特性
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 调用 _mkldnn_pooling 函数进行池化操作，选择平均池化算法，根据 count_include_pad 决定是否包括填充部分
  return _mkldnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      /*dilation*/ std::vector<int64_t>{1, 1},   // 固定 dilation 为 {1, 1}
      ceil_mode,
      count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                        : ideep::algorithm::pooling_avg_exclude_padding);
}

// 使用 MKLDNN 库实现 2D 平均池化操作的输出版本，不支持原地操作
Tensor& mkldnn_avg_pool2d_out(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& output) {

  // 报错，暂时不支持原地操作
  TORCH_CHECK(false, "mkldnn_avg_pool2d_out: in-place mkldnn operations are not supported yet");
}

// 使用 MKLDNN 库实现 3D 平均池化操作
Tensor mkldnn_avg_pool3d(
    const Tensor& input,                        // 输入张量
    IntArrayRef kernel_size,                     // 池化核大小
    IntArrayRef stride,                          // 步幅大小
    IntArrayRef padding,                         // 填充大小
    bool ceil_mode,                              // 是否使用 ceil 模式
    bool count_include_pad,                      // 是否包括填充部分计数
    std::optional<int64_t> divisor_override) {   // 除数覆盖选项

  // 检查是否存在除数覆盖选项，该操作不支持除数
  TORCH_CHECK(!divisor_override.has_value(), "mkldnn_avg_pool3d operator does not support divisor");

  // 如果输入张量是 BFloat16 类型，检查是否支持相应的 CPU 特性
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_avg_pool3d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // 调用 _mkldnn_pooling 函数进行池化操作，选择平均池化算法，根据 count_include_pad 决定是否包括填充部分
  return _mkldnn_pooling(
      input,
      kernel_size,
      stride,
      padding,
      /*dilation*/ std::vector<int64_t>{1, 1, 1},   // 固定 dilation 为 {1, 1, 1}
      ceil_mode,
      count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                        : ideep::algorithm::pooling_avg_exclude_padding);
}

// 使用 MKLDNN 库实现 3D 平均池化操作的输出版本，不支持原地操作
Tensor& mkldnn_avg_pool3d_out(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    Tensor& output) {
    # 使用 TORCH_CHECK 断言函数，检查条件为 false
    # 输出错误信息表明不支持 mkldnn_avg_pool3d_out 的原地操作
  TORCH_CHECK(false, "mkldnn_avg_pool3d_out: in-place mkldnn operations are not supported yet");
}

// 计算 MKLDNN 自适应平均池化的结果
Tensor mkldnn_adaptive_avg_pool2d(
    // 输入张量
    Tensor const& input,
    // 输出大小
    IntArrayRef output_size) {
  // 检查输入张量是否为四维
  TORCH_CHECK(input.dim() == 4, "mkldnn_adaptive_avg_pool2d: Expect 2D input");
  // 如果输入张量的数据类型是 BFloat16
  if (input.scalar_type() == ScalarType::BFloat16) {
    // 检查是否支持 bf16 路径
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  // 根据需要扩展输出大小的参数
  auto output_size_vec =
      expand_param_if_needed(output_size, "output_size", input.dim() - 2);
  // 初始化池化核大小的向量
  std::vector<int64_t> kernel_size(input.dim() - 2);
  // 遍历输入张量的维度，计算每个维度的池化核大小
  for (const auto i : c10::irange(2, input.dim())) {
    auto s1 = input.size(i);
    auto s2 = output_size_vec[i - 2];
    // 检查输出大小不能为零
    TORCH_CHECK(s2 != 0, "output size can not be zero");
    // 检查输入大小能否被输出大小整除
    TORCH_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    // 计算池化核大小
    kernel_size[i - 2] = s1 / s2;
  }
  // 调用具体的 MKLDNN 池化操作函数
  return _mkldnn_pooling(
      input,
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ {0, 0},
      /*dilation*/ {1, 1},
      /*ceil_mode*/ false,
      /*algo*/ ideep::algorithm::pooling_avg_exclude_padding);
}

// 原地计算 MKLDNN 自适应平均池化的输出（存根函数）
Tensor& mkldnn_adaptive_avg_pool2d_out_stub(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  // 报错，不支持原地执行的 MKLDNN 操作
  TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_out_stub: in-place mkldnn operations are not supported yet");
}

// 计算 MKLDNN 自适应平均池化的输出
Tensor& mkldnn_adaptive_avg_pool2d_out(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  // 调用非原地版本的函数获取临时输出
  auto tmp_output = at::native::mkldnn_adaptive_avg_pool2d(input, output_size);
  // 调整输出张量的大小
  at::native::resize_output(output, tmp_output.sizes());
  // 将临时输出复制到输出张量
  output.copy_(tmp_output);
  // 返回输出张量
  return output;
}

// 计算 MKLDNN 最大池化反向传播的结果
Tensor mkldnn_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 调用具体的 MKLDNN 最大池化反向传播操作函数
  return _mkldnn_pooling_backward(
      grad_output,
      output,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

// 计算 MKLDNN 三维最大池化反向传播的结果
Tensor mkldnn_max_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 调用具体的 MKLDNN 三维最大池化反向传播操作函数
  return _mkldnn_pooling_backward(
      grad_output,
      output,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

// 计算 MKLDNN 平均池化反向传播的结果
Tensor mkldnn_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    // 使用 MKL-DNN 库进行池化操作的反向传播计算
    return _mkldnn_pooling_backward(
        // 反向传播时的输入梯度
        grad_output,
        // 反向传播时的输出梯度（通常与输入梯度相同）
        grad_output,
        // 池化操作的输入数据
        input,
        // 池化核大小
        kernel_size,
        // 池化操作的步幅
        stride,
        // 池化操作的填充
        padding,
        // 池化操作的扩张（在此处固定为 [1, 1]）
        /*dilation*/ std::vector<int64_t>{1, 1},
        // 是否启用 ceil 模式
        ceil_mode,
        // 是否包含填充值在内的计数方式
        count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                          : ideep::algorithm::pooling_avg_exclude_padding);
} // 关闭命名空间 'native'

} // 关闭命名空间 'at'

#endif // 如果 MKLDNN 功能已启用，则关闭宏定义
```