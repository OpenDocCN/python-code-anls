# `.\pytorch\aten\src\ATen\native\quantized\cpu\AveragePool2d.cpp`

```
// 定义宏，用于限定仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入头文件，包括 ATen 库的基本组件
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Pool.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

// 根据条件选择不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/avg_pool2d_native.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// 定义命名空间 at 和 native
namespace at {
namespace native {

// 定义分发函数 qavg_pool2d_nhwc_stub
DEFINE_DISPATCH(qavg_pool2d_nhwc_stub);

// 匿名命名空间，包含平均池化操作的模板函数定义
namespace {

// 模板函数，计算平均池化的输出帧
template <typename scalar_t>
static void avg_pool2d_out_frame(
    const Tensor& input,           // 输入张量
    Tensor& output,                // 输出张量
    int64_t nInputPlane,           // 输入通道数
    int64_t inputWidth,            // 输入宽度
    int64_t inputHeight,           // 输入高度
    int64_t outputWidth,           // 输出宽度
    int64_t outputHeight,          // 输出高度
    int kW,                        // 池化窗口宽度
    int kH,                        // 池化窗口高度
    int dW,                        // 宽度步长
    int dH,                        // 高度步长
    int padW,                      // 宽度填充
    int padH,                      // 高度填充
    bool count_include_pad,        // 是否包括填充
    std::optional<int64_t> divisor_override) {  // 可选的除数覆盖

  // 将输入张量转换为连续存储的张量
  Tensor input_contig = input.contiguous();
  // 获取输入和输出数据指针
  auto input_data = input_contig.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  // 计算输入和输出的量化比例因子
  const auto scale_factor = input.q_scale() / output.q_scale();
  // 获取输入和输出的量化零点
  const auto input_zero_point = input.q_zero_point();
  const auto output_zero_point = output.q_zero_point();

  // 并行计算，遍历每个输入通道
  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    // 对于给定范围内的每个索引 k
    for (const auto k : c10::irange(start, end)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 定义变量 xx 和 yy
      int64_t xx, yy;
      /* 对于所有输出像素... */
      // 计算输出数据指针的起始位置
      scalar_t* ptr_output = output_data + k * outputWidth * outputHeight;
      // 计算输入数据指针的起始位置
      const scalar_t* ptr_input = input_data + k * inputWidth * inputHeight;
      // 计算 scalar_t 类型的最小值和最大值
      auto minimum =
          std::numeric_limits<typename scalar_t::underlying>::lowest();
      auto maximum = std::numeric_limits<typename scalar_t::underlying>::max();

      // 遍历输出图像的高度和宽度
      for (yy = 0; yy < outputHeight; yy++) {
        for (xx = 0; xx < outputWidth; xx++) {
          /* 计算输入图像的均值... */
          // 计算输入图像的高度和宽度的起始和结束位置
          int64_t hstart = yy * dH - padH;
          int64_t wstart = xx * dW - padW;
          int64_t hend = std::min(hstart + kH, inputHeight + padH);
          int64_t wend = std::min(wstart + kW, inputWidth + padW);
          int64_t pool_size = (hend - hstart) * (wend - wstart);
          // 确保起始和结束位置在有效范围内
          hstart = std::max(hstart, (int64_t)0);
          wstart = std::max(wstart, (int64_t)0);
          hend = std::min(hend, inputHeight);
          wend = std::min(wend, inputWidth);

          int sum_int = 0;
          // 初始化输出指针的值为 0
          ptr_output->val_ = 0;

          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t divide_factor;
          int64_t size = (hend - hstart) * (wend - wstart);
          // 根据是否有覆盖的除数值来确定分母
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if (count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }

          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t kx, ky;
          // 计算输入图像指定区域内的像素值总和
          for (ky = hstart; ky < hend; ky++) {
            for (kx = wstart; kx < wend; kx++)
              sum_int += (ptr_input + ky * inputWidth + kx)->val_;
          }
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          // 计算乘数，用于缩放总和以及更新输出像素值
          float multiplier = scale_factor / divide_factor;

          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          // 将输入零点值减去到总和，然后转换为浮点数
          sum_int -= size * input_zero_point;
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          float sum = sum_int * 1.0;
          /* 通过重新量化结果更新输出 */
          // 将计算后的值重新量化并更新输出指针的值
          ptr_output->val_ =
              static_cast<typename scalar_t::underlying>(std::min<int32_t>(
                  std::max<int32_t>(
                      std::nearbyint(sum * multiplier + output_zero_point),
                      minimum),
                  maximum));
          ptr_output++;
        }
      }
    }
  });


这段代码是一个嵌套的循环结构，用于对输入图像的不同区域进行池化操作，并将结果存储在输出数据中。
}

// 获取卷积核大小的函数
inline std::pair<int, int> get_kernel(IntArrayRef kernel_size) {
  // 检查 kernel_size 的大小，必须为 1 或 2
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  // 将第一个维度转换为整数作为卷积核的高度 kH
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  // 如果 kernel_size 的大小为 1，则宽度 kW 也为 kH；否则将第二个维度转换为整数作为卷积核的宽度 kW
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);
  // 返回卷积核大小的 pair 对象
  return std::make_pair(kW, kH);
}

// 获取步长的函数
inline std::pair<int, int> get_stride(IntArrayRef stride, int kW, int kH) {
  // 检查 stride 是否为空，或者大小为 1 或 2
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  // 如果 stride 为空，则步长 dH 和 dW 均为卷积核的高度 kH
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  // 如果 stride 为空，则步长 dW 和 dH 均为卷积核的宽度 kW；否则将第二个维度转换为整数作为步长 dW
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);
  // 返回步长的 pair 对象
  return std::make_pair(dW, dH);
}

// 获取填充的函数
inline std::pair<int, int> get_padding(IntArrayRef padding) {
  // 检查 padding 的大小，必须为 1 或 2
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  // 将第一个维度转换为整数作为填充的高度 padH
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  // 如果 padding 的大小为 1，则宽度 padW 也为 padH；否则将第二个维度转换为整数作为填充的宽度 padW
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  // 返回填充的 pair 对象
  return std::make_pair(padW, padH);
}

// 获取输出形状的函数
std::vector<int64_t> get_output_shape(
    const Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool ceil_mode) {
  // 获取输入张量的批次大小、输入通道数、输入高度和宽度
  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);
  // 计算输出高度和宽度
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  // 如果输入张量维度为 3，则返回输出形状的向量 {nInputPlane, outputHeight, outputWidth}
  if (input_.ndimension() == 3) {
    return {nInputPlane, outputHeight, outputWidth};
  }
  // 否则返回 {nbatch, nInputPlane, outputHeight, outputWidth}
  return {nbatch, nInputPlane, outputHeight, outputWidth};
}

// 定义量化平均池化函数的模板
template <typename scalar_t>
Tensor q_avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    // 使用 std::optional<int64_t> 类型的 divisor_override 参数，表示一个可选的除数重写值
    std::optional<int64_t> divisor_override) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      // 从 kernel_size 获取卷积核的宽度和高度
      auto [kW, kH] = get_kernel(kernel_size);
      // 从 stride 和已知的卷积核大小获取步长的宽度和高度
      auto [dW, dH] = get_stride(stride, kW, kH);
      // 从 padding 获取填充的宽度和高度
      auto [padW, padH] = get_padding(padding);
    
      // 确定输入张量的批次大小（如果是四维张量）或默认为1
      const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
      // 确定输入张量的输入通道数
      const int64_t nInputPlane = input.size(-3);
      // 确定输入张量的高度
      const int64_t inputHeight = input.size(-2);
      // 确定输入张量的宽度
      const int64_t inputWidth = input.size(-1);
    
      // 检查 divisor_override 是否有值且不为零，否则抛出错误
      TORCH_CHECK(
          !divisor_override.has_value() || divisor_override.value() != 0,
          "divisor must be not zero");
    
      // 获取输出形状，根据输入、卷积核、步长、填充和 ceil_mode 计算得到
      auto output_shape =
          get_output_shape(input, kW, kH, dW, dH, padW, padH, ceil_mode);
      // 确定输出张量的高度
      const int64_t outputHeight = output_shape[output_shape.size() - 2];
      // 确定输出张量的宽度
      const int64_t outputWidth = output_shape[output_shape.size() - 1];
    
      // 如果输入张量以 ChannelsLast 的内存格式存储
      if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
        // 创建一个空的量化仿射输出张量，使用输入建议的内存格式和量化参数
        auto output = at::_empty_affine_quantized(
            output_shape,
            input.options().memory_format(input.suggest_memory_format()),
            input.q_scale(),
            input.q_zero_point(),
            c10::nullopt);
        // 对于通道最后存储方式的快速路径：调用 qavg_pool_2d_nhwc_stub 进行平均池化操作
        qavg_pool2d_nhwc_stub(
            input.device().type(),
            input,
            output,
            nbatch,
            nInputPlane,
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            count_include_pad,
            divisor_override);
        // 返回输出张量
        return output;
      } else {
        // 创建一个空的仿射量化输出张量，使用输入的选项和量化参数
        auto output = at::_empty_affine_quantized(
            output_shape, input.options(), input.q_scale(), input.q_zero_point());
        // 对于非通道最后存储方式：调用 avg_pool2d_out_frame 进行平均池化操作
        avg_pool2d_out_frame<scalar_t>(
            input,
            output,
            // 将批次和通道合并为一个维度
            nbatch * nInputPlane,
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            count_include_pad,
            divisor_override);
        // 返回输出张量
        return output;
      }
    }
#ifdef USE_PYTORCH_QNNPACK
// 如果使用了 QNNPACK，并且输入张量为 kQUInt8 类型且未开启 ceil_mode
if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
    input.scalar_type() == kQUInt8 && !ceil_mode) {
    // 调用 QNNPACK 版本的平均池化函数，并返回其结果张量
    return at::native::qnnp_avgpool_helper::qnnpack_avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
}
#endif

// 根据输入的标量类型分发到不同的平均池化函数
AT_DISPATCH_QINT_TYPES(input.scalar_type(), "avg_pool2d_quantized_cpu", [&]() {
    // 调用对应标量类型的平均池化函数，并将结果赋给 output
    output = q_avg_pool2d<scalar_t>(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
});

// 返回计算得到的输出张量
return output;
```