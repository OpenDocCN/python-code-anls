# `.\pytorch\aten\src\ATen\native\cpu\PixelShuffleKernel.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cpu/PixelShuffleKernel.h>

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

// 定义一个模板函数，用于在 CPU 上执行像素混洗操作
template <typename scalar_t>
void cpu_pixel_shuffle(
    TensorBase& output,                    // 输出张量的引用
    const TensorBase& input,               // 输入张量的常量引用
    int64_t upscale_factor) {              // 缩放因子

  auto input_data = input.const_data_ptr<scalar_t>();   // 获取输入张量的数据指针
  auto output_data = output.data_ptr<scalar_t>();       // 获取输出张量的数据指针

  // 输入张量的通道数、高度、宽度
  int64_t channels = input.size(-3);
  int64_t height = input.size(-2);
  int64_t width = input.size(-1);

  // 计算每个小块的通道数
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);

  // 输入张量的元素总数
  int64_t numel = input.numel();

  // 输入张量的批次数
  int64_t nbatch = numel / (channels * height * width);

  // 缩放因子
  int64_t S = upscale_factor;

  // 计算输入张量的步长
  int64_t stride_n = channels * height * width;   // 批次的步长
  int64_t stride_c = S * S * height * width;      // 通道的步长
  int64_t stride_s1 = S * height * width;         // 第一个缩放方向的步长
  int64_t stride_s2 = height * width;             // 第二个缩放方向的步长
  int64_t stride_h = width;                       // 高度的步长

  // 使用并行处理遍历输入张量
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    // 初始化索引变量
    int64_t n{0}, c{0}, h{0}, s1{0}, w{0}, s2{0};
    data_index_init(begin, n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);

    // 遍历指定范围内的元素
    for (const auto i : c10::irange(begin, end)) {
      // 计算输入偏移量
      int64_t input_offset = n * stride_n + c * stride_c + s1 * stride_s1 +
          s2 * stride_s2 + h * stride_h + w;

      // 将输入张量中的数据复制到输出张量中
      output_data[i] = input_data[input_offset];

      // 更新索引，准备处理下一个元素
      data_index_step(n, nbatch, c, sub_channels, h, height, s1, S, w, width, s2, S);
    }
  });
}

// 定义另一个模板函数，用于在 CPU 上执行通道为最后维度的像素混洗操作
template <typename scalar_t>
void cpu_pixel_shuffle_channels_last(
    TensorBase& output,                    // 输出张量的引用
    const TensorBase& input,               // 输入张量的常量引用
    int64_t upscale_factor) {              // 缩放因子

  // 检查输入张量的维度是否为4
  TORCH_CHECK(input.ndimension() == 4,
              "pixel shuffle with channels last format supports tensors with 4 dims");

  auto input_data = input.const_data_ptr<scalar_t>();   // 获取输入张量的数据指针
  auto output_data = output.data_ptr<scalar_t>();       // 获取输出张量的数据指针

  // 获取输入张量的各维度大小
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);

  // 计算每个小块的通道数
  int64_t sub_channels = channels / (upscale_factor * upscale_factor);

  // 缩放因子
  int64_t S = upscale_factor;

  // 使用并行处理遍历输入张量的前两个维度
  at::parallel_for(0, nbatch * height, 0, [&](int64_t begin, int64_t end) {
    // 创建一个临时缓冲区以保存每个通道的数据
    auto buffer = std::make_unique<scalar_t []>(channels);
    scalar_t* buffer_ptr = buffer.get();

    // 初始化索引变量
    int64_t n{0}, h{0};
    data_index_init(begin, n, nbatch, h, height);

    // 循环处理指定范围内的元素
    // 对于范围 [begin, end) 中的每个索引 i 循环执行
    for (const auto i : c10::irange(begin, end)) {
      // 对于宽度范围内的每个索引 w 循环执行
      for (const auto w : c10::irange(width)) {
        // 计算输入数据中特定位置的指针 input_ptr
        const scalar_t* input_ptr = input_data + n * height * width * channels + h * width * channels + w * channels;

        // 步骤 1: 转置每个通道的数据
        //   输入格式: [c, s1*s2]
        //   输出格式: [s1*s2, c]
        utils::transpose(sub_channels, S * S, input_ptr, S * S, buffer_ptr, sub_channels);

        // 步骤 2: 将临时缓冲区中的数据复制到输出数据中
        for (const auto s1 : c10::irange(S)) {
          // 定义临时缓冲区和输出数据的指针
          scalar_t* x_ptr = buffer_ptr + s1 * S * sub_channels;
          scalar_t* y_ptr = output_data + i * width * channels + s1 * width * S * sub_channels + w * S * sub_channels;

          // 计算要复制的数据大小和初始偏移量
          int64_t size = S * sub_channels;
          int64_t d = 0;
          // 使用 SIMD 加速，逐块复制数据
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(x_ptr + d);
            data_vec.store(y_ptr + d);
          }
          // 处理剩余的非对齐数据
          for (; d < size; d++) {
            y_ptr[d] = x_ptr[d];
          }
        }
      }

      // 调用 data_index_step 函数，更新数据索引
      data_index_step(n, nbatch, h, height);
    }
  });


这段代码实现了一个复杂的数据处理流程，包括矩阵转置和数据复制操作。
}

template <typename scalar_t>
void cpu_pixel_unshuffle(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  auto input_data = input.const_data_ptr<scalar_t>();  // 获取输入张量的常量数据指针
  auto output_data = output.data_ptr<scalar_t>();  // 获取输出张量的数据指针

  // [(B1...Bn), C, H, W] => [N, C, H, W]
  int64_t sub_channels = input.size(-3);  // 获取输入张量中子通道数
  int64_t height = input.size(-2) / downscale_factor;  // 计算输出高度
  int64_t width = input.size(-1) / downscale_factor;  // 计算输出宽度
  int64_t channels = sub_channels * downscale_factor * downscale_factor;  // 计算通道数
  int64_t numel = input.numel();  // 获取输入张量的元素总数
  int64_t nbatch = numel / (channels * height * width);  // 计算批次数
  int64_t S = downscale_factor;  // 缩小因子

  // input strides
  int64_t stride_n = channels * height * width;  // 批次步长
  int64_t stride_c = height * S * width * S;  // 通道步长
  int64_t stride_h = S * width * S;  // 高度步长
  int64_t stride_s1 = width * S;  // S1步长
  int64_t stride_w = S;  // 宽度步长
  int64_t stride_s2 = 1;  // S2步长

  // input tensor shape of [n, c, h, s1, w, s2]
  // output tensor shape of [n, c, s1, s2, h, w]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0}, s1{0}, s2{0}, h{0}, w{0};
    data_index_init(begin, n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);  // 初始化索引值

    for (const auto i : c10::irange(begin, end)) {
      int64_t input_offset = n * stride_n + c * stride_c + h * stride_h +
          s1 * stride_s1 + w * stride_w + s2 * stride_s2;  // 计算输入数据偏移量
      output_data[i] = input_data[input_offset];  // 执行像素解压操作

      data_index_step(n, nbatch, c, sub_channels, s1, S, s2, S, h, height, w, width);  // 更新索引值
    }
  });
}

template <typename scalar_t>
void cpu_pixel_unshuffle_channels_last(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  TORCH_CHECK(input.ndimension() == 4,
              "pixel unshuffle with channels last format supports tensors with 4 dims");  // 检查输入张量维度是否为4
  auto input_data = input.const_data_ptr<scalar_t>();  // 获取输入张量的常量数据指针
  auto output_data = output.data_ptr<scalar_t>();  // 获取输出张量的数据指针

  int64_t nbatch = input.size(0);  // 获取批次数
  int64_t sub_channels = input.size(1);  // 获取子通道数
  int64_t height = input.size(2) / downscale_factor;  // 计算输出高度
  int64_t width = input.size(3) / downscale_factor;  // 计算输出宽度
  int64_t channels = sub_channels * downscale_factor * downscale_factor;  // 计算通道数
  int64_t numel = input.numel();  // 获取输入张量的元素总数
  int64_t S = downscale_factor;  // 缩小因子

  // input strides
  int64_t stride_n = height * width * channels;  // 批次步长
  int64_t stride_h = S * width * S * sub_channels;  // 高度步长
  int64_t stride_s1 = width * S * sub_channels;  // S1步长
  int64_t stride_w = S * sub_channels;  // 宽度步长
  int64_t stride_s2 = sub_channels;  // S2步长
  int64_t stride_c = 1;  // 通道步长

  // input tensor shape of [n, h, s1, w, s2, c]
  // output tensor shape of [n, h, w, c, s1, s2]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, h{0}, w{0}, c{0}, s1{0}, s2{0};
    data_index_init(begin, n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);  // 初始化索引值
    // 使用范围遍历每个元素 i 在 [begin, end) 范围内
    for (const auto i : c10::irange(begin, end)) {
        // 计算输入数据中的偏移量，根据给定的索引 n, h, s1, w, s2, c，以及各自的步长
        int64_t input_offset = n * stride_n + h * stride_h + s1 * stride_s1 +
                               w * stride_w + s2 * stride_s2 + c * stride_c;
        // 将计算出的输入数据偏移量对应的数据复制到输出数据的位置 i
        output_data[i] = input_data[input_offset];

        // 调用 data_index_step 函数，计算下一个数据索引的步进
        data_index_step(n, nbatch, h, height, w, width, c, sub_channels, s1, S, s2, S);
    }
  });
} // 匿名命名空间结束

void pixel_shuffle_kernel_impl(
    TensorBase& output,
    const TensorBase& input,
    int64_t upscale_factor) {
  // 根据输入张量的建议内存格式进行分支处理
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 当内存格式为连续时执行像素重排操作
      // 使用宏处理所有数据类型，调用cpu_pixel_shuffle函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_shuffle", [&] {
        cpu_pixel_shuffle<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 当内存格式为通道最后时执行通道最后的像素重排操作
      // 使用宏处理所有数据类型，调用cpu_pixel_shuffle_channels_last函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_shuffle_channels_last", [&] {
        cpu_pixel_shuffle_channels_last<scalar_t>(output, input, upscale_factor);
      });
      break;
    }
    default:
      // 若内存格式不支持，抛出错误信息
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void pixel_unshuffle_kernel_impl(
    TensorBase& output,
    const TensorBase& input,
    int64_t downscale_factor) {
  // 根据输入张量的建议内存格式进行分支处理
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 当内存格式为连续时执行像素反重排操作
      // 使用宏处理所有数据类型，调用cpu_pixel_unshuffle函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_unshuffle", [&] {
        cpu_pixel_unshuffle<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 当内存格式为通道最后时执行通道最后的像素反重排操作
      // 使用宏处理所有数据类型，调用cpu_pixel_unshuffle_channels_last函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "pixel_unshuffle_channels_last", [&] {
        cpu_pixel_unshuffle_channels_last<scalar_t>(output, input, downscale_factor);
      });
      break;
    }
    default:
      // 若内存格式不支持，抛出错误信息
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // at::native

REGISTER_DISPATCH(pixel_shuffle_kernel, &pixel_shuffle_kernel_impl);
REGISTER_DISPATCH(pixel_unshuffle_kernel, &pixel_unshuffle_kernel_impl);

} // at::native 命名空间结束
```