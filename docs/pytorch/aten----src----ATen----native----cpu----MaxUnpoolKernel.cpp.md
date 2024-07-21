# `.\pytorch\aten\src\ATen\native\cpu\MaxUnpoolKernel.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/MaxUnpoolKernel.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include <c10/util/Optional.h>

namespace at::native {

namespace {

template <typename scalar_t, bool is_3d = false>
void cpu_max_unpool(
    Tensor& output_,                         // 输出张量的引用
    const Tensor& input,                     // 输入张量的常量引用
    const Tensor& indices) {                 // 索引张量的常量引用

  auto output = output_.contiguous();        // 将输出张量进行内存连续化处理

  auto input_data = input.const_data_ptr<scalar_t>();     // 获取输入张量数据的指针
  auto indices_data = indices.const_data_ptr<int64_t>();  // 获取索引张量数据的指针
  auto output_data = output.data_ptr<scalar_t>();         // 获取输出张量数据的指针

  // NB: input tensor dimensions:
  // MaxUnpool2d:
  //    dim = 3: CHW
  //    dim = 4: NCHW
  // MaxUnpool3d:
  //    dim = 4: CDHW
  //    dim = 5: NCDHW

  int64_t numel = input.numel();              // 输入张量的元素总数
  int64_t ndim = input.ndimension();          // 输入张量的维度数

  // treat batch size and channels as one dimension
  // and the feature map as another dimension
  int64_t channels, output_depth, output_height, output_width;
  if (is_3d) {
    TORCH_CHECK(ndim == 4 || ndim == 5, "MaxUnpool3d: expect input to be 4d or 5d tensor.");
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
    output_depth = output.size(-3);           // 输出深度
    output_height = output.size(-2);          // 输出高度
    output_width = output.size(-1);           // 输出宽度
  } else {
    TORCH_CHECK(ndim == 3 || ndim == 4, "MaxUnpool2d: expect input to be 3d or 4d tensor.");
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
    output_depth = 1;                         // 输出深度（2D情况下为1）
    output_height = output.size(-2);          // 输出高度
    output_width = output.size(-1);           // 输出宽度
  }
  int64_t input_image_size = numel / channels;        // 单个图像的输入张量大小
  int64_t output_image_size = output.numel() / channels;  // 单个图像的输出张量大小

  std::optional<int64_t> optional_error_index;    // 可选的错误索引

  // parallel on dim N, C, D, H, W: [channels, input_image_size]
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t ip = 0;
    data_index_init(begin, c, channels, ip, input_image_size);  // 初始化数据索引

    for (const auto i : c10::irange(begin, end)) {
      scalar_t* output_ptr = output_data + c * output_image_size;  // 输出指针位置

      int64_t maxp = indices_data[i];           // 获取最大池化索引
      if (maxp < 0 || maxp >= output_image_size) {  // 检查最大池化索引的有效性
        optional_error_index = maxp;
        std::atomic_thread_fence(std::memory_order_release);  // 设置内存顺序
      } else {
        output_ptr[maxp] = input_data[i];       // 写入输出张量的值
      }

      // move on to next input index
      data_index_step(c, channels, ip, input_image_size);  // 移动到下一个输入索引位置
    }
  });

  if (optional_error_index) {                   // 检查是否存在错误索引
    if (is_3d) {
      AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
          " (output volumes are of size ", output_depth,
          "x", output_height, "x", output_width);
    } else {
      AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
          " (output volumes are of size ", output_height,
          "x", output_width);
    }
  }

  if (!output_.is_contiguous()) {               // 检查输出张量是否连续
    output_.copy_(output);                      // 如果不连续，进行复制操作
  }
}

template <typename scalar_t>
// 检查输入张量的维度是否为4，因为channels last格式的max_unpool2d仅支持4维张量
void cpu_max_unpool_channels_last(
    Tensor& output_,
    const Tensor& input,
    const Tensor& indices) {
  TORCH_CHECK(input.ndimension() == 4,
              "max_unpool2d with channels last format supports tensors with 4 dims");

  // 将输出张量转换为指定内存格式（channels last）
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto output = output_.contiguous(memory_format);

  // 获取输入、索引和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入张量的维度信息
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);
  int64_t input_image_size = input_height * input_width;
  int64_t output_image_size = output_height * output_width;

  // 可选的错误索引，用于标记无效的最大池化索引
  std::optional<int64_t> optional_error_index;

  // 并行处理N、H、W维度上的计算
  at::parallel_for(0, nbatch * input_image_size, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t ip = 0;
    data_index_init(begin, n, nbatch, ip, input_image_size);

    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + i * channels;
      const int64_t* indices_ptr = indices_data + i * channels;
      scalar_t* output_ptr = output_data + n * output_image_size * channels;

      // 在AVX2上无法执行scatter操作（仅在AVX512上可用）
      for (const auto c : c10::irange(channels)) {
        int64_t maxp = indices_ptr[c];
        // 检查最大池化索引是否有效，如果无效则记录错误索引
        if (maxp < 0 || maxp >= output_image_size) {
          optional_error_index = maxp;
          std::atomic_thread_fence(std::memory_order_release);
        } else {
          output_ptr[maxp * channels + c] = input_ptr[c];
        }
      }

      // 移动到下一个输入索引
      data_index_step(n, nbatch, ip, input_image_size);
    }
  });

  // 如果存在错误索引，抛出错误信息
  if (optional_error_index) {
    AT_ERROR("Found an invalid max index: ", optional_error_index.value(),
        " (output volumes are of size ", output_height,
        "x", output_width, ")");
  }

  // 如果输出张量未按照指定的内存格式连续，则进行复制操作
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}
    // 如果输入张量是四维的，计算通道数；如果是三维的，计算通道数乘以第二维度大小
    channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
    
    // 获取输出张量的深度（第一个轴的大小）
    output_depth = grad_output.size(-3);
    
    // 获取输出张量的高度（倒数第二个轴的大小）
    output_height = grad_output.size(-2);
    
    // 获取输出张量的宽度（最后一个轴的大小）
    output_width = grad_output.size(-1);
    } else {
    // 如果不是四维张量，检查是否是三维或四维张量，否则报错
    TORCH_CHECK(ndim == 3 || ndim == 4, "MaxUnpool2d_backward: expect grad_output to be 3d or 4d tensor.");
    
    // 如果是三维张量，计算通道数；如果是四维张量，计算通道数乘以第二维度大小
    channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
    
    // 对于三维情况下，输出深度为1；在四维情况下，使用输出张量的第一个轴的大小作为输出深度
    output_depth = 1;
    
    // 获取输出张量的高度（倒数第二个轴的大小）
    output_height = grad_output.size(-2);
    
    // 获取输出张量的宽度（最后一个轴的大小）
    output_width = grad_output.size(-1);
    }
    
    // 计算输入图像的大小，即总元素数除以通道数
    int64_t input_image_size = numel / channels;
    
    // 计算输出图像的大小，即grad_output张量的总元素数除以通道数
    int64_t output_image_size = grad_output.numel() / channels;
    
    // 初始化一个可选的错误索引，用于记录并行计算中出现的错误
    std::optional<int64_t> optional_error_index;
    
    // 在N、C、D、H、W这五个维度上并行执行
    at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
      int64_t c = 0; // 初始化通道数索引
      int64_t ip = 0; // 初始化输入图像像素索引
      data_index_init(begin, c, channels, ip, input_image_size); // 初始化数据索引
    
      for (const auto i : c10::irange(begin, end)) {
        scalar_t* grad_output_ptr = grad_output_data + c * output_image_size; // 指向当前通道的grad_output数据指针
    
        int64_t maxp = indices_data[i]; // 获取当前索引处的最大池化位置索引
    
        // 检查最大池化位置索引是否有效
        if (maxp < 0 || maxp >= output_image_size) {
            optional_error_index = maxp; // 记录错误索引
            std::atomic_thread_fence(std::memory_order_release); // 确保错误索引的写入操作完全完成
        } else {
          grad_input_data[i] = grad_output_ptr[maxp]; // 将grad_output中的梯度值写入到grad_input中
        }
    
        // 移动到下一个输入索引位置
        data_index_step(c, channels, ip, input_image_size);
      }
    });
    
    // 如果存在错误索引，根据是否是三维情况报错
    if (optional_error_index) {
      if (is_3d) {
        AT_ERROR("invalid max index ", optional_error_index.value(),
            ", odepth= ", output_depth,
            ", owidth= ", output_width,
            ", oheight= ", output_height);
      } else {
        AT_ERROR("invalid max index ", optional_error_index.value(),
            ", owidth= ", output_width,
            ", oheight= ", output_height);
      }
    }
    
    // 如果grad_input_不是连续的，将grad_input的内容复制到grad_input_
    if (!grad_input_.is_contiguous()) {
      grad_input_.copy_(grad_input);
    }
} // anonymous namespace



REGISTER_DISPATCH(max_unpool2d_kernel, &max_unpool2d_kernel_impl);



REGISTER_DISPATCH(max_unpool3d_kernel, &max_unpool3d_kernel_impl);



} // at::native


这些代码片段属于一个 C++ 程序，主要涉及注册分发函数和命名空间的结尾标记。

- `}` // anonymous namespace
  - 匿名命名空间的结束标记，用于限制命名空间的作用域和可见性。

- `REGISTER_DISPATCH(max_unpool2d_kernel, &max_unpool2d_kernel_impl);`
  - 注册函数 `max_unpool2d_kernel_impl` 作为 `max_unpool2d_kernel` 的分发实现。

- `REGISTER_DISPATCH(max_unpool3d_kernel, &max_unpool3d_kernel_impl);`
  - 注册函数 `max_unpool3d_kernel_impl` 作为 `max_unpool3d_kernel` 的分发实现。

- `} // at::native`
  - 命名空间 `at::native` 的结束标记，用于限制其中的函数和变量的作用域和可见性。
  
这些代码片段一起组成了一个完整的 C++ 文件尾部，包括匿名命名空间的关闭和分发函数的注册，以及命名空间 `at::native` 的结束。
```