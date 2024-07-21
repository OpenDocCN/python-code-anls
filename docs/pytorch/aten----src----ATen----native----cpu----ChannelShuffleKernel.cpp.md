# `.\pytorch\aten\src\ATen\native\cpu\ChannelShuffleKernel.cpp`

```
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cpu/ChannelShuffleKernel.h>

#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

// CPU实现通道重排函数，用于处理输出和输入张量的数据重排
template <typename scalar_t>
void cpu_channel_shuffle(
    TensorBase& output,  // 输出张量的引用
    const TensorBase& input,  // 输入张量的常量引用
    int64_t groups) {  // 分组数

  auto input_data = input.data_ptr<scalar_t>();  // 获取输入数据的指针
  auto output_data = output.data_ptr<scalar_t>();  // 获取输出数据的指针

  int64_t nbatch = input.size(0);  // 批量大小
  int64_t channels = input.size(1);  // 通道数
  int64_t channels_per_group = channels / groups;  // 每个分组中的通道数
  int64_t image_size = input.numel() / nbatch / channels;  // 每个图像的大小

  // 以 [n, g, oc, ...] 的形状处理输入张量，以 [n, oc, g, ...] 的形状处理输出张量
  // 对于3D、4D、5D张量，在n和c维度上并行处理
  using Vec = vec::Vectorized<scalar_t>;
  int64_t inner_size = image_size - (image_size % Vec::size());
  at::parallel_for (0, nbatch * /* oc*g */channels, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oc = 0;
    int64_t g = 0;
    data_index_init(begin, n, nbatch, oc, channels_per_group, g, groups);

    // 遍历每个输出索引范围内的元素
    for (const auto i : c10::irange(begin, end)) {
      scalar_t* output_ptr = output_data + i * image_size;  // 输出指针位置
      scalar_t* input_ptr = input_data + n * channels * image_size +
          g * channels_per_group * image_size + oc * image_size;  // 输入指针位置

      int64_t d = 0;
      // 使用向量化加载和存储来处理内部大小
      for (; d < inner_size; d += Vec::size()) {
        Vec data_vec = Vec::loadu(input_ptr + d);  // 加载输入数据向量
        data_vec.store(output_ptr + d);  // 存储输出数据向量
      }
      // 处理剩余的不是向量大小的部分
      for (; d < image_size; d++) {
        output_ptr[d] = input_ptr[d];  // 直接复制数据
      }

      // 移动到下一个输出索引
      data_index_step(n, nbatch, oc, channels_per_group, g, groups);  // 更新索引位置
    }
  });
}

// CPU实现通道重排的特殊情况，用于处理数据的转置
template <typename scalar_t>
void cpu_channel_shuffle_cl(
    TensorBase& output,  // 输出张量的引用
    const TensorBase& input,  // 输入张量的常量引用
    int64_t groups) {  // 分组数

  auto input_data = input.data_ptr<scalar_t>();  // 获取输入数据的指针
  auto output_data = output.data_ptr<scalar_t>();  // 获取输出数据的指针

  int64_t nbatch = input.size(0);  // 批量大小
  int64_t channels = input.size(1);  // 通道数
  int64_t channels_per_group = channels / groups;  // 每个分组中的通道数
  int64_t image_size = input.numel() / nbatch / channels;  // 每个图像的大小

  // 对于4D张量，在n、h、w维度上并行处理；对于5D张量，在n、d、h、w维度上并行处理
  at::parallel_for(0, nbatch * image_size, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      scalar_t* output_ptr = output_data + i * channels;  // 输出指针位置
      scalar_t* input_ptr = input_data + i * channels;  // 输入指针位置

      // 对每个通道进行转置操作：从 [groups, channels_per_group] 到 [channels_per_group, groups]
      utils::transpose(groups, channels_per_group, input_ptr, channels_per_group, output_ptr, groups);
    }
  });
}

// 通道重排内核实现函数，根据输入张量的内存格式选择合适的重排实现方式
void channel_shuffle_kernel_impl(
    TensorBase& output,  // 输出张量的引用
    const TensorBase& input,  // 输入张量的常量引用
    int64_t groups) {  // 分组数

  switch (input.suggest_memory_format()) {  // 根据输入张量的内存格式选择实现方式
    # 处理内存格式为Contiguous时的情况
    case at::MemoryFormat::Contiguous: {
      # 使用AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3宏，根据输入张量的数据类型调度通道重排函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "channel_shuffle", [&] {
        # 调用CPU版本的通道重排函数，处理输出、输入和分组数
        cpu_channel_shuffle<scalar_t>(output, input, groups);
      });
      break;
    }
    # 处理内存格式为ChannelsLast或ChannelsLast3d时的情况
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d: {
      # 使用AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3宏，根据输入张量的数据类型调度ChannelsLast内存格式的通道重排函数
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
          input.scalar_type(), "channel_shuffle_cl", [&] {
        # 调用CPU版本的ChannelsLast内存格式的通道重排函数，处理输出、输入和分组数
        cpu_channel_shuffle_cl<scalar_t>(output, input, groups);
      });
      break;
    }
    # 处理默认情况，即不支持的内存格式
    default:
      # 抛出错误，提示不支持的内存格式，仅支持ChannelsLast、ChannelsLast3d和Contiguous
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
  }
}

} // anonymous namespace



REGISTER_DISPATCH(channel_shuffle_kernel, &channel_shuffle_kernel_impl);



} // at::native


注释：

// 结束匿名命名空间，这里闭合了之前的匿名命名空间定义
}

// 注册通道重排核心函数，将 channel_shuffle_kernel_impl 函数与 channel_shuffle_kernel 绑定注册
REGISTER_DISPATCH(channel_shuffle_kernel, &channel_shuffle_kernel_impl);

// 结束 at::native 命名空间
} // at::native
```