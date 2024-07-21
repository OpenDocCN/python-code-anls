# `.\pytorch\aten\src\ATen\native\cpu\AdaptiveMaxPoolKernel.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

namespace at::native {

namespace {

// 定义模板函数：对输入的 2D 自适应最大池化进行 CPU 计算
template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_max_pool2d(
    const Tensor& output_,                  // 输出张量（最大池化后的结果）
    const Tensor& indices_,                 // 输出索引张量（记录最大值位置）
    const Tensor& input_,                   // 输入张量（原始数据）
    IntArrayRef output_size) {              // 输出大小的数组

  auto input = input_.contiguous();         // 保证输入张量是连续的
  auto output = output_.contiguous();       // 保证输出张量是连续的
  auto indices = indices_.contiguous();     // 保证索引张量是连续的

  auto input_data = input.const_data_ptr<scalar_t>();     // 获取输入张量的常量数据指针
  auto output_data = output.data_ptr<scalar_t>();         // 获取输出张量的数据指针
  auto indices_data = indices.data_ptr<int64_t>();        // 获取索引张量的数据指针

  int64_t ndim = input.ndimension();                      // 获取输入张量的维度数
  // 将批次大小和通道视为一个维度处理
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);                  // 输入张量的高度
  int64_t input_width = input.size(-1);                   // 输入张量的宽度
  int64_t output_height = output_size[0];                  // 输出的高度
  int64_t output_width = output_size[1];                   // 输出的宽度

  // 在 N、C 的维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + c * input_height * input_width;       // 当前通道的输入数据指针
      scalar_t* output_ptr = output_data + c * output_height * output_width;          // 当前通道的输出数据指针
      int64_t* indices_ptr = indices_data + c * output_height * output_width;         // 当前通道的索引数据指针

      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);       // 计算输出高度范围的起始索引
        int64_t ih1 = end_index(oh, output_height, input_height);         // 计算输出高度范围的结束索引

        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);       // 计算输出宽度范围的起始索引
          int64_t iw1 = end_index(ow, output_width, input_width);         // 计算输出宽度范围的结束索引

          // 计算局部最大值
          int64_t maxindex = ih0 * input_width + iw0;                    // 初始化最大值的索引
          accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();   // 初始化最大值

          for (int64_t ih = ih0; ih < ih1; ih ++) {
            for (int64_t iw = iw0; iw < iw1; iw ++) {
              int64_t index = ih * input_width + iw;                    // 计算当前位置的索引
              scalar_t val = input_ptr[index];                          // 获取当前位置的值
              if ((val > maxval) || std::isnan(val)) {                  // 如果当前值大于最大值或者为 NaN
                maxval = val;                                           // 更新最大值
                maxindex = index;                                       // 更新最大值的索引
              }
            }
          }

          // 将输出设置为局部最大值，并存储最大值的位置索引
          output_ptr[oh * output_width + ow] = maxval;                  // 将最大值写入输出张量
          indices_ptr[oh * output_width + ow] = scalar_t(maxindex);     // 将最大值的索引写入索引张量
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);      // 如果输出张量不是连续的，则复制连续版本的数据到它
  }
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);    // 如果索引张量不是连续的，则复制连续版本的数据到它
  }
}

// 模板特化函数：针对 opmath_type<scalar_t> 类型进行处理
template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_max_pool2d_channels_last(
    const Tensor& output_,        // 输出张量（最大池化后的结果）
    const Tensor& indices_,                          // 输入参数：索引张量的引用
    const Tensor& input_,                            // 输入参数：输入张量的引用
    IntArrayRef output_size) {                       // 输入参数：输出尺寸的引用数组
  TORCH_CHECK(input_.ndimension() == 4,              // 检查输入张量的维度是否为4，如果不是则抛出错误信息
              "2d adaptive max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast; // 设置内存格式为通道最后格式
  auto input = input_.contiguous(memory_format);     // 使输入张量按照指定的内存格式连续化
  auto output = output_.contiguous(memory_format);   // 使输出张量按照指定的内存格式连续化
  auto indices = indices_.contiguous(memory_format); // 使索引张量按照指定的内存格式连续化

  auto input_data = input.const_data_ptr<scalar_t>(); // 获取输入张量的常量数据指针
  auto output_data = output.data_ptr<scalar_t>();     // 获取输出张量的数据指针
  auto indices_data = indices.data_ptr<int64_t>();    // 获取索引张量的数据指针

  int64_t nbatch = input.size(0);                    // 获取输入张量的批次大小
  int64_t channels = input.size(1);                  // 获取输入张量的通道数
  int64_t input_height = input.size(2);              // 获取输入张量的高度
  int64_t input_width = input.size(3);               // 获取输入张量的宽度
  int64_t output_height = output_size[0];            // 获取输出的高度
  int64_t output_width = output_size[1];             // 获取输出的宽度

  using Vec = vec::Vectorized<scalar_t>;              // 使用向量化的数据类型 Vec，根据标量类型确定
  using integer_t = vec::int_same_size_t<scalar_t>;   // 使用与标量类型相同大小的整数类型 integer_t
  using iVec = vec::Vectorized<integer_t>;            // 使用向量化的整数类型 iVec

  // 为了方便向量化，使用与标量类型大小相同的整数类型，
  // 例如，对于 float 使用 int32_t，对于 double 使用 int64_t
  // 需要确保不会溢出

  TORCH_CHECK(input_height * input_width <= std::numeric_limits<integer_t>::max());
  // 检查输入张量的尺寸乘积不超过整数类型 integer_t 的最大值，以防止溢出

  // 在 N、H、W 维度上并行处理
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);
    // 初始化数据索引 n、oh、ow，用于并行处理中的索引计算

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // 计算通道数 size，以及使得 len 为 Vec 大小的整数倍的值

    // 临时缓冲区，用于保存整数类型 integer_t 的索引
    auto index_buffer = std::make_unique<integer_t[]>(len);
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前迭代中的输出行和列的起始和结束索引
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);

      // 计算当前迭代中输出数据和索引的起始位置
      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: 初始化输出向量和索引向量
      iVec index0_vec = iVec(ih0 * input_width + iw0);
      Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
      int64_t d1 = 0;
      // 使用向量化操作填充部分数据
      for (; d1 < len; d1 += Vec::size()) {
        index0_vec.store(index_buffer.get() + d1);
        out_vec.store(out + d1);
      }
      // 处理剩余的非向量化数据
      for (; d1 < size; d1++) {
        ind[d1] = ih0 * input_width + iw0;
        out[d1] = -std::numeric_limits<scalar_t>::infinity();
      }

      // Pass II: 计算局部最大值
      for (int64_t ih = ih0; ih < ih1; ih ++) {
        for (int64_t iw = iw0; iw < iw1; iw ++) {
          // 计算输入数据的起始位置
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          // 使用向量化操作处理部分数据
          for (; d2 < len; d2 += Vec::size()) {
            iVec index_vec = iVec(ih * input_width + iw);
            Vec val_vec = Vec::loadu(in + d2);
            iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
            Vec maxval_vec = Vec::loadu(out + d2);

            // 计算掩码，用于选择最大值
            Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
            iVec imask = vec::cast<integer_t>(mask);
            Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
            iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

            // 存储计算结果
            out_vec.store(out + d2);
            ind_vec.store(index_buffer.get() + d2);
          }
          // 处理剩余的非向量化数据
          for (; d2 < size; d2++) {
            int64_t index = ih * input_width + iw;
            scalar_t val = in[d2];
            int64_t maxindex = ind[d2];
            scalar_t maxval = out[d2];

            // 判断是否更新最大值和索引
            bool mask = (val > maxval) || std::isnan(val);
            out[d2] = mask ? val : maxval;
            ind[d2] = mask ? index : maxindex;
          }
        }
      }

      // 将索引数据类型转换为 integer_t 类型
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // 移动到下一个输出索引
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }

  // 如果输出张量不是按照指定的内存格式连续存储，则进行复制操作
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  // 如果索引张量不是按照指定的内存格式连续存储，则进行复制操作
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
// 结束上一个函数定义的花括号

template <typename scalar_t>
// 使用 SFINAE 技术，当 scalar_t 不是 at::opmath_type<scalar_t> 类型时，该函数无效
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 定义名为 cpu_adaptive_max_pool2d_channels_last 的模板函数
cpu_adaptive_max_pool2d_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  // 检查输入张量 input_ 的维度是否为 4
  TORCH_CHECK(input_.ndimension() == 4,
              "2d adaptive max pooling with channels last format supports tensors with 4 dims");
  // 设置内存格式为 ChannelsLast
  auto memory_format = at::MemoryFormat::ChannelsLast;
  // 使 input_ 张量连续化，并使用 ChannelsLast 内存格式
  auto input = input_.contiguous(memory_format);
  // 使 output_ 张量连续化，并使用 ChannelsLast 内存格式
  auto output = output_.contiguous(memory_format);
  // 使 indices_ 张量连续化，并使用 ChannelsLast 内存格式
  auto indices = indices_.contiguous(memory_format);

  // 获取 input 数据的指针，类型为 scalar_t
  auto input_data = input.const_data_ptr<scalar_t>();
  // 获取 output 数据的指针，类型为 scalar_t
  auto output_data = output.data_ptr<scalar_t>();
  // 获取 indices 数据的指针，类型为 int64_t
  auto indices_data = indices.data_ptr<int64_t>();

  // 获取 input 张量的批次大小
  int64_t nbatch = input.size(0);
  // 获取 input 张量的通道数
  int64_t channels = input.size(1);
  // 获取 input 张量的高度
  int64_t input_height = input.size(2);
  // 获取 input 张量的宽度
  int64_t input_width = input.size(3);
  // 获取输出高度
  int64_t output_height = output_size[0];
  // 获取输出宽度
  int64_t output_width = output_size[1];

  // 使用 Vectorized 类型别名定义 bVec, fVec, iVec 以优化向量化计算
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  using iVec = vec::Vectorized<int32_t>;

  // 检查输入高度和宽度的乘积是否超出 int32_t 的上限
  TORCH_CHECK(input_height * input_width <= std::numeric_limits<int32_t>::max());

  // 使用并行计算在 N, H, W 的维度上进行操作
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化索引变量 n, oh, ow
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 调用 data_index_init 函数进行索引初始化
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    // 定义 size 为通道数
    int64_t size = channels;
    // 计算 len 为 size 对 bVec::size() 取余的结果
    int64_t len = size - (size % bVec::size());
    // 创建存放整数索引的临时缓冲区
    auto index_buffer = std::make_unique<int32_t []>(len);
    // 创建存放最大值的临时缓冲区
    auto max_arr = std::make_unique<float []>(size);
    // 获取 max 指针指向的数组首地址
    float* max = max_arr.get();

    // 此处为并行计算的任务实现代码，未完整显示
  });

  // 如果 output_ 不是使用 memory_format 内存格式存储，则将 output 复制到 output_
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  // 如果 indices_ 不是使用 memory_format 内存格式存储，则将 indices 复制到 indices_
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
// 定义名为 cpu_adaptive_max_pool2d_backward 的模板函数
void cpu_adaptive_max_pool2d_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  // 使 grad_output_ 张量连续化
  auto grad_output = grad_output_.contiguous();
  // 使 indices_ 张量连续化
  auto indices = indices_.contiguous();
  // 使 grad_input_ 张量连续化
  auto grad_input = grad_input_.contiguous();

  // 获取 grad_output 数据的指针，类型为 scalar_t
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  // 获取 indices 数据的指针，类型为 int64_t
  auto indices_data = indices.const_data_ptr<int64_t>();
  // 获取 grad_input 数据的指针，类型为 scalar_t
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取 grad_output 的维度数
  int64_t ndim = grad_output.ndimension();
  // 如果 ndim 为 3，则将通道数和批次大小视为一个维度
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  // 获取 grad_input 的高度
  int64_t input_height = grad_input.size(-2);
  // 获取 grad_input 的宽度
  int64_t input_width = grad_input.size(-1);
  // 获取 grad_output 的高度
  int64_t output_height = grad_output.size(-2);
  // 获取 grad_output 的宽度
  int64_t output_width = grad_output.size(-1);

  // 使用并行计算在 N, C 的维度上进行操作
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 此处为并行计算的任务实现代码，未完整显示
  });
}
    // 遍历输入通道范围内的每个通道
    for (const auto c : c10::irange(begin, end)) {
      // 计算当前通道在梯度输入数据中的起始位置
      scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
      // 获取当前通道在梯度输出数据和索引数据中的起始位置
      const scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;
      const int64_t* indices_ptr = indices_data + c * output_height * output_width;

      // 遍历输出特征图的高度维度
      for (const auto oh : c10::irange(output_height)) {
        // 遍历输出特征图的宽度维度
        for (const auto ow : c10::irange(output_width)) {
          // 计算输出特征图中当前位置的索引
          int64_t index = oh * output_width + ow;
          // 获取最大值索引，用于从梯度输入中找到对应位置
          int64_t maxindex = indices_ptr[index];

          // 更新梯度输入，将梯度输出加到梯度输入的对应最大值索引位置上
          grad_input_ptr[maxindex] += grad_output_ptr[index];
        }
      }
    }
  });

  // 如果梯度输入不是连续存储的，则进行复制操作使其连续
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
void cpu_adaptive_max_pool2d_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& indices_) {
  // 检查输出梯度张量是否为四维，仅支持通道在最后的格式
  TORCH_CHECK(grad_output_.ndimension() == 4,
              "2d adaptive max pooling backward with channels last format supports tensors with 4 dims.");
  
  // 设置内存格式为通道在最后
  auto memory_format = at::MemoryFormat::ChannelsLast;
  // 使用指定内存格式创建连续的梯度输入、梯度输出和索引张量
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  // 获取梯度输入、梯度输出和索引张量的数据指针
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto indices_data = indices.const_data_ptr<int64_t>();

  // 获取张量的维度信息
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // 在 N 维度上并行处理
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      // 计算当前批次中梯度输入、梯度输出和索引的指针位置
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      const scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;
      const int64_t* indices_ptr = indices_data + n * output_height * output_width * channels;

      // 遍历输出的高度和宽度
      for (const auto oh : c10::irange(output_height)) {
        for (const auto ow : c10::irange(output_width)) {
          // 获取当前位置的梯度输出和索引指针
          const scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          const int64_t* ind = indices_ptr + oh * output_width * channels + ow * channels;
          // TODO: gcc 向量化
          // 遍历通道维度，将梯度输出加到梯度输入的最大索引位置
          for (const auto c : c10::irange(channels)) {
            int64_t maxindex = ind[c];
            grad_input_ptr[maxindex * channels + c] += gout[c];
          }
        }
      }
    }
  });

  // 如果梯度输入张量不是连续的，将其拷贝回原始张量
  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void adaptive_max_pool2d_kernel_impl(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    IntArrayRef output_size) {
  // 根据输入张量的推荐内存格式选择执行路径
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 如果内存格式为连续，执行标准的自适应最大池化操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool2d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        cpu_adaptive_max_pool2d<scalar_t, /*accscalar_t*/param_t>(output, indices, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 如果内存格式为通道在最后，执行通道在最后的自适应最大池化操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool2d_channels_last", [&]{
        cpu_adaptive_max_pool2d_channels_last<scalar_t>(output, indices, input, output_size);
      });
      break;
    }
    default:
      # 如果进入了默认情况，表示遇到了不支持的内存格式
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adaptive_max_pool2d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  // 无法根据 grad_output 的内存格式在此处切换，因为 grad_output 可能是 NC11
  switch (grad_input.suggest_memory_format()) {
    // 如果 grad_input 是连续内存格式
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool2d_backward", [&] {
        // 调用 CPU 实现的 adaptive_max_pool2d_backward 函数，使用 scalar_t 类型
        cpu_adaptive_max_pool2d_backward<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    // 如果 grad_input 是通道最后内存格式
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool2d_backward_channels_last", [&]{
        // 调用 CPU 实现的 adaptive_max_pool2d_backward_channels_last 函数，使用 scalar_t 类型
        cpu_adaptive_max_pool2d_backward_channels_last<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    // 如果 grad_input 不是支持的内存格式，则抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_max_pool3d(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t ndim = input.ndimension();
  // 将批量大小和通道视为一个维度处理
  int64_t channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_depth = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // 在 N、C 的维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 对输入张量的每个通道进行迭代
    for (const auto c : c10::irange(begin, end)) {
      // 计算当前通道在输入数据中的起始指针位置
      scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
      // 计算当前通道在输出数据中的起始指针位置
      scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;
      // 计算当前通道在索引数据中的起始指针位置
      int64_t* indices_ptr = indices_data + c * output_depth * output_height * output_width;

      // 对输出张量的深度维度进行迭代
      for (const auto od : c10::irange(output_depth)) {
        // 计算当前深度维度在输入数据中的起始索引
        int64_t id0 = start_index(od, output_depth, input_depth);
        // 计算当前深度维度在输入数据中的结束索引
        int64_t id1 = end_index(od, output_depth, input_depth);
        
        // 对输出张量的高度维度进行迭代
        for (const auto oh : c10::irange(output_height)) {
          // 计算当前高度维度在输入数据中的起始索引
          int64_t ih0 = start_index(oh, output_height, input_height);
          // 计算当前高度维度在输入数据中的结束索引
          int64_t ih1 = end_index(oh, output_height, input_height);

          // 对输出张量的宽度维度进行迭代
          for (const auto ow : c10::irange(output_width)) {
            // 计算当前宽度维度在输入数据中的起始索引
            int64_t iw0 = start_index(ow, output_width, input_width);
            // 计算当前宽度维度在输入数据中的结束索引
            int64_t iw1 = end_index(ow, output_width, input_width);

            // 计算当前窗口内的最大值及其索引
            int64_t maxindex = id0 * input_height * input_width + ih0 * input_width + iw0;
            accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();
            // 遍历当前窗口内的元素
            for (int64_t id = id0; id < id1; id ++) {
              for (int64_t ih = ih0; ih < ih1; ih ++) {
                for (int64_t iw = iw0; iw < iw1; iw ++) {
                  int64_t index = id * input_height * input_width + ih * input_width + iw;
                  scalar_t val = input_ptr[index];
                  // 更新最大值及其索引
                  if ((val > maxval) || std::isnan(val)) {
                    maxval = val;
                    maxindex = index;
                  }
                }
              }
            }

            // 将当前窗口内的局部最大值写入输出张量
            output_ptr[od * output_height * output_width + oh * output_width + ow] = maxval;
            // 将局部最大值的索引写入索引张量
            indices_ptr[od * output_height * output_width + oh * output_width + ow] = scalar_t(maxindex);
          }
        }
      }
    }
  });

  // 如果输出张量不是连续的，则进行复制以保证连续性
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  // 如果索引张量不是连续的，则进行复制以保证连续性
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_max_pool3d_channels_last(
    const Tensor& output_,                                  // 输入参数：输出张量
    const Tensor& indices_,                                 // 输入参数：索引张量
    const Tensor& input_,                                   // 输入参数：输入张量
    IntArrayRef output_size) {                              // 输入参数：输出尺寸数组

  TORCH_CHECK(input_.ndimension() == 5,
              "3d adaptive max pooling with channels last format supports tensors with 5 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast3d;    // 设置内存格式为ChannelsLast3d
  auto input = input_.contiguous(memory_format);           // 将输入张量按照指定内存格式连续化
  auto output = output_.contiguous(memory_format);         // 将输出张量按照指定内存格式连续化
  auto indices = indices_.contiguous(memory_format);       // 将索引张量按照指定内存格式连续化

  auto input_data = input.data_ptr<scalar_t>();            // 获取输入张量数据指针
  auto output_data = output.data_ptr<scalar_t>();          // 获取输出张量数据指针
  auto indices_data = indices.data_ptr<int64_t>();         // 获取索引张量数据指针

  int64_t nbatch = input.size(0);                          // 获取输入张量的批次大小
  int64_t channels = input.size(1);                        // 获取输入张量的通道数
  int64_t input_depth = input.size(2);                     // 获取输入张量的深度
  int64_t input_height = input.size(3);                    // 获取输入张量的高度
  int64_t input_width = input.size(4);                     // 获取输入张量的宽度
  int64_t output_depth = output_size[0];                   // 获取输出尺寸的深度
  int64_t output_height = output_size[1];                  // 获取输出尺寸的高度
  int64_t output_width = output_size[2];                   // 获取输出尺寸的宽度

  using Vec = vec::Vectorized<scalar_t>;                    // 使用向量化类型Vec来处理scalar_t
  using integer_t = vec::int_same_size_t<scalar_t>;         // 定义与scalar_t相同大小的整数类型integer_t
  using iVec = vec::Vectorized<integer_t>;                  // 使用向量化类型iVec来处理integer_t

  // 为了方便向量化，使用与scalar_t相同大小的整数类型，
  // 例如，float对应int32_t，double对应int64_t
  TORCH_CHECK(input_height * input_width <= std::numeric_limits<integer_t>::max());

  // 在N、H、W维度上并行处理
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    // 临时缓冲区，用于存储整数类型的索引
    auto index_buffer = std::make_unique<integer_t []>(len);
    // 遍历从 begin 到 end 的范围内的每个索引 i
    for (const auto i : c10::irange(begin, end)) {
      // 计算起始和结束索引 id0 和 id1，以及对应的高度和宽度 ih0, ih1, iw0, iw1
      int64_t id0 = start_index(od, output_depth, input_depth);
      int64_t id1 = end_index(od, output_depth, input_depth);

      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);

      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);

      // 计算输出数据和索引的起始位置
      scalar_t* out = output_data + i * channels;
      int64_t* ind = indices_data + i * channels;

      // Pass I: 初始化输出向量
      // 设置初始索引向量和输出向量
      iVec index0_vec = iVec(id0 * input_height * input_width + ih0 * input_width + iw0);
      Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
      int64_t d1 = 0;
      // 对于每个 Vec::size 大小的元素块，存储索引和输出值
      for (; d1 < len; d1 += Vec::size()) {
        index0_vec.store(index_buffer.get() + d1);
        out_vec.store(out + d1);
      }
      // 处理余下的元素，单独存储索引和输出值
      for (; d1 < size; d1++) {
        ind[d1] = id0 * input_height * input_width + ih0 * input_width + iw0;
        out[d1] = -std::numeric_limits<scalar_t>::infinity();
      }

      // Pass II: 计算局部最大值
      // 遍历输出深度、高度、宽度的范围，计算每个位置的最大值
      for (int64_t id = id0; id < id1; id ++) {
        for (int64_t ih = ih0; ih < ih1; ih ++) {
          for (int64_t iw = iw0; iw < iw1; iw ++) {
            // 计算输入数据的位置和相关向量的起始位置
            scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                id * input_height * input_width * channels + ih * input_width * channels + iw * channels;

            int64_t d2 = 0;
            // 对于每个 Vec::size 大小的元素块，加载输入数据和相关向量
            for (; d2 < len; d2 += Vec::size()) {
              iVec index_vec = iVec(id * input_height * input_width + ih * input_width + iw);
              Vec val_vec = Vec::loadu(in + d2);
              iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
              Vec maxval_vec = Vec::loadu(out + d2);

              // 创建掩码，用于检查是否要更新最大值
              Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
              iVec imask = vec::cast<integer_t>(mask);
              Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
              iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

              // 存储更新后的输出和索引向量
              out_vec.store(out + d2);
              ind_vec.store(index_buffer.get() + d2);
            }
            // 处理余下的元素，比较并更新最大值和对应索引
            for (; d2 < size; d2++) {
              int64_t index = id * input_height * input_width + ih * input_width + iw;
              scalar_t val = in[d2];
              int64_t maxindex = ind[d2];
              scalar_t maxval = out[d2];

              bool mask = (val > maxval) || std::isnan(val);
              out[d2] = mask ? val : maxval;
              ind[d2] = mask ? index : maxindex;
            }
          }
        }
      }

      // 将 index_buffer 中的数据类型转换为 integer_t，并存储到 ind 中
      vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

      // 移动到下一个输出索引
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  // 检查输出张量是否按照指定的内存格式进行连续存储
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  # 如果输出张量output不是按照指定的内存格式（memory_format）是连续的，则复制输出到output_张量
  if (!indices_.is_contiguous(memory_format)) {
    # 如果索引张量indices不是按照指定的内存格式（memory_format）是连续的，则复制索引到indices_张量
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_max_pool3d_channels_last(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    IntArrayRef output_size) {
  // 检查输入张量维度是否为5，表示3D自适应最大池化与通道末尾格式支持5维张量
  TORCH_CHECK(input_.ndimension() == 5,
              "3d adaptive max pooling with channels last format supports tensors with 5 dims");
  // 设定内存格式为ChannelsLast3d
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  // 使用指定的内存格式创建连续的输入、输出和索引张量
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  // 获取输入、输出和索引张量的数据指针
  auto input_data = input.data_ptr<BFloat16>();
  auto output_data = output.data_ptr<BFloat16>();
  auto indices_data = indices.data_ptr<int64_t>();

  // 获取张量的各维度大小信息
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // 定义向量化类型别名
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  using iVec = vec::Vectorized<int32_t>;
  // 确保乘积不会溢出
  TORCH_CHECK(input_height * input_width <= std::numeric_limits<int32_t>::max());

  // 在 N、H、W 维度上进行并行操作
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化数据索引起始值
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    // 计算需要处理的通道数和向量化长度
    int64_t size = channels;
    int64_t len = size - (size % bVec::size());
    // 创建存放索引的临时缓冲区
    auto index_buffer = std::make_unique<int32_t []>(len);
    // 创建存放最大值的临时缓冲区
    auto max_arr = std::make_unique<float []>(size);
    float* max = max_arr.get();

    }
  });

  // 如果输出张量不是连续的，则复制连续的数据到输出张量
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  // 如果索引张量不是连续的，则复制连续的数据到索引张量
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}
    // 根据输入的张量 indices_ 获取其梯度 grad_output_，并保证是连续的
    const Tensor& indices_) {
  // 将 grad_output_ 张量变为连续存储
  auto grad_output = grad_output_.contiguous();
  // 将 indices_ 张量变为连续存储
  auto indices = indices_.contiguous();
  // 将 grad_input_ 张量变为连续存储
  auto grad_input = grad_input_.contiguous();

  // 获取 grad_output 的数据指针，指定数据类型为 scalar_t
  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  // 获取 indices 的数据指针，指定数据类型为 int64_t
  auto indices_data = indices.data_ptr<int64_t>();
  // 获取 grad_input 的可变数据指针，指定数据类型为 scalar_t
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取 grad_output 的维度数
  int64_t ndim = grad_output.ndimension();
  // 如果 ndim 为 3，则将通道数 channels 设置为 grad_output 的第一维大小，否则计算总通道数
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  // 获取 grad_input 的深度、高度、宽度
  int64_t input_depth = grad_input.size(-3);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  // 获取 grad_output 的深度、高度、宽度
  int64_t output_depth = grad_output.size(-3);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N、C 维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 对于每个通道 c
    for (const auto c : c10::irange(begin, end)) {
      // 获取当前通道对应的 grad_input_ptr、grad_output_ptr 和 indices_ptr
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;
      int64_t* indices_ptr = indices_data + c * output_depth * output_height * output_width;

      // 遍历输出张量的深度、高度、宽度
      for (const auto od : c10::irange(output_depth)) {
        for (const auto oh : c10::irange(output_height)) {
          for (const auto ow : c10::irange(output_width)) {
            // 计算在展平后的一维数组中的索引
            int64_t index = od * output_height * output_width + oh * output_width + ow;
            // 获取对应位置的最大值索引
            int64_t maxindex = indices_ptr[index];

            // 更新 grad_input 对应位置的梯度
            grad_input_ptr[maxindex] += grad_output_ptr[index];
          }
        }
      }
    }
  });

  // 如果 grad_input_ 不是连续的，则将其内容复制为 grad_input 的内容
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
// 定义了 CPU 上处理 channels last 格式 3D 自适应最大池化反向传播的函数
void cpu_adaptive_max_pool3d_backward_channels_last(
    const Tensor& grad_input_,   // 梯度输入张量
    const Tensor& grad_output_,  // 梯度输出张量
    const Tensor& indices_) {    // 池化索引张量
  TORCH_CHECK(grad_output_.ndimension() == 5,
              "3d adaptive max pooling backward with channels last format supports tensors with 5 dims.");
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto grad_input = grad_input_.contiguous(memory_format);  // 保证梯度输入张量是连续的，按照 channels last 格式
  auto grad_output = grad_output_.contiguous(memory_format);  // 保证梯度输出张量是连续的，按照 channels last 格式
  auto indices = indices_.contiguous(memory_format);  // 保证池化索引张量是连续的，按照 channels last 格式

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();  // 获取可变的梯度输入数据指针
  auto grad_output_data = grad_output.data_ptr<scalar_t>();        // 获取梯度输出数据指针
  auto indices_data = indices.data_ptr<int64_t>();                 // 获取池化索引数据指针

  int64_t nbatch = grad_input.size(0);         // 获取批次大小
  int64_t channels = grad_input.size(1);       // 获取通道数
  int64_t input_depth = grad_input.size(2);    // 获取输入深度
  int64_t input_height = grad_input.size(3);   // 获取输入高度
  int64_t input_width = grad_input.size(4);    // 获取输入宽度
  int64_t output_depth = grad_output.size(2);  // 获取输出深度
  int64_t output_height = grad_output.size(3); // 获取输出高度
  int64_t output_width = grad_output.size(4);  // 获取输出宽度

  // 在 N 维度上并行处理
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;  // 计算当前批次的梯度输入指针
      scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;  // 计算当前批次的梯度输出指针
      int64_t* indices_ptr = indices_data + n * output_depth * output_height * output_width * channels;  // 计算当前批次的池化索引指针

      // 遍历输出体积的每个点
      for (const auto od : c10::irange(output_depth)) {
        for (const auto oh : c10::irange(output_height)) {
          for (const auto ow : c10::irange(output_width)) {
            // 计算当前位置在梯度输出和池化索引中的偏移量
            scalar_t* gout = grad_output_ptr + od * output_height * output_width * channels + oh * output_width * channels + ow * channels;
            int64_t* ind = indices_ptr + od * output_height * output_width * channels + oh * output_width * channels + ow * channels;

            // TODO: gcc vectorization
            // 对每个通道进行反向传播计算
            for (const auto c : c10::irange(channels)) {
              int64_t maxindex = ind[c];  // 获取池化索引中的最大值索引
              grad_input_ptr[maxindex * channels + c] += gout[c];  // 累加梯度输入中对应最大值索引的梯度
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);  // 如果梯度输入不是按 channels last 格式连续的，则复制重新排列后的梯度输入
  }
}

// 实现 3D 自适应最大池化的内核函数
void adaptive_max_pool3d_kernel_impl(
    const Tensor& output,    // 输出张量
    const Tensor& indices,   // 池化索引张量
    const Tensor& input,     // 输入张量
    IntArrayRef output_size) {  // 输出尺寸
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 分发函数，根据输入张量的类型调用适当的处理函数
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool3d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        cpu_adaptive_max_pool3d<scalar_t, /*accscalar_t*/param_t>(output, indices, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      // 处理内存格式为 ChannelsLast3d 的情况
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_max_pool3d_channels_last", [&]{
        // 使用 Lambda 表达式，根据输入张量的数据类型，调用 CPU 上的自适应最大池化函数
        cpu_adaptive_max_pool3d_channels_last<scalar_t>(output, indices, input, output_size);
      });
      // 结束当前 case 分支
      break;
    }
    default:
      // 如果不支持当前内存格式，抛出错误信息
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
// 定义一个匿名命名空间，用于限定此处代码的作用域，避免命名冲突
void adaptive_max_pool3d_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& indices) {
  // 不能依据 grad_output 的内存格式来切换，因为 grad_output 可能是 NC11 格式
  switch (grad_input.suggest_memory_format()) {
    // 如果 grad_input 推荐使用连续内存格式
    case at::MemoryFormat::Contiguous: {
      // 根据 grad_output 的数据类型进行分发，执行 adaptive_max_pool3d_backward 函数
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool3d_backward", [&] {
        cpu_adaptive_max_pool3d_backward<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    // 如果 grad_input 推荐使用 ChannelsLast3d 内存格式
    case at::MemoryFormat::ChannelsLast3d: {
      // 根据 grad_output 的数据类型进行分发，执行 adaptive_max_pool3d_backward_channels_last 函数
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_max_pool3d_backward_channels_last", [&]{
        cpu_adaptive_max_pool3d_backward_channels_last<scalar_t>(grad_input, grad_output, indices);
      });
      break;
    }
    // 如果推荐的内存格式既不是连续的也不是 ChannelsLast3d
    default:
      // 抛出错误，提示不支持的内存格式，仅支持 ChannelsLast 和连续格式
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 结束匿名命名空间

// 注册 adaptive_max_pool2d_kernel 的分发函数实现
REGISTER_DISPATCH(adaptive_max_pool2d_kernel, &adaptive_max_pool2d_kernel_impl);
// 注册 adaptive_max_pool2d_backward_kernel 的分发函数实现
REGISTER_DISPATCH(adaptive_max_pool2d_backward_kernel, &adaptive_max_pool2d_backward_kernel_impl);
// 注册 adaptive_max_pool3d_kernel 的分发函数实现
REGISTER_DISPATCH(adaptive_max_pool3d_kernel, &adaptive_max_pool3d_kernel_impl);
// 注册 adaptive_max_pool3d_backward_kernel 的分发函数实现
REGISTER_DISPATCH(adaptive_max_pool3d_backward_kernel, &adaptive_max_pool3d_backward_kernel_impl);

// 结束 at::native 命名空间
```