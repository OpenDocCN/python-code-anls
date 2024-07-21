# `.\pytorch\aten\src\ATen\native\cpu\AdaptiveAvgPoolKernel.cpp`

```
# 定义一个宏，用于在编译时仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

# 包含 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>

# 包含 ATen 库中的调度相关功能
#include <ATen/Dispatch.h>

# 包含 ATen 库中的自适应池化函数定义
#include <ATen/native/AdaptivePooling.h>

# 包含 ATen 库中的并行计算支持
#include <ATen/Parallel.h>

# 包含 ATen 库中的向量化功能
#include <ATen/cpu/vec/vec.h>

# 包含 ATen 库中的向量化功能的函数定义
#include <ATen/cpu/vec/functional.h>

# 包含 ATen 库中的 CPU 相关实用函数
#include <ATen/native/cpu/utils.h>

# 包含 C10 库中的整数范围函数
#include <c10/util/irange.h>

# 包含 ATen 库中的操作数数学类型定义
#include <ATen/OpMathType.h>

# 定义 ATen 库中 native 命名空间
namespace at::native {

# 匿名命名空间，用于定义局部函数或变量
namespace {

# 定义 CPU 上的自适应平均池化函数，使用模板类型 scalar_t 和 accscalar_t
template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool2d(
    Tensor& output_,                     // 输出 Tensor
    const Tensor& input_,                // 输入 Tensor
    IntArrayRef output_size) {           // 输出大小的整数数组引用

  auto input = input_.contiguous();      // 使输入 Tensor 连续存储
  auto output = output_.contiguous();    // 使输出 Tensor 连续存储

  auto input_data = input.const_data_ptr<scalar_t>();    // 获取输入数据指针
  auto output_data = output.data_ptr<scalar_t>();         // 获取输出数据指针

  int64_t ndim = input.ndimension();      // 获取输入 Tensor 的维度数
  // 将批次大小和通道视为一个维度
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);  // 获取输入 Tensor 的高度维度大小
  int64_t input_width = input.size(-1);   // 获取输入 Tensor 的宽度维度大小
  int64_t output_height = output_size[0]; // 获取输出 Tensor 的高度维度大小
  int64_t output_width = output_size[1];   // 获取输出 Tensor 的宽度维度大小

  // 在 N、C 维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      const scalar_t* input_ptr = input_data + c * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_height * output_width;

      // 遍历输出 Tensor 的高度维度
      for (const auto oh : c10::irange(output_height)) {
        int64_t ih0 = start_index(oh, output_height, input_height);  // 计算输入索引起始位置
        int64_t ih1 = end_index(oh, output_height, input_height);    // 计算输入索引结束位置
        int64_t kh = ih1 - ih0;  // 计算高度维度的长度

        // 遍历输出 Tensor 的宽度维度
        for (const auto ow : c10::irange(output_width)) {
          int64_t iw0 = start_index(ow, output_width, input_width);   // 计算输入索引起始位置
          int64_t iw1 = end_index(ow, output_width, input_width);     // 计算输入索引结束位置
          int64_t kw = iw1 - iw0;  // 计算宽度维度的长度

          // 计算局部平均值
          accscalar_t sum = 0;
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              sum += accscalar_t(input_ptr[ih * input_width + iw]);
            }
          }
          output_ptr[oh * output_width + ow] = scalar_t(sum / kh / kw);  // 存储平均值到输出 Tensor
        }
      }
    }
  });

  // 如果输出 Tensor 不是连续存储，进行复制
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

# 定义模板特化，针对 scalar_t 类型为 at::opmath_type<scalar_t> 的情况
template <typename scalar_t>
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
cpu_adaptive_avg_pool2d_channels_last(
    Tensor& output_,                     // 输出 Tensor
    const Tensor& input_,                // 输入 Tensor
  IntArrayRef output_size) {
  // 设置内存格式为通道在最后
  auto memory_format = at::MemoryFormat::ChannelsLast;
  // 将输入张量转换为连续内存格式，使用指定的内存格式
  auto input = input_.contiguous(memory_format);
  // 将输出张量转换为连续内存格式，使用指定的内存格式
  auto output = output_.contiguous(memory_format);

  // 获取输入数据的常量指针
  auto input_data = input.const_data_ptr<scalar_t>();
  // 获取输出数据的指针
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入张量的批次大小、通道数、输入高度和宽度
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  // 获取输出的高度和宽度
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // 使用 Vectorized 类别进行向量化计算
  using Vec = vec::Vectorized<scalar_t>;
  // 在 N、H、W 维度上并行执行操作
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 初始化数据索引
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    // 遍历输出张量的每个元素
    for (const auto i : c10::irange(begin, end)) {
      // 计算输入张量在高度方向上的起始和结束索引
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      // 计算输入张量在宽度方向上的起始和结束索引
      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      // 获取输出张量的起始地址
      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // 注意: 对于普通的使用场景，每个输出通道应该适合 L1 缓存；否则考虑块维度 C。
      // 第一遍: 将输出通道置零
      int64_t d1 = 0;
      for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }

      // 第二遍: 计算局部和
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          // 获取输入张量的起始地址
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
            out_vec.store(out + d2);
          }
          for (; d2 < size; d2++) {
            out[d2] += in[d2];
          }
        }
      }

      // 第三遍: 计算局部平均值
      int64_t d3 = 0;
      for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kh * kw));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / kh / kw;
      }

      // 移动到下一个输出索引
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量不是连续的指定内存格式，则复制输出数据到输出张量
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
    // 结束上一行并开始一个新的代码块，定义一个模板特化函数用于 scalar_t 类型不是 at::opmath_type<scalar_t> 的情况，返回 void
    template <typename scalar_t>
    typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
    // 函数定义，对 channels_last 内存格式进行自适应平均池化操作
    cpu_adaptive_avg_pool2d_channels_last(
        // 输出张量的引用
        Tensor& output_,
        // 输入张量的常量引用
        const Tensor& input_,
        // 输出尺寸的整数数组引用
        IntArrayRef output_size) {
      // 设定内存格式为 channels_last
      auto memory_format = at::MemoryFormat::ChannelsLast;
      // 使用 channels_last 内存格式对输入张量进行连续性处理
      auto input = input_.contiguous(memory_format);
      // 使用 channels_last 内存格式对输出张量进行连续性处理
      auto output = output_.contiguous(memory_format);

      // 获取输入数据的常量指针，类型为 scalar_t
      auto input_data = input.const_data_ptr<scalar_t>();
      // 获取输出数据的指针，类型为 scalar_t
      auto output_data = output.data_ptr<scalar_t>();

      // 获取输入张量的批量大小
      int64_t nbatch = input.size(0);
      // 获取输入张量的通道数
      int64_t channels = input.size(1);
      // 获取输入张量的高度
      int64_t input_height = input.size(2);
      // 获取输入张量的宽度
      int64_t input_width = input.size(3);
      // 获取输出张量的高度
      int64_t output_height = output_size[0];
      // 获取输出张量的宽度
      int64_t output_width = output_size[1];

      // 使用 scalar_t 类型的向量化数据类型别名 bVec
      using bVec = vec::Vectorized<scalar_t>;
      // 使用 float 类型的向量化数据类型别名 fVec
      using fVec = vec::Vectorized<float>;
      // 在 N、H、W 维度上并行执行
      // 使用 parallel_for 函数，范围为 0 到 nbatch * output_height * output_width
      at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
        // 初始化索引
        int64_t n = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        // 初始化数据索引
        data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

        // 用于存储求和的临时缓冲区，使用 float 作为累积类型
        // 不能重用输出缓冲区来存储求和，因为它可能是 BFloat16 或 Half 类型
        auto sum_arr = std::make_unique<float []>(channels);
        // 指向 sum_arr 的指针，用于存储求和结果
        float* sum = sum_arr.get();
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前输出行的起始和结束索引，以及高度
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;

      // 计算当前输出列的起始和结束索引，以及宽度
      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;

      // 计算当前输出索引对应的输出数据的起始地址
      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Pass I: 将输出通道中的数据清零
      int64_t d1 = 0;
      for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
        // 使用 fVec 类型的向量清零
        fVec sum_fvec = fVec(float(0));
        sum_fvec.store(sum + d1);
      }
      // 处理剩余不足一个向量大小的数据
      for (; d1 < size; d1++) {
        sum[d1] = float(0);
      }

      // Pass II: 计算局部和
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          // 计算输入数据的起始地址
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          // 处理每个通道的数据
          int64_t d2 = 0;
          for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
            // 加载并转换为浮点数向量
            bVec data_bvec = bVec::loadu(in + d2);
            fVec data_fvec0, data_fvec1;
            std::tie(data_fvec0, data_fvec1) = convert_to_float<scalar_t>(data_bvec);

            // 加载累加和向量，并加上输入数据向量
            fVec sum_fvec0 = fVec::loadu(sum + d2) + data_fvec0;
            fVec sum_fvec1 = fVec::loadu(sum + d2 + fVec::size()) + data_fvec1;
            // 存储累加和向量
            sum_fvec0.store(sum + d2);
            sum_fvec1.store(sum + d2 + fVec::size());
          }
          // 处理剩余不足一个向量大小的数据
          for (; d2 < size; d2++) {
            sum[d2] += float(in[d2]);
          }
        }
      }

      // Pass III: 计算局部平均值
      int64_t d3 = 0;
      for (; d3 < size - (size % bVec::size()); d3 += bVec::size()) {
        // 计算每个通道的局部平均值，并存储到输出向量
        fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(kh * kw));
        fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) / fVec(float(kh * kw));
        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(out + d3);
      }
      // 处理剩余不足一个向量大小的数据
      for (; d3 < size; d3++) {
        out[d3] = scalar_t(sum[d3] / kh / kw);
      }

      // 移动到下一个输出索引
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  // 如果输出不是连续存储的，则进行复制操作
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
// 结束 CPU 自适应平均池化层的反向传播函数的声明
template <typename scalar_t>
void cpu_adaptive_avg_pool2d_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  // 强制复制输入的梯度张量为连续的
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取梯度张量的数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取梯度张量的维度数
  int64_t ndim = grad_output.ndimension();
  // 将批量大小和通道数视为一个维度处理
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N、C 维度上并行执行
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      // 计算当前通道的梯度输入和梯度输出的指针
      scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
      const scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;

      // 对输出高度上的每个元素进行迭代
      for (const auto oh : c10::irange(output_height)) {
        // 计算输入高度的起始和结束索引
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        // 对输出宽度上的每个元素进行迭代
        for (const auto ow : c10::irange(output_width)) {
          // 计算输入宽度的起始和结束索引
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          // 计算当前输出元素对应的梯度增量
          scalar_t grad_delta = grad_output_ptr[oh * output_width + ow] / kh / kw;

          // 在输入张量的相应位置上累加梯度增量
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              grad_input_ptr[ih * input_width + iw] += grad_delta;
            }
          }
        }
      }
    }
  });

  // 如果输出的梯度张量不是连续的，则复制连续的数据到原始张量
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

// 开始使用通道最后内存格式的 CPU 自适应平均池化层的反向传播函数的声明
template <typename scalar_t>
void cpu_adaptive_avg_pool2d_backward_channels_last(
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  // 设置内存格式为通道最后
  auto memory_format = at::MemoryFormat::ChannelsLast;
  // 强制复制输入的梯度张量为通道最后内存格式的连续的
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  // 获取梯度张量的数据指针
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();

  // 获取张量的维度信息
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // 使用通道最后内存格式并行执行在 N 维度上的操作
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    // 对于输入数据的每个索引 n 在范围 [begin, end) 内进行循环
    for (const auto n : c10::irange(begin, end)) {
      // 计算当前输入数据对应的梯度输入指针位置
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
      // 计算当前输入数据对应的梯度输出指针位置
      const scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

      // 对于输出的每一行 oh 在 output_height 范围内进行循环
      for (const auto oh : c10::irange(output_height)) {
        // 计算输入的起始和结束索引 ih0 和 ih1
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0; // 计算高度上的尺寸

        // 对于输出的每一列 ow 在 output_width 范围内进行循环
        for (const auto ow : c10::irange(output_width)) {
          // 计算输入的起始和结束索引 iw0 和 iw1
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0; // 计算宽度上的尺寸

          // 计算当前梯度输出指针位置 gout
          const scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t size = channels; // 计算通道数

          // 对于输入的每一行 ih 在范围 [ih0, ih1) 内进行循环
          for (const auto ih : c10::irange(ih0, ih1)) {
            // 对于输入的每一列 iw 在范围 [iw0, iw1) 内进行循环
            for (const auto iw : c10::irange(iw0, iw1)) {
              // 计算当前梯度输入指针位置 gin
              scalar_t* gin = grad_input_ptr + ih * input_width * channels + iw * channels;

              int64_t d = 0;
              // 使用 SIMD 加速计算，每次处理 Vec::size() 个元素
              for (; d < size - (size % Vec::size()); d += Vec::size()) {
                // 加载当前位置的数据到 SIMD 向量 gin_vec，并执行计算
                Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(kh * kw));
                // 将计算结果存储回原位置
                gin_vec.store(gin + d);
              }
              // 处理剩余不足 Vec::size() 的部分
              for (; d < size; d++) {
                // 执行普通的非 SIMD 计算
                gin[d] += gout[d] / kh / kw;
              }
            }
          }
        }
      }
    }
  });

  // 如果梯度输入不是按照给定的内存格式（memory_format）进行连续存储
  if (!grad_input_.is_contiguous(memory_format)) {
    // 则通过拷贝将 grad_input_ 调整为按照 memory_format 连续存储的形式
    grad_input_.copy_(grad_input);
  }
}

// 定义了一个函数，实现了自适应平均池化操作
void adaptive_avg_pool2d_kernel_impl(
    Tensor& output, // 输出张量，存储池化后的结果
    const Tensor& input, // 输入张量，进行池化的原始数据
    IntArrayRef output_size) { // 池化后的输出大小

  // 根据输入张量的内存格式选择不同的处理方式
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: { // 如果是连续内存格式
      // 使用宏展开，处理浮点类型和半精度类型的数据
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool2d", [&] {
        using param_t = at::opmath_type<scalar_t>;
        // 调用CPU实现的自适应平均池化函数
        cpu_adaptive_avg_pool2d<scalar_t, /*accscalar_t*/param_t>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: { // 如果是通道最后的内存格式
      // 使用宏展开，处理浮点类型和半精度类型的数据
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool2d_channels_last", [&]{
        // 调用CPU实现的通道最后内存格式的自适应平均池化函数
        cpu_adaptive_avg_pool2d_channels_last<scalar_t>(output, input, output_size);
      });
      break;
    }
    default:
      // 如果内存格式不支持，抛出异常
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 定义了一个函数，实现了自适应平均池化的反向传播操作
void adapative_avg_pool2d_backward_kernel_impl(
    Tensor& grad_input, // 输入梯度张量，用于接收反向传播的梯度
    const Tensor& grad_output) { // 输出梯度张量，反向传播的梯度来源

  // 根据输出梯度张量的内存格式选择不同的处理方式
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: { // 如果是连续内存格式
      // 使用宏展开，处理浮点类型和半精度类型的数据
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool2d_backward", [&] {
        // 调用CPU实现的自适应平均池化反向传播函数
        cpu_adaptive_avg_pool2d_backward<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: { // 如果是通道最后的内存格式
      // 使用宏展开，处理浮点类型和半精度类型的数据
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool2d_backward_channels_last", [&]{
        // 调用CPU实现的通道最后内存格式的自适应平均池化反向传播函数
        cpu_adaptive_avg_pool2d_backward_channels_last<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    default:
      // 如果内存格式不支持，抛出异常
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 定义了一个模板函数，实现了自适应平均池化的三维操作
template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool3d(
    Tensor& output_, // 输出张量，存储池化后的结果
    const Tensor& input_, // 输入张量，进行池化的原始数据
    IntArrayRef output_size) { // 池化后的输出大小

  auto input = input_.contiguous(); // 将输入张量转为连续内存格式
  auto output = output_.contiguous(); // 将输出张量转为连续内存格式

  auto input_data = input.data_ptr<scalar_t>(); // 获取输入张量数据指针
  auto output_data = output.data_ptr<scalar_t>(); // 获取输出张量数据指针

  int64_t ndim = input.ndimension(); // 获取输入张量的维度数
  // 将批处理大小和通道数看作一个维度处理
  int64_t channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_depth = input.size(-3); // 输入张量的深度维度大小
  int64_t input_height = input.size(-2); // 输入张量的高度维度大小
  int64_t input_width = input.size(-1); // 输入张量的宽度维度大小
  int64_t output_depth = output_size[0]; // 输出张量的深度维度大小
  int64_t output_height = output_size[1]; // 输出张量的高度维度大小
  int64_t output_width = output_size[2]; // 输出张量的宽度维度大小

  // 在N和C维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 遍历输入张量的通道维度
    for (const auto c : c10::irange(begin, end)) {
        // 计算当前通道在输入数据中的起始指针位置
        scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;
        // 计算当前通道在输出数据中的起始指针位置
        scalar_t* output_ptr = output_data + c * output_depth * output_height * output_width;

        // 遍历输出张量的深度维度
        for (const auto od : c10::irange(output_depth)) {
            // 计算当前输出深度对应的输入深度的起始和结束索引
            int64_t id0 = start_index(od, output_depth, input_depth);
            int64_t id1 = end_index(od, output_depth, input_depth);
            int64_t kd = id1 - id0;

            // 遍历输出张量的高度维度
            for (const auto oh : c10::irange(output_height)) {
                // 计算当前输出高度对应的输入高度的起始和结束索引
                int64_t ih0 = start_index(oh, output_height, input_height);
                int64_t ih1 = end_index(oh, output_height, input_height);
                int64_t kh = ih1 - ih0;

                // 遍历输出张量的宽度维度
                for (const auto ow : c10::irange(output_width)) {
                    // 计算当前输出宽度对应的输入宽度的起始和结束索引
                    int64_t iw0 = start_index(ow, output_width, input_width);
                    int64_t iw1 = end_index(ow, output_width, input_width);
                    int64_t kw = iw1 - iw0;

                    // 计算局部平均值
                    accscalar_t sum = 0;
                    // 遍历输入张量的深度、高度、宽度维度，累加输入数据
                    for (const auto id : c10::irange(id0, id1)) {
                        for (const auto ih : c10::irange(ih0, ih1)) {
                            for (const auto iw : c10::irange(iw0, iw1)) {
                                sum += accscalar_t(input_ptr[id * input_height * input_width + ih * input_width + iw]);
                            }
                        }
                    }
                    // 计算并存储输出张量中的值，即平均值
                    output_ptr[od * output_height * output_width + oh * output_width + ow] = scalar_t(sum / kd / kh / kw);
                }
            }
        }
    }
  });

  // 如果输出张量不是连续存储的，则进行拷贝操作使其变为连续存储
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  // 结束 CPU 自适应平均池化函数的定义，使用模板类型 scalar_t
template <typename scalar_t>
  // 如果 scalar_t 和 at::opmath_type<scalar_t> 相同，则返回 void
typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
  // 定义函数 cpu_adaptive_avg_pool3d_channels_last，处理输出张量、输入张量和输出大小
cpu_adaptive_avg_pool3d_channels_last(
    // 输出张量的引用
    Tensor& output_,
    // 输入张量的常量引用
    const Tensor& input_,
    // 输出大小的整数数组引用
    IntArrayRef output_size) {
  // 内存格式设置为 ChannelsLast3d
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  // 使用 ChannelsLast3d 内存格式对输入张量进行连续化处理
  auto input = input_.contiguous(memory_format);
  // 使用 ChannelsLast3d 内存格式对输出张量进行连续化处理
  auto output = output_.contiguous(memory_format);

  // 获取输入张量数据的指针，并转换为 scalar_t 类型
  auto input_data = input.data_ptr<scalar_t>();
  // 获取输出张量数据的指针，并转换为 scalar_t 类型
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入张量的批次大小、通道数、深度、高度和宽度
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  // 获取输出张量的深度、高度和宽度
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  // 使用 Vec 类型别名定义向量化操作器，以便在 N、H、W 维度上并行执行
  using Vec = vec::Vectorized<scalar_t>;
  // 在 N、H、W 维度上并行执行的 lambda 函数
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化数据索引，以便在并行执行期间处理 N、H、W 维度
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 调用 data_index_init 函数，初始化数据索引的起始值
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    for (const auto i : c10::irange(begin, end)) {
        // 获取当前循环迭代的索引 i
        int64_t id0 = start_index(od, output_depth, input_depth);
        // 计算起始深度索引 id0
        int64_t id1 = end_index(od, output_depth, input_depth);
        // 计算结束深度索引 id1
        int64_t kd = id1 - id0;
        // 计算深度维度大小 kd

        int64_t ih0 = start_index(oh, output_height, input_height);
        // 计算起始高度索引 ih0
        int64_t ih1 = end_index(oh, output_height, input_height);
        // 计算结束高度索引 ih1
        int64_t kh = ih1 - ih0;
        // 计算高度维度大小 kh

        int64_t iw0 = start_index(ow, output_width, input_width);
        // 计算起始宽度索引 iw0
        int64_t iw1 = end_index(ow, output_width, input_width);
        // 计算结束宽度索引 iw1
        int64_t kw = iw1 - iw0;
        // 计算宽度维度大小 kw

        scalar_t* out = output_data + i * channels;
        // 获取输出数据的起始地址，考虑偏移量 i * channels
        int64_t size = channels;
        // 确定通道数目 size

        // Note: For ordinary usage scenario, each out lane should
        //   fit in L1 cache; otherwise consider block dim C.
        // Pass I: zero the out lane
        int64_t d1 = 0;
        // 初始化循环变量 d1
        for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
            // 以向量化方式处理数据，每次处理 Vec::size() 个元素
            Vec out_vec = Vec(scalar_t(0));
            // 创建全零向量 out_vec
            out_vec.store(out + d1);
            // 将 out_vec 存储到 out 中的偏移位置 d1
        }
        for (; d1 < size; d1++) {
            // 处理剩余不足 Vec::size() 的元素
            out[d1] = scalar_t(0);
            // 将 out 中剩余位置置零
        }
        // Pass II: compute local sum
        for (const auto id : c10::irange(id0, id1)) {
            // 遍历深度范围 [id0, id1)
            for (const auto ih : c10::irange(ih0, ih1)) {
                // 遍历高度范围 [ih0, ih1)
                for (const auto iw : c10::irange(iw0, iw1)) {
                    // 遍历宽度范围 [iw0, iw1)
                    scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                        id * input_height * input_width * channels + ih * input_width * channels + iw * channels;
                    // 获取输入数据的地址，考虑输入数据的多维索引

                    int64_t d2 = 0;
                    // 初始化循环变量 d2
                    for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
                        // 以向量化方式处理数据，每次处理 Vec::size() 个元素
                        Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
                        // 加载 out 和 in 的向量数据，执行加法操作
                        out_vec.store(out + d2);
                        // 将结果存储到 out 中的偏移位置 d2
                    }
                    for (; d2 < size; d2++) {
                        // 处理剩余不足 Vec::size() 的元素
                        out[d2] += in[d2];
                        // 执行逐元素加法
                    }
                }
            }
        }
        // Pass III: compute local average
        int64_t d3 = 0;
        // 初始化循环变量 d3
        for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
            // 以向量化方式处理数据，每次处理 Vec::size() 个元素
            Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kd * kh * kw));
            // 加载 out 的向量数据，执行除法操作
            out_vec.store(out + d3);
            // 将结果存储到 out 中的偏移位置 d3
        }
        for (; d3 < size; d3++) {
            // 处理剩余不足 Vec::size() 的元素
            out[d3] = out[d3] / kd / kh / kw;
            // 执行逐元素平均操作
        }

        // move on to next output index
        // 移动到下一个输出索引位置
        data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
});

if (!output_.is_contiguous(memory_format)) {
    // 检查输出是否按照给定的内存格式连续
    output_.copy_(output);
    // 如果不连续，执行数据拷贝操作
}
// 结束模板函数的定义，模板函数为了处理不同类型的数据进行了特化
template <typename scalar_t>
// 如果输入数据类型不是 at::opmath_type<scalar_t>，则函数返回 void
typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
// 函数名称，计算 CPU 上的自适应平均池化，数据格式为 ChannelsLast3d
cpu_adaptive_avg_pool3d_channels_last(
    // 输出 Tensor 的引用
    Tensor& output_,
    // 输入 Tensor 的常量引用
    const Tensor& input_,
    // 输出大小的引用
    IntArrayRef output_size) {
  // 设定内存格式为 ChannelsLast3d
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  // 对输入进行连续性处理，以指定的内存格式存储
  auto input = input_.contiguous(memory_format);
  // 对输出进行连续性处理，以指定的内存格式存储
  auto output = output_.contiguous(memory_format);

  // 获取输入数据的指针，指向 scalar_t 类型的数据
  auto input_data = input.data_ptr<scalar_t>();
  // 获取输出数据的指针，指向 scalar_t 类型的数据
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入 Tensor 的批次大小
  int64_t nbatch = input.size(0);
  // 获取输入 Tensor 的通道数
  int64_t channels = input.size(1);
  // 获取输入 Tensor 的深度（第三维度）大小
  int64_t input_depth = input.size(2);
  // 获取输入 Tensor 的高度（第四维度）大小
  int64_t input_height = input.size(3);
  // 获取输入 Tensor 的宽度（第五维度）大小
  int64_t input_width = input.size(4);
  // 获取输出 Tensor 的深度大小
  int64_t output_depth = output_size[0];
  // 获取输出 Tensor 的高度大小
  int64_t output_height = output_size[1];
  // 获取输出 Tensor 的宽度大小
  int64_t output_width = output_size[2];

  // 使用标量类型 scalar_t 的向量化操作
  using bVec = vec::Vectorized<scalar_t>;
  // 使用浮点类型 float 的向量化操作
  using fVec = vec::Vectorized<float>;

  // 并行处理维度 N, D, H, W
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化索引变量
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    int64_t od = 0;
    // 初始化数据索引，根据 begin 和 end 确定处理的范围
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    // 用于临时存储求和结果的缓冲区，使用 float 作为累加类型
    // 不能重用输出缓冲区存储求和结果，因为它可能是 BFloat16 或 Half 类型
    auto sum_arr = std::make_unique<float []>(channels);
    // 指向求和缓冲区的指针
    float* sum = sum_arr.get();
    // 对于给定的范围 [begin, end)，迭代每个索引 i
    for (const auto i : c10::irange(begin, end)) {
      // 计算当前输出通道的起始和结束索引
      int64_t id0 = start_index(od, output_depth, input_depth);
      int64_t id1 = end_index(od, output_depth, input_depth);
      int64_t kd = id1 - id0;  // 计算深度方向的尺寸

      // 计算当前输出高度的起始和结束索引
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);
      int64_t kh = ih1 - ih0;  // 计算高度方向的尺寸

      // 计算当前输出宽度的起始和结束索引
      int64_t iw0 = start_index(ow, output_width, input_width);
      int64_t iw1 = end_index(ow, output_width, input_width);
      int64_t kw = iw1 - iw0;  // 计算宽度方向的尺寸

      // 定位到输出数据的起始位置，每个通道的数据
      scalar_t* out = output_data + i * channels;
      int64_t size = channels;

      // Pass I: 清零输出通道的数据
      int64_t d1 = 0;
      for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
        fVec sum_fvec = fVec(float(0));
        sum_fvec.store(sum + d1);  // 将零向量存储到 sum 数组的对应位置
      }
      for (; d1 < size; d1++) {
        sum[d1] = float(0);  // 将零值存储到 sum 数组的剩余位置
      }

      // Pass II: 计算局部和
      for (const auto id : c10::irange(id0, id1)) {
        for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
                // 定位输入数据中当前位置的起始位置
                scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                    id * input_height * input_width * channels +
                    ih * input_width * channels + iw * channels;

                int64_t d2 = 0;
                // 对于每个通道，使用向量化方式计算和
                for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
                    bVec data_bvec = bVec::loadu(in + d2);
                    fVec data_fvec0, data_fvec1;
                    std::tie(data_fvec0, data_fvec1) = convert_to_float<scalar_t>(data_bvec);

                    // 加载当前位置的和到 fVec 对象中，并加上对应的输入数据
                    fVec sum_fvec0 = fVec::loadu(sum + d2) + data_fvec0;
                    fVec sum_fvec1 = fVec::loadu(sum + d2 + fVec::size()) + data_fvec1;
                    // 将更新后的和存储回 sum 数组的对应位置
                    sum_fvec0.store(sum + d2);
                    sum_fvec1.store(sum + d2 + fVec::size());
                }
                // 处理剩余不足一个向量的部分
                for (; d2 < size; d2++) {
                    sum[d2] += float(in[d2]);
                }
            }
        }
      }

      // Pass III: 计算局部平均值
      int64_t d3 = 0;
      for (; d3 < size - (size % bVec::size()); d3 += bVec::size()) {
        // 使用向量化方式计算和的平均值，并存储到输出数据中
        fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(kd * kh * kw));
        fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) / fVec(float(kd * kh * kw));

        bVec out_bvec = convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_bvec.store(out + d3);  // 存储转换后的结果到输出数据中
      }
      // 处理剩余不足一个向量的部分
      for (; d3 < size; d3++) {
        out[d3] = scalar_t(sum[d3] / kd / kh / kw);  // 计算平均值并存储到输出数据中
      }

      // 移动到下一个输出索引位置
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量不是按照给定的内存格式连续的，则进行复制操作
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
// 结束上一个函数的定义，这是一个 C++ 函数
template <typename scalar_t>
void cpu_adaptive_avg_pool3d_backward(
    // 用 grad_output_ 的连续版本来初始化 grad_output
    Tensor& grad_input_,
    const Tensor& grad_output_) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  int64_t ndim = grad_output.ndimension();
  // 将批量大小和通道数视为一个维度处理
  int64_t channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_depth = grad_input.size(-3);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = grad_output.size(-3);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N 和 C 的维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 对每个通道 c 执行循环
    for (const auto c : c10::irange(begin, end)) {
      // 获取 grad_input 和 grad_output 的指针
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;

      // 对输出的深度维度 od 执行循环
      for (const auto od : c10::irange(output_depth)) {
        // 计算输入的起始和结束索引
        int64_t id0 = start_index(od, output_depth, input_depth);
        int64_t id1 = end_index(od, output_depth, input_depth);
        int64_t kd = id1 - id0;
        
        // 对输出的高度维度 oh 执行循环
        for (const auto oh : c10::irange(output_height)) {
          // 计算输入的起始和结束索引
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          // 对输出的宽度维度 ow 执行循环
          for (const auto ow : c10::irange(output_width)) {
            // 计算输入的起始和结束索引
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            // 计算梯度增量，由 grad_output_ptr 中的值除以 kd、kh 和 kw 得到
            scalar_t grad_delta = grad_output_ptr[od * output_width * output_height + oh * output_width + ow] / kd / kh / kw;

            // 对输入的深度、高度和宽度维度执行嵌套循环
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  // 将 grad_delta 添加到 grad_input_ptr 的相应位置
                  grad_input_ptr[id * input_height * input_width + ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });

  // 如果 grad_input_ 不是连续的，则复制 grad_input 到 grad_input_
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}
    // 定义了一个内存格式为ChannelsLast3d的Tensor
    const Tensor& grad_output_) {
      auto memory_format = at::MemoryFormat::ChannelsLast3d;
      // 使用指定的内存格式将grad_input_张量进行连续化处理
      auto grad_input = grad_input_.contiguous(memory_format);
      // 使用指定的内存格式将grad_output_张量进行连续化处理
      auto grad_output = grad_output_.contiguous(memory_format);
    
      // 获取grad_input和grad_output的数据指针
      auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
      auto grad_output_data = grad_output.data_ptr<scalar_t>();
    
      // 获取grad_input和grad_output的各维度大小
      int64_t nbatch = grad_input.size(0);
      int64_t channels = grad_input.size(1);
      int64_t input_depth = grad_input.size(2);
      int64_t input_height = grad_input.size(3);
      int64_t input_width = grad_input.size(4);
      int64_t output_depth = grad_output.size(2);
      int64_t output_height = grad_output.size(3);
      int64_t output_width = grad_output.size(4);
    
      using Vec = vec::Vectorized<scalar_t>;
      // 在N维度上进行并行处理
      at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
        // 遍历batch中的每个元素n
        for (const auto n : c10::irange(begin, end)) {
          // 计算grad_input和grad_output的指针位置
          scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
          scalar_t* grad_output_ptr = grad_output_data + n * output_depth * output_height * output_width * channels;
    
          // 遍历输出的深度维度
          for (const auto od : c10::irange(output_depth)) {
            // 计算输入和输出的深度索引范围
            int64_t id0 = start_index(od, output_depth, input_depth);
            int64_t id1 = end_index(od, output_depth, input_depth);
            int64_t kd = id1 - id0;
            // 遍历输出的高度维度
            for (const auto oh : c10::irange(output_height)) {
              // 计算输入和输出的高度索引范围
              int64_t ih0 = start_index(oh, output_height, input_height);
              int64_t ih1 = end_index(oh, output_height, input_height);
              int64_t kh = ih1 - ih0;
    
              // 遍历输出的宽度维度
              for (const auto ow : c10::irange(output_width)) {
                // 计算输入和输出的宽度索引范围
                int64_t iw0 = start_index(ow, output_width, input_width);
                int64_t iw1 = end_index(ow, output_width, input_width);
                int64_t kw = iw1 - iw0;
    
                // 计算当前位置的grad_output指针
                scalar_t* gout = grad_output_ptr + od * output_depth * channels + oh * output_width * channels + ow * channels;
                int64_t size = channels;
                // 遍历深度、高度、宽度的输入数据
                for (const auto id : c10::irange(id0, id1)) {
                  for (const auto ih : c10::irange(ih0, ih1)) {
                    for (const auto iw : c10::irange(iw0, iw1)) {
                      // 计算当前位置的grad_input指针
                      scalar_t* gin = grad_input_ptr + id * input_width * input_height * channels + ih * input_width * channels + iw * channels;
    
                      int64_t d = 0;
                      // 使用Vectorized类进行向量化处理
                      for (; d < size - (size % Vec::size()); d += Vec::size()) {
                        // 加载并处理向量化的数据
                        Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(kd * kh * kw));
                        // 存储处理后的向量化数据到grad_input中
                        gin_vec.store(gin + d);
                      }
                      // 处理剩余部分的数据
                      for (; d < size; d++) {
                        gin[d] += gout[d] / kd / kh / kw;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      });
    
      // 如果grad_input_不是按照指定的内存格式连续的，则复制grad_input
      if (!grad_input_.is_contiguous(memory_format)) {
        grad_input_.copy_(grad_input);
      }
} // anonymous namespace



void adaptive_avg_pool3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  // 根据输入张量建议的内存格式选择执行不同的操作
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 如果是连续内存格式，执行以下操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool3d", [&] {
        // 定义参数类型为当前数据类型的数学运算类型
        using param_t = at::opmath_type<scalar_t>;
        // 调用 CPU 版本的三维自适应平均池化函数
        cpu_adaptive_avg_pool3d<scalar_t, /*accscalar_t*/param_t>(output, input, output_size);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      // 如果是三维通道最后内存格式，执行以下操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, input.scalar_type(), "adaptive_avg_pool3d_channels_last", [&]{
        // 调用 CPU 版本的通道最后内存格式的三维自适应平均池化函数
        cpu_adaptive_avg_pool3d_channels_last<scalar_t>(output, input, output_size);
      });
      break;
    }
    default:
      // 如果不支持的内存格式，抛出错误信息
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adapative_avg_pool3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output) {
  // 根据梯度输出张量建议的内存格式选择执行不同的操作
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 如果是连续内存格式，执行以下操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool3d_backward", [&] {
        // 调用 CPU 版本的三维自适应平均池化反向传播函数
        cpu_adaptive_avg_pool3d_backward<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      // 如果是三维通道最后内存格式，执行以下操作
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, grad_output.scalar_type(), "adaptive_avg_pool3d_backward_channels_last", [&]{
        // 调用 CPU 版本的通道最后内存格式的三维自适应平均池化反向传播函数
        cpu_adaptive_avg_pool3d_backward_channels_last<scalar_t>(grad_input, grad_output);
      });
      break;
    }
    default:
      // 如果不支持的内存格式，抛出错误信息
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

// 注册适用于二维自适应平均池化的分发函数
REGISTER_DISPATCH(adaptive_avg_pool2d_kernel, &adaptive_avg_pool2d_kernel_impl);
// 注册适用于二维自适应平均池化反向传播的分发函数
REGISTER_DISPATCH(adaptive_avg_pool2d_backward_kernel, &adapative_avg_pool2d_backward_kernel_impl);
// 注册适用于三维自适应平均池化的分发函数
REGISTER_DISPATCH(adaptive_avg_pool3d_kernel, &adaptive_avg_pool3d_kernel_impl);
// 注册适用于三维自适应平均池化反向传播的分发函数
REGISTER_DISPATCH(adaptive_avg_pool3d_backward_kernel, &adapative_avg_pool3d_backward_kernel_impl);

} // at::native
```