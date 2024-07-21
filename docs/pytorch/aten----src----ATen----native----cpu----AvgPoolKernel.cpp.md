# `.\pytorch\aten\src\ATen\native\cpu\AvgPoolKernel.cpp`

```py
// 定义宏，用于只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 ATen 库中的各种头文件
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

// ATen 库的命名空间 at::native
namespace at::native {

// 匿名命名空间，限定作用域，仅在当前文件内可见
namespace {

// 模板函数，用于执行 CPU 上的平均池化操作
template <typename scalar_t>
void cpu_avg_pool2d(
    const Tensor& output_,      // 输出张量
    const Tensor& input_,       // 输入张量
    int64_t kW, int64_t kH,     // 池化核大小 (width, height)
    int64_t dW, int64_t dH,     // 步幅 (width, height)
    int64_t padW, int64_t padH, // 填充 (width, height)
    bool count_include_pad,     // 是否包括填充在内的数值计算
    std::optional<int64_t> divisor_override) { // 覆盖除数（可选）

  // 定义累加类型为 scalar_t 对应的数学操作类型
  using acc_t = at::opmath_type<scalar_t>;

  // 对输入和输出张量进行内存连续性检查并获取连续化后的副本
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  // 获取输入和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 计算输出张量中元素的总数
  int64_t numel = output.numel();

  // 获取输入张量的维度数
  int64_t ndim = input.ndimension();

  // 如果输入张量是三维的，通道数是 input.size(0)，否则是 input.size(0) * input.size(1)
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);

  // 获取输入和输出张量的高度和宽度
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  // 并行执行在 N（batch 大小）、C（通道数）、H（高度）、W（宽度）上的操作
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    // 初始化数据索引，设置起始位置
    data_index_init(begin, c, channels, oh, output_height, ow, output_width);

    // 遍历当前并行操作的每一个输出索引
    for (const auto i : c10::irange(begin, end)) {
      output_data[i] = static_cast<scalar_t>(0);

      // 指向输入数据的本地指针
      const scalar_t* input_ptr = input_data + c * input_height * input_width;

      // 计算输入图像的平均值...
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      // 如果池化区域为空，则跳过当前输出索引
      if (ih0 >= ih1 || iw0 >= iw1) {
        // 移动到下一个输出索引
        data_index_step(c, channels, oh, output_height, ow, output_width);
        continue;
      }

      // 累加类型为 acc_t 的和
      acc_t sum = 0;

      // 计算除数因子
      int64_t divide_factor;
      if (divisor_override.has_value()) {
        divide_factor = divisor_override.value();
      } else {
        if(count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (ih1 - ih0) * (iw1 - iw0);
        }
      }

      // 遍历计算池化区域内的和
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          sum += input_ptr[ih * input_width + iw];
        }
      }
      
      // 将平均值添加到输出中
      output_data[i] += scalar_t(sum / divide_factor);

      // 移动到下一个输出索引
      data_index_step(c, channels, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量不是内存连续的，则需要进一步处理...
    output_.copy_(output);



# 使用 PyTorch 中的张量操作，将 output 张量的值复制到 output_ 张量中
output_.copy_(output);


这行代码使用了 PyTorch 提供的张量方法 `copy_()`，它用来将一个张量 (`output`) 的值复制到另一个张量 (`output_`) 中。注意，`copy_()` 是一个就地操作，即它会修改 `output_` 张量的内容而不返回任何新的对象。
// 模板函数：在通道为最后维度的情况下，执行二维平均池化操作
template <typename scalar_t,
          typename std::enable_if<!is_reduced_floating_point<scalar_t>::value, int>::type = 0>
void cpu_avg_pool2d_channels_last(
    const Tensor& output_,                // 输出张量
    const Tensor& input_,                 // 输入张量
    int64_t kW, int64_t kH,               // 池化窗口的宽度和高度
    int64_t dW, int64_t dH,               // 池化步幅的宽度和高度
    int64_t padW, int64_t padH,           // 填充的宽度和高度
    bool count_include_pad,               // 是否包括填充像素在内
    std::optional<int64_t> divisor_override) {  // 可选的除数覆盖
  // 检查输入张量是否为四维，即(batch, channels, height, width)
  TORCH_CHECK(input_.ndimension() == 4,
              "2d average pooling with channels last format supports tensors with 4 dims");
  // 指定内存布局格式为ChannelsLast，并确保输入张量使用该格式连续存储
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  // 获取输入和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取张量的维度信息
  int64_t nbatch = input.size(0);         // batch size
  int64_t channels = input.size(1);       // 通道数
  int64_t input_height = input.size(2);   // 输入图像的高度
  int64_t input_width = input.size(3);    // 输入图像的宽度
  int64_t output_height = output.size(2); // 输出图像的高度
  int64_t output_width = output.size(3);  // 输出图像的宽度

  // 使用Vectorized类别别名Vec来处理标量数据类型的并行向量化操作
  using Vec = vec::Vectorized<scalar_t>;
  // 在维度N（batch）、H（height）、W（width）上并行执行
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化索引变量
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    // 计算向量化操作所需的尺寸
    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    for (const auto i : c10::irange(begin, end)) {
      // 对输入图像计算均值...

      // 计算输出图像的起始位置
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      // 计算输出图像的结束位置（考虑padding）
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      // 计算池化区域的大小
      int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
      // 确保计算的区域在有效范围内
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      int64_t divide_factor;
      // 根据是否有自定义的除数来确定划分因子
      if (divisor_override.has_value()) {
        divide_factor = divisor_override.value();
      } else {
        // 根据是否包含padding来决定划分因子
        if(count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (ih1 - ih0) * (iw1 - iw0);
        }
      }

      // 指向输出数据的指针
      scalar_t* out = output_data + i * channels;

      // 第一步：将输出的通道置零
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }

      // 如果计算区域为空，则直接处理下一个输出索引
      if (ih0 >= ih1 || iw0 >= iw1) {
        // 跳到下一个输出索引
        data_index_step(n, nbatch, oh, output_height, ow, output_width);
        continue;
      }

      // 第二步：计算局部和
      for (const auto ih : c10::irange(ih0, ih1)) {
        for (const auto iw : c10::irange(iw0, iw1)) {
          // 指向输入数据的指针
          const scalar_t* in = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;

          int64_t d2 = 0;
          for (; d2 < len; d2 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
            out_vec.store(out + d2);
          }
          for (; d2 < size; d2++) {
            out[d2] += in[d2];
          }
        }
      }

      // 第三步：计算局部平均值
      int64_t d3 = 0;
      for (; d3 < len; d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(divide_factor));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / divide_factor;
      }

      // 跳到下一个输出索引
      data_index_step(n, nbatch, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量不是按照指定内存格式连续，则进行复制操作
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
// 模板函数，用于在通道最后格式上执行 2D 平均池化的 CPU 实现
template <typename scalar_t,
          typename std::enable_if<is_reduced_floating_point<scalar_t>::value, int>::type = 0>
void cpu_avg_pool2d_channels_last(
    const Tensor& output_,
    const Tensor& input_,
    int64_t kW, int64_t kH,
    int64_t dW, int64_t dH,
    int64_t padW, int64_t padH,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  // 检查输入张量维度是否为4，即支持通道最后格式的 2D 平均池化
  TORCH_CHECK(input_.ndimension() == 4,
              "2d average pooling with channels last format supports tensors with 4 dims");

  // 设置内存格式为通道最后格式，对输入和输出张量进行连续化操作
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  // 获取输入和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入和输出张量的尺寸信息
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  // 使用向量化类型定义
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;

  // 在 N, H, W 维度上并行处理
  at::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化循环索引
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

    // 为求和准备临时缓冲区，使用 float 作为累加类型
    // 由于输出张量的数据类型可能为 BFloat16 或 Half，不能直接重用输出缓冲区
    auto sum_arr = std::make_unique<float []>(channels);
    float* sum = sum_arr.get();

    int64_t size = channels;
    }
  });

  // 如果输出张量不是按照通道最后格式连续的，则复制到输出张量
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

// 模板函数，用于执行 2D 平均池化的反向传播（梯度计算）的 CPU 实现
template <typename scalar_t>
void cpu_avg_pool2d_backward(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  // 对梯度输入和梯度输出张量进行连续化操作
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取梯度输出和梯度输入张量的数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取梯度输出张量的维度
  int64_t ndim = grad_output.ndimension();
  // 将批次大小和通道数视为一个维度处理
  int64_t channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N, C 维度上并行处理
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    `
        // 遍历通道数，从 begin 到 end
        for (const auto c : c10::irange(begin, end)) {
          // 计算当前通道的梯度输入数据指针
          scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
          // 计算当前通道的梯度输出数据指针
          const scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;
    
          // 遍历输出高度，从 0 到 output_height - 1
          for (const auto oh : c10::irange(output_height)) {
            // 遍历输出宽度，从 0 到 output_width - 1
            for (const auto ow : c10::irange(output_width)) {
              // 计算输入高度的起始位置
              int64_t ih0 = oh * dH - padH;
              // 计算输入宽度的起始位置
              int64_t iw0 = ow * dW - padW;
              // 计算输入高度的结束位置
              int64_t ih1 = std::min(ih0 + kH, input_height + padH);
              // 计算输入宽度的结束位置
              int64_t iw1 = std::min(iw0 + kW, input_width + padW);
              // 计算池化区域的大小
              int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
              // 确保输入高度起始位置不小于0
              ih0 = std::max(ih0, (int64_t) 0);
              // 确保输入宽度起始位置不小于0
              iw0 = std::max(iw0, (int64_t) 0);
              // 确保输入高度结束位置不超过输入高度
              ih1 = std::min(ih1, input_height);
              // 确保输入宽度结束位置不超过输入宽度
              iw1 = std::min(iw1, input_width);
    
              // 定义除法因子
              int64_t divide_factor;
              // 如果 divisor_override 有值，使用其值作为除法因子
              if (divisor_override.has_value()) {
                divide_factor = divisor_override.value();
              } else {
                // 根据 count_include_pad 的值，决定除法因子的计算方式
                if(count_include_pad) {
                  divide_factor = pool_size;
                } else {
                  divide_factor = (ih1 - ih0) * (iw1 - iw0);
                }
              }
    
              // 计算梯度增量
              scalar_t grad_delta = grad_output_ptr[oh * output_width + ow] / divide_factor;
              // 遍历输入高度范围从 ih0 到 ih1 - 1
              for (const auto ih : c10::irange(ih0, ih1)) {
                // 遍历输入宽度范围从 iw0 到 iw1 - 1
                for (const auto iw : c10::irange(iw0, iw1)) {
                  // 更新梯度输入数据
                  grad_input_ptr[ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      });
    
      // 如果梯度输入数据不是连续的内存布局，拷贝数据到连续内存
      if (!grad_input_.is_contiguous()) {
        grad_input_.copy_(grad_input);
      }
  }

template <typename scalar_t>
void cpu_avg_pool2d_backward_channels_last(
    const Tensor& grad_input_,                             // grad_input_ 参数：反向传播的梯度输入张量
    const Tensor& grad_output_,                            // grad_output_ 参数：反向传播的梯度输出张量
    int kW, int kH,                                         // kW, kH 参数：池化窗口的宽度和高度
    int dW, int dH,                                         // dW, dH 参数：池化窗口的水平和垂直步长
    int padW, int padH,                                     // padW, padH 参数：输入的填充宽度和高度
    bool count_include_pad,                                 // count_include_pad 参数：是否包含填充像素计算池化平均值
    std::optional<int64_t> divisor_override) {              // divisor_override 参数：覆盖池化平均值分母的可选值

  auto memory_format = at::MemoryFormat::ChannelsLast;       // 设置内存格式为 ChannelsLast
  auto grad_input = grad_input_.contiguous(memory_format);   // 将梯度输入张量转换为连续的 ChannelsLast 内存格式
  auto grad_output = grad_output_.contiguous(memory_format); // 将梯度输出张量转换为连续的 ChannelsLast 内存格式

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();    // 获取可变的梯度输入数据指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();    // 获取常量的梯度输出数据指针

  int64_t nbatch = grad_input.size(0);                    // 获取批量大小
  int64_t channels = grad_input.size(1);                  // 获取通道数
  int64_t input_height = grad_input.size(2);              // 获取输入高度
  int64_t input_width = grad_input.size(3);               // 获取输入宽度
  int64_t output_height = grad_output.size(2);            // 获取输出高度
  int64_t output_width = grad_output.size(3);             // 获取输出宽度

  using Vec = vec::Vectorized<scalar_t>;                   // 使用 Vectorized 类型别名简化向量化操作的类型

  // 在维度 N 上并行执行
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {         // 遍历每个批次中的每个样本
      scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;    // 计算当前批次中梯度输入的指针位置
      const scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;  // 计算当前批次中梯度输出的指针位置

      for (const auto oh : c10::irange(output_height)) {   // 遍历输出的高度维度
        for (const auto ow : c10::irange(output_width)) {  // 遍历输出的宽度维度
          int64_t ih0 = oh * dH - padH;                    // 计算池化窗口在输入张量中的起始高度位置
          int64_t iw0 = ow * dW - padW;                    // 计算池化窗口在输入张量中的起始宽度位置
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);  // 计算池化窗口在输入张量中的结束高度位置
          int64_t iw1 = std::min(iw0 + kW, input_width + padW);   // 计算池化窗口在输入张量中的结束宽度位置
          int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);    // 计算池化窗口的大小
          ih0 = std::max(ih0, (int64_t) 0);                 // 确保起始高度在有效范围内
          iw0 = std::max(iw0, (int64_t) 0);                 // 确保起始宽度在有效范围内
          ih1 = std::min(ih1, input_height);                // 确保结束高度在有效范围内
          iw1 = std::min(iw1, input_width);                 // 确保结束宽度在有效范围内

          int64_t divide_factor;
          if (divisor_override.has_value()) {               // 如果覆盖值存在
            divide_factor = divisor_override.value();       // 使用覆盖值作为分母
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;                    // 如果包含填充，则分母为池化窗口大小
            } else {
              divide_factor = (ih1 - ih0) * (iw1 - iw0);    // 否则分母为有效池化窗口大小
            }
          }

          const scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;  // 获取当前输出位置的梯度输出指针
          int64_t size = channels;                          // 设置当前通道数
          int64_t len = size - (size % Vec::size());        // 计算可以向量化处理的最大长度

          for (const auto ih : c10::irange(ih0, ih1)) {      // 在高度范围内循环
            for (const auto iw : c10::irange(iw0, iw1)) {    // 在宽度范围内循环
              scalar_t* gin = grad_input_ptr + ih * input_width * channels + iw * channels;  // 获取当前输入位置的梯度输入指针

              int64_t d = 0;
              for (; d < len; d += Vec::size()) {           // 向量化处理主循环
                Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(divide_factor));  // 加载、计算和存储向量化数据
                gin_vec.store(gin + d);                     // 存储处理后的向量化数据
              }
              for (; d < size; d++) {                       // 处理剩余的非向量化数据
                gin[d] += gout[d] / divide_factor;          // 计算梯度输入的贡献
              }
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    # 使用 PyTorch 的张量操作，将 grad_input 的值复制到 grad_input_ 中
    grad_input_.copy_(grad_input);
}

// 定义了一个函数模板，用于计算二维平均池化的核心实现
void avg_pool2d_kernel_impl(
    const Tensor& output,  // 输出张量，用于存储池化后的结果
    const Tensor& input,   // 输入张量，即待池化的原始数据
    int64_t kW, int64_t kH,  // 池化窗口的宽度和高度
    int64_t dW, int64_t dH,  // 池化窗口的步长（宽度和高度方向）
    int64_t padW, int64_t padH,  // 水平和垂直方向的填充大小
    bool count_include_pad,  // 是否包括填充在内进行计算
    std::optional<int64_t> divisor_override) {  // 可选参数，覆盖默认的除数

  // 根据输入张量的内存格式选择相应的池化操作
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 使用宏来分发不同数据类型的实现，调用 CPU 上的平均池化函数
      AT_DISPATCH_FLOATING_TYPES_AND3(kLong, kBFloat16, kHalf, input.scalar_type(), "avg_pool2d", [&] {
        cpu_avg_pool2d<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 对于通道最后的内存格式，调用相应的通道最后的平均池化函数
      AT_DISPATCH_FLOATING_TYPES_AND3(kLong, kBFloat16, kHalf, input.scalar_type(), "avg_pool2d_channels_last", [&] {
        cpu_avg_pool2d_channels_last<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    default:
      // 如果输入的内存格式不支持，抛出异常
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 定义了一个函数模板，用于反向传播的二维平均池化核心实现
void avg_pool2d_backward_kernel_impl(
    const Tensor& grad_input,  // 梯度输入张量，用于计算反向传播的梯度
    const Tensor& grad_output,  // 梯度输出张量，即池化层的反向传播梯度
    int kW, int kH,  // 池化窗口的宽度和高度
    int dW, int dH,  // 池化窗口的步长（宽度和高度方向）
    int padW, int padH,  // 水平和垂直方向的填充大小
    bool count_include_pad,  // 是否包括填充在内进行计算
    std::optional<int64_t> divisor_override) {  // 可选参数，覆盖默认的除数

  // 根据梯度输出张量的内存格式选择相应的反向传播池化操作
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // 使用宏来分发不同数据类型的实现，调用 CPU 上的平均池化反向传播函数
      AT_DISPATCH_FLOATING_TYPES_AND3(kLong, kBFloat16, kHalf, grad_output.scalar_type(), "avg_pool2d_backward", [&] {
        cpu_avg_pool2d_backward<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      // 对于通道最后的内存格式，调用相应的通道最后的平均池化反向传播函数
      AT_DISPATCH_FLOATING_TYPES_AND3(kLong, kBFloat16, kHalf, grad_output.scalar_type(), "avg_pool2d_backward_channels_last", [&] {
        cpu_avg_pool2d_backward_channels_last<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
      });
      break;
    }
    default:
      // 如果梯度输出的内存格式不支持，抛出异常
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 定义了一个函数模板，用于计算三维平均池化的核心实现
template <typename scalar_t>
void cpu_avg_pool3d(
    const Tensor& output_,  // 输出张量，用于存储三维平均池化的结果
    const Tensor& input_,   // 输入张量，即待池化的三维原始数据
    int64_t kW, int64_t kH, int64_t kD,  // 池化窗口的宽度、高度和深度
    int64_t dW, int64_t dH, int64_t dD,  // 池化窗口的步长（宽度、高度和深度方向）
    int64_t padW, int64_t padH, int64_t padD,  // 填充的大小（宽度、高度和深度方向）
    bool count_include_pad,
  std::optional<int64_t> divisor_override) {
  // 使用at命名空间的opmath_type模板类，将scalar_t类型定义为acc_t
  using acc_t = at::opmath_type<scalar_t>;

  // 将输入张量(input_)和输出张量(output_)转换为连续内存布局
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  // 获取输入和输出张量的数据指针
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输出张量的元素数量和输入张量的维度数
  int64_t numel = output.numel();
  int64_t ndim = input.ndimension();

  // 计算通道数，如果输入张量是四维的，则通道数为第一维的大小；否则为第一维和第二维大小的乘积
  int64_t channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);

  // 获取输入和输出张量在深度、高度和宽度方向上的大小
  int64_t input_depth = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = output.size(-3);
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  // 并行处理N、C、D、H、W维度上的操作
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    // 初始化索引变量c、od、oh、ow
    int64_t c = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;

    // 使用data_index_init函数初始化数据索引
    data_index_init(begin, c, channels, od, output_depth, oh, output_height, ow, output_width);

    // 遍历指定范围内的索引
    for (const auto i : c10::irange(begin, end)) {
      // 将输出张量中的当前元素置为0
      output_data[i] = static_cast<scalar_t>(0);

      // 计算输入指针的起始位置
      scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width;

      // 计算输入图像的均值...
      int64_t id0 = od * dD - padD;
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t id1 = std::min(id0 + kD, input_depth + padD);
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);

      // 确保边界不超出输入图像的范围
      id0 = std::max(id0, (int64_t) 0);
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      id1 = std::min(id1, input_depth);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      // 如果某些维度的范围无效，跳过当前输出索引
      if (id0 >= id1 || ih0 >= ih1 || iw0 >= iw1) {
        // 移动到下一个输出索引
        data_index_step(c, channels, od, output_depth, oh, output_height, ow, output_width);
        continue;
      }

      // 初始化累加器sum
      acc_t sum = 0;

      // 初始化除数因子divide_factor
      int64_t divide_factor;
      // 根据是否提供了divisor_override来确定divide_factor的值
      if (divisor_override.has_value()) {
        divide_factor = divisor_override.value();
      } else {
        // 根据count_include_pad确定是否包括填充值在内来计算divide_factor
        if(count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
        }
      }

      // 遍历计算输入指针范围内的所有像素，并累加到sum中
      for (const auto id : c10::irange(id0, id1)) {
        for (const auto ih : c10::irange(ih0, ih1)) {
          for (const auto iw : c10::irange(iw0, iw1)) {
            sum += input_ptr[id * input_height * input_width + ih * input_width + iw];
          }
        }
      }

      // 将sum除以divide_factor，并加到输出张量对应位置上
      output_data[i] += scalar_t(sum / divide_factor);

      // 移动到下一个输出索引
      data_index_step(c, channels, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  // 如果输出张量不是连续的，则复制连续的output到输出张量中
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}
  // 模板函数，用于在不是降低浮点数精度的情况下执行3D平均池化操作
template <typename scalar_t,
          // 如果不是降低浮点数精度，则启用SFINAE，设置为0
          typename std::enable_if<!is_reduced_floating_point<scalar_t>::value, int>::type = 0>
void cpu_avg_pool3d_channels_last(
    // 输出张量的引用，用于存储池化后的结果
    const Tensor& output_,
    // 输入张量的引用，作为池化操作的输入
    const Tensor& input_,
    // 池化核的宽、高、深度
    int64_t kW, int64_t kH, int64_t kD,
    // 池化步长的宽、高、深度
    int64_t dW, int64_t dH, int64_t dD,
    // 输入张量填充的宽、高、深度
    int64_t padW, int64_t padH, int64_t padD,
    // 是否包含填充值在内的计数
    bool count_include_pad,
    // 可选的覆盖除法器
    std::optional<int64_t> divisor_override) {
  // 检查输入张量的维度是否为5维
  TORCH_CHECK(input_.ndimension() == 5,
              "3d average pooling with channels last format supports tensors with 5 dims");
  // 设置内存格式为ChannelsLast3d，并将输入张量转换为该格式的连续张量
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  // 获取输入和输出张量的数据指针
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取输入和输出张量的各个维度大小
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  int64_t output_depth = output.size(2);
  int64_t output_height = output.size(3);
  int64_t output_width = output.size(4);

  // 使用Vectorized<scalar_t>作为Vec的别名
  using Vec = vec::Vectorized<scalar_t>;
  // 并行处理在维度N、H、W上的操作
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    // 初始化数据索引，开始在N、OD、OH、OW上迭代
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    // 计算每个向量化操作的大小
    int64_t size = channels;
    int64_t len = size - (size % Vec::size());
    for (const auto i : c10::irange(begin, end)) {
      // 遍历从 begin 到 end 的索引 i

      // 计算输入图像的均值...
      int64_t id0 = od * dD - padD;
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t id1 = std::min(id0 + kD, input_depth + padD);
      int64_t ih1 = std::min(ih0 + kH, input_height + padH);
      int64_t iw1 = std::min(iw0 + kW, input_width + padW);
      int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
      id0 = std::max(id0, (int64_t) 0);
      ih0 = std::max(ih0, (int64_t) 0);
      iw0 = std::max(iw0, (int64_t) 0);
      id1 = std::min(id1, input_depth);
      ih1 = std::min(ih1, input_height);
      iw1 = std::min(iw1, input_width);

      int64_t divide_factor;
      if (divisor_override.has_value()) {
        // 如果存在 divisor_override，使用其值作为除数
        divide_factor = divisor_override.value();
      } else {
        if(count_include_pad) {
          // 如果计数包括填充，则使用 pool_size 作为除数
          divide_factor = pool_size;
        } else {
          // 否则使用池化区域的体积作为除数
          divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
        }
      }

      scalar_t* out = output_data + i * channels;

      // Pass I: zero the out lane
      // Pass I：将输出通道置零
      int64_t d1 = 0;
      for (; d1 < len; d1 += Vec::size()) {
        Vec out_vec = Vec(scalar_t(0));
        out_vec.store(out + d1);
      }
      for (; d1 < size; d1++) {
        out[d1] = scalar_t(0);
      }

      if (id0 >= id1 || ih0 >= ih1 || iw0 >= iw1) {
        // 如果池化区域为空，跳到下一个输出索引
        data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
        continue;
      }

      // Pass II: compute local sum
      // Pass II：计算局部和
      for (const auto id : c10::irange(id0, id1)) {
        for (const auto ih : c10::irange(ih0, ih1)) {
          for (const auto iw : c10::irange(iw0, iw1)) {
            // 计算输入的地址
            scalar_t* in = input_data + n * input_depth * input_height * input_width * channels +
                id * input_height * input_width * channels + ih * input_width * channels + iw * channels;

            int64_t d2 = 0;
            for (; d2 < len; d2 += Vec::size()) {
              Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
              out_vec.store(out + d2);
            }
            for (; d2 < size; d2++) {
              out[d2] += in[d2];
            }
          }
        }
      }

      // Pass III: compute local average
      // Pass III：计算局部平均值
      int64_t d3 = 0;
      for (; d3 < len; d3 += Vec::size()) {
        Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(divide_factor));
        out_vec.store(out + d3);
      }
      for (; d3 < size; d3++) {
        out[d3] = out[d3] / divide_factor;
      }

      // move on to next output index
      // 移动到下一个输出索引
      data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous(memory_format)) {
    // 如果输出不是按照指定的内存格式连续存储，则复制输出数据
    output_.copy_(output);
  }
// 结束函数模板
}

// 3D 平均池化的 CPU 实现，处理通道在最后的内存格式
template <typename scalar_t,
          typename std::enable_if<is_reduced_floating_point<scalar_t>::value, int>::type = 0>
void cpu_avg_pool3d_channels_last(
    const Tensor& output_,                          // 输出张量
    const Tensor& input_,                           // 输入张量
    int64_t kW, int64_t kH, int64_t kD,              // 池化核的宽、高、深度
    int64_t dW, int64_t dH, int64_t dD,              // 步长的宽、高、深度
    int64_t padW, int64_t padH, int64_t padD,        // 填充的宽、高、深度
    bool count_include_pad,                         // 是否包含填充值在内
    std::optional<int64_t> divisor_override) {       // 可选的覆盖除数

  // 检查输入张量的维度是否为 5
  TORCH_CHECK(input_.ndimension() == 5,
              "3d average pooling with channels last format supports tensors with 5 dims");

  // 设置内存格式为 ChannelsLast3d，并确保输入输出张量是连续的
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  // 获取输入和输出数据指针，使用 BFloat16 类型
  auto input_data = input.data_ptr<BFloat16>();
  auto output_data = output.data_ptr<BFloat16>();

  // 获取输入和输出张量的维度信息
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  int64_t output_depth = output.size(2);
  int64_t output_height = output.size(3);
  int64_t output_width = output.size(4);

  // 使用 Vectorized 类型来进行向量化操作
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;

  // 在 N、H、W 维度上并行执行操作
  at::parallel_for(0, nbatch * output_depth * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

    // 用于临时存储求和结果的缓冲区，使用 float 作为累加类型
    // 由于输出张量是 BFloat16 类型，不能重用输出缓冲区存储求和结果
    auto sum_arr = std::make_unique<float []>(channels);
    float* sum = sum_arr.get();

    int64_t size = channels;
  });

  // 如果输出张量不是按照指定的内存格式连续，则将计算结果复制回输出张量
  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

// 3D 平均池化的反向传播，处理任意类型的标量
template <typename scalar_t>
void cpu_avg_pool3d_backward(
    const Tensor& grad_input_,                      // 梯度输入张量
    const Tensor& grad_output_,                     // 梯度输出张量
    int kW, int kH, int kD,                          // 池化核的宽、高、深度
    int dW, int dH, int dD,                          // 步长的宽、高、深度
    int padW, int padH, int padD,                    // 填充的宽、高、深度
    bool count_include_pad,                         // 是否包含填充值在内
    std::optional<int64_t> divisor_override) {       // 可选的覆盖除数

  // 确保梯度输出张量是连续的
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取梯度输出和梯度输入数据指针，使用标量类型作为模板参数
  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();

  // 获取梯度输出张量的维度信息
  int64_t ndim = grad_output.ndimension();
  // 将批量大小和通道数视为一个维度处理
  int64_t channels = ndim == 4 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
  int64_t input_depth = grad_input.size(-3);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = grad_output.size(-3);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // 在 N、C 维度上并行执行反向传播操作
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    // 对于每个通道 c，在输入和输出之间计算梯度
    for (const auto c : c10::irange(begin, end)) {
      // 计算当前通道在梯度输入数据和梯度输出数据中的起始位置指针
      scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;

      // 遍历输出的深度、高度和宽度
      for (const auto od : c10::irange(output_depth)) {
        for (const auto oh : c10::irange(output_height)) {
          for (const auto ow : c10::irange(output_width)) {
            // 计算池化窗口在输入中的起始和结束位置
            int64_t id0 = od * dD - padD;
            int64_t ih0 = oh * dH - padH;
            int64_t iw0 = ow * dW - padW;
            int64_t id1 = std::min(id0 + kD, input_depth + padD);
            int64_t ih1 = std::min(ih0 + kH, input_height + padH);
            int64_t iw1 = std::min(iw0 + kW, input_width + padW);

            // 计算池化窗口大小
            int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);

            // 确保池化窗口在输入尺寸范围内
            id0 = std::max(id0, (int64_t) 0);
            ih0 = std::max(ih0, (int64_t) 0);
            iw0 = std::max(iw0, (int64_t) 0);
            ih1 = std::min(ih1, input_height);
            iw1 = std::min(iw1, input_width);

            int64_t divide_factor;

            // 根据是否有覆盖的除法因子来确定分母
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              // 根据是否包括填充来选择池化窗口大小作为分母
              if (count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
              }
            }

            // 计算当前输出位置的梯度与分母的比率
            scalar_t grad_delta = grad_output_ptr[od * output_height * output_width + oh * output_width + ow] / divide_factor;

            // 在梯度输入的相应区域累加梯度
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  grad_input_ptr[id * input_height * input_width + ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });

  // 如果梯度输入不是连续的，则进行复制操作使其连续
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
  // 结束 CPU 平均池化反向传播函数的声明
template <typename scalar_t>
void cpu_avg_pool3d_backward_channels_last(
    const Tensor& grad_input_,  // 输入梯度张量
    const Tensor& grad_output_,  // 输出梯度张量
    int kW, int kH, int kD,  // 池化核的宽度、高度和深度
    int dW, int dH, int dD,  // 池化步长的宽度、高度和深度
    int padW, int padH, int padD,  // 池化填充的宽度、高度和深度
    bool count_include_pad,  // 是否包含填充值在内的计数
    std::optional<int64_t> divisor_override) {  // 可选的覆盖除数

  auto memory_format = at::MemoryFormat::ChannelsLast3d;  // 内存格式为 ChannelsLast3d
  auto grad_input = grad_input_.contiguous(memory_format);  // 使用指定的内存格式创建连续的输入梯度张量
  auto grad_output = grad_output_.contiguous(memory_format);  // 使用指定的内存格式创建连续的输出梯度张量

  auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();  // 获取可变的输入梯度数据指针
  auto grad_output_data = grad_output.data_ptr<scalar_t>();  // 获取输出梯度数据指针

  int64_t nbatch = grad_input.size(0);  // 获取输入梯度张量的批次维度大小
  int64_t channels = grad_input.size(1);  // 获取输入梯度张量的通道维度大小
  int64_t input_depth = grad_input.size(2);  // 获取输入梯度张量的深度维度大小
  int64_t input_height = grad_input.size(3);  // 获取输入梯度张量的高度维度大小
  int64_t input_width = grad_input.size(4);  // 获取输入梯度张量的宽度维度大小
  int64_t output_depth = grad_output.size(2);  // 获取输出梯度张量的深度维度大小
  int64_t output_height = grad_output.size(3);  // 获取输出梯度张量的高度维度大小
  int64_t output_width = grad_output.size(4);  // 获取输出梯度张量的宽度维度大小

  using Vec = vec::Vectorized<scalar_t>;  // 使用 Vectorized 类型进行向量化操作

  // 在 N 维度上并行处理
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      // 计算当前样本在梯度张量中的起始位置
      scalar_t* grad_input_ptr = grad_input_data + n * input_depth * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

      // 遍历输出张量的深度、高度、宽度维度
      for (const auto od : c10::irange(output_depth)) {
        for (const auto oh : c10::irange(output_height)) {
          for (const auto ow : c10::irange(output_width)) {
            // 计算池化窗口在输入张量中的起始和结束位置
            int64_t id0 = od * dD - padD;
            int64_t ih0 = oh * dH - padH;
            int64_t iw0 = ow * dW - padW;
            int64_t id1 = std::min(id0 + kD, input_depth + padD);
            int64_t ih1 = std::min(ih0 + kH, input_height + padH);
            int64_t iw1 = std::min(iw0 + kW, input_width + padW);
            int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);

            // 确保池化窗口在输入张量内部
            id0 = std::max(id0, (int64_t) 0);
            ih0 = std::max(ih0, (int64_t) 0);
            iw0 = std::max(iw0, (int64_t) 0);
            id1 = std::min(id1, input_depth);
            ih1 = std::min(ih1, input_height);
            iw1 = std::min(iw1, input_width);

            int64_t divide_factor;
            // 根据是否有 divisor_override 值来确定分母因子
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              // 根据 count_include_pad 来选择是否包含 padding 的像素数
              if(count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
              }
            }

            // 计算当前位置在梯度输出张量中的指针位置
            scalar_t* gout = grad_output_ptr + od * output_height * output_width * channels + oh * output_width * channels + ow * channels;
            int64_t size = channels;
            int64_t len = size - (size % Vec::size());

            // 遍历池化窗口内的像素点
            for (const auto id : c10::irange(id0, id1)) {
              for (const auto ih : c10::irange(ih0, ih1)) {
                for (const auto iw : c10::irange(iw0, iw1)) {
                  // 计算当前像素在梯度输入张量中的指针位置
                  scalar_t* gin = grad_input_ptr + id * input_height * input_width * channels + ih * input_width * channels + iw * channels;

                  int64_t d = 0;
                  // 使用向量化指令处理通道数据
                  for (; d < len; d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(divide_factor));
                    gin_vec.store(gin + d);
                  }
                  // 处理剩余通道数据
                  for (; d < size; d++) {
                    gin[d] += gout[d] / divide_factor;
                  }
                }
              }
            }
          }
        }
      }
    }
  });

  // 确保梯度输入张量是按指定内存格式连续的
  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
} // anonymous namespace



// 注册 avg_pool2d_kernel 的分发函数指针，调用 avg_pool2d_kernel_impl 实现池化操作
REGISTER_DISPATCH(avg_pool2d_kernel, &avg_pool2d_kernel_impl);



// 注册 avg_pool2d_backward_kernel 的分发函数指针，调用 avg_pool2d_backward_kernel_impl 实现池化反向传播操作
REGISTER_DISPATCH(avg_pool2d_backward_kernel, &avg_pool2d_backward_kernel_impl);



// 注册 avg_pool3d_kernel 的分发函数指针，调用 avg_pool3d_kernel_impl 实现三维池化操作
REGISTER_DISPATCH(avg_pool3d_kernel, &avg_pool3d_kernel_impl);



// 注册 avg_pool3d_backward_kernel 的分发函数指针，调用 avg_pool3d_backward_kernel_impl 实现三维池化反向传播操作
REGISTER_DISPATCH(avg_pool3d_backward_kernel, &avg_pool3d_backward_kernel_impl);



} // at::native


这段代码主要用于注册和分发池化（pooling）操作的函数指针，通过调用对应的实现函数来处理不同的池化操作和反向传播操作。
```