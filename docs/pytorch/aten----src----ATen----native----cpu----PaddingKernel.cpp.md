# `.\pytorch\aten\src\ATen\native\cpu\PaddingKernel.cpp`

```py
// 定义宏，仅在 Torch 的断言方法操作器中启用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入 ATen 库的 Tensor 类定义
#include <ATen/core/Tensor.h>

// 引入 ATen 库的调度机制
#include <ATen/Dispatch.h>

// 引入 ATen 库的并行处理功能
#include <ATen/Parallel.h>

// 引入 ATen 库的向量化处理支持
#include <ATen/cpu/vec/vec.h>

// 引入 ATen 库的填充操作相关功能
#include <ATen/native/Padding.h>

// 引入 ATen 库的 CPU 工具函数
#include <ATen/native/cpu/utils.h>

// 引入 C10 实用工具的范围处理
#include <c10/util/irange.h>

// 定义 at::native 命名空间
namespace at::native {

// 匿名命名空间，内部实现相关函数和结构体
namespace {

// 填充参数结构体，用于存储填充操作相关的参数信息
struct PaddingParams {
  int ndim;  // 维度数
  int64_t nbatch;  // 批次大小
  int64_t channels;  // 通道数

  // 当输出索引位于 [pad, input_width + pad) 范围内时，对宽度应用向量化逻辑
  // 仅适用于 Channels First 格式，当 pad_l 和 pad_r 均为正数时有效
  bool is_padding_positive_width;

  // 输入形状、输出形状、填充值、偏移值的小型向量数组
  c10::SmallVector<int64_t, 3u> ishape;
  c10::SmallVector<int64_t, 3u> oshape;
  c10::SmallVector<int64_t, 3u> pads;
  c10::SmallVector<int64_t, 3u> offsets;

  // 构造函数，根据输入、输出 Tensor 和填充参数初始化结构体
  PaddingParams(const Tensor& input, const Tensor& output, IntArrayRef padding) {
    ndim = padding.size() / 2;

    bool is_batch = input.dim() == ndim + 2;
    nbatch = is_batch ? input.size(0) : 1;
    channels = is_batch ? input.size(1) : input.size(0);

    // 判断填充宽度是否为正数
    is_padding_positive_width = padding[0] >= 0 && padding[1] >=0;

    // 处理批次模式和非批次模式的大小
    int ind = is_batch ? 2 : 1;
    for (const auto d : c10::irange(ndim)) {
      ishape.emplace_back(input.size(ind + d));
      oshape.emplace_back(output.size(ind + d));
    }

    // 将填充顺序重新组织为 {depth, height, width} 的顺序
    if (ndim == 1) {
      pads.emplace_back(padding[0]);
    } else if (ndim == 2) {
      pads.emplace_back(padding[2]);
      pads.emplace_back(padding[0]);
    } else {
      pads.emplace_back(padding[4]);
      pads.emplace_back(padding[2]);
      pads.emplace_back(padding[0]);
    }
    for (const auto d : c10::irange(ndim)) {
      int64_t pad = pads[d];
      auto i_start = std::max(int64_t(0), -pad);
      auto o_start = std::max(int64_t(0), pad);
      offsets.emplace_back(i_start - o_start);
    }
  };
};

// 反射填充结构体，实现反射填充索引计算的静态方法
struct ReflectionPad {
  static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
    int64_t i;
    if (j < pad) {
      i = pad * 2 - j;
    } else if (j >= pad && j < size + pad) {
      i = j;
    } else {
      i = (size + pad - 1) * 2 - j;
    }
    return i + offset;
  }
};

// 复制填充结构体，实现复制填充索引计算的静态方法
struct ReplicationPad {
  static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
    int64_t i;
    if (j < pad) {
      i = pad;
    } else if (j >= pad && j < size + pad) {
      i = j;
    } else {
      i = size + pad - 1;
    }
    return i + offset;
  }
};

// 标量复制函数的模板，实现标量复制操作
template <typename scalar_t>
static inline void copy_stub(scalar_t* out, const scalar_t* in, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  int64_t d = 0;
  // 使用向量化类 Vec 复制输入向量 in 到输出向量 out，处理对齐的部分
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec in_vec = Vec::loadu(in + d);
    in_vec.store(out + d);
  }
  // 处理剩余不满足对齐的部分，使用标准循环复制
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    out[d] = in[d];


// 将输入数组中索引为d的元素赋值给输出数组中索引为d的位置


  }


// 循环结束，代码块结束
// 并行处理一维情况下的填充操作，涵盖了批次和通道，并对宽度进行处理
at::parallel_for(0, channels * output_width, 1, [&](int64_t begin, int64_t end) {
  // 初始化索引：c 对应通道数，ow 对应输出宽度
  int64_t c{0}, ow{0};
  data_index_init(begin, c, channels, ow, output_width);

  // 遍历给定范围的索引
  for (const auto i : c10::irange(begin, end)) {
    // 计算输入中的索引 iw，基于 PaddingType 类型和给定的填充参数
    int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
    // 将输入数据复制到输出数据中
    output_data[i] = input_data[c * input_width + iw];
    // 更新数据索引
    data_index_step(c, channels, ow, output_width);
  }
});
    // 并行处理每个输出通道和输出高度的循环
    at::parallel_for(0, channels * output_height, 1, [&](int64_t begin, int64_t end) {
      // 初始化循环变量 c 和 oh
      int64_t c{0}, oh{0};
      // 初始化数据索引，确定起始位置
      data_index_init(begin, c, channels, oh, output_height);

      // 遍历当前范围内的元素
      for (const auto i : c10::irange(begin, end)) {
        // 计算输入高度的索引，考虑填充类型
        int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
        // 设置输出数据指针，指向当前输出元素的起始位置
        scalar_t* output_ptr = output_data + i * output_width;
        // 设置输入数据指针，指向当前输入元素的起始位置
        const scalar_t* input_ptr = input_data + c * input_height * input_width + ih * input_width;

        // 调用循环函数，执行数据复制或处理，根据是否是正向填充
        loop(output_ptr, input_ptr, p.is_padding_positive_width);
        // 更新数据索引，准备处理下一个元素
        data_index_step(c, channels, oh, output_height);
      }
    });
  } else if (ndim == 3) {
    // 并行处理每个输出通道、深度和高度的循环
    // 在 N、C、D、H 上并行，而在 W 上向量化
    at::parallel_for(0, channels * output_depth * output_height, 1, [&](int64_t begin, int64_t end) {
      // 初始化循环变量 c、od 和 oh
      int64_t c{0}, od{0}, oh{0};
      // 初始化数据索引，确定起始位置
      data_index_init(begin, c, channels, od, output_depth, oh, output_height);

      // 遍历当前范围内的元素
      for (const auto i : c10::irange(begin, end)) {
        // 计算输入深度的索引，考虑填充类型
        int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
        // 计算输入高度的索引，考虑填充类型
        int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
        // 设置输出数据指针，指向当前输出元素的起始位置
        scalar_t* output_ptr = output_data + i * output_width;
        // 设置输入数据指针，指向当前输入元素的起始位置
        const scalar_t* input_ptr = input_data + c * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width;

        // 调用循环函数，执行数据复制或处理，根据是否是正向填充
        loop(output_ptr, input_ptr, p.is_padding_positive_width);
        // 更新数据索引，准备处理下一个元素
        data_index_step(c, channels, od, output_depth, oh, output_height);
      }
    });
  } else {
    // 如果输入维度不是 1d、2d 或 3d，则抛出错误
    TORCH_INTERNAL_ASSERT(false, "expect input dim to be 1d, 2d or 3d.");
  }

  // 如果输出张量不是连续的，则进行拷贝操作使其连续
  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
// 结束之前的函数定义
}

// 根据输出和输入张量的内存格式，将输入张量转换为连续的指定内存格式
template <typename scalar_t, typename PaddingType>
void cpu_padding_channels_last(
    const Tensor& output_,                    // 输出张量的引用
    const Tensor& input_,                     // 输入张量的引用
    PaddingParams& p) {                       // 填充参数的引用

  // 根据填充参数的维度确定内存格式为ChannelsLast或ChannelsLast3d
  auto memory_format = p.ndim == 2
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::ChannelsLast3d;

  // 将输入张量转换为连续的指定内存格式
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  // 获取输入和输出张量的数据指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // 获取填充参数中的批次数和通道数
  int64_t nbatch = p.nbatch;
  int64_t channels = p.channels;

  // 获取填充参数中的维度和相关尺寸
  int ndim = p.ndim;
  int64_t input_depth = ndim == 3 ? p.ishape[ndim - 3] : 1;
  int64_t input_height = ndim >= 2 ? p.ishape[ndim - 2] : 1;
  int64_t input_width = p.ishape[ndim - 1];
  int64_t output_depth = ndim == 3 ? p.oshape[ndim - 3] : 1;
  int64_t output_height = ndim >= 2 ? p.oshape[ndim - 2] : 1;
  int64_t output_width = p.oshape[ndim - 1];
  int64_t pad_d = ndim == 3 ? p.pads[ndim - 3] : 0;
  int64_t pad_h = ndim >= 2 ? p.pads[ndim - 2] : 0;
  int64_t pad_w = p.pads[ndim - 1];
  int64_t offset_d = ndim == 3 ? p.offsets[ndim - 3] : 0;
  int64_t offset_h = ndim >= 2 ? p.offsets[ndim - 2] : 0;
  int64_t offset_w = p.offsets[ndim - 1];

  // 根据输入张量的维度选择不同的并行方式和索引初始化方法
  if (ndim == 2) {
    // 在N,H,W上并行执行，C上向量化
    at::parallel_for(0, nbatch * output_height * output_width, 1, [&](int64_t begin, int64_t end) {
      int64_t n{0}, oh{0}, ow{0};
      data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

      // 遍历输出张量的元素并进行填充
      for (const auto i : c10::irange(begin, end)) {
        // 计算输入张量中的索引
        int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
        int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);

        // 获取当前输出张量元素的指针和对应输入张量元素的指针，然后复制数据
        scalar_t* output_ptr = output_data + i * channels;
        const scalar_t* input_ptr = input_data + (n * input_height * input_width + ih * input_width + iw) * channels;
        copy_stub(output_ptr, input_ptr, channels);

        // 更新索引
        data_index_step(n, nbatch, oh, output_height, ow, output_width);
      }
    });
  } else if (ndim == 3) {
    // 在N,D,H,W上并行执行，C上向量化
    at::parallel_for(0, nbatch * output_depth * output_height * output_width, 1, [&](int64_t begin, int64_t end) {
      int64_t n{0}, od{0}, oh{0}, ow{0};
      data_index_init(begin, n, nbatch, od, output_depth, oh, output_height, ow, output_width);

      // 遍历输出张量的元素并进行填充
      for (const auto i : c10::irange(begin, end)) {
        // 计算输入张量中的索引
        int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
        int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
        int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);

        // 获取当前输出张量元素的指针和对应输入张量元素的指针，然后复制数据
        scalar_t* output_ptr = output_data + i * channels;
        const scalar_t* input_ptr = input_data + (n * input_depth * input_height * input_width +
            id * input_height * input_width + ih * input_width + iw) * channels;
        copy_stub(output_ptr, input_ptr, channels);

        // 更新索引
        data_index_step(n, nbatch, od, output_depth, oh, output_height, ow, output_width);
      }
    });
  }
}
    });
  } else {
    // 如果条件不成立，即输入维度不是2D或3D，则触发内部断言错误，打印错误信息
    TORCH_INTERNAL_ASSERT(false, "expect input dim to be 2d or 3d.");
  }

  // 如果输出张量不是按指定的内存格式（memory_format）连续存储
  if (!output_.is_contiguous(memory_format)) {
    // 将按照默认顺序存储的输出张量复制为按照指定内存格式存储的输出张量
    output_.copy_(output);
  }
// 定义了一个函数，用于在 CPU 上进行反向填充操作，用于梯度计算
template <typename scalar_t, typename PaddingType>
void cpu_padding_backward(
    const Tensor& grad_input_,  // 输入的梯度张量
    const Tensor& grad_output_,  // 输出的梯度张量
    PaddingParams& p) {  // 填充参数结构体的引用

  // 保证梯度张量是连续的，以便后续操作
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  // 获取梯度张量数据的指针
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  // 将 nbatch 和 channels 折叠成单个维度，用于通道优先的计算
  int64_t channels = p.nbatch * p.channels;

  // 获取填充维度
  int ndim = p.ndim;
  int64_t input_depth = ndim == 3 ? p.ishape[ndim - 3] : 1;
  int64_t input_height = ndim >=2 ? p.ishape[ndim - 2] : 1;
  int64_t input_width = p.ishape[ndim - 1];
  int64_t output_depth = ndim == 3 ? p.oshape[ndim - 3] : 1;
  int64_t output_height = ndim >= 2 ? p.oshape[ndim - 2] : 1;
  int64_t output_width = p.oshape[ndim - 1];
  int64_t pad_d = ndim == 3 ? p.pads[ndim - 3] : 0;
  int64_t pad_h = ndim >= 2 ? p.pads[ndim - 2] : 0;
  int64_t pad_w = p.pads[ndim - 1];
  int64_t offset_d = ndim == 3 ? p.offsets[ndim - 3] : 0;
  int64_t offset_h = ndim >= 2 ? p.offsets[ndim - 2] : 0;
  int64_t offset_w = p.offsets[ndim - 1];

  // 根据维度不同进行不同的填充反向计算方式
  if (ndim == 1) {
    // 在 N,C 上并行，在 W 上顺序执行
    at::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
      for (const auto c : c10::irange(begin, end)) {
        for (const auto ow : c10::irange(output_width)) {
          // 计算输入张量的索引位置
          int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
          // 累加梯度值到输入张量的相应位置
          grad_input_data[c * input_width + iw] += grad_output_data[c * output_width + ow];
        }
      }
    });
  } else if (ndim == 2) {
    // 在 N,C 上并行，在 H,W 上顺序执行
    at::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
      for (const auto c : c10::irange(begin, end)) {
        const scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;
        scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;

        for (const auto oh : c10::irange(output_height)) {
          // 计算输入张量的索引位置
          int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
          for (const auto ow : c10::irange(output_width)) {
            int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
            // 累加梯度值到输入张量的相应位置
            grad_input_ptr[ih * input_width + iw] += grad_output_ptr[oh * output_width + ow];
          }
        }
      }
    });
  } else if (p.ndim == 3) {
    // 在 N,C 上并行，在 D,H,W 上顺序执行
    // 使用 ATen 库中的并行函数 parallel_for 进行多线程并行处理，循环范围是 [0, channels)，步长为 1
    at::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
      // 遍历每个通道 c，其中 c 范围是 [begin, end)
      for (const auto c : c10::irange(begin, end)) {
        // 计算梯度输出在当前通道 c 上的起始指针位置
        const scalar_t* grad_output_ptr = grad_output_data + c * output_depth * output_height * output_width;
        // 计算梯度输入在当前通道 c 上的起始指针位置
        scalar_t* grad_input_ptr = grad_input_data + c * input_depth * input_height * input_width;

        // 遍历输出张量的深度维度 od
        for (const auto od : c10::irange(output_depth)) {
          // 根据填充类型计算输入张量对应的深度维度 id
          int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
          // 遍历输出张量的高度维度 oh
          for (const auto oh : c10::irange(output_height)) {
            // 根据填充类型计算输入张量对应的高度维度 ih
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            // 遍历输出张量的宽度维度 ow
            for (const auto ow : c10::irange(output_width)) {
              // 根据填充类型计算输入张量对应的宽度维度 iw
              int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
              // 将梯度输出的对应位置加到梯度输入的对应位置上
              grad_input_ptr[id * input_height * input_width + ih * input_width + iw] +=
                  grad_output_ptr[od * output_height * output_width + oh * output_width + ow];
            }
          }
        }
      }
    });
  } else {
    // 如果输入张量的数据不是连续的，抛出内部断言错误，提示期望输入维度为 1D、2D 或 3D
    TORCH_INTERNAL_ASSERT(false, "expect input dim to be 1d, 2d, or 3d.");
  }

  // 如果梯度输入张量不是连续的，则使用其副本 grad_input 替换 grad_input_
  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}



template <typename scalar_t, typename PaddingType>
void cpu_padding_backward_channels_last(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    PaddingParams& p) {

// 定义一个模板函数，用于计算通道在最后的情况下的反向填充操作。接受梯度输入、梯度输出张量和填充参数结构体作为参数。


  auto memory_format = p.ndim == 2
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::ChannelsLast3d;

// 根据输入张量的维度选择内存格式，如果维度为2，则使用ChannelsLast格式；如果维度为3，则使用ChannelsLast3d格式。


  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

// 将输入的梯度张量和输出的梯度张量按照选择的内存格式进行连续化处理，确保数据在内存中的连续存储。


  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();

// 获取处理后的输入和输出梯度张量的数据指针，分别用于读写操作。


  int64_t nbatch = p.nbatch;
  int64_t channels = p.channels;

  int ndim = p.ndim;
  int64_t input_depth = ndim == 3 ? p.ishape[ndim - 3] : 1;
  int64_t input_height = ndim >= 2 ? p.ishape[ndim - 2] : 1;
  int64_t input_width = p.ishape[ndim - 1];
  int64_t output_depth = ndim == 3 ? p.oshape[ndim - 3] : 1;
  int64_t output_height = ndim >= 2 ? p.oshape[ndim - 2] : 1;
  int64_t output_width = p.oshape[ndim - 1];
  int64_t pad_d = ndim == 3 ? p.pads[ndim - 3] : 0;
  int64_t pad_h = ndim >= 2 ? p.pads[ndim - 2] : 0;
  int64_t pad_w = p.pads[ndim - 1];
  int64_t offset_d = ndim == 3 ? p.offsets[ndim - 3] : 0;
  int64_t offset_h = ndim >= 2 ? p.offsets[ndim - 2] : 0;
  int64_t offset_w = p.offsets[ndim - 1];

// 根据填充参数结构体 `p` 中的信息，获取批次大小、通道数、输入和输出的深度、高度、宽度等维度相关的参数。


  if (ndim == 2) {
    // parallel on N, sequential on H,W, vectorize on C
    at::parallel_for(0, nbatch, 1, [&](int64_t begin, int64_t end) {
      for (const auto n : c10::irange(begin, end)) {
        for (const auto oh : c10::irange(output_height)) {
          int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
          for (const auto ow : c10::irange(output_width)) {
            int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
            scalar_t* grad_input_ptr = grad_input_data +
                (n * input_height * input_width + ih * input_width + iw) * channels;
            const scalar_t* grad_output_ptr = grad_output_data +
                (n * output_height * output_width + oh * output_width + ow) * channels;
            add_stub(grad_input_ptr, grad_output_ptr, channels);
          }
        }
      }
    });
  } else if (ndim == 3) {
    // parallel on N, sequential on D,H,W, vectorize on C

// 如果维度为2，则对批次N并行处理，对高度H和宽度W进行顺序处理，在通道C上进行向量化操作。
    # 使用 ATen 库的并行执行函数 `parallel_for`，对 `nbatch` 次迭代进行并行处理
    at::parallel_for(0, nbatch, 1, [&](int64_t begin, int64_t end) {
      # 遍历 batch 内的每个样本 `n`
      for (const auto n : c10::irange(begin, end)) {
        # 遍历输出的深度维度 `output_depth`
        for (const auto od : c10::irange(output_depth)) {
          # 根据填充类型计算输入的深度维度 `id`
          int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
          # 遍历输出的高度维度 `output_height`
          for (const auto oh : c10::irange(output_height)) {
            # 根据填充类型计算输入的高度维度 `ih`
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            # 遍历输出的宽度维度 `output_width`
            for (const auto ow : c10::irange(output_width)) {
              # 根据填充类型计算输入的宽度维度 `iw`
              int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
              # 计算输入梯度张量的指针位置
              scalar_t* grad_input_ptr = grad_input_data +
                  (n * input_depth * input_height * input_width + id * input_height * input_width +
                   ih * input_width + iw) * channels;
              # 计算输出梯度张量的指针位置
              const scalar_t* grad_output_ptr = grad_output_data +
                  (n * output_depth * output_height * output_width + od * output_height * output_width +
                   oh * output_width + ow) * channels;
              # 调用 `add_stub` 函数，将输出梯度加到输入梯度上
              add_stub(grad_input_ptr, grad_output_ptr, channels);
            }
          }
        }
      }
    });
    # 如果输入梯度不是按照指定内存格式（`memory_format`）连续存储，则执行复制操作
    } else {
      # 抛出内部断言错误，表明期望输入的维度为 2D 或 3D
      TORCH_INTERNAL_ASSERT(false, "expect input dim to be 2d or 3d.");
    }
    
    # 如果 `grad_input_` 张量不按照指定内存格式（`memory_format`）连续存储，则执行复制操作
    if (!grad_input_.is_contiguous(memory_format)) {
      grad_input_.copy_(grad_input);
    }
}

// 对于非批处理模式下的4维输入，将被视为CDHW格式中的连续格式
at::MemoryFormat padding_memory_format_3d(const Tensor& input) {
  return input.dim() == 4 ? at::MemoryFormat::Contiguous : input.suggest_memory_format();
}

// 反射填充的1维核心实现
void reflection_pad1d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 构造填充参数对象
  PaddingParams param{input, output, padding};
  // 如果输入是量化的
  if (input.is_quantized()) {
    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad1d", [&] {
      // 调用CPU填充函数，使用反射填充方式
      cpu_padding<scalar_t, ReflectionPad>(output, input, param);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
        "reflection_pad1d", [&] {
      // 调用CPU填充函数，使用反射填充方式
      cpu_padding<scalar_t, ReflectionPad>(output, input, param);
    });
  }
}

// 反射填充的1维反向传播核心实现
void reflection_pad1d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 构造填充参数对象
  PaddingParams param{grad_input, grad_output, padding};
  // 根据输出梯度的数据类型分发到相应的CPU反向填充函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
      "reflection_pad1d_backward", [&] {
    cpu_padding_backward<scalar_t, ReflectionPad>(grad_input, grad_output, param);
  });
}

// 反射填充的2维核心实现
void reflection_pad2d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 构造填充参数对象
  PaddingParams param{input, output, padding};
  // 如果输入是量化的
  if (input.is_quantized()) {
    // 原始的量化实现不支持通道最后的格式，
    // 如果有意为之，请在此处进行切换。
    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
      // 调用CPU填充函数，使用反射填充方式
      cpu_padding<scalar_t, ReflectionPad>(output, input, param);
    });
  } else {
    switch (input.suggest_memory_format()) {
      case at::MemoryFormat::Contiguous: {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
            "reflection_pad2d", [&] {
          // 调用CPU填充函数，使用反射填充方式
          cpu_padding<scalar_t, ReflectionPad>(output, input, param);
        });
        break;
      }
      case at::MemoryFormat::ChannelsLast: {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
            "reflection_pad2d_channels_last", [&]{
          // 调用CPU填充函数，使用通道最后的反射填充方式
          cpu_padding_channels_last<scalar_t, ReflectionPad>(output, input, param);
        });
        break;
      }
      default:
        // 如果不支持的内存格式，则抛出错误
        TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
    }
  }
}

// 反射填充的2维反向传播核心实现
void reflection_pad2d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 构造填充参数对象
  PaddingParams param{grad_input, grad_output, padding};
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "reflection_pad2d_backward", [&] {
        // 调用CPU反向填充函数，使用反射填充方式
        cpu_padding_backward<scalar_t, ReflectionPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 其他内存格式的情况下，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only Contiguous");
  }
}
    // 在内存格式为ChannelsLast时执行以下代码块
    case at::MemoryFormat::ChannelsLast: {
      // 使用AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1宏根据grad_output的数据类型进行分发，执行lambda表达式
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "reflection_pad2d_backward_channels_last", [&]{
        // 调用cpu_padding_backward_channels_last函数，对grad_input和grad_output进行反向填充操作，使用ReflectionPad类型的填充参数param
        cpu_padding_backward_channels_last<scalar_t, ReflectionPad>(grad_input, grad_output, param);
      });
      // 退出当前case语句块
      break;
    }
    // 如果内存格式不是ChannelsLast，则执行以下代码块
    default:
      // 抛出错误，指示不支持当前的内存格式，只支持ChannelsLast和Contiguous两种
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
// 定义函数 reflection_pad3d_kernel_impl，实现 3D 反射填充操作
void reflection_pad3d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 创建填充参数对象 param，包括输入、输出张量及填充信息
  PaddingParams param{input, output, padding};
  // 根据输入张量的内存格式选择操作方式
  switch (padding_memory_format_3d(input)) {
    // 如果内存格式为连续存储
    case at::MemoryFormat::Contiguous: {
      // 使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 遍历所有数据类型
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input.scalar_type(),
          "reflection_pad3d", [&] {
        // 调用 CPU 实现的反射填充函数，根据输入、输出张量及填充参数 param
        cpu_padding<scalar_t, ReflectionPad>(output, input, param);
      });
      break;
    }
    // 如果内存格式为 ChannelsLast3d
    case at::MemoryFormat::ChannelsLast3d: {
      // 使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 遍历所有数据类型
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, input.scalar_type(),
          "reflection_pad3d_channels_last", [&]{
        // 调用 CPU 实现的通道最后一维反射填充函数，根据输入、输出张量及填充参数 param
        cpu_padding_channels_last<scalar_t, ReflectionPad>(output, input, param);
      });
      break;
    }
    // 如果出现不支持的内存格式，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

// 定义函数 reflection_pad3d_backward_kernel_impl，实现 3D 反射填充的反向传播
void reflection_pad3d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 创建填充参数对象 param，包括梯度输入、梯度输出张量及填充信息
  PaddingParams param{grad_input, grad_output, padding};
  // 根据梯度输出张量的内存格式选择操作方式
  switch (padding_memory_format_3d(grad_output)) {
    // 如果内存格式为连续存储
    case at::MemoryFormat::Contiguous: {
      // 使用宏 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 遍历浮点型数据类型
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, grad_output.scalar_type(),
          "reflection_pad3d_backward", [&] {
        // 调用 CPU 实现的反射填充反向传播函数，根据梯度输入、梯度输出张量及填充参数 param
        cpu_padding_backward<scalar_t, ReflectionPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果内存格式为 ChannelsLast3d
    case at::MemoryFormat::ChannelsLast3d: {
      // 使用宏 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 遍历浮点型数据类型
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, grad_output.scalar_type(),
          "reflection_pad3d_backward_channels_last", [&]{
        // 调用 CPU 实现的通道最后一维反射填充反向传播函数，根据梯度输入、梯度输出张量及填充参数 param
        cpu_padding_backward_channels_last<scalar_t, ReflectionPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果出现不支持的内存格式，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

// replication padding
// 定义函数 replication_pad1d_kernel_impl，实现 1D 复制填充操作
void replication_pad1d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 创建填充参数对象 param，包括输入、输出张量及填充信息
  PaddingParams param{input, output, padding};
  // 使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND 遍历所有数据类型
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
      "replication_pad1d", [&] {
    // 调用 CPU 实现的复制填充函数，根据输入、输出张量及填充参数 param
    cpu_padding<scalar_t, ReplicationPad>(output, input, param);
  });
}

// 定义函数 replication_pad1d_backward_kernel_impl，实现 1D 复制填充的反向传播
void replication_pad1d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 创建填充参数对象 param，包括梯度输入、梯度输出张量及填充信息
  PaddingParams param{grad_input, grad_output, padding};
  // 使用宏 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1 遍历浮点型数据类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
      "replication_pad1d_backward", [&] {
    // 调用 CPU 实现的复制填充反向传播函数，根据梯度输入、梯度输出张量及填充参数 param
    cpu_padding_backward<scalar_t, ReplicationPad>(grad_input, grad_output, param);
  });
}

// 定义函数 replication_pad2d_kernel_impl，实现 2D 复制填充操作
void replication_pad2d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 创建填充参数对象 param，包括输入、输出张量及填充信息
  PaddingParams param{input, output, padding};
  // 根据输入张量推荐的内存格式选择操作方式
  switch (input.suggest_memory_format()) {
    // 对于内存格式为连续的情况执行以下操作
    case at::MemoryFormat::Contiguous: {
      // 使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(), "replication_pad2d", [&] {...})，根据输入张量的数据类型分发执行 replication_pad 函数，该函数在 CPU 上进行填充操作
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
          "replication_pad2d", [&] {
        cpu_padding<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    // 对于内存格式为 ChannelsLast 的情况执行以下操作
    case at::MemoryFormat::ChannelsLast: {
      // 使用模板函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(), "replication_pad2d_channels_last", [&] {...})，根据输入张量的数据类型分发执行 cpu_padding_channels_last 函数，该函数在 CPU 上进行通道最后填充操作
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
          "replication_pad2d_channels_last", [&]{
        cpu_padding_channels_last<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    // 默认情况下，若内存格式不支持，则抛出错误信息
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// 定义了匿名命名空间，用于限定内部函数和变量的作用域
void replication_pad2d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 创建填充参数对象，用于处理梯度输入、梯度输出和填充信息
  PaddingParams param{grad_input, grad_output, padding};
  // 根据梯度输出的内存格式选择相应的处理方式
  switch (grad_output.suggest_memory_format()) {
    // 如果梯度输出内存格式为连续存储
    case at::MemoryFormat::Contiguous: {
      // 根据梯度输出的数据类型分发 CPU 处理函数，处理反向填充操作
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "replication_pad2d_backward", [&] {
        cpu_padding_backward<scalar_t, ReplicationPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果梯度输出内存格式为通道最后存储
    case at::MemoryFormat::ChannelsLast: {
      // 根据梯度输出的数据类型分发 CPU 处理函数，处理通道最后存储的反向填充操作
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "replication_pad2d_backward_channels_last", [&]{
        cpu_padding_backward_channels_last<scalar_t, ReplicationPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果遇到不支持的内存格式，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

// replication_pad3d_kernel_impl 函数的定义，处理 3D 复制填充操作
void replication_pad3d_kernel_impl(const Tensor& output, const Tensor& input, IntArrayRef padding) {
  // 创建填充参数对象，用于处理输入、输出和填充信息
  PaddingParams param{input, output, padding};
  // 根据输入张量的内存格式选择相应的处理方式
  switch (padding_memory_format_3d(input)) {
    // 如果输入内存格式为连续存储
    case at::MemoryFormat::Contiguous: {
      // 根据输入张量的数据类型分发 CPU 处理函数，处理 3D 填充操作
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
          "replication_pad3d", [&] {
        cpu_padding<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    // 如果输入内存格式为 3D 通道最后存储
    case at::MemoryFormat::ChannelsLast3d: {
      // 根据输入张量的数据类型分发 CPU 处理函数，处理通道最后存储的 3D 填充操作
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, input.scalar_type(),
          "replication_pad3d_channels_last", [&]{
        cpu_padding_channels_last<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    // 如果遇到不支持的内存格式，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

// replication_pad3d_backward_kernel_impl 函数的定义，处理 3D 反向复制填充操作
void replication_pad3d_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding) {
  // 创建填充参数对象，用于处理梯度输入、梯度输出和填充信息
  PaddingParams param{grad_input, grad_output, padding};
  // 根据梯度输出的内存格式选择相应的处理方式
  switch (padding_memory_format_3d(grad_output)) {
    // 如果梯度输出内存格式为连续存储
    case at::MemoryFormat::Contiguous: {
      // 根据梯度输出的数据类型分发 CPU 处理函数，处理反向 3D 复制填充操作
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "replication_pad3d_backward", [&] {
        cpu_padding_backward<scalar_t, ReplicationPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果梯度输出内存格式为 3D 通道最后存储
    case at::MemoryFormat::ChannelsLast3d: {
      // 根据梯度输出的数据类型分发 CPU 处理函数，处理通道最后存储的反向 3D 复制填充操作
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, grad_output.scalar_type(),
          "replication_pad3d_backward_channels_last", [&]{
        cpu_padding_backward_channels_last<scalar_t, ReplicationPad>(grad_input, grad_output, param);
      });
      break;
    }
    // 如果遇到不支持的内存格式，抛出错误
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

} // 匿名命名空间结束，限定内部函数和变量的作用域

// reflection_pad1d_kernel 的注册调度，指向 reflection_pad1d_kernel_impl 函数
REGISTER_DISPATCH(reflection_pad1d_kernel, &reflection_pad1d_kernel_impl);
// reflection_pad1d_backward_kernel 的注册调度，指向 reflection_pad1d_backward_kernel_impl 函数
REGISTER_DISPATCH(reflection_pad1d_backward_kernel, &reflection_pad1d_backward_kernel_impl);
# 注册反射填充二维卷积操作的分发函数
REGISTER_DISPATCH(reflection_pad2d_kernel, &reflection_pad2d_kernel_impl);

# 注册反射填充二维卷积反向传播操作的分发函数
REGISTER_DISPATCH(reflection_pad2d_backward_kernel, &reflection_pad2d_backward_kernel_impl);

# 注册反射填充三维卷积操作的分发函数
REGISTER_DISPATCH(reflection_pad3d_kernel, &reflection_pad3d_kernel_impl);

# 注册反射填充三维卷积反向传播操作的分发函数
REGISTER_DISPATCH(reflection_pad3d_backward_kernel, &reflection_pad3d_backward_kernel_impl);

# 注册复制填充一维卷积操作的分发函数
REGISTER_DISPATCH(replication_pad1d_kernel, &replication_pad1d_kernel_impl);

# 注册复制填充一维卷积反向传播操作的分发函数
REGISTER_DISPATCH(replication_pad1d_backward_kernel, &replication_pad1d_backward_kernel_impl);

# 注册复制填充二维卷积操作的分发函数
REGISTER_DISPATCH(replication_pad2d_kernel, &replication_pad2d_kernel_impl);

# 注册复制填充二维卷积反向传播操作的分发函数
REGISTER_DISPATCH(replication_pad2d_backward_kernel, &replication_pad2d_backward_kernel_impl);

# 注册复制填充三维卷积操作的分发函数
REGISTER_DISPATCH(replication_pad3d_kernel, &replication_pad3d_kernel_impl);

# 注册复制填充三维卷积反向传播操作的分发函数
REGISTER_DISPATCH(replication_pad3d_backward_kernel, &replication_pad3d_backward_kernel_impl);
```