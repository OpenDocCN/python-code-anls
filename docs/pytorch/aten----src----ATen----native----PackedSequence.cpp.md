# `.\pytorch\aten\src\ATen\native\PackedSequence.cpp`

```py
// 定义宏，仅启用方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量操作的头文件
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含常用的张量函数和本地函数头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含特定操作的本地函数头文件
#include <ATen/ops/_pack_padded_sequence_backward_native.h>
#include <ATen/ops/_pack_padded_sequence_native.h>
#include <ATen/ops/_pad_packed_sequence_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/pad_sequence_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <c10/util/irange.h> // 包含 C10 库中的整数范围迭代器

namespace at::native {

// 静态函数，用于检查传入的张量是否为长整型的一维 CPU 张量
static void checkLongTensor(const Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 1 && tensor.device().type() == at::kCPU && tensor.scalar_type() == at::kLong,
           "'lengths' argument should be a 1D CPU int64 tensor, but got ",
            tensor.dim(), "D ", tensor.device().str(), " ", tensor.scalar_type(), " tensor");
}

// 此方法返回元组 `(data, batch_sizes)`，这些值随后传递给 `PackedSequence` 构造函数。
// `data` 可能位于任意设备上并具有任意数据类型，但 `batch_sizes` 必须是 CPU 上的 int64 张量。
// 参见 NOTE [ device and dtype of a PackedSequence ]
std::tuple<Tensor, Tensor> _pack_padded_sequence(const Tensor& _input, const Tensor& _lengths, bool batch_first) {
  TORCH_CHECK(_input.numel() > 0, "Cannot pack empty tensors.");
  auto input = batch_first ? _input.transpose(0, 1) : _input; // 根据 batch_first 参数转置输入张量
  auto lengths_t = _lengths.contiguous(); // 返回一个连续的 `_lengths` 张量
  checkLongTensor(lengths_t); // 检查 `_lengths` 张量是否符合要求

  int64_t batch_size = input.size(1); // 获取输入张量的 batch 大小
  int64_t * lengths = lengths_t.data_ptr<int64_t>(); // 获取 `_lengths` 数据指针

  TORCH_CHECK(lengths_t.size(0) == batch_size,
           "Expected `len(lengths)` to be equal to batch_size, but got ", lengths_t.size(0),
           " (batch_size=", batch_size, ")");
  TORCH_CHECK(lengths[batch_size - 1] > 0,
           "Length of all samples has to be greater than 0, but found an element "
           "in 'lengths' that is <= 0");
  for (const auto i : c10::irange(batch_size - 1)) {
    if (lengths[batch_size - 1 - i] > lengths[batch_size - 2 - i]) {
      // 注意：如果 enforce_sorted=False，则 sortedness 检查由 Python 层实现，
      // 但此处存在检查。如果 enforce_sorted=False，则此错误不应出现。
      AT_ERROR("`lengths` array must be sorted in decreasing order when "
               "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
               "to pack_padded_sequence and/or pack_sequence to sidestep this "
               "requirement if you do not need ONNX exportability.");
    }
  }

  std::vector<at::Tensor> steps; // 创建一个张量向量 steps
  steps.reserve(batch_size); // 预留足够的空间以容纳 batch_size 个张量
  at::Tensor batch_sizes_t = at::empty(lengths[0], _lengths.options()); // 创建一个与 `_lengths` 相同大小和类型的空张量
  int64_t * batch_sizes = batch_sizes_t.mutable_data_ptr<int64_t>(); // 获取 batch_sizes_t 的可变数据指针

  std::vector<int64_t> step_shape; // 定义一个整型向量 step_shape
  {
    auto input_sizes = input.sizes(); // 获取输入张量的尺寸
    step_shape.reserve(input_sizes.size()); // 预留足够空间以容纳 input_sizes 的维度数
    auto s_input_sizes = input_sizes.slice(2); // 获取输入张量维度从第二个开始的切片
    step_shape.push_back(-1); // 在 step_shape 的开头添加 -1
    // 将 s_input_sizes 开始到结束的内容插入到 step_shape 的末尾
    step_shape.insert(step_shape.end(), s_input_sizes.begin(), s_input_sizes.end());
  }

  // 理解此循环的关键在于假设输入是一个填充的二维数组，如下所示（x = 有效条目，. = 填充）
  //
  //  1 1 1 1 1
  //  2 2 2 . .
  //  2 2 2 . .
  //  4 . . . .
  //  4 . . . .
  //
  // 其中垂直维度对应时间，水平维度对应批次。在此示例中，长度数组将等于 [5, 3, 3, 1, 1]，
  // 我们将以逆序方式遍历它们（从最右侧列到最左侧）。我们希望避免在每个时间步骤都急切地对输入进行切片，
  // 而是等待长度增加的时刻。在这个示例中，这将发生在第一、第二和第四步。然后，我们切片出对应于此长度且尚未切片的整个输入块
  // （上面数组中每个元素的切片步骤都有注释）。你可以将其想象成我们从最短的序列开始扫描，并且每当我们意识到某列下面还有更多元素时，
  // 我们就降低计数器（prev_l），并将新的块附加到输出中。
  int64_t prev_l = 0;
  for (const auto i : c10::irange(batch_size)) {
    int64_t l = lengths[batch_size - 1 - i];
    // 如果当前长度大于前一个长度prev_l，则进行处理
    if (l > prev_l) {
      auto current_batch_size = batch_size - i;
      // 在第0维度切片，从prev_l到l，第1维度切片从0到current_batch_size，然后重新视图化为step_shape
      steps.push_back(input.slice(0, prev_l, l).slice(1, 0, current_batch_size).contiguous().view(step_shape));
      // 对于(l - prev_l)个元素，将当前批次大小添加到batch_sizes
      for (int64_t j = 0; j < (l - prev_l); ++j) {
        (*batch_sizes++) = current_batch_size;
      }
      // 更新prev_l为当前长度l
      prev_l = l;
    }
    // 断言当前长度l大于等于前一个长度prev_l
    TORCH_CHECK(l >= prev_l);
  }

  // 返回一个元组，包含所有步骤的连接结果和batch_sizes_t
  return std::make_tuple(at::cat(steps), batch_sizes_t);
}

// `grad` could be on arbitrary device and of arbitrary dtype, but `_batch_sizes`
// is guaranteed to be a CPU int64 tensor.
// See NOTE [ device and dtype of a PackedSequence ]
// 定义一个函数 `_pack_padded_sequence_backward_symint`，用于计算反向的填充序列的梯度
Tensor _pack_padded_sequence_backward_symint(const Tensor& grad, c10::SymIntArrayRef input_size, const Tensor& _batch_sizes, bool batch_first) {
  // 将输入的 `input_size` 转换为标准的 `std::vector` 形式
  std::vector<c10::SymInt> input_size_after_t = input_size.vec();
  // 如果 `batch_first` 参数为真，则交换 `input_size_after_t` 的第一个和第二个元素
  if (batch_first) {
    TORCH_CHECK(input_size.size() >= 2);
    std::swap(input_size_after_t[0], input_size_after_t[1]);
  }
  // 创建一个和 `grad` 大小相同的零填充的 `grad_input` 张量
  auto grad_input = at::zeros_symint(input_size_after_t, grad.options());
  // 将 `_batch_sizes` 张量变成连续的张量
  auto batch_sizes_t = _batch_sizes.contiguous();
  // 检查 `_batch_sizes` 是否为 int64 类型的张量
  checkLongTensor(batch_sizes_t);

  // 初始化偏移量为 0
  int64_t offset = 0;
  // NOTE: this op advertises as CompositeImplicitAutograd, but uses data_ptr().
  // we should fix this.
  // 获取 `batch_sizes_t` 的第一个维度大小，即序列的最大长度
  auto max_seq_len = batch_sizes_t.size(0);
  // 获取 `batch_sizes_t` 的数据指针，以获取每个批次的大小
  int64_t * batch_sizes = batch_sizes_t.data_ptr<int64_t>();
  // 遍历每个序列长度
  for (const auto i : c10::irange(max_seq_len)) {
    // 将 `grad` 的部分切片复制到 `grad_input` 的相应部分
    grad_input[i].slice(0, 0, batch_sizes[i]).copy_(grad.slice(0, offset, offset + batch_sizes[i]));
    // 更新偏移量
    offset += batch_sizes[i];
  }

  // 如果 `batch_first` 为真，则对 `grad_input` 进行转置
  if (batch_first) {
    grad_input = grad_input.transpose(0, 1);
  }

  // 返回计算得到的 `grad_input` 张量
  return grad_input;
}

// 定义一个函数 `_pad_packed_sequence`，用于填充打包的序列
std::tuple<Tensor, Tensor> _pad_packed_sequence(const Tensor& data, const Tensor& _batch_sizes, bool batch_first, const Scalar& padding_value, int64_t total_length) {
  // 将 `_batch_sizes` 张量变成连续的张量
  auto batch_sizes_t = _batch_sizes.contiguous();
  // 检查 `_batch_sizes` 是否为 int64 类型的张量
  checkLongTensor(batch_sizes_t);

  // 获取 `batch_sizes_t` 的数据指针，以获取每个批次的大小
  int64_t * batch_sizes = batch_sizes_t.data_ptr<int64_t>();
  // 获取第一个批次的大小
  int64_t max_batch_size = batch_sizes[0];
  // 获取实际序列长度的最大值
  int64_t max_real_seq_length = batch_sizes_t.size(0);
  // 初始化最大序列长度为实际序列长度
  int64_t max_seq_length = max_real_seq_length;
  // 如果 `total_length` 大于 0，则根据需要更新最大序列长度
  if (total_length > 0) {
    TORCH_CHECK(total_length >= max_seq_length,
             "Expected total_length to be at least the length of the longest "
             "sequence in input, but got total_length=", total_length, " and "
             "max sequence length being ", max_seq_length);
    max_seq_length = total_length;
  }

  // 初始化输出张量的大小向量 `output_size`
  std::vector<int64_t> output_size; // == [max_seq_length, max_batch_size, *var_data.size()[1:]]
  {
    output_size.reserve(data.dim() + 1);
    output_size.push_back(max_seq_length);
    output_size.push_back(max_batch_size);
    auto s_data_size = data.sizes().slice(1);
    output_size.insert(output_size.end(), s_data_size.begin(), s_data_size.end());
  }
  // 使用指定的填充值和数据选项创建一个全填充的输出张量 `output`
  auto output = at::full(output_size, padding_value, data.options());

  // 初始化长度张量 `lengths_t`，其大小为 `max_batch_size`
  at::Tensor lengths_t = at::empty(max_batch_size, batch_sizes_t.options());
  // 获取 `lengths_t` 的数据指针
  int64_t * lengths = lengths_t.mutable_data_ptr<int64_t>() + max_batch_size - 1;
  // 初始化数据偏移量为 0，前一批次大小为 `max_batch_size`，前一个索引为 0
  int64_t data_offset = 0;
  int64_t prev_batch_size = max_batch_size;
  int64_t prev_i = 0;
  // 遍历每个序列
  for (int64_t i = 0; i <= max_real_seq_length; ++i) {
    // 获取当前批次的大小，如果是最后一个序列，则批次大小为 0
    int64_t batch_size = i != max_real_seq_length ? batch_sizes[i] : 0;
    // 如果当前批次大小与前一批次大小不同
    if (batch_size != prev_batch_size) {
      // 计算前一批次数据的长度
      int64_t l = prev_batch_size * (i - prev_i);
      // 下面的代码等效于：
      // output[prev_i:i, :prev_batch_size] = tmp.view(i - prev_i, prev_batch_size, *input.shape[2:])
      // 从数据中切片出临时数据块
      auto tmp = data.slice(0, data_offset, data_offset + l);
      // 设置临时视图大小
      tmp_view_size[0] = i - prev_i;
      tmp_view_size[1] = prev_batch_size;
      // 将临时数据块复制到输出的指定位置
      output.slice(0, prev_i, i).slice(1, 0, prev_batch_size).copy_(tmp.view(tmp_view_size));
      // 更新数据偏移量
      data_offset += l;
      // 更新前一批次结束索引
      prev_i = i;
    }
    // 计算批次大小的差值
    int64_t dec = prev_batch_size - batch_size;
    // 如果差值大于零
    if (dec > 0) {
      // 逆序迭代，将长度设为当前索引 i
      for (C10_UNUSED const auto j : c10::irange(dec)) {
        (*lengths--) = i;
      }
    }
    // 更新前一批次大小为当前批次大小
    prev_batch_size = batch_size;
  }

  // 如果设置了 batch_first 标志
  if (batch_first) {
    // 调整输出的维度顺序为 batch_first
    output = output.transpose(0, 1);
  }

  // 返回输出和长度的元组
  return std::make_tuple(output, lengths_t);
// 结束命名空间 at::native

Tensor pad_sequence(TensorList sequences, bool batch_first, double padding_value) {
  // 获取序列列表的大小
  const int64_t sequences_size = sequences.size();
  // 检查序列列表是否为空
  TORCH_CHECK(sequences_size > 0, "received an empty list of sequences");
  // 获取第一个序列的大小，不包括批处理维度
  IntArrayRef max_size = sequences[0].sizes();
  // 获取除了批处理维度之外的其他维度
  IntArrayRef trailing_dims = max_size.slice(1);
  // 计算所有序列中的最大长度
  int64_t max_len = std::max_element(
    sequences.begin(),
    sequences.end(),
    // 比较函数，用于找出序列长度最大的 Tensor
    [](const Tensor &a, const Tensor &b) {
      return a.size(0) < b.size(0);
    }
  )->size(0);

  // 定义输出 Tensor 的维度
  DimVector out_dims;
  // 根据 batch_first 参数确定输出维度的顺序
  if (batch_first) {
    out_dims = {sequences_size, max_len};
  } else {
    out_dims = {max_len, sequences_size};
  }
  // 将除了批处理维度之外的其他维度添加到输出维度中
  out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());

  // 创建指定维度的全填充 Tensor，使用指定的填充值和第一个序列的选项
  Tensor out = at::full(out_dims, padding_value, sequences[0].options());
  // 遍历每个序列
  for (const auto i : c10::irange(sequences_size)) {
    // 获取当前序列的引用
    const Tensor& currseq = sequences[i];
    // 获取当前序列的长度
    const int64_t length_i = currseq.size(0);
    // 使用索引表示法来防止对 Tensor 的重复引用
    if (batch_first) {
      // 如果 batch_first 为 true，按批处理维度在输出 Tensor 中选择子张量并复制当前序列数据
      out.select(0, i).narrow(0, 0, length_i).copy_(currseq);
    } else {
      // 如果 batch_first 为 false，按序列长度在输出 Tensor 中选择子张量并复制当前序列数据
      out.narrow(0, 0, length_i).select(1, i).copy_(currseq);
    }
  }
  // 返回填充后的输出 Tensor
  return out;
}
```