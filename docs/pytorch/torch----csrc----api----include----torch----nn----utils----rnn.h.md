# `.\pytorch\torch\csrc\api\include\torch\nn\utils\rnn.h`

```
// 防止头文件重复包含，只在第一次包含时有效
#pragma once

// 包含来自C10库的irange.h头文件和来自torch库的types.h头文件
#include <c10/util/irange.h>
#include <torch/types.h>

// 使用命名空间torch::nn::utils::rnn以便于定位和管理相关函数和类
#include <utility>

namespace torch {
namespace nn {
namespace utils {
namespace rnn {

// 在inline函数内部反转给定张量的排列顺序
inline Tensor invert_permutation(const Tensor& permutation) {
  // 如果给定张量未定义，则返回一个未定义的Tensor
  if (!permutation.defined()) {
    return torch::Tensor();
  }
  // 创建一个与permutation形状相同的空张量，使用Contiguous内存格式
  Tensor output =
      torch::empty_like(permutation, torch::MemoryFormat::Contiguous);
  // 将顺序排列的索引按照给定的permutation张量进行散列
  output.scatter_(
      0,
      permutation,
      torch::arange(0, permutation.numel(), permutation.device()));
  // 返回反转后的张量
  return output;
}

/// 存储打包序列的数据和`batch_sizes`列表。
///
/// 所有RNN模块都接受打包序列作为输入。
///
/// 注意：
///     不应手动创建此类的实例。它们应由像`pack_padded_sequence`这样的函数实例化。
///
///     batch_sizes表示批次中每个序列步骤的元素数量，而不是传递给`pack_padded_sequence`的可变序列长度。
///     例如，给定数据``abc``和``x``，:class:`PackedSequence`将包含数据``axbc``和``batch_sizes=[2,1,1]``。
///
/// 属性：
///     data (Tensor): 包含打包序列的张量
///     batch_sizes (Tensor): 包含每个序列步骤的批次大小信息的整数张量
///     sorted_indices (Tensor, optional): 描述如何从序列构造此:class:`PackedSequence`的整数张量
///     unsorted_indices (Tensor, optional): 描述如何以正确顺序恢复原始序列的整数张量
///
/// .. 注意::
///     `data`可以位于任意设备上，并且具有任意的dtype。
///     `sorted_indices`和`unsorted_indices`必须是位于与`data`相同设备上的``torch::kInt64``张量。
///
///     但是，`batch_sizes`应始终是CPU上的``torch::kInt64``张量。
///
///     这个不变量贯穿于`PackedSequence`类的整个实现中，
///     所有在libtorch中构造`PackedSequence`的函数（例如，它们只传递符合此约束的张量）。
class PackedSequence {
 public:
  // 显式构造函数，用于初始化PackedSequence实例
  explicit PackedSequence(
      Tensor data,
      Tensor batch_sizes,
      Tensor sorted_indices = {},
      Tensor unsorted_indices = {}) {
    // 注意：如果提供了unsorted_indices，它应该是sorted_indices的反转置换。
    // 这里不进行断言检查，因为PackedSequence的构造函数应仅在内部使用。
    if (!unsorted_indices.defined()) {
      unsorted_indices = invert_permutation(sorted_indices);
    }
    // 断言检查，确保batch_sizes位于CPU上
    TORCH_CHECK(
        batch_sizes.device().type() == kCPU,
        "batch_sizes should always be on CPU. "
        "Instances of PackedSequence should never be created manually. "
        "They should be instantiated by functions like pack_sequence "
        "and pack_padded_sequences in nn::utils::rnn. "
        "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence");
    data_ = std::move(data);
    batch_sizes_ = std::move(batch_sizes);
    sorted_indices_ = std::move(sorted_indices);
    unsorted_indices_ = std::move(unsorted_indices);
  }



    // 将参数 `data` 的所有权移动给成员变量 `data_`
    // 将参数 `batch_sizes` 的所有权移动给成员变量 `batch_sizes_`
    // 将参数 `sorted_indices` 的所有权移动给成员变量 `sorted_indices_`
    // 将参数 `unsorted_indices` 的所有权移动给成员变量 `unsorted_indices`
  }



  const Tensor& data() const {
    return data_;
  }



  // 返回成员变量 `data_` 的常量引用
  const Tensor& data() const {
    return data_;
  }



  const Tensor& batch_sizes() const {
    return batch_sizes_;
  }



  // 返回成员变量 `batch_sizes_` 的常量引用
  const Tensor& batch_sizes() const {
    return batch_sizes_;
  }



  const Tensor& sorted_indices() const {
    return sorted_indices_;
  }



  // 返回成员变量 `sorted_indices_` 的常量引用
  const Tensor& sorted_indices() const {
    return sorted_indices_;
  }



  const Tensor& unsorted_indices() const {
    return unsorted_indices_;
  }



  // 返回成员变量 `unsorted_indices_` 的常量引用
  const Tensor& unsorted_indices() const {
    return unsorted_indices_;
  }



  PackedSequence pin_memory() const {
    // 为什么不转换 `batch_sizes`？
    // 参见 NOTE [ device and dtype of a PackedSequence ]
    // 返回一个使用 `data_.pin_memory()` 创建的 `PackedSequence` 对象，
    // 其中 `batch_sizes_` 可能不会被转换
    return PackedSequence(
        data_.pin_memory(),
        batch_sizes_,
        sorted_indices_.defined() ? sorted_indices_.pin_memory() : Tensor(),
        unsorted_indices_.defined() ? unsorted_indices_.pin_memory() : Tensor());
  }



  PackedSequence to(TensorOptions options) const {
    // 在 `data_` 上执行 dtype 和/或 device 转换。
    //
    // 如果 `data_` Tensor 已经具有正确的 `torch::Dtype` 和 `torch::Device`，
    // 则返回 `*this`。
    // 否则，返回一个具有所需配置的副本。
    
    // 为什么不转换 `batch_sizes`？
    // 参见 NOTE [ device and dtype of a PackedSequence ]
    Tensor data = data_.to(options);
    if (data.is_same(data_)) {
      return *this;
    } else {
      // 不转发 device 或 dtype 参数，device 是从 `data.device()` 设置的
      Tensor sorted_indices = sorted_indices_.defined()
          ? sorted_indices_.to(options.device(data.device()).dtype(sorted_indices_.dtype()))
          : Tensor();
      Tensor unsorted_indices = unsorted_indices_.defined()
          ? unsorted_indices_.to(options.device(data.device()).dtype(unsorted_indices_.dtype()))
          : Tensor();
      return PackedSequence(
          std::move(data),
          batch_sizes_,
          std::move(sorted_indices),
          std::move(unsorted_indices));
    }
  }



  PackedSequence cuda() const {
    // 返回一个使用 CUDA 设备的 `PackedSequence` 对象
    return to(kCUDA);
  }



  PackedSequence cpu() const {
    // 返回一个使用 CPU 设备的 `PackedSequence` 对象
    return to(kCPU);
  }



  /// 如果 `data_` 存储在 GPU 上，则返回 true
  bool is_cuda() const {
    return data_.is_cuda();
  }



  /// 如果 `data_` 存储在固定内存中，则返回 true
  bool is_pinned() const {
    return data_.is_pinned();
  }



 private:
  Tensor data_;
  Tensor batch_sizes_;
  Tensor sorted_indices_;
  Tensor unsorted_indices_;



 private:
  // 成员变量声明
  Tensor data_;
  Tensor batch_sizes_;
  Tensor sorted_indices_;
  Tensor unsorted_indices_;
/// Pads a packed batch of variable length sequences.
///
/// It is an inverse operation to `pack_padded_sequence`.
///
/// The returned Tensor's data will be of size ``T x B x *``, where `T` is the
/// length of the longest sequence and `B` is the batch size. If ``batch_first``
/// is true, the data will be transposed into ``B x T x *`` format.
///
/// Batch elements will be ordered decreasingly by their length.
///
/// Arguments:
///     sequence (PackedSequence): batch to pad
///     batch_first (bool, optional): if ``true``, the output will be in ``B x T
///     x *``
///         format.
///     padding_value (double, optional): values for padded elements.
///     total_length (int64_t, optional): if specified, the output will be
///         padded to have this length. This argument is useful when the
///         unpadding operation requires a specific length.
///
inline Tensor pad_packed_sequence(
    PackedSequence sequence,
    bool batch_first = false,
    double padding_value = 0.0,
    int64_t total_length = 0) {
  // Unpack the PackedSequence into data and batch_sizes
  auto [data, batch_sizes, sorted_indices, unsorted_indices] = sequence;
  
  // Call the internal Torch function to pad the sequence
  auto padded = torch::_pad_packed_sequence(data, batch_sizes, batch_first,
                                            padding_value, total_length);
  
  // If there were sorted indices, unsort the padded data
  if (!unsorted_indices.empty()) {
    int64_t batch_dim = batch_first ? 0 : 1;
    padded = padded.index_select(batch_dim, unsorted_indices);
  }
  
  return padded;
}
/// pad_packed_sequence函数：对一个压缩的序列进行填充，使其具有指定的总长度。
/// 如果指定的总长度小于序列中的最大序列长度，将会抛出错误。
///
/// 参数：
///     sequence (PackedSequence): 压缩的序列对象
///     batch_first (bool, optional): 如果为true，则返回的Tensor维度为``B x T x *``；否则为``T x B x *``。
///                                   默认为false。
///     padding_value (double, optional): 填充值。默认为0.0。
///     total_length (std::optional<int64_t>): 序列的总长度，可以为null。
///
/// 返回：
///     Tuple<Tensor, Tensor>: 包含填充序列和每个序列长度的Tensor的元组。
inline std::tuple<Tensor, Tensor> pad_packed_sequence(
    PackedSequence sequence,
    bool batch_first = false,
    double padding_value = 0.0,
    std::optional<int64_t> total_length = torch::nullopt) {
  // 获取序列中最长序列的长度
  int64_t max_seq_length = sequence.batch_sizes().size(0);
  // 如果指定了total_length，则进行长度检查和调整
  if (total_length.has_value()) {
    int64_t total_length_val = total_length.value();
    // 检查总长度是否大于等于最大序列长度，否则抛出错误
    TORCH_CHECK(
        total_length_val >= max_seq_length,
        "Expected total_length to be at least the length "
        "of the longest sequence in input, but got "
        "total_length=",
        total_length_val,
        " and max sequence length being ",
        max_seq_length);
    // 调整最大序列长度为指定的总长度
    max_seq_length = total_length_val;
  }
  // 调用PyTorch的内部函数进行序列填充
  auto [padded_output, lengths] = torch::_pad_packed_sequence(
      sequence.data(),
      sequence.batch_sizes(),
      batch_first,
      padding_value,
      max_seq_length);
  // 获取未排序的索引
  const Tensor& unsorted_indices = sequence.unsorted_indices();
  // 如果存在未排序的索引，则按照其重新排序填充后的序列和长度
  if (unsorted_indices.defined()) {
    int64_t batch_dim = batch_first ? 0 : 1;
    return std::make_tuple(
        padded_output.index_select(batch_dim, unsorted_indices),
        lengths.index({unsorted_indices.cpu()}));
  }
  // 否则返回填充后的序列和长度
  return std::make_tuple(padded_output, lengths);
}

/// pad_sequence函数：使用指定的填充值对长度不同的Tensor列表进行填充。
///
/// 参数：
///     sequences (torch::ArrayRef<Tensor>): 需要填充的变长序列列表。
///     batch_first (bool, optional): 如果为true，则输出的Tensor维度为``B x T x *``；否则为``T x B x *``。
///                                   默认为false。
///     padding_value (double, optional): 填充值。默认为0.0。
///
/// 返回：
///     Tensor: 填充后的Tensor，维度为``T x B x *``（如果batch_first为false）或``B x T x *``（如果batch_first为true）。
inline Tensor pad_sequence(
    ArrayRef<Tensor> sequences,
    bool batch_first = false,
    double padding_value = 0) {
  // 调用PyTorch的pad_sequence函数进行填充操作
  return at::pad_sequence(sequences, batch_first, padding_value);
}
/// Packs a list of variable length Tensors into a PackedSequence object.
///
/// ``sequences`` should be a list of Tensors where each Tensor has dimensions
/// of the form ``L x *``, with `L` being the length of the sequence and `*`
/// representing any number of trailing dimensions, including zero.
///
/// For unsorted sequences, set `enforce_sorted` to `false`. If ``enforce_sorted``
/// is `true`, the sequences are expected to be sorted in decreasing order of length.
///
///
/// Arguments:
///     sequences (torch::ArrayRef<Tensor>): A list of variable length sequences.
///     enforce_sorted (bool, optional): If `true`, checks that the input sequences
///         are sorted by length in decreasing order. Default: `true`.
///
/// Returns:
///     A `PackedSequence` object containing the packed data.
inline PackedSequence pack_sequence(
    ArrayRef<Tensor> sequences,
    bool enforce_sorted = true) {
  
  // Create a Tensor `lengths` to store the lengths of each sequence
  Tensor lengths = torch::empty({(int64_t)sequences.size()}, kInt64);
  
  // Iterate over each sequence and store its length in the `lengths` Tensor
  for (const auto i : c10::irange(sequences.size())) {
    lengths[i] = sequences[i].size(0);  // Store the length of the i-th sequence
  }
  
  // Pack the sequences into a PackedSequence using `pack_padded_sequence`
  return pack_padded_sequence(
      at::pad_sequence(sequences),  // Pad sequences to create a contiguous Tensor
      std::move(lengths),           // Pass lengths Tensor to `pack_padded_sequence`
      /*batch_first=*/false,        // Indicate that the batch dimension is not the first
      /*enforce_sorted=*/enforce_sorted);  // Pass enforce_sorted flag to `pack_padded_sequence`
}

} // namespace rnn
} // namespace utils
} // namespace nn
} // namespace torch
```