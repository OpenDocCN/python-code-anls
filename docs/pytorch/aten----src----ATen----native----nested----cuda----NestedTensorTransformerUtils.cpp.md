# `.\pytorch\aten\src\ATen\native\nested\cuda\NestedTensorTransformerUtils.cpp`

```py
namespace {

/**
 * This builds up the cumulative sequence length for a batch of sequences.
 * This is not very dry, but in the backward pass we already have cumulative_seq_len
 * on device. And all we need on CPU to launch the kernel is NNz. We could refactor the
 * the below function but it adds more complexity than I think is needed.
 */
int64_t get_nnz(Tensor nestedtensor) {
  auto* nt_impl = get_nested_tensor_impl(nestedtensor);
  const auto& sizes = nt_impl->get_nested_sizes();
  auto size_tensor_stride = sizes.stride(0);
  const int64_t batch_size = nestedtensor.size(0);
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  int64_t cumulative_sequence_length = 0;
  for (const auto i : c10::irange(batch_size)) {
    // Calculate the cumulative sum of the sequence lengths
    int64_t current_seq_len = sizes_ptr[(i * size_tensor_stride)];
    cumulative_sequence_length += current_seq_len;
  }
  return cumulative_sequence_length;
}

  /**
   * This function is used to calculate two pieces of metadata that are needed
   * for use with flash-attention and efficient_attention kernels. They are the
   * cumulative sequence_length over a batch of sequences and the maximum
   * sequence length.
   *
   * @return A tuple of cumulative sequence lengths and the maximum sequence
   * length, and the last element in the cumulative_sequence_lengths
   */
  std::tuple<Tensor, int64_t, int64_t> cumulative_and_max_seq_len_nnz(Tensor qkv) {
    TORCH_CHECK(
        qkv.is_nested(),
        "QKV must be nested for flash cumulative_seq_len calculation.")
    auto* nt_impl = get_nested_tensor_impl(qkv);
    const auto& sizes = nt_impl->get_nested_sizes();
    auto size_tensor_stride = sizes.stride(0);

    const int64_t batch_size = qkv.size(0);
    auto cumulative_seqlen = at::zeros(
        {batch_size + 1}, TensorOptions().device(at::kCPU).dtype(at::kInt));

    auto* sizes_ptr = sizes.data_ptr<int64_t>();
    auto* cumulative_seqlen_ptr = cumulative_seqlen.data_ptr<int32_t>();

    int32_t sum = 0;
    int64_t max_seqlen = -1;
    cumulative_seqlen_ptr[0] = sum;
    for (const auto i : c10::irange(batch_size)) {
      // Calculate the cumulative sum of the sequence lengths
      auto current_seq_len = sizes_ptr[(i * size_tensor_stride)];
      sum += current_seq_len;
      cumulative_seqlen_ptr[i + 1] = sum;

      // Find the max element while we traverse
      max_seqlen = std::max(max_seqlen, current_seq_len);
    }
    // Send to GPU, this is pretty light weight calc for normal batch size
    // but maybe this needs to be on gpu
    cumulative_seqlen = cumulative_seqlen.to(TensorOptions().device(at::kCUDA));
  /**
   * 返回一个元组，包含累积序列长度、最大序列长度和总和
   */
  return std::tuple<Tensor, int64_t, int64_t>{
      cumulative_seqlen, max_seqlen, sum};
}

/**
 * 此函数检查嵌套张量是否可以安全地用于flash-attention和efficient_attention内核，
 * 而无需对嵌套张量输入调用contiguous。
 * 它检查存储偏移的相邻差异是否是前一个张量的常数倍数，并且检查步幅是否单调递减。
 * 此检查在对嵌套张量进行转置后进行，结果为形状为[bsz, {seq_len}, num_heads, dim]的Nt。
 *
 * @return 表示是否需要为输入调用contiguous的布尔值
 */
bool is_safe_to_get_storage_as_tensor(const NestedTensorImpl* tensor) {
  const int64_t* tensor_offsets_ptr =
      tensor->get_storage_offsets().data_ptr<int64_t>();
  const Tensor& tensor_sizes = tensor->get_nested_sizes();
  const Tensor& tensor_strides = tensor->get_nested_strides();

  const int64_t n_tensors = tensor_strides.size(0);
  constexpr int64_t n_dims = 3;
  // 这是安全的，因为head_dim保证一致
  const int64_t num_heads = tensor -> opt_size(2).value();
  const int64_t tensor_stride_0 = tensor_strides.stride(0);

  if (n_tensors <= 1) {
    return true;
  }

  int64_t* previous_tensor_stride = tensor_strides.data_ptr<int64_t>();

  // 首先检查第一个张量的步幅是否严格降序排列
  // 注意：如果num_heads等于1，则跳过stride[0]
  if (num_heads == 1) {
    if (previous_tensor_stride[0] <= previous_tensor_stride[2]) {
      // 这意味着最后一个步幅大于seq_len的步幅
      return false;
    }
  } else {
    for (int i{1}; i < n_dims; i++) {
      if (previous_tensor_stride[i - 1] <= previous_tensor_stride[i]) {
        return false;
      }
    }
    // 检查嵌套张量中的每个张量i是否具有相同的步幅
    for (int i{1}; i < n_tensors; i++) {
      for (const int64_t j : c10::irange(n_dims)) {
        if (previous_tensor_stride[j] !=
            previous_tensor_stride[i * tensor_stride_0 + j]) {
          return false;
        }
      }
    }
  }

  // 检查偏移是否是前一个numels的常数倍数
  const int64_t* tensor_size_ptr = tensor_sizes.const_data_ptr<int64_t>();
  const int64_t* tensor_stride_ptr = tensor_strides.const_data_ptr<int64_t>();

  int64_t numel_0 = (tensor_size_ptr[0] * tensor_stride_ptr[0]);
  TORCH_INTERNAL_ASSERT(numel_0 > 0, "numels must be positive!");

  int64_t offset_constant =
      (tensor_offsets_ptr[1] - tensor_offsets_ptr[0]) / numel_0;
    for (int64_t i = 2; i < n_tensors; i++) {
      // 循环遍历从第二个开始的所有张量
      // TODO: 当允许存在 0 个序列长度的嵌套张量时，我们需要进行保护处理
      // 计算前一个张量的元素数量
      int64_t previous_numel = tensor_size_ptr[(i - 1) * tensor_stride_0] *
          tensor_stride_ptr[(i - 1) * tensor_stride_0];
      // 断言前一个张量的元素数量大于 0，否则输出错误信息
      TORCH_INTERNAL_ASSERT(previous_numel > 0, "numels must be positive!");
      // 计算当前偏移常量
      int64_t current_offset_constant =
          (tensor_offsets_ptr[i] - tensor_offsets_ptr[i - 1]) / previous_numel;
      // 如果当前偏移常量与之前计算的偏移常量不相等，返回 false
      if (current_offset_constant != offset_constant) {
        return false;
      }
    }
    // 循环结束，表示所有张量偏移常量一致，返回 true
    // 恭喜，你做到了！
    return true;
  }

  /**
   * 处理一个单独的 NestedTensor，将其重塑并视为 DenseTensor
   * 通常对于 q、k、v 的处理方式是：
   * (1) 获取连续嵌套张量的存储
   * (2) 视图重塑为形状 {output_batch_size, {*}_t.size(1), output_num_heads, head_dim_{*}}
   *    和步长 {0, nnz_{*}_stride, head_{*}_stride, head_dim_stride}，
   *    其中 head_{*}_stride 如果 {*}_num_heads_needs_broadcast 为 true 则为 0
   * (3) 如果 {*}_t.size(1)（即 seq_len 为 1），则通过重塑将前两个维度合并为 {Nnz_{*}, output_num_heads, head_dim_{*}}
   *     重塑应为视图，不应造成复制到稠密张量而不获取存储
   */
  at::Tensor view_as_dense(
      const at::Tensor& input_nestedtensor,
      const int64_t Nnz,
      const int64_t num_heads,
      const int64_t head_dim,
      const bool batch_needs_broadcast = false,
      const bool num_heads_needs_broadcast = false) {
    // 获取输入 NestedTensor 的实现
    const auto* tensor_impl = get_nested_tensor_impl(input_nestedtensor);
    // 获取作为张量的存储
    Tensor storage_as_tensor = tensor_impl->get_unsafe_storage_as_tensor();

    // 定义头维度步长为 1
    constexpr int64_t head_dim_stride = 1;
    // 获取嵌套张量的步长和偏移
    const int64_t* nt_strides =
        tensor_impl->get_nested_strides().data_ptr<int64_t>();
    const int64_t* nt_offsets_ptr =
        tensor_impl->get_storage_offsets().data_ptr<int64_t>();

    // 获取非零元素的步长和头部步长，根据需要广播确定是否为 0
    const int64_t nnz_stride = nt_strides[0];
    const int64_t head_stride = num_heads_needs_broadcast ? 0 : nt_strides[1];

    // 如果需要广播批次维度
    if (batch_needs_broadcast) {
      // 重塑输入缓冲区为指定的形状和步长
      Tensor input_buffer_reshaped = storage_as_tensor.as_strided(
          {Nnz, input_nestedtensor.size(1), num_heads, head_dim},
          {0, nnz_stride, head_stride, head_dim_stride},
          nt_offsets_ptr[0]);
      // 将重塑后的张量继续重塑为 {-1, num_heads, head_dim} 的形状
      return input_buffer_reshaped.reshape({-1, num_heads, head_dim});
    }

      // 如果不需要广播批次维度，直接返回重塑后的张量
      return storage_as_tensor.as_strided(
          {Nnz, input_nestedtensor.size(1), num_heads, head_dim},
          {0, nnz_stride, head_stride, head_dim_stride},
          nt_offsets_ptr[0]).reshape({-1, num_heads, head_dim});
    }
  // 返回一个存储为张量的视图，用于存储多维数据，并且可能包含内存重叠
  return storage_as_tensor.as_strided(
      {Nnz, num_heads, head_dim},
      {nnz_stride, head_stride, head_dim_stride},
      nt_offsets_ptr[0]);
}

/**
 * 这个函数是一个辅助函数，用于处理需要在批次或头数维度上广播的嵌套查询、键和值，
 * 并且将其预处理以便与flash-attention或efficient-attention内核一起运行。
 * @return 包含运行融合内核所需所有数据的元组
 */
auto sdpa_nested_preprocessing_with_broadcast(
    const Tensor& query, const Tensor& key, const Tensor& value) {
  // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
  // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);
  const int64_t v_batch_size = value.size(0);

  const int64_t output_batch_size =
      std::max({q_batch_size, k_batch_size, v_batch_size});

  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);

  const int64_t output_num_heads =
      std::max({q_num_heads, k_num_heads, v_num_heads});

  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  // 将查询、键和值的维度重新排列，将头数维度移动到第二个位置
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  // 在sdp_utils中的检查确保，如果{*}_batch_size/{*}_num_heads !=
  // output_batch_size/num_heads，则它们为1
  bool q_batch_size_needs_broadcast = q_batch_size != output_batch_size;
  bool k_batch_size_needs_broadcast = k_batch_size != output_batch_size;
  bool v_batch_size_needs_broadcast = v_batch_size != output_batch_size;

  // 如果{*}_batch_size_needs_broadcast为true，则：
  // (1) max_seqlen_batch_{*}由{*}_t.size(1)给出，因为needs_broadcast表示batch_size为1，
  //     因此seq_len只有一个值
  // (2) 累积的序列长度由[0, {*}_t.size(1), 2 * {*}_t.size(1), ..., output_batch_size * {*}_t.size(1)]给出
  // (3) Nnz_{*}由output_batch_size * {*}_t.size(1)给出；
  
  int64_t max_seqlen_batch_q = 0, Nnz_q = 0;
  Tensor cumulative_sequence_length_q;
  if (q_batch_size_needs_broadcast || !q_t.is_nested()) {
    max_seqlen_batch_q = q_t.size(1);
    cumulative_sequence_length_q = at::arange(
        0,
        (output_batch_size + 1) * max_seqlen_batch_q,
        max_seqlen_batch_q,
        TensorOptions().device(at::kCUDA).dtype(at::kInt));
    Nnz_q = output_batch_size * max_seqlen_batch_q;
    // 如果不需要广播 k 和 v 的批量大小，则计算 k 和 v 的累积序列长度、最大序列长度和非零元素个数
    } else {
      auto cumulative_and_max_q_and_nnz_q = cumulative_and_max_seq_len_nnz(q_t);
      cumulative_sequence_length_q =
          std::get<0>(cumulative_and_max_q_and_nnz_q);
      max_seqlen_batch_q = std::get<1>(cumulative_and_max_q_and_nnz_q);
      Nnz_q = std::get<2>(cumulative_and_max_q_and_nnz_q);
    }

    // 初始化 max_seqlen_batch_kv 和 Nnz_kv 为 0，以及 cumulative_sequence_length_kv 为 Tensor
    int64_t max_seqlen_batch_kv = 0, Nnz_kv = 0;
    Tensor cumulative_sequence_length_kv;
    
    // 如果需要广播 k 和 v 的批量大小
    if (k_batch_size_needs_broadcast && v_batch_size_needs_broadcast) {
      // 检查 k 和 v 的第二维是否相等
      TORCH_CHECK(k_t.size(1) == v_t.size(1));
      // 设置 max_seqlen_batch_kv 为 k_t 的第二维大小
      max_seqlen_batch_kv = k_t.size(1);
      // 生成 cumulative_sequence_length_kv，使用 CUDA 设备和整数类型
      cumulative_sequence_length_kv = at::arange(
          0,
          (output_batch_size + 1) * max_seqlen_batch_kv,
          max_seqlen_batch_kv,
          TensorOptions().device(at::kCUDA).dtype(at::kInt));
      // 计算 Nnz_kv，为输出批量大小乘以最大序列长度
      Nnz_kv = output_batch_size * max_seqlen_batch_kv;
    } else {
      // 否则，根据需要广播的情况选择 k_t 或 v_t 计算累积序列长度、最大序列长度和非零元素个数
      auto cumulative_and_max_kv_and_nnz_kv = k_batch_size_needs_broadcast
          ? cumulative_and_max_seq_len_nnz(v_t)
          : cumulative_and_max_seq_len_nnz(k_t);
      cumulative_sequence_length_kv =
          std::get<0>(cumulative_and_max_kv_and_nnz_kv);
      max_seqlen_batch_kv = std::get<1>(cumulative_and_max_kv_and_nnz_kv);
      Nnz_kv = std::get<2>(cumulative_and_max_kv_and_nnz_kv);
    }

    // 检查是否需要广播 q_num_heads、k_num_heads 和 v_num_heads
    bool q_num_heads_needs_broadcast = q_num_heads != output_num_heads;
    bool k_num_heads_needs_broadcast = k_num_heads != output_num_heads;
    bool v_num_heads_needs_broadcast = v_num_heads != output_num_heads;

    // 初始化 query_buffer_reshaped、key_buffer_reshaped 和 value_buffer_reshaped 为 Tensor
    Tensor query_buffer_reshaped;
    Tensor key_buffer_reshaped;
    Tensor value_buffer_reshaped;

    // 如果 q_t 不是嵌套的
    if (!q_t.is_nested()) {
      // 将 q_t 扩展到指定维度
      query_buffer_reshaped = q_t.expand(
          {output_batch_size, q_t.size(1), output_num_heads, head_dim_qk});
      // 将其形状重新整理为指定形状
      query_buffer_reshaped =
          query_buffer_reshaped.reshape({Nnz_q, output_num_heads, head_dim_qk});
    } else {
      // 否则，获取嵌套张量的实现
      const auto* query_impl = get_nested_tensor_impl(q_t);
      // 如果 q_t 不是连续的且不安全获取存储作为张量，则使其连续
      if (!q_t.is_contiguous() &&
          !is_safe_to_get_storage_as_tensor(query_impl)) {
        q_t = q_t.contiguous();
      }
      // 如果需要广播 q 的批量大小，则有效批量大小为 output_batch_size，否则为 Nnz_q
      const int64_t effective_batch_size_q =
          q_batch_size_needs_broadcast ? output_batch_size : Nnz_q;
      // 使用视图作为密集张量，调整形状
      query_buffer_reshaped = view_as_dense(
          q_t,
          effective_batch_size_q,
          output_num_heads,
          head_dim_qk,
          q_batch_size_needs_broadcast,
          q_num_heads_needs_broadcast);
    }

    // 获取 key_t 和 value_t 的实现
    const auto* key_impl = get_nested_tensor_impl(k_t);
    const auto* value_impl = get_nested_tensor_impl(v_t);

    // 如果嵌套张量的存储物理布局不是 batch, {seq_len}, num_heads, head_dim，则需要调用 contiguous
    if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
      k_t = k_t.contiguous();
    }
    if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
      v_t = v_t.contiguous();
    }
    // 计算有效批次大小 k，如果需要广播批次大小，则使用输出批次大小，否则使用 Nnz_kv
    const int64_t effective_batch_size_k =
        k_batch_size_needs_broadcast ? output_batch_size : Nnz_kv;
    // 将 key_buffer_reshaped 视图转换为稠密张量，重新形状为指定的参数
    key_buffer_reshaped = view_as_dense(
        k_t,
        effective_batch_size_k,
        output_num_heads,
        head_dim_qk,
        k_batch_size_needs_broadcast,
        k_num_heads_needs_broadcast);

    // 计算有效批次大小 v，如果需要广播批次大小，则使用输出批次大小，否则使用 Nnz_kv
    const int64_t effective_batch_size_v =
        v_batch_size_needs_broadcast ? output_batch_size : Nnz_kv;
    // 将 value_buffer_reshaped 视图转换为稠密张量，重新形状为指定的参数
    value_buffer_reshaped = view_as_dense(
        v_t,
        effective_batch_size_v,
        output_num_heads,
        head_dim_v,
        v_batch_size_needs_broadcast,
        v_num_heads_needs_broadcast);

    Tensor output_shape;
    // 如果不需要广播 q 的批次大小
    if (!q_batch_size_needs_broadcast) {
      // 获取 q_t 的嵌套尺寸并克隆为 output_shape
      output_shape = get_nested_sizes(q_t).clone();
      // 如果 head_dim_v 不等于 head_dim_qk，则填充 output_shape 的最后一维为 head_dim_v
      if (head_dim_v != head_dim_qk) {
        output_shape.select(1, -1).fill_(head_dim_v);
      }
      // 如果 q_num_heads 需要广播，则填充 output_shape 的第二维为 output_num_heads
      if (q_num_heads_needs_broadcast) {
        output_shape.select(1, 1).fill_(output_num_heads);
      }
    } else {
      // 如果需要广播 q 的批次大小，则创建一个空的 output_shape 张量
      output_shape = at::empty(
          {output_batch_size, 3}, TensorOptions().dtype(kLong).device(kCPU));
      // 填充 output_shape 的第一维为 q_t 的第二维大小
      output_shape.select(1, 0).fill_(q_t.size(1));
      // 填充 output_shape 的第二维为 output_num_heads
      output_shape.select(1, 1).fill_(output_num_heads);
      // 填充 output_shape 的第三维为 head_dim_v
      output_shape.select(1, 2).fill_(head_dim_v);
    }

    // 返回一个包含多个元素的元组，包括重新形状后的查询、键、值缓冲，以及其他元信息
    return std::make_tuple(
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_shape);
}
} // namespace

// 定义函数 sdpa_nested_preprocessing，接收三个张量作为输入，并返回包含多个张量和整数的元组
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor>
sdpa_nested_preprocessing(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
  // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // 计算查询、键、值张量的批量大小和头数
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);
  const int64_t v_batch_size = value.size(0);

  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);

  // 如果批量大小或头数不一致，则调用 sdpa_nested_preprocessing_with_broadcast 处理
  if (!(q_batch_size == k_batch_size && q_batch_size == v_batch_size) ||
      !(q_num_heads == k_num_heads && k_num_heads == v_num_heads)) {
    return sdpa_nested_preprocessing_with_broadcast(query, key, value);
  }

  // 获取头数和每个头的维度大小
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  // 将查询、键、值张量的头维度和序列长度维度交换
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  // 调用 cumulative_and_max_seq_len_nnz 函数获取累计序列长度和最大非零元素数
  auto cumulative_and_max_q_and_nnz_q = cumulative_and_max_seq_len_nnz(q_t);
  auto cumulative_and_max_kv_and_nnz_kv = cumulative_and_max_seq_len_nnz(k_t);

  // 获取累计序列长度张量
  Tensor cumulative_sequence_length_q =
      std::get<0>(cumulative_and_max_q_and_nnz_q);
  Tensor cumulative_sequence_length_kv =
      std::get<0>(cumulative_and_max_kv_and_nnz_kv);

  // 获取每批次的最大序列长度和非零元素数
  const int64_t max_seqlen_batch_q =
      std::get<1>(cumulative_and_max_q_and_nnz_q);
  const int64_t max_seqlen_batch_kv =
      std::get<1>(cumulative_and_max_kv_and_nnz_kv);

  const int64_t Nnz_q = std::get<2>(cumulative_and_max_q_and_nnz_q);
  const int64_t Nnz_kv = std::get<2>(cumulative_and_max_kv_and_nnz_kv);

  // 声明用于重塑的缓冲张量
  Tensor query_buffer_reshaped;
  Tensor key_buffer_reshaped;
  Tensor value_buffer_reshaped;

  // 获取 NestedTensor 的实现指针
  const auto* query_impl = get_nested_tensor_impl(q_t);
  const auto* key_impl = get_nested_tensor_impl(k_t);
  const auto* value_impl = get_nested_tensor_impl(v_t);

  // 如果张量不是连续的，并且不安全直接获取存储作为张量，则需要进行连续化操作
  if (!q_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(query_impl)) {
    q_t = q_t.contiguous();
  }
  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
  }
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
  }

  // 将张量按照给定的非零元素数、头数和头维度重塑为稠密张量
  query_buffer_reshaped = view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk);
  key_buffer_reshaped = view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk);
  value_buffer_reshaped = view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v);

  // 获取 NestedTensor 的大小并克隆形状
  auto output_shape = get_nested_sizes(q_t).clone();

  // 如果头维度不一致，则执行以下操作
  if (head_dim_v != head_dim_qk) {
    output_shape.select(1, -1).fill_(head_dim_v);

用于在 `output_shape` 张量的第一维上选择所有元素的最后一个维度，并将其填充为 `head_dim_v`。


  }

结束函数或代码块的大括号。


  return std::make_tuple(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      output_shape);

返回一个 `std::tuple` 对象，包含了多个变量：
- `query_buffer_reshaped`
- `key_buffer_reshaped`
- `value_buffer_reshaped`
- `cumulative_sequence_length_q`
- `cumulative_sequence_length_kv`
- `max_seqlen_batch_q`
- `max_seqlen_batch_kv`
- `output_shape`

这些变量被打包成元组返回给调用者。
  // 定义函数 sdpa_nested_preprocessing_backward，接受多个张量作为输入并返回一个包含五个张量的元组
  const at::Tensor& grad_out_,
  const at::Tensor& query,
  const at::Tensor& key,
  const at::Tensor& value,
  const at::Tensor& out,
  const Tensor& cumulative_sequence_length_q,
  const Tensor& cumulative_sequence_length_kv,
  const int64_t max_seqlen_batch_q,
  const int64_t max_seqlen_batch_kv) {
  
  // 计算查询、键、值张量的批次大小
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);
  const int64_t v_batch_size = value.size(0);
  
  // 计算查询、键、值张量的头数
  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);
  
  // 检查批次大小和头数是否一致，否则抛出错误
  if (!(q_batch_size == k_batch_size && q_batch_size == v_batch_size) ||
      !(q_num_heads == k_num_heads && k_num_heads == v_num_heads)) {
    TORCH_CHECK(false, "Broadcasted NestedTensor inputs is currently not supported for backwards.");
  }
  
  // 获取头数和头维度
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  
  // 对查询、键、值、梯度输出和输出张量进行转置
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  Tensor grad_out_t = grad_out_.transpose(1, 2);
  Tensor out_t = out.transpose(1, 2);
  
  // 计算查询张量和键张量的非零元素数量
  const int64_t Nnz_q = get_nnz(q_t);
  const int64_t Nnz_kv = get_nnz(k_t);
  
  // 定义用于缓存的重塑后的张量变量
  Tensor query_buffer_reshaped;
  Tensor key_buffer_reshaped;
  Tensor value_buffer_reshaped;
  Tensor grad_out_buffer_reshaped;
  Tensor output_buffer_reshaped;
  
  // 获取查询、键、值、梯度输出和输出张量的 NestedTensor 实现指针
  const auto* query_impl = get_nested_tensor_impl(q_t);
  const auto* key_impl = get_nested_tensor_impl(k_t);
  const auto* value_impl = get_nested_tensor_impl(v_t);
  const auto* grad_out_impl = get_nested_tensor_impl(grad_out_t);
  const auto* out_impl = get_nested_tensor_impl(out_t);
  
  // 如果查询张量不是连续的且不安全获取存储作为张量，则进行连续化操作
  if (!q_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(query_impl)) {
    q_t = q_t.contiguous();
  }
  // 如果键张量不是连续的且不安全获取存储作为张量，则进行连续化操作
  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
  }
  // 如果值张量不是连续的且不安全获取存储作为张量，则进行连续化操作
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
  }
  // 如果梯度输出张量不是连续的且不安全获取存储作为张量，则进行连续化操作
  if (!grad_out_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(grad_out_impl)) {
    grad_out_t = grad_out_t.contiguous();
  }
  // 如果输出张量不是连续的且不安全获取存储作为张量，则进行连续化操作
  if (!out_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(out_impl)) {
    # 将输出张量转换为连续存储，以提高访问效率
    out_t = out_t.contiguous();
  }

  # 将查询张量重新形状为稠密张量，以便后续操作
  query_buffer_reshaped = view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk);
  # 将键张量重新形状为稠密张量，以便后续操作
  key_buffer_reshaped = view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk);
  # 将值张量重新形状为稠密张量，以便后续操作
  value_buffer_reshaped = view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v);

  # 将梯度输出张量重新形状为稠密张量，以便后续操作
  grad_out_buffer_reshaped =
      view_as_dense(grad_out_t, Nnz_q, num_heads, head_dim_v);
  # 将输出张量重新形状为稠密张量，以便后续操作
  output_buffer_reshaped = view_as_dense(out_t, Nnz_q, num_heads, head_dim_v);

  # 获取查询张量的嵌套尺寸，并复制其形状作为输出形状
  auto output_shape = get_nested_sizes(q_t).clone();
  # 如果值的头维度不等于查询和键的头维度，填充输出形状的最后一个维度为值的头维度
  if (head_dim_v != head_dim_qk) {
    output_shape.select(1, -1).fill_(head_dim_v);
  }

  # 返回重塑后的张量作为元组的一部分
  return std::make_tuple(
      grad_out_buffer_reshaped,
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      output_buffer_reshaped);
}

// 结束预处理命名空间
} // namespace preprocessing

// 结束本地命名空间
} // namespace native

// 结束 AT（Assistive Technology 辅助技术）命名空间
} // namespace at
```