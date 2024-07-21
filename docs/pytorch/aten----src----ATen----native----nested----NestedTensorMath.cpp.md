# `.\pytorch\aten\src\ATen\native\nested\NestedTensorMath.cpp`

```py
namespace at {
namespace native {
namespace {

// 计算给定大小的张量所需的字节数
int64_t num_bytes(IntArrayRef sizes) {
  // 初始化结果为1，表示空张量至少有1个字节的内存
  int64_t result = 1;
  int64_t stride = 1;
  // 从最后一个维度向前计算
  for (int ii = sizes.size() - 1; ii >= 0; --ii) {
    // 更新结果，考虑每个维度的大小和步长
    result += (sizes[ii] - 1) * stride;
    // TODO: 当支持非连续内存时，接受步长作为输入
    stride *= sizes[ii];
  }
  return result;
}

// 将张量填充为目标形状
Tensor pad_tensor_to_shape(
    const Tensor& t,
    IntArrayRef goal_shape,
    double value = 0) {
  // 存储填充量的向量
  std::vector<int64_t> padd;
  // 获取当前张量的尺寸
  auto tup = t.sizes();
  // 检查张量维度是否与目标形状的长度匹配
  TORCH_CHECK(
      t.dim() == (int64_t)(goal_shape.size()),
      "dimension ",
      t.dim(),
      " doesn't match length ",
      goal_shape.size(),
      " of goal shape.");
  // 从最后一个维度向前计算填充量
  for (int64_t i = tup.size() - 1; i >= 0; i--) {
    padd.push_back(0);
    padd.push_back(goal_shape[i] - tup[i]);
  }
  // 使用指定值对张量进行常数填充
  Tensor new_tensor = at::constant_pad_nd(t, IntArrayRef(padd), value);
  // 重新整形填充后的张量为目标形状
  new_tensor = new_tensor.reshape(goal_shape);
  return new_tensor;
}

} // namespace
} // namespace native
} // namespace at

// 从掩码创建嵌套张量
Tensor NestedTensor_nested_tensor_from_mask(const Tensor& t, const Tensor& mask, bool mask_check) {
    // 检查掩码张量的类型和维度是否符合要求
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Expected mask to be of ScalarType Bool, but got ", mask.scalar_type(), " instead.");
    TORCH_CHECK(mask.dim() == 2, "Padding mask should be 2D");
    // 检查输入张量的维度是否为3
    TORCH_CHECK(t.dim() == 3, "Input should be a 3D tensor, N * L * D");
    auto N = t.size(0), L = t.size(1), D = t.size(2);
    auto NN = mask.size(0), LL = mask.size(1);
    // 检查掩码尺寸是否与输入尺寸匹配
    TORCH_CHECK(N == NN && L == LL, "Mask size should match input size");

    // 获取掩码的大小作为填充的标记
    Tensor sizes = mask;
    Tensor tmp_pad = at::zeros({N, 1}, mask.options());
    // 确保填充仅在掩码末尾添加
    Tensor nums = at::cat({sizes, tmp_pad}, 1).to(kInt).argmin(1);

    // 计算每行的累积和作为填充的大小标记
    sizes = sizes.cumsum(1).select(1, L - 1);
    nums = nums.to(sizes.options());

    // 如果需要检查掩码，则验证填充是否是左对齐且没有间隙
    if (mask_check)
      TORCH_CHECK(sizes.equal(nums), "Mask must be left-aligned without gaps");

    sizes = sizes.reshape({N, 1});
    // 创建填充维度为目标形状的张量
    Tensor d = at::full_like(sizes, D);

    // 将填充维度与目标形状的张量连接起来
    sizes = at::cat({sizes, d}, 1).to(kCPU);

    // 使用内部函数创建嵌套张量
    return at::_nested_from_padded(t, sizes, false);
}
// 从给定的张量和掩码创建嵌套张量，掩码要求左对齐
bool NestedTensor_nested_tensor_from_mask_left_aligned(const Tensor& t, const Tensor& mask) {
    // 检查掩码的标量类型是否为布尔型
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Expected mask to be of ScalarType Bool, but got ", mask.scalar_type(), " instead.");
    // 检查掩码的维度是否为2
    TORCH_CHECK(mask.dim() == 2, "Padding mask should be 2D");
    // 检查输入张量的维度是否为3，表示为 N * L * D
    TORCH_CHECK(t.dim() == 3, "Input should be a 3D tensor, N * L * D");
    // 获取输入张量和掩码的尺寸
    auto N = t.size(0), L = t.size(1);
    auto NN = mask.size(0), LL = mask.size(1);
    // 检查输入张量和掩码的尺寸是否匹配
    TORCH_CHECK(N == NN && L == LL, "Mask size should match input size");

    // 将掩码作为尺寸张量
    Tensor sizes = mask;
    // 创建一个与掩码相同的尺寸的零张量，并将其作为临时填充
    Tensor tmp_pad = at::zeros({N, 1}, mask.options());
    // 确保填充仅添加在掩码的末尾
    Tensor nums = at::cat({sizes, tmp_pad}, 1).to(kInt).argmin(1);

    // 计算每个示例的大小并选择最后一个尺寸
    sizes = sizes.cumsum(1).select(1, L - 1);
    nums = nums.to(sizes.options());

    // 检查计算出的尺寸是否与填充的位置匹配
    return sizes.equal(nums);
}

// 从张量列表创建嵌套张量
Tensor _nested_tensor_from_tensor_list(
    TensorList list,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 遍历张量列表
  for (const auto i : c10::irange(list.size())) {
    // 对于索引大于0的张量，检查其维度是否与前一个张量相同
    if (i > 0) {
      int64_t dim_i = list[i].dim();
      int64_t dim_prev = list[i - 1].dim();
      TORCH_CHECK(
          dim_i == dim_prev,
          "All Tensors given to nested_tensor must have the same dimension. ",
          "Found dimension ",
          dim_i,
          " for Tensor at index ",
          i,
          " and dimension ",
          dim_prev,
          " for Tensor at index ",
          i - 1,
          ".");
    }
  }
  // 将张量列表封装成张量节点，并返回
  return impl::wrap_tensor_node(
      impl::TensorNode(list),
      dtype,
      layout,
      device,
      pin_memory);
}

// 执行嵌套层归一化
std::tuple<Tensor, Tensor, Tensor> nested_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    //
    // 检查权重和偏置是否存在，若不存在则抛出错误信息
    TORCH_CHECK(weight_opt && bias_opt, "NestedTensor layer_norm requires weight and bias");
    // 获取权重和偏置的引用
    const auto& weight = *weight_opt;
    const auto& bias = *bias_opt;
    // 检查权重是否为嵌套张量，不支持嵌套张量的情况
    TORCH_CHECK(!weight.is_nested(), "NestedTensor weight not supported for layer_norm");
    // 检查偏置是否为嵌套张量，不支持嵌套张量的情况
    TORCH_CHECK(!bias.is_nested(), "NestedTensor bias not supported for layer_norm");
    // 获取输入张量的嵌套张量实现指针
    auto* nt_input = get_nested_tensor_impl(input);
    // 检查输入张量是否是连续的嵌套张量
    TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
    // 获取输入张量的缓冲区
    const auto& input_buffer = nt_input->get_buffer();
    // 检查输入张量、规范化形状、权重和偏置的有效性，并返回 M 和 N 的值
    auto M_N = _check_nested_layer_norm_inputs(*nt_input, normalized_shape, weight, bias);
    auto M = M_N.first;
    auto N = M_N.second;
    // 获取连续的权重和偏置
    const auto weight_contig = weight.expect_contiguous();
    const auto bias_contig = bias.expect_contiguous();
    // 根据输入缓冲区的形状创建一个空的输出缓冲区，保持连续性
    auto output_buffer = at::native::empty_like(
        input_buffer,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
    // 获取输入缓冲区的选项
    auto options = input_buffer.options();
    // 如果输入缓冲区在 GPU 上，将累加类型设置为累加类型，并更新选项
    if (input_buffer.is_cuda()) {
        auto acc_type = at::toAccumulateType(input_buffer.scalar_type(), true);
        options = options.dtype(acc_type);
    }
    // 创建一个空的张量用于存储均值
    Tensor mean = at::empty({M}, options);
    // 创建一个空的张量用于存储归一化标准差的倒数
    Tensor rstd = at::empty({M}, options);
    // 调用 LayerNormKernel 进行层归一化计算
    LayerNormKernel(
        input_buffer.is_cuda() ? kCUDA : kCPU, // 确定计算设备是 CUDA 还是 CPU
        input_buffer, // 输入缓冲区
        *weight_contig, // 权重张量
        *bias_contig, // 偏置张量
        M, // M 的值
        N, // N 的值
        eps, // ε 参数
        &output_buffer, // 输出缓冲区
        &mean, // 存储均值的张量
        &rstd); // 存储归一化标准差的倒数的张量
    // 返回一个元组，包含封装后的输出缓冲区、均值和归一化标准差的倒数
    return std::make_tuple(
        wrap_buffer(output_buffer, nt_input->get_nested_sizes()), // 封装后的输出缓冲区
        mean, // 均值
        rstd // 归一化标准差的倒数
    );
}

// 从填充的张量和嵌套示例中创建嵌套张量
Tensor NestedTensor_from_padded_and_nested_example(
    const Tensor& padded,
    const Tensor& nt_example) {
  // 调用内部函数 _nested_from_padded，传入填充的张量和嵌套示例的嵌套尺寸
  return _nested_from_padded(padded, get_nested_tensor_impl(nt_example)->get_nested_sizes());
}

// 通用函数，从填充的张量和尺寸张量创建嵌套张量
Tensor nested_from_padded_generic(
    const Tensor& padded,
    const Tensor& sizes,
    const bool do_transform_0213) {
  // 检查并执行 0213 变换
  auto padded_transformed = padded;
  if (do_transform_0213) {
    // 对填充的张量进行维度置换、连续化和形状重塑
    padded_transformed = padded.permute({0, 2, 1, 3})
      .contiguous()
      .view(
          {padded.size(0),
           padded.size(2),
           padded.size(1) * padded.size(3)});
  }
  // 获取目标尺寸，用于匹配填充的张量可能超出嵌套张量的最大尺寸
  auto target_size = NestedTensor_get_max_size_from_size_tensor(sizes);
  // 检查维度匹配
  const size_t dim = padded_transformed.dim();
  TORCH_CHECK(dim - 1 == target_size.size(), "dim: ", dim, "target_size: ", target_size.size());
  // 更新目标尺寸，确保与填充的张量的每个维度匹配
  for (size_t ii = 0; ii < dim - 1; ++ii) {
    const auto padded_size_i = padded_transformed.sizes()[ii + 1];
    if (target_size[ii] < padded_size_i) {
      target_size[ii] = padded_size_i;
    }
  }
  IntArrayRef target_size_arr(target_size);
  // 创建掩码张量列表
  std::vector<at::Tensor> masks;
  std::vector<at::Tensor> all_sizes = sizes.unbind();
  // 对每个尺寸张量创建对应的掩码张量，并进行形状填充
  for (const auto& size : all_sizes) {
    IntArrayRef sizes_i(
        size.data_ptr<int64_t>(), size.data_ptr<int64_t>() + size.numel());
    at::Tensor mask_i = padded_transformed.new_full(
        sizes_i, true, kBool, c10::nullopt, c10::nullopt, c10::nullopt);
    masks.push_back(pad_tensor_to_shape(mask_i, target_size_arr));
  }
  // 将所有掩码张量堆叠在一起形成最终的掩码张量
  at::Tensor final_mask = at::stack(masks);
  // 根据最终的掩码张量选择填充的张量中的有效数据，并根据填充张量的设备返回新的缓冲区张量
  at::Tensor new_buffer = padded_transformed.masked_select(final_mask).to(padded.device());
  // 使用内部函数构造嵌套张量的实现
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(new_buffer), sizes);
}

// 通用函数，将嵌套张量转换为填充张量
Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
  // TODO: 支持非连续情况
  // 目前仅支持连续嵌套张量，否则报错
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(get_nested_tensor_impl(t)),
      "for now to_padded_tensor only supports contiguous nested tensor");
  // TODO: 跳过所有 1x1 张量的优化
  auto& nt = *get_nested_tensor_impl(t);
  // 获取嵌套张量的最大尺寸和嵌套尺寸
  auto max_size = NestedTensor_get_max_size(nt);
  auto sizes = nt.get_nested_sizes();

  if (sizes.numel() == 0 || sizes.dim() == 0) {
    // 内部断言，确保嵌套张量的缓冲区为空
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nt.get_buffer().numel() == 0);


这些注释解释了每行代码的作用和意图，帮助理解每个函数和操作的目的。
    return nt.get_buffer().clone();
  }


// 返回 NestedTensor 的缓冲区的克隆副本
return nt.get_buffer().clone();



  TORCH_CHECK(
      t.numel() > 0,
      "to_padded_tensor: at least one constituent tensor should have non-zero numel"
  )


// 检查张量 t 的元素数量是否大于 0，否则抛出错误信息
TORCH_CHECK(
    t.numel() > 0,
    "to_padded_tensor: at least one constituent tensor should have non-zero numel"
)



  // TODO: doesn't handle empty/scalar entries because we don't need
  // it for transformers; see to_padded_tensor in
  // pytorch/nestedtensor's masking.cpp.


// TODO: 不处理空的或标量条目，因为在 transformers 中不需要，
// 参见 pytorch/nestedtensor 的 masking.cpp 中的 to_padded_tensor 函数。



  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_num_columns = sizes.sizes()[1];
  const auto sizes_data_start = sizes.data_ptr<int64_t>();
  const auto sizes_data_end = sizes_data_start + sizes.numel();
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(sizes_num_rows);
  for (auto sizes_data = sizes_data_start; sizes_data != sizes_data_end;
       sizes_data += sizes_num_columns) {
    split_sizes.push_back(
        num_bytes(IntArrayRef(sizes_data, sizes_num_columns)));
  }


// 计算输入张量 sizes 的行数和列数，并准备相关数据
const auto sizes_num_rows = sizes.sizes()[0];
const auto sizes_num_columns = sizes.sizes()[1];
const auto sizes_data_start = sizes.data_ptr<int64_t>();
const auto sizes_data_end = sizes_data_start + sizes.numel();
std::vector<int64_t> split_sizes;
split_sizes.reserve(sizes_num_rows);
// 遍历 sizes 的数据，根据列数计算每行的字节数并存储在 split_sizes 中
for (auto sizes_data = sizes_data_start; sizes_data != sizes_data_end;
     sizes_data += sizes_num_columns) {
  split_sizes.push_back(
      num_bytes(IntArrayRef(sizes_data, sizes_num_columns)));
}



  std::vector<int64_t> nonzero_split_sizes;
  for (const auto split_size : split_sizes) {
    if (split_size > 0) {
      nonzero_split_sizes.push_back(split_size);
    }
  }


// 从 split_sizes 中筛选出大于 0 的分割大小，存储在 nonzero_split_sizes 中
std::vector<int64_t> nonzero_split_sizes;
for (const auto split_size : split_sizes) {
  if (split_size > 0) {
    nonzero_split_sizes.push_back(split_size);
  }
}



  const auto buffer = nt.get_buffer();
  std::vector<Tensor> buffers_;
  if (!nonzero_split_sizes.empty()) {
    buffers_ = at::split_with_sizes(buffer, nonzero_split_sizes, 0);
  }


// 获取 NestedTensor 的缓冲区，并根据 nonzero_split_sizes 分割缓冲区
const auto buffer = nt.get_buffer();
std::vector<Tensor> buffers_;
if (!nonzero_split_sizes.empty()) {
  buffers_ = at::split_with_sizes(buffer, nonzero_split_sizes, 0);
}



  std::vector<Tensor> buffers;
  buffers.reserve(split_sizes.size());
  int64_t next_buffer = 0;
  auto sizes_ptr = sizes_data_start;
  for (const auto split_size : split_sizes) {
    Tensor to_pad;
    IntArrayRef tensor_sizes(sizes_ptr, sizes_num_columns);
    if (split_size > 0) {
      to_pad = buffers_[next_buffer++].reshape(tensor_sizes);
    } else {
      to_pad = at::empty(tensor_sizes, buffer.options());
    }
    buffers.push_back(pad_tensor_to_shape(to_pad, max_size, padding));
    sizes_ptr += sizes_num_columns;
  }


// 准备存储填充后张量的容器 buffers，并根据 split_sizes 遍历
std::vector<Tensor> buffers;
buffers.reserve(split_sizes.size());
int64_t next_buffer = 0;
auto sizes_ptr = sizes_data_start;
for (const auto split_size : split_sizes) {
  Tensor to_pad;
  IntArrayRef tensor_sizes(sizes_ptr, sizes_num_columns);
  if (split_size > 0) {
    // 如果 split_size 大于 0，则从 buffers_ 中获取张量并重塑到指定大小
    to_pad = buffers_[next_buffer++].reshape(tensor_sizes);
  } else {
    // 如果 split_size 等于 0，则创建一个空的张量
    to_pad = at::empty(tensor_sizes, buffer.options());
  }
  // 将填充后的张量添加到 buffers 中
  buffers.push_back(pad_tensor_to_shape(to_pad, max_size, padding));
  sizes_ptr += sizes_num_columns;
}



  auto ret_val = at::stack(buffers);


// 使用 at::stack 将填充后的张量列表 buffers 堆叠成一个张量 ret_val
auto ret_val = at::stack(buffers);



  // Pad output tensor to output_size if provided
  if (output_size.has_value()) {
    auto output_size_ = output_size.value();
    TORCH_CHECK(
        (int64_t)output_size_.size() == ret_val.dim(),
        "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
    for (int64_t i = 0; i < (int64_t)ret_val.dim(); i++) {
      TORCH_CHECK(
          output_size_[i] >= ret_val.size(i),
          "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
    }
    return pad_tensor_to_shape(ret_val, output_size_, padding);
  }
  return ret_val;


// 如果提供了 output_size，则将输出张量填充到指定的 output_size
if (output_size.has_value()) {
  auto output_size_ = output_size.value();
  // 检查 output_size 的长度是否与 NestedTensor 的维度匹配
  TORCH_CHECK(
      (int64_t)output_size_.size() == ret_val.dim(),
      "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
  // 检查 output_size 的每个维度是否大于等于 ret_val 对应维度的大小
  for (int64_t i = 0; i < (int64_t)ret_val.dim(); i++) {
    TORCH_CHECK(
        output_size_[i] >= ret_val.size(i),
        "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
  }
  // 返回填充到指定形状的输出张量
  return pad_tensor_to_shape(ret_val, output_size_, padding);
}
// 否则直接返回 ret_val
return ret_val;
// 定义嵌套张量的嵌入函数，用于在给定权重和索引的情况下进行嵌入操作
Tensor NestedTensor_embedding(
    const Tensor& weight,                      // 嵌入操作所使用的权重张量
    const Tensor& indices,                     // 嵌入操作的索引张量
    int64_t padding_idx,                       // 嵌入操作中的填充索引
    bool scale_grad_by_freq,                   // 是否按频率缩放梯度
    bool sparse) {                             // 是否使用稀疏张量进行嵌入

  const auto* nt_indices = get_nested_tensor_impl(indices);  // 获取索引的嵌套张量实现
  TORCH_CHECK(
      !weight.is_nested(),                     // 检查权重张量是否是嵌套张量
      "NestedTensor weight not supported for embedding"); // 权重张量不能是嵌套张量的错误提示

  TORCH_CHECK(indices.dim() < 3);              // 检查索引张量的维度小于3
  TORCH_CHECK(
      indices.dim() > 0,                       // 检查索引张量的维度大于0
      "NestedTensor embedding doesn't support empty indices."); // 索引张量为空时的错误提示

  TORCH_CHECK(weight.dim() == 2);              // 检查权重张量的维度为2
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_indices));  // 检查索引的嵌套张量实现是否是连续的
  TORCH_CHECK(
      weight.is_contiguous());                 // 检查权重张量是否是连续的

  const auto& indices_buffer = nt_indices->get_buffer();  // 获取索引张量的缓冲区
  auto result_buffer = at::embedding(           // 执行嵌入操作
      weight, indices_buffer, padding_idx, scale_grad_by_freq, sparse);

  const auto& sizes = nt_indices->get_nested_sizes();  // 获取嵌套张量索引的大小
  auto new_sizes = at::empty({sizes.size(0)}, sizes.options());  // 创建新的大小张量
  new_sizes.fill_(weight.sizes()[1]);            // 填充新大小张量
  new_sizes = new_sizes.reshape({new_sizes.size(0), 1});  // 调整新大小张量的形状
  new_sizes = at::cat({sizes, new_sizes}, 1);     // 拼接原大小和新大小张量
  // 创建并返回嵌套张量实现的张量
  return at::detail::make_tensor<NestedTensorImpl>(
      result_buffer.reshape({-1}), std::move(new_sizes));
}

// 用于使用torch_scatter.segment_reduce原型的非常基础的sum_dim函数
Tensor NestedTensor_sum_dim_CPU(
    const Tensor& self,                         // 待操作的嵌套张量
    OptionalIntArrayRef opt_dims,               // 可选的维度参数
    bool keepdim,                               // 是否保持减少后的维度
    std::optional<ScalarType> dtype) {          // 可选的数据类型参数

  // 只允许对最后一个维度进行减少操作
  auto dims = opt_dims.value_or(IntArrayRef{}); // 获取维度参数的值或空数组
  TORCH_CHECK(
      dims.size() == 1,                        // 检查维度参数的大小为1
      "NestedTensor only allows reduction of a single dimension for now."); // 维度参数错误的错误提示

  auto dim = maybe_wrap_dim(dims[0], self.dim());  // 获取包装后的维度索引
  TORCH_CHECK(
      dim == self.dim() - 1,                   // 检查减少操作是否在最后一个维度上进行
      "NestedTensor can only be reduced across the last dimension for now ",
      "got dimension ",
      dim,
      " instead.");                            // 维度不符合要求的错误提示

  // 目前始终保持减少后的维度
  // 避免嵌套张量为1维且keepdim=False的情况
  TORCH_CHECK(
      keepdim,                                 // 检查是否保持减少后的维度
      "NestedTensor always requires keepdim=True for now."); // keepdim=False的错误提示

  // 目前不支持acc_dtype参数
  TORCH_CHECK(
      !dtype,                                  // 检查是否提供了acc_dtype参数
      "NestedTensor does not support dtype argument for now."); // 不支持acc_dtype的错误提示

  auto nt_input = get_nested_tensor_impl(self); // 获取输入嵌套张量的实现
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_input),  // 检查输入的嵌套张量实现是否是连续的
      "NestedTensor does not support reductions when the input is noncontiguous for now."); // 输入非连续时的错误提示

  int64_t ntensors = nt_input->size(0);         // 获取嵌套张量的数量
  if (ntensors == 0) {
    return self;
  }
  const Tensor& buffer = nt_input->get_buffer();

  auto sizemat = nt_input->get_nested_sizes();
  // create output size tensor for keepdim=True
  // 根据 keepdim=True 创建输出尺寸张量
  auto output_sizemat = sizemat.clone();
  // 将输出尺寸张量的最后一列填充为1
  output_sizemat.select(1, -1).fill_(1);

  // 计算输出张量中的段数
  auto num_segments = at::prod(output_sizemat, -1);
  // 获取各段的长度
  auto segment_lengths = sizemat.select(1, -1);
  // 计算新张量的元素数
  const int64_t new_numel = at::sum(num_segments).item<int64_t>();
  // 创建一个空的输出缓冲区
  auto output_buffer = buffer.new_empty(IntArrayRef(new_numel));

  // 这段逻辑目前假设：
  // (1) 所有嵌套张量都是连续的
  // (2) 嵌套张量在缓冲区中是连续存储的
  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, buffer.scalar_type(), "nested_sum_dim_cpu", [&]() {
    auto* output_data = output_buffer.data_ptr<scalar_t>();
    const auto* input_data = buffer.const_data_ptr<scalar_t>();
    int64_t out_idx = 0, in_idx = 0;
    // 遍历每个嵌套张量
    for (const auto i : c10::irange(ntensors)) {
      // 获取当前嵌套张量的段数和段长度
      int64_t segments = num_segments[i].item<int64_t>();
      int64_t segment_length = segment_lengths[i].item<int64_t>();
      // 对每个段进行求和操作
      for (auto j = 0; j < segments; j++) {
        scalar_t res = 0;
        // 对当前段中的元素进行累加
        for (auto k = 0; k < segment_length; k++) {
          res += input_data[in_idx];
          in_idx += 1;
        }
        // 将求和结果存入输出缓冲区
        output_data[out_idx] = res;
        out_idx += 1;
      }
    }
  });

  // 将输出缓冲区和输出尺寸张量封装成输出张量并返回
  return wrap_buffer(output_buffer, output_sizemat);
}

// 从嵌套张量中选择特定维度上的索引值
Tensor select_nested(const Tensor& self, int64_t dim, int64_t index) {
  // 获取嵌套张量的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 获取嵌套张量的大小和步长信息
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
                           strides = NestedTensor_get_strides(self_ptr);
  // 获取存储偏移指针
  int64_t *offsets_ptr = self_ptr->get_storage_offsets().data_ptr<int64_t>();
  // 获取底层存储的缓冲区作为张量
  const at::Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor();
  // 将维度索引转换为正数
  int64_t positive_dim = at::maybe_wrap_dim(dim, self_ptr->dim());
  // 获取嵌套张量的数量
  int64_t ntensors = self_ptr->size(0);
  // 检查嵌套张量是否非空
  TORCH_CHECK_INDEX(ntensors > 0, "You can only select when the NT is not empty.");
  // 获取维度数
  int64_t ndims = static_cast<long>(sizes[0].size());
  // 如果选择的是第一个维度
  if (positive_dim == 0) {
    // 检查索引是否在有效范围内
    TORCH_CHECK_INDEX(
        index >= -ntensors && index < ntensors,
        "index ",
        index,
        " is out of bounds for dimension 0 with size ",
        ntensors);
    // 计算正数索引
    int64_t positive_index = index < 0 ? index + ntensors : index;
    // 返回按给定大小、步长和偏移创建的新视图张量
    return buffer.as_strided(
        sizes[positive_index],
        strides[positive_index],
        offsets_ptr[positive_index]);
  } else {
    // 创建新的大小、步长和偏移张量
    auto new_sizes = at::empty({ntensors, ndims-1}, TensorOptions().dtype(kLong));
    auto new_strides = at::empty({ntensors, ndims-1}, TensorOptions().dtype(kLong));
    auto new_offsets = at::empty({ntensors}, TensorOptions().dtype(kLong));
    // 遍历每个嵌套张量
    for (int64_t i : c10::irange(ntensors)) {
      int64_t *size_ptr = new_sizes[i].data_ptr<int64_t>();
      int64_t *stride_ptr = new_strides[i].data_ptr<int64_t>();

      int64_t dim_idx = 0;
      // 遍历每个维度
      for (int64_t j : c10::irange(ndims)) {
        // 如果不是选定的维度
        if (j != dim - 1) {
          size_ptr[dim_idx] = sizes[i][j];
          stride_ptr[dim_idx] = strides[i][j];
          ++dim_idx;
        } else {
          // 检查索引是否在有效范围内
          TORCH_CHECK_INDEX(
              index >= 0 && index < sizes[i][j],
              "index ",
              index,
              " is out of bounds for dimension ",
              j,
              " of the ",
              i,
              "th constituent tensor with size ",
              sizes[i][j]);
          // 计算新的偏移量
          new_offsets[i] = offsets_ptr[i] + index * strides[i][j];
        }
      }
    }
    // 创建新的嵌套视图张量
    return create_nested_view_tensor(self, new_sizes, new_strides, new_offsets);
  }

}

// 原生的嵌套张量的 dropout 函数
std::tuple<Tensor,Tensor> native_dropout_nested(const Tensor& input, double p, std::optional<bool> train) {
  auto input_ptr = get_nested_tensor_impl(input);
  // 获取底层存储的缓冲区、大小矩阵和步长矩阵
  const Tensor& input_buffer = input_ptr-> get_unsafe_storage_as_tensor(),
      & sizemat = input_ptr->get_nested_sizes(),
      & stridemat = input_ptr->get_nested_strides();
  // 获取存储偏移量
  const auto offsets = input_ptr->get_storage_offsets();
  Tensor output_buffer, mask_buffer;
  // 如果输入缓冲区为空
  if (input_buffer.numel() == 0) {
    // 克隆空的输入缓冲区和掩码缓冲区
    output_buffer = input_buffer.clone();
    mask_buffer = input_buffer.clone();
  }
  else {

    // 如果输入缓冲区不为空，则执行以下操作
    // 调用PyTorch的native_dropout函数，进行张量的dropout操作，并将结果分别存储在output_buffer和mask_buffer中
    std::tie(output_buffer, mask_buffer) = at::native_dropout(input_buffer, p, train);
  }
  // 对于常规的张量dropout，使用输入的大小和步幅来包装输出
  // 即如果输入不是连续的，输出也将不是连续的
  Tensor output = wrap_buffer(output_buffer, sizemat.clone(), stridemat.clone(), offsets.clone()),
      // 使用相同的大小和步幅包装mask_buffer，创建mask张量
      mask = wrap_buffer(mask_buffer, sizemat.clone(), stridemat.clone(), offsets.clone());
  // 返回output和mask的元组作为结果
  return std::make_tuple(output, mask);
// 定义一个函数 `softmax_nested`，用于在嵌套张量中沿指定维度应用 softmax
Tensor softmax_nested(
    // 输入参数 `input` 是待处理的嵌套张量
    const Tensor& input,
    // 参数 `dim` 表示要应用 softmax 的维度
    const int64_t dim,
    // 参数 `half_to_float` 表示是否将输入从半精度转换为单精度
    const bool half_to_float) {
  // 获取输入的嵌套张量实现指针
  auto input_ptr = get_nested_tensor_impl(input);
  // 获取嵌套张量中的张量数量
  int64_t ntensors = input_ptr->size(0);
  // 如果张量数量为 0，直接返回输入的克隆副本
  if (ntensors == 0) {
    return input.clone();
  }
  // 确保维度 `dim` 是正数
  int64_t positive_dim = at::maybe_wrap_dim(dim, input_ptr->dim());
  TORCH_CHECK(
      positive_dim >= 1,
      "Cannot apply softmax across nested dimension 0");
  
  // 创建一个连续的输出缓冲区
  // TODO: 在此处理想情况下应使用 `empty_like`，但嵌套张量尚不支持此功能
  // 因此在此处使用 `unsafe_storage_as_tensor` 是可以接受的，仅用于缓冲区选项和大小
  const Tensor& buffer = input_ptr->get_unsafe_storage_as_tensor(),
      & sizemat = input_ptr->get_nested_sizes();
  Tensor output_buffer = buffer.new_empty(buffer.sizes());
  // 使用缓冲区和尺寸矩阵创建输出嵌套视图张量
  Tensor output = wrap_buffer(output_buffer, sizemat.clone());
  
  // 调用张量 softmax
  // TODO: 对于 CPU，如果基准测试显示有必要，可能要使用 `parallel_for`
  //       来调用 `aten/src/ATen/native/cpu/SoftMaxKernel.cpp/softmax_kernel`
  //       1. 它包含 `parallel_for`，我们无法在多线程中再次多线程执行
  //       2. 不能在多线程中分发（在这种情况下为 `at::_softmax_out`）
  std::vector<Tensor> input_unbind = input.unbind(),
      output_unbind = output.unbind();
  // 对每个张量应用 `_softmax_out`，沿着 `positive_dim - 1` 的维度
  for (int64_t i = 0; i < ntensors; i++) {
    at::_softmax_out(
        output_unbind[i],
        input_unbind[i],
        positive_dim - 1,
        half_to_float);
  }
  // 返回应用 softmax 后的输出张量
  return output;
}

// 定义一个函数 `transpose_nested`，用于在嵌套张量中交换指定维度的位置
Tensor transpose_nested(const Tensor& self, int64_t dim0, int64_t dim1) {
  // 获取输入的嵌套张量实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 检查输入张量的维度数
  int64_t ndims = self_ptr->dim();
  // 确保维度 `dim0` 和 `dim1` 是正数，并且不相等
  int64_t positive_dim0 = at::maybe_wrap_dim(dim0, ndims),
      positive_dim1 = at::maybe_wrap_dim(dim1, ndims);
  if (positive_dim0 == positive_dim1) {
    return self;
  }
  TORCH_CHECK(positive_dim0 > 0 && positive_dim1 > 0, "Nested tensor dimension 0 cannot be transposed");
  
  // 从维度数中排除隐式批处理维度
  ndims--;
  positive_dim0--;
  positive_dim1--;
  
  // 获取嵌套张量的尺寸矩阵和步幅矩阵
  const Tensor& sizemat = self_ptr->get_nested_sizes(),
      & stridemat = self_ptr->get_nested_strides();
  
  // 创建一个列索引张量，用于交换 `dim0` 和 `dim1` 的列
  Tensor column_indices = sizemat.new_empty(ndims);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  std::iota(column_indices_ptr, column_indices_ptr + ndims, 0);
  column_indices_ptr[positive_dim0] = positive_dim1;
  column_indices_ptr[positive_dim1] = positive_dim0;
  
  // 创建交换维度后的尺寸矩阵和步幅矩阵
  Tensor sizemat_transposed = at::index_select(sizemat, 1, column_indices),
      stridemat_transposed = at::index_select(stridemat, 1, column_indices);
  
  // 创建一个新的嵌套视图张量，使用交换后的尺寸矩阵和步幅矩阵
  return create_nested_view_tensor(
      self, sizemat_transposed, stridemat_transposed, self_ptr->get_storage_offsets().clone());
}
// 当前函数用于处理嵌套张量的挤压操作，移除指定维度上的大小为1的维度。
Tensor squeeze_nested(const Tensor& self) {
  // 对于嵌套张量，不支持在没有指定维度参数的情况下进行挤压操作，因此抛出错误信息
  TORCH_CHECK(false,
    "squeeze(): For nested tensors, squeeze without the dim argument is not supported ",
    "at the moment, however you can use squeeze(Tensor self, int dim) instead ",
    "if you need this feature, please open an issue on github describing your use case.");
  // 返回原始张量
  return self;
}

// 对于给定的维度列表，挤压嵌套张量的指定维度
Tensor squeeze_dim_nested(const Tensor& self, IntArrayRef dims) {
  // 获取嵌套张量的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 获取张量的维度数
  int64_t ndim = self_ptr->dim();
  // 根据指定的维度列表创建掩码
  auto mask = at::dim_list_to_bitset(dims, ndim);
  // 检查是否对第一个维度进行挤压操作，如果是则抛出错误信息
  TORCH_CHECK(!mask.test(0),
    "squeeze(): For nested tensors, squeezing dimension 0 is not supported at the moment ",
    "if you need this feature, please open an issue on github describing your use case.");
  // 获取嵌套张量的大小和步长信息
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  const Tensor& stridemat = self_ptr->get_nested_strides();

  // 如果指定的维度不为1，则将对应位置的掩码位清除
  for (const auto d : c10::irange(ndim)) {
    if (mask.test(d)) {
      std::optional<int64_t> size_dim = self_ptr->opt_size(d);
      if (!(size_dim.has_value() && *size_dim == 1)) {
        mask.reset(d);
      }
    }
  }

  // 如果掩码中没有任何位被设置，则返回与原始张量分离的张量
  if (!mask.any()) {
    // 分离张量以避免触发 throw_error_if_base_and_tensor_are_same
    return self.detach();
  }

  // 如果维度数为2并且通过上述条件，则应该有一组嵌套的单例张量
  TORCH_CHECK(ndim > static_cast<int64_t>(1 + dims.size()),
    "squeeze(): For nested tensors, squeezing a nested tensor of singleton tensors is not ",
    "supported at the moment, if you need this feature, please open an issue on github",
    "describing your use case.");

  // 计算新的维度数，排除掉被挤压的维度
  const auto new_ndim = ndim - mask.count();
  // 创建列索引来选择保留的维度
  auto column_indices = sizemat.new_empty(new_ndim - 1);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  for (const auto d : c10::irange(1, ndim)) {
    if (!mask.test(d)) {
      *column_indices_ptr++ = d - 1;
    }
  }

  // 根据列索引选择大小和步长的子集
  auto sizemat_squeezed = at::index_select(sizemat, 1, column_indices);
  auto stridemat_squeezed = at::index_select(stridemat, 1, column_indices);

  // 创建一个新的挤压视图张量并返回
  return create_nested_view_tensor(
    self, sizemat_squeezed, stridemat_squeezed, self_ptr->get_storage_offsets().clone());
}

// 对给定的单个维度进行挤压嵌套张量的操作
Tensor squeeze_dim_nested(const Tensor& self, int64_t dim) {
  return squeeze_dim_nested(self, IntArrayRef{dim});
}
// 在嵌套张量中添加维度的操作函数，返回处理后的张量
Tensor unsqueeze_nested(const Tensor& self, int64_t dim) {
  // 获取嵌套张量的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 获取嵌套张量的维度
  int64_t ndim = self_ptr->dim();
  // 根据传入的维度参数，计算出正确的包装后的维度
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, ndim + 1);
  // 检查 wrapped_dim 是否大于 0，如果不是，抛出错误信息
  TORCH_CHECK(wrapped_dim > 0,
    "unsqueeze(): For nested tensors, unsqueezing dimension 0 is not supported at the moment ",
    "if you need this feature, please open an issue on github describing your use case.");
  
  // 获取嵌套张量的尺寸矩阵和步长矩阵
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  const Tensor& stridemat = self_ptr->get_nested_strides();
  
  // 计算包含新维度的尺寸矩阵
  auto mat_dim = wrapped_dim - 1;
  Tensor new_size = sizemat.new_ones({sizemat.size(0), 1});
  Tensor sizemat_unsqueezed = at::cat({sizemat.slice(1, 0, mat_dim),
                                       new_size,
                                       sizemat.slice(1, mat_dim, ndim)}, 1);
  
  // 计算包含新维度的步长矩阵
  Tensor new_stride;
  if (wrapped_dim == ndim) {
    new_stride = stridemat.new_ones({stridemat.size(0), 1});
  } else {
    new_stride = (stridemat.select(1, mat_dim) * sizemat.select(1, mat_dim)).unsqueeze(-1);
  }
  Tensor stridemat_unsqueezed = at::cat({stridemat.slice(1, 0, mat_dim),
                                         new_stride,
                                         stridemat.slice(1, mat_dim, ndim)}, 1);
  
  // 创建并返回嵌套视图张量
  return create_nested_view_tensor(
      self, sizemat_unsqueezed, stridemat_unsqueezed, self_ptr->get_storage_offsets().clone());
}

// 支持 `view_nested` 和 `reshape_nested` 的工具函数
namespace {
// 计算尺寸和步长矩阵的函数，用于处理视图重塑操作
// 参数:
//     sizes: 原始嵌套张量的尺寸
//     strides: 原始嵌套张量的步长
//     proposed_shape: 用户建议的新形状
//     op: 新尺寸和步长矩阵的选项
// 返回:
//     是否可视
//     重塑后的尺寸矩阵
//     重塑后的步长矩阵（如果不可视则未完全填充）
inline std::tuple<bool, Tensor, Tensor> NestedTensor_compute_size_stride(
    const std::vector<IntArrayRef>& sizes,
    const std::vector<IntArrayRef>& strides,
    const IntArrayRef& proposed_shape,
    const c10::TensorOptions& op) {
  // 计算张量数量、底层维度、重塑后的底层维度
  int64_t ntensors = sizes.size(),
      ndims_underlying = sizes[0].size(),
      ndims_underlying_reshaped = proposed_shape.size() - 1;
  bool viewable = true;
  
  // 创建空的重塑后尺寸和步长矩阵
  Tensor sizemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op),
      stridemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op);
  
  // 获取可变指针以填充新的尺寸和步长矩阵
  int64_t* sizemat_reshaped_ptr = sizemat_reshaped.mutable_data_ptr<int64_t>(),
      * stridemat_reshaped_ptr = stridemat_reshaped.mutable_data_ptr<int64_t>();
  
  // 遍历每个张量进行重塑计算
  for (int64_t itensor = 0; itensor < ntensors; itensor++) {
    const IntArrayRef& size = sizes[itensor],
        & stride = strides[itensor];
    
    // 计算重塑后的尺寸
    std::vector<int64_t> size_reshaped_vector(proposed_shape.begin() + 1, proposed_shape.end());
    
    // 仅允许一个已存在维度具有 proposed_shape == -1
    int64_t infer_index_old = -1;
    
    // 保留一些负尺寸以供推断
    // 这里应该有更多的代码，但由于限制，无法显示完整的函数内容
    // 如果底层张量的维度少于重塑后的维度
    if (ndims_underlying < ndims_underlying_reshaped) {
      int64_t numel = 1, numel_reshaped = 1;
      // 替换旧维度的负大小为旧大小
      for (int64_t idim = 0; idim < ndims_underlying; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        // 检查重塑后的大小不为负
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          // 检查仅有一个维度可以推断
          TORCH_CHECK(infer_index_old == -1, "only one dimension can be inferred");
          size_reshaped = size[idim];
          infer_index_old = idim;
        }
        numel *= size[idim];
        numel_reshaped *= size_reshaped;
      }
      // 推断新维度的负大小
      int64_t infer_index = -1;
      for (int64_t idim = ndims_underlying; idim < ndims_underlying_reshaped; idim++) {
        const int64_t& size_reshaped = size_reshaped_vector[idim];
        if (size_reshaped >= 0) {
          numel_reshaped *= size_reshaped;
        }
        else if (size_reshaped == -1) {
          if (infer_index > -1) {
            throw std::runtime_error("only one dimension can be inferred");
          }
          else {
            infer_index = idim;
          }
        }
        else {
          AT_ERROR("invalid shape dimension ", size_reshaped);
        }
      }
      // 见注解 [嵌套张量的特殊大小规则]
      TORCH_CHECK(infer_index == -1, "nested tensor does not infer shape");
      TORCH_CHECK(
          numel == numel_reshaped,
          "shape '", proposed_shape, "' ",
          "is invalid for input of size ", numel);
    }
    // 所有负大小可以被替换
    else {
      int64_t numel = 1, numel_reshaped = 1;
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        // 检查重塑后的大小不为负
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          size_reshaped = size[idim];
        }
        numel *= size[idim];
        numel_reshaped *= size_reshaped;
      }
      for (int64_t idim = ndims_underlying_reshaped; idim < ndims_underlying; idim++) {
        numel *= size[idim];
      }
      TORCH_CHECK(
          numel == numel_reshaped,
          "shape '", proposed_shape, "' ",
          "is invalid for input of size ", numel);
    }
    // 将 size_reshaped_vector 转换为 IntArrayRef 对象
    IntArrayRef size_reshaped(size_reshaped_vector);
    // 计算重塑后的步长
    auto opt_stride_reshaped = at::detail::computeStride(size, stride, size_reshaped);
    // 可以作为视图进行重塑
    // 如果 opt_stride_reshaped 包含值
    if (opt_stride_reshaped.has_value()) {
      // 解引用得到 stride_reshaped 引用
      const IntArrayRef& stride_reshaped = *opt_stride_reshaped;
      // 将重塑后的尺寸和步长填充到 sizemat_reshaped_ptr 和 stridemat_reshaped_ptr 中
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        sizemat_reshaped_ptr[idim] = size_reshaped[idim];
        stridemat_reshaped_ptr[idim] = stride_reshaped[idim];
      }
      // 移动 sizemat_reshaped_ptr 和 stridemat_reshaped_ptr 指针到下一个位置
      sizemat_reshaped_ptr += ndims_underlying_reshaped;
      stridemat_reshaped_ptr += ndims_underlying_reshaped;
    }
    // 如果 opt_stride_reshaped 不包含值，即无法重塑为视图
    else {
      // 设置 viewable 为 false，表示无法重塑为视图
      viewable = false;
      // 将重塑后的尺寸填充到 sizemat_reshaped_ptr 中
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        sizemat_reshaped_ptr[idim] = size_reshaped[idim];
      }
      // 移动 sizemat_reshaped_ptr 指针到下一个位置
      sizemat_reshaped_ptr += ndims_underlying_reshaped;
    }
  }
  // 返回包含 viewable 状态以及 sizemat_reshaped 和 stridemat_reshaped 的元组
  return std::make_tuple(viewable, sizemat_reshaped, stridemat_reshaped);
} // namespace

// Note [Special size rule for nested tensor]
// Instead of inferring size, -1 means "inherit the old size", so:
// * negative size is legal for a ragged dimension
// * however, we only allow one -1
// In principle we could still infer a dimension,
// we are designing a better semantics to include both inheritance and inference

// 定义一个函数 view_nested，用于将输入的嵌套张量按照指定的形状进行视图重塑
Tensor view_nested(const Tensor& self, IntArrayRef proposed_shape) {
  // 检查 proposed_shape 不为空，否则抛出错误信息
  TORCH_CHECK(
      !proposed_shape.empty(),
      "shape '[]' is invalid for a nested tensor");
  // 获取 self 的 NestedTensor 实现指针
  auto self_ptr = get_nested_tensor_impl(self);

  // 获取重塑前的基本信息
  int64_t ntensors = self_ptr->size(0);
  // 检查 ntensors 必须大于 0，否则抛出错误信息
  TORCH_CHECK(
      ntensors > 0,
      "empty nested tensor cannot be reshaped");

  // 获取重塑后的基本信息
  int64_t ntensors_reshaped = proposed_shape[0];
  // 检查 ntensors 是否等于 ntensors_reshaped，否则抛出错误信息
  TORCH_CHECK(
      ntensors == ntensors_reshaped,
      "view: For now nested view cannot change or infer the implicit batch dimension");

  // 获取 NestedTensor 的 sizes 和 strides
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);

  // 重塑底层张量的维度不会改变偏移量
  // 确定重塑后的大小和步幅
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  bool viewable;
  Tensor sizemat_reshaped, stridemat_reshaped;
  // 调用 NestedTensor_compute_size_stride 计算重塑后的大小和步幅
  std::tie(viewable, sizemat_reshaped, stridemat_reshaped) = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());

  // 检查是否可以进行视图重塑，否则抛出错误信息
  TORCH_CHECK(
      viewable,
      "view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces). "
      "Use .reshape(...) instead.");

  // 创建一个新的嵌套视图张量并返回
  return create_nested_view_tensor(self, sizemat_reshaped, stridemat_reshaped, self_ptr->get_storage_offsets().clone());
}

/**
 * Create a buffer tensor that is a view of self
 *
 * This serves as the boundary between nested and non nested tensor
 * view conversions
 *
 * @return Returns a new non nested tensor that
 * aliases the same storage as self
 */
// 创建一个函数 values_nested，用于从嵌套张量创建一个缓冲张量视图
Tensor values_nested(const Tensor& self) {
  // 内部断言检查 self 是否为嵌套张量，否则抛出错误信息
  TORCH_INTERNAL_ASSERT(self.is_nested(), "Can only create a buffer from Nested Tensor");
  // 获取 self 的 NestedTensor 实现指针
  auto* nt_self = get_nested_tensor_impl(self);
  // 返回从嵌套张量获取不安全存储作为张量的结果
  return nt_self->get_unsafe_storage_as_tensor();
}

/**
 * Create a nested tensor that is a view of a buffer
 *
 * This serves as the boundary between non nested tensor and nested
 * view conversions
 *
 * @return Returns a nested tensor that
 * aliases the same storage as buffer
 */
// 创建一个函数 _nested_view_from_buffer，用于从缓冲区创建一个嵌套张量视图
Tensor _nested_view_from_buffer(
    const Tensor& buffer,
    const Tensor& nested_sizes,
    const Tensor& nested_strides,
    ...
    # 确保输入的 buffer 不是嵌套的，而是普通的张量缓冲区
    TORCH_INTERNAL_ASSERT(
        !buffer.is_nested(),
        "Can only a create Nested Tensor from a normal tensor buffer");
    
    # 确保输入的 buffer 是一维的
    TORCH_INTERNAL_ASSERT(buffer.dim() == 1, "The input buffer must be flat");
    
    # 确保 nested_sizes 张量是二维的
    TORCH_INTERNAL_ASSERT(nested_sizes.dim() == 2, "Expected the nested size tensor to be two dimensional.");
    
    # 计算 nested_sizes 张量中所有元素的总数
    uint64_t num_elements_nested_size = at::prod(nested_sizes, 1).sum().item<int64_t>();
    
    # 计算 buffer 中存储的总字节数除以元素的字节大小，得到 buffer 的存储大小
    uint64_t buffer_storage_size = buffer.storage().nbytes() / buffer.dtype().itemsize();
    
    # 确保 buffer 的存储大小等于 nested_sizes 中的元素总数
    TORCH_INTERNAL_ASSERT(
        buffer_storage_size == num_elements_nested_size,
        "The number of elements in the buffer must equal the nested tensor size but buffer size: ",
        buffer_storage_size,
        " and nested tensor size: ",
        num_elements_nested_size,
        ".");
    
    # 确保 nested_strides 张量是二维的
    TORCH_INTERNAL_ASSERT(nested_strides.dim() == 2, "Expected the nested stride tensor to be two dimensional.");
    
    # 确保 nested_sizes 和 nested_strides 张量的第一维度大小相等
    TORCH_INTERNAL_ASSERT(nested_sizes.size(0) == nested_strides.size(0), "Expected the first dimension of nested size and nested stride tensor to be equal.");
    
    # 确保 nested_strides 张量的第一维度大小与 storage_offsets 张量的长度相等
    TORCH_INTERNAL_ASSERT(nested_strides.size(0) == storage_offsets.size(0), "Expected the first dimension of nested stride tensor to equal the length of offsets.");
    
    # 使用给定的参数创建一个 NestedTensorImpl 张量对象并返回
    return at::detail::make_tensor<NestedTensorImpl>(
      c10::TensorImpl::VIEW,
      buffer,
      nested_sizes,
      nested_strides,
      storage_offsets);
}

// 返回一个包含计算后的连续步长和偏移量的元组
std::tuple<Tensor, Tensor> _nested_compute_contiguous_strides_offsets(const Tensor& nested_size) {
  return std::make_tuple(
      construct_nested_strides(nested_size),  // 构建嵌套步长
      construct_offsets(nested_size));        // 构建偏移量
}

// 查看“嵌套张量特殊尺寸规则”注释
Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape) {
  TORCH_CHECK(
      !proposed_shape.empty(),  // 检查是否提供了有效的形状
      "shape '[]' is invalid for a nested tensor");
  auto self_ptr = get_nested_tensor_impl(self);
  // 重塑前的基本信息
  int64_t ntensors = self_ptr->size(0);  // 获取嵌套张量的数量
  TORCH_CHECK(
      ntensors > 0,  // 检查空嵌套张量无法重塑
      "empty nested tensor cannot be reshaped");
  // 重塑后的基本信息
  int64_t ntensors_reshaped = proposed_shape[0];  // 获取重塑后的张量数量
  TORCH_CHECK(
      ntensors == ntensors_reshaped,  // 检查是否改变或推断了隐式批处理维度
      "reshape: For now nested reshape cannot change or infer the implicit batch dimension");
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  // 重塑底层张量维度不会改变偏移量
  // 确定重塑的大小和步长
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  bool viewable{false};
  Tensor sizemat_reshaped, stridemat_reshaped;
  std::tie(viewable, sizemat_reshaped, stridemat_reshaped) = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  if (viewable) {
    return self.view(proposed_shape);  // 如果可以直接查看，返回视图
  }
  else {
    return self.clone(at::MemoryFormat::Contiguous).view(proposed_shape);  // 否则克隆为连续内存格式再返回视图
  }
}

// Jagged布局的重塑操作，支持符号整数数组的形状
Tensor reshape_nested_symint(const Tensor& self, SymIntArrayRef proposed_shape) {
  // 如果布局是Jagged
  if (self.layout() == at::kJagged) {
    // TODO: 扩展解析以处理其他可视情况
    bool viewable = self.is_contiguous();
    return (
        viewable ? self.view_symint(proposed_shape) :  // 如果可视，直接返回符号整数视图
        self.clone(at::MemoryFormat::Contiguous).view_symint(proposed_shape)  // 否则克隆为连续内存格式再返回符号整数视图
    );
  }

  return reshape_nested(self, C10_AS_INTARRAYREF_SLOW(proposed_shape));  // 否则使用慢速转换为整数数组的形状进行一般重塑
}

// 将张量重塑为与其他张量相同的嵌套结构
Tensor reshape_as_nested(const Tensor& self, const Tensor& other) {
  // 如果布局是Jagged
  if (self.layout() == at::kJagged) {
    return self.reshape_symint(other.sym_sizes());  // 使用其他张量的符号尺寸进行符号整数重塑
  }

  auto other_ptr = get_nested_tensor_impl(other);
  // TODO: 这是为了复制其他ptr->opt_sizes_
  //       如果未来提供了访问器，可以替换此操作
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < other_ptr->dim(); i++) {
    std::optional<int64_t> opt_size = other_ptr->opt_size(i);
    if (opt_size.has_value()) {
      sizes.push_back(*opt_size);
    }
    else {
      sizes.push_back(-1);
    }
  }
  // 使用其他.opt_sizes_进行重塑
  return self.reshape(sizes);
}

// 将嵌套张量标准化为正态分布
Tensor& normal_nested_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.normal_(mean, std, gen);  // 对内部缓冲区进行正态分布标准化
  return self;
}
// 如果两个嵌套张量的大小在指定维度上兼容，则返回 true
// 在指定的连接维度之外，张量的大小应该匹配
static bool can_cat_nested_sizes(const Tensor& nested_sizes1, const Tensor& nested_sizes2, int64_t cat_dim) {
  // 检查两个嵌套张量的总大小是否相同
  if (nested_sizes1.sizes() != nested_sizes2.sizes()) {
    return false;
  }

  auto nested_sizes1_ptr = nested_sizes1.data_ptr<int64_t>();
  auto nested_sizes2_ptr = nested_sizes2.data_ptr<int64_t>();
  const auto num_components = nested_sizes1.size(0);
  const auto num_dims = nested_sizes1.size(1);
  for (auto c : c10::irange(num_components)) {
    for (auto d : c10::irange(num_dims)) {
      // 对于每个组件和维度，检查是否需要跳过连接维度
      // 由于减去 1 用于批量维度
      auto component_cat_dim = cat_dim - 1;
      if (d == component_cat_dim) {
        continue;
      }
      // 检查除连接维度之外的维度是否匹配
      if (nested_sizes1_ptr[c * num_dims + d] != nested_sizes2_ptr[c * num_dims + d]) {
        return false;
      }
    }
  }

  return true;
}

// 将表示为不规则（jagged）的 NTs 列表进行连接
static Tensor cat_nested_as_jagged(
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  // 获取列表中第一个张量的引用和其维度信息
  const auto first_item = tensors[0].get();
  const auto first_item_dim = first_item.dim();
  const auto first_item_batch_size = first_item.size(0);
  // 用于存储不规则视图的向量
  std::vector<Tensor> jagged_views;
  for (auto i : c10::irange(tensors.size())) {
    auto t = tensors[i].get();
    // 断言每个张量都是嵌套的
    TORCH_CHECK(t.is_nested(),
        "cat(): expected each tensor in given list to be nested");
    // 断言每个张量是连续的
    TORCH_CHECK(t.is_contiguous(),
        "cat(): only contiguous nested tensors are supported");
    if (i > 0) {
      // 检查所有嵌套张量在连接维度之外是否具有匹配的不规则结构
      TORCH_CHECK(
          can_cat_nested_sizes(
              get_nested_tensor_impl(first_item)->get_nested_sizes(),
              get_nested_tensor_impl(t)->get_nested_sizes(),
              dim),
          "cat(): expected all nested tensors to have matching ragged structures outside of the concatenated dim");
    }
    // 仅支持输入格式 (B, *, D_0, D_1, ...)
    // 即在批量维度旁边最多有一个不规则维度
    auto *nt_impl = get_nested_tensor_impl(t);
    std::vector<int64_t> jagged_size;
    jagged_size.push_back(-1);
    for (auto d : c10::irange(first_item_dim - 2)) {
      // 断言只支持具有单个不规则维度的嵌套张量，紧邻批量维度
      TORCH_CHECK(nt_impl->opt_size(d + 2).has_value(),
          "cat(): only nested tensors with a single ragged dim next to the batch dim are supported");
      jagged_size.push_back(nt_impl->size(d + 2));
    }
    // 获取嵌套张量的缓冲区，并按照不规则大小进行视图
    auto jagged = nt_impl->get_buffer().view(jagged_size);
    jagged_views.push_back(jagged);
  }

  // 在不规则视图上进行连接操作
  auto new_buffer = at::cat(jagged_views, dim - 1);

  // 将结果封装成嵌套张量
  const auto component_dim = first_item_dim - 1;
  auto new_dim_size = new_buffer.size(dim - 1);
  auto new_sizes = get_nested_tensor_impl(tensors[0].get())->get_nested_sizes().clone();
  auto new_sizes_ptr = new_sizes.data_ptr<int64_t>();
  for (const auto i : c10::irange(first_item_batch_size)) {
    # 将新维度尺寸写入指针数组中的特定位置
    new_sizes_ptr[i * component_dim + (dim - 1)] = new_dim_size;
  }
  # 使用新的缓冲区视图和更新后的尺寸创建一个 NestedTensorImpl 类型的张量
  return at::detail::make_tensor<NestedTensorImpl>(
      new_buffer.view(-1), new_sizes);
} // 关闭 namespace at

static Tensor cat_nested_impl(
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  // 根据第一个张量的情况，可能调整维度参数 dim
  dim = maybe_wrap_dim(dim, tensors[0].get());
  if (dim == 0) {
    // 处理 dim=0 的简单情况：连接 NT 组件
    std::vector<at::Tensor> buffers;
    std::vector<at::Tensor> sizes;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      // 检查每个给定列表中的张量是否为嵌套张量
      TORCH_CHECK(
          t.is_nested(), "Expected each tensor in given list to be nested.");
      // 检查每个给定列表中的张量是否是连续的
      TORCH_CHECK(
          t.is_contiguous(),
          "Expected each tensor in given list to be contiguous.");
      auto t_ptr = get_nested_tensor_impl(t);
      // 将嵌套张量的缓冲区视图加入到缓冲区列表中
      buffers.push_back(t_ptr->get_buffer().view({-1}));
      // 将嵌套张量的嵌套尺寸加入到尺寸列表中
      sizes.push_back(t_ptr->get_nested_sizes());
    }
    // 创建并返回嵌套张量的新实现
    return at::detail::make_tensor<NestedTensorImpl>(
        at::cat(buffers).view({-1}), at::cat(sizes, 0));
  }

  // 注意：对于其他维度的支持仅限于可表示为不规则数组的嵌套张量
  // 调用函数以将嵌套张量连接为不规则数组形式的张量
  return cat_nested_as_jagged(tensors, dim);
}

// 对外接口函数：连接给定张量列表中的嵌套张量
Tensor cat_nested(const ITensorListRef& tensors, int64_t dim) {
  // 将输入张量列表实例化为 MaterializedITensorListRef
  auto materialized = tensors.materialize();
  // 调用内部实现函数以执行连接操作，处理维度参数
  return cat_nested_impl(materialized, at::legacy_cat_wrap_dim(dim, materialized));
}

} // 关闭 namespace native
} // 关闭 namespace at
```