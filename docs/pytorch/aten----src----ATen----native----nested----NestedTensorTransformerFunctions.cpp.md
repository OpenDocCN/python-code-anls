# `.\pytorch\aten\src\ATen\native\nested\NestedTensorTransformerFunctions.cpp`

```
# 包含 ATen 库的头文件，用于深度学习框架 PyTorch 的张量操作
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

# 包含 ATen 库的主头文件和相关功能函数
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

# 包含字符串视图和异常处理的实用工具
#include <c10/util/string_view.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

# 定义 ATen 命名空间
namespace at {
# 定义 native 命名空间，包含 PyTorch 的原生实现
namespace native {
# 匿名命名空间，用于限制函数的作用域
namespace {

# 检查嵌套张量与矩阵约束条件的函数
inline void check_nested_tensor_matrix_constraints(
    const Tensor& nested_tensor,
    const Tensor& dense_matrix,
    c10::string_view caller) {
  # 获取嵌套张量的实现指针
  auto* nt_input = get_nested_tensor_impl(nested_tensor);
  # 内部断言，确保嵌套张量实现指针非空
  TORCH_INTERNAL_ASSERT(nt_input != nullptr);
  # 检查是否支持输入为嵌套张量时，权重矩阵不能为嵌套矩阵
  TORCH_CHECK(
      !dense_matrix.is_nested(),
      caller,
      " does not support nested weight when input is a nested tensor.")
  # TODO: 支持非连续情况
  # 暂时报错，线性函数只支持连续的嵌套张量
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_input),
      "for now linear only supports contiguous nested tensor");
  # 检查嵌套张量的维度是否为3，密集矩阵的维度是否为2
  TORCH_CHECK(
      nested_tensor.dim() == 3 && dense_matrix.dim() == 2,
      caller,
      " requires nested_tensor.dim == 3 and dense_matrix.dim == 2."
      " Nested tensor dim: ",
      nested_tensor.dim(),
      ". Dense tensor dim: ",
      dense_matrix.dim());
  # 获取一致的嵌套张量最后一个维度
  const auto last_dim = get_consistent_last_dim_of_nested_tensor(*nt_input);
  # 对于线性函数，检查第二个维度，因为会转置后进行矩阵乘法
  int64_t dim_constraint = (caller == "Linear") ? 1 : 0;
  auto dense_size = dense_matrix.size(dim_constraint);
  # 检查嵌套张量的 'last_dim' 是否与权重矩阵的大小匹配
  TORCH_CHECK(
      last_dim == dense_size,
      "Shape mismatch for NestedTensor ",
      caller,
      ": Expected input's (a nested tensor) 'last_dim' to equal 'weight.size(",
      dim_constraint,
      "),",
      " but got: last_dim = ",
      last_dim,
      ", and weight.size(",
      dim_constraint,
      ") = ",
      dense_size);
}

} // namespace

# 嵌套线性操作函数，接收输入张量、权重矩阵和可选的偏置张量
Tensor nested_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  # 检查嵌套张量与矩阵约束条件，调用者为 "Linear"
  check_nested_tensor_matrix_constraints(input, weight, c10::string_view{"Linear"});
  # 获取嵌套张量的实现指针
  auto* nt_input = get_nested_tensor_impl(input);
  # 获取嵌套张量的缓冲区
  const Tensor& input_buffer = nt_input->get_buffer();
  # 使用 ATen 提供的线性函数进行矩阵乘法，结果保存在结果缓冲区中
  Tensor result_buffer =
      at::linear(input_buffer.reshape({-1, weight.size(1)}), weight, bias_opt);
  # 将结果缓冲区重塑为一维张量
  result_buffer = result_buffer.reshape({-1});
  # 获取权重矩阵的第一个维度大小
  int64_t weight_size_1 = weight.size(0);
  # 复制嵌套张量的嵌套尺寸
  Tensor new_sizes = nt_input->get_nested_sizes().clone();
  # 现在 new_sizes 每行的最后一个条目应该是 weight_size_1
  new_sizes.index_put_({at::indexing::Slice(), -1}, weight_size_1);
  # 将结果缓冲区和新的尺寸包装成嵌套张量
  return wrap_buffer(result_buffer, new_sizes);
}
Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  // 检查输入张量是否符合嵌套张量矩阵乘法的约束条件
  check_nested_tensor_matrix_constraints(self, other, c10::string_view{"Matmul"});
  // 获取自身的嵌套张量实现指针
  auto* nt_self = get_nested_tensor_impl_or_null(self);
  // 获取自身的缓冲区张量
  const Tensor& self_buffer = nt_self->get_buffer();
  // 计算矩阵乘法 self_buffer.reshape({-1, other.sizes()[0]}) * other
  Tensor result_buffer =
      at::mm(self_buffer.reshape({-1, other.sizes()[0]}), other);
  // 将结果张量重新调整形状为一维
  result_buffer = result_buffer.reshape({-1});
  // 获取 other 张量的第二个维度大小
  int64_t other_size_1 = other.sizes()[1];
  // 复制嵌套张量的尺寸信息
  Tensor new_sizes = nt_self->get_nested_sizes().clone();
  // 更新 new_sizes 的每一行的最后一个条目为 other_size_1
  new_sizes.index_put_({at::indexing::Slice(), -1}, other_size_1);
  // 使用结果缓冲区和新的尺寸信息包装成嵌套张量并返回
  return wrap_buffer(result_buffer, new_sizes);
}

Tensor NestedTensor_times_Tensor_plus_Tensor_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    std::optional<bool> use_gelu) {
  // 特殊情况：alpha * NT * T + beta * T
  const auto* nt_mat1 = get_nested_tensor_impl_or_null(mat1);
  // 断言 nt_mat1 不为空
  TORCH_INTERNAL_ASSERT(nt_mat1 != nullptr);
  // 断言 mat2 不是嵌套张量
  TORCH_INTERNAL_ASSERT(!mat2.is_nested());
  // 断言 self 不是嵌套张量
  TORCH_INTERNAL_ASSERT(!self.is_nested());
  // 断言 nt_mat1 是连续的嵌套张量实现
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(nt_mat1));
  // 断言 mat1 和 mat2 的维度分别为 3 和 2
  TORCH_INTERNAL_ASSERT(mat1.dim() == 3 && mat2.dim() == 2);
  // 断言 nt_mat1 的最后一个维度与 mat2 的第一个维度大小一致
  TORCH_INTERNAL_ASSERT(
      get_consistent_last_dim_of_nested_tensor(*nt_mat1) == mat2.sizes()[0]);
  // 获取 mat1 的缓冲区张量
  const Tensor& mat1_buffer = nt_mat1->get_buffer();
  // 根据 use_gelu 的值选择调用 addmm 或 _addmm_activation
  Tensor result_buffer = !use_gelu.has_value()
      ? at::addmm(
            self, mat1_buffer.reshape({-1, mat2.sizes()[0]}), mat2, beta, alpha)
      : at::_addmm_activation(
            self,
            mat1_buffer.reshape({-1, mat2.sizes()[0]}),
            mat2,
            beta,
            alpha,
            *use_gelu);
  // 将结果张量重新调整形状为一维
  result_buffer = result_buffer.reshape({-1});
  // 获取 mat2 的第二个维度大小
  int64_t other_size_1 = mat2.sizes()[1];
  // 复制 nt_mat1 的尺寸信息
  Tensor new_sizes = nt_mat1->get_nested_sizes().clone();
  // 更新 new_sizes 的每一行的最后一个条目为 other_size_1
  new_sizes.index_put_({at::indexing::Slice(), -1}, other_size_1);
  // 使用结果缓冲区和新的尺寸信息创建嵌套张量并返回
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(result_buffer), std::move(new_sizes));
}

Tensor NestedTensor_add_NestedTensor_in_place(
    const Tensor& self,
    const Tensor& other) {
  // 断言 self 和 other 是嵌套张量
  TORCH_INTERNAL_ASSERT(self.is_nested() && other.is_nested());
  // 获取 self 和 other 的嵌套张量实现引用
  const auto& nt_self = *get_nested_tensor_impl(self);
  const auto& nt_other = *get_nested_tensor_impl(other);

  // 获取 self 和 other 的尺寸信息
  const auto& self_sizes = nt_self.get_nested_sizes();
  const auto& other_sizes = nt_other.get_nested_sizes();

  // 断言 self 和 other 的尺寸信息相等
  TORCH_CHECK(at::equal(self_sizes, other_sizes));
  // 断言 nt_self 和 nt_other 是连续的嵌套张量实现
  TORCH_INTERNAL_ASSERT(
      nested_tensor_impl_is_contiguous(&nt_self) &&
      nested_tensor_impl_is_contiguous(&nt_other));
  // 将 nt_other 的缓冲区张量的每一行加到 nt_self 的缓冲区张量的每一行上
  nt_self.get_buffer().view({-1}).add_(nt_other.get_buffer().view({-1}));
  // 返回 self 本身
  return self;
}
# 计算嵌套张量的 softmax dropout 结果
Tensor NestedTensor_softmax_dropout(const Tensor& self, const Tensor& query) {
  # 获取查询张量的嵌套张量实现
  const auto* query_nt = get_nested_tensor_impl_or_null(query);
  # 内部断言：查询张量的嵌套实现不为空
  TORCH_INTERNAL_ASSERT(query_nt != nullptr);
  # 内部断言：查询张量的嵌套实现是连续的
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(query_nt));

  # 获取嵌套张量的大小信息
  const Tensor& sizes = query_nt->get_nested_sizes();
  # 获取嵌套张量中张量的数量
  const auto num_tensors = sizes.sizes()[0];

  # 创建一个与输入张量相同形状的空张量作为输出，使用连续的内存格式
  auto output = at::empty_like(self, {}, at::MemoryFormat::Contiguous);
  # 内部断言：输出张量是连续的
  TORCH_INTERNAL_ASSERT(output.is_contiguous());

  # 获取输入张量的最大序列长度
  const auto max_seq_len = self.sizes()[2];

  # 遍历每个嵌套张量中的子张量
  for (int64_t i = 0; i < num_tensors; i++) {
    # 获取当前子张量的序列长度
    auto seq_len = sizes.index({i, 0}).item<int64_t>();
    # 对输入张量的子序列进行索引，从而获取子序列的视图
    auto subseq = self.index(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)});
    # 对子序列进行 softmax 操作，沿着最后一个维度进行计算
    auto subscores = at::softmax(subseq, subseq.dim() - 1);
    # 将计算得到的 softmax 分数放回到输出张量的相应位置
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)},
        subscores);
    # 将超出序列长度的部分填充为零
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(seq_len, max_seq_len)},
        0);
    # 将超出序列长度的部分填充为零
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(seq_len, max_seq_len),
         indexing::Slice(0, max_seq_len)},
        0);
  }
  # 返回最终的输出张量
  return output;
}

# 在 CUDA 上执行嵌套张量的 softmax dropout
Tensor NestedTensor_softmax_dropout_cuda(const Tensor& self, const Tensor& query) {
  # 初始化一个空的注意力掩码作为可选项
  std::optional<Tensor> attn_mask;

  # 将查询张量转换为掩码张量，对应于 CUDA 设备
  attn_mask = NestedTensor_to_mask(query, 2, self.size(2));
  attn_mask = attn_mask->to(query.device(), /*non-blocking=*/true);
  # 调用底层函数对输入张量执行带掩码的 softmax 操作，指定掩码类型为 1
  return _masked_softmax(self, *attn_mask, self.dim() - 1, /*mask type */ 1 );  // NestedTensor_to_mask produces a BxT mask
}

# 从大小张量计算嵌套张量的批次偏移量
Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements) {
  # 获取大小张量的数据指针
  int64_t* const sizes_ptr = sizes.data_ptr<int64_t>();
  # 创建一个空张量用于存储偏移量，长度为 sizes.size(0) + 1 + extra_elements
  Tensor offsets = at::empty({1 + sizes.size(0) + extra_elements}, at::kInt);
  # 获取偏移量张量的可变数据指针
  int32_t* const offsets_ptr = offsets.mutable_data_ptr<int32_t>();
  # 初始化第一个偏移量为 0
  offsets_ptr[0] = 0;
  # 获取 sizes 张量的第二个维度大小
  const auto sizes_size_1 = sizes.size(1);
  # 获取 sizes 张量的第一个维度大小
  const auto sizes_size_0 = sizes.size(0);
  # 遍历大小张量的第一个维度
  for (const auto i : c10::irange(sizes_size_0)) {
    # 初始化一个累积乘积
    int64_t prod = 1;
    # 遍历大小张量的第二个维度
    for (const auto j : c10::irange(sizes_size_1)) {
      # 计算当前索引位置处的乘积
      prod *= sizes_ptr[i * sizes_size_1 + j];
    }
    # 计算并存储当前偏移量
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  # 返回最终计算得到的偏移量张量
  return offsets;
}
// 将嵌套张量转换为掩码张量，其中nt为输入嵌套张量，mask_dim为可选参数，表示掩码的维度，mask_dim_length为可选参数，表示掩码维度的长度
Tensor NestedTensor_to_mask(const Tensor& nt, std::optional<int64_t> mask_dim, std::optional<int64_t> mask_dim_length) {
  // 获取嵌套张量的实现对象
  auto* nt_impl = get_nested_tensor_impl(nt);
  // 检查嵌套张量是否是连续的
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_impl), "to_mask only works on contiguous NestedTensors.");
  // 检查mask_dim参数是否有效，即是否小于嵌套张量的维度
  TORCH_CHECK(
      !mask_dim || *mask_dim < nt.dim(),
      "Requested mask dimension ",
      *mask_dim,
      " is bigger than dimension ",
      nt.dim(),
      " of given NestedTensor.");

  // TODO: port optimization for 1x1 tensors from
  // pytorch/nestedtensor's version.

  // 仅支持特殊情况：mask_dim == 2，且嵌套张量为3维
  TORCH_CHECK(
      mask_dim && *mask_dim == 2 && nt.dim() == 3,
      "Only the special case of mask_dim == 2 on a 3-D NestedTensor is supported right now.")
  // 获取嵌套张量实现对象的嵌套尺寸信息
  const auto& sizes = nt_impl->get_nested_sizes();
  // 创建用于存储结果的张量，形状为[嵌套张量中张量的数量, result_size_1]
  // result_size_1由mask_dim_length参数指定，或者计算得到嵌套张量中最大尺寸的第一个维度
  const auto result_size_1 = mask_dim_length ? *mask_dim_length : NestedTensor_get_max_size(*nt_impl)[0];
  auto result = at::ones({sizes.sizes()[0], result_size_1}, at::kBool);
  // 断言检查嵌套尺寸的维度是否为2
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  // 获取结果张量的数据指针和嵌套尺寸数据指针
  auto* result_data = result.data_ptr<bool>();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_1 = sizes.sizes()[1];
  // 遍历嵌套尺寸，为每个张量的每个元素设置为false
  for (const auto ii : c10::irange(sizes.sizes()[0])) {
    auto length = sizes_ptr[ii * sizes_size_1];
    for (const auto jj : c10::irange(length)) {
      result_data[ii * result_size_1 + jj] = false;
    }
  }
  // 返回结果张量
  return result;
}

// 将不规则的张量转换为填充后的密集张量（CPU版本）
Tensor _jagged_to_padded_dense_forward_cpu(
    const Tensor& values,
    TensorList offsets_list,
    c10::IntArrayRef max_lengths,
    const double padding_value) {
  // TODO: Make this kernel more efficient using TensorIterator or something.
  // 断言检查是否只支持单一的不规则维度和单一的最大长度
  TORCH_INTERNAL_ASSERT(
      offsets_list.size() == 1 && max_lengths.size() == 1,
      "_jagged_to_padded_dense_forward(): only a single jagged dim is supported for now");

  // 分配适当大小的填充张量
  auto offsets = offsets_list[0];
  TORCH_CHECK(
      offsets.dim() == 1,
      "_jagged_to_padded_dense_forward(): expected 1D offsets, but got offsets.dim() == ",
      offsets.dim());

  auto batch_size = offsets.size(0) - 1;
  auto max_length = max_lengths[0];
  auto values_shape = values.sizes().vec();
  std::vector<int64_t> padded_shape;
  padded_shape.reserve(values.dim() + 1);
  padded_shape.push_back(batch_size);
  padded_shape.push_back(max_length);
  padded_shape.insert(padded_shape.end(), values_shape.begin() + 1, values_shape.end());
  // 创建填充值为padding_value的新张量
  Tensor padded = values.new_full(padded_shape, padding_value);

  // 将数据复制到填充张量中
  for (auto i : c10::irange(batch_size)) {
    auto start_offset = offsets[i].item<int64_t>();
    auto end_offset = offsets[i + 1].item<int64_t>();
    auto length = end_offset - start_offset;
    // 注意：截断到最大长度以匹配CUDA核函数的行为
    length = std::min(length, max_length);
    // 从 'values' 张量中切片出指定范围的数据作为源数据
    auto source = values.slice(0, start_offset, start_offset + length);
    // 从 'padded' 张量中选择第 i 列，并切片出指定范围的数据作为目标数据
    auto dst = padded.select(0, i).slice(0, 0, length);
    // 将源数据复制到目标数据中
    dst.copy_(source);
  }

  // 返回填充后的 'padded' 张量作为结果
  return padded;
} // 结束函数 _padded_dense_to_jagged_forward_cpu

} // 结束命名空间 native
} // 结束命名空间 at
```