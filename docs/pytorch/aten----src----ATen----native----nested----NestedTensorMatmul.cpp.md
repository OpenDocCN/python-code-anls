# `.\pytorch\aten\src\ATen\native\nested\NestedTensorMatmul.cpp`

```
// 包含 ATen 库的各种头文件，用于张量操作和函数定义
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>

namespace at {
namespace native {

// 函数定义：对嵌套张量进行批次矩阵乘法（Batch Matrix Multiplication）
Tensor bmm_nested(const Tensor& self, const Tensor& mat2) {
  // 检查输入张量维度是否为3，如果不是则抛出错误信息
  TORCH_CHECK(self.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2.dim() == 3, "batch2 must be a 3D tensor");

  // 获取嵌套张量中第一个维度的大小，用于检查两个输入张量的批次大小是否相同
  int64_t ntensors = self.is_nested() ? get_nested_tensor_impl(self)->size(0) : self.size(0);
  int64_t ntensors2 = mat2.is_nested() ? get_nested_tensor_impl(mat2)->size(0) : mat2.size(0);

  // 检查两个输入张量的批次大小是否相同，如果不同则抛出错误信息
  TORCH_CHECK(ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ", ntensors,
      " but got: ", ntensors2, ".");

  // 根据输入张量是否为嵌套张量选择性地获取其底层存储张量
  const Tensor& self_buffer = self.is_nested() ? get_nested_tensor_impl(self)->get_unsafe_storage_as_tensor() : self;
  const Tensor& mat2_buffer = mat2.is_nested() ? get_nested_tensor_impl(mat2)->get_unsafe_storage_as_tensor() : mat2;

  // 创建一个空的输出张量，确保其是连续的
  int64_t out_numel = 0;
  const Tensor& self_sizemat = self.is_nested() ?
      get_nested_tensor_impl(self)->get_nested_sizes() : get_nested_tensor_impl(mat2)->get_nested_sizes();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  
  // 遍历每个批次中的嵌套张量，计算输出张量的形状和元素个数
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_shape = get_size_for_index(self, i);
    const IntArrayRef& mat2_shape = get_size_for_index(mat2, i);
    const int64_t& self_size0 = self_shape[0], & self_size1 = self_shape[1],
        & mat2_size0 = mat2_shape[0], & mat2_size1 = mat2_shape[1];
    
    // 检查当前批次中两个嵌套张量是否可以进行矩阵乘法，如果不能则抛出错误信息
    TORCH_CHECK(self_size1 == mat2_size0,
        i, "-th nested matrices in batch cannot be multiplied (",
        self_size0, "x", self_size1, " and ",
        mat2_size0, "x", mat2_size1, ")");
    
    // 设置输出张量的形状信息
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    
    // 累加输出张量的元素个数
    out_numel += self_size0 * mat2_size1;
  }

  // 根据输入张量是否为嵌套张量选择性地创建输出缓冲区
  Tensor out_buffer = self.is_nested() ? self_buffer.new_empty(out_numel) : mat2_buffer.new_empty(out_numel);
  
  // 将输出缓冲区包装为嵌套张量，并返回结果
  Tensor output = wrap_buffer(out_buffer, out_sizemat);

  // 调用张量的矩阵乘法函数 mm
  // TODO: `padding nested tensor -> bmm -> remove padding` 可能更有效
  //       直到有专门的嵌套张量 bmm 内核为止
  //       有用资源: `aten/src/ATen/native/cpu/LinearAlgebra.cpp/bmm_out_or_baddbmm_`
  //                `aten/src/ATen/native/cuda/Blas.cpp/baddbmm_out_cuda_impl`
  std::vector<Tensor> output_unbind = output.unbind();
  
  // 遍历每个批次中的嵌套张量，进行批次矩阵乘法操作
  for (int64_t i = 0; i < ntensors; i++) {
    // 这里应该是继续批次矩阵乘法的逻辑，但代码被截断了，需要继续完善
    # 使用 ATen（PyTorch Tensor Library）中的矩阵乘法函数 mm_out，计算输出张量的第 i 个分块
    at::mm_out(output_unbind[i],
              # 使用 self_buffer 的视图，根据索引 i 计算其尺寸大小、步长和偏移量，以构造相应的张量视图
              self_buffer.as_strided(get_size_for_index(self, i), get_stride_for_index(self, i), get_offset_for_index(self, i)),
              # 使用 mat2_buffer 的视图，根据索引 i 计算其尺寸大小、步长和偏移量，以构造相应的张量视图
              mat2_buffer.as_strided(get_size_for_index(mat2, i), get_stride_for_index(mat2, i), get_offset_for_index(mat2, i)));
  }
  # 返回计算后的输出张量
  return output;
// 定义静态函数 matmul_with_bmm_nested，执行嵌套张量的批矩阵乘法
static Tensor matmul_with_bmm_nested(const Tensor& self, const Tensor& mat2) {
  // Tensor self = self_.contiguous();
  // Tensor mat2 = mat2_.contiguous();
  // 对输入张量进行连续性操作，确保数据在内存中连续存储
  // self [N, n_heads, *, head_dim]
  // mat2 [N, n_heads, head_dim, *]

  // 获取 self 和 mat2 的 NestedTensor 实现指针
  const auto self_ptr = get_nested_tensor_impl(self);
  const auto mat2_ptr = get_nested_tensor_impl(mat2);

  // 获取 self 的尺寸和步长信息
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  // 获取 self 的存储偏移数组指针
  int64_t* self_offsets_ptr =
      self_ptr->get_storage_offsets().data_ptr<int64_t>();
  auto opt = self_ptr->get_nested_sizes().options();

  // 获取 mat2 的尺寸和步长信息
  std::vector<IntArrayRef> mat2_sizes = NestedTensor_get_sizes(mat2_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  // 获取 mat2 的存储偏移数组指针
  int64_t* mat2_offsets_ptr =
      mat2_ptr->get_storage_offsets().data_ptr<int64_t>();
  auto opt2 = mat2_ptr->get_nested_sizes().options();

  // 计算 self 的总体尺寸
  int64_t N = self_sizes.size();
  int64_t n_heads = self_sizes[0][0];

  // 创建用于 self 视图的新尺寸和步长元数据
  auto self_new_sizes = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_sizes_ptr = self_new_sizes.mutable_data_ptr<int64_t>();

  auto self_new_strides = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_strides_ptr = self_new_strides.mutable_data_ptr<int64_t>();
  auto self_new_offsets = at::empty({N * n_heads}, opt);
  int64_t* self_new_offsets_ptr = self_new_offsets.mutable_data_ptr<int64_t>();

  // 创建用于 mat2 视图的新尺寸和步长元数据
  auto mat2_new_sizes = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_sizes_ptr = mat2_new_sizes.mutable_data_ptr<int64_t>();

  auto mat2_new_strides = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_strides_ptr = mat2_new_strides.mutable_data_ptr<int64_t>();
  auto mat2_new_offsets = at::empty({N * n_heads}, opt);
  int64_t* mat2_new_offsets_ptr = mat2_new_offsets.mutable_data_ptr<int64_t>();

  // 遍历每个 N 对应的维度，获取相关的尺寸、步长和偏移信息
  for (int64_t i = 0; i < N; i++) {
    const IntArrayRef& self_size_i = self_sizes[i];
    const IntArrayRef& self_stride_i = self_strides[i];
    int64_t self_offset = self_offsets_ptr[i];

    const IntArrayRef& mat2_size_i = mat2_sizes[i];
    const IntArrayRef& mat2_stride_i = mat2_strides[i];
    int64_t mat2_offset = mat2_offsets_ptr[i];
    // 循环处理每个头部
    for (int64_t j = 0; j < n_heads; j++) {
      // 计算索引，用于访问self和mat2的大小和步长
      auto idx = (i * n_heads + j) * 2;
      // 设置self的新大小
      self_new_sizes_ptr[idx] = self_size_i[1];
      self_new_sizes_ptr[idx + 1] = self_size_i[2];
      // 设置self的新步长
      self_new_strides_ptr[idx] = self_stride_i[1];
      self_new_strides_ptr[idx + 1] = self_stride_i[2];
      // 设置self的新偏移量
      auto offset_idx = i * n_heads + j;
      self_new_offsets_ptr[offset_idx] = self_offset;
      self_offset += self_stride_i[0];

      // 设置mat2的新大小
      mat2_new_sizes_ptr[idx] = mat2_size_i[1];
      mat2_new_sizes_ptr[idx + 1] = mat2_size_i[2];
      // 设置mat2的新步长
      mat2_new_strides_ptr[idx] = mat2_stride_i[1];
      mat2_new_strides_ptr[idx + 1] = mat2_stride_i[2];
      // 设置mat2的新偏移量
      mat2_new_offsets_ptr[offset_idx] = mat2_offset;
      mat2_offset += mat2_stride_i[0];
    }
  }

  // 将self视图为[N * n_heads, *, head_dim] (折叠前两个维度)
  auto viewed_self = create_nested_view_tensor(
      self, self_new_sizes, self_new_strides, self_new_offsets);

  // 将mat2视图为[N * n_heads, head_dim, *] (折叠前两个维度)
  auto viewed_mat2 = create_nested_view_tensor(
      mat2, mat2_new_sizes, mat2_new_strides, mat2_new_offsets);

  // 进行矩阵乘法，输出 [N * n_heads, *, *]
  auto bmm_output = at::bmm(viewed_self, viewed_mat2);

  // 生成元数据，将输出视图为[N, n_heads, *, *]
  // 保证bmm的输出是连续的，因此步长计算应该正确
  auto out_new_sizes = at::empty({N, 3}, opt);
  auto out_new_strides = at::empty({N, 3}, opt);
  auto out_new_offsets = at::empty({N}, opt);
  int64_t* out_new_offsets_ptr = out_new_offsets.mutable_data_ptr<int64_t>();

  int64_t* out_new_sizes_ptr = out_new_sizes.data_ptr<int64_t>();
  int64_t* out_new_strides_ptr = out_new_strides.data_ptr<int64_t>();

  int64_t out_offset = 0;
  // 遍历每个N，设置输出的大小、步长和偏移量
  for (int64_t i = 0; i < N; i++) {
    out_new_offsets_ptr[i] = out_offset;
    const IntArrayRef& self_size_i = self_sizes[i];
    const IntArrayRef& mat2_size_i = mat2_sizes[i];
    auto idx = i * 3;
    out_new_sizes_ptr[idx] = n_heads;
    out_new_sizes_ptr[idx + 1] = self_size_i[1];
    out_new_sizes_ptr[idx + 2] = mat2_size_i[2];
    out_new_strides_ptr[idx] = self_size_i[1] * mat2_size_i[2];
    out_new_strides_ptr[idx + 1] = mat2_size_i[2];
    out_new_strides_ptr[idx + 2] = 1;
    out_offset += n_heads * (self_size_i[1] * mat2_size_i[2]);
  }

  // 创建输出的嵌套视图张量
  auto viewed_out = create_nested_view_tensor(
      bmm_output, out_new_sizes, out_new_strides, out_new_offsets);

  // 返回视图输出
  return viewed_out;
}

// nt: NT的形状为(B, *, C, D)
// other: 形状为(D, E)的密集张量
// output: NT的形状为(B, *, C, E)
static Tensor matmul_nested_with_broadcasted_dense(
    const Tensor& nt,
    const Tensor& other) {
  // 将nt的缓冲区视为3D不规则张量进行矩阵乘法
  auto* nt_impl = get_nested_tensor_impl(nt);
  auto jagged = nt_impl->get_buffer().view({-1, nt.size(2), nt.size(3)});
  // 执行矩阵乘法操作
  auto new_buffer = at::matmul(jagged, other);

  // 将结果包装为嵌套张量
  const auto E = other.size(-1);
  const auto component_dim = nt.dim() - 1;
  auto new_sizes = nt_impl->get_nested_sizes().clone();
  auto new_sizes_ptr = new_sizes.data_ptr<int64_t>();
  for (const auto i : c10::irange(nt.size(0))) {
    // 更新嵌套张量的尺寸，将最后一维的大小设置为E
    new_sizes_ptr[i * component_dim + 2] = E;
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      new_buffer.view(-1), new_sizes);
}

// Note [nested tensor matmul]
// 这实际上是针对嵌套张量的广义批次矩阵乘法，
// 其中`self`和`mat2`具有相同数量（>= 3）的维度。
// 最后两个维度将被视为矩阵维度，
// 因此它们应该是可以进行矩阵乘法的。
// 前导维度被视为批次维度，
// 由于嵌套张量目前不支持广播，
// 对于每个批次维度，`self`和`mat2`必须具有相同的大小。
// TODO: 应该在某一天完全支持矩阵乘法的语义
Tensor matmul_nested(const Tensor& self, const Tensor& mat2) {
  // NT（B, *, C, D）与广播的密集张量（D, E）的特殊情况
  if (self.is_nested() && self.is_contiguous() && !mat2.is_nested() &&
      self.dim() == 4 && mat2.dim() == 2 &&
      get_nested_tensor_impl(self)->opt_size(2).has_value() &&
      get_nested_tensor_impl(self)->opt_size(3).has_value() &&
      self.size(3) == mat2.size(0)) {
    // 调用处理嵌套张量与广播密集张量乘法的函数
    return matmul_nested_with_broadcasted_dense(self, mat2);
  }
  if (self.is_nested() && !mat2.is_nested()) {
    // 报错：预期两者都是嵌套的，但得到了一个嵌套的self和非嵌套的other
    AT_ERROR("Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    // 在self不是嵌套的而mat2是嵌套的情况下报错
    AT_ERROR("Expected both to be nested, but got a non-nested self and nested other");
  }
    // 抛出错误，指示期望两个张量是嵌套的，但得到的是非嵌套的 self 和嵌套的 other
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // to_padded_tensor 只支持连续输入
  auto self_contig = self.contiguous();
  auto mat2_contig = mat2.contiguous();
  // 调度器应该保证至少一个张量是嵌套的
  const auto self_ptr = get_nested_tensor_impl(self_contig);
  const auto mat2_ptr = get_nested_tensor_impl(mat2_contig);
  int64_t self_dim = self_ptr->dim(), mat2_dim = mat2_ptr->dim();
  // 检查第一个输入张量是否具有至少 3 维，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      self_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: ",
      self_dim);
  // 检查第二个输入张量是否具有至少 3 维，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      mat2_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: ",
      mat2_dim);
  // 检查两个输入张量的维度是否相等，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      self_dim == mat2_dim, "matmul: both inputs must have the same rank");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  // 检查第一个维度的大小是否相等，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      ntensors == ntensors2,
      "matmul: Expected size for the 1st dimension of 2nd input tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");
  // 确保批次维度具有相同的大小（不进行广播）
  const auto& self_sizes = self_ptr->get_nested_sizes();
  const auto& mat2_sizes = mat2_ptr->get_nested_sizes();
  const auto& self_batch_sizes = self_sizes.narrow(1, 0, self_dim - 3);
  const auto& mat2_batch_sizes = mat2_sizes.narrow(1, 0, mat2_dim - 3);
  // 检查批次维度是否具有相同的大小，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      at::equal(self_batch_sizes, mat2_batch_sizes),
      "matmul: For nested tensors, batch dimensions must have the same sizes, ",
      "no broadcasting is currently performed. Got batch shapes for self ",
      self_batch_sizes,
      " and batch shapes for mat2 ",
      mat2_batch_sizes);
  // 确保 self 的最后一个维度和 mat2 的倒数第二个维度具有相同的大小
  const auto& self_dim_size = self_sizes.select(1, -1);
  const auto& mat2_dim_size = mat2_sizes.select(1, -2);
  // 检查最后一个维度是否具有相同的大小，用于嵌套张量的矩阵乘法
  TORCH_CHECK(
      at::equal(self_dim_size, mat2_dim_size),
      "matmul: Nested tensors cannot be matrix multiplied, last dimension of self has sizes",
      self_dim_size,
      "second last dimension of mat2 has sizes",
      mat2_dim_size);

  // 使用 bmm 推理专用快速路径，用于形状为 [N, n_heads, *, head_dim] 和 [N, n_heads, head_dim, *] 的张量
  if (self.is_cuda() && self_dim == 4 && self.is_contiguous() &&
      mat2_dim == 4 && mat2.is_contiguous() &&
      !(GradMode::is_enabled() &&
        (self.requires_grad() || mat2.requires_grad()))) {
    const auto& self_opt_head_dim = self_ptr->opt_size(1);
    const auto& mat2_opt_head_dim = mat2_ptr->opt_size(1);
    // 如果头维度大小相等，则使用 matmul_with_bmm_nested 函数执行 bmm 矩阵乘法
    if (self_opt_head_dim.has_value() && mat2_opt_head_dim.has_value() &&
        self_opt_head_dim.value() == mat2_opt_head_dim.value()) {
      return matmul_with_bmm_nested(self, mat2);
  }
}

// 从输入尺寸构造输出尺寸
Tensor output_sizes = self_sizes.clone();
// output_sizes 的每一行的最后一个条目应与 mat2_sizes 的最后一列相对应
output_sizes.index_put_(
    {at::indexing::Slice(), -1}, mat2_sizes.select(1, -1).clone());

auto self_padded = self_contig.to_padded_tensor(0.);
auto mat2_padded = mat2_contig.to_padded_tensor(0.);
auto output_padded = at::matmul(self_padded, mat2_padded);
// 通过通用方法从填充后的张量中生成嵌套的输出
auto output_nested = nested_from_padded_generic(output_padded, output_sizes);
return output_nested;



// 从输入尺寸构造输出尺寸
Tensor output_sizes = self_sizes.clone();
// output_sizes 的每一行的最后一个条目应与 mat2_sizes 的最后一列相对应
output_sizes.index_put_(
    {at::indexing::Slice(), -1}, mat2_sizes.select(1, -1).clone());

auto self_padded = self_contig.to_padded_tensor(0.);
auto mat2_padded = mat2_contig.to_padded_tensor(0.);
auto output_padded = at::matmul(self_padded, mat2_padded);
// 通过通用方法从填充后的张量中生成嵌套的输出
auto output_nested = nested_from_padded_generic(output_padded, output_sizes);
return output_nested;
}

Tensor& matmul_out_nested(
    const Tensor& tensor1,
    const Tensor& tensor2,
    Tensor& result) {
  // TODO: 这是一个非常快速且粗糙的实现
  //       应该改进以避免中间内存的使用

  // 执行张量相乘操作，得到结果张量 function_result
  Tensor function_result = at::matmul(tensor1, tensor2);

  // 获取 function_result 的嵌套张量实现指针
  auto function_result_ptr = get_nested_tensor_impl(function_result);

  // TODO: 这段代码是为了复制 function_result_ptr->opt_sizes_
  //       如果将来提供了访问器，可以替换这部分逻辑
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < function_result_ptr->dim(); i++) {
    // 获取第 i 维的可选尺寸
    std::optional<int64_t> opt_size = function_result_ptr->opt_size(i);
    if (opt_size.has_value()) {
      sizes.push_back(*opt_size);
    } else {
      sizes.push_back(-1);
    }
  }

  // 重新调整结果张量 result 的形状
  result.reshape(sizes);

  // 将 function_result 的数据复制到 result 中
  result.copy_(function_result);

  // 返回结果张量 result 的引用
  return result;
}

} // namespace native
} // namespace at
```