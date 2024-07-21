# `.\pytorch\aten\src\ATen\native\TensorShape.cpp`

```
// 定义宏，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的 Tensor 类和相关功能的头文件
#include <ATen/core/Tensor.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/functional.h>
#include <ATen/core/IListRef.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>

// 包含 ATen 库的 CPU 相关操作头文件
#include <ATen/native/Copy.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>

// 包含 ATen 库的量化张量实现头文件
#include <ATen/quantized/QTensorImpl.h>

// 包含 C10 实用工具的异常处理和可选值处理头文件
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

// 根据预处理器宏选择性地包含 ATen 库的完整功能或特定操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_chunk_cat_native.h>
#include <ATen/ops/_conj_copy_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/ops/_fw_primal_copy_native.h>
#include <ATen/ops/_indices_copy_native.h>
#include <ATen/ops/_make_dual.h>
#include <ATen/ops/_make_dual_copy_native.h>
#include <ATen/ops/_mkldnn_reshape.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/_neg_view_copy_native.h>
#include <ATen/ops/_reshape_alias_copy_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_reshape_copy_native.h>
#include <ATen/ops/_reshape_from_tensor_native.h>
#include <ATen/ops/_shape_as_tensor_native.h>
#include <ATen/ops/_sparse_broadcast_to.h>
#include <ATen/ops/_sparse_broadcast_to_copy_native.h>
#include <ATen/ops/_sparse_broadcast_to_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_stack_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/_values_copy_native.h>
#include <ATen/ops/adjoint_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/as_strided_copy_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/as_strided_scatter_native.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/atleast_2d.h>
#endif


这段代码是一组 C++ 头文件的包含语句，用于引入 ATen（PyTorch C++ 前端）和 C10 库中的各种功能和操作。
// 导入 ATen 库中的模块，这些模块提供了各种张量操作函数
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/block_diag_native.h>
#include <ATen/ops/broadcast_tensors_native.h>
#include <ATen/ops/broadcast_to_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/col_indices_copy_native.h>
#include <ATen/ops/column_stack_native.h>
#include <ATen/ops/concat_native.h>
#include <ATen/ops/concatenate_native.h>
#include <ATen/ops/crow_indices_copy_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/detach_copy_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diag_embed_native.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/diagflat_native.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_backward.h>
#include <ATen/ops/diagonal_backward_native.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/diagonal_copy_native.h>
#include <ATen/ops/diagonal_native.h>
#include <ATen/ops/diagonal_scatter_native.h>
#include <ATen/ops/dsplit_native.h>
#include <ATen/ops/dstack_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/expand_as_native.h>
#include <ATen/ops/expand_copy_native.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/flatten_dense_tensors_native.h>
#include <ATen/ops/flatten_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/hsplit_native.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/hstack_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/indices_copy_native.h>
#include <ATen/ops/lift_fresh_native.h>
#include <ATen/ops/lift_native.h>
#include <ATen/ops/mH_native.h>
#include <ATen/ops/mT_native.h>
#include <ATen/ops/matrix_H_native.h>
#include <ATen/ops/meshgrid_native.h>
#include <ATen/ops/moveaxis_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/movedim_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/narrow_copy_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/numpy_T_native.h>
#include <ATen/ops/permute_copy_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/ravel_native.h>
#include <ATen/ops/repeat_native.h>
#include <ATen/ops/reshape_as_native.h>
#include <ATen/ops/reshape_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_stack_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/select_backward_native.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_scatter_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/slice_backward_native.h>
#include <ATen/ops/slice_copy_native.h>
#include <ATen/ops/slice_inverse_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/slice_scatter_native.h>
// 包含稀疏 COO 张量操作的头文件
#include <ATen/ops/sparse_coo_tensor.h>
// 包含稀疏 COO 张量原生操作的头文件
#include <ATen/ops/sparse_coo_tensor_native.h>
// 包含稀疏维度原生操作的头文件
#include <ATen/ops/sparse_dim_native.h>
// 包含拆分复制原生操作的头文件
#include <ATen/ops/split_copy_native.h>
// 包含拆分原生操作的头文件
#include <ATen/ops/split_native.h>
// 包含按尺寸拆分操作的头文件
#include <ATen/ops/split_with_sizes.h>
// 包含按尺寸拆分复制原生操作的头文件
#include <ATen/ops/split_with_sizes_copy_native.h>
// 包含按尺寸拆分原生操作的头文件
#include <ATen/ops/split_with_sizes_native.h>
// 包含压缩复制原生操作的头文件
#include <ATen/ops/squeeze_copy_native.h>
// 包含压缩原生操作的头文件
#include <ATen/ops/squeeze_native.h>
// 包含压缩操作的头文件
#include <ATen/ops/squeeze.h>
// 包含堆栈原生操作的头文件
#include <ATen/ops/stack_native.h>
// 包含子操作的头文件
#include <ATen/ops/sub.h>
// 包含求和操作的头文件
#include <ATen/ops/sum.h>
// 包含调整大小原生操作的头文件
#include <ATen/ops/sum_to_size_native.h>
// 包含交换轴原生操作的头文件
#include <ATen/ops/swapaxes_native.h>
// 包含交换维度原生操作的头文件
#include <ATen/ops/swapdims_native.h>
// 包含转置复制原生操作的头文件
#include <ATen/ops/t_copy_native.h>
// 包含转置原生操作的头文件
#include <ATen/ops/t_native.h>
// 包含张量操作的头文件
#include <ATen/ops/tensor.h>
// 包含张量拆分操作的头文件
#include <ATen/ops/tensor_split.h>
// 包含张量拆分原生操作的头文件
#include <ATen/ops/tensor_split_native.h>
// 包含平铺原生操作的头文件
#include <ATen/ops/tile_native.h>
// 包含转置操作的头文件
#include <ATen/ops/transpose.h>
// 包含转置复制原生操作的头文件
#include <ATen/ops/transpose_copy_native.h>
// 包含转置原生操作的头文件
#include <ATen/ops/transpose_native.h>
// 包含解绑操作的头文件
#include <ATen/ops/unbind.h>
// 包含解绑复制原生操作的头文件
#include <ATen/ops/unbind_copy_native.h>
// 包含解绑原生操作的头文件
#include <ATen/ops/unbind_native.h>
// 包含解压稠密张量原生操作的头文件
#include <ATen/ops/unflatten_dense_tensors_native.h>
// 包含解压原生操作的头文件
#include <ATen/ops/unflatten_native.h>
// 包含展开复制原生操作的头文件
#include <ATen/ops/unfold_copy_native.h>
// 包含展开原生操作的头文件
#include <ATen/ops/unfold_native.h>
// 包含不安全块原生操作的头文件
#include <ATen/ops/unsafe_chunk_native.h>
// 包含不安全拆分原生操作的头文件
#include <ATen/ops/unsafe_split_native.h>
// 包含不安全按尺寸拆分原生操作的头文件
#include <ATen/ops/unsafe_split_with_sizes_native.h>
// 包含增加维度复制原生操作的头文件
#include <ATen/ops/unsqueeze_copy_native.h>
// 包含增加维度原生操作的头文件
#include <ATen/ops/unsqueeze_native.h>
// 包含值复制原生操作的头文件
#include <ATen/ops/values_copy_native.h>
// 包含视图作为复杂类型的头文件
#include <ATen/ops/view_as_complex.h>
// 包含视图作为复杂类型复制原生操作的头文件
#include <ATen/ops/view_as_complex_copy_native.h>
// 包含视图作为原生操作的头文件
#include <ATen/ops/view_as_native.h>
// 包含视图作为实数的头文件
#include <ATen/ops/view_as_real.h>
// 包含视图作为实数复制原生操作的头文件
#include <ATen/ops/view_as_real_copy_native.h>
// 包含视图复制原生操作的头文件
#include <ATen/ops/view_copy_native.h>
// 包含视图原生操作的头文件
#include <ATen/ops/view_native.h>
// 包含垂直分割原生操作的头文件
#include <ATen/ops/vsplit_native.h>
// 包含垂直堆叠操作的头文件
#include <ATen/ops/vstack.h>
// 包含垂直堆叠原生操作的头文件
#include <ATen/ops/vstack_native.h>
// 包含零操作的头文件
#include <ATen/ops/zeros.h>
// 包含类似零操作的头文件
#include <ATen/ops/zeros_like.h>
// 包含零原生操作的头文件
#include <ATen/ops/zeros_native.h>
// 如果未定义结束符号，则结束符号
#endif

// 包含标准库中的算法、整型、实用工具和向量
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// 在 at 命名空间下定义元函数
namespace at::meta {
// 检查连接时没有零维的张量
inline void cat_check_no_zero_dim(const MaterializedITensorListRef& tensors) {
  // 初始化索引计数器
  size_t i = 0;
  // 遍历张量列表中的每一个张量
  for (const Tensor& t : tensors) {
    // 使用 TORCH_CHECK 检查张量维度是否大于零
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
    // 更新索引
    i++;
  }
}

// 计算连接操作的输出内存格式
inline c10::MemoryFormat cat_compute_output_memory_format(const MaterializedITensorListRef& inputs) {
  // 初始化内存格式为无
  std::optional<c10::MemoryFormat> format = c10::nullopt;
  // 遍历输入张量列表中的每一个张量
  for (const Tensor& t : inputs) {
    // 建议张量的内存格式
    auto f = t.suggest_memory_format();
    // 如果建议的内存格式是连续的，直接返回该格式
    if (f == c10::MemoryFormat::Contiguous) {
        return f;
    }
    // 如果已经有存储的内存格式，并且当前张量的格式与之不同，返回连续的内存格式
    if (format.has_value() && format.value() != f) {
        return c10::MemoryFormat::Contiguous;
    }
    // 更新存储的内存格式
    format = f;
  }
  // 返回最后一个张量的内存格式
  return format.value();
}
// 定义一个名为 TORCH_PRECOMPUTE_META_FUNC 的宏函数，接受两个参数：tensors（一个 ITensorListRef 类型的引用）和 dim（一个 int64_t 类型的整数）
TORCH_PRECOMPUTE_META_FUNC(cat)(const ITensorListRef& tensors, int64_t dim) {
  // 以前，大小为 [0] 的张量是唯一可能为空的张量；因此，除非所有其他张量都是一维的，否则不能连接空张量，因此我们允许这些张量被“跳过”。
  // 我们保持这种行为以保证向后兼容性，但仅适用于这个特定的大小（即其他空大小不被跳过）。
  auto materialized = tensors.materialize();

  // 调用函数检查所有张量都不是零维
  cat_check_no_zero_dim(materialized);

  // 使用 at::legacy_cat_wrap_dim 函数将维度 dim 调整到有效的范围内
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // 在实际计算维度之前，检查名称
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);

  // 检查材料化后的张量列表是否为空
  TORCH_CHECK(
      !materialized.empty(), "torch.cat(): expected a non-empty list of Tensors");

  // 查找第一个有效的张量
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  // 初始化标志，用于判断所有张量是否具有相同的内存布局
  bool all_contiguous = true;
  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;

  // 计算输出内存布局格式
  auto memory_format = cat_compute_output_memory_format(materialized);

  // 计算输出张量的数据类型
  const auto& result = maybe_get_output();
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // 如果输出张量已定义，则需要在计算实际输出数据类型和标志时考虑它
  if (is_out_defined) {
    // 如果输出张量已定义，则检查类型推广
    TORCH_CHECK(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    // 更新输出数据类型为结果张量的数据类型
    out_dtype = result.scalar_type();
    // 检查结果张量是否按指定的内存格式连续
    all_contiguous = result.is_contiguous(memory_format);
  }

  // 设置输出张量的维度向量
  DimVector sizes {0};

  // 初始化张量选项，使用第一个材料化张量的选项作为基础设置
  TensorOptions options = materialized[0].get().options()
      .dtype(out_dtype)
      .memory_format(memory_format);

  // 检查是否找到有效的张量
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    // 检查维度 dim 是否在有效张量的维度范围内
    TORCH_CHECK(
        dim <= materialized[valid].get().dim(), "torch.cat(): dimension ", dim, "out of range");

    // 计算输出张量的大小
    // 它应该与任何其他有效张量具有相同的形状，除了维度 'dim'
    size_t size_at_dim = 0;
    // 遍历 materialized 向量中的每个索引 i
    for (const auto i : c10::irange(materialized.size())) {
      // 获取索引 i 处的张量 t
      const Tensor& t = materialized[i];
      // 更新 all_same_dtype 标志，检查所有张量的数据类型是否与输出数据类型一致
      all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
      // 如果张量 t 不应跳过拼接操作
      if (!at::native::cat_should_skip_tensor(t)) {
        // 检查除了指定维度 dim 外的形状是否匹配
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        // 累加在指定维度 dim 上的大小
        size_at_dim += t.size(dim);
        // 更新 all_contiguous 标志，检查所有张量是否按指定内存格式连续存储
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        // 更新 all_same_sizes_and_stride 标志，检查张量的大小和步长是否与参考张量相同
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        // 如果张量 t 应跳过拼接操作，则设置 all_contiguous 为 false
        all_contiguous = false;
      }
    }

    // 实际设置输出张量的大小、选项（数据类型和内存格式），并可能指定输出张量的名称
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options = materialized[valid].get().options()
        .dtype(out_dtype)
        .memory_format(memory_format);
  }

  // 调用 set_output_raw_strided 方法设置输出张量的大小、步长、选项和可能的输出张量名称
  set_output_raw_strided(0, sizes, {}, options, maybe_outnames);
  // 检查输入张量与输出张量之间是否存在重叠区域
  if (is_out_defined && found_valid_tensor) {
    // 断言输出张量内部不存在重叠区域
    at::assert_no_internal_overlap(result);
    // 检查每个 materialized 张量与输出张量之间是否存在重叠区域
    for (const Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }

  // 返回一个预先计算好的 torch::Tensor 结构对象，设置其维度、有效性、连续性、数据类型一致性和内存格式
  return TORCH_PRECOMPUTE_STRUCT(cat)()
      .set_dim(dim)
      .set_valid(valid)
      .set_all_contiguous(all_contiguous)
      .set_all_same_dtype(all_same_dtype)
      .set_all_same_sizes_and_stride(all_same_sizes_and_stride)
      .set_memory_format(memory_format);
// 结束 at::native 命名空间的定义

// 定义分发函数 cat_serial_stub 和 stack_serial_stub

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor) {
  // 检查 shape_tensor 的维度是否为 1
  TORCH_CHECK(shape_tensor.dim() == 1);
  // 创建空的 int64_t 向量 shape
  std::vector<int64_t> shape;
  // 使用 accessor 获取 shape_tensor 的数据
  auto accessor = shape_tensor.accessor<int64_t, 1>();
  // 遍历 shape_tensor 的每个元素，将其添加到 shape 向量中
  for (const auto i : c10::irange(shape_tensor.numel())) {
    shape.push_back(accessor[i]);
  }
  // 调用 self 的 reshape 方法，返回重新形状后的 Tensor
  return self.reshape(IntArrayRef(shape));
}

Tensor _shape_as_tensor(const Tensor& self) {
  // 创建类型为 kLong 的 TensorOptions
  auto options = TensorOptions(at::kLong);
  // 使用 self 的 sizes 创建一个新的 Tensor
  return at::tensor(self.sizes(), options);
}

Tensor& set_(Tensor& result, Storage source) {
  // 计算新的 size，以便从 source 的 nbytes 和 result 的 dtype.itemsize 推导出来
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  // 调用 result 的 set_ 方法，设置新的 Storage
  return result.set_(std::move(source), 0, new_size, {});
}


// 是否与 cuda 实现统一？为了避免在 resize_impl_cpu_ 中进行分发而未统一
Tensor& set_storage_cpu_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  // 检查设置 storage 是否符合条件
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);

  // 设置 result 的 storage_offset
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  // 如果 stride.data() 不为空，将其转换为 OptionalIntArrayRef；否则设为 nullptr
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  // 调用 resize_impl_cpu_ 进行大小调整，传递 size、stride_opt 和 resize_storage 参数
  // 如果 result 不是 meta tensor，就调整其 storage
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(), size, stride_opt, /*resize_storage=*/!result.is_meta());
  // 返回设置后的 result
  return result;
}

Tensor& set_storage_meta__symint(Tensor& result, Storage storage, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  // 检查设置 storage 是否符合条件
  checkSetStorage(result, storage, storage_offset, size, stride);

  // 创建 contiguous_strides，如果 stride.data() 为空，基于 size 创建连续步长
  c10::SymDimVector contiguous_strides;
  if (stride.data() == nullptr) {
    int64_t dim = size.size();
    contiguous_strides.resize(dim);
    if (dim > 0) {
      const auto last_idx = dim - 1;
      contiguous_strides.at(last_idx) = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        contiguous_strides.at(i) = contiguous_strides.at(i+1) * size.at(i+1);
      }
    }
    stride = contiguous_strides;
  }

  // 在设置 storage 前运行此操作，以便访问 numel
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, storage_offset);

  // 如果符合大小约束，就运行 maybe_resize_storage_cpu，处理没有 numel 的情况
  if (TORCH_GUARD_SIZE_OBLIVIOUS(result.sym_numel().sym_ne(0))) {
    TORCH_INTERNAL_ASSERT(storage);
    // 检查 storage 是否可调整大小
    TORCH_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");
    // 所有 meta 数据指针相同，因此不需要 "重新" 分配它们
    // TODO: 如果使用特殊
    // 定义变量 itemsize 来存储结果张量元素的大小（字节数）
    const auto itemsize = result.dtype().itemsize();
    // 计算新的存储字节数 new_size_bytes，具体取决于结果张量是否是连续的
    c10::SymInt new_size_bytes = result.is_contiguous()
      ? at::detail::computeStorageNbytesContiguous(size, itemsize, std::move(storage_offset))
      : at::detail::computeStorageNbytes(size, stride, itemsize, std::move(storage_offset));
    // TODO: 当存在未备份的 SymInts 时，我们无条件跳过设置器。
    // 这在技术上是不正确的，但在许多情况下我们无法方便地测试真实条件，
    // 因为很多人仅仅使用 set_ 来交换张量上的元数据，而不是真正想要调整存储空间大小。
    //
    // 旧的行为是无条件地设置字节大小，但我认为不设置更加安全。
    // 检查新的存储字节数是否有提示，并且当前存储的符号字节数也有提示，并且如果新的符号字节数大于当前存储的符号字节数
    if (new_size_bytes.has_hint() && storage.sym_nbytes().has_hint() && TORCH_GUARD_SIZE_OBLIVIOUS(new_size_bytes.sym_gt(storage.sym_nbytes()))) {
      // 设置当前存储的字节数为新计算的存储字节数
      storage.set_nbytes(std::move(new_size_bytes));
    }
  }
  // 返回处理后的结果张量
  return result;
}

// 设置符号整数Tensor，用于描述非连续存储结构的Tensor
Tensor& set__symint(Tensor& result, const Tensor& storage, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  // 检查传入的存储Tensor是否是连续的
  TORCH_CHECK(storage.is_contiguous(), "passed in tensor to be used as storage must be contiguous");
  // 调用result的set__symint方法，设置其为符号整数Tensor
  return result.set__symint(storage.storage(), storage_offset + storage.sym_storage_offset(), size, stride);
}

// 将一个Tensor设置为另一个Tensor的值
Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  // 检查result与source的Tensor实现是否相同
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    // 调用result的set__symint方法，使用source作为存储，并复制其符号信息
    return result.set__symint(source.storage(), source.sym_storage_offset(), source.sym_sizes(), source.sym_strides());
  }
  // 若result与source相同，直接返回result
  return result;
}

// 在CPU上设置Tensor
Tensor& set_cpu_(Tensor& result) {
  // 获取result的数据类型
  caffe2::TypeMeta dtype = result.dtype();
  // 创建一个空的Storage对象，使用CPU的分配器
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::GetAllocator(kCPU),
      true);
  // 将result设置为使用新的Storage对象
  result.set_(std::move(storage), 0, {0}, {});
  // 内部断言确保设置后的数据类型与之前相同
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  // 返回设置后的result
  return result;
}

// 在元数据（meta）上设置Tensor
Tensor& set_meta_(Tensor& result) {
  // 获取result的数据类型
  caffe2::TypeMeta dtype = result.dtype();
  // 创建一个空的Storage对象，使用元数据的分配器
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::GetAllocator(kMeta),
      true);
  // 将result设置为使用新的Storage对象
  result.set_(std::move(storage), 0, {0}, {});
  // 内部断言确保设置后的数据类型与之前相同
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  // 返回设置后的result
  return result;
}

// 将稀疏Tensor广播到指定大小
Tensor sparse_broadcast_to(const Tensor& self, IntArrayRef size) {
  // 检查输入Tensor是否为稀疏Tensor
  TORCH_CHECK(self.is_sparse(), "input must be sparse tensor");
  // 计算额外的稀疏维度和稠密维度数量
  int64_t sparse_extra_ndim = size.size() - self.dim();
  int64_t sparse_ndim = size.size() - self.dense_dim();
  // 检查输入Tensor是否可以广播到较小维度的大小
  TORCH_CHECK(sparse_extra_ndim >= 0, "input not broadcastable to size with smaller dimensionality");
  // 获取稀疏Tensor的索引和值
  Tensor indices = self._indices();
  Tensor values = self._values();
  auto nnz = values.size(0);

  // 初始化广播尺寸、稠密尺寸、广播维度和不变维度
  std::vector<int64_t> broadcast_sizes;
  std::vector<int64_t> broadcast_dense_sizes;
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> unchanged_dims;
  broadcast_sizes.reserve(sparse_ndim);
  broadcast_dense_sizes.reserve(self.dense_dim() + 1);
  broadcast_dims.reserve(self.sparse_dim());
  unchanged_dims.reserve(self.sparse_dim());
  int64_t nnz_factor = 1;
  int64_t min_broadcast_dim = (sparse_extra_ndim > 0 ? 0: -1);
  int64_t max_unchanged_dim = -1;

  // 循环处理额外的稀疏维度
  for (int64_t i=0; i<sparse_extra_ndim; i++) {
    auto d = size[i];
    nnz_factor *= d;
    broadcast_sizes.emplace_back(d);
  }

  // 循环处理稀疏Tensor的稀疏维度
  for (int64_t i=0; i<self.sparse_dim(); i++) {
    auto d = size[sparse_extra_ndim + i];
    // ...
    // 如果当前维度i的大小不等于d，则执行以下逻辑
    if (self.size(i) != d) {
      // 检查当前维度i的大小是否为1，若不是则抛出异常
      TORCH_CHECK(self.size(i) == 1,
                  "The expanded size of the tensor (",size[sparse_extra_ndim + i],") ",
                  "must match the existing size (",self.size(i),")");
      // 更新非零元素因子
      nnz_factor *= d;
      // 将扩展的大小添加到广播尺寸列表
      broadcast_sizes.emplace_back(d);
      // 如果最小广播维度尚未设置，则设置为当前维度的索引
      if (min_broadcast_dim == -1) {
        min_broadcast_dim = sparse_extra_ndim + i;
      }
      // 将当前维度索引添加到广播维度列表
      broadcast_dims.emplace_back(i);
    } else {
      // 如果当前维度i的大小等于d，则将其索引添加到未更改维度列表
      unchanged_dims.emplace_back(i);
      // 更新最大未更改维度
      max_unchanged_dim = sparse_extra_ndim + i;
    }
  }
  // 计算是否稀疏张量已经被压缩，如果张量维度为0或者当前稀疏张量已经被压缩且未更改的最大维度小于最小广播维度或者最小广播维度为-1，则为true
  bool is_coalesced = self.dim()==0 || (self.is_coalesced() && (max_unchanged_dim < min_broadcast_dim || min_broadcast_dim == -1));

  // 将nnz添加到广播稠密尺寸列表中
  broadcast_dense_sizes.emplace_back(nnz);
  // 将扩展的稠密维度尺寸添加到广播稠密尺寸列表中
  for (int64_t i=0; i<self.dense_dim(); i++) {
    broadcast_dense_sizes.emplace_back(size[sparse_extra_ndim + self.sparse_dim() + i]);
  }

  // 创建新的指数和值的尺寸向量
  std::vector<int64_t> new_indices_size{sparse_ndim, nnz * nnz_factor};
  std::vector<int64_t> new_values_size(values.sizes().vec());
  new_values_size[0] = new_indices_size[1];

  // 根据广播稠密尺寸扩展值，并重复插入以创建新的值张量
  Tensor new_values = values.expand(broadcast_dense_sizes).repeat_interleave(nnz_factor, 0);
  // 创建一个新的空指数张量
  Tensor new_indices = indices.new_empty(new_indices_size);
  // 如果广播尺寸不为空，则创建广播指数张量
  if (!broadcast_sizes.empty()) {
    // 根据广播尺寸创建全COO格式的指数张量，并扩展以匹配nnz
    Tensor broadcast_indices = at::sparse::full_coo_indices(broadcast_sizes, indices.options()).tile(nnz);
    // 复制广播的前sparse_extra_ndim维度的索引
    new_indices.narrow(0, 0, sparse_extra_ndim).copy_(broadcast_indices.narrow(0, 0, sparse_extra_ndim));
    // 复制广播的每个维度索引
    for (size_t i=0; i<broadcast_dims.size(); i++) {
      int64_t j=broadcast_dims[i];
      new_indices.select(0, sparse_extra_ndim + j).copy_(broadcast_indices.select(0, sparse_extra_ndim + i));
    }
  }
  // 复制未更改维度的索引，以匹配nnz_factor
  for (int64_t j:unchanged_dims) {
    new_indices.select(0, sparse_extra_ndim + j).copy_(indices.select(0, j).repeat_interleave(nnz_factor));
  }
  // 返回新的稀疏COO张量，指数为new_indices，值为new_values，尺寸为size，选项为self的选项，是否已压缩为is_coalesced
  return at::sparse_coo_tensor(new_indices, new_values, size, self.options(), is_coalesced);
}

// 函数：将输入张量按照指定的大小扩展为符号整数数组，返回扩展后的张量
Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);
}

// 函数：广播多个张量，返回广播后的张量列表
std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  return expand_outplace(tensors);
}

// 静态函数：在第0维上快速连接输出张量和输入张量的内容
static void fastCatOutDim0(const Tensor& out, const MaterializedITensorListRef& inputs) {
  auto outBytes = out.nbytes();  // 输出张量的字节数
  char* dataPtr = reinterpret_cast<char*>(out.data_ptr());  // 输出数据的指针
  size_t totalBytes = 0;  // 累计已复制的字节数
  for (const Tensor& input : inputs) {  // 遍历输入张量列表
    TORCH_CHECK(outBytes >= totalBytes);  // 检查输出张量是否有足够的空间
    if (input.nbytes() > 0) {  // 如果输入张量非空
      std::memcpy(dataPtr + totalBytes, input.const_data_ptr(), input.nbytes());  // 将输入张量的数据复制到输出张量中
    }
    totalBytes += input.nbytes();  // 更新已复制的字节数
  }
  TORCH_CHECK(outBytes == totalBytes);  // 最终检查输出张量是否完全填充
}

// 函数：CPU上的concat操作的实现
TORCH_IMPL_FUNC(cat_out_cpu)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {  // 如果结果张量元素个数为0，则直接返回
    return;
  }

  auto materialized = tensors.materialize();  // 实例化输入张量列表

  bool use_serial_kernel = result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;  // 是否使用串行内核

  ScalarType dtype = materialized[valid].get().scalar_type();  // 取出有效输入张量的数据类型
  bool serial_dtype = at::isFloatingType(dtype);  // 是否为浮点数类型

  // 在结果和所有输入都是连续的，并且内存布局为Contiguous时，使用快速连接的优化路径（仅限于dim=0的情况）
  if (use_serial_kernel && all_contiguous && all_same_dtype && (MemoryFormat::Contiguous == memory_format)) {
    if (dim == 0) {
      fastCatOutDim0(result, materialized);  // 在第0维上快速连接
      return;
    }
    // TODO: 为更高维度的情况添加快速连接支持和多线程支持
  }

  // 在结果和所有输入都是连续的，并且数据类型相同时，使用串行内核
  if (use_serial_kernel && all_contiguous && all_same_dtype && serial_dtype) {
    cat_serial_stub(kCPU, result, materialized, dim);  // 调用串行内核连接操作
    return;
  }

  // 如果所有输入大小和步长都相同，并且结果张量在指定内存格式下是连续的，并且所有输入数据类型相同
  int64_t offset = 0;  // 偏移量初始化为0
  if (all_same_sizes_and_stride && result.is_contiguous(memory_format) &&
      all_same_dtype) {
    const Tensor& source_slice = materialized[valid];  // 取出有效输入张量的引用
    auto slice_dim_size = source_slice.sizes()[dim];  // 取出指定维度的大小
    auto result_slice = result.narrow(dim, 0, slice_dim_size);  // 对结果张量进行narrow操作得到切片
    auto result_slice_data = result_slice.data_ptr();  // 结果切片的数据指针
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());  // 结果张量在指定维度上的步长字节数

    auto iter = TensorIteratorConfig()  // 创建张量迭代器配置对象
      .set_check_mem_overlap(false)  // 不检查内存重叠
      .resize_outputs(false)  // 不调整输出大小
      .add_output(result_slice)  // 添加输出张量切片
      .add_const_input(source_slice)  // 添加常量输入张量
      .enforce_safe_casting_to_output(true)  // 强制安全的类型转换
      .build();  // 构建张量迭代器
    // 遍历 materialized 中的每一个 Tensor 对象
    for (const Tensor& tensor : materialized) {
      // 检查是否应该跳过当前的 tensor
      if (cat_should_skip_tensor(tensor)) {
        continue;  // 如果应该跳过，则继续下一个循环
      }
      // 获取当前 tensor 的数据作为源数据
      auto source_data = static_cast<const char*>(tensor.const_data_ptr());
      // 计算结果数据在结果切片中的位置偏移量
      auto result_data = static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      // 替换迭代器中的操作数，将结果数据和源数据作为操作数
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, const_cast<char*>(source_data));
      // 执行数据拷贝操作
      copy_stub(iter.device_type(), iter, false);
      // 更新偏移量，准备处理下一个切片
      offset += slice_dim_size;
    }
  } else {
    // 如果 materialized 中的 tensor 不需要合并，则执行以下操作
    for (const Tensor& tensor: materialized) {
      // 检查是否应该跳过当前的 tensor
      if (cat_should_skip_tensor(tensor)) {
        continue;  // 如果应该跳过，则继续下一个循环
      }
      // 获取当前 tensor 在指定维度上的大小作为切片的维度大小
      auto slice_dim_size = tensor.sizes()[dim];
      // 根据切片的维度大小在结果张量上创建一个切片
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      // 配置一个张量迭代器，用于执行拷贝操作
      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // 上面已经检查过了
        .resize_outputs(false)          // 不调整输出大小
        .add_output(result_slice)       // 添加结果切片作为输出
        .add_const_input(tensor)        // 添加当前 tensor 作为常量输入
        .promote_inputs_to_common_dtype(true)  // 将输入升级到共同的数据类型
        .cast_common_dtype_to_outputs(true)    // 将共同的数据类型转换为输出类型
        .enforce_safe_casting_to_output(true)  // 强制安全地将类型转换为输出类型
        .build();
      // 执行数据拷贝操作
      copy_stub(iter.device_type(), iter, false);
      // 更新偏移量，准备处理下一个切片
      offset += slice_dim_size;
    }
  }
}

// 将给定的张量列表沿指定维度进行拼接，将结果存入指定的结果张量中
Tensor& cat_out(TensorList tensors, Dimname dim, Tensor& result) {
  // 检查张量列表不为空
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  // 调用 ATen 库的 cat_out 函数进行张量拼接，并返回结果张量的引用
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

// 将给定的张量列表沿指定维度进行拼接，返回拼接后的结果张量
Tensor cat(TensorList tensors, Dimname dim) {
  // 检查张量列表不为空
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  // 调用 ATen 库的 cat 函数进行张量拼接，并返回结果张量
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

// torch.concat 的别名，调用 cat_out 进行张量拼接
Tensor& concat_out(TensorList tensors, Dimname dim, Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

// torch.cat 的别名，调用 cat 进行张量拼接
Tensor concat(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

// 将给定的张量列表沿指定维度进行拼接，将结果存入指定的结果张量中
Tensor & concat_out(TensorList tensors, int64_t dim, Tensor & result) {
  return at::cat_out(result, tensors, dim);
}

// 将给定的张量列表沿指定维度进行拼接，返回拼接后的结果张量
Tensor concat(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

// torch.concatenate 的别名，调用 cat_out 进行张量拼接
Tensor& concatenate_out(TensorList tensors, Dimname dim, Tensor& result) {
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

// torch.cat 的别名，调用 cat 进行张量拼接
Tensor concatenate(TensorList tensors, Dimname dim) {
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

// 将给定的张量列表沿指定维度进行拼接，将结果存入指定的结果张量中
Tensor& concatenate_out(TensorList tensors, int64_t dim, Tensor & result) {
  return at::cat_out(result, tensors, dim);
}

// 将给定的张量列表沿指定维度进行拼接，返回拼接后的结果张量
Tensor concatenate(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

// 检查两个形状是否除了指定维度外其他维度大小都相同
static bool sizes_match_except(IntArrayRef s1, IntArrayRef s2, int64_t dim_except /* should already be wrapped */) {
  // 如果两个形状的维度数不同，则返回 false
  if (s1.size() != s2.size()) {
    return false;
  }
  // 遍历形状的每一个维度，检查除了指定的维度外是否所有维度大小都相同
  for (const auto i : c10::irange(static_cast<int64_t>(s1.size()))) {
    if (i != dim_except && s1[i] != s2[i]) {
      return false;
    }
  }
  // 所有维度都匹配，返回 true
  return true;
}

// 检查稀疏张量的拼接维度是否匹配，包括形状、稀疏维度和密集维度
static void check_cat_sparse_dims(Tensor const &t,
  int64_t pos /* used only for debug messages */,
  IntArrayRef sizes,
  int64_t wrapped,
  int64_t sparse_dim,
  int64_t dense_dim) {
    // 检查张量是否为稀疏张量
    TORCH_CHECK(t.is_sparse(),
            "Concatenating sparse tensors, but a dense tensor was found at position ", pos, ".");
    // 检查张量的形状是否与期望的形状除了拼接维度外都匹配
    TORCH_CHECK(sizes_match_except(sizes, t.sizes(), wrapped),
            "All tensors must have the same shape: ", sizes, " (except in the concatenating dimension),"
            " but found shape: ", t.sizes(), " at position ", pos, ".");
    // 检查稀疏维度和密集维度是否与期望值相符
    TORCH_CHECK(t.sparse_dim() == sparse_dim && t.dense_dim() == dense_dim,
            "All tensors must have the same sparse_dim and dense_dim: ", sparse_dim, ", ", dense_dim,
            ", but tensor at position ", pos, " has ", t.sparse_dim(), ", ", t.dense_dim(), ".");
}
// 将稀疏张量列表沿指定维度拼接起来
static Tensor cat_sparse_impl(const MaterializedITensorListRef& tensors, int64_t dim) {
  // 存储拼接后的索引和数值张量
  std::vector<Tensor> indices;
  std::vector<Tensor> values;

  // 计算可能被包裹的维度
  int64_t wrapped = maybe_wrap_dim(dim, tensors[0].get().dim());
  // 获取第一个张量的稀疏和密集维度
  int64_t sparse_dim = tensors[0].get().sparse_dim();
  int64_t dense_dim = tensors[0].get().dense_dim();
  // 获取第一个张量的尺寸
  IntArrayRef sizes = tensors[0].get().sizes();

  // 如果包裹后的维度小于稀疏维度
  if (wrapped < sparse_dim) {
    // 遍历稀疏张量列表
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      // 检查拼接时的稀疏维度
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      // 将每个张量的索引和数值张量存入对应的向量中
      indices.push_back(t._indices());
      values.push_back(t._values());
    }

    // 沿着第一个维度（dim=1）拼接所有的索引和数值张量
    Tensor idxs = at::cat(indices, 1);
    Tensor vals = at::cat(values, 0);

    // 需要将每个输入张量的索引沿着指定维度 `dim` 上移一定量
    int64_t col = 0;
    int64_t cumulative_offset = 0;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      int64_t this_piece_size = t._nnz();
      // 对第一块不需要做任何操作，因此只处理 i > 0 的情况
      if (i > 0) {
        // 增加相应数量的累积偏移量到 idxs 的指定切片
        idxs[wrapped].narrow(0, col, this_piece_size) += cumulative_offset;
      }
      // 更新累积偏移量
      cumulative_offset += t.size(wrapped);
      col += this_piece_size;
    }

    // 复制尺寸数组，并更新沿指定维度的尺寸为累积偏移量
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = cumulative_offset;

    // 创建并返回新的稀疏 COO 张量
    return native::sparse_coo_tensor(
        idxs,
        vals,
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  }
  else {
    // 在密集维度上拼接需要创建新的数值张量
    // 例如，考虑稀疏三维张量 t1 和 t2，它们在维度 2 上的拼接
    // 创建新的值张量的示例
    // 返回一个新的稀疏 COO 张量
    // 确定在每个张量的值对象中，对应于我们要进行连接的整体维度
    int64_t values_dim = wrapped - sparse_dim + 1;
    // 计算连接后沿连接维度的总尺寸
    const int64_t total_size = std::accumulate(
        tensors.begin(),
        tensors.end(),
        static_cast<int64_t>(0),
        [values_dim](int64_t l, const Tensor& r) {
          return l + r._values().size(values_dim);
        });
    // 获取第一个张量的值对象的大小
    auto zeros_sizes = tensors[0].get()._values().sizes().vec();
    int64_t cumulative_size = 0;
    // 存储分片的值张量和索引张量
    std::vector<Tensor> vals_pieces;
    std::vector<Tensor> idxs_pieces;
    // 遍历每个张量
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      // 检查稀疏张量的连接维度
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      // 将值对象的维度0设为值的数量，而非稀疏张量的任何逻辑维度
      zeros_sizes[0] = t._values().size(0);
      // 设置连接维度的大小
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t._values().size(values_dim);
      // 创建填充为零的张量 z1
      auto z1 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      // 调整连接维度的大小以创建另一个填充为零的张量 z2
      zeros_sizes[values_dim] = total_size - cumulative_size;
      auto z2 = at::zeros(
          zeros_sizes,
          optTypeMetaToScalarType(t._values().options().dtype_opt()),
          t._values().options().layout_opt(),
          t._values().options().device_opt(),
          t._values().options().pinned_memory_opt());
      // 将 z1、t 的值和 z2 沿着连接维度连接起来，存入 vals_pieces
      vals_pieces.push_back(at::cat({z1, t._values(), z2}, values_dim));
      // 将索引张量存入 idxs_pieces
      idxs_pieces.push_back(t._indices());
    }
    // 复制 sizes 到 sizes_copy
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = total_size;
    // 创建一个 COO 格式的稀疏张量，连接所有的索引和值分片
    // 这可能会创建一个未合并的张量
    return native::sparse_coo_tensor(
        at::cat(idxs_pieces, 1),
        at::cat(vals_pieces),
        sizes_copy,
        optTypeMetaToScalarType(tensors[0].get().options().dtype_opt()),
        tensors[0].get().options().layout_opt(),
        tensors[0].get().options().device_opt(),
        tensors[0].get().options().pinned_memory_opt());
  }
}

Tensor block_diag(TensorList tensors) {
  // 定义结果张量
  Tensor result;
  // 如果输入张量列表为空，则创建一个空张量返回
  if (tensors.empty()) {
    result = at::empty({1, 0});
    return result;
  }

  // 获取第一个张量的设备，确保所有输入张量都在同一设备上
  const Device& device = tensors[0].device();
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];

    // 检查每个张量是否在相同的设备上
    TORCH_CHECK(
      tensor.device() == device,
      "torch.block_diag: input tensors must all be on the same device.",
      " Input 0 is on device ", device,
      " and input ", tensor_idx, " is on device ", tensor.device()
    );
  }

  // 确定输出张量的标量类型
  ScalarType output_scalar_type = native::result_type(tensors);
  // 初始化结果张量的维度
  int64_t result_dim0 = 0;
  int64_t result_dim1 = 0;
  // 用于存储所有张量的二维形式
  std::vector<Tensor> tensors_2D(tensors.size());

  // 对每个张量执行以下操作：
  // - 计算张量的维度
  // - 扩展所有0维和1维张量，使其变为2维
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];
    int64_t ndims = tensor.dim();
    // 张量必须是2维或更少维度
    TORCH_CHECK(
      ndims <= 2,
      "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
      tensor_idx, " has ", ndims, " dimensions"
    );

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (ndims == 2) {
      dim0 = tensor.size(0);
      dim1 = tensor.size(1);
      tensors_2D[tensor_idx] = tensor;
    } else if (ndims == 1) {
      // 将第0维度切换为第1维度是有意义的
      dim1 = tensor.size(0);
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    } else {
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    }
    // 累加结果张量的维度
    result_dim0 += dim0;
    result_dim1 += dim1;
  }

  // 创建一个全零的结果张量，与第一个张量的数据类型匹配
  result = at::zeros(
    {result_dim0, result_dim1},
    tensors[0].options().dtype(output_scalar_type)
  );

  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // 将每个二维张量复制到结果矩阵的适当位置
  for (const auto& tensor : tensors_2D) {
    int64_t dim0 = tensor.size(0);
    int64_t dim1 = tensor.size(1);
    result.slice(0, cur_dim0, cur_dim0+dim0).slice(1, cur_dim1, cur_dim1+dim1).copy_(tensor);

    // 更新当前位置
    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }

  // 返回结果张量
  return result;
}
`
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  // 检查输入张量的维度是否大于 0，否则抛出异常
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  // 检查分块数量是否大于 0，否则抛出异常
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  // 获取指定维度的大小
  const auto dim_size = self.sym_size(dim);
  // 计算每个分块的大小，向上取整
  auto split_size = (dim_size + chunks - 1) / chunks;

  // 当 split_size 和维度大小都为 0 时，调用 split_with_sizes 方法处理，避免分块数量丢失的问题
  if (split_size == 0 && dim_size == 0) {
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.split_with_sizes_symint(split_sizes, dim);
  } else {
    // 否则调用 split_symint 方法进行分块
    return self.split_symint(std::move(split_size), dim);
  }
}

std::vector<Tensor> tensor_split_sections_symint(const Tensor& self, c10::SymInt sym_sections, int64_t dim) {
  // 检查输入张量的维度是否大于 0，否则抛出异常
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  // 将 dim 转换为有效的维度索引
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  // 确保 sections 是整数值
  int64_t sections = sym_sections.guard_int(__FILE__, __LINE__);
  // 检查分段数是否大于 0，否则抛出异常
  TORCH_CHECK(sections > 0, "number of sections must be larger than 0, got ", sections);
  // 获取指定维度的大小
  const auto dim_size = self.sym_size(dim_);
  std::vector<Tensor> splits(sections);
  // 计算每个分块的最小大小
  auto min_split_size = dim_size / sections;
  // 计算有额外元素的分块数
  auto num_splits_one_extra = dim_size % sections;
  c10::SymInt start_idx = 0;
  // 遍历所有分段索引，创建各个分块
  for (const auto split_idx : c10::irange(sections)) {
    auto split_size = (num_splits_one_extra > split_idx) ? (min_split_size + 1) : min_split_size;
    splits[split_idx] = at::slice_symint(self, dim_, start_idx, start_idx + split_size);
    start_idx += split_size;
  }
  return splits;
}

template <typename T>
std::vector<Tensor> _tensor_split_indices(const Tensor& self, ArrayRef<T> indices, int64_t dim) {
  // 检查输入张量的维度是否大于 0，否则抛出异常
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  // 将 dim 转换为有效的维度索引
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  // 获取索引数量
  int64_t num_indices = indices.size();
  std::vector<Tensor> splits(num_indices + 1);
  T start_idx(0);
  // 遍历所有索引，创建各个分块
  for (const auto split_idx : c10::irange(num_indices)) {
    auto end_idx = indices[split_idx];
    splits[split_idx] = at::symint::slice<T>(self, dim_, start_idx, end_idx);
    start_idx = end_idx;
  }
  // 添加最后一个分块
  splits[num_indices] = at::symint::slice<T>(self, dim_, start_idx, at::symint::size<T>(self, dim_));
  return splits;
}

std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim) {
  // 调用 _tensor_split_indices 函数进行分块
  return _tensor_split_indices(self, indices, dim);
}
# 使用给定的索引或分段在指定维度上分割张量，返回分割后的张量列表
std::vector<Tensor> tensor_split_indices_symint(const Tensor& self, SymIntArrayRef indices, int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

# 根据给定的索引或分段在指定维度上分割张量
std::vector<Tensor> tensor_split(const Tensor& self, const Tensor& tensor_indices_or_sections, int64_t dim) {
  # 检查张量的维度是否大于0
  TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
  # 获取tensor_indices_or_sections的设备类型
  auto split_device = tensor_indices_or_sections.device();
  # 检查tensor_indices_or_sections是否在CPU上
  TORCH_CHECK(split_device == kCPU,
    "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on ", split_device);
  # 获取tensor_indices_or_sections的数据类型
  auto split_dtype = tensor_indices_or_sections.scalar_type();
  # 检查tensor_indices_or_sections是否为长整型
  TORCH_CHECK(split_dtype == at::kLong,
    "tensor_split expected tensor_indices_or_sections to have dtype of long, but got ", split_dtype);
  # 获取tensor_indices_or_sections的维度
  auto split_dim = tensor_indices_or_sections.dim();
  # 检查tensor_indices_or_sections是否为零维或一维张量
  TORCH_CHECK(split_dim == 1 || split_dim == 0,
    "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with ", split_dim, " dims");

  if (split_dim == 0) {
    # 如果tensor_indices_or_sections是零维张量，则使用其值作为分割段数
    int64_t sections = tensor_indices_or_sections.item<int64_t>();
    return self.tensor_split(sections, dim);
  } else {
    # 否则，从tensor_indices_or_sections中获取索引数据并分割张量
    auto indices_data = tensor_indices_or_sections.const_data_ptr<int64_t>();
    auto stride = tensor_indices_or_sections.stride(0);
    auto numel = tensor_indices_or_sections.numel();
    std::vector<int64_t> indices(numel);
    for (const auto offset : c10::irange(numel)) {
      // 索引张量可能是非连续的
      indices[offset] = *(indices_data + offset * stride);
    }
    return self.tensor_split(indices, dim);
  }
}

# 将张量不安全地按指定数量和维度分块
std::vector<Tensor> unsafe_chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  # 检查张量的维度是否大于0
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  # 检查chunks是否大于0
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.size(dim);
  # 计算每块的大小
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  // 参见上述chunk(...)函数中的注释
  if (split_size == 0 && dim_size == 0) {
    # 处理特殊情况：如果split_size为0且dim_size也为0，则手动计算每块的大小
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.unsafe_split_with_sizes(split_sizes, dim);
  } else {
    # 否则，按计算得到的split_size分割张量
    return self.unsafe_split(split_size, dim);
  }
}

# 将输入张量视图连续化后，对其进行偏移，返回偏移后的对角线张量
Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}
// 返回具有给定偏移量的张量的对角线视图
Tensor diagonal(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  // 获取张量的维度数
  int64_t nDims = self.dim();
  // 将维度 dim1_ 转换为有效的维度索引
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  // 将维度 dim2_ 转换为有效的维度索引
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  // 检查对角线的维度不能相同
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  // 计算对角线的输出维度名称
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  // 保证返回的张量没有命名
  NoNamesGuard no_names_guard;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t diag_size;
  // 获取张量的存储偏移量
  int64_t storage_offset = self.storage_offset();
  // 计算对角线的存储偏移量和大小
  // 对于正的偏移量（在主对角线之上），删除沿 dim2 的左侧列
  // 对于负的偏移量（在主对角线之下），删除沿 dim1 的顶部行
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(self.size(dim1), self.size(dim2)-offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(self.size(dim1)+offset, self.size(dim2)), 0);
  }

  // 如果对角线大小为 0，则跳过
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    // 根据正偏移量更新存储偏移量
    storage_offset += offset * self.stride(dim2);
  } else {
    // 根据负偏移量更新存储偏移量
    storage_offset -= offset * self.stride(dim1);
  }

  // 构造新的大小和步长：删除 dim1 和 dim2，并将新维度追加到形状和步长的末尾，以匹配 numpy 语义
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1)+self.stride(dim2));

  // 返回具有新参数的视图
  auto result = self.as_strided(sizes, strides, storage_offset);

  // 重置无命名保护
  no_names_guard.reset();
  // 如果输出名称非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}

// 返回具有命名维度的张量的对角线视图
Tensor diagonal(const Tensor& self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  // 调用底层的对角线函数，根据命名维度计算索引位置
  auto result = at::diagonal(
      self,
      offset,
      dimname_to_position(self, dim1),
      dimname_to_position(self, dim2));
  // 目前没有方法可以原地修改张量的名称，因此此处的操作速度较慢。
  // 未来可能考虑提供该功能。
  // 更新最后一个维度的名称为输出维度的名称
  std::vector<Dimname> new_names = result.names().vec();
  new_names[new_names.size() - 1] = outdim;
  return result.refine_names(new_names);
}
// 构造函数 diag_embed，用于在张量中插入对角线值
Tensor diag_embed(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  // 计算输入张量的维度数
  int64_t nDims = self.dim() + 1;
  // 确保维度 dim1_ 在有效范围内
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  // 确保维度 dim2_ 在有效范围内
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  // 检查对角线维度是否不同
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  // 计算新对角线长度
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  // 获取输入张量的大小，并移除最后一个维度
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  // 在合适的位置插入新的对角线长度
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  // 创建一个全零张量，大小与修改后的 sizes 相同
  auto result = at::zeros(sizes, self.options());
  // 获取结果张量的对角线视图，并复制输入张量的值到对角线
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  // 返回结果张量
  return result;
}

// 函数 expand，用于扩展张量的大小
Tensor expand(const Tensor& self, c10::IntArrayRef size, bool /*unused*/) {
  // 检查提供的大小是否足够扩展张量的所有维度
  TORCH_CHECK(size.size() >= (size_t)self.dim(),
           "expand(", self.toString(), "{", self.sizes(), "}, size=", size,
           "): the number of sizes provided (", size.size(), ") ",
           "must be greater or equal to the number of dimensions in the tensor (",
           self.dim(), ")");
  // 检查张量是否是稀疏张量或压缩稀疏张量，这些情况下不支持扩展操作
  TORCH_CHECK(!self.is_sparse() && !at::sparse_csr::is_sparse_compressed(self),
            "expand is unsupported for ", self.layout(), " tensors");

  // 推断扩展后张量的大小和步长
  auto expandedSizesAndStrides = inferExpandGeometry_dimvector(self.sizes(), self.strides(), size);

  // 使用推断得到的大小和步长创建视图张量
  auto result = self.as_strided(
      expandedSizesAndStrides.sizes, expandedSizesAndStrides.strides);
  // 传播扩展后的张量的命名属性
  namedinference::propagate_names_for_expand(result, self);
  // 返回结果张量
  return result;
}

// 函数 expand_as，将张量扩展为另一个张量的大小
Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand_symint(other.sym_sizes());
}

// 函数 sum_to_size_symint，将张量按照给定大小求和
Tensor sum_to_size_symint(const Tensor& self, SymIntArrayRef size) {
  // 检查是否可以将张量扩展到给定大小
  TORCH_CHECK(is_expandable_to(size, self.sym_sizes()),
           "size {", size, "} is not expandable to size {", self.sizes(), "}.");

  // 将张量按照给定大小进行求和
  return sum_to(self, size);
}

// 不支持对 unfold、diagonal、expand、permute 进行通道级量化
// TODO: 将此函数转换为 ATen 函数，并在完成后替换 as_strided_qtensorimpl。
// 创建量化张量的函数 make_qtensor
static Tensor make_qtensor(const Tensor& self, IntArrayRef size, IntArrayRef stride, QuantizerPtr quantizer) {
  // 使用指定的存储、数据类型和量化器创建量化张量
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  // 设置量化张量的大小、步长和存储偏移
  setStrided(result, size, stride, self.storage_offset());
  // 返回创建的量化张量
  return result;
}

// 函数 as_strided_tensorimpl，创建具有指定大小和步长的张量
Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  // 内部断言：不支持 MPS 的情况下使用 as_strided_tensorimpl，建议使用 self.as_strided(...)
  TORCH_INTERNAL_ASSERT(!self.is_mps(), "as_strided_tensorimpl does not work with MPS; call self.as_strided(...) instead");
  // 获取存储偏移量，如果未提供则使用张量的默认偏移量
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  // 使用指定的存储、数据类型和量化器创建张量
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  // 设置张量的大小、步长和存储偏移
  setStrided(result, size, stride, storage_offset);
  // 返回创建的张量
  return result;
}

template <typename T>
inline void setStridedUnchecked(
    const Tensor& self,
    // 使用模板参数 T 创建数组引用 size，表示张量的尺寸
    ArrayRef<T> size,
    // 使用模板参数 T 创建数组引用 stride，表示张量的步长
    ArrayRef<T> stride,
    // 使用右值引用 T&& 创建 storage_offset，表示张量的存储偏移量
    T&& storage_offset) {
  // 获取 self 对象的底层张量实现指针
  auto* self_ = self.unsafeGetTensorImpl();
  // 设置张量实现的尺寸、步长及存储偏移量
  self_->set_sizes_and_strides(size, stride, c10::make_optional(std::forward<T>(storage_offset)));
// 返回一个新的 Tensor，其元数据被设置为 VIEW 模式，使用指定的符号大小、符号步长和可选的符号存储偏移量
Tensor as_strided_tensorimpl_meta_symint(const Tensor& self, SymIntArrayRef sym_size, SymIntArrayRef sym_stride, optional<c10::SymInt> sym_storage_offset_) {
  // 如果未提供符号存储偏移量，使用 self 的符号存储偏移量
  auto sym_storage_offset = sym_storage_offset_.value_or(self.sym_storage_offset());
  // 创建一个新的 TensorImpl，其类型为 VIEW，使用 self 的存储，键集合和数据类型
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  // 设置未经检查的步长（strided），使用指定的符号大小、符号步长和符号存储偏移量
  setStridedUnchecked(result, sym_size, sym_stride, std::move(sym_storage_offset));
  // 返回新创建的 Tensor
  return result;
}

// 返回一个新的 QTensorImpl，其元数据被设置为 VIEW 模式，使用指定的大小、步长和可选的存储偏移量
Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  // 如果未提供存储偏移量，使用 self 的存储偏移量
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  // 获取量化器
  auto quantizer = get_qtensorimpl(self)->quantizer();
  // 检查量化方案是否为 PER_TENSOR_AFFINE，否则抛出错误
  TORCH_CHECK(
      quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE,
      "Setting strides is possible only on uniformly quantized tensor");
  // 创建一个新的 QTensorImpl，其类型为 VIEW，使用 self 的存储，键集合，数据类型和量化器
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  // 设置步长（strided），使用指定的大小、步长和存储偏移量
  setStrided(result, size, stride, storage_offset);
  // 返回新创建的 QTensorImpl
  return result;
}

// 此函数重载了上一个函数，额外接受一个量化器参数，并且不通过分派器可用
// 返回一个新的 QTensorImpl，其元数据被设置为 VIEW 模式，使用指定的大小、步长、存储偏移量和量化器
// TODO: 使此函数与分派器兼容
static Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_,
  QuantizerPtr quantizer) {
  // 如果未提供存储偏移量，使用 self 的存储偏移量
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  // 检查量化方案是否为 PER_TENSOR_AFFINE 或 PER_CHANNEL_AFFINE，否则抛出错误
  TORCH_CHECK(
      (quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE) ||
      (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE),
      "Setting strides is possible only on uniformly or per channel quantized tensors");
  // 创建一个新的 QTensorImpl，其类型为 VIEW，使用 self 的存储，键集合，数据类型和量化器
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  // 设置步长（strided），使用指定的大小、步长和存储偏移量
  setStrided(result, size, stride, storage_offset);
  // 返回新创建的 QTensorImpl
  return result;
}

// 返回一个新的 Tensor，其元数据被设置为 VIEW 模式，使用指定的符号大小、符号步长和可选的符号存储偏移量
const Tensor &as_strided__symint(const Tensor& self, SymIntArrayRef size, SymIntArrayRef stride, optional<c10::SymInt> storage_offset_) {
  // 如果未提供符号存储偏移量，使用 self 的符号存储偏移量
  auto storage_offset = storage_offset_.value_or(self.sym_storage_offset());
  // 设置步长（strided），使用指定的符号大小、符号步长和符号存储偏移量
  setStrided(self, size, stride, std::move(storage_offset));
  // 返回原始的 Tensor 引用
  return self;
}
// 在 CPU 上实现稠密张量的窄切片操作，返回一个新的张量
Tensor narrow_copy_dense_cpu(const Tensor& self, int64_t dim, int64_t start, int64_t length){
  // narrow_copy_dense_cpu_out 总是会调整输出张量的大小，因此这里只创建一个大小为零的张量。
  auto output = at::empty({0}, self.options());
  // 调用 narrow_copy_dense_cpu_out 函数执行实际的窄切片操作，并将结果返回
  return narrow_copy_dense_cpu_out(self, dim, start, length, output);
}

// 在稀疏张量上实现窄切片操作，返回一个新的稀疏张量
Tensor narrow_copy_sparse(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  int64_t allDim = self.dim();
  int64_t end = start+length;
  // 检查张量维度是否大于 0
  TORCH_CHECK(allDim > 0, "narrow() cannot be applied to a 0-dim tensor.");
  // 检查长度是否为非负数
  TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
  // 检查切片维度是否在有效范围内
  TORCH_CHECK(dim >= 0 && dim < allDim,
    "Dimension ", dim, " out of range. Expecting 0 <= dim < ", allDim, ".");
  // 检查切片范围是否有效
  TORCH_CHECK(start >= 0 && end <= self.size(dim),
    "Invalid range to narrow. range(start, start+length) must be a subset of range(0, ", self.size(dim), ").")
  // 获取稀疏张量的索引
  Tensor indices = self._indices();
  int64_t sparse_dim = self.sparse_dim();

  // 创建新的大小向量，用于存储切片后的张量大小
  std::vector<int64_t> new_sizes = self.sizes().vec();
  new_sizes[dim] = length;

  Tensor new_values;
  Tensor new_indices;
  // 如果切片的维度小于稀疏维度，执行稀疏切片操作
  if (dim < sparse_dim) {
    // 创建一个掩码，标记符合切片条件的索引
    Tensor mask = (indices[dim] >= start).__and__((indices[dim] < end));
    // 根据掩码选择符合条件的索引，并重新视图化为稀疏张量的索引结构
    new_indices = indices.masked_select(mask).view({sparse_dim, -1});
    // 调整新索引中的维度，以适应切片后的起始位置
    new_indices[dim].sub_(start);
    // 根据非零掩码选择对应的值
    Tensor nzIndices = mask.nonzero().view(-1);
    new_values = self._values().index_select(0, nzIndices);
  } else {
    // 否则，执行在稠密维度上的切片操作，实际上是在 _values() 上进行切片
    new_indices = indices;
    int64_t dense_dim = dim - sparse_dim + 1;
    new_values = self._values().narrow_copy(dense_dim, start, length);
  }

  // 根据新的索引、值和大小创建一个新的稀疏 COO 张量，并返回
  return at::sparse_coo_tensor(new_indices, new_values, new_sizes, self.options(), self.is_coalesced());
}

// 这个函数实际上应该使用 narrow_copy_out，但是这个 API 在 Meta 内部使用：
// https://github.com/pytorch/pytorch/pull/87045#issuecomment-1309353561
// 在 CPU 上实现稠密张量的窄切片操作，并将结果存储到指定的输出张量中
Tensor& narrow_copy_dense_cpu_out(
  const Tensor& self, int64_t dim, int64_t start, int64_t length, Tensor& output
) {
  // 检查张量维度是否大于 0
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  // 检查输出张量的数据类型是否与输入张量相同
  TORCH_CHECK(self.dtype() == output.dtype());

  // 获取连续化的自身张量，并获取其大小
  auto self_contig = self.expect_contiguous();
  const auto self_sizes = self_contig->sizes();

  // 如果维度是负数，则进行维度包装并进行边界检查
  if (dim < 0) {
    dim = at::maybe_wrap_dim(dim, self_sizes.size());
  } else {
    TORCH_CHECK(dim < static_cast<int64_t>(self_sizes.size()));
  }

  // 包装起始位置并进行边界检查
  const auto cur_size = self_sizes[dim];
  TORCH_CHECK_INDEX(
    -cur_size <= start && start <= cur_size,
    "start out of range (expected to be in range of [", -cur_size, ", ", cur_size, "], but got ", start, ")"
  )
  if (start < 0) {
    start = start + cur_size;
  }
  // 检查起始位置和长度是否符合要求
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");

  // 调整输出张量的大小
  auto output_sizes = self_sizes.vec();
  output_sizes[dim] = length;
  at::native::resize_(output, output_sizes);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 定义单位大小，用于计算块的大小
  const int64_t unit = c10::size_from_dim_(dim + 1, self_sizes);
  const int64_t num_blocks = c10::size_to_dim_(dim, self_sizes);

  // 计算元素大小
  const auto itemsize = self_contig->dtype().itemsize();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  // 计算源张量和目标张量的字节大小
  size_t src_nbytes = itemsize * self_contig->numel();
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  size_t dst_nbytes = itemsize * output.numel();

  // 计算源块和目标块的大小
  size_t src_block_size = unit * self_sizes[dim];
  size_t dst_block_size = unit * length;

  // 如果没有块或目标块大小为0，则直接返回输出张量
  if (num_blocks == 0 || dst_block_size == 0) {
    return output;
  }

  // 获取源张量和目标张量的字节指针
  const char* src_bytes = static_cast<const char*>(self_contig->const_data_ptr());
  char* dst_bytes = static_cast<char*>(output.data_ptr());

  // 计算源块和目标块的字节大小
  size_t src_block_size_bytes = itemsize * src_block_size;
  size_t dst_block_size_bytes = itemsize * dst_block_size;
  size_t src_offset = unit * start;

  // 计算起始字节的偏移量
  const char* src_offset_bytes = src_bytes + itemsize * src_offset;
  char* dst_offset_bytes = dst_bytes;

  // 遍历每个块，执行内存拷贝操作
  for (const auto i : c10::irange(num_blocks)) {
    const char* local_src_offset_bytes = src_offset_bytes + i * src_block_size_bytes;
    char* local_dst_offset_bytes = dst_offset_bytes + i * dst_block_size_bytes;

    // 内部断言，确保内存访问的合法性
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<const void*>(local_src_offset_bytes + dst_block_size_bytes) <=
        static_cast<const void*>(src_bytes + src_nbytes));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<void*>(local_dst_offset_bytes + dst_block_size_bytes) <=
        static_cast<void*>(dst_bytes + dst_nbytes));

    // 执行内存拷贝操作
    memcpy(
        local_dst_offset_bytes, local_src_offset_bytes, dst_block_size_bytes);
  }
  // 返回调整大小后的输出张量
  return output;
}

// 定义了一个名为 narrow 的函数，用于在指定维度上对张量进行切片操作
Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  // 检查张量 self 的维度是否大于 0
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  // 检查切片长度是否为非负数
  TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
  auto cur_size = self.size(dim);
  // 检查 start 是否在合法范围内
  TORCH_CHECK_INDEX(
    -cur_size <= start && start <= cur_size,
    "start out of range (expected to be in range of [", -cur_size, ", ", cur_size, "], but got ", start, ")"
  )
  // 如果 start 是负数，将其转换为非负数索引
  if (start < 0) {
    start = start + cur_size;
  }
  // 检查切片的起始索引和长度是否超出维度的范围
  TORCH_CHECK(start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  // 调用 ATen 库的 slice 函数执行张量的切片操作
  return at::slice(self, dim, start, start + length, 1);
}

// 定义了一个名为 narrow_symint 的函数，用于在指定维度上对符号整数类型的张量进行切片操作
Tensor narrow_symint(const Tensor& self, int64_t dim, SymInt start, SymInt length) {
  // 检查张量 self 的维度是否大于 0
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  // 检查切片长度是否为非负数
  TORCH_SYM_CHECK(length.sym_ge(0), "narrow(): length must be non-negative.");
  auto cur_size = self.sym_size(dim);
  // 检查 start 是否在合法范围内，使用符号整数的符号比较函数
  TORCH_CHECK_INDEX(
    ((-cur_size).sym_le(start).sym_and(start.sym_le(cur_size))).expect_true(__FILE__, __LINE__),
    "start out of range (expected to be in range of [", -cur_size, ", ", cur_size, "], but got ", start, ")"
  )
  // 如果 start 是负数，将其转换为非负数索引
  if (start < 0) {
    start = start + cur_size;
  }
  // 检查切片的起始索引和长度是否超出维度的范围，使用符号整数的符号比较函数
  TORCH_SYM_CHECK(start.sym_le(cur_size - length),
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  // 调用 ATen 库的 slice_symint 函数执行符号整数类型张量的切片操作
  return at::slice_symint(self, dim, start, start + length, 1);
}

// 该函数是 narrow 的重载版本，用于处理在 XLA 中传递符号整数类型的 start 参数
// 该函数仅在 XLA 中使用，将 start 参数转换为符号整数类型后再调用 narrow_symint 函数
Tensor narrow_tensor_symint(const Tensor& self, int64_t dim, const Tensor& start, SymInt length) {
  // 检查 start 张量是否为 0 维度的整数张量
  TORCH_CHECK(start.dim() == 0 && isIntegralType(start.scalar_type(), /*includeBool=*/false),
              "start must be an 0-dim integral Tensor.");
  // 将 start 张量的值转换为 int64_t 类型
  int64_t st = start.item<int64_t>();
  // 调用 narrow_symint 函数执行符号整数类型张量的切片操作
  return at::narrow_symint(self, dim, c10::SymInt(st), std::move(length));
}

// 该函数用于在进行维度重排（permute）操作时，估计新张量的大小和步长
std::tuple<DimVector, DimVector, std::vector<int64_t>>
static _permute_size_stride_estimation(const Tensor& self, IntArrayRef dims) {
  // 获取张量的当前维度数
  const auto ndim = self.dim();
  // 检查张量的维度数与指定维度数组的长度是否相等
  TORCH_CHECK(ndim == static_cast<int64_t>(dims.size()),
      "permute(sparse_coo): number of dimensions in the tensor input ",
      "does not match the length of the desired ordering of dimensions ",
      "i.e. input.dim() = ", ndim, " is not equal to len(dims) = ", dims.size());

  // 检查张量是否使用步进布局（strided layout）
  const auto is_strided_layout = self.options().layout() == at::kStrided;
  // 获取张量的原始大小和步长
  const auto old_sizes = self.sizes();
  const auto old_strides = is_strided_layout ? self.strides() : IntArrayRef{};

  // 创建新张量的大小向量和步长向量
  auto new_sizes = DimVector(ndim);
  auto new_strides = DimVector(is_strided_layout ? ndim : 0);
  // 创建用于包装维度的数组和用于记录维度是否已经出现的数组
  auto wrapped_dims = std::vector<int64_t>(ndim);
  std::vector<bool> seen_dims(ndim);

  // 遍历指定的维度数组进行处理
  for (const auto i : c10::irange(ndim)) {
    // 获取包装后的维度值
    const auto d = maybe_wrap_dim(dims[i], ndim);
    // 检查是否有重复的维度
    TORCH_CHECK(!seen_dims[d],
        "permute(): duplicate dims are not allowed.");
    seen_dims[d] = true;
    # 将 wrapped_dims 数组的第 i 个位置设为 d
    wrapped_dims[i] = d;
    # 将 new_sizes 数组的第 i 个位置设为 old_sizes 数组中索引为 d 的值
    new_sizes[i] = old_sizes[d];
    # 如果当前布局是跨步布局
    if (is_strided_layout) {
      # 将 new_strides 数组的第 i 个位置设为 old_strides 数组中索引为 d 的值
      new_strides[i] = old_strides[d];
    }
  }

  # 返回一个包含 new_sizes、new_strides 和 wrapped_dims 的元组
  return std::make_tuple(new_sizes, new_strides, wrapped_dims);
}

// 函数：根据给定的维度重新排列稀疏 COO 张量
Tensor permute(const Tensor& self, IntArrayRef dims) {
  // 估计新的大小和步长信息
  auto [new_sizes, new_strides, _] = _permute_size_stride_estimation(self, dims);
  // 返回按照新的大小和步长重新排列的张量
  return self.as_strided(new_sizes, new_strides);
}

// 函数：根据给定的维度重新排列稀疏 COO 张量
Tensor permute_sparse_coo(const Tensor& self, IntArrayRef dims) {
  // 估计新的大小、步长以及封装后的维度信息
  auto [new_sizes, _, wrapped_dims] = _permute_size_stride_estimation(self, dims);

  // 获取张量的总维度数、稀疏维度数和密集维度数
  const auto ndim = self.dim();
  const auto sparse_ndim = self.sparse_dim();
  const auto dense_ndim = self.dense_dim();

  // 初始化维度和稀疏-密集维度的排列索引
  auto dims_id_perm = std::vector<int64_t>(ndim);
  auto dims_sparse_dense_id_perm = std::vector<int64_t>(ndim);
  for (const auto i : c10::irange(ndim)) {
    dims_id_perm[i] = i;
    dims_sparse_dense_id_perm[i] = wrapped_dims[i];
  }
  // 对稀疏维度和密集维度的排列索引进行排序
  std::sort(dims_sparse_dense_id_perm.begin(), dims_sparse_dense_id_perm.begin() + sparse_ndim);
  std::sort(dims_sparse_dense_id_perm.begin() + sparse_ndim, dims_sparse_dense_id_perm.end());
  // 检查排序后的稀疏-密集维度索引是否与原始索引相匹配
  TORCH_CHECK(dims_sparse_dense_id_perm == dims_id_perm,
      "permute(sparse_coo): transpositions between sparse and dense dimensions are not allowed.",
      "Only transpositions within sparse and dense dimensions are supported.");

  // 函数：从 vector 中提取指定范围的元素
  const auto slice = [](std::vector<int64_t> v, size_t begin, size_t len) -> decltype(v) {
    return std::vector<int64_t>{v.begin() + begin, v.begin() + begin + len};
  };

  // 提取旧的稀疏维度和密集维度
  auto old_sparse_dims = slice(dims_id_perm, 0, sparse_ndim);
  auto old_dense_dims = slice(std::move(dims_id_perm), sparse_ndim, ndim - sparse_ndim);
  // 提取新的稀疏维度和密集维度
  auto new_sparse_dims = slice(wrapped_dims, 0, sparse_ndim);
  auto new_dense_dims = slice(std::move(wrapped_dims), sparse_ndim, ndim - sparse_ndim);

  // 获取旧的索引和值
  auto old_indices = self._indices();
  auto old_values = self._values();

  // 创建新的稀疏维度索引
  const auto new_indices = (new_sparse_dims == old_sparse_dims)
    ? std::move(old_indices)
    : [&]() -> Tensor {
      auto sparse_perm_tensor = at::from_blob(reinterpret_cast<void*>(new_sparse_dims.data()),
          {sparse_ndim}, old_indices.options().device(at::kCPU));
      // 如果允许 COO 存储一个排列向量，则可以避免创建新的索引
      return old_indices.index_select(0, sparse_perm_tensor.to(self.device().type()));
    }();

  // 创建新的密集维度值
  const auto new_values = (new_dense_dims == old_dense_dims)
    ? std::move(old_values)
    : [&]() -> Tensor {
      auto values_perm = std::vector<int64_t>(dense_ndim + 1);
      for (const auto i : c10::irange(dense_ndim)) {
        values_perm[i + 1] = new_dense_dims[i] - sparse_ndim + 1;
      }
      return old_values.permute(values_perm);
    }();

  // 检查是否需要进行稀疏张量的合并操作
  const auto is_coalesced = self.is_coalesced() && (dims.empty() || dims[0] == 0);
  // TODO: 应用 `is_coalesced ||= new_values.size(0) < 2`.
  // 返回按照新的维度和张量创建的稀疏 COO 张量
  return _sparse_coo_tensor_with_dims_and_tensors(
       sparse_ndim, dense_ndim, new_sizes, new_indices, new_values, self.options(), is_coalesced);
}
// 重复给定的张量 `self`，使其维度与 `repeats` 相匹配
Tensor repeat(const Tensor& self, IntArrayRef repeats) {
  // 检查重复的维度数量不能少于张量 `self` 的维度数
  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // 如果目标维度数大于源张量的维度数，则在张量前面添加新的前导维度
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for(const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // 扩展张量 `self` 到 `padded_size`
  Tensor xtensor = self.expand(padded_size);

  Tensor result;
  // 根据张量 `self` 是否量化来创建空张量 `result`
  if (self.is_quantized()) {
    result = at::empty_quantized(target_size, self);
  } else {
    result = at::empty(target_size, self.options());
  }

  // 如果 `repeats` 中有维度为零，则返回一个空张量
  if (zero_tensor) {
    return result;
  }

  // 别名张量 `result` 为 `urtensor`，用于后续操作
  Tensor urtensor = at::alias(result);
  for (const auto i : c10::irange(xtensor.dim())) {
    // 如果步长为0，则展开时设置步长为至少1
    auto size_i = xtensor.sizes()[i];
    urtensor = urtensor.unfold(i, size_i, std::max<int64_t>(size_i, 1));
  }

  // 将 `xtensor` 的数据复制到 `urtensor` 中，以达到重复的效果
  urtensor.copy_(xtensor.expand_as(urtensor));

  return result;
}

// 使用符号整数数组 `reps` 对张量 `self` 进行平铺
Tensor tile_symint(const Tensor& self, SymIntArrayRef reps){
  // 如果 `self` 的维度大于 `reps` 的长度，将 `reps` 前置1直到维度与 `self` 匹配
  const int64_t size_diff = self.dim() - static_cast<int64_t>(reps.size());
  if (size_diff > 0){
    std::vector<c10::SymInt> new_reps(size_diff, 1);
    for (const auto i : c10::irange(reps.size())) {
      new_reps.emplace_back(reps[i]);
    }
    // 使用新的 `reps` 对 `self` 进行符号整数重复操作
    return self.repeat_symint(SymIntArrayRef(new_reps));
  }
  // 对 `self` 使用符号整数数组 `reps` 进行重复操作
  // `torch.tile` 相当于已实现的 `torch.Tensor.repeat`
  return self.repeat_symint(reps);
}

//
// 用于 ArrayRef<int64_t> 和 SmallVector<int64_t> 用例的模板
//
template <typename Vec>
Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const Vec& sizes,
    const Vec& strides) {
  // 调用者应确保 sizes 和 strides 对于 self 是有效的
  // （存储足够，步长非负，大小和步长数组的大小相同）
  Tensor self_;
  if (self.is_quantized()) {
    // 创建量化张量视图 `self_`，并设置存储偏移量
    self_ = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), get_qtensorimpl(self)->quantizer());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());

    // 创建量化张量视图 `self_`，并设置存储偏移量
    self_ = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), get_qtensorimpl(self)->quantizer());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
  } else {
    // 创建普通张量视图 `self_`，并设置存储偏移量
    self_ = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
  }

  // 返回新创建的视图 `self_`
  return self_;
}
    self_tmp_->set_sizes_and_strides(sizes, strides);

# 设置当前张量实现对象的尺寸和步长

  } else {

# 如果不是共享模式，创建一个新的张量实现对象作为视图

    self_ = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());

# 使用当前张量的存储、键集和数据类型创建一个新的张量实现对象作为视图

    auto* self_tmp_ = self_.unsafeGetTensorImpl();

# 获取新创建视图的张量实现对象的指针

    self_tmp_->set_storage_offset(self.storage_offset());

# 设置视图的存储偏移量

    self_tmp_->set_sizes_and_strides(sizes, strides);

# 设置视图的尺寸和步长

  }

# 结束条件分支

  namedinference::propagate_names(self_, self);

# 根据推断的命名信息，将命名信息传播到当前张量视图

  return self_;

# 返回创建或更新后的张量视图```cpp
    self_tmp_->set_sizes_and_strides(sizes, strides);

# 设置当前张量实现对象的尺寸和步长


  } else {

# 如果不是共享模式，创建一个新的张量实现对象作为视图


    self_ = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());

# 使用当前张量的存储、键集和数据类型创建一个新的张量实现对象作为视图


    auto* self_tmp_ = self_.unsafeGetTensorImpl();

# 获取新创建视图的张量实现对象的指针


    self_tmp_->set_storage_offset(self.storage_offset());

# 设置视图的存储偏移量


    self_tmp_->set_sizes_and_strides(sizes, strides);

# 设置视图的尺寸和步长


  }

# 结束条件分支


  namedinference::propagate_names(self_, self);

# 根据推断的命名信息，将命名信息传播到当前张量视图


  return self_;

# 返回创建或更新后的张量视图
// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
// 根据给定的sizes和strides对self进行大小和步长的别名设置
Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides) {
  //caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array size is the same)
  // 调用者应确保sizes和strides对self有效
  Tensor self_;
  if (self.is_quantized()) {
    // 使用make_tensor创建一个量化张量，使用给定的sizes和strides进行初始化
    self_ = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype(), get_qtensorimpl(self)->quantizer());
    // 设置张量的sizes和strides，以及符号存储偏移
    self_.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, self.sym_storage_offset());
  } else {
    // 使用make_tensor创建一个标准张量，使用给定的sizes和strides进行初始化
    self_ = at::detail::make_tensor<TensorImpl>(
    c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
    // 设置张量的sizes和strides，以及符号存储偏移
    self_.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, self.sym_storage_offset());
  }
  // 将self的命名信息传播到self_
  namedinference::propagate_names(self_, self);
  // 返回设置好sizes和strides的self_
  return self_;
}

// 对于给定的self和proposed_shape，重塑张量（包含符号整数）
Tensor reshape_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    // 对于稀疏张量，不支持reshape操作，抛出错误
    TORCH_CHECK(false, "reshape is not implemented for sparse tensors");
  }

  if (self.is_contiguous() && !self.is_mkldnn()) {
    // 如果张量是连续的且不是mkldnn格式，直接使用view_symint进行视图重塑
    return self.view_symint(proposed_shape);
  }

  // 推断出形状的SymDimVector
  c10::SymDimVector shape = infer_size_dv(proposed_shape, self.sym_numel());

  if (self.is_mkldnn()) {
    // 如果是mkldnn格式的张量，使用_mkldnn_reshape进行重塑
    return at::_mkldnn_reshape(self, C10_AS_INTARRAYREF_SLOW(shape));
  }

  // `computeStride`返回适合这个reshape的合适步长
  auto stride = at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape);

  // 注意：即使我们已经有了可视几何和目标步长，
  // 我们不直接在self上调用as_strided，因为它的反向传播效率不如view（因为as_strided处理一般情况）。
  //
  // 同样，我们不调用view，因为它会重复我们已经完成的一些工作，
  // 而是调用我们的内部/私有操作符 `_reshape_alias`，
  // 它本质上与view和as_strided做相同的事情，但没有额外的开销。
  if (stride.has_value()) {
    // 临时检查，如果设备不支持（例如对于XLA，该操作不受支持，因此我们使用`view`）
    // 我们需要在这里进行检查，而不是在`native_functions.yaml`中，
    // 以保持向后兼容性。
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu() && !at::isTensorSubclassLike(self)) {
      // 在不支持的设备上使用_reshape_alias_symint进行重塑
      return self._reshape_alias_symint(shape, stride.value());
    } else {
      // 否则使用view_symint进行视图重塑
      return self.view_symint(shape);
    }
  }
  // 如果无法计算出合适的步长，返回不安全的视图重塑结果
  return at::_unsafe_view_symint(self.clone(at::MemoryFormat::Contiguous), shape);
}
// 对具有符号整数类型的张量进行重新形状操作，并返回重新形状后的副本
Tensor _reshape_copy_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  // 如果张量是稀疏张量，则抛出错误，因为不支持稀疏张量的 reshape 操作
  if (self.is_sparse()) {
    TORCH_CHECK(0, "_reshape_copy is not implemented for sparse tensors");
  }
  // 根据给定的 proposed_shape 推断出最终的形状 shape
  c10::SymDimVector shape = infer_size_dv(proposed_shape, self.sym_numel());

  // 如果张量是 mkldnn 类型，则抛出错误，因为不支持 mkldnn 张量的 reshape 操作
  if (self.is_mkldnn()) {
    TORCH_CHECK(0, "_reshape_copy not implemented for mkldnn tensors");
  }

  // 如果张量是连续的，则返回其按照指定形状重新形状后的克隆，并指定内存格式为连续
  if (self.is_contiguous()) {
    return self.view_symint(shape).clone(at::MemoryFormat::Contiguous);
  } else {
    // 如果张量不是连续的，则先克隆成连续的张量，再调用 _unsafe_view_symint 进行形状变换
    return at::_unsafe_view_symint(self.clone(at::MemoryFormat::Contiguous), shape);
  }
}

// 针对非符号整数类型的张量的重新形状操作的重复实现，保留是为了向后兼容和减少破坏
Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  // 如果张量是稀疏张量，则抛出错误，因为不支持稀疏张量的 reshape 操作
  if (self.is_sparse()) {
    TORCH_CHECK(false, "reshape is not implemented for sparse tensors");
  }
  // 根据给定的 proposed_shape 推断出最终的形状 shape
  DimVector shape = infer_size_dv(proposed_shape, self.numel());

  // 如果张量是 mkldnn 类型，则调用对应的 mkldnn reshape 函数
  if (self.is_mkldnn()) {
    return at::_mkldnn_reshape(self, shape);
  }

  // 计算出正确的步幅 stride，用于确定是否可以直接视图重塑
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);

  // 注意：即使在这里有可视化的几何形状和目标步幅，我们也不直接在 self 上调用 as_strided，
  //     因为 as_strided 的反向传播不如 view 高效，所以我们调用 _reshape_alias，
  //     它本质上与 view 和 as_strided 做的事情相同，但没有额外的开销。
  if (stride.has_value()) {
    // 临时检查，根据设备支持情况，决定使用 _reshape_alias 或者 view
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu()) {
      return self._reshape_alias(shape, stride.value());
    } else {
      return self.view(shape);
    }
  }
  // 如果不能直接视图重塑，则使用不安全的视图函数进行张量的重塑
  return at::_unsafe_view(self.clone(at::MemoryFormat::Contiguous), shape);
}

// 在 reshape 函数中使用的内部私有操作符，用于在可以调用 view 的情况下执行 reshape 操作
Tensor _reshape_alias(const Tensor& self, IntArrayRef sizes, IntArrayRef strides) {
  // 只有在 reshape 函数中会使用，用于替代调用 view，以减少重复工作
  return alias_with_sizes_and_strides(self, sizes, strides);
}

// 将张量按照另一个张量的符号整数形状进行重新形状操作
Tensor reshape_as(const Tensor& self, const Tensor& other) {
  return self.reshape_symint(other.sym_sizes());
}
// 选择稀疏张量的子张量，根据给定的维度和索引进行操作
static Tensor select_sparse(const Tensor& self, int64_t dim, int64_t index) {
  // 获取张量的稀疏维度和密集维度
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  // 断言维度在有效范围内
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < sparse_dim + dense_dim);

  // 获取张量的索引和值
  auto indices = self._indices();
  auto values = self._values();
  // 复制张量的大小，并删除指定维度
  auto new_sizes = self.sizes().vec();
  new_sizes.erase(new_sizes.begin() + dim);

  if (dim < sparse_dim) {
    // 如果维度在稀疏维度内
    // 找到符合条件的非零索引
    auto nzIndices = (indices[dim] == index).nonzero().view(-1);
    // 根据非零索引选择新的值
    auto new_values = values.index_select(0, nzIndices);
    if (sparse_dim == 1) {
      // 如果只有一个稀疏维度，返回密集部分的值
      if (new_values.size(0) == 1) {
        return new_values[0];
      } else {
        // 对值进行求和操作
        return at::sum(new_values, 0, false, new_values.scalar_type());
      }
    } else {
      // 如果有多个稀疏维度，调整索引和值，创建新的稀疏 COO 张量
      auto dimIndices = (arange(
                             0,
                             sparse_dim,
                             c10::nullopt /* dtype */,
                             c10::nullopt /* layout */,
                             self.device(),
                             c10::nullopt /* pin_memory */) != dim)
                            .nonzero()
                            .view(-1);
      auto new_indices = indices.index_select(1, nzIndices).index_select(0, dimIndices);
      return _sparse_coo_tensor_with_dims_and_tensors(
            sparse_dim - 1, dense_dim, new_sizes, new_indices, new_values, self.options());
    }
  } else {
    // 如果维度在密集维度内，选择新的值并创建新的稀疏 COO 张量
    auto new_values = values.select(dim - sparse_dim + 1, index);
    return _sparse_coo_tensor_with_dims_and_tensors(
         sparse_dim, dense_dim - 1, new_sizes, indices, new_values, self.options());
  }
}

// 这是一个辅助函数，由选择和切片方法调用，根据给定输入创建新的量化器
// 如果调用函数是 select()，is_select 为 true
static QuantizerPtr create_subtensor_quantizer(const Tensor& self, bool is_select, int64_t start,
  int64_t end, int64_t dim, int64_t step) {
  // 获取当前量化器
  auto quantizer_prev = get_qtensorimpl(self)->quantizer();
  // 如果量化方案是 PER_TENSOR_AFFINE，则返回原量化器
  if (quantizer_prev->qscheme() == QScheme::PER_TENSOR_AFFINE) {
    return quantizer_prev;
  }
  // 其他情况，创建新的量化器
  QuantizerPtr quantizer;
  auto temp = static_cast<PerChannelAffineQuantizer*>(quantizer_prev.get());
  auto axis = temp->axis();
  auto scales = temp->scales();
  auto zero_points = temp->zero_points();
  if (dim == axis) {
    // 如果维度等于轴，则计算子张量的 scales 和 zero_points
    // *.select(0, start) 可以用 *.slice(0, start, end, step) 替换，但 select() 的开销更小
    scales = is_select ? scales.select(0, start) : scales.slice(0, start, end, step);
    zero_points = is_select ? zero_points.select(0, start) : zero_points.slice(0, start, end, step);
  }
  if (scales.numel() > 1) {
    // 如果 scales 的元素数量大于 1，调整轴的位置，根据是否是 select() 调整 axis
    // slice() 函数不会改变轴的位置，而 select() 会减少张量的维度并保持轴不变
    # 如果条件成立，创建基于通道的仿射量化器；否则创建基于张量的仿射量化器
    if (is_select) {
        # 根据每个通道的量化参数（缩放因子、零点、选择的轴减去1或选择的轴），以及前一个量化器的标量类型，创建仿射量化器
        quantizer = make_per_channel_affine_quantizer(scales, zero_points, (axis - 1), quantizer_prev->scalar_type());
    } else {
        # 根据张量的量化参数（缩放因子和零点），以及前一个量化器的标量类型，创建仿射量化器
        quantizer = make_per_tensor_affine_quantizer(scales.item().to<double>(), zero_points.item().to<int64_t>(), quantizer_prev->scalar_type());
    }
    # 返回创建的仿射量化器
    return quantizer;
// 选择稀疏张量的指定维度的索引值，返回结果张量
Tensor index_select_sparse_cpu(const Tensor& self, int64_t dim, const Tensor& index) {
  /*
    算法：
    1. 检查索引张量的维度和形状是否与输入张量的指定维度匹配
    2. 创建一个与输入张量同类型和设备的零张量 grad_input，其形状与输入张量的大小相同
    3. 调用 select_symint 方法，将指定维度和索引应用于 grad_input，并将 grad 张量的值复制到结果中
    4. 返回结果 grad_input
  */
  auto grad_input = at::zeros_symint(input_sizes, grad.options());
  grad_input.select_symint(dim, std::move(index)).copy_(grad);
  return grad_input;
}
    // index - 一个形状为 (n,) 的一维张量，包含要索引的索引值
    // self - 稀疏张量，其形状为 sizes = sparse_shape + dense_shape
    //   indices - 2-D 张量，形状为 (sparse_dims, nnz)，包含稀疏张量的索引
    //   values - (1+len(dense_shape)) 维张量，形状为 (nnz,) + dense_shape，包含稀疏张量的值
    index_select(dim, index) 返回一个稀疏张量，具有以下数据
      new_sizes = sizes[:dim] + (n,) + sizes[dim+1:]
      new_indices - 形状为 (sparse_dims, new_nnz) 的张量
      new_values - 形状为 (new_nnz,) + dense_shape 的张量

      if dim < len(sparse_shape):
          // 找到输出稀疏张量的 new_indices[dim] 和要选择值/索引的位置
          // CPP 代码使用二进制搜索或计数表来查找匹配项，并可能为了获得更好的算法复杂度而交换循环顺序
          new_dim_indices = []
          selected_dim_indices = []
          // 这是一种暴力算法，传达主要思想
          // 下面的 CPP 代码更有效但更复杂
          for i, i_idx in enumerate(indices[dim]):
              for j, j_idx in enumerate(index):
                  if i_idx == j_idx:
                      new_dim_indices.append(j)
                      selected_dim_indices.append(i)
          new_indices = indices.index_select(1, selected_dim_indices)
          new_values = values.index_select(0, selected_dim_indices)
          new_indices[dim] = new_dim_indices
      else:
          new_indices = indices
          new_values = values.index_select(dim - sparse_dim + 1, index);
    */
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong && index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // Equivalent to t.index_select(dim, idx), but vanilla index_select is not parallel,
  // so we use gather instead.
  // We use this method to select relevant indices/values
  // from the intersection between indices[dim] and the index.
  const auto index_select = [](const Tensor& t, int64_t dim, const Tensor& idx) -> Tensor {
    const auto idx_len = idx.numel();
    auto out_shape = t.sizes().vec();
    out_shape[dim] = idx_len;
    auto idx_shape = std::vector<int64_t>(t.dim(), 1);
    idx_shape[dim] = idx_len;
    return t.gather(dim, idx.view(idx_shape).expand(out_shape));
  };

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    // short-circuit if index is empty
    // 如果 index_len 为零，则进行以下操作
    if (!index_len) {
      // 从 indices 中选择指定维度的索引，形成新的 res_indices
      auto res_indices = index_select(indices, 1, index);
      // 将 res_indices 中指定维度的索引修改为 index
      res_indices[dim] = index;
      // 从 values 中选择对应的值，形成新的 res_values
      const auto res_values = index_select(values, 0, index);

      // 调用 _sparse_coo_tensor_with_dims_and_tensors 函数创建稀疏张量，并返回
      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    }

    // 定义 lambda 函数 nneg_index，用于生成处理后的索引
    const auto nneg_index = [&index, index_len, &self, size, dim]() -> Tensor {
      // 将 index 转换为连续存储的张量
      const auto index_contiguous = index.contiguous();
      // 创建一个与 index_contiguous 相同大小的空张量 nneg_index
      auto nneg_index = at::empty_like(index_contiguous);
      // 获取指向 index_contiguous 和 nneg_index 数据的指针
      auto* ptr_index = index_contiguous.data_ptr<int64_t>();
      auto* ptr_nneg_index = nneg_index.data_ptr<int64_t>();
      // 并行处理 index 的每个元素
      at::parallel_for(0, index_len, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
          const auto* src = ptr_index + start;
          auto* dst = ptr_nneg_index + start;
          // 遍历处理每个索引
          for (C10_UNUSED const auto _ : c10::irange(start, end)) {
            auto idx = *src++;
            // 检查索引是否超出 tensor 的大小范围，如果超出则抛出错误信息
            if (idx < -size || idx >= size) {
               // 如果编译时包含 STRIP_ERROR_MESSAGES，则标记 self 和 dim 为已使用
              (void)dim;
              (void)self;
              TORCH_CHECK_INDEX(false,
                  "index_select(): index contains ", idx, " that is out of range for tensor of size ",
                  self.sizes(), " at dimension ", dim
              );
            }
            // 如果索引为负数，则将其加上 size，转换为非负数索引
            if (idx < 0) {
              idx += size;
            }
            *dst++ = idx; // 将处理后的索引存入 nneg_index
          }
      });

      return nneg_index; // 返回处理后的索引张量
    }();

    // 获取 indices[dim] 的连续存储张量 dim_indices
    const auto dim_indices = indices[dim].contiguous();

    // 如果 nnz 小于 size，则对 indices[dim] 或 index 进行排序，并执行二分查找以找到交集。
    };

    // 将一维排序的索引转换为压缩的一维索引，用于 CSR 格式中的行索引。在并行化和无同步的情况下获取计数表非常有用。
    // TODO: 此函数等同于 _convert_indices_from_coo_to_csr。但该函数尚未公开。
    // 定义了一个 lambda 函数 sorted_idx_to_cidx，用于生成索引到计数索引的映射
    const auto sorted_idx_to_cidx = [](
        const Tensor& idx,
        int64_t len,
        bool run_in_parallel = true) -> Tensor {
      // 创建一个空的张量 cidx，用于存储计数索引
      auto cidx = at::empty({len + 1}, idx.options());

      // 获取输入张量 idx 的指针和 cidx 的指针
      const auto* ptr_idx = idx.const_data_ptr<int64_t>();
      auto* ptr_cidx = cidx.data_ptr<int64_t>();

      // 获取输入张量 idx 的元素个数
      const auto idx_len = idx.numel();

      // 初始化 cidx 中的前部分为 0
      std::fill_n(ptr_cidx, ptr_idx[0] + 1, 0);
      // 初始化 cidx 中的后部分为 idx_len
      std::fill_n(ptr_cidx + ptr_idx[idx_len - 1] + 1, len - ptr_idx[idx_len - 1], idx_len);

      // 确定并行执行的粒度
      const auto grain_size = run_in_parallel ? at::internal::GRAIN_SIZE : idx_len;
      // 并行处理 idx 中的元素，构建计数索引
      at::parallel_for(0, idx_len, grain_size, [&](int64_t start, int64_t end) {
          auto* ptr_curr_cidx = ptr_cidx + ptr_idx[start] + 1;
          for (int64_t i = start; i < std::min(end, idx_len - 1); ++i) {
            const auto diff = ptr_idx[i + 1] - ptr_idx[i];
            std::fill_n(ptr_curr_cidx, diff, i + 1);
            ptr_curr_cidx += diff;
          }
      });

      // 返回生成的计数索引张量 cidx
      return cidx;
    };

    // 如果 nnz（非零元素数）远大于 size，则 indices[dim] 和 index 都被排序
    // 使用计数排序（更快速，且不需要大量的大小为 nnz 的内存分配）
    // 计算两个计数表的元素积，得到所有的交集
    };

    // 定义了一个 lambda 函数 make_output，用于生成输出张量
    const auto make_output = [&](
        const Tensor& selected_dim_indices,
        const Tensor& res_dim_indices) -> Tensor {
      // 根据 selected_dim_indices 从 indices 中选择结果索引
      auto res_indices = index_select(indices, 1, selected_dim_indices);
      // 将 res_dim_indices 存储到 res_indices 的 dim 维度中
      res_indices[dim] = res_dim_indices;
      // 根据 selected_dim_indices 从 values 中选择结果值
      const auto res_values = index_select(values, 0, selected_dim_indices);

      // 使用给定的参数创建一个稀疏 COO 张量，并返回结果
      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    };

    // 对于 nnz 和 index_len 较小的情况，采用蛮力法求解
    const auto get_result_small_nnz_small_index = [&]()
      -> Tensor {
      // 检查条件，确定是否在内循环中处理维度索引
      const auto dim_indices_in_inner_loop = nnz >= index_len;
      // 根据条件选择外部索引和内部索引
      auto [outer, inner] = [&]() -> std::tuple<Tensor, Tensor> {
        if (dim_indices_in_inner_loop) {
          return std::make_tuple(nneg_index, dim_indices);
        }
        else {
          return std::make_tuple(dim_indices, nneg_index);
        }
      }();

      // 获取外部索引和内部索引的指针
      const auto* ptr_outer = outer.const_data_ptr<int64_t>();
      const auto* ptr_inner = inner.const_data_ptr<int64_t>();
      // 如果对性能要求极高，可以替换 std::vector
      // 为在栈上操作的数据结构，设置某些限制。
      // 初始化存储匹配索引的容器
      auto outer_selected_idx = std::vector<int64_t>();
      auto inner_selected_idx = std::vector<int64_t>();
      int64_t res_len = 0;
      // 遍历外部和内部索引，寻找匹配项
      for (const auto i : c10::irange(outer.numel())) {
        for (const auto j : c10::irange(inner.numel())) {
          if (ptr_outer[i] == ptr_inner[j]) {
            ++res_len;
            outer_selected_idx.push_back(i);
            inner_selected_idx.push_back(j);
          }
        }
      }

      // 将匹配的索引转换为 Tensor 类型
      const auto outer_selected_idx_tensor = at::from_blob(
          outer_selected_idx.data(), {res_len}, at::kLong
      );
      const auto inner_selected_idx_tensor = at::from_blob(
          inner_selected_idx.data(), {res_len}, at::kLong
      );

      // 根据条件选择输出结果
      return dim_indices_in_inner_loop
        ? make_output(inner_selected_idx_tensor, outer_selected_idx_tensor)
        : make_output(outer_selected_idx_tensor, inner_selected_idx_tensor);
    };

    // 定义暴力算法大小限制常量
    constexpr int64_t BRUTE_FORCE_SIZE_LIMIT = 2 << 14; // 16384
    // 用于避免 (nnz * index_len) 过大的条件判断
    // 如果条件满足，则调用处理小 nnz 和小 index_len 的结果函数
    if (nnz <= BRUTE_FORCE_SIZE_LIMIT && index_len <= BRUTE_FORCE_SIZE_LIMIT
        && (nnz * index_len) <= BRUTE_FORCE_SIZE_LIMIT) {
      return get_result_small_nnz_small_index();
    }
    else {
      Tensor selected_dim_indices;
      Tensor res_dim_indices;

      // 更精确的决策可能是 `nnz < C(nnz, size) * size` 的形式，
      // 但需要进行大量基准测试。
      // 我们选择 `nnz < size`，它衡量理论复杂性而不依赖于运行时性能。
      // TODO: 进行此分析，并找到更好的 C(nnz, size)。
      // 根据 nnz 的大小选择处理函数
      if (nnz <= size) {
        std::tie(selected_dim_indices, res_dim_indices) = get_selected_indices_small_nnz_large_size();
      }
      else {
        std::tie(selected_dim_indices, res_dim_indices) = get_selected_indices_large_nnz_small_size();
      }

      // 根据选择生成输出结果
      return make_output(selected_dim_indices, res_dim_indices);
    }
  }
  // 如果是对稠密维度进行索引
  else {
    // 对值进行 `index_select` 操作
    const auto res_values = index_select(values, dim - sparse_dim + 1, index);

    // 返回稀疏 COO 张量，带有维度和张量
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
Tensor slice(
    const Tensor& self,  // 输入参数：要切片的张量
    int64_t dim,  // 输入参数：切片的维度
    std::optional<int64_t> start,  // 输入参数（可选）：切片的起始位置
    std::optional<int64_t> end,  // 输入参数（可选）：切片的结束位置
    int64_t step) {  // 输入参数：切片的步长
  int64_t ndim = self.dim();  // 获取张量的维度数
  if (ndim == 0) {  // 如果张量维度为0，则抛出错误
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);  // 处理维度索引，确保在有效范围内
  DimVector sizes(self.sizes().begin(), self.sizes().end());  // 获取张量的尺寸大小
  DimVector strides(self.strides().begin(), self.strides().end());  // 获取张量的步幅大小
  // 处理可选参数
  int64_t start_val = start.has_value() ? start.value() : 0;  // 获取切片起始位置，默认为0
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;  // 获取切片结束位置，默认为INT64_MAX

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");  // 检查步长必须为正数

  if (start_val < 0) {  // 处理负数的起始位置
    start_val += sizes[dim];
  }
  if (end_val < 0) {  // 处理负数的结束位置
    end_val += sizes[dim];
  }
  if (start_val < 0) {  // 确保起始位置不小于0
    start_val = 0;
  } else if (start_val >= sizes[dim]) {  // 确保起始位置不超过张量维度的大小
    start_val = sizes[dim];
  }
  if (end_val < start_val) {  // 确保结束位置不小于起始位置
    end_val = start_val;
  } else if (end_val >= sizes[dim]) {  // 确保结束位置不超过张量维度的大小
    end_val = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start_val * strides[dim];  // 计算存储偏移量
  auto len = end_val - start_val;  // 计算切片的长度
  sizes[dim] = (len + step - 1) / step;  // 根据步长计算新的维度大小（向上取整）
  strides[dim] *= step;  // 更新步幅

  Tensor result;  // 定义返回的张量
  if (self.is_quantized()) {  // 如果输入张量是量化的
    auto quantizer = create_subtensor_quantizer(self, false, start_val, end_val, dim, step);  // 创建量化子张量的量化器
    result = as_strided_qtensorimpl(self, sizes, strides, storage_offset, std::move(quantizer));  // 根据新的尺寸和步幅创建量化的子张量
  } else {
    // NB: it is extremely important to perform a redispatch here for
    // the MPS backend; if you call directly to as_strided_tensorimpl,
    // the necessary metadata for MPS will not get setup and you will
    // get silently wrong results
    result = self.as_strided(sizes, strides, storage_offset);  // 根据新的尺寸和步幅创建子张量
  }
  namedinference::propagate_names(result, self);  // 根据原张量的命名信息推断子张量的命名信息
  return result;  // 返回切片后的结果张量
}

Tensor slice_inverse_symint(
    const Tensor& self,  // 输入参数：要切片的张量
    const Tensor& base,  // 输入参数：基本张量，用于提供元数据
    int64_t /* dim */,  // 输入参数：切片的维度（此参数未使用）
    std::optional<SymInt> /* start */,  // 输入参数（可选）：切片的起始位置（此参数未使用）
    std::optional<SymInt> /* end */,  // 输入参数（可选）：切片的结束位置（此参数未使用）
    SymInt /* step */) {  // 输入参数：切片的步长（此参数未使用）
  // assume self has enough to storage to be viewed with base's metadata
  return self.as_strided_symint(base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());  // 返回使用基本张量元数据的符号整数切片后的张量
}

Tensor slice_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());  // 根据输入尺寸和梯度的选项创建零张量
  grad_input.slice(dim, start, end, step).copy_(grad);  // 对梯度进行切片操作，并复制到新创建的零张量中
  return grad_input;  // 返回包含梯度切片的结果张量
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  const auto num_splits = get_num_splits(self, split_size, dim);  // 计算在给定维度上分割的次数
  std::vector<Tensor> splits(num_splits);  // 创建用于存储分割结果的张量向量
  int64_t last_split_size = split_size - (split_size * num_splits - self.size(dim));  // 计算最后一个分割的大小

  for (const auto i : c10::irange(num_splits)) {  // 遍历每个分割
    auto length = i < num_splits - 1 ? split_size : last_split_size;  // 确定当前分割的长度
    splits[i] = self.narrow(dim, i * split_size, length);  // 对输入张量进行狭窄操作，以获取当前分割
  }
  return splits;  // 返回分割后的张量向量
}
// 使用给定大小和维度对输入张量进行符号整数大小的分割，并返回分割后的张量向量
std::vector<Tensor> split_symint(const Tensor& self, c10::SymIntArrayRef sizes, int64_t dim) {
  // 调用ATen库函数，使用指定的大小和维度进行符号整数大小的分割，并返回结果向量
  return at::split_with_sizes_symint(self, sizes, dim);
}

// 使用给定大小和维度对输入张量进行非安全分割，并可能设置版本计数器
std::vector<Tensor> unsafe_split(const Tensor& self, int64_t split_size, int64_t dim) {
  // 调用ATen库函数进行非安全分割
  auto result = at::native::split(self, split_size, dim);
  // 遍历分割后的张量向量，检查每个张量是否需要设置版本计数器
  for (auto& t : result) {
    // 如果张量不是推断状态，则设置其版本计数器为指定的值
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(/*version=*/0));
    }
  }
  // 返回非安全分割后的结果张量向量
  return result;
}

// 使用给定大小对输入张量进行水平分割
std::vector<Tensor> hsplit(const Tensor& self, int64_t split_size) {
  // 检查输入张量的维度是否至少为1，否则抛出错误
  TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 确定分割的维度是0还是1
  int64_t dim = (self.dim() == 1) ? 0 : 1;
  // 检查分割大小是否合法：非零且指定维度的符号大小可被分割大小整除
  TORCH_CHECK(split_size != 0 && self.sym_sizes()[dim] % split_size == 0,
    "torch.hsplit attempted to split along dimension ", dim,", but the size of the dimension ", self.sizes()[dim], " is not divisible by the split_size ", split_size, "!");
  // 调用ATen库函数进行水平分割，并返回结果张量向量
  return at::tensor_split(self, split_size, dim);
}

// 使用给定大小对输入张量进行垂直分割
std::vector<Tensor> vsplit(const Tensor& self, int64_t split_size) {
  // 检查输入张量的维度是否至少为2，否则抛出错误
  TORCH_CHECK(self.dim() >= 2, "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 检查分割大小是否合法：非零且第一维度的符号大小可被分割大小整除
  TORCH_CHECK(split_size != 0 && self.sym_sizes()[0] % split_size == 0,
    "torch.vsplit attempted to split along dimension ", 0,", but the size of the dimension ", self.sizes()[0], " is not divisible by the split_size ", split_size, "!");
  // 调用ATen库函数进行垂直分割，并返回结果张量向量
  return at::tensor_split(self, split_size, 0);
}

// 使用给定大小对输入张量进行深度分割
std::vector<Tensor> dsplit(const Tensor& self, int64_t split_size) {
  // 检查输入张量的维度是否至少为3，否则抛出错误
  TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 检查分割大小是否合法：非零且第三维度的符号大小可被分割大小整除
  TORCH_CHECK(split_size != 0 && self.sym_sizes()[2] % split_size == 0,
    "torch.dsplit attempted to split along dimension ", 2,", but the size of the dimension ", self.sizes()[2], " is not divisible by the split_size ", split_size, "!");
  // 调用ATen库函数进行深度分割，并返回结果张量向量
  return at::tensor_split(self, split_size, 2);
}

// 使用给定大小和维度对输入张量进行分割
std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  // 检查输入张量是否至少为1维度，否则抛出错误
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  // 获取指定维度的大小
  const int64_t dim_size = self.size(dim);
  // 获取分割大小的数量
  const int64_t num_splits = split_sizes.size();
  // 初始化起始索引
  int64_t start_idx = 0;

  // 创建空的张量向量用于存储分割后的结果
  std::vector<Tensor> splits;
  splits.reserve(num_splits);
  // 遍历每个分割大小
  for (const auto i : c10::irange(num_splits)) {
    auto length = split_sizes[i];
    // 检查分割大小是否非负
    TORCH_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    // 调用ATen库函数进行分割，将分割后的张量加入结果向量
    splits.push_back(at::native::slice(self, dim, start_idx, start_idx + length, 1));
    // 更新起始索引
    start_idx += length;
  }
  // 返回分割后的结果张量向量
  return splits;
}
    # 更新起始索引，使其跳过当前分割块的长度
    start_idx += length;
  }
  # 使用 TORCH_CHECK 确保所有分割大小的总和等于维度上的原始大小
  TORCH_CHECK(start_idx == dim_size,
           "split_with_sizes expects split_sizes to sum exactly to ", dim_size,
           " (input tensor's size at dimension ", dim, "), ", "but got split_sizes=", split_sizes);
  # 返回分割后的张量列表
  return splits;
}

// 按照给定的大小列表在指定维度上切割张量，并设置版本计数器（如果需要）
std::vector<Tensor> unsafe_split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  // 调用 ATen 中的 split_with_sizes 函数进行张量分割
  auto result = at::native::split_with_sizes(self, split_sizes, dim);
  // 遍历分割后的每个张量
  for (auto& t : result) {
    // 如果张量不是推理张量，则设置其版本计数器
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(/*version=*/0));
    }
  }
  // 返回分割后的张量列表
  return result;
}

// 水平分割（按列）张量
std::vector<Tensor> hsplit(const Tensor& self, IntArrayRef split_sizes) {
  // 检查张量的维度是否至少为 1
  TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 调用 ATen 中的 tensor_split 函数进行水平分割
  return at::tensor_split(self, split_sizes, (self.dim() == 1) ? 0 : 1);
}

// 垂直分割（按行）张量
std::vector<Tensor> vsplit(const Tensor& self, IntArrayRef split_sizes) {
  // 检查张量的维度是否至少为 2
  TORCH_CHECK(self.dim() >= 2, "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 调用 ATen 中的 tensor_split 函数进行垂直分割
  return at::tensor_split(self, split_sizes, 0);
}

// 深度分割（按深度）张量
std::vector<Tensor> dsplit(const Tensor& self, IntArrayRef split_sizes) {
  // 检查张量的维度是否至少为 3
  TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
  // 调用 ATen 中的 tensor_split 函数进行深度分割
  return at::tensor_split(self, split_sizes, 2);
}

// 前提条件：tensors 非空
static inline std::vector<Tensor> get_stack_inputs(TensorList tensors, int64_t dim) {
  // 创建一个与输入张量列表大小相同的输入张量列表
  std::vector<Tensor> inputs(tensors.size());
  // 获取第一个张量的形状作为参考形状
  at::IntArrayRef entry_shape = tensors[0].sizes();
  // 在指定维度上对第一个张量进行unsqueeze操作，并加入到输入列表中
  inputs[0] = tensors[0].unsqueeze(dim);
  // 遍历剩余张量，检查其形状是否与第一个张量相同，并进行unsqueeze操作，加入到输入列表中
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(tensors[i].sizes() == entry_shape,
      "stack expects each tensor to be equal size, but got ", entry_shape,
      " at entry 0 and ", tensors[i].sizes(), " at entry ", i);
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  // 返回处理后的输入张量列表
  return inputs;
}

// 可能使用本地序列化堆叠张量
bool inline maybe_native_stack(Tensor& result, TensorList tensors, int64_t dim) {
  // 调整 dim 以确保在有效范围内
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  // 检查是否可以使用本地序列化堆叠操作
  if (detail::CanUseNativeSerialStack<TensorList, /*skip_overlap_check*/ false>::call(result, tensors, dim)) {
    // 计算结果张量的大小
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + dim, tensors.size());

    // 如果结果张量的大小与预期不同，则重新调整大小
    if (result.sizes() != result_sizes) {
      result.resize_(result_sizes);
    }

    // 调用本地序列化堆叠的具体实现
    stack_serial_stub(kCPU, result, tensors, dim);
    return true;
  }
  return false;
}

// 对张量列表进行堆叠操作
Tensor _stack(TensorList tensors, int64_t dim) {
  // 确定结果张量的高级类型
  ScalarType high_type = result_type(tensors);
  // 创建一个空张量作为结果张量
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  // 调用 ATen 中的 _stack_out 函数进行堆叠操作
  return at::native::_stack_out(get_stack_inputs(tensors, dim), dim, result);
}
// 将给定张量列表沿指定维度堆叠，返回结果张量
Tensor _stack_cpu(TensorList tensors, int64_t dim) {
  // 确定张量列表中张量的高级数据类型
  ScalarType high_type = result_type(tensors);
  // 创建一个空张量作为结果，使用与第一个张量相同的数据类型
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  // 调用 _stack_out_cpu 函数执行实际的堆叠操作，并将结果返回
  return at::native::_stack_out_cpu(tensors, dim, result);
}

// 检查堆叠操作的输入张量列表是否满足条件
static void check_stack_inputs(TensorList tensors, int64_t dim) {
  // 获取第一个张量的形状作为参考形状
  at::IntArrayRef entry_shape = tensors[0].sizes();
  // 遍历张量列表中的每个张量，确保它们的形状与第一个张量相同
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(tensors[i].sizes() == entry_shape,
      "stack expects each tensor to be equal size, but got ", entry_shape,
      " at entry 0 and ", tensors[i].sizes(), " at entry ", i);
  }
}

// 在指定维度上对每个张量进行填充，使得填充后的维度能整除 num_chunks
static std::vector<Tensor> _pad_chunk(TensorList tensors, int64_t dim, int64_t num_chunks) {
  auto num_tensors = tensors.size();
  std::vector<Tensor> padded_tensors;
  padded_tensors.reserve(num_tensors);
  // 遍历每个张量进行填充
  for (const auto & tensor : tensors) {
    auto tensor_size = tensor.sizes();
    // 计算填充后的维度大小
    std::vector<int64_t> padded_size(tensor_size.vec());
    padded_size[dim] = (tensor_size[dim] + num_chunks - 1) / num_chunks * num_chunks;
    Tensor padded_tensor = tensor;
    // 如果填充后的维度与原始维度不同，则进行填充操作
    if (padded_size != tensor_size) {
      padded_tensor = tensor.new_zeros(padded_size);
      padded_tensor.narrow(dim, 0, tensor_size[dim]).copy_(tensor);
    }
    // 将填充后的张量进行视图重塑，并加入到结果向量中
    std::vector<int64_t> view_sizes(tensor_size.begin(), tensor_size.begin()+dim);
    view_sizes.insert(view_sizes.end(), {num_chunks, -1});
    padded_tensors.push_back(padded_tensor.view(view_sizes));
  }
  return padded_tensors;
}

// 对张量列表进行分块连接操作，返回连接后的张量
Tensor _chunk_cat(TensorList tensors, int64_t dim, int64_t num_chunks) {
  // 对输入进行预处理，确保能正确进行分块连接
  auto wrapped_dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  // 执行分块连接，并在连接后的维度上进行填充操作
  return at::cat(_pad_chunk(tensors, wrapped_dim, num_chunks), wrapped_dim+1);
}

// 对张量列表进行分块连接操作，将结果存储到预分配的输出张量中
Tensor& _chunk_cat_out(TensorList tensors, int64_t dim, int64_t num_chunks, Tensor& out) {
  // 对输入进行预处理，确保能正确进行分块连接
  auto wrapped_dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  // 将分块连接后的结果存储到预分配的输出张量中，并在连接后的维度上进行填充操作
  at::cat_out(out, _pad_chunk(tensors, wrapped_dim, num_chunks), wrapped_dim+1);
  return out;
}

// 对张量列表进行堆叠操作，返回堆叠后的结果张量
Tensor stack(TensorList tensors, int64_t dim) {
  // 检查输入张量列表是否为空
  TORCH_CHECK(!tensors.empty(),
           "stack expects a non-empty TensorList");
  // 确定堆叠操作的维度，并进行必要的输入检查
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension()+1);
  // 如果堆叠维度小于第一个张量的维度数，并且第一个张量不是稀疏张量
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    // 检查每个输入张量的形状是否一致
    check_stack_inputs(tensors, wrapped_dim);
    // 计算堆叠后的结果张量的形状
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    // 执行堆叠操作，并通过视图重塑结果张量的形状
    auto out = at::cat(tensors, wrapped_dim);
    return out.view(result_sizes); // one can always split a dimension with view
  } else { // 如果堆叠维度等于第一个张量的维度数，或者第一个张量是稀疏张量
    // 调用辅助函数获取适合堆叠的输入，并执行堆叠操作
    return at::cat(get_stack_inputs(tensors, dim), dim);
  }
}

// CPU 特定的实现，对输入张量列表进行堆叠操作，并将结果存储到预分配的输出张量中
Tensor& _stack_out_cpu(TensorList tensors, int64_t dim, Tensor& result) {
  // 尝试在 CPU 上使用原生的堆叠实现
  if (maybe_native_stack(result, tensors, dim)) {
    return result;
  } else {
    return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
  }



// 如果条件满足，则返回 result 变量的值
    return result;
  // 如果条件不满足，则调用 at::cat_out 函数，将 result 和通过 get_stack_inputs 函数获取的输入张量进行连接，并沿指定维度进行拼接
  } else {
    return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
  }
}

// 默认的后端函数，将给定的张量列表沿指定维度堆叠到结果张量中
Tensor& _stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  // 调用 ATen 库的 cat_out 函数，将张量列表 tensors 沿维度 dim 进行连接，并将结果存入 result
  return at::cat_out(result, tensors, dim);
}

// TODO：(msubkhankulov) 重构以使用 _stack_out
// 将给定的张量列表沿指定维度堆叠到结果张量中
Tensor& stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "stack expects a non-empty TensorList");
  // 可能会对维度进行包装
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension()+1);
  // 如果包装后的维度小于第一个张量的维度并且第一个张量不是稀疏张量
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    // 检查堆叠输入的有效性
    check_stack_inputs(tensors, wrapped_dim);
    // 调整输出张量的尺寸
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    at::native::resize_output(result, result_sizes);
    // 计算用于 cat 操作的尺寸
    auto cat_sizes = tensors[0].sizes().vec();
    cat_sizes[wrapped_dim] *= tensors.size();
    // 计算张量的步幅
    auto strides = at::detail::computeStride(result.sizes(), result.strides(), cat_sizes);
    // 如果存在步幅值
    if (strides.has_value()) {
      // 可以使用快速 cat 路径
      // 创建结果视图并进行 cat 操作
      auto result_view = result.view(cat_sizes);
      at::cat_out(result_view, tensors, wrapped_dim);
      // 返回结果张量
      return result;
    }
  }
  // 否则调用 cat_out 进行堆叠操作
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

// 将张量列表沿第0维度（水平方向）堆叠为一个张量
Tensor hstack(TensorList tensors) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "hstack expects a non-empty TensorList");
  // 至少将所有张量转为1维张量
  auto rep = at::atleast_1d(tensors);
  // 如果第一个张量的维度为1，则沿第0维度堆叠
  if (rep[0].dim() == 1) {
    return at::cat(rep, 0);
  }
  // 否则沿第1维度堆叠
  return at::cat(rep, 1);
}

// 将张量列表沿第0维度（水平方向）堆叠到给定结果张量中
Tensor& hstack_out(TensorList tensors, Tensor& result) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "hstack expects a non-empty TensorList");
  // 至少将所有张量转为1维张量
  auto rep = at::atleast_1d(tensors);
  // 如果第一个张量的维度为1，则沿第0维度堆叠到结果张量中
  if (rep[0].dim() == 1) {
    return at::cat_out(result, rep, 0);
  }
  // 否则沿第1维度堆叠到结果张量中
  return at::cat_out(result, rep, 1);
}

// 将张量列表沿第0维度（垂直方向）堆叠为一个张量
Tensor vstack(TensorList tensors) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "vstack expects a non-empty TensorList");
  // 至少将所有张量转为2维张量
  auto rep = at::atleast_2d(tensors);
  // 沿第0维度堆叠
  return at::cat(rep, 0);
}

// 将张量列表沿第0维度（垂直方向）堆叠到给定结果张量中
Tensor& vstack_out(TensorList tensors, Tensor& result) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "vstack expects a non-empty TensorList");
  // 至少将所有张量转为2维张量
  auto rep = at::atleast_2d(tensors);
  // 沿第0维度堆叠到结果张量中
  return at::cat_out(result, rep, 0);
}

// 将张量列表沿第2维度（深度方向）堆叠为一个张量
Tensor dstack(TensorList tensors) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "dstack expects a non-empty TensorList");
  // 至少将所有张量转为3维张量
  auto rep = at::atleast_3d(tensors);
  // 沿第2维度堆叠
  return at::cat(rep, 2);
}

// 将张量列表沿第2维度（深度方向）堆叠到给定结果张量中
Tensor& dstack_out(TensorList tensors, Tensor& result) {
  // 检查张量列表不能为空
  TORCH_CHECK(!tensors.empty(),
           "dstack expects a non-empty TensorList");
  // 至少将所有张量转为3维张量
  auto rep = at::atleast_3d(tensors);
  // 沿第2维度堆叠到结果张量中
  return at::cat_out(result, rep, 2);
}

// 稀疏张量的转置操作
static inline Tensor & sparse_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  // 获取稀疏张量的稀疏维度数
  int64_t nsparse_dim = self.sparse_dim();
  // 检查转置的维度必须是稀疏维度
  TORCH_CHECK(dim0 < nsparse_dim && dim1 < nsparse_dim,
           "sparse transpose: transposed dimensions must be sparse ",
           "Got sparse_dim: ", nsparse_dim, ", d0: ", dim0, ", d1: ", dim1);

  // 如果稀疏张量的索引和数值均为空
  if (self._indices().numel() == 0 && self._values().numel() == 0) {
    // 交换指定维度的尺寸
    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);
    // ...
    // 如果稀疏张量的维度数量大于0
    if (at::sparse::get_sparse_impl(self)->nnz() > 0) {
        // 获取稀疏张量的尺寸
        auto sizes = self.sizes().vec();
        // 交换给定的两个维度的尺寸大小
        std::swap(sizes[dim0], sizes[dim1]);
        
        // 调整稀疏张量的内部表示以适应新的尺寸
        at::sparse::get_sparse_impl(self)->raw_resize_(self._indices().size(0), self._values().dim() - 1, sizes);
    } else {
        // 否则，获取稀疏张量的索引
        auto indices = self._indices();
        // 选择索引的第一个维度中的 dim0 和 dim1 行
        auto row0 = indices.select(0, dim0);
        auto row1 = indices.select(0, dim1);

        // 交换 row0 和 row1 的内容
        auto tmp = at::zeros_like(row0, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        tmp.copy_(row0);
        row0.copy_(row1);
        row1.copy_(tmp);

        // 标记稀疏张量未压缩状态
        self._coalesced_(false);

        // 获取稀疏张量的尺寸
        auto sizes = self.sizes().vec();
        // 交换给定的两个维度的尺寸大小
        std::swap(sizes[dim0], sizes[dim1]);

        // 调整稀疏张量的内部表示以适应新的尺寸
        at::sparse::get_sparse_impl(self)->raw_resize_(self._indices().size(0), self._values().dim() - 1, sizes);
    }
    // 返回更新后的稀疏张量对象
    return self;
}

// 结束函数定义

// torch.row_stack, torch.vstack 的别名
Tensor& row_stack_out(TensorList tensors, Tensor& result) {
  // 调用 torch.vstack_out 函数
  return at::vstack_out(result, tensors);
}

// torch.row_stack, torch.vstack 的别名
Tensor row_stack(TensorList tensors) {
  // 调用 torch.vstack 函数
  return at::vstack(tensors);
}

// 重新整形输入以用于 column_stack 函数
static std::vector<Tensor> reshape_input_for_column_stack(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  // 定义 lambda 函数 transform_lambda
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    // 将 0 维或 1 维的张量重塑为 (t.numel(), 1)
    if (input.dim() <= 1) {
      return input.reshape_symint({input.sym_numel(), 1});
    }
    return input;
  };
  // 对输入张量列表应用 lambda 函数
  std::transform(tensors.cbegin(),
                 tensors.cend(),
                 result.begin(),
                 transform_lambda);
  return result;
}

// column_stack 函数的输出版本
Tensor& column_stack_out(TensorList tensors, Tensor& result) {
  // 检查张量列表不为空
  TORCH_CHECK(!tensors.empty(),
              "column_stack expects a non-empty TensorList");

  // 重新整形输入张量以准备进行 column_stack
  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  // 调用 torch.hstack_out 函数
  return at::hstack_out(result, reshaped_tensors);
}

// column_stack 函数
Tensor column_stack(TensorList tensors) {
  // 检查张量列表不为空
  TORCH_CHECK(!tensors.empty(),
              "column_stack expects a non-empty TensorList");

  // 重新整形输入张量以准备进行 column_stack
  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  // 调用 torch.hstack 函数
  return at::hstack(reshaped_tensors);
}

// 传播转置后的名称
static Tensor& propagate_transposed_names(
    Tensor& result,
    const Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  // 如果 other 张量有命名，则交换 dim0 和 dim1 的名称
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    // 根据非空名称传播命名到 result 张量
    namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

// transpose 函数的实现
Tensor transpose(const Tensor& self, Dimname dim0, Dimname dim1) {
  // 调用 torch.transpose 函数
  return at::transpose(
      self, dimname_to_position(self, dim0), dimname_to_position(self, dim1));
}

// 原地 transpose 函数的实现
Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  // 检查稀疏张量类型，不支持原地转置操作
  TORCH_CHECK(
      !(self.layout() == kSparseCsr || self.layout() == kSparseCsc ||
        self.layout() == kSparseBsr || self.layout() == kSparseBsc),
      "torch.transpose_: in-place transposition is not supported for ",
      self.layout(),
      " layout");

  auto ndims = self.dim();
  // 包裹 dim0 和 dim1 以适应张量维度
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  // 对于稀疏 COO 格式，支持原地转置
  if (self.is_sparse()) {
    return sparse_transpose_(self, dim0, dim1);
  }

  // 对于 MKL-DNN 张量，处理特殊情况
    # 调用ATen库中的_mkldnn_transpose_函数，执行张量的转置操作，并返回结果
    return at::_mkldnn_transpose_(self, dim0, dim1);
  }

  # 获取张量的尺寸大小并存储在sizes中
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  # 获取张量的步长信息并存储在strides中
  DimVector strides(self.strides().begin(), self.strides().end());
  # 交换指定维度dim0和dim1的步长信息
  std::swap(strides[dim0], strides[dim1]);
  # 交换指定维度dim0和dim1的尺寸大小信息
  std::swap(sizes[dim0], sizes[dim1]);
  # 使用新的尺寸和步长信息对当前张量进行重构
  self.as_strided_(sizes, strides);
  # 返回重构后的张量
  return self;
}



namespace {
// Transpose implementation for sparse compressed layouts
// NB: We assume that dim1,dim0 have already been wrapped
static inline Tensor sparse_compressed_transpose(
    const Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  auto compressed_inds = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "compressed_inds",
      [&self]() { return self.crow_indices(); },
      [&self]() { return self.ccol_indices(); });

  auto plain_inds = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "plain_inds",
      [&self]() { return self.col_indices(); },
      [&self]() { return self.row_indices(); });

  const auto n_batch_dim = compressed_inds.dim() - 1;
  const auto dense_dim = self.dim() - n_batch_dim - 2;

  // In theory it works, but missing to_dense coverage to test
  TORCH_CHECK(
      dense_dim == 0,
      "transpose(): hybrid sparse compressed tensors with dense dimensions are not supported");

  // Classify transpose "type"
  enum class TransposeDim : uint8_t { Batch, Sparse, Dense };
  auto classify_dim = [&n_batch_dim](const int64_t dim) {
    if (dim < n_batch_dim) {
      return TransposeDim::Batch;
    } else if (dim > n_batch_dim + 1) {
      return TransposeDim::Dense;
    } else {
      return TransposeDim::Sparse;
    }
  };

  const auto transpose_type = classify_dim(dim0);

  // Validate that dimensions are of the same type for transposition
  {
    auto dim_type_name = [](const TransposeDim dim) {
      switch (dim) {
        case TransposeDim::Batch:
          return "Batch";
        case TransposeDim::Dense:
          return "Dense";
        case TransposeDim::Sparse:
          return "Sparse";
        default:
          TORCH_INTERNAL_ASSERT(
              false,
              "Impossible TransposeDim value: ",
              static_cast<std::underlying_type_t<TransposeDim>>(dim));
      }
    };
    const auto dim1_type = classify_dim(dim1);
    TORCH_CHECK(
        dim1_type == transpose_type,
        "transpose(): can only transpose dimensions of the same type (Batch, Sparse, Dense), got ",
        dim0,
        "(",
        dim_type_name(transpose_type),
        ")",
        " and ",
        dim1,
        "(",
        dim_type_name(dim1_type),
        ")");
  }

  // We have validated everything, early exit for equal dims (no effect)
  if (dim0 == dim1) {
    return self.clone();
  }

  auto result_sizes = DimVector(self.sizes());
  std::swap(result_sizes[dim0], result_sizes[dim1]);
  Tensor result_vals;
  auto result_layout = self.layout();

  // Perform transposition based on transpose type
  if (transpose_type == TransposeDim::Batch) {
    compressed_inds = compressed_inds.transpose(dim0, dim1).contiguous();
    plain_inds = plain_inds.transpose(dim0, dim1).contiguous();
    result_vals = self.values().transpose(dim0, dim1).contiguous();

  } else if (transpose_type == TransposeDim::Dense) {
    // NB: This code should work, but is untestable due to lack of support for
    // dense dimensions in to_dense. The Debug assert is present to emphasize
    // 使用 TORCH_INTERNAL_ASSERT 确保不会执行到此代码块
    TORCH_INTERNAL_ASSERT(
        false, "transpose(): Shouldn't have reached this point");
    // 根据当前的稀疏张量布局调用 AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS 宏，
    // 用于执行稀疏转置操作
    result_vals = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "sparse_transpose",
        // 对于未阻塞的情况：两个稀疏维度映射到单个非零元素维度，因此密集维度 dim0/1 向左移动一个位置
        [&]() { return self.values().transpose(dim0 - 1, dim1 - 1); },
        // 对于阻塞的情况：两个稀疏维度映射到三个（非零元素, ）+ 块大小维度，因此密集维度 dim0/1 向右移动一个位置
        [&]() { return self.values().transpose(dim0 + 1, dim1 + 1); });
  } else /*if (transpose_type == TransposeDim::Sparse) */ {
    // 翻转布局
    result_layout = sparse_csr::flip_compressed_layout(self.layout());
    // 根据当前的稀疏张量布局调用 AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS 宏，
    // 执行稀疏转置操作
    result_vals = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "sparse_transpose",
        // 对于未阻塞的情况：值没有变化，布局被翻转
        [&]() { return self.values(); },
        // 对于阻塞的情况：块嵌套在稀疏维度下，因此它们也必须进行转置
        [&]() {
          return self.values().transpose(-2 - dense_dim, -1 - dense_dim);
        });
  }
  // 返回不安全的稀疏压缩张量，包括压缩的指标、普通的指标、处理后的值、结果大小和可能的布局
  return at::_sparse_compressed_tensor_unsafe(
      compressed_inds,
      plain_inds,
      result_vals,
      result_sizes,
      self.options().layout(result_layout));
static void check_t(const Tensor& self, const char *fn) {
  // 检查稀疏张量的维度限制
  if (self.is_sparse()) {
    int64_t sparse_dim = self.sparse_dim();
    int64_t dense_dim = self.dense_dim();
    // 使用 TORCH_CHECK 确保稀疏维度不超过2且稠密维度为0
    TORCH_CHECK(sparse_dim <= 2 && dense_dim == 0,
             fn, " expects a tensor with <= 2 sparse and 0 dense dimensions, but got ",
             sparse_dim, " sparse and ", dense_dim, " dense dimensions");
  } else {
    // 使用 TORCH_CHECK 确保张量维度不超过2
    TORCH_CHECK(self.dim() <= 2,
             fn, " expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");
  }
}

Tensor t(const Tensor & self) {
  // 检查输入张量是否符合转置操作的要求
  check_t(self, "t()");
  // 返回输入张量的转置结果，如果维度小于2，则转置(0,0)，否则转置(0,1)
  return self.transpose(0, self.dim() < 2 ? 0 : 1);
}

Tensor & t_(Tensor & self) {
  // 检查输入张量是否符合原地转置操作的要求
  check_t(self, "t_()");
  // 原地转置输入张量，如果维度小于2，则转置(0,0)，否则转置(0,1)
  return self.transpose_(0, self.dim() < 2 ? 0 : 1);
}

std::tuple<SymDimVector, SymDimVector>
static inferSqueezeGeometry(const Tensor &tensor) {
  // 推断张量压缩后的几何特性：尺寸和步长
  SymDimVector sizes;
  SymDimVector strides;

  // 遍历张量的维度
  for(const auto d : c10::irange(tensor.dim())) {
    // 如果当前维度的尺寸不为1，则添加到尺寸和步长向量中
    if(tensor.sym_sizes()[d] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }

  // 返回尺寸和步长的元组
  return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<SymDimVector, SymDimVector>
static inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  // 推断在指定维度上压缩后的张量几何特性：尺寸和步长
  SymDimVector sizes;
  SymDimVector strides;

  // 遍历张量的维度
  for(const auto d : c10::irange(tensor.dim())) {
    // 如果当前维度不是指定的维度或者指定维度的尺寸不为1，则添加到尺寸和步长向量中
    if(d != dim || tensor.sym_sizes()[dim] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }

  // 返回尺寸和步长的元组
  return std::make_tuple(std::move(sizes), std::move(strides));
}
// 推断压缩后张量的几何特性，返回压缩后尺寸和步长
static inferSqueezeGeometry(const Tensor &tensor, std::bitset<dim_bitset_size> dim_mask) {
  // 获取张量的维度
  const auto ndim = tensor.dim();
  // 获取张量的符号大小
  const auto sym_sizes = tensor.sym_sizes();
  // 获取张量的符号步长
  const auto sym_strides = tensor.sym_strides();

  // 存储压缩后的尺寸和步长
  SymDimVector out_sizes, out_strides;
  // 遍历张量的每一个维度
  for (const auto d: c10::irange(ndim)) {
    // 如果当前维度不在压缩掩码中或者对应的符号大小不为1
    if (!dim_mask.test(d) || sym_sizes[d] != 1) {
      // 将符号大小和符号步长添加到输出的尺寸和步长中
      out_sizes.push_back(sym_sizes[d]);
      out_strides.push_back(sym_strides[d]);
    }
  }
  // 返回压缩后的尺寸和步长作为元组
  return std::make_tuple(std::move(out_sizes), std::move(out_strides));
}

namespace {
// 命名类型，而不是使用pair/tuple，以确保在此处构造向量并获得NRVO（具名返回值优化）
template <typename T>
struct InferUnsqueezeGeometryResult {
  // 存储尺寸和步长的小型向量，使用静态大小kDimVectorStaticSize
  SmallVector<T, kDimVectorStaticSize> sizes;
  SmallVector<T, kDimVectorStaticSize> strides;
  // 构造函数，从张量大小和步长的数组引用初始化sizes和strides
  InferUnsqueezeGeometryResult(ArrayRef<T> tensor_sizes, ArrayRef<T> tensor_strides)
      : sizes(tensor_sizes.begin(), tensor_sizes.end())
      , strides(tensor_strides.begin(), tensor_strides.end()) {}
};

// 推断未压缩后的符号整数几何特性
InferUnsqueezeGeometryResult<c10::SymInt>
inferUnsqueezeGeometry_symint(const Tensor& tensor, int64_t dim) {
  // 创建结果对象，使用张量的符号大小和步长初始化
  InferUnsqueezeGeometryResult<c10::SymInt> result(tensor.sym_sizes(), tensor.sym_strides());
  // 计算新的步长
  c10::SymInt new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  // 在指定维度插入大小为1的维度和新的步长
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  // 返回更新后的结果对象
  return result;
}

// 推断未压缩后的整数几何特性
InferUnsqueezeGeometryResult<int64_t>
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  // 创建结果对象，使用张量的大小和步长初始化
  InferUnsqueezeGeometryResult<int64_t> result(tensor.sizes(), tensor.strides());
  // 计算新的步长
  int64_t new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  // 在指定维度插入大小为1的维度和新的步长
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  // 返回更新后的结果对象
  return result;
}

// 对量化张量进行压缩操作
Tensor squeeze_qtensor(const Tensor& self, c10::OptionalIntArrayRef dims) {
  // 获取张量的量化器
  auto quantizer = get_qtensorimpl(self)->quantizer();
  // 获取张量的维度
  const auto ndim = self.dim();
  // 如果存在压缩维度，则将其转换为位掩码，否则使用所有维度
  auto mask = dims.has_value()
      ? dim_list_to_bitset(dims, self.dim())
      : std::bitset<dim_bitset_size>((1ull << self.dim()) - 1);
  // 推断压缩后的尺寸和步长
  auto [sizes, strides] = inferSqueezeGeometry(self, mask);

  // 如果量化方案为每通道仿射量化
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    // 强制转换为PerChannelAffineQuantizer
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    // 获取量化的轴
    auto axis = per_channel_quantizer->axis();
    // 初始化偏移量为0
    int64_t shift = 0;
    // 遍历张量的每一个维度
    for (const auto d : c10::irange(ndim)) {
      // 如果在掩码中且该维度的大小为1
      if (mask.test(d) && self.sizes()[d] == 1) {
        // 检查压缩只能在非轴维度进行，对于每通道量化的张量
        TORCH_CHECK(axis != d, "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
        // 如果当前维度小于量化轴，则增加偏移量
        if (d < axis) {
          ++shift;
        }
      }
    }
    // 调整量化轴
    axis -= shift;
  quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                per_channel_quantizer->zero_points(),
                                                axis,
                                                quantizer->scalar_type());

// 使用 `per_channel_quantizer` 的 `scales()` 和 `zero_points()` 方法创建一个按通道的仿射量化器，并指定轴和量化器的标量类型。


// TODO: quantized Tensor support for SymInt needs to be added but basic building blocs
// are missing for now.

// TODO: 需要添加对 SymInt 的量化张量支持，但目前缺少基本的构建模块。


  auto result = make_qtensor(self, C10_AS_INTARRAYREF_SLOW(sizes), C10_AS_INTARRAYREF_SLOW(strides), std::move(quantizer));

// 使用 `make_qtensor` 函数创建一个量化张量 `result`，使用 `self` 作为原始张量，`sizes` 和 `strides` 作为尺寸和步幅，移动之前创建的量化器 `quantizer`。


  auto maybe_outnames = namedinference::compute_squeeze_outnames(self, mask);

// 使用 `namedinference::compute_squeeze_outnames` 函数计算 `self` 张量在应用掩码 `mask` 后的可能输出名称。


  namedinference::propagate_names_if_nonempty(result, maybe_outnames);

// 如果 `maybe_outnames` 非空，则将这些输出名称传播到 `result` 张量中，使用 `namedinference::propagate_names_if_nonempty` 函数。


  return result;

// 返回创建的量化张量 `result`。
// 返回输入张量的压缩版本，移除维度为1的轴
Tensor squeeze(const Tensor& self) {
  // 推断要压缩的几何形状
  auto g = inferSqueezeGeometry(self);
  // 使用推断的几何形状创建新的张量
  at::Tensor result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  // 计算可能的输出名称
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
  // 如果可能的输出名称非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回压缩后的张量
  return result;
}

// 返回量化输入张量的压缩版本，移除维度为1的轴
Tensor squeeze_quantized(const Tensor& self) {
  // 调用通用的压缩函数，并传入空的维度选项
  return squeeze_qtensor(self, c10::nullopt);
}

// 返回输入张量在给定维度上的压缩版本
Tensor squeeze(const Tensor& self, int64_t dim) {
  // 获取张量的维度数
  int64_t dims = self.dim();
  // 将dim映射到有效范围内
  dim = maybe_wrap_dim(dim, dims);
  
  // 如果张量为空或者指定维度上的大小不为1，则不进行压缩
  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    // 返回指定维度上的张量数据
    return self.as_strided_symint(self.sym_sizes(), self.sym_strides());
  }
  
  // 推断指定维度上的压缩几何形状
  auto g = inferSqueezeGeometry(self, dim);
  // 使用推断的几何形状创建新的张量
  auto result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  // 传播名称，但排除指定的维度
  namedinference::propagate_names_except(result, self, {dim});
  // 返回压缩后的张量
  return result;
}

// 返回量化输入张量在给定维度上的压缩版本
Tensor squeeze_quantized(const Tensor& self, int64_t dim) {
  // 调用量化压缩函数，传入指定的维度
  return squeeze_qtensor(self, dim);
}

// 返回输入张量在给定维度列表上的压缩版本
Tensor squeeze(const Tensor& self, IntArrayRef dims) {
  // 将维度列表转换为位集
  auto mask = dim_list_to_bitset(dims, self.dim());
  // 推断在给定位集上的压缩几何形状
  auto g = inferSqueezeGeometry(self, mask);
  // 使用推断的几何形状创建新的张量
  at::Tensor result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  // 计算可能的输出名称，考虑给定的位集
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self, mask);
  // 如果可能的输出名称非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回压缩后的张量
  return result;
}

// 返回量化输入张量在给定维度列表上的压缩版本
Tensor squeeze_quantized(const Tensor& self, IntArrayRef dim) {
  // 调用量化压缩函数，传入指定的维度列表
  return squeeze_qtensor(self, dim);
}

// 在原地操作，返回输入张量的压缩版本，移除维度为1的轴
Tensor & squeeze_(Tensor& self) {
  // 推断要压缩的几何形状
  auto g = inferSqueezeGeometry(self);
  // 原地操作，使用推断的几何形状创建新的张量
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  // 返回原地操作后的张量
  return self;
}

// 在原地操作，返回输入张量在给定维度上的压缩版本
Tensor & squeeze_(Tensor& self, int64_t dim) {
  // 获取张量的维度数
  int64_t dims = self.dim();
  // 将dim映射到有效范围内
  dim = maybe_wrap_dim(dim, self.dim());

  // 如果张量为空或者指定维度上的大小不为1，则不进行压缩
  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    // 原地操作，返回指定维度上的张量数据
    self.as_strided__symint(self.sym_sizes(), self.sym_strides());
    // 返回原地操作后的张量
    return self;
  }
  
  // 推断指定维度上的压缩几何形状
  auto g = inferSqueezeGeometry(self, dim);
  // 原地操作，使用推断的几何形状创建新的张量
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  // 返回原地操作后的张量
  return self;
}

// 在原地操作，返回输入张量在给定维度列表上的压缩版本
Tensor & squeeze_(Tensor &self, IntArrayRef dims) {
  // 将维度列表转换为位集
  auto mask = dim_list_to_bitset(dims, self.dim());
  // 推断在给定位集上的压缩几何形状
  auto g = inferSqueezeGeometry(self, mask);
  // 原地操作，使用推断的几何形状创建新的张量
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  // 返回原地操作后的张量
  return self;
}

// 注意事项 [ 不安全的视图 ]
// _unsafe_view() 与 view() 不同之处在于，返回的张量不会被视为自动微分的视图。
// 它只能在 `self` 张量是临时的情况下安全使用。例如，被视图化的张量 (a + b) 在视图后立即被丢弃：
//
//  res = at::_unsafe_view(a + b, size);
//
// 这是一种 hack，因为对被视为视图的张量进行原地操作可能比对非视图张量昂贵得多。
# 实现张量的视图操作，改变其尺寸为指定大小
inline Tensor view_impl(const Tensor& self, IntArrayRef size) {

  # 推断新的尺寸向量，以确保张量的元素数量与指定尺寸兼容
  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  # 计算新的步长，以确保视图操作的有效性
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  # 检查计算的步长是否存在，以保证视图大小与输入张量的大小和步长兼容
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  # 返回一个具有指定大小和步长的视图张量，与原张量共享数据
  return alias_with_sizes_and_strides(self, inferred_size, *stride);

}

# 返回不安全版本的视图操作，直接调用安全版本的实现
Tensor _unsafe_view(const Tensor& self, IntArrayRef size) {
  return view_impl(self, size);
}

# 在指定维度上对稠密张量进行unsqueeze操作
Tensor unsqueeze(const Tensor& self, int64_t dim) {
  # 确定维度是否需要包装，以确保在有效范围内
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  # 推断unsqueeze操作后的几何属性
  auto g = inferUnsqueezeGeometry_symint(self, dim);
  # 返回一个以指定尺寸和步长重构的张量
  return self.as_strided_symint(g.sizes, g.strides);
}

# 在指定维度上对稀疏张量进行unsqueeze操作
Tensor unsqueeze_sparse(Tensor const &self, int64_t dim) {
  # 确定维度是否需要包装，以确保在有效范围内
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  # 获取稀疏和稠密维度
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  # 获取稀疏张量的索引
  auto indices = self._indices();
  auto sizes = self.sizes().vec();
  # 在指定位置插入尺寸为1的新维度
  sizes.insert(sizes.begin() + dim, 1);
  if (dim <= sparse_dim) {
    # 在指定位置插入尺寸为1的新索引，返回一个具有修改后尺寸的稀疏张量
    auto new_indices = at::cat(
        {indices.narrow(0, 0, dim),
         at::zeros(
             {1, indices.size(1)},
             kLong,
             indices.options().layout_opt(),
             indices.options().device_opt(),
             indices.options().pinned_memory_opt()),
         indices.narrow(0, dim, indices.size(0) - dim)});
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim + 1, dense_dim, sizes, new_indices, self._values(), self.options());
  } else {
    # 在指定位置插入尺寸为1的新稠密维度，返回一个具有修改后尺寸的稀疏张量
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim + 1, sizes, indices, self._values().unsqueeze(dim - sparse_dim + 1), self.options());
  }
}

# 在指定维度上对量化张量进行unsqueeze操作
Tensor unsqueeze_quantized(const Tensor& self, int64_t dim) {
  # 确定维度是否需要包装，以确保在有效范围内
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  # 推断unsqueeze操作后的几何属性
  auto g = inferUnsqueezeGeometry(self, dim);
  # 获取量化器
  auto quantizer = get_qtensorimpl(self)->quantizer();
  # 如果量化方案为逐通道仿射量化，则调整轴
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    # 创建新的逐通道仿射量化器
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  # 返回一个新的量化张量，具有指定的尺寸、步长和量化器
  return make_qtensor(self, g.sizes, g.strides, std::move(quantizer));
}

# 对张量进行原位unsqueeze操作
Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  # 确定维度是否需要包装，以确保在有效范围内
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  # 推断unsqueeze操作后的几何属性
  auto g = inferUnsqueezeGeometry(self, dim);
  # 原位重构张量的尺寸和步长
  self.as_strided_(g.sizes, g.strides);
  # 返回原张量的引用
  return self;
}
// 将张量 self 沿着指定的 start_dim 和 end_dim 范围内的维度展平，并返回展平后的张量
Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  // 将 start_dim 和 end_dim 转换为有效的维度索引，确保不超出张量维度的范围
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  // 检查 start_dim 是否小于等于 end_dim，否则抛出错误信息
  TORCH_CHECK(start_dim <= end_dim, "flatten() has invalid args: start_dim cannot come after end_dim");

  // 如果张量 self 是零维的，将其变形为包含一个元素的张量并返回
  if (self.dim() == 0) {
    return self.reshape({1});
  }
  // 如果 start_dim 等于 end_dim，说明不需要展平，直接返回原始张量
  if (start_dim == end_dim) {
    return self;
  }

  // 计算展平后张量的 slice 数量，并根据 slice 数量构建新的形状 shape
  auto slice_numel = c10::multiply_integers(self.sym_sizes().slice(start_dim, end_dim - start_dim + 1));
  std::vector<c10::SymInt> shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  // 将 start_dim 之前的维度大小添加到 shape 中
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sym_sizes()[i]);
  }
  // 添加 slice_numel 作为新的维度大小
  shape.push_back(slice_numel);
  // 将 end_dim 之后的维度大小添加到 shape 中
  for (const auto i : c10::irange(end_dim + 1, self.dim())) {
    shape.push_back(self.sym_sizes()[i]);
  }

  // 调用 native::reshape_symint 函数，根据新的形状 shape 对张量 self 进行形状重塑，并返回重塑后的张量
  return native::reshape_symint(self, shape);
}

// 将张量 self 沿着指定的 start_dim 和 end_dim 范围内的维度展平，并指定展平后的结果维度 out_dim
Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  // 将 start_dim 和 end_dim 转换为有效的维度索引，确保不超出张量维度的范围
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  // 检查 start_dim 是否小于等于 end_dim，否则抛出错误信息
  TORCH_CHECK(start_dim <= end_dim, "flatten() has invalid args: start_dim cannot come after end_dim");

  // 获取张量 self 当前的维度命名列表，并创建一个可修改的副本 outnames
  auto outnames = self.names().vec();
  // 删除 outnames 中 start_dim 到 end_dim 之间的所有维度命名
  outnames.erase(outnames.begin() + start_dim, outnames.begin() + end_dim + 1);
  // 在 outnames 的 start_dim 位置插入新的输出维度命名 out_dim
  outnames.insert(outnames.begin() + start_dim, out_dim);

  // 定义一个 Tensor 变量 result
  Tensor result;
  {
    // 创建 NoNamesGuard 对象 guard，确保后续操作不受命名的影响
    NoNamesGuard guard;
    // 调用 native::flatten 函数，对张量 self 进行展平操作，并将结果赋给 result
    result = native::flatten(self, start_dim, end_dim);
  }
  // 在展平后的结果 result 中应用新的维度命名 outnames
  internal_set_names_inplace(result, outnames);
  // 返回展平后带有新维度命名的张量 result
  return result;
}

// 将张量 self 沿着指定的 start_dim 和 end_dim 范围内的维度展平，并指定展平后的结果维度 out_dim
Tensor flatten(const Tensor& self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  // 将维度命名转换为对应的位置索引
  auto start_pos = dimname_to_position(self, start_dim);
  auto end_pos  = dimname_to_position(self, end_dim);
  // 调用 native::flatten 函数，根据位置索引对张量 self 进行展平操作，并指定结果维度命名 out_dim
  return native::flatten(self, start_pos, end_pos, out_dim);
}

// 将张量 self 沿着指定的 dims 中的维度进行展平，并指定展平后的结果维度 out_dim
Tensor flatten(const Tensor& self, DimnameList dims, Dimname out_dim) {
  // 将维度命名列表转换为对应的位置索引列表
  auto positions = dimnames_to_positions(self, dims);
  // 检查 positions 列表不能为空
  TORCH_CHECK(!positions.empty(),
      "flatten(tensor, dims, out_dim): dims cannot be empty");
  // 检查 dims 中的维度是否连续
  for (const auto i : c10::irange(positions.size() - 1)) {
    if (positions[i] + 1 == positions[i + 1]) continue;
    // 若维度不连续，抛出错误信息
    TORCH_CHECK(positions[i] + 1 == positions[i + 1],
        "flatten(tensor, dims, out_dim): dims ", dims, " must be consecutive ",
        "in Tensor", self.names());
  }
  // 调用 native::flatten 函数，根据位置索引对张量 self 进行展平操作，并指定结果维度命名 out_dim
  return native::flatten(self, *dims.begin(), *(dims.end() - 1), out_dim);
}

// 将张量 self 进行展平为一维张量，并返回展平后的结果
Tensor ravel(const Tensor& self) {
  // 对张量进行连续化操作，并将其视图变形为一维张量并返回
  return self.contiguous().view(-1);
}
static inline void handle_unflatten_exception(const std::runtime_error &e,
                                              const Tensor &self,
                                              int64_t dim,
                                              SymIntArrayRef sizes,
                                              std::optional<DimnameList> names) {
  // 如果异常信息中不包含特定字符串，则抛出 TORCH_CHECK 错误
  if (!strstr(e.what(), "is invalid for input of size")) {
    TORCH_CHECK(false, "unflatten got an unexpected error:\n", e.what());
  }

  // 如果 Tensor 对象有命名信息
  if (self.has_names()) {
    // 抛出 TORCH_CHECK 错误，说明提供的 sizes 不符合在指定维度上的期望
    TORCH_CHECK(false,
                "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
                dim, " (", self.names()[dim], ": ", self.sym_size(dim), ") in Tensor", self.names());

  } else {
    // 抛出 TORCH_CHECK 错误，说明提供的 sizes 不符合在指定维度上的期望（对于无命名的 Tensor）
    TORCH_CHECK(false,
                "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
                dim, " (", self.sym_size(dim), ") in the input tensor");
  }
}

// 根据指定的 Tensor 和维度对其进行 unflatten 操作，返回重塑后的 Tensor 对象
static Tensor unflatten_impl(const Tensor& self, int64_t dim, SymIntArrayRef sizes, std::optional<DimnameList> names) {
  // 根据可能需要包装的维度，确定最终的操作维度
  dim = maybe_wrap_dim(dim, self.dim());

  // 检查 sizes 是否为空
  TORCH_CHECK(!sizes.empty(), "unflatten: sizes must be non-empty");
  // 内部断言：如果 names 存在，则其大小必须与 sizes 相同
  TORCH_INTERNAL_ASSERT(!names || names->size() == sizes.size());
  // 如果 Tensor 对象有命名信息，则确保在 unflatten 操作中也提供了命名信息
  if (self.has_names()) {
    TORCH_CHECK(names, "unflatten: input is a named tensor but no names were given for unflattened sizes");
  }

  SymDimVector inferred_size;
  try {
    // 使用 sizes 和当前维度的大小来推断新的维度大小
    inferred_size = at::infer_size_dv(sizes, self.sym_size(dim));
  } catch (const std::runtime_error& e) {
    // at::infer_size 可能会抛出 std::runtime_error 异常，需要处理该异常
    // 调用 handle_unflatten_exception 处理异常信息
    handle_unflatten_exception(e, self, dim, sizes, names);
  }

  // 创建一个新的 shape，用于存储 unflatten 后的 Tensor 的形状
  SymDimVector shape(self.sym_sizes().begin(), self.sym_sizes().end());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, inferred_size.begin(), inferred_size.end());

  Tensor result;
  {
    // 使用 NoNamesGuard 确保在操作期间不修改命名信息
    NoNamesGuard guard;
    // 使用 view_symint 方法对 Tensor 进行 reshape 操作
    result = self.view_symint(shape);
  }

  // 如果提供了命名信息，则在结果中设置相应的命名信息
  if (names) {
    auto outnames = self.names().vec();
    outnames.erase(outnames.begin() + dim);
    outnames.insert(outnames.begin() + dim, names->begin(), names->end());
    // 在结果 Tensor 中设置新的命名信息
    at::internal_set_names_inplace(result, outnames);
  }

  // 返回最终的 unflatten 后的 Tensor 对象
  return result;
}

// 使用 unflatten_impl 对非命名 Tensor 进行 unflatten 操作
Tensor unflatten_symint(const Tensor& self, int64_t dim, SymIntArrayRef sizes) {
  return native::unflatten_impl(self, dim, sizes, c10::nullopt);
}

// 使用 unflatten_impl 对命名 Tensor 进行 unflatten 操作
Tensor unflatten_dimname_symint(const Tensor& self, Dimname dim, SymIntArrayRef sizes, DimnameList names) {
  // 将 Dimname 转换为对应的位置索引，然后调用 unflatten_impl
  return native::unflatten_impl(self, dimname_to_position(self, dim), sizes, names);
}

// 将当前 Tensor 按照另一个 Tensor 的形状进行 reshape
Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view_symint(other.sym_sizes());
}

// 将 Tensor 按照指定维度进行 unbind 操作，返回多个分割后的 Tensor
std::vector<Tensor> unbind(const Tensor &self, int64_t dim) {
  // 根据可能需要包装的维度，确定最终的操作维度
  dim = maybe_wrap_dim(dim, self.dim());
  // 获取指定维度的大小
  int64_t size = self.size(dim);
  // 创建存储分割后 Tensor 的向量
  std::vector<Tensor> tensors(size);
  // 遍历分割后的每个 Tensor
  for (const auto i : c10::irange(size)) {
    # 将第 i 个张量设置为在指定维度 dim 上选择的第 i 个子张量
    tensors[i] = self.select(dim, i);
  }
  # 返回更新后的张量列表
  return tensors;
}

// 解绑定操作，将给定维度上的张量解绑定为一个张量向量
std::vector<Tensor> unbind(const Tensor& self, Dimname dim) {
  return at::unbind(self, dimname_to_position(self, dim));
}

// 创建网格操作，使用输入张量列表创建网格
std::vector<Tensor> meshgrid(TensorList tensors) {
  // 输出警告信息，提示未来版本可能需要传递索引参数
  TORCH_WARN_ONCE("torch.meshgrid: in an upcoming release, it will be required to pass the "
                  "indexing argument.");
  // 调用本地实现的meshgrid函数，使用默认的"ij"索引方式
  return native::meshgrid(tensors, /*indexing=*/"ij");
}

// 创建网格操作，使用输入张量列表和指定的索引方式创建网格
std::vector<Tensor> meshgrid(TensorList tensors,
                             c10::string_view indexing) {
  // 获取输入张量列表的大小
  int64_t size = tensors.size();
  // 检查输入张量列表不能为空
  TORCH_CHECK(size > 0, "meshgrid expects a non-empty TensorList");

  // 检查所有张量是否具有相同的数据类型和设备
  for(const auto i: c10::irange(size - 1)){
    TORCH_CHECK(tensors[i].dtype() == tensors[i+1].dtype(), "meshgrid expects all tensors to have the same dtype");
    TORCH_CHECK(tensors[i].device() == tensors[i+1].device(), "meshgrid expects all tensors to have the same device");
  }

  // 创建常量引用的张量向量，以便在不同索引方式下可能需要重新排序
  std::vector<std::reference_wrapper<const Tensor>> tensor_refs(tensors.begin(),
                                                                tensors.end());

  // 是否需要交换第一个和第二个张量的标志
  bool swap_first_and_second_tensors = false;

  // 如果索引方式为"xy"，则可能需要交换第一个和第二个张量
  if (indexing == "xy") {
    // 只有在输入张量列表中有至少两个张量时才进行交换
    swap_first_and_second_tensors = size >= 2;
    if (swap_first_and_second_tensors) {
      std::swap(tensor_refs[0], tensor_refs[1]);
    }
  } else {
    // 对于不支持的索引方式，抛出错误信息，目前仅支持"xy"和"ij"
    TORCH_CHECK(false, "meshgrid only supports 'xy' and 'ij' indexing");
  }
    // 检查索引方式是否为 "ij"，如果不是则抛出错误信息
    TORCH_CHECK(indexing == "ij",
                "torch.meshgrid: indexing must be one of \"xy\" or \"ij\", "
                "but received: ", indexing);
  }

  // 创建一个存储各维度大小的 SymInt 向量
  std::vector<c10::SymInt> shape(size);
  // 遍历每个张量引用，检查其维度是否不大于1，并获取对应的符号化元素个数
  for(const auto i: c10::irange(size)){
    TORCH_CHECK(tensor_refs[i].get().dim() <= 1,
                "torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: ", tensor_refs[i]);
    shape[i] = tensor_refs[i].get().sym_numel();  // 将0维张量视为1维张量处理其符号化元素个数
  }
  // 创建一个 Tensor 向量用于存储网格
  std::vector<Tensor> grids;
  grids.reserve(size);
  // 创建一个存储视图形状的 SymInt 向量，并初始化为全1
  std::vector<c10::SymInt> view_shape(size, 1);
  // 遍历每个张量引用，为每个维度的视图形状设置为-1（以便自动推断）
  for(const auto i: c10::irange(size)){
    view_shape[i] = -1;  // 选择该维度用于推断
    // 根据视图形状创建符号整数视图，并扩展为指定形状的符号整数
    grids.push_back(tensor_refs[i].get().view_symint(view_shape).expand_symint(shape));
    view_shape[i] = 1;  // 恢复到先前的值
  }

  // 如果需要交换第一个和第二个张量，则交换对应的网格输出
  if (swap_first_and_second_tensors) {
    std::swap(grids[0], grids[1]);
  }
  // 返回生成的网格张量向量
  return grids;
// Numpy 风格的 `a.T`：返回维度颠倒的张量
Tensor numpy_T(const Tensor &self) {
  // 获取张量的维度
  const auto n = self.dim();
  // 如果维度不是 2 或 0，则发出警告
  if (n != 2 && n != 0) {
    TORCH_WARN_ONCE(
        "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated ",
        "and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices ",
        "or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor."
    );
  }
  // 如果维度为 0，则发出特定警告
  if (n == 0) {
    TORCH_WARN_ONCE("Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases.");
  }
  // 创建一个向量来保存转置后的维度顺序
  DimVector transpose_dims;
  // 从最后一个维度开始向前遍历，将维度顺序反转
  for (int64_t i = n - 1; i >= 0; --i) {
    transpose_dims.push_back(i);
  }
  // 返回按照新维度顺序排列的张量
  return self.permute(transpose_dims);
}

// 返回矩阵的共轭转置
Tensor matrix_H(const Tensor &self) {
  // 获取张量的维度
  const auto ndim = self.dim();
  // 如果维度为 0，则发出特定警告
  if (ndim == 0) {
    TORCH_WARN_ONCE("Tensor.H is deprecated on 0-D tensors. Consider using x.conj().");
  }
  // 检查张量是否为复数类型，根据情况返回转置或共轭转置
  if (self.is_complex()) {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  } else {
    return ndim == 0 ? self : self.transpose(-2, -1);
  }
}

// 返回张量的转置或共轭转置
namespace {
Tensor _adjoint(const Tensor &self, const bool transpose, const char* const name) {
  // 获取张量的维度
  const auto ndim = self.dim();
  // 如果维度为 1，则抛出错误
  TORCH_CHECK(ndim != 1,
      "tensor.", name, " is only supported on matrices or batches of matrices. Got 1-D tensor.");
  // 根据参数决定返回转置或共轭转置的张量
  if (transpose || !self.is_complex()) {
    return ndim == 0 ? self : self.transpose(-2, -1);
  } else {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  }
}
} // 匿名命名空间结束

// 返回矩阵的转置
Tensor mT(const Tensor &self) {
  // 如果维度为 0，则发出特定警告
  if (self.dim() == 0) {
    TORCH_WARN_ONCE("Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases.");
  }
  // 调用 _adjoint 函数，要求返回转置的张量
  return _adjoint(self, /*transpose=*/true, "mT");
}

// 返回矩阵的共轭转置
Tensor mH(const Tensor &self) {
  // 如果维度为 0，则发出特定警告
  if (self.dim() == 0) {
    TORCH_WARN_ONCE("Tensor.mH is deprecated on 0-D tensors. Consider using x.conj().");
  }
  // 调用 _adjoint 函数，要求返回共轭转置的张量
  return _adjoint(self, /*transpose=*/false, "mH");
}

// 返回矩阵的转置或共轭转置
Tensor adjoint(const Tensor &self) {
  // 如果维度为 0，则发出特定警告
  if (self.dim() == 0) {
    TORCH_WARN_ONCE("adjoint() is deprecated on 0-D tensors. Consider using x.conj().");
  }
  // 调用 _adjoint 函数，根据情况返回转置或共轭转置的张量
  return _adjoint(self, /*transpose=*/false, "adjoint()");
}

// 返回具有指定大小的张量视图
Tensor view(const Tensor& self,
            at::IntArrayRef size) {
  return view_impl(self, size);
}

// 返回与输入张量具有相同大小和步幅的别名张量
Tensor alias(const Tensor& self) {
  return alias_with_sizes_and_strides(self, self.sym_sizes(), self.sym_strides());
}
Tensor detach(const Tensor& self) {
  // detach()不同于alias()！主要区别在于detach()不允许元数据更改，而alias()允许。
  return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
    // ADInplaceOrView逻辑将在运行时覆盖这些值；否则这些是默认值。
    /*version_counter=*/0,
    /*allow_tensor_metadata_change=*/false));
}

Tensor unfold(const Tensor& self, int64_t d, int64_t size, int64_t step) {
  // 处理当self.dim() == 0且d == 0的特殊情况
  auto ndim = self.dim();
  d = at::maybe_wrap_dim(d, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  int64_t max_size = self.dim() == 0 ? 1 : sizes[d];
  TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", d,
                                " is ", max_size, " but size is ", size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");
  sizes.push_back(size);
  strides.push_back(self.dim() == 0 ? 1 : strides[d]);
  // 处理self.dim() == 0的情况
  if (d < ndim) {
    sizes[d] = (sizes[d] - size) / step + 1;
    strides[d] *= step;
  }
  return self.as_strided(sizes, strides);
}

Tensor diag(const Tensor& self, int64_t offset) {
  auto ndim = self.dim();
  TORCH_CHECK(ndim == 1 || ndim == 2, "diag(): Supports 1D or 2D tensors. Got ", self.dim(), "D");
  if (ndim == 1) {
    // 返回self的对角线嵌入（1D情况）
    return at::diag_embed(self, offset);
  } else {
    // 返回self的对角线的拷贝（2D情况）
    return at::diagonal_copy(self, offset);
  }
}

Tensor& diag_out(const Tensor& self, int64_t offset, Tensor& out) {
  auto ndim = self.dim();
  TORCH_CHECK(ndim == 1 || ndim == 2, "Supports 1D or 2D tensors. Got ", self.dim(), "D");
  if (ndim == 1) {
    TORCH_CHECK(
        canCast(self.scalar_type(), out.scalar_type()),
        "diag: result type ", self.scalar_type(), " can't be cast to the desired out= type ",
        out.scalar_type());
    // 将self的对角线嵌入到输出张量out中（1D情况）
    return at::diag_embed_out(out, self, offset);
  } else {
    // 将self的对角线拷贝到输出张量out中（2D情况）
    return at::diagonal_copy_out(out, self, offset);
  }
}

Tensor diagonal_backward_symint(const Tensor & grad, SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  auto grad_input = at::zeros_symint(input_sizes, grad.options());
  auto diag = grad_input.diagonal(offset, dim1, dim2);
  diag.copy_(grad);
  return grad_input;
}

Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  // 检查源和目标维度数组的大小是否相同
  TORCH_CHECK(src.size() == dst.size(), "movedim: Invalid source or destination dims: source (",
              src, " dims) should contain the same number of dims as destination (", dst, " dims)");

  size_t self_dim = self.dim();
  DimVector normalized_src(src.size());
  DimVector normalized_dst(dst.size());

  auto wrap_dims = [&self_dim](const IntArrayRef& vec, DimVector& normalized_vec) {
  for (const auto i : c10::irange(vec.size())) {
    // 对于输入向量 `vec` 的每个索引 `i`，执行以下操作：
    normalized_vec[i] = maybe_wrap_dim(vec[i], self_dim);
  };



  wrap_dims(src, normalized_src);
  // 将 `src` 向量使用 `normalized_src` 进行维度包装

  wrap_dims(dst, normalized_dst);
  // 将 `dst` 向量使用 `normalized_dst` 进行维度包装

  auto all_unique = [](const DimVector& dims) {
    // 匿名函数 `all_unique` 接受维度向量 `dims` 作为参数
    DimVector copy = dims;
    // 复制 `dims` 到 `copy` 中
    std::sort(copy.begin(), copy.end());
    // 对 `copy` 中的元素进行排序
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    // 查找相邻重复的元素
    return duplicate == copy.end();
    // 如果没有找到相邻重复的元素，则返回 true，否则返回 false
  };

  TORCH_CHECK(all_unique(normalized_src), "movedim: repeated dim in `source` (", src, ")");
  // 检查 `normalized_src` 是否包含重复的维度，并在发现重复时抛出错误信息

  TORCH_CHECK(all_unique(normalized_dst), "movedim: repeated dim in `destination` (", dst, ")");
  // 检查 `normalized_dst` 是否包含重复的维度，并在发现重复时抛出错误信息

  // 处理标量张量的情况，作为无操作返回
  if (self_dim == 0)
    return self.alias();

  // TODO: 下面的算法可能可以进行优化。
  // 参考：https://github.com/pytorch/pytorch/pull/41480#discussion_r456100505

  // 算法步骤
  // 示例输入
  // 变量状态：
  //     normalized_src = 0, 1
  //     normalized_dst = 2, 4
  //     self_dim = 5
  DimVector order(self_dim);
  // 创建大小为 `self_dim` 的向量 `order`
  DimVector source_dims(self_dim);
  // 创建大小为 `self_dim` 的向量 `source_dims`
  DimVector destination_dims(self_dim);
  // 创建大小为 `self_dim` 的向量 `destination_dims`

  // 初始化两个向量以跟踪维度更新
  // `order` 包含维度位置的最终顺序。
  // 变量状态：
  //     order = NA, NA, NA, NA, NA
  //     source_dims = 0, 1, 2, 3, 4
  //     destination_dims = 0, 1, 2, 3, 4
  std::iota(source_dims.begin(), source_dims.end(), 0);
  // 用起始值初始化 `source_dims`
  std::iota(destination_dims.begin(), destination_dims.end(), 0);
  // 用起始值初始化 `destination_dims`

  // 标记并更新用户提供的维度位置
  // 即 `normalized_src` 和 `normalized_dims`
  // 变量状态：
  //     order = NA, NA, 0, NA, 1
  //     source_dims = -1, -1, 2, 3, 4
  //     destination_dims = 0, 1, -1, 3, -1
  for (const auto i : c10::irange(src.size())) {
      order[normalized_dst[i]] = normalized_src[i];
      // 将 `normalized_src[i]` 的值放入 `order` 的 `normalized_dst[i]` 位置
      source_dims[normalized_src[i]] = -1;
      // 将 `normalized_src[i]` 的位置标记为 `-1`
      destination_dims[normalized_dst[i]] = -1;
      // 将 `normalized_dst[i]` 的位置标记为 `-1`
  }

  // 移除我们已知位置的维度，上一步中标记为 `-1` 的维度
  // 变量状态：
  //     source_dims = 2, 3, 4
  //     destination_dims = 0, 1, 3
  auto source_iter = std::remove(source_dims.begin(), source_dims.end(), -1);
  // 移除 `source_dims` 中的 `-1` 值
  auto destination_iter = std::remove(destination_dims.begin(), destination_dims.end(), -1);
  // 移除 `destination_dims` 中的 `-1` 值

  int64_t rest_dim = self.dim() - src.size();
  // 计算剩余的维度数量
  TORCH_INTERNAL_ASSERT(std::distance(source_dims.begin(), source_iter)  == rest_dim);
  // 内部断言，确保 `source_dims` 中的有效维度数量与剩余维度数量相同
  TORCH_INTERNAL_ASSERT(std::distance(destination_dims.begin(), destination_iter)  == rest_dim);
  // 内部断言，确保 `destination_dims` 中的有效维度数量与剩余维度数量相同

  // 更新剩余维度的位置。
  // `source_dims` 现在包含原始位置
  // `destination_dims` 包含考虑用户输入后将要移动到的新位置
  // 变量状态：
  //     order = 2, 3, 0, 4, 1
  for (const auto i : c10::irange(rest_dim)) {
      order[destination_dims[i]] = source_dims[i];
      // 将 `source_dims[i]` 的值放入 `order` 的 `destination_dims[i]` 位置
  }

  return self.permute(order);
  // 返回按 `order` 排列的张量 `self`
}

// 移动张量的维度，将指定的维度从 src 移动到 dst
Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

// 移动张量的维度，将多个指定的维度从 src 移动到 dst
Tensor moveaxis(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  return at::movedim(self, src, dst);
}

// 移动张量的维度，将一个指定的维度从 src 移动到 dst
Tensor moveaxis(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

// 交换张量的两个轴（维度）
Tensor swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose(axis0, axis1);
}

// 原地交换张量的两个轴（维度）
Tensor& swapaxes_(Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose_(axis0, axis1);
}

// 交换张量的两个维度
Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose(dim0, dim1);
}

// 原地交换张量的两个维度
Tensor& swapdims_(Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose_(dim0, dim1);
}

// 将一组密集张量展平为一个连续的张量
Tensor flatten_dense_tensors(TensorList tensors) {
  static auto flatten = [](const Tensor &t) { return t.contiguous().view({-1}); };
  if (tensors.size() == 1)
    return flatten(tensors[0]);
  return at::cat(fmap(tensors, flatten));
}

// 将一个扁平化的张量重新展开为一组张量
std::vector<Tensor> unflatten_dense_tensors(const Tensor& flat, TensorList tensors) {
  std::vector<Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = tensor.numel();
    // 如果张量是空的，则创建一个新的空张量，使用与 flat 张量相同的选项
    // 这可以避免未展开的空张量与其他未展开的张量共享相同的存储
    if (numel == 0) {
      outputs.push_back(at::empty({0}, flat.options()));
    } else {
      outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
      offset += numel;
    }
  }
  return outputs;
}

// 克隆张量，通过克隆其所在的基础存储来复制确切的步幅和存储偏移
// 注意：*_scatter 操作需要保持步幅的正确性
// 函数化操作应该保持输入的步幅行为不变
// 具体来说，*_scatter(base, mutated_view, ...) 的输出应该与 "base" 具有相同的大小/步幅/存储偏移
// 克隆输入张量，并尽可能保持其步幅（strides）和存储偏移（storage_offset）。
at::Tensor clone_preserve_strides(const at::Tensor& self) {
  // 确保输入张量有有效的存储
  TORCH_INTERNAL_ASSERT(self.has_storage());
  
  // 如果输入张量存在内存重叠，无法保持步幅和存储偏移，
  // 因为 *_scatter 操作会尝试复制到克隆的张量中。
  // 但是，在功能化用户代码中**绝不**应该出现这种情况；
  // 大多数尝试变异有内存重叠的张量的 aten 操作本身会报错。
  //
  // 唯一可能出现问题的地方是在自动求导中 - 如果在前向传播中有 select_scatter，
  // 那么自动求导将为反向传播生成一个对应的操作。
  // 如果 select_scatter 的输入是 grad_output，则可能是一个具有内部重叠的扩展张量。
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    return self.clone();  // 如果存在内存重叠，则直接克隆输入张量
  }
  
  // 计算数据类型的大小和存储器的总字节数
  auto dtype_size = self.dtype().itemsize();
  auto nbytes = self.storage().sym_nbytes();
  TORCH_INTERNAL_ASSERT(nbytes % dtype_size == 0);
  auto numel = nbytes / dtype_size;
  
  // 根据大小和步幅创建一个完整的自定义符号整数（symint）张量
  auto self_full_size = self.as_strided_symint({std::move(numel)}, {1}, 0);
  
  // 克隆完整尺寸的张量
  auto clone = self_full_size.clone();
  
  // 根据原始张量的符号大小、步幅和存储偏移创建一个自定义符号整数（symint）张量
  auto out = clone.as_strided_symint(self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());
  
  // 返回克隆的张量
  return out;
}

// 对指定维度进行切片并进行散射操作
at::Tensor slice_scatter(const at::Tensor& self, const at::Tensor& src, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) {
  // 查看注释 [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  
  // 对输出张量进行切片操作
  auto slice = output.slice(dim, start, end, step);
  
  // 检查切片张量的尺寸与源张量的尺寸是否相等
  TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
  
  // 将源张量的内容复制到切片张量中
  slice.copy_(src);
  
  // 返回结果张量
  return output;
}

// 对指定维度进行选择并进行符号整数散射操作
at::Tensor select_scatter_symint(const at::Tensor& self, const at::Tensor& src, int64_t dim, c10::SymInt index) {
  // 查看注释 [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  
  // 对输出张量进行选择符号整数操作
  auto slice = output.select_symint(dim, std::move(index));
  
  // 检查选择张量的尺寸与源张量的尺寸是否相等
  TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
  
  // 将源张量的内容复制到选择张量中
  slice.copy_(src);
  
  // 返回结果张量
  return output;
}

// 对指定维度进行对角线散射操作
at::Tensor diagonal_scatter(const at::Tensor& self, const at::Tensor& src, int64_t offset, int64_t dim1, int64_t dim2) {
  // 查看注释 [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  
  // 对输出张量进行对角线散射操作
  auto slice = output.diagonal(offset, dim1, dim2);
  
  // 检查对角线张量的尺寸与源张量的尺寸是否相等
  TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
  
  // 将源张量的内容复制到对角线张量中
  slice.copy_(src);
  
  // 返回结果张量
  return output;
}

// 对指定大小、步幅和存储偏移进行符号整数散射操作
at::Tensor as_strided_scatter_symint(const at::Tensor& self, const at::Tensor& src, at::SymIntArrayRef size, at::SymIntArrayRef stride, std::optional<c10::SymInt> storage_offset) {
  // 查看注释 [as_strided_scatter backward support]
    // 断言：如果张量不需要梯度或者是连续的，则满足条件；否则抛出错误信息
    TORCH_INTERNAL_ASSERT(!self.requires_grad() || self.is_contiguous(), "as_strided_scatter is currently only supported for contiguous inputs");
    // 查看注释 [*_scatter ops preserve strides]
    // 克隆张量并保留其步长信息
    auto output = clone_preserve_strides(self);
    // 使用符号化大小创建输出张量的切片
    auto slice = output.as_strided_symint(size, stride, std::move(storage_offset));
    // 检查切片的符号化大小与源张量的符号化大小是否相等，否则抛出错误信息
    TORCH_CHECK(slice.sym_sizes() == src.sym_sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sym_sizes(), ", slice size = ", slice.sym_sizes());
    // 在切片上执行数据拷贝操作
    slice.copy_(src);
    // 返回经过操作后的输出张量
    return output;
// 默认的 lift 函数是一个空操作。
// 如果 TLS 设置得当（用于包装张量键，如 Functionalize 或 functorch 转换），
// 那么我们将分派到它们的一个实现，该实现将正确地将张量提升到一个包装器中。
at::Tensor lift(const at::Tensor& self) {
    return self;
}

// 参见 native_functions.yaml 中的注释
at::Tensor lift_fresh(const at::Tensor& self) {
    return self;
}

// 自动化生成的张量列表操作内核在 XLA 上不起作用。TODO(jakeszwe)
void split_copy_Tensor_out(const at::Tensor & self, int64_t split_size, int64_t dim, at::TensorList  out) {
  auto tmp = self.split(split_size, dim);

  TORCH_CHECK(out.size() == tmp.size(), "split_copy_Tensor_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}

// 使用指定的尺寸列表进行分割并复制到输出张量列表中
void split_with_sizes_copy_out(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim, at::TensorList  out) {
  auto tmp = self.split_with_sizes(split_sizes, dim);

  TORCH_CHECK(out.size() == tmp.size(), "split_with_sizes_copy_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    // 如果需要，检查输出张量尺寸是否需要调整
    if (resize_output_check(out[i], tmp[i].sizes())) {
      out[i].resize_(tmp[i].sizes());
    }
    // 检查输出张量的数据类型是否符合预期
    TORCH_CHECK(out[i].dtype() == tmp[i].dtype(),
        "Expected out tensor to have dtype ", tmp[i].dtype(), ", but got ", out[i].dtype(), " instead");
    // 检查输出张量的设备是否符合预期
    TORCH_CHECK(out[i].device() == tmp[i].device(),
        "Expected out tensor to have device ", tmp[i].device(), ", but got ", out[i].device(), " instead");
    // 将分割后的张量复制到输出张量中
    out[i].copy_(tmp[i]);
  }
}

// 按指定维度解绑并复制到输出张量列表中
void unbind_copy_int_out(const at::Tensor & self, int64_t dim, at::TensorList  out) {
  auto tmp = self.unbind(dim);

  TORCH_CHECK(out.size() == tmp.size(), "unbind_copy_int_out() expected an out= argument of size ", tmp.size(), ", got size ", out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}

// 默认情况下稀疏维度的查询函数
int64_t sparse_dim_default(const Tensor& self) {
  TORCH_CHECK(self.layout() == kStrided, "sparse_dim expected sparse or strided tensor layout but got ", self.layout());
  return 0;
}

// 默认情况下稠密维度的查询函数
int64_t dense_dim_default(const Tensor& self) {
  TORCH_CHECK(self.layout() == kStrided, "dense_dim expected sparse or strided tensor layout but got ", self.layout());
  return self.dim();
}

} // namespace at::native
```