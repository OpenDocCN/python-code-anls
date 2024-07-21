# `.\pytorch\aten\src\ATen\native\TensorAdvancedIndexing.cpp`

```
// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containing kLong, kBool or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]
//
// The code contains two implementations of indexing. The more efficient
// implementation treats indexing like an elementwise operation over the
// tensors `result`, `x`, `ind_1`, `ind_2`, etc. This implementation does
// not work for index_put_ with accumulate=True. The other implementation
// combines the indexed tensors into a single linear index that is used
// with Tensor.put_. This is used for index_put_ with accumulate=True.
//
// The more efficient implementation takes the following steps for the
// above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to a common shape
// 2) Record x.stride(i) for each indexed dimension `i`
// 3) Replace the indexed subspace of `x` with the shape of the corresponding
//    subspace of `result` but with stride 0
// 4) Add dimensions of size 1 to the index tensors (ind_1, ind_2, etc.) so
//    that their shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes element-wise over the broadcasted
// and restrided result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of and indirection operations.

// Including necessary headers for tensor operations and indexing
#include <ATen/ATen.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/IListRef.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScatterGatherChecks.h>
// 包含 ATen 库的头文件，用于张量操作的高级索引和并行计算
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含一组通用函数和原生函数的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含一组特定功能的头文件，用于稀疏张量的相关操作和高级索引的实现
#else
#include <ATen/ops/_gather_sparse_backward.h>
#include <ATen/ops/_gather_sparse_backward_native.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/_index_put_impl_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_unsafe_index_native.h>
#include <ATen/ops/_unsafe_index_put_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argwhere_native.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/broadcast_to.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gather_backward_native.h>
#include <ATen/ops/gather_meta.h>
#include <ATen/ops/gather_native.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_add_meta.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_copy_meta.h>
#include <ATen/ops/index_copy_native.h>
#include <ATen/ops/index_fill_native.h>
#include <ATen/ops/index_meta.h>
#include <ATen/ops/index_native.h>
#include <ATen/ops/index_put_native.h>
#include <ATen/ops/index_reduce_meta.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_backward_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_backward_native.h>
#include <ATen/ops/masked_select_native.h>
#include <ATen/ops/nested_to_padded_tensor_native.h>
#include <ATen/ops/nonzero_native.h>
#include <ATen/ops/nonzero_numpy_native.h>
#include <ATen/ops/nonzero_static_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/put_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/scatter_add_meta.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_meta.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/scatter_reduce_meta.h>
#include <ATen/ops/scatter_reduce_native.h>
#include <ATen/ops/take_along_dim_native.h>
#include <ATen/ops/take_native.h>
#include <ATen/ops/zeros_like.h>
#endif

// 如果定义了 USE_FBGEMM 宏，则包含 FBGEMM 库的 Utils 头文件
#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif

// 包含 C++ STL 头文件
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

// 定义 at::native 命名空间，包含张量操作的相关函数
namespace at::native {

// 定义一个函数，将张量列表的形状信息转换为字符串
std::string shapes_as_str(TensorList tensors);

// 定义一个函数，创建并返回一个 AdvancedIndex 结构，用于高级索引
AdvancedIndex make_info(Tensor self, IOptTensorListRef orig);

} // namespace at::native

// 定义 at::meta 命名空间，用于元数据函数的相关定义
namespace at::meta {

// TORCH_META_FUNC 宏生成的元数据函数，用于 gather 操作
TORCH_META_FUNC(gather)
// 定义名为 scatter_meta_impl 的模板函数，用于执行 scatter 和 gather 操作的元数据操作
template <bool use_new_options = false, typename Meta>
void scatter_meta_impl(
    Meta& meta,                         // 元数据对象的引用
    const Tensor& self,                 // 输入张量 self
    int64_t dim,                        // 操作维度 dim
    const Tensor& index,                // 索引张量 index
    const std::optional<Tensor>& src = nullopt,  // 可选的源张量 src，默认为空
    const std::optional<c10::string_view> reduce = nullopt) {  // 可选的减少操作 reduce，默认为空
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());  // 对维度 dim 进行边界处理

  // 检查 scatter 操作中的数据类型匹配
  at::native::scatter_gather_dtype_check("scatter", self, index, src);
  // 检查 scatter 操作中的形状匹配
  at::native::scatter_shape_check(self, wrapped_dim, index, src);

  // 获取输出张量，如果已定义，则执行内部重叠检查
  auto output = meta.maybe_get_output(0);
  if (output.defined()) {
    at::assert_no_internal_overlap(output);  // 确保输出张量内部无重叠
    at::assert_no_overlap(output, index);    // 确保输出张量与索引张量无重叠
    if (src.has_value()) {
      at::assert_no_overlap(output, src.value());  // 如果有源张量，则确保输出张量与源张量无重叠
    }
  }

  // 设置输出张量的原始步长方式
  meta.set_output_raw_strided(0, self.sizes(), {}, self.options());

  // 如果提供了减少操作类型，则检查其有效性
  if (reduce.has_value()) {
    at::native::get_operator_enum(reduce.value(), use_new_options);
  }
}

// 定义 scatter 函数的元数据实现，处理张量 src 作为输入
TORCH_META_FUNC2(scatter, src)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_meta_impl(*this, self, dim, index, src);  // 调用 scatter_meta_impl 处理 scatter 操作
}

// 定义 scatter 函数的元数据实现，处理标量 value 作为输入
TORCH_META_FUNC2(scatter, value)
(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  scatter_meta_impl(*this, self, dim, index);  // 调用 scatter_meta_impl 处理 scatter 操作
}

// 定义 scatter 函数的元数据实现，处理张量 src 和减少操作类型 reduce 作为输入
TORCH_META_FUNC2(scatter, reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce) {
  // 发出一次警告，表明将来的版本中不再支持使用 Tensor src 的 reduce 参数
  TORCH_WARN_ONCE(
      "The reduce argument of torch.scatter with Tensor src is deprecated and will be removed ",
      "in a future PyTorch release. Use torch.scatter_reduce instead for more reduction options."
  );
  scatter_meta_impl(*this, self, dim, index, src, reduce);  // 调用 scatter_meta_impl 处理 scatter 操作
}

// 定义 scatter 函数的元数据实现，处理标量 src 和减少操作类型 reduce 作为输入
TORCH_META_FUNC2(scatter, value_reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& src,
 const c10::string_view reduce) {
  scatter_meta_impl(*this, self, dim, index, nullopt, reduce);  // 调用 scatter_meta_impl 处理 scatter 操作
}
}

TORCH_META_FUNC(scatter_add)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  // 调用 scatter_meta_impl 函数，执行 scatter_add 操作的元数据处理
  scatter_meta_impl(*this, self, dim, index, src, "add");
}

TORCH_META_FUNC2(scatter_reduce, two)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce,
 bool include_self) {
  // 调用 scatter_meta_impl 函数，执行 scatter_reduce 操作的元数据处理，使用新选项
  scatter_meta_impl</*use_new_options=*/true>(*this, self, dim, index, src, reduce);
}

TORCH_PRECOMPUTE_META_FUNC(index_copy)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  // 调整维度，确保维度在有效范围内
  dim = maybe_wrap_dim(dim, self.dim());

  const Tensor& result = maybe_get_output(0);

  // 在调整大小（如果需要）后进行内存重叠检查
  // 但只有在 result 已定义时才有意义进行这些检查，因此这里使用布尔变量 `check_result`
  // 更多细节请参见：https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832
  // 和 https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  // 设置输出张量的原始步长和大小
  set_output_raw_strided(0, self.sizes(), {}, self.options());
  if (check_result) {
    // 检查 result 是否有内部重叠
    at::assert_no_internal_overlap(result);
    // 检查 result 和 index 是否有重叠
    at::assert_no_overlap(result, index);
    // 检查 result 和 source 是否有重叠
    at::assert_no_overlap(result, source);
  }

  // 检查 index 的维度是否小于 2
  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");

  // 检查当 source 是标量时，index 是否只有一个元素
  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  } else if ((source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    // 检查当 source 和 destination 不是标量时，它们的维度必须匹配
    TORCH_CHECK_INDEX(false, "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
                   source.dim(), "), destination dimensionality (", self.dim(), ")");
  }

  // 检查 index 是否是长整型张量
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_copy_(): Expected a long tensor for index, but got ", index.scalar_type());
  // 检查 self 和 source 是否有相同的数据类型
  TORCH_CHECK(self.scalar_type() == source.scalar_type(), "index_copy_(): self and source expected to have the same dtype, but got (self) ", self.scalar_type(), " and (source) ", source.scalar_type());
  // 检查 self、index 和 source 是否在相同的设备上
  TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(), ", (index) ", index.device(), ", and (source) ", source.device());

  // 检查源和目标切片是否具有相同的大小
  auto selfSlicedSizes = self.sizes().vec();
  if (!selfSlicedSizes.empty()) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (!sourceSlicedSizes.empty()) {
    // 删除源张量的相应维度大小
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);

删除 `sourceSlicedSizes` 中索引为 `dim` 的元素。


  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {

如果 `selfSlicedSizes` 的大小不等于 `sourceSlicedSizes` 的大小，或者它们的元素不相等：


    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";

创建一个 `std::stringstream` 对象 `ss`，构建一条错误信息字符串，指出源张量和目标张量必须具有相同的切片形状。该字符串包含目标切片形状 `selfSlicedSizes` 在维度 `dim` 处的信息，以及源切片形状 `sourceSlicedSizes` 在维度 0 处的信息。


    TORCH_CHECK(false, ss.str());

使用 `TORCH_CHECK` 断言检查条件为 `false`，并输出 `ss` 中的错误信息字符串。


  TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
          "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");

使用 `TORCH_CHECK_INDEX` 断言检查条件，确保 `source` 张量的维度为 0 或者 `numIndices` 等于 `source` 在维度 `dim` 处的大小。如果条件不满足，输出相应的错误信息。


  return TORCH_PRECOMPUTE_STRUCT(index_copy)().set_dim(dim);

返回一个调用 `TORCH_PRECOMPUTE_STRUCT(index_copy)` 的临时对象，并调用其 `set_dim` 方法设置维度参数 `dim`。
}

template <typename Meta>
void index_func_meta_impl(
  Meta& meta,
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  c10::string_view func) {
  auto numel = index.numel();

  // 检查索引张量维度是否不超过1，输出错误信息包含索引维度、类型和大小
  TORCH_CHECK_INDEX(index.dim() <= 1, func, "_(): Index is supposed to be a vector, but got dim: ",
                    index.dim(), " with type: ", index.scalar_type(), " and size: ", index.sizes());
  // 检查索引张量数据类型是否为int32或int64
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              func, "_(): Expected dtype int32/int64 for index but got: ", index.scalar_type());
  // 检查self张量和source张量是否具有相同的数据类型
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              func, "_(): self (", self.scalar_type(), ") and source (", source.scalar_type(),
              ") must have the same scalar type");
  // 检查dim是否在有效范围内，输出错误信息包含dim和source张量的维度
  TORCH_CHECK(dim == 0 || dim < source.dim(),
              func, "_(): Indexing dim ", dim, " is out of bounds of the source tensor with dim ",
              source.dim());
  // 检查索引数量是否与source张量在指定维度上的大小匹配，输出错误信息包含索引数量和source张量在dim上的大小
  TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
              func, "_(): Number of indices (", numel, ") should be equal to source.size(dim): (",
              source.size(dim), "), for dim: ", dim);

  auto self_sizes = self.sizes().vec();
  auto source_sizes = source.sizes().vec();
  if (source.dim() != 0 && self.dim() != 0) {
    // 如果source和self张量的维度均不为0，则删除对应维度上的大小
    self_sizes.erase(self_sizes.begin() + dim);
    source_sizes.erase(source_sizes.begin() + dim);
  }
  // 检查self张量和source张量的形状是否匹配，输出错误信息包含self和source张量的形状
  TORCH_CHECK(
      self_sizes == source_sizes,
      "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
      self.sizes(),
      " source.shape = ",
      source.sizes());

  auto& result = meta.maybe_get_output(0);
  bool is_defined = result.defined();
  // 设置meta函数的输出张量的大小、步幅和选项
  meta.set_output_raw_strided(0, self.sizes(), {}, self.options());
  if (is_defined) {
    // 如果结果张量已定义，则执行不重叠检查
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  // 一个用于在meta函数中运行TensorIterator检查的hack
  // 参见注释：https://github.com/pytorch/pytorch/pull/65993#discussion_r760307417
  // TODO: (@krshrimali) 尝试从TensorIteratorBase继承而来
  if (result.device() == kMeta && result.dim() > 0) {
    // 如果结果张量的设备为kMeta并且维度大于0，则选择结果张量和source张量在指定维度上的切片
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    // 借用TensorIterator执行二进制操作
    auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
  }
}

TORCH_PRECOMPUTE_META_FUNC(index_add)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha) {
  // 对dim进行包装，确保在有效维度范围内
  dim = maybe_wrap_dim(dim, self.dim());
  // 调用index_func_meta_impl函数，执行索引操作的元信息实现
  index_func_meta_impl(*this, self, dim, index, source, "index_add");
  // 返回包含更新维度信息的TORCH_PRECOMPUTE_STRUCT结构体
  return TORCH_PRECOMPUTE_STRUCT(index_add)().set_dim(dim);
}

TORCH_PRECOMPUTE_META_FUNC(index_reduce)
(const Tensor& self,                                   // 函数参数：当前张量
 int64_t dim,                                           // 函数参数：索引的维度
 const Tensor& index,                                   // 函数参数：索引张量
 const Tensor& source,                                  // 函数参数：源张量
 const c10::string_view reduce,                         // 函数参数：约简操作类型
 bool include_self) {                                   // 函数参数：是否包含自身

  (void)include_self;                                   // 忽略未使用的参数

  TORCH_CHECK(reduce == "prod" || reduce == "mean" || reduce == "amax" || reduce == "amin",  // 检查约简操作类型的有效性
              "index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got ", reduce, ".");

  dim = maybe_wrap_dim(dim, self.dim());                // 调整索引维度，确保在张量维度范围内
  index_func_meta_impl(*this, self, dim, index, source, "index_reduce");  // 调用索引功能的元函数实现
  return TORCH_PRECOMPUTE_STRUCT(index_reduce)().set_dim(dim);  // 预计算结构，设置维度并返回
}

static void build_index_op(
    TensorIteratorBase& iter,                           // 函数参数：迭代器基类的引用
    const at::native::AdvancedIndex& info,              // 函数参数：高级索引信息
    const Tensor& result) {                            // 函数参数：结果张量

  // 'TensorIterator' 需要拥有来自 'info' 的东西，因为 'info' 在 META 函数之后将被销毁。
  TensorIteratorConfig config;

  // info.src 是结果的限制视图
  config.set_check_mem_overlap(false)                  // 设置不检查内存重叠
      .check_all_same_dtype(false)                     // 设置不检查所有输入的数据类型是否相同
      .add_output(result)                              // 添加输出结果
      .add_owned_const_input(info.src);                // 添加拥有的常量输入：info.src

  for (auto& index : info.indices) {                   // 遍历索引列表
    config.add_owned_const_input(index);               // 添加拥有的常量输入：每个索引
  }

  if (!result.defined()) {                             // 如果结果未定义
    config.declare_static_dtype_and_device(info.src.scalar_type(), info.src.device());  // 声明静态数据类型和设备
  }

  iter.build(config);                                  // 使用配置构建迭代器
}

static void check_indices_on_cpu_or_selfdevice(
    const Tensor& self,                                // 函数参数：当前张量
    const at::MaterializedIOptTensorListRef& indices) { // 函数参数：索引张量列表的材料化引用

  auto dev = self.device();                            // 获取当前张量的设备
  bool indices_on_cpu_or_dev = std::all_of(            // 检查所有索引是否在 CPU 或与当前张量相同设备上
      indices.begin(), indices.end(), [=](const at::OptionalTensorRef& opt) {
        return opt.has_value() ? (opt->is_cpu() || opt->device() == dev) : true;
      });

  TORCH_CHECK(
      indices_on_cpu_or_dev,                           // 检查索引是否在合适的设备上
      "indices should be either on ", kCPU,            // 错误信息：索引应该在 CPU 上
      " or on the same device as the indexed tensor (", dev, ")");  // 错误信息：或者与索引张量相同的设备
}

TORCH_PRECOMPUTE_META_FUNC2(index, Tensor)
(const Tensor& self, at::IOptTensorListRef indices) {   // 函数参数：当前张量，可选索引张量列表的引用

  auto materialized = indices.materialize();            // 材料化索引列表

  TORCH_CHECK_INDEX(
      materialized.size() <= (size_t)self.dim(),        // 检查索引数量是否超过张量维度
      "too many indices for tensor of dimension ",
      self.dim(), " (got ", materialized.size(), ")");

  // 只允许：`dev_tensor[{cpu,dev}_tensor]`.
  // 参见：https://github.com/pytorch/pytorch/pull/69607
  check_indices_on_cpu_or_selfdevice(self, materialized);  // 检查索引是否在 CPU 或与当前张量相同设备上

  const auto& result = maybe_get_output();             // 获取可能的输出结果

  if (result.defined()) {                              // 如果结果已定义
    TORCH_CHECK(self.scalar_type() == result.scalar_type(),  // 检查当前张量和结果张量的标量类型是否相同
                "index_out: self (", self.scalar_type(), ") and result (", result.scalar_type(),
                ") must have the same scalar type");
    at::assert_no_internal_overlap(result);            // 确保结果没有内部重叠
    at::assert_no_overlap(result, self);               // 确保结果和当前张量没有重叠
    for (const at::OptionalTensorRef& index : materialized) {  // 遍历材料化后的索引列表
      if (index.has_value()) {
        at::assert_no_overlap(result, *index);         // 确保结果和每个索引没有重叠
      }
    }
    }
  }  // 结束 for 循环和函数定义

  // 调用 make_info 函数生成 Tensor 元数据信息
  auto info = at::native::make_info(self, std::move(indices));

  // 调用 build_index_op 函数构建索引操作
  build_index_op(*this, info, result);

  // 返回一个 TORCH_PRECOMPUTE_STRUCT2(index, Tensor) 的实例
  // 设置其尺寸为 info.indexed_sizes，并移动所有权
  // 设置其步幅为 info.indexed_strides，并移动所有权
  return TORCH_PRECOMPUTE_STRUCT2(index, Tensor)()
      .set_sizes(std::move(info.indexed_sizes))
      .set_strides(std::move(info.indexed_strides));
}

} // namespace at::meta

namespace at::native {

// 定义分发函数的实现
DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_fill_stub);
DEFINE_DISPATCH(index_copy_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(put_stub);
DEFINE_DISPATCH(take_stub);
DEFINE_DISPATCH(masked_fill_stub);
REGISTER_NO_CPU_DISPATCH(index_put_with_sort_stub);  // 注册不使用 CPU 分发的函数
REGISTER_NO_CPU_DISPATCH(index_put_with_sort_quantized_stub);
DEFINE_DISPATCH(masked_select_serial_stub);
DEFINE_DISPATCH(masked_select_stub);
DEFINE_DISPATCH(masked_scatter_stub);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(scatter_reduce_stub);
DEFINE_DISPATCH(scatter_scalar_reduce_stub);
DEFINE_DISPATCH(scatter_reduce_two_stub);

DEFINE_DISPATCH(scatter_add_expanded_index_stub);
DEFINE_DISPATCH(scatter_reduce_expanded_index_stub);
DEFINE_DISPATCH(gather_expanded_index_stub);

// 检查多个张量的步长是否匹配
static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());  // 确保张量列表非空
  auto strides = tensors[0].strides();  // 获取第一个张量的步长
  for (auto& tensor : tensors.slice(1)) {  // 遍历其余张量
    if (!strides.equals(tensor.strides())) {  // 如果步长不匹配
      return false;  // 返回假
    }
  }
  return true;  // 返回真，所有步长匹配
}

// 将张量的形状转换为字符串
inline std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;  // 创建输出流对象
  bool first = true;  // 标记是否第一个张量
  for (auto& tensor : tensors) {  // 遍历张量列表
    if (tensor.defined()) {  // 如果张量已定义
      if (!first) {  // 如果不是第一个张量
        os << ", ";  // 输出逗号分隔符
      }
      os << tensor.sizes();  // 输出张量的尺寸
      first = false;  // 将第一个张量标记设置为假
    }
  }
  return os.str();  // 返回形状字符串
}

// 重新调整源张量的形状以支持高级索引操作
static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed,
                           IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());  // 获取源张量的尺寸
  auto strides = DimVector(src.strides());  // 获取源张量的步长
  int64_t end = dims_before + dims_indexed;  // 计算索引维度的结束位置
  shape.erase(shape.begin() + dims_before, shape.begin() + end);  // 删除索引维度的尺寸
  strides.erase(strides.begin() + dims_before, strides.begin() + end);  // 删除索引维度的步长
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());  // 插入替换后的形状
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);  // 插入步长为零的索引维度
  return src.as_strided(shape, strides);  // 返回重新调整形状后的张量
}

// 将索引张量的形状重塑为可以与结果张量和重新调整后的源张量广播的形状
static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();  // 获取索引张量的原始形状
  auto shape = DimVector();  // 创建形状向量
  shape.append(dims_before, 1);  // 添加前置维度为1
  shape.append(orig_shape.begin(), orig_shape.end());  // 添加索引张量的原始形状
  shape.append(dims_after, 1);  // 添加后置维度为1
  return index.reshape(shape);  // 返回重塑后的索引张量
}

// 构造函数，用于高级索引操作
AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list)
{
  // 计算源张量元素的字节大小
  int64_t element_size_bytes = src.element_size();
  // 初始化计数器：dims_before 表示索引前的维度数，dims_after 表示索引后的维度数，dims_indexed 表示索引的维度数
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  // 用于存储替换形状的 IntArrayRef
  IntArrayRef replacement_shape;
  // 遍历索引列表中的每个维度
  for (const auto dim : c10::irange(indices_list.size())) {
    // 如果当前维度没有定义索引
    if (!indices_list[dim].defined()) {
      // 如果是第一个未定义的维度
      if (dims_indexed == 0) {
        dims_before++;  // 增加索引前的维度计数
      } else {
        dims_after++;   // 增加索引后的维度计数
      }
    } else {
      dims_indexed++;  // 增加索引的维度计数
      replacement_shape = indices_list[dim].sizes();  // 获取替换的形状
      indexed_sizes.push_back(src.size(dim));  // 记录索引的维度大小
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);  // 记录索引的维度步长
    }
  }

  // 检查索引子空间中是否包含大小为 0 的维度，但替换形状中没有。这意味着索引越界，因为空张量没有有效的索引。
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  // 更新对象成员变量 dims_before 和 dims_after
  this->dims_before = dims_before;
  this->dims_after = dims_after;
  // 使用 restride_src 函数对源张量进行重新调整，根据索引情况进行调整
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  // 对每个定义了索引的张量进行形状重塑，添加到 indices 列表中
  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // 对于 CUDA/MPS 张量，强制所有索引张量具有相同的步长，以简化 CUDA/MPS 核心的操作
  if (indices.size() >= 2 && (this->src.device().type() == kCUDA || this->src.device().type() == kMPS)) {
    if (!all_strides_match(indices)) {
      for (auto & indice : indices) {
        indice = indice.contiguous();  // 强制索引张量为连续的
      }
    }
  }
}

// 创建并返回一个 TensorIterator 对象，用于执行索引赋值操作
static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  // 检查值张量是否可以广播到索引结果的形状
  TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  // 检查值张量和源张量的数据类型是否匹配
  TORCH_CHECK(value.scalar_type() == info.src.scalar_type(),
              "Index put requires the source and destination dtypes match, "
              "got ", info.src.scalar_type(), " for the destination "
              "and ", value.scalar_type(), " for the source.");
  // 配置 TensorIteratorConfig 对象
  TensorIteratorConfig config;
  config.set_check_mem_overlap(false);  // 禁止内存重叠检查
  config.resize_outputs(false);  // 不调整输出大小
  config.check_all_same_dtype(false);  // 不检查所有张量的数据类型是否相同
  config.add_output(info.src);  // 添加输出张量
  config.add_const_input(value);  // 添加常量输入张量 value
  for (auto& index : info.indices) {
    config.add_const_input(index);  // 添加索引张量作为常量输入
  }
  return config.build();  // 构建并返回 TensorIterator 对象
}

// 实现索引赋值的函数，对应于 Torch 库的 index_out 操作
TORCH_IMPL_FUNC(index_out)
(const Tensor& self,
 DimVector sizes,
 DimVector strides,
 const Tensor& result) {
  index_stub(device_type(), *this, sizes, strides);  // 调用底层的 index_stub 函数进行索引操作
}
// 返回类型为 Tensor 的函数，接受一个常量引用 self 和一个索引列表 indices
Tensor quantized_index(const Tensor & self, const torch::List<std::optional<Tensor>>& indices) {
  // 内部断言，检查 self 的量化方案是否为每张量仿射或对称，否则抛出异常
  TORCH_INTERNAL_ASSERT(
      self.qscheme() == c10::kPerTensorAffine ||
      self.qscheme() == c10::kPerTensorSymmetric,
      "Indexing is only supported for per-Tensor quantized Tensors.");

  // 目前的简单实现：先去量化，再进行索引操作，最后再量化回去。
  // TODO(未来的PR)：通过消除拷贝来提升性能。
  // 获取 self 的去量化版本
  const auto& self_dq = self.dequantize();
  // 对 self_dq 进行索引操作，使用 indices
  auto result = at::index(self_dq, indices);
  // 对结果 result 进行张量的每张量量化
  return at::quantize_per_tensor(
      result, self.q_scale(), self.q_zero_point(), self.scalar_type());
}

// 返回类型为 Tensor 的函数，接受一个常量引用 self 和一个索引列表 indices
Tensor _unsafe_index(const Tensor& self, const torch::List<std::optional<Tensor>>& indices) {
  // 禁止布尔类型索引，因为它会导致动态输出形状
  for (auto i : c10::irange(indices.size())) {
    auto index = indices.get(i);
    if (index.has_value()) {
      auto dtype = index->scalar_type();
      // 检查索引类型是否为 kLong 或 kInt
      TORCH_CHECK(dtype == kLong || dtype == kInt,
                  "_unsafe_index found unexpected index type ", dtype);
    }
  }
  // 对 self 进行索引操作，使用 indices
  return at::index(self, indices);
}

// 返回类型为 Tensor 的函数，接受一个常量引用 self，一个掩码 mask，一个索引列表 indices 和一个标量 fill
Tensor _unsafe_masked_index(const Tensor& self, const Tensor& mask, const torch::List<c10::optional<Tensor>>& indices, const Scalar& fill) {
  // 不安全的掩码索引等同于 where(mask, self[indices], fill)
  // 区别在于当 mask 为 false 时，不会使用 indices 对 self 进行索引操作。
  // 这允许 indices 超出边界。当 mask 为 true 时，预期 indices 在边界内，不做检查。
  //
  // 此函数不适用于急切模式。这里提供了一个未经优化的版本。
  //
  // 编译器后端应实现此操作，以便在 mask 为 true 时不加载 self[indices]。参见归纳器作为参考。

  // clamp 函数，对 index 进行截断处理，确保类型为 kLong 或 kInt
  auto clamp = [](const c10::optional<Tensor>& index, auto size) -> c10::optional<Tensor> {
    if (!index) {
      return index;
    }
    // 禁止布尔类型
    auto dtype = index->scalar_type();
    TORCH_CHECK(dtype == kLong || dtype == kInt,
                "_unsafe_masked_index found unexpected index type ", dtype);
    // 返回索引值在指定范围内的结果，使用 at::clamp 进行限制
    return at::clamp(*index, -size, size - 1);
  };

  // 创建一个列表，存放索引的可选张量
  torch::List<c10::optional<Tensor>> clamped_indices(indices);
  // 使用 clamp 函数对索引进行限制，将结果存放在 clamped_indices 中
  std::transform(indices.begin(), indices.end(), self.sizes().begin(), clamped_indices.begin(), clamp);

  // 如果张量 self 中元素数目为 0
  if (self.numel() == 0) {
      // 返回一个填充了 fill 值的张量
      // 由于没有方法直接获取张量的正确大小（除非使用不适用于移动构建的 meta 实现），因此在这里使用了一个技巧
      std::vector<int64_t> new_sizes(self.dim());
      // 定义计算新大小的函数，根据索引和当前大小计算新的维度大小
      auto compute_new_size = [](const c10::optional<Tensor>& index, auto size) -> int64_t {
          if (index && size == 0) {
              return 1;
          } else {
              return size;
          }
      };
      // 使用 compute_new_size 函数计算新的维度大小，存放在 new_sizes 中
      std::transform(indices.begin(), indices.end(), self.sizes().begin(), new_sizes.begin(), compute_new_size);
      // 使用 new_sizes 和 fill 值创建一个新的填充张量
      auto result = self.new_full(new_sizes, fill);
      // 返回在 result 上进行不安全索引操作后的结果，使用 clamped_indices 作为索引
      return at::_unsafe_index(result, clamped_indices);
  }

  // 在 self 上进行不安全索引操作，使用 clamped_indices 作为索引
  auto result = at::_unsafe_index(self, clamped_indices);
  // 使用 mask 对结果进行逻辑非操作，将结果中的 False 值用 fill 填充
  return result.masked_fill(at::logical_not(mask), fill);
}

// 不安全的索引放置累积操作的反向操作函数。
// 此函数不应在急切模式下执行。

Tensor _unsafe_masked_index_put_accumulate(const Tensor& self, const Tensor& mask, const torch::List<c10::optional<Tensor>>& indices, const Tensor& values) {
  // 如果 self 张量的元素数为零，则返回其克隆
  if (self.numel() == 0) {
    return self.clone();
  }

  // 定义一个闭包函数 clamp，重新计算索引并依赖于 CSE 来共享计算结果
  auto clamp = [](const c10::optional<Tensor>& index, auto size) -> c10::optional<Tensor> {
    // 如果索引为空，则直接返回
    if (!index) {
      return index;
    }
    // 禁止布尔类型的索引
    auto dtype = index->scalar_type();
    TORCH_CHECK(dtype == kLong || dtype == kInt,
                "_unsafe_masked_index found unexpected index type ", dtype);
    // 将索引限制在 -size 和 size - 1 之间
    return at::clamp(*index, -size, size - 1);
  };

  // 使用 clamp 函数对 indices 进行重新计算
  torch::List<c10::optional<Tensor>> clamped_indices(indices);
  std::transform(indices.begin(), indices.end(), self.sizes().begin(), clamped_indices.begin(), clamp);

  // 根据 mask 对 values 进行遮蔽，将未被遮蔽的部分填充为 0
  auto masked_value = values.masked_fill(at::logical_not(mask), 0);
  // 调用底层的 _unsafe_index_put 函数进行索引放置操作
  return at::_unsafe_index_put(self, clamped_indices, masked_value, true);
}

// 在自身张量的给定索引处放置 source 张量的值
Tensor & put_(Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
  // 参见注释 [Writing Nondeterministic Operations]
  // 当索引包含重复条目且我们不累积时，操作是不确定性的
  // 如果在 GPU 上累积，我们使用 atomicGPUAdd，这是不确定性的

  if (!accumulate || (accumulate && self.device().type() == DeviceType::CUDA)) {
    // 如果不累积或在 CUDA 设备上累积，则发出警告
    at::globalContext().alertNotDeterministic("put_");
  }

  // 类型和设备检查
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "put_(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == source.scalar_type(), "put_(): self and source expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and source.dtype = ", source.scalar_type());
  TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
      "put_(): self, index and source expected to be in the same device, but got self.device = ",
      self.device(), ", index.device = ", index.device(), ", and source.device = ", source.device());

  // 索引检查
  TORCH_CHECK_INDEX(source.numel() == index.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ", source.numel(), ", index.numel() = ", index.numel());
  TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "put_(): Tried to put elements into an empty tensor");

  // 确保没有内部重叠
  at::assert_no_internal_overlap(self);
  // 确保 self 和 index 之间没有重叠
  at::assert_no_overlap(self, index);
  // 确保 self 和 source 之间没有重叠
  at::assert_no_overlap(self, source);

  // 如果索引的元素数为零，则提前返回
  if (index.numel() == 0) {
    return self;
  }

  // 将 index 重塑为与 source 相同的大小
  auto index_reshaped = index.reshape(source.sizes());
  // 不要迭代 self，我们将手动计算偏移量
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_const_input(source);
    .add_const_input(index_reshaped)
    .build();



# 添加经常不变的输入（使用 index_reshaped），然后构建计算图
.add_const_input(index_reshaped)
.build();


```  
  put_stub(iter.device_type(), iter, self, accumulate);




# 使用 put_stub 函数将 iter 对象放入计算图中
put_stub(iter.device_type(), iter, self, accumulate);


```  
  return self;



# 返回当前对象自身，通常用于链式调用
return self;
}

// 在给定的张量上执行类似字典操作的索引赋值，返回修改后的张量
Tensor put(const Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
  // 使用保留内存格式克隆张量，避免原地操作带来的影响
  return self.clone(at::MemoryFormat::Preserve).put_(index, source, accumulate);
}

// 在给定的张量上执行索引赋值，支持多个索引，返回修改后的张量
Tensor index_put(const Tensor & self, const torch::List<std::optional<Tensor>>& indices, const Tensor & value, bool accumulate) {
  // 使用保留内存格式克隆张量，避免原地操作带来的影响
  return self.clone(at::MemoryFormat::Preserve).index_put_(indices, value, accumulate);
}

// 在给定的张量上执行不安全的索引赋值，返回修改后的张量
Tensor _unsafe_index_put(const Tensor& self, const torch::List<std::optional<Tensor>>& indices, const Tensor& value, bool accumulate) {
  // 调用ATen库的不安全索引赋值函数
  return at::index_put(self, indices, value, accumulate);
}

// 在给定的张量上执行索引赋值的实现，支持多个索引和累加操作，可能会修改原张量
Tensor & _index_put_impl_(Tensor & self, const torch::List<std::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  // 检查索引数量是否超过张量维度的合法性
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  // 检查张量是否存在内部重叠，若存在则发出警告
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  // 如果不进行累加操作，检查是否可以优化为masked_fill_操作
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  // 如果value张量的设备与self不同，并且value是标量，则将value转移到self设备上
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  // 确保self和value张量没有重叠
  at::assert_no_overlap(self, value);
  // 依次检查indices中的每个索引，确保与self没有重叠
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const std::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  // 如果self在CUDA设备上，并且要求累加或使用确定性算法，则调用排序后的索引赋值函数
  if (self.device().type() == DeviceType::CUDA && (accumulate || globalContext().deterministicAlgorithms())) {
      TORCH_CHECK(value_.device() == self.device(), "expected device ", self.device(), " but got device ",
      value_.device(), " for value tensor");
      index_put_with_sort_stub(self.device().type(), self, indices, value_, accumulate, unsafe);
      return self;
  }

  // 否则，根据索引和值创建索引赋值的迭代器，并调用对应的底层函数执行赋值操作
  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}
// 从输入张量 `self` 中取出索引 `index` 指定的元素，将结果存入输出张量 `out` 中
Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // 类型和设备检查
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "take(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == out.scalar_type(), "take(): self and out expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and out.dtype = ", out.scalar_type());
  TORCH_CHECK(self.device() == out.device() && self.device() == index.device(),
      "take(): self, index and out expected to be in the same device, but got self.device = ",
      self.device(), ", index.device = ", index.device(), ", and out.device = ", out.device());

  // 索引检查
  TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");

  // 确保输出张量 `out` 不重叠
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, self);

  // 不需要遍历 `self`，我们将手动计算偏移量
  // `out` 将在 tensor_iterator 中被重新调整大小
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_output(out)
    .add_const_input(index)
    .build();

  // 在 `out` 被调整大小后立即返回
  if (index.numel() == 0) {
    return out;
  }

  // 调用实际的 take 函数实现
  take_stub(iter.device_type(), iter, self);

  return out;
}

// 从输入张量 `self` 中取出索引 `index` 指定的元素，返回结果作为新的张量
Tensor take(const Tensor& self, const Tensor& index) {
    // 创建一个与 `index` 相同大小的空张量 `out`，并使用与 `self` 相同的选项
    auto out = at::empty(index.sizes(), self.options());
    // 调用 take_out 函数，将结果存入 `out` 中
    at::native::take_out(self, index, out);
    // 返回结果张量 `out`
    return out;
}

// 在 `self` 上执行索引操作，根据 `indices` 指定的位置将 `value` 放置到 `self` 中
Tensor & index_put_(Tensor & self, const torch::List<std::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  // 调用内部的 _index_put_impl_ 函数实现索引操作
  return at::_index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
}

// index_copy_out 函数的 Torch 实现
TORCH_IMPL_FUNC(index_copy_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Tensor& result) {
    // 如果 `result` 不是 `self`，则将 `self` 的内容复制到 `result` 中
    if (!result.is_same(self)) result.copy_(self);

    // 检查是否启用了确定性操作
    if (result.is_cuda() && globalContext().deterministicAlgorithms()){
        // 构建用于索引操作的索引列表 `indices`
        torch::List<std::optional<Tensor>> indices;
        indices.reserve(dim + 1);
        for (const auto i: c10::irange(dim)) {
          (void)i;
          indices.emplace_back();
        }
        indices.emplace_back(index);
        // 在 `result` 上执行索引操作，将 `source` 放置到指定位置
        result.index_put_(indices, source, false);
        return;
    }

    // 处理当 `self` 或 `source` 是零维的情况
    Tensor result_nonzero = result.dim() == 0 ? result.unsqueeze(0) : result;
    Tensor source_nonzero = source.dim() == 0 ? source.unsqueeze(0) : source;

    // 准备用于 TensorIterator 的 `index`
    // 保证其在 TensorIterator 中可广播到 `self`
    auto index_sizes = std::vector<int64_t>(result_nonzero.dim(), 1);
    // 创建一个长度为 result_nonzero 的维度数的向量，所有元素初始化为 0
    auto index_strides = std::vector<int64_t>(result_nonzero.dim(), 0);
    // 将 index 的元素数目设置为 index_sizes 的 dim 维度
    index_sizes[dim] = index.numel();
    // 如果 index 是一维数组或标量，将 index_strides 的 dim 维度设置为 index 的步长；否则设置为 1
    index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
    // 将 index 按照 index_sizes 和 index_strides 进行重新排列
    auto index_restrided = index.as_strided(index_sizes, index_strides);

    // 为 TensorIterator 准备 result
    // 将 result_nonzero 在 dim 维度上重新排列，使其不在该维度上前进
    // 这里不使用 squash_dim 是因为 index 在这个维度上需要前进
    // 注意，self_sizes[dim] 被设置为 index 的元素数目
    auto result_sizes = result_nonzero.sizes().vec();
    auto result_strides = result_nonzero.strides().vec();
    result_sizes[dim] = index.numel();
    result_strides[dim] = 0;
    auto result_restrided = result_nonzero.as_strided(result_sizes, result_strides);

    // 创建 TensorIteratorConfig 对象 iter
    auto iter = TensorIteratorConfig()
      // 不检查内存重叠，因为 result 被重新排列为零步长
      .set_check_mem_overlap(false)
      // 不检查所有输入是否具有相同的数据类型
      .check_all_same_dtype(false)
      // 不调整输出的大小
      .resize_outputs(false)
      // 将 result_restrided 添加为输出
      .add_output(result_restrided)
      // 将 index_restrided 添加为常量输入
      .add_const_input(index_restrided)
      // 将 source_nonzero 添加为常量输入
      .add_const_input(source_nonzero)
      // 构建 TensorIterator
      .build();

    // 获取 result_nonzero 在 dim 维度上的大小和步长
    auto result_dim_size = result_nonzero.size(dim);
    auto result_dim_stride = result_nonzero.stride(dim);
    // 使用 index_copy_stub 处理 TensorIterator iter
    index_copy_stub(
      iter.device_type(),
      iter,
      dim,
      result_dim_size,
      result_dim_stride);
// 结束函数的定义，此处为函数实现结束的大括号
}

// 由于不同的数据类型分发，没有调用到 index_reduce_func_impl
TORCH_IMPL_FUNC(index_add_cpu_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  // 如果 result 不是 self 本身，则将 self 的内容复制到 result
  if (!result.is_same(self)) {
     result.copy_(self);
  }
  // 获取索引张量的元素个数
  auto numel = index.numel();

  // 将索引张量转换为连续的张量
  auto index_contig = index.contiguous();

  // 如果 result 张量的维度大于 1
  if (result.dim() > 1) {
    // 如果索引张量的元素个数为 0，或者 self 张量的元素个数为 0，则直接返回
    if (numel == 0 || self.numel() == 0) {
      return;
    }

    // 对维度 dim 进行可能的包装，确保在有效范围内
    dim = maybe_wrap_dim(dim, self.dim());

    // 当 source 或 result 的切片是非连续的时，
    // 原始的 index_add 操作较慢，因为它使用对切片张量的串行加法，
    // 这会导致在索引上并行，在切片张量上避免写冲突。
    // 对切片张量进行并行操作并不是最优的，因为切片张量的大小可能不足以并行，
    // 同时会导致多重并行化。为了加速这种情况，使用 scatter_add。
    // scatter_add 在输入的外部维度上并行，并在内部维度上串行，以避免写冲突。
    // scatter_add 只需要一次并行操作，外部维度的大小足够大可以进行并行操作。

    if ((dim == 0 || dim == self.dim() - 1) &&
        // 索引的数据类型应为 long，alpha 应为 1 才能使用 scatter_add
        alpha.equal(1.0) && index_contig.scalar_type() == ScalarType::Long &&
        // scatter_add 不支持 ComplexHalf 类型
        source.scalar_type() != ScalarType::ComplexHalf &&
        result.scalar_type() != ScalarType::ComplexHalf) {
      // 创建用于存储大小和步长的向量
      std::vector<int64_t> ep_sizes(result.sizes().size());
      std::vector<int64_t> ep_strides(source.sizes().size());

      // 检查 result 和 source 在维度 dim 以外的匹配性
      auto check_sizes = [&ep_sizes, &ep_strides, &numel](IntArrayRef a, IntArrayRef b, int64_t dim) -> bool {

        ep_sizes[dim] = numel;
        ep_strides[dim] = 1;
        // 遍历维度，检查是否匹配
        for (const int64_t i : c10::irange(a.size())) {
          if (i == dim) {
            continue;
          }

          // 如果维度大小不匹配，则返回 false
          if (a[i] != b[i]) {
            return false;
          }
          ep_sizes[i] = a[i];
          ep_strides[i] = 0;

        }
        return true;
      };

      // 如果大小匹配，则创建扩展索引并进行 scatter_add 操作
      if (check_sizes(result.sizes(), source.sizes(), dim)) {
        auto ep_index = index_contig.as_strided(ep_sizes, ep_strides);
        result.scatter_add_(dim, ep_index, source);
        return;
      }
    }

    // 如果上述条件不满足，则获取 result 在维度 dim 上的第一个切片
    auto selfSlice = result.select(dim, 0);
    // 从源张量中选择特定维度的切片
    auto sourceSlice = source.select(dim, 0);
    // 计算结果张量在指定维度上的步长字节数
    auto self_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    // 计算源张量在指定维度上的步长字节数
    auto source_stride_bytes = source.stride(dim) * elementSize(source.scalar_type());
    // 获取结果张量在指定维度上的大小
    auto self_dim_size = result.size(dim);
    // 创建一个 TensorIterator 对象，用于操作两个张量的二进制操作
    auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);

    // 根据索引类型分发函数 index_add_cpu_
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cpu_", [&] () {
      // 获取索引数据的指针
      auto index_data = index_contig.const_data_ptr<index_t>();
      // 遍历索引范围内的元素个数
      for (const auto i : c10::irange(numel)) {
          // 获取当前索引值
          auto self_i = index_data[i];
          // 检查索引是否超出结果张量的有效范围
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          // 计算结果张量中对应索引位置的数据指针
          auto self_data = static_cast<char*>(selfSlice.data_ptr()) + self_i * self_stride_bytes;
          // 计算源张量中对应索引位置的常量数据指针
          auto source_data = static_cast<const char*>(sourceSlice.const_data_ptr()) + i * source_stride_bytes;
          // 替换 TensorIterator 对象中的操作数
          iter.unsafe_replace_operand(0, self_data);
          iter.unsafe_replace_operand(1, self_data);
          iter.unsafe_replace_operand(2, const_cast<char*>(source_data));
          // 调用 add_stub 执行张量操作
          add_stub(iter.device_type(), iter, alpha);
      }
    });
  } else {
    // 检查条件：source 张量的维度必须为一维或零维
    TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

    // 显式捕获所有必要变量以解决 Windows 构建问题
    // TODO: 当 Windows 正确捕获嵌套 lambda 中的变量时修复此问题
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, ScalarType::ComplexHalf,
      result.scalar_type(), "index_add_", [&result, &source, &dim, &index_contig, &numel, &alpha] {
      // 将 alpha 转换为当前标量类型
      auto alpha_value = alpha.to<scalar_t>();
      // 计算结果张量在指定维度上的步长
      auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
      // 计算源张量在指定维度上的步长
      auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
      // 获取结果张量的数据指针
      auto* result_ptr = result.data_ptr<scalar_t>();
      // 获取源张量的常量数据指针
      auto* source_ptr = source.const_data_ptr<scalar_t>();
      // 根据索引类型分发函数 index_add_cpu_
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_add_cpu_",
        [&index_contig, &numel, &result, &result_ptr, &result_stride, &source_ptr, &source_stride, &alpha_value] {
        // 获取索引数据的指针
        auto index_data = index_contig.const_data_ptr<index_t>();
        // 遍历索引范围内的元素个数
        for (const auto i : c10::irange(numel)) {
            // 获取当前索引值
            auto self_i = index_data[i];
            // 检查索引是否超出结果张量的有效范围
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < result.numel()), "index out of range in self");
            // 计算结果张量中对应索引位置的数据指针
            scalar_t *self_ip = result_ptr + self_i * result_stride;
            // 执行索引加法操作，更新结果张量数据
            *self_ip += *(source_ptr + i * source_stride) * alpha_value;
        }
      });
    });
  }
// 实现一个静态函数，用于对张量 self 进行索引约简操作，dim 表示约简的维度，index 是索引张量，source 是源张量，
// include_self 表示是否包含 self，result 是结果张量，op 是约简操作的类型
static void index_reduce_func_impl(
  const Tensor& self,  // 输入张量 self
  int64_t dim,          // 约简操作的维度
  const Tensor& index,  // 索引张量
  const Tensor& source, // 源张量
  bool include_self,    // 是否包含 self
  const Tensor& result, // 结果张量
  const ReductionType& op) { // 约简操作的类型
  // 如果 result 不是 self 自身，则将 self 的内容复制到 result 中
  if (!result.is_same(self)) result.copy_(self);
  
  // 如果不包含 self
  if (!include_self) {
    // 根据不同的约简类型初始化初始值 init_val
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "index_reduce_func_exclude_input_init", [&] {
        scalar_t init_val;
        switch (op) {
          case ReductionType::PROD:
            init_val = (scalar_t)1; // PROD 约简初始值为 1
            break;
          case ReductionType::MAX:
            // MAX 约简初始值为负无穷
            init_val = std::numeric_limits<scalar_t>::has_infinity ? -std::numeric_limits<scalar_t>::infinity()
                       : std::numeric_limits<scalar_t>::lowest();
            break;
          case ReductionType::MIN:
            // MIN 约简初始值为正无穷
            init_val = std::numeric_limits<scalar_t>::has_infinity ? std::numeric_limits<scalar_t>::infinity()
                       : std::numeric_limits<scalar_t>::max();
            break;
          default:
            init_val = (scalar_t)0; // 默认初始值为 0
            break;
        }
        // 使用初始值 init_val 对 result 进行索引填充操作，index 需要是 LongTensor 类型
        result.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
    });
  }

  // 计算索引张量的元素个数
  auto numel = index.numel();

  // 将索引张量转换为连续存储
  auto index_contig = index.contiguous();

  // 如果结果张量的维度大于 1
  if (result.dim() > 1) {
    // 如果 numel 为 0，则直接返回
    if (numel == 0) {
      return;
    }
    // 获取结果张量在 dim 维度上的第一个切片
    auto selfSlice = result.select(dim, 0);
    // 获取源张量在 dim 维度上的第一个切片
    auto sourceSlice = source.select(dim, 0);
    // 计算结果张量在 dim 维度上的步长（字节数）
    auto self_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    // 计算源张量在 dim 维度上的步长（字节数）
    auto source_stride_bytes = source.stride(dim) * elementSize(source.scalar_type());
    // 获取结果张量在 dim 维度上的大小
    auto self_dim_size = result.size(dim);
    // 使用二进制操作符的迭代器进行张量迭代，这里是二进制操作 selfSlice.op_(sourceSlice) 的重用
    auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_func_cpu_", [&] () {
      // 根据索引类型分派处理器，并定义 Lambda 函数
      auto index_data = index_contig.const_data_ptr<index_t>();
      // 获取索引数据的常量指针
      for (const auto i : c10::irange(numel)) {
        // 遍历索引范围
        auto self_i = index_data[i];
        // 获取当前索引值
        TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
        // 检查索引范围是否合法
        auto self_data = static_cast<char*>(selfSlice.data_ptr()) + self_i * self_stride_bytes;
        // 计算当前 selfSlice 数据的偏移量
        auto source_data = static_cast<const char*>(sourceSlice.const_data_ptr()) + i * source_stride_bytes;
        // 计算当前 sourceSlice 数据的偏移量
        iter.unsafe_replace_operand(0, self_data);
        // 替换操作数 0 为 self_data
        iter.unsafe_replace_operand(1, self_data);
        // 替换操作数 1 为 self_data
        iter.unsafe_replace_operand(2, const_cast<char*>(source_data));
        // 替换操作数 2 为 source_data

        switch (op) {
          case ReductionType::PROD :
            // 如果操作类型为 PROD，调用 mul_stub 函数
            mul_stub(iter.device_type(), iter);
            break;
          case ReductionType::MIN :
            // 如果操作类型为 MIN，调用 minimum_stub 函数
            minimum_stub(iter.device_type(), iter);
            break;
          case ReductionType::MAX :
            // 如果操作类型为 MAX，调用 maximum_stub 函数
            maximum_stub(iter.device_type(), iter);
            break;
          default :
            // 默认情况下，调用 add_stub 函数，参数为 iter 和 1
            add_stub(iter.device_type(), iter, 1);
            break;
        }
      }
    });

    if (op == ReductionType::MEAN) {
      // 如果操作类型为 MEAN
      auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
      // 根据 include_self 判断是否使用 ones_like 或 zeros_like 函数生成 counts
      counts.index_add_(dim, index, at::ones_like(source));
      // 对 counts 使用 index_add_ 函数，参数包括 dim、index 和 ones_like(source)
      counts.masked_fill_(counts == 0, 1);
      // 使用 masked_fill_ 函数，将 counts 中值为 0 的位置填充为 1
      if (result.is_floating_point() || result.is_complex()) {
        // 如果 result 是浮点数或复数类型
        result.div_(counts);
        // 使用 div_ 函数对 result 进行除法运算，参数为 counts
      } else {
        // 否则
        result.div_(counts, "floor");
        // 使用 div_ 函数对 result 进行除法运算，参数为 counts 和 "floor"
      }
    }
  }
  else {
    TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
    // 检查条件：source 的维度必须小于等于 1
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    // 根据 include_self 判断是否使用 ones_like 或 zeros_like 函数生成 counts
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
      result.scalar_type(), "index_func_", [&result, &source, &dim, &index_contig, &numel, &op, &counts] {
      // 获取结果张量在指定维度上的步长，如果维度为0，则步长为1
      auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
      // 获取源张量在指定维度上的步长，如果维度为0，则步长为1
      auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
      // 获取计数张量在指定维度上的步长，如果维度为0，则步长为1
      auto counts_stride = counts.dim() == 0 ? 1 : counts.stride(dim);
      // TODO: Maybe TensorAccessor can be used here?
      // 获取结果张量的数据指针，类型为当前标量类型的指针
      auto* result_ptr = result.data_ptr<scalar_t>();
      // 获取源张量的常量数据指针，类型为当前标量类型的常量指针
      auto* source_ptr = source.const_data_ptr<scalar_t>();
      // 获取计数张量的数据指针，类型为当前标量类型的指针
      auto counts_ptr = counts.data_ptr<scalar_t>();
      // 根据索引类型处理连续索引，命名为"index_func_cpu_"
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_func_cpu_",
        [&index_contig, &numel, &result, &result_ptr, &result_stride, &source_ptr, &source_stride, &op, &counts_ptr, &counts_stride] {
        // 获取连续索引数据的常量数据指针，类型为当前索引类型的常量指针
        auto index_data = index_contig.const_data_ptr<index_t>();
        // 遍历索引范围内的每个元素
        for (const auto i : c10::irange(numel)) {
            // 获取当前索引值
            auto self_i = index_data[i];
            // 检查索引值是否在结果张量范围内
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < result.numel()), "index out of range in self");
            // 指向结果张量中当前位置的指针
            scalar_t *self_ip = result_ptr + self_i * result_stride;
            // 指向计数张量中当前位置的指针
            scalar_t *count_ip;
            scalar_t val;
            // 根据操作类型选择操作
            switch (op) {
              // 求均值操作
              case ReductionType::MEAN :
                // 在结果张量中对应位置累加源张量中的值
                *self_ip += *(source_ptr + i * source_stride);
                // 在计数张量中对应位置增加计数
                count_ip = counts_ptr + self_i * counts_stride;
                *count_ip += 1;
                break;
              // 求积操作
              case ReductionType::PROD :
                // 在结果张量中对应位置乘以源张量中的值
                *self_ip *= *(source_ptr + i * source_stride);
                break;
              // 求最小值操作
              case ReductionType::MIN :
                // 获取源张量中的值
                val = *(source_ptr + i * source_stride);
                // 如果值为 NaN，则直接赋值，否则取最小值
                *self_ip = at::_isnan<scalar_t>(val) ? val : std::min(*self_ip, val);
                break;
              // 求最大值操作
              case ReductionType::MAX :
                // 获取源张量中的值
                val = *(source_ptr + i * source_stride);
                // 如果值为 NaN，则直接赋值，否则取最大值
                *self_ip = at::_isnan<scalar_t>(val) ? val : std::max(*self_ip, val);
                break;
              default:
                break;
            }
        }
      });
    });
    // 如果操作为均值操作
    if (op == ReductionType::MEAN) {
      // 将计数张量中为0的位置填充为1
      counts.masked_fill_(counts == 0, 1);
      // 如果结果张量为浮点型或复数型，则对结果张量进行除法操作
      if (result.is_floating_point() || result.is_complex()) {
        result.div_(counts);
      } else {
        // 否则使用 floor 操作对结果张量进行除法操作
        result.div_(counts, "floor");
      }
    }
  }
}

// 实现 CPU 索引缩减的函数，将结果输出到预分配的 Tensor
TORCH_IMPL_FUNC(index_reduce_cpu_out)
(const Tensor& self,  // 输入张量
 int64_t dim,  // 索引的维度
 const Tensor& index,  // 索引张量
 const Tensor& source,  // 源张量
 const c10::string_view reduce,  // 缩减操作类型
 bool include_input,  // 是否包含输入
 const Tensor& result) {  // 输出张量
  TORCH_WARN_ONCE("index_reduce() is in beta and the API may change at any time.");  // 发出警告，函数处于 beta 阶段，API 可能随时更改
  auto op = get_operator_enum(reduce, true);  // 获取操作枚举
  index_reduce_func_impl(self, dim, index, source, include_input, result, op);  // 调用实现索引缩减的函数
}

// 检查索引数组中的索引是否落在维度大小范围内
// 避免重新调度调用最小值/最大值函数
template <typename IndexType>
static void check_indexarray_range(
    const IndexType* indices,  // 索引数组
    int64_t n,  // 数组长度
    IndexType indexing_axis_dim) {  // 索引维度大小
  for (const auto i : c10::irange(n)) {
    auto idx = indices[i];  // 获取当前索引值
    TORCH_CHECK(
        0 <= idx && idx < indexing_axis_dim,  // 检查索引值是否在范围内
        "INDICES element is out of DATA bounds, id=",  // 错误消息：索引元素超出数据范围，id=
        idx,  // 输出索引值
        " axis_dim=",  // 错误消息：轴维度=
        indexing_axis_dim);  // 输出索引维度大小
  }
}

// 在维度为1的情况下实现 CPU 索引选择操作，输出到预分配的 Tensor
static Tensor & index_select_out_cpu_dim1_(
    Tensor & result_contig,  // 输出结果张量（连续存储）
    const Tensor & self,  // 输入张量
    const Tensor & index_contig) {  // 索引张量（连续存储）

  auto self_contig = self.contiguous();  // 确保输入张量连续
  const caffe2::TypeMeta dataType = self_contig.dtype();  // 获取数据类型
  size_t item_bytesize = dataType.itemsize();  // 获取每个元素的字节大小

  auto out = static_cast<char*>(result_contig.data_ptr());  // 输出结果的数据指针

  auto src_base = static_cast<const char*>(self_contig.const_data_ptr());  // 输入张量的常量数据指针

  auto self_sizes = self_contig.sizes();  // 输入张量的尺寸
  auto outer_dims_product = c10::size_to_dim_(1, self_sizes);  // 外部维度的乘积
  auto block_size = c10::size_from_dim_(2, self_sizes);  // 块的大小
  auto block_bytesize = block_size * item_bytesize;  // 块的字节大小

  auto src_indexing_axis_dim = self_sizes[1];  // 输入张量在索引轴上的维度大小
  auto src_batch_bytesize = self_sizes[1] * block_bytesize;  // 输入张量批次的字节大小
  auto N = index_contig.numel();  // 索引张量中的元素数量

  auto gathered_batch_bytesize = N * block_bytesize;  // 被收集的批次的字节大小

  AT_DISPATCH_INDEX_TYPES(
    index_contig.scalar_type(), "batch_index_select_compute", [&]() {
      // 获取索引数组的指针
      const auto* idxs = index_contig.const_data_ptr<index_t>();
      // 检查索引数组的有效范围是否在合理范围内
      check_indexarray_range<index_t>(idxs, N, src_indexing_axis_dim);

      // 对于单精度浮点数且块大小为1的特殊情况，采用高效的单浮点数复制
      if (self.scalar_type() == ScalarType::Float && block_size == 1) {
        // 遍历所有批次的外部维度
        for (const auto batch : c10::irange(outer_dims_product)) {
          // 获取当前批次的源浮点数指针和目标浮点数指针
          const float* src_floats = (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

          // 根据索引数组将源数据复制到目标位置
          for (const auto i : c10::irange(N)) {
            auto idx = idxs[i];
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product 指定了内部维度重复的次数，因此迭代它以覆盖所有外部维度
        for (const auto batch : c10::irange(outer_dims_product)) {
          // 根据索引数组和块大小计算源和目标地址，并使用 memcpy 复制数据块
          for (const auto i : c10::irange(N)) {
            auto idx = idxs[i];
            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            memcpy(dst, src, block_bytesize);
          }
        }
      }
  });
  // 返回结果张量的连续化视图
  return result_contig;
  // 如果输入张量是量化的，则进行量化校验
  if (self.is_quantized()) {
    TORCH_CHECK(
        self.qscheme() == kPerTensorAffine,
        "Only per_tensor quantized quantized tensors are supported by index_select.")
  }

  // 将维度值转换为有效的维度索引
  dim = maybe_wrap_dim(dim, self.dim());

  // 获取索引张量的元素数量
  auto numel = index.numel();

  // 检查索引张量维度是否为1，确保其为向量
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");

  // 检查处理标量情况下的特殊索引要求
  TORCH_CHECK(!(self.dim() == 0 && numel != 1), "index_select(): Index to scalar can have only 1 value, got ", numel, " value(s)");

  // 检查索引张量的数据类型，应为 int32 或 int64
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");

  // 检查结果张量与输入张量的数据类型需一致
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "index_select(): self and result must have the same scalar type");

  // 检查结果张量与输入张量、索引张量之间不存在内部重叠
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, index);

  // 根据输入张量的尺寸构建结果张量的尺寸向量
  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  at::native::resize_output(result, result_size);

  // 确保索引张量是连续的
  auto index_contig = index.contiguous();

  // 处理输入张量维度大于1的情况
  if (self.dim() > 1) {
    // 若索引张量为空，则直接返回结果张量
    if (numel == 0) {
      return result;
    }

    // 若输入张量为空，则进行边界检查
    if (self.numel() == 0) {
      auto src_indexing_axis_dim = self.size(dim);
      TORCH_CHECK(src_indexing_axis_dim > 0,
                  "index_select(): self indexing axis dim should be positive");
      AT_DISPATCH_INDEX_TYPES(
      index_contig.scalar_type(), "index_select_empty_self_bound_check", [&]() {
        const auto* idxs = index_contig.const_data_ptr<index_t>();
        check_indexarray_range<index_t>(idxs, numel, src_indexing_axis_dim);
      });
      return result;
    }

    // 在维度为1且结果张量是连续的情况下，使用快速路径处理
    if (dim == 1 && result.is_contiguous()) {
      return index_select_out_cpu_dim1_(result, self, index_contig);
    }

    // 非快速路径下，分别获取输入张量和结果张量在指定维度上的切片
    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);

    // 获取输入张量和结果张量切片的数据指针及字节步长
    auto selfSlice_data = selfSlice.const_data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();

    // 配置张量迭代器，以便处理张量上的操作
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(resultSlice)
      .add_const_input(selfSlice)
      .build();

    // 设置操作的粒度大小
    auto grain_size = at::internal::GRAIN_SIZE;
    auto outer_loop =
      // 显式捕获所有必需变量以解决 Windows 构建问题
      // TODO: 当 Windows 能正确捕获嵌套 lambda 中的变量时，修复此问题
      [&index_contig, &iter, &self_dim_size, &selfSlice_data, &self_stride_bytes, &resultSlice_data,
        &result_stride_bytes](int64_t start, int64_t end) {
      // 使用 TensorIterator 创建子迭代器
      auto sub_iter = TensorIterator(iter);
      // 根据索引类型分派操作，此处为 index_select_out_cpu_
      AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
        [&index_contig, &start, &end, &sub_iter, &self_dim_size, &selfSlice_data, &self_stride_bytes,
          &resultSlice_data, &result_stride_bytes] () {
        // 获取索引数据指针
        auto index_data = index_contig.const_data_ptr<index_t>();
        // 对于范围内的每个索引执行操作
        for (const auto i : c10::irange(start, end)) {
          // 获取当前索引值
          auto self_i = index_data[i];
          // 检查索引值是否在有效范围内
          TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          // 计算 self 数据的地址
          auto self_data = static_cast<const char*>(selfSlice_data) + self_i * self_stride_bytes;
          // 计算 result 数据的地址
          auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
          // 替换子迭代器的操作数，将 result_data 作为第一个操作数，self_data 作为第二个操作数
          sub_iter.unsafe_replace_operand(0, result_data);
          sub_iter.unsafe_replace_operand(1, const_cast<char*>(self_data));
          // 根据设备类型调用复制函数
          copy_stub(sub_iter.device_type(), sub_iter, false);
        };
      });
    };

    // 如果切片大小大于等于指定粒度，则在内部循环上并行执行；
    // 否则在外部循环上并行执行
    if (slice_size >= grain_size) {
      // 调用外部循环函数，范围从 0 到 numel
      outer_loop(0, numel);
    } else {
      // 如果 self 和 result 是连续的且数据类型相同，则使用快速循环
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        // 计算切片大小的字节数
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        // 由于 Windows 构建的问题，显式捕获所有必需变量
        // TODO: 当 Windows 能正确捕获嵌套 lambda 中的变量时，修复此问题
        // 使用并行处理，划分任务范围从 0 到 numel，每个任务的粒度为 grain_size / slice_size
        at::parallel_for(0, numel, grain_size / slice_size,
          [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
            &self_stride_bytes, &resultSlice_data, &result_stride_bytes](int64_t start, int64_t end) {
          // 根据索引类型分派具体操作的 Lambda 函数
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
              &self_stride_bytes, &resultSlice_data, &result_stride_bytes, &start, &end] () {
            auto index_data = index_contig.const_data_ptr<index_t>();
            // 在给定的任务范围内进行循环处理
            for (const auto i : c10::irange(start, end)) {
              auto self_i = index_data[i];
              // 检查索引是否超出 self 的范围
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              // 计算 self 数据的偏移地址
              auto self_data = static_cast<const char*>(selfSlice_data) + self_i * self_stride_bytes;
              // 计算 result 数据的偏移地址
              auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
              // 执行数据拷贝操作
              memcpy(result_data, self_data, slice_size_bytes);
            }
          });
        });
      } else {
        // 使用并行处理，划分任务范围从 0 到 numel，每个任务的粒度为 grain_size / slice_size，调用外部循环函数
        at::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    // 检查结果张量的维度不超过 1，否则抛出错误
    TORCH_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
    // 由于 Windows 构建的问题，显式捕获所有必需变量
    // TODO: 当 Windows 能正确捕获嵌套 lambda 中的变量时，修复此问题
    // 检查当前张量是否量化，如果是，则使用量化类型进行索引选择操作
    if(self.is_quantized()){
      // 使用宏，根据量化整型类型调度索引选择量化操作
      AT_DISPATCH_QINT_TYPES(self.scalar_type(), "index_select_quant", [&index_contig, &self, &result, &dim, &numel] {
        // 获取当前张量和结果张量在指定维度上的步长
        auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
        auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
        // 获取当前张量和结果张量的数据指针
        auto self_data_ptr = self.const_data_ptr<scalar_t>();
        auto result_data_ptr = result.data_ptr<scalar_t>();
        auto self_numel = self.numel();
        // 使用宏，根据索引类型调度具体的操作
        AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_quant_",
          [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
          // 获取索引数据的指针
          auto index_data = index_contig.const_data_ptr<index_t>();
          // 遍历索引范围内的元素
          for (const auto i : c10::irange(numel)) {
            auto self_i = index_data[i];
            // 检查索引是否在当前张量的范围内
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
            // 计算当前张量中指定位置的数据指针
            const scalar_t *self_ip = self_data_ptr + self_i * self_stride;
            // 将数据复制到结果张量中指定位置
            *(result_data_ptr + i * result_stride) = *self_ip;
          }
        });
      });
    } else {
      // 如果当前张量未量化，则调度普通类型的索引选择操作
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        self.scalar_type(), "index_select", [&index_contig, &self, &result, &dim, &numel] {
        // 获取当前张量和结果张量在指定维度上的步长
        auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
        auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);

        // 获取当前张量和结果张量的数据指针
        auto self_data_ptr = self.const_data_ptr<scalar_t>();
        auto result_data_ptr = result.data_ptr<scalar_t>();
        auto self_numel = self.numel();
        // 使用宏，根据索引类型调度具体的操作
        AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
          [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
          // 获取索引数据的指针
          auto index_data = index_contig.const_data_ptr<index_t>();
          // 遍历索引范围内的元素
          for (const auto i : c10::irange(numel)) {
            auto self_i = index_data[i];
            // 检查索引是否在当前张量的范围内
            TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
            // 计算当前张量中指定位置的数据指针
            const scalar_t *self_ip = self_data_ptr + self_i * self_stride;
            // 将数据复制到结果张量中指定位置
            *(result_data_ptr + i * result_stride) = *self_ip;
          }
        });
      });
    }
  }

  // 返回索引选择操作后得到的结果张量
  return result;
}

Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  // 创建一个空的张量作为结果
  Tensor result = at::empty({0}, self.options());
  // 调用 ATen 库中的 index_select_out_cpu_ 函数，将结果存储在 result 中并返回
  return at::native::index_select_out_cpu_(self, dim, index, result);
}

Tensor index_select_quantized_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  // 检查张量的量化方案是否为 kPerTensorAffine
  TORCH_CHECK(self.qscheme() == kPerTensorAffine,
              "Only per_tensor quantized quantized tensors are supported by index_select.")
  // 创建一个空的量化张量作为结果
  Tensor result = at::empty_quantized({0}, self);
  // 调用 ATen 库中的 index_select_out_cpu_ 函数，将结果存储在 result 中并返回
  return at::native::index_select_out_cpu_(self, dim, index, result);
}

Tensor index_select_backward_symint(const Tensor& grad, c10::SymIntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  // 如果 index 是 Tensor 子类，则使用 out-of-place 变体的 index_add
  if (isTensorSubclassLike(index)) {
    return grad.new_zeros_symint(self_sizes, grad.options()).index_add(dim, index, grad);
  }
  // 否则使用 in-place 变体的 index_add
  return grad.new_zeros_symint(self_sizes, grad.options()).index_add_(dim, index, grad);
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  // 禁用名称保护
  at::NoNamesGuard guard;

  // 检查索引张量的数据类型是否为 int64
  TORCH_CHECK_INDEX(
    index.scalar_type() == ScalarType::Long,
    "index_fill_(): Expected dtype int64 for index.");

  // 检查是否有内部重叠
  at::assert_no_overlap(self, index);
  // 如果存在内部重叠，发出警告
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
      "Use of index_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  // 如果 self 不是复数，但 source 是复数，则抛出错误
  if (!self.is_complex() && source.isComplex()) {
    TORCH_CHECK(false, "index_fill_(): Converting complex Scalar to non-complex type is not supported");
  }

  // 处理 self 是 0 维的情况，将其扩展为至少 1 维
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;

  // 确保 dim 在有效范围内
  dim = at::maybe_wrap_dim(dim, self_nonzero_dim);
  // 确保索引是一维或标量
  TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");

  // 准备 index 以便用于 TensorIterator，使其能够广播到 self 上
  auto index_sizes = std::vector<int64_t>(self_nonzero_dim.dim(), 1);
  auto index_strides = std::vector<int64_t>(self_nonzero_dim.dim(), 0);
  index_sizes[dim] = index.numel();
  // 如果 index 是一维的，则使用其步幅；否则步幅为 1
  index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1;
  // 重新排列 index，使其具有与 self 兼容的广播形状
  auto index_restrided = index.as_strided(
  // 准备 `self` 对象给 TensorIterator 使用。
  // 在维度 `dim` 上重新设置 `self` 对象，不会在该维度上前进。
  // 这里不使用 squash_dim，因为 `index` 需要在该维度上前进。
  // 注意 self_sizes[dim] 被设置为 index 的元素数。
  // 这样做是为了确保 self_sizes[dim] 和 index_sizes[dim] 匹配，
  // 符合 TensorIterator 的要求（输入形状应严格广播到输出形状，即
  // output.shape[i] >= input.shape[i] 对于所有 i 在维度上成立）。
  auto self_sizes = self_nonzero_dim.sizes().vec();
  auto self_strides = self_nonzero_dim.strides().vec();
  self_sizes[dim] = index.numel();  // 设置 self_sizes 中的 dim 维度为 index 的元素数
  self_strides[dim] = 0;  // 在 self_strides 中的 dim 维度设置为 0，即不增加步幅
  auto self_restrided = self_nonzero_dim.as_strided(self_sizes, self_strides);  // 对 self_nonzero_dim 进行重新步幅化

  auto iter = TensorIteratorConfig()
    // 不检查内存重叠，因为 `self` 已经被重新步幅化为零步幅。
    // 零步幅会触发 TensorIterator 中的内存重叠断言。
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)  // 检查所有输入的数据类型是否相同
    .resize_outputs(false)  // 不调整输出尺寸
    .add_output(self_restrided)  // 将重新步幅化后的 self 添加为输出
    .add_const_input(index_restrided)  // 将重新步幅化后的 index 添加为常量输入
    .build();  // 构建 TensorIterator

  auto self_dim_size = (self_nonzero_dim.sizes())[dim];  // 获取 self_nonzero_dim 在 dim 维度上的尺寸
  auto self_dim_stride = (self_nonzero_dim.strides())[dim];  // 获取 self_nonzero_dim 在 dim 维度上的步幅
  index_fill_stub(
    iter.device_type(),  // 使用 iter 的设备类型
    iter,
    dim,  // 操作的维度
    self_dim_size,  // self 在 dim 维度上的尺寸
    self_dim_stride,  // self 在 dim 维度上的步幅
    source);  // 填充的源数据

  return self;  // 返回修改后的 self 对象
// 结束函数体的大括号，标志函数实现的结束
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  // 检查源张量的维度是否为0，index_fill_ 只支持0维值张量，否则抛出错误信息
  TORCH_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", source.dim(), " dimension(s).");
  // 调用底层的 index_fill_ 方法，并返回结果张量的引用
  return self.index_fill_(dim, index, source.item());
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  // 克隆原张量并保持内存格式，然后调用其 index_fill_ 方法，并返回结果张量
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  // 克隆原张量并保持内存格式，然后调用其 index_fill_ 方法，并返回结果张量
  return self.clone(at::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

// GNN 使用的快速路径
static bool can_use_expanded_index_path(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    bool is_scatter_like) {
#ifdef USE_FBGEMM
  // 如果未使用 FBGEMM 的 OpenMP 加速，则无法使用扩展索引路径，直接返回 false
  if (!fbgemm::is_radix_sort_accelerated_with_openmp()) {
    return false;
  }
#else
  // 如果未定义 USE_FBGEMM，则无法使用扩展索引路径，直接返回 false
  return false;
#endif

  // 如果张量不在 CPU 设备上，则无法使用扩展索引路径，返回 false
  if (!self.device().is_cpu()) {
    return false;
  }

  // 检查张量的标量类型，只有浮点类型（除半精度外）才能使用扩展索引路径，否则返回 false
  const auto st = self.scalar_type();
  if (!(c10::isFloatingType(st)) || st == ScalarType::Half) {
    return false;
  }

  // 如果任何一个张量是空的，则无法使用扩展索引路径，返回 false
  if (self.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return false;
  }

  // 如果任何一个张量的维度为0，则无法使用扩展索引路径，返回 false
  if (self.ndimension() == 0 || index.ndimension() == 0 || src.ndimension() == 0) {
    return false;
  }

  // 仅允许在 dim 0 上 src 和 index 的大小不同，其他维度大小必须一致
  // 参考：https://github.com/pytorch/pytorch/issues/99595
  for (const auto dim : c10::irange(1, index.dim())) {
    if (src.size(dim) != index.size(dim)) {
      return false;
    }
  }

  // 如果是 scatter 操作，并且 index 的元素数量除以第一维大小小于阈值 16，则不使用扩展索引路径
  if (is_scatter_like) {
    constexpr int64_t threshold = 16;
    if (index.numel() / index.size(0) < threshold) {
      return false;
    }
  }

  // 检查索引张量是否是扩展的，通常第一维度的步长为1，其他维度的步长为0或1
  auto index_strides = index.strides().vec();
  bool is_index_expanded = index_strides[0] == 1;
  for (const auto dim : c10::irange(1, index_strides.size())) {
    if (index_strides[dim] > 1) { is_index_expanded = false; }
  }

  // 当前索引是扩展的条件：dim为0、索引是扁平的、源张量和结果张量都是连续的
  return dim == 0 && is_index_expanded && src.is_contiguous() && self.is_contiguous();
}

// gather_out_cpu_cuda
TORCH_IMPL_FUNC(gather_out)
(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, const Tensor& result) {
  // 如果索引张量是空的，则直接返回，不执行操作
  if (index.numel() == 0) return;
  // 确保维度索引在有效范围内
  dim = at::maybe_wrap_dim(dim, self.dim());
  // 如果可以使用扩展索引路径，则调用对应的 gather_expanded_index_stub 方法
  if (can_use_expanded_index_path(result, dim, index, self, /*is_scatter_like=*/false)) {
    gather_expanded_index_stub(result.device().type(), result, self, index);
  } else {
    gather_stub(result.device().type(), result, self, dim, index);
// 索引和梯度张量任一是张量子类时，为了符合复合、vmap 和引线器的要求，使用 `scatter_add` 的无输出位置变体。
static void _scatter_via_index_put(
  // 指定的张量 `self`，用于散播操作的源张量
  const Tensor& self,
  // 散播的维度
  int64_t dim,
  // 用于指定位置的索引张量
  const Tensor& index,
  // 要散播的源张量
  const Tensor& src,
  // 输出张量的可变版本
  const Tensor& mut_out,
  // 是否累加到输出中
  bool accumulate) {
  
  // 如果 `self` 是一维张量
  if (self.dim() == 1) {
    // 创建一个 Torch 列表，用于存放一个可选的张量索引
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(1);
    indices.push_back(index);
    // 对输出张量进行索引放置操作，根据累加标志决定是否累加
    mut_out.index_put_(indices, src, accumulate);
  } else {
    // 创建一个连续版本的输出张量 `mut_out_contig`
    Tensor mut_out_contig = mut_out.contiguous();

    // 创建存储索引坐标的张量 `index_coords`
    auto index_coords_sizes = index.sizes().vec();
    index_coords_sizes.push_back(self.dim());
    auto index_coords = at::empty(
      index_coords_sizes,
      at::TensorOptions().dtype(at::ScalarType::Long).device(self.device()));
      
    // ...
    // 以下部分代码被省略，根据提示继续编写注释
    // 遍历所有维度，除了当前维度 `dim` 外
    for (int64_t dim_other = 0; dim_other < self.dim(); dim_other++) {
      // 如果当前维度 `dim_other` 等于 `dim`，则跳过本次循环
      if (dim_other == dim) {
        continue;
      }
      // 生成一个张量 `dim_coord_vals`，包含从 0 到 index.size(dim_other)-1 的整数，张量位于当前设备上
      auto dim_coord_vals = at::arange(
        index.size(dim_other),
        at::TensorOptions().device(self.device()));

      // 在当前维度上插入维度，使得 dim_coord_vals 的维度与 self 的维度一致
      for (int64_t dim_unsqueeze = 0; dim_unsqueeze < self.dim() - 1; dim_unsqueeze++) {
        dim_coord_vals = dim_coord_vals.unsqueeze((dim_unsqueeze >= dim_other) ? -1 : 0);
      }

      // 创建一个与 index_coords 尺寸和步长相符的视图，目标维度是 dim_other
      auto view_sizes = index.sizes().vec();
      view_sizes.push_back(1);
      auto view_strides = index_coords.strides().vec();
      view_strides[self.dim()] = self.dim();

      // 将 dim_coord_vals 的数据复制到 index_coords 的对应视图中
      at::as_strided(
        index_coords,
        view_sizes,
        view_strides,
        dim_other
      ).copy_(dim_coord_vals.unsqueeze(-1));
    }

    // 为 index_coords 创建一个与 index 尺寸相同的视图，目标维度是 dim
    auto view_sizes = index.sizes().vec();
    view_sizes.push_back(1);
    auto view_strides = index_coords.strides().vec();
    view_strides[self.dim()] = self.dim();

    // 将 index 的数据复制到 index_coords 的对应视图中
    at::as_strided(
      index_coords,
      view_sizes,
      view_strides,
      dim
    ).copy_(index.unsqueeze(-1));

    // 将 index_coords 展平为一个一维张量
    Tensor index_coords_flat = index_coords.flatten(0, -2);

    // 复制 mut_out_contig 的步长到一个张量中
    // TODO: 是否有现成的实用函数可以完成这一步骤？
    IntArrayRef mut_out_contig_strides = mut_out_contig.strides();
    // 创建一个长整型张量 coord_strides，与 mut_out_contig 的维度一致，位于 CPU 上
    Tensor coord_strides = at::empty(
      {mut_out_contig.dim()},
      TensorOptions().dtype(at::ScalarType::Long).device(at::kCPU));
    // 使用 memcpy 将 mut_out_contig 的步长数据复制到 coord_strides 中
    std::memcpy(
      coord_strides.mutable_data_ptr(),
      mut_out_contig_strides.data(),
      coord_strides.nbytes());
    // 将 coord_strides 转移到 mut_out_contig 的设备上
    coord_strides = coord_strides.to(mut_out_contig.device());

    // index_flat 包含与展平后的 mut_out 对应的一维索引
    Tensor index_flat = (index_coords_flat * coord_strides).sum({-1});
    // 展平 mut_out_contig 为一个一维张量
    Tensor mut_out_flat = mut_out_contig.flatten();
    // 创建一个与 src 尺寸相符的视图，并展平为一维张量
    Tensor src_flat = at::as_strided(
      src,
      index.sizes(),
      src.strides()
    ).flatten();

    // 创建一个包含一个 Tensor 的 Torch 列表 indices，并预留空间
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(1);
    // 将 index_flat 加入 indices 列表中
    indices.push_back(index_flat);

    // 使用 index_flat 和 src_flat 更新 mut_out_flat 的数据，使用累积方法 accumulate
    mut_out_flat.index_put_(indices, src_flat, accumulate);

    // 如果 mut_out 不是连续的，则将 mut_out_flat 重塑为与 mut_out 尺寸相同的张量并复制到 mut_out
    if (!mut_out.is_contiguous()) {
      mut_out.copy_(mut_out_flat.reshape(mut_out.sizes()));
    }
  }
template <bool use_new_options = false, typename T, typename ReduceStub, typename FillStub>
void scatter_impl(
    const Tensor& self,                           // 输入张量
    int64_t dim,                                  // 指定的维度
    const Tensor& index,                          // 索引张量
    const T& src,                                 // 源数据
    const Tensor& out,                            // 输出张量
    ReduceStub& reduce_stub,                      // 缩减操作的stub
    FillStub& fill_stub,                          // 填充操作的stub
    const std::optional<c10::string_view> reduce = nullopt,  // 可选的缩减操作类型
    bool reduce_includes_self = true) {           // 是否包含自身在内进行缩减操作

  dim = at::maybe_wrap_dim(dim, self.dim());      // 根据张量的维度调整dim的值
  auto mut_out = const_cast<Tensor&>(out);        // 创建可变的输出张量

  if (!self.is_same(mut_out)) {                   // 如果输入张量不同于输出张量
    mut_out.copy_(self);                          // 复制输入张量到输出张量
  }

  if (index.numel() == 0) return;                 // 如果索引张量元素个数为0，则直接返回

  auto op = ReductionType::SUM;                   // 默认缩减操作为求和
  bool deterministic = globalContext().deterministicAlgorithms() && self.device().type() == DeviceType::CUDA;  // 确定性标志

  if (reduce.has_value()) {                       // 如果指定了缩减操作类型
    op = get_operator_enum(reduce.value(), use_new_options);  // 获取缩减操作类型的枚举值
    if (!reduce_includes_self) {
      // scatter inits for reduction to appropriate indices (used by scatter_reduce.two)
      // 如果不包含自身在内进行缩减操作，则调用辅助函数初始化
      scatter_reduce_exclude_self_helper(mut_out, dim, index, op);
    }
    // _scatter_via_index_put can only handle sum and mean reduction type
    // 如果缩减操作为SUM或MEAN，则标志为确定性操作
    deterministic = deterministic && (op == ReductionType::SUM || op == ReductionType::MEAN);
  }

  // Scalar src should already be deterministic
  // 标量源数据应该已经是确定性的
  if (deterministic && std::is_same_v<T, Tensor>) {
    // both runtime and compile check are required
    // 需要运行时和编译时检查
    if constexpr (std::is_same_v<T, Tensor>) {
      bool accumulate = reduce.has_value();
      // 根据索引张量使用索引放置方法进行分散操作
      _scatter_via_index_put(self, dim, index, src, mut_out, accumulate);
      return;
    }
  }

  if (reduce.has_value()) {
    // 如果指定了缩减操作类型，则调用相应的缩减stub函数
    reduce_stub(self.device().type(), mut_out, dim, index, src, op);
  } else {
    // 否则调用填充stub函数
    fill_stub(self.device().type(), mut_out, dim, index, src);
  }
}
(const Tensor& self,                          // 原始张量的常量引用
 int64_t dim,                                  // 操作的维度
 const Tensor& index,                          // 索引张量
 const Tensor& src,                            // 源张量
 const Tensor& out) {                          // 输出张量
  auto mut_out = const_cast<Tensor&>(out);     // 创建可变引用以修改输出张量
  dim = maybe_wrap_dim(dim, self.dim());       // 确保维度值在有效范围内

  if (!self.is_same(mut_out)) {                // 如果输出张量和原始张量不同
    mut_out.copy_(self);                       // 将原始张量复制到输出张量
  }

  if (index.numel() == 0) return;              // 如果索引张量为空，则直接返回

  // 查看是否启用确定性操作，避免在 CUDA 上使用 gpuAtomicAdd
  if (globalContext().deterministicAlgorithms() && self.device().type() == DeviceType::CUDA) {
    _scatter_via_index_put(self, dim, index, src, mut_out, /*accumulate*/true);  // 使用索引放置操作
  } else {
    // 检查是否可以使用扩展索引路径来执行 scatter_add 操作
    if (can_use_expanded_index_path(mut_out, dim, index, src, /*is_scatter_like*/true)) {
      scatter_add_expanded_index_stub(self.device().type(), mut_out, index, src);  // 使用扩展索引的 scatter_add 操作
    } else {
      scatter_add_stub(self.device().type(), mut_out, dim, index, src);  // 常规 scatter_add 操作
    }
  }
}

TORCH_IMPL_FUNC(scatter_reduce_two)
(const Tensor& self,                          // 原始张量的常量引用
 int64_t dim,                                  // 操作的维度
 const Tensor& index,                          // 索引张量
 const Tensor& src,                            // 源张量
 const c10::string_view reduce,                // 减少操作的名称
 bool include_self,                            // 是否包含自身
 const Tensor& out) {                          // 输出张量
  dim = at::maybe_wrap_dim(dim, self.dim());   // 确保维度值在有效范围内

  if (!self.is_same(out)) {                    // 如果输出张量和原始张量不同
    out.copy_(self);                           // 将原始张量复制到输出张量
  }

  const auto op = get_operator_enum(reduce, true);  // 获取减少操作的枚举值

  // 检查是否可以使用扩展索引路径来执行 scatter_reduce 操作
  if (can_use_expanded_index_path(out, dim, index, src, /*is_scatter_like*/true)) {
    scatter_reduce_expanded_index_stub(self.device().type(), out, index, src, op, include_self);  // 使用扩展索引的 scatter_reduce 操作
    return;
  }

  // 使用 scatter_impl 执行 scatter 操作或 reduce 操作
  scatter_impl</*use_new_options=*/true>(self, dim, index, src, out,
                                         scatter_reduce_two_stub,
                                         scatter_stub,
                                         reduce,
                                         include_self);

  if (op == ReductionType::MEAN) {             // 如果是 MEAN 减少操作
    auto ones = at::ones_like(src);            // 创建一个与源张量相同形状的全 1 张量
    auto count = include_self ? at::ones_like(out) : at::zeros_like(out);  // 根据 include_self 创建计数张量
    count.scatter_add_(dim, index, ones);      // 根据索引在指定维度上累加计数
    count.masked_fill_(count == 0, 1);         // 将计数为零的位置填充为 1

    if (out.is_floating_point() || out.is_complex()) {
      out.div_(count);                         // 若输出张量为浮点数或复数，执行除法操作
    } else {
      out.div_(count, "floor");                // 否则，执行向下取整的除法操作
    }
  }
}

Tensor masked_scatter(const Tensor & self,      // 原始张量的常量引用
                      const Tensor & mask,      // 掩码张量
                      const Tensor & source) {  // 源张量
  auto [_mask, _self] = expand_outplace(mask, self);  // 将掩码张量和原始张量扩展到相同形状
  return _self->clone(at::MemoryFormat::Contiguous).masked_scatter_(*_mask, source);  // 执行 masked_scatter 操作并返回结果张量
}

Tensor masked_scatter_backward_symint(
    const Tensor& grad,                        // 梯度张量
    const Tensor& mask,                        // 掩码张量
    c10::SymIntArrayRef sizes) {               // 尺寸数组的常量引用
  c10::SymInt numel = 1;                      // 初始化元素数量为 1
  for (const auto& size : sizes) {            // 遍历尺寸数组
    numel *= size;                            // 计算总元素数量
  }
  auto mask_selected = grad.masked_select(mask);  // 根据掩码选择梯度张量中的元素
  auto diff_nelem = numel - mask_selected.sym_numel();  // 计算未选择元素的数量差

  if (diff_nelem > 0) {
    // 因为 mask_selected 返回的是一个 1 维张量，其大小等于选择的元素数量，我们需要用零填充其余元素，
    // 然后将其重新调整为张量2的大小。
    auto zeros_fillin =
        at::zeros_symint({std::move(diff_nelem)}, grad.options());  // 创建零填充张量
    # 将 `mask_selected` 和 `zeros_fillin` 沿着维度 0 进行连接，并且使用 `std::move` 将 `zeros_fillin` 的所有权转移
    mask_selected = at::cat({mask_selected, std::move(zeros_fillin)}, 0);
  }
  # 返回一个使用 `sizes` 参数重新视图化的 `mask_selected` 张量
  return mask_selected.view_symint(sizes);
}

static Tensor & masked_fill_impl_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  // 禁用命名保护，允许在此范围内进行操作
  NoNamesGuard guard;
  // 检查掩码的数据类型是否为布尔型
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_fill_ only supports boolean masks, but got mask "
      "with dtype ", mask.dtype());

  // 检查是否存在张量的内部重叠
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    // 发出警告信息，提示不建议在扩展的张量上使用 masked_fill_
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  // 确保张量 self 和 mask 不存在部分重叠
  at::assert_no_partial_overlap(self, mask);

  // 设置张量迭代器的配置
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 已废弃，但不是严格的错误
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self)
    .add_const_input(mask)
    .build();

  // 调用底层的 masked_fill_stub 函数来执行填充操作
  masked_fill_stub(iter.device_type(), iter, value);
  // 返回修改后的自身张量
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
  // 通过广播扩展确定输出张量的名称推断
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  // 调用 CPU 版本的 masked_fill_impl_cpu 函数执行填充操作
  masked_fill_impl_cpu(self, mask, value);
  // 如果可能的话，传播非空的名称到自身张量
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  // 返回修改后的自身张量
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  // 通过广播扩展确定输出张量的名称推断
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  // 检查值张量的维度是否为 0
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  // 调用 CPU 版本的 masked_fill_impl_cpu 函数执行填充操作，使用标量值
  masked_fill_impl_cpu(self, mask, value.item());
  // 如果可能的话，传播非空的名称到自身张量
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  // 返回修改后的自身张量
  return self;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Scalar& source) {
  // 定义结果张量
  Tensor result;
  // 通过广播扩展确定输出张量的名称推断
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    // 禁用命名保护，允许在此范围内进行操作
    NoNamesGuard guard;
    // 在此范围内，扩展掩码和自身张量
    auto [_mask, _self] = expand_outplace(mask, self);
    // 克隆自身张量，使用连续的内存格式
    result = _self->clone(at::MemoryFormat::Contiguous);
    // 对结果张量执行 masked_fill_ 操作
    result.masked_fill_(mask, source);
  }
  // 如果可能的话，传播非空的名称到结果张量
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回结果张量
  return result;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & source) {
  // 定义结果张量
  Tensor result;
  // 通过广播扩展确定输出张量的名称推断
  auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
  {
    // 禁用命名保护，允许在此范围内进行操作
    NoNamesGuard guard;
    // 在此范围内，扩展掩码和自身张量
    auto [_mask, _self] = expand_outplace(mask, self);
    // 克隆自身张量，使用连续的内存格式
    result = _self->clone(at::MemoryFormat::Contiguous);
    // 对结果张量执行 masked_fill_ 操作，使用张量作为填充源
    result.masked_fill_(mask, source);
  }
  // 如果可能的话，传播非空的名称到结果张量
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回结果张量
  return result;
}
// 在 CPU 上实现 masked_select 操作，将结果存储在 result 中
static Tensor & masked_select_out_impl_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;  // 禁止自动命名保护

  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");  // 检查 mask 张量的数据类型是否为布尔类型
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");  // 检查 self 和 result 张量的数据类型是否相同

  at::assert_no_internal_overlap(result);  // 确保 result 张量内部没有重叠
  at::assert_no_overlap(result, self);     // 确保 result 张量与 self 张量没有重叠
  at::assert_no_overlap(result, mask);     // 确保 result 张量与 mask 张量没有重叠

  auto [_mask, _self] = expand_outplace(mask, self);  // 扩展 mask 和 self 张量的形状以匹配

  auto shape = _self->sizes();  // 获取 _self 张量的大小
  int64_t numel = _mask->sum().item().toLong();  // 计算 mask 中值为 true 的元素数量
  at::native::resize_output(result, {numel});  // 调整 result 张量的大小以容纳选取出的元素
  if (numel == 0) {
    return result;  // 如果选取的元素数量为 0，则直接返回 result
  }

  // 在输入进入 TensorIterator 之前创建 result 的跨步视图
  auto strides = DimVector(shape.size(), 0);  // 创建一个跨步向量，初始化为零
  auto orig_stride = result.strides()[0];  // 获取 result 张量的原始跨步
  auto result_strided = result.as_strided(shape, strides);  // 创建 result 的跨步视图

  // 使用串行核心
  // 串行核心要求 src 在其逻辑顺序下遍历。然而，TensorIterator 可能重新排序维度，
  // 使得 src 按其物理顺序遍历，导致错误的答案。一个足够的条件是 _self 和 _mask 都是连续的。
  // 如果不满足，使用正确处理排列的并行核心。
  bool use_serial_kernel = (self.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ) &&
                           _self->is_contiguous() && _mask->is_contiguous();
  if (use_serial_kernel) {
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // 上述故意使 result 为零跨步
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(result_strided)
      .add_const_input(*_self)
      .add_const_input(*_mask)
      .build();

    masked_select_serial_stub(iter.device_type(), iter, orig_stride);  // 使用串行核心执行选择操作
    return result;  // 返回结果张量
  }

  // 使用前缀和记录被选元素的输出位置，以与 TensorIterator 并行化
  auto mask_long = at::empty(shape, self.options().dtype(at::kLong)).copy_(*_mask);  // 创建 mask 的长整型副本
  auto mask_prefix_sum = at::empty(shape, self.options().dtype(at::kLong));  // 创建用于存储前缀和的张量
  auto mask_long_data = mask_long.data_ptr<int64_t>();  // 获取 mask_long 数据指针
  auto mask_prefix_sum_data = mask_prefix_sum.data_ptr<int64_t>();  // 获取 mask_prefix_sum 数据指针
  // TODO: 这里在 C++14 中只能使用 std::partial_sum，
  // 当 PyTorch 升级到 C++17 时，可以使用 std::exclusive_scan，它具有更好的性能。
  std::partial_sum(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data);  // 计算 mask_long 的前缀和

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // 上述故意使 result 为零跨步
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(result_strided)
    .add_const_input(*_self)
    .add_const_input(*_mask)
    .add_const_input(mask_prefix_sum);  // 添加前缀和作为额外的输入维度
    .build();



// 调用一个对象的 build() 方法，可能用于完成对象的构建或初始化操作
.build();



  masked_select_stub(iter.device_type(), iter, orig_stride);
  return result;



// 使用 masked_select_stub 函数处理给定的参数，并返回结果
masked_select_stub(iter.device_type(), iter, orig_stride);
// 返回函数的结果
return result;
}

Tensor & masked_select_out_cpu(const Tensor & self, const Tensor & mask, Tensor & result) {
  // Perform named inference to determine output tensor names based on input tensors.
  namedinference::compute_broadcast_outnames(self, mask);
  // Call the implementation function for masked_select operation on CPU with output tensor.
  return masked_select_out_impl_cpu(result, self, mask);
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
  // Create an empty tensor `result` with zero elements and same options as `self`.
  Tensor result = at::empty({0}, self.options());
  // Call the out-of-place CPU function for masked_select operation.
  return at::native::masked_select_out_cpu(self, mask, result);
}

Tensor masked_select_backward(const Tensor& grad, const Tensor& input, const Tensor& mask) {
  // Create a zero-initialized tensor `result` with the same shape as `input` and broadcasted sizes.
  auto result = at::zeros_like(
      input.expand(at::infer_size(input.sizes(), mask.sizes())), at::MemoryFormat::Preserve);

  // Use out-of-place variant of masked_scatter if either `grad` or `mask` is subclass-like.
  if (areAnyTensorSubclassLike({grad, mask})) {
    return result.masked_scatter(mask, grad);
  }
  // Use in-place variant of masked_scatter for efficiency, handling broadcasting implicitly.
  result.masked_scatter_(mask, grad);
  return result;
}

namespace {

inline std::tuple<Tensor, Tensor, int64_t> _take_along_dim_helper(
    const Tensor& self,
    const Tensor& indices,
    int64_t dim) {
  // Check that input and indices tensors have matching dimensions.
  TORCH_CHECK(
      self.dim() == indices.dim(),
      "torch.take_along_dim(): input and indices should have the same number of dimensions, ",
      "but got ", self.dim(), " dimensions for input, and ", indices.dim(), " dimensions for indices")
  // Check that indices tensor has Long dtype.
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "torch.take_along_dim(): dtype of indices should be Long but got ", indices.scalar_type())

  // Ensure dimension `dim` is within bounds for self tensor.
  dim = at::maybe_wrap_dim(dim, self.dim());

  // Create a vector of symbolic sizes for self tensor.
  SymDimVector self_sizes{self.sym_sizes()};
  // Update the size at dimension `dim` based on indices tensor.
  self_sizes[dim] = indices.sym_size(dim);
  // Infer broadcast shape for self tensor and indices tensor.
  auto broadcast_shape = infer_size_symint(self_sizes, indices.sym_sizes());
  // Broadcast indices tensor to the inferred shape.
  auto indices_broadcasted = at::broadcast_to_symint(indices, broadcast_shape);

  // Create a vector of symbolic sizes for indices tensor.
  SymDimVector indices_sizes{indices.sym_sizes()};
  // Update the size at dimension `dim` based on self tensor.
  indices_sizes[dim] = self.sym_size(dim);
  // Infer broadcast shape for indices tensor and self tensor.
  broadcast_shape = infer_size_symint(indices_sizes, self.sym_sizes());
  // Broadcast self tensor to the inferred shape.
  auto self_broadcasted = at::broadcast_to_symint(self, broadcast_shape);

  // Return a tuple containing the broadcasted tensors and dimension `dim`.
  return std::make_tuple(std::move(self_broadcasted),
                         std::move(indices_broadcasted),
                         std::move(dim));
}

static inline void checkDevice(CheckedFrom c, const Tensor& t, Device device) {
  // Check if tensor `t` is defined and verify its device matches the expected `device`.
  TORCH_CHECK(
      !t.defined() || t.device() == device,
      "Expected tensor to have ", device,
      " Device, but got tensor with ", t.device(), " Device ",
      "(while checking arguments for ", c, ")");
}

static inline void checkDevice(CheckedFrom c, at::ArrayRef<Tensor> tensors, Device device) {
  // Iterate through each tensor in the array `tensors` and verify device consistency.
  for (auto &t : tensors) {
    // 调用名为 checkDevice 的函数，传入参数 c, t, device
    checkDevice(c, t, device);
  }
}

} // anonymous namespace

// 定义函数 take_along_dim，用于沿指定维度从输入张量中获取数据
Tensor take_along_dim(const Tensor& self, const Tensor& indices, std::optional<int64_t> opt_dim) {
  // 检查输入张量的设备是否一致，并打印相关信息
  checkDevice("torch.take_along_dim():", {self, indices}, self.device());
  
  // 如果指定了维度
  if (opt_dim.has_value()) {
    // 调用辅助函数 _take_along_dim_helper 处理输入张量和索引张量，并获取相应的广播后的张量和维度
    auto [self_broadcasted, indices_broadcasted, dim] =
        _take_along_dim_helper(self, indices, opt_dim.value());
    // 使用 gather 操作从广播后的输入张量 self_broadcasted 中收集索引张量 indices_broadcasted 的数据，并返回结果
    return self_broadcasted.gather(dim, indices_broadcasted);
  }

  // 类似于 take 操作，但是支持与 gather 相同的数据类型
  return self.view(-1).gather(0, indices.view(-1));
}

// 定义函数 take_along_dim_out，用于沿指定维度从输入张量中获取数据，并将结果存储到指定的输出张量 result 中
Tensor& take_along_dim_out(const Tensor& self, const Tensor& indices, std::optional<int64_t> opt_dim, Tensor& result) {
  // 检查输入张量和输出张量的设备是否一致，并打印相关信息
  checkDevice("torch.take_along_dim():", {self, indices, result}, self.device());
  
  // 如果指定了维度
  if (opt_dim.has_value()) {
    // 调用辅助函数 _take_along_dim_helper 处理输入张量和索引张量，并获取相应的广播后的张量和维度
    auto [self_broadcasted, indices_broadcasted, dim] =
        _take_along_dim_helper(self, indices, opt_dim.value());
    // 使用 gather_out 操作从广播后的输入张量 self_broadcasted 中收集索引张量 indices_broadcasted 的数据，并将结果存储到输出张量 result 中
    return at::gather_out(result, self_broadcasted, dim, indices_broadcasted);
  }

  // 类似于 take 操作，但是支持与 gather 相同的数据类型
  return at::gather_out(result, self.view(-1), 0, indices.view(-1));
}

// 定义函数 _gather_sparse_backward，用于稀疏张量的反向传播操作
Tensor _gather_sparse_backward(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& grad){
  // 处理特殊情况：输入张量或索引张量是标量的情况
  if (self.ndimension() == 0) return at::_sparse_coo_tensor_unsafe_symint(at::empty_symint({0,grad.sym_numel()}, index.options()), grad, self.sym_sizes());
  if (grad.ndimension() == 0) return at::_sparse_coo_tensor_unsafe_symint(index.view({1,1}), grad, self.sym_sizes());
  
  // 创建用于存储稀疏索引的张量 sparse_ind
  Tensor sparse_ind = at::empty_symint({self.ndimension(), grad.sym_numel()}, self.options().dtype(at::kLong));
  SymInt grad_numel = grad.sym_numel();
  if (grad_numel > 0) {
    SymInt n_above = grad_numel;
    SymInt n_below = 1;
    if (dim < 0) dim += self.ndimension();
    // 遍历输入张量的各个维度
    for (const auto i : c10::irange(self.ndimension())) {
        n_above /= grad.sym_size(i);
        // 如果当前维度是指定的 dim 维度，则将索引张量 index 重塑为一维，并赋给 sparse_ind[i]
        if (i == dim) {
            sparse_ind[i] = index.reshape(-1);
        } else {
            // 否则，生成一维的索引序列，并根据 grad.sym_size(i) 和 n_above 进行广播和重塑，最后重复 n_below 次
            sparse_ind[i] = at::arange(grad.sym_size(i),self.options().dtype(at::kLong)).unsqueeze(1).expand_symint({grad.sym_size(i), n_above}).reshape(-1).repeat_symint(n_below);
        }
        n_below *= grad.sym_size(i);
    }
  }
  // 返回稀疏张量，其中稀疏索引为 sparse_ind，梯度为 grad
  return at::_sparse_coo_tensor_unsafe_symint(sparse_ind, grad.reshape(-1), self.sym_sizes());
}

// 定义模板函数 count_nonzero_impl，用于计算迭代器中非零元素的数量
template <typename scalar_t>
int64_t count_nonzero_impl(TensorIteratorBase& iter, Range range) {
  int64_t num_nonzero = 0;

  // 定义循环操作
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    constexpr int ilp_factor = 4;
    const char* ptr = data[0];
    const auto stride = strides[0];
    int64_t nonzero[ilp_factor] = {0};

    int64_t i = 0;
    // 循环并行处理数组，每次处理 ilp_factor 个元素
    for (; i + (ilp_factor - 1) < n; i += ilp_factor) {
      // 强制展开循环以提高并行效率，处理 ilp_factor 个元素
      c10::ForcedUnroll<ilp_factor>{}([&](int k) {
        // 加载内存中的 scalar_t 类型数据到 val
        const auto& val = c10::load<scalar_t>(ptr + k * stride);
        // 如果 val 不为 0，则增加 nonzero[k] 的计数
        if (val != scalar_t(0)) {
          ++nonzero[k];
        }
      });
      // 指针移动到下一个 ilp_factor 个元素的位置
      ptr += ilp_factor * stride;
    }

    // 处理剩余的元素，逐个处理直到末尾
    for (; i < n; ++i) {
      // 加载内存中的 scalar_t 类型数据到 val
      const auto& val = c10::load<scalar_t>(ptr);
      // 如果 val 不为 0，则增加 nonzero[0] 的计数
      if (val != scalar_t(0)) {
        ++nonzero[0];
      }
      // 指针移动到下一个元素的位置
      ptr += stride;
    }

    // 将 nonzero 数组中从 1 到 ilp_factor-1 的元素累加到 nonzero[0]
    for (const auto k : c10::irange(1, ilp_factor)) {
      nonzero[0] += nonzero[k];
    }

    // 将当前线程计算得到的非零元素个数累加到总的非零元素个数
    num_nonzero += nonzero[0];
  };

  // 执行串行循环迭代，对 range 范围内的数据进行处理
  iter.serial_for_each(loop, range);

  // 返回计算得到的总的非零元素个数
  return num_nonzero;
}

// 在 CUDA 上计算非零元素数量的函数，输入是张量 self 和维度列表 dims
Tensor count_nonzero_cuda(const Tensor& self, IntArrayRef dims){
  return (self != 0).sum(dims);
}

// 在 CPU 上计算非零元素数量的函数，输入是张量 self 和维度列表 dims
Tensor count_nonzero_cpu(const Tensor& self, IntArrayRef dims){
  if (!dims.empty()) {
    // 如果 dims 非空，则按照给定维度 dims 计算非零元素数量
    return (self != 0).sum(dims);
  }

  // 优化的全局归约计算
  auto iter = TensorIteratorConfig()
      .add_const_input(self)
      .build();

  const auto num_threads = at::get_num_threads();
  DimVector thread_count_nonzero(num_threads);

  // 遍历所有类型，使用并行计算每个线程的非零元素数量
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
    at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      const auto tid = at::get_thread_num();
      thread_count_nonzero[tid] = count_nonzero_impl<scalar_t>(iter, {begin, end});
    });
  });

  // 将各线程的非零计数累加到第一个线程
  for (const auto i : c10::irange(1, num_threads)) {
    thread_count_nonzero[0] += thread_count_nonzero[i];
  }
  
  // 创建一个与输入张量类型相同的长整型张量作为输出
  auto out = at::empty({}, self.options().dtype(kLong));
  *out.mutable_data_ptr<int64_t>() = thread_count_nonzero[0];
  return out;
}

// 计算张量 self 在指定维度 dim 上的非零元素数量
Tensor count_nonzero(const Tensor& self, std::optional<int64_t> dim) {
  if (dim) {
    return at::count_nonzero(self, IntArrayRef{*dim});
  }
  return at::count_nonzero(self, IntArrayRef{});
}

// 在 CPU 上实现 nonzero 操作的输出到 result 张量
Tensor& nonzero_out_cpu(const Tensor& self, Tensor& result) {
  // 检查输出张量 result 的类型是否为 Long
  TORCH_CHECK(result.scalar_type() == kLong,
              "nonzero: Expected out tensor to have scalar type Long "
              "but got scalar type", result.scalar_type());
  // 检查输出张量 result 与输入张量 self 没有内部重叠
  at::assert_no_internal_overlap(result);
  // 检查输出张量 result 与输入张量 self 没有重叠
  at::assert_no_overlap(result, self);

  // 配置张量迭代器以支持线性迭代
  auto iter = TensorIteratorConfig()
    .add_const_input(self)
    .enforce_linear_iteration()
    .build();

  const auto numel = iter.numel();
  const auto num_threads = at::get_num_threads();
  DimVector thread_begin(num_threads, -1);
  DimVector thread_count_nonzero(num_threads + 1);

  // 第一遍：计算每个线程中非零元素的数量
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      const auto tid = at::get_thread_num();
      thread_begin[tid] = begin;
      thread_count_nonzero[tid + 1] = count_nonzero_impl<scalar_t>(iter, {begin, end});
    });
  });

  // 将每个线程的非零计数转换为累计和
  for (const auto i : c10::irange(1, thread_count_nonzero.size())) {
    thread_count_nonzero[i] += thread_count_nonzero[i - 1];
  }

  // 获取输入张量 self 的大小
  const auto self_sizes = self.sizes();
  const auto total_nonzero = thread_count_nonzero.back();
  const int64_t ndim = self_sizes.size();

  // 如果需要重新调整输出张量 result 的大小，则进行调整
  if (resize_output(result, {total_nonzero, ndim})) {
    // 默认使用 Fortran 连续布局的输出张量（参见 gh-46224）
    result.as_strided_({total_nonzero, ndim}, {1, total_nonzero});
  }

  // 如果输出张量的元素数量为零，则...
    return result;
  }

  auto out_accessor = result.accessor<int64_t, 2>();

  // Pass 2: Write indexes
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_cpu", [&] {
    at::parallel_for(0, numel, internal::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
      auto tid = at::get_thread_num();
      // Work needs to be distributed the same on both passes
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(begin == thread_begin[tid]);

      // +1 faster than additional condition check inside loop
      c10::SmallVector<int64_t, 33> sizes(ndim + 1, -1);
      std::copy(self_sizes.begin(), self_sizes.end(), sizes.begin() + 1);
      c10::SmallVector<int64_t, 33> current_idx(ndim + 1);
      if (begin > 0) {
        auto idx = begin;
        for (int64_t k = ndim; idx > 0 && k > 0; --k) {
          current_idx[k] = idx % sizes[k];
          idx /= sizes[k];
        }
      }

      auto out_ptr = out_accessor[thread_count_nonzero[tid]].data();

      auto loop = [&](char** data, const int64_t* strides, int64_t n1, int64_t n2) {
        // Copy into local variables to improve compiler alias analysis
        int64_t* C10_RESTRICT local_idx = current_idx.data() + 1;
        const int64_t* C10_RESTRICT local_sizes = sizes.data() + 1;
        // Compute strides for input and output tensors
        const auto in_stride = strides[0];
        const auto out_stride1 = out_accessor.stride(1);
        const auto out_stride0 = out_accessor.stride(0) - ndim * out_stride1;
        const auto ndim = out_accessor.size(1);  // Number of dimensions in output tensor
        int64_t* out = out_ptr;  // Pointer to current output position

        // Iterate over input data in chunks
        for (const auto i : c10::irange(n2)) {
          const char* ptr = data[0] + i * strides[1];
          for (C10_UNUSED const auto j : c10::irange(n1)) {
            const auto& val = c10::load<scalar_t>(ptr);
            // If nonzero, write index to output tensor
            if (val != scalar_t(0)) {
              for (const auto k : c10::irange(ndim)) {
                *out = local_idx[k];  // Write current index value
                out += out_stride1;   // Move to next position in the current dimension
              }
              out += out_stride0;  // Move to next position in the outermost dimension
            }
            ptr += in_stride;

            // Advance current index
            int64_t k = ndim - 1;
            ++local_idx[k];
            while (C10_UNLIKELY(local_idx[k] == local_sizes[k])) {
              local_idx[k] = 0;
              --k;
              ++local_idx[k];
            }
          }
        }
        out_ptr = out;  // Update the pointer to current output position
      };
      // Execute the loop in parallel
      iter.serial_for_each(loop, {begin, end});
      // Assertion to ensure correctness of output pointer after loop execution
      TORCH_INTERNAL_ASSERT(out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data());
    });
  });
  return result;
}

Tensor nonzero_cpu(const Tensor& self) {
  // 创建一个空的张量 `result`，使用与 `self` 相同的选项和长整型数据类型
  auto result = at::empty({0}, self.options().dtype(kLong));
  // 调用非零元素索引函数 `nonzero_out_cpu`，将结果存储到 `result` 中
  nonzero_out_cpu(self, result);
  // 返回结果张量 `result`
  return result;
}

Tensor& nonzero_static_out_cpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& result) {
  // 检查 `size` 是否为非负数
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  // 检查输出张量 `result` 的数据类型是否为长整型
  TORCH_CHECK(
      result.scalar_type() == kLong,
      "nonzero_static: Expected out tensor to have scalar type Long "
      "but got scalar type",
      result.scalar_type());

  // 获取输入张量 `self` 的维度
  int64_t ndim = self.dim();
  // 如果 `result` 的维度不是2，或者其大小与预期不符，则重新调整大小
  if (result.dim() != 2 || result.size(0) != size || result.size(1) != ndim) {
    at::native::resize_output(result, {size, ndim});
  }
  // 检查输出张量 `result` 是否为2维张量
  TORCH_CHECK(
      result.dim() == 2,
      "nonzero_static: Expected out tensor to be a 2D tensor but got a ",
      result.dim(),
      "D tensor");
  // 检查输出张量 `result` 的大小是否符合预期
  TORCH_CHECK(
      result.size(0) == size && result.size(1) == ndim,
      "nonzero_static: Expected out tensor to have Size([",
      size,
      ", ",
      ndim,
      "]) but got Size([",
      result.size(0),
      ", ",
      result.size(1),
      "]) ");
  // 确保输出张量 `result` 没有内部重叠
  at::assert_no_internal_overlap(result);
  // 确保输出张量 `result` 与输入张量 `self` 没有重叠
  at::assert_no_overlap(result, self);

  // 如果输出张量任一维度为0，则直接返回
  if (result.size(0) == 0 || result.size(1) == 0) {
    return result;
  }

  // 调用 `nonzero_cpu` 函数获取动态结果
  auto dyn_result = nonzero_cpu(self);
  int64_t num_nonzeros = dyn_result.size(0);
  int64_t copy_len = std::min(size, num_nonzeros);
  // 将动态结果复制到固定大小的张量 `result` 中
  result.narrow(0, 0, copy_len).copy_(dyn_result.narrow(0, 0, copy_len));
  // 如果 `size` 大于 `copy_len`，则使用 `fill_value` 填充 `result`
  if (size > copy_len) {
    result.narrow(0, copy_len, size - copy_len).fill_(fill_value);
  }
  // 返回结果张量 `result`
  return result;
}

Tensor nonzero_static_cpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value) {
  // 检查 `size` 是否为非负数
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  // 获取输入张量 `self` 的维度
  int64_t ndim = self.dim();
  // 创建一个空的长整型张量 `result`，大小为 `(size, ndim)`
  auto result = at::empty(
      {size, ndim},
      at::TensorOptions().dtype(at::ScalarType::Long).device(at::kCPU));
  // 调用 `nonzero_static_out_cpu` 函数，填充 `result` 的值
  nonzero_static_out_cpu(self, size, fill_value, result);
  // 返回结果张量 `result`
  return result;
}

std::vector<Tensor> nonzero_numpy(const Tensor& self) {
  // 处理标量情况，以兼容 numpy：
  // 如果输入张量 `self` 的维度为0，则添加一个维度后调用 `nonzero`，并解绑第1维
  if (self.dim() == 0) {
    return self.unsqueeze(0).nonzero().unbind(1);
  }

  // 对于其他情况，直接调用 `nonzero` 并解绑第1维
  return self.nonzero().unbind(1);
}

Tensor argwhere(const Tensor& self) {
  // 直接调用 `nonzero` 返回结果
  return self.nonzero();
}
// 按位覆盖 CPU 上的张量 `self`，使用 `mask` 来选择性地从 `source` 中复制数据
Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  // 断言自身张量不存在内部重叠
  at::assert_no_internal_overlap(self);
  // 检查自身张量和源张量 `source` 的数据类型是否相同
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  // 检查自身张量 `self` 的设备类型是否为 CPU
  TORCH_CHECK(self.device().type() == at::kCPU, "device type of self (", self.device().type(), ") is not CPU");
  // 检查掩码张量 `mask` 的设备类型是否为 CPU
  TORCH_CHECK(mask.device().type() == at::kCPU, "device type of mask (", mask.device().type(), ") is not CPU");
  // 检查源张量 `source` 的设备类型是否为 CPU
  TORCH_CHECK(source.device().type() == at::kCPU, "device type of source (", source.device().type(), ") is not CPU");

  // 将掩码张量 `mask` 扩展到与自身张量 `self` 的形状，并可能共享存储
  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  // 如果掩码张量 `mask` 的数据类型为 torch.uint8，则发出警告，该行为已弃用
  if (b_mask->dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated, please use a mask with dtype torch.bool instead.");
  }

  // 将源张量 `source` 进行连续化处理
  auto src_cont = source.contiguous();

  // 配置张量迭代器，用于处理张量的迭代操作
  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // 不检查内存重叠
      .check_all_same_dtype(false)   // 不要求所有输入张量具有相同的数据类型
      .resize_outputs(false)         // 不调整输出张量的大小
      .enforce_linear_iteration()    // 强制线性迭代顺序
      .add_output(self)              // 添加输出张量 `self` 作为输出
      .add_const_input(*b_mask)      // 将扩展后的掩码张量作为输入之一
      .build();

  // 调用底层的 masked_scatter_stub 函数，执行按位覆盖操作
  masked_scatter_stub(iter.device_type(), iter, src_cont);
  // 返回修改后的自身张量 `self`
  return self;
}
```