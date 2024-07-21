# `.\pytorch\aten\src\ATen\native\sparse\SparseTensor.cpp`

```py
// 定义宏，限制在这些头文件中只使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入各种头文件，用于稀疏张量的基本操作
#include <ATen/core/Tensor.h>                    // 引入张量的核心头文件
#include <ATen/Dispatch.h>                       // 分发操作头文件
#include <ATen/InitialTensorOptions.h>           // 初始化张量选项头文件
#include <ATen/Layout.h>                         // 引入张量布局相关头文件
#include <ATen/Parallel.h>                       // 并行操作头文件
#include <ATen/SparseTensorImpl.h>               // 稀疏张量实现相关头文件
#include <ATen/native/SparseTensorUtils.h>       // 稀疏张量工具函数头文件
#include <ATen/native/sparse/SparseStubs.h>      // 稀疏张量存根函数头文件
#include <ATen/native/IndexingUtils.h>           // 索引工具函数头文件
#include <ATen/native/NonSymbolicBC.h>           // 非符号边界条件头文件
#include <ATen/NamedTensorUtils.h>               // 命名张量工具函数头文件

#include <ATen/native/Copy.h>                    // 复制操作头文件
#include <ATen/native/CPUBlas.h>                 // CPU Blas 头文件
#include <c10/util/irange.h>                    // C10 范围工具头文件

// 根据宏的定义选择性地引入操作或者完整的操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce.h>
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_dimI_native.h>
#include <ATen/ops/_dimV_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/coalesce_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/_sparse_mask_projection_native.h>
#include <ATen/ops/sparse_resize_and_clear_native.h>
#include <ATen/ops/sparse_resize_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/unique_dim.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#endif

// 进入 at::native 命名空间
namespace at::native {

// 使用 at::sparse 命名空间下的内容
using namespace at::sparse;

/******************************************************************************
 * access methods
 ******************************************************************************/

// 获取稀疏张量 self 的稀疏维度
int64_t sparse_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->sparse_dim();
}

// 获取稀疏张量 self 的密集维度
int64_t dense_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->dense_dim();
}
// 检查给定稀疏张量是否已经压缩稀疏格式
bool is_coalesced_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->coalesced();
}

// 检查给定张量是否为默认的稀疏坐标张量布局，预期为稀疏坐标格式
bool is_coalesced_default(const Tensor& self) {
  TORCH_CHECK(false, "is_coalesced expected sparse coordinate tensor layout but got ", self.layout());
  return false;
}

// 获取给定稀疏张量的非零元素个数
int64_t _nnz_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->nnz();
}

// 获取给定稀疏张量的索引
Tensor _indices_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->indices();
}

// 获取给定稀疏张量的值
Tensor _values_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->values();
}

// 设置给定稀疏张量是否为压缩稀疏格式
Tensor& _coalesced_sparse_(SparseTensor& self, bool coalesced) {
  get_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

// 获取默认张量的索引，预期为稀疏坐标张量布局
Tensor indices_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->indices().alias();
}

// 报告默认张量的索引预期为稀疏坐标张量布局
Tensor indices_default(const Tensor& self) {
  TORCH_CHECK(false, "indices expected sparse coordinate tensor layout but got ", self.layout());
}

// 获取稀疏张量的值，预期为稀疏坐标张量布局
Tensor values_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->values().alias();
}

// 报告默认张量的值预期为稀疏张量布局
Tensor values_default(const Tensor& self) {
  TORCH_CHECK(false, "values expected sparse tensor layout but got ", self.layout());
}

/******************************************************************************
 * creation methods
 * See NOTE [ Sparse: autograd and API ] for details
 ******************************************************************************/

/*** Helper methods ***/

// 创建新的稀疏张量，指定dtype、layout、device和pin_memory
static SparseTensor new_sparse(
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 断言layout值必须为稀疏布局
  AT_ASSERT(layout.has_value() && *layout == kSparse);
  DispatchKey dispatch_key;
  // 根据device类型选择对应的DispatchKey
  switch (device_or_default(device).type()) {
#define DO_CASE(device, _) \
    case DeviceType::device: \
      dispatch_key = DispatchKey::Sparse##device; \
      break;
    C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
    default:
      // 报告不支持给定device类型的稀疏张量
      TORCH_CHECK(false, "device type not supported for sparse ", device_or_default(device))
  }
  // 使用指定的DispatchKeySet和dtype创建SparseTensorImpl对象
  return detail::make_tensor<SparseTensorImpl>(
      DispatchKeySet(dispatch_key),
      scalarTypeToTypeMeta(dtype_or_default(dtype)));
}

/** Actual dispatched creation methods ***/

// 创建指定维度的稀疏张量，指定sparse_dim、dense_dim、size、dtype、layout和device
SparseTensor new_with_dims_sparse(
    int64_t sparse_dim,
    int64_t dense_dim,
    ArrayRef<int64_t> size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    // 创建一个新的稀疏张量 `self`，并初始化为稀疏张量的标准布局和数据类型
    SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
    // 获取稀疏张量 `self` 的实现，并调用其方法 resize_and_clear_ 进行大小调整和清空操作
    get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
    // 返回调整大小和清空后的稀疏张量 `self`
    return self;
SparseTensor new_with_dims_and_tensor_sparse_symint(
    int64_t sparse_dim,                                 // 定义稀疏张量的稀疏维度
    int64_t dense_dim,                                  // 定义稀疏张量的密集维度
    c10::SymIntArrayRef size,                           // 使用对称整数数组引用定义稀疏张量的大小
    const Tensor& indices,                              // 稀疏张量的索引张量
    const Tensor& values,                               // 稀疏张量的值张量
    std::optional<ScalarType> dtype,                    // 可选的数据类型
    std::optional<Layout> layout,                       // 可选的布局
    std::optional<Device> device,                       // 可选的设备
    std::optional<bool> pin_memory,                     // 可选的内存固定标志
    std::optional<bool> is_coalesced) {                 // 可选的合并标志
  // 使用给定的参数创建一个新的稀疏张量
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  // 获取稀疏张量的实现对象
  auto impl = get_sparse_impl(self);
  // 调整稀疏张量的维度和大小
  impl->resize_(sparse_dim, dense_dim, size);
  // 注意: 不能保证 `indices` 和 `values` 不包含 AutogradMeta。
  // 我们希望保持稀疏张量的不变量，即 `indices_` 和 `values_` 不包含 AutogradMeta。
  // 为了达到这个目的，在这里浅拷贝 `indices` 和 `values`。
  auto indices_shallow_copy =
      Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy =
      Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  // 将浅拷贝的 `indices` 和 `values` 别名到稀疏张量 `self` 中
  alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
  // alias_into_sparse 方法会覆盖合并标志，因此在此处重置标志到期望的状态
  if (is_coalesced.has_value()) {
    impl->set_coalesced(*is_coalesced);
  }
  // TODO: alias_into_sparse 方法设置合并标志为 `self._values().shape[0] < 2`。
  // 存在某些方法（例如在 COO 张量上的置换），即使 `dims[0] != 0` 也会强制合并标志为 false。
  // 在估计 is_coalesced 状态时，这些方法可能过于限制性。
  // 条件 `!is_coalesced && self._nnz() < 2` 提供了一种检测和优化这类方法的方法，
  // 关于估计 is_coalesced 状态。

  // 返回创建的稀疏张量
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
Tensor empty_sparse(
    IntArrayRef size,                                   // 稀疏张量的大小
    std::optional<ScalarType> dtype,                    // 可选的数据类型
    std::optional<Layout> layout,                       // 可选的布局
    std::optional<Device> device,                       // 可选的设备
    std::optional<bool> pin_memory,                     // 可选的内存固定标志
    std::optional<MemoryFormat> optional_memory_format) {
  // 检查内存固定标志是否有效，只有密集的 CPU 张量可以被固定
  TORCH_CHECK(
      !pin_memory.has_value() || !*pin_memory,
      "Only dense CPU tensors can be pinned");
  // 调用带有指定参数的 new_with_dims_sparse 方法来创建空的稀疏张量
  return new_with_dims_sparse(
      size.size(), 0, size, dtype, layout, device, pin_memory);
}

/* Shape init */
Tensor sparse_coo_tensor(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
    // 定义一个函数，返回一个 std::optional<bool> 类型的对象，可能包含布尔值，代表是否将内存固定
    std::optional<bool> pin_memory) {
      // 查看注释：TensorOptions 的 hacky wrapper 的移除
      // 创建一个 TensorOptions 对象，设置其数据类型、布局、设备，并根据 pin_memory 参数设置是否固定内存
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    
      // 调用 at::_sparse_coo_tensor_with_dims 函数，创建一个稀疏 COO 张量，设置其维度为 size 的大小，稀疏格式的布局
      return at::_sparse_coo_tensor_with_dims(size.size(), 0, size, options.layout(at::kSparse));
    }
}

/* Pointer-copy init */

// 命名空间，用于定义辅助函数
namespace {
// 如果需要，扩展值的函数
static inline Tensor expand_values_if_needed(const Tensor& values) {
  // 如果值的维度为0
  if (values.dim() == 0) {
    // 模仿 NumPy 的行为，将其视为一维张量进行扩展
    return values.expand({1});
  } else {
    return values;
  }
}
} // namespace

// 创建稀疏 COO 张量的函数
Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values_,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // 构建张量选项
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 如果需要，扩展值
  Tensor values = expand_values_if_needed(values_);

  // 参数检查
  TORCH_CHECK(
      !options.has_layout() || options.layout() == kSparse,
      "expected sparse layout, but got layout ",
      options.layout());
  // 下面的检查是多余的，因为它们也在 SparseTensorImpl::set_indices_and_values_unsafe 中检查，
  // 但我们需要确保它们以便推断形状。
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());

  // 如果没有给出大小，将推断为每个维度的最大索引值
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  // 计算推断的大小
  std::vector<int64_t> computed_sizes(sparse_dim + dense_dim);
  if (indices.numel() > 0) {
    // 如果索引中有元素，则推断最小稀疏维度的大小为索引每个维度的最大值加一
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor computed_indices_sizes =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    computed_indices_sizes.add_(1); // 长度 = 最大索引 + 1
    Tensor cpu_min_indices = min_indices.to(at::DeviceType::CPU);
    Tensor cpu_computed_indices_sizes =
        computed_indices_sizes.to(at::DeviceType::CPU);
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_computed_indices_sizes_accessor =
        cpu_computed_indices_sizes.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      computed_sizes[static_cast<size_t>(d)] =
          cpu_computed_indices_sizes_accessor[d];
    }
  } else {
    // 如果索引中没有元素，则没有足够信息来确定最小稀疏维度的大小，
    // 此时通过推断是不可能的。
    // 这段代码可能不会执行，因为在实际调用中，索引通常是非空的。
    // 对稀疏维度的尺寸进行初始化，设置为0
    for (const auto d : c10::irange(sparse_dim)) {
      computed_sizes[static_cast<size_t>(d)] = 0;
    }
  }
  // 对于密集维度，根据值的尺寸设置对应维度的尺寸
  for (const auto d : c10::irange(dense_dim)) {
    computed_sizes[static_cast<size_t>(sparse_dim + d)] = values.size(d + 1);
  }

  // 调用函数创建一个 COO（Coordinate Format）稀疏张量，并返回结果
  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim,                                      // 稀疏维度
      dense_dim,                                       // 密集维度
      computed_sizes,                                  // 计算得到的各维度的尺寸
      indices,                                         // 稀疏张量的索引
      values,                                          // 稀疏张量的值
      values.options().layout(kSparse),                // 稀疏张量使用稀疏布局
      is_coalesced);                                   // 是否是 coalesced（合并）的张量
// 验证稀疏 COO 张量的参数有效性
void _validate_sparse_coo_tensor_args(
    // indices 参数是稀疏张量的索引
    const Tensor& indices,
    // values_ 参数是稀疏张量的值
    const Tensor& values_,
    // size 参数是张量的尺寸，由稀疏维度和密集维度组成
    ArrayRef<int64_t> size,
    // is_coalesced_ 参数是一个可选的布尔值，指示张量是否已压缩
    std::optional<bool> is_coalesced_) {
  
  // 将 values_ 参数根据需要扩展为 values 张量
  Tensor values = expand_values_if_needed(values_);
  // 如果 is_coalesced_ 未提供值，则默认为 false
  bool is_coalesced = is_coalesced_.value_or(false);

  // 以下检查是冗余的，因为它们也在 SparseTensorImpl::set_indices_and_values_unsafe 中检查，
  // 但我们需要确保它们以推断形状。
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());
  
  // 稀疏维度和密集维度的数量
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  TORCH_CHECK(
      static_cast<int64_t>(size.size()) == sparse_dim + dense_dim,
      "number of dimensions must be sparse_dim (",
      sparse_dim,
      ") + dense_dim (",
      dense_dim,
      "), but got ",
      size.size());

  // 检查确保所有索引在 `size` 的边界内
  if (indices.numel() > 0) {
    // 计算索引的最小和最大值
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor max_indices =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    Tensor cpu_min_indices, cpu_max_indices;
    // 如果索引不在 CPU 上，则将其复制到 CPU
    if (!indices.is_cpu()) {
      cpu_min_indices = min_indices.to(at::DeviceType::CPU);
      cpu_max_indices = max_indices.to(at::DeviceType::CPU);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    // 创建 CPU 访问器以便访问最小和最大索引
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      // 确保在每个维度中索引都在合法范围内
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = size[static_cast<size_t>(d)];
      TORCH_CHECK(
          max_index_in_dim < dim_size,
          "size is inconsistent with indices: for dim ",
          d,
          ", size is ",
          dim_size,
          " but found index ",
          max_index_in_dim);
    }
    // 如果张量已压缩且 values 大小大于 1，则进行额外的一致性检查
    if (is_coalesced && values.size(0) > 1) {
      // 将索引扁平化，并计算相邻索引的差值
      Tensor indices_scalar = flatten_indices(indices, size);
      Tensor diff = indices_scalar.diff();
      TORCH_CHECK(diff.min().item().toLong() > 0, "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor");
    }
  }
}
    // 创建一个 TensorOptions 对象，设置其数据类型、布局、设备和是否固定内存
    TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    // 参数检查
    TORCH_CHECK(
        // 如果 options 指定了布局，则要求布局为稀疏布局
        !options.has_layout() || options.layout() == kSparse,
        // 报错信息：期望稀疏布局，但得到的布局是
        "expected sparse layout, but got layout ",
        options.layout());
    // 调用底层的 _sparse_coo_tensor_unsafe 函数，创建稀疏张量
    return at::native::_sparse_coo_tensor_unsafe(
        // 稀疏张量的索引
        indices,
        // 稀疏张量的值
        values,
        // 稀疏张量的大小
        size,
        // 从 options 中获取的数据类型元数据转换为标量类型
        optTypeMetaToScalarType(options.dtype_opt()),
        // 从 options 中获取的布局选择项
        options.layout_opt(),
        // 从 options 中获取的设备选择项
        options.device_opt(),
        // 从 options 中获取的固定内存选择项
        options.pinned_memory_opt(),
        // 是否强制合并操作
        is_coalesced);
}

// 定义 _sparse_coo_tensor_unsafe 函数，接受稀疏张量的索引、数值、大小等参数
Tensor _sparse_coo_tensor_unsafe(const Tensor& indices, const Tensor& values_, at::IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // 如果全局上下文要求检查稀疏张量不变性，则调用验证参数函数
  if (at::globalContext().checkSparseTensorInvariants()) {
    at::native::_validate_sparse_coo_tensor_args(indices, values_, size, is_coalesced);
  }
  // 调用内部的 _sparse_coo_tensor_unsafe_symint 函数，传递参数并返回稀疏张量
  return at::native::_sparse_coo_tensor_unsafe_symint(indices, values_, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory, is_coalesced);
}

// 注意：_sparse_coo_tensor_unsafe() 与 sparse_coo_tensor() 的不同之处在于，它不检查索引是否超出 `size` 的边界，从而避免从 CUDA 到 CPU 的复制。
// 但是，这个函数应仅在我们确保索引保证在边界内或者调用者在使用张量之前会调用 _validate_sparse_coo_tensor_args 的情况下使用。
// 注意：已经删除了 size == NULL 的情况
Tensor _sparse_coo_tensor_unsafe_symint(const Tensor& indices, const Tensor& values_, c10::SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<bool> is_coalesced) {
  // 见 [Note: hacky wrapper removal for TensorOptions]

  // 如果需要，扩展值张量
  Tensor values = expand_values_if_needed(values_);

  // 这个守卫是有意的：我们不支持沿着索引维度的动态形状，因为这意味着可变的维度性
  auto sparse_dim = indices.sym_size(0).guard_int(__FILE__, __LINE__);
  auto dense_dim = values.dim() - 1;

  // 调用 _sparse_coo_tensor_with_dims_and_tensors_symint 创建稀疏张量，并返回
  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      values.options().layout(kSparse),
      is_coalesced);
}

// 注意：删除了 newWithSizeNd 变体

// 克隆稀疏张量 self，返回另一个稀疏张量 other
SparseTensor clone_sparse(
    const SparseTensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 检查不支持的内存格式选项
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  // 使用 new_with_dims_sparse 创建一个新的稀疏张量 other，并复制数据
  SparseTensor other = new_with_dims_sparse(
      self.sparse_dim(),
      self.dense_dim(),
      self.sizes(),
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  copy_into_sparse(other, self._indices(), self._values(), true);
  // 返回已经合并过的 other 稀疏张量
  return other._coalesced_(self.is_coalesced());
}

/******************************************************************************
 * 重塑方法
 ******************************************************************************/

// 稀疏张量 self 调整大小，并返回自身的引用
const SparseTensor& sparse_resize_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  // 调用 get_sparse_impl(self)->resize_ 来实现重新调整稀疏张量的大小
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  // 返回调整大小后的稀疏张量自身的引用
  return self;
}
const SparseTensor& sparse_resize_and_clear_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  // 调用底层实现，调整稀疏张量的大小并清空内容
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  // 返回调整后的稀疏张量
  return self;
}

namespace {
bool _is_same_size_as_sparse(
    const SparseTensor& self,
    const SparseTensor& src) {
  // 检查两个稀疏张量的稀疏维度、密集维度和大小是否相同
  return self.sparse_dim() == src.sparse_dim() &&
      self.dense_dim() == src.dense_dim() && self.sizes().equals(src.sizes());
}
} // namespace

// 从 native/Resize.cpp 调用（无需动态调度）
const SparseTensor& resize_as_sparse_(const SparseTensor& self, const SparseTensor& src) {
  // 如果 self 和 src 的大小不同，则调整 self 的大小以匹配 src
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
  }
  // 返回调整后的稀疏张量 self
  return self;
}

// 注意：省略了 resizeNd 的变体

SparseTensor& copy_sparse_wrapper_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  // TODO: 一旦 copy_ 完全迁移到使用调度程序，使用调度程序处理命名推断，而不是在每处都进行处理
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    // 如果 self 或 src 不是稀疏张量，则抛出错误
    if (!self.is_sparse() || !src.is_sparse()) {
      AT_ERROR(
          "copy_() between dense and sparse Tensors is not implemented! Found self type = ",
          self.toString(),
          " and src type = ",
          src.toString());
    }
    // 将稀疏张量 src 复制到稀疏张量 self
    at::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  // 如果可能的输出名称不为空，则传播名称
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  // 返回复制后的稀疏张量 self
  return self;
}

SparseTensor& copy_sparse_(
    SparseTensor& self,
    const SparseTensor& src,
    bool non_blocking) {
  // 如果 self 和 src 是同一个张量，则直接返回 self
  if (is_same_tensor(self, src))
    return self;
  // 调整 self 的大小以匹配 src，并将 src 的数据复制到 self 中
  get_sparse_impl(self)->resize_(
      src.sparse_dim(), src.dense_dim(), src.sizes());
  copy_into_sparse(self, src._indices(), src._values(), non_blocking);
  // 返回调整和复制后的稀疏张量 self，并标记为 coalesced 状态与 src 一致
  return self._coalesced_(src.is_coalesced());
}

SparseTensor coalesce(const SparseTensor& self) {
  // 检查输入张量是否为稀疏坐标张量布局
  TORCH_CHECK(self.layout() == kSparse, "coalesce expected sparse coordinate tensor layout but got ", self.layout());
  // 见注释: [ coalesce autograd ]
  // 如果张量已经是 coalesced 状态，则直接返回
  if (self.is_coalesced()) {
    return self;
  }
  // 否则，在原始张量的副本上执行 coalesce 操作
  return at::_coalesce(self);
}

SparseTensor _coalesce_sparse_cpu(const SparseTensor& self) {
  // 断言输入张量已定义
  AT_ASSERT(self.defined());
  // 内部断言，用于排除变量是否在调度程序之外
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  // 断言输入张量是稀疏张量
  AT_ASSERT(self.is_sparse());
  // 内部断言，确保输入张量未 coalesced
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());

  // 注意：由于当 is_coalesced 为 false 时，coalesce 不是原地操作，
  // 我们应该保持原始张量不变，在其副本上执行 coalesce 操作
  // 如果非零元素数量小于 2，则在输入张量的克隆上执行 coalesce 操作
  if (self._nnz() < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
  return dst;
}

// 获取当前 SparseTensor 的索引张量
Tensor indices = self._indices();
// 获取当前 SparseTensor 的值张量，并确保连续存储
Tensor values = self._values().contiguous();
// 获取稀疏张量的稀疏维度和密集维度
int64_t sparse_dim = self.sparse_dim();
int64_t dense_dim = self.dense_dim();
// 获取当前 SparseTensor 的非零元素个数
int64_t nnz = self._nnz();

// 将索引张量展平，以便在一维空间中操作
Tensor indices_scalar = flatten_indices(indices, self.sizes());

// 创建一个新的稀疏张量 dst，使用指定的数据类型、布局、设备和固定内存选项
SparseTensor dst = new_sparse(
    optTypeMetaToScalarType(self.options().dtype_opt()),
    self.options().layout_opt(),
    self.options().device_opt(),
    self.options().pinned_memory_opt());
// 调整 dst 的稀疏表示，设置其稀疏维度、密集维度和尺寸
get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
// 将新创建的稀疏张量与新索引和新值关联起来
alias_into_sparse(dst, newIndices, newValues);

// 对索引张量进行排序，返回排序后的索引缓冲区和排序后的索引排列
auto [indicesBuffer, indicesPermutation] = indices_scalar.sort(0);
// 注：以下访问器依赖于 self._nnz() > 0（在函数前面已经测试过）
auto newIndicesAccessor = newIndices.accessor<int64_t, 2>();
auto indicesAccessor = indices.accessor<int64_t, 2>();
auto indicesPermutationAccessor = indicesPermutation.accessor<int64_t, 1>();
auto indicesBufferAccessor = indicesBuffer.accessor<int64_t, 1>();

int64_t i = -1;
// 使用 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4 宏遍历 values 的数据类型
AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    at::ScalarType::ComplexHalf, at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Bool,
    values.scalar_type(), "coalesce", [&] {
  int64_t prev = -1;
  int64_t blockSize = values.stride(0);
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  scalar_t* newValues_ptr = newValues.data_ptr<scalar_t>();
  // 遍历索引的排序结果
  for (const auto j : c10::irange(nnz)) {
    int64_t pos = indicesPermutationAccessor[j];
    int64_t curr = indicesBufferAccessor[j];
    // 如果当前索引与前一个索引相同，将值复制到新值张量中
    if (curr == prev) {
      // 如果 values 张量不为空，则进行值的累加
      if (values.numel() > 0) {
        at::native::cpublas::axpy<scalar_t>(
            blockSize,
            static_cast<scalar_t>(1),
            values_ptr + pos * blockSize,
            1,
            newValues_ptr + i * blockSize,
            1);
      }
    } else {
      ++i;
      // 复制当前索引到新索引张量中的对应位置
      for (const auto d : c10::irange(sparse_dim)) {
        newIndicesAccessor[d][i] = indicesAccessor[d][pos];
      }
      // 如果 values 张量不为空，则将对应的值复制到新值张量中
      if (values.numel() > 0) {
        at::native::cpublas::copy<scalar_t>(
            blockSize,
            values_ptr + pos * blockSize,
            1,
            newValues_ptr + i * blockSize,
            1);
      }
    }
    prev = curr;
  }
});

// 标记 dst 为紧缩状态（coalesced）
dst._coalesced_(true);
// 设置 dst 的非零元素个数并缩小其内部表示的大小
get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

// 返回处理后的稀疏张量 dst
return dst;
}

DEFINE_DISPATCH(sparse_mask_intersection_out_stub);
// 定义一个分发函数的调度器，用于稀疏掩码交集的操作

DEFINE_DISPATCH(sparse_mask_projection_out_stub);
// 定义一个分发函数的调度器，用于稀疏掩码投影的操作

using OptTensor = std::optional<Tensor>;
// 使用类型别名定义一个可选的张量类型

static std::tuple<Tensor, Tensor, OptTensor> sparse_mask_like_prepare_sparse_inputs(
    const std::string& method_name,
    const Tensor& t,
    const Tensor& mask) {
  // 这是一个辅助函数，用于实现类似 "sparse_mask" 功能的操作，主要是将一个张量的值投影到另一个张量上。
  // 这些操作主要依赖于 COO（Coordinate Format）交集原语，充分利用紧凑的输入以避免任何同步和排序调用。
  // 问题在于，这些原语可能会根据哪些参数被合并以及哪些参数更大，将第一个参数投影到第二个参数，或者反过来。
  // 此函数准备了 `sparse_mask` 的输入，使得 `t` 如果未被合并则通过对 `t` 进行排序来投影到 `mask` 上，
  // 同时将 `mask` 设置为未被合并。
  // 该投影的结果将是未被合并的，因此用户需要根据操作的语义正确设置相应的标志。

  // 我们已经假设 t.sizes() == mask.sizes()
  TORCH_CHECK(t.sparse_dim() == mask.sparse_dim(),
              method_name, "(): the number of sparse dimensions in `self` ",
              "should match that of the `mask`. ",
              "Got `self.sparse_dim() == ", t.sparse_dim(), "` != ",
              "`mask.sparse_dim() == ", mask.sparse_dim(), "`.");

  const auto wrapped_tensor = [](const Tensor& t,
                                 const OptTensor& indices = c10::nullopt,
                                 const OptTensor& values = c10::nullopt) -> Tensor {
    auto res = at::empty({0}, t.options());
    auto* res_sparse_impl = get_sparse_impl(res);
    res_sparse_impl->raw_resize_(t.sparse_dim(), t.dense_dim(), t.sizes());
    const auto res_indices = indices.has_value() ? *indices : t._indices();
    const auto res_values = values.has_value() ? *values : t._values();
    res_sparse_impl->set_indices_and_values_unsafe(res_indices, res_values);
    res_sparse_impl->set_nnz_and_narrow(t._nnz());
    res._coalesced_(false);
    return res;
  };

  auto [lhs, lhs_hash_opt, lhs_is_movable] = [&]() -> auto {
    if (t.is_coalesced()) {
      return std::make_tuple(t, static_cast<OptTensor>(c10::nullopt), false);
      // 如果 t 已经被合并，则直接返回 t，并且索引和值为可选空，不可移动为 false
    } else {
      // 否则，根据 t 的情况返回相应的结果
      // 这里是一个占位符，具体逻辑在实际代码中完成
      // 返回 t 本身，以及其哈希值和可移动性的状态
      // 这里的逻辑尚未完全定义，需要进一步的实现
    }
  }();

  // 返回一个包含 t，mask 和一个可选张量的元组，用于准备稀疏输入
  return std::make_tuple(lhs, mask, lhs_hash_opt);
}
    } else {
      // 计算稀疏张量的压缩表示，并根据其哈希索引排序
      const auto indices_hash = at::sparse::flatten_indices(t._indices(), t.sizes());
      // 对哈希索引进行排序，获取排序后的索引
      const auto argsort_indices_hash = std::get<1>(indices_hash.sort(0));
      // 根据排序后的索引重新选择稀疏张量的索引和数值
      const auto res_indices = t._indices().index_select(1, argsort_indices_hash);
      const auto res_values = t._values().index_select(0, argsort_indices_hash);
      // 根据排序后的索引重新选择哈希索引
      const auto indices_hash_sorted = indices_hash.index_select(0, argsort_indices_hash);
      // 注意：res 可能不是连续的，但是已经排序了。
      // 将 res 标记为“连续”，以便在交集内核中跳过排序步骤。
      auto res = wrapped_tensor(t, res_indices, res_values)._coalesced_(true);
      // 返回结果元组，包括重新组织后的稀疏张量 res、排序后的哈希索引以及 true（表示成功）
      return std::make_tuple(std::move(res), static_cast<OptTensor>(std::move(indices_hash_sorted)), true);
    }
  }();

  // 检查 mask 张量是否是连续的，如果是，使用其包装后的版本 rhs，否则直接使用 mask
  const auto rhs = mask.is_coalesced() ? wrapped_tensor(mask) : mask;
  // 检查 mask 是否可以移动，如果是连续的，rhs_is_movable 为 true，否则为 false
  const auto rhs_is_movable = mask.is_coalesced() ? true : false;

  // 返回最终的元组，包括 lhs（可能为移动或非移动）、rhs（可能为移动或非移动）、以及 lhs 的哈希索引
  return std::make_tuple(lhs_is_movable ? std::move(lhs) : lhs,
                         rhs_is_movable ? std::move(rhs) : rhs,
                         lhs_hash_opt);
}

// 对稀疏张量 t 应用稀疏掩码 mask
SparseTensor sparse_mask(const Tensor& t, const SparseTensor& mask) {
  // 检查稀疏张量和掩码的尺寸是否一致
  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  // 如果 t 和 mask 是同一个对象，则直接返回 t
  if (t.is_same(mask)) {
    return t;
  }

  // 如果 mask 是空或者没有非零元素，则返回 mask 的克隆，并转换到 t 的设备和标量类型
  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(t.device(), t.scalar_type());
  }

  // 如果 t 的布局是稀疏
  if (t.layout() == at::kSparse) {
    // 如果 t 没有非零元素，则创建一个与 mask 相同设备和标量类型的克隆，并将其值置为零
    if (!t._nnz()) {
      auto res = mask.clone().to(t.device(), t.scalar_type());
      res._values().zero_();
      return res;
    }

    // 否则，创建一个空的张量 res，并准备稀疏输入以进行稀疏掩码操作
    auto res = at::empty({0}, t.options());
    auto [lhs, rhs, lhs_hash_opt] = sparse_mask_like_prepare_sparse_inputs("sparse_mask", t, mask);
    sparse_mask_intersection_out_stub(res.device().type(), res, lhs, rhs, lhs_hash_opt);
    // 返回稀疏张量 res，并根据 mask 的合并状态进行整合
    return res._coalesced_(mask.is_coalesced());
  }

  // 如果 t 的布局不是稀疏，生成 mask 的稀疏模板，并使用该模板与 t 进行逐元素乘法操作
  const auto mask_values = mask._values();
  auto mask_template = at::sparse_coo_tensor(
      mask._indices(),
      at::ones({1}, mask_values.options()).expand_as(mask_values),
      mask.sizes())._coalesced_(mask.is_coalesced());
  return t.mul(mask_template).to(t.scalar_type());
}

// 对稀疏张量 t 应用稀疏投影掩码 mask，并返回结果张量
Tensor sparse_mask_projection(const Tensor& t, const Tensor& mask, bool accumulate_matches) {
  // 内部断言确保 t 和 mask 是稀疏张量
  TORCH_INTERNAL_ASSERT(t.is_sparse());
  TORCH_INTERNAL_ASSERT(mask.is_sparse());

  // 检查稀疏投影掩码的尺寸是否与稀疏张量 t 的尺寸一致
  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "_sparse_mask_projection(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  // 如果 t 或者 t 或 mask 中没有非零元素，则返回 t 的克隆并将其值置为零
  if (!t.numel() || !t._nnz() || !mask._nnz()) {
    auto res = t.clone();
    res._values().zero_();
    return res;
  }

  // 否则，创建一个空的张量 res，并准备稀疏输入以进行稀疏投影掩码操作
  auto res = at::empty({0}, t.options());
  auto [lhs, rhs, lhs_hash_opt] = sparse_mask_like_prepare_sparse_inputs("_sparse_mask_projection", mask, t);
  sparse_mask_projection_out_stub(res.device().type(), res, lhs, rhs, lhs_hash_opt, accumulate_matches);
  // 返回稀疏张量 res，并根据 t 的合并状态进行整合
  return res._coalesced_(t.is_coalesced());
}

// 创建与给定稀疏张量 self 类似的空张量
Tensor empty_like_sparse_coo(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 创建张量选项 options_
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查不能同时在选项和显式参数中设置内存格式
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");

  // 合并稀疏张量 self 的选项和提供的选项，并合并内存格式
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  // 检查如果布局不是步进式（strided），则不支持内存格式选项
  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  // 如果选项中布局是稀疏，则创建一个空的稀疏 COO 张量并返回
  if (options.layout() == kSparse) {
    auto result = at::empty({0}, options);
    // 调用result对象的sparse_resize_and_clear_方法，对其进行稀疏张量的重置和清除操作，
    // 参数依次为当前对象的尺寸、稀疏维度和密集维度
    result.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    // 返回经操作后的result对象
    return result;
  } else {
    // 若条件不满足，调用at::native::empty_like函数创建一个与self相同类型和布局的空张量，
    // 可指定数据类型、布局、设备、是否锁定内存以及内存格式等选项
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  }
}

} // namespace at::native
```