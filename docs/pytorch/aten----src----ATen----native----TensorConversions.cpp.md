# `.\pytorch\aten\src\ATen\native\TensorConversions.cpp`

```
// 定义是否仅为 Torch 断言方法运算符开启宏
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的头文件
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 决定是否包含所有操作的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_autocast_to_full_precision_native.h>
#include <ATen/ops/_autocast_to_reduced_precision_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_to_cpu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/to_dense_backward_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_mkldnn_backward_native.h>
#include <ATen/ops/to_native.h>
#include <ATen/ops/to_sparse_bsc_native.h>
#include <ATen/ops/to_sparse_bsr_native.h>
#include <ATen/ops/to_sparse_csc_native.h>
#include <ATen/ops/to_sparse_csr_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zeros.h>
#endif

// 包含稀疏 CSR 张量的实用函数和索引工具
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

// 包含标准库头文件
#include <algorithm>
#include <numeric>

// 定义 ATen native 命名空间
namespace at::native {

// 以下是一个未命名的命名空间，包含了稀疏矩阵转换的辅助函数和准备工作的注释
namespace {
// dense_to_sparse_{csr,bsr,csc,bsc} common helpers

// 准备 N 维稠密到稀疏压缩转换的前期工作。
// 将 N 维输入转换为 3 维（单个批次维度），检查批次维度的乘积是否非零，
// 并且每个批次中包含的稀疏矩阵具有相同数量的非零元素。
// 批次沿着压缩轴连接。可以一次性生成此矩阵的索引，然后进行单步操作。
// 将稠密表示的张量转换为稀疏压缩表示前的准备和检查掩码值，支持批处理。
void dense_to_sparse_compressed_prepare_check_mask_values_batched(
    const Layout& target_layout,  // 目标布局类型，指定稀疏矩阵的布局
    Tensor& values,               // 输入的值张量，可能包含批处理维度
    Tensor& mask,                 // 输入的掩码张量，用于指示稀疏位置，可能包含批处理维度
    const int64_t& n_batch_dim)   // 批处理维度的数量
{
  if (n_batch_dim > 1) {
    // 对于具有多个批处理维度的输入，将它们展平。
    // 输入形状 (b0, b1 ..., bn, r, c) -> (b0 * b1 * ... * bn, r ,c)
    values = values.flatten(0, n_batch_dim - 1);
    mask = mask.flatten(0, n_batch_dim - 1);
  }

  // 根据函数名称生成信息性消息，格式为 "to_sparse_{csr,csc,bsr,bsc}"。
  TORCH_CHECK(
      mask.size(0) > 0,
      "to_sparse_",
      // 我们希望消息与函数名称匹配，因此生成布局的小写首字母缩写
      sparse_csr::layoutToString(target_layout, false, true),
      ": Expected product of batch dimensions to be non-zero.");

  // 计算第一个批次中非零元素的数量，并扩展到整体大小
  auto nse_per_batch = mask.select(0, 0).sum().expand(mask.size(0));
  TORCH_CHECK(
      mask.sum({-2, -1}).equal(nse_per_batch),
      "Expect the same number of specified elements per batch.");

  // 我们需要将批次合并为矩阵，增加压缩轴的长度。这允许我们为压缩矩阵创建索引，
  // 稍后再去除批次 (两个内核)。否则，我们将不得不为每个批次单独创建索引，
  // 需要 n_batch 个内核。对于 csr/bsr，批处理维度已与压缩轴相邻，可以一起展平。
  // 对于 csc/bsc，我们需要首先进行转置。
  // 对于 BSR/CSR (b, r, c) -> (b*r, c)
  // 对于 BSC/CSC (b, c, r) -> (r, b*c)
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      target_layout,
      "dense_to_sparse_compressed",
      [&]() {
        values = values.flatten(0, 1);  // 展平第 0 和 1 维度
        mask = mask.flatten(0, 1);
      },
      [&]() {
        values = values.transpose(0, 1).flatten(1, 2);  // 转置后展平第 1 和 2 维度
        mask = mask.transpose(0, 1).flatten(1, 2);
      });
}

// 此函数将压缩稀疏矩阵的压缩索引展开为批处理的压缩稀疏张量。
// 这类似于一个 unflatten 操作：
// unflatten(0, {b, r}) 适用于 csr/bsr，输入形状为 (r*b, c)
//          (输出形状为 (b, r, c))
// unflatten(1, {b, c}).transpose(0,1) 适用于 csc/bsc，输入形状为 (r, c*b)
//          (输出形状为 (r, b, c) unflatten, (b, r, c) unflatten + 转置)
// 这仅对压缩索引进行操作，因为普通索引和值可以按上述方式处理而无需特殊处理。
// 在进行转换之前，必须确保批处理形状的稀疏模式是合理的前提条件。即每个批次具有相同数量的非零元素。
Tensor compressed_to_batched_compressed_indices(
    const Tensor& compressed_in,
    const int64_t& n_batch,
    // 计算每个批次中压缩数据的数量，除去第一个元素后再分批
    auto n_compressed_per_batch = (compressed_in.size(0) - 1) / n_batch;
    // 根据 out_int32 决定输出的数据类型是 Int 还是 Long
    ScalarType out_type = out_int32 ? ScalarType::Int : ScalarType::Long;
    // 创建一个全零张量作为输出，形状为 (n_batch, n_compressed_per_batch + 1)
    auto batched_out = at::zeros(
        {n_batch, n_compressed_per_batch + 1},
        compressed_in.options().dtype(out_type));

    // 如果每个批次的压缩维度长度大于零，则执行以下操作
    if (n_compressed_per_batch > 0) {
        // 切片压缩索引，忽略第一个零元素并重塑为 n_batch 行
        auto trailing_slice =
            compressed_in.slice(0, 1, c10::nullopt, 1).reshape({n_batch, -1});
        // 再次切片压缩索引，选择对应于批次边界的元素，这些值将是 nnz per batch 的递增倍数。重塑为 n_batch 行（1 列）以进行广播。
        // 这相当于使用相同的重塑形状 arange(n_batch) * nnz_per_batch
        auto offsets = compressed_in.slice(0, 0, -1, n_compressed_per_batch)
                           .reshape({n_batch, -1});
        // 从重塑的压缩索引的每一行中减去偏移量，得到批次内的压缩索引。每行的第一个元素不计算，因为它始终为零。将结果复制到输出缓冲区的视图中。
        batched_out.narrow(-1, 1, n_compressed_per_batch)
            .copy_(trailing_slice - offsets);
    }
    // 返回批次输出张量
    return batched_out;
// After generating member tensors for sparse_compressed matrix, if the target
// shape is N-D we must reform the batch dimensions.
// Single kernel is used to restore one batch dimension in the compressed
// indices. From there full batch shape is restored by reshape. No special
// handling is needed for restoring batch dimensions of the values or
// plain_indices it can be done with reshape/unflatten.
void reshape_2d_sparse_compressed_members_to_nd_batched(
    const IntArrayRef full_sizes,
    const int64_t& n_batch_dim,
    Tensor& compressed_indices,
    Tensor& plain_indices,
    Tensor& values) {
  // Extract the batch shape from full_sizes based on the number of batch dimensions.
  auto batch_shape = full_sizes.slice(0, n_batch_dim);
  
  // Calculate the total number of batches by multiplying the dimensions in batch_shape.
  auto n_batch = std::accumulate(
      batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int64_t>());
  
  // Convert compressed indices to batched compressed indices, assuming consistent nnz per batch.
  compressed_indices = compressed_to_batched_compressed_indices(
      compressed_indices, n_batch, /*out_int32*/ false);

  // Infer the last dimension of batched plain indices, based on nnz or nrow/ncol+1.
  auto batchsize_infer_last = DimVector(batch_shape);
  batchsize_infer_last.push_back(-1);
  plain_indices = plain_indices.reshape(batchsize_infer_last);

  // Infer the last dimension of batched compressed indices, based on nnz or nrow/ncol+1.
  compressed_indices = compressed_indices.reshape(batchsize_infer_last);

  // Unflatten values tensor to handle nnz per batch, compatible with blocked and unblocked layouts.
  values = values.unflatten(0, batchsize_infer_last);
}
    // 检查是否存在布局选项，并确保如果设置了，self 的布局与其一致
    // 如果布局选项与 self 的布局不一致，则抛出错误信息
    TORCH_CHECK(!layout.has_value() || self.layout() == layout.value(),
               "to(options) doesn't support converting to a different layout, "
               "but got self.layout being ", self.layout(),
               " and options.layout set as ", layout.value());
    
    // 创建 TensorOptions 对象 options，并设置其数据类型、布局、设备和固定内存等选项
    auto options = TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device)
        .pinned_memory(pin_memory);
    
    // 如果 options 中指定了设备，则确保设备索引有效
    if (options.has_device()) {
        options = options.device(ensure_has_index(options.device()));
    }
    
    // 将当前张量的选项与新的 options 合并，并设置内存格式为 null
    options = self.options().merge_in(options).memory_format(c10::nullopt);
    
    // 确定最终的内存格式，如果用户没有指定，则使用 Preserve
    auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
    
    // TODO: 使用分发器来处理此部分逻辑。
    // 当前存在未列举的可扩展性问题，阻止了此操作的实现。
    if (self.layout() == kSparse) {
        // 如果当前张量为稀疏张量，确保内存格式为 Preserve，否则抛出错误信息
        TORCH_CHECK(
            memory_format == MemoryFormat::Preserve,
            "to(options): COO only supports memory format Preserve, but got ", memory_format,
            " instead.");
    
        // 如果选项中的设备是元设备，则返回一个与 self 形状一致的零张量
        if (options.device().is_meta()) {
            return zeros_like(self, options);
        }
    
        // 获取当前张量的索引，并将其转换为新的内存格式、数据类型和设备
        auto indices = self._indices();
        const auto new_indices = at::native::to(
            indices,
            indices.scalar_type(),
            c10::kStrided,
            device,
            pin_memory,
            non_blocking,
            true, // 由于在 _to_copy 中，强制复制
            memory_format);
    
        // 获取当前张量的值，并将其转换为新的内存格式、数据类型和设备
        const auto new_values = at::native::to(
            self._values(),
            dtype,
            c10::kStrided,
            device,
            pin_memory,
            non_blocking,
            true, // 由于在 _to_copy 中，强制复制
            memory_format);
  // 如果输入稀疏 COO 格式，则转换为新的稀疏 COO 张量
  return at::_sparse_coo_tensor_unsafe(
      new_indices,
      new_values,
      self.sizes(),
      options, self.is_coalesced());
} else if (at::sparse_csr::is_sparse_compressed(self)) {
  // 检查是否为 CSR 格式，要求内存格式必须为 Preserve
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve,
      "to(options): ", at::sparse_csr::layoutToString(self.layout()),
      " only supports memory format Preserve, but got ", memory_format,
      " instead.");

  // 若 options.device() 为元设备，则返回与 self 相同尺寸的全零张量
  if (options.device().is_meta()) {
    return zeros_like(self, options);
  }

  // 获取压缩索引和普通索引
  auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(self);

  // 将 self.values() 转换为指定的数据类型和设备，要求强制复制
  const auto new_values = at::native::to(
      self.values(),
      dtype,
      c10::kStrided,
      device,
      pin_memory,
      non_blocking,
      true, // 强制复制，因为在 _to_copy 中
      memory_format);

  // 将压缩索引转换为指定的数据类型和设备，要求强制复制
  const auto new_compressed_indices = at::native::to(
      compressed_indices,
      compressed_indices.scalar_type(),
      c10::kStrided,
      device,
      pin_memory,
      non_blocking,
      true, // 强制复制，因为在 _to_copy 中
      memory_format);

  // 将普通索引转换为指定的数据类型和设备，要求强制复制
  const auto new_plain_indices = at::native::to(
      plain_indices,
      plain_indices.scalar_type(),
      c10::kStrided,
      device,
      pin_memory,
      non_blocking,
      true, // 强制复制，因为在 _to_copy 中
      memory_format);

  // 返回新的稀疏压缩张量
  return at::_sparse_compressed_tensor_unsafe(
      new_compressed_indices,
      new_plain_indices,
      new_values,
      self.sizes(),
      options);
}

// 检查是否需要将输出固定在内存中，对于 CUDA 设备上的非重叠且密集张量，并且内存格式为 kStrided
bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                (options.layout() == c10::kStrided));

// 如果内存格式为 Preserve
if (memory_format == MemoryFormat::Preserve) {
  // 如果设备支持按步幅分布，并且输入张量是非重叠且密集的
  if (options.device().supports_as_strided()) {
    if (self.is_non_overlapping_and_dense()) {
      // 如果输入张量是量化的，则创建一个与 self 大小相同的量化张量
      Tensor r;
      if (self.is_quantized()) {
        r = at::empty_quantized(self.sizes(), self, options);
        at::QuantizerPtr quantizer = r.quantizer();
        r.copy_(self, non_blocking);
        set_quantizer_(r, quantizer);
      } else {
        // 否则，创建一个与 self 相同大小和步幅的张量
        r = at::empty_strided(
            self.sizes(),
            self.strides(),
            options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
      }
      // 返回结果张量 r
      return r;
    } else if (!self.is_quantized() && self.layout() == kStrided) {
        // 如果输入张量不是量化的，并且布局为 kStrided，则推断出步幅，并创建相应的张量
        Tensor r;
        auto strides = infer_dense_strides(self.sizes(), self.strides());
        r = at::empty_strided(
            self.sizes(),
            strides,
            options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
        // 返回结果张量 r
        return r;
    } else {
      // 否则，建议使用自动推断的内存格式
      memory_format = self.suggest_memory_format();
    }
  } else {
    // 否则，建议使用自动推断的内存格式
    memory_format = self.suggest_memory_format();

      // 建议使用自动推断的内存格式
      memory_format = self.suggest_memory_format();
    }
  } else {
    // 建议使用自动推断的内存格式
    memory_format = self.suggest_memory_format();
  }
}

// 返回经过格式转换后的张量或建议的内存格式
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  // 根据特定注释说明，此处需要明确传递 nullopt 作为 MemoryFormat 参数
  // TODO: 在这里不能使用 empty_quantized。在调用 empty_affine_quantized/_empty_per_channel_affine_quantized 之前，会在 CheckMemoryFormat.h 中引发异常
  // at::empty 在这里也不适用，因为对量化张量没有适当的 at::empty 支持，会返回一个具有 UnknownQuantizer 的量化张量
  auto r = self.is_quantized() ? at::empty_like(self, memory_format)
                               : at::empty_symint(self.sym_sizes(),
                                 options.memory_format(memory_format).pinned_memory(pin_out), c10::nullopt);
  // 将当前张量 self 的数据异步复制到 r 中
  r.copy_(self, non_blocking);
  // 返回复制后的新张量 r
  return r;
}

// 检查给定的可选值是否为空或与指定值相等
template <typename T>
static inline bool is_null_or_equal_to(const std::optional<T>& test, const T& value) {
  // 如果可选值为空，则返回 true
  if (!test.has_value()) {
    return true;
  }
  // 否则返回可选值是否等于指定值的比较结果
  return test.value() == value;
}

// NOTE: static runtime's to_maybe_copy_out relies on details of this
// check; if you change how it works, please update static runtime as
// well.

// 检查是否可以安全地进行内存别名
bool to_will_alias(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // 获取内存格式，如果未指定则使用默认保留格式
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // 检查数据类型、布局、设备、拷贝标志以及内存格式是否允许内存别名
  return is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
    is_null_or_equal_to(layout, self.layout()) &&
    is_null_or_equal_to(device, self.device()) &&
    !copy &&
    (memory_format == MemoryFormat::Preserve ||
     self.suggest_memory_format() == memory_format);
}

// 对象类型转换的实现，根据给定的参数进行选择性转换
static inline Tensor to_impl(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {

  // 快速路径：如果不需要转换，则直接返回原始张量
  if (to_will_alias(self, dtype, layout, device, copy, optional_memory_format)) {
    return self;
  }
  // 否则调用 _to_copy 进行复制操作
  return at::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, optional_memory_format);
}

// 如果输入张量是 fp32 类型，则将其转换为 fp16 类型，否则保持不变
// （这是 JIT 自动类型转换实现内部使用的函数）
Tensor _autocast_to_reduced_precision(const Tensor& self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) {
  if (self.dtype() == at::ScalarType::Float &&
      ((self.device().is_cuda() && cuda_enabled) ||
      (self.device().is_cpu() && cpu_enabled))
      ) {
    at::ScalarType target = at::ScalarType::Undefined;
    // 根据设备选择目标类型
    if (self.device().is_cuda()) {
      target = cuda_dtype;
    } else if (self.device().is_cpu()) {
      target = cpu_dtype;
    }

    // 断言目标类型合法性
    TORCH_INTERNAL_ASSERT(target != at::ScalarType::Undefined, "_autocast_to_reduced_precision requires legit ScalarType argument for given device");

    // 执行类型转换
    return to_impl(
        self, target, c10::nullopt, c10::nullopt, c10::nullopt, false, false, c10::nullopt);
  } else {
    // 不需要转换时返回原始张量
    return self;
  }
}

// 如果输入张量是 fp16 类型，则将其转换为 fp32 类型，否则保持不变
// （这是 JIT 自动类型转换实现内部使用的函数）
Tensor _autocast_to_full_precision(const Tensor& self, bool cuda_enabled, bool cpu_enabled) {
  if ((self.dtype() == at::ScalarType::Half || self.dtype() == at::ScalarType::BFloat16) &&
      ((self.device().is_cuda() && cuda_enabled) ||
      (self.device().is_cpu() && cpu_enabled))
      ) {
    // 执行类型转换
    return to_impl(
        self, at::ScalarType::Float, c10::nullopt, c10::nullopt, c10::nullopt, false, false, c10::nullopt);
  } else {
    // 不需要转换时返回原始张量
    return self;
  }
}
    return self;
  }



// 返回当前对象的引用，通常用于方法链式调用中
return self;
// 结束当前方法并返回对象自身的引用
}
`
}
# 定义 Tensor 对象的转换函数，接收多个可选参数，包括数据类型、布局、设备、内存 pin 状态、非阻塞标志和内存格式
Tensor to(
  const Tensor& self,                       # 输入 Tensor 对象 self
  std::optional<ScalarType> dtype,           # 可选参数，数据类型 ScalarType
  std::optional<Layout> layout,              # 可选参数，布局 Layout
  std::optional<Device> device,              # 可选参数，设备 Device
  std::optional<bool> pin_memory,            # 可选参数，是否 pin 内存
  bool non_blocking,                          # 非阻塞标志
  bool copy,                                  # 是否复制数据
  std::optional<c10::MemoryFormat> optional_memory_format  # 可选参数，内存格式 MemoryFormat
) {
  # 调用具体实现函数 to_impl，传递参数 self、dtype、layout、设备信息、内存 pin 状态、非阻塞标志、复制标志和内存格式
  return to_impl(
      self,
      dtype,
      layout,
      ensure_has_index(device),               # 确保设备索引有效
      pin_memory,
      non_blocking,
      copy,
      optional_memory_format);
}

# 重载的 Tensor 转换函数，接收 Device 类型的设备参数
Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  # 调用具体实现函数 to_impl，传递参数 self、dtype、布局为空、设备信息、内存 pin 状态为空、非阻塞标志、复制标志和内存格式
  return to_impl(
      self,
      dtype,
      nullopt,                                 # 布局为空
      ensure_has_index(device),                # 确保设备索引有效
      nullopt,                                 # 内存 pin 状态为空
      non_blocking,
      copy,
      optional_memory_format);
}

# 重载的 Tensor 转换函数，仅接收数据类型、非阻塞标志、复制标志和内存格式
Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  # 调用具体实现函数 to_impl，传递参数 self、dtype、布局为空、设备为空、内存 pin 状态为空、非阻塞标志、复制标志和内存格式
  return to_impl(
      self,
      dtype,
      nullopt,                                 # 布局为空
      nullopt,                                 # 设备为空
      nullopt,                                 # 内存 pin 状态为空
      non_blocking,
      copy,
      optional_memory_format);
}

# 重载的 Tensor 转换函数，接收另一个 Tensor 对象作为设备参数
Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  # 获取其他 Tensor 对象的选项（数据类型、布局、设备、内存 pin 状态）
  auto options = other.options();
  # 调用具体实现函数 to_impl，传递参数 self、其他 Tensor 的数据类型、布局、设备、内存 pin 状态、非阻塞标志、复制标志和内存格式
  return to_impl(
      self,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      non_blocking,
      copy,
      optional_memory_format);
}

# 该操作对于懒惰/图形后端特别重要，逐个 Tensor 转换到 CPU
std::vector<Tensor> _to_cpu(TensorList tensors) {
    std::vector<Tensor> cpu_tensors;          # 创建一个用于存储 CPU Tensor 的向量
    for (const auto& t : tensors) {            # 遍历输入的 Tensor 列表
        cpu_tensors.push_back(t.cpu());        # 将每个 Tensor 转换到 CPU 并添加到向量中
    }
    return cpu_tensors;                       # 返回包含所有 CPU Tensor 的向量
}

# 定义 to_dense 的反向传播函数，接收梯度、输入 Tensor 和可选的遮罩标记
Tensor to_dense_backward(const Tensor& grad, const Tensor& input_, std::optional<bool> masked_grad_) {
  /*
    For historical reasons, to_dense backward implements masked
    semantics for sparse tensors, that is, gradients with respect to
    unspecified elements are ignored.  The masked_grad kw argument of
    to_dense is introduced to allow to_dense to be used in the
    non-masked semantics context. However, for BC reasons, the default
    value to masked_grad kw argument is set True as a first instance.
    Eventually, we should eliminate the masked_grad kw argument and
    let to_dense backward to behave according to non-masked
    semantics. Masked semantics of tensors is implemented in the
    framework of masked tensors.
  */
  const auto input_layout = input_.layout();  # 获取输入 Tensor 的布局
  const bool masked_grad = masked_grad_.value_or(true);  # 获取或设置遮罩标记，默认为 true
  switch (input_layout) {                      # 根据输入 Tensor 的布局执行不同的操作
    case kStrided:
      # TODO: 返回梯度本身，尚未实现
      return grad.to_dense(input_.scalar_type(), masked_grad_);
    case kSparse:
      // 如果输入稀疏，则执行以下操作。自动求导假设无重复值。
      if (masked_grad) {
        // 如果需要掩码梯度，则对输入进行稀疏化并返回稀疏掩码后的梯度
        return grad.sparse_mask(input_.coalesce());
      } else {
        // 否则，返回输入的稀疏表示的梯度
        // TODO: return grad as it is
        return grad.to_sparse(input_.sparse_dim());
      }
    case kSparseCsr:
    case kSparseCsc:
      // 如果输入是 CSR 或者 CSC 格式的稀疏张量
      // TODO: 为 sparse_mask 添加高效的 CSR/CSC 支持
      if (masked_grad) {
        // 如果需要掩码梯度，则对输入进行稀疏化，并根据输入布局返回稀疏掩码后的梯度
        return grad.sparse_mask(input_.to_sparse(input_.sparse_dim())).to_sparse(input_layout);
      } else {
        // 否则，返回输入的稀疏表示的梯度
        // TODO: return grad as it is
        return grad.to_sparse(input_layout, /*blocksize=*/c10::nullopt, /*dense_dim=*/input_.dense_dim());
      }
    case kSparseBsr:
    case kSparseBsc: {
      // 如果输入是 BSR 或者 BSC 格式的稀疏张量
      // TODO: 为 sparse_mask 添加高效的 BSR/BSC 支持
      const auto blocksize = at::sparse_csr::getBlockSize(input_);
      if (masked_grad) {
        // 如果需要掩码梯度，则对输入进行稀疏化，并根据输入布局和块大小返回稀疏掩码后的梯度
        return grad.sparse_mask(input_.to_sparse(input_.sparse_dim())).to_sparse(input_layout, blocksize);
      } else {
        // 否则，返回输入的稀疏表示的梯度
        // TODO: return grad as it is
        return grad.to_sparse(input_layout, blocksize, input_.dense_dim());
      }
    }
    case kMkldnn:
      // 如果输入是 MKLDNN 格式的张量，则将梯度转换为 MKLDNN 格式
      return grad.to_mkldnn(input_.scalar_type());
    default:
      // 如果输入布局不受支持，则抛出错误并返回空张量
      AT_ERROR("to_dense_backward: Unsupported input layout: ", input_layout);
      return Tensor{};
  }
}

// 将稀疏格式的自动求导张量转换为密集张量
Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  // 断言输入张量的布局是连续的
  AT_ASSERT(input_.layout() == c10::kStrided);
  // 将梯度张量转换为与输入张量相同数据类型的密集张量
  return grad.to_dense(input_.scalar_type());
}

// 将张量转换为密集布局，支持可选的数据类型和梯度遮罩
Tensor to_dense(const Tensor& tensor, std::optional<c10::ScalarType> dtype, std::optional<bool> masked_grad) {
  // 如果输入张量是稀疏布局，则调用对应的_to_dense方法
  if (tensor.layout() == c10::kSparse) {
    return tensor._to_dense(dtype, masked_grad);
  }
  // 如果输入张量是CSR、CSC、BSR或BSC稀疏布局之一，则同样调用对应的_to_dense方法
  if (tensor.layout() == c10::kSparseCsr ||
      tensor.layout() == c10::kSparseCsc ||
      tensor.layout() == c10::kSparseBsr ||
      tensor.layout() == c10::kSparseBsc) {
    return tensor._to_dense(dtype, masked_grad);
  }
  // 如果输入张量是MKLDNN布局，则同样调用对应的_to_dense方法
  if (tensor.layout() == c10::kMkldnn) {
    return tensor._to_dense(dtype, masked_grad);
  }
  // 如果输入张量是连续布局，则根据是否指定数据类型返回对应的张量
  TORCH_CHECK(
      tensor.layout() == c10::kStrided,
      "to_dense does not support layout ",
      tensor.layout());
  if (dtype) {
    return tensor.to(*dtype);
  }
  return tensor;
}

// 将稀疏张量转换为密集张量，不支持数据类型参数
Tensor sparse_to_dense(const Tensor& self, std::optional<ScalarType> dtype, std::optional<bool> masked) {
  TORCH_CHECK(
      !dtype.has_value(), "dtype argument is not supported by sparse_to_dense");
  // 根据稀疏张量的尺寸创建全零的密集张量，并将稀疏张量加到其中
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

// 将压缩稀疏张量转换为密集张量，不支持数据类型参数
Tensor sparse_compressed_to_dense(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<bool> masked_grad) {
  TORCH_CHECK(
      !dtype.has_value(),
      "dtype argument is not supported by sparse_csr_to_dense");

  // 如果张量元素数量为零，返回全零的密集张量
  if (self.numel() == 0) {
    return at::zeros(self.sizes(), self.options().layout(kStrided));
  }

  auto batch_ndim = sparse_csr::numBatchDimensions(self);

  auto compressed_rows = self.layout() == kSparseCsr || self.layout() == kSparseBsr;
  auto block_sparse = self.layout() == kSparseBsr || self.layout() == kSparseBsc;

  auto [compressed_indices, plain_indices] =
      sparse_csr::getCompressedPlainIndices(self);

  auto values = self.values();
  Tensor dense = at::zeros(self.sizes(), self.options().layout(kStrided));

  // 如果存在多个批次维度，将它们展平为单一的批次维度
  if (batch_ndim == 0) {
    // 扩展形状以处理非批次化张量，最终会消除虚构的末尾批次维度
    compressed_indices.unsqueeze_(0);
    plain_indices.unsqueeze_(0);
    values.unsqueeze_(0);
    dense.unsqueeze_(0);
  }
  if (batch_ndim > 1) {
    // 展平批次维度
    compressed_indices = compressed_indices.flatten(0, batch_ndim - 1);
    plain_indices = plain_indices.flatten(0, batch_ndim - 1);
    values = values.flatten(0, batch_ndim - 1);
    dense = dense.flatten(0, batch_ndim - 1);
  }

  // 现在只有一个批次维度，可能已经存在或从多个批次维度中展平。现在，重塑结果密集矩阵，使单一批次维度与稀疏维度结合成一个维度，以便最终剩下的维度只有块维度和密集维度。
  auto n_batch = values.size(0);
  int64_t nrows, ncols;
  auto dense_reshaped_sizes = dense.sizes().vec();
  if (!block_sparse) {
  // 获取当前维度的大小作为行数
  nrows = self.size(batch_ndim);
  // 获取下一个维度的大小作为列数
  ncols = self.size(batch_ndim + 1);
  // 移除 dense_reshaped_sizes 的前两个元素
  dense_reshaped_sizes.erase(dense_reshaped_sizes.begin(), dense_reshaped_sizes.begin() + 2);
} else {
  // 创建一个包含两个元素的数组，作为块的大小，基于 values 的维度信息
  std::array<int64_t, 2> blocksize = {values.size(2), values.size(3)};
  // 计算行数为当前维度大小除以块的第一个维度大小
  nrows = self.size(batch_ndim) / blocksize[0];
  // 计算列数为下一个维度大小除以块的第二个维度大小
  ncols = self.size(batch_ndim + 1) / blocksize[1];
  // 更新 dense_reshaped_sizes 的第二个和第三个元素为块的大小
  dense_reshaped_sizes[1] = blocksize[0];
  dense_reshaped_sizes[2] = blocksize[1];
}
// 计算 dense 的第一个维度大小为批次数乘以行数乘以列数
dense_reshaped_sizes[0] = n_batch * nrows * ncols;
// 重新整形 dense 张量为指定大小的形状
dense = dense.reshape(dense_reshaped_sizes);

// 计算稀疏矩阵中非零元素的批次、行和列索引，
// 并利用这些索引计算在上述重塑后的 dense 矩阵中的对应索引。
// 然后，通过将稀疏矩阵的值添加到以这种方式计算的元素中，更新 dense 矩阵。
auto options = compressed_indices.options();
// 获取每个批次中非零元素的数量
auto nnz_per_batch = values.size(1);
// 创建批次索引张量，重复非零元素数量次数
auto batch_indices = at::arange(0, n_batch, options).repeat_interleave(nnz_per_batch);
// 确定压缩索引的长度
auto ncompressed = compressed_rows ? nrows : ncols;
// 计算跨所有批次的压缩索引
auto compressed_indices_over_all_batches =
  at::cat({compressed_indices.slice(1, 0, ncompressed).flatten()
          + nnz_per_batch * at::arange(0, n_batch, options).repeat_interleave(ncompressed),
          n_batch * nnz_per_batch * at::ones({1}, options)});
// 转换稀疏矩阵的索引表示，从 CSR 格式转换为 COO 格式
Tensor indices = at::_convert_indices_from_csr_to_coo(
    compressed_indices_over_all_batches,
    plain_indices.flatten(),
    false,
    !compressed_rows);
// 获取行索引和列索引
auto row_indices = indices.select(0, 0);
auto col_indices = indices.select(0, 1);
// 如果使用压缩行格式，则减去批次索引乘以行数
if (compressed_rows) {
  row_indices -= batch_indices * nrows;
} else {
  // 否则减去批次索引乘以列数
  col_indices -= batch_indices * ncols;
}
// 计算偏移量
auto offsets = col_indices + row_indices * ncols + batch_indices * nrows * ncols;
// 将 values 展平并按照指定维度进行加法索引更新 dense
dense.index_add_(0, offsets, values.flatten(0, 1));

// 反展开结果。如果不是块稀疏，则使用原始的 self.sizes() 进行最终重塑，可能会消除额外的批次维度。
if (!block_sparse) {
  return dense.reshape(self.sizes());
} else {
  // 否则，使用 unflatten 将 dense 进行未展开，然后转置并按原始形状进行重塑
  return dense
    .unflatten(0, {-1, nrows, ncols})
      .transpose(2, 3)
      .reshape(self.sizes());
}
}

// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
inline SymDimVector compute_strides_for_view_dtype_downsize(SymIntArrayRef old_strides, int64_t size_ratio, ScalarType old_dtype, ScalarType new_dtype) {
  // 获取张量维度数
  const int64_t ndim = old_strides.size();

  // 检查最后一个维度步长是否为1，用于确保能将旧数据类型视作新数据类型
  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  // 创建新的步长向量，对每个维度进行计算
  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    // 根据尺寸比率计算每个维度的新步长
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  // 最后一个维度步长设为1
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
inline SymDimVector compute_strides_for_view_dtype_upsize(SymIntArrayRef old_strides, int64_t size_ratio, ScalarType old_dtype, ScalarType new_dtype) {
  // 获取张量维度数
  const int64_t ndim = old_strides.size();

  // 检查最后一个维度步长是否为1，用于确保能将旧数据类型视作新数据类型
  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  // 创建新的步长向量，对每个维度进行计算
  SymDimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    // 检查每个维度的旧步长是否能被尺寸比率整除
    TORCH_CHECK(
      (old_strides[dim_idx] % size_ratio) == 0,
      "self.stride(", dim_idx, ") must be divisible by ", size_ratio,
      " to view ", old_dtype, " as ", new_dtype, " (different element sizes), ",
      "but got ", old_strides[dim_idx]);

    // 根据尺寸比率计算每个维度的新步长
    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  // 最后一个维度步长设为1
  new_strides[ndim - 1] = 1;
  return new_strides;
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  // 如果张量已经是目标数据类型，则直接返回
  if (self.scalar_type() == dtype) {
    return self;
  }

  // 获取目标数据类型的元数据
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);

  // 检查是否支持共轭视图的张量转换
  TORCH_CHECK(!self.is_conj(),
    "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");

  // 检查是否支持带有负数位设置的张量转换
  TORCH_CHECK(!self.is_neg(),
    "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  // 获取张量元素的大小和目标元素的大小
  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  // 获取张量的存储
  Storage storage = self.storage();

  // 创建新的张量对象
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();

  // 根据元素大小不同的情况设置张量的大小和步长
  if (self_element_size == new_element_size) {
    impl->set_sizes_and_strides(self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());

  } else if (self.dim() == 0) {
    // 张量维度为0时，不能进行视图转换
    TORCH_CHECK(false,
      "self.dim() cannot be 0 to view ", self.scalar_type(), " as ",
      dtype, " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // 当元素大小减小时

    // 计算元素大小的比率
    int64_t size_ratio = self_element_size / new_element_size;
    // 计算新视图的步长，用于降低数据类型大小
    auto new_strides = compute_strides_for_view_dtype_downsize(
      self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    // 获取当前张量的符号化尺寸
    auto old_sizes = self.sym_sizes();
    // 创建一个新的符号化尺寸向量，并复制旧尺寸数据
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    // 调整最后一个维度的尺寸，使其乘以大小比率
    new_sizes[self.dim() - 1] *= size_ratio;

    // 计算新的符号化存储偏移量
    auto new_storage_offset = size_ratio * self.sym_storage_offset();

    // 调用实现的方法，设置新的尺寸、步长和存储偏移量
    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);

  } else {
    // 增大元素大小的情况

    // 计算元素大小的比率
    int64_t size_ratio = new_element_size / self_element_size;

    // 检查最后一个维度的尺寸是否能被比率整除
    TORCH_CHECK(
      (self.sym_size(-1) % size_ratio) == 0,
      "self.size(-1) must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), ",
      "but got ", self.sym_size(-1));

    // 检查符号化存储偏移量是否能被比率整除
    TORCH_CHECK(
      (self.sym_storage_offset() % size_ratio) == 0,
      "self.storage_offset() must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), but got ",
      self.sym_storage_offset());

    // 计算新视图的步长，用于增大数据类型大小
    auto new_strides = compute_strides_for_view_dtype_upsize(
      self.sym_strides(), size_ratio, self.scalar_type(), dtype);

    // 获取当前张量的符号化尺寸
    auto old_sizes = self.sym_sizes();
    // 创建一个新的符号化尺寸向量，并复制旧尺寸数据
    SymDimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    // 调整最后一个维度的尺寸，使其除以大小比率
    new_sizes[self.dim() - 1] /= size_ratio;

    // 计算新的符号化存储偏移量
    auto new_storage_offset = self.sym_storage_offset() / size_ratio;

    // 调用实现的方法，设置新的尺寸、步长和存储偏移量
    impl->set_sizes_and_strides(new_sizes, new_strides, new_storage_offset);
  }

  // 返回新的张量对象
  return new_tensor;
{



// 此函数将矩阵转换为一系列块
//
// 给定矩阵：
//
//  1  2  3  4
//  5  6  7  8
//  9 10 11 12
// 14 15 16 17
//
// _tile_tensor(matrix, {2, 2}) 将产生以下 2x2 块
//
//  1  2 |  3  4 |  9 10 | 11 12
//  5  6 |  7  8 | 14 15 | 16 17
//
//  通过一个形状为 (2, 2, 2, 2) 的四维张量
//
static Tensor _tile_tensor(const Tensor& self, IntArrayRef blocksize) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);  // 断言块大小的第一个维度大于 0
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);  // 断言块大小的第二个维度大于 0
    auto block_size_0 = self.size(0) / blocksize[0];  // 计算第一个块的大小
    auto block_size_1 = self.size(1) / blocksize[1];  // 计算第二个块的大小

    auto new_shape = DimVector({block_size_0, blocksize[0], block_size_1, blocksize[1]});
    new_shape.append(DimVector(self.sizes().slice(2, self.dim() - 2)));
    return self.reshape(new_shape)
        .transpose(1, 2)
        .contiguous();
}

// 如果 self 是三维的，则与 _tile_tensor 相同，但是针对 self 的每个矩阵条目
static Tensor _batch_tile_tensor(const Tensor& self, IntArrayRef blocksize, const int64_t dense_dim) {
    if (self.dim() == 2 + dense_dim) {
        return _tile_tensor(self, blocksize);  // 如果 self 是二维加 dense_dim，则调用 _tile_tensor
    }
    auto n_batch_dim = self.dim() - 2 - dense_dim;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[0] > 0);  // 断言块大小的第一个维度大于 0
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(blocksize[1] > 0);  // 断言块大小的第二个维度大于 0
    auto block_size_0 = self.size(n_batch_dim) / blocksize[0];  // 计算第一个块的大小
    auto block_size_1 = self.size(n_batch_dim + 1) / blocksize[1];  // 计算第二个块的大小
    auto tiled_sizes = DimVector(self.sizes().slice(0, n_batch_dim));
    tiled_sizes.push_back(block_size_0);
    tiled_sizes.push_back(blocksize[0]);
    tiled_sizes.push_back(block_size_1);
    tiled_sizes.push_back(blocksize[1]);
    tiled_sizes.append(DimVector(self.sizes().slice(n_batch_dim + 2, dense_dim)));
    return self.reshape(tiled_sizes).transpose(n_batch_dim + 1, n_batch_dim + 2).contiguous();
}

// 此函数返回一个向量，其中包含给定布尔掩码为 true 的索引位置
static Tensor _mask_to_indices(const Tensor& mask) {
    TORCH_CHECK(mask.dim() == 1, "Currently _mask_to_indices only supports 1-d masks.");  // 断言掩码是一维的
    TORCH_CHECK(mask.dtype() == at::kBool, "Expected mask to be of dtype bool.");  // 断言掩码的数据类型为 bool
    return at::native::arange(
        mask.numel(), at::kLong, kStrided, mask.device())
        .masked_select(mask);  // 返回掩码为 true 的索引
}

// 此函数将非零掩码转换为列和行索引对
static std::pair<Tensor, Tensor> _not_zero_mask_to_col_row_indices(
    Tensor not_zero_mask,
    ScalarType index_dtype,
    // 根据非零掩码的尺寸，使用给定的索引数据类型和设备创建列索引
    auto col_indices =
        at::native::arange(not_zero_mask.size(-1), index_dtype, kStrided, index_device)
            // 将一维索引视图扩展为与非零掩码相同的形状
            .view({1, not_zero_mask.size(-1)})
            // 按照非零掩码进行选择，仅保留非零元素的索引
            .expand_as(not_zero_mask)
            .masked_select(not_zero_mask);
    
    // 根据非零掩码的尺寸，使用给定的索引数据类型和设备创建行索引
    auto row_indices =
        at::native::arange(
            not_zero_mask.size(-2), index_dtype, kStrided, index_device)
            // 将一维索引视图扩展为与非零掩码相同的形状
            .view({not_zero_mask.size(-2), 1})
            // 按照非零掩码进行选择，仅保留非零元素的索引
            .expand_as(not_zero_mask)
            .masked_select(not_zero_mask);
    
    // 返回由列索引和行索引组成的标准库中的 pair 对象
    return std::pair<Tensor, Tensor>(col_indices, row_indices);
}

// Sparse layout conversions Start
// 定义内联函数，检查稀疏布局转换函数的参数
static inline
void _to_sparse_check_arguments(const std::string& funcname, const Tensor& self, const int64_t sparse_dim) {
  // 获取输入张量的布局信息
  auto layout_from = self.layout();
  // 目标布局设为稀疏布局
  auto layout_to = kSparse;

  // 检查输入布局是否有效，必须是Strided、Sparse或稀疏压缩格式之一
  auto layout_from_valid = layout_from == kStrided || layout_from == kSparse || at::sparse_csr::is_sparse_compressed(layout_from);
  if (!layout_from_valid) {
    // 抛出错误，显示无效的源布局信息
    AT_ERROR(funcname, ": unexpected source layout ", layout_from);
  }

  // 如果源布局是Strided
  if (layout_from == kStrided) {
    // 如果sparse_dim为0且张量维度大于0，则抛出错误
    if (sparse_dim == 0 && self.dim() > 0) {
      AT_ERROR(funcname, ": sparse_dim argument must be in >0 when self.dim()>0");
    }
    // 如果sparse_dim小于0或大于张量的维度数，则抛出错误
    if (sparse_dim < 0 || sparse_dim > self.dim()) {
      AT_ERROR(funcname, ": sparse_dim argument must be in [0,", self.dim(), "] range, but ", sparse_dim, " is given");
    }
  } 
  // 如果源布局是Sparse
  else if (layout_from == kSparse) {
    // 如果sparse_dim与张量的稀疏维度不匹配，则抛出错误
    if (sparse_dim != self.sparse_dim()) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " with sparse_dim argument !=self.sparse_dim() is not supported");
    }
  } 
  // 如果源布局是稀疏压缩格式
  else if (at::sparse_csr::is_sparse_compressed(layout_from)) {
    // 如果sparse_dim不等于2，则抛出错误，因为这种情况下只支持sparse_dim为2
    if (sparse_dim != 2) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " with sparse_dim argument !=2 is not supported");
    }
  }
}

// 定义内联函数，检查稀疏布局转换函数的参数
static inline
void _to_sparse_check_arguments(const std::string& funcname, const Tensor& self, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 获取输入张量的布局信息
  auto layout_from = self.layout();
  // 如果未指定目标布局，则默认为稀疏布局
  auto layout_to = layout.value_or(kSparse);

  // 检查输入布局是否有效，必须是Strided、Sparse或稀疏压缩格式之一
  auto layout_from_valid = layout_from == kStrided || layout_from == kSparse || at::sparse_csr::is_sparse_compressed(layout_from);
  if (!layout_from_valid) {
    // 抛出错误，显示无效的源布局信息
    AT_ERROR(funcname, ": unexpected source layout ", layout_from);
  }

  // 检查目标布局是否有效，必须是Strided、Sparse或稀疏压缩格式之一
  auto layout_to_valid = layout_to == kStrided || layout_to == kSparse || at::sparse_csr::is_sparse_compressed(layout_to);
  if (!layout_to_valid) {
    // 抛出错误，显示无效的目标布局信息
    AT_ERROR(funcname, ": unexpected source layout ", layout_from);
  }

  // 如果源布局是Sparse且目标布局不是Sparse
  if (layout_from == kSparse && layout_to != kSparse) {
    // 如果张量的稀疏维度不等于2，则抛出错误
    if (self.sparse_dim() != 2) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " for input tensors with sparse_dim()!=2 is not supported");
    }
  }

  // 如果源布局是SparseCsr或SparseCsc且目标布局是SparseBsr或SparseBsc
  if ((layout_from == kSparseCsr || layout_from == kSparseCsc) &&
      (layout_to == kSparseBsr || layout_to == kSparseBsc)) {
    // 如果输入张量具有批处理维度，则抛出错误，因为不支持批处理输入
    if (sparse_csr::numBatchDimensions(self) > 0) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " for batched inputs is not supported");
    }
  }

  // 如果指定了块大小
  if (blocksize.has_value()) {
    // 如果块大小不是大小为2的元组，则抛出错误
    if (blocksize.value().size() != 2) {
      AT_ERROR(funcname, ": blocksize needs to be a tuple of size 2, but got ", blocksize.value().size());
    }
    // 获取块大小并进行检查，必须为正数
    auto blocksize_to = *blocksize;
    if (blocksize_to[0] <= 0 || blocksize_to[1] <= 0) {
      AT_ERROR(funcname, ": blocksize needs to be positive, but got ", blocksize_to);
    }
  }
    # 检查目标布局是否为稀疏的 Bsr 或 Bsc
    if (layout_to == kSparseBsr || layout_to == kSparseBsc) {
      # 如果源布局也是稀疏的 Bsr 或 Bsc
      if (layout_from == kSparseBsr || layout_from == kSparseBsc) {
        # 获取源张量的块大小
        auto blocksize_from = at::sparse_csr::getBlockSize(self);
        # 如果目标布局的块大小与源布局的不同，则抛出错误
        if (!(blocksize_to == blocksize_from)) {
          AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " with blocksize changed from ", blocksize_from, " to ", blocksize_to, " is not supported");
        }
      } else {
        # 如果源布局不是稀疏的 Bsr 或 Bsc，则确定稠密维度并计算稀疏行和列的维度
        auto dense_dim = (layout_from == kStrided) ? dense_dim_opt.value_or(0) : self.dense_dim();
        auto sparse_row_dim = -(dense_dim + 2);
        auto sparse_col_dim = -(dense_dim + 1);
        # 检查张量的稀疏行和列维度是否能够被给定的块大小整除，否则抛出错误
        if ((self.size(sparse_row_dim) % blocksize_to[0] != 0) ||
            (self.size(sparse_col_dim) % blocksize_to[1] != 0)) {
            AT_ERROR(funcname, ": tensor sparse size (", self.size(sparse_row_dim), ",", self.size(sparse_row_dim), ") must be divisible by given blocksize (", blocksize_to[0], ",", blocksize_to[1], ")");
        }
      }
    } else {
      # 如果目标布局不是稀疏的 Bsr 或 Bsc，则抛出错误，说明不支持带有块大小参数的转换
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " with blocksize argument given is not supported");
    }
  } else {
    # 如果目标布局不是稀疏的 Bsr 或 Bsc，并且源布局也不是稀疏的 Bsr 或 Bsc，则抛出错误，说明不支持未提供块大小参数的转换
    if ((layout_to == kSparseBsr || layout_to == kSparseBsc) &&
        !(layout_from == kSparseBsr && layout_from == kSparseBsc)) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " without blocksize argument given is not supported");
    }
  }

  # 如果提供了 dense_dim_opt 参数
  if (dense_dim_opt.has_value()) {
    # 如果源布局不是 kStrided，则抛出错误，说明不支持带有 dense_dim 参数的转换
    if (layout_from != kStrided) {
      AT_ERROR(funcname, ": conversion from ", layout_from, " to ", layout_to, " with dense_dim argument given is not supported");
    }

    # 获取 dense_dim 参数的值
    auto dense_dim = *dense_dim_opt;
    # 如果目标布局是稀疏的
    if (layout_to == kSparse) {
      # 如果 dense_dim 等于张量的维度且张量维度大于0，则抛出错误，说明 dense_dim 参数必须不等于 self.dim() 当 self.dim()>0
      if (dense_dim == self.dim() && self.dim() > 0) {
        AT_ERROR(funcname, ": dense_dim argument must be !=self.dim() when self.dim()>0");
      }
      # 如果 dense_dim 小于0或大于张量的维度，则抛出错误，说明 dense_dim 参数必须在 [0, self.dim()] 范围内
      if (dense_dim < 0 || dense_dim > self.dim()) {
        AT_ERROR(funcname, ": dense_dim argument must be in [0,", self.dim(), "] range, but ", dense_dim, " is given");
      }
    } else {
      # 如果目标布局不是稀疏的，则检查 dense_dim 参数是否在有效范围内
      if (dense_dim < 0 || dense_dim > self.dim() - 2) {
        AT_ERROR(funcname, ": dense_dim argument must be in [0,", self.dim() - 2, "] range, but ", dense_dim, " is given");
      }
    }
  }
}

template<Layout target_layout>
static Tensor dense_to_sparse_compressed(const Tensor& self, const Tensor& self_mask, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确保目标布局参数为稀疏矩阵的有效布局之一
  static_assert(target_layout == Layout::SparseCsr || target_layout == Layout::SparseCsc
                || target_layout == Layout::SparseBsr || target_layout == Layout::SparseBsc,
                "invalid layout template parameter for dense_to_sparse_compressed");
  // 确定是否为压缩的行布局和是否为块布局
  constexpr auto compressed_rows_layout = target_layout == Layout::SparseCsr || target_layout == Layout::SparseBsr;
  constexpr auto blocked_layout = target_layout == Layout::SparseBsr || target_layout == Layout::SparseBsc;

  // 获取稠密维度的值或使用默认值0
  int64_t dense_dim = dense_dim_opt.value_or(0);

  // 重新整形值张量，确保块维度被显式添加，并计算一个仅包含批处理和稀疏维度的掩码张量
  auto n_batch_dim = self.dim() - 2 - dense_dim;
  auto is_batched = n_batch_dim > 0;
  auto values = blocked_layout ? _batch_tile_tensor(self, blocksize, dense_dim) :  self;
  auto not_zero_mask = blocked_layout ? _batch_tile_tensor(self_mask, blocksize, dense_dim) : self_mask;
  if (blocked_layout || dense_dim > 0) {
    // 创建一个减少维度的向量，用于求和操作
    std::vector<int64_t> reduce_dim((blocked_layout ? 2 : 0) + dense_dim);
    std::iota(reduce_dim.begin(), reduce_dim.end(), n_batch_dim + 2);
    // 计算非零掩码的和，以确定稀疏矩阵是否具有非零元素
    not_zero_mask = not_zero_mask.sum(reduce_dim) != 0;
  }

  if (is_batched) {
    // 如果存在批处理维度，准备进行转换
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        target_layout, values, not_zero_mask, n_batch_dim);
  }

  // 计算稀疏矩阵的行和列索引，然后根据目标布局计算相应的压缩和稀疏索引
  // 使用上述的掩码张量生成稀疏矩阵的值张量
  Tensor row_indices;
  Tensor col_indices;
  Tensor compressed_indices;
  if (compressed_rows_layout) {
    // 对于压缩的行布局，将非零掩码转换为行和列索引
    std::tie(col_indices, row_indices) = _not_zero_mask_to_col_row_indices(
        not_zero_mask, at::kLong, not_zero_mask.device());
    // 将 COO 格式的索引转换为 CSR 格式的压缩索引
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        row_indices, not_zero_mask.size(0), false /*out_int32*/);
    {
      // 将掩码索引转换为扁平化的值张量，并根据掩码进行选择
      auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
      values = values.flatten(0, 1).index_select(0, mask_indices);
    }
  } else {
    // 对于非压缩的行布局，转置掩码张量并获取列和行索引
    std::tie(row_indices, col_indices) = _not_zero_mask_to_col_row_indices(
       not_zero_mask.transpose(1, 0), at::kLong, not_zero_mask.device());
    // 将 COO 格式的索引转换为 CSR 格式的压缩索引
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        col_indices, not_zero_mask.size(-1), false /*out_int32*/);
    {
      // 计算非零元素的位置索引，并根据转置后的掩码生成
      auto mask_indices = _mask_to_indices(not_zero_mask.transpose(0, 1).flatten());
      // 根据掩码索引选择对应的数值，进行展平和重新排序
      values = values.transpose(0, 1).flatten(0, 1).index_select(0, mask_indices);
    }
    
    Tensor& plain_indices = compressed_rows_layout ? col_indices : row_indices;
    
    if (is_batched) {
      // 如果数据是分批次的，恢复批次维度和压缩维度
      reshape_2d_sparse_compressed_members_to_nd_batched(
          self.sizes(), n_batch_dim, compressed_indices, plain_indices, values);
    }
    
    // 使用目标布局选项创建压缩稀疏矩阵
    return at::_sparse_compressed_tensor_unsafe(
          compressed_indices,
          plain_indices,
          values,
          self.sizes(),
          self.options().layout(target_layout));
// 将稠密张量按照给定的掩码转换为稀疏张量，支持不同的稀疏布局
Tensor dense_to_sparse_with_mask(const Tensor& self, const Tensor& mask, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确定输出的稀疏布局，如果未指定则使用默认的稀疏布局 kSparse
  auto layout_to = layout.value_or(kSparse);
  // 检查输入张量与输出布局不同的情况，应该不相同
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "dense_to_sparse: unexpected same input and output layout");
  // 检查输入张量与掩码张量的布局是否相同
  TORCH_INTERNAL_ASSERT(self.layout() == mask.layout(),
                        "dense_to_sparse_with_mask: expected mask layout ", self.layout(), ", got ", mask.layout());
  // 检查输入张量与掩码张量的尺寸是否相同
  TORCH_INTERNAL_ASSERT(self.sizes() == mask.sizes(),
                        "dense_to_sparse_with_mask: expected mask size ", self.sizes(), ", got ", mask.sizes());
  // 检查并初始化稀疏转换的相关参数
  _to_sparse_check_arguments("dense_to_sparse_with_mask", self, layout, blocksize, dense_dim_opt);

  // 根据目标布局类型进行不同的稀疏转换
  switch (layout_to) {
  case kSparse:
    // 对于 kSparse 布局，使用掩码将输入张量转换为稀疏张量
    return self.sparse_mask(mask.to_sparse(self.dim() - dense_dim_opt.value_or(0)));
  case kSparseCsr:
    // 对于 kSparseCsr 布局，调用压缩稀疏转换函数
    return dense_to_sparse_compressed<Layout::SparseCsr>(self, mask, {}, dense_dim_opt);
  case kSparseCsc:
    // 对于 kSparseCsc 布局，调用压缩稀疏转换函数
    return dense_to_sparse_compressed<Layout::SparseCsc>(self, mask, {}, dense_dim_opt);
  case kSparseBsr:
    // 对于 kSparseBsr 布局，调用压缩稀疏转换函数
    return dense_to_sparse_compressed<Layout::SparseBsr>(self, mask, *blocksize, dense_dim_opt);
  case kSparseBsc:
    // 对于 kSparseBsc 布局，调用压缩稀疏转换函数
    return dense_to_sparse_compressed<Layout::SparseBsc>(self, mask, *blocksize, dense_dim_opt);
  default:
    break;
  }

  // 如果未识别的布局类型，抛出错误
  AT_ERROR("dense_to_sparse_with_mask: ", self.layout(), " to ", layout_to, " conversion not supported");
  return Tensor{};  // 返回空张量作为默认返回值
}

// 将稠密张量转换为 CSR 格式的稀疏张量
Tensor dense_to_sparse_csr(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsr;
  // 检查并初始化稀疏转换的相关参数
  _to_sparse_check_arguments("dense_to_sparse_csr", self, layout_to, {}, dense_dim_opt);

  // 调用压缩稀疏转换函数，将输入张量按照 CSR 布局转换为稀疏张量
  return dense_to_sparse_compressed<Layout::SparseCsr>(self, self != 0, {}, dense_dim_opt);
}

// 将稠密张量转换为 CSC 格式的稀疏张量
Tensor dense_to_sparse_csc(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsc;
  // 检查并初始化稀疏转换的相关参数
  _to_sparse_check_arguments("dense_to_sparse_csc", self, layout_to, {}, dense_dim_opt);

  // 调用压缩稀疏转换函数，将输入张量按照 CSC 布局转换为稀疏张量
  return dense_to_sparse_compressed<Layout::SparseCsc>(self, self != 0, {}, dense_dim_opt);
}

// 将稠密张量转换为 BSR 格式的稀疏张量
Tensor dense_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsr;
  // 检查并初始化稀疏转换的相关参数
  _to_sparse_check_arguments("dense_to_sparse_bsr", self, layout_to, blocksize, dense_dim_opt);

  // 调用压缩稀疏转换函数，将输入张量按照 BSR 布局转换为稀疏张量
  return dense_to_sparse_compressed<Layout::SparseBsr>(self, self != 0, blocksize, dense_dim_opt);
}

// 将稠密张量转换为 BSC 格式的稀疏张量
Tensor dense_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsc;
  // 检查并初始化稀疏转换的相关参数
  _to_sparse_check_arguments("dense_to_sparse_bsc", self, layout_to, blocksize, dense_dim_opt);

  // 调用压缩稀疏转换函数，将输入张量按照 BSC 布局转换为稀疏张量
  return dense_to_sparse_compressed<Layout::SparseBsc>(self, self != 0, blocksize, dense_dim_opt);
}
// 将稠密张量转换为稀疏张量
Tensor dense_to_sparse(const Tensor& self, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确定输出的布局，默认为稀疏布局
  auto layout_to = layout.value_or(kSparse);
  // 检查输入张量的布局与输出布局是否相同，如果相同则抛出错误
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "dense_to_sparse: unexpected same input and output layout");
  // 检查转换参数的有效性
  _to_sparse_check_arguments("dense_to_sparse", self, layout, blocksize, dense_dim_opt);

  // 根据输出布局类型进行不同的稀疏张量转换
  switch (layout_to) {
  case kSparse:
    // 对稠密张量进行稀疏表示转换，减少的维度为 dense_dim_opt 指定的维度或者默认为 0
    return self.to_sparse(self.dim() - dense_dim_opt.value_or(0));
  case kSparseCsr:
    // 使用 CSR 格式将稠密张量转换为稀疏张量
    return self.to_sparse_csr(dense_dim_opt);
  case kSparseCsc:
    // 使用 CSC 格式将稠密张量转换为稀疏张量
    return self.to_sparse_csc(dense_dim_opt);
  case kSparseBsr:
    // 使用 BSR 格式将稠密张量转换为稀疏张量，使用指定的块大小
    return self.to_sparse_bsr(*blocksize, dense_dim_opt);
  case kSparseBsc:
    // 使用 BSC 格式将稠密张量转换为稀疏张量，使用指定的块大小
    return self.to_sparse_bsc(*blocksize, dense_dim_opt);
  default:
    break;
  }

  // 如果指定的输出布局类型不在已支持的类型中，抛出错误
  AT_ERROR("dense_to_sparse: ", self.layout(), " to ", layout_to, " conversion not supported");
  return Tensor{};  // 返回空张量作为默认值
}

// 将稠密张量转换为稀疏张量，只指定稀疏维度
Tensor dense_to_sparse(const Tensor& self, int64_t sparse_dim) {
  // 检查转换参数的有效性
  _to_sparse_check_arguments("dense_to_sparse", self, sparse_dim);

  // 获取输入张量的维度
  int64_t dims = self.dim();
  // 设置稀疏张量的选项，布局为稀疏
  at::TensorOptions sparse_options = self.options().layout(kSparse);
  // 获取输入张量的尺寸
  std::vector<int64_t> sizes = self.sizes().vec();
  // 获取非零元素的索引并转置
  Tensor nz = self.nonzero().transpose(0, 1);

  // 如果非零元素个数为 0，则创建一个空的稀疏张量
  if (nz.size(1) == 0) {
    auto sparse = new_with_dims_sparse(
        sparse_dim,
        dims - sparse_dim,
        sizes,
        optTypeMetaToScalarType(sparse_options.dtype_opt()),
        sparse_options.layout_opt(),
        sparse_options.device_opt(),
        sparse_options.pinned_memory_opt());
    return sparse._coalesced_(true);  // 返回合并的稀疏张量
  }

  Tensor indices;
  // 如果稀疏维度等于输入张量的维度，则直接复制非零元素的索引
  if (sparse_dim == dims) {
    indices = nz.clone();
  } else {
    // 否则，只取前 sparse_dim 维度的非重复索引
    Tensor i = nz.narrow(0, 0, sparse_dim);
    std::tie(indices, std::ignore, std::ignore) = unique_dim(i, 1);
    indices = indices.contiguous(); // 许多稀疏 CUDA 内核要求连续性，参见问题 #12633
  }

  Tensor values;
  // 如果输入张量的维度大于 0，则根据索引获取对应的值
  if (self.dim() > 0) {
    auto ix = toListOfOptionalTensors(indices.chunk(indices.size(0), 0));
    values = self.index(ix).squeeze(0).clone(at::MemoryFormat::Preserve);
  } else {
    // 如果输入张量维度为 0，说明 nz 是形状为 (0, 1) 的克隆张量
    AT_ASSERT(nz.sizes().equals({0, 1}));
    // 根据稀疏张量不变性，values 应该是形状为 (1,) 的张量
    values = self.unsqueeze(0).clone(at::MemoryFormat::Preserve);
  }

  // 创建 COO 格式的稀疏张量
  Tensor sparse = at::sparse_coo_tensor(indices, values, sizes, sparse_options);
  return sparse._coalesced_(true);  // 返回合并的稀疏张量
}

// 辅助函数，将压缩稀疏张量转换为反向的稀疏张量
static Tensor sparse_compressed_to_flipped(
    const Tensor& self,
    std::optional<IntArrayRef> blocksize,
    const std::string& name) {
  // 获取当前张量的布局
  const auto layout = self.layout();
  // 注意：仅适用于非压缩稀疏布局的情况。
  const auto flipped_layout = at::sparse_csr::flip_compressed_layout(layout);

  // 假设compressed_indices表示输入的行在CSR或BSR稀疏压缩格式中的批处理。
  // 为了将批处理的CSR/BSR索引转换为批处理的CSC/BSC索引，执行以下步骤：
  // 1. 将表示形状为(b, r, c)矩阵批次的稀疏压缩索引转换为表示形状为(b * r, c)单个矩阵的稀疏压缩索引。
  // 2. 将形状为(b * r, c)矩阵的压缩索引转换为COO索引。
  // 3. 将这些COO索引映射到形状为(r, b * c)矩阵的COO索引，使得如果A是形状为(b * r, c)的矩阵，
  //    B是形状为(r, b * c)的矩阵，且对所有k在arange(b)中，
  //    A[(k * r):(k * r + r), :] = B[:, (k * c):(k * c + c)]，则A[i, j] = B[i', j']。
  //    这相当于查找与垂直平铺的矩阵值匹配的与水平平铺的相同矩阵值的索引。
  // 4. 将COO索引转换为CSC/BSC索引并形成输出。
  //
  // 注意：垂直/水平平铺的原因是为了能够在单个内核调用中转换整个批次中所有矩阵的索引，
  //       因为所有现有的coo <-> 压缩索引转换方法都假定单个矩阵。
  //
  // 对于CSC/BSC输入，类似地处理带有“transposed”参数。
  // 详细说明每个步骤如何执行，请参见下面的注释。

  Tensor compressed_indices, plain_indices;
  // 获取压缩的稀疏索引和普通的索引
  std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(self);
  // 获取张量的值
  auto values = self.values();
  // 获取非零元素的数量
  const auto nnz = plain_indices.size(-1);

  // 获取批次数目，减去1是因为压缩索引张量的最后一个维度不是批次维度
  const auto n_batches = compressed_indices.dim() - 1;
  auto n_batches_nonzero = n_batches;
  // 为了简化操作，在没有批次时插入虚拟批次维度
  if (!n_batches) {
    n_batches_nonzero = 1;
    compressed_indices.unsqueeze_(0);
    plain_indices.unsqueeze_(0);
    values.unsqueeze_(0);
  }

  // 注意：这些sparse_dim仅适用于CSR/CSC输入的真实稀疏维度。
  // 对于BSR/BSC输入，这些是真实稀疏维度/块大小。换句话说，sparse_dim存储行/列维度中有效索引的范围。
  const auto sparse_dim = [&]() -> at::DimVector {
    auto sparse_dim = at::DimVector(self.sizes().slice(n_batches, 2));
    if (layout == at::kSparseBsr || layout == at::kSparseBsc) {
      auto blocksize = at::sparse_csr::getBlockSize(self);
      sparse_dim[0] /= blocksize[0];
      sparse_dim[1] /= blocksize[1];
    }
    // 返回计算后的稀疏维度向量
    return sparse_dim;
  };
  return sparse_dim;
}();

// batch_sizes_nonempty stores at least one, potentially fake, batch dimension.
// rebatch_sizes_nonempty is equivalent to batch_sizes_nonempty.push_back(-1),
// and is used to unflatten batch dimensions from a dimension of size
// (batch_numel * dim_size,) for some dim_size.
const auto batch_sizes_nonempty = at::DimVector(plain_indices.sizes().slice(0, n_batches_nonzero));
auto rebatch_sizes_nonempty = at::DimVector(batch_sizes_nonempty);
rebatch_sizes_nonempty.push_back(-1);
const auto batch_numel_nonzero = std::accumulate(
    batch_sizes_nonempty.begin(),
    batch_sizes_nonempty.begin() + n_batches_nonzero,
    1,
    std::multiplies<int64_t>());

// Equivalent to (arange(batch_numel_nonzero).mul_(nnz)).reshape(batch_sizes_nonempty).
// We just compute it differently to use `add` kernel in place of `mul` for better
// performance.
const auto batch_nnz_offset = [&]() -> Tensor {
  // Create a tensor containing `nnz` using the options of `compressed_indices`.
  const auto wrapped_nnz = at::tensor({nnz}, compressed_indices.options());
  // Expand `wrapped_nnz` to match the size of `batch_numel_nonzero`,
  // compute cumulative sum along the last dimension, subtract `wrapped_nnz`,
  // and reshape it to `batch_sizes_nonempty`.
  auto offset = wrapped_nnz
    .expand({batch_numel_nonzero})
    .cumsum(-1).sub_(wrapped_nnz)
    .reshape(batch_sizes_nonempty);
  return offset;
}();

// Step 1 for CSR/BSR inputs:
// Convert a sparse compressed index representing batches of matrices of
// shape (b, r, c) to a sparse compressed index that represents a single
// matrix of shape (b * r, c).
// The algorithm is identical for CSC/BSC inputs, with the batch dimensions
// flattened in the "transposed" dimension.
const auto compressed_indices_2d = [&]() -> Tensor {
  // Extract offsets only relevant for the first :-1 elements in a row/col.
  const auto compressed_offsets = compressed_indices.slice(-1, 0, -1);
  // `batch_nnz_offset` offsets each individual matrix row/col offsets by the total
  // sum of nnz's of all the matrices with the smaller batch index.
  const auto batch_offsets = batch_nnz_offset
    .unsqueeze(-1).expand_as(compressed_offsets);
  // `compressed_offsets + batch_offsets` creates an offset vector for a 2d matrix
  // that is stored in a compressed sparse format.
  const auto compressed_offsets_2d = compressed_offsets.add(batch_offsets).reshape({-1});
  const auto offsets_len = compressed_offsets_2d.numel();
  // Create an empty tensor to hold the final compressed indices for the 2D matrix.
  auto res = at::empty({offsets_len + 1}, compressed_indices.options());
  // Copy `compressed_offsets_2d` into `res`, leaving the last element to be filled.
  res.slice(-1, 0, -1).copy_(compressed_offsets_2d);
  // Fill the last element of `res` with `nnz * batch_numel_nonzero`.
  res.slice(-1, -1).fill_(nnz * batch_numel_nonzero);
    return res;
  }();
  // 对于压缩索引更为复杂，但对于 plain_indices 和 values 相对简单：
  // 只需压缩批处理维度。
  const auto plain_indices_2d = plain_indices.flatten(0, n_batches_nonzero);
  // 注意：values 并非二维！它们只表示稀疏压缩的二维矩阵的值。
  const auto values_2d = values.flatten(0, n_batches_nonzero);

  const auto is_out_int32 = compressed_indices.scalar_type() == ScalarType::Int;

  // Step 2 & 3:
  //
  // 将形状为 (b * r, c) 的压缩索引转换为 COO 索引。
  //
  // 将这些 COO 索引映射到形状为 (r, b * c) 的矩阵的 COO 索引，
  // 使得如果 A 是形状为 (b * r, c) 的矩阵，B 是形状为 (r, b * c) 的矩阵，
  // 满足 A[(k * r):(k * r + r), :] = B[:, (k * c):(k * c + c)] 对于所有 k 属于 arange(b)，
  // 那么 A[i, j] = B[i', j']。
  // 这等同于找到与垂直堆叠的矩阵的值匹配的水平堆叠的相同矩阵的索引。

  // COO <-> 稀疏索引转换假设 CSR/BSR 输入。
  // 对于 CSC/BSC 输入，这些索引将呈现“转置”状态。
  const auto is_transposed_indices = layout == at::kSparseCsc || layout == at::kSparseBsc;
  const auto coo_indices_2d_transposed = [&]() -> Tensor {
    auto coo_indices_2d = _convert_indices_from_csr_to_coo(
        compressed_indices_2d,
        plain_indices_2d,
        is_out_int32,
        /*transpose=*/true); // 翻转行和列以方便操作。
    // 将形状为 (b * r, c) 的 COO 索引转换为 (r, b * c)。
    // 这是一个映射 (i, j) -> {
    //    b = i // r
    //    i' = i % r
    //    j' = j + b * c
    //    return (i', j')
    // }
    // 注意：我们在上面使用了 transpose=true！
    auto i = coo_indices_2d.select(0, 1);
    auto j = coo_indices_2d.select(0, 0);
    auto b = i.div(is_transposed_indices ? sparse_dim[1] : sparse_dim[0], "trunc");
    // 原地修改 i, j。
    i.fmod_(is_transposed_indices ? sparse_dim[1] : sparse_dim[0]);
    j.add_(b * (is_transposed_indices ? sparse_dim[0] : sparse_dim[1]));
    return coo_indices_2d;
  }();

  // Step 4:
  // 将 COO 索引转换为 CSC/BSC 索引并形成输出。
  // 需要沿着“转置”维度对 COO 索引进行排序，以满足排序后的普通索引不变性。
  // 使用哈希函数将 COO 索引转换为线性偏移，其中在“转置”维度上放置更多的“权重”（即步长）。
  const auto coo_indices_2d_transposed_hashed = at::sparse::flatten_indices(
      coo_indices_2d_transposed,
      is_transposed_indices ? at::DimVector({sparse_dim[0], sparse_dim[1] * batch_numel_nonzero})
                            : at::DimVector({sparse_dim[1], sparse_dim[0] * batch_numel_nonzero}));

  // 对哈希后的 COO 索引进行排序
  const auto hash_argsort = std::get<1>(coo_indices_2d_transposed_hashed.sort());

  // 根据排序后的索引重新排列 COO 索引
  const auto coo_indices_2d_transposed_sorted = coo_indices_2d_transposed.index_select(1, hash_argsort);

  // 提取新的压缩索引和普通索引
  const auto new_compressed_indices_coo_2d = coo_indices_2d_transposed_sorted.select(0, 0);
  const auto new_plain_indices_2d = coo_indices_2d_transposed_sorted.select(0, 1);

  // 根据排序后的索引重新排列数值
  const auto new_values_2d = values_2d.index_select(0, hash_argsort);

  // 转换新的压缩索引为批处理压缩索引
  auto new_compressed_indices = compressed_to_batched_compressed_indices(
      _convert_indices_from_coo_to_csr(
        new_compressed_indices_coo_2d,
        is_transposed_indices
          ? batch_numel_nonzero * sparse_dim[0]
          : batch_numel_nonzero * sparse_dim[1],
        is_out_int32),
      batch_numel_nonzero,
      is_out_int32)
    .unflatten(0, batch_sizes_nonempty);

  // 根据重新排列的尺寸重新排列普通索引
  auto new_plain_indices = new_plain_indices_2d.unflatten(0, rebatch_sizes_nonempty);

  // 根据重新排列的尺寸重新排列数值
  auto new_values = new_values_2d.unflatten(0, rebatch_sizes_nonempty);

  // 如果没有批次，则删除插入的虚假批次维度
  if (!n_batches) {
    new_compressed_indices.squeeze_(0);
    new_plain_indices.squeeze_(0);
    new_values.squeeze_(0);
  }

  // 返回稀疏压缩张量的不安全版本
  return _sparse_compressed_tensor_unsafe(
      new_compressed_indices,
      new_plain_indices,
      new_values,
      self.sizes(),
      self.options().layout(flipped_layout));
}

// 将稀疏压缩格式转换为稀疏 CSR 格式的张量
Tensor sparse_compressed_to_sparse_csr(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  // 目标布局设置为 SparseCsr
  auto layout_to = kSparseCsr;
  // 检查输入张量的布局是否与目标布局不同，否则抛出错误
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_compressed_to_sparse_csr: unexpected same input and output layout");
  // 检查转换参数并验证输入张量
  _to_sparse_check_arguments("sparse_compressed_to_sparse_csr", self, layout_to, {}, dense_dim_opt);

  // 如果输入张量布局为 SparseCsc，则调用相应函数进行转换
  if (self.layout() == kSparseCsc) {
    return sparse_compressed_to_flipped(self, c10::nullopt, "to_sparse_csr");
  }

  // 如果布局不是 SparseCsr 或 SparseCsc，则抛出错误
  AT_ERROR("sparse_compressed_to_sparse_csr: expected SparseCsr or SparseCsc layout but got ", self.layout());
  return Tensor{};
}

// 将稀疏压缩格式转换为稀疏 CSC 格式的张量
Tensor sparse_compressed_to_sparse_csc(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  // 目标布局设置为 SparseCsc
  auto layout_to = kSparseCsc;
  // 检查输入张量的布局是否与目标布局不同，否则抛出错误
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_compressed_to_sparse_csc: unexpected same input and output layout");
  // 检查转换参数并验证输入张量
  _to_sparse_check_arguments("sparse_compressed_to_sparse_csc", self, layout_to, {}, dense_dim_opt);

  // 如果输入张量布局为 SparseCsr，则调用相应函数进行转换
  if (self.layout() == kSparseCsr) {
    return sparse_compressed_to_flipped(self, c10::nullopt, "to_sparse_csc");
  }

  // 如果布局不是 SparseCsr 或 SparseCsc，则抛出错误
  AT_ERROR("sparse_compressed_to_sparse_csc: expected SparseCsr or SparseCsc layout but got ", self.layout());
  return Tensor{};
}

// 将 COO 格式的张量转换为稀疏 CSR 格式的张量
Tensor coo_to_sparse_csr(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  // 目标布局设置为 SparseCsr
  auto layout_to = kSparseCsr;
  // 检查转换参数并验证输入张量
  _to_sparse_check_arguments("coo_to_sparse_csr", self, layout_to, {}, dense_dim_opt);

  // 对输入张量进行合并操作，得到稀疏张量
  auto coalesced_self = self.coalesce();
  // 提取行索引
  auto row_indices = coalesced_self.indices()[0];
  // 判断行索引是否为 int32 类型
  bool out_int32 = (row_indices.scalar_type() == at::kInt);
  // 将 COO 格式的行索引转换为 CSR 格式的行偏移索引
  auto crow_indices = at::_convert_indices_from_coo_to_csr(
      row_indices, self.size(0), out_int32);
  // 使用不安全操作创建稀疏 CSR 格式的张量
  return at::native::_sparse_csr_tensor_unsafe(
      crow_indices,
      coalesced_self.indices()[1].contiguous(),
      coalesced_self.values(),
      coalesced_self.sizes(),
      coalesced_self.scalar_type(),
      c10::kSparseCsr,
      coalesced_self.device());
}

// 将 COO 格式的张量转换为稀疏 CSC 格式的张量
Tensor coo_to_sparse_csc(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  // 目标布局设置为 SparseCsc
  auto layout_to = kSparseCsc;
  // 检查转换参数并验证输入张量
  _to_sparse_check_arguments("coo_to_sparse_csc", self, layout_to, {}, dense_dim_opt);

  // 对输入张量进行转置并转换为稀疏 CSR 格式的张量
  auto transposed_csr = self.transpose(0, 1).to_sparse_csr(dense_dim_opt);
  // 使用不安全操作创建稀疏 CSC 格式的张量
  return at::native::_sparse_csc_tensor_unsafe(
      transposed_csr.crow_indices(),
      transposed_csr.col_indices(),
      transposed_csr.values(),
      self.sizes(),
      transposed_csr.scalar_type(),
      c10::kSparseCsc,
      transposed_csr.device());
}

// 将 COO 格式的张量转换为稀疏 BSR 格式的张量
Tensor coo_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 目标布局设置为 SparseBsr
  auto layout_to = kSparseBsr;
  // 检查转换参数并验证输入张量
  _to_sparse_check_arguments("coo_to_sparse_bsr", self, layout_to, blocksize, dense_dim_opt);

  // 将输入张量先转换为稀疏 CSR 格式，再转换为稀疏 BSR 格式
  return self.to_sparse_csr(dense_dim_opt).to_sparse_bsr(blocksize);
}
// 定义函数 coo_to_sparse_bsc，将 COO 格式的稀疏张量转换为块压缩稀疏（BSC）格式
Tensor coo_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 设置目标稀疏张量格式为块压缩稀疏（BSC），并检查参数有效性
  auto layout_to = kSparseBsc;
  _to_sparse_check_arguments("coo_to_sparse_bsc", self, layout_to, blocksize, dense_dim_opt);

  // 调用 self 的方法将其转换为压缩稀疏列（CSC）格式，再转换为块压缩稀疏（BSC）格式并返回
  return self.to_sparse_csc(dense_dim_opt).to_sparse_bsc(blocksize);
}

namespace {
// 定义模板函数 convert_indices_from_coo_to_csr_cpu，用于在 CPU 上从 COO 到 CSR 索引的转换
template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cpu(
    const Tensor& result,          // 输出结果张量
    const Tensor& input,           // 输入 COO 格式的索引张量
    const int64_t size) {          // 结果张量的大小
  int64_t numel = input.numel();   // 获取输入张量的元素个数
  const input_t* data_in = input.const_data_ptr<input_t>();  // 获取输入张量的数据指针
  output_t* data_out = result.data_ptr<output_t>();          // 获取输出结果张量的数据指针

  // 如果输入张量为空，将结果张量置零并返回
  if (numel == 0) {
    result.zero_();
    return;
  }

  // 初始化 CSR 索引的第一行
  for (int64_t i = 0; i <= data_in[0]; i++)
    data_out[i] = static_cast<output_t>(0);

  // 并行处理主体部分的 CSR 索引转换过程
  at::parallel_for(
      0, numel - 1, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
        input_t curr_value = data_in[start], next_value;
        for (const auto i : c10::irange(start, end)) {
          next_value = data_in[i + 1];
          for (; curr_value < next_value; curr_value++)
            data_out[curr_value + 1] = static_cast<output_t>(i + 1);
        }
      });

  // 处理最后一行的 CSR 索引
  for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++) {
    data_out[i] = static_cast<output_t>(numel);
  }
}

// 定义模板函数 convert_indices_from_csr_to_coo_cpu，用于在 CPU 上从 CSR 到 COO 索引的转换
template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cpu(
    const Tensor& indices,         // 输出 COO 格式的索引张量
    const Tensor& crow_indices,    // 输入 CSR 格式的行索引张量
    const Tensor& col_indices,     // 输入 CSR 格式的列索引张量
    const bool transpose = false) {// 是否转置的标志，默认为 false
  int64_t nrows = crow_indices.size(-1) - 1;  // 获取行数
  int64_t nnz = col_indices.size(-1);         // 获取非零元素个数

  // 当行数或非零元素个数为 0 时，将输出索引张量置零并返回
  if (nrows == 0 || nnz == 0) {
    indices.zero_();
    return;
  }

  // 确保行索引张量是连续的
  auto crow_indices_ = crow_indices.expect_contiguous();

  int64_t total_nnz = col_indices.numel();     // 获取总的非零元素个数
  int64_t batch_ndim = crow_indices.dim() - 1; // 获取批处理维度数

  // 如果存在批处理维度，将部分张量进行缩窄以处理批处理
  if (batch_ndim > 0) {
    auto batch_indices = indices.narrow(0, 0, batch_ndim);
    batch_indices.copy_(at::sparse::full_coo_indices(crow_indices.sizes().slice(0, batch_ndim), crow_indices.options())
                        .repeat_interleave(nnz, 1));

# 复制操作：使用稀疏张量的 COO 索引生成函数创建一个与 `crow_indices` 相同大小的张量，并在第一维度上重复 `nnz` 次数。

  }
  const input_t* crow_indices_data_in = crow_indices_->const_data_ptr<input_t>();
  // 获取 `crow_indices_` 的常量数据指针，类型为 `input_t`

  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  // 断言张量 `indices` 是连续存储的

  auto row0 = indices.select(0, transpose ? batch_ndim + 1 : batch_ndim + 0);
  // 根据 `transpose` 的值选择 `indices` 的第一维度中的子张量 `row0`

  auto row1 = indices.select(0, transpose ? batch_ndim + 0 : batch_ndim + 1);
  // 根据 `transpose` 的值选择 `indices` 的第一维度中的另一个子张量 `row1`

  output_t* data_out = row0.data_ptr<output_t>();
  // 获取 `row0` 的数据指针，类型为 `output_t`

  auto col_indices_ = col_indices.expect_contiguous();
  // 获取列索引 `col_indices_`，确保其是连续存储的

  row1.copy_(col_indices_->view({-1}));
  // 将 `col_indices_` 的视图复制到 `row1`

  at::parallel_for(
                   0, nrows * total_nnz / nnz, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
        for (const auto i_  : c10::irange(start, end)) {
          auto b = i_ / nrows;
          auto i = i_ % nrows;
          std::fill(
              &data_out[b * nnz + crow_indices_data_in[b * (nrows + 1) + i]],
              &data_out[b * nnz + crow_indices_data_in[b * (nrows + 1) + i + 1]],
              static_cast<output_t>(i));
        }
      });

# 使用并行循环填充操作：在多线程环境下，为输出张量 `data_out` 的特定位置范围赋值，使用 `crow_indices_data_in` 来确定填充的位置。
}
} // namespace

// 定义 Torch 实现函数 _convert_indices_from_coo_to_csr_structured_cpu
TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cpu)
// 参数列表：输入张量 input，尺寸 size，是否输出 int32 结果 out_int32，结果张量 result
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  // 如果输出 int32 结果
  if (out_int32) {
    // 调度并执行整数类型的分发，调用 convert_indices_from_coo_to_csr_cpu 函数
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int32_t>(
              result, input, size);
        });
  } else {
    // 否则调度并执行整数类型的分发，调用 convert_indices_from_coo_to_csr_cpu 函数
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int64_t>(
              result, input, size);
        });
  }
}

// 定义 Torch 实现函数 _convert_indices_from_csr_to_coo_structured_cpu
TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_cpu)
// 参数列表：压缩行索引 crow_indices，列索引 col_indices，是否输出 int32 结果 out_int32，
// 是否转置 transpose，结果张量 result
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  // 如果输出 int32 结果
  if (out_int32) {
    // 调度并执行整数类型的分发，调用 convert_indices_from_csr_to_coo_cpu 函数
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int32_t>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    // 否则调度并执行整数类型的分发，调用 convert_indices_from_csr_to_coo_cpu 函数
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
}

/*
 * 基于 https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h
 * 修改以确保排序的 BSR 列索引。
 */
// 压缩到块压缩的 CPU 内核函数模板
template <class index_t, class scalar_t, bool compressed_rows>
void _compressed_to_block_compressed_cpu_kernel(
    const index_t n_compressed, // 压缩维度上的张量大小
    const index_t n_plain, // 平面维度上的张量大小
    const index_t C, // 压缩维度上的块大小
    const index_t P, // 平面维度上的块大小
    const index_t D, // 稠密维度中的元素数量
    const index_t* input_compressed_indices, // 输入的压缩索引数组
    const index_t* input_plain_indices, // 输入的平面索引数组
    const scalar_t* input_values, // 输入的值数组
    index_t* result_compressed_indices, // 结果的压缩索引数组
    index_t* result_plain_indices, // 结果的平面索引数组
    scalar_t* result_values) { // 结果的值数组
  // 所有块都是可能的，即如果单个非零值位于其中，则可以分配它们。否则它们不是。

  // 为所有可能的平面块和一个额外的指针分配空间
  std::vector<scalar_t*> blocks(n_plain / P + 1, nullptr);

  // 断言保证压缩维度和平面维度可以被块大小整除
  assert(n_compressed % C == 0);
  assert(n_plain % P == 0);

  // 压缩维度上的块数
  index_t n_bcompressed = n_compressed / C;
  // 平面维度上的块数
  index_t n_bplain = n_plain / P;

  // 每个块的元素数量
  index_t CPD = C * P * D;
  // 总共的块数
  index_t n_blks = 0;

  // 初始化结果的压缩索引数组的第一个元素为 0
  result_compressed_indices[0] = 0;

  // 遍历压缩维度上的块
  for (index_t block_c = 0; block_c < n_bcompressed; block_c++) {
    // 迭代处理压缩维度块中的非零块，确保按照排序后的普通维度索引进行操作
    for (index_t block_p = 0; block_p < n_bplain; block_p ++) {
      // 遍历当前压缩块内的压缩维度索引范围
      for (index_t i = input_compressed_indices[C * block_c]; i < input_compressed_indices[C * (block_c + 1)]; i++) {
        index_t p = input_plain_indices[i]; // 普通维度元素索引
        // 如果当前普通维度元素索引在当前块内，则进行处理
        if (p / P == block_p) {
          // 将结果值数组中的块指针指向当前处理的结果值块
          blocks[block_p] = result_values + CPD * n_blks;
          // 记录当前处理块的普通维度索引
          result_plain_indices[n_blks] = block_p;
          // 块计数增加
          n_blks++;
          // 跳出当前循环，继续处理下一个块
          break;
        }
      }
    }

    // 迭代处理块内的压缩维度
    for (index_t cb = 0; cb < C; cb++) {
      index_t c = C * block_c + cb; // 压缩维度索引
      // 遍历当前压缩维度块内的压缩索引范围
      for (index_t i = input_compressed_indices[c]; i < input_compressed_indices[c + 1]; i++) {
        index_t p = input_plain_indices[i]; // 普通维度索引

        // 计算普通维度索引对应的块索引
        index_t block_p = p / P;
        // 计算普通维度索引在块内的偏移
        index_t pb = p % P;

        // 复制当前压缩索引对应的数据到结果块中
        std::copy(input_values + i * D, input_values + (i + 1) * D,
                  blocks[block_p] + (compressed_rows ? P * cb + pb : C * pb + cb) * D);
      }
    }

    // Scipy代码包含以下部分，但修改后的代码（参见block_p循环以上的部分）不需要评估`blocks[block_p] == 0`，因此不需要这部分。
    // 用于设置结果压缩索引中的块计数
    result_compressed_indices[block_c + 1] = n_blks;
/*
 * 根据给定的压缩格式（CSR）转换为块压缩稀疏格式（BSR），并计算非零块的数量。
 * 返回转换后的稀疏张量。
 */
template<Layout target_layout>
Tensor sparse_compressed_to_sparse_bsr(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsr;
  // 检查输入和输出布局是否相同
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_compressed_to_sparse_bsr: unexpected same input and output layout");
  // 检查参数的有效性
  _to_sparse_check_arguments("sparse_compressed_to_sparse_bsr", self, layout_to, blocksize, dense_dim_opt);

  // 如果输入布局为 Bsc，则转换为 Bsr
  if (self.layout() == kSparseBsc) {
    return sparse_compressed_to_flipped(self, blocksize, "to_sparse_bsr");
  }
  // 如果输入布局为 Csr
  if (self.layout() == kSparseCsr) {
    // 如果不在 CPU 设备上，则发出警告
    if (self.device() != kCPU) {
      TORCH_WARN("sparse_compressed_to_sparse_bsr executing on the CPU device, the performance may be sub-optimal");
    }
    // 在 CPU 上执行压缩到块压缩的转换，并将结果移到原始设备
    return _compressed_to_block_compressed_cpu<kSparseBsr>(self.cpu(), blocksize).to(self.device());
  }
  // 如果输入布局为 Csc
  if (self.layout() == kSparseCsc) {
    // 先将输入转换为 Csr，再转换为 Bsr
    return self.to_sparse_csr(dense_dim_opt).to_sparse_bsr(blocksize);
  }

  // 如果输入布局不是 Csr、Csc、Bsr、Bsc 中的任何一种，则抛出错误
  AT_ERROR("sparse_compressed_to_sparse_bsr: expected SparseCsr, SparseCsc, SparseBsr or SparseBsc layout but got ", self.layout());
  return Tensor{};
}

/*
 * 根据给定的压缩格式（CSC）转换为块压缩稀疏格式（BSC），并计算非零块的数量。
 * 返回转换后的稀疏张量。
 */
Tensor sparse_compressed_to_sparse_bsc(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsc;
  // 检查输入和输出布局是否相同
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_compressed_to_sparse_bsc: unexpected same input and output layout");
  // 检查参数的有效性
  _to_sparse_check_arguments("sparse_compressed_to_sparse_bsc", self, layout_to, blocksize, dense_dim_opt);

  // 如果输入布局为 Bsr，则转换为 Bsc
  if (self.layout() == kSparseBsr) {
    return sparse_compressed_to_flipped(self, blocksize, "to_sparse_bsc");
  }
  // 如果输入布局为 Csc
  if (self.layout() == kSparseCsc) {
    // 如果不在 CPU 设备上，则发出警告
    if (self.device() != kCPU) {
      TORCH_WARN("sparse_compressed_to_sparse_bsc executing on the CPU device, the performance may be sub-optimal");
    }
    // 在 CPU 上执行压缩到块压缩的转换，并将结果移到原始设备
    return _compressed_to_block_compressed_cpu<kSparseBsc>(self.cpu(), blocksize).to(self.device());
  }
  // 如果输入布局为 Csr
  if (self.layout() == kSparseCsr) {
    // 先将输入转换为 Csc，再转换为 Bsc
    return self.to_sparse_csr(dense_dim_opt).to_sparse_bsc(blocksize);
  }

  // 如果输入布局不是 Csr、Csc、Bsr、Bsc 中的任何一种，则抛出错误
  AT_ERROR("sparse_compressed_to_sparse_bsc: expected SparseCsr, SparseCsc, SparseBsr or SparseBsc layout but got ", self.layout());
  return Tensor{};
}
    // 将稠密格式转换为稀疏的块压缩列存储（BSC），并返回结果
    return self.to_sparse_csc(dense_dim_opt).to_sparse_bsc(blocksize);
  }

  // 如果当前稀疏张量不是稀疏的CSR、CSC、BSR或BSC布局，则抛出错误并返回空张量
  AT_ERROR("sparse_compressed_to_sparse_bsc: expected SparseCsr, SparseCsc, SparseBsr or SparseBsc layout but got ", self.layout());
  // 返回空的Tensor对象作为错误处理的结果
  return Tensor{};
}

// 将稀疏 COO 格式的张量转换为稀疏格式的张量
Tensor sparse_coo_to_sparse(const Tensor& self, const int64_t sparse_dim) {
  // 设置目标布局为稀疏
  auto layout_to = kSparse;
  // 检查转换参数的有效性
  _to_sparse_check_arguments("sparse_coo_to_sparse", self, sparse_dim);

  // 抛出错误，指出不支持从当前布局到目标布局的转换
  AT_ERROR("sparse_coo_to_sparse: ", self.layout(), " to ", layout_to, " conversion not supported");
  // 返回空张量
  return Tensor{};
}

// 将压缩稀疏格式的张量转换为稀疏格式的张量
Tensor sparse_compressed_to_sparse(const Tensor& self, const int64_t sparse_dim) {
  // 检查转换参数的有效性
  _to_sparse_check_arguments("sparse_compressed_to_sparse", self, sparse_dim);

  // 获取当前张量的布局
  Layout layout = self.layout();
  // 获取压缩格式和普通索引
  auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(self);
  Tensor values;
  // 将压缩格式的索引转换为 COO 格式的索引
  Tensor indices = at::_convert_indices_from_csr_to_coo(compressed_indices, plain_indices,
                                                        false, (layout == kSparseCsc || layout == kSparseBsc));
  // 计算批处理维度数
  const auto batch_ndim = compressed_indices.dim() - 1;
  // 只有 CSR 格式可以简单地合并
  bool coalesced = layout == kSparseCsr || self.numel() == 0 || self._nnz() == 1;
  // 根据当前布局选择不同的处理路径
  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "sparse_compressed_to_sparse",
    [&] { values = self.values().flatten(0, batch_ndim); },
    [&] {
      // 计算块的大小
      auto blocksize = DimVector(self.values().sizes().slice(batch_ndim + 1, 2));
      DimVector batch_blocksize;
      batch_blocksize.append(batch_ndim, 1);
      batch_blocksize.append(blocksize);
      // 创建块 COO 索引
      const auto block_coo_indices = at::zeros({batch_ndim + 2, blocksize[0] * blocksize[1]}, indices.options());
      block_coo_indices.narrow(0, batch_ndim, 2).copy_(at::sparse::full_coo_indices(blocksize, indices.options()));
      // 对索引进行变换，以便获得块的元素坐标
      indices = indices
        .mul(at::tensor(batch_blocksize, indices.options()).unsqueeze_(1))
        .unsqueeze_(-1).add(block_coo_indices.unsqueeze_(1))
        .flatten(-2, -1);

      // 展平值张量
      values = self.values().flatten(0, batch_ndim + 2);

      // 对于 BSR 格式且块大小为 1 且批处理维度为 0 的情况，产生合并的结果
      coalesced |= (layout == kSparseBsr && blocksize[0] == 1 && batch_ndim == 0);
    });
  // 创建稀疏 COO 张量并标记是否合并
  return at::native::_sparse_coo_tensor_unsafe(indices, values, self.sizes())._coalesced_(coalesced);
}
// 将稀疏压缩表示的张量转换为稀疏表示的张量
Tensor sparse_compressed_to_sparse(const Tensor& self, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确定转换后的布局，默认为稀疏布局
  auto layout_to = layout.value_or(kSparse);
  // 断言输入张量的布局与目标布局不同，否则抛出异常
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_compressed_to_sparse: unexpected same input and output layout");
  // 检查参数有效性
  _to_sparse_check_arguments("sparse_compressed_to_sparse", self, layout_to, blocksize, dense_dim_opt);

  // 确定块大小，默认为自适应或者1x1的大小，根据输入张量的布局确定
  auto blocksize_ = blocksize.value_or((self.layout() == kSparseBsr || self.layout() == kSparseBsc) ? at::sparse_csr::getBlockSize(self) : at::DimVector({1, 1}));
  
  // 根据目标布局执行相应的转换操作
  switch (layout_to) {
  case kStrided:
    // 转换为密集张量
    return sparse_compressed_to_dense(self, /*dtype=*/c10::nullopt, /*masked_grad=*/c10::nullopt);
  case kSparse:
    // 转换为稀疏张量，密度为2
    return sparse_compressed_to_sparse(self, 2);
  case kSparseCsr:
    // 转换为CSR格式的稀疏张量
    return sparse_compressed_to_sparse_csr(self, dense_dim_opt);
  case kSparseCsc:
    // 转换为CSC格式的稀疏张量
    return sparse_compressed_to_sparse_csc(self, dense_dim_opt);
  case kSparseBsr:
    // 转换为BSR格式的稀疏张量，指定块大小和稠密维度
    return sparse_compressed_to_sparse_bsr(self, blocksize_, dense_dim_opt);
  case kSparseBsc:
    // 转换为BSC格式的稀疏张量，指定块大小和稠密维度
    return sparse_compressed_to_sparse_bsc(self, blocksize_, dense_dim_opt);
  default:
    break;
  }

  // 如果遇到未支持的布局转换，抛出错误
  AT_ERROR("sparse_compressed_to_sparse: ", self.layout(), " to ", layout_to, " conversion not supported");
  // 返回空张量
  return Tensor{};
}

// 将COO格式的稀疏张量转换为目标布局的稀疏张量
Tensor sparse_coo_to_sparse(const Tensor& self, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确定转换后的布局，默认为稀疏布局
  auto layout_to = layout.value_or(kSparse);
  // 断言输入张量的布局与目标布局不同，否则抛出异常
  TORCH_INTERNAL_ASSERT(self.layout() != layout_to, "sparse_coo_to_sparse: unexpected same input and output layout");
  // 检查参数有效性
  _to_sparse_check_arguments("sparse_coo_to_sparse", self, layout_to, blocksize, dense_dim_opt);

  // 根据目标布局执行相应的转换操作
  switch (layout_to) {
  case kStrided:
    // 转换为密集张量
    return self.to_dense(c10::nullopt, c10::nullopt);
  case kSparseCsr:
    // 转换为CSR格式的稀疏张量，指定稠密维度
    return self.to_sparse_csr(dense_dim_opt);
  case kSparseCsc:
    // 转换为CSC格式的稀疏张量，指定稠密维度
    return self.to_sparse_csc(dense_dim_opt);
  case kSparseBsr:
    // 转换为BSR格式的稀疏张量，指定块大小和稠密维度
    return self.to_sparse_bsr(*blocksize, dense_dim_opt);
  case kSparseBsc:
    // 转换为BSC格式的稀疏张量，指定块大小和稠密维度
    return self.to_sparse_bsc(*blocksize, dense_dim_opt);
  default:
    break;
  }

  // 如果遇到未支持的布局转换，抛出错误
  AT_ERROR("sparse_coo_to_sparse: ", self.layout(), " to ", layout_to, " conversion not supported");
  // 返回空张量
  return Tensor{};
}

// 将输入张量转换为指定稀疏度的稀疏张量
Tensor to_sparse(const Tensor& self, const int64_t sparse_dim) {
  // 默认转换为稀疏布局
  auto layout_to = kSparse;
  // 如果输入张量已经是稀疏布局，则直接返回
  if (self.layout() == layout_to) {
    // 检查参数有效性
    _to_sparse_check_arguments("to_sparse", self, sparse_dim);
    // 返回原始稀疏张量
    return self;
  }
  // 执行稀疏转换操作
  return self._to_sparse(sparse_dim);
}

// 将输入张量转换为指定布局的稀疏张量
Tensor to_sparse(const Tensor& self, std::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  // 确定转换后的布局，默认为稀疏布局
  auto layout_to = layout.value_or(kSparse);
  // 如果输入张量已经是目标布局，则直接返回
  if (self.layout() == layout_to) {
    // 检查参数有效性
    _to_sparse_check_arguments("to_sparse", self, layout, blocksize, dense_dim_opt);
    // 返回原始稀疏张量
    return self;
  }
  // 执行稀疏转换操作
  return self._to_sparse(layout, blocksize, dense_dim_opt);
}
// 将稀疏张量转换为压缩稀疏行 (CSR) 格式
Tensor to_sparse_csr(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsr;
  // 如果张量已经是 CSR 布局，则直接返回自身
  if (self.layout() == layout_to) {
    // 检查参数并确保转换的一致性
    _to_sparse_check_arguments("to_sparse_csr", self, layout_to, {}, dense_dim_opt);
    return self;
  }
  // 否则调用内部方法进行 CSR 转换
  return self._to_sparse_csr(dense_dim_opt);
}

// 将稀疏张量转换为压缩稀疏列 (CSC) 格式
Tensor to_sparse_csc(const Tensor& self, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseCsc;
  // 如果张量已经是 CSC 布局，则直接返回自身
  if (self.layout() == layout_to) {
    // 检查参数并确保转换的一致性
    _to_sparse_check_arguments("to_sparse_csc", self, layout_to, {}, dense_dim_opt);
    return self;
  }
  // 否则调用内部方法进行 CSC 转换
  return self._to_sparse_csc(dense_dim_opt);
}

// 将稀疏张量转换为块稀疏行 (BSR) 格式
Tensor to_sparse_bsr(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsr;
  // 如果张量已经是 BSR 布局，则直接返回自身
  if (self.layout() == layout_to) {
    // 检查参数并确保转换的一致性，包括块大小信息
    _to_sparse_check_arguments("to_sparse_bsr", self, layout_to, blocksize, dense_dim_opt);
    return self;
  }
  // 否则调用内部方法进行 BSR 转换
  return self._to_sparse_bsr(blocksize, dense_dim_opt);
}

// 将稀疏张量转换为块稀疏列 (BSC) 格式
Tensor to_sparse_bsc(const Tensor& self, IntArrayRef blocksize, std::optional<int64_t> dense_dim_opt) {
  auto layout_to = kSparseBsc;
  // 如果张量已经是 BSC 布局，则直接返回自身
  if (self.layout() == layout_to) {
    // 检查参数并确保转换的一致性，包括块大小信息
    _to_sparse_check_arguments("to_sparse_bsc", self, layout_to, blocksize, dense_dim_opt);
    return self;
  }
  // 否则调用内部方法进行 BSC 转换
  return self._to_sparse_bsc(blocksize, dense_dim_opt);
}

// 将张量转换为元数据张量
Tensor to_meta(const Tensor& tensor) {
  // 使用张量的符号大小、符号步长、数据类型、布局和设备信息创建一个元数据张量
  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), \
/*dtype=*/c10::make_optional(tensor.scalar_type()), /*layout=*/c10::make_optional(tensor.layout()), \
/*device=*/c10::make_optional(c10::Device(c10::kMeta)), /*pin_memory=*/c10::nullopt);
  // 如果张量是包装数值类型，需要设置元数据张量也是包装数值类型
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}

// 将可选的张量转换为元数据张量
std::optional<Tensor> to_meta(const std::optional<Tensor>& tensor) {
  // 如果输入的可选张量有值，则调用单参数版本的 to_meta 函数进行转换
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  // 否则返回空值
  return c10::nullopt;
}

// 将张量列表转换为元数据张量列表
std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outs;
  outs.reserve(t_list.size());
  // 遍历输入的张量列表，将每个张量转换为元数据张量并存储到输出列表中
  for (const auto& tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
```