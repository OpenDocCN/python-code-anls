# `.\pytorch\aten\src\ATen\native\sparse\SparseCsrTensor.cpp`

```
// 基本稀疏张量函数

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>  // 引入张量核心头文件
#include <ATen/Dispatch.h>  // 分发机制头文件
#include <ATen/InitialTensorOptions.h>  // 初始张量选项头文件
#include <ATen/Layout.h>  // 张量布局头文件
#include <ATen/Parallel.h>  // 并行处理头文件
#include <ATen/SparseCsrTensorImpl.h>  // 稀疏 CSR 张量实现头文件
#include <ATen/SparseCsrTensorUtils.h>  // 稀疏 CSR 张量工具函数头文件
#include <ATen/SparseTensorImpl.h>  // 稀疏张量实现头文件
#include <ATen/native/LinearAlgebraUtils.h>  // 线性代数工具函数头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>  // 引入 ATen 函数头文件
#include <ATen/NativeFunctions.h>  // 引入 ATen 本地函数头文件
#else
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csc_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsc_tensor_args_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_copy.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/sparse_compressed_tensor_native.h>
#include <ATen/ops/sparse_csr_tensor_native.h>
#include <ATen/ops/sparse_csc_tensor_native.h>
#include <ATen/ops/sparse_bsr_tensor_native.h>
#include <ATen/ops/sparse_bsc_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

using namespace at::sparse_csr;

namespace {

bool solve_arange(const Tensor& input, int64_t& start, int64_t& end, int64_t& step) {
  /*
    This function solves the equation

      input == arange(start, end, step)

    for integers start, end, and step, if possible. If the solution
    exists, returns true.
  */
  int64_t n = input.numel();  // 计算张量中元素的总数
  if (n == 0) {
    // 简单情况：张量为空
    start = end = 0;
    step = 1;
  } else if (n == 1) {
    // 简单情况：张量只有一个元素
    start = input[0].item<int64_t>();  // 获取张量中的第一个元素作为起始值
    end = start + 1;  // 结束值为起始值加一
    step = 1;  // 步长为1
  } else {
    Tensor first_last = input.slice(0, 0, n, n - 1).cpu();  // 获取张量的第一个和最后一个元素，并移到 CPU 上处理


这段代码涵盖了头文件的引入和一个解方程的函数实现，需要进一步的注释以完整解释其余部分的功能和作用。
    // 将第一个和最后一个元素转换为 int64_t 类型，并分别赋给起始和结束候选值
    int64_t start_candidate = first_last[0].item<int64_t>();
    int64_t end_candidate = first_last[1].item<int64_t>() + 1;
    
    // 如果结束候选值减去起始候选值等于 n，表示找到特殊解
    if (end_candidate - start_candidate == n) {
      // 特殊解情况下的处理
      start = start_candidate;
      end = end_candidate;
      step = 1;
    } else {
      // 检测是否存在一般解
      // 计算可能的步长，即输入的第一个到第 n-1 个元素的差值
      Tensor possible_steps = input.slice(0, 1).sub(input.slice(0, 0, n - 1));
      // 取第一个可能的步长值
      Tensor possible_step = possible_steps[0];
      // 如果所有的可能步长值都相等，则存在一般解
      if ((possible_steps.eq(possible_step)).all().item<bool>()) {
        start = start_candidate;
        end = end_candidate;
        step = possible_step.item<int64_t>();
      } else {
        // 没有解决方案
        return false;
      }
    }
  }
  // 函数执行成功，返回 true
  return true;
} // end anonymous namespace



/*
  Validate the arguments to sparse compressed (CSR, CSC, BSR, and BSC)
  tensor factory functions.

  The CSR and BSR invariants for PyTorch are outlined in

    https://pearu.github.io/csr_tensor_invariants.html
    https://pearu.github.io/bsr_tensor_invariants.html

  that in what follows are generalized for all sparse compressed
  formats with support to batched and dense dimensions.
*/



TORCH_CHECK(values_nnz == 0, "expected nnz to be 0 for sparse ", layout_name, " meta tensor but got ", values_nnz);

检查稀疏张量的非零值数是否为0，用于验证稀疏格式的元数据张量。


} else {

如果条件不满足，则执行以下代码块。


// Indices invariants
at::_validate_compressed_sparse_indices(
    /*is_crow = */layout == kSparseCsr || layout == kSparseBsr,
    compressed_indices,
    plain_indices,
    compressed_dim_size,
    plain_dim_size,
    values_nnz);

验证稀疏压缩索引的不变性，根据布局类型（CSR 或 BSR）调用相应的验证函数。


// Device Invariants
// 4.1
TORCH_CHECK(
    values.device().type() == kCPU || values.device().type() == kCUDA || values.device().type() == kMeta,
    "device type of values (",
    values.device().type(),
    ") must be CPU or CUDA or Meta");

验证设备的不变性：确保值张量的设备类型是 CPU、CUDA 或 Meta。


// 4.2, 4.3, 4.4
TORCH_CHECK(
    compressed_indices.get_device() == values.get_device(),
    "device of ", compressed_indices_name, " (=",
    compressed_indices.device(),
    ") must match device of values (=",
    values.device(),
    ")");
TORCH_CHECK(
    compressed_indices.get_device() == plain_indices.get_device(),
    "device of ", compressed_indices_name, " (=",
    compressed_indices.device(),
    ") must match device of ", plain_indices_name," (=",
    plain_indices.device(),
    ")");

验证设备的不变性：确保压缩索引张量和值张量、压缩索引张量和普通索引张量在同一设备上。


// Autograd Invariants
//
// These are internal asserts because users should not be able to
// create non-floating point dtype tensors with requires_grad flag
// set to true.
TORCH_INTERNAL_ASSERT(!compressed_indices.requires_grad());
TORCH_INTERNAL_ASSERT(!plain_indices.requires_grad());

验证自动求导的不变性：确保压缩索引张量和普通索引张量不需要梯度，以防止用户使用非浮点数类型的张量并将 requires_grad 标志设置为 true。


}

void _validate_sparse_compressed_tensor_args(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, IntArrayRef size, Layout layout) {
  _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout);
}



void _validate_sparse_csr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseCsr);
}



void _validate_sparse_csc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseCsc);
}



void _validate_sparse_bsr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseBsr);
}
``` 

这些函数定义用于验证稀疏张量（CSR、CSC、BSR）的参数，并调用相应的工作函数进行详细的验证。
// 验证稀疏压缩张量参数的有效性
void _validate_sparse_bsc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size) {
    // 调用工作函数，验证稀疏压缩张量的参数，并指定为稀疏BSC格式
    _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseBsc);
}

// 构建CSR、CSC、BSR和BSC张量。

// 注意：在名称中使用"Csr"（如SparseCsrTensor、SparseCsrCPU、SparseCsrCUDA和SparseCsrTensorImpl）是出于历史原因
// （未来应该删除），并不意味着相应的功能只适用于CSR布局。
static SparseCsrTensor new_compressed_tensor(const TensorOptions& options) {
    // TODO: 在启用CSR张量的自动求导支持后，删除此注释。
    // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
    
    // 从选项中获取布局，根据稀疏压缩布局调度CSR、CSC、BSR和BSC的构建
    Layout layout = AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(options.layout(), "new_compressed_tensor", [&] { return the_layout; });
    DispatchKey dispatch_key = DispatchKey::Undefined;

    // 根据设备类型选择适当的调度键
    switch(options.device().type()) {
    case kCPU:
        dispatch_key = DispatchKey::SparseCsrCPU;
        break;
    case kCUDA:
        dispatch_key = DispatchKey::SparseCsrCUDA;
        break;
    case kMeta:
        dispatch_key = DispatchKey::SparseCsrMeta;
        break;
    case kPrivateUse1:
        dispatch_key = DispatchKey::SparseCsrPrivateUse1;
        break;
    default:
        // 如果设备类型不支持，则抛出错误
        TORCH_CHECK_NOT_IMPLEMENTED(false, "Could not run 'new_compressed_tensor' from the '", options.device(), "' device.)");
    }

    // 使用指定的调度键、设备、布局和数据类型创建稀疏CSR张量的实现
    return detail::make_tensor<SparseCsrTensorImpl>(DispatchKeySet(dispatch_key), options.device(), layout, options.dtype());
}
// sparse_compressed_tensor_with_dims 函数用于创建指定稀疏压缩张量的通用化版本，允许设置 nnz、dense_dim、blocksize 和 index_dtype。
Tensor sparse_compressed_tensor_with_dims(
     int64_t nnz,                                      // 非零元素的数量
     int64_t dense_dim,                                 // 密集维度的数量
     c10::IntArrayRef size,                             // 张量的维度尺寸
     c10::IntArrayRef blocksize,                        // 块大小
     ScalarType index_dtype,                            // 索引的数据类型
     std::optional<ScalarType> dtype,                   // 张量数据类型（可选）
     std::optional<Layout> layout,                      // 张量布局（可选）
     std::optional<Device> device,                      // 设备（可选）
     std::optional<bool> pin_memory) {                  // 是否锁定内存（可选）
  // sparse_compressed_tensor_with_dims 是 empty 函数的泛化版本，
  // 允许指定 nnz、dense_dim、blocksize 和 index_dtype 以创建稀疏压缩张量。
  //
  // sparse_compressed_tensor_with_dims 的 indices 和 values 张量被创建为空张量，
  // 因此返回的稀疏压缩张量不会满足稀疏压缩张量的不变性条件。
  // 调用者负责适当初始化 indices 张量。

  TORCH_CHECK(layout, "sparse_compressed_tensor_with_dims: expected sparse compressed tensor layout but got none");
  // 检查是否提供了稀疏压缩张量的布局，否则抛出错误信息。

  Layout layout_ = layout.value();
  // 获取布局的值。

  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor_with_dims", [&]{});
  // 根据布局调度特定的稀疏压缩张量计算，目前不包含实际操作。

  constexpr int64_t sparse_dim = 2;
  // 声明稀疏维度的常量为 2。
  int64_t batch_dim = size.size() - dense_dim - sparse_dim;
  // 计算批处理维度的数量。
  TORCH_CHECK(batch_dim >= 0, "sparse_compressed_tensor_with_dims: dimensionality must be at least dense_dim(=", dense_dim, ") + sparse_dim(=", sparse_dim, "), but got ", size.size());
  // 检查维度的合法性，确保至少包括 dense_dim 和 sparse_dim。

  TORCH_CHECK(nnz >= 0, "sparse_compressed_tensor_with_dims: nnz must be non-negative, got ", nnz);
  // 检查 nnz 是否为非负数。

  auto plain_indices_size = DimVector(size.slice(0, batch_dim));
  auto compressed_indices_size = DimVector(size.slice(0, batch_dim));
  auto values_size = DimVector(size.slice(0, batch_dim));
  // 初始化各种索引和值的尺寸向量。

  plain_indices_size.push_back(nnz);
  values_size.push_back(nnz);
  // 更新普通索引和值的尺寸，增加 nnz。

  if (layout_ == kSparseBsr || layout_ == kSparseBsc) {
    TORCH_CHECK(blocksize.size() == (size_t)sparse_dim, "sparse_compressed_tensor_with_dims: blocksize needs to be a tuple of size ",
                sparse_dim, ", but got ", blocksize.size());
    // 检查块大小是否符合要求，必须与 sparse_dim 大小相同。

    auto d0 = (layout_ == kSparseBsr ? 0 : 1);
    auto d1 = (layout_ == kSparseBsr ? 1 : 0);
    // 根据布局设置 d0 和 d1 的索引。

    TORCH_CHECK(blocksize[0] > 0 && blocksize[1] > 0, "sparse_compressed_tensor_with_dims: blocksize needs to be positive, but got ", blocksize);
    // 检查块大小是否为正数。

    auto compressed_size = size[compressedDimension(layout_, size, dense_dim)];
    auto plain_size = size[plainDimension(layout_, size, dense_dim)];
    // 获取压缩和普通维度的尺寸。

    TORCH_CHECK(compressed_size % blocksize[d0] == 0, "sparse_compressed_tensor_with_dims: dimension ",
                compressedDimension(layout_, size, dense_dim), " must be multiple of blocksize[", d0, "](=", blocksize[d0], ") but got ", compressed_size);
    // 检查压缩维度是否是块大小 d0 的倍数。

    TORCH_CHECK(plain_size % blocksize[d1] == 0, "sparse_compressed_tensor_with_dims: dimension ", plainDimension(layout_, size, dense_dim),
                " must be multiple of blocksize[", d1, "](=", blocksize[d1], ") but got ", plain_size);
    // 检查普通维度是否是块大小 d1 的倍数。

    compressed_indices_size.push_back(compressed_size / blocksize[d0] + 1);
    // 更新压缩索引的尺寸。

    values_size.append(DimVector(blocksize));
    // 更新值的尺寸，添加块大小信息。
  } else {
    # 检查块大小的尺寸是否为零，如果不是零则抛出异常，指示非块布局下不能指定块大小
    TORCH_CHECK(blocksize.size() == 0, "sparse_compressed_tensor_with_dims: blocksize cannot be specified for non-block layout ", layout_);
    # 将压缩后的指标尺寸添加到列表末尾，包括在布局中压缩维度的尺寸加一
    compressed_indices_size.push_back(size[compressedDimension(layout_, size, dense_dim)] + 1);
  }

  # 将值的尺寸添加到值的尺寸列表中，这些尺寸从批次维度和稀疏维度到稠密维度的切片中提取
  values_size.append(DimVector(size.slice(batch_dim + sparse_dim, dense_dim)));
  # 检查索引的数据类型是否为 Int 或 Long，否则抛出异常
  TORCH_CHECK(
      index_dtype == ScalarType::Int || index_dtype == ScalarType::Long,
      "indices dtype must be Int or Long, but got ", index_dtype);

  # 定义张量选项，包括布局为 Strided，设备为指定设备，内存为固定内存
  TensorOptions options_ = TensorOptions().layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  # 创建一个空张量用于存储压缩后的索引，尺寸为 compressed_indices_size，数据类型为 index_dtype
  auto compressed_indices = at::empty(compressed_indices_size, options_.dtype(index_dtype));
  # 创建一个空张量用于存储非压缩索引，尺寸为 plain_indices_size，数据类型为 index_dtype
  auto plain_indices = at::empty(plain_indices_size, options_.dtype(index_dtype));
  # 创建一个空张量用于存储值，尺寸为 values_size，数据类型为 dtype
  auto values = at::empty(values_size, options_.dtype(dtype));

  # 定义张量选项，包括数据类型为 dtype，布局为 layout_，设备为指定设备，内存为固定内存
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  # 创建一个新的稀疏 CSR 张量 self
  SparseCsrTensor self = new_compressed_tensor(options);
  # 设置 self 的成员张量为压缩后的索引、非压缩索引、值以及尺寸信息
  get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  # 返回创建的稀疏 CSR 张量 self
  return self;
}

# 定义一个函数 _sparse_compressed_tensor_unsafe_symint，接受多个参数并返回一个 Tensor
Tensor _sparse_compressed_tensor_unsafe_symint(
     const Tensor& compressed_indices,        # 压缩索引张量，存储压缩的稀疏张量索引
     const Tensor& plain_indices,             # 普通索引张量，存储非压缩的稀疏张量索引
     const Tensor& values,                    # 值张量，存储稀疏张量的值
     c10::SymIntArrayRef size,                # 稀疏张量的尺寸信息
     std::optional<ScalarType> dtype,         # 可选参数，张量的数据类型
     std::optional<Layout> layout,            # 可选参数，稀疏张量的布局
     std::optional<Device> device,            # 可选参数，张量的设备类型
     std::optional<bool> pin_memory) {        # 可选参数，是否使用固定内存

  # 如果布局参数为空，抛出错误信息
  if (!layout) {
    AT_ERROR("sparse_compressed_tensor_unsafe expected sparse compressed tensor layout but got none");
  }
  
  # 获取布局的实际值
  Layout layout_ = layout.value();

  # 根据布局类型分发处理函数
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor_unsafe", [&]{});

  # 如果全局上下文需要检查稀疏张量的不变性，则执行参数验证函数
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, C10_AS_INTARRAYREF_SLOW(size), layout_);
  }

  # 设置张量的选项，包括数据类型、布局、设备类型和是否固定内存
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  # 创建一个新的稀疏 CSR 张量
  SparseCsrTensor self = new_compressed_tensor(options);

  # 设置稀疏 CSR 张量的成员张量，包括压缩索引、普通索引、值和尺寸信息
  get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);

  # 返回创建的稀疏 CSR 张量
  return self;
}

# 模板函数 _sparse_compressed_tensor_unsafe_template，接受多个参数并返回一个 Tensor
template <Layout required_layout>
Tensor _sparse_compressed_tensor_unsafe_template(const Tensor& compressed_indices,
                                                 const Tensor& plain_indices,
                                                 const Tensor& values,
                                                 IntArrayRef size,
                                                 std::optional<ScalarType> dtype,
                                                 std::optional<Layout> layout,
                                                 std::optional<Device> device,
                                                 std::optional<bool> pin_memory) {

  # 获取布局的实际值或者使用模板参数中指定的布局
  Layout layout_ = layout.value_or(required_layout);

  # 检查布局是否符合模板参数中要求的布局类型，如果不符则抛出错误信息
  TORCH_CHECK(layout_ == required_layout, "sparse compressed layout must be ",required_layout, " but got ", layout_);

  # 如果全局上下文需要检查稀疏张量的不变性，则执行参数验证函数
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout_);
  }

  # 设置张量的选项，包括数据类型、布局、设备类型和是否固定内存
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  # 创建一个新的稀疏 CSR 张量
  SparseCsrTensor self = new_compressed_tensor(options);

  # 设置稀疏 CSR 张量的成员张量，包括压缩索引、普通索引、值和尺寸信息
  get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);

  # 返回创建的稀疏 CSR 张量
  return self;
}
# 定义一个宏，用于生成不同类型的稀疏压缩张量（不安全版本）
#define SPARSE_COMPRESSED_TENSOR_UNSAFE(KIND, REQUIRED_LAYOUT)          \
  Tensor _sparse_##KIND##_tensor_unsafe(const Tensor& compressed_indices, \
                                        const Tensor& plain_indices,    \
                                        const Tensor& values,           \
                                        IntArrayRef size,               \
                                        std::optional<ScalarType> dtype, \
                                        std::optional<Layout> layout,   \
                                        std::optional<Device> device,   \
                                        std::optional<bool> pin_memory) { \
    # 调用模板函数 _sparse_compressed_tensor_unsafe_template 生成指定类型的稀疏压缩张量
    return _sparse_compressed_tensor_unsafe_template<REQUIRED_LAYOUT>(compressed_indices, plain_indices, values, size, dtype, layout, device, pin_memory); \
  }

# 使用宏 SPARSE_COMPRESSED_TENSOR_UNSAFE 生成不同类型的稀疏压缩张量函数
SPARSE_COMPRESSED_TENSOR_UNSAFE(csr, kSparseCsr);
SPARSE_COMPRESSED_TENSOR_UNSAFE(csc, kSparseCsc);
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsr, kSparseBsr);
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsc, kSparseBsc);

# 定义一个静态函数，用于估算稀疏压缩张量的大小
static DimVector _estimate_sparse_compressed_tensor_size(
    const Tensor& compressed_indices,   # 稀疏张量的压缩索引
    const Tensor& plain_indices,        # 稀疏张量的普通索引
    const Tensor& values,               # 稀疏张量的值
    // 根据布局类型调度处理，计算块的维度数
    const int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size", [&] { return 0; }, [&] { return 2; });
    // 基础维度为2，对应压缩和普通索引
    const int base_ndim = 2;
    // 计算批处理维度，减去1是因为索引从0开始
    const auto batch_ndim = compressed_indices.dim() - 1;
    // 获取压缩索引的名称
    const std::string compressed_indices_name = compressedIndicesName(layout);
    // 获取普通索引的名称
    const std::string plain_indices_name = plainIndicesName(layout);
    // 检查压缩索引的维度是否至少为1
    TORCH_CHECK(
                batch_ndim >= 0,
                compressed_indices_name, " must have dimensionality >= 1 but got ", compressed_indices.dim());
    // 检查压缩索引和普通索引的维度是否相等
    TORCH_CHECK(
                compressed_indices.dim() == plain_indices.dim(),
                compressed_indices_name, " and ", plain_indices_name, " dimensionalities must be equal but got ",
                compressed_indices.dim(), " and ", plain_indices.dim(), ", respectively");
    // 计算稠密数据的维度，减去批处理和块的维度，再减1是因为索引从0开始
    const int64_t dense_ndim = values.dim() - batch_ndim - block_ndim - 1;
    // 检查稠密数据的维度是否大于等于0
    TORCH_CHECK(
                dense_ndim >= 0,
                "values must have dimensionality > sum of batch and block dimensionalities (=",
                batch_ndim, " + ", block_ndim, ") but got ", values.dim());
    // 根据块的维度确定块的大小
    DimVector blocksize{
                        (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 1)) : 1),
                        (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 2)) : 1)
    };
    // 初始化大小向量，从压缩索引的尺寸中获取前几个尺寸作为大小向量的维度
    DimVector size = DimVector(compressed_indices.sizes().slice(0, batch_ndim));
    // 计算压缩维度的尺寸，确保维度大于0且尺寸大于0时，计算尺寸值
    int64_t compressed_dim_size = (compressed_indices.dim() > 0 && compressed_indices.size(-1) > 0 ? compressed_indices.size(-1) - 1 : 0);
    // 根据普通索引的数据类型计算普通维度的尺寸
    int64_t plain_dim_size = AT_DISPATCH_INTEGRAL_TYPES(plain_indices.scalar_type(), "estimate_sparse_compressed_tensor_size",
                                                        [&]() -> int64_t {
                                                          if (plain_indices.numel() > 0) {
                                                            return plain_indices.max().item<scalar_t>() + 1;
                                                          } else {
                                                            return 0;
                                                          }
                                                        });
    // 根据布局类型调度处理，更新大小向量的内容
    AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size",
        [&]{
          // 对于行压缩布局，更新大小向量的内容
          size.push_back(compressed_dim_size * blocksize[0]);
          size.push_back(plain_dim_size * blocksize[1]);
        },
        [&]{
          // 对于列压缩布局，更新大小向量的内容
          size.push_back(plain_dim_size * blocksize[0]);
          size.push_back(compressed_dim_size * blocksize[1]);
        });
    // 遍历稠密维度，计算其大小
    for (int i=0; i<dense_ndim; i++) {
      // 计算稠密维度的索引位置
      int64_t j = batch_ndim + 1 + block_ndim + i;
    // 将当前维度的尺寸值添加到 `size` 向量中，如果维度 `j` 小于 `values` 的维度数，则使用 `values.size(j)`，否则使用 1
    size.push_back((j < values.dim() ? values.size(j) : 1));
  }
  // 使用 Torch 的断言检查，确保 `size` 向量的大小等于批量维度、基础维度和密集维度的总和
  TORCH_CHECK(
              static_cast<int>(size.size()) == batch_ndim + base_ndim + dense_ndim,
              "tensor dimensionality must be sum of batch, base, and dense dimensionalities (=",
              batch_ndim, " + ", base_ndim, " + ", dense_ndim, ") but got ", size.size());
  // 返回 `size` 向量作为函数结果
  return size;
// 定义稀疏压缩张量生成函数的宏，根据 KIND 和 REQUIRED_LAYOUT 创建不同的稀疏张量类型
#define SPARSE_COMPRESSED_TENSOR(KIND, REQUIRED_LAYOUT)                 \
  // 定义函数 sparse_##KIND##_tensor，接受压缩索引、普通索引、值、dtype、布局、设备和固定内存等可选参数
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                std::optional<ScalarType> dtype,        \
                                std::optional<Layout> layout,           \
                                std::optional<Device> device,           \
                                std::optional<bool> pin_memory) {       \
    // 如果布局参数 layout 存在
    if (layout) {                                                       \
      // 检查布局是否等于 REQUIRED_LAYOUT，否则抛出错误信息
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    // 设置局部变量 layout_ 为 REQUIRED_LAYOUT
    std::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    // 调用原生函数创建稀疏压缩张量，传递压缩索引、普通索引、数值、数据类型、布局、设备、是否内存钉住
    return at::native::sparse_compressed_tensor(compressed_indices,       \
                                plain_indices,            \
                                values,                   \
                                IntArrayRef size,                       \
                                std::optional<ScalarType> dtype,        \
                                std::optional<Layout> layout,           \
                                std::optional<Device> device,           \
                                std::optional<bool> pin_memory);       \
  }                                                                     \
  // 定义 sparse_##KIND##_tensor 函数，接收压缩索引、普通索引、数值、大小、数据类型、布局、设备、是否内存钉住作为参数
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                IntArrayRef size,                       \
                                std::optional<ScalarType> dtype,        \
                                std::optional<Layout> layout,           \
                                std::optional<Device> device,           \
                                std::optional<bool> pin_memory) {       \
    // 如果布局参数 layout 存在
    if (layout) {                                                       \
      // 检查布局是否等于 REQUIRED_LAYOUT，否则抛出错误信息
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    // 设置局部变量 layout_ 为 REQUIRED_LAYOUT
    std::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    // 调用原生函数创建稀疏压缩张量，传递压缩索引、普通索引、数值、大小、数据类型、布局、设备、是否内存钉住
    return at::native::sparse_compressed_tensor(compressed_indices, plain_indices, values, size, dtype, layout_, device, pin_memory); \
  }


注释：
这段代码定义了两个函数 `sparse_##KIND##_tensor`，用于创建稀疏张量。首先检查布局是否符合要求，然后调用原生函数创建对应的稀疏压缩张量。
SPARSE`
// 定义宏以创建稀疏压缩张量，使用给定的格式参数
SPARSE_COMPRESSED_TENSOR(csr, kSparseCsr)
SPARSE_COMPRESSED_TENSOR(csc, kSparseCsc)
SPARSE_COMPRESSED_TENSOR(bsr, kSparseBsr)
SPARSE_COMPRESSED_TENSOR(bsc, kSparseBsc)

// 警告：理想情况下，torch.empty(...) 应不支持稀疏压缩格式，因为它未初始化压缩索引，不能返回有效的稀疏压缩张量。
// 下面的实现保留了对向后兼容性的支持。
Tensor empty_sparse_compressed(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  
  // 检查尺寸非负
  check_size_nonnegative(size);
  // 检查尺寸是否至少为2，仅支持批处理的非块状稀疏压缩张量
  TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);

  // 默认布局为 Strided
  Layout layout_ = layout.value_or(Layout::Strided);

  // torch.empty 不能用于创建块状张量，因为其 API 缺少指定块大小的方法
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(layout_, "empty_sparse_compressed", [&]{});

  // 初始化非零元素数量为0，设置压缩索引和普通索引/值的尺寸
  int64_t nnz = 0;
  auto compressed_indices_size = DimVector(size.slice(0, size.size() - 2));
  auto plain_indices_and_values_size = DimVector(size.slice(0, size.size() - 2));
  compressed_indices_size.push_back(size[compressedDimension(layout_, size)] + 1);
  plain_indices_and_values_size.push_back(nnz);

  // 设置张量选项
  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  // 创建空的压缩索引、普通索引和值张量
  auto compressed_indices = at::empty(compressed_indices_size, options);
  auto plain_indices = at::empty(plain_indices_and_values_size, options);
  auto values = at::empty(plain_indices_and_values_size, options.dtype(dtype));

  // torch.empty 生成垃圾数据，因此结果的空稀疏压缩张量可能无法满足以下压缩稀疏张量的不变性：
  //
  //   compressed_indices[..., 0] == 0
  //   compressed_indices[..., -1] == nnz.
  //   compressed_indices 必须是非递减序列
  //
  // 因此，避免使用 empty 创建稀疏压缩张量。可以直接使用压缩稀疏构造函数或其他工厂函数如 torch.zeros 等。
  
  // 返回使用不安全的方式创建稀疏压缩张量的结果
  return at::_sparse_compressed_tensor_unsafe(compressed_indices,
                                              plain_indices,
                                              values,
                                              size,
                                              dtype,
                                              layout,
                                              device,
                                              pin_memory);
}

// 调整稀疏 CSR 张量的大小的函数
const Tensor& resize_sparse_csr_(
    const Tensor& self,
    IntArrayRef size,
    ...
    // 检查尺寸是否为非负数
    check_size_nonnegative(size);
    // 检查尺寸的维度是否至少为2，只支持批处理的稀疏CSR矩阵
    TORCH_CHECK(size.size() >= 2, "torch.resize_: Only batched sparse CSR matrices are supported, but got size ", size);
    // 检查要调整的稀疏CSR张量的列数是否小于等于新尺寸的最后一维
    TORCH_CHECK(
        self.size(-1) <= size[size.size() - 1],
        "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported. ",
        "The original number of columns is ",
        self.size(-1),
        " while the requested new number of columns is ", size[size.size() - 1], ".");
    // 调用底层实现函数，调整稀疏CSR张量的大小，保持其非零元素数量不变
    get_sparse_csr_impl(self)->resize_(self._nnz(), size);
    // 返回调整大小后的稀疏CSR张量本身
    return self;
}

Tensor& copy_sparse_compressed_(Tensor& self, const Tensor& src, bool non_blocking) {
  // 使用宏展开，根据self的稀疏压缩布局分派不同的操作
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_", [&]{});
  // 检查self和src的布局是否相同，不同则报错
  TORCH_CHECK(
      self.layout() == src.layout(),
      "torch.copy_: copy of sparse compressed tensors having different layouts is not supported.",
      " self layout is ", self.layout(), " and src layout is ", src.layout());
  // 检查self和src稀疏元素的数量是否相同，不同则报错
  TORCH_CHECK(
      self._nnz() == src._nnz(),  // 实际上，值的复制允许不同形状，只要操作数是可广播的
      "torch.copy_: only sparse compressed tensors with the same number of specified elements are supported.");
  // 获取self和src的压缩维度
  auto self_compressed_dim = compressedDimension(self.layout(), self.sizes());
  auto src_compressed_dim = compressedDimension(src.layout(), src.sizes());
  // 获取self和src在压缩维度上的大小
  auto self_compressed_dims = self.size(self_compressed_dim);
  auto src_compressed_dims = src.size(compressedDimension(src.layout(), src.sizes()));
  // 如果self和src的压缩维度相同，检查它们在该维度上的大小是否一致
  if (self_compressed_dim == src_compressed_dim) {
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimension ",
                self_compressed_dim, " for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  } else {
    # 使用 TORCH_CHECK 函数检查 self_compressed_dims 和 src_compressed_dims 是否相等
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimensions ",
                self_compressed_dim, " and ", src_compressed_dim, ", respectively, for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  }
  # 根据当前 self 的布局分发任务，名称为 "copy_sparse_compressed_"
  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                              [&]{},
                                              [&]{
                                                # 获取 self 和 src 的 values，并获取它们的最后两个维度作为块大小
                                                auto self_values = self.values();
                                                auto src_values = src.values();
                                                auto self_blocksize = DimVector(self_values.sizes().slice(self_values.dim()-2, 2));
                                                auto src_blocksize = DimVector(src_values.sizes().slice(src_values.dim()-2, 2));
                                                # 使用 TORCH_CHECK 函数检查 self_blocksize 和 src_blocksize 是否相等
                                                TORCH_CHECK(self_blocksize == src_blocksize,
                                                            "torch.copy_: copy of sparse compressed tensors having different block sizes is not supported.",
                                                            " self and src block sizes are ", self_blocksize, " and ", src_blocksize, ", respectively.");
                                              });
  # 根据当前 self 的布局分发任务，名称为 "copy_sparse_compressed_"
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                            [&]{
                                              # 如果是行稀疏压缩布局，复制 src 的 crow_indices 到 self 的 crow_indices
                                              self.crow_indices().copy_(src.crow_indices(), non_blocking);
                                              # 复制 src 的 col_indices 到 self 的 col_indices
                                              self.col_indices().copy_(src.col_indices(), non_blocking);
                                            },
                                            [&]{
                                              # 如果是列稀疏压缩布局，复制 src 的 ccol_indices 到 self 的 ccol_indices
                                              self.ccol_indices().copy_(src.ccol_indices(), non_blocking);
                                              # 复制 src 的 row_indices 到 self 的 row_indices
                                              self.row_indices().copy_(src.row_indices(), non_blocking);
                                            });
  # 复制 src 的 values 到 self 的 values
  self.values().copy_(src.values(), non_blocking);
  # 返回更新后的 self 张量
  return self;
}

// 访问 CSR 张量的成员。
// 返回稀疏 CSR 张量的非零元素数量。
int64_t _nnz_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->nnz();
}

// 返回稀疏 CSR 张量的值。
Tensor values_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->values().alias();
}

// 返回稀疏 CSR 张量的行索引。
Tensor crow_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "crow_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

// 返回稀疏 CSR 张量的列索引。
Tensor col_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "col_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

// 返回稀疏 CSR 张量的列压缩列索引。
Tensor ccol_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "ccol_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

// 返回稀疏 CSR 张量的行索引。
Tensor row_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "row_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

// 默认情况下返回稀疏张量的行索引。
Tensor crow_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "crow_indices expected sparse row compressed tensor layout but got ", self.layout());
}

// 默认情况下返回稀疏张量的列索引。
Tensor col_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "col_indices expected sparse row compressed tensor layout but got ", self.layout());
}

// 默认情况下返回稀疏张量的列压缩列索引。
Tensor ccol_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "ccol_indices expected sparse column compressed tensor layout but got ", self.layout());
}

// 默认情况下返回稀疏张量的行索引。
Tensor row_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "row_indices expected sparse column compressed tensor layout but got ", self.layout());
}

// 返回稀疏 CSR 张量的稀疏维度。
int64_t sparse_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->sparse_dim();
}

// 返回稀疏 CSR 张量的稠密维度。
int64_t dense_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->dense_dim();
}

// 调整自身与给定稀疏 CSR 张量大小相同。
const SparseCsrTensor& resize_as_sparse_compressed_(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  auto src_layout = src.layout();
  auto self_layout = self.layout();
  // 调整为与给定稀疏 CSR 张量 src 相同大小。
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      src_layout, "resize_as_sparse_compressed_: src ", []() {});
  // 调整为与自身稀疏 CSR 张量相同大小。
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self_layout, "resize_as_sparse_compressed_: self ", []() {});
  // 注意：实现方法会检查是否需要对成员张量进行大小调整或数据复制。
  get_sparse_csr_impl(self)->resize_as_sparse_compressed_tensor_(src);
  return self;
}
# 根据给定的稀疏 CSR 张量 self 克隆出一个新的稀疏压缩张量
SparseCsrTensor clone_sparse_compressed(
    const SparseCsrTensor& self,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  # 检查是否提供了内存格式选项，如果提供则抛出错误信息
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());

  # 获取 self 的选项信息
  TensorOptions options = self.options();

  # 根据 self 的布局调度获取压缩的索引
  auto compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                      "clone_sparse_compressed",
                                                                      [&]{ return self.crow_indices(); },
                                                                      [&]{ return self.ccol_indices(); });

  # 根据 self 的布局调度获取普通的索引
  auto plain_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                 "clone_sparse_compressed",
                                                                 [&]{ return self.col_indices(); },
                                                                 [&]{ return self.row_indices(); });

  # 调用底层函数创建并返回稀疏压缩张量
  return at::_sparse_compressed_tensor_unsafe(
       compressed_indices.clone(),               # 克隆压缩的索引
       plain_indices.clone(),                    # 克隆普通的索引
       self.values().clone(),                    # 克隆数值部分
       self.sizes(),                             # 原张量的尺寸
       optTypeMetaToScalarType(options.dtype_opt()),  # 转换数据类型
       options.layout_opt(),                     # 选项中的布局
       options.device_opt(),                     # 选项中的设备
       options.pinned_memory_opt());             # 选项中的固定内存标志
}

# 根据输入的稀疏张量 self 创建一个相同形状的空稀疏张量
Tensor empty_like_sparse_csr(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  # 构建包含给定选项的 TensorOptions 对象
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  # 将 self 的选项与上述选项进行合并，并加入内存格式选项
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  # 检查是否请求了不同的稀疏布局，如果是则抛出错误
  TORCH_CHECK(options.layout() == self.layout(),
    "empty_like with different sparse layout is not supported (self is ",
    self.layout(), " but you requested ", options.layout(), ")");

  # 如果选项中的布局是稀疏 CSR
  if (options.layout() == kSparseCsr) {
    # 创建并返回一个新的稀疏 CSR 张量
    auto result = at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices().to(options.device(), self.crow_indices().dtype(), false, true),  # 将行索引移动到指定设备
        self.col_indices().to(options.device(), self.col_indices().dtype(), false, true),    # 将列索引移动到指定设备
        at::empty(self.values().sizes(), options.layout(kStrided)),                         # 创建空的数值部分张量
        self.sizes(),                             # 原张量的尺寸
        optTypeMetaToScalarType(options.dtype()),   # 转换数据类型
        self.layout(),                            # 原张量的布局
        options.device());                        # 选项中的设备
    return result;
  } else if (options.layout() == kSparseCsc) {
    `
        // 根据输入的选项和张量布局调用相应的函数创建稀疏张量，返回结果
        auto result = at::native::_sparse_csc_tensor_unsafe(
            // 将自变量的列索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.ccol_indices().to(options.device(), self.ccol_indices().dtype(), false, true),
            // 将自变量的行索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.row_indices().to(options.device(), self.row_indices().dtype(), false, true),
            // 创建一个空的张量，大小与值张量相同，布局为 Strided
            at::empty(self.values().sizes(), options.layout(kStrided)),
            // 获取自变量的大小
            self.sizes(),
            // 将选项的数据类型转换为标量类型
            optTypeMetaToScalarType(options.dtype()),
            // 获取自变量的布局
            self.layout(),
            // 设置张量的设备
            options.device());
        // 返回创建的稀疏 CSC 张量
        return result;
      } else if (options.layout() == kSparseBsr) {
        // 根据输入的选项和张量布局调用相应的函数创建稀疏张量，返回结果
        auto result = at::native::_sparse_bsr_tensor_unsafe(
            // 将自变量的行索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.crow_indices().to(options.device(), self.crow_indices().dtype(), false, true),
            // 将自变量的列索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.col_indices().to(options.device(), self.col_indices().dtype(), false, true),
            // 创建一个空的张量，大小与值张量相同，布局为 Strided
            at::empty(self.values().sizes(), options.layout(kStrided)),
            // 获取自变量的大小
            self.sizes(),
            // 将选项的数据类型转换为标量类型
            optTypeMetaToScalarType(options.dtype()),
            // 获取自变量的布局
            self.layout(),
            // 设置张量的设备
            options.device());
        // 返回创建的稀疏 BSR 张量
        return result;
      } else if (options.layout() == kSparseBsc) {
        // 根据输入的选项和张量布局调用相应的函数创建稀疏张量，返回结果
        auto result = at::native::_sparse_bsc_tensor_unsafe(
            // 将自变量的列索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.ccol_indices().to(options.device(), self.ccol_indices().dtype(), false, true),
            // 将自变量的行索引转换到指定设备、数据类型，设置非拷贝和使用原始内存
            self.row_indices().to(options.device(), self.row_indices().dtype(), false, true),
            // 创建一个空的张量，大小与值张量相同，布局为 Strided
            at::empty(self.values().sizes(), options.layout(kStrided)),
            // 获取自变量的大小
            self.sizes(),
            // 将选项的数据类型转换为标量类型
            optTypeMetaToScalarType(options.dtype()),
            // 获取自变量的布局
            self.layout(),
            // 设置张量的设备
            options.device());
        // 返回创建的稀疏 BSC 张量
        return result;
      } else if (options.layout() == kStrided) {
        // 根据选项创建一个与自变量相同但指定布局的张量，返回结果
        return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
      } else {
        // 如果布局不被支持，抛出错误
        TORCH_CHECK(false, "Layout ", options.layout(), " is not supported");
      }
template <bool require_view, bool require_copy>
// 定义模板函数 `select_sparse_csr_worker`，接受两个布尔模板参数 `require_view` 和 `require_copy`
Tensor select_sparse_csr_worker(const Tensor& self, int64_t dim, int64_t index) {
  // 声明常量字符串指针 `select_name`，根据 `require_view` 条件设置选择名称
  constexpr const char* select_name = (require_view ? "select()" : "select_copy()");
  // 根据稀疏张量的布局类型分发操作，当前操作为 `select`，但实际无操作内容
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "select", []() { return; });
  // 检查张量的维度不为 0，否则报错，显示不能对 0 维张量应用该操作
  TORCH_CHECK_INDEX(
      self.dim() != 0, select_name, " cannot be applied to a 0-dim tensor.");
  // 对维度进行范围调整，确保在有效范围内
  dim = maybe_wrap_dim(dim, self.dim());
  // 获取指定维度的尺寸
  auto size = self.size(dim);
  // 如果索引超出范围，报错并显示详细信息
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        select_name, ": index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  // 如果索引为负数，将其转换为正数索引
  if (index < 0) {
    index += size;
  }

  // 定义 lambda 函数 `select_strided`，根据 `require_copy` 条件选择相应的选择操作
  auto select_strided = [](const Tensor& self, int64_t dim, int64_t index) {
    if (require_copy) {
      return at::select_copy(self, dim, index);
    } else {
      return self.select(dim, index);
    }
  };

  // 内部断言，确保维度在有效范围内
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < self.dim());

  // 创建新尺寸向量 `new_sizes`，删除指定维度后的尺寸
  auto new_sizes = DimVector(self.sizes());
  new_sizes.erase(new_sizes.begin() + dim);
  // 获取张量的选项（如数据类型、布局、设备等）
  auto options = self.options();

  // 分发操作获取压缩指数和普通指数，根据当前操作类型确定返回的结果
  auto [compressed_indices, plain_indices] =
      AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
          self.layout(),
          "select",
          [&]() {
            return std::make_pair(self.crow_indices(), self.col_indices());
          },
          [&]() {
            return std::make_pair(self.ccol_indices(), self.row_indices());
          });
  // 获取批处理的数量
  auto n_batch = compressed_indices.dim() - 1;

  // 根据维度是否小于批处理数来选择操作类型
  if (dim < n_batch) {
    // 选择批处理维度
    return at::_sparse_compressed_tensor_unsafe(
        compressed_indices.select(dim, index),
        plain_indices.select(dim, index),
        select_strided(self.values(), dim, index),
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  } else if (dim < n_batch + 2) {
    // 选择稀疏维度
    TORCH_CHECK(
        n_batch == 0,
        select_name, ": selecting sparse dimensions is not supported for batched sparse compressed tensors.")
    TORCH_INTERNAL_ASSERT(dim == 0 || dim == 1);

    // 定义块尺寸向量 `blocksize`，根据当前操作类型确定其尺寸
    DimVector blocksize{1, 1};
    AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "select", [&] {}, [&] {
      blocksize[0] = std::max<int64_t>(1, self.values().size(n_batch + 1));
      blocksize[1] = std::max<int64_t>(1, self.values().size(n_batch + 2));
    });

    // 获取压缩指数的选项
    auto indices_options = compressed_indices.options();
    // 获取快速维度，根据当前操作类型选择返回结果
    int64_t fast_dim = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "select", [&]() { return 0; }, [&]() { return 1; });
    int64_t other_dim = (dim == 0 ? 1 : 0);
    // 定义张量 `indices` 和 `values`
    Tensor indices;
    Tensor values;
    // 判断是否为视图，视图的维度为快速维度
    bool is_view = dim == fast_dim;
    if (is_view) {
      // 如果是视图操作，则进行以下步骤
      // 从压缩的索引中选择对应块的起止位置，并在CPU上进行计算
      Tensor start_end = compressed_indices.narrow(0, index / blocksize[dim], 2).cpu();
      // 获取起始位置
      int64_t start = start_end[0].item<int64_t>();
      // 获取结束位置
      int64_t end = start_end[1].item<int64_t>();
      // 根据起止位置切片出索引
      indices = plain_indices.slice(0, start, end);
      // 根据起止位置切片出值
      values = self.values().slice(0, start, end);
    } else {
      // 如果不是视图操作，则进行以下步骤
      // 将压缩的索引转换为COO格式的索引
      Tensor decompressed_indices = at::_convert_indices_from_csr_to_coo(compressed_indices, plain_indices)
        .select(0, 0);
      
      // 找出等于 index / blocksize[dim] 的维度索引
      Tensor dim_indices = at::where(plain_indices.eq(index / blocksize[dim]))[0];
      // 注意，dim_indices 是非负整数的有序序列。下面我们将尝试解决 `dim_indices == arange(start, stop, step)` 的问题。
      // 如果找到解，则 select 操作也将成为视图操作，即使对于 `dim != fast_dim` 的情况也是如此。
      int64_t start{}, end{}, step{};
      // 尝试解决 arange(start, stop, step) 的问题
      if (solve_arange(dim_indices, start, end, step)) {
        // 根据解决方案进行切片操作
        indices = decompressed_indices.slice(0, start, end, step);
        values = self.values().slice(0, start, end, step);
        // 将 is_view 设置为 true
        is_view = true;
      } else {
        // 由于 index_select，select 将进行复制操作
        indices = decompressed_indices.index_select(0, dim_indices);
        values = self.values().index_select(0, dim_indices);
      }
    }

    // 如果需要视图，并且实际上不存在视图，则发出警告
    if (require_view) {
      TORCH_CHECK(values.is_alias_of(self.values()), select_name,
                  ": no view exists for the given input, consider using torch.select_copy.");
    }

    // 将 indices 扩展为一维并转换为 kLong 类型
    indices = indices.unsqueeze(0).to(kLong);
    // 如果需要复制并且是视图，则克隆值
    if (require_copy && is_view) {
      values = values.clone();
    }
    // 返回稀疏 COO 张量
    return at::_sparse_coo_tensor_unsafe(indices, values, new_sizes)._coalesced_(true);
  } else {
    // 如果选择密集维度，则进行以下步骤
    // 根据布局选择相应的策略来进行选择操作
    Tensor new_values = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "select",
        // 非块布局（两个稀疏维度在值中变为一个非零值维度，所以维度在左边找到）
        [&]() { return select_strided(self.values(), dim - 1, index); },
        // 块布局（两个稀疏维度在值中变为一个非零值维度加两个块形状维度，所以维度在右边找到）
        [&]() { return select_strided(self.values(), dim + 1, index); });
    // 返回不安全的压缩稀疏张量
    return at::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        new_values,
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  }
}

// 选择稀疏张量的 CSR 格式的特定维度和索引
Tensor select_sparse_csr(const Tensor& self, int64_t dim, int64_t index) {
    // 调用稀疏 CSR 格式选择器的工作函数，选择并返回结果张量
    return select_sparse_csr_worker<true, false>(self, dim, index);
}

// 复制稀疏张量的 CSR 格式的特定维度和索引
Tensor select_copy_sparse_csr(const Tensor& self, int64_t dim, int64_t index) {
    // 调用稀疏 CSR 格式选择器的工作函数，并复制选择结果返回
    return select_sparse_csr_worker<false, true>(self, dim, index);
}

// 命名空间结束标记，结束了在 at::native 命名空间内的定义
} // namespace at::native
```