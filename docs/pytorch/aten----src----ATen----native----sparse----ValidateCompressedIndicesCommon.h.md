# `.\pytorch\aten\src\ATen\native\sparse\ValidateCompressedIndicesCommon.h`

```
#pragma once
#include <ATen/Dispatch.h>  // ATen 分发机制的头文件
#include <ATen/Tensor.h>  // ATen 张量的头文件
#include <ATen/Utils.h>  // ATen 实用工具的头文件
#include <ATen/native/TensorIterator.h>  // ATen 张量迭代器的头文件
#include <ATen/native/sparse/Macros.h>  // ATen 稀疏张量宏定义的头文件
#include <ATen/native/SparseTensorUtils.h>  // ATen 稀疏张量工具的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>  // ATen 函数的头文件
#include <ATen/NativeFunctions.h>  // ATen 原生函数的头文件
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>  // ATen 稀疏 COO 张量创建的头文件
#include <ATen/ops/arange.h>  // ATen arange 操作的头文件
#include <ATen/ops/empty.h>  // ATen empty 操作的头文件
#include <ATen/ops/tensor.h>  // ATen tensor 操作的头文件
#endif

#ifdef GPUCC
#define NAME "compressed_index_invariance_checks_cuda"  // 定义 GPU 环境下的名称宏
#else
#define NAME "compressed_index_invariance_checks_cpu"  // 定义 CPU 环境下的名称宏
#endif

#define INVARIANT_CHECK_FUNC_API static INLINE FUNCAPI void  // 定义不变性检查函数的 API

namespace at::native {

namespace {

// NOTE: all the checks but the very last one are designed
// to work with vectors.
// To enable vectorization one would need to write a conversion
// Vec -> bool and make kernel launchers call into vectorized
// execution paths.

// All the invariants are described in
// https://pearu.github.io/bsr_tensor_invariants.html NOTE: in the code we also
// use `cidx/idx` to refer to `compressed_indices/plain_indices` respectively.

// 检查函数，用于在 CUDA 或 CPU 环境下进行断言检查
INVARIANT_CHECK_FUNC_API
_assert(const bool cond, const char* const message) {
#ifdef GPUCC
  CUDA_KERNEL_ASSERT(cond && message);  // 在 GPU 环境下使用 CUDA_KERNEL_ASSERT 断言
#else
  TORCH_CHECK(cond, message);  // 在 CPU 环境下使用 TORCH_CHECK 断言
#endif
}

enum class CDimName : bool { CRow, CCol };  // 枚举类型，表示压缩张量的行或列维度

// 不变性检查 5.1
// 压缩索引的第一个元素必须为 0
template <CDimName cdim_name, typename index_t>
INVARIANT_CHECK_FUNC_API _check_first_cidx_is_zero(
    const index_t& cidx,
    const index_t& zero) {
  const bool invariant = cidx == zero;
  if (cdim_name == CDimName::CRow) {
    _assert(invariant, "`crow_indices[..., 0] == 0` is not satisfied.");  // 断言行压缩索引的第一个元素为 0
  } else {
    _assert(invariant, "`ccol_indices[..., 0] == 0` is not satisfied.");  // 断言列压缩索引的第一个元素为 0
  }
}

// 不变性检查 5.2
// 压缩索引的最后一个元素必须为 nnz（非零元素数量）
template <CDimName cdim_name, typename index_t>
INVARIANT_CHECK_FUNC_API _check_last_cidx_is_nnz(
    const index_t& cidx,
    const index_t& nnz) {
  const bool invariant = cidx == nnz;
  if (cdim_name == CDimName::CRow) {
    _assert(invariant, "`crow_indices[..., -1] == nnz` is not satisfied.");  // 断言行压缩索引的最后一个元素为 nnz
  } else {
    _assert(invariant, "`ccol_indices[..., -1] == nnz` is not satisfied.");  // 断言列压缩索引的最后一个元素为 nnz
  }
}

// 不变性检查 5.3
// 压缩索引中相邻元素的差值必须在 [0, dim] 范围内
template <CDimName cdim_name, typename index_t>
INVARIANT_CHECK_FUNC_API _check_cidx_nondecreasing_locally_bounded_sequence(
    const index_t& cidx,
    const index_t& cidx_next,
    const index_t& zero,
    const index_t& dim) {
  const auto s_cidx = cidx_next - cidx;
  const bool invariant = zero <= s_cidx && s_cidx <= dim;
  if (cdim_name == CDimName::CRow) {
    _assert(
        invariant,
        "`0 <= crow_indices[..., 1:] - crow_indices[..., :-1] <= ncols` is not satisfied.");  // 断言行压缩索引中相邻元素的差值在 [0, ncols] 范围内
  } else {
    _assert(
        invariant,
        "`0 <= ccol_indices[..., 1:] - ccol_indices[..., :-1] <= nrows` is not satisfied.");  // 断言列压缩索引中相邻元素的差值在 [0, nrows] 范围内
  }
}
// Invariants 5.4 and 5.5
// 0 <= plain_index < plain_dim.
template <CDimName cdim_name, typename index_t>
INVARIANT_CHECK_FUNC_API _check_idx_bounds(
    const index_t& idx,
    const index_t& zero,
    const index_t& dim) {
  // 检查索引是否在指定范围内
  const bool invariant = zero <= idx && idx < dim;
  // 根据维度名称检查不变量并断言
  if (cdim_name == CDimName::CRow) {
    _assert(invariant, "`0 <= col_indices < ncols` is not satisfied.");
  } else {
    _assert(invariant, "`0 <= row_indices < nrows` is not satisfied.");
  }
}

// Invariant 5.6
// plain_indices[..., compressed_indices[..., i - 1]:compressed_indices[..., i]]
// for all i = 1, ..., compressed_dim
// are sorted and distinct along the last dimension values.
template <CDimName cdim_name, typename index_t>
INVARIANT_CHECK_FUNC_API _check_idx_sorted_distinct_vals_slices_with_cidx(
    const index_t* RESTRICT ptr_idx_batch,
    const index_t cidx,
    const index_t cidx_next) {
  // 注意 ptr_idx_batch = &idx[batch_idx] 是连续的
  const auto* RESTRICT slice_begin = ptr_idx_batch + cidx;
  const auto* RESTRICT slice_end = ptr_idx_batch + cidx_next;
  // 检查每个切片是否在最后一个维度上按顺序且唯一
  for (auto* RESTRICT curr = slice_begin; (slice_begin < slice_end) && (curr + 1 < slice_end); ++curr) {
    const auto invariant = *curr < *(curr + 1);
    if (cdim_name == CDimName::CRow) {
      _assert(
          invariant,
          "`col_indices[..., crow_indices[..., i - 1]:crow_indices[..., i]] "
          "for all i = 1, ..., nrows "
          "are sorted and distinct along the last dimension values` "
          "is not satisfied.");
    } else {
      _assert(
          invariant,
          "`row_indices[..., ccol_indices[..., i - 1]:ccol_indices[..., i]] "
          "for all i = 1, ..., ncols "
          "are sorted and distinct along the last dimension values` "
          "is not satisfied.");
    }
  }
}

static inline int64_t indexCount(IntArrayRef sizes) {
  // 计算数组 sizes 中所有元素的乘积
  int64_t res = 1;
  for (const auto& s : sizes) {
    res *= s;
  }
  return res;
}

template <typename func_t, typename vec_func_t>
struct EmptyVecKernel {
  // 空的向量操作核函数，不执行任何操作
  static void launch(
      TensorIteratorBase& iter,
      const func_t& f,
      const vec_func_t& vec_f) {}
};

template <typename scalar_t>
using DummyVec = scalar_t;

template <
    template <typename func_t>
    class kernel_t,
    template <typename func_t, typename vec_func_t>
    class vec_kernel_t>
struct KernelLauncher {
  // 核函数启动器，根据传入的核函数类型选择启动适当的实现
  template <typename func_t, typename vec_func_t>
  static void launch(
      TensorIteratorBase& iter,
      const func_t& f,
      const vec_func_t& vec_f) {
    vec_kernel_t<func_t, vec_func_t>::launch(iter, f, vec_f);
  }

  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

template <
    CDimName cdim_name,
    template <typename func_t>
    class kernel_t,
    template <typename func_t, typename vec_func_t>
    class vec_kernel_t = EmptyVecKernel,
    template <typename scalar_t> class Vec = DummyVec,
    size_t static_shape_max_len = 0>
// 验证压缩稀疏索引核心函数，用于检查稀疏张量索引的形状是否正确
void _validate_compressed_sparse_indices_kernel(
    const Tensor& cidx,  // 压缩的行或列索引张量
    const Tensor& idx,   // 索引张量
    const int64_t cdim,  // 压缩维度的大小（行或列）
    const int64_t dim,   // 维度大小（行或列）
    const int64_t nnz) { // 非零元素数量

  if (cdim_name == CDimName::CRow) {  // 如果压缩维度名为行
    // 检查行索引张量的最后一个维度是否为 cdim + 1
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "crow_indices have wrong shape: ",
        "crow_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "nrows + 1 = ",
        cdim + 1);
    // 检查列索引张量的最后一个维度是否为 nnz
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "col_indices have wrong shape: ",
        "col_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  } else {  // 如果压缩维度名为列
    // 检查列索引张量的最后一个维度是否为 cdim + 1
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "ccol_indices have wrong shape: ",
        "ccol_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "ncols + 1 = ",
        cdim + 1);
    // 检查行索引张量的最后一个维度是否为 nnz
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "row_indices have wrong shape: ",
        "row_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  }

  // 使用 KernelLauncher 类调用相应的内核函数，确保输出不是 void 类型的 lambda 函数
  using KernelLauncher = KernelLauncher<kernel_t, vec_kernel_t>;

  // 创建一个与 cidx 张量相同选项的空张量，用于 TensorIterator 的输出
  const auto dummy = at::empty({1}, cidx.options());

  // 捕获来自大尺寸维度的整数溢出。否则，在将 int64_t 类型的维度强制转换为 index 类型（例如 int32_t）时，
  // 不变式检查可能会因为假异常而失败，或者当维度被强制转换为较小的整数类型时会成功产生假阳性结果。
  {
    AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [cdim, dim, nnz]() {
      if (cdim_name == CDimName::CRow) {
        // 检查列维度是否有 64 位整数溢出
        TORCH_CHECK(static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
                    sizeof(index_t) * 8, "-bit integer overflow in column dimension = ", dim);
        // 检查行维度是否有 64 位整数溢出
        TORCH_CHECK(static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
                    sizeof(index_t) * 8, "-bit integer overflow in row dimension = ", cdim);
      } else {
        // 检查行维度是否有 64 位整数溢出
        TORCH_CHECK(static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
                    sizeof(index_t) * 8, "-bit integer overflow in row dimension = ", dim);
        // 检查列维度是否有 64 位整数溢出
        TORCH_CHECK(static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
                    sizeof(index_t) * 8, "-bit integer overflow in column dimension = ", cdim);
      }
      // 检查 nnz 是否有 64 位整数溢出
      TORCH_CHECK(static_cast<int64_t>(static_cast<index_t>(nnz)) == nnz,
                  sizeof(index_t) * 8, "-bit integer overflow in nnz = ", nnz);
    });
  }

  // 不变式 5.4 和 5.5
  {
    // 设置 TensorIteratorConfig 配置，其中不检查内存重叠，添加一个拥有的输出（与 idx 张量展开后相同），并添加 idx 作为输入
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .add_owned_output(dummy.expand_as(idx))
                    .add_input(idx)
                    .build();
  // 使用 AT_DISPATCH_INDEX_TYPES 宏展开不同索引类型的操作，其中 idx.scalar_type() 返回索引类型
  AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [&iter, dim]() {
    // 定义零常量，类型为 index_t
    const auto zero = index_t{0};
    // 使用 KernelLauncher 启动核函数，对每个索引进行操作
    KernelLauncher::launch(iter, [zero, dim] FUNCAPI(index_t idx) -> index_t {
      // 检查索引 idx 是否在指定的边界范围内
      _check_idx_bounds<cdim_name, index_t>(idx, zero, dim);
      // 返回固定的索引值 0
      return 0;
    });
  });

  // Invariants 5.1, 5.2, 5.3, 5.6
  {
    // 从 cidx 中切片获取第一个索引
    const auto cidx_first = cidx.slice(-1, 0, 1);
    // 从 cidx 中切片获取最后一个索引
    const auto cidx_last = cidx.slice(-1, cdim, cdim + 1);

    // 从 cidx 中切片获取当前索引范围
    const auto cidx_curr = cidx.slice(-1, 0, cdim);
    // 从 cidx 中切片获取下一个索引范围
    const auto cidx_next = cidx.slice(-1, 1, cdim + 1);

    // 计算批处理维度，即从 cidx 的大小中移除最后一个维度
    const auto batch_dims = cidx.sizes().slice(0, cidx.dim() - 1);
    // 计算批处理的数量
    const auto batch_count = indexCount(batch_dims);
    // 创建批处理索引张量，形状为 batch_dims，并在最后增加一个维度
    const auto batch_idx =
        at::arange(batch_count, cidx.options()).view(batch_dims).unsqueeze_(-1);

    // 获取 idx 的维度数量
    const auto idx_ndims = idx.dim();

    // 创建 idx 的几何信息持有者，其中包括大小和步幅
    const auto idx_geometry_holder = at::sparse::TensorGeometryHolder<static_shape_max_len>(idx);
    // 获取 idx 的大小
    const auto idx_sizes = std::get<0>(*idx_geometry_holder);
    // 获取 idx 的步幅
    const auto idx_strides = std::get<1>(*idx_geometry_holder);

    // 创建张量迭代器配置
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)  // 设置不检查内存重叠
                    .add_owned_output(dummy.expand_as(cidx_curr))  // 添加输出张量
                    .add_input(cidx_first)  // 添加输入张量 cidx_first
                    .add_input(cidx_last)   // 添加输入张量 cidx_last
                    .add_input(cidx_curr)   // 添加输入张量 cidx_curr
                    .add_input(cidx_next)   // 添加输入张量 cidx_next
                    .add_input(batch_idx)   // 添加输入张量 batch_idx
                    .build();               // 构建迭代器配置
  }
    // 使用 AT_DISPATCH_INDEX_TYPES 宏来根据索引类型分派任务
    AT_DISPATCH_INDEX_TYPES(
        idx.scalar_type(),
        NAME,
        [&iter, &idx, dim, nnz, idx_ndims, &idx_sizes, &idx_strides]() {
          // 获取索引数据的指针，并声明零值
          const auto* RESTRICT ptr_idx = idx.const_data_ptr<index_t>();
          const auto zero = index_t{0};
          // 调用 KernelLauncher 的 launch 方法，启动核函数
          KernelLauncher::launch(
              iter,
              // 定义核函数的 lambda 函数
              [zero, dim, nnz, idx_ndims, idx_sizes, idx_strides, ptr_idx] FUNCAPI(
                  index_t cidx_first,
                  index_t cidx_last,
                  index_t cidx_curr,
                  index_t cidx_next,
                  index_t batch_idx) -> index_t {
                // 不变式 5.1：检查第一个 cidx 是否为零
                _check_first_cidx_is_zero<cdim_name, index_t>(cidx_first, zero);
                // 不变式 5.2：检查最后一个 cidx 是否为 nnz
                _check_last_cidx_is_nnz<cdim_name, index_t>(cidx_last, nnz);
                // 不变式 5.3：检查局部递增且受限制的 cidx 序列
                _check_cidx_nondecreasing_locally_bounded_sequence<
                    cdim_name,
                    index_t>(cidx_curr, cidx_next, zero, dim);
                // 不变式 5.6
                // 注意：下面的实现是无同步的，但不保证工作在不同线程之间良好平衡。
                // 注意：当 nnz==0 时不应测试 5.6。幸运的是，当 nnz==0 时，下面的代码无操作。
                // 初始化索引偏移量
                int64_t idx_offset = 0;
                // 假设每个批次中的索引是连续的：
                int64_t tmp = batch_idx * nnz;
                // 在 nnz > 0 时，根据索引大小和步长计算索引偏移量
                for (int i = idx_ndims - 1;
                     i >= 0 && nnz > 0;  // 当 nnz==0 时提前退出循环
                     i--) {
                  int64_t div = tmp / idx_sizes[i];
                  idx_offset += (tmp - div * idx_sizes[i]) * idx_strides[i];
                  tmp = div;
                }
                // 获取当前批次的指针
                const auto* RESTRICT ptr_idx_batch = ptr_idx + idx_offset;
                // 检查带有 cidx 的排序和唯一值分片的索引
                _check_idx_sorted_distinct_vals_slices_with_cidx<
                    cdim_name,
                    index_t>(ptr_idx_batch, cidx_curr, cidx_next);
                return 0;
              });
        });
} // 结束 validate_compressed_sparse_indices_kernel 函数定义

template <
    // 定义模板参数 kernel_t，表示内核类型
    template <typename func_t>
    class kernel_t,
    // 定义模板参数 vec_kernel_t，默认为 EmptyVecKernel，表示向量化内核类型
    template <typename func_t, typename vec_func_t>
    class vec_kernel_t = EmptyVecKernel,
    // 定义模板参数 Vec，默认为 DummyVec，表示向量类型
    template <typename scalar_t> class Vec = DummyVec>
// 定义 validate_compressed_sparse_indices_kernel 函数，验证压缩稀疏索引的内核
void validate_compressed_sparse_indices_kernel(
    // 是否为行压缩表示
    const bool is_crow,
    // 压缩行索引张量
    const Tensor& cidx,
    // 索引张量
    const Tensor& idx,
    // 压缩维度
    const int64_t cdim,
    // 维度
    const int64_t dim,
    // 非零元素个数
    const int64_t nnz) {
  // 最大索引维度为 8，支持最多 7 维批处理
  constexpr size_t idx_max_ndims = 8; // up to 7-dim batch.
  // 索引张量的维度
  const size_t idx_ndims = static_cast<size_t>(idx.dim());

  // 如果是行压缩表示
  if (is_crow) {
    // 如果索引张量的维度不超过最大索引维度
    if (idx_ndims <= idx_max_ndims) {
      // 调用特化的压缩稀疏索引验证内核函数，使用 CRow 作为行压缩名称
      _validate_compressed_sparse_indices_kernel<
          CDimName::CRow,
          kernel_t,
          vec_kernel_t,
          Vec,
          idx_max_ndims>(cidx, idx, cdim, dim, nnz);
    }
    // 否则
    else {
      // 调用特化的压缩稀疏索引验证内核函数，使用 CRow 作为行压缩名称，不指定最大索引维度
      _validate_compressed_sparse_indices_kernel<
          CDimName::CRow,
          kernel_t,
          vec_kernel_t,
          Vec>(cidx, idx, cdim, dim, nnz);
    }
  } 
  // 如果是列压缩表示
  else {
    // 如果索引张量的维度不超过最大索引维度
    if (idx_ndims <= idx_max_ndims) {
      // 调用特化的压缩稀疏索引验证内核函数，使用 CCol 作为列压缩名称
      _validate_compressed_sparse_indices_kernel<
          CDimName::CCol,
          kernel_t,
          vec_kernel_t,
          Vec,
          idx_max_ndims>(cidx, idx, cdim, dim, nnz);
    }
    // 否则
    else {
      // 调用特化的压缩稀疏索引验证内核函数，使用 CCol 作为列压缩名称，不指定最大索引维度
      _validate_compressed_sparse_indices_kernel<
          CDimName::CCol,
          kernel_t,
          vec_kernel_t,
          Vec>(cidx, idx, cdim, dim, nnz);
    }
  }
}

} // 结束命名空间 at::native

} // 结束命名空间 namespace
```