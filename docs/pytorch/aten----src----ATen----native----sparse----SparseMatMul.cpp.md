# `.\pytorch\aten\src\ATen\native\sparse\SparseMatMul.cpp`

```py
// 定义宏以仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 核心头文件
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
// 包含范围工具
#include <c10/util/irange.h>
// 包含无序映射的头文件
#include <unordered_map>

// 如果未定义每个操作符头文件，则包含函数和本地函数头文件，否则包含特定操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_sparse_matmul_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#endif

// ATen 的 native 命名空间
namespace at::native {

// ATen 稀疏模块命名空间
using namespace at::sparse;

/*
    这是 SMMP 算法的实现：
     "Sparse Matrix Multiplication Package (SMMP)"

      Randolph E. Bank 和 Craig C. Douglas
      https://doi.org/10.1007/BF02070824
*/
namespace {
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
// 将 CSR 格式转换为 COO 格式的函数
void csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[]) {
  /*
    将压缩的行指针转换为行索引数组
    输入：
      `n_row` 是 `Ap` 中的行数
      `Ap` 是行指针

    输出：
      `Bi` 是行索引
  */
  for (const auto i : c10::irange(n_row)) {
    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      Bi[jj] = i;
    }
  }
}

// 模板函数，计算矩阵 `C = A@B` 操作中矩阵 `C` 所需的缓冲区大小
template<typename index_t_ptr = int64_t*>
int64_t _csr_matmult_maxnnz(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const index_t_ptr Bp,
    const index_t_ptr Bj) {
  /*
    计算矩阵 `C = A@B` 操作中 `C` 矩阵所需的缓冲区大小

    要求矩阵必须是适当的 CSR 结构，并且它们的维度应该是兼容的。
  */
  // 创建大小为 `n_col`，初始值为 `-1` 的掩码向量
  std::vector<int64_t> mask(n_col, -1);
  int64_t nnz = 0;
  for (const auto i : c10::irange(n_row)) {
    int64_t row_nnz = 0;

    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      int64_t j = Aj[jj];
      for (int64_t kk = Bp[j]; kk < Bp[j + 1]; kk++) {
        int64_t k = Bj[kk];
        if (mask[k] != i) {
          mask[k] = i;
          row_nnz++;
        }
      }
    }
    int64_t next_nnz = nnz + row_nnz;
    nnz = next_nnz;
  }
  return nnz;
}

// 模板函数，计算矩阵 `C = A@B` 操作中的 CSR 条目
template<typename index_t_ptr, typename scalar_t_ptr>
void _csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const scalar_t_ptr Ax,
    const index_t_ptr Bp,
    const index_t_ptr Bj,
    const scalar_t_ptr Bx,
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cp[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cj[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename scalar_t_ptr::value_type Cx[]) {
  /*
    计算矩阵 `C = A@B` 操作的 CSR 条目
  */
    /*
    The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
    should be compatible.

    Inputs:
      `n_row`         - number of rows in matrix A
      `n_col`         - number of columns in matrix B
      `Ap[n_row+1]`   - row pointer of matrix A
      `Aj[nnz(A)]`    - column indices of matrix A
      `Ax[nnz(A)]`    - nonzeros of matrix A
      `Bp[?]`         - row pointer of matrix B
      `Bj[nnz(B)]`    - column indices of matrix B
      `Bx[nnz(B)]`    - nonzeros of matrix B
    Outputs:
      `Cp[n_row+1]`   - row pointer of resulting matrix C
      `Cj[nnz(C)]`    - column indices of resulting matrix C
      `Cx[nnz(C)]`    - nonzeros of resulting matrix C

    Note:
      Output arrays Cp, Cj, and Cx must be preallocated
    */
    using index_t = typename index_t_ptr::value_type;
    using scalar_t = typename scalar_t_ptr::value_type;

    // Initialize vectors to manage linked lists and accumulate sums
    std::vector<index_t> next(n_col, -1); // next pointer for each column
    std::vector<scalar_t> sums(n_col, 0); // accumulated sums for each column

    int64_t nnz = 0; // Initialize the number of nonzeros in C

    Cp[0] = 0; // The first row pointer of C starts at index 0

    // Iterate over each row of matrix A
    for (const auto i : c10::irange(n_row)) {
      index_t head = -2; // Initialize head of linked list to -2 (sentinel value)
      index_t length = 0; // Initialize length of linked list

      index_t jj_start = Ap[i];    // Start index for columns in current row of A
      index_t jj_end = Ap[i + 1];  // End index for columns in current row of A
      // Iterate over each column index and value in current row of A
      for (const auto jj : c10::irange(jj_start, jj_end)) {
        index_t j = Aj[jj];      // Column index in A
        scalar_t v = Ax[jj];     // Value in A corresponding to column j

        index_t kk_start = Bp[j];    // Start index for columns in row j of B
        index_t kk_end = Bp[j + 1];  // End index for columns in row j of B
        // Iterate over each column index and value in row j of B
        for (const auto kk : c10::irange(kk_start, kk_end)) {
          index_t k = Bj[kk];         // Column index in B
          sums[k] += v * Bx[kk];      // Accumulate sum for corresponding entry in C

          if (next[k] == -1) {        // If next pointer for column k is -1 (unset)
            next[k] = head;           // Set next pointer to current head of list
            head = k;                 // Update head to current column k
            length++;                 // Increment length of linked list
          }
        }
      }

      // Iterate over linked list to populate Cj and Cx
      for (C10_UNUSED const auto jj : c10::irange(length)) {
        // Store column index and corresponding value in C
        Cj[nnz] = head;
        Cx[nnz] = sums[head];
        nnz++;                      // Increment number of nonzeros in C

        index_t temp = head;
        head = next[head];           // Move head to next entry in linked list

        next[temp] = -1;             // Clear next pointer
        sums[temp] = 0;              // Clear accumulated sum
      }

      // Sort column indices in C to ensure they are in ascending order
      auto col_indices_accessor = StridedRandomAccessor<int64_t>(Cj + nnz - length, 1);
      auto val_accessor = StridedRandomAccessor<scalar_t>(Cx + nnz - length, 1);
      auto kv_accessor = CompositeRandomAccessorCPU<
        decltype(col_indices_accessor), decltype(val_accessor)
      >(col_indices_accessor, val_accessor);
      std::sort(kv_accessor, kv_accessor + length, [](const auto& lhs, const auto& rhs) -> bool {
          return get<0>(lhs) < get<0>(rhs);
      });

      Cp[i + 1] = nnz; // Update row pointer for next row in C
    }
}
// 结束匿名命名空间，这是 C++ 中用来限定变量和函数作用域的一种方式

template <typename scalar_t>
void sparse_matmul_kernel(
    Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
  /*
    计算稀疏矩阵 mat1 和 mat2 之间的稀疏矩阵乘法，这两个矩阵都以 COO 格式表示。
  */

  auto M = mat1.size(0);  // 获取矩阵 mat1 的行数
  auto N = mat2.size(1);  // 获取矩阵 mat2 的列数

  const auto mat1_csr = mat1.to_sparse_csr();  // 将 mat1 转换为 CSR 格式的稀疏矩阵
  const auto mat2_csr = mat2.to_sparse_csr();  // 将 mat2 转换为 CSR 格式的稀疏矩阵

  // 使用 StridedRandomAccessor 封装 CSR 矩阵的各个数据指针和步长
  auto mat1_crow_indices_ptr = StridedRandomAccessor<int64_t>(
      mat1_csr.crow_indices().data_ptr<int64_t>(),
      mat1_csr.crow_indices().stride(-1));
  auto mat1_col_indices_ptr = StridedRandomAccessor<int64_t>(
      mat1_csr.col_indices().data_ptr<int64_t>(),
      mat1_csr.col_indices().stride(-1));
  auto mat1_values_ptr = StridedRandomAccessor<scalar_t>(
      mat1_csr.values().data_ptr<scalar_t>(),
      mat1_csr.values().stride(-1));
  auto mat2_crow_indices_ptr = StridedRandomAccessor<int64_t>(
      mat2_csr.crow_indices().data_ptr<int64_t>(),
      mat2_csr.crow_indices().stride(-1));
  auto mat2_col_indices_ptr = StridedRandomAccessor<int64_t>(
      mat2_csr.col_indices().data_ptr<int64_t>(),
      mat2_csr.col_indices().stride(-1));
  auto mat2_values_ptr = StridedRandomAccessor<scalar_t>(
      mat2_csr.values().data_ptr<scalar_t>(),
      mat2_csr.values().stride(-1));

  // 计算输出稀疏矩阵的非零元素个数上限
  const auto nnz = _csr_matmult_maxnnz(
      M,
      N,
      mat1_crow_indices_ptr,
      mat1_col_indices_ptr,
      mat2_crow_indices_ptr,
      mat2_col_indices_ptr);

  // 获取输出稀疏矩阵的索引和值
  auto output_indices = output._indices();
  auto output_values = output._values();

  // 创建输出稀疏矩阵的指针数组，并重新调整输出的索引和值的大小
  Tensor output_indptr = at::empty({M + 1}, kLong);
  at::native::resize_output(output_indices, {2, nnz});
  at::native::resize_output(output_values, nnz);

  // 获取输出稀疏矩阵的行索引和列索引
  Tensor output_row_indices = output_indices.select(0, 0);
  Tensor output_col_indices = output_indices.select(0, 1);

  // 使用 _csr_matmult 函数执行 CSR 格式稀疏矩阵乘法计算
  // TODO: 为了提升性能，考虑用 CSR @ CSC 内核替代。
  _csr_matmult(
      M,
      N,
      mat1_crow_indices_ptr,
      mat1_col_indices_ptr,
      mat1_values_ptr,
      mat2_crow_indices_ptr,
      mat2_col_indices_ptr,
      mat2_values_ptr,
      output_indptr.data_ptr<int64_t>(),
      output_col_indices.data_ptr<int64_t>(),
      output_values.data_ptr<scalar_t>());

  // 将输出矩阵的 CSR 格式转换为 COO 格式
  csr_to_coo(M, output_indptr.data_ptr<int64_t>(), output_row_indices.data_ptr<int64_t>());
  output._coalesced_(true);  // 将输出稀疏矩阵进行合并
}

} // 结束匿名命名空间
// 使用稀疏矩阵在 CPU 上进行矩阵乘法，返回结果稀疏张量
Tensor sparse_sparse_matmul_cpu(const Tensor& mat1_, const Tensor& mat2_) {
  // 断言输入的第一个和第二个张量是稀疏张量
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());

  // 检查输入张量的维度是否为2
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);

  // 检查输入张量是否为标量值（dense_dim == 0）
  TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat1_.dense_dim(), "D values");
  TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat2_.dense_dim(), "D values");

  // 检查矩阵乘法是否可行，即 mat1 的列数必须等于 mat2 的行数
  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  // 检查两个输入张量的数据类型是否相同
  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  // 创建一个和 mat1_ 相同大小的空张量作为输出
  auto output = at::native::empty_like(mat1_);

  // 调整输出张量为稀疏张量，并清除其中的元素
  output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

  // 根据 mat1_ 和 mat2_ 的数据类型分发到相应的计算内核进行稀疏矩阵乘法
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_matmul_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });

  // 返回计算结果的稀疏张量
  return output;
}
```