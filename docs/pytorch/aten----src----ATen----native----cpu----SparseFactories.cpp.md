# `.\pytorch\aten\src\ATen\native\cpu\SparseFactories.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/sparse/SparseFactories.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

// 进入 ATen 库的 native 命名空间
namespace at::native {

// 声明一个私有函数 _spdiags_kernel_cpu，处理稀疏对角线矩阵的计算
namespace {
void _spdiags_kernel_cpu(
    TensorIterator& iter,             // 张量迭代器，用于操作张量迭代
    const TensorBase& diagonals,      // 输入的对角线张量
    TensorBase& values,               // 输出的值张量
    TensorBase& indices) {            // 输出的索引张量
  auto* row_index_write_ptr = indices.data_ptr<int64_t>();   // 索引张量的写指针
  auto* col_index_write_ptr = row_index_write_ptr ? row_index_write_ptr + indices.stride(0) : nullptr;  // 列索引的写指针，根据行索引偏移得到
  const int64_t diagonals_index_stride = diagonals.stride(0);  // 对角线张量在第一维上的步长
  const int64_t diagonals_read_stride = diagonals.stride(1);   // 对角线张量在第二维上的步长
  // 使用宏展开，处理所有数据类型（包括复数和特殊类型）
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      diagonals.scalar_type(),
      "spdiags_cpu",   // 分派名称
      [&] {   // lambda 函数开始
        auto* const values_write_ptr = values.data_ptr<scalar_t>();   // 值张量的写指针
        const auto* const diagonals_ptr = diagonals.const_data_ptr<scalar_t>();   // 对角线张量的常量数据指针

        // 调用 CPU 内核函数进行计算
        cpu_kernel(
            iter,
            [&](int64_t diag_index,
                int64_t diag_offset,
                int64_t out_offset,
                int64_t n_out) -> int64_t {   // lambda 函数开始，处理每个对角线
              if (n_out > 0) {   // 如果有输出
                auto* rows_start = row_index_write_ptr + out_offset;   // 行索引的起始位置
                auto* cols_start = col_index_write_ptr + out_offset;   // 列索引的起始位置
                auto* vals_start = values_write_ptr + out_offset;      // 值的起始位置
                const int64_t first_col = std::max<int64_t>(diag_offset, 0);  // 第一个列的位置
                const int64_t first_row = first_col - diag_offset;    // 第一个行的位置
                auto* data_read = (diagonals_ptr +
                                   diagonals_index_stride * diag_index +
                                   first_col * diagonals_read_stride);  // 对角线数据的读指针
                for (int64_t i = 0; i < n_out; ++i) {   // 对每个输出执行循环
                  rows_start[i] = first_row + i;    // 设置行索引
                  cols_start[i] = first_col + i;    // 设置列索引
                  vals_start[i] = data_read[i * diagonals_read_stride];  // 设置值
                }
              }
              // 返回虚拟值，表示处理完成
              return 0;
            });
      });
}

} // namespace

// 注册 spdiags_kernel_stub 分发函数，指向 _spdiags_kernel_cpu 函数
REGISTER_DISPATCH(spdiags_kernel_stub, &_spdiags_kernel_cpu)

} // namespace at::native
```