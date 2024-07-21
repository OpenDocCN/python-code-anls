# `.\pytorch\aten\src\ATen\native\cpu\SampledAddmmKernel.cpp`

```
// 使用宏定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS，可能用于指定仅用于方法操作符的相关设置

// 引入 ATen 库中所需的头文件
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/SampledAddmmKernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

// 定义 at::native 命名空间
namespace at::native {

// 定义匿名命名空间，用于实现局部函数
namespace {

// 实现 sampled_addmm_sparse_csr_kernel_impl 函数模板
template <typename scalar_t, typename index_t>
void sampled_addmm_sparse_csr_kernel_impl(
    const Tensor& mat1,          // 第一个稀疏张量
    const Tensor& mat2,          // 第二个稀疏张量
    const Scalar& beta,          // 乘法参数 beta
    const Scalar& alpha,         // 乘法参数 alpha
    const Tensor& result) {      // 结果张量

  int64_t nnz = result._nnz();  // 获取结果张量的非零元素数

  auto beta_ = beta.to<scalar_t>();   // 将 beta 转换为 scalar_t 类型
  auto alpha_ = alpha.to<scalar_t>(); // 将 alpha 转换为 scalar_t 类型

  // 获取 mat1 和 mat2 的数据指针
  const scalar_t* mat1_data = mat1.const_data_ptr<scalar_t>();
  const scalar_t* mat2_data = mat2.const_data_ptr<scalar_t>();

  // 获取 mat1 和 mat2 的维度信息
  int64_t M = mat1.size(-2);   // mat1 的倒数第二维度大小
  int64_t K = mat1.size(-1);   // mat1 和 mat2 的最后一维度大小
  int64_t N = mat2.size(-2);   // mat2 的倒数第二维度大小
  int64_t B = mat1.numel() / M / K;  // mat1 的元素数除以 M*K，得到 B 的大小

  // 重塑结果张量的值、行偏移和列索引张量的形状
  auto values = result.values().reshape({-1, nnz});
  auto crow = result.crow_indices().reshape({-1, M + 1});
  auto col = result.col_indices().reshape({-1, nnz});

  // 获取 values、crow 和 col 张量的访问器
  auto values_acc = values.accessor<scalar_t, 2>();
  auto crow_acc = crow.accessor<const index_t, 2>();
  auto col_acc = col.accessor<const index_t, 2>();

  // 使用 Vec 类型进行向量化操作
  using Vec = vec::Vectorized<scalar_t>;

  // 遍历 B 的范围
  for (const auto b : c10::irange(B)) {
    auto crow_slice = crow_acc[b];   // 获取当前 batch 中的 crow 切片
    auto col_slice = col_acc[b];     // 获取当前 batch 中的 col 切片
    auto values_slice = values_acc[b]; // 获取当前 batch 中的 values 切片
    const scalar_t* mat1_ptr = mat1_data + b * M * K; // 计算 mat1 数据指针偏移量
    const scalar_t* mat2_ptr = mat2_data + b * N * K; // 计算 mat2 数据指针偏移量

    // 并行处理稀疏矩阵的 CSR 格式
    utils::parallel_sparse_csr(crow_slice, M, nnz, [&](int64_t begin, int64_t end) {
      // 遍历当前 batch 的每一行
      for (const auto m : c10::irange(begin, end)) {
        int64_t row_start = crow_slice[m];       // 当前行在 values 和 col 中的起始位置
        int64_t row_end = crow_slice[m + 1];     // 当前行在 values 和 col 中的结束位置

        // 遍历当前行的每个非零元素
        for (const auto e : c10::irange(row_start, row_end)) {
          int64_t n = col_slice[e];         // 当前非零元素的列索引
          scalar_t val = values_slice[e];   // 当前非零元素的值
          
          // 计算 mat1_ptr[m * K] 和 mat2_ptr[n * K] 的内积
          scalar_t dot = vec::map2_reduce_all<scalar_t>(
              [](Vec x, Vec y) { return x * y; },    // 乘法运算
              [](Vec x, Vec y) { return x + y; },    // 加法运算
              mat1_ptr + m * K,                      // 第一个矩阵的起始地址
              mat2_ptr + n * K,                      // 第二个矩阵的起始地址
              K);                                    // 矩阵的宽度

          // 更新 values_slice[e] 的值
          val = alpha_ * dot + beta_ * val;
          values_slice[e] = val;
        }
      }
    });
  }
}

// 实现 sampled_addmm_sparse_csr_kernel 函数
void sampled_addmm_sparse_csr_kernel(
    const Tensor& mat1,        // 第一个稀疏张量
    const Tensor& mat2,        // 第二个稀疏张量
    const Scalar& beta,        // 乘法参数 beta
    const Scalar& alpha,       // 乘法参数 alpha
    const Tensor& result) {    // 结果张量
  const auto index_type = result.crow_indices().scalar_type();   // 获取行偏移的标量类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1.scalar_type(), "sampled_addmm_sparse_csr_kernel", [&]() {
    // 使用宏 AT_DISPATCH_INDEX_TYPES(index_type, "sampled_addmm_sparse_csr_index", [&]() { 开始索引类型的分发调度，
    // 这里将调用 sampled_addmm_sparse_csr_kernel_impl<scalar_t, index_t>(mat1, mat2, beta, alpha, result);
    // 作为该索引类型的实现函数。
    AT_DISPATCH_INDEX_TYPES(index_type, "sampled_addmm_sparse_csr_index", [&]() {
      // 在 lambda 函数中调用 sampled_addmm_sparse_csr_kernel_impl<scalar_t, index_t>，该函数用于实现稀疏矩阵乘法的加法和乘法操作。
      sampled_addmm_sparse_csr_kernel_impl<scalar_t, index_t>(mat1, mat2, beta, alpha, result);
    });
    // Lambda 表达式结束
}

} // anonymous namespace



REGISTER_DISPATCH(sampled_addmm_sparse_csr_stub, &sampled_addmm_sparse_csr_kernel);



} // at::native


注释：


// 结束了一个匿名命名空间，将其中定义的内容限制在当前文件范围内
}

// 注册一个函数调度器，将指向 sampled_addmm_sparse_csr_kernel 函数的指针与 sampled_addmm_sparse_csr_stub 绑定
REGISTER_DISPATCH(sampled_addmm_sparse_csr_stub, &sampled_addmm_sparse_csr_kernel);

} // 结束了 at::native 命名空间
```