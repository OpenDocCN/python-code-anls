# `.\pytorch\aten\src\ATen\native\sparse\SparseBlas.cpp`

```
// 定义宏，仅用于方法操作符断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入必要的头文件
#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseBlas.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/native/cpu/SampledAddmmKernel.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则引入常用的 ATen 函数和原生函数
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，引入特定的 ATen 操作函数
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

// 引入 C10 工具中的 MaybeOwned 类
#include <c10/util/MaybeOwned.h>

// ATen 的命名空间
namespace at::native {

// 实现稀疏矩阵向量加法的输出函数
Tensor& addmv_out_sparse_compressed(
    const Tensor& self,              // 输入张量 self
    const Tensor& mat,               // 稀疏矩阵 mat
    const Tensor& vec,               // 向量 vec
    const Scalar& beta,              // 标量 beta
    const Scalar& alpha,             // 标量 alpha
    Tensor& result) {                // 输出张量 result

  // 检查矩阵 mat 的布局，不支持 SparseBsc 布局
  TORCH_CHECK(
      mat.layout() != kSparseBsc,
      "torch.addmv: operation not supported for mat with SparseBsc layout");

  // 如果矩阵 mat 是 SparseCsc 布局，转换为 SparseCsr 后调用自身以优化处理
  if (mat.layout() == kSparseCsc) {
    // TODO: Add native CSC support to avoid this expensive conversion
    return addmv_out_sparse_compressed(
        self, mat.to_sparse_csr(), vec, beta, alpha, result);
  }

  // 断言调试模式下矩阵 mat 的布局为 SparseCsr 或 SparseBsr
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mat.layout() == kSparseCsr || mat.layout() == kSparseBsr);

  // 检查矩阵 mat 和向量 vec 的维度是否符合预期
  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  // 扩展 self 的大小以匹配 mat 的行数，并使用 MaybeOwned 类进行可能的拷贝
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});

  // 将 beta 转换为复数双精度数值
  auto betaval = beta.toComplexDouble();

  // 如果 result 不是 self，则调整 result 的大小以匹配 self，并复制 self 的值到 result
  if (&result != &self) {
    at::native::resize_output(result, self_->sizes());
    if (betaval != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  // 如果矩阵 mat 的非零元素数量为 0，处理空矩阵的特殊情况
  if (mat._nnz() == 0) {
    // 当 beta == 0 时，忽略 self 中的值；确保 NaN 和 Inf 不传播
    if (betaval == 0.0) {
      return result.zero_();  // 将 result 的值置零
    } else {
      // 否则，使用 beta 的标量张量形式进行乘法运算并赋值给 result
      return at::mul_out(
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta,
              self.scalar_type(),
              c10::nullopt /*layout*/,
              at::kCPU,
              c10::nullopt /* pin_memory */));
    }
  }

  // 使用稀疏矩阵-向量乘法的 CPU 实现来计算输出 result
  sparse::impl::cpu::addmv_out_sparse_csr(mat, vec, beta, alpha, result);

  // 返回输出张量 result 的引用
  return result;
}

} // namespace at::native
/*
  解决由稀疏三角矩阵 A 表示的系数组成的线性方程组：op(A) X = B。

  Args:
  * `B` - 大小为 m × nrhs 的密集张量。
  * `A` - 大小为 m × m 的稀疏张量。
  * `upper` - 控制计算中是使用 A 的上三角部分还是下三角部分。
  * `transpose` - 如果为 true，则 op(A) = A^T。
  * `unitriangular` - 如果为 true，则假定 A 的对角元素为一。
  * `X` - 大小为 m × nrhs 的密集张量。
  * `clone_A` - 克隆的矩阵 A，仅用于与分布式布局接口兼容性。

  返回:
  * 返回一个元组，包含 X 和 clone_A 的引用。
*/
std::tuple<Tensor&, Tensor&> triangular_solve_out_sparse_csr_cpu(
    const Tensor& B,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    Tensor& X,
    Tensor& clone_A) {
  // 调用稀疏三角求解的 CPU 实现
  sparse::impl::cpu::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
  return std::tuple<Tensor&, Tensor&>(X, clone_A);
}

/*
  计算 `result` <- α*(A @ B) * spy(C) + β*C，其中 spy(C) 是 C 的稀疏模式矩阵。

  Args:
  * `mat1` - [in] 大小为 m × k 的密集张量 A。
  * `mat2` - [in] 大小为 k × n 的密集张量 B。
  * `self` - [in] 大小为 m × n 的稀疏张量 C。
  * `result` - [out] 大小为 m × n 的稀疏张量。

  Returns:
  * 返回计算结果的稀疏张量 result 的引用。
*/
Tensor& sparse_sampled_addmm_out_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // 检查输入，确保符合稀疏加法矩阵乘法的要求
  at::native::sparse::sparse_sampled_addmm_check_inputs(self, mat1, mat2, beta, alpha, result);

  // 仅允许与 CUDA 路径相同类型的稀疏操作
  auto t = self.scalar_type();
  TORCH_CHECK(t == ScalarType::Double || t == ScalarType::Float ||
    t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble,
    "sparse_sampled_addmm: Expected self to be a floating-point or complex tensor, but got ", t);

  // 如果 result 不是 self 的引用，则调整 result 的稀疏 CSR 实现
  if (&result != &self) {
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(self._nnz(), result_sizes);
    result.copy_(self);
  }

  // 若 mat1、mat2 或 result 为空，则只乘以 beta 并返回
  if (mat1.numel() == 0 || mat2.numel() == 0 || result._nnz() == 0) {
    result.mul_(beta);
    return result;
  }

  // 将 mat2 转置为 [b, n, k]，以优化性能
  auto mat2_t = mat2.transpose(-1, -2).contiguous();
  // 调用稀疏 CSR 的加权矩阵乘法操作
  sampled_addmm_sparse_csr_stub(kCPU, mat1.contiguous(), mat2_t, beta, alpha, result);

  return result;
}


这段代码注释了两个函数，分别用于解决稀疏三角矩阵线性方程组和稀疏加权矩阵乘法运算。
    # 定义一个函数，执行稀疏矩阵乘法加法操作，并将结果存储在指定的输出张量中
    const Scalar& alpha) {
      # 创建一个空的张量，用于存储结果，维度为 {0, 0}，使用 self 张量的选项
      auto result = at::empty({0, 0}, self.options());
      # 调用本地（native）库中的函数，执行稀疏采样乘加操作，结果存储在 result 中
      at::native::sparse_sampled_addmm_out_sparse_csr_cpu(self, mat1, mat2, beta, alpha, result);
      # 返回存储结果的张量
      return result;
}

namespace sparse {

// 检查输入参数是否符合要求
void sparse_sampled_addmm_check_inputs(
    const Tensor& self,                // 第一个稀疏张量
    const Tensor& mat1,                // 第一个矩阵张量
    const Tensor& mat2,                // 第二个矩阵张量
    const Scalar& beta,                // beta 参数
    const Scalar& alpha,               // alpha 参数
    const Tensor& result) {            // 结果张量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.is_sparse_csr());  // 断言确保稀疏张量为 CSR 格式

  // 检查 mat1 和 mat2 的布局是否为 strided
  TORCH_CHECK(
      mat1.layout() == kStrided,
      "sampled_addmm: Expected mat1 to have strided layout, but got ",
      mat1.layout());
  TORCH_CHECK(
      mat2.layout() == kStrided,
      "sampled_addmm: Expected mat2 to have strided layout, but got ",
      mat2.layout());

  // 检查 result 张量是否为稀疏 CSR 格局
  TORCH_CHECK(
      result.layout() == kSparseCsr,
      "sampled_addmm: Expected result to have sparse csr layout, but got ",
      result.layout());
  
  // 检查 self 张量的 dense 维度是否为 0
  TORCH_CHECK(self.dense_dim() == 0,
      "sampled_addmm: Expected non-hybrid self tensor");
  
  // 检查 result 张量的 dense 维度是否为 0
  TORCH_CHECK(result.dense_dim() == 0,
      "sampled_addmm: Expected non-hybrid result tensor");

  // 检查 mat1 和 mat2 的数据类型是否相同
  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "sampled_addmm: Expected mat1 and mat2 to have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      mat2.scalar_type());

  // 检查 mat1 和 self 的数据类型是否相同
  TORCH_CHECK(
      mat1.scalar_type() == self.scalar_type(),
      "sampled_addmm: Expected mat1 and self to have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      self.scalar_type());

  // 检查 result 和 self 的数据类型是否相同
  TORCH_CHECK(
      result.scalar_type() == self.scalar_type(),
      "sampled_addmm: Expected result and self to have the same dtype, but got ",
      result.scalar_type(),
      " and ",
      self.scalar_type());

  // 检查 mat1 的维度是否大于等于 2
  TORCH_CHECK(
      mat1.dim() >= 2,
      "sampled_addmm: Expected mat1 to be a matrix, got ",
      mat1.dim(),
      "-D tensor");

  // 检查 mat2 的维度是否大于等于 2
  TORCH_CHECK(
      mat2.dim() >= 2,
      "sampled_addmm: Expected mat2 to be a matrix, got ",
      mat2.dim(),
      "-D tensor");

  // 检查 result 的维度是否大于等于 2
  TORCH_CHECK(
      result.dim() >= 2,
      "sampled_addmm: Expected result to be a matrix, got ",
      result.dim(),
      "-D tensor");

  // 检查 mat1 和 mat2 的批处理大小是否相同
  TORCH_CHECK(
    mat1.sizes().slice(0, mat1.dim() - 2) == mat2.sizes().slice(0, mat2.dim() - 2),
    "sampled_addmm: Expected mat1 and mat2 to have the same batch size, but got ",
    mat1.sizes().slice(0, mat1.dim() - 2),
    " and ",
    mat2.sizes().slice(0, mat2.dim() - 2));

  // 检查 self 和 mat1 的批处理大小是否相同
  TORCH_CHECK(
    !(self.dim() > 2 && self.sizes().slice(0, self.dim() - 2) != mat1.sizes().slice(0, mat1.dim() - 2)),
    "sampled_addmm: Expected self and mat1 to have the same batch size, but got ",
    self.sizes().slice(0, self.dim() - 2),
    " and ",
    // 获取 mat1 的尺寸，去掉最后两个维度
    mat1.sizes().slice(0, mat1.dim() - 2));
    
    // 获取 mat1 和 mat2 的尺寸
    IntArrayRef mat1_sizes = mat1.sizes();
    IntArrayRef mat2_sizes = mat2.sizes();
    
    // 检查 mat1 和 mat2 的倒数第二个维度是否相等，如果不相等则报错
    TORCH_CHECK(
        mat1_sizes[mat1.dim() - 1] == mat2_sizes[mat2.dim() - 2],
        "sampled_addmm: mat1 and mat2 shapes cannot be multiplied (",
        mat1_sizes[mat1.dim() - 2],
        "x",
        mat1_sizes[mat1.dim() - 1],
        " and ",
        mat2_sizes[mat2.dim() - 2],
        "x",
        mat2_sizes[mat2.dim() - 1],
        ")");
    
    // 获取 self 的尺寸
    IntArrayRef self_sizes = self.sizes();
    
    // 检查 self 的倒数第二个维度是否与 mat1 的倒数第二个维度相等，如果不相等则报错
    TORCH_CHECK(
        self_sizes[self.dim() - 2] == mat1_sizes[mat1.dim() - 2],
        "sampled_addmm: self.shape[-2] must match mat1.shape[-2]");
    
    // 检查 self 的最后一个维度是否与 mat2 的最后一个维度相等，如果不相等则报错
    TORCH_CHECK(
        self_sizes[self.dim() - 1] == mat2_sizes[mat2.dim() - 1],
        "sampled_addmm: self.shape[-1] must match mat2.shape[-1]");
}

} // namespace sparse



DEFINE_DISPATCH(sampled_addmm_sparse_csr_stub);



} // namespace at::native


注释：

} // 关闭 sparse 命名空间

DEFINE_DISPATCH(sampled_addmm_sparse_csr_stub);
// 定义一个名为 sampled_addmm_sparse_csr_stub 的调度分发函数

} // 关闭 at::native 命名空间


这段代码示例中包含了 C++ 的命名空间和函数定义。注释描述了每一行的具体作用，包括了命名空间的开启和关闭以及函数定义的目的。
```