# `.\pytorch\aten\src\ATen\native\sparse\SparseTensorMath.h`

```
#pragma once

# 防止头文件被重复包含，仅编译一次


#include <ATen/native/SparseTensorUtils.h>

# 包含 ATen 库中稀疏张量工具的头文件


namespace at::native {

# 进入 at::native 命名空间


TORCH_API sparse::SparseTensor& mul_out_sparse_scalar(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Scalar& value);

# 定义一个函数 mul_out_sparse_scalar，用于将稀疏张量 t 与标量值 value 相乘，并将结果存入稀疏张量 r


TORCH_API sparse::SparseTensor& mul_out_sparse_zerodim(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Tensor& value);

# 定义一个函数 mul_out_sparse_zerodim，用于将稀疏张量 t 与零维张量 value 相乘，并将结果存入稀疏张量 r


TORCH_API sparse::SparseTensor& _mul_dense_sparse_out(const Tensor& d, const Tensor& s, Tensor& res);

# 定义一个函数 _mul_dense_sparse_out，用于计算稠密张量 d 与稀疏张量 s 的乘积，并将结果存入张量 res


TORCH_API sparse::SparseTensor& _mul_sparse_sparse_zero_dim_out(const Tensor& zero_dim, const Tensor& other, Tensor& res);

# 定义一个函数 _mul_sparse_sparse_zero_dim_out，用于计算零维稀疏张量 zero_dim 与其他稀疏张量 other 的乘积，并将结果存入张量 res


TORCH_API sparse::SparseTensor& _mul_sparse_sparse_out(const Tensor& x, const Tensor& y, Tensor& res);

# 定义一个函数 _mul_sparse_sparse_out，用于计算稀疏张量 x 与稀疏张量 y 的乘积，并将结果存入张量 res


} // namespace at::native

# 结束 at::native 命名空间
```