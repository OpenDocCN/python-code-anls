# `.\pytorch\aten\src\ATen\native\cuda\ReduceOps.h`

```py
# 声明命名空间 at 下的 TensorIterator 结构体
namespace at {
    struct TensorIterator;
}

# 声明命名空间 c10 下的 Scalar 类
namespace c10 {
    class Scalar;
}

# 声明命名空间 at::native 下的函数原型

# 启动核函数，用于计算张量迭代器 iter 中每个元素的范数
void norm_launch_kernel(TensorIterator &iter, double val);

# 启动核函数，用于计算张量迭代器 iter 中的最小值
void min_launch_kernel(TensorIterator &iter);

# 启动核函数，用于计算张量迭代器 iter 中的最大值
void max_launch_kernel(TensorIterator &iter);

# 启动核函数，用于计算张量迭代器 iter 中的最小值和最大值
void aminmax_launch_kernel(TensorIterator &iter);

# 启动核函数，用于计算张量迭代器 iter 中的所有最小值
void min_all_launch_kernel(TensorIterator &iter);

# 启动核函数，用于计算张量迭代器 iter 中的所有最大值
void max_all_launch_kernel(TensorIterator &iter);

# 启动核函数，用于计算张量迭代器 iter 中所有最小值和最大值的归约
void aminmax_allreduce_launch_kernel(TensorIterator &iter);
```