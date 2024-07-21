# `.\pytorch\functorch\notebooks\_src\plot_jacobians_and_hessians.py`

```py
"""
=============================
Jacobians, hessians, and more
=============================

Computing jacobians or hessians are useful in a number of non-traditional
deep learning models. It is difficult (or annoying) to compute these quantities
efficiently using a standard autodiff system like PyTorch Autograd; functorch
provides ways of computing various higher-order autodiff quantities efficiently.
"""
# 导入 functools 模块中的 partial 函数
from functools import partial

# 导入 PyTorch 库
import torch
import torch.nn.functional as F

# 设置随机种子，以保证结果的可重复性
torch.manual_seed(0)


######################################################################
# Setup: Comparing functorch vs the naive approach
# --------------------------------------------------------------------
# 定义一个需要计算其 Jacobian 矩阵的函数
# 这是一个简单的线性函数，带有非线性激活函数
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()


# 创建一些虚拟数据: 权重、偏置和特征向量
D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)

# 将 predict 函数视为将输入 x 从 R^D 映射到 R^D 的函数
# PyTorch Autograd 只能计算向量-Jacobian积，为了计算这个 R^D -> R^D 函数的完整 Jacobian 矩阵
# 我们需要逐行计算，每次使用一个不同的单位向量
xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)


def compute_jac(xp):
    # 使用 PyTorch Autograd 逐行计算 Jacobian 矩阵
    jacobian_rows = [
        torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
        for vec in unit_vectors
    ]
    return torch.stack(jacobian_rows)


# 计算 Jacobian 矩阵
jacobian = compute_jac(xp)

# 不再逐行计算 Jacobian，而是使用 ``vmap`` 函数来消除循环，并向量化计算
# 不能直接将 ``vmap`` 应用于 PyTorch Autograd，因此使用 functorch 提供的 ``vjp`` 变换
from functorch import vjp, vmap

_, vjp_fn = vjp(partial(predict, weight, bias), x)
(ft_jacobian,) = vmap(vjp_fn)(unit_vectors)
assert torch.allclose(ft_jacobian, jacobian)

# 在另一个教程中，反向模式自动微分和 vmap 的组合为我们提供了每样本梯度。
# 在本教程中，组合反向模式自动微分和 vmap 提供了 Jacobian 计算！
# vmap 和自动微分变换的各种组合可以给我们不同的有趣量化值。
#
# functorch 提供了 ``jacrev`` 作为一个便捷函数，执行 vmap-vjp 组合来计算 Jacobian 矩阵。
# ``jacrev`` 接受一个 argnums 参数，指定我们要相对于哪个参数计算 Jacobian 矩阵。
from functorch import jacrev

# 使用 functorch 的 jacrev 函数计算 Jacobian 矩阵
ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)
assert torch.allclose(ft_jacobian, jacobian)

# 比较两种计算 Jacobian 矩阵的性能。
# functorch 版本要快得多（并且随着输出量的增加而变得更快）。
# 通常情况下，我们期望通过 ``vmap`` 实现向量化，可以帮助消除开销，并更好地利用硬件资源。
from torch.utils.benchmark import Timer
# 导入性能基准模块Timer

without_vmap = Timer(stmt="compute_jac(xp)", globals=globals())
# 创建不使用vmap的性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
with_vmap = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
# 创建使用vmap的性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
print(without_vmap.timeit(500))
# 执行不使用vmap的计时器500次，并打印结果
print(with_vmap.timeit(500))
# 执行使用vmap的计时器500次，并打印结果

# It's pretty easy to flip the problem around and say we want to compute
# Jacobians of the parameters to our model (weight, bias) instead of the input.
# 计算模型参数（weight, bias）的雅可比矩阵而不是输入的雅可比矩阵。

ft_jac_weight, ft_jac_bias = jacrev(predict, argnums=(0, 1))(weight, bias, x)
# 使用jacrev计算predict函数关于参数weight和bias的雅可比矩阵，将结果分别赋给ft_jac_weight和ft_jac_bias变量

######################################################################
# reverse-mode Jacobian (jacrev) vs forward-mode Jacobian (jacfwd)
# --------------------------------------------------------------------
# We offer two APIs to compute jacobians: jacrev and jacfwd:
# - jacrev uses reverse-mode AD. As you saw above it is a composition of our
#   vjp and vmap transforms.
# - jacfwd uses forward-mode AD. It is implemented as a composition of our
#   jvp and vmap transforms.
# jacfwd and jacrev can be subsituted for each other and have different
# performance characteristics.
#
# As a general rule of thumb, if you're computing the jacobian of an R^N -> R^M
# function, if there are many more outputs than inputs (i.e. M > N) then jacfwd is
# preferred, otherwise use jacrev. There are exceptions to this rule, but a
# non-rigorous argument for this follows:
#
# In reverse-mode AD, we are computing the jacobian row-by-row, while in
# forward-mode AD (which computes Jacobian-vector products), we are computing
# it column-by-column. The Jacobian matrix has M rows and N columns.
from functorch import jacfwd, jacrev
# 从functorch模块中导入jacfwd和jacrev函数

# Benchmark with more inputs than outputs
Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)
# 定义输入输出维度，生成随机权重、偏置和输入数据

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
# 创建使用jacfwd计算性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
# 创建使用jacrev计算性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
print(f"jacfwd time: {using_fwd.timeit(500)}")
# 执行使用jacfwd的计时器500次，并打印结果
print(f"jacrev time: {using_bwd.timeit(500)}")
# 执行使用jacrev的计时器500次，并打印结果

# Benchmark with more outputs than inputs
Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)
# 定义输入输出维度，生成随机权重、偏置和输入数据

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
# 创建使用jacfwd计算性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
# 创建使用jacrev计算性能计时器，stmt指定执行的语句，globals=globals()用于访问全局变量
print(f"jacfwd time: {using_fwd.timeit(500)}")
# 执行使用jacfwd的计时器500次，并打印结果
print(f"jacrev time: {using_bwd.timeit(500)}")
# 执行使用jacrev的计时器500次，并打印结果

######################################################################
# Hessian computation with functorch.hessian
# --------------------------------------------------------------------
# We offer a convenience API to compute hessians: functorch.hessian.
# Hessians are the jacobian of the jacobian, which suggests that one can just
# compose functorch's jacobian transforms to compute one.
# Indeed, under the hood, ``hessian(f)`` is simply ``jacfwd(jacrev(f))``
#
# Depending on your model, you may want to use ``jacfwd(jacfwd(f))`` or
# functorch.hessian, depending on your needs.
# 导入functorch库中的hessian函数，用于计算函数的Hessian矩阵
from functorch import hessian

# 使用双向自动微分计算函数predict关于其第二个参数的二阶Jacobi矩阵，并传入weight、bias、x作为参数
# hess0 = hessian(predict, argnums=2)(weight, bias, x)

# 使用双向自动微分两次计算函数predict关于其第二个参数的二阶前向Jacobi矩阵，并传入weight、bias、x作为参数
# hess1 = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)

# 使用双向自动微分两次计算函数predict关于其第二个参数的二阶反向Jacobi矩阵，并传入weight、bias、x作为参数
hess2 = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

######################################################################
# Batch Jacobian (and Batch Hessian)
# --------------------------------------------------------------------
# 在上面的示例中，我们操作的是单个特征向量。
# 在某些情况下，您可能希望对一批输出关于一批输入计算Jacobi矩阵，
# 其中每个输入产生独立的输出。也就是说，给定形状为(B, N)的输入批次和
# 一个从(B, N)到(B, M)的函数，我们希望得到形状为(B, M, N)的Jacobi矩阵。
# 最简单的方法是在批次维度上求和，然后计算该函数的Jacobi矩阵。

def predict_with_output_summed(weight, bias, x):
    # 返回predict函数关于weight、bias、x的预测值之和
    return predict(weight, bias, x).sum(0)

batch_size = 64
Din = 31
Dout = 33
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(batch_size, Din)

# 使用反向自动微分计算函数predict_with_output_summed关于其第二个参数的Jacobi矩阵，并传入weight、bias、x作为参数
batch_jacobian0 = jacrev(predict_with_output_summed, argnums=2)(weight, bias, x)

# 如果您有一个从R^N到R^M的函数，但输入是批处理的，则可以将vmap与jacrev结合使用来计算批处理Jacobi矩阵
compute_batch_jacobian = vmap(jacrev(predict, argnums=2), in_dims=(None, None, 0))
# 使用vmap计算predict函数关于其第二个参数的批处理Jacobi矩阵，并传入weight、bias、x作为参数
batch_jacobian1 = compute_batch_jacobian(weight, bias, x)
# 断言两个批处理Jacobi矩阵是否接近（数值上相等）
assert torch.allclose(batch_jacobian0, batch_jacobian1)

# 最后，批处理Hessian矩阵的计算方式类似。使用vmap来批处理Hessian计算是最简单的方法，
# 但在某些情况下，求和技巧也有效。
compute_batch_hessian = vmap(hessian(predict, argnums=2), in_dims=(None, None, 0))
# 使用vmap计算predict函数关于其第二个参数的批处理Hessian矩阵，并传入weight、bias、x作为参数
batch_hess = compute_batch_hessian(weight, bias, x)
```