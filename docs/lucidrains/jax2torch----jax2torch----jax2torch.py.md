# `.\lucidrains\jax2torch\jax2torch\jax2torch.py`

```
# 导入需要的库
import torch
from torch.utils import dlpack as torch_dlpack

import jax
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
from jax.tree_util import tree_map

from inspect import signature
from functools import wraps

# 定义将 JAX 数组转换为 PyTorch 张量的函数
def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

# 定义将 PyTorch 张量转换为 JAX 数组的函数
def t2j(x_torch):
    x_torch = x_torch.contiguous() # 保证张量是连续的
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

# 定义将树状结构中的 PyTorch 张量转换为 JAX 数组的函数
def tree_t2j(x_torch):
    return tree_map(lambda t: t2j(t) if isinstance(t, torch.Tensor) else t, x_torch)

# 定义将树状结构中的 JAX 数组转换为 PyTorch 张量的函数
def tree_j2t(x_jax):
    return tree_map(lambda t: j2t(t) if isinstance(t, jnp.ndarray) else t, x_jax)

# 定义装饰器，将 JAX 函数转换为 PyTorch 函数
def jax2torch(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        # 定义一个继承自 torch.autograd.Function 的类
        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                # 将输入参数转换为 JAX 数组
                args = tree_t2j(args)
                # 调用 JAX 的 vjp 函数计算函数值和梯度
                y_, ctx.fun_vjp = jax.vjp(fn, *args)
                # 将结果转换为 PyTorch 张量
                return tree_j2t(y_)

            @staticmethod
            def backward(ctx, *grad_args):
                # 将梯度参数转换为 JAX 数组
                grad_args = tree_t2j(grad_args) if len(grad_args) > 1 else t2j(grad_args[0])
                # 计算梯度
                grads = ctx.fun_vjp(grad_args)
                # 将梯度转换为 PyTorch 张量
                grads = tuple(map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads))
                return tree_j2t(grads)

        # 获取函数的参数签名
        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        # 调用 JaxFun 类的 apply 方法
        return JaxFun.apply(*bound.arguments.values())
    return inner
```