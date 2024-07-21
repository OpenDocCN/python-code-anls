# `.\pytorch\torch\distributed\_shard\sharded_optim\__init__.py`

```
# 从 typing 模块导入必要的类型提示
from typing import Iterator, Tuple, Union

# 导入 PyTorch 中的 nn 模块
import torch.nn as nn
# 导入 ShardedTensor 类型，用于分布式训练
from torch.distributed._shard.sharded_tensor import ShardedTensor
# 导入自定义的 ShardedOptimizer 类
from .api import ShardedOptimizer

# 定义一个函数，返回一个迭代器，遍历模块中的参数以及 ShardedTensor 参数
def named_params_with_sharded_tensor(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ShardedTensor]]]:
    r"""Returns an iterator over module parameters (together with the
    ShardedTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torch.distributed._shard.sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (str, Union[Tensor, ShardedTensor]): Tuple containing
            the name and parameter (or ShardedTensor parameter)

    Example::

        >>> # xdoctest: +SKIP
        >>> model = torch.nn.Linear(*linear_size)
        >>> shard_parameter(model, "weight", spec)
        >>> for name, param in named_params_with_sharded_tensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    # 如果设置了递归，使用 named_modules 方法获取所有子模块
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    # 记录已经遍历过的 ShardedTensor，避免重复处理
    memo = set()
    for mod_prefix, mod in modules:
        # 遍历当前模块的所有属性
        for name, val in vars(mod).items():
            # 如果属性是 ShardedTensor 类型且未处理过，则加入结果迭代器
            if isinstance(val, ShardedTensor) and val not in memo:
                memo.add(val)
                # 构造完整的参数名，包括可能存在的前缀
                name = mod_prefix + ("." if mod_prefix else "") + name
                yield name, val

    # 最后遍历当前模块的所有 nn.Parameter 类型的参数，加入结果迭代器
    for name, val in module.named_parameters():
        yield name, val
```