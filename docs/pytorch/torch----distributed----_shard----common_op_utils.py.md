# `.\pytorch\torch\distributed\_shard\common_op_utils.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类型声明
from typing import Optional

import torch
from torch.utils import _pytree as pytree


def _basic_validation(op, args=(), kwargs=None):
    """
    Common validation across all ops go in here.
    """
    # 导入特定的分布式张量类型
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    # 检查是否没有输入参数
    if len(args) == 0 and (kwargs is None or len(kwargs) == 0):
        raise ValueError(f" No input for '{op.__name__}'!")

    # 验证参数的类型
    has_distributed_tensor = False

    def is_distributed_tensor(e):
        nonlocal has_distributed_tensor
        # 检查是否是分布式张量
        if isinstance(e, ShardedTensor):
            has_distributed_tensor = True

    # 在参数列表和关键字参数中进行递归映射，检查是否存在分布式张量
    pytree.tree_map_(is_distributed_tensor, args)
    pytree.tree_map_(is_distributed_tensor, kwargs)

    # 如果没有分布式张量，则抛出类型错误
    if not has_distributed_tensor:
        raise TypeError(
            f"torch function '{op.__name__}', with args: {args} and "
            f"kwargs: {kwargs} are called without any distributed tensor!"
        )

    # 验证所有分布式张量是否使用相同的过程组
    cur_pg: Optional[torch.distributed.ProcessGroup] = None

    def validate_pg(e):
        nonlocal cur_pg
        # 检查是否是分布式张量，并验证过程组是否相同
        if isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError(
                    "All distributed tensors should use the "
                    "same ProcessGroup if used together in an op."
                )
            cur_pg = e._process_group

    # 在参数列表和关键字参数中进行递归映射，验证过程组
    pytree.tree_map_(validate_pg, args)
    pytree.tree_map_(validate_pg, kwargs)


def _register_default_op(op, decorator):
    # 定义默认操作的注册函数
    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunctionSubclass context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        # 如果没有传入关键字参数，设为一个空字典
        if kwargs is None:
            kwargs = {}

        # 使用 DisableTorchFunctionSubclass 上下文，将操作委托给真正的操作调用
        with torch._C.DisableTorchFunctionSubclass():
            return op(*args, **kwargs)
```