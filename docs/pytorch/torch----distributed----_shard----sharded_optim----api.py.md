# `.\pytorch\torch\distributed\_shard\sharded_optim\api.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类型声明
from typing import Any, Dict, List, Mapping, Union

# 导入 PyTorch 中的优化器和张量类
import torch.optim as optim
from torch import Tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor


# 定义 ShardedOptimizer 类，继承自 optim.Optimizer
class ShardedOptimizer(optim.Optimizer):
    def __init__(
        self,
        named_params: Mapping[str, Union[Tensor, ShardedTensor]],
        optimizer_class,  # 传入的优化器类
        *optimizer_args,  # 优化器的位置参数
        **optimizer_kwargs,  # 优化器的关键字参数
    ):
        """
        ShardedOptimizer collects all tensors and local shard tensors of
        ShardedTensor, then use these tensors as ``params`` for optimizers

        Args:
            named_params (Dict[str, Union[Tensor, ShardedTensor]]) : a Dict
                of parameters, where key is the parameter key, value is either
                Tensor or ShardedTensor parameter.
            optimizer_class (torch.optim.Optimizer): the Optimizer to use
                locally, i.e. torch.optim.SGD, torch.optim.Adagrad, etc.
            *optimizer_args: the arguments to initialize the optimizer.
            **optimizer_kwargs: the key-word arguments to initialize the optimizer.

        """
        # 初始化函数，收集所有张量和 ShardedTensor 的本地片段作为优化器的参数
        tensors: List[Tensor] = []
        for value in named_params.values():
            if isinstance(value, ShardedTensor):
                for local_shard in value.local_shards():
                    tensors.append(local_shard.tensor)
            else:
                tensors.append(value)

        # 保存传入的参数和初始化优化器实例
        self.named_params = named_params
        self._optim = optimizer_class(tensors, *optimizer_args, **optimizer_kwargs)
        self.param_groups = self._optim.param_groups  # 继承优化器的参数组信息
        self.state = self._optim.state  # 继承优化器的状态信息

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        # 重置所有优化的张量的梯度
        self._optim.zero_grad(set_to_none)
    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        # 调用底层优化器的step方法，执行一次优化步骤
        self._optim.step(closure)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices like torch.optim.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.
        """
        # TODO: 实现state_dict方法
        raise NotImplementedError("ShardedOptimizer state_dict not implemented yet!")

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        r"""Loads the ShardedOptimizer state.

        Args:
            state_dict (dict): ShardedOptimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # TODO: 实现load_state_dict方法
        raise NotImplementedError(
            "ShardedOptimizer load_state_dict not implemented yet!"
        )

    def add_param_group(self, param_group: Any):
        r"""Add a new param group"""
        # TODO: 实现add_param_group方法
        raise NotImplementedError(
            "ShardedOptimizer add_param_group not implemented yet!"
        )
```