# `.\pytorch\torch\distributed\_shard\sharding_plan\api.py`

```py
import abc  # 导入抽象基类模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, List, Optional, Union  # 导入类型提示模块

import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._shard.sharder import Sharder  # 导入分片器类
from torch.distributed._shard.sharding_spec import ShardingSpec  # 导入分片规格类

@dataclass
class ShardingPlan:
    """
    表示分片计划的数据类，描述如何在主机之间分片一个模块。
    `plan` 用于根据提供的规格分片模块参数，
    `output_plan` 和 `return_local_tensor` 是可选的，它们用于指定使用规格时模块的输出布局，
    以及何时将其转换回数据并行模式。

    Args:
        plan (Dict[str, Union[ShardingSpec, Sharder]]):
            描述如何分片模块的字典，当前有两种分片模块的方式：
                1. 使用 `ShardingSpec` 直接分片模块参数，按参数名称键入 `ShardingSpec`。
                2. 使用 `Sharder` 在子模块上应用分片器，按模块名称键入 `Sharder` 对象。
        output_plan (Optional[Dict[str, ShardingSpec]]):
            指定产生分片张量的模块输出布局的字典，按模块名称键入 `ShardingSpec`（"" 键表示根模块）。
            默认值：`None`
        return_local_tensor (Optional[List[str]]):
            一个字符串列表，每个元素使模块的分片输出作为张量返回到本地分片，以确保在数据并行模式下进一步处理。（"" 在列表中表示根模块）。
            默认值：`None`

    Example:
      假设我们想要对一个包含两个线性层的模块进行分片，然后使用DDP运行它，我们还希望将第二个线性层的输出转换回DDP，可以按以下方式操作：

        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> class MyModule(nn.Module):
        >>>     def __init__(self):
        >>>        super().__init__()
        >>>        self.fc1 = nn.Linear()
        >>>        self.gelu = nn.GELU()
        >>>        self.fc2 = nn.Linear()
        >>>        self.relu = nn.Linear()
        >>>
        >>>     def forward(self, input):
        >>>         return self.relu(self.fc2(self.gelu(self.fc1(input))))


        >>> # xdoctest: +SKIP("Undefined spec1, spec2)
        >>> sharding_plan = ShardingPlan(
        >>>    plan={
        >>>        "fc1.weight": spec1,
        >>>        "fc2.weight": spec2
        >>>    },
        >>>    output_plan={
        >>>        "fc2": output_spec
        >>>    },
        >>>    return_local_tensor=["fc2"]
        >>> )
    """
    plan: Dict[str, Union[ShardingSpec, Sharder]]
    output_plan: Optional[Dict[str, ShardingSpec]] = None
    return_local_tensor: Optional[List[str]] = None
class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """

    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        """
        Given a nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """
        pass



# 定义一个抽象基类 ShardingPlanner，用于实现分片策略的接口
class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """

    @abc.abstractmethod
    # 定义一个抽象方法 build_plan，接受一个 nn.Module 模块作为参数，返回一个 ShardingPlan 对象
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        """
        Given a nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """
        pass
```