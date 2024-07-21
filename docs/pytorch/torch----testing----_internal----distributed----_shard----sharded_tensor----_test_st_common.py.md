# `.\pytorch\torch\testing\_internal\distributed\_shard\sharded_tensor\_test_st_common.py`

```py
# 忽略类型检查错误，这通常用于类型检查工具
# 导入所需的库和模块
import copy  # 导入深拷贝模块，用于复制对象
import random  # 导入随机数模块，用于生成随机数
import torch  # 导入PyTorch库
from torch.distributed._shard import sharded_tensor  # 导入分布式张量模块

from torch.distributed._shard.sharding_spec import (  # 从分片规范模块导入
    ChunkShardingSpec,  # 导入块分片规范类
)

# 定义GPU放置策略列表
PLACEMENTS = [
    "rank:0/cuda:0",
    "rank:1/cuda:1",
    "rank:2/cuda:2",
    "rank:3/cuda:3",
]

# 默认的GPU数量
DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0):
    """
    生成用于测试的块分片规范列表。

    Args:
        sharding_dims (list): 分片维度列表
        seed (int): 随机种子，默认为0

    Returns:
        list: 块分片规范对象列表
    """
    # 初始化空的规范列表
    spec_list = []
    # 对每个分片维度进行循环处理
    for i in range(len(sharding_dims)):
        # 使用不同的随机种子对GPU放置策略进行洗牌
        random.Random(seed + i).shuffle(PLACEMENTS)
        # 创建并添加块分片规范对象到列表中
        spec_list.append(
            ChunkShardingSpec(
                dim=sharding_dims[i],
                placements=copy.deepcopy(PLACEMENTS),
            )
        )
    return spec_list


class MyShardedModel2(torch.nn.Module):
    def __init__(
        self,
        spec=None,
        group=None,
        init_rrefs=True
    ) -> None:
        """
        初始化函数，用于创建分片模型2的实例。

        Args:
            spec (ChunkShardingSpec or None): 分片规范对象或None
            group (ProcessGroup or None): 进程组对象或None
            init_rrefs (bool): 是否初始化RRef，默认为True
        """
        super().__init__()
        # 如果提供了分片规范对象，则创建相应的分片张量
        if spec is not None:
            self.sharded_tensor2 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor2 = None
        # 创建一个随机张量参数
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
    def __init__(
        self,
        spec=None,
        group=None,
        init_rrefs=True
    ) -> None:
        """
        初始化函数，用于创建分片模型1的实例。

        Args:
            spec (ChunkShardingSpec or None): 分片规范对象或None
            group (ProcessGroup or None): 进程组对象或None
            init_rrefs (bool): 是否初始化RRef，默认为True
        """
        super().__init__()
        # 如果提供了分片规范对象，则创建相应的分片张量
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor1 = None
        # 创建一个随机张量参数
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        # 创建分片模型2的实例作为子模块
        self.submodule = MyShardedModel2(spec, group, init_rrefs)
```