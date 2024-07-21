# `.\pytorch\torch\utils\data\datapipes\iter\sharding.py`

```
"""
# mypy: allow-untyped-defs
从 enum 模块中导入 IntEnum 类型，用于定义枚举值
从 typing 模块中导入 Dict、Sized、Tuple 类型，用于类型提示

从 torch.utils.data.datapipes._decorator 模块中导入 functional_datapipe 装饰器
从 torch.utils.data.datapipes.datapipe 模块中导入 IterDataPipe 类型

__all__ 列表，包含导出的符号名称
"""
from enum import IntEnum
from typing import Dict, Sized, Tuple

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


class SHARDING_PRIORITIES(IntEnum):
    """
    枚举类型 SHARDING_PRIORITIES，定义了分片优先级：
    - DEFAULT 默认优先级
    - DISTRIBUTED 分布式优先级
    - MULTIPROCESSING 多进程优先级
    """
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


class _ShardingIterDataPipe(IterDataPipe):
    """
    _ShardingIterDataPipe 类，继承自 IterDataPipe 类

    定义了抽象方法 apply_sharding，用于应用分片操作
    """
    def apply_sharding(
        self,
        num_of_instances: int,
        instance_id: int,
        sharding_group: SHARDING_PRIORITIES,
    ):
        """
        抽象方法 apply_sharding，用于应用分片操作

        参数：
        - num_of_instances: 实例数量
        - instance_id: 实例 ID
        - sharding_group: 分片优先级枚举值
        """
        raise NotImplementedError


@functional_datapipe("sharding_filter")
class ShardingFilterIterDataPipe(_ShardingIterDataPipe):
    """
    ShardingFilterIterDataPipe 类，继承自 _ShardingIterDataPipe 类

    作为一个装饰器 @functional_datapipe("sharding_filter")，用于允许数据管道进行分片

    初始化方法 __init__：
    参数：
    - source_datapipe: 将被分片的可迭代数据管道
    - sharding_group_filter: 分片组过滤器，默认为 None
    """
    def __init__(self, source_datapipe: IterDataPipe, sharding_group_filter=None):
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: Dict[int, Tuple[int, int]] = {}  # 组信息字典，存储分片组信息
        self.num_of_instances = 1  # 实例数量，默认为 1
        self.instance_id = 0  # 实例 ID，默认为 0
        self._update_num_of_instances()  # 更新实例数量信息

    def apply_sharding(
        self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT
    ):
        """
        apply_sharding 方法，用于应用分片操作

        参数：
        - num_of_instances: 实例数量
        - instance_id: 实例 ID
        - sharding_group: 分片优先级枚举值，默认为 DEFAULT

        异常：
        - 如果 instance_id 大于等于 num_of_instances，则抛出 ValueError 异常
        - 如果 sharding_group 为 DEFAULT 且组信息字典中存在非 DEFAULT 组，则抛出 RuntimeError 异常
        - 如果 sharding_group 不为 DEFAULT 且组信息字典中存在 DEFAULT 组，则抛出 RuntimeError 异常
        """
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()  # 更新实例数量信息

    def _update_num_of_instances(self):
        """
        _update_num_of_instances 方法，用于更新实例数量信息

        根据分片组信息，更新实例数量和实例 ID
        """
        sorted_sharding_groups = []
        for key in sorted(self.groups.keys()):
            if self.sharding_group_filter is None or key == self.sharding_group_filter:
                sorted_sharding_groups.append(self.groups[key])

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances
    # 定义迭代器方法，使对象可迭代
    def __iter__(self):
        # 使用 enumerate 函数遍历 self.source_datapipe 中的元素及其索引
        for i, item in enumerate(self.source_datapipe):
            # 如果索引 i 能整除 self.num_of_instances，并且余数等于 self.instance_id，则返回该元素
            if i % self.num_of_instances == self.instance_id:
                yield item  # 生成器语法，返回符合条件的元素

    # 定义长度方法，返回对象的长度
    def __len__(self):
        # 检查 self.source_datapipe 是否可获取长度
        if isinstance(self.source_datapipe, Sized):
            # 返回计算得到的长度，该长度为 self.source_datapipe 的总长度除以 self.num_of_instances 向下取整，再加 1 或 0
            # 如果当前实例编号小于 self.source_datapipe 长度对 self.num_of_instances 的取模结果
            return len(self.source_datapipe) // self.num_of_instances + (
                1 if (self.instance_id < len(self.source_datapipe) % self.num_of_instances) else 0
            )
        # 如果 self.source_datapipe 不可获取长度，则抛出错误
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
```