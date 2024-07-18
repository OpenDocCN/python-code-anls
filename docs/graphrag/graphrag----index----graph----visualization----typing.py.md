# `.\graphrag\graphrag\index\graph\visualization\typing.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入 dataclasses 模块中的 dataclass 装饰器
from dataclasses import dataclass

# 定义一个数据类 NodePosition，用于表示节点位置信息
@dataclass
class NodePosition:
    """Node position class definition."""
    
    # 节点标签
    label: str
    # 所属集群
    cluster: str
    # 大小
    size: float
    
    # x 坐标
    x: float
    # y 坐标
    y: float
    # 可选的 z 坐标，默认为 None
    z: float | None = None

    # 将对象转换为 pandas 格式的元组的方法定义
    def to_pandas(self) -> tuple[str, float, float, str, float]:
        """To pandas method definition."""
        return self.label, self.x, self.y, self.cluster, self.size

# 定义一个别名类型 GraphLayout，表示由 NodePosition 对象组成的列表
GraphLayout = list[NodePosition]
```