# `.\pytorch\torch\_export\pass_infra\proxy_value.py`

```py
# mypy: allow-untyped-defs
# pyre-strict
# 引入必要的模块和类型定义
from typing import Union

import torch

# 定义代理值类，用于包装数据和代理对象或节点
class ProxyValue:
    # pyre-ignore
    # 初始化方法，接受数据和代理对象或节点
    def __init__(self, data, proxy: Union[torch.fx.Proxy, torch.fx.Node]):
        # pyre-ignore
        # 存储数据
        self.data = data
        # 存储代理对象或节点
        self.proxy_or_node = proxy

    # 获取节点属性的方法，返回代理值所关联的节点对象
    @property
    def node(self) -> torch.fx.Node:
        # 检查当前代理值关联的对象类型，如果是节点则直接返回
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        # 否则断言代理值关联的是代理对象
        assert isinstance(self.proxy_or_node, torch.fx.Proxy)
        # 返回代理对象对应的节点
        return self.proxy_or_node.node

    # 获取代理对象属性的方法，返回代理值所关联的代理对象
    @property
    def proxy(self) -> torch.fx.Proxy:
        # 检查当前代理值关联的对象类型，如果不是代理对象则引发运行时错误
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        # 返回代理对象
        return self.proxy_or_node

    # 将代理值转换为张量的方法，确保数据类型为张量
    def to_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor)
        return self.data

    # 检查代理值是否为张量的方法，返回布尔值
    def is_tensor(self) -> bool:
        return isinstance(self.data, torch.Tensor)

    # pyre-ignore
    # 定义迭代器方法，允许迭代代理值中的数据
    def __iter__(self):
        yield from self.data

    # 定义布尔转换方法，判断代理值是否为真
    def __bool__(self) -> bool:
        return bool(self.data)
```