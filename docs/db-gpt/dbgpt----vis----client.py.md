# `.\DB-GPT-src\dbgpt\vis\client.py`

```py
"""Client for vis protocol."""
# 导入必要的类型模块
from typing import Dict, Type

# 导入基础 Vis 类
from .base import Vis
# 导入各种 vis 标签相关模块
from .tags.vis_agent_message import VisAgentMessages
from .tags.vis_agent_plans import VisAgentPlans
from .tags.vis_chart import VisChart
from .tags.vis_code import VisCode
from .tags.vis_dashboard import VisDashboard
from .tags.vis_plugin import VisPlugin


class VisClient:
    """Client for vis protocol."""

    def __init__(self):
        """Client for vis protocol."""
        # 初始化一个空字典用于存储 Vis 类实例
        self._vis_tag: Dict[str, Vis] = {}

    def register(self, vis_cls: Type[Vis]):
        """Register the vis protocol."""
        # 将 vis_cls 类的实例存储在 _vis_tag 字典中，键为其 vis_tag 方法返回的字符串
        self._vis_tag[vis_cls.vis_tag()] = vis_cls()

    def get(self, tag_name):
        """Get the vis protocol by tag name."""
        # 如果给定的 tag_name 不在 _vis_tag 字典中，则抛出 ValueError 异常
        if tag_name not in self._vis_tag:
            raise ValueError(f"Vis protocol tags not yet supported！[{tag_name}]")
        # 返回对应 tag_name 的 Vis 实例
        return self._vis_tag[tag_name]

    def tag_names(self):
        """Return the tag names of the vis protocol."""
        # 返回 _vis_tag 字典的所有键，即所有注册的 vis 标签名
        return self._vis_tag.keys()


# 创建 VisClient 的实例 vis_client
vis_client = VisClient()

# 依次注册以下各个 Vis 类到 vis_client 中
vis_client.register(VisCode)
vis_client.register(VisChart)
vis_client.register(VisDashboard)
vis_client.register(VisAgentPlans)
vis_client.register(VisAgentMessages)
vis_client.register(VisPlugin)
```