# `.\AutoGPT\autogpts\autogpt\autogpt\agents\features\context.py`

```py
from __future__ import annotations



# 导入未来版本的注解特性

from typing import TYPE_CHECKING, Any, Optional



# 导入类型检查相关的模块和类型

if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent



# 如果是类型检查模式，导入 ChatPrompt, ContextItem, BaseAgent 类

from autogpt.core.resource.model_providers import ChatMessage



# 导入 ChatMessage 类

class AgentContext:
    items: list[ContextItem]



# 定义 AgentContext 类，包含一个 ContextItem 类型的列表 items

    def __init__(self, items: Optional[list[ContextItem]] = None):
        self.items = items or []



# 初始化 AgentContext 类，如果没有传入 items 参数，则初始化为空列表

    def __bool__(self) -> bool:
        return len(self.items) > 0



# 定义 __bool__ 方法，判断 items 是否为空

    def __contains__(self, item: ContextItem) -> bool:
        return any([i.source == item.source for i in self.items])



# 定义 __contains__ 方法，判断是否包含指定的 ContextItem 对象

    def add(self, item: ContextItem) -> None:
        self.items.append(item)



# 定义 add 方法，向 items 列表中添加 ContextItem 对象

    def close(self, index: int) -> None:
        self.items.pop(index - 1)



# 定义 close 方法，根据索引从 items 列表中移除对应的 ContextItem 对象

    def clear(self) -> None:
        self.items.clear()



# 定义 clear 方法，清空 items 列表

    def format_numbered(self) -> str:
        return "\n\n".join([f"{i}. {c.fmt()}" for i, c in enumerate(self.items, 1)])



# 定义 format_numbered 方法，将 items 列表中的 ContextItem 对象格式化为带编号的字符串

class ContextMixin:
    """Mixin that adds context support to a BaseAgent subclass"""

    context: AgentContext



# 定义 ContextMixin 类，为 BaseAgent 的子类添加上下文支持

    def __init__(self, **kwargs: Any):
        self.context = AgentContext()

        super(ContextMixin, self).__init__(**kwargs)



# 初始化 ContextMixin 类，创建 AgentContext 对象并将其赋值给 context 属性

    def build_prompt(
        self,
        *args: Any,
        extra_messages: Optional[list[ChatMessage]] = None,
        **kwargs: Any,



# 定义 build_prompt 方法，构建提示信息，接受任意参数和关键字参数
    # 定义一个方法，返回ChatPrompt对象
    ) -> ChatPrompt:
        # 如果没有额外消息，则将额外消息列表设置为空列表
        if not extra_messages:
            extra_messages = []

        # 如果存在上下文信息，则将上下文信息插入到额外消息列表的第一个位置
        if self.context:
            extra_messages.insert(
                0,
                ChatMessage.system(
                    "## Context\n"
                    f"{self.context.format_numbered()}\n\n"
                    "When a context item is no longer needed and you are not done yet, "
                    "you can hide the item by specifying its number in the list above "
                    "to `hide_context_item`.",
                ),
            )

        # 调用父类的build_prompt方法，传入参数args和kwargs，同时传入额外消息列表
        return super(ContextMixin, self).build_prompt(
            *args,
            extra_messages=extra_messages,
            **kwargs,
        )  # type: ignore
# 获取代理的上下文信息，如果代理实现了ContextMixin接口，则返回其上下文信息，否则返回None
def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    # 检查代理是否实现了ContextMixin接口
    if isinstance(agent, ContextMixin):
        # 如果是，则返回代理的上下文信息
        return agent.context

    # 如果代理没有实现ContextMixin接口，则返回None
    return None
```