# `.\DB-GPT-src\dbgpt\core\awel\resource\base.py`

```py
"""Base class for resource group."""
# 导入抽象基类（ABC）和抽象方法装饰器（abstractmethod）
from abc import ABC, abstractmethod

# 定义一个抽象基类（ABC），表示资源组的基础类
class ResourceGroup(ABC):
    """Base class for resource group.

    A resource group is a group of resources that are related to each other.
    It contains the all resources that are needed to run a workflow.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of current resource group."""
        # 这是一个抽象属性方法，子类必须实现它来返回当前资源组的名称
```