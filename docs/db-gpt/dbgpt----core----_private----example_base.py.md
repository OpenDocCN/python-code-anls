# `.\DB-GPT-src\dbgpt\core\_private\example_base.py`

```py
"""Example selector base class"""

# 引入抽象基类和枚举类型
from abc import ABC
from enum import Enum
from typing import List, Optional

# 引入数据模型基类
from dbgpt._private.pydantic import BaseModel


class ExampleType(Enum):
    """Example type"""

    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"


class ExampleSelector(BaseModel, ABC):
    """Example selector base class"""

    # examples_record 用于存储示例数据的列表
    examples_record: List[dict]
    # use_example 表示是否使用示例，默认为 False
    use_example: bool = False
    # type 表示示例类型，默认为 ONE_SHOT
    type: str = ExampleType.ONE_SHOT.value

    def examples(self, count: int = 2):
        """Return examples"""
        # 根据示例类型选择不同的上下文生成方法
        if ExampleType.ONE_SHOT.value == self.type:
            return self.__one_shot_context()
        else:
            return self.__few_shot_context(count)

    def __few_shot_context(self, count: int = 2) -> Optional[List[dict]]:
        """
        Use 2 or more examples, default 2
        Returns: example text
        """
        # 如果 use_example 为真，则返回指定数量的示例文本
        if self.use_example:
            need_use = self.examples_record[:count]
            return need_use
        return None

    def __one_shot_context(self) -> Optional[dict]:
        """
         Use one examples
        Returns:

        """
        # 如果 use_example 为真，则返回最后一个示例数据
        if self.use_example:
            need_use = self.examples_record[-1]
            return need_use

        return None
```