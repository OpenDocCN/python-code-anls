# `.\DB-GPT-src\dbgpt\core\interface\operators\retriever.py`

```py
"""The Abstract Retriever Operator."""
# 导入抽象方法装饰器
from abc import abstractmethod

# 导入父类 MapOperator 和相关模块
from dbgpt.core.awel import MapOperator
from dbgpt.core.awel.task.base import IN, OUT


class RetrieverOperator(MapOperator[IN, OUT]):
    """The Abstract Retriever Operator."""

    async def map(self, input_value: IN) -> OUT:
        """Map input value to output value.

        Args:
            input_value (IN): The input value.

        Returns:
            OUT: The output value.
        """
        # 将阻塞函数 retrieve 包装成异步函数
        return await self.blocking_func_to_async(self.retrieve, input_value)

    @abstractmethod
    def retrieve(self, input_value: IN) -> OUT:
        """Retrieve data for input value."""
```