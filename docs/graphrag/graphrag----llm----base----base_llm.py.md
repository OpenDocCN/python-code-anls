# `.\graphrag\graphrag\llm\base\base_llm.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入用于异常追踪的模块
import traceback
# 引入抽象基类和类型变量支持
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# 引入用于类型展开的扩展
from typing_extensions import Unpack

# 引入自定义类型和函数类型
from graphrag.llm.types import (
    LLM,                # 导入LLM接口类型
    ErrorHandlerFn,     # 导入错误处理函数类型
    LLMInput,           # 导入LLM输入类型
    LLMOutput,          # 导入LLM输出类型
)

# 定义类型变量
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

# LLM基类的实现，继承自抽象基类ABC和泛型类Generic
class BaseLLM(ABC, LLM[TIn, TOut], Generic[TIn, TOut]):
    """LLM Implementation class definition."""

    # 错误处理函数变量
    _on_error: ErrorHandlerFn | None

    # 设置错误处理函数
    def on_error(self, on_error: ErrorHandlerFn | None) -> None:
        """Set the error handler function."""
        self._on_error = on_error

    # 抽象方法，子类需要实现具体的LLM执行逻辑
    @abstractmethod
    async def _execute_llm(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> TOut | None:
        pass

    # 调用LLM的主入口点，根据是否需要JSON输出选择不同的调用方式
    async def __call__(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[TOut]:
        """Invoke the LLM."""
        # 判断是否需要JSON输出
        is_json = kwargs.get("json") or False
        if is_json:
            return await self._invoke_json(input, **kwargs)
        return await self._invoke(input, **kwargs)

    # 执行LLM的基本方法，捕获异常并处理
    async def _invoke(self, input: TIn, **kwargs: Unpack[LLMInput]) -> LLMOutput[TOut]:
        try:
            # 调用具体的LLM执行方法
            output = await self._execute_llm(input, **kwargs)
            # 封装执行结果到LLMOutput对象中并返回
            return LLMOutput(output=output)
        except Exception as e:
            # 获取异常堆栈信息
            stack_trace = traceback.format_exc()
            # 如果定义了错误处理函数，则调用它来处理异常
            if self._on_error:
                self._on_error(e, stack_trace, {"input": input})
            # 将异常继续抛出
            raise

    # 当需要JSON输出时调用的方法，目前未实现，抛出NotImplementedError
    async def _invoke_json(
        self, input: TIn, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        msg = "JSON output not supported by this LLM"
        raise NotImplementedError(msg)
```