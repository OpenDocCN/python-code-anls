# `.\DB-GPT-src\dbgpt\agent\resource\tool\tests\test_base_tool.py`

```py
import asyncio  # 导入异步IO模块
import json  # 导入JSON处理模块
from typing import Dict, List, Optional  # 导入类型提示相关的模块

import pytest  # 导入pytest测试框架
from typing_extensions import Annotated, Doc  # 导入类型提示的扩展模块

from dbgpt._private.pydantic import BaseModel, Field  # 导入Pydantic相关模块

from ..base import BaseTool, FunctionTool, ToolParameter, tool  # 导入本地自定义模块


class TestBaseTool(BaseTool):
    @property
    def name(self):
        return "test_tool"  # 返回工具的名称

    @property
    def description(self):
        return "This is a test tool."  # 返回工具的描述信息

    @property
    def args(self):
        return {}  # 返回工具的参数字典

    def execute(self, *args, **kwargs):
        return "executed"  # 执行工具同步方法

    async def async_execute(self, *args, **kwargs):
        return "async executed"  # 执行工具异步方法


def test_base_tool():
    tool = TestBaseTool()  # 创建测试用的BaseTool实例
    assert tool.name == "test_tool"  # 断言工具名称为"test_tool"
    assert tool.description == "This is a test tool."  # 断言工具描述正确
    assert tool.execute() == "executed"  # 断言同步执行方法正确返回
    assert asyncio.run(tool.async_execute()) == "async executed"  # 使用asyncio运行异步执行方法


def test_function_tool_sync() -> None:
    def two_sum(a: int, b: int) -> int:
        """Add two numbers."""  # 定义一个函数用于求两数之和
        return a + b  # 返回两数之和

    ft = FunctionTool(name="sample", func=two_sum)  # 创建同步函数工具实例
    assert ft.execute(1, 2) == 3  # 断言工具正确执行两数之和
    with pytest.raises(ValueError):  # 使用pytest断言抛出值错误
        asyncio.run(ft.async_execute(1, 2))  # 尝试异步运行同步方法


@pytest.mark.asyncio
async def test_function_tool_async() -> None:
    async def sample_async_func(a: int, b: int) -> int:
        """Add two numbers asynchronously."""  # 异步方式定义一个函数用于求两数之和
        return a + b  # 返回两数之和

    ft = FunctionTool(name="sample_async", func=sample_async_func)  # 创建异步函数工具实例
    with pytest.raises(ValueError):  # 使用pytest断言抛出值错误
        ft.execute(1, 2)  # 尝试同步运行异步方法
    assert await ft.async_execute(1, 2) == 3  # 等待异步执行方法正确返回


@pytest.mark.asyncio
async def test_function_tool_sync_with_args() -> None:
    def two_sum(a: int, b: int) -> int:
        """Add two numbers."""  # 定义一个函数用于求两数之和
        return a + b  # 返回两数之和

    ft = FunctionTool(
        name="sample",
        func=two_sum,
        args={
            "a": {"type": "integer", "name": "a", "description": "The first number."},
            "b": {"type": "integer", "name": "b", "description": "The second number."},
        },
    )  # 创建带参数的同步函数工具实例
    ft1 = FunctionTool(
        name="sample",
        func=two_sum,
        args={
            "a": ToolParameter(
                type="integer", name="a", description="The first number."
            ),
            "b": ToolParameter(
                type="integer", name="b", description="The second number."
            ),
        },
    )  # 创建带参数类对象的同步函数工具实例
    assert ft.description == "Add two numbers."  # 断言工具描述正确
    assert ft.args.keys() == {"a", "b"}  # 断言工具参数名称正确
    assert ft.args["a"].type == "integer"  # 断言参数a类型为整数
    assert ft.args["a"].name == "a"  # 断言参数a名称为a
    assert ft.args["a"].description == "The first number."  # 断言参数a描述正确
    assert ft.args["a"].title == "A"  # 断言参数a标题为A
    dict_params = [  # 定义参数字典列表
        {
            "name": "a",
            "type": "integer",
            "description": "The first number.",
            "required": True,
        },
        {
            "name": "b",
            "type": "integer",
            "description": "The second number.",
            "required": True,
        },
    ]
    json_params = json.dumps(dict_params, ensure_ascii=False)  # 将参数字典列表转换为JSON格式
    # 定义期望的提示消息，包含关于样本API的描述和参数信息
    expected_prompt = (
        f"sample: Call this tool to interact with the sample API. What is the "
        f"sample API useful for? Add two numbers. Parameters: {json_params}"
    )
    # 断言调用 ft 对象的 get_prompt 方法返回的结果与期望的提示消息相等
    assert await ft.get_prompt() == expected_prompt
    # 断言调用 ft1 对象的 get_prompt 方法返回的结果与期望的提示消息相等
    assert await ft1.get_prompt() == expected_prompt
    # 断言调用 ft 对象的 execute 方法，传入参数 1 和 2，返回值应为 3
    assert ft.execute(1, 2) == 3
    # 使用 pytest 来确保以下代码块抛出 ValueError 异常
    with pytest.raises(ValueError):
        await ft.async_execute(1, 2)
# 定义测试函数，用于测试复杂类型的函数工具同步
def test_function_tool_sync_with_complex_types() -> None:
    # 定义一个复杂函数，使用装饰器 @tool 包裹
    @tool
    def complex_func(
        a: int,
        b: Annotated[int, Doc("The second number.")],
        c: Annotated[str, Doc("The third string.")],
        d: List[int],
        e: Annotated[Dict[str, int], Doc("A dictionary of integers.")],
        f: Optional[float] = None,
        g: str | None = None,
    ) -> int:
        """A complex function."""
        # 计算并返回各参数的复杂计算结果
        return (
            a + b + len(c) + sum(d) + sum(e.values()) + (f or 0) + (len(g) if g else 0)
        )

    # 获取函数工具对象
    ft: FunctionTool = complex_func._tool
    # 断言函数描述正确
    assert ft.description == "A complex function."
    # 断言参数名称集合正确
    assert ft.args.keys() == {"a", "b", "c", "d", "e", "f", "g"}
    # 断言参数类型及描述正确
    assert ft.args["a"].type == "integer"
    assert ft.args["a"].description == "A"
    assert ft.args["b"].type == "integer"
    assert ft.args["b"].description == "The second number."
    assert ft.args["c"].type == "string"
    assert ft.args["c"].description == "The third string."
    assert ft.args["d"].type == "array"
    assert ft.args["d"].description == "D"
    assert ft.args["e"].type == "object"
    assert ft.args["e"].description == "A dictionary of integers."
    assert ft.args["f"].type == "float"
    assert ft.args["f"].description == "F"
    assert ft.args["g"].type == "string"
    assert ft.args["g"].description == "G"


# 定义测试函数，用于测试与参数模式的函数工具同步
def test_function_tool_sync_with_args_schema() -> None:
    # 定义参数模式类
    class ArgsSchema(BaseModel):
        a: int = Field(description="The first number.")
        b: int = Field(description="The second number.")
        c: Optional[str] = Field(None, description="The third string.")
        d: List[int] = Field(description="Numbers.")

    # 定义一个复杂函数，使用装饰器 @tool 包裹，并指定参数模式为 ArgsSchema
    @tool(args_schema=ArgsSchema)
    def complex_func(a: int, b: int, c: Optional[str] = None) -> int:
        """A complex function."""
        # 计算并返回两个参数的和，如果存在第三个参数，则加上其长度
        return a + b + len(c) if c else 0

    # 获取函数工具对象
    ft: FunctionTool = complex_func._tool
    # 断言函数描述正确
    assert ft.description == "A complex function."
    # 断言参数名称集合正确
    assert ft.args.keys() == {"a", "b", "c", "d"}
    # 断言参数类型及描述正确
    assert ft.args["a"].type == "integer"
    assert ft.args["a"].description == "The first number."
    assert ft.args["b"].type == "integer"
    assert ft.args["b"].description == "The second number."
    assert ft.args["c"].type == "string"
    assert ft.args["c"].description == "The third string."
    assert ft.args["d"].type == "array"
    assert ft.args["d"].description == "Numbers."


# 定义测试函数，用于测试装饰器功能
def test_tool_decorator() -> None:
    # 定义一个简单的加法函数，使用装饰器 @tool 包裹
    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # 断言函数计算结果正确
    assert add(1, 2) == 3
    # 断言函数工具对象的名称和描述正确
    assert add._tool.name == "add"
    assert add._tool.description == "Add two numbers"


# 异步测试函数，用于测试异步函数的装饰器功能
@pytest.mark.asyncio
async def test_tool_decorator_async() -> None:
    # 定义一个异步加法函数，使用装饰器 @tool 包裹
    @tool
    async def async_add(a: int, b: int) -> int:
        """Asynchronously add two numbers."""
        return a + b

    # 断言异步函数计算结果正确
    assert await async_add(1, 2) == 3
```