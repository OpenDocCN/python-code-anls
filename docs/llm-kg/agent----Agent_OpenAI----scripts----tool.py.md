# `.\agent\Agent_OpenAI\scripts\tool.py`

```
import colorama
import json
from colorama import Fore
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import OpenAI
from pydantic.v1 import BaseModel
from datetime import datetime
from typing import Type, Callable

# 定义一个基础工具结果的数据模型，包含内容和成功状态
class ToolResult(BaseModel):
    content: str
    success: bool

# 定义一个工具类，包含名称、数据模型、执行函数及验证缺失值的标志
class Tool(BaseModel):
    name: str                     # 工具名称
    model: Type[BaseModel]        # 数据模型类型
    function: Callable            # 可调用函数
    validate_missing: bool = True # 是否验证缺失值，默认为True

    class Config:
        arbitrary_types_allowed = True

    # 执行工具功能，接收关键字参数，返回ToolResult对象
    def run(self, **kwargs) -> ToolResult:
        # 如果需要验证缺失值
        if self.validate_missing:
            # 调用验证输入函数，检查缺失值
            missing_values = self.validate_input(**kwargs)
            # 如果有缺失值，返回带有缺失值信息的ToolResult对象
            if missing_values:
                content = f"Missing values: {', '.join(missing_values)}"
                return ToolResult(content=content, success=False)

        # 执行工具的主要功能，调用指定的函数，并将结果转换为字符串
        result = self.function(**kwargs)
        return ToolResult(content=str(result), success=True)

    # 验证输入参数，检查是否缺少必要的参数，返回缺失的参数名列表
    def validate_input(self, **kwargs):
        missing_values = []

        # 遍历数据模型的注解，检查是否所有的键都在参数中
        for key, value in self.model.__annotations__.items():
            if key not in kwargs:
                missing_values.append(key)

        return missing_values

    # 属性装饰器，生成OpenAI工具的模式数据
    @property
    def openai_tool_schema(self):
        # 调用函数将数据模型转换为OpenAI工具的模式
        schema = convert_to_openai_tool(self.model)
        # 设置函数名
        schema["function"]["name"] = self.name

        # 如果模式中的参数有"required"字段，则删除该字段
        if schema["function"]["parameters"].get("required"):
            del schema["function"]["parameters"]["required"]
        
        return schema
```