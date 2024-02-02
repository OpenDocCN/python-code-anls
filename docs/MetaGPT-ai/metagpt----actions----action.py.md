# `MetaGPT\metagpt\actions\action.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : action.py
"""

# 导入必要的模块和类
from __future__ import annotations
from typing import Optional, Union
from pydantic import ConfigDict, Field, model_validator
from metagpt.actions.action_node import ActionNode
from metagpt.llm import LLM
from metagpt.provider.base_llm import BaseLLM
from metagpt.schema import (
    CodeSummarizeContext,
    CodingContext,
    RunCodeContext,
    SerializationMixin,
    TestingContext,
)

# 定义 Action 类
class Action(SerializationMixin, is_polymorphic_base=True):
    # 定义模型配置
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude=["llm"])

    # 定义属性
    name: str = ""
    llm: BaseLLM = Field(default_factory=LLM, exclude=True)
    context: Union[dict, CodingContext, CodeSummarizeContext, TestingContext, RunCodeContext, str, None] = ""
    prefix: str = ""  # aask*时会加上prefix，作为system_message
    desc: str = ""  # for skill manager
    node: ActionNode = Field(default=None, exclude=True)

    # 模型验证器，设置 name 属性如果为空
    @model_validator(mode="before")
    def set_name_if_empty(cls, values):
        if "name" not in values or not values["name"]:
            values["name"] = cls.__name__
        return values

    # 模型验证器，根据指令初始化节点
    @model_validator(mode="before")
    def _init_with_instruction(cls, values):
        if "instruction" in values:
            name = values["name"]
            i = values["instruction"]
            values["node"] = ActionNode(key=name, expected_type=str, instruction=i, example="", schema="raw")
        return values

    # 设置前缀的方法
    def set_prefix(self, prefix):
        """Set prefix for later usage"""
        self.prefix = prefix
        self.llm.system_prompt = prefix
        if self.node:
            self.node.llm = self.llm
        return self

    # 返回类名的字符串表示
    def __str__(self):
        return self.__class__.__name__

    # 返回类的字符串表示
    def __repr__(self):
        return self.__str__()

    # 异步方法，用于询问用户输入
    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """Append default prefix"""
        return await self.llm.aask(prompt, system_msgs)

    # 异步方法，运行动作节点
    async def _run_action_node(self, *args, **kwargs):
        """Run action node"""
        msgs = args[0]
        context = "## History Messages\n"
        context += "\n".join([f"{idx}: {i}" for idx, i in enumerate(reversed(msgs))])
        return await self.node.fill(context=context, llm=self.llm)

    # 异步方法，运行动作
    async def run(self, *args, **kwargs):
        """Run action"""
        if self.node:
            return await self._run_action_node(*args, **kwargs)
        raise NotImplementedError("The run method should be implemented in a subclass.")

```