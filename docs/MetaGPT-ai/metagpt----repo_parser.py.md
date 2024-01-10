# `MetaGPT\metagpt\repo_parser.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/17 17:58
@Author  : alexanderwu
@File    : repo_parser.py
"""
# 导入必要的模块
from __future__ import annotations  # 导入未来的注解特性
import ast  # 导入抽象语法树模块
import json  # 导入 JSON 模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程模块
from pathlib import Path  # 导入路径模块
from typing import Dict, List, Optional  # 导入类型提示模块

import pandas as pd  # 导入 pandas 模块
from pydantic import BaseModel, Field  # 导入 pydantic 模块中的基本模型和字段
# 导入自定义常量
from metagpt.const import AGGREGATION, COMPOSITION, GENERALIZATION
# 导入日志模块
from metagpt.logs import logger
# 导入通用工具模块
from metagpt.utils.common import any_to_str, aread
# 导入异常处理模块
from metagpt.utils.exceptions import handle_exception


# 定义文件信息的基本模型
class RepoFileInfo(BaseModel):
    file: str
    classes: List = Field(default_factory=list)
    functions: List = Field(default_factory=list)
    globals: List = Field(default_factory=list)
    page_info: List = Field(default_factory=list)


# 定义代码块信息的基本模型
class CodeBlockInfo(BaseModel):
    lineno: int
    end_lineno: int
    type_name: str
    tokens: List = Field(default_factory=list)
    properties: Dict = Field(default_factory=dict)


# 定义类信息的基本模型
class ClassInfo(BaseModel):
    name: str
    package: Optional[str] = None
    attributes: Dict[str, str] = Field(default_factory=dict)
    methods: Dict[str, str] = Field(default_factory=dict)


# 定义类关系的基本模型
class ClassRelationship(BaseModel):
    src: str = ""
    dest: str = ""
    relationship: str = ""
    label: Optional[str] = None


# 判断节点是否为函数
def is_func(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

```