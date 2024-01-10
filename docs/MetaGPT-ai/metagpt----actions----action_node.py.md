# `MetaGPT\metagpt\actions\action_node.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/11 18:45
@Author  : alexanderwu
@File    : action_node.py

NOTE: You should use typing.List instead of list to do type annotation. Because in the markdown extraction process,
  we can use typing to extract the type of the node, but we cannot use built-in list to extract.
"""
# 导入所需的模块
import json
from typing import Any, Dict, List, Optional, Tuple, Type  # 引入类型注解所需的模块

from pydantic import BaseModel, create_model, model_validator  # 导入 pydantic 模块中的类和函数
from tenacity import retry, stop_after_attempt, wait_random_exponential  # 导入 tenacity 模块中的类和函数

from metagpt.config import CONFIG  # 从 metagpt 包中导入 CONFIG 对象
from metagpt.llm import BaseLLM  # 从 metagpt 包中导入 BaseLLM 类
from metagpt.logs import logger  # 从 metagpt 包中导入 logger 对象
from metagpt.provider.postprocess.llm_output_postprocess import llm_output_postprocess  # 从 metagpt.provider.postprocess 包中导入 llm_output_postprocess 函数
from metagpt.utils.common import OutputParser, general_after_log  # 从 metagpt.utils.common 包中导入 OutputParser 和 general_after_log 函数

TAG = "CONTENT"  # 定义 TAG 常量为 "CONTENT"
LANGUAGE_CONSTRAINT = "Language: Please use the same language as Human INPUT."  # 定义 LANGUAGE_CONSTRAINT 常量
FORMAT_CONSTRAINT = f"Format: output wrapped inside [{TAG}][/{TAG}] like format example, nothing else."  # 定义 FORMAT_CONSTRAINT 常量

SIMPLE_TEMPLATE = """
## context
{context}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow instructions of nodes, generate output and make sure it follows the format example.
"""  # 定义 SIMPLE_TEMPLATE 常量为多行字符串模板

# 定义函数 dict_to_markdown，将字典转换为 Markdown 格式的字符串
def dict_to_markdown(d, prefix="- ", kv_sep="\n", postfix="\n"):
    markdown_str = ""
    for key, value in d.items():
        markdown_str += f"{prefix}{key}{kv_sep}{value}{postfix}"
    return markdown_str

```