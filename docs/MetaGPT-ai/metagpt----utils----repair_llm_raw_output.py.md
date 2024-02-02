# `MetaGPT\metagpt\utils\repair_llm_raw_output.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : repair llm raw output with particular conditions

import copy  # 导入copy模块，用于复制对象
from enum import Enum  # 导入Enum枚举类
from typing import Callable, Union  # 导入类型提示相关的模块

import regex as re  # 导入regex模块，用于正则表达式操作
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed  # 导入tenacity模块，用于重试操作

from metagpt.config import CONFIG  # 导入CONFIG配置
from metagpt.logs import logger  # 导入logger日志模块
from metagpt.utils.custom_decoder import CustomDecoder  # 导入自定义解码器模块


# 定义修复类型的枚举类
class RepairType(Enum):
    CS = "case sensitivity"  # 大小写敏感
    RKPM = "required key pair missing"  # 缺少必需的键对
    SCM = "special character missing"  # 缺少特殊字符
    JSON = "json format"  # JSON格式修复


# 修复大小写敏感的函数
def repair_case_sensitivity(output: str, req_key: str) -> str:
    # 修复大小写敏感的情况
    # ...
    return output


# 修复缺少特殊字符的函数
def repair_special_character_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    # 修复缺少特殊字符的情况
    # ...
    return output


# 修复缺少必需键对的函数
def repair_required_key_pair_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    # 修复缺少必需键对的情况
    # ...
    return output


# 修复JSON格式的函数
def repair_json_format(output: str) -> str:
    # 修复JSON格式的情况
    # ...
    return output


# 修复LLM原始输出的函数
def _repair_llm_raw_output(output: str, req_key: str, repair_type: RepairType = None) -> str:
    # 修复LLM原始输出的情况
    # ...
    return output


# 修复LLM原始输出的函数（包含多个修复类型）
def repair_llm_raw_output(output: str, req_keys: list[str], repair_type: RepairType = None) -> str:
    # 修复LLM原始输出的情况
    # ...
    return output


# 修复无效JSON的函数
def repair_invalid_json(output: str, error: str) -> str:
    # 修复无效JSON的情况
    # ...
    return output


# 运行后执行并传递到下一个重试的函数
def run_after_exp_and_passon_next_retry(logger: "loguru.Logger") -> Callable[["RetryCallState"], None]:
    # 运行后执行并传递到下一个重试的情况
    # ...
    return run_and_passon


# 重试解析JSON文本的函数
@retry(
    stop=stop_after_attempt(3 if CONFIG.repair_llm_output else 0),
    wait=wait_fixed(1),
    after=run_after_exp_and_passon_next_retry(logger),
)
def retry_parse_json_text(output: str) -> Union[list, dict]:
    # 重试解析JSON文本的情况
    # ...
    return parsed_data


# 从输出中提取内容的函数
def extract_content_from_output(content: str, right_key: str = "[/CONTENT]"):
    # 从输出中提取内容的情况
    # ...
    return new_content


# 从输出中提取状态值的函数
def extract_state_value_from_output(content: str) -> str:
    # 从输出中提取状态值的情况
    # ...
    return state

```