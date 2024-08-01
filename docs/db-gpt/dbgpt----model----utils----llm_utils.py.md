# `.\DB-GPT-src\dbgpt\model\utils\llm_utils.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 导入必要的模块和函数
from pathlib import Path
from typing import Dict, List

import cachetools  # 导入缓存工具模块

# 导入配置和函数
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG, LLM_MODEL_CONFIG
from dbgpt.model.base import SupportedModel
from dbgpt.util.parameter_utils import _get_parameter_descriptions


def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@cachetools.cached(cachetools.TTLCache(maxsize=100, ttl=60 * 5))
def list_supported_models():
    """List supported models with caching."""
    # 导入内部模块
    from dbgpt.model.parameter import WorkerType

    # 获取语言模型配置和嵌入模型配置中的模型列表
    models = _list_supported_models(WorkerType.LLM.value, LLM_MODEL_CONFIG)
    models += _list_supported_models(WorkerType.TEXT2VEC.value, EMBEDDING_MODEL_CONFIG)
    return models


def _list_supported_models(
    worker_type: str, model_config: Dict[str, str]
) -> List[SupportedModel]:
    """Internal function to list supported models."""
    # 导入必要的模块和函数
    from dbgpt.model.adapter.loader import _get_model_real_path
    from dbgpt.model.adapter.model_adapter import get_llm_model_adapter

    ret = []  # 初始化返回列表
    # 遍历模型配置字典
    for model_name, model_path in model_config.items():
        # 获取真实的模型路径
        model_path = _get_model_real_path(model_name, model_path)
        # 创建 SupportedModel 对象
        model = SupportedModel(
            model=model_name,
            path=model_path,
            worker_type=worker_type,
            path_exist=False,
            proxy=False,
            enabled=False,
            params=None,
        )
        # 判断是否为代理模型
        if "proxyllm" in model_name:
            model.proxy = True
        else:
            # 检查模型路径是否存在
            path = Path(model_path)
            model.path_exist = path.exists()
        param_cls = None
        try:
            # 获取语言模型适配器并获取参数类
            llm_adapter = get_llm_model_adapter(model_name, model_path)
            param_cls = llm_adapter.model_param_class()
            model.enabled = True
            # 获取模型参数描述
            params = _get_parameter_descriptions(
                param_cls, model_name=model_name, model_path=model_path
            )
            model.params = params
        except Exception:
            pass
        # 将模型对象添加到返回列表中
        ret.append(model)
    return ret  # 返回模型列表
```