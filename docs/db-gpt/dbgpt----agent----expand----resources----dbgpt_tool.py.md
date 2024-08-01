# `.\DB-GPT-src\dbgpt\agent\expand\resources\dbgpt_tool.py`

```py
"""Some internal tools for the DB-GPT project."""

# 导入必要的模块和库
from typing_extensions import Annotated, Doc

# 导入自定义的工具函数装饰器
from ...resource.tool.base import tool


# 使用装饰器定义一个工具函数，描述为列出 DB-GPT 项目中支持的模型
@tool(description="List the supported models in DB-GPT project.")
def list_dbgpt_support_models(
    # 定义函数参数 model_type，使用 Annotated 类型注解和 Doc 描述
    model_type: Annotated[
        str, Doc("The model type, LLM(Large Language Model) and EMBEDDING).")
    ] = "LLM",
) -> str:
    """List the supported models in dbgpt."""
    
    # 在函数内部导入模型配置信息
    from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG, LLM_MODEL_CONFIG

    # 根据 model_type 的值选择不同的支持模型列表
    if model_type.lower() == "llm":
        supports = list(LLM_MODEL_CONFIG.keys())  # 获取大语言模型配置中的所有键
    elif model_type.lower() == "embedding":
        supports = list(EMBEDDING_MODEL_CONFIG.keys())  # 获取嵌入模型配置中的所有键
    else:
        raise ValueError(f"Unsupported model type: {model_type}")  # 抛出异常，说明不支持的模型类型
    
    # 将支持的模型列表转换为以两个换行符分隔的字符串形式返回
    return "\n\n".join(supports)
```