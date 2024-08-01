# `.\DB-GPT-src\dbgpt\model\proxy\__init__.py`

```py
# 代理模型

# 根据给定的名称延迟导入对应的模块
def __lazy_import(name):
    # 定义模块名称到模块路径的映射
    module_path = {
        "OpenAILLMClient": "dbgpt.model.proxy.llms.chatgpt",
        "GeminiLLMClient": "dbgpt.model.proxy.llms.gemini",
        "SparkLLMClient": "dbgpt.model.proxy.llms.spark",
        "TongyiLLMClient": "dbgpt.model.proxy.llms.tongyi",
        "WenxinLLMClient": "dbgpt.model.proxy.llms.wenxin",
        "ZhipuLLMClient": "dbgpt.model.proxy.llms.zhipu",
        "YiLLMClient": "dbgpt.model.proxy.llms.yi",
        "MoonshotLLMClient": "dbgpt.model.proxy.llms.moonshot",
        "OllamaLLMClient": "dbgpt.model.proxy.llms.ollama",
        "DeepseekLLMClient": "dbgpt.model.proxy.llms.deepseek",
    }
    
    # 如果给定的名称在映射中，则导入对应的模块并返回模块中的指定名称
    if name in module_path:
        module = __import__(module_path[name], fromlist=[name])
        return getattr(module, name)
    else:
        # 如果名称不在映射中，抛出属性错误异常
        raise AttributeError(f"module {__name__} has no attribute {name}")


# 根据名称获取对应的属性
def __getattr__(name):
    # 调用延迟导入函数获取属性
    return __lazy_import(name)


# 指定可以从当前模块导出的所有符号的列表
__all__ = [
    "OpenAILLMClient",
    "GeminiLLMClient",
    "TongyiLLMClient",
    "ZhipuLLMClient",
    "WenxinLLMClient",
    "SparkLLMClient",
    "YiLLMClient",
    "MoonshotLLMClient",
    "OllamaLLMClient",
    "DeepseekLLMClient",
]
```