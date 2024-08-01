# `.\DB-GPT-src\dbgpt\model\__init__.py`

```py
# 尝试导入调试模块中的客户端类：DefaultLLMClient 和 RemoteLLMClient
try:
    from dbgpt.model.cluster.client import DefaultLLMClient, RemoteLLMClient
# 如果导入失败，记录导入异常
except ImportError as exc:
    # 设置 DefaultLLMClient 和 RemoteLLMClient 为 None
    DefaultLLMClient = None
    RemoteLLMClient = None

# 定义一个空列表，用于存储需要导出的符号（类、函数等）
_exports = []

# 如果 DefaultLLMClient 存在（即导入成功），将其添加到导出列表中
if DefaultLLMClient:
    _exports.append("DefaultLLMClient")
# 如果 RemoteLLMClient 存在（即导入成功），将其添加到导出列表中
if RemoteLLMClient:
    _exports.append("RemoteLLMClient")

# 设置 __ALL__ 为 _exports 列表，以便模块导入时限制导出的符号
__ALL__ = _exports
```