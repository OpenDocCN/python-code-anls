# `.\graphrag\graphrag\llm\base\_create_cache_key.py`

```py
# 导入 hashlib 模块，用于生成哈希值
import hashlib

# 定义一个私有函数 _llm_string，用于生成语言模型参数的字符串表示
def _llm_string(params: dict) -> str:
    # 如果参数字典中包含 "max_tokens" 键而不包含 "n" 键，将 "n" 设置为 None
    # 这样可以避免为相同的提示创建新的缓存键
    if "max_tokens" in params and "n" not in params:
        params["n"] = None
    # 返回按键值对排序后的参数字典的字符串表示
    return str(sorted((k, v) for k, v in params.items()))

# 定义一个私有函数 _hash，用于对输入字符串进行哈希处理
def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    # 使用 MD5 哈希算法对输入字符串进行编码，并返回其十六进制表示的哈希值
    return hashlib.md5(_input.encode()).hexdigest()  # noqa S324

# 定义函数 create_hash_key，用于根据提示和相关模型及设置计算缓存键
def create_hash_key(operation: str, prompt: str, parameters: dict) -> str:
    """Compute cache key from prompt and associated model and settings.

    Args:
        operation (str): 操作的描述性字符串，用于标识缓存键的作用
        prompt (str): 通过语言模型运行的提示文本
        parameters (dict): 与语言模型相关的参数字典，用于生成模型设置的字符串表示

    Returns:
        str: 缓存键，由操作描述和经过哈希处理的提示及参数字符串组成
    """
    # 生成语言模型参数的字符串表示
    llm_string = _llm_string(parameters)
    # 返回由操作描述、经过哈希处理的提示及参数字符串组成的缓存键
    return f"{operation}-{_hash(prompt + llm_string)}"
```