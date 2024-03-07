# `.\PokeLLMon\poke_env\data\normalize.py`

```
# 导入 functools 模块中的 lru_cache 装饰器
from functools import lru_cache

# 使用 lru_cache 装饰器缓存函数的结果，缓存大小为 2 的 13 次方
@lru_cache(2**13)
# 定义函数 to_id_str，将全名转换为对应的 ID 字符串
def to_id_str(name: str) -> str:
    """Converts a full-name to its corresponding id string.
    :param name: The name to convert.
    :type name: str
    :return: The corresponding id string.
    :rtype: str
    """
    # 将输入的名字中的字母和数字提取出来，转换为小写，并拼接成字符串
    return "".join(char for char in name if char.isalnum()).lower()
```