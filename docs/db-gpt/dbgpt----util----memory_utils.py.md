# `.\DB-GPT-src\dbgpt\util\memory_utils.py`

```py
# 导入所需的模块和函数
from typing import Any  # 导入类型提示模块中的 Any 类型
from pympler import asizeof  # 从 pympler 模块导入 asizeof 函数

# 定义一个函数，用于计算对象在内存中所占的字节数
def _get_object_bytes(obj: Any) -> int:
    """Get the bytes of a object in memory

    Args:
        obj (Any): The object to return the bytes
            接收一个任意类型的参数 obj，用于获取其在内存中占用的字节数
    """
    return asizeof.asizeof(obj)  # 调用 pympler 的 asizeof 函数，返回 obj 对象在内存中的字节数
```