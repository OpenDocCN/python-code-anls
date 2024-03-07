# `.\PokeLLMon\poke_env\exceptions.py`

```
"""
This module contains exceptions.
"""

# 定义一个自定义异常类 ShowdownException，继承自内置异常类 Exception
class ShowdownException(Exception):
    """
    This exception is raised when a non-managed message
    is received from the server.
    """
    # 当从服务器接收到非受控消息时引发此异常

    pass
```