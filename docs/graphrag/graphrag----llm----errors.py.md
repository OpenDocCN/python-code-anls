# `.\graphrag\graphrag\llm\errors.py`

```py
# 定义一个自定义错误类 RetriesExhaustedError，继承自 RuntimeError
class RetriesExhaustedError(RuntimeError):
    """Retries exhausted error."""

    # 初始化方法，接受参数 name（字符串）和 num_retries（整数）
    def __init__(self, name: str, num_retries: int) -> None:
        """Init method definition."""
        # 调用父类 RuntimeError 的初始化方法，设置错误消息
        super().__init__(f"Operation '{name}' failed - {num_retries} retries exhausted")
```