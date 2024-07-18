# `.\graphrag\graphrag\llm\types\llm_callbacks.py`

```py
# 版权声明，指明此代码版权归 Microsoft Corporation 所有，基于 MIT 许可证发布

"""OpenAI DataShaper 包的类型定义。"""

# 导入所需模块中的 Callable 类
from collections.abc import Callable

# 导入本地模块中的 LLMInvocationResult 类
from .llm_invocation_result import LLMInvocationResult

# 定义 ErrorHandlerFn 类型别名，表示错误处理函数的类型定义
ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]

# 定义 LLMInvocationFn 类型别名，表示处理 LLM 调用结果的函数类型
LLMInvocationFn = Callable[[LLMInvocationResult], None]

# 定义 OnCacheActionFn 类型别名，表示处理缓存命中的函数类型
OnCacheActionFn = Callable[[str, str | None], None]

# 定义 IsResponseValidFn 类型别名，表示检查 LLM 响应是否有效的函数类型
IsResponseValidFn = Callable[[dict], bool]
```