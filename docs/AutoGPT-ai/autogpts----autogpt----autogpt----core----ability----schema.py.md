# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\schema.py`

```py
# 导入必要的模块
import enum
from typing import Any

from pydantic import BaseModel

# 定义内容类型枚举类
class ContentType(str, enum.Enum):
    # 待定这些实际上是什么。
    TEXT = "text"
    CODE = "code"

# 定义知识类，包含内容、内容类型和内容元数据
class Knowledge(BaseModel):
    content: str
    content_type: ContentType
    content_metadata: dict[str, Any]

# 定义能力结果类，作为能力的标准响应结构
class AbilityResult(BaseModel):
    """The AbilityResult is a standard response struct for an ability."""

    ability_name: str
    ability_args: dict[str, str]
    success: bool
    message: str
    new_knowledge: Knowledge = None

    # 定义一个方法，返回能力结果的摘要信息
    def summary(self) -> str:
        # 将能力参数转换为字符串形式
        kwargs = ", ".join(f"{k}={v}" for k, v in self.ability_args.items())
        # 返回摘要信息
        return f"{self.ability_name}({kwargs}): {self.message}"
```