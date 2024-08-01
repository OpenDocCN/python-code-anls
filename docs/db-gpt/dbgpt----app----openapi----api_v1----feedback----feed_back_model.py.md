# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\feedback\feed_back_model.py`

```py
# 引入必要的模块和类
from typing import Optional  # 用于类型提示
from dbgpt._private.pydantic import BaseModel  # 导入 Pydantic 的 BaseModel 类，用于定义数据模型


class FeedBackBody(BaseModel):
    """反馈体数据模型"""

    conv_uid: str  # 对话 ID，字符串类型

    """会话索引，整数类型"""
    conv_index: int

    """人类提出的问题，字符串类型"""
    question: str

    """LLM 答案的评分，整数类型"""
    score: int

    """问题类型，字符串类型"""
    ques_type: str

    user_name: Optional[str] = None  # 可选的用户名称，字符串类型，默认为 None

    """评分详细信息，可选的字符串类型"""
    messages: Optional[str] = None

    """知识空间，可选的字符串类型"""
    knowledge_space: Optional[str] = None
```