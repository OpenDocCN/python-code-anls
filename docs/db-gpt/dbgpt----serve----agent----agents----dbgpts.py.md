# `.\DB-GPT-src\dbgpt\serve\agent\agents\dbgpts.py`

```py
# 从 __future__ 导入 annotations，确保在 Python 3.7 及以上版本中支持类型注解
from __future__ import annotations

# 导入必要的模块和类
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 数据类定义：DbGptsMessage，用于表示数据库中的消息
@dataclass
class DbGptsMessage:
    sender: str         # 消息发送者
    receiver: str       # 消息接收者
    content: str        # 消息内容
    action_report: str  # 操作报告信息

    # 从字典构建 DbGptsMessage 对象的静态方法
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> DbGptsMessage:
        return DbGptsMessage(
            sender=d["sender"],
            receiver=d["receiver"],
            content=d["content"],
            action_report=d["action_report"],  # 修正了错误，应为 action_report
        )

    # 将对象转换为字典的方法
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# 数据类定义：DbGptsTaskStep，用于表示数据库中的任务步骤
@dataclass
class DbGptsTaskStep:
    task_num: str       # 任务编号
    task_content: str   # 任务内容
    state: str          # 任务状态
    result: str         # 任务结果
    agent_name: str     # 代理名称
    model_name: str     # 模型名称

    # 从字典构建 DbGptsTaskStep 对象的静态方法
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> DbGptsTaskStep:
        return DbGptsTaskStep(
            task_num=d["task_num"],
            task_content=d["task_content"],
            state=d["state"],
            result=d["result"],
            agent_name=d["agent_name"],
            model_name=d["model_name"],
        )

    # 将对象转换为字典的方法
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# 数据类定义：DbGptsCompletion，用于表示数据库中的任务完成情况
@dataclass
class DbGptsCompletion:
    conv_id: str                        # 对话 ID
    task_steps: Optional[List[DbGptsTaskStep]]  # 任务步骤列表，可选
    messages: Optional[List[DbGptsMessage]]    # 消息列表，可选

    # 从字典构建 DbGptsCompletion 对象的静态方法
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> DbGptsCompletion:
        return DbGptsCompletion(
            conv_id=d.get("conv_id"),
            task_steps=[DbGptsTaskStep.from_dict(step) for step in d["task_steps"]],
            messages=[DbGptsMessage.from_dict(msg) for msg in d["messages"]],
        )

    # 将对象转换为字典的方法
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
```