# `.\DB-GPT-src\dbgpt\model\cluster\base.py`

```py
from typing import Any, Dict, List, Optional  # 导入需要的类型定义

from dbgpt._private.pydantic import BaseModel  # 导入 Pydantic 的 BaseModel 类
from dbgpt.core.interface.message import ModelMessage  # 导入消息模型
from dbgpt.model.base import WorkerApplyType  # 导入工作器应用类型
from dbgpt.model.parameter import WorkerType  # 导入工作器类型枚举

WORKER_MANAGER_SERVICE_TYPE = "service"  # 定义工作器管理服务类型
WORKER_MANAGER_SERVICE_NAME = "WorkerManager"  # 定义工作器管理服务名称


class PromptRequest(BaseModel):
    messages: List[ModelMessage]  # 消息列表
    model: str  # 模型名称
    prompt: str = None  # 提示文本，默认为 None
    temperature: float = None  # 生成温度，默认为 None
    max_new_tokens: int = None  # 最大生成令牌数，默认为 None
    stop: str = None  # 停止条件，默认为 None
    stop_token_ids: List[int] = []  # 停止令牌 ID 列表，默认为空列表
    context_len: int = None  # 上下文长度，默认为 None
    echo: bool = True  # 是否回显，默认为 True
    span_id: str = None  # 跨度 ID，默认为 None
    metrics: bool = False  # 是否返回推理指标，默认为 False
    """Whether to return metrics of inference"""
    version: str = "v2"  # 消息版本，默认为 "v2"
    """Message version, default to v2"""
    context: Dict[str, Any] = None  # 模型的上下文信息，默认为 None
    """Context information for the model"""


class EmbeddingsRequest(BaseModel):
    model: str  # 模型名称
    input: List[str]  # 输入文本列表
    span_id: Optional[str] = None  # 跨度 ID，可选，默认为 None
    query: Optional[str] = None  # 对于重新排序模型，需要查询文本
    """For rerank model, query is required"""


class CountTokenRequest(BaseModel):
    model: str  # 模型名称
    prompt: str  # 提示文本


class ModelMetadataRequest(BaseModel):
    model: str  # 模型名称


class WorkerApplyRequest(BaseModel):
    model: str  # 模型名称
    apply_type: WorkerApplyType  # 工作器应用类型
    worker_type: WorkerType = WorkerType.LLM  # 工作器类型，默认为语言模型
    params: Dict = None  # 参数字典，可选，默认为 None
    apply_user: str = None  # 应用用户，可选，默认为 None


class WorkerParameterRequest(BaseModel):
    model: str  # 模型名称
    worker_type: WorkerType = WorkerType.LLM  # 工作器类型，默认为语言模型


class WorkerStartupRequest(BaseModel):
    host: str  # 主机地址
    port: int  # 端口号
    model: str  # 模型名称
    worker_type: WorkerType  # 工作器类型
    params: Dict  # 参数字典
```