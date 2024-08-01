# `.\DB-GPT-src\dbgpt\model\base.py`

```py
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from dbgpt.util.parameter_utils import ParameterDescription

# 导入必要的模块和类

class ModelType:
    """ModelType枚举定义了模型的类型"""

    HF = "huggingface"
    LLAMA_CPP = "llama.cpp"
    PROXY = "proxy"
    VLLM = "vllm"
    # TODO, support more model type
    # 各种支持的模型类型，可以根据需要添加更多类型

@dataclass
class ModelInstance:
    """ModelInstance表示一个模型实例的详细信息"""

    model_name: str
    host: str
    port: int
    weight: Optional[float] = 1.0
    check_healthy: Optional[bool] = True
    healthy: Optional[bool] = False
    enabled: Optional[bool] = True
    prompt_template: Optional[str] = None
    last_heartbeat: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """将ModelInstance转换为字典"""
        return asdict(self)

class WorkerApplyType(str, Enum):
    """WorkerApplyType枚举定义了工作器实例的应用操作类型"""

    START = "start"
    STOP = "stop"
    RESTART = "restart"
    UPDATE_PARAMS = "update_params"

@dataclass
class WorkerApplyOutput:
    """WorkerApplyOutput表示工作器应用操作的输出结果"""

    message: str
    success: Optional[bool] = True
    # The seconds cost to apply some action to worker instances
    timecost: Optional[int] = -1

    @staticmethod
    def reduce(outs: List["WorkerApplyOutput"]) -> "WorkerApplyOutput":
        """合并所有WorkerApplyOutput的输出结果

        Args:
            outs (List["WorkerApplyOutput"]): WorkerApplyOutput列表
        """
        if not outs:
            return WorkerApplyOutput("Not outputs")
        combined_success = all(out.success for out in outs)
        max_timecost = max(out.timecost for out in outs)
        combined_message = "\n;".join(out.message for out in outs)
        return WorkerApplyOutput(combined_message, combined_success, max_timecost)

@dataclass
class SupportedModel:
    """SupportedModel表示支持的模型的信息"""

    model: str
    path: str
    worker_type: str
    path_exist: bool
    proxy: bool
    enabled: bool
    params: List[ParameterDescription]

    @classmethod
    def from_dict(cls, model_data: Dict) -> "SupportedModel":
        """从字典数据创建SupportedModel对象

        Args:
            model_data (Dict): 包含模型信息的字典

        Returns:
            SupportedModel: 创建的SupportedModel对象
        """
        params = model_data.get("params", [])
        if params:
            params = [ParameterDescription(**param) for param in params]
        model_data["params"] = params
        return cls(**model_data)

@dataclass
class WorkerSupportedModel:
    """WorkerSupportedModel表示支持的工作器模型的信息"""

    host: str
    port: int
    models: List[SupportedModel]

    @classmethod
    def from_dict(cls, worker_data: Dict) -> "WorkerSupportedModel":
        """从字典数据创建WorkerSupportedModel对象

        Args:
            worker_data (Dict): 包含工作器模型信息的字典

        Returns:
            WorkerSupportedModel: 创建的WorkerSupportedModel对象
        """
        models = [
            SupportedModel.from_dict(model_data) for model_data in worker_data["models"]
        ]
        worker_data["models"] = models
        return cls(**worker_data)

@dataclass
class FlatSupportedModel(SupportedModel):
    """FlatSupportedModel用于Web端展示支持的模型信息"""

    host: str
    port: int

    @staticmethod
    def from_supports(
        supports: List[WorkerSupportedModel],
        # 将WorkerSupportedModel列表转换为FlatSupportedModel对象
    ) -> List["FlatSupportedModel"]:
        results = []  # 初始化一个空列表用于存储结果
        for s in supports:  # 遍历参数 supports 中的每个元素 s
            host, port, models = s.host, s.port, s.models  # 从 s 中解构出 host、port 和 models
            for m in models:  # 遍历 models 中的每个元素 m
                kwargs = asdict(m)  # 将 m 转换为关键字参数字典 kwargs
                kwargs["host"] = host  # 将 host 添加到 kwargs 中
                kwargs["port"] = port  # 将 port 添加到 kwargs 中
                results.append(FlatSupportedModel(**kwargs))  # 使用 kwargs 创建 FlatSupportedModel 对象，并添加到 results 列表中
        return results  # 返回包含所有 FlatSupportedModel 对象的列表 results
```