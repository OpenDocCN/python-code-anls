# `.\yolov8\ultralytics\nn\__init__.py`

```py
# 导入模块中的特定类和函数，包括：
#   - BaseModel: 基础模型类
#   - ClassificationModel: 分类模型类
#   - DetectionModel: 目标检测模型类
#   - SegmentationModel: 分割模型类
#   - attempt_load_one_weight: 尝试加载单个权重函数
#   - attempt_load_weights: 尝试加载权重函数
#   - guess_model_scale: 推测模型规模函数
#   - guess_model_task: 推测模型任务函数
#   - parse_model: 解析模型函数
#   - torch_safe_load: 安全加载 Torch 模型函数
#   - yaml_model_load: 加载 YAML 格式模型函数
from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_scale,
    guess_model_task,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

# 模块中可以直接访问的全部对象的元组，包括类和函数
__all__ = (
    "attempt_load_one_weight",
    "attempt_load_weights",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
)
```