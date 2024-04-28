# `.\transformers\models\nllb_moe\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入自定义的异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_nllb_moe": [
        "NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射表
        "NllbMoeConfig",  # NLLB MOE 模型配置类
    ]
}

# 尝试导入 PyTorch，如果不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用，则添加模型相关内容到导入结构中
    _import_structure["modeling_nllb_moe"] = [
        "NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "NllbMoeForConditionalGeneration",  # 用于条件生成的 NLLB MOE 模型
        "NllbMoeModel",  # NLLB MOE 模型基类
        "NllbMoePreTrainedModel
```