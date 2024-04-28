# `.\transformers\models\mobilevitv2\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构字典
_import_structure = {
    "configuration_mobilevitv2": [
        "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件存档映射
        "MobileViTV2Config",  # MobileViTV2 配置类
        "MobileViTV2OnnxConfig",  # MobileViTV2 ONNX 配置类
    ],
}

# 检查是否可用 torch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加模型相关的导入结构
    _import_structure["modeling_mobilevitv2"] = [
        "MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "MobileViTV2ForImageClassification",  # 用于图像分类的 MobileViTV2 模型
        "MobileViTV2ForSemanticSegmentation",  # 用于语义分割的 MobileViTV2 模型
        "MobileViTV2Model",  # MobileViTV2 模型
        "MobileViTV2PreTrainedModel",  # MobileViTV2 预训练模型
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关类
    from .configuration_mobilevitv2 import (
        MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件存档映射
        MobileViTV2Config,  # MobileViTV2 配置类
        MobileViTV2OnnxConfig,  # MobileViTV2 ONNX 配置类
    )

    # 再次检查是否可用 torch
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关类
        from .modeling_mobilevitv2 import (
            MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            MobileViTV2ForImageClassification,  # 用于图像分类的 MobileViTV2 模型
            MobileViTV2ForSemanticSegmentation,  # 用于语义分割的 MobileViTV2 模型
            MobileViTV2Model,  # MobileViTV2 模型
            MobileViTV2PreTrainedModel,  # MobileViTV2 预训练模型
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块替换为延迟加载的模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```