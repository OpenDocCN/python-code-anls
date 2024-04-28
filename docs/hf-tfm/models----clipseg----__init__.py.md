# `.\transformers\models\clipseg\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING
# 导入自定义的异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包含了各个子模块的名称列表
_import_structure = {
    "configuration_clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件存档映射
        "CLIPSegConfig",  # CLIPSeg 模型配置
        "CLIPSegTextConfig",  # CLIPSeg 文本模型配置
        "CLIPSegVisionConfig",  # CLIPSeg 视觉模型配置
    ],
    "processing_clipseg": ["CLIPSegProcessor"],  # CLIPSeg 处理器
}

# 检查是否导入了 Torch 库，如果未导入则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 Torch 库，则将模型相关的名称添加到导入结构中
    _import_structure["modeling_clipseg"] = [
        "CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "CLIPSegModel",  # CLIPSeg 模型
        "CLIPSegPreTrainedModel",  # CLIPSeg 预训练模型
        "CLIPSegTextModel",  # CLIPSeg 文本模型
        "CLIPSegVisionModel",  # CLIPSeg 视觉模型
        "CLIPSegForImageSegmentation",  # 用于图像分割的 CLIPSeg 模型
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入配置、处理器和模型相关的类和常量
    from .configuration_clipseg import (
        CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件存档映射
        CLIPSegConfig,  # CLIPSeg 模型配置
        CLIPSegTextConfig,  # CLIPSeg 文本模型配置
        CLIPSegVisionConfig,  # CLIPSeg 视觉模型配置
    )
    from .processing_clipseg import CLIPSegProcessor  # CLIPSeg 处理器

    # 再次检查是否导入了 Torch 库，如果未导入则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和常量
        from .modeling_clipseg import (
            CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            CLIPSegForImageSegmentation,  # 用于图像分割的 CLIPSeg 模型
            CLIPSegModel,  # CLIPSeg 模型
            CLIPSegPreTrainedModel,  # CLIPSeg 预训练模型
            CLIPSegTextModel,  # CLIPSeg 文本模型
            CLIPSegVisionModel,  # CLIPSeg 视觉模型
        )

# 如果不是类型检查环境，则将当前模块替换为 LazyModule 类的实例
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```