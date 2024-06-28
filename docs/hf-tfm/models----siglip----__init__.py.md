# `.\models\siglip\__init__.py`

```
# 引入必要的依赖和模块结构定义
from typing import TYPE_CHECKING

# 从 HuggingFace 的 utils 模块中导入所需的工具和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构，用于动态导入所需的类和函数
_import_structure = {
    "configuration_siglip": [
        "SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SiglipConfig",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "processing_siglip": ["SiglipProcessor"],
}

# 检查是否存在 sentencepiece，若不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在，将 SiglipTokenizer 加入导入结构
    _import_structure["tokenization_siglip"] = ["SiglipTokenizer"]

# 检查是否存在 vision 库，若不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在，将 SiglipImageProcessor 加入导入结构
    _import_structure["image_processing_siglip"] = ["SiglipImageProcessor"]

# 检查是否存在 torch 库，若不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在，将 Siglip 相关的模型和类加入导入结构
    _import_structure["modeling_siglip"] = [
        "SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SiglipModel",
        "SiglipPreTrainedModel",
        "SiglipTextModel",
        "SiglipVisionModel",
        "SiglipForImageClassification",
    ]

# 如果是类型检查阶段，导入所需的具体类和函数
if TYPE_CHECKING:
    from .configuration_siglip import (
        SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    from .processing_siglip import SiglipProcessor

    # 检查是否存在 sentencepiece，若存在则导入 SiglipTokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_siglip import SiglipTokenizer

    # 检查是否存在 vision 库，若存在则导入 SiglipImageProcessor
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_siglip import SiglipImageProcessor

    # 检查是否存在 torch 库，若存在则导入 Siglip 相关的模型和类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_siglip import (
            SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            SiglipForImageClassification,
            SiglipModel,
            SiglipPreTrainedModel,
            SiglipTextModel,
            SiglipVisionModel,
        )

else:
    # 如果不是类型检查阶段，则什么都不做，这部分代码不会执行
    pass
    import sys
    导入 sys 模块，用于访问和操作 Python 解释器的运行时环境和变量。
    
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    将当前模块注册为一个懒加载模块。这行代码的作用是将当前模块的模块对象（通过__name__获取）关联到一个自定义的 _LazyModule 实例，这个实例接受当前模块的名称、文件路径（通过 globals()["__file__"] 获取）、一个导入结构（_import_structure）、和一个模块规范（module_spec=__spec__）作为参数。
```