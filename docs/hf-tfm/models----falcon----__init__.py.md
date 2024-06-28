# `.\models\falcon\__init__.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，标明 Falcon 作者和 HuggingFace Inc. 团队的版权
# 根据 Apache License, Version 2.0 许可证，使用该文件需要遵循许可证规定
# 可以在指定许可证网址获取许可证的副本
# 根据适用法律或书面同意的情况下，本软件按"原样"提供，无任何明示或暗示的担保
# 详见许可证以获取特定语言的权限说明
from typing import TYPE_CHECKING

# 从 utils 模块导入必要的依赖项
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],
}

# 检查是否存在 torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加相关模型的导入结构
    _import_structure["modeling_falcon"] = [
        "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FalconForCausalLM",
        "FalconModel",
        "FalconPreTrainedModel",
        "FalconForSequenceClassification",
        "FalconForTokenClassification",
        "FalconForQuestionAnswering",
    ]

# 如果是类型检查阶段，从 configuration_falcon 模块导入相关的配置和类
if TYPE_CHECKING:
    from .configuration_falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig

    # 检查是否存在 torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 库，则从 modeling_falcon 模块导入相关的模型类
        from .modeling_falcon import (
            FALCON_PRETRAINED_MODEL_ARCHIVE_LIST,
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )

# 如果不是类型检查阶段，则动态地将当前模块设置为 LazyModule，并指定导入结构和模块规范
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```