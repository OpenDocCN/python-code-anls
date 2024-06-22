# `.\models\git\__init__.py`

```py
# 版权声明和许可信息
# 版权所有 © 2022 The HuggingFace Team。
# 根据 Apache 许可证 2.0 版本许可。
from typing import TYPE_CHECKING
# 引入类型检查模块

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available
# 从相对路径中引入依赖模块和自定义模块

_import_structure = {
    "configuration_git": ["GIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "GitConfig", "GitVisionConfig"],
    # 定义导入结构字典，包括configuration_git的相关类和属性
    "processing_git": ["GitProcessor"],
    # processing_git的相关类和属性
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
    # 如果torch不可用，则抛出OptionalDependencyNotAvailable异常
except OptionalDependencyNotAvailable:
    pass
    # 捕获OptionalDependencyNotAvailable异常后不做任何操作
else:
    _import_structure["modeling_git"] = [
        "GIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GitForCausalLM",
        "GitModel",
        "GitPreTrainedModel",
        "GitVisionModel",
    ]
    # 如果torch可用，则将modeling_git的相关类和属性添加到导入结构字典

if TYPE_CHECKING:
    from .configuration_git import GIT_PRETRAINED_CONFIG_ARCHIVE_MAP, GitConfig, GitVisionConfig
    from .processing_git import GitProcessor
    # 如果是类型检查，从相对路径导入配置和处理类

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_git import (
            GIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GitForCausalLM,
            GitModel,
            GitPreTrainedModel,
            GitVisionModel,
        )
    # 如果是类型检查，并且torch可用，则从相对路径导入建模相关类和属性
else:
    import sys
    # 引入sys模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 设置当前模块的名称和LazyModule相关参数，将此模块添加到sys.modules中
```