# `.\models\git\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和模块延迟加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_git": ["GIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "GitConfig", "GitVisionConfig"],
    "processing_git": ["GitProcessor"],
}

# 检查是否可以导入 Torch，如果不行则抛出自定义的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 Torch，则增加一些模型相关的导入结构
    _import_structure["modeling_git"] = [
        "GIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GitForCausalLM",
        "GitModel",
        "GitPreTrainedModel",
        "GitVisionModel",
    ]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的类和常量
    from .configuration_git import GIT_PRETRAINED_CONFIG_ARCHIVE_MAP, GitConfig, GitVisionConfig
    # 导入处理相关的类
    from .processing_git import GitProcessor

    # 再次检查 Torch 是否可用，如果不行则忽略模型相关导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和常量
        from .modeling_git import (
            GIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GitForCausalLM,
            GitModel,
            GitPreTrainedModel,
            GitVisionModel,
        )

# 如果不是类型检查模式，则将当前模块设为一个延迟加载模块
else:
    import sys

    # 使用 _LazyModule 实现模块的延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```