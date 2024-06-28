# `.\models\m2m_100\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入可选的依赖未安装异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_m2m_100": ["M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP", "M2M100Config", "M2M100OnnxConfig"],
    "tokenization_m2m_100": ["M2M100Tokenizer"],
}

# 检查是否有 torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加模型相关的导入
    _import_structure["modeling_m2m_100"] = [
        "M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST",
        "M2M100ForConditionalGeneration",
        "M2M100Model",
        "M2M100PreTrainedModel",
    ]

# 如果是类型检查阶段，导入具体的模型配置和标记器
if TYPE_CHECKING:
    from .configuration_m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config, M2M100OnnxConfig
    from .tokenization_m2m_100 import M2M100Tokenizer

    # 再次检查是否有 torch 可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则导入模型相关的类
        from .modeling_m2m_100 import (
            M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST,
            M2M100ForConditionalGeneration,
            M2M100Model,
            M2M100PreTrainedModel,
        )

# 如果不是类型检查阶段，将当前模块设为延迟加载模块
else:
    import sys

    # 使用延迟加载模块的方式加载当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```