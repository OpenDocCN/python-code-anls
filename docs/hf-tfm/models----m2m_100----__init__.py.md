# `.\transformers\models\m2m_100\__init__.py`

```
# 版权声明和许可证信息

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的模块和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_m2m_100": ["M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP", "M2M100Config", "M2M100OnnxConfig"],
    "tokenization_m2m_100": ["M2M100Tokenizer"],
}

# 检查 torch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果 torch 可用，则添加 modeling_m2m_100 到导入结构
else:
    _import_structure["modeling_m2m_100"] = [
        "M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST",
        "M2M100ForConditionalGeneration",
        "M2M100Model",
        "M2M100PreTrainedModel",
    ]

# 如果是类型检查阶段，则导入相关模块
if TYPE_CHECKING:
    from .configuration_m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config, M2M100OnnxConfig
    from .tokenization_m2m_100 import M2M100Tokenizer

    # 再次检查 torch 是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 可用，则导入 modeling_m2m_100 模块
    else:
        from .modeling_m2m_100 import (
            M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST,
            M2M100ForConditionalGeneration,
            M2M100Model,
            M2M100PreTrainedModel,
        )

# 如果不是类型检查阶段，则将当前模块设为懒加载模块
else:
    import sys
    # 动态添加模块到当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```