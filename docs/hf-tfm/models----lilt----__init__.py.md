# `.\models\lilt\__init__.py`

```py
# 引入所需的模块和函数
from typing import TYPE_CHECKING

# 从自定义的工具包中引入异常处理类和延迟加载模块类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置和模型的名称
_import_structure = {
    "configuration_lilt": ["LILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LiltConfig"],
}

# 尝试检查是否可用 Torch 库，如果不可用则引发自定义的异常类
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加相关模型的导入结构
    _import_structure["modeling_lilt"] = [
        "LILT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LiltForQuestionAnswering",
        "LiltForSequenceClassification",
        "LiltForTokenClassification",
        "LiltModel",
        "LiltPreTrainedModel",
    ]

# 如果是类型检查阶段，则从配置和模型模块中导入特定的类和常量
if TYPE_CHECKING:
    from .configuration_lilt import LILT_PRETRAINED_CONFIG_ARCHIVE_MAP, LiltConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_lilt import (
            LILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LiltForQuestionAnswering,
            LiltForSequenceClassification,
            LiltForTokenClassification,
            LiltModel,
            LiltPreTrainedModel,
        )

# 如果不是类型检查阶段，则设置模块为延迟加载模式
else:
    import sys

    # 将当前模块设置为 LazyModule，以便在需要时按需加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```