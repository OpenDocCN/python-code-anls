# `.\models\mpt\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_mpt": ["MPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MptConfig", "MptOnnxConfig"],
}

# 检查是否导入了torch，若未导入则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加模型相关的导入结构
    _import_structure["modeling_mpt"] = [
        "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MptForCausalLM",
        "MptModel",
        "MptPreTrainedModel",
        "MptForSequenceClassification",
        "MptForTokenClassification",
        "MptForQuestionAnswering",
    ]

# 如果是类型检查模式，导入特定的配置和模型类
if TYPE_CHECKING:
    from .configuration_mpt import MPT_PRETRAINED_CONFIG_ARCHIVE_MAP, MptConfig, MptOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mpt import (
            MPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MptForCausalLM,
            MptForQuestionAnswering,
            MptForSequenceClassification,
            MptForTokenClassification,
            MptModel,
            MptPreTrainedModel,
        )

# 在非类型检查模式下，将当前模块设置为一个懒加载模块
else:
    import sys

    # 将当前模块替换为一个懒加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```