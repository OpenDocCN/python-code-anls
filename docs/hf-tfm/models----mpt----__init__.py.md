# `.\transformers\models\mpt\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入自定义的异常，用于处理可选依赖不可用的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包含模型配置和模型
_import_structure = {
    "configuration_mpt": ["MPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MptConfig", "MptOnnxConfig"],
}

# 检查是否存在 torch 库，如果不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在 torch 库，则添加模型到导入结构中
else:
    _import_structure["modeling_mpt"] = [
        "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MptForCausalLM",
        "MptModel",
        "MptPreTrainedModel",
        "MptForSequenceClassification",
        "MptForTokenClassification",
        "MptForQuestionAnswering",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入模型配置和模型
    from .configuration_mpt import MPT_PRETRAINED_CONFIG_ARCHIVE_MAP, MptConfig, MptOnnxConfig
    # 再次检查是否存在 torch 库，如果不存在则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果存在 torch 库，则导入模型
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

# 如果不是类型检查环境，则将当前模块设为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```