# `.\transformers\models\lilt\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入自定义的异常，用于处理可选依赖不可用情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置和模型
_import_structure = {
    "configuration_lilt": ["LILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LiltConfig"],  # 配置部分
}

# 检查是否存在 torch，若不存在，则抛出自定义的可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch，则添加模型部分到导入结构
    _import_structure["modeling_lilt"] = [
        "LILT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型列表
        "LiltForQuestionAnswering",  # 用于问答任务的 LILT 模型
        "LiltForSequenceClassification",  # 用于序列分类任务的 LILT 模型
        "LiltForTokenClassification",  # 用于标记分类任务的 LILT 模型
        "LiltModel",  # LILT 模型
        "LiltPreTrainedModel",  # 预训练的 LILT 模型
    ]

# 如果处于类型检查模式，则导入配置和模型相关内容
if TYPE_CHECKING:
    # 导入配置相关内容
    from .configuration_lilt import LILT_PRETRAINED_CONFIG_ARCHIVE_MAP, LiltConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
        from .modeling_lilt import (
            LILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LiltForQuestionAnswering,
            LiltForSequenceClassification,
            LiltForTokenClassification,
            LiltModel,
            LiltPreTrainedModel,
        )

# 如果不处于类型检查模式，则将当前模块设为懒加载模块
else:
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```