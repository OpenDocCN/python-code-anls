# `.\transformers\models\luke\__init__.py`

```
# 导入 TYPE_CHECKING 模块，用于在类型检查时导入依赖项
from typing import TYPE_CHECKING

# 导入 OptionalDependencyNotAvailable 异常和 _LazyModule 类，以及 is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_luke": ["LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP", "LukeConfig"],  # 配置模块导入结构
    "tokenization_luke": ["LukeTokenizer"],  # 分词模块导入结构
}

# 检查是否有 torch 库可用，如果不可用，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果引发了异常，则忽略
else:
    # 如果没有引发异常，则添加模型模块导入结构
    _import_structure["modeling_luke"] = [
        "LUKE_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "LukeForEntityClassification",  # 实体分类模型
        "LukeForEntityPairClassification",  # 实体对分类模型
        "LukeForEntitySpanClassification",  # 实体跨度分类模型
        "LukeForMultipleChoice",  # 多项选择模型
        "LukeForQuestionAnswering",  # 问答模型
        "LukeForSequenceClassification",  # 序列分类模型
        "LukeForTokenClassification",  # 标记分类模型
        "LukeForMaskedLM",  # 掩码语言模型
        "LukeModel",  # LUKE 模型
        "LukePreTrainedModel",  # LUKE 预训练模型
    ]

# 如果是类型检查状态
if TYPE_CHECKING:
    # 导入配置模块和分词模块的特定类和对象
    from .configuration_luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig
    from .tokenization_luke import LukeTokenizer

    # 再次检查是否有 torch 库可用，如果不可用，则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 如果引发了异常，则忽略
    else:
        # 如果没有引发异常，则导入模型模块的特定类和对象
        from .modeling_luke import (
            LUKE_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            LukeForEntityClassification,  # 实体分类模型
            LukeForEntityPairClassification,  # 实体对分类模型
            LukeForEntitySpanClassification,  # 实体跨度分类模型
            LukeForMaskedLM,  # 掩码语言模型
            LukeForMultipleChoice,  # 多项选择模型
            LukeForQuestionAnswering,  # 问答模型
            LukeForSequenceClassification,  # 序列分类模型
            LukeForTokenClassification,  # 标记分类模型
            LukeModel,  # LUKE 模型
            LukePreTrainedModel,  # LUKE 预训练模型
        )

# 如果不是类型检查状态
else:
    import sys  # 导入 sys 模块

    # 将当前模块替换为懒加载模块，以延迟导入依赖项
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```