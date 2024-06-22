# `.\transformers\models\tapas\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的依赖项检查模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义并初始化需要导入的模块及其子模块的结构
_import_structure = {
    "configuration_tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig"],
    "tokenization_tapas": ["TapasTokenizer"],
}

# 检查是否有可用的 torch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tapas"] = [
        "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TapasForMaskedLM",
        "TapasForQuestionAnswering",
        "TapasForSequenceClassification",
        "TapasModel",
        "TapasPreTrainedModel",
        "load_tf_weights_in_tapas",
    ]

# 检查是否有可用的 tensorflow
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_tapas"] = [
        "TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFTapasForMaskedLM",
        "TFTapasForQuestionAnswering",
        "TFTapasForSequenceClassification",
        "TFTapasModel",
        "TFTapasPreTrainedModel",
    ]

# 如果是类型检查环境，导入所需模块和子模块
if TYPE_CHECKING:
    # 导入 tapas 的相关配置
    from .configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
    # 导入 tapas 的 tokenizer
    from .tokenization_tapas import TapasTokenizer

    # 检查是否有可用的 torch
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tapas 的建模相关模块
        from .modeling_tapas import (
            TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
            TapasModel,
            TapasPreTrainedModel,
            load_tf_weights_in_tapas,
        )

    # 检查是否有可用的 tensorflow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tapas 的 tensorflow 建模相关模块
        from .modeling_tf_tapas import (
            TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFTapasForMaskedLM,
            TFTapasForQuestionAnswering,
            TFTapasForSequenceClassification,
            TFTapasModel,
            TFTapasPreTrainedModel,
        )

# 如果不是类型检查环境，使用 _LazyModule 创建模块并懒加载所需的模块
else:
    import sys

    # 创建懒加载模块 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```