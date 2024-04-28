# `.\transformers\models\xlm_roberta_xl\__init__.py`

```
# 这是一个 Hugging Face 模型库的 Python 模块，包含了 XLM-RoBERTa-XL 模型的相关内容
# 它首先定义了该模块的导入结构，指定了哪些类和常量可以从该模块中导入

# 导入所需的类型提示
from typing import TYPE_CHECKING

# 导入工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义该模块的导入结构
_import_structure = {
    # 配置相关的类和常量
    "configuration_xlm_roberta_xl": [
        "XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XLMRobertaXLConfig",
        "XLMRobertaXLOnnxConfig",
    ],
}

# 如果 PyTorch 不可用，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用，则添加模型相关的类和常量到导入结构中
    _import_structure["modeling_xlm_roberta_xl"] = [
        "XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMRobertaXLForCausalLM",
        "XLMRobertaXLForMaskedLM",
        "XLMRobertaXLForMultipleChoice",
        "XLMRobertaXLForQuestionAnswering",
        "XLMRobertaXLForSequenceClassification",
        "XLMRobertaXLForTokenClassification",
        "XLMRobertaXLModel",
        "XLMRobertaXLPreTrainedModel",
    ]

# 如果进入了类型检查阶段
if TYPE_CHECKING:
    # 导入配置相关的类和常量
    from .configuration_xlm_roberta_xl import (
        XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaXLConfig,
        XLMRobertaXLOnnxConfig,
    )

    # 如果 PyTorch 不可用，则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 PyTorch 可用，则导入模型相关的类和常量
        from .modeling_xlm_roberta_xl import (
            XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaXLForCausalLM,
            XLMRobertaXLForMaskedLM,
            XLMRobertaXLForMultipleChoice,
            XLMRobertaXLForQuestionAnswering,
            XLMRobertaXLForSequenceClassification,
            XLMRobertaXLForTokenClassification,
            XLMRobertaXLModel,
            XLMRobertaXLPreTrainedModel,
        )

else:
    # 如果不是类型检查阶段，则使用延迟导入机制
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```