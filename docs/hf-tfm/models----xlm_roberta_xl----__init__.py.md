# `.\models\xlm_roberta_xl\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义的异常和模块懒加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包含了配置和模型的映射和类名
_import_structure = {
    "configuration_xlm_roberta_xl": [
        "XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XLMRobertaXLConfig",
        "XLMRobertaXLOnnxConfig",
    ],
}

# 尝试检测是否存在 torch 库，如果不存在则抛出自定义的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 存在，则添加相关模型的导入结构
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

# 如果处于类型检查模式
if TYPE_CHECKING:
    # 从配置模块中导入特定的符号（符号已在上面定义）
    from .configuration_xlm_roberta_xl import (
        XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaXLConfig,
        XLMRobertaXLOnnxConfig,
    )

    # 再次尝试检查 torch 库的存在，如果不存在则抛出自定义的异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型定义模块中导入特定的符号（符号已在上面定义）
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

# 如果不处于类型检查模式
else:
    # 引入系统模块
    import sys

    # 动态地将当前模块设置为一个懒加载模块，延迟加载导入的模块和符号
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```