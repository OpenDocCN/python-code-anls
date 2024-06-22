# `.\models\layoutlm\__init__.py`

```py
# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入必要的工具类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {
    "configuration_layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMOnnxConfig"],
    "tokenization_layoutlm": ["LayoutLMTokenizer"],
}

# 检查是否安装了 tokenizers，如果未安装则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_layoutlm_fast"] = ["LayoutLMTokenizerFast"]

# 检查是否安装了 torch，如果未安装则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlm"] = [
        "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMForMaskedLM",
        "LayoutLMForSequenceClassification",
        "LayoutLMForTokenClassification",
        "LayoutLMForQuestionAnswering",
        "LayoutLMModel",
        "LayoutLMPreTrainedModel",
    ]

# 检查是否安装了 tensorflow，如果未安装则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_layoutlm"] = [
        "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMForMaskedLM",
        "TFLayoutLMForSequenceClassification",
        "TFLayoutLMForTokenClassification",
        "TFLayoutLMForQuestionAnswering",
        "TFLayoutLMMainLayer",
        "TFLayoutLMModel",
        "TFLayoutLMPreTrainedModel",
    ]

# 如果当前环境支持类型检查，则执行以下导入操作
if TYPE_CHECKING:
    from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMOnnxConfig
    from .tokenization_layoutlm import LayoutLMTokenizer

    # 检查是否安装了 tokenizers，如果未安装则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlm_fast import LayoutLMTokenizerFast

    # 检查是否安装了 torch，如果未安装则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是条件为真，执行下面的代码块
    else:
        # 从modeling_layoutlm中导入所需的模块和类
        from .modeling_layoutlm import (
            LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMForMaskedLM,
            LayoutLMForQuestionAnswering,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
    # 尝试执行下面的代码块
    try:
        # 如果没有安装tensorflow，引发OptionalDependencyNotAvailable错误
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获OptionalDependencyNotAvailable错误
    except OptionalDependencyNotAvailable:
        # 什么也不做，继续执行代码
        pass
    # 如果上述代码块没有引发错误，执行下面的代码块
    else:
        # 从modeling_tf_layoutlm中导入所需的模块和类
        from .modeling_tf_layoutlm import (
            TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMForMaskedLM,
            TFLayoutLMForQuestionAnswering,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )
# 如果上述条件不成立，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```