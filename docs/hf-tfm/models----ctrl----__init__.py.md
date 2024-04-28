# `.\models\ctrl\__init__.py`

```py
# 2020年由HuggingFace团队版权所有
#
# 根据Apache许可证2.0版本（“许可证”）获得许可；
# 您不得使用此文件，除非符合许可证的要求。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则在“AS IS”基础上分发软件，
# 没有任何形式的担保或条件，无论是明示或暗示的。
# 有关特定语言的权限和限制，请参阅许可证。
#
# 从类型检查导入TYPE_CHECKING
from typing import TYPE_CHECKING

从...工具中导入可选依赖未可用，_LazyModule，is_tf_available，is_torch_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 这是一个导入结构字典，将要导入的模块和对应的方法放在一起
_import_structure = {
    "configuration_ctrl": ["CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CTRLConfig"],
    "tokenization_ctrl": ["CTRLTokenizer"],
}

# 尝试导入torch，如果不可用则引发OptionalDependencyNotAvailable
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    如果可用，则将"modeling_ctrl"添加到导入结构中
    _import_structure["modeling_ctrl"] = [
        "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CTRLForSequenceClassification",
        "CTRLLMHeadModel",
        "CTRLModel",
        "CTRLPreTrainedModel",
    ]

# 尝试导入tensorflow，如果不可用则引发OptionalDependencyNotAvailable
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    如果可用，则将"modeling_tf_ctrl"添加到导入结构中
    _import_structure["modeling_tf_ctrl"] = [
        "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCTRLForSequenceClassification",
        "TFCTRLLMHeadModel",
        "TFCTRLModel",
        "TFCTRLPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    从.configuration_ctrl中导入CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP，CTRLConfig
    from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
    从.tokenization_ctrl中导入CTRLTokenizer
    from .tokenization_ctrl import CTRLTokenizer

    # 尝试导入torch，如果不可用则引发OptionalDependencyNotAvailable
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        从.modeling_ctrl中导入CTRL_PRETRAINED_MODEL_ARCHIVE_LIST，CTRLForSequenceClassification，CTRLLMHeadModel，CTRLModel，CTRLPreTrainedModel
        from .modeling_ctrl import (
            CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            CTRLForSequenceClassification,
            CTRLLMHeadModel,
            CTRLModel,
            CTRLPreTrainedModel,
        )

    # 尝试导入tensorflow，如果不可用则引发OptionalDependencyNotAvailable
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        从.modeling_tf_ctrl中导入TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST，TFCTRLForSequenceClassification，TFCTRLLMHeadModel，TFCTRLModel，TFCTRLPreTrainedModel
        from .modeling_tf_ctrl import (
            TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCTRLForSequenceClassification,
            TFCTRLLMHeadModel,
            TFCTRLModel,
            TFCTRLPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    导入sys模块
    import sys

    将当前模块的名称和全局变量__file__传入_LazyModule，并指定导入结构_import_structure，模块规范为__spec__
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```