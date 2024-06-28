# `.\models\xmod\__init__.py`

```py
# flake8: noqa
# 禁用 flake8 检查，因为无法忽略 "F401 '...' imported but unused" 警告，但要保留其他警告。因此完全不检查此模块。

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本许可。除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"原样"提供的，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。

from typing import TYPE_CHECKING

# 导入异常处理模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_xmod": [
        "XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XmodConfig",
        "XmodOnnxConfig",
    ],
}

# 检查是否导入了 torch，如果未导入则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 torch，则添加以下模块到导入结构中
    _import_structure["modeling_xmod"] = [
        "XMOD_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XmodForCausalLM",
        "XmodForMaskedLM",
        "XmodForMultipleChoice",
        "XmodForQuestionAnswering",
        "XmodForSequenceClassification",
        "XmodForTokenClassification",
        "XmodModel",
        "XmodPreTrainedModel",
    ]

# 如果 TYPE_CHECKING 为真，则导入配置和建模模块
if TYPE_CHECKING:
    from .configuration_xmod import XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP, XmodConfig, XmodOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_xmod import (
            XMOD_PRETRAINED_MODEL_ARCHIVE_LIST,
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodModel,
            XmodPreTrainedModel,
        )

# 如果不是 TYPE_CHECKING 模式，则使用 _LazyModule 模块化导入结构
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```