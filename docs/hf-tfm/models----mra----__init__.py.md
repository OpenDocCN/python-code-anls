# `.\models\mra\__init__.py`

```py
# flake8: noqa
# 无法在此模块中忽略 "F401 '...' imported but unused" 警告，以保留其他警告。因此，完全不检查此模块。

# 版权 2023 年 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件是基于"按现状"提供的，
# 没有任何形式的明示或暗示担保或条件。
# 有关特定语言的权限，请参阅许可证。

from typing import TYPE_CHECKING

# 导入自定义的异常类和模块懒加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义导入结构，用于延迟加载模块
_import_structure = {"configuration_mra": ["MRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MraConfig"]}

# 检查是否可用 torch，若不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，添加模型相关的导入结构
    _import_structure["modeling_mra"] = [
        "MRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MraForMaskedLM",
        "MraForMultipleChoice",
        "MraForQuestionAnswering",
        "MraForSequenceClassification",
        "MraForTokenClassification",
        "MraLayer",
        "MraModel",
        "MraPreTrainedModel",
    ]

# 如果是类型检查模式，导入必要的类型声明
if TYPE_CHECKING:
    from .configuration_mra import MRA_PRETRAINED_CONFIG_ARCHIVE_MAP, MraConfig

    # 再次检查 torch 是否可用，并导入相关的模型类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mra import (
            MRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MraForMaskedLM,
            MraForMultipleChoice,
            MraForQuestionAnswering,
            MraForSequenceClassification,
            MraForTokenClassification,
            MraLayer,
            MraModel,
            MraPreTrainedModel,
        )
# 如果不是类型检查模式，设置当前模块为懒加载模式
else:
    import sys

    # 使用 _LazyModule 将当前模块设置为懒加载模式
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```