# `.\models\nystromformer\__init__.py`

```py
# 版权声明和许可证信息，版权归 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的保证或条件
# 请参阅许可证，了解详细信息
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的工具函数和异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_nystromformer": ["NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "NystromformerConfig"],
}

# 尝试检查 torch 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 modeling_nystromformer 模块到导入结构中
    _import_structure["modeling_nystromformer"] = [
        "NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NystromformerForMaskedLM",
        "NystromformerForMultipleChoice",
        "NystromformerForQuestionAnswering",
        "NystromformerForSequenceClassification",
        "NystromformerForTokenClassification",
        "NystromformerLayer",
        "NystromformerModel",
        "NystromformerPreTrainedModel",
    ]

# 如果是类型检查模式，则从相应模块导入必要的类和变量
if TYPE_CHECKING:
    from .configuration_nystromformer import NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, NystromformerConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_nystromformer import (
            NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            NystromformerForMaskedLM,
            NystromformerForMultipleChoice,
            NystromformerForQuestionAnswering,
            NystromformerForSequenceClassification,
            NystromformerForTokenClassification,
            NystromformerLayer,
            NystromformerModel,
            NystromformerPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块设置为 LazyModule 的实例，以支持延迟加载
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```