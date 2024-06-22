# `.\transformers\models\nystromformer\__init__.py`

```py
# 声明脚本版权信息
# 版权 2022 年 HuggingFace 团队。保留所有权利。
# 根据 Apache 许可证第 2.0 版（“许可证”）许可;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下位置获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依照许可证分发的软件
# 在“按原样”基础上分发，无论是明示的还是暗示的； 
# 查看许可证以获取特定语言的权限和限制

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入依赖项检查和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 声明模块的导入结构
_import_structure = {
    "configuration_nystromformer": ["NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "NystromformerConfig"],
}

# 检查是否存在 torch 库，若不存在则引发可选依赖项不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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

# 如果是类型检查模式，则进行进一步导入
if TYPE_CHECKING:
    from .configuration_nystromformer import NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, NystromformerConfig
    # 检查是否存在 torch 库，若不存在则引发可选依赖项不可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容
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

# 若不是类型检查模式，则进行动态模块创建
else:
    import sys
    # 创建懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```