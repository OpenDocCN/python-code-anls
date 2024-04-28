# `.\transformers\models\qdqbert\__init__.py`

```py
# 版权声明和许可证信息
# NVIDIA 公司和 HuggingFace 团队版权所有
#
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证，否则不得使用本文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除法律要求或书面约定，否则在"按原样"基础上分发
# 没有任何明示或暗示的担保或条件
# 查看许可证以了解权限和限制
from typing import TYPE_CHECKING

# 导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {"configuration_qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"]}

# 尝试导入 Torch，如果无法导入则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入 Torch，则添加 QDQBERT 模型相关内容到导入结构中
    _import_structure["modeling_qdqbert"] = [
        "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "QDQBertForMaskedLM",
        "QDQBertForMultipleChoice",
        "QDQBertForNextSentencePrediction",
        "QDQBertForQuestionAnswering",
        "QDQBertForSequenceClassification",
        "QDQBertForTokenClassification",
        "QDQBertLayer",
        "QDQBertLMHeadModel",
        "QDQBertModel",
        "QDQBertPreTrainedModel",
        "load_tf_weights_in_qdqbert",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置文件中导入 QDQBERT 相关内容
    from .configuration_qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig

    # 尝试导入 Torch，如果无法导入则捕获异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果成功导入 Torch，则从模型文件中导入 QDQBERT 相关内容
        from .modeling_qdqbert import (
            QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLayer,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 动态创建一个 LazyModule 对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```