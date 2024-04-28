# `.\models\distilbert\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的
# 没有任何明示或暗示的担保或条件，包括但不限于特定用途的适用性
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_distilbert": [
        "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DistilBertConfig",
        "DistilBertOnnxConfig",
    ],
    "tokenization_distilbert": ["DistilBertTokenizer"],
}

# 检查是否存在 tokenizers 库，若不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_distilbert_fast"] = ["DistilBertTokenizerFast"]

# 检查是否存在 torch 库，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_distilbert"] = [
        "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DistilBertForMaskedLM",
        "DistilBertForMultipleChoice",
        "DistilBertForQuestionAnswering",
        "DistilBertForSequenceClassification",
        "DistilBertForTokenClassification",
        "DistilBertModel",
        "DistilBertPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，若不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_distilbert"] = [
        "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDistilBertForMaskedLM",
        "TFDistilBertForMultipleChoice",
        "TFDistilBertForQuestionAnswering",
        "TFDistilBertForSequenceClassification",
        "TFDistilBertForTokenClassification",
        "TFDistilBertMainLayer",
        "TFDistilBertModel",
        "TFDistilBertPreTrainedModel",
    ]

# 检查是否存在 flax 库，若不存在则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_distilbert"] = [
        "FlaxDistilBertForMaskedLM",
        "FlaxDistilBertForMultipleChoice",
        "FlaxDistilBertForQuestionAnswering",
        "FlaxDistilBertForSequenceClassification",
        "FlaxDistilBertForTokenClassification",
        "FlaxDistilBertModel",
        "FlaxDistilBertPreTrainedModel",
    ]

# 如果是类型检查，则执行以下代码
if TYPE_CHECKING:
    # 从configuration_distilbert模块中导入相关内容
    from .configuration_distilbert import (
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入预训练配置映射
        DistilBertConfig,  # 导入DistilBert配置类
        DistilBertOnnxConfig,  # 导入DistilBert ONNX配置类
    )
    # 从tokenization_distilbert模块中导入DistilBertTokenizer类
    from .tokenization_distilbert import DistilBertTokenizer

    # 尝试检查是否存在tokenizers库，如果不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在tokenizers库，则从tokenization_distilbert_fast模块中导入DistilBertTokenizerFast类
        from .tokenization_distilbert_fast import DistilBertTokenizerFast

    # 尝试检查是否存在torch库，如果不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在torch库，则从modeling_distilbert模块中导入相关内容
        from .modeling_distilbert import (
            DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型存档列表
            DistilBertForMaskedLM,  # 导入DistilBert用于Masked Language Model的类
            DistilBertForMultipleChoice,  # 导入DistilBert用于多选题的类
            DistilBertForQuestionAnswering,  # 导入DistilBert用于问答任务的类
            DistilBertForSequenceClassification,  # 导入DistilBert用于序列分类的类
            DistilBertForTokenClassification,  # 导入DistilBert用于标记分类的类
            DistilBertModel,  # 导入DistilBert模型类
            DistilBertPreTrainedModel,  # 导入DistilBert预训练模型类
        )

    # 尝试检查是否存在tensorflow库，如果不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在tensorflow库，则从modeling_tf_distilbert模块中导入相关内容
        from .modeling_tf_distilbert import (
            TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型存档列表
            TFDistilBertForMaskedLM,  # 导入TF版DistilBert用于Masked Language Model的类
            TFDistilBertForMultipleChoice,  # 导入TF版DistilBert用于多选题的类
            TFDistilBertForQuestionAnswering,  # 导入TF版DistilBert用于问答任务的类
            TFDistilBertForSequenceClassification,  # 导入TF版DistilBert用于序列分类的类
            TFDistilBertForTokenClassification,  # 导入TF版DistilBert用于标记分类的类
            TFDistilBertMainLayer,  # 导入TF版DistilBert主层类
            TFDistilBertModel,  # 导入TF版DistilBert模型类
            TFDistilBertPreTrainedModel,  # 导入TF版DistilBert预训练模型类
        )

    # 尝试检查是否存在flax库，如果不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在flax库，则从modeling_flax_distilbert模块中导入相关内容
        from .modeling_flax_distilbert import (
            FlaxDistilBertForMaskedLM,  # 导入Flax版DistilBert用于Masked Language Model的类
            FlaxDistilBertForMultipleChoice,  # 导入Flax版DistilBert用于多选题的类
            FlaxDistilBertForQuestionAnswering,  # 导入Flax版DistilBert用于问答任务的类
            FlaxDistilBertForSequenceClassification,  # 导入Flax版DistilBert用于序列分类的类
            FlaxDistilBertForTokenClassification,  # 导入Flax版DistilBert用于标记分类的类
            FlaxDistilBertModel,  # 导入Flax版DistilBert模型类
            FlaxDistilBertPreTrainedModel,  # 导入Flax版DistilBert预训练模型类
        )
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```