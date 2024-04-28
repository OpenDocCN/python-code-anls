# `.\models\funnel\__init__.py`

```py
# 导入必要的模块和类型提示
from typing import TYPE_CHECKING

# 从工具包中导入必要的模块和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_funnel": ["FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP", "FunnelConfig"],  # FUNNEL模型配置的导入结构
    "convert_funnel_original_tf_checkpoint_to_pytorch": [],  # 将FUNNEL原始TF检查点转换为PyTorch的导入结构
    "tokenization_funnel": ["FunnelTokenizer"],  # FUNNEL模型的标记化导入结构
}

# 检查是否有标记工具可用，如果不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_funnel_fast"] = ["FunnelTokenizerFast"]  # 快速标记化的FUNNEL模型的导入结构

# 检查是否有PyTorch可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_funnel"] = [  # FUNNEL模型的建模导入结构
        "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FunnelBaseModel",
        "FunnelForMaskedLM",
        "FunnelForMultipleChoice",
        "FunnelForPreTraining",
        "FunnelForQuestionAnswering",
        "FunnelForSequenceClassification",
        "FunnelForTokenClassification",
        "FunnelModel",
        "FunnelPreTrainedModel",
        "load_tf_weights_in_funnel",
    ]

# 检查是否有TensorFlow可用，如果不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_funnel"] = [  # FUNNEL模型的TensorFlow建模导入结构
        "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFFunnelBaseModel",
        "TFFunnelForMaskedLM",
        "TFFunnelForMultipleChoice",
        "TFFunnelForPreTraining",
        "TFFunnelForQuestionAnswering",
        "TFFunnelForSequenceClassification",
        "TFFunnelForTokenClassification",
        "TFFunnelModel",
        "TFFunnelPreTrainedModel",
    ]

# 如果是类型检查模式，则导入特定的模块和类
if TYPE_CHECKING:
    from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig  # 导入FUNNEL模型的配置
    from .tokenization_funnel import FunnelTokenizer  # 导入FUNNEL模型的标记化

    # 检查是否有标记工具可用，如果不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_funnel_fast import FunnelTokenizerFast  # 导入快速标记化的FUNNEL模型的类

    # 检查是否有PyTorch可用，如果不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从modeling_funnel模块中导入所需的类和变量
        from .modeling_funnel import (
            FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
            FunnelBaseModel,
            FunnelForMaskedLM,
            FunnelForMultipleChoice,
            FunnelForPreTraining,
            FunnelForQuestionAnswering,
            FunnelForSequenceClassification,
            FunnelForTokenClassification,
            FunnelModel,
            FunnelPreTrainedModel,
            load_tf_weights_in_funnel,
        )

    try:
        # 检查是否存在 TensorFlow 库
        if not is_tf_available():
            # 如果不存在，抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 捕获异常，不做任何操作
        pass
    else:
        # 如果存在 TensorFlow 库，则从modeling_tf_funnel模块中导入所需的类和变量
        from .modeling_tf_funnel import (
            TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFFunnelBaseModel,
            TFFunnelForMaskedLM,
            TFFunnelForMultipleChoice,
            TFFunnelForPreTraining,
            TFFunnelForQuestionAnswering,
            TFFunnelForSequenceClassification,
            TFFunnelForTokenClassification,
            TFFunnelModel,
            TFFunnelPreTrainedModel,
        )
# 如果不是以上任何情况，则引入sys模块
import sys
# 将当前模块注册为 LazyModule 的实例
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```