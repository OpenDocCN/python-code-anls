# `.\transformers\models\lxmert\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的依赖和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义需要导入的模块结构
_import_structure = {
    "configuration_lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig"],
    "tokenization_lxmert": ["LxmertTokenizer"],
}

# 检查是否存在 Tokenizers，若不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Tokenizers，则导入快速 Tokenizers 模块
    _import_structure["tokenization_lxmert_fast"] = ["LxmertTokenizerFast"]

# 检查是否存在 Torch，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Torch，则导入相关建模模块
    _import_structure["modeling_lxmert"] = [
        "LxmertEncoder",
        "LxmertForPreTraining",
        "LxmertForQuestionAnswering",
        "LxmertModel",
        "LxmertPreTrainedModel",
        "LxmertVisualFeatureEncoder",
        "LxmertXLayer",
    ]

# 检查是否存在 TensorFlow，若不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 TensorFlow，则导入相关建模模块
    _import_structure["modeling_tf_lxmert"] = [
        "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLxmertForPreTraining",
        "TFLxmertMainLayer",
        "TFLxmertModel",
        "TFLxmertPreTrainedModel",
        "TFLxmertVisualFeatureEncoder",
    ]

# 类型检查时导入相关模块
if TYPE_CHECKING:
    from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
    from .tokenization_lxmert import LxmertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_lxmert_fast import LxmertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_lxmert import (
            LxmertEncoder,
            LxmertForPreTraining,
            LxmertForQuestionAnswering,
            LxmertModel,
            LxmertPreTrainedModel,
            LxmertVisualFeatureEncoder,
            LxmertXLayer,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 若存在 TensorFlow，则不进行任何导入操作
    # 如果不在顶层目录下，则从当前目录中的 modeling_tf_lxmert 模块中导入以下内容
    from .modeling_tf_lxmert import (
        TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表
        TFLxmertForPreTraining,  # 导入用于预训练的 TFLxmertForPreTraining 类
        TFLxmertMainLayer,  # 导入 TFLxmertMainLayer 类
        TFLxmertModel,  # 导入 TFLxmertModel 类
        TFLxmertPreTrainedModel,  # 导入 TFLxmertPreTrainedModel 类
        TFLxmertVisualFeatureEncoder,  # 导入 TFLxmertVisualFeatureEncoder 类
    )
# 如果
```