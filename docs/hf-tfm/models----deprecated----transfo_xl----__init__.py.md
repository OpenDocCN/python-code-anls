# `.\models\deprecated\transfo_xl\__init__.py`

```py
# 导入必要的模块和函数

# 引入类型检查器的模块
from typing import TYPE_CHECKING

# 引入自定义的异常：OptionalDependencyNotAvailable，_LazyModule等
from ....utils import OptionalDependencyNotAvailable, _LazyModule

# 引入判断是否可用的函数：is_tf_available, is_torch_available
from ....utils import is_tf_available, is_torch_available

# 定义导入结构的字典，包含不同模块对应的导入内容
_import_structure = {
    "configuration_transfo_xl": ["TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "TransfoXLConfig"],
    "tokenization_transfo_xl": ["TransfoXLCorpus", "TransfoXLTokenizer"],
}

# 尝试导入 Torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加相关模块到导入结构中
    _import_structure["modeling_transfo_xl"] = [
        "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AdaptiveEmbedding",
        "TransfoXLForSequenceClassification",
        "TransfoXLLMHeadModel",
        "TransfoXLModel",
        "TransfoXLPreTrainedModel",
        "load_tf_weights_in_transfo_xl",
    ]

# 尝试导入 TensorFlow 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加相关模块到导入结构中
    _import_structure["modeling_tf_transfo_xl"] = [
        "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFAdaptiveEmbedding",
        "TFTransfoXLForSequenceClassification",
        "TFTransfoXLLMHeadModel",
        "TFTransfoXLMainLayer",
        "TFTransfoXLModel",
        "TFTransfoXLPreTrainedModel",
    ]

# 如果是类型检查环境，导入类型相关的模块
if TYPE_CHECKING:
    from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
    from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer

    # 尝试导入 Torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则导入相关模块到当前作用域
        from .modeling_transfo_xl import (
            TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )

    # 尝试导入 TensorFlow 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果条件不满足之前的任何情况，则执行以下导入操作
        # 从当前目录的.modeling_tf_transfo_xl模块中导入以下内容：
        from .modeling_tf_transfo_xl import (
            TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFAdaptiveEmbedding,
            TFTransfoXLForSequenceClassification,
            TFTransfoXLLMHeadModel,
            TFTransfoXLMainLayer,
            TFTransfoXLModel,
            TFTransfoXLPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统相关功能
    import sys

    # 设置当前模块的名称对应的模块对象为 _LazyModule 类的实例，用于延迟加载模块
    # 这里将当前模块的名称、文件路径、导入结构和模块规范传递给 _LazyModule 构造函数
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```