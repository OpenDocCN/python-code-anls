# `.\models\deprecated\transfo_xl\__init__.py`

```py
# 版权声明和许可协议信息
# 从其他模块导入的模块和函数
from typing import TYPE_CHECKING
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_transfo_xl": ["TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "TransfoXLConfig"],
    "tokenization_transfo_xl": ["TransfoXLCorpus", "TransfoXLTokenizer"],
}

# 尝试导入 torch，如果不可用，则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_transfo_xl"] = [
        "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AdaptiveEmbedding",
        "TransfoXLForSequenceClassification",
        "TransfoXLLMHeadModel",
        "TransfoXLModel",
        "TransfoXLPreTrainedModel",
        "load_tf_weights_in_transfo_xl",
    ]

# 尝试导入 tensorflow，如果不可用，则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_transfo_xl"] = [
        "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFAdaptiveEmbedding",
        "TFTransfoXLForSequenceClassification",
        "TFTransfoXLLMHeadModel",
        "TFTransfoXLMainLayer",
        "TFTransfoXLModel",
        "TFTransfoXLPreTrainedModel",
    ]

# 如果是类型检查阶段，导入特定模块
if TYPE_CHECKING:
    from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
    from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_transfo_xl import (
            TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不成立，则从当前目录下的modeling_tf_transfo_xl模块中导入以下内容：
    # TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST：TRANSFO_XL模型预训练模型文件存档列表
    # TFAdaptiveEmbedding：可适应的嵌入层
    # TFTransfoXLForSequenceClassification：用于序列分类的TransfoXL模型
    # TFTransfoXLLMHeadModel：TransfoXL的语言建模头模型
    # TFTransfoXLMainLayer：TransfoXL的主要层
    # TFTransfoXLModel：TransfoXL模型
    # TFTransfoXLPreTrainedModel：TransfoXL的预训练模型
```  
# 否则，如果进入了这个分支，说明没有找到要导入的模块
else:
    # 导入 sys 模块，用于访问和操作 Python 解释器的运行时环境
    import sys
    # 使用 sys.modules[__name__] 将当前模块替换为 _LazyModule 类的实例
    # __name__ 是当前模块的名称，__file__ 是当前模块的文件名，_import_structure 是一个字典
    # module_spec=__spec__ 将 __spec__ 参数传递给 _LazyModule 类的构造函数
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```