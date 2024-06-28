# `.\models\layoutlmv3\__init__.py`

```py
# 版权声明及许可信息
# 2022年由HuggingFace团队版权所有。
# 根据Apache许可证2.0版（“许可证”）授权；
# 您只能在遵守许可证的情况下使用此文件。
# 您可以通过访问以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
#

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义了导入结构的字典，用于模块化地导入布局LMv3相关模块和类
_import_structure = {
    "configuration_layoutlmv3": [
        "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LayoutLMv3Config",
        "LayoutLMv3OnnxConfig",
    ],
    "processing_layoutlmv3": ["LayoutLMv3Processor"],
    "tokenization_layoutlmv3": ["LayoutLMv3Tokenizer"],
}

# 检查是否安装了tokenizers库，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了tokenizers，则添加"tokenization_layoutlmv3_fast"到_import_structure字典
    _import_structure["tokenization_layoutlmv3_fast"] = ["LayoutLMv3TokenizerFast"]

# 检查是否安装了torch库，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了torch，则添加"modeling_layoutlmv3"到_import_structure字典
    _import_structure["modeling_layoutlmv3"] = [
        "LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv3ForQuestionAnswering",
        "LayoutLMv3ForSequenceClassification",
        "LayoutLMv3ForTokenClassification",
        "LayoutLMv3Model",
        "LayoutLMv3PreTrainedModel",
    ]

# 检查是否安装了tensorflow库，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了tensorflow，则添加"modeling_tf_layoutlmv3"到_import_structure字典
    _import_structure["modeling_tf_layoutlmv3"] = [
        "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMv3ForQuestionAnswering",
        "TFLayoutLMv3ForSequenceClassification",
        "TFLayoutLMv3ForTokenClassification",
        "TFLayoutLMv3Model",
        "TFLayoutLMv3PreTrainedModel",
    ]

# 检查是否安装了vision库，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了vision，则添加"feature_extraction_layoutlmv3"和"image_processing_layoutlmv3"到_import_structure字典
    _import_structure["feature_extraction_layoutlmv3"] = ["LayoutLMv3FeatureExtractor"]
    _import_structure["image_processing_layoutlmv3"] = ["LayoutLMv3ImageProcessor"]

# 如果是类型检查阶段，则从各模块导入对应的类和常量
if TYPE_CHECKING:
    from .configuration_layoutlmv3 import (
        LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv3Config,
        LayoutLMv3OnnxConfig,
    )
    from .processing_layoutlmv3 import LayoutLMv3Processor
    from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入 LayoutLMv3TokenizerFast，如果 OptionalDependencyNotAvailable 异常抛出则跳过
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast

    # 尝试检查是否 Torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Torch 版本的 LayoutLMv3 模型和相关类
        from .modeling_layoutlmv3 import (
            LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv3ForQuestionAnswering,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Model,
            LayoutLMv3PreTrainedModel,
        )

    # 尝试检查是否 TensorFlow 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 TensorFlow 版本的 LayoutLMv3 模型和相关类
        from .modeling_tf_layoutlmv3 import (
            TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
            TFLayoutLMv3Model,
            TFLayoutLMv3PreTrainedModel,
        )

    # 尝试检查是否 Vision 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 LayoutLMv3 的图像特征提取器和图像处理器
        from .feature_extraction_layoutlmv3 import LayoutLMv3FeatureExtractor
        from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor
else:
    # 如果不在前面的任何一个条件分支中，则执行以下操作
    import sys
    # 导入sys模块，用于访问系统相关的功能

    # 将当前模块注册到sys.modules中，使用_LazyModule延迟加载模式
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # __name__表示当前模块名，__file__表示当前模块的文件名
    # _LazyModule是一个延迟加载模块的类，用于按需加载模块的内容
    # _import_structure和__spec__是用于模块导入和规范的参数
```