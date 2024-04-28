# `.\transformers\models\squeezebert\__init__.py`

```py
# 版权声明和许可证信息
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    # 配置文件
    "configuration_squeezebert": [
        "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
        "SqueezeBertConfig",  # SqueezeBert 配置
        "SqueezeBertOnnxConfig",  # SqueezeBert ONNX 配置
    ],
    # SqueezeBert 分词器
    "tokenization_squeezebert": ["SqueezeBertTokenizer"],
}

# 检查 Tokenizers 库是否可用
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_squeezebert_fast"] = ["SqueezeBertTokenizerFast"]

# 检查 Torch 库是否可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_squeezebert"] = [
        "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型归档列表
        "SqueezeBertForMaskedLM",  # 用于 Masked LM 任务的 SqueezeBert 模型
        "SqueezeBertForMultipleChoice",  # 用于多选题任务的 SqueezeBert 模型
        "SqueezeBertForQuestionAnswering",  # 用于问答任务的 SqueezeBert 模型
        "SqueezeBertForSequenceClassification",  # 用于序列分类任务的 SqueezeBert 模型
        "SqueezeBertForTokenClassification",  # 用于标记分类任务的 SqueezeBert 模型
        "SqueezeBertModel",  # SqueezeBert 模型
        "SqueezeBertModule",  # SqueezeBert 模块
        "SqueezeBertPreTrainedModel",  # SqueezeBert 预训练模型
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入必要的模块和类
    from .configuration_squeezebert import (
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SqueezeBertConfig,
        SqueezeBertOnnxConfig,
    )
    from .tokenization_squeezebert import SqueezeBertTokenizer

    # 检查 Tokenizers 库是否可用
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_squeezebert_fast import SqueezeBertTokenizerFast

    # 检查 Torch 库是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_squeezebert import (
            SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForQuestionAnswering,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
            SqueezeBertModel,
            SqueezeBertModule,
            SqueezeBertPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 创建一个LazyModule对象作为当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```