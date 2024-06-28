# `.\models\fnet\__init__.py`

```py
# 版权声明和许可证信息
#
# 版权所有 2021 年 HuggingFace 团队。保留所有权利。
# 
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件分发时基于“原样”提供，
# 没有任何形式的明示或暗示保证或条件。
# 有关特定语言的详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 中导入相关的函数和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构的字典，用于组织导入的模块和类
_import_structure = {"configuration_fnet": ["FNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"]}

# 检查是否有 sentencepiece 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 FNetTokenizer 加入导入结构
    _import_structure["tokenization_fnet"] = ["FNetTokenizer"]

# 检查是否有 tokenizers 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 FNetTokenizerFast 加入导入结构
    _import_structure["tokenization_fnet_fast"] = ["FNetTokenizerFast"]

# 检查是否有 torch 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将一系列 FNet 模型和类加入导入结构
    _import_structure["modeling_fnet"] = [
        "FNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FNetForMaskedLM",
        "FNetForMultipleChoice",
        "FNetForNextSentencePrediction",
        "FNetForPreTraining",
        "FNetForQuestionAnswering",
        "FNetForSequenceClassification",
        "FNetForTokenClassification",
        "FNetLayer",
        "FNetModel",
        "FNetPreTrainedModel",
    ]

# 如果是类型检查模式，则导入配置类和相关依赖
if TYPE_CHECKING:
    from .configuration_fnet import FNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_fnet import FNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_fnet_fast import FNetTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从当前目录导入指定模块和变量
        from .modeling_fnet import (
            FNET_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表
            FNetForMaskedLM,  # 导入用于Masked Language Modeling的FNet模型
            FNetForMultipleChoice,  # 导入用于多项选择任务的FNet模型
            FNetForNextSentencePrediction,  # 导入用于下一句预测任务的FNet模型
            FNetForPreTraining,  # 导入用于预训练的FNet模型
            FNetForQuestionAnswering,  # 导入用于问答任务的FNet模型
            FNetForSequenceClassification,  # 导入用于序列分类任务的FNet模型
            FNetForTokenClassification,  # 导入用于标记分类任务的FNet模型
            FNetLayer,  # 导入FNet的层类
            FNetModel,  # 导入通用的FNet模型类
            FNetPreTrainedModel,  # 导入预训练模型的基类
        )
else:
    # 导入 sys 模块，用于在运行时操作 Python 解释器
    import sys

    # 将当前模块注册到 sys.modules 中，使得模块在运行时可以被动态加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```