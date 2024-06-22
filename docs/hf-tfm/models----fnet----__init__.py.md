# `.\models\fnet\__init__.py`

```py
# 引入需要的模块和类
from typing import TYPE_CHECKING

# 从工具模块中引入所需内容
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {"configuration_fnet": ["FNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"]}

# 尝试检查是否安装了 SentencePiece 库
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 FNetTokenizer 到导入结构中
    _import_structure["tokenization_fnet"] = ["FNetTokenizer"]

# 尝试检查是否安装了 Tokenizers 库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 FNetTokenizerFast 到导入结构中
    _import_structure["tokenization_fnet_fast"] = ["FNetTokenizerFast"]

# 尝试检查是否安装了 PyTorch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 FNet 相关模块到导入结构中
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

# 若是类型检查模式，则添加更多导入
if TYPE_CHECKING:
    # 从配置模块中导入配置相关内容
    from .configuration_fnet import FNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig

    # 尝试检查是否安装了 SentencePiece 库
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则添加 FNetTokenizer 到导入结构中
        from .tokenization_fnet import FNetTokenizer

    # 尝试检查是否安装了 Tokenizers 库
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则添加 FNetTokenizerFast 到导入结构中
        from .tokenization_fnet_fast import FNetTokenizerFast

    # 尝试检查是否安装了 PyTorch 库
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
```  
    # 否则，从当前目录的modeling_fnet模块中导入以下内容
    from .modeling_fnet import (
        FNET_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型的存档列表
        FNetForMaskedLM,  # 用于遮蔽语言建模的FNet模型
        FNetForMultipleChoice,  # 用于多项选择任务的FNet模型
        FNetForNextSentencePrediction,  # 用于下一个句子预测的FNet模型
        FNetForPreTraining,  # 用于预训练的FNet模型
        FNetForQuestionAnswering,  # 用于问答任务的FNet模型
        FNetForSequenceClassification,  # 用于序列分类任务的FNet模型
        FNetForTokenClassification,  # 用于令牌分类任务的FNet模型
        FNetLayer,  # FNet模型的层
        FNetModel,  # FNet模型
        FNetPreTrainedModel,  # FNet预训练模型
    )
# 如果之前的条件均不满足，即找不到模块，则导入 sys 模块
import sys
# 将当前模块设为 sys.modules 中的项，其值为 _LazyModule 类的实例，该实例在被调用时会根据需要加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```