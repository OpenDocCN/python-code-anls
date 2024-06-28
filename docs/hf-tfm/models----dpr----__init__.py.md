# `.\models\dpr\__init__.py`

```
# 引入必要的模块和类型检查
from typing import TYPE_CHECKING

# 从相对路径引入工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构的字典，用于延迟加载模块
_import_structure = {
    "configuration_dpr": ["DPR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPRConfig"],
    "tokenization_dpr": [
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
}

# 检查是否可用 tokenizers，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加快速 tokenization_dpr_fast 模块到导入结构字典中
    _import_structure["tokenization_dpr_fast"] = [
        "DPRContextEncoderTokenizerFast",
        "DPRQuestionEncoderTokenizerFast",
        "DPRReaderTokenizerFast",
    ]

# 检查是否可用 torch，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_dpr 模块到导入结构字典中
    _import_structure["modeling_dpr"] = [
        "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPRContextEncoder",
        "DPRPretrainedContextEncoder",
        "DPRPreTrainedModel",
        "DPRPretrainedQuestionEncoder",
        "DPRPretrainedReader",
        "DPRQuestionEncoder",
        "DPRReader",
    ]

# 检查是否可用 tensorflow，若不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_tf_dpr 模块到导入结构字典中
    _import_structure["modeling_tf_dpr"] = [
        "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDPRContextEncoder",
        "TFDPRPretrainedContextEncoder",
        "TFDPRPretrainedQuestionEncoder",
        "TFDPRPretrainedReader",
        "TFDPRQuestionEncoder",
        "TFDPRReader",
    ]

# 如果是类型检查阶段，导入必要的类型和模块
if TYPE_CHECKING:
    from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
    from .tokenization_dpr import (
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )

    # 检查是否可用 tokenizers，在类型检查阶段也进行检查
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从当前目录下的tokenization_dpr_fast模块中导入以下快速tokenizer类
    from .tokenization_dpr_fast import (
        DPRContextEncoderTokenizerFast,
        DPRQuestionEncoderTokenizerFast,
        DPRReaderTokenizerFast,
    )

try:
    # 检查是否已经安装了torch依赖
    if not is_torch_available():
        # 如果没有安装，抛出OptionalDependencyNotAvailable异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到OptionalDependencyNotAvailable异常，则不进行任何操作
    pass
else:
    # 否则，从当前目录下的modeling_dpr模块中导入以下内容
    from .modeling_dpr import (
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPRContextEncoder,
        DPRPretrainedContextEncoder,
        DPRPreTrainedModel,
        DPRPretrainedQuestionEncoder,
        DPRPretrainedReader,
        DPRQuestionEncoder,
        DPRReader,
    )

try:
    # 检查是否已经安装了tensorflow依赖
    if not is_tf_available():
        # 如果没有安装，抛出OptionalDependencyNotAvailable异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到OptionalDependencyNotAvailable异常，则不进行任何操作
    pass
else:
    # 否则，从当前目录下的modeling_tf_dpr模块中导入以下内容
    from .modeling_tf_dpr import (
        TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDPRContextEncoder,
        TFDPRPretrainedContextEncoder,
        TFDPRPretrainedQuestionEncoder,
        TFDPRPretrainedReader,
        TFDPRQuestionEncoder,
        TFDPRReader,
    )
else:
    # 导入 sys 模块，用于在运行时动态操作 Python 解释器
    import sys

    # 将当前模块的名称添加到 sys.modules 中，并指定为一个懒加载模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```