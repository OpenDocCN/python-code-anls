# `.\models\dpr\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_dpr": ["DPR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPRConfig"],
    "tokenization_dpr": [
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
}

# 尝试检查是否可用tokenizers库，若不可用则抛出可选依赖未安装异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
# 若可用则忽略异常
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加tokenizers模块的导入结构
    _import_structure["tokenization_dpr_fast"] = [
        "DPRContextEncoderTokenizerFast",
        "DPRQuestionEncoderTokenizerFast",
        "DPRReaderTokenizerFast",
    ]

# 尝试检查是否可用torch库，若不可用则抛出可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 若可用则忽略异常
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加torch模块的导入结构
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

# 尝试检查是否可用tensorflow库，若不可用则抛出可选依赖未安装异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
# 若可用则忽略异常
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加tensorflow模块的导入结构
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

# 如果是类型检查，则进行类型相关的导入
if TYPE_CHECKING:
    # 导入DPR配置和类型化的tokenizers模块
    from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
    from .tokenization_dpr import (
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )

    # 尝试检查是否可用tokenizers库，若不可用则忽略异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果没有导入 tokenization_dpr_fast 模块，尝试从当前目录下导入相关内容
        from .tokenization_dpr_fast import (
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoderTokenizerFast,
            DPRReaderTokenizerFast,
        )

    try:
        # 检查是否导入了 torch 库，如果没有则抛出异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常 OptionalDependencyNotAvailable 被捕获，则什么也不做
        pass
    else:
        # 如果未捕获异常，则从 modeling_dpr 模块导入相关内容
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
        # 检查是否导入了 tensorflow 库，如果没有则抛出异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常 OptionalDependencyNotAvailable 被捕获，则什么也不做
        pass
    else:
        # 如果未捕获异常，则从 modeling_tf_dpr 模块导入相关内容
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
```  
# 如果不在if分支中，则导入sys模块
import sys
# 将当前模块对象设置为_LazyModule对象，用于延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```