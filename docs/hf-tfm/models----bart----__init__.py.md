# `.\transformers\models\bart\__init__.py`

```
# 导入类型检查模块，用于检查类型
from typing import TYPE_CHECKING
# 导入工具模块
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入自定义异常类
    _LazyModule,  # 导入懒加载模块
    is_flax_available,  # 检查是否可用 Flax 框架
    is_tf_available,  # 检查是否可用 TensorFlow 框架
    is_tokenizers_available,  # 检查是否可用 Tokenizers 库
    is_torch_available,  # 检查是否可用 PyTorch 库
)

# 定义模块的导入结构
_import_structure = {
    "configuration_bart": ["BART_PRETRAINED_CONFIG_ARCHIVE_MAP", "BartConfig", "BartOnnxConfig"],  # Bart 相关配置
    "tokenization_bart": ["BartTokenizer"],  # Bart 模型的标记器
}

# 检查 Tokenizers 库是否可用，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加快速标记器到导入结构中
    _import_structure["tokenization_bart_fast"] = ["BartTokenizerFast"]

# 检查 PyTorch 库是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 PyTorch 版本的 Bart 模型到导入结构中
    _import_structure["modeling_bart"] = [
        "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BartForCausalLM",
        "BartForConditionalGeneration",
        "BartForQuestionAnswering",
        "BartForSequenceClassification",
        "BartModel",
        "BartPreTrainedModel",
        "BartPretrainedModel",
        "PretrainedBartModel",
    ]

# 检查 TensorFlow 库是否可用，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 TensorFlow 版本的 Bart 模型到导入结构中
    _import_structure["modeling_tf_bart"] = [
        "TFBartForConditionalGeneration",
        "TFBartForSequenceClassification",
        "TFBartModel",
        "TFBartPretrainedModel",
    ]

# 检查 Flax 库是否可用，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 Flax 版本的 Bart 模型到导入结构中
    _import_structure["modeling_flax_bart"] = [
        "FlaxBartDecoderPreTrainedModel",
        "FlaxBartForCausalLM",
        "FlaxBartForConditionalGeneration",
        "FlaxBartForQuestionAnswering",
        "FlaxBartForSequenceClassification",
        "FlaxBartModel",
        "FlaxBartPreTrainedModel",
    ]

# 如果是类型检查模式，则导入额外的类型相关模块
if TYPE_CHECKING:
    # 导入 Bart 相关配置和标记器
    from .configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig, BartOnnxConfig
    from .tokenization_bart import BartTokenizer
    
    # 检查 Tokenizers 库是否可用，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入快速 Bart 标记器
        from .tokenization_bart_fast import BartTokenizerFast
    
    # 检查 PyTorch 库是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入可选的依赖项，如果依赖项不可用，则跳过
        try:
            # 尝试导入可选依赖项的异常
            except OptionalDependencyNotAvailable:
                pass
            else:
                # 如果没有异常，则导入 BART 模型的相关内容
                from .modeling_bart import (
                    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
                    BartForCausalLM,
                    BartForConditionalGeneration,
                    BartForQuestionAnswering,
                    BartForSequenceClassification,
                    BartModel,
                    BartPreTrainedModel,
                    BartPretrainedModel,
                    PretrainedBartModel,
                )
    
        # 尝试检查 TensorFlow 是否可用，如果不可用则跳过
        try:
            if not is_tf_available():
                # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            # 捕获到异常则继续执行
            pass
        else:
            # 如果没有异常，则导入 TensorFlow 版本的 BART 模型相关内容
            from .modeling_tf_bart import (
                TFBartForConditionalGeneration,
                TFBartForSequenceClassification,
                TFBartModel,
                TFBartPretrainedModel,
            )
    
        # 尝试检查 Flax 是否可用，如果不可用则跳过
        try:
            if not is_flax_available():
                # 如果 Flax 不可用，则引发 OptionalDependencyNotAvailable 异常
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            # 捕获到异常则继续执行
            pass
        else:
            # 如果没有异常，则导入 Flax 版本的 BART 模型相关内容
            from .modeling_flax_bart import (
                FlaxBartDecoderPreTrainedModel,
                FlaxBartForCausalLM,
                FlaxBartForConditionalGeneration,
                FlaxBartForQuestionAnswering,
                FlaxBartForSequenceClassification,
                FlaxBartModel,
                FlaxBartPreTrainedModel,
            )
# 否则，如果进入了这个分支，则导入 sys 模块，用于进行模块级别的操作
import sys

# 将当前模块设置为一个 LazyModule 实例，用于延迟加载模块内容
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```