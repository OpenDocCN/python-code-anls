# `.\models\xlm_roberta\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 从相对路径导入工具函数和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，用于存储不同模块的导入结构
_import_structure = {
    "configuration_xlm_roberta": [
        "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XLMRobertaConfig",
        "XLMRobertaOnnxConfig",
    ],
}

# 检查是否安装了 sentencepiece 库，如果没有则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 XLMRobertaTokenizer 添加到导入结构中
    _import_structure["tokenization_xlm_roberta"] = ["XLMRobertaTokenizer"]

# 检查是否安装了 tokenizers 库，如果没有则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 XLMRobertaTokenizerFast 添加到导入结构中
    _import_structure["tokenization_xlm_roberta_fast"] = ["XLMRobertaTokenizerFast"]

# 检查是否安装了 torch 库，如果没有则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 XLMRoBerta 相关模型和类添加到导入结构中
    _import_structure["modeling_xlm_roberta"] = [
        "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMRobertaForCausalLM",
        "XLMRobertaForMaskedLM",
        "XLMRobertaForMultipleChoice",
        "XLMRobertaForQuestionAnswering",
        "XLMRobertaForSequenceClassification",
        "XLMRobertaForTokenClassification",
        "XLMRobertaModel",
        "XLMRobertaPreTrainedModel",
    ]

# 检查是否安装了 tensorflow 库，如果没有则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 TFXLMRoberta 相关模型和类添加到导入结构中
    _import_structure["modeling_tf_xlm_roberta"] = [
        "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLMRobertaForCausalLM",
        "TFXLMRobertaForMaskedLM",
        "TFXLMRobertaForMultipleChoice",
        "TFXLMRobertaForQuestionAnswering",
        "TFXLMRobertaForSequenceClassification",
        "TFXLMRobertaForTokenClassification",
        "TFXLMRobertaModel",
        "TFXLMRobertaPreTrainedModel",
    ]

# 检查是否安装了 flax 库，如果没有则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，继续处理 flax 相关的导入结构（此处代码省略）
    pass
    # 将一组模块名称添加到 _import_structure 字典中的 "modeling_flax_xlm_roberta" 键下
    _import_structure["modeling_flax_xlm_roberta"] = [
        "FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlaxXLMRobertaForMaskedLM",
        "FlaxXLMRobertaForCausalLM",
        "FlaxXLMRobertaForMultipleChoice",
        "FlaxXLMRobertaForQuestionAnswering",
        "FlaxXLMRobertaForSequenceClassification",
        "FlaxXLMRobertaForTokenClassification",
        "FlaxXLMRobertaModel",
        "FlaxXLMRobertaPreTrainedModel",
    ]
if TYPE_CHECKING:
    # 引入需要的配置和模型类映射
    from .configuration_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaConfig,
        XLMRobertaOnnxConfig,
    )

    try:
        # 检查是否安装了 sentencepiece
        if not is_sentencepiece_available():
            # 如果未安装，抛出可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 sentencepiece，引入 XLMRobertaTokenizer
        from .tokenization_xlm_roberta import XLMRobertaTokenizer

    try:
        # 检查是否安装了 tokenizers
        if not is_tokenizers_available():
            # 如果未安装，抛出可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 tokenizers，引入 XLMRobertaTokenizerFast
        from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

    try:
        # 检查是否安装了 torch
        if not is_torch_available():
            # 如果未安装，抛出可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 torch，引入 XLM-Roberta 模型和相关类
        from .modeling_xlm_roberta import (
            XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )

    try:
        # 检查是否安装了 tensorflow
        if not is_tf_available():
            # 如果未安装，抛出可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 tensorflow，引入 TF 版本的 XLM-Roberta 模型和相关类
        from .modeling_tf_xlm_roberta import (
            TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLMRobertaForCausalLM,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
            TFXLMRobertaPreTrainedModel,
        )

    try:
        # 检查是否安装了 flax
        if not is_flax_available():
            # 如果未安装，抛出可选依赖不可用的异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 flax，引入 Flax 版本的 XLM-Roberta 模型和相关类
        from .modeling_flax_xlm_roberta import (
            FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaxXLMRobertaForCausalLM,
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
            FlaxXLMRobertaPreTrainedModel,
        )

else:
    # 如果不是类型检查阶段，则将当前模块设置为懒加载模块
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```