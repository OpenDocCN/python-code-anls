# `.\models\megatron_bert\__init__.py`

```py
# 引入必要的模块和函数，包括类型检查和依赖检查
from typing import TYPE_CHECKING
# 从相对路径导入自定义的异常和模块加载器
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构的字典，包括配置和模型名称
_import_structure = {
    "configuration_megatron_bert": ["MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MegatronBertConfig"],
}

# 尝试检查是否存在 Torch 库，如果不存在则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加 Megatron BERT 模型相关的模块和类到导入结构中
    _import_structure["modeling_megatron_bert"] = [
        "MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MegatronBertForCausalLM",
        "MegatronBertForMaskedLM",
        "MegatronBertForMultipleChoice",
        "MegatronBertForNextSentencePrediction",
        "MegatronBertForPreTraining",
        "MegatronBertForQuestionAnswering",
        "MegatronBertForSequenceClassification",
        "MegatronBertForTokenClassification",
        "MegatronBertModel",
        "MegatronBertPreTrainedModel",
    ]

# 如果是类型检查阶段，则执行以下导入
if TYPE_CHECKING:
    # 从相对路径导入 Megatron BERT 的配置和模型定义
    from .configuration_megatron_bert import MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MegatronBertConfig

    # 尝试检查是否存在 Torch 库，如果不存在则引发自定义异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从相对路径导入 Megatron BERT 的模型定义
        from .modeling_megatron_bert import (
            MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MegatronBertForCausalLM,
            MegatronBertForMaskedLM,
            MegatronBertForMultipleChoice,
            MegatronBertForNextSentencePrediction,
            MegatronBertForPreTraining,
            MegatronBertForQuestionAnswering,
            MegatronBertForSequenceClassification,
            MegatronBertForTokenClassification,
            MegatronBertModel,
            MegatronBertPreTrainedModel,
        )

# 如果不是类型检查阶段，则执行以下懒加载模块的设置
else:
    # 导入 sys 模块用于动态修改当前模块对象
    import sys

    # 使用自定义的惰性模块加载器将当前模块替换为 LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```