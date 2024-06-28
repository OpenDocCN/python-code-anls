# `.\models\flaubert\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 引入自定义的异常类和模块加载函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertOnnxConfig"],
    "tokenization_flaubert": ["FlaubertTokenizer"],
}

# 检查是否有 Torch 库可用，如果不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加 Flaubert 相关的模型类到导入结构中
    _import_structure["modeling_flaubert"] = [
        "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlaubertForMultipleChoice",
        "FlaubertForQuestionAnswering",
        "FlaubertForQuestionAnsweringSimple",
        "FlaubertForSequenceClassification",
        "FlaubertForTokenClassification",
        "FlaubertModel",
        "FlaubertWithLMHeadModel",
        "FlaubertPreTrainedModel",
    ]

# 检查是否有 TensorFlow 库可用，如果不可用则抛出自定义异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加 TensorFlow 下的 Flaubert 相关模型类到导入结构中
    _import_structure["modeling_tf_flaubert"] = [
        "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFFlaubertForMultipleChoice",
        "TFFlaubertForQuestionAnsweringSimple",
        "TFFlaubertForSequenceClassification",
        "TFFlaubertForTokenClassification",
        "TFFlaubertModel",
        "TFFlaubertPreTrainedModel",
        "TFFlaubertWithLMHeadModel",
    ]

# 如果是类型检查模式，则导入特定的配置和模型类
if TYPE_CHECKING:
    from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertOnnxConfig
    from .tokenization_flaubert import FlaubertTokenizer

    # 检查 Torch 是否可用，如果可用则导入 Flaubert 相关模型类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flaubert import (
            FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaubertForMultipleChoice,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertModel,
            FlaubertPreTrainedModel,
            FlaubertWithLMHeadModel,
        )

    # 检查 TensorFlow 是否可用，如果可用则导入 TensorFlow 下的 Flaubert 相关模型类
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入所需模块和类，这些来自于当前包的子模块 modeling_tf_flaubert
        from .modeling_tf_flaubert import (
            TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表常量
            TFFlaubertForMultipleChoice,  # 导入用于多项选择任务的 Flaubert 模型类
            TFFlaubertForQuestionAnsweringSimple,  # 导入用于简单问答任务的 Flaubert 模型类
            TFFlaubertForSequenceClassification,  # 导入用于序列分类任务的 Flaubert 模型类
            TFFlaubertForTokenClassification,  # 导入用于标记分类任务的 Flaubert 模型类
            TFFlaubertModel,  # 导入 Flaubert 模型基类
            TFFlaubertPreTrainedModel,  # 导入预训练 Flaubert 模型基类
            TFFlaubertWithLMHeadModel,  # 导入带有语言模型头的 Flaubert 模型类
        )
else:
    # 导入系统模块 sys
    import sys

    # 将当前模块 (__name__) 的模块对象替换为一个懒加载模块对象 (_LazyModule)
    # _LazyModule 的参数依次为模块名 (__name__), 模块所在文件名 (__file__), 导入结构 (_import_structure)
    # module_spec 参数指定模块规范 (__spec__)
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```