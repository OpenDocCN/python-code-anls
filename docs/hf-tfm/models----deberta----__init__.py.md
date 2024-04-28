# `.\models\deberta\__init__.py`

```
# 导入模块中的类型检查
from typing import TYPE_CHECKING
# 从模块中导入实用工具函数
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入可选依赖未安装异常类
    _LazyModule,  # 导入惰性加载模块类
    is_tf_available,  # 导入检查是否可用 TensorFlow 的函数
    is_tokenizers_available,  # 导入检查是否可用 Tokenizers 的函数
    is_torch_available,  # 导入检查是否可用 PyTorch 的函数
)


# 定义模块导入结构，指定了各模块中需要导入的内容
_import_structure = {
    "configuration_deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig", "DebertaOnnxConfig"],  # 定义 DEBERTA 配置相关内容的导入结构
    "tokenization_deberta": ["DebertaTokenizer"],  # 定义 DEBERTA 分词器的导入结构
}

# 检查 Tokenizers 是否可用，如果不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Tokenizers 可用，则添加对应的导入结构
    _import_structure["tokenization_deberta_fast"] = ["DebertaTokenizerFast"]

# 检查 PyTorch 是否可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用，则添加对应的导入结构
    _import_structure["modeling_deberta"] = [
        "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",  # 定义 DEBERTA 预训练模型存档列表的导入结构
        "DebertaForMaskedLM",  # 定义 DEBERTA 用于遮蔽语言建模的模型的导入结构
        "DebertaForQuestionAnswering",  # 定义 DEBERTA 用于问答任务的模型的导入结构
        "DebertaForSequenceClassification",  # 定义 DEBERTA 用于序列分类任务的模型的导入结构
        "DebertaForTokenClassification",  # 定义 DEBERTA 用于标记分类任务的模型的导入结构
        "DebertaModel",  # 定义 DEBERTA 模型的导入结构
        "DebertaPreTrainedModel",  # 定义 DEBERTA 预训练模型的导入结构
    ]

# 检查 TensorFlow 是否可用，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加对应的导入结构
    _import_structure["modeling_tf_deberta"] = [
        "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",  # 定义 TensorFlow 下 DEBERTA 预训练模型存档列表的导入结构
        "TFDebertaForMaskedLM",  # 定义 TensorFlow 下 DEBERTA 用于遮蔽语言建模的模型的导入结构
        "TFDebertaForQuestionAnswering",  # 定义 TensorFlow 下 DEBERTA 用于问答任务的模型的导入结构
        "TFDebertaForSequenceClassification",  # 定义 TensorFlow 下 DEBERTA 用于序列分类任务的模型的导入结构
        "TFDebertaForTokenClassification",  # 定义 TensorFlow 下 DEBERTA 用于标记分类任务的模型的导入结构
        "TFDebertaModel",  # 定义 TensorFlow 下 DEBERTA 模型的导入结构
        "TFDebertaPreTrainedModel",  # 定义 TensorFlow 下 DEBERTA 预训练模型的导入结构
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从 DEBERTA 配置模块中导入相关内容
    from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig, DebertaOnnxConfig
    # 从 DEBERTA 分词模块中导入相关内容
    from .tokenization_deberta import DebertaTokenizer

    # 检查 Tokenizers 是否可用，如果不可用则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Tokenizers 可用，则从 DEBERTA 快速分词模块中导入相关内容
        from .tokenization_deberta_fast import DebertaTokenizerFast

    # 检查 PyTorch 是否可用，如果不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从相对路径中导入Deberta相关模块
        from .modeling_deberta import (
            DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DebertaForMaskedLM,
            DebertaForQuestionAnswering,
            DebertaForSequenceClassification,
            DebertaForTokenClassification,
            DebertaModel,
            DebertaPreTrainedModel,
        )

    try:
        # 检查是否存在TensorFlow可用
        if not is_tf_available():
            # 如果TensorFlow不可用，引发OptionalDependencyNotAvailable异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了OptionalDependencyNotAvailable异常，则跳过
        pass
    else:
        # 从相对路径中导入TensorFlow版本的Deberta相关模块
        from .modeling_tf_deberta import (
            TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDebertaForMaskedLM,
            TFDebertaForQuestionAnswering,
            TFDebertaForSequenceClassification,
            TFDebertaForTokenClassification,
            TFDebertaModel,
            TFDebertaPreTrainedModel,
        )
# 如果模块没有被导入过，则执行以下代码
else:
    # 获取当前模块的 sys.modules 引用
    import sys

    # 使用 _LazyModule 创建一个延迟加载的模块对象，并将其设置为当前模块的引用
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```