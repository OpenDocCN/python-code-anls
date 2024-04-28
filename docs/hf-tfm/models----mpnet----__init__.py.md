# `.\transformers\models\mpnet\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义需要导入的结构
_import_structure = {
    "configuration_mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig"],
    "tokenization_mpnet": ["MPNetTokenizer"],
}

# 检查tokenizers是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tokenizers可用，添加"tokenization_mpnet_fast"结构
    _import_structure["tokenization_mpnet_fast"] = ["MPNetTokenizerFast"]

# 检查torch是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，添加"modeling_mpnet"结构
    _import_structure["modeling_mpnet"] = [
        "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MPNetForMaskedLM",
        "MPNetForMultipleChoice",
        "MPNetForQuestionAnswering",
        "MPNetForSequenceClassification",
        "MPNetForTokenClassification",
        "MPNetLayer",
        "MPNetModel",
        "MPNetPreTrainedModel",
    ]

# 检查tf是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tf可用，添加"modeling_tf_mpnet"结构
    _import_structure["modeling_tf_mpnet"] = [
        "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFMPNetEmbeddings",
        "TFMPNetForMaskedLM",
        "TFMPNetForMultipleChoice",
        "TFMPNetForQuestionAnswering",
        "TFMPNetForSequenceClassification",
        "TFMPNetForTokenClassification",
        "TFMPNetMainLayer",
        "TFMPNetModel",
        "TFMPNetPreTrainedModel",
    ]

# 如果是类型检查模式，引入相关类型
if TYPE_CHECKING:
    from .configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
    from .tokenization_mpnet import MPNetTokenizer

    # 检查tokenizers是否可用，若不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若tokenizers可用，引入tokenization_mpnet_fast中的相关类型
        from .tokenization_mpnet_fast import MPNetTokenizerFast

    # 检查torch是否可用，若不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果当前模块是主模块，则导入以下模块
    else:
        # 从当前目录下的 modeling_mpnet 模块中导入以下内容
        from .modeling_mpnet import (
            MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,  # MPNet 预训练模型存档列表
            MPNetForMaskedLM,  # 用于遮蔽语言建模的 MPNet 模型
            MPNetForMultipleChoice,  # 用于多项选择任务的 MPNet 模型
            MPNetForQuestionAnswering,  # 用于问答任务的 MPNet 模型
            MPNetForSequenceClassification,  # 用于序列分类任务的 MPNet 模型
            MPNetForTokenClassification,  # 用于标记分类任务的 MPNet 模型
            MPNetLayer,  # MPNet 模型的一层
            MPNetModel,  # MPNet 模型
            MPNetPreTrainedModel,  # MPNet 预训练模型
        )

    # 尝试检查 TensorFlow 是否可用，若不可用则抛出异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果 TensorFlow 可用，则导入以下模块
    else:
        # 从当前目录下的 modeling_tf_mpnet 模块中导入以下内容
        from .modeling_tf_mpnet import (
            TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,  # TF_MPNet 预训练模型存档列表
            TFMPNetEmbeddings,  # TFMPNet 的 embeddings
            TFMPNetForMaskedLM,  # 用于遮蔽语言建模的 TFMPNet 模型
            TFMPNetForMultipleChoice,  # 用于多项选择任务的 TFMPNet 模型
            TFMPNetForQuestionAnswering,  # 用于问答任务的 TFMPNet 模型
            TFMPNetForSequenceClassification,  # 用于序列分类任务的 TFMPNet 模型
            TFMPNetForTokenClassification,  # 用于标记分类任务的 TFMPNet 模型
            TFMPNetMainLayer,  # TFMPNet 的主层
            TFMPNetModel,  # TFMPNet 模型
            TFMPNetPreTrainedModel,  # TFMPNet 预训练模型
        )
# 如果条件不满足，则执行以下代码块
else:
    # 导入 sys 模块，用于操作 Python 解释器的运行时环境
    import sys
    # 用当前模块的名字作为键，将当前模块替换为 LazyModule 的实例
    # LazyModule 用于延迟导入模块中的对象，以提高性能和启动速度
    # __name__ 表示当前模块的名称，__file__ 表示当前模块的文件路径
    # _import_structure 表示要导入的模块的结构
    # module_spec 表示当前模块的模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```