# `.\models\layoutlm\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 从工具模块中导入所需的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，包含模块导入结构
_import_structure = {
    "configuration_layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMOnnxConfig"],
    "tokenization_layoutlm": ["LayoutLMTokenizer"],
}

# 检查是否存在 tokenizers 库，如果不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 库，则添加对应的模块到导入结构字典中
    _import_structure["tokenization_layoutlm_fast"] = ["LayoutLMTokenizerFast"]

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加对应的模块到导入结构字典中
    _import_structure["modeling_layoutlm"] = [
        "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMForMaskedLM",
        "LayoutLMForSequenceClassification",
        "LayoutLMForTokenClassification",
        "LayoutLMForQuestionAnswering",
        "LayoutLMModel",
        "LayoutLMPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则添加对应的模块到导入结构字典中
    _import_structure["modeling_tf_layoutlm"] = [
        "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMForMaskedLM",
        "TFLayoutLMForSequenceClassification",
        "TFLayoutLMForTokenClassification",
        "TFLayoutLMForQuestionAnswering",
        "TFLayoutLMMainLayer",
        "TFLayoutLMModel",
        "TFLayoutLMPreTrainedModel",
    ]

# 如果是类型检查模式，导入类型检查所需的模块和类
if TYPE_CHECKING:
    from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMOnnxConfig
    from .tokenization_layoutlm import LayoutLMTokenizer

    # 检查是否存在 tokenizers 库，如果不存在则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 tokenizers 库，则导入对应的模块
        from .tokenization_layoutlm_fast import LayoutLMTokenizerFast

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是 TensorFlow 可用状态，则引发 OptionalDependencyNotAvailable 异常
    else:
        # 从当前包中导入相关模块和符号
        from .modeling_layoutlm import (
            LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表
            LayoutLMForMaskedLM,  # 导入用于遮蔽语言建模的 LayoutLM 模型
            LayoutLMForQuestionAnswering,  # 导入用于问答任务的 LayoutLM 模型
            LayoutLMForSequenceClassification,  # 导入用于序列分类任务的 LayoutLM 模型
            LayoutLMForTokenClassification,  # 导入用于标记分类任务的 LayoutLM 模型
            LayoutLMModel,  # 导入 LayoutLM 的基础模型
            LayoutLMPreTrainedModel,  # 导入 LayoutLM 的预训练模型基类
        )
    try:
        # 检查 TensorFlow 是否可用，如果不可用，则触发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，直接跳过
        pass
    else:
        # 从当前包中导入 TensorFlow 版本的相关模块和符号
        from .modeling_tf_layoutlm import (
            TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入 TensorFlow 版本的预训练模型存档列表
            TFLayoutLMForMaskedLM,  # 导入用于遮蔽语言建模的 TensorFlow 版本的 LayoutLM 模型
            TFLayoutLMForQuestionAnswering,  # 导入用于问答任务的 TensorFlow 版本的 LayoutLM 模型
            TFLayoutLMForSequenceClassification,  # 导入用于序列分类任务的 TensorFlow 版本的 LayoutLM 模型
            TFLayoutLMForTokenClassification,  # 导入用于标记分类任务的 TensorFlow 版本的 LayoutLM 模型
            TFLayoutLMMainLayer,  # 导入 TensorFlow 版本的 LayoutLM 主层
            TFLayoutLMModel,  # 导入 TensorFlow 版本的 LayoutLM 基础模型
            TFLayoutLMPreTrainedModel,  # 导入 TensorFlow 版本的 LayoutLM 预训练模型基类
        )
# 否则（即非if条件下的情况），导入sys模块
else:
    # 使用sys.modules字典，将当前模块名(__name__)映射到一个_LazyModule对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```