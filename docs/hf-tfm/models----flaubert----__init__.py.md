# `.\models\flaubert\__init__.py`

```
# 版权声明和许可证信息
# 在特定条件下使用此文件
# 获取许可证信息的链接
# 分发本文件基于 ASIS 基础
# 没有保证或条件，明示或暗示
# 查看许可证以查看特定语言的权限和限制

# 导入必要的库和模块
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertOnnxConfig"],
    "tokenization_flaubert": ["FlaubertTokenizer"],
}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 torch 模型相关的导入结构
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

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 tensorflow 模型相关的导入结构
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

# 如果是类型检查，则导入特定的模块
if TYPE_CHECKING:
    from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertOnnxConfig
    from .tokenization_flaubert import FlaubertTokenizer

    # 如果存在 torch 库，则导入相关模块
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

    # 如果存在 tensorflow 库，则导入相关模块
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从当前目录下的模块 "modeling_tf_flaubert" 中导入以下内容：
    # TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST: TF Flaubert 预训练模型的存档列表
    # TFFlaubertForMultipleChoice: 用于多选题的 TF Flaubert 模型
    # TFFlaubertForQuestionAnsweringSimple: 用于简单问答的 TF Flaubert 模型
    # TFFlaubertForSequenceClassification: 用于序列分类的 TF Flaubert 模型
    # TFFlaubertForTokenClassification: 用于标记分类的 TF Flaubert 模型
    # TFFlaubertModel: TF Flaubert 模型
    # TFFlaubertPreTrainedModel: TF Flaubert 预训练模型的基类
    # TFFlaubertWithLMHeadModel: 带有语言模型头部的 TF Flaubert 模型
# 如果不满足前面的条件，则导入 sys 模块
import sys
# 将当前模块的名称添加到 sys.modules 中，使用 LazyModule 进行延迟加载
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```