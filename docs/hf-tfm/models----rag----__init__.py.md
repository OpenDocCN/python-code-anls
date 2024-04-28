# `.\transformers\models\rag\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入自定义异常OptionalDependencyNotAvailable
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块导入结构字典
_import_structure = {
    "configuration_rag": ["RagConfig"],  # 配置 RAG 模块
    "retrieval_rag": ["RagRetriever"],  # RAG 检索模块
    "tokenization_rag": ["RagTokenizer"],  # RAG 分词模块
}

# 检查是否可用 torch 模块
try:
    if not is_torch_available():
        # 如果不可用，则引发自定义异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 torch 模块到导入结构字典中
    _import_structure["modeling_rag"] = [
        "RagModel",
        "RagPreTrainedModel",
        "RagSequenceForGeneration",
        "RagTokenForGeneration",
    ]

# 检查是否可用 tf 模块
try:
    if not is_tf_available():
        # 如果不可用，则引发自定义异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tf 模块到导入结构字典中
    _import_structure["modeling_tf_rag"] = [
        "TFRagModel",
        "TFRagPreTrainedModel",
        "TFRagSequenceForGeneration",
        "TFRagTokenForGeneration",
    ]


# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入 RAG 配置、检索和分词模块
    from .configuration_rag import RagConfig
    from .retrieval_rag import RagRetriever
    from .tokenization_rag import RagTokenizer

    # 检查是否可用 torch 模块
    try:
        if not is_torch_available():
            # 如果不可用，则引发自定义异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 torch RAG 模型相关模块
        from .modeling_rag import RagModel, RagPreTrainedModel, RagSequenceForGeneration, RagTokenForGeneration

    # 检查是否可用 tf 模块
    try:
        if not is_tf_available():
            # 如果不可用，则引发自定义异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 tf RAG 模型相关模块
        from .modeling_tf_rag import (
            TFRagModel,
            TFRagPreTrainedModel,
            TFRagSequenceForGeneration,
            TFRagTokenForGeneration,
        )

# 如果不是类型检查环境
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为惰性模块，按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```