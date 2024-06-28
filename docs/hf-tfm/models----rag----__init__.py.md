# `.\models\rag\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available


# 定义模块的导入结构字典
_import_structure = {
    "configuration_rag": ["RagConfig"],  # 配置模块中的 RagConfig 类
    "retrieval_rag": ["RagRetriever"],   # 检索模块中的 RagRetriever 类
    "tokenization_rag": ["RagTokenizer"],  # 分词模块中的 RagTokenizer 类
}

# 尝试导入 Torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果异常发生，则继续执行后续代码
else:
    # 如果 Torch 可用，则更新导入结构字典中的建模模块
    _import_structure["modeling_rag"] = [
        "RagModel",
        "RagPreTrainedModel",
        "RagSequenceForGeneration",
        "RagTokenForGeneration",
    ]

# 尝试导入 TensorFlow 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果异常发生，则继续执行后续代码
else:
    # 如果 TensorFlow 可用，则更新导入结构字典中的 TensorFlow 建模模块
    _import_structure["modeling_tf_rag"] = [
        "TFRagModel",
        "TFRagPreTrainedModel",
        "TFRagSequenceForGeneration",
        "TFRagTokenForGeneration",
    ]


# 如果是类型检查模式，导入特定的模块
if TYPE_CHECKING:
    from .configuration_rag import RagConfig
    from .retrieval_rag import RagRetriever
    from .tokenization_rag import RagTokenizer

    # 尝试导入 Torch 模型模块，如果不可用则继续执行后续代码
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则导入建模相关的 Torch 模块
        from .modeling_rag import RagModel, RagPreTrainedModel, RagSequenceForGeneration, RagTokenForGeneration

    # 尝试导入 TensorFlow 模型模块，如果不可用则继续执行后续代码
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 TensorFlow 可用，则导入建模相关的 TensorFlow 模块
        from .modeling_tf_rag import (
            TFRagModel,
            TFRagPreTrainedModel,
            TFRagSequenceForGeneration,
            TFRagTokenForGeneration,
        )

# 如果不是类型检查模式，则将当前模块设置为一个延迟加载模块
else:
    import sys

    # 将当前模块注册为一个延迟加载模块，使用 LazyModule 类进行管理
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```