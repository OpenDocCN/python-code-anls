# `.\models\xglm\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从工具包中导入相关依赖项
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包含了所需的模块和函数
_import_structure = {"configuration_xglm": ["XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XGLMConfig"]}

# 检查是否存在 sentencepiece 库，如果不存在则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 XGLMTokenizer 加入到导入结构中
    _import_structure["tokenization_xglm"] = ["XGLMTokenizer"]

# 检查是否存在 tokenizers 库，如果不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 XGLMTokenizerFast 加入到导入结构中
    _import_structure["tokenization_xglm_fast"] = ["XGLMTokenizerFast"]

# 检查是否存在 torch 库，如果不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 modeling_xglm 相关模块加入到导入结构中
    _import_structure["modeling_xglm"] = [
        "XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XGLMForCausalLM",
        "XGLMModel",
        "XGLMPreTrainedModel",
    ]

# 检查是否存在 flax 库，如果不存在则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 modeling_flax_xglm 相关模块加入到导入结构中
    _import_structure["modeling_flax_xglm"] = [
        "FlaxXGLMForCausalLM",
        "FlaxXGLMModel",
        "FlaxXGLMPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，如果不存在则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 modeling_tf_xglm 相关模块加入到导入结构中
    _import_structure["modeling_tf_xglm"] = [
        "TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXGLMForCausalLM",
        "TFXGLMModel",
        "TFXGLMPreTrainedModel",
    ]

# 如果是类型检查阶段，导入额外的类型定义和模块
if TYPE_CHECKING:
    from .configuration_xglm import XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XGLMConfig

    # 检查是否存在 sentencepiece 库，如果可用则导入 XGLMTokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xglm import XGLMTokenizer

    # 检查是否存在 tokenizers 库，如果可用则导入 XGLMTokenizerFast
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xglm_fast import XGLMTokenizerFast

    # 检查是否存在 torch 库，如果可用则导入相关模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被抛出，则忽略并继续执行
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则导入相关模块和类
    else:
        from .modeling_xglm import XGLM_PRETRAINED_MODEL_ARCHIVE_LIST, XGLMForCausalLM, XGLMModel, XGLMPreTrainedModel

    # 尝试检查是否Flax库可用，如果不可用则忽略并继续执行
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被抛出，则忽略并继续执行
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则导入相关模块和类
    else:
        from .modeling_flax_xglm import FlaxXGLMForCausalLM, FlaxXGLMModel, FlaxXGLMPreTrainedModel

    # 尝试检查是否TensorFlow库可用，如果不可用则忽略并继续执行
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被抛出，则忽略并继续执行
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则导入相关模块和类
    else:
        from .modeling_tf_xglm import (
            TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXGLMForCausalLM,
            TFXGLMModel,
            TFXGLMPreTrainedModel,
        )
else:
    # 导入内置模块 sys
    import sys

    # 将当前模块(__name__)的引用替换为一个延迟加载的模块对象 _LazyModule
    # _LazyModule 的构造参数依次为模块名称(__name__)、模块文件路径(__file__)、导入结构(_import_structure)
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```