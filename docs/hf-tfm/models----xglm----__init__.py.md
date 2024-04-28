# `.\transformers\models\xglm\__init__.py`

```py
# 引入必要的依赖库和模块
from typing import TYPE_CHECKING

# 从 HuggingFace 工具包中导入必要的模块和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构的字典
_import_structure = {"configuration_xglm": ["XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XGLMConfig"]}

# 检查是否可用 sentencepiece 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构字典中添加 tokenization_xglm 模块
    _import_structure["tokenization_xglm"] = ["XGLMTokenizer"]

# 检查是否可用 tokenizers 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构字典中添加 tokenization_xglm_fast 模块
    _import_structure["tokenization_xglm_fast"] = ["XGLMTokenizerFast"]

# 检查是否可用 torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构字典中添加 modeling_xglm 模块
    _import_structure["modeling_xglm"] = [
        "XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XGLMForCausalLM",
        "XGLMModel",
        "XGLMPreTrainedModel",
    ]

# 检查是否可用 flax 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构字典中添加 modeling_flax_xglm 模块
    _import_structure["modeling_flax_xglm"] = [
        "FlaxXGLMForCausalLM",
        "FlaxXGLMModel",
        "FlaxXGLMPreTrainedModel",
    ]

# 检查是否可用 tensorflow 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构字典中添加 modeling_tf_xglm 模块
    _import_structure["modeling_tf_xglm"] = [
        "TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXGLMForCausalLM",
        "TFXGLMModel",
        "TFXGLMPreTrainedModel",
    ]

# 若当前环境为类型检查环境，则执行以下代码块
if TYPE_CHECKING:
    # 从 configuration_xglm 模块中导入所需类和常量
    from .configuration_xglm import XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XGLMConfig

    # 检查是否可用 sentencepiece 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 tokenization_xglm 模块中导入 XGLMTokenizer 类
        from .tokenization_xglm import XGLMTokenizer

    # 检查是否可用 tokenizers 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 tokenization_xglm_fast 模块中导入 XGLMTokenizerFast 类
        from .tokenization_xglm_fast import XGLMTokenizerFast

    # 检查是否可用 torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被捕获，则执行空语句
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有捕获到OptionalDependencyNotAvailable异常，则执行下面的代码块
    else:
        # 从modeling_xglm模块中导入指定的类和变量
        from .modeling_xglm import XGLM_PRETRAINED_MODEL_ARCHIVE_LIST, XGLMForCausalLM, XGLMModel, XGLMPreTrainedModel

    # 尝试检查是否Flax可用，如果不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被捕获，则执行空语句
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有捕获到OptionalDependencyNotAvailable异常，则执行下面的代码块
    else:
        # 从modeling_flax_xglm模块中导入指定的类
        from .modeling_flax_xglm import FlaxXGLMForCausalLM, FlaxXGLMModel, FlaxXGLMPreTrainedModel

    # 尝试检查是否TensorFlow可用，如果不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果OptionalDependencyNotAvailable异常被捕获，则执行空语句
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有捕获到OptionalDependencyNotAvailable异常，则执行下面的代码块
    else:
        # 从modeling_tf_xglm模块中导入指定的类和变量
        from .modeling_tf_xglm import TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST, TFXGLMForCausalLM, TFXGLMModel, TFXGLMPreTrainedModel
# 如果不在主程序中，则导入sys模块
import sys
# 将当前模块的结构存储在_import_structure中，并以懒加载模块的方式将当前模块替换为_LazyModule对象
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```