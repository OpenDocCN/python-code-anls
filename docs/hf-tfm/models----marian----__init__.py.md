# `.\models\marian\__init__.py`

```py
# 版权声明和许可条款声明，指出版权归 HuggingFace Team 所有，采用 Apache License 2.0
#
# 此部分导入所需的模块和函数，从 utils 模块导入必要的依赖检查和工具函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构，用于组织导入的模块和函数
_import_structure = {
    "configuration_marian": ["MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarianConfig", "MarianOnnxConfig"],
}

# 检查 sentencepiece 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_marian 模块到导入结构中
    _import_structure["tokenization_marian"] = ["MarianTokenizer"]

# 检查 torch 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_marian 模块到导入结构中，包含多个类和常量
    _import_structure["modeling_marian"] = [
        "MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarianForCausalLM",
        "MarianModel",
        "MarianMTModel",
        "MarianPreTrainedModel",
    ]

# 检查 tensorflow 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_tf_marian 模块到导入结构中，包含多个 TensorFlow 相关的类
    _import_structure["modeling_tf_marian"] = ["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"]

# 检查 flax 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_flax_marian 模块到导入结构中，包含多个 Flax 相关的类
    _import_structure["modeling_flax_marian"] = ["FlaxMarianModel", "FlaxMarianMTModel", "FlaxMarianPreTrainedModel"]

# 如果是类型检查阶段，执行以下导入语句
if TYPE_CHECKING:
    # 从 configuration_marian 模块导入特定类和常量
    from .configuration_marian import MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP, MarianConfig, MarianOnnxConfig

    # 检查 sentencepiece 是否可用，如果可用则从 tokenization_marian 模块导入 MarianTokenizer 类
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_marian import MarianTokenizer

    # 检查 torch 是否可用，如果可用则从 modeling_marian 模块导入多个类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_marian import (
            MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarianForCausalLM,
            MarianModel,
            MarianMTModel,
            MarianPreTrainedModel,
        )

    # 不再检查 tensorflow 是否可用，因为这部分代码不在类型检查块内，避免导入 TensorFlow 相关模块
    else:
        # 如果前面的导入失败，则尝试从当前目录下导入相关模块
        from .modeling_tf_marian import TFMarianModel, TFMarianMTModel, TFMarianPreTrainedModel

    try:
        # 检查是否没有安装 Flax，如果没有安装则抛出 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果检测到 OptionalDependencyNotAvailable 异常，则不做任何处理
        pass
    else:
        # 如果没有抛出异常，则尝试从当前目录下导入 Flax 版本的 Marian 模型相关模块
        from .modeling_flax_marian import FlaxMarianModel, FlaxMarianMTModel, FlaxMarianPreTrainedModel
else:
    # 如果以上所有的条件都不满足，则执行以下代码块

    # 导入 sys 模块，用于对当前模块进行操作
    import sys

    # 将当前模块名对应的模块对象替换为一个懒加载模块对象 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```