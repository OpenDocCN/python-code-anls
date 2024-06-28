# `.\models\blenderbot_small\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 从相对路径的utils模块导入所需的函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典结构，用于存储模块导入的结构信息
_import_structure = {
    "configuration_blenderbot_small": [
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlenderbotSmallConfig",
        "BlenderbotSmallOnnxConfig",
    ],
    "tokenization_blenderbot_small": ["BlenderbotSmallTokenizer"],
}

# 检查tokenizers库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tokenizers库可用，则更新_import_structure字典，添加tokenization_blenderbot_small_fast模块的导入信息
    _import_structure["tokenization_blenderbot_small_fast"] = ["BlenderbotSmallTokenizerFast"]

# 检查torch库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch库可用，则更新_import_structure字典，添加modeling_blenderbot_small模块的导入信息
    _import_structure["modeling_blenderbot_small"] = [
        "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BlenderbotSmallForCausalLM",
        "BlenderbotSmallForConditionalGeneration",
        "BlenderbotSmallModel",
        "BlenderbotSmallPreTrainedModel",
    ]

# 检查tensorflow库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tensorflow库可用，则更新_import_structure字典，添加modeling_tf_blenderbot_small模块的导入信息
    _import_structure["modeling_tf_blenderbot_small"] = [
        "TFBlenderbotSmallForConditionalGeneration",
        "TFBlenderbotSmallModel",
        "TFBlenderbotSmallPreTrainedModel",
    ]

# 检查flax库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若flax库可用，则更新_import_structure字典，添加modeling_flax_blenderbot_small模块的导入信息
    _import_structure["modeling_flax_blenderbot_small"] = [
        "FlaxBlenderbotSmallForConditionalGeneration",
        "FlaxBlenderbotSmallModel",
        "FlaxBlenderbotSmallPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相对路径的configuration_blenderbot_small模块导入所需的类和常量
    from .configuration_blenderbot_small import (
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotSmallConfig,
        BlenderbotSmallOnnxConfig,
    )
    # 从相对路径的tokenization_blenderbot_small模块导入所需的类
    from .tokenization_blenderbot_small import BlenderbotSmallTokenizer

    # 检查tokenizers库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若tokenizers库可用，则从tokenization_blenderbot_small_fast模块导入所需的类
        from .tokenization_blenderbot_small_fast import BlenderbotSmallTokenizerFast
    # 尝试检查是否安装了 Torch 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Torch 可用，则导入相关的 Blenderbot Small 模型类和常量
    else:
        from .modeling_blenderbot_small import (
            BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotSmallForCausalLM,
            BlenderbotSmallForConditionalGeneration,
            BlenderbotSmallModel,
            BlenderbotSmallPreTrainedModel,
        )

    # 尝试检查是否安装了 TensorFlow 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果 TensorFlow 可用，则导入相关的 TensorFlow 版 Blenderbot Small 模型类和常量
    else:
        from .modeling_tf_blenderbot_small import (
            TFBlenderbotSmallForConditionalGeneration,
            TFBlenderbotSmallModel,
            TFBlenderbotSmallPreTrainedModel,
        )

    # 尝试检查是否安装了 Flax 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Flax 可用，则导入相关的 Flax 版 Blenderbot Small 模型类和常量
    else:
        from .modeling_flax_blenderbot_small import (
            FlaxBlenderbotSmallForConditionalGeneration,
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallPreTrainedModel,
        )
else:
    # 如果前面的条件不满足，则执行以下代码块
    import sys
    # 导入 sys 模块，用于处理 Python 解释器的系统参数和功能

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 包装当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # __name__ 表示当前模块的名称
    # globals()["__file__"] 获取当前模块的文件路径
    # _import_structure 可能是一个函数或对象，用于指定模块的导入结构
    # module_spec=__spec__ 指定当前模块的规范对象，用于描述模块的详细信息
```