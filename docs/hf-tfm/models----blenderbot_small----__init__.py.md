# `.\transformers\models\blenderbot_small\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖项
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入可选依赖未安装异常类
    _LazyModule,  # 导入懒加载模块
    is_flax_available,  # 检查是否可用 Flax
    is_tf_available,  # 检查是否可用 TensorFlow
    is_tokenizers_available,  # 检查是否可用 Tokenizers
    is_torch_available,  # 检查是否可用 PyTorch
)

# 定义导入结构字典，用于存储各模块的导入信息
_import_structure = {
    "configuration_blenderbot_small": [  # BlenderBot Small 配置模块
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "BlenderbotSmallConfig",  # BlenderBot Small 配置类
        "BlenderbotSmallOnnxConfig",  # BlenderBot Small ONNX 配置类
    ],
    "tokenization_blenderbot_small": [  # BlenderBot Small 分词模块
        "BlenderbotSmallTokenizer",  # BlenderBot Small 分词器类
    ],
}

# 检查 Tokenizers 是否可用，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将分词模块加入导入结构字典
    _import_structure["tokenization_blenderbot_small_fast"] = ["BlenderbotSmallTokenizerFast"]

# 检查 PyTorch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 PyTorch 模型相关模块加入导入结构字典
    _import_structure["modeling_blenderbot_small"] = [
        "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型归档列表
        "BlenderbotSmallForCausalLM",  # 用于因果语言建模的 BlenderBot Small 模型类
        "BlenderbotSmallForConditionalGeneration",  # 用于条件生成的 BlenderBot Small 模型类
        "BlenderbotSmallModel",  # BlenderBot Small 模型基类
        "BlenderbotSmallPreTrainedModel",  # BlenderBot Small 预训练模型基类
    ]

# 检查 TensorFlow 是否可用，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 TensorFlow 模型相关模块加入导入结构字典
    _import_structure["modeling_tf_blenderbot_small"] = [
        "TFBlenderbotSmallForConditionalGeneration",  # 用于条件生成的 TensorFlow 版 BlenderBot Small 模型类
        "TFBlenderbotSmallModel",  # TensorFlow 版 BlenderBot Small 模型基类
        "TFBlenderbotSmallPreTrainedModel",  # TensorFlow 版 BlenderBot Small 预训练模型基类
    ]

# 检查 Flax 是否可用，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 Flax 模型相关模块加入导入结构字典
    _import_structure["modeling_flax_blenderbot_small"] = [
        "FlaxBlenderbotSmallForConditionalGeneration",  # 用于条件生成的 Flax 版 BlenderBot Small 模型类
        "FlaxBlenderbotSmallModel",  # Flax 版 BlenderBot Small 模型基类
        "FlaxBlenderbotSmallPreTrainedModel",  # Flax 版 BlenderBot Small 预训练模型基类
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入 BlenderBot Small 配置相关类
    from .configuration_blenderbot_small import (
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置归档映射
        BlenderbotSmallConfig,  # BlenderBot Small 配置类
        BlenderbotSmallOnnxConfig,  # BlenderBot Small ONNX 配置类
    )
    # 导入 BlenderBot Small 分词器相关类
    from .tokenization_blenderbot_small import BlenderbotSmallTokenizer

    # 检查 Tokenizers 是否可用，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 BlenderBot Small 快速分词器相关类
        from .tokenization_blenderbot_small_fast import BlenderbotSmallTokenizerFast
    # 尝试检查是否有 Torch 库可用
    try:
        # 如果 Torch 库不可用，引发自定义的异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获自定义异常
    except OptionalDependencyNotAvailable:
        # 如果 Torch 库不可用，则什么都不做，继续执行后续代码
        pass
    else:
        # 如果 Torch 库可用，则导入相关模块和对象
        from .modeling_blenderbot_small import (
            BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotSmallForCausalLM,
            BlenderbotSmallForConditionalGeneration,
            BlenderbotSmallModel,
            BlenderbotSmallPreTrainedModel,
        )

    # 尝试检查是否有 TensorFlow 库可用
    try:
        # 如果 TensorFlow 库不可用，引发自定义的异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获自定义异常
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 库不可用，则什么都不做，继续执行后续代码
        pass
    else:
        # 如果 TensorFlow 库可用，则导入相关模块和对象
        from .modeling_tf_blenderbot_small import (
            TFBlenderbotSmallForConditionalGeneration,
            TFBlenderbotSmallModel,
            TFBlenderbotSmallPreTrainedModel,
        )

    # 尝试检查是否有 Flax 库可用
    try:
        # 如果 Flax 库不可用，引发自定义的异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获自定义异常
    except OptionalDependencyNotAvailable:
        # 如果 Flax 库不可用，则什么都不做，继续执行后续代码
        pass
    else:
        # 如果 Flax 库可用，则导入相关模块和对象
        from .modeling_flax_blenderbot_small import (
            FlaxBlenderbotSmallForConditionalGeneration,
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallPreTrainedModel,
        )
# 如果不在顶层模块中，则导入 sys 模块
import sys
# 使用 sys.modules 将当前模块注册为 _LazyModule 类型的模块，使得可以延迟加载
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```