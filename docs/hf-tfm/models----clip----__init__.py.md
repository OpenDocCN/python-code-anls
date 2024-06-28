# `.\models\clip\__init__.py`

```py
# 版权声明和许可信息，告知此文件受 Apache 2.0 许可证保护
# 详情可参阅 http://www.apache.org/licenses/LICENSE-2.0
#
# 在这里导入必要的模块和函数
from typing import TYPE_CHECKING
# 从当前项目的工具模块中导入所需的函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义一个字典，描述导入结构和所需的组件
_import_structure = {
    "configuration_clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPConfig",
        "CLIPOnnxConfig",
        "CLIPTextConfig",
        "CLIPVisionConfig",
    ],
    "processing_clip": ["CLIPProcessor"],
    "tokenization_clip": ["CLIPTokenizer"],
}

# 尝试导入 tokenizers 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 CLIPTokenizerFast 添加到导入结构中
    _import_structure["tokenization_clip_fast"] = ["CLIPTokenizerFast"]

# 尝试导入 vision 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 CLIPFeatureExtractor 和 CLIPImageProcessor 添加到导入结构中
    _import_structure["feature_extraction_clip"] = ["CLIPFeatureExtractor"]
    _import_structure["image_processing_clip"] = ["CLIPImageProcessor"]

# 尝试导入 torch 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 CLIP 相关的模型和预训练模型添加到导入结构中
    _import_structure["modeling_clip"] = [
        "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CLIPModel",
        "CLIPPreTrainedModel",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "CLIPVisionModel",
        "CLIPVisionModelWithProjection",
        "CLIPForImageClassification",
    ]

# 尝试导入 tensorflow 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 TFCLIP 相关的模型和预训练模型添加到导入结构中
    _import_structure["modeling_tf_clip"] = [
        "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCLIPModel",
        "TFCLIPPreTrainedModel",
        "TFCLIPTextModel",
        "TFCLIPVisionModel",
    ]

# 尝试导入 flax 库，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 FlaxCLIP 相关的模型和预训练模型添加到导入结构中
    _import_structure["modeling_flax_clip"] = [
        "FlaxCLIPModel",
        "FlaxCLIPPreTrainedModel",
        "FlaxCLIPTextModel",
        "FlaxCLIPTextPreTrainedModel",
        "FlaxCLIPTextModelWithProjection",
        "FlaxCLIPVisionModel",
        "FlaxCLIPVisionPreTrainedModel",
    ]

# 如果是类型检查阶段，则需要进一步的导入，暂时略过
if TYPE_CHECKING:
    pass
    # 从本地引入所需的配置信息和类
    from .configuration_clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPConfig,
        CLIPOnnxConfig,
        CLIPTextConfig,
        CLIPVisionConfig,
    )
    # 从本地引入处理 CLIP 模型所需的处理器类
    from .processing_clip import CLIPProcessor
    # 从本地引入处理 CLIP 模型所需的分词器类
    from .tokenization_clip import CLIPTokenizer

    try:
        # 检查是否安装了 tokenizers 库，如果未安装，则抛出异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被抛出，不执行任何操作
        pass
    else:
        # 如果没有异常，则从本地引入加速版的 CLIPTokenizerFast 类
        from .tokenization_clip_fast import CLIPTokenizerFast

    try:
        # 检查是否安装了 vision 库，如果未安装，则抛出异常
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被抛出，不执行任何操作
        pass
    else:
        # 如果没有异常，则从本地引入 CLIPFeatureExtractor 和 CLIPImageProcessor 类
        from .feature_extraction_clip import CLIPFeatureExtractor
        from .image_processing_clip import CLIPImageProcessor

    try:
        # 检查是否安装了 torch 库，如果未安装，则抛出异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被抛出，不执行任何操作
        pass
    else:
        # 如果没有异常，则从本地引入相关的 CLIP 模型类和预训练模型列表
        from .modeling_clip import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPForImageClassification,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )

    try:
        # 检查是否安装了 TensorFlow 库，如果未安装，则抛出异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被抛出，不执行任何操作
        pass
    else:
        # 如果没有异常，则从本地引入 TensorFlow 版本的 CLIP 相关模型类和预训练模型列表
        from .modeling_tf_clip import (
            TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )

    try:
        # 检查是否安装了 Flax 库，如果未安装，则抛出异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被抛出，不执行任何操作
        pass
    else:
        # 如果没有异常，则从本地引入 Flax 版本的 CLIP 相关模型类和预训练模型列表
        from .modeling_flax_clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态配置模块
    import sys
    
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```