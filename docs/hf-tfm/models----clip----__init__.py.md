# `.\transformers\models\clip\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构
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

# 检查 tokenizers 是否可用
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_clip_fast 到导入结构
    _import_structure["tokenization_clip_fast"] = ["CLIPTokenizerFast"]

# 检查 vision 是否可用
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 feature_extraction_clip 和 image_processing_clip 到导入结构
    _import_structure["feature_extraction_clip"] = ["CLIPFeatureExtractor"]
    _import_structure["image_processing_clip"] = ["CLIPImageProcessor"]

# 检查 torch 是否可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_clip 到导入结构
    _import_structure["modeling_clip"] = [
        "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CLIPModel",
        "CLIPPreTrainedModel",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "CLIPVisionModel",
        "CLIPVisionModelWithProjection",
    ]

# 检查 tensorflow 是否可用
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_tf_clip 到导入结构
    _import_structure["modeling_tf_clip"] = [
        "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCLIPModel",
        "TFCLIPPreTrainedModel",
        "TFCLIPTextModel",
        "TFCLIPVisionModel",
    ]

# 检查 flax 是否可用
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_flax_clip 到导入结构
    _import_structure["modeling_flax_clip"] = [
        "FlaxCLIPModel",
        "FlaxCLIPPreTrainedModel",
        "FlaxCLIPTextModel",
        "FlaxCLIPTextPreTrainedModel",
        "FlaxCLIPTextModelWithProjection",
        "FlaxCLIPVisionModel",
        "FlaxCLIPVisionPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从configuration_clip模块中导入相关内容，包括预训练配置文件映射、CLIP配置等
    from .configuration_clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPConfig,
        CLIPOnnxConfig,
        CLIPTextConfig,
        CLIPVisionConfig,
    )
    
    # 从processing_clip模块中导入CLIP处理器
    from .processing_clip import CLIPProcessor
    
    # 从tokenization_clip模块中导入CLIPTokenizer
    from .tokenization_clip import CLIPTokenizer
    
    # 尝试检查是否存在tokenizers库，如果不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则执行以下代码块
    else:
        # 从tokenization_clip_fast模块中导入CLIPTokenizerFast
        from .tokenization_clip_fast import CLIPTokenizerFast
    
    # 尝试检查是否存在vision库，如果不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则执行以下代码块
    else:
        # 从feature_extraction_clip模块中导入CLIPFeatureExtractor和CLIPImageProcessor
        from .feature_extraction_clip import CLIPFeatureExtractor
        from .image_processing_clip import CLIPImageProcessor
    
    # 尝试检查是否存在torch库，如果不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则执行以下代码块
    else:
        # 从modeling_clip模块中导入相关内容，包括预训练模型存档列表和CLIP模型等
        from .modeling_clip import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )
    
    # 尝试检查是否存在tf库，如果不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则执行以下代码块
    else:
        # 从modeling_tf_clip模块中导入相关内容，包括TF预训练模型存档列表和TFCLIP模型等
        from .modeling_tf_clip import (
            TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )
    
    # 尝试检查是否存在flax库，如果不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出异常，则执行以下代码块
    else:
        # 从modeling_flax_clip模块中导入相关内容，包括FlaxCLIP模型等
        from .modeling_flax_clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```