# `.\transformers\models\seamless_m4t\__init__.py`

```py
# 版权声明信息
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# 许可证信息
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 根据许可证分发的是按"原样"分发的，
# 没有任何明示或暗示的担保或条件。
# 参见许可证了解管理权限和
# 限制。

# 导入所需的类型提示
from typing import TYPE_CHECKING

# 导入一些必需的工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {
    "configuration_seamless_m4t": ["SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP", "SeamlessM4TConfig"],
    "feature_extraction_seamless_m4t": ["SeamlessM4TFeatureExtractor"],
    "processing_seamless_m4t": ["SeamlessM4TProcessor"],
}

# 检查是否可以使用 SentencePiece 相关功能
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_seamless_m4t"] = ["SeamlessM4TTokenizer"]

# 检查是否可以使用 Tokenizers 相关功能 
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_seamless_m4t_fast"] = ["SeamlessM4TTokenizerFast"]

# 检查是否可以使用 PyTorch 相关功能
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_seamless_m4t"] = [
        "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SeamlessM4TForTextToSpeech",
        "SeamlessM4TForSpeechToSpeech",
        "SeamlessM4TForTextToText",
        "SeamlessM4TForSpeechToText",
        "SeamlessM4TModel",
        "SeamlessM4TPreTrainedModel",
        "SeamlessM4TCodeHifiGan",
        "SeamlessM4THifiGan",
        "SeamlessM4TTextToUnitForConditionalGeneration",
        "SeamlessM4TTextToUnitModel",
    ]

# 根据类型提示导入相关类
if TYPE_CHECKING:
    from .configuration_seamless_m4t import SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP, SeamlessM4TConfig
    from .feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
    from .processing_seamless_m4t import SeamlessM4TProcessor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t import SeamlessM4TTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t_fast import SeamlessM4TTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass


这段代码是一个 Python 文件的开头部分,主要包含以下内容:

1. 版权声明和许可证信息。
2. 导入必需的类型提示和工具函数。
3. 定义导入结构,指明了各种组件的位置。
4. 检查是否可以使用 SentencePiece、Tokenizers 和 PyTorch 相关功能,并根据可用性动态构建导入结构。
5. 根据类型提示导入相关类。

这些代码主要用于管理和导入 Seamless M4T 模型相关的各种组件,如配置、特征提取器、处理器、Tokenizer 和模型等。它确保在使用这些组件时,会首先检查必要的依赖是否可用,并根据实际情况动态构建导入结构。
    # 尝试捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 如果捕获到 OptionalDependencyNotAvailable 异常，则忽略，不做任何处理
        pass
    # 如果未捕获到 OptionalDependencyNotAvailable 异常
    else:
        # 从 modeling_seamless_m4t 模块中导入以下内容：
        # SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST
        # SeamlessM4TCodeHifiGan
        # SeamlessM4TForSpeechToSpeech
        # SeamlessM4TForSpeechToText
        # SeamlessM4TForTextToSpeech
        # SeamlessM4TForTextToText
        # SeamlessM4THifiGan
        # SeamlessM4TModel
        # SeamlessM4TPreTrainedModel
        # SeamlessM4TTextToUnitForConditionalGeneration
        # SeamlessM4TTextToUnitModel
        from .modeling_seamless_m4t import (
            SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST,
            SeamlessM4TCodeHifiGan,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
            SeamlessM4THifiGan,
            SeamlessM4TModel,
            SeamlessM4TPreTrainedModel,
            SeamlessM4TTextToUnitForConditionalGeneration,
            SeamlessM4TTextToUnitModel,
        )
else:
    # 导入系统模块 sys
    import sys
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 类封装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```