# `.\models\seamless_m4t\__init__.py`

```
# 版权声明和许可信息
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型检查工具
from typing import TYPE_CHECKING

# 从 utils 中导入相关工具和依赖
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典
_import_structure = {
    "configuration_seamless_m4t": ["SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP", "SeamlessM4TConfig"],
    "feature_extraction_seamless_m4t": ["SeamlessM4TFeatureExtractor"],
    "processing_seamless_m4t": ["SeamlessM4TProcessor"],
}

# 尝试导入句子分割模块，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将对应的模块添加到导入结构字典中
    _import_structure["tokenization_seamless_m4t"] = ["SeamlessM4TTokenizer"]

# 尝试导入分词器模块，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将对应的模块添加到导入结构字典中
    _import_structure["tokenization_seamless_m4t_fast"] = ["SeamlessM4TTokenizerFast"]

# 尝试导入 Torch 模块，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将对应的模块添加到导入结构字典中
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

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从对应模块中导入所需的符号
    from .configuration_seamless_m4t import SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP, SeamlessM4TConfig
    from .feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
    from .processing_seamless_m4t import SeamlessM4TProcessor

    # 尝试导入句子分割模块，如果不可用则忽略
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t import SeamlessM4TTokenizer

    # 尝试导入分词器模块，如果不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t_fast import SeamlessM4TTokenizerFast

    # 尝试导入 Torch 模块，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果发生 OptionalDependencyNotAvailable 异常，则忽略并继续执行
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则导入以下模块
    else:
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
    # 导入 sys 模块，用于动态设置当前模块
    import sys

    # 将当前模块名设置为 _LazyModule 的实例，并替换 sys.modules 中的当前模块条目
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```