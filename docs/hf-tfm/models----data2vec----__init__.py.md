# `.\models\data2vec\__init__.py`

```py
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

# 从 HuggingFace 的 utils 模块导入必要的异常和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义一个结构，用于存储不同模块的导入信息
_import_structure = {
    "configuration_data2vec_audio": ["DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Data2VecAudioConfig"],
    "configuration_data2vec_text": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecTextConfig",
        "Data2VecTextOnnxConfig",
    ],
    "configuration_data2vec_vision": [
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecVisionConfig",
        "Data2VecVisionOnnxConfig",
    ],
}

# 尝试检查是否 Torch 可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果 Torch 不可用，则忽略异常继续执行
    pass
else:
    # 如果 Torch 可用，则扩展 _import_structure 添加相关的模型定义
    _import_structure["modeling_data2vec_audio"] = [
        "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Data2VecAudioForAudioFrameClassification",
        "Data2VecAudioForCTC",
        "Data2VecAudioForSequenceClassification",
        "Data2VecAudioForXVector",
        "Data2VecAudioModel",
        "Data2VecAudioPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_text"] = [
        "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Data2VecTextForCausalLM",
        "Data2VecTextForMaskedLM",
        "Data2VecTextForMultipleChoice",
        "Data2VecTextForQuestionAnswering",
        "Data2VecTextForSequenceClassification",
        "Data2VecTextForTokenClassification",
        "Data2VecTextModel",
        "Data2VecTextPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_vision"] = [
        "DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Data2VecVisionForImageClassification",
        "Data2VecVisionForMaskedImageModeling",
        "Data2VecVisionForSemanticSegmentation",
        "Data2VecVisionModel",
        "Data2VecVisionPreTrainedModel",
    ]

# 如果是在类型检查模式下，导入额外的类型相关信息
if TYPE_CHECKING:
    from .configuration_data2vec_audio import DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP, Data2VecAudioConfig

# 注意：此处的代码没有返回值，仅用于定义模块导入结构和在特定条件下导入额外的类型信息
    # 从配置文件中导入文本数据2vec的预训练配置映射和相关类
    from .configuration_data2vec_text import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecTextConfig,
        Data2VecTextOnnxConfig,
    )
    # 从配置文件中导入视觉数据2vec的预训练配置映射和相关类
    from .configuration_data2vec_vision import (
        DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecVisionConfig,
        Data2VecVisionOnnxConfig,
    )
    
    try:
        # 检查是否已经安装了torch，如果没有则引发OptionalDependencyNotAvailable异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果OptionalDependencyNotAvailable异常被引发，则什么都不做，继续执行后续代码
        pass
    else:
        # 如果没有异常发生，则导入音频数据2vec的预训练模型和相关类
        from .modeling_data2vec_audio import (
            DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
        )
        # 导入文本数据2vec的预训练模型和相关类
        from .modeling_data2vec_text import (
            DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
        )
        # 导入视觉数据2vec的预训练模型和相关类
        from .modeling_data2vec_vision import (
            DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecVisionForImageClassification,
            Data2VecVisionForMaskedImageModeling,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )
    
    # 如果TensorFlow可用，导入TensorFlow版本的视觉数据2vec模型和相关类
    if is_tf_available():
        from .modeling_tf_data2vec_vision import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态操作模块信息
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```