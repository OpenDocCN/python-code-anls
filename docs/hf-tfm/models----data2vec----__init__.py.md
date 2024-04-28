# `.\models\data2vec\__init__.py`

```
# 版权声明及许可信息

from typing import TYPE_CHECKING  # 导入类型检查模块

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available  # 从内部模块导入一些函数和类

# 定义一个字典，存储模块的导入结构
_import_structure = {
    "configuration_data2vec_audio": ["DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Data2VecAudioConfig"],  # 音频数据模型的配置
    "configuration_data2vec_text": [  # 文本数据模型的配置
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecTextConfig",
        "Data2VecTextOnnxConfig",
    ],
    "configuration_data2vec_vision": [  # 视觉数据模型的配置
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecVisionConfig",
        "Data2VecVisionOnnxConfig",
    ],
}

# 检查是否导入了 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:  # 如果依赖项不可用，则捕获并忽略异常
    pass
else:  # 如果依赖项可用，则执行以下代码
    _import_structure["modeling_data2vec_audio"] = [
        "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",  # 音频数据模型的预训练模型档案列表
        "Data2VecAudioForAudioFrameClassification",
        "Data2VecAudioForCTC",
        "Data2VecAudioForSequenceClassification",
        "Data2VecAudioForXVector",
        "Data2VecAudioModel",
        "Data2VecAudioPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_text"] = [
        "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 文本数据模型的预训练模型档案列表
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
        "DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST",  # 视觉数据模型的预训练模型档案列表
        "Data2VecVisionForImageClassification",
        "Data2VecVisionForMaskedImageModeling",
        "Data2VecVisionForSemanticSegmentation",
        "Data2VecVisionModel",
        "Data2VecVisionPreTrainedModel",
    ]

if is_tf_available():  # 如果导入了 TensorFlow 库
    _import_structure["modeling_tf_data2vec_vision"] = [
        "TFData2VecVisionForImageClassification",
        "TFData2VecVisionForSemanticSegmentation",
        "TFData2VecVisionModel",
        "TFData2VecVisionPreTrainedModel",
    ]

if TYPE_CHECKING:  # 如果是类型检查
    from .configuration_data2vec_audio import DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP, Data2VecAudioConfig  # 导入音频数据模型的配置
    # 从configuration_data2vec_text模块中导入需要的变量和类
    from .configuration_data2vec_text import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecTextConfig,
        Data2VecTextOnnxConfig,
    )
    # 从configuration_data2vec_vision模块中导入需要的变量和类
    from .configuration_data2vec_vision import (
        DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecVisionConfig,
        Data2VecVisionOnnxConfig,
    )
    
    # 检查是否有torch可用，如果没有则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果有torch可用，则从modeling_data2vec_audio模块中导入需要的变量和类
        from .modeling_data2vec_audio import (
            DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
        )
        # 从modeling_data2vec_text模块中导入需要的变量和类
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
        # 从modeling_data2vec_vision模块中导入需要的变量和类
        from .modeling_data2vec_vision import (
            DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecVisionForImageClassification,
            Data2VecVisionForMaskedImageModeling,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )
    if is_tf_available():
        # 如果有tensorflow可用，则从modeling_tf_data2vec_vision模块中导入需要的变量和类
        from .modeling_tf_data2vec_vision import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )
# 如果不是第一次导入该模块，执行以下操作
else:
    # 导入sys模块，用于访问Python解释器的功能
    import sys
    # 使用sys.modules字典，将当前模块替换为_LazyModule对象
    # __name__是模块的名称，__file__是模块的文件名，_import_structure是模块的导入结构
    # __spec__是模块的规范对象，包含有关模块的信息
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```