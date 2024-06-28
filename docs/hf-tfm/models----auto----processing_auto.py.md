# `.\models\auto\processing_auto.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" AutoProcessor class."""
# 导入必要的库和模块
import importlib  # 导入动态导入模块的功能
import inspect  # 导入用于检查对象的属性和方法的模块
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入与操作系统交互的功能
import warnings  # 导入警告处理模块
from collections import OrderedDict  # 导入有序字典类型

# 导入其他本地库和模块
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 导入动态模块加载函数
from ...feature_extraction_utils import FeatureExtractionMixin  # 导入特征提取混合类
from ...image_processing_utils import ImageProcessingMixin  # 导入图像处理混合类
from ...processing_utils import ProcessorMixin  # 导入处理混合类
from ...tokenization_utils import TOKENIZER_CONFIG_FILE  # 导入分词配置文件
from ...utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, get_file_from_repo, logging  # 导入工具函数和日志记录

# 导入本地自动化处理工厂和配置模块
from .auto_factory import _LazyAutoMapping  # 导入惰性自动映射类
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,  # 导入配置映射名称
    AutoConfig,  # 导入自动配置类
    model_type_to_module_name,  # 导入模型类型到模块名称的映射函数
    replace_list_option_in_docstrings,  # 导入替换文档字符串中列表选项的函数
)
from .feature_extraction_auto import AutoFeatureExtractor  # 导入自动特征提取器类
from .image_processing_auto import AutoImageProcessor  # 导入自动图像处理器类
from .tokenization_auto import AutoTokenizer  # 导入自动分词器类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义处理器映射名称，使用有序字典存储
PROCESSOR_MAPPING_NAMES = OrderedDict(
    # 定义一个包含处理器名称和对应处理器类的元组列表
    [
        # ('align', 'AlignProcessor')：处理器名称为 'align'，对应处理器类为 'AlignProcessor'
        ("align", "AlignProcessor"),
        # ('altclip', 'AltCLIPProcessor')：处理器名称为 'altclip'，对应处理器类为 'AltCLIPProcessor'
        ("altclip", "AltCLIPProcessor"),
        # ('bark', 'BarkProcessor')：处理器名称为 'bark'，对应处理器类为 'BarkProcessor'
        ("bark", "BarkProcessor"),
        # ('blip', 'BlipProcessor')：处理器名称为 'blip'，对应处理器类为 'BlipProcessor'
        ("blip", "BlipProcessor"),
        # ('blip-2', 'Blip2Processor')：处理器名称为 'blip-2'，对应处理器类为 'Blip2Processor'
        ("blip-2", "Blip2Processor"),
        # ('bridgetower', 'BridgeTowerProcessor')：处理器名称为 'bridgetower'，对应处理器类为 'BridgeTowerProcessor'
        ("bridgetower", "BridgeTowerProcessor"),
        # ('chinese_clip', 'ChineseCLIPProcessor')：处理器名称为 'chinese_clip'，对应处理器类为 'ChineseCLIPProcessor'
        ("chinese_clip", "ChineseCLIPProcessor"),
        # ('clap', 'ClapProcessor')：处理器名称为 'clap'，对应处理器类为 'ClapProcessor'
        ("clap", "ClapProcessor"),
        # ('clip', 'CLIPProcessor')：处理器名称为 'clip'，对应处理器类为 'CLIPProcessor'
        ("clip", "CLIPProcessor"),
        # ('clipseg', 'CLIPSegProcessor')：处理器名称为 'clipseg'，对应处理器类为 'CLIPSegProcessor'
        ("clipseg", "CLIPSegProcessor"),
        # ('clvp', 'ClvpProcessor')：处理器名称为 'clvp'，对应处理器类为 'ClvpProcessor'
        ("clvp", "ClvpProcessor"),
        # ('flava', 'FlavaProcessor')：处理器名称为 'flava'，对应处理器类为 'FlavaProcessor'
        ("flava", "FlavaProcessor"),
        # ('fuyu', 'FuyuProcessor')：处理器名称为 'fuyu'，对应处理器类为 'FuyuProcessor'
        ("fuyu", "FuyuProcessor"),
        # ('git', 'GitProcessor')：处理器名称为 'git'，对应处理器类为 'GitProcessor'
        ("git", "GitProcessor"),
        # ('groupvit', 'CLIPProcessor')：处理器名称为 'groupvit'，对应处理器类为 'CLIPProcessor'
        ("groupvit", "CLIPProcessor"),
        # ('hubert', 'Wav2Vec2Processor')：处理器名称为 'hubert'，对应处理器类为 'Wav2Vec2Processor'
        ("hubert", "Wav2Vec2Processor"),
        # ('idefics', 'IdeficsProcessor')：处理器名称为 'idefics'，对应处理器类为 'IdeficsProcessor'
        ("idefics", "IdeficsProcessor"),
        # ('instructblip', 'InstructBlipProcessor')：处理器名称为 'instructblip'，对应处理器类为 'InstructBlipProcessor'
        ("instructblip", "InstructBlipProcessor"),
        # ('kosmos-2', 'Kosmos2Processor')：处理器名称为 'kosmos-2'，对应处理器类为 'Kosmos2Processor'
        ("kosmos-2", "Kosmos2Processor"),
        # ('layoutlmv2', 'LayoutLMv2Processor')：处理器名称为 'layoutlmv2'，对应处理器类为 'LayoutLMv2Processor'
        ("layoutlmv2", "LayoutLMv2Processor"),
        # ('layoutlmv3', 'LayoutLMv3Processor')：处理器名称为 'layoutlmv3'，对应处理器类为 'LayoutLMv3Processor'
        ("layoutlmv3", "LayoutLMv3Processor"),
        # ('llava', 'LlavaProcessor')：处理器名称为 'llava'，对应处理器类为 'LlavaProcessor'
        ("llava", "LlavaProcessor"),
        # ('llava_next', 'LlavaNextProcessor')：处理器名称为 'llava_next'，对应处理器类为 'LlavaNextProcessor'
        ("llava_next", "LlavaNextProcessor"),
        # ('markuplm', 'MarkupLMProcessor')：处理器名称为 'markuplm'，对应处理器类为 'MarkupLMProcessor'
        ("markuplm", "MarkupLMProcessor"),
        # ('mctct', 'MCTCTProcessor')：处理器名称为 'mctct'，对应处理器类为 'MCTCTProcessor'
        ("mctct", "MCTCTProcessor"),
        # ('mgp-str', 'MgpstrProcessor')：处理器名称为 'mgp-str'，对应处理器类为 'MgpstrProcessor'
        ("mgp-str", "MgpstrProcessor"),
        # ('oneformer', 'OneFormerProcessor')：处理器名称为 'oneformer'，对应处理器类为 'OneFormerProcessor'
        ("oneformer", "OneFormerProcessor"),
        # ('owlv2', 'Owlv2Processor')：处理器名称为 'owlv2'，对应处理器类为 'Owlv2Processor'
        ("owlv2", "Owlv2Processor"),
        # ('owlvit', 'OwlViTProcessor')：处理器名称为 'owlvit'，对应处理器类为 'OwlViTProcessor'
        ("owlvit", "OwlViTProcessor"),
        # ('pix2struct', 'Pix2StructProcessor')：处理器名称为 'pix2struct'，对应处理器类为 'Pix2StructProcessor'
        ("pix2struct", "Pix2StructProcessor"),
        # ('pop2piano', 'Pop2PianoProcessor')：处理器名称为 'pop2piano'，对应处理器类为 'Pop2PianoProcessor'
        ("pop2piano", "Pop2PianoProcessor"),
        # ('sam', 'SamProcessor')：处理器名称为 'sam'，对应处理器类为 'SamProcessor'
        ("sam", "SamProcessor"),
        # ('seamless_m4t', 'SeamlessM4TProcessor')：处理器名称为 'seamless_m4t'，对应处理器类为 'SeamlessM4TProcessor'
        ("seamless_m4t", "SeamlessM4TProcessor"),
        # ('sew', 'Wav2Vec2Processor')：处理器名称为 'sew'，对应处理器类为 'Wav2Vec2Processor'
        ("sew", "Wav2Vec2Processor"),
        # ('sew-d', 'Wav2Vec2Processor')：处理器名称为 'sew-d'，对应处理器类为 'Wav2Vec2Processor'
        ("sew-d", "Wav2Vec2Processor"),
        # ('siglip', 'SiglipProcessor')：处理器名称为 'siglip'，对应处理器类为 'SiglipProcessor'
        ("siglip", "SiglipProcessor"),
        # ('speech_to_text', 'Speech2TextProcessor')：处理器名称为 'speech_to_text'，对应处理器类为 'Speech2TextProcessor'
        ("speech_to_text", "Speech2TextProcessor"),
        # ('speech_to_text_2', 'Speech2Text2Processor')：处理器名称为 'speech_to_text_2'，对应处理器类为 'Speech2Text2Processor'
        ("speech_to_text_2", "Speech2Text2Processor"),
        # ('speecht5', 'SpeechT5Processor')：处理器名称为 'speecht5'，对应处理器类为 'SpeechT5Processor'
        ("speecht5", "SpeechT5Processor"),
        # ('trocr', 'TrOCRProcessor')：处理器名称为 'trocr'，对应处理器类为 'TrOCRProcessor'
        ("trocr", "TrOCRProcessor"),
        # ('tvlt', 'TvltProcessor')：处理器名称为 'tvlt'，对应处理器类为 'TvltProcessor'
        ("tvlt", "TvltProcessor"),
        # ('tvp', 'TvpProcessor')：处理器名称为 'tvp'，对应处理器类为 'TvpProcessor'
        ("tvp", "TvpProcessor"),
        # ('unispeech', 'Wav2Vec2Processor')：处理器名称为 'unispeech'，对应处理器类为 'Wav2Vec2Processor'
        ("unispeech", "Wav2Vec2Processor"),
        # ('un
# 这里导入了_LazyAutoMapping和replace_list_option_in_docstrings函数，以及CONFIG_MAPPING_NAMES和PROCESSOR_MAPPING_NAMES变量。
PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)

# 根据给定的类名查找并返回对应的处理器类。
def processor_class_from_name(class_name: str):
    # 遍历PROCESSOR_MAPPING_NAMES中的模块名和处理器列表
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        # 如果class_name在当前处理器列表中
        if class_name in processors:
            # 将模块名转换为对应的模块路径
            module_name = model_type_to_module_name(module_name)

            # 动态导入transformers.models下的特定模块
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                # 返回模块中的class_name类对象
                return getattr(module, class_name)
            except AttributeError:
                # 如果属性错误，则继续下一个模块的尝试
                continue

    # 如果在PROCESSOR_MAPPING的额外内容中找到class_name对应的处理器，则返回该处理器
    for processor in PROCESSOR_MAPPING._extra_content.values():
        if getattr(processor, "__name__", None) == class_name:
            return processor

    # 如果以上都找不到，则尝试从transformers主模块中导入class_name类
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    # 如果还是找不到，则返回None
    return None


class AutoProcessor:
    r"""
    This is a generic processor class that will be instantiated as one of the processor classes of the library when
    created with the [`AutoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    # AutoProcessor的构造函数，抛出环境错误，指导使用`AutoProcessor.from_pretrained(pretrained_model_name_or_path)`方法来实例化
    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    # 装饰器，用于在文档字符串中替换列表选项，使用PROCESSOR_MAPPING_NAMES参数
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    # 静态方法，注册新的处理器类到PROCESSOR_MAPPING中
    def register(config_class, processor_class, exist_ok=False):
        """
        Register a new processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            processor_class ([`FeatureExtractorMixin`]): The processor to register.
        """
        PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)
```