# `.\pipelines\__init__.py`

```
# 导入所需的模块和函数

import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关的功能模块
import warnings  # 导入警告处理模块
from pathlib import Path  # 导入处理路径的模块 Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的功能

from huggingface_hub import model_info  # 从 huggingface_hub 模块导入 model_info

# 从不同模块中导入所需的类和函数
from ..configuration_utils import PretrainedConfig  # 导入预训练配置类
from ..dynamic_module_utils import get_class_from_dynamic_module  # 导入从动态模块获取类的函数
from ..feature_extraction_utils import PreTrainedFeatureExtractor  # 导入预训练特征提取器类
from ..image_processing_utils import BaseImageProcessor  # 导入基础图像处理器类
from ..models.auto.configuration_auto import AutoConfig  # 导入自动配置类
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor  # 导入自动特征提取映射和自动特征提取器类
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor  # 导入自动图像处理映射和自动图像处理器类
from ..models.auto.modeling_auto import AutoModelForDepthEstimation, AutoModelForImageToImage  # 导入自动深度估计模型和自动图像转换模型
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer  # 导入自动分词映射和自动分词器类
from ..tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器类
from ..utils import (
    CONFIG_NAME,  # 导入配置文件名常量
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,  # 导入 Hugging Face 协作解决端点常量
    cached_file,  # 导入缓存文件函数
    extract_commit_hash,  # 导入提取提交哈希函数
    find_adapter_config_file,  # 导入查找适配器配置文件函数
    is_kenlm_available,  # 导入检查 kenlm 是否可用函数
    is_offline_mode,  # 导入检查是否离线模式函数
    is_peft_available,  # 导入检查 peft 是否可用函数
    is_pyctcdecode_available,  # 导入检查 pyctcdecode 是否可用函数
    is_tf_available,  # 导入检查是否有 TensorFlow 函数
    is_torch_available,  # 导入检查是否有 PyTorch 函数
    logging,  # 导入日志记录模块
)

# 从不同子模块导入具体的任务流水线类
from .audio_classification import AudioClassificationPipeline  # 导入音频分类任务流水线类
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline  # 导入自动语音识别任务流水线类
from .base import (  # 从基础模块导入多个类和函数
    ArgumentHandler,  # 导入参数处理器类
    CsvPipelineDataFormat,  # 导入 CSV 数据格式流水线类
    JsonPipelineDataFormat,  # 导入 JSON 数据格式流水线类
    PipedPipelineDataFormat,  # 导入管道数据格式流水线类
    Pipeline,  # 导入任务流水线基类
    PipelineDataFormat,  # 导入任务流水线数据格式基类
    PipelineException,  # 导入任务流水线异常类
    PipelineRegistry,  # 导入任务流水线注册表类
    get_default_model_and_revision,  # 导入获取默认模型和版本函数
    infer_framework_load_model,  # 导入推断框架加载模型函数
)

# 从不同子模块导入特定任务流水线类
from .conversational import Conversation, ConversationalPipeline  # 导入对话任务流水线类
from .depth_estimation import DepthEstimationPipeline  # 导入深度估计任务流水线类
from .document_question_answering import DocumentQuestionAnsweringPipeline  # 导入文档问答任务流水线类
from .feature_extraction import FeatureExtractionPipeline  # 导入特征提取任务流水线类
from .fill_mask import FillMaskPipeline  # 导入填充掩码任务流水线类
from .image_classification import ImageClassificationPipeline  # 导入图像分类任务流水线类
from .image_feature_extraction import ImageFeatureExtractionPipeline  # 导入图像特征提取任务流水线类
from .image_segmentation import ImageSegmentationPipeline  # 导入图像分割任务流水线类
from .image_to_image import ImageToImagePipeline  # 导入图像到图像任务流水线类
from .image_to_text import ImageToTextPipeline  # 导入图像到文本任务流水线类
from .mask_generation import MaskGenerationPipeline  # 导入生成掩码任务流水线类
from .object_detection import ObjectDetectionPipeline  # 导入对象检测任务流水线类
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline  # 导入问答任务流水线相关类和函数
# 导入表格问答模块中的参数处理器和管道
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
# 导入文本到文本生成模块中的摘要生成管道、文本到文本生成管道和翻译管道
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
# 导入文本分类模块中的文本分类管道
from .text_classification import TextClassificationPipeline
# 导入文本生成模块中的文本生成管道
from .text_generation import TextGenerationPipeline
# 导入文本到音频模块中的文本到音频管道
from .text_to_audio import TextToAudioPipeline
# 导入标记分类模块中的聚合策略、命名实体识别管道、标记分类参数处理器和标记分类管道
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
# 导入视频分类模块中的视频分类管道
from .video_classification import VideoClassificationPipeline
# 导入视觉问答模块中的视觉问答管道
from .visual_question_answering import VisualQuestionAnsweringPipeline
# 导入零样本音频分类模块中的零样本音频分类管道
from .zero_shot_audio_classification import ZeroShotAudioClassificationPipeline
# 导入零样本分类模块中的零样本分类参数处理器和零样本分类管道
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
# 导入零样本图像分类模块中的零样本图像分类管道
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline
# 导入零样本目标检测模块中的零样本目标检测管道
from .zero_shot_object_detection import ZeroShotObjectDetectionPipeline

# 如果 TensorFlow 可用，则导入相关模块和类
if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForImageClassification,
        TFAutoModelForMaskedLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTableQuestionAnswering,
        TFAutoModelForTokenClassification,
        TFAutoModelForVision2Seq,
        TFAutoModelForZeroShotImageClassification,
    )

# 如果 PyTorch 可用，则导入相关模块和类
if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        AutoModel,
        AutoModelForAudioClassification,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForDocumentQuestionAnswering,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedLM,
        AutoModelForMaskGeneration,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTableQuestionAnswering,
        AutoModelForTextToSpectrogram,
        AutoModelForTextToWaveform,
        AutoModelForTokenClassification,
        AutoModelForVideoClassification,
        AutoModelForVision2Seq,
        AutoModelForVisualQuestionAnswering,
        AutoModelForZeroShotImageClassification,
        AutoModelForZeroShotObjectDetection,
    )

# 如果支持类型检查，则导入必要的模块
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_fast import PreTrainedTokenizerFast

# 获取日志记录器并命名空间化
logger = logging.get_logger(__name__)

# 注册所有支持的任务别名
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",  # 情感分析任务的别名为文本分类
    "ner": "token-classification",  # 命名实体识别任务的别名为标记分类
    "vqa": "visual-question-answering",  # 视觉问答任务的别名为视觉问答
    "text-to-speech": "text-to-audio",  # 文本转语音任务的别名为文本到音频
}
# 支持的任务及其配置信息字典，每个任务对应一个字典条目
SUPPORTED_TASKS = {
    # 音频分类任务
    "audio-classification": {
        # 实现类为 AudioClassificationPipeline
        "impl": AudioClassificationPipeline,
        # TensorFlow 空元组，无特定的 TensorFlow 模型
        "tf": (),
        # 如果 Torch 可用，包含 AutoModelForAudioClassification 类
        "pt": (AutoModelForAudioClassification,) if is_torch_available() else (),
        # 默认模型为 wav2vec2-base-superb-ks，版本为 "372e048"
        "default": {"model": {"pt": ("superb/wav2vec2-base-superb-ks", "372e048")}},
        # 类型为音频
        "type": "audio",
    },
    # 自动语音识别任务
    "automatic-speech-recognition": {
        # 实现类为 AutomaticSpeechRecognitionPipeline
        "impl": AutomaticSpeechRecognitionPipeline,
        # TensorFlow 空元组，无特定的 TensorFlow 模型
        "tf": (),
        # 如果 Torch 可用，包含 AutoModelForCTC 和 AutoModelForSpeechSeq2Seq 类
        "pt": (AutoModelForCTC, AutoModelForSpeechSeq2Seq) if is_torch_available() else (),
        # 默认模型为 wav2vec2-base-960h，版本为 "55bb623"
        "default": {"model": {"pt": ("facebook/wav2vec2-base-960h", "55bb623")}},
        # 类型为多模态
        "type": "multimodal",
    },
    # 文本转音频任务
    "text-to-audio": {
        # 实现类为 TextToAudioPipeline
        "impl": TextToAudioPipeline,
        # TensorFlow 空元组，无特定的 TensorFlow 模型
        "tf": (),
        # 如果 Torch 可用，包含 AutoModelForTextToWaveform 和 AutoModelForTextToSpectrogram 类
        "pt": (AutoModelForTextToWaveform, AutoModelForTextToSpectrogram) if is_torch_available() else (),
        # 默认模型为 bark-small，版本为 "645cfba"
        "default": {"model": {"pt": ("suno/bark-small", "645cfba")}},
        # 类型为文本
        "type": "text",
    },
    # 特征提取任务
    "feature-extraction": {
        # 实现类为 FeatureExtractionPipeline
        "impl": FeatureExtractionPipeline,
        # 如果 TensorFlow 可用，包含 TFAutoModel 类
        "tf": (TFAutoModel,) if is_tf_available() else (),
        # 如果 Torch 可用，包含 AutoModel 类
        "pt": (AutoModel,) if is_torch_available() else (),
        # 默认模型为 distilbert-base-cased，版本为 "935ac13"，同时支持 TensorFlow 和 Torch
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-cased", "935ac13"),
                "tf": ("distilbert/distilbert-base-cased", "935ac13"),
            }
        },
        # 类型为多模态
        "type": "multimodal",
    },
    # 文本分类任务
    "text-classification": {
        # 实现类为 TextClassificationPipeline
        "impl": TextClassificationPipeline,
        # 如果 TensorFlow 可用，包含 TFAutoModelForSequenceClassification 类
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        # 如果 Torch 可用，包含 AutoModelForSequenceClassification 类
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        # 默认模型为 distilbert-base-uncased-finetuned-sst-2-english，版本为 "af0f99b"，同时支持 TensorFlow 和 Torch
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
                "tf": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
            },
        },
        # 类型为文本
        "type": "text",
    },
    # 标记分类任务
    "token-classification": {
        # 实现类为 TokenClassificationPipeline
        "impl": TokenClassificationPipeline,
        # 如果 TensorFlow 可用，包含 TFAutoModelForTokenClassification 类
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        # 如果 Torch 可用，包含 AutoModelForTokenClassification 类
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        # 默认模型为 bert-large-cased-finetuned-conll03-english，版本为 "f2482bf"，同时支持 TensorFlow 和 Torch
        "default": {
            "model": {
                "pt": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
                "tf": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
            },
        },
        # 类型为文本
        "type": "text",
    },
    # 问答任务
    "question-answering": {
        # 实现类为 QuestionAnsweringPipeline
        "impl": QuestionAnsweringPipeline,
        # 如果 TensorFlow 可用，包含 TFAutoModelForQuestionAnswering 类
        "tf": (TFAutoModelForQuestionAnswering,) if is_tf_available() else (),
        # 如果 Torch 可用，包含 AutoModelForQuestionAnswering 类
        "pt": (AutoModelForQuestionAnswering,) if is_torch_available() else (),
        # 默认模型为 distilbert-base-cased-distilled-squad，版本为 "626af31"，同时支持 TensorFlow 和 Torch
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-cased-distilled-squad", "626af31"),
                "tf": ("distilbert/distilbert-base-cased-distilled-squad", "626af31"),
            },
        },
        # 类型为文本
        "type": "text",
    },
    # 定义 table-question-answering 任务配置项
    "table-question-answering": {
        # 使用 TableQuestionAnsweringPipeline 处理该任务
        "impl": TableQuestionAnsweringPipeline,
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForTableQuestionAnswering,) if is_torch_available() else (),
        # 如果有 TensorFlow 可用，则提供 TensorFlow 模型
        "tf": (TFAutoModelForTableQuestionAnswering,) if is_tf_available() else (),
        # 默认模型设定
        "default": {
            "model": {
                # Torch 模型及其版本
                "pt": ("google/tapas-base-finetuned-wtq", "69ceee2"),
                # TensorFlow 模型及其版本
                "tf": ("google/tapas-base-finetuned-wtq", "69ceee2"),
            },
        },
        # 任务类型为文本处理
        "type": "text",
    },
    
    # 定义 visual-question-answering 任务配置项
    "visual-question-answering": {
        # 使用 VisualQuestionAnsweringPipeline 处理该任务
        "impl": VisualQuestionAnsweringPipeline,
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForVisualQuestionAnswering,) if is_torch_available() else (),
        # TensorFlow 模型部分为空，表示无 TensorFlow 模型
        "tf": (),
        # 默认模型设定
        "default": {
            "model": {
                # Torch 模型及其版本
                "pt": ("dandelin/vilt-b32-finetuned-vqa", "4355f59"),
            },
        },
        # 任务类型为多模态处理
        "type": "multimodal",
    },
    
    # 定义 document-question-answering 任务配置项
    "document-question-answering": {
        # 使用 DocumentQuestionAnsweringPipeline 处理该任务
        "impl": DocumentQuestionAnsweringPipeline,
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForDocumentQuestionAnswering,) if is_torch_available() else (),
        # TensorFlow 模型部分为空，表示无 TensorFlow 模型
        "tf": (),
        # 默认模型设定
        "default": {
            "model": {
                # Torch 模型及其版本
                "pt": ("impira/layoutlm-document-qa", "52e01b3"),
            },
        },
        # 任务类型为多模态处理
        "type": "multimodal",
    },
    
    # 定义 fill-mask 任务配置项
    "fill-mask": {
        # 使用 FillMaskPipeline 处理该任务
        "impl": FillMaskPipeline,
        # 如果有 TensorFlow 可用，则提供 TensorFlow 模型
        "tf": (TFAutoModelForMaskedLM,) if is_tf_available() else (),
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForMaskedLM,) if is_torch_available() else (),
        # 默认模型设定
        "default": {
            "model": {
                # Torch 模型及其版本
                "pt": ("distilbert/distilroberta-base", "ec58a5b"),
                # TensorFlow 模型及其版本
                "tf": ("distilbert/distilroberta-base", "ec58a5b"),
            }
        },
        # 任务类型为文本处理
        "type": "text",
    },
    
    # 定义 summarization 任务配置项
    "summarization": {
        # 使用 SummarizationPipeline 处理该任务
        "impl": SummarizationPipeline,
        # 如果有 TensorFlow 可用，则提供 TensorFlow 模型
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        # 默认模型设定
        "default": {
            "model": {
                # Torch 模型及其版本
                "pt": ("sshleifer/distilbart-cnn-12-6", "a4f8f3e"),
                # TensorFlow 模型及其版本
                "tf": ("google-t5/t5-small", "d769bba")
            }
        },
        # 任务类型为文本处理
        "type": "text",
    },
    
    # translation 任务是特殊情况，参数化为 SRC 和 TGT 语言
    "translation": {
        # 使用 TranslationPipeline 处理该任务
        "impl": TranslationPipeline,
        # 如果有 TensorFlow 可用，则提供 TensorFlow 模型
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        # 如果有 Torch 可用，则提供 Torch 模型
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        # 默认模型设定
        "default": {
            # 设定不同的 SRC 和 TGT 语言对应的模型
            ("en", "fr"): {"model": {"pt": ("google-t5/t5-base", "686f1db"), "tf": ("google-t5/t5-base", "686f1db")}},
            ("en", "de"): {"model": {"pt": ("google-t5/t5-base", "686f1db"), "tf": ("google-t5/t5-base", "686f1db")}},
            ("en", "ro"): {"model": {"pt": ("google-t5/t5-base", "686f1db"), "tf": ("google-t5/t5-base", "686f1db")}},
        },
        # 任务类型为文本处理
        "type": "text",
    },
    "text2text-generation": {  # 文本到文本生成任务配置
        "impl": Text2TextGenerationPipeline,  # 使用 Text2TextGenerationPipeline 类实现
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),  # 如果 TensorFlow 可用，使用 TFAutoModelForSeq2SeqLM 模型
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModelForSeq2SeqLM 模型
        "default": {"model": {"pt": ("google-t5/t5-base", "686f1db"), "tf": ("google-t5/t5-base", "686f1db")}},  # 默认模型配置
        "type": "text",  # 任务类型为文本生成
    },
    "text-generation": {  # 文本生成任务配置
        "impl": TextGenerationPipeline,  # 使用 TextGenerationPipeline 类实现
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),  # 如果 TensorFlow 可用，使用 TFAutoModelForCausalLM 模型
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModelForCausalLM 模型
        "default": {"model": {"pt": ("openai-community/gpt2", "6c0e608"), "tf": ("openai-community/gpt2", "6c0e608")}},  # 默认模型配置
        "type": "text",  # 任务类型为文本生成
    },
    "zero-shot-classification": {  # 零样本分类任务配置
        "impl": ZeroShotClassificationPipeline,  # 使用 ZeroShotClassificationPipeline 类实现
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),  # 如果 TensorFlow 可用，使用 TFAutoModelForSequenceClassification 模型
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModelForSequenceClassification 模型
        "default": {  # 默认配置
            "model": {  # 模型配置
                "pt": ("facebook/bart-large-mnli", "c626438"),  # PyTorch 使用 Facebook BART 大型 MNLI 模型
                "tf": ("FacebookAI/roberta-large-mnli", "130fb28"),  # TensorFlow 使用 Facebook RoBERTa 大型 MNLI 模型
            },
            "config": {  # 额外配置
                "pt": ("facebook/bart-large-mnli", "c626438"),  # PyTorch 使用相同的 BART 大型 MNLI 模型
                "tf": ("FacebookAI/roberta-large-mnli", "130fb28"),  # TensorFlow 使用相同的 RoBERTa 大型 MNLI 模型
            },
        },
        "type": "text",  # 任务类型为文本分类
    },
    "zero-shot-image-classification": {  # 零样本图像分类任务配置
        "impl": ZeroShotImageClassificationPipeline,  # 使用 ZeroShotImageClassificationPipeline 类实现
        "tf": (TFAutoModelForZeroShotImageClassification,) if is_tf_available() else (),  # 如果 TensorFlow 可用，使用 TFAutoModelForZeroShotImageClassification 模型
        "pt": (AutoModelForZeroShotImageClassification,) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModelForZeroShotImageClassification 模型
        "default": {  # 默认配置
            "model": {  # 模型配置
                "pt": ("openai/clip-vit-base-patch32", "f4881ba"),  # PyTorch 使用 OpenAI CLIP-ViT Base 模型
                "tf": ("openai/clip-vit-base-patch32", "f4881ba"),  # TensorFlow 使用相同的 CLIP-ViT Base 模型
            }
        },
        "type": "multimodal",  # 任务类型为多模态
    },
    "zero-shot-audio-classification": {  # 零样本音频分类任务配置
        "impl": ZeroShotAudioClassificationPipeline,  # 使用 ZeroShotAudioClassificationPipeline 类实现
        "tf": (),  # TensorFlow 不适用于此任务，设为空元组
        "pt": (AutoModel,) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModel 模型
        "default": {  # 默认配置
            "model": {  # 模型配置
                "pt": ("laion/clap-htsat-fused", "973b6e5"),  # PyTorch 使用 Laion CLAP-HTSAT-Fused 模型
            }
        },
        "type": "multimodal",  # 任务类型为多模态
    },
    "conversational": {  # 对话生成任务配置
        "impl": ConversationalPipeline,  # 使用 ConversationalPipeline 类实现
        "tf": (TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM) if is_tf_available() else (),  # 如果 TensorFlow 可用，使用 TFAutoModelForSeq2SeqLM 和 TFAutoModelForCausalLM 模型
        "pt": (AutoModelForSeq2SeqLM, AutoModelForCausalLM) if is_torch_available() else (),  # 如果 PyTorch 可用，使用 AutoModelForSeq2SeqLM 和 AutoModelForCausalLM 模型
        "default": {  # 默认配置
            "model": {"pt": ("microsoft/DialoGPT-medium", "8bada3b"), "tf": ("microsoft/DialoGPT-medium", "8bada3b")}  # 使用 Microsoft DialoGPT 中等模型
        },
        "type": "text",  # 任务类型为文本生成
    },
    {
        # 图像分类任务的配置
        "image-classification": {
            # 实现图像分类任务的流水线
            "impl": ImageClassificationPipeline,
            # TensorFlow 可用时的模型配置，包含自动图像分类模型
            "tf": (TFAutoModelForImageClassification,) if is_tf_available() else (),
            # PyTorch 可用时的模型配置，包含自动图像分类模型
            "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
            # 默认模型配置
            "default": {
                "model": {
                    # PyTorch 的默认模型为 VIT-base-patch16-224，版本为 5dca96d
                    "pt": ("google/vit-base-patch16-224", "5dca96d"),
                    # TensorFlow 的默认模型为 VIT-base-patch16-224，版本为 5dca96d
                    "tf": ("google/vit-base-patch16-224", "5dca96d"),
                }
            },
            # 任务类型为图像处理
            "type": "image",
        },
        # 图像特征提取任务的配置
        "image-feature-extraction": {
            # 实现图像特征提取任务的流水线
            "impl": ImageFeatureExtractionPipeline,
            # TensorFlow 可用时的模型配置，包含自动模型
            "tf": (TFAutoModel,) if is_tf_available() else (),
            # PyTorch 可用时的模型配置，包含自动模型
            "pt": (AutoModel,) if is_torch_available() else (),
            # 默认模型配置
            "default": {
                "model": {
                    # PyTorch 的默认模型为 VIT-base-patch16-224，版本为 29e7a1e183
                    "pt": ("google/vit-base-patch16-224", "29e7a1e183"),
                    # TensorFlow 的默认模型为 VIT-base-patch16-224，版本为 29e7a1e183
                    "tf": ("google/vit-base-patch16-224", "29e7a1e183"),
                }
            },
            # 任务类型为图像处理
            "type": "image",
        },
        # 图像分割任务的配置
        "image-segmentation": {
            # 实现图像分割任务的流水线
            "impl": ImageSegmentationPipeline,
            # TensorFlow 可用时的模型配置为空元组，表示不可用
            "tf": (),
            # PyTorch 可用时的模型配置，包含自动目标分割和语义分割模型
            "pt": (AutoModelForImageSegmentation, AutoModelForSemanticSegmentation) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 DETR-resnet-50-panoptic，版本为 fc15262
            "default": {"model": {"pt": ("facebook/detr-resnet-50-panoptic", "fc15262")}},
            # 任务类型为多模态处理
            "type": "multimodal",
        },
        # 图像到文本任务的配置
        "image-to-text": {
            # 实现图像到文本任务的流水线
            "impl": ImageToTextPipeline,
            # TensorFlow 可用时的模型配置，包含自动视觉到序列模型
            "tf": (TFAutoModelForVision2Seq,) if is_tf_available() else (),
            # PyTorch 可用时的模型配置，包含自动视觉到序列模型
            "pt": (AutoModelForVision2Seq,) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 VIT-GPT2-COCO-en，版本为 65636df
            "default": {
                "model": {
                    "pt": ("ydshieh/vit-gpt2-coco-en", "65636df"),
                    "tf": ("ydshieh/vit-gpt2-coco-en", "65636df"),
                }
            },
            # 任务类型为多模态处理
            "type": "multimodal",
        },
        # 目标检测任务的配置
        "object-detection": {
            # 实现目标检测任务的流水线
            "impl": ObjectDetectionPipeline,
            # TensorFlow 可用时的模型配置为空元组，表示不可用
            "tf": (),
            # PyTorch 可用时的模型配置，包含自动目标检测模型
            "pt": (AutoModelForObjectDetection,) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 DETR-resnet-50，版本为 2729413
            "default": {"model": {"pt": ("facebook/detr-resnet-50", "2729413")}},
            # 任务类型为多模态处理
            "type": "multimodal",
        },
        # 零样本目标检测任务的配置
        "zero-shot-object-detection": {
            # 实现零样本目标检测任务的流水线
            "impl": ZeroShotObjectDetectionPipeline,
            # TensorFlow 可用时的模型配置为空元组，表示不可用
            "tf": (),
            # PyTorch 可用时的模型配置，包含自动零样本目标检测模型
            "pt": (AutoModelForZeroShotObjectDetection,) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 OWL-ViT-base-patch32，版本为 17740e1
            "default": {"model": {"pt": ("google/owlvit-base-patch32", "17740e1")}},
            # 任务类型为多模态处理
            "type": "multimodal",
        },
        # 深度估计任务的配置
        "depth-estimation": {
            # 实现深度估计任务的流水线
            "impl": DepthEstimationPipeline,
            # TensorFlow 可用时的模型配置为空元组，表示不可用
            "tf": (),
            # PyTorch 可用时的模型配置，包含自动深度估计模型
            "pt": (AutoModelForDepthEstimation,) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 DPT-large，版本为 e93beec
            "default": {"model": {"pt": ("Intel/dpt-large", "e93beec")}},
            # 任务类型为图像处理
            "type": "image",
        },
        # 视频分类任务的配置
        "video-classification": {
            # 实现视频分类任务的流水线
            "impl": VideoClassificationPipeline,
            # TensorFlow 可用时的模型配置为空元组，表示不可用
            "tf": (),
            # PyTorch 可用时的模型配置，包含自动视频分类模型
            "pt": (AutoModelForVideoClassification,) if is_torch_available() else (),
            # 默认模型配置，PyTorch 的默认模型为 VideoMae-base-finetuned-kinetics，版本为 4800870
            "default": {"model": {"pt": ("MCG-NJU/videomae-base-finetuned-kinetics", "4800870")}},
            # 任务类型为视频处理
            "type": "video",
        },
    }
    # "mask-generation"任务配置
    "mask-generation": {
        # 使用MaskGenerationPipeline作为实现
        "impl": MaskGenerationPipeline,
        # TensorFlow环境下不需要额外模型
        "tf": (),
        # 如果有PyTorch环境，使用AutoModelForMaskGeneration作为模型
        "pt": (AutoModelForMaskGeneration,) if is_torch_available() else (),
        # 默认模型配置，使用Facebook的"facebook/sam-vit-huge"模型
        "default": {"model": {"pt": ("facebook/sam-vit-huge", "997b15")}},
        # 任务类型为多模态处理
        "type": "multimodal",
    },
    
    # "image-to-image"任务配置
    "image-to-image": {
        # 使用ImageToImagePipeline作为实现
        "impl": ImageToImagePipeline,
        # TensorFlow环境下不需要额外模型
        "tf": (),
        # 如果有PyTorch环境，使用AutoModelForImageToImage作为模型
        "pt": (AutoModelForImageToImage,) if is_torch_available() else (),
        # 默认模型配置，使用"caidas/swin2SR-classical-sr-x2-64"模型
        "default": {"model": {"pt": ("caidas/swin2SR-classical-sr-x2-64", "4aaedcb")}},
        # 任务类型为图像处理
        "type": "image",
    },
}

# 初始化空集合，用于存放没有特征提取器的任务
NO_FEATURE_EXTRACTOR_TASKS = set()
# 初始化空集合，用于存放没有图像处理器的任务
NO_IMAGE_PROCESSOR_TASKS = set()
# 初始化空集合，用于存放没有分词器的任务
NO_TOKENIZER_TASKS = set()

# 下面这些模型配置是特殊的，它们是通用的，适用于多种任务，意味着任何分词器/特征提取器都可能用于给定的模型，
# 因此我们无法使用静态定义的 TOKENIZER_MAPPING 和 FEATURE_EXTRACTOR_MAPPING 来查看模型是否定义了这些对象。
MULTI_MODEL_AUDIO_CONFIGS = {"SpeechEncoderDecoderConfig"}
MULTI_MODEL_VISION_CONFIGS = {"VisionEncoderDecoderConfig", "VisionTextDualEncoderConfig"}

# 遍历 SUPPORTED_TASKS 中的任务及其值
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        # 如果任务类型为文本，将其添加到没有特征提取器的任务集合中
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        # 如果任务类型为文本，将其添加到没有图像处理器的任务集合中
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] in {"image", "video"}:
        # 如果任务类型为图像或视频，将其添加到没有分词器的任务集合中
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] in {"audio"}:
        # 如果任务类型为音频，将其添加到没有分词器的任务集合中
        NO_TOKENIZER_TASKS.add(task)
        # 如果任务类型为音频，将其添加到没有图像处理器的任务集合中
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] != "multimodal":
        # 如果任务类型不是多模态，抛出异常，说明不支持的任务类型
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")

# 创建管道注册对象，使用支持的任务和任务别名作为参数
PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)


def get_supported_tasks() -> List[str]:
    """
    返回支持的任务列表。
    """
    return PIPELINE_REGISTRY.get_supported_tasks()


def get_task(model: str, token: Optional[str] = None, **deprecated_kwargs) -> str:
    """
    根据模型和令牌返回任务字符串，支持废弃的参数。
    """
    # 弹出废弃的参数 use_auth_token，并赋值给 use_auth_token
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    
    # 如果 use_auth_token 不为 None，发出废弃警告信息
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 不为 None，引发值错误，说明同时指定了 token 和 use_auth_token 参数
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 use_auth_token 赋值给 token
        token = use_auth_token

    # 如果处于离线模式，引发运行时错误，说明不能在离线模式下自动推断任务
    if is_offline_mode():
        raise RuntimeError("You cannot infer task automatically within `pipeline` when using offline mode")
    
    # 尝试获取模型信息，如果出现异常，引发运行时错误
    try:
        info = model_info(model, token=token)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    
    # 如果信息中没有 pipeline_tag 属性，引发运行时错误，说明模型没有正确设置 pipeline_tag 来自动推断任务
    if not info.pipeline_tag:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    
    # 如果 info 的 library_name 属性不是 "transformers"，引发运行时错误，说明该模型应该使用其他库而不是 transformers
    if getattr(info, "library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {info.library_name} not with transformers")
    
    # 返回从 info 中推断的 pipeline_tag 作为任务
    task = info.pipeline_tag
    return task


def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    检查传入的任务字符串，验证其正确性，并返回默认的管道和模型类，以及默认模型（如果存在）。
    """
    Args:
        task (`str`):
            指定要返回的流水线的任务。目前接受的任务包括：

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-feature-extraction"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"`（别名为 `"sentiment-analysis"` 可用）
            - `"text-generation"`
            - `"text-to-audio"`（别名为 `"text-to-speech"` 可用）
            - `"token-classification"`（别名为 `"ner"` 可用）
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"`（别名为 `"vqa"` 可用）
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        返回一个元组，包含标准化后的任务名称 `normalized_task`（去除了别名和选项）、任务默认设置字典 `task_defaults`，以及一些额外的任务选项 `task_options`（对于像 "translation_XX_to_YY" 这样带参数的任务）。

    """
    return PIPELINE_REGISTRY.check_task(task)
def clean_custom_task(task_info):
    import transformers  # 导入transformers库

    # 检查任务信息中是否包含实现信息，如果没有则抛出运行时错误
    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    
    pt_class_names = task_info.get("pt", ())  # 获取pt_class_names，如果不存在则默认为空元组
    if isinstance(pt_class_names, str):
        pt_class_names = [pt_class_names]  # 如果pt_class_names是字符串，转换为列表
    # 将pt_class_names中每个类名对应的类对象存入task_info["pt"]中
    task_info["pt"] = tuple(getattr(transformers, c) for c in pt_class_names)
    
    tf_class_names = task_info.get("tf", ())  # 获取tf_class_names，如果不存在则默认为空元组
    if isinstance(tf_class_names, str):
        tf_class_names = [tf_class_names]  # 如果tf_class_names是字符串，转换为列表
    # 将tf_class_names中每个类名对应的类对象存入task_info["tf"]中
    task_info["tf"] = tuple(getattr(transformers, c) for c in tf_class_names)
    
    return task_info, None  # 返回更新后的task_info和None作为第二个返回值


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```"""
    if model_kwargs is None:
        model_kwargs = {}
    
    # 确保只将use_auth_token作为一个关键字参数传递（以前可以将其传递给model_kwargs，为了保持向后兼容性）
    use_auth_token = model_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 参数不为 None，则发出警告，提醒该参数在 Transformers v5 版本中将被移除，建议使用 `token` 参数代替
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 参数也不为 None，则抛出 ValueError，说明同时指定了 `token` 和 `use_auth_token` 参数，应只设置 `token` 参数
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 use_auth_token 的值赋给 token 参数
        token = use_auth_token

    # 从 kwargs 字典中弹出 code_revision 和 _commit_hash 参数的值
    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)

    # 创建 hub_kwargs 字典，用于存储 revision、token、trust_remote_code 和 _commit_hash 参数的值
    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    # 如果既未指定 task 参数也未指定 model 参数，则抛出 RuntimeError，说明无法实例化 Pipeline
    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    # 如果未指定 model 参数但指定了 tokenizer 参数，则抛出 RuntimeError，说明无法实例化 Pipeline
    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )

    # 如果未指定 model 参数但指定了 feature_extractor 参数，则抛出 RuntimeError，说明无法实例化 Pipeline
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
            " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
            " or a path/identifier to a pretrained model when providing feature_extractor."
        )

    # 如果 model 参数的类型是 Path 对象，则将其转换为字符串类型
    if isinstance(model, Path):
        model = str(model)

    # 如果 commit_hash 参数为 None
    if commit_hash is None:
        # 预先训练的模型名或路径名为 None
        pretrained_model_name_or_path = None
        # 如果 config 参数是字符串类型，则将其赋值给 pretrained_model_name_or_path
        if isinstance(config, str):
            pretrained_model_name_or_path = config
        # 如果 config 参数为 None 且 model 参数为字符串类型，则将 model 参数赋值给 pretrained_model_name_or_path
        elif config is None and isinstance(model, str):
            pretrained_model_name_or_path = model

        # 如果 config 参数不是 PretrainedConfig 类型且 pretrained_model_name_or_path 不为 None
        if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
            # 首先调用配置文件 (可能不存在) 获取 commit hash
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            # 从配置文件中提取 commit hash，更新 hub_kwargs 中的 _commit_hash 参数
            hub_kwargs["_commit_hash"] = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            # 否则，从 config 对象中获取 _commit_hash 属性的值，更新 hub_kwargs 中的 _commit_hash 参数
            hub_kwargs["_commit_hash"] = getattr(config, "_commit_hash", None)

    # 配置是最原始的信息项。
    # 如有需要则实例化配置
    # 如果配置是字符串，则根据预训练模型配置自动生成配置对象
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # 更新 hub_kwargs 中的 _commit_hash
        hub_kwargs["_commit_hash"] = config._commit_hash
    # 如果配置为 None 且模型路径是字符串
    elif config is None and isinstance(model, str):
        # 如果 PEFT 可用，检查模型路径中是否存在适配器文件
        if is_peft_available():
            # 在模型路径中查找适配器配置文件，不包括 `trust_remote_code` 参数
            _hub_kwargs = {k: v for k, v in hub_kwargs.items() if k != "trust_remote_code"}
            maybe_adapter_path = find_adapter_config_file(
                model,
                token=hub_kwargs["token"],
                revision=hub_kwargs["revision"],
                _commit_hash=hub_kwargs["_commit_hash"],
            )

            # 如果找到适配器路径，则加载适配器配置文件中的基础模型名称或路径
            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                    model = adapter_config["base_model_name_or_path"]

        # 根据模型路径加载自动配置对象
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # 更新 hub_kwargs 中的 _commit_hash
        hub_kwargs["_commit_hash"] = config._commit_hash

    # 自定义任务字典初始化为空
    custom_tasks = {}
    # 如果配置对象不为空且存在自定义流水线，则获取自定义流水线任务
    if config is not None and len(getattr(config, "custom_pipelines", {})) > 0:
        custom_tasks = config.custom_pipelines
        # 如果任务为 None 且不禁止远程代码，则尝试自动推断任务
        if task is None and trust_remote_code is not False:
            # 如果只有一个自定义任务，则自动选择该任务
            if len(custom_tasks) == 1:
                task = list(custom_tasks.keys())[0]
            else:
                # 如果存在多个自定义任务，则抛出运行时错误，要求手动选择任务
                raise RuntimeError(
                    "We can't infer the task automatically for this model as there are multiple tasks available. Pick "
                    f"one in {', '.join(custom_tasks.keys())}"
                )

    # 如果任务仍为 None 且模型不为空，则尝试获取任务
    if task is None and model is not None:
        # 如果模型不是字符串，则抛出运行时错误
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`. "
                f"{model} is not a valid model_id."
            )
        # 根据模型 ID 和 token 获取任务
        task = get_task(model, token)

    # 获取任务后的处理流程
    if task in custom_tasks:
        # 标准化任务名称
        normalized_task = task
        # 清理自定义任务，获取目标任务和任务选项
        targeted_task, task_options = clean_custom_task(custom_tasks[task])
        # 如果未指定流水线类，则根据情况抛出 ValueError
        if pipeline_class is None:
            # 如果不信任远程代码，则要求设置 `trust_remote_code=True` 以移除错误
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
            # 从动态模块中获取类引用
            class_ref = targeted_task["impl"]
            pipeline_class = get_class_from_dynamic_module(
                class_ref,
                model,
                code_revision=code_revision,
                **hub_kwargs,
            )
    else:
        # 检查任务并返回标准化的任务、目标任务和任务选项
        normalized_task, targeted_task, task_options = check_task(task)
        # 如果未指定流水线类，则使用目标任务的实现类作为默认流水线类
        if pipeline_class is None:
            pipeline_class = targeted_task["impl"]

    # 如果未提供模型，则使用任务的默认模型、配置和分词器
    if model is None:
        # 获取任务的默认模型及其修订版本
        model, default_revision = get_default_model_and_revision(targeted_task, framework, task_options)
        # 如果未指定修订版本，则使用默认修订版本
        revision = revision if revision is not None else default_revision
        # 记录警告信息，指出未提供模型，使用默认模型和修订版本
        logger.warning(
            f"No model was supplied, defaulted to {model} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        # 如果未提供配置且模型名称为字符串，则从预训练模型中创建配置对象
        if config is None and isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **model_kwargs)
            # 将配置的提交哈希记录到 hub_kwargs 中
            hub_kwargs["_commit_hash"] = config._commit_hash

    # 如果设备映射不为空，则处理相关参数
    if device_map is not None:
        # 如果模型参数中已包含 device_map，抛出错误
        if "device_map" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... device_map=..., model_kwargs={"device_map":...})` as those'
                " arguments might conflict, use only one.)"
            )
        # 如果同时指定了 device 和 device_map，则发出警告
        if device is not None:
            logger.warning(
                "Both `device` and `device_map` are specified. `device` will override `device_map`. You"
                " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`."
            )
        # 将 device_map 添加到模型参数中
        model_kwargs["device_map"] = device_map

    # 如果 torch 数据类型不为空，则处理相关参数
    if torch_dtype is not None:
        # 如果模型参数中已包含 torch_dtype，抛出错误
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        # 如果 torch_dtype 是字符串且存在于 torch 模块中，则转换成相应的 torch 数据类型
        if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
            torch_dtype = getattr(torch, torch_dtype)
        # 将 torch_dtype 添加到模型参数中
        model_kwargs["torch_dtype"] = torch_dtype

    # 如果模型名称是字符串，则推断框架并加载模型
    if isinstance(model, str) or framework is None:
        # 定义模型类别（TensorFlow 或 PyTorch）并根据模型加载相应的框架和模型
        model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
        framework, model = infer_framework_load_model(
            model,
            model_classes=model_classes,
            config=config,
            framework=framework,
            task=task,
            **hub_kwargs,
            **model_kwargs,
        )

    # 获取模型的配置信息
    model_config = model.config
    # 将配置的提交哈希记录到 hub_kwargs 中
    hub_kwargs["_commit_hash"] = model.config._commit_hash
    # 判断是否需要加载分词器
    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    # 判断是否需要加载特征提取器
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None
    # 检查是否需要加载图像处理器，条件为模型配置在图像处理器映射中或者图像处理器不为空
    load_image_processor = type(model_config) in IMAGE_PROCESSOR_MAPPING or image_processor is not None

    # 如果传入的`model`（`PretrainedModel`的实例而不是字符串），并且`image_processor`或`feature_extractor`为空，
    # 则加载将失败。这在某些视觉任务中特别发生，当使用`pipeline()`函数时传入`model`和其中一个`image_processor`或`feature_extractor`时。
    # TODO: 我们需要使`NO_IMAGE_PROCESSOR_TASKS`和`NO_FEATURE_EXTRACTOR_TASKS`更加健壮，以避免这种问题。
    # 这段代码仅用于临时使CI通过。
    if load_image_processor and load_feature_extractor:
        load_feature_extractor = False

    # 如果`tokenizer`为空，并且不需要加载`tokenizer`，并且`normalized_task`不在`NO_TOKENIZER_TASKS`中，
    # 并且`model_config`的类名在`MULTI_MODEL_AUDIO_CONFIGS`或`MULTI_MODEL_VISION_CONFIGS`中，
    # 则尝试强制加载`tokenizer`。
    if (
        tokenizer is None
        and not load_tokenizer
        and normalized_task not in NO_TOKENIZER_TASKS
        # 使用类名来避免导入真实类。
        and (
            model_config.__class__.__name__ in MULTI_MODEL_AUDIO_CONFIGS
            or model_config.__class__.__name__ in MULTI_MODEL_VISION_CONFIGS
        )
    ):
        load_tokenizer = True

    # 如果`image_processor`为空，并且不需要加载`image_processor`，并且`normalized_task`不在`NO_IMAGE_PROCESSOR_TASKS`中，
    # 并且`model_config`的类名在`MULTI_MODEL_VISION_CONFIGS`中，
    # 则尝试强制加载`image_processor`。
    if (
        image_processor is None
        and not load_image_processor
        and normalized_task not in NO_IMAGE_PROCESSOR_TASKS
        # 使用类名来避免导入真实类。
        and model_config.__class__.__name__ in MULTI_MODEL_VISION_CONFIGS
    ):
        load_image_processor = True

    # 如果`feature_extractor`为空，并且不需要加载`feature_extractor`，并且`normalized_task`不在`NO_FEATURE_EXTRACTOR_TASKS`中，
    # 并且`model_config`的类名在`MULTI_MODEL_AUDIO_CONFIGS`中，
    # 则尝试强制加载`feature_extractor`。
    if (
        feature_extractor is None
        and not load_feature_extractor
        and normalized_task not in NO_FEATURE_EXTRACTOR_TASKS
        # 使用类名来避免导入真实类。
        and model_config.__class__.__name__ in MULTI_MODEL_AUDIO_CONFIGS
    ):
        load_feature_extractor = True

    # 如果任务在`NO_TOKENIZER_TASKS`中，则不需要加载`tokenizer`。
    if task in NO_TOKENIZER_TASKS:
        load_tokenizer = False

    # 如果任务在`NO_FEATURE_EXTRACTOR_TASKS`中，则不需要加载`feature_extractor`。
    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False

    # 如果任务在`NO_IMAGE_PROCESSOR_TASKS`中，则不需要加载`image_processor`。
    if task in NO_IMAGE_PROCESSOR_TASKS:
        load_image_processor = False
    # 如果需要加载分词器
    if load_tokenizer:
        # 尝试根据模型名称或配置名称推断分词器（如果提供的话）
        if tokenizer is None:
            # 如果 model_name 是字符串，则尝试使用其作为分词器
            if isinstance(model_name, str):
                tokenizer = model_name
            # 如果 config 是字符串，则尝试使用其作为分词器
            elif isinstance(config, str):
                tokenizer = config
            else:
                # 在这里无法猜测应该使用哪个分词器
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # 如果需要，实例化分词器
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # 对于元组，格式为（分词器名称，{kwargs}）
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs.copy()
                tokenizer_kwargs.pop("torch_dtype", None)

            # 根据给定的参数创建 AutoTokenizer 实例
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )

    # 如果需要加载图像处理器
    if load_image_processor:
        # 尝试根据模型名称或配置名称推断图像处理器（如果提供的话）
        if image_processor is None:
            # 如果 model_name 是字符串，则尝试使用其作为图像处理器
            if isinstance(model_name, str):
                image_processor = model_name
            # 如果 config 是字符串，则尝试使用其作为图像处理器
            elif isinstance(config, str):
                image_processor = config
            # 为了向后兼容，如果 feature_extractor 是 BaseImageProcessor 的实例，则使用其作为图像处理器
            elif feature_extractor is not None and isinstance(feature_extractor, BaseImageProcessor):
                image_processor = feature_extractor
            else:
                # 在这里无法猜测应该使用哪个图像处理器
                raise Exception(
                    "Impossible to guess which image processor to use. "
                    "Please provide a PreTrainedImageProcessor class or a path/identifier "
                    "to a pretrained image processor."
                )

        # 如果需要，实例化图像处理器
        if isinstance(image_processor, (str, tuple)):
            # 根据给定的参数创建 AutoImageProcessor 实例
            image_processor = AutoImageProcessor.from_pretrained(
                image_processor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
    # 如果需要加载特征提取器
    if load_feature_extractor:
        # 尝试从模型名称或配置名称（如果是字符串）推断特征提取器
        if feature_extractor is None:
            # 如果模型名称是字符串，则将其作为特征提取器
            if isinstance(model_name, str):
                feature_extractor = model_name
            # 如果配置是字符串，则将其作为特征提取器
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # 在此无法猜测正确的特征提取器
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # 如果特征提取器是字符串或元组，则实例化特征提取器
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )

            # 如果特征提取器包含语言模型且模型名称是字符串
            if (
                feature_extractor._processor_class
                and feature_extractor._processor_class.endswith("WithLM")
                and isinstance(model_name, str)
            ):
                try:
                    import kenlm  # 触发 `ImportError` 如果未安装
                    from pyctcdecode import BeamSearchDecoderCTC

                    # 如果模型名称是目录或文件
                    if os.path.isdir(model_name) or os.path.isfile(model_name):
                        decoder = BeamSearchDecoderCTC.load_from_dir(model_name)
                    else:
                        # 语言模型的全局路径及字母表文件名
                        language_model_glob = os.path.join(
                            BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                        )
                        alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                        allow_patterns = [language_model_glob, alphabet_filename]
                        # 从 HF Hub 加载模型名称对应的解码器
                        decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name, allow_patterns=allow_patterns)

                    # 将解码器加入参数中
                    kwargs["decoder"] = decoder
                except ImportError as e:
                    # 如果无法加载 `decoder`，则记录警告信息，并默认使用原始 CTC
                    logger.warning(f"Could not load the `decoder` for {model_name}. Defaulting to raw CTC. Error: {e}")
                    # 如果未安装 kenlm
                    if not is_kenlm_available():
                        logger.warning("Try to install `kenlm`: `pip install kenlm")

                    # 如果未安装 pyctcdecode
                    if not is_pyctcdecode_available():
                        logger.warning("Try to install `pyctcdecode`: `pip install pyctcdecode")

    # 如果任务是翻译且模型配置具有特定任务参数
    if task == "translation" and model.config.task_specific_params:
        # 遍历模型配置的特定任务参数
        for key in model.config.task_specific_params:
            # 如果参数以 "translation" 开头
            if key.startswith("translation"):
                # 将任务设为该参数值，并发出警告
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    # 如果存在分词器，则将其加入参数中
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    # 如果提供了特征提取器，则将其添加到 kwargs 字典中
    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    # 如果提供了 torch 的数据类型，则将其添加到 kwargs 字典中
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    # 如果提供了图像处理器，则将其添加到 kwargs 字典中
    if image_processor is not None:
        kwargs["image_processor"] = image_processor

    # 如果提供了设备信息，则将其添加到 kwargs 字典中
    if device is not None:
        kwargs["device"] = device

    # 使用给定的参数和 kwargs 字典创建一个新的 pipeline_class 对象并返回
    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
```