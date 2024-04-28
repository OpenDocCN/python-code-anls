# `.\transformers\pipelines\__init__.py`

```py
# 导入所需的库和模块
import io  # 用于输入输出流的操作
import json  # 用于 JSON 数据的处理
import os  # 用于操作文件和目录的功能
import warnings  # 用于警告处理
from pathlib import Path  # 用于操作文件路径和目录路径中的不同操作系统上的特定路径操作
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union  # 用于类型提示和类型检查的工具

from huggingface_hub import model_info  # 从 Hugging Face Hub 中导入模型信息

# 从 numpy 库中导入 isin 函数
from numpy import isin

# 从其它模块中导入所需的类和函数
from ..configuration_utils import PretrainedConfig  # 导入预训练配置类
from ..dynamic_module_utils import get_class_from_dynamic_module  # 从动态模块中获取类
from ..feature_extraction_utils import PreTrainedFeatureExtractor  # 导入预训练特征提取器类
from ..image_processing_utils import BaseImageProcessor  # 导入基础图像处理器类
from ..models.auto.configuration_auto import AutoConfig  # 从自动配置中导入自动配置类
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor  # 从自动特征提取中导入特征提取映射和自动特征提取类
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor  # 从自动图像处理中导入图像处理映射和自动图像处理类
from ..models.auto.modeling_auto import AutoModelForDepthEstimation, AutoModelForImageToImage  # 从自动建模中导入深度估计模型和图像转图像模型
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer  # 从自动标记化中导入标记器映射和自动标记器
from ..tokenization_utils import PreTrainedTokenizer  # 导入预训练标记器类
from ..utils import (
    CONFIG_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    cached_file,
    extract_commit_hash,
    find_adapter_config_file,
    is_kenlm_available,
    is_offline_mode,
    is_peft_available,
    is_pyctcdecode_available,
    is_tf_available,
    is_torch_available,
    logging,
)
from .audio_classification import AudioClassificationPipeline  # 导入音频分类管道类
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline  # 导入自动语音识别管道类
from .base import (
    ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    PipelineException,
    PipelineRegistry,
    get_default_model_and_revision,
    infer_framework_load_model,
)
from .conversational import Conversation, ConversationalPipeline  # 导入对话和对话管道类
from .depth_estimation import DepthEstimationPipeline  # 导入深度估计管道类
from .document_question_answering import DocumentQuestionAnsweringPipeline  # 导入文档问答管道类
from .feature_extraction import FeatureExtractionPipeline  # 导入特征提取管道类
from .fill_mask import FillMaskPipeline  # 导入填充掩码管道类
from .image_classification import ImageClassificationPipeline  # 导入图像分类管道类
from .image_segmentation import ImageSegmentationPipeline  # 导入图像分割管道类
from .image_to_image import ImageToImagePipeline  # 导入图像转图像管道类
from .image_to_text import ImageToTextPipeline  # 导入图像转文本管道类
from .mask_generation import MaskGenerationPipeline  # 导入掩码生成管道类
from .object_detection import ObjectDetectionPipeline  # 导入目标检测管道类
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline  # 导入问答处理器和问答管道类
# 从不同的模块中导入各种任务的Pipeline类和ArgumentHandler类
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .text_to_audio import TextToAudioPipeline
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from .video_classification import VideoClassificationPipeline
from .visual_question_answering import VisualQuestionAnsweringPipeline
from .zero_shot_audio_classification import ZeroShotAudioClassificationPipeline
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline
from .zero_shot_object_detection import ZeroShotObjectDetectionPipeline

# 如果TensorFlow可用，则导入相关的模型类
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

# 如果PyTorch可用，则导入相关的模型类
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

# 如果存在TYPE_CHECKING，则导入相应的类型
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_fast import PreTrainedTokenizerFast

# 获取logging的logger实例
logger = logging.get_logger(__name__)

# 不同任务的别名映射字典
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
    "text-to-speech": "text-to-audio",
}
# 支持的任务列表，每个任务包含实现、所需的框架、默认模型和类型信息
SUPPORTED_TASKS = {
    # 音频分类任务
    "audio-classification": {
        "impl": AudioClassificationPipeline,  # 实现的类
        "tf": (),  # TensorFlow 相关内容
        "pt": (AutoModelForAudioClassification,) if is_torch_available() else (),  # PyTorch 相关内容
        "default": {"model": {"pt": ("superb/wav2vec2-base-superb-ks", "372e048")}},  # 默认模型
        "type": "audio",  # 任务类型
    },
    # 自动语音识别任务
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        "tf": (),
        "pt": (AutoModelForCTC, AutoModelForSpeechSeq2Seq) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/wav2vec2-base-960h", "55bb623")}},
        "type": "multimodal",
    },
    # 文本转音频任务
    "text-to-audio": {
        "impl": TextToAudioPipeline,
        "tf": (),
        "pt": (AutoModelForTextToWaveform, AutoModelForTextToSpectrogram) if is_torch_available() else (),
        "default": {"model": {"pt": ("suno/bark-small", "645cfba")}},
        "type": "text",
    },
    # 特征提取任务
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {"model": {"pt": ("distilbert-base-cased", "935ac13"), "tf": ("distilbert-base-cased", "935ac13")}},
        "type": "multimodal",
    },
    # 文本分类任务
    "text-classification": {
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
                "tf": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
            },
        },
        "type": "text",
    },
    # 标记分类任务
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
                "tf": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
            },
        },
        "type": "text",
    },
    # 问答任务
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": (TFAutoModelForQuestionAnswering,) if is_tf_available() else (),
        "pt": (AutoModelForQuestionAnswering,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert-base-cased-distilled-squad", "626af31"),
                "tf": ("distilbert-base-cased-distilled-squad", "626af31"),
            },
        },
        "type": "text",
    },
    "table-question-answering": {  # 表格问答任务
        "impl": TableQuestionAnsweringPipeline,  # 使用TableQuestionAnsweringPipeline实现
        "pt": (AutoModelForTableQuestionAnswering,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForTableQuestionAnswering
        "tf": (TFAutoModelForTableQuestionAnswering,) if is_tf_available() else (),  # TensorFlow模型可用时使用TFAutoModelForTableQuestionAnswering
        "default": {  # 默认模型
            "model": {  # 模型参数配置
                "pt": ("google/tapas-base-finetuned-wtq", "69ceee2"),  # PyTorch模型参数配置
                "tf": ("google/tapas-base-finetuned-wtq", "69ceee2"),  # TensorFlow模型参数配置
            },
        },
        "type": "text",  # 任务类型为文本
    },
    "visual-question-answering": {  # 视觉问答任务
        "impl": VisualQuestionAnsweringPipeline,  # 使用VisualQuestionAnsweringPipeline实现
        "pt": (AutoModelForVisualQuestionAnswering,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForVisualQuestionAnswering
        "tf": (),  # TensorFlow模型为空
        "default": {  # 默认模型
            "model": {"pt": ("dandelin/vilt-b32-finetuned-vqa", "4355f59")},  # PyTorch模型参数配置
        },
        "type": "multimodal",  # 任务类型为多模态
    },
    "document-question-answering": {  # 文档问答任务
        "impl": DocumentQuestionAnsweringPipeline,  # 使用DocumentQuestionAnsweringPipeline实现
        "pt": (AutoModelForDocumentQuestionAnswering,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForDocumentQuestionAnswering
        "tf": (),  # TensorFlow模型为空
        "default": {  # 默认模型
            "model": {"pt": ("impira/layoutlm-document-qa", "52e01b3")},  # PyTorch模型参数配置
        },
        "type": "multimodal",  # 任务类型为多模态
    },
    "fill-mask": {  # 填充掩码任务
        "impl": FillMaskPipeline,  # 使用FillMaskPipeline实现
        "tf": (TFAutoModelForMaskedLM,) if is_tf_available() else (),  # TensorFlow模型可用时使用TFAutoModelForMaskedLM
        "pt": (AutoModelForMaskedLM,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForMaskedLM
        "default": {"model": {"pt": ("distilroberta-base", "ec58a5b"), "tf": ("distilroberta-base", "ec58a5b")}},  # 默认模型参数配置
        "type": "text",  # 任务类型为文本
    },
    "summarization": {  # 文本摘要任务
        "impl": SummarizationPipeline,  # 使用SummarizationPipeline实现
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),  # TensorFlow模型可用时使用TFAutoModelForSeq2SeqLM
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForSeq2SeqLM
        "default": {"model": {"pt": ("sshleifer/distilbart-cnn-12-6", "a4f8f3e"), "tf": ("t5-small", "d769bba")}},  # 默认模型参数配置
        "type": "text",  # 任务类型为文本
    },
    # This task is a special case as it's parametrized by SRC, TGT languages.
    "translation": {  # 翻译任务
        "impl": TranslationPipeline,  # 使用TranslationPipeline实现
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),  # TensorFlow模型可用时使用TFAutoModelForSeq2SeqLM
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForSeq2SeqLM
        "default": {  # 默认模型
            ("en", "fr"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")},  # 英法语对的模型参数配置
            ("en", "de"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")},  # 英德语对的模型参数配置
            ("en", "ro"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")},  # 英罗马尼亚语对的模型参数配置
        },
        "type": "text",  # 任务类型为文本
    },
    "text2text-generation": {  # 文本生成任务
        "impl": Text2TextGenerationPipeline,  # 使用Text2TextGenerationPipeline实现
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),  # TensorFlow模型可用时使用TFAutoModelForSeq2SeqLM
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),  # PyTorch模型可用时使用AutoModelForSeq2SeqLM
        "default": {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},  # 默认模型参数配置
        "type": "text",  # 任务类型为文本
    },
    # 文本生成任务配置
    "text-generation": {
        # 使用TextGenerationPipeline实现文本生成
        "impl": TextGenerationPipeline,
        # 如果有TensorFlow可用，则使用TFAutoModelForCausalLM
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),
        # 如果有PyTorch可用，则使用AutoModelForCausalLM
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        # 默认模型配置
        "default": {"model": {"pt": ("gpt2", "6c0e608"), "tf": ("gpt2", "6c0e608")}},
        # 任务类型为文本
        "type": "text",
    },
    
    # 零样本分类任务配置
    "zero-shot-classification": {
        # 使用ZeroShotClassificationPipeline实现零样本分类
        "impl": ZeroShotClassificationPipeline,
        # 如果有TensorFlow可用，则使用TFAutoModelForSequenceClassification
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        # 如果有PyTorch可用，则使用AutoModelForSequenceClassification
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        # 默认模型和配置
        "default": {
            "model": {"pt": ("facebook/bart-large-mnli", "c626438"), "tf": ("roberta-large-mnli", "130fb28")},
            "config": {"pt": ("facebook/bart-large-mnli", "c626438"), "tf": ("roberta-large-mnli", "130fb28")},
        },
        # 任务类型为文本
        "type": "text",
    },
    
    # 零样本图像分类任务配置
    "zero-shot-image-classification": {
        # 使用ZeroShotImageClassificationPipeline实现零样本图像分类
        "impl": ZeroShotImageClassificationPipeline,
        # 如果有TensorFlow可用，则使用TFAutoModelForZeroShotImageClassification
        "tf": (TFAutoModelForZeroShotImageClassification,) if is_tf_available() else (),
        # 如果有PyTorch可用，则使用AutoModelForZeroShotImageClassification
        "pt": (AutoModelForZeroShotImageClassification,) if is_torch_available() else (),
        # 默认模型配置
        "default": {
            "model": {
                "pt": ("openai/clip-vit-base-patch32", "f4881ba"),
                "tf": ("openai/clip-vit-base-patch32", "f4881ba"),
            }
        },
        # 任务类型为多模态
        "type": "multimodal",
    },
    
    # 零样本音频分类任务配置
    "zero-shot-audio-classification": {
        # 使用ZeroShotAudioClassificationPipeline实现零样本音频分类
        "impl": ZeroShotAudioClassificationPipeline,
        # TensorFlow不适用
        "tf": (),
        # 如果有PyTorch可用，则使用AutoModel
        "pt": (AutoModel,) if is_torch_available() else (),
        # 默认模型配置
        "default": {
            "model": {
                "pt": ("laion/clap-htsat-fused", "973b6e5"),
            }
        },
        # 任务类型为多模态
        "type": "multimodal",
    },
    
    # 对话任务配置
    "conversational": {
        # 使用ConversationalPipeline实现对话任务
        "impl": ConversationalPipeline,
        # 如果有TensorFlow可用，则使用TFAutoModelForSeq2SeqLM和TFAutoModelForCausalLM
        "tf": (TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM) if is_tf_available() else (),
        # 如果有PyTorch可用，则使用AutoModelForSeq2SeqLM和AutoModelForCausalLM
        "pt": (AutoModelForSeq2SeqLM, AutoModelForCausalLM) if is_torch_available() else (),
        # 默认模型配置
        "default": {
            "model": {"pt": ("microsoft/DialoGPT-medium", "8bada3b"), "tf": ("microsoft/DialoGPT-medium", "8bada3b")}
        },
        # 任务类型为文本
        "type": "text",
    },
    
    # 图像分类任务配置
    "image-classification": {
        # 使用ImageClassificationPipeline实现图像分类任务
        "impl": ImageClassificationPipeline,
        # 如果有TensorFlow可用，则使用TFAutoModelForImageClassification
        "tf": (TFAutoModelForImageClassification,) if is_tf_available() else (),
        # 如果有PyTorch可用，则使用AutoModelForImageClassification
        "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
        # 默认模型配置
        "default": {
            "model": {
                "pt": ("google/vit-base-patch16-224", "5dca96d"),
                "tf": ("google/vit-base-patch16-224", "5dca96d"),
            }
        },
        # 任务类型为图像
        "type": "image",
    },
    
    # 图像分割任务配置
    "image-segmentation": {
        # 使用ImageSegmentationPipeline实现图像分割任务
        "impl": ImageSegmentationPipeline,
        # TensorFlow不适用
        "tf": (),
        # 如果有PyTorch可用，则使用AutoModelForImageSegmentation和AutoModelForSemanticSegmentation
        "pt": (AutoModelForImageSegmentation, AutoModelForSemanticSegmentation) if is_torch_available() else (),
        # 默认模型配置
        "default": {"model": {"pt": ("facebook/detr-resnet-50-panoptic", "fc15262")},
        # 任务类型为多模态
        "type": "multimodal",
    },
    "image-to-text": {
        # 图像转文本任务的实现类
        "impl": ImageToTextPipeline,
        # 如果 TensorFlow 可用，则包含 TFAutoModelForVision2Seq 模型，否则为空元组
        "tf": (TFAutoModelForVision2Seq,) if is_tf_available() else (),
        # 如果 PyTorch 可用，则包含 AutoModelForVision2Seq 模型，否则为空元组
        "pt": (AutoModelForVision2Seq,) if is_torch_available() else (),
        # 默认模型配置
        "default": {
            "model": {
                # PyTorch 默认模型为 ("ydshieh/vit-gpt2-coco-en", "65636df")
                "pt": ("ydshieh/vit-gpt2-coco-en", "65636df"),
                # TensorFlow 默认模型为 ("ydshieh/vit-gpt2-coco-en", "65636df")
                "tf": ("ydshieh/vit-gpt2-coco-en", "65636df"),
            }
        },
        # 任务类型为多模态
        "type": "multimodal",
    },
    "object-detection": {
        # 目标检测任务的实现类
        "impl": ObjectDetectionPipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForObjectDetection 模型，否则为空元组
        "pt": (AutoModelForObjectDetection,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("facebook/detr-resnet-50", "2729413")}
        "default": {"model": {"pt": ("facebook/detr-resnet-50", "2729413")}},
        # 任务类型为多模态
        "type": "multimodal",
    },
    "zero-shot-object-detection": {
        # 零样本目标检测任务的实现类
        "impl": ZeroShotObjectDetectionPipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForZeroShotObjectDetection 模型，否则为空元组
        "pt": (AutoModelForZeroShotObjectDetection,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("google/owlvit-base-patch32", "17740e1")}
        "default": {"model": {"pt": ("google/owlvit-base-patch32", "17740e1")}},
        # 任务类型为多模态
        "type": "multimodal",
    },
    "depth-estimation": {
        # 深度估计任务的实现类
        "impl": DepthEstimationPipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForDepthEstimation 模型，否则为空元组
        "pt": (AutoModelForDepthEstimation,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("Intel/dpt-large", "e93beec")}
        "default": {"model": {"pt": ("Intel/dpt-large", "e93beec")}},
        # 任务类型为图像
        "type": "image",
    },
    "video-classification": {
        # 视频分类任务的实现类
        "impl": VideoClassificationPipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForVideoClassification 模型，否则为空元组
        "pt": (AutoModelForVideoClassification,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("MCG-NJU/videomae-base-finetuned-kinetics", "4800870")}
        "default": {"model": {"pt": ("MCG-NJU/videomae-base-finetuned-kinetics", "4800870")}},
        # 任务类型为视频
        "type": "video",
    },
    "mask-generation": {
        # 掩模生成任务的实现类
        "impl": MaskGenerationPipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForMaskGeneration 模型，否则为空元组
        "pt": (AutoModelForMaskGeneration,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("facebook/sam-vit-huge", "997b15")}
        "default": {"model": {"pt": ("facebook/sam-vit-huge", "997b15")}},
        # 任务类型为多模态
        "type": "multimodal",
    },
    "image-to-image": {
        # 图像到图像任务的实现类
        "impl": ImageToImagePipeline,
        # TensorFlow 模型为空元组
        "tf": (),
        # 如果 PyTorch 可用，则包含 AutoModelForImageToImage 模型，否则为空元组
        "pt": (AutoModelForImageToImage,) if is_torch_available() else (),
        # 默认模型配置为 {"pt": ("caidas/swin2SR-classical-sr-x2-64", "4aaedcb")}
        "default": {"model": {"pt": ("caidas/swin2SR-classical-sr-x2-64", "4aaedcb")}},
        # 任务类型为图像
        "type": "image",
    },
# 定义三个空集合，用于存储不需要特征提取器、图像处理器和分词器的任务
NO_FEATURE_EXTRACTOR_TASKS = set()
NO_IMAGE_PROCESSOR_TASKS = set()
NO_TOKENIZER_TASKS = set()

# 这些模型配置是特殊的，它们对其任务是通用的，意味着任何分词器/特征提取器都可能用于给定模型，因此我们不能使用静态定义的TOKENIZER_MAPPING和FEATURE_EXTRACTOR_MAPPING来查看模型是否定义了这些对象。
MULTI_MODEL_CONFIGS = {"SpeechEncoderDecoderConfig", "VisionEncoderDecoderConfig", "VisionTextDualEncoderConfig"}

# 遍历支持的任务，根据任务类型将任务添加到相应的集合中
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] in {"image", "video"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] in {"audio"}:
        NO_TOKENIZER_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")

# 创建PipelineRegistry对象，传入支持的任务和任务别名
PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)

# 获取支持的任务列表
def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return PIPELINE_REGISTRY.get_supported_tasks()

# 获取任务
def get_task(model: str, token: Optional[str] = None, **deprecated_kwargs) -> str:
    # 弹出deprecated_kwargs中的"use_auth_token"参数
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果use_auth_token不为None，则发出警告
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果token不为None，则引发ValueError
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # 如果处于离线模式，则引发RuntimeError
    if is_offline_mode():
        raise RuntimeError("You cannot infer task automatically within `pipeline` when using offline mode")
    try:
        # 获取模型信息
        info = model_info(model, token=token)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    # 如果info.pipeline_tag不存在，则引发RuntimeError
    if not info.pipeline_tag:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    # 如果info.library_name不是"transformers"，则引发RuntimeError
    if getattr(info, "library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {info.library_name} not with transformers")
    # 返回任务
    task = info.pipeline_tag
    return task

# 检查任务
def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.
    # 定义函数参数和返回值的说明
    Args:
        task (`str`):
            指定要返回的管道的任务。当前接受的任务有：

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` 可用)
            - `"text-generation"`
            - `"text-to-audio"` (alias `"text-to-speech"` 可用)
            - `"token-classification"` (alias `"ner"` 可用)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"`
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) 返回规范化的任务名称
        (移除别名和选项)。初始化管道所需的实际字典和一些额外的任务选项，用于参数化任务如 "translation_XX_to_YY"


    """
    # 检查任务是否在管道注册表中，并返回结果
    return PIPELINE_REGISTRY.check_task(task)
# 清理自定义任务信息，确保任务信息中包含实现信息
def clean_custom_task(task_info):
    # 导入transformers库
    import transformers

    # 如果任务信息中没有实现信息，则抛出运行时错误
    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    
    # 获取任务信息中的pt类名列表，并转换为列表类型
    pt_class_names = task_info.get("pt", ())
    if isinstance(pt_class_names, str):
        pt_class_names = [pt_class_names]
    
    # 将transformers库中对应类名转换为实际类对象，并更新任务信息中的pt字段
    task_info["pt"] = tuple(getattr(transformers, c) for c in pt_class_names)
    
    # 获取任务信息中的tf类名列表，并转换为列表类型
    tf_class_names = task_info.get("tf", ())
    if isinstance(tf_class_names, str):
        tf_class_names = [tf_class_names]
    
    # 将transformers库中对应类名转换为实际类对象，并更新任务信息中的tf字段
    task_info["tf"] = tuple(getattr(transformers, c) for c in tf_class_names)
    
    # 返回更新后的任务信息和空值
    return task_info, None


# 构建Pipeline的实用工厂方法
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
    ...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```py"""
    
    # 如果model_kwargs为None，则初始化为空字典
    if model_kwargs is None:
        model_kwargs = {}
    
    # 确保只传递一次use_auth_token作为关键字参数（以前可以将其作为model_kwargs传递，为了保持向后兼容性）
    use_auth_token = model_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 不为 None，则发出警告，提示该参数已被弃用，并将在 Transformers 的 v5 版本中移除，建议使用 token 参数代替
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 也不为 None，则抛出数值错误，提示只能设置一个参数 token
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 设置为 use_auth_token
        token = use_auth_token

    # 从 kwargs 中弹出 code_revision 和 _commit_hash 参数
    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)

    # 构建 hub_kwargs 字典，包含 revision、token、trust_remote_code 和 _commit_hash 参数
    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    # 如果 task 和 model 都为 None，则抛出运行时错误，提示需要指定 task 类或 model
    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    # 如果 model 为 None 且 tokenizer 不为 None，则抛出运行时错误，提示需要提供 PreTrainedModel 类或预训练模型的路径/标识符
    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )
    # 如果 model 为 None 且 feature_extractor 不为 None，则抛出运行时错误，提示需要提供 PreTrainedModel 类或预训练模型的路径/标识符
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
            " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
            " or a path/identifier to a pretrained model when providing feature_extractor."
        )
    # 如果 model 是 Path 类型，则将其转换为字符串类型
    if isinstance(model, Path):
        model = str(model)

    # 如果 commit_hash 为 None
    if commit_hash is None:
        pretrained_model_name_or_path = None
        # 如果 config 是字符串类型，则将其赋值给 pretrained_model_name_or_path
        if isinstance(config, str):
            pretrained_model_name_or_path = config
        # 如果 config 为 None 且 model 是字符串类型，则将 model 赋值给 pretrained_model_name_or_path
        elif config is None and isinstance(model, str):
            pretrained_model_name_or_path = model

        # 如果 config 不是 PretrainedConfig 类型且 pretrained_model_name_or_path 不为 None
        if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
            # 首先调用配置文件以尽早获取提交哈希值
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            # 从配置文件中提取提交哈希值
            hub_kwargs["_commit_hash"] = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            # 将 config 的 _commit_hash 赋值给 hub_kwargs 的 _commit_hash
            hub_kwargs["_commit_hash"] = getattr(config, "_commit_hash", None)

    # Config 是原始信息项。
    # 如果需要，实例化 config
    # 检查 config 是否为字符串类型
    if isinstance(config, str):
        # 如果是字符串类型，则从预训练模型中加载配置信息
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # 将配置信息的提交哈希值存储到 hub_kwargs 中
        hub_kwargs["_commit_hash"] = config._commit_hash
    # 如果 config 为 None 且 model 是字符串类型
    elif config is None and isinstance(model, str):
        # 如果 PEFT 可用，则检查模型路径中是否存在适配器文件
        if is_peft_available():
            # 从 hub_kwargs 中排除 "trust_remote_code" 键值对，因为 `find_adapter_config_file` 不接受该参数
            _hub_kwargs = {k: v for k, v in hub_kwargs.items() if k != "trust_remote_code"}
            # 查找适配器配置文件
            maybe_adapter_path = find_adapter_config_file(
                model,
                token=hub_kwargs["token"],
                revision=hub_kwargs["revision"],
                _commit_hash=hub_kwargs["_commit_hash"],
            )

            if maybe_adapter_path is not None:
                # 如果找到适配器配置文件，则读取其中的配置信息
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                    model = adapter_config["base_model_name_or_path"]

        # 从预训练模型中加载配置信息
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        # 将配置信息的提交哈希值存储到 hub_kwargs 中
        hub_kwargs["_commit_hash"] = config._commit_hash

    # 初始化自定义任务字典
    custom_tasks = {}
    # 如果配置信息不为 None 且 custom_pipelines 不为空
    if config is not None and len(getattr(config, "custom_pipelines", {})) > 0:
        # 获取自定义任务字典
        custom_tasks = config.custom_pipelines
        # 如果任务为 None 且 trust_remote_code 不为 False
        if task is None and trust_remote_code is not False:
            # 如果只有一个自定义任务，则自动选择该任务
            if len(custom_tasks) == 1:
                task = list(custom_tasks.keys())[0]
            else:
                # 如果存在多个自定义任务，则抛出异常
                raise RuntimeError(
                    "We can't infer the task automatically for this model as there are multiple tasks available. Pick "
                    f"one in {', '.join(custom_tasks.keys())}"
                )

    # 如果任务为 None 且模型不为 None
    if task is None and model is not None:
        # 如果模型不是字符串类型，则抛出异常
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`. "
                f"{model} is not a valid model_id."
            )
        # 根据模型和 token 获取任务
        task = get_task(model, token)

    # 获取任务
    if task in custom_tasks:
        # 标准化任务名称
        normalized_task = task
        # 清理自定义任务信息
        targeted_task, task_options = clean_custom_task(custom_tasks[task])
        # 如果 pipeline_class 为 None
        if pipeline_class is None:
            # 如果不信任远程代码，则抛出异常
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
            # 获取类引用并创建动态模块
            class_ref = targeted_task["impl"]
            pipeline_class = get_class_from_dynamic_module(
                class_ref,
                model,
                code_revision=code_revision,
                **hub_kwargs,
            )
    # 如果条件不成立，则执行以下代码
    else:
        # 对任务进行规范化，获取目标任务和任务选项
        normalized_task, targeted_task, task_options = check_task(task)
        # 如果管道类为空，则使用目标任务的实现类
        if pipeline_class is None:
            pipeline_class = targeted_task["impl"]

    # 如果没有提供模型，则使用任务的默认模型/配置/分词器
    if model is None:
        # 在这一点上，框架可能仍未确定
        model, default_revision = get_default_model_and_revision(targeted_task, framework, task_options)
        # 如果未提供修订版本，则使用默认修订版本
        revision = revision if revision is not None else default_revision
        # 输出警告信息，指示默认模型和修订版本
        logger.warning(
            f"No model was supplied, defaulted to {model} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        # 如果配置为空且模型是字符串类型，则从预训练模型加载配置
        if config is None and isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **model_kwargs)
            hub_kwargs["_commit_hash"] = config._commit_hash

    # 如果设备映射不为空
    if device_map is not None:
        # 如果模型参数中包含设备映射，则引发值错误
        if "device_map" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... device_map=..., model_kwargs={"device_map":...})` as those'
                " arguments might conflict, use only one.)"
            )
        # 如果设备不为空，则输出警告信息
        if device is not None:
            logger.warning(
                "Both `device` and `device_map` are specified. `device` will override `device_map`. You"
                " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`."
            )
        # 将设备映射添加到模型参数中
        model_kwargs["device_map"] = device_map
    # 如果 torch 数据类型不为空
    if torch_dtype is not None:
        # 如果模型参数中包含 torch 数据类型，则引发值错误
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        # 将 torch 数据类型添加到模型参数中
        model_kwargs["torch_dtype"] = torch_dtype

    # 如果模型是字符串类型或框架为空
    model_name = model if isinstance(model, str) else None

    # 加载正确的模型（如果可能）
    # 如果框架未定义，则从模型推断框架
    if isinstance(model, str) or framework is None:
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

    # 获取模型配置
    model_config = model.config
    hub_kwargs["_commit_hash"] = model.config._commit_hash
    # 判断是否需要加载分词器
    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    # 判断是否需要加载特征提取器
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None
    # 判断是否需要加载图像处理器
    load_image_processor = type(model_config) in IMAGE_PROCESSOR_MAPPING or image_processor is not None
    # 如果传递了`model`（`PretrainedModel`的实例而不是`str`）（以及/或者对于配置也是如此），同时`image_processor`或`feature_extractor`为`None`，加载将失败。特别是在某些视觉任务中，当使用`pipeline()`调用`model`和`image_processor`和`feature_extractor`中的一个时会发生这种情况。
    # TODO: 我们需要使`NO_IMAGE_PROCESSOR_TASKS`和`NO_FEATURE_EXTRACTOR_TASKS`更健壮，以避免这种问题。这个代码块只是为了暂时使CI变绿。
    if load_image_processor and load_feature_extractor:
        load_feature_extractor = False

    if (
        tokenizer is None
        and not load_tokenizer
        and normalized_task not in NO_TOKENIZER_TASKS
        # 使用类名来避免导入真实类。
        and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS
    ):
        # 这是一类特殊的模型，它们是多个模型的融合，因此model_config可能不定义一个tokenizer，但对于任务来说似乎是必要的，因此我们强制尝试加载它。
        load_tokenizer = True
    if (
        image_processor is None
        and not load_image_processor
        and normalized_task not in NO_IMAGE_PROCESSOR_TASKS
        # 使用类名来避免导入真实类。
        and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS
        and normalized_task != "automatic-speech-recognition"
    ):
        # 这是一类特殊的模型，它们是多个模型的融合，因此model_config可能不定义一个image_processor，但对于任务来说似乎是必要的，因此我们强制尝试加载它。
        load_image_processor = True
    if (
        feature_extractor is None
        and not load_feature_extractor
        and normalized_task not in NO_FEATURE_EXTRACTOR_TASKS
        # 使用类名来避免导入真实类。
        and model_config.__class__.__name__ in MULTI_MODEL_CONFIGS
    ):
        # 这是一类特殊的模型，它们是多个模型的融合，因此model_config可能不定义一个feature_extractor，但对于任务来说似乎是必要的，因此我们强制尝试加载它。
        load_feature_extractor = True

    if task in NO_TOKENIZER_TASKS:
        # 这些任务永远不需要一个tokenizer。
        # 另一方面，模型可能有一个tokenizer，但是文件可能在hub中丢失，而不是在这些存储库上失败，我们只是强制不加载它。
        load_tokenizer = False

    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    if task in NO_IMAGE_PROCESSOR_TASKS:
        load_image_processor = False
    # 如果需要加载分词器
    if load_tokenizer:
        # 尝试从模型或配置名称中推断分词器（如果提供为字符串）
        if tokenizer is None:
            # 如果 model_name 是字符串，则将其作为分词器
            if isinstance(model_name, str):
                tokenizer = model_name
            # 如果 config 是字符串，则将其作为分词器
            elif isinstance(config, str):
                tokenizer = config
            else:
                # 无法猜测应该使用哪个分词器
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # 如果分词器是字符串或元组，则实例化分词器
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

            # 从预训练模型加载自动分词器
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )

    # 如果需要加载图像处理器
    if load_image_processor:
        # 尝试从模型或配置名称中推断图像处理器（如果提供为字符串）
        if image_processor is None:
            # 如果 model_name 是字符串，则将其作为图像处理器
            if isinstance(model_name, str):
                image_processor = model_name
            # 如果 config 是字符串，则将其作为图像处理器
            elif isinstance(config, str):
                image_processor = config
            # 向后兼容，因为 `feature_extractor` 曾经是 `ImageProcessor` 的名称
            elif feature_extractor is not None and isinstance(feature_extractor, BaseImageProcessor):
                image_processor = feature_extractor
            else:
                # 无法猜测应该使用哪个图像处理器
                raise Exception(
                    "Impossible to guess which image processor to use. "
                    "Please provide a PreTrainedImageProcessor class or a path/identifier "
                    "to a pretrained image processor."
                )

        # 如果图像处理器是字符串或元组，则实例化图像处理器
        if isinstance(image_processor, (str, tuple)):
            # 从预训练模型加载自动图像处理器
            image_processor = AutoImageProcessor.from_pretrained(
                image_processor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
    # 如果需要加载特征提取器
    if load_feature_extractor:
        # 尝试从模型或配置名称中推断特征提取器（如果提供为字符串）
        if feature_extractor is None:
            # 如果模型名称是字符串，则将其作为特征提取器
            if isinstance(model_name, str):
                feature_extractor = model_name
            # 如果配置是字符串，则将其作为特征提取器
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # 无法猜测应该使用哪个特征提取器
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # 如果需要，实例化特征提取器
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )

            # 如果特征提取器的处理器类以 "WithLM" 结尾，并且模型名称是字符串
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
                        language_model_glob = os.path.join(
                            BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                        )
                        alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                        allow_patterns = [language_model_glob, alphabet_filename]
                        decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name, allow_patterns=allow_patterns)

                    kwargs["decoder"] = decoder
                except ImportError as e:
                    logger.warning(f"Could not load the `decoder` for {model_name}. Defaulting to raw CTC. Error: {e}")
                    if not is_kenlm_available():
                        logger.warning("Try to install `kenlm`: `pip install kenlm")

                    if not is_pyctcdecode_available():
                        logger.warning("Try to install `pyctcdecode`: `pip install pyctcdecode")

    # 如果任务是 "translation" 并且模型配置具有特定任务参数
    if task == "translation" and model.config.task_specific_params:
        # 遍历模型配置的特定任务参数
        for key in model.config.task_specific_params:
            # 如果参数以 "translation" 开头
            if key.startswith("translation"):
                task = key
                # 发出警告，使用 "translation" 任务而不是 "translation_XX_to_YY"，默认使用特定任务参数
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    # 如果存在分词器
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    # 如果特征提取器不为空，则将其添加到参数字典中
    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    # 如果 torch 数据类型不为空，则将其添加到参数字典中
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    # 如果图像处理器不为空，则将其添加到参数字典中
    if image_processor is not None:
        kwargs["image_processor"] = image_processor

    # 如果设备不为空，则将其添加到参数字典中
    if device is not None:
        kwargs["device"] = device

    # 返回使用给定参数实例化的 pipeline_class 对象
    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
```