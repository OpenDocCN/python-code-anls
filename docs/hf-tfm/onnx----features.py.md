# `.\transformers\onnx\features.py`

```
# 导入必要的模块和库
import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union

import transformers

# 从 transformers 包导入相关模块和函数
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig

# 如果是类型检查环境，则导入相关的类型注释
if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

# 获取日志记录器，并禁用无效名称警告
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 如果 PyTorch 可用，则导入相关的 AutoModel 类
if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTokenClassification,
        AutoModelForVision2Seq,
    )
# 如果 TensorFlow 可用，则导入相关的 TFAutoModel 类
if is_tf_available():
    from transformers.models.auto import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForMultipleChoice,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSemanticSegmentation,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
    )
# 如果 PyTorch 和 TensorFlow 都不可用，则输出警告
if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The ONNX export features are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )


def supported_features_mapping(
    *supported_features: str, onnx_config_cls: str = None
) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    # 如果未提供 OnnxConfig 类名，则抛出一个异常
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    # 获取 OnnxConfig 类
    config_cls = transformers
    for attr_name in onnx_config_cls.split("."):
        config_cls = getattr(config_cls, attr_name)

    # 创建一个字典，将每个特性映射到对应的 OnnxConfig 构造函数
    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)

    return mapping


class FeaturesManager:
    # 用于存储各任务对应的 AutoModel 和 TFAutoModel 的字典
    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
    # 如果已经安装了torch库
    if is_torch_available():
        # 设置任务类型到自动模型类的映射关系
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,  # 默认类型的自动模型
            "masked-lm": AutoModelForMaskedLM,  # 掩码语言模型的自动模型
            "causal-lm": AutoModelForCausalLM,  # 因果语言模型的自动模型
            "seq2seq-lm": AutoModelForSeq2SeqLM,  # 序列到序列语言模型的自动模型
            "sequence-classification": AutoModelForSequenceClassification,  # 序列分类的自动模型
            "token-classification": AutoModelForTokenClassification,  # 标记分类的自动模型
            "multiple-choice": AutoModelForMultipleChoice,  # 多项选择的自动模型
            "object-detection": AutoModelForObjectDetection,  # 目标检测的自动模型
            "question-answering": AutoModelForQuestionAnswering,  # 问答的自动模型
            "image-classification": AutoModelForImageClassification,  # 图像分类的自动模型
            "image-segmentation": AutoModelForImageSegmentation,  # 图像分割的自动模型
            "masked-im": AutoModelForMaskedImageModeling,  # 掩模图像建模的自动模型
            "semantic-segmentation": AutoModelForSemanticSegmentation,  # 语义分割的自动模型
            "vision2seq-lm": AutoModelForVision2Seq,  # 视觉到序列语言模型的自动模型
            "speech2seq-lm": AutoModelForSpeechSeq2Seq,  # 语音到序列语言模型的自动模型
        }
    # 如果已经安装了tensorflow库
    if is_tf_available():
        # 设置任务类型到tensorflow自动模型类的映射关系
        _TASKS_TO_TF_AUTOMODELS = {
            "default": TFAutoModel,  # 默认类型的tensorflow自动模型
            "masked-lm": TFAutoModelForMaskedLM,  # 掩码语言模型的tensorflow自动模型
            "causal-lm": TFAutoModelForCausalLM,  # 因果语言模型的tensorflow自动模型
            "seq2seq-lm": TFAutoModelForSeq2SeqLM,  # 序列到序列语言模型的tensorflow自动模型
            "sequence-classification": TFAutoModelForSequenceClassification,  # 序列分类的tensorflow自动模型
            "token-classification": TFAutoModelForTokenClassification,  # 标记分类的tensorflow自动模型
            "multiple-choice": TFAutoModelForMultipleChoice,  # 多项选择的tensorflow自动模型
            "question-answering": TFAutoModelForQuestionAnswering,  # 问答的tensorflow自动模型
            "semantic-segmentation": TFAutoModelForSemanticSegmentation,  # 语义分割的tensorflow自动模型
        }

    # 支持的模型类型和其支持的特性以及工厂的集合
    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    # 定义一个静态方法，用于检索模型类型的特性与对应的OnnxConfig构造函数映射
    # 参数model_type（`str`）：要检索支持特性的模型类型
    # 参数model_name（`str`，*可选*）：模型对象的名称属性，仅用于异常消息
    # 返回值：每个特性到对应的OnnxConfig构造函数的字典
    def get_feature_to_config_map(model_type: str, model_name: str = None) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
        # 将模型类型转换为小写
        model_type = model_type.lower()
        # 如果模型类型不在SUPPORTED_MODEL_TYPE中，则抛出KeyError异常
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        # 返回SUPPORTED_MODEL_TYPE中模型类型对应的值
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]

    # 定义一个静态方法，用于将特性名称中的"-with-past"替换为空字符串
    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    # 定义一个静态方法，用于验证导出的框架选择是否正确可用，否则抛出异常
    @staticmethod
    def _validate_framework_choice(framework: str):
        if framework not in ["pt", "tf"]:
            raise ValueError(
                f"Only two frameworks are supported for ONNX export: pt or tf, but {framework} was provided."
            )
        elif framework == "pt" and not is_torch_available():
            raise RuntimeError("Cannot export model to ONNX using PyTorch because no PyTorch package was found.")
        elif framework == "tf" and not is_tf_available():
            raise RuntimeError("Cannot export model to ONNX using TensorFlow because no TensorFlow package was found.")

    # 定义一个静态方法
    @staticmethod
    def get_model_class_for_feature(feature: str, framework: str = "pt") -> Type:
        """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        # 将特征名称映射到任务类型
        task = FeaturesManager.feature_to_task(feature)
        # 验证框架选择的有效性
        FeaturesManager._validate_framework_choice(framework)
        # 根据框架类型选择任务到AutoModel类的映射字典
        if framework == "pt":
            task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        else:
            task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
        # 检查任务是否在映射字典中
        if task not in task_to_automodel:
            raise KeyError(
                f"Unknown task: {feature}. Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )

        return task_to_automodel[task]

    @staticmethod
    def determine_framework(model: str, framework: str = None) -> str:
        """
        Determines the framework to use for the export.

        The priority is in the following order:
            1. User input via `framework`.
            2. If local checkpoint is provided, use the same framework as the checkpoint.
            3. Available framework in environment, with priority given to PyTorch

        Args:
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See above for priority if none provided.

        Returns:
            The framework to use for the export.

        """
        # 如果用户指定了框架，直接返回用户指定的框架
        if framework is not None:
            return framework

        # 框架映射字典，用于判断模型所使用的框架
        framework_map = {"pt": "PyTorch", "tf": "TensorFlow"}
        # 导出器映射字典，用于导出模型到ONNX格式
        exporter_map = {"pt": "torch", "tf": "tf2onnx"}

        # 判断模型是否是一个目录
        if os.path.isdir(model):
            # 检查模型目录中是否有PyTorch的权重文件
            if os.path.isfile(os.path.join(model, WEIGHTS_NAME)):
                framework = "pt"
            # 检查模型目录中是否有TensorFlow的权重文件
            elif os.path.isfile(os.path.join(model, TF2_WEIGHTS_NAME)):
                framework = "tf"
            else:
                raise FileNotFoundError(
                    "Cannot determine framework from given checkpoint location."
                    f" There should be a {WEIGHTS_NAME} for PyTorch"
                    f" or {TF2_WEIGHTS_NAME} for TensorFlow."
                )
            logger.info(f"Local {framework_map[framework]} model found.")
        else:
            # 如果当前环境支持PyTorch，选择PyTorch作为默认框架
            if is_torch_available():
                framework = "pt"
            # 如果当前环境支持TensorFlow，选择TensorFlow作为默认框架
            elif is_tf_available():
                framework = "tf"
            else:
                raise EnvironmentError("Neither PyTorch nor TensorFlow found in environment. Cannot export to ONNX.")

        logger.info(f"Framework not requested. Using {exporter_map[framework]} to export to ONNX.")

        return framework
    def get_model_from_feature(
        feature: str, model: str, framework: str = None, cache_dir: str = None
    ) -> Union["PreTrainedModel", "TFPreTrainedModel"]:
        """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See `FeaturesManager.determine_framework` for the priority should
                none be provided.

        Returns:
            The instance of the model.

        """
        framework = FeaturesManager.determine_framework(model, framework)
        # 根据模型名称和特征尝试检索模型
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        try:
            # 尝试从已训练好的模型中加载模型
            model = model_class.from_pretrained(model, cache_dir=cache_dir)
        except OSError:
            # 在导出到 ONNX 之前在 PyTorch 中加载 TensorFlow 模型
            if framework == "pt":
                logger.info("Loading TensorFlow model in PyTorch before exporting to ONNX.")
                model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
            else:
                # 在导出到 ONNX 之前在 TensorFlow 中加载 PyTorch 模型
                logger.info("Loading PyTorch model in TensorFlow before exporting to ONNX.")
                model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir)
        return model

    @staticmethod
    def check_supported_model_or_raise(
        model: Union["PreTrainedModel", "TFPreTrainedModel"], feature: str = "default"
    ) -> Tuple[str, Callable]:
        """
        Check whether or not the model has the requested features.

        Args:
            model: The model to export.
            feature: The name of the feature to check if it is available.

        Returns:
            (str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties.

        """
        model_type = model.config.model_type.replace("_", "-")
        model_name = getattr(model, "name", "")
        # 获取模型类型和模型名称支持的特性
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        if feature not in model_features:
            # 如果请求的特性不在支持列表中，则引发错误
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]

    def get_config(model_type: str, feature: str) -> OnnxConfig:
        """
        Gets the OnnxConfig for a model_type and feature combination.

        Args:
            model_type (`str`):
                The model type to retrieve the config for.
            feature (`str`):
                The feature to retrieve the config for.

        Returns:
            `OnnxConfig`: config for the combination
        """
        # 获取给定模型类型和特性组合的配置
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]
```