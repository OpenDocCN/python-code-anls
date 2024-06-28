# `.\onnx\features.py`

```py
import os  # 导入标准库 os，用于与操作系统交互
from functools import partial, reduce  # 从 functools 模块导入 partial 和 reduce 函数
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union  # 导入类型提示相关的库

import transformers  # 导入 transformers 库，用于自然语言处理模型

from .. import PretrainedConfig, is_tf_available, is_torch_available  # 导入相对路径下的模块和函数
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging  # 导入相对路径下的工具函数和常量
from .config import OnnxConfig  # 导入当前目录下的 config 模块中的 OnnxConfig 类


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel  # 根据 TYPE_CHECKING 导入相关类型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象，用于记录日志信息，名称为当前模块名，如果名字无效则禁用


if is_torch_available():  # 如果系统支持 torch
    from transformers.models.auto import (  # 导入 torch 下的自动模型选择
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

if is_tf_available():  # 如果系统支持 tensorflow
    from transformers.models.auto import (  # 导入 tensorflow 下的自动模型选择
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

if not is_torch_available() and not is_tf_available():  # 如果系统既不支持 torch 也不支持 tensorflow
    logger.warning(  # 记录警告信息，提醒用户无法导出模型
        "The ONNX export features are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )


def supported_features_mapping(  # 定义函数 supported_features_mapping，用于生成支持特性与其对应 OnnxConfig 的映射关系
    *supported_features: str, onnx_config_cls: str = None  # 支持的特性名称（可变参数），以及指定的 OnnxConfig 类的全名
) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:  # 返回字典类型，键为特性名称，值为对应的 OnnxConfig 构造函数
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    if onnx_config_cls is None:  # 如果未提供 OnnxConfig 类的全名，则抛出 ValueError 异常
        raise ValueError("A OnnxConfig class must be provided")

    config_cls = transformers  # 初始化配置类为 transformers 模块
    for attr_name in onnx_config_cls.split("."):  # 根据类名字符串分割，逐层获取属性
        config_cls = getattr(config_cls, attr_name)
    mapping = {}  # 初始化空字典，用于存储特性与构造函数的映射关系
    for feature in supported_features:  # 遍历所有支持的特性名称
        if "-with-past" in feature:  # 如果特性名称包含 "-with-past"
            task = feature.replace("-with-past", "")  # 提取任务名称
            mapping[feature] = partial(config_cls.with_past, task=task)  # 使用部分函数生成配置类的构造函数
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)  # 使用部分函数生成配置类的构造函数

    return mapping  # 返回特性与构造函数的映射字典


class FeaturesManager:  # 定义特性管理器类
    _TASKS_TO_AUTOMODELS = {}  # 空字典，用于存储任务与自动模型的映射关系
    _TASKS_TO_TF_AUTOMODELS = {}  # 空字典，用于存储任务与 TensorFlow 自动模型的映射关系
    # 如果 torch 库可用，则定义一个任务到自动模型类的映射字典
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,
            "masked-lm": AutoModelForMaskedLM,
            "causal-lm": AutoModelForCausalLM,
            "seq2seq-lm": AutoModelForSeq2SeqLM,
            "sequence-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "multiple-choice": AutoModelForMultipleChoice,
            "object-detection": AutoModelForObjectDetection,
            "question-answering": AutoModelForQuestionAnswering,
            "image-classification": AutoModelForImageClassification,
            "image-segmentation": AutoModelForImageSegmentation,
            "masked-im": AutoModelForMaskedImageModeling,
            "semantic-segmentation": AutoModelForSemanticSegmentation,
            "vision2seq-lm": AutoModelForVision2Seq,
            "speech2seq-lm": AutoModelForSpeechSeq2Seq,
        }
    
    # 如果 tensorflow 库可用，则定义一个任务到 TensorFlow 自动模型类的映射字典
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            "default": TFAutoModel,
            "masked-lm": TFAutoModelForMaskedLM,
            "causal-lm": TFAutoModelForCausalLM,
            "seq2seq-lm": TFAutoModelForSeq2SeqLM,
            "sequence-classification": TFAutoModelForSequenceClassification,
            "token-classification": TFAutoModelForTokenClassification,
            "multiple-choice": TFAutoModelForMultipleChoice,
            "question-answering": TFAutoModelForQuestionAnswering,
            "semantic-segmentation": TFAutoModelForSemanticSegmentation,
        }

    # 定义一个集合，包含所有支持的特性，特性由各个模型类型支持的特性的并集组成
    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    # 静态方法：根据模型类型获取支持的特性列表
    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
        """
        Tries to retrieve the feature -> OnnxConfig constructor map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported features for.
            model_name (`str`, *optional*):
                The name attribute of the model object, only used for the exception message.

        Returns:
            The dictionary mapping each feature to a corresponding OnnxConfig constructor.
        """
        # 将 model_type 转换为小写
        model_type = model_type.lower()
        # 检查 model_type 是否在支持的模型类型中
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            # 准备错误信息，如果提供了 model_name，则将其包含在错误信息中
            model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
            # 抛出 KeyError 异常，说明给定的模型类型不被支持
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        # 返回 model_type 对应的 OnnxConfig 构造函数字典
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]

    @staticmethod
    def feature_to_task(feature: str) -> str:
        """
        Converts a feature string by removing the '-with-past' suffix.

        Args:
            feature (`str`):
                The feature string to be converted.

        Returns:
            The feature string with '-with-past' suffix removed.
        """
        return feature.replace("-with-past", "")

    @staticmethod
    def _validate_framework_choice(framework: str):
        """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.

        Args:
            framework (`str`):
                The framework requested for ONNX export.

        Raises:
            ValueError: If the provided framework is not 'pt' or 'tf'.
            RuntimeError: If the requested framework is 'pt' but PyTorch is not available,
                          or if the requested framework is 'tf' but TensorFlow is not available.
        """
        # 检查 framework 是否在支持的框架列表中
        if framework not in ["pt", "tf"]:
            # 抛出 ValueError 异常，说明只支持 'pt' 或 'tf' 两种框架
            raise ValueError(
                f"Only two frameworks are supported for ONNX export: pt or tf, but {framework} was provided."
            )
        # 如果 framework 是 'pt'，检查是否可以导出到 ONNX
        elif framework == "pt" and not is_torch_available():
            # 抛出 RuntimeError 异常，说明无法使用 PyTorch 导出模型到 ONNX
            raise RuntimeError("Cannot export model to ONNX using PyTorch because no PyTorch package was found.")
        # 如果 framework 是 'tf'，检查是否可以导出到 ONNX
        elif framework == "tf" and not is_tf_available():
            # 抛出 RuntimeError 异常，说明无法使用 TensorFlow 导出模型到 ONNX
            raise RuntimeError("Cannot export model to ONNX using TensorFlow because no TensorFlow package was found.")
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
        # 根据特征名称获取对应的任务
        task = FeaturesManager.feature_to_task(feature)
        # 验证选择的框架是否有效
        FeaturesManager._validate_framework_choice(framework)
        # 根据选择的框架确定任务到AutoModel类的映射
        if framework == "pt":
            task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        else:
            task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
        # 如果任务不在映射中，则抛出KeyError异常
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
        # 如果用户指定了框架，则直接返回该框架
        if framework is not None:
            return framework

        # 框架映射关系
        framework_map = {"pt": "PyTorch", "tf": "TensorFlow"}
        # 导出器映射关系
        exporter_map = {"pt": "torch", "tf": "tf2onnx"}

        # 如果模型路径是一个目录
        if os.path.isdir(model):
            # 检查是否存在PyTorch的权重文件
            if os.path.isfile(os.path.join(model, WEIGHTS_NAME)):
                framework = "pt"
            # 检查是否存在TensorFlow的权重文件
            elif os.path.isfile(os.path.join(model, TF2_WEIGHTS_NAME)):
                framework = "tf"
            else:
                # 如果无法确定框架，则抛出FileNotFoundError异常
                raise FileNotFoundError(
                    "Cannot determine framework from given checkpoint location."
                    f" There should be a {WEIGHTS_NAME} for PyTorch"
                    f" or {TF2_WEIGHTS_NAME} for TensorFlow."
                )
            # 记录日志，表示找到本地模型
            logger.info(f"Local {framework_map[framework]} model found.")
        else:
            # 如果PyTorch可用，则选择PyTorch框架
            if is_torch_available():
                framework = "pt"
            # 如果TensorFlow可用，则选择TensorFlow框架
            elif is_tf_available():
                framework = "tf"
            else:
                # 如果环境中既没有PyTorch也没有TensorFlow，则抛出EnvironmentError异常
                raise EnvironmentError("Neither PyTorch nor TensorFlow found in environment. Cannot export to ONNX.")

        # 记录日志，表示使用导出器将模型导出为ONNX格式
        logger.info(f"Framework not requested. Using {exporter_map[framework]} to export to ONNX.")

        return framework
    def get_model_from_feature(
        feature: str, model: str, framework: str = None, cache_dir: str = None
    ) -> Union["PreTrainedModel", "TFPreTrainedModel"]:
        """
        Attempts to retrieve a model instance based on the given feature and model name.

        Args:
            feature (`str`):
                The specific feature required by the model.
            model (`str`):
                The name of the model to retrieve.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for model instantiation. If not provided, determined by `FeaturesManager.determine_framework`.

        Returns:
            Union["PreTrainedModel", "TFPreTrainedModel"]: The instantiated model object.
        """
        framework = FeaturesManager.determine_framework(model, framework)
        # 获取特定 feature 对应的模型类
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        try:
            # 尝试从预训练模型加载指定模型
            model = model_class.from_pretrained(model, cache_dir=cache_dir)
        except OSError:
            if framework == "pt":
                # 若出错且框架为 PyTorch，尝试加载 TensorFlow 模型并转换为 PyTorch 格式
                logger.info("Loading TensorFlow model in PyTorch before exporting to ONNX.")
                model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
            else:
                # 若出错且框架不是 PyTorch，尝试加载 PyTorch 模型并转换为 TensorFlow 格式
                logger.info("Loading PyTorch model in TensorFlow before exporting to ONNX.")
                model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir)
        return model

    @staticmethod
    def check_supported_model_or_raise(
        model: Union["PreTrainedModel", "TFPreTrainedModel"], feature: str = "default"
    ) -> Tuple[str, Callable]:
        """
        Checks if a given model supports a specified feature.

        Args:
            model (Union["PreTrainedModel", "TFPreTrainedModel"]):
                The model instance to check.
            feature (`str`, *optional*, defaults to `"default"`):
                The feature name to verify if supported.

        Returns:
            Tuple[str, Callable]:
                - The type of the model (`str`).
                - Callable function from `FeaturesManager._SUPPORTED_MODEL_TYPE` corresponding to the feature.
        """
        # 获取模型类型并替换下划线为破折号
        model_type = model.config.model_type.replace("_", "-")
        # 获取模型名称（如果有）
        model_name = getattr(model, "name", "")
        # 获取模型支持的特性列表
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        # 检查指定特性是否在支持的特性列表中
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]

    def get_config(model_type: str, feature: str) -> OnnxConfig:
        """
        Retrieves the configuration for a specified model type and feature combination.

        Args:
            model_type (`str`):
                The type of model to fetch the configuration for.
            feature (`str`):
                The feature to retrieve the configuration for.

        Returns:
            `OnnxConfig`: Configuration object for the specified model type and feature.
        """
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]
```