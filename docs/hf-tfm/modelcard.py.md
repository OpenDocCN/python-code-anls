# `.\modelcard.py`

```py
# 导入所需的模块和库
import copy  # 导入深拷贝函数
import json  # 导入处理 JSON 的库
import os  # 导入操作系统相关的功能
import warnings  # 导入警告处理模块
from dataclasses import dataclass  # 导入 dataclass 用于创建数据类
from pathlib import Path  # 导入 Path 类用于处理文件路径
from typing import Any, Dict, List, Optional, Union  # 导入类型提示相关功能

import requests  # 导入处理 HTTP 请求的库
import yaml  # 导入处理 YAML 文件的库
from huggingface_hub import model_info  # 导入 Hugging Face Hub 的模型信息功能
from huggingface_hub.utils import HFValidationError  # 导入 Hugging Face Hub 的验证错误处理

from . import __version__  # 导入当前包的版本信息
from .models.auto.modeling_auto import (  # 导入自动生成模型的相关映射名称
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
)
from .training_args import ParallelMode  # 导入并行模式参数
from .utils import (  # 导入工具函数和常量
    MODEL_CARD_NAME,
    cached_file,
    is_datasets_available,
    is_offline_mode,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    logging,
)


TASK_MAPPING = {  # 定义任务与模型映射关系的字典
    "text-generation": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    "image-segmentation": MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
    "fill-mask": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    "object-detection": MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
    "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    "text2text-generation": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    "text-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    "table-question-answering": MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
    "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    "automatic-speech-recognition": {**MODEL_FOR_CTC_MAPPING_NAMES, **MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES},
    "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
}

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
    # 定义结构化的模型卡片类。存储模型卡片以及加载/下载/保存模型卡片的方法。

    # 请阅读以下论文以获取关于各部分的详细信息和解释：“Model Cards for Model Reporting” 作者包括 Margaret Mitchell, Simone Wu, Andrew Zaldivar, 等人提出了模型卡片的建议。链接：https://arxiv.org/abs/1810.03993

    # 注意：可以加载和保存模型卡片到磁盘上。
    """
    
    # 初始化方法，用于创建模型卡片对象
    def __init__(self, **kwargs):
        # 发出警告，表示该类 `ModelCard` 已被弃用，并将在 Transformers 的第五版中移除
        warnings.warn(
            "The class `ModelCard` is deprecated and will be removed in version 5 of Transformers", FutureWarning
        )
        # 推荐的属性来源于 https://arxiv.org/abs/1810.03993（见论文）
        # 设置模型细节
        self.model_details = kwargs.pop("model_details", {})
        # 设置预期使用
        self.intended_use = kwargs.pop("intended_use", {})
        # 设置因素
        self.factors = kwargs.pop("factors", {})
        # 设置度量
        self.metrics = kwargs.pop("metrics", {})
        # 设置评估数据
        self.evaluation_data = kwargs.pop("evaluation_data", {})
        # 设置训练数据
        self.training_data = kwargs.pop("training_data", {})
        # 设置定量分析
        self.quantitative_analyses = kwargs.pop("quantitative_analyses", {})
        # 设置伦理考虑
        self.ethical_considerations = kwargs.pop("ethical_considerations", {})
        # 设置注意事项和建议
        self.caveats_and_recommendations = kwargs.pop("caveats_and_recommendations", {})

        # 打开额外的属性
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                # 如果无法设置属性，则记录错误信息并抛出异常
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    # 将模型卡片对象保存到指定的目录或文件
    def save_pretrained(self, save_directory_or_file):
        """Save a model card object to the directory or file `save_directory_or_file`."""
        # 如果保存目录存在，则使用预定义的文件名保存，方便使用 `from_pretrained` 加载
        if os.path.isdir(save_directory_or_file):
            output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
        else:
            output_model_card_file = save_directory_or_file

        # 将模型卡片对象保存为 JSON 文件
        self.to_json_file(output_model_card_file)
        logger.info(f"Model card saved in {output_model_card_file}")

    # 从 Python 字典中构造一个 `ModelCard` 对象的类方法
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelCard` from a Python dictionary of parameters."""
        return cls(**json_object)

    # 从 JSON 文件中构造一个 `ModelCard` 对象的类方法
    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelCard` from a json file of parameters."""
        # 读取 JSON 文件内容
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        # 解析 JSON 文本为 Python 字典对象
        dict_obj = json.loads(text)
        # 使用字典对象构造一个新的 `ModelCard` 对象
        return cls(**dict_obj)

    # 判断两个 `ModelCard` 对象是否相等的特殊方法
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    # 返回 `ModelCard` 对象的字符串表示形式的特殊方法
    def __repr__(self):
        return str(self.to_json_string())
    # 将当前对象实例序列化为一个 Python 字典
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        # 深拷贝当前对象的所有属性到 output 字典中
        output = copy.deepcopy(self.__dict__)
        return output

    # 将当前对象实例序列化为 JSON 字符串
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        # 调用 to_dict 方法获取对象的字典表示，转换为带缩进和排序键的 JSON 字符串，并添加换行符
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    # 将当前对象实例保存到一个 JSON 文件中
    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        # 打开指定路径的 JSON 文件，使用 UTF-8 编码写入对象的 JSON 字符串表示
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
AUTOGENERATED_TRAINER_COMMENT = """
<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->
"""

AUTOGENERATED_KERAS_COMMENT = """
<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->
"""


TASK_TAG_TO_NAME_MAPPING = {
    "fill-mask": "Masked Language Modeling",  # 映射任务标签 "fill-mask" 到任务名称 "Masked Language Modeling"
    "image-classification": "Image Classification",  # 映射任务标签 "image-classification" 到任务名称 "Image Classification"
    "image-segmentation": "Image Segmentation",  # 映射任务标签 "image-segmentation" 到任务名称 "Image Segmentation"
    "multiple-choice": "Multiple Choice",  # 映射任务标签 "multiple-choice" 到任务名称 "Multiple Choice"
    "object-detection": "Object Detection",  # 映射任务标签 "object-detection" 到任务名称 "Object Detection"
    "question-answering": "Question Answering",  # 映射任务标签 "question-answering" 到任务名称 "Question Answering"
    "summarization": "Summarization",  # 映射任务标签 "summarization" 到任务名称 "Summarization"
    "table-question-answering": "Table Question Answering",  # 映射任务标签 "table-question-answering" 到任务名称 "Table Question Answering"
    "text-classification": "Text Classification",  # 映射任务标签 "text-classification" 到任务名称 "Text Classification"
    "text-generation": "Causal Language Modeling",  # 映射任务标签 "text-generation" 到任务名称 "Causal Language Modeling"
    "text2text-generation": "Sequence-to-sequence Language Modeling",  # 映射任务标签 "text2text-generation" 到任务名称 "Sequence-to-sequence Language Modeling"
    "token-classification": "Token Classification",  # 映射任务标签 "token-classification" 到任务名称 "Token Classification"
    "translation": "Translation",  # 映射任务标签 "translation" 到任务名称 "Translation"
    "zero-shot-classification": "Zero Shot Classification",  # 映射任务标签 "zero-shot-classification" 到任务名称 "Zero Shot Classification"
    "automatic-speech-recognition": "Automatic Speech Recognition",  # 映射任务标签 "automatic-speech-recognition" 到任务名称 "Automatic Speech Recognition"
    "audio-classification": "Audio Classification",  # 映射任务标签 "audio-classification" 到任务名称 "Audio Classification"
}


METRIC_TAGS = [
    "accuracy",  # 表示度量标签 "accuracy"，用于评估模型准确性
    "bleu",  # 表示度量标签 "bleu"，用于评估机器翻译质量
    "f1",  # 表示度量标签 "f1"，用于评估分类和信息检索等任务的准确性
    "matthews_correlation",  # 表示度量标签 "matthews_correlation"，用于评估二分类问题中的相关性
    "pearsonr",  # 表示度量标签 "pearsonr"，用于评估两个变量之间的线性相关性
    "precision",  # 表示度量标签 "precision"，用于评估分类模型中的精确性
    "recall",  # 表示度量标签 "recall"，用于评估分类模型中的召回率
    "rouge",  # 表示度量标签 "rouge"，用于评估文本摘要生成模型的质量
    "sacrebleu",  # 表示度量标签 "sacrebleu"，用于机器翻译任务中的 BLEU 得分
    "spearmanr",  # 表示度量标签 "spearmanr"，用于评估两个变量的非线性相关性
    "wer",  # 表示度量标签 "wer"，用于评估自动语音识别中的词错误率
]


def _listify(obj):
    if obj is None:
        return []  # 如果对象为 None，则返回空列表
    elif isinstance(obj, str):
        return [obj]  # 如果对象为字符串，则返回包含该字符串的列表
    else:
        return obj  # 否则返回原始对象


def _insert_values_as_list(metadata, name, values):
    if values is None:
        return metadata  # 如果值为 None，则返回元数据本身
    if isinstance(values, str):
        values = [values]  # 如果值为字符串，则转换成单元素列表
    values = [v for v in values if v is not None]  # 过滤掉值中的 None 元素
    if len(values) == 0:
        return metadata  # 如果列表为空，则返回元数据本身
    metadata[name] = values  # 将处理后的列表赋给元数据对应的名称
    return metadata  # 返回更新后的元数据


def infer_metric_tags_from_eval_results(eval_results):
    if eval_results is None:
        return {}  # 如果评估结果为 None，则返回空字典
    result = {}  # 初始化结果字典
    for key in eval_results.keys():
        if key.lower().replace(" ", "_") in METRIC_TAGS:
            result[key.lower().replace(" ", "_")] = key  # 将符合度量标签的键添加到结果字典中
        elif key.lower() == "rouge1":
            result["rouge"] = key  # 特别处理 "rouge1"，将其映射为 "rouge"
    return result  # 返回最终的结果字典


def _insert_value(metadata, name, value):
    if value is None:
        return metadata  # 如果值为 None，则返回元数据本身
    metadata[name] = value  # 将值插入到元数据中对应的名称
    return metadata  # 返回更新后的元数据


def is_hf_dataset(dataset):
    if not is_datasets_available():
        return False  # 如果 datasets 库不可用，则返回 False

    from datasets import Dataset, IterableDataset

    return isinstance(dataset, (Dataset, IterableDataset))  # 判断 dataset 是否是 Dataset 或 IterableDataset 类的实例


def _get_mapping_values(mapping):
    result = []  # 初始化结果列表
    for v in mapping.values():
        if isinstance(v, (tuple, list)):
            result += list(v)  # 如果值是元组或列表，则将其展开并添加到结果列表中
        else:
            result.append(v)  # 否则直接添加到结果列表中
    return result  # 返回所有映射值组成的列表


@dataclass
class TrainingSummary:
    model_name: str  # 模型名称
    language: Optional[Union[str, List[str]]] = None  # 语言属性，可以是字符串或字符串列表，默认为 None
    license: Optional[str] = None  # 许可证信息，默认为 None
"""
    # 标签，可以是字符串或字符串列表，用于标识模型的类别或特征
    tags: Optional[Union[str, List[str]]] = None
    # 微调自哪个模型而来的信息
    finetuned_from: Optional[str] = None
    # 任务，可以是字符串或字符串列表，描述模型训练的任务类型
    tasks: Optional[Union[str, List[str]]] = None
    # 数据集，可以是字符串或字符串列表，指定用于训练的数据集名称或描述
    dataset: Optional[Union[str, List[str]]] = None
    # 数据集标签，可以是字符串或字符串列表，用于描述数据集的特征或类别
    dataset_tags: Optional[Union[str, List[str]]] = None
    # 数据集参数，可以是字符串或字符串列表，指定数据集的详细参数
    dataset_args: Optional[Union[str, List[str]]] = None
    # 数据集元数据，是一个字典，包含关于数据集的其他信息
    dataset_metadata: Optional[Dict[str, Any]] = None
    # 评估结果，是一个字典，包含模型评估的指标和结果
    eval_results: Optional[Dict[str, float]] = None
    # 评估结果的行信息，是一个字符串列表，记录评估结果的详细信息
    eval_lines: Optional[List[str]] = None
    # 超参数，是一个字典，包含模型训练时使用的超参数信息
    hyperparameters: Optional[Dict[str, Any]] = None
    # 模型的来源，通常为字符串 "trainer"
    source: Optional[str] = "trainer"

    def __post_init__(self):
        # 根据微调自的模型信息推断默认许可证
        if (
            self.license is None  # 如果许可证为空
            and not is_offline_mode()  # 并且不是离线模式
            and self.finetuned_from is not None  # 并且有微调自的模型信息
            and len(self.finetuned_from) > 0  # 并且微调自的模型信息不为空字符串
        ):
            try:
                # 获取微调自模型的信息
                info = model_info(self.finetuned_from)
                # 遍历模型信息的标签
                for tag in info.tags:
                    # 如果标签以 "license:" 开头
                    if tag.startswith("license:"):
                        # 设置许可证为标签中 "license:" 后的内容
                        self.license = tag[8:]
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, HFValidationError):
                # 处理可能的网络请求错误或验证错误
                pass
    def create_model_index(self, metric_mapping):
        # 初始化模型索引，包含模型名称
        model_index = {"name": self.model_name}

        # 将数据集相关信息转换为列表形式
        dataset_names = _listify(self.dataset)
        dataset_tags = _listify(self.dataset_tags)
        dataset_args = _listify(self.dataset_args)
        dataset_metadata = _listify(self.dataset_metadata)

        # 如果参数数量不足，则用 None 补齐
        if len(dataset_args) < len(dataset_tags):
            dataset_args = dataset_args + [None] * (len(dataset_tags) - len(dataset_args))

        # 创建数据集映射字典，将标签映射到名称
        dataset_mapping = dict(zip(dataset_tags, dataset_names))
        dataset_arg_mapping = dict(zip(dataset_tags, dataset_args))
        dataset_metadata_mapping = dict(zip(dataset_tags, dataset_metadata))

        # 创建任务映射字典，将任务标签映射到任务名称
        task_mapping = {
            task: TASK_TAG_TO_NAME_MAPPING[task] for task in _listify(self.tasks) if task in TASK_TAG_TO_NAME_MAPPING
        }

        # 初始化结果列表
        model_index["results"] = []

        # 如果任务映射和数据集映射都为空，则返回只包含模型名称的列表
        if len(task_mapping) == 0 and len(dataset_mapping) == 0:
            return [model_index]

        # 如果任务映射为空，则将其设置为包含 None 的字典
        if len(task_mapping) == 0:
            task_mapping = {None: None}

        # 如果数据集映射为空，则将其设置为包含 None 的字典
        if len(dataset_mapping) == 0:
            dataset_mapping = {None: None}

        # 遍历所有可能的任务和数据集组合
        all_possibilities = [(task_tag, ds_tag) for task_tag in task_mapping for ds_tag in dataset_mapping]
        for task_tag, ds_tag in all_possibilities:
            result = {}

            # 如果任务标签不为空，则设置任务名称和类型
            if task_tag is not None:
                result["task"] = {"name": task_mapping[task_tag], "type": task_tag}

            # 如果数据集标签不为空，则设置数据集名称、类型以及元数据
            if ds_tag is not None:
                metadata = dataset_metadata_mapping.get(ds_tag, {})
                result["dataset"] = {
                    "name": dataset_mapping[ds_tag],
                    "type": ds_tag,
                    **metadata,
                }
                # 如果数据集参数不为空，则设置参数
                if dataset_arg_mapping[ds_tag] is not None:
                    result["dataset"]["args"] = dataset_arg_mapping[ds_tag]

            # 如果度量映射不为空，则设置度量结果
            if len(metric_mapping) > 0:
                result["metrics"] = []
                for metric_tag, metric_name in metric_mapping.items():
                    result["metrics"].append(
                        {
                            "name": metric_name,
                            "type": metric_tag,
                            "value": self.eval_results[metric_name],
                        }
                    )

            # 如果结果中包含任务、数据集和度量，则将结果添加到模型索引中
            if "task" in result and "dataset" in result and "metrics" in result:
                model_index["results"].append(result)
            else:
                # 否则，记录日志并丢弃结果以避免模型卡片被拒绝
                logger.info(f"Dropping the following result as it does not have all the necessary fields:\n{result}")

        # 返回包含模型索引的列表
        return [model_index]
    # 创建元数据的方法，用于生成模型相关的元数据信息
    def create_metadata(self):
        # 从评估结果推断度量标签的映射关系
        metric_mapping = infer_metric_tags_from_eval_results(self.eval_results)

        # 初始化一个空的元数据字典
        metadata = {}

        # 将语言信息插入元数据字典，作为列表形式存储
        metadata = _insert_values_as_list(metadata, "language", self.language)
        
        # 将许可证信息插入元数据字典，作为单一值存储
        metadata = _insert_value(metadata, "license", self.license)
        
        # 如果模型是从某个基础模型微调而来，且基础模型为非空字符串，则插入基础模型信息
        if self.finetuned_from is not None and isinstance(self.finetuned_from, str) and len(self.finetuned_from) > 0:
            metadata = _insert_value(metadata, "base_model", self.finetuned_from)
        
        # 将标签信息插入元数据字典，作为列表形式存储
        metadata = _insert_values_as_list(metadata, "tags", self.tags)
        
        # 将数据集标签信息插入元数据字典，作为列表形式存储
        metadata = _insert_values_as_list(metadata, "datasets", self.dataset_tags)
        
        # 将度量标签映射中的键（度量名称）作为列表插入元数据字典
        metadata = _insert_values_as_list(metadata, "metrics", list(metric_mapping.keys()))
        
        # 创建模型索引并插入元数据字典中
        metadata["model-index"] = self.create_model_index(metric_mapping)

        # 返回生成的元数据字典
        return metadata

    @classmethod
    def from_trainer(
        cls,
        trainer,
        language=None,
        license=None,
        tags=None,
        model_name=None,
        finetuned_from=None,
        tasks=None,
        dataset_tags=None,
        dataset_metadata=None,
        dataset=None,
        dataset_args=None,
    ):
        # 推断默认数据集
        one_dataset = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
        # 如果数据集来自 HF 数据集且缺少标签、参数或元数据，则推断默认标签
        if is_hf_dataset(one_dataset) and (dataset_tags is None or dataset_args is None or dataset_metadata is None):
            default_tag = one_dataset.builder_name
            # 排除不是来自 Hub 的虚构数据集
            if default_tag not in ["csv", "json", "pandas", "parquet", "text"]:
                # 如果缺少元数据，则创建包含配置名和分割信息的元数据列表
                if dataset_metadata is None:
                    dataset_metadata = [{"config": one_dataset.config_name, "split": str(one_dataset.split)}]
                # 如果缺少标签，则使用默认标签
                if dataset_tags is None:
                    dataset_tags = [default_tag]
                # 如果缺少参数，则使用配置名作为参数
                if dataset_args is None:
                    dataset_args = [one_dataset.config_name]

        # 如果未指定数据集但指定了数据集标签，则将数据集设为数据集标签
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags

        # 推断默认微调自
        if (
            finetuned_from is None
            and hasattr(trainer.model.config, "_name_or_path")
            and not os.path.isdir(trainer.model.config._name_or_path)
        ):
            # 使用模型配置的名称或路径作为微调来源
            finetuned_from = trainer.model.config._name_or_path

        # 推断默认任务标签
        if tasks is None:
            model_class_name = trainer.model.__class__.__name__
            # 遍历任务映射表，根据模型类名获取任务标签
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task

        # 如果未指定模型名称，则使用输出目录的名称作为模型名称
        if model_name is None:
            model_name = Path(trainer.args.output_dir).name
        # 如果模型名称为空字符串，则使用微调来源作为模型名称
        if len(model_name) == 0:
            model_name = finetuned_from

        # 将 `generated_from_trainer` 添加到标签中
        if tags is None:
            tags = ["generated_from_trainer"]
        elif isinstance(tags, str) and tags != "generated_from_trainer":
            tags = [tags, "generated_from_trainer"]
        elif "generated_from_trainer" not in tags:
            tags.append("generated_from_trainer")

        # 解析训练状态日志历史，获取日志行和评估结果
        _, eval_lines, eval_results = parse_log_history(trainer.state.log_history)
        # 从训练器中提取超参数
        hyperparameters = extract_hyperparameters_from_trainer(trainer)

        # 返回构造的类对象，初始化各个参数
        return cls(
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset=dataset,
            dataset_tags=dataset_tags,
            dataset_args=dataset_args,
            dataset_metadata=dataset_metadata,
            eval_results=eval_results,
            eval_lines=eval_lines,
            hyperparameters=hyperparameters,
        )

    @classmethod
    def from_keras(
        cls,
        model,
        model_name,
        keras_history=None,
        language=None,
        license=None,
        tags=None,
        finetuned_from=None,
        tasks=None,
        dataset_tags=None,
        dataset=None,
        dataset_args=None,
        # 接受以下参数并返回新的 HFModelArguments 对象
        ):
        # 如果给定了 dataset 参数：
        if dataset is not None:
            # 如果 dataset 是 HF dataset 并且 dataset_tags 或 dataset_args 为 None：
            if is_hf_dataset(dataset) and (dataset_tags is None or dataset_args is None):
                # 使用 dataset 的构建器名称作为默认标签
                default_tag = dataset.builder_name
                # 排除不是来自 Hub 的虚构数据集
                if default_tag not in ["csv", "json", "pandas", "parquet", "text"]:
                    # 如果 dataset_tags 为 None，则设为默认标签列表
                    if dataset_tags is None:
                        dataset_tags = [default_tag]
                    # 如果 dataset_args 为 None，则设为 dataset 的配置名称列表
                    if dataset_args is None:
                        dataset_args = [dataset.config_name]

        # 如果 dataset 为 None 而 dataset_tags 不为 None，则将 dataset 设置为 dataset_tags
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags

        # 推断默认的 finetuned_from
        if (
            finetuned_from is None
            and hasattr(model.config, "_name_or_path")
            and not os.path.isdir(model.config._name_or_path)
        ):
            # 使用 model.config 的 _name_or_path 属性作为 finetuned_from
            finetuned_from = model.config._name_or_path

        # 推断默认的任务标签:
        if tasks is None:
            # 获取 model 的类名
            model_class_name = model.__class__.__name__
            # 遍历 TASK_MAPPING 中的任务映射
            for task, mapping in TASK_MAPPING.items():
                # 如果 model_class_name 在映射值中
                if model_class_name in _get_mapping_values(mapping):
                    # 设置任务为当前的 task

                    Add ` generated_from_keras_callback to
# 解析 `logs` 参数，该参数可以是 `model.fit()` 返回的 `keras.History` 对象，也可以是传递给 `PushToHubCallback` 的累积日志字典
def parse_keras_history(logs):
    if hasattr(logs, "history"):
        # 如果 `logs` 对象有 `history` 属性，则看起来像是一个 `History` 对象
        if not hasattr(logs, "epoch"):
            # 如果 `logs` 对象没有 `epoch` 属性，表示历史记录为空，返回空结果
            return None, [], {}
        # 将 `epoch` 属性添加到 `logs.history` 字典中
        logs.history["epoch"] = logs.epoch
        # 使用 `logs.history` 替换 `logs`，统一处理为字典格式
        logs = logs.history
    else:
        # 如果 `logs` 不是 `History` 对象，则假设它是一个包含字典列表的训练日志，我们将其转换为字典的列表格式，以匹配 `History` 对象的结构
        logs = {log_key: [single_dict[log_key] for single_dict in logs] for log_key in logs[0]}

    # 初始化空列表 `lines`，用于存储解析后的日志信息
    lines = []
    # 遍历 `epoch` 列表的长度，即遍历每个周期的日志
    for i in range(len(logs["epoch"])):
        # 创建当前周期的字典 `epoch_dict`，将每个日志键值对应到当前周期的值
        epoch_dict = {log_key: log_value_list[i] for log_key, log_value_list in logs.items()}
        # 初始化空字典 `values`，用于存储当前周期的解析后的键值对
        values = {}
        # 遍历 `epoch_dict` 中的每个键值对
        for k, v in epoch_dict.items():
            if k.startswith("val_"):
                # 如果键以 "val_" 开头，将其改为以 "validation_" 开头
                k = "validation_" + k[4:]
            elif k != "epoch":
                # 如果键不是 "epoch"，则将其改为以 "train_" 开头
                k = "train_" + k
            # 将键名按照下划线分割后，每个部分首字母大写，形成更友好的名称
            splits = k.split("_")
            name = " ".join([part.capitalize() for part in splits])
            # 将处理后的键值对加入到 `values` 字典中
            values[name] = v
        # 将当前周期解析后的字典 `values` 添加到 `lines` 列表中
        lines.append(values)

    # 提取评估结果，即最后一个周期解析后的结果
    eval_results = lines[-1]

    # 返回原始日志字典、解析后的周期信息列表 `lines` 和评估结果 `eval_results`
    return logs, lines, eval_results


# 解析 `log_history` 参数，获取 `Trainer` 的中间和最终评估结果
def parse_log_history(log_history):
    # 初始化索引 `idx`，从头开始查找直到找到包含 "train_runtime" 的日志条目
    idx = 0
    while idx < len(log_history) and "train_runtime" not in log_history[idx]:
        idx += 1

    # 如果没有训练日志
    if idx == len(log_history):
        # 将索引减一，从最后一个日志向前查找包含 "eval_loss" 的日志条目
        idx -= 1
        while idx >= 0 and "eval_loss" not in log_history[idx]:
            idx -= 1

        # 如果找到了包含 "eval_loss" 的日志条目，则返回 `None`、`None` 和该日志条目
        if idx >= 0:
            return None, None, log_history[idx]
        else:
            # 如果没有找到包含 "eval_loss" 的日志条目，则返回三个 `None`
            return None, None, None

    # 现在我们可以假设存在训练日志
    # 获取训练日志 `train_log`，即包含 "train_runtime" 的日志条目
    train_log = log_history[idx]
    # 初始化空列表 `lines`，用于存储解析后的日志信息
    lines = []
    # 初始化训练损失为 "No log"
    training_loss = "No log"
    # 遍历日志历史记录中索引范围内的每一个索引 i
    for i in range(idx):
        # 如果当前索引 i 的日志记录包含 "loss" 键
        if "loss" in log_history[i]:
            # 将训练损失记录下来
            training_loss = log_history[i]["loss"]
        
        # 如果当前索引 i 的日志记录包含 "eval_loss" 键
        if "eval_loss" in log_history[i]:
            # 复制当前日志记录中的所有项到 metrics 字典中
            metrics = log_history[i].copy()
            # 移除不需要的项目
            _ = metrics.pop("total_flos", None)
            epoch = metrics.pop("epoch", None)
            step = metrics.pop("step", None)
            _ = metrics.pop("eval_runtime", None)
            _ = metrics.pop("eval_samples_per_second", None)
            _ = metrics.pop("eval_steps_per_second", None)
            _ = metrics.pop("eval_jit_compilation_time", None)
            
            # 初始化一个空字典 values，用于存储需要的指标
            values = {"Training Loss": training_loss, "Epoch": epoch, "Step": step}
            
            # 遍历 metrics 字典中的每一项
            for k, v in metrics.items():
                # 如果当前项的键是 "eval_loss"
                if k == "eval_loss":
                    # 将其值存入 values 字典中作为 "Validation Loss"
                    values["Validation Loss"] = v
                else:
                    # 如果键不是 "eval_loss"，将键按照下划线分割为列表
                    splits = k.split("_")
                    # 将分割后的每个部分的首字母大写，并连接起来作为指标名称
                    name = " ".join([part.capitalize() for part in splits[1:]])
                    # 将该指标的值存入 values 字典中
                    values[name] = v
            
            # 将 values 字典存入 lines 列表中
            lines.append(values)

    # 将 idx 设置为日志历史记录的长度减一
    idx = len(log_history) - 1
    
    # 当 idx 大于等于 0 且日志历史记录中索引为 idx 的项不包含 "eval_loss" 键时循环
    while idx >= 0 and "eval_loss" not in log_history[idx]:
        # 减小 idx 的值
        idx -= 1

    # 如果 idx 大于 0
    if idx > 0:
        # 初始化一个空字典 eval_results，用于存储评估结果
        eval_results = {}
        
        # 遍历日志历史记录中索引为 idx 的项的每一个键值对
        for key, value in log_history[idx].items():
            # 如果键以 "eval_" 开头，去除开头的 "eval_"
            if key.startswith("eval_"):
                key = key[5:]
            # 如果键不是 ["runtime", "samples_per_second", "steps_per_second", "epoch", "step"] 中的一员
            if key not in ["runtime", "samples_per_second", "steps_per_second", "epoch", "step"]:
                # 将键按照下划线分割为列表，每个部分首字母大写，并连接起来作为新键
                camel_cased_key = " ".join([part.capitalize() for part in key.split("_")])
                # 将该键及其对应的值存入 eval_results 字典中
                eval_results[camel_cased_key] = value
        
        # 返回训练日志 train_log，行列表 lines，以及评估结果 eval_results
        return train_log, lines, eval_results
    else:
        # 如果 idx 不大于 0，则返回训练日志 train_log，行列表 lines，以及空的评估结果
        return train_log, lines, None
def extract_hyperparameters_from_keras(model):
    # 导入 keras 模块中的函数和类
    from .modeling_tf_utils import keras

    # 创建一个空字典用于存储超参数
    hyperparameters = {}

    # 检查模型是否具有优化器，并且获取其配置信息
    if hasattr(model, "optimizer") and model.optimizer is not None:
        hyperparameters["optimizer"] = model.optimizer.get_config()
    else:
        hyperparameters["optimizer"] = None

    # 获取全局训练精度策略的名称
    hyperparameters["training_precision"] = keras.mixed_precision.global_policy().name

    # 返回提取的超参数字典
    return hyperparameters


def _maybe_round(v, decimals=4):
    # 如果 v 是浮点数且有小数部分超过指定的小数位数，则返回按小数位数四舍五入后的字符串
    if isinstance(v, float) and len(str(v).split(".")) > 1 and len(str(v).split(".")[1]) > decimals:
        return f"{v:.{decimals}f}"
    # 否则返回 v 的字符串形式
    return str(v)


def _regular_table_line(values, col_widths):
    # 生成 Markdown 表格的一行，包括表格的普通行格式
    values_with_space = [f"| {v}" + " " * (w - len(v) + 1) for v, w in zip(values, col_widths)]
    return "".join(values_with_space) + "|\n"


def _second_table_line(col_widths):
    # 生成 Markdown 表格的第二行，包括表头和数据之间的分隔线格式
    values = ["|:" + "-" * w + ":" for w in col_widths]
    return "".join(values) + "|\n"


def make_markdown_table(lines):
    """
    Create a nice Markdown table from the results in `lines`.
    """
    # 如果 lines 为空或者 None，则返回空字符串
    if lines is None or len(lines) == 0:
        return ""

    # 初始化列宽字典，计算每列的最大宽度
    col_widths = {key: len(str(key)) for key in lines[0].keys()}
    for line in lines:
        for key, value in line.items():
            if col_widths[key] < len(_maybe_round(value)):
                col_widths[key] = len(_maybe_round(value))

    # 构建 Markdown 表格的内容
    table = _regular_table_line(list(lines[0].keys()), list(col_widths.values()))
    table += _second_table_line(list(col_widths.values()))
    for line in lines:
        table += _regular_table_line([_maybe_round(v) for v in line.values()], list(col_widths.values()))
    return table


_TRAINING_ARGS_KEYS = [
    "learning_rate",
    "train_batch_size",
    "eval_batch_size",
    "seed",
]


def extract_hyperparameters_from_trainer(trainer):
    # 从训练器对象中提取超参数，使用预定义的训练参数键
    hyperparameters = {k: getattr(trainer.args, k) for k in _TRAINING_ARGS_KEYS}

    # 如果并行模式不是单GPU模式或非分布式模式，则添加分布式类型
    if trainer.args.parallel_mode not in [ParallelMode.NOT_PARALLEL, ParallelMode.NOT_DISTRIBUTED]:
        hyperparameters["distributed_type"] = (
            "multi-GPU" if trainer.args.parallel_mode == ParallelMode.DISTRIBUTED else trainer.args.parallel_mode.value
        )

    # 如果使用多个设备进行训练，则添加设备数量
    if trainer.args.world_size > 1:
        hyperparameters["num_devices"] = trainer.args.world_size

    # 如果梯度累积步数大于1，则添加梯度累积步数
    if trainer.args.gradient_accumulation_steps > 1:
        hyperparameters["gradient_accumulation_steps"] = trainer.args.gradient_accumulation_steps

    # 计算总的训练批次大小，如果不等于预定义的训练批次大小，则添加总训练批次大小
    total_train_batch_size = (
        trainer.args.train_batch_size * trainer.args.world_size * trainer.args.gradient_accumulation_steps
    )
    if total_train_batch_size != hyperparameters["train_batch_size"]:
        hyperparameters["total_train_batch_size"] = total_train_batch_size

    # 计算总的评估批次大小，如果不等于预定义的评估批次大小，则添加总评估批次大小
    total_eval_batch_size = trainer.args.eval_batch_size * trainer.args.world_size
    if total_eval_batch_size != hyperparameters["eval_batch_size"]:
        hyperparameters["total_eval_batch_size"] = total_eval_batch_size
    # 如果训练器的参数中指定了使用 Adafactor 优化器
    if trainer.args.adafactor:
        # 将超参数中的优化器设置为 Adafactor
        hyperparameters["optimizer"] = "Adafactor"
    else:
        # 否则，使用带有指定参数的 Adam 优化器
        hyperparameters["optimizer"] = (
            f"Adam with betas=({trainer.args.adam_beta1},{trainer.args.adam_beta2}) and"
            f" epsilon={trainer.args.adam_epsilon}"
        )

    # 设置学习率调度器的类型为训练器参数中指定的值
    hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    
    # 如果训练器参数中指定了非零的预热比例
    if trainer.args.warmup_ratio != 0.0:
        # 将预热比例设置到学习率调度器的预热比例参数中
        hyperparameters["lr_scheduler_warmup_ratio"] = trainer.args.warmup_ratio
    
    # 如果训练器参数中指定了非零的预热步数
    if trainer.args.warmup_steps != 0.0:
        # 将预热步数设置到学习率调度器的预热步数参数中
        hyperparameters["lr_scheduler_warmup_steps"] = trainer.args.warmup_steps
    
    # 如果训练器参数中指定了最大步数不等于 -1
    if trainer.args.max_steps != -1:
        # 将最大步数设置到超参数的训练步数参数中
        hyperparameters["training_steps"] = trainer.args.max_steps
    else:
        # 否则，将训练轮数设置到超参数的训练轮数参数中
        hyperparameters["num_epochs"] = trainer.args.num_train_epochs

    # 如果训练器参数中指定了使用混合精度训练
    if trainer.args.fp16:
        # 如果使用了 Apex 框架
        if trainer.use_apex:
            # 将混合精度训练设置为 Apex，并包括指定的优化级别
            hyperparameters["mixed_precision_training"] = f"Apex, opt level {trainer.args.fp16_opt_level}"
        else:
            # 否则，将混合精度训练设置为本地 AMP 支持
            hyperparameters["mixed_precision_training"] = "Native AMP"

    # 如果训练器参数中指定了标签平滑因子不等于 0.0
    if trainer.args.label_smoothing_factor != 0.0:
        # 将标签平滑因子设置到超参数的标签平滑因子参数中
        hyperparameters["label_smoothing_factor"] = trainer.args.label_smoothing_factor

    # 返回设置好的超参数字典
    return hyperparameters
```