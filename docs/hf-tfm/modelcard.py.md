# `.\transformers\modelcard.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
# 配置基类和实用程序

# 导入必要的库
import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError

# 导入自定义模块
from . import __version__
from .models.auto.modeling_auto import (
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
from .training_args import ParallelMode
from .utils import (
    MODEL_CARD_NAME,
    cached_file,
    is_datasets_available,
    is_offline_mode,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    logging,
)

# 定义任务映射关系
TASK_MAPPING = {
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义模型卡片类
class ModelCard:
    r"""
    class ModelCard:
        """
        结构化的模型卡片类。存储模型卡片以及加载/下载/保存模型卡片的方法。
    
        请阅读以下论文以获取有关各节的详细信息和解释: "Model Cards for Model Reporting"，作者为
        Margaret Mitchell、Simone Wu、Andrew Zaldivar、Parker Barnes、Lucy Vasserman、Ben Hutchinson、Elena Spitzer、
        Inioluwa Deborah Raji 和 Timnit Gebru，提出了模型卡片的建议。链接: https://arxiv.org/abs/1810.03993
    
        注意: 模型卡片可以加载和保存到磁盘上。
        """
    
        def __init__(self, **kwargs):
            warnings.warn(
                "The class `ModelCard` is deprecated and will be removed in version 5 of Transformers", FutureWarning
            )
            # 推荐的属性来自 https://arxiv.org/abs/1810.03993 (参见论文)
            self.model_details = kwargs.pop("model_details", {})
            self.intended_use = kwargs.pop("intended_use", {})
            self.factors = kwargs.pop("factors", {})
            self.metrics = kwargs.pop("metrics", {})
            self.evaluation_data = kwargs.pop("evaluation_data", {})
            self.training_data = kwargs.pop("training_data", {})
            self.quantitative_analyses = kwargs.pop("quantitative_analyses", {})
            self.ethical_considerations = kwargs.pop("ethical_considerations", {})
            self.caveats_and_recommendations = kwargs.pop("caveats_and_recommendations", {})
    
            # 开放额外属性
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err
    
        def save_pretrained(self, save_directory_or_file):
            """将模型卡片对象保存到目录或文件 `save_directory_or_file` 中。"""
            if os.path.isdir(save_directory_or_file):
                # 如果使用预定义的名称保存，可以使用 `from_pretrained` 加载
                output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
            else:
                output_model_card_file = save_directory_or_file
    
            self.to_json_file(output_model_card_file)
            logger.info(f"模型卡片已保存在 {output_model_card_file}")
    
        @classmethod
        def from_dict(cls, json_object):
            """从 Python 字典参数构造一个 `ModelCard`。"""
            return cls(**json_object)
    
        @classmethod
        def from_json_file(cls, json_file):
            """从参数的 json 文件构造一个 `ModelCard`。"""
            with open(json_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            dict_obj = json.loads(text)
            return cls(**dict_obj)
    
        def __eq__(self, other):
            return self.__dict__ == other.__dict__
    
        def __repr__(self):
            return str(self.to_json_string())
    # 将实例序列化为 Python 字典
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        # 深拷贝实例的属性字典
        output = copy.deepcopy(self.__dict__)
        # 返回深拷贝后的字典
        return output

    # 将实例序列化为 JSON 字符串
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        # 调用 to_dict 方法将实例转换为字典，然后使用 json.dumps 方法将字典转换为 JSON 字符串
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    # 将实例保存到 JSON 文件中
    def to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        # 打开指定路径的 JSON 文件，以写入模式，编码为 utf-8
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 将实例序列化为 JSON 字符串后写入文件
            writer.write(self.to_json_string())
# 自动生成的 Trainer 注释
AUTOGENERATED_TRAINER_COMMENT = """
<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->
"""

# 自动生成的 Keras 注释
AUTOGENERATED_KERAS_COMMENT = """
<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->
"""

# 任务标签到任务名称的映射
TASK_TAG_TO_NAME_MAPPING = {
    "fill-mask": "Masked Language Modeling",
    "image-classification": "Image Classification",
    "image-segmentation": "Image Segmentation",
    "multiple-choice": "Multiple Choice",
    "object-detection": "Object Detection",
    "question-answering": "Question Answering",
    "summarization": "Summarization",
    "table-question-answering": "Table Question Answering",
    "text-classification": "Text Classification",
    "text-generation": "Causal Language Modeling",
    "text2text-generation": "Sequence-to-sequence Language Modeling",
    "token-classification": "Token Classification",
    "translation": "Translation",
    "zero-shot-classification": "Zero Shot Classification",
    "automatic-speech-recognition": "Automatic Speech Recognition",
    "audio-classification": "Audio Classification",
}

# 指标标签列表
METRIC_TAGS = [
    "accuracy",
    "bleu",
    "f1",
    "matthews_correlation",
    "pearsonr",
    "precision",
    "recall",
    "rouge",
    "sacrebleu",
    "spearmanr",
    "wer",
]

# 将对象转换为列表
def _listify(obj):
    if obj is None:
        return []
    elif isinstance(obj, str):
        return [obj]
    else:
        return obj

# 将值插入为列表
def _insert_values_as_list(metadata, name, values):
    if values is None:
        return metadata
    if isinstance(values, str):
        values = [values]
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return metadata
    metadata[name] = values
    return metadata

# 从评估结果推断指标标签
def infer_metric_tags_from_eval_results(eval_results):
    if eval_results is None:
        return {}
    result = {}
    for key in eval_results.keys():
        if key.lower().replace(" ", "_") in METRIC_TAGS:
            result[key.lower().replace(" ", "_")] = key
        elif key.lower() == "rouge1":
            result["rouge"] = key
    return result

# 将值插入
def _insert_value(metadata, name, value):
    if value is None:
        return metadata
    metadata[name] = value
    return metadata

# 检查是否为 HF 数据集
def is_hf_dataset(dataset):
    if not is_datasets_available():
        return False

    from datasets import Dataset, IterableDataset

    return isinstance(dataset, (Dataset, IterableDataset))

# 获取映射值
def _get_mapping_values(mapping):
    result = []
    for v in mapping.values():
        if isinstance(v, (tuple, list)):
            result += list(v)
        else:
            result.append(v)
    return result

# 训练摘要数据类
@dataclass
class TrainingSummary:
    model_name: str
    language: Optional[Union[str, List[str]]] = None
    license: Optional[str] = None
    # 定义可选参数 tags，可以是字符串或字符串列表，默认为 None
    tags: Optional[Union[str, List[str]]] = None
    # 定义可选参数 finetuned_from，表示从哪个模型微调而来，默认为 None
    finetuned_from: Optional[str] = None
    # 定义可选参数 tasks，可以是字符串或字符串列表，默认为 None
    tasks: Optional[Union[str, List[str]]] = None
    # 定义可选参数 dataset，可以是字符串或字符串列表，默认为 None
    dataset: Optional[Union[str, List[str]]] = None
    # 定义可选参数 dataset_tags，可以是字符串或字符串列表，默认为 None
    dataset_tags: Optional[Union[str, List[str]]] = None
    # 定义可选参数 dataset_args，可以是字符串或字符串列表，默认为 None
    dataset_args: Optional[Union[str, List[str]]] = None
    # 定义可选参数 dataset_metadata，表示数据集的元数据，默认为 None
    dataset_metadata: Optional[Dict[str, Any]] = None
    # 定义可选参数 eval_results，表示评估结果的字典，默认为 None
    eval_results: Optional[Dict[str, float]] = None
    # 定义可选参数 eval_lines，表示评估结果的字符串列表，默认为 None
    eval_lines: Optional[List[str]] = None
    # 定义可选参数 hyperparameters，表示超参数的字典，默认为 None
    hyperparameters: Optional[Dict[str, Any]] = None
    # 定义可选参数 source，表示来源，默认为 "trainer"
    source: Optional[str] = "trainer"

    # 初始化函数，在创建对象后自动调用
    def __post_init__(self):
        # 如果 license 为空且不是离线模式且 finetuned_from 不为空且长度大于0
        if (
            self.license is None
            and not is_offline_mode()
            and self.finetuned_from is not None
            and len(self.finetuned_from) > 0
        ):
            try:
                # 获取 finetuned_from 模型的信息
                info = model_info(self.finetuned_from)
                # 遍历模型信息的标签
                for tag in info.tags:
                    # 如果标签以 "license:" 开头
                    if tag.startswith("license:"):
                        # 将 license 设置为标签中 "license:" 后的内容
                        self.license = tag[8:]
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, HFValidationError):
                pass
    # 创建模型索引，用于记录模型相关信息和评估结果
    def create_model_index(self, metric_mapping):
        # 初始化模型索引字典，包含模型名称
        model_index = {"name": self.model_name}

        # 将数据集相关信息映射到字典中，包括标签到名称、标签到参数、标签到元数据的映射
        dataset_names = _listify(self.dataset)
        dataset_tags = _listify(self.dataset_tags)
        dataset_args = _listify(self.dataset_args)
        dataset_metadata = _listify(self.dataset_metadata)
        if len(dataset_args) < len(dataset_tags):
            dataset_args = dataset_args + [None] * (len(dataset_tags) - len(dataset_args))
        dataset_mapping = dict(zip(dataset_tags, dataset_names))
        dataset_arg_mapping = dict(zip(dataset_tags, dataset_args))
        dataset_metadata_mapping = dict(zip(dataset_tags, dataset_metadata))

        # 将任务标签映射到任务名称，仅保留已知映射
        task_mapping = {
            task: TASK_TAG_TO_NAME_MAPPING[task] for task in _listify(self.tasks) if task in TASK_TAG_TO_NAME_MAPPING
        }

        # 初始化结果列表
        model_index["results"] = []

        # 若任务映射和数据集映射均为空，则返回包含模型名称的列表
        if len(task_mapping) == 0 and len(dataset_mapping) == 0:
            return [model_index]
        # 若任务映射为空，则添加一个空映射
        if len(task_mapping) == 0:
            task_mapping = {None: None}
        # 若数据集映射为空，则添加一个空映射
        if len(dataset_mapping) == 0:
            dataset_mapping = {None: None}

        # 遍历所有可能的任务和数据集组合
        all_possibilities = [(task_tag, ds_tag) for task_tag in task_mapping for ds_tag in dataset_mapping]
        for task_tag, ds_tag in all_possibilities:
            result = {}
            # 如果任务标签不为空，则添加任务名称和类型到结果字典中
            if task_tag is not None:
                result["task"] = {"name": task_mapping[task_tag], "type": task_tag}

            # 如果数据集标签不为空，则添加数据集名称、类型和元数据到结果字典中，并检查是否有数据集参数
            if ds_tag is not None:
                metadata = dataset_metadata_mapping.get(ds_tag, {})
                result["dataset"] = {
                    "name": dataset_mapping[ds_tag],
                    "type": ds_tag,
                    **metadata,
                }
                if dataset_arg_mapping[ds_tag] is not None:
                    result["dataset"]["args"] = dataset_arg_mapping[ds_tag]

            # 如果存在度量映射，则添加度量结果到结果字典中
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

            # 移除部分结果以避免模型卡片被拒绝
            if "task" in result and "dataset" in result and "metrics" in result:
                model_index["results"].append(result)
            else:
                # 如果结果缺少必要字段，则记录日志并丢弃该结果
                logger.info(f"Dropping the following result as it does not have all the necessary fields:\n{result}")

        # 返回包含模型索引的列表
        return [model_index]
    # 创建模型元数据
    def create_metadata(self):
        # 从评估结果推断度量标签映射关系
        metric_mapping = infer_metric_tags_from_eval_results(self.eval_results)

        # 创建空的元数据字典
        metadata = {}
        # 将语言插入元数据字典中
        metadata = _insert_values_as_list(metadata, "language", self.language)
        # 将许可证插入元数据字典中
        metadata = _insert_value(metadata, "license", self.license)
        # 如果模型是从其他模型微调而来，则插入基础模型信息
        if self.finetuned_from is not None and isinstance(self.finetuned_from, str) and len(self.finetuned_from) > 0:
            metadata = _insert_value(metadata, "base_model", self.finetuned_from)
        # 将标签插入元数据字典中
        metadata = _insert_values_as_list(metadata, "tags", self.tags)
        # 将数据集标签插入元数据字典中
        metadata = _insert_values_as_list(metadata, "datasets", self.dataset_tags)
        # 将度量标签插入元数据字典中
        metadata = _insert_values_as_list(metadata, "metrics", list(metric_mapping.keys()))
        # 创建模型索引并插入元数据字典中
        metadata["model-index"] = self.create_model_index(metric_mapping)

        # 返回元数据字典
        return metadata

    # 从训练器创建模型
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
        # 推断默认数据集
        one_dataset = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
        # 检查数据集是否为 HF 数据集，并且 dataset_tags、dataset_args、dataset_metadata 为空时
        if is_hf_dataset(one_dataset) and (dataset_tags is None or dataset_args is None or dataset_metadata is None):
            # 获取默认标签
            default_tag = one_dataset.builder_name
            # 排除不是 Hub 中真实数据集的情况
            if default_tag not in ["csv", "json", "pandas", "parquet", "text"]:
                # 如果 dataset_metadata 为空，则创建包含配置和拆分信息的列表
                if dataset_metadata is None:
                    dataset_metadata = [{"config": one_dataset.config_name, "split": str(one_dataset.split)}]
                # 如果 dataset_tags 为空，则使用默认标签
                if dataset_tags is None:
                    dataset_tags = [default_tag]
                # 如果 dataset_args 为空，则使用配置名称
                if dataset_args is None:
                    dataset_args = [one_dataset.config_name]

        # 如果 dataset 为空且 dataset_tags 不为空，则将 dataset 设置为 dataset_tags
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags

        # 推断默认 finetuned_from
        if (
            finetuned_from is None
            and hasattr(trainer.model.config, "_name_or_path")
            and not os.path.isdir(trainer.model.config._name_or_path)
        ):
            finetuned_from = trainer.model.config._name_or_path

        # 推断默认任务标签
        if tasks is None:
            model_class_name = trainer.model.__class__.__name__
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task

        # 如果 model_name 为空，则使用输出目录的名称
        if model_name is None:
            model_name = Path(trainer.args.output_dir).name
        # 如果 model_name 长度为 0，则使用 finetuned_from
        if len(model_name) == 0:
            model_name = finetuned_from

        # 将 "generated_from_trainer" 添加到标签中
        if tags is None:
            tags = ["generated_from_trainer"]
        elif isinstance(tags, str) and tags != "generated_from_trainer":
            tags = [tags, "generated_from_trainer"]
        elif "generated_from_trainer" not in tags:
            tags.append("generated_from_trainer")

        # 解析训练日志历史，获取评估行和结果
        _, eval_lines, eval_results = parse_log_history(trainer.state.log_history)
        # 从训练器中提取超参数
        hyperparameters = extract_hyperparameters_from_trainer(trainer)

        # 返回一个新的 AutoModelConfig 对象
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
        # 推断默认值从数据集中获取
        if dataset is not None:
            # 检查数据集是否为 HF 数据集，并且 dataset_tags 或 dataset_args 为 None
            if is_hf_dataset(dataset) and (dataset_tags is None or dataset_args is None):
                # 默认标签为数据集的构建器名称
                default_tag = dataset.builder_name
                # 排除不是 Hub 中真实数据集的情况
                if default_tag not in ["csv", "json", "pandas", "parquet", "text"]:
                    # 如果 dataset_tags 为 None，则设置为 default_tag
                    if dataset_tags is None:
                        dataset_tags = [default_tag]
                    # 如果 dataset_args 为 None，则设置为 dataset.config_name
                    if dataset_args is None:
                        dataset_args = [dataset.config_name]

        # 如果 dataset 为 None 且 dataset_tags 不为 None，则将 dataset 设置为 dataset_tags
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags

        # 推断默认的 finetuned_from
        if (
            finetuned_from is None
            and hasattr(model.config, "_name_or_path")
            and not os.path.isdir(model.config._name_or_path)
        ):
            finetuned_from = model.config._name_or_path

        # 推断默认的任务标签
        if tasks is None:
            model_class_name = model.__class__.__name__
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task

        # 将 "generated_from_keras_callback" 添加到标签中
        if tags is None:
            tags = ["generated_from_keras_callback"]
        elif isinstance(tags, str) and tags != "generated_from_keras_callback":
            tags = [tags, "generated_from_keras_callback"]
        elif "generated_from_keras_callback" not in tags:
            tags.append("generated_from_keras_callback")

        # 如果 keras_history 不为 None，则解析 keras_history
        if keras_history is not None:
            _, eval_lines, eval_results = parse_keras_history(keras_history)
        else:
            eval_lines = []
            eval_results = {}
        # 从 Keras 模型中提取超参数
        hyperparameters = extract_hyperparameters_from_keras(model)

        # 返回新的实例
        return cls(
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
            eval_results=eval_results,
            eval_lines=eval_lines,
            hyperparameters=hyperparameters,
            source="keras",
        )
# 解析 `logs`，可以是由 `model.fit()` 返回的 `tf.keras.History` 对象的 `logs` 或者传递给 `PushToHubCallback` 的累积日志 `dict`
# 返回与 `parse_log_history` 返回的行和日志兼容的结果
def parse_keras_history(logs):
    # 检查是否具有 "history" 属性，看起来像一个 `History` 对象
    if hasattr(logs, "history"):
        # 这看起来像一个 `History` 对象
        if not hasattr(logs, "epoch"):
            # 这个历史记录看起来是空的，返回空结果
            return None, [], {}
        # 将 "epoch" 属性添加到 logs 字典中
        logs.history["epoch"] = logs.epoch
        # 将 logs 赋值为 logs 字典
        logs = logs.history
    else:
        # 训练日志是一个字典列表，让我们将其反转为一个字典列表以匹配 History 对象
        logs = {log_key: [single_dict[log_key] for single_dict in logs] for log_key in logs[0]}

    lines = []
    # 遍历 epoch 数组的长度次数
    for i in range(len(logs["epoch"])):
        # 为每个 epoch 创建一个字典
        epoch_dict = {log_key: log_value_list[i] for log_key, log_value_list in logs.items()}
        values = {}
        # 对每个键值对进行处理
        for k, v in epoch_dict.items():
            if k.startswith("val_"):
                # 如果键以 "val_" 开头，则替换为 "validation_"，否则添加 "train_"
                k = "validation_" + k[4:]
            elif k != "epoch":
                k = "train_" + k
            splits = k.split("_")
            # 将每个部分的首字母大写，然后以空格连接起来
            name = " ".join([part.capitalize() for part in splits])
            values[name] = v
        # 将每个 epoch 的结果添加到 lines 中
        lines.append(values)

    # 获取最后一个 epoch 的评估结果
    eval_results = lines[-1]

    return logs, lines, eval_results


# 解析 Trainer 的 `log_history`，以获取中间和最终评估结果
def parse_log_history(log_history):
    idx = 0
    # 查找第一个包含 "train_runtime" 的日志
    while idx < len(log_history) and "train_runtime" not in log_history[idx]:
        idx += 1

    # 如果没有训练日志
    if idx == len(log_history):
        idx -= 1
        # 回退查找最后一个包含 "eval_loss" 的日志
        while idx >= 0 and "eval_loss" not in log_history[idx]:
            idx -= 1

        if idx >= 0:
            # 返回最后一个 "eval_loss" 的日志
            return None, None, log_history[idx]
        else:
            # 如果没有找到 "eval_loss"，则返回空结果
            return None, None, None

    # 从这里开始我们可以假设我们有训练日志了
    train_log = log_history[idx]
    lines = []
    # 初始化训练损失为 "No log"
    training_loss = "No log"
```  
    # 遍历日志历史记录中的索引范围
    for i in range(idx):
        # 如果当前记录包含"loss"字段
        if "loss" in log_history[i]:
            # 获取训练损失值
            training_loss = log_history[i]["loss"]
        # 如果当前记录包含"eval_loss"字段
        if "eval_loss" in log_history[i]:
            # 复制当前记录的所有指标
            metrics = log_history[i].copy()
            # 移除不需要的字段
            _ = metrics.pop("total_flos", None)
            epoch = metrics.pop("epoch", None)
            step = metrics.pop("step", None)
            _ = metrics.pop("eval_runtime", None)
            _ = metrics.pop("eval_samples_per_second", None)
            _ = metrics.pop("eval_steps_per_second", None)
            _ = metrics.pop("eval_jit_compilation_time", None)
            # 创建包含训练损失、epoch和step的字典
            values = {"Training Loss": training_loss, "Epoch": epoch, "Step": step}
            # 遍历剩余的指标
            for k, v in metrics.items():
                # 如果指标是"eval_loss"
                if k == "eval_loss":
                    # 添加验证损失值到字典
                    values["Validation Loss"] = v
                else:
                    # 将指标名称转换为首字母大写的形式
                    splits = k.split("_")
                    name = " ".join([part.capitalize() for part in splits[1:]])
                    values[name] = v
            # 将当前记录的指标值添加到列表中
            lines.append(values)

    # 重置索引为日志历史记录的长度减一
    idx = len(log_history) - 1
    # 从最后一个记录开始向前查找，直到找到包含"eval_loss"字段的记录
    while idx >= 0 and "eval_loss" not in log_history[idx]:
        idx -= 1

    # 如果找到包含"eval_loss"字段的记录
    if idx > 0:
        # 创建一个空字典用于存储评估结果
        eval_results = {}
        # 遍历最后一个包含"eval_loss"字段的记录的所有指标
        for key, value in log_history[idx].items():
            # 如果指标以"eval_"开头
            if key.startswith("eval_"):
                key = key[5:]
            # 如果指标不是特定的字段
            if key not in ["runtime", "samples_per_second", "steps_per_second", "epoch", "step"]:
                # 将指标名称转换为首字母大写的形式，并添加到评估结果字典中
                camel_cased_key = " ".join([part.capitalize() for part in key.split("_")])
                eval_results[camel_cased_key] = value
        # 返回训练日志、指标值列表和评估结果字典
        return train_log, lines, eval_results
    else:
        # 如果未找到包含"eval_loss"字段的记录，则返回训练日志、指标值列表和空值
        return train_log, lines, None
def extract_hyperparameters_from_keras(model):
    # 导入 TensorFlow 库
    import tensorflow as tf

    # 初始化超参数字典
    hyperparameters = {}
    # 检查模型是否有优化器属性，并且不为空
    if hasattr(model, "optimizer") and model.optimizer is not None:
        # 将优化器配置添加到超参数字典中
        hyperparameters["optimizer"] = model.optimizer.get_config()
    else:
        hyperparameters["optimizer"] = None
    # 获取训练精度
    hyperparameters["training_precision"] = tf.keras.mixed_precision.global_policy().name

    return hyperparameters


def _maybe_round(v, decimals=4):
    # 如果值是浮点数且小数位数超过指定精度，则四舍五入
    if isinstance(v, float) and len(str(v).split(".")) > 1 and len(str(v).split(".")[1]) > decimals:
        return f"{v:.{decimals}f}"
    return str(v)


def _regular_table_line(values, col_widths):
    # 生成普通表格行
    values_with_space = [f"| {v}" + " " * (w - len(v) + 1) for v, w in zip(values, col_widths)]
    return "".join(values_with_space) + "|\n"


def _second_table_line(col_widths):
    # 生成表格第二行，用于分隔表头和内容
    values = ["|:" + "-" * w + ":" for w in col_widths]
    return "".join(values) + "|\n"


def make_markdown_table(lines):
    """
    Create a nice Markdown table from the results in `lines`.
    """
    # 如果行为空，则返回空字符串
    if lines is None or len(lines) == 0:
        return ""
    # 计算每列的最大宽度
    col_widths = {key: len(str(key)) for key in lines[0].keys()}
    for line in lines:
        for key, value in line.items():
            if col_widths[key] < len(_maybe_round(value)):
                col_widths[key] = len(_maybe_round(value))

    # 生成表格内容
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
    # 初始化超参数字典，从训练器中提取指定键的值
    hyperparameters = {k: getattr(trainer.args, k) for k in _TRAINING_ARGS_KEYS}

    # 如果并行模式不是非并行或非分布式
    if trainer.args.parallel_mode not in [ParallelMode.NOT_PARALLEL, ParallelMode.NOT_DISTRIBUTED]:
        # 添加分布式类型到超参数字典中
        hyperparameters["distributed_type"] = (
            "multi-GPU" if trainer.args.parallel_mode == ParallelMode.DISTRIBUTED else trainer.args.parallel_mode.value
        )
    # 如果世界大小大于1
    if trainer.args.world_size > 1:
        # 添加设备数量到超参数字典中
        hyperparameters["num_devices"] = trainer.args.world_size
    # 如果梯度累积步数大于1
    if trainer.args.gradient_accumulation_steps > 1:
        # 添加梯度累积步数到超参数字典中
        hyperparameters["gradient_accumulation_steps"] = trainer.args.gradient_accumulation_steps

    # 计算总训练批次大小和总评估批次大小
    total_train_batch_size = (
        trainer.args.train_batch_size * trainer.args.world_size * trainer.args.gradient_accumulation_steps
    )
    if total_train_batch_size != hyperparameters["train_batch_size"]:
        hyperparameters["total_train_batch_size"] = total_train_batch_size
    total_eval_batch_size = trainer.args.eval_batch_size * trainer.args.world_size
    if total_eval_batch_size != hyperparameters["eval_batch_size"]:
        hyperparameters["total_eval_batch_size"] = total_eval_batch_size

    # 如果使用 Adafactor 优化器
    if trainer.args.adafactor:
        hyperparameters["optimizer"] = "Adafactor"
    # 如果没有指定优化器，则使用默认的 Adam 优化器，并设置相关参数
    else:
        hyperparameters["optimizer"] = (
            f"Adam with betas=({trainer.args.adam_beta1},{trainer.args.adam_beta2}) and"
            f" epsilon={trainer.args.adam_epsilon}"
        )

    # 设置学习率调度器的类型
    hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    # 如果设置了学习率预热比例，则记录下来
    if trainer.args.warmup_ratio != 0.0:
        hyperparameters["lr_scheduler_warmup_ratio"] = trainer.args.warmup_ratio
    # 如果设置了学习率预热步数，则记录下来
    if trainer.args.warmup_steps != 0.0:
        hyperparameters["lr_scheduler_warmup_steps"] = trainer.args.warmup_steps
    # 如果设置了最大训练步数，则记录下来
    if trainer.args.max_steps != -1:
        hyperparameters["training_steps"] = trainer.args.max_steps
    # 否则记录下训练的总轮数
    else:
        hyperparameters["num_epochs"] = trainer.args.num_train_epochs

    # 如果启用了混合精度训练
    if trainer.args.fp16:
        # 如果使用 Apex 混合精度训练，则记录下来
        if trainer.use_apex:
            hyperparameters["mixed_precision_training"] = f"Apex, opt level {trainer.args.fp16_opt_level}"
        # 否则记录下使用 Native AMP 进行混合精度训练
        else:
            hyperparameters["mixed_precision_training"] = "Native AMP"

    # 如果设置了标签平滑因子，则记录下来
    if trainer.args.label_smoothing_factor != 0.0:
        hyperparameters["label_smoothing_factor"] = trainer.args.label_smoothing_factor

    # 返回记录了超参数信息的字典
    return hyperparameters
```