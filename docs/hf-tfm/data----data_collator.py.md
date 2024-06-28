# `.\data\data_collator.py`

```py
# 导入必要的模块和类
import random  # 导入随机数模块
import warnings  # 导入警告模块
from collections.abc import Mapping  # 从collections.abc模块导入Mapping类
from dataclasses import dataclass  # 导入dataclass装饰器
from random import randint  # 从random模块导入randint函数
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union  # 导入多种类型声明

import numpy as np  # 导入NumPy模块

from ..models.bert import BertTokenizer, BertTokenizerFast  # 从上级目录的models.bert模块导入BertTokenizer和BertTokenizerFast类
from ..tokenization_utils_base import PreTrainedTokenizerBase  # 从上级目录的tokenization_utils_base模块导入PreTrainedTokenizerBase类
from ..utils import PaddingStrategy  # 从上级目录的utils模块导入PaddingStrategy类

InputDataClass = NewType("InputDataClass", Any)  # 定义新类型InputDataClass

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])  # 定义新类型DataCollator


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        # 确定返回的张量类型，默认与实例的return_tensors属性相同
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 如果返回类型为"tf"，调用tf_call方法处理features
        if return_tensors == "tf":
            return self.tf_call(features)
        # 如果返回类型为"pt"，调用torch_call方法处理features
        elif return_tensors == "pt":
            return self.torch_call(features)
        # 如果返回类型为"np"，调用numpy_call方法处理features
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            # 如果返回类型不是预期的类型，抛出值错误异常
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """
    
    # 避免使用快速分词器时触发的填充警告
    # 如果tokenizer没有deprecation_warnings属性，直接调用pad方法进行填充
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # 保存警告状态，并且禁用相关警告
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        # 调用tokenizer的pad方法进行填充
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # 恢复警告状态
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    """
    """
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    
    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    
    # 根据 `return_tensors` 参数的值选择合适的数据收集器函数并返回结果

    if return_tensors == "pt":
        # 如果 `return_tensors` 是 "pt"，则使用 PyTorch 默认的数据收集器
        return torch_default_data_collator(features)
    elif return_tensors == "tf":
        # 如果 `return_tensors` 是 "tf"，则使用 TensorFlow 默认的数据收集器
        return tf_default_data_collator(features)
    elif return_tensors == "np":
        # 如果 `return_tensors` 是 "np"，则使用 NumPy 默认的数据收集器
        return numpy_default_data_collator(features)
@dataclass
class DefaultDataCollator(DataCollatorMixin):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.

    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.

    Args:
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # If return_tensors is not provided, default to the value set during initialization
        if return_tensors is None:
            return_tensors = self.return_tensors
        # Call the default_data_collator function with the specified return_tensors value
        return default_data_collator(features, return_tensors)


def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    # If features list contains objects that are not mappings, convert them to dictionaries
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    # Retrieve the first feature dictionary
    first = features[0]
    # Initialize an empty batch dictionary
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        # Extract the label value and determine its dtype (long or float)
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        # Create a tensor batch["labels"] containing labels from all features
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        # Handle case where label_ids are present
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        # Process each key-value pair in the first feature dictionary
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def tf_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import tensorflow as tf

    # This function is intended to collate data for TensorFlow, but its implementation is incomplete.
    # Further code would handle collation similar to the torch_default_data_collator.
    pass
    # 检查 features 列表中第一个元素是否是 Mapping 类型（字典类型）
    if not isinstance(features[0], Mapping):
        # 如果不是 Mapping 类型，则将 features 中的每个元素转换为字典类型
        features = [vars(f) for f in features]
    
    # 获取 features 中的第一个元素
    first = features[0]
    
    # 初始化空字典 batch
    batch = {}

    # 处理标签数据的特殊情况。
    # 确保使用正确的数据类型创建张量
    # （通常应该自动处理，但我们需要确保这一点。）
    if "label" in first and first["label"] is not None:
        label_col_name = "label"
    elif "label_ids" in first and first["label_ids"] is not None:
        label_col_name = "label_ids"
    elif "labels" in first and first["labels"] is not None:
        label_col_name = "labels"
    else:
        label_col_name = None
    
    # 如果存在标签列名
    if label_col_name is not None:
        # 根据第一个元素的标签数据类型，确定 dtype
        if isinstance(first[label_col_name], tf.Tensor):
            dtype = tf.int64 if first[label_col_name].dtype.is_integer else tf.float32
        elif isinstance(first[label_col_name], np.ndarray) or isinstance(first[label_col_name], np.generic):
            dtype = tf.int64 if np.issubdtype(first[label_col_name].dtype, np.integer) else tf.float32
        elif isinstance(first[label_col_name], (tuple, list)):
            dtype = tf.int64 if isinstance(first[label_col_name][0], int) else tf.float32
        else:
            dtype = tf.int64 if isinstance(first[label_col_name], int) else tf.float32
        
        # 将 features 中的标签数据转换为张量，存储在 batch 中的 "labels" 键下
        batch["labels"] = tf.convert_to_tensor([f[label_col_name] for f in features], dtype=dtype)
    
    # 处理除标签以外的所有可能键。
    # 再次使用第一个元素来确定哪些键/值对在此模型中不为 None。
    for k, v in first.items():
        if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
            # 如果值是张量或者 numpy 数组，则将 features 中的相应值堆叠为张量
            if isinstance(v, (tf.Tensor, np.ndarray)):
                batch[k] = tf.stack([f[k] for f in features])
            else:
                # 否则，将 features 中的相应值转换为张量
                batch[k] = tf.convert_to_tensor([f[k] for f in features])

    # 返回构建好的 batch 字典
    return batch
# 根据输入特征列表创建批处理数据字典，适用于 NumPy 默认数据格式
def numpy_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    # 如果第一个特征不是映射类型，则将每个特征对象转换为其变量字典表示
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    # 获取第一个特征对象
    first = features[0]
    # 初始化批处理字典
    batch = {}

    # 处理标签的特殊情况
    # 确保使用正确类型创建张量
    # （虽然通常应该自动转换，但我们还是确保类型正确）
    if "label" in first and first["label"] is not None:
        # 如果标签是 NumPy 数组，则将其转换为标量
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        # 确定标签数据类型
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        # 如果标签 IDs 是 NumPy 数组，则堆叠它们
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            # 否则，确定标签 IDs 的数据类型
            dtype = np.int64 if isinstance(first["label_ids"][0], int) else np.float32
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    # 处理所有其他可能的键
    # 再次使用第一个元素确定该模型中哪些键/值不为 None
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, np.ndarray):
                # 如果值是 NumPy 数组，则堆叠它们
                batch[k] = np.stack([f[k] for f in features])
            else:
                # 否则，将值转换为数组
                batch[k] = np.array([f[k] for f in features])

    # 返回批处理数据字典
    return batch
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            用于编码数据的分词器。
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            选择一种策略来对返回的序列进行填充（根据模型的填充位置和填充索引），可选值包括：
            
            - `True` 或 `'longest'`（默认）：对批次中最长的序列进行填充（如果只提供一个序列，则不进行填充）。
            - `'max_length'`：按照参数 `max_length` 指定的最大长度进行填充，或者如果未提供该参数，则按照模型可接受的最大输入长度进行填充。
            - `False` 或 `'do_not_pad'`：不进行填充（即可以输出长度不同的序列批次）。
        max_length (`int`, *optional*):
            返回列表的最大长度，也可选用于填充长度（见上文）。
        pad_to_multiple_of (`int`, *optional*):
            如果设置，将序列填充到提供的值的倍数。
            
            这对于在具有计算能力 >= 7.5（Volta）的 NVIDIA 硬件上启用 Tensor Core 特别有用。
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            要返回的张量类型。允许的值有 "np"、"pt" 和 "tf"。
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 调用 pad_without_fast_tokenizer_warning 函数进行批量填充
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # 如果 batch 中有 "label" 键，将其重命名为 "labels"，并删除 "label"
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        # 如果 batch 中有 "label_ids" 键，将其重命名为 "labels"，并删除 "label_ids"
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        # 返回处理后的 batch 字典
        return batch
@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase  # Tokenizer对象，用于数据编码
    padding: Union[bool, str, PaddingStrategy] = True  # 序列填充策略：可以是布尔值、字符串或填充策略对象，默认为True
    max_length: Optional[int] = None  # 返回列表的最大长度及填充长度（可选）
    pad_to_multiple_of: Optional[int] = None  # 如果设置，将序列填充到提供的值的倍数（可选）
    label_pad_token_id: int = -100  # 标签填充时使用的ID，默认为-100，PyTorch损失函数会自动忽略这些ID
    return_tensors: str = "pt"  # 返回的Tensor类型，默认为"pt"，可选值有"np"、"pt"和"tf"
    # 定义一个使用 Torch 处理特征的方法
    def torch_call(self, features):
        # 导入 PyTorch 库
        import torch

        # 确定标签名称是 "label" 还是 "labels"
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果特征中包含标签，提取所有标签值到列表；否则设置标签为 None
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # 剔除特征中的标签，生成没有标签的特征字典列表
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # 使用自定义函数进行填充（此处函数的具体实现未显示），生成批处理数据
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",  # 返回 PyTorch 张量
        )

        # 如果没有标签，直接返回批处理数据
        if labels is None:
            return batch

        # 获取输入序列的长度
        sequence_length = batch["input_ids"].shape[1]
        # 获取填充的位置（左或右）
        padding_side = self.tokenizer.padding_side

        # 定义一个函数，将张量或可迭代对象转换为列表
        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        # 根据填充的位置对标签进行填充
        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        # 将填充后的标签转换为 PyTorch 的 int64 类型张量
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch

    # 定义一个使用 TensorFlow 处理特征的方法
    def tf_call(self, features):
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 确定标签名称是 "label" 还是 "labels"
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果特征中包含标签，提取所有标签值到列表；否则设置标签为 None
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # 使用自定义函数进行填充（此处函数的具体实现未显示），生成批处理数据
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # 如果标签为 None，则返回 TensorFlow 张量；否则不进行转换
            return_tensors="tf" if labels is None else None,
        )

        # 如果没有标签，直接返回批处理数据
        if labels is None:
            return batch

        # 获取输入序列的长度
        sequence_length = tf.convert_to_tensor(batch["input_ids"]).shape[1]
        # 获取填充的位置（左或右）
        padding_side = self.tokenizer.padding_side

        # 根据填充的位置对标签进行填充
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        # 将填充后的标签转换为 TensorFlow 的 int64 类型张量，并将批处理字典中的所有值转换为 TensorFlow 张量
        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        return batch
    # 定义一个方法，用于处理特征数据并返回一个批次的 numpy 数组
    def numpy_call(self, features):
        # 确定标签名称是 "label" 还是 "labels"
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果特征中包含标签信息，则从每个特征中提取标签，否则设为 None
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # 使用自定义的 pad_without_fast_tokenizer_warning 函数对特征进行填充，转换成批次
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # 如果没有标签信息，则返回的批次为 numpy 数组
            return_tensors="np" if labels is None else None,
        )

        # 如果特征中没有标签信息，则直接返回批次
        if labels is None:
            return batch

        # 计算输入序列长度
        sequence_length = np.array(batch["input_ids"]).shape[1]
        # 获取填充位置（左侧或右侧）
        padding_side = self.tokenizer.padding_side
        # 根据填充位置，为每个标签添加填充标记，使它们长度相同
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        # 将批次中的每个键值对的值转换为 numpy 数组，类型为 int64
        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        # 返回处理后的批次
        return batch
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def _tf_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    import tensorflow as tf

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.convert_to_tensor(e, dtype=tf.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    
    # Prepare paddings based on tensor rank.
    result = []
    rank = tf.rank(examples[0])
    paddings = np.zeros((rank, 2), dtype=np.int32)
    # 遍历给定的示例列表
    for example in examples:
        # 检查分词器的填充位置是否在右侧
        if tokenizer.padding_side == "right":
            # 如果填充在右侧，计算需要填充的长度并更新填充数组的第一行第二列
            paddings[0, 1] = max_length - len(example)
        else:
            # 如果填充在左侧，计算需要填充的长度并更新填充数组的第一行第一列
            paddings[0, 0] = max_length - len(example)
        # 使用 TensorFlow 的填充函数对示例进行填充，使用填充数组和分词器的填充标记值
        result.append(tf.pad(example, paddings, constant_values=tokenizer.pad_token_id))
    # 将填充后的示例堆叠成一个张量，沿着第一个维度（批次维度）
    return tf.stack(result, axis=0)
# 定义一个数据收集器，用于序列到序列模型的数据处理
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    # 定义函数参数和类型注解，说明函数的输入参数和可选参数
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            用于对数据进行编码的分词器。
        model ([`PreTrainedModel`], *optional*):
            正在训练的模型。如果设置并且具有 *prepare_decoder_input_ids_from_labels* 方法，
            则使用它来准备 *decoder_input_ids*。
            
            当使用 *label_smoothing* 时，这很有用，可以避免重复计算损失。
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            选择一种策略来填充返回的序列（根据模型的填充方向和填充索引），可选值包括：
    
            - `True` 或 `'longest'`（默认）：填充到批次中最长的序列（如果只提供单个序列，则不填充）。
            - `'max_length'`：填充到指定的最大长度（通过参数 `max_length` 提供），或者如果未提供该参数，则填充到模型的最大可接受输入长度。
            - `False` 或 `'do_not_pad'`：不填充（即可以输出具有不同长度的序列批次）。
        max_length (`int`, *optional*):
            返回列表的最大长度，也可以作为填充长度（参见上文）。
        pad_to_multiple_of (`int`, *optional*):
            如果设置，将序列填充到提供的值的倍数。
    
            这对于在具有计算能力 >= 7.5（Volta）的 NVIDIA 硬件上启用张量核心特别有用。
        label_pad_token_id (`int`, *optional*, defaults to -100):
            用于填充标签时的标识符（-100 将被 PyTorch 损失函数自动忽略）。
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            要返回的张量类型。允许的值有 "np"、"pt" 和 "tf"。
    def __call__(self, features, return_tensors=None):
        # 如果没有指定返回张量类型，则使用预设的 return_tensors
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 从 features 中提取标签列表，如果 features 的第一个元素包含 "labels" 键
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # 在调用 `tokenizer.pad` 之前，需要对标签进行填充，因为该方法不会进行填充，并且需要所有标签长度相同以返回张量
        if labels is not None:
            # 计算最长的标签长度
            max_label_length = max(len(l) for l in labels)
            # 如果指定了 pad_to_multiple_of，调整最大标签长度使其成为该值的倍数
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            # 获取填充的位置（左侧或右侧）
            padding_side = self.tokenizer.padding_side
            # 对每个 feature 进行标签填充
            for feature in features:
                # 计算需要填充的空位，用 label_pad_token_id 填充
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                # 如果标签是列表
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                # 如果填充在右侧
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                # 如果填充在左侧
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # 使用 pad_without_fast_tokenizer_warning 函数对 features 进行填充，避免使用快速分词器警告
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # 准备 decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            # 根据标签准备 decoder_input_ids
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        # 返回处理后的 features
        return features
    """
    Language modeling数据收集器。如果输入的长度不同，输入会被动态填充到一个batch的最大长度。

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            用于编码数据的分词器。
        mlm (`bool`, *optional*, defaults to `True`):
            是否使用masked language modeling。如果设置为`False`，则标签与输入相同，忽略填充的标记（通过将它们设置为-100）。否则，非masked的标记为-100，要预测的masked标记为其他值。
        mlm_probability (`float`, *optional*, defaults to 0.15):
            当`mlm`设置为`True`时，随机mask输入中的token的概率。
        pad_to_multiple_of (`int`, *optional*):
            如果设置，则将序列填充到提供的值的倍数。
        return_tensors (`str`):
            要返回的Tensor类型。允许的值为"np"、"pt"和"tf"。

    <Tip>

    为了最佳性能，此数据收集器应与具有项为字典或BatchEncoding的数据集一起使用，这些数据集具有"special_tokens_mask"键，该键由[`PreTrainedTokenizer`]或[`PreTrainedTokenizerFast`]返回，参数`return_special_tokens_mask=True`。

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    @staticmethod
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ):
        """
        用于在TensorFlow中执行token masking的函数。

        Args:
            inputs (Any): 输入的数据（例如token IDs）。
            vocab_size (int): 词汇表的大小。
            mask_token_id (int): 要用作masked token的token ID。
            special_tokens_mask (Optional[Any]): 特殊token的mask，与词汇表外的token相关。

        """
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        # 将 mask_token_id 转换为与 inputs 相同的数据类型
        mask_token_id = tf.cast(mask_token_id, inputs.dtype)

        # 获取输入张量的形状
        input_shape = tf.shape(inputs)

        # 为了进行 MLM 训练，以概率 self.mlm_probability 对每个序列中的部分 token 进行掩码操作
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask

        # 用 -100 替换 labels 中未被掩码的位置，因为损失只计算掩码 token 的损失
        labels = tf.where(masked_indices, inputs, -100)

        # 80% 的概率用 mask_token_id 替换掩码的输入 token
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices
        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # 10% 的概率用随机单词替换掩码的输入 token
        indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=inputs.dtype)
        inputs = tf.where(indices_random, random_words, inputs)

        # 剩余 10% 的概率保持掩码的输入 token 不变
        return inputs, labels
    # 定义一个 TensorFlow 模型的调用函数，处理输入的样本列表并返回预测结果字典
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import tensorflow as tf

        # 根据输入样本类型的不同进行处理：如果是字典，则使用自定义函数进行填充和转换为张量
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 否则，将样本列表转换为包含 "input_ids" 键的字典，使用内置函数进行填充
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果预处理了特殊 token 掩码，则从字典中移除该项
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        # 如果采用 MLM（Masked Language Modeling），则进行相应处理
        if self.mlm:
            if special_tokens_mask is None:
                # 如果没有预处理特殊 token 掩码，根据输入的 input_ids 创建掩码
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                # 将掩码转换为 TensorFlow 中的布尔类型张量
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                # 否则，直接将已有的特殊 token 掩码转换为 TensorFlow 的布尔类型张量
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            
            # 使用 TensorFlow 函数 tf_mask_tokens 处理 input_ids 和 labels，并更新 batch 字典
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            # 如果不是 MLM 模式，则直接将 input_ids 作为 labels，同时处理 padding 的情况
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                # 将 padding 的位置替换为 -100
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                # 如果没有定义 pad_token_id，创建 labels 的深拷贝以防万一
                labels = tf.identity(labels)
            batch["labels"] = labels
        
        # 返回处理后的 batch 字典，其中包含处理过的 input_ids 和相应的 labels
        return batch
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 处理输入数据 examples，可以是字典或列表，根据不同类型进行填充和转换为张量。
        if isinstance(examples[0], Mapping):
            # 对字典类型的 examples 进行填充，使用自定义的 pad_without_fast_tokenizer_warning 函数进行填充，并返回 PyTorch 张量。
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 对列表类型的 examples 进行处理，仅填充 "input_ids" 键，调用 _torch_collate_batch 函数进行填充。
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果特殊标记掩码已经预处理，则从字典中移除。
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            # 如果是 MLM（Masked Language Modeling）任务，调用 torch_mask_tokens 函数处理 input_ids 和 labels。
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # 如果不是 MLM 任务，将 input_ids 复制到 labels，并根据 tokenizer 的 pad_token_id 设置 labels 中相应位置的值为 -100。
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        准备用于掩码语言建模的输入/标签：80% MASK，10% 随机词，10% 原始词。
        """
        import torch

        labels = inputs.clone()
        # 对每个序列进行 MLM 训练时，以概率 self.mlm_probability 对输入进行掩码。
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            # 如果特殊标记掩码为空，则使用 tokenizer 获取每个序列的特殊标记掩码。
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 根据特殊标记掩码，将概率矩阵中的特定位置置为 0.0。
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # 使用伯努利分布生成掩码索引。
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算掩码位置的损失

        # 80% 的时间，用 tokenizer.mask_token ([MASK]) 替换掩码位置的输入标记。
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% 的时间，用随机词替换掩码位置的输入标记。
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余 10% 的时间，保持掩码位置的输入标记不变。
        return inputs, labels
    # 定义一个方法，用于处理包含各种数据结构的示例列表，并返回处理后的结果字典
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 如果第一个示例是字典类型，则使用适当的填充方式和转换为张量
        if isinstance(examples[0], Mapping):
            # 使用适当的填充方法（避免快速分词器警告），将示例列表转换为 NumPy 张量
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 如果示例不是字典类型，则只包含输入 ID 的批次
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果已预处理特殊标记掩码，则从字典中弹出
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # 如果是 MLM 任务，调用方法对输入 ID 进行掩码处理，并将结果存入 batch 中
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # 如果不是 MLM 任务，则创建 labels 副本，并将填充标记转换为 -100
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        # 返回处理后的批次数据字典
        return batch
    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # 创建输入的副本作为标签
        labels = np.copy(inputs)

        # 创建一个与输入形状相同的概率矩阵，每个位置的值为 self.mlm_probability
        probability_matrix = np.full(labels.shape, self.mlm_probability)

        # 如果特殊标记掩码为空，则根据每个序列的值获取特殊标记掩码
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        # 将特殊标记掩码位置的概率设为 0，这些位置不会被选为被屏蔽的位置
        probability_matrix[special_tokens_mask] = 0

        # 使用二项分布随机生成屏蔽的索引
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)

        # 将未屏蔽的标签设为 -100，用于损失计算
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% 的概率，将屏蔽的输入标记替换为 tokenizer.mask_token_id
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% 的概率，将屏蔽的输入标记替换为随机词
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # 剩余的 10% 的概率，保持屏蔽的输入标记不变

        # 返回处理后的输入和标签
        return inputs, labels
@DataCollatorForWholeWordMask
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Determine if examples are provided as a list of mappings or as a list of input_ids
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]  # Extract input_ids from each example mapping
        else:
            input_ids = examples  # Examples are directly input_ids, wrap each in a mapping

        # Collate input_ids into a batch tensor respecting tokenizer's padding rules
        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):  # Convert input_ids to tokens using tokenizer
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, mark sub-words with "##", e.g., [喜,欢]->[喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])  # Positions in input_ids that are sub-words
                len_seq = len(e["input_ids"])  # Length of the input sequence
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]  # Prefix sub-word tokens with "##"

            mask_labels.append(self._whole_word_mask(ref_tokens))  # Apply whole word masking to tokens

        # Collate mask_labels into a batch tensor respecting tokenizer's padding rules
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # Mask input_ids and create labels for masked language modeling
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)

        return {"input_ids": inputs, "labels": labels}
    # 定义 TensorFlow 版本的调用方法，接受一个例子列表，并返回处理后的输入和标签字典
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 检查第一个例子的类型，若为映射类型（字典），则提取其中的 "input_ids" 列表
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            # 否则，假设每个例子本身就是一个 input_ids 列表，将其赋值给 input_ids，并用例子包装成带 "input_ids" 键的字典列表
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        # 调用内部函数 _tf_collate_batch，将 input_ids 列表和 tokenizer 进行批处理
        batch_input = _tf_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # 初始化一个空列表，用于存储每个例子的掩码标签
        mask_labels = []
        for e in examples:
            ref_tokens = []
            # 遍历每个例子中的 input_ids，将每个 id 转换为对应的 token
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # 对于中文 token，如果指定了 "chinese_ref" 键，需添加额外的标记 "##" 标识子词，例如 [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # 将处理后的 token 列表传入 _whole_word_mask 方法，得到该例子的掩码标签，添加到 mask_labels 列表中
            mask_labels.append(self._whole_word_mask(ref_tokens))

        # 再次调用 _tf_collate_batch，将 mask_labels 列表和 tokenizer 进行批处理，得到批量化的掩码标签
        batch_mask = _tf_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        # 调用对象自身的 tf_mask_tokens 方法，传入批量化的输入和掩码标签，得到 inputs 和 labels，返回作为字典的 "input_ids" 和 "labels"
        inputs, labels = self.tf_mask_tokens(tf.cast(batch_input, tf.int64), batch_mask)
        return {"input_ids": inputs, "labels": labels}

    # 定义 NumPy 版本的调用方法，接受一个例子列表，并返回处理后的输入和标签字典
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 检查第一个例子的类型，若为映射类型（字典），则提取其中的 "input_ids" 列表
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            # 否则，假设每个例子本身就是一个 input_ids 列表，将其赋值给 input_ids，并用例子包装成带 "input_ids" 键的字典列表
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        # 调用内部函数 _numpy_collate_batch，将 input_ids 列表和 tokenizer 进行批处理
        batch_input = _numpy_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # 初始化一个空列表，用于存储每个例子的掩码标签
        mask_labels = []
        for e in examples:
            ref_tokens = []
            # 遍历每个例子中的 input_ids，将每个 id 转换为对应的 token
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # 对于中文 token，如果指定了 "chinese_ref" 键，需添加额外的标记 "##" 标识子词，例如 [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # 将处理后的 token 列表传入 _whole_word_mask 方法，得到该例子的掩码标签，添加到 mask_labels 列表中
            mask_labels.append(self._whole_word_mask(ref_tokens))

        # 再次调用 _numpy_collate_batch，将 mask_labels 列表和 tokenizer 进行批处理，得到批量化的掩码标签
        batch_mask = _numpy_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        # 调用对象自身的 numpy_mask_tokens 方法，传入批量化的输入和掩码标签，得到 inputs 和 labels，返回作为字典的 "input_ids" 和 "labels"
        inputs, labels = self.numpy_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # 如果当前的分词器不是BertTokenizer或BertTokenizerFast，则发出警告
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        # 初始化候选索引列表
        cand_indexes = []
        # 遍历输入的token列表
        for i, token in enumerate(input_tokens):
            # 跳过特殊token，如"[CLS]"和"[SEP]"
            if token == "[CLS]" or token == "[SEP]":
                continue

            # 如果当前候选索引列表不为空且当前token是一个以"##"开头的部分token，则将当前token加入最后一个候选索引的列表中
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                # 否则，创建一个新的候选索引列表并加入当前token的索引
                cand_indexes.append([i])

        # 随机打乱候选索引列表
        random.shuffle(cand_indexes)
        # 计算应该预测的masked token数量，取最小值为max_predictions和输入token数量乘以mlm_probability的整数部分
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        # 初始化masked tokens列表
        masked_lms = []
        # 初始化已覆盖索引的集合
        covered_indexes = set()
        # 遍历候选索引列表
        for index_set in cand_indexes:
            # 如果已经预测的masked token数量达到了num_to_predict，则退出循环
            if len(masked_lms) >= num_to_predict:
                break
            # 如果当前候选索引集合加上已预测的masked token数量超过了num_to_predict，则跳过该候选集合
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            # 检查当前候选索引集合中是否有已覆盖的索引
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            # 如果有任何已覆盖的索引，则跳过该候选索引集合
            if is_any_index_covered:
                continue
            # 否则，将候选索引集合中的每个索引加入已覆盖索引集合，并将其加入masked tokens列表
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        # 如果已覆盖索引的数量不等于masked tokens列表的长度，则抛出异常
        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        # 根据已覆盖的索引集合生成mask标签列表，即标记哪些token是masked的
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        # 返回mask标签列表作为结果
        return mask_labels
    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        # 检查当前的分词器是否有掩码标记，这是进行掩码语言建模所必需的
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        # 复制输入以保留原始标签
        labels = inputs.clone()

        # 我们在每个序列中随机抽样几个标记，用于掩码语言建模训练（概率默认为0.15，适用于Bert/RoBERTa）
        probability_matrix = mask_labels

        # 获取特殊标记的掩码，用于排除掉特殊标记的影响
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 如果存在填充标记，将其添加到掩码中
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 确定要掩码的索引
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # 只计算掩码标记上的损失

        # 80%的时间，将掩码输入标记替换为tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的时间，将掩码输入标记替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余的时间（10%），保持掩码输入标记不变
        return inputs, labels
    def tf_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import tensorflow as tf  # 导入 TensorFlow 库

        input_shape = tf.shape(inputs)  # 获取输入张量的形状
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = tf.identity(inputs)  # 创建输入张量的副本作为标签

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = tf.cast(mask_labels, tf.bool)  # 将掩码标签转换为布尔类型张量

        # Exclude special tokens from masking
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels
        ]  # 获取特殊标记的掩码，排除已有的特殊标记
        masked_indices = masked_indices & ~tf.cast(special_tokens_mask, dtype=tf.bool)  # 更新掩码索引，排除特殊标记

        if self.tokenizer._pad_token is not None:
            padding_mask = inputs == self.tokenizer.pad_token_id  # 获取填充标记的掩码
            masked_indices = masked_indices & ~padding_mask  # 更新掩码索引，排除填充标记

        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)  # 根据掩码索引，将未掩码的位置在标签中替换为-100，仅计算掩码位置的损失

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices  # 使用伯努利采样确定掩码位置，80%的时间用[MASK]标记替换掩码输入
        inputs = tf.where(indices_replaced, self.tokenizer.mask_token_id, inputs)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = self.tf_bernoulli(input_shape, 0.5) & masked_indices & ~indices_replaced  # 使用伯努利采样确定掩码位置，10%的时间用随机词替换掩码输入
        random_words = tf.random.uniform(input_shape, maxval=len(self.tokenizer), dtype=tf.int64)  # 生成随机词
        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels  # 返回处理后的输入和标签
    def numpy_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        # Convert mask_labels to boolean array indicating which tokens to mask
        masked_indices = mask_labels.astype(bool)

        # Mask special tokens so they are not selected for masking
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        masked_indices[np.array(special_tokens_mask, dtype=bool)] = 0
        
        # If there is a padding token, mask it so it's not selected for masking
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0

        # Set labels of unmasked tokens to -100 to compute loss only on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random words
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(low=0, high=len(self.tokenizer), size=labels.shape, dtype=np.int64)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time), keep the masked input tokens unchanged
        return inputs, labels
@Dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """

    def __init__(self, *args, **kwargs):
        # 发出警告信息，提示该类即将被弃用，并建议使用DataCollatorForLanguageModeling代替
        warnings.warn(
            "DataCollatorForSOP is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling instead.",
            FutureWarning,
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        # 从每个示例中提取input_ids列表
        input_ids = [example["input_ids"] for example in examples]
        # 调用内部方法进行批量处理和填充
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        # 对input_ids进行遮蔽处理，生成labels和attention_mask
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)

        # 从每个示例中提取token_type_ids列表
        token_type_ids = [example["token_type_ids"] for example in examples]
        # 使用pad_sequence函数对token_type_ids进行填充，保证每个批次的长度一致
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # 从每个示例中提取sentence_order_label列表，并转换成tensor
        sop_label_list = [example["sentence_order_label"] for example in examples]
        sentence_order_label = torch.stack(sop_label_list)

        # 返回包含处理后数据的字典
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_order_label": sentence_order_label,
        }
    def mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
        import torch  # 导入PyTorch库，用于张量操作

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()  # 复制输入作为标签

        # 构建一个概率矩阵，决定哪些位置进行掩码处理，默认使用的概率为self.mlm_probability（通常为0.15）
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 获取输入序列中的特殊标记（如起始标记、结束标记等），并在概率矩阵中将这些位置的概率设为0
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 如果存在填充标记，则在概率矩阵中将填充标记位置的概率设为0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 使用伯努利分布生成一个掩码的布尔张量
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 根据模型的需求，调整注意力掩码的值（有些模型中，0表示被掩码）
        attention_mask = (~masked_indices).float()

        # 如果存在填充标记，则在注意力掩码中将填充标记位置的值设为1.0
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)

        # 将非掩码的位置的标签值设为-100，用于计算交叉熵损失时忽略这些位置
        labels[~masked_indices] = -100

        # 80%的情况下，将掩码的输入标记替换为特定的掩码标记（如[MASK]）
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的情况下，将掩码的输入标记替换为随机的单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余10%的情况下，保持掩码的输入标记不变

        # 返回处理后的输入、标签和注意力掩码
        return inputs, labels, attention_mask
# 使用 dataclass 装饰器定义一个数据类 DataCollatorForPermutationLanguageModeling，
# 用于处理排列语言建模的数据。
@dataclass
class DataCollatorForPermutationLanguageModeling(DataCollatorMixin):
    """
    Data collator used for permutation language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """

    # 初始化函数参数：tokenizer 表示预训练的分词器，plm_probability 表示置换语言建模的概率，默认为 1/6，
    # max_span_length 表示最大掩码标记序列的长度，默认为 5，
    # return_tensors 表示返回的张量类型，默认为 "pt"（PyTorch 张量）。
    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens
    return_tensors: str = "pt"

    # 定义 torch_call 方法，接收一个例子列表 examples，
    # 如果 examples 中的第一个元素是字典类型，则提取它们的 "input_ids" 字段作为例子列表的新内容。
    # 然后使用 _torch_collate_batch 函数对 examples 进行批量处理，结合 tokenizer 进行处理。
    # 最后调用 torch_mask_tokens 方法生成输入、掩码、目标映射和标签，并以字典形式返回结果。
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        batch = _torch_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.torch_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    # 定义 tf_call 方法，功能与 torch_call 方法类似，不同之处在于使用 _tf_collate_batch 函数处理 examples。
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        batch = _tf_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.tf_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    # 定义 numpy_call 方法，功能与前两者相似，使用 _numpy_collate_batch 处理 examples，
    # 并调用 numpy_mask_tokens 方法生成相应的输入、掩码、目标映射和标签，以字典形式返回结果。
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        batch = _numpy_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.numpy_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}
```