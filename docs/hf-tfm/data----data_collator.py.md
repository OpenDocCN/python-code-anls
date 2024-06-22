# `.\transformers\data\data_collator.py`

```py
# 导入必要的模块和库
import random  # 导入随机模块
import warnings  # 导入警告模块
from collections.abc import Mapping  # 导入 Mapping 抽象基类
from dataclasses import dataclass  # 导入 dataclass 装饰器
from random import randint  # 从随机模块导入 randint 函数
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库

from ..models.bert import BertTokenizer, BertTokenizerFast  # 导入 BERT 分词器
from ..tokenization_utils_base import PreTrainedTokenizerBase  # 导入基础分词器类
from ..utils import PaddingStrategy  # 导入填充策略类

InputDataClass = NewType("InputDataClass", Any)  # 定义 InputDataClass 类型别名

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])  # 定义 DataCollator 类型别名


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        # 如果未指定返回张量类型，则使用设定的默认类型
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 根据返回张量类型的不同，调用不同的处理函数
        if return_tensors == "tf":
            return self.tf_call(features)  # 调用 TensorFlow 处理函数
        elif return_tensors == "pt":
            return self.torch_call(features)  # 调用 PyTorch 处理函数
        elif return_tensors == "np":
            return self.numpy_call(features)  # 调用 NumPy 处理函数
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")  # 抛出错误，指示不支持的返回张量类型


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """
    # 避免使用快速分词器时触发关于 pad 函数子优化的警告
    # 针对特征提取器的错误处理
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # 保存警告状态，然后禁用警告
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)  # 执行填充操作
    finally:
        # 恢复警告状态
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded  # 返回填充后的结果


def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    """
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    # 定义了一组可能的键名，用于处理对象的属性
    # - `label`: 处理每个对象的单个值（int 或 float）
    # - `label_ids`: 处理每个对象的值列表
    # 不执行任何额外的预处理：输入对象的属性名称将用作模型的对应输入。查看 glue 和 ner 的示例以了解其有用性。

    # 在这个函数中，我们假设批处理中的所有 `features` 都具有相同的属性。
    # 因此，我们将查看第一个元素作为整个批次存在的属性的代理。

    # 如果 return_tensors 参数为 "pt"，则返回 PyTorch 默认的数据收集器处理结果
    if return_tensors == "pt":
        return torch_default_data_collator(features)
    # 如果 return_tensors 参数为 "tf"，则返回 TensorFlow 默认的数据收集器处理结果
    elif return_tensors == "tf":
        return tf_default_data_collator(features)
    # 如果 return_tensors 参数为 "np"，则返回 NumPy 默认的数据收集器处理结果
    elif return_tensors == "np":
        return numpy_default_data_collator(features)
from dataclasses import dataclass
from typing import List, Dict, Any, Mapping, Tuple
import torch
import numpy as np

# 使用dataclass装饰器定义DefaultDataCollator类，实现了DataCollatorMixin接口
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

    # 定义默认的返回张量类型为pytorch张量
    return_tensors: str = "pt"

    # 实现类的可调用方法
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # 如果未指定返回张量类型，则使用默认的类型
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 调用default_data_collator函数进行数据拼接和处理
        return default_data_collator(features, return_tensors)


# 定义针对torch张量的默认数据拼接函数
def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    # 若输入的特征列表中的第一个元素不是映射类型，则将其转换为字典
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    # 取出特征列表中的第一个元素
    first = features[0]
    # 创建一个空字典用于存储批次数据
    batch = {}

    # 特殊处理标签
    if "label" in first and first["label"] is not None:
        # 如果标签是torch张量，则提取其值
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        # 根据标签的类型设置数据类型
        dtype = torch.long if isinstance(label, int) else torch.float
        # 将所有特征中的标签拼接成张量并存储到批次字典中
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        # 如果标签IDs是torch张量，则将其拼接成张量
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            # 根据标签IDs的类型设置数据类型
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            # 将所有特征中的标签IDs拼接成张量并存储到批次字典中
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # 处理其他可能的键
    for k, v in first.items():
        # 排除特殊处理的键和字符串类型的键，并且值不为空
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                # 如果值是torch张量，则拼接成张量并存储到批次字典中
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                # 如果值是numpy数组，则转换为torch张量并存储到批次字典中
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # 将其他类型的值转换为torch张量并存储到批次字典中
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def tf_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import tensorflow as tf
    # 检查第一个特征是否为映射类型，如果不是，则将特征列表中的每个元素转换为字典
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    # 获取第一个特征
    first = features[0]
    # 初始化批处理字典
    batch = {}

    # 处理标签的特殊情况
    # 确保使用正确的类型创建张量
    if "label" in first and first["label"] is not None:
        label_col_name = "label"
    elif "label_ids" in first and first["label_ids"] is not None:
        label_col_name = "label_ids"
    elif "labels" in first and first["labels"] is not None:
        label_col_name = "labels"
    else:
        label_col_name = None
    if label_col_name is not None:
        # 根据标签数据类型确定张量的数据类型
        if isinstance(first[label_col_name], tf.Tensor):
            dtype = tf.int64 if first[label_col_name].dtype.is_integer else tf.float32
        elif isinstance(first[label_col_name], np.ndarray) or isinstance(first[label_col_name], np.generic):
            dtype = tf.int64 if np.issubdtype(first[label_col_name].dtype, np.integer) else tf.float32
        elif isinstance(first[label_col_name], (tuple, list)):
            dtype = tf.int64 if isinstance(first[label_col_name][0], int) else tf.float32
        else:
            dtype = tf.int64 if isinstance(first[label_col_name], int) else tf.float32
        # 将标签数据转换为张量并存储在批处理字典中
        batch["labels"] = tf.convert_to_tensor([f[label_col_name] for f in features], dtype=dtype)
    
    # 处理所有其他可能的键
    # 再次使用第一个元素来确定哪些键/值对对于此模型不为 None
    for k, v in first.items():
        if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
            if isinstance(v, (tf.Tensor, np.ndarray)):
                # 如果值是张量或数组，则将其堆叠为张量并存储在批处理字典中
                batch[k] = tf.stack([f[k] for f in features])
            else:
                # 否则将值转换为张量并存储在批处理字典中
                batch[k] = tf.convert_to_tensor([f[k] for f in features])

    # 返回批处理字典
    return batch
# 定义一个函数，用于将输入的特征列表转换为批量数据字典
def numpy_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    # 如果特征列表中的第一个元素不是映射类型，则将特征列表中的每个元素转换为字典
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    # 取第一个特征作为参考
    first = features[0]
    # 创建一个空的批量数据字典
    batch = {}

    # 处理标签的特殊情况
    # 确保正确创建张量
    if "label" in first and first["label"] is not None:
        # 如果标签是 ndarray 类型，则转换为 Python 中的标量
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        # 确定数据类型
        dtype = np.int64 if isinstance(label, int) else np.float32
        # 将标签转换为数组，指定数据类型
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        # 如果标签 ID 是 ndarray 类型，则堆叠成二维数组
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            # 否则，根据第一个标签 ID 元素的类型确定数据类型
            dtype = np.int64 if isinstance(first["label_ids"][0], int) else np.float32
            # 将标签 ID 转换为数组，指定数据类型
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    # 处理所有其他可能的键
    for k, v in first.items():
        # 如果键不是 "label" 或 "label_ids"，且值不为 None 且不是字符串类型
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            # 如果值是 ndarray 类型，则堆叠成二维数组
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                # 否则，将值转换为数组
                batch[k] = np.array([f[k] for f in features])

    # 返回批量数据字典
    return batch


# 定义一个数据收集器类，用于动态填充接收到的输入
@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            用于对数据进行编码的分词器。
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            选择一种策略来填充返回的序列（根据模型的填充位置和填充索引），其中：

            - `True` 或 `'longest'`（默认）：填充到批次中最长的序列（或者如果仅提供单个序列则不填充）。
            - `'max_length'`：填充到使用参数 `max_length` 指定的最大长度，或者如果未提供该参数，则填充到模型的最大可接受输入长度。
            - `False` 或 `'do_not_pad'`：不填充（即，可以输出长度不同的序列批次）。
        max_length (`int`, *optional*):
            返回列表的最大长度，以及可选的填充长度（见上文）。
        pad_to_multiple_of (`int`, *optional*):
            如果设置，将序列填充到提供的值的倍数。

            这对于在具有计算能力 >= 7.5（Volta）的 NVIDIA 硬件上启用张量核心尤其有用。
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            要返回的张量类型。可接受的值为 "np"、"pt" 和 "tf"。
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 使用自定义的 pad_without_fast_tokenizer_warning 函数填充序列
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # 如果批次中有 "label" 键，则将其改为 "labels"
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        # 如果批次中有 "label_ids" 键，则将其改为 "labels"
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        # 返回填充后的批次
        return batch
# 定义一个数据收集器类，用于在接收到数据时动态填充输入和标签
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

    # 用于编码数据的分词器
    tokenizer: PreTrainedTokenizerBase
    # 填充策略，可以是布尔值、字符串或填充策略对象，默认为True
    padding: Union[bool, str, PaddingStrategy] = True
    # 返回列表的最大长度和可选的填充长度，默认为None
    max_length: Optional[int] = None
    # 将序列填充到提供的值的倍数，特别适用于启用 NVIDIA 硬件上的 Tensor Cores
    pad_to_multiple_of: Optional[int] = None
    # 填充标签时使用的 ID，默认为-100，PyTorch 损失函数会自动忽略-100
    label_pad_token_id: int = -100
    # 返回的张量类型，默认为"pt"，可选值为"np"、"pt"和"tf"
    return_tensors: str = "pt"
    # 定义一个函数用于处理 PyTorch 引擎的数据
    def torch_call(self, features):
        # 导入 PyTorch 库
        import torch

        # 如果 features 中包含 'label' 键，则将 label_name 设置为 'label'，否则设置为 'labels'
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果 features 中的第一个元素包含 label_name，则提取所有 features 中的 labels，否则将 labels 设置为 None
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # 从 features 中移除 label_name 键的所有内容，得到没有 labels 的 features
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # 使用填充函数对特征进行填充，返回 PyTorch 张量
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 如果 labels 为 None，则返回批处理张量
        if labels is None:
            return batch

        # 获取序列长度
        sequence_length = batch["input_ids"].shape[1]
        # 获取填充方向
        padding_side = self.tokenizer.padding_side

        # 将 PyTorch 张量或可迭代对象转换为列表
        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        # 如果填充方向为右侧
        if padding_side == "right":
            # 对 labels 进行填充，使其与序列长度相同
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            # 对 labels 进行填充，使其与序列长度相同
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        # 将 labels 转换为 PyTorch 张量
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        # 返回填充后的批处理张量
        return batch

    # 定义一个函数用于处理 TensorFlow 引擎的数据
    def tf_call(self, features):
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 如果 features 中包含 'label' 键，则将 label_name 设置为 'label'，否则设置为 'labels'
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果 features 中的第一个元素包含 label_name，则提取所有 features 中的 labels，否则将 labels 设置为 None
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # 使用填充函数对特征进行填充，返回 TensorFlow 张量
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # 如果 labels 为 None，则直接返回 TensorFlow 张量，否则不转换为张量
            return_tensors="tf" if labels is None else None,
        )

        # 如果 labels 为 None，则返回批处理张量
        if labels is None:
            return batch

        # 获取序列长度
        sequence_length = tf.convert_to_tensor(batch["input_ids"]).shape[1]
        # 获取填充方向
        padding_side = self.tokenizer.padding_side
        # 如果填充方向为右侧
        if padding_side == "right":
            # 对 labels 进行填充，使其与序列长度相同
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            # 对 labels 进行填充，使其与序列长度相同
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        # 将 batch 中的所有值转换为 TensorFlow 张量
        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        # 返回填充后的批处理张量
        return batch
    # 定义一个方法，用于处理输入特征并返回处理后的批数据
    def numpy_call(self, features):
        # 确定标签名称是"label"还是"labels"
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 如果特征中包含标签，则将所有标签提取出来
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # 对特征进行填充处理，根据参数设置进行填充
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # 如果没有标签，则返回numpy数组
            return_tensors="np" if labels is None else None,
        )

        # 如果没有标签，则直接返回处理后的批数据
        if labels is None:
            return batch

        # 获取输入序列的长度
        sequence_length = np.array(batch["input_ids"]).shape[1]
        # 获取填充的位置（左边或右边）
        padding_side = self.tokenizer.padding_side
        # 根据填充位置对标签进行填充处理
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        # 将所有数据转换为numpy数组，并指定数据类型为int64
        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        # 返回处理后的批数据
        return batch
# 将示例数据集合成一个批次，根据需要使用 tokenizer 进行填充
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # 如果示例是列表、元组或者 numpy 数组，则转换为 torch 张量
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # 获取第一个示例的长度
    length_of_first = examples[0].size(0)

    # 检查是否需要填充

    # 检查所有张量是否具有相同的长度
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    # 如果所有张量长度相同且不需要填充，则直接堆叠张量
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # 如果需要填充，检查是否有填充标记
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # 创建完整的张量并用数据填充
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
    # 如果示例是列表或元组，则转换为 TensorFlow 张量
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.convert_to_tensor(e, dtype=tf.int64) for e in examples]

    # 检查是否需要填充
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)

    # 如果需要填充，检查���否有填充标记
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # 创建完整的张量并用数据填充
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    # result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    result = []
    rank = tf.rank(examples[0])
    paddings = np.zeros((rank, 2), dtype=np.int32)
    # 遍历给定的示例列表
    for example in examples:
        # 如果分词器的填充位置在右侧
        if tokenizer.padding_side == "right":
            # 计算右侧填充的数量
            paddings[0, 1] = max_length - len(example)
        else:
            # 计算左侧填充的数量
            paddings[0, 0] = max_length - len(example)
        # 对示例进行填充，并将结果添加到结果列表中
        result.append(tf.pad(example, paddings, constant_values=tokenizer.pad_token_id))
    # 使用 TensorFlow 在新维度上叠加填充后的示例，沿着轴0叠加
    return tf.stack(result, axis=0)
# 将示例列表`examples`整理成一个批次，如果需要，使用`tokenizer`中的信息进行填充
def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    # 如果示例是列表或元组，则将其转换为NumPy数组
    if isinstance(examples[0], (list, tuple)):
        examples = [np.array(e, dtype=np.int64) for e in examples]

    # 检查是否需要填充
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return np.stack(examples, axis=0)

    # 如果需要填充，检查是否有`pad_token`
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # 创建完整的张量并填充数据
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = np.full(shape=(len(examples), max_length), fill_value=tokenizer.pad_token_id, dtype=examples[0].dtype)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


# 将输入`x`转换为列表
def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # 检查是否为TF张量，无需导入TF库
        x = x.numpy()
    return x.tolist()


# 用于序列到序列模型的数据整理器
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            用于对数据进行编码的分词器。
        model ([`PreTrainedModel`], *optional*):
            正在训练的模型。如果设置并且具有 *prepare_decoder_input_ids_from_labels*，则使用它来准备 *decoder_input_ids*。

            在使用 *label_smoothing* 时很有用，可以避免两次计算损失。
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            选择一种策略来对返回的序列进行填充（根据模型的填充方向和填充索引），可选值包括：

            - `True` 或 `'longest'`（默认）：填充为批次中最长的序列（如果仅提供单个序列，则不进行填充）。
            - `'max_length'`：填充到指定的最大长度（通过参数 `max_length` 提供）或者填充到模型的最大可接受输入长度（如果未提供该参数）。
            - `False` 或 `'do_not_pad'`：不进行填充（即可以输出具有不同长度序列的批次）。
        max_length (`int`, *optional*):
            返回列表的最大长度，可选地作为填充长度（见上文）。
        pad_to_multiple_of (`int`, *optional*):
            如果设置，将序列填充到提供的值的倍数。

            这对于启用具有计算能力 >= 7.5（Volta）的 NVIDIA 硬件上的 Tensor Cores 特别有用。
        label_pad_token_id (`int`, *optional*, defaults to -100):
            用于填充标签时使用的标识符（-100 将被 PyTorch 损失函数自动忽略）。
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            要返回的张量类型。允许的值为 "np"、"pt" 和 "tf"。
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    # 定义 __call__ 方法，用于对输入的 features 进行处理，并返回处理后的 features
    def __call__(self, features, return_tensors=None):
        # 如果 return_tensors 为 None，则使用默认值 self.return_tensors
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 如果 features 中包含 "labels" 键，则提取出所有 labels，并存储在列表中；否则 labels 为 None
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # 在调用 `tokenizer.pad` 方法之前，需要对 labels 进行填充，因为该方法不会进行填充，而且需要输入相同长度的 labels 才能返回张量
        if labels is not None:
            # 计算 labels 中最长的标签长度
            max_label_length = max(len(l) for l in labels)
            # 如果指定了 pad_to_multiple_of，则将 max_label_length 向上取整至其倍数
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            # 获取填充的位置（左侧或右侧）
            padding_side = self.tokenizer.padding_side
            # 遍历 features 中的每个 feature
            for feature in features:
                # 计算需要填充的标签部分，使其与最长标签长度相等
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                # 如果 feature["labels"] 是列表，则根据填充位置在左右两侧进行填充
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                # 如果填充在右侧，则将填充的部分与原标签连接起来，并转换为 int64 类型
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                # 如果填充在左侧，则将填充的部分与原标签连接起来，并转换为 int64 类型
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # 使用 pad_without_fast_tokenizer_warning 函数对 features 进行填充处理
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # 准备 decoder_input_ids
        # 如果 labels 不为空，且 model 存在，并且 model 具有 prepare_decoder_input_ids_from_labels 方法
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            # 使用 model 的 prepare_decoder_input_ids_from_labels 方法根据 labels 准备 decoder_input_ids
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            # 将准备好的 decoder_input_ids 存储在 features 中
            features["decoder_input_ids"] = decoder_input_ids

        # 返回处理后的 features
        return features
# 定义一个数据收集器，用于语言建模。如果输入数据的长度不相同，将动态地将其填充到批次的最大长度。

# 引入必要的库
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    # 初始化函数，用于检查是否可以执行 MLM，并进行一些设置
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        # 如果启用 MLM 但 Tokenizer 没有掩码标记，则引发错误
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        # 如果启用 TensorFlow 实验性编译，则导入 TensorFlow 库并编译 tf_mask_tokens 方法
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    @staticmethod
    # 生成一个服从伯努利分布的张量，用于随机掩码
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    # TensorFlow 下的掩码处理函数，用于对输入进行掩码处理
    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 将 mask_token_id 转换为与 inputs 相同的数据类型
        mask_token_id = tf.cast(mask_token_id, inputs.dtype)

        # 获取输入的形状
        input_shape = tf.shape(inputs)
        # 1 表示特殊标记，0 表示普通标记在特殊标记掩码中
        # 我们在每个序列中随机选择一些标记进行 MLM 训练（概率为 self.mlm_probability）
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        # 将未掩码的索引替换为 -100，因为我们只在掩码标记上计算损失
        labels = tf.where(masked_indices, inputs, -100)

        # 80% 的时间，我们将掩码的输入标记替换为 tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # 10% 的时间，我们将掩码的输入标记替换为随机单词
        indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=inputs.dtype)

        inputs = tf.where(indices_random, random_words, inputs)

        # 剩余的时间（10% 的时间），我们保持掩码的输入标记不变
        return inputs, labels
    # 定义 TensorFlow 推断函数，接受 examples 参数作为输入，返回预测结果字典
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 处理字典或列表，进行适当的填充和转换为张量
        if isinstance(examples[0], Mapping):
            # 使用 pad_without_fast_tokenizer_warning 函数进行填充，返回 TensorFlow 格式的批量数据
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 对输入进行拼接并进行填充，返回 input_ids 键的批量数据
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果特殊标记掩码已经预处理完毕，从字典中弹出该项
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # 如果启用了 MLM（Masked Language Modeling），则执行以下逻辑
        if self.mlm:
            if special_tokens_mask is None:
                # 获取已经存在特殊标记的数据的特殊标记掩码
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                # 将列表转换为 TensorFlow 张量，并将数据类型转换为布尔类型
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                # 将特殊标记掩码转换为 TensorFlow 张量，并将数据类型转换为布尔类型
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            # 对输入数据进行 MLM 掩码处理，返回处理后的输入和标签
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            # 如果未启用 MLM，则将输入数据直接作为标签
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                # 用 -100 替换标记为 pad_token_id 的位置
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                # 创建标签的副本，以防万一
                labels = tf.identity(labels)
            batch["labels"] = labels
        # 返回处理后的批量数据
        return batch
    # 定义 torch_call 方法，接收一个包含列表、任意类型或字典的列表作为参数，并返回字典类型的结果
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 如果 examples 中的第一个元素是字典，则进行特殊处理，包括适当的填充和转换为张量
        if isinstance(examples[0], Mapping):
            # 使用 pad_without_fast_tokenizer_warning 函数对 examples 进行填充，返回张量，pad_to_multiple_of 参数用于指定填充的倍数
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 否则，创建一个包含 "input_ids" 键的字典，值为调用 _torch_collate_batch 函数的结果
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果已经预处理了特殊的 token mask，则从字典中删除它
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # 如果启用了 MLM，则调用 torch_mask_tokens 函数进行处理
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # 否则，将 "input_ids" 的值复制给 labels
            labels = batch["input_ids"].clone()
            # 如果 tokenizer 的 pad_token_id 不为 None，则将 labels 中等于 pad_token_id 的值设置为 -100
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        # 返回处理后的 batch
        return batch

    # 定义 torch_mask_tokens 方法，接收 inputs 和 special_tokens_mask 两个参数，并返回一个元组
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        # 克隆 inputs，作为 labels
        labels = inputs.clone()
        # 创建一个与 labels 形状相同的全为 self.mlm_probability 的张量
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 如果 special_tokens_mask 为 None，则根据 labels 的值获取特殊 token mask
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 将 probability_matrix 中与 special_tokens_mask 对应位置的值设为 0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # 根据概率矩阵生成掩码
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 将 labels 中非掩码位置的值设为 -100，用于计算损失
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% 的概率将掩码位置的输入 tokens 替换为 tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% 的概率将掩码位置的输入 tokens 替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余 10% 的概率保持掩码位置的输入 tokens 不变
        return inputs, labels
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 处理字典或列表，进行适当的填充和转换为张量
        if isinstance(examples[0], Mapping):
            # 使用适当的填充和转换为张量处理字典
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            # 处理列表
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # 如果特殊标记掩码已经预处理，从字典中弹出
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            # 如果是MLM任务，调用numpy_mask_tokens函数处理input_ids和special_tokens_mask
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            # 复制input_ids作为labels
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                # 将labels中等于pad_token_id的值设为-100
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        # 返回处理后的batch
        return batch
    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # 复制输入作为标签
        labels = np.copy(inputs)
        
        # 创建一个与标签形状相同的概率矩阵，用于确定哪些标记要被掩盖
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        
        # 如果没有提供特殊标记掩码，则根据标签获取特殊标记掩码
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        # 将特殊标记位置的概率设为0
        probability_matrix[special_tokens_mask] = 0
        
        # 使用二项分布生成掩盖标记的索引
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # 仅在掩盖标记上计算损失

        # 80%的情况下，将掩盖的输入标记替换为tokenizer.mask_token_id ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10%的情况下，将掩盖的输入标记替换为随机单词
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # 剩余的情况下（10%的情况下），保持掩盖的输入标记不变
        return inputs, labels
@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>"""

    # 定义 torch_call 方法，接受 examples 参数，返回字典类型
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]) -> Dict[str, Any]:
        # 如果 examples 的第一个元素是 Mapping 类型
        if isinstance(examples[0], Mapping):
            # 提取每个元素中的 "input_ids" 字段
            input_ids = [e["input_ids"] for e in examples]
        else:
            # 否则，直接使用 examples 作为 input_ids，并将每个元素包装成字典
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        # 调用 _torch_collate_batch 方法，将 input_ids 转换为 batch_input
        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # 初始化 mask_labels 列表
        mask_labels = []
        # 遍历 examples
        for e in examples:
            # 初始化 ref_tokens 列表
            ref_tokens = []
            # 遍历 e["input_ids"] 中的每个 id
            for id in tolist(e["input_ids"]):
                # 将 id 转换为 token，并添加到 ref_tokens 中
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # 对于中文 tokens，需要额外的信息来标记子词，例如 [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                # 提取中文参考信息
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                # 遍历序列长度
                for i in range(len_seq):
                    # 如果 i 在 ref_pos 中
                    if i in ref_pos:
                        # 在对应的 token 前加上 "##"
                        ref_tokens[i] = "##" + ref_tokens[i]
            # 将整个词掩盖后的结果添加到 mask_labels 中
            mask_labels.append(self._whole_word_mask(ref_tokens))
        
        # 调用 _torch_collate_batch 方法，将 mask_labels 转换为 batch_mask
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        # 调用 torch_mask_tokens 方法，获取 inputs 和 labels
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        # 返回包含 inputs 和 labels 的字典
        return {"input_ids": inputs, "labels": labels}
    # 定义一个 TensorFlow 接口函数，接受一个包含输入数据的列表，返回一个字典
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 检查输入数据的类型，如果是 Mapping 类型，则提取其中的 "input_ids" 字段
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        # 调用 _tf_collate_batch 函数，将输入数据整理成批量输入
        batch_input = _tf_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # 初始化一个空列表用于存储 mask 标签
        mask_labels = []
        # 遍历每个示例
        for e in examples:
            ref_tokens = []
            # 遍历示例中的每个输入 id，并将其转换为对应的 token
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # 对于中文 token，需要额外的信息来标记子词，例如 [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # 将处理后的 token 序列传入 _whole_word_mask 函数，生成 mask 标签
            mask_labels.append(self._whole_word_mask(ref_tokens))
        # 调用 _tf_collate_batch 函数，将 mask 标签整理成批量输入
        batch_mask = _tf_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        # 调用 tf_mask_tokens 函数，对输入数据进行 mask 处理，返回处理后的输入和标签
        inputs, labels = self.tf_mask_tokens(tf.cast(batch_input, tf.int64), batch_mask)
        # 返回处理后的结果字典
        return {"input_ids": inputs, "labels": labels}

    # 定义一个 NumPy 接口函数，接受一个包含输入数据的列表，返回一个字典
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 检查输入数据的类型，如果是 Mapping 类型，则提取其中的 "input_ids" 字段
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        # 调用 _numpy_collate_batch 函数，将输入数据整理成批量输入
        batch_input = _numpy_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        # 初始化一个空列表用于存储 mask 标签
        mask_labels = []
        # 遍历每个示例
        for e in examples:
            ref_tokens = []
            # 遍历示例中的每个输入 id，并将其转换为对应的 token
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # 对于中文 token，需要额外的信息来标记子词，例如 [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # 将处理���的 token 序列传入 _whole_word_mask 函数，生成 mask 标签
            mask_labels.append(self._whole_word_mask(ref_tokens))
        # 调用 _numpy_collate_batch 函数，将 mask 标签整理成批量输入
        batch_mask = _numpy_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        # 调用 numpy_mask_tokens 函数，对输入数据进行 mask 处理，返回处理后的输入和标签
        inputs, labels = self.numpy_mask_tokens(batch_input, batch_mask)
        # 返回处理后的结果字典
        return {"input_ids": inputs, "labels": labels}
    # 使用整个词掩盖代理获取被掩盖的标记的0/1标签
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        # 如果当前的分词器不是BertTokenizer或BertTokenizerFast类型的，则发出警告
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        # 存储候选的词索引
        cand_indexes = []
        # 遍历输入的标记列表
        for i, token in enumerate(input_tokens):
            # 如果标记是"[CLS]"或"[SEP]"，则跳过
            if token == "[CLS]" or token == "[SEP]":
                continue

            # 如果候选索引列表不为空且当前标记以"##"开头，则将当前索引添加到上一个候选索引列表中
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            # 否则，将当前索引作为新的候选索引列表的第一个元素
            else:
                cand_indexes.append([i])

        # 随机打乱候选索引列表
        random.shuffle(cand_indexes)
        # 计算要预测的标记数量，取最小值为最大预测数量和输入标记数量乘以掩码概率的四舍五入整数
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        # 存储被掩盖的标记的索引
        masked_lms = []
        # 存储已覆盖的索引
        covered_indexes = set()
        # 遍历候选索引列表
        for index_set in cand_indexes:
            # 如果已经预测的标记数量大于等于要预测的标记数量，则跳出循环
            if len(masked_lms) >= num_to_predict:
                break
            # 如果添加一个整个词掩码会超过最大预测数量，则跳过此候选索引列表
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            # 判断当前候选索引集合中是否有任何一个索引已经被覆盖
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            # 如果已覆盖任何一个索引，则跳过此候选索引列表
            if is_any_index_covered:
                continue
            # 将候选索引集合中的索引添加到被掩盖的标记列表中，并标记为已覆盖
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        # 如果已覆盖的索引数量不等于被掩盖的标记数量，则抛出值错误
        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        # 构建标记掩码列表，如果索引在已覆盖的索引集合中，则为1，否则为0
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        # 返回标记掩码列表
        return mask_labels
    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        准备用于掩码语言建模的掩码标记输入/标签：80% MASK，10% 随机，10% 原始。设置 'mask_labels' 意味着我们使用整词掩码 (wwm)，我们根据其引用直接掩码 idxs。
        """
        import torch

        如果标记器没有掩码标记：
            抛出值错误异常
        labels = inputs.clone()
        # 我们在每个序列中对一些标记进行掩码-LM训练（概率为 args.mlm_probability，默认为 0.15 在 Bert/RoBERTa）

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        如果标记器的填充标记不是 None：
            创建填充掩码
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # 我们只计算掩码标记上的损失

        # 80% 的时间，我们用 tokenizer.mask_token ([MASK]) 替换掩码输入标记
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% 的时间，我们用随机单词替换掩码输入标记
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩下的时间（10% 的时间），我们保持掩码输入标记不变
        return inputs, labels
    # 定义方法，用于为掩码语言建模准备掩码的标记输入/标签：80% MASK，10% 随机，10% 原始。设置'mask_labels'意味着我们使用整词掩码(wwm)，我们直接根据其参考掩码索引。
    def tf_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        # 导入 TensorFlow 库
        import tensorflow as tf

        # 获取输入的形状
        input_shape = tf.shape(inputs)
        # 如果分词器没有掩码标记，则引发错误
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        # 将标签设置为输入的副本
        labels = tf.identity(inputs)
        # 从每个序列中随机抽样一些标记，用于掩码语言建模训练（默认情况下，Bert/RoBERTa 中的 args.mlm_probability 为 0.15）

        # 将掩码标签转换为布尔型
        masked_indices = tf.cast(mask_labels, tf.bool)

        # 获取特殊标记掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels
        ]
        # 掩码索引为掩码索引且不是特殊标记掩码
        masked_indices = masked_indices & ~tf.cast(special_tokens_mask, dtype=tf.bool)
        # 如果分词器的填充标记不是 None
        if self.tokenizer._pad_token is not None:
            # 获取填充掩码
            padding_mask = inputs == self.tokenizer.pad_token_id
            # 掩码索引为掩码索引且不是填充掩码
            masked_indices = masked_indices & ~padding_mask

        # 将未掩码的索引在标签中替换为 -100，因为我们只在掩码标记上计算损失
        labels = tf.where(masked_indices, inputs, -100)

        # 80% 的时间，我们将掩码的输入标记替换为分词器的掩码标记 ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

        inputs = tf.where(indices_replaced, self.tokenizer.mask_token_id, inputs)

        # 10% 的时间，我们将掩码的输入标记替换为随机词
        indices_random = self.tf_bernoulli(input_shape, 0.5) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, maxval=len(self.tokenizer), dtype=tf.int64)
        inputs = tf.where(indices_random, random_words, inputs)

        # 剩余的时间（10% 的时间），我们保持掩码的输入标记不变
        return inputs, labels
``` 
    def numpy_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        # 检查当前分词器是否有掩码标记，用于掩码语言建模
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        # 复制输入以获取标签
        labels = np.copy(inputs)
        # 确定要掩码的索引
        masked_indices = mask_labels.astype(bool)

        # 获取特殊标记的掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 掩码特殊标记的索引
        masked_indices[np.array(special_tokens_mask, dtype=bool)] = 0
        # 如果存在填充标记，将其掩码
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0

        # 将非掩码标记的标签设置为-100，用于计算损失
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80%的情况下，将掩码的输入标记替换为tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的情况下，将掩码的输入标记替换为随机单词
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(low=0, high=len(self.tokenizer), size=labels.shape, dtype=np.int64)
        inputs[indices_random] = random_words[indices_random]

        # 剩余情况（10%的概率）下，保持掩码的输入标记不变
        return inputs, labels
@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """

    def __init__(self, *args, **kwargs):
        # 发出警告，表明 DataCollatorForSOP 类已被弃用，建议使用 DataCollatorForLanguageModeling 替代
        warnings.warn(
            "DataCollatorForSOP is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling instead.",
            FutureWarning,
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        # 提取 examples 中每个示例的 input_ids，并组成列表
        input_ids = [example["input_ids"] for example in examples]
        # 调用 _torch_collate_batch 函数对 input_ids 列表进行处理，保证张量的拼接
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        # 调用 mask_tokens 方法对 input_ids 进行处理，生成 mask 用于 masked language modeling
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)

        # 提取 examples 中每个示例的 token_type_ids，并组成列表
        token_type_ids = [example["token_type_ids"] for example in examples]
        # 对 token_type_ids 列表进行填充，使其长度相同，以适应模型的输入
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # 提取 examples 中每个示例的 sentence_order_label，并组成列表
        sop_label_list = [example["sentence_order_label"] for example in examples]
        # 使用 torch.stack 将 sop_label_list 列表中的张量堆叠成一个张量
        sentence_order_label = torch.stack(sop_label_list)

        # 返回字典，包含处理后的数据
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
        import torch

        # 检查当前的分词器是否有掩码标记，用于掩码语言建模
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )

        # 复制输入作为标签
        labels = inputs.clone()
        
        # 为每个序列中的一些标记准备掩码语言建模的输入/标签/注意力掩码：80% MASK，10% 随机，10% 原始
        # 创建一个概率矩阵，用于确定哪些标记要被掩码
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 获取特殊标记的掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 如果有填充标记，则将填充标记的概率设为0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # 通过伯努利分布生成掩码的索引
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 生成注意力掩码，将掩码的标记设为0，反转值
        attention_mask = (~masked_indices).float()
        
        # 如果有填充标记，则将填充标记的注意力掩码设为1
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)
        
        # 将未掩码的标记设为-100，用于计算损失
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80%的情况下，用tokenizer.mask_token ([MASK])替换掩码的输入标记
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的情况下，用随机单词替换掩码的输入标记
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余的10%的情况下，保持掩码的输入标记不变
        return inputs, labels, attention_mask
from dataclasses import dataclass  # 导入 dataclass 装饰器用于定义数据类

@dataclass
class DataCollatorForPermutationLanguageModeling(DataCollatorMixin):
    """
    Data collator used for permutation language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """

    tokenizer: PreTrainedTokenizerBase  # 用于分词的预训练分词器
    plm_probability: float = 1 / 6  # PLM 模型生成掩码的概率，默认为 1/6
    max_span_length: int = 5  # 最大生成掩码的 token 数
    return_tensors: str = "pt"  # 返回的张量类型，默认为 PyTorch 张量

    # 定义用于处理 Torch 张量的方法
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 如果输入的示例为字典类型，则提取其中的 input_ids
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        # 对示例进行 Torch 张量化处理
        batch = _torch_collate_batch(examples, self.tokenizer)
        # 生成掩码，并返回处理结果字典
        inputs, perm_mask, target_mapping, labels = self.torch_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    # 定义用于处理 TensorFlow 张量的方法
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 如果输入的示例为字典类型，则提取其中的 input_ids
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        # 对示例进行 TensorFlow 张量化处理
        batch = _tf_collate_batch(examples, self.tokenizer)
        # 生成掩码，并返回处理结果字典
        inputs, perm_mask, target_mapping, labels = self.tf_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    # 定义用于处理 NumPy 数组的方法
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # 如果输入的示例为字典类型，则提取其中的 input_ids
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        # 对示例进行 NumPy 数组化处理
        batch = _numpy_collate_batch(examples, self.tokenizer)
        # 生成掩码，并返回处理结果字典
        inputs, perm_mask, target_mapping, labels = self.numpy_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}
```