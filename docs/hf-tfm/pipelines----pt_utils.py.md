# `.\pipelines\pt_utils.py`

```
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.generic import ModelOutput


class PipelineDataset(Dataset):
    def __init__(self, dataset, process, params):
        self.dataset = dataset  # 存储数据集对象
        self.process = process  # 存储数据处理函数
        self.params = params    # 存储参数

    def __len__(self):
        return len(self.dataset)  # 返回数据集的长度

    def __getitem__(self, i):
        item = self.dataset[i]                      # 获取索引为i的数据集元素
        processed = self.process(item, **self.params)  # 使用给定的处理函数和参数处理数据
        return processed  # 返回处理后的数据


class PipelineIterator(IterableDataset):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        大致相当于

        ```
        for item in loader:
            yield infer(item, **params)
        ```

        参数:
            loader (`torch.utils.data.DataLoader` 或任何迭代器):
                将应用 `infer` 函数的迭代器。
            infer (任何函数):
                要应用于 `loader` 每个元素的函数。
            params (`dict`):
                传递给 `infer` 函数的参数。
            loader_batch_size (`int`, *可选*):
                如果指定，则假设 `loader` 中的项作为批次进行处理，并在此处加载批次，
                使其大致行为为

        ```
        for items in loader:
            for i in loader_batch_size:
                item = items[i]
                yield infer(item, **params)
        ```"""
        self.loader = loader                  # 存储数据加载器对象
        self.infer = infer                    # 存储推断函数
        self.params = params                  # 存储参数
        if loader_batch_size == 1:
            # 省略一些时间通过全部停用
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size  # 存储加载器批次大小

        # 内部记录
        self._loader_batch_index = None   # 加载器批次索引
        self._loader_batch_data = None    # 加载器批次数据

    def __len__(self):
        return len(self.loader)  # 返回加载器中的元素数量

    def __iter__(self):
        self.iterator = iter(self.loader)  # 创建加载器的迭代器
        return self
    def loader_batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        # 如果 `_loader_batch_data` 是 torch.Tensor 类型
        if isinstance(self._loader_batch_data, torch.Tensor):
            # 批处理数据是简单的张量，直接获取切片
            result = self._loader_batch_data[self._loader_batch_index].unsqueeze(0)
        else:
            # 批处理数据假定为 BaseModelOutput（或字典）
            loader_batched = {}
            # 遍历 `_loader_batch_data` 的项目
            for k, element in self._loader_batch_data.items():
                # 如果 element 是 ModelOutput 类型
                if isinstance(element, ModelOutput):
                    # 首先将 ModelOutput 转换为元组
                    element = element.to_tuple()
                    # 如果元组中第一个元素是 torch.Tensor
                    if isinstance(element[0], torch.Tensor):
                        # 将每个元素的正确批处理数据提取出来，并在第一维上增加维度为 1
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        # 将每个元素的正确批处理数据提取出来，并在第一维上增加维度为 1
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    continue
                # 如果 k 是 {"hidden_states", "past_key_values", "attentions"} 之一，并且 element 是元组
                if k in {"hidden_states", "past_key_values", "attentions"} and isinstance(element, tuple):
                    # 这些通常存储为张量列表，因此需要特定的解批处理操作
                    if isinstance(element[0], torch.Tensor):
                        # 将每个元素的正确批处理数据提取出来，并在第一维上增加维度为 1
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        # 将每个元素的正确批处理数据提取出来，并在第一维上增加维度为 1
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    continue
                # 如果 element 是 None，通常表示可选数据
                if element is None:
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    # 取出正确的批处理数据，但使其看起来像 batch_size=1
                    # 以便与 transformers 中的其他方法兼容
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    # 取出正确的批处理数据，但使其看起来像 batch_size=1
                    # 以便与 transformers 中的其他方法兼容
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    # 这通常是一个列表，因此无需 `unsqueeze` 操作
                    loader_batched[k] = element[self._loader_batch_index]
            # 通过使用原始类重新创建元素，使其看起来像 batch_size=1
            result = self._loader_batch_data.__class__(loader_batched)
        # 增加 `_loader_batch_index` 以便下次调用获取下一个批处理项
        self._loader_batch_index += 1
        return result
    # 定义迭代器的下一个元素方法
    def __next__(self):
        # 检查是否当前正在展开一个批次，并且批次内部索引小于批次大小
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            # 当前正在展开批次，直接返回批次内的当前项
            return self.loader_batch_item()

        # 批次内的项已经用完
        # 获取下一个迭代器中的项
        item = next(self.iterator)
        # 对获取的项进行推断处理，使用预定义的参数
        processed = self.infer(item, **self.params)
        # 现在有了一批推断出的数据
        if self.loader_batch_size is not None:
            # 尝试推断出批次的大小
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                # 可能是最后一个批次，因此无法展开太多元素
                self.loader_batch_size = observed_batch_size
            # 设置内部索引以展开批次数据
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            # 返回批次内的当前项
            return self.loader_batch_item()
        else:
            # 不需要展开批次
            return processed
class PipelineChunkIterator(PipelineIterator):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```

        Arguments:
            loader (`torch.utils.data.DataLoader` or any iterator):
                The iterator that will be used to apply `infer` on.
            infer (any function):
                The function to apply of each element of `loader`.
            params (`dict`):
                The parameters passed to `infer` along with every item
        """
        super().__init__(loader, infer, params)

    def __iter__(self):
        # Initialize the main iterator over the loader
        self.iterator = iter(self.loader)
        # Initialize subiterator to None
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            # If subiterator is None, start the preprocessing on the next item
            self.subiterator = self.infer(next(self.iterator), **self.params)
        
        try:
            # Try to retrieve the next processed item from subiterator
            processed = next(self.subiterator)
        except StopIteration:
            # If subiterator is exhausted, move to the next item in the main iterator
            # This is akin to flattening nested iterators into a single sequence
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        
        return processed


class PipelinePackIterator(PipelineIterator):
    """
    Roughly equivalent to

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```

    but it also handles cases where `item` are batched (meaning it's a dict of Tensor with first dimension > 1). In
    that case it does

    ```
    packed =  []
    for batch in loader:
        # item is batched
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```

    Arguments:
        loader (`torch.utils.data.DataLoader` or any iterator):
            The iterator that will be used to apply `infer` on.
        infer (any function):
            The function to apply to each element of `loader`.
        params (`dict`):
            The parameters passed to `infer` along with every item
        loader_batch_size (`int`, *optional*):
            If specified, the items of `loader` are supposed to come as batch, and are loader_batched here making
            it roughly behave as

    """
    for items in loader:
        for i in loader_batch_size:
            item = items[i]
            yield infer(item, **params)



    def __iter__(self):
        # 设置迭代器为加载器的迭代器
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        # 与 PipelineIterator 非常相似的解包机制
        # 但是，我们有一个额外的必需项，即 `is_last` 的存在
        # 这是因为一切都被 `PipelineChunkIterator` 扁平化了，我们需要在这里
        # 在原始的 `process` 边界处跟踪如何重新分组，以便 `process` 和 `postprocess` 看到相同的数据。

        # 此迭代器累积项目（可能在取消批处理时），直到遇到 `is_last`，然后将其传递给调用者。
        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                # 获取加载器批处理项
                item = self.loader_batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            # 处理下一个加载器项
            processed = self.infer(next(self.iterator), **self.params)
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:
                    # 可能是最后一个批次，因此我们不能展开太多元素。
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    # 获取加载器批处理项
                    item = self.loader_batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                # 单个加载器项处理
                item = processed
                is_last = item.pop("is_last")
                accumulator.append(item)
        return accumulator
# 定义一个名为 KeyDataset 的类，继承自 Dataset 类
class KeyDataset(Dataset):
    # 初始化方法，接收一个 dataset 和一个 key 参数
    def __init__(self, dataset: Dataset, key: str):
        # 将传入的 dataset 参数赋值给实例变量 self.dataset
        self.dataset = dataset
        # 将传入的 key 参数赋值给实例变量 self.key
        self.key = key

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 根据索引 i 返回对应元素的 key 字段值
    def __getitem__(self, i):
        return self.dataset[i][self.key]


# 定义一个名为 KeyPairDataset 的类，继承自 Dataset 类
class KeyPairDataset(Dataset):
    # 初始化方法，接收一个 dataset 和两个 key 参数
    def __init__(self, dataset: Dataset, key1: str, key2: str):
        # 将传入的 dataset 参数赋值给实例变量 self.dataset
        self.dataset = dataset
        # 将传入的 key1 参数赋值给实例变量 self.key1
        self.key1 = key1
        # 将传入的 key2 参数赋值给实例变量 self.key2
        self.key2 = key2

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 根据索引 i 返回一个字典，包含 text 键和 text_pair 键，分别对应 dataset 中的 key1 和 key2 字段值
    def __getitem__(self, i):
        return {"text": self.dataset[i][self.key1], "text_pair": self.dataset[i][self.key2]}
```