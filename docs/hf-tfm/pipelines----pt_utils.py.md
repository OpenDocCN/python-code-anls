# `.\transformers\pipelines\pt_utils.py`

```py
# 导入所需的库
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
# 导入自定义的ModelOutput类
from ..utils.generic import ModelOutput

# 定义PipelineDataset类
class PipelineDataset(Dataset):
    # 初始化方法，接受dataset、process和params作为输入
    def __init__(self, dataset, process, params):
        # 将dataset赋值给实例变量self.dataset
        self.dataset = dataset
        # 将process赋值给实例变量self.process
        self.process = process
        # 将params赋值给实例变量self.params
        self.params = params

    # 返回dataset的长度
    def __len__(self):
        return len(self.dataset)

    # 返回经过process处理后的数据
    def __getitem__(self, i):
        # 获取第i个元素
        item = self.dataset[i]
        # 使用process处理item并传入params，将结果赋值给processed
        processed = self.process(item, **self.params)
        return processed

# 定义PipelineIterator类
class PipelineIterator(IterableDataset):
    # 初始化方法，接受loader、infer、params和loader_batch_size作为输入
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        粗略相当于

        ```
        for item in loader:
            yield infer(item, **params)
        ```py

        参数:
            loader (`torch.utils.data.DataLoader` or any iterator):
                将应用于`infer`的迭代器
            infer (any function):
                应用于`loader`每个元素的函数
            params (`dict`):
                传递给`infer`的参数
            loader_batch_size (`int`, *optional*):
                如果指定，假设`loader`的项将作为批处理形式传入，并在此处进行批处理处理，使其大致行为如下

        """
        # 将loader赋值给实例变量self.loader
        self.loader = loader
        # 将infer赋值给实例变量self.infer
        self.infer = infer
        # 将params赋值给实例变量self.params
        self.params = params
        # 如果loader_batch_size等于1，则将其设置为None
        if loader_batch_size == 1:
            loader_batch_size = None
        # 将loader_batch_size赋值给实例变量self.loader_batch_size
        self.loader_batch_size = loader_batch_size

        # 内部记录
        self._loader_batch_index = None
        self._loader_batch_data = None

    # 返回loader的长度
    def __len__(self):
        return len(self.loader)

    # 返回迭代器对象
    def __iter__(self):
        # 生成迭代器对象
        self.iterator = iter(self.loader)
        return self
    def loader_batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        # 如果 loader_batch_data 是 torch.Tensor 类型
        if isinstance(self._loader_batch_data, torch.Tensor):
            # 批处理数据是简单的张量，直接获取片段
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            # 批处理数据被假定为BaseModelOutput（或字典）
            loader_batched = {}
            # 遍历 loader_batch_data 的项目
            for k, element in self._loader_batch_data.items():
                if isinstance(element, ModelOutput):
                    # 首先将 ModelOutput 转换为元组
                    element = element.to_tuple()
                    # 如果元素的第一个值为 torch.Tensor 类型
                    if isinstance(element[0], torch.Tensor):
                        # 将元素中每个元素取出指定索引并在 batch 维度上添加，形成元素为1的元组
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    # 如果元素的第一个值为 np.ndarray 类型
                    elif isinstance(element[0], np.ndarray):
                        # 将元素中每个元素取出指定索引并在 batch 维度上添加，形成元素为1的元组
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    # 继续下一次循环
                    continue
                if k in {"hidden_states", "past_key_values", "attentions"} and isinstance(element, tuple):
                    # 这些存储为张量列表，需要特定的解批处理。
                    if isinstance(element[0], torch.Tensor):
                        # 将元素中每个元素取出指定索引并在 batch 维度上添加，形成元素为1的元组
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        # 将元素中每个元素取出指定索引并在 batch 维度上添加，形成元素为1的元组
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    # 继续下一次循环
                    continue
                if element is None:
                    # 对于可选数据，可能为 None
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    # 取出正确的批处理数据，但使其看起来像 batch_size=1
                    # 与transformers内部���他方法兼容
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    # 取出正确的批处理数据，但使其看起来像 batch_size=1
                    # 与transformers内部其他方法兼容
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    # 这通常是一个列表，所以不需要 `unsqueeze`.
                    loader_batched[k] = element[self._loader_batch_index]
            # 通过重用原始类，重新创建元素，使其看起来像 batch_size=1
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result
    # 定义迭代器的下一个元素的方法
    def __next__(self):
        # 检查当前是否在解开批次内的元素，如果是，则返回当前批次内的元素
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            # We are currently unrolling a batch so we just need to return
            # the current item within a batch
            return self.loader_batch_item()

        # 当前批次内的元素已经遍历完毕
        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        # 现在我们有了一批“推断出的东西”
        
        # 如果定义了批次的大小
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
                # 可能是最后一批，所以我们不能解开太多元素
                self.loader_batch_size = observed_batch_size
            # 设置内部索引以解开批次数据
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            # 我们不进行批次解开
            return processed
# 创建一个 PipelineChunkIterator 类，它是 PipelineIterator 的子类
class PipelineChunkIterator(PipelineIterator):
    # 初始化方法
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        类似于以下代码

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```py

        参数:
            loader (`torch.utils.data.DataLoader` or any iterator):
                用于应用 `infer` 的迭代器。
            infer (any function):
                应用于 `loader` 中每个元素的函数。
            params (`dict`):
                与每个条目一起传递给 `infer` 的参数
            loader_batch_size (`int`, *可选*):
                如果指定，`loader` 的项目应为批处理，并在此处作为 loader_batched，从而使其大致行为如下


        """

        # 调用父类的初始化方法
        super().__init__(loader, infer, params)

    # 迭代器方法
    def __iter__(self):
        # 将 self.loader 转换为迭代器
        self.iterator = iter(self.loader)
        # 子迭代器设置为 None
        self.subiterator = None
        return self

    # 迭代器方法
    def __next__(self):
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            # 尝试返回下一个条目
            processed = next(self.subiterator)
        except StopIteration:
            # 当前处理迭代器结束后，我们可以开始查看下一个条目
            # ChunkIterator 将一直进行处理，直到 iterator 的所有元素都创建了子迭代器并已进行迭代。
            #
            # 另一种看法是，我们基本上在使用生成器将列表的列表展开为单个列表
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


# 创建一个 PipelinePackIterator 类，它是 PipelineIterator 的子类
class PipelinePackIterator(PipelineIterator):
    """
    类似于以下代码

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```py

    但它还处理了 `item` 为批处理的情况（这意味着它是一个具有大于 1 的第一维张量的字典）。在这种情况下，它执行以下操作

    ```
    packed =  []
    for batch in loader:
        # item 是批处理的
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```py

    参数:
        loader (`torch.utils.data.DataLoader` or any iterator):
            用于应用 `infer` 的迭代器。
        infer (any function):
            应用于 `loader` 中每个元素的函数。
        params (`dict`):
            与每个条目一起传递给 `infer` 的参数
        loader_batch_size (`int`, *可选*):
            如果指定，`loader` 的项目应为批处理，并在此处作为 loader_batched，从而使其大致行为如下


    """
    # 遍历 loader，获取其中的 items
    for items in loader:
        # 遍历 loader_batch_size，处理每个 item
        for i in loader_batch_size:
            # 获取指定索引处的 item
            item = items[i]
            # 调用 infer 方法处理 item，并使用参数 params
            yield infer(item, **params)
    ```

    # 实现迭代器方法，将其作为迭代器使用
    def __iter__(self):
        # 初始化迭代器为 loader 对象的迭代器
        self.iterator = iter(self.loader)
        # 返回迭代器对象
        return self

    # 实现迭代器的下一个值方法
    def __next__(self):
        # 这里的逻辑与 PipelineIterator 非常相似，但是有一个额外的必需项，即 `is_last` 的存在
        # 这是因为 `PipelineChunkIterator` 把所有东西展开了，我们需要在原始 `process` 
        # 边界中跟踪如何重新分组，以便 `process` 和 `postprocess` 看到相同的数据
        # 这个迭代器累积项目（可能在取消批处理时），直到遇到 `is_last`，然后将其传递给调用者
        is_last = False
        accumulator = []
        # 如果 _loader_batch_index 不为空，并且小于 loader_batch_size
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            # 当 _loader_batch_index 小于 loader_batch_size 时，继续循环
            while self._loader_batch_index < self.loader_batch_size:
                # 获取 loader_batch_item 中的 item
                item = self.loader_batch_item()
                # 弹出 `is_last` 字段，判断是否为最后一个
                is_last = item.pop("is_last")
                # 将 item 添加至 accumulator
                accumulator.append(item)
                # 如果是最后一个，返回 accumulator
                if is_last:
                    return accumulator

        # 当 is_last 为 False 时，继续循环
        while not is_last:
            # 使用 infer 方法处理下一个迭代器的值，传入参数 self.params
            processed = self.infer(next(self.iterator), **self.params)
            # 如果 loader_batch_size 不为 None
            if self.loader_batch_size is not None:
                # 如果 processed 是 torch.Tensor 类型
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                # 如果 first_tensor 是列表
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                # 如果 observed_batch_size 在 0 和 loader_batch_size 之间
                if 0 < observed_batch_size < self.loader_batch_size:
                    # 可能是最后一个批次，因此不能展开太多元素
                    # 将 loader_batch_size 更新为 observed_batch_size
                    self.loader_batch_size = observed_batch_size
                # 将 processed 保存到 _loader_batch_data
                self._loader_batch_data = processed
                # 将 _loader_batch_index 重置为 0
                self._loader_batch_index = 0
                # 当 _loader_batch_index 小于 loader_batch_size 时，继续循环
                while self._loader_batch_index < self.loader_batch_size:
                    # 获取 loader_batch_item 中的 item
                    item = self.loader_batch_item()
                    # 弹出 `is_last` 字段，判断是否为最后一个
                    is_last = item.pop("is_last")
                    # 将 item 添加至 accumulator
                    accumulator.append(item)
                    # 如果是最后一个，返回 accumulator
                    if is_last:
                        return accumulator
            # 如果 loader_batch_size 为 None
            else:
                # 获取 processed 的 item
                item = processed
                # 弹出 `is_last` 字段，判断是否为最后一个
                is_last = item.pop("is_last")
                # 将 item 添加至 accumulator
                accumulator.append(item)
        # 返回 accumulator
        return accumulator
# 定义一个 KeyDataset 类，继承自 Dataset 类
class KeyDataset(Dataset):
    # 初始化方法，接收参数 dataset（数据集）和 key（键）
    def __init__(self, dataset: Dataset, key: str):
        # 将参数 dataset 存储到实例变量中
        self.dataset = dataset
        # 将参数 key 存储到实例变量中
        self.key = key

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 根据索引 i 获取数据集中对应键 key 的值
    def __getitem__(self, i):
        return self.dataset[i][self.key]


# 定义一个 KeyPairDataset 类，继承自 Dataset 类
class KeyPairDataset(Dataset):
    # 初始化方法，接收参数 dataset（数据集）、key1（键1）和 key2（键2）
    def __init__(self, dataset: Dataset, key1: str, key2: str):
        # 将参数 dataset 存储到实例变量中
        self.dataset = dataset
        # 将参数 key1 存储到实例变量中
        self.key1 = key1
        # 将参数 key2 存储到实例变量中
        self.key2 = key2

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 根据索引 i 获取数据集中对应键 key1 和 key2 的值，并以字典的形式返回
    def __getitem__(self, i):
        return {"text": self.dataset[i][self.key1], "text_pair": self.dataset[i][self.key2]}
```