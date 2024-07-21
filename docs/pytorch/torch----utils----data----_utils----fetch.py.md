# `.\pytorch\torch\utils\data\_utils\fetch.py`

```
# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

# 定义基础数据集获取器类，用于从可迭代或映射样式的数据集中获取数据
class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset              # 初始化数据集
        self.auto_collation = auto_collation  # 是否自动整合数据的标志
        self.collate_fn = collate_fn        # 数据整合函数
        self.drop_last = drop_last          # 是否丢弃最后一个批次的标志

    # 抽象方法，需要在子类中实现
    def fetch(self, possibly_batched_index):
        raise NotImplementedError


# 继承自 _BaseDatasetFetcher 的可迭代数据集获取器类
class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)   # 使用数据集创建迭代器
        self.ended = False                  # 迭代器结束标志

    # 从数据集中获取数据的方法
    def fetch(self, possibly_batched_index):
        if self.ended:                      # 如果迭代器已结束则抛出 StopIteration 异常
            raise StopIteration

        if self.auto_collation:             # 如果自动整合数据
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))  # 从迭代器中获取下一个数据
                except StopIteration:
                    self.ended = True           # 如果迭代结束则设置结束标志为 True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration           # 如果数据为空或者需要丢弃最后一个批次且数据不足，则抛出 StopIteration 异常
        else:
            data = next(self.dataset_iter)    # 否则直接从迭代器中获取下一个数据
        return self.collate_fn(data)          # 返回整合后的数据


# 继承自 _BaseDatasetFetcher 的映射数据集获取器类
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:               # 如果自动整合数据
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)  # 如果数据集有 __getitems__ 方法则调用
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]  # 否则遍历索引列表获取数据
        else:
            data = self.dataset[possibly_batched_index]  # 直接根据索引列表获取数据
        return self.collate_fn(data)          # 返回整合后的数据
```