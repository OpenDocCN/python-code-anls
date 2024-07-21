# `.\pytorch\torch\utils\data\datapipes\dataframe\datapipes.py`

```
# 导入必要的模块和函数
# mypy: allow-untyped-defs
import random

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe

# 定义模块中公开的类名
__all__ = [
    "ConcatDataFramesPipe",
    "DataFramesAsTuplesPipe",
    "ExampleAggregateAsDataFrames",
    "FilterDataFramesPipe",
    "PerRowDataFramesPipe",
    "ShuffleDataFramesPipe",
]

# 将 DataFrames 转换为元组的数据管道类
@functional_datapipe("_dataframes_as_tuples")
class DataFramesAsTuplesPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # 将 DataFrame 转换为元组并迭代返回
            yield from df_wrapper.iterate(df)

# 按行处理 DataFrames 的数据管道类
@functional_datapipe("_dataframes_per_row", enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for df in self.source_datapipe:
            # 按行迭代返回 DataFrame
            for i in range(len(df)):
                yield df[i : i + 1]

# 连接多个 DataFrames 的数据管道类
@functional_datapipe("_dataframes_concat", enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, batch=3):
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                # 将多个 DataFrame 连接起来并返回
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)

# 对 DataFrames 进行随机打乱的数据管道类
@functional_datapipe("_dataframes_shuffle", enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        size = None
        all_buffer = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            for i in range(df_wrapper.get_len(df)):
                all_buffer.append(df_wrapper.get_item(df, i))
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                # 打乱后的 DataFrame 返回
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)

# 对 DataFrames 进行过滤的数据管道类
@functional_datapipe("_dataframes_filter", enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):
    def __init__(self, source_datapipe, filter_fn):
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn
    # 定义一个迭代器方法，用于迭代处理self.source_datapipe中的数据
    def __iter__(self):
        # 初始化变量size为None，用于存储数据集的大小
        size = None
        # all_buffer用于存储所有的数据块
        all_buffer = []
        # filter_res用于存储筛选函数self.filter_fn对应用结果的列表
        filter_res = []
        
        # 遍历数据管道self.source_datapipe中的每个数据帧df
        for df in self.source_datapipe:
            # 如果size为None，则设置为当前数据帧df的索引长度
            if size is None:
                size = len(df.index)
            # 遍历当前数据帧df的索引
            for i in range(len(df.index)):
                # 将当前行i到i+1的数据块添加到all_buffer中
                all_buffer.append(df[i : i + 1])
                # 将self.filter_fn应用于当前行df.iloc[i]，并将结果添加到filter_res中
                filter_res.append(self.filter_fn(df.iloc[i]))

        # 初始化一个空列表buffer，用于存储满足条件的数据块
        buffer = []
        # 遍历all_buffer和filter_res中的元素
        for df, res in zip(all_buffer, filter_res):
            # 如果res为True，将当前数据块df添加到buffer中
            if res:
                buffer.append(df)
                # 如果buffer的长度等于size，生成一个包含buffer中所有数据块的数据帧，并清空buffer
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        # 如果buffer中仍有剩余数据块，生成一个包含buffer中所有数据块的数据帧
        if len(buffer):
            yield df_wrapper.concat(buffer)
# 使用自定义装饰器将类标记为数据管道，并启用 DataFrame API 跟踪功能
@functional_datapipe("_to_dataframes_pipe", enable_df_api_tracing=True)
# 定义一个继承自 DFIterDataPipe 的类 ExampleAggregateAsDataFrames
class ExampleAggregateAsDataFrames(DFIterDataPipe):
    # 初始化方法，接受数据源数据管道、数据帧大小和列名作为参数
    def __init__(self, source_datapipe, dataframe_size=10, columns=None):
        self.source_datapipe = source_datapipe  # 设置数据源数据管道
        self.columns = columns  # 设置列名
        self.dataframe_size = dataframe_size  # 设置数据帧大小

    # 辅助函数，尝试将输入转换为列表，捕获所有异常并返回包含单个元素的列表
    def _as_list(self, item):
        try:
            return list(item)
        except (
            Exception
        ):  # 处理所有异常，应替换为更具体的可迭代对象异常
            return [item]

    # 实现迭代器接口的方法，用于生成聚合后的数据帧
    def __iter__(self):
        aggregate = []  # 初始化聚合列表
        # 遍历数据源数据管道中的每个项
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))  # 将每个项转换为列表并添加到聚合列表中
            if len(aggregate) == self.dataframe_size:  # 如果聚合列表大小达到指定数据帧大小
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)  # 生成数据帧并返回
                aggregate = []  # 重置聚合列表
        # 处理剩余的不足数据帧大小的项
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)  # 生成最后一个数据帧并返回
```