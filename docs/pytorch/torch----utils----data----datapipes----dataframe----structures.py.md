# `.\pytorch\torch\utils\data\datapipes\dataframe\structures.py`

```
# 导入数据处理模块中的 DataFrame 封装器
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
# 导入数据处理模块中的 DataChunk 类
from torch.utils.data.datapipes.datapipe import DataChunk

# 声明模块中公开的类名列表，只包含 DataChunkDF 这一个类
__all__ = ["DataChunkDF"]

# 定义一个新的类 DataChunkDF，继承自 DataChunk 类
class DataChunkDF(DataChunk):
    """DataChunkDF iterating over individual items inside of DataFrame containers, to access DataFrames user `raw_iterator`."""

    # 实现迭代器接口，迭代处理 items 中的每一个 DataFrame
    def __iter__(self):
        for df in self.items:
            # 使用 DataFrame 封装器中的 iterate 函数来迭代处理 DataFrame
            yield from df_wrapper.iterate(df)

    # 实现长度接口，返回所有 items 中 DataFrame 的总长度
    def __len__(self):
        total_len = 0
        for df in self.items:
            # 使用 DataFrame 封装器中的 get_len 函数获取 DataFrame 的长度并累加
            total_len += df_wrapper.get_len(df)
        return total_len
```