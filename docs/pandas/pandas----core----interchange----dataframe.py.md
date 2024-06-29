# `D:\src\scipysrc\pandas\pandas\core\interchange\dataframe.py`

```
from __future__ import annotations
# 导入了新版本的特性，使得可以在类型提示中使用自身类的类型声明

from collections import abc
# 导入了 collections.abc 模块，用于支持抽象基类的集合类型

from typing import TYPE_CHECKING
# 导入了 TYPE_CHECKING，用于类型检查时避免循环导入问题

from pandas.core.interchange.column import PandasColumn
# 从 pandas 的 interchange.column 模块中导入 PandasColumn 类

from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
# 从 pandas 的 interchange.dataframe_protocol 模块中导入 DataFrame 类，并重命名为 DataFrameXchg

from pandas.core.interchange.utils import maybe_rechunk
# 从 pandas 的 interchange.utils 模块中导入 maybe_rechunk 函数

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )
    # 如果是类型检查模式，导入 collections.abc 中的 Iterable 和 Sequence 类型

    from pandas import (
        DataFrame,
        Index,
    )
    # 如果是类型检查模式，导入 pandas 中的 DataFrame 和 Index 类型


class PandasDataFrameXchg(DataFrameXchg):
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.
    Instances of this (private) class are returned from
    ``pd.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """
    
    def __init__(self, df: DataFrame, allow_copy: bool = True) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `pd.DataFrame.__dataframe__`.
        """
        self._df = df.rename(columns=str)
        # 使用 str 函数重命名 DataFrame 的列名，并保存到私有属性 _df 中
        self._allow_copy = allow_copy
        # 将 allow_copy 参数保存到私有属性 _allow_copy 中
        for i, _col in enumerate(self._df.columns):
            rechunked = maybe_rechunk(self._df.iloc[:, i], allow_copy=allow_copy)
            # 对每一列进行可能的重块化处理，返回处理后的结果
            if rechunked is not None:
                self._df.isetitem(i, rechunked)
                # 如果重块化后的结果不为空，则更新 DataFrame 的第 i 列数据

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> PandasDataFrameXchg:
        # `nan_as_null` can be removed here once it's removed from
        # Dataframe.__dataframe__
        # 返回当前对象的副本，允许指定是否将 NaN 视为空值，并允许指定是否复制数据
        return PandasDataFrameXchg(self._df, allow_copy)

    @property
    def metadata(self) -> dict[str, Index]:
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as Pandas-specific metadata here.
        # 返回一个包含 Pandas 特定元数据的字典，其中包括 DataFrame 的索引信息
        return {"pandas.index": self._df.index}

    def num_columns(self) -> int:
        # 返回 DataFrame 的列数
        return len(self._df.columns)

    def num_rows(self) -> int:
        # 返回 DataFrame 的行数
        return len(self._df)

    def num_chunks(self) -> int:
        # 返回 DataFrame 的块数，此处默认为 1 块
        return 1

    def column_names(self) -> Index:
        # 返回 DataFrame 的列名索引
        return self._df.columns

    def get_column(self, i: int) -> PandasColumn:
        # 返回指定索引 i 的列作为 PandasColumn 对象
        return PandasColumn(self._df.iloc[:, i], allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> PandasColumn:
        # 返回指定名称 name 的列作为 PandasColumn 对象
        return PandasColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> list[PandasColumn]:
        # 返回所有列作为 PandasColumn 对象列表
        return [
            PandasColumn(self._df[name], allow_copy=self._allow_copy)
            for name in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> PandasDataFrameXchg:
        # 选择指定索引列表 indices 对应的列，并返回新的 PandasDataFrameXchg 对象
        if not isinstance(indices, abc.Sequence):
            raise ValueError("`indices` is not a sequence")
            # 如果 indices 不是序列类型，则抛出 ValueError 异常
        if not isinstance(indices, list):
            indices = list(indices)
            # 如果 indices 不是列表类型，则将其转换为列表

        return PandasDataFrameXchg(
            self._df.iloc[:, indices], allow_copy=self._allow_copy
        )
    # 定义一个方法，根据列名列表从当前数据框中选择列，并返回一个新的 PandasDataFrameXchg 对象
    def select_columns_by_name(self, names: list[str]) -> PandasDataFrameXchg:  # type: ignore[override]
        # 检查传入的 `names` 是否为序列类型，若不是则抛出数值错误
        if not isinstance(names, abc.Sequence):
            raise ValueError("`names` is not a sequence")
        # 如果 `names` 不是列表类型，则将其转换为列表
        if not isinstance(names, list):
            names = list(names)

        # 返回根据列名列表选择的数据框的新实例，允许复制数据
        return PandasDataFrameXchg(self._df.loc[:, names], allow_copy=self._allow_copy)

    # 定义一个方法，返回一个可迭代对象，该对象产生数据框的分块
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[PandasDataFrameXchg]:
        """
        Return an iterator yielding the chunks.
        """
        # 如果指定了 n_chunks 且 n_chunks 大于 1
        if n_chunks and n_chunks > 1:
            # 计算数据框的长度
            size = len(self._df)
            # 计算每个块的大小
            step = size // n_chunks
            # 如果数据框长度不能被 n_chunks 整除，则增加步长以容纳余数部分
            if size % n_chunks != 0:
                step += 1
            # 生成器：从 0 开始，以步长为单位，生成 n_chunks 个数据框块的迭代器
            for start in range(0, step * n_chunks, step):
                yield PandasDataFrameXchg(
                    self._df.iloc[start : start + step, :],  # 使用 iloc 切片选取数据框的片段
                    allow_copy=self._allow_copy,
                )
        else:
            # 如果未指定或者 n_chunks 不大于 1，则返回当前对象自身作为唯一的数据框块
            yield self
```