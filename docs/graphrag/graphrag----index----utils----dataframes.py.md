# `.\graphrag\graphrag\index\utils\dataframes.py`

```py
# 版权声明及许可信息

"""包含 DataFrame 工具的模块."""

# 导入所需的模块和类型声明
from collections.abc import Callable
from typing import Any, cast

import pandas as pd
from pandas._typing import MergeHow


def drop_columns(df: pd.DataFrame, *column: str) -> pd.DataFrame:
    """从数据框中删除列."""
    return df.drop(list(column), axis=1)


def where_column_equals(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """返回一个根据列等于特定值筛选后的数据框."""
    return cast(pd.DataFrame, df[df[column] == value])


def antijoin(df: pd.DataFrame, exclude: pd.DataFrame, column: str) -> pd.DataFrame:
    """返回一个反连接后的数据框.

    参数:
    * df: 要应用排除操作的数据框
    * exclude: 包含要移除行的数据框.
    * column: 连接列.
    """
    # 使用外连接合并 df 和 exclude 数据框，并增加指示器列
    result = df.merge(
        exclude[[column]],
        on=column,
        how="outer",
        indicator=True,
    )
    # 如果结果中包含 "_merge" 列，则保留左连接的行，并移除 "_merge" 列
    if "_merge" in result.columns:
        result = result[result["_merge"] == "left_only"].drop("_merge", axis=1)
    return cast(pd.DataFrame, result)


def transform_series(series: pd.Series, fn: Callable[[Any], Any]) -> pd.Series:
    """对序列应用转换函数."""
    return cast(pd.Series, series.apply(fn))


def join(
    left: pd.DataFrame, right: pd.DataFrame, key: str, strategy: MergeHow = "left"
) -> pd.DataFrame:
    """执行表连接操作."""
    return left.merge(right, on=key, how=strategy)


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    """对给定的一组数据框执行并集操作."""
    return pd.concat(list(frames))


def select(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """从数据框中选择列."""
    return cast(pd.DataFrame, df[list(columns)])
```