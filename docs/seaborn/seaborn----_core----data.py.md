# `D:\src\scipysrc\seaborn\seaborn\_core\data.py`

```
"""
Components for parsing variable assignments and internally representing plot data.
"""
# 导入必要的模块和类
from __future__ import annotations  # 导入用于支持类型提示的模块

from collections.abc import Mapping, Sized  # 导入用于抽象基类的 Mapping 和 Sized
from typing import cast  # 导入用于类型强制转换的 cast 函数
import warnings  # 导入警告模块

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import DataFrame  # 导入 DataFrame 类

from seaborn._core.typing import DataSource, VariableSpec, ColumnName  # 导入 seaborn 中的类型定义
from seaborn.utils import _version_predates  # 导入 seaborn 中的版本预测函数


class PlotData:
    """
    Data table with plot variable schema and mapping to original names.

    Contains logic for parsing variable specification arguments and updating
    the table with layer-specific data and/or mappings.

    Parameters
    ----------
    data
        Input data where variable names map to vector values.
    variables
        Keys are names of plot variables (x, y, ...) each value is one of:

        - name of a column (or index level, or dictionary entry) in `data`
        - vector in any format that can construct a :class:`pandas.DataFrame`

    Attributes
    ----------
    frame
        Data table with column names having defined plot variables.
    names
        Dictionary mapping plot variable names to names in source data structure(s).
    ids
        Dictionary mapping plot variable names to unique data source identifiers.

    """

    frame: DataFrame  # 存储具有定义的绘图变量列名的数据表
    frames: dict[tuple, DataFrame]  # 存储多个帧的字典，用于支持特定操作
    names: dict[str, str | None]  # 将绘图变量名映射到源数据结构中的名称的字典
    ids: dict[str, str | int]  # 将绘图变量名映射到唯一数据源标识符的字典
    source_data: DataSource  # 存储源数据
    source_vars: dict[str, VariableSpec]  # 存储绘图变量规范的字典

    def __init__(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec],
    ):
        # 处理输入的数据源
        data = handle_data_source(data)
        # 分配变量到数据表中，并获取结果帧、名称映射和标识符字典
        frame, names, ids = self._assign_variables(data, variables)

        self.frame = frame  # 将结果帧存储在对象属性中
        self.names = names  # 将名称映射存储在对象属性中
        self.ids = ids  # 将标识符映射存储在对象属性中

        # 初始化一个空的帧字典，用于特定操作时支持多个帧
        self.frames = {}

        self.source_data = data  # 存储源数据到对象属性中
        self.source_vars = variables  # 存储变量规范到对象属性中

    def __contains__(self, key: str) -> bool:
        """
        Boolean check on whether a variable is defined in this dataset.
        """
        if self.frame is None:
            # 如果主帧为空，则检查所有子帧中是否包含指定的键
            return any(key in df for df in self.frames.values())
        # 否则，在主帧中检查指定的键是否存在
        return key in self.frame

    def join(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec] | None,
        # 继续编写 join 方法的注释
        ) -> PlotData:
        """Add, replace, or drop variables and return as a new dataset."""
        # 如果未提供数据（data），则默认使用上游数据源的内容
        if data is None:
            data = self.source_data

        # TODO 允许 `data` 是一个函数（在源数据上调用该函数？）

        # 如果没有传入变量（variables），则使用原始的变量集合（source_vars）
        if not variables:
            variables = self.source_vars

        # 找出需要从当前层级继承的变量，即值为 None 的变量
        disinherit = [k for k, v in variables.items() if v is None]

        # 使用传入的数据和变量创建一个新的数据集对象
        new = PlotData(data, variables)

        # -- 使用新信息更新继承的数据源（DataSource）

        # 计算需要从当前数据框（frame）中丢弃的列
        drop_cols = [k for k in self.frame if k in new.frame or k in disinherit]
        parts = [self.frame.drop(columns=drop_cols), new.frame]

        # 因为我们在合并不同的列，可能更自然地认为是 "merge"/"join" 操作。
        # 但是使用 concat，因为一些简单的测试表明它可能稍微更快一些。
        frame = pd.concat(parts, axis=1, sort=False, copy=False)

        # 更新变量名称映射，排除不需要继承的变量
        names = {k: v for k, v in self.names.items() if k not in disinherit}
        names.update(new.names)

        # 更新变量标识映射，排除不需要继承的变量
        ids = {k: v for k, v in self.ids.items() if k not in disinherit}
        ids.update(new.ids)

        # 将新的数据框、变量名称映射和变量标识映射分配给新数据集对象
        new.frame = frame
        new.names = names
        new.ids = ids

        # 多个链式操作应始终继承自原始对象的源数据和变量集合
        new.source_data = self.source_data
        new.source_vars = self.source_vars

        return new
# 将数据源对象转换为通用的联合表示形式（DataFrame、Mapping 或 None）
def handle_data_source(data: object) -> pd.DataFrame | Mapping | None:
    """Convert the data source object to a common union representation."""
    # 如果数据源是 pd.DataFrame 类型或者具有 "__dataframe__" 属性
    if isinstance(data, pd.DataFrame) or hasattr(data, "__dataframe__"):
        # 转换为 Pandas DataFrame 对象
        data = convert_dataframe_to_pandas(data)
    # 如果数据源不为 None 且不是 Mapping 类型，则抛出类型错误
    elif data is not None and not isinstance(data, Mapping):
        err = f"Data source must be a DataFrame or Mapping, not {type(data)!r}."
        raise TypeError(err)

    # 返回处理后的数据源对象
    return data


# 将非 Pandas DataFrame 对象转换为 Pandas DataFrame
def convert_dataframe_to_pandas(data: object) -> pd.DataFrame:
    """Use the DataFrame exchange protocol, or fail gracefully."""
    # 如果数据源已经是 Pandas DataFrame，则直接返回
    if isinstance(data, pd.DataFrame):
        return data

    # 如果当前 pandas 版本不支持 DataFrame 交换协议
    if not hasattr(pd.api, "interchange"):
        msg = (
            "Support for non-pandas DataFrame objects requires a version of pandas "
            "that implements the DataFrame interchange protocol. Please upgrade "
            "your pandas version or coerce your data to pandas before passing "
            "it to seaborn."
        )
        raise TypeError(msg)

    # 如果当前 pandas 版本早于 2.0.2，给出警告信息
    if _version_predates(pd, "2.0.2"):
        msg = (
            "DataFrame interchange with pandas<2.0.2 has some known issues. "
            f"You are using pandas {pd.__version__}. "
            "Continuing, but it is recommended to carefully inspect the results and to "
            "consider upgrading."
        )
        warnings.warn(msg, stacklevel=2)

    try:
        # 使用 DataFrame 交换协议将输入的数据源转换为 Pandas DataFrame
        # 这将转换输入 DataFrame 中的所有列，尽管在 Plot() 时可能只使用一两列。
        # 在调用 Plot() 之前选择将在绘图中使用的列可能更有效。
        # 在对象接口中，可能只有在稍后的 Plot.add() 中引用的变量。如果这是一个瓶颈，解决这个问题通常是一个困难的问题。
        return pd.api.interchange.from_dataframe(data)
    except Exception as err:
        # 遇到异常时，抛出运行时错误并提供详细信息
        msg = (
            "Encountered an exception when converting data source "
            "to a pandas DataFrame. See traceback above for details."
        )
        raise RuntimeError(msg) from err
```