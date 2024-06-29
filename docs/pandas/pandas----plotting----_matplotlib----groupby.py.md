# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\groupby.py`

```
# 从未来导入类型提示中的注解支持
from __future__ import annotations

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入numpy库，并使用np作为别名
import numpy as np

# 导入pandas库中处理缺失值的模块
from pandas.core.dtypes.missing import remove_na_arraylike

# 导入pandas库中的MultiIndex和concat函数
from pandas import (
    MultiIndex,
    concat,
)

# 导入pandas库中的私有模块，用于单个字符串列表的解包
from pandas.plotting._matplotlib.misc import unpack_single_str_list

# 如果是类型检查模式，导入必要的类型
if TYPE_CHECKING:
    from collections.abc import Hashable

    # 导入pandas库中的IndexLabel类型
    from pandas._typing import IndexLabel

    # 导入pandas库中的DataFrame和Series类型
    from pandas import (
        DataFrame,
        Series,
    )


def create_iter_data_given_by(
    data: DataFrame, kind: str = "hist"
) -> dict[Hashable, DataFrame | Series]:
    """
    根据`by`参数是否被赋值，为迭代创建数据，仅适用于`hist`和`boxplot`。

    如果`by`被赋值，则返回一个DataFrame字典，字典的键是组中的值。
    如果`by`未被赋值，则原样返回输入数据，保持当前iter_data的状态。

    参数
    ----------
    data : 从`_compute_plot_data`方法重格式化的分组数据。
    kind : str，绘图类型。该函数仅用于`hist`和`box`图。

    返回
    -------
    iter_data : DataFrame或DataFrame字典

    示例
    --------
    如果`by`被赋值：

    >>> import numpy as np
    >>> tuples = [("h1", "a"), ("h1", "b"), ("h2", "a"), ("h2", "b")]
    >>> mi = pd.MultiIndex.from_tuples(tuples)
    >>> value = [[1, 3, np.nan, np.nan], [3, 4, np.nan, np.nan], [np.nan, np.nan, 5, 6]]
    >>> data = pd.DataFrame(value, columns=mi)
    >>> create_iter_data_given_by(data)
    {'h1':     h1
         a    b
    0  1.0  3.0
    1  3.0  4.0
    2  NaN  NaN, 'h2':     h2
         a    b
    0  NaN  NaN
    1  NaN  NaN
    2  5.0  6.0}
    """

    # 对于`hist`图，转换之前，level 0中的值是组中的值和子图标题，后续用于列子选择和迭代；
    # 对于`box`图，level 1中的值是要显示的列名，并且用于迭代和作为子图标题。
    if kind == "hist":
        level = 0
    else:
        level = 1

    # 根据MI的level值选择子列，并且如果`by`被赋值，data必须是MultiIndex DataFrame
    assert isinstance(data.columns, MultiIndex)
    return {
        col: data.loc[:, data.columns.get_level_values(level) == col]
        for col in data.columns.levels[level]
    }


def reconstruct_data_with_by(
    data: DataFrame, by: IndexLabel, cols: IndexLabel
) -> DataFrame:
    """
    内部函数，用于对数据进行分组，并将多级索引列名重新分配到结果中，以便让分组数据在_compute_plot_data方法中使用。

    参数
    ----------
    data : 原始要绘制的DataFrame
    by : 用户选择的分组`by`参数
    cols : 数据集的列（不包括`by`中使用的列）

    返回
    -------
    重构后带有MultiIndex列的DataFrame。第一级
    of MI is unique values of groups, and second level of MI is the columns
    selected by users.

    Examples
    --------
    >>> d = {"h": ["h1", "h1", "h2"], "a": [1, 3, 5], "b": [3, 4, 6]}
    >>> df = pd.DataFrame(d)
    >>> reconstruct_data_with_by(df, by="h", cols=["a", "b"])
       h1      h2
       a     b     a     b
    0  1.0   3.0   NaN   NaN
    1  3.0   4.0   NaN   NaN
    2  NaN   NaN   5.0   6.0
    """
    # 将字符串或字符串列表转换为统一的字符串列表
    by_modified = unpack_single_str_list(by)
    # 按照指定的列（by_modified）对数据进行分组
    grouped = data.groupby(by_modified)

    data_list = []
    for key, group in grouped:
        # 创建多级索引，第一级为分组的键值，第二级为用户选择的列
        columns = MultiIndex.from_product([[key], cols])  # type: ignore[list-item]
        # 提取分组中指定的列数据
        sub_group = group[cols]
        # 设置子组的列名为创建的多级索引
        sub_group.columns = columns
        # 将处理后的子组添加到数据列表中
        data_list.append(sub_group)

    # 沿着轴 1 连接数据列表中的各个子组，形成最终重构后的数据
    data = concat(data_list, axis=1)
    return data
# 定义一个函数用于重新格式化给定的 y 数组，根据是否应用 by 参数来进行处理，用于直方图绘制

def reformat_hist_y_given_by(y: np.ndarray, by: IndexLabel | None) -> np.ndarray:
    """Internal function to reformat y given `by` is applied or not for hist plot.

    If by is None, input y is 1-d with NaN removed; and if by is not None, groupby
    will take place and input y is multi-dimensional array.
    """
    # 如果 by 不为 None 并且 y 的维度大于 1，则执行以下代码块
    if by is not None and len(y.shape) > 1:
        # 对 y 的每一列应用 remove_na_arraylike 函数，并重新组合成多维数组返回
        return np.array([remove_na_arraylike(col) for col in y.T]).T
    # 否则，调用 remove_na_arraylike 函数处理 y，并返回处理后的数组
    return remove_na_arraylike(y)
```