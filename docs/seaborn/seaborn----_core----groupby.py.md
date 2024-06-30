# `D:\src\scipysrc\seaborn\seaborn\_core\groupby.py`

```
"""Simplified split-apply-combine paradigm on dataframes for internal use."""
# 导入必要的模块和函数

from __future__ import annotations

from typing import cast, Iterable

import pandas as pd

# 导入用于排序的相关函数
from seaborn._core.rules import categorical_order

# 类型检查相关导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from pandas import DataFrame, MultiIndex, Index

# 定义 GroupBy 类，用于封装 Pandas GroupBy 操作
class GroupBy:
    """
    Interface for Pandas GroupBy operations allowing specified group order.

    Writing our own class to do this has a few advantages:
    - It constrains the interface between Plot and Stat/Move objects
    - It allows control over the row order of the GroupBy result, which is
      important when using in the context of some Move operations (dodge, stack, ...)
    - It simplifies some complexities regarding the return type and Index contents
      one encounters with Pandas, especially for DataFrame -> DataFrame applies
    - It increases future flexibility regarding alternate DataFrame libraries

    """
    def __init__(self, order: list[str] | dict[str, list | None]):
        """
        Initialize the GroupBy from grouping variables and optional level orders.

        Parameters
        ----------
        order
            List of variable names or dict mapping names to desired level orders.
            Level order values can be None to use default ordering rules. The
            variables can include names that are not expected to appear in the
            data; these will be dropped before the groups are defined.

        """
        # 检查是否有有效的分组变量
        if not order:
            raise ValueError("GroupBy requires at least one grouping variable")

        # 如果 order 是列表，则转换为字典，使用默认的 level order 规则
        if isinstance(order, list):
            order = {k: None for k in order}
        self.order = order

    def _get_groups(
        self, data: DataFrame
    ) -> tuple[str | list[str], Index | MultiIndex]:
        """Return index with Cartesian product of ordered grouping variable levels."""
        # 初始化 levels 字典来存储变量和其对应的排序顺序
        levels = {}
        # 遍历定义的分组变量和对应的排序规则
        for var, order in self.order.items():
            # 如果变量存在于数据中
            if var in data:
                # 如果没有指定排序规则，则使用默认的分类顺序
                if order is None:
                    order = categorical_order(data[var])
                levels[var] = order

        # 初始化 grouper 和 groups 变量
        grouper: str | list[str]
        groups: Index | MultiIndex
        # 如果 levels 为空字典，则 grouper 为空列表，groups 为空的索引
        if not levels:
            grouper = []
            groups = pd.Index([])
        # 如果 levels 中有多个变量，则 grouper 是变量名列表，groups 是多级索引
        elif len(levels) > 1:
            grouper = list(levels)
            groups = pd.MultiIndex.from_product(levels.values(), names=grouper)
        # 否则，只有一个变量，则 grouper 是该变量名，groups 是单级索引
        else:
            grouper, = list(levels)
            groups = pd.Index(levels[grouper], name=grouper)
        return grouper, groups

    def _reorder_columns(self, res, data):
        """Reorder result columns to match original order with new columns appended."""
        # 将结果列重新排序，使其与原始数据列顺序一致，并追加新列
        cols = [c for c in data if c in res]
        cols += [c for c in res if c not in data]
        return res.reindex(columns=pd.Index(cols))
    def agg(self, data: DataFrame, *args, **kwargs) -> DataFrame:
        """
        Reduce each group to a single row in the output.

        The output will have a row for each unique combination of the grouping
        variable levels with null values for the aggregated variable(s) where
        those combinations do not appear in the dataset.

        """
        # 获取分组器和分组信息
        grouper, groups = self._get_groups(data)

        if not grouper:
            # 如果没有分组变量存在于数据框中，则抛出值错误异常
            raise ValueError("No grouping variables are present in dataframe")

        # 对数据按照分组器进行分组聚合操作，并重新索引为指定的分组集合
        res = (
            data
            .groupby(grouper, sort=False, observed=False)
            .agg(*args, **kwargs)
            .reindex(groups)
            .reset_index()
            .pipe(self._reorder_columns, data)
        )

        # 返回聚合结果
        return res

    def apply(
        self, data: DataFrame, func: Callable[..., DataFrame],
        *args, **kwargs,
    ) -> DataFrame:
        """Apply a DataFrame -> DataFrame mapping to each group."""
        # 获取分组器和分组信息
        grouper, groups = self._get_groups(data)

        if not grouper:
            # 如果没有分组变量存在于数据框中，则直接对整个数据框应用函数并重新排序列
            return self._reorder_columns(func(data, *args, **kwargs), data)

        # 初始化一个空字典用于存储各个分组的处理结果
        parts = {}
        # 遍历按分组器分组后的数据框
        for key, part_df in data.groupby(grouper, sort=False, observed=False):
            # 对每个分组应用指定函数并存储结果
            parts[key] = func(part_df, *args, **kwargs)

        # 初始化一个空列表用于存储最终结果
        stack = []
        # 遍历预定义的分组集合
        for key in groups:
            # 如果当前分组在处理结果中存在
            if key in parts:
                # 如果分组器是列表，则说明有多级索引，需要转换为字典格式
                if isinstance(grouper, list):
                    group_ids = dict(zip(grouper, cast(Iterable, key)))
                else:
                    group_ids = {grouper: key}
                # 将当前分组的结果与分组信息合并，并加入到结果栈中
                stack.append(parts[key].assign(**group_ids))

        # 将所有结果拼接为一个数据框，并忽略索引
        res = pd.concat(stack, ignore_index=True)
        # 对最终结果重新排序列，并返回
        return self._reorder_columns(res, data)
```