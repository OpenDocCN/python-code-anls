# `.\graphrag\graphrag\index\graph\extractors\community_reports\prep_community_report_context.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_community_reports and load_strategy methods definition."""

import logging
from typing import cast

import pandas as pd

# 导入本地模块和函数
import graphrag.index.graph.extractors.community_reports.schemas as schemas
from graphrag.index.utils.dataframes import (
    antijoin,
    drop_columns,
    join,
    select,
    transform_series,
    union,
    where_column_equals,
)

# 导入相关模块
from .build_mixed_context import build_mixed_context
from .sort_context import sort_context
from .utils import set_context_size

# 设置日志记录器
log = logging.getLogger(__name__)


def prep_community_report_context(
    report_df: pd.DataFrame | None,
    community_hierarchy_df: pd.DataFrame,
    local_context_df: pd.DataFrame,
    level: int | str,
    max_tokens: int,
) -> pd.DataFrame:
    """
    Prep context for each community in a given level.

    For each community:
    - Check if local context fits within the limit, if yes use local context
    - If local context exceeds the limit, iteratively replace local context with sub-community reports, starting from the biggest sub-community
    """
    # 如果没有报告数据框，则创建一个空的数据框
    if report_df is None:
        report_df = pd.DataFrame()

    # 将级别转换为整数
    level = int(level)
    # 获取特定级别的上下文数据框
    level_context_df = _at_level(level, local_context_df)
    # 过滤有效的上下文数据框
    valid_context_df = _within_context(level_context_df)
    # 过滤无效的上下文数据框
    invalid_context_df = _exceeding_context(level_context_df)

    # 如果不存在无效的上下文记录，则返回有效的上下文数据框
    # 这种情况仅发生在社区层次结构的最底层，即没有子社区的地方
    if invalid_context_df.empty:
        return valid_context_df

    # 如果报告数据框为空，则修剪无效上下文记录的本地上下文
    invalid_context_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
        invalid_context_df, max_tokens
    )
    set_context_size(invalid_context_df)
    invalid_context_df[schemas.CONTEXT_EXCEED_FLAG] = 0
    return union(valid_context_df, invalid_context_df)

    # 从级别上下文数据框中排除报告数据框中的记录
    level_context_df = _antijoin_reports(level_context_df, report_df)

    # 对于每个无效的上下文记录，尝试使用子社区报告替换
    # 首先获取每个子社区的本地上下文和报告（如果可用）
    sub_context_df = _get_subcontext_df(level + 1, report_df, local_context_df)
    community_df = _get_community_df(
        level, invalid_context_df, sub_context_df, community_hierarchy_df, max_tokens
    )

    # 处理任何剩余无效记录，这些记录无法用子社区报告替换
    # 这种情况很少发生，但如果发生，我们将修剪本地上下文以适应限制
    remaining_df = _antijoin_reports(invalid_context_df, community_df)
    remaining_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
        remaining_df, max_tokens
    )

    result = union(valid_context_df, community_df, remaining_df)
    set_context_size(result)
    result[schemas.CONTEXT_EXCEED_FLAG] = 0
    return result


注释：


# 返回函数的结果


这行代码简单地返回函数中定义的变量 `result` 的值作为函数的返回结果。
def _drop_community_level(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the community level column from the dataframe."""
    # 调用 drop_columns 函数，删除数据框中的社区级别列
    return drop_columns(df, schemas.COMMUNITY_LEVEL)


def _at_level(level: int, df: pd.DataFrame) -> pd.DataFrame:
    """Return records at the given level."""
    # 调用 where_column_equals 函数，返回数据框中社区级别与指定级别相符的记录
    return where_column_equals(df, schemas.COMMUNITY_LEVEL, level)


def _exceeding_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return records where the context exceeds the limit."""
    # 调用 where_column_equals 函数，返回数据框中上下文超过限制的记录
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 1)


def _within_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return records where the context is within the limit."""
    # 调用 where_column_equals 函数，返回数据框中上下文在限制范围内的记录
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 0)


def _antijoin_reports(df: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Return records in df that are not in reports."""
    # 调用 antijoin 函数，返回数据框中不在报告数据框中的记录
    return antijoin(df, reports, schemas.NODE_COMMUNITY)


def _sort_and_trim_context(df: pd.DataFrame, max_tokens: int) -> pd.Series:
    """Sort and trim context to fit the limit."""
    # 提取数据框中的全部上下文列，然后通过 sort_context 函数排序和修剪以符合最大标记数
    series = cast(pd.Series, df[schemas.ALL_CONTEXT])
    return transform_series(series, lambda x: sort_context(x, max_tokens=max_tokens))


def _build_mixed_context(df: pd.DataFrame, max_tokens: int) -> pd.Series:
    """Sort and trim context to fit the limit."""
    # 提取数据框中的全部上下文列，然后通过 build_mixed_context 函数排序和修剪以符合最大标记数
    series = cast(pd.Series, df[schemas.ALL_CONTEXT])
    return transform_series(
        series, lambda x: build_mixed_context(x, max_tokens=max_tokens)
    )


def _get_subcontext_df(
    level: int, report_df: pd.DataFrame, local_context_df: pd.DataFrame
) -> pd.DataFrame:
    """Get sub-community context for each community."""
    # 获取报告数据框中指定级别的子社区上下文数据框
    sub_report_df = _drop_community_level(_at_level(level, report_df))
    # 获取本地上下文数据框中指定级别的记录
    sub_context_df = _at_level(level, local_context_df)
    # 将子社区上下文数据框和本地上下文数据框按节点社区进行连接
    sub_context_df = join(sub_context_df, sub_report_df, schemas.NODE_COMMUNITY)
    # 重命名连接后的数据框的节点社区列为子社区列，并返回结果
    sub_context_df.rename(
        columns={schemas.NODE_COMMUNITY: schemas.SUB_COMMUNITY}, inplace=True
    )
    return sub_context_df


def _get_community_df(
    level: int,
    invalid_context_df: pd.DataFrame,
    sub_context_df: pd.DataFrame,
    community_hierarchy_df: pd.DataFrame,
    max_tokens: int,
) -> pd.DataFrame:
    """Get community context for each community."""
    # 获取社区层次结构数据框中指定级别的社区数据框
    community_df = _drop_community_level(_at_level(level, community_hierarchy_df))
    # 从无效上下文数据框中选择节点社区列
    invalid_community_ids = select(invalid_context_df, schemas.NODE_COMMUNITY)
    # 从子社区上下文数据框中选择子社区、完整内容、全部上下文和上下文大小列
    subcontext_selection = select(
        sub_context_df,
        schemas.SUB_COMMUNITY,
        schemas.FULL_CONTENT,
        schemas.ALL_CONTEXT,
        schemas.CONTEXT_SIZE,
    )
    # 内连接无效社区和社区数据框，获取无效社区信息
    invalid_communities = join(
        community_df, invalid_community_ids, schemas.NODE_COMMUNITY, "inner"
    )
    # 使用子社区选择数据连接社区和无效社区数据，获取社区数据
    community_df = join(
        invalid_communities, subcontext_selection, schemas.SUB_COMMUNITY
    )
    # 使用 apply 方法将每行数据按照指定规则转换成新的列 ALL_CONTEXT
    community_df[schemas.ALL_CONTEXT] = community_df.apply(
        lambda x: {
            schemas.SUB_COMMUNITY: x[schemas.SUB_COMMUNITY],  # 将 SUB_COMMUNITY 列的值作为新列的一个子键
            schemas.ALL_CONTEXT: x[schemas.ALL_CONTEXT],  # 将 ALL_CONTEXT 列的值作为新列的一个子键
            schemas.FULL_CONTENT: x[schemas.FULL_CONTENT],  # 将 FULL_CONTENT 列的值作为新列的一个子键
            schemas.CONTEXT_SIZE: x[schemas.CONTEXT_SIZE],  # 将 CONTEXT_SIZE 列的值作为新列的一个子键
        },
        axis=1,
    )
    # 对 NODE_COMMUNITY 进行分组，并将 ALL_CONTEXT 列的值组成列表
    community_df = (
        community_df.groupby(schemas.NODE_COMMUNITY)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    # 使用 _build_mixed_context 方法构建混合内容，并存储到 CONTEXT_STRING 列
    community_df[schemas.CONTEXT_STRING] = _build_mixed_context(
        community_df, max_tokens
    )
    # 将 level 存储到 COMMUNITY_LEVEL 列
    community_df[schemas.COMMUNITY_LEVEL] = level
    # 返回结果数据框
    return community_df
```