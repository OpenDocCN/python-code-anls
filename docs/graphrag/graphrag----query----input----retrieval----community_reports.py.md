# `.\graphrag\graphrag\query\input\retrieval\community_reports.py`

```py
# 导入必要的模块和类型定义
from typing import Any, cast

import pandas as pd  # 导入 pandas 库

from graphrag.model import CommunityReport, Entity  # 导入自定义模块和类


def get_candidate_communities(
    selected_entities: list[Entity],  # 输入参数：选定的实体列表
    community_reports: list[CommunityReport],  # 输入参数：社区报告列表
    include_community_rank: bool = False,  # 是否包含社区排名，默认为 False
    use_community_summary: bool = False,  # 是否使用社区摘要，默认为 False
) -> pd.DataFrame:
    """获取与选定实体相关的所有社区报告。"""
    # 收集所有选定实体关联的社区 ID
    selected_community_ids = [
        entity.community_ids for entity in selected_entities if entity.community_ids
    ]
    # 展开嵌套列表
    selected_community_ids = [
        item for sublist in selected_community_ids for item in sublist
    ]
    # 筛选出与选定社区 ID 相匹配的社区报告
    selected_reports = [
        community
        for community in community_reports
        if community.id in selected_community_ids
    ]
    # 转换为 DataFrame 并返回
    return to_community_report_dataframe(
        reports=selected_reports,
        include_community_rank=include_community_rank,
        use_community_summary=use_community_summary,
    )


def to_community_report_dataframe(
    reports: list[CommunityReport],  # 输入参数：社区报告列表
    include_community_rank: bool = False,  # 是否包含社区排名，默认为 False
    use_community_summary: bool = False,  # 是否使用社区摘要，默认为 False
) -> pd.DataFrame:
    """将社区报告列表转换为 pandas DataFrame。"""
    if len(reports) == 0:
        return pd.DataFrame()  # 如果没有报告，返回空 DataFrame

    # 构建表头
    header = ["id", "title"]  # 固定列：ID 和标题
    # 获取第一个报告的属性字段列表
    attribute_cols = list(reports[0].attributes.keys()) if reports[0].attributes else []
    # 筛选出不在固定列中的属性字段
    attribute_cols = [col for col in attribute_cols if col not in header]
    # 添加属性列和摘要或内容列
    header.extend(attribute_cols)
    header.append("summary" if use_community_summary else "content")
    # 如果包含社区排名，则添加排名列
    if include_community_rank:
        header.append("rank")

    # 构建记录列表
    records = []
    for report in reports:
        # 创建新记录
        new_record = [
            report.short_id if report.short_id else "",  # 报告的短 ID，如果有的话
            report.title,  # 报告标题
            *[
                str(report.attributes.get(field, ""))
                if report.attributes and report.attributes.get(field)
                else ""
                for field in attribute_cols
            ],  # 属性字段值列表
        ]
        # 添加摘要或全文内容
        new_record.append(
            report.summary if use_community_summary else report.full_content
        )
        # 如果包含社区排名，则添加排名
        if include_community_rank:
            new_record.append(str(report.rank))
        records.append(new_record)  # 添加记录到记录列表

    # 返回构建的 DataFrame
    return pd.DataFrame(records, columns=cast(Any, header))
```