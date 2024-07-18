# `.\graphrag\graphrag\query\indexer_adapters.py`

```py
# 导入必要的库和模块
from typing import cast  # 引入类型转换的辅助函数 cast

import pandas as pd  # 引入 pandas 库

# 导入数据模型相关的类和函数
from graphrag.model import CommunityReport, Covariate, Entity, Relationship, TextUnit
# 导入读取函数
from graphrag.query.input.loaders.dfs import (
    read_community_reports,
    read_covariates,
    read_entities,
    read_relationships,
    read_text_units,
)


def read_indexer_text_units(final_text_units: pd.DataFrame) -> list[TextUnit]:
    """Read in the Text Units from the raw indexing outputs."""
    return read_text_units(
        df=final_text_units,
        short_id_col=None,
        # 期望一个类型到 ID 的 Covariate 映射
        covariates_col=None,
    )


def read_indexer_covariates(final_covariates: pd.DataFrame) -> list[Covariate]:
    """Read in the Claims from the raw indexing outputs."""
    covariate_df = final_covariates
    covariate_df["id"] = covariate_df["id"].astype(str)
    return read_covariates(
        df=covariate_df,
        short_id_col="human_readable_id",
        attributes_cols=[
            "object_id",
            "status",
            "start_date",
            "end_date",
            "description",
        ],
        text_unit_ids_col=None,
    )


def read_indexer_relationships(final_relationships: pd.DataFrame) -> list[Relationship]:
    """Read in the Relationships from the raw indexing outputs."""
    return read_relationships(
        df=final_relationships,
        short_id_col="human_readable_id",
        description_embedding_col=None,
        document_ids_col=None,
        attributes_cols=["rank"],
    )


def read_indexer_reports(
    final_community_reports: pd.DataFrame,
    final_nodes: pd.DataFrame,
    community_level: int,
) -> list[CommunityReport]:
    """Read in the Community Reports from the raw indexing outputs."""
    report_df = final_community_reports
    entity_df = final_nodes

    # 根据 community_level 过滤 entity_df 中低于指定 community_level 的条目
    entity_df = _filter_under_community_level(entity_df, community_level)
    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)

    # 按 title 分组并取 community 最大值作为新的 community_df
    entity_df = entity_df.groupby(["title"]).agg({"community": "max"}).reset_index()
    entity_df["community"] = entity_df["community"].astype(str)
    filtered_community_df = entity_df["community"].drop_duplicates()

    # 根据 community_level 过滤 report_df 中低于指定 community_level 的条目
    report_df = _filter_under_community_level(report_df, community_level)
    # 内连接合并 report_df 和 filtered_community_df，基于 community 列
    report_df = report_df.merge(filtered_community_df, on="community", how="inner")

    # 读取符合条件的社区报告数据
    return read_community_reports(
        df=report_df,
        id_col="community",
        short_id_col="community",
        summary_embedding_col=None,
        content_embedding_col=None,
    )


def read_indexer_entities(
    final_nodes: pd.DataFrame,
    final_entities: pd.DataFrame,
):
    """Read in the Entities from the raw indexing outputs."""
    # 此函数目前未定义
    community_level: int,



# 声明一个变量 community_level，类型为整数 (int)
# 从原始索引输出中读取实体列表
def read_entities_from_indexing_outputs(
    final_nodes: list[Entity],
    final_entities: pd.DataFrame,
    community_level: int,
) -> list[Entity]:
    """Read in the Entities from the raw indexing outputs."""
    
    # 使用最终节点数据作为实体数据框
    entity_df = final_nodes
    # 使用最终实体数据作为实体嵌入数据框
    entity_embedding_df = final_entities

    # 根据社区级别过滤实体数据框
    entity_df = _filter_under_community_level(entity_df, community_level)
    
    # 选择并重命名实体数据框的列
    entity_df = cast(pd.DataFrame, entity_df[["title", "degree", "community"]]).rename(
        columns={"title": "name", "degree": "rank"}
    )

    # 填充并转换社区和等级列的数据类型为整数
    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)
    entity_df["rank"] = entity_df["rank"].astype(int)

    # 对于重复的实体，保留社区级别最高的一个
    entity_df = (
        entity_df.groupby(["name", "rank"]).agg({"community": "max"}).reset_index()
    )
    # 将社区列转换为字符串列表格式
    entity_df["community"] = entity_df["community"].apply(lambda x: [str(x)])
    # 合并实体数据框和实体嵌入数据框，保留唯一值
    entity_df = entity_df.merge(
        entity_embedding_df, on="name", how="inner"
    ).drop_duplicates(subset=["name"])

    # 将实体数据框读取为知识模型对象
    return read_entities(
        df=entity_df,
        id_col="id",
        title_col="name",
        type_col="type",
        short_id_col="human_readable_id",
        description_col="description",
        community_col="community",
        rank_col="rank",
        name_embedding_col=None,
        description_embedding_col="description_embedding",
        graph_embedding_col=None,
        text_unit_ids_col="text_unit_ids",
        document_ids_col=None,
    )


# 根据社区级别过滤数据框中的数据
def _filter_under_community_level(
    df: pd.DataFrame, community_level: int
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        df[df.level <= community_level],
    )
```