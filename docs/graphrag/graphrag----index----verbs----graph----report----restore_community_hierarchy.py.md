# `.\graphrag\graphrag\index\verbs\graph\report\restore_community_hierarchy.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_graph, _get_node_attributes, _get_edge_attributes and _get_attribute_column_mapping methods definition."""

import logging  # 导入日志模块
from typing import cast  # 引入类型转换

import pandas as pd  # 导入 pandas 库
from datashaper import TableContainer, VerbInput, verb  # 从 datashaper 库中导入 TableContainer、VerbInput 和 verb

import graphrag.index.graph.extractors.community_reports.schemas as schemas  # 导入 community_reports.schemas 模块

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@verb(name="restore_community_hierarchy")  # 声明一个名为 restore_community_hierarchy 的装饰器函数
def restore_community_hierarchy(
    input: VerbInput,  # 输入参数为 VerbInput 类型
    name_column: str = schemas.NODE_NAME,  # 名称列，默认为 schemas 模块中的 NODE_NAME 值
    community_column: str = schemas.NODE_COMMUNITY,  # 社区列，默认为 schemas 模块中的 NODE_COMMUNITY 值
    level_column: str = schemas.NODE_LEVEL,  # 层级列，默认为 schemas 模块中的 NODE_LEVEL 值
    **_kwargs,  # 其它关键字参数
) -> TableContainer:  # 返回值为 TableContainer 类型
    """Restore the community hierarchy from the node data."""
    node_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())  # 强制类型转换为 pd.DataFrame，并获取输入数据
    community_df = (
        node_df.groupby([community_column, level_column])  # 按社区列和层级列分组
        .agg({name_column: list})  # 聚合名称列为列表
        .reset_index()  # 重置索引
    )
    community_levels = {}  # 初始化空字典用于存储社区层级关系

    for _, row in community_df.iterrows():  # 遍历社区数据框的每一行
        level = row[level_column]  # 获取层级值
        name = row[name_column]  # 获取名称列表
        community = row[community_column]  # 获取社区值

        if community_levels.get(level) is None:  # 如果当前层级的社区字典为空
            community_levels[level] = {}  # 初始化当前层级的社区字典
        community_levels[level][community] = name  # 将社区和名称列表存入对应层级的字典中

    # 获取唯一的层级列表，并按升序排序
    levels = sorted(community_levels.keys())

    community_hierarchy = []  # 初始化空列表，用于存储社区层级关系

    for idx in range(len(levels) - 1):  # 遍历层级列表，注意最后一个层级不需要处理
        level = levels[idx]  # 当前层级
        log.debug("Level: %s", level)  # 记录调试信息，显示当前层级
        next_level = levels[idx + 1]  # 下一个层级
        current_level_communities = community_levels[level]  # 当前层级的社区字典
        next_level_communities = community_levels[next_level]  # 下一个层级的社区字典
        log.debug(
            "Number of communities at level %s: %s",
            level,
            len(current_level_communities),
        )  # 记录调试信息，显示当前层级的社区数量

        for current_community in current_level_communities:  # 遍历当前层级的每个社区
            current_entities = current_level_communities[current_community]  # 当前社区的实体列表

            # 遍历下一个层级的每个社区，找到所有的子社区
            entities_found = 0
            for next_level_community in next_level_communities:
                next_entities = next_level_communities[next_level_community]
                if set(next_entities).issubset(set(current_entities)):  # 如果下一个社区的实体是当前社区实体的子集
                    community_hierarchy.append({
                        community_column: current_community,
                        schemas.COMMUNITY_LEVEL: level,
                        schemas.SUB_COMMUNITY: next_level_community,
                        schemas.SUB_COMMUNITY_SIZE: len(next_entities),
                    })  # 将社区层级关系添加到列表中

                    entities_found += len(next_entities)
                    if entities_found == len(current_entities):
                        break  # 如果找到了所有实体，退出内循环

    return TableContainer(table=pd.DataFrame(community_hierarchy))  # 返回社区层级关系的数据框封装
```