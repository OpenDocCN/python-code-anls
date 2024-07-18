# `.\graphrag\graphrag\index\graph\extractors\community_reports\sort_context.py`

```py
# 版权声明和许可声明
# 版权所有 (c) 2024 Microsoft Corporation.
# 根据 MIT 许可证授权

"""按照度数降序对上下文进行排序。"""

# 导入 pandas 库
import pandas as pd

# 导入特定路径的模块
import graphrag.index.graph.extractors.community_reports.schemas as schemas
# 导入 num_tokens 函数
from graphrag.query.llm.text_utils import num_tokens


def sort_context(
    local_context: list[dict],  # 本地上下文的列表，每个元素是一个字典
    sub_community_reports: list[dict] | None = None,  # 子社区报告的列表，可选参数，默认为 None
    max_tokens: int | None = None,  # 最大 tokens 数量，可选参数，默认为 None
    node_id_column: str = schemas.NODE_ID,  # 节点 ID 列的名称，默认为 schemas 模块中的 NODE_ID
    node_name_column: str = schemas.NODE_NAME,  # 节点名称列的名称，默认为 schemas 模块中的 NODE_NAME
    node_details_column: str = schemas.NODE_DETAILS,  # 节点详情列的名称，默认为 schemas 模块中的 NODE_DETAILS
    edge_id_column: str = schemas.EDGE_ID,  # 边 ID 列的名称，默认为 schemas 模块中的 EDGE_ID
    edge_details_column: str = schemas.EDGE_DETAILS,  # 边详情列的名称，默认为 schemas 模块中的 EDGE_DETAILS
    edge_degree_column: str = schemas.EDGE_DEGREE,  # 边度数列的名称，默认为 schemas 模块中的 EDGE_DEGREE
    edge_source_column: str = schemas.EDGE_SOURCE,  # 边源节点列的名称，默认为 schemas 模块中的 EDGE_SOURCE
    edge_target_column: str = schemas.EDGE_TARGET,  # 边目标节点列的名称，默认为 schemas 模块中的 EDGE_TARGET
    claim_id_column: str = schemas.CLAIM_ID,  # 主张 ID 列的名称，默认为 schemas 模块中的 CLAIM_ID
    claim_details_column: str = schemas.CLAIM_DETAILS,  # 主张详情列的名称，默认为 schemas 模块中的 CLAIM_DETAILS
    community_id_column: str = schemas.COMMUNITY_ID,  # 社区 ID 列的名称，默认为 schemas 模块中的 COMMUNITY_ID
) -> str:
    """按照度数降序对上下文进行排序。

    如果提供了 max tokens 参数，将返回符合 token 限制的上下文字符串。
    """

    def _get_context_string(
        entities: list[dict],  # 实体列表，每个元素是一个字典
        edges: list[dict],  # 边列表，每个元素是一个字典
        claims: list[dict],  # 主张列表，每个元素是一个字典
        sub_community_reports: list[dict] | None = None,  # 子社区报告的列表，可选参数，默认为 None
        max_tokens: int | None = None,  # 最大 tokens 数量，可选参数，默认为 None
    ) -> str:
        """根据提供的实体、边和主张，生成表示上下文的字符串。

        如果提供了 sub_community_reports 参数，将考虑其内容。
        如果提供了 max_tokens 参数，将限制字符串的长度。
        """
    ) -> str:
        """将结构化数据连接成上下文字符串。"""
        # 初始化一个空列表用于存储上下文信息
        contexts = []

        # 处理子社区报告数据
        if sub_community_reports:
            # 筛选有效的子社区报告，确保社区 ID 列存在且不为空白
            sub_community_reports = [
                report
                for report in sub_community_reports
                if community_id_column in report
                and report[community_id_column]
                and str(report[community_id_column]).strip() != ""
            ]
            # 将筛选后的报告数据转换为 DataFrame，并去重
            report_df = pd.DataFrame(sub_community_reports).drop_duplicates()
            # 如果 DataFrame 不为空
            if not report_df.empty:
                # 如果社区 ID 列的数据类型为浮点数，则转换为整数类型
                if report_df[community_id_column].dtype == float:
                    report_df[community_id_column] = report_df[
                        community_id_column
                    ].astype(int)
                # 生成报告字符串，并添加到上下文列表中
                report_string = (
                    f"----Reports-----\n{report_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(report_string)

        # 处理实体数据
        entities = [
            entity
            for entity in entities
            if node_id_column in entity
            and entity[node_id_column]
            and str(entity[node_id_column]).strip() != ""
        ]
        # 将筛选后的实体数据转换为 DataFrame，并去重
        entity_df = pd.DataFrame(entities).drop_duplicates()
        # 如果 DataFrame 不为空
        if not entity_df.empty:
            # 如果节点 ID 列的数据类型为浮点数，则转换为整数类型
            if entity_df[node_id_column].dtype == float:
                entity_df[node_id_column] = entity_df[node_id_column].astype(int)
            # 生成实体字符串，并添加到上下文列表中
            entity_string = (
                f"-----Entities-----\n{entity_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(entity_string)

        # 处理声明数据
        if claims and len(claims) > 0:
            # 筛选有效的声明数据，确保声明 ID 列存在且不为空白
            claims = [
                claim
                for claim in claims
                if claim_id_column in claim
                and claim[claim_id_column]
                and str(claim[claim_id_column]).strip() != ""
            ]
            # 将筛选后的声明数据转换为 DataFrame，并去重
            claim_df = pd.DataFrame(claims).drop_duplicates()
            # 如果 DataFrame 不为空
            if not claim_df.empty:
                # 如果声明 ID 列的数据类型为浮点数，则转换为整数类型
                if claim_df[claim_id_column].dtype == float:
                    claim_df[claim_id_column] = claim_df[claim_id_column].astype(int)
                # 生成声明字符串，并添加到上下文列表中
                claim_string = (
                    f"-----Claims-----\n{claim_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(claim_string)

        # 处理边关系数据
        edges = [
            edge
            for edge in edges
            if edge_id_column in edge
            and edge[edge_id_column]
            and str(edge[edge_id_column]).strip() != ""
        ]
        # 将筛选后的边关系数据转换为 DataFrame，并去重
        edge_df = pd.DataFrame(edges).drop_duplicates()
        # 如果 DataFrame 不为空
        if not edge_df.empty:
            # 如果边 ID 列的数据类型为浮点数，则转换为整数类型
            if edge_df[edge_id_column].dtype == float:
                edge_df[edge_id_column] = edge_df[edge_id_column].astype(int)
            # 生成边关系字符串，并添加到上下文列表中
            edge_string = (
                f"-----Relationships-----\n{edge_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(edge_string)

        # 将所有上下文信息用双换行符连接成一个字符串并返回
        return "\n\n".join(contexts)

    # 根据节点的度数降序排列节点详情
    edges = []
    node_details = {}
    # 初始化空字典用于存储索赔详情
    claim_details = {}
    
    # 遍历本地上下文中的记录
    for record in local_context:
        # 获取节点名称
        node_name = record[node_name_column]
        # 获取边缘详情，如果存在则筛选非空元素
        record_edges = record.get(edge_details_column, [])
        record_edges = [e for e in record_edges if not pd.isna(e)]
        # 获取节点详情
        record_node_details = record[node_details_column]
        # 获取索赔详情，如果存在则筛选非空元素
        record_claims = record.get(claim_details_column, [])
        record_claims = [c for c in record_claims if not pd.isna(c)]
    
        # 将记录中的边缘详情扩展到全局列表中
        edges.extend(record_edges)
        # 将节点名称和节点详情映射存储到全局字典中
        node_details[node_name] = record_node_details
        # 将节点名称和索赔详情映射存储到索赔详情字典中
        claim_details[node_name] = record_claims
    
    # 筛选出列表中的字典类型边缘，并按照边缘度排序（降序）
    edges = [edge for edge in edges if isinstance(edge, dict)]
    edges = sorted(edges, key=lambda x: x[edge_degree_column], reverse=True)
    
    # 初始化排序后的边缘、节点和索赔列表，以及上下文字符串
    sorted_edges = []
    sorted_nodes = []
    sorted_claims = []
    context_string = ""
    
    # 遍历排序后的边缘列表
    for edge in edges:
        # 获取边缘源节点和目标节点的详情
        source_details = node_details.get(edge[edge_source_column], {})
        target_details = node_details.get(edge[edge_target_column], {})
        # 将源节点和目标节点的详情加入到排序后的节点列表中
        sorted_nodes.extend([source_details, target_details])
        # 将边缘加入到排序后的边缘列表中
        sorted_edges.append(edge)
        # 获取边缘源节点和目标节点的索赔详情，并加入到排序后的索赔列表中
        source_claims = claim_details.get(edge[edge_source_column], [])
        target_claims = claim_details.get(edge[edge_target_column], [])
        sorted_claims.extend(source_claims if source_claims else [])
        sorted_claims.extend(target_claims if target_claims else [])
        # 如果设置了最大标记数，则更新上下文字符串并检查是否超过最大标记数
        if max_tokens:
            new_context_string = _get_context_string(
                sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
            )
            if num_tokens(context_string) > max_tokens:
                break
            context_string = new_context_string
    
    # 如果上下文字符串为空，则生成新的上下文字符串
    if context_string == "":
        return _get_context_string(
            sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
        )
    
    # 返回已生成的上下文字符串
    return context_string
```