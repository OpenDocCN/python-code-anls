# `.\graphrag\graphrag\query\input\loaders\dfs.py`

```py
# 加载 pandas 库，用于处理数据帧
import pandas as pd

# 导入图数据模型相关类
from graphrag.model import (
    Community,
    CommunityReport,
    Covariate,
    Document,
    Entity,
    Relationship,
    TextUnit,
)

# 导入数据加载工具函数
from graphrag.query.input.loaders.utils import (
    to_list,
    to_optional_dict,
    to_optional_float,
    to_optional_int,
    to_optional_list,
    to_optional_str,
    to_str,
)

# 导入向量存储相关模块
from graphrag.vector_stores import BaseVectorStore, VectorStoreDocument

# 定义函数 read_entities，从数据帧中读取实体信息并返回实体对象列表
def read_entities(
    df: pd.DataFrame,  # 输入参数 df，数据帧对象
    id_col: str = "id",  # 实体标识列，默认为 "id"
    short_id_col: str | None = "short_id",  # 短标识列，可选，默认为 "short_id"
    title_col: str = "title",  # 标题列，默认为 "title"
    type_col: str | None = "type",  # 类型列，可选，默认为 "type"
    description_col: str | None = "description",  # 描述列，可选，默认为 "description"
    name_embedding_col: str | None = "name_embedding",  # 名称嵌入列，可选，默认为 "name_embedding"
    description_embedding_col: str | None = "description_embedding",  # 描述嵌入列，可选，默认为 "description_embedding"
    graph_embedding_col: str | None = "graph_embedding",  # 图嵌入列，可选，默认为 "graph_embedding"
    community_col: str | None = "community_ids",  # 社区标识列，可选，默认为 "community_ids"
    text_unit_ids_col: str | None = "text_unit_ids",  # 文本单元标识列，可选，默认为 "text_unit_ids"
    document_ids_col: str | None = "document_ids",  # 文档标识列，可选，默认为 "document_ids"
    rank_col: str | None = "degree",  # 排名列，可选，默认为 "degree"
    attributes_cols: list[str] | None = None,  # 属性列列表，可选，默认为空列表
) -> list[Entity]:
    """从数据帧中读取实体信息并返回实体对象列表."""
    entities = []
    # 遍历数据帧中的每一行
    for idx, row in df.iterrows():
        # 创建 Entity 对象，并根据数据帧中的列填充属性值
        entity = Entity(
            id=to_str(row, id_col),  # 实体标识，转换为字符串
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 短标识，可选
            title=to_str(row, title_col),  # 标题，转换为字符串
            type=to_optional_str(row, type_col),  # 类型，可选
            description=to_optional_str(row, description_col),  # 描述，可选
            name_embedding=to_optional_list(row, name_embedding_col, item_type=float),  # 名称嵌入，可选
            description_embedding=to_optional_list(
                row, description_embedding_col, item_type=float
            ),  # 描述嵌入，可选
            graph_embedding=to_optional_list(row, graph_embedding_col, item_type=float),  # 图嵌入，可选
            community_ids=to_optional_list(row, community_col, item_type=str),  # 社区标识，可选
            text_unit_ids=to_optional_list(row, text_unit_ids_col),  # 文本单元标识，可选
            document_ids=to_optional_list(row, document_ids_col),  # 文档标识，可选
            rank=to_optional_int(row, rank_col),  # 排名，可选
            attributes=(
                {col: row.get(col) for col in attributes_cols}  # 属性字典，如果属性列列表存在则创建字典
                if attributes_cols
                else None
            ),
        )
        entities.append(entity)  # 将创建的实体对象添加到列表中
    return entities  # 返回实体对象列表


# 定义函数 store_entity_semantic_embeddings，将实体的语义嵌入存储到向量存储中
def store_entity_semantic_embeddings(
    entities: list[Entity],  # 输入参数 entities，实体对象列表
    vectorstore: BaseVectorStore,  # 输入参数 vectorstore，基础向量存储对象
) -> BaseVectorStore:
    """将实体的语义嵌入存储到向量存储中."""
    # 创建文档对象列表，每个实体对应一个文档对象
    documents = [
        VectorStoreDocument(
            id=entity.id,  # 文档标识为实体的标识
            text=entity.description,  # 文本为实体的描述
            vector=entity.description_embedding,  # 向量为实体的描述嵌入
            attributes=(
                {"title": entity.title, **entity.attributes}  # 属性字典包括标题和其他属性
                if entity.attributes
                else {"title": entity.title}  # 如果没有其他属性，则只包括标题
            ),
        )
        for entity in entities  # 遍历输入的实体列表
    ]
    # 载入文档到向量存储器中
    vectorstore.load_documents(documents=documents)
    # 返回载入文档后的向量存储器对象
    return vectorstore
# 将实体行为嵌入存储到向量存储中
def store_entity_behavior_embeddings(
    entities: list[Entity],                 # 输入参数：实体对象列表
    vectorstore: BaseVectorStore,           # 输入参数：向量存储对象

) -> BaseVectorStore:                      # 返回类型：向量存储对象

    """Store entity behavior embeddings in a vectorstore."""
    # 根据实体对象列表创建文档对象列表
    documents = [
        VectorStoreDocument(
            id=entity.id,                   # 实体ID
            text=entity.description,        # 实体描述文本
            vector=entity.graph_embedding,  # 实体图嵌入向量
            attributes=(                    # 实体属性字典
                {"title": entity.title, **entity.attributes}
                if entity.attributes
                else {"title": entity.title}
            ),
        )
        for entity in entities            # 遍历实体对象列表
    ]
    # 载入文档到向量存储
    vectorstore.load_documents(documents=documents)
    # 返回更新后的向量存储对象
    return vectorstore


# 从数据框中读取关系数据并返回关系对象列表
def read_relationships(
    df: pd.DataFrame,                       # 输入参数：数据框
    id_col: str = "id",                     # 关系ID列，默认为"id"
    short_id_col: str | None = "short_id",  # 短ID列，可选，默认为"short_id"
    source_col: str = "source",             # 源列，默认为"source"
    target_col: str = "target",             # 目标列，默认为"target"
    description_col: str | None = "description",  # 描述列，可选，默认为"description"
    description_embedding_col: str | None = "description_embedding",  # 描述嵌入列，可选，默认为"description_embedding"
    weight_col: str | None = "weight",      # 权重列，可选，默认为"weight"
    text_unit_ids_col: str | None = "text_unit_ids",  # 文本单元ID列，可选，默认为"text_unit_ids"
    document_ids_col: str | None = "document_ids",    # 文档ID列，可选，默认为"document_ids"
    attributes_cols: list[str] | None = None,          # 属性列列表，可选，默认为None
) -> list[Relationship]:                    # 返回类型：关系对象列表

    """Read relationships from a dataframe."""
    relationships = []
    # 遍历数据框的行索引和每一行数据
    for idx, row in df.iterrows():
        # 创建关系对象
        rel = Relationship(
            id=to_str(row, id_col),                         # 关系ID
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 短ID
            source=to_str(row, source_col),                 # 源
            target=to_str(row, target_col),                 # 目标
            description=to_optional_str(row, description_col),  # 描述
            description_embedding=to_optional_list(         # 描述嵌入向量列表
                row, description_embedding_col, item_type=float
            ),
            weight=to_optional_float(row, weight_col),      # 权重
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),  # 文本单元ID列表
            document_ids=to_optional_list(row, document_ids_col, item_type=str),    # 文档ID列表
            attributes=(                                    # 属性字典
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        relationships.append(rel)  # 将关系对象添加到列表中
    return relationships          # 返回关系对象列表


# 从数据框中读取协变量数据并返回协变量对象列表
def read_covariates(
    df: pd.DataFrame,                       # 输入参数：数据框
    id_col: str = "id",                     # 协变量ID列，默认为"id"
    short_id_col: str | None = "short_id",  # 短ID列，可选，默认为"short_id"
    subject_col: str = "subject_id",        # 主题ID列，默认为"subject_id"
    subject_type_col: str | None = "subject_type",   # 主题类型列，可选，默认为"subject_type"
    covariate_type_col: str | None = "covariate_type",  # 协变量类型列，可选，默认为"covariate_type"
    text_unit_ids_col: str | None = "text_unit_ids",     # 文本单元ID列，可选，默认为"text_unit_ids"
    document_ids_col: str | None = "document_ids",       # 文档ID列，可选，默认为"document_ids"
    attributes_cols: list[str] | None = None,            # 属性列列表，可选，默认为None
) -> list[Covariate]:                      # 返回类型：协变量对象列表

    """Read covariates from a dataframe."""
    covariates = []
    # 遍历数据框的行索引和每一行数据
    for idx, row in df.iterrows():
        # 创建协变量对象
        covar = Covariate(
            id=to_str(row, id_col),                         # 协变量ID
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 短ID
            subject_id=to_str(row, subject_col),            # 主题ID
            subject_type=to_optional_str(row, subject_type_col),  # 主题类型
            covariate_type=to_optional_str(row, covariate_type_col),  # 协变量类型
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),  # 文本单元ID列表
            document_ids=to_optional_list(row, document_ids_col, item_type=str),    # 文档ID列表
            attributes=(                                    # 属性字典
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        covariates.append(covar)  # 将协变量对象添加到列表中
    return covariates            # 返回协变量对象列表
    # 遍历数据框的每一行及其索引
    for idx, row in df.iterrows():
        # 创建 Covariate 对象，从数据行中提取属性值，并根据需要转换为指定类型
        cov = Covariate(
            id=to_str(row, id_col),  # 提取并转换 ID 列的值为字符串
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 如果有短 ID 列，则转换为可选字符串；否则使用索引转换为字符串
            subject_id=to_str(row, subject_col),  # 提取并转换主体 ID 列的值为字符串
            subject_type=(
                to_str(row, subject_type_col) if subject_type_col else "entity"  # 如果有主体类型列，则转换为字符串；否则默认为 "entity"
            ),
            covariate_type=(
                to_str(row, covariate_type_col) if covariate_type_col else "claim"  # 如果有协变量类型列，则转换为字符串；否则默认为 "claim"
            ),
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),  # 提取并转换文本单元 ID 列的值为可选字符串列表
            document_ids=to_optional_list(row, document_ids_col, item_type=str),  # 提取并转换文档 ID 列的值为可选字符串列表
            attributes=(
                {col: row.get(col) for col in attributes_cols}  # 如果有属性列，则创建属性字典，每列作为键，对应值为行中的值
                if attributes_cols  # 如果存在属性列
                else None  # 否则属性设为 None
            ),
        )
        # 将创建的 Covariate 对象添加到 covariates 列表中
        covariates.append(cov)
    # 返回填充完毕的 covariates 列表
    return covariates
# 从给定的 DataFrame 中读取社群信息并返回社群对象的列表
def read_communities(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    title_col: str = "title",
    level_col: str = "level",
    entities_col: str | None = "entity_ids",
    relationships_col: str | None = "relationship_ids",
    covariates_col: str | None = "covariate_ids",
    attributes_cols: list[str] | None = None,
) -> list[Community]:
    """Read communities from a dataframe."""
    communities = []
    # 遍历 DataFrame 的每一行
    for idx, row in df.iterrows():
        # 根据行数据创建 Community 对象，并加入到 communities 列表中
        comm = Community(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            level=to_str(row, level_col),
            entity_ids=to_optional_list(row, entities_col, item_type=str),
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        communities.append(comm)
    return communities


# 从给定的 DataFrame 中读取社群报告信息并返回社群报告对象的列表
def read_community_reports(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    title_col: str = "title",
    community_col: str = "community",
    summary_col: str = "summary",
    content_col: str = "full_content",
    rank_col: str | None = "rank",
    summary_embedding_col: str | None = "summary_embedding",
    content_embedding_col: str | None = "full_content_embedding",
    attributes_cols: list[str] | None = None,
) -> list[CommunityReport]:
    """Read community reports from a dataframe."""
    reports = []
    # 遍历 DataFrame 的每一行
    for idx, row in df.iterrows():
        # 根据行数据创建 CommunityReport 对象，并加入到 reports 列表中
        report = CommunityReport(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            community_id=to_str(row, community_col),
            summary=to_str(row, summary_col),
            full_content=to_str(row, content_col),
            rank=to_optional_float(row, rank_col),
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),
            full_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        reports.append(report)
    return reports


# 从给定的 DataFrame 中读取文本单元信息并返回文本单元对象的列表
def read_text_units(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str | None = "short_id",
    text_col: str = "text",
    entities_col: str | None = "entity_ids",
    relationships_col: str | None = "relationship_ids",
):
    """
    Read text units from a dataframe.
    Note: This function definition is incomplete in the provided snippet.
    """
    # 这里的函数体未完全给出，无法完整注释其功能
    pass  # 占位符，因为函数体未完全给出，暂无法提供完整的注释
    covariates_col: str | None = "covariate_ids",
    # 定义变量 covariates_col，用于指定协变量列的名称，默认为 "covariate_ids"，可以是字符串或者空值

    tokens_col: str | None = "n_tokens",
    # 定义变量 tokens_col，用于指定标记列的名称，默认为 "n_tokens"，可以是字符串或者空值

    document_ids_col: str | None = "document_ids",
    # 定义变量 document_ids_col，用于指定文档ID列的名称，默认为 "document_ids"，可以是字符串或者空值

    embedding_col: str | None = "text_embedding",
    # 定义变量 embedding_col，用于指定嵌入列的名称，默认为 "text_embedding"，可以是字符串或者空值

    attributes_cols: list[str] | None = None,
    # 定义变量 attributes_cols，用于指定属性列的列表，默认为空列表，可以是字符串列表或者空值
# 从数据框中读取文本单元并返回列表
def read_text_units(df: pd.DataFrame) -> list[TextUnit]:
    """Read text units from a dataframe."""
    text_units = []
    # 遍历数据框中的每一行
    for idx, row in df.iterrows():
        # 创建文本单元对象
        chunk = TextUnit(
            id=to_str(row, id_col),  # 提取并转换id列数据为字符串
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 提取并转换short_id列数据为可选字符串或者使用索引
            text=to_str(row, text_col),  # 提取并转换text_col列数据为字符串
            entity_ids=to_optional_list(row, entities_col, item_type=str),  # 提取并转换entities_col列数据为可选字符串列表
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),  # 提取并转换relationships_col列数据为可选字符串列表
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),  # 提取并转换covariates_col列数据为可选键为字符串值为字符串的字典
            text_embedding=to_optional_list(row, embedding_col, item_type=float),  # type: ignore  # 提取并转换embedding_col列数据为可选浮点数列表
            n_tokens=to_optional_int(row, tokens_col),  # 提取并转换tokens_col列数据为可选整数
            document_ids=to_optional_list(row, document_ids_col, item_type=str),  # 提取并转换document_ids_col列数据为可选字符串列表
            attributes=(
                {col: row.get(col) for col in attributes_cols}  # 如果attributes_cols不为None，则提取每个列的数据并形成字典
                if attributes_cols
                else None
            ),
        )
        text_units.append(chunk)  # 将创建的文本单元添加到列表中
    return text_units  # 返回文本单元列表


# 从数据框中读取文档并返回列表
def read_documents(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str = "short_id",
    title_col: str = "title",
    type_col: str = "type",
    summary_col: str | None = "entities",
    raw_content_col: str | None = "relationships",
    summary_embedding_col: str | None = "summary_embedding",
    content_embedding_col: str | None = "raw_content_embedding",
    text_units_col: str | None = "text_units",
    attributes_cols: list[str] | None = None,
) -> list[Document]:
    """Read documents from a dataframe."""
    docs = []
    # 遍历数据框中的每一行
    for idx, row in df.iterrows():
        # 创建文档对象
        doc = Document(
            id=to_str(row, id_col),  # 提取并转换id_col列数据为字符串
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),  # 提取并转换short_id_col列数据为可选字符串或者使用索引
            title=to_str(row, title_col),  # 提取并转换title_col列数据为字符串
            type=to_str(row, type_col),  # 提取并转换type_col列数据为字符串
            summary=to_optional_str(row, summary_col),  # 提取并转换summary_col列数据为可选字符串
            raw_content=to_str(row, raw_content_col),  # 提取并转换raw_content_col列数据为字符串
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),  # 提取并转换summary_embedding_col列数据为可选浮点数列表
            raw_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),  # 提取并转换content_embedding_col列数据为可选浮点数列表
            text_units=to_list(row, text_units_col, item_type=str),  # type: ignore  # 提取并转换text_units_col列数据为列表形式的字符串
            attributes=(
                {col: row.get(col) for col in attributes_cols}  # 如果attributes_cols不为None，则提取每个列的数据并形成字典
                if attributes_cols
                else None
            ),
        )
        docs.append(doc)  # 将创建的文档对象添加到列表中
    return docs  # 返回文档对象列表
```