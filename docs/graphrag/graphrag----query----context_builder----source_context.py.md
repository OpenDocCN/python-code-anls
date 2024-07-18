# `.\graphrag\graphrag\query\context_builder\source_context.py`

```py
# 导入随机模块，用于数据洗牌
import random
# 引入类型提示相关的模块
from typing import Any, cast

# 导入 pandas 库，用于处理数据
import pandas as pd
# 导入 tiktoken 库，可能用于编码处理
import tiktoken

# 从 graphrag.model 中导入 Entity, Relationship, TextUnit 类
from graphrag.model import Entity, Relationship, TextUnit
# 从 graphrag.query.llm.text_utils 中导入 num_tokens 函数
from graphrag.query.llm.text_utils import num_tokens

# 定义一个文本单元上下文构建的实用函数
def build_text_unit_context(
    text_units: list[TextUnit],  # 文本单元的列表
    token_encoder: tiktoken.Encoding | None = None,  # 可选的编码器对象，默认为 None
    column_delimiter: str = "|",  # 列分隔符，默认为竖线 |
    shuffle_data: bool = True,  # 是否对数据进行洗牌，默认为 True
    max_tokens: int = 8000,  # 最大 token 数量，默认为 8000
    context_name: str = "Sources",  # 上下文名称，默认为 "Sources"
    random_state: int = 86,  # 随机数种子，默认为 86
) -> tuple[str, dict[str, pd.DataFrame]]:
    """准备文本单元数据表作为系统提示的上下文数据。"""
    # 如果文本单元为空，则返回空字符串和空字典
    if text_units is None or len(text_units) == 0:
        return ("", {})

    # 如果需要洗牌数据，则使用指定的随机种子进行洗牌
    if shuffle_data:
        random.seed(random_state)
        random.shuffle(text_units)

    # 添加上下文标题
    current_context_text = f"-----{context_name}-----" + "\n"

    # 添加表头
    header = ["id", "text"]
    # 获取文本单元属性列，如果存在的话
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    # 过滤掉表头已经包含的属性列
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    # 构建当前上下文文本
    current_context_text += column_delimiter.join(header) + "\n"
    # 计算当前上下文的 token 数量
    current_tokens = num_tokens(current_context_text, token_encoder)
    # 存储所有上下文记录的列表，包含表头
    all_context_records = [header]

    # 遍历文本单元列表，构建每条记录的文本和数据
    for unit in text_units:
        # 构建新的上下文记录
        new_context = [
            unit.short_id,
            unit.text,
            *[
                str(unit.attributes.get(field, "")) if unit.attributes else ""
                for field in attribute_cols
            ],
        ]
        # 将新记录转换为文本格式
        new_context_text = column_delimiter.join(new_context) + "\n"
        # 计算新记录的 token 数量
        new_tokens = num_tokens(new_context_text, token_encoder)

        # 如果添加新记录后超过最大 token 数量，则结束添加
        if current_tokens + new_tokens > max_tokens:
            break

        # 更新当前上下文文本和 token 数量
        current_context_text += new_context_text
        current_tokens += new_tokens
        # 将新记录添加到所有上下文记录列表中
        all_context_records.append(new_context)

    # 如果记录数大于1，则创建 pandas DataFrame 对象
    if len(all_context_records) > 1:
        record_df = pd.DataFrame(
            all_context_records[1:], columns=cast(Any, all_context_records[0])
        )
    else:
        record_df = pd.DataFrame()

    # 返回当前上下文文本和以上下文名称为 key 的 DataFrame 字典
    return current_context_text, {context_name.lower(): record_df}


# 定义一个函数，用于计算与文本单元相关的选择实体的关系数量
def count_relationships(
    text_unit: TextUnit,  # 文本单元对象
    entity: Entity,  # 实体对象
    relationships: dict[str, Relationship]  # 包含关系的字典
) -> int:
    """计算与选定实体相关联的文本单元的关系数量。"""
    # 初始化一个空的匹配关系列表
    matching_relationships = list[Relationship]()
    # 如果文本单元的关系标识为 None，则进行以下操作
    if text_unit.relationship_ids is None:
        # 获取所有与实体相关的关系
        entity_relationships = [
            rel
            for rel in relationships.values()  # 遍历所有关系
            if rel.source == entity.title or rel.target == entity.title  # 筛选与实体标题相关的关系
        ]
        # 进一步筛选出具有文本单元标识符的关系
        entity_relationships = [
            rel for rel in entity_relationships if rel.text_unit_ids
        ]
        # 找到包含文本单元标识符的关系，形成列表
        matching_relationships = [
            rel
            for rel in entity_relationships
            if text_unit.id in rel.text_unit_ids  # 检查文本单元标识符是否在关系的文本单元标识符列表中
        ]  # type: ignore
    else:
        # 根据文本单元的关系标识符获取相关的关系
        text_unit_relationships = [
            relationships[rel_id]
            for rel_id in text_unit.relationship_ids  # 遍历文本单元的关系标识符列表
            if rel_id in relationships  # 确保关系标识符存在于关系字典中
        ]
        # 筛选与实体标题相关的关系
        matching_relationships = [
            rel
            for rel in text_unit_relationships
            if rel.source == entity.title or rel.target == entity.title  # 检查关系的源或目标是否与实体标题匹配
        ]
    # 返回匹配关系的数量
    return len(matching_relationships)
```