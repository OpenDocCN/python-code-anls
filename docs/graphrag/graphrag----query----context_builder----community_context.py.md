# `.\graphrag\graphrag\query\context_builder\community_context.py`

```py
# 导入日志记录模块
import logging
# 导入随机数生成模块
import random
# 导入类型提示模块
from typing import Any, cast

# 导入 pandas 库并使用别名 pd
import pandas as pd
# 导入 tiktoken 模块
import tiktoken

# 从 graphrag.model 模块导入 CommunityReport 和 Entity 类
from graphrag.model import CommunityReport, Entity
# 从 graphrag.query.llm.text_utils 模块导入 num_tokens 函数

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 定义函数 build_community_context，准备社区报告数据表作为系统提示的上下文数据
def build_community_context(
    community_reports: list[CommunityReport],  # 社区报告对象的列表
    entities: list[Entity] | None = None,  # 实体对象的列表，可选，默认为 None
    token_encoder: tiktoken.Encoding | None = None,  # TikToken 编码对象，可选，默认为 None
    use_community_summary: bool = True,  # 是否使用社区摘要，布尔值，默认为 True
    column_delimiter: str = "|",  # 列分隔符，默认为竖线 "|"
    shuffle_data: bool = True,  # 是否对数据进行洗牌，布尔值，默认为 True
    include_community_rank: bool = False,  # 是否包括社区排名，布尔值，默认为 False
    min_community_rank: int = 0,  # 最小社区排名，整数，默认为 0
    community_rank_name: str = "rank",  # 社区排名名称，字符串，默认为 "rank"
    include_community_weight: bool = True,  # 是否包括社区权重，布尔值，默认为 True
    community_weight_name: str = "occurrence weight",  # 社区权重名称，字符串，默认为 "occurrence weight"
    normalize_community_weight: bool = True,  # 是否对社区权重进行归一化，布尔值，默认为 True
    max_tokens: int = 8000,  # 最大 tokens 数量，整数，默认为 8000
    single_batch: bool = True,  # 是否使用单批次处理，布尔值，默认为 True
    context_name: str = "Reports",  # 上下文名称，字符串，默认为 "Reports"
    random_state: int = 86,  # 随机数种子，整数，默认为 86
) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
    """
    准备社区报告数据表作为系统提示的上下文数据。

    如果提供了实体信息，则社区权重将被计算为与社区内实体相关联的文本单位计数。

    计算的权重将添加为社区报告的属性，并添加到上下文数据表中。
    """
    # 如果提供了实体信息，且社区报告列表不为空，且需要包括社区权重，
    # 并且社区报告的第一个报告的属性为空或者不包含社区权重属性时
    if (
        entities
        and len(community_reports) > 0
        and include_community_weight
        and (
            community_reports[0].attributes is None
            or community_weight_name not in community_reports[0].attributes
        )
    ):
        # 记录日志信息：计算社区权重
        log.info("Computing community weights...")
        # 调用内部函数 _compute_community_weights 计算社区权重
        community_reports = _compute_community_weights(
            community_reports=community_reports,
            entities=entities,
            weight_attribute=community_weight_name,
            normalize=normalize_community_weight,
        )

    # 选择符合条件的社区报告，即排名大于等于 min_community_rank 的报告
    selected_reports = [
        report
        for report in community_reports
        if report.rank and report.rank >= min_community_rank
    ]
    # 如果没有符合条件的报告或者选中报告列表为空，则返回空列表和空字典
    if selected_reports is None or len(selected_reports) == 0:
        return ([], {})

    # 如果需要对数据进行洗牌
    if shuffle_data:
        # 使用给定的随机数种子进行随机化
        random.seed(random_state)
        random.shuffle(selected_reports)

    # 添加上下文数据表的标题行
    current_context_text = f"-----{context_name}-----" + "\n"

    # 添加表头
    header = ["id", "title"]
    # 获取属性列名列表，如果社区报告包含属性信息
    attribute_cols = (
        list(selected_reports[0].attributes.keys())
        if selected_reports[0].attributes
        else []
    )
    # 如果不需要包括社区权重，则从属性列名列表中移除社区权重列
    if not include_community_weight:
        attribute_cols = [col for col in attribute_cols if col != community_weight_name]
    # 将属性列名添加到表头中
    header.extend(attribute_cols)
    # 添加摘要列或内容列到表头中
    header.append("summary" if use_community_summary else "content")
    # 如果需要包括社区排名，则添加社区排名列到表头中
    if include_community_rank:
        header.append(community_rank_name)

    # 将表头以指定的列分隔符连接，并添加到当前上下文文本中
    current_context_text += column_delimiter.join(header) + "\n"
    # 计算当前上下文文本的令牌数
    current_tokens = num_tokens(current_context_text, token_encoder)
    # 初始化当前上下文记录列表，包含表头
    current_context_records = [header]
    # 初始化所有上下文文本列表
    all_context_text = []
    # 初始化所有上下文记录列表
    all_context_records = []

    # 遍历选定的报告列表
    for report in selected_reports:
        # 创建新的上下文信息列表
        new_context = [
            report.short_id,  # 报告的短ID
            report.title,  # 报告的标题
            *[
                str(report.attributes.get(field, "")) if report.attributes else ""
                for field in attribute_cols
            ],  # 报告的属性信息
        ]
        # 根据使用社区摘要的选择，添加报告的摘要或完整内容到新上下文信息列表
        new_context.append(
            report.summary if use_community_summary else report.full_content
        )
        # 如果需要包含社区排名，将排名转换为字符串并添加到新上下文信息列表
        if include_community_rank:
            new_context.append(str(report.rank))
        # 将新上下文信息列表转换为文本形式
        new_context_text = column_delimiter.join(new_context) + "\n"

        # 计算新上下文文本的令牌数
        new_tokens = num_tokens(new_context_text, token_encoder)
        # 检查是否添加新上下文会导致超过最大令牌数限制
        if current_tokens + new_tokens > max_tokens:
            # 如果当前上下文记录超过表头，则转换为 pandas 数据框，并根据权重和排名（如果存在）进行排序
            if len(current_context_records) > 1:
                record_df = _convert_report_context_to_df(
                    context_records=current_context_records[1:],
                    header=current_context_records[0],
                    weight_column=community_weight_name
                    if entities and include_community_weight
                    else None,
                    rank_column=community_rank_name if include_community_rank else None,
                )
            else:
                # 否则创建空的 pandas 数据框
                record_df = pd.DataFrame()
            # 将当前上下文记录转换为 CSV 格式的文本
            current_context_text = record_df.to_csv(index=False, sep=column_delimiter)

            # 如果是单批次处理，返回当前上下文文本和以小写形式命名的记录数据框字典
            if single_batch:
                return current_context_text, {context_name.lower(): record_df}

            # 将当前上下文文本和记录数据框添加到所有上下文文本和记录列表中
            all_context_text.append(current_context_text)
            all_context_records.append(record_df)

            # 开始一个新的批次，重新初始化当前上下文文本和记录列表
            current_context_text = (
                f"-----{context_name}-----"
                + "\n"
                + column_delimiter.join(header)
                + "\n"
            )
            current_tokens = num_tokens(current_context_text, token_encoder)
            current_context_records = [header]
        else:
            # 如果未超过最大令牌数限制，将新上下文文本和记录添加到当前上下文文本和记录列表中
            current_context_text += new_context_text
            current_tokens += new_tokens
            current_context_records.append(new_context)

    # 如果最后一个批次尚未添加，将其添加到所有上下文文本和记录列表中
    # 如果当前上下文文本不在所有上下文文本列表中
    if current_context_text not in all_context_text:
        # 如果当前上下文记录数大于1
        if len(current_context_records) > 1:
            # 将当前上下文记录转换为数据框
            record_df = _convert_report_context_to_df(
                context_records=current_context_records[1:],  # 使用除第一个记录外的其余记录
                header=current_context_records[0],  # 使用第一个记录作为表头
                weight_column=community_weight_name  # 如果存在实体和包括社区权重，则使用指定的权重列名
                              if entities and include_community_weight
                              else None,  # 否则权重列为空
                rank_column=community_rank_name  # 如果包括社区等级，则使用指定的等级列名
                            if include_community_rank
                            else None,  # 否则等级列为空
            )
        else:
            # 如果当前上下文记录数不大于1，则创建一个空的数据框
            record_df = pd.DataFrame()
        
        # 将当前数据框添加到所有上下文记录列表中
        all_context_records.append(record_df)
        
        # 将当前数据框转换为 CSV 格式的文本，并存储到当前上下文文本列表中
        current_context_text = record_df.to_csv(index=False, sep=column_delimiter)
        all_context_text.append(current_context_text)

    # 返回所有上下文文本列表和一个字典，其中键为小写的上下文名，值为合并所有上下文记录的数据框
    return all_context_text, {
        context_name.lower(): pd.concat(all_context_records, ignore_index=True)
    }
# 计算社区权重，即与社区内实体相关联的文本单位的数量
def _compute_community_weights(
    community_reports: list[CommunityReport],
    entities: list[Entity],
    weight_attribute: str = "occurrence",
    normalize: bool = True,
) -> list[CommunityReport]:
    """Calculate a community's weight as count of text units associated with entities within the community."""
    
    # 创建一个字典来存储每个社区中关联的文本单位
    community_text_units = {}
    for entity in entities:
        if entity.community_ids:
            for community_id in entity.community_ids:
                if community_id not in community_text_units:
                    community_text_units[community_id] = []
                # 将实体的文本单位添加到对应社区的列表中
                community_text_units[community_id].extend(entity.text_unit_ids)
    
    # 更新社区报告对象中的权重属性
    for report in community_reports:
        if not report.attributes:
            report.attributes = {}
        # 将社区报告的权重属性设为对应社区中文本单位的数量（去重）
        report.attributes[weight_attribute] = len(
            set(community_text_units.get(report.community_id, []))
        )
    
    if normalize:
        # 如果需要归一化，按照最大权重进行归一化
        all_weights = [
            report.attributes[weight_attribute]
            for report in community_reports
            if report.attributes
        ]
        max_weight = max(all_weights)
        # 对每个社区报告的权重属性进行归一化处理
        for report in community_reports:
            if report.attributes:
                report.attributes[weight_attribute] = (
                    report.attributes[weight_attribute] / max_weight
                )
    
    return community_reports


# 将报告内容按社区权重和排名（如果存在）进行排序
def _rank_report_context(
    report_df: pd.DataFrame,
    weight_column: str | None = "occurrence weight",
    rank_column: str | None = "rank",
) -> pd.DataFrame:
    """Sort report context by community weight and rank if exist."""
    
    rank_attributes = []
    if weight_column:
        rank_attributes.append(weight_column)
        # 将权重列的数据类型转换为浮点型
        report_df[weight_column] = report_df[weight_column].astype(float)
    if rank_column:
        rank_attributes.append(rank_column)
        # 将排名列的数据类型转换为浮点型
        report_df[rank_column] = report_df[rank_column].astype(float)
    
    # 根据指定的排序属性对报告数据框进行降序排序
    if len(rank_attributes) > 0:
        report_df.sort_values(by=rank_attributes, ascending=False, inplace=True)
    
    return report_df


# 将报告内容记录转换为 pandas 数据框，并按权重和排名（如果存在）进行排序
def _convert_report_context_to_df(
    context_records: list[list[str]],
    header: list[str],
    weight_column: str | None = None,
    rank_column: str | None = None,
) -> pd.DataFrame:
    """Convert report context records to pandas dataframe and sort by weight and rank if exist."""
    
    # 创建 pandas 数据框，使用给定的记录和标题
    record_df = pd.DataFrame(
        context_records,
        columns=cast(Any, header),
    )
    
    # 调用 _rank_report_context 函数，对数据框按指定的权重和排名列进行排序
    return _rank_report_context(
        report_df=record_df,
        weight_column=weight_column,
        rank_column=rank_column,
    )
```