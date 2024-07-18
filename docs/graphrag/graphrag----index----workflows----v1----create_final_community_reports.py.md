# `.\graphrag\graphrag\index\workflows\v1\create_final_community_reports.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""一个包含 build_steps 方法定义的模块。"""

# 从 graphrag.index.config 模块导入 PipelineWorkflowConfig 和 PipelineWorkflowStep
from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

# 定义工作流名称为 "create_final_community_reports"
workflow_name = "create_final_community_reports"


def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    创建最终的社区报告表格。

    ## Dependencies
    * `workflow:create_base_entity_graph`
    """
    # 检索配置中是否启用协变量，默认为 False
    covariates_enabled = config.get("covariates_enabled", False)
    # 检索创建社区报告的配置信息，默认为空字典
    create_community_reports_config = config.get("create_community_reports", {})
    # 检索文本嵌入的基础配置信息，默认为 {}
    base_text_embed = config.get("text_embed", {})
    # 检索社区报告全文嵌入的配置信息，如果未配置则使用 base_text_embed
    community_report_full_content_embed_config = config.get(
        "community_report_full_content_embed", base_text_embed
    )
    # 检索社区报告摘要嵌入的配置信息，如果未配置则使用 base_text_embed
    community_report_summary_embed_config = config.get(
        "community_report_summary_embed", base_text_embed
    )
    # 检索社区报告标题嵌入的配置信息，如果未配置则使用 base_text_embed
    community_report_title_embed_config = config.get(
        "community_report_title_embed", base_text_embed
    )
    # 检索是否跳过标题嵌入的配置信息，默认为 False
    skip_title_embedding = config.get("skip_title_embedding", False)
    # 检索是否跳过摘要嵌入的配置信息，默认为 False
    skip_summary_embedding = config.get("skip_summary_embedding", False)
    # 检索是否跳过全文嵌入的配置信息，默认为 False
    skip_full_content_embedding = config.get("skip_full_content_embedding", False)

    # 返回一个 PipelineWorkflowStep 对象的列表
    return [
```