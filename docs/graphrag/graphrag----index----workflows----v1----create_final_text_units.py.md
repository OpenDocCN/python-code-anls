# `.\graphrag\graphrag\index\workflows\v1\create_final_text_units.py`

```py
# 从 graphrag.index.config 导入 PipelineWorkflowConfig 和 PipelineWorkflowStep 类
from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep

# 定义工作流名称为 "create_final_text_units"
workflow_name = "create_final_text_units"

# 定义 build_steps 函数，接收一个 PipelineWorkflowConfig 类型的参数 config，并返回 PipelineWorkflowStep 类型的列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    创建最终的文本单元表。

    ## Dependencies
    * `workflow:create_base_text_units`
    * `workflow:create_final_entities`
    * `workflow:create_final_communities`
    """
    # 从配置中获取 "text_embed" 键对应的值，若不存在则为空字典
    base_text_embed = config.get("text_embed", {})
    # 获取 "text_unit_text_embed" 键对应的值，若不存在则使用 base_text_embed 的值
    text_unit_text_embed_config = config.get("text_unit_text_embed", base_text_embed)
    # 获取 "covariates_enabled" 键对应的值，若不存在则为 False
    covariates_enabled = config.get("covariates_enabled", False)
    # 获取 "skip_text_unit_embedding" 键对应的值，若不存在则为 False
    skip_text_unit_embedding = config.get("skip_text_unit_embedding", False)
    # 检查是否正在使用向量存储，从 "text_unit_text_embed" 中获取 "strategy" 下的 "vector_store" 值
    is_using_vector_store = (
        text_unit_text_embed_config.get("strategy", {}).get("vector_store", None)
        is not None
    )

    # 返回构建好的工作流步骤列表
    return [
```