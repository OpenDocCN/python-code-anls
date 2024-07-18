# `.\graphrag\graphrag\index\workflows\v1\create_final_entities.py`

```py
# 定义了一个名称为 `build_steps` 的函数，用于构建流水线工作流步骤列表
def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    创建最终实体表。

    ## Dependencies
    * `workflow:create_base_entity_graph`
    """

    # 从配置中获取文本嵌入的基础配置
    base_text_embed = config.get("text_embed", {})
    
    # 获取实体名称嵌入的配置，如果未指定，则使用基础文本嵌入配置
    entity_name_embed_config = config.get("entity_name_embed", base_text_embed)
    
    # 获取实体名称描述嵌入的配置，如果未指定，则使用基础文本嵌入配置
    entity_name_description_embed_config = config.get(
        "entity_name_description_embed", base_text_embed
    )
    
    # 检查是否跳过名称嵌入步骤
    skip_name_embedding = config.get("skip_name_embedding", False)
    
    # 检查是否跳过描述嵌入步骤
    skip_description_embedding = config.get("skip_description_embedding", False)
    
    # 检查是否使用向量存储作为策略的一部分
    is_using_vector_store = (
        entity_name_embed_config.get("strategy", {}).get("vector_store", None)
        is not None
    )

    # 返回构建的流水线工作流步骤列表
    return [
        # 未完整的列表，需要在此处继续添加步骤
    ]
```