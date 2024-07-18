# `.\graphrag\graphrag\index\workflows\default_workflows.py`

```py
# 版权声明和许可证信息
# 版权所有 (c) 2024 Microsoft Corporation.
# 根据 MIT 许可证发布

# 导入自定义类型定义
from .typing import WorkflowDefinitions

# 导入创建基础文档工作流相关函数和变量
from .v1.create_base_documents import (
    build_steps as build_create_base_documents_steps,
)
from .v1.create_base_documents import (
    workflow_name as create_base_documents,
)

# 导入创建基础实体图工作流相关函数和变量
from .v1.create_base_entity_graph import (
    build_steps as build_create_base_entity_graph_steps,
)
from .v1.create_base_entity_graph import (
    workflow_name as create_base_entity_graph,
)

# 导入创建基础提取实体工作流相关函数和变量
from .v1.create_base_extracted_entities import (
    build_steps as build_create_base_extracted_entities_steps,
)
from .v1.create_base_extracted_entities import (
    workflow_name as create_base_extracted_entities,
)

# 导入创建基础文本单元工作流相关函数和变量
from .v1.create_base_text_units import (
    build_steps as build_create_base_text_units_steps,
)
from .v1.create_base_text_units import (
    workflow_name as create_base_text_units,
)

# 导入创建最终社区工作流相关函数和变量
from .v1.create_final_communities import (
    build_steps as build_create_final_communities_steps,
)
from .v1.create_final_communities import (
    workflow_name as create_final_communities,
)

# 导入创建最终社区报告工作流相关函数和变量
from .v1.create_final_community_reports import (
    build_steps as build_create_final_community_reports_steps,
)
from .v1.create_final_community_reports import (
    workflow_name as create_final_community_reports,
)

# 导入创建最终协变量工作流相关函数和变量
from .v1.create_final_covariates import (
    build_steps as build_create_final_covariates_steps,
)
from .v1.create_final_covariates import (
    workflow_name as create_final_covariates,
)

# 导入创建最终文档工作流相关函数和变量
from .v1.create_final_documents import (
    build_steps as build_create_final_documents_steps,
)
from .v1.create_final_documents import (
    workflow_name as create_final_documents,
)

# 导入创建最终实体工作流相关函数和变量
from .v1.create_final_entities import (
    build_steps as build_create_final_entities_steps,
)
from .v1.create_final_entities import (
    workflow_name as create_final_entities,
)

# 导入创建最终节点工作流相关函数和变量
from .v1.create_final_nodes import (
    build_steps as build_create_final_nodes_steps,
)
from .v1.create_final_nodes import (
    workflow_name as create_final_nodes,
)

# 导入创建最终关系工作流相关函数和变量
from .v1.create_final_relationships import (
    build_steps as build_create_final_relationships_steps,
)
from .v1.create_final_relationships import (
    workflow_name as create_final_relationships,
)

# 导入创建最终文本单元工作流相关函数和变量
from .v1.create_final_text_units import (
    build_steps as build_create_final_text_units_steps,
)
from .v1.create_final_text_units import (
    workflow_name as create_final_text_units,
)

# 导入创建汇总实体工作流相关函数和变量
from .v1.create_summarized_entities import (
    build_steps as build_create_summarized_entities_steps,
)
from .v1.create_summarized_entities import (
    workflow_name as create_summarized_entities,
)

# 导入将文本单元与协变量 ID 关联工作流相关函数和变量
from .v1.join_text_units_to_covariate_ids import (
    build_steps as join_text_units_to_covariate_ids_steps,
)
from .v1.join_text_units_to_covariate_ids import (
    workflow_name as join_text_units_to_covariate_ids,
)

# 导入将文本单元与实体 ID 关联工作流相关函数和变量
from .v1.join_text_units_to_entity_ids import (
    # 将变量名 build_steps 重命名为 join_text_units_to_entity_ids_steps
    build_steps as join_text_units_to_entity_ids_steps,
# 从当前目录的 v1 子模块中导入指定模块的 workflow_name 别名作为 join_text_units_to_entity_ids
from .v1.join_text_units_to_entity_ids import (
    workflow_name as join_text_units_to_entity_ids,
)

# 从当前目录的 v1 子模块中导入指定模块的 build_steps 别名作为 join_text_units_to_relationship_ids_steps
from .v1.join_text_units_to_relationship_ids import (
    build_steps as join_text_units_to_relationship_ids_steps,
)

# 从当前目录的 v1 子模块中导入指定模块的 workflow_name 别名作为 join_text_units_to_relationship_ids
from .v1.join_text_units_to_relationship_ids import (
    workflow_name as join_text_units_to_relationship_ids,
)

# 定义一个默认的工作流定义字典，键是函数名，值是对应函数构建步骤的引用
default_workflows: WorkflowDefinitions = {
    create_base_extracted_entities: build_create_base_extracted_entities_steps,
    create_base_entity_graph: build_create_base_entity_graph_steps,
    create_base_text_units: build_create_base_text_units_steps,
    create_final_text_units: build_create_final_text_units,
    create_final_community_reports: build_create_final_community_reports_steps,
    create_final_nodes: build_create_final_nodes_steps,
    create_final_relationships: build_create_final_relationships_steps,
    create_final_documents: build_create_final_documents_steps,
    create_final_covariates: build_create_final_covariates_steps,
    create_base_documents: build_create_base_documents_steps,
    create_final_entities: build_create_final_entities_steps,
    create_final_communities: build_create_final_communities_steps,
    create_summarized_entities: build_create_summarized_entities_steps,
    join_text_units_to_entity_ids: join_text_units_to_entity_ids_steps,
    join_text_units_to_covariate_ids: join_text_units_to_covariate_ids_steps,
    join_text_units_to_relationship_ids: join_text_units_to_relationship_ids_steps,
}
```