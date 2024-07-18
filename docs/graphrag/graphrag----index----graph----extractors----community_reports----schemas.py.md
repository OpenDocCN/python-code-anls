# `.\graphrag\graphrag\index\graph\extractors\community_reports\schemas.py`

```py
# POST-PREP NODE TABLE SCHEMA
# 节点表结构定义

NODE_ID = "human_readable_id"
# 节点 ID 字段，用人类可读的 ID 表示

NODE_NAME = "title"
# 节点名称字段，表示标题或名称

NODE_DESCRIPTION = "description"
# 节点描述字段，表示描述信息

NODE_DEGREE = "degree"
# 节点度字段，表示节点的度或相关度

NODE_DETAILS = "node_details"
# 节点详细信息字段，存储与节点相关的详细数据

NODE_COMMUNITY = "community"
# 节点社区字段，表示节点所属的社区或群体

NODE_LEVEL = "level"
# 节点级别字段，表示节点在层次结构中的级别

# POST-PREP EDGE TABLE SCHEMA
# 边表结构定义

EDGE_ID = "human_readable_id"
# 边 ID 字段，用人类可读的 ID 表示

EDGE_SOURCE = "source"
# 边的源节点字段，表示边的起始节点

EDGE_TARGET = "target"
# 边的目标节点字段，表示边的结束节点

EDGE_DESCRIPTION = "description"
# 边的描述字段，表示边的说明信息

EDGE_DEGREE = "rank"
# 边的度字段，表示边的权重或等级

EDGE_DETAILS = "edge_details"
# 边的详细信息字段，存储与边相关的详细数据

EDGE_WEIGHT = "weight"
# 边的权重字段，表示边的权值或重要性

# POST-PREP CLAIM TABLE SCHEMA
# 索赔表结构定义

CLAIM_ID = "human_readable_id"
# 索赔 ID 字段，用人类可读的 ID 表示

CLAIM_SUBJECT = "subject_id"
# 索赔主体字段，表示索赔所涉及的主体 ID

CLAIM_TYPE = "type"
# 索赔类型字段，表示索赔的类型或种类

CLAIM_STATUS = "status"
# 索赔状态字段，表示索赔的状态信息

CLAIM_DESCRIPTION = "description"
# 索赔描述字段，表示索赔的详细描述信息

CLAIM_DETAILS = "claim_details"
# 索赔详细信息字段，存储与索赔相关的详细数据

# COMMUNITY HIERARCHY TABLE SCHEMA
# 社区层级表结构定义

SUB_COMMUNITY = "sub_communitty"
# 子社区字段，表示社区内的子社区或分支

SUB_COMMUNITY_SIZE = "sub_community_size"
# 子社区大小字段，表示子社区的规模或大小

COMMUNITY_LEVEL = "level"
# 社区级别字段，表示社区在整体结构中的层级

# COMMUNITY CONTEXT TABLE SCHEMA
# 社区上下文表结构定义

ALL_CONTEXT = "all_context"
# 所有上下文字段，表示包含所有上下文信息的字段

CONTEXT_STRING = "context_string"
# 上下文字符串字段，表示特定上下文的字符串表达

CONTEXT_SIZE = "context_size"
# 上下文大小字段，表示上下文的大小或容量

CONTEXT_EXCEED_FLAG = "context_exceed_limit"
# 上下文超出标志字段，表示上下文是否超出了限制的标志位

# COMMUNITY REPORT TABLE SCHEMA
# 社区报告表结构定义

REPORT_ID = "id"
# 报告 ID 字段，表示报告的唯一标识符

COMMUNITY_ID = "id"
# 社区 ID 字段，表示报告所属的社区或群体的标识符

COMMUNITY_LEVEL = "level"
# 社区级别字段，表示报告所属社区的级别

TITLE = "title"
# 标题字段，表示报告的标题或名称

SUMMARY = "summary"
# 摘要字段，表示报告的摘要或总结

FINDINGS = "findings"
# 发现字段，表示报告中的发现或结果

RATING = "rank"
# 等级字段，表示报告的等级或评分

EXPLANATION = "rating_explanation"
# 评分解释字段，表示评分或等级的解释说明

FULL_CONTENT = "full_content"
# 完整内容字段，表示报告的完整内容

FULL_CONTENT_JSON = "full_content_json"
# 完整内容 JSON 字段，表示报告完整内容的 JSON 格式
```