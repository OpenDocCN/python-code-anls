# `.\graphrag\graphrag\prompt_tune\template\entity_extraction.py`

```py
# 定义用于文本实体抽取的提示信息，包含目标、步骤和示例等详细说明
GRAPH_EXTRACTION_PROMPT = """
-Goal-
给定一个可能与此活动相关的文本文档和一组实体类型，从文本中识别所有这些类型的实体以及识别出的所有实体之间的所有关系。

-Steps-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体的名称，大写
- entity_type: 下列类型之一: [{entity_types}]
- entity_description: 实体属性和活动的详细描述
将每个实体格式化为 ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>

2. 从步骤 1 中识别出的实体中，识别所有明显相关的（source_entity, target_entity）对。
对于每对相关的实体，提取以下信息：
- source_entity: 源实体的名称，如步骤 1 中识别的
- target_entity: 目标实体的名称，如步骤 1 中识别的
- relationship_description: 解释为什么认为源实体和目标实体彼此相关
- relationship_strength: 介于 1 到 10 之间的整数分数，表示源实体和目标实体之间关系的强度

将每个关系格式化为 ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

3. 以 {language} 格式返回输出，作为步骤 1 和步骤 2 中识别的所有实体和关系的单个列表。使用 **{{record_delimiter}}** 作为列表分隔符。如果需要翻译，只翻译描述，不做其他修改！

4. 完成时输出 {{completion_delimiter}}

-Examples-
######################
{examples}

-Real Data-
######################
entity_types: [{entity_types}]
text: {{input_text}}
######################
output:"""

# 定义用于文本实体抽取的 JSON 格式提示信息，包含目标、步骤等详细说明
GRAPH_EXTRACTION_JSON_PROMPT = """
-Goal-
给定一个可能与此活动相关的文本文档和一组实体类型，从文本中识别所有这些类型的实体以及识别出的所有实体之间的所有关系。

-Steps-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体的名称，大写
- entity_type: 下列类型之一: [{entity_types}]
- entity_description: 实体属性和活动的详细描述
将每个实体输出格式化为以下 JSON 格式条目：

{{"name": <entity name>, "type": <type>, "description": <entity description>}}

2. 从步骤 1 中识别出的实体中，识别所有明显相关的（source_entity, target_entity）对。
对于每对相关的实体，提取以下信息：
# 基于给定的文本文档，首先识别所有必需的实体，以捕获文本中的信息和思想。
# 接下来，报告所有已识别实体之间的关系。

-Goal-
给定一个可能与此活动相关的文本文档，首先识别文本中需要的所有实体，以便捕获文本中的信息和思想。
接下来，报告所识别实体之间的所有关系。

-Steps-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体的名称，大写
- entity_type: 建议实体的几个标签或类别。类别不应过于具体，应尽可能一般化。
- entity_description: 实体属性和活动的全面描述
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. 从第1步识别出的实体中，识别所有明显相关的（source_entity，target_entity）对。
对于每对相关实体，提取以下信息：
- source_entity: 根据第1步识别出的源实体的名称
- target_entity: 根据第1步识别出的目标实体的名称
- relationship_description: 解释为何认为源实体和目标实体彼此相关的说明
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分数
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
# 返回以 {language} 语言编写的输出，将步骤1和2中识别的所有实体和关系作为一个列表输出。使用 {{record_delimiter}} 作为列表分隔符。
# 如果需要翻译，请仅翻译描述部分，不要改变其他内容！

# 当完成时，输出 {{completion_delimiter}}

# 示例
######################
{examples}

# 实际数据
######################
text: {{input_text}}
######################
output:
"""
```