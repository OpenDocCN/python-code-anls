# `.\graphrag\graphrag\prompt_tune\prompt\entity_types.py`

```py
"""Fine-tuning prompts for entity types generation."""

# 定义一个多行字符串常量，用于生成实体类型的提示文本
ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
And remember, it is ENTITY TYPES what we need.
Return the entity types in as a list of comma sepparated of strings.
=====================================================================
EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.
"""

# 定义另一个多行字符串常量，用于生成实体类型的提示文本，JSON格式
ENTITY_TYPE_GENERATION_JSON_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
"""

# 注释结束
# 分析任务描述和输入文本，需要识别文本中存在的实体类型
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
JSON response:
{{"entity_types": [<entity_types>] }}

# 返回 JSON 格式的结果，包含实体类型的数组作为 "entity_types" 键的值
{{"entity_types": [organization, person] }}
```