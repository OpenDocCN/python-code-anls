# `.\graphrag\graphrag\prompt_tune\prompt\community_reporter_role.py`

```py
# 为社区记者角色生成微调提示文本

GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT = """
{persona}
给定一个示例文本，帮助用户创建一个将负责社区分析的角色定义。
分析示例文本的关键部分，并使用提供的领域和您的专业知识，为提供的输入创建一个新的角色定义，其结构和内容与示例相同。
请记住，您的输出应与提供的示例在结构和内容上保持一致。

Example:
A technologist reporter that is analyzing Kevin Scott's "Behind the Tech Podcast", given a list of entities
that belong to the community as well as their relationships and optional associated claims.
The report will be used to inform decision-makers about significant developments associated with the community and their potential impact.


Domain: {domain}
Text: {input_text}
Role:"""
```