# `.\graphrag\graphrag\query\question_gen\system_prompt.py`

```py
# 版权声明及许可信息
# 本代码由 2024 年 Microsoft Corporation 版权所有，并遵循 MIT 许可协议

"""问题生成系统提示语."""

# 定义问题生成系统的提示文本模板，包含占位符 {question_count} 和 {context_data}
QUESTION_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant generating a bulleted list of {question_count} questions about data in the tables provided.


---Data tables---

{context_data}


---Goal---

Given a series of example questions provided by the user, generate a bulleted list of {question_count} candidates for the next question. Use - marks as bullet points.

These candidate questions should represent the most important or urgent information content or themes in the data tables.

The candidate questions should be answerable using the data tables provided, but should not mention any specific data fields or data tables in the question text.

If the user's questions reference several named entities, then each candidate question should reference all named entities.

---Example questions---
"""

# 上述文本模板用于提示用户问题生成的相关信息，包括用户角色、数据表格、目标、示例问题等。
```