# `.\graphrag\graphrag\index\verbs\text\translate\strategies\defaults.py`

```py
# 定义一个常量 TRANSLATION_PROMPT，其包含一个多行的字符串值，用于提示翻译任务的内容。
"""A file containing TRANSLATION_PROMPT value definition."""

# 定义一个多行字符串常量 TRANSLATION_PROMPT，提示用户翻译任务的详细说明。
TRANSLATION_PROMPT = """
    You are a helpful assistant. Translate into {language} the following text, and make sure all of the text is in {language}.
    """.strip()
```