# `.\graphrag\graphrag\prompt_tune\prompt\language.py`

```py
# 设置用于语言检测的提示文本模板
DETECT_LANGUAGE_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by determining what's the primary language of the provided texts.
Examples are: "English", "Spanish", "Japanese", "Portuguese" among others.

Text: {input_text}
Language:"""
```