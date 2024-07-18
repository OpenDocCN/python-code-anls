# `.\graphrag\graphrag\prompt_tune\prompt\domain.py`

```py
# 定义一个多行字符串常量，用于生成领域提示文本模板
GENERATE_DOMAIN_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by assigning a descriptive domain that summarizes what the text is about.
Example domains are: "Social studies", "Algorithmic analysis", "Medical science", among others.

Text: {input_text}
Domain:"""
```