# `.\graphrag\graphrag\llm\openai\_prompts.py`

```py
# 声明一个长字符串，用于提供JSON检查的提示信息和示例
JSON_CHECK_PROMPT = """
You are going to be given a malformed JSON string that threw an error during json.loads.
It probably contains unnecessary escape sequences, or it is missing a comma or colon somewhere.
Your task is to fix this string and return a well-formed JSON string containing a single object.
Eliminate any unnecessary escape sequences.
Only return valid JSON, parseable with json.loads, without commentary.

# Examples
-----------
Text: {{ \\"title\\": \\"abc\\", \\"summary\\": \\"def\\" }}
Output: {{"title": "abc", "summary": "def"}}
-----------
Text: {{"title": "abc", "summary": "def"
Output: {{"title": "abc", "summary": "def"}}
-----------
Text: {{"title': "abc", 'summary": "def"
Output: {{"title": "abc", "summary": "def"}}
-----------
Text: "{{"title": "abc", "summary": "def"}}"
Output: {{"title": "abc", "summary": "def"}}
-----------
Text: [{{"title": "abc", "summary": "def"}}]
Output: [{{"title": "abc", "summary": "def"}}]
-----------
Text: [{{"title": "abc", "summary": "def"}}, {{ \\"title\\": \\"abc\\", \\"summary\\": \\"def\\" }}]
Output: [{{"title": "abc", "summary": "def"}}, {{"title": "abc", "summary": "def"}}]
-----------
Text: ```json\n[{{"title": "abc", "summary": "def"}}, {{ \\"title\\": \\"abc\\", \\"summary\\": \\"def\\" }}]```py
Output: [{{"title": "abc", "summary": "def"}}, {{"title": "abc", "summary": "def"}}]


# Real Data
Text: {input_text}
Output:"
"""
```