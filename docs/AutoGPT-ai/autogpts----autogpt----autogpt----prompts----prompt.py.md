# `.\AutoGPT\autogpts\autogpt\autogpt\prompts\prompt.py`

```py
# 默认的触发提示，指导用户根据给定的目标和已经取得的进展，确定下一步要使用的命令，并使用之前指定的 JSON 模式进行响应
DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use next based on the given goals "
    "and the progress you have made so far, "
    "and respond using the JSON schema specified previously:"
)
```