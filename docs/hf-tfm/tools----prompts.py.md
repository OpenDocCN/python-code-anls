# `.\transformers\tools\prompts.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 导入 re 模块
import re

# 从上级目录的 utils 模块中导入 cached_file 函数
from ..utils import cached_file

# 定义文本提示消息字符串
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """

# 定义默认的提示仓库地址
DEFAULT_PROMPTS_REPO = "huggingface-tools/default-prompts"
# 定义不同类型的提示文件名
PROMPT_FILES = {"chat": "chat_prompt_template.txt", "run": "run_prompt_template.txt"}

# 定义下载提示的函数
def download_prompt(prompt_or_repo_id, agent_name, mode="run"):
    """
    Downloads and caches the prompt from a repo and returns it contents (if necessary)
    """
    # 如果未提供提示或者仓库地址，则使用默认仓库地址
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO

    # 当提示中不包含空格时，认为是仓库地址
    if re.search("\\s", prompt_or_repo_id) is not None:
        return prompt_or_repo_id

    # 使用 cached_file 函数下载并缓存提示文件，并返回其内容
    prompt_file = cached_file(
        prompt_or_repo_id, PROMPT_FILES[mode], repo_type="dataset", user_agent={"agent": agent_name}
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
```