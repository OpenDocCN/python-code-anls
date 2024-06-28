# `.\tools\prompts.py`

```py
# 消息提示模板，包含了人类消息和助手回复的基本结构
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """
# 默认的提示信息仓库地址
DEFAULT_PROMPTS_REPO = "huggingface-tools/default-prompts"
# 不同模式下的提示文件名映射
PROMPT_FILES = {"chat": "chat_prompt_template.txt", "run": "run_prompt_template.txt"}

def download_prompt(prompt_or_repo_id, agent_name, mode="run"):
    """
    根据提示信息或仓库 ID 下载并缓存提示信息，并返回其内容（如果需要）
    """
    # 如果未提供提示信息或仓库 ID，则使用默认的提示信息仓库地址
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO

    # 当提示信息中包含空格时，视其为仓库 ID 而非具体的提示信息
    if re.search("\\s", prompt_or_repo_id) is not None:
        return prompt_or_repo_id

    # 使用 cached_file 函数下载指定模式下的提示文件，并返回其文件路径
    prompt_file = cached_file(
        prompt_or_repo_id, PROMPT_FILES[mode], repo_type="dataset", user_agent={"agent": agent_name}
    )
    # 打开并读取下载的提示文件内容
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
```