# `.\AutoGPT\autogpts\autogpt\autogpt\app\utils.py`

```py
# 导入日志、操作系统、正则表达式、系统模块
import logging
import os
import re
import sys

# 导入请求模块、颜色模块、Git 模块、提示工具模块
import requests
from colorama import Fore, Style
from git import InvalidGitRepositoryError, Repo
from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.history import InMemoryHistory

# 从自定义模块中导入 Config 类
from autogpt.config import Config

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 创建一个提示会话对象，用于用户输入历史记录
session = PromptSession(history=InMemoryHistory())

# 异步函数，用于清理用户输入
async def clean_input(config: Config, prompt: str = ""):
    try:
        # 如果启用了聊天消息功能
        if config.chat_messages_enabled:
            # 遍历配置中的插件
            for plugin in config.plugins:
                # 如果插件没有 can_handle_user_input 方法，则跳过
                if not hasattr(plugin, "can_handle_user_input"):
                    continue
                # 如果插件能够处理用户输入
                if plugin.can_handle_user_input(user_input=prompt):
                    # 调用插件的 user_input 方法处理用户输入
                    plugin_response = plugin.user_input(user_input=prompt)
                    # 如果插件响应为空，则继续下一个插件
                    if not plugin_response:
                        continue
                    # 如果插件响应为肯定回答，则返回授权密钥
                    if plugin_response.lower() in [
                        "yes",
                        "yeah",
                        "y",
                        "ok",
                        "okay",
                        "sure",
                        "alright",
                    ]:
                        return config.authorise_key
                    # 如果插件响应为否定回答，则返回退出密钥
                    elif plugin_response.lower() in [
                        "no",
                        "nope",
                        "n",
                        "negative",
                    ]:
                        return config.exit_key
                    # 其他情况下返回插件响应
                    return plugin_response

        # 如果未启用聊天消息功能，则通过键盘询问用户
        logger.debug("Asking user via keyboard...")

        # 调用提示会话对象的 prompt_async 方法，等待用户输入
        # handle_sigint 必须设置为 False，以便在 autogpt/main.py 中正确使用信号处理程序
        answer = await session.prompt_async(ANSI(prompt + " "), handle_sigint=False)
        return answer
    # 捕获键盘中断异常，用户按下Ctrl+C时触发
    except KeyboardInterrupt:
        # 记录信息，提示用户中断了AutoGPT
        logger.info("You interrupted AutoGPT")
        # 记录信息，提示正在退出程序
        logger.info("Quitting...")
        # 退出程序，返回状态码0表示正常退出
        exit(0)
# 从指定网址获取公告内容
def get_bulletin_from_web():
    try:
        # 发送 GET 请求获取网址内容
        response = requests.get(
            "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpts/autogpt/BULLETIN.md"  # noqa: E501
        )
        # 如果响应状态码为 200，返回响应文本
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException:
        pass

    return ""


# 获取当前 Git 分支名称
def get_current_git_branch() -> str:
    try:
        # 在父目录中搜索 Git 仓库
        repo = Repo(search_parent_directories=True)
        # 获取当前活动分支名称
        branch = repo.active_branch
        return branch.name
    except InvalidGitRepositoryError:
        return ""


# 获取最新公告内容和是否为新消息的元组
def get_latest_bulletin() -> tuple[str, bool]:
    # 检查当前目录下是否存在 CURRENT_BULLETIN.md 文件
    exists = os.path.exists("data/CURRENT_BULLETIN.md")
    current_bulletin = ""
    # 如果文件存在，读取当前公告内容
    if exists:
        current_bulletin = open(
            "data/CURRENT_BULLETIN.md", "r", encoding="utf-8"
        ).read()
    # 获取网络上的最新公告内容
    new_bulletin = get_bulletin_from_web()
    # 判断是否有新消息
    is_new_news = new_bulletin != "" and new_bulletin != current_bulletin

    # 设置公告标题样式
    news_header = Fore.YELLOW + "Welcome to AutoGPT!\n"
    if new_bulletin or current_bulletin:
        news_header += (
            "Below you'll find the latest AutoGPT News and feature updates!\n"
            "If you don't wish to see this message, you "
            "can run AutoGPT with the *--skip-news* flag.\n"
        )

    # 如果有新公告且为新消息，将新公告内容写入文件，并更新当前公告内容
    if new_bulletin and is_new_news:
        open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8").write(new_bulletin)
        current_bulletin = f"{Fore.RED}::NEW BULLETIN::{Fore.RESET}\n\n{new_bulletin}"

    return f"{news_header}\n{current_bulletin}", is_new_news


# 将 Markdown 格式转换为 ANSI 样式
def markdown_to_ansi_style(markdown: str):
    ansi_lines: list[str] = []
    # 遍历 markdown 文本的每一行
    for line in markdown.split("\n"):
        # 初始化行的样式为空字符串
        line_style = ""

        # 如果行以 "# " 开头，则添加亮色样式
        if line.startswith("# "):
            line_style += Style.BRIGHT
        else:
            # 使用正则表达式匹配行中的斜体格式，添加亮色样式
            line = re.sub(
                r"(?<!\*)\*(\*?[^*]+\*?)\*(?!\*)",
                rf"{Style.BRIGHT}\1{Style.NORMAL}",
                line,
            )

        # 如果行以 "#+ " 开头，则添加青色样式，并去除行中的 "#+ " 标记
        if re.match(r"^#+ ", line) is not None:
            line_style += Fore.CYAN
            line = re.sub(r"^#+ ", "", line)

        # 将带有样式的行添加到 ansi_lines 列表中
        ansi_lines.append(f"{line_style}{line}{Style.RESET_ALL}")
    # 将带有样式的行连接成一个字符串并返回
    return "\n".join(ansi_lines)
# 定义一个函数，返回法律警告文本
def get_legal_warning() -> str:
    # 法律警告文本内容，包含免责和赔偿协议
    legal_text = """
## DISCLAIMER AND INDEMNIFICATION AGREEMENT
### PLEASE READ THIS DISCLAIMER AND INDEMNIFICATION AGREEMENT CAREFULLY BEFORE USING THE AUTOGPT SYSTEM. BY USING THE AUTOGPT SYSTEM, YOU AGREE TO BE BOUND BY THIS AGREEMENT.

## Introduction
AutoGPT (the "System") is a project that connects a GPT-like artificial intelligence system to the internet and allows it to automate tasks. While the System is designed to be useful and efficient, there may be instances where the System could perform actions that may cause harm or have unintended consequences.

## No Liability for Actions of the System
The developers, contributors, and maintainers of the AutoGPT project (collectively, the "Project Parties") make no warranties or representations, express or implied, about the System's performance, accuracy, reliability, or safety. By using the System, you understand and agree that the Project Parties shall not be liable for any actions taken by the System or any consequences resulting from such actions.

## User Responsibility and Respondeat Superior Liability
As a user of the System, you are responsible for supervising and monitoring the actions of the System while it is operating on your
behalf. You acknowledge that using the System could expose you to potential liability including but not limited to respondeat superior and you agree to assume all risks and liabilities associated with such potential liability.

## Indemnification
By using the System, you agree to indemnify, defend, and hold harmless the Project Parties from and against any and all claims, liabilities, damages, losses, or expenses (including reasonable attorneys' fees and costs) arising out of or in connection with your use of the System, including, without limitation, any actions taken by the System on your behalf, any failure to properly supervise or monitor the System, and any resulting harm or unintended consequences.
    """  # noqa: E501
    # 返回 legal_text 变量的值
    return legal_text
# 打印最新的消息，如果有的话
def print_motd(config: Config, logger: logging.Logger):
    # 获取最新的公告消息和是否是新消息
    motd, is_new_motd = get_latest_bulletin()
    # 如果有消息
    if motd:
        # 将消息内容转换为 ANSI 格式
        motd = markdown_to_ansi_style(motd)
        # 按行遍历消息内容
        for motd_line in motd.split("\n"):
            # 打印消息到日志中
            logger.info(
                extra={
                    "title": "NEWS:",
                    "title_color": Fore.GREEN,
                    "preserve_color": True,
                },
                msg=motd_line,
            )
        # 如果是新消息且聊天消息未启用
        if is_new_motd and not config.chat_messages_enabled:
            # 提示用户按 Enter 键继续
            input(
                Fore.MAGENTA
                + Style.BRIGHT
                + "NEWS: Bulletin was updated! Press Enter to continue..."
                + Style.RESET_ALL
            )

# 打印当前 Git 分支信息
def print_git_branch_info(logger: logging.Logger):
    # 获取当前 Git 分支
    git_branch = get_current_git_branch()
    # 如果存在 Git 分支且不是 master 分支
    if git_branch and git_branch != "master":
        # 打印警告信息
        logger.warning(
            f"You are running on `{git_branch}` branch"
            " - this is not a supported branch."
        )

# 打印 Python 版本信息
def print_python_version_info(logger: logging.Logger):
    # 如果 Python 版本低于 3.10
    if sys.version_info < (3, 10):
        # 打印错误信息，建议升级到 Python 3.10 或更高版本
        logger.error(
            "WARNING: You are running on an older version of Python. "
            "Some people have observed problems with certain "
            "parts of AutoGPT with this version. "
            "Please consider upgrading to Python 3.10 or higher.",
        )
```