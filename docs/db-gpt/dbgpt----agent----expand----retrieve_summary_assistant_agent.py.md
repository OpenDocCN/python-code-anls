# `.\DB-GPT-src\dbgpt\agent\expand\retrieve_summary_assistant_agent.py`

```py
"""Retrieve Summary Assistant Agent."""

# 导入所需的模块
import glob  # 用于查找文件路径模式匹配
import json  # 用于 JSON 数据的解析和生成
import logging  # 日志记录模块
import os  # 提供与操作系统交互的功能
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 强类型注解相关的导入
from urllib.parse import urlparse  # URL 解析模块

from dbgpt.configs.model_config import PILOT_PATH  # 从 dbgpt 包中导入 PILOT_PATH 变量
from dbgpt.core import ModelMessageRoleType  # 从 dbgpt 包中导入 ModelMessageRoleType 类

from ..core.action.base import Action, ActionOutput  # 导入核心功能的基础类和动作输出
from ..core.agent import Agent, AgentMessage, AgentReviewInfo  # 导入代理、消息和评审信息类
from ..core.base_agent import ConversableAgent  # 导入可对话代理的基类
from ..core.profile import ProfileConfig  # 导入配置文件类
from ..resource.base import AgentResource  # 导入代理资源类
from ..util.cmp import cmp_string_equal  # 导入字符串比较函数

try:
    from unstructured.partition.auto import partition  # 尝试导入未结构化数据处理的分区功能

    HAS_UNSTRUCTURED = True  # 设置标志指示是否成功导入未结构化功能
except ImportError:
    HAS_UNSTRUCTURED = False  # 设置标志指示未导入未结构化功能

logger = logging.getLogger()  # 获取全局日志记录器对象

TEXT_FORMATS = [  # 文本文件格式列表
    "txt", "json", "csv", "tsv", "md", "html", "htm", "rtf", "rst", "jsonl", "log", "xml", "yaml", "yml", "pdf"
]

UNSTRUCTURED_FORMATS = [  # 未结构化文件格式列表，如果安装了 'unstructured' 库则会被处理
    "doc", "docx", "epub", "msg", "odt", "org", "pdf", "ppt", "pptx", "rtf", "rst", "xlsx"
]

if HAS_UNSTRUCTURED:
    TEXT_FORMATS += UNSTRUCTURED_FORMATS  # 如果有未结构化支持，则将未结构化格式添加到文本格式列表中
    TEXT_FORMATS = list(set(TEXT_FORMATS))  # 去除重复格式并转换为列表

VALID_CHUNK_MODES = frozenset({"one_line", "multi_lines"})  # 有效的文本块模式集合


def _get_max_tokens(model="gpt-3.5-turbo"):
    """Get the maximum number of tokens for a given model."""
    if "32k" in model:  # 如果模型名称中包含 '32k'，返回 32000
        return 32000
    elif "16k" in model:  # 如果模型名称中包含 '16k'，返回 16000
        return 16000
    elif "gpt-4" in model:  # 如果模型名称中包含 'gpt-4'，返回 8000
        return 8000
    else:  # 默认情况下返回 4000
        return 4000


_NO_RESPONSE = "NO RELATIONSHIP.UPDATE TEXT CONTENT."


class RetrieveSummaryAssistantAgent(ConversableAgent):
    """Assistant agent, designed to solve a task with LLM.

    AssistantAgent is a subclass of ConversableAgent configured with a default
    system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    """
    # 定义包含用户问题和提供文本内容的提示信息
    PROMPT_QA: str = (
        "You are a great summary writer to summarize the provided text content "
        "according to user questions.\n"
        "User's Question is: {input_question}\n\n"
        "Provided text content is: {input_context}\n\n"
        "Please complete this task step by step following instructions below:\n"
        "   1. You need to first detect user's question that you need to answer with "
        "your summarization.\n"
        "   2. Then you need to summarize the provided text content that ONLY CAN "
        "ANSWER user's question and filter useless information as possible as you can. "
        "YOU CAN ONLY USE THE PROVIDED TEXT CONTENT!! DO NOT CREATE ANY SUMMARIZATION "
        "WITH YOUR OWN KNOWLEDGE!!!\n"
        "   3. Output the content of summarization that ONLY CAN ANSWER user's question"
        " and filter useless information as possible as you can. The output language "
        "must be the same to user's question language!! You must give as short an "
        "summarization as possible!!! DO NOT CREATE ANY SUMMARIZATION WITH YOUR OWN "
        "KNOWLEDGE!!!\n\n"
        "####Important Notice####\n"
        "If the provided text content CAN NOT ANSWER user's question, ONLY output "
        "'NO RELATIONSHIP.UPDATE TEXT CONTENT.'!!."
    )
    
    # 定义检查结果系统消息，用于分析摘要任务结果
    CHECK_RESULT_SYSTEM_MESSAGE: str = (
        "You are an expert in analyzing the results of a summary task."
        "Your responsibility is to check whether the summary results can summarize the "
        "input provided by the user, and then make a judgment. You need to answer "
        "according to the following rules:\n"
        "    Rule 1: If you think the summary results can summarize the input provided"
        " by the user, only return True.\n"
        "    Rule 2: If you think the summary results can NOT summarize the input "
        "provided by the user, return False and the reason, split by | and ended "
        "by TERMINATE. For instance: False|Some important concepts in the input are "
        "not summarized. TERMINATE"
    )

    # 默认描述信息，用于根据用户问题和提供的文件路径进行摘要
    DEFAULT_DESCRIBE: str = (
        "Summarize provided content according to user's questions and "
        "the provided file paths."
    )
    
    # 配置文件信息，包括名称、角色、目标和描述
    profile: ProfileConfig = ProfileConfig(
        name="RetrieveSummarizer",
        role="Assistant",
        goal="You're an extraction expert. You need to extract Please complete this "
        "task step by step following instructions below:\n"
        "   1. You need to first ONLY extract user's question that you need to answer "
        "without ANY file paths and URLs. \n"
        "   2. Extract the provided file paths and URLs.\n"
        "   3. Construct the extracted file paths and URLs as a list of strings.\n"
        "   4. ONLY output the extracted results with the following json format: "
        "{{ response }}.",
        desc=DEFAULT_DESCRIBE,
    )
    
    # 定义分块的令牌大小和模式
    chunk_token_size: int = 4000
    chunk_mode: str = "multi_lines"
    
    # 模型名称
    _model: str = "gpt-3.5-turbo-16k"
    # 使用 _get_max_tokens 函数获取模型的最大令牌数，并赋值给 _max_tokens 变量
    _max_tokens: int = _get_max_tokens(_model)
    # 计算上下文最大令牌数为 _max_tokens 的 80%
    context_max_tokens: int = int(_max_tokens * 0.8)

    def __init__(
        self,
        **kwargs,
    ):
        """Create a new instance of the agent."""
        # 调用父类的初始化方法，传入任意关键字参数
        super().__init__(
            **kwargs,
        )
        # 初始化代理操作，包括 SummaryAction
        self._init_actions([SummaryAction])

    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        # 调用父类的方法初始化回复消息
        reply_message = super()._init_reply_message(received_message)
        # 创建一个包含用户问题和文件列表的 JSON 数据
        json_data = {"user_question": "user's question", "file_list": "file&URL list"}
        # 将 JSON 数据转换为字符串，并添加到回复消息的上下文中
        reply_message.context = {"response": json.dumps(json_data, ensure_ascii=False)}
        return reply_message

    async def generate_reply(
        self,
        received_message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        **kwargs,
    ):
        # 异步方法，用于生成回复消息，接收消息、发送者代理、审核者代理等参数
        ...

    async def correctness_check(
        self, message: AgentMessage
        # 异步方法，用于执行正确性检查，接收消息对象作为参数
    # 定义一个方法，用于验证结果的正确性，返回一个布尔值和一个可选的失败原因字符串。
        ) -> Tuple[bool, Optional[str]]:
            """Verify the correctness of the results."""
            # 获取消息中的动作报告
            action_report = message.action_report
            # 初始化任务结果字符串
            task_result = ""
            # 如果存在动作报告，则从中获取内容作为任务结果
            if action_report:
                task_result = action_report.get("content", "")
    
            # 调用 self.thinking 方法进行处理
            check_result, model = await self.thinking(
                # 向模型发送消息，请求判断结果
                messages=[
                    AgentMessage(
                        role=ModelMessageRoleType.HUMAN,
                        content=(
                            "Please understand the following user input and summary results"
                            " and give your judgment:\n"
                            f"User Input: {message.current_goal}\n"
                            f"Summary Results: {task_result}"
                        ),
                    )
                ],
                # 设置提示信息为 CHECK_RESULT_SYSTEM_MESSAGE
                prompt=self.CHECK_RESULT_SYSTEM_MESSAGE,
            )
            # 初始化失败原因字符串
            fail_reason = ""
            # 根据判断结果设置成功状态和失败原因
            if check_result and (
                "true" in check_result.lower() or "yes" in check_result.lower()
            ):
                success = True
            elif not check_result:
                success = False
                # 如果判断结果为空，则设置失败状态和原因
                fail_reason = (
                    "The summary results cannot summarize the user input. "
                    "Please re-understand and complete the summary task."
                )
            else:
                success = False
                try:
                    # 尝试解析判断结果中的失败原因
                    _, fail_reason = check_result.split("|")
                    # 格式化失败原因字符串
                    fail_reason = (
                        "The summary results cannot summarize the user input due"
                        f" to: {fail_reason}. Please re-understand and complete the summary"
                        " task."
                    )
                except Exception:
                    # 如果解析失败，则记录警告日志
                    logger.warning(
                        "The model thought the results are irrelevant but did not give the"
                        " correct format of results."
                    )
                    # 设置默认的失败原因字符串
                    fail_reason = (
                        "The summary results cannot summarize the user input. "
                        "Please re-understand and complete the summary task."
                    )
            # 返回最终的成功状态和失败原因字符串
            return success, fail_reason
    
        # 定义一个方法，用于从指定目录中获取文件列表
        def _get_files_from_dir(
            self,
            dir_path: Union[str, List[str]],
            types: list = TEXT_FORMATS,
            recursive: bool = True,
    ):
        """Return a list of all the files in a given directory.

        A url, a file path or a list of them.
        """
        # 检查 types 是否为空，若为空则抛出数值错误异常
        if len(types) == 0:
            raise ValueError("types cannot be empty.")
        
        # 将 types 中的文件类型统一转换为小写，并加入对应的大写形式，放入集合中
        types = [t[1:].lower() if t.startswith(".") else t.lower() for t in set(types)]
        types += [t.upper() for t in types]

        # 初始化文件列表
        files = []

        # 如果 dir_path 是一个文件列表或 URL 列表，则逐个处理并返回文件列表
        if isinstance(dir_path, list):
            for item in dir_path:
                # 如果 item 是文件路径，则直接加入文件列表
                if os.path.isfile(item):
                    files.append(item)
                # 如果 item 是 URL，则下载文件并加入文件列表
                elif self._is_url(item):
                    files.append(self._get_file_from_url(item))
                # 如果 item 是存在的路径，则尝试递归获取文件列表
                elif os.path.exists(item):
                    try:
                        files.extend(self._get_files_from_dir(item, types, recursive))
                    except ValueError:
                        logger.warning(f"Directory {item} does not exist. Skipping.")
                else:
                    logger.warning(f"File {item} does not exist. Skipping.")
            return files

        # 如果 dir_path 是一个文件路径，则直接返回包含该路径的列表
        if os.path.isfile(dir_path):
            return [dir_path]

        # 如果 dir_path 是一个 URL，则下载文件并返回包含该文件的列表
        if self._is_url(dir_path):
            return [self._get_file_from_url(dir_path)]

        # 如果 dir_path 是一个存在的路径，则根据文件类型和递归标志获取文件列表
        if os.path.exists(dir_path):
            for type in types:
                if recursive:
                    # 使用 glob 模块获取匹配的文件列表（支持递归）
                    files += glob.glob(
                        os.path.join(dir_path, f"**/*.{type}"), recursive=True
                    )
                else:
                    # 使用 glob 模块获取匹配的文件列表（不递归）
                    files += glob.glob(
                        os.path.join(dir_path, f"*.{type}"), recursive=False
                    )
        else:
            # 若 dir_path 路径不存在，则记录错误并抛出数值错误异常
            logger.error(f"Directory {dir_path} does not exist.")
            raise ValueError(f"Directory {dir_path} does not exist.")
        
        # 返回最终的文件列表
        return files
    def _get_file_from_url(self, url: str, save_path: Optional[str] = None):
        """Download a file from a URL."""
        import requests  # 导入requests库，用于发送HTTP请求
        from bs4 import BeautifulSoup  # 导入BeautifulSoup库，用于解析HTML内容

        if save_path is None:
            # 如果save_path未指定，则保存到默认目录下的"data"文件夹中
            target_directory = os.path.join(PILOT_PATH, "data")
            os.makedirs(target_directory, exist_ok=True)
            save_path = os.path.join(target_directory, os.path.basename(url))
        else:
            # 如果指定了save_path，则确保其父文件夹存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        proxies: Dict[str, Any] = {}
        # 如果环境变量中有http_proxy或https_proxy，则设置相应的代理
        if os.getenv("http_proxy"):
            proxies["http"] = os.getenv("http_proxy")
        if os.getenv("https_proxy"):
            proxies["https"] = os.getenv("https_proxy")

        with requests.get(url, proxies=proxies, timeout=10, stream=True) as r:
            r.raise_for_status()  # 检查是否有HTTP错误，如果有则抛出异常
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)  # 将下载的文件内容写入本地文件

        with open(save_path, "r", encoding="utf-8") as file:
            html_content = file.read()  # 读取下载的HTML文件内容

        soup = BeautifulSoup(html_content, "html.parser")  # 创建BeautifulSoup对象解析HTML内容

        # 从Beautiful Soup对象中提取所有<p>标签，获取段落文本
        paragraphs = soup.find_all("p")

        # 将解析后的段落文本重新写入到相同的save_path文件中
        with open(save_path, "w", encoding="utf-8") as f:
            for paragraph in paragraphs:
                f.write(paragraph.get_text() + "\n")  # 将段落文本写入文件，每个段落占一行

        return save_path  # 返回保存的文件路径

    def _is_url(self, string: str):
        """Return True if the string is a valid URL."""
        try:
            result = urlparse(string)  # 解析URL字符串
            return all([result.scheme, result.netloc])  # 检查解析结果中是否包含scheme和netloc
        except ValueError:
            return False  # 如果解析失败，则返回False

    async def _split_text_to_chunks(
        self,
        text: str,
        chunk_mode: str = "multi_lines",
        must_break_at_empty_line: bool = True,
        chunk_length: int = 1000
    ):
        """Split text into chunks based on specified parameters."""
        # 这里可以实现将文本根据指定参数拆分成多个片段的功能，具体实现根据函数的调用和需求来决定
        pass  # 占位符，表示这里的功能实现留待后续完善
    ):
        """Split a long text into chunks of max_tokens."""
        # 设置每个块的最大标记数
        max_tokens = self.chunk_token_size
        # 检查分块模式是否有效
        if chunk_mode not in VALID_CHUNK_MODES:
            raise AssertionError
        # 如果分块模式是 "one_line"，则不需要在空行处分割
        if chunk_mode == "one_line":
            must_break_at_empty_line = False
        # 初始化空的块列表
        chunks = []
        # 将文本按行分割
        lines = text.split("\n")
        # 计算每行的标记数
        lines_tokens = [await self._count_token(line) for line in lines]
        # 计算所有行的总标记数
        sum_tokens = sum(lines_tokens)
        # 当总标记数超过最大标记数时执行循环
        while sum_tokens > max_tokens:
            # 根据分块模式确定预估的行数切割
            if chunk_mode == "one_line":
                estimated_line_cut = 2
            else:
                estimated_line_cut = int(max_tokens / sum_tokens * len(lines)) + 1
            cnt = 0
            prev = ""
            # 反向遍历预估的行数切割
            for cnt in reversed(range(estimated_line_cut)):
                # 如果需要在空行处分割，并且当前行不是空行，则继续循环
                if must_break_at_empty_line and lines[cnt].strip() != "":
                    continue
                # 如果当前行之前的所有行的标记数总和小于等于最大标记数，则记录前面的文本
                if sum(lines_tokens[:cnt]) <= max_tokens:
                    prev = "\n".join(lines[:cnt])
                    break
            # 如果没有找到合适的切割点，记录警告信息并尝试其他策略
            if cnt == 0:
                logger.warning(
                    f"max_tokens is too small to fit a single line of text. Breaking "
                    f"this line:\n\t{lines[0][:100]} ..."
                )
                if not must_break_at_empty_line:
                    # 计算新的切割长度
                    split_len = int(max_tokens / lines_tokens[0] * 0.9 * len(lines[0]))
                    # 切割文本并更新相关信息
                    prev = lines[0][:split_len]
                    lines[0] = lines[0][split_len:]
                    lines_tokens[0] = await self._count_token(lines[0])
                else:
                    # 如果必须在空行处分割文档但失败，则设置为不需要在空行处分割
                    logger.warning(
                        "Failed to split docs with must_break_at_empty_line being True,"
                        " set to False."
                    )
                    must_break_at_empty_line = False
            (
                chunks.append(prev) if len(prev) > 10 else None
            )  # 不添加长度小于10的块
            # 更新剩余的行和标记数
            lines = lines[cnt:]
            lines_tokens = lines_tokens[cnt:]
            sum_tokens = sum(lines_tokens)
        # 将剩余的文本作为一个块添加到列表中
        text_to_chunk = "\n".join(lines)
        (
            chunks.append(text_to_chunk) if len(text_to_chunk) > 10 else None
        )  # 不添加长度小于10的块
        # 返回所有的块
        return chunks
    async def _extract_text_from_pdf(self, file: str) -> str:
        """Extract text from PDF files."""
        text = ""
        import pypdf  # 导入 PyPDF 模块用于处理 PDF 文件

        with open(file, "rb") as f:
            reader = pypdf.PdfReader(f)  # 使用 PyPDF 打开 PDF 文件
            if reader.is_encrypted:  # 检查 PDF 是否已加密
                try:
                    reader.decrypt("")  # 尝试解密 PDF 文件
                except pypdf.errors.FileNotDecryptedError as e:
                    logger.warning(f"Could not decrypt PDF {file}, {e}")
                    return text  # 如果解密失败，记录警告并返回空文本

            for page_num in range(len(reader.pages)):  # 遍历 PDF 的每一页
                page = reader.pages[page_num]
                text += page.extract_text()  # 提取每页的文本内容并拼接到 text 变量中

        if not text.strip():  # 调试用：检查提取的文本是否为空
            logger.warning(f"Could not decrypt PDF {file}")

        return text  # 返回提取的文本内容

    async def _split_files_to_chunks(
        self,
        files: list,
        chunk_mode: str = "multi_lines",
        must_break_at_empty_line: bool = True,
        custom_text_split_function: Optional[Callable] = None,
    ):
        """Split a list of files into chunks of max_tokens."""
        chunks = []

        for file in files:
            _, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if HAS_UNSTRUCTURED and file_extension[1:] in UNSTRUCTURED_FORMATS:
                text = partition(file)  # 对非结构化格式文件进行处理
                text = "\n".join([t.text for t in text]) if len(text) > 0 else ""  # 将处理后的文本连接成一个字符串
            elif file_extension == ".pdf":
                text = self._extract_text_from_pdf(file)  # 提取 PDF 文件的文本内容
            else:  # 对于非 PDF 的文本文件
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()  # 读取文本文件的内容

            if (
                not text.strip()
            ):  # 调试用：检查读取的文本内容是否为空
                logger.warning(f"No text available in file: {file}")
                continue  # 如果文本内容为空，则跳过当前文件

            if custom_text_split_function is not None:
                chunks += custom_text_split_function(text)  # 使用自定义的文本分割函数分割文本
            else:
                chunks += await self._split_text_to_chunks(
                    text, chunk_mode, must_break_at_empty_line
                )  # 使用内置的方法将文本分割成块

        return chunks  # 返回所有文件的分块内容列表

    async def _count_token(
        self, input: Union[str, List, Dict], model: str = "gpt-3.5-turbo-0613"
    ):
        """Count tokens in the input text."""
        # 实现计算输入文本中 token 数量的功能
    # 定义一个方法，用于计算 OpenAI 模型使用的令牌数目
    async def count_tokens(self, input: Union[str, list, dict], model: str) -> int:
        """Count number of tokens used by an OpenAI model.

        Args:
            input: (str, list, dict): Input to the model.
            model: (str): Model name.

        Returns:
            int: Number of tokens from the input.
        """
        # 获取非空的 OpenAI 模型客户端
        _llm_client = self.not_null_llm_client
        # 如果输入是字符串，则直接使用模型客户端计算令牌数目并返回
        if isinstance(input, str):
            return await _llm_client.count_token(model, input)
        # 如果输入是列表，则对列表中每个元素调用模型客户端计算令牌数目，并求和返回
        elif isinstance(input, list):
            return sum([await _llm_client.count_token(model, i) for i in input])
        # 如果输入既不是字符串也不是列表，则抛出值错误异常
        else:
            raise ValueError("input must be str or list")
class SummaryAction(Action[None]):
    """Simple Summary Action."""

    def __init__(self):
        """Create a new instance of the action."""
        super().__init__()

    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
    ) -> ActionOutput:
        """Perform the action."""
        
        # 初始化失败原因为 None
        fail_reason = None
        # 默认操作成功
        response_success = True
        # 视图和内容初始化为空
        view = None
        content = None
        
        # 如果 AI 消息为空
        if ai_message is None:
            # 设置失败原因为消息未被总结，提示用户检查输入
            fail_reason = "Nothing is summarized, please check your input."
            # 操作标记为失败
            response_success = False
        else:
            try:
                # 如果 AI 消息中包含 "NO RELATIONSHIP."
                if "NO RELATIONSHIP." in ai_message:
                    # 设置失败原因为内容与用户问题无关联，终止操作
                    fail_reason = (
                        "Return summarization error, the provided text "
                        "content has no relationship to user's question. TERMINATE."
                    )
                    # 操作标记为失败
                    response_success = False
                else:
                    # 将内容设为 AI 消息
                    content = ai_message
                    # 将视图设为内容
                    view = content
            except Exception as e:
                # 捕获到异常时，设置失败原因为总结出错并包含异常信息
                fail_reason = f"Return summarization error, {str(e)}"
                # 操作标记为失败
                response_success = False

        # 如果操作未成功，则将内容设为失败原因
        if not response_success:
            content = fail_reason
        
        # 返回操作结果对象，包括操作是否成功标志，内容和视图
        return ActionOutput(is_exe_success=response_success, content=content, view=view)
```