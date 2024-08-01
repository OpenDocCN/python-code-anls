# `.\DB-GPT-src\dbgpt\util\code_utils.py`

```py
import logging  # 导入日志模块
import os  # 导入操作系统功能模块
import pathlib  # 导入路径操作模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块
import time  # 导入时间模块
from concurrent.futures import ThreadPoolExecutor, TimeoutError  # 导入线程池和超时错误
from hashlib import md5  # 导入MD5哈希模块
from typing import Callable, Dict, List, Optional, Tuple, Union  # 导入类型注解

# 正则表达式，用于匹配代码块
# ```[ \t]*(\w+)?[ \t]*\r?\n(.*?)[ \t]*\r?\n```py 匹配多行代码块。
#   [ \t]* 匹配语言名前的可能空格或制表符。
#   (\w+)? 匹配语言名，? 表示可选。
#   [ \t]* 匹配语言名后可能的空格（非换行符）。
#   \r?\n 确保 ``` 后有换行符。
#   (.*?) 匹配代码本身（非贪婪模式）。
#   \r?\n 确保 ```py 前有换行符。
#   [ \t]* 匹配 ``` 关闭前可能的缩进（规范允许缩进）。
CODE_BLOCK_PATTERN = r"```py[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"

WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extensions")  # 设置工作目录为脚本文件所在目录下的 extensions 子目录
UNKNOWN = "unknown"  # 定义未知字符串常量
TIMEOUT_MSG = "Timeout"  # 定义超时消息字符串常量
DEFAULT_TIMEOUT = 60  # 定义默认超时时间为60秒
WIN32 = sys.platform == "win32"  # 检测是否运行在Windows平台
PATH_SEPARATOR = WIN32 and "\\" or "/"  # 根据平台设置路径分隔符

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def content_str(content: Union[str, List]) -> str:
    """Convert content to a string representation.

    Args:
        content (Union[str, List]): The content to convert, can be a string or list.

    Returns:
        str: The concatenated string representation of the content.
    """
    if type(content) is str:
        return content
    rst = ""
    for item in content:
        if item["type"] == "text":
            rst += item["text"]
        else:
            assert (
                isinstance(item, dict) and item["type"] == "image_url"
            ), "Wrong content format."
            rst += "<image>"
    return rst


def infer_lang(code):
    """Infer the programming language of the given code snippet.

    Args:
        code (str): The code snippet to infer language for.

    Returns:
        str: The inferred programming language.
    """
    if (
        code.startswith("python ")
        or code.startswith("pip")
        or code.startswith("python3 ")
    ):
        return "sh"  # 如果代码以 "python ", "pip" 或 "python3 " 开头，则推断为 shell 脚本语言

    # 检查代码是否是有效的Python代码
    try:
        compile(code, "test", "exec")
        return "python"  # 如果可以编译通过，则推断为Python语言
    except SyntaxError:
        # 不是有效的Python代码
        return UNKNOWN  # 返回未知语言


# TODO: In the future move, to better support https://spec.commonmark.org/0.30/#fenced-code-blocks
#       perhaps by using a full Markdown parser.
def extract_code(
    text: Union[str, List],
    pattern: str = CODE_BLOCK_PATTERN,
    detect_single_line_code: bool = False,
    default_lang: str = "python",
) -> List[Tuple[str, str]]:
    """Extract code blocks from the given text.

    Args:
        text (Union[str, List]): The text to extract code blocks from. Can be a string or a list.
        pattern (str, optional): The regex pattern to find code blocks. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable extraction of single line code blocks. Defaults to False.
        default_lang (str, optional): The default language to use when language cannot be determined.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the language and corresponding code block.
    """
    # 实现代码未提供，预留作为TODO注释
    """
    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is a code block but the language is not specified, the language would be "".
    """
    # 对输入的文本进行预处理，确保文本是字符串类型
    text = content_str(text)
    
    # 如果不需要检测单行代码块，则进行正则表达式匹配以找出所有代码块
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        # 如果找到代码块，则返回匹配结果；否则返回默认语言和原始文本组成的列表
        return match if match else [(default_lang, text)]

    # 定义用于匹配多行和单行代码块的正则表达式模式，以及内联代码的模式
    # CODE_BLOCK_PATTERN 是一个包含多行和单行代码块的正则表达式模式
    # `([^`]+)`: 匹配内联代码块
    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    # 在文本中查找所有匹配的代码块
    code_blocks = code_pattern.findall(text)

    # 提取每个匹配组中的语言和代码块内容，并存储在 extracted 列表中
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    # 返回提取的代码块列表，其中每个元素是一个元组，包含语言和代码内容
    return extracted
if __name__ == "__main__":
    # 当脚本作为主程序执行时，执行以下操作

    # 打印提取的代码结果
    print(
        extract_code(
            """```py import requests from bs4 import BeautifulSoup from datetime import datetime, timedelta  # Define the search query query = "LLM application"  # Define the time range (last week) end_date = datetime.now().strftime("%Y-%m-%d") start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Create the search URL url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={query}&terms-0-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=specific_date&date-year=&date-from_date={start_date}&date-to_date={end_date}&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"  # Send a GET request to the search URL response = requests.get(url)  # Parse the HTML content soup = BeautifulSoup(response.content, "html.parser")  # Find all the paper titles and authors titles = soup.find_all("p", class_="title is-5 mathjax") authors = soup.find_all("p", class_="authors")  # Print the results for i in range(len(titles)):     print(f"Title: {titles[i].text.strip()}")     print(f"Authors: {authors[i].text.strip()}")     print("-------------------------") ```  This code uses the `requests` library to send a GET request to the advanced search page of arXiv. It searches for papers with the specified query ("LLM application") that were submitted in the last week. The code then uses `BeautifulSoup` to parse the HTML content of the search results page and extracts the paper titles and authors. Finally, it prints the titles and authors of the found papers."""
        )
    )

# 用于改进函数配置的字典
_IMPROVE_FUNCTION_CONFIG = {
    "prompt": """Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}""",
    "model": "DEFAULT_MODEL",
    "request_timeout": 600,
}

# 用于改进代码配置的字典
_IMPROVE_CODE_CONFIG = {
    "prompt": """Analyze the code in the following files and return a list of suggestions for improvement{followup}, to achieve the objective of '{objective}'.
{code}
""",
    "model": "DEFAULT_MODEL",
    "request_timeout": 900,
}

def timeout_handler(signum, frame):
    # 超时处理函数，当超时时抛出 TimeoutError
    raise TimeoutError("Timed out!")

def _cmd(lang):
    # 根据语言选择执行命令
    if lang.startswith("python") or lang in ["bash", "sh", "powershell"]:
        return lang
    if lang in ["shell"]:
        return "sh"
    if lang in ["ps1"]:
        return "powershell"
    # 抛出异常，提示不支持的语言
    raise NotImplementedError(f"{lang} not recognized in code execution")

def execute_code(
    code: Optional[str] = None,
    timeout: Optional[int] = None,
    filename: Optional[str] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[Union[List[str], str, bool]] = None,
    lang: Optional[str] = "python",
) -> Tuple[int, str, str]:
    """Execute code in a docker container.
    This function is not tested on MacOS.
    # 检查是否同时未提供 code 和 filename，若是则记录错误并抛出异常
    if all((code is None, filename is None)):
        error_msg = f"Either {code=} or {filename=} must be provided."
        logger.error(error_msg)
        raise AssertionError(error_msg)

    # 如果未指定 use_docker（默认为 None），并且 Docker 包不可用，则发出警告
    # 此时默认行为是以本地方式运行代码，但此行为可能会发生变化。
    try:
        # 尝试导入 Docker 模块
        import docker
        
        # 检查 Docker 模块是否正确导入，若导入错误，则将 docker 设为 None
        try:
            docker.version
        except AttributeError:
            docker = None
    except ImportError:
        # 若未安装 Docker 模块，则将 docker 设为 None
        docker = None
    # 检查是否未指定 use_docker
    if use_docker is None:
        # 如果未安装 docker 包，则默认不使用 Docker
        if docker is None:
            use_docker = False
            # 记录警告日志，说明未指定 use_docker，且未安装 python docker 包，代码将在本地环境中运行
            logger.warning(
                "execute_code was called without specifying a value for use_docker. Since the python docker package is not available, code will be run natively. Note: this fallback behavior is subject to change"
            )
        else:
            # 如果安装了 docker 包，默认使用 Docker
            use_docker = True
    
    # 设置超时时间，默认使用全局设置中的超时时间
    timeout = timeout or DEFAULT_TIMEOUT
    # 保存原始文件名
    original_filename = filename
    
    # 如果在 Windows 平台且语言为 shell 类型且不使用 Docker，则将语言设置为 PowerShell
    if WIN32 and lang in ["sh", "shell"] and (not use_docker):
        lang = "ps1"
    
    # 如果未指定文件名，则根据代码内容生成一个临时文件名
    if filename is None:
        code_hash = md5(code.encode()).hexdigest()
        # 使用自动生成的文件名来创建临时文件
        filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
    
    # 如果未指定工作目录，则使用默认工作目录
    if work_dir is None:
        work_dir = WORKING_DIR
    
    # 构建文件路径，并创建文件所在目录（如果不存在）
    filepath = os.path.join(work_dir, filename)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)
    
    # 如果有代码内容，则将代码写入文件中
    if code is not None:
        with open(filepath, "w", encoding="utf-8") as fout:
            fout.write(code)
    
    # 检查是否正在运行在 Docker 容器中
    in_docker_container = os.path.exists("/.dockerenv")
    
    # 如果不使用 Docker 或者已经在 Docker 容器中运行
    if not use_docker or in_docker_container:
        # 已经在 Docker 容器中运行
        # 构建执行命令
        cmd = [
            sys.executable if lang.startswith("python") else _cmd(lang),
            f".\\{filename}" if WIN32 else filename,
        ]
        
        # 如果在 Windows 下，记录警告日志，因为 Windows 不支持 SIGALRM 信号，无法强制执行超时
        if WIN32:
            logger.warning(
                "SIGALRM is not supported on Windows. No timeout will be enforced."
            )
            
            # 在 Windows 下执行命令
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
        else:
            # 在其他平台下，使用线程池执行命令，并设置超时时间
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    subprocess.run,
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                try:
                    result = future.result(timeout=timeout)
                except TimeoutError:
                    # 如果超时，删除临时文件并返回超时错误
                    if original_filename is None:
                        os.remove(filepath)
                    return 1, TIMEOUT_MSG, None
        
        # 如果未指定原始文件名，则删除临时文件
        if original_filename is None:
            os.remove(filepath)
        
        # 如果执行结果返回非零状态码，则记录错误日志
        if result.returncode:
            logs = result.stderr
            # 如果未指定原始文件名，修正日志中的文件路径信息
            if original_filename is None:
                abs_path = str(pathlib.Path(filepath).absolute())
                logs = logs.replace(str(abs_path), "").replace(filename, "")
            else:
                abs_path = str(pathlib.Path(work_dir).absolute()) + PATH_SEPARATOR
                logs = logs.replace(str(abs_path), "")
        else:
            # 执行成功，获取标准输出作为日志
            logs = result.stdout
        
        # 返回执行结果状态码、日志和空值
        return result.returncode, logs, None
    
    # 如果需要使用 Docker，则创建 Docker 客户端连接
    client = docker.from_env()
    # 根据 use_docker 变量的类型确定要使用的 Docker 镜像列表
    image_list = (
        ["python:3-alpine", "python:3", "python:3-windowsservercore"]
        if use_docker is True  # 如果 use_docker 是布尔型 True，则使用默认的镜像列表
        else [use_docker]  # 如果 use_docker 是字符串，则使用单个镜像
        if isinstance(use_docker, str)  # 如果 use_docker 是字符串类型
        else use_docker  # 否则使用 use_docker 中提供的列表
    )
    for image in image_list:
        # 检查镜像是否存在
        try:
            client.images.get(image)
            break  # 如果镜像存在，则终止循环
        except docker.errors.ImageNotFound:
            # 如果镜像不存在，则尝试拉取镜像
            print("Pulling image", image)
            try:
                client.images.pull(image)
                break  # 如果成功拉取镜像，则终止循环
            except docker.errors.DockerException:
                print("Failed to pull image", image)
    # 生成一个基于当前时间的随机字符串，用于包装退出码
    exit_code_str = f"exitcode{time.time()}"
    # 获取工作目录的绝对路径
    abs_path = pathlib.Path(work_dir).absolute()
    # 构造命令列表，用于在 Docker 容器中执行脚本
    cmd = [
        "sh",
        "-c",
        f"{_cmd(lang)} {filename}; exit_code=$?; echo -n {exit_code_str}; echo -n $exit_code; echo {exit_code_str}",
    ]
    # 创建一个 Docker 容器实例
    container = client.containers.run(
        image,
        command=cmd,
        working_dir="/workspace",
        detach=True,  # 在后台运行容器
        # 将工作目录绑定到容器中，以读写模式挂载
        volumes={abs_path: {"bind": "/workspace", "mode": "rw"}},
    )
    start_time = time.time()
    # 在规定的超时时间内等待容器执行完毕
    while container.status != "exited" and time.time() - start_time < timeout:
        container.reload()  # 重新加载容器状态
    if container.status != "exited":
        # 如果超时仍未退出，则停止并删除容器
        container.stop()
        container.remove()
        if original_filename is None:
            os.remove(filepath)  # 如果没有指定原始文件名，则删除文件
        return 1, TIMEOUT_MSG, image  # 返回超时退出码、超时消息和使用的镜像名称
    # 获取容器的日志信息
    logs = container.logs().decode("utf-8").rstrip()
    # 提取文件名中的斜杠并替换为空格
    tag = filename.replace("/", "")
    # 提交容器为新的镜像
    container.commit(repository="python", tag=tag)
    # 删除容器
    container.remove()
    # 检查容器的退出码
    exit_code = container.attrs["State"]["ExitCode"]
    if exit_code == 0:
        # 从日志中提取退出码
        pattern = re.compile(f"{exit_code_str}(\\d+){exit_code_str}")
        match = pattern.search(logs)
        exit_code = 1 if match is None else int(match.group(1))
        # 从日志中删除退出码信息
        logs = logs if match is None else pattern.sub("", logs)

    if original_filename is None:
        os.remove(filepath)  # 如果没有指定原始文件名，则删除文件
    if exit_code:
        # 删除日志中的工作目录路径信息
        logs = logs.replace(
            f"/workspace/{filename if original_filename is None else ''}", ""
        )
    # 返回最终的退出码、日志和使用的镜像名称
    return exit_code, logs, f"python:{tag}"
_GENERATE_ASSERTIONS_CONFIG = {
    "prompt": """Given the signature and docstring, write the exactly same number of assertion(s) for the provided example(s) in the docstring, without assertion messages.

func signature:
{definition}
assertions:""",
    "model": "FAST_MODEL",
    "max_tokens": 256,
    "stop": "\n\n",
}

def _remove_check(response):
    """Remove the check function from the response."""
    # find the position of the check function
    pos = response.find("def check(")
    if pos == -1:
        return response
    return response[:pos]


def eval_function_completions(
    responses: List[str],
    definition: str,
    test: Optional[str] = None,
    entry_point: Optional[str] = None,
    assertions: Optional[Union[str, Callable[[str], Tuple[str, float]]]] = None,
    timeout: Optional[float] = 3,
    use_docker: Optional[bool] = True,
) -> Dict:
    """(openai<1) Select a response from a list of responses for the function completion task (using generated assertions), and/or evaluate if the task is successful using a gold test.

    Args:
        responses (list): The list of responses.
        definition (str): The input definition.
        test (Optional, str): The test code.
        entry_point (Optional, str): The name of the function.
        assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.
            When provided, only the responses that pass the assertions will be considered for the actual test (if provided).
        timeout (Optional, float): The timeout for executing the code.

    Returns:
        dict: The success metrics.
    """
    # Determine the number of responses
    n = len(responses)
    
    # If no assertions provided, evaluate all responses
    if assertions is None:
        success_list = []
        # Iterate through each response
        for i in range(n):
            # Remove any 'check' function from the response
            response = _remove_check(responses[i])
            # Construct the code to execute
            code = (
                f"{response}\n{test}\ncheck({entry_point})"
                if response.startswith("def")
                else f"{definition}{response}\n{test}\ncheck({entry_point})"
            )
            # Execute the constructed code and check success
            success = execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
            success_list.append(success)
        
        # Calculate and return success metrics
        return {
            "expected_success": 1 - pow(1 - sum(success_list) / n, n),
            "success": any(s for s in success_list),
        }
    
    # If assertions is callable and multiple responses, generate assertions
    if callable(assertions) and n > 1:
        assertions, gen_cost = assertions(definition)
    else:
        assertions, gen_cost = None, 0
    # 如果 n 大于 1 或者 test 为 None，则执行以下逻辑
    if n > 1 or test is None:
        # 循环 n 次
        for i in range(n):
            # 对 responses[i] 应用 _remove_check 函数，并将结果保存到 responses[i] 中
            response = responses[i] = _remove_check(responses[i])
            # 根据 response 的开头判断，构建不同的代码段
            code = (
                f"{response}\n{assertions}"
                if response.startswith("def")
                else f"{definition}{response}\n{assertions}"
            )
            # 执行构建的代码段，并检查执行结果是否成功
            succeed_assertions = (
                execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
            )
            # 如果执行成功，则跳出循环
            if succeed_assertions:
                break
    else:
        # 如果 n <= 1 且 test 不为 None，则执行以下逻辑
        # 只是测试，无需检查断言
        succeed_assertions = False
        i, response = 0, responses[0]
    
    # 如果 test 为 None，则执行以下逻辑
    if test is None:
        # 没有测试代码，返回以下结果字典
        return {
            "index_selected": i,
            "succeed_assertions": succeed_assertions,
            "gen_cost": gen_cost,
            "assertions": assertions,
        }
    
    # 构建包含测试代码的代码段
    code_test = (
        f"{response}\n{test}\ncheck({entry_point})"
        if response.startswith("def")
        else f"{definition}{response}\n{test}\ncheck({entry_point})"
    )
    # 执行包含测试代码的代码段，并检查执行结果是否成功
    success = execute_code(code_test, timeout=timeout, use_docker=use_docker)[0] == 0
    
    # 返回以下结果字典
    return {
        "index_selected": i,
        "succeed_assertions": succeed_assertions,
        "success": success,
        "gen_cost": gen_cost,
        "assertions": assertions,
    }
# 设置用于Python函数完成的提示信息，在这里是针对Python 3的定义。
_FUNC_COMPLETION_PROMPT = "# Python 3{definition}"

# 定义触发停止完成的关键词列表，包括class、def、if和print，每个关键词前面有换行符。
_FUNC_COMPLETION_STOP = ["\nclass", "\ndef", "\nif", "\nprint"]
```