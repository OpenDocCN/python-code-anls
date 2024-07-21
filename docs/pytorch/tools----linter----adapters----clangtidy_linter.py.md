# `.\pytorch\tools\linter\adapters\clangtidy_linter.py`

```py
# 引入未来的注释语法支持，使得 Annotations 可以作为类型提示使用
from __future__ import annotations

# 引入用于命令行参数解析的模块
import argparse
# 引入用于并发任务处理的模块
import concurrent.futures
# 引入处理 JSON 数据的模块
import json
# 引入日志记录模块
import logging
# 引入操作系统相关功能的模块
import os
# 引入正则表达式处理模块
import re
# 引入文件和目录操作相关功能的模块
import shutil
# 引入子进程管理模块
import subprocess
# 引入系统相关功能的模块
import sys
# 引入时间相关功能的模块
import time
# 引入枚举类型的支持
from enum import Enum
# 引入处理文件路径相关功能的模块
from pathlib import Path
# 引入系统配置信息获取的函数
from sysconfig import get_paths as gp
# 引入类型提示相关功能
from typing import Any, NamedTuple

# PyTorch 根目录的确定函数
def scm_root() -> str:
    path = os.path.abspath(os.getcwd())
    while True:
        # 检查当前路径是否包含 .git 目录，如果是则返回该路径作为 SCM 根目录
        if os.path.exists(os.path.join(path, ".git")):
            return path
        # 检查当前路径是否包含 .hg 目录，如果是则返回该路径作为 SCM 根目录
        if os.path.isdir(os.path.join(path, ".hg")):
            return path
        # 记录当前路径长度
        n = len(path)
        # 将当前路径更新为其父目录
        path = os.path.dirname(path)
        # 如果路径没有变化，则抛出运行时错误，无法找到 SCM 根目录
        if len(path) == n:
            raise RuntimeError("Unable to find SCM root")

# 将 Python 包含目录的路径返回，格式为 '/usr/local/include/python<version number>'
def get_python_include_dir() -> str:
    return gp()["include"]

# 将内容输出到标准错误输出的函数
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)

# 代码检查严重性的枚举定义
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 代码检查消息的命名元组定义
class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None

# 将路径分隔符转换为 POSIX 风格的函数
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name

# 编译正则表达式，用于解析代码检查结果
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):      # 匹配文件名部分，以冒号结尾
    (?P<line>\d+):      # 匹配行号部分，以冒号结尾
    (?:(?P<column>-?\d+):)?  # 可选的匹配列号部分，以冒号结尾
    \s(?P<severity>\S+?):?  # 匹配严重性部分，以空白字符结尾，可能带冒号
    \s(?P<message>.*)   # 匹配消息部分，匹配剩余部分
    \s(?P<code>\[.*\])  # 匹配代码部分，以方括号包裹
    $
    """
)

# 运行命令的函数，返回一个完成的子进程对象
def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))  # 记录调试信息，显示执行的命令
    start_time = time.monotonic()  # 记录开始时间
    try:
        return subprocess.run(
            args,
            capture_output=True,  # 捕获命令的输出结果
            check=False,  # 不检查命令的返回状态
        )
    finally:
        end_time = time.monotonic()  # 记录结束时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 记录命令执行所花费的时间

# 将字符串严重性映射为枚举类型的字典
severities = {
    "error": LintSeverity.ERROR,
    "warning": LintSeverity.WARNING,
}

# 返回用于搜索 clang 的目录列表
def clang_search_dirs() -> list[str]:
    # 定义编译器的候选列表，按照优先级排序
    compilers = ["clang", "gcc", "cpp", "cc"]
    # 过滤出系统中存在的编译器
    compilers = [c for c in compilers if shutil.which(c) is not None]
    # 如果没有找到任何编译器，则抛出运行时错误
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    # 选择第一个找到的编译器
    compiler = compilers[0]

    # 运行编译器的命令，获取其搜索路径
    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    # 将标准错误输出解码为字符串，去除首尾空白字符，并按行拆分为列表
    stderr = result.stderr.decode().strip().split("\n")
    
    # 定义用于匹配搜索路径起始和结束的正则表达式模式
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    # 初始化变量，用于标记是否开始添加搜索路径和存储搜索路径的列表
    append_path = False
    search_paths = []

    # 遍历标准错误输出的每一行
    for line in stderr:
        # 如果当前行匹配搜索路径起始的正则表达式模式
        if re.match(search_start, line):
            # 如果已经在添加搜索路径，则继续下一次循环
            if append_path:
                continue
            else:
                # 否则，标记开始添加搜索路径
                append_path = True
        # 如果当前行匹配搜索路径结束的正则表达式模式
        elif re.match(search_end, line):
            # 结束循环，不再添加更多搜索路径
            break
        # 如果标记指示正在添加搜索路径
        elif append_path:
            # 将当前行去除首尾空白字符后，添加到搜索路径列表中
            search_paths.append(line.strip())

    # 返回存储了搜索路径的列表
    return search_paths
# 初始化空列表以存储额外参数
include_args = []
# 列出包含的目录列表，包括固定的路径和动态生成的路径
include_dir = [
    "/usr/lib/llvm-11/include/openmp",
    get_python_include_dir(),  # 获取 Python 的包含目录路径
    os.path.join(PYTORCH_ROOT, "third_party/pybind11/include"),
] + clang_search_dirs()  # 添加额外的搜索路径
# 遍历包含目录列表，将每个目录转换为 --extra-arg 参数格式并添加到 include_args 列表中
for dir in include_dir:
    include_args += ["--extra-arg", f"-I{dir}"]

# 定义检查文件的函数，返回LintMessage对象的列表
def check_file(
    filename: str,
    binary: str,
    build_dir: Path,
) -> list[LintMessage]:
    try:
        # 运行指定的二进制文件，传入一系列参数和文件名
        proc = run_command(
            [binary, f"-p={build_dir}", *include_args, filename],
        )
    except OSError as err:
        # 如果运行失败，返回一个包含错误信息的LintMessage对象列表
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    lint_messages = []
    try:
        # 更改当前工作目录到构建目录，因为clang-tidy会相对于构建目录报告文件
        saved_cwd = os.getcwd()
        os.chdir(build_dir)

        # 遍历clang-tidy输出的结果，使用正则表达式匹配
        for match in RESULTS_RE.finditer(proc.stdout.decode()):
            # 将报告的路径转换为绝对路径
            abs_path = str(Path(match["file"]).resolve())
            # 创建LintMessage对象，存储Lint检查的信息
            message = LintMessage(
                path=abs_path,
                name=match["code"],
                description=match["message"],
                line=int(match["line"]),
                char=int(match["column"])
                if match["column"] is not None and not match["column"].startswith("-")
                else None,
                code="CLANGTIDY",
                severity=severities.get(match["severity"], LintSeverity.ERROR),
                original=None,
                replacement=None,
            )
            lint_messages.append(message)
    finally:
        # 恢复原来的工作目录
        os.chdir(saved_cwd)

    return lint_messages


# 主函数入口，不返回任何内容
def main() -> None:
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        description="clang-tidy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    # 添加命令行参数
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-tidy binary path",
    )
    parser.add_argument(
        "--build-dir",
        "--build_dir",
        required=True,
        help=(
            "Where the compile_commands.json file is located. "
            "Gets passed to clang-tidy -p"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志格式和级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET  # 根据参数设置日志级别
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000  # 根据文件数量设置不同的日志级别
        else logging.INFO,
        stream=sys.stderr,
    )
    # 检查给定的二进制文件路径是否存在，如果不存在则输出错误消息并退出程序
    if not os.path.exists(args.binary):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code="CLANGTIDY",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find clang-tidy binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        # 将错误消息以 JSON 格式打印到标准输出，并刷新缓冲区
        print(json.dumps(err_msg._asdict()), flush=True)
        # 退出程序
        sys.exit(0)

    # 获取构建目录的绝对路径
    abs_build_dir = Path(args.build_dir).resolve()

    # 获取 clang-tidy 的绝对路径，替换相对路径（如 .lintbin/clang-tidy）
    # 这是因为 os.chdir 是每个进程的，而 linter 使用它在当前目录和构建文件夹之间切换。
    # 构建文件夹中没有 .lintbin 目录，如果发生竞态条件，linter 命令将会因为找不到 '.lintbin/clang-tidy' 而失败。
    binary_path = os.path.abspath(args.binary)

    # 使用线程池执行 lint 操作，每个文件一个线程，最多使用 CPU 核心数的线程
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        # 提交 lint 操作的任务，并记录每个任务对应的文件名
        futures = {
            executor.submit(
                check_file,
                filename,
                binary_path,
                abs_build_dir,
            ): filename
            for filename in args.filenames
        }
        # 遍历已完成的任务
        for future in concurrent.futures.as_completed(futures):
            try:
                # 获取 lint 操作的结果并将结果以 JSON 格式打印到标准输出
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 如果出现异常，记录临界错误日志并抛出异常
                logging.critical('Failed at "%s".', futures[future])
                raise
# 如果这个脚本被直接执行（而不是作为模块导入），那么执行以下代码块
if __name__ == "__main__":
    # 调用主函数 main() 来执行主程序逻辑
    main()
```