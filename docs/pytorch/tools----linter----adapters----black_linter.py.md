# `.\pytorch\tools\linter\adapters\black_linter.py`

```py
# 从未来导入注释支持，以便在函数签名中使用类型注释
from __future__ import annotations

# 导入所需模块
import argparse                 # 用于解析命令行参数
import concurrent.futures       # 提供并发执行任务的功能
import json                     # 用于处理 JSON 格式数据
import logging                  # 日志记录模块
import os                       # 提供与操作系统交互的功能
import subprocess               # 提供执行外部命令的功能
import sys                      # 提供与解释器交互的功能
import time                     # 提供时间相关的功能
from enum import Enum           # 枚举类型支持
from typing import Any, BinaryIO, NamedTuple  # 类型提示相关


# 判断操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"


# 输出错误信息到标准错误流
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


# 定义代码检查结果的严重程度枚举类型
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


# 定义代码检查结果消息的命名元组
class LintMessage(NamedTuple):
    path: str | None            # 文件路径，可为空
    line: int | None            # 行号，可为空
    char: int | None            # 字符位置，可为空
    code: str                   # 错误代码
    severity: LintSeverity      # 错误严重程度
    name: str                   # 错误名称
    original: str | None        # 原始内容，可为空
    replacement: str | None     # 建议的替换内容，可为空
    description: str | None     # 错误描述，可为空


# 将路径名统一使用 POSIX 风格的斜杠路径分隔符（仅在 Windows 下生效）
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# 内部函数：运行外部命令并返回完整的执行结果
def _run_command(
    args: list[str],            # 外部命令及其参数列表
    *,
    stdin: BinaryIO,            # 标准输入流
    timeout: int,               # 超时时间（秒）
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))  # 记录调试信息：输出执行的命令
    start_time = time.monotonic()           # 记录开始执行的时间
    try:
        return subprocess.run(
            args,                           # 执行的命令及参数
            stdin=stdin,                    # 指定标准输入流
            capture_output=True,            # 捕获命令的输出
            shell=IS_WINDOWS,               # 是否使用 shell 执行（针对 Windows 的批处理脚本）
            timeout=timeout,                # 指定超时时间
            check=True,                     # 是否检查命令执行的返回状态
        )
    finally:
        end_time = time.monotonic()         # 记录命令执行结束的时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 记录执行耗时


# 外部可调用函数：运行外部命令，支持重试机制
def run_command(
    args: list[str],            # 外部命令及其参数列表
    *,
    stdin: BinaryIO,            # 标准输入流
    retries: int,               # 重试次数
    timeout: int,               # 超时时间（秒）
) -> subprocess.CompletedProcess[bytes]:
    remaining_retries = retries        # 剩余重试次数初始化为总重试次数
    while True:
        try:
            return _run_command(args, stdin=stdin, timeout=timeout)  # 调用内部函数执行命令
        except subprocess.TimeoutExpired as err:
            if remaining_retries == 0:  # 如果没有剩余重试次数，则抛出超时异常
                raise err
            remaining_retries -= 1      # 否则减少剩余重试次数
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )                          # 记录警告日志：因命令执行失败而重试
            time.sleep(1)               # 等待1秒后重试


# 外部函数：检查指定文件并返回代码检查结果列表
def check_file(
    filename: str,              # 待检查的文件名
    retries: int,               # 重试次数
    timeout: int,               # 超时时间（秒）
) -> list[LintMessage]:
    try:
        with open(filename, "rb") as f:    # 以二进制只读模式打开文件
            original = f.read()            # 读取文件内容作为原始数据
        with open(filename, "rb") as f:    # 再次以二进制只读模式打开文件
            proc = run_command(
                [sys.executable, "-mblack", "--stdin-filename", filename, "-"],  # 使用 black 模块格式化文件内容
                stdin=f,                      # 指定标准输入为文件流
                retries=retries,              # 指定重试次数
                timeout=timeout,              # 指定超时时间
            )
    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="BLACK",
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=(
                    "black timed out while trying to process a file. "
                    "Please report an issue in pytorch/pytorch with the "
                    "label 'module: lint'"
                ),
            )
        ]



# 处理超时异常情况，返回一个LintMessage对象的列表，表示lint过程中的错误信息
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="BLACK",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
        ]



# 处理OSError和subprocess.CalledProcessError异常情况，返回一个LintMessage对象的列表，表示lint过程中的建议信息
    replacement = proc.stdout
    if original == replacement:
        return []



# 如果原始内容和替换内容相同，则返回空列表，表示没有需要警告或建议的修正
    return [
        LintMessage(
            path=filename,
            line=None,
            char=None,
            code="BLACK",
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="Run `lintrunner -a` to apply this patch.",
        )
    ]



# 如果有内容需要修正，则返回一个LintMessage对象的列表，表示lint过程中的警告信息，包含原始和替换的内容以及修复方法描述
def main() -> None:
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser(
        description="Format files with black.",
        fromfile_prefix_chars="@",
    )
    # 添加命令行参数：重试次数，默认为3
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out black",
    )
    # 添加命令行参数：超时时间，默认为90秒
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for black",
    )
    # 添加命令行参数：是否启用详细日志
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    # 添加命令行参数：待检查文件的路径列表
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志输出格式和日志级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET  # 如果启用详细日志则为 NOTSET，否则根据文件数决定是 DEBUG 还是 INFO
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # 使用线程池执行文件检查任务
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),  # 最大工作线程数为 CPU 核心数
        thread_name_prefix="Thread",  # 线程名称前缀为 "Thread"
    ) as executor:
        # 提交文件检查任务，并将 future 与文件名关联起来
        futures = {
            executor.submit(check_file, x, args.retries, args.timeout): x
            for x in args.filenames
        }
        # 遍历已完成的 future
        for future in concurrent.futures.as_completed(futures):
            try:
                # 处理 future 返回的 lint_message，将其转换为 JSON 格式输出
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 如果出现异常，记录日志并抛出异常
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
```