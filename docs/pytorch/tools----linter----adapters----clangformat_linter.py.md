# `.\pytorch\tools\linter\adapters\clangformat_linter.py`

```py
# 从未来模块中导入注解支持
from __future__ import annotations

# 导入必要的模块
import argparse                   # 解析命令行参数
import concurrent.futures         # 并发执行任务
import json                       # 处理 JSON 数据
import logging                    # 记录日志
import os                         # 提供操作系统相关功能
import subprocess                 # 执行外部命令
import sys                        # 提供对 Python 运行时系统的访问
import time                       # 时间相关功能
from enum import Enum             # 枚举类型支持
from pathlib import Path         # 处理文件路径
from typing import Any, NamedTuple  # 类型提示支持

# 判断操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"

# 打印到标准错误流的函数
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)

# 定义代码检查严重性的枚举类型
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

# 将路径转换为 POSIX 风格的字符串函数
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name

# 内部运行外部命令的函数
def _run_command(
    args: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))  # 记录调试信息：运行的命令
    start_time = time.monotonic()          # 记录开始时间
    try:
        return subprocess.run(
            args,
            capture_output=True,
            shell=IS_WINDOWS,  # 使用 Windows 的命令行解释器执行命令
            timeout=timeout,   # 设置命令超时时间
            check=True,        # 检查命令执行结果，如果出错则抛出异常
        )
    finally:
        end_time = time.monotonic()        # 记录结束时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 记录命令执行耗时

# 运行外部命令的函数，支持重试
def run_command(
    args: list[str],
    *,
    retries: int,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    remaining_retries = retries
    while True:
        try:
            return _run_command(args, timeout=timeout)  # 调用内部函数执行命令
        except subprocess.TimeoutExpired as err:
            if remaining_retries == 0:  # 如果没有剩余重试次数，则抛出超时异常
                raise err
            remaining_retries -= 1  # 减少剩余重试次数
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",  # 记录警告信息：因命令执行失败而重试
                retries - remaining_retries,
                retries,
                err,
            )
            time.sleep(1)  # 等待1秒后再重试

# 检查指定文件的函数，返回代码检查消息列表
def check_file(
    filename: str,
    binary: str,
    retries: int,
    timeout: int,
) -> list[LintMessage]:
    try:
        with open(filename, "rb") as f:
            original = f.read()  # 读取文件内容作为原始数据
        proc = run_command(
            [binary, filename],  # 执行外部命令，传递文件名作为参数
            retries=retries,     # 设置重试次数
            timeout=timeout,     # 设置命令超时时间
        )
    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGFORMAT",
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=(
                    "clang-format timed out while trying to process a file. "
                    "Please report an issue in pytorch/pytorch with the "
                    "label 'module: lint'"
                ),
            )
        ]  # 如果命令执行超时，则返回一个具有超时信息的 lint 消息
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            # 处理异常情况，返回一个LintMessage对象的列表
            LintMessage(
                path=filename,  # 文件路径
                line=None,  # 行号为空
                char=None,  # 字符位置为空
                code="CLANGFORMAT",  # 错误码为CLANGFORMAT
                severity=LintSeverity.ADVICE,  # 严重程度为建议
                name="command-failed",  # 错误名称为command-failed
                original=None,  # 原始内容为空
                replacement=None,  # 替换内容为空
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"  # 描述失败原因和具体错误信息
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),  # 将命令转为字符串格式
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",  # 解码并处理标准错误流
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",  # 解码并处理标准输出流
                    )
                ),
            )
        ]

    replacement = proc.stdout  # 设置替换内容为进程的标准输出

    if original == replacement:  # 如果原始内容等于替换内容
        return []  # 返回空列表，表示无需进行修正

    return [
        # 返回一个LintMessage对象的列表，表示需要进行代码格式化的警告信息
        LintMessage(
            path=filename,  # 文件路径
            line=None,  # 行号为空
            char=None,  # 字符位置为空
            code="CLANGFORMAT",  # 错误码为CLANGFORMAT
            severity=LintSeverity.WARNING,  # 严重程度为警告
            name="format",  # 错误名称为format
            original=original.decode("utf-8"),  # 原始内容解码为UTF-8格式
            replacement=replacement.decode("utf-8"),  # 替换内容解码为UTF-8格式
            description="See https://clang.llvm.org/docs/ClangFormat.html.\nRun `lintrunner -a` to apply this patch.",  # 描述信息，指向ClangFormat文档和应用补丁的命令
        )
    ]
# 定义程序的主函数
def main() -> None:
    # 创建命令行参数解析器对象，描述为格式化文件使用 clang-format
    parser = argparse.ArgumentParser(
        description="Format files with clang-format.",
        fromfile_prefix_chars="@",
    )

    # 添加命令行参数：指定 clang-format 的可执行文件路径，必需参数
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-format binary path",
    )

    # 添加命令行参数：重试超时 clang-format 的次数，默认为 3 次
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out clang-format",
    )

    # 添加命令行参数：等待 clang-format 的超时时间，默认为 90 秒
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for clang-format",
    )

    # 添加命令行参数：是否输出详细日志信息
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )

    # 添加命令行参数：待处理的文件路径列表，数量至少为一个
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 解析命令行参数并存储到 args 对象中
    args = parser.parse_args()

    # 配置日志输出格式和级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        # 如果启用了 verbose 参数，则日志级别为 NOTSET；否则，如果文件数小于 1000，则为 DEBUG，否则为 INFO
        level=logging.NOTSET if args.verbose else (logging.DEBUG if len(args.filenames) < 1000 else logging.INFO),
        stream=sys.stderr,
    )

    # 根据操作系统确定 clang-format 可执行文件的路径
    binary = os.path.normpath(args.binary) if IS_WINDOWS else args.binary

    # 如果指定路径的二进制文件不存在
    if not Path(binary).exists():
        # 创建 lint_message 对象，报错指出找不到 clang-format 的二进制文件路径
        lint_message = LintMessage(
            path=None,
            line=None,
            char=None,
            code="CLANGFORMAT",
            severity=LintSeverity.ERROR,
            name="init-error",
            original=None,
            replacement=None,
            description=(
                f"Could not find clang-format binary at {binary}, "
                "did you forget to run `lintrunner init`?"
            ),
        )
        # 打印 lint_message 对象的 JSON 格式输出，并刷新输出缓冲区
        print(json.dumps(lint_message._asdict()), flush=True)
        # 退出程序，返回状态码 0
        sys.exit(0)

    # 使用线程池执行 lint 检查任务
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        # 提交 lint 检查任务到线程池，并记录每个任务对应的文件路径
        futures = {
            executor.submit(check_file, x, binary, args.retries, args.timeout): x
            for x in args.filenames
        }
        # 等待所有 lint 检查任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                # 获取 lint 检查任务的结果，遍历 lint_message 对象列表，并以 JSON 格式输出
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 捕获异常情况，记录日志并重新抛出异常
                logging.critical('Failed at "%s".', futures[future])
                raise

# 如果当前脚本作为主程序运行，则调用主函数 main()
if __name__ == "__main__":
    main()
```