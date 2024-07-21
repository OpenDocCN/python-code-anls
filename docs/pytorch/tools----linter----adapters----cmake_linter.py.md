# `.\pytorch\tools\linter\adapters\cmake_linter.py`

```
from __future__ import annotations

import argparse  # 导入用于解析命令行参数的模块
import concurrent.futures  # 导入用于并发执行的模块
import json  # 导入处理 JSON 格式数据的模块
import logging  # 导入日志记录模块
import os  # 导入与操作系统交互的模块
import re  # 导入正则表达式模块
import subprocess  # 导入执行外部命令的模块
import time  # 导入处理时间的模块
from enum import Enum  # 导入定义枚举类型的模块
from typing import NamedTuple  # 导入定义命名元组的模块


LINTER_CODE = "CMAKE"  # 定义常量，表示使用的 linter 代码为 CMAKE


class LintSeverity(str, Enum):
    ERROR = "error"  # 定义 lint 的严重错误类型
    WARNING = "warning"  # 定义 lint 的警告类型
    ADVICE = "advice"  # 定义 lint 的建议类型
    DISABLED = "disabled"  # 定义 lint 的禁用类型


class LintMessage(NamedTuple):
    path: str | None  # 文件路径或 None
    line: int | None  # 行号或 None
    char: int | None  # 字符位置或 None
    code: str  # 代码标识
    severity: LintSeverity  # lint 的严重程度
    name: str  # lint 消息名称
    original: str | None  # 原始内容或 None
    replacement: str | None  # 替换内容或 None
    description: str | None  # 描述信息或 None


# 匹配 cmakelint 输出结果的正则表达式模式
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: list[str],  # 外部命令及其参数列表
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))  # 记录调试信息，显示执行的命令及参数
    start_time = time.monotonic()  # 记录命令开始执行的时间点
    try:
        return subprocess.run(
            args,
            capture_output=True,  # 捕获命令的输出结果
        )
    finally:
        end_time = time.monotonic()  # 记录命令执行结束的时间点
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 记录命令执行时间


def check_file(
    filename: str,  # 要检查的文件名
    config: str,  # cmakelint 的配置文件路径
) -> list[LintMessage]:  # 返回 lint 消息列表
    try:
        proc = run_command(
            ["cmakelint", f"--config={config}", filename],  # 执行 cmakelint 命令
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),  # 记录命令执行失败的详细信息
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()  # 获取命令执行结果的标准输出并转换为字符串格式
    return [
        LintMessage(
            path=match["file"],  # lint 错误所在文件路径
            name=match["code"],  # lint 错误代码
            description=match["message"],  # lint 错误信息描述
            line=int(match["line"]),  # lint 错误所在行号
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,  # 错误严重程度为 ERROR
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)  # 使用正则表达式匹配 cmakelint 输出的结果
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cmakelint runner",  # 脚本描述信息
        fromfile_prefix_chars="@",  # 参数文件的前缀字符
    )
    parser.add_argument(
        "--config",  # cmakelint 配置文件的路径
        required=True,  # 必须提供该参数
        help="location of cmakelint config",  # 参数的帮助信息
    )
    parser.add_argument(
        "filenames",  # 要进行 lint 检查的文件路径列表
        nargs="+",  # 至少需要提供一个文件路径
        help="paths to lint",  # 参数的帮助信息
    )

    args = parser.parse_args()  # 解析命令行参数

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),  # 最大线程数为当前 CPU 的核心数
        thread_name_prefix="Thread",  # 线程名称前缀
        # 继续 ThreadPoolExecutor 的代码略，不在注释范围内
    # 使用 `concurrent.futures.ProcessPoolExecutor()` 创建一个进程池执行器，上下文管理器确保在退出时正确关闭
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用字典推导式创建一个字典 `futures`，键为每个 `executor.submit()` 返回的 Future 对象，
        # 值为相应的文件名 `filename`
        futures = {
            executor.submit(
                # 调用 `check_file` 函数，传递参数 `filename` 和 `args.config`
                check_file,
                filename,
                args.config,
            ): filename
            # 遍历参数 `args.filenames` 中的每个文件名 `filename`
            for filename in args.filenames
        }
        # 遍历由 `concurrent.futures.as_completed(futures)` 返回的迭代器
        for future in concurrent.futures.as_completed(futures):
            try:
                # 尝试获取每个 future 的结果，并遍历其中的每条 `lint_message`
                for lint_message in future.result():
                    # 将 `lint_message` 转换为字典，并以 JSON 格式打印到标准输出，同时刷新缓冲区
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 捕获异常，并记录关键信息，包括发生异常的文件名
                logging.critical('Failed at "%s".', futures[future])
                # 重新抛出异常，向上层传递
                raise
```