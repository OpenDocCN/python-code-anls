# `.\pytorch\tools\linter\adapters\ufmt_linter.py`

```py
# 从未来导入类型提示注解
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入并发执行模块
import concurrent.futures
# 导入文件名匹配模块
import fnmatch
# 导入 JSON 操作模块
import json
# 导入日志记录模块
import logging
# 导入操作系统相关模块
import os
# 导入正则表达式模块
import re
# 导入系统相关模块
import sys
# 导入枚举类型模块
from enum import Enum
# 导入路径操作模块
from pathlib import Path
# 导入任意类型注解
from typing import Any, NamedTuple

# 导入 isort 模块
import isort
# 导入 isort 的配置类
from isort import Config as IsortConfig
# 导入 ufmt_string 函数
from ufmt.core import ufmt_string
# 导入生成黑名单配置的函数
from ufmt.util import make_black_config
# 导入 usort 模块
from usort import Config as UsortConfig

# 判断操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"
# 获取代码仓库的根目录
REPO_ROOT = Path(__file__).absolute().parents[3]

# 定义用于 isort 的白名单正则表达式，包含一系列文件和目录的匹配规则
ISORT_WHITELIST = re.compile(
    "|".join(
        (
            r"\A\Z",  # 空字符串
            *map(
                fnmatch.translate,
                [
                    # **
                    "**",
                    # .ci/**
                    ".ci/**",
                    # .github/**
                    ".github/**",
                    # benchmarks/**
                    "benchmarks/**",
                    # functorch/**
                    "functorch/**",
                    # tools/**
                    "tools/**",
                    # torchgen/**
                    "torchgen/**",
                    # test/**
                    "test/**",
                    # test/[a-c]*/**
                    "test/[a-c]*/**",
                    # test/d*/**
                    "test/d*/**",
                    # test/dy*/**
                    "test/dy*/**",
                    # test/[e-h]*/**
                    "test/[e-h]*/**",
                    # test/i*/**
                    "test/i*/**",
                    # test/j*/**
                    "test/j*/**",
                    # test/[k-p]*/**
                    "test/[k-p]*/**",
                    # test/[q-z]*/**
                    "test/[q-z]*/**",
                    # torch/**
                    "torch/**",
                    # torch/_[a-c]*/**
                    "torch/_[a-c]*/**",
                    # torch/_d*/**
                    "torch/_d*/**",
                    # torch/_[e-h]*/**
                    "torch/_[e-h]*/**",
                    # torch/_i*/**
                    "torch/_i*/**",
                    # torch/_[j-z]*/**
                    "torch/_[j-z]*/**",
                    # torch/[a-c]*/**
                    "torch/[a-c]*/**",
                    # torch/d*/**
                    "torch/d*/**",
                    # torch/[e-n]*/**
                    "torch/[e-n]*/**",
                    # torch/[o-z]*/**
                    "torch/[o-z]*/**",
                ],
            ),
        )
    )
)


def eprint(*args: Any, **kwargs: Any) -> None:
    # 打印错误信息到标准错误流
    print(*args, file=sys.stderr, flush=True, **kwargs)


class LintSeverity(str, Enum):
    # 定义 lint 消息的严重程度枚举类型
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # 定义 lint 消息的命名元组，包含多个字段
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None
# 定义一个函数，根据操作系统类型将反斜杠替换为斜杠，用于处理文件路径
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name

# 定义一个函数，格式化错误信息并返回LintMessage对象
def format_error_message(filename: str, err: Exception) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="UFMT",
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )

# 定义一个函数，检查文件格式并返回LintMessage对象的列表
def check_file(filename: str) -> list[LintMessage]:
    # 获取文件的绝对路径并读取其内容
    path = Path(filename).absolute()
    original = path.read_text(encoding="utf-8")

    try:
        # 查找并加载usort配置
        usort_config = UsortConfig.find(path)
        # 创建black配置
        black_config = make_black_config(path)

        # 如果文件路径不是当前文件且不在ISORT_WHITELIST中
        if not path.samefile(__file__) and not ISORT_WHITELIST.match(
            path.absolute().relative_to(REPO_ROOT).as_posix()
        ):
            # 使用正则表达式替换isort: split为usort: skip，并进行格式化处理
            isorted_replacement = re.sub(
                r"(#.*\b)isort: split\b",
                r"\g<1>usort: skip",
                isort.code(
                    re.sub(r"(#.*\b)usort:\s*skip\b", r"\g<1>isort: split", original),
                    config=IsortConfig(settings_path=str(REPO_ROOT)),
                    file_path=path,
                ),
            )
        else:
            # 使用原始内容作为替换内容
            isorted_replacement = original

        # 使用UFMT API调用usort和black，并生成替换后的内容
        replacement = ufmt_string(
            path=path,
            content=isorted_replacement,
            usort_config=usort_config,
            black_config=black_config,
        )

        # 如果原始内容和替换后的内容相同，返回空列表
        if original == replacement:
            return []

        # 如果内容有变化，返回一个LintMessage对象的列表，提醒运行`lintrunner -a`来应用这个补丁
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="UFMT",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as err:
        # 捕获异常并返回格式化的错误消息LintMessage对象
        return [format_error_message(filename, err)]

# 主函数，负责解析命令行参数并调用相应的函数处理文件
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with ufmt (black + usort).",
        fromfile_prefix_chars="@",
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
    args = parser.parse_args()

    # 配置日志格式和日志级别
    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # 使用多进程池执行Lint检查
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    # 使用 concurrent.futures 模块创建一个线程池 executor，并通过上下文管理器确保在使用完后正确关闭
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用 executor.submit() 方法提交每个文件的检查任务，并使用字典推导式将 future 对象与文件名关联起来
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        # 遍历每个 future 对象，处理完成的任务
        for future in concurrent.futures.as_completed(futures):
            try:
                # 获取任务的返回结果，并遍历 lint_message 列表
                for lint_message in future.result():
                    # 将 lint_message 转换为字典，并将其 JSON 格式化输出
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 如果出现异常，记录错误日志，指明失败的文件名
                logging.critical('Failed at "%s".', futures[future])
                # 将异常抛出，向上层传递
                raise
# 如果这个脚本作为主程序运行
if __name__ == "__main__":
    # 调用主函数，程序的入口点
    main()
```