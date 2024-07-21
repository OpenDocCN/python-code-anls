# `.\pytorch\tools\linter\adapters\ruff_linter.py`

```
"""Adapter for https://github.com/charliermarsh/ruff."""

# 引入未来的注解支持
from __future__ import annotations

# 引入需要的模块
import argparse  # 命令行参数解析模块
import concurrent.futures  # 并发执行模块
import dataclasses  # 数据类支持
import enum  # 枚举类型支持
import json  # JSON 数据处理模块
import logging  # 日志记录模块
import os  # 操作系统相关功能
import subprocess  # 子进程管理模块
import sys  # 系统相关功能
import time  # 时间处理模块
from typing import Any, BinaryIO  # 类型提示相关

# 定义一个全局变量，用于标识 Linter 的代码类型
LINTER_CODE = "RUFF"

# 判断操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr."""
    # 打印内容到标准错误流
    print(*args, file=sys.stderr, flush=True, **kwargs)


# 枚举类型，表示 Lint 消息的严重性级别
class LintSeverity(str, enum.Enum):
    """Severity of a lint message."""

    ERROR = "error"  # 错误
    WARNING = "warning"  # 警告
    ADVICE = "advice"  # 建议
    DISABLED = "disabled"  # 禁用


# 数据类，表示 Lint 消息的详细信息
@dataclasses.dataclass(frozen=True)
class LintMessage:
    """A lint message defined by https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html."""

    path: str | None  # 文件路径，可为空
    line: int | None  # 行号，可为空
    char: int | None  # 字符位置，可为空
    code: str  # 错误码
    severity: LintSeverity  # 错误级别
    name: str  # 名称
    original: str | None  # 原始内容，可为空
    replacement: str | None  # 替换内容，可为空
    description: str | None  # 描述信息，可为空

    def asdict(self) -> dict[str, Any]:
        # 将数据类转换为字典形式
        return dataclasses.asdict(self)

    def display(self) -> None:
        """Print to stdout for lintrunner to consume."""
        # 打印 JSON 格式的 lint 消息到标准输出，供 lint 运行器使用
        print(json.dumps(self.asdict()), flush=True)


def as_posix(name: str) -> str:
    # 将路径分隔符统一为 POSIX 风格的斜杠 '/'，用于跨平台兼容
    return name.replace("\\", "/") if IS_WINDOWS else name


def _run_command(
    args: list[str],
    *,
    timeout: int | None,
    stdin: BinaryIO | None,
    input: bytes | None,
    check: bool,
    cwd: os.PathLike[Any] | None,
) -> subprocess.CompletedProcess[bytes]:
    # 执行系统命令，并记录调试信息
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        if input is not None:
            # 如果提供了输入数据，则使用输入数据执行命令
            return subprocess.run(
                args,
                capture_output=True,
                shell=False,
                input=input,
                timeout=timeout,
                check=check,
                cwd=cwd,
            )

        # 否则，使用标准输入流或无输入数据执行命令
        return subprocess.run(
            args,
            stdin=stdin,
            capture_output=True,
            shell=False,
            timeout=timeout,
            check=check,
            cwd=cwd,
        )
    finally:
        # 记录命令执行时间
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: list[str],
    *,
    retries: int = 0,
    timeout: int | None = None,
    stdin: BinaryIO | None = None,
    input: bytes | None = None,
    check: bool = False,
    cwd: os.PathLike[Any] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    # 执行系统命令，支持重试机制和超时控制
    remaining_retries = retries
    # 无限循环，持续尝试运行命令直到成功或超时
    while True:
        try:
            # 调用 _run_command 函数来执行指定的命令
            return _run_command(
                args, timeout=timeout, stdin=stdin, input=input, check=check, cwd=cwd
            )
        except subprocess.TimeoutExpired as err:
            # 如果命令超时，则进入异常处理
            if remaining_retries == 0:
                # 如果没有剩余重试次数，则抛出超时异常
                raise err
            # 减少剩余重试次数
            remaining_retries -= 1
            # 记录警告日志，表明正在进行重试
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )
            # 等待1秒后继续下一轮循环
            time.sleep(1)
# 添加默认选项到命令行参数解析器
def add_default_options(parser: argparse.ArgumentParser) -> None:
    """Add default options to a parser.

    This should be called the last in the chain of add_argument calls.
    """
    # 添加 --retries 参数，指定整数类型，默认为 3，用于指定 linter 超时时重试的次数
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of times to retry if the linter times out.",
    )
    # 添加 --verbose 参数，用于开启详细日志记录
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    # 添加位置参数 "filenames"，接收多个路径用于 lint
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )


def explain_rule(code: str) -> str:
    # 运行命令 ["ruff", "rule", "--output-format=json", code] 获取规则输出为 JSON 格式
    proc = run_command(
        ["ruff", "rule", "--output-format=json", code],
        check=True,
    )
    # 将 JSON 结果解析为字典
    rule = json.loads(str(proc.stdout, "utf-8").strip())
    # 返回规则的提示信息，格式为 "<linter>: <summary>"
    return f"\n{rule['linter']}: {rule['summary']}"


def get_issue_severity(code: str) -> LintSeverity:
    # 根据错误码判断错误的严重程度
    # 如果错误码以以下任一字符串开头，则返回建议级别 LintSeverity.ADVICE
    if any(
        code.startswith(x)
        for x in (
            "B9",
            "C4",
            "C9",
            "E2",
            "E3",
            "E5",
            "T400",
            "T49",
            "PLC",
            "PLR",
        )
    ):
        return LintSeverity.ADVICE

    # 如果错误码以 "F821" 或 "E999" 开头，则返回错误级别 LintSeverity.ERROR
    if any(code.startswith(x) for x in ("F821", "E999", "PLE")):
        return LintSeverity.ERROR

    # 其他情况下，返回警告级别 LintSeverity.WARNING
    return LintSeverity.WARNING


def format_lint_message(
    message: str, code: str, rules: dict[str, str], show_disable: bool
) -> str:
    # 如果有规则存在，将规则附加到消息末尾
    if rules:
        message += f".\n{rules.get(code) or ''}"
    # 在消息末尾添加固定的信息，指向 Ruff 文档中的规则说明页面
    message += ".\nSee https://beta.ruff.rs/docs/rules/"
    # 如果需要显示如何禁用此消息，添加相应信息
    if show_disable:
        message += f".\n\nTo disable, use `  # noqa: {code}`"
    return message


def check_files(
    filenames: list[str],
    severities: dict[str, LintSeverity],
    *,
    config: str | None,
    retries: int,
    timeout: int,
    explain: bool,
    show_disable: bool,
) -> list[LintMessage]:
    try:
        # 运行命令 ["python3", "-m", "ruff", "check", "--exit-zero", "--quiet", "--output-format=json", *filenames]
        # 用于检查文件，并以 JSON 格式输出结果
        proc = run_command(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--exit-zero",
                "--quiet",
                "--output-format=json",
                *([f"--config={config}"] if config else []),
                *filenames,
            ],
            retries=retries,
            timeout=timeout,
            check=True,
        )
    # 处理可能的 OSError 或 subprocess.CalledProcessError 异常
    except (OSError, subprocess.CalledProcessError) as err:
        # 返回一个包含LintMessage对象的列表，表示 lint 过程中的错误信息
        return [
            LintMessage(
                path=None,  # 文件路径为空，因为这是 lint 过程中的一个错误信息
                line=None,  # 行号为空，因为这是 lint 过程中的一个错误信息
                char=None,  # 字符位置为空，因为这是 lint 过程中的一个错误信息
                code=LINTER_CODE,  # 错误码，指示 lint 的特定类型
                severity=LintSeverity.ERROR,  # 错误严重性为 ERROR 级别
                name="command-failed",  # 错误名称，指示命令执行失败
                original=None,  # 原始信息为空，因为这是 lint 过程中的一个错误信息
                replacement=None,  # 替换信息为空，因为这是 lint 过程中的一个错误信息
                description=(  # 错误描述，包括具体的错误信息和可能的修复建议
                    f"Failed due to {err.__class__.__name__}:\n{err}"  # 若错误不是 subprocess.CalledProcessError，则显示错误类名和详细错误信息
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        f"COMMAND (exit code {err.returncode})\n"  # 若是 subprocess.CalledProcessError，则显示命令和其返回码
                        f"{' '.join(as_posix(x) for x in err.cmd)}\n\n"  # 命令及其参数列表
                        f"STDERR\n{err.stderr.decode('utf-8').strip() or '(empty)'}\n\n"  # 标准错误输出内容
                        f"STDOUT\n{err.stdout.decode('utf-8').strip() or '(empty)'}"  # 标准输出内容
                    )
                ),
            )
        ]

    # 从进程的标准输出中获取字符串，解析为 JSON 格式的漏洞信息
    stdout = str(proc.stdout, "utf-8").strip()
    vulnerabilities = json.loads(stdout)

    # 如果需要解释漏洞信息
    if explain:
        # 获取所有漏洞代码的集合
        all_codes = {v["code"] for v in vulnerabilities}
        # 创建包含每个漏洞代码对应解释的字典
        rules = {code: explain_rule(code) for code in all_codes}
    else:
        rules = {}  # 否则，解释规则为空字典

    # 返回一个包含LintMessage对象的列表，表示每个漏洞的 lint 结果信息
    return [
        LintMessage(
            path=vuln["filename"],  # 漏洞所在文件路径
            name=vuln["code"],  # 漏洞代码
            description=(  # 漏洞描述，格式化 lint 消息的具体内容
                format_lint_message(
                    vuln["message"],  # 漏洞消息
                    vuln["code"],  # 漏洞代码
                    rules,  # 漏洞代码对应的解释规则
                    show_disable,  # 是否显示禁用信息
                )
            ),
            line=int(vuln["location"]["row"]),  # 漏洞所在行号
            char=int(vuln["location"]["column"]),  # 漏洞所在字符位置
            code=LINTER_CODE,  # 错误码，指示 lint 的特定类型
            severity=severities.get(vuln["code"], get_issue_severity(vuln["code"])),  # 漏洞严重性
            original=None,  # 原始信息为空
            replacement=None,  # 替换信息为空
        )
        for vuln in vulnerabilities  # 对于每个漏洞信息
    ]
def check_file_for_fixes(
    filename: str,
    *,
    config: str | None,  # 接收配置文件路径或空值作为参数
    retries: int,  # 重试次数
    timeout: int,  # 超时时间
) -> list[LintMessage]:  # 返回LintMessage对象列表
    try:
        with open(filename, "rb") as f:  # 打开文件以二进制模式读取
            original = f.read()  # 读取文件内容保存到original变量中
        with open(filename, "rb") as f:  # 再次打开文件以二进制模式读取
            proc_fix = run_command(  # 运行外部命令，执行lint修复操作
                [
                    sys.executable,  # 使用系统Python解释器运行
                    "-m",
                    "ruff",  # 使用ruff模块
                    "check",  # 执行检查操作
                    "--fix-only",  # 仅执行修复操作
                    "--exit-zero",  # 退出码为零
                    *([f"--config={config}"] if config else []),  # 如果配置路径存在，则添加配置参数
                    "--stdin-filename",  # 从标准输入读取文件名
                    filename,  # 要处理的文件名
                    "-",  # 标准输入作为命令的输入
                ],
                stdin=f,  # 标准输入来自打开的文件f
                retries=retries,  # 重试次数
                timeout=timeout,  # 超时时间
                check=True,  # 检查命令是否成功执行
            )
    except (OSError, subprocess.CalledProcessError) as err:  # 处理可能发生的异常
        return [  # 返回包含LintMessage对象的列表
            LintMessage(
                path=None,  # 文件路径为空
                line=None,  # 行号为空
                char=None,  # 字符位置为空
                code=LINTER_CODE,  # Linter代码
                severity=LintSeverity.ERROR,  # 错误严重程度
                name="command-failed",  # 错误名称
                original=None,  # 原始内容为空
                replacement=None,  # 替换内容为空
                description=(  # 错误描述信息，根据不同异常类型进行组合
                    f"Failed due to {err.__class__.__name__}:\n{err}"  # 如果不是CalledProcessError，显示基本异常信息
                    if not isinstance(err, subprocess.CalledProcessError) else (  # 如果是CalledProcessError，则显示详细信息
                        f"COMMAND (exit code {err.returncode})\n"
                        f"{' '.join(as_posix(x) for x in err.cmd)}\n\n"
                        f"STDERR\n{err.stderr.decode('utf-8').strip() or '(empty)'}\n\n"
                        f"STDOUT\n{err.stdout.decode('utf-8').strip() or '(empty)'}"
                    )
                ),
            )
        ]

    replacement = proc_fix.stdout  # 获取修复后的输出内容
    if original == replacement:  # 如果修复后的内容与原始内容相同
        return []  # 返回空列表，表示没有进行任何修复

    return [  # 返回包含LintMessage对象的列表
        LintMessage(
            path=filename,  # 文件路径为filename
            name="format",  # 错误名称为format
            description="Run `lintrunner -a` to apply this patch.",  # 描述如何应用修补程序
            line=None,  # 行号为空
            char=None,  # 字符位置为空
            code=LINTER_CODE,  # Linter代码
            severity=LintSeverity.WARNING,  # 警告严重程度
            original=original.decode("utf-8"),  # 原始内容解码为UTF-8格式字符串
            replacement=replacement.decode("utf-8"),  # 替换内容解码为UTF-8格式字符串
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(  # 创建参数解析器对象
        description=f"Ruff linter. Linter code: {LINTER_CODE}. Use with RUFF-FIX to auto-fix issues.",  # 程序描述信息
        fromfile_prefix_chars="@",  # 读取文件前缀字符为@
    )
    parser.add_argument(  # 添加命令行参数
        "--config",  # 配置文件路径参数
        default=None,  # 默认值为空
        help="Path to the `pyproject.toml` or `ruff.toml` file to use for configuration",  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--explain",  # 解释规则的选项
        action="store_true",  # 设置为True表示存在即可，无需值
        help="Explain a rule",  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--show-disable",  # 显示如何禁用lint消息的选项
        action="store_true",  # 设置为True表示存在即可，无需值
        help="Show how to disable a lint message",  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--timeout",  # 超时时间参数
        default=90,  # 默认值为90
        type=int,  # 参数类型为整数
        help="Seconds to wait for ruff",  # 帮助信息
    )
    parser.add_argument(
        "--severity",
        action="append",
        help="map code to severity (e.g. `F401:advice`). This option can be used multiple times.",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Do not suggest fixes",
    )
    add_default_options(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET   # 设置日志的默认级别为NOTSET，如果args.verbose未设置，则根据文件名数量动态调整
        if args.verbose        # 如果启用了详细模式
        else logging.DEBUG     # 如果文件名数量少于1000个，则设置为DEBUG级别
        if len(args.filenames) < 1000
        else logging.INFO,     # 否则设置为INFO级别
        stream=sys.stderr,
    )

    severities: dict[str, LintSeverity] = {}
    if args.severity:          # 如果用户指定了severity参数
        for severity in args.severity:   # 遍历每个指定的severity
            parts = severity.split(":", 1)   # 将参数以":"分割成两部分
            assert len(parts) == 2, f"invalid severity `{severity}`"   # 如果分割后不是两部分，抛出异常
            severities[parts[0]] = LintSeverity(parts[1])   # 将分割后的两部分分别作为键值对存入severities字典

    lint_messages = check_files(
        args.filenames,
        severities=severities,
        config=args.config,
        retries=args.retries,
        timeout=args.timeout,
        explain=args.explain,
        show_disable=args.show_disable,
    )
    for lint_message in lint_messages:   # 遍历lint_messages中的每个lint_message对象
        lint_message.display()   # 显示lint_message的内容

    if args.no_fix or not lint_messages:   # 如果用户设置了--no-fix参数或lint_messages为空
        # 如果不进行修复，则可以提前退出
        return

    files_with_lints = {lint.path for lint in lint_messages if lint.path is not None}   # 收集lint_messages中有路径的文件集合
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:   # 创建一个线程池执行器
        futures = {   # 创建一个字典，用于存储每个文件路径对应的执行future
            executor.submit(
                check_file_for_fixes,
                path,
                config=args.config,
                retries=args.retries,
                timeout=args.timeout,
            ): path   # 每个future与其对应的文件路径path关联
            for path in files_with_lints   # 遍历有lint的文件路径集合
        }
        for future in concurrent.futures.as_completed(futures):   # 遍历已完成的future
            try:
                for lint_message in future.result():   # 获取future的结果，遍历每个lint_message
                    lint_message.display()   # 显示lint_message的内容
            except Exception:   # 捕获所有异常，用于lintrunner
                logging.critical('Failed at "%s".', futures[future])   # 记录关键错误信息
                raise   # 抛出异常
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 main() 函数
if __name__ == "__main__":
    main()
```