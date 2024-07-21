# `.\pytorch\tools\linter\adapters\bazel_linter.py`

```
"""
This linter ensures that users don't set a SHA hash checksum in Bazel for the http_archive.
Although the security practice of setting the checksum is good, it doesn't work when the
archive is downloaded from some sites like GitHub because it can change. Specifically,
GitHub gives no guarantee to keep the same value forever. Check for more details at
https://github.com/community/community/discussions/46034.
"""

# 导入必要的库和模块
from __future__ import annotations

import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import re  # 用于正则表达式操作
import shlex  # 用于分割命令行参数
import subprocess  # 用于执行外部命令
import sys  # 提供对 Python 解释器的访问和控制
import xml.etree.ElementTree as ET  # 用于处理 XML 数据
from enum import Enum  # 用于定义枚举类型
from typing import NamedTuple  # 用于定义命名元组
from urllib.parse import urlparse  # 用于解析 URL

# 定义常量和全局变量
LINTER_CODE = "BAZEL_LINTER"
SHA256_REGEX = re.compile(r"\s*sha256\s*=\s*['\"](?P<sha256>[a-zA-Z0-9]{64})['\"]\s*,")
DOMAINS_WITH_UNSTABLE_CHECKSUM = {"github.com"}


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


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


def is_required_checksum(urls: list[str | None]) -> bool:
    # 检查 URL 列表中是否存在需要进行 SHA 校验的 URL
    if not urls:
        return False

    for url in urls:
        if not url:
            continue

        parsed_url = urlparse(url)
        if parsed_url.hostname in DOMAINS_WITH_UNSTABLE_CHECKSUM:
            return False

    return True


def get_disallowed_checksums(
    binary: str,
) -> set[str]:
    """
    Return the set of disallowed checksums from all http_archive rules
    """
    # 使用 Bazel 查询所有 http_archive 规则的外部依赖列表，并以 XML 格式输出
    proc = subprocess.run(
        [binary, "query", "kind(http_archive, //external:*)", "--output=xml"],
        capture_output=True,
        check=True,
        text=True,
    )

    root = ET.fromstring(proc.stdout)

    disallowed_checksums = set()
    # 解析 XML 输出中的所有 http_archive 规则
    for rule in root.findall('.//rule[@class="http_archive"]'):
        urls_node = rule.find('.//list[@name="urls"]')
        if urls_node is None:
            continue
        urls = [n.get("value") for n in urls_node.findall(".//string")]

        checksum_node = rule.find('.//string[@name="sha256"]')
        if checksum_node is None:
            continue
        checksum = checksum_node.get("value")

        if not checksum:
            continue

        if not is_required_checksum(urls):
            disallowed_checksums.add(checksum)

    return disallowed_checksums


def check_bazel(
    filename: str,
    disallowed_checksums: set[str],
) -> list[LintMessage]:
    original = ""
    replacement = ""
    # 使用文件名打开文件，并将其内容逐行读取
    with open(filename) as f:
        # 初始化两个空字符串，用于存储原始内容和替换后的内容
        for line in f:
            original += f"{line}"

            # 对每一行使用 SHA256_REGEX 进行匹配
            m = SHA256_REGEX.match(line)
            if m:
                # 如果匹配成功，获取 SHA256 值
                sha256 = m.group("sha256")

                # 如果 SHA256 值在不允许的校验和列表中，则跳过当前行
                if sha256 in disallowed_checksums:
                    continue

            # 将每一行添加到替换后的字符串中
            replacement += f"{line}"

        # 如果原始内容和替换后的内容相同，则返回空列表
        if original == replacement:
            return []

        # 构建LintMessage对象列表，用于报告发现的问题
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ADVICE,
                name="format",
                original=original,
                replacement=replacement,
                description="Found redundant SHA checksums. Run `lintrunner -a` to apply this patch.",
            )
        ]
# 定义主函数，不返回任何内容
def main() -> None:
    # 创建参数解析器对象，设置描述和文件前缀字符
    parser = argparse.ArgumentParser(
        description="A custom linter to detect redundant SHA checksums in Bazel",
        fromfile_prefix_chars="@",
    )
    # 添加必需的参数--binary，用于指定 bazel 二进制路径
    parser.add_argument(
        "--binary",
        required=True,
        help="bazel binary path",
    )
    # 添加参数filenames，接受多个路径用于进行 lint
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    # 解析命令行参数
    args = parser.parse_args()

    try:
        # 获取不允许的校验和列表，根据传入的二进制路径
        disallowed_checksums = get_disallowed_checksums(args.binary)
    except subprocess.CalledProcessError as err:
        # 处理 subprocess 调用过程中的错误，生成相应的 lint 消息
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=__file__,
            severity=LintSeverity.ADVICE,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"COMMAND (exit code {err.returncode})\n"
                f"{shlex.join(err.cmd)}\n\n"
                f"STDERR\n{err.stderr or '(empty)'}\n\n"
                f"STDOUT\n{err.stdout or '(empty)'}"
            ),
        )
        # 打印错误消息的 JSON 格式
        print(json.dumps(err_msg._asdict()))
        return
    except Exception as e:
        # 处理其他异常情况，生成相应的 lint 消息
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(f"Failed due to {e.__class__.__name__}:\n{e}"),
        )
        # 打印错误消息的 JSON 格式
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    # 遍历传入的文件路径列表
    for filename in args.filenames:
        # 对每个文件进行 bazel 校验，获取 lint 消息列表
        for lint_message in check_bazel(filename, disallowed_checksums):
            # 打印 lint 消息的 JSON 格式
            print(json.dumps(lint_message._asdict()), flush=True)


# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```