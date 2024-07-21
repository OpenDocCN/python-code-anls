# `.\pytorch\tools\setup_helpers\gen_version_header.py`

```
# Ideally, there would be a way in Bazel to parse version.txt
# and use the version numbers from there as substitutions for
# an expand_template action. Since there isn't, this silly script exists.

from __future__ import annotations

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关功能模块
from typing import cast, Tuple  # 导入类型提示相关模块


Version = Tuple[int, int, int]  # 定义一个元组类型，表示版本号（major, minor, patch）


def parse_version(version: str) -> Version:
    """
    Parses a version string into (major, minor, patch) version numbers.

    Args:
      version: Full version number string, possibly including revision / commit hash.

    Returns:
      An int 3-tuple of (major, minor, patch) version numbers.
    """
    # 提取版本号部分（去除可能包含的修订号或提交哈希部分）
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return cast(Version, tuple([int(n) for n in version_number_str.split(".")]))


def apply_replacements(replacements: dict[str, str], text: str) -> str:
    """
    Applies the given replacements within the text.

    Args:
      replacements (dict): Mapping of str -> str replacements.
      text (str): Text in which to make replacements.

    Returns:
      Text with replacements applied, if any.
    """
    # 遍历替换字典，将文本中的对应内容进行替换
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def main(args: argparse.Namespace) -> None:
    with open(args.version_path) as f:
        version = f.read().strip()  # 读取版本文件并去除首尾空白字符
    (major, minor, patch) = parse_version(version)  # 解析版本号字符串为 (major, minor, patch)

    replacements = {
        "@TORCH_VERSION_MAJOR@": str(major),  # 定义替换字典，替换主版本号
        "@TORCH_VERSION_MINOR@": str(minor),  # 替换次版本号
        "@TORCH_VERSION_PATCH@": str(patch),  # 替换修订版本号
    }

    # 如果输出路径不存在，则创建对应目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.template_path) as input:
        with open(args.output_path, "w") as output:
            for line in input:  # 遍历模板文件的每一行
                output.write(apply_replacements(replacements, line))  # 对每一行应用替换后写入输出文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate version.h from version.h.in template",
    )
    parser.add_argument(
        "--template-path",
        required=True,
        help="Path to the template (i.e. version.h.in)",
    )
    parser.add_argument(
        "--version-path",
        required=True,
        help="Path to the file specifying the version",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for expanded template (i.e. version.h)",
    )
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 执行主程序逻辑
```