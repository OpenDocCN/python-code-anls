# `.\pytorch\tools\linter\adapters\pip_init.py`

```
"""
Initializer script that installs stuff to pip.
"""

# 引入从 Python 3.7 开始支持的类型提示
from __future__ import annotations

import argparse  # 导入处理命令行参数的模块
import logging   # 导入日志记录模块
import os        # 导入操作系统相关功能的模块
import shutil    # 导入高级文件操作功能的模块
import subprocess  # 导入子进程管理功能的模块
import sys       # 导入与 Python 解释器交互的模块
import time      # 导入时间相关功能的模块


def run_command(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    # 记录要执行的命令
    logging.debug("$ %s", " ".join(args))
    # 记录命令执行开始时间
    start_time = time.monotonic()
    try:
        # 执行命令并返回结果
        return subprocess.run(args, check=True)
    finally:
        # 计算命令执行时间并记录
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(description="pip initializer")
    # 添加位置参数，指定要安装的 pip 包列表
    parser.add_argument(
        "packages",
        nargs="+",
        help="pip packages to install",
    )
    # 添加可选参数，用于开启详细日志记录
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    # 添加可选参数，用于模拟安装过程而不实际安装
    parser.add_argument(
        "--dry-run", help="do not install anything, just print what would be done."
    )
    # 添加可选参数，禁用从 pip 安装 black 的预编译二进制文件
    parser.add_argument(
        "--no-black-binary",
        help="do not use pre-compiled binaries from pip for black.",
        action="store_true",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志输出格式和级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    # 检查是否可以找到 uv 命令
    uv_available = shutil.which("uv") is not None

    if uv_available:
        # 如果找到 uv 命令，则将 uv pip 作为安装 pip 包的前缀
        pip_args = ["uv", "pip", "install"]
    else:
        # 如果未找到 uv 命令，则直接使用 pip install
        pip_args = ["pip", "install"]

    # 如果不在 conda 环境和虚拟环境中，则使用 --user 选项安装以避免需要 root 权限
    in_conda = os.environ.get("CONDA_PREFIX") is not None
    in_virtualenv = os.environ.get("VIRTUAL_ENV") is not None
    if not in_conda and not in_virtualenv:
        pip_args.append("--user")

    # 将位置参数中的包名添加到安装命令中
    pip_args.extend(args.packages)

    # 遍历要安装的包列表，处理版本号和禁用二进制选项
    for package in args.packages:
        package_name, _, version = package.partition("=")
        if version == "":
            # 如果包名未指定版本号，则抛出错误
            raise RuntimeError(
                f"Package {package_name} did not have a version specified. "
                "Please specify a version to produce a consistent linting experience."
            )
        if args.no_black_binary and "black" in package_name:
            # 如果禁用了 black 的预编译二进制选项，则添加对应的命令行参数
            pip_args.append(f"--no-binary={package_name}")

    # 判断是否处于模拟运行模式
    dry_run = args.dry_run == "1"
    if dry_run:
        # 如果是模拟运行模式，则打印即将执行的安装命令并退出
        print(f"Would have run: {pip_args}")
        sys.exit(0)

    # 执行 pip 安装命令
    run_command(pip_args)
```