# `.\pytorch\tools\generate_torch_version.py`

```
# 从未来版本导入 annotations，支持类型提示
from __future__ import annotations

# 导入必要的库
import argparse  # 提供命令行参数解析功能
import os  # 提供操作系统相关的功能
import re  # 提供正则表达式匹配功能
import subprocess  # 提供执行外部命令的功能
from pathlib import Path  # 提供操作路径的功能

# 导入 setuptools 库的 distutils 模块（类型提示忽略导入错误）
from setuptools import distutils  # type: ignore[import]

# 未知版本号的标识
UNKNOWN = "Unknown"
# 用于匹配版本号的正则表达式模式
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")

# 获取当前代码库的 SHA 标识
def get_sha(pytorch_root: str | Path) -> str:
    try:
        rev = None
        # 如果存在 .git 目录，使用 git 命令获取 SHA
        if os.path.exists(os.path.join(pytorch_root, ".git")):
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=pytorch_root
            )
        # 如果存在 .hg 目录，使用 hg 命令获取 SHA
        elif os.path.exists(os.path.join(pytorch_root, ".hg")):
            rev = subprocess.check_output(
                ["hg", "identify", "-r", "."], cwd=pytorch_root
            )
        # 如果成功获取到 SHA，则返回解码后的字符串
        if rev:
            return rev.decode("ascii").strip()
    except Exception:
        pass
    return UNKNOWN

# 获取当前代码库的版本标签
def get_tag(pytorch_root: str | Path) -> str:
    try:
        # 使用 git describe 命令获取版本标签
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=pytorch_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        # 如果标签符合 RELEASE_PATTERN 模式，则返回该标签
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN

# 获取 PyTorch 的版本号
def get_torch_version(sha: str | None = None) -> str:
    # 获取 PyTorch 根目录路径
    pytorch_root = Path(__file__).absolute().parent.parent
    # 从 version.txt 文件中读取版本号
    version = open(pytorch_root / "version.txt").read().strip()

    # 如果环境变量中指定了 PYTORCH_BUILD_VERSION，则使用其构建版本号
    if os.getenv("PYTORCH_BUILD_VERSION"):
        assert os.getenv("PYTORCH_BUILD_NUMBER") is not None
        build_number = int(os.getenv("PYTORCH_BUILD_NUMBER", ""))
        version = os.getenv("PYTORCH_BUILD_VERSION", "")
        # 如果构建号大于 1，则添加后缀 .post + 构建号
        if build_number > 1:
            version += ".post" + str(build_number)
    # 否则，如果 SHA 不是未知的，则添加 +git 后缀和 SHA 的前七位
    elif sha != UNKNOWN:
        if sha is None:
            sha = get_sha(pytorch_root)
        version += "+git" + sha[:7]
    return version

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Generate torch/version.py from build and environment metadata."
    )
    # 添加命令行参数：是否调试模式
    parser.add_argument(
        "--is-debug",
        "--is_debug",
        type=distutils.util.strtobool,
        help="Whether this build is debug mode or not.",
    )
    # 添加命令行参数：CUDA 版本
    parser.add_argument("--cuda-version", "--cuda_version", type=str)
    # 添加命令行参数：HIP 版本
    parser.add_argument("--hip-version", "--hip_version", type=str)

    # 解析命令行参数
    args = parser.parse_args()

    # 断言确保 is_debug 参数不为 None
    assert args.is_debug is not None
    # 如果 CUDA 版本为空字符串，则设置为 None
    args.cuda_version = None if args.cuda_version == "" else args.cuda_version
    # 如果 HIP 版本为空字符串，则设置为 None
    args.hip_version = None if args.hip_version == "" else args.hip_version

    # 获取 PyTorch 根目录路径
    pytorch_root = Path(__file__).parent.parent
    # 设置版本文件路径
    version_path = pytorch_root / "torch" / "version.py"
    # 首先尝试获取版本标签，如果未找到，则使用 SHA
    tagged_version = get_tag(pytorch_root)
    sha = get_sha(pytorch_root)
    if tagged_version == UNKNOWN:
        version = get_torch_version(sha)
    else:
        version = tagged_version
    # 使用写入模式打开文件 version_path，文件句柄为 f
    with open(version_path, "w") as f:
        # 写入类型提示和空行到文件
        f.write("from typing import Optional\n\n")
        # 写入定义 __all__ 的列表
        f.write("__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']\n")
        # 写入当前版本号到 __version__ 变量
        f.write(f"__version__ = '{version}'\n")
        # 写入是否为调试模式的布尔值，根据 args.is_debug 决定
        # 注意：这并不完全准确，因为库代码可能是用 DEBUG 编译的，但 csrc 可能没有 DEBUG（这种情况下，
        # 该代码会错误地声称是发布版）
        f.write(f"debug = {repr(bool(args.is_debug))}\n")
        # 写入 CUDA 版本号的可选字符串表示，根据 args.cuda_version 决定
        f.write(f"cuda: Optional[str] = {repr(args.cuda_version)}\n")
        # 写入 Git 版本号的字符串表示，根据 sha 决定
        f.write(f"git_version = {repr(sha)}\n")
        # 写入 HIP 版本号的可选字符串表示，根据 args.hip_version 决定
        f.write(f"hip: Optional[str] = {repr(args.hip_version)}\n")
```