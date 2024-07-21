# `.\pytorch\tools\linter\clang_tidy\generate_build_files.py`

```py
# 导入未来的类型注解特性
from __future__ import annotations

# 导入操作系统相关功能
import os
# 导入子进程管理功能
import subprocess
# 导入系统相关功能
import sys

# 定义一个函数，用于运行命令并打印命令内容
def run_cmd(cmd: list[str]) -> None:
    # 打印当前正在运行的命令
    print(f"Running: {cmd}")
    # 运行指定的命令，并捕获其输出
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    # 解码并去除输出结果的空白字符，获取标准输出和标准错误输出
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    # 打印标准输出和标准错误输出
    print(stdout)
    print(stderr)
    # 如果命令返回值不为 0，打印失败信息并退出程序
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        sys.exit(1)

# 定义一个函数，用于更新子模块
def update_submodules() -> None:
    # 运行命令：git submodule update --init --recursive
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])

# 定义一个函数，用于生成编译命令
def gen_compile_commands() -> None:
    # 设置环境变量
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_PRECOMPILED_HEADERS"] = "1"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    # 运行命令：python3 setup.py --cmake-only build
    run_cmd([sys.executable, "setup.py", "--cmake-only", "build"])

# 定义一个函数，用于运行自动生成工具
def run_autogen() -> None:
    # 运行命令：python3 -m torchgen.gen -s aten/src/ATen -d build/aten/src/ATen --per-operator-headers
    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
            "--per-operator-headers",
        ]
    )

    # 运行命令：python3 tools/setup_helpers/generate_code.py --native-functions-path aten/src/ATen/native/native_functions.yaml --tags-path aten/src/ATen/native/tags.yaml --gen-lazy-ts-backend
    run_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen-lazy-ts-backend",
        ]
    )

# 定义一个函数，用于生成构建文件
def generate_build_files() -> None:
    # 更新子模块
    update_submodules()
    # 生成编译命令
    gen_compile_commands()
    # 运行自动生成工具
    run_autogen()

# 如果当前文件作为主程序运行，则执行生成构建文件的操作
if __name__ == "__main__":
    generate_build_files()
```