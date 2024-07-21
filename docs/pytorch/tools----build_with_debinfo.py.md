# `.\pytorch\tools\build_with_debinfo.py`

```py
#!/usr/bin/env python3
# Tool quickly rebuild one or two files with debug info
# Mimics following behavior:
# - touch file
# - ninja -j1 -v -n torch_python | sed -e 's/-O[23]/-g/g' -e 's#\[[0-9]\+\/[0-9]\+\] \+##' |sh
# - Copy libs from build/lib to torch/lib folder

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


PYTORCH_ROOTDIR = Path(__file__).resolve().parent.parent
TORCH_DIR = PYTORCH_ROOTDIR / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
BUILD_DIR = PYTORCH_ROOTDIR / "build"
BUILD_LIB_DIR = BUILD_DIR / "lib"


def check_output(args: list[str], cwd: str | None = None) -> str:
    """Execute a command and return its output as a decoded string."""
    return subprocess.check_output(args, cwd=cwd).decode("utf-8")


def parse_args() -> Any:
    """Parse command line arguments."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Incremental build PyTorch with debinfo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def get_lib_extension() -> str:
    """Return the file extension for shared libraries based on the platform."""
    if sys.platform == "linux":
        return "so"
    if sys.platform == "darwin":
        return "dylib"
    raise RuntimeError(f"Usupported platform {sys.platform}")


def create_symlinks() -> None:
    """Create symbolic links from build/lib to torch/lib."""
    if not TORCH_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {TORCH_LIB_DIR} does not exist")
    if not BUILD_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {BUILD_LIB_DIR} does not exist")
    for torch_lib in TORCH_LIB_DIR.glob(f"*.{get_lib_extension()}"):
        if torch_lib.is_symlink():
            continue
        build_lib = BUILD_LIB_DIR / torch_lib.name
        if not build_lib.exists():
            raise RuntimeError(f"Can't find {build_lib} corresponding to {torch_lib}")
        torch_lib.unlink()
        torch_lib.symlink_to(build_lib)


def has_build_ninja() -> bool:
    """Check if the build directory contains the 'build.ninja' file."""
    return (BUILD_DIR / "build.ninja").exists()


def is_devel_setup() -> bool:
    """Check if the current setup is for development."""
    output = check_output([sys.executable, "-c", "import torch;print(torch.__file__)"])
    return output.strip() == str(TORCH_DIR / "__init__.py")


def create_build_plan() -> list[tuple[str, str]]:
    """Create a build plan by parsing the output of 'ninja -j1 -v -n torch_python'."""
    output = check_output(
        ["ninja", "-j1", "-v", "-n", "torch_python"], cwd=str(BUILD_DIR)
    )
    rc = []
    for line in output.split("\n"):
        if not line.startswith("["):
            continue
        line = line.split("]", 1)[1].strip()
        if line.startswith(": &&") and line.endswith("&& :"):
            line = line[4:-4]
        line = line.replace("-O2", "-g").replace("-O3", "-g")
        name = line.split("-o ", 1)[1].split(" ")[0]
        rc.append((name, line))
    return rc


def main() -> None:
    """Main function to execute the script."""
    if sys.platform == "win32":
        print("Not supported on Windows yet")
        sys.exit(-95)
    # 如果不是开发设置，请输出提示信息并退出程序
    if not is_devel_setup():
        print(
            "Not a devel setup of PyTorch, please run `python3 setup.py develop --user` first"
        )
        sys.exit(-1)
    
    # 如果没有安装 Ninja 构建系统，请输出提示信息并退出程序
    if not has_build_ninja():
        print("Only ninja build system is supported at the moment")
        sys.exit(-1)
    
    # 解析命令行参数
    args = parse_args()
    
    # 遍历命令行参数中的文件列表，如果文件为 None 则跳过
    for file in args.files:
        if file is None:
            continue
        # 创建一个空文件（如果文件不存在则创建）
        Path(file).touch()
    
    # 创建构建计划
    build_plan = create_build_plan()
    
    # 如果构建计划为空，则打印信息并返回
    if len(build_plan) == 0:
        return print("Nothing to do")
    
    # 如果构建计划中的项超过 100 个，输出提示信息并退出程序
    if len(build_plan) > 100:
        print("More than 100 items needs to be rebuild, run `ninja torch_python` first")
        sys.exit(-1)
    
    # 遍历构建计划，按顺序构建每一项
    for idx, (name, cmd) in enumerate(build_plan):
        # 打印当前构建进度和构建项的名称
        print(f"[{idx + 1} / {len(build_plan)}] Building {name}")
        # 如果命令行参数中指定了详细输出，打印构建命令
        if args.verbose:
            print(cmd)
        # 使用 subprocess 调用指定的命令来执行构建，设置工作目录为 BUILD_DIR
        subprocess.check_call(["sh", "-c", cmd], cwd=BUILD_DIR)
    
    # 创建符号链接
    create_symlinks()
# 如果这个模块被直接执行（而不是被导入到其他模块中执行），那么执行以下代码
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```