# `.\pytorch\tools\nvcc_fix_deps.py`

```py
"""Tool to fix the nvcc's dependecy file output

Usage: python nvcc_fix_deps.py nvcc [nvcc args]...

This wraps nvcc to ensure that the dependency file created by nvcc with the
-MD flag always uses absolute paths. nvcc sometimes outputs relative paths,
which ninja interprets as an unresolved dependency, so it triggers a rebuild
of that file every time.

The easiest way to use this is to define:

CMAKE_CUDA_COMPILER_LAUNCHER="python;tools/nvcc_fix_deps.py;ccache"

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TextIO


def resolve_include(path: Path, include_dirs: list[Path]) -> Path:
    # 遍历 include_dirs 中的路径，尝试解析给定的相对路径 path
    for include_path in include_dirs:
        abs_path = include_path / path
        # 如果找到了存在的绝对路径，直接返回
        if abs_path.exists():
            return abs_path

    # 如果所有路径均不存在，抛出运行时错误并展示尝试过的路径列表
    paths = "\n    ".join(str(d / path) for d in include_dirs)
    raise RuntimeError(
        f"""
ERROR: Failed to resolve dependency:
    {path}
Tried the following paths, but none existed:
    {paths}
"""
    )


def repair_depfile(depfile: TextIO, include_dirs: list[Path]) -> None:
    changes_made = False
    out = ""
    for line in depfile:
        # 如果行中包含 ":"，截取到最后一个 ":"，保留前面的内容并记录下来
        if ":" in line:
            colon_pos = line.rfind(":")
            out += line[: colon_pos + 1]
            line = line[colon_pos + 1 :]

        line = line.strip()

        # 如果行以 "\\" 结尾，去掉 "\\" 并标记 end 为 " \\"
        if line.endswith("\\"):
            end = " \\"
            line = line[:-1].strip()
        else:
            end = ""

        # 将当前行转换为 Path 对象
        path = Path(line)

        # 如果路径不是绝对路径，尝试用 include_dirs 解析路径
        if not path.is_absolute():
            changes_made = True
            path = resolve_include(path, include_dirs)
        
        # 构造修复后的行并添加到输出中
        out += f"    {path}{end}\n"

    # 如果有路径被修改，重新写入整个文件
    if changes_made:
        depfile.seek(0)
        depfile.write(out)
        depfile.truncate()


PRE_INCLUDE_ARGS = ["-include", "--pre-include"]
POST_INCLUDE_ARGS = ["-I", "--include-path", "-isystem", "--system-include"]


def extract_include_arg(include_dirs: list[Path], i: int, args: list[str]) -> None:
    def extract_one(name: str, i: int, args: list[str]) -> str | None:
        arg = args[i]
        if arg == name:
            return args[i + 1]
        if arg.startswith(name):
            arg = arg[len(name) :]
            return arg[1:] if arg[0] == "=" else arg
        return None

    # 根据参数名称在 args 中提取路径信息，更新 include_dirs
    for name in PRE_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.insert(0, Path(path).resolve())
            return

    for name in POST_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.append(Path(path).resolve())
            return


if __name__ == "__main__":
    ret = subprocess.run(
        sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr
    )

    depfile_path = None
    include_dirs = []

    # 解析 nvcc 相关参数，更新 include_dirs
    args = sys.argv[2:]
    # 使用 enumerate 遍历参数列表 args，i 是索引，arg 是当前参数值
    for i, arg in enumerate(args):
        # 如果当前参数是 "-MF"，则下一个参数 args[i + 1] 是依赖文件的路径
        if arg == "-MF":
            depfile_path = Path(args[i + 1])
        # 如果当前参数是 "-c"
        elif arg == "-c":
            # 将 CUDA 文件的基础路径添加到 include_dirs 中
            include_dirs.append(Path(args[i + 1]).resolve().parent)
        else:
            # 对于其他参数调用 extract_include_arg 函数处理
            extract_include_arg(include_dirs, i, args)

    # 如果存在依赖文件路径 depfile_path 并且该路径存在
    if depfile_path is not None and depfile_path.exists():
        # 以读写方式打开依赖文件
        with depfile_path.open("r+") as f:
            # 调用 repair_depfile 函数修复依赖文件，传入 include_dirs
            repair_depfile(f, include_dirs)

    # 退出程序，使用 ret.returncode 作为退出码
    sys.exit(ret.returncode)
```