# `D:\src\scipysrc\pandas\generate_version.py`

```
#!/usr/bin/env python3

# Note: This file has to live next to setup.py or versioneer will not work

# 导入 argparse 库，用于命令行参数解析
import argparse
# 导入 os 和 sys 库，用于操作系统相关的功能和系统路径
import os
import sys

# 导入 versioneer 库，用于自动化版本控制
import versioneer

# 将当前路径添加到系统路径中，确保能够找到需要的模块
sys.path.insert(0, "")

# 定义一个函数，用于写入版本信息到指定路径的文件中
def write_version_info(path) -> None:
    version = None
    git_version = None

    try:
        # 尝试导入 _version_meson 模块，获取版本信息和 Git 版本信息
        import _version_meson
        version = _version_meson.__version__
        git_version = _version_meson.__git_version__
    except ImportError:
        # 如果导入失败，使用 versioneer 库获取版本信息和 Git 版本信息
        version = versioneer.get_version()
        git_version = versioneer.get_versions()["full-revisionid"]

    # 如果设置了 MESON_DIST_ROOT 环境变量，将文件写入到指定路径
    if os.environ.get("MESON_DIST_ROOT"):
        path = os.path.join(os.environ.get("MESON_DIST_ROOT"), path)
    
    # 打开指定路径的文件，写入版本信息和 Git 版本信息
    with open(path, "w", encoding="utf-8") as file:
        file.write(f'__version__="{version}"\n')
        file.write(f'__git_version__="{git_version}"\n')

# 主函数，用于解析命令行参数并执行相应操作
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数选项，指定输出文件路径
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to write version info to",
        required=False,
    )
    
    # 添加命令行参数选项，是否打印版本信息
    parser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="Whether to print out the version",
        required=False,
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 如果指定了输出文件路径
    if args.outfile:
        # 检查输出文件是否以 .py 结尾，否则抛出 ValueError 异常
        if not args.outfile.endswith(".py"):
            raise ValueError(
                f"Output file must be a Python file. "
                f"Got: {args.outfile} as filename instead"
            )
        
        # 调用写入版本信息的函数，将版本信息写入到指定的输出文件中
        write_version_info(args.outfile)

    # 如果指定了打印版本信息的选项
    if args.print:
        try:
            # 尝试导入 _version_meson 模块，获取版本信息
            import _version_meson
            version = _version_meson.__version__
        except ImportError:
            # 如果导入失败，使用 versioneer 库获取版本信息
            version = versioneer.get_version()
        
        # 打印获取到的版本信息
        print(version)

# 调用主函数，开始执行程序
main()
```