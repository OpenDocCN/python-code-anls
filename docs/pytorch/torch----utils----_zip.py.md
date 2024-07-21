# `.\pytorch\torch\utils\_zip.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块
import argparse  # 用于解析命令行参数的模块
import glob  # 用于查找文件路径模式匹配的模块
import os  # 提供了与操作系统交互的功能
from pathlib import Path  # 提供了操作路径的对象导向的API
from zipfile import ZipFile  # 用于创建和读取ZIP文件的模块

# 被排除在外的标准库模块列表，用于：
# 1. 减小最终压缩文件的大小
# 2. 移除我们不想支持的功能。
DENY_LIST = [
    # Unix数据库接口
    "dbm",
    # ncurses绑定（终端界面）
    "curses",
    # Tcl/Tk图形用户界面
    "tkinter",
    # 测试标准库的模块
    "test",
    "tests",
    "idle_test",
    "__phello__.foo.py",
    # importlib冻结模块。这些已经内置到CPython中。
    "_bootstrap.py",
    "_bootstrap_external.py",
]

strip_file_dir = ""


def remove_prefix(text, prefix):
    # 如果文本以指定前缀开头，则移除前缀返回剩余部分
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def write_to_zip(file_path, strip_file_path, zf, prepend_str=""):
    # 构造处理后的文件路径，去掉指定目录前缀
    stripped_file_path = prepend_str + remove_prefix(file_path, strip_file_dir + "/")
    path = Path(stripped_file_path)
    # 如果文件名在DENY_LIST中，则跳过不处理
    if path.name in DENY_LIST:
        return
    # 向ZIP文件中写入文件，使用处理后的文件路径作为存储路径
    zf.write(file_path, stripped_file_path)


def main() -> None:
    global strip_file_dir
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Zip py source")
    parser.add_argument("paths", nargs="*", help="Paths to zip.")
    parser.add_argument(
        "--install-dir", "--install_dir", help="Root directory for all output files"
    )
    parser.add_argument(
        "--strip-dir",
        "--strip_dir",
        help="The absolute directory we want to remove from zip",
    )
    parser.add_argument(
        "--prepend-str",
        "--prepend_str",
        help="A string to prepend onto all paths of a file in the zip",
        default="",
    )
    parser.add_argument("--zip-name", "--zip_name", help="Output zip name")

    args = parser.parse_args()

    # 组合ZIP文件名，安装目录+输出文件名
    zip_file_name = args.install_dir + "/" + args.zip_name
    strip_file_dir = args.strip_dir
    prepend_str = args.prepend_str
    # 创建一个新的ZIP文件对象
    zf = ZipFile(zip_file_name, mode="w")

    # 遍历所有指定的路径
    for p in sorted(args.paths):
        if os.path.isdir(p):
            # 如果路径是目录，则递归查找所有.py文件
            files = glob.glob(p + "/**/*.py", recursive=True)
            for file_path in sorted(files):
                # 去掉绝对路径中的指定目录前缀，并写入ZIP文件
                write_to_zip(
                    file_path, strip_file_dir + "/", zf, prepend_str=prepend_str
                )
        else:
            # 如果路径是文件，则去掉指定目录前缀，并写入ZIP文件
            write_to_zip(p, strip_file_dir + "/", zf, prepend_str=prepend_str)


if __name__ == "__main__":
    main()  # pragma: no cover
```