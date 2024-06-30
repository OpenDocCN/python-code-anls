# `D:\src\scipysrc\scipy\tools\check_python_h_first.py`

```
#!/usr/bin/env python
"""Check that Python.h is included before any stdlib headers.

May be a bit overzealous, but it should get the job done.
"""

import argparse
import fnmatch
import os.path
import re
import subprocess
import sys

HEADER_PATTERN = re.compile(
    r'^\s*#\s*include\s*[<"]((?:\w+/)*\w+(?:\.h[hp+]{0,2})?)[>"]\s*$'
)

PYTHON_INCLUDING_HEADERS = [
    "Python.h",
    # This isn't all of Python.h, but it is the visibility macros
    "pyconfig.h",
    "numpy/arrayobject.h",
    "numpy/ndarrayobject.h",
    "numpy/npy_common.h",
    "numpy/npy_math.h",
    "numpy/random/distributions.h",
    "pybind11/pybind11.h",
    # Boost::Python
    "boost/python.hpp",
    "boost/python/args.hpp",
    "boost/python/detail/prefix.hpp",
    "boost/python/detail/wrap_python.hpp",
    "boost/python/ssize_t.hpp",
    "boost/python/object.hpp",
    "boost/mpi/python.hpp",
    # Pythran
    "pythonic/core.hpp",
    # Python-including headers the sort doesn't pick up
    "ni_support.h",
]
LEAF_HEADERS = []

C_CPP_EXTENSIONS = (".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx")
# check against list in diff_files

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument(
    "--diff-against",
    dest="branch",
    type=str,
    default=None,
    help="Diff against "
    "this branch and lint modified files. Use either "
    "`--diff-against` or `--files`, but not both. "
    "Likely to produce false positives.",
)
PARSER.add_argument(
    "files",
    nargs="*",
    help="Lint these files or directories; " "use **/*.py to lint all files",
)


def check_python_h_included_first(name_to_check: str) -> int:
    """Check that the passed file includes Python.h first if it does at all.

    Perhaps overzealous, but that should work around concerns with
    recursion.

    Parameters
    ----------
    name_to_check : str
        The name of the file to check.

    Returns
    -------
    int
        The number of headers before Python.h
    """
    included_python = False  # 是否已经包含了 Python.h
    included_non_python_header = []  # 除 Python.h 外的其他头文件列表
    warned_python_construct = False  # 是否已经发出 Python 构造的警告
    basename_to_check = os.path.basename(name_to_check)  # 获取文件的基本名称
    in_comment = False  # 是否当前在注释中
    includes_headers = False  # 是否存在头文件的包含操作
    # 打开需要检查的文件，使用默认的只读模式，上下文管理器确保文件关闭
    with open(name_to_check) as in_file:
        # 遍历文件的每一行，同时记录行号，enumerate 的第二个参数指定起始行号为 1
        for i, line in enumerate(in_file, 1):
            # 简单的注释解析
            # 假定 /*...*/ 样式的注释占据单独的一行
            if "/*" in line:
                # 如果当前行包含 /* 但不包含 */，表示进入了注释块
                if "*/" not in line:
                    in_comment = True
                # 如果当前行同时包含 */，则注释块结束，重置标志
                continue  # 继续处理下一行

            # 如果当前处于注释块中，寻找注释块结束标志 */
            if in_comment:
                if "*/" in line:
                    in_comment = False
                continue  # 继续处理下一行

            # 使用正则表达式匹配当前行是否符合 HEADER_PATTERN
            match = HEADER_PATTERN.match(line)
            if match:
                includes_headers = True
                # 获取匹配到的头文件名
                this_header = match.group(1)
                # 如果头文件名在 PYTHON_INCLUDING_HEADERS 中
                if this_header in PYTHON_INCLUDING_HEADERS:
                    # 如果已经包含非 Python 头文件但未包含 Python 头文件
                    if included_non_python_header and not included_python:
                        # 输出警告信息，指明文件名及行号
                        print(
                            f"Header before Python.h in file {name_to_check:s}\n"
                            f"Python.h on line {i:d}, other header(s) on line(s)"
                            f" {included_non_python_header}",
                            file=sys.stderr,
                        )
                    # 标记已经包含 Python 头文件
                    included_python = True
                    # 将当前文件名添加到 PYTHON_INCLUDING_HEADERS 中
                    PYTHON_INCLUDING_HEADERS.append(basename_to_check)
                # 如果头文件不是 Python 头文件，并且满足特定条件
                elif not included_python and (
                    "numpy" in this_header
                    and this_header != "numpy/utils.h"
                    or "python" in this_header
                ):
                    # 输出警告信息，指明文件名、头文件名及行号
                    print(
                        f"Python.h not included before python-including header "
                        f"in file {name_to_check:s}\n"
                        f"{this_header:s} on line {i:d}",
                        file=sys.stderr,
                    )
                # 如果头文件不是 Python 头文件，并且不在 LEAF_HEADERS 中
                elif not included_python and this_header not in LEAF_HEADERS:
                    # 记录未包含的非 Python 头文件行号
                    included_non_python_header.append(i)
            # 如果未包含 Python 头文件，并且未发出 Python 构造函数的警告，并且不是 .h 文件
            elif (
                not included_python
                and not warned_python_construct
                and ".h" not in basename_to_check
            ) and ("py::" in line or "PYBIND11_" in line or "npy_" in line):
                # 输出警告信息，指明文件名及行号
                print(
                    "Python-including header not used before python constructs "
                    f"in file {name_to_check:s}\nConstruct on line {i:d}",
                    file=sys.stderr,
                )
                # 标记已发出 Python 构造函数的警告
                warned_python_construct = True

    # 如果存在头文件包含，将当前头文件名添加到 LEAF_HEADERS 中
    if includes_headers:
        LEAF_HEADERS.append(this_header)

    # 返回是否包含了 Python 头文件以及未包含的非 Python 头文件的数量
    return included_python and len(included_non_python_header)
# 处理给定文件列表，返回不按顺序的文件数量
def process_files(file_list: list[str]) -> int:
    # 初始化计数器
    n_out_of_order = 0
    # 对文件列表按照是否包含'h'在文件扩展名中排序
    for name_to_check in sorted(
        file_list, key=lambda name: "h" not in os.path.splitext(name)[1].lower()
    ):
        try:
            # 调用函数检查是否Python头文件优先包含
            n_out_of_order += check_python_h_included_first(name_to_check)
        except UnicodeDecodeError:
            # 捕获Unicode解码错误，打印文件名和错误信息到标准输出
            print(f"File {name_to_check:s} not utf-8", sys.stdout)
    # 返回不按顺序的文件数量
    return n_out_of_order


# 在指定根目录中查找C和C++文件并返回它们的列表
def find_c_cpp_files(root: str) -> list[str]:

    result = []

    # 递归遍历根目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk("scipy"):
        # 假设其他人已经检查了boost，排除"build", ".git", "boost"目录
        for name in ("build", ".git", "boost"):
            try:
                dirnames.remove(name)
            except ValueError:
                pass
        # 移除dirnames中以'.p'结尾的目录名
        for name in fnmatch.filter(dirnames, "*.p"):
            dirnames.remove(name)
        # 将满足C_CPP_EXTENSIONS中文件扩展名条件的文件路径添加到result列表中
        result.extend(
            [
                os.path.join(dirpath, name)
                for name in filenames
                if os.path.splitext(name)[1].lower() in C_CPP_EXTENSIONS
            ]
        )
    # 返回找到的C和C++文件列表
    return result


# 根据给定SHA找到与之前差异的文件列表
def diff_files(sha: str) -> list[str]:
    """Find the diff since the given SHA.

    Adapted from lint.py
    """
    # 运行git diff命令获取变更的文件列表
    res = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            sha,
            "--",
            # 根据C_CPP_EXTENSIONS检查文件
            "*.[ch]",
            "*.[ch]pp",
            "*.[ch]xx",
            "*.cc",
            "*.hh",
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    # 检查命令运行的返回码，如果有错误将会抛出异常
    res.check_returncode()
    # 返回变更文件列表，去除末尾的空字符分隔符
    return [f for f in res.stdout.split("\0") if f]


if __name__ == "__main__":
    # 导入lint.py中的find_branch_point函数
    from lint import find_branch_point

    # 解析命令行参数
    args = PARSER.parse_args()

    # 根据参数情况确定要处理的文件列表
    if not ((len(args.files) == 0) ^ (args.branch is None)):
        # 如果没有文件参数且分支参数不为空，则找到分支点之后的文件变更列表
        files = find_c_cpp_files("scipy")
    elif args.branch:
        # 否则，根据分支名找到分支点，并计算分支点后的文件变更列表
        branch_point = find_branch_point(args.branch)
        files = diff_files(branch_point)
    else:
        # 否则使用命令行传入的文件列表
        files = args.files

    # 检查头文件中是否包含Python.h并计算不按顺序的文件数量
    n_out_of_order = process_files(files)
    # 程序正常退出，返回不按顺序的文件数量作为退出码
    sys.exit(n_out_of_order)
```