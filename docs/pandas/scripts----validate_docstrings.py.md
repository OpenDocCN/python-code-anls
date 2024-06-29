# `D:\src\scipysrc\pandas\scripts\validate_docstrings.py`

```
# 指定脚本的解释器为 Python 3
#!/usr/bin/env python3
"""
Analyze docstrings to detect errors.

If no argument is provided, it does a quick check of docstrings and returns
a csv with all API functions and results of basic checks.

If a function or method is provided in the form "pandas.function",
"pandas.module.class.method", etc. a list of all errors in the docstring for
the specified function or method.

Usage::
    $ ./validate_docstrings.py
    $ ./validate_docstrings.py pandas.DataFrame.head
"""

# 导入未来的注释支持
from __future__ import annotations

# 导入必要的模块
import argparse  # 处理命令行参数的库
import collections  # Python 的集合数据类型的扩展
import doctest  # Python 自带的模块，用于测试文档中的代码示例
import importlib  # 提供了用于加载和重载 Python 模块的功能
import json  # 处理 JSON 数据的库
import os  # 提供与操作系统交互的功能
import pathlib  # 提供了操作文件和目录路径的类
import subprocess  # 启动和控制子进程的库
import sys  # 提供了与 Python 解释器相关的函数和变量
import tempfile  # 创建临时文件和目录的库

# 导入 matplotlib 的子模块 pyplot 和 matplotlib 本身
import matplotlib
import matplotlib.pyplot as plt

# 导入 numpydoc 的相关模块和函数
from numpydoc.docscrape import get_doc_object  # 获取文档对象的函数
from numpydoc.validate import (
    ERROR_MSGS as NUMPYDOC_ERROR_MSGS,  # numpydoc 的错误信息字典
    Validator,  # numpydoc 的验证器类
    validate,  # numpydoc 的验证函数
)

# 使用模板后端，matplotlib 不绘制图形
matplotlib.use("template")

# 忽略验证的 Styler 方法，这些方法的文档字符串不由我们拥有
IGNORE_VALIDATION = {
    "Styler.env",
    "Styler.template_html",
    "Styler.template_html_style",
    "Styler.template_html_table",
    "Styler.template_latex",
    "Styler.template_string",
    "Styler.loader",
    "errors.InvalidComparison",
    "errors.LossySetitemError",
    "errors.NoBufferPresent",
    "errors.IncompatibilityWarning",
    "errors.PyperclipException",
    "errors.PyperclipWindowsException",
}

# 私有类的名称列表，不应出现在公共文档字符串中
PRIVATE_CLASSES = ["NDFrame", "IndexOpsMixin"]

# 自定义的错误信息字典
ERROR_MSGS = {
    "GL04": "Private classes ({mentioned_private_classes}) should not be "
            "mentioned in public docstrings",
    "PD01": "Use 'array-like' rather than 'array_like' in docstrings.",
    "SA05": "{reference_name} in `See Also` section does not need `pandas` "
            "prefix, use {right_reference} instead.",
    "EX03": "flake8 error: line {line_number}, col {col_number}: {error_code} "
            "{error_message}",
    "EX04": "Do not import {imported_library}, as it is imported "
            "automatically for the examples (numpy as np, pandas as pd)",
}

# 所有错误信息的集合，包括 numpydoc 和自定义的错误信息
ALL_ERRORS = set(NUMPYDOC_ERROR_MSGS).union(set(ERROR_MSGS))

# 检查是否有重复的错误信息存在于 numpydoc 和自定义错误中
duplicated_errors = set(NUMPYDOC_ERROR_MSGS).intersection(set(ERROR_MSGS))
assert not duplicated_errors, (f"Errors {duplicated_errors} exist in both pandas "
                               "and numpydoc, should they be removed from pandas?")


def pandas_error(code, **kwargs):
    """
    Copy of the numpydoc error function, since ERROR_MSGS can't be updated
    with our custom errors yet.
    """
    # 自定义的 pandas 错误信息生成函数，根据错误码和参数生成具体的错误消息
    return code, ERROR_MSGS[code].format(**kwargs)


def get_api_items(api_doc_fd):
    """
    Yield information about all public API items.

    Parse api.rst file from the documentation, and extract all the functions,
    methods, classes, attributes... This should include all pandas public API.

    Parameters
    ----------
    api_doc_fd : file descriptor
        A file descriptor of the API documentation page, containing the table
        of contents with all the public API.

    Yields
    ------
    """
    """
    name : str
        The name of the object (e.g. 'pandas.Series.str.upper').
    func : function
        The object itself. In most cases this will be a function or method,
        but it can also be classes, properties, cython objects...
    section : str
        The name of the section in the API page where the object item is
        located.
    subsection : str
        The name of the subsection in the API page where the object item is
        located.
    """
    # 初始化变量，设置默认值
    current_module = "pandas"
    previous_line = current_section = current_subsection = ""
    position = None

    # 遍历 API 文档文件的每一行
    for line in api_doc_fd:
        line_stripped = line.strip()

        # 检查是否为分隔线，用来确定当前所属的节和子节
        if len(line_stripped) == len(previous_line):
            if set(line_stripped) == set("-"):
                current_section = previous_line
                continue
            if set(line_stripped) == set("~"):
                current_subsection = previous_line
                continue

        # 处理当前模块声明
        if line_stripped.startswith(".. currentmodule::"):
            current_module = line_stripped.replace(".. currentmodule::", "").strip()
            continue

        # 处理 ".. autosummary::" 指令
        if line_stripped == ".. autosummary::":
            position = "autosummary"
            continue

        # 处理自动摘要之后的项目列表
        if position == "autosummary":
            if line_stripped == "":
                position = "items"
                continue

        # 处理项目条目
        if position == "items":
            if line_stripped == "":
                position = None
                continue
            # 忽略在验证集合中的条目
            if line_stripped in IGNORE_VALIDATION:
                continue
            # 动态导入当前模块下的函数或对象
            func = importlib.import_module(current_module)
            for part in line_stripped.split("."):
                func = getattr(func, part)

            # 返回生成器结果，包括对象的完整名称、对象本身、当前节和子节名称
            yield (
                f"{current_module}.{line_stripped}",
                func,
                current_section,
                current_subsection,
            )

        previous_line = line_stripped
class PandasDocstring(Validator):
    # 定义一个名为 PandasDocstring 的类，继承自 Validator 类
    def __init__(self, func_name: str, doc_obj=None) -> None:
        # 初始化方法，接收函数名 func_name 和文档对象 doc_obj，默认为 None
        self.func_name = func_name
        # 将 func_name 存储在实例变量中
        if doc_obj is None:
            # 如果 doc_obj 为空，通过 func_name 获取文档对象
            doc_obj = get_doc_object(Validator._load_obj(func_name))
        super().__init__(doc_obj)
        # 调用父类 Validator 的初始化方法，传入 doc_obj

    @property
    def name(self):
        # 属性装饰器，返回实例的 func_name 属性
        return self.func_name

    @property
    def mentioned_private_classes(self):
        # 属性装饰器，返回 raw_doc 中提到的私有类列表
        return [klass for klass in PRIVATE_CLASSES if klass in self.raw_doc]

    @property
    def examples_source_code(self):
        # 属性装饰器，解析文档中的 doctest 示例代码，返回源代码列表
        lines = doctest.DocTestParser().get_examples(self.raw_doc)
        return [line.source for line in lines]

    def validate_pep8(self):
        # 方法，验证文档中的示例代码是否符合 PEP8 规范
        if not self.examples:
            return

        # 构建导入语句和示例代码的完整内容，避免 flake8 报错
        content = "".join(
            (
                "import numpy as np  # noqa: F401\n",
                "import pandas as pd  # noqa: F401\n",
                *self.examples_source_code,
            )
        )

        error_messages = []

        # 创建临时文件，写入 content 内容
        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        try:
            file.write(content)
            file.flush()
            # 构建 flake8 命令
            cmd = [
                sys.executable,
                "-m",
                "flake8",
                "--format=%(row)d\t%(col)d\t%(code)s\t%(text)s",
                "--max-line-length=88",
                "--ignore=E203,E3,W503,W504,E402,E731,E128,E124,E704",
                file.name,
            ]
            # 运行 flake8 命令，捕获输出
            response = subprocess.run(cmd, capture_output=True, check=False, text=True)
            # 处理 stdout 和 stderr 的输出
            for output in ("stdout", "stderr"):
                out = getattr(response, output)
                out = out.replace(file.name, "")
                messages = out.strip("\n").splitlines()
                if messages:
                    error_messages.extend(messages)
        finally:
            # 关闭并删除临时文件
            file.close()
            os.unlink(file.name)

        # 解析错误信息，返回错误码、消息、行号和列号
        for error_message in error_messages:
            line_number, col_number, error_code, message = error_message.split(
                "\t", maxsplit=3
            )
            # 注意：从行号中减去 2，因为示例代码前面有两行导入语句
            yield error_code, message, int(line_number) - 2, int(col_number)

    def non_hyphenated_array_like(self):
        # 方法，检查文档中是否提到了非连字符的 array_like
        return "array_like" in self.raw_doc


def pandas_validate(func_name: str):
    """
    调用 numpydoc 进行验证，并添加特定于 pandas 的错误。

    Parameters
    ----------
    func_name : str
        要验证其文档字符串的对象名称。

    Returns
    -------
    dict
        包含文档字符串信息和发现的错误信息。
    """
    func_obj = Validator._load_obj(func_name)
    # 加载函数对象，有些对象是实例，如 IndexSlice，numpydoc 无法验证
    # 获取函数对象的文档对象，用于后续处理
    doc_obj = get_doc_object(func_obj, doc=func_obj.__doc__)
    # 根据函数名和文档对象创建 PandasDocstring 对象
    doc = PandasDocstring(func_name, doc_obj)
    # 对文档对象进行验证，返回验证结果
    result = validate(doc_obj)
    # 获取文档中提到的私有类的列表
    mentioned_errs = doc.mentioned_private_classes
    # 如果存在提到的私有类，则将错误信息添加到结果中
    if mentioned_errs:
        result["errors"].append(
            pandas_error("GL04", mentioned_private_classes=", ".join(mentioned_errs))
        )

    # 如果文档中存在参考链接
    if doc.see_also:
        # 对每个参考链接进行处理，检查是否符合 "pandas." 开头的命名空间
        result["errors"].extend(
            pandas_error(
                "SA05",
                reference_name=rel_name,
                right_reference=rel_name[len("pandas."):],
            )
            for rel_name in doc.see_also
            if rel_name.startswith("pandas.")
        )

    # 初始化示例错误信息字符串
    result["examples_errs"] = ""
    # 如果文档中存在示例代码
    if doc.examples:
        # 验证示例代码是否符合 PEP8 规范，获取错误信息并添加到结果中
        for error_code, error_message, line_number, col_number in doc.validate_pep8():
            result["errors"].append(
                pandas_error(
                    "EX03",
                    error_code=error_code,
                    error_message=error_message,
                    line_number=line_number,
                    col_number=col_number,
                )
            )
        # 获取示例代码的源代码文本
        examples_source_code = "".join(doc.examples_source_code)
        # 检查示例代码中是否存在错误的库导入，并添加相应错误信息到结果中
        result["errors"].extend(
            pandas_error("EX04", imported_library=wrong_import)
            for wrong_import in ("numpy", "pandas")
            if f"import {wrong_import}" in examples_source_code
        )

    # 如果文档中存在非连字符形式的类数组，则添加相应错误信息到结果中
    if doc.non_hyphenated_array_like():
        result["errors"].append(pandas_error("PD01"))

    # 关闭所有的 matplotlib 图形窗口
    plt.close("all")
    # 返回最终的结果字典
    return result
def validate_all(prefix, ignore_deprecated=False):
    """
    Execute the validation of all docstrings, and return a dict with the
    results.

    Parameters
    ----------
    prefix : str or None
        If provided, only the docstrings that start with this pattern will be
        validated. If None, all docstrings will be validated.
    ignore_deprecated: bool, default False
        If True, deprecated objects are ignored when validating docstrings.

    Returns
    -------
    dict
        A dictionary with an item for every function/method... containing
        all the validation information.
    """
    # 初始化一个空字典来存储验证结果
    result = {}
    # 用于记录已经处理过的函数名及其对应的共享代码位置
    seen = {}

    # 遍历获取所有的 API 项
    for func_name, _, section, subsection in get_all_api_items():
        # 如果有指定前缀且函数名不以该前缀开头，则跳过
        if prefix and not func_name.startswith(prefix):
            continue
        # 对函数名进行文档字符串验证，返回验证信息
        doc_info = pandas_validate(func_name)
        # 如果设置了忽略已弃用，并且当前函数被标记为已弃用，则跳过
        if ignore_deprecated and doc_info["deprecated"]:
            continue
        # 将验证信息存入结果字典中以函数名为键
        result[func_name] = doc_info

        # 构建共享代码键，用文件名及行号标识共享的代码位置
        shared_code_key = doc_info["file"], doc_info["file_line"]
        # 获取已经记录的共享代码，若没有则为空字符串
        shared_code = seen.get(shared_code_key, "")
        # 更新结果字典中当前函数的信息，加入是否在 API 中的标记、部分和子部分名称，以及共享代码位置
        result[func_name].update(
            {
                "in_api": True,
                "section": section,
                "subsection": subsection,
                "shared_code_with": shared_code,
            }
        )

        # 记录已处理过的共享代码位置及对应的函数名
        seen[shared_code_key] = func_name

    # 返回最终的验证结果字典
    return result


def get_all_api_items():
    # 获取当前文件的父目录的父目录，作为基础路径
    base_path = pathlib.Path(__file__).parent.parent
    # 构建 API 文档文件夹的完整路径
    api_doc_fnames = pathlib.Path(base_path, "doc", "source", "reference")
    # 遍历 API 文档文件夹下的所有 .rst 文件
    for api_doc_fname in api_doc_fnames.glob("*.rst"):
        # 打开每个 .rst 文件，使用 UTF-8 编码读取
        with open(api_doc_fname, encoding="utf-8") as f:
            # 调用函数从打开的文件中获取 API 项
            yield from get_api_items(f)


def print_validate_all_results(
    output_format: str,
    prefix: str | None,
    ignore_deprecated: bool,
    ignore_errors: dict[str, set[str]],
):
    # 检查输出格式是否为预定义的类型之一，否则抛出值错误异常
    if output_format not in ("default", "json", "actions"):
        raise ValueError(f'Unknown output_format "{output_format}"')
    # 如果忽略错误参数为 None，则初始化为空字典
    if ignore_errors is None:
        ignore_errors = {}

    # 执行验证所有函数文档字符串的操作，获取验证结果
    result = validate_all(prefix, ignore_deprecated)

    # 如果输出格式为 JSON，则将结果以 JSON 格式写入标准输出并返回 0
    if output_format == "json":
        sys.stdout.write(json.dumps(result))
        return 0

    # 根据输出格式设定前缀字符串，用于后续输出错误信息
    prefix = "##[error]" if output_format == "actions" else ""
    # 初始化退出状态为 0
    exit_status = 0
    # 对于每个函数名和其检查结果的字典项进行迭代
    for func_name, res in result.items():
        # 提取错误消息字典
        error_messages = dict(res["errors"])
        # 获取实际失败的错误码集合
        actual_failures = set(error_messages)
        # 获取应忽略的错误码集合，可能是特定函数名的或通用的
        expected_failures = (ignore_errors.get(func_name, set())
                             | ignore_errors.get(None, set()))
        # 检查实际失败但不应忽略的错误码
        for err_code in actual_failures - expected_failures:
            # 输出错误信息到标准输出
            sys.stdout.write(
                f'{prefix}{res["file"]}:{res["file_line"]}:'
                f'{err_code}:{func_name}:{error_messages[err_code]}\n'
            )
            # 增加退出状态计数
            exit_status += 1
        # 检查应忽略的错误码但实际未失败的情况
        for err_code in ignore_errors.get(func_name, set()) - actual_failures:
            # 输出预期不失败但未失败的消息到标准输出
            sys.stdout.write(
                f'{prefix}{res["file"]}:{res["file_line"]}:'
                f"{err_code}:{func_name}:"
                "EXPECTED TO FAIL, BUT NOT FAILING\n"
            )
            # 增加退出状态计数
            exit_status += 1
    
    # 返回最终的退出状态码
    return exit_status
# 定义打印验证结果的函数，接受函数名、错误忽略字典作为参数，并返回整数结果
def print_validate_one_results(func_name: str,
                               ignore_errors: dict[str, set[str]]) -> int:
    # 定义一个内部函数 header，用于生成指定标题的装饰性文本
    def header(title, width=80, char="#") -> str:
        # 构造完整长度的分隔线
        full_line = char * width
        # 计算标题两侧填充字符的长度
        side_len = (width - len(title) - 2) // 2
        # 根据计算结果生成格式化后的标题行
        adj = "" if len(title) % 2 == 0 else " "
        title_line = f"{char * side_len} {title}{adj} {char * side_len}"

        return f"\n{full_line}\n{title_line}\n{full_line}\n\n"

    # 调用外部函数执行验证函数，返回验证结果
    result = pandas_validate(func_name)

    # 过滤掉在忽略错误字典中指定的错误代码
    result["errors"] = [(code, message) for code, message in result["errors"]
                        if code not in ignore_errors.get(None, set())]

    # 输出格式化的文档字符串相关信息到标准错误流
    sys.stderr.write(header(f"Docstring ({func_name})"))
    sys.stderr.write(f"{result['docstring']}\n")

    # 输出验证结果相关信息到标准错误流
    sys.stderr.write(header("Validation"))
    if result["errors"]:
        sys.stderr.write(f'{len(result["errors"])} Errors found for `{func_name}`:\n')
        for err_code, err_desc in result["errors"]:
            sys.stderr.write(f"\t{err_code}\t{err_desc}\n")
    else:
        sys.stderr.write(f'Docstring for "{func_name}" correct. :)\n')

    # 如果存在文档测试的错误信息，则输出到标准错误流
    if result["examples_errs"]:
        sys.stderr.write(header("Doctests"))
        sys.stderr.write(result["examples_errs"])

    # 返回错误数量的总和（验证错误数加文档测试错误数）
    return len(result["errors"]) + len(result["examples_errs"])


# 内部函数，用于格式化处理原始的错误忽略设置，返回格式化后的忽略错误字典
def _format_ignore_errors(raw_ignore_errors):
    ignore_errors = collections.defaultdict(set)
    if raw_ignore_errors:
        # 遍历原始的错误忽略设置列表
        for error_codes in raw_ignore_errors:
            obj_name = None
            # 如果错误代码中包含空格，则将对象名称和错误代码分开
            if " " in error_codes:
                obj_name, error_codes = error_codes.split(" ")

            # 处理特定对象的错误忽略设置，如 "pandas.Series PR01,SA01"
            if obj_name:
                # 如果对象名称已经存在于字典中，则抛出数值错误
                if obj_name in ignore_errors:
                    raise ValueError(
                        f"Object `{obj_name}` is present in more than one "
                        "--ignore_errors argument. Please use it once and specify "
                        "the errors separated by commas.")
                # 将指定对象的错误代码集合化存储在字典中
                ignore_errors[obj_name] = set(error_codes.split(","))

                # 检查未知的错误代码是否在已知错误集合中
                unknown_errors = ignore_errors[obj_name] - ALL_ERRORS
                if unknown_errors:
                    raise ValueError(
                        f"Object `{obj_name}` is ignoring errors {unknown_errors} "
                        f"which are not known. Known errors are: {ALL_ERRORS}")

            # 处理全局的错误忽略设置，如 "PR02,ES01"
            else:
                # 将全局错误代码集合化存储在字典中
                ignore_errors[None].update(set(error_codes.split(",")))

        # 检查未知的全局错误代码是否在已知错误集合中
        unknown_errors = ignore_errors["*"] - ALL_ERRORS
        if unknown_errors:
            raise ValueError(
                f"Unknown errors {unknown_errors} specified using --ignore_errors "
                "Known errors are: {ALL_ERRORS}")

    # 返回格式化后的忽略错误字典
    return ignore_errors


# 主函数，接受多个参数作为输入，无返回值，用于调度验证功能的执行
def main(
    func_name,
    output_format,
    prefix,
    ignore_deprecated,
    ignore_errors
):
    """
    Main entry point. Call the validation for one or for all docstrings.
    """
    # 如果 func_name 参数为 None，则调用 print_validate_all_results 函数
    if func_name is None:
        # 返回 print_validate_all_results 函数的结果，传入以下参数：
        # - output_format: 输出格式
        # - prefix: 输出前缀
        # - ignore_deprecated: 是否忽略已弃用的内容
        # - ignore_errors: 是否忽略错误
        return print_validate_all_results(
            output_format,
            prefix,
            ignore_deprecated,
            ignore_errors
        )
    # 如果 func_name 参数不为 None，则调用 print_validate_one_results 函数
    else:
        # 返回 print_validate_one_results 函数的结果，传入以下参数：
        # - func_name: 要验证的函数名
        # - ignore_errors: 是否忽略错误
        return print_validate_one_results(func_name, ignore_errors)
if __name__ == "__main__":
    # 定义格式选项
    format_opts = "default", "json", "actions"
    # 帮助文本，描述了要验证的函数或方法（例如 pandas.DataFrame.head）
    # 如果未提供函数名，则验证所有文档字符串并返回 JSON
    func_help = (
        "function or method to validate (e.g. pandas.DataFrame.head) "
        "if not provided, all docstrings are validated and returned "
        "as JSON"
    )
    # 创建命令行参数解析器，描述为验证 pandas 文档字符串
    argparser = argparse.ArgumentParser(description="validate pandas docstrings")
    # 添加位置参数，即要验证的函数或方法的名称
    argparser.add_argument("function", nargs="?", default=None, help=func_help)
    # 添加可选参数 --format，用于指定验证多个文档字符串时的输出格式
    # 默认为 "default"，可选值有 "default", "json", "actions"
    argparser.add_argument(
        "--format",
        default="default",
        choices=format_opts,
        help="format of the output when validating "
        "multiple docstrings (ignored when validating one). "
        "It can be {str(format_opts)[1:-1]}",
    )
    # 添加可选参数 --prefix，用于指定要验证的文档字符串名称的模式
    # 例如 "pandas.Series.str." 将验证以此模式开头的所有方法的文档字符串
    # 如果提供了 --function 参数，则忽略此参数
    argparser.add_argument(
        "--prefix",
        default=None,
        help="pattern for the "
        "docstring names, in order to decide which ones "
        'will be validated. A prefix "pandas.Series.str." '
        "will make the script validate all the docstrings "
        "of methods starting by this pattern. It is "
        "ignored if parameter function is provided",
    )
    # 添加可选参数 --ignore_deprecated，若设置此标志则忽略已弃用对象的文档字符串验证
    argparser.add_argument(
        "--ignore_deprecated",
        default=False,
        action="store_true",
        help="if this flag is set, "
        "deprecated objects are ignored when validating "
        "all docstrings",
    )
    # 添加可选参数 --ignore_errors 或简写 -i，用于指定要忽略的错误代码列表
    # 例如 'PR02,SA01'，并可选择性地指定对象路径以仅忽略单个对象的错误
    # 通过重复此参数可以部分验证多个函数
    argparser.add_argument(
        "--ignore_errors",
        "-i",
        default=None,
        action="append",
        help="comma-separated list of error codes "
        "(e.g. 'PR02,SA01'), with optional object path "
        "to ignore errors for a single object "
        "(e.g. pandas.DataFrame.head PR02,SA01). "
        "Partial validation for more than one function "
        "can be achieved by repeating this parameter.",
    )
    # 解析命令行参数
    args = argparser.parse_args(sys.argv[1:])

    # 调用 main 函数，并传递命令行参数
    sys.exit(
        main(args.function,
             args.format,
             args.prefix,
             args.ignore_deprecated,
             _format_ignore_errors(args.ignore_errors),
             )
    )
```