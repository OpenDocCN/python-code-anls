# `D:\src\scipysrc\pandas\scripts\validate_unwanted_patterns.py`

```
"""
Unwanted patterns test cases.

The reason this file exist despite the fact we already have
`ci/code_checks.sh`,
(see https://github.com/pandas-dev/pandas/blob/master/ci/code_checks.sh)

is that some of the test cases are more complex/impossible to validate via regex.
So this file is somewhat an extensions to `ci/code_checks.sh`
"""

import argparse  # 导入用于解析命令行参数的模块
import ast  # 导入用于抽象语法树操作的模块
from collections.abc import (  # 导入抽象基类中的具体集合类
    Callable,
    Iterable,
)
import sys  # 导入与系统交互的模块
import token  # 导入 Python 词法分析的标记化常量
import tokenize  # 导入用于词法分析的模块
from typing import IO  # 从 typing 模块中导入 IO 泛型

PRIVATE_IMPORTS_TO_IGNORE: set[str] = {
    "_extension_array_shared_docs",  # 忽略的私有导入名称集合
    "_index_shared_docs",
    "_interval_shared_docs",
    "_merge_doc",
    "_shared_docs",
    "_new_Index",
    "_new_PeriodIndex",
    "_agg_template_series",
    "_agg_template_frame",
    "_pipe_template",
    "_apply_groupings_depr",
    "__main__",
    "_transform_template",
    "_get_plot_backend",
    "_matplotlib",
    "_arrow_utils",
    "_registry",
    "_test_parse_iso8601",
    "_testing",
    "_test_decorators",
    "__version__",  # 在 compat.numpy.function 中检查 np.__version__
    "__git_version__",
    "_arrow_dtype_mapping",
    "_global_config",
    "_chained_assignment_msg",
    "_chained_assignment_method_msg",
    "_version_meson",
    # numba 扩展需要此项来模拟 iloc 对象
    "_iLocIndexer",
    # TODO(4.0): GH#55043 - 删除 CoW 选项后移除
    "_get_option",
    "_fill_limit_area_1d",
    "_make_block",
}


def _get_literal_string_prefix_len(token_string: str) -> int:
    """
    获取字面字符串前缀的长度。

    Parameters
    ----------
    token_string : str
        要检查的字符串。

    Returns
    -------
    int
        字面字符串前缀的长度。

    Examples
    --------
    >>> example_string = "'Hello world'"
    >>> _get_literal_string_prefix_len(example_string)
    0
    >>> example_string = "r'Hello world'"
    >>> _get_literal_string_prefix_len(example_string)
    1
    """
    try:
        return min(
            token_string.find(quote)
            for quote in (r"'", r'"')
            if token_string.find(quote) >= 0
        )
    except ValueError:
        return 0


PRIVATE_FUNCTIONS_ALLOWED = {"sys._getframe"}  # 允许的私有函数名称集合，没有已知的替代方法


def private_function_across_module(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    """
    检查私有函数是否跨模块使用。

    Parameters
    ----------
    file_obj : IO
        包含要验证的 Python 代码的类文件对象。

    Yields
    ------
    line_number : int
        跨模块使用的私有函数的行号。
    msg : str
        错误的解释信息。
    """
    contents = file_obj.read()  # 读取文件内容到字符串中
    tree = ast.parse(contents)  # 解析文件内容生成抽象语法树

    imported_modules: set[str] = set()  # 存储导入的模块名称的集合
    # 遍历抽象语法树中的每个节点
    for node in ast.walk(tree):
        # 如果节点是导入模块或者从模块导入
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # 遍历节点中的每个模块
            for module in node.names:
                # 获取模块的完全限定名称，如果有重命名则使用重命名的名称
                module_fqdn = module.name if module.asname is None else module.asname
                # 将完全限定名称添加到已导入模块集合中
                imported_modules.add(module_fqdn)

        # 如果节点不是函数调用，则跳过
        if not isinstance(node, ast.Call):
            continue

        try:
            # 尝试获取函数调用的模块名称和函数名称
            module_name = node.func.value.id
            function_name = node.func.attr
        except AttributeError:
            # 如果获取失败，则跳过此节点
            continue

        # 异常情况处理 #

        # (值得商榷的) 类名情况，跳过大写字母开头的模块名
        if module_name[0].isupper():
            continue
        # (值得商榷的) 魔术方法情况，跳过以双下划线开头和结尾的函数名
        elif function_name.startswith("__") and function_name.endswith("__"):
            continue
        # 如果模块名和函数名的组合在允许的私有函数集合中，则跳过
        elif module_name + "." + function_name in PRIVATE_FUNCTIONS_ALLOWED:
            continue

        # 如果模块名在已导入模块集合中且函数名以下划线开头，则生成警告
        if module_name in imported_modules and function_name.startswith("_"):
            yield (node.lineno, f"Private function '{module_name}.{function_name}'")
# 定义一个函数，用于检查是否在跨模块导入中引用了私有函数
def private_import_across_module(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    # 读取文件对象中的内容
    contents = file_obj.read()
    # 解析文件内容形成语法树
    tree = ast.parse(contents)

    # 遍历语法树中的每个节点
    for node in ast.walk(tree):
        # 如果节点不是导入语句（包括普通导入和从某个模块导入）
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        # 遍历导入的每个模块
        for module in node.names:
            # 获取模块名，去掉可能存在的包路径部分，保留最后的模块名
            module_name = module.name.split(".")[-1]
            # 如果模块名在忽略的私有导入列表中，则继续下一个模块的检查
            if module_name in PRIVATE_IMPORTS_TO_IGNORE:
                continue

            # 如果模块名以下划线开头，表示是私有函数的导入
            if module_name.startswith("_"):
                # 返回导入语句所在的行号和错误信息字符串
                yield (node.lineno, f"Import of internal function {module_name!r}")


# 定义一个函数，用于检查字符串中不正确放置的空白字符
def strings_with_wrong_placed_whitespace(
    file_obj: IO[str],
) -> Iterable[tuple[int, str]]:
    """
    Test case for leading spaces in concatenated strings.

    For example:

    >>> rule = (
    ...    "We want the space at the end of the line, "
    ...    "not at the beginning"
    ... )

    Instead of:

    >>> rule = (
    ...    "We want the space at the end of the line,"
    ...    " not at the beginning"
    ... )

    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.

    Yields
    ------
    line_number : int
        Line number of unconcatenated string.
    msg : str
        Explanation of the error.
    """
    def has_wrong_whitespace(first_line: str, second_line: str) -> bool:
        """
        检查两行文本是否存在不良空白模式。

        Parameters
        ----------
        first_line : str
            待检查的第一行文本。
        second_line : str
            待检查的第二行文本。

        Returns
        -------
        bool
            如果两行文本匹配不良模式，则返回True。

        Notes
        -----
        我们要捕捉的不良模式是，如果一个跨多行连接的字符串中，空格位于每行字符串的末尾，
        除非该字符串以换行符（\n）结尾。

        例如，这是不好的示例：

        >>> rule = (
        ...    "We want the space at the end of the line,"
        ...    " not at the beginning"
        ... )

        而我们期望的是：

        >>> rule = (
        ...    "We want the space at the end of the line, "
        ...    "not at the beginning"
        ... )

        如果字符串以换行符（\n）结尾，我们不希望在其后有任何尾随空格。

        例如，这是不好的示例：

        >>> rule = (
        ...    "We want the space at the begging of "
        ...    "the line if the previous line is ending with a \n "
        ...    "not at the end, like always"
        ... )

        而我们希望的是：

        >>> rule = (
        ...    "We want the space at the begging of "
        ...    "the line if the previous line is ending with a \n"
        ...    " not at the end, like always"
        ... )
        """
        # 检查第一行是否以 \n 结尾，如果是则不符合不良模式
        if first_line.endswith(r"\n"):
            return False
        # 如果第一行或第二行以两个空格开头，则不符合不良模式
        elif first_line.startswith("  ") or second_line.startswith("  "):
            return False
        # 如果第一行或第二行以一个空格结尾，则不符合不良模式
        elif first_line.endswith("  ") or second_line.endswith("  "):
            return False
        # 如果第一行不以空格结尾，但第二行以空格开头，则符合不良模式
        elif (not first_line.endswith(" ")) and second_line.startswith(" "):
            return True
        # 默认情况下，返回 False，表示不符合不良模式
        return False

    tokens: list = list(tokenize.generate_tokens(file_obj.readline))
    # 遍历 tokens 列表中每三个相邻的元素
    for first_token, second_token, third_token in zip(tokens, tokens[1:], tokens[2:]):
        # 检查是否处于连续字符串块中
        if (
            first_token.type == third_token.type == token.STRING
            and second_token.type == token.NL
        ):
            # 去除字符串的引号及前缀
            first_string: str = first_token.string[
                _get_literal_string_prefix_len(first_token.string) + 1 : -1
            ]
            second_string: str = third_token.string[
                _get_literal_string_prefix_len(third_token.string) + 1 : -1
            ]

            # 检查第一个字符串和第二个字符串之间是否有错误的空白符
            if has_wrong_whitespace(first_string, second_string):
                # 生成一个元组，包含错误位置和错误信息
                yield (
                    third_token.start[0],
                    (
                        "String has a space at the beginning instead "
                        "of the end of the previous string."
                    ),
                )
def main(
    function: Callable[[IO[str]], Iterable[tuple[int, str]]],
    source_path: str,
    output_format: str,
) -> bool:
    """
    Main entry point of the script.

    Parameters
    ----------
    function : Callable
        Function to execute for the specified validation type.
    source_path : str
        Source path representing path to a file/directory.
    output_format : str
        Output format of the error message.
    """
    # 初始化失败标志
    is_failed: bool = False

    # 遍历每个文件路径
    for file_path in source_path:
        # 打开文件对象
        with open(file_path, encoding="utf-8") as file_obj:
            # 对文件中的每一行进行函数调用并迭代处理结果
            for line_number, msg in function(file_obj):
                # 设置失败标志为 True
                is_failed = True
                # 打印格式化的错误消息
                print(
                    output_format.format(
                        source_path=file_path, line_number=line_number, msg=msg
                    )
                )
    # 返回变量 is_failed 的值作为函数的返回结果
    return is_failed
if __name__ == "__main__":
    # 定义可用的验证类型列表
    available_validation_types: list[str] = [
        "private_function_across_module",
        "private_import_across_module",
        "strings_with_wrong_placed_whitespace",
        "nodefault_used_not_only_for_typing",
    ]

    # 创建命令行解析器对象，并设置描述信息
    parser = argparse.ArgumentParser(description="Unwanted patterns checker.")

    # 添加位置参数 paths，表示要检查的文件的源路径
    parser.add_argument("paths", nargs="*", help="Source paths of files to check.")

    # 添加可选参数 --format 或 -f，用于指定错误消息的输出格式
    parser.add_argument(
        "--format",
        "-f",
        default="{source_path}:{line_number}: {msg}",
        help="Output format of the error message.",
    )

    # 添加必选参数 --validation-type 或 -vt，用于指定要执行的验证测试类型
    parser.add_argument(
        "--validation-type",
        "-vt",
        choices=available_validation_types,
        required=True,
        help="Validation test case to check.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用全局函数 globals() 获取命令行参数指定的验证函数，并执行主函数 main
    sys.exit(
        main(
            function=globals().get(args.validation_type),  # 获取指定的验证函数
            source_path=args.paths,  # 获取源文件路径列表
            output_format=args.format,  # 获取输出格式
        )
    )
```