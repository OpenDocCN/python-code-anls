# `.\pytorch\torch\_sources.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类
import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple

# 导入 Torch 的错误处理模块
from torch._C import ErrorReport
# 导入 Torch 的 JIT 编译相关模块
from torch._C._jit_tree_views import SourceRangeFactory


def get_source_lines_and_file(
    obj: Any,
    error_msg: Optional[str] = None,
) -> Tuple[List[str], int, Optional[str]]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        # 获取对象的源文件路径
        filename = inspect.getsourcefile(obj)
        # 获取对象的源代码行和起始行号
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        # 处理获取源文件失败的异常情况
        msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
        if error_msg:
            msg += "\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


def normalize_source_lines(sourcelines: List[str]) -> List[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '\n' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]

    # 查找包含函数定义的行和行号
    idx = None
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith("def"):
            idx = i
            break

    # 处理无法找到函数定义的情况（例如 lambda 函数）
    if idx is None:
        return sourcelines

    # 获取函数定义行的缩进
    fn_def = sourcelines[idx]
    whitespace = fn_def.split("def")[0]

    # 对函数定义前后的代码行进行缩进调整
    aligned_prefix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]
    ]
    aligned_suffix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1 :]
    ]

    # 返回调整后的源代码行列表
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix


# SourceContext 类，继承自 SourceRangeFactory，用于存储将要编译的函数的额外元数据
# 该类为对 SourceRangeFactory 的轻量封装
class SourceContext(SourceRangeFactory):
    # 初始化方法，用于创建一个新的对象实例
    def __init__(
        self,
        source,
        filename,
        file_lineno,
        leading_whitespace_len,
        uses_true_division=True,
        funcname=None,
    ):
        # 调用父类的初始化方法，传递相应的参数
        super().__init__(source, filename, file_lineno, leading_whitespace_len)
        # 设置当前对象的属性 uses_true_division，用于指示是否使用真除法
        self.uses_true_division = uses_true_division
        # 设置当前对象的属性 filename，表示文件名
        self.filename = filename
        # 设置当前对象的属性 funcname，表示函数名，可以为 None
        self.funcname = funcname
# 使用 functools 模块的 lru_cache 装饰器，将 make_source_context 函数设置为带有无限大小缓存的函数
@functools.lru_cache(maxsize=None)
# 定义 make_source_context 函数，接受任意数量的参数，并返回一个 SourceContext 对象
def make_source_context(*args):
    return SourceContext(*args)


# 定义 fake_range 函数，返回一个空字符串的 SourceContext 对象，并调用其 make_raw_range 方法生成一个伪造的范围
def fake_range():
    return SourceContext("", None, 0, 0).make_raw_range(0, 1)


# 使用 NamedTuple 创建 ParsedDef 类型，包含 ast、ctx、source、filename 和 file_lineno 五个字段
class ParsedDef(NamedTuple):
    ast: ast.Module  # AST 抽象语法树表示的模块对象
    ctx: SourceContext  # 与代码源相关的上下文对象
    source: str  # 源代码的字符串表示
    filename: Optional[str]  # 可选的源文件名
    file_lineno: int  # 源文件中的行号


# 定义 parse_def 函数，接受一个函数 fn 作为参数
def parse_def(fn):
    # 获取函数 fn 的源代码行、调用栈中的错误报告和源文件信息
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        fn, ErrorReport.call_stack()
    )
    # 标准化源代码行，确保格式一致性
    sourcelines = normalize_source_lines(sourcelines)
    # 将标准化后的源代码行连接成一个字符串
    source = "".join(sourcelines)
    # 对源代码进行去除缩进处理
    dedent_src = dedent(source)
    # 解析去除缩进后的 Python 抽象语法树
    py_ast = ast.parse(dedent_src)
    # 如果 AST 的主体不止一个或者不是 ast.FunctionDef 类型，则抛出运行时错误
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError(
            f"Expected a single top-level function: {filename}:{file_lineno}"
        )
    # 计算源代码开头的空白字符长度
    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    # 创建源码上下文对象 ctx，调用 make_source_context 函数
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, True, fn.__name__
    )
    # 返回 ParsedDef 对象，包含 AST、上下文对象 ctx、源代码字符串、源文件名和文件行号
    return ParsedDef(py_ast, ctx, source, filename, file_lineno)
```