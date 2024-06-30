# `D:\src\scipysrc\scikit-learn\doc\sphinxext\github_link.py`

```
import inspect  # 导入 inspect 模块，用于检查和获取对象的信息
import os  # 导入 os 模块，用于与操作系统交互
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import sys  # 导入 sys 模块，用于访问系统相关的参数和功能
from functools import partial  # 导入 functools 模块中的 partial 函数，用于部分应用函数
from operator import attrgetter  # 导入 operator 模块中的 attrgetter 函数，用于获取对象的属性

REVISION_CMD = "git rev-parse --short HEAD"  # 定义获取 Git 提交短哈希的命令字符串


def _get_git_revision():
    """获取当前代码库的 Git 提交短哈希

    使用 subprocess 执行 Git 命令，获取输出的字节流，并转换为 UTF-8 编码的字符串
    """
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()  # 执行 Git 命令获取短哈希
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")  # 如果执行失败，则打印错误信息
        return None  # 返回 None
    return revision.decode("utf-8")  # 返回 UTF-8 编码的短哈希字符串


def _linkcode_resolve(domain, info, package, url_fmt, revision):
    """确定类/方法/函数在线源代码的链接地址

    这个函数通常被 sphinx.ext.linkcode 调用

    参数示例：一个长期未更改的模块的示例
    """
    if revision is None:  # 如果未提供有效的 revision 参数，则返回
        return
    if domain not in ("py", "pyx"):  # 如果域不在有效的范围内，则返回
        return
    if not info.get("module") or not info.get("fullname"):  # 如果没有有效的模块名或完整名称，则返回
        return

    class_name = info["fullname"].split(".")[0]  # 提取类名（全名的第一部分）
    module = __import__(info["module"], fromlist=[class_name])  # 动态导入模块
    obj = attrgetter(info["fullname"])(module)  # 使用 attrgetter 获取指定属性的对象

    # 解包对象以获取正确的源文件（如果由装饰器包装）
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)  # 获取对象关联的源文件名
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])  # 获取对象模块的源文件名
        except Exception:
            fn = None
    if not fn:  # 如果找不到源文件，则返回
        return

    # 计算相对路径并尝试获取行号
    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))  # 计算相对路径
    try:
        lineno = inspect.getsourcelines(obj)[1]  # 获取对象定义的起始行号
    except Exception:
        lineno = ""  # 获取不到行号时为空字符串
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)  # 格式化 URL 返回


def make_linkcode_resolve(package, url_fmt):
    """生成一个 linkcode_resolve 函数，用于给定的 URL 格式

    revision 是 Git 提交引用（哈希或名称）

    package 是包的根模块的名称

    url_fmt 通常为 'https://github.com/USER/PROJECT/blob/{revision}/{package}/{path}#L{lineno}'
    """
    revision = _get_git_revision()  # 获取当前代码库的 Git 提交短哈希
    return partial(
        _linkcode_resolve, revision=revision, package=package, url_fmt=url_fmt
    )  # 返回部分应用了 _linkcode_resolve 函数的函数对象
```