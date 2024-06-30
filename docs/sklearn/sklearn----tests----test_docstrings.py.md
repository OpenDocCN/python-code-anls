# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_docstrings.py`

```
# 导入正则表达式模块
import re
# 导入检查函数签名的模块
from inspect import signature
# 导入类型提示中的 Optional 类型
from typing import Optional

# 导入 pytest 模块
import pytest

# 使得在调用 `all_estimators` 时能够发现实验性评估器
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
# 导入用于发现显示功能的辅助函数
from sklearn.utils.discovery import all_displays, all_estimators, all_functions

# 导入 numpydoc 的验证模块，如果不存在则跳过（importorskip）
numpydoc_validation = pytest.importorskip("numpydoc.validate")


def get_all_methods():
    # 获取所有的评估器
    estimators = all_estimators()
    # 获取所有的显示功能
    displays = all_displays()
    # 遍历所有评估器和显示功能
    for name, Klass in estimators + displays:
        # 如果类名以 "_" 开头，则跳过，不处理私有类
        if name.startswith("_"):
            continue
        # 初始化空方法列表
        methods = []
        # 遍历类中的所有属性名
        for name in dir(Klass):
            # 如果属性名以 "_" 开头，则跳过
            if name.startswith("_"):
                continue
            # 获取属性对应的对象
            method_obj = getattr(Klass, name)
            # 如果对象是可调用的或者是属性，则加入到方法列表中
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                methods.append(name)
        # 添加一个空值作为结束标记
        methods.append(None)

        # 按照字符串排序方法列表，并逐个生成
        for method in sorted(methods, key=str):
            yield Klass, method


def get_all_functions_names():
    # 获取所有函数列表
    functions = all_functions()
    # 遍历函数列表
    for _, func in functions:
        # 如果函数不是来自 utils.fixes 模块，则生成其模块名与函数名的组合字符串
        if "utils.fixes" not in func.__module__:
            yield f"{func.__module__}.{func.__name__}"


def filter_errors(errors, method, Klass=None):
    """
    根据方法类型忽略一些错误。

    这些规则特定适用于 scikit-learn。
    """
    for code, message in errors:
        # 遍历错误列表，每个错误包含错误代码和错误信息

        # 忽略以下错误代码：
        # - RT02: 返回部分的第一行应只包含类型，..
        #   (因为我们可能需要引用返回对象的名称)
        # - GL01: 文档字符串文本（摘要）应从开头引号的下一行开始
        #   （不应在同一行，或者在中间留有空行）
        # - GL02: 如果有空行，应在返回部分的第一行之前，而不是之后
        #   （允许为属性使用简短的文档字符串）。
        if code in ["RT02", "GL01", "GL02"]:
            continue

        # 忽略以下错误代码：
        # - PR02: 对于属性的未知参数。有时我们使用属性进行鸭子类型检查，例如 SGDClassifier.predict_proba
        # - GL08: 方法签名解析失败，可能是因为这是一个属性。属性有时用于已弃用的属性，并且该属性已在类文档字符串中记录。
        #
        # 所有错误代码参考：
        # https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
        if code in ("PR02", "GL08") and Klass is not None and method is not None:
            # 如果 Klass 和 method 非空，则获取方法对象
            method_obj = getattr(Klass, method)
            # 如果方法对象是属性，则继续下一个循环
            if isinstance(method_obj, property):
                continue

        # 仅对顶层类的文档字符串考虑以下错误代码：
        # - ES01: 没有找到扩展摘要
        # - SA01: 没有找到参见部分
        # - EX01: 没有找到示例部分
        if method is not None and code in ["EX01", "SA01", "ES01"]:
            continue

        # 产生当前错误代码和信息的生成器对象
        yield code, message
def repr_errors(res, Klass=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    Klass : {Estimator, Display, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    # 如果 method 参数为 None，则根据 Klass 对象的情况确定 method 的值
    if method is None:
        # 如果 Klass 对象有 "__init__" 属性，则将 method 设为 "__init__"
        if hasattr(Klass, "__init__"):
            method = "__init__"
        # 如果 Klass 为 None，则抛出 ValueError 异常
        elif Klass is None:
            raise ValueError("At least one of Klass, method should be provided")
        # 否则，抛出 NotImplementedError 异常（尚未实现）
        else:
            raise NotImplementedError

    # 如果 Klass 不为 None，则获取 Klass 对象的 method 方法
    if Klass is not None:
        obj = getattr(Klass, method)
        try:
            # 获取方法的签名字符串表示
            obj_signature = str(signature(obj))
        except TypeError:
            # 特别地，无法解析属性方法的签名
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        # 构建对象名称，格式为 类名.method_name
        obj_name = Klass.__name__ + "." + method
    else:
        # 如果 Klass 为 None，则对象签名为空字符串
        obj_signature = ""
        obj_name = method

    # 构建输出消息，包括文件名、对象名称及其签名、原始文档字符串、错误信息
    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),  # 输出文件名
            obj_name + obj_signature,  # 输出对象名称及其签名
            res["docstring"],  # 输出原始文档字符串
            "# Errors",  # 错误标题
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),  # 输出所有错误信息
        ]
    )
    # 返回最终构建的消息字符串
    return msg


@pytest.mark.parametrize("function_name", get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""
    # 使用 numpydoc_validation 模块验证函数文档字符串
    res = numpydoc_validation.validate(function_name)

    # 过滤仅保留函数相关的错误信息
    res["errors"] = list(filter_errors(res["errors"], method="function"))

    # 如果存在错误信息，则生成错误消息并抛出 ValueError 异常
    if res["errors"]:
        msg = repr_errors(res, method=f"Tested function: {function_name}")
        raise ValueError(msg)


@pytest.mark.parametrize("Klass, method", get_all_methods())
def test_docstring(Klass, method, request):
    # 构建 Klass 的完整导入路径
    base_import_path = Klass.__module__
    import_path = [base_import_path, Klass.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    # 使用 numpydoc_validation 模块验证导入路径对应的文档字符串
    res = numpydoc_validation.validate(import_path)

    # 过滤仅保留特定方法和 Klass 的错误信息
    res["errors"] = list(filter_errors(res["errors"], method, Klass=Klass))

    # 如果存在错误信息，则生成错误消息并抛出 ValueError 异常
    if res["errors"]:
        msg = repr_errors(res, Klass, method)
        raise ValueError(msg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate docstring with numpydoc.")
    parser.add_argument("import_path", help="Import path to validate")

    args = parser.parse_args()

    # 使用 numpydoc_validation 模块验证传入的导入路径
    res = numpydoc_validation.validate(args.import_path)

    import_path_sections = args.import_path.split(".")
    # 当应用于类时，检测类方法。对于函数，method 为 None。
    # TODO: 检测可以改进。当前假设我们有类方法，如果导入路径的倒数第二个元素是驼峰命名。
    if len(import_path_sections) >= 2 and re.match(
        r"(?:[A-Z][a-z]*)+", import_path_sections[-2]
    ):
        # 如果满足条件，将最后一个路径元素作为方法名
        method = import_path_sections[-1]
    else:
        # 否则将方法名设为 None
        method = None

    # 使用过滤函数从结果错误列表中过滤特定方法相关的错误
    res["errors"] = list(filter_errors(res["errors"], method))

    # 如果存在错误
    if res["errors"]:
        # 生成错误消息，包含特定方法的错误信息
        msg = repr_errors(res, method=args.import_path)

        # 输出错误消息并退出程序
        print(msg)
        sys.exit(1)
    else:
        # 如果没有错误，输出检查通过的消息
        print("All docstring checks passed for {}!".format(args.import_path))
```