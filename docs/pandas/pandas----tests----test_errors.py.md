# `D:\src\scipysrc\pandas\pandas\tests\test_errors.py`

```
# 引入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 pandas.errors 模块中导入以下异常类
from pandas.errors import (
    AbstractMethodError,
    UndefinedVariableError,
)

# 导入 pandas 库，并使用别名 pd
import pandas as pd

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，参数是异常类名的列表
@pytest.mark.parametrize(
    "exc",
    [
        "AttributeConflictWarning",
        "CSSWarning",
        "CategoricalConversionWarning",
        "ClosedFileError",
        "DataError",
        "DatabaseError",
        "DtypeWarning",
        "EmptyDataError",
        "IncompatibilityWarning",
        "IndexingError",
        "InvalidColumnName",
        "InvalidComparison",
        "InvalidVersion",
        "LossySetitemError",
        "MergeError",
        "NoBufferPresent",
        "NumExprClobberingError",
        "NumbaUtilError",
        "OptionError",
        "OutOfBoundsDatetime",
        "ParserError",
        "ParserWarning",
        "PerformanceWarning",
        "PossibleDataLossError",
        "PossiblePrecisionLoss",
        "PyperclipException",
        "SpecificationError",
        "UnsortedIndexError",
        "UnsupportedFunctionCall",
        "ValueLabelTypeMismatch",
    ],
)
# 定义一个名为 test_exception_importable 的测试函数，参数是 exc
def test_exception_importable(exc):
    # 从 pandas.errors 模块中动态获取异常类
    from pandas import errors

    # 获取异常类的实际对象
    err = getattr(errors, exc)
    # 断言异常类对象不为空
    assert err is not None

    # 使用 pytest.raises 检查是否可以成功抛出异常
    msg = "^$"
    with pytest.raises(err, match=msg):
        # 抛出异常实例
        raise err()

# 定义一个测试函数 test_catch_oob，测试 pandas.errors 中的 OutOfBoundsDatetime 异常
def test_catch_oob():
    # 从 pandas.errors 模块中导入 OutOfBoundsDatetime 异常类
    from pandas import errors

    # 定义匹配的错误信息
    msg = "Cannot cast 1500-01-01 00:00:00 to unit='ns' without overflow"
    # 使用 pytest.raises 检查是否可以捕获 OutOfBoundsDatetime 异常，并匹配指定的错误信息
    with pytest.raises(errors.OutOfBoundsDatetime, match=msg):
        # 尝试创建一个超出范围的时间戳
        pd.Timestamp("15000101").as_unit("ns")

# 参数化测试函数 test_catch_undefined_variable_error，测试 UndefinedVariableError 异常
@pytest.mark.parametrize("is_local", [True, False])
def test_catch_undefined_variable_error(is_local):
    # 定义变量名
    variable_name = "x"
    # 根据 is_local 参数选择错误信息
    if is_local:
        msg = f"local variable '{variable_name}' is not defined"
    else:
        msg = f"name '{variable_name}' is not defined"

    # 使用 pytest.raises 检查是否可以捕获 UndefinedVariableError 异常，并匹配指定的错误信息
    with pytest.raises(UndefinedVariableError, match=msg):
        # 抛出 UndefinedVariableError 异常实例
        raise UndefinedVariableError(variable_name, is_local)

# 定义一个类 Foo，用于测试抽象方法相关的异常
class Foo:
    # 类方法抛出 AbstractMethodError 异常
    @classmethod
    def classmethod(cls):
        raise AbstractMethodError(cls, methodtype="classmethod")

    # 属性方法抛出 AbstractMethodError 异常
    @property
    def property(self):
        raise AbstractMethodError(self, methodtype="property")

    # 实例方法抛出 AbstractMethodError 异常
    def method(self):
        raise AbstractMethodError(self)

# 测试类 Foo 中抛出 AbstractMethodError 异常的情况
def test_AbstractMethodError_classmethod():
    # 测试类方法抛出异常
    xpr = "This classmethod must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo.classmethod()

    # 测试属性方法抛出异常
    xpr = "This property must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().property

    # 测试实例方法抛出异常
    xpr = "This method must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().method()
```