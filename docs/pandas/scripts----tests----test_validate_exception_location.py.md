# `D:\src\scipysrc\pandas\scripts\tests\test_validate_exception_location.py`

```
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从指定路径导入 validate_exception_and_warning_placement 函数及相关常量
from scripts.validate_exception_location import (
    ERROR_MESSAGE,
    validate_exception_and_warning_placement,
)

# 定义路径常量 PATH，指定待测试的文件路径
PATH = "t.py"

# 自定义常量说明：
# CUSTOM_EXCEPTION_NOT_IN_TESTING_RST 表示不在 testing.rst 中的自定义异常
# CUSTOM_EXCEPTION__IN_TESTING_RST 表示在 testing.rst 中的旧版自定义异常
# ERRORS_IN_TESTING_RST 是从 testing.rst 解析出的异常集合，包括 CUSTOM_EXCEPTION__IN_TESTING_RST
CUSTOM_EXCEPTION_NOT_IN_TESTING_RST = "MyException"
CUSTOM_EXCEPTION__IN_TESTING_RST = "MyOldException"
ERRORS_IN_TESTING_RST = {CUSTOM_EXCEPTION__IN_TESTING_RST}

# 定义测试用例的代码模板，包含一个简单的函数和一个自定义异常类
TEST_CODE = """
import numpy as np
import sys

def my_func():
  pass

class {custom_name}({error_type}):
  pass

"""

# 使用 pytest 的 fixture 装饰器定义一个参数化的测试用例，用来测试各种 Python 内置异常
@pytest.fixture(params=["Exception", "ValueError", "Warning", "UserWarning"])
def error_type(request):
    return request.param

# 测试用例：检查未在 testing.rst 中声明但继承异常的类是否被正确标记
def test_class_that_inherits_an_exception_and_is_not_in_the_testing_rst_is_flagged(
    capsys, error_type
) -> None:
    # 根据模板生成具体内容，替换自定义异常名称和异常类型
    content = TEST_CODE.format(
        custom_name=CUSTOM_EXCEPTION_NOT_IN_TESTING_RST, error_type=error_type
    )
    # 期待的错误消息格式，包含未在 testing.rst 中声明的自定义异常名
    expected_msg = ERROR_MESSAGE.format(errors=CUSTOM_EXCEPTION_NOT_IN_TESTING_RST)
    # 断言捕获到 SystemExit 异常，并匹配期待的错误消息
    with pytest.raises(SystemExit, match=None):
        validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)
    # 读取并断言 capsys 捕获的输出与期待的错误消息一致
    result_msg, _ = capsys.readouterr()
    assert result_msg == expected_msg

# 测试用例：检查在 testing.rst 中声明且继承异常的类是否未被错误标记
def test_class_that_inherits_an_exception_but_is_in_the_testing_rst_is_not_flagged(
    capsys, error_type
) -> None:
    # 根据模板生成具体内容，替换自定义异常名称和异常类型
    content = TEST_CODE.format(
        custom_name=CUSTOM_EXCEPTION__IN_TESTING_RST, error_type=error_type
    )
    # 调用验证函数，检查是否不会捕获 SystemExit 异常，表示不会错误标记
    validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)

# 测试用例：检查不继承任何异常的类是否未被错误标记
def test_class_that_does_not_inherit_an_exception_is_not_flagged(capsys) -> None:
    # 定义一个简单的类定义字符串
    content = "class MyClass(NonExceptionClass): pass"
    # 调用验证函数，检查是否不会捕获 SystemExit 异常，表示不会错误标记
    validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)
```