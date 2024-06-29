# `D:\src\scipysrc\pandas\pandas\tests\util\test_util.py`

```
# 导入标准库中的 os 模块
import os

# 导入第三方库 pytest
import pytest

# 从 pandas 库中导入 array 和 compat 模块
from pandas import (
    array,
    compat,
)

# 导入 pandas 内部的测试工具模块
import pandas._testing as tm


# 定义测试函数，验证 numpy 的错误状态是否为默认设置
def test_numpy_err_state_is_default():
    # 预期的 numpy 错误状态字典
    expected = {"over": "warn", "divide": "warn", "invalid": "warn", "under": "ignore"}
    # 导入 numpy 库
    import numpy as np

    # 断言当前 numpy 的错误状态与预期一致
    assert np.geterr() == expected


# 定义测试函数，验证将行列表转换为 CSV 字符串的功能
def test_convert_rows_list_to_csv_str():
    # 给定的行列表
    rows_list = ["aaa", "bbb", "ccc"]
    # 调用测试工具模块中的函数进行转换
    ret = tm.convert_rows_list_to_csv_str(rows_list)

    # 根据操作系统平台判断预期的转换结果
    if compat.is_platform_windows():
        expected = "aaa\r\nbbb\r\nccc\r\n"
    else:
        expected = "aaa\nbbb\nccc\n"

    # 断言转换后的结果与预期相符
    assert ret == expected


# 使用 pytest 的参数化装饰器，定义测试函数，验证 datapath 函数对文件缺失的处理
@pytest.mark.parametrize("strict_data_files", [True, False])
def test_datapath_missing(datapath):
    # 使用 pytest 断言捕获 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match="Could not find file"):
        datapath("not_a_file")


# 定义测试函数，验证 datapath 函数返回正确的文件路径
def test_datapath(datapath):
    # 给定的路径参数
    args = ("io", "data", "csv", "iris.csv")

    # 调用 datapath 函数获取结果
    result = datapath(*args)
    # 构建预期的文件路径
    expected = os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

    # 断言结果与预期路径相等
    assert result == expected


# 定义测试函数，验证 external_error_raised 上下文管理器能正确捕获特定异常类型
def test_external_error_raised():
    # 使用 external_error_raised 上下文管理器，验证是否能捕获指定类型的异常
    with tm.external_error_raised(TypeError):
        # 抛出一个 TypeError 异常，但不会检查其错误消息内容
        raise TypeError("Should not check this error message, so it will pass")


# 定义测试函数，验证 assert_is_sorted 函数能正确判断数组是否排序
def test_is_sorted():
    # 创建一个已排序的数组，并调用 assert_is_sorted 进行验证
    arr = array([1, 2, 3], dtype="Int64")
    tm.assert_is_sorted(arr)

    # 创建一个未排序的数组，并使用 pytest 断言捕获 Assertion 错误，并匹配特定错误消息
    arr = array([4, 2, 3], dtype="Int64")
    with pytest.raises(AssertionError, match="ExtensionArray are different"):
        tm.assert_is_sorted(arr)
```