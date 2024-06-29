# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_numpy_array_equal.py`

```
# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象

import numpy as np  # 导入NumPy库，并用np作为别名
import pytest  # 导入pytest库，用于编写和运行测试

import pandas as pd  # 导入Pandas库，并用pd作为别名
from pandas import Timestamp  # 从Pandas中导入Timestamp类
import pandas._testing as tm  # 导入Pandas内部测试模块

# 定义测试函数，测试当NumPy数组不等时的断言行为
def test_assert_numpy_array_equal_shape_mismatch():
    msg = """numpy array are different

numpy array shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]))

# 定义测试函数，测试当输入的不是NumPy数组时的断言行为
def test_assert_numpy_array_equal_bad_type():
    expected = "Expected type"
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息expected
    with pytest.raises(AssertionError, match=expected):
        tm.assert_numpy_array_equal(1, 2)

# 使用@pytest.mark.parametrize装饰器指定多组参数进行参数化测试
@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
# 定义测试函数，测试当NumPy数组类别不匹配时的断言行为
def test_assert_numpy_array_equal_class_mismatch(a, b, klass1, klass2):
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况1）
def test_assert_numpy_array_equal_value_mismatch1():
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况2）
def test_assert_numpy_array_equal_value_mismatch2():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([1, 3]))

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况3）
def test_assert_numpy_array_equal_value_mismatch3():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况4）
def test_assert_numpy_array_equal_value_mismatch4():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1\\.1, 2\\.000001\\]
\\[right\\]: \\[1\\.1, 2.0\\]"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1.1, 2.000001]), np.array([1.1, 2.0]))

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况5）
def test_assert_numpy_array_equal_value_mismatch5():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""
    
    # 使用pytest.raises检测是否抛出AssertionError异常，并匹配预期的错误信息msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )

# 定义测试函数，测试当NumPy数组元素值不匹配时的断言行为（情况6）
def test_assert_numpy_array_equal_value_mismatch6():
    # 待实现的测试函数，暂时留空
    pass
    # 定义一个多行字符串变量，用于存储错误消息或文本
    msg = """numpy array are different
# 测试确保当 numpy 数组包含不同的 Unicode 对象时，`tm.assert_numpy_array_equal` 正确引发异常。
msg = """numpy array are different

numpy array values are different \(33\.33333 %\)
\[left\]:  \[á, à, ä\]
\[right\]: \[á, à, å\]"""
# 断言应该引发 AssertionError，并且匹配预期的错误消息 `msg`
with pytest.raises(AssertionError, match=msg):
    # 调用 `tm.assert_numpy_array_equal` 比较两个包含不同 Unicode 对象的 numpy 数组
    tm.assert_numpy_array_equal(
        np.array(["á", "à", "ä"]), np.array(["á", "à", "å"])
    )
    # 检查 nulls_fixture 是否具有 "copy" 属性（即是否可复制）
    if hasattr(nulls_fixture, "copy"):
        # 如果可以复制，则使用对象的 copy 方法创建副本
        other = nulls_fixture.copy()
    else:
        # 如果没有 copy 方法，则使用 copy 模块的通用复制方法创建副本
        other = copy.copy(nulls_fixture)
    # 使用对象 other 创建一个 numpy 数组 a，其中元素类型为 object
    a = np.array([other], dtype=object)
    # 使用测试框架 tm 来断言 numpy 数组 a 和 b 相等
    tm.assert_numpy_array_equal(a, b)
# 定义一个测试函数，用于验证当numpy数组包含不同类型的NA值时是否引发断言错误
def test_numpy_array_equal_different_na():
    # 创建包含NaN的numpy数组a，数据类型为object
    a = np.array([np.nan], dtype=object)
    # 创建包含pd.NA的numpy数组b，数据类型为object
    b = np.array([pd.NA], dtype=object)

    # 设置错误消息，用于断言异常时的匹配
    msg = """numpy array are different

numpy array values are different \(100.0 %\)
\[left\]:  \[nan\]
\[right\]: \[<NA>\]"""

    # 使用pytest断言，验证调用tm.assert_numpy_array_equal(a, b)时是否会引发AssertionError，并且错误消息匹配msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


这段代码是一个测试函数，用于验证当两个包含不同类型NA值的numpy数组进行比较时，是否会引发AssertionError异常，同时验证异常消息是否与预期的msg匹配。
```