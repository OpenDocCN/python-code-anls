# `D:\src\scipysrc\pandas\pandas\tests\test_aggregation.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 从 pandas 的核心模块中导入特定函数
from pandas.core.apply import (
    _make_unique_kwarg_list,  # 导入函数 _make_unique_kwarg_list
    maybe_mangle_lambdas,      # 导入函数 maybe_mangle_lambdas
)


# 定义单元测试函数：测试 maybe_mangle_lambdas 函数的行为是否符合预期
def test_maybe_mangle_lambdas_passthrough():
    # 断言：对于字符串 "mean"，函数应该返回 "mean"
    assert maybe_mangle_lambdas("mean") == "mean"
    # 断言：对于 lambda 函数，检查其名称是否为 "<lambda>"
    assert maybe_mangle_lambdas(lambda x: x).__name__ == "<lambda>"
    # 断言：对于包含单个 lambda 函数的列表，不应该进行函数名称修改
    assert maybe_mangle_lambdas([lambda x: x])[0].__name__ == "<lambda>"


# 定义单元测试函数：测试 maybe_mangle_lambdas 函数处理列表类对象的情况
def test_maybe_mangle_lambdas_listlike():
    # 定义包含两个 lambda 函数的列表
    aggfuncs = [lambda x: 1, lambda x: 2]
    # 调用函数进行处理
    result = maybe_mangle_lambdas(aggfuncs)
    # 断言：处理后的第一个函数名称应为 "<lambda_0>"
    assert result[0].__name__ == "<lambda_0>"
    # 断言：处理后的第二个函数名称应为 "<lambda_1>"
    assert result[1].__name__ == "<lambda_1>"
    # 断言：验证原始函数列表和处理后函数列表的执行结果一致性
    assert aggfuncs[0](None) == result[0](None)
    assert aggfuncs[1](None) == result[1](None)


# 定义单元测试函数：测试 maybe_mangle_lambdas 函数处理字典类对象的情况
def test_maybe_mangle_lambdas():
    # 定义包含 lambda 函数列表的字典
    func = {"A": [lambda x: 0, lambda x: 1]}
    # 调用函数进行处理
    result = maybe_mangle_lambdas(func)
    # 断言：处理后的第一个函数名称应为 "<lambda_0>"
    assert result["A"][0].__name__ == "<lambda_0>"
    # 断言：处理后的第二个函数名称应为 "<lambda_1>"    
    assert result["A"][1].__name__ == "<lambda_1>"


# 定义单元测试函数：测试 maybe_mangle_lambdas 函数处理带参数的 lambda 函数
def test_maybe_mangle_lambdas_args():
    # 定义包含带参数 lambda 函数列表的字典
    func = {"A": [lambda x, a, b=1: (0, a, b), lambda x: 1]}
    # 调用函数进行处理
    result = maybe_mangle_lambdas(func)
    # 断言：处理后的第一个函数名称应为 "<lambda_0>"
    assert result["A"][0].__name__ == "<lambda_0>"
    # 断言：处理后的第二个函数名称应为 "<lambda_1>"
    assert result["A"][1].__name__ == "<lambda_1>"
    # 断言：验证 lambda 函数的参数和默认参数的执行结果
    assert func["A"][0](0, 1) == (0, 1, 1)
    assert func["A"][0](0, 1, 2) == (0, 1, 2)
    assert func["A"][0](0, 2, b=3) == (0, 2, 3)


# 定义单元测试函数：测试 maybe_mangle_lambdas 函数对于已命名函数的处理
def test_maybe_mangle_lambdas_named():
    # 定义包含 numpy.mean 函数的字典
    func = {"C": np.mean, "D": {"foo": np.mean, "bar": np.mean}}
    # 调用函数进行处理
    result = maybe_mangle_lambdas(func)
    # 断言：处理结果应与原始输入相等
    assert result == func


# 使用 pytest 的参数化装饰器，定义多个测试用例
@pytest.mark.parametrize(
    "order, expected_reorder",  # 参数化的变量名和期望的重排序结果
    [
        (   # 第一个测试用例
            [   # 输入的原始顺序列表
                ("height", "<lambda>"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>"),
                ("weight", "<lambda>"),
            ],
            [   # 预期的重排序结果列表
                ("height", "<lambda>_0"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>_1"),
                ("weight", "<lambda>"),
            ],
        ),
        (   # 第二个测试用例
            [   # 输入的原始顺序列表
                ("col2", "min"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
            ],
            [   # 预期的重排序结果列表
                ("col2", "min"),
                ("col1", "<lambda>_0"),
                ("col1", "<lambda>_1"),
                ("col1", "<lambda>_2"),
            ],
        ),
        (   # 第三个测试用例
            [   # 输入的原始顺序列表
                ("col", "<lambda>"),
                ("col", "<lambda>"),
                ("col", "<lambda>"),
            ],
            [   # 预期的重排序结果列表
                ("col", "<lambda>_0"),
                ("col", "<lambda>_1"),
                ("col", "<lambda>_2"),
            ],
        ),
    ],
)
def test_make_unique(order, expected_reorder):
    # GH 27519, 测试 make_unique 函数是否正确重排序
    result = _make_unique_kwarg_list(order)

    # 断言：函数返回的结果应与预期的重排序结果相等
    assert result == expected_reorder
```