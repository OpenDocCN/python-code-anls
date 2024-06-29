# `.\numpy\numpy\_core\tests\test_cpu_dispatcher.py`

```
from numpy._core._multiarray_umath import (
    __cpu_features__, __cpu_baseline__, __cpu_dispatch__
)
from numpy._core import _umath_tests
from numpy.testing import assert_equal

def test_dispatcher():
    """
    Testing the utilities of the CPU dispatcher
    """
    # 定义处理器支持的目标特性列表
    targets = (
        "SSE2", "SSE41", "AVX2",
        "VSX", "VSX2", "VSX3",
        "NEON", "ASIMD", "ASIMDHP",
        "VX", "VXE"
    )
    highest_sfx = "" # 用于存储最高特性的后缀，默认为空表示基线没有后缀
    all_sfx = []  # 存储所有特性的后缀列表

    # 逆序遍历目标特性列表
    for feature in reversed(targets):
        # 如果特性在基线特性中，则跳过
        if feature in __cpu_baseline__:
            continue
        # 检查编译器和运行机器是否支持该特性
        if feature not in __cpu_dispatch__ or not __cpu_features__[feature]:
            continue

        # 如果当前没有最高特性后缀，则使用当前特性作为最高特性后缀
        if not highest_sfx:
            highest_sfx = "_" + feature
        # 将当前特性添加到所有特性后缀列表中
        all_sfx.append("func" + "_" + feature)

    # 调用_umath_tests模块中的test_dispatch函数进行测试
    test = _umath_tests.test_dispatch()

    # 断言检查测试结果中的函数名称是否匹配最高特性后缀
    assert_equal(test["func"], "func" + highest_sfx)
    # 断言检查测试结果中的变量名称是否匹配最高特性后缀
    assert_equal(test["var"], "var"  + highest_sfx)

    # 如果存在最高特性后缀，则进一步检查带基线的函数和变量名称是否匹配最高特性后缀
    if highest_sfx:
        assert_equal(test["func_xb"], "func" + highest_sfx)
        assert_equal(test["var_xb"], "var"  + highest_sfx)
    else:
        # 如果不存在最高特性后缀，则函数和变量名称应该为"nobase"
        assert_equal(test["func_xb"], "nobase")
        assert_equal(test["var_xb"], "nobase")

    # 添加基线特性的函数名称到所有特性后缀列表中
    all_sfx.append("func")
    # 断言检查测试结果中的"all"键对应的值是否等于所有特性后缀列表
    assert_equal(test["all"], all_sfx)
```