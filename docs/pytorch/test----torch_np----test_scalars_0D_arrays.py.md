# `.\pytorch\test\torch_np\test_scalars_0D_arrays.py`

```
# Owner(s): ["module: dynamo"]

"""
Basic tests to assert and illustrate the behavior around the decision to use 0D
arrays in place of array scalars.

Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*

Also test the isscalar function (which is deliberately a bit more lax).
"""

# 导入必要的模块和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)

# 根据测试环境选择正确的 numpy 包
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal

# 定义参数化的值列表，包括多种类型的 subtest
parametrize_value = parametrize(
    "value",
    [
        subtest(np.int64(42), name="int64"),       # 使用 int64 创建的 subtest
        subtest(np.array(42), name="array"),       # 使用 array 创建的 subtest
        subtest(np.asarray(42), name="asarray"),   # 使用 asarray 创建的 subtest
        subtest(np.asarray(np.int64(42)), name="asarray_int"),  # 使用 asarray 转换 int64 创建的 subtest
    ],
)

# 定义一个测试类，用于测试数组标量的行为
@instantiate_parametrized_tests
class TestArrayScalars(TestCase):
    
    @parametrize_value
    def test_array_scalar_basic(self, value):
        # 断言标量的属性
        assert value.ndim == 0             # 标量的维度应为 0
        assert value.shape == ()           # 标量的形状应为空元组
        assert value.size == 1             # 标量的尺寸应为 1
        assert value.dtype == np.dtype("int64")  # 标量的数据类型应为 int64

    @parametrize_value
    def test_conversion_to_int(self, value):
        # 测试将标量转换为整数的行为
        py_scalar = int(value)
        assert py_scalar == 42            # 转换后的整数应为 42
        assert isinstance(py_scalar, int) # 转换后的结果应为整数类型
        assert not isinstance(value, int) # 原始值不应为整数类型

    @parametrize_value
    def test_decay_to_py_scalar(self, value):
        # 测试标量的衰减行为，即标量与列表相乘的结果
        # NumPy 区分标量和 0D 数组。例如，`scalar * list` 等同于 `int(scalar) * list`，
        # 但 `0D array * list` 等同于 `0D array * np.asarray(list)`。
        # 我们的标量遵循 0D 数组的行为（因为它们本质上也是 0D 数组）。
        lst = [1, 2, 3]

        product = value * lst
        assert isinstance(product, np.ndarray)  # 乘积应为 NumPy 数组
        assert product.shape == (3,)            # 数组形状应为 (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])  # 数组内容应与预期相等

        # 右乘相同测试
        product = lst * value
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])

    def test_scalar_comparisons(self):
        # 测试标量的比较行为
        scalar = np.int64(42)
        arr = np.array(42)

        assert arr == scalar   # 数组与标量相等比较
        assert arr >= scalar   # 数组大于等于标量比较
        assert arr <= scalar   # 数组小于等于标量比较

        assert scalar == 42    # 标量与整数相等比较
        assert arr == 42       # 数组与整数相等比较


# @xfailIfTorchDynamo
@instantiate_parametrized_tests
class TestIsScalar(TestCase):
    #
    # np.isscalar(...) checks that its argument is a numeric object with exactly one element.
    #
    # This differs from NumPy which also requires that shape == ().
    #
    # 创建一个包含不同数据类型的标量测试用例列表
    scalars = [
        subtest(42, "literal"),  # 测试整型字面量 42
        subtest(int(42.0), "int"),  # 测试整型数值 42.0
        subtest(np.float32(42), "float32"),  # 测试单精度浮点数 42
        subtest(np.array(42), "array_0D", decorators=[xfailIfTorchDynamo]),  # 测试零维数组 [42]
        subtest([42], "list", decorators=[xfailIfTorchDynamo]),  # 测试列表 [42]
        subtest([[42]], "list-list", decorators=[xfailIfTorchDynamo]),  # 测试嵌套列表 [[42]]
        subtest(np.array([42]), "array_1D", decorators=[xfailIfTorchDynamo]),  # 测试一维数组 [42]
        subtest(np.array([[42]]), "array_2D", decorators=[xfailIfTorchDynamo]),  # 测试二维数组 [[42]]
    ]

    # 导入 math 模块
    import math

    # 创建一个包含非标量的测试用例列表
    not_scalars = [
        int,  # 整型类型本身
        np.float32,  # 单精度浮点数类型本身
        subtest("s", decorators=[xfailIfTorchDynamo]),  # 字符串类型的 subtest 测试用例
        subtest("string", decorators=[xfailIfTorchDynamo]),  # 字符串类型的 subtest 测试用例
        (),  # 空元组
        [],  # 空列表
        math.sin,  # math.sin 函数
        np,  # NumPy 模块本身
        np.transpose,  # NumPy 的转置函数
        [1, 2],  # 包含两个元素的列表
        np.asarray([1, 2]),  # 将 [1, 2] 转换为 NumPy 数组
        np.float32([1, 2]),  # 创建单精度浮点数类型的数组 [1.0, 2.0]
    ]

    # 参数化装饰器，用于测试标量的函数
    @parametrize("value", scalars)
    def test_is_scalar(self, value):
        assert np.isscalar(value)

    # 参数化装饰器，用于测试非标量的函数
    @parametrize("value", not_scalars)
    def test_is_not_scalar(self, value):
        assert not np.isscalar(value)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```