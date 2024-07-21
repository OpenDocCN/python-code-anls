# `.\pytorch\test\torch_np\numpy_tests\lib\test_arraypad.py`

```py
# Owner(s): ["module: dynamo"]

# 引入 skipIf 别名 skipif 用于条件性跳过测试
from unittest import skipIf as skipif

# 从 torch.testing._internal.common_utils 模块导入所需的符号
from torch.testing._internal.common_utils import (
    run_tests,  # 导入 run_tests 函数
    TEST_WITH_TORCHDYNAMO,  # 导入测试标志 TEST_WITH_TORCHDYNAMO
    TestCase,  # 导入 TestCase 类
    xpassIfTorchDynamo,  # 导入 xpassIfTorchDynamo 装饰器
)

# 根据测试标志决定使用 NumPy 还是 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 NumPy
    from numpy.testing import assert_allclose, assert_array_equal  # 导入 NumPy 的测试函数
else:
    import torch._numpy as np  # 导入 torch._numpy
    from torch._numpy.testing import assert_allclose, assert_array_equal  # 导入 torch._numpy 的测试函数


class TestConstant(TestCase):
    @xpassIfTorchDynamo  # 使用 xpassIfTorchDynamo 装饰器标记的测试方法
    def test_check_constant_float(self):
        # 如果输入数组是整数，但常数值是浮点数，则填充数组的数据类型保持不变
        arr = np.arange(30).reshape(5, 6)
        test = np.pad(arr, (1, 2), mode="constant", constant_values=1.1)
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 2, 3, 4, 5, 1, 1],
                [1, 6, 7, 8, 9, 10, 11, 1, 1],
                [1, 12, 13, 14, 15, 16, 17, 1, 1],
                [1, 18, 19, 20, 21, 22, 23, 1, 1],
                [1, 24, 25, 26, 27, 28, 29, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        assert_allclose(test, expected)  # 断言 test 与 expected 的接近程度

    def test_check_constant_float2(self):
        # 如果输入数组是浮点数，并且常数值也是浮点数，则填充数组的数据类型保持不变 - 在这里保留浮点数常数
        arr = np.arange(30).reshape(5, 6)
        arr_float = arr.astype(np.float64)  # 将 arr 转换为 float64 类型
        test = np.pad(arr_float, ((1, 2), (1, 2)), mode="constant", constant_values=1.1)
        expected = np.array(
            [
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                [1.1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 1.1],
                [1.1, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1.1, 1.1],
                [1.1, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 1.1, 1.1],
                [1.1, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 1.1, 1.1],
                [1.1, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 1.1, 1.1],
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            ]
        )
        assert_allclose(test, expected)  # 断言 test 与 expected 的接近程度

    @xpassIfTorchDynamo  # 使用 xpassIfTorchDynamo 装饰器标记的测试方法
    # 定义测试函数，用于检查使用常数填充的奇数填充量
    def test_check_constant_odd_pad_amount(self):
        # 创建一个形状为 (5, 6) 的 NumPy 数组，包含数字 0 到 29
        arr = np.arange(30).reshape(5, 6)
        # 对数组进行常数填充，左右各增加 2 列，上方增加 1 行，下方增加 1 行，填充值为 3
        test = np.pad(arr, ((1,), (2,)), mode="constant", constant_values=3)
        # 预期的填充后的数组
        expected = np.array(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 3, 0, 1, 2, 3, 4, 5, 3, 3],
                [3, 3, 6, 7, 8, 9, 10, 11, 3, 3],
                [3, 3, 12, 13, 14, 15, 16, 17, 3, 3],
                [3, 3, 18, 19, 20, 21, 22, 23, 3, 3],
                [3, 3, 24, 25, 26, 27, 28, 29, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]
        )
        # 断言填充后的数组与预期的数组相等
        assert_allclose(test, expected)

    @xpassIfTorchDynamo  # (reason="tuple values")
    # 标记为 xpassIfTorchDynamo 的测试函数，用于检查使用常数填充的二维数组
    def test_check_constant_pad_2d(self):
        # 创建一个形状为 (2, 2) 的 NumPy 数组，包含数字 0 到 3
        arr = np.arange(4).reshape(2, 2)
        # 对数组进行常数填充，上方增加 1 行，下方增加 2 行，左侧增加 1 列，右侧增加 3 列，填充值为 ((1, 2), (3, 4))
        test = np.lib.pad(
            arr, ((1, 2), (1, 3)), mode="constant", constant_values=((1, 2), (3, 4))
        )
        # 预期的填充后的数组
        expected = np.array(
            [
                [3, 1, 1, 4, 4, 4],
                [3, 0, 1, 4, 4, 4],
                [3, 2, 3, 4, 4, 4],
                [3, 2, 2, 4, 4, 4],
                [3, 2, 2, 4, 4, 4],
            ]
        )
        # 断言填充后的数组与预期的数组相等
        assert_allclose(test, expected)

    @skipif(
        True, reason="passes on MacOS, fails otherwise"
    )  # (reason="int64 overflow")
    # 标记为 skipif 的测试函数，用于检查大整数的填充行为
    def test_check_large_integers(self):
        # 计算 int64 类型的最大值
        int64_max = 2**63 - 1
        # 创建一个包含 5 个 int64 类型最大值的数组
        arr = np.full(5, int64_max, dtype=np.int64)
        # 对数组进行常数填充，左右各增加 1 个元素，填充值为数组的最小值（即 int64 类型的最小值）
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        # 预期的填充后的数组，所有元素为 int64 类型的最大值
        expected = np.full(7, int64_max, dtype=np.int64)
        # 断言填充后的数组与预期的数组相等
        assert_array_equal(test, expected)

    # 定义测试函数，用于检查空维度的填充行为
    def test_pad_empty_dimension(self):
        # 创建一个形状为 (3, 0, 2) 的全零数组
        arr = np.zeros((3, 0, 2))
        # 对数组进行常数填充，第一个维度不填充，第二个维度左右各增加 2 个元素，第三个维度左右各增加 1 个元素
        result = np.pad(arr, [(0,), (2,), (1,)], mode="constant")
        # 断言填充后的数组形状为 (3, 4, 4)
        assert result.shape == (3, 4, 4)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```