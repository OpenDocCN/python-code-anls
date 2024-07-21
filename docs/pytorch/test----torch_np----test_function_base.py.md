# `.\pytorch\test\torch_np\test_function_base.py`

```
# Owner(s): ["module: dynamo"]

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 torch.testing._internal.common_utils 中导入需要的函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

# 如果 TEST_WITH_TORCHDYNAMO 为真，则使用 NumPy，否则使用 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np  # 导入 NumPy 库
    from numpy.testing import assert_equal  # 导入 NumPy 的断言函数
else:
    import torch._numpy as np  # 导入 Torch NumPy 接口
    from torch._numpy.testing import assert_equal  # 导入 Torch NumPy 的断言函数


class TestAppend(TestCase):
    # tests taken from np.append docstring
    
    # 测试基本的 append 操作
    def test_basic(self):
        # 在 [1, 2, 3] 后面追加 [[4, 5, 6], [7, 8, 9]]，结果应该是 [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        assert_equal(result, np.arange(1, 10, dtype=int))  # 断言结果与期望值一致

        # 指定 axis=0，在 [[1, 2, 3], [4, 5, 6]] 后面追加 [7, 8, 9]，结果应该是 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        assert_equal(result, np.arange(1, 10, dtype=int).reshape((3, 3)))  # 断言结果与期望值一致

        # 使用 pytest 检查当 axis=0 时，错误地尝试追加一个一维数组 [7, 8, 9] 到二维数组的情况
        with pytest.raises((RuntimeError, ValueError)):
            np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)


if __name__ == "__main__":
    run_tests()  # 执行所有的测试用例
```