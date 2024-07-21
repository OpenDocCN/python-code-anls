# `.\pytorch\test\torch_np\test_random.py`

```py
# Owner(s): ["module: dynamo"]

"""Light smoke test switching between numpy to pytorch random streams.
"""
# 导入必要的库和模块
from contextlib import contextmanager  # 上下文管理器
from functools import partial  # 函数偏应用

import numpy as _np  # 导入NumPy，别名_np
import pytest  # 导入pytest测试框架

import torch._dynamo.config as config  # 导入torch._dynamo.config模块

import torch._numpy as tnp  # 导入torch._numpy模块，并用别名tnp
from torch._numpy.testing import assert_equal  # 导入assert_equal函数

from torch.testing._internal.common_utils import (  # 导入测试工具函数和类
    instantiate_parametrized_tests,  # 参数化测试实例化装饰器
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试函数
    subtest,  # 子测试装饰器
    TestCase,  # 测试用例类
)


@contextmanager
def control_stream(use_numpy=False):
    """Context manager to control the random stream type."""
    with config.patch(use_numpy_random_stream=use_numpy):  # 使用config.patch设置numpy随机流
        yield  # 执行被管理的代码块


@instantiate_parametrized_tests
class TestScalarReturn(TestCase):
    """Test case class for scalar return types."""
    
    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,  # 正态分布随机数生成器
            tnp.random.rand,  # 均匀分布随机数生成器
            partial(tnp.random.randint, 0, 5),  # 随机整数生成器
            tnp.random.randn,  # 标准正态分布随机数生成器
            subtest(tnp.random.random, name="random_random"),  # 子测试：均匀分布随机数生成器
            subtest(tnp.random.random_sample, name="random_sample"),  # 子测试：均匀分布随机数生成器
            tnp.random.sample,  # 随机抽样函数
            tnp.random.uniform,  # 均匀分布随机数生成器
        ],
    )
    def test_rndm_scalar(self, func, use_numpy):
        """Test function to verify scalar return types."""
        # default `size` means a python scalar return
        with control_stream(use_numpy):  # 使用控制随机流的上下文管理器
            r = func()  # 调用指定的随机数生成函数
        assert isinstance(r, (int, float))  # 断言返回值是整数或浮点数类型

    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,  # 正态分布随机数生成器
            tnp.random.rand,  # 均匀分布随机数生成器
            partial(tnp.random.randint, 0, 5),  # 随机整数生成器
            tnp.random.randn,  # 标准正态分布随机数生成器
            subtest(tnp.random.random, name="random_random"),  # 子测试：均匀分布随机数生成器
            subtest(tnp.random.random_sample, name="random_sample"),  # 子测试：均匀分布随机数生成器
            tnp.random.sample,  # 随机抽样函数
            tnp.random.uniform,  # 均匀分布随机数生成器
        ],
    )
    def test_rndm_array(self, func, use_numpy):
        """Test function to verify array return types."""
        with control_stream(use_numpy):  # 使用控制随机流的上下文管理器
            if func in (tnp.random.rand, tnp.random.randn):  # 如果是均匀分布或标准正态分布生成器
                r = func(10)  # 生成长度为10的随机数组
            else:
                r = func(size=10)  # 使用指定大小生成随机数组
        assert isinstance(r, tnp.ndarray)  # 断言返回值是NumPy数组类型


@instantiate_parametrized_tests
class TestShuffle(TestCase):
    """Test case class for testing shuffle operations."""
    
    @parametrize("use_numpy", [True, False])
    def test_1d(self, use_numpy):
        """Test function for 1-dimensional shuffle."""
        ax = tnp.asarray([1, 2, 3, 4, 5, 6])  # 创建NumPy数组
        ox = ax.copy()  # 复制原始数组

        tnp.random.seed(1234)  # 设置随机数种子
        tnp.random.shuffle(ax)  # 打乱数组顺序

        assert isinstance(ax, tnp.ndarray)  # 断言结果是NumPy数组类型
        assert not (ax == ox).all()  # 断言打乱后的数组与原数组不全等

    @parametrize("use_numpy", [True, False])
    def test_2d(self, use_numpy):
        """Test function for 2-dimensional shuffle."""
        # np.shuffle only shuffles the first axis
        ax = tnp.asarray([[1, 2, 3], [4, 5, 6]])  # 创建二维NumPy数组
        ox = ax.copy()  # 复制原始数组

        tnp.random.seed(1234)  # 设置随机数种子
        tnp.random.shuffle(ax)  # 打乱数组顺序

        assert isinstance(ax, tnp.ndarray)  # 断言结果是NumPy数组类型
        assert not (ax == ox).all()  # 断言打乱后的数组与原数组不全等

    @parametrize("use_numpy", [True, False])
    # 定义一个测试函数，用于测试列表的随机打乱操作，接受一个布尔值参数 use_numpy
    def test_shuffle_list(self, use_numpy):
        # 在 eager 模式下，拒绝对列表进行打乱操作
        # 在 dynamo 模式下，总是回退到使用 numpy
        # 注意：当 USE_NUMPY_STREAM == False 时，这意味着对列表或数组进行打乱时使用的随机流是不同的
        # 创建一个包含元素 1, 2, 3 的列表 x
        x = [1, 2, 3]
        # 使用 pytest 模块断言抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            # 调用 numpy 的随机打乱函数对列表 x 进行打乱
            tnp.random.shuffle(x)
# 为了实现参数化测试的自动化装饰器
@instantiate_parametrized_tests
# 定义一个名为 TestChoice 的测试类，继承自 TestCase
class TestChoice(TestCase):
    
    # 使用 parametrize 装饰器，为 test_choice 方法提供两种参数化选项：True 和 False
    @parametrize("use_numpy", [True, False])
    # 定义一个测试方法 test_choice，接受 use_numpy 参数
    def test_choice(self, use_numpy):
        # 定义一个关键字参数字典 kwds，包括 size=3, replace=False, p=[0.1, 0, 0.3, 0.6, 0]
        kwds = dict(size=3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        
        # 控制随机数流，根据 use_numpy 参数选择使用 numpy 还是其它随机数生成器
        with control_stream(use_numpy):
            # 设定 numpy 随机数种子为 12345
            tnp.random.seed(12345)
            # 使用 tnp.random.choice 方法从 0 到 4 之间选择一个数字，根据 kwds 参数
            x = tnp.random.choice(5, **kwds)
            # 再次设定 numpy 随机数种子为 12345
            tnp.random.seed(12345)
            # 使用 tnp.random.choice 方法从数组 tnp.arange(5) 中选择一个数字，根据 kwds 参数
            x_1 = tnp.random.choice(tnp.arange(5), **kwds)
            # 断言两次生成的随机数结果 x 和 x_1 相等
            assert_equal(x, x_1)

# 定义一个名为 TestNumpyGlobal 的测试类，继承自 TestCase
class TestNumpyGlobal(TestCase):
    
    # 定义一个测试方法 test_numpy_global
    def test_numpy_global(self):
        # 控制随机数流，强制使用 numpy
        with control_stream(use_numpy=True):
            # 设定 numpy 随机数种子为 12345
            tnp.random.seed(12345)
            # 使用 tnp.random.uniform 方法生成一个 0 到 1 之间的均匀分布随机数数组，大小为 11
            x = tnp.random.uniform(0, 1, size=11)

        # 断言生成的随机数数组 x 与 numpy 生成的相同
        # 比较 tnp.asarray(x_np) 与 x 是否相等
        _np.random.seed(12345)
        x_np = _np.random.uniform(0, 1, size=11)
        assert_equal(x, tnp.asarray(x_np))

        # 切换到非 numpy 的随机数流
        with control_stream(use_numpy=False):
            # 设定非 numpy 随机数种子为 12345
            tnp.random.seed(12345)
            # 使用 tnp.random.uniform 方法生成一个 0 到 1 之间的均匀分布随机数数组，大小为 11
            x_1 = tnp.random.uniform(0, 1, size=11)

        # 断言两次生成的随机数数组 x 和 x_1 不全相等
        assert not (x_1 == x).all()

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```