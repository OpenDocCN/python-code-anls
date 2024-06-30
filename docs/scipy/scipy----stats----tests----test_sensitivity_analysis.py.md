# `D:\src\scipysrc\scipy\scipy\stats\tests\test_sensitivity_analysis.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose, assert_array_less  # 导入 NumPy 测试模块中的函数
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from scipy import stats  # 导入 SciPy 统计模块
from scipy.stats import sobol_indices  # 导入 Sobol 指数计算函数
from scipy.stats._resampling import BootstrapResult  # 导入 Bootstrap 计算结果类
from scipy.stats._sensitivity_analysis import (  # 导入敏感性分析相关函数和类
    BootstrapSobolResult, f_ishigami, sample_AB, sample_A_B
)


@pytest.fixture(scope='session')  # 定义 pytest 的 session 级别的 fixture
def ishigami_ref_indices():
    """Reference values for Ishigami from Saltelli2007.

    Chapter 4, exercise 5 pages 179-182.
    """
    a = 7.  # 参数 a
    b = 0.1  # 参数 b

    var = 0.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18  # 方差计算
    v1 = 0.5 + b*np.pi**4/5 + b**2*np.pi**8/50  # 第一个 Sobol 指数
    v2 = a**2/8  # 第二个 Sobol 指数
    v3 = 0  # 第三个 Sobol 指数
    v12 = 0  # 交叉 Sobol 指数
    # v13: mistake in the book, see other derivations e.g. in 10.1002/nme.4856
    v13 = b**2*np.pi**8*8/225  # 另一个 Sobol 指数
    v23 = 0  # 另一个交叉 Sobol 指数

    s_first = np.array([v1, v2, v3])/var  # 第一阶段 Sobol 指数
    s_second = np.array([
        [0., 0., v13],
        [v12, 0., v23],
        [v13, v23, 0.]
    ])/var  # 第二阶段 Sobol 指数
    s_total = s_first + s_second.sum(axis=1)  # 总 Sobol 指数

    return s_first, s_total  # 返回 Sobol 指数的计算结果


def f_ishigami_vec(x):
    """Output of shape (2, n)."""
    res = f_ishigami(x)  # 计算 Ishigami 函数的输出
    return res, res  # 返回形状为 (2, n) 的结果


class TestSobolIndices:
    dists = [
        stats.uniform(loc=-np.pi, scale=2*np.pi)  # 定义均匀分布的统计对象
    ] * 3

    def test_sample_AB(self):
        # (d, n)
        A = np.array(
            [[1, 4, 7, 10],
             [2, 5, 8, 11],
             [3, 6, 9, 12]]
        )  # 定义矩阵 A
        B = A + 100  # 创建矩阵 B，元素为 A 中元素加 100
        # (d, d, n)
        ref = np.array(
            [[[101, 104, 107, 110],
              [2, 5, 8, 11],
              [3, 6, 9, 12]],
             [[1, 4, 7, 10],
              [102, 105, 108, 111],
              [3, 6, 9, 12]],
             [[1, 4, 7, 10],
              [2, 5, 8, 11],
              [103, 106, 109, 112]]]
        )  # 定义期望的结果矩阵
        AB = sample_AB(A=A, B=B)  # 调用 sample_AB 函数进行计算
        assert_allclose(AB, ref)  # 使用 NumPy 测试模块中的 assert_allclose 检查 AB 和 ref 是否接近

    @pytest.mark.xslow  # 标记测试为特别慢的测试用例
    @pytest.mark.xfail_on_32bit("Can't create large array for test")  # 如果在 32 位系统上失败，则标记测试为失败
    @pytest.mark.parametrize(
        'func',
        [f_ishigami, pytest.param(f_ishigami_vec, marks=pytest.mark.slow)],
        ids=['scalar', 'vector']
    )  # 参数化测试，运行 scalar 和 vector 版本的 f_ishigami 函数测试
    # 定义测试函数 test_ishigami，接受三个参数：ishigami_ref_indices、func 和 self
    def test_ishigami(self, ishigami_ref_indices, func):
        # 使用指定的种子创建随机数生成器对象 rng
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 调用 sobol_indices 函数进行计算，返回 Sobol 指数的结果 res
        res = sobol_indices(
            func=func, n=4096,
            dists=self.dists,
            random_state=rng
        )

        # 如果 func 的名称为 'f_ishigami_vec'，则调整 ishigami_ref_indices 的值
        if func.__name__ == 'f_ishigami_vec':
            ishigami_ref_indices = [
                    [ishigami_ref_indices[0], ishigami_ref_indices[0]],
                    [ishigami_ref_indices[1], ishigami_ref_indices[1]]
            ]

        # 断言第一阶和总体阶的结果与参考值 ishigami_ref_indices 相似，允许的误差为 1e-2
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)

        # 断言 bootstrap 过程中的结果为 None
        assert res._bootstrap_result is None
        # 运行 bootstrap 方法生成 BootstrapSobolResult 对象，并进行类型检查
        bootstrap_res = res.bootstrap(n_resamples=99)
        assert isinstance(bootstrap_res, BootstrapSobolResult)
        assert isinstance(res._bootstrap_result, BootstrapResult)

        # 断言 bootstrap 结果中置信区间的维度为 2
        assert res._bootstrap_result.confidence_interval.low.shape[0] == 2
        # 断言置信区间的维度与第一阶的结果相同
        assert res._bootstrap_result.confidence_interval.low[1].shape \
               == res.first_order.shape

        # 断言 bootstrap_res 的第一阶和总体阶置信区间的维度与结果的维度相同
        assert bootstrap_res.first_order.confidence_interval.low.shape \
               == res.first_order.shape
        assert bootstrap_res.total_order.confidence_interval.low.shape \
               == res.total_order.shape

        # 断言第一阶和总体阶的置信区间下界比 res 的第一阶和总体阶更小
        assert_array_less(
            bootstrap_res.first_order.confidence_interval.low, res.first_order
        )
        assert_array_less(
            res.first_order, bootstrap_res.first_order.confidence_interval.high
        )
        assert_array_less(
            bootstrap_res.total_order.confidence_interval.low, res.total_order
        )
        assert_array_less(
            res.total_order, bootstrap_res.total_order.confidence_interval.high
        )

        # 再次调用 bootstrap 方法，使用先前的结果并改变一个参数，断言返回的类型是 BootstrapSobolResult
        assert isinstance(
            res.bootstrap(confidence_level=0.9, n_resamples=99),
            BootstrapSobolResult
        )
        assert isinstance(res._bootstrap_result, BootstrapResult)
    # 定义一个测试函数，用于计算 Sobol 灵敏度指数，并与参考指数进行比较
    def test_func_dict(self, ishigami_ref_indices):
        # 使用特定种子初始化随机数生成器
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 设置样本数量
        n = 4096
        # 定义三个均匀分布的随机变量
        dists = [
            stats.uniform(loc=-np.pi, scale=2*np.pi),
            stats.uniform(loc=-np.pi, scale=2*np.pi),
            stats.uniform(loc=-np.pi, scale=2*np.pi)
        ]

        # 从定义的分布中采样生成 A 和 B
        A, B = sample_A_B(n=n, dists=dists, random_state=rng)
        # 根据 A 和 B 生成 AB
        AB = sample_AB(A=A, B=B)

        # 构建函数字典，包含 f_A、f_B 和 f_AB 的计算结果
        func = {
            'f_A': f_ishigami(A).reshape(1, -1),
            'f_B': f_ishigami(B).reshape(1, -1),
            'f_AB': f_ishigami(AB).reshape((3, 1, -1))
        }

        # 计算 Sobol 灵敏度指数，使用自定义的分布和随机数生成器
        res = sobol_indices(
            func=func, n=n,
            dists=dists,
            random_state=rng
        )
        # 断言第一阶灵敏度指数与参考值的接近程度
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)

        # 再次计算 Sobol 灵敏度指数，只使用随机数生成器作为输入
        res = sobol_indices(
            func=func, n=n,
            random_state=rng
        )
        # 断言第一阶灵敏度指数与参考值的接近程度
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)

    # 定义另一个测试方法，用于测试具有自定义方法的 Sobol 灵敏度指数计算
    def test_method(self, ishigami_ref_indices):
        # 定义用于计算 Sobol 灵敏度指数的 Jansen 方法
        def jansen_sobol(f_A, f_B, f_AB):
            """Jansen for S and Sobol' for St.

            From Saltelli2010, table 2 formulations (c) and (e)."""
            # 计算变量的方差
            var = np.var([f_A, f_B], axis=(0, -1))

            # 计算 S 指数
            s = (var - 0.5*np.mean((f_B - f_AB)**2, axis=-1)) / var
            # 计算 St 指数
            st = np.mean(f_A*(f_A - f_AB), axis=-1) / var

            return s.T, st.T

        # 使用特定种子初始化随机数生成器
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 计算 Sobol 灵敏度指数，使用自定义的方法 jansen_sobol
        res = sobol_indices(
            func=f_ishigami, n=4096,
            dists=self.dists,
            method=jansen_sobol,
            random_state=rng
        )

        # 断言第一阶和总阶灵敏度指数与参考值的接近程度
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)

        # 定义类型化的 Jansen Sobol 方法，用于特定输入和输出类型的 Sobol 灵敏度指数计算
        def jansen_sobol_typed(
            f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            return jansen_sobol(f_A, f_B, f_AB)

        # 再次计算 Sobol 灵敏度指数，使用类型化的方法 jansen_sobol_typed
        _ = sobol_indices(
            func=f_ishigami, n=8,
            dists=self.dists,
            method=jansen_sobol_typed,
            random_state=rng
        )

    # 定义用于测试归一化函数的测试方法
    def test_normalization(self, ishigami_ref_indices):
        # 使用特定种子初始化随机数生成器
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 计算 Sobol 灵敏度指数，使用带有常数增量的函数进行归一化
        res = sobol_indices(
            func=lambda x: f_ishigami(x) + 1000, n=4096,
            dists=self.dists,
            random_state=rng
        )

        # 断言第一阶和总阶灵敏度指数与参考值的接近程度
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)
    # 定义一个测试函数，用于测试常数函数的 Sobol 灵敏度指数计算
    def test_constant_function(self, ishigami_ref_indices):

        # 定义一个向量化的常数函数，返回形状为 (3, n) 的结果
        def f_ishigami_vec_const(x):
            """Output of shape (3, n)."""
            # 调用真实的石神真函数计算结果
            res = f_ishigami(x)
            # 返回计算结果、以及一个全部为 10 的矩阵和原始计算结果
            return res, res * 0 + 10, res

        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 使用 Sobol 算法计算指定函数的灵敏度指数
        res = sobol_indices(
            func=f_ishigami_vec_const, n=4096,
            dists=self.dists,
            random_state=rng
        )

        # 准备预期的石神真函数灵敏度指数的参考值列表
        ishigami_vec_indices = [
                [ishigami_ref_indices[0], [0, 0, 0], ishigami_ref_indices[0]],
                [ishigami_ref_indices[1], [0, 0, 0], ishigami_ref_indices[1]]
        ]

        # 断言计算结果的一阶和总阶灵敏度指数与预期值非常接近
        assert_allclose(res.first_order, ishigami_vec_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_vec_indices[1], atol=1e-2)

    # 使用 32 位系统时预期测试失败，并附加一条提示信息
    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    # 定义一个测试函数，用于测试更多收敛的情况下的 Sobol 灵敏度指数计算
    def test_more_converged(self, ishigami_ref_indices):
        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        # 使用 Sobol 算法计算石神真函数在更多收敛情况下的灵敏度指数
        res = sobol_indices(
            func=f_ishigami, n=2**19,  # 524288
            dists=self.dists,
            random_state=rng
        )

        # 断言计算结果的一阶和总阶灵敏度指数与预期值非常接近
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-4)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-4)
    # 定义测试函数 test_raises，用于测试异常情况
    def test_raises(self):

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"Each distribution in `dists` must have method `ppf`"
        # 测试当 dists 参数为字符串时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, dists="uniform")

        # 测试当 dists 参数为包含一个 lambda 函数的列表时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, dists=[lambda x: x])

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"The balance properties of Sobol'"
        # 测试当 dists 参数为包含 stats.uniform() 分布对象的列表时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=7, func=f_ishigami, dists=[stats.uniform()])

        # 测试当 n 参数为浮点数时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=4.1, func=f_ishigami, dists=[stats.uniform()])

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"'toto' is not a valid 'method'"
        # 测试当 method 参数为 'toto' 时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, method='toto')

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"must have the following signature"
        # 测试当 method 参数为 lambda 函数时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, method=lambda x: x)

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"'dists' must be defined when 'func' is a callable"
        # 测试当 func 参数为函数 f_ishigami 且没有指定 dists 参数时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami)

        # 定义一个返回形状不符合要求的函数 func_wrong_shape_output
        def func_wrong_shape_output(x):
            return x.reshape(-1, 1)

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"'func' output should have a shape"
        # 测试当 func 参数为 func_wrong_shape_output 函数时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(
                n=2, func=func_wrong_shape_output, dists=[stats.uniform()]
            )

        # 设置错误信息字符串模板，用于匹配异常信息
        message = r"When 'func' is a dictionary"
        # 测试当 func 参数为字典类型时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            sobol_indices(
                n=2, func={'f_A': [], 'f_AB': []}, dists=[stats.uniform()]
            )

        # 测试当 func 参数为字典类型且 f_B 键对应的值长度不合规范时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # f_B malformed
            sobol_indices(
                n=2,
                func={'f_A': [1, 2], 'f_B': [3], 'f_AB': [5, 6, 7, 8]},
            )

        # 测试当 func 参数为字典类型且 f_AB 键对应的值长度不合规范时，抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # f_AB malformed
            sobol_indices(
                n=2,
                func={'f_A': [1, 2], 'f_B': [3, 4], 'f_AB': [5, 6, 7]},
            )
```