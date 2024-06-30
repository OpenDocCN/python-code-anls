# `D:\src\scipysrc\scipy\scipy\special\tests\test_kolmogorov.py`

```
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
import sys  # 导入 sys 模块，提供对 Python 解释器的访问
import pytest  # 导入 pytest 模块，用于编写和运行测试用例

import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_  # 导入 NumPy 测试模块中的 assert_

from scipy.special._testutils import FuncData  # 从 scipy.special._testutils 模块中导入 FuncData 类

from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi  # 导入 SciPy 库中的特殊函数
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
                                   _smirnovc, _smirnovci, _smirnovp)  # 从 scipy.special._ufuncs 中导入一些函数

_rtol = 1e-10  # 设置相对容差

class TestSmirnov:
    def test_nan(self):
        assert_(np.isnan(smirnov(1, np.nan)))  # 断言检查 smirnov 函数对 NaN 的处理

    def test_basic(self):
        dataset = [(1, 0.1, 0.9),  # 创建包含元组的列表
                   (1, 0.875, 0.125),
                   (2, 0.875, 0.125 * 0.125),
                   (3, 0.875, 0.125 * 0.125 * 0.125)]

        dataset = np.asarray(dataset)  # 转换为 NumPy 数组
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 smirnov 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期
        dataset[:, -1] = 1 - dataset[:, -1]  # 修改最后一列数据
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 _smirnovc 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期

    def test_x_equals_0(self):
        dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]  # 生成元组列表
        dataset = np.asarray(dataset)  # 转换为 NumPy 数组
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 smirnov 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期
        dataset[:, -1] = 1 - dataset[:, -1]  # 修改最后一列数据
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 _smirnovc 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期

    def test_x_equals_1(self):
        dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]  # 生成元组列表
        dataset = np.asarray(dataset)  # 转换为 NumPy 数组
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 smirnov 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期
        dataset[:, -1] = 1 - dataset[:, -1]  # 修改最后一列数据
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 _smirnovc 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期

    def test_x_equals_0point5(self):
        dataset = [(1, 0.5, 0.5),  # 创建包含元组的列表
                   (2, 0.5, 0.25),
                   (3, 0.5, 0.166666666667),
                   (4, 0.5, 0.09375),
                   (5, 0.5, 0.056),
                   (6, 0.5, 0.0327932098765),
                   (7, 0.5, 0.0191958707681),
                   (8, 0.5, 0.0112953186035),
                   (9, 0.5, 0.00661933257355),
                   (10, 0.5, 0.003888705)]

        dataset = np.asarray(dataset)  # 转换为 NumPy 数组
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 smirnov 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期
        dataset[:, -1] = 1 - dataset[:, -1]  # 修改最后一列数据
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol  # 使用 FuncData 对 _smirnovc 函数进行测试
        ).check(dtypes=[int, float, float])  # 检查数据类型是否符合预期
    # 定义一个测试函数，用于测试当 n = 1 时的情况
    def test_n_equals_1(self):
        # 生成一个从 0 到 1 的等间距数列，包括端点，共 101 个数
        x = np.linspace(0, 1, 101, endpoint=True)
        # 构建数据集，第一列为全 1，第二列为 x，第三列为 1-x
        dataset = np.column_stack([[1]*len(x), x, 1-x])
        # 创建 FuncData 对象，使用 smirnov 函数进行检查
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 将数据集的最后一列修改为 1 减去原始最后一列的值
        dataset[:, -1] = 1 - dataset[:, -1]
        # 再次创建 FuncData 对象，使用 _smirnovc 函数进行检查
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 定义一个测试函数，用于测试当 n = 2 时的情况
    def test_n_equals_2(self):
        # 生成一个从 0.5 到 1 的等间距数列，包括端点，共 101 个数
        x = np.linspace(0.5, 1, 101, endpoint=True)
        # 计算 p = (1-x)^2
        p = np.power(1-x, 2)
        # 创建长度与 x 相同的数组 n，每个元素为 2
        n = np.array([2] * len(x))
        # 构建数据集，第一列为 n，第二列为 x，第三列为 p
        dataset = np.column_stack([n, x, p])
        # 创建 FuncData 对象，使用 smirnov 函数进行检查
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 将数据集的最后一列修改为 1 减去原始最后一列的值
        dataset[:, -1] = 1 - dataset[:, -1]
        # 再次创建 FuncData 对象，使用 _smirnovc 函数进行检查
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 定义一个测试函数，用于测试当 n = 3 时的情况
    def test_n_equals_3(self):
        # 生成一个从 0.7 到 1 的等间距数列，包括端点，共 31 个数
        x = np.linspace(0.7, 1, 31, endpoint=True)
        # 计算 p = (1-x)^3
        p = np.power(1-x, 3)
        # 创建长度与 x 相同的数组 n，每个元素为 3
        n = np.array([3] * len(x))
        # 构建数据集，第一列为 n，第二列为 x，第三列为 p
        dataset = np.column_stack([n, x, p])
        # 创建 FuncData 对象，使用 smirnov 函数进行检查
        FuncData(
            smirnov, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 将数据集的最后一列修改为 1 减去原始最后一列的值
        dataset[:, -1] = 1 - dataset[:, -1]
        # 再次创建 FuncData 对象，使用 _smirnovc 函数进行检查
        FuncData(
            _smirnovc, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 定义一个测试函数，用于测试 n 较大时的情况
    def test_n_large(self):
        # 测试 n 较大时的情况
        # 概率值应随着 n 的增加而减小
        x = 0.4
        # 生成一个包含各种 n 值的数组，计算对应的 smirnov(n, x) 的值
        pvals = np.array([smirnov(n, x) for n in range(400, 1100, 20)])
        # 计算概率值数组的差分
        dfs = np.diff(pvals)
        # 断言所有的差分值都小于等于 0
        assert_(np.all(dfs <= 0), msg='Not all diffs negative %s' % dfs)
class TestSmirnovi:
    # 测试函数，验证当数据为 NaN 时函数的行为是否符合预期
    def test_nan(self):
        # 断言：调用 smirnovi 函数，期望返回的结果应该是 NaN
        assert_(np.isnan(smirnovi(1, np.nan)))

    # 测试基本情况，使用不同的数据集来验证函数的正确性
    def test_basic(self):
        # 准备数据集，包含不同的元组，每个元组表示一个测试用例
        dataset = [(1, 0.4, 0.6),
                   (1, 0.6, 0.4),
                   (1, 0.99, 0.01),
                   (1, 0.01, 0.99),
                   (2, 0.125 * 0.125, 0.875),
                   (3, 0.125 * 0.125 * 0.125, 0.875),
                   (10, 1.0 / 16 ** 10, 1 - 1.0 / 16)]

        # 将数据集转换为 NumPy 数组
        dataset = np.asarray(dataset)
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 测试当第二列数据 x 等于 0 时的情况
    def test_x_equals_0(self):
        # 准备数据集，生成元组，每个元组的第二列为 0，第三列为 1
        dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        # 将数据集转换为 NumPy 数组
        dataset = np.asarray(dataset)
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 测试当第二列数据 x 等于 1 时的情况
    def test_x_equals_1(self):
        # 准备数据集，生成元组，每个元组的第二列为 1，第三列为 0
        dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        # 将数据集转换为 NumPy 数组
        dataset = np.asarray(dataset)
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 测试当第一列数据 n 等于 1 时的情况
    def test_n_equals_1(self):
        # 生成一个从 0 到 1 等分为 101 个数的数组
        pp = np.linspace(0, 1, 101, endpoint=True)
        # 生成数据集，每个元组的第一列为 1，第二列为 pp 数组，第三列为 1 减去 pp 数组
        dataset = np.column_stack([[1]*len(pp), pp, 1-pp])
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 测试当第一列数据 n 等于 2 时的情况
    def test_n_equals_2(self):
        # 生成一个从 0.5 到 1 等分为 101 个数的数组
        x = np.linspace(0.5, 1, 101, endpoint=True)
        # 计算 p，p 等于 (1-x) 的平方
        p = np.power(1-x, 2)
        # 生成数据集，每个元组的第一列为 2，第二列为 p 数组，第三列为 x 数组
        n = np.array([2] * len(x))
        dataset = np.column_stack([n, p, x])
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 测试当第一列数据 n 等于 3 时的情况
    def test_n_equals_3(self):
        # 生成一个从 0.7 到 1 等分为 31 个数的数组
        x = np.linspace(0.7, 1, 31, endpoint=True)
        # 计算 p，p 等于 (1-x) 的三次方
        p = np.power(1-x, 3)
        # 生成数据集，每个元组的第一列为 3，第二列为 p 数组，第三列为 x 数组
        n = np.array([3] * len(x))
        dataset = np.column_stack([n, p, x])
        # 使用 FuncData 类来检查 smirnovi 函数的行为，验证结果是否符合预期
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        # 调整数据集的第二列，将每个元素替换为其补数
        dataset[:, 1] = 1 - dataset[:, 1]
        # 再次使用 FuncData 类来检查 _smirnovci 函数的行为，验证结果是否符合预期
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rt
    # 定义测试方法：验证统计量在不同情况下的正确性
    def test_round_trip(self):
        # 定义内部函数，用于计算 Smirnov 检验的结果
        def _sm_smi(n, p):
            return smirnov(n, smirnovi(n, p))

        # 定义内部函数，用于计算 Smirnov 检验的互补结果
        def _smc_smci(n, p):
            return _smirnovc(n, _smirnovci(n, p))

        # 准备测试数据集，包含不同情况下的样本数量和两种概率值
        dataset = [(1, 0.4, 0.4),
                   (1, 0.6, 0.6),
                   (2, 0.875, 0.875),
                   (3, 0.875, 0.875),
                   (3, 0.125, 0.125),
                   (10, 0.999, 0.999),
                   (10, 0.0001, 0.0001)]

        # 将数据集转换为 NumPy 数组格式
        dataset = np.asarray(dataset)
        
        # 创建 FuncData 对象，检验 _sm_smi 函数的输出
        FuncData(
            _sm_smi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        
        # 创建 FuncData 对象，检验 _smc_smci 函数的输出
        FuncData(
            _smc_smci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    # 定义测试方法：验证在 x = 0.5 情况下的 Smirnov 检验结果
    def test_x_equals_0point5(self):
        # 准备测试数据集，包含不同情况下的样本数量和固定的中间概率值
        dataset = [(1, 0.5, 0.5),
                   (2, 0.5, 0.366025403784),
                   (2, 0.25, 0.5),
                   (3, 0.5, 0.297156508177),
                   (4, 0.5, 0.255520481121),
                   (5, 0.5, 0.234559536069),
                   (6, 0.5, 0.21715965898),
                   (7, 0.5, 0.202722580034),
                   (8, 0.5, 0.190621765256),
                   (9, 0.5, 0.180363501362),
                   (10, 0.5, 0.17157867006)]

        # 将数据集转换为 NumPy 数组格式
        dataset = np.asarray(dataset)
        
        # 创建 FuncData 对象，检验 smirnovi 函数的输出
        FuncData(
            smirnovi, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
        
        # 将数据集中的第二列数据变为 1 减去原始值（求互补概率）
        dataset[:, 1] = 1 - dataset[:, 1]
        
        # 创建 FuncData 对象，检验 _smirnovci 函数的输出
        FuncData(
            _smirnovci, dataset, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])
class TestSmirnovp:
    # 定义测试类 TestSmirnovp
    def test_nan(self):
        # 测试处理 NaN 值的情况
        assert_(np.isnan(_smirnovp(1, np.nan)))

    def test_basic(self):
        # 检查在端点处的导数
        n1_10 = np.arange(1, 10)
        dataset0 = np.column_stack([n1_10,
                                    np.full_like(n1_10, 0),
                                    np.full_like(n1_10, -1)])
        FuncData(
            _smirnovp, dataset0, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

        n2_10 = np.arange(2, 10)
        dataset1 = np.column_stack([n2_10,
                                    np.full_like(n2_10, 1.0),
                                    np.full_like(n2_10, 0)])
        FuncData(
            _smirnovp, dataset1, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    def test_oneminusoneovern(self):
        # 检查在 x=1-1/n 处的导数
        n = np.arange(1, 20)
        x = 1.0/n
        xm1 = 1-1.0/n
        pp1 = -n * x**(n-1)
        pp1 -= (1-np.sign(n-2)**2) * 0.5  # n=2, x=0.5, 1-1/n = 0.5, 需要调整
        dataset1 = np.column_stack([n, xm1, pp1])
        FuncData(
            _smirnovp, dataset1, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    def test_oneovertwon(self):
        # 检查在 x=1/2n 处的导数（在 x=1/n 处不连续，因此在 x=1/2n 处进行检查）
        n = np.arange(1, 20)
        x = 1.0/2/n
        pp = -(n*x+1) * (1+x)**(n-2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(
            _smirnovp, dataset0, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    def test_oneovern(self):
        # 检查在 x=1/n 处的导数
        # （在 x=1/n 处不连续，难以确定是否在 x==1/n，仅使用 2 的幂次方的 n）
        n = 2**np.arange(1, 10)
        x = 1.0/n
        pp = -(n*x+1) * (1+x)**(n-2) + 0.5
        dataset0 = np.column_stack([n, x, pp])
        FuncData(
            _smirnovp, dataset0, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

    @pytest.mark.xfail(sys.maxsize <= 2**32,
                       reason="requires 64-bit platform")
    def test_oneovernclose(self):
        # 检查在 x=1/n 处的导数
        # （在 x=1/n 处不连续，在 x=1/n +/- 2epsilon 的两侧进行测试）
        n = np.arange(3, 20)

        x = 1.0/n - 2*np.finfo(float).eps
        pp = -(n*x+1) * (1+x)**(n-2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(
            _smirnovp, dataset0, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])

        x = 1.0/n + 2*np.finfo(float).eps
        pp = -(n*x+1) * (1+x)**(n-2) + 1
        dataset1 = np.column_stack([n, x, pp])
        FuncData(
            _smirnovp, dataset1, (0, 1), 2, rtol=_rtol
        ).check(dtypes=[int, float, float])


class TestKolmogorov:
    # 定义测试类 TestKolmogorov
    def test_nan(self):
        # 测试处理 NaN 值的情况
        assert_(np.isnan(kolmogorov(np.nan)))
    # 定义测试基本功能的方法
    def test_basic(self):
        # 定义数据集，包含一系列元组，每个元组有两个元素
        dataset = [(0, 1.0),
                   (0.5, 0.96394524366487511),
                   (0.8275735551899077, 0.5000000000000000),
                   (1, 0.26999967167735456),
                   (2, 0.00067092525577969533)]

        # 将数据集转换为 NumPy 数组
        dataset = np.asarray(dataset)
        # 使用 FuncData 类创建实例，并调用其 check 方法进行检查
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    # 定义测试 linspace 方法的方法
    def test_linspace(self):
        # 创建从 0 到 2.0 的等差数列，共 21 个数
        x = np.linspace(0, 2.0, 21)
        # 定义一个包含浮点数的数据集列表
        dataset = [1.0000000000000000, 1.0000000000000000, 0.9999999999994950,
                   0.9999906941986655, 0.9971923267772983, 0.9639452436648751,
                   0.8642827790506042, 0.7112351950296890, 0.5441424115741981,
                   0.3927307079406543, 0.2699996716773546, 0.1777181926064012,
                   0.1122496666707249, 0.0680922218447664, 0.0396818795381144,
                   0.0222179626165251, 0.0119520432391966, 0.0061774306344441,
                   0.0030676213475797, 0.0014636048371873, 0.0006709252557797]

        # 定义另一个包含浮点数的数据集列表
        dataset_c = [0.0000000000000000, 6.609305242245699e-53, 5.050407338670114e-13,
                     9.305801334566668e-06, 0.0028076732227017, 0.0360547563351249,
                     0.1357172209493958, 0.2887648049703110, 0.4558575884258019,
                     0.6072692920593457, 0.7300003283226455, 0.8222818073935988,
                     0.8877503333292751, 0.9319077781552336, 0.9603181204618857,
                     0.9777820373834749, 0.9880479567608034, 0.9938225693655559,
                     0.9969323786524203, 0.9985363951628127, 0.9993290747442203]

        # 将 x 和 dataset 组合成二维数组
        dataset = np.column_stack([x, dataset])
        # 使用 FuncData 类创建实例，并调用其 check 方法进行检查
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()
        # 将 x 和 dataset_c 组合成二维数组
        dataset_c = np.column_stack([x, dataset_c])
        # 使用 FuncData 类创建实例，并调用其 check 方法进行检查
        FuncData(_kolmogc, dataset_c, (0,), 1, rtol=_rtol).check()
    # 定义一个测试方法，用于测试 linspacei 函数的行为
    def test_linspacei(self):
        # 生成一个包含 21 个数的数组，从 0 到 1.0（包括），间隔均匀
        p = np.linspace(0, 1.0, 21, endpoint=True)
        
        # 定义一个数据集，包含浮点数和特定值
        dataset = [np.inf, 1.3580986393225507, 1.2238478702170823,
                   1.1379465424937751, 1.0727491749396481, 1.0191847202536859,
                   0.9730633753323726, 0.9320695842357622, 0.8947644549851197,
                   0.8601710725555463, 0.8275735551899077, 0.7964065373291559,
                   0.7661855555617682, 0.7364542888171910, 0.7067326523068980,
                   0.6764476915028201, 0.6448126061663567, 0.6105590999244391,
                   0.5711732651063401, 0.5196103791686224, 0.0000000000000000]

        # 定义另一个数据集，与上述数据集类似，但顺序稍有不同
        dataset_c = [0.0000000000000000, 0.5196103791686225, 0.5711732651063401,
                     0.6105590999244391, 0.6448126061663567, 0.6764476915028201,
                     0.7067326523068980, 0.7364542888171910, 0.7661855555617682,
                     0.7964065373291559, 0.8275735551899077, 0.8601710725555463,
                     0.8947644549851196, 0.9320695842357622, 0.9730633753323727,
                     1.0191847202536859, 1.0727491749396481, 1.1379465424937754,
                     1.2238478702170825, 1.3580986393225509, np.inf]

        # 将 p 和 dataset 的部分数据以列的形式组合成新的数组
        dataset = np.column_stack([p[1:], dataset[1:]])
        
        # 使用 FuncData 对象测试 kolmogi 函数在 dataset 上的行为
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        
        # 将 p 和 dataset_c 的部分数据以列的形式组合成新的数组
        dataset_c = np.column_stack([p[:-1], dataset_c[:-1]])
        
        # 使用 FuncData 对象测试 _kolmogci 函数在 dataset_c 上的行为
        FuncData(_kolmogci, dataset_c, (0,), 1, rtol=_rtol).check()

    # 定义另一个测试方法，用于测试 smallx 函数的行为
    def test_smallx(self):
        # 生成一个包含 13 个数的数组，指数级别递减
        epsilon = 0.1 ** np.arange(1, 14)
        
        # 定义一个包含特定浮点数的数组
        x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217,
                      0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254,
                      0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658,
                      0.19487060742])

        # 将 x 和 1 减去 epsilon 后的结果以列的形式组合成新的数组
        dataset = np.column_stack([x, 1-epsilon])
        
        # 使用 FuncData 对象测试 kolmogorov 函数在 dataset 上的行为
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    # 定义另一个测试方法，用于测试 round_trip 函数的行为
    def test_round_trip(self):
        # 定义一个内部函数 _ki_k，用于将输入值通过 kolmogi 和 kolmogorov 函数进行转换
        def _ki_k(_x):
            return kolmogi(kolmogorov(_x))

        # 定义一个内部函数 _kci_kc，用于将输入值通过 _kolmogc 和 _kolmogci 函数进行转换
        def _kci_kc(_x):
            return _kolmogci(_kolmogc(_x))

        # 生成一个包含 21 个数的数组，从 0.0 到 2.0（包括），间隔均匀
        x = np.linspace(0.0, 2.0, 21, endpoint=True)
        
        # 从 x 中排除值为 0 和大于 0.21 的元素
        x02 = x[(x == 0) | (x > 0.21)]
        
        # 将 x02 和 x02 以列的形式组合成新的数组
        dataset02 = np.column_stack([x02, x02])
        
        # 使用 FuncData 对象测试 _ki_k 函数在 dataset02 上的行为
        FuncData(_ki_k, dataset02, (0,), 1, rtol=_rtol).check()

        # 将 x 和 x 以列的形式组合成新的数组
        dataset = np.column_stack([x, x])
        
        # 使用 FuncData 对象测试 _kci_kc 函数在 dataset 上的行为
        FuncData(_kci_kc, dataset, (0,), 1, rtol=_rtol).check()
class TestKolmogi:
    # 测试处理 NaN 的情况
    def test_nan(self):
        assert_(np.isnan(kolmogi(np.nan)))

    # 测试基本功能
    def test_basic(self):
        # 定义数据集
        dataset = [(1.0, 0),
                   (0.96394524366487511, 0.5),
                   (0.9, 0.571173265106),
                   (0.5000000000000000, 0.8275735551899077),
                   (0.26999967167735456, 1),
                   (0.00067092525577969533, 2)]
        
        # 转换数据集为 NumPy 数组
        dataset = np.asarray(dataset)
        # 使用 FuncData 类检查 kolmogi 函数的行为
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()

    # 测试小概率累积分布函数
    def test_smallpcdf(self):
        epsilon = 0.5 ** np.arange(1, 55, 3)
        # kolmogi(1-p) == _kolmogci(p) 当且仅当 1-(1-p) == p，否则不一定成立
        # 使用 epsilon 使得 1-(1-epsilon) == epsilon，
        # 因此可以在同一 x 数组上使用相同的结果

        x = np.array([0.8275735551899077, 0.5345255069097583, 0.4320114038786941,
                      0.3736868442620478, 0.3345161714909591, 0.3057833329315859,
                      0.2835052890528936, 0.2655578150208676, 0.2506869966107999,
                      0.2380971058736669, 0.2272549289962079, 0.2177876361600040,
                      0.2094254686862041, 0.2019676748836232, 0.1952612948137504,
                      0.1891874239646641, 0.1836520225050326, 0.1785795904846466])

        # 使用 column_stack 将 epsilon 和 x 组合成数据集
        dataset = np.column_stack([1-epsilon, x])
        # 使用 FuncData 类检查 kolmogi 函数在数据集上的行为
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()

        # 使用 column_stack 将 epsilon 和 x 组合成数据集
        dataset = np.column_stack([epsilon, x])
        # 使用 FuncData 类检查 _kolmogci 函数在数据集上的行为
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    # 测试小概率生存函数
    def test_smallpsf(self):
        epsilon = 0.5 ** np.arange(1, 55, 3)
        # kolmogi(p) == _kolmogci(1-p) 当且仅当 1-(1-p) == p，否则不一定成立
        # 使用 epsilon 使得 1-(1-epsilon) == epsilon，
        # 因此可以在同一 x 数组上使用相同的结果

        x = np.array([0.8275735551899077, 1.3163786275161036, 1.6651092133663343,
                      1.9525136345289607, 2.2027324540033235, 2.4272929437460848,
                      2.6327688477341593, 2.8233300509220260, 3.0018183401530627,
                      3.1702735084088891, 3.3302184446307912, 3.4828258153113318,
                      3.6290214150152051, 3.7695513262825959, 3.9050272690877326,
                      4.0359582187082550, 4.1627730557884890, 4.2858371743264527])

        # 使用 column_stack 将 epsilon 和 x 组合成数据集
        dataset = np.column_stack([epsilon, x])
        # 使用 FuncData 类检查 kolmogi 函数在数据集上的行为
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()

        # 使用 column_stack 将 1-epsilon 和 x 组合成数据集
        dataset = np.column_stack([1-epsilon, x])
        # 使用 FuncData 类检查 _kolmogci 函数在数据集上的行为
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    # 测试来回计算
    def test_round_trip(self):
        # 定义内部函数，用于测试 round trip
        def _k_ki(_p):
            return kolmogorov(kolmogi(_p))

        # 生成 p 的等间距数组
        p = np.linspace(0.1, 1.0, 10, endpoint=True)
        # 使用 column_stack 将 p 和 p 组合成数据集
        dataset = np.column_stack([p, p])
        # 使用 FuncData 类检查 _k_ki 函数在数据集上的行为
        FuncData(_k_ki, dataset, (0,), 1, rtol=_rtol).check()


class TestKolmogp:
    # 测试处理 NaN 的情况
    def test_nan(self):
        assert_(np.isnan(_kolmogp(np.nan)))
    # 定义一个名为 test_basic 的测试方法，通常用于测试某个功能或模块的基本行为
    def test_basic(self):
        # 定义一个数据集，包含一系列元组，每个元组包含两个浮点数
        dataset = [(0.000000, -0.0),
                   (0.200000, -1.532420541338916e-10),
                   (0.400000, -0.1012254419260496),
                   (0.600000, -1.324123244249925),
                   (0.800000, -1.627024345636592),
                   (1.000000, -1.071948558356941),
                   (1.200000, -0.538512430720529),
                   (1.400000, -0.2222133182429472),
                   (1.600000, -0.07649302775520538),
                   (1.800000, -0.02208687346347873),
                   (2.000000, -0.005367402045629683)]
        
        # 将数据集转换为 NumPy 数组
        dataset = np.asarray(dataset)
        
        # 调用 FuncData 类的构造函数，传入参数 _kolmogp, dataset, (0,), 1, rtol=_rtol，并调用 check 方法进行检查
        FuncData(_kolmogp, dataset, (0,), 1, rtol=_rtol).check()
```