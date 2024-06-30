# `D:\src\scipysrc\scipy\scipy\stats\tests\test_multivariate.py`

```
"""
Test functions for multivariate normal distributions.

"""
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle

# 从 numpy.testing 中导入断言函数，用于进行各种数值测试断言
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal, assert_equal,
                           assert_array_less, assert_)
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 从 pytest 模块导入 raises 别名 assert_raises，用于断言是否抛出指定异常
from pytest import raises as assert_raises

# 从当前目录下的 test_continuous_basic 模块中导入 check_distribution_rvs 函数
from .test_continuous_basic import check_distribution_rvs

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 scipy.linalg 模块，用于线性代数运算
import scipy.linalg
# 从 scipy.stats._multivariate 模块中导入 _PSD, _lnB 和 multivariate_normal_frozen
from scipy.stats._multivariate import (_PSD,
                                       _lnB,
                                       multivariate_normal_frozen)
# 从 scipy.stats 模块中导入多变量分布相关函数
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
                         matrix_normal, special_ortho_group, ortho_group,
                         random_correlation, unitary_group, dirichlet,
                         beta, wishart, multinomial, invwishart, chi2,
                         invgamma, norm, uniform, ks_2samp, kstest, binom,
                         hypergeom, multivariate_t, cauchy, normaltest,
                         random_table, uniform_direction, vonmises_fisher,
                         dirichlet_multinomial, vonmises)

# 从 scipy.stats 模块中导入 _covariance 和 Covariance 类
from scipy.stats import _covariance, Covariance
# 从 scipy 模块中导入 stats 子模块
from scipy import stats

# 从 scipy.integrate 模块中导入 romb, qmc_quad 和 tplquad 函数
from scipy.integrate import romb, qmc_quad, tplquad
# 从 scipy.special 模块中导入 multigammaln 函数
from scipy.special import multigammaln

# 从当前目录下的 common_tests 模块中导入 check_random_state_property 函数
from .common_tests import check_random_state_property
# 从当前目录下的 data._mvt 模块中导入 _qsimvtv 函数
from .data._mvt import _qsimvtv

# 从 unittest.mock 模块中导入 patch 函数，用于单元测试时的模拟
from unittest.mock import patch


# 定义 assert_close 函数，用于比较两个数组是否在给定精度内相等
def assert_close(res, ref, *args, **kwargs):
    # 将 res 和 ref 转换为 NumPy 数组
    res, ref = np.asarray(res), np.asarray(ref)
    # 使用 assert_allclose 断言两个数组在指定精度内元素相等
    assert_allclose(res, ref, *args, **kwargs)
    # 使用 assert_equal 断言两个数组形状相同
    assert_equal(res.shape, ref.shape)


# 定义 TestCovariance 类，用于测试协方差相关功能
class TestCovariance:
    # 定义一个测试方法，用于验证输入的有效性
    def test_input_validation(self):

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The input `precision` must be a square, two-dimensional..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaPrecision，传入一个全为1的数组
            _covariance.CovViaPrecision(np.ones(2))

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "`precision.shape` must equal `covariance.shape`."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaPrecision，传入两个不同形状的单位矩阵
            _covariance.CovViaPrecision(np.eye(3), covariance=np.eye(2))

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The input `diagonal` must be a one-dimensional array..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaDiagonal，传入一个字符串而非数组
            _covariance.CovViaDiagonal("alpaca")

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The input `cholesky` must be a square, two-dimensional..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaCholesky，传入一个全为1的数组
            _covariance.CovViaCholesky(np.ones(2))

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The input `eigenvalues` must be a one-dimensional..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaEigendecomposition，传入一个字符串和单位矩阵
            _covariance.CovViaEigendecomposition(("alpaca", np.eye(2)))

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The input `eigenvectors` must be a square..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaEigendecomposition，传入一个全为1的数组和字符串
            _covariance.CovViaEigendecomposition((np.ones(2), "alpaca"))

        # 设定错误信息，用于匹配抛出的 ValueError 异常
        message = "The shapes of `eigenvalues` and `eigenvectors` must be..."
        # 使用 pytest.raises 检查是否抛出指定异常，并匹配特定消息
        with pytest.raises(ValueError, match=message):
            # 调用 _covariance.CovViaEigendecomposition，传入一个列表和单位矩阵
            _covariance.CovViaEigendecomposition(([1, 2, 3], np.eye(2)))

    # 定义一个字典，包含预处理函数名称和相应的 NumPy 函数对象
    _covariance_preprocessing = {"Diagonal": np.diag,
                                 "Precision": np.linalg.inv,
                                 "Cholesky": np.linalg.cholesky,
                                 "Eigendecomposition": np.linalg.eigh,
                                 "PSD": lambda x:
                                     _PSD(x, allow_singular=True)}

    # 创建一个包含所有预处理函数名称的 NumPy 数组
    _all_covariance_types = np.array(list(_covariance_preprocessing))

    # 定义一个字典，包含不同类型的协方差矩阵示例
    _matrices = {"diagonal full rank": np.diag([1, 2, 3]),
                 "general full rank": [[5, 1, 3], [1, 6, 4], [3, 4, 7]],
                 "diagonal singular": np.diag([1, 0, 3]),
                 "general singular": [[5, -1, 0], [-1, 5, 0], [0, 0, 0]]}

    # 定义一个字典，包含不同类型的协方差矩阵和其对应的预处理函数名称
    _cov_types = {"diagonal full rank": _all_covariance_types,
                  "general full rank": _all_covariance_types[1:],
                  "diagonal singular": _all_covariance_types[[0, -2, -1]],
                  "general singular": _all_covariance_types[-2:]}

    # 使用 pytest.mark.parametrize 装饰器，对 _all_covariance_types[:-1] 进行参数化测试
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types[:-1])
    # 定义一个测试方法，测试不同协方差类型的工厂方法
    def test_factories(self, cov_type_name):
        # 创建一个对角矩阵 A
        A = np.diag([1, 2, 3])
        # 创建一个向量 x
        x = [-4, 2, 5]

        # 获取 _covariance 模块中对应协方差类型的函数对象
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        # 获取 _covariance_preprocessing 中对应协方差类型的预处理函数对象
        preprocessing = self._covariance_preprocessing[cov_type_name]
        # 获取 Covariance 类中对应协方差类型的工厂方法对象
        factory = getattr(Covariance, f"from_{cov_type_name.lower()}")

        # 使用工厂方法创建一个协方差对象 res
        res = factory(preprocessing(A))
        # 使用 _covariance 模块中的对应函数创建一个参考协方差对象 ref
        ref = cov_type(preprocessing(A))
        
        # 断言 res 和 ref 的类型相同
        assert type(res) == type(ref)
        # 断言 res 对象的 whiten 方法对 x 的处理结果与 ref 对象相同
        assert_allclose(res.whiten(x), ref.whiten(x))

    # 使用 pytest.mark.parametrize 装饰器，对 _matrices 的键进行参数化测试
    @pytest.mark.parametrize("matrix_type", list(_matrices)))
    # 使用 pytest.mark.parametrize 装饰器，对 test_covariance 方法进行参数化测试，参数为 _all_covariance_types 列表中的每个元素
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    def test_covariance(self, matrix_type, cov_type_name):
        # 根据 matrix_type 和 cov_type_name 构造错误消息
        message = (f"CovVia{cov_type_name} does not support {matrix_type} "
                   "matrices")
        # 如果 cov_type_name 不在 self._cov_types[matrix_type] 中，则跳过该测试并输出错误消息
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)

        # 获取 matrix_type 对应的矩阵 A
        A = self._matrices[matrix_type]
        # 根据 cov_type_name 动态获取对应的协方差类
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        # 获取 cov_type_name 对应的预处理函数
        preprocessing = self._covariance_preprocessing[cov_type_name]

        # 创建 _PSD 类对象 psd，使用矩阵 A，允许矩阵奇异性
        psd = _PSD(A, allow_singular=True)

        # 测试属性
        # 创建 cov_type 对象，使用 preprocessing 函数处理 A
        cov_object = cov_type(preprocessing(A))
        # 断言 log_pdet 属性相等
        assert_close(cov_object.log_pdet, psd.log_pdet)
        # 断言 rank 属性相等
        assert_equal(cov_object.rank, psd.rank)
        # 断言 shape 属性与 A 的形状相等
        assert_equal(cov_object.shape, np.asarray(A).shape)
        # 断言 covariance 属性与 A 数组相等
        assert_close(cov_object.covariance, np.asarray(A))

        # 测试 whitening/coloring 1D x
        rng = np.random.default_rng(5292808890472453840)
        x = rng.random(size=3)
        res = cov_object.whiten(x)
        ref = x @ psd.U
        # 在一般情况下 res != ref，但满足 res @ res == ref @ ref
        assert_close(res @ res, ref @ ref)
        # 如果 cov_object 有 "_colorize" 方法，并且 matrix_type 不包含 "singular"，则进行以下断言
        if hasattr(cov_object, "_colorize") and "singular" not in matrix_type:
            # 断言 colorize 方法的结果与 x 近似相等
            assert_close(cov_object.colorize(res), x)

        # 测试 whitening/coloring 3D x
        x = rng.random(size=(2, 4, 3))
        res = cov_object.whiten(x)
        ref = x @ psd.U
        # 断言 (res 的平方和的轴和) 与 (ref 的平方和的轴和) 近似相等
        assert_close((res**2).sum(axis=-1), (ref**2).sum(axis=-1))
        # 如果 cov_object 有 "_colorize" 方法，并且 matrix_type 不包含 "singular"，则进行以下断言
        if hasattr(cov_object, "_colorize") and "singular" not in matrix_type:
            # 断言 colorize 方法的结果与 x 近似相等
            assert_close(cov_object.colorize(res), x)

        # gh-19197 报告多变量正态分布的 rvs 方法在使用 from_eigenvalues 产生奇异协方差对象时生成错误结果。
        # 这是由于 colorize 方法在处理奇异协方差矩阵时存在问题。检查这个边界情况，前面的测试已跳过。
        if hasattr(cov_object, "_colorize"):
            # 使用单位矩阵进行 colorize 方法测试
            res = cov_object.colorize(np.eye(len(A)))
            # 断言 res 的转置与 A 的近似相等
            assert_close(res.T @ res, A)

    # 使用 pytest.mark.parametrize 装饰器，对 test_covariance 方法进行参数化测试，参数为 matrix_type 列表中的每个元素和 _all_covariance_types 列表中的每个元素
    @pytest.mark.parametrize("size", [None, tuple(), 1, (2, 4, 3)])
    @pytest.mark.parametrize("matrix_type", list(_matrices))
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    # 定义一个测试方法，用于测试多变量正态分布的协方差类型
    def test_mvn_with_covariance(self, size, matrix_type, cov_type_name):
        # 生成错误消息，指出该协方差类型不支持指定的矩阵类型
        message = (f"CovVia{cov_type_name} does not support {matrix_type} "
                   "matrices")
        # 如果指定的协方差类型不在该矩阵类型支持的列表中，则跳过测试，并输出错误消息
        if cov_type_name not in self._cov_types[matrix_type]:
            pytest.skip(message)

        # 获取指定矩阵类型对应的矩阵 A
        A = self._matrices[matrix_type]
        # 根据字符串构造对应的协方差类型类对象
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        # 获取协方差预处理函数
        preprocessing = self._covariance_preprocessing[cov_type_name]

        # 设置多变量正态分布的均值向量
        mean = [0.1, 0.2, 0.3]
        # 使用预处理后的矩阵 A 构造协方差对象
        cov_object = cov_type(preprocessing(A))
        # 多变量正态分布对象
        mvn = multivariate_normal
        
        # 使用指定均值和协方差矩阵 A 构造多变量正态分布 dist0
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        # 使用指定均值和协方差对象 cov_object 构造多变量正态分布 dist1
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)

        # 使用指定种子生成器创建随机数生成器对象 rng
        rng = np.random.default_rng(5292808890472453840)
        # 生成符合多变量正态分布的随机样本 x
        x = rng.multivariate_normal(mean, A, size=size)
        
        # 使用相同的种子生成器创建随机数生成器对象 rng
        rng = np.random.default_rng(5292808890472453840)
        # 生成符合多变量正态分布的随机样本 x1，使用协方差对象 cov_object
        x1 = mvn.rvs(mean, cov_object, size=size, random_state=rng)
        
        # 使用相同的种子生成器创建随机数生成器对象 rng
        rng = np.random.default_rng(5292808890472453840)
        # 生成符合多变量正态分布的随机样本 x2，使用协方差对象 cov_object
        x2 = mvn(mean, cov_object, seed=rng).rvs(size=size)
        
        # 如果协方差对象是 _covariance.CovViaPSD 类的实例
        if isinstance(cov_object, _covariance.CovViaPSD):
            # 断言 x1 与 np.squeeze(x) 在数值上相近，用于向后兼容性
            assert_close(x1, np.squeeze(x))
            # 断言 x2 与 np.squeeze(x) 在数值上相近
            assert_close(x2, np.squeeze(x))
        else:
            # 断言 x1 的形状与 x 相同
            assert_equal(x1.shape, x.shape)
            # 断言 x2 的形状与 x 相同
            assert_equal(x2.shape, x.shape)
            # 断言 x2 与 x1 在数值上相近
            assert_close(x2, x1)

        # 断言 mvn.pdf 计算出的概率密度函数值与 dist0.pdf 计算出的值在数值上相近
        assert_close(mvn.pdf(x, mean, cov_object), dist0.pdf(x))
        # 断言 dist1.pdf 计算出的概率密度函数值与 dist0.pdf 计算出的值在数值上相近
        assert_close(dist1.pdf(x), dist0.pdf(x))
        # 断言 mvn.logpdf 计算出的对数概率密度函数值与 dist0.logpdf 计算出的值在数值上相近
        assert_close(mvn.logpdf(x, mean, cov_object), dist0.logpdf(x))
        # 断言 dist1.logpdf 计算出的对数概率密度函数值与 dist0.logpdf 计算出的值在数值上相近
        assert_close(dist1.logpdf(x), dist0.logpdf(x))
        # 断言 mvn.entropy 计算出的熵与 dist0.entropy 计算出的值在数值上相近
        assert_close(mvn.entropy(mean, cov_object), dist0.entropy())
        # 断言 dist1.entropy 计算出的熵与 dist0.entropy 计算出的值在数值上相近
        assert_close(dist1.entropy(), dist0.entropy())

    # 使用 pytest 的参数化装饰器，对 size 参数进行参数化测试
    @pytest.mark.parametrize("size", [tuple(), (2, 4, 3)])
    # 使用 pytest 的参数化装饰器，对 cov_type_name 参数进行参数化测试，参数列表为 _all_covariance_types
    @pytest.mark.parametrize("cov_type_name", _all_covariance_types)
    # 测试多元正态分布的累积分布函数（CDF）与对角协方差矩阵的类型
    def test_mvn_with_covariance_cdf(self, size, cov_type_name):
        # 由于所有矩阵类型的运行速度较慢，并且 _mvn.mvnun 已经进行了计算，
        # 因此这里单独拆分测试。Covariance 只需提供 `covariance` 属性即可。
        
        # 选择矩阵类型为对角全秩
        matrix_type = "diagonal full rank"
        # 获取特定协方差类型的类
        A = self._matrices[matrix_type]
        cov_type = getattr(_covariance, f"CovVia{cov_type_name}")
        # 获取协方差类型的预处理方法
        preprocessing = self._covariance_preprocessing[cov_type_name]

        mean = [0.1, 0.2, 0.3]
        # 使用预处理后的 A 创建 cov_object
        cov_object = cov_type(preprocessing(A))
        mvn = multivariate_normal
        
        # 创建两个多元正态分布对象，一个使用均值和 A，另一个使用均值和 cov_object
        dist0 = multivariate_normal(mean, A, allow_singular=True)
        dist1 = multivariate_normal(mean, cov_object, allow_singular=True)

        # 生成服从多元正态分布的随机样本 x
        rng = np.random.default_rng(5292808890472453840)
        x = rng.multivariate_normal(mean, A, size=size)

        # 断言两种分布的 CDF 相近
        assert_close(mvn.cdf(x, mean, cov_object), dist0.cdf(x))
        assert_close(dist1.cdf(x), dist0.cdf(x))
        
        # 断言两种分布的 logCDF 相近
        assert_close(mvn.logcdf(x, mean, cov_object), dist0.logcdf(x))
        assert_close(dist1.logcdf(x), dist0.logcdf(x))

    # 测试协方差实例化
    def test_covariance_instantiation(self):
        message = "The `Covariance` class cannot be instantiated directly."
        # 使用 pytest 检查是否抛出预期的 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=message):
            Covariance()

    # 忽略运行时警告，矩阵不是半正定的情况
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_gh9942(self):
        # 原本在 `multivariate_normal_frozen` 的 `rvs` 方法中存在错误，
        # 导致所有协方差对象都被处理为 `_CovViaPSD`。确保问题已解决。
        
        # 创建一个对角矩阵 A
        A = np.diag([1, 2, -1e-8])
        n = A.shape[0]
        mean = np.zeros(n)

        # 如果输入矩阵被处理为 `_CovViaPSD`，则会抛出 ValueError
        with pytest.raises(ValueError, match="The input matrix must be..."):
            multivariate_normal(mean, A).rvs()

        # 如果以 `CovViaEigendecomposition` 形式提供，则不会抛出错误
        seed = 3562050283508273023
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        # 从特征分解创建 Covariance 对象
        cov = Covariance.from_eigendecomposition(np.linalg.eigh(A))
        # 创建多元正态分布对象 rv
        rv = multivariate_normal(mean, cov)
        # 生成随机样本
        res = rv.rvs(random_state=rng1)
        # 使用相同的随机种子生成参考随机样本
        ref = multivariate_normal.rvs(mean, cov, random_state=rng2)
        # 断言生成的随机样本与参考随机样本相等
        assert_equal(res, ref)
    # 定义一个测试方法，用于验证解决 GitHub issue 19197 报告的问题：
    # 当使用 `from_eigenvalues` 生成一个奇异的协方差对象时，多元正态分布 `rvs` 方法产生错误的结果。
    # 检查特定问题是否已解决；更一般的测试包含在 `test_covariance` 中。
    def test_gh19197(self):
        # 创建一个均值为全1的2维数组
        mean = np.ones(2)
        # 使用 `from_eigendecomposition` 方法创建一个协方差对象 `cov`，
        # 传入一个零数组和单位矩阵作为参数
        cov = Covariance.from_eigendecomposition((np.zeros(2), np.eye(2)))
        # 使用给定的均值和协方差对象 `cov` 创建一个多元正态分布对象 `dist`
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        # 从多元正态分布中生成一个样本，大小为 `None` 表示生成一个单个样本
        rvs = dist.rvs(size=None)
        # 断言生成的样本 `rvs` 的值等于均值 `mean`
        assert_equal(rvs, mean)

        # 使用 `from_eigendecomposition` 方法创建另一个协方差对象 `cov`，
        # 传入一个具有特定值的数组和一个特定的2x2矩阵
        cov = scipy.stats.Covariance.from_eigendecomposition(
            (np.array([1., 0.]), np.array([[1., 0.], [0., 400.]])))
        # 使用相同的均值 `mean` 和新的协方差对象 `cov` 创建另一个多元正态分布对象 `dist`
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        # 从新的多元正态分布中生成一个样本，大小为 `None` 表示生成一个单个样本
        rvs = dist.rvs(size=None)
        # 断言生成的样本 `rvs` 的第一个值不等于均值 `mean` 的第一个值
        assert rvs[0] != mean[0]
        # 断言生成的样本 `rvs` 的第二个值等于均值 `mean` 的第二个值
        assert rvs[1] == mean[1]
# 生成具有给定维度 `dim` 和特征值 `evals` 的随机协方差矩阵，使用提供的生成器 `rng`。
# 如果 `singular` 为 True，则随机将一些特征值设为零。
def _random_covariance(dim, evals, rng, singular=False):
    # 生成随机矩阵 A，维度为 (dim, dim)
    A = rng.random((dim, dim))
    # 计算 A 的转置与自身的乘积，得到对称正定矩阵 A
    A = A @ A.T
    # 计算 A 的特征值和特征向量
    _, v = np.linalg.eigh(A)
    # 如果 singular 为 True，则随机选择一些特征值设为零
    if singular:
        zero_eigs = rng.normal(size=dim) > 0
        evals[zero_eigs] = 0
    # 计算协方差矩阵，使用特征向量和特征值构造
    cov = v @ np.diag(evals) @ v.T
    return cov


# 生成一个随机的正交矩阵，维度为 n x n
def _sample_orthonormal_matrix(n):
    # 生成一个随机矩阵 M，维度为 (n, n)
    M = np.random.randn(n, n)
    # 使用 scipy 的 SVD 分解获取 M 的正交基 U
    u, s, v = scipy.linalg.svd(M)
    return u


class TestMultivariateNormal:
    # 测试输入形状是否正确
    def test_input_shape(self):
        # 设置均值向量 mu 和单位协方差矩阵 cov
        mu = np.arange(3)
        cov = np.identity(2)
        # 测试对于不同维度的输入是否抛出 ValueError 异常
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.pdf, (0, 1, 2), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1), mu, cov)
        assert_raises(ValueError, multivariate_normal.cdf, (0, 1, 2), mu, cov)

    # 测试标量值的情况
    def test_scalar_values(self):
        np.random.seed(1234)

        # 当输入标量数据时，pdf 应返回标量
        x, mean, cov = 1.5, 1.7, 2.5
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)

        # 当输入单个向量时，pdf 应返回标量
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))  # 对角线上的值用于协方差矩阵
        pdf = multivariate_normal.pdf(x, mean, cov)
        assert_equal(pdf.ndim, 0)

        # 当输入标量数据时，cdf 应返回标量
        x, mean, cov = 1.5, 1.7, 2.5
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

        # 当输入单个向量时，cdf 应返回标量
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))  # 对角线上的值用于协方差矩阵
        cdf = multivariate_normal.cdf(x, mean, cov)
        assert_equal(cdf.ndim, 0)

    # 测试 logpdf 方法
    def test_logpdf(self):
        # 检查 logpdf 是否正确返回 pdf 的对数值
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        d1 = multivariate_normal.logpdf(x, mean, cov)
        d2 = multivariate_normal.pdf(x, mean, cov)
        assert_allclose(d1, np.log(d2))

    # 测试 logpdf 方法默认参数的情况
    def test_logpdf_default_values(self):
        # 检查 logpdf 是否正确返回 pdf 的对数值
        # 使用默认参数 Mean=None 和 cov=1
        np.random.seed(1234)
        x = np.random.randn(5)
        d1 = multivariate_normal.logpdf(x)
        d2 = multivariate_normal.pdf(x)
        # 检查是否使用了默认值
        d3 = multivariate_normal.logpdf(x, None, 1)
        d4 = multivariate_normal.pdf(x, None, 1)
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))
    # 测试 logcdf 方法
    def test_logcdf(self):
        # 检查 logcdf 是否正确计算
        np.random.seed(1234)
        # 生成随机数据
        x = np.random.randn(5)
        # 随机生成均值和方差
        mean = np.random.randn(5)
        cov = np.abs(np.random.randn(5))
        # 计算 logcdf
        d1 = multivariate_normal.logcdf(x, mean, cov)
        # 计算 cdf
        d2 = multivariate_normal.cdf(x, mean, cov)
        # 断言 logcdf 和 log(cdf) 的接近程度
        assert_allclose(d1, np.log(d2))

    # 测试 logcdf 方法的默认参数
    def test_logcdf_default_values(self):
        # 检查 logcdf 在默认参数 Mean=None 和 cov=1 的情况下是否正确计算
        np.random.seed(1234)
        # 生成随机数据
        x = np.random.randn(5)
        # 计算 logcdf
        d1 = multivariate_normal.logcdf(x)
        # 计算 cdf
        d2 = multivariate_normal.cdf(x)
        # 检查默认参数是否被使用
        d3 = multivariate_normal.logcdf(x, None, 1)
        d4 = multivariate_normal.cdf(x, None, 1)
        # 断言 logcdf 和 log(cdf) 的接近程度
        assert_allclose(d1, np.log(d2))
        assert_allclose(d3, np.log(d4))

    # 测试 rank 方法
    def test_rank(self):
        # 检查是否正确检测 rank
        np.random.seed(1234)
        n = 4
        # 随机生成均值
        mean = np.random.randn(n)
        # 循环检查不同的预期 rank
        for expected_rank in range(1, n + 1):
            # 生成随机数据并计算协方差矩阵
            s = np.random.randn(n, expected_rank)
            cov = np.dot(s, s.T)
            # 创建多元正态分布对象
            distn = multivariate_normal(mean, cov, allow_singular=True)
            # 断言分布对象的 rank 是否与预期 rank 相等
            assert_equal(distn.cov_object.rank, expected_rank)
    # 定义一个测试函数，用于测试退化分布的情况
    def test_degenerate_distributions(self):

        # 循环遍历不同维度的情况
        for n in range(1, 5):
            # 生成一个大小为 n 的随机向量 z
            z = np.random.randn(n)
            # 再次循环，用于生成小型协方差矩阵
            for k in range(1, n):
                # 生成一个 k x k 大小的随机矩阵 s
                s = np.random.randn(k, k)
                # 计算 s 和其转置的乘积，得到小型协方差矩阵 cov_kk
                cov_kk = np.dot(s, s.T)

                # 创建一个大小为 n x n 的零矩阵 cov_nn
                cov_nn = np.zeros((n, n))
                # 将小型协方差矩阵 cov_kk 嵌入到 cov_nn 中
                cov_nn[:k, :k] = cov_kk

                # 创建一个大小为 n 的零向量 x
                x = np.zeros(n)
                # 将向量 z 的前 k 个元素嵌入到向量 x 中
                x[:k] = z[:k]

                # 生成一个大小为 n x n 的正交矩阵 u
                u = _sample_orthonormal_matrix(n)
                # 计算旋转后的大型低秩矩阵的协方差 cov_rr
                cov_rr = np.dot(u, np.dot(cov_nn, u.T))
                # 计算旋转后的向量 y
                y = np.dot(u, x)

                # 创建一个 k 维多元正态分布对象 distn_kk，其协方差矩阵为 cov_kk
                distn_kk = multivariate_normal(np.zeros(k), cov_kk,
                                               allow_singular=True)
                # 创建一个 n 维多元正态分布对象 distn_nn，其协方差矩阵为 cov_nn
                distn_nn = multivariate_normal(np.zeros(n), cov_nn,
                                               allow_singular=True)
                # 创建一个 n 维多元正态分布对象 distn_rr，其协方差矩阵为 cov_rr
                distn_rr = multivariate_normal(np.zeros(n), cov_rr,
                                               allow_singular=True)
                
                # 断言 distn_kk 的秩为 k
                assert_equal(distn_kk.cov_object.rank, k)
                # 断言 distn_nn 的秩为 k
                assert_equal(distn_nn.cov_object.rank, k)
                # 断言 distn_rr 的秩为 k
                assert_equal(distn_rr.cov_object.rank, k)
                
                # 计算 distn_kk 在 x 的前 k 个元素上的概率密度函数值 pdf_kk
                pdf_kk = distn_kk.pdf(x[:k])
                # 计算 distn_nn 在整个向量 x 上的概率密度函数值 pdf_nn
                pdf_nn = distn_nn.pdf(x)
                # 计算 distn_rr 在向量 y 上的概率密度函数值 pdf_rr
                pdf_rr = distn_rr.pdf(y)
                # 断言 pdf_kk 和 pdf_nn 的值近似相等
                assert_allclose(pdf_kk, pdf_nn)
                # 断言 pdf_kk 和 pdf_rr 的值近似相等
                assert_allclose(pdf_kk, pdf_rr)
                
                # 计算 distn_kk 在 x 的前 k 个元素上的对数概率密度函数值 logpdf_kk
                logpdf_kk = distn_kk.logpdf(x[:k])
                // 计算 distn_nn 在整个向量 x 上的对数概率密度函数值 logpdf_nn
    def test_degenerate_array(self):
        # 测试从退化多变量正态分布生成随机变量数组，并且这些样本的概率密度函数非零
        # （即分布的样本位于子空间上）
        k = 10
        for n in range(2, 6):
            for r in range(1, n):
                mn = np.zeros(n)
                # 生成一个 n × r 的正交矩阵 u
                u = _sample_orthonormal_matrix(n)[:, :r]
                # 计算投影矩阵 vr = u * u^T
                vr = np.dot(u, u.T)
                # 从多元正态分布中抽取 k 个样本
                X = multivariate_normal.rvs(mean=mn, cov=vr, size=k)

                # 计算概率密度函数值
                pdf = multivariate_normal.pdf(X, mean=mn, cov=vr,
                                              allow_singular=True)
                # 断言 pdf 数组大小为 k
                assert_equal(pdf.size, k)
                # 断言所有概率密度函数值大于 0
                assert np.all(pdf > 0.0)

                # 计算对数概率密度函数值
                logpdf = multivariate_normal.logpdf(X, mean=mn, cov=vr,
                                                    allow_singular=True)
                # 断言 logpdf 数组大小为 k
                assert_equal(logpdf.size, k)
                # 断言所有对数概率密度函数值大于负无穷
                assert np.all(logpdf > -np.inf)

    def test_large_pseudo_determinant(self):
        # 检查大伪行列式是否被适当处理

        # 构造一个奇异对角协方差矩阵
        # 其伪行列式溢出双精度范围
        large_total_log = 1000.0
        npos = 100
        nzero = 2
        large_entry = np.exp(large_total_log / npos)
        n = npos + nzero
        cov = np.zeros((n, n), dtype=float)
        np.fill_diagonal(cov, large_entry)
        cov[-nzero:, -nzero:] = 0

        # 检查一些行列式值
        assert_equal(scipy.linalg.det(cov), 0)
        assert_equal(scipy.linalg.det(cov[:npos, :npos]), np.inf)
        assert_allclose(np.linalg.slogdet(cov[:npos, :npos]),
                        (1, large_total_log))

        # 检查伪行列式值
        psd = _PSD(cov)
        assert_allclose(psd.log_pdet, large_total_log)

    def test_broadcasting(self):
        np.random.seed(1234)
        n = 4

        # 构造一个随机协方差矩阵
        data = np.random.randn(n, n)
        cov = np.dot(data, data.T)
        mean = np.random.randn(n)

        # 构造一个 ndarray，它可以被解释为一个 2x3 的数组，其元素是随机数据向量
        X = np.random.randn(2, 3, n)

        # 检查可以同时评估多个数据点
        desired_pdf = multivariate_normal.pdf(X, mean, cov)
        desired_cdf = multivariate_normal.cdf(X, mean, cov)
        for i in range(2):
            for j in range(3):
                actual = multivariate_normal.pdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_pdf[i,j])
                # 对于累积分布函数也重复相同的检查
                actual = multivariate_normal.cdf(X[i, j], mean, cov)
                assert_allclose(actual, desired_cdf[i,j], rtol=1e-3)
    def test_normal_1D(self):
        # 1D 正态分布变量的概率密度函数应与 scipy.stats.distributions 中的标准正态分布一致
        x = np.linspace(0, 2, 10)  # 在 [0, 2] 区间均匀分布的 10 个点
        mean, cov = 1.2, 0.9  # 正态分布的均值和方差
        scale = cov**0.5  # 标准差
        d1 = norm.pdf(x, mean, scale)  # 计算正态分布的概率密度函数
        d2 = multivariate_normal.pdf(x, mean, cov)  # 计算多元正态分布的概率密度函数
        assert_allclose(d1, d2)  # 断言两个概率密度函数的值应接近
        # 累积分布函数也应该满足相同的条件
        d1 = norm.cdf(x, mean, scale)  # 计算正态分布的累积分布函数
        d2 = multivariate_normal.cdf(x, mean, cov)  # 计算多元正态分布的累积分布函数
        assert_allclose(d1, d2)  # 断言两个累积分布函数的值应接近

    def test_marginalization(self):
        # 对二维高斯分布的一个变量进行积分应该得到一个一维高斯分布
        mean = np.array([2.5, 3.5])  # 高斯分布的均值向量
        cov = np.array([[.5, 0.2], [0.2, .6]])  # 高斯分布的协方差矩阵
        n = 2 ** 8 + 1  # 样本数
        delta = 6 / (n - 1)  # 网格间距

        v = np.linspace(0, 6, n)  # 在 [0, 6] 区间均匀分布的 n 个点
        xv, yv = np.meshgrid(v, v)  # 创建网格
        pos = np.empty((n, n, 2))  # 创建空数组用于存放位置信息
        pos[:, :, 0] = xv  # x 坐标
        pos[:, :, 1] = yv  # y 坐标
        pdf = multivariate_normal.pdf(pos, mean, cov)  # 计算多元正态分布的概率密度函数

        # 沿着 x 和 y 轴进行边际化
        margin_x = romb(pdf, delta, axis=0)  # 沿着 x 轴边际化
        margin_y = romb(pdf, delta, axis=1)  # 沿着 y 轴边际化

        # 与标准正态分布进行比较
        gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)  # 计算 x 边际化后的高斯分布
        gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)  # 计算 y 边际化后的高斯分布
        assert_allclose(margin_x, gauss_x, rtol=1e-2, atol=1e-2)  # 断言 x 边际化后的结果与高斯分布接近
        assert_allclose(margin_y, gauss_y, rtol=1e-2, atol=1e-2)  # 断言 y 边际化后的结果与高斯分布接近

    def test_frozen(self):
        # 冻结的分布应与常规分布一致
        np.random.seed(1234)  # 设置随机种子
        x = np.random.randn(5)  # 生成随机数
        mean = np.random.randn(5)  # 随机均值
        cov = np.abs(np.random.randn(5))  # 随机协方差的绝对值
        norm_frozen = multivariate_normal(mean, cov)  # 创建冻结的多元正态分布对象
        assert_allclose(norm_frozen.pdf(x), multivariate_normal.pdf(x, mean, cov))  # 断言冻结的概率密度函数与常规计算的结果接近
        assert_allclose(norm_frozen.logpdf(x),
                        multivariate_normal.logpdf(x, mean, cov))  # 断言冻结的对数概率密度函数与常规计算的结果接近
        assert_allclose(norm_frozen.cdf(x), multivariate_normal.cdf(x, mean, cov))  # 断言冻结的累积分布函数与常规计算的结果接近
        assert_allclose(norm_frozen.logcdf(x),
                        multivariate_normal.logcdf(x, mean, cov))  # 断言冻结的对数累积分布函数与常规计算的结果接近

    @pytest.mark.parametrize(
        'covariance',
        [
            np.eye(2),  # 2x2 单位矩阵
            Covariance.from_diagonal([1, 1]),  # 对角线元素为 [1, 1] 的协方差矩阵
        ]
    )
    def test_frozen_multivariate_normal_exposes_attributes(self, covariance):
        mean = np.ones((2,))  # 高斯分布的均值向量
        cov_should_be = np.eye(2)  # 期望的协方差矩阵
        norm_frozen = multivariate_normal(mean, covariance)  # 创建冻结的多元正态分布对象
        assert np.allclose(norm_frozen.mean, mean)  # 断言冻结分布的均值与给定的均值接近
        assert np.allclose(norm_frozen.cov, cov_should_be)  # 断言冻结分布的协方差矩阵与期望的协方差矩阵接近
    # 定义测试方法，验证伪逆和伪行列式在截断值上的一致性
    def test_pseudodet_pinv(self):
        # 创建一个随机的协方差矩阵，包含大和小的特征值
        np.random.seed(1234)
        n = 7
        x = np.random.randn(n, n)
        cov = np.dot(x, x.T)
        # 计算协方差矩阵的特征值和特征向量
        s, u = scipy.linalg.eigh(cov)
        # 设置所有特征值为0.5，但将第一个特征值设为1.0，最后一个设为1e-7
        s = np.full(n, 0.5)
        s[0] = 1.0
        s[-1] = 1e-7
        # 重新生成协方差矩阵，确保特征值变化
        cov = np.dot(u, np.dot(np.diag(s), u.T))

        # 设置截断条件，使得最小特征值低于截断值
        cond = 1e-5
        # 创建一个 _PSD 对象，传入协方差矩阵和截断条件
        psd = _PSD(cov, cond=cond)
        # 创建一个 _PSD 对象，传入伪逆的协方差矩阵和截断条件
        psd_pinv = _PSD(psd.pinv, cond=cond)

        # 验证伪行列式的对数与除了最小特征值外所有特征值的对数和一致
        assert_allclose(psd.log_pdet, np.sum(np.log(s[:-1])))
        # 验证伪逆的伪行列式与原矩阵的伪行列式的负数相一致
        assert_allclose(-psd.log_pdet, psd_pinv.log_pdet)

    # 测试异常情况：非方阵协方差矩阵
    def test_exception_nonsquare_cov(self):
        cov = [[1, 2, 3], [4, 5, 6]]
        # 断言应该抛出 ValueError 异常
        assert_raises(ValueError, _PSD, cov)

    # 测试异常情况：协方差矩阵包含非有限数值（NaN）
    def test_exception_nonfinite_cov(self):
        cov_nan = [[1, 0], [0, np.nan]]
        # 断言应该抛出 ValueError 异常
        assert_raises(ValueError, _PSD, cov_nan)
        cov_inf = [[1, 0], [0, np.inf]]
        # 断言应该抛出 ValueError 异常
        assert_raises(ValueError, _PSD, cov_inf)

    # 测试异常情况：非正定协方差矩阵
    def test_exception_non_psd_cov(self):
        cov = [[1, 0], [0, -1]]
        # 断言应该抛出 ValueError 异常
        assert_raises(ValueError, _PSD, cov)

    # 测试异常情况：奇异协方差矩阵
    def test_exception_singular_cov(self):
        np.random.seed(1234)
        x = np.random.randn(5)
        mean = np.random.randn(5)
        cov = np.ones((5, 5))
        e = np.linalg.LinAlgError
        # 断言应该抛出 LinAlgError 异常
        assert_raises(e, multivariate_normal, mean, cov)
        assert_raises(e, multivariate_normal.pdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logpdf, x, mean, cov)
        assert_raises(e, multivariate_normal.cdf, x, mean, cov)
        assert_raises(e, multivariate_normal.logcdf, x, mean, cov)

        # 消息之前是 "singular matrix"，但这更加准确。详见 gh-15508
        cov = [[1., 0.], [1., 1.]]
        msg = "When `allow_singular is False`, the input matrix"
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            multivariate_normal(cov=cov)
    def test_R_values(self):
        # 比较多变量概率密度函数与预先在 R 版本 3.0.1 (2013-05-16) 上 Mac OS X 10.6 上计算的值

        # 下面的值是通过以下 R 脚本生成的：
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > z <- x + cos(y)
        # > mu <- c(1, 3, 2)
        # > Sigma <- matrix(c(1,2,0,2,5,0.5,0,0.5,3), 3, 3)
        # > r_pdf <- dmnorm(cbind(x,y,z), mu, Sigma)
        r_pdf = np.array([0.0002214706, 0.0013819953, 0.0049138692,
                          0.0103803050, 0.0140250800])

        x = np.linspace(0, 2, 5)
        y = 3 * x - 2
        z = x + np.cos(y)
        r = np.array([x, y, z]).T

        mean = np.array([1, 3, 2], 'd')
        cov = np.array([[1, 2, 0], [2, 5, .5], [0, .5, 3]], 'd')

        # 计算多变量正态分布的概率密度函数
        pdf = multivariate_normal.pdf(r, mean, cov)
        # 使用 assert_allclose 检查计算出的概率密度函数与预先计算的值是否在指定的容差范围内相等
        assert_allclose(pdf, r_pdf, atol=1e-10)

        # 比较多变量累积分布函数与预先在 R 版本 3.3.2 (2016-10-31) 上 Debian GNU/Linux 上计算的值

        # 下面的值是通过以下 R 脚本生成的：
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > z <- x + cos(y)
        # > mu <- c(1, 3, 2)
        # > Sigma <- matrix(c(1,2,0,2,5,0.5,0,0.5,3), 3, 3)
        # > r_cdf <- pmnorm(cbind(x,y,z), mu, Sigma)
        r_cdf = np.array([0.0017866215, 0.0267142892, 0.0857098761,
                          0.1063242573, 0.2501068509])

        # 计算多变量正态分布的累积分布函数
        cdf = multivariate_normal.cdf(r, mean, cov)
        # 使用 assert_allclose 检查计算出的累积分布函数与预先计算的值是否在指定的容差范围内相等
        assert_allclose(cdf, r_cdf, atol=2e-5)

        # 还要测试二元累积分布函数，与预先在 R 版本 3.3.2 (2016-10-31) 上 Debian GNU/Linux 上计算的值比较

        # 下面的值是通过以下 R 脚本生成的：
        # > library(mnormt)
        # > x <- seq(0, 2, length=5)
        # > y <- 3*x - 2
        # > mu <- c(1, 3)
        # > Sigma <- matrix(c(1,2,2,5), 2, 2)
        # > r_cdf2 <- pmnorm(cbind(x,y), mu, Sigma)
        r_cdf2 = np.array([0.01262147, 0.05838989, 0.18389571,
                           0.40696599, 0.66470577])

        r2 = np.array([x, y]).T

        mean2 = np.array([1, 3], 'd')
        cov2 = np.array([[1, 2], [2, 5]], 'd')

        # 计算二元多变量正态分布的累积分布函数
        cdf2 = multivariate_normal.cdf(r2, mean2, cov2)
        # 使用 assert_allclose 检查计算出的累积分布函数与预先计算的值是否在指定的容差范围内相等
        assert_allclose(cdf2, r_cdf2, atol=1e-5)

    def test_multivariate_normal_rvs_zero_covariance(self):
        mean = np.zeros(2)
        covariance = np.zeros((2, 2))
        model = multivariate_normal(mean, covariance, allow_singular=True)
        # 生成多元正态分布的随机样本，均值为零，协方差矩阵为零
        sample = model.rvs()
        # 使用 assert_equal 检查生成的随机样本是否等于预期的值 [0, 0]
        assert_equal(sample, [0, 0])
    def test_rvs_shape(self):
        # 检查 rvs 方法是否正确解析平均值和协方差，并返回正确形状的数组
        N = 300  # 样本数量
        d = 4    # 维度
        # 生成多元正态分布的样本，指定平均值为零向量，单位方差，样本数为 N
        sample = multivariate_normal.rvs(mean=np.zeros(d), cov=1, size=N)
        assert_equal(sample.shape, (N, d))  # 断言样本形状是否为 (N, d)

        # 生成多元正态分布的样本，指定平均值为 None，协方差矩阵为特定值，样本数为 N
        sample = multivariate_normal.rvs(mean=None,
                                         cov=np.array([[2, .1], [.1, 1]]),
                                         size=N)
        assert_equal(sample.shape, (N, 2))  # 断言样本形状是否为 (N, 2)

        # 创建一个多元正态分布对象，平均值为 0，单位方差
        u = multivariate_normal(mean=0, cov=1)
        # 从该分布对象中生成样本，样本数为 N
        sample = u.rvs(N)
        assert_equal(sample.shape, (N, ))  # 断言样本形状是否为 (N, )

    def test_large_sample(self):
        # 生成大样本并比较样本均值和样本协方差与给定的均值和协方差矩阵

        np.random.seed(2846)

        n = 3
        mean = np.random.randn(n)  # 生成随机的均值向量
        M = np.random.randn(n, n)   # 生成随机的 n x n 矩阵
        cov = np.dot(M, M.T)        # 计算其协方差矩阵
        size = 5000                 # 样本大小

        sample = multivariate_normal.rvs(mean, cov, size)  # 从指定多元正态分布中生成样本

        # 断言样本的协方差矩阵是否接近给定的协方差矩阵
        assert_allclose(np.cov(sample.T), cov, rtol=1e-1)
        # 断言样本均值是否接近给定的均值向量
        assert_allclose(sample.mean(0), mean, rtol=1e-1)

    def test_entropy(self):
        np.random.seed(2846)

        n = 3
        mean = np.random.randn(n)  # 生成随机的均值向量
        M = np.random.randn(n, n)   # 生成随机的 n x n 矩阵
        cov = np.dot(M, M.T)        # 计算其协方差矩阵

        rv = multivariate_normal(mean, cov)  # 创建多元正态分布对象

        # 断言冻结的分布对象的熵与熵函数计算的结果是否接近
        assert_almost_equal(rv.entropy(), multivariate_normal.entropy(mean, cov))
        # 比较熵与手动计算的表达式，涉及协方差矩阵特征值对数和的一半
        eigs = np.linalg.eig(cov)[0]  # 计算协方差矩阵的特征值
        desired = 1 / 2 * (n * (np.log(2 * np.pi) + 1) + np.sum(np.log(eigs)))
        assert_almost_equal(desired, rv.entropy())

    def test_lnB(self):
        alpha = np.array([1, 1, 1])  # 设定 alpha 向量
        desired = .5  # 对于 [1, 1, 1]，期望 e^lnB = 1/2

        assert_almost_equal(np.exp(_lnB(alpha)), desired)

    def test_cdf_with_lower_limit_arrays(self):
        # 测试在多个维度下具有下限的累积分布函数 (CDF)

        rng = np.random.default_rng(2408071309372769818)
        mean = [0, 0]              # 平均值向量
        cov = np.eye(2)            # 单位方差矩阵
        a = rng.random((4, 3, 2))*6 - 3  # 生成随机下限数组 a
        b = rng.random((4, 3, 2))*6 - 3  # 生成随机上限数组 b

        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)  # 计算带下限的 CDF

        cdf2a = multivariate_normal.cdf(b, mean, cov)        # 计算不带下限的 CDF
        cdf2b = multivariate_normal.cdf(a, mean, cov)        # 计算不带下限的 CDF
        ab1 = np.concatenate((a[..., 0:1], b[..., 1:2]), axis=-1)  # 组合下限和上限
        ab2 = np.concatenate((a[..., 1:2], b[..., 0:1]), axis=-1)  # 组合下限和上限
        cdf2ab1 = multivariate_normal.cdf(ab1, mean, cov)     # 计算组合的 CDF
        cdf2ab2 = multivariate_normal.cdf(ab2, mean, cov)     # 计算组合的 CDF
        cdf2 = cdf2a + cdf2b - cdf2ab1 - cdf2ab2              # 组合结果

        assert_allclose(cdf1, cdf2)  # 断言带下限的 CDF 与组合结果是否接近
    def test_cdf_with_lower_limit_consistency(self):
        # 检查多变量正态分布的累积分布函数的一致性
        rng = np.random.default_rng(2408071309372769818)
        # 生成均值向量，长度为3的随机数组
        mean = rng.random(3)
        # 生成3x3的随机矩阵，并确保其为协方差矩阵
        cov = rng.random((3, 3))
        cov = cov @ cov.T
        # 生成2x3的随机数组，并缩放到 [-3, 3] 的范围内
        a = rng.random((2, 3))*6 - 3
        b = rng.random((2, 3))*6 - 3

        # 使用 lower_limit=a 计算多变量正态分布的累积分布函数
        cdf1 = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        # 使用 multivariate_normal(mean, cov) 对象的方法计算累积分布函数，指定 lower_limit=a
        cdf2 = multivariate_normal(mean, cov).cdf(b, lower_limit=a)
        # 使用 lower_limit=a 计算多变量正态分布的对数累积分布函数，并取指数
        cdf3 = np.exp(multivariate_normal.logcdf(b, mean, cov, lower_limit=a))
        # 使用 multivariate_normal(mean, cov) 对象的方法计算对数累积分布函数，并取指数，指定 lower_limit=a
        cdf4 = np.exp(multivariate_normal(mean, cov).logcdf(b, lower_limit=a))

        # 检验结果的一致性，允许相对误差为1e-4
        assert_allclose(cdf2, cdf1, rtol=1e-4)
        assert_allclose(cdf3, cdf1, rtol=1e-4)
        assert_allclose(cdf4, cdf1, rtol=1e-4)

    def test_cdf_signs(self):
        # 检查当 np.any(lower > x) 时输出符号的正确性
        mean = np.zeros(3)
        cov = np.eye(3)
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
        # 当 b 中的奇数个元素小于对应的 a 时，输出应为负数
        expected_signs = np.array([1, -1, -1, 1])
        # 使用 lower_limit=a 计算多变量正态分布的累积分布函数
        cdf = multivariate_normal.cdf(b, mean, cov, lower_limit=a)
        # 检验计算结果与预期乘以符号是否一致
        assert_allclose(cdf, cdf[0]*expected_signs)

    def test_mean_cov(self):
        # 测试 Covariance 对象和均值之间的交互作用
        P = np.diag(1 / np.array([1, 2, 3]))
        cov_object = _covariance.CovViaPrecision(P)

        message = "`cov` represents a covariance matrix in 3 dimensions..."
        # 确保在特定条件下会引发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            multivariate_normal.entropy([0, 0], cov_object)

        with pytest.raises(ValueError, match=message):
            multivariate_normal([0, 0], cov_object)

        x = [0.5, 0.5, 0.5]
        # 计算给定均值和 Covariance 对象的多变量正态分布概率密度函数，并与参考值对比
        ref = multivariate_normal.pdf(x, [0, 0, 0], cov_object)
        assert_equal(multivariate_normal.pdf(x, cov=cov_object), ref)

        ref = multivariate_normal.pdf(x, [1, 1, 1], cov_object)
        assert_equal(multivariate_normal.pdf(x, 1, cov=cov_object), ref)

    def test_fit_wrong_fit_data_shape(self):
        data = [1, 3]
        error_msg = "`x` must be two-dimensional."
        # 确保在特定条件下会引发 ValueError 异常
        with pytest.raises(ValueError, match=error_msg):
            multivariate_normal.fit(data)

    @pytest.mark.parametrize('dim', (3, 5))
    def test_fit_correctness(self, dim):
        rng = np.random.default_rng(4385269356937404)
        # 生成指定维度和数量的随机数据
        x = rng.random((100, dim))
        # 使用 multivariate_normal.fit 估计均值和协方差矩阵
        mean_est, cov_est = multivariate_normal.fit(x)
        # 计算样本均值和协方差矩阵的参考值
        mean_ref, cov_ref = np.mean(x, axis=0), np.cov(x.T, ddof=0)
        # 检验估计值与参考值是否在数值上一致，允许的绝对误差和相对误差都很小
        assert_allclose(mean_est, mean_ref, atol=1e-15)
        assert_allclose(cov_est, cov_ref, rtol=1e-15)
    # 定义一个测试方法，用于测试 multivariate_normal.fit 函数在给定固定参数时的行为
    def test_fit_both_parameters_fixed(self):
        # 创建一个形状为 (2, 1)，元素全为 3 的数组作为测试数据
        data = np.full((2, 1), 3)
        # 指定固定的均值 mean_fixed
        mean_fixed = 1.
        # 将 1. 转换成至少二维数组作为固定的协方差 cov_fixed
        cov_fixed = np.atleast_2d(1.)
        # 调用 multivariate_normal.fit 函数，传入数据 data 和固定的 mean_fixed、cov_fixed
        mean, cov = multivariate_normal.fit(data, fix_mean=mean_fixed,
                                            fix_cov=cov_fixed)
        # 断言返回的均值 mean 与指定的 mean_fixed 相等
        assert_equal(mean, mean_fixed)
        # 断言返回的协方差 cov 与指定的 cov_fixed 相等
        assert_equal(cov, cov_fixed)

    # 使用 pytest 参数化装饰器标记，测试 multivariate_normal.fit 函数对 fix_mean 输入的验证
    @pytest.mark.parametrize('fix_mean', [np.zeros((2, 2)),
                                          np.zeros((3, ))])
    def test_fit_fix_mean_input_validation(self, fix_mean):
        # 准备错误消息字符串，指出 fix_mean 必须是与输入数据 x 维度相同长度的一维数组
        msg = ("`fix_mean` must be a one-dimensional array the same "
                "length as the dimensionality of the vectors `x`.")
        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证错误消息匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 multivariate_normal.fit 函数，传入单位矩阵和不合规的 fix_mean
            multivariate_normal.fit(np.eye(2), fix_mean=fix_mean)

    # 使用 pytest 参数化装饰器标记，测试 multivariate_normal.fit 函数对 fix_cov 输入维度的验证
    @pytest.mark.parametrize('fix_cov', [np.zeros((2, )),
                                         np.zeros((3, 2)),
                                         np.zeros((4, 4))])
    def test_fit_fix_cov_input_validation_dimension(self, fix_cov):
        # 准备错误消息字符串，指出 fix_cov 必须是二维方阵，且边长与输入数据 x 维度相同
        msg = ("`fix_cov` must be a two-dimensional square array "
                "of same side length as the dimensionality of the "
                "vectors `x`.")
        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证错误消息匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 multivariate_normal.fit 函数，传入单位矩阵和不合规的 fix_cov
            multivariate_normal.fit(np.eye(3), fix_cov=fix_cov)

    # 测试 multivariate_normal.fit 函数对非正半定的 fix_cov 参数的处理
    def test_fit_fix_cov_not_positive_semidefinite(self):
        # 准备错误消息字符串，指出 fix_cov 必须是对称正半定的
        error_msg = "`fix_cov` must be symmetric positive semidefinite."
        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证错误消息匹配
        with pytest.raises(ValueError, match=error_msg):
            # 准备一个非对称正半定的 fix_cov
            fix_cov = np.array([[1., 0.], [0., -1.]])
            # 调用 multivariate_normal.fit 函数，传入单位矩阵和非正半定的 fix_cov
            multivariate_normal.fit(np.eye(2), fix_cov=fix_cov)
    def test_fit_fix_mean(self):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(4385269356937404)
        # 生成一个包含三个随机数的一维数组，表示均值的位置参数
        loc = rng.random(3)
        # 生成一个3x3的随机矩阵A，并计算其协方差矩阵
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        # 从多元正态分布中抽取样本，使用给定的均值和协方差矩阵
        samples = multivariate_normal.rvs(mean=loc, cov=cov, size=100,
                                          random_state=rng)
        # 用最大似然法拟合多元正态分布，返回拟合得到的均值和协方差矩阵
        mean_free, cov_free = multivariate_normal.fit(samples)
        # 计算使用自由参数拟合得到的对数概率密度的总和
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free,
                                               cov=cov_free).sum()
        # 使用固定均值进行拟合，并断言拟合得到的均值等于预期的固定均值
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_mean=loc)
        assert_equal(mean_fix, loc)
        # 计算使用固定参数拟合得到的对数概率密度的总和
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix,
                                              cov=cov_fix).sum()
        # 断言固定参数的对数概率密度总和比自由参数的小
        assert logp_fix < logp_free
        # 对拟合得到的参数进行微小扰动，并计算扰动后的对数概率密度总和
        A = rng.random((3, 3))
        m = 1e-8 * np.dot(A, A.T)
        cov_perturbed = cov_fix + m
        logp_perturbed = (multivariate_normal.logpdf(samples,
                                                     mean=mean_fix,
                                                     cov=cov_perturbed)
                                                     ).sum()
        # 断言扰动后的对数概率密度总和比固定参数的对数概率密度总和小
        assert logp_perturbed < logp_fix


    def test_fit_fix_cov(self):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(4385269356937404)
        # 生成一个包含三个随机数的一维数组，表示均值的位置参数
        loc = rng.random(3)
        # 生成一个3x3的随机矩阵A，并计算其协方差矩阵
        A = rng.random((3, 3))
        cov = np.dot(A, A.T)
        # 从多元正态分布中抽取样本，使用给定的均值和协方差矩阵
        samples = multivariate_normal.rvs(mean=loc, cov=cov,
                                          size=100, random_state=rng)
        # 用最大似然法拟合多元正态分布，返回拟合得到的均值和协方差矩阵
        mean_free, cov_free = multivariate_normal.fit(samples)
        # 计算使用自由参数拟合得到的对数概率密度的总和
        logp_free = multivariate_normal.logpdf(samples, mean=mean_free,
                                               cov=cov_free).sum()
        # 使用固定协方差矩阵进行拟合，并断言拟合得到的均值等于样本均值，协方差矩阵等于预期的固定协方差矩阵
        mean_fix, cov_fix = multivariate_normal.fit(samples, fix_cov=cov)
        assert_equal(mean_fix, np.mean(samples, axis=0))
        assert_equal(cov_fix, cov)
        # 计算使用固定参数拟合得到的对数概率密度的总和
        logp_fix = multivariate_normal.logpdf(samples, mean=mean_fix,
                                              cov=cov_fix).sum()
        # 断言固定参数的对数概率密度总和比自由参数的小
        assert logp_fix < logp_free
        # 对拟合得到的均值参数进行微小扰动，并计算扰动后的对数概率密度总和
        mean_perturbed = mean_fix + 1e-8 * rng.random(3)
        logp_perturbed = (multivariate_normal.logpdf(samples,
                                                     mean=mean_perturbed,
                                                     cov=cov_fix)
                                                     ).sum()
        # 断言扰动后的对数概率密度总和比固定参数的对数概率密度总和小
        assert logp_perturbed < logp_fix
class TestMatrixNormal:

    def test_bad_input(self):
        # Check that bad inputs raise errors

        num_rows = 4
        num_cols = 3
        # Create a matrix M filled with 0.3, of size num_rows x num_cols
        M = np.full((num_rows,num_cols), 0.3)
        # Create a matrix U with diagonal elements 0.5 and off-diagonal elements 0.5
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)
        # Create a matrix V with diagonal elements 0.7 and off-diagonal elements 0.3
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)

        # Incorrect dimensions: Check ValueError is raised for invalid inputs
        assert_raises(ValueError, matrix_normal, np.zeros((5,4,3)))
        assert_raises(ValueError, matrix_normal, M, np.zeros(10), V)
        assert_raises(ValueError, matrix_normal, M, U, np.zeros(10))
        assert_raises(ValueError, matrix_normal, M, U, U)
        assert_raises(ValueError, matrix_normal, M, V, V)
        assert_raises(ValueError, matrix_normal, M.T, U, V)

        e = np.linalg.LinAlgError
        # Singular covariance for the rvs method of a non-frozen instance
        assert_raises(e, matrix_normal.rvs,
                      M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal.rvs,
                      M, np.ones((num_rows, num_rows)), V)
        # Singular covariance for a frozen instance
        assert_raises(e, matrix_normal, M, U, np.ones((num_cols, num_cols)))
        assert_raises(e, matrix_normal, M, np.ones((num_rows, num_rows)), V)
    def test_default_inputs(self):
        # Check that default argument handling works

        # 设置测试数据的行数和列数
        num_rows = 4
        num_cols = 3

        # 创建均值矩阵 M，元素值均为 0.3
        M = np.full((num_rows,num_cols), 0.3)

        # 创建行协方差矩阵 U，对角线元素为 0.5，其余元素为 0.5
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)

        # 创建列协方差矩阵 V，对角线元素为 0.7，其余元素为 0.3
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)

        # 创建全零矩阵 Z，大小为 num_rows x num_cols
        Z = np.zeros((num_rows, num_cols))

        # 创建全零列向量 Zr，大小为 num_rows x 1
        Zr = np.zeros((num_rows, 1))

        # 创建全零行向量 Zc，大小为 1 x num_cols
        Zc = np.zeros((1, num_cols))

        # 创建 num_rows x num_rows 的单位矩阵 Ir
        Ir = np.identity(num_rows)

        # 创建 num_cols x num_cols 的单位矩阵 Ic
        Ic = np.identity(num_cols)

        # 创建大小为 1 x 1 的单位矩阵 I1
        I1 = np.identity(1)

        # 测试不同参数组合下 matrix_normal.rvs 函数的输出形状是否符合预期
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U, colcov=V).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U).shape,
                     (num_rows, 1))
        assert_equal(matrix_normal.rvs(colcov=V).shape,
                     (1, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, colcov=V).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(mean=M, rowcov=U).shape,
                     (num_rows, num_cols))
        assert_equal(matrix_normal.rvs(rowcov=U, colcov=V).shape,
                     (num_rows, num_cols))

        # 测试不同参数组合下 matrix_normal 类的属性值是否符合预期
        assert_equal(matrix_normal(mean=M).rowcov, Ir)
        assert_equal(matrix_normal(mean=M).colcov, Ic)
        assert_equal(matrix_normal(rowcov=U).mean, Zr)
        assert_equal(matrix_normal(rowcov=U).colcov, I1)
        assert_equal(matrix_normal(colcov=V).mean, Zc)
        assert_equal(matrix_normal(colcov=V).rowcov, I1)
        assert_equal(matrix_normal(mean=M, rowcov=U).colcov, Ic)
        assert_equal(matrix_normal(mean=M, colcov=V).rowcov, Ir)
        assert_equal(matrix_normal(rowcov=U, colcov=V).mean, Z)

    def test_covariance_expansion(self):
        # Check that covariance can be specified with scalar or vector

        # 设置测试数据的行数和列数
        num_rows = 4
        num_cols = 3

        # 创建均值矩阵 M，元素值均为 0.3
        M = np.full((num_rows, num_cols), 0.3)

        # 创建行方差向量 Uv，每个元素值为 0.2
        Uv = np.full(num_rows, 0.2)

        # 创建行方差标量 Us，值为 0.2
        Us = 0.2

        # 创建列方差向量 Vv，每个元素值为 0.1
        Vv = np.full(num_cols, 0.1)

        # 创建列方差标量 Vs，值为 0.1
        Vs = 0.1

        # 创建 num_rows x num_rows 的单位矩阵 Ir
        Ir = np.identity(num_rows)

        # 创建 num_cols x num_cols 的单位矩阵 Ic
        Ic = np.identity(num_cols)

        # 测试不同参数组合下 matrix_normal 类的属性值是否符合预期
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).rowcov,
                     0.2*Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Uv, colcov=Vv).colcov,
                     0.1*Ic)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).rowcov,
                     0.2*Ir)
        assert_equal(matrix_normal(mean=M, rowcov=Us, colcov=Vs).colcov,
                     0.1*Ic)
    # 定义一个测试方法，用于测试 matrix_normal 分布的特性，特别是与多元正态分布的一致性
    def test_frozen_matrix_normal(self):
        # 对于每个可能的矩阵维度 (i, j) 进行测试
        for i in range(1,5):
            for j in range(1,5):
                # 创建一个 (i, j) 维度的矩阵 M，每个元素为 0.3
                M = np.full((i,j), 0.3)
                # 创建一个 (i, i) 维度的单位矩阵 U，并加上每个元素为 0.5 的矩阵
                U = 0.5 * np.identity(i) + np.full((i,i), 0.5)
                # 创建一个 (j, j) 维度的单位矩阵 V，并加上每个元素为 0.3 的矩阵
                V = 0.7 * np.identity(j) + np.full((j,j), 0.3)

                # 使用给定的 M, U, V 参数创建一个 matrix_normal 冻结分布对象
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)

                # 生成服从该分布的随机变量 rvs1 和 rvs2，应当相等
                rvs1 = frozen.rvs(random_state=1234)
                rvs2 = matrix_normal.rvs(mean=M, rowcov=U, colcov=V,
                                         random_state=1234)
                assert_equal(rvs1, rvs2)

                # 再次生成随机变量 X，用于后续的 PDF 和 logPDF 比较
                X = frozen.rvs(random_state=1234)

                # 计算生成随机变量 X 的概率密度函数值 pdf1 和 pdf2，应当相等
                pdf1 = frozen.pdf(X)
                pdf2 = matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(pdf1, pdf2)

                # 计算生成随机变量 X 的对数概率密度函数值 logpdf1 和 logpdf2，应当相等
                logpdf1 = frozen.logpdf(X)
                logpdf2 = matrix_normal.logpdf(X, mean=M, rowcov=U, colcov=V)
                assert_equal(logpdf1, logpdf2)

    # 定义另一个测试方法，验证 matrix_normal 分布的 PDF、logPDF 和熵与多元正态分布的一致性
    def test_matches_multivariate(self):
        # 检查 PDF 是否与通过向量化和视为多元正态分布得到的 PDF 一致
        for i in range(1,5):
            for j in range(1,5):
                # 创建一个 (i, j) 维度的矩阵 M，每个元素为 0.3
                M = np.full((i,j), 0.3)
                # 创建一个 (i, i) 维度的单位矩阵 U，并加上每个元素为 0.5 的矩阵
                U = 0.5 * np.identity(i) + np.full((i,i), 0.5)
                # 创建一个 (j, j) 维度的单位矩阵 V，并加上每个元素为 0.3 的矩阵
                V = 0.7 * np.identity(j) + np.full((j,j), 0.3)

                # 使用给定的 M, U, V 参数创建一个 matrix_normal 冻结分布对象
                frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
                # 生成服从该分布的随机变量 X
                X = frozen.rvs(random_state=1234)
                # 计算生成随机变量 X 的概率密度函数值 pdf1 和对数概率密度函数值 logpdf1，以及熵 entropy1
                pdf1 = frozen.pdf(X)
                logpdf1 = frozen.logpdf(X)
                entropy1 = frozen.entropy()

                # 将 X 扁平化为向量 vecX，将 M 扁平化为向量 vecM
                vecX = X.T.flatten()
                vecM = M.T.flatten()
                # 计算多元正态分布的 PDF 和 logPDF 值 pdf2 和 logpdf2，以及熵 entropy2
                cov = np.kron(V,U)
                pdf2 = multivariate_normal.pdf(vecX, mean=vecM, cov=cov)
                logpdf2 = multivariate_normal.logpdf(vecX, mean=vecM, cov=cov)
                entropy2 = multivariate_normal.entropy(mean=vecM, cov=cov)

                # 使用 assert_allclose 函数比较两种分布的 PDF、logPDF 和熵值，设置容忍度 rtol=1E-10
                assert_allclose(pdf1, pdf2, rtol=1E-10)
                assert_allclose(logpdf1, logpdf2, rtol=1E-10)
                assert_allclose(entropy1, entropy2)
    def test_array_input(self):
        # 检查输入数组是否与单独输入的结果相同
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)  # 创建一个 num_rows x num_cols 大小的数组 M，填充值为 0.3
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)  # 创建一个 num_rows x num_rows 的单位矩阵 U，加上填充值为 0.5 的数组
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)  # 创建一个 num_cols x num_cols 的单位矩阵 V，加上填充值为 0.3 的数组
        N = 10

        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)  # 创建一个 matrix_normal 分布对象 frozen，指定均值为 M，行协方差为 U，列协方差为 V
        X1 = frozen.rvs(size=N, random_state=1234)  # 从 frozen 分布中生成大小为 N 的随机样本 X1，使用种子 1234
        X2 = frozen.rvs(size=N, random_state=4321)  # 从 frozen 分布中生成大小为 N 的随机样本 X2，使用种子 4321
        X = np.concatenate((X1[np.newaxis,:,:,:],X2[np.newaxis,:,:,:]), axis=0)  # 将 X1 和 X2 沿着第一个轴进行连接，形成一个新的数组 X，维度为 (2, N, num_rows, num_cols)
        assert_equal(X.shape, (2, N, num_rows, num_cols))  # 断言 X 的形状为 (2, N, num_rows, num_cols)

        array_logpdf = frozen.logpdf(X)  # 计算 X 的对数概率密度函数值，存储在 array_logpdf 中
        assert_equal(array_logpdf.shape, (2, N))  # 断言 array_logpdf 的形状为 (2, N)
        for i in range(2):
            for j in range(N):
                separate_logpdf = matrix_normal.logpdf(X[i,j], mean=M,
                                                       rowcov=U, colcov=V)  # 计算单独样本 X[i,j] 的对数概率密度函数值，使用均值 M，行协方差 U，列协方差 V
                assert_allclose(separate_logpdf, array_logpdf[i,j], 1E-10)  # 断言单独计算的对数概率密度函数值与数组计算的值相近，精度为 1E-10

    def test_moments(self):
        # 检查样本矩是否与参数匹配
        num_rows = 4
        num_cols = 3
        M = np.full((num_rows,num_cols), 0.3)  # 创建一个 num_rows x num_cols 大小的数组 M，填充值为 0.3
        U = 0.5 * np.identity(num_rows) + np.full((num_rows, num_rows), 0.5)  # 创建一个 num_rows x num_rows 的单位矩阵 U，加上填充值为 0.5 的数组
        V = 0.7 * np.identity(num_cols) + np.full((num_cols, num_cols), 0.3)  # 创建一个 num_cols x num_cols 的单位矩阵 V，加上填充值为 0.3 的数组
        N = 1000

        frozen = matrix_normal(mean=M, rowcov=U, colcov=V)  # 创建一个 matrix_normal 分布对象 frozen，指定均值为 M，行协方差为 U，列协方差为 V
        X = frozen.rvs(size=N, random_state=1234)  # 从 frozen 分布中生成大小为 N 的随机样本 X，使用种子 1234

        sample_mean = np.mean(X,axis=0)  # 计算 X 按列的均值，存储在 sample_mean 中
        assert_allclose(sample_mean, M, atol=0.1)  # 断言 sample_mean 与 M 的值在绝对误差容忍度 0.1 内相等

        sample_colcov = np.cov(X.reshape(N*num_rows,num_cols).T)  # 计算 X 按行重塑后的协方差矩阵，存储在 sample_colcov 中
        assert_allclose(sample_colcov, V, atol=0.1)  # 断言 sample_colcov 与 V 的值在绝对误差容忍度 0.1 内相等

        sample_rowcov = np.cov(np.swapaxes(X,1,2).reshape(
                                                        N*num_cols,num_rows).T)  # 计算 X 按列和行交换后重塑的协方差矩阵，存储在 sample_rowcov 中
        assert_allclose(sample_rowcov, U, atol=0.1)  # 断言 sample_rowcov 与 U 的值在绝对误差容忍度 0.1 内相等

    def test_samples(self):
        # 回归测试以确保我们始终生成相同的随机变量流
        actual = matrix_normal.rvs(
            mean=np.array([[1, 2], [3, 4]]),  # 指定均值矩阵为 [[1, 2], [3, 4]]
            rowcov=np.array([[4, -1], [-1, 2]]),  # 指定行协方差矩阵为 [[4, -1], [-1, 2]]
            colcov=np.array([[5, 1], [1, 10]]),  # 指定列协方差矩阵为 [[5, 1], [1, 10]]
            random_state=np.random.default_rng(0),  # 指定随机数生成器的种子为默认随机数生成器的种子 0
            size=2  # 指定生成的样本数量为 2
        )
        expected = np.array(
            [[[1.56228264238181, -1.24136424071189],  # 预期结果的第一个样本的第一个随机变量
              [2.46865788392114, 6.22964440489445]],  # 预期结果的第一个样本的第二个随机变量
             [[3.86405716144353, 10.73714311429529],  # 预期结果的第二个样本的第一个随机变量
              [2.59428444080606, 5.79987854490876]]]  # 预期结果的第二个样本的第二个随机变量
        )
        assert_allclose(actual, expected)  # 断言生成的实际随机变量与预期的随机变量相近
class TestDirichlet:

    def test_frozen_dirichlet(self):
        # 设置随机种子以保证测试的可重复性
        np.random.seed(2846)

        # 随机生成一个整数 n，范围在 [1, 32)
        n = np.random.randint(1, 32)
        # 随机生成一个长度为 n 的 alpha 数组，元素均匀分布在 [10e-10, 100) 区间内
        alpha = np.random.uniform(10e-10, 100, n)

        # 使用给定的 alpha 创建一个 Dirichlet 分布对象 d
        d = dirichlet(alpha)

        # 断言 d 的方差与 dirichlet 分布的方差函数计算结果相等
        assert_equal(d.var(), dirichlet.var(alpha))
        # 断言 d 的均值与 dirichlet 分布的均值函数计算结果相等
        assert_equal(d.mean(), dirichlet.mean(alpha))
        # 断言 d 的熵与 dirichlet 分布的熵函数计算结果相等
        assert_equal(d.entropy(), dirichlet.entropy(alpha))

        # 设定测试次数为 10 次
        num_tests = 10
        for i in range(num_tests):
            # 随机生成一个长度为 n 的数组 x，元素均匀分布在 [10e-10, 100) 区间内
            x = np.random.uniform(10e-10, 100, n)
            # 将 x 规范化为概率分布，即除以其总和
            x /= np.sum(x)
            # 断言 d 对 x 的前 n-1 个元素的概率密度函数与 dirichlet 分布的计算结果相等
            assert_equal(d.pdf(x[:-1]), dirichlet.pdf(x[:-1], alpha))
            # 断言 d 对 x 的前 n-1 个元素的对数概率密度函数与 dirichlet 分布的计算结果相等
            assert_equal(d.logpdf(x[:-1]), dirichlet.logpdf(x[:-1], alpha))

    def test_numpy_rvs_shape_compatibility(self):
        # 设置随机种子以保证测试的可重复性
        np.random.seed(2846)
        # 设定 alpha 数组为 [1.0, 2.0, 3.0]
        alpha = np.array([1.0, 2.0, 3.0])
        # 使用 numpy 随机生成符合 Dirichlet 分布的样本矩阵 x，形状为 (7, 3)
        x = np.random.dirichlet(alpha, size=7)

        # 断言 x 的形状为 (7, 3)
        assert_equal(x.shape, (7, 3))
        # 断言对于 dirichlet.pdf 和 dirichlet.logpdf 函数对 x 和 alpha 的调用会引发 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

        # 对 x 的转置进行函数调用，不应该引发异常
        dirichlet.pdf(x.T, alpha)
        dirichlet.pdf(x.T[:-1], alpha)
        dirichlet.logpdf(x.T, alpha)
        dirichlet.logpdf(x.T[:-1], alpha)

    def test_alpha_with_zeros(self):
        # 设置随机种子以保证测试的可重复性
        np.random.seed(2846)
        # 设定 alpha 数组包含一个零值的情况
        alpha = [1.0, 0.0, 3.0]
        # 不要将无效的 alpha 传递给 np.random.dirichlet
        x = np.random.dirichlet(np.maximum(1e-9, alpha), size=7).T
        # 断言对于 dirichlet.pdf 和 dirichlet.logpdf 函数对 x 和 alpha 的调用会引发 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_alpha_with_negative_entries(self):
        # 设置随机种子以保证测试的可重复性
        np.random.seed(2846)
        # 设定 alpha 数组包含负值的情况
        alpha = [1.0, -2.0, 3.0]
        # 不要将无效的 alpha 传递给 np.random.dirichlet
        x = np.random.dirichlet(np.maximum(1e-9, alpha), size=7).T
        # 断言对于 dirichlet.pdf 和 dirichlet.logpdf 函数对 x 和 alpha 的调用会引发 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    def test_data_with_zeros(self):
        # 设定 alpha 和 x 数组
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([0.1, 0.0, 0.2, 0.7])
        # 对 dirichlet.pdf 和 dirichlet.logpdf 函数进行调用
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)
        # 当 alpha 全部为 1 时，进行额外的断言验证
        alpha = np.array([1.0, 1.0, 1.0, 1. deeply
    # 测试函数：检查给定参数 alpha 和 x 是否导致 Dirichlet 分布函数引发 ValueError 异常
    def test_data_too_deep_c(self):
        # 创建一个长度为 3 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1.0, 2.0, 3.0])
        # 创建一个形状为 (2, 7, 7) 的 NumPy 数组，每个元素为 1/14，作为参数 x
        x = np.full((2, 7, 7), 1 / 14)
        # 断言 dirichlet.pdf 和 dirichlet.logpdf 对参数 x 和 alpha 抛出 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    # 测试函数：检查给定参数 alpha 和 x 是否导致 Dirichlet 分布函数引发 ValueError 异常
    def test_alpha_too_deep(self):
        # 创建一个形状为 (2, 2) 的二维 NumPy 数组，作为参数 alpha
        alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        # 创建一个形状为 (2, 2, 7) 的 NumPy 数组，每个元素为 1/4，作为参数 x
        x = np.full((2, 2, 7), 1 / 4)
        # 断言 dirichlet.pdf 和 dirichlet.logpdf 对参数 x 和 alpha 抛出 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    # 测试函数：使用合适的参数 alpha 和 x 调用 dirichlet.pdf 和 dirichlet.logpdf 函数
    def test_alpha_correct_depth(self):
        # 创建一个长度为 3 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1.0, 2.0, 3.0])
        # 创建一个形状为 (3, 7) 的 NumPy 数组，每个元素为 1/3，作为参数 x
        x = np.full((3, 7), 1 / 3)
        # 调用 dirichlet.pdf 和 dirichlet.logpdf 函数，传入参数 x 和 alpha
        dirichlet.pdf(x, alpha)
        dirichlet.logpdf(x, alpha)

    # 测试函数：检查给定参数 alpha 和 x 是否导致 Dirichlet 分布函数引发 ValueError 异常
    def test_non_simplex_data(self):
        # 创建一个长度为 3 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1.0, 2.0, 3.0])
        # 创建一个形状为 (3, 7) 的 NumPy 数组，每个元素为 1/2，作为参数 x
        x = np.full((3, 7), 1 / 2)
        # 断言 dirichlet.pdf 和 dirichlet.logpdf 对参数 x 和 alpha 抛出 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    # 测试函数：检查给定参数 alpha 和 x 是否导致 Dirichlet 分布函数引发 ValueError 异常
    def test_data_vector_too_short(self):
        # 创建一个长度为 4 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        # 创建一个形状为 (2, 7) 的 NumPy 数组，每个元素为 1/2，作为参数 x
        x = np.full((2, 7), 1 / 2)
        # 断言 dirichlet.pdf 和 dirichlet.logpdf 对参数 x 和 alpha 抛出 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    # 测试函数：检查给定参数 alpha 和 x 是否导致 Dirichlet 分布函数引发 ValueError 异常
    def test_data_vector_too_long(self):
        # 创建一个长度为 4 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1.0, 2.0, 3.0, 4.0])
        # 创建一个形状为 (5, 7) 的 NumPy 数组，每个元素为 1/5，作为参数 x
        x = np.full((5, 7), 1 / 5)
        # 断言 dirichlet.pdf 和 dirichlet.logpdf 对参数 x 和 alpha 抛出 ValueError 异常
        assert_raises(ValueError, dirichlet.pdf, x, alpha)
        assert_raises(ValueError, dirichlet.logpdf, x, alpha)

    # 测试函数：验证 Dirichlet 分布的均值、方差和协方差是否与手工计算的参考值一致
    def test_mean_var_cov(self):
        # 创建一个长度为 3 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([1., 0.8, 0.2])
        # 创建一个 Dirichlet 分布对象，以 alpha 作为参数
        d = dirichlet(alpha)

        # 预期的均值、方差和协方差的值
        expected_mean = [0.5, 0.4, 0.1]
        expected_var = [1. / 12., 0.08, 0.03]
        expected_cov = [
                [ 1. / 12, -1. / 15, -1. / 60],
                [-1. / 15,  2. / 25, -1. / 75],
                [-1. / 60, -1. / 75,  3. / 100],
        ]

        # 断言 Dirichlet 对象的均值、方差和协方差与预期值几乎相等
        assert_array_almost_equal(d.mean(), expected_mean)
        assert_array_almost_equal(d.var(), expected_var)
        assert_array_almost_equal(d.cov(), expected_cov)

    # 测试函数：验证对于长度为 1 的 alpha，Dirichlet 分布的均值和方差应为标量
    def test_scalar_values(self):
        # 创建一个长度为 1 的一维 NumPy 数组，作为参数 alpha
        alpha = np.array([0.2])
        # 创建一个 Dirichlet 分布对象，以 alpha 作为参数
        d = dirichlet(alpha)

        # 断言 Dirichlet 对象的均值和方差的维度应为 0
        assert_equal(d.mean().ndim, 0)
        assert_equal(d.var().ndim, 0)

        # 断言 Dirichlet 对象对输入 [1.] 的概率密度和对数概率密度的维度应为 0
        assert_equal(d.pdf([1.]).ndim, 0)
        assert_equal(d.logpdf([1.]).ndim, 0)

    # 测试函数：验证使用 K 和 K-1 个参数调用 Dirichlet 分布的 pdf 函数返回几乎相等的结果
    def test_K_and_K_minus_1_calls_equal(self):
        # 设置随机数生成种子
        np.random.seed(2846)

        # 随机生成一个介于 1 到 32 之间的整数，作为参数 n
        n = np.random.randint(1, 32)
        # 随机生成一个长度为 n 的一维 NumPy 数组，元素取值介于 10e-10 到 100 之间，作为参数 alpha
        alpha = np.random.uniform(10e-10, 100, n)

        # 创建一个 Dirichlet 分布对象，以 alpha 作为参数
        d = dirichlet(alpha)
        # 设置测试次数为 10
        num_tests = 10
        # 执行 num_tests 次测试
        for i in range(num_tests):
            # 随机生成一个长度为 n 的一维 NumPy 数组，元素取值介于 10e-10 到 100 之间，作为参数 x
            x = np.random.uniform(10e-10, 100, n)
            x /= np.sum(x)  # 将 x 归一化为概率向量

            # 断言 Dirichlet 对象对 x 和 x[:-1] 的
    # 定义一个测试函数，用于测试多个输入向量作为矩阵的情况
    def test_multiple_entry_calls(self):
        # 设置随机数种子以保证结果可重复
        np.random.seed(2846)

        # 生成一个随机整数 n，范围在 [1, 32) 内
        n = np.random.randint(1, 32)
        # 生成一个长度为 n 的随机数组 alpha，元素取值范围在 [10e-10, 100) 内
        alpha = np.random.uniform(10e-10, 100, n)
        # 使用参数 alpha 创建一个 Dirichlet 分布对象 d
        d = dirichlet(alpha)

        # 定义测试的次数
        num_tests = 10
        # 定义每次测试中多个向量的数量
        num_multiple = 5
        # 初始化一个空的矩阵 xm，用于存储多个向量
        xm = None
        # 开始循环进行多次测试
        for i in range(num_tests):
            # 在每次测试中，生成多个长度为 n 的随机数组 x
            for m in range(num_multiple):
                x = np.random.uniform(10e-10, 100, n)
                # 将 x 向量归一化
                x /= np.sum(x)
                # 将 x 加入到矩阵 xm 中
                if xm is not None:
                    xm = np.vstack((xm, x))
                else:
                    xm = x
            # 计算矩阵 xm 的每行的密度值，返回一个行向量 rm
            rm = d.pdf(xm.T)
            # 初始化一个空数组 rs，用于存储每个单独向量的密度值
            rs = None
            # 遍历矩阵 xm 中的每个向量 xs
            for xs in xm:
                # 计算向量 xs 的密度值 r
                r = d.pdf(xs)
                # 将 r 添加到数组 rs 中
                if rs is not None:
                    rs = np.append(rs, r)
                else:
                    rs = r
            # 断言矩阵 rm 与数组 rs 的密度值近似相等
            assert_array_almost_equal(rm, rs)

    # 定义一个测试函数，用于验证二维 Dirichlet 分布是否等价于 Beta 分布
    def test_2D_dirichlet_is_beta(self):
        # 设置随机数种子以保证结果可重复
        np.random.seed(2846)

        # 生成一个长度为 2 的随机数组 alpha，元素取值范围在 [10e-10, 100) 内
        alpha = np.random.uniform(10e-10, 100, 2)
        # 使用参数 alpha 创建一个 Dirichlet 分布对象 d
        d = dirichlet(alpha)
        # 使用参数 alpha 的第一个和第二个元素创建一个 Beta 分布对象 b
        b = beta(alpha[0], alpha[1])

        # 定义测试的次数
        num_tests = 10
        # 开始循环进行多次测试
        for i in range(num_tests):
            # 生成一个长度为 2 的随机数组 x
            x = np.random.uniform(10e-10, 100, 2)
            # 将 x 向量归一化
            x /= np.sum(x)
            # 断言 Beta 分布 b 在向量 x 上的概率密度函数值与 Dirichlet 分布 d 在向量 [x] 上的值近似相等
            assert_almost_equal(b.pdf(x), d.pdf([x]))

        # 断言 Beta 分布 b 的均值与 Dirichlet 分布 d 的第一个维度均值近似相等
        assert_almost_equal(b.mean(), d.mean()[0])
        # 断言 Beta 分布 b 的方差与 Dirichlet 分布 d 的第一个维度方差近似相等
        assert_almost_equal(b.var(), d.var()[0])
# 定义一个测试函数，用于检查多变量正态分布函数在维度不匹配时是否能够正确引发 ValueError 异常，并包含信息丰富的错误消息。
def test_multivariate_normal_dimensions_mismatch():
    # 设置均值 mu 为二维数组 [0.0, 0.0]
    mu = np.array([0.0, 0.0])
    # 设置协方差矩阵 sigma 为二维数组 [[1.0]]
    sigma = np.array([[1.0]])

    # 断言调用 multivariate_normal 函数时，使用 mu 和 sigma 参数会引发 ValueError 异常
    assert_raises(ValueError, multivariate_normal, mu, sigma)

    # 进行简单的检查，确认错误消息被正确传递。由于完全匹配错误消息可能会很脆弱，因此只检查前导部分是否匹配。
    try:
        multivariate_normal(mu, sigma)
    except ValueError as e:
        msg = "Dimension mismatch"
        # 断言错误消息的前缀部分与预期的错误信息 "Dimension mismatch" 匹配
        assert_equal(str(e)[:len(msg)], msg)


class TestWishart:
    # 定义一个测试类 TestWishart
    def test_scale_dimensions(self):
        # 测试不同尺度维度的 Wishart 分布调用是否正常工作

        # 测试案例: dim=1, scale=1
        true_scale = np.array(1, ndmin=2)
        scales = [
            1,                    # 标量
            [1],                  # 可迭代对象
            np.array(1),          # 0 维数组
            np.r_[1],             # 1 维数组
            np.array(1, ndmin=2)  # 2 维数组
        ]
        for scale in scales:
            # 使用 wishart 函数创建一个 Wishart 分布对象 w
            w = wishart(1, scale)
            # 断言 w 的 scale 属性与预期的 true_scale 相等
            assert_equal(w.scale, true_scale)
            # 断言 w 的 scale 属性形状与预期的 true_scale 形状相等
            assert_equal(w.scale.shape, true_scale.shape)

        # 测试案例: dim=2, scale=[[1,0]
        #                          [0,2]
        true_scale = np.array([[1,0],
                               [0,2]])
        scales = [
            [1,2],             # 可迭代对象
            np.r_[1,2],        # 1 维数组
            np.array([[1,0],   # 2 维数组
                      [0,2]])
        ]
        for scale in scales:
            # 使用 wishart 函数创建一个 Wishart 分布对象 w
            w = wishart(2, scale)
            # 断言 w 的 scale 属性与预期的 true_scale 相等
            assert_equal(w.scale, true_scale)
            # 断言 w 的 scale 属性形状与预期的 true_scale 形状相等
            assert_equal(w.scale.shape, true_scale.shape)

        # 对于 df < dim - 1 时，调用 wishart 函数应该引发 ValueError 异常
        assert_raises(ValueError, wishart, 1, np.eye(2))

        # 但是对于 dim - 1 < df < dim 时，调用 wishart 函数不应该引发异常
        wishart(1.1, np.eye(2))  # 没有错误
        # see gh-5562

        # 对于 3 维数组作为尺度参数时，调用 wishart 函数应该引发 ValueError 异常
        scale = np.array(1, ndmin=3)
        assert_raises(ValueError, wishart, 1, scale)
    def test_quantile_dimensions(self):
        # 测试使用不同分位数维度调用 Wishart 随机变量的功能

        # 如果 dim == 1，考虑 x.shape = [1,1,1]
        X = [
            1,                      # 标量
            [1],                    # 可迭代对象
            np.array(1),            # 0 维数组
            np.r_[1],               # 1 维数组
            np.array(1, ndmin=2),   # 2 维数组
            np.array([1], ndmin=3)  # 3 维数组
        ]

        # 创建一个 Wishart 分布对象，自由度为 1，规模矩阵为单位矩阵
        w = wishart(1,1)
        # 计算密度函数值，传入一个 3 维数组
        density = w.pdf(np.array(1, ndmin=3))
        for x in X:
            # 断言每个 x 的密度函数值与 density 相等
            assert_equal(w.pdf(x), density)

        # 如果 dim == 1，考虑 x.shape = [1,1,*]
        X = [
            [1,2,3],                     # 可迭代对象
            np.r_[1,2,3],                # 1 维数组
            np.array([1,2,3], ndmin=3)   # 3 维数组
        ]

        # 创建一个新的 Wishart 分布对象，自由度为 1，规模矩阵为单位矩阵
        w = wishart(1,1)
        # 计算密度函数值，传入一个 [1,2,3] 形状的 3 维数组
        density = w.pdf(np.array([1,2,3], ndmin=3))
        for x in X:
            # 断言每个 x 的密度函数值与 density 相等
            assert_equal(w.pdf(x), density)

        # 如果 dim == 2，考虑 x.shape = [2,2,1]
        # 其中 x[:,:,*] = np.eye(1)*2
        X = [
            2,                    # 标量
            [2,2],                # 可迭代对象
            np.array(2),          # 0 维数组
            np.r_[2,2],           # 1 维数组
            np.array([[2,0],
                      [0,2]]),    # 2 维数组
            np.array([[2,0],
                      [0,2]])[:,:,np.newaxis]  # 3 维数组
        ]

        # 创建一个新的 Wishart 分布对象，自由度为 2，规模矩阵为单位矩阵的扩展形式
        w = wishart(2,np.eye(2))
        # 计算密度函数值，传入一个 [2,2,1] 形状的 3 维数组
        density = w.pdf(np.array([[2,0],
                                  [0,2]])[:,:,np.newaxis])
        for x in X:
            # 断言每个 x 的密度函数值与 density 相等
            assert_equal(w.pdf(x), density)

    def test_frozen(self):
        # 测试冻结与非冻结状态下 Wishart 分布给出相同答案的功能

        # 构造任意正定的规模矩阵
        dim = 4
        scale = np.diag(np.arange(dim)+1)
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim-1) // 2)
        scale = np.dot(scale.T, scale)

        # 构造一组正定矩阵来测试概率密度函数
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim)+(i+1)**2)
            x[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim-1) // 2)
            x = np.dot(x.T, x)
            X.append(x)
        X = np.array(X).T

        # 构造 1 维和 2 维参数集合
        parameters = [
            (10, 1, np.linspace(0.1, 10, 5)),  # 1 维情况
            (10, scale, X)
        ]

        for (df, scale, x) in parameters:
            # 创建一个冻结的 Wishart 分布对象
            w = wishart(df, scale)
            # 断言冻结与非冻结状态下的方差相等
            assert_equal(w.var(), wishart.var(df, scale))
            # 断言冻结与非冻结状态下的均值相等
            assert_equal(w.mean(), wishart.mean(df, scale))
            # 断言冻结与非冻结状态下的众数相等
            assert_equal(w.mode(), wishart.mode(df, scale))
            # 断言冻结与非冻结状态下的熵相等
            assert_equal(w.entropy(), wishart.entropy(df, scale))
            # 断言冻结与非冻结状态下的概率密度函数值相等
            assert_equal(w.pdf(x), wishart.pdf(x, df, scale))
    # 定义一个测试函数，用于测试二维 Wishart 分布的随机变量生成
    def test_wishart_2D_rvs(self):
        # 设置维度和自由度
        dim = 3
        df = 10

        # 构造一个简单的非对角正定矩阵
        scale = np.eye(dim)
        scale[0,1] = 0.5
        scale[1,0] = 0.5

        # 构造冻结的 Wishart 随机变量
        w = wishart(df, scale)

        # 从已知种子生成随机变量
        np.random.seed(248042)
        w_rvs = wishart.rvs(df, scale)
        np.random.seed(248042)
        frozen_w_rvs = w.rvs()

        # 根据 Bartlett (1933) 分解 Wishart 分布为 D A A' D' 手动计算应该的随机变量
        np.random.seed(248042)
        covariances = np.random.normal(size=3)
        variances = np.r_[
            np.random.chisquare(df),
            np.random.chisquare(df-1),
            np.random.chisquare(df-2),
        ]**0.5

        # 构造下三角矩阵 A
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances

        # Wishart 随机变量
        D = np.linalg.cholesky(scale)
        DA = D.dot(A)
        manual_w_rvs = np.dot(DA, DA.T)

        # 测试随机变量的相等性
        assert_allclose(w_rvs, manual_w_rvs)
        assert_allclose(frozen_w_rvs, manual_w_rvs)

    # 定义一个测试函数，用于测试一维 Wishart 分布是否等价于卡方分布
    def test_1D_is_chisquared(self):
        # 当标度矩阵为单位矩阵时，一维 Wishart 分布等价于卡方分布
        # 测试方差、均值、熵和概率密度函数
        # Kolgomorov-Smirnov 测试随机变量
        np.random.seed(482974)

        sn = 500
        dim = 1
        scale = np.eye(dim)

        df_range = np.arange(1, 10, 2, dtype=float)
        X = np.linspace(0.1,10,num=10)
        for df in df_range:
            w = wishart(df, scale)
            c = chi2(df)

            # 统计量
            assert_allclose(w.var(), c.var())
            assert_allclose(w.mean(), c.mean())
            assert_allclose(w.entropy(), c.entropy())

            # 概率密度函数
            assert_allclose(w.pdf(X), c.pdf(X))

            # 随机变量
            rvs = w.rvs(size=sn)
            args = (df,)
            alpha = 0.01
            check_distribution_rvs('chi2', args, alpha, rvs)
    # 定义测试函数，用于验证特定数学分布的性质
    def test_is_scaled_chisquared(self):
        # 通过随机数种子设定确定性随机种子，保证结果的可复现性
        np.random.seed(482974)

        # 设置参数
        sn = 500  # 样本数量
        df = 10   # 自由度
        dim = 4   # 维度

        # 构造任意正定矩阵
        scale = np.diag(np.arange(4)+1)  # 对角矩阵，对角元素为1至4
        scale[np.tril_indices(4, k=-1)] = np.arange(6)  # 设置下三角部分为0至5
        scale = np.dot(scale.T, scale)  # 计算得到正定矩阵

        # 使用单位向量作为 lambda
        lamda = np.ones((dim,1))  # 维度为4的全1列向量
        sigma_lamda = lamda.T.dot(scale).dot(lamda).squeeze()  # 计算 lambda' * scale * lambda

        # 生成 Wishart 分布和对应的 scaled chi-squared 分布
        w = wishart(df, sigma_lamda)
        c = chi2(df, scale=sigma_lamda)

        # 验证统计量
        assert_allclose(w.var(), c.var())  # 验证方差是否接近
        assert_allclose(w.mean(), c.mean())  # 验证均值是否接近
        assert_allclose(w.entropy(), c.entropy())  # 验证熵是否接近

        # 验证概率密度函数
        X = np.linspace(0.1,10,num=10)  # 在指定区间内生成10个均匀分布的点
        assert_allclose(w.pdf(X), c.pdf(X))  # 验证概率密度函数在给定点上的值是否接近

        # 生成随机样本
        rvs = w.rvs(size=sn)  # 生成大小为sn的随机样本
        args = (df, 0, sigma_lamda)  # chi-squared 分布的参数
        alpha = 0.01  # 显著性水平
        check_distribution_rvs('chi2', args, alpha, rvs)  # 验证生成的随机样本是否符合 chi-squared 分布
class TestMultinomial:
    # 测试 Multinomial 分布的各种方法

    def test_logpmf(self):
        # 测试 logpmf 方法
        vals1 = multinomial.logpmf((3,4), 7, (0.3, 0.7))
        assert_allclose(vals1, -1.483270127243324, rtol=1e-8)

        vals2 = multinomial.logpmf([3, 4], 0, [.3, .7])
        assert vals2 == -np.inf

        vals3 = multinomial.logpmf([0, 0], 0, [.3, .7])
        assert vals3 == 0

        vals4 = multinomial.logpmf([3, 4], 0, [-2, 3])
        assert_allclose(vals4, np.nan, rtol=1e-8)

    def test_reduces_binomial(self):
        # 测试 Multinomial 分布在二维情况下是否可以归约为二项分布
        val1 = multinomial.logpmf((3, 4), 7, (0.3, 0.7))
        val2 = binom.logpmf(3, 7, 0.3)
        assert_allclose(val1, val2, rtol=1e-8)

        val1 = multinomial.pmf((6, 8), 14, (0.1, 0.9))
        val2 = binom.pmf(6, 14, 0.1)
        assert_allclose(val1, val2, rtol=1e-8)

    def test_R(self):
        # 测试与 R 代码产生的值是否一致
        # (https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Multinom.html)
        # X <- t(as.matrix(expand.grid(0:3, 0:3))); X <- X[, colSums(X) <= 3]
        # X <- rbind(X, 3:3 - colSums(X)); dimnames(X) <- list(letters[1:3], NULL)
        # X
        # apply(X, 2, function(x) dmultinom(x, prob = c(1,2,5)))

        n, p = 3, [1./8, 2./8, 5./8]
        r_vals = {(0, 0, 3): 0.244140625, (1, 0, 2): 0.146484375,
                  (2, 0, 1): 0.029296875, (3, 0, 0): 0.001953125,
                  (0, 1, 2): 0.292968750, (1, 1, 1): 0.117187500,
                  (2, 1, 0): 0.011718750, (0, 2, 1): 0.117187500,
                  (1, 2, 0): 0.023437500, (0, 3, 0): 0.015625000}
        for x in r_vals:
            assert_allclose(multinomial.pmf(x, n, p), r_vals[x], atol=1e-14)

    @pytest.mark.parametrize("n", [0, 3])
    def test_rvs_np(self, n):
        # 测试 .rvs 方法是否与 numpy 一致
        sc_rvs = multinomial.rvs(n, [1/4.]*3, size=7, random_state=123)
        rndm = np.random.RandomState(123)
        np_rvs = rndm.multinomial(n, [1/4.]*3, size=7)
        assert_equal(sc_rvs, np_rvs)
    # 定义一个测试方法，测试 multinomial.pmf 函数的不同输入情况
    def test_pmf(self):
        # 计算 multinomial 分布的概率质量函数，参数为 (5,)，总试验次数为 5，概率为 (1,)
        vals0 = multinomial.pmf((5,), 5, (1,))
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals0, 1, rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 (3, 4)，总试验次数为 7，概率为 (.3, .7)
        vals1 = multinomial.pmf((3, 4), 7, (.3, .7))
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals1, .22689449999999994, rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [[[3, 5], [0, 8]], [[-1, 9], [1, 1]]]，总试验次数为 8，概率为 (.1, .9)
        vals2 = multinomial.pmf([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], 8, (.1, .9))
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals2, [[.03306744, .43046721], [0, 0]], rtol=1e-8)

        # 创建一个空的 NumPy 数组 x，形状为 (0, 2)，数据类型为 np.float64
        x = np.empty((0, 2), dtype=np.float64)
        # 计算 multinomial 分布的概率质量函数，参数为 x，总试验次数为 4，概率为 (.3, .7)
        vals3 = multinomial.pmf(x, 4, (.3, .7))
        # 断言计算得到的结果与空的相同数据类型的 NumPy 数组相等
        assert_equal(vals3, np.empty([], dtype=np.float64))

        # 计算 multinomial 分布的概率质量函数，参数为 [1, 2]，总试验次数为 4，概率为 (.3, .7)
        vals4 = multinomial.pmf([1, 2], 4, (.3, .7))
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals4, 0, rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [3, 3, 0]，总试验次数为 6，概率为 [2/3.0, 1/3.0, 0]
        vals5 = multinomial.pmf([3, 3, 0], 6, [2/3.0, 1/3.0, 0])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals5, 0.219478737997, rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [0, 0, 0]，总试验次数为 0，概率为 [2/3.0, 1/3.0, 0]
        vals5 = multinomial.pmf([0, 0, 0], 0, [2/3.0, 1/3.0, 0])
        # 断言计算得到的概率值为 1，与预期相等
        assert vals5 == 1

        # 计算 multinomial 分布的概率质量函数，参数为 [2, 1, 0]，总试验次数为 0，概率为 [2/3.0, 1/3.0, 0]
        vals6 = multinomial.pmf([2, 1, 0], 0, [2/3.0, 1/3.0, 0])
        # 断言计算得到的概率值为 0，与预期相等
        assert vals6 == 0

    # 定义一个测试方法，测试 multinomial.pmf 函数在广播（broadcasting）情况下的表现
    def test_pmf_broadcasting(self):
        # 计算 multinomial 分布的概率质量函数，参数为 [1, 2]，总试验次数为 3，概率为 [[.1, .9], [.2, .8]]
        vals0 = multinomial.pmf([1, 2], 3, [[.1, .9], [.2, .8]])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals0, [.243, .384], rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [1, 2]，总试验次数为 [3, 4]，概率为 [.1, .9]
        vals1 = multinomial.pmf([1, 2], [3, 4], [.1, .9])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals1, [.243, 0], rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [[[1, 2], [1, 1]]]，总试验次数为 3，概率为 [.1, .9]
        vals2 = multinomial.pmf([[[1, 2], [1, 1]]], 3, [.1, .9])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals2, [[.243, 0]], rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [1, 2]，总试验次数为 [[[3], [4]]]，概率为 [.1, .9]
        vals3 = multinomial.pmf([1, 2], [[[3], [4]]], [.1, .9])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals3, [[[.243], [0]]], rtol=1e-8)

        # 计算 multinomial 分布的概率质量函数，参数为 [[1, 2], [1, 1]]，总试验次数为 [[[[3]]]]，概率为 [.1, .9]
        vals4 = multinomial.pmf([[1, 2], [1,1]], [[[[3]]]], [.1, .9])
        # 断言计算得到的概率与预期值在一定的相对误差范围内相等
        assert_allclose(vals4, [[[[.243, 0]]]], rtol=1e-8)

    # 使用 pytest 提供的参数化装饰器，测试 multinomial.cov 函数在不同 n 值下的表现
    @pytest.mark.parametrize("n", [0, 5])
    def test_cov(self, n):
        # 计算 multinomial 分布的协方差矩阵，参数为 n=0 或 n=5，概率为 (.2, .3, .5)
        cov1 = multinomial.cov(n, (.2, .3, .5))
        # 预期的协方差矩阵结果
        cov2 = [[n*.2*.8, -n*.2*.3, -n*.2*.5],
                [-n*.3*.2, n*.3*.7, -n*.3*.5],
                [-n*.5*.2, -n*.5*.3
    # 定义一个测试方法，用于测试多项分布的熵的计算与广播特性
    def test_entropy_broadcasting(self):
        # 计算给定参数下的多项分布的熵
        ent0 = multinomial.entropy([2, 3], [.2, .3])
        # 使用assert_allclose函数检查计算结果是否接近给定的二项分布熵
        assert_allclose(ent0, [binom.entropy(2, .2), binom.entropy(3, .2)],
                        rtol=1e-8)

        # 再次计算多项分布的熵，这次参数包含二维数组，测试广播性质
        ent1 = multinomial.entropy([7, 8], [[.3, .7], [.4, .6]])
        # 使用assert_allclose函数检查计算结果是否接近给定的二项分布熵
        assert_allclose(ent1, [binom.entropy(7, .3), binom.entropy(8, .4)],
                        rtol=1e-8)

        # 再次计算多项分布的熵，这次参数包含二维数组，测试广播性质
        ent2 = multinomial.entropy([[7], [8]], [[.3, .7], [.4, .6]])
        # 使用assert_allclose函数检查计算结果是否接近给定的二项分布熵
        assert_allclose(ent2,
                        [[binom.entropy(7, .3), binom.entropy(7, .4)],
                         [binom.entropy(8, .3), binom.entropy(8, .4)]],
                        rtol=1e-8)

    # 使用pytest的参数化装饰器，定义一个测试方法，用于测试多项分布的均值计算
    @pytest.mark.parametrize("n", [0, 5])
    def test_mean(self, n):
        # 计算给定参数下的多项分布的均值
        mean1 = multinomial.mean(n, [.2, .8])
        # 使用assert_allclose函数检查计算结果是否接近给定的均值
        assert_allclose(mean1, [n*.2, n*.8], rtol=1e-8)

    # 定义一个测试方法，用于测试多项分布均值的广播特性
    def test_mean_broadcasting(self):
        # 计算给定参数下的多项分布的均值
        mean1 = multinomial.mean([5, 6], [.2, .8])
        # 使用assert_allclose函数检查计算结果是否接近给定的均值
        assert_allclose(mean1, [[5*.2, 5*.8], [6*.2, 6*.8]], rtol=1e-8)

    # 定义一个测试方法，用于测试多项分布的冻结分布特性
    def test_frozen(self):
        # 设置随机数种子，确保结果可复现
        np.random.seed(1234)
        # 定义多项分布的参数
        n = 12
        pvals = (.1, .2, .3, .4)
        # 定义测试数据集合，包含不同的观测值
        x = [[0,0,0,12],[0,0,1,11],[0,1,1,10],[1,1,1,9],[1,1,2,8]]
        # 将数据集合转换为NumPy数组，并指定数据类型为float64
        x = np.asarray(x, dtype=np.float64)
        # 创建一个冻结的多项分布对象
        mn_frozen = multinomial(n, pvals)
        # 使用assert_allclose函数检查冻结分布的pmf方法的计算结果是否接近原始多项分布的结果
        assert_allclose(mn_frozen.pmf(x), multinomial.pmf(x, n, pvals))
        # 使用assert_allclose函数检查冻结分布的logpmf方法的计算结果是否接近原始多项分布的结果
        assert_allclose(mn_frozen.logpmf(x), multinomial.logpmf(x, n, pvals))
        # 使用assert_allclose函数检查冻结分布的entropy方法的计算结果是否接近原始多项分布的结果
        assert_allclose(mn_frozen.entropy(), multinomial.entropy(n, pvals))

    # 定义一个测试方法，用于验证GitHub问题#11860的修复情况
    def test_gh_11860(self):
        # GitHub问题#11860报告了在多项分布调整概率参数时可能会导致nan的情况，即使输入基本有效。
        # 检查一个病态情况，确保返回有限且非零的结果。（在修复之前会在主分支失败。）
        n = 88
        # 使用随机数生成器创建随机数种子
        rng = np.random.default_rng(8879715917488330089)
        # 生成n个随机概率值
        p = rng.random(n)
        # 将最后一个概率设置为1e-30，然后归一化确保概率和为1
        p[-1] = 1e-30
        p /= np.sum(p)
        # 创建一个全1的数组作为观测值
        x = np.ones(n)
        # 计算多项分布的logpmf，并检查结果是否有限
        logpmf = multinomial.logpmf(x, n, p)
        assert np.isfinite(logpmf)
class TestInvwishart:
    def test_frozen(self):
        # Test that the frozen and non-frozen inverse Wishart gives the same
        # answers

        # Construct an arbitrary positive definite scale matrix
        dim = 4
        scale = np.diag(np.arange(dim)+1)  # 创建一个对角矩阵，对角线元素为 1 到 dim
        scale[np.tril_indices(dim, k=-1)] = np.arange(dim*(dim-1)/2)  # 将下三角部分填充为从 0 开始的连续索引
        scale = np.dot(scale.T, scale)  # 计算 scale 的转置与自身的乘积，确保是正定矩阵

        # Construct a collection of positive definite matrices to test the PDF
        X = []
        for i in range(5):
            x = np.diag(np.arange(dim)+(i+1)**2)  # 创建对角矩阵，对角线元素为 (i+1)^2
            x[np.tril_indices(dim, k=-1)] = np.arange(dim*(dim-1)/2)  # 填充下三角部分为从 0 开始的连续索引
            x = np.dot(x.T, x)  # 计算 x 的转置与自身的乘积，确保是正定矩阵
            X.append(x)
        X = np.array(X).T  # 转换为 NumPy 数组，并进行转置操作

        # Construct a 1D and 2D set of parameters
        parameters = [
            (10, 1, np.linspace(0.1, 10, 5)),  # 1维情况的参数组合
            (10, scale, X)  # 2维情况的参数组合
        ]

        for (df, scale, x) in parameters:
            iw = invwishart(df, scale)  # 创建逆 Wishart 分布对象
            assert_equal(iw.var(), invwishart.var(df, scale))  # 检查方差是否一致
            assert_equal(iw.mean(), invwishart.mean(df, scale))  # 检查均值是否一致
            assert_equal(iw.mode(), invwishart.mode(df, scale))  # 检查众数是否一致
            assert_allclose(iw.pdf(x), invwishart.pdf(x, df, scale))  # 检查概率密度函数是否一致

    def test_1D_is_invgamma(self):
        # The 1-dimensional inverse Wishart with an identity scale matrix is
        # just an inverse gamma distribution.
        # Test variance, mean, pdf, entropy
        # Kolgomorov-Smirnov test for rvs
        np.random.seed(482974)

        sn = 500
        dim = 1
        scale = np.eye(dim)  # 创建一个单位矩阵作为 scale

        df_range = np.arange(5, 20, 2, dtype=float)  # 创建一个浮点数数组，作为自由度的范围
        X = np.linspace(0.1,10,num=10)  # 创建一个包含 10 个元素的等间隔数组

        for df in df_range:
            iw = invwishart(df, scale)  # 创建逆 Wishart 分布对象
            ig = invgamma(df/2, scale=1./2)  # 创建逆 Gamma 分布对象

            # Statistics
            assert_allclose(iw.var(), ig.var())  # 检查方差是否一致
            assert_allclose(iw.mean(), ig.mean())  # 检查均值是否一致

            # PDF
            assert_allclose(iw.pdf(X), ig.pdf(X))  # 检查概率密度函数是否一致

            # rvs
            rvs = iw.rvs(size=sn)  # 生成随机变量
            args = (df/2, 0, 1./2)  # 定义参数元组
            alpha = 0.01  # 设置显著性水平
            check_distribution_rvs('invgamma', args, alpha, rvs)  # 进行分布检验

            # entropy
            assert_allclose(iw.entropy(), ig.entropy())  # 检查熵是否一致
    def test_invwishart_2D_rvs(self):
        dim = 3
        df = 10

        # 构造一个简单的非对角正定矩阵
        scale = np.eye(dim)
        scale[0,1] = 0.5
        scale[1,0] = 0.5

        # 构造冻结的逆-Wishart随机变量
        iw = invwishart(df, scale)

        # 从已知种子获取生成的随机变量
        np.random.seed(608072)
        iw_rvs = invwishart.rvs(df, scale)
        np.random.seed(608072)
        frozen_iw_rvs = iw.rvs()

        # 根据 https://arxiv.org/abs/2310.15884 中逆-Wishart分解的方法手动计算期望值
        np.random.seed(608072)
        covariances = np.random.normal(size=3)
        variances = np.r_[
            np.random.chisquare(df-2),
            np.random.chisquare(df-1),
            np.random.chisquare(df),
        ]**0.5

        # 构造下三角矩阵A
        A = np.diag(variances)
        A[np.tril_indices(dim, k=-1)] = covariances

        # 逆-Wishart随机变量
        D = np.linalg.cholesky(scale)
        L = np.linalg.solve(A.T, D.T).T
        manual_iw_rvs = np.dot(L, L.T)

        # 测试是否相等
        assert_allclose(iw_rvs, manual_iw_rvs)
        assert_allclose(frozen_iw_rvs, manual_iw_rvs)

    def test_sample_mean(self):
        """测试样本均值是否与已知均值一致。"""
        # 构造任意的正定尺度矩阵
        df = 10
        sample_size = 20_000
        for dim in [1, 5]:
            scale = np.diag(np.arange(dim) + 1)
            scale[np.tril_indices(dim, k=-1)] = np.arange(dim * (dim - 1) / 2)
            scale = np.dot(scale.T, scale)

            dist = invwishart(df, scale)
            Xmean_exp = dist.mean()
            Xvar_exp = dist.var()
            Xmean_std = (Xvar_exp / sample_size)**0.5  # 均值估计的渐近标准误差

            X = dist.rvs(size=sample_size, random_state=1234)
            Xmean_est = X.mean(axis=0)

            ntests = dim*(dim + 1)//2
            fail_rate = 0.01 / ntests  # 对多次测试进行修正
            max_diff = norm.ppf(1 - fail_rate / 2)
            assert np.allclose(
                (Xmean_est - Xmean_exp) / Xmean_std,
                0,
                atol=max_diff,
            )
    def test_logpdf_4x4(self):
        """Regression test for gh-8844."""
        # 创建一个 4x4 的 numpy 数组 X，用于测试
        X = np.array([[2, 1, 0, 0.5],
                      [1, 2, 0.5, 0.5],
                      [0, 0.5, 3, 1],
                      [0.5, 0.5, 1, 2]])
        # 创建一个 4x4 的 numpy 数组 Psi，用于测试
        Psi = np.array([[9, 7, 3, 1],
                        [7, 9, 5, 1],
                        [3, 5, 8, 2],
                        [1, 1, 2, 9]])
        # 设置自由度 nu 为 6
        nu = 6
        # 计算逆 Wishart 分布的对数概率密度函数值
        prob = invwishart.logpdf(X, nu, Psi)
        # 根据维基百科上的公式进行显式计算
        p = X.shape[0]
        # 计算 X 的行列式的符号和自然对数
        sig, logdetX = np.linalg.slogdet(X)
        # 计算 Psi 的行列式的符号和自然对数
        sig, logdetPsi = np.linalg.slogdet(Psi)
        # 解 X * M = Psi，计算 M 的矩阵
        M = np.linalg.solve(X, Psi)
        # 计算期望值
        expected = ((nu/2)*logdetPsi
                    - (nu*p/2)*np.log(2)
                    - multigammaln(nu/2, p)
                    - (nu + p + 1)/2*logdetX
                    - 0.5*M.trace())
        # 使用 numpy 的 assert_allclose 函数检查 prob 和 expected 的近似程度
        assert_allclose(prob, expected)
class TestSpecialOrthoGroup:
    def test_reproducibility(self):
        np.random.seed(514)  # 设定随机种子，确保结果可复现
        x = special_ortho_group.rvs(3)  # 生成一个3阶特殊正交群的随机矩阵
        expected = np.array([[-0.99394515, -0.04527879, 0.10011432],
                             [0.04821555, -0.99846897, 0.02711042],
                             [0.09873351, 0.03177334, 0.99460653]])
        assert_array_almost_equal(x, expected)  # 断言生成的随机矩阵与预期矩阵近似相等

        random_state = np.random.RandomState(seed=514)  # 使用随机状态对象设定种子
        x = special_ortho_group.rvs(3, random_state=random_state)  # 使用指定随机状态生成随机矩阵
        assert_array_almost_equal(x, expected)  # 再次断言生成的随机矩阵与预期矩阵近似相等

    def test_invalid_dim(self):
        assert_raises(ValueError, special_ortho_group.rvs, None)  # 测试无效维度输入抛出值错误异常
        assert_raises(ValueError, special_ortho_group.rvs, (2, 2))  # 测试无效维度输入抛出值错误异常
        assert_raises(ValueError, special_ortho_group.rvs, 1)  # 测试无效维度输入抛出值错误异常
        assert_raises(ValueError, special_ortho_group.rvs, 2.5)  # 测试无效维度输入抛出值错误异常

    def test_frozen_matrix(self):
        dim = 7  # 设定矩阵维度
        frozen = special_ortho_group(dim)  # 创建一个冻结的特殊正交群对象

        rvs1 = frozen.rvs(random_state=1234)  # 使用指定随机状态生成随机矩阵
        rvs2 = special_ortho_group.rvs(dim, random_state=1234)  # 使用指定随机状态生成随机矩阵

        assert_equal(rvs1, rvs2)  # 断言两个生成的随机矩阵相等

    def test_det_and_ortho(self):
        xs = [special_ortho_group.rvs(dim)
              for dim in range(2,12)
              for i in range(3)]  # 生成多个不同维度的随机正交矩阵

        # 检验行列式始终为+1
        dets = [np.linalg.det(x) for x in xs]  # 计算每个随机矩阵的行列式值
        assert_allclose(dets, [1.]*30, rtol=1e-13)  # 断言所有行列式值接近1

        # 检验这些矩阵是否为正交矩阵
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T),
                                      np.eye(x.shape[0]))  # 断言每个随机矩阵乘以其转置结果接近单位矩阵

    def test_haar(self):
        # 检验旋转下分布是否恒定
        # 每列应具有相同分布
        # 此外，另一次旋转下分布应保持不变

        # 生成样本
        dim = 5  # 设置矩阵维度
        samples = 1000  # 样本数，不宜太多以避免测试时间过长
        ks_prob = .05  # Kolmogorov-Smirnov检验的显著性水平
        np.random.seed(514)  # 设定随机种子
        xs = special_ortho_group.rvs(dim, size=samples)  # 生成指定大小的随机正交矩阵样本

        # 通过单位向量的点积检验多行(0, 1, 2)与多列(0, 2, 4, 3)的投影
        #   有效地挑选出矩阵xs中的条目
        #   这些投影应具有相同的分布，建立旋转不变性。我们使用双侧KS检验来确认这一点。
        #   我们也可以测试随机向量之间的角度是否均匀分布，但下面的测试已经足够了。
        #   不可能考虑所有配对，因此选择几个。
        els = ((0,0), (0,2), (1,4), (2,3))
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}  # 对选定的投影进行排序
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]  # 生成配对组合
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]  # 计算每对投影的KS检验的p值
        assert_array_less([ks_prob]*len(pairs), ks_tests)  # 断言所有KS检验的p值均小于设定的显著性水平
    # 测试矩阵的可重现性
    def test_reproducibility(self):
        seed = 514
        np.random.seed(seed)  # 设置随机种子
        x = ortho_group.rvs(3)  # 生成一个3阶正交矩阵
        x2 = ortho_group.rvs(3, random_state=seed)  # 使用指定种子生成另一个3阶正交矩阵
        # 注意：这个矩阵的行列式为-1，区分了O(N)与SO(N)
        assert_almost_equal(np.linalg.det(x), -1)  # 断言行列式接近于-1
        expected = np.array([[0.381686, -0.090374, 0.919863],
                             [0.905794, -0.161537, -0.391718],
                             [-0.183993, -0.98272, -0.020204]])
        assert_array_almost_equal(x, expected)  # 断言生成的矩阵接近期望值
        assert_array_almost_equal(x2, expected)  # 断言另一个生成的矩阵接近期望值

    # 测试无效维度参数的情况
    def test_invalid_dim(self):
        assert_raises(ValueError, ortho_group.rvs, None)  # 断言当维度为None时抛出值错误
        assert_raises(ValueError, ortho_group.rvs, (2, 2))  # 断言当维度为元组时抛出值错误
        assert_raises(ValueError, ortho_group.rvs, 1)  # 断言当维度为1时抛出值错误
        assert_raises(ValueError, ortho_group.rvs, 2.5)  # 断言当维度为浮点数时抛出值错误

    # 测试冻结正交矩阵的生成与性质
    def test_frozen_matrix(self):
        dim = 7
        frozen = ortho_group(dim)  # 创建一个固定的7阶正交矩阵对象
        frozen_seed = ortho_group(dim, seed=1234)  # 使用特定种子创建另一个7阶正交矩阵对象

        rvs1 = frozen.rvs(random_state=1234)  # 使用特定种子生成矩阵
        rvs2 = ortho_group.rvs(dim, random_state=1234)  # 使用特定种子生成矩阵
        rvs3 = frozen_seed.rvs(size=1)  # 使用另一个种子生成矩阵

        assert_equal(rvs1, rvs2)  # 断言两个用相同种子生成的矩阵相等
        assert_equal(rvs1, rvs3)  # 断言一个冻结对象和使用固定种子的对象生成的矩阵相等

    # 测试正交矩阵的行列式与正交性质
    def test_det_and_ortho(self):
        xs = [[ortho_group.rvs(dim)
               for i in range(10)]
              for dim in range(2,12)]

        # 测试行列式的绝对值始终为+1
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)

        # 测试这些矩阵是否为正交矩阵
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T),
                                          np.eye(x.shape[0]))

    # 使用参数化测试（针对不同维度）测试行列式分布的问题
    @pytest.mark.parametrize("dim", [2, 5, 10, 20])
    def test_det_distribution_gh18272(self, dim):
        # 测试行列式为正和负的概率是否相等
        rng = np.random.default_rng(6796248956179332344)
        dist = ortho_group(dim=dim)
        rvs = dist.rvs(size=5000, random_state=rng)
        dets = scipy.linalg.det(rvs)
        k = np.sum(dets > 0)
        n = len(dets)
        res = stats.binomtest(k, n)
        low, high = res.proportion_ci(confidence_level=0.95)
        assert low < 0.5 < high  # 断言置信区间包含0.5，即行列式为正和负的概率相等
    def test_haar(self):
        # 测试在旋转下分布是否恒定
        # 每一列应具有相同的分布
        # 此外，分布在另一次旋转下也应保持不变

        # 生成样本
        dim = 5
        samples = 1000  # 样本数量不宜过多，否则测试时间太长
        np.random.seed(518)  # 注意测试对随机种子敏感
        xs = ortho_group.rvs(dim, size=samples)

        # 将几行（0, 1, 2）与单位向量（0, 2, 4, 3）点乘，
        # 从而选取出矩阵 xs 中的条目。
        # 这些投影应具有相同的分布，确立旋转不变性。我们使用双侧 KS 检验来确认这一点。
        # 我们也可以测试随机向量之间的角度是否均匀分布，但下面的方法已足够。
        # 考虑到不可能考虑所有的配对，因此只选择了几个。
        els = ((0, 0), (0, 2), (1, 4), (2, 3))
        #proj = {(er, ec): [x[er][ec] for x in xs] for er, ec in els}
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob] * len(pairs), ks_tests)

    @pytest.mark.slow
    def test_pairwise_distances(self):
        # 测试成对距离的分布是否接近正确

        np.random.seed(514)

        def random_ortho(dim):
            u, _s, v = np.linalg.svd(np.random.normal(size=(dim, dim)))
            return np.dot(u, v)

        for dim in range(2, 6):
            def generate_test_statistics(rvs, N=1000, eps=1e-10):
                stats = np.array([
                    np.sum((rvs(dim=dim) - rvs(dim=dim)) ** 2)
                    for _ in range(N)
                ])
                # 为了考虑数值精度，增加一点噪声。
                stats += np.random.uniform(-eps, eps, size=stats.shape)
                return stats

            expected = generate_test_statistics(random_ortho)
            actual = generate_test_statistics(scipy.stats.ortho_group.rvs)

            _D, p = scipy.stats.ks_2samp(expected, actual)

            assert_array_less(.05, p)
class TestRandomCorrelation:
    def test_reproducibility(self):
        # 设置随机种子，确保结果可复现
        np.random.seed(514)
        # 定义特征值
        eigs = (.5, .8, 1.2, 1.5)
        # 生成随机相关矩阵
        x = random_correlation.rvs(eigs)
        # 使用相同的种子生成另一个随机相关矩阵
        x2 = random_correlation.rvs(eigs, random_state=514)
        # 预期的相关矩阵
        expected = np.array([[1., -0.184851, 0.109017, -0.227494],
                             [-0.184851, 1., 0.231236, 0.326669],
                             [0.109017, 0.231236, 1., -0.178912],
                             [-0.227494, 0.326669, -0.178912, 1.]])
        # 断言两个生成的相关矩阵与预期结果相近
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_eigs(self):
        # 测试无效特征值的情况
        assert_raises(ValueError, random_correlation.rvs, None)
        assert_raises(ValueError, random_correlation.rvs, 'test')
        assert_raises(ValueError, random_correlation.rvs, 2.5)
        assert_raises(ValueError, random_correlation.rvs, [2.5])
        assert_raises(ValueError, random_correlation.rvs, [[1,2],[3,4]])
        assert_raises(ValueError, random_correlation.rvs, [2.5, -.5])
        assert_raises(ValueError, random_correlation.rvs, [1, 2, .1])

    def test_frozen_matrix(self):
        # 测试固定相关矩阵的生成
        eigs = (.5, .8, 1.2, 1.5)
        # 创建一个固定的相关矩阵对象
        frozen = random_correlation(eigs)
        # 使用特定种子创建另一个固定的相关矩阵对象
        frozen_seed = random_correlation(eigs, seed=514)

        # 使用相同种子生成随机相关矩阵
        rvs1 = random_correlation.rvs(eigs, random_state=514)
        # 从固定相关矩阵对象中生成随机相关矩阵
        rvs2 = frozen.rvs(random_state=514)
        # 从使用特定种子的固定相关矩阵对象中生成随机相关矩阵
        rvs3 = frozen_seed.rvs()

        # 断言生成的三个相关矩阵在相同种子下是相等的
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_definition(self):
        # 测试多维度相关矩阵的定义：
        #
        # 1. 行列式是特征值的乘积（在示例中是正的）
        # 2. 对角线上是1
        # 3. 矩阵是对称的

        def norm(i, e):
            return i * e / sum(e)

        # 设置随机种子
        np.random.seed(123)

        # 生成不同维度下的特征值数组
        eigs = [norm(i, np.random.uniform(size=i)) for i in range(2, 6)]
        eigs.append([4, 0, 0, 0])

        # 对角线上都是1的矩阵数组
        ones = [[1.] * len(e) for e in eigs]
        # 生成多个随机相关矩阵
        xs = [random_correlation.rvs(e) for e in eigs]

        # 断言行列式是特征值乘积，精度要求为1e-13
        dets = [np.fabs(np.linalg.det(x)) for x in xs]
        dets_known = [np.prod(e) for e in eigs]
        assert_allclose(dets, dets_known, rtol=1e-13, atol=1e-13)

        # 断言对角线上的元素是1
        diags = [np.diag(x) for x in xs]
        for a, b in zip(diags, ones):
            assert_allclose(a, b, rtol=1e-13)

        # 断言相关矩阵是对称的
        for x in xs:
            assert_allclose(x, x.T, rtol=1e-13)
    def test_to_corr(self):
        # Check some corner cases in to_corr

        # ajj == 1
        # 创建一个2x2的numpy数组，表示一个特定的相关矩阵
        m = np.array([[0.1, 0], [0, 1]], dtype=float)
        # 调用_to_corr函数，将给定的数组转换为相关矩阵
        m = random_correlation._to_corr(m)
        # 断言转换后的结果与预期的相关矩阵近似相等
        assert_allclose(m, np.array([[1, 0], [0, 0.1]]))

        # Floating point overflow; fails to compute the correct
        # rotation, but should still produce some valid rotation
        # rather than infs/nans
        # 忽略浮点溢出的错误
        with np.errstate(over='ignore'):
            # 定义一个2x2的矩阵g，用于旋转矩阵的计算
            g = np.array([[0, 1], [-1, 0]])

            # 创建一个具有极大值的矩阵m0，测试_to_corr函数的行为
            m0 = np.array([[1e300, 0], [0, np.nextafter(1, 0)]], dtype=float)
            # 将m0的副本传递给_to_corr函数，生成相关矩阵m
            m = random_correlation._to_corr(m0.copy())
            # 断言旋转后的结果m与期望的结果近似相等
            assert_allclose(m, g.T.dot(m0).dot(g))

            # 创建一个具有极大值的矩阵m0，测试_to_corr函数的行为
            m0 = np.array([[0.9, 1e300], [1e300, 1.1]], dtype=float)
            # 将m0的副本传递给_to_corr函数，生成相关矩阵m
            m = random_correlation._to_corr(m0.copy())
            # 断言旋转后的结果m与期望的结果近似相等
            assert_allclose(m, g.T.dot(m0).dot(g))

        # Zero discriminant; should set the first diag entry to 1
        # 创建一个具有零判别式的矩阵m0，测试_to_corr函数的行为
        m0 = np.array([[2, 1], [1, 2]], dtype=float)
        # 将m0的副本传递给_to_corr函数，生成相关矩阵m
        m = random_correlation._to_corr(m0.copy())
        # 断言相关矩阵m的第一个对角线条目为1
        assert_allclose(m[0,0], 1)

        # Slightly negative discriminant; should be approx correct still
        # 创建一个具有稍微负判别式的矩阵m0，测试_to_corr函数的行为
        m0 = np.array([[2 + 1e-7, 1], [1, 2]], dtype=float)
        # 将m0的副本传递给_to_corr函数，生成相关矩阵m
        m = random_correlation._to_corr(m0.copy())
        # 断言相关矩阵m的第一个对角线条目为1
        assert_allclose(m[0,0], 1)
class TestUniformDirection:
    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("size", [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        # test that samples have correct shape and norm 1
        # 使用指定的随机数生成器创建 RNG 对象
        rng = np.random.default_rng(2777937887058094419)
        # 创建一个指定维度的均匀方向分布对象
        uniform_direction_dist = uniform_direction(dim, seed=rng)
        # 生成样本数据
        samples = uniform_direction_dist.rvs(size)
        # 创建均值和协方差矩阵
        mean, cov = np.zeros(dim), np.eye(dim)
        # 计算期望的数据形状
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        # 断言样本数据的形状符合期望
        assert samples.shape == expected_shape
        # 计算样本数据的范数
        norms = np.linalg.norm(samples, axis=-1)
        # 断言样本数据的范数均为1
        assert_allclose(norms, 1.)

    @pytest.mark.parametrize("dim", [None, 0, (2, 2), 2.5])
    def test_invalid_dim(self, dim):
        # 定义错误消息
        message = ("Dimension of vector must be specified, "
                   "and must be an integer greater than 0.")
        # 使用 pytest 断言捕获 ValueError 异常并检查错误消息
        with pytest.raises(ValueError, match=message):
            uniform_direction.rvs(dim)

    def test_frozen_distribution(self):
        dim = 5
        # 创建冻结的均匀方向分布对象
        frozen = uniform_direction(dim)
        frozen_seed = uniform_direction(dim, seed=514)

        # 使用指定随机种子生成随机样本数据
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = uniform_direction.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs()

        # 断言三种方法生成的随机样本数据相等
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    @pytest.mark.parametrize("dim", [2, 5, 8])
    def test_uniform(self, dim):
        # 使用指定的随机数生成器创建 RNG 对象
        rng = np.random.default_rng(1036978481269651776)
        # 创建一个指定维度的均匀方向分布对象
        spherical_dist = uniform_direction(dim, seed=rng)
        # 生成两个随机正交向量
        v1, v2 = spherical_dist.rvs(size=2)
        # 使 v2 正交于 v1
        v2 -= v1 @ v2 * v1
        v2 /= np.linalg.norm(v2)
        # 断言 v1 和 v2 正交
        assert_allclose(v1 @ v2, 0, atol=1e-14)
        # 生成数据并投影到正交向量上
        samples = spherical_dist.rvs(size=10000)
        s1 = samples @ v1
        s2 = samples @ v2
        angles = np.arctan2(s1, s2)
        # 将角度归一化到 [0, 1] 范围内
        angles += np.pi
        angles /= 2*np.pi
        # 执行 Kolmogorov-Smirnov 检验
        uniform_dist = uniform()
        kstest_result = kstest(angles, uniform_dist.cdf)
        # 断言 KS 检验的 p 值大于 0.05
        assert kstest_result.pvalue > 0.05


class TestUnitaryGroup:
    def test_reproducibility(self):
        # 设置随机种子确保结果可复现
        np.random.seed(514)
        # 生成一个 3x3 的酉矩阵
        x = unitary_group.rvs(3)
        x2 = unitary_group.rvs(3, random_state=514)

        # 预期的酉矩阵
        expected = np.array(
            [[0.308771+0.360312j, 0.044021+0.622082j, 0.160327+0.600173j],
             [0.732757+0.297107j, 0.076692-0.4614j, -0.394349+0.022613j],
             [-0.148844+0.357037j, -0.284602-0.557949j, 0.607051+0.299257j]]
        )

        # 断言生成的酉矩阵与预期相等
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)
    # 测试无效维度输入的情况
    def test_invalid_dim(self):
        # 断言期望引发 ValueError 异常，当维度为 None 时调用 unitary_group.rvs(None)
        assert_raises(ValueError, unitary_group.rvs, None)
        # 断言期望引发 ValueError 异常，当维度为 (2, 2) 时调用 unitary_group.rvs((2, 2))
        assert_raises(ValueError, unitary_group.rvs, (2, 2))
        # 断言期望引发 ValueError 异常，当维度为 1 时调用 unitary_group.rvs(1)
        assert_raises(ValueError, unitary_group.rvs, 1)
        # 断言期望引发 ValueError 异常，当维度为 2.5 时调用 unitary_group.rvs(2.5)
        assert_raises(ValueError, unitary_group.rvs, 2.5)

    # 测试冻结矩阵的生成和随机数生成的一致性
    def test_frozen_matrix(self):
        # 设定维度为 7
        dim = 7
        # 创建一个维度为 7 的冻结单位群对象
        frozen = unitary_group(dim)
        # 创建一个带种子的冻结单位群对象
        frozen_seed = unitary_group(dim, seed=514)

        # 使用相同的种子生成随机数
        rvs1 = frozen.rvs(random_state=514)
        # 直接调用静态方法生成随机数
        rvs2 = unitary_group.rvs(dim, random_state=514)
        # 使用带种子的对象生成随机数
        rvs3 = frozen_seed.rvs(size=1)

        # 断言生成的随机数相等
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    # 测试生成的单位矩阵是否确实是单位的
    def test_unitarity(self):
        # 生成多个维度在 2 到 11 之间的随机单位矩阵
        xs = [unitary_group.rvs(dim)
              for dim in range(2, 12)
              for i in range(3)]

        # 遍历每个生成的单位矩阵，检查其是否满足单位性质
        for x in xs:
            # 断言近似相等，dot(x, x.conj().T) 应接近单位矩阵 np.eye(x.shape[0])
            assert_allclose(np.dot(x, x.conj().T), np.eye(x.shape[0]), atol=1e-15)

    # 测试生成的随机矩阵的特性，例如其特征值在单位圆上分布
    def test_haar(self):
        # 测试特征值的角度 x 在复平面上是否是不相关的

        # 设定维度为 5
        dim = 5
        # 生成指定数量的样本
        samples = 1000  # 不要生成太多，否则测试时间过长
        np.random.seed(514)  # 注意测试对种子敏感

        # 生成多个维度为 5 的随机单位矩阵
        xs = unitary_group.rvs(dim, size=samples)

        # 提取特征值的角度 x，这些角度应均匀分布
        eigs = np.vstack([scipy.linalg.eigvals(x) for x in xs])
        x = np.arctan2(eigs.imag, eigs.real)

        # 执行 Kolmogorov-Smirnov 测试，验证 x 是否符合均匀分布
        res = kstest(x.ravel(), uniform(-np.pi, 2*np.pi).cdf)
        # 断言检验的 p-value 大于 0.05，表明特征值角度 x 是均匀分布的
        assert_(res.pvalue > 0.05)
class TestMultivariateT:

    # These tests were created by running vpa(mvtpdf(...)) in MATLAB. The
    # function takes no `mu` parameter. The tests were run as
    #
    # >> ans = vpa(mvtpdf(x - mu, shape, df));
    #

    # 定义测试数据集合，每个元组包含了 x, loc, shape, df, ans 五个参数
    PDF_TESTS = [(
        # x
        [
            [1, 2],             # 第一个样本点
            [4, 1],             # 第二个样本点
            [2, 1],             # 第三个样本点
            [2, 4],             # 第四个样本点
            [1, 4],             # 第五个样本点
            [4, 1],             # 第六个样本点
            [3, 2],             # 第七个样本点
            [3, 3],             # 第八个样本点
            [4, 4],             # 第九个样本点
            [5, 1],             # 第十个样本点
        ],
        # loc
        [0, 0],                 # 分布的位置参数 loc
        # shape
        [
            [1, 0],             # 第一个维度的形状参数
            [0, 1]              # 第二个维度的形状参数
        ],
        # df
        4,                      # 自由度参数 df
        # ans
        [
            0.013972450422333741737457302178882,    # 第一个样本点的期望概率密度
            0.0010998721906793330026219646100571,   # 第二个样本点的期望概率密度
            0.013972450422333741737457302178882,    # 第三个样本点的期望概率密度
            0.00073682844024025606101402363634634,  # 第四个样本点的期望概率密度
            0.0010998721906793330026219646100571,   # 第五个样本点的期望概率密度
            0.0010998721906793330026219646100571,   # 第六个样本点的期望概率密度
            0.0020732579600816823488240725481546,   # 第七个样本点的期望概率密度
            0.00095660371505271429414668515889275,  # 第八个样本点的期望概率密度
            0.00021831953784896498569831346792114,  # 第九个样本点的期望概率密度
            0.00037725616140301147447000396084604   # 第十个样本点的期望概率密度
        ]

    ), (
        # x
        [
            [0.9718, 0.1298, 0.8134],               # 第一个样本点
            [0.4922, 0.5522, 0.7185],               # 第二个样本点
            [0.3010, 0.1491, 0.5008],               # 第三个样本点
            [0.5971, 0.2585, 0.8940],               # 第四个样本点
            [0.5434, 0.5287, 0.9507],               # 第五个样本点
        ],
        # loc
        [-1, 1, 50],                                # 分布的位置参数 loc
        # shape
        [
            [1.0000, 0.5000, 0.2500],               # 第一个维度的形状参数
            [0.5000, 1.0000, -0.1000],              # 第二个维度的形状参数
            [0.2500, -0.1000, 1.0000],              # 第三个维度的形状参数
        ],
        # df
        8,                                          # 自由度参数 df
        # ans
        [
            0.00000000000000069609279697467772867405511133763,  # 第一个样本点的期望概率密度
            0.00000000000000073700739052207366474839369535934,  # 第二个样本点的期望概率密度
            0.00000000000000069522909962669171512174435447027,  # 第三个样本点的期望概率密度
            0.00000000000000074212293557998314091880208889767,  # 第四个样本点的期望概率密度
            0.00000000000000077039675154022118593323030449058   # 第五个样本点的期望概率密度
        ]
    )]

    @pytest.mark.parametrize("x, loc, shape, df, ans", PDF_TESTS)
    # 测试概率密度函数的正确性
    def test_pdf_correctness(self, x, loc, shape, df, ans):
        # 创建多变量 t 分布对象
        dist = multivariate_t(loc, shape, df, seed=0)
        # 计算概率密度函数值
        val = dist.pdf(x)
        # 断言计算结果与预期结果的近似性
        assert_array_almost_equal(val, ans)

    @pytest.mark.parametrize("x, loc, shape, df, ans", PDF_TESTS)
    # 测试对数概率密度函数的正确性
    def test_logpdf_correct(self, x, loc, shape, df, ans):
        # 创建多变量 t 分布对象
        dist = multivariate_t(loc, shape, df, seed=0)
        # 计算概率密度函数值
        val1 = dist.pdf(x)
        # 计算对数概率密度函数值
        val2 = dist.logpdf(x)
        # 断言对数概率密度函数值与自然对数后的概率密度函数值的近似性
        assert_array_almost_equal(np.log(val1), val2)

    # https://github.com/scipy/scipy/issues/10042#issuecomment-576795195
    # 测试自由度为 1 时的 t 分布是否退化为柯西分布
    def test_mvt_with_df_one_is_cauchy(self):
        x = [9, 7, 4, 1, -3, 9, 0, -3, -1, 3]
        # 计算自由度为 1 时的 t 分布的概率密度函数值
        val = multivariate_t.pdf(x, df=1)
        # 计算柯西分布在同一样本下的概率密度函数值
        ans = cauchy.pdf(x)
        # 断言两者的近似性
        assert_array_almost_equal(val, ans)
    def test_mvt_with_high_df_is_approx_normal(self):
        # `normaltest`函数返回卡方统计量和相关的 p 值。零假设是 `x` 来自正态分布，
        # 因此较低的 p 值表示拒绝零假设，即 `x` 不太可能来自正态分布。
        P_VAL_MIN = 0.1

        # 创建自由度为 100000 的多元 t 分布对象
        dist = multivariate_t(0, 1, df=100000, seed=1)
        # 从分布中生成样本
        samples = dist.rvs(size=100000)
        # 对样本进行正态性检验
        _, p = normaltest(samples)
        # 断言 p 值大于设定的最小 p 值阈值
        assert (p > P_VAL_MIN)

        # 创建多元 t 分布对象，具有指定均值、协方差和自由度
        dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000,
                              seed=42)
        # 从分布中生成样本
        samples = dist.rvs(size=100000)
        # 对样本进行正态性检验
        _, p = normaltest(samples)
        # 断言所有 p 值大于设定的最小 p 值阈值
        assert ((p > P_VAL_MIN).all())

    @patch('scipy.stats.multivariate_normal._logpdf')
    def test_mvt_with_inf_df_calls_normal(self, mock):
        # 创建自由度为无穷大的多元 t 分布对象
        dist = multivariate_t(0, 1, df=np.inf, seed=7)
        # 断言分布对象类型为 multivariate_normal_frozen
        assert isinstance(dist, multivariate_normal_frozen)
        # 调用 multivariate_t.pdf() 一次，检查 mock 对象调用次数
        multivariate_t.pdf(0, df=np.inf)
        assert mock.call_count == 1
        # 再次调用 multivariate_t.logpdf()，检查 mock 对象调用次数
        multivariate_t.logpdf(0, df=np.inf)
        assert mock.call_count == 2

    def test_shape_correctness(self):
        # 当 x 中的样本数为一时，pdf 和 logpdf 应返回标量值。
        dim = 4
        loc = np.zeros(dim)
        shape = np.eye(dim)
        df = 4.5
        x = np.zeros(dim)
        res = multivariate_t(loc, shape, df).pdf(x)
        assert np.isscalar(res)
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert np.isscalar(res)

        # 当 x 中的样本数为 n_samples 时，pdf() 和 logpdf() 应返回形状为 (n_samples,) 的概率值。
        n_samples = 7
        x = np.random.random((n_samples, dim))
        res = multivariate_t(loc, shape, df).pdf(x)
        assert (res.shape == (n_samples,))
        res = multivariate_t(loc, shape, df).logpdf(x)
        assert (res.shape == (n_samples,))

        # 当未指定 size 参数时，rvs() 应返回标量值。
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs()
        assert np.isscalar(res)

        # 当指定 size 参数时，rvs() 应返回形状为 (size,) 的向量。
        size = 7
        res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs(size=size)
        assert (res.shape == (size,))

    def test_default_arguments(self):
        # 测试默认参数情况下的多元 t 分布对象属性
        dist = multivariate_t()
        assert_equal(dist.loc, [0])
        assert_equal(dist.shape, [[1]])
        assert (dist.df == 1)

    DEFAULT_ARGS_TESTS = [
        # 默认参数测试用例列表，每个元组包含不同的输入值和预期输出值
        (None, None, None, 0, 1, 1),
        (None, None, 7, 0, 1, 7),
        (None, [[7, 0], [0, 7]], None, [0, 0], [[7, 0], [0, 7]], 1),
        (None, [[7, 0], [0, 7]], 7, [0, 0], [[7, 0], [0, 7]], 7),
        ([7, 7], None, None, [7, 7], [[1, 0], [0, 1]], 1),
        ([7, 7], None, 7, [7, 7], [[1, 0], [0, 1]], 7),
        ([7, 7], [[7, 0], [0, 7]], None, [7, 7], [[7, 0], [0, 7]], 1),
        ([7, 7], [[7, 0], [0, 7]], 7, [7, 7], [[7, 0], [0, 7]], 7)
    ]
    # 使用 pytest 的 parametrize 装饰器，为 test_default_args 方法参数化多组测试数据
    @pytest.mark.parametrize("loc, shape, df, loc_ans, shape_ans, df_ans",
                             DEFAULT_ARGS_TESTS)
    def test_default_args(self, loc, shape, df, loc_ans, shape_ans, df_ans):
        # 创建 multivariate_t 分布对象，使用给定的 loc、shape 和 df 参数
        dist = multivariate_t(loc=loc, shape=shape, df=df)
        # 断言分布对象的 loc 属性与预期值 loc_ans 相等
        assert_equal(dist.loc, loc_ans)
        # 断言分布对象的 shape 属性与预期值 shape_ans 相等
        assert_equal(dist.shape, shape_ans)
        # 断言分布对象的 df 属性与预期值 df_ans 相等
        assert (dist.df == df_ans)

    # 定义多组参数化测试数据，用于 test_scalar_list_and_ndarray_arguments 方法
    ARGS_SHAPES_TESTS = [
        (-1, 2, 3, [-1], [[2]], 3),
        ([-1], [2], 3, [-1], [[2]], 3),
        (np.array([-1]), np.array([2]), 3, [-1], [[2]], 3)
    ]

    # 使用 pytest 的 parametrize 装饰器，为 test_scalar_list_and_ndarray_arguments 方法参数化多组测试数据
    @pytest.mark.parametrize("loc, shape, df, loc_ans, shape_ans, df_ans",
                             ARGS_SHAPES_TESTS)
    def test_scalar_list_and_ndarray_arguments(self, loc, shape, df, loc_ans,
                                               shape_ans, df_ans):
        # 创建 multivariate_t 分布对象，使用给定的 loc、shape 和 df 参数
        dist = multivariate_t(loc, shape, df)
        # 断言分布对象的 loc 属性与预期值 loc_ans 相等
        assert_equal(dist.loc, loc_ans)
        # 断言分布对象的 shape 属性与预期值 shape_ans 相等
        assert_equal(dist.shape, shape_ans)
        # 断言分布对象的 df 属性与预期值 df_ans 相等
        assert_equal(dist.df, df_ans)

    # 测试参数错误处理的方法 test_argument_error_handling
    def test_argument_error_handling(self):
        # 测试当 loc 是一个二维向量时抛出 ValueError 异常
        loc = [[1, 1]]
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc))

        # 测试当 shape 不是标量或方阵时抛出 ValueError 异常
        shape = [[1, 1], [2, 2], [3, 3]]
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape))

        # 测试当 df 小于等于零时抛出 ValueError 异常
        loc = np.zeros(2)
        shape = np.eye(2)
        df = -1
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape, df=df))
        df = 0
        assert_raises(ValueError,
                      multivariate_t,
                      **dict(loc=loc, shape=shape, df=df))

    # 测试分布对象的可重现性的方法 test_reproducibility
    def test_reproducibility(self):
        # 创建随机数生成器对象 rng，并设置种子为 4
        rng = np.random.RandomState(4)
        # 生成一个长度为 3 的均匀分布的随机向量 loc
        loc = rng.uniform(size=3)
        # 创建一个 3x3 的单位矩阵 shape
        shape = np.eye(3)
        # 使用相同的 loc、shape 和种子 2 创建两个 multivariate_t 分布对象 dist1 和 dist2
        dist1 = multivariate_t(loc, shape, df=3, seed=2)
        dist2 = multivariate_t(loc, shape, df=3, seed=2)
        # 从 dist1 和 dist2 中分别生成 10 个样本数据
        samples1 = dist1.rvs(size=10)
        samples2 = dist2.rvs(size=10)
        # 断言生成的两组样本数据 samples1 和 samples2 相等
        assert_equal(samples1, samples2)

    # 测试允许奇异矩阵的方法 test_allow_singular
    def test_allow_singular(self):
        # 定义一个参数字典 args，使得 shape 成为奇异矩阵，并验证是否引发 np.linalg.LinAlgError 异常
        args = dict(loc=[0,0], shape=[[0,0],[0,1]], df=1, allow_singular=False)
        assert_raises(np.linalg.LinAlgError, multivariate_t, **args)

    # 使用多个嵌套的 parametrize 装饰器，为 test_rvs 方法参数化测试数据
    @pytest.mark.parametrize("size", [(10, 3), (5, 6, 4, 3)])
    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    @pytest.mark.parametrize("df", [1., 2., np.inf])
    def test_rvs(self, size, dim, df):
        # 创建 multivariate_t 分布对象，使用给定的零均值、单位方差和 df 参数
        dist = multivariate_t(np.zeros(dim), np.eye(dim), df)
        # 从 dist 中生成 size 大小的随机样本
        rvs = dist.rvs(size=size)
        # 断言生成的随机样本 rvs 的形状与预期的 size 和 dim 对应
        assert rvs.shape == size + (dim, )
    def test_cdf_signs(self):
        # 检查当 np.any(lower > x) 时输出的符号是否正确
        mean = np.zeros(3)  # 创建一个三维零向量作为均值
        cov = np.eye(3)  # 创建一个3x3的单位矩阵作为协方差矩阵
        df = 10  # 自由度设为10
        b = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]]  # 一个包含4个三维向量的列表b
        a = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]  # 一个包含4个三维向量的列表a
        # 当b中的元素比a中的元素少的时候，预期结果是负数
        expected_signs = np.array([1, -1, -1, 1])  # 预期符号数组
        cdf = multivariate_normal.cdf(b, mean, cov, df, lower_limit=a)  # 使用多元正态分布计算累积分布函数
        assert_allclose(cdf, cdf[0]*expected_signs)  # 断言cdf和cdf[0]乘以预期符号数组相等

    @pytest.mark.parametrize('dim', [1, 2, 5])
    def test_cdf_against_multivariate_normal(self, dim):
        # 随机生成情况下，检查与多元正态分布的累积分布函数的准确性
        self.cdf_against_mvn_test(dim)

    @pytest.mark.parametrize('dim', [3, 6, 9])
    def test_cdf_against_multivariate_normal_singular(self, dim):
        # 针对随机生成的奇异情况，检查与多元正态分布的累积分布函数的准确性
        self.cdf_against_mvn_test(3, True)

    def cdf_against_mvn_test(self, dim, singular=False):
        # 在df -> oo和MVT -> MVN的极限情况下，检查准确性
        rng = np.random.default_rng(413722918996573)  # 创建随机数生成器对象
        n = 3  # 设置n为3

        w = 10**rng.uniform(-2, 1, size=dim)  # 从指数分布中生成权重w
        cov = _random_covariance(dim, w, rng, singular)  # 生成随机协方差矩阵

        mean = 10**rng.uniform(-1, 2, size=dim) * np.sign(rng.normal(size=dim))  # 生成均值mean
        a = -10**rng.uniform(-1, 2, size=(n, dim)) + mean  # 生成下限a
        b = 10**rng.uniform(-1, 2, size=(n, dim)) + mean  # 生成上限b

        res = stats.multivariate_t.cdf(b, mean, cov, df=10000, lower_limit=a,
                                       allow_singular=True, random_state=rng)  # 使用多元t分布计算累积分布函数
        ref = stats.multivariate_normal.cdf(b, mean, cov, allow_singular=True,
                                            lower_limit=a)  # 使用多元正态分布计算累积分布函数
        assert_allclose(res, ref, atol=5e-4)  # 断言res和ref在误差容限内相等

    def test_cdf_against_univariate_t(self):
        rng = np.random.default_rng(413722918996573)  # 创建随机数生成器对象
        cov = 2  # 设置协方差为2
        mean = 0  # 设置均值为0
        x = rng.normal(size=10, scale=np.sqrt(cov))  # 生成服从正态分布的随机数x
        df = 3  # 自由度设为3

        res = stats.multivariate_t.cdf(x, mean, cov, df, lower_limit=-np.inf,
                                       random_state=rng)  # 使用多元t分布计算累积分布函数
        ref = stats.t.cdf(x, df, mean, np.sqrt(cov))  # 使用t分布计算累积分布函数
        incorrect = stats.norm.cdf(x, mean, np.sqrt(cov))  # 使用正态分布计算累积分布函数

        assert_allclose(res, ref, atol=5e-4)  # 断言res和ref在误差容限内相等
        assert np.all(np.abs(res - incorrect) > 1e-3)  # 断言res与incorrect的差异大于指定值

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    @pytest.mark.parametrize("seed", [3363958638, 7891119608, 3887698049,
                                      5013150848, 1495033423, 6170824608])
    @pytest.mark.parametrize("singular", [False, True])
    # 定义测试函数，用于测试多元 t 分布的累积分布函数（CDF）与 qsimvtv 的一致性
    def test_cdf_against_qsimvtv(self, dim, seed, singular):
        # 如果测试为奇异情况且种子不等于指定值，则跳过测试
        if singular and seed != 3363958638:
            pytest.skip('Agreement with qsimvtv is not great in singular case')
        
        # 使用 numpy 随机数生成器创建 RNG 对象
        rng = np.random.default_rng(seed)
        
        # 根据给定维度生成权重 w，取值范围为 10^-2 到 10^2 之间
        w = 10**rng.uniform(-2, 2, size=dim)
        
        # 生成随机协方差矩阵
        cov = _random_covariance(dim, w, rng, singular)
        
        # 生成随机均值向量
        mean = rng.random(dim)
        
        # 生成随机向量 a 和 b，分别作为上下限
        a = -rng.random(dim)
        b = rng.random(dim)
        
        # 生成随机的自由度参数 df
        df = rng.random() * 5
        
        # 计算多元 t 分布的 CDF 值，允许奇异情况
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng,
                                       allow_singular=True)
        
        # 使用 qsimvtv 方法计算参考值 ref，忽略无效值警告
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, np.inf*a, b - mean, rng)[0]
        
        # 使用 assert_allclose 断言实际值与参考值在一定容差范围内一致
        assert_allclose(res, ref, atol=2e-4, rtol=1e-3)

        # 带有下限的情况
        res = stats.multivariate_t.cdf(b, mean, cov, df, lower_limit=a,
                                       random_state=rng, allow_singular=True)
        
        # 使用 qsimvtv 方法计算带下限的参考值 ref，忽略无效值警告
        with np.errstate(invalid='ignore'):
            ref = _qsimvtv(20000, df, cov, a - mean, b - mean, rng)[0]
        
        # 使用 assert_allclose 断言实际值与带下限的参考值在一定容差范围内一致
        assert_allclose(res, ref, atol=1e-4, rtol=1e-3)

    @pytest.mark.slow
    # 测试多元 t 分布的 CDF 与通用数值积分器的一致性
    def test_cdf_against_generic_integrators(self):
        # 指定维度为 3
        dim = 3
        
        # 使用特定种子创建 RNG 对象
        rng = np.random.default_rng(41372291899657)
        
        # 生成权重 w，取值范围为 10^-1 到 10^1 之间
        w = 10 ** rng.uniform(-1, 1, size=dim)
        
        # 生成随机奇异协方差矩阵
        cov = _random_covariance(dim, w, rng, singular=True)
        
        # 生成随机均值向量
        mean = rng.random(dim)
        
        # 生成随机向量 a 和 b，分别作为上下限
        a = -rng.random(dim)
        b = rng.random(dim)
        
        # 生成随机的自由度参数 df
        df = rng.random() * 5
        
        # 计算多元 t 分布的 CDF 值，允许奇异情况
        res = stats.multivariate_t.cdf(b, mean, cov, df, random_state=rng,
                                       lower_limit=a)
        
        # 定义积分函数 integrand，用于计算多元 t 分布的概率密度函数值
        def integrand(x):
            return stats.multivariate_t.pdf(x.T, mean, cov, df)
        
        # 使用 qmc_quad 方法计算积分的参考值 ref，使用 Halton 序列作为积分采样点
        ref = qmc_quad(integrand, a, b, qrng=stats.qmc.Halton(d=dim, seed=rng))
        
        # 使用 assert_allclose 断言实际值与积分结果的一致性，指定容差范围
        assert_allclose(res, ref.integral, rtol=1e-3)

        # 定义三维积分函数 integrand
        def integrand(*zyx):
            return stats.multivariate_t.pdf(zyx[::-1], mean, cov, df)
        
        # 使用 tplquad 方法计算三维积分的参考值 ref
        ref = tplquad(integrand, a[0], b[0], a[1], b[1], a[2], b[2])
        
        # 使用 assert_allclose 断言实际值与三维积分结果的一致性，指定容差范围
        assert_allclose(res, ref[0], rtol=1e-3)

    # 测试多元 t 分布的 CDF 与 Matlab 的 mvtcdf 方法的一致性
    def test_against_matlab(self):
        # 使用特定种子创建 RNG 对象
        rng = np.random.default_rng(2967390923)
        
        # 指定协方差矩阵 cov
        cov = np.array([[ 6.21786909,  0.2333667 ,  7.95506077],
                        [ 0.2333667 , 29.67390923, 16.53946426],
                        [ 7.95506077, 16.53946426, 19.17725252]])
        
        # 指定自由度参数 df
        df = 1.9559939787727658
        
        # 使用指定的协方差矩阵和自由度参数创建多元 t 分布对象 dist
        dist = stats.multivariate_t(shape=cov, df=df)
        
        # 计算多元 t 分布的 CDF 值，使用指定的随机数种子
        res = dist.cdf([0, 0, 0], random_state=rng)
        
        # 指定 Matlab 的 mvtcdf 方法的参考值 ref
        ref = 0.2523
        
        # 使用 assert_allclose 断言实际值与 Matlab 方法的参考值在一定容差范围内一致
        assert_allclose(res, ref, rtol=1e-3)
    # 定义测试函数，用于测试多变量 t 分布的冻结功能
    def test_frozen(self):
        # 设定随机数生成器的种子
        seed = 4137229573
        rng = np.random.default_rng(seed)
        # 生成均匀分布随机数作为 loc
        loc = rng.uniform(size=3)
        # 生成均匀分布随机数作为 x，并加上 loc
        x = rng.uniform(size=3) + loc
        # 生成单位矩阵
        shape = np.eye(3)
        # 生成随机数作为 df
        df = rng.random()
        # 将 loc、shape、df 组成参数元组
        args = (loc, shape, df)

        # 使用相同的种子重新生成一个随机数生成器
        rng_frozen = np.random.default_rng(seed)
        rng_unfrozen = np.random.default_rng(seed)
        # 创建多变量 t 分布对象
        dist = stats.multivariate_t(*args, seed=rng_frozen)
        # 断言冻结和非冻结状态下的累积分布函数值相等
        assert_equal(dist.cdf(x),
                     multivariate_t.cdf(x, *args, random_state=rng_unfrozen))

    # 定义测试函数，测试向量化功能
    def test_vectorized(self):
        # 设定维度和 n
        dim = 4
        n = (2, 3)
        rng = np.random.default_rng(413722918996573)
        # 生成随机矩阵 A
        A = rng.random(size=(dim, dim))
        # 计算协方差矩阵
        cov = A @ A.T
        # 生成随机均值向量
        mean = rng.random(dim)
        # 生成随机向量 x
        x = rng.random(n + (dim,))
        # 生成随机数作为 df
        df = rng.random() * 5

        # 计算多变量 t 分布的累积分布函数值
        res = stats.multivariate_t.cdf(x, mean, cov, df, random_state=rng)

        # 定义用于一维情况下的累积分布函数计算函数
        def _cdf_1d(x):
            return _qsimvtv(10000, df, cov, -np.inf*x, x-mean, rng)[0]

        # 对 x 的每一行应用 _cdf_1d 函数，作为参考值
        ref = np.apply_along_axis(_cdf_1d, -1, x)
        # 断言向量化计算结果与参考值的接近度
        assert_allclose(res, ref, atol=1e-4, rtol=1e-3)

    # 根据参数化测试，测试与解析结果的比较
    @pytest.mark.parametrize("dim", (3, 7))
    def test_against_analytical(self, dim):
        rng = np.random.default_rng(413722918996573)
        # 生成 toeplitz 矩阵 A
        A = scipy.linalg.toeplitz(c=[1] + [0.5] * (dim - 1))
        # 计算多变量 t 分布的累积分布函数值
        res = stats.multivariate_t(shape=A).cdf([0] * dim, random_state=rng)
        ref = 1 / (dim + 1)
        # 断言计算结果与解析结果的接近度
        assert_allclose(res, ref, rtol=5e-5)

    # 定义测试函数，测试无穷自由度下的熵
    def test_entropy_inf_df(self):
        # 生成单位协方差矩阵和无穷自由度
        cov = np.eye(3, 3)
        df = np.inf
        # 计算多变量 t 分布的熵和多变量正态分布的熵
        mvt_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mvn_entropy = stats.multivariate_normal.entropy(None, cov)
        # 断言两者相等
        assert mvt_entropy == mvn_entropy

    # 根据参数化测试，测试一维情况下的熵
    @pytest.mark.parametrize("df", [1, 10, 100])
    def test_entropy_1d(self, df):
        # 计算多变量 t 分布的熵和 t 分布的熵
        mvt_entropy = stats.multivariate_t.entropy(shape=1., df=df)
        t_entropy = stats.t.entropy(df=df)
        # 断言计算结果的接近度
        assert_allclose(mvt_entropy, t_entropy, rtol=1e-13)

    # entropy 参考值通过数值积分计算得到
    #
    # def integrand(x, y, mvt):
    #     vec = np.array([x, y])
    #     return mvt.logpdf(vec) * mvt.pdf(vec)
    #
    # def multivariate_t_entropy_quad_2d(df, cov):
    #     dim = cov.shape[0]
    #     loc = np.zeros((dim, ))
    #     mvt = stats.multivariate_t(loc, cov, df)
    #     limit = 100
    #     return -integrate.dblquad(integrand, -limit, limit, -limit, limit,
    #                               args=(mvt, ))[0]

    # 根据参数化测试，测试熵与数值积分结果的比较
    @pytest.mark.parametrize("df, cov, ref, tol",
                             [(10, np.eye(2, 2), 3.0378770664093313, 1e-14),
                              (100, np.array([[0.5, 1], [1, 10]]),
                               3.55102424550609, 1e-8)])
    def test_entropy_vs_numerical_integration(self, df, cov, ref, tol):
        loc = np.zeros((2, ))
        # 生成多变量 t 分布对象
        mvt = stats.multivariate_t(loc, cov, df)
        # 断言计算的熵与数值积分结果的接近度
        assert_allclose(mvt.entropy(), ref, rtol=tol)
    @pytest.mark.parametrize(
        "df, dim, ref, tol",
        [
            (10, 1, 1.5212624929756808, 1e-15),  # 参数化测试参数1
            (100, 1, 1.4289633653182439, 1e-13),  # 参数化测试参数2
            (500, 1, 1.420939531869349, 1e-14),  # 参数化测试参数3
            (1e20, 1, 1.4189385332046727, 1e-15),  # 参数化测试参数4
            (1e100, 1, 1.4189385332046727, 1e-15),  # 参数化测试参数5
            (10, 10, 15.069150450832911, 1e-15),  # 参数化测试参数6
            (1000, 10, 14.19936546446673, 1e-13),  # 参数化测试参数7
            (1e20, 10, 14.189385332046728, 1e-15),  # 参数化测试参数8
            (1e100, 10, 14.189385332046728, 1e-15),  # 参数化测试参数9
            (10, 100, 148.28902883192654, 1e-15),  # 参数化测试参数10
            (1000, 100, 141.99155538003762, 1e-14),  # 参数化测试参数11
            (1e20, 100, 141.8938533204673, 1e-15),  # 参数化测试参数12
            (1e100, 100, 141.8938533204673, 1e-15),  # 参数化测试参数13
        ]
    )
    def test_extreme_entropy(self, df, dim, ref, tol):
        # 引用值是使用 mpmath 计算的：
        # from mpmath import mp
        # mp.dps = 500
        #
        # def mul_t_mpmath_entropy(dim, df=1):
        #     dim = mp.mpf(dim)
        #     df = mp.mpf(df)
        #     halfsum = (dim + df)/2
        #     half_df = df/2
        #
        #     return float(
        #         -mp.loggamma(halfsum) + mp.loggamma(half_df)
        #         + dim / 2 * mp.log(df * mp.pi)
        #         + halfsum * (mp.digamma(halfsum) - mp.digamma(half_df))
        #         + 0.0
        #     )
        # 使用 stats.multivariate_t 创建多变量 t 分布对象
        mvt = stats.multivariate_t(shape=np.eye(dim), df=df)
        # 断言多变量 t 分布的熵值接近给定的参考值 ref
        assert_allclose(mvt.entropy(), ref, rtol=tol)

    def test_entropy_with_covariance(self):
        # 使用 np.randn(5, 5) 生成随机矩阵，然后四舍五入到两位小数
        _A = np.array([
            [1.42, 0.09, -0.49, 0.17, 0.74],
            [-1.13, -0.01,  0.71, 0.4, -0.56],
            [1.07, 0.44, -0.28, -0.44, 0.29],
            [-1.5, -0.94, -0.67, 0.73, -1.1],
            [0.17, -0.08, 1.46, -0.32, 1.36]
        ])
        # 将 _A 乘以其转置得到一个对称半正定矩阵作为协方差矩阵 cov
        cov = _A @ _A.T

        # 测试渐近情况。对于大自由度，多变量 t 分布的熵值趋近于多变量正态分布的熵值。
        df = 1e20
        mul_t_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
        mul_norm_entropy = multivariate_normal(None, cov=cov).entropy()
        # 断言多变量 t 分布的熵值与多变量正态分布的熵值接近
        assert_allclose(mul_t_entropy, mul_norm_entropy, rtol=1e-15)

        # 测试常规情况。对于维度为 5，阈值约为 766.45。因此使用略有不同的自由度来比较熵值。
        df1 = 765
        df2 = 768
        # 计算不同自由度下的多变量 t 分布的熵值
        _entropy1 = stats.multivariate_t.entropy(shape=cov, df=df1)
        _entropy2 = stats.multivariate_t.entropy(shape=cov, df=df2)
        # 断言两个不同自由度下的多变量 t 分布的熵值接近
        assert_allclose(_entropy1, _entropy2, rtol=1e-5)
# 定义一个测试类 TestMultivariateHypergeom，用于测试多元超几何分布的相关函数
class TestMultivariateHypergeom:

    # 使用 pytest 的 parametrize 装饰器为 test_logpmf 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            # 从 R 的 dmvhyper 函数中得到的基准值
            ([3, 4], [5, 10], 7, -1.119814),
            # 测试当 n=0 时的情况
            ([3, 4], [5, 10], 0, -np.inf),
            # 测试当 x 中有负数时的情况
            ([-3, 4], [5, 10], 7, -np.inf),
            # 测试当 m 中有负数时（会引发 RuntimeWarning）的情况
            ([3, 4], [-5, 10], 7, np.nan),
            # 测试所有 m 中均为负数且 x.sum() 不等于 n 的情况
            ([[1, 2], [3, 4]], [[-4, -6], [-5, -10]],
             [3, 7], [np.nan, np.nan]),
            # 测试当 x 和 m 均有负数时（会引发 RuntimeWarning）的情况
            ([-3, 4], [-5, 10], 1, np.nan),
            # 测试当 x 中的值大于对应的 m 值时的情况
            ([1, 11], [10, 1], 12, np.nan),
            # 测试当 m 中有负数时（会引发 RuntimeWarning）的情况
            ([1, 11], [10, -1], 12, np.nan),
            # 测试当 n 为负数时的情况
            ([3, 4], [5, 10], -7, np.nan),
            # 测试当 x.sum() 不等于 n 的情况
            ([3, 3], [5, 10], 7, -np.inf)
        ]
    )
    # 定义 test_logpmf 方法，用于测试 multivariate_hypergeom.logpmf 函数的输出是否符合预期
    def test_logpmf(self, x, m, n, expected):
        vals = multivariate_hypergeom.logpmf(x, m, n)
        # 断言计算出的值 vals 与预期值 expected 接近（相对误差小于等于 1e-6）
        assert_allclose(vals, expected, rtol=1e-6)

    # 定义 test_reduces_hypergeom 方法，测试多元超几何分布在二维情况下是否归约到超几何分布
    def test_reduces_hypergeom(self):
        # 测试 multivariate_hypergeom.pmf 在 x=[3, 1], m=[10, 5], n=4 时是否与 hypergeom.pmf 结果接近
        val1 = multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
        val2 = hypergeom.pmf(k=3, M=15, n=4, N=10)
        assert_allclose(val1, val2, rtol=1e-8)

        # 测试 multivariate_hypergeom.pmf 在 x=[7, 3], m=[15, 10], n=10 时是否与 hypergeom.pmf 结果接近
        val1 = multivariate_hypergeom.pmf(x=[7, 3], m=[15, 10], n=10)
        val2 = hypergeom.pmf(k=7, M=25, n=10, N=15)
        assert_allclose(val1, val2, rtol=1e-8)

    # 定义 test_rvs 方法，测试 multivariate_hypergeom.rvs 方法的随机样本生成是否无偏且样本量大时是否收敛到真实均值
    def test_rvs(self):
        rv = multivariate_hypergeom(m=[3, 5], n=4)
        rvs = rv.rvs(size=1000, random_state=123)
        # 断言随机样本的均值 rvs.mean(0) 接近真实分布的均值 rv.mean()（相对误差小于等于 1e-2）
        assert_allclose(rvs.mean(0), rv.mean(), rtol=1e-2)

    # 定义 test_rvs_broadcasting 方法，测试 multivariate_hypergeom.rvs 方法在广播情况下的样本生成
    def test_rvs_broadcasting(self):
        rv = multivariate_hypergeom(m=[[3, 5], [5, 10]], n=[4, 9])
        rvs = rv.rvs(size=(1000, 2), random_state=123)
        # 断言广播后的随机样本的均值 rvs.mean(0) 接近真实分布的均值 rv.mean()（相对误差小于等于 1e-2）
        assert_allclose(rvs.mean(0), rv.mean(), rtol=1e-2)

    # 使用 pytest 的 parametrize 装饰器为 test_rvs_gh16171 方法提供多组参数化测试数据
    @pytest.mark.parametrize('m, n', (
        ([0, 0, 20, 0, 0], 5), ([0, 0, 0, 0, 0], 0),
        ([0, 0], 0), ([0], 0)
    ))
    # 定义 test_rvs_gh16171 方法，测试 multivariate_hypergeom.rvs 方法在特定条件下的输出是否符合预期
    def test_rvs_gh16171(self, m, n):
        res = multivariate_hypergeom.rvs(m, n)
        m = np.asarray(m)
        res_ex = m.copy()
        res_ex[m != 0] = n
        # 断言计算结果 res 与预期结果 res_ex 相等
        assert_equal(res, res_ex)
    # 使用 pytest 的 parametrize 装饰器为 test_pmf 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            # 单元素列表示例，期望返回值为1
            ([5], [5], 5, 1),
            # 多元素列表示例，期望返回值为0.3263403
            ([3, 4], [5, 10], 7, 0.3263403),
            # 复杂嵌套列表示例，与 R 中的 dmvhyper 函数返回值对比
            # 输入为三维列表和对应的期望输出
            ([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]],
             [5, 10], [[8, 8], [8, 2]],
             [[0.3916084, 0.006993007], [0, 0.4761905]]),
            # 空数组示例，期望返回空列表
            (np.array([], dtype=int), np.array([], dtype=int), 0, []),
            # 多元素列表示例，期望返回值为0
            ([1, 2], [4, 5], 5, 0),
            # 复杂列表示例，与 R 中的 dmvhyper 函数返回值对比
            ([3, 3, 0], [5, 6, 7], 6, 0.01077354)
        ]
    )
    # 测试多元超几何分布的概率质量函数 pmf
    def test_pmf(self, x, m, n, expected):
        # 调用 multivariate_hypergeom 模块的 pmf 函数计算概率质量函数值
        vals = multivariate_hypergeom.pmf(x, m, n)
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(vals, expected, rtol=1e-7)

    # 使用 pytest 的 parametrize 装饰器为 test_pmf_broadcasting 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "x, m, n, expected",
        [
            # 多元素列表示例，期望返回多个概率值
            ([3, 4], [[5, 10], [10, 15]], 7, [0.3263403, 0.3407531]),
            # 多维列表示例，期望返回多个概率值
            ([[1], [2]], [[3], [4]], [1, 3], [1., 0.]),
            # 复杂嵌套列表示例，期望返回多维概率值
            ([[[1], [2]]], [[3], [4]], [1, 3], [[1., 0.]]),
            # 多维列表示例，期望返回多维概率值
            ([[1], [2]], [[[[3]]]], [1, 3], [[[1., 0.]]])
        ]
    )
    # 测试多元超几何分布的概率质量函数 pmf，支持广播特性
    def test_pmf_broadcasting(self, x, m, n, expected):
        # 调用 multivariate_hypergeom 模块的 pmf 函数计算概率质量函数值
        vals = multivariate_hypergeom.pmf(x, m, n)
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(vals, expected, rtol=1e-7)

    # 测试多元超几何分布的协方差计算 cov
    def test_cov(self):
        # 调用 multivariate_hypergeom 模块的 cov 函数计算协方差矩阵
        cov1 = multivariate_hypergeom.cov(m=[3, 7, 10], n=12)
        # 预期的协方差矩阵
        cov2 = [[0.64421053, -0.26526316, -0.37894737],
                [-0.26526316, 1.14947368, -0.88421053],
                [-0.37894737, -0.88421053, 1.26315789]]
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(cov1, cov2, rtol=1e-8)

    # 测试多元超几何分布的协方差计算 cov，支持广播特性
    def test_cov_broadcasting(self):
        # 调用 multivariate_hypergeom 模块的 cov 函数计算协方差矩阵
        cov1 = multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
        # 预期的协方差矩阵
        cov2 = [[[1.05, -1.05], [-1.05, 1.05]],
                [[1.56, -1.56], [-1.56, 1.56]]]
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(cov1, cov2, rtol=1e-8)

        # 调用 multivariate_hypergeom 模块的 cov 函数计算协方差矩阵
        cov3 = multivariate_hypergeom.cov(m=[[4], [5]], n=[4, 5])
        # 预期的协方差矩阵
        cov4 = [[[0.]], [[0.]]]
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(cov3, cov4, rtol=1e-8)

        # 调用 multivariate_hypergeom 模块的 cov 函数计算协方差矩阵
        cov5 = multivariate_hypergeom.cov(m=[7, 9], n=[8, 12])
        # 预期的协方差矩阵
        cov6 = [[[1.05, -1.05], [-1.05, 1.05]],
                [[0.7875, -0.7875], [-0.7875, 0.7875]]]
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(cov5, cov6, rtol=1e-8)

    # 测试多元超几何分布的方差计算 var
    def test_var(self):
        # 调用 multivariate_hypergeom 模块的 var 函数计算方差
        # 使用 hypergeom 模块中的 var 函数作为参考值进行验证
        var0 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var1 = hypergeom.var(M=15, n=4, N=10)
        # 使用 assert_allclose 断言方法验证计算结果与期望值的接近程度
        assert_allclose(var0, var1, rtol=1e-8)
    def test_var_broadcasting(self):
        # 计算多元超几何分布的方差，分别对应不同的参数组合
        var0 = multivariate_hypergeom.var(m=[10, 5], n=[4, 8])
        var1 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var2 = multivariate_hypergeom.var(m=[10, 5], n=8)
        # 使用 assert_allclose 检查计算结果与期望值的接近程度
        assert_allclose(var0[0], var1, rtol=1e-8)
        assert_allclose(var0[1], var2, rtol=1e-8)

        # 进行多维情况下的方差计算，并与预期结果 var4 进行比较
        var3 = multivariate_hypergeom.var(m=[[10, 5], [10, 14]], n=[4, 8])
        var4 = [[0.6984127, 0.6984127], [1.352657, 1.352657]]
        assert_allclose(var3, var4, rtol=1e-8)

        # 计算另一组参数下的方差，并与预期结果 var6 进行比较
        var5 = multivariate_hypergeom.var(m=[[5], [10]], n=[5, 10])
        var6 = [[0.], [0.]]
        assert_allclose(var5, var6, rtol=1e-8)

    def test_mean(self):
        # 使用 multivariate_hypergeom 计算多元超几何分布的期望
        mean0 = multivariate_hypergeom.mean(m=[10, 5], n=4)
        # 使用 hypergeom 计算超几何分布的期望
        mean1 = hypergeom.mean(M=15, n=4, N=10)
        # 使用 assert_allclose 检查计算结果与期望值的接近程度
        assert_allclose(mean0[0], mean1, rtol=1e-8)

        # 计算另一组参数下的期望，并与预期结果 mean3 进行比较
        mean2 = multivariate_hypergeom.mean(m=[12, 8], n=10)
        mean3 = [12.*10./20., 8.*10./20.]
        assert_allclose(mean2, mean3, rtol=1e-8)

    def test_mean_broadcasting(self):
        # 使用 multivariate_hypergeom 计算多维情况下的期望，并与预期结果 mean1 进行比较
        mean0 = multivariate_hypergeom.mean(m=[[3, 5], [10, 5]], n=[4, 8])
        mean1 = [[3.*4./8., 5.*4./8.], [10.*8./15., 5.*8./15.]]
        assert_allclose(mean0, mean1, rtol=1e-8)

    def test_mean_edge_cases(self):
        # 处理期望计算中的边缘情况：全零参数
        mean0 = multivariate_hypergeom.mean(m=[0, 0, 0], n=0)
        assert_equal(mean0, [0., 0., 0.])

        # 处理期望计算中的边缘情况：除以零的情况
        mean1 = multivariate_hypergeom.mean(m=[1, 0, 0], n=2)
        assert_equal(mean1, [np.nan, np.nan, np.nan])

        # 处理期望计算中的多维边缘情况，与预期结果 mean2 进行比较
        mean2 = multivariate_hypergeom.mean(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(mean2, [[np.nan, np.nan, np.nan], [1., 0., 1.]],
                        rtol=1e-17)

        # 处理期望计算中的空参数情况，并验证结果的形状
        mean3 = multivariate_hypergeom.mean(m=np.array([], dtype=int), n=0)
        assert_equal(mean3, [])
        assert_(mean3.shape == (0, ))

    def test_var_edge_cases(self):
        # 处理方差计算中的边缘情况：全零参数
        var0 = multivariate_hypergeom.var(m=[0, 0, 0], n=0)
        assert_allclose(var0, [0., 0., 0.], rtol=1e-16)

        # 处理方差计算中的边缘情况：除以零的情况
        var1 = multivariate_hypergeom.var(m=[1, 0, 0], n=2)
        assert_equal(var1, [np.nan, np.nan, np.nan])

        # 处理方差计算中的多维边缘情况，与预期结果 var2 进行比较
        var2 = multivariate_hypergeom.var(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(var2, [[np.nan, np.nan, np.nan], [0., 0., 0.]],
                        rtol=1e-17)

        # 处理方差计算中的空参数情况，并验证结果的形状
        var3 = multivariate_hypergeom.var(m=np.array([], dtype=int), n=0)
        assert_equal(var3, [])
        assert_(var3.shape == (0, ))

    def test_cov_edge_cases(self):
        # 处理协方差计算中的边缘情况：全零参数
        cov0 = multivariate_hypergeom.cov(m=[1, 0, 0], n=1)
        cov1 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        assert_allclose(cov0, cov1, rtol=1e-17)

        # 处理协方差计算中的边缘情况：除以零的情况
        cov3 = multivariate_hypergeom.cov(m=[0, 0, 0], n=0)
        cov4 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        assert_equal(cov3, cov4)

        # 处理协方差计算中的空参数情况，并验证结果的形状
        cov5 = multivariate_hypergeom.cov(m=np.array([], dtype=int), n=0)
        cov6 = np.array([], dtype=np.float64).reshape(0, 0)
        assert_allclose(cov5, cov6, rtol=1e-17)
        assert_(cov5.shape == (0, 0))
    # 定义一个测试方法，用于验证多元超几何分布的冻结对象与常规对象的一致性
    def test_frozen(self):
        # 设置随机种子以确保可重复性
        np.random.seed(1234)
        # 定义超几何分布的参数
        n = 12  # 总抽样数
        m = [7, 9, 11, 13]  # 各组别的总体大小
        # 定义抽样数据集
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10],
             [1, 1, 1, 9], [1, 1, 2, 8]]
        # 将数据转换为 NumPy 数组，并指定数据类型为整数
        x = np.asarray(x, dtype=int)
        
        # 使用给定的参数创建冻结的多元超几何分布对象
        mhg_frozen = multivariate_hypergeom(m, n)
        
        # 断言冻结对象的概率质量函数与常规对象计算的结果一致
        assert_allclose(mhg_frozen.pmf(x),
                        multivariate_hypergeom.pmf(x, m, n))
        # 断言冻结对象的对数概率质量函数与常规对象计算的结果一致
        assert_allclose(mhg_frozen.logpmf(x),
                        multivariate_hypergeom.logpmf(x, m, n))
        # 断言冻结对象的方差与常规对象计算的结果一致
        assert_allclose(mhg_frozen.var(), multivariate_hypergeom.var(m, n))
        # 断言冻结对象的协方差与常规对象计算的结果一致
        assert_allclose(mhg_frozen.cov(), multivariate_hypergeom.cov(m, n))

    # 定义一个测试方法，用于验证对于无效的参数，是否能正确抛出异常
    def test_invalid_params(self):
        # 断言在参数为无效值时，调用多元超几何分布函数会引发 ValueError 异常
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, 10, 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, [10], 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, [5, 4], [10], 5)
        # 断言在参数类型不符合要求时，调用多元超几何分布函数会引发 TypeError 异常
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5.5, 4.5],
                      [10, 15], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4],
                      [10.5, 15.5], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4],
                      [10, 15], 5.5)
class TestRandomTable:
    # 测试用例类 TestRandomTable，用于测试 random_table 函数的各种情况

    def get_rng(self):
        # 返回一个指定种子的 NumPy 随机数生成器对象
        return np.random.default_rng(628174795866951638)

    def test_process_parameters(self):
        # 测试 random_table 函数对参数的处理是否正确

        message = "`row` must be one-dimensional"
        # 定义错误消息，用于断言 row 必须是一维数组
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([[1, 2]], [1, 2])

        message = "`col` must be one-dimensional"
        # 定义错误消息，用于断言 col 必须是一维数组
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([1, 2], [[1, 2]])

        message = "each element of `row` must be non-negative"
        # 定义错误消息，用于断言 row 的每个元素必须为非负数
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([1, -1], [1, 2])

        message = "each element of `col` must be non-negative"
        # 定义错误消息，用于断言 col 的每个元素必须为非负数
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([1, 2], [1, -2])

        message = "sums over `row` and `col` must be equal"
        # 定义错误消息，用于断言 row 和 col 的总和必须相等
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([1, 2], [1, 0])

        message = "each element of `row` must be an integer"
        # 定义错误消息，用于断言 row 的每个元素必须为整数
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([2.1, 2.1], [1, 1, 2])

        message = "each element of `col` must be an integer"
        # 定义错误消息，用于断言 col 的每个元素必须为整数
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table([1, 2], [1.1, 1.1, 1])

        row = [1, 3]
        col = [2, 1, 1]
        r, c, n = random_table._process_parameters([1, 3], [2, 1, 1])
        # 调用 random_table._process_parameters 函数处理参数，并分别赋值给 r, c, n

        assert_equal(row, r)
        # 断言 r 应该等于 row
        assert_equal(col, c)
        # 断言 c 应该等于 col
        assert n == np.sum(row)
        # 断言 n 应该等于 row 数组元素的总和

    @pytest.mark.parametrize("scale,method",
                             ((1, "boyett"), (100, "patefield")))
    def test_process_rvs_method_on_None(self, scale, method):
        # 测试 random_table.rvs 方法在 method=None 时的处理是否正确

        row = np.array([1, 3]) * scale
        # 计算缩放后的行向量
        col = np.array([2, 1, 1]) * scale
        # 计算缩放后的列向量

        ct = random_table
        # 使用 random_table 别名 ct

        expected = ct.rvs(row, col, method=method, random_state=1)
        # 调用 random_table.rvs 方法计算期望结果
        got = ct.rvs(row, col, method=None, random_state=1)
        # 调用 random_table.rvs 方法计算实际结果

        assert_equal(expected, got)
        # 断言期望结果与实际结果相等

    def test_process_rvs_method_bad_argument(self):
        # 测试 random_table.rvs 方法传入错误参数时的处理是否正确

        row = [1, 3]
        # 定义行向量
        col = [2, 1, 1]
        # 定义列向量

        # order of items in set is random, so cannot check that
        message = "'foo' not recognized, must be one of"
        # 定义错误消息，用于断言 method 参数传入 'foo' 时抛出异常
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 断言，期望抛出 ValueError 异常并匹配给定消息
            random_table.rvs(row, col, method="foo")

    @pytest.mark.parametrize('frozen', (True, False))
    @pytest.mark.parametrize('log', (True, False))
    # 参数化测试，对 frozen 和 log 参数进行组合测试
    # 定义测试函数，用于测试概率质量函数（pmf）或对数概率质量函数（logpmf）
    def test_pmf_logpmf(self, frozen, log):
        # 通过随机样本生成来测试概率质量函数（pmf）
        # 使用 Boyett 算法实现，其实现简单且可以手动验证其正确性。
        
        # 获取随机数生成器
        rng = self.get_rng()
        
        # 指定行和列
        row = [2, 6]
        col = [1, 3, 4]
        
        # 使用 random_table.rvs 方法生成随机样本
        rvs = random_table.rvs(row, col, size=1000,
                               method="boyett", random_state=rng)

        # 根据是否冻结对象选择合适的对象
        obj = random_table(row, col) if frozen else random_table
        
        # 获取要调用的方法，根据 log 参数选择 pmf 或 logpmf
        method = getattr(obj, "logpmf" if log else "pmf")
        
        # 如果对象未冻结，则定义一个包装方法，用于传入额外的行和列参数
        if not frozen:
            original_method = method

            def method(x):
                return original_method(x, row, col)
        
        # 定义 pmf 函数，如果 log 参数为 True，则返回 np.exp(method(x))
        pmf = (lambda x: np.exp(method(x))) if log else method

        # 计算唯一随机样本和它们的计数
        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)

        # 粗略的精度检查
        p = pmf(unique_rvs)
        assert_allclose(p * len(rvs), counts, rtol=0.1)

        # 接受任何可迭代对象作为输入
        p2 = pmf(list(unique_rvs[0]))
        assert_equal(p2, p[0])

        # 接受高维输入和二维输入
        rvs_nd = rvs.reshape((10, 100) + rvs.shape[1:])
        p = pmf(rvs_nd)
        assert p.shape == (10, 100)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                pij = p[i, j]
                rvij = rvs_nd[i, j]
                qij = pmf(rvij)
                assert_equal(pij, qij)

        # 如果列边际不匹配，则概率为零
        x = [[0, 1, 1], [2, 1, 3]]
        assert_equal(np.sum(x, axis=-1), row)
        p = pmf(x)
        assert p == 0

        # 如果行边际不匹配，则概率为零
        x = [[0, 1, 2], [1, 2, 2]]
        assert_equal(np.sum(x, axis=-2), col)
        p = pmf(x)
        assert p == 0

        # 对无效输入的响应
        message = "`x` must be at least two-dimensional"
        with pytest.raises(ValueError, match=message):
            pmf([1])

        message = "`x` must contain only integral values"
        with pytest.raises(ValueError, match=message):
            pmf([[1.1]])

        message = "`x` must contain only integral values"
        with pytest.raises(ValueError, match=message):
            pmf([[np.nan]])

        message = "`x` must contain only non-negative values"
        with pytest.raises(ValueError, match=message):
            pmf([[-1]])

        message = "shape of `x` must agree with `row`"
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2, 3]])

        message = "shape of `x` must agree with `col`"
        with pytest.raises(ValueError, match=message):
            pmf([[1, 2],
                 [3, 4]])

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_mean(self, method):
        # 测试 `rvs` 是否无偏且大样本大小是否收敛到真实均值。

        # 获取随机数生成器
        rng = self.get_rng()

        # 定义行和列
        row = [2, 6]
        col = [1, 3, 4]

        # 生成随机变量样本
        rvs = random_table.rvs(row, col, size=1000, method=method,
                               random_state=rng)

        # 计算均值
        mean = random_table.mean(row, col)

        # 断言：总和应该相等于行的总和
        assert_equal(np.sum(mean), np.sum(row))

        # 断言：使用容差 0.05，检查 `rvs` 的均值是否接近真实均值
        assert_allclose(rvs.mean(0), mean, atol=0.05)

        # 断言：检查按最后一个轴求和后是否广播到给定的行
        assert_equal(rvs.sum(axis=-1), np.broadcast_to(row, (1000, 2)))

        # 断言：检查按倒数第二个轴求和后是否广播到给定的列
        assert_equal(rvs.sum(axis=-2), np.broadcast_to(col, (1000, 3)))

    def test_rvs_cov(self):
        # 测试使用 Patefield 和 Boyett 算法生成的 `rvs` 是否产生大致相同的协方差矩阵。

        # 获取随机数生成器
        rng = self.get_rng()

        # 定义行和列
        row = [2, 6]
        col = [1, 3, 4]

        # 使用 Boyett 算法生成 `rvs1`
        rvs1 = random_table.rvs(row, col, size=10000, method="boyett",
                                random_state=rng)

        # 使用 Patefield 算法生成 `rvs2`
        rvs2 = random_table.rvs(row, col, size=10000, method="patefield",
                                random_state=rng)

        # 计算 `rvs1` 和 `rvs2` 的方差
        cov1 = np.var(rvs1, axis=0)
        cov2 = np.var(rvs2, axis=0)

        # 断言：使用容差 0.02，检查 `cov1` 和 `cov2` 是否大致相等
        assert_allclose(cov1, cov2, atol=0.02)

    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_size(self, method):
        # 测试不同的 `size` 参数对生成的 `rvs` 的形状是否正确。

        # 定义行和列
        row = [2, 6]
        col = [1, 3, 4]

        # 测试 `size=None`
        rv = random_table.rvs(row, col, method=method,
                              random_state=self.get_rng())
        assert rv.shape == (2, 3)

        # 测试 `size=1`
        rv2 = random_table.rvs(row, col, size=1, method=method,
                               random_state=self.get_rng())
        assert rv2.shape == (1, 2, 3)
        assert_equal(rv, rv2[0])

        # 测试 `size=0`
        rv3 = random_table.rvs(row, col, size=0, method=method,
                               random_state=self.get_rng())
        assert rv3.shape == (0, 2, 3)

        # 测试其他有效的 `size`
        rv4 = random_table.rvs(row, col, size=20, method=method,
                               random_state=self.get_rng())
        assert rv4.shape == (20, 2, 3)

        rv5 = random_table.rvs(row, col, size=(4, 5), method=method,
                               random_state=self.get_rng())
        assert rv5.shape == (4, 5, 2, 3)

        # 断言：使用容差 1e-15，检查 `rv5` 重塑后的形状是否等于 `rv4`
        assert_allclose(rv5.reshape(20, 2, 3), rv4, rtol=1e-15)

        # 测试无效的 `size`
        message = "`size` must be a non-negative integer or `None`"
        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=-1, method=method,
                             random_state=self.get_rng())

        with pytest.raises(ValueError, match=message):
            random_table.rvs(row, col, size=np.nan, method=method,
                             random_state=self.get_rng())
    # 定义一个测试方法，用于测试随机变量生成的方法
    def test_rvs_method(self, method):
        # 该测试假设 pmf 是正确的，并检查随机样本是否遵循该概率分布。
        # 这似乎是一个循环论证，因为在 test_pmf_logpmf 中用 rvs 方法生成的随机样本会检查 pmf。
        # 这个测试不是多余的，因为 test_pmf_logpmf 故意只使用 Boyett 方法生成随机样本，
        # 但这里我们测试 Boyett 和 Patefield 两种方法。
        row = [2, 6]
        col = [1, 3, 4]

        # 从 random_table 中获取随机变量样本
        ct = random_table
        rvs = ct.rvs(row, col, size=100000, method=method,
                     random_state=self.get_rng())

        # 计算唯一随机变量样本和它们的出现次数
        unique_rvs, counts = np.unique(rvs, axis=0, return_counts=True)

        # 生成的频率应该与期望的频率相匹配
        p = ct.pmf(unique_rvs, row, col)
        assert_allclose(p * len(rvs), counts, rtol=0.02)

    # 使用参数化测试，测试在列和行中包含零值时的 rvs 方法
    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    def test_rvs_with_zeros_in_col_row(self, method):
        row = [0, 1, 0]
        col = [1, 0, 0, 0]
        
        # 创建 random_table 实例
        d = random_table(row, col)
        
        # 生成随机变量样本
        rv = d.rvs(1000, method=method, random_state=self.get_rng())
        
        # 期望的输出数组，用于断言比较
        expected = np.zeros((1000, len(row), len(col)))
        expected[...] = [[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0]]
        
        # 断言生成的随机变量与期望的输出相等
        assert_equal(rv, expected)

    # 使用参数化测试，测试在边缘情况下的 rvs 方法
    @pytest.mark.parametrize("method", (None, "boyett", "patefield"))
    @pytest.mark.parametrize("col", ([], [0]))
    @pytest.mark.parametrize("row", ([], [0]))
    def test_rvs_with_edge_cases(self, method, row, col):
        # 创建 random_table 实例
        d = random_table(row, col)
        
        # 生成随机变量样本
        rv = d.rvs(10, method=method, random_state=self.get_rng())
        
        # 期望的输出数组，用于断言比较
        expected = np.zeros((10, len(row), len(col)))
        
        # 断言生成的随机变量与期望的输出相等
        assert_equal(rv, expected)

    # 测试冻结的 random_table 实例
    def test_frozen(self):
        row = [2, 6]
        col = [1, 3, 4]
        
        # 创建带有随机种子的 random_table 实例
        d = random_table(row, col, seed=self.get_rng())

        # 生成随机变量样本
        sample = d.rvs()

        # 断言生成的随机变量的平均值与期望相等
        expected = random_table.mean(row, col)
        assert_equal(expected, d.mean())

        # 断言生成的随机变量的概率质量函数与期望相等
        expected = random_table.pmf(sample, row, col)
        assert_equal(expected, d.pmf(sample))

        # 断言生成的随机变量的对数概率质量函数与期望相等
        expected = random_table.logpmf(sample, row, col)
        assert_equal(expected, d.logpmf(sample))

    # 使用参数化测试，测试不同方法的 rvs_rcont 方法
    @pytest.mark.parametrize("method", ("boyett", "patefield"))
    # 定义测试函数，测试随机变量表格的生成
    def test_rvs_frozen(self, method):
        # 设定行和列的数量
        row = [2, 6]
        col = [1, 3, 4]
        # 使用指定种子生成随机表格数据
        d = random_table(row, col, seed=self.get_rng())

        # 生成预期结果，调用 random_table.rvs 方法
        expected = random_table.rvs(row, col, size=10, method=method,
                                    random_state=self.get_rng())
        # 调用被测试对象的 rvs 方法生成实际结果
        got = d.rvs(size=10, method=method)
        # 断言预期结果与实际结果相等
        assert_equal(expected, got)
def check_pickling(distfn, args):
    # 检查分布实例是否能够正确进行序列化和反序列化
    # 特别注意 random_state 属性的处理

    # 保存 random_state 以便稍后恢复
    rndm = distfn.random_state

    # 将 random_state 设为特定值 1234
    distfn.random_state = 1234
    # 调用分布函数的 rvs 方法生成随机样本
    distfn.rvs(*args, size=8)
    # 对 distfn 进行序列化
    s = pickle.dumps(distfn)
    # 再次调用分布函数的 rvs 方法生成随机样本
    r0 = distfn.rvs(*args, size=8)

    # 反序列化得到的对象
    unpickled = pickle.loads(s)
    # 对反序列化后的对象调用 rvs 方法生成随机样本
    r1 = unpickled.rvs(*args, size=8)
    # 断言序列化前后生成的随机样本应该相等
    assert_equal(r0, r1)

    # 恢复之前保存的 random_state
    distfn.random_state = rndm
    # 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 kappa 为 0 和 0. 的情况
    @pytest.mark.parametrize("kappa", [0, 0.])
    def test_kappa_zero(self, kappa):
        # 定义异常消息，指出当 kappa 为 0 时，von Mises-Fisher 分布变为球面上的均匀分布
        msg = ("For 'kappa=0' the von Mises-Fisher distribution "
               "becomes the uniform distribution on the sphere "
               "surface. Consider using 'scipy.stats.uniform_direction' "
               "instead.")
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，并匹配指定的异常消息
        with pytest.raises(ValueError, match=msg):
            # 调用 vonmises_fisher 函数，传入参数 [1, 0] 和 kappa
            vonmises_fisher([1, 0], kappa)


    # 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 method 参数为 vonmises_fisher.pdf 和 vonmises_fisher.logpdf 的情况
    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_invalid_shapes_pdf_logpdf(self, method):
        # 创建一个 numpy 数组 x，其维度为 [1., 0., 0]
        x = np.array([1., 0., 0])
        # 定义异常消息，指出 x 的最后一个轴的维度必须与 von Mises Fisher 分布的维度匹配
        msg = ("The dimensionality of the last axis of 'x' must "
               "match the dimensionality of the von Mises Fisher "
               "distribution.")
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，并匹配指定的异常消息
        with pytest.raises(ValueError, match=msg):
            # 调用传入的 method 函数，传入 x, [1, 0], 1 作为参数
            method(x, [1, 0], 1)

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 method 参数为 vonmises_fisher.pdf 和 vonmises_fisher.logpdf 的情况
    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_unnormalized_input(self, method):
        # 创建一个 numpy 数组 x，其维度为 [0.5, 0.]
        x = np.array([0.5, 0.])
        # 定义异常消息，指出 x 必须是最后一个维度上的单位向量
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，并匹配指定的异常消息
        with pytest.raises(ValueError, match=msg):
            # 调用传入的 method 函数，传入 x, [1, 0], 1 作为参数
            method(x, [1, 0], 1)

    # 下面的代码段是一个函数的定义，用于计算 von Mises-Fisher 分布的对数概率密度函数
    # 期望值是通过 mpmath 计算得到的
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def logpdf_mpmath(x, mu, kappa):
    #     dim = mu.size
    #     halfdim = mp.mpf(0.5 * dim)
    #     kappa = mp.mpf(kappa)
    #     const = (kappa**(halfdim - mp.one)/((2*mp.pi)**halfdim * \
    #              mp.besseli(halfdim -mp.one, kappa)))
    #     return float(const * mp.exp(kappa*mp.fdot(x, mu)))
    @pytest.mark.parametrize('x, mu, kappa, reference',
                             [(np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               1e-4, 0.0795854295583605),  # 参数化测试的参数和期望结果
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               1e-4, 0.07957747141331854),
                              (np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               100, 15.915494309189533),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               100, 5.920684802611232e-43),
                              (np.array([1., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.]),
                               2000, 5.930499050746588e-07),
                              (np.array([1., 0., 0]), np.array([1., 0., 0.]),
                               2000, 318.3098861837907),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([1., 0., 0., 0., 0.]),
                               2000, 101371.86957712633),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.,
                                         0, 0.]),
                               2000, 0.00018886808182653578),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.8), np.sqrt(0.2), 0.,
                                         0, 0.]),
                               2000, 2.0255393314603194e-87)])
    def test_pdf_accuracy(self, x, mu, kappa, reference):
        # 计算 von Mises-Fisher 分布的概率密度函数并断言结果接近参考值
        pdf = vonmises_fisher(mu, kappa).pdf(x)
        assert_allclose(pdf, reference, rtol=1e-13)

    # 通过 mpmath 计算 von Mises-Fisher 分布的对数概率密度函数的期望值
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def logpdf_mpmath(x, mu, kappa):
    #     dim = mu.size
    #     halfdim = mp.mpf(0.5 * dim)
    #     kappa = mp.mpf(kappa)
    #     two = mp.mpf(2.)
    #     const = (kappa**(halfdim - mp.one)/((two*mp.pi)**halfdim * \
    #              mp.besseli(halfdim - mp.one, kappa)))
    #     return float(mp.log(const * mp.exp(kappa*mp.fdot(x, mu))))
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义测试参数化
    @pytest.mark.parametrize('x, mu, kappa, reference',
                             [(np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               1e-4, -2.5309242486359573),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               1e-4, -2.5310242486359575),
                              (np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                               100, 2.767293119578746),
                              (np.array([1., 0., 0]), np.array([0., 0., 1.]),
                               100, -97.23270688042125),
                              (np.array([1., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.]),
                               2000, -14.337987284534103),
                              (np.array([1., 0., 0]), np.array([1., 0., 0.]),
                               2000, 5.763025393132737),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([1., 0., 0., 0., 0.]),
                               2000, 11.526550911307156),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.98), np.sqrt(0.02), 0.,
                                         0, 0.]),
                               2000, -8.574461766359684),
                              (np.array([1., 0., 0., 0., 0.]),
                               np.array([np.sqrt(0.8), np.sqrt(0.2), 0.,
                                         0, 0.]),
                               2000, -199.61906708886113)])
    def test_logpdf_accuracy(self, x, mu, kappa, reference):
        # 计算 von Mises-Fisher 分布的对数概率密度函数值
        logpdf = vonmises_fisher(mu, kappa).logpdf(x)
        # 使用 assert_allclose 断言函数来验证计算结果与参考值的接近程度
        assert_allclose(logpdf, reference, rtol=1e-14)
    
    
    # 计算 von Mises-Fisher 分布的熵的预期值，通过 mpmath 计算
    # from mpmath import mp
    # import numpy as np
    # mp.dps = 50
    # def entropy_mpmath(dim, kappa):
    #     mu = np.full((dim, ), 1/np.sqrt(dim))
    #     kappa = mp.mpf(kappa)
    #     halfdim = mp.mpf(0.5 * dim)
    #     logconstant = (mp.log(kappa**(halfdim - mp.one)
    #                    /((2*mp.pi)**halfdim
    #                    * mp.besseli(halfdim -mp.one, kappa)))
    #     return float(-logconstant - kappa * mp.besseli(halfdim, kappa)/
    #             mp.besseli(halfdim -1, kappa))
    
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器定义测试参数化
    @pytest.mark.parametrize('dim, kappa, reference',
                             [(3, 1e-4, 2.531024245302624),
                              (3, 100, -1.7672931195787458),
                              (5, 5000, -11.359032310024453),
                              (8, 1, 3.4189526482545527)])
    def test_entropy_accuracy(self, dim, kappa, reference):
        # 创建单位向量 mu
        mu = np.full((dim, ), 1/np.sqrt(dim))
        # 计算 von Mises-Fisher 分布的熵
        entropy = vonmises_fisher(mu, kappa).entropy()
        # 使用 assert_allclose 断言函数来验证计算结果与参考值的接近程度
        assert_allclose(entropy, reference, rtol=2e-14)
    @pytest.mark.parametrize("method", [vonmises_fisher.pdf,
                                        vonmises_fisher.logpdf])
    def test_broadcasting(self, method):
        # 测试 PDF 和 logPDF 的正确广播
        testshape = (2, 2)
        rng = np.random.default_rng(2777937887058094419)
        # 生成一个形状为 testshape 的随机向量
        x = uniform_direction(3).rvs(testshape, random_state=rng)
        # 设置方向向量的均值
        mu = np.full((3, ), 1/np.sqrt(3))
        kappa = 5
        # 计算所有数据点的结果
        result_all = method(x, mu, kappa)
        # 断言结果的形状与预期一致
        assert result_all.shape == testshape
        # 遍历每个数据点
        for i in range(testshape[0]):
            for j in range(testshape[1]):
                # 计算当前数据点的值
                current_val = method(x[i, j, :], mu, kappa)
                # 断言当前值与预期结果的接近程度
                assert_allclose(current_val, result_all[i, j], rtol=1e-15)

    def test_vs_vonmises_2d(self):
        # 测试在二维中，von Mises-Fisher 分布与 von Mises 分布产生相同结果
        rng = np.random.default_rng(2777937887058094419)
        mu = np.array([0, 1])
        mu_angle = np.arctan2(mu[1], mu[0])
        kappa = 20
        # 创建 von Mises-Fisher 分布对象
        vmf = vonmises_fisher(mu, kappa)
        # 创建 von Mises 分布对象
        vonmises_dist = vonmises(loc=mu_angle, kappa=kappa)
        # 生成均匀分布的方向向量
        vectors = uniform_direction(2).rvs(10, random_state=rng)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        # 断言两个分布的熵接近
        assert_allclose(vonmises_dist.entropy(), vmf.entropy())
        # 断言两个分布在给定角度处的概率密度函数接近
        assert_allclose(vonmises_dist.pdf(angles), vmf.pdf(vectors))
        # 断言两个分布在给定角度处的对数概率密度函数接近
        assert_allclose(vonmises_dist.logpdf(angles), vmf.logpdf(vectors))

    @pytest.mark.parametrize("dim", [2, 3, 6])
    @pytest.mark.parametrize("kappa, mu_tol, kappa_tol",
                             [(1, 5e-2, 5e-2),
                              (10, 1e-2, 1e-2),
                              (100, 5e-3, 2e-2),
                              (1000, 1e-3, 2e-2)])
    def test_fit_accuracy(self, dim, kappa, mu_tol, kappa_tol):
        # 测试拟合的准确性
        mu = np.full((dim, ), 1/np.sqrt(dim))
        # 创建 von Mises-Fisher 分布对象
        vmf_dist = vonmises_fisher(mu, kappa)
        rng = np.random.default_rng(2777937887058094419)
        n_samples = 10000
        # 从分布中生成样本数据
        samples = vmf_dist.rvs(n_samples, random_state=rng)
        # 进行拟合，获取拟合得到的 mu 和 kappa
        mu_fit, kappa_fit = vonmises_fisher.fit(samples)
        # 计算角度误差
        angular_error = np.arccos(mu.dot(mu_fit))
        # 断言角度误差接近零
        assert_allclose(angular_error, 0., atol=mu_tol, rtol=0)
        # 断言拟合得到的 kappa 接近预期值
        assert_allclose(kappa, kappa_fit, rtol=kappa_tol)

    def test_fit_error_one_dimensional_data(self):
        # 测试拟合函数对于一维数据的错误处理
        x = np.zeros((3, ))
        msg = "'x' must be two dimensional."
        # 断言拟合函数对于一维数据抛出 ValueError 异常，并匹配指定错误消息
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_fit_error_unnormalized_data(self):
        # 测试拟合函数对于非单位向量的错误处理
        x = np.ones((3, 3))
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        # 断言拟合函数对于非单位向量抛出 ValueError 异常，并匹配指定错误消息
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)
    # 定义一个测试方法，用于测试 vonmises_fisher 分布的特性
    def test_frozen_distribution(self):
        # 设置分布的参数 mu，这里是一个包含三个元素的 numpy 数组
        mu = np.array([0, 0, 1])
        # 设置分布的参数 kappa，表示分布的集中度
        kappa = 5
        # 使用给定的 mu 和 kappa 创建一个冻结的 vonmises_fisher 分布对象
        frozen = vonmises_fisher(mu, kappa)
        # 使用给定的 mu、kappa 和种子创建另一个冻结的 vonmises_fisher 分布对象
        frozen_seed = vonmises_fisher(mu, kappa, seed=514)

        # 从第一个冻结分布对象中生成随机变量，使用种子 514
        rvs1 = frozen.rvs(random_state=514)
        # 从 vonmises_fisher 类方法中直接生成随机变量，使用相同的 mu、kappa 和种子 514
        rvs2 = vonmises_fisher.rvs(mu, kappa, random_state=514)
        # 从第二个冻结分布对象中生成随机变量，不指定种子（使用默认的种子）
        rvs3 = frozen_seed.rvs()

        # 断言生成的随机变量应该相等
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)
class TestDirichletMultinomial:
    @classmethod
    def get_params(self, m):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(28469824356873456)
        # 生成一个包含两个元素的均匀分布随机数数组，范围在0到100之间
        alpha = rng.uniform(0, 100, size=2)
        # 生成一个m行2列的二维整数数组，元素范围在1到20之间
        x = rng.integers(1, 20, size=(m, 2))
        # 计算每行元素的和，形成一个包含m个元素的一维数组
        n = x.sum(axis=-1)
        return rng, m, alpha, n, x

    def test_frozen(self):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(28469824356873456)
        
        # 生成一个包含10个元素的均匀分布随机数数组，范围在0到100之间
        alpha = rng.uniform(0, 100, 10)
        # 生成一个包含10个元素的均匀分布随机整数数组，范围在0到10之间
        x = rng.integers(0, 10, 10)
        # 计算数组x所有元素的和
        n = np.sum(x, axis=-1)

        # 使用给定的参数alpha和n创建一个Dirichlet-Multinomial分布对象d
        d = dirichlet_multinomial(alpha, n)
        
        # 断言Dirichlet-Multinomial分布对象d的对数概率质量函数与直接计算结果的一致性
        assert_equal(d.logpmf(x), dirichlet_multinomial.logpmf(x, alpha, n))
        # 断言Dirichlet-Multinomial分布对象d的概率质量函数与直接计算结果的一致性
        assert_equal(d.pmf(x), dirichlet_multinomial.pmf(x, alpha, n))
        # 断言Dirichlet-Multinomial分布对象d的均值与直接计算结果的一致性
        assert_equal(d.mean(), dirichlet_multinomial.mean(alpha, n))
        # 断言Dirichlet-Multinomial分布对象d的方差与直接计算结果的一致性
        assert_equal(d.var(), dirichlet_multinomial.var(alpha, n))
        # 断言Dirichlet-Multinomial分布对象d的协方差矩阵与直接计算结果的一致性
        assert_equal(d.cov(), dirichlet_multinomial.cov(alpha, n))

    def test_pmf_logpmf_against_R(self):
        # 比较Dirichlet-Multinomial分布的概率质量函数与R语言extraDistr包中ddirmnom函数的结果

        x = np.array([1, 2, 3])
        n = np.sum(x)
        alpha = np.array([3, 4, 5])
        
        # 计算Dirichlet-Multinomial分布的概率质量函数与对数概率质量函数
        res = dirichlet_multinomial.pmf(x, alpha, n)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 0.08484162895927638
        
        # 断言计算结果与预期结果的接近程度
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))
        assert res.shape == logres.shape == ()

        # 使用额外的R语言extraDistr包作为参考
        # options(digits=16)
        # ddirmnom(c(4, 3, 2, 0, 2, 3, 5, 7, 4, 7), 37,
        #          c(45.01025314, 21.98739582, 15.14851365, 80.21588671,
        #            52.84935481, 25.20905262, 53.85373737, 4.88568118,
        #            89.06440654, 20.11359466))
        
        # 使用相同的随机数生成器对象生成随机参数
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        
        # 计算Dirichlet-Multinomial分布的概率质量函数与对数概率质量函数
        res = dirichlet_multinomial(alpha, n).pmf(x)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 3.65409306285992e-16
        
        # 断言计算结果与预期结果的接近程度
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))

    def test_pmf_logpmf_support(self):
        # 当类别计数的总和不等于试验次数时，概率质量函数为零，对数概率质量函数为负无穷

        # 获取随机数生成器对象和参数
        rng, m, alpha, n, x = self.get_params(1)
        n += 1
        
        # 断言Dirichlet-Multinomial分布对象的概率质量函数为零，对数概率质量函数为负无穷
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x), 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x), -np.inf)

        # 获取随机数生成器对象和参数
        rng, m, alpha, n, x = self.get_params(10)
        # 随机选择一些索引
        i = rng.random(size=10) > 0.5
        # 对所选索引对应的x值进行调整，使其总和不等于n
        x[i] = np.round(x[i] * 2)
        
        # 断言Dirichlet-Multinomial分布对象的概率质量函数为零，对数概率质量函数为负无穷
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x)[i], 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x)[i], -np.inf)
        # 断言Dirichlet-Multinomial分布对象的概率质量函数大于零，对数概率质量函数大于负无穷
        assert np.all(dirichlet_multinomial(alpha, n).pmf(x)[~i] > 0)
        assert np.all(dirichlet_multinomial(alpha, n).logpmf(x)[~i] > -np.inf)
    # 定义测试函数，测试维度为一的情况
    def test_dimensionality_one(self):
        # 如果维度为一，只有一种可能的结果
        n = 6  # 试验次数
        alpha = [10]  # 浓度参数
        x = np.asarray([n])  # 计数
        dist = dirichlet_multinomial(alpha, n)

        # 断言期望概率质量函数值为1
        assert_equal(dist.pmf(x), 1)
        # 断言期望概率质量函数值为0（超过一个计数）
        assert_equal(dist.pmf(x+1), 0)
        # 断言期望对数概率质量函数值为0
        assert_equal(dist.logpmf(x), 0)
        # 断言期望对数概率质量函数值为负无穷（超过一个计数）
        assert_equal(dist.logpmf(x+1), -np.inf)
        # 断言期望均值为n
        assert_equal(dist.mean(), n)
        # 断言期望方差为0
        assert_equal(dist.var(), 0)
        # 断言期望协方差为0
        assert_equal(dist.cov(), 0)

    # 使用参数化测试，针对 'pmf' 和 'logpmf' 方法进行测试
    @pytest.mark.parametrize('method_name', ['pmf', 'logpmf'])
    def test_against_betabinom_pmf(self, method_name):
        # 获取参数
        rng, m, alpha, n, x = self.get_params(100)

        # 获取当前方法的函数引用
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        # 获取参考方法的函数引用
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)

        # 计算当前方法的结果
        res = method(x)
        # 计算参考方法的结果
        ref = ref_method(x.T[0])
        # 断言两者结果近似相等
        assert_allclose(res, ref)

    # 使用参数化测试，针对 'mean' 和 'var' 方法进行测试
    @pytest.mark.parametrize('method_name', ['mean', 'var'])
    def test_against_betabinom_moments(self, method_name):
        # 获取参数
        rng, m, alpha, n, x = self.get_params(100)

        # 获取当前方法的函数引用
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        # 获取参考方法的函数引用
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)

        # 计算当前方法的结果
        res = method()[:, 0]
        # 计算参考方法的结果
        ref = ref_method()
        # 断言两者结果近似相等
        assert_allclose(res, ref)

    # 测试生成的分布的矩
    def test_moments(self):
        # 设置随机数生成器和维度
        rng = np.random.default_rng(28469824356873456)
        dim = 5
        # 生成随机的试验次数n
        n = rng.integers(1, 100)
        # 生成随机的浓度参数alpha
        alpha = rng.random(size=dim) * 10
        # 创建 Dirichlet-Multinomial 分布对象
        dist = dirichlet_multinomial(alpha, n)

        # 使用 NumPy 生成从分布中抽取的随机样本
        m = 100000
        p = rng.dirichlet(alpha, size=m)
        x = rng.multinomial(n, p, size=m)

        # 断言分布的均值近似等于样本均值的均值
        assert_allclose(dist.mean(), np.mean(x, axis=0), rtol=5e-3)
        # 断言分布的方差近似等于样本方差的方差
        assert_allclose(dist.var(), np.var(x, axis=0), rtol=1e-2)
        # 断言分布的均值和方差的形状正确
        assert dist.mean().shape == dist.var().shape == (dim,)

        # 计算分布的协方差矩阵
        cov = dist.cov()
        # 断言协方差矩阵的形状正确
        assert cov.shape == (dim, dim)
        # 断言分布的协方差矩阵近似等于样本协方差矩阵
        assert_allclose(cov, np.cov(x.T), rtol=2e-2)
        # 断言协方差矩阵的对角线元素等于分布的方差
        assert_equal(np.diag(cov), dist.var())
        # 断言协方差矩阵的特征值都大于0（正定性）
        assert np.all(scipy.linalg.eigh(cov)[0] > 0)
    @pytest.mark.parametrize('method', ['pmf', 'logpmf'])
    def test_broadcasting_pmf(self, method):
        # 创建一个包含多种方法的参数化测试
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        # 创建包含不同维度的 alpha 数组
        n = np.array([[6], [7], [8]])
        # 创建包含不同维度的 n 数组
        x = np.array([[1, 2, 3], [2, 2, 3]]).reshape((2, 1, 1, 3))
        # 创建包含不同维度的 x 数组
        method = getattr(dirichlet_multinomial, method)
        # 根据方法名获取相应的方法对象
        res = method(x, alpha, n)
        # 调用方法对象，计算结果
        assert res.shape == (2, 3, 4)
        # 断言结果的形状符合预期
        for i in range(len(x)):
            for j in range(len(n)):
                for k in range(len(alpha)):
                    # 迭代遍历计算每个索引的结果并比较
                    res_ijk = res[i, j, k]
                    ref = method(x[i].squeeze(), alpha[k].squeeze(), n[j].squeeze())
                    # 断言每个计算结果与参考结果的接近程度
                    assert_allclose(res_ijk, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var', 'cov'])
    def test_broadcasting_moments(self, method_name):
        # 创建一个包含多种方法名的参数化测试
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        # 创建包含不同维度的 alpha 数组
        n = np.array([[6], [7], [8]])
        # 创建包含不同维度的 n 数组
        method = getattr(dirichlet_multinomial, method_name)
        # 根据方法名获取相应的方法对象
        res = method(alpha, n)
        # 调用方法对象，计算结果
        assert res.shape == (3, 4, 3) if method_name != 'cov' else (3, 4, 3, 3)
        # 断言结果的形状符合预期
        for j in range(len(n)):
            for k in range(len(alpha)):
                # 迭代遍历计算每个索引的结果并比较
                res_ijk = res[j, k]
                ref = method(alpha[k].squeeze(), n[j].squeeze())
                # 断言每个计算结果与参考结果的接近程度
                assert_allclose(res_ijk, ref)
```