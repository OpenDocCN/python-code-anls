# `D:\src\scipysrc\scipy\scipy\stats\tests\test_mstats_basic.py`

```
"""
Tests for the stats.mstats module (support for masked arrays)
"""
# 导入警告模块
import warnings
# 导入平台信息模块
import platform

# 导入 NumPy 库并指定别名 np
import numpy as np
# 从 NumPy 中导入 nan 函数
from numpy import nan
# 导入 NumPy 中的 masked array 模块并指定别名 ma
import numpy.ma as ma
# 从 NumPy 中导入 masked 和 nomask 对象
from numpy.ma import masked, nomask

# 导入 SciPy 库中的 stats.mstats 模块并指定别名 mstats
import scipy.stats.mstats as mstats
# 从 SciPy 中导入 stats 模块
from scipy import stats
# 导入 common_tests 模块中的 check_named_results 函数
from .common_tests import check_named_results
# 导入 pytest 测试框架
import pytest
# 从 pytest 中导入 raises 函数并指定别名 assert_raises
from pytest import raises as assert_raises
# 从 numpy.ma.testutils 中导入断言函数
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
                                assert_array_almost_equal,
                                assert_array_almost_equal_nulp, assert_,
                                assert_allclose, assert_array_equal)
# 从 NumPy 中导入测试工具模块 suppress_warnings
from numpy.testing import suppress_warnings
# 从 SciPy 中导入 _mstats_basic 和 _stats_py 模块
from scipy.stats import _mstats_basic, _stats_py
# 从 scipy.conftest 中导入 skip_xp_invalid_arg 装饰器
from scipy.conftest import skip_xp_invalid_arg
# 从 scipy.stats._axis_nan_policy 中导入警告类 SmallSampleWarning 和函数 too_small_1d_not_omit

# 定义测试类 TestMquantiles
class TestMquantiles:
    # 定义测试方法 test_mquantiles_limit_keyword
    def test_mquantiles_limit_keyword(self):
        # 数据数组
        data = np.array([[6., 7., 1.],
                         [47., 15., 2.],
                         [49., 36., 3.],
                         [15., 39., 4.],
                         [42., 40., -999.],
                         [41., 41., -999.],
                         [7., -999., -999.],
                         [39., -999., -999.],
                         [43., -999., -999.],
                         [40., -999., -999.],
                         [36., -999., -999.]])
        # 预期结果数组
        desired = [[19.2, 14.6, 1.45],
                   [40.0, 37.5, 2.5],
                   [42.8, 40.05, 3.55]]
        # 计算分位数并存储结果
        quants = mstats.mquantiles(data, axis=0, limit=(0, 50))
        # 断言计算结果与预期结果的接近程度
        assert_almost_equal(quants, desired)


# 定义函数 check_equal_gmean，用于测试几何平均值
def check_equal_gmean(array_like, desired, axis=None, dtype=None, rtol=1e-7):
    # 计算给定数组的几何平均值
    x = mstats.gmean(array_like, axis=axis, dtype=dtype)
    # 断言计算结果与预期结果的接近程度
    assert_allclose(x, desired, rtol=rtol)
    # 断言计算结果的数据类型与指定的数据类型一致
    assert_equal(x.dtype, dtype)


# 定义函数 check_equal_hmean，用于测试调和平均值
def check_equal_hmean(array_like, desired, axis=None, dtype=None, rtol=1e-7):
    # 计算给定数组的调和平均值
    x = stats.hmean(array_like, axis=axis, dtype=dtype)
    # 断言计算结果与预期结果的接近程度
    assert_allclose(x, desired, rtol=rtol)
    # 断言计算结果的数据类型与指定的数据类型一致
    assert_equal(x.dtype, dtype)


# 装饰器，用于跳过无效参数的测试
@skip_xp_invalid_arg
# 定义几何平均值测试类 TestGeoMean
class TestGeoMean:
    # 定义测试方法 test_1d，用于测试一维数组的几何平均值计算
    def test_1d(self):
        # 输入数组
        a = [1, 2, 3, 4]
        # 预期结果
        desired = np.power(1*2*3*4, 1./4.)
        # 调用函数检查几何平均值
        check_equal_gmean(a, desired, rtol=1e-14)

    # 定义测试方法 test_1d_ma，用于测试带有掩码的一维 masked array 的几何平均值计算
    def test_1d_ma(self):
        # 输入数组
        a = ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # 预期结果
        desired = 45.2872868812
        # 调用函数检查几何平均值
        check_equal_gmean(a, desired)

        # 带有掩码的输入数组
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        # 预期结果
        desired = np.power(1*2*3, 1./3.)
        # 调用函数检查几何平均值
        check_equal_gmean(a, desired, rtol=1e-14)

    # 定义测试方法 test_1d_ma_value，用于测试带有掩码值的一维 masked array 的几何平均值计算
    def test_1d_ma_value(self):
        # 带有掩码值的输入数组
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        # 预期结果
        desired = 41.4716627439
        # 调用函数检查几何平均值
        check_equal_gmean(a, desired)
    # 定义一个测试函数，测试含有零元素的一维掩码数组
    def test_1d_ma0(self):
        # 创建一个一维掩码数组，包含数值 [10, 20, 30, 40, 50, 60, 70, 80, 90, 0]
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 0])
        # 期望的结果为零
        desired = 0
        # 调用函数 check_equal_gmean 来验证结果
        check_equal_gmean(a, desired)

    # 定义一个测试函数，测试含有负数元素的一维掩码数组
    def test_1d_ma_inf(self):
        # 创建一个一维掩码数组，包含数值 [10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, -1])
        # 期望的结果为 NaN（不是一个数字）
        desired = np.nan
        # 在忽略无效操作警告的情况下，调用函数 check_equal_gmean 来验证结果
        with np.errstate(invalid='ignore'):
            check_equal_gmean(a, desired)

    # 标记，如果 numpy 没有 float96 属性，则跳过该测试
    @pytest.mark.skipif(not hasattr(np, 'float96'),
                        reason='cannot find float96 so skipping')
    # 定义一个测试函数，测试 float96 类型的一维掩码数组
    def test_1d_float96(self):
        # 创建一个一维掩码数组，包含数值 [1, 2, 3, 4]，其中第四个元素被屏蔽
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        # 期望的结果为计算结果的 float96 类型
        desired_dt = np.power(1*2*3, 1./3.).astype(np.float96)
        # 调用函数 check_equal_gmean 来验证结果，指定数据类型和相对容差
        check_equal_gmean(a, desired_dt, dtype=np.float96, rtol=1e-14)

    # 定义一个测试函数，测试二维掩码数组
    def test_2d_ma(self):
        # 创建一个二维掩码数组
        a = ma.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                     mask=[[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        # 期望的结果为一维数组 [1, 2, 3, 4]，在轴 0 上验证结果，指定相对容差
        desired = np.array([1, 2, 3, 4])
        check_equal_gmean(a, desired, axis=0, rtol=1e-14)

        # 期望的结果为包含特定计算的一维掩码数组，在轴 -1 上验证结果，指定相对容差
        desired = ma.array([np.power(1*2*3*4, 1./4.),
                            np.power(2*3, 1./2.),
                            np.power(1*4, 1./2.)])
        check_equal_gmean(a, desired, axis=-1, rtol=1e-14)

        # 创建一个二维数组，并指定期望的结果
        # 测试一个二维掩码数组
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        # 调用函数 check_equal_gmean 来验证结果，将数组转换为掩码数组
        check_equal_gmean(np.ma.array(a), desired)
# 应用装饰器 @skip_xp_invalid_arg，跳过不符合要求的测试
@skip_xp_invalid_arg
# 定义测试类 TestHarMean，用于测试调和平均数计算函数
class TestHarMean:
    # 测试一维数组情况
    def test_1d(self):
        # 创建带遮盖值的一维 ma 数组
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        # 期望的调和平均数结果
        desired = 3. / (1./1 + 1./2 + 1./3)
        # 调用函数 check_equal_hmean 检查调和平均数计算结果是否与期望相等
        check_equal_hmean(a, desired, rtol=1e-14)

        # 创建不带遮盖值的一维 np.ma 数组
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # 期望的调和平均数结果
        desired = 34.1417152147
        # 再次调用函数 check_equal_hmean 检查结果是否符合期望
        check_equal_hmean(a, desired)

        # 创建带遮盖值的一维 np.ma 数组
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        # 期望的调和平均数结果
        desired = 31.8137186141
        # 再次调用函数 check_equal_hmean 检查结果是否符合期望
        check_equal_hmean(a, desired)

    # 如果系统中没有 np.float96 属性，则跳过此测试
    @pytest.mark.skipif(not hasattr(np, 'float96'),
                        reason='cannot find float96 so skipping')
    # 测试一维数组情况，使用 np.float96 类型数据
    def test_1d_float96(self):
        # 创建带遮盖值的一维 ma 数组
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        # 期望的调和平均数结果，使用 np.float96 数据类型
        desired_dt = np.asarray(3. / (1./1 + 1./2 + 1./3), dtype=np.float96)
        # 调用函数 check_equal_hmean 检查调和平均数计算结果是否与期望相等，使用 np.float96 数据类型
        check_equal_hmean(a, desired_dt, dtype=np.float96)

    # 测试二维数组情况
    def test_2d(self):
        # 创建带遮盖值的二维 ma 数组
        a = ma.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                     mask=[[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        # 期望的调和平均数结果，沿 axis=0 方向计算
        desired = ma.array([1, 2, 3, 4])
        # 调用函数 check_equal_hmean 检查调和平均数计算结果是否与期望相等，指定 axis=0
        check_equal_hmean(a, desired, axis=0, rtol=1e-14)

        # 期望的调和平均数结果列表，沿 axis=-1 方向计算
        desired = [4./(1/1.+1/2.+1/3.+1/4.), 2./(1/2.+1/3.), 2./(1/1.+1/4.)]
        # 再次调用函数 check_equal_hmean 检查调和平均数计算结果是否与期望相等，指定 axis=-1
        check_equal_hmean(a, desired, axis=-1, rtol=1e-14)

        # 创建普通的二维数组
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        # 期望的调和平均数结果
        desired = 38.6696271841
        # 调用函数 check_equal_hmean 检查调和平均数计算结果是否与期望相等
        check_equal_hmean(np.ma.array(a), desired)


# 定义测试类 TestRanking，用于测试排名函数
class TestRanking:
    # 测试排名函数
    def test_ranking(self):
        # 创建带遮盖值的一维 ma 数组
        x = ma.array([0,1,1,1,2,3,4,5,5,6,])
        # 断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x),
                            [1,3,3,3,5,6,7,8.5,8.5,10])
        
        # 将数组中索引为 3 和 4 的值设置为遮盖值
        x[[3,4]] = masked
        # 再次断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x),
                            [1,2.5,2.5,0,0,4,5,6.5,6.5,8])
        
        # 带有缺失值处理的排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x, use_missing=True),
                            [1,2.5,2.5,4.5,4.5,4,5,6.5,6.5,8])
        
        # 创建带遮盖值的一维 ma 数组
        x = ma.array([0,1,5,1,2,4,3,5,1,6,])
        # 断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x),
                            [1,3,8.5,3,5,7,6,8.5,3,10])
        
        # 创建二维 ma 数组
        x = ma.array([[0,1,1,1,2], [3,4,5,5,6,]])
        # 断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x),
                            [[1,3,3,3,5], [6,7,8.5,8.5,10]])
        
        # 沿 axis=1 方向断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x, axis=1),
                            [[1,3,3,3,5], [1,2,3.5,3.5,5]])
        
        # 沿 axis=0 方向断言排名结果与期望值几乎相等
        assert_almost_equal(mstats.rankdata(x,axis=0),
                            [[1,1,1,1,1], [2,2,2,2,2,]])

# 定义测试类 TestCorr
class TestCorr:
    def test_pearsonr(self):
        # Tests some computations of Pearson's r

        # 创建一个长度为10的掩码数组
        x = ma.arange(10)

        with warnings.catch_warnings():
            # 在这个上下文中的测试是边界情况，具有完全的相关性或反相关性，或者是完全掩盖的数据。
            # 这些情况不应该触发 RuntimeWarning。
            warnings.simplefilter("error", RuntimeWarning)

            # 断言计算的Pearson相关系数几乎等于1.0
            assert_almost_equal(mstats.pearsonr(x, x)[0], 1.0)
            # 断言计算的Pearson相关系数几乎等于-1.0
            assert_almost_equal(mstats.pearsonr(x, x[::-1])[0], -1.0)

            # 创建一个具有掩码的掩码数组
            x = ma.array(x, mask=True)
            # 计算掩码数组的Pearson相关系数
            pr = mstats.pearsonr(x, x)
            # 断言相关系数和p值都是掩码值
            assert_(pr[0] is masked)
            assert_(pr[1] is masked)

        # 创建两个掩码数组，计算它们的Pearson相关系数和p值
        x1 = ma.array([-1.0, 0.0, 1.0])
        y1 = ma.array([0, 0, 3])
        r, p = mstats.pearsonr(x1, y1)
        # 断言计算的Pearson相关系数几乎等于sqrt(3)/2
        assert_almost_equal(r, np.sqrt(3)/2)
        # 断言计算的p值几乎等于1.0/3
        assert_almost_equal(p, 1.0/3)

        # 创建具有不匹配掩码的两个掩码数组，计算它们的Pearson相关系数和p值
        mask = [False, False, False, True]
        x2 = ma.array([-1.0, 0.0, 1.0, 99.0], mask=mask)
        y2 = ma.array([0, 0, 3, -1], mask=mask)
        r, p = mstats.pearsonr(x2, y2)
        # 断言计算的Pearson相关系数几乎等于sqrt(3)/2
        assert_almost_equal(r, np.sqrt(3)/2)
        # 断言计算的p值几乎等于1.0/3

    def test_pearsonr_misaligned_mask(self):
        # 测试处理不对齐掩码的情况

        # 创建一个具有掩码的MaskedArray对象
        mx = np.ma.masked_array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 0, 0, 0])
        my = np.ma.masked_array([9, 8, 7, 6, 5, 9], mask=[0, 0, 1, 0, 0, 0])
        
        # 创建两个普通的NumPy数组
        x = np.array([1, 4, 5, 6])
        y = np.array([9, 6, 5, 9])

        # 计算MaskedArray对象的Pearson相关系数和p值
        mr, mp = mstats.pearsonr(mx, my)
        # 使用普通的NumPy数组计算Pearson相关系数和p值
        r, p = stats.pearsonr(x, y)
        
        # 断言计算得到的相关系数和p值相等
        assert_equal(mr, r)
        assert_equal(mp, p)
    # 定义测试函数 test_spearmanr
    def test_spearmanr(self):
        # 测试 Spearman's rho 的计算
        
        # 定义数据集 (x, y)
        (x, y) = ([5.05,6.75,3.21,2.66], [1.65,2.64,2.64,6.95])
        # 断言 Spearman's rho 的计算结果与预期值接近
        assert_almost_equal(mstats.spearmanr(x,y)[0], -0.6324555)
        
        # 定义包含 NaN 的数据集 (x, y)，并修复 NaN 值
        (x, y) = ([5.05,6.75,3.21,2.66,np.nan],[1.65,2.64,2.64,6.95,np.nan])
        (x, y) = (ma.fix_invalid(x), ma.fix_invalid(y))
        # 断言修复后的数据的 Spearman's rho 计算结果与预期值接近
        assert_almost_equal(mstats.spearmanr(x,y)[0], -0.6324555)

        # 定义另一组数据集 (x, y)
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
             1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
             0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]
        # 断言 Spearman's rho 的计算结果与预期值接近
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.6887299)
        
        # 定义包含 NaN 的数据集 (x, y)，并修复 NaN 值
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
             1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7, np.nan]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
             0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4, np.nan]
        (x, y) = (ma.fix_invalid(x), ma.fix_invalid(y))
        # 断言修复后的数据的 Spearman's rho 计算结果与预期值接近
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.6887299)
        
        # 下一个测试确保计算使用足够的精度。
        # 分母的值约为 n^3，之前被表示为一个整数。2000^3 > 2^32，因此这些数组可能在某些机器上导致溢出。
        x = list(range(2000))
        y = list(range(2000))
        y[0], y[9] = y[9], y[0]
        y[10], y[434] = y[434], y[10]
        y[435], y[1509] = y[1509], y[435]
        # rho = 1 - 6 * (2 * (9^2 + 424^2 + 1074^2))/(2000 * (2000^2 - 1))
        #     = 1 - (1 / 500)
        #     = 0.998
        # 断言计算出的 Spearman's rho 与预期值接近
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.998)

        # 测试命名元组属性
        # 调用 Spearman's rho 函数计算结果
        res = mstats.spearmanr(x, y)
        # 定义预期的命名元组属性
        attributes = ('correlation', 'pvalue')
        # 检查计算结果是否符合预期的命名元组属性
        check_named_results(res, attributes, ma=True)
    def test_spearmanr_alternative(self):
        # 检查与 R 的结果是否一致
        # 设置 R 中的数字显示精度为16位
        # 进行 Spearman 相关性检验，使用两个数组 x 和 y
        # 采用双侧检验，Spearman 方法进行计算
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
             1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
             0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]

        # 预期的 Spearman 相关系数，从 cor.test 中获得
        r_exp = 0.6887298747763864  # from cor.test

        # 使用 scipy.stats.mstats.spearmanr 计算实际的 Spearman 相关系数 r 和 p 值
        r, p = mstats.spearmanr(x, y)
        assert_allclose(r, r_exp)  # 断言实际的 r 值与预期相等
        assert_allclose(p, 0.004519192910756)  # 断言实际的 p 值与预期相等

        # 对于 alternative='greater' 的 Spearman 相关性检验
        r, p = mstats.spearmanr(x, y, alternative='greater')
        assert_allclose(r, r_exp)  # 断言实际的 r 值与预期相等
        assert_allclose(p, 0.002259596455378)  # 断言实际的 p 值与预期相等

        # 对于 alternative='less' 的 Spearman 相关性检验
        r, p = mstats.spearmanr(x, y, alternative='less')
        assert_allclose(r, r_exp)  # 断言实际的 r 值与预期相等
        assert_allclose(p, 0.9977404035446)  # 断言实际的 p 值与预期相等

        # 直观测试（显然具有正相关性）
        n = 100
        x = np.linspace(0, 5, n)
        y = 0.1*x + np.random.rand(n)  # y 与 x 显著正相关

        # 计算 Spearman 相关性及 p 值
        stat1, p1 = mstats.spearmanr(x, y)

        # 对于 alternative="greater" 的 Spearman 相关性检验，断言其 p 值
        stat2, p2 = mstats.spearmanr(x, y, alternative="greater")
        assert_allclose(p2, p1 / 2)  # 正相关性导致较小的 p 值

        # 对于 alternative="less" 的 Spearman 相关性检验，断言其 p 值
        stat3, p3 = mstats.spearmanr(x, y, alternative="less")
        assert_allclose(p3, 1 - p1 / 2)  # 正相关性导致较大的 p 值

        # 断言所有的统计量相等
        assert stat1 == stat2 == stat3

        # 使用 pytest 断言 ValueError 异常被触发，并检查其错误信息
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            mstats.spearmanr(x, y, alternative="ekki-ekki")

    @pytest.mark.skipif(platform.machine() == 'ppc64le',
                        reason="fails/crashes on ppc64le")
    @pytest.mark.skipif(platform.machine() == 'ppc64le',
                        reason="fails/crashes on ppc64le")
    @pytest.mark.slow
    def test_kendalltau_large(self):
        # 确保在处理较大数组时内部变量使用正确的精度
        x = np.arange(2000, dtype=float)
        x = ma.masked_greater(x, 1995)
        y = np.arange(2000, dtype=float)
        y = np.concatenate((y[1000:], y[:1000]))
        assert_(np.isfinite(mstats.kendalltau(x, y)[1]))

    def test_kendalltau_seasonal(self):
        # 测试季节性 Kendall tau 相关性
        x = [[nan, nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan],
             [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
        # 修正 x 中的无效值并转置
        x = ma.fix_invalid(x).T
        # 执行 mstats.kendalltau_seasonal 函数并断言结果
        output = mstats.kendalltau_seasonal(x)
        assert_almost_equal(output['global p-value (indep)'], 0.008, 3)
        assert_almost_equal(output['seasonal p-value'].round(2),
                            [0.18,0.53,0.20,0.04])
    # 使用 pytest.mark.parametrize 装饰器为测试方法添加参数化测试用例
    @pytest.mark.parametrize("method", ("exact", "asymptotic"))
    @pytest.mark.parametrize("alternative", ("two-sided", "greater", "less"))
    def test_kendalltau_mstats_vs_stats(self, method, alternative):
        # 测试 mstats.kendalltau 和 stats.kendalltau 在 nan_policy='omit' 下的行为是否匹配 stats.kendalltau
        # 精度的替代方法在 stats/tests/test_stats.py 中进行测试

        # 设置随机种子
        np.random.seed(0)
        # 创建长度为 50 的随机数组 x 和 y
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        # 创建随机掩码，大于 0.5 的位置为 True
        mask = np.random.rand(n) > 0.5

        # 使用掩码创建掩码数组 x_masked 和 y_masked
        x_masked = ma.array(x, mask=mask)
        y_masked = ma.array(y, mask=mask)
        # 调用 mstats.kendalltau 计算带有掩码的 Kendall Tau 相关系数
        res_masked = mstats.kendalltau(
            x_masked, y_masked, method=method, alternative=alternative)

        # 压缩掩码数组 x_masked 和 y_masked
        x_compressed = x_masked.compressed()
        y_compressed = y_masked.compressed()
        # 调用 stats.kendalltau 计算压缩后的 Kendall Tau 相关系数
        res_compressed = stats.kendalltau(
            x_compressed, y_compressed, method=method, alternative=alternative)

        # 将 mask 位置的元素置为 NaN
        x[mask] = np.nan
        y[mask] = np.nan
        # 调用 stats.kendalltau 计算忽略 NaN 值的 Kendall Tau 相关系数
        res_nan = stats.kendalltau(
            x, y, method=method, nan_policy='omit', alternative=alternative)

        # 断言带有掩码的结果与压缩后的结果的近似性
        assert_allclose(res_masked, res_compressed)
        # 断言忽略 NaN 值的结果与压缩后的结果的近似性
        assert_allclose(res_nan, res_compressed)

    def test_kendall_p_exact_medium(self):
        # 测试中等样本大小（某些 n >= 171）下的 exact 方法
        # 使用 SymPy 生成的预期值进行测试

        expectations = {(100, 2393): 0.62822615287956040664,
                        (101, 2436): 0.60439525773513602669,
                        (170, 0): 2.755801935583541e-307,
                        (171, 0): 0.0,
                        (171, 1): 2.755801935583541e-307,
                        (172, 1): 0.0,
                        (200, 9797): 0.74753983745929675209,
                        (201, 9656): 0.40959218958120363618}
        # 遍历预期值字典
        for nc, expected in expectations.items():
            # 调用 _mstats_basic._kendall_p_exact 计算 Kendall Tau 的精确 p 值
            res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
            # 断言计算结果与预期值的近似性
            assert_almost_equal(res, expected)

    @pytest.mark.xslow
    def test_kendall_p_exact_large(self):
        # 测试大样本大小（n >= 171）下的 exact 方法
        # 使用 SymPy 生成的预期值进行测试

        expectations = {(400, 38965): 0.48444283672113314099,
                        (401, 39516): 0.66363159823474837662,
                        (800, 156772): 0.42265448483120932055,
                        (801, 157849): 0.53437553412194416236,
                        (1600, 637472): 0.84200727400323538419,
                        (1601, 630304): 0.34465255088058593946}

        # 遍历预期值字典
        for nc, expected in expectations.items():
            # 调用 _mstats_basic._kendall_p_exact 计算 Kendall Tau 的精确 p 值
            res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
            # 断言计算结果与预期值的近似性
            assert_almost_equal(res, expected)

    @skip_xp_invalid_arg
    # mstats.pointbiserialr 返回一个 NumPy 浮点数作为统计量，但在调用 `special.betainc` 之前会将其转换为无掩码元素的掩码数组，
    # 当 `SCIPY_ARRAY_API=1` 时，`special.betainc` 不会接受掩码数组。
    # 定义测试函数 test_pointbiserial，用于测试 pointbiserialr 函数
    def test_pointbiserial(self):
        # 定义变量 x，包含一组二进制值和一个异常值
        x = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1]
        # 定义变量 y，包含一组浮点数和一个 NaN 值
        y = [14.8, 13.8, 12.4, 10.1, 7.1, 6.1, 5.8, 4.6, 4.3, 3.5, 3.3, 3.2,
             3.0, 2.8, 2.8, 2.5, 2.4, 2.3, 2.1, 1.7, 1.7, 1.5, 1.3, 1.3, 1.2,
             1.2, 1.1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1, np.nan]
        # 断言 pointbiserialr 函数计算得到的相关系数接近于指定值
        assert_almost_equal(mstats.pointbiserialr(x, y)[0], 0.36149, 5)

        # 测试 namedtuple 的属性
        # 调用 pointbiserialr 函数，并将结果赋给变量 res
        res = mstats.pointbiserialr(x, y)
        # 定义 namedtuple 的属性列表
        attributes = ('correlation', 'pvalue')
        # 调用自定义函数 check_named_results，验证 res 的属性是否符合预期
        check_named_results(res, attributes, ma=True)
# 装饰器，用于跳过某些测试函数，这里用于测试跳过测试中的无效参数
@skip_xp_invalid_arg
# 定义名为 TestTrimming 的测试类
class TestTrimming:

    # 定义名为 test_trim 的测试方法
    def test_trim(self):
        # 创建一个 MaskedArray 对象 a，包含从 0 到 9 的整数
        a = ma.arange(10)
        # 断言调用 mstats.trim 函数对 a 进行修剪后的结果与预期结果相等
        assert_equal(mstats.trim(a), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # 将 a 重新赋值为包含从 0 到 9 的整数的 MaskedArray 对象
        a = ma.arange(10)
        # 断言调用 mstats.trim 函数对 a 在指定范围 (2, 8) 进行修剪后的结果与预期结果相等
        assert_equal(mstats.trim(a, (2, 8)), [None, None, 2, 3, 4, 5, 6, 7, 8, None])
        # 将 a 重新赋值为包含从 0 到 9 的整数的 MaskedArray 对象
        a = ma.arange(10)
        # 断言调用 mstats.trim 函数对 a 在指定范围 (2, 8)、不包括边界的条件下进行修剪后的结果与预期结果相等
        assert_equal(mstats.trim(a, limits=(2, 8), inclusive=(False, False)),
                     [None, None, None, 3, 4, 5, 6, 7, None, None])
        # 将 a 重新赋值为包含从 0 到 9 的整数的 MaskedArray 对象
        a = ma.arange(10)
        # 断言调用 mstats.trim 函数对 a 在相对范围 (0.1, 0.2) 的条件下进行修剪后的结果与预期结果相等
        assert_equal(mstats.trim(a, limits=(0.1, 0.2), relative=True),
                     [None, 1, 2, 3, 4, 5, 6, 7, None, None])

        # 将 a 重新赋值为包含从 0 到 11 的整数的 MaskedArray 对象，并对部分元素进行遮蔽
        a = ma.arange(12)
        a[[0, -1]] = a[5] = masked
        # 断言调用 mstats.trim 函数对 a 在指定范围 (2, 8) 进行修剪后的结果与预期结果相等
        assert_equal(mstats.trim(a, (2, 8)),
                     [None, None, 2, 3, 4, None, 6, 7, 8, None, None, None])

        # 创建一个 10x10 的 MaskedArray 对象 x，其中包含从 0 到 99 的整数
        x = ma.arange(100).reshape(10, 10)
        # 创建一个预期的掩码列表，以指定的相对范围 (0.1, 0.2) 对 x 进行修剪
        expected = [1] * 10 + [0] * 70 + [1] * 20
        # 调用 mstats.trim 函数对 x 在相对范围 (0.1, 0.2) 进行修剪，并预期返回的掩码列表
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
        # 断言修剪后的结果的掩码与预期的掩码列表相等
        assert_equal(trimx._mask.ravel(), expected)
        # 同上，但是在 axis=0 上进行修剪
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        # 同上，但是在 axis=-1 上进行修剪
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=-1)
        assert_equal(trimx._mask.T.ravel(), expected)

        # 与上述相同，但是在 x 中插入了一个额外的遮蔽行
        x = ma.arange(110).reshape(11, 10)
        x[1] = masked
        # 创建一个预期的掩码列表，以指定的相对范围 (0.1, 0.2) 对 x 进行修剪
        expected = [1] * 20 + [0] * 70 + [1] * 20
        # 调用 mstats.trim 函数对 x 在相对范围 (0.1, 0.2) 进行修剪，并预期返回的掩码列表
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=None)
        # 断言修剪后的结果的掩码与预期的掩码列表相等
        assert_equal(trimx._mask.ravel(), expected)
        # 同上，但是在 axis=0 上进行修剪
        trimx = mstats.trim(x, (0.1, 0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        # 同上，但是在 axis=-1 上进行修剪，先对 x 进行转置
        trimx = mstats.trim(x.T, (0.1, 0.2), relative=True, axis=-1)
        assert_equal(trimx.T._mask.ravel(), expected)

    # 定义名为 test_trim_old 的测试方法
    def test_trim_old(self):
        # 创建一个包含从 0 到 99 的整数的 MaskedArray 对象 x
        x = ma.arange(100)
        # 断言调用 mstats.trimboth 函数对 x 进行修剪后的非遮蔽元素个数与预期结果相等
        assert_equal(mstats.trimboth(x).count(), 60)
        # 断言调用 mstats.trimtail 函数对 x 在尾部 'r' 进行修剪后的非遮蔽元素个数与预期结果相等
        assert_equal(mstats.trimtail(x, tail='r').count(), 80)
        # 将 x 的第 50 到 69 个元素设置为遮蔽
        x[50:70] = masked
        # 调用 mstats.trimboth 函数对 x 进行修剪
        trimx = mstats.trimboth(x)
        # 断言修剪后的非遮蔽元素个数与预期结果相等
        assert_equal(trimx.count(), 48)
        # 断言修剪后的遮蔽状态与预期的列表相等
        assert_equal(trimx._mask, [1]*16 + [0]*34 + [1]*20 + [0]*14 + [1]*16)
        # 清除 x 的遮蔽
        x._mask = nomask
        # 将 x 的形状改为 (10, 10)
        x.shape = (10, 10)
        # 断言调用 mstats.trimboth 函数对 x 进行修剪后的非遮蔽元素个数与预期结果相等
        assert_equal(mstats.trimboth(x).count(), 60)
        # 断言调用 mstats.trimtail 函数对 x 进行修剪后的非遮蔽元素个数与预期结果相等
        assert_equal(mstats.trimtail(x).count(), 80)

    # 定义名为 test_trimr 的测试方法
    def test_trimr(self):
        # 创建一个包含从 0 到 9 的整数的 MaskedArray 对象 x
        x = ma.arange(10)
        # 调用 mstats.trimr 函数对 x 在指定范围 (0.15, 0.14)、不包括边界的条件下进行修剪
        result = mstats.trimr(x, limits=(0.15, 0.14), inclusive=(False, False))
        # 创建一个预期的 MaskedArray 对象，其中包含修剪后的数据和
    # 定义一个测试函数，用于测试 mstats.trimmed_var 函数的功能
    def test_trimmedvar(self):
        # 基本测试。需要对所有参数、边界情况、输入验证以及对掩码数组的正确处理进行额外测试。
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(3262323289434724460)
        # 生成包含20个随机数的原始数据数组
        data_orig = rng.random(size=20)
        # 对原始数据数组进行排序，生成排序后的数据数组
        data = np.sort(data_orig)
        # 将数据数组包装成掩码数组，其中指定的索引位置被掩盖
        data = ma.array(data, mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # 断言修剪方差函数的结果与排序后数据数组的方差相等
        assert_allclose(mstats.trimmed_var(data_orig, 0.1), data.var())

    # 定义一个测试函数，用于测试 mstats.trimmed_std 函数的功能
    def test_trimmedstd(self):
        # 基本测试。需要对所有参数、边界情况、输入验证以及对掩码数组的正确处理进行额外测试。
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(7121029245207162780)
        # 生成包含20个随机数的原始数据数组
        data_orig = rng.random(size=20)
        # 对原始数据数组进行排序，生成排序后的数据数组
        data = np.sort(data_orig)
        # 将数据数组包装成掩码数组，其中指定的索引位置被掩盖
        data = ma.array(data, mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # 断言修剪标准差函数的结果与排序后数据数组的标准差相等
        assert_allclose(mstats.trimmed_std(data_orig, 0.1), data.std())

    # 定义一个测试函数，用于测试 mstats.trimmed_stde 函数的功能
    def test_trimmed_stde(self):
        # 创建一个包含掩码的数据数组
        data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
                         296,299,306,376,428,515,666,1310,2611])
        # 断言修剪标准误函数的结果与预期值在一定精度内相等
        assert_almost_equal(mstats.trimmed_stde(data,(0.2,0.2)), 56.13193, 5)
        # 断言修剪标准误函数的结果与预期值在一定精度内相等
        assert_almost_equal(mstats.trimmed_stde(data,0.2), 56.13193, 5)

    # 定义一个测试函数，用于测试 mstats.winsorize 函数的功能
    def test_winsorization(self):
        # 创建一个包含掩码的数据数组
        data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
                         296,299,306,376,428,515,666,1310,2611])
        # 断言修剪函数后的数据的方差与预期值在一定精度内相等
        assert_almost_equal(mstats.winsorize(data,(0.2,0.2)).var(ddof=1),
                            21551.4, 1)
        # 断言修剪函数后的数据的方差与预期值在一定精度内相等
        assert_almost_equal(
            mstats.winsorize(data, (0.2, 0.2),(False,False)).var(ddof=1),
            11887.3, 1)
        # 将数据数组中的第5个元素设为掩码值
        data[5] = masked
        # 对数据进行修剪，返回修剪后的数据数组
        winsorized = mstats.winsorize(data)
        # 断言修剪后的数组的掩码属性与原数据数组的掩码属性相等
        assert_equal(winsorized.mask, data.mask)

    # 定义一个测试函数，用于测试 mstats.winsorize 函数在处理 NaN 值时的功能
    def test_winsorization_nan(self):
        # 创建一个包含 NaN 值的数据数组
        data = ma.array([np.nan, np.nan, 0, 1, 2])
        # 断言在处理 NaN 值时，mstats.winsorize 函数会抛出 ValueError 异常
        assert_raises(ValueError, mstats.winsorize, data, (0.05, 0.05),
                      nan_policy='raise')
        # 测试默认情况下处理 NaN 值时的行为
        assert_equal(mstats.winsorize(data, (0.4, 0.4)),
                     ma.array([2, 2, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8)),
                     ma.array([np.nan, np.nan, np.nan, np.nan, np.nan]))
        assert_equal(mstats.winsorize(data, (0.4, 0.4), nan_policy='omit'),
                     ma.array([np.nan, np.nan, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8), nan_policy='omit'),
                     ma.array([np.nan, np.nan, 2, 2, 2]))
# 装饰器，用于跳过测试中的无效参数
@skip_xp_invalid_arg
# 定义一个测试类 TestMoments
class TestMoments:
    # 比较用的数值从 R v.1.5.1 中获得
    # 注意 testcase 的长度为 4
    # testmathworks 来自于 Matlab 统计工具箱的文档，可以在以下链接找到:
    # https://www.mathworks.com/help/stats/kurtosis.html
    # https://www.mathworks.com/help/stats/skewness.html
    # 需要注意，这两个测试案例都来自这里。
    testcase = [1, 2, 3, 4]
    # 2D 测试用例，使用了 ma.fix_invalid 进行修正
    testcase_2d = ma.array(
        np.array([[0.05245846, 0.50344235, 0.86589117, 0.36936353, 0.46961149],
                  [0.11574073, 0.31299969, 0.45925772, 0.72618805, 0.75194407],
                  [0.67696689, 0.91878127, 0.09769044, 0.04645137, 0.37615733],
                  [0.05903624, 0.29908861, 0.34088298, 0.66216337, 0.83160998],
                  [0.64619526, 0.94894632, 0.27855892, 0.0706151, 0.39962917]]),
        mask=np.array([[True, False, False, True, False],
                       [True, True, True, False, True],
                       [False, False, False, False, False],
                       [True, True, True, True, True],
                       [False, False, True, False, False]], dtype=bool))

    # 断言函数，用于比较 actual 和 expect 是否相等
    def _assert_equal(self, actual, expect, *, shape=None, dtype=None):
        # 将 expect 转换为 NumPy 数组
        expect = np.asarray(expect)
        # 如果指定了 shape，则广播 expect 到相应形状
        if shape is not None:
            expect = np.broadcast_to(expect, shape)
        # 使用 assert_array_equal 断言 actual 和 expect 相等
        assert_array_equal(actual, expect)
        # 如果未指定 dtype，则使用 expect 的 dtype
        if dtype is None:
            dtype = expect.dtype
        # 断言 actual 的 dtype 与指定的 dtype 相等
        assert actual.dtype == dtype
    # 测试 mstats 库中的 moment 函数
    def test_moment(self):
        # 计算一阶矩
        y = mstats.moment(self.testcase, 1)
        # 断言一阶矩近似为 0.0，允许误差为 10 的小数位
        assert_almost_equal(y, 0.0, 10)
        # 计算二阶矩
        y = mstats.moment(self.testcase, 2)
        # 断言二阶矩精确为 1.25
        assert_almost_equal(y, 1.25)
        # 计算三阶矩
        y = mstats.moment(self.testcase, 3)
        # 断言三阶矩近似为 0.0
        assert_almost_equal(y, 0.0)
        # 计算四阶矩
        y = mstats.moment(self.testcase, 4)
        # 断言四阶矩精确为 2.5625
        assert_almost_equal(y, 2.5625)

        # 检查 moment 函数接受 array_like 类型的输入
        y = mstats.moment(self.testcase, [1, 2, 3, 4])
        # 断言结果与期望的数组接近
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # 检查 moment 函数输入只能是整数
        y = mstats.moment(self.testcase, 0.0)
        # 断言结果接近 1.0
        assert_allclose(y, 1.0)
        # 断言当输入不是整数时抛出 ValueError 异常
        assert_raises(ValueError, mstats.moment, self.testcase, 1.2)
        y = mstats.moment(self.testcase, [1.0, 2, 3, 4.0])
        # 断言结果与期望的数组接近
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # 测试空输入的情况
        y = mstats.moment([])
        # 使用自定义的断言方法，断言结果为 NaN，数据类型为 np.float64
        self._assert_equal(y, np.nan, dtype=np.float64)
        y = mstats.moment(np.array([], dtype=np.float32))
        # 使用自定义的断言方法，断言结果为 NaN，数据类型为 np.float32
        self._assert_equal(y, np.nan, dtype=np.float32)
        y = mstats.moment(np.zeros((1, 0)), axis=0)
        # 使用自定义的断言方法，断言结果为一个空列表，数据类型为 np.float64，形状为 (0,)
        self._assert_equal(y, [], shape=(0,), dtype=np.float64)
        y = mstats.moment([[]], axis=1)
        # 使用自定义的断言方法，断言结果为 NaN，形状为 (1,)，数据类型为 np.float64
        self._assert_equal(y, np.nan, shape=(1,), dtype=np.float64)
        y = mstats.moment([[]], moment=[0, 1], axis=0)
        # 使用自定义的断言方法，断言结果为一个空列表，形状为 (2, 0)
        self._assert_equal(y, [], shape=(2, 0))

        x = np.arange(10.)
        x[9] = np.nan
        # 断言当输入包含 NaN 值时，moment 函数返回 ma.masked
        assert_equal(mstats.moment(x, 2), ma.masked)  # NaN value is ignored

    # 测试 mstats 库中的 variation 函数
    def test_variation(self):
        # 计算 variation
        y = mstats.variation(self.testcase)
        # 断言结果近似为 0.44721359549996，允许误差为 10 的小数位
        assert_almost_equal(y, 0.44721359549996, 10)

    # 测试 mstats 库中的 variation 函数，带有自由度修正参数 ddof
    def test_variation_ddof(self):
        # regression test for gh-13341，测试带有自由度修正参数的 variation 函数
        a = np.array([1, 2, 3, 4, 5])
        y = mstats.variation(a, ddof=1)
        # 断言结果近似为 0.5270462766947299
        assert_almost_equal(y, 0.5270462766947299)
    # 定义一个测试方法，用于测试偏度计算函数的准确性
    def test_skewness(self):
        # 计算一维数组 self.testmathworks 的偏度，与预期值比较精确度
        y = mstats.skew(self.testmathworks)
        assert_almost_equal(y, -0.29322304336607, 10)
        
        # 使用指定的偏差参数计算 self.testmathworks 的偏度，与预期值比较精确度
        y = mstats.skew(self.testmathworks, bias=0)
        assert_almost_equal(y, -0.437111105023940, 10)
        
        # 计算另一组一维数组 self.testcase 的偏度，与预期值比较精确度
        y = mstats.skew(self.testcase)
        assert_almost_equal(y, 0.0, 10)

        # 测试多维遮蔽数组的偏度计算功能
        
        # 定义一个预期的二维遮蔽数组，包括数据和遮蔽信息
        correct_2d = ma.array(
            np.array([0.6882870394455785, 0, 0.2665647526856708,
                      0, -0.05211472114254485]),
            mask=np.array([False, False, False, True, False], dtype=bool)
        )
        # 检查多维数组 self.testcase_2d 按行计算偏度，与预期结果比较
        assert_allclose(mstats.skew(self.testcase_2d, 1), correct_2d)
        for i, row in enumerate(self.testcase_2d):
            # 逐行计算偏度，与预期结果比较
            assert_almost_equal(mstats.skew(row), correct_2d[i])

        # 使用不偏估计计算多维数组 self.testcase_2d 的偏度
        
        # 定义一个带不偏估计的预期二维遮蔽数组，包括数据和遮蔽信息
        correct_2d_bias_corrected = ma.array(
            np.array([1.685952043212545, 0.0, 0.3973712716070531, 0,
                      -0.09026534484117164]),
            mask=np.array([False, False, False, True, False], dtype=bool)
        )
        # 检查多维数组 self.testcase_2d 按行计算不偏估计的偏度，与预期结果比较
        assert_allclose(mstats.skew(self.testcase_2d, 1, bias=False),
                        correct_2d_bias_corrected)
        for i, row in enumerate(self.testcase_2d):
            # 逐行计算不偏估计的偏度，与预期结果比较
            assert_almost_equal(mstats.skew(row, bias=False),
                                correct_2d_bias_corrected[i])

        # 检查统计和 mstats 实现之间的一致性，比较单行数据的偏度计算结果
        assert_allclose(mstats.skew(self.testcase_2d[2, :]),
                        stats.skew(self.testcase_2d[2, :]))
    # 定义一个测试函数，测试峰度计算
    def test_kurtosis(self):
        # 设置参数：axis=0（对列进行计算），fisher=0（使用Pearson的峰度定义，与Matlab兼容）
        y = mstats.kurtosis(self.testmathworks, 0, fisher=0, bias=1)
        # 断言计算结果与预期值的接近程度，精确到小数点后10位
        assert_almost_equal(y, 2.1658856802973, 10)
        
        # 注意：MATLAB对以下情况的文档描述可能令人困惑：
        # kurtosis(x,0) 给出Pearson偏度的无偏估计
        # kurtosis(x) 给出Fisher偏度的有偏估计（即Pearson-3）
        # MATLAB文档暗示两者都应该给出Fisher偏度
        y = mstats.kurtosis(self.testmathworks, fisher=0, bias=0)
        # 断言计算结果与预期值的接近程度，精确到小数点后10位
        assert_almost_equal(y, 3.663542721189047, 10)
        
        # 计算峰度，axis=0，bias=0
        y = mstats.kurtosis(self.testcase, 0, 0)
        # 断言计算结果与预期值的接近程度
        assert_almost_equal(y, 1.64)

        # 测试多维掩码数组的峰度计算
        correct_2d = ma.array(np.array([-1.5, -3., -1.47247052385, 0.,
                                        -1.26979517952]),
                              mask=np.array([False, False, False, True,
                                             False], dtype=bool))
        # 断言多维数组按行计算峰度的结果与预期值的接近程度
        assert_array_almost_equal(mstats.kurtosis(self.testcase_2d, 1),
                                  correct_2d)
        # 对于每一行，断言按行计算峰度的结果与预期值的接近程度
        for i, row in enumerate(self.testcase_2d):
            assert_almost_equal(mstats.kurtosis(row), correct_2d[i])

        # 使用无偏校正计算多维数组的峰度
        correct_2d_bias_corrected = ma.array(
            np.array([-1.5, -3., -1.88988209538, 0., -0.5234638463918877]),
            mask=np.array([False, False, False, True, False], dtype=bool))
        # 断言使用无偏校正按行计算峰度的结果与预期值的接近程度
        assert_array_almost_equal(mstats.kurtosis(self.testcase_2d, 1,
                                                  bias=False),
                                  correct_2d_bias_corrected)
        # 对于每一行，断言使用无偏校正按行计算峰度的结果与预期值的接近程度
        for i, row in enumerate(self.testcase_2d):
            assert_almost_equal(mstats.kurtosis(row, bias=False),
                                correct_2d_bias_corrected[i])

        # 检查统计与mstats实现之间的一致性
        assert_array_almost_equal_nulp(mstats.kurtosis(self.testcase_2d[2, :]),
                                       stats.kurtosis(self.testcase_2d[2, :]),
                                       nulp=4)
class TestMode:
    def test_mode(self):
        # 创建一个包含重复元素的列表
        a1 = [0,0,0,1,1,1,2,3,3,3,3,4,5,6,7]
        # 将a1重塑为3行5列的NumPy数组
        a2 = np.reshape(a1, (3,5))
        # 创建一个包含整数的NumPy数组
        a3 = np.array([1,2,3,4,5,6])
        # 将a3重塑为3行2列的NumPy数组
        a4 = np.reshape(a3, (3,2))
        # 创建一个掩码数组，根据条件屏蔽元素
        ma1 = ma.masked_where(ma.array(a1) > 2, a1)
        # 创建一个掩码数组，根据条件屏蔽元素
        ma2 = ma.masked_where(a2 > 2, a2)
        # 创建一个掩码数组，根据条件屏蔽元素
        ma3 = ma.masked_where(a3 < 2, a3)
        # 创建一个掩码数组，根据条件屏蔽元素
        ma4 = ma.masked_where(ma.array(a4) < 2, a4)
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a1, axis=None), (3,4))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a1, axis=0), (3,4))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma1, axis=None), (0,3))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a2, axis=None), (3,4))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma2, axis=None), (0,3))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a3, axis=None), (1,1))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma3, axis=None), (2,1))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a2, axis=0), ([[0,0,0,1,1]], [[1,1,1,1,1]]))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma2, axis=0), ([[0,0,0,1,1]], [[1,1,1,1,1]]))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(a2, axis=-1), ([[0],[3],[3]], [[3],[3],[1]]))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma2, axis=-1), ([[0],[1],[0]], [[3],[1],[0]]))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma4, axis=0), ([[3,2]], [[1,1]]))
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.mode(ma4, axis=-1), ([[2],[3],[5]], [[1],[1],[1]]))

        # 调用函数并获取返回值
        a1_res = mstats.mode(a1, axis=None)

        # 对返回的命名元组进行属性检查
        attributes = ('mode', 'count')
        check_named_results(a1_res, attributes, ma=True)

    def test_mode_modifies_input(self):
        # 回归测试：mode(..., axis=None) 不应修改输入数组
        # 创建一个全零数组
        im = np.zeros((100, 100))
        # 修改数组的部分元素值
        im[:50, :] += 1
        im[:, :50] += 1
        # 复制数组的副本
        cp = im.copy()
        # 调用函数不应修改输入数组
        mstats.mode(im, None)
        # 断言函数未修改输入数组
        assert_equal(im, cp)


class TestPercentile:
    def setup_method(self):
        # 初始化测试数据
        self.a1 = [3, 4, 5, 10, -3, -5, 6]
        self.a2 = [3, -6, -2, 8, 7, 4, 2, 1]
        self.a3 = [3., 4, 5, 10, -3, -5, -6, 7.0]

    def test_percentile(self):
        # 创建一个NumPy数组
        x = np.arange(8) * 0.5
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.scoreatpercentile(x, 0), 0.)
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.scoreatpercentile(x, 100), 3.5)
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.scoreatpercentile(x, 50), 1.75)

    def test_2D(self):
        # 创建一个掩码数组
        x = ma.array([[1, 1, 1],
                      [1, 1, 1],
                      [4, 4, 3],
                      [1, 1, 1],
                      [1, 1, 1]])
        # 断言函数的输出与预期结果相等
        assert_equal(mstats.scoreatpercentile(x, 50), [1, 1, 1])


@skip_xp_invalid_arg
class TestVariability:
    """  Comparison numbers are found using R v.1.5.1
         note that length(testcase) = 4
    """
    # 创建一个包含修复无效值后的数组
    testcase = ma.fix_invalid([1,2,3,4,np.nan])

    def test_sem(self):
        # 这个测试不在R语言中，所以用以下公式替代：sqrt(var(testcase)*3/4) / sqrt(3)
        y = mstats.sem(self.testcase)
        # 断言函数的输出与预期结果近似相等
        assert_almost_equal(y, 0.6454972244)
        # 计算数组中非掩码元素的数量
        n = self.testcase.count()
        # 断言函数的输出与预期结果非常接近
        assert_allclose(mstats.sem(self.testcase, ddof=0) * np.sqrt(n/(n-2)),
                        mstats.sem(self.testcase, ddof=2))
    def test_zmap(self):
        # 测试 mstats 库中的 zmap 函数
        # 这里的测试方法是用以下公式，而不是使用 R 中的方法：
        #    (testcase[i]-mean(testcase,axis=0)) / sqrt(var(testcase)*3/4)
        y = mstats.zmap(self.testcase, self.testcase)
        # 期望得到的未屏蔽数值
        desired_unmaskedvals = ([-1.3416407864999, -0.44721359549996,
                                 0.44721359549996, 1.3416407864999])
        # 使用 assert_array_almost_equal 函数检查结果的准确性
        assert_array_almost_equal(desired_unmaskedvals,
                                  y.data[y.mask == False], decimal=12)  # noqa: E712

    def test_zscore(self):
        # 测试 mstats 库中的 zscore 函数
        # 这里的测试方法是用以下公式，而不是使用 R 中的方法：
        #     (testcase[i]-mean(testcase,axis=0)) / sqrt(var(testcase)*3/4)
        y = mstats.zscore(self.testcase)
        # 期望得到的结果，包括修正无效值
        desired = ma.fix_invalid([-1.3416407864999, -0.44721359549996,
                                  0.44721359549996, 1.3416407864999, np.nan])
        # 使用 assert_almost_equal 函数检查结果的准确性
        assert_almost_equal(desired, y, decimal=12)
# 装饰器，用于跳过特定的测试
@skip_xp_invalid_arg
# 定义一个测试类 TestMisc
class TestMisc:

    # 测试 obrientransform 函数
    def test_obrientransform(self):
        # 定义参数 args 和期望结果 result
        args = [[5]*5+[6]*11+[7]*9+[8]*3+[9]*2+[10]*2,
                [6]+[7]*2+[8]*4+[9]*9+[10]*16]
        result = [5*[3.1828]+11*[0.5591]+9*[0.0344]+3*[1.6086]+2*[5.2817]+2*[11.0538],
                  [10.4352]+2*[4.8599]+4*[1.3836]+9*[0.0061]+16*[0.7277]]
        # 断言 obrientransform 函数的输出与期望结果 result 接近，精度为四位小数
        assert_almost_equal(np.round(mstats.obrientransform(*args).T, 4),
                            result, 4)

    # 测试 ks_2samp 函数
    def test_ks_2samp(self):
        # 定义包含 NaN 值的数据 x
        x = [[nan,nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan],
             [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
        # 使用 ma.fix_invalid 处理数据 x，转置后重新赋值给 x
        x = ma.fix_invalid(x).T
        # 分别将数据 x 的四个列赋值给 winter, spring, summer, fall
        (winter, spring, summer, fall) = x.T

        # 断言 ks_2samp 函数计算的两个样本 winter 和 spring 的 Kolmogorov-Smirnov 统计量和 p 值接近于 (0.1818, 0.9628)
        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring), 4),
                            (0.1818, 0.9628))
        # 断言 ks_2samp 函数计算的两个样本 winter 和 spring 的 Kolmogorov-Smirnov 统计量和 p 值接近于 (0.1469, 0.6886)，使用 'g' 表示两样本间较大的差距
        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring, 'g'), 4),
                            (0.1469, 0.6886))
        # 断言 ks_2samp 函数计算的两个样本 winter 和 spring 的 Kolmogorov-Smirnov 统计量和 p 值接近于 (0.1818, 0.6011)，使用 'l' 表示两样本间较小的差距
        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring, 'l'), 4),
                            (0.1818, 0.6011))

    # 测试 friedmanchisq 函数
    def test_friedmanchisq(self):
        # 定义不含缺失值的参数 args
        args = ([9.0,9.5,5.0,7.5,9.5,7.5,8.0,7.0,8.5,6.0],
                [7.0,6.5,7.0,7.5,5.0,8.0,6.0,6.5,7.0,7.0],
                [6.0,8.0,4.0,6.0,7.0,6.5,6.0,4.0,6.5,3.0])
        # 使用 mstats.friedmanchisquare 计算 args 的 Friedman 卡方值和 p 值
        result = mstats.friedmanchisquare(*args)
        # 断言计算结果的 Friedman 卡方值接近于 10.4737，精度为四位小数
        assert_almost_equal(result[0], 10.4737, 4)
        # 断言计算结果的 p 值接近于 0.005317，精度为六位小数
        assert_almost_equal(result[1], 0.005317, 6)

        # 定义包含缺失值的数据 x
        x = [[nan,nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan],
             [nan, 6, 11, 4, 17,nan, 6, 1, 1, 2, 5, 1, 1]]
        # 使用 ma.fix_invalid 处理数据 x
        x = ma.fix_invalid(x)
        # 使用 mstats.friedmanchisquare 计算修正后的数据 x 的 Friedman 卡方值和 p 值
        result = mstats.friedmanchisquare(*x)
        # 断言计算结果的 Friedman 卡方值接近于 2.0156，精度为四位小数
        assert_almost_equal(result[0], 2.0156, 4)
        # 断言计算结果的 p 值接近于 0.5692，精度为四位小数
        assert_almost_equal(result[1], 0.5692, 4)

        # 检查 namedtuple 的属性是否符合预期，期望的属性为 ('statistic', 'pvalue')
        attributes = ('statistic', 'pvalue')
        check_named_results(result, attributes, ma=True)

# 定义测试函数 test_regress_simple
def test_regress_simple():
    # 创建一个包含 0 到 100 的等差数列 x
    x = np.linspace(0, 100, 100)
    # 创建一个斜率为 0.2、截距为 10 的直线 y，并加上正弦噪声
    y = 0.2 * np.linspace(0, 100, 100) + 10
    y += np.sin(np.linspace(0, 20, 100))

    # 使用 mstats.linregress 对 x, y 进行线性回归
    result = mstats.linregress(x, y)

    # 断言结果 result 的类型为 _stats_py.LinregressResult 类型
    lr = _stats_py.LinregressResult
    assert_(isinstance(result, lr))
    # 检查结果 result 的命名属性是否符合预期，期望的属性为 ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
    attributes = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
    check_named_results(result, attributes, ma=True)
    # 断言结果 result 的截距 stderr 的属性存在
    assert 'intercept_stderr' in dir(result)

    # 断言结果 result 的斜率 slope、截距 intercept、标准误差 stderr、截距的标准误差 intercept_stderr 是否与预期值接近
    assert_almost_equal(result.slope, 0.19644990055858422)
    assert_almost_equal(result.intercept, 10.211269918932341)
    assert_almost_equal(result.stderr, 0.002395781449783862)
    assert_almost_equal(result.intercept_stderr, 0.13866936078570702)
def test_linregress_identical_x():
    # 创建一个包含 10 个零的 numpy 数组作为 x 值
    x = np.zeros(10)
    # 创建一个包含 10 个随机数的 numpy 数组作为 y 值
    y = np.random.random(10)
    # 定义一个错误消息，用于断言抛出 ValueError 异常
    msg = "Cannot calculate a linear regression if all x values are identical"
    # 使用 assert_raises 断言会抛出 ValueError 异常，并且异常消息与 msg 匹配
    with assert_raises(ValueError, match=msg):
        # 调用 mstats.linregress 计算 x 和 y 的线性回归
        mstats.linregress(x, y)


class TestTheilslopes:
    def test_theilslopes(self):
        # 测试基本的斜率和截距
        slope, intercept, lower, upper = mstats.theilslopes([0, 1, 1])
        assert_almost_equal(slope, 0.5)
        assert_almost_equal(intercept, 0.5)

        # 使用 'joint' 方法测试斜率和截距
        slope, intercept, lower, upper = mstats.theilslopes([0, 1, 1],
                                                            method='joint')
        assert_almost_equal(slope, 0.5)
        assert_almost_equal(intercept, 0.0)

        # 测试正确的屏蔽效果
        y = np.ma.array([0, 1, 100, 1], mask=[False, False, True, False])
        slope, intercept, lower, upper = mstats.theilslopes(y)
        assert_almost_equal(slope, 1./3)
        assert_almost_equal(intercept, 2./3)

        slope, intercept, lower, upper = mstats.theilslopes(y,
                                                            method='joint')
        assert_almost_equal(slope, 1./3)
        assert_almost_equal(intercept, 0.0)

        # 在 Sen (1968) 的示例中测试置信区间
        x = [1, 2, 3, 4, 10, 12, 18]
        y = [9, 15, 19, 20, 45, 55, 78]
        slope, intercept, lower, upper = mstats.theilslopes(y, x, 0.07)
        assert_almost_equal(slope, 4)
        assert_almost_equal(intercept, 4.0)
        assert_almost_equal(upper, 4.38, decimal=2)
        assert_almost_equal(lower, 3.71, decimal=2)

        slope, intercept, lower, upper = mstats.theilslopes(y, x, 0.07,
                                                            method='joint')
        assert_almost_equal(slope, 4)
        assert_almost_equal(intercept, 6.0)
        assert_almost_equal(upper, 4.38, decimal=2)
        assert_almost_equal(lower, 3.71, decimal=2)


    def test_theilslopes_warnings(self):
        # 使用 degenerate 输入测试 `theilslopes`；参见 gh-15943
        msg = "All `x` coordinates.*|Mean of empty slice.|invalid value encountered.*"
        # 断言会触发 RuntimeWarning，并且警告消息与 msg 匹配
        with pytest.warns(RuntimeWarning, match=msg):
            # 调用 mstats.theilslopes 对输入进行计算
            res = mstats.theilslopes([0, 1], [0, 0])
            # 断言结果数组中所有元素都是 NaN
            assert np.all(np.isnan(res))
        # 使用 suppress_warnings 上下文管理器，过滤特定的 RuntimeWarning
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered...")
            # 再次调用 mstats.theilslopes 进行计算
            res = mstats.theilslopes([0, 0, 0], [0, 1, 0])
            # 断言结果数组与给定值在允许误差内全等
            assert_allclose(res, (0, 0, np.nan, np.nan))
    def test_theilslopes_namedtuple_consistency(self):
        """
        Simple test to ensure tuple backwards-compatibility of the returned
        TheilslopesResult object
        """
        # 定义测试数据
        y = [1, 2, 4]
        x = [4, 6, 8]
        
        # 调用 theilslopes 函数并获取返回的斜率、截距、低斜率和高斜率
        slope, intercept, low_slope, high_slope = mstats.theilslopes(y, x)
        
        # 再次调用 theilslopes 函数获取返回结果对象
        result = mstats.theilslopes(y, x)

        # 断言确保返回的四个值与结果对象中的对应字段相等
        assert_equal(slope, result.slope)
        assert_equal(intercept, result.intercept)
        assert_equal(low_slope, result.low_slope)
        assert_equal(high_slope, result.high_slope)

    def test_gh19678_uint8(self):
        # 当 y 是无符号类型时，theilslopes 返回意外结果。验证此问题是否已解决。
        rng = np.random.default_rng(2549824598234528)
        
        # 生成一个无符号整数类型的随机数组 y
        y = rng.integers(0, 255, size=10, dtype=np.uint8)
        
        # 调用 theilslopes 函数计算斜率
        res = stats.theilslopes(y, y)
        
        # 使用 np.testing.assert_allclose 函数验证斜率的数值是否接近于 1
        np.testing.assert_allclose(res.slope, 1)
# 定义用于测试 `siegelslopes` 函数的测试函数
def test_siegelslopes():
    # 对于直线，方法应该是精确的
    y = 2 * np.arange(10) + 0.5
    # 断言使用 `siegelslopes` 计算斜率和截距，应该得到 (2.0, 0.5)
    assert_equal(mstats.siegelslopes(y), (2.0, 0.5))
    # 同样的测试，指定方法为 'separate'
    assert_equal(mstats.siegelslopes(y, method='separate'), (2.0, 0.5))

    x = 2 * np.arange(10)
    y = 5 * x - 3.0
    # 断言使用 `siegelslopes` 计算斜率和截距，应该得到 (5.0, -3.0)
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))
    # 同样的测试，指定方法为 'separate'
    assert_equal(mstats.siegelslopes(y, x, method='separate'), (5.0, -3.0))

    # 方法对异常值是鲁棒的：断裂点为 50%
    y[:4] = 1000
    # 断言使用 `siegelslopes` 计算斜率和截距，应该得到 (5.0, -3.0)，即使存在异常值
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))

    # 如果没有异常值，结果应该与 linregress 相当
    x = np.arange(10)
    y = -2.3 + 0.3*x + stats.norm.rvs(size=10, random_state=231)
    slope_ols, intercept_ols, _, _, _ = stats.linregress(x, y)

    slope, intercept = mstats.siegelslopes(y, x)
    # 使用 `assert_allclose` 检查斜率和截距是否与 linregress 的结果相似，相对误差不超过 0.1
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)

    slope, intercept = mstats.siegelslopes(y, x, method='separate')
    # 同样的比较，指定方法为 'separate'
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)


def test_siegelslopes_namedtuple_consistency():
    """
    简单测试以确保返回的 SiegelslopesResult 对象与元组的向后兼容性。
    """
    y = [1, 2, 4]
    x = [4, 6, 8]
    slope, intercept = mstats.siegelslopes(y, x)
    result = mstats.siegelslopes(y, x)

    # 注意这里返回的两个值是不同的
    assert_equal(slope, result.slope)
    assert_equal(intercept, result.intercept)


def test_sen_seasonal_slopes():
    rng = np.random.default_rng(5765986256978575148)
    x = rng.random(size=(100, 4))
    intra_slope, inter_slope = mstats.sen_seasonal_slopes(x)

    # 从 `sen_seasonal_slopes` 文档中的参考实现
    def dijk(yi):
        n = len(yi)
        x = np.arange(n)
        dy = yi - yi[:, np.newaxis]
        dx = x - x[:, np.newaxis]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return dy[mask]/dx[mask]

    for i in range(4):
        # 使用 `assert_allclose` 检查 `intra_slope` 是否与 `dijk` 函数计算的中位数接近
        assert_allclose(np.median(dijk(x[:, i])), intra_slope[i])

    all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])
    # 使用 `assert_allclose` 检查所有斜率的中位数是否接近 `inter_slope`
    assert_allclose(np.median(all_slopes), inter_slope)


def test_plotting_positions():
    # 对于 issue #1256 的回归测试
    pos = mstats.plotting_positions(np.arange(3), 0, 0)
    # 使用 `assert_array_almost_equal` 检查结果数组是否接近预期值 [0.25, 0.5, 0.75]
    assert_array_almost_equal(pos.data, np.array([0.25, 0.5, 0.75]))
    def test_vs_nonmasked(self):
        # 创建一个包含整数平方的 NumPy 数组
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        # 使用 mstats 中的 normaltest 函数和 stats 中的 normaltest 函数比较结果的近似性
        assert_array_almost_equal(mstats.normaltest(x),
                                  stats.normaltest(x))
        # 使用 mstats 中的 skewtest 函数和 stats 中的 skewtest 函数比较结果的近似性
        assert_array_almost_equal(mstats.skewtest(x),
                                  stats.skewtest(x))
        # 使用 mstats 中的 kurtosistest 函数和 stats 中的 kurtosistest 函数比较结果的近似性
        assert_array_almost_equal(mstats.kurtosistest(x),
                                  stats.kurtosistest(x))

        # 将 stats 中的 normaltest、skewtest 和 kurtosistest 函数放入列表 funcs
        funcs = [stats.normaltest, stats.skewtest, stats.kurtosistest]
        # 将 mstats 中的 normaltest、skewtest 和 kurtosistest 函数放入列表 mfuncs
        mfuncs = [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]
        # 创建一个简单的整数列表 x
        x = [1, 2, 3, 4]
        # 对于 funcs 和 mfuncs 中对应的每一对函数 func 和 mfunc
        for func, mfunc in zip(funcs, mfuncs):
            # 检查是否会触发 SmallSampleWarning 警告，且警告消息中包含 too_small_1d_not_omit
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                # 对 x 执行 func 函数，并检查结果中的 statistic 是否为 NaN
                res = func(x)
                assert np.isnan(res.statistic)
                # 检查结果中的 pvalue 是否为 NaN
                assert np.isnan(res.pvalue)
            # 检查对于 mfunc 函数对 x 执行时是否会引发 ValueError 异常
            assert_raises(ValueError, mfunc, x)

    def test_axis_None(self):
        # 测试 axis=None 的情况（对于 1-D 输入等同于 axis=0）
        x = np.array((-2,-1,0,1,2,3)*4)**2
        # 检查 mstats 中的 normaltest 函数在 axis=None 时与在默认 axis=0 时的结果是否全部近似
        assert_allclose(mstats.normaltest(x, axis=None), mstats.normaltest(x))
        # 检查 mstats 中的 skewtest 函数在 axis=None 时与在默认 axis=0 时的结果是否全部近似
        assert_allclose(mstats.skewtest(x, axis=None), mstats.skewtest(x))
        # 检查 mstats 中的 kurtosistest 函数在 axis=None 时与在默认 axis=0 时的结果是否全部近似
        assert_allclose(mstats.kurtosistest(x, axis=None),
                        mstats.kurtosistest(x))

    def test_maskedarray_input(self):
        # 添加一些掩码值，测试结果不应改变
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        # 创建一个包含掩码的掩码数组 xm
        xm = np.ma.array(np.r_[np.inf, x, 10],
                         mask=np.r_[True, [False] * x.size, True])
        # 检查使用 mstats 中的 normaltest 函数和 stats 中的 normaltest 函数比较结果的近似性
        assert_allclose(mstats.normaltest(xm), stats.normaltest(x))
        # 检查使用 mstats 中的 skewtest 函数和 stats 中的 skewtest 函数比较结果的近似性
        assert_allclose(mstats.skewtest(xm), stats.skewtest(x))
        # 检查使用 mstats 中的 kurtosistest 函数和 stats 中的 kurtosistest 函数比较结果的近似性
        assert_allclose(mstats.kurtosistest(xm), stats.kurtosistest(x))

    def test_nd_input(self):
        # 创建一个整数平方数组 x 和一个其转置的 2D 数组 x_2d
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        x_2d = np.vstack([x] * 2).T
        # 对于 mstats 中的 normaltest、skewtest 和 kurtosistest 函数中的每一个
        for func in [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]:
            # 对 x 执行 func 函数，并将结果分别存储为 res_1d 和 res_2d
            res_1d = func(x)
            res_2d = func(x_2d)
            # 检查 2D 结果中的第一个元素是否与 1D 结果的第一个元素的复制相等
            assert_allclose(res_2d[0], [res_1d[0]] * 2)
            # 检查 2D 结果中的第二个元素是否与 1D 结果的第二个元素的复制相等
            assert_allclose(res_2d[1], [res_1d[1]] * 2)

    def test_normaltest_result_attributes(self):
        # 创建一个整数平方数组 x
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        # 对 x 执行 mstats 中的 normaltest 函数，将结果存储为 res
        res = mstats.normaltest(x)
        # 定义需要检查的结果属性列表
        attributes = ('statistic', 'pvalue')
        # 使用自定义函数 check_named_results 检查 res 中的属性是否符合预期，包括支持掩码数组
        check_named_results(res, attributes, ma=True)

    def test_kurtosistest_result_attributes(self):
        # 创建一个整数平方数组 x
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        # 对 x 执行 mstats 中的 kurtosistest 函数，将结果存储为 res
        res = mstats.kurtosistest(x)
        # 定义需要检查的结果属性列表
        attributes = ('statistic', 'pvalue')
        # 使用自定义函数 check_named_results 检查 res 中的属性是否符合预期，包括支持掩码数组
        check_named_results(res, attributes, ma=True)

    def test_regression_9033(self):
        # 对 x 进行偏态检验，检查是否显著偏态
        counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
        x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
        # 检查 mstats 中的 kurtosistest 函数计算结果中的 p 值是否小于 0.01
        assert_equal(mstats.kurtosistest(x)[1] < 0.01, True)

    @pytest.mark.parametrize("test", ["skewtest", "kurtosistest"])
    # 使用 pytest 的参数化装饰器，为 test_alternative 方法提供两个参数化的测试用例：'less' 和 'greater'
    @pytest.mark.parametrize("alternative", ["less", "greater"])
    # 定义测试方法 test_alternative，接受参数 test 和 alternative
    def test_alternative(self, test, alternative):
        # 生成服从正态分布的随机数据，均值为 10，标准差为 2.5，共 30 个样本，随机种子为 123
        x = stats.norm.rvs(loc=10, scale=2.5, size=30, random_state=123)

        # 根据字符串 test 获取 scipy.stats 中的对应统计方法
        stats_test = getattr(stats, test)
        # 根据字符串 test 获取 scipy.stats.mstats 中的对应统计方法
        mstats_test = getattr(mstats, test)

        # 使用 scipy.stats 中的统计方法计算测试值和 p 值，使用 alternative 参数指定检验类型（'less' 或 'greater'）
        z_ex, p_ex = stats_test(x, alternative=alternative)
        # 使用 scipy.stats.mstats 中的统计方法计算测试值和 p 值，使用 alternative 参数指定检验类型（'less' 或 'greater'）
        z, p = mstats_test(x, alternative=alternative)

        # 断言两种方法计算得到的 z 值非常接近，允许误差为 1e-12
        assert_allclose(z, z_ex, atol=1e-12)
        # 断言两种方法计算得到的 p 值非常接近，允许误差为 1e-12
        assert_allclose(p, p_ex, atol=1e-12)

        # 测试带有掩码数组的情况
        # 将部分数据设为 NaN
        x[1:5] = np.nan
        # 创建掩码数组，将 NaN 值掩盖起来
        x = np.ma.masked_array(x, mask=np.isnan(x))
        # 使用 scipy.stats 中的统计方法计算掩码数组去除掩盖后的测试值和 p 值，使用 alternative 参数指定检验类型（'less' 或 'greater'）
        z_ex, p_ex = stats_test(x.compressed(), alternative=alternative)
        # 使用 scipy.stats.mstats 中的统计方法计算掩码数组的测试值和 p 值，使用 alternative 参数指定检验类型（'less' 或 'greater'）
        z, p = mstats_test(x, alternative=alternative)

        # 断言两种方法计算得到的 z 值非常接近，允许误差为 1e-12
        assert_allclose(z, z_ex, atol=1e-12)
        # 断言两种方法计算得到的 p 值非常接近，允许误差为 1e-12
        assert_allclose(p, p_ex, atol=1e-12)

    # 定义测试方法 test_bad_alternative，用于测试错误的 alternative 参数情况
    def test_bad_alternative(self):
        # 生成服从正态分布的随机数据，均值为 0，标准差为 1，共 20 个样本，随机种子为 123
        x = stats.norm.rvs(size=20, random_state=123)
        # 定义预期的错误消息内容
        msg = r"`alternative` must be..."

        # 使用 pytest 的上下文管理器检查 mstats.skewtest 方法是否会抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            mstats.skewtest(x, alternative='error')

        # 使用 pytest 的上下文管理器检查 mstats.kurtosistest 方法是否会抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            mstats.kurtosistest(x, alternative='error')
class TestFOneway:
    def test_result_attributes(self):
        # 创建包含 uint16 数据的 NumPy 数组 a
        a = np.array([655, 788], dtype=np.uint16)
        # 创建包含 uint16 数据的 NumPy 数组 b
        b = np.array([789, 772], dtype=np.uint16)
        # 使用 scipy.stats.mstats 中的 f_oneway 函数计算单向方差分析结果
        res = mstats.f_oneway(a, b)
        # 指定需要检查的结果属性为 statistic 和 pvalue
        attributes = ('statistic', 'pvalue')
        # 调用自定义函数 check_named_results，检查 res 的属性是否包含指定的属性名
        check_named_results(res, attributes, ma=True)


class TestMannwhitneyu:
    # data from gh-1428
    # 创建包含数据的 NumPy 数组 x，数据来自 gh-1428
    x = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 2., 1., 1., 2.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1.])

    # 创建包含数据的 NumPy 数组 y，数据来自 gh-1428
    y = np.array([1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1., 1., 1., 1.,
                  2., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2., 1., 1., 3.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1.,
                  1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2.,
                  2., 1., 1., 2., 1., 1., 2., 1., 2., 1., 1., 1., 1., 2.,
                  2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 2., 1., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  2., 1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 2., 1., 1.,
                  1., 1., 1., 1.])

    def test_result_attributes(self):
        # 使用 scipy.stats.mstats 中的 mannwhitneyu 函数计算 Mann-Whitney U 检验结果
        res = mstats.mannwhitneyu(self.x, self.y)
        # 指定需要检查的结果属性为 statistic 和 pvalue
        attributes = ('statistic', 'pvalue')
        # 调用自定义函数 check_named_results，检查 res 的属性是否包含指定的属性名
        check_named_results(res, attributes, ma=True)
    def test_against_stats(self):
        # 通过 gh-4641 报告，发现 stats.mannwhitneyu 返回的 p 值是 mstats.mannwhitneyu 的一半
        # 现在 stats.mannwhitneyu 的默认 alternative 参数为双侧，因此它们的结果现在一致了。
        # 使用 mstats.mannwhitneyu 对象计算非参数 Mann-Whitney U 检验的统计量和 p 值
        res1 = mstats.mannwhitneyu(self.x, self.y)
        # 使用 stats.mannwhitneyu 对象计算非参数 Mann-Whitney U 检验的统计量和 p 值
        res2 = stats.mannwhitneyu(self.x, self.y)
        # 断言两种方法计算的统计量相等
        assert res1.statistic == res2.statistic
        # 断言两种方法计算的 p 值在接受的误差范围内相等
        assert_allclose(res1.pvalue, res2.pvalue)
class TestKruskal:
    # 定义测试方法，验证 Kruskal-Wallis 检验的结果属性
    def test_result_attributes(self):
        # 创建两个样本数据
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]

        # 执行 Kruskal-Wallis 检验，返回结果对象
        res = mstats.kruskal(x, y)
        
        # 定义结果对象应包含的属性
        attributes = ('statistic', 'pvalue')

        # 调用自定义函数检查结果对象是否包含指定属性
        check_named_results(res, attributes, ma=True)


# TODO: for all ttest functions, add tests with masked array inputs
class TestTtest_rel:
    # 测试相依样本 t 检验与非掩码输入的比较
    def test_vs_nonmasked(self):
        # 设置随机种子以保证可复现性
        np.random.seed(1234567)
        # 生成服从标准正态分布的随机数据
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 对 1-D 输入进行 t 检验
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1])
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        assert_allclose(res1, res2)

        # 对 2-D 输入进行 t 检验，指定 axis=None
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)
        res1 = stats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2)

        # 检查默认 axis=0 的情况
        res3 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

    # 测试完全掩码输入的情况
    def test_fully_masked(self):
        np.random.seed(1234567)
        # 创建一个完全掩码的掩码数组
        outcome = ma.masked_array(np.random.randn(3, 2),
                                  mask=[[1, 1, 1], [0, 0, 0]])
        with suppress_warnings() as sup:
            # 忽略运行时警告，如"在计算中遇到无效值"
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            for pair in [(outcome[:, 0], outcome[:, 1]),
                         ([np.nan, np.nan], [1.0, 2.0])]:
                # 执行 t 检验，期望结果为 NaN
                t, p = mstats.ttest_rel(*pair)
                assert_array_equal(t, (np.nan, np.nan))
                assert_array_equal(p, (np.nan, np.nan))

    # 测试结果属性的验证
    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 执行相依样本 t 检验
        res = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        attributes = ('statistic', 'pvalue')

        # 调用自定义函数检查结果对象是否包含指定属性
        check_named_results(res, attributes, ma=True)

    # 测试无效输入尺寸的情况
    def test_invalid_input_size(self):
        # 断言输入为不同尺寸的数组时会引发 ValueError
        assert_raises(ValueError, mstats.ttest_rel,
                      np.arange(10), np.arange(11))
        x = np.arange(24)
        assert_raises(ValueError, mstats.ttest_rel,
                      x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=1)
        assert_raises(ValueError, mstats.ttest_rel,
                      x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=2)

    # 测试空输入的情况
    def test_empty(self):
        # 对空输入进行 t 检验，预期结果全部为 NaN
        res1 = mstats.ttest_rel([], [])
        assert_(np.all(np.isnan(res1)))

    # 测试除零情况
    def test_zero_division(self):
        # 执行独立样本 t 检验，比较处理全零输入和全不同输入的情况
        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1])
        assert_equal((np.abs(t), p), (np.inf, 0))

        with suppress_warnings() as sup:
            # 忽略运行时警告，如"在计算中遇到无效值"
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            t, p = mstats.ttest_ind([0, 0, 0], [0, 0, 0])
            assert_array_equal(t, np.array([np.nan, np.nan]))
            assert_array_equal(p, np.array([np.nan, np.nan]))
    # 定义一个测试方法，用于测试当 alternative 参数传递不合法值时是否抛出 ValueError 异常
    def test_bad_alternative(self):
        # 设置预期的异常消息
        msg = r"alternative must be 'less', 'greater' or 'two-sided'"
        # 使用 pytest 模块的 raises 方法验证是否抛出指定异常，并匹配预期的异常消息
        with pytest.raises(ValueError, match=msg):
            # 调用 mstats.ttest_ind 执行 t 检验，alternative 参数传递 'foo'（非法值）
            mstats.ttest_ind([1, 2, 3], [4, 5, 6], alternative='foo')

    # 使用 pytest.mark.parametrize 注解标记的测试方法，针对不同的 alternative 参数进行多组参数化测试
    @pytest.mark.parametrize("alternative", ["less", "greater"])
    # 定义 t 检验的测试方法，测试相关性 t 检验的功能
    def test_alternative(self, alternative):
        # 生成符合正态分布的随机数据 x 和 y
        x = stats.norm.rvs(loc=10, scale=5, size=25, random_state=42)
        y = stats.norm.rvs(loc=8, scale=2, size=25, random_state=42)

        # 使用 scipy.stats 下的 ttest_rel 函数执行相关性 t 检验，指定 alternative 参数
        t_ex, p_ex = stats.ttest_rel(x, y, alternative=alternative)
        # 使用 mstats 下的 ttest_rel 函数执行相关性 t 检验，指定 alternative 参数
        t, p = mstats.ttest_rel(x, y, alternative=alternative)
        # 断言两个实现的 t 值和 p 值在指定的相对误差范围内一致
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        # 测试使用掩码数组（masked arrays）
        # 将 x 和 y 中的部分数据置为 NaN
        x[1:10] = np.nan
        y[1:10] = np.nan
        # 创建掩码数组，将 NaN 值掩盖起来
        x = np.ma.masked_array(x, mask=np.isnan(x))
        y = np.ma.masked_array(y, mask=np.isnan(y))
        # 使用 mstats 下的 ttest_rel 函数执行相关性 t 检验，指定 alternative 参数
        t, p = mstats.ttest_rel(x, y, alternative=alternative)
        # 对比使用掩码后的 t 和 p 值与未使用掩码时的 t 和 p 值
        t_ex, p_ex = stats.ttest_rel(x.compressed(), y.compressed(),
                                     alternative=alternative)
        # 断言两组 t 和 p 值在指定的相对误差范围内一致
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)
class TestTtest_ind:
    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 1-D inputs
        # 对于一维输入，使用 scipy.stats 的 ttest_ind 进行独立双样本 t 检验
        res1 = stats.ttest_ind(outcome[:, 0], outcome[:, 1])
        res2 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1])
        assert_allclose(res1, res2)

        # 2-D inputs
        # 对于二维输入，设置 axis=None，进行整体独立双样本 t 检验
        res1 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)
        # 沿着 axis=0 进行独立双样本 t 检验
        res1 = stats.ttest_ind(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_ind(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2)

        # 检查默认 axis=0 的情况
        res3 = mstats.ttest_ind(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

        # 检查 equal_var 参数为 True 的情况
        res4 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=True)
        res5 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=True)
        assert_allclose(res4, res5)
        # 检查 equal_var 参数为 False 的情况
        res4 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=False)
        res5 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=False)
        assert_allclose(res4, res5)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3, 2), mask=[[1, 1, 1], [0, 0, 0]])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            for pair in [(outcome[:, 0], outcome[:, 1]),
                         ([np.nan, np.nan], [1.0, 2.0])]:
                # 使用 mstats.ttest_ind 处理完全被屏蔽的数据
                t, p = mstats.ttest_ind(*pair)
                assert_array_equal(t, (np.nan, np.nan))
                assert_array_equal(p, (np.nan, np.nan))

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 检查返回结果的属性
        res = mstats.ttest_ind(outcome[:, 0], outcome[:, 1])
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        # 处理空输入的情况
        res1 = mstats.ttest_ind([], [])
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        # 处理除以零的情况
        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1])
        assert_equal((np.abs(t), p), (np.inf, 0))

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            # 处理计算绝对值时出现无效值的情况
            t, p = mstats.ttest_ind([0, 0, 0], [0, 0, 0])
            assert_array_equal(t, (np.nan, np.nan))
            assert_array_equal(p, (np.nan, np.nan))

        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1], equal_var=False)
        assert_equal((np.abs(t), p), (np.inf, 0))
        assert_array_equal(mstats.ttest_ind([0, 0, 0], [0, 0, 0],
                                            equal_var=False), (np.nan, np.nan))
    # 定义一个测试方法，用于测试当 alternative 参数不合法时是否抛出 ValueError 异常
    def test_bad_alternative(self):
        # 定义错误消息，用于匹配异常信息
        msg = r"alternative must be 'less', 'greater' or 'two-sided'"
        # 使用 pytest 的断言检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 mstats.ttest_ind 函数，传入不合法的 alternative 参数 'foo'
            mstats.ttest_ind([1, 2, 3], [4, 5, 6], alternative='foo')

    # 使用 pytest 的参数化装饰器，定义一个参数化测试方法，用于测试合法的 alternative 参数
    @pytest.mark.parametrize("alternative", ["less", "greater"])
    def test_alternative(self, alternative):
        # 生成两组服从正态分布的随机样本数据
        x = stats.norm.rvs(loc=10, scale=2, size=100, random_state=123)
        y = stats.norm.rvs(loc=8, scale=2, size=100, random_state=123)

        # 使用 scipy.stats.ttest_ind 计算给定 alternative 参数下的 t 统计量和 p 值
        t_ex, p_ex = stats.ttest_ind(x, y, alternative=alternative)
        # 使用 mstats.ttest_ind 函数计算给定 alternative 参数下的 t 统计量和 p 值
        t, p = mstats.ttest_ind(x, y, alternative=alternative)
        # 使用 numpy.testing.assert_allclose 断言函数检查两种方法计算得到的 t 和 p 值的近似性
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)

        # 测试使用掩码数组的情况
        # 将 x 和 y 数组的部分元素设置为 NaN
        x[1:10] = np.nan
        y[80:90] = np.nan
        # 创建掩码数组，屏蔽 NaN 元素
        x = np.ma.masked_array(x, mask=np.isnan(x))
        y = np.ma.masked_array(y, mask=np.isnan(y))
        # 使用 scipy.stats.ttest_ind 计算掩码数组下的 t 统计量和 p 值
        t_ex, p_ex = stats.ttest_ind(x.compressed(), y.compressed(),
                                     alternative=alternative)
        # 使用 mstats.ttest_ind 函数计算掩码数组下的 t 统计量和 p 值
        t, p = mstats.ttest_ind(x, y, alternative=alternative)
        # 使用 numpy.testing.assert_allclose 断言函数检查两种方法计算得到的 t 和 p 值的近似性
        assert_allclose(t, t_ex, rtol=1e-14)
        assert_allclose(p, p_ex, rtol=1e-14)
class TestTtest_1samp:
    def test_vs_nonmasked(self):
        # 设置随机数种子，以便结果可重复
        np.random.seed(1234567)
        # 生成服从标准正态分布的随机数据，形状为 (20, 4)，并偏移 [0, 0, 1, 2]
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 对第一列数据进行单样本 t 检验
        res1 = stats.ttest_1samp(outcome[:, 0], 1)
        # 使用 mstats 中的函数进行单样本 t 检验
        res2 = mstats.ttest_1samp(outcome[:, 0], 1)
        # 断言两个结果的近似相等
        assert_allclose(res1, res2)

    def test_fully_masked(self):
        # 设置随机数种子，以便结果可重复
        np.random.seed(1234567)
        # 创建一个完全遮蔽的掩码数组，形状为 (3)
        outcome = ma.masked_array(np.random.randn(3), mask=[1, 1, 1])
        # 期望的结果为 (NaN, NaN)
        expected = (np.nan, np.nan)
        # 使用警告抑制上下文
        with suppress_warnings() as sup:
            # 过滤特定的运行时警告信息
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            # 对两个测试对 (np.nan, np.nan) 和 (outcome, 0.0) 分别进行单样本 t 检验
            for pair in [((np.nan, np.nan), 0.0), (outcome, 0.0)]:
                # 执行单样本 t 检验，返回统计量 t 和 p 值 p
                t, p = mstats.ttest_1samp(*pair)
                # 断言 p 值近似等于期望值
                assert_array_equal(p, expected)
                # 断言统计量 t 近似等于期望值
                assert_array_equal(t, expected)

    def test_result_attributes(self):
        # 设置随机数种子，以便结果可重复
        np.random.seed(1234567)
        # 生成服从标准正态分布的随机数据，形状为 (20, 4)，并偏移 [0, 0, 1, 2]
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 对第一列数据进行单样本 t 检验，返回统计量 t 和 p 值 p
        res = mstats.ttest_1samp(outcome[:, 0], 1)
        # 定义要检查的结果属性
        attributes = ('statistic', 'pvalue')
        # 检查返回的结果是否包含指定的属性，并且支持遮蔽数组
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        # 对空列表进行单样本 t 检验
        res1 = mstats.ttest_1samp([], 1)
        # 断言结果中所有值都为 NaN
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        # 对全零列表进行单样本 t 检验
        t, p = mstats.ttest_1samp([0, 0, 0], 1)
        # 断言 t 的绝对值为无穷大，p 值为 0
        assert_equal((np.abs(t), p), (np.inf, 0))

        # 使用警告抑制上下文
        with suppress_warnings() as sup:
            # 过滤特定的运行时警告信息
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            # 对全零列表进行单样本 t 检验，但指定的值为 0
            t, p = mstats.ttest_1samp([0, 0, 0], 0)
            # 断言 t 的值为 NaN
            assert_(np.isnan(t))
            # 断言 p 值的数组等于 (NaN, NaN)
            assert_array_equal(p, (np.nan, np.nan))

    def test_bad_alternative(self):
        # 定义错误的备择假设信息
        msg = r"alternative must be 'less', 'greater' or 'two-sided'"
        # 使用 pytest 框架断言引发特定异常信息
        with pytest.raises(ValueError, match=msg):
            # 调用单样本 t 检验函数，指定错误的备择假设
            mstats.ttest_1samp([1, 2, 3], 4, alternative='foo')

    @pytest.mark.parametrize("alternative", ["less", "greater"])
    def test_alternative(self, alternative):
        # 生成服从正态分布的随机数据，均值为 10，标准差为 2，大小为 100，以便结果可重复
        x = stats.norm.rvs(loc=10, scale=2, size=100, random_state=123)

        # 使用 stats 中的单样本 t 检验函数计算参考值
        t_ex, p_ex = stats.ttest_1samp(x, 9, alternative=alternative)
        # 使用 mstats 中的单样本 t 检验函数计算测试值
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        # 断言测试值的 t 统计量近似等于参考值
        assert_allclose(t, t_ex, rtol=1e-14)
        # 断言测试值的 p 值近似等于参考值
        assert_allclose(p, p_ex, rtol=1e-14)

        # 将部分数据设置为 NaN，创建遮蔽数组
        x[1:10] = np.nan
        x = np.ma.masked_array(x, mask=np.isnan(x))
        # 使用 stats 中的单样本 t 检验函数计算参考值
        t_ex, p_ex = stats.ttest_1samp(x.compressed(), 9,
                                       alternative=alternative)
        # 使用 mstats 中的单样本 t 检验函数计算测试值
        t, p = mstats.ttest_1samp(x, 9, alternative=alternative)
        # 断言测试值的 t 统计量近似等于参考值
        assert_allclose(t, t_ex, rtol=1e-14)
        # 断言测试值的 p 值近似等于参考值
        assert_allclose(p, p_ex, rtol=1e-14)


class TestDescribe:
    """
    Tests for mstats.describe.

    Note that there are also tests for `mstats.describe` in the
    class TestCompareWithStats.
    """
    # 定义一个测试方法，测试带有轴参数的基本功能
    def test_basic_with_axis(self):
        # 这是一个基本测试，也是 gh-7303 的回归测试
        # 创建一个掩码数组 a，包含两行六列的数据，并设置掩码
        a = np.ma.masked_array([[0, 1, 2, 3, 4, 9],
                                [5, 5, 0, 9, 3, 3]],
                               mask=[[0, 0, 0, 0, 0, 1],
                                     [0, 0, 1, 1, 0, 0]])
        # 对数组 a 进行描述统计，指定 axis=1 表示按行进行统计
        result = mstats.describe(a, axis=1)
        # 断言结果的观测值属性与预期的列表相等
        assert_equal(result.nobs, [5, 4])
        # 获取结果的最小值和最大值，并与预期的列表进行比较
        amin, amax = result.minmax
        assert_equal(amin, [0, 3])
        assert_equal(amax, [4, 5])
        # 断言结果的均值属性与预期的列表相等
        assert_equal(result.mean, [2.0, 4.0])
        # 断言结果的方差属性与预期的列表相等
        assert_equal(result.variance, [2.0, 1.0])
        # 断言结果的偏度属性与预期的列表相等
        assert_equal(result.skewness, [0.0, 0.0])
        # 断言结果的峰度属性与预期的列表相近（使用 assert_allclose 进行比较）
        assert_allclose(result.kurtosis, [-1.3, -2.0])
# 装饰器函数，用于跳过 XP 无效参数
@skip_xp_invalid_arg
# 定义一个测试类 TestCompareWithStats
class TestCompareWithStats:
    """
    Class to compare mstats results with stats results.

    It is in general assumed that scipy.stats is at a more mature stage than
    stats.mstats.  If a routine in mstats results in similar results like in
    scipy.stats, this is considered also as a proper validation of scipy.mstats
    routine.

    Different sample sizes are used for testing, as some problems between stats
    and mstats are dependent on sample size.

    Author: Alexander Loew

    NOTE that some tests fail. This might be caused by
    a) actual differences or bugs between stats and mstats
    b) numerical inaccuracies
    c) different definitions of routine interfaces

    These failures need to be checked. Current workaround is to have disabled these
    tests, but issuing reports on scipy-dev
    """

    # 返回用于比较的样本大小列表
    def get_n(self):
        """ Returns list of sample sizes to be used for comparison. """
        return [1000, 100, 10, 5]

    # 生成带有相同数据但包含额外掩码值的 numpy 数组和对应的掩码数组
    def generate_xy_sample(self, n):
        # 初始化随机种子
        np.random.seed(1234567)
        # 生成大小为 n 的随机标准正态分布数组 x 和 y
        x = np.random.randn(n)
        y = x + np.random.randn(n)
        # 创建比 x 和 y 长 5 的全为 1e16 的数组 xm 和 ym
        xm = np.full(len(x) + 5, 1e16)
        ym = np.full(len(y) + 5, 1e16)
        # 将 x 和 y 的数据复制到 xm 和 ym 中对应的位置
        xm[0:len(x)] = x
        ym[0:len(y)] = y
        # 创建掩码条件，将大于 9e15 的值标记为掩码
        mask = xm > 9e15
        # 创建掩码数组
        xm = np.ma.array(xm, mask=mask)
        ym = np.ma.array(ym, mask=mask)
        return x, y, xm, ym

    # 生成 2D 样本数据，返回 numpy 数组和掩码数组
    def generate_xy_sample2D(self, n, nx):
        # 创建大小为 (n, nx) 的 NaN 值数组 x 和 y
        x = np.full((n, nx), np.nan)
        y = np.full((n, nx), np.nan)
        # 创建大小为 (n+5, nx) 的 NaN 值数组 xm 和 ym
        xm = np.full((n+5, nx), np.nan)
        ym = np.full((n+5, nx), np.nan)

        # 循环生成 nx 个样本数据
        for i in range(nx):
            # 调用 generate_xy_sample 生成单个样本数据
            x[:, i], y[:, i], dx, dy = self.generate_xy_sample(n)

        # 将 x 和 y 的数据复制到 xm 和 ym 中对应的位置
        xm[0:n, :] = x[0:n]
        ym[0:n, :] = y[0:n]
        # 创建掩码数组
        xm = np.ma.array(xm, mask=np.isnan(xm))
        ym = np.ma.array(ym, mask=np.isnan(ym))
        return x, y, xm, ym

    # 测试线性回归
    def test_linregress(self):
        # 遍历样本大小列表
        for n in self.get_n():
            # 生成样本数据
            x, y, xm, ym = self.generate_xy_sample(n)
            # 使用 scipy.stats 计算线性回归结果
            result1 = stats.linregress(x, y)
            # 使用 scipy.mstats 计算掩码数组上的线性回归结果
            result2 = stats.mstats.linregress(xm, ym)
            # 断言两个结果数组近似相等
            assert_allclose(np.asarray(result1), np.asarray(result2))

    # 测试 Pearson 相关系数
    def test_pearsonr(self):
        # 遍历样本大小列表
        for n in self.get_n():
            # 生成样本数据
            x, y, xm, ym = self.generate_xy_sample(n)
            # 使用 scipy.stats 计算 Pearson 相关系数和 p 值
            r, p = stats.pearsonr(x, y)
            # 使用 scipy.mstats 计算掩码数组上的 Pearson 相关系数和 p 值
            rm, pm = stats.mstats.pearsonr(xm, ym)
            # 断言两个结果的数值近似相等
            assert_almost_equal(r, rm, decimal=14)
            assert_almost_equal(p, pm, decimal=14)

    # 测试 Spearman 秩相关系数
    def test_spearmanr(self):
        # 遍历样本大小列表
        for n in self.get_n():
            # 生成样本数据
            x, y, xm, ym = self.generate_xy_sample(n)
            # 使用 scipy.stats 计算 Spearman 秩相关系数和 p 值
            r, p = stats.spearmanr(x, y)
            # 使用 scipy.mstats 计算掩码数组上的 Spearman 秩相关系数和 p 值
            rm, pm = stats.mstats.spearmanr(xm, ym)
            # 断言两个结果的数值近似相等
            assert_almost_equal(r, rm, 14)
            assert_almost_equal(p, pm, 14)
    # 定义一个测试方法，用于验证 spearmanr 函数在向后兼容性上的表现
    def test_spearmanr_backcompat_useties(self):
        # 创建一个长度为6的数组
        x = np.arange(6)
        # 断言调用 spearmanr 函数时传入相同数组 x 两次会引发 ValueError 异常
        assert_raises(ValueError, mstats.spearmanr, x, x, False)

    # 定义一个测试方法，用于验证 gmean 函数的计算几何平均值的正确性
    def test_gmean(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算 x 的绝对值的几何平均值
            r = stats.gmean(abs(x))
            # 计算样本 x 的 mstats 模块的几何平均值
            rm = stats.mstats.gmean(abs(xm))
            # 断言两个几何平均值的接近程度
            assert_allclose(r, rm, rtol=1e-13)

            # 计算 y 的绝对值的几何平均值
            r = stats.gmean(abs(y))
            # 计算样本 y 的 mstats 模块的几何平均值
            rm = stats.mstats.gmean(abs(ym))
            # 断言两个几何平均值的接近程度
            assert_allclose(r, rm, rtol=1e-13)

    # 定义一个测试方法，用于验证 hmean 函数的计算调和平均值的正确性
    def test_hmean(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)

            # 计算 x 的绝对值的调和平均值
            r = stats.hmean(abs(x))
            # 计算样本 x 的 mstats 模块的调和平均值
            rm = stats.mstats.hmean(abs(xm))
            # 断言两个调和平均值的接近程度
            assert_almost_equal(r, rm, 10)

            # 计算 y 的绝对值的调和平均值
            r = stats.hmean(abs(y))
            # 计算样本 y 的 mstats 模块的调和平均值
            rm = stats.mstats.hmean(abs(ym))
            # 断言两个调和平均值的接近程度
            assert_almost_equal(r, rm, 10)

    # 定义一个测试方法，用于验证 skew 函数的计算偏度的正确性
    def test_skew(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)

            # 计算 x 的偏度
            r = stats.skew(x)
            # 计算样本 x 的 mstats 模块的偏度
            rm = stats.mstats.skew(xm)
            # 断言两个偏度的接近程度
            assert_almost_equal(r, rm, 10)

            # 计算 y 的偏度
            r = stats.skew(y)
            # 计算样本 y 的 mstats 模块的偏度
            rm = stats.mstats.skew(ym)
            # 断言两个偏度的接近程度
            assert_almost_equal(r, rm, 10)

    # 定义一个测试方法，用于验证 moment 函数的计算矩的正确性
    def test_moment(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)

            # 计算 x 的矩
            r = stats.moment(x)
            # 计算样本 x 的 mstats 模块的矩
            rm = stats.mstats.moment(xm)
            # 断言两个矩的接近程度
            assert_almost_equal(r, rm, 10)

            # 计算 y 的矩
            r = stats.moment(y)
            # 计算样本 y 的 mstats 模块的矩
            rm = stats.mstats.moment(ym)
            # 断言两个矩的接近程度
            assert_almost_equal(r, rm, 10)

    # 定义一个测试方法，用于验证 zscore 函数的计算 Z 分数的正确性
    def test_zscore(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)

            # 计算参考解法下 x 的 Z 分数
            zx = (x - x.mean()) / x.std()
            zy = (y - y.mean()) / y.std()

            # 验证 stats 模块计算的 Z 分数
            assert_allclose(stats.zscore(x), zx, rtol=1e-10)
            assert_allclose(stats.zscore(y), zy, rtol=1e-10)

            # 比较 stats 和 mstats 模块计算的 Z 分数
            assert_allclose(stats.zscore(x), stats.mstats.zscore(xm[0:len(x)]),
                            rtol=1e-10)
            assert_allclose(stats.zscore(y), stats.mstats.zscore(ym[0:len(y)]),
                            rtol=1e-10)

    # 定义一个测试方法，用于验证 kurtosis 函数的计算峰度的正确性
    def test_kurtosis(self):
        # 遍历获取样本大小的生成器
        for n in self.get_n():
            # 生成 x, y 及其样本的绝对值和均值
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算 x 的峰度
            r = stats.kurtosis(x)
            # 计算样本 x 的 mstats 模块的峰度
            rm = stats.mstats.kurtosis(xm)
            # 断言两个峰度的接近程度
            assert_almost_equal(r, rm, 10)

            # 计算 y 的峰度
            r = stats.kurtosis(y)
            # 计算样本 y 的 mstats 模块的峰度
            rm = stats.mstats.kurtosis(ym)
            # 断言两个峰度的接近程度
            assert_almost_equal(r, rm, 10)
    def test_sem(self):
        # example from stats.sem doc
        # 创建一个 5x4 的数组 a
        a = np.arange(20).reshape(5, 4)
        # 将数组 a 转换为掩码数组 am
        am = np.ma.array(a)
        # 计算数组 a 的标准误差 r，自由度为 1
        r = stats.sem(a, ddof=1)
        # 计算掩码数组 am 的标准误差 rm，自由度为 1
        rm = stats.mstats.sem(am, ddof=1)

        # 断言 r 与预期值接近，允许的绝对误差为 1e-5
        assert_allclose(r, 2.82842712, atol=1e-5)
        # 断言 rm 与预期值接近，允许的绝对误差为 1e-5
        assert_allclose(rm, 2.82842712, atol=1e-5)

        # 遍历 self.get_n() 返回的值
        for n in self.get_n():
            # 生成样本 x, y, xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 xm 的标准误差（无轴向，自由度为 0）与 x 的相同
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=0),
                                stats.sem(x, axis=None, ddof=0), decimal=13)
            # 断言 ym 的标准误差（无轴向，自由度为 0）与 y 的相同
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=0),
                                stats.sem(y, axis=None, ddof=0), decimal=13)
            # 断言 xm 的标准误差（无轴向，自由度为 1）与 x 的相同
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=1),
                                stats.sem(x, axis=None, ddof=1), decimal=13)
            # 断言 ym 的标准误差（无轴向，自由度为 1）与 y 的相同
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=1),
                                stats.sem(y, axis=None, ddof=1), decimal=13)

    def test_describe(self):
        # 遍历 self.get_n() 返回的值
        for n in self.get_n():
            # 生成样本 x, y, xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算 x 的描述统计信息，自由度为 1
            r = stats.describe(x, ddof=1)
            # 计算掩码数组 xm 的描述统计信息，自由度为 1
            rm = stats.mstats.describe(xm, ddof=1)
            # 对于前 6 个元素，断言 r 和 rm 数组的值接近
            for ii in range(6):
                assert_almost_equal(np.asarray(r[ii]),
                                    np.asarray(rm[ii]),
                                    decimal=12)

    def test_describe_result_attributes(self):
        # 调用 mstats.describe() 计算包含 0 到 4 的数组的描述统计信息
        actual = mstats.describe(np.arange(5))
        # 定义期望的属性
        attributes = ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                      'kurtosis')
        # 检查返回结果 actual 是否具有指定的命名属性
        check_named_results(actual, attributes, ma=True)

    def test_rankdata(self):
        # 遍历 self.get_n() 返回的值
        for n in self.get_n():
            # 生成样本 x, y, xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算数组 x 的秩
            r = stats.rankdata(x)
            # 计算掩码数组 xm 的秩
            rm = stats.mstats.rankdata(x)
            # 断言 r 与 rm 数组的值接近
            assert_allclose(r, rm)

    def test_tmean(self):
        # 遍历 self.get_n() 返回的值
        for n in self.get_n():
            # 生成样本 x, y, xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 x 的修正均值与 xm 的修正均值接近，精度为 14
            assert_almost_equal(stats.tmean(x),stats.mstats.tmean(xm), 14)
            # 断言 y 的修正均值与 ym 的修正均值接近，精度为 14
            assert_almost_equal(stats.tmean(y),stats.mstats.tmean(ym), 14)

    def test_tmax(self):
        # 遍历 self.get_n() 返回的值
        for n in self.get_n():
            # 生成样本 x, y, xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 x 的最大值（上限为 2.）与 xm 的最大值接近，精度为 10
            assert_almost_equal(stats.tmax(x,2.),
                                stats.mstats.tmax(xm,2.), 10)
            # 断言 y 的最大值（上限为 2.）与 ym 的最大值接近，精度为 10
            assert_almost_equal(stats.tmax(y,2.),
                                stats.mstats.tmax(ym,2.), 10)

            # 断言 x 的最大值（上限为 3.）与 xm 的最大值接近，精度为 10
            assert_almost_equal(stats.tmax(x, upperlimit=3.),
                                stats.mstats.tmax(xm, upperlimit=3.), 10)
            # 断言 y 的最大值（上限为 3.）与 ym 的最大值接近，精度为 10
            assert_almost_equal(stats.tmax(y, upperlimit=3.),
                                stats.mstats.tmax(ym, upperlimit=3.), 10)
    # 测试函数，用于测试 stats.tmin 函数
    def test_tmin(self):
        # 遍历 self.get_n() 返回的迭代器
        for n in self.get_n():
            # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 stats.tmin(x) 等于 stats.mstats.tmin(xm)
            assert_equal(stats.tmin(x), stats.mstats.tmin(xm))
            # 断言 stats.tmin(y) 等于 stats.mstats.tmin(ym)
            assert_equal(stats.tmin(y), stats.mstats.tmin(ym))

            # 断言 stats.tmin(x, lowerlimit=-1.) 与 stats.mstats.tmin(xm, lowerlimit=-1.) 相近，精度为 10
            assert_almost_equal(stats.tmin(x, lowerlimit=-1.),
                                stats.mstats.tmin(xm, lowerlimit=-1.), 10)
            # 断言 stats.tmin(y, lowerlimit=-1.) 与 stats.mstats.tmin(ym, lowerlimit=-1.) 相近，精度为 10
            assert_almost_equal(stats.tmin(y, lowerlimit=-1.),
                                stats.mstats.tmin(ym, lowerlimit=-1.), 10)

    # 测试函数，用于测试 stats.zmap 函数
    def test_zmap(self):
        # 遍历 self.get_n() 返回的迭代器
        for n in self.get_n():
            # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算 stats.zmap(x, y)
            z = stats.zmap(x, y)
            # 计算 stats.mstats.zmap(xm, ym)
            zm = stats.mstats.zmap(xm, ym)
            # 断言 z 与 zm 的前 len(z) 个元素在容差 1e-10 内相等
            assert_allclose(z, zm[0:len(z)], atol=1e-10)

    # 测试函数，用于测试 stats.variation 函数
    def test_variation(self):
        # 遍历 self.get_n() 返回的迭代器
        for n in self.get_n():
            # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 stats.variation(x) 等于 stats.mstats.variation(xm)，精度为 12
            assert_almost_equal(stats.variation(x), stats.mstats.variation(xm),
                                decimal=12)
            # 断言 stats.variation(y) 等于 stats.mstats.variation(ym)，精度为 12
            assert_almost_equal(stats.variation(y), stats.mstats.variation(ym),
                                decimal=12)

    # 测试函数，用于测试 stats.tvar 函数
    def test_tvar(self):
        # 遍历 self.get_n() 返回的迭代器
        for n in self.get_n():
            # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 stats.tvar(x) 等于 stats.mstats.tvar(xm)，精度为 12
            assert_almost_equal(stats.tvar(x), stats.mstats.tvar(xm),
                                decimal=12)
            # 断言 stats.tvar(y) 等于 stats.mstats.tvar(ym)，精度为 12
            assert_almost_equal(stats.tvar(y), stats.mstats.tvar(ym),
                                decimal=12)

    # 测试函数，用于测试 stats.trimboth 函数
    def test_trimboth(self):
        # 创建数组 a，其中包含 0 到 19 的整数
        a = np.arange(20)
        # 使用 stats.trimboth 函数对数组 a 进行修剪，修剪比例为 0.1
        b = stats.trimboth(a, 0.1)
        # 使用 stats.mstats.trimboth 函数对数组 a 进行修剪，修剪比例为 0.1
        bm = stats.mstats.trimboth(a, 0.1)
        # 断言 np.sort(b) 与 bm.data[~bm.mask] 的排序后内容相近
        assert_allclose(np.sort(b), bm.data[~bm.mask])

    # 测试函数，用于测试 stats.tsem 函数
    def test_tsem(self):
        # 遍历 self.get_n() 返回的迭代器
        for n in self.get_n():
            # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
            x, y, xm, ym = self.generate_xy_sample(n)
            # 断言 stats.tsem(x) 等于 stats.mstats.tsem(xm)，精度为 14
            assert_almost_equal(stats.tsem(x), stats.mstats.tsem(xm),
                                decimal=14)
            # 断言 stats.tsem(y) 等于 stats.mstats.tsem(ym)，精度为 14
            assert_almost_equal(stats.tsem(y), stats.mstats.tsem(ym),
                                decimal=14)
            # 断言 stats.tsem(x, limits=(-2., 2.)) 等于 stats.mstats.tsem(xm, limits=(-2., 2.))，精度为 14
            assert_almost_equal(stats.tsem(x, limits=(-2., 2.)),
                                stats.mstats.tsem(xm, limits=(-2., 2.)),
                                decimal=14)

    # 测试函数，用于测试 stats.skewtest 函数
    def test_skewtest(self):
        # 仅适用于一维数据的测试
        for n in self.get_n():
            if n > 8:
                # 使用 self.generate_xy_sample(n) 生成样本数据 x, y 和样本均值数据 xm, ym
                x, y, xm, ym = self.generate_xy_sample(n)
                # 计算 stats.skewtest(x)
                r = stats.skewtest(x)
                # 计算 stats.mstats.skewtest(xm)
                rm = stats.mstats.skewtest(xm)
                # 断言 r 与 rm 的结果相近
                assert_allclose(r, rm)

    # 测试函数，用于测试 stats.skewtest 函数返回结果的属性
    def test_skewtest_result_attributes(self):
        # 创建一维数组 x
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        # 对数组 x 运行 mstats.skewtest 函数
        res = mstats.skewtest(x)
        # 定义要检查的结果属性
        attributes = ('statistic', 'pvalue')
        # 使用 check_named_results 函数检查 res 的属性
        check_named_results(res, attributes, ma=True)

    # 测试函数，用于测试 stats.skewtest 函数处理二维非遮蔽数据
    def test_skewtest_2D_notmasked(self):
        # 创建一个 20x2 的随机二维数组 x
        x = np.random.random((20, 2)) * 20.
        # 计算 stats.skewtest(x)
        r = stats.skewtest(x)
        # 计算 stats.mstats.skewtest(x)
        rm = stats.mstats.skewtest(x)
        # 断言 r 与 rm 的结果相等
        assert_allclose(np.asarray(r), np.asarray(rm))
    # 定义一个测试方法，用于测试带有掩码的二维偏斜测试
    def test_skewtest_2D_WithMask(self):
        # 设置样本数量 nx 为 2
        nx = 2
        # 遍历生成样本数量的迭代器
        for n in self.get_n():
            # 如果样本数量大于 8
            if n > 8:
                # 使用 generate_xy_sample2D 方法生成 x, y, xm, ym 四个样本数据
                x, y, xm, ym = self.generate_xy_sample2D(n, nx)
                # 对 x 进行偏斜测试，返回结果 r
                r = stats.skewtest(x)
                # 对 xm 进行偏斜测试，返回结果 rm
                rm = stats.mstats.skewtest(xm)

                # 断言 r[0][0] 与 rm[0][0] 相近，设置相对容差 rtol=1e-14
                assert_allclose(r[0][0], rm[0][0], rtol=1e-14)
                # 断言 r[0][1] 与 rm[0][1] 相近，设置相对容差 rtol=1e-14
                assert_allclose(r[0][1], rm[0][1], rtol=1e-14)

    # 定义一个测试方法，用于测试正态性测试
    def test_normaltest(self):
        # 设置 numpy 的错误状态，在发生溢出时抛出异常
        with np.errstate(over='raise'), suppress_warnings() as sup:
            # 过滤掉 'kurtosistest' 可能不准确的警告
            sup.filter(UserWarning, "`kurtosistest` p-value may be inaccurate")
            # 过滤掉只适用于 n>=20 的 'kurtosistest' 警告
            sup.filter(UserWarning, "kurtosistest only valid for n>=20")
            # 遍历生成样本数量的迭代器
            for n in self.get_n():
                # 如果样本数量大于 8
                if n > 8:
                    # 使用 generate_xy_sample 方法生成 x, y, xm, ym 四个样本数据
                    x, y, xm, ym = self.generate_xy_sample(n)
                    # 对 x 进行正态性测试，返回结果 r
                    r = stats.normaltest(x)
                    # 对 xm 进行正态性测试，返回结果 rm
                    rm = stats.mstats.normaltest(xm)
                    # 将 r 和 rm 转换为数组后，断言它们相等
                    assert_allclose(np.asarray(r), np.asarray(rm))

    # 定义一个测试方法，用于测试重复值查找
    def test_find_repeats(self):
        # 创建一个浮点数 numpy 数组 x
        x = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]).astype('float')
        # 创建一个临时的浮点数 numpy 数组 tmp
        tmp = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]).astype('float')
        # 创建一个掩码，用于标记 tmp 中值为 5.0 的位置
        mask = (tmp == 5.)
        # 使用掩码创建一个带掩码的 numpy.ma 数组 xm
        xm = np.ma.array(tmp, mask=mask)
        # 复制 x 和 xm 到原始变量 x_orig 和 xm_orig
        x_orig, xm_orig = x.copy(), xm.copy()

        # 使用 find_repeats 函数查找 x 中的重复值，返回结果 r
        r = stats.find_repeats(x)
        # 使用 mstats 模块中的 find_repeats 函数查找 xm 中的重复值，返回结果 rm
        rm = stats.mstats.find_repeats(xm)

        # 断言 r 和 rm 相等
        assert_equal(r, rm)
        # 断言 x 和 x_orig 相等
        assert_equal(x, x_orig)
        # 断言 xm 和 xm_orig 相等
        assert_equal(xm, xm_orig)

        # 预期 count_tied_groups 函数的特殊行为，不在文档字符串中...
        # 对一个空列表使用 find_repeats 函数，期望返回空数组
        _, counts = stats.mstats.find_repeats([])
        assert_equal(counts, np.array(0, dtype=np.intp))

    # 定义一个测试方法，用于测试 Kendall Tau 相关性
    def test_kendalltau(self):
        # 遍历生成样本数量的迭代器
        for n in self.get_n():
            # 使用 generate_xy_sample 方法生成 x, y, xm, ym 四个样本数据
            x, y, xm, ym = self.generate_xy_sample(n)
            # 计算 x 和 y 的 Kendall Tau 相关性，返回结果 r
            r = stats.kendalltau(x, y)
            # 计算 xm 和 ym 的 Kendall Tau 相关性，返回结果 rm
            rm = stats.mstats.kendalltau(xm, ym)
            # 断言 r[0] 和 rm[0] 相等，设置精度到小数点后第 10 位
            assert_almost_equal(r[0], rm[0], decimal=10)
            # 断言 r[1] 和 rm[1] 相等，设置精度到小数点后第 7 位
            assert_almost_equal(r[1], rm[1], decimal=7)

    # 定义一个测试方法，用于测试 O'Brien 转换
    def test_obrientransform(self):
        # 遍历生成样本数量的迭代器
        for n in self.get_n():
            # 使用 generate_xy_sample 方法生成 x, y, xm, ym 四个样本数据
            x, y, xm, ym = self.generate_xy_sample(n)
            # 对 x 进行 O'Brien 转换，返回结果 r
            r = stats.obrientransform(x)
            # 对 xm 进行 O'Brien 转换，返回结果 rm
            rm = stats.mstats.obrientransform(xm)
            # 断言 r.T 和 rm[0:len(x)] 相等，转置 r 后与 rm 的部分长度进行比较
            assert_almost_equal(r.T, rm[0:len(x)])
    # 定义一个测试函数，用于验证 mstats.ks_1samp 和 stats.ks_1samp 在掩码数组上的一致性
    def test_ks_1samp(self):
        """Checks that mstats.ks_1samp and stats.ks_1samp agree on masked arrays."""
        # 遍历三种模式：'auto', 'exact', 'asymp'
        for mode in ['auto', 'exact', 'asymp']:
            # 忽略警告上下文
            with suppress_warnings():
                # 遍历三种假设检验类型：'less', 'greater', 'two-sided'
                for alternative in ['less', 'greater', 'two-sided']:
                    # 获取样本大小的生成器，并遍历样本大小 n
                    for n in self.get_n():
                        # 生成样本 x, y 以及掩码后的样本 xm, ym
                        x, y, xm, ym = self.generate_xy_sample(n)
                        # 使用 stats.ks_1samp 计算检验结果 res1
                        res1 = stats.ks_1samp(x, stats.norm.cdf,
                                              alternative=alternative, mode=mode)
                        # 使用 stats.mstats.ks_1samp 在掩码数组 xm 上计算检验结果 res2
                        res2 = stats.mstats.ks_1samp(xm, stats.norm.cdf,
                                                     alternative=alternative, mode=mode)
                        # 断言 res1 和 res2 的数组形式相等
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        # 再次使用 stats.ks_1samp 在掩码数组 xm 上计算检验结果 res3
                        res3 = stats.ks_1samp(xm, stats.norm.cdf,
                                              alternative=alternative, mode=mode)
                        # 断言 res1 和 res3 的数组形式相等
                        assert_equal(np.asarray(res1), np.asarray(res3))

    # 定义一个测试函数，用于验证 mstats.kstest 和 stats.kstest 在掩码数组上的一致性
    def test_kstest_1samp(self):
        """
        Checks that 1-sample mstats.kstest and stats.kstest agree on masked arrays.
        """
        # 遍历三种模式：'auto', 'exact', 'asymp'
        for mode in ['auto', 'exact', 'asymp']:
            # 忽略警告上下文
            with suppress_warnings():
                # 遍历三种假设检验类型：'less', 'greater', 'two-sided'
                for alternative in ['less', 'greater', 'two-sided']:
                    # 获取样本大小的生成器，并遍历样本大小 n
                    for n in self.get_n():
                        # 生成样本 x, y 以及掩码后的样本 xm, ym
                        x, y, xm, ym = self.generate_xy_sample(n)
                        # 使用 stats.kstest 计算检验结果 res1
                        res1 = stats.kstest(x, 'norm',
                                            alternative=alternative, mode=mode)
                        # 使用 stats.mstats.kstest 在掩码数组 xm 上计算检验结果 res2
                        res2 = stats.mstats.kstest(xm, 'norm',
                                                   alternative=alternative, mode=mode)
                        # 断言 res1 和 res2 的数组形式相等
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        # 再次使用 stats.kstest 在掩码数组 xm 上计算检验结果 res3
                        res3 = stats.kstest(xm, 'norm',
                                            alternative=alternative, mode=mode)
                        # 断言 res1 和 res3 的数组形式相等
                        assert_equal(np.asarray(res1), np.asarray(res3))
    # 定义一个测试方法，用于验证 mstats.ks_2samp 和 stats.ks_2samp 在掩码数组上的一致性
    def test_ks_2samp(self):
        """Checks that mstats.ks_2samp and stats.ks_2samp agree on masked arrays.
        gh-8431"""
        # 遍历不同的计算模式：'auto', 'exact', 'asymp'
        for mode in ['auto', 'exact', 'asymp']:
            # 使用 suppress_warnings 上下文管理器，用于处理特定模式下的运行时警告
            with suppress_warnings() as sup:
                # 如果模式为 'auto' 或 'exact'，过滤特定的运行时警告信息
                if mode in ['auto', 'exact']:
                    message = "ks_2samp: Exact calculation unsuccessful."
                    sup.filter(RuntimeWarning, message)
                # 遍历不同的假设检验类型：'less', 'greater', 'two-sided'
                for alternative in ['less', 'greater', 'two-sided']:
                    # 遍历生成不同样本大小 n 的样本数据
                    for n in self.get_n():
                        # 生成样本数据 x, y 和相应的掩码数组 xm, ym
                        x, y, xm, ym = self.generate_xy_sample(n)
                        # 使用 stats.ks_2samp 计算两个样本 x, y 的 KS 检验结果
                        res1 = stats.ks_2samp(x, y,
                                              alternative=alternative, mode=mode)
                        # 使用 mstats.ks_2samp 计算两个掩码数组 xm, ym 的 KS 检验结果
                        res2 = stats.mstats.ks_2samp(xm, ym,
                                                     alternative=alternative, mode=mode)
                        # 断言两个结果数组的一致性
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        # 再次使用 stats.ks_2samp 计算掩码数组 xm, y 的 KS 检验结果
                        res3 = stats.ks_2samp(xm, y,
                                              alternative=alternative, mode=mode)
                        # 断言结果数组 res1 与 res3 的一致性
                        assert_equal(np.asarray(res1), np.asarray(res3))

    # 定义一个测试方法，用于验证 mstats.kstest 和 stats.kstest 在掩码数组上的一致性
    def test_kstest_2samp(self):
        """
        Checks that 2-sample mstats.kstest and stats.kstest agree on masked arrays.
        """
        # 遍历不同的计算模式：'auto', 'exact', 'asymp'
        for mode in ['auto', 'exact', 'asymp']:
            # 使用 suppress_warnings 上下文管理器，用于处理特定模式下的运行时警告
            with suppress_warnings() as sup:
                # 如果模式为 'auto' 或 'exact'，过滤特定的运行时警告信息
                if mode in ['auto', 'exact']:
                    message = "ks_2samp: Exact calculation unsuccessful."
                    sup.filter(RuntimeWarning, message)
                # 遍历不同的假设检验类型：'less', 'greater', 'two-sided'
                for alternative in ['less', 'greater', 'two-sided']:
                    # 遍历生成不同样本大小 n 的样本数据
                    for n in self.get_n():
                        # 生成样本数据 x, y 和相应的掩码数组 xm, ym
                        x, y, xm, ym = self.generate_xy_sample(n)
                        # 使用 stats.kstest 计算两个样本 x, y 的 KS 检验结果
                        res1 = stats.kstest(x, y,
                                            alternative=alternative, mode=mode)
                        # 使用 mstats.kstest 计算两个掩码数组 xm, ym 的 KS 检验结果
                        res2 = stats.mstats.kstest(xm, ym,
                                                   alternative=alternative, mode=mode)
                        # 断言两个结果数组的一致性
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        # 再次使用 stats.kstest 计算掩码数组 xm, y 的 KS 检验结果
                        res3 = stats.kstest(xm, y,
                                            alternative=alternative, mode=mode)
                        # 断言结果数组 res1 与 res3 的一致性
                        assert_equal(np.asarray(res1), np.asarray(res3))
class TestBrunnerMunzel:
    # Data from (Lumley, 1996)
    # 定义两个包含数据的掩码数组，用于测试布伦纳-门泽尔检验
    X = np.ma.masked_invalid([1, 2, 1, 1, 1, np.nan, 1, 1,
                              1, 1, 1, 2, 4, 1, 1, np.nan])
    Y = np.ma.masked_invalid([3, 3, 4, 3, np.nan, 1, 2, 3, 1, 1, 5, 4])
    significant = 14  # 设定显著性水平为14

    def test_brunnermunzel_one_sided(self):
        # Results are compared with R's lawstat package.
        # 使用布伦纳-门泽尔检验计算给定数据的统计量和 p 值，设置单侧检验
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, alternative='less')
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, alternative='greater')
        u3, p3 = mstats.brunnermunzel(self.X, self.Y, alternative='greater')
        u4, p4 = mstats.brunnermunzel(self.Y, self.X, alternative='less')

        # 断言：两个单侧检验得到的 p 值应该相等
        assert_almost_equal(p1, p2, decimal=self.significant)
        assert_almost_equal(p3, p4, decimal=self.significant)
        assert_(p1 != p3)  # 断言：两个不同的单侧检验得到的 p 值不相等
        # 断言：检验统计量的值应与预期值接近
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u3, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u4, -3.1374674823029505,
                            decimal=self.significant)
        # 断言：两个单侧检验得到的 p 值应与预期值接近
        assert_almost_equal(p1, 0.0028931043330757342,
                            decimal=self.significant)
        assert_almost_equal(p3, 0.99710689566692423,
                            decimal=self.significant)

    def test_brunnermunzel_two_sided(self):
        # Results are compared with R's lawstat package.
        # 使用布伦纳-门泽尔检验计算给定数据的统计量和 p 值，设置双侧检验
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, alternative='two-sided')
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, alternative='two-sided')

        # 断言：两个双侧检验得到的 p 值应该相等
        assert_almost_equal(p1, p2, decimal=self.significant)
        # 断言：检验统计量的值应与预期值接近
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        # 断言：双侧检验得到的 p 值应与预期值接近
        assert_almost_equal(p1, 0.0057862086661515377,
                            decimal=self.significant)

    def test_brunnermunzel_default(self):
        # The default value for alternative is two-sided
        # 使用布伦纳-门泽尔检验计算给定数据的统计量和 p 值，使用默认的双侧检验
        u1, p1 = mstats.brunnermunzel(self.X, self.Y)
        u2, p2 = mstats.brunnermunzel(self.Y, self.X)

        # 断言：两个默认双侧检验得到的 p 值应该相等
        assert_almost_equal(p1, p2, decimal=self.significant)
        # 断言：检验统计量的值应与预期值接近
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        # 断言：默认双侧检验得到的 p 值应与预期值接近
        assert_almost_equal(p1, 0.0057862086661515377,
                            decimal=self.significant)
    # 定义测试函数，测试当alternative为"error"时的情况
    def test_brunnermunzel_alternative_error(self):
        # 设置alternative为"error"，distribution为"t"
        alternative = "error"
        distribution = "t"
        # 断言alternative不在["two-sided", "greater", "less"]中
        assert_(alternative not in ["two-sided", "greater", "less"])
        # 断言调用mstats.brunnermunzel函数时会抛出ValueError异常，传入参数为self.X, self.Y, alternative, distribution
        assert_raises(ValueError,
                      mstats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution)

    # 定义测试函数，测试当distribution为"normal"时的情况
    def test_brunnermunzel_distribution_norm(self):
        # 调用mstats.brunnermunzel函数，计算u1, p1，传入参数为self.X, self.Y, distribution="normal"
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, distribution="normal")
        # 调用mstats.brunnermunzel函数，计算u2, p2，传入参数为self.Y, self.X, distribution="normal"
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, distribution="normal")
        # 断言p1与p2几乎相等，精度为self.significant
        assert_almost_equal(p1, p2, decimal=self.significant)
        # 断言u1几乎等于3.1374674823029505，精度为self.significant
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        # 断言u2几乎等于-3.1374674823029505，精度为self.significant
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        # 断言p1几乎等于0.0017041417600383024，精度为self.significant
        assert_almost_equal(p1, 0.0017041417600383024,
                            decimal=self.significant)

    # 定义测试函数，测试当distribution为"error"时的情况
    def test_brunnermunzel_distribution_error(self):
        # 设置alternative为"two-sided"，distribution为"error"
        alternative = "two-sided"
        distribution = "error"
        # 断言alternative不在["t", "normal"]中
        assert_(alternative not in ["t", "normal"])
        # 断言调用mstats.brunnermunzel函数时会抛出ValueError异常，传入参数为self.X, self.Y, alternative, distribution
        assert_raises(ValueError,
                      mstats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution)

    # 定义测试函数，测试当输入为空列表时的情况
    def test_brunnermunzel_empty_imput(self):
        # 调用mstats.brunnermunzel函数，计算u1, p1，传入参数为self.X, []
        u1, p1 = mstats.brunnermunzel(self.X, [])
        # 调用mstats.brunnermunzel函数，计算u2, p2，传入参数为[], self.Y
        u2, p2 = mstats.brunnermunzel([], self.Y)
        # 调用mstats.brunnermunzel函数，计算u3, p3，传入参数为[], []
        u3, p3 = mstats.brunnermunzel([], [])

        # 断言u1, p1, u2, p2, u3, p3都是NaN
        assert_(np.isnan(u1))
        assert_(np.isnan(p1))
        assert_(np.isnan(u2))
        assert_(np.isnan(p2))
        assert_(np.isnan(u3))
        assert_(np.isnan(p3))
```