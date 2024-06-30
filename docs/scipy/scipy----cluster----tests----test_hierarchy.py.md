# `D:\src\scipysrc\scipy\scipy\cluster\tests\test_hierarchy.py`

```
#
# Author: Damian Eads
# Date: April 17, 2008
#
# Copyright (C) 2008 Damian Eads
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises

import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
    ClusterWarning, linkage, from_mlab_linkage, to_mlab_linkage,
    num_obs_linkage, inconsistent, cophenet, fclusterdata, fcluster,
    is_isomorphic, single, leaders,
    correspond, is_monotonic, maxdists, maxinconsts, maxRstat,
    is_valid_linkage, is_valid_im, to_tree, leaves_list, dendrogram,
    set_link_color_palette, cut_tree, optimal_leaf_ordering,
    _order_cluster_tree, _hierarchy, _LINKAGE_METHODS)
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close, xp_assert_equal

from . import hierarchy_test_data


# Matplotlib is not a scipy dependency but is optionally used in dendrogram, so
# check if it's available
try:
    import matplotlib
    # and set the backend to be Agg (no gui)
    matplotlib.use('Agg')
    # before importing pyplot
    import matplotlib.pyplot as plt
    have_matplotlib = True
except Exception:
    have_matplotlib = False


pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
skip_xp_backends = pytest.mark.skip_xp_backends


class TestLinkage:

    @skip_xp_backends(cpu_only=True)
    # 定义一个测试类 TestLinkage，用于测试 scipy.cluster.hierarchy 模块的相关功能
    # 使用 skip_xp_backends 装饰器标记，表示这些测试用例仅在非 XP 后端环境下运行
    def test_linkage_non_finite_elements_in_distance_matrix(self, xp):
        # 测试 linkage(Y)，其中 Y 包含非有限元素（例如 NaN 或 Inf）。预期会抛出异常。
        y = xp.asarray([xp.nan] + [0.0]*5)
        assert_raises(ValueError, linkage, y)

    @skip_xp_backends(cpu_only=True)
    def test_linkage_empty_distance_matrix(self, xp):
        # 测试 linkage(Y)，其中 Y 是一个 0x4 的链接矩阵。预期会抛出异常。
        y = xp.zeros((0,))
        assert_raises(ValueError, linkage, y)

    @skip_xp_backends(cpu_only=True)
    def test_linkage_tdist(self, xp):
        for method in ['single', 'complete', 'average', 'weighted']:
            # 对 tdist 数据集使用指定的方法测试 linkage(Y, method)。
            self.check_linkage_tdist(method, xp)

    def check_linkage_tdist(self, method, xp):
        # 在 tdist 数据集上测试 linkage(Y, method)。
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_' + method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)

    @skip_xp_backends(cpu_only=True)
    def test_linkage_X(self, xp):
        for method in ['centroid', 'median', 'ward']:
            # 对 X 数据集使用指定的方法测试 linkage(Y, method)。
            self.check_linkage_q(method, xp)

    def check_linkage_q(self, method, xp):
        # 在 Q 数据集上测试 linkage(Y, method)。
        Z = linkage(xp.asarray(hierarchy_test_data.X), method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_X_' + method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)

        y = scipy.spatial.distance.pdist(hierarchy_test_data.X,
                                         metric="euclidean")
        Z = linkage(xp.asarray(y), method)
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)

    @skip_xp_backends(cpu_only=True)
    def test_compare_with_trivial(self, xp):
        rng = np.random.RandomState(0)
        n = 20
        X = rng.rand(n, 2)
        d = pdist(X)

        for method, code in _LINKAGE_METHODS.items():
            # 使用不同方法比较 linkage(d, n, code) 和 linkage(Y, method) 的结果。
            Z_trivial = _hierarchy.linkage(d, n, code)
            Z = linkage(xp.asarray(d), method)
            xp_assert_close(Z, xp.asarray(Z_trivial), rtol=1e-14, atol=1e-15)

    @skip_xp_backends(cpu_only=True)
    def test_optimal_leaf_ordering(self, xp):
        # 在 ytdist 数据集上测试 linkage(Y, optimal_ordering=True)。
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), optimal_ordering=True)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_single_olo')
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)
@skip_xp_backends(cpu_only=True)
class TestLinkageTies:
    
    # 预期的聚类链接矩阵，包含不同方法的预期结果
    _expectations = {
        'single': np.array([[0, 1, 1.41421356, 2],
                            [2, 3, 1.41421356, 3]]),
        'complete': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.82842712, 3]]),
        'average': np.array([[0, 1, 1.41421356, 2],
                             [2, 3, 2.12132034, 3]]),
        'weighted': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.12132034, 3]]),
        'centroid': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.12132034, 3]]),
        'median': np.array([[0, 1, 1.41421356, 2],
                            [2, 3, 2.12132034, 3]]),
        'ward': np.array([[0, 1, 1.41421356, 2],
                          [2, 3, 2.44948974, 3]]),
    }
    
    # 测试聚类链接方法的方法
    def test_linkage_ties(self, xp):
        for method in ['single', 'complete', 'average', 'weighted',
                       'centroid', 'median', 'ward']:
            self.check_linkage_ties(method, xp)
    
    # 检查给定方法的聚类链接矩阵是否与预期接近
    def check_linkage_ties(self, method, xp):
        X = xp.asarray([[-1, -1], [0, 0], [1, 1]])
        Z = linkage(X, method=method)
        expectedZ = self._expectations[method]
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)


@skip_xp_backends(cpu_only=True)
class TestInconsistent:
    
    # 测试不一致指数函数在不同深度的表现
    def test_inconsistent_tdist(self, xp):
        for depth in hierarchy_test_data.inconsistent_ytdist:
            self.check_inconsistent_tdist(depth, xp)
    
    # 检查给定深度的不一致指数计算是否与预期接近
    def check_inconsistent_tdist(self, depth, xp):
        Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
        xp_assert_close(inconsistent(Z, depth),
                        xp.asarray(hierarchy_test_data.inconsistent_ytdist[depth]))


@skip_xp_backends(cpu_only=True)
class TestCopheneticDistance:
    
    # 测试 cophenet(Z) 在 tdist 数据集上的表现
    def test_linkage_cophenet_tdist_Z(self, xp):
        expectedM = xp.asarray([268, 295, 255, 255, 295, 295, 268, 268, 295, 295,
                                295, 138, 219, 295, 295])
        Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
        M = cophenet(Z)
        xp_assert_close(M, xp.asarray(expectedM, dtype=xp.float64), atol=1e-10)
    
    # 测试 cophenet(Z, Y) 在 tdist 数据集上的表现
    def test_linkage_cophenet_tdist_Z_Y(self, xp):
        Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
        (c, M) = cophenet(Z, xp.asarray(hierarchy_test_data.ytdist))
        expectedM = xp.asarray([268, 295, 255, 255, 295, 295, 268, 268, 295, 295,
                                295, 138, 219, 295, 295], dtype=xp.float64)
        expectedc = xp.asarray(0.639931296433393415057366837573, dtype=xp.float64)[()]
        xp_assert_close(c, expectedc, atol=1e-10)
        xp_assert_close(M, expectedM, atol=1e-10)


class TestMLabLinkageConversion:
    pass  # 此类未包含需要注释的代码，因此保持空白
    def test_mlab_linkage_conversion_empty(self, xp):
        # Tests from/to_mlab_linkage on empty linkage array.
        # 创建一个空的 Numpy 数组 X，数据类型为 float64
        X = xp.asarray([], dtype=xp.float64)
        # 调用 from_mlab_linkage 函数，期望其返回值与 X 相等
        xp_assert_equal(from_mlab_linkage(X), X)
        # 调用 to_mlab_linkage 函数，期望其返回值与 X 相等
        xp_assert_equal(to_mlab_linkage(X), X)

    @skip_xp_backends(cpu_only=True)
    def test_mlab_linkage_conversion_single_row(self, xp):
        # Tests from/to_mlab_linkage on linkage array with single row.
        # 创建一个包含单行数据的 Numpy 数组 Z
        Z = xp.asarray([[0., 1., 3., 2.]])
        # 创建一个对应的单行 mlab linkage 数组 Zm
        Zm = xp.asarray([[1, 2, 3]])
        # 调用 from_mlab_linkage 函数，期望其返回值与 Z 相等，数据类型为 float64
        xp_assert_close(from_mlab_linkage(Zm), xp.asarray(Z, dtype=xp.float64),
                        rtol=1e-15)
        # 调用 to_mlab_linkage 函数，期望其返回值与 Zm 相等，数据类型为 float64
        xp_assert_close(to_mlab_linkage(Z), xp.asarray(Zm, dtype=xp.float64),
                        rtol=1e-15)

    @skip_xp_backends(cpu_only=True)
    def test_mlab_linkage_conversion_multiple_rows(self, xp):
        # Tests from/to_mlab_linkage on linkage array with multiple rows.
        # 创建一个包含多行数据的 mlab linkage 数组 Zm
        Zm = xp.asarray([[3, 6, 138], [4, 5, 219],
                         [1, 8, 255], [2, 9, 268], [7, 10, 295]])
        # 创建一个对应的 Numpy 数组 Z，数据类型为 float64
        Z = xp.asarray([[2., 5., 138., 2.],
                        [3., 4., 219., 2.],
                        [0., 7., 255., 3.],
                        [1., 8., 268., 4.],
                        [6., 9., 295., 6.]],
                       dtype=xp.float64)
        # 调用 from_mlab_linkage 函数，期望其返回值与 Z 相等，rtol 设置为 1e-15
        xp_assert_close(from_mlab_linkage(Zm), Z, rtol=1e-15)
        # 调用 to_mlab_linkage 函数，期望其返回值与 Zm 相等，数据类型为 float64，rtol 设置为 1e-15
        xp_assert_close(to_mlab_linkage(Z), xp.asarray(Zm, dtype=xp.float64),
                        rtol=1e-15)
@skip_xp_backends(cpu_only=True)
class TestFcluster:
    
    def test_fclusterdata(self, xp):
        # 遍历 hierarchy_test_data.fcluster_inconsistent 中的数据进行测试
        for t in hierarchy_test_data.fcluster_inconsistent:
            # 调用 self.check_fclusterdata 方法，测试 fclusterdata(X, criterion='inconsistent', t=t)
            self.check_fclusterdata(t, 'inconsistent', xp)
        # 遍历 hierarchy_test_data.fcluster_distance 中的数据进行测试
        for t in hierarchy_test_data.fcluster_distance:
            # 调用 self.check_fclusterdata 方法，测试 fclusterdata(X, criterion='distance', t=t)
            self.check_fclusterdata(t, 'distance', xp)
        # 遍历 hierarchy_test_data.fcluster_maxclust 中的数据进行测试
        for t in hierarchy_test_data.fcluster_maxclust:
            # 调用 self.check_fclusterdata 方法，测试 fclusterdata(X, criterion='maxclust', t=t)
            self.check_fclusterdata(t, 'maxclust', xp)

    def check_fclusterdata(self, t, criterion, xp):
        # 在随机生成的三类数据集上测试 fclusterdata(X, criterion=criterion, t=t)
        expectedT = xp.asarray(getattr(hierarchy_test_data, 'fcluster_' + criterion)[t])
        X = xp.asarray(hierarchy_test_data.Q_X)
        # 调用 fclusterdata 方法，返回结果赋给 T
        T = fclusterdata(X, criterion=criterion, t=t)
        # 断言 T 与预期结果 expectedT 的同构性
        assert_(is_isomorphic(T, expectedT))

    def test_fcluster(self, xp):
        # 遍历 hierarchy_test_data.fcluster_inconsistent 中的数据进行测试
        for t in hierarchy_test_data.fcluster_inconsistent:
            # 调用 self.check_fcluster 方法，测试 fcluster(Z, criterion='inconsistent', t=t)
            self.check_fcluster(t, 'inconsistent', xp)
        # 遍历 hierarchy_test_data.fcluster_distance 中的数据进行测试
        for t in hierarchy_test_data.fcluster_distance:
            # 调用 self.check_fcluster 方法，测试 fcluster(Z, criterion='distance', t=t)
            self.check_fcluster(t, 'distance', xp)
        # 遍历 hierarchy_test_data.fcluster_maxclust 中的数据进行测试
        for t in hierarchy_test_data.fcluster_maxclust:
            # 调用 self.check_fcluster 方法，测试 fcluster(Z, criterion='maxclust', t=t)
            self.check_fcluster(t, 'maxclust', xp)

    def check_fcluster(self, t, criterion, xp):
        # 在随机生成的三类数据集上测试 fcluster(Z, criterion=criterion, t=t)
        expectedT = xp.asarray(getattr(hierarchy_test_data, 'fcluster_' + criterion)[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        # 调用 fcluster 方法，返回结果赋给 T
        T = fcluster(Z, criterion=criterion, t=t)
        # 断言 T 与预期结果 expectedT 的同构性
        assert_(is_isomorphic(T, expectedT))

    def test_fcluster_monocrit(self, xp):
        # 遍历 hierarchy_test_data.fcluster_distance 中的数据进行测试
        for t in hierarchy_test_data.fcluster_distance:
            # 调用 self.check_fcluster_monocrit 方法，测试 fcluster(Z, t, criterion='monocrit', monocrit=maxdists(Z))
            self.check_fcluster_monocrit(t, xp)
        # 遍历 hierarchy_test_data.fcluster_maxclust 中的数据进行测试
        for t in hierarchy_test_data.fcluster_maxclust:
            # 调用 self.check_fcluster_maxclust_monocrit 方法，测试 fcluster(Z, t, criterion='maxclust_monocrit', monocrit=maxdists(Z))
            self.check_fcluster_maxclust_monocrit(t, xp)

    def check_fcluster_monocrit(self, t, xp):
        # 测试 fcluster(Z, t, criterion='monocrit', monocrit=maxdists(Z)) 在 hierarchy_test_data.fcluster_distance[t] 上的表现
        expectedT = xp.asarray(hierarchy_test_data.fcluster_distance[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        # 调用 fcluster 方法，返回结果赋给 T
        T = fcluster(Z, t, criterion='monocrit', monocrit=maxdists(Z))
        # 断言 T 与预期结果 expectedT 的同构性
        assert_(is_isomorphic(T, expectedT))

    def check_fcluster_maxclust_monocrit(self, t, xp):
        # 测试 fcluster(Z, t, criterion='maxclust_monocrit', monocrit=maxdists(Z)) 在 hierarchy_test_data.fcluster_maxclust[t] 上的表现
        expectedT = xp.asarray(hierarchy_test_data.fcluster_maxclust[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        # 调用 fcluster 方法，返回结果赋给 T
        T = fcluster(Z, t, criterion='maxclust_monocrit', monocrit=maxdists(Z))
        # 断言 T 与预期结果 expectedT 的同构性
        assert_(is_isomorphic(T, expectedT))


@skip_xp_backends(cpu_only=True)
class TestLeaders:

    def test_leaders_single(self, xp):
        # 使用 single linkage 生成的平坦聚类测试 leaders 方法
        X = hierarchy_test_data.Q_X
        Y = pdist(X)
        Y = xp.asarray(Y)
        Z = linkage(Y)
        # 调用 fcluster 方法，返回结果赋给 T
        T = fcluster(Z, criterion='maxclust', t=3)
        Lright = (xp.asarray([53, 55, 56]), xp.asarray([2, 3, 1]))
        T = xp.asarray(T, dtype=xp.int32)
        # 调用 leaders 方法，返回结果赋给 L
        L = leaders(Z, T)
        # 断言 L 与预期结果 Lright 的吻合度
        assert_allclose(np.concatenate(L), np.concatenate(Lright), rtol=1e-15)
@skip_xp_backends(np_only=True,
                   reasons=['`is_isomorphic` only supports NumPy backend'])
class TestIsIsomorphic:
    @skip_xp_backends(np_only=True,
                       reasons=['array-likes only supported for NumPy backend'])
    def test_array_like(self, xp):
        # 断言对于非空列表输入，is_isomorphic 函数返回 True
        assert is_isomorphic([1, 1, 1], [2, 2, 2])
        # 断言对于空列表输入，is_isomorphic 函数返回 True
        assert is_isomorphic([], [])

    def test_is_isomorphic_1(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #1 上（一个平坦聚类，不同标签）
        a = xp.asarray([1, 1, 1])
        b = xp.asarray([2, 2, 2])
        assert is_isomorphic(a, b)
        assert is_isomorphic(b, a)

    def test_is_isomorphic_2(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #2 上（两个平坦聚类，不同标签）
        a = xp.asarray([1, 7, 1])
        b = xp.asarray([2, 3, 2])
        assert is_isomorphic(a, b)
        assert is_isomorphic(b, a)

    def test_is_isomorphic_3(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #3 上（没有平坦聚类）
        a = xp.asarray([])
        b = xp.asarray([])
        assert is_isomorphic(a, b)

    def test_is_isomorphic_4A(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #4A 上
        # （3个平坦聚类，不同标签，同构）
        a = xp.asarray([1, 2, 3])
        b = xp.asarray([1, 3, 2])
        assert is_isomorphic(a, b)
        assert is_isomorphic(b, a)

    def test_is_isomorphic_4B(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #4B 上
        # （3个平坦聚类，不同标签，非同构）
        a = xp.asarray([1, 2, 3, 3])
        b = xp.asarray([1, 3, 2, 3])
        assert is_isomorphic(a, b) is False
        assert is_isomorphic(b, a) is False

    def test_is_isomorphic_4C(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #4C 上
        # （3个平坦聚类，不同标签，同构）
        a = xp.asarray([7, 2, 3])
        b = xp.asarray([6, 3, 2])
        assert is_isomorphic(a, b)
        assert is_isomorphic(b, a)

    def test_is_isomorphic_5(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #5 上
        # （1000个观测，2/3/5个随机聚类，标签随机排列）
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc, xp=xp)

    def test_is_isomorphic_6(self, xp):
        # 测试 is_isomorphic 函数在测试用例 #5A 上
        # （1000个观测，2/3/5个随机聚类，标签随机排列，略微非同构）
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc, True, 5, xp=xp)

    def test_is_isomorphic_7(self, xp):
        # gh-6271 的回归测试
        a = xp.asarray([1, 2, 3])
        b = xp.asarray([1, 1, 1])
        assert not is_isomorphic(a, b)
    # 定义一个方法 help_is_isomorphic_randperm，接受多个参数：观测数目 nobs、簇数目 nclusters、是否非同构 noniso、错误数目 nerrors、关键字参数 xp
    def help_is_isomorphic_randperm(self, nobs, nclusters, noniso=False, nerrors=0, *, xp):
        # 进行三次循环，每次生成一个随机数组 a，其元素为 nobs 个 [0, nclusters) 范围内的整数
        for k in range(3):
            # 生成随机数组 a，使用 numpy.random.rand 生成 nobs 个 [0, 1) 之间的随机数，乘以 nclusters 后取整作为数组 a 的元素
            a = (np.random.rand(nobs) * nclusters).astype(int)
            # 初始化数组 b，长度与 a 相同，元素类型为整数
            b = np.zeros(a.size, dtype=int)
            # 生成一个 nclusters 的随机排列 P，作为置换矩阵
            P = np.random.permutation(nclusters)
            # 根据置换矩阵 P 对数组 a 进行置换，得到数组 b
            for i in range(0, a.shape[0]):
                b[i] = P[a[i]]
            # 如果 noniso 为 True，则生成一个 nobs 的随机排列 Q，将 b 的前 nerrors 个元素增加 1 并对 nclusters 取模
            if noniso:
                Q = np.random.permutation(nobs)
                b[Q[0:nerrors]] += 1
                b[Q[0:nerrors]] %= nclusters
            # 将数组 a 和 b 转换为 xp (可能是 numpy 或 cupy) 的数组
            a = xp.asarray(a)
            b = xp.asarray(b)
            # 断言 is_isomorphic(a, b) 的结果与 (not noniso) 相同，用于验证是否同构
            assert is_isomorphic(a, b) == (not noniso)
            # 断言 is_isomorphic(b, a) 的结果与 (not noniso) 相同，用于验证是否同构
            assert is_isomorphic(b, a) == (not noniso)
# 使用装饰器 @skip_xp_backends，设置仅在 CPU 模式下运行测试
@skip_xp_backends(cpu_only=True)
# 定义测试类 TestIsValidLinkage
class TestIsValidLinkage:

    # 定义测试方法 test_is_valid_linkage_various_size，接受参数 xp
    def test_is_valid_linkage_various_size(self, xp):
        # 遍历包含不同参数组合的列表
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False),
                                  (1, 4, True), (2, 4, True)]:
            # 调用 self.check_is_valid_linkage_various_size 方法进行测试
            self.check_is_valid_linkage_various_size(nrow, ncol, valid, xp)

    # 定义方法 check_is_valid_linkage_various_size，接受参数 nrow, ncol, valid, xp
    def check_is_valid_linkage_various_size(self, nrow, ncol, valid, xp):
        # 创建一个二维数组 Z，使用 xp.asarray 转换为 xp 的数组，数据类型为 xp.float64
        Z = xp.asarray([[0, 1, 3.0, 2, 5],
                        [3, 2, 4.0, 3, 3]], dtype=xp.float64)
        # 根据给定的 nrow 和 ncol 对 Z 进行切片操作
        Z = Z[:nrow, :ncol]
        # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否与 valid 相同
        assert_(is_valid_linkage(Z) == valid)
        # 如果 valid 为 False，则使用 assert_raises 检查是否抛出 ValueError 异常
        if not valid:
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    # 定义方法 test_is_valid_linkage_int_type，接受参数 xp
    def test_is_valid_linkage_int_type(self, xp):
        # 创建一个二维数组 Z，使用 xp.asarray 转换为 xp 的数组，数据类型为 xp.int64
        Z = xp.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=xp.int64)
        # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否为 False
        assert_(is_valid_linkage(Z) is False)
        # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, is_valid_linkage, Z, throw=True)

    # 定义方法 test_is_valid_linkage_empty，接受参数 xp
    def test_is_valid_linkage_empty(self, xp):
        # 创建一个形状为 (0, 4) 的零数组 Z，数据类型为 xp.float64
        Z = xp.zeros((0, 4), dtype=xp.float64)
        # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否为 False
        assert_(is_valid_linkage(Z) is False)
        # 使用 assert_raises 检查是否抛出 ValueError 异常
        assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    # 定义方法 test_is_valid_linkage_4_and_up，接受参数 xp
    def test_is_valid_linkage_4_and_up(self, xp):
        # 遍历从 4 到 15（步长为 3）的整数序列
        for i in range(4, 15, 3):
            # 创建一个长度为 i*(i-1)//2 的随机数组 y，转换为 xp 的数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z
            Z = linkage(y)
            # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否为 True
            assert_(is_valid_linkage(Z) is True)

    # 使用装饰器 @skip_xp_backends，设置仅在 CPU 模式下运行测试，并附带原因说明
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    # 定义方法 test_is_valid_linkage_4_and_up_neg_index_left，接受参数 xp
    def test_is_valid_linkage_4_and_up_neg_index_left(self, xp):
        # 遍历从 4 到 15（步长为 3）的整数序列
        for i in range(4, 15, 3):
            # 创建一个长度为 i*(i-1)//2 的随机数组 y，转换为 xp 的数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z
            Z = linkage(y)
            # 修改 Z[i//2,0] 的值为 -2
            Z[i//2,0] = -2
            # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否为 False
            assert_(is_valid_linkage(Z) is False)
            # 使用 assert_raises 检查是否抛出 ValueError 异常
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    # 使用装饰器 @skip_xp_backends，设置仅在 CPU 模式下运行测试，并附带原因说明
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    # 定义方法 test_is_valid_linkage_4_and_up_neg_index_right，接受参数 xp
    def test_is_valid_linkage_4_and_up_neg_index_right(self, xp):
        # 遍历从 4 到 15（步长为 3）的整数序列
        for i in range(4, 15, 3):
            # 创建一个长度为 i*(i-1)//2 的随机数组 y，转换为 xp 的数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z
            Z = linkage(y)
            # 修改 Z[i//2,1] 的值为 -2
            Z[i//2,1] = -2
            # 使用 assert_ 验证 is_valid_linkage(Z) 的结果是否为 False
            assert_(is_valid_linkage(Z) is False)
            # 使用 assert_raises 检查是否抛出 ValueError 异常
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_is_valid_linkage_4_and_up_neg_dist(self, xp):
        # 使用装饰器跳过特定的后端（如'jax.numpy'），因为它们不支持项赋值操作，仅限于 CPU 模式
        # 定义测试函数，检查 is_valid_linkage(Z) 在链接观察集合大小在4到15之间（步长为3）且具有负距离时的行为。
        for i in range(4, 15, 3):
            # 创建大小为 i*(i-1)//2 的随机数组 y，并将其转换为特定后端的数组表示（xp）
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 根据数组 y 计算其层次聚类链接 Z
            Z = linkage(y)
            # 修改链接矩阵 Z 中索引为 [i//2, 2] 处的值为 -0.5
            Z[i//2,2] = -0.5
            # 断言 is_valid_linkage(Z) 的返回值为 False
            assert_(is_valid_linkage(Z) is False)
            # 断言调用 is_valid_linkage(Z, throw=True) 会引发 ValueError 异常
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_is_valid_linkage_4_and_up_neg_counts(self, xp):
        # 使用装饰器跳过特定的后端（如'jax.numpy'），因为它们不支持项赋值操作，仅限于 CPU 模式
        # 定义测试函数，检查 is_valid_linkage(Z) 在链接观察集合大小在4到15之间（步长为3）且具有负计数时的行为。
        for i in range(4, 15, 3):
            # 创建大小为 i*(i-1)//2 的随机数组 y，并将其转换为特定后端的数组表示（xp）
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 根据数组 y 计算其层次聚类链接 Z
            Z = linkage(y)
            # 修改链接矩阵 Z 中索引为 [i//2, 3] 处的值为 -2
            Z[i//2,3] = -2
            # 断言 is_valid_linkage(Z) 的返回值为 False
            assert_(is_valid_linkage(Z) is False)
            # 断言调用 is_valid_linkage(Z, throw=True) 会引发 ValueError 异常
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)
# 使用装饰器跳过特定的 XP 后端，限定仅使用 CPU
@skip_xp_backends(cpu_only=True)
# 定义一个测试类 TestIsValidInconsistent
class TestIsValidInconsistent:

    # 定义测试方法 test_is_valid_im_int_type，测试 is_valid_im(R) 对整数类型的处理
    def test_is_valid_im_int_type(self, xp):
        # 创建一个二维数组 R，包含整数和浮点数混合的数据
        R = xp.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=xp.int64)
        # 断言 is_valid_im(R) 返回 False
        assert_(is_valid_im(R) is False)
        # 断言 is_valid_im(R) 抛出 TypeError 异常
        assert_raises(TypeError, is_valid_im, R, throw=True)

    # 定义测试方法 test_is_valid_im_various_size，测试不同大小的链接矩阵对 is_valid_im(R) 的影响
    def test_is_valid_im_various_size(self, xp):
        # 遍历不同的 nrow、ncol 组合及其有效性
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False),
                                  (1, 4, True), (2, 4, True)]:
            # 调用 check_is_valid_im_various_size 方法进行测试
            self.check_is_valid_im_various_size(nrow, ncol, valid, xp)

    # 定义检查链接矩阵各种大小的方法 check_is_valid_im_various_size
    def check_is_valid_im_various_size(self, nrow, ncol, valid, xp):
        # 创建一个二维数组 R，包含浮点数数据
        R = xp.asarray([[0, 1, 3.0, 2, 5],
                        [3, 2, 4.0, 3, 3]], dtype=xp.float64)
        # 裁剪 R 为指定的 nrow 行和 ncol 列
        R = R[:nrow, :ncol]
        # 断言 is_valid_im(R) 返回有效性 valid
        assert_(is_valid_im(R) == valid)
        # 如果有效性为 False，断言 is_valid_im(R) 抛出 ValueError 异常
        if not valid:
            assert_raises(ValueError, is_valid_im, R, throw=True)

    # 定义测试方法 test_is_valid_im_empty，测试空的不一致性矩阵对 is_valid_im(R) 的影响
    def test_is_valid_im_empty(self, xp):
        # 创建一个形状为 (0, 4) 的全零浮点数数组 R
        R = xp.zeros((0, 4), dtype=xp.float64)
        # 断言 is_valid_im(R) 返回 False
        assert_(is_valid_im(R) is False)
        # 断言 is_valid_im(R) 抛出 ValueError 异常
        assert_raises(ValueError, is_valid_im, R, throw=True)

    # 定义测试方法 test_is_valid_im_4_and_up，测试不同观测集大小（4 到 15，步长为 3）的链接矩阵对 is_valid_im(R) 的影响
    def test_is_valid_im_4_and_up(self, xp):
        # 遍历观测集大小从 4 到 15，步长为 3
        for i in range(4, 15, 3):
            # 创建一个包含随机浮点数的数组 y，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z，再计算其不一致性矩阵 R
            Z = linkage(y)
            R = inconsistent(Z)
            # 断言 is_valid_im(R) 返回 True
            assert_(is_valid_im(R) is True)

    # 使用装饰器跳过特定的 XP 后端，并提供跳过的原因
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    # 定义测试方法 test_is_valid_im_4_and_up_neg_index_left，测试带有负索引高度的链接矩阵对 is_valid_im(R) 的影响
    def test_is_valid_im_4_and_up_neg_index_left(self, xp):
        # 遍历观测集大小从 4 到 15，步长为 3
        for i in range(4, 15, 3):
            # 创建一个包含随机浮点数的数组 y，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z，再计算其不一致性矩阵 R
            Z = linkage(y)
            R = inconsistent(Z)
            # 修改 R 的指定索引处的值为 -2.0
            R[i//2,0] = -2.0
            # 断言 is_valid_im(R) 返回 False
            assert_(is_valid_im(R) is False)
            # 断言 is_valid_im(R) 抛出 ValueError 异常
            assert_raises(ValueError, is_valid_im, R, throw=True)

    # 使用装饰器跳过特定的 XP 后端，并提供跳过的原因
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    # 定义测试方法 test_is_valid_im_4_and_up_neg_index_right，测试带有负索引标准差的链接矩阵对 is_valid_im(R) 的影响
    def test_is_valid_im_4_and_up_neg_index_right(self, xp):
        # 遍历观测集大小从 4 到 15，步长为 3
        for i in range(4, 15, 3):
            # 创建一个包含随机浮点数的数组 y，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 linkage 函数生成 Z，再计算其不一致性矩阵 R
            Z = linkage(y)
            R = inconsistent(Z)
            # 修改 R 的指定索引处的值为 -2.0
            R[i//2,1] = -2.0
            # 断言 is_valid_im(R) 返回 False
            assert_(is_valid_im(R) is False)
            # 断言 is_valid_im(R) 抛出 ValueError 异常
            assert_raises(ValueError, is_valid_im, R, throw=True)
    # 使用装饰器 @skip_xp_backends 来跳过特定的 Xarray 后端，针对 'jax.numpy' 这里禁止使用
    # 给装饰器传递参数 reasons 和 cpu_only，reasons 参数说明了跳过的原因是 'jax arrays do not support item assignment'，
    # cpu_only 参数指定只在 CPU 上运行这个测试函数
    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    # 定义测试函数 test_is_valid_im_4_and_up_neg_dist，测试 is_valid_im(R) 函数对不同大小（从4到15，步长为3）的 im 对象的有效性
    def test_is_valid_im_4_and_up_neg_dist(self, xp):
        # 循环遍历从4到15（步长为3）的整数范围
        for i in range(4, 15, 3):
            # 生成随机数组 y，长度为 i*(i-1)//2
            y = np.random.rand(i*(i-1)//2)
            # 将随机数组 y 转换为 xp 对象（根据传入的 xp 参数可以是 numpy 或者其他）
            y = xp.asarray(y)
            # 对数组 y 进行层次聚类，生成链接矩阵 Z
            Z = linkage(y)
            # 计算不一致系数矩阵 R
            R = inconsistent(Z)
            # 修改 R 的某个元素，将 R[i//2,2] 设置为 -0.5
            R[i//2,2] = -0.5
            # 断言 is_valid_im(R) 返回 False
            assert_(is_valid_im(R) is False)
            # 使用 assert_raises 断言 is_valid_im(R) 抛出 ValueError 异常，确保 throw=True
            assert_raises(ValueError, is_valid_im, R, throw=True)
class TestNumObsLinkage:

    @skip_xp_backends(cpu_only=True)
    def test_num_obs_linkage_empty(self, xp):
        # Tests num_obs_linkage(Z) with empty linkage.
        Z = xp.zeros((0, 4), dtype=xp.float64)
        # Asserts that calling num_obs_linkage with empty linkage raises ValueError
        assert_raises(ValueError, num_obs_linkage, Z)

    def test_num_obs_linkage_1x4(self, xp):
        # Tests num_obs_linkage(Z) on linkage over 2 observations.
        Z = xp.asarray([[0, 1, 3.0, 2]], dtype=xp.float64)
        # Asserts that num_obs_linkage returns 2 for a linkage of 2 observations
        assert_equal(num_obs_linkage(Z), 2)

    def test_num_obs_linkage_2x4(self, xp):
        # Tests num_obs_linkage(Z) on linkage over 3 observations.
        Z = xp.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=xp.float64)
        # Asserts that num_obs_linkage returns 3 for a linkage of 3 observations
        assert_equal(num_obs_linkage(Z), 3)

    @skip_xp_backends(cpu_only=True)
    def test_num_obs_linkage_4_and_up(self, xp):
        # Tests num_obs_linkage(Z) on linkage on observation sets between sizes
        # 4 and 15 (step size 3).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            Z = linkage(y)
            # Asserts that num_obs_linkage returns i for linkages of sizes 4, 7, 10, 13
            assert_equal(num_obs_linkage(Z), i)


@skip_xp_backends(cpu_only=True)
class TestLeavesList:

    def test_leaves_list_1x4(self, xp):
        # Tests leaves_list(Z) on a 1x4 linkage.
        Z = xp.asarray([[0, 1, 3.0, 2]], dtype=xp.float64)
        # Converts linkage matrix Z to a tree representation
        to_tree(Z)
        # Asserts that leaves_list returns [0, 1] for a linkage of 2 observations
        assert_allclose(leaves_list(Z), [0, 1], rtol=1e-15)

    def test_leaves_list_2x4(self, xp):
        # Tests leaves_list(Z) on a 2x4 linkage.
        Z = xp.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=xp.float64)
        # Converts linkage matrix Z to a tree representation
        to_tree(Z)
        # Asserts that leaves_list returns [0, 1, 2] for a linkage of 3 observations
        assert_allclose(leaves_list(Z), [0, 1, 2], rtol=1e-15)

    def test_leaves_list_Q(self, xp):
        for method in ['single', 'complete', 'average', 'weighted', 'centroid',
                       'median', 'ward']:
            self.check_leaves_list_Q(method, xp)

    def check_leaves_list_Q(self, method, xp):
        # Tests leaves_list(Z) on the Q data set
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, method)
        node = to_tree(Z)
        # Asserts that leaves_list returns the pre-order traversal of nodes for Q data
        assert_allclose(node.pre_order(), leaves_list(Z), rtol=1e-15)

    def test_Q_subtree_pre_order(self, xp):
        # Tests that pre_order() works when called on sub-trees.
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, 'single')
        node = to_tree(Z)
        # Asserts that pre_order() of the root node equals the concatenation of
        # pre_order() results from its left and right subtrees
        assert_allclose(node.pre_order(), (node.get_left().pre_order()
                                           + node.get_right().pre_order()),
                        rtol=1e-15)


@skip_xp_backends(cpu_only=True)
class TestCorrespond:

    def test_correspond_empty(self, xp):
        # Tests correspond(Z, y) with empty linkage and condensed distance matrix.
        y = xp.zeros((0,), dtype=xp.float64)
        Z = xp.zeros((0,4), dtype=xp.float64)
        # Asserts that calling correspond with empty linkage and y raises ValueError
        assert_raises(ValueError, correspond, Z, y)
    def test_correspond_2_and_up(self, xp):
        # 对 linkage 和 CDMs 在不同大小的观测集上进行 correspond(Z, y) 测试
        for i in range(2, 4):
            # 生成大小为 i*(i-1)//2 的随机数组 y，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 y 计算 linkage
            Z = linkage(y)
            # 断言 correspond(Z, y) 返回 True
            assert_(correspond(Z, y))
        for i in range(4, 15, 3):
            # 生成大小为 i*(i-1)//2 的随机数组 y，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y = xp.asarray(y)
            # 使用 y 计算 linkage
            Z = linkage(y)
            # 断言 correspond(Z, y) 返回 True
            assert_(correspond(Z, y))

    def test_correspond_4_and_up(self, xp):
        # 对 linkage 和 CDMs 在不同大小的观测集上进行 correspond(Z, y) 测试，预期返回 False
        for (i, j) in (list(zip(list(range(2, 4)), list(range(3, 5)))) +
                       list(zip(list(range(3, 5)), list(range(2, 4))))):
            # 生成大小为 i*(i-1)//2 的随机数组 y 和大小为 j*(j-1)//2 的随机数组 y2，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y2 = np.random.rand(j*(j-1)//2)
            y = xp.asarray(y)
            y2 = xp.asarray(y2)
            # 使用 y 计算 linkage 和 y2 计算 linkage
            Z = linkage(y)
            Z2 = linkage(y2)
            # 断言 correspond(Z, y2) 返回 False，以及 correspond(Z2, y) 返回 False
            assert not correspond(Z, y2)
            assert not correspond(Z2, y)

    def test_correspond_4_and_up_2(self, xp):
        # 对 linkage 和 CDMs 在不同大小的观测集上进行 correspond(Z, y) 测试，预期返回 False
        for (i, j) in (list(zip(list(range(2, 7)), list(range(16, 21)))) +
                       list(zip(list(range(2, 7)), list(range(16, 21))))):
            # 生成大小为 i*(i-1)//2 的随机数组 y 和大小为 j*(j-1)//2 的随机数组 y2，并转换为 xp 数组
            y = np.random.rand(i*(i-1)//2)
            y2 = np.random.rand(j*(j-1)//2)
            y = xp.asarray(y)
            y2 = xp.asarray(y2)
            # 使用 y 计算 linkage 和 y2 计算 linkage
            Z = linkage(y)
            Z2 = linkage(y2)
            # 断言 correspond(Z, y2) 返回 False，以及 correspond(Z2, y) 返回 False
            assert not correspond(Z, y2)
            assert not correspond(Z2, y)

    def test_num_obs_linkage_multi_matrix(self, xp):
        # 对具有多种大小观测矩阵的 num_obs_linkage 进行测试
        for n in range(2, 10):
            # 生成大小为 n x 4 的随机矩阵 X，并计算其距离矩阵 Y
            X = np.random.rand(n, 4)
            Y = pdist(X)
            Y = xp.asarray(Y)
            # 使用 Y 计算 linkage
            Z = linkage(Y)
            # 断言 num_obs_linkage(Z) 返回值等于 n
            assert_equal(num_obs_linkage(Z), n)
@skip_xp_backends(cpu_only=True)
class TestIsMonotonic:
    # 定义一个测试类 TestIsMonotonic，并标记为在 CPU 上运行的测试

    def test_is_monotonic_empty(self, xp):
        # 测试 is_monotonic(Z) 在空 linkage 上的行为。期望引发 ValueError 异常。
        Z = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, is_monotonic, Z)

    def test_is_monotonic_1x4(self, xp):
        # 测试 is_monotonic(Z) 在 1x4 linkage 上的行为。期望返回 True。
        Z = xp.asarray([[0, 1, 0.3, 2]], dtype=xp.float64)
        assert is_monotonic(Z)

    def test_is_monotonic_2x4_T(self, xp):
        # 测试 is_monotonic(Z) 在 2x4 linkage 上的行为。期望返回 True。
        Z = xp.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 3]], dtype=xp.float64)
        assert is_monotonic(Z)

    def test_is_monotonic_2x4_F(self, xp):
        # 测试 is_monotonic(Z) 在 2x4 linkage 上的行为。期望返回 False。
        Z = xp.asarray([[0, 1, 0.4, 2],
                        [2, 3, 0.3, 3]], dtype=xp.float64)
        assert not is_monotonic(Z)

    def test_is_monotonic_3x4_T(self, xp):
        # 测试 is_monotonic(Z) 在 3x4 linkage 上的行为。期望返回 True。
        Z = xp.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.6, 4]], dtype=xp.float64)
        assert is_monotonic(Z)

    def test_is_monotonic_3x4_F1(self, xp):
        # 测试 is_monotonic(Z) 在 3x4 linkage 上的行为（案例 1）。期望返回 False。
        Z = xp.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.2, 2],
                        [4, 5, 0.6, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    def test_is_monotonic_3x4_F2(self, xp):
        # 测试 is_monotonic(Z) 在 3x4 linkage 上的行为（案例 2）。期望返回 False。
        Z = xp.asarray([[0, 1, 0.8, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.6, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    def test_is_monotonic_3x4_F3(self, xp):
        # 测试 is_monotonic(Z) 在 3x4 linkage 上的行为（案例 3）。期望返回 False。
        Z = xp.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.2, 4]], dtype=xp.float64)
        assert not is_monotonic(Z)

    def test_is_monotonic_tdist_linkage1(self, xp):
        # 测试 is_monotonic(Z) 在使用 tdist 数据集上单链接聚类生成的 linkage 上的行为。期望返回 True。
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        assert is_monotonic(Z)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_is_monotonic_tdist_linkage2(self, xp):
        # 测试 is_monotonic(Z) 在使用 tdist 数据集上单链接聚类生成的 linkage 上的行为，进行扰动。期望返回 False。
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        Z[2,2] = 0.0
        assert not is_monotonic(Z)
    # 定义测试方法 test_is_monotonic_Q_linkage，用于测试在 Q 数据集上通过单链接聚类生成的聚类结果 Z 是否单调性的检查
    def test_is_monotonic_Q_linkage(self, xp):
        # 将 Q 数据集的特征矩阵转换为数组，并用 xp.asarray() 方法封装
        X = xp.asarray(hierarchy_test_data.Q_X)
        # 使用单链接法进行层次聚类，生成聚类结果 Z
        Z = linkage(X, 'single')
        # 断言聚类结果 Z 是否满足单调性条件，即聚类距离严格单调递增
        assert is_monotonic(Z)
@skip_xp_backends(cpu_only=True)
class TestMaxDists:

    def test_maxdists_empty_linkage(self, xp):
        # Tests maxdists(Z) on empty linkage. Expecting exception.
        Z = xp.zeros((0, 4), dtype=xp.float64)
        # 断言调用 maxdists 函数时会抛出 ValueError 异常
        assert_raises(ValueError, maxdists, Z)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxdists_one_cluster_linkage(self, xp):
        # Tests maxdists(Z) on linkage with one cluster.
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        # 计算 maxdists(Z) 的结果
        MD = maxdists(Z)
        # 计算预期的最大距离值
        expectedMD = calculate_maximum_distances(Z, xp)
        # 断言 MD 与 expectedMD 在给定容差下的接近程度
        xp_assert_close(MD, expectedMD, atol=1e-15)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxdists_Q_linkage(self, xp):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            # 对每种聚类方法调用 check_maxdists_Q_linkage 方法
            self.check_maxdists_Q_linkage(method, xp)

    def check_maxdists_Q_linkage(self, method, xp):
        # Tests maxdists(Z) on the Q data set
        X = xp.asarray(hierarchy_test_data.Q_X)
        # 对数据集 X 应用给定方法计算 linkage 矩阵 Z
        Z = linkage(X, method)
        # 计算 maxdists(Z) 的结果
        MD = maxdists(Z)
        # 计算预期的最大距离值
        expectedMD = calculate_maximum_distances(Z, xp)
        # 断言 MD 与 expectedMD 在给定容差下的接近程度
        xp_assert_close(MD, expectedMD, atol=1e-15)


class TestMaxInconsts:

    @skip_xp_backends(cpu_only=True)
    def test_maxinconsts_empty_linkage(self, xp):
        # Tests maxinconsts(Z, R) on empty linkage. Expecting exception.
        Z = xp.zeros((0, 4), dtype=xp.float64)
        R = xp.zeros((0, 4), dtype=xp.float64)
        # 断言调用 maxinconsts 函数时会抛出 ValueError 异常
        assert_raises(ValueError, maxinconsts, Z, R)

    def test_maxinconsts_difrow_linkage(self, xp):
        # Tests maxinconsts(Z, R) on linkage and inconsistency matrices with
        # different numbers of clusters. Expecting exception.
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = np.random.rand(2, 4)
        R = xp.asarray(R)
        # 断言调用 maxinconsts 函数时会抛出 ValueError 异常
        assert_raises(ValueError, maxinconsts, Z, R)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxinconsts_one_cluster_linkage(self, xp):
        # Tests maxinconsts(Z, R) on linkage with one cluster.
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
        # 计算 maxinconsts(Z, R) 的结果
        MD = maxinconsts(Z, R)
        # 计算预期的最大不一致性值
        expectedMD = calculate_maximum_inconsistencies(Z, R, xp=xp)
        # 断言 MD 与 expectedMD 在给定容差下的接近程度
        xp_assert_close(MD, expectedMD, atol=1e-15)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxinconsts_Q_linkage(self, xp):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            # 对每种聚类方法调用 check_maxinconsts_Q_linkage 方法
            self.check_maxinconsts_Q_linkage(method, xp)
    # 定义一个方法用于测试 maxinconsts(Z, R) 在 Q 数据集上的表现
    def check_maxinconsts_Q_linkage(self, method, xp):
        # 使用 hierarchy_test_data.Q_X 创建一个 NumPy 数组 X
        X = xp.asarray(hierarchy_test_data.Q_X)
        # 对数组 X 使用给定的聚类方法 method 进行层次聚类，返回聚类结果 Z
        Z = linkage(X, method)
        # 计算聚类结果 Z 的不一致性值 R
        R = inconsistent(Z)
        # 计算 Z 和 R 的最大一致性不足 MD
        MD = maxinconsts(Z, R)
        # 计算预期的最大一致性不足值 expectedMD
        expectedMD = calculate_maximum_inconsistencies(Z, R, xp=xp)
        # 使用 xp_assert_close 函数检查 MD 和 expectedMD 的近似性，允许误差为 1e-15
        xp_assert_close(MD, expectedMD, atol=1e-15)
class TestMaxRStat:

    def test_maxRstat_invalid_index(self, xp):
        # 对于不合法的索引进行测试，期望抛出异常
        for i in [3.3, -1, 4]:
            self.check_maxRstat_invalid_index(i, xp)

    def check_maxRstat_invalid_index(self, i, xp):
        # 测试 maxRstat(Z, R, i)，期望抛出异常
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
        if isinstance(i, int):
            # 如果 i 是整数，预期抛出 ValueError 异常
            assert_raises(ValueError, maxRstat, Z, R, i)
        else:
            # 否则，预期抛出 TypeError 异常
            assert_raises(TypeError, maxRstat, Z, R, i)

    @skip_xp_backends(cpu_only=True)
    def test_maxRstat_empty_linkage(self, xp):
        # 对空链接进行测试，期望抛出异常
        for i in range(4):
            self.check_maxRstat_empty_linkage(i, xp)

    def check_maxRstat_empty_linkage(self, i, xp):
        # 测试 maxRstat(Z, R, i) 在空链接上，期望抛出 ValueError 异常
        Z = xp.zeros((0, 4), dtype=xp.float64)
        R = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, maxRstat, Z, R, i)

    def test_maxRstat_difrow_linkage(self, xp):
        # 对链接和不一致性矩阵具有不同聚类数的情况进行测试，期望抛出异常
        for i in range(4):
            self.check_maxRstat_difrow_linkage(i, xp)

    def check_maxRstat_difrow_linkage(self, i, xp):
        # 测试 maxRstat(Z, R, i) 在链接和不一致性矩阵具有不同行数的情况下，期望抛出 ValueError 异常
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = np.random.rand(2, 4)
        R = xp.asarray(R)
        assert_raises(ValueError, maxRstat, Z, R, i)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxRstat_one_cluster_linkage(self, xp):
        # 对只有一个聚类的链接进行测试
        for i in range(4):
            self.check_maxRstat_one_cluster_linkage(i, xp)

    def check_maxRstat_one_cluster_linkage(self, i, xp):
        # 测试 maxRstat(Z, R, i) 在只有一个聚类的链接上
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_maxRstat_Q_linkage(self, xp):
        # 对 Q 数据集进行链接测试
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            for i in range(4):
                self.check_maxRstat_Q_linkage(method, i, xp)

    def check_maxRstat_Q_linkage(self, method, i, xp):
        # 测试 maxRstat(Z, R, i) 在 Q 数据集上
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, method)
        R = inconsistent(Z)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)


@skip_xp_backends(cpu_only=True)
class TestDendrogram:
    # 测试在单链式聚类中使用 tdist 数据集计算树状图
    def test_dendrogram_single_linkage_tdist(self, xp):
        # 将 hierarchy_test_data.ytdist 转换为 NumPy 数组并进行单链式聚类计算
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        # 获取树状图的信息，但不绘制图形
        R = dendrogram(Z, no_plot=True)
        # 获取叶子节点的顺序
        leaves = R["leaves"]
        # 断言叶子节点的顺序是否符合预期
        assert_equal(leaves, [2, 5, 1, 0, 3, 4])

    # 测试在指定不支持的方向参数时是否引发 ValueError 异常
    def test_valid_orientation(self, xp):
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        assert_raises(ValueError, dendrogram, Z, orientation="foo")

    # 测试使用数组或列表作为标签参数时的一致性
    def test_labels_as_array_or_list(self, xp):
        # test for gh-12418
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        # 创建标签数组
        labels = xp.asarray([1, 3, 2, 6, 4, 5])
        # 使用数组作为标签计算树状图，并获取结果
        result1 = dendrogram(Z, labels=labels, no_plot=True)
        # 使用列表作为标签计算树状图，并获取结果
        result2 = dendrogram(Z, labels=list(labels), no_plot=True)
        # 断言两种方式计算的结果是否相同
        assert result1 == result2

    # 测试在不支持的标签大小时是否引发 ValueError 异常
    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_valid_label_size(self, xp):
        # 创建包含链接信息的 NumPy 数组
        link = xp.asarray([
            [0, 1, 1.0, 4],
            [2, 3, 1.0, 5],
            [4, 5, 2.0, 6],
        ])
        # 创建新的 matplotlib 图形对象
        plt.figure()
        # 测试在 Z 和标签大小不一致时是否引发异常
        with pytest.raises(ValueError) as exc_info:
            dendrogram(link, labels=list(range(100)))
        # 断言异常消息中包含特定文本
        assert "Dimensions of Z and labels must be consistent."\
               in str(exc_info.value)

        # 使用匹配方式断言异常消息中是否包含特定文本
        with pytest.raises(
                ValueError,
                match="Dimensions of Z and labels must be consistent."):
            dendrogram(link, labels=[])

        # 关闭当前 matplotlib 图形对象
        plt.close()

    # 测试树状图的绘制
    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_dendrogram_plot(self, xp):
        # 遍历不同的方向参数进行树状图绘制测试
        for orientation in ['top', 'bottom', 'left', 'right']:
            # 使用指定的方向参数检查树状图的绘制
            self.check_dendrogram_plot(orientation, xp)
    # 定义一个方法，用于测试树状图的绘制
    def check_dendrogram_plot(self, orientation, xp):
        # 使用 hierarchy_test_data.ytdist 创建层次聚类的 Z 矩阵
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
        # 预期的树状图结构，包括颜色列表、坐标数据、节点坐标、叶子标签等
        expected = {'color_list': ['C1', 'C0', 'C0', 'C0', 'C0'],
                    'dcoord': [[0.0, 138.0, 138.0, 0.0],
                               [0.0, 219.0, 219.0, 0.0],
                               [0.0, 255.0, 255.0, 219.0],
                               [0.0, 268.0, 268.0, 255.0],
                               [138.0, 295.0, 295.0, 268.0]],
                    'icoord': [[5.0, 5.0, 15.0, 15.0],
                               [45.0, 45.0, 55.0, 55.0],
                               [35.0, 35.0, 50.0, 50.0],
                               [25.0, 25.0, 42.5, 42.5],
                               [10.0, 10.0, 33.75, 33.75]],
                    'ivl': ['2', '5', '1', '0', '3', '4'],
                    'leaves': [2, 5, 1, 0, 3, 4],
                    'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0', 'C0'],
                    }

        # 创建一个新的图形对象
        fig = plt.figure()
        # 添加一个子图到图形中
        ax = fig.add_subplot(221)

        # 测试 dendrogram 函数是否接受 ax 关键字参数
        R1 = dendrogram(Z, ax=ax, orientation=orientation)
        # 将 dcoord 转换为 NumPy 数组
        R1['dcoord'] = np.asarray(R1['dcoord'])
        # 断言 dendrogram 返回的结果与预期的结果相等
        assert_equal(R1, expected)

        # 测试 dendrogram 是否接受并正确处理 leaf_font_size 和 leaf_rotation 关键字参数
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_font_size=20, leaf_rotation=90)
        # 获取标签并检查其旋转角度是否为 90 度
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_rotation(), 90)
        assert_equal(testlabel.get_size(), 20)

        # 测试 dendrogram 是否正确处理 leaf_rotation 关键字参数
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_rotation=90)
        # 再次获取标签并检查其旋转角度是否为 90 度
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_rotation(), 90)

        # 测试 dendrogram 是否正确处理 leaf_font_size 关键字参数
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_font_size=20)
        # 获取标签并检查其字体大小是否为 20
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_size(), 20)

        # 关闭图形对象
        plt.close()

        # 测试将绘图输出到当前轴（gca），这将导入 pylab
        R2 = dendrogram(Z, orientation=orientation)
        # 关闭图形对象
        plt.close()
        # 将 dcoord 转换为 NumPy 数组
        R2['dcoord'] = np.asarray(R2['dcoord'])
        # 断言 dendrogram 返回的结果与预期的结果相等
        assert_equal(R2, expected)

    # 使用 pytest.mark.skipif 进行条件跳过，如果没有 matplotlib 库，则跳过该测试
    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    # 定义测试方法：测试在截断模式下的树状图生成
    def test_dendrogram_truncate_mode(self, xp):
        # 从给定的距离矩阵生成层次聚类树，并使用单链接法
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')

        # 生成截断后的树状图，显示部分被合并的节点
        R = dendrogram(Z, 2, 'lastp', show_contracted=True)
        # 关闭生成的图形窗口
        plt.close()

        # 将树状图结果中的坐标数据转换为 NumPy 数组
        R['dcoord'] = np.asarray(R['dcoord'])

        # 断言树状图的结果与预期的字典形式相等
        assert_equal(R, {'color_list': ['C0'],
                         'dcoord': [[0.0, 295.0, 295.0, 0.0]],
                         'icoord': [[5.0, 5.0, 15.0, 15.0]],
                         'ivl': ['(2)', '(4)'],
                         'leaves': [6, 9],
                         'leaves_color_list': ['C0', 'C0'],
                         })

        # 生成截断后的树状图，使用 mtica 模式
        R = dendrogram(Z, 2, 'mtica', show_contracted=True)
        # 关闭生成的图形窗口
        plt.close()

        # 将树状图结果中的坐标数据转换为 NumPy 数组
        R['dcoord'] = np.asarray(R['dcoord'])

        # 断言树状图的结果与预期的字典形式相等
        assert_equal(R, {'color_list': ['C1', 'C0', 'C0', 'C0'],
                         'dcoord': [[0.0, 138.0, 138.0, 0.0],
                                    [0.0, 255.0, 255.0, 0.0],
                                    [0.0, 268.0, 268.0, 255.0],
                                    [138.0, 295.0, 295.0, 268.0]],
                         'icoord': [[5.0, 5.0, 15.0, 15.0],
                                    [35.0, 35.0, 45.0, 45.0],
                                    [25.0, 25.0, 40.0, 40.0],
                                    [10.0, 10.0, 32.5, 32.5]],
                         'ivl': ['2', '5', '1', '0', '(2)'],
                         'leaves': [2, 5, 1, 0, 7],
                         'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0'],
                         })

    # 定义测试方法：测试树状图叶节点颜色设置
    def test_dendrogram_colors(self, xp):
        # 从给定的距离矩阵生成层次聚类树，并使用单链接法
        Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')

        # 设置连接线颜色的全局调色板
        set_link_color_palette(['c', 'm', 'y', 'k'])

        # 生成树状图，但不显示，设置某些参数如阈值颜色和阈值
        R = dendrogram(Z, no_plot=True,
                       above_threshold_color='g', color_threshold=250)

        # 恢复连接线颜色的全局调色板
        set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])

        # 获取树状图中连接线的颜色列表
        color_list = R['color_list']

        # 断言连接线颜色列表与预期相符
        assert_equal(color_list, ['c', 'm', 'g', 'g', 'g'])

        # 重置连接线颜色的全局调色板
        set_link_color_palette(None)

    # 定义测试方法：测试树状图叶节点颜色设置，当距离为零时
    def test_dendrogram_leaf_colors_zero_dist(self, xp):
        # 测试包含两个相同点的树的叶节点颜色是否正确
        x = xp.asarray([[1, 0, 0],
                        [0, 0, 1],
                        [0, 2, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0]])

        # 从给定的数据矩阵生成层次聚类树，并使用单链接法
        z = linkage(x, "single")

        # 生成树状图，但不显示
        d = dendrogram(z, no_plot=True)

        # 预期的叶节点颜色列表
        exp_colors = ['C0', 'C1', 'C1', 'C0', 'C2', 'C2']

        # 获取树状图中叶节点的颜色列表
        colors = d["leaves_color_list"]

        # 断言叶节点颜色列表与预期相符
        assert_equal(colors, exp_colors)
    # 定义一个测试函数，用于验证树的叶子节点颜色是否正确
    def test_dendrogram_leaf_colors(self, xp):
        # 定义一个示例输入数据 x，包含六个三维点
        x = xp.asarray([[1, 0, 0],
                        [0, 0, 1.1],
                        [0, 2, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0]])
        # 对输入数据进行层次聚类，使用单链接方法
        z = linkage(x, "single")
        # 根据层次聚类结果生成树状图，但不显示（no_plot=True）
        d = dendrogram(z, no_plot=True)
        # 预期的叶子节点颜色列表，用于与生成的树状图进行比较
        exp_colors = ['C0', 'C1', 'C1', 'C0', 'C2', 'C2']
        # 获取实际生成的叶子节点颜色列表
        colors = d["leaves_color_list"]
        # 断言实际颜色列表与预期颜色列表相等
        assert_equal(colors, exp_colors)
# 计算给定距离矩阵的最大距离
def calculate_maximum_distances(Z, xp):
    # 获取样本点数目
    n = Z.shape[0] + 1
    # 创建一个零数组 B，用于存储最大距离
    B = xp.zeros((n-1,), dtype=Z.dtype)
    # 创建一个长度为 3 的零数组 q，用于暂存左右节点的距离和当前节点的高度
    q = xp.zeros((3,))
    # 遍历每一个内部节点
    for i in range(0, n - 1):
        # 将 q 数组置零
        q[:] = 0.0
        # 获取当前节点的左右节点
        left = Z[i, 0]
        right = Z[i, 1]
        # 如果左节点或右节点超过了样本数目 n，则从 B 数组中取出其对应的最大距离
        if left >= n:
            q[0] = B[xp.asarray(left, dtype=xp.int64) - n]
        if right >= n:
            q[1] = B[xp.asarray(right, dtype=xp.int64) - n]
        # 将当前节点的高度放入 q 数组的第三个位置
        q[2] = Z[i, 2]
        # 计算并存储当前节点的最大距离
        B[i] = xp.max(q)
    # 返回最大距离数组 B
    return B


# 计算给定距离矩阵的最大不一致性
def calculate_maximum_inconsistencies(Z, R, k=3, xp=np):
    # 获取样本点数目
    n = Z.shape[0] + 1
    # 确定结果数据类型
    dtype = xp.result_type(Z, R)
    # 创建一个零数组 B，用于存储最大不一致性
    B = xp.zeros((n-1,), dtype=dtype)
    # 创建一个长度为 3 的零数组 q，用于暂存左右节点的距离和当前节点的 R 值
    q = xp.zeros((3,))
    # 遍历每一个内部节点
    for i in range(0, n - 1):
        # 将 q 数组置零
        q[:] = 0.0
        # 获取当前节点的左右节点
        left = Z[i, 0]
        right = Z[i, 1]
        # 如果左节点或右节点超过了样本数目 n，则从 B 数组中取出其对应的最大不一致性
        if left >= n:
            q[0] = B[xp.asarray(left, dtype=xp.int64) - n]
        if right >= n:
            q[1] = B[xp.asarray(right, dtype=xp.int64) - n]
        # 将当前节点的 R 值放入 q 数组的第三个位置
        q[2] = R[i, k]
        # 计算并存储当前节点的最大不一致性
        B[i] = xp.max(q)
    # 返回最大不一致性数组 B
    return B


# 测试不支持的未压缩距离矩阵链接警告
@skip_xp_backends(cpu_only=True)
def test_unsupported_uncondensed_distance_matrix_linkage_warning(xp):
    # 断言会产生 ClusterWarning 警告
    assert_warns(ClusterWarning, linkage, xp.asarray([[0, 1], [1, 0]]))


# 测试欧氏距离链接值错误
def test_euclidean_linkage_value_error(xp):
    # 对每种欧氏距离计算方法进行测试
    for method in scipy.cluster.hierarchy._EUCLIDEAN_METHODS:
        # 断言会产生 ValueError 错误
        assert_raises(ValueError, linkage, xp.asarray([[1, 1], [1, 1]]),
                      method=method, metric='cityblock')


# 测试 2x2 距离矩阵链接
@skip_xp_backends(cpu_only=True)
def test_2x2_linkage(xp):
    # 使用单个样本测试 single 方法和欧氏距离
    Z1 = linkage(xp.asarray([1]), method='single', metric='euclidean')
    # 使用 2x2 距离矩阵测试 single 方法和欧氏距离
    Z2 = linkage(xp.asarray([[0, 1], [0, 0]]), method='single', metric='euclidean')
    # 断言两者结果在指定的误差范围内相等
    xp_assert_close(Z1, Z2, rtol=1e-15)


# 测试节点比较
@skip_xp_backends(np_only=True, reasons=['`cut_tree` uses non-standard indexing'])
def test_node_compare(xp):
    # 设置随机种子
    np.random.seed(23)
    # 创建随机数据集 X
    nobs = 50
    X = np.random.randn(nobs, 4)
    X = xp.asarray(X)
    # 使用 ward 方法进行层次聚类
    Z = scipy.cluster.hierarchy.ward(X)
    # 将层次聚类结果转换为树形结构
    tree = to_tree(Z)
    # 对树的节点进行一系列比较操作
    assert_(tree > tree.get_left())
    assert_(tree.get_right() > tree.get_left())
    assert_(tree.get_right() == tree.get_right())
    assert_(tree.get_right() != tree.get_left())


# 测试 cut_tree 函数
@skip_xp_backends(np_only=True, reasons=['`cut_tree` uses non-standard indexing'])
def test_cut_tree(xp):
    # 设置随机种子
    np.random.seed(23)
    # 创建随机数据集 X
    nobs = 50
    X = np.random.randn(nobs, 4)
    X = xp.asarray(X)
    # 使用 ward 方法进行层次聚类
    Z = scipy.cluster.hierarchy.ward(X)
    # 利用 cut_tree 函数进行切割
    cutree = cut_tree(Z)

    # 断言切割结果的第一列等于从零到 nobs 的整数序列
    xp_assert_close(cutree[:, 0], xp.arange(nobs), rtol=1e-15, check_dtype=False)
    # 断言切割结果的最后一列全为零
    xp_assert_close(cutree[:, -1], xp.zeros(nobs), rtol=1e-15, check_dtype=False)
    # 断言切割结果的最大值为从 nobs-1 到 0 的逆序整数序列
    assert_equal(np.asarray(cutree).max(0), np.arange(nobs - 1, -1, -1))

    # 对切割结果的倒数第五列进行断言
    xp_assert_close(cutree[:, [-5]], cut_tree(Z, n_clusters=5), rtol=1e-15)
    # 对切割结果的倒数第五列和倒数第十列进行断言
    xp_assert_close(cutree[:, [-5, -10]], cut_tree(Z, n_clusters=[5, 10]), rtol=1e-15)
    # 对切割结果的倒数第十列和倒数第五列进行断言
    xp_assert_close(cutree[:, [-10, -5]], cut_tree(Z, n_clusters=[10, 5]), rtol=1e-15)

    # 对聚类树进行排序
    nodes = _order_cluster_tree(Z)
    # 将节点的距离数组转换为NumPy数组，存储在heights中
    heights = xp.asarray([node.dist for node in nodes])
    
    # 调用xp_assert_close函数，验证cut_tree函数的输出与指定高度参数的预期输出在数值上的接近度
    xp_assert_close(
        cutree[:, np.searchsorted(heights, [5])],
        cut_tree(Z, height=5),
        rtol=1e-15
    )
    
    # 同上，验证cut_tree函数对多个高度参数的输出
    xp_assert_close(
        cutree[:, np.searchsorted(heights, [5, 10])],
        cut_tree(Z, height=[5, 10]),
        rtol=1e-15
    )
    
    # 同上，验证cut_tree函数对多个高度参数的输出，参数顺序不同
    xp_assert_close(
        cutree[:, np.searchsorted(heights, [10, 5])],
        cut_tree(Z, height=[10, 5]),
        rtol=1e-15
    )
# 标记为跳过在 CPU 上运行的后端测试
@skip_xp_backends(cpu_only=True)
def test_optimal_leaf_ordering(xp):
    # 使用距离向量 y 进行测试
    Z = optimal_leaf_ordering(linkage(xp.asarray(hierarchy_test_data.ytdist)),
                              xp.asarray(hierarchy_test_data.ytdist))
    # 期望的 Z 值来自 hierarchy_test_data.linkage_ytdist_single_olo
    expectedZ = hierarchy_test_data.linkage_ytdist_single_olo
    # 断言 Z 与期望值非常接近，允许的绝对误差为 1e-10
    xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)

    # 使用观测矩阵 X 进行测试
    Z = optimal_leaf_ordering(linkage(xp.asarray(hierarchy_test_data.X), 'ward'),
                              xp.asarray(hierarchy_test_data.X))
    # 期望的 Z 值来自 hierarchy_test_data.linkage_X_ward_olo
    expectedZ = hierarchy_test_data.linkage_X_ward_olo
    # 断言 Z 与期望值非常接近，允许的绝对误差为 1e-06
    xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)


# 标记为跳过在 NumPy 上运行的后端测试，并提供原因
@skip_xp_backends(np_only=True, reasons=['`Heap` only supports NumPy backend'])
def test_Heap(xp):
    # 将列表 [2, -1, 0, -1.5, 3] 转换为 XP 数组并赋值给 values
    values = xp.asarray([2, -1, 0, -1.5, 3])
    # 创建 Heap 对象
    heap = Heap(values)

    # 获取最小值对
    pair = heap.get_min()
    # 断言最小值对的键为 3，值为 -1.5
    assert_equal(pair['key'], 3)
    assert_equal(pair['value'], -1.5)

    # 移除最小值
    heap.remove_min()
    # 再次获取最小值对
    pair = heap.get_min()
    # 断言最小值对的键为 1，值为 -1
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], -1)

    # 修改索引为 1 的值为 2.5
    heap.change_value(1, 2.5)
    # 再次获取最小值对
    pair = heap.get_min()
    # 断言最小值对的键为 2，值为 0
    assert_equal(pair['key'], 2)
    assert_equal(pair['value'], 0)

    # 连续移除两次最小值
    heap.remove_min()
    heap.remove_min()

    # 修改索引为 1 的值为 10
    heap.change_value(1, 10)
    # 再次获取最小值对
    pair = heap.get_min()
    # 断言最小值对的键为 4，值为 3
    assert_equal(pair['key'], 4)
    assert_equal(pair['value'], 3)

    # 最后一次移除最小值对
    heap.remove_min()
    pair = heap.get_min()
    # 断言最小值对的键为 1，值为 10
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], 10)
```