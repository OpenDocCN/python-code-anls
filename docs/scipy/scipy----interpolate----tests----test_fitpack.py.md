# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_fitpack.py`

```
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import os  # 导入 os 模块，用于处理操作系统相关的功能

import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import (assert_equal, assert_allclose, assert_,
                           assert_almost_equal, assert_array_almost_equal)  # 导入 NumPy 测试模块中的断言函数
from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数，并起别名为 assert_raises
import pytest  # 导入 pytest 库，用于编写和运行测试用例
from scipy._lib._testutils import check_free_memory  # 从 SciPy 库中导入检查空闲内存的函数

from scipy.interpolate import RectBivariateSpline  # 导入 SciPy 中的二维矩形插值函数

from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
     sproot, splprep, splint, spalde, splder, splantider, insert, dblint)  # 从 SciPy 的插值模块中导入各种插值函数
from scipy.interpolate._dfitpack import regrid_smth  # 从 SciPy 的插值模块中导入重网格平滑函数
from scipy.interpolate._fitpack2 import dfitpack_int  # 从 SciPy 的插值模块中导入双精度积分函数


def data_file(basename):
    """返回基于当前文件目录的 data 文件夹中的文件路径."""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)


def norm2(x):
    """计算向量 x 的二范数."""
    return np.sqrt(np.dot(x.T, x))


def f1(x, d=0):
    """计算 sin->cos->-sin->-cos 的导数."""
    if d % 4 == 0:
        return np.sin(x)
    if d % 4 == 1:
        return np.cos(x)
    if d % 4 == 2:
        return -np.sin(x)
    if d % 4 == 3:
        return -np.cos(x)


def makepairs(x, y):
    """辅助函数，创建 x 和 y 的配对数组."""
    xy = np.array(list(itertools.product(np.asarray(x), np.asarray(y))))
    return xy.T


class TestSmokeTests:
    """
    Smoke tests (with a few asserts) for fitpack routines -- mostly
    check that they are runnable
    """
    def check_1(self, per=0, s=0, a=0, b=2*np.pi, at_nodes=False,
                xb=None, xe=None):
        """检查插值函数 splrep 的精度，用于逼近指定的函数."""
        if xb is None:
            xb = a
        if xe is None:
            xe = b

        N = 20
        # 节点和节点中间的点
        x = np.linspace(a, b, N + 1)
        x1 = a + (b - a) * np.arange(1, N, dtype=float) / float(N - 1)
        v = f1(x)

        def err_est(k, d):
            """估算误差上限."""
            # 假设 f 的所有导数均小于 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            if s > 0:
                tol += 1e5*s
            return tol

        for k in range(1, 6):
            tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
            tt = tck[0][k:-k] if at_nodes else x1

            for d in range(k+1):
                tol = err_est(k, d)
                err = norm2(f1(tt, d) - splev(tt, tck, d)) / norm2(f1(tt, d))
                assert err < tol

    def check_2(self, per=0, N=20, ia=0, ib=2*np.pi):
        """检查插值函数 splrep 的另一组参数下的精度."""
        a, b, dx = 0, 2*np.pi, 0.2*np.pi
        x = np.linspace(a, b, N+1)    # 节点
        v = np.sin(x)

        def err_est(k, d):
            """估算误差上限."""
            # 假设 f 的所有导数均小于 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            return tol

        nk = []
        for k in range(1, 6):
            tck = splrep(x, v, s=0, per=per, k=k, xe=b)
            nk.append([splint(ia, ib, tck), spalde(dx, tck)])

        k = 1
        for r in nk:
            d = 0
            for dr in r[1]:
                tol = err_est(k, d)
                assert_allclose(dr, f1(dx, d), atol=0, rtol=tol)
                d = d+1
            k = k+1
    # 调用 self.check_1 方法进行测试，设置参数 s=1e-6
    self.check_1(s=1e-6)
    # 调用 self.check_1 方法进行测试，设置参数 b=1.5*np.pi
    self.check_1(b=1.5*np.pi)
    # 调用 self.check_1 方法进行测试，设置参数 b=1.5*np.pi, xe=2*np.pi, per=1, s=1e-1
    self.check_1(b=1.5*np.pi, xe=2*np.pi, per=1, s=1e-1)

@pytest.mark.parametrize('per', [0, 1])
@pytest.mark.parametrize('at_nodes', [True, False])
def test_smoke_splrep_splev_2(self, per, at_nodes):
    # 调用 self.check_1 方法进行测试，设置参数 per=per, at_nodes=at_nodes
    self.check_1(per=per, at_nodes=at_nodes)

@pytest.mark.parametrize('N', [20, 50])
@pytest.mark.parametrize('per', [0, 1])
def test_smoke_splint_spalde(self, N, per):
    # 调用 self.check_2 方法进行测试，设置参数 N=N, per=per
    self.check_2(per=per, N=N)

@pytest.mark.parametrize('N', [20, 50])
@pytest.mark.parametrize('per', [0, 1])
def test_smoke_splint_spalde_iaib(self, N, per):
    # 调用 self.check_2 方法进行测试，设置参数 ia=0.2*np.pi, ib=np.pi, N=N, per=per
    self.check_2(ia=0.2*np.pi, ib=np.pi, N=N, per=per)

def test_smoke_sproot(self):
    # sproot 仅实现在 k=3 时有效
    a, b = 0.1, 15
    x = np.linspace(a, b, 20)
    v = np.sin(x)

    # 遍历不同的 k 值，验证是否抛出 ValueError 异常
    for k in [1, 2, 4, 5]:
        tck = splrep(x, v, s=0, per=0, k=k, xe=b)
        with assert_raises(ValueError):
            sproot(tck)

    # 对于 k=3 的情况，计算其根并验证其值接近于 0
    k = 3
    tck = splrep(x, v, s=0, k=3)
    roots = sproot(tck)
    assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
    assert_allclose(roots, np.pi * np.array([1, 2, 3, 4]), rtol=1e-3)

@pytest.mark.parametrize('N', [20, 50])
@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_smoke_splprep_splrep_splev(self, N, k):
    a, b, dx = 0, 2.*np.pi, 0.2*np.pi
    x = np.linspace(a, b, N+1)    # nodes
    v = np.sin(x)

    # 使用 splprep 创建参数化的三次样条曲线，并验证误差 err1
    tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
    uv = splev(dx, tckp)
    err1 = abs(uv[1] - np.sin(uv[0]))
    assert err1 < 1e-2

    # 使用 splrep 创建普通的三次样条曲线，并验证误差 err2
    tck = splrep(x, v, s=0, per=0, k=k)
    err2 = abs(splev(uv[0], tck) - np.sin(uv[0]))
    assert err2 < 1e-2

    # 对于 k=3 的情况，计算参数化三次样条曲线在 u 处的导数
    if k == 3:
        tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
        for d in range(1, k+1):
            uv = splev(dx, tckp, d)

def test_smoke_bisplrep_bisplev(self):
    xb, xe = 0, 2.*np.pi
    yb, ye = 0, 2.*np.pi
    kx, ky = 3, 3
    Nx, Ny = 20, 20

    def f2(x, y):
        return np.sin(x+y)

    x = np.linspace(xb, xe, Nx + 1)
    y = np.linspace(yb, ye, Ny + 1)
    xy = makepairs(x, y)

    # 使用 bisplrep 创建二维样条曲线，并计算误差 norm2
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
    tt = [tck[0][kx:-kx], tck[1][ky:-ky]]
    t2 = makepairs(tt[0], tt[1])
    v1 = bisplev(tt[0], tt[1], tck)
    v2 = f2(t2[0], t2[1])
    v2.shape = len(tt[0]), len(tt[1])

    assert norm2(np.ravel(v1 - v2)) < 1e-2
class TestSplev:
    # 定义测试类 TestSplev
    def test_1d_shape(self):
        # 定义测试方法 test_1d_shape
        x = [1,2,3,4,5]  # 定义输入数据 x
        y = [4,5,6,7,8]  # 定义输入数据 y
        tck = splrep(x, y)  # 使用 x 和 y 进行样条插值，返回插值对象 tck
        z = splev([1], tck)  # 对插值对象 tck 进行求值，得到 z
        assert_equal(z.shape, (1,))  # 断言 z 的形状为 (1,)
        z = splev(1, tck)  # 对插值对象 tck 进行求值，得到 z
        assert_equal(z.shape, ())  # 断言 z 的形状为空元组

    def test_2d_shape(self):
        # 定义测试方法 test_2d_shape
        x = [1, 2, 3, 4, 5]  # 定义输入数据 x
        y = [4, 5, 6, 7, 8]  # 定义输入数据 y
        tck = splrep(x, y)  # 使用 x 和 y 进行样条插值，返回插值对象 tck
        t = np.array([[1.0, 1.5, 2.0, 2.5],
                      [3.0, 3.5, 4.0, 4.5]])  # 定义二维数组 t
        z = splev(t, tck)  # 对插值对象 tck 进行求值，得到 z
        z0 = splev(t[0], tck)  # 对插值对象 tck 进行求值，得到 z0
        z1 = splev(t[1], tck)  # 对插值对象 tck 进行求值，得到 z1
        assert_equal(z, np.vstack((z0, z1)))  # 断言 z 等于 z0 和 z1 垂直堆叠后的结果

    def test_extrapolation_modes(self):
        # 定义测试方法 test_extrapolation_modes
        # 测试外推模式
        #    * 如果 ext=0，则返回外推值。
        #    * 如果 ext=1，则返回 0。
        #    * 如果 ext=2，则引发 ValueError。
        #    * 如果 ext=3，则返回边界值。
        x = [1,2,3]  # 定义输入数据 x
        y = [0,2,4]  # 定义输入数据 y
        tck = splrep(x, y, k=1)  # 使用 x 和 y 进行一阶样条插值，返回插值对象 tck

        rstl = [[-2, 6], [0, 0], None, [0, 4]]  # 定义期望结果 rstl
        for ext in (0, 1, 3):
            # 对于每种外推模式进行断言
            assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])

        assert_raises(ValueError, splev, [0, 4], tck, ext=2)


class TestSplder:
    # 定义测试类 TestSplder
    def setup_method(self):
        # 设置方法，使用非均匀网格
        x = np.linspace(0, 1, 100)**3  # 在区间 [0, 1] 内生成非均匀网格 x
        y = np.sin(20 * x)  # 计算 x 上的正弦函数值，得到 y
        self.spl = splrep(x, y)  # 使用 x 和 y 进行样条插值，保存插值对象到 self.spl

        # 确保节点是非均匀的断言
        assert_(np.ptp(np.diff(self.spl[0])) > 0)

    def test_inverse(self):
        # 检查反函数和导数是否为恒等式
        for n in range(5):
            spl2 = splantider(self.spl, n)  # 对插值对象 self.spl 进行 n 次反函数插值
            spl3 = splder(spl2, n)  # 对插值对象 spl2 进行 n 次导数插值
            assert_allclose(self.spl[0], spl3[0])  # 断言插值对象的节点相等
            assert_allclose(self.spl[1], spl3[1])  # 断言插值对象的系数相等
            assert_equal(self.spl[2], spl3[2])  # 断言插值对象的阶数相等

    def test_splder_vs_splev(self):
        # 检查导数与 FITPACK 的比较

        for n in range(3+1):
            # 包括外推!
            xx = np.linspace(-1, 2, 2000)
            if n == 3:
                # ... 除了 FITPACK 对于 n=0 的外推行为奇怪，所以我们不检查那个。
                xx = xx[(xx >= 0) & (xx <= 1)]

            dy = splev(xx, self.spl, n)  # 对插值对象 self.spl 进行 n 次导数求值，得到 dy
            spl2 = splder(self.spl, n)  # 对插值对象 self.spl 进行 n 次导数插值，得到 spl2
            dy2 = splev(xx, spl2)  # 对插值对象 spl2 进行求值，得到 dy2
            if n == 1:
                assert_allclose(dy, dy2, rtol=2e-6)  # 对于 n=1，使用相对误差进行断言
            else:
                assert_allclose(dy, dy2)  # 对于其他情况，使用默认精度进行断言

    def test_splantider_vs_splint(self):
        # 检查反函数与 FITPACK 的比较
        spl2 = splantider(self.spl)  # 对插值对象 self.spl 进行反函数插值

        # 无外推，splint 假设函数在范围外为零
        xx = np.linspace(0, 1, 20)

        for x1 in xx:
            for x2 in xx:
                y1 = splint(x1, x2, self.spl)  # 计算插值对象 self.spl 在 [x1, x2] 上的积分值，得到 y1
                y2 = splev(x2, spl2) - splev(x1, spl2)  # 计算反函数插值对象 spl2 在 [x1, x2] 上的差值，得到 y2
                assert_allclose(y1, y2)  # 断言 y1 和 y2 的近似相等

    def test_order0_diff(self):
        assert_raises(ValueError, splder, self.spl, 4)  # 断言对于 n=4，导致 ValueError 异常
    def test_kink(self):
        # 检查是否拒绝对带拐点的样条进行求导

        # 在 self.spl 中插入一个拐点，并尝试进行二阶导数操作，应该成功
        spl2 = insert(0.5, self.spl, m=2)
        splder(spl2, 2)  # 应该成功
        # 断言应该抛出 ValueError 异常，因为试图对拥有三个拐点的样条进行三阶导数操作
        assert_raises(ValueError, splder, spl2, 3)

        # 在 self.spl 中插入一个拐点，并尝试进行一阶导数操作，应该成功
        spl2 = insert(0.5, self.spl, m=3)
        splder(spl2, 1)  # 应该成功
        # 断言应该抛出 ValueError 异常，因为试图对拥有四个拐点的样条进行二阶导数操作
        assert_raises(ValueError, splder, spl2, 2)

        # 在 self.spl 中插入一个拐点，断言应该抛出 ValueError 异常，因为试图对拥有五个拐点的样条进行一阶导数操作
        spl2 = insert(0.5, self.spl, m=4)
        assert_raises(ValueError, splder, spl2, 1)

    def test_multidim(self):
        # c 可以具有额外的尾随维度
        for n in range(3):
            t, c, k = self.spl
            # 将 c 扩展为多列，并且通过叠加形成一个三维数组
            c2 = np.c_[c, c, c]
            c2 = np.dstack((c2, c2))

            # 对扩展后的样条进行反向积分操作
            spl2 = splantider((t, c2, k), n)
            # 对反向积分后的样条进行 n 阶导数操作
            spl3 = splder(spl2, n)

            # 断言样条的时间参数 t 与求导后的结果的第一个元素相近
            assert_allclose(t, spl3[0])
            # 断言扩展后的控制点 c2 与求导后的结果的第二个元素相近
            assert_allclose(c2, spl3[1])
            # 断言样条的阶数 k 与求导后的结果的第三个元素相等
            assert_equal(k, spl3[2])
class TestSplint:
    # 定义一个测试类 TestSplint

    def test_len_c(self):
        # 定义测试方法 test_len_c

        n, k = 7, 3
        # 设置变量 n 和 k 分别为 7 和 3

        x = np.arange(n)
        # 创建一个包含 n 个元素的 NumPy 数组 x，其元素为 0 到 n-1 的整数

        y = x**3
        # 创建数组 y，其元素为 x 中每个元素的立方

        t, c, k = splrep(x, y, s=0)
        # 调用 splrep 函数拟合曲线，返回 t, c, k 作为样条曲线的参数

        # note that len(c) == len(t) == 11 (== len(x) + 2*(k-1))
        # 断言 c 的长度等于 t 的长度等于 11，即 len(x) + 2*(k-1)
        assert len(t) == len(c) == n + 2*(k-1)

        # integrate directly: $\int_0^6 x^3 dx = 6^4 / 4$
        # 直接积分计算：$\int_0^6 x^3 dx = 6^4 / 4$
        res = splint(0, 6, (t, c, k))
        # 调用 splint 函数计算从 0 到 6 的积分，使用样条曲线的参数 (t, c, k)
        assert_allclose(res, 6**4 / 4, atol=1e-15)

        # check that the coefficients past len(t) - k - 1 are ignored
        # 检查超过 len(t) - k - 1 的系数是否被忽略
        c0 = c.copy()
        # 复制数组 c 到 c0
        c0[len(t)-k-1:] = np.nan
        # 将 c0 中从 len(t)-k-1 开始的元素设置为 NaN
        res0 = splint(0, 6, (t, c0, k))
        # 使用修改后的 c0 计算从 0 到 6 的积分
        assert_allclose(res0, 6**4 / 4, atol=1e-15)

        # however, all other coefficients *are* used
        # 但是，所有其他系数确实被使用
        c0[6] = np.nan
        # 将 c0 中索引为 6 的元素设置为 NaN
        assert np.isnan(splint(0, 6, (t, c0, k)))

        # check that the coefficient array can have length `len(t) - k - 1`
        # 检查系数数组的长度可以是 `len(t) - k - 1`
        c1 = c[:len(t) - k - 1]
        # 取数组 c 的前 len(t) - k - 1 个元素赋值给 c1
        res1 = splint(0, 6, (t, c1, k))
        # 使用 c1 计算从 0 到 6 的积分
        assert_allclose(res1, 6**4 / 4, atol=1e-15)

        # however shorter c arrays raise. The error from f2py is a
        # `dftipack.error`, which is an Exception but not ValueError etc.
        # 但是，较短的 c 数组会引发错误。来自 f2py 的错误是 `dftipack.error`，它是一个异常，但不是 ValueError 等。
        with assert_raises(Exception, match=r">=n-k-1"):
            # 使用 assert_raises 检测是否引发指定异常
            splint(0, 1, (np.ones(10), np.ones(5), 3))


class TestBisplrep:
    # 定义一个测试类 TestBisplrep

    def test_overflow(self):
        # 定义测试方法 test_overflow

        from numpy.lib.stride_tricks import as_strided
        # 导入 as_strided 函数来创建视图

        if dfitpack_int.itemsize == 8:
            size = 1500000**2
        else:
            size = 400**2
        # 根据 dfitpack_int 的 itemsize 大小设置 size 变量的值

        # Don't allocate a real array, as it's very big, but rely
        # on that it's not referenced
        # 不要分配一个真实的数组，因为它很大，但是依赖于它不被引用

        x = as_strided(np.zeros(()), shape=(size,))
        # 利用 as_strided 创建一个形状为 (size,) 的数组 x

        assert_raises(OverflowError, bisplrep, x, x, x, w=x,
                      xb=0, xe=1, yb=0, ye=1, s=0)
        # 使用 assert_raises 检测是否引发 OverflowError 异常


    def test_regression_1310(self):
        # 定义回归测试方法 test_regression_1310

        # Regression test for gh-1310
        # gh-1310 的回归测试

        with np.load(data_file('bug-1310.npz')) as loaded_data:
            data = loaded_data['data']
        # 使用 np.load 加载数据文件 'bug-1310.npz'，并将数据赋给变量 data

        # Shouldn't crash -- the input data triggers work array sizes
        # that caused previously some data to not be aligned on
        # sizeof(double) boundaries in memory, which made the Fortran
        # code to crash when compiled with -O3
        # 不应崩溃 - 输入数据触发了工作数组大小，之前导致一些数据在内存中未对齐 sizeof(double) 边界，导致 Fortran 代码在使用 -O3 编译时崩溃
        bisplrep(data[:,0], data[:,1], data[:,2], kx=3, ky=3, s=0,
                 full_output=True)

    @pytest.mark.skipif(dfitpack_int != np.int64, reason="needs ilp64 fitpack")
    # 使用 pytest.mark.skipif 装饰器，条件为 dfitpack_int 不等于 np.int64
    def test_ilp64_bisplrep(self):
        # 定义测试方法 test_ilp64_bisplrep

        check_free_memory(28000)  # VM size, doesn't actually use the pages
        # 检查空闲内存是否足够大（28000）

        x = np.linspace(0, 1, 400)
        # 创建一个包含 400 个元素的数组 x，范围从 0 到 1

        y = np.linspace(0, 1, 400)
        # 创建一个包含 400 个元素的数组 y，范围从 0 到 1

        x, y = np.meshgrid(x, y)
        # 创建 x 和 y 的网格

        z = np.zeros_like(x)
        # 创建一个与 x 同样形状的全零数组 z

        tck = bisplrep(x, y, z, kx=3, ky=3, s=0)
        # 使用 bisplrep 函数进行二维样条曲线拟合，返回 tck 作为结果

        assert_allclose(bisplev(0.5, 0.5, tck), 0.0)
        # 使用 bisplev 函数计算在点 (0.5, 0.5) 处的二维样条曲线插值，并断言结果接近于 0.0


def test_dblint():
    # 定义函数 test_dblint

    # Basic test to see it runs and gives the correct result on a trivial
    # problem. Note that `dblint` is not exposed in the interpolate namespace.
    # 基本测试以确保它在简单问题上运行并给出正确结果。注意 `dblint` 不在 interpolate 命名空间中暴露。

    x = np.linspace(0, 1)
    # 创建一个包含 50 个元素的数组 x，范围从 0 到 1

    y = np.linspace(0, 1)
    # 创建一个包含 50 个元素的数组 y，范围从 0 到 1

    xx, yy = np.meshgrid(x, y)
    # 创建 x 和 y 的网格

    rect = RectBivariateSpline(x
    # 将 rect.degrees 中的角度值添加到 tck 列表中
    tck.extend(rect.degrees)

    # 断言双重积分函数 dblint 对于指定区域和 tck 曲线的计算结果几乎等于预期值
    assert_almost_equal(dblint(0, 1, 0, 1, tck), 1)
    assert_almost_equal(dblint(0, 0.5, 0, 1, tck), 0.25)
    assert_almost_equal(dblint(0.5, 1, 0, 1, tck), 0.75)
    assert_almost_equal(dblint(-100, 100, -100, 100, tck), 1)
def test_splev_der_k():
    # regression test for gh-2188: splev(x, tck, der=k) gives garbage or crashes
    # for x outside of knot range

    # test case from gh-2188
    tck = (np.array([0., 0., 2.5, 2.5]),   # 节点向量 t
           np.array([-1.56679978, 2.43995873, 0., 0.]),   # 控制点向量 c
           1)   # 阶数 k
    t, c, k = tck   # 分别解包节点向量、控制点向量和阶数

    x = np.array([-3, 0, 2.5, 3])   # 测试点向量 x

    # an explicit form of the linear spline
    # 使用线性样条的显式形式进行断言验证
    assert_allclose(splev(x, tck), c[0] + (c[1] - c[0]) * x/t[2])
    # 验证一阶导数的值
    assert_allclose(splev(x, tck, 1), (c[1]-c[0]) / t[2])

    # now check a random spline vs splder
    # 现在检查随机样条与 splder 的对比
    np.random.seed(1234)
    x = np.sort(np.random.random(30))
    y = np.random.random(30)
    t, c, k = splrep(x, y)   # 使用 splrep 拟合样条曲线

    x = [t[0] - 1., t[-1] + 1.]
    tck2 = splder((t, c, k), k)   # 计算导数为 k 的样条曲线
    assert_allclose(splev(x, (t, c, k), k), splev(x, tck2))


def test_splprep_segfault():
    # regression test for gh-3847: splprep segfaults if knots are specified
    # for task=-1
    t = np.arange(0, 1.1, 0.1)
    x = np.sin(2*np.pi*t)
    y = np.cos(2*np.pi*t)
    tck, u = splprep([x, y], s=0)   # 使用 splprep 拟合参数化的 B 样条曲线

    np.arange(0, 1.01, 0.01)

    uknots = tck[0]   # 使用之前拟合结果的节点向量
    tck, u = splprep([x, y], task=-1, t=uknots)   # 在指定的节点处执行 task=-1 的拟合，这里会导致崩溃


def test_bisplev_integer_overflow():
    np.random.seed(1)

    x = np.linspace(0, 1, 11)
    y = x
    z = np.random.randn(11, 11).ravel()
    kx = 1
    ky = 1

    nx, tx, ny, ty, c, fp, ier = regrid_smth(
        x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)   # 构造 B 样条曲线的表达式

    xp = np.zeros([2621440])
    yp = np.zeros([2621440])

    assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)


@pytest.mark.xslow
def test_gh_1766():
    # this should fail gracefully instead of segfaulting (int overflow)
    size = 22
    kx, ky = 3, 3
    def f2(x, y):
        return np.sin(x+y)

    x = np.linspace(0, 10, size)
    y = np.linspace(50, 700, size)
    xy = makepairs(x, y)   # 生成 x 和 y 的所有组合对
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)   # 用二维 B 样条曲线拟合

    # the size value here can either segfault
    # or produce a MemoryError on main
    tx_ty_size = 500000
    tck[0] = np.arange(tx_ty_size)   # 在此处使用大数组来测试内存错误
    tck[1] = np.arange(tx_ty_size) * 4
    tt_0 = np.arange(50)
    tt_1 = np.arange(50) * 3
    with pytest.raises(MemoryError):
        bisplev(tt_0, tt_1, tck, 1, 1)


def test_spalde_scalar_input():
    # Ticket #629
    x = np.linspace(0, 10)
    y = x**3
    tck = splrep(x, y, k=3, t=[5])   # 用给定的 t 值拟合样条曲线
    res = spalde(np.float64(1), tck)   # 对标量输入进行求导
    des = np.array([1., 3., 6., 6.])   # 预期的导数值
    assert_almost_equal(res, des)


def test_spalde_nc():
    # regression test for https://github.com/scipy/scipy/issues/19002
    # here len(t) = 29 and len(c) = 25 (== len(t) - k - 1) 
    x = np.asarray([-10., -9., -8., -7., -6., -5., -4., -3., -2.5, -2., -1.5,
                    -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6.],
                    dtype="float")
    # 定义变量 t，表示一个包含浮点数的列表，代表样条插值中的节点（节点的横坐标）
    t = [-10.0, -10.0, -10.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
         -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
         5.0, 6.0, 6.0, 6.0, 6.0]
    
    # 定义变量 c，使用 numpy 转换为数组，表示样条插值的系数
    c = np.asarray([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    # 定义变量 k，表示样条插值的阶数
    k = 3
    
    # 调用 spalde 函数计算样条插值的导数值
    res = spalde(x, (t, c, k))
    
    # 使用列表推导式生成一个 numpy 数组，计算样条插值在给定节点处的函数值
    res_splev = np.asarray([splev(x, (t, c, k), nu) for nu in range(4)])
    
    # 使用 assert_allclose 函数验证两个结果的近似程度，设置容差为 1e-15
    assert_allclose(res, res_splev.T, atol=1e-15)
```