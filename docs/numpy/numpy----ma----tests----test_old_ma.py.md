# `.\numpy\numpy\ma\tests\test_old_ma.py`

```py
# 导入 reduce 函数用于累积计算，pickle 用于对象序列化
from functools import reduce
import pickle

# 导入 pytest 测试框架
import pytest

# 导入 numpy 库及其子模块
import numpy as np
import numpy._core.umath as umath
import numpy._core.fromnumeric as fromnumeric

# 导入 numpy.testing 模块下的各种断言函数
from numpy.testing import (
    assert_, assert_raises, assert_equal,
    )

# 导入 numpy.ma 模块下的函数和类
from numpy.ma import (
    MaskType, MaskedArray, absolute, add, all, allclose, allequal, alltrue,
    arange, arccos, arcsin, arctan, arctan2, array, average, choose,
    concatenate, conjugate, cos, cosh, count, divide, equal, exp, filled,
    getmask, greater, greater_equal, inner, isMaskedArray, less,
    less_equal, log, log10, make_mask, masked, masked_array, masked_equal,
    masked_greater, masked_greater_equal, masked_inside, masked_less,
    masked_less_equal, masked_not_equal, masked_outside,
    masked_print_option, masked_values, masked_where, maximum, minimum,
    multiply, nomask, nonzero, not_equal, ones, outer, product, put, ravel,
    repeat, resize, shape, sin, sinh, sometrue, sort, sqrt, subtract, sum,
    take, tan, tanh, transpose, where, zeros,
    )

# 设置 pi 的值为 numpy 中的圆周率常量
pi = np.pi

# 自定义函数 eq，用于比较两个值或数组是否近似相等
def eq(v, w, msg=''):
    # 使用 numpy 的 allclose 函数判断两个数组或值是否近似相等
    result = allclose(v, w)
    # 如果不相等，则打印消息和不相等的值，并返回 False；否则返回 True
    if not result:
        print(f'Not eq:{msg}\n{v}\n----{w}')
    return result

# 定义一个测试类 TestMa
class TestMa:

    # 初始化方法，在每个测试方法运行之前被调用，设置测试数据
    def setup_method(self):
        # 初始化测试数据
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = array(x, mask=m1)  # 创建一个带有掩码的 MaskedArray 对象 xm
        ym = array(y, mask=m2)  # 创建一个带有掩码的 MaskedArray 对象 ym
        z = np.array([-.5, 0., .5, .8])
        zm = array(z, mask=[0, 1, 0, 0])  # 创建一个带有掩码的 MaskedArray 对象 zm
        xf = np.where(m1, 1e+20, x)  # 根据掩码 m1 条件填充数组 x 中的元素
        s = x.shape  # 获取数组 x 的形状
        xm.set_fill_value(1e+20)  # 设置 MaskedArray xm 的填充值
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf, s)  # 将数据存储在实例属性 d 中

    # 测试方法，用于测试在 1 维情况下的基本数组创建和属性
    def test_testBasic1d(self):
        # 获取初始化设置的测试数据
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        assert_(not isMaskedArray(x))  # 断言 x 不是 MaskedArray 对象
        assert_(isMaskedArray(xm))  # 断言 xm 是 MaskedArray 对象
        assert_equal(shape(xm), s)  # 断言 xm 的形状与 s 相等
        assert_equal(xm.shape, s)  # 断言 xm 的形状与 s 相等
        assert_equal(xm.dtype, x.dtype)  # 断言 xm 的数据类型与 x 相等
        assert_equal(xm.size, reduce(lambda x, y:x * y, s))  # 断言 xm 的大小与 s 元素数的乘积相等
        assert_equal(count(xm), len(m1) - reduce(lambda x, y:x + y, m1))  # 断言 xm 中非掩码元素的数量
        assert_(eq(xm, xf))  # 断言 xm 与填充后的 xf 近似相等
        assert_(eq(filled(xm, 1.e20), xf))  # 断言填充值为 1.e20 后的 xm 与 xf 近似相等
        assert_(eq(x, xm))  # 断言 x 与 xm 近似相等

    @pytest.mark.parametrize("s", [(4, 3), (6, 2)])
    def test_testBasic2d(self, s):
        # Test of basic array creation and properties in 2 dimensions.
        # 解包测试数据元组
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        # 设置数组 x, y, xm, ym, xf 的形状为 s
        x.shape = s
        y.shape = s
        xm.shape = s
        ym.shape = s
        xf.shape = s

        # 断言 x 不是 MaskedArray
        assert_(not isMaskedArray(x))
        # 断言 xm 是 MaskedArray
        assert_(isMaskedArray(xm))
        # 断言 xm 的形状等于 s
        assert_equal(shape(xm), s)
        # 断言 xm 的形状等于 s
        assert_equal(xm.shape, s)
        # 断言 xm 的大小等于 s 元素数的乘积
        assert_equal(xm.size, reduce(lambda x, y: x * y, s))
        # 断言 xm 中填充的值个数等于 m1 中未填充值的个数
        assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
        # 断言 xm 和 xf 相等
        assert_(eq(xm, xf))
        # 断言 xm 和填充值为 1.e20 的 xf 相等
        assert_(eq(filled(xm, 1.e20), xf))
        # 断言 x 和 xm 相等
        assert_(eq(x, xm))

    def test_testArithmetic(self):
        # Test of basic arithmetic.
        # 解包测试数据元组
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        # 创建一个二维数组 a2d
        a2d = array([[1, 2], [0, 4]])
        # 使用 a2d 创建一个带掩码的数组 a2dm
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        # 断言 a2d 和 a2dm 的乘法结果相等
        assert_(eq(a2d * a2d, a2d * a2dm))
        # 断言 a2d 和 a2dm 的加法结果相等
        assert_(eq(a2d + a2d, a2d + a2dm))
        # 断言 a2d 和 a2dm 的减法结果相等
        assert_(eq(a2d - a2d, a2d - a2dm))

        # 遍历不同的形状 s 进行测试
        for s in [(12,), (4, 3), (2, 6)]:
            # 重塑数组 x, y, xm, ym, xf 的形状为 s
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            # 断言 x 的相反数等于 xm 的相反数
            assert_(eq(-x, -xm))
            # 断言 x 和 y 的加法结果等于 xm 和 ym 的加法结果
            assert_(eq(x + y, xm + ym))
            # 断言 x 和 y 的减法结果等于 xm 和 ym 的减法结果
            assert_(eq(x - y, xm - ym))
            # 断言 x 和 y 的乘法结果等于 xm 和 ym 的乘法结果
            assert_(eq(x * y, xm * ym))
            # 使用忽略除零和无效的错误状态，断言 x 和 y 的除法结果等于 xm 和 ym 的除法结果
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(x / y, xm / ym))
            # 断言 a10 和 y 的加法结果等于 a10 和 ym 的加法结果
            assert_(eq(a10 + y, a10 + ym))
            # 断言 a10 和 y 的减法结果等于 a10 和 ym 的减法结果
            assert_(eq(a10 - y, a10 - ym))
            # 断言 a10 和 y 的乘法结果等于 a10 和 ym 的乘法结果
            assert_(eq(a10 * y, a10 * ym))
            # 使用忽略除零和无效的错误状态，断言 a10 和 y 的除法结果等于 a10 和 ym 的除法结果
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(a10 / y, a10 / ym))
            # 断言 x 和 a10 的加法结果等于 xm 和 a10 的加法结果
            assert_(eq(x + a10, xm + a10))
            # 断言 x 和 a10 的减法结果等于 xm 和 a10 的减法结果
            assert_(eq(x - a10, xm - a10))
            # 断言 x 和 a10 的乘法结果等于 xm 和 a10 的乘法结果
            assert_(eq(x * a10, xm * a10))
            # 断言 x 和 a10 的除法结果等于 xm 和 a10 的除法结果
            assert_(eq(x / a10, xm / a10))
            # 断言 x 的平方结果等于 xm 的平方结果
            assert_(eq(x ** 2, xm ** 2))
            # 断言 x 的绝对值的2.5次方等于 xm 的绝对值的2.5次方
            assert_(eq(abs(x) ** 2.5, abs(xm) ** 2.5))
            # 断言 x 的 y 次方结果等于 xm 的 ym 次方结果
            assert_(eq(x ** y, xm ** ym))
            # 断言 np.add(x, y) 的结果等于 add(xm, ym) 的结果
            assert_(eq(np.add(x, y), add(xm, ym)))
            # 断言 np.subtract(x, y) 的结果等于 subtract(xm, ym) 的结果
            assert_(eq(np.subtract(x, y), subtract(xm, ym)))
            # 断言 np.multiply(x, y) 的结果等于 multiply(xm, ym) 的结果
            assert_(eq(np.multiply(x, y), multiply(xm, ym)))
            # 使用忽略除零和无效的错误状态，断言 np.divide(x, y) 的结果等于 divide(xm, ym) 的结果
            with np.errstate(divide='ignore', invalid='ignore'):
                assert_(eq(np.divide(x, y), divide(xm, ym)))

    def test_testMixedArithmetic(self):
        # Test of mixed arithmetic operations between ndarray and MaskedArray.
        # 创建一个包含单个元素 1 的 ndarray 对象 na
        na = np.array([1])
        # 创建一个包含单个元素 1 的 MaskedArray 对象 ma
        ma = array([1])
        # 断言 na 和 ma 的加法结果是 MaskedArray 类型
        assert_(isinstance(na + ma, MaskedArray))
        # 断言 ma 和 na 的加法结果是 MaskedArray 类型
        assert_(isinstance(ma + na, MaskedArray))
    def test_testUfuncs1(self):
        # 测试各种函数，如 sin、cos 等。
        # 从元组 self.d 中获取变量 x, y, a10, m1, m2, xm, ym, z, zm, xf, s
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        
        # 断言 np.cos(x) 等于 cos(xm)
        assert_(eq(np.cos(x), cos(xm)))
        # 断言 np.cosh(x) 等于 cosh(xm)
        assert_(eq(np.cosh(x), cosh(xm)))
        # 断言 np.sin(x) 等于 sin(xm)
        assert_(eq(np.sin(x), sin(xm)))
        # 断言 np.sinh(x) 等于 sinh(xm)
        assert_(eq(np.sinh(x), sinh(xm)))
        # 断言 np.tan(x) 等于 tan(xm)
        assert_(eq(np.tan(x), tan(xm)))
        # 断言 np.tanh(x) 等于 tanh(xm)
        assert_(eq(np.tanh(x), tanh(xm)))
        
        # 忽略除零和无效值的错误状态
        with np.errstate(divide='ignore', invalid='ignore'):
            # 断言 np.sqrt(abs(x)) 等于 sqrt(xm)
            assert_(eq(np.sqrt(abs(x)), sqrt(xm)))
            # 断言 np.log(abs(x)) 等于 log(xm)
            assert_(eq(np.log(abs(x)), log(xm)))
            # 断言 np.log10(abs(x)) 等于 log10(xm)
            assert_(eq(np.log10(abs(x)), log10(xm)))
        
        # 断言 np.exp(x) 等于 exp(xm)
        assert_(eq(np.exp(x), exp(xm)))
        # 断言 np.arcsin(z) 等于 arcsin(zm)
        assert_(eq(np.arcsin(z), arcsin(zm)))
        # 断言 np.arccos(z) 等于 arccos(zm)
        assert_(eq(np.arccos(z), arccos(zm)))
        # 断言 np.arctan(z) 等于 arctan(zm)
        assert_(eq(np.arctan(z), arctan(zm)))
        # 断言 np.arctan2(x, y) 等于 arctan2(xm, ym)
        assert_(eq(np.arctan2(x, y), arctan2(xm, ym)))
        # 断言 np.absolute(x) 等于 absolute(xm)
        assert_(eq(np.absolute(x), absolute(xm)))
        # 断言 np.equal(x, y) 等于 equal(xm, ym)
        assert_(eq(np.equal(x, y), equal(xm, ym)))
        # 断言 np.not_equal(x, y) 等于 not_equal(xm, ym)
        assert_(eq(np.not_equal(x, y), not_equal(xm, ym)))
        # 断言 np.less(x, y) 等于 less(xm, ym)
        assert_(eq(np.less(x, y), less(xm, ym)))
        # 断言 np.greater(x, y) 等于 greater(xm, ym)
        assert_(eq(np.greater(x, y), greater(xm, ym)))
        # 断言 np.less_equal(x, y) 等于 less_equal(xm, ym)
        assert_(eq(np.less_equal(x, y), less_equal(xm, ym)))
        # 断言 np.greater_equal(x, y) 等于 greater_equal(xm, ym)
        assert_(eq(np.greater_equal(x, y), greater_equal(xm, ym)))
        # 断言 np.conjugate(x) 等于 conjugate(xm)
        assert_(eq(np.conjugate(x), conjugate(xm)))
        # 断言 np.concatenate((x, y)) 等于 concatenate((xm, ym))
        assert_(eq(np.concatenate((x, y)), concatenate((xm, ym))))
        # 断言 np.concatenate((x, y)) 等于 concatenate((x, y))
        assert_(eq(np.concatenate((x, y)), concatenate((x, y))))
        # 断言 np.concatenate((x, y)) 等于 concatenate((xm, y))
        assert_(eq(np.concatenate((x, y)), concatenate((xm, y))))
        # 断言 np.concatenate((x, y, x)) 等于 concatenate((x, ym, x))
        assert_(eq(np.concatenate((x, y, x)), concatenate((x, ym, x))))

    def test_xtestCount(self):
        # 测试计数功能
        # 创建一个 array ott，包含数值 [0., 1., 2., 3.]，并设定部分元素被屏蔽（mask=[1, 0, 0, 0]）
        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        
        # 断言 count(ott) 的 dtype 是 np.intp 类型
        assert_(count(ott).dtype.type is np.intp)
        # 断言 count(ott) 的结果为 3
        assert_equal(3, count(ott))
        # 断言 count(1) 的结果为 1
        assert_equal(1, count(1))
        # 断言 array(1, mask=[1]) 的结果等于 0
        assert_(eq(0, array(1, mask=[1])))
        
        # 将 ott 重新调整为形状为 (2, 2)
        ott = ott.reshape((2, 2))
        # 断言 count(ott) 的 dtype 是 np.intp 类型
        assert_(count(ott).dtype.type is np.intp)
        # 断言 count(ott, 0) 返回的类型是 np.ndarray
        assert_(isinstance(count(ott, 0), np.ndarray))
        # 断言 count(ott) 的 dtype 是 np.intp 类型
        assert_(count(ott).dtype.type is np.intp)
        # 断言 count(ott) 的结果为 3
        assert_(eq(3, count(ott)))
        # 断言 count(ott, 0) 的 mask 为 nomask
        assert_(getmask(count(ott, 0)) is nomask)
        # 断言 count(ott, 0) 的结果为 [1, 2]
        assert_(eq([1, 2], count(ott, 0)))

    def test_testMinMax(self):
        # 测试最小值和最大值
        # 从元组 self.d 中获取变量 x, y, a10, m1, m2, xm, ym, z, zm, xf, s
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        
        # 将 x 扁平化为 xr（如果形状不同，max 将无法正常工作）
        xr = np.ravel(x)
        # 将 xm 扁平化为 xmr
        xmr = ravel(xm)
        
        # 断言 max(xr) 等于 maximum.reduce(xmr)
        assert_(eq(max(xr), maximum.reduce(xmr)))
        # 断言 min(xr) 等于 minimum.reduce(xmr))
        assert_(eq(min(xr), minimum.reduce(xmr)))
    def test_testAddSumProd(self):
        # Test add, sum, product.
        # 从测试数据集中获取变量（x, y, a10, m1, m2, xm, ym, z, zm, xf, s）
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        # 断言：numpy 的 add.reduce 和自定义的 add.reduce 的结果相等
        assert_(eq(np.add.reduce(x), add.reduce(x)))
        # 断言：numpy 的 add.accumulate 和自定义的 add.accumulate 的结果相等
        assert_(eq(np.add.accumulate(x), add.accumulate(x)))
        # 断言：对 array(4) 沿着 axis=0 的求和结果等于 4
        assert_(eq(4, sum(array(4), axis=0)))
        # 断言：对 array(4) 沿着 axis=0 的求和结果等于 4
        assert_(eq(4, sum(array(4), axis=0)))
        # 断言：numpy 的 sum 函数和自定义的 sum 函数结果相等，沿着 axis=0
        assert_(eq(np.sum(x, axis=0), sum(x, axis=0)))
        # 断言：numpy 的 sum 函数和自定义的 sum 函数结果相等，对填充了 0 的 xm，沿着 axis=0
        assert_(eq(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0)))
        # 断言：numpy 的 sum 函数和自定义的 sum 函数结果相等，沿着 axis=0
        assert_(eq(np.sum(x, 0), sum(x, 0)))
        # 断言：numpy 的 prod 函数和自定义的 product 函数结果相等，沿着 axis=0
        assert_(eq(np.prod(x, axis=0), product(x, axis=0)))
        # 断言：numpy 的 prod 函数和自定义的 product 函数结果相等，沿着 axis=0
        assert_(eq(np.prod(x, 0), product(x, 0)))
        # 断言：numpy 的 prod 函数和自定义的 product 函数结果相等，对填充了 1 的 xm，沿着 axis=0
        assert_(eq(np.prod(filled(xm, 1), axis=0),
                           product(xm, axis=0)))
        # 如果 s 的长度大于 1，则执行以下断言
        if len(s) > 1:
            # 断言：numpy 的 concatenate 函数和自定义的 concatenate 函数结果相等，沿着 axis=1
            assert_(eq(np.concatenate((x, y), 1),
                               concatenate((xm, ym), 1)))
            # 断言：numpy 的 add.reduce 函数和自定义的 add.reduce 函数结果相等，沿着 axis=1
            assert_(eq(np.add.reduce(x, 1), add.reduce(x, 1)))
            # 断言：numpy 的 sum 函数和自定义的 sum 函数结果相等，沿着 axis=1
            assert_(eq(np.sum(x, 1), sum(x, 1)))
            # 断言：numpy 的 prod 函数和自定义的 product 函数结果相等，沿着 axis=1
            assert_(eq(np.prod(x, 1), product(x, 1)))

    def test_testCI(self):
        # Test of conversions and indexing
        # 创建一个 numpy 数组 x1
        x1 = np.array([1, 2, 4, 3])
        # 使用带有屏蔽值的 array 函数创建数组 x2
        x2 = array(x1, mask=[1, 0, 0, 0])
        # 使用带有屏蔽值的 array 函数创建数组 x3
        x3 = array(x1, mask=[0, 1, 0, 1])
        # 使用 str 函数测试将 x2 转换为字符串，可能会引发异常
        str(x2)  # raises?
        # 使用 repr 函数测试将 x2 转换为字符串，可能会引发异常
        repr(x2)  # raises?
        # 断言：numpy 的 sort 函数和自定义的 sort 函数结果相等，对 x1 排序，使用 fill_value=0
        assert_(eq(np.sort(x1), sort(x2, fill_value=0)))
        # 断言：检查索引后的类型是否相等
        assert_(type(x2[1]) is type(x1[1]))
        # 断言：检查索引后的值是否相等
        assert_(x1[1] == x2[1])
        # 断言：检查 x2 的第一个元素是否为屏蔽值
        assert_(x2[0] is masked)
        # 断言：检查索引后的值是否相等
        assert_(eq(x1[2], x2[2]))
        # 断言：检查切片后的值是否相等
        assert_(eq(x1[2:5], x2[2:5]))
        # 断言：检查全切片后的值是否相等
        assert_(eq(x1[:], x2[:]))
        # 断言：检查 x1 和 x2 的第 1 到第 3 个元素是否相等
        assert_(eq(x1[1:3], x2[1:3]))
        # 修改 x1 和 x2 的第 2 个元素为 9，并断言它们相等
        x1[2] = 9
        x2[2] = 9
        assert_(eq(x1, x2))
        # 将 x1 和 x2 的第 1 到第 3 个元素修改为 99，并断言它们相等
        x1[1:3] = 99
        x2[1:3] = 99
        assert_(eq(x1, x2))
        # 将 x2 的第 2 个元素设为屏蔽值，并断言它们相等
        x2[1] = masked
        assert_(eq(x1, x2))
        # 将 x2 的第 1 到第 3 个元素设为屏蔽值，并断言它们相等
        x2[1:3] = masked
        assert_(eq(x1, x2))
        # 将 x2 的所有元素设为 x1 的值，并将第 2 个元素设为屏蔽值，并断言屏蔽数组的形状为 (0,)
        x2[:] = x1
        x2[1] = masked
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        # 将 x3 的所有元素设为带有屏蔽值的数组，并断言屏蔽数组的形状为 (0, 1, 1, 0)
        x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        # 将 x4 的所有元素设为带有屏蔽值的数组，并断言它们相等
        x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
        assert_(allequal(x4, array([1, 2, 3, 4])))
        # 创建一个 numpy 数组 x1，数据类型为 object
        x1 = array([1, 'hello', 2, 3], object)
        # 创建一个 numpy 数组 x2，数据类型为 object
        x2 = np.array([1, 'hello', 2, 3], object)
        # 获取 x1 和 x2 的第 1 个元素
        s1 = x1[1]
        s2 = x2[1]
        # 断言：获取的 s1 和 s2 的类型为字符串
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        # 断言：获取的 s1 和 s2 的值相等
        assert_equal(s1, s2)
        # 断言：切片 x1 的索引 1 到 1 的形状为 (0,)
        assert_(x1[1:1].shape == (0,))
    def test_testCopySize(self):
        # Tests of some subtle points of copying and sizing.

        # 创建一个包含五个元素的列表
        n = [0, 0, 1, 0, 0]
        # 调用 make_mask 函数创建掩码数组 m
        m = make_mask(n)
        # 将 m 作为参数再次调用 make_mask 函数，返回值赋给 m2
        m2 = make_mask(m)
        # 断言 m 和 m2 引用的是同一个对象
        assert_(m is m2)
        # 使用 copy=True 参数再次调用 make_mask 函数，返回值赋给 m3
        m3 = make_mask(m, copy=True)
        # 断言 m 和 m3 引用的不是同一个对象
        assert_(m is not m3)

        # 创建一个包含 0 到 4 的一维数组 x1
        x1 = np.arange(5)
        # 使用掩码 m 创建一个 MaskedArray 对象 y1
        y1 = array(x1, mask=m)
        # 断言 y1 的数据不是 x1
        assert_(y1._data is not x1)
        # 断言 y1 的数据和 x1 的数据相等
        assert_(allequal(x1, y1._data))
        # 断言 y1 的掩码属性是 m
        assert_(y1._mask is m)

        # 使用 copy=0 参数创建一个新的 MaskedArray 对象 y1a
        y1a = array(y1, copy=0)
        # 断言 y1a 的掩码的数组接口和 y1 的掩码的数组接口相等
        assert_(y1a._mask.__array_interface__ ==
                y1._mask.__array_interface__)

        # 使用 copy=0 和 m3 参数创建一个新的 MaskedArray 对象 y2
        y2 = array(x1, mask=m3, copy=0)
        # 断言 y2 的掩码是 m3
        assert_(y2._mask is m3)
        # 断言 y2 的第二个元素是 masked
        assert_(y2[2] is masked)
        # 修改 y2 的第二个元素为 9
        y2[2] = 9
        # 断言 y2 的第二个元素不是 masked
        assert_(y2[2] is not masked)
        # 断言 y2 的掩码是 m3
        assert_(y2._mask is m3)
        # 断言 y2 的所有掩码元素都是 0
        assert_(allequal(y2.mask, 0))

        # 使用 copy=1 和 m 参数创建一个新的 MaskedArray 对象 y2a
        y2a = array(x1, mask=m, copy=1)
        # 断言 y2a 的掩码不是 m
        assert_(y2a._mask is not m)
        # 断言 y2a 的第二个元素是 masked
        assert_(y2a[2] is masked)
        # 修改 y2a 的第二个元素为 9
        y2a[2] = 9
        # 断言 y2a 的第二个元素不是 masked
        assert_(y2a[2] is not masked)
        # 断言 y2a 的掩码不是 m
        assert_(y2a._mask is not m)
        # 断言 y2a 的所有掩码元素都是 0
        assert_(allequal(y2a.mask, 0))

        # 创建一个包含浮点数的一维数组 y3，数据与 x1 的数据相同，掩码为 m
        y3 = array(x1 * 1.0, mask=m)
        # 断言 filled(y3) 的数据类型与 (x1 * 1.0) 的数据类型相同
        assert_(filled(y3).dtype is (x1 * 1.0).dtype)

        # 创建一个一维数组 x4 包含四个元素
        x4 = arange(4)
        # 将 x4 的第三个元素设为 masked
        x4[2] = masked
        # 调整 x4 的大小为 (8,)，赋给 y4
        y4 = resize(x4, (8,))
        # 断言 concatenate([x4, x4]) 等于 y4
        assert_(eq(concatenate([x4, x4]), y4))
        # 断言 y4 的掩码为 [0, 0, 1, 0, 0, 0, 1, 0]
        assert_(eq(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0]))
        # 将 x4 沿轴 0 重复 [2, 2, 2, 2]，赋给 y5
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        # 断言 y5 等于 [0, 0, 1, 1, 2, 2, 3, 3]
        assert_(eq(y5, [0, 0, 1, 1, 2, 2, 3, 3]))
        # 将 x4 沿轴 0 重复 2 次，赋给 y6
        y6 = repeat(x4, 2, axis=0)
        # 断言 y5 等于 y6
        assert_(eq(y5, y6))

    def test_testPut(self):
        # Test of put

        # 创建一个包含五个元素的一维数组 d
        d = arange(5)
        # 创建一个包含 [0, 0, 0, 1, 1] 的列表，赋给 n
        n = [0, 0, 0, 1, 1]
        # 调用 make_mask 函数创建掩码数组 m
        m = make_mask(n)
        # 复制 m，并赋给 m2
        m2 = m.copy()
        # 使用 m 创建一个 MaskedArray 对象 x
        x = array(d, mask=m)
        # 断言 x 的第四个元素是 masked
        assert_(x[3] is masked)
        # 断言 x 的第五个元素是 masked
        assert_(x[4] is masked)
        # 将 x 的第二个和第五个元素分别设为 10 和 40
        x[[1, 4]] = [10, 40]
        # 断言 x 的掩码是 m
        assert_(x._mask is m)
        # 断言 x 的第四个元素是 masked
        assert_(x[3] is masked)
        # 断言 x 的第五个元素不是 masked
        assert_(x[4] is not masked)
        # 断言 x 的元素与 [0, 10, 2, -1, 40] 相等
        assert_(eq(x, [0, 10, 2, -1, 40]))

        # 使用 copy=True 和 m2 创建一个新的 MaskedArray 对象 x
        x = array(d, mask=m2, copy=True)
        # 使用 put 方法将 [-1, 100, 200] 分别放置在 x 的第一个、第二个和第三个位置
        x.put([0, 1, 2], [-1, 100, 200])
        # 断言 x 的掩码不是 m2
        assert_(x._mask is not m2)
        # 断言 x 的第四个元素是 masked
        assert_(x[3] is masked)
        # 断言 x 的第五个元素是 masked
        assert_(x[4] is masked)
        # 断言 x 的元素与 [-1, 100, 200, 0, 0] 相等
        assert_(eq(x, [-1, 100, 200, 0, 0]))
    def test_testPut2(self):
        # Test of put
        # 创建一个包含0到4的数组
        d = arange(5)
        # 创建一个带有掩码的数组，掩码表明所有元素都未被屏蔽
        x = array(d, mask=[0, 0, 0, 0, 0])
        # 创建另一个带有掩码的数组，其中第一个元素被屏蔽
        z = array([10, 40], mask=[1, 0])
        # 断言：x的第2个元素未被屏蔽
        assert_(x[2] is not masked)
        # 断言：x的第3个元素未被屏蔽
        assert_(x[3] is not masked)
        # 将z的值放入x的第2到第4个位置
        x[2:4] = z
        # 断言：x的第2个元素已被屏蔽
        assert_(x[2] is masked)
        # 断言：x的第3个元素未被屏蔽
        assert_(x[3] is not masked)
        # 断言：x是否等于[0, 1, 10, 40, 4]
        assert_(eq(x, [0, 1, 10, 40, 4]))

        # 重新初始化变量d和x，重复上述步骤，但不修改原始数组x
        d = arange(5)
        x = array(d, mask=[0, 0, 0, 0, 0])
        # 从x中获取第2到第4个元素形成一个新的数组y
        y = x[2:4]
        # 创建另一个带有掩码的数组z
        z = array([10, 40], mask=[1, 0])
        # 断言：x的第2个元素未被屏蔽
        assert_(x[2] is not masked)
        # 断言：x的第3个元素未被屏蔽
        assert_(x[3] is not masked)
        # 将z的值放入y中
        y[:] = z
        # 断言：y的第一个元素已被屏蔽
        assert_(y[0] is masked)
        # 断言：y的第二个元素未被屏蔽
        assert_(y[1] is not masked)
        # 断言：y是否等于[10, 40]
        assert_(eq(y, [10, 40]))
        # 断言：x的第2个元素已被屏蔽
        assert_(x[2] is masked)
        # 断言：x的第3个元素未被屏蔽
        assert_(x[3] is not masked)
        # 断言：x是否等于[0, 1, 10, 40, 4]

        assert_(eq(x, [0, 1, 10, 40, 4]))

    def test_testMaPut(self):
        # Test of put for masked arrays
        # 从self.d中解包变量(x, y, a10, m1, m2, xm, ym, z, zm, xf, s)
        (x, y, a10, m1, m2, xm, ym, z, zm, xf, s) = self.d
        # 创建一个掩码列表m
        m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        # 找到掩码列表m中非零元素的索引
        i = np.nonzero(m)[0]
        # 使用put函数将zm放入ym的指定索引位置
        put(ym, i, zm)
        # 断言：从ym中取出指定索引位置的值与zm相等
        assert_(all(take(ym, i, axis=0) == zm))

    def test_testMinMax2(self):
        # Test of minimum, maximum.
        # 断言：对于输入的两个数组，返回元素级的最小值
        assert_(eq(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3]))
        # 断言：对于输入的两个数组，返回元素级的最大值
        assert_(eq(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9]))
        # 创建数组x和y，分别为0到4和-2到2的范围
        x = arange(5)
        y = arange(5) - 2
        # 将x的第3个元素屏蔽
        x[3] = masked
        # 将y的第一个元素屏蔽
        y[0] = masked
        # 断言：对x和y中每对元素，返回小于的元素
        assert_(eq(minimum(x, y), where(less(x, y), x, y)))
        # 断言：对x和y中每对元素，返回大于的元素
        assert_(eq(maximum(x, y), where(greater(x, y), x, y)))
        # 断言：返回数组x中的最小值
        assert_(minimum.reduce(x) == 0)
        # 断言：返回数组x中的最大值
        assert_(maximum.reduce(x) == 4)

    def test_testTakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        # 创建一个包含0到23的数组x和y
        x = arange(24)
        y = np.arange(24)
        # 将x的第5到第6个元素屏蔽，并将其重塑为2x3x4的数组
        x[5:6] = masked
        x = x.reshape(2, 3, 4)
        # 将y重塑为与x相同的2x3x4的数组
        y = y.reshape(2, 3, 4)
        # 断言：通过维度重新排列，确保x和y的内部结构相同
        assert_(eq(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1))))
        # 断言：使用指定索引从数组中获取元素，并确保x和y的输出相同
        assert_(eq(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1)))
        # 断言：返回填充为0的x和y的内积
        assert_(eq(np.inner(filled(x, 0), filled(y, 0)),
                   inner(x, y)))
        # 断言：返回填充为0的x和y的外积
        assert_(eq(np.outer(filled(x, 0), filled(y, 0)),
                   outer(x, y)))
        # 创建一个对象数组y，并将其第2个元素屏蔽
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked
        # 使用指定索引从y中获取元素形成一个新数组t
        t = take(y, [0, 3, 4])
        # 断言：确保数组t的第一个元素为'abc'
        assert_(t[0] == 'abc')
        # 断言：确保数组t的第二个元素为2
        assert_(t[1] == 2)
        # 断言：确保数组t的第三个元素为3
        assert_(t[2] == 3)
    def test_testInplace(self):
        # 测试原地操作和丰富比较
        y = arange(10)  # 创建一个包含0到9的数组y

        x = arange(10)  # 创建一个包含0到9的数组x
        xm = arange(10)  # 创建一个包含0到9的数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x += 1  # 将数组x中的每个元素都加1
        assert_(eq(x, y + 1))  # 断言x与y加1后的结果相等
        xm += 1  # 将数组xm中的每个元素都加1
        assert_(eq(x, y + 1))  # 断言xm与y加1后的结果相等

        x = arange(10)  # 创建一个包含0到9的数组x
        xm = arange(10)  # 创建一个包含0到9的数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x -= 1  # 将数组x中的每个元素都减1
        assert_(eq(x, y - 1))  # 断言x与y减1后的结果相等
        xm -= 1  # 将数组xm中的每个元素都减1
        assert_(eq(xm, y - 1))  # 断言xm与y减1后的结果相等

        x = arange(10) * 1.0  # 创建一个包含0.0到9.0的浮点数组x
        xm = arange(10) * 1.0  # 创建一个包含0.0到9.0的浮点数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x *= 2.0  # 将数组x中的每个元素都乘以2.0
        assert_(eq(x, y * 2))  # 断言x与y乘以2后的结果相等
        xm *= 2.0  # 将数组xm中的每个元素都乘以2.0
        assert_(eq(xm, y * 2))  # 断言xm与y乘以2后的结果相等

        x = arange(10) * 2  # 创建一个包含0到18（步长为2）的数组x
        xm = arange(10)  # 创建一个包含0到9的数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x //= 2  # 将数组x中的每个元素都整除以2
        assert_(eq(x, y))  # 断言x与y整除以2后的结果相等
        xm //= 2  # 将数组xm中的每个元素都整除以2
        assert_(eq(x, y))  # 断言xm与y整除以2后的结果相等

        x = arange(10) * 1.0  # 创建一个包含0.0到9.0的浮点数组x
        xm = arange(10) * 1.0  # 创建一个包含0.0到9.0的浮点数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x /= 2.0  # 将数组x中的每个元素都除以2.0
        assert_(eq(x, y / 2.0))  # 断言x与y除以2.0后的结果相等
        xm /= arange(10)  # 将数组xm中的每个元素分别除以相应的索引值
        assert_(eq(xm, ones((10,))))  # 断言xm与一个全1数组相等

        x = arange(10).astype(np.float32)  # 创建一个包含0到9的浮点数组x
        xm = arange(10)  # 创建一个包含0到9的数组xm
        xm[2] = masked  # 将xm的索引为2的位置设为masked（掩码值）
        x += 1.  # 将数组x中的每个元素都加1.0
        assert_(eq(x, y + 1.))  # 断言x与y加1.0后的结果相等
    def test_testAverage2(self):
        # 定义测试函数 test_testAverage2
        # 测试 average 函数的不同用例

        # 定义权重和数组
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = arange(6)

        # 断言：验证在指定轴上求平均值，预期结果为 2.5
        assert_(allclose(average(x, axis=0), 2.5))
        
        # 断言：验证在指定轴上使用权重求平均值，预期结果为 2.5
        assert_(allclose(average(x, axis=0, weights=w1), 2.5))
        
        # 定义二维数组 y
        y = array([arange(6), 2.0 * arange(6)])

        # 断言：验证在全局求平均值，预期结果为 (0+1+2+3+4+5) * 3 / 12 = 7.5
        assert_(allclose(average(y, None), np.add.reduce(np.arange(6)) * 3. / 12.))
        
        # 断言：验证在指定轴上求平均值，预期结果为 [0*3/2, 1*3/2, ..., 5*3/2]
        assert_(allclose(average(y, axis=0), np.arange(6) * 3. / 2.))
        
        # 断言：验证在指定轴上求平均值，预期结果为 [平均(x, axis=0), 平均(x, axis=0)*2.0]
        assert_(allclose(average(y, axis=1), [average(x, axis=0), average(x, axis=0)*2.0]))
        
        # 断言：验证在全局使用权重求平均值，预期结果为 20 / 6.0
        assert_(allclose(average(y, None, weights=w2), 20. / 6.))
        
        # 断言：验证在指定轴上使用权重求平均值，预期结果为 [0, 1, 2, 3, 4, 10]
        assert_(allclose(average(y, axis=0, weights=w2), [0., 1., 2., 3., 4., 10.]))
        
        # 以下几个断言均为重复验证，省略重复注释
        m1 = zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        assert_(allclose(average(masked_array(x, m1), axis=0), 2.5))
        assert_(allclose(average(masked_array(x, m2), axis=0), 2.5))
        assert_(average(masked_array(x, m4), axis=0) is masked)
        assert_equal(average(masked_array(x, m5), axis=0), 0.0)
        assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
        
        z = masked_array(y, m3)
        assert_(allclose(average(z, None), 20. / 6.))
        assert_(allclose(average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5]))
        assert_(allclose(average(z, axis=1), [2.5, 5.0]))
        assert_(allclose(average(z, axis=0, weights=w2), [0., 1., 99., 99., 4.0, 10.0]))

        a = arange(6)
        b = arange(6) * 3
        r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
        assert_equal(shape(r1), shape(w1))
        assert_equal(r1.shape, w1.shape)
        r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
        assert_(shape(w2) == shape(r2))

        a2d = array([[1, 2], [0, 4]], float)
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        a2da = average(a2d, axis=0)
        assert_(eq(a2da, [0.5, 3.0]))
        a2dma = average(a2dm, axis=0)
        assert_(eq(a2dma, [1.0, 3.0]))
        a2dma = average(a2dm, axis=None)
        assert_(eq(a2dma, 7. / 3.))
        a2dma = average(a2dm, axis=1)
        assert_(eq(a2dma, [1.5, 4.0]))
    # 测试将数值转换为数组，并验证转换后的类型是否正确
    def test_testToPython(self):
        # 验证将整数转换为数组后，类型为整数
        assert_equal(1, int(array(1)))
        # 验证将整数转换为浮点数数组后，类型为浮点数
        assert_equal(1.0, float(array(1)))
        # 验证将多维整数数组转换为整数后，类型为整数
        assert_equal(1, int(array([[[1]]])))
        # 验证将二维整数数组转换为浮点数后，类型为浮点数
        assert_equal(1.0, float(array([[1]]))))
        # 验证传递非法参数时是否引发TypeError异常
        assert_raises(TypeError, float, array([1, 1]))
        # 验证传递非法值时是否引发ValueError异常
        assert_raises(ValueError, bool, array([0, 1]))
        # 验证传递带掩码的数组时是否引发ValueError异常
        assert_raises(ValueError, bool, array([0, 0], mask=[0, 1]))

    # 测试标量运算
    def test_testScalarArithmetic(self):
        xm = array(0, mask=1)
        #TODO FIXME: 在r8247中找出以下内容为何引发警告
        # 使用特定的错误状态处理，忽略除以零的警告
        with np.errstate(divide='ignore'):
            assert_((1 / array(0)).mask)
        # 验证在标量加法中是否生成了正确的掩码
        assert_((1 + xm).mask)
        # 验证在标量取反操作中是否生成了正确的掩码
        assert_((-xm).mask)
        # 再次验证在标量取反操作中是否生成了正确的掩码
        assert_((-xm).mask)
        # 验证在取两个掩码的最大值时是否生成了正确的掩码
        assert_(maximum(xm, xm).mask)
        # 验证在取两个掩码的最小值时是否生成了正确的掩码
        assert_(minimum(xm, xm).mask)
        # 验证填充后的数据类型是否与原始数据类型相同
        assert_(xm.filled().dtype is xm._data.dtype)
        # 验证未掩码数组与其数据的相等性
        x = array(0, mask=0)
        assert_(x.filled() == x._data)
        # 验证掩码数组转换为字符串是否与预期的打印选项相等
        assert_equal(str(xm), str(masked_print_option))

    # 测试数组方法
    def test_testArrayMethods(self):
        a = array([1, 3, 2])
        # 验证任意值是否与原始数据的任意值相等
        assert_(eq(a.any(), a._data.any()))
        # 验证所有值是否与原始数据的所有值相等
        assert_(eq(a.all(), a._data.all()))
        # 验证最大值的索引是否与原始数据的最大值索引相等
        assert_(eq(a.argmax(), a._data.argmax()))
        # 验证最小值的索引是否与原始数据的最小值索引相等
        assert_(eq(a.argmin(), a._data.argmin()))
        # 验证选择操作的结果是否与原始数据的选择操作结果相等
        assert_(eq(a.choose(0, 1, 2, 3, 4),
                           a._data.choose(0, 1, 2, 3, 4)))
        # 验证压缩操作的结果是否与原始数据的压缩操作结果相等
        assert_(eq(a.compress([1, 0, 1]), a._data.compress([1, 0, 1])))
        # 验证共轭操作的结果是否与原始数据的共轭操作结果相等
        assert_(eq(a.conj(), a._data.conj()))
        # 验证共轭操作的结果是否与原始数据的共轭操作结果相等
        assert_(eq(a.conjugate(), a._data.conjugate()))
        # 验证对角线元素的结果是否与原始数据的对角线元素结果相等
        m = array([[1, 2], [3, 4]])
        assert_(eq(m.diagonal(), m._data.diagonal()))
        # 验证数组求和的结果是否与原始数据的求和结果相等
        assert_(eq(a.sum(), a._data.sum()))
        # 验证取数组元素的结果是否与原始数据的取元素结果相等
        assert_(eq(a.take([1, 2]), a._data.take([1, 2])))
        # 验证数组转置的结果是否与原始数据的转置结果相等
        assert_(eq(m.transpose(), m._data.transpose()))

    # 测试数组属性
    def test_testArrayAttributes(self):
        a = array([1, 3, 2])
        # 验证数组的维度是否为1
        assert_equal(a.ndim, 1)

    # 测试API
    def test_testAPI(self):
        # 验证不在MaskedArray中且不以下划线开头的所有方法是否都在ndarray中
        assert_(not [m for m in dir(np.ndarray)
                     if m not in dir(MaskedArray) and
                     not m.startswith('_')])

    # 测试单个元素下标
    def test_testSingleElementSubscript(self):
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        # 验证单个元素下标的形状是否为空元组
        assert_equal(a[0].shape, ())
        # 验证带掩码的数组单个元素下标的形状是否为空元组
        assert_equal(b[0].shape, ())
        # 验证带掩码的数组第二个元素下标的形状是否为单元素元组
        assert_equal(b[1].shape, ())

    # 测试条件赋值
    def test_assignment_by_condition(self):
        # 测试gh-18951
        a = array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        # 创建条件，选择满足条件的元素进行赋值
        c = a >= 3
        a[c] = 5
        # 验证条件赋值后的特定索引是否被遮蔽
        assert_(a[2] is masked)

    # 测试条件赋值2
    def test_assignment_by_condition_2(self):
        # gh-19721
        a = masked_array([0, 1], mask=[False, False])
        b = masked_array([0, 1], mask=[True, True])
        # 创建条件，选择满足条件的元素进行赋值
        mask = a < 1
        b[mask] = a[mask]
        expected_mask = [False, True]
        # 验证条件赋值后的掩码是否与预期相符
        assert_equal(b.mask, expected_mask)
class TestUfuncs:
    def setup_method(self):
        # 设置测试方法的初始化数据
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)

    def test_testUfuncRegression(self):
        # 定义无效忽略函数列表
        f_invalid_ignore = [
            'sqrt', 'arctanh', 'arcsin', 'arccos',
            'arccosh', 'arctanh', 'log', 'log10', 'divide',
            'true_divide', 'floor_divide', 'remainder', 'fmod']
        # 遍历每个函数并进行测试
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor']:
            try:
                # 尝试从umath模块获取函数
                uf = getattr(umath, f)
            except AttributeError:
                # 如果umath模块中不存在，则从fromnumeric模块获取
                uf = getattr(fromnumeric, f)
            # 从np.ma模块获取函数
            mf = getattr(np.ma, f)
            # 准备参数，使用uf函数的输入数量切片数据d
            args = self.d[:uf.nin]
            # 设置numpy错误状态
            with np.errstate():
                if f in f_invalid_ignore:
                    np.seterr(invalid='ignore')
                if f in ['arctanh', 'log', 'log10']:
                    np.seterr(divide='ignore')
                # 调用uf函数和mf函数，得到结果
                ur = uf(*args)
                mr = mf(*args)
            # 断言ur的填充0后与mr的填充0后相等，使用函数名作为描述信息
            assert_(eq(ur.filled(0), mr.filled(0), f))
            # 断言ur的掩码与mr的掩码相等
            assert_(eqmask(ur.mask, mr.mask))

    def test_reduce(self):
        # 获取数组self.d的第一个元素
        a = self.d[0]
        # 断言a在轴0上不全为真
        assert_(not alltrue(a, axis=0))
        # 断言a在轴0上至少有一个为真
        assert_(sometrue(a, axis=0))
        # 断言a的前三个元素在轴0上的和为0
        assert_equal(sum(a[:3], axis=0), 0)
        # 断言a在轴0上的乘积为0
        assert_equal(product(a, axis=0), 0)

    def test_minmax(self):
        # 创建一个3x4的数组
        a = arange(1, 13).reshape(3, 4)
        # 使用条件掩码创建一个数组amask
        amask = masked_where(a < 5, a)
        # 断言amask的最大值与a的最大值相等
        assert_equal(amask.max(), a.max())
        # 断言amask的最小值为5
        assert_equal(amask.min(), 5)
        # 断言amask在轴0上的最大值与a在轴0上的最大值全部相等
        assert_((amask.max(0) == a.max(0)).all())
        # 断言amask在轴0上的最小值与[5, 6, 7, 8]全部相等
        assert_((amask.min(0) == [5, 6, 7, 8]).all())
        # 断言amask在轴1上的第一个元素的最大值具有掩码
        assert_(amask.max(1)[0].mask)
        # 断言amask在轴1上的第一个元素的最小值具有掩码
        assert_(amask.min(1)[0].mask)

    def test_nonzero(self):
        # 遍历不同类型的数据类型
        for t in "?bhilqpBHILQPfdgFDGO":
            # 创建一个数组x，包含数据和掩码
            x = array([1, 0, 2, 0], mask=[0, 0, 1, 1])
            # 断言x的非零元素索引为[0]
            assert_(eq(nonzero(x), [0]))
    # 设置测试方法的初始化，准备测试数据
    def setup_method(self):
        # 创建包含浮点数的 NumPy 数组 x
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        # 将 x 重塑为 6x6 的数组 X
        X = x.reshape(6, 6)
        # 将 x 重塑为 3x2x2x3 的数组 XX
        XX = x.reshape(3, 2, 2, 3)

        # 创建包含 0 和 1 的 NumPy 数组 m
        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        # 使用 x 和 m 创建掩码数组 mx
        mx = array(data=x, mask=m)
        # 使用 X 和 m 重塑的形状创建掩码数组 mX
        mX = array(data=X, mask=m.reshape(X.shape))
        # 使用 XX 和 m 重塑的形状创建掩码数组 mXX
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        # 将所有数据存储在 self.d 中
        self.d = (x, X, XX, m, mx, mX, mXX)

    # 测试计算对角线元素和的方法
    def test_trace(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 计算 mX 的对角线元素
        mXdiag = mX.diagonal()
        # 断言 mX 的迹等于其对角线元素去除掩码后的和
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        # 断言 mX 的迹等于 X 的迹减去按位乘积结果的和
        assert_(eq(mX.trace(),
                           X.trace() - sum(mXdiag.mask * X.diagonal(),
                                           axis=0)))

    # 测试剪裁方法
    def test_clip(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 对 mx 应用剪裁操作，剪裁范围为 [2, 8]
        clipped = mx.clip(2, 8)
        # 断言剪裁后的掩码与 mx 的掩码相等
        assert_(eq(clipped.mask, mx.mask))
        # 断言剪裁后的数据与 x 的剪裁结果相等
        assert_(eq(clipped._data, x.clip(2, 8)))
        # 断言剪裁后的数据与 mx 的剪裁结果相等
        assert_(eq(clipped._data, mx._data.clip(2, 8)))

    # 测试峰值到峰值方法
    def test_ptp(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 计算 mx 的峰值到峰值
        assert_equal(mx.ptp(), np.ptp(mx.compressed()))
        # 创建一个全零的行和列数组，用于存储结果
        rows = np.zeros(n, np.float64)
        cols = np.zeros(m, np.float64)
        # 计算 mX 每列的峰值到峰值，并存储在 cols 数组中
        for k in range(m):
            cols[k] = np.ptp(mX[:, k].compressed())
        # 计算 mX 每行的峰值到峰值，并存储在 rows 数组中
        for k in range(n):
            rows[k] = np.ptp(mX[k].compressed())
        # 断言 mX 沿指定轴的峰值到峰值等于预期结果
        assert_(eq(mX.ptp(0), cols))
        assert_(eq(mX.ptp(1), rows))

    # 测试交换轴方法
    def test_swapaxes(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 对 mX 执行轴交换操作
        mXswapped = mX.swapaxes(0, 1)
        # 断言交换后的最后一个轴与原始数据的最后一个轴相等
        assert_(eq(mXswapped[-1], mX[:, -1]))
        # 对 mXX 执行轴交换操作，并断言结果形状符合预期
        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    # 测试累积乘积方法
    def test_cumprod(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 对 mX 执行沿第一个轴的累积乘积操作
        mXcp = mX.cumprod(0)
        # 断言累积乘积结果的数据部分与填充后的 mX 结果相等
        assert_(eq(mXcp._data, mX.filled(1).cumprod(0)))
        # 对 mX 执行沿第二个轴的累积乘积操作
        mXcp = mX.cumprod(1)
        # 断言累积乘积结果的数据部分与填充后的 mX 结果相等
        assert_(eq(mXcp._data, mX.filled(1).cumprod(1)))

    # 测试累积求和方法
    def test_cumsum(self):
        # 从 self.d 中解包所需的数据
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 对 mX 执行沿第一个轴的累积求和操作
        mXcp = mX.cumsum(0)
        # 断言累积求和结果的数据部分与填充后的 mX 结果相等
        assert_(eq(mXcp._data, mX.filled(0).cumsum(0)))
        # 对 mX 执行沿第二个轴的累积求和操作
        mXcp = mX.cumsum(1)
        # 断言累积求和结果的数据部分与填充后的 mX 结果相等
        assert_(eq(mXcp._data, mX.filled(0).cumsum(1)))
    # 定义一个测试函数，用于验证变量标准差的计算是否正确
    def test_varstd(self):
        # 从数据集 self.d 中解包变量
        (x, X, XX, m, mx, mX, mXX,) = self.d
        # 断言检查：比较 mX 沿着所有轴的方差和经过压缩后的 mX 的方差是否相等
        assert_(eq(mX.var(axis=None), mX.compressed().var()))
        # 断言检查：比较 mX 沿着所有轴的标准差和经过压缩后的 mX 的标准差是否相等
        assert_(eq(mX.std(axis=None), mX.compressed().std()))
        # 断言检查：比较 mXX 沿着第 3 个轴计算方差后的形状和 XX 沿着第 3 个轴计算方差后的形状是否相等
        assert_(eq(mXX.var(axis=3).shape, XX.var(axis=3).shape))
        # 断言检查：比较 mX 沿着所有轴计算方差的形状和 X 沿着所有轴计算方差的形状是否相等
        assert_(eq(mX.var().shape, X.var().shape))
        # 解包 mX 沿着第 0 轴和第 1 轴的方差
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        # 循环检查每个索引 k
        for k in range(6):
            # 断言检查：比较 mXvar1[k] 和经过压缩后的 mX[k] 的方差是否相等
            assert_(eq(mXvar1[k], mX[k].compressed().var()))
            # 断言检查：比较 mXvar0[k] 和经过压缩后的 mX[:, k] 的方差是否相等
            assert_(eq(mXvar0[k], mX[:, k].compressed().var()))
            # 断言检查：比较 mXvar0[k] 的平方根和经过压缩后的 mX[:, k] 的标准差是否相等
            assert_(eq(np.sqrt(mXvar0[k]),
                               mX[:, k].compressed().std()))
# 定义一个函数，用于比较两个掩码（mask）是否相等
def eqmask(m1, m2):
    # 如果 m1 是 nomask，则判断 m2 是否也是 nomask，若是则返回 True
    if m1 is nomask:
        return m2 is nomask
    # 如果 m2 是 nomask，则判断 m1 是否也是 nomask，若是则返回 True
    if m2 is nomask:
        return m1 is nomask
    # 如果 m1 和 m2 都不是 nomask，则比较它们的所有元素是否相等，返回比较结果
    return (m1 == m2).all()
```