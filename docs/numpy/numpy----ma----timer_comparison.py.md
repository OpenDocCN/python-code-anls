# `.\numpy\numpy\ma\timer_comparison.py`

```py
# 导入用于计时的模块
import timeit
# 导入 reduce 函数，用于对序列进行累积计算
from functools import reduce

# 导入 NumPy 库，并指定别名 np
import numpy as np
# 导入 NumPy 的核心模块 fromnumeric
import numpy._core.fromnumeric as fromnumeric

# 导入 NumPy 的测试模块中的错误消息构建函数
from numpy.testing import build_err_msg

# 设置常数 pi 为 NumPy 中定义的圆周率值
pi = np.pi

# 定义 ModuleTester 类，用于测试指定模块中的函数
class ModuleTester:
    def __init__(self, module):
        # 初始化类的实例，传入要测试的模块
        self.module = module
        # 以下是从模块中获取和设置的一系列函数和属性，用于后续的测试操作
        self.allequal = module.allequal
        self.arange = module.arange
        self.array = module.array
        self.concatenate = module.concatenate
        self.count = module.count
        self.equal = module.equal
        self.filled = module.filled
        self.getmask = module.getmask
        self.getmaskarray = module.getmaskarray
        self.id = id
        self.inner = module.inner
        self.make_mask = module.make_mask
        self.masked = module.masked
        self.masked_array = module.masked_array
        self.masked_values = module.masked_values
        self.mask_or = module.mask_or
        self.nomask = module.nomask
        self.ones = module.ones
        self.outer = module.outer
        self.repeat = module.repeat
        self.resize = module.resize
        self.sort = module.sort
        self.take = module.take
        self.transpose = module.transpose
        self.zeros = module.zeros
        self.MaskType = module.MaskType
        # 尝试获取 umath 属性，如果不存在则使用 module.core.umath
        try:
            self.umath = module.umath
        except AttributeError:
            self.umath = module.core.umath
        # 初始化测试用例名称列表
        self.testnames = []
    def assert_array_compare(self, comparison, x, y, err_msg='', header='',
                             fill_value=True):
        """
        Assert that a comparison of two masked arrays is satisfied elementwise.

        """
        # 将输入的两个数组 x 和 y 填充为非掩码状态的数组
        xf = self.filled(x)
        yf = self.filled(y)
        # 使用 mask_or 方法生成 x 和 y 的掩码，并合并
        m = self.mask_or(self.getmask(x), self.getmask(y))

        # 使用填充后的数组 xf 和 yf 创建新的 MaskedArray 对象，再次填充，使用 fill_value 参数
        x = self.filled(self.masked_array(xf, mask=m), fill_value)
        y = self.filled(self.masked_array(yf, mask=m), fill_value)

        # 如果 x 和 y 的 dtype 不是 "O"（即不是对象型），将它们转换为 np.float64 类型
        if (x.dtype.char != "O"):
            x = x.astype(np.float64)
            # 如果 x 是 ndarray 类型且大小大于 1，则将其中的 NaN 值替换为 0
            if isinstance(x, np.ndarray) and x.size > 1:
                x[np.isnan(x)] = 0
            # 如果 x 是 NaN，则将其替换为 0
            elif np.isnan(x):
                x = 0

        # 类似地处理 y
        if (y.dtype.char != "O"):
            y = y.astype(np.float64)
            if isinstance(y, np.ndarray) and y.size > 1:
                y[np.isnan(y)] = 0
            elif np.isnan(y):
                y = 0

        try:
            # 检查 x 和 y 的形状是否相同，或其中一个是标量
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
            # 如果形状不匹配，生成错误消息
            if not cond:
                msg = build_err_msg([x, y],
                                    err_msg
                                    + f'\n(shapes {x.shape}, {y.shape} mismatch)',
                                    header=header,
                                    names=('x', 'y'))
                assert cond, msg

            # 进行比较操作 comparison(x, y)
            val = comparison(x, y)

            # 如果存在掩码并且 fill_value 为 True，则创建一个 MaskedArray 对象
            if m is not self.nomask and fill_value:
                val = self.masked_array(val, mask=m)

            # 如果 val 是布尔值，判断其条件并设置 reduced 列表
            if isinstance(val, bool):
                cond = val
                reduced = [0]
            else:
                reduced = val.ravel()
                cond = reduced.all()
                reduced = reduced.tolist()

            # 如果条件不满足，计算不匹配百分比，并生成错误消息
            if not cond:
                match = 100-100.0*reduced.count(1)/len(reduced)
                msg = build_err_msg([x, y],
                                    err_msg
                                    + '\n(mismatch %s%%)' % (match,),
                                    header=header,
                                    names=('x', 'y'))
                assert cond, msg

        # 捕获 ValueError 异常，生成错误消息
        except ValueError as e:
            msg = build_err_msg([x, y], err_msg, header=header, names=('x', 'y'))
            raise ValueError(msg) from e

    def assert_array_equal(self, x, y, err_msg=''):
        """
        Checks the elementwise equality of two masked arrays.

        """
        # 调用 assert_array_compare 方法进行两个数组 x 和 y 的元素比较
        self.assert_array_compare(self.equal, x, y, err_msg=err_msg,
                                  header='Arrays are not equal')

    @np.errstate(all='ignore')
    def test_0(self):
        """
        Tests creation

        """
        # 创建包含浮点数和特定掩码的 numpy 数组 x，并使用 self.masked_array 方法创建 MaskedArray 对象
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        xm = self.masked_array(x, mask=m)
        # 访问 xm 的第一个元素
        xm[0]

    @np.errstate(all='ignore')
    @np.errstate(all='ignore')
    # 使用 NumPy 的错误状态管理器，忽略所有错误
    def test_1(self):
        """
        Tests creation

        """
        # 创建测试数据
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        # 定义两个掩码
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        # 使用自定义方法创建掩码数组
        xm = self.masked_array(x, mask=m1)
        ym = self.masked_array(y, mask=m2)
        # 创建一个填充值为 1.e+20 的新数组
        xf = np.where(m1, 1.e+20, x)
        # 设置 xm 的填充值为 1.e+20
        xm.set_fill_value(1.e+20)

        # 断言 xm 和 ym 的差异中任何一个被填充后不为零
        assert((xm-ym).filled(0).any())
        # 获取 x 的形状
        s = x.shape
        # 断言 xm 的大小等于其形状元素数的乘积
        assert(xm.size == reduce(lambda x, y:x*y, s))
        # 断言计算 xm 中未掩码元素的数量等于 m1 中 1 的数量
        assert(self.count(xm) == len(m1) - reduce(lambda x, y:x+y, m1))

        # 遍历不同形状并重置数组形状进行断言
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s
            # 再次断言计算 xm 中未掩码元素的数量等于 m1 中 1 的数量
            assert(self.count(xm) == len(m1) - reduce(lambda x, y:x+y, m1))

    @np.errstate(all='ignore')
    # 使用 NumPy 的错误状态管理器，忽略所有错误
    def test_2(self):
        """
        Tests conversions and indexing.

        """
        # 创建初始数组
        x1 = np.array([1, 2, 4, 3])
        # 使用自定义方法创建带有掩码的数组
        x2 = self.array(x1, mask=[1, 0, 0, 0])
        x3 = self.array(x1, mask=[0, 1, 0, 1])
        x4 = self.array(x1)
        # 测试转换为字符串，确保没有错误
        str(x2)
        repr(x2)
        # 索引测试
        assert type(x2[1]) is type(x1[1])
        assert x1[1] == x2[1]
        x1[2] = 9
        x2[2] = 9
        self.assert_array_equal(x1, x2)
        x1[1:3] = 99
        x2[1:3] = 99
        x2[1] = self.masked
        x2[1:3] = self.masked
        x2[:] = x1
        x2[1] = self.masked
        x3[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        x4[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        x1 = np.arange(5)*1.0
        # 使用自定义方法创建带有特定屏蔽值的数组
        x2 = self.masked_values(x1, 3.0)
        x1 = self.array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        # 检查是否没有发生错误
        x1[1]
        x2[1]
        # 断言切片后的形状为空元组
        assert x1[1:1].shape == (0,)
        # 复制大小测试
        n = [0, 0, 1, 0, 0]
        m = self.make_mask(n)
        m2 = self.make_mask(m)
        # 断言 m 和 m2 是同一个对象
        assert(m is m2)
        m3 = self.make_mask(m, copy=1)
        # 断言 m 和 m3 不是同一个对象
        assert(m is not m3)

    @np.errstate(all='ignore')
    # 使用 NumPy 的错误状态管理器，忽略所有错误
    def test_3(self):
        """
        Tests resize/repeat

        """
        # 创建数组并将索引为 2 的元素设置为掩码
        x4 = self.arange(4)
        x4[2] = self.masked
        # 调整数组大小至 (8,)
        y4 = self.resize(x4, (8,))
        # 断言连接 x4 两次的结果等于 y4
        assert self.allequal(self.concatenate([x4, x4]), y4)
        # 断言 y4 的掩码为 [0, 0, 1, 0, 0, 0, 1, 0]
        assert self.allequal(self.getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        # 将 x4 按指定重复次数重复，并进行断言
        y5 = self.repeat(x4, (2, 2, 2, 2), axis=0)
        self.assert_array_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = self.repeat(x4, 2, axis=0)
        assert self.allequal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert self.allequal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert self.allequal(y5, y8)
    # 定义一个测试方法，用于测试 take、transpose、inner、outer 函数的功能
    def test_4(self):
        """
        Test of take, transpose, inner, outer products.
        
        测试 take、transpose、inner、outer 函数的功能。
        """
        # 创建一个包含 24 个元素的数组 x，元素从 0 到 23
        x = self.arange(24)
        # 创建一个包含 24 个元素的数组 y，元素从 0 到 23
        y = np.arange(24)
        # 将 x 数组中索引为 5 的元素替换为 self.masked
        x[5:6] = self.masked
        # 将 x 数组重塑为形状为 (2, 3, 4) 的三维数组
        x = x.reshape(2, 3, 4)
        # 将 y 数组重塑为形状为 (2, 3, 4) 的三维数组
        y = y.reshape(2, 3, 4)
        
        # 断言 np.transpose(y, (2, 0, 1)) 与 self.transpose(x, (2, 0, 1)) 相等
        assert self.allequal(np.transpose(y, (2, 0, 1)), self.transpose(x, (2, 0, 1)))
        # 断言 np.take(y, (2, 0, 1), 1) 与 self.take(x, (2, 0, 1), 1) 相等
        assert self.allequal(np.take(y, (2, 0, 1), 1), self.take(x, (2, 0, 1), 1))
        # 断言 np.inner(self.filled(x, 0), self.filled(y, 0)) 与 self.inner(x, y) 相等
        assert self.allequal(np.inner(self.filled(x, 0), self.filled(y, 0)),
                            self.inner(x, y))
        # 断言 np.outer(self.filled(x, 0), self.filled(y, 0)) 与 self.outer(x, y) 相等
        assert self.allequal(np.outer(self.filled(x, 0), self.filled(y, 0)),
                            self.outer(x, y))
        
        # 创建一个包含字符串和整数的对象数组 y
        y = self.array(['abc', 1, 'def', 2, 3], object)
        # 将 y 数组中索引为 2 的元素替换为 self.masked
        y[2] = self.masked
        # 从 y 数组中取出索引为 [0, 3, 4] 的元素
        t = self.take(y, [0, 3, 4])
        # 断言取出的 t 数组的第一个元素等于 'abc'
        assert t[0] == 'abc'
        # 断言取出的 t 数组的第二个元素等于 2
        assert t[1] == 2
        # 断言取出的 t 数组的第三个元素等于 3
        assert t[2] == 3

    @np.errstate(all='ignore')
    # 在错误状态中运行测试方法，忽略所有错误
    def test_5(self):
        """
        Tests inplace w/ scalar
        
        测试原地操作与标量。
        """
        # 创建一个包含 10 个元素的数组 x，元素从 0 到 9
        x = self.arange(10)
        # 创建一个包含 10 个元素的数组 y，元素从 0 到 9
        y = self.arange(10)
        # 创建一个包含 10 个元素的数组 xm，并将索引为 2 的元素替换为 self.masked
        xm = self.arange(10)
        xm[2] = self.masked
        # 对 x 数组执行原地加法操作，每个元素加 1
        x += 1
        # 断言加法操作后的 x 数组与 y+1 数组相等
        assert self.allequal(x, y+1)
        # 对 xm 数组执行原地加法操作，每个元素加 1
        xm += 1
        # 断言加法操作后的 xm 数组与 y+1 数组相等
        assert self.allequal(xm, y+1)

        # 重复上述过程，分别测试减法、乘法、除法的原地操作
        x = self.arange(10)
        xm = self.arange(10)
        xm[2] = self.masked
        x -= 1
        assert self.allequal(x, y-1)
        xm -= 1
        assert self.allequal(xm, y-1)

        x = self.arange(10)*1.0
        xm = self.arange(10)*1.0
        xm[2] = self.masked
        x *= 2.0
        assert self.allequal(x, y*2)
        xm *= 2.0
        assert self.allequal(xm, y*2)

        x = self.arange(10)*2
        xm = self.arange(10)*2
        xm[2] = self.masked
        x /= 2
        assert self.allequal(x, y)
        xm /= 2
        assert self.allequal(xm, y)

        x = self.arange(10)*1.0
        xm = self.arange(10)*1.0
        xm[2] = self.masked
        x /= 2.0
        assert self.allequal(x, y/2.0)
        xm /= self.arange(10)
        self.assert_array_equal(xm, self.ones((10,)))

        x = self.arange(10).astype(np.float64)
        xm = self.arange(10)
        xm[2] = self.masked
        x += 1.
        assert self.allequal(x, y + 1.)
    # 定义测试函数 test_6，测试原地操作与数组的相关功能

    x = self.arange(10, dtype=np.float64)
    # 创建长度为 10 的浮点数数组 x

    y = self.arange(10)
    # 创建长度为 10 的整数数组 y

    xm = self.arange(10, dtype=np.float64)
    # 创建长度为 10 的浮点数数组 xm

    xm[2] = self.masked
    # 将 xm 数组的第 2 个位置设置为特定的掩码值 self.masked

    m = xm.mask
    # 获取数组 xm 的掩码

    a = self.arange(10, dtype=np.float64)
    # 创建长度为 10 的浮点数数组 a

    a[-1] = self.masked
    # 将数组 a 的最后一个位置设置为特定的掩码值 self.masked

    x += a
    # 将数组 x 与数组 a 原地相加

    xm += a
    # 将数组 xm 与数组 a 原地相加

    assert self.allequal(x, y+a)
    # 断言数组 x 应该与数组 y 加上数组 a 后相等

    assert self.allequal(xm, y+a)
    # 断言数组 xm 应该与数组 y 加上数组 a 后相等

    assert self.allequal(xm.mask, self.mask_or(m, a.mask))
    # 断言数组 xm 的掩码应该与数组 xm 的旧掩码 m 或上数组 a 的掩码后相等

    x = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 x

    xm = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 xm

    xm[2] = self.masked
    # 将 xm 数组的第 2 个位置设置为特定的掩码值 self.masked

    m = xm.mask
    # 获取数组 xm 的掩码

    a = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 a

    a[-1] = self.masked
    # 将数组 a 的最后一个位置设置为特定的掩码值 self.masked

    x -= a
    # 将数组 x 减去数组 a

    xm -= a
    # 将数组 xm 减去数组 a

    assert self.allequal(x, y-a)
    # 断言数组 x 应该与数组 y 减去数组 a 后相等

    assert self.allequal(xm, y-a)
    # 断言数组 xm 应该与数组 y 减去数组 a 后相等

    assert self.allequal(xm.mask, self.mask_or(m, a.mask))
    # 断言数组 xm 的掩码应该与数组 xm 的旧掩码 m 或上数组 a 的掩码后相等

    x = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 x

    xm = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 xm

    xm[2] = self.masked
    # 将 xm 数组的第 2 个位置设置为特定的掩码值 self.masked

    m = xm.mask
    # 获取数组 xm 的掩码

    a = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 a

    a[-1] = self.masked
    # 将数组 a 的最后一个位置设置为特定的掩码值 self.masked

    x *= a
    # 将数组 x 与数组 a 原地相乘

    xm *= a
    # 将数组 xm 与数组 a 原地相乘

    assert self.allequal(x, y*a)
    # 断言数组 x 应该与数组 y 乘以数组 a 后相等

    assert self.allequal(xm, y*a)
    # 断言数组 xm 应该与数组 y 乘以数组 a 后相等

    assert self.allequal(xm.mask, self.mask_or(m, a.mask))
    # 断言数组 xm 的掩码应该与数组 xm 的旧掩码 m 或上数组 a 的掩码后相等

    x = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 x

    xm = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 xm

    xm[2] = self.masked
    # 将 xm 数组的第 2 个位置设置为特定的掩码值 self.masked

    m = xm.mask
    # 获取数组 xm 的掩码

    a = self.arange(10, dtype=np.float64)
    # 重新创建长度为 10 的浮点数数组 a

    a[-1] = self.masked
    # 将数组 a 的最后一个位置设置为特定的掩码值 self.masked

    x /= a
    # 将数组 x 除以数组 a

    xm /= a
    # 将数组 xm 除以数组 a
# 循环遍历定义的一组数学函数名称列表
for f in [
             'sin', 'cos', 'tan',
             'arcsin', 'arccos', 'arctan',
             'sinh', 'cosh', 'tanh',
             'arcsinh',
             'arccosh',
             'arctanh',
             'absolute', 'fabs', 'negative',
             # 'nonzero', 'around',
             'floor', 'ceil',
             # 'sometrue', 'alltrue',
             'logical_not',
             'add', 'subtract', 'multiply',
             'divide', 'true_divide', 'floor_divide',
             'remainder', 'fmod', 'hypot', 'arctan2',
             'equal', 'not_equal', 'less_equal', 'greater_equal',
             'less', 'greater',
             'logical_and', 'logical_or', 'logical_xor',
         ]:
    try:
        # 尝试从 self.umath 中获取对应名称的函数对象
        uf = getattr(self.umath, f)
    except AttributeError:
        # 如果在 self.umath 中找不到，则从 fromnumeric 中获取对应名称的函数对象
        uf = getattr(fromnumeric, f)
    # 从 self.module 中获取对应名称的函数对象
    mf = getattr(self.module, f)
    # 从参数字典 d 中获取前 uf.nin 个参数作为函数调用的参数
    args = d[:uf.nin]
    # 调用 uf 函数并存储结果
    ur = uf(*args)
    # 调用 mf 函数并存储结果
    mr = mf(*args)
    # 使用 self.assert_array_equal 方法比较 ur 和 mr 对象的填充后值是否相等，并打印出错信息 f
    self.assert_array_equal(ur.filled(0), mr.filled(0), f)
    # 使用 self.assert_array_equal 方法比较 ur 和 mr 对象的掩码是否相等
    self.assert_array_equal(ur._mask, mr._mask)

# 在 numpy 中忽略所有错误状态
@np.errstate(all='ignore')
    # 定义一个测试方法，用于测试average函数的不同用例
    def test_99(self):
        # 创建一个带有部分掩码的数组ott，测试average函数在axis=0时的平均值
        ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        self.assert_array_equal(2.0, self.average(ott, axis=0))
        # 测试带权重的average函数，验证平均值计算是否正确
        self.assert_array_equal(2.0, self.average(ott, weights=[1., 1., 2., 1.]))
        # 测试返回权重的average函数，确保返回的结果和权重值正确
        result, wts = self.average(ott, weights=[1., 1., 2., 1.], returned=1)
        self.assert_array_equal(2.0, result)
        assert(wts == 4.0)
        # 将ott数组全部置为masked，验证average函数在axis=0时的返回值是否正确为masked
        ott[:] = self.masked
        assert(self.average(ott, axis=0) is self.masked)
        # 将ott数组重新定义为未masked的数组，并reshape为2x2的矩阵
        ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        ott = ott.reshape(2, 2)
        # 将ott的第二列置为masked，测试average函数在axis=0时的平均值计算
        ott[:, 1] = self.masked
        self.assert_array_equal(self.average(ott, axis=0), [2.0, 0.0])
        # 验证average函数在axis=1时，第一行的返回值是否正确为masked
        assert(self.average(ott, axis=1)[0] is self.masked)
        # 再次验证average函数在axis=0时的平均值计算是否正确
        self.assert_array_equal([2., 0.], self.average(ott, axis=0))
        # 测试返回权重的average函数在axis=0时，确保返回的权重值正确
        result, wts = self.average(ott, axis=0, returned=1)
        self.assert_array_equal(wts, [1., 0.])
        # 定义权重数组w1和w2，并创建一个包含0到5的数组x
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = self.arange(6)
        # 验证average函数在axis=0时，计算数组x的平均值是否正确
        self.assert_array_equal(self.average(x, axis=0), 2.5)
        # 验证带权重的average函数在axis=0时，计算数组x的平均值是否正确
        self.assert_array_equal(self.average(x, axis=0, weights=w1), 2.5)
        # 创建一个包含两行的数组y，并验证average函数在None条件下的计算结果是否正确
        y = self.array([self.arange(6), 2.0*self.arange(6)])
        self.assert_array_equal(self.average(y, None), np.add.reduce(np.arange(6))*3./12.)
        # 验证average函数在axis=0时，计算数组y的平均值是否正确
        self.assert_array_equal(self.average(y, axis=0), np.arange(6) * 3./2.)
        # 验证average函数在axis=1时，计算数组y的平均值是否正确
        self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
        # 验证带权重的average函数在None条件下，计算数组y的平均值是否正确
        self.assert_array_equal(self.average(y, None, weights=w2), 20./6.)
        # 验证带权重的average函数在axis=0时，计算数组y的平均值是否正确
        self.assert_array_equal(self.average(y, axis=0, weights=w2), [0., 1., 2., 3., 4., 10.])
        # 定义一些掩码数组m1到m5，并验证average函数在对应掩码条件下的平均值计算结果是否正确
        m1 = self.zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = self.ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        self.assert_array_equal(self.average(self.masked_array(x, m1), axis=0), 2.5)
        self.assert_array_equal(self.average(self.masked_array(x, m2), axis=0), 2.5)
        self.assert_array_equal(self.average(self.masked_array(x, m5), axis=0), 0.0)
        self.assert_array_equal(self.count(self.average(self.masked_array(x, m4), axis=0)), 0)
        # 创建一个包含掩码数组m3的数组z，并验证average函数在不同axis条件下的计算结果是否正确
        z = self.masked_array(y, m3)
        self.assert_array_equal(self.average(z, None), 20./6.)
        self.assert_array_equal(self.average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
        self.assert_array_equal(self.average(z, axis=1), [2.5, 5.0])
        # 验证带权重的average函数在axis=0时，计算数组z的平均值是否正确
        self.assert_array_equal(self.average(z, axis=0, weights=w2), [0., 1., 99., 99., 4.0, 10.0])

    # 定义一个测试方法，用于测试在忽略所有错误情况下，数组操作的功能
    @np.errstate(all='ignore')
    def test_A(self):
        # 创建一个包含掩码值的数组x，并对其中一个元素应用masked操作
        x = self.arange(24)
        x[5:6] = self.masked
        # 将数组x重新reshape为2x3x4的三维数组
        x = x.reshape(2, 3, 4)
if __name__ == '__main__':
    # 如果当前脚本作为主程序执行
    setup_base = ("from __main__ import ModuleTester \n"
                  "import numpy\n"
                  "tester = ModuleTester(module)\n")
    # 设置基础的导入语句和模块测试器的初始化
    setup_cur = "import numpy.ma.core as module\n" + setup_base
    # 设置当前的导入语句，包括numpy.ma.core作为module，并继承基础设置

    (nrepeat, nloop) = (10, 10)
    # 定义重复执行和循环次数的元组

    for i in range(1, 8):
        # 循环执行以下操作，i从1到7
        func = 'tester.test_%i()' % i
        # 生成当前测试函数的字符串表示形式，例如'tester.test_1()'
        cur = timeit.Timer(func, setup_cur).repeat(nrepeat, nloop*10)
        # 使用timeit.Timer来测量func函数的执行时间，重复nrepeat次，每次循环执行nloop*10次
        cur = np.sort(cur)
        # 对测量结果进行排序
        print("#%i" % i + 50*'.')
        # 打印当前测试编号，以及一行分隔符
        print(eval("ModuleTester.test_%i.__doc__" % i))
        # 打印当前测试函数的文档字符串
        print(f'core_current : {cur[0]:.3f} - {cur[1]:.3f}')
        # 打印测量结果的最小值和最大值，保留三位小数精度
```