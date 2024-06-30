# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_bracket.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_array_less, assert_allclose, assert_equal  # 从 NumPy.testing 模块导入三个断言方法

from scipy.optimize._bracket import _bracket_root, _bracket_minimum, _ELIMITS  # 导入 SciPy 中的优化模块中的函数和常量
import scipy._lib._elementwise_iterative_method as eim  # 导入 SciPy 私有库中的迭代方法模块别名为 eim
from scipy import stats  # 导入 SciPy 统计模块中的 stats 子模块

class TestBracketRoot:
    @pytest.mark.parametrize("seed", (615655101, 3141866013, 238075752))  # 参数化装饰器，为 seed 参数设置多个值进行测试
    @pytest.mark.parametrize("use_xmin", (False, True))  # 参数化装饰器，为 use_xmin 参数设置 False 和 True 进行测试
    @pytest.mark.parametrize("other_side", (False, True))  # 参数化装饰器，为 other_side 参数设置 False 和 True 进行测试
    @pytest.mark.parametrize("fix_one_side", (False, True))  # 参数化装饰器，为 fix_one_side 参数设置 False 和 True 进行测试
    def f(self, q, p):
        return stats.norm.cdf(q) - p  # 定义函数 f，计算标准正态分布的累积分布函数值减去 p 的结果

    @pytest.mark.parametrize('p', [0.6, np.linspace(0.05, 0.95, 10)])  # 参数化装饰器，为 p 参数设置多个值进行测试
    @pytest.mark.parametrize('xmin', [-5, None])  # 参数化装饰器，为 xmin 参数设置 -5 和 None 进行测试
    @pytest.mark.parametrize('xmax', [5, None])  # 参数化装饰器，为 xmax 参数设置 5 和 None 进行测试
    @pytest.mark.parametrize('factor', [1.2, 2])  # 参数化装饰器，为 factor 参数设置 1.2 和 2 进行测试
    def test_basic(self, p, xmin, xmax, factor):
        # 测试基本功能以找到根（分布的 PPF）
        res = _bracket_root(self.f, -0.01, 0.01, xmin=xmin, xmax=xmax,
                            factor=factor, args=(p,))
        assert_equal(-np.sign(res.fl), np.sign(res.fr))  # 断言左右界限 fl 和 fr 的符号相反

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    # 定义一个测试向量化功能的方法，用于检查不同输入形状的正确功能、输出形状和数据类型。
    def test_vectorization(self, shape):
        # 创建一个线性空间的数组，用于测试目的，如果有指定的形状则使用该形状，否则使用标量值0.6
        p = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        # 参数元组，包含单个参数p
        args = (p,)
        # 最大迭代次数设为10
        maxiter = 10

        # 定义一个使用向量化修饰器的函数，用于计算根的区间
        @np.vectorize
        def bracket_root_single(xl0, xr0, xmin, xmax, factor, p):
            return _bracket_root(self.f, xl0, xr0, xmin=xmin, xmax=xmax,
                                 factor=factor, args=(p,),
                                 maxiter=maxiter)

        # 定义一个函数f，用于包装self.f，并计数函数调用次数
        def f(*args, **kwargs):
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0

        # 使用指定种子创建一个随机数生成器对象
        rng = np.random.default_rng(2348234)
        # 创建随机数数组xl0，大小与指定形状相同
        xl0 = -rng.random(size=shape)
        # 创建随机数数组xr0，大小与指定形状相同
        xr0 = rng.random(size=shape)
        # xmin和xmax分别为xl0和xr0乘以1000
        xmin, xmax = 1e3*xl0, 1e3*xr0
        # 如果有指定形状，则对部分元素执行条件操作，将其xmin和xmax设置为负无穷和正无穷
        if shape:
            i = rng.random(size=shape) > 0.5
            xmin[i], xmax[i] = -np.inf, np.inf
        # 创建随机数数组factor，大小与指定形状相同，并加上1.5
        factor = rng.random(size=shape) + 1.5
        # 调用_bracket_root函数，计算根的区间，返回结果存储在res中
        res = _bracket_root(f, xl0, xr0, xmin=xmin, xmax=xmax, factor=factor,
                            args=args, maxiter=maxiter)
        # 调用bracket_root_single函数，计算根的区间，将结果展平并存储在refs中
        refs = bracket_root_single(xl0, xr0, xmin, xmax, factor, p).ravel()

        # 属性列表，用于比较res和refs的属性值
        attrs = ['xl', 'xr', 'fl', 'fr', 'success', 'nfev', 'nit']
        # 遍历属性列表，比较每个属性的值
        for attr in attrs:
            # 获取refs中每个对象的attr属性值，并存储在ref_attr中
            ref_attr = [getattr(ref, attr) for ref in refs]
            # 获取res对象的attr属性值
            res_attr = getattr(res, attr)
            # 使用assert_allclose断言函数，比较res_attr和ref_attr的值是否接近
            assert_allclose(res_attr.ravel(), ref_attr)
            # 使用assert_equal断言函数，比较res_attr的形状是否与指定形状相同
            assert_equal(res_attr.shape, shape)

        # 使用assert断言函数，检查res.success的数据类型是否为布尔类型
        assert np.issubdtype(res.success.dtype, np.bool_)
        # 如果有指定形状，则使用assert检查res.success数组的第2到倒数第2个元素是否全部为True
        if shape:
            assert np.all(res.success[1:-1])
        # 使用assert断言函数，检查res.status的数据类型是否为整数类型
        assert np.issubdtype(res.status.dtype, np.integer)
        # 使用assert断言函数，检查res.nfev的数据类型是否为整数类型
        assert np.issubdtype(res.nfev.dtype, np.integer)
        # 使用assert断言函数，检查res.nit的数据类型是否为整数类型
        assert np.issubdtype(res.nit.dtype, np.integer)
        # 使用assert_equal断言函数，比较res.nit中的最大值是否等于f.f_evals减2
        assert_equal(np.max(res.nit), f.f_evals - 2)
        # 使用assert_array_less断言函数，比较res.xl和res.xr的对应元素是否满足xl小于xr的关系
        assert_array_less(res.xl, res.xr)
        # 使用assert_allclose断言函数，比较res.fl和self.f(res.xl, *args)的值是否接近
        assert_allclose(res.fl, self.f(res.xl, *args))
        # 使用assert_allclose断言函数，比较res.fr和self.f(res.xr, *args)的值是否接近
        assert_allclose(res.fr, self.f(res.xr, *args))
    def test_flags(self):
        # Test cases that should produce different status flags; show that all
        # can be produced simultaneously.
        
        # 定义一个内部函数 f，接受两个参数 xs 和 js，返回根据不同函数操作后的列表
        def f(xs, js):
            # 定义一组函数列表，每个函数对输入 x 进行不同的操作
            funcs = [lambda x: x - 1.5,
                     lambda x: x - 1000,
                     lambda x: x - 1000,
                     lambda x: np.nan,
                     lambda x: x]

            # 使用 zip 将 xs 和 js 合并，对每个组合应用对应的函数，并返回结果列表
            return [funcs[j](x) for x, j in zip(xs, js)]

        # 设置参数 args 为一个包含 np.arange(5) 的元组
        args = (np.arange(5, dtype=np.int64),)
        
        # 调用 _bracket_root 函数，并传入多个参数
        res = _bracket_root(f,
                            xl0=[-1, -1, -1, -1, 4],
                            xr0=[1, 1, 1, 1, -4],
                            xmin=[-np.inf, -1, -np.inf, -np.inf, 6],
                            xmax=[np.inf, 1, np.inf, np.inf, 2],
                            args=args, maxiter=3)

        # 创建一个参考的状态标志数组 ref_flags
        ref_flags = np.array([eim._ECONVERGED,
                              _ELIMITS,
                              eim._ECONVERR,
                              eim._EVALUEERR,
                              eim._EINPUTERR])

        # 使用 assert_equal 检查 res.status 是否等于 ref_flags
        assert_equal(res.status, ref_flags)

    # 使用 pytest.mark.parametrize 标记参数化测试
    @pytest.mark.parametrize("root", (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize('xmin', [-5, None])
    @pytest.mark.parametrize('xmax', [5, None])
    @pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
    def test_dtype(self, root, xmin, xmax, dtype):
        # Test that dtypes are preserved
        
        # 将 xmin 和 xmax 转换为 dtype 类型，如果它们不是 None 的话
        xmin = xmin if xmin is None else dtype(xmin)
        xmax = xmax if xmax is None else dtype(xmax)
        root = dtype(root)
        
        # 定义函数 f，接受参数 x 和 root，计算 ((x - root) ** 3) 并将结果转换为 dtype 类型
        def f(x, root):
            return ((x - root) ** 3).astype(dtype)

        # 创建一个 dtype 类型的数组 bracket，并将其作为参数传递给 _bracket_root 函数
        bracket = np.asarray([-0.01, 0.01], dtype=dtype)
        
        # 调用 _bracket_root 函数，并传入各种参数，存储结果到 res
        res = _bracket_root(f, *bracket, xmin=xmin, xmax=xmax, args=(root,))
        
        # 使用 assert 检查所有成功标志是否为 True
        assert np.all(res.success)
        
        # 使用 assert 检查 res.xl 和 res.xr 的数据类型是否都为 dtype
        assert res.xl.dtype == res.xr.dtype == dtype
        
        # 使用 assert 检查 res.fl 和 res.fr 的数据类型是否都为 dtype
        assert res.fl.dtype == res.fr.dtype == dtype
    # 定义单元测试函数，用于测试输入验证的准确错误信息

    # 测试当 `func` 不是可调用对象时是否会引发 ValueError 异常，匹配给定的错误消息
    message = '`func` must be callable.'
    with pytest.raises(ValueError, match=message):
        _bracket_root(None, -4, 4)

    # 测试当传入的参数不是实数时是否会引发 ValueError 异常，匹配给定的错误消息
    message = '...must be numeric and real.'
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4+1j, 4)
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 'hello')
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, xmin=np)
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, xmax=object())
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, factor=sum)

    # 测试当传入的 `factor` 参数中有非正数值时是否会引发 ValueError 异常，匹配给定的错误消息
    message = "All elements of `factor` must be greater than 1."
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, factor=0.5)

    # 测试当传入的参数不满足广播要求时是否会引发 ValueError 异常，匹配给定的错误消息
    message = "shape mismatch: objects cannot be broadcast"
    # 由 `np.broadcast` 引发，但回溯信息可读性较好
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, [-2, -3], [3, 4, 5])
    # 考虑对此情况提供更具可读性的错误消息
    # with pytest.raises(ValueError, match=message):
    #     _bracket_root(lambda x: [x[0], x[1], x[1]], [-3, -3], [5, 5])

    # 测试当传入的 `maxiter` 不是非负整数时是否会引发 ValueError 异常，匹配给定的错误消息
    message = '`maxiter` must be a non-negative integer.'
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, maxiter=1.5)
    with pytest.raises(ValueError, match=message):
        _bracket_root(lambda x: x, -4, 4, maxiter=-1)
    def test_special_cases(self):
        # Test edge cases and other special cases

        # Test that integers are not passed to `f`
        # (otherwise this would overflow)
        def f(x):
            # Ensure that x is a floating-point number
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1

        # Perform root bracketing and check for success
        res = _bracket_root(f, -7, 5)
        assert res.success

        # Test maxiter = 0. Should do nothing to bracket.
        def f(x):
            return x - 10

        # Define initial bracket
        bracket = (-3, 5)
        # Call root bracketing function with maxiter set to 0
        res = _bracket_root(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket  # Check if bracket remains unchanged
        assert res.nit == 0  # Ensure no iterations were performed
        assert res.nfev == 2  # Check the number of function evaluations
        assert res.status == -2  # Verify the status code

        # Test scalar `args` (not in tuple)
        def f(x, c):
            return c*x - 1

        # Perform root bracketing with additional argument
        res = _bracket_root(f, -1, 1, args=3)
        assert res.success
        assert_allclose(res.fl, f(res.xl, 3))  # Check the function value at xl

        # Test other edge cases

        def f(x):
            f.count += 1
            return x

        # 1. root lies within guess of bracket
        f.count = 0
        _bracket_root(f, -10, 20)
        assert_equal(f.count, 2)  # Ensure two function evaluations were done

        # 2. bracket endpoint hits root exactly
        f.count = 0
        res = _bracket_root(f, 5, 10, factor=2)
        bracket = (res.xl, res.xr)
        assert_equal(res.nfev, 4)  # Verify the number of function evaluations
        assert_allclose(bracket, (0, 5), atol=1e-15)  # Check bracket endpoints

        # 3. bracket limit hits root exactly
        with np.errstate(over='ignore'):
            res = _bracket_root(f, 5, 10, xmin=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[0], 0, atol=1e-15)  # Check if xl hits the limit
        with np.errstate(over='ignore'):
            res = _bracket_root(f, -10, -5, xmax=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[1], 0, atol=1e-15)  # Check if xr hits the limit

        # 4. bracket not within min, max
        with np.errstate(over='ignore'):
            res = _bracket_root(f, 5, 10, xmin=1)
        assert not res.success  # Ensure the function did not succeed
# 定义一个测试类 TestBracketMinimum，用于测试括号最小化算法

class TestBracketMinimum:
    
    # 初始化函数 init_f，返回一个函数 f(x, a, b)，并初始化计数器 f.count
    def init_f(self):
        def f(x, a, b):
            f.count += 1  # 每调用一次 f 函数，计数器 f.count 自增一次
            return (x - a)**2 + b
        f.count = 0  # 初始化 f 函数的调用次数为 0
        return f

    # 断言验证函数 assert_valid_bracket，用于验证结果是否符合括号最小化算法的条件
    def assert_valid_bracket(self, result):
        assert np.all(
            (result.xl < result.xm) & (result.xm < result.xr)
        )  # 检查左、右和中点的顺序关系是否满足 xl < xm < xr
        assert np.all(
            (result.fl >= result.fm) & (result.fr > result.fm)
            | (result.fl > result.fm) & (result.fr > result.fm)
        )  # 检查左右端点和中点的函数值关系是否满足 fl >= fm > fr 或 fl > fm >= fr

    # 获取参数字典的函数 get_kwargs，用于提取参数并生成关键字参数字典
    def get_kwargs(
            self, *, xl0=None, xr0=None, factor=None, xmin=None, xmax=None, args=()
    ):
        # 参数名称列表
        names = ("xl0", "xr0", "xmin", "xmax", "factor", "args")
        # 使用列表推导式生成关键字参数字典，筛选出数组或标量值，且不为 None、空元组的参数
        return {
            name: val for name, val in zip(names, (xl0, xr0, xmin, xmax, factor, args))
            if isinstance(val, np.ndarray) or np.isscalar(val)
            or val not in [None, ()]
        }

    # 使用 pytest 的 parametrize 装饰器多参数化测试函数，分别测试不同的种子和参数选项
    @pytest.mark.parametrize(
        "seed",
        (
            307448016549685229886351382450158984917,
            11650702770735516532954347931959000479,
            113767103358505514764278732330028568336,
        )
    )
    @pytest.mark.parametrize("use_xmin", (False, True))
    @pytest.mark.parametrize("other_side", (False, True))
    # 定义测试函数，用于验证期望的 nfev 值是否正确
    def test_nfev_expected(self, seed, use_xmin, other_side):
        # 使用指定种子创建随机数生成器
        rng = np.random.default_rng(seed)
        # 设置测试函数的参数，这里是 f(x) = x^2，其最小值在 x = 0 处
        args = (0, 0)

        # 随机生成 xl0, d1, d2, factor 四个数值，用于构建初始搜索区间
        xl0, d1, d2, factor = rng.random(size=4) * [1e5, 10, 10, 5]
        xm0 = xl0 + d1  # 确定中间点 xm0
        xr0 = xm0 + d2  # 确定右侧点 xr0

        # 确保 factor 大于 1
        factor += 1

        # 如果 use_xmin 为 True，则设置最小值 xmin 并调整搜索区间
        if use_xmin:
            xmin = -rng.random() * 5
            # 计算所需的迭代次数 n
            n = int(np.ceil(np.log(-(xl0 - xmin) / xmin) / np.log(factor)))
            lower = xmin + (xl0 - xmin)*factor**-n
            middle = xmin + (xl0 - xmin)*factor**-(n-1)
            upper = xmin + (xl0 - xmin)*factor**-(n-2) if n > 1 else xm0
            # 若中间点 middle 的平方大于下界 lower 的平方，则增加迭代次数 n
            if middle**2 > lower**2:
                n += 1
                lower, middle, upper = (
                    xmin + (xl0 - xmin)*factor**-n, lower, middle
                )
        else:
            # 如果 use_xmin 为 False，则不设置最小值 xmin
            xmin = None
            # 计算所需的迭代次数 n
            n = int(np.ceil(np.log(xl0 / d1) / np.log(factor)))
            lower = xl0 - d1*factor**n
            middle = xl0 - d1*factor**(n-1) if n > 1 else xl0
            upper = xl0 - d1*factor**(n-2) if n > 1 else xm0
            # 若中间点 middle 的平方大于下界 lower 的平方，则增加迭代次数 n
            if middle**2 > lower**2:
                n += 1
                lower, middle, upper = (
                    xl0 - d1*factor**n, lower, middle
                )

        # 初始化测试函数 f
        f = self.init_f()

        # 如果 other_side 为 True，则反转搜索区间以测试另一侧情况
        if other_side:
            xl0, xm0, xr0 = -xr0, -xm0, -xl0
            xmin, xmax = None, -xmin if xmin is not None else None
            lower, middle, upper = -upper, -middle, -lower

        # 获取函数调用的关键字参数
        kwargs = self.get_kwargs(
            xl0=xl0, xr0=xr0, xmin=xmin, xmax=xmax, factor=factor, args=args
        )

        # 调用函数 _bracket_minimum 进行搜索
        result = _bracket_minimum(f, xm0, **kwargs)

        # 检查 nfev 和 nit 的关系是否正确
        assert result.nfev == result.nit + 3
        # 检查 nfev 是否报告了正确数量的函数评估次数
        assert result.nfev == f.count
        # 检查迭代次数 nit 是否与理论值 n 相符
        assert result.nit == n

        # 比较报告的搜索区间与理论搜索区间，并比较函数值与搜索区间上的实际函数值
        bracket = np.asarray([result.xl, result.xm, result.xr])
        assert_allclose(bracket, (lower, middle, upper))
        f_bracket = np.asarray([result.fl, result.fm, result.fr])
        assert_allclose(f_bracket, f(bracket, *args))

        # 确保返回的结果 result 是有效的搜索区间
        self.assert_valid_bracket(result)
        # 断言搜索是否成功完成
        assert result.status == 0
        assert result.success
    def test_flags(self):
        # Test cases that should produce different status flags; show that all
        # can be produced simultaneously
        # 定义一个嵌套函数f，接受两个参数xs和js，分别为一组数和索引
        def f(xs, js):
            # 定义一个包含5个lambda函数的列表，每个lambda函数接受一个参数x
            # 函数功能分别是：计算(x - 1.5)^2, 返回x本身, 返回x本身, 返回NaN, 计算x^2
            funcs = [lambda x: (x - 1.5)**2,
                     lambda x: x,
                     lambda x: x,
                     lambda x: np.nan,
                     lambda x: x**2]

            # 返回一个列表，通过zip将xs和js中的元素逐一传给funcs中对应的函数进行计算
            return [funcs[j](x) for x, j in zip(xs, js)]

        # 定义元组args，包含一个numpy数组，数组内容为0到4的整数
        args = (np.arange(5, dtype=np.int64),)
        # 定义四个列表，分别为xl0, xm0, xr0, xmin，每个列表包含5个浮点数
        xl0 = [-1.0, -1.0, -1.0, -1.0, 6.0]
        xm0 = [0.0, 0.0, 0.0, 0.0, 4.0]
        xr0 = [1.0, 1.0, 1.0, 1.0, 2.0]
        xmin = [-np.inf, -1.0, -np.inf, -np.inf, 8.0]

        # 调用_bracket_minimum函数，传入函数f、xm0以及一些其他参数
        result = _bracket_minimum(f, xm0, xl0=xl0, xr0=xr0, xmin=xmin,
                                  args=args, maxiter=3)

        # 定义一个numpy数组reference_flags，包含5个特定的标志常量
        reference_flags = np.array([eim._ECONVERGED, _ELIMITS,
                                    eim._ECONVERR, eim._EVALUEERR,
                                    eim._EINPUTERR])
        # 使用assert_equal断言，验证result.status与reference_flags相等
        assert_equal(result.status, reference_flags)

    @pytest.mark.parametrize("minimum", (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
    @pytest.mark.parametrize("xmin", [-5, None])
    @pytest.mark.parametrize("xmax", [5, None])
    # 测试不同的数据类型、最小值、最大值组合
    def test_dtypes(self, minimum, xmin, xmax, dtype):
        # 将xmin和xmax转换为dtype类型，如果它们不是None的话
        xmin = xmin if xmin is None else dtype(xmin)
        xmax = xmax if xmax is None else dtype(xmax)
        # 将minimum转换为dtype类型
        minimum = dtype(minimum)

        # 定义一个函数f，接受两个参数x和minimum，返回一个以dtype类型为元素类型的数组
        def f(x, minimum):
            return ((x - minimum)**2).astype(dtype)

        # 定义三个numpy数组xl0, xm0, xr0，每个数组包含一个dtype类型的浮点数
        xl0, xm0, xr0 = np.array([-0.01, 0.0, 0.01], dtype=dtype)
        # 调用_bracket_minimum函数，传入函数f、xm0以及一些其他参数
        result = _bracket_minimum(
            f, xm0, xl0=xl0, xr0=xr0, xmin=xmin, xmax=xmax, args=(minimum, )
        )
        # 使用assert断言，验证result.success中所有元素都为True
        assert np.all(result.success)
        # 使用assert断言，验证result.xl、result.xm、result.xr的数据类型都为dtype
        assert result.xl.dtype == result.xm.dtype == result.xr.dtype == dtype
        # 使用assert断言，验证result.fl、result.fm、result.fr的数据类型都为dtype
        assert result.fl.dtype == result.fm.dtype == result.fr.dtype == dtype
    def test_input_validation(self):
        # 测试输入验证以获取适当的错误消息

        message = '`func` must be callable.'
        # 测试当 `func` 不可调用时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(None, -4, xl0=4)

        message = '...must be numeric and real.'
        # 测试当输入参数不是数值或实数时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, 4+1j)
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, xl0='hello')
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, xmin=np)
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, xmax=object())
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, factor=sum)

        message = "All elements of `factor` must be greater than 1."
        # 测试当 `factor` 中的所有元素不大于1时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x, -4, factor=0.5)

        message = "shape mismatch: objects cannot be broadcast"
        # 测试当参数在广播时形状不匹配时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, [-2, -3], xl0=[-3, -4, -5])

        message = '`maxiter` must be a non-negative integer.'
        # 测试当 `maxiter` 不是非负整数时是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, xr0=4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            _bracket_minimum(lambda x: x**2, -4, xr0=4, maxiter=-1)

    @pytest.mark.parametrize("xl0", [0.0, None])
    @pytest.mark.parametrize("xm0", (0.05, 0.1, 0.15))
    @pytest.mark.parametrize("xr0", (0.2, 0.4, 0.6, None))
    # 最小值为 `a`，对于每个元组 `(a, b)` 测试最小值在初始区间内、左侧或右侧的不同距离处
    @pytest.mark.parametrize(
        "args",
        (
            (1.2, 0), (-0.5, 0), (0.1, 0), (0.2, 0), (3.6, 0), (21.4, 0),
            (121.6, 0), (5764.1, 0), (-6.4, 0), (-12.9, 0), (-146.2, 0)
        )
    )
    def test_scalar_no_limits(self, xl0, xm0, xr0, args):
        # 准备测试函数对象
        f = self.init_f()
        # 获取测试用的关键字参数
        kwargs = self.get_kwargs(xl0=xl0, xr0=xr0, args=args)
        # 调用 _bracket_minimum 函数进行测试，验证返回结果的有效性
        result = _bracket_minimum(f, xm0, **kwargs)
        self.assert_valid_bracket(result)
        # 断言结果对象的状态为成功
        assert result.status == 0
        assert result.success
        # 断言计算函数调用次数与预期一致
        assert result.nfev == f.count
    @pytest.mark.parametrize(
        # 定义参数化测试，测试不同情况下的xl0, xm0, xr0, xmin
        "xl0,xm0,xr0,xmin",
        (
            # 初始区间，与xmin的距离不同
            (0.5, 0.75, 1.0, 0.0),
            (1.0, 2.5, 4.0, 0.0),
            (2.0, 4.0, 6.0, 0.0),
            (12.0, 16.0, 20.0, 0.0),
            # 测试默认左端点选择。应保证不小于xmin
            (None, 0.75, 1.0, 0.0),
            (None, 2.5, 4.0, 0.0),
            (None, 4.0, 6.0, 0.0),
            (None, 16.0, 20.0, 0.0),
        )
    )
    @pytest.mark.parametrize(
        # 定义参数化测试，测试不同情况下的args
        "args", (
            (0.0, 0.0), # 最小值直接在xmin处
            (1e-300, 0.0), # 最小值非常接近xmin
            (1e-20, 0.0), # 最小值非常接近xmin
            # 最小值与xmin的距离不同
            (0.1, 0.0),
            (0.2, 0.0),
            (0.4, 0.0)
        )
    )
    def test_scalar_with_limit_left(self, xl0, xm0, xr0, xmin, args):
        # 初始化测试函数
        f = self.init_f()
        # 获取关键字参数
        kwargs = self.get_kwargs(xl0=xl0, xr0=xr0, xmin=xmin, args=args)
        # 调用_bracket_minimum函数进行测试
        result = _bracket_minimum(f, xm0, **kwargs)
        # 断言有效的区间结果
        self.assert_valid_bracket(result)
        # 断言结果状态为成功
        assert result.status == 0
        # 断言成功
        assert result.success
        # 断言函数计算次数与预期一致
        assert result.nfev == f.count

    @pytest.mark.parametrize(
        # 定义参数化测试，测试不同情况下的xl0, xm0, xr0, xmax
        "xl0,xm0,xr0,xmax",
        (
            # 区间与xmax的距离不同
            (0.2, 0.3, 0.4, 1.0),
            (0.05, 0.075, 0.1, 1.0),
            (-0.2, -0.1, 0.0, 1.0),
            (-21.2, -17.7, -14.2, 1.0),
            # 测试默认右端点选择。应保证不超过xmax
            (0.2, 0.3, None, 1.0),
            (0.05, 0.075, None, 1.0),
            (-0.2, -0.1, None, 1.0),
            (-21.2, -17.7, None, 1.0),
        )
    )
    @pytest.mark.parametrize(
        # 定义参数化测试，测试不同情况下的args
        "args", (
            (0.9999999999999999, 0.0), # 最小值非常接近xmax
            # 最小值与xmax的距离不同
            (0.9, 0.0),
            (0.7, 0.0),
            (0.5, 0.0)
        )
    )
    def test_scalar_with_limit_right(self, xl0, xm0, xr0, xmax, args):
        # 初始化测试函数
        f = self.init_f()
        # 获取关键字参数
        kwargs = self.get_kwargs(xl0=xl0, xr0=xr0, xmax=xmax, args=args)
        # 调用_bracket_minimum函数进行测试
        result = _bracket_minimum(f, xm0, **kwargs)
        # 断言有效的区间结果
        self.assert_valid_bracket(result)
        # 断言结果状态为成功
        assert result.status == 0
        # 断言成功
        assert result.success
        # 断言函数计算次数与预期一致
        assert result.nfev == f.count
    @pytest.mark.parametrize(
        "xl0,xm0,xr0,xmin,xmax,args",
        (
            (   # Case 1:
                # 初始括号。
                0.2, 
                0.3,
                0.4,
                # 函数从括号向右下降到最小值1.0处。xmax也在1.0处。
                None, 
                1.0,
                (1.0, 0.0)
            ),
            (   # Case 2:
                # 初始括号。
                1.4,
                1.95,
                2.5,
                # 函数从括号向左下降到最小值0.3处，xmin设为0.3。
                0.3,
                None,
                (0.3, 0.0)
            ),
            (
                # Case 3:
                # 初始括号。
                2.6,
                3.25,
                3.9,
                # 函数向右下降到最小值99.4处，xmax设为99.4。测试最小值相对于括号位置较远的情况。
                None,
                99.4,
                (99.4, 0)
            ),
            (
                # Case 4:
                # 初始括号。
                4,
                4.5,
                5,
                # 函数向左下降，最小值为-26.3，xmin设为-26.3。测试最小值相对于括号位置较远的情况。
                -26.3,
                None,
                (-26.3, 0)
            ),
            (
                # Case 5:
                # 类似于上面的Case 1，但测试xl0和xr0的默认值。
                None,
                0.3,
                None,
                None,
                1.0,
                (1.0, 0.0)
            ),
            (   # Case 6:
                # 类似于上面的Case 2，但测试xl0和xr0的默认值。
                None,
                1.95,
                None,
                0.3,
                None,
                (0.3, 0.0)
            ),
            (
                # Case 7:
                # 类似于上面的Case 3，但测试xl0和xr0的默认值。
                None,
                3.25,
                None,
                None,
                99.4,
                (99.4, 0)
            ),
            (
                # Case 8:
                # 类似于上面的Case 4，但测试xl0和xr0的默认值。
                None,
                4.5,
                None,
                -26.3,
                None,
                (-26.3, 0)
            ),
        )
    )
    def test_minimum_at_boundary_point(self, xl0, xm0, xr0, xmin, xmax, args):
        # 初始化测试函数
        f = self.init_f()
        # 获取参数字典
        kwargs = self.get_kwargs(xr0=xr0, xmin=xmin, xmax=xmax, args=args)
        # 调用 _bracket_minimum 函数进行最小值搜索
        result = _bracket_minimum(f, xm0, **kwargs)
        # 断言检查返回的状态是否为 -1
        assert result.status == -1
        # 断言检查结果中 xl 或 xr 是否与参数 args[0] 相等
        assert args[0] in (result.xl, result.xr)
        # 断言检查函数计算次数是否正确
        assert result.nfev == f.count

    @pytest.mark.parametrize('shape', [tuple(), (12, ), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        # 测试不同输入形状的正确功能、输出形状和数据类型
        a = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (a, 0.0)
        maxiter = 10

        @np.vectorize
        def bracket_minimum_single(xm0, xl0, xr0, xmin, xmax, factor, a):
            # 使用单个参数调用 _bracket_minimum 函数
            return _bracket_minimum(self.init_f(), xm0, xl0=xl0, xr0=xr0, xmin=xmin,
                                    xmax=xmax, factor=factor, maxiter=maxiter,
                                    args=(a, 0.0))

        # 初始化测试函数
        f = self.init_f()

        # 随机数生成器
        rng = np.random.default_rng(2348234)
        xl0 = -rng.random(size=shape)
        xr0 = rng.random(size=shape)
        xm0 = xl0 + rng.random(size=shape) * (xr0 - xl0)
        xmin, xmax = 1e3*xl0, 1e3*xr0
        if shape:  # 使部分元素为无穷大
            i = rng.random(size=shape) > 0.5
            xmin[i], xmax[i] = -np.inf, np.inf
        factor = rng.random(size=shape) + 1.5
        # 调用 _bracket_minimum 函数
        res = _bracket_minimum(f, xm0, xl0=xl0, xr0=xr0, xmin=xmin, xmax=xmax,
                               factor=factor, args=args, maxiter=maxiter)
        # 调用 bracket_minimum_single 函数获取参考结果
        refs = bracket_minimum_single(xm0, xl0, xr0, xmin, xmax, factor, a).ravel()

        # 结果属性列表
        attrs = ['xl', 'xm', 'xr', 'fl', 'fm', 'fr', 'success', 'nfev', 'nit']
        for attr in attrs:
            # 获取参考结果的属性值列表
            ref_attr = [getattr(ref, attr) for ref in refs]
            # 获取结果对象的属性值
            res_attr = getattr(res, attr)
            # 断言检查属性值的近似相等性
            assert_allclose(res_attr.ravel(), ref_attr)
            # 断言检查属性值的形状是否正确
            assert_equal(res_attr.shape, shape)

        # 断言检查 success 属性的数据类型是否为布尔型
        assert np.issubdtype(res.success.dtype, np.bool_)
        if shape:
            # 断言检查 success 属性的中间元素是否全部为 True
            assert np.all(res.success[1:-1])
        # 断言检查 status 属性的数据类型是否为整数型
        assert np.issubdtype(res.status.dtype, np.integer)
        # 断言检查 nfev 属性的数据类型是否为整数型
        assert np.issubdtype(res.nfev.dtype, np.integer)
        # 断言检查 nit 属性的数据类型是否为整数型
        assert np.issubdtype(res.nit.dtype, np.integer)
        # 断言检查 nit 属性的最大值是否等于函数计算次数减 3
        assert_equal(np.max(res.nit), f.count - 3)
        # 检查有效区间是否合法
        self.assert_valid_bracket(res)
        # 断言检查 fl 属性的近似相等性
        assert_allclose(res.fl, f(res.xl, *args))
        # 断言检查 fm 属性的近似相等性
        assert_allclose(res.fm, f(res.xm, *args))
        # 断言检查 fr 属性的近似相等性
        assert_allclose(res.fr, f(res.xr, *args))
    def test_special_cases(self):
        # Test edge cases and other special cases.

        # Test that integers are not passed to `f`
        # (otherwise this would overflow)
        # 定义函数 f，检查传入的参数 x 是否是浮点数类型，避免整数溢出
        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 98 - 1

        # 调用 _bracket_minimum 函数，传入参数 f、-7 以及 xr0=5，获取结果
        result = _bracket_minimum(f, -7, xr0=5)
        # 断言结果对象的 success 属性为真
        assert result.success

        # Test maxiter = 0. Should do nothing to bracket.
        # 定义函数 f，计算 x 的平方减去 10
        def f(x):
            return x**2 - 10

        # 初始化 xl0、xm0、xr0 的值为 -3、-1、2
        xl0, xm0, xr0 = -3, -1, 2
        # 调用 _bracket_minimum 函数，传入参数 f、xm0，以及 xl0、xr0 和 maxiter=0
        result = _bracket_minimum(f, xm0, xl0=xl0, xr0=xr0, maxiter=0)
        # 断言 result 对象的 xl、xm、xr 属性与 [xl0, xm0, xr0] 相等
        assert_equal([result.xl, result.xm, result.xr], [xl0, xm0, xr0])

        # Test scalar `args` (not in tuple)
        # 定义函数 f，接受参数 x 和常数 c，计算 c*x 的平方减去 1
        def f(x, c):
            return c*x**2 - 1

        # 调用 _bracket_minimum 函数，传入参数 f、-1 以及 args=3
        result = _bracket_minimum(f, -1, args=3)
        # 断言 result 对象的 success 属性为真
        assert result.success
        # 断言 result 对象的 fl 属性与 f(result.xl, 3) 的近似值相等
        assert_allclose(result.fl, f(result.xl, 3))

        # Initial bracket is valid.
        # 获取 self.init_f() 返回的函数对象 f
        f = self.init_f()
        # 初始化 xl0、xm0、xr0 的值为 [-1.0, -0.2, 1.0]
        xl0, xm0, xr0 = [-1.0, -0.2, 1.0]
        # 初始化 args 为 (0, 0)
        args = (0, 0)
        # 调用 _bracket_minimum 函数，传入参数 f、xm0，以及 xl0、xr0 和 args=args
        result = _bracket_minimum(f, xm0, xl0=xl0, xr0=xr0, args=args)
        # 断言函数 f 的 count 属性为 3
        assert f.count == 3

        # 断言 result 对象的 xl、xm、xr 属性与 [xl0, xm0, xr0] 相等
        assert_equal(
            [result.xl, result.xm, result.xr],
            [xl0, xm0, xr0],
        )
        # 断言 result 对象的 fl、fm、fr 属性与 [f(xl0, *args), f(xm0, *args), f(xr0, *args)] 相等
        assert_equal(
            [result.fl, result.fm, result.fr],
            [f(xl0, *args), f(xm0, *args), f(xr0, *args)],
        )

    def test_gh_20562_left(self):
        # Regression test for https://github.com/scipy/scipy/issues/20562
        # minimum of f in [xmin, xmax] is at xmin.
        # 定义函数 f，计算以 xmin 和 xmax 为参数的对数的差值乘以 x 的倒数的负数
        xmin, xmax = 0.21933608, 1.39713606

        def f(x):
            log_a, log_b = np.log([xmin, xmax])
            return -((log_b - log_a)*x)**-1

        # 调用 _bracket_minimum 函数，传入参数 f、0.5535723499480897，以及 xmin=xmin、xmax=xmax
        result = _bracket_minimum(f, 0.5535723499480897, xmin=xmin, xmax=xmax)
        # 断言 xmin 等于 result 对象的 xl 属性
        assert xmin == result.xl

    def test_gh_20562_right(self):
        # Regression test for https://github.com/scipy/scipy/issues/20562
        # minimum of f in [xmin, xmax] is at xmax.
        # 定义函数 f，计算以 -xmax 和 -xmin 为参数的对数的差值乘以 x 的倒数
        xmin, xmax = -1.39713606, -0.21933608,

        def f(x):
            log_a, log_b = np.log([-xmax, -xmin])
            return ((log_b - log_a)*x)**-1

        # 调用 _bracket_minimum 函数，传入参数 f、-0.5535723499480897，以及 xmin=xmin、xmax=xmax
        result = _bracket_minimum(f, -0.5535723499480897, xmin=xmin, xmax=xmax)
        # 断言 xmax 等于 result 对象的 xr 属性
        assert xmax == result.xr
```