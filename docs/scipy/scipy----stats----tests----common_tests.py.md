# `D:\src\scipysrc\scipy\scipy\stats\tests\common_tests.py`

```
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle

# 导入numpy库，并使用别名np
import numpy as np

# 导入numpy.testing模块，并使用别名npt，用于进行数值测试和断言
import numpy.testing as npt

# 从numpy.testing模块导入assert_allclose和assert_equal函数，用于数值近似比较和相等性断言
from numpy.testing import assert_allclose, assert_equal

# 从pytest库导入raises函数，并使用别名assert_raises，用于检测异常的断言
from pytest import raises as assert_raises

# 导入numpy.ma.testutils模块，并使用别名ma_npt，用于掩码数组的数值测试
import numpy.ma.testutils as ma_npt

# 从scipy._lib._util模块中导入getfullargspec_no_self函数，并使用别名_getfullargspec，用于获取函数的参数规范
from scipy._lib._util import getfullargspec_no_self as _getfullargspec

# 从scipy._lib._array_api模块中导入xp_assert_equal函数，用于跨平台的数组相等性断言
from scipy._lib._array_api import xp_assert_equal

# 从scipy库导入stats模块，用于统计函数和分布
from scipy import stats


# 定义函数check_named_results，用于检查具有命名结果的对象列表
def check_named_results(res, attributes, ma=False, xp=None):
    # 遍历attributes列表中的每个属性
    for i, attr in enumerate(attributes):
        # 如果ma标志为True，则使用ma_npt.assert_equal函数进行断言
        if ma:
            ma_npt.assert_equal(res[i], getattr(res, attr))
        # 如果xp不为None，则使用xp_assert_equal函数进行断言
        elif xp is not None:
            xp_assert_equal(res[i], getattr(res, attr))
        # 否则使用npt.assert_equal函数进行断言
        else:
            npt.assert_equal(res[i], getattr(res, attr))


# 定义函数check_normalization，用于检查分布函数的归一化
def check_normalization(distfn, args, distname):
    # 计算分布函数的零阶矩，期望为1.0
    norm_moment = distfn.moment(0, *args)
    npt.assert_allclose(norm_moment, 1.0)

    # 根据distname设置atol和rtol的值
    if distname == "rv_histogram_instance":
        atol, rtol = 1e-5, 0
    else:
        atol, rtol = 1e-7, 1e-7

    # 计算分布函数的期望，期望为1.0
    normalization_expect = distfn.expect(lambda x: 1, args=args)
    npt.assert_allclose(normalization_expect, 1.0, atol=atol, rtol=rtol,
                        err_msg=distname, verbose=True)

    # 获取分布函数的支持区间，并计算其累积分布函数的值，期望为1.0
    _a, _b = distfn.support(*args)
    normalization_cdf = distfn.cdf(_b, *args)
    npt.assert_allclose(normalization_cdf, 1.0)


# 定义函数check_moment，用于检查分布函数的矩
def check_moment(distfn, arg, m, v, msg):
    # 计算分布函数的一阶矩和二阶矩
    m1 = distfn.moment(1, *arg)
    m2 = distfn.moment(2, *arg)

    # 如果m不是无穷大，则断言一阶矩近似等于m
    if not np.isinf(m):
        npt.assert_almost_equal(m1, m, decimal=10,
                                err_msg=msg + ' - 1st moment')
    else:  # 如果m是无穷大，则断言m1也是无穷大
        npt.assert_(np.isinf(m1),
                    msg + ' - 1st moment -infinite, m1=%s' % str(m1))

    # 如果v不是无穷大，则断言二阶中心矩减去一阶矩的平方近似等于v
    if not np.isinf(v):
        npt.assert_almost_equal(m2 - m1 * m1, v, decimal=10,
                                err_msg=msg + ' - 2ndt moment')
    else:  # 如果v是无穷大，则断言m2也是无穷大
        npt.assert_(np.isinf(m2), msg + ' - 2nd moment -infinite, {m2=}')


# 定义函数check_mean_expect，用于检查分布函数的期望
def check_mean_expect(distfn, arg, m, msg):
    # 如果m是有限的，则计算分布函数的期望，并断言其近似等于m
    if np.isfinite(m):
        m1 = distfn.expect(lambda x: x, arg)
        npt.assert_almost_equal(m1, m, decimal=5,
                                err_msg=msg + ' - 1st moment (expect)')


# 定义函数check_var_expect，用于检查分布函数的方差期望
def check_var_expect(distfn, arg, m, v, msg):
    # 对于一些分布函数，使用更宽松的公差
    dist_looser_tolerances = {"rv_histogram_instance" , "ksone"}
    kwargs = {'rtol': 5e-6} if msg in dist_looser_tolerances else {}
    
    # 如果v是有限的，则计算分布函数的期望，并断言其近似等于v + m的平方
    if np.isfinite(v):
        m2 = distfn.expect(lambda x: x*x, arg)
        npt.assert_allclose(m2, v + m*m, **kwargs)


# 定义函数check_skew_expect，用于检查分布函数的偏度期望
def check_skew_expect(distfn, arg, m, v, s, msg):
    # 如果s是有限的，则计算分布函数的偏度期望，并断言其近似等于s乘以v的1.5次方
    if np.isfinite(s):
        m3e = distfn.expect(lambda x: np.power(x-m, 3), arg)
        npt.assert_almost_equal(m3e, s * np.power(v, 1.5),
                                decimal=5, err_msg=msg + ' - skew')
    else:
        # 如果s是无限大，则断言s是NaN
        npt.assert_(np.isnan(s))


# 定义函数check_kurt_expect，用于检查分布函数的峰度期望
def check_kurt_expect(distfn, arg, m, v, k, msg):
    # 如果 k 是有限的（不是无穷大或 NaN），则执行以下代码块
    if np.isfinite(k):
        # 计算分布函数 distfn 中使用 lambda 函数计算的期望，lambda 函数是计算 (x-m)^4 的值
        m4e = distfn.expect(lambda x: np.power(x-m, 4), arg)
        # 使用 assert_allclose 函数验证 m4e 与 (k + 3.) * np.power(v, 2) 的近似性
        npt.assert_allclose(m4e, (k + 3.) * np.power(v, 2),
                            atol=1e-5, rtol=1e-5,
                            err_msg=msg + ' - kurtosis')
    # 如果 k 不是正无穷（np.isposinf(k) 返回 False），则执行以下代码块
    elif not np.isposinf(k):
        # 断言 k 是 NaN（非有限的）
        npt.assert_(np.isnan(k))
# 检查分布的_munp方法是否被重写，若是，则测试更高的矩（moment）。
# （在 gh-18634 之前，一些分布在矩 5 及更高时存在问题。）
def check_munp_expect(dist, args, msg):
    if dist._munp.__func__ != stats.rv_continuous._munp:
        # 调用分布的moment方法计算第5阶矩，预期不会引发错误
        res = dist.moment(5, *args)
        # 使用expect方法计算期望，函数为 x^5，限定在无穷大范围内
        ref = dist.expect(lambda x: x ** 5, args, lb=-np.inf, ub=np.inf)
        # 如果res不是有限数，可能是有效的；自动化测试无法确定
        if not np.isfinite(res):
            return
        # 使用assert_allclose检查res和ref的接近程度，设定宽松的公差
        assert_allclose(res, ref, atol=1e-10, rtol=1e-4, err_msg=msg + ' - higher moment / _munp')


# 检查分布的熵是否为NaN，预期不是NaN
def check_entropy(distfn, arg, msg):
    ent = distfn.entropy(*arg)
    npt.assert_(not np.isnan(ent), msg + 'test Entropy is nan')


# 检查分布的私有_entropy方法与其超类中的_entropy方法的一致性
def check_private_entropy(distfn, args, superclass):
    npt.assert_allclose(distfn._entropy(*args),
                        superclass._entropy(distfn, *args))


# 检查分布的熵在不同的缩放(scale)下的一致性
def check_entropy_vect_scale(distfn, arg):
    # 检查二维情况下的缩放
    sc = np.asarray([[1, 2], [3, 4]])
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc.ravel()]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)

    # 检查无效值和类型转换
    sc = [1, 2, -3]
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)


# 检查分布在边界支持值处的行为是否正确
def check_edge_support(distfn, args):
    x = distfn.support(*args)
    if isinstance(distfn, stats.rv_discrete):
        x = x[0]-1, x[1]

    # 检查累积分布函数在支持上的取值
    npt.assert_equal(distfn.cdf(x, *args), [0.0, 1.0])
    # 检查生存函数在支持上的取值
    npt.assert_equal(distfn.sf(x, *args), [1.0, 0.0])

    if distfn.name not in ('skellam', 'dlaplace'):
        # 当a=-inf时，logcdf生成警告，检查其取值
        npt.assert_equal(distfn.logcdf(x, *args), [-np.inf, 0.0])
        npt.assert_equal(distfn.logsf(x, *args), [0.0, -np.inf])

    # 检查分位点函数和逆生存函数在0和1处的取值
    npt.assert_equal(distfn.ppf([0.0, 1.0], *args), x)
    npt.assert_equal(distfn.isf([0.0, 1.0], *args), x[::-1])

    # 检查分位点函数和逆生存函数在超出界限值时的行为
    npt.assert_(np.isnan(distfn.isf([-1, 2], *args)).all())
    npt.assert_(np.isnan(distfn.ppf([-1, 2], *args)).all())


# 检查使用命名参数调用分布函数的一致性和正确性
def check_named_args(distfn, x, shape_args, defaults, meths):
    ## Check calling w/ named arguments.

    # 检查参数的形状一致性，参数数量和_parse签名
    signature = _getfullargspec(distfn._parse_args)
    npt.assert_(signature.varargs is None)
    npt.assert_(signature.varkw is None)
    npt.assert_(not signature.kwonlyargs)
    npt.assert_(list(signature.defaults) == list(defaults))

    # 根据分布的shapes属性，检查形状参数名的一致性
    shape_argnames = signature.args[:-len(defaults)]  # a, b, loc=0, scale=1
    if distfn.shapes:
        shapes_ = distfn.shapes.replace(',', ' ').split()
    else:
        shapes_ = ''
    # 确保 shapes_ 的长度与分布函数的参数数量相同
    npt.assert_(len(shapes_) == distfn.numargs)
    # 确保 shapes_ 的长度与 shape_argnames 的长度相同
    npt.assert_(len(shapes_) == len(shape_argnames))

    # 检查使用命名参数调用的情况
    shape_args = list(shape_args)

    # 对于每个方法 meths 中的方法，使用 meth(x, *shape_args) 计算 vals 列表
    vals = [meth(x, *shape_args) for meth in meths]
    # 确保 vals 中的所有值都是有限的
    npt.assert_(np.all(np.isfinite(vals)))

    # 初始化 names, a, k，分别赋值为 shape_argnames 的拷贝，shape_args 的拷贝，空字典
    names, a, k = shape_argnames[:], shape_args[:], {}

    # 当 names 非空时循环
    while names:
        # 将 names 的最后一个元素作为键，a 的最后一个元素作为值加入字典 k 中
        k.update({names.pop(): a.pop()})
        # 使用当前的参数 a, k 调用 meths 中的方法，计算 v 列表
        v = [meth(x, *a, **k) for meth in meths]
        # 确保 vals 和 v 的数组内容相等
        npt.assert_array_equal(vals, v)
        # 如果 k 中没有键为 'n'，则进行以下断言
        if 'n' not in k.keys():
            # 'n' 是 moment() 方法的第一个参数，所以不能作为命名参数使用
            npt.assert_equal(distfn.moment(1, *a, **k),
                             distfn.moment(1, *shape_args))

    # 更新 k 字典，加入一个未知的参数 'kaboom'
    k.update({'kaboom': 42})
    # 确保调用 distfn.cdf(x, **k) 会引发 TypeError 异常
    assert_raises(TypeError, distfn.cdf, x, **k)
# 检查分布函数实例的 random_state 属性
def check_random_state_property(distfn, args):
    # 保存原始的 distfn.random_state，以便稍后恢复
    rndm = distfn.random_state

    # 使用全局种子1234重新设置随机数生成器状态，确保基准测试使用全局状态
    np.random.seed(1234)
    # 将 distfn.random_state 设置为 None，使用全局状态进行随机变量生成
    distfn.random_state = None
    r0 = distfn.rvs(*args, size=8)

    # 使用实例级别的随机数生成器状态1234生成随机变量，与全局状态下的结果进行比较
    distfn.random_state = 1234
    r1 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r1)

    # 使用自定义的 np.random.RandomState(1234) 实例生成随机数，与全局状态下的结果进行比较
    distfn.random_state = np.random.RandomState(1234)
    r2 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r2)

    # 检查是否可以使用 np.random.Generator（numpy >= 1.17）
    if hasattr(np.random, 'default_rng'):
        # 获取一个 np.random.Generator 对象
        rng = np.random.default_rng(1234)
        distfn.rvs(*args, size=1, random_state=rng)

    # 可以为单独的 .rvs 调用覆盖实例级别的 random_state
    distfn.random_state = 2
    orig_state = distfn.random_state.get_state()

    # 使用 np.random.RandomState(1234) 作为 random_state 调用 .rvs，与全局状态下的结果进行比较
    r3 = distfn.rvs(*args, size=8, random_state=np.random.RandomState(1234))
    npt.assert_equal(r0, r3)

    # 确保这不会改变实例级别的 random_state
    npt.assert_equal(distfn.random_state.get_state(), orig_state)

    # 最后，恢复原始的 random_state
    distfn.random_state = rndm


# 检查分布函数的方法返回值的数据类型
def check_meth_dtype(distfn, arg, meths):
    q0 = [0.25, 0.5, 0.75]
    x0 = distfn.ppf(q0, *arg)
    # 将 x0 转换为指定的数据类型列表
    x_cast = [x0.astype(tp) for tp in (np_long, np.float16, np.float32,
                                       np.float64)]

    for x in x_cast:
        # 检查是否已截断值，排除已截断的值
        distfn._argcheck(*arg)
        x = x[(distfn.a < x) & (x < distfn.b)]
        for meth in meths:
            # 对每种方法 meth 应用到 x 上，检查返回值的数据类型是否为 np.float64
            val = meth(x, *arg)
            npt.assert_(val.dtype == np.float64)


# 检查分布函数的 percent point function (ppf) 方法返回值的数据类型
def check_ppf_dtype(distfn, arg):
    q0 = np.asarray([0.25, 0.5, 0.75])
    # 将 q0 转换为指定的数据类型列表
    q_cast = [q0.astype(tp) for tp in (np.float16, np.float32, np.float64)]
    for q in q_cast:
        for meth in [distfn.ppf, distfn.isf]:
            # 对每种方法 meth 应用到 q 上，检查返回值的数据类型是否为 np.float64
            val = meth(q, *arg)
            npt.assert_(val.dtype == np.float64)


# 检查分布函数允许复数参数时的导数
def check_cmplx_deriv(distfn, arg):
    # 定义一个函数 deriv，计算复数参数下的导数
    def deriv(f, x, *arg):
        x = np.asarray(x)
        h = 1e-10
        return (f(x + h*1j, *arg)/h).imag

    # 计算 ppf 方法在不同数据类型下的返回值，并进行类型转换
    x0 = distfn.ppf([0.25, 0.51, 0.75], *arg)
    x_cast = [x0.astype(tp) for tp in (np_long, np.float16, np.float32,
                                       np.float64)]
    # 对于每个变量 x 在 x_cast 中进行迭代处理
    for x in x_cast:
        # 检查参数是否符合分布函数的要求
        distfn._argcheck(*arg)
        # 根据分布函数的定义域条件，筛选出符合条件的 x 值
        x = x[(distfn.a < x) & (x < distfn.b)]

        # 计算概率密度函数（PDF）、累积分布函数（CDF）、生存函数（SF）在 x 处的值
        pdf, cdf, sf = distfn.pdf(x, *arg), distfn.cdf(x, *arg), distfn.sf(x, *arg)
        # 断言：利用数值方法验证 PDF 的导数值是否接近于 PDF 本身，相对误差不超过 1e-5
        assert_allclose(deriv(distfn.cdf, x, *arg), pdf, rtol=1e-5)
        # 断言：利用数值方法验证对数CDF的导数值是否接近于 PDF/CDF，相对误差不超过 1e-5
        assert_allclose(deriv(distfn.logcdf, x, *arg), pdf/cdf, rtol=1e-5)

        # 断言：利用数值方法验证 SF 的导数值是否接近于 -PDF，相对误差不超过 1e-5
        assert_allclose(deriv(distfn.sf, x, *arg), -pdf, rtol=1e-5)
        # 断言：利用数值方法验证对数SF的导数值是否接近于 -PDF/SF，相对误差不超过 1e-5
        assert_allclose(deriv(distfn.logsf, x, *arg), -pdf/sf, rtol=1e-5)

        # 断言：利用数值方法验证对数PDF的导数值是否接近于 PDF 导数除以 PDF 值本身，相对误差不超过 1e-5
        assert_allclose(deriv(distfn.logpdf, x, *arg),
                        deriv(distfn.pdf, x, *arg) / distfn.pdf(x, *arg),
                        rtol=1e-5)
# 检查分布实例是否能够正确进行序列化和反序列化
# 特别关注 random_state 属性

# 保存 random_state（稍后恢复）
rndm = distfn.random_state

# 检查未冻结状态下的分布
distfn.random_state = 1234
distfn.rvs(*args, size=8)
s = pickle.dumps(distfn)
r0 = distfn.rvs(*args, size=8)

# 反序列化并再次生成随机数，确保结果一致
unpickled = pickle.loads(s)
r1 = unpickled.rvs(*args, size=8)
npt.assert_equal(r0, r1)

# 进行一些方法的简单测试
medians = [distfn.ppf(0.5, *args), unpickled.ppf(0.5, *args)]
npt.assert_equal(medians[0], medians[1])
npt.assert_equal(distfn.cdf(medians[0], *args),
                 unpickled.cdf(medians[1], *args))

# 检查冻结状态下通过 rvs 方法的序列化和反序列化
frozen_dist = distfn(*args)
pkl = pickle.dumps(frozen_dist)
unpickled = pickle.loads(pkl)

r0 = frozen_dist.rvs(size=8)
r1 = unpickled.rvs(size=8)
npt.assert_equal(r0, r1)

# 检查 .fit 方法的序列化和反序列化
if hasattr(distfn, "fit"):
    fit_function = distfn.fit
    pickled_fit_function = pickle.dumps(fit_function)
    unpickled_fit_function = pickle.loads(pickled_fit_function)
    assert fit_function.__name__ == unpickled_fit_function.__name__ == "fit"

# 恢复之前保存的 random_state
distfn.random_state = rndm
```