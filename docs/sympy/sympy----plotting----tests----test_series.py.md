# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_series.py`

```
# 导入必要的符号计算库和函数
from sympy import (
    latex, exp, symbols, I, pi, sin, cos, tan, log, sqrt,
    re, im, arg, frac, Sum, S, Abs, lambdify,
    Function, dsolve, Eq, floor, Tuple
)
# 导入外部模块导入函数
from sympy.external import import_module
# 导入符号绘图相关的类和函数
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
    ImplicitSeries, _set_discretization_points, List2DSeries
)
# 导入测试相关的函数和装饰器
from sympy.testing.pytest import raises, warns, XFAIL, skip, ignore_warnings

# 导入numpy模块并重命名为np
np = import_module('numpy')

# 定义测试函数test_adaptive
def test_adaptive():
    # 如果numpy未安装，则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量x和y
    x, y = symbols("x, y")

    # 创建自适应线性序列对象s1，s2，s3，并验证其长度
    s1 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True,
        depth=2)
    x1, _ = s1.get_data()
    s2 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True,
        depth=5)
    x2, _ = s2.get_data()
    s3 = LineOver1DRangeSeries(sin(x), (x, -10, 10), "", adaptive=True)
    x3, _ = s3.get_data()
    assert len(x1) < len(x2) < len(x3)

    # 创建自适应参数化二维线性序列对象s1，s2，s3，并验证其长度
    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True, depth=2)
    x1, _, _, = s1.get_data()
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True, depth=5)
    x2, _, _ = s2.get_data()
    s3 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=True)
    x3, _, _ = s3.get_data()
    assert len(x1) < len(x2) < len(x3)

# 定义测试函数test_detect_poles
def test_detect_poles():
    # 如果numpy未安装，则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量x和u
    x, u = symbols("x, u")

    # 创建禁用极点检测的线性序列对象s1，s2，s3，s4，并验证其数据
    s1 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=False)
    xx1, yy1 = s1.get_data()
    s2 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=True, eps=0.01)
    xx2, yy2 = s2.get_data()
    # eps值太小，未检测到任何极点
    s3 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles=True, eps=1e-06)
    xx3, yy3 = s3.get_data()
    s4 = LineOver1DRangeSeries(tan(x), (x, -pi, pi),
        adaptive=False, n=1000, detect_poles="symbolic")
    xx4, yy4 = s4.get_data()

    # 验证数据相等性和极点的正确性
    assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3) and np.allclose(xx1, xx4)
    assert not np.any(np.isnan(yy1))
    assert not np.any(np.isnan(yy3))
    assert np.any(np.isnan(yy2))
    assert np.any(np.isnan(yy4))
    assert len(s2.poles_locations) == len(s3.poles_locations) == 0
    assert len(s4.poles_locations) == 2
    assert np.allclose(np.abs(s4.poles_locations), np.pi / 2)
    # 使用 `warns` 上下文管理器捕获特定的用户警告消息
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
            test_stacklevel=False,
        ):
        # 创建 `LineOver1DRangeSeries` 对象 `s1`，设置不同的参数组合
        s1 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles=False)
        # 创建 `LineOver1DRangeSeries` 对象 `s2`，设置不同的参数组合
        s2 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles=True, eps=0.05)
        # 创建 `LineOver1DRangeSeries` 对象 `s3`，设置不同的参数组合
        s3 = LineOver1DRangeSeries(frac(x), (x, -10, 10),
            adaptive=False, n=1000, detect_poles="symbolic")
        # 获取 `s1` 对象的数据
        xx1, yy1 = s1.get_data()
        # 获取 `s2` 对象的数据
        xx2, yy2 = s2.get_data()
        # 获取 `s3` 对象的数据
        xx3, yy3 = s3.get_data()
        # 断言 `xx1`, `xx2`, `xx3` 数据相似
        assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3)
        # 断言 `yy1` 没有 NaN 值
        assert not np.any(np.isnan(yy1))
        # 断言 `yy2` 或 `yy3` 存在 NaN 值
        assert np.any(np.isnan(yy2)) and np.any(np.isnan(yy2))
        # 断言 `yy1` 和 `yy2` 在考虑 NaN 值的情况下不相等
        assert not np.allclose(yy1, yy2, equal_nan=True)
        # 断言 `s3` 中的极点位置数量为 21
        assert len(s3.poles_locations) == 21

    # 创建 `LineOver1DRangeSeries` 对象 `s1`，使用不同的参数组合
    s1 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=False)
    # 获取 `s1` 对象的数据
    xx1, yy1 = s1.get_data()
    # 创建 `LineOver1DRangeSeries` 对象 `s2`，使用不同的参数组合
    s2 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=True, eps=0.01)
    # 获取 `s2` 对象的数据
    xx2, yy2 = s2.get_data()
    # 创建 `LineOver1DRangeSeries` 对象 `s3`，使用不同的参数组合
    s3 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles=True, eps=1e-06)
    # 获取 `s3` 对象的数据
    xx3, yy3 = s3.get_data()
    # 创建 `LineOver1DRangeSeries` 对象 `s4`，使用不同的参数组合
    s4 = LineOver1DRangeSeries(tan(u * x), (x, -pi, pi), params={u: 1},
        adaptive=False, n=1000, detect_poles="symbolic")
    # 获取 `s4` 对象的数据
    xx4, yy4 = s4.get_data()

    # 断言 `xx1`, `xx2`, `xx3`, `xx4` 数据相似
    assert np.allclose(xx1, xx2) and np.allclose(xx1, xx3) and np.allclose(xx1, xx4)
    # 断言 `yy1` 和 `yy3` 没有 NaN 值
    assert not np.any(np.isnan(yy1))
    assert not np.any(np.isnan(yy3))
    # 断言 `yy2` 或 `yy4` 存在 NaN 值
    assert np.any(np.isnan(yy2))
    assert np.any(np.isnan(yy4))
    # 断言 `s2` 和 `s3` 中的极点位置数量为 0
    assert len(s2.poles_locations) == len(s3.poles_locations) == 0
    # 断言 `s4` 中的极点位置数量为 2，并且它们的绝对值近似为 pi / 2
    assert len(s4.poles_locations) == 2
    assert np.allclose(np.abs(s4.poles_locations), np.pi / 2)

    # 使用 `warns` 上下文管理器捕获特定的用户警告消息
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
            test_stacklevel=False,
        ):
        # 声明符号变量 `u` 和 `v`，限定其为实数
        u, v = symbols("u, v", real=True)
        # 定义复数幂 `f`
        n = S(1) / 3
        f = (u + I * v)**n
        # 提取 `f` 的实部和虚部
        r, i = re(f), im(f)
        # 创建 `Parametric2DLineSeries` 对象 `s1`，使用不同的参数组合
        s1 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2), (v, -2, 2),
            adaptive=False, n=1000, detect_poles=False)
        # 创建 `Parametric2DLineSeries` 对象 `s2`，使用不同的参数组合
        s2 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2), (v, -2, 2),
            adaptive=False, n=1000, detect_poles=True)
    
    # 忽略 `RuntimeWarning` 类型的警告消息
    with ignore_warnings(RuntimeWarning):
        # 获取 `s1` 对象的数据
        xx1, yy1, pp1 = s1.get_data()
        # 断言 `yy1` 没有 NaN 值
        assert not np.isnan(yy1).any()
        # 获取 `s2` 对象的数据
        xx2, yy2, pp2 = s2.get_data()
        # 断言 `yy2` 中存在 NaN 值
        assert np.isnan(yy2).any()
    # 使用 warns 上下文管理器捕获 UserWarning 异常，并匹配特定的警告信息
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some of",
            test_stacklevel=False,
        ):
        # 计算复杂表达式 (x * u + x * I * v)**n
        f = (x * u + x * I * v)**n
        # 提取复数表达式 f 的实部和虚部
        r, i = re(f), im(f)
        # 创建 Parametric2DLineSeries 对象 s1，表示二维参数化曲线
        s1 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2),
            (v, -2, 2), params={x: 1},
            adaptive=False, n1=1000, detect_poles=False)
        # 创建 Parametric2DLineSeries 对象 s2，表示二维参数化曲线
        s2 = Parametric2DLineSeries(r.subs(u, -2), i.subs(u, -2),
            (v, -2, 2), params={x: 1},
            adaptive=False, n1=1000, detect_poles=True)
    
    # 使用 ignore_warnings 上下文管理器忽略 RuntimeWarning 异常
    with ignore_warnings(RuntimeWarning):
        # 获取 s1 曲线的数据 xx1, yy1, pp1
        xx1, yy1, pp1 = s1.get_data()
        # 断言 yy1 中不含 NaN 值
        assert not np.isnan(yy1).any()
        # 获取 s2 曲线的数据 xx2, yy2, pp2
        xx2, yy2, pp2 = s2.get_data()
        # 断言 yy2 中含有 NaN 值
        assert np.isnan(yy2).any()
def test_number_discretization_points():
    # 验证不同设置离散化点数的方式是否一致。
    if not np:
        skip("numpy not installed.")

    # 定义符号变量 x, y, z
    x, y, z = symbols("x:z")

    # 对于一维线性系列，验证离散化点数的设置
    for pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
        Parametric3DLineSeries]:
        # 设置离散化点数为 10，生成关键字参数 kw1
        kw1 = _set_discretization_points({"n": 10}, pt)
        # 设置离散化点数为 [10, 20, 30]，生成关键字参数 kw2
        kw2 = _set_discretization_points({"n": [10, 20, 30]}, pt)
        # 设置离散化点数为 10，生成关键字参数 kw3
        kw3 = _set_discretization_points({"n1": 10}, pt)
        # 确保所有关键字参数中都有 "n1"，且其值为 10
        assert all(("n1" in kw) and kw["n1"] == 10 for kw in [kw1, kw2, kw3])

    # 对于二维表面系列，轮廓系列，二维参数化表面系列，隐式系列，验证离散化点数的设置
    for pt in [SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries,
        ImplicitSeries]:
        # 设置离散化点数为 10，生成关键字参数 kw1
        kw1 = _set_discretization_points({"n": 10}, pt)
        # 设置离散化点数为 [10, 20, 30]，生成关键字参数 kw2
        kw2 = _set_discretization_points({"n": [10, 20, 30]}, pt)
        # 设置离散化点数为 {"n1": 10, "n2": 20}，生成关键字参数 kw3
        kw3 = _set_discretization_points({"n1": 10, "n2": 20}, pt)
        # 确保 kw1 的 "n1" 和 "n2" 值都为 10
        assert kw1["n1"] == kw1["n2"] == 10
        # 确保所有关键字参数中 "n1" 的值为 10，"n2" 的值为 20
        assert all((kw["n1"] == 10) and (kw["n2"] == 20) for kw in [kw2, kw3])

    # 验证线性相关系列能够处理大量浮点数离散化点
    LineOver1DRangeSeries(cos(x), (x, -5, 5), adaptive=False, n=1e04).get_data()


def test_list2dseries():
    if not np:
        skip("numpy not installed.")

    # 在区间 [-3, 3] 上生成 10 个点的等间距数列 xx
    xx = np.linspace(-3, 3, 10)
    # 计算 xx 对应的 cos 值 yy1
    yy1 = np.cos(xx)
    # 在区间 [-3, 3] 上生成 20 个点的等间距数列 yy2
    yy2 = np.linspace(-3, 3, 20)

    # 当元素个数相同时，正常生成 List2DSeries 对象
    s = List2DSeries(xx, yy1)
    assert not s.is_parametric
    # 当元素个数不同时，抛出 ValueError
    raises(ValueError, lambda: List2DSeries(xx, yy2))

    # 当没有颜色函数时，返回仅包含 x, y 组件的数据，并且 s 不是参数化的
    s = List2DSeries(xx, yy1)
    xxs, yys = s.get_data()
    assert np.allclose(xx, xxs)
    assert np.allclose(yy1, yys)
    assert not s.is_parametric


def test_interactive_vs_noninteractive():
    # 验证如果 *Series 类接收到 `params` 字典，则设置 is_interactive=True
    x, y, z, u, v = symbols("x, y, z, u, v")

    # 对于一维线性系列，验证 is_interactive 属性的设置
    s = LineOver1DRangeSeries(cos(x), (x, -5, 5))
    assert not s.is_interactive
    s = LineOver1DRangeSeries(u * cos(x), (x, -5, 5), params={u: 1})
    assert s.is_interactive

    # 对于二维参数化线性系列，验证 is_interactive 属性的设置
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5))
    assert not s.is_interactive
    s = Parametric2DLineSeries(u * cos(x), u * sin(x), (x, -5, 5),
        params={u: 1})
    assert s.is_interactive

    # 对于三维参数化线性系列，验证 is_interactive 属性的设置
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5))
    assert not s.is_interactive
    s = Parametric3DLineSeries(u * cos(x), u * sin(x), x, (x, -5, 5),
        params={u: 1})
    assert s.is_interactive

    # 对于二维表面系列，验证 is_interactive 属性的设置
    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -5, 5), (y, -5, 5))
    assert not s.is_interactive
    s = SurfaceOver2DRangeSeries(u * cos(x * y), (x, -5, 5), (y, -5, 5),
        params={u: 1})
    assert s.is_interactive

    # 对于轮廓系列，验证 is_interactive 属性的设置
    s = ContourSeries(cos(x * y), (x, -5, 5), (y, -5, 5))
    assert not s.is_interactive
    s = ContourSeries(u * cos(x * y), (x, -5, 5), (y, -5, 5),
        params={u: 1})
    assert s.is_interactive
    # 断言检查是否处于交互模式
    assert s.is_interactive
    
    # 创建 ParametricSurfaceSeries 对象，并设置其参数为 u * cos(v)，v * sin(u)，u + v
    # 参数范围为 u 从 -5 到 5，v 从 -5 到 5
    s = ParametricSurfaceSeries(u * cos(v), v * sin(u), u + v,
        (u, -5, 5), (v, -5, 5))
    
    # 再次断言检查是否不处于交互模式
    assert not s.is_interactive
    
    # 创建 ParametricSurfaceSeries 对象，并设置其参数为 u * cos(v * x)，v * sin(u)，u + v
    # 参数范围为 u 从 -5 到 5，v 从 -5 到 5，并额外指定参数 x 的值为 1
    s = ParametricSurfaceSeries(u * cos(v * x), v * sin(u), u + v,
        (u, -5, 5), (v, -5, 5), params={x: 1})
    
    # 再次断言检查是否处于交互模式
    assert s.is_interactive
# 验证数据序列在不同比例尺下的创建是否正确
def test_lin_log_scale():
    # 如果没有安装 numpy，则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量 x, y, z
    x, y, z = symbols("x, y, z")

    # 创建线性比例尺的一维范围数据序列，固定非自适应，50个数据点
    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50,
        xscale="linear")
    xx, _ = s.get_data()
    # 断言相邻数据点之间的间距相等
    assert np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    # 创建对数比例尺的一维范围数据序列，固定非自适应，50个数据点
    s = LineOver1DRangeSeries(x, (x, 1, 10), adaptive=False, n=50,
        xscale="log")
    xx, _ = s.get_data()
    # 断言相邻数据点之间的间距不相等
    assert not np.isclose(xx[1] - xx[0], xx[-1] - xx[-2])

    # 创建线性比例尺的二维参数曲线序列，固定非自适应，50个数据点
    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="linear")
    _, _, param = s.get_data()
    # 断言参数值的相邻间距相等
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    # 创建对数比例尺的二维参数曲线序列，固定非自适应，50个数据点
    s = Parametric2DLineSeries(
        cos(x), sin(x), (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="log")
    _, _, param = s.get_data()
    # 断言参数值的相邻间距不相等
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    # 创建线性比例尺的三维参数曲线序列，固定非自适应，50个数据点
    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="linear")
    _, _, _, param = s.get_data()
    # 断言参数值的相邻间距相等
    assert np.isclose(param[1] - param[0], param[-1] - param[-2])

    # 创建对数比例尺的三维参数曲线序列，固定非自适应，50个数据点
    s = Parametric3DLineSeries(
        cos(x), sin(x), x, (x, pi / 2, 1.5 * pi), adaptive=False, n=50,
        xscale="log")
    _, _, _, param = s.get_data()
    # 断言参数值的相邻间距不相等
    assert not np.isclose(param[1] - param[0], param[-1] - param[-2])

    # 创建线性比例尺的二维范围数据曲面序列，固定非自适应，10x10个数据点
    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, 1, 5), (y, 1, 5), n=10,
        xscale="linear", yscale="linear")
    xx, yy, _ = s.get_data()
    # 断言 x 和 y 方向上相邻数据点的间距相等
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    # 创建对数比例尺的二维范围数据曲面序列，固定非自适应，10x10个数据点
    s = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, 1, 5), (y, 1, 5), n=10,
        xscale="log", yscale="log")
    xx, yy, _ = s.get_data()
    # 断言 x 和 y 方向上相邻数据点的间距不相等
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert not np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    # 创建线性比例尺的隐式二维范围数据曲面序列，固定非自适应，10x10个数据点
    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0, (x, 1, 5), (y, 1, 5),
        n1=10, n2=10, xscale="linear", yscale="linear", adaptive=False)
    xx, yy, _, _ = s.get_data()
    # 断言 x 和 y 方向上相邻数据点的间距相等
    assert np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])

    # 创建对数比例尺的隐式二维范围数据曲面序列，固定非自适应，10x10个数据点
    s = ImplicitSeries(
        cos(x ** 2 + y ** 2) > 0, (x, 1, 5), (y, 1, 5),
        n=10, xscale="log", yscale="log", adaptive=False)
    xx, yy, _, _ = s.get_data()
    # 断言 x 和 y 方向上相邻数据点的间距不相等
    assert not np.isclose(xx[0, 1] - xx[0, 0], xx[0, -1] - xx[0, -2])
    assert not np.isclose(yy[1, 0] - yy[0, 0], yy[-1, 0] - yy[-2, 0])


def test_rendering_kw():
    # 验证每个数据序列是否暴露了 `rendering_kw` 属性
    if not np:
        skip("numpy not installed.")

    # 定义符号变量 u, v, x, y, z
    u, v, x, y, z = symbols("u, v, x:z")

    # 创建简单的二维列表数据序列
    s = List2DSeries([1, 2, 3], [4, 5, 6])
    # 断言序列对象的 `rendering_kw` 属性是一个字典
    assert isinstance(s.rendering_kw, dict)

    # 创建一维范围线性数据序列，仅指定 x 变量
    s = LineOver1DRangeSeries(1, (x, -5, 5))
    # 断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
    
    # 创建一个 Parametric2DLineSeries 对象 `s`，参数为 sin(x), cos(x)，x 的范围是 0 到 π
    s = Parametric2DLineSeries(sin(x), cos(x), (x, 0, pi))
    # 再次断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
    
    # 创建一个 Parametric3DLineSeries 对象 `s`，参数为 cos(x), sin(x), x 的范围是 0 到 2π
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2 * pi))
    # 再次断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
    
    # 创建一个 SurfaceOver2DRangeSeries 对象 `s`，参数为 x + y，x 范围是 -2 到 2，y 范围是 -3 到 3
    s = SurfaceOver2DRangeSeries(x + y, (x, -2, 2), (y, -3, 3))
    # 再次断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
    
    # 创建一个 ContourSeries 对象 `s`，参数为 x + y，x 范围是 -2 到 2，y 范围是 -3 到 3
    s = ContourSeries(x + y, (x, -2, 2), (y, -3, 3))
    # 再次断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
    
    # 创建一个 ParametricSurfaceSeries 对象 `s`，参数为 1, x, y，x 范围是 0 到 1，y 范围是 0 到 1
    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1))
    # 再次断言确保 `s.rendering_kw` 是一个字典类型
    assert isinstance(s.rendering_kw, dict)
def test_data_shape():
    # 验证当输入表达式为数字时，系列产生正确的数据形状

    if not np:
        # 如果没有安装 numpy，则跳过测试
        skip("numpy not installed.")

    u, x, y, z = symbols("u, x:z")

    # scalar expression: it should return a numpy ones array
    # 标量表达式：应返回一个 numpy 的全为 1 的数组
    s = LineOver1DRangeSeries(1, (x, -5, 5))
    xx, yy = s.get_data()
    assert len(xx) == len(yy)
    assert np.all(yy == 1)

    s = LineOver1DRangeSeries(1, (x, -5, 5), adaptive=False, n=10)
    xx, yy = s.get_data()
    assert len(xx) == len(yy) == 10
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi))
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric2DLineSeries(sin(x), 1, (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric2DLineSeries(1, sin(x), (x, 0, pi), adaptive=False)
    xx, yy, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = Parametric3DLineSeries(cos(x), sin(x), 1, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(zz == 1)

    s = Parametric3DLineSeries(cos(x), 1, x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(yy == 1)

    s = Parametric3DLineSeries(1, sin(x), x, (x, 0, 2 * pi))
    xx, yy, zz, param = s.get_data()
    assert (len(xx) == len(yy)) and (len(xx) == len(zz)) and (len(xx) == len(param))
    assert np.all(xx == 1)

    s = SurfaceOver2DRangeSeries(1, (x, -2, 2), (y, -3, 3))
    xx, yy, zz = s.get_data()
    assert (xx.shape == yy.shape) and (xx.shape == zz.shape)
    assert np.all(zz == 1)

    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(xx == 1)

    s = ParametricSurfaceSeries(1, 1, y, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(yy == 1)

    s = ParametricSurfaceSeries(x, 1, 1, (x, 0, 1), (y, 0, 1))
    xx, yy, zz, uu, vv = s.get_data()
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape
    assert np.all(zz == 1)


def test_only_integers():
    if not np:
        skip("numpy not installed.")

    x, y, u, v = symbols("x, y, u, v")

    # Create a LineOver1DRangeSeries with sine of x, range from -5.5 to 4.5, empty string title,
    # non-adaptive mode, only integers option set to True.
    s = LineOver1DRangeSeries(sin(x), (x, -5.5, 4.5), "",
        adaptive=False, only_integers=True)
    xx, _ = s.get_data()
    assert len(xx) == 10
    # 断言检查列表 xx 的第一个和最后一个元素是否分别为 -5 和 4
    assert xx[0] == -5 and xx[-1] == 4

    # 创建 Parametric2DLineSeries 对象，表示二维参数化线系列
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    # 获取该线系列的数据，返回元组 (x 数据, y 数据, 参数值列表)
    _, _, p = s.get_data()
    # 断言检查参数值列表 p 的长度是否为 7
    assert len(p) == 7
    # 断言检查参数值列表 p 的第一个和最后一个元素是否分别为 0 和 6
    assert p[0] == 0 and p[-1] == 6

    # 创建 Parametric3DLineSeries 对象，表示三维参数化线系列
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    # 获取该线系列的数据，返回元组 (x 数据, y 数据, z 数据, 参数值列表)
    _, _, _, p = s.get_data()
    # 断言检查参数值列表 p 的长度是否为 7
    assert len(p) == 7
    # 断言检查参数值列表 p 的第一个和最后一个元素是否分别为 0 和 6
    assert p[0] == 0 and p[-1] == 6

    # 创建 SurfaceOver2DRangeSeries 对象，表示二维范围上的表面系列
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -5.5, 5.5),
        (y, -3.5, 3.5), "",
        adaptive=False, only_integers=True)
    # 获取该表面系列的数据，返回元组 (x 数据, y 数据, z 数据)
    xx, yy, _ = s.get_data()
    # 断言检查 xx 和 yy 的形状是否为 (7, 11)
    assert xx.shape == yy.shape == (7, 11)
    # 断言检查 xx 的第一列是否接近于 -5 的线性空间数组
    assert np.allclose(xx[:, 0] - (-5) * np.ones(7), 0)
    # 断言检查 xx 的第一行是否接近于从 -5 到 5 的线性空间数组
    assert np.allclose(xx[0, :] - np.linspace(-5, 5, 11), 0)
    # 断言检查 yy 的第一列是否接近于 -3 的线性空间数组
    assert np.allclose(yy[:, 0] - np.linspace(-3, 3, 7), 0)
    # 断言检查 yy 的第一行是否接近于 -3 的线性空间数组
    assert np.allclose(yy[0, :] - (-3) * np.ones(11), 0)

    # 计算参数表达式中的 r 值
    r = 2 + sin(7 * u + 5 * v)
    # 定义三维参数表面系列的表达式
    expr = (
        r * cos(u) * sin(v),
        r * sin(u) * sin(v),
        r * cos(v)
    )
    # 创建 ParametricSurfaceSeries 对象，表示三维参数化表面系列
    s = ParametricSurfaceSeries(*expr, (u, 0, 2 * pi), (v, 0, pi), "",
        adaptive=False, only_integers=True)
    # 获取该表面系列的数据，返回元组 (x 数据, y 数据, z 数据, u 数据, v 数据)
    xx, yy, zz, uu, vv = s.get_data()
    # 断言检查 xx, yy, zz, uu, vv 的形状是否为 (4, 7)
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape == (4, 7)

    # only_integers 参数也适用于标量表达式
    # 创建 LineOver1DRangeSeries 对象，表示一维范围上的线系列
    s = LineOver1DRangeSeries(1, (x, -5.5, 4.5), "",
        adaptive=False, only_integers=True)
    # 获取该线系列的数据，返回元组 (x 数据, y 数据)
    xx, _ = s.get_data()
    # 断言检查 xx 的长度是否为 10
    assert len(xx) == 10
    # 断言检查 xx 的第一个和最后一个元素是否分别为 -5 和 4
    assert xx[0] == -5 and xx[-1] == 4

    # 创建 Parametric2DLineSeries 对象，表示二维参数化线系列
    s = Parametric2DLineSeries(cos(x), 1, (x, 0, 2 * pi), "",
        adaptive=False, only_integers=True)
    # 获取该线系列的数据，返回元组 (x 数据, y 数据, 参数值列表)
    _, _, p = s.get_data()
    # 断言检查参数值列表 p 的长度是否为 7
    assert len(p) == 7
    # 断言检查参数值列表 p 的第一个和最后一个元素是否分别为 0 和 6
    assert p[0] == 0 and p[-1] == 6

    # 创建 SurfaceOver2DRangeSeries 对象，表示二维范围上的表面系列
    s = SurfaceOver2DRangeSeries(1, (x, -5.5, 5.5), (y, -3.5, 3.5), "",
        adaptive=False, only_integers=True)
    # 获取该表面系列的数据，返回元组 (x 数据, y 数据, z 数据)
    xx, yy, _ = s.get_data()
    # 断言检查 xx 和 yy 的形状是否为 (7, 11)
    assert xx.shape == yy.shape == (7, 11)
    # 断言检查 xx 的第一列是否接近于 -5 的线性空间数组
    assert np.allclose(xx[:, 0] - (-5) * np.ones(7), 0)
    # 断言检查 xx 的第一行是否接近于从 -5 到 5 的线性空间数组
    assert np.allclose(xx[0, :] - np.linspace(-5, 5, 11), 0)
    # 断言检查 yy 的第一列是否接近于 -3 的线性空间数组
    assert np.allclose(yy[:, 0] - np.linspace(-3, 3, 7), 0)
    # 断言检查 yy 的第一行是否接近于 -3 的线性空间数组
    assert np.allclose(yy[0, :] - (-3) * np.ones(11), 0)

    # 计算参数表达式中的 r 值
    r = 2 + sin(7 * u + 5 * v)
    # 定义三维参数表面系列的表达式
    expr = (
        r * cos(u) * sin(v),
        1,
        r * cos(v)
    )
    # 创建 ParametricSurfaceSeries 对象，表示三维参数化表面系列
    s = ParametricSurfaceSeries(*expr, (u, 0, 2 * pi), (v, 0, pi), "",
        adaptive=False, only_integers=True)
    # 获取该表面系列的数据，返回元组 (x 数据, y 数据, z 数据, u 数据, v 数据)
    xx, yy, zz, uu, vv = s.get_data()
    # 断言检查 xx, yy, zz, uu, vv 的形状是否为 (4, 7)
    assert xx.shape == yy.shape == zz.shape == uu.shape == vv.shape == (4, 7)
def test_is_point_is_filled():
    # 验证 `is_point` 和 `is_filled` 是属性，并且它们接收到正确的值

    if not np:
        skip("numpy not installed.")

    x, u = symbols("x, u")

    # 创建一个 LineOver1DRangeSeries 实例，表示一维范围上的线性系列，使用余弦函数作为数据，范围是 -5 到 5
    # is_point=False 表示不是点状系列，is_filled=True 表示填充
    s = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled

    # 创建另一个 LineOver1DRangeSeries 实例，is_point=True 表示是点状系列，is_filled=False 表示不填充
    s = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    # 创建一个 List2DSeries 实例，表示二维列表系列，x 轴数据是 [0, 1, 2]，y 轴数据是 [3, 4, 5]
    # is_point=False 表示不是点状系列，is_filled=True 表示填充
    s = List2DSeries([0, 1, 2], [3, 4, 5],
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled

    # 创建另一个 List2DSeries 实例，is_point=True 表示是点状系列，is_filled=False 表示不填充
    s = List2DSeries([0, 1, 2], [3, 4, 5],
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    # 创建一个 Parametric2DLineSeries 实例，表示二维参数线性系列，使用余弦和正弦函数作为数据，范围是 -5 到 5
    # is_point=False 表示不是点状系列，is_filled=True 表示填充
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled

    # 创建另一个 Parametric2DLineSeries 实例，is_point=True 表示是点状系列，is_filled=False 表示不填充
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)

    # 创建一个 Parametric3DLineSeries 实例，表示三维参数线性系列，使用余弦和正弦函数作为数据，范围是 -5 到 5
    # is_point=False 表示不是点状系列，is_filled=True 表示填充
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        is_point=False, is_filled=True)
    assert (not s.is_point) and s.is_filled

    # 创建另一个 Parametric3DLineSeries 实例，is_point=True 表示是点状系列，is_filled=False 表示不填充
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        is_point=True, is_filled=False)
    assert s.is_point and (not s.is_filled)


def test_is_filled_2d():
    # 验证 is_filled 属性是否由以下系列类所暴露

    x, y = symbols("x, y")

    expr = cos(x**2 + y**2)
    ranges = (x, -2, 2), (y, -2, 2)

    # 创建 ContourSeries 实例，表示轮廓系列，使用表达式 expr 和给定的范围
    s = ContourSeries(expr, *ranges)
    assert s.is_filled

    # 创建 ContourSeries 实例，is_filled=True 表示填充
    s = ContourSeries(expr, *ranges, is_filled=True)
    assert s.is_filled

    # 创建 ContourSeries 实例，is_filled=False 表示不填充
    s = ContourSeries(expr, *ranges, is_filled=False)
    assert not s.is_filled


def test_steps():
    if not np:
        skip("numpy not installed.")

    x, u = symbols("x, u")

    def do_test(s1, s2):
        # 如果 s1 不是参数化且是二维线性系列，获取数据并进行比较
        if (not s1.is_parametric) and s1.is_2Dline:
            xx1, _ = s1.get_data()
            xx2, _ = s2.get_data()
        # 如果 s1 是参数化且是二维线性系列，获取数据并进行比较
        elif s1.is_parametric and s1.is_2Dline:
            xx1, _, _ = s1.get_data()
            xx2, _, _ = s2.get_data()
        # 如果 s1 不是参数化且是三维线性系列，获取数据并进行比较
        elif (not s1.is_parametric) and s1.is_3Dline:
            xx1, _, _ = s1.get_data()
            xx2, _, _ = s2.get_data()
        # 其他情况，假定为四维数据，获取数据并进行比较
        else:
            xx1, _, _, _ = s1.get_data()
            xx2, _, _, _ = s2.get_data()
        
        # 确保两个数据集的长度不相等
        assert len(xx1) != len(xx2)

    # 创建 LineOver1DRangeSeries 实例，表示一维范围上的线性系列，使用余弦函数作为数据，范围是 -5 到 5
    # adaptive=False 表示非自适应，n=40 表示40个数据点，steps=False 表示不启用步进
    s1 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=False)
    # 创建另一个 LineOver1DRangeSeries 实例，steps=True 表示启用步进
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), "",
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)

    # 创建 List2DSeries 实例，表示二维列表系列，x 轴数据是 [0, 1, 2]，y 轴数据是 [3, 4, 5]
    # steps=False 表示不启用步进
    s1 = List2DSeries([0, 1, 2], [3, 4, 5], steps=False)
    # 创建另一个 List2DSeries 实例，steps=True 表示启用步进
    s2 = List2DSeries([0, 1, 2], [3, 4, 5], steps=True)
    do_test(s1, s2)

    # 创建 Parametric2DLineSeries 实例，表示二维参数线性系列，使用余弦和正弦函数作为数据，范围是 -5 到 5
    # adaptive=False 表示非自适应，n=40 表示40个数据点，steps=False 表示不启用步进
    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=40, steps=False)
    # 创建另一个 Parametric2DLineSeries 实例，steps=True 表示启用步进
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=40, steps=True)
    do_test(s1, s2)
    `
    # 创建一个 Parametric3DLineSeries 对象 s1，使用函数 cos(x), sin(x), x 作为参数，范围在 x 从 -5 到 5
    s1 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=40, steps=False)
    # 创建一个 Parametric3DLineSeries 对象 s2，使用函数 cos(x), sin(x), x 作为参数，范围在 x 从 -5 到 5
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=40, steps=True)
    # 调用 do_test 函数，传入参数 s1 和 s2
    do_test(s1, s2)
def test_interactive_data():
    # verify that InteractiveSeries produces the same numerical data as their
    # corresponding non-interactive series.
    # 检查 InteractiveSeries 生成的数值数据与其对应的非交互系列是否相同

    if not np:
        skip("numpy not installed.")
        # 如果没有安装 numpy，则跳过测试

    u, x, y, z = symbols("u, x:z")

    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)
        # 检查两组数据是否长度相同，并逐个元素检查它们是否非常接近

    s1 = LineOver1DRangeSeries(u * cos(x), (x, -5, 5), params={u: 1}, n=50)
    s2 = LineOver1DRangeSeries(cos(x), (x, -5, 5), adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    # 测试 LineOver1DRangeSeries 类的两个实例生成的数据是否一致

    s1 = Parametric2DLineSeries(
        u * cos(x), u * sin(x), (x, -5, 5), params={u: 1}, n=50)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, -5, 5),
        adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    # 测试 Parametric2DLineSeries 类的两个实例生成的数据是否一致

    s1 = Parametric3DLineSeries(
        u * cos(x), u * sin(x), u * x, (x, -5, 5),
        params={u: 1}, n=50)
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, -5, 5),
        adaptive=False, n=50)
    do_test(s1.get_data(), s2.get_data())
    # 测试 Parametric3DLineSeries 类的两个实例生成的数据是否一致

    s1 = SurfaceOver2DRangeSeries(
        u * cos(x ** 2 + y ** 2), (x, -3, 3), (y, -3, 3),
        params={u: 1}, n1=50, n2=50,)
    s2 = SurfaceOver2DRangeSeries(
        cos(x ** 2 + y ** 2), (x, -3, 3), (y, -3, 3),
        adaptive=False, n1=50, n2=50)
    do_test(s1.get_data(), s2.get_data())
    # 测试 SurfaceOver2DRangeSeries 类的两个实例生成的数据是否一致

    s1 = ParametricSurfaceSeries(
        u * cos(x + y), sin(x + y), x - y, (x, -3, 3), (y, -3, 3),
        params={u: 1}, n1=50, n2=50,)
    s2 = ParametricSurfaceSeries(
        cos(x + y), sin(x + y), x - y, (x, -3, 3), (y, -3, 3),
        adaptive=False, n1=50, n2=50,)
    do_test(s1.get_data(), s2.get_data())
    # 测试 ParametricSurfaceSeries 类的两个实例生成的数据是否一致

    # real part of a complex function evaluated over a real line with numpy
    # 使用 numpy 计算复数函数在实数线上的实部

    expr = re((z ** 2 + 1) / (z ** 2 - 1))
    s1 = LineOver1DRangeSeries(u * expr, (z, -3, 3), adaptive=False, n=50,
        modules=None, params={u: 1})
    s2 = LineOver1DRangeSeries(expr, (z, -3, 3), adaptive=False, n=50,
        modules=None)
    do_test(s1.get_data(), s2.get_data())
    # 测试 LineOver1DRangeSeries 类的两个实例生成的数据是否一致

    # real part of a complex function evaluated over a real line with mpmath
    # 使用 mpmath 计算复数函数在实数线上的实部

    expr = re((z ** 2 + 1) / (z ** 2 - 1))
    s1 = LineOver1DRangeSeries(u * expr, (z, -3, 3), n=50, modules="mpmath",
        params={u: 1})
    s2 = LineOver1DRangeSeries(expr, (z, -3, 3),
        adaptive=False, n=50, modules="mpmath")
    do_test(s1.get_data(), s2.get_data())
    # 测试 LineOver1DRangeSeries 类的两个实例生成的数据是否一致


def test_list2dseries_interactive():
    if not np:
        skip("numpy not installed.")
        # 如果没有安装 numpy，则跳过测试

    x, y, u = symbols("x, y, u")

    s = List2DSeries([1, 2, 3], [1, 2, 3])
    assert not s.is_interactive
    # 断言 List2DSeries 实例不是交互式的

    # symbolic expressions as coordinates, but no ``params``
    # 符号表达式作为坐标，但没有 params 参数
    raises(ValueError, lambda: List2DSeries([cos(x)], [sin(x)]))
    # 预期抛出 ValueError 异常，因为没有给定 params 参数

    # too few parameters
    # 参数太少
    raises(ValueError,
        lambda: List2DSeries([cos(x), y], [sin(x), 2], params={u: 1}))
    # 预期抛出 ValueError 异常，因为参数个数不匹配

    s = List2DSeries([cos(x)], [sin(x)], params={x: 1})
    assert s.is_interactive
    # 断言 List2DSeries 实例是交互式的
    # 创建一个 List2DSeries 对象，使用指定的列表作为 x 和 y 数据，同时传入参数字典 {x: 3}
    s = List2DSeries([x, 2, 3, 4], [4, 3, 2, x], params={x: 3})
    # 调用 List2DSeries 对象的 get_data 方法，获取返回的 xx 和 yy 数据
    xx, yy = s.get_data()
    # 使用 numpy 的 allclose 函数检查 xx 是否与期望的数值列表 [3, 2, 3, 4] 接近
    assert np.allclose(xx, [3, 2, 3, 4])
    # 使用 numpy 的 allclose 函数检查 yy 是否与期望的数值列表 [4, 3, 2, 3] 接近
    assert np.allclose(yy, [4, 3, 2, 3])
    # 断言对象 s 的 is_parametric 属性为 False
    assert not s.is_parametric

    # 创建另一个 List2DSeries 对象，使用 [1, 2, 3] 作为 x 和 y 数据，并传入参数字典 {x: 1}
    s = List2DSeries([1, 2, 3], [1, 2, 3], params={x: 1})
    # 断言对象 s 的 is_interactive 属性为 True，表示该对象是交互式的
    assert s.is_interactive
    # 断言对象 s 的 list_x 属性为 Tuple 类型，表明 x 数据被转换为元组
    assert isinstance(s.list_x, Tuple)
    # 断言对象 s 的 list_y 属性为 Tuple 类型，表明 y 数据被转换为元组
    assert isinstance(s.list_y, Tuple)
def test_mpmath():
    # 测试复数函数的参数在使用 mpmath 计算时与使用 Numpy 计算可能不同（在分支切割处行为不同）
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    z, u = symbols("z, u")  # 定义符号变量 z 和 u

    # 创建线性范围上的线性系列对象 s1 和 s2，使用 mpmath 模块计算复数函数 im(sqrt(-z))
    s1 = LineOver1DRangeSeries(im(sqrt(-z)), (z, 1e-03, 5),
        adaptive=True, modules=None, force_real_eval=True)
    s2 = LineOver1DRangeSeries(im(sqrt(-z)), (z, 1e-03, 5),
        adaptive=True, modules="mpmath", force_real_eval=True)
    xx1, yy1 = s1.get_data()  # 获取 s1 的数据
    xx2, yy2 = s2.get_data()  # 获取 s2 的数据
    assert np.all(yy1 < 0)  # 断言 s1 中所有的 yy1 值均小于 0
    assert np.all(yy2 > 0)  # 断言 s2 中所有的 yy2 值均大于 0

    # 创建线性范围上的线性系列对象 s1 和 s2，使用 mpmath 模块计算复数函数 im(sqrt(-z))
    s1 = LineOver1DRangeSeries(im(sqrt(-z)), (z, -5, 5),
        adaptive=False, n=20, modules=None, force_real_eval=True)
    s2 = LineOver1DRangeSeries(im(sqrt(-z)), (z, -5, 5),
        adaptive=False, n=20, modules="mpmath", force_real_eval=True)
    xx1, yy1 = s1.get_data()  # 获取 s1 的数据
    xx2, yy2 = s2.get_data()  # 获取 s2 的数据
    assert np.allclose(xx1, xx2)  # 断言 s1 和 s2 的 xx1 数据近似相等
    assert not np.allclose(yy1, yy2)  # 断言 s1 和 s2 的 yy1 数据不近似相等


def test_str():
    u, x, y, z = symbols("u, x:z")  # 定义符号变量 u, x, y, z

    # 创建线性范围上的线性系列对象 s，计算余弦函数 cos(x)
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3))
    assert str(s) == "cartesian line: cos(x) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    d = {"return": "real"}
    # 创建线性范围上的线性系列对象 s，计算余弦函数的实部 re(cos(x))
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: re(cos(x)) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    d = {"return": "imag"}
    # 创建线性范围上的线性系列对象 s，计算余弦函数的虚部 im(cos(x))
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: im(cos(x)) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    d = {"return": "abs"}
    # 创建线性范围上的线性系列对象 s，计算余弦函数的绝对值 abs(cos(x))
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: abs(cos(x)) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    d = {"return": "arg"}
    # 创建线性范围上的线性系列对象 s，计算余弦函数的幅角 arg(cos(x))
    s = LineOver1DRangeSeries(cos(x), (x, -4, 3), **d)
    assert str(s) == "cartesian line: arg(cos(x)) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    # 创建交互式线性范围上的线性系列对象 s，计算余弦函数 cos(u*x)，参数为 u
    s = LineOver1DRangeSeries(cos(u * x), (x, -4, 3), params={u: 1})
    assert str(s) == "interactive cartesian line: cos(u*x) for x over (-4.0, 3.0) and parameters (u,)"  # 断言 s 的字符串表示符合预期

    # 创建交互式线性范围上的线性系列对象 s，计算余弦函数 cos(u*x)，参数为 u 和 y
    s = LineOver1DRangeSeries(cos(u * x), (x, -u, 3*y), params={u: 1, y: 1})
    assert str(s) == "interactive cartesian line: cos(u*x) for x over (-u, 3*y) and parameters (u, y)"  # 断言 s 的字符串表示符合预期

    # 创建参数化二维线性系列对象 s，计算二维余弦函数曲线 (cos(x), sin(x))
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3))
    assert str(s) == "parametric cartesian line: (cos(x), sin(x)) for x over (-4.0, 3.0)"  # 断言 s 的字符串表示符合预期

    # 创建交互式参数化二维线性系列对象 s，计算二维余弦函数曲线 (cos(u*x), sin(x))，参数为 u
    s = Parametric2DLineSeries(cos(u * x), sin(x), (x, -4, 3), params={u: 1})
    assert str(s) == "interactive parametric cartesian line: (cos(u*x), sin(x)) for x over (-4.0, 3.0) and parameters (u,)"  # 断言 s 的字符串表示符合预期

    # 创建交互式参数化二维线性系列对象 s，计算二维余弦函数曲线 (cos(u*x), sin(x))，参数为 u 和 y
    s = Parametric2DLineSeries(cos(u * x), sin(x), (x, -u, 3*y), params={u: 1, y: 1})
    assert str(s) == "interactive parametric cartesian line: (cos(u*x), sin(x)) for x over (-u, 3*y) and parameters (u, y)"  # 断言 s 的字符串表示符合预期

    # 创建参数化三维线性系列对象 s，计算三维余弦函数曲线 (cos(x), sin(x), x)
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3))
    assert str(s) == "3D parametric cartesian line: (cos(x), sin(x), x) for x over (-4.0, 3.0)"
    # 断言 s 的字符串表示符合预期
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive 3D parametric cartesian line: (cos(u*x), sin(x), x) for x over (-4.0, 3.0) and parameters (u,)"

    # 创建 Parametric3DLineSeries 对象，定义一个参数化的三维线系列
    s = Parametric3DLineSeries(cos(u*x), sin(x), x, (x, -u, 3*y), params={u: 1, y: 1})
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive 3D parametric cartesian line: (cos(u*x), sin(x), x) for x over (-u, 3*y) and parameters (u, y)"

    # 创建 SurfaceOver2DRangeSeries 对象，定义一个二维范围上的表面系列
    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5))
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "cartesian surface: cos(x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    # 创建 SurfaceOver2DRangeSeries 对象，定义一个带参数的二维范围上的表面系列
    s = SurfaceOver2DRangeSeries(cos(u * x * y), (x, -4, 3), (y, -2, 5), params={u: 1})
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive cartesian surface: cos(u*x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    # 创建 SurfaceOver2DRangeSeries 对象，定义一个带参数的二维范围上的表面系列
    s = SurfaceOver2DRangeSeries(cos(u * x * y), (x, -4*u, 3), (y, -2, 5*u), params={u: 1})
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive cartesian surface: cos(u*x*y) for x over (-4*u, 3.0) and y over (-2.0, 5*u) and parameters (u,)"

    # 创建 ContourSeries 对象，定义一个等高线系列
    s = ContourSeries(cos(x * y), (x, -4, 3), (y, -2, 5))
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "contour: cos(x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    # 创建 ContourSeries 对象，定义一个带参数的等高线系列
    s = ContourSeries(cos(u * x * y), (x, -4, 3), (y, -2, 5), params={u: 1})
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive contour: cos(u*x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    # 创建 ParametricSurfaceSeries 对象，定义一个参数化的二维范围上的表面系列
    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5))
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "parametric cartesian surface: (cos(x*y), sin(x*y), x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0)"

    # 创建 ParametricSurfaceSeries 对象，定义一个带参数的参数化二维范围上的表面系列
    s = ParametricSurfaceSeries(cos(u * x * y), sin(x * y), x * y,
        (x, -4, 3), (y, -2, 5), params={u: 1})
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "interactive parametric cartesian surface: (cos(u*x*y), sin(x*y), x*y) for x over (-4.0, 3.0) and y over (-2.0, 5.0) and parameters (u,)"

    # 创建 ImplicitSeries 对象，定义一个隐式表达式系列
    s = ImplicitSeries(x < y, (x, -5, 4), (y, -3, 2))
    # 断言，验证字符串表达式是否与预期相符
    assert str(s) == "Implicit expression: x < y for x over (-5.0, 4.0) and y over (-3.0, 2.0)"
def test_use_cm():
    # 验证 `use_cm` 属性是否被正确实现

    # 检查是否安装了 numpy，如果没有则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量
    u, x, y, z = symbols("u, x:z")

    # 测试 List2DSeries 类的 use_cm 参数
    s = List2DSeries([1, 2, 3, 4], [5, 6, 7, 8], use_cm=True)
    assert s.use_cm
    s = List2DSeries([1, 2, 3, 4], [5, 6, 7, 8], use_cm=False)
    assert not s.use_cm

    # 测试 Parametric2DLineSeries 类的 use_cm 参数
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3), use_cm=True)
    assert s.use_cm
    s = Parametric2DLineSeries(cos(x), sin(x), (x, -4, 3), use_cm=False)
    assert not s.use_cm

    # 测试 Parametric3DLineSeries 类的 use_cm 参数
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3), use_cm=True)
    assert s.use_cm
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, -4, 3), use_cm=False)
    assert not s.use_cm

    # 测试 SurfaceOver2DRangeSeries 类的 use_cm 参数
    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5), use_cm=True)
    assert s.use_cm
    s = SurfaceOver2DRangeSeries(cos(x * y), (x, -4, 3), (y, -2, 5), use_cm=False)
    assert not s.use_cm

    # 测试 ParametricSurfaceSeries 类的 use_cm 参数
    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y, (x, -4, 3), (y, -2, 5), use_cm=True)
    assert s.use_cm
    s = ParametricSurfaceSeries(cos(x * y), sin(x * y), x * y, (x, -4, 3), (y, -2, 5), use_cm=False)
    assert not s.use_cm


def test_surface_use_cm():
    # 验证 SurfaceOver2DRangeSeries 和 ParametricSurfaceSeries 类的 use_cm 属性一致性

    # 定义符号变量
    x, y, u, v = symbols("x, y, u, v")

    # 默认设置下，确保两个对象读取相同的 use_cm 值
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2))
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u, (u, 0, 1), (v, 0 , 2*pi))
    assert s1.use_cm == s2.use_cm

    # 指定 use_cm=False，确保两个对象的 use_cm 值相同
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), use_cm=False)
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u, (u, 0, 1), (v, 0 , 2*pi), use_cm=False)
    assert s1.use_cm == s2.use_cm

    # 指定 use_cm=True，确保两个对象的 use_cm 值相同
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2), use_cm=True)
    s2 = ParametricSurfaceSeries(u * cos(v), u * sin(v), u, (u, 0, 1), (v, 0 , 2*pi), use_cm=True)
    assert s1.use_cm == s2.use_cm


def test_sums():
    # 测试数据序列能否处理求和

    # 检查是否安装了 numpy，如果没有则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量
    x, y, u = symbols("x, y, u")

    # 定义内部测试函数，比较两组数据是否相等
    def do_test(data1, data2):
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    # 测试 LineOver1DRangeSeries 类的 use_cm 参数
    s = LineOver1DRangeSeries(Sum(1 / x ** y, (x, 1, 1000)), (y, 2, 10), adaptive=False, only_integers=True)
    xx, yy = s.get_data()

    # 测试 LineOver1DRangeSeries 类的 use_cm 参数
    s1 = LineOver1DRangeSeries(Sum(1 / x, (x, 1, y)), (y, 2, 10), adaptive=False, only_integers=True)
    xx1, yy1 = s1.get_data()

    # 测试 LineOver1DRangeSeries 类的 use_cm 参数
    s2 = LineOver1DRangeSeries(Sum(u / x, (x, 1, y)), (y, 2, 10), params={u: 1}, only_integers=True)
    xx2, yy2 = s2.get_data()
    xx1 = xx1.astype(float)
    xx2 = xx2.astype(float)
    # 调用函数 do_test，传入两个参数列表 [xx1, yy1] 和 [xx2, yy2]
    do_test([xx1, yy1], [xx2, yy2])

    # 创建 LineOver1DRangeSeries 对象 s，计算表达式 Sum(1 / x, (x, 1, y)) 在变量 y 范围为 2 到 10 的值，并使用自适应模式
    s = LineOver1DRangeSeries(Sum(1 / x, (x, 1, y)), (y, 2, 10),
        adaptive=True)

    # 使用 warns 上下文管理器，检查是否会引发 UserWarning 异常，并匹配给定的消息 "The evaluation with NumPy/SciPy failed"，同时禁用测试相关的堆栈信息
    with warns(
        UserWarning,
        match="The evaluation with NumPy/SciPy failed",
        test_stacklevel=False,
    ):
        # 断言调用 s.get_data() 会引发 TypeError 异常
        raises(TypeError, lambda: s.get_data())
def test_apply_transforms():
    # 验证变换函数是否应用于数据序列的输出

    # 检查是否安装了 numpy 库，如果没有则跳过测试
    if not np:
        skip("numpy not installed.")

    # 定义符号变量
    x, y, z, u, v = symbols("x:z, u, v")

    # 创建不同的一维范围线性系列对象，不应用任何转换函数
    s1 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10)
    
    # 创建一维范围线性系列对象，应用角度转换函数 np.rad2deg 到 x 轴
    s2 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        tx=np.rad2deg)
    
    # 创建一维范围线性系列对象，应用角度转换函数 np.rad2deg 到 y 轴
    s3 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        ty=np.rad2deg)
    
    # 创建一维范围线性系列对象，应用角度转换函数 np.rad2deg 到 x 和 y 轴
    s4 = LineOver1DRangeSeries(cos(x), (x, -2*pi, 2*pi), adaptive=False, n=10,
        tx=np.rad2deg, ty=np.rad2deg)

    # 获取各个系列对象的数据
    x1, y1 = s1.get_data()
    x2, y2 = s2.get_data()
    x3, y3 = s3.get_data()
    x4, y4 = s4.get_data()

    # 断言：检查第一个系列的数据范围和最大最小值
    assert np.isclose(x1[0], -2*np.pi) and np.isclose(x1[-1], 2*np.pi)
    assert (y1.min() < -0.9) and (y1.max() > 0.9)

    # 断言：检查第二个系列的数据范围和最大最小值，以及 x 轴的角度转换
    assert np.isclose(x2[0], -360) and np.isclose(x2[-1], 360)
    assert (y2.min() < -0.9) and (y2.max() > 0.9)

    # 断言：检查第三个系列的数据范围和最大最小值，以及 y 轴的角度转换
    assert np.isclose(x3[0], -2*np.pi) and np.isclose(x3[-1], 2*np.pi)
    assert (y3.min() < -52) and (y3.max() > 52)

    # 断言：检查第四个系列的数据范围和最大最小值，以及 x 和 y 轴的角度转换
    assert np.isclose(x4[0], -360) and np.isclose(x4[-1], 360)
    assert (y4.min() < -52) and (y4.max() > 52)

    # 生成均匀分布的数据点，并创建二维列表数据系列对象，不应用任何转换函数
    xx = np.linspace(-2*np.pi, 2*np.pi, 10)
    yy = np.cos(xx)
    s1 = List2DSeries(xx, yy)
    
    # 创建二维列表数据系列对象，应用角度转换函数 np.rad2deg 到 x 和 y 轴
    s2 = List2DSeries(xx, yy, tx=np.rad2deg, ty=np.rad2deg)
    
    # 获取各个系列对象的数据
    x1, y1 = s1.get_data()
    x2, y2 = s2.get_data()

    # 断言：检查第一个系列的数据范围和最大最小值
    assert np.isclose(x1[0], -2*np.pi) and np.isclose(x1[-1], 2*np.pi)
    assert (y1.min() < -0.9) and (y1.max() > 0.9)

    # 断言：检查第二个系列的数据范围和最大最小值，以及 x 和 y 轴的角度转换
    assert np.isclose(x2[0], -360) and np.isclose(x2[-1], 360)
    assert (y2.min() < -52) and (y2.max() > 52)

    # 创建二维参数化线性系列对象，不应用任何转换函数
    s1 = Parametric2DLineSeries(
        sin(x), cos(x), (x, -pi, pi), adaptive=False, n=10)
    
    # 创建二维参数化线性系列对象，应用角度转换函数 np.rad2deg 到 x、y 和参数 t 轴
    s2 = Parametric2DLineSeries(
        sin(x), cos(x), (x, -pi, pi), adaptive=False, n=10,
        tx=np.rad2deg, ty=np.rad2deg, tp=np.rad2deg)
    
    # 获取各个系列对象的数据
    x1, y1, a1 = s1.get_data()
    x2, y2, a2 = s2.get_data()

    # 断言：检查 x 轴数据与角度转换后的 x 轴数据是否近似相等
    assert np.allclose(x1, np.deg2rad(x2))
    # 断言：检查 y 轴数据与角度转换后的 y 轴数据是否近似相等
    assert np.allclose(y1, np.deg2rad(y2))
    # 断言：检查参数 t 轴数据与角度转换后的参数 t 轴数据是否近似相等
    assert np.allclose(a1, np.deg2rad(a2))

    # 创建三维参数化线性系列对象，不应用任何转换函数
    s1 =  Parametric3DLineSeries(
        sin(x), cos(x), x, (x, -pi, pi), adaptive=False, n=10)
    
    # 创建三维参数化线性系列对象，应用角度转换函数 np.rad2deg 到参数 t 轴
    s2 = Parametric3DLineSeries(
        sin(x), cos(x), x, (x, -pi, pi), adaptive=False, n=10, tp=np.rad2deg)
    
    # 获取各个系列对象的数据
    x1, y1, z1, a1 = s1.get_data()
    x2, y2, z2, a2 = s2.get_data()

    # 断言：检查 x、y、z 轴数据是否近似相等
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    assert np.allclose(z1, z2)
    # 断言：检查参数 t 轴数据与角度转换后的参数 t 轴数据是否近似相等
    assert np.allclose(a1, np.deg2rad(a2))

    # 创建二维范围曲面系列对象，不应用任何转换函数
    s1 = SurfaceOver2DRangeSeries(
        cos(x**2 + y**2), (x, -2*pi, 2*pi), (y, -2*pi, 2*pi),
        adaptive=False, n1=10, n2=10)
    
    # 创建二维范围曲面系列对象，应用角度转换函数 np.rad2deg 到 x 轴，lambda 函数 2*x 到 y 轴，lambda 函数 3*x 到 z 轴
    s2 = SurfaceOver2DRangeSeries(
        cos(x**2 + y**2), (x, -2*pi, 2*pi), (y, -2*pi, 2*pi),
        adaptive=False, n1=10, n2=10,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)
    
    # 获取各个系列对象的数据
    x1, y1, z1 = s1.get_data()
    x2, y2, z2 = s2.get_data()

    # 断言：检查 x 轴数据与角度转换后的 x 轴数据是否近似相等
    assert np.allclose(x1, np.deg2rad(x2))
    #
    # 创建第一个 ParametricSurfaceSeries 对象，定义参数化曲面的表达式和参数范围
    s1 = ParametricSurfaceSeries(
        u + v, u - v, u * v, (u, 0, 2*pi), (v, 0, pi),
        adaptive=False, n1=10, n2=10)

    # 创建第二个 ParametricSurfaceSeries 对象，与第一个对象相似但包含额外的变换函数
    s2 = ParametricSurfaceSeries(
        u + v, u - v, u * v, (u, 0, 2*pi), (v, 0, pi),
        adaptive=False, n1=10, n2=10,
        tx=np.rad2deg, ty=lambda x: 2*x, tz=lambda x: 3*x)

    # 从第一个 ParametricSurfaceSeries 对象中获取数据
    x1, y1, z1, u1, v1 = s1.get_data()

    # 从第二个 ParametricSurfaceSeries 对象中获取数据
    x2, y2, z2, u2, v2 = s2.get_data()

    # 使用 NumPy 的 allclose 函数检查 x1 和 np.deg2rad(x2) 是否在容差范围内相等
    assert np.allclose(x1, np.deg2rad(x2))

    # 使用 NumPy 的 allclose 函数检查 y1 和 y2/2 是否在容差范围内相等
    assert np.allclose(y1, y2 / 2)

    # 使用 NumPy 的 allclose 函数检查 z1 和 z2/3 是否在容差范围内相等
    assert np.allclose(z1, z2 / 3)

    # 使用 NumPy 的 allclose 函数检查 u1 和 u2 是否在容差范围内相等
    assert np.allclose(u1, u2)

    # 使用 NumPy 的 allclose 函数检查 v1 和 v2 是否在容差范围内相等
    assert np.allclose(v1, v2)
def test_series_labels():
    # 验证系列根据绘图类型和输入参数返回正确的标签。如果用户在数据系列上设置了自定义标签，
    # 则应返回未修改的标签。
    if not np:
        skip("numpy not installed.")

    # 定义符号变量
    x, y, z, u, v = symbols("x, y, z, u, v")
    # LaTeX包装器
    wrapper = "$%s$"

    # 单变量函数表达式
    expr = cos(x)
    # 创建一维范围上的线性系列对象，未指定标签
    s1 = LineOver1DRangeSeries(expr, (x, -2, 2), None)
    # 创建一维范围上的线性系列对象，指定标签为"test"
    s2 = LineOver1DRangeSeries(expr, (x, -2, 2), "test")
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == str(expr)
    # 验证使用数学包装的标签
    assert s1.get_label(True) == wrapper % latex(expr)
    # 验证未使用数学包装的自定义标签
    assert s2.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s2.get_label(True) == "test"

    # 创建二维列表系列对象，指定标签为"test"
    s1 = List2DSeries([0, 1, 2, 3], [0, 1, 2, 3], "test")
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == "test"
    # 验证使用数学包装的标签
    assert s1.get_label(True) == "test"

    # 多变量函数表达式
    expr = (cos(x), sin(x))
    # 创建二维参数化线性系列对象，未指定标签
    s1 = Parametric2DLineSeries(*expr, (x, -2, 2), None, use_cm=True)
    # 创建二维参数化线性系列对象，指定标签为"test"
    s2 = Parametric2DLineSeries(*expr, (x, -2, 2), "test", use_cm=True)
    # 创建二维参数化线性系列对象，未使用色彩映射
    s3 = Parametric2DLineSeries(*expr, (x, -2, 2), None, use_cm=False)
    # 创建二维参数化线性系列对象，指定标签为"test"，未使用色彩映射
    s4 = Parametric2DLineSeries(*expr, (x, -2, 2), "test", use_cm=False)
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == "x"
    # 验证使用数学包装的标签
    assert s1.get_label(True) == wrapper % "x"
    # 验证未使用数学包装的自定义标签
    assert s2.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s2.get_label(True) == "test"
    # 验证未使用数学包装的标签
    assert s3.get_label(False) == str(expr)
    # 验证使用数学包装的标签
    assert s3.get_label(True) == wrapper % latex(expr)
    # 验证未使用数学包装的自定义标签
    assert s4.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s4.get_label(True) == "test"

    # 多变量函数表达式
    expr = (cos(x), sin(x), x)
    # 创建三维参数化线性系列对象，未指定标签
    s1 = Parametric3DLineSeries(*expr, (x, -2, 2), None, use_cm=True)
    # 创建三维参数化线性系列对象，指定标签为"test"
    s2 = Parametric3DLineSeries(*expr, (x, -2, 2), "test", use_cm=True)
    # 创建三维参数化线性系列对象，未使用色彩映射
    s3 = Parametric3DLineSeries(*expr, (x, -2, 2), None, use_cm=False)
    # 创建三维参数化线性系列对象，指定标签为"test"，未使用色彩映射
    s4 = Parametric3DLineSeries(*expr, (x, -2, 2), "test", use_cm=False)
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == "x"
    # 验证使用数学包装的标签
    assert s1.get_label(True) == wrapper % "x"
    # 验证未使用数学包装的自定义标签
    assert s2.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s2.get_label(True) == "test"
    # 验证未使用数学包装的标签
    assert s3.get_label(False) == str(expr)
    # 验证使用数学包装的标签
    assert s3.get_label(True) == wrapper % latex(expr)
    # 验证未使用数学包装的自定义标签
    assert s4.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s4.get_label(True) == "test"

    # 二维曲面方程式表达式
    expr = cos(x**2 + y**2)
    # 创建二维范围上的曲面系列对象，未指定标签
    s1 = SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -2, 2), None)
    # 创建二维范围上的曲面系列对象，指定标签为"test"
    s2 = SurfaceOver2DRangeSeries(expr, (x, -2, 2), (y, -2, 2), "test")
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == str(expr)
    # 验证使用数学包装的标签
    assert s1.get_label(True) == wrapper % latex(expr)
    # 验证未使用数学包装的自定义标签
    assert s2.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s2.get_label(True) == "test"

    # 多变量方程式表达式
    expr = (cos(x - y), sin(x + y), x - y)
    # 创建二维参数化曲面系列对象，未指定标签
    s1 = ParametricSurfaceSeries(*expr, (x, -2, 2), (y, -2, 2), None)
    # 创建二维参数化曲面系列对象，指定标签为"test"
    s2 = ParametricSurfaceSeries(*expr, (x, -2, 2), (y, -2, 2), "test")
    # 验证未使用数学包装的标签
    assert s1.get_label(False) == str(expr)
    # 验证使用数学包装的标签
    assert s1.get_label(True) == wrapper % latex(expr)
    # 验证未使用数学包装的自定义标签
    assert s2.get_label(False) == "test"
    # 验证使用数学包装的自定义标签
    assert s2.get_label(True) == "test"

    # 隐式方程式表达式
    expr = Eq(cos(x - y), 0)
    # 创建二维范围上的隐式曲线系列对象，未指定标签
    s1 = ImplicitSeries(expr, (x, -10, 10), (y, -10, 10), None)
    # 创建二维范围上的隐式曲线系列对象，指定标签为"test"
    s2 = ImplicitSeries(expr, (x, -10, 10), (y, -10, 10), "test")
    # 断言：检查 s1 对象的 get_label(False) 方法返回的结果是否等于 expr 对象的字符串表示形式
    assert s1.get_label(False) == str(expr)
    
    # 断言：检查 s1 对象的 get_label(True) 方法返回的结果是否等于用 latex 函数处理 expr 对象后的字符串表示形式
    assert s1.get_label(True) == wrapper % latex(expr)
    
    # 断言：检查 s2 对象的 get_label(False) 方法返回的结果是否等于字符串 "test"
    assert s2.get_label(False) == "test"
    
    # 断言：检查 s2 对象的 get_label(True) 方法返回的结果是否等于字符串 "test"
    assert s2.get_label(True) == "test"
def test_is_polar_2d_parametric():
    # verify that Parametric2DLineSeries is able to apply polar discretization,
    # which is used when polar_plot is executed with polar_axis=True
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    t, u = symbols("t u")

    # NOTE: a sufficiently big n must be provided, or else tests
    # are going to fail
    # No colormap
    f = sin(4 * t)
    s1 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=False, use_cm=False)
    x1, y1, p1 = s1.get_data()  # 获取 s1 的数据：x 坐标、y 坐标、参数值

    s2 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=True, use_cm=False)
    th, r, p2 = s2.get_data()  # 获取 s2 的数据：极角坐标、极径、参数值

    assert (not np.allclose(x1, th)) and (not np.allclose(y1, r))  # 断言：x1 与 th 不完全相等，y1 与 r 不完全相等
    assert np.allclose(p1, p2)  # 断言：p1 与 p2 完全相等

    # With colormap
    s3 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=False, color_func=lambda t: 2*t)
    x3, y3, p3 = s3.get_data()  # 获取 s3 的数据：x 坐标、y 坐标、参数值

    s4 = Parametric2DLineSeries(f * cos(t), f * sin(t), (t, 0, 2*pi),
        adaptive=False, n=10, is_polar=True, color_func=lambda t: 2*t)
    th4, r4, p4 = s4.get_data()  # 获取 s4 的数据：极角坐标、极径、参数值

    assert np.allclose(p3, p4) and (not np.allclose(p1, p3))  # 断言：p3 与 p4 完全相等，且 p1 与 p3 不完全相等
    assert np.allclose(x3, x1) and np.allclose(y3, y1)  # 断言：x3 与 x1 完全相等，y3 与 y1 完全相等
    assert np.allclose(th4, th) and np.allclose(r4, r)  # 断言：th4 与 th 完全相等，r4 与 r 完全相等


def test_is_polar_3d():
    # verify that SurfaceOver2DRangeSeries is able to apply
    # polar discretization
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x, y, t = symbols("x, y, t")
    expr = (x**2 - 1)**2
    s1 = SurfaceOver2DRangeSeries(expr, (x, 0, 1.5), (y, 0, 2 * pi),
        n=10, adaptive=False, is_polar=False)
    s2 = SurfaceOver2DRangeSeries(expr, (x, 0, 1.5), (y, 0, 2 * pi),
        n=10, adaptive=False, is_polar=True)
    x1, y1, z1 = s1.get_data()  # 获取 s1 的数据：x 坐标、y 坐标、z 坐标
    x2, y2, z2 = s2.get_data()  # 获取 s2 的数据：x 坐标、y 坐标、z 坐标
    x22, y22 = x1 * np.cos(y1), x1 * np.sin(y1)
    assert np.allclose(x2, x22)  # 断言：x2 与 x22 完全相等
    assert np.allclose(y2, y22)  # 断言：y2 与 y22 完全相等


def test_color_func():
    # verify that eval_color_func produces the expected results in order to
    # maintain back compatibility with the old sympy.plotting module
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x, y, z, u, v = symbols("x, y, z, u, v")

    # color func: returns x, y, color and s is parametric
    xx = np.linspace(-3, 3, 10)
    yy1 = np.cos(xx)
    s = List2DSeries(xx, yy1, color_func=lambda x, y: 2 * x, use_cm=True)
    xxs, yys, col = s.get_data()  # 获取 s 的数据：x 坐标、y 坐标、颜色值
    assert np.allclose(xx, xxs)  # 断言：xx 与 xxs 完全相等
    assert np.allclose(yy1, yys)  # 断言：yy1 与 yys 完全相等
    assert np.allclose(2 * xx, col)  # 断言：2 * xx 与 col 完全相等
    assert s.is_parametric  # 断言：s 是参数化的

    s = List2DSeries(xx, yy1, color_func=lambda x, y: 2 * x, use_cm=False)
    assert len(s.get_data()) == 2  # 断言：s 的数据长度为 2
    assert not s.is_parametric  # 断言：s 不是参数化的

    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: t)
    xx, yy, col = s.get_data()  # 获取 s 的数据：x 坐标、y 坐标、颜色值
    assert (not np.allclose(xx, col)) and (not np.allclose(yy, col))  # 断言：xx 与 col 不完全相等，yy 与 col 不完全相等
    # 创建一个二维参数化线系列对象，使用余弦函数和正弦函数作为参数化函数，定义在区间 [0, 2π]
    # 参数设置为非自适应，n=10 表示采样点数，color_func 定义了颜色函数为 x * y
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y: x * y)
    # 获取参数化线系列对象的数据 xx, yy, col
    xx, yy, col = s.get_data()
    # 断言：验证 col 是否近似等于 xx * yy 的每个元素
    assert np.allclose(col, xx * yy)

    # 创建另一个二维参数化线系列对象，使用相同的参数化函数，但颜色函数定义为 x * y * t
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, t: x * y * t)
    # 获取参数化线系列对象的数据 xx, yy, col
    xx, yy, col = s.get_data()
    # 断言：验证 col 是否近似等于 xx * yy * linspace(0, 2π, 10) 的每个元素
    assert np.allclose(col, xx * yy * np.linspace(0, 2*np.pi, 10))

    # 创建一个三维参数化线系列对象，使用余弦函数和正弦函数作为参数化函数，x 作为颜色函数的参数
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: t)
    # 获取参数化线系列对象的数据 xx, yy, zz, col
    xx, yy, zz, col = s.get_data()
    # 断言：验证 col 是否不近似等于 xx 和 yy 的每个元素
    # 即 col 和 xx, yy 不完全相同
    assert (not np.allclose(xx, col)) and (not np.allclose(yy, col))

    # 创建另一个三维参数化线系列对象，使用相同的参数化函数和颜色函数定义为 x * y * z
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, z: x * y * z)
    # 获取参数化线系列对象的数据 xx, yy, zz, col
    xx, yy, zz, col = s.get_data()
    # 断言：验证 col 是否近似等于 xx * yy * zz 的每个元素
    assert np.allclose(col, xx * yy * zz)

    # 创建另一个三维参数化线系列对象，使用相同的参数化函数和颜色函数定义为 x * y * z * t
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda x, y, z, t: x * y * z * t)
    # 获取参数化线系列对象的数据 xx, yy, zz, col
    xx, yy, zz, col = s.get_data()
    # 断言：验证 col 是否近似等于 xx * yy * zz * linspace(0, 2π, 10) 的每个元素
    assert np.allclose(col, xx * yy * zz * np.linspace(0, 2*np.pi, 10))

    # 创建一个二维参数化曲面系列对象，使用 cos(x^2 + y^2) 作为参数化函数，定义在区间 [-2, 2] x [-2, 2]
    # 参数设置为非自适应，n1=10, n2=10 表示采样点数，color_func 定义了颜色函数为 x
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x: x)
    # 获取参数化曲面系列对象的数据 xx, yy, zz
    xx, yy, zz = s.get_data()
    # 计算颜色函数在每个点的值
    col = s.eval_color_func(xx, yy, zz)
    # 断言：验证 xx 是否近似等于 col 的每个元素
    assert np.allclose(xx, col)

    # 创建另一个二维参数化曲面系列对象，使用相同的参数化函数，但颜色函数定义为 x * y
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x, y: x * y)
    # 获取参数化曲面系列对象的数据 xx, yy, zz
    xx, yy, zz = s.get_data()
    # 计算颜色函数在每个点的值
    col = s.eval_color_func(xx, yy, zz)
    # 断言：验证 xx * yy 是否近似等于 col 的每个元素
    assert np.allclose(xx * yy, col)

    # 创建另一个二维参数化曲面系列对象，使用相同的参数化函数，但颜色函数定义为 x * y * z
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x, y, z: x * y * z)
    # 获取参数化曲面系列对象的数据 xx, yy, zz
    xx, yy, zz = s.get_data()
    # 计算颜色函数在每个点的值
    col = s.eval_color_func(xx, yy, zz)
    # 断言：验证 xx * yy * zz 是否近似等于 col 的每个元素
    assert np.allclose(xx * yy * zz, col)

    # 创建另一个二维参数化曲面系列对象，使用相同的参数化函数，但颜色函数定义为 x * y * z * u * v
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x, y, z, u, v: x * y * z * u * v)
    # 获取参数化曲面系列对象的数据 xx, yy, zz
    xx, yy, zz = s.get_data()
    # 计算颜色函数在每个点的值
    col = s.eval_color_func(xx, yy, zz)
    # 断言：验证 xx * yy * zz * uu * vv 是否近似等于 col 的每个元素
    assert np.allclose(xx * yy * zz * uu * vv, col)
    # 创建一个 List2DSeries 对象，用指定的 x 值和列表作为数据点，使用指定的颜色函数和参数，启用色彩映射
    s = List2DSeries([0, 1, 2, x], [x, 2, 3, 4],
                     color_func=lambda x, y: 2 * x, params={x: 1}, use_cm=True)
    # 获取 List2DSeries 对象的数据：x 值、y 值和颜色值
    xx, yy, col = s.get_data()
    # 断言 x 值数组与预期值 [0, 1, 2, 1] 接近
    assert np.allclose(xx, [0, 1, 2, 1])
    # 断言 y 值数组与预期值 [x, 2, 3, 4] 接近
    assert np.allclose(yy, [x, 2, 3, 4])
    # 断言颜色值数组是 2 倍的 x 值数组
    assert np.allclose(2 * xx, col)
    # 断言 List2DSeries 对象是参数化的并且启用了色彩映射
    assert s.is_parametric and s.use_cm

    # 创建另一个 List2DSeries 对象，使用相同的 x 值和列表作为数据点，但不启用色彩映射
    s = List2DSeries([0, 1, 2, x], [x, 2, 3, 4],
                     color_func=lambda x, y: 2 * x, params={x: 1}, use_cm=False)
    # 断言获取的数据元组长度为 2
    assert len(s.get_data()) == 2
    # 断言 List2DSeries 对象不是参数化的
    assert not s.is_parametric
def test_color_func_scalar_val():
    # 验证 eval_color_func 返回一个 numpy 数组，即使 color_func 返回一个标量值
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x, y = symbols("x, y")

    # 创建一个 Parametric2DLineSeries 实例，设置 color_func 返回固定值 1
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: 1)
    xx, yy, col = s.get_data()
    assert np.allclose(col, np.ones(xx.shape))  # 断言颜色数组 col 的所有值都接近于 1

    # 创建一个 Parametric3DLineSeries 实例，设置 color_func 返回固定值 1
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 2*pi),
        adaptive=False, n=10, color_func=lambda t: 1)
    xx, yy, zz, col = s.get_data()
    assert np.allclose(col, np.ones(xx.shape))  # 断言颜色数组 col 的所有值都接近于 1

    # 创建一个 SurfaceOver2DRangeSeries 实例，设置 color_func 返回固定值 1
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        adaptive=False, n1=10, n2=10, color_func=lambda x: 1)
    xx, yy, zz = s.get_data()
    assert np.allclose(s.eval_color_func(xx), np.ones(xx.shape))  # 断言 eval_color_func 返回的颜色数组与 1 接近

    # 创建一个 ParametricSurfaceSeries 实例，设置 color_func 返回固定值 1
    s = ParametricSurfaceSeries(1, x, y, (x, 0, 1), (y, 0, 1), adaptive=False,
        n1=10, n2=10, color_func=lambda u: 1)
    xx, yy, zz, uu, vv = s.get_data()
    col = s.eval_color_func(xx, yy, zz, uu, vv)
    assert np.allclose(col, np.ones(xx.shape))  # 断言颜色数组 col 的所有值都接近于 1


def test_color_func_expression():
    # 验证 color_func 能够处理 Expr 实例：它们将使用与主表达式相同的签名进行 lambdify。
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x, y = symbols("x, y")

    # 创建 Parametric2DLineSeries 实例，设置 color_func 为 sin(x)
    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        color_func=sin(x), adaptive=False, n=10, use_cm=True)
    # 创建 Parametric2DLineSeries 实例，设置 color_func 为 lambda 表达式 np.cos(x)
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        color_func=lambda x: np.cos(x), adaptive=False, n=10, use_cm=True)
    # 断言 color_func 是可调用的，并且两个实例的最后一个数据数组不全等
    d1 = s1.get_data()
    assert callable(s1.color_func)
    d2 = s2.get_data()
    assert not np.allclose(d1[-1], d2[-1])

    # 创建 SurfaceOver2DRangeSeries 实例，设置 color_func 为 sin(x**2 + y**2)
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        color_func=sin(x**2 + y**2), adaptive=False, n1=5, n2=5)
    # 断言 color_func 是可调用的
    s.get_data()
    assert callable(s.color_func)

    xx = [1, 2, 3, 4, 5]
    yy = [1, 2, 3, 4, 5]
    # 断言调用 List2DSeries 时会引发 TypeError，其中的 color_func 为 sin(x)


def test_line_surface_color():
    # 验证与旧 sympy.plotting 模块的后向兼容性。
    # 通过将 line_color 或 surface_color 设置为可调用函数，将设置 color_func 属性。

    x, y, z = symbols("x, y, z")

    # 创建 LineOver1DRangeSeries 实例，设置 line_color 为 lambda 表达式 x
    s = LineOver1DRangeSeries(sin(x), (x, -5, 5), adaptive=False, n=10,
        line_color=lambda x: x)
    assert (s.line_color is None) and callable(s.color_func)

    # 创建 Parametric2DLineSeries 实例，设置 line_color 为 lambda 表达式 t
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 2*pi),
        adaptive=False, n=10, line_color=lambda t: t)
    assert (s.line_color is None) and callable(s.color_func)

    # 创建 SurfaceOver2DRangeSeries 实例，设置 surface_color 为 lambda 表达式 x
    s = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -2, 2), (y, -2, 2),
        n1=10, n2=10, surface_color=lambda x: x)
    assert (s.surface_color is None) and callable(s.color_func)
def test_complex_adaptive_false():
    # 验证在 adaptive=False 的情况下，带有复数类型的离散化范围的系列是否被评估。
    # 检查是否安装了 numpy，如果没有则跳过测试。
    if not np:
        skip("numpy not installed.")

    x, y, u = symbols("x y u")

    def do_test(data1, data2):
        # 断言两个数据列表长度相同
        assert len(data1) == len(data2)
        # 逐对检查数据列表中的元素是否全部接近
        for d1, d2 in zip(data1, data2):
            assert np.allclose(d1, d2)

    expr1 = sqrt(x) * exp(-x**2)
    expr2 = sqrt(u * x) * exp(-x**2)

    # 创建 LineOver1DRangeSeries 对象 s1 和 s2，设置 adaptive=False 和参数 n=10
    s1 = LineOver1DRangeSeries(im(expr1), (x, -5, 5), adaptive=False, n=10)
    s2 = LineOver1DRangeSeries(im(expr2), (x, -5, 5),
        adaptive=False, n=10, params={u: 1})
    # 获取数据并进行测试
    data1 = s1.get_data()
    data2 = s2.get_data()
    do_test(data1, data2)
    # 断言数据列表中索引为1的元素不全部接近0
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))

    # 创建 Parametric2DLineSeries 对象 s1 和 s2，设置 adaptive=False 和参数 n=10
    s1 = Parametric2DLineSeries(re(expr1), im(expr1), (x, -pi, pi),
        adaptive=False, n=10)
    s2 = Parametric2DLineSeries(re(expr2), im(expr2), (x, -pi, pi),
        adaptive=False, n=10, params={u: 1})
    # 获取数据并进行测试
    data1 = s1.get_data()
    data2 = s2.get_data()
    do_test(data1, data2)
    # 断言数据列表中索引为1的元素不全部接近0
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))

    # 创建 SurfaceOver2DRangeSeries 对象 s1 和 s2，设置 adaptive=False 和参数 n1=30, n2=3
    s1 = SurfaceOver2DRangeSeries(im(expr1), (x, -5, 5), (y, -10, 10),
        adaptive=False, n1=30, n2=3)
    s2 = SurfaceOver2DRangeSeries(im(expr2), (x, -5, 5), (y, -10, 10),
        adaptive=False, n1=30, n2=3, params={u: 1})
    # 获取数据并进行测试
    data1 = s1.get_data()
    data2 = s2.get_data()
    do_test(data1, data2)
    # 断言数据列表中索引为1的元素不全部接近0
    assert (not np.allclose(data1[1], 0)) and (not np.allclose(data2[1], 0))


def test_expr_is_lambda_function():
    # 验证当提供 numpy 函数时，系列能够进行评估。
    # 同时，为了防止某些后端崩溃，标签应为空。
    if not np:
        skip("numpy not installed.")

    # 定义 lambda 函数 f
    f = lambda x: np.cos(x)
    # 创建 LineOver1DRangeSeries 对象 s1 和 s2，设置 adaptive=True 和深度 depth=3
    s1 = LineOver1DRangeSeries(f, ("x", -5, 5), adaptive=True, depth=3)
    s1.get_data()
    # 创建 LineOver1DRangeSeries 对象 s2，设置 adaptive=False 和参数 n=10
    s2 = LineOver1DRangeSeries(f, ("x", -5, 5), adaptive=False, n=10)
    s2.get_data()
    # 断言 s1 和 s2 的标签均为空字符串
    assert s1.label == s2.label == ""

    # 定义 lambda 函数 fx 和 fy
    fx = lambda x: np.cos(x)
    fy = lambda x: np.sin(x)
    # 创建 Parametric2DLineSeries 对象 s1 和 s2，设置 adaptive=True 和 adaptive_goal=0.1
    s1 = Parametric2DLineSeries(fx, fy, ("x", 0, 2*pi),
        adaptive=True, adaptive_goal=0.1)
    s1.get_data()
    # 创建 Parametric2DLineSeries 对象 s2，设置 adaptive=False 和参数 n=10
    s2 = Parametric2DLineSeries(fx, fy, ("x", 0, 2*pi),
        adaptive=False, n=10)
    s2.get_data()
    # 断言 s1 和 s2 的标签均为空字符串
    assert s1.label == s2.label == ""

    # 定义 lambda 函数 fz
    fz = lambda x: x
    # 创建 Parametric3DLineSeries 对象 s1 和 s2，设置 adaptive=True 和 adaptive_goal=0.1
    s1 = Parametric3DLineSeries(fx, fy, fz, ("x", 0, 2*pi),
        adaptive=True, adaptive_goal=0.1)
    s1.get_data()
    # 创建 Parametric3DLineSeries 对象 s2，设置 adaptive=False 和参数 n=10
    s2 = Parametric3DLineSeries(fx, fy, fz, ("x", 0, 2*pi),
        adaptive=False, n=10)
    s2.get_data()
    # 断言 s1 和 s2 的标签均为空字符串
    assert s1.label == s2.label == ""

    # 定义 lambda 函数 f
    f = lambda x, y: np.cos(x**2 + y**2)
    # 创建 SurfaceOver2DRangeSeries 对象 s1 和 ContourSeries 对象 s2，设置 adaptive=False 和参数 n1=10, n2=10
    s1 = SurfaceOver2DRangeSeries(f, ("a", -2, 2), ("b", -3, 3),
        adaptive=False, n1=10, n2=10)
    s1.get_data()
    s2 = ContourSeries(f, ("a", -2, 2), ("b", -3, 3),
        adaptive=False, n1=10, n2=10)
    s2.get_data()
    # 断言 s1 和 s2 的标签均为空字符串
    assert s1.label == s2.label == ""
    # 定义一个 lambda 函数 fx，表示参数化表面的 x 坐标
    fx = lambda u, v: np.cos(u + v)
    
    # 定义一个 lambda 函数 fy，表示参数化表面的 y 坐标
    fy = lambda u, v: np.sin(u - v)
    
    # 定义一个 lambda 函数 fz，表示参数化表面的 z 坐标
    fz = lambda u, v: u * v
    
    # 创建 ParametricSurfaceSeries 对象 s1，用来表示参数化表面
    # fx, fy, fz 分别为参数化表面的 x、y、z 坐标函数
    # ("u", 0, pi) 和 ("v", 0, 2*pi) 是参数范围
    # adaptive=False 表示不使用自适应采样
    # n1=10 和 n2=10 分别表示 u 和 v 方向上的采样点数量
    s1 = ParametricSurfaceSeries(fx, fy, fz, ("u", 0, pi), ("v", 0, 2*pi),
                                adaptive=False, n1=10, n2=10)
    
    # 获取参数化表面 s1 的数据
    s1.get_data()
    
    # 断言参数化表面 s1 的标签为空字符串
    assert s1.label == ""
    
    # 检查 List2DSeries 构造函数是否会引发 TypeError 异常，应当抛出异常
    raises(TypeError, lambda: List2DSeries(lambda t: t, lambda t: t))
    
    # 检查 ImplicitSeries 构造函数是否会引发 TypeError 异常，应当抛出异常
    raises(TypeError, lambda : ImplicitSeries(lambda t: np.sin(t),
                                             ("x", -5, 5), ("y", -6, 6)))
def test_show_in_legend_lines():
    # verify that lines series correctly set the show_in_legend attribute

    # 导入符号计算中的符号变量 x, u
    x, u = symbols("x, u")

    # 创建一维范围上的线性系列对象 s，设置 show_in_legend 为 True，并断言其为 True
    s = LineOver1DRangeSeries(cos(x), (x, -2, 2), "test", show_in_legend=True)
    assert s.show_in_legend
    # 创建一维范围上的线性系列对象 s，设置 show_in_legend 为 False，并断言其为 False
    s = LineOver1DRangeSeries(cos(x), (x, -2, 2), "test", show_in_legend=False)
    assert not s.show_in_legend

    # 创建二维参数化线性系列对象 s，设置 show_in_legend 为 True，并断言其为 True
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), "test",
        show_in_legend=True)
    assert s.show_in_legend
    # 创建二维参数化线性系列对象 s，设置 show_in_legend 为 False，并断言其为 False
    s = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), "test",
        show_in_legend=False)
    assert not s.show_in_legend

    # 创建三维参数化线性系列对象 s，设置 show_in_legend 为 True，并断言其为 True
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1), "test",
        show_in_legend=True)
    assert s.show_in_legend
    # 创建三维参数化线性系列对象 s，设置 show_in_legend 为 False，并断言其为 False
    s = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1), "test",
        show_in_legend=False)
    assert not s.show_in_legend


@XFAIL
def test_particular_case_1_with_adaptive_true():
    # Verify that symbolic expressions and numerical lambda functions are
    # evaluated with the same algorithm.
    if not np:
        skip("numpy not installed.")

    # NOTE: xfail because sympy's adaptive algorithm is not deterministic

    # 定义测试函数 do_test，比较两个对象的数据，检查是否有警告信息
    def do_test(a, b):
        with warns(
            RuntimeWarning,
            match="invalid value encountered in scalar power",
            test_stacklevel=False,
        ):
            d1 = a.get_data()
            d2 = b.get_data()
            for t, v in zip(d1, d2):
                assert np.allclose(t, v)

    # 定义符号变量 n
    n = symbols("n")
    # 设定常量 a 和 epsilon
    a = S(2) / 3
    epsilon = 0.01
    # 定义符号表达式 xn
    xn = (n**3 + n**2)**(S(1)/3) - (n**3 - n**2)**(S(1)/3)
    expr = Abs(xn - a) - epsilon
    # 创建一维范围上的线性系列对象 s1，使用 adaptive=True 和 depth=3
    s1 = LineOver1DRangeSeries(expr, (n, -10, 10), "",
        adaptive=True, depth=3)
    # 创建数值 lambda 函数的线性系列对象 s2，使用 adaptive=True 和 depth=3
    s2 = LineOver1DRangeSeries(math_func, ("n", -10, 10), "",
        adaptive=True, depth=3)
    # 执行测试函数 do_test
    do_test(s1, s2)


def test_particular_case_1_with_adaptive_false():
    # Verify that symbolic expressions and numerical lambda functions are
    # evaluated with the same algorithm. In particular, uniform evaluation
    # is going to use np.vectorize, which correctly evaluates the following
    # mathematical function.
    if not np:
        skip("numpy not installed.")

    # 定义测试函数 do_test，比较两个对象的数据
    def do_test(a, b):
        d1 = a.get_data()
        d2 = b.get_data()
        for t, v in zip(d1, d2):
            assert np.allclose(t, v)

    # 定义符号变量 n
    n = symbols("n")
    # 设定常量 a 和 epsilon
    a = S(2) / 3
    epsilon = 0.01
    # 定义符号表达式 xn
    xn = (n**3 + n**2)**(S(1)/3) - (n**3 - n**2)**(S(1)/3)
    expr = Abs(xn - a) - epsilon
    # 创建一维范围上的线性系列对象 s3，使用 adaptive=False 和 n=10
    s3 = LineOver1DRangeSeries(expr, (n, -10, 10), "",
        adaptive=False, n=10)
    # 创建数值 lambda 函数的线性系列对象 s4，使用 adaptive=False 和 n=10
    s4 = LineOver1DRangeSeries(math_func, ("n", -10, 10), "",
        adaptive=False, n=10)
    # 执行测试函数 do_test
    do_test(s3, s4)


def test_complex_params_number_eval():
    # The main expression contains terms like sqrt(xi - 1), with
    # parameter (0 <= xi <= 1).
    # There shouldn't be any NaN values on the output.
    # 检查是否导入了 numpy 库，如果没有则跳过执行并输出提示信息
    if not np:
        skip("numpy not installed.")

    # 定义符号变量 xi, wn, x0, v0, t，并创建一个关于 t 的函数 x
    xi, wn, x0, v0, t = symbols("xi, omega_n, x0, v0, t")
    x = Function("x")(t)
    # 构建微分方程，包括二阶导数、一阶导数和常数项
    eq = x.diff(t, 2) + 2 * xi * wn * x.diff(t) + wn**2 * x
    # 求解微分方程，并指定初始条件
    sol = dsolve(eq, x, ics={x.subs(t, 0): x0, x.diff(t).subs(t, 0): v0})
    # 定义参数字典，包括 wn, xi, x0, v0 的值
    params = {
        wn: 0.5,
        xi: 0.25,
        x0: 0.45,
        v0: 0.0
    }
    # 创建一个 LineOver1DRangeSeries 对象，用于绘制解 sol.rhs 的曲线
    s = LineOver1DRangeSeries(sol.rhs, (t, 0, 100), adaptive=False, n=5,
        params=params)
    # 获取曲线数据
    x, y = s.get_data()
    # 断言确保曲线数据中没有 NaN 值
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()


    # 锯齿波的傅里叶级数
    # 主要表达式包含一个求和，其上限为符号变量 m
    # lambdify 的代码看起来像:
    #       sum(blablabla for for n in range(1, m+1))
    # 但是 range 要求整数参数，而根据上面的示例，该级数将参数转换为复数。验证级数能够检测到
    # 求和的上限，并将其转换为整数以成功进行求值。
    x, T, n, m = symbols("x, T, n, m")
    # 定义锯齿波的傅里叶级数表达式
    fs = S(1) / 2 - (1 / pi) * Sum(sin(2 * n * pi * x / T) / n, (n, 1, m))
    # 定义参数字典，包括 T 和 m 的值
    params = {
        T: 4.5,
        m: 5
    }
    # 创建一个 LineOver1DRangeSeries 对象，用于绘制 fs 的曲线
    s = LineOver1DRangeSeries(fs, (x, 0, 10), adaptive=False, n=5,
        params=params)
    # 获取曲线数据
    x, y = s.get_data()
    # 断言确保曲线数据中没有 NaN 值
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()
def test_complex_range_line_plot_1():
    # 验证在复杂数据范围（虚部为零）下评估单变量函数。输出中不应包含任何 NaN 值。
    if not np:
        skip("numpy not installed.")

    # 符号定义
    x, u = symbols("x, u")
    
    # 表达式计算
    expr1 = im(sqrt(x) * exp(-x**2))
    expr2 = im(sqrt(u * x) * exp(-x**2))
    
    # 创建线性数据序列对象，自适应方式
    s1 = LineOver1DRangeSeries(expr1, (x, -10, 10), adaptive=True,
        adaptive_goal=0.1)
    
    # 创建线性数据序列对象，非自适应方式，指定点数
    s2 = LineOver1DRangeSeries(expr1, (x, -10, 10), adaptive=False, n=30)
    
    # 创建线性数据序列对象，非自适应方式，指定参数
    s3 = LineOver1DRangeSeries(expr2, (x, -10, 10), adaptive=False, n=30,
        params={u: 1})
    
    # 忽略运行时警告
    with ignore_warnings(RuntimeWarning):
        # 获取数据
        data1 = s1.get_data()
    data2 = s2.get_data()
    data3 = s3.get_data()

    # 断言：数据中不存在 NaN 值
    assert not np.isnan(data1[1]).any()
    assert not np.isnan(data2[1]).any()
    assert not np.isnan(data3[1]).any()
    
    # 断言：数据的前两列相似
    assert np.allclose(data2[0], data3[0]) and np.allclose(data2[1], data3[1])


@XFAIL
def test_complex_range_line_plot_2():
    # 验证在复杂数据范围（虚部非零）下评估单变量函数。输出中不应包含任何 NaN 值。
    if not np:
        skip("numpy not installed.")

    # 注意：xfail，因为 sympy 的自适应算法无法处理复数。

    # 符号定义
    x, u = symbols("x, u")

    # 自适应和均匀网格化应该产生相同的数据。
    # 由于自适应性质，比较两个序列的第一个和最后一个点。
    s1 = LineOver1DRangeSeries(abs(sqrt(x)), (x, -5-2j, 5-2j), adaptive=True)
    s2 = LineOver1DRangeSeries(abs(sqrt(x)), (x, -5-2j, 5-2j), adaptive=False,
        n=10)
    
    # 忽略特定的运行时警告
    with warns(
            RuntimeWarning,
            match="invalid value encountered in sqrt",
            test_stacklevel=False,
        ):
        # 获取数据
        d1 = s1.get_data()
        d2 = s2.get_data()
        xx1 = [d1[0][0], d1[0][-1]]
        xx2 = [d2[0][0], d2[0][-1]]
        yy1 = [d1[1][0], d1[1][-1]]
        yy2 = [d2[1][0], d2[1][-1]]
        
        # 断言：前后两列数据近似相等
        assert np.allclose(xx1, xx2)
        assert np.allclose(yy1, yy2)


def test_force_real_eval():
    # 验证 force_real_eval=True 在与复数域评估比较时会产生不一致的结果。
    if not np:
        skip("numpy not installed.")

    # 符号定义
    x = symbols("x")

    # 表达式计算
    expr = im(sqrt(x) * exp(-x**2))
    
    # 创建线性数据序列对象，非自适应方式，指定点数和参数
    s1 = LineOver1DRangeSeries(expr, (x, -10, 10), adaptive=False, n=10,
        force_real_eval=False)
    
    # 创建线性数据序列对象，非自适应方式，指定点数和参数，强制实数评估
    s2 = LineOver1DRangeSeries(expr, (x, -10, 10), adaptive=False, n=10,
        force_real_eval=True)
    
    # 获取数据
    d1 = s1.get_data()
    with ignore_warnings(RuntimeWarning):
        d2 = s2.get_data()
    
    # 断言：第一列数据不全近似为零，第二列数据全近似为零
    assert not np.allclose(d1[1], 0)
    assert np.allclose(d2[1], 0)


def test_contour_series_show_clabels():
    # 验证等高线系列能够设置标签对等高线的可见性

    # 符号定义
    x, y = symbols("x, y")
    
    # 创建等高线数据序列对象
    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2))
    
    # 断言：等高线显示标签
    assert s.show_clabels
    # 创建一个 ContourSeries 对象 s，该对象显示 x*y 的余弦函数在给定范围内的等高线图，并显示标签
    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2), clabels=True)
    # 断言检查，确保等高线图 s 显示标签
    assert s.show_clabels
    
    # 创建一个 ContourSeries 对象 s，该对象显示 x*y 的余弦函数在给定范围内的等高线图，并不显示标签
    s = ContourSeries(cos(x*y), (x, -2, 2), (y, -2, 2), clabels=False)
    # 断言检查，确保等高线图 s 不显示标签
    assert not s.show_clabels
def test_LineOver1DRangeSeries_complex_range():
    # 验证 LineOver1DRangeSeries 能够接受复杂的范围
    # 如果起始值和结束值的虚部相同

    x = symbols("x")

    # 创建一个基于一维范围的数据系列，函数为 sqrt(x)，范围为 x 从 -10 到 10
    LineOver1DRangeSeries(sqrt(x), (x, -10, 10))
    # 创建一个基于一维范围的数据系列，函数为 sqrt(x)，范围为 x 从 -10-2j 到 10-2j
    LineOver1DRangeSeries(sqrt(x), (x, -10-2j, 10-2j))
    # 预期会引发 ValueError 错误，因为复数范围的起始值和结束值的虚部不同
    raises(ValueError,
        lambda : LineOver1DRangeSeries(sqrt(x), (x, -10-2j, 10+2j)))


def test_symbolic_plotting_ranges():
    # 验证数据系列能够使用符号化的绘图范围
    if not np:
        # 如果 numpy 没有安装，则跳过测试
        skip("numpy not installed.")

    x, y, z, a, b = symbols("x, y, z, a, b")

    def do_test(s1, s2, new_params):
        # 执行测试函数，比较两个数据系列的数据是否相似
        d1 = s1.get_data()
        d2 = s2.get_data()
        for u, v in zip(d1, d2):
            assert np.allclose(u, v)
        # 设置 s2 的参数为 new_params
        s2.params = new_params
        d2 = s2.get_data()
        for u, v in zip(d1, d2):
            assert not np.allclose(u, v)

    # 创建一个基于一维范围的数据系列，函数为 sin(x)，范围为 x 从 0 到 1，不自适应，点数为 10
    s1 = LineOver1DRangeSeries(sin(x), (x, 0, 1), adaptive=False, n=10)
    # 创建一个基于一维范围的数据系列，函数为 sin(x)，范围为 x 从 a 到 b，参数为 {a: 0, b: 1}，不自适应，点数为 10
    s2 = LineOver1DRangeSeries(sin(x), (x, a, b), params={a: 0, b: 1},
        adaptive=False, n=10)
    # 执行数据比较测试，参数为 {a: 0.5, b: 1.5}
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # 缺少一个参数，预期会引发 ValueError 错误
    raises(ValueError,
        lambda : LineOver1DRangeSeries(sin(x), (x, a, b), params={a: 1}, n=10))

    # 创建一个二维参数化曲线数据系列，参数为 cos(x), sin(x)，范围为 x 从 0 到 1，不自适应，点数为 10
    s1 = Parametric2DLineSeries(cos(x), sin(x), (x, 0, 1), adaptive=False, n=10)
    # 创建一个二维参数化曲线数据系列，参数为 cos(x), sin(x)，范围为 x 从 a 到 b，参数为 {a: 0, b: 1}，不自适应，点数为 10
    s2 = Parametric2DLineSeries(cos(x), sin(x), (x, a, b), params={a: 0, b: 1},
        adaptive=False, n=10)
    # 执行数据比较测试，参数为 {a: 0.5, b: 1.5}
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # 缺少一个参数，预期会引发 ValueError 错误
    raises(ValueError,
        lambda : Parametric2DLineSeries(cos(x), sin(x), (x, a, b),
            params={a: 0}, adaptive=False, n=10))

    # 创建一个三维参数化曲线数据系列，参数为 cos(x), sin(x), x，范围为 x 从 0 到 1，不自适应，点数为 10
    s1 = Parametric3DLineSeries(cos(x), sin(x), x, (x, 0, 1),
        adaptive=False, n=10)
    # 创建一个三维参数化曲线数据系列，参数为 cos(x), sin(x), x，范围为 x 从 a 到 b，参数为 {a: 0, b: 1}，不自适应，点数为 10
    s2 = Parametric3DLineSeries(cos(x), sin(x), x, (x, a, b),
        params={a: 0, b: 1}, adaptive=False, n=10)
    # 执行数据比较测试，参数为 {a: 0.5, b: 1.5}
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # 缺少一个参数，预期会引发 ValueError 错误
    raises(ValueError,
        lambda : Parametric3DLineSeries(cos(x), sin(x), x, (x, a, b),
            params={a: 0}, adaptive=False, n=10))

    # 创建一个二维参数化表面数据系列，参数为 cos(x**2 + y**2)，范围为 x 从 -pi 到 pi，y 从 -pi 到 pi，不自适应，点数为 5x5
    s1 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi),
        adaptive=False, n1=5, n2=5)
    # 创建一个二维参数化表面数据系列，参数为 cos(x**2 + y**2)，范围为 x 从 -pi*a 到 pi*a，y 从 -pi*b 到 pi*b，参数为 {a: 1, b: 1}，不自适应，点数为 5x5
    s2 = SurfaceOver2DRangeSeries(cos(x**2 + y**2), (x, -pi * a, pi * a),
        (y, -pi * b, pi * b), params={a: 1, b: 1},
        adaptive=False, n1=5, n2=5)
    # 执行数据比较测试，参数为 {a: 0.5, b: 1.5}
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # 缺少一个参数，预期会引发 ValueError 错误
    raises(ValueError,
        lambda : SurfaceOver2DRangeSeries(cos(x**2 + y**2),
        (x, -pi * a, pi * a), (y, -pi * b, pi * b), params={a: 1},
        adaptive=False, n1=5, n2=5))
    # 预期会引发 ValueError 错误，因为一个范围符号包含在另一个范围的最小或最大值内
    raises(ValueError,
        lambda : SurfaceOver2DRangeSeries(cos(x**2 + y**2),
        (x, -pi * a + y, pi * a), (y, -pi * b, pi * b), params={a: 1},
        adaptive=False, n1=5, n2=5))

    # 创建一个二维参数化表面数据系列，参数为 cos(x - y)，sin(x + y)，范围为 x 从 -2 到 2，y 从 -2 到 2，点数为 5x5
    s1 = ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2, 2), (y, -2, 2), n1=5, n2=5)
    # 创建第二个 ParametricSurfaceSeries 对象，定义了参数和范围
    s2 = ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2 * a, 2), (y, -2, 2 * b),
        params={a: 1, b: 1}, n1=5, n2=5)
    # 执行测试函数，比较两个 ParametricSurfaceSeries 对象的输出
    do_test(s1, s2, {a: 0.5, b: 1.5})

    # 引发 ValueError 异常，因为缺少参数 b
    raises(ValueError,
        lambda : ParametricSurfaceSeries(
        cos(x - y), sin(x + y), x - y, (x, -2 * a, 2), (y, -2, 2 * b),
        params={a: 1}, n1=5, n2=5))
def test_exclude_points():
    # verify that exclude works as expected
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x = symbols("x")  # 定义符号变量 x

    expr = (floor(x) + S.Half) / (1 - (x - S.Half)**2)  # 定义表达式 expr

    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some",
            test_stacklevel=False,
        ):
        # 创建 LineOver1DRangeSeries 对象 s，排除指定范围的点
        s = LineOver1DRangeSeries(expr, (x, -3.5, 3.5), adaptive=False, n=100,
            exclude=list(range(-3, 4)))
        xx, yy = s.get_data()  # 获取数据点 xx 和 yy
        assert not np.isnan(xx).any()  # 确保 xx 中没有 NaN 值
        assert np.count_nonzero(np.isnan(yy)) == 7  # 确保 yy 中有 7 个 NaN 值
        assert len(xx) > 100  # 确保 xx 的长度大于 100

    e1 = log(floor(x)) * cos(x)  # 定义 e1 表达式
    e2 = log(floor(x)) * sin(x)  # 定义 e2 表达式
    with warns(
            UserWarning,
            match="NumPy is unable to evaluate with complex numbers some",
            test_stacklevel=False,
        ):
        # 创建 Parametric2DLineSeries 对象 s，排除指定范围的点
        s = Parametric2DLineSeries(e1, e2, (x, 1, 12), adaptive=False, n=100,
            exclude=list(range(1, 13)))
        xx, yy, pp = s.get_data()  # 获取数据点 xx, yy 和 pp
        assert not np.isnan(pp).any()  # 确保 pp 中没有 NaN 值
        assert np.count_nonzero(np.isnan(xx)) == 11  # 确保 xx 中有 11 个 NaN 值
        assert np.count_nonzero(np.isnan(yy)) == 11  # 确保 yy 中有 11 个 NaN 值
        assert len(xx) > 100  # 确保 xx 的长度大于 100


def test_unwrap():
    # verify that unwrap works as expected
    if not np:
        skip("numpy not installed.")  # 如果没有安装 numpy，则跳过测试

    x, y = symbols("x, y")  # 定义符号变量 x 和 y

    expr = 1 / (x**3 + 2*x**2 + x)  # 定义表达式 expr
    expr = arg(expr.subs(x, I*y*2*pi))  # 对表达式 expr 进行符号替换
    # 创建三个 LineOver1DRangeSeries 对象 s1, s2, s3，分别设置 unwrap 参数为 False, True, {"period": 4}
    s1 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap=False)
    s2 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap=True)
    s3 = LineOver1DRangeSeries(expr, (y, 1e-05, 1e05), xscale="log",
        adaptive=False, n=10, unwrap={"period": 4})
    x1, y1 = s1.get_data()  # 获取数据点 x1, y1
    x2, y2 = s2.get_data()  # 获取数据点 x2, y2
    x3, y3 = s3.get_data()  # 获取数据点 x3, y3
    assert np.allclose(x1, x2)  # 确保 x1 和 x2 在数值上相等
    # 确保这些评估结果中没有 NaN 值
    assert all(not np.isnan(t).any() for t in [y1, y2, y3])
    assert not np.allclose(y1, y2)  # 确保 y1 和 y2 在数值上不完全相等
    assert not np.allclose(y1, y3)  # 确保 y1 和 y3 在数值上不完全相等
    assert not np.allclose(y2, y3)  # 确保 y2 和 y3 在数值上不完全相等
```