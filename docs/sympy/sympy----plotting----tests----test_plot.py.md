# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_plot.py`

```
# 导入操作系统接口模块
import os
# 导入临时目录模块
from tempfile import TemporaryDirectory
# 导入 pytest 测试框架
import pytest
# 导入 Sympy 的求和模块
from sympy.concrete.summations import Sum
# 导入 Sympy 的常数模块（虚数单位 i、无穷大 oo、圆周率 pi）
from sympy.core.numbers import (I, oo, pi)
# 导入 Sympy 的不等式模块
from sympy.core.relational import Ne
# 导入 Sympy 的符号模块（单个符号、多个符号）
from sympy.core.symbol import Symbol, symbols
# 导入 Sympy 的指数函数模块（Lambert W 函数、指数函数、极坐标指数函数、对数函数）
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
# 导入 Sympy 的元素函数模块（实数根函数、平方根函数）
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
# 导入 Sympy 的分段函数模块
from sympy.functions.elementary.piecewise import Piecewise
# 导入 Sympy 的三角函数模块（余弦函数、正弦函数）
from sympy.functions.elementary.trigonometric import (cos, sin)
# 导入 Sympy 的元素函数模块（最小值函数）
from sympy.functions.elementary.miscellaneous import Min
# 导入 Sympy 的超函模块（Meijer G 函数）
from sympy.functions.special.hyper import meijerg
# 导入 Sympy 的积分模块
from sympy.integrals.integrals import Integral
# 导入 Sympy 的布尔逻辑模块（与操作）
from sympy.logic.boolalg import And
# 导入 Sympy 的单例模块
from sympy.core.singleton import S
# 导入 Sympy 的符号化函数模块
from sympy.core.sympify import sympify
# 导入 Sympy 的外部模块（导入指定模块）
from sympy.external import import_module
# 导入 Sympy 的绘图模块（绘图对象、普通绘图、参数绘图、三维参数线绘图、三维绘图、三维参数面绘图）
from sympy.plotting.plot import (
    Plot, plot, plot_parametric, plot3d_parametric_line, plot3d,
    plot3d_parametric_surface)
# 导入 Sympy 的绘图模块（取消显示、绘制等高线图、绘图网格、Matplotlib 后端、文本后端）
from sympy.plotting.plot import (
    unset_show, plot_contour, PlotGrid, MatplotlibBackend, TextBackend)
# 导入 Sympy 的绘图序列模块（一维范围序列、二维参数线序列、三维参数线序列、三维参数面序列、二维范围面序列）
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries)
# 导入 Sympy 的测试模块（跳过、警告、异常、警告过时）
from sympy.testing.pytest import skip, warns, raises, warns_deprecated_sympy
# 导入 Sympy 的工具模块（lambda 化函数、忽略警告）
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings

# 取消显示函数，用于取消显示默认绘图
unset_show()

# 尝试导入 Matplotlib 模块，如果版本不符或运行时异常则捕获并忽略
matplotlib = import_module(
    'matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))


# 定义一个用于测试的不支持显示、保存和关闭的绘图后端类
class DummyBackendNotOk(Plot):
    """ 用于验证用户是否可以创建自己的后端。
    这个后端意图上会为 `show`、`save`、`close` 方法抛出 NotImplementedError 异常。
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


# 定义一个用于测试支持显示、保存和关闭的绘图后端类
class DummyBackendOk(Plot):
    """ 用于验证用户是否可以创建自己的后端。
    这个后端意图上会通过所有的测试。
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def show(self):
        pass

    def save(self):
        pass

    def close(self):
        pass


# 测试基本的绘图后端功能
def test_basic_plotting_backend():
    x = Symbol('x')
    # 绘制 x 在 [0, 3] 范围内的图像，使用文本后端
    plot(x, (x, 0, 3), backend='text')
    # 绘制 x^2 + 1 在 [0, 3] 范围内的图像，使用文本后端
    plot(x**2 + 1, (x, 0, 3), backend='text')


# 参数化测试，测试绘图和保存功能
@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_1(adaptive):
    # 如果没有安装 Matplotlib 模块，则跳过这个测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        ###
        # 使用临时目录创建一个作为工作空间，名称以'sympy_'为前缀
        ###
        
        # 绘制函数 x 的图像，带有图例，标签为'f1'，自适应参数为adaptive，点数为10
        p = plot(x, legend=True, label='f1', adaptive=adaptive, n=10)
        
        # 绘制函数 x*sin(x) 和 x*cos(x) 的图像，各自带有标签'f2'，自适应参数为adaptive，点数为10
        p = plot(x*sin(x), x*cos(x), label='f2', adaptive=adaptive, n=10)
        
        # 将第一个图像复制一份添加到图像列表中
        p.extend(p)
        
        # 设置第一个图像的线条颜色为一种依赖于参数的函数
        p[0].line_color = lambda a: a
        
        # 设置第二个图像的线条颜色为蓝色
        p[1].line_color = 'b'
        
        # 设置图像的标题
        p.title = 'Big title'
        
        # 设置 x 轴的标签
        p.xlabel = 'the x axis'
        
        # 设置第二个图像的标签
        p[1].label = 'straight line'
        
        # 显示图例
        p.legend = True
        
        # 设置图像的长宽比为1:1
        p.aspect_ratio = (1, 1)
        
        # 设置 x 轴的显示范围
        p.xlim = (-15, 20)
        
        # 将图像保存为文件，文件名为'test_basic_options_and_colors.png'
        filename = 'test_basic_options_and_colors.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 将另一个函数的图像添加到已有图像列表中
        p.extend(plot(x + 1, adaptive=adaptive, n=10))
        
        # 将第二个函数的图像添加到图像列表的末尾
        p.append(plot(x + 3, x**2, adaptive=adaptive, n=10)[1])
        
        # 将图像保存为文件，文件名为'test_plot_extend_append.png'
        filename = 'test_plot_extend_append.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 修改图像列表中第三个图像的内容为 x^2 的函数图像
        p[2] = plot(x**2, (x, -2, 3), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_plot_setitem.png'
        filename = 'test_plot_setitem.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 绘制 sin(x) 函数在指定范围内的图像
        p = plot(sin(x), (x, -2*pi, 4*pi), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_line_explicit.png'
        filename = 'test_line_explicit.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 绘制默认范围内 sin(x) 函数的图像
        p = plot(sin(x), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_line_default_range.png'
        filename = 'test_line_default_range.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 绘制多个函数在指定范围内的图像
        p = plot((x**2, (x, -5, 5)), (x**3, (x, -3, 3)), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_line_multiple_range.png'
        filename = 'test_line_multiple_range.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 测试异常情况，预期引发 ValueError 异常
        raises(ValueError, lambda: plot(x, y))

        # 绘制分段函数的图像
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_plot_piecewise.png'
        filename = 'test_plot_piecewise.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 绘制另一个分段函数的图像
        p = plot(Piecewise((x, x < 1), (x**2, True)), (x, -3, 3), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_plot_piecewise_2.png'
        filename = 'test_plot_piecewise_2.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 测试问题 #7471
        p1 = plot(x, adaptive=adaptive, n=10)
        p2 = plot(3, adaptive=adaptive, n=10)
        
        # 将 p2 的图像添加到 p1 的图像列表中
        p1.extend(p2)
        
        # 将图像保存为文件，文件名为'test_horizontal_line.png'
        filename = 'test_horizontal_line.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()

        # 测试问题 #10925
        f = Piecewise((-1, x < -1), (x, And(-1 <= x, x < 0)), \
            (x**2, And(0 <= x, x < 1)), (x**3, x >= 1))
        
        # 绘制分段函数 f 的图像
        p = plot(f, (x, -3, 3), adaptive=adaptive, n=10)
        
        # 将图像保存为文件，文件名为'test_plot_piecewise_3.png'
        filename = 'test_plot_piecewise_3.png'
        p.save(os.path.join(tmpdir, filename))
        
        # 关闭图像后端以释放资源
        p._backend.close()
@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_2(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y, z
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_3(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y, z
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')


@pytest.mark.parametrize("adaptive", [True])
def test_plot_and_save_4(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y
    x = Symbol('x')
    y = Symbol('y')

    ###
    # Examples from the 'advanced' notebook
    ###

    # 使用临时目录创建一个上下文环境
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        # 定义积分表达式
        i = Integral(log((sin(x)**2 + 1)*sqrt(x**2 + 1)), (x, 0, y))
        # 绘制并保存图形，文件名为 'test_advanced_integral.png'
        p = plot(i, (y, 1, 5), adaptive=adaptive, n=10, force_real_eval=True)
        filename = 'test_advanced_integral.png'
        p.save(os.path.join(tmpdir, filename))
        # 关闭图形后端
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_5(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y
    x = Symbol('x')
    y = Symbol('y')

    # 使用临时目录创建一个上下文环境
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        # 定义无穷级数表达式
        s = Sum(1/x**y, (x, 1, oo))
        # 绘制并保存图形，文件名为 'test_advanced_inf_sum.png'
        p = plot(s, (y, 2, 10), adaptive=adaptive, n=10)
        filename = 'test_advanced_inf_sum.png'
        p.save(os.path.join(tmpdir, filename))
        # 关闭图形后端
        p._backend.close()

        # 绘制有限级数表达式，并保存为图像
        p = plot(Sum(1/x, (x, 1, y)), (y, 2, 10), show=False,
            adaptive=adaptive, n=10)
        # 设置只显示整数和步骤
        p[0].only_integers = True
        p[0].steps = True
        filename = 'test_advanced_fin_sum.png'

        # XXX: This should be fixed in experimental_lambdify or by using
        # ordinary lambdify so that it doesn't warn. The error results from
        # passing an array of values as the integration limit.
        #
        # UserWarning: The evaluation of the expression is problematic. We are
        # trying a failback method that may still work. Please report this as a
        # bug.
        # 忽略用户警告，保存图像
        with ignore_warnings(UserWarning):
            p.save(os.path.join(tmpdir, filename))

        # 关闭图形后端
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_6(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        filename = 'test.png'
        ###
        # 创建临时目录，并设置文件名为'test.png'用于保存绘图结果
        ###

        # 绘制 sin(x) + I*cos(x) 的图像并保存到临时目录中
        p = plot(sin(x) + I*cos(x))
        p.save(os.path.join(tmpdir, filename))

        # 忽略运行时警告，并尝试绘制 sqrt(sqrt(-x)) 的图像并保存
        with ignore_warnings(RuntimeWarning):
            p = plot(sqrt(sqrt(-x)))
            p.save(os.path.join(tmpdir, filename))

        # 绘制 LambertW(x) 的图像并保存
        p = plot(LambertW(x))
        p.save(os.path.join(tmpdir, filename))

        # 绘制 sqrt(LambertW(x)) 的图像并保存
        p = plot(sqrt(LambertW(x)))
        p.save(os.path.join(tmpdir, filename))

        # 创建关于学生 t 分布的特征函数，使用 Meijer G 函数进行计算
        x1 = 5 * x**2 * exp_polar(-I*pi)/2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5*x**2 * exp_polar(I*pi)/2
        m2 = meijerg(((1/2,), ()), ((5, 0, 1/2), ()), x2)
        expr = (m1 + m2) / (48 * pi)

        # 在评估时出现 NumPy/SciPy 失败的警告情况下，绘制表达式的图像并保存
        with warns(
            UserWarning,
            match="The evaluation with NumPy/SciPy failed",
            test_stacklevel=False,
        ):
            p = plot(expr, (x, 1e-6, 1e-2), adaptive=adaptive, n=10)
            p.save(os.path.join(tmpdir, filename))
@pytest.mark.parametrize("adaptive", [True, False])
def test_plotgrid_and_save(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')

    # 使用临时目录创建绘图
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        # 绘制第一个子图 p1
        p1 = plot(x, adaptive=adaptive, n=10)
        # 绘制第二个子图 p2
        p2 = plot_parametric((sin(x), cos(x)), (x, sin(x)), show=False,
            adaptive=adaptive, n=10)
        # 绘制第三个子图 p3
        p3 = plot_parametric(
            cos(x), sin(x), adaptive=adaptive, n=10, show=False)
        # 绘制第四个子图 p4
        p4 = plot3d_parametric_line(sin(x), cos(x), x, show=False,
            adaptive=adaptive, n=10)
        
        # 创建一个 2x2 的绘图网格 p
        p = PlotGrid(2, 2, p1, p2, p3, p4)
        # 设置保存文件名
        filename = 'test_grid1.png'
        # 保存网格 p 到临时目录
        p.save(os.path.join(tmpdir, filename))
        # 关闭后端绘图对象
        p._backend.close()

        # 创建一个 3x4 的绘图网格 p
        p = PlotGrid(3, 4, p1, p2, p3, p4)
        # 设置保存文件名
        filename = 'test_grid2.png'
        # 保存网格 p 到临时目录
        p.save(os.path.join(tmpdir, filename))
        # 关闭后端绘图对象
        p._backend.close()

        # 绘制第五个子图 p5
        p5 = plot(cos(x),(x, -pi, pi), show=False, adaptive=adaptive, n=10)
        # 设置线条颜色函数
        p5[0].line_color = lambda a: a
        # 绘制第六个子图 p6
        p6 = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), show=False,
            adaptive=adaptive, n=10)
        # 绘制第七个子图 p7
        p7 = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)), show=False,
            adaptive=adaptive, n=10)
        
        # 创建一个 1x3 的绘图网格 p
        p = PlotGrid(1, 3, p5, p6, p7)
        # 设置保存文件名
        filename = 'test_grid3.png'
        # 保存网格 p 到临时目录
        p.save(os.path.join(tmpdir, filename))
        # 关闭后端绘图对象
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_append_issue_7140(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')
    
    # 绘制函数 x
    p1 = plot(x, adaptive=adaptive, n=10)
    # 绘制函数 x^2
    p2 = plot(x**2, adaptive=adaptive, n=10)
    # 绘制函数 x + 2
    plot(x + 2, adaptive=adaptive, n=10)

    # 将 p1 添加到 p2 的系列中
    p2.append(p1[0])
    # 断言 p2 的系列长度为 2
    assert len(p2._series) == 2

    # 测试错误情况：不能将 p2 添加到 p1 中
    with raises(TypeError):
        p1.append(p2)

    # 测试错误情况：不能将 p2 的系列添加到 p1 中
    with raises(TypeError):
        p1.append(p2._series)


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_15265(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')
    # 定义函数 eqn = sin(x)
    eqn = sin(x)

    # 绘制函数 eqn，设置 x 轴范围和 y 轴范围
    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1), adaptive=adaptive, n=10)
    # 关闭后端绘图对象
    p._backend.close()

    # 绘制函数 eqn，设置 x 轴范围和 y 轴范围
    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi), adaptive=adaptive, n=10)
    # 关闭后端绘图对象
    p._backend.close()

    # 绘制函数 eqn，设置 x 轴范围和 y 轴范围
    p = plot(eqn, xlim=(-1, 1), adaptive=adaptive, n=10,
        ylim=(sympify('-3.14'), sympify('3.14')))
    # 关闭后端绘图对象
    p._backend.close()

    # 绘制函数 eqn，设置 x 轴范围和 y 轴范围
    p = plot(eqn, adaptive=adaptive, n=10,
        xlim=(sympify('-3.14'), sympify('3.14')), ylim=(-1, 1))
    # 关闭后端绘图对象
    p._backend.close()

    # 断言值错误：xlim 中不能包含 S.ImaginaryUnit
    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))
    # 调用 plot 函数并检查是否引发 ValueError 异常，用于测试参数有效性
    raises(ValueError,
        # 使用 lambda 匿名函数调用 plot 函数，传递参数 eqn、adaptive、n，以及 xlim=(-1, 1) 和 ylim=(-1, S.ImaginaryUnit)
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))

    # 调用 plot 函数并检查是否引发 ValueError 异常，用于测试参数有效性
    raises(ValueError,
        # 使用 lambda 匿名函数调用 plot 函数，传递参数 eqn、adaptive、n，以及 xlim=(S.NegativeInfinity, 1) 和 ylim=(-1, 1)
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))

    # 调用 plot 函数并检查是否引发 ValueError 异常，用于测试参数有效性
    raises(ValueError,
        # 使用 lambda 匿名函数调用 plot 函数，传递参数 eqn、adaptive、n，以及 xlim=(-1, 1) 和 ylim=(-1, S.Infinity)
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.Infinity)))
# 测试一个空的 Plot 对象的行为
def test_empty_Plot():
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 调用 plot() 函数，没有异常会显示一个空的图形
    plot()

    # 创建一个 Plot 对象，它只是一个基类，没有实现显示图像的逻辑
    p = Plot()

    # 使用 lambda 函数和 raises() 断言来验证调用 p.show() 会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: p.show())


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_17405(adaptive):
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 定义一个符号表达式 f
    f = x**0.3 - 10*x**3 + x**2

    # 调用 plot() 函数创建一个图形对象 p，显示范围为 (-10, 10)，自适应参数为 adaptive，点数为 30，但不显示图形
    p = plot(f, (x, -10, 10), adaptive=adaptive, n=30, show=False)

    # 使用 ignore_warnings() 上下文管理器忽略 RuntimeWarning 异常
    with ignore_warnings(RuntimeWarning):
        # 断言 p[0].get_data()[0] 返回的数据点数至少为 30
        assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_logplot_PR_16796(adaptive):
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 使用对数坐标系绘制 x 的图形，范围为 (0.001, 100)，自适应参数为 adaptive，点数为 30，但不显示图形
    p = plot(x, (x, .001, 100), adaptive=adaptive, n=30,
             xscale='log', show=False)

    # 断言 p[0].get_data()[0] 返回的数据点数至少为 30
    assert len(p[0].get_data()[0]) >= 30

    # 断言 p[0] 的结束点为 100.0
    assert p[0].end == 100.0

    # 断言 p[0] 的起始点为 0.001
    assert p[0].start == .001


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_16572(adaptive):
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 调用 LambertW 函数并绘制其图形，自适应参数为 adaptive，点数为 30，但不显示图形
    p = plot(LambertW(x), show=False, adaptive=adaptive, n=30)

    # 断言 p[0].get_data()[0] 返回的数据点数至少为 30
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11865(adaptive):
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义整数符号变量 k
    k = Symbol('k', integer=True)

    # 使用 Piecewise 定义符号表达式 f，并绘制其图形，自适应参数为 adaptive，点数为 30，但不显示图形
    f = Piecewise((-I*exp(I*pi*k)/k + I*exp(-I*pi*k)/k, Ne(k, 0)), (2*pi, True))
    p = plot(f, show=False, adaptive=adaptive, n=30)

    # 断言 p[0].get_data()[0] 返回的数据点数至少为 30
    assert len(p[0].get_data()[0]) >= 30


def test_issue_11461():
    # 如果没有安装 matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 使用 real_root 函数绘制 log(x/(x-2)) 的三个实数根的图形，自适应参数为 True，但不显示图形
    p = plot(real_root((log(x/(x-2))), 3), show=False, adaptive=True)

    # 使用 warns() 上下文管理器捕获 RuntimeWarning 异常，并匹配特定的警告消息
    with warns(
        RuntimeWarning,
        match="invalid value encountered in",
        test_stacklevel=False,
    ):
        # 断言 p[0].get_data()[0] 返回的数据点数至少为 30
        assert len(p[0].get_data()[0]) >= 30
@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11764(adaptive):
    # 如果没有导入 matplotlib 模块，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')
    # 绘制参数化图像 cos(x), sin(x)，x 范围为 [0, 2*pi]
    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
                        aspect_ratio=(1,1), show=False, adaptive=adaptive, n=30)
    # 断言绘图的纵横比为 (1, 1)
    assert p.aspect_ratio == (1, 1)
    # 断言生成的线段数量至少为 30 条
    # 通常会生成超过 100 条线段，我们检查是否生成了线段，以验证修复前后的差异
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_13516(adaptive):
    # 如果没有导入 matplotlib 模块，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 使用 matplotlib 后端绘制 sin(x) 图像，不显示，自适应曲线密度为 n=30
    pm = plot(sin(x), backend="matplotlib", show=False, adaptive=adaptive, n=30)
    # 断言使用的后端是 MatplotlibBackend
    assert pm.backend == MatplotlibBackend
    # 断言生成的线段数量至少为 30 条
    assert len(pm[0].get_data()[0]) >= 30

    # 使用文本后端绘制 sin(x) 图像，不显示，自适应曲线密度为 n=30
    pt = plot(sin(x), backend="text", show=False, adaptive=adaptive, n=30)
    # 断言使用的后端是 TextBackend
    assert pt.backend == TextBackend
    # 断言生成的线段数量至少为 30 条
    assert len(pt[0].get_data()[0]) >= 30

    # 使用默认后端绘制 sin(x) 图像，不显示，自适应曲线密度为 n=30
    pd = plot(sin(x), backend="default", show=False, adaptive=adaptive, n=30)
    # 断言使用的后端是 MatplotlibBackend
    assert pd.backend == MatplotlibBackend
    # 断言生成的线段数量至少为 30 条
    assert len(pd[0].get_data()[0]) >= 30

    # 绘制 sin(x) 图像，不显示，自适应曲线密度为 n=30（使用默认后端）
    p = plot(sin(x), show=False, adaptive=adaptive, n=30)
    # 断言使用的后端是 MatplotlibBackend
    assert p.backend == MatplotlibBackend
    # 断言生成的线段数量至少为 30 条
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_limits(adaptive):
    # 如果没有导入 matplotlib 模块，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')
    # 绘制 x 和 x^2 的图像，x 范围为 [-10, 10]，自适应曲线密度为 n=10
    p = plot(x, x**2, (x, -10, 10), adaptive=adaptive, n=10)
    # 获取绘图的后端对象
    backend = p._backend

    # 断言 x 轴的显示范围在 [-10, 10] 内（误差小于 2）
    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    # 断言 y 轴的显示范围在 [0, 100] 内（误差小于 10）
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot3d_parametric_line_limits(adaptive):
    # 如果没有导入 matplotlib 模块，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x
    x = Symbol('x')

    # 定义参数化线的两个向量 v1 和 v2
    v1 = (2*cos(x), 2*sin(x), 2*x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    # 绘制参数化三维线 v1 和 v2，自适应曲线密度为 n=60
    p = plot3d_parametric_line(v1, v2, adaptive=adaptive, n=60)
    # 获取绘图的后端对象
    backend = p._backend

    # 断言 x 轴的显示范围在 [-2, 2] 内（误差小于 1e-2）
    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    # 断言 y 轴的显示范围在 [-2, 2] 内（误差小于 1e-2）
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    # 断言 z 轴的显示范围在 [-10, 10] 内（误差小于 1e-2）
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    # 交换参数顺序，绘制参数化三维线 v2 和 v1，自适应曲线密度为 n=60
    p = plot3d_parametric_line(v2, v1, adaptive=adaptive, n=60)
    # 获取绘图的后端对象
    backend = p._backend

    # 断言 x 轴的显示范围在 [-2, 2] 内（误差小于 1e-2）
    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    # 断言 y 轴的显示范围在 [-2, 2] 内（误差小于 1e-2）
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    # 断言 z 轴的显示范围在 [-10, 10] 内（误差小于 1e-2）
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2
    # 如果没有导入 matplotlib 模块，抛出跳过的消息
    if not matplotlib:
        skip("Matplotlib not the default backend")
    
    # 定义符号变量 x
    x = Symbol('x')
    
    # 创建一个绘图对象 p1，绘制 sin(x)，使用 matplotlib 后端，设置尺寸为 (8, 4)，并根据 adaptive 参数进行自适应
    p1 = plot(sin(x), backend="matplotlib", size=(8, 4),
        adaptive=adaptive, n=10)
    # 获取 p1 绘图对象的尺寸大小
    s1 = p1._backend.fig.get_size_inches()
    # 断言 p1 的宽度为 8，高度为 4
    assert (s1[0] == 8) and (s1[1] == 4)
    
    # 创建另一个绘图对象 p2，绘制 sin(x)，使用 matplotlib 后端，设置尺寸为 (5, 10)，并根据 adaptive 参数进行自适应
    p2 = plot(sin(x), backend="matplotlib", size=(5, 10),
        adaptive=adaptive, n=10)
    # 获取 p2 绘图对象的尺寸大小
    s2 = p2._backend.fig.get_size_inches()
    # 断言 p2 的宽度为 5，高度为 10
    assert (s2[0] == 5) and (s2[1] == 10)
    
    # 创建一个 PlotGrid 绘图对象 p3，包含两个子图 p1 和 p2，设置整体尺寸为 (6, 2)，并根据 adaptive 参数进行自适应
    p3 = PlotGrid(2, 1, p1, p2, size=(6, 2),
        adaptive=adaptive, n=10)
    # 获取 p3 绘图对象的尺寸大小
    s3 = p3._backend.fig.get_size_inches()
    # 断言 p3 的宽度为 6，高度为 2
    assert (s3[0] == 6) and (s3[1] == 2)
    
    # 使用 raises() 上下文管理器，预期会抛出 ValueError 异常
    with raises(ValueError):
        # 绘制 sin(x)，使用 matplotlib 后端，设置尺寸为 (-1, 3)，这里会引发异常
        plot(sin(x), backend="matplotlib", size=(-1, 3))
# 定义测试函数，用于验证GitHub问题20113的修复情况
def test_issue_20113():
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 创建符号变量x
    x = Symbol('x')

    # 使用自定义后端绘制sin(x)函数图像，不显示
    plot(sin(x), backend=Plot, show=False)
    
    # 使用Matplotlib后端绘制sin(x)函数图像，不显示
    p2 = plot(sin(x), backend=MatplotlibBackend, show=False)
    # 断言使用了正确的后端（MatplotlibBackend）
    assert p2.backend == MatplotlibBackend
    # 断言绘制的曲线数据点数至少为30个
    assert len(p2[0].get_data()[0]) >= 30
    
    # 使用DummyBackendOk后端绘制sin(x)函数图像，不显示
    p3 = plot(sin(x), backend=DummyBackendOk, show=False)
    # 断言使用了正确的后端（DummyBackendOk）
    assert p3.backend == DummyBackendOk
    # 断言绘制的曲线数据点数至少为30个
    assert len(p3[0].get_data()[0]) >= 30

    # 使用DummyBackendNotOk后端绘制sin(x)函数图像，不显示
    p4 = plot(sin(x), backend=DummyBackendNotOk, show=False)
    # 断言使用了正确的后端（DummyBackendNotOk）
    assert p4.backend == DummyBackendNotOk
    # 断言绘制的曲线数据点数至少为30个
    assert len(p4[0].get_data()[0]) >= 30
    # 使用raises检查是否抛出NotImplementedError异常
    with raises(NotImplementedError):
        p4.show()
    with raises(NotImplementedError):
        p4.save("test/path")
    with raises(NotImplementedError):
        p4._backend.close()


# 定义测试函数，验证自定义颜色功能是否正常
def test_custom_coloring():
    # 创建符号变量x和y
    x = Symbol('x')
    y = Symbol('y')

    # 绘制cos(x)函数图像，使用lambda函数设置线条颜色为参数a
    plot(cos(x), line_color=lambda a: a)
    # 绘制cos(x)函数图像，设置线条颜色为数字1
    plot(cos(x), line_color=1)
    # 绘制cos(x)函数图像，设置线条颜色为字符串"r"（红色）
    plot(cos(x), line_color="r")

    # 绘制参数化曲线(cos(x), sin(x))，使用lambda函数设置线条颜色为参数a
    plot_parametric(cos(x), sin(x), line_color=lambda a: a)
    # 绘制参数化曲线(cos(x), sin(x))，设置线条颜色为数字1
    plot_parametric(cos(x), sin(x), line_color=1)
    # 绘制参数化曲线(cos(x), sin(x))，设置线条颜色为字符串"r"（红色）
    plot_parametric(cos(x), sin(x), line_color="r")

    # 绘制3D参数化线(cos(x), sin(x), x)，使用lambda函数设置线条颜色为参数a
    plot3d_parametric_line(cos(x), sin(x), x, line_color=lambda a: a)
    # 绘制3D参数化线(cos(x), sin(x), x)，设置线条颜色为数字1
    plot3d_parametric_line(cos(x), sin(x), x, line_color=1)
    # 绘制3D参数化线(cos(x), sin(x), x)，设置线条颜色为字符串"r"（红色）
    plot3d_parametric_line(cos(x), sin(x), x, line_color="r")

    # 绘制3D参数化曲面(cos(x+y), sin(x-y), x-y)，设置表面颜色为lambda函数计算结果
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    # 绘制3D参数化曲面(cos(x+y), sin(x-y), x-y)，设置表面颜色为数字1
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=1)
    # 绘制3D参数化曲面(cos(x+y), sin(x-y), x-y)，设置表面颜色为字符串"r"（红色）
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color="r")

    # 绘制3D函数图像x*y，设置表面颜色为lambda函数计算结果
    plot3d(x*y, (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    # 绘制3D函数图像x*y，设置表面颜色为数字1
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color=1)
    # 绘制3D函数图像x*y，设置表面颜色为字符串"r"（红色）
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color="r")


# 使用参数化测试adaptive为True和False的情况，测试已弃用的get_segments方法
@pytest.mark.parametrize("adaptive", [True, False])
def test_deprecated_get_segments(adaptive):
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 创建符号变量x，定义函数f=sin(x)
    x = Symbol('x')
    f = sin(x)

    # 绘制sin(x)函数图像，x取值范围为-10到10，不显示，使用自定义参数adaptive和n=10
    p = plot(f, (x, -10, 10), show=False, adaptive=adaptive, n=10)
    # 使用warns_deprecated_sympy检查是否已发出弃用警告
    with warns_deprecated_sympy():
        p[0].get_segments()


# 使用参数化测试adaptive为True和False的情况，验证通用数据系列的功能
@pytest.mark.parametrize("adaptive", [True, False])
def test_generic_data_series(adaptive):
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 创建符号变量x
    x = Symbol("x")
    # 绘制x函数图像，设置各种图形属性，并使用自定义参数adaptive和n=10
    p = plot(x,
        markers=[{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}],
        annotations=[{"text": "test", "xy": (0, 0)}],
        fill={"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]},
        rectangles=[{"xy": (0, 0), "width": 5, "height": 1}],
        adaptive=adaptive, n=10)
    # 断言绘图对象中包含的绘图集合数量为1
    assert len(p._backend.ax.collections) == 1
    # 断言检查图形对象中的补丁数量是否为1
    assert len(p._backend.ax.patches) == 1
    # 断言检查图形对象中的线条数量是否为2
    assert len(p._backend.ax.lines) == 2
    # 断言检查图形对象中的文本数量是否为1
    assert len(p._backend.ax.texts) == 1
# 测试函数，用于测试在不推荐使用的情况下设置标记、注释、填充和矩形的行为
def test_deprecated_markers_annotations_rectangles_fill():
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量x
    x = Symbol('x')
    # 绘制sin(x)函数的图像，范围为x从-10到10
    p = plot(sin(x), (x, -10, 10), show=False)
    # 在警告被弃用的情况下设置标记为星号
    with warns_deprecated_sympy():
        p.markers = [{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}]
    # 断言_p_对象的系列数为2
    assert len(p._series) == 2
    # 在警告被弃用的情况下添加注释
    with warns_deprecated_sympy():
        p.annotations = [{"text": "test", "xy": (0, 0)}]
    # 断言_p_对象的系列数为3
    assert len(p._series) == 3
    # 在警告被弃用的情况下设置填充属性
    with warns_deprecated_sympy():
        p.fill = {"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]}
    # 断言_p_对象的系列数为4
    assert len(p._series) == 4
    # 在警告被弃用的情况下设置矩形属性
    with warns_deprecated_sympy():
        p.rectangles = [{"xy": (0, 0), "width": 5, "height": 1}]
    # 断言_p_对象的系列数为5
    assert len(p._series) == 5


# 测试函数，用于测试向后兼容性
def test_back_compatibility():
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量x和y
    x = Symbol('x')
    y = Symbol('y')

    # 绘制sin(x)函数的图像，禁用自适应，n=5
    p = plot(sin(x), adaptive=False, n=5)
    # 断言_p_对象第一个元素的点数为2
    assert len(p[0].get_points()) == 2
    # 断言_p_对象第一个元素的数据长度为2
    assert len(p[0].get_data()) == 2

    # 绘制参数方程cos(x), sin(x)，范围为x从0到2，禁用自适应，n=5
    p = plot_parametric(cos(x), sin(x), (x, 0, 2), adaptive=False, n=5)
    # 断言_p_对象第一个元素的点数为2
    assert len(p[0].get_points()) == 2
    # 断言_p_对象第一个元素的数据长度为3
    assert len(p[0].get_data()) == 3

    # 绘制三维参数线cos(x), sin(x), x，范围为x从0到2，禁用自适应，n=5
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2),
        adaptive=False, n=5)
    # 断言_p_对象第一个元素的点数为3
    assert len(p[0].get_points()) == 3
    # 断言_p_对象第一个元素的数据长度为4
    assert len(p[0].get_data()) == 4

    # 绘制三维图像cos(x**2 + y**2)，范围为x和y均从-pi到pi，n=5
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    # 断言_p_对象第一个元素的网格数为3
    assert len(p[0].get_meshes()) == 3
    # 断言_p_对象第一个元素的数据长度为3
    assert len(p[0].get_data()) == 3

    # 绘制等高线图cos(x**2 + y**2)，范围为x和y均从-pi到pi，n=5
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    # 断言_p_对象第一个元素的网格数为3
    assert len(p[0].get_meshes()) == 3
    # 断言_p_对象第一个元素的数据长度为3
    assert len(p[0].get_data()) == 3

    # 绘制三维参数曲面x * cos(y), x * sin(y), x * cos(4 * y) / 2，
    # 范围为x从0到pi，y从0到2*pi，n=5
    p = plot3d_parametric_surface(x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        (x, 0, pi), (y, 0, 2*pi), n=5)
    # 断言_p_对象第一个元素的网格数为3
    assert len(p[0].get_meshes()) == 3
    # 断言_p_对象第一个元素的数据长度为5


def test_plot_arguments():
    ### 测试plot()函数的参数
    # 如果没有导入matplotlib，则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量x和y
    x, y = symbols("x, y")

    # 单个表达式x + 1的绘制，范围默认为x从-10到10
    p = plot(x + 1)
    # 断言_p_对象的第一个元素是LineOver1DRangeSeries类型
    assert isinstance(p[0], LineOver1DRangeSeries)
    # 断言_p_对象的第一个元素的表达式为x + 1
    assert p[0].expr == x + 1
    # 断言_p_对象的第一个元素的范围为[(x, -10, 10)]
    assert p[0].ranges == [(x, -10, 10)]
    # 断言_p_对象的第一个元素的标签为"x + 1"
    assert p[0].get_label(False) == "x + 1"
    # 断言_p_对象的第一个元素的渲染关键词为空字典
    assert p[0].rendering_kw == {}

    # 单个表达式x + 1的绘制，自定义标签为"label"
    p = plot(x + 1, "label")
    # 断言_p_对象的第一个元素是LineOver1DRangeSeries类型
    assert isinstance(p[0], LineOver1DRangeSeries)
    # 断言_p_对象的第一个元素的表达式为x + 1
    assert p[0].expr == x + 1
    # 断言_p_对象的第一个元素的范围为[(x, -10, 10)]
    assert p[0].ranges == [(x, -10, 10)]
    # 断言_p_对象的第一个元素的标签为"label"
    assert p[0].get_label(False) == "label"
    # 断言_p_对象的第一个元素的渲染关键词为空字典
    assert p[0].rendering_kw == {}

    # 单个表达式x + 1的绘制，范围为x从-2到2
    p = plot(x + 1, (x, -2, 2))
    # 断言_p_对象的第一个元素的范围为[(x, -2, 2)]
    assert p[0].ranges == [(x, -2, 2)]

    # 单个表达式x + 1的绘制，范围为x从-2到2，自定义标签为"test"，渲染关键词为{"color": "r"}
    p = plot(x
    # 断言第一个图形对象的标签不带括号的表达式 "x + 1"
    assert p[0].get_label(False) == "x + 1"
    # 断言第一个图形对象的渲染关键字为空字典
    assert p[0].rendering_kw == {}
    # 断言第二个图形对象是 LineOver1DRangeSeries 类的实例
    assert isinstance(p[1], LineOver1DRangeSeries)
    # 断言第二个图形对象的表达式为 x**2
    assert p[1].expr == x**2
    # 断言第二个图形对象的范围为 [(x, -10, 10)]
    assert p[1].ranges == [(x, -10, 10)]
    # 断言第二个图形对象的标签为 "x**2"
    assert p[1].get_label(False) == "x**2"
    # 断言第二个图形对象的渲染关键字为空字典
    assert p[1].rendering_kw == {}

    # 在相同范围内绘制多个表达式
    p = plot(x + 1, x**2, (x, 0, 5))
    # 断言第一个图形对象的范围为 [(x, 0, 5)]
    assert p[0].ranges == [(x, 0, 5)]
    # 断言第二个图形对象的范围为 [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]

    # 在相同范围内绘制多个表达式，并使用相同的渲染关键字
    p = plot(x + 1, x**2, (x, 0, 5), {"color": "r"})
    # 断言第一个图形对象的范围为 [(x, 0, 5)]
    assert p[0].ranges == [(x, 0, 5)]
    # 断言第二个图形对象的范围为 [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]
    # 断言第一个图形对象的渲染关键字为 {"color": "r"}
    assert p[0].rendering_kw == {"color": "r"}
    # 断言第二个图形对象的渲染关键字为 {"color": "r"}
    assert p[1].rendering_kw == {"color": "r"}

    # 绘制多个表达式，每个表达式具有不同的范围、标签和渲染关键字
    p = plot(
        (x + 1, (x, 0, 5)),
        (x**2, (x, -2, 2), "test", {"color": "r"}))
    # 断言第一个图形对象是 LineOver1DRangeSeries 类的实例
    assert isinstance(p[0], LineOver1DRangeSeries)
    # 断言第一个图形对象的表达式为 x + 1
    assert p[0].expr == x + 1
    # 断言第一个图形对象的范围为 [(x, 0, 5)]
    assert p[0].ranges == [(x, 0, 5)]
    # 断言第一个图形对象的标签为 "x + 1"
    assert p[0].get_label(False) == "x + 1"
    # 断言第一个图形对象的渲染关键字为空字典
    assert p[0].rendering_kw == {}
    # 断言第二个图形对象是 LineOver1DRangeSeries 类的实例
    assert isinstance(p[1], LineOver1DRangeSeries)
    # 断言第二个图形对象的表达式为 x**2
    assert p[1].expr == x**2
    # 断言第二个图形对象的范围为 [(x, -2, 2)]
    assert p[1].ranges == [(x, -2, 2)]
    # 断言第二个图形对象的标签为 "test"
    assert p[1].get_label(False) == "test"
    # 断言第二个图形对象的渲染关键字为 {"color": "r"}
    assert p[1].rendering_kw == {"color": "r"}

    # 绘制一个 lambda 函数的图形
    f = lambda t: t
    p = plot(lambda t: t)
    # 断言第一个图形对象是 LineOver1DRangeSeries 类的实例
    assert isinstance(p[0], LineOver1DRangeSeries)
    # 断言第一个图形对象的表达式是可调用的 lambda 函数
    assert callable(p[0].expr)
    # 断言第一个图形对象的范围是 (-10, 10)，与默认的 x 范围匹配
    assert p[0].ranges[0][1:] == (-10, 10)
    # 断言第一个图形对象的标签为空字符串
    assert p[0].get_label(False) == ""
    # 断言第一个图形对象的渲染关键字为空字典
    assert p[0].rendering_kw == {}

    # 绘制一个 lambda 函数的图形，并指定自定义的范围和标签
    p = plot(f, ("t", -5, 6), "test")
    # 断言第一个图形对象的范围为 [("t", -5, 6)]
    assert p[0].ranges[0][1:] == (-5, 6)
    # 断言第一个图形对象的标签为 "test"
    assert p[0].get_label(False) == "test"
def test_plot_parametric_arguments():
    ### Test arguments for plot_parametric()

    # 检查是否导入了 matplotlib，如果没有则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y
    x, y = symbols("x, y")

    # 单个参数化表达式的测试
    p = plot_parametric(x + 1, x)
    # 断言第一个图表对象是 Parametric2DLineSeries 类的实例
    assert isinstance(p[0], Parametric2DLineSeries)
    # 断言参数化表达式和范围设置正确
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    # 断言标签和渲染参数为空字典
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # 单个参数化表达式，自定义范围、标签和渲染参数
    p = plot_parametric(x + 1, x, (x, -2, 2), "test",
        {"cmap": "Reds"})
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # 参数化表达式，包含多个参数组合
    p = plot_parametric((x + 1, x), (x, -2, 2), "test")
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # 多个参数化表达式使用相同的符号
    p = plot_parametric((x + 1, x), (x ** 2, x + 1))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # 多个参数化表达式使用不同的符号
    p = plot_parametric((x + 1, x), (y ** 2, y + 1, "test"))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, y + 1)
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # 多个参数化表达式使用相同的范围
    p = plot_parametric((x + 1, x), (x ** 2, x + 1), (x, -2, 2))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # 多个参数化表达式，自定义范围和标签，带渲染参数
    p = plot_parametric(
        (x + 1, x, (x, -2, 2), "test1"),
        (x ** 2, x + 1, (x, -3, 3), "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test1"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -3, 3)]
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # 单个参数：lambda 函数
    fx = lambda t: t
    fy = lambda t: 2 * t
    p = plot_parametric(fx, fy)
    # 断言所有表达式都是可调用的函数
    assert all(callable(t) for t in p[0].expr)
    # 断言范围设置正确
    assert p[0].ranges[0][1:] == (-10, 10)
    # 断言：检查标签是否包含 "Dummy"
    assert "Dummy" in p[0].get_label(False)
    # 断言：检查 rendering_kw 是否为空字典
    assert p[0].rendering_kw == {}

    # 创建参数绘图对象：使用 lambda 函数、自定义范围和标签
    p = plot_parametric(fx, fy, ("t", 0, 2), "test")
    # 断言：检查所有表达式是否都是可调用的 lambda 函数
    assert all(callable(t) for t in p[0].expr)
    # 断言：检查第一个参数的范围是否为 (0, 2)
    assert p[0].ranges[0][1:] == (0, 2)
    # 断言：检查获取标签是否为 "test"
    assert p[0].get_label(False) == "test"
    # 断言：检查 rendering_kw 是否为空字典
    assert p[0].rendering_kw == {}
# 测试 plot3d_parametric_line() 函数的不同参数情况

if not matplotlib:
    # 如果没有导入 matplotlib，则跳过测试
    skip("Matplotlib not the default backend")

x, y = symbols("x, y")

# 单个参数化表达式
p = plot3d_parametric_line(x + 1, x, sin(x))
# 断言第一个元素是 Parametric3DLineSeries 类的实例
assert isinstance(p[0], Parametric3DLineSeries)
# 断言表达式与参数的一致性
assert p[0].expr == (x + 1, x, sin(x))
# 断言范围的一致性
assert p[0].ranges == [(x, -10, 10)]
# 断言标签的一致性
assert p[0].get_label(False) == "x"
# 断言渲染关键字参数的一致性
assert p[0].rendering_kw == {}

# 单个参数化表达式，包括自定义范围、标签和渲染参数
p = plot3d_parametric_line(x + 1, x, sin(x), (x, -2, 2),
    "test", {"cmap": "Reds"})
assert isinstance(p[0], Parametric3DLineSeries)
assert p[0].expr == (x + 1, x, sin(x))
assert p[0].ranges == [(x, -2, 2)]
assert p[0].get_label(False) == "test"
assert p[0].rendering_kw == {"cmap": "Reds"}

p = plot3d_parametric_line((x + 1, x, sin(x)), (x, -2, 2), "test")
assert p[0].expr == (x + 1, x, sin(x))
assert p[0].ranges == [(x, -2, 2)]
assert p[0].get_label(False) == "test"
assert p[0].rendering_kw == {}

# 多个参数化表达式，使用相同的符号
p = plot3d_parametric_line(
    (x + 1, x, sin(x)), (x ** 2, 1, cos(x), {"cmap": "Reds"}))
assert p[0].expr == (x + 1, x, sin(x))
assert p[0].ranges == [(x, -10, 10)]
assert p[0].get_label(False) == "x"
assert p[0].rendering_kw == {}
assert p[1].expr == (x ** 2, 1, cos(x))
assert p[1].ranges == [(x, -10, 10)]
assert p[1].get_label(False) == "x"
assert p[1].rendering_kw == {"cmap": "Reds"}

# 多个参数化表达式，使用不同的符号
p = plot3d_parametric_line((x + 1, x, sin(x)), (y ** 2, 1, cos(y)))
assert p[0].expr == (x + 1, x, sin(x))
assert p[0].ranges == [(x, -10, 10)]
assert p[0].get_label(False) == "x"
assert p[0].rendering_kw == {}
assert p[1].expr == (y ** 2, 1, cos(y))
assert p[1].ranges == [(y, -10, 10)]
assert p[1].get_label(False) == "y"
assert p[1].rendering_kw == {}

# 多个参数化表达式，自定义范围和标签
p = plot3d_parametric_line(
    (x + 1, x, sin(x)),
    (x ** 2, 1, cos(x), (x, -2, 2), "test", {"cmap": "Reds"}))
assert p[0].expr == (x + 1, x, sin(x))
assert p[0].ranges == [(x, -10, 10)]
assert p[0].get_label(False) == "x"
assert p[0].rendering_kw == {}
assert p[1].expr == (x ** 2, 1, cos(x))
assert p[1].ranges == [(x, -2, 2)]
assert p[1].get_label(False) == "test"
assert p[1].rendering_kw == {"cmap": "Reds"}

# 单参数：lambda 函数
fx = lambda t: t
fy = lambda t: 2 * t
fz = lambda t: 3 * t
p = plot3d_parametric_line(fx, fy, fz)
# 断言所有表达式都是可调用的函数
assert all(callable(t) for t in p[0].expr)
# 断言范围的一致性
assert p[0].ranges[0][1:] == (-10, 10)
# 断言标签中包含 "Dummy"
assert "Dummy" in p[0].get_label(False)
# 断言渲染关键字参数的一致性
assert p[0].rendering_kw == {}

# 单参数：lambda 函数 + 自定义范围 + 标签
    # 使用 plot3d_parametric_line 函数创建一个三维参数化曲线对象 p，
    # 参数 fx, fy, fz 分别表示曲线的 x, y, z 分量函数，("t", 0, 2) 指定参数范围为 t 从 0 到 2，
    # "test" 是曲线对象的标签
    p = plot3d_parametric_line(fx, fy, fz, ("t", 0, 2), "test")
    
    # 断言：检查 p[0].expr 中的每个表达式是否可调用（即是否为函数）
    assert all(callable(t) for t in p[0].expr)
    
    # 断言：检查 p[0].ranges[0][1:] 是否等于 (0, 2)，即参数范围是否正确
    assert p[0].ranges[0][1:] == (0, 2)
    
    # 断言：检查 p[0].get_label(False) 是否返回 "test"，即标签是否正确
    assert p[0].get_label(False) == "test"
    
    # 断言：检查 p[0].rendering_kw 是否为空字典
    assert p[0].rendering_kw == {}
# 定义测试函数，测试 plot3d() 和 plot_contour() 的参数
def test_plot3d_plot_contour_arguments():
    ### Test arguments for plot3d() and plot_contour()
    # 检查是否导入了 matplotlib 模块，如果没有则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x 和 y
    x, y = symbols("x, y")

    # 单个表达式的测试
    # 使用默认范围生成 3D 图表对象 p，验证其类型为 SurfaceOver2DRangeSeries
    p = plot3d(x + y)
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    # 验证默认范围在 (-10, 10) 内
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    # 验证标签为 "x + y"
    assert p[0].get_label(False) == "x + y"
    # 验证渲染关键字为空字典
    assert p[0].rendering_kw == {}

    # 单个表达式的测试，自定义范围、标签和渲染关键字
    p = plot3d(x + y, (x, -2, 2), "test", {"cmap": "Reds"})
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # 多个表达式的测试
    p = plot3d(x + y, x * y)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # 多个表达式的测试，相同的自定义范围
    p = plot3d(x + y, x * y, (x, -2, 2), (y, -4, 4))
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -2, 2)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # 多个表达式的测试，自定义范围、标签和渲染关键字
    p = plot3d(
        (x + y, (x, -2, 2), (y, -4, 4)),
        (x * y, (x, -3, 3), (y, -6, 6), "test", {"cmap": "Reds"})
    )
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -6, 6)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # 单个表达式：使用 lambda 函数
    f = lambda x, y: x + y
    p = plot3d(f)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].ranges[1][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}
    # 使用 plot3d 函数创建 3D 绘图对象，并指定函数 f，参数范围和标题
    p = plot3d(f, ("a", -5, 3), ("b", -2, 1), "test")
    # 断言第一个图表对象的表达式是可调用的
    assert callable(p[0].expr)
    # 断言第一个图表对象的参数范围与预期相符
    assert p[0].ranges[0][1:] == (-5, 3)
    assert p[0].ranges[1][1:] == (-2, 1)
    # 断言第一个图表对象的标签为 "test"
    assert p[0].get_label(False) == "test"
    # 断言第一个图表对象的渲染参数为空字典
    assert p[0].rendering_kw == {}

    # 测试问题编号 25818
    # 单个表达式，自定义范围，最小/最大函数
    p = plot3d(Min(x, y), (x, 0, 10), (y, 0, 10))
    # 断言第一个图表对象是 SurfaceOver2DRangeSeries 类的实例
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    # 断言第一个图表对象的表达式与 Min(x, y) 相符
    assert p[0].expr == Min(x, y)
    # 断言第一个图表对象的参数范围与预期相符
    assert p[0].ranges[0] == (x, 0, 10)
    assert p[0].ranges[1] == (y, 0, 10)
    # 断言第一个图表对象的标签为 "Min(x, y)"
    assert p[0].get_label(False) == "Min(x, y)"
    # 断言第一个图表对象的渲染参数为空字典
    assert p[0].rendering_kw == {}
def test_plot3d_parametric_surface_arguments():
    ### Test arguments for plot3d_parametric_surface()

    # 检查是否导入了 matplotlib，如果没有则跳过测试
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 定义符号变量 x, y
    x, y = symbols("x, y")

    # 单个参数表达式
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y))
    # 断言第一个图表对象是 ParametricSurfaceSeries 类型
    assert isinstance(p[0], ParametricSurfaceSeries)
    # 断言参数表达式与期望的相同
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    # 断言参数范围为默认值 (-10, 10)
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    # 断言图表标签与预期相同
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    # 断言渲染关键字为空字典
    assert p[0].rendering_kw == {}

    # 单个参数表达式，自定义范围、标签和渲染关键字
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y),
        (x, -2, 2), (y, -4, 4), "test", {"cmap": "Reds"})
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # 多个参数表达式
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y)),
        (x - y, cos(x - y), sin(x - y), "test"))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # 多个参数表达式，自定义范围和标签
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
        (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4),
            "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)  # 注意：这里原注释可能有错误，应为 (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # 使用 lambda 函数代替符号表达式来定义单个 3D 参数化曲面
    p = plot3d_parametric_surface(
        lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4))
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}
    # 使用 lambda 函数代替符号表达式来定义多个三维参数化表面
    p = plot3d_parametric_surface(
        (lambda u, v: u, lambda u, v: v, lambda u, v: u + v,  # 参数化表面的 x, y, z 函数
        ("u", 0, 2), ("v", -3, 4)),  # 参数范围 u 从 0 到 2，v 从 -3 到 4
        (lambda u, v: v, lambda u, v: u, lambda u, v: u - v,  # 参数化表面的 x, y, z 函数
        ("u", -2, 3), ("v", -4, 5), "test"))  # 参数范围 u 从 -2 到 3，v 从 -4 到 5，表面的名称为 "test"
    # 断言检查以下各项是否成立
    assert all(callable(t) for t in p[0].expr)  # 检查第一个参数化表面的表达式是否为可调用函数
    assert p[0].ranges[0][1:] == (0, 2)  # 检查第一个参数化表面的 u 参数范围是否为 (0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)  # 检查第一个参数化表面的 v 参数范围是否为 (-3, 4)
    assert p[0].get_label(False) == ""  # 检查第一个参数化表面的标签是否为空字符串
    assert p[0].rendering_kw == {}  # 检查第一个参数化表面的渲染关键字是否为空字典
    assert all(callable(t) for t in p[1].expr)  # 检查第二个参数化表面的表达式是否为可调用函数
    assert p[1].ranges[0][1:] == (-2, 3)  # 检查第二个参数化表面的 u 参数范围是否为 (-2, 3)
    assert p[1].ranges[1][1:] == (-4, 5)  # 检查第二个参数化表面的 v 参数范围是否为 (-4, 5)
    assert p[1].get_label(False) == "test"  # 检查第二个参数化表面的标签是否为 "test"
    assert p[1].rendering_kw == {}  # 检查第二个参数化表面的渲染关键字是否为空字典
```