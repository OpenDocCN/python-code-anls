# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_plot_implicit.py`

```
# 从 sympy 库中导入各种符号、函数和绘图相关的模块和函数
from sympy.core.numbers import (I, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import (And, Or)
from sympy.plotting.plot_implicit import plot_implicit
from sympy.plotting.plot import unset_show
from tempfile import NamedTemporaryFile, mkdtemp
from sympy.testing.pytest import skip, warns, XFAIL
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
import os

# 设置图形不显示
unset_show()

# 定义一个函数，创建临时文件并返回其文件名
def tmp_file(dir=None, name=''):
    return NamedTemporaryFile(suffix='.png', dir=dir, delete=False).name

# 定义一个函数，绘制隐式图并保存到临时文件
def plot_and_save(expr, *args, name='', dir=None, **kwargs):
    p = plot_implicit(expr, *args, **kwargs)  # 绘制隐式图
    p.save(tmp_file(dir=dir, name=name))  # 保存绘制的图形到临时文件
    p._backend.close()  # 关闭绘图以避免 matplotlib 的警告

# 定义一个函数，执行隐式图的测试并保存结果
def plot_implicit_tests(name):
    temp_dir = mkdtemp()  # 创建临时文件夹
    TmpFileManager.tmp_folder(temp_dir)  # 设置临时文件夹
    x = Symbol('x')  # 定义符号变量 x
    y = Symbol('y')  # 定义符号变量 y

    # 执行隐式绘图测试，并保存结果到临时文件夹
    plot_and_save(Eq(y, cos(x)), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), (x, -5, 5), (y, -4, 4), name=name, dir=temp_dir)
    plot_and_save(y > 1 / x, (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y < 1 / tan(x), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y >= 2 * sin(x) * cos(x), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y <= x**2, (x, -3, 3), (y, -1, 5), name=name, dir=temp_dir)

    # 测试 plot_implicit 函数的所有输入参数
    plot_and_save(Eq(y**2, x**3 - x), dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, n=500, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), dir=temp_dir)
    plot_and_save(And(y > exp(x), y > x + 2), dir=temp_dir)
    plot_and_save(Or(y > x, y > -x), dir=temp_dir)
    plot_and_save(x**2 - 1, (x, -5, 5), dir=temp_dir)
    plot_and_save(x**2 - 1, dir=temp_dir)
    plot_and_save(y > x, depth=-5, dir=temp_dir)
    plot_and_save(y > x, depth=5, dir=temp_dir)
    plot_and_save(y > cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(y < cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(And(y > cos(x), Or(y > x, Eq(y, x))), dir=temp_dir)
    plot_and_save(y - cos(pi / x), dir=temp_dir)

    # 添加一个带标题的隐式绘图测试
    plot_and_save(x**2 - 1, title='An implicit plot', dir=temp_dir)

# 标记为预期失败的测试函数装饰器
@XFAIL
def test_no_adaptive_meshing():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    # 如果 matplotlib 模块存在，则执行以下代码块
    if matplotlib:
        try:
            # 创建临时目录并注册到临时文件管理器
            temp_dir = mkdtemp()
            TmpFileManager.tmp_folder(temp_dir)
            # 创建符号变量 x 和 y
            x = Symbol('x')
            y = Symbol('y')
            
            # 测试绘图，使用自适应算法无法渲染的情况

            # 下面的代码可以工作，但会触发 sympify() 的弃用警告。
            # 需要更新代码，以在不依赖随机 AttributeError 的情况下检测是否支持区间数学。
            with warns(UserWarning, match="Adaptive meshing could not be applied"):
                # 绘制并保存图形，显示 y = re(cos(x) + I*sin(x)) 的等式
                plot_and_save(Eq(y, re(cos(x) + I*sin(x))), name='test', dir=temp_dir)
        finally:
            # 清理临时文件管理器
            TmpFileManager.cleanup()
    else:
        # 如果 matplotlib 不是默认后端，则跳过执行并输出提示信息
        skip("Matplotlib not the default backend")
# 定义一个测试函数，用于测试绘图的线条颜色设定
def test_line_color():
    # 导入符号变量 x 和 y
    x, y = symbols('x, y')
    # 绘制隐式函数 x**2 + y**2 - 1 = 0 的图像，线条颜色设定为绿色，不立即显示
    p = plot_implicit(x**2 + y**2 - 1, line_color="green", show=False)
    # 断言第一个数据系列的线条颜色是否为 "green"
    assert p._series[0].line_color == "green"
    # 再次绘制相同的隐式函数图像，线条颜色设定为红色，不立即显示
    p = plot_implicit(x**2 + y**2 - 1, line_color='r', show=False)
    # 断言第一个数据系列的线条颜色是否为 "r"（红色）
    assert p._series[0].line_color == "r"

# 定义一个测试函数，用于测试 matplotlib 库是否可用
def test_matplotlib():
    # 导入 matplotlib 库，要求最低版本为 '1.1.0'，捕获 RuntimeError 异常
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    # 如果 matplotlib 库可用
    if matplotlib:
        try:
            # 执行隐式函数绘图的测试函数
            plot_implicit_tests('test')
            # 执行线条颜色设置测试函数
            test_line_color()
        finally:
            # 清理临时文件管理器的临时文件
            TmpFileManager.cleanup()
    else:
        # 如果 matplotlib 不是默认后端，则跳过测试，并给出相应信息
        skip("Matplotlib not the default backend")

# 定义一个测试函数，测试逻辑与操作下的区域绘图
def test_region_and():
    # 导入 matplotlib 库，如果不可用则跳过测试
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # 导入用于图像比较的函数
    from matplotlib.testing.compare import compare_images
    # 获取当前测试文件所在的目录
    test_directory = os.path.dirname(os.path.abspath(__file__))

    try:
        # 创建临时文件夹
        temp_dir = mkdtemp()
        # 将临时文件夹注册到临时文件管理器中
        TmpFileManager.tmp_folder(temp_dir)

        # 导入符号变量 x 和 y
        x, y = symbols('x y')

        # 定义两个圆形区域的隐式函数表示
        r1 = (x - 1)**2 + y**2 < 2
        r2 = (x + 1)**2 + y**2 < 2

        # 创建用于保存测试结果图像的临时文件，并指定文件名
        test_filename = tmp_file(dir=temp_dir, name="test_region_and")
        # 指定用于比较的标准图像文件
        cmp_filename = os.path.join(test_directory, "test_region_and.png")
        # 绘制 r1 和 r2 的逻辑与操作的图像
        p = plot_implicit(r1 & r2, x, y)
        # 保存绘制结果图像到指定的临时文件中
        p.save(test_filename)
        # 比较生成的测试图像和标准图像，容差为 0.005
        compare_images(cmp_filename, test_filename, 0.005)

        # 重复上述步骤，测试逻辑或操作下的区域绘图
        test_filename = tmp_file(dir=temp_dir, name="test_region_or")
        cmp_filename = os.path.join(test_directory, "test_region_or.png")
        p = plot_implicit(r1 | r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        # 测试逻辑非操作下的区域绘图
        test_filename = tmp_file(dir=temp_dir, name="test_region_not")
        cmp_filename = os.path.join(test_directory, "test_region_not.png")
        p = plot_implicit(~r1, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        # 测试逻辑异或操作下的区域绘图
        test_filename = tmp_file(dir=temp_dir, name="test_region_xor")
        cmp_filename = os.path.join(test_directory, "test_region_xor.png")
        p = plot_implicit(r1 ^ r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
    finally:
        # 清理临时文件管理器的临时文件
        TmpFileManager.cleanup()
```