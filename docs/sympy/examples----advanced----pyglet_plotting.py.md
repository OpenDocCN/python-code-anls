# `D:\src\scipysrc\sympy\examples\advanced\pyglet_plotting.py`

```
#!/usr/bin/env python

"""
Plotting Examples

Suggested Usage:    python -i pyglet_plotting.py
"""

# 从 sympy 中导入需要的符号和函数
from sympy import symbols, sin, cos, pi, sqrt
# 导入 PygletPlot 类和相关函数
from sympy.plotting.pygletplot import PygletPlot
# 导入 sleep 和 perf_counter 函数
from time import sleep, perf_counter

# 主函数定义
def main():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')

    # 设置轴的显示选项，包括是否可见、颜色、标签等
    axes_options = 'visible=false; colored=true; label_ticks=true; label_axes=true; overlay=true; stride=0.5'
    # 可以根据需要切换不同的轴选项
    # axes_options = 'colored=false; overlay=false; stride=(1.0, 0.5, 0.5)'

    # 创建 PygletPlot 对象实例 p
    p = PygletPlot(
        width=600,                       # 绘图宽度
        height=500,                      # 绘图高度
        ortho=False,                     # 是否使用正交投影
        invert_mouse_zoom=False,         # 是否反转鼠标缩放
        axes=axes_options,               # 轴的显示选项
        antialiasing=True                # 是否开启抗锯齿
    )

    # 存储示例函数的列表
    examples = []

    # 定义示例函数的装饰器
    def example_wrapper(f):
        examples.append(f)
        return f

    # 示例函数：镜像鞍点
    @example_wrapper
    def mirrored_saddles():
        p[5] = x**2 - y**2, [20], [20]
        p[6] = y**2 - x**2, [20], [20]

    # 示例函数：保存镜像鞍点图像
    @example_wrapper
    def mirrored_saddles_saveimage():
        p[5] = x**2 - y**2, [20], [20]
        p[6] = y**2 - x**2, [20], [20]
        p.wait_for_calculations()
        # 等待计算完成，确保图像渲染完毕
        sleep(1)
        p.saveimage("plot_example.png")

    # 示例函数：镜像椭球面
    @example_wrapper
    def mirrored_ellipsoids():
        p[2] = x**2 + y**2, [40], [40], 'color=zfade'
        p[3] = -x**2 - y**2, [40], [40], 'color=zfade'

    # 示例函数：通过导数给鞍点上色
    @example_wrapper
    def saddle_colored_by_derivative():
        f = x**2 - y**2
        p[1] = f, 'style=solid'
        p[1].color = abs(f.diff(x)), abs(f.diff(x) + f.diff(y)), abs(f.diff(y))

    # 示例函数：ding dong 表面
    @example_wrapper
    def ding_dong_surface():
        f = sqrt(1.0 - y)*y
        p[1] = f, [x, 0, 2*pi,
                   40], [y, -
                             1, 4, 100], 'mode=cylindrical; style=solid; color=zfade4'

    # 示例函数：极坐标下的圆
    @example_wrapper
    def polar_circle():
        p[7] = 1, 'mode=polar'

    # 示例函数：极坐标下的花朵形状
    @example_wrapper
    def polar_flower():
        p[8] = 1.5*sin(4*x), [160], 'mode=polar'
        p[8].color = z, x, y, (0.5, 0.5, 0.5), (
            0.8, 0.8, 0.8), (x, y, None, z)  # z 用于 t

    # 示例函数：简单圆柱体
    @example_wrapper
    def simple_cylinder():
        p[9] = 1, 'mode=cylindrical'

    # 示例函数：圆柱体上的双曲线
    @example_wrapper
    def cylindrical_hyperbola():
        # 注意极坐标是圆柱坐标的别名
        p[10] = 1/y, 'mode=polar', [x], [y, -2, 2, 20]

    # 示例函数：外扩的双曲线
    @example_wrapper
    def extruded_hyperbolas():
        p[11] = 1/x, [x, -10, 10, 100], [1], 'style=solid'
        p[12] = -1/x, [x, -10, 10, 100], [1], 'style=solid'

    # 示例函数：环面
    @example_wrapper
    def torus():
        a, b = 1, 0.5  # 半径和厚度
        p[13] = (a + b*cos(x))*cos(y), (a + b*cos(x)) *\
            sin(y), b*sin(x), [x, 0, pi*2, 40], [y, 0, pi*2, 40]

    # 示例函数：扭曲的环面
    @example_wrapper
    def warped_torus():
        a, b = 2, 1  # 半径和厚度
        p[13] = (a + b*cos(x))*cos(y), (a + b*cos(x))*sin(y), b *\
            sin(x) + 0.5*sin(4*y), [x, 0, pi*2, 40], [y, 0, pi*2, 40]
    # 将 parametric_spiral 函数包装为示例函数
    @example_wrapper
    def parametric_spiral():
        # 将参数设置为一个元组，包含了 cos(y), sin(y), y / 10.0 以及用于 y 变量的范围和步长
        p[14] = cos(y), sin(y), y / 10.0, [y, -4*pi, 4*pi, 100]
        # 设置 p[14] 对象的颜色属性，包含 x、y、z 轴的范围和颜色值范围
        p[14].color = x, (0.1, 0.9), y, (0.1, 0.9), z, (0.1, 0.9)
    
    # 将 multistep_gradient 函数包装为示例函数
    @example_wrapper
    def multistep_gradient():
        # 设置 p[1] 对象的参数，包括表达式、参数范围和样式
        p[1] = 1, 'mode=spherical', 'style=both'
        # 定义渐变列表，每对数值和颜色定义一个颜色点
        gradient = [0.0, (0.3, 0.3, 1.0),
                    0.30, (0.3, 1.0, 0.3),
                    0.55, (0.95, 1.0, 0.2),
                    0.65, (1.0, 0.95, 0.2),
                    0.85, (1.0, 0.7, 0.2),
                    1.0, (1.0, 0.3, 0.2)]
        # 设置 p[1] 对象的颜色属性，使用 z 变量作为混合的关键参数
        p[1].color = z, [None, None, z], gradient
    
    # 将 lambda_vs_sympy_evaluation 函数包装为示例函数
    @example_wrapper
    def lambda_vs_sympy_evaluation():
        # 开始计时
        start = perf_counter()
        # 设置 p[4] 对象的表达式和参数范围，样式设置为 solid
        p[4] = x**2 + y**2, [100], [100], 'style=solid'
        # 等待计算完成
        p.wait_for_calculations()
        # 输出 lambda-based 计算所用时间
        print("lambda-based calculation took %s seconds." % (perf_counter() - start))
    
        # 再次计时
        start = perf_counter()
        # 设置 p[4] 对象的表达式和参数范围，样式设置为 solid，使用 sympy 计算
        p[4] = x**2 + y**2, [100], [100], 'style=solid; use_sympy_eval'
        # 等待计算完成
        p.wait_for_calculations()
        # 输出 sympy substitution-based 计算所用时间
        print(
            "sympy substitution-based calculation took %s seconds." %
            (perf_counter() - start))
    def gradient_vectors():
        def gradient_vectors_inner(f, i):
            # 导入所需的库和模块
            from sympy import lambdify  # 导入lambdify函数，用于将SymPy表达式转换为可调用的函数
            from sympy.plotting.plot_interval import PlotInterval  # 导入PlotInterval类，用于定义绘图区间
            from pyglet.gl import glBegin, glColor3f  # 导入pyglet库中的绘图函数和颜色设置函数
            from pyglet.gl import glVertex3f, glEnd, GL_LINES  # 导入绘制顶点和绘制模式相关的常量

            def draw_gradient_vectors(f, iu, iv):
                """
                创建一个绘制梯度向量的函数。
                """
                # 计算函数f对x和y的偏导数
                dx, dy, dz = f.diff(x), f.diff(y), 0
                # 将函数f和其偏导数转换为可调用的函数
                FF = lambdify([x, y], [x, y, f])
                FG = lambdify([x, y], [dx, dy, dz])
                # 调整绘图区间的步长
                iu.v_steps /= 5
                iv.v_steps /= 5
                # 生成梯度向量列表
                Gvl = [[[FF(u, v), FG(u, v)]
                                for v in iv.frange()]
                           for u in iu.frange()]

                def draw_arrow(p1, p2):
                    """
                    绘制单个向量。
                    """
                    # 设置箭头的颜色
                    glColor3f(0.4, 0.4, 0.9)
                    # 绘制向量的起点
                    glVertex3f(*p1)
                    # 设置箭头的颜色
                    glColor3f(0.9, 0.4, 0.4)
                    # 绘制向量的终点
                    glVertex3f(*p2)

                def draw():
                    """
                    迭代绘制计算得到的向量。
                    """
                    glBegin(GL_LINES)
                    for u in Gvl:
                        for v in u:
                            point = [[v[0][0], v[0][1], v[0][2]],
                                     [v[0][0] + v[1][0], v[0][1] + v[1][1], v[0][2] + v[1][2]]]
                            draw_arrow(point[0], point[1])
                    glEnd()

                return draw
            # 将函数f和其绘图区间添加到列表p中
            p[i] = f, [-0.5, 0.5, 25], [-0.5, 0.5, 25], 'style=solid'
            # 创建函数f对应的绘图区间对象
            iu = PlotInterval(p[i].intervals[0])
            iv = PlotInterval(p[i].intervals[1])
            # 将绘制梯度向量的函数添加到绘制后操作列表中
            p[i].postdraw.append(draw_gradient_vectors(f, iu, iv))

        # 调用gradient_vectors_inner函数，分别计算两个示例函数的梯度向量
        gradient_vectors_inner(x**2 + y**2, 1)
        gradient_vectors_inner(-x**2 - y**2, 2)

    def help_str():
        # 返回一个帮助信息字符串，列出了可用的命令和示例
        s = ("\nPlot p has been created. Useful commands: \n"
             "    help(p), p[1] = x**2, print(p), p.clear() \n\n"
             "Available examples (see source in plotting.py):\n\n")
        # 迭代添加示例函数的名称到字符串s中
        for i in range(len(examples)):
            s += "(%i) %s\n" % (i, examples[i].__name__)
        s += "\n"
        s += "e.g. >>> example(2)\n"
        s += "     >>> ding_dong_surface()\n"
        return s

    def example(i):
        # 根据参数i执行相应的示例函数或清空p对象
        if callable(i):
            p.clear()
            i()
        elif i >= 0 and i < len(examples):
            p.clear()
            examples[i]()
        else:
            print("Not a valid example.\n")
        # 打印p对象的内容
        print(p)

    # 执行第一个示例函数
    example(0)  # 0 - 15 are defined above
    # 打印帮助信息字符串
    print(help_str())
if __name__ == "__main__":
    # 如果当前脚本被直接执行而不是被导入为模块，则执行以下代码块
    main()
```