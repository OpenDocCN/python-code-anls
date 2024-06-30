# `D:\src\scipysrc\sympy\examples\advanced\autowrap_integrators.py`

```
# 指定 Python 解释器位置，用于 Unix/Linux 系统
#!/usr/bin/env python

"""
数值积分与自动包装
-----------------------------------

该示例演示如何使用 SymPy 中的 autowrap 模块，
创建可以从 Python 中调用的快速数值积分例程。详细解释了各个步骤。
通过 autowrap 包装的 SymPy 表达式，可以显著提高速度，
远远快于仅使用 numpy 提供的一系列 ufunc 函数。

我们将找到用于以谐振子解的形式近似量子力学氢波函数所需的系数。
为了演示，我们将设置一个简单的数值积分方案作为 SymPy 表达式，并使用 autowrap 获得二进制实现。

在运行此示例之前，需要安装 numpy 和有效的 Fortran 编译器。
如果安装了 pylab，最后将得到一个漂亮的图形。

[0]:
http://ojensen.wordpress.com/2010/08/10/fast-ufunc-ish-hydrogen-solutions/

----
"""

import sys
from sympy.external import import_module

# 尝试导入 numpy
np = import_module('numpy')
if not np:
    # 如果导入失败，打印错误信息并退出程序
    sys.exit("Cannot import numpy. Exiting.")

# 尝试导入 pylab，警告未安装
pylab = import_module('pylab', warn_not_installed=True)

# 导入 SymPy 中的相关模块和函数
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import autowrap, ufuncify
from sympy import Idx, IndexedBase, Lambda, pprint, Symbol, oo, Integral,\
    Function

# 导入 SymPy 中的量子力学相关模块
from sympy.physics.sho import R_nl
from sympy.physics.hydrogen import R_nl as hydro_nl


# ***************************************************************************
# 计算参数设置
# ***************************************************************************

# 基函数维度大小，即谐振子基底的大小 (n < basis_dimension)
basis_dimension = 5

# 原子单位中的频率的两倍，即振子频率的两倍
omega2 = 0.1

# 轨道角动量量子数 `l` 的值
orbital_momentum_l = 1

# 氢波函数的节点量子数
hydrogen_n = 2

# 径向方向的截断距离
rmax = 20

# 网格中的点数
gridsize = 200

# ***************************************************************************


def main():

    print(__doc__)

    # 使用 IndexedBase 表示数组，Idx 表示索引
    m = Symbol('m', integer=True)  # 符号 m，整数类型
    i = Idx('i', m)                 # 索引符号 i，依赖于 m
    A = IndexedBase('A')            # IndexedBase 对象 A
    B = IndexedBase('B')            # IndexedBase 对象 B
    x = Symbol('x')                 # 符号 x

    print("Compiling ufuncs for radial harmonic oscillator solutions")

    # 设置谐振子解的基础 (对于 l=0)
    basis_ho = {}
    # 对于每个基函数的索引 n，执行以下操作：

        # 为当前 n 的径向谐振子解设置表达式
        expr = R_nl(n, orbital_momentum_l, omega2, x)

        # 将表达式的操作减少到浮点数的 evalf 形式
        expr = expr.evalf(15)

        # 打印输出当前轨道角动量 l 和主量子数 n 的谐振子波函数
        print("The h.o. wave function with l = %i and n = %i is" % (
            orbital_momentum_l, n))
        # 漂亮打印（pretty print）表达式
        pprint(expr)

        # 实现、编译并封装为一个 ufunc 函数
        basis_ho[n] = ufuncify(x, expr)

    # 现在我们尝试用谐振子基函数表达氢原子径向波函数。这是我们将近似的解决方案：
    H_ufunc = ufuncify(x, hydro_nl(hydrogen_n, orbital_momentum_l, 1, x))

    # 转换到另一组基函数可以如下表示，
    #
    #   psi(r) = sum_i c(i) phi_i(r)
    #
    # 其中 psi(r) 是氢解，phi_i(r) 是谐振子解，c(i) 是标量系数。
    #
    # 因此，为了用谐振子基函数表达氢解，我们需要确定系数 c(i)。在位置空间中，这意味着我们需要评估一个积分：
    #
    #  psi(r) = sum_i Integral(R**2*conj(phi(R))*psi(R), (R, 0, oo)) phi_i(r)
    #
    # 使用 autowrap 计算积分时，我们注意到它包含对所有向量的逐元素求和。
    # 使用 Indexed 类，可以生成在低级别代码中执行求和的 autowrapped 函数。
    # （实际上，生成求和非常简单，正如我们将看到的，通常需要额外步骤来避免它们。）
    # 对于谐振子基函数中的每个波函数，我们需要一个积分 ufunc 函数
    binary_integrator = {}
    for n in range(basis_dimension):
        #
        # 设置基础波函数
        #
        # 为了在低级代码中获得内联表达式，我们使用implemented_function将波函数表达式附加到常规的SymPy函数上。
        # 这是一个额外的步骤，用于避免波函数表达式中的错误求和。
        #
        # 这样的函数对象携带它们表示的表达式，但除非采取显式措施，否则不会公开表达式。
        # 好处是搜索重复索引以进行收缩的例程不会搜索波函数表达式。
        psi_ho = implemented_function('psi_ho',
                Lambda(x, R_nl(n, orbital_momentum_l, omega2, x)))

        # 我们用一个数组表示氢原子函数，这将成为二进制例程的输入参数。
        # 这样做可以让积分器为我们抛出的任何波函数找到 h.o. 基础系数。
        psi = IndexedBase('psi')

        #
        # 设置积分表达式
        #

        step = Symbol('step')  # 使用符号步长以增加灵活性

        # 让 i 表示网格数组的索引，A 表示网格数组。然后我们可以通过以下表达式的求和来近似积分
        # （简化的矩形法则，忽略端点修正）：
        expr = A[i]**2 * psi_ho(A[i]) * psi[i] * step

        if n == 0:
            print("Setting up binary integrators for the integral:")
            pprint(Integral(x**2 * psi_ho(x) * Function('psi')(x), (x, 0, oo)))

        # 自动包装它。对于接受多个参数的函数，最好使用'args'关键字，这样您就知道包装函数的签名。
        # （维度 m 将是一个可选参数，但必须在 args 列表中存在。）
        binary_integrator[n] = autowrap(expr, args=[A.label, psi.label, step, m])

        # 看看它在网格维度上的收敛性
        print("Checking convergence of integrator for n = %i" % n)
        for g in range(3, 8):
            grid, step = np.linspace(0, rmax, 2**g, retstep=True)
            print("grid dimension %5i, integral = %e" % (2**g,
                    binary_integrator[n](grid, H_ufunc(grid), step)))

    print("A binary integrator has been set up for each basis state")
    print("We will now use them to reconstruct a hydrogen solution.")

    # 注意：到目前为止，我们不需要指定网格或使用 gridsize
    grid, stepsize = np.linspace(0, rmax, gridsize, retstep=True)

    print("Calculating coefficients with gridsize = %i and stepsize %f" % (
        len(grid), stepsize))

    coeffs = {}
    # 对每个基函数的系数进行计算
    for n in range(basis_dimension):
        # 使用对应的二进制积分器计算系数
        coeffs[n] = binary_integrator[n](grid, H_ufunc(grid), stepsize)
        # 打印每个系数的值
        print("c(%i) = %e" % (n, coeffs[n]))

    # 构造近似的氢波函数
    print("Constructing the approximate hydrogen wave")
    hydro_approx = 0
    all_steps = {}
    # 根据基函数和系数构建近似的氢波函数
    for n in range(basis_dimension):
        hydro_approx += basis_ho[n](grid)*coeffs[n]
        # 记录每个步骤的近似值
        all_steps[n] = hydro_approx.copy()
        # 如果需要绘图，则在图上绘制当前 n 的近似曲线
        if pylab:
            line = pylab.plot(grid, all_steps[n], ':', label='max n = %i' % n)

    # 数值上检查误差
    diff = np.max(np.abs(hydro_approx - H_ufunc(grid)))
    # 打印估计的误差，显示最大偏差的元素
    print("Error estimate: the element with largest deviation misses by %f" % diff)
    if diff > 0.01:
        print("This is much, try to increase the basis size or adjust omega")
    else:
        print("Ah, that's a pretty good approximation!")

    # 可视化检查
    if pylab:
        # 如果需要绘图，则展示每个 n 的贡献
        print("Here's a plot showing the contribution for each n")
        # 将线条样式设为实线
        line[0].set_linestyle('-')
        # 在图上绘制精确的 H_ufunc 曲线
        pylab.plot(grid, H_ufunc(grid), 'r-', label='exact')
        # 添加图例
        pylab.legend()
        # 显示图形
        pylab.show()

    # 打印关于二进制积分器的说明
    print("""Note:
    These binary integrators were specialized to find coefficients for a
    harmonic oscillator basis, but they can process any wave function as long
    as it is available as a vector and defined on a grid with equidistant
    points. That is, on any grid you get from numpy.linspace.

    To make the integrators even more flexible, you can setup the harmonic
    oscillator solutions with symbolic parameters omega and l.  Then the
    autowrapped binary routine will take these scalar variables as arguments,
    so that the integrators can find coefficients for *any* isotropic harmonic
    oscillator basis.

    """)
# 如果当前脚本作为主程序运行（而不是被导入到其他模块中），则执行下面的代码块
if __name__ == '__main__':
    # 调用主函数 main()
    main()
```