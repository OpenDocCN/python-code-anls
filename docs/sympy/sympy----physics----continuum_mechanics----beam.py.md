# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\beam.py`

```
"""
This module can be used to solve 2D beam bending problems with
singularity functions in mechanics.
"""

from sympy.core import S, Symbol, diff, symbols  # 导入 SymPy 核心模块的相关功能
from sympy.core.add import Add  # 导入 SymPy 加法模块
from sympy.core.expr import Expr  # 导入 SymPy 表达式模块
from sympy.core.function import (Derivative, Function)  # 导入 SymPy 函数模块
from sympy.core.mul import Mul  # 导入 SymPy 乘法模块
from sympy.core.relational import Eq  # 导入 SymPy 关系运算模块
from sympy.core.sympify import sympify  # 导入 SymPy 符号化模块
from sympy.solvers import linsolve  # 导入 SymPy 线性求解模块
from sympy.solvers.ode.ode import dsolve  # 导入 SymPy 求解常微分方程模块
from sympy.solvers.solvers import solve  # 导入 SymPy 求解模块
from sympy.printing import sstr  # 导入 SymPy 打印模块
from sympy.functions import SingularityFunction, Piecewise, factorial  # 导入 SymPy 函数模块
from sympy.integrals import integrate  # 导入 SymPy 积分模块
from sympy.series import limit  # 导入 SymPy 极限模块
from sympy.plotting import plot, PlotGrid  # 导入 SymPy 绘图模块
from sympy.geometry.entity import GeometryEntity  # 导入 SymPy 几何模块
from sympy.external import import_module  # 导入 SymPy 外部模块导入功能
from sympy.sets.sets import Interval  # 导入 SymPy 集合模块
from sympy.utilities.lambdify import lambdify  # 导入 SymPy 函数转换为可调用函数模块
from sympy.utilities.decorator import doctest_depends_on  # 导入 SymPy 装饰器模块
from sympy.utilities.iterables import iterable  # 导入 SymPy 可迭代模块
import warnings  # 导入警告模块


__doctest_requires__ = {
    ('Beam.draw',
     'Beam.plot_bending_moment',
     'Beam.plot_deflection',
     'Beam.plot_ild_moment',
     'Beam.plot_ild_shear',
     'Beam.plot_shear_force',
     'Beam.plot_shear_stress',
     'Beam.plot_slope'): ['matplotlib'],
}

numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})  # 导入 NumPy 模块，并设置导入参数

class Beam:
    """
    A Beam is a structural element that is capable of withstanding load
    primarily by resisting against bending. Beams are characterized by
    their cross sectional profile(Second moment of area), their length
    and their material.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention. However, the
       chosen sign convention must respect the rule that, on the positive
       side of beam's axis (in respect to current section), a loading force
       giving positive shear yields a negative moment, as below (the
       curved arrow shows the positive moment and rotation):

    .. image:: allowed-sign-conventions.png

    Examples
    ========
    There is a beam of length 4 meters. A constant distributed load of 6 N/m
    is applied from half of the beam till the end. There are two simple supports
    below the beam, one at the starting point and another at the ending point
    of the beam. The deflection of the beam at the end is restricted.

    Using the sign convention of downwards forces being positive.

    >>> from sympy.physics.continuum_mechanics.beam import Beam  # 导入 Beam 类
    >>> from sympy import symbols, Piecewise  # 导入符号和分段函数
    >>> E, I = symbols('E, I')  # 定义符号 E 和 I
    >>> R1, R2 = symbols('R1, R2')  # 定义符号 R1 和 R2
    >>> b = Beam(4, E, I)  # 创建长度为 4 米，弹性模量为 E，惯性矩 I 的梁对象
    >>> b.apply_load(R1, 0, -1)  # 在位置 0 处施加反向的载荷 R1
    >>> b.apply_load(6, 2, 0)  # 在位置 2 处施加大小为 6 的分布载荷
    >>> b.apply_load(R2, 4, -1)  # 在位置 4 处施加反向的载荷 R2
    >>> b.bc_deflection = [(0, 0), (4, 0)]  # 设置边界条件，即挠度为 0 的位置
    >>> b.boundary_conditions
    {'bending_moment': [], 'deflection': [(0, 0), (4, 0)], 'shear_force': [], 'slope': []}
    >>> b.load  # 输出应用的载荷
"""
    R1*SingularityFunction(x, 0, -1) + R2*SingularityFunction(x, 4, -1) + 6*SingularityFunction(x, 2, 0)
    # 计算受力平衡方程中的加载项，包括反力 R1 和 R2，以及分布载荷的 SingularityFunction
    >>> b.solve_for_reaction_loads(R1, R2)
    # 调用方法计算反力 R1 和 R2 的值，使受力平衡方程成立
    >>> b.load
    # 返回受力平衡方程中的加载项表达式
    -3*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 2, 0) - 9*SingularityFunction(x, 4, -1)
    # 返回负载的合成表达式，包括 SingularityFunction 的项
    >>> b.shear_force()
    # 计算梁的剪力图表达式
    3*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 2, 1) + 9*SingularityFunction(x, 4, 0)
    # 返回梁的弯矩图表达式
    >>> b.bending_moment()
    3*SingularityFunction(x, 0, 1) - 3*SingularityFunction(x, 2, 2) + 9*SingularityFunction(x, 4, 1)
    # 返回梁的挠度图表达式
    >>> b.slope()
    (-3*SingularityFunction(x, 0, 2)/2 + SingularityFunction(x, 2, 3) - 9*SingularityFunction(x, 4, 2)/2 + 7)/(E*I)
    # 返回梁的挠度图表达式，采用 Piecewise 形式
    >>> b.deflection()
    (7*x - SingularityFunction(x, 0, 3)/2 + SingularityFunction(x, 2, 4)/4 - 3*SingularityFunction(x, 4, 3)/2)/(E*I)
    # 返回梁的挠度图表达式，采用 Piecewise 形式
    >>> b.deflection().rewrite(Piecewise)
    (7*x - Piecewise((x**3, x >= 0), (0, True))/2
         - 3*Piecewise(((x - 4)**3, x >= 4), (0, True))/2
         + Piecewise(((x - 2)**4, x >= 2), (0, True))/4)/(E*I)

    # 计算完全符号化梁的支持反力，其长度为 L
    # 梁的下方有两个简支，一个位于起点，另一个位于终点
    # 梁的端点挠度被限制
    # 梁的加载包括：
    # * 在 L/4 处施加的向下点载荷 P1
    # * 在 L/8 处施加的向上点载荷 P2
    # * 在 L/2 处施加的逆时针矩 M1
    # * 在 3*L/4 处施加的顺时针矩 M2
    # * 从 L/2 到 3*L/4 区间内施加的向下常量分布载荷 q1
    # * 从 3*L/4 到 L 区间内施加的向上常量分布载荷 q2
    # 对符号化载荷不做任何假设，但定义正长度有助于算法计算解决方案

    >>> E, I = symbols('E, I')
    # 定义符号 E 和 I
    >>> L = symbols("L", positive=True)
    # 定义正的符号长度 L
    >>> P1, P2, M1, M2, q1, q2 = symbols("P1, P2, M1, M2, q1, q2")
    # 定义符号载荷 P1, P2, M1, M2, q1, q2
    >>> R1, R2 = symbols('R1, R2')
    # 定义符号反力 R1 和 R2
    >>> b = Beam(L, E, I)
    # 创建一个梁对象 b，具有长度 L、弹性模量 E 和惯性矩 I
    >>> b.apply_load(R1, 0, -1)
    # 在梁的起点施加反力 R1
    >>> b.apply_load(R2, L, -1)
    # 在梁的终点施加反力 R2
    >>> b.apply_load(P1, L/4, -1)
    # 在梁的长度 L/4 处施加向下点载荷 P1
    >>> b.apply_load(-P2, L/8, -1)
    # 在梁的长度 L/8 处施加向上点载荷 -P2
    >>> b.apply_load(M1, L/2, -2)
    # 在梁的长度 L/2 处施加逆时针矩 M1
    >>> b.apply_load(-M2, 3*L/4, -2)
    # 在梁的长度 3*L/4 处施加顺时针矩 -M2
    >>> b.apply_load(q1, L/2, 0, 3*L/4)
    # 在梁的长度 L/2 到 3*L/4 区间内施加向下常量分布载荷 q1
    >>> b.apply_load(-q2, 3*L/4, 0, L)
    # 在梁的长度 3*L/4 到 L 区间内施加向上常量分布载荷 -q2
    >>> b.bc_deflection = [(0, 0), (L, 0)]
    # 定义梁的挠度边界条件为起点和终点挠度为零
    >>> b.solve_for_reaction_loads(R1, R2)
    # 调用方法计算反力 R1 和 R2 的值，使受力平衡方程成立
    >>> print(b.reaction_loads[R1])
    # 打印反力 R1 的计算结果
    (-3*L**2*q1 + L**2*q2 - 24*L*P1 + 28*L*P2 - 32*M1 + 32*M2)/(32*L)
    # 计算并打印反力 R1 的值
    >>> print(b.reaction_loads[R2])
    # 打印反力 R2 的计算结果
    (-5*L**2*q1 + 7*L**2*q2 - 8*L*P1 + 4*L*P2 + 32*M1 - 32*M2)/(32*L)
    # 计算并打印反力 R2 的值

    def __str__(self):
    # 定义类方法 __str__，返回对象的描述字符串
        shape_description = self._cross_section if self._cross_section else self._second_moment
        # 根据横截面或第二矩的存在选择描述信息
        str_sol = 'Beam({}, {}, {})'.format(sstr(self._length), sstr(self._elastic_modulus), sstr(shape_description))
        # 格式化字符串，描述梁的长度、弹性模量和形状描述
        return str_sol
        # 返回格式化后的描述字符串

    @property
    def reaction_loads(self):
    # 定义属性方法 reaction_loads，返回对象的反力字典
        """ Returns the reaction forces in a dictionary."""
        # 返回值说明：返回一个包含反力的字典
        return self._reaction_loads
        # 返回对象内部存储的反力字典
    @property
    def rotation_jumps(self):
        """
        Returns the value for the rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._rotation_jumps

    @property
    def deflection_jumps(self):
        """
        Returns the deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._deflection_jumps

    @property
    def ild_shear(self):
        """
        Returns the I.L.D. shear equation.
        """
        return self._ild_shear

    @property
    def ild_reactions(self):
        """
        Returns the I.L.D. reaction forces in a dictionary.
        """
        return self._ild_reactions

    @property
    def ild_rotation_jumps(self):
        """
        Returns the I.L.D. rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._ild_rotations_jumps

    @property
    def ild_deflection_jumps(self):
        """
        Returns the I.L.D. deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._ild_deflection_jumps

    @property
    def ild_moment(self):
        """
        Returns the I.L.D. moment equation.
        """
        return self._ild_moment

    @property
    def length(self):
        """
        Length of the Beam.
        """
        return self._length

    @length.setter
    def length(self, l):
        """
        Setter for the length of the Beam.
        """
        self._length = sympify(l)

    @property
    def area(self):
        """
        Cross-sectional area of the Beam.
        """
        return self._area

    @area.setter
    def area(self, a):
        """
        Setter for the cross-sectional area of the Beam.
        """
        self._area = sympify(a)

    @property
    def variable(self):
        """
        A symbol that can be used as a variable along the length of the beam
        while representing load distribution, shear force curve, bending
        moment, slope curve and the deflection curve. By default, it is set
        to ``Symbol('x')``, but this property is mutable.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I, A = symbols('E, I, A')
        >>> x, y, z = symbols('x, y, z')
        >>> b = Beam(4, E, I)
        >>> b.variable
        x
        >>> b.variable = y
        >>> b.variable
        y
        >>> b = Beam(4, E, I, A, z)
        >>> b.variable
        z
        """
        return self._variable

    @variable.setter
    def variable(self, v):
        # 如果输入的变量 v 是 Symbol 对象，则将其赋值给实例变量 _variable
        if isinstance(v, Symbol):
            self._variable = v
        else:
            # 如果 v 不是 Symbol 对象，则抛出 TypeError 异常
            raise TypeError("""The variable should be a Symbol object.""")

    @property
    def elastic_modulus(self):
        """返回梁的杨氏模量。"""
        # 返回实例变量 _elastic_modulus 的值
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, e):
        # 使用 sympify 将输入的 e 转换为符号表达式，并赋值给实例变量 _elastic_modulus
        self._elastic_modulus = sympify(e)

    @property
    def second_moment(self):
        """返回梁的截面二阶矩。"""
        # 返回实例变量 _second_moment 的值
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        # 设置实例变量 _cross_section 为 None
        self._cross_section = None
        # 如果输入的 i 是 GeometryEntity 对象，则抛出 ValueError 异常
        if isinstance(i, GeometryEntity):
            raise ValueError("To update cross-section geometry use `cross_section` attribute")
        else:
            # 使用 sympify 将输入的 i 转换为符号表达式，并赋值给实例变量 _second_moment
            self._second_moment = sympify(i)

    @property
    def cross_section(self):
        """返回梁的截面形状。"""
        # 返回实例变量 _cross_section 的值
        return self._cross_section

    @cross_section.setter
    def cross_section(self, s):
        # 如果输入的 s 不为 None，则计算其二阶矩并赋值给实例变量 _second_moment
        if s:
            self._second_moment = s.second_moment_of_area()[0]
        # 将输入的 s 赋值给实例变量 _cross_section
        self._cross_section = s

    @property
    def boundary_conditions(self):
        """
        返回施加在梁上的边界条件字典。
        字典包含三个关键字：moment（弯矩）、slope（斜率）、deflection（挠度）。
        每个关键字的值是一个元组列表，其中每个元组包含边界条件的位置和值，格式为 (位置, 值)。
        """
        # 返回实例变量 _boundary_conditions 的值
        return self._boundary_conditions

    @property
    def bc_shear_force(self):
        # 返回 _boundary_conditions 字典中 'shear_force' 键的值
        return self._boundary_conditions['shear_force']

    @bc_shear_force.setter
    def bc_shear_force(self, sf_bcs):
        # 将输入的 sf_bcs 赋值给 _boundary_conditions 字典中 'shear_force' 键
        self._boundary_conditions['shear_force'] = sf_bcs

    @property
    def bc_bending_moment(self):
        # 返回 _boundary_conditions 字典中 'bending_moment' 键的值
        return self._boundary_conditions['bending_moment']

    @bc_bending_moment.setter
    def bc_bending_moment(self, bm_bcs):
        # 将输入的 bm_bcs 赋值给 _boundary_conditions 字典中 'bending_moment' 键
        self._boundary_conditions['bending_moment'] = bm_bcs

    @property
    def bc_slope(self):
        # 返回 _boundary_conditions 字典中 'slope' 键的值
        return self._boundary_conditions['slope']

    @bc_slope.setter
    def bc_slope(self, s_bcs):
        # 将输入的 s_bcs 赋值给 _boundary_conditions 字典中 'slope' 键
        self._boundary_conditions['slope'] = s_bcs
    # 定义一个方法 `bc_deflection`，用于获取 `self._boundary_conditions` 中 'deflection' 键对应的值
    def bc_deflection(self):
        return self._boundary_conditions['deflection']

    # 使用 `@property.setter` 装饰器定义 `bc_deflection` 方法的 setter 方法
    @bc_deflection.setter
    # 设置 `bc_deflection` 方法的 setter 方法，用于设置 `self._boundary_conditions` 中 'deflection' 键对应的值
    def bc_deflection(self, d_bcs):
        self._boundary_conditions['deflection'] = d_bcs
    def apply_support(self, loc, type="fixed"):
        """
        This method applies support to a particular beam object and returns
        the symbol of the unknown reaction load(s).

        Parameters
        ==========
        loc : Sympifyable
            Location of point at which support is applied.
        type : String
            Determines type of Beam support applied. To apply support structure
            with
            - zero degree of freedom, type = "fixed"
            - one degree of freedom, type = "pin"
            - two degrees of freedom, type = "roller"

        Returns
        =======
        Symbol or tuple of Symbol
            The unknown reaction load as a symbol.
            - Symbol(reaction_force) if type = "pin" or "roller"
            - Symbol(reaction_force), Symbol(reaction_moment) if type = "fixed"

        Examples
        ========
        There is a beam of length 20 meters. A moment of magnitude 100 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at a distance of 10 meters.
        There is one fixed support at the start of the beam and a roller at the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(20, E, I)
        >>> p0, m0 = b.apply_support(0, 'fixed')
        >>> p1 = b.apply_support(20, 'roller')
        >>> b.apply_load(-8, 10, -1)
        >>> b.apply_load(100, 20, -2)
        >>> b.solve_for_reaction_loads(p0, m0, p1)
        >>> b.reaction_loads
        {M_0: 20, R_0: -2, R_20: 10}
        >>> b.reaction_loads[p0]
        -2
        >>> b.load
        20*SingularityFunction(x, 0, -2) - 2*SingularityFunction(x, 0, -1)
        - 8*SingularityFunction(x, 10, -1) + 100*SingularityFunction(x, 20, -2)
        + 10*SingularityFunction(x, 20, -1)
        """
        # 将位置参数转换为符号表达式
        loc = sympify(loc)

        # 将支持的位置和类型添加到支持列表中
        self._applied_supports.append((loc, type))

        # 根据支持类型决定处理方式
        if type in ("pin", "roller"):
            # 如果是pin或roller支持，创建反力符号并将其应用在指定位置上
            reaction_load = Symbol('R_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            # 将边界条件加入到挠度边界条件列表中
            self.bc_deflection.append((loc, 0))
        else:
            # 如果是fixed支持，创建反力和反弯矩符号并将其应用在指定位置上
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.apply_load(reaction_moment, loc, -2)
            # 将边界条件加入到挠度和斜率边界条件列表中
            self.bc_deflection.append((loc, 0))
            self.bc_slope.append((loc, 0))
            # 将支持作为外力应用到加载列表中
            self._support_as_loads.append((reaction_moment, loc, -2, None))

        # 将反力作为外力应用到加载列表中
        self._support_as_loads.append((reaction_load, loc, -1, None))

        # 根据支持类型返回相应的反力符号或反力和反弯矩符号
        if type in ("pin", "roller"):
            return reaction_load
        else:
            return reaction_load, reaction_moment
    def _get_I(self, loc):
        """
        Helper function that returns the Second moment (I) at a location in the beam.
        """
        # 获取给定位置 loc 处的截面惯性矩 I
        I = self.second_moment
        # 如果 I 不是 Piecewise 对象，则直接返回 I
        if not isinstance(I, Piecewise):
            return I
        else:
            # 遍历 Piecewise 对象的参数
            for i in range(len(I.args)):
                # 如果 loc 小于等于当前参数的上限，返回该参数的第一个值（即截面惯性矩）
                if loc <= I.args[i][1].args[1]:
                    return I.args[i][0]

    def apply_rotation_hinge(self, loc):
        """
        This method applies a rotation hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown rotation jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 15 meters. Pin supports are placed at distances
        of 0 and 10 meters. There is a fixed support at the end. There are two rotation hinges
        in the structure, one at 5 meters and one at 10 meters. A pointload of magnitude
        10 kN is applied on the hinge at 5 meters. A distributed load of 5 kN works on
        the structure from 10 meters to the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import Symbol
        >>> E = Symbol('E')
        >>> I = Symbol('I')
        >>> b = Beam(15, E, I)
        >>> r0 = b.apply_support(0, type='pin')
        >>> r10 = b.apply_support(10, type='pin')
        >>> r15, m15 = b.apply_support(15, type='fixed')
        >>> p5 = b.apply_rotation_hinge(5)
        >>> p12 = b.apply_rotation_hinge(12)
        >>> b.apply_load(-10, 5, -1)
        >>> b.apply_load(-5, 10, 0, 15)
        >>> b.solve_for_reaction_loads(r0, r10, r15, m15)
        >>> b.reaction_loads
        {M_15: -75/2, R_0: 0, R_10: 40, R_15: -5}
        >>> b.rotation_jumps
        {P_12: -1875/(16*E*I), P_5: 9625/(24*E*I)}
        >>> b.rotation_jumps[p12]
        -1875/(16*E*I)
        >>> b.bending_moment()
        -9625*SingularityFunction(x, 5, -1)/24 + 10*SingularityFunction(x, 5, 1)
        - 40*SingularityFunction(x, 10, 1) + 5*SingularityFunction(x, 10, 2)/2
        + 1875*SingularityFunction(x, 12, -1)/16 + 75*SingularityFunction(x, 15, 0)/2
        + 5*SingularityFunction(x, 15, 1) - 5*SingularityFunction(x, 15, 2)/2
        """
        # 将 loc 转换为 sympy 符号
        loc = sympify(loc)
        # 获取 loc 处的弹性模量 E 和截面惯性矩 I
        E = self.elastic_modulus
        I = self._get_I(loc)

        # 创建旋转跳跃符号 P_loc
        rotation_jump = Symbol('P_'+str(loc))
        # 记录应用的旋转铰链位置
        self._applied_rotation_hinges.append(loc)
        # 将旋转跳跃符号添加到旋转铰链符号列表
        self._rotation_hinge_symbols.append(rotation_jump)
        # 应用加载，加载大小为 E * I * rotation_jump，作用于 loc 处，加载类型为 -3
        self.apply_load(E * I * rotation_jump, loc, -3)
        # 将 loc 添加到弯曲矩边界条件的列表中，边界条件为 (loc, 0)
        self.bc_bending_moment.append((loc, 0))
        # 返回旋转跳跃符号
        return rotation_jump
    def apply_sliding_hinge(self, loc):
        """
        This method applies a sliding hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown deflection jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 13 meters. A fixed support is placed at the beginning.
        There is a pin support at the end. There is a sliding hinge at a location of 8 meters.
        A pointload of magnitude 10 kN is applied on the hinge at 5 meters.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> b = Beam(13, 20, 20)
        >>> r0, m0 = b.apply_support(0, type="fixed")
        >>> s8 = b.apply_sliding_hinge(8)
        >>> r13 = b.apply_support(13, type="pin")
        >>> b.apply_load(-10, 5, -1)
        >>> b.solve_for_reaction_loads(r0, m0, r13)
        >>> b.reaction_loads
        {M_0: -50, R_0: 10, R_13: 0}
        >>> b.deflection_jumps
        {W_8: 85/24}
        >>> b.deflection_jumps[s8]
        85/24
        >>> b.bending_moment()
        50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
        + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3
        >>> b.deflection()
        -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
        - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24
        """
        # 使用 sympify 将 loc 转换为符号表达式
        loc = sympify(loc)
        # 获取弹性模量
        E = self.elastic_modulus
        # 获取惯性矩
        I = self._get_I(loc)

        # 定义偏转跳跃为以 loc 为位置的符号变量
        deflection_jump = Symbol('W_' + str(loc))
        # 将 loc 添加到已应用的滑动铰链列表中
        self._applied_sliding_hinges.append(loc)
        # 将偏转跳跃符号添加到滑动铰链符号列表中
        self._sliding_hinge_symbols.append(deflection_jump)
        # 在 loc 处施加载荷，大小为 E * I * deflection_jump，作用方向为 -4
        self.apply_load(E * I * deflection_jump, loc, -4)
        # 将 loc 处的横向剪力为零添加到边界条件的横向剪力列表中
        self.bc_shear_force.append((loc, 0))
        # 返回偏转跳跃符号
        return deflection_jump
    # 定义一个方法，用于将加载应用到特定的梁对象上
    """
    This method adds up the loads given to a particular beam object.

    Parameters
    ==========
    value : Sympifyable
        The value inserted should have the units [Force/(Distance**(n+1)]
        where n is the order of applied load.
        Units for applied loads:

           - For moments, unit = kN*m
           - For point loads, unit = kN
           - For constant distributed load, unit = kN/m
           - For ramp loads, unit = kN/m/m
           - For parabolic ramp loads, unit = kN/m/m/m
           - ... so on.

    start : Sympifyable
        The starting point of the applied load. For point moments and
        point forces this is the location of application.
    order : Integer
        The order of the applied load.

           - For moments, order = -2
           - For point loads, order =-1
           - For constant distributed load, order = 0
           - For ramp loads, order = 1
           - For parabolic ramp loads, order = 2
           - ... so on.

    end : Sympifyable, optional
        An optional argument that can be used if the load has an end point
        within the length of the beam.

    Examples
    ========
    There is a beam of length 4 meters. A moment of magnitude 3 Nm is
    applied in the clockwise direction at the starting point of the beam.
    A point load of magnitude 4 N is applied from the top of the beam at
    2 meters from the starting point and a parabolic ramp load of magnitude
    2 N/m is applied below the beam starting from 2 meters to 3 meters
    away from the starting point of the beam.

    >>> from sympy.physics.continuum_mechanics.beam import Beam
    >>> from sympy import symbols
    >>> E, I = symbols('E, I')
    >>> b = Beam(4, E, I)
    >>> b.apply_load(-3, 0, -2)
    >>> b.apply_load(4, 2, -1)
    >>> b.apply_load(-2, 2, 2, end=3)
    >>> b.load
    -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)

    """
    # 获取梁对象的变量 x
    x = self.variable
    # 将 value、start、order 转换为符号表达式
    value = sympify(value)
    start = sympify(start)
    order = sympify(order)

    # 将加载的信息添加到应用加载列表中
    self._applied_loads.append((value, start, order, end))
    # 更新总加载量，加上新加载的 SingularityFunction
    self._load += value*SingularityFunction(x, start, order)
    # 更新原始加载量，同样加上新加载的 SingularityFunction
    self._original_load += value*SingularityFunction(x, start, order)

    if end:
        # 如果加载有一个结束点在梁的长度范围内，则处理结束点
        self._handle_end(x, value, start, order, end, type="apply")
    def remove_load(self, value, start, order, end=None):
        """
        This method removes a particular load present on the beam object.
        Returns a ValueError if the load passed as an argument is not
        present on the beam.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.
            - For moments, order= -2
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
        >>> b.remove_load(-2, 2, 2, end = 3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)

        Removes a load from the beam object identified by its value, start point,
        order, and optionally its end point. If the specified load is found,
        it updates internal load representations and removes the load from the
        list of applied loads.

        """

        x = self.variable
        value = sympify(value)  # Convert value to a sympy expression
        start = sympify(start)  # Convert start point to a sympy expression
        order = sympify(order)  # Convert order to a sympy expression

        # Check if the load (value, start, order, end) exists in the list of applied loads
        if (value, start, order, end) in self._applied_loads:
            # Subtract the load from internal load representations
            self._load -= value * SingularityFunction(x, start, order)
            self._original_load -= value * SingularityFunction(x, start, order)
            # Remove the load from the list of applied loads
            self._applied_loads.remove((value, start, order, end))
        else:
            # Raise an error if the specified load doesn't exist on the beam object
            msg = "No such load distribution exists on the beam object."
            raise ValueError(msg)

        if end:
            # If end point is provided, handle the end point of the load removal
            self._handle_end(x, value, start, order, end, type="remove")
    @property
    def load(self):
        """
        Returns a Singularity Function expression which represents
        the load distribution curve of the Beam object.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 3 meters away from the
        starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 3, 2)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 3, 2)
        """
        return self._load



# 返回 Beam 对象的加载分布曲线，由 Singularity Function 表达式表示
@property
def load(self):
    # 返回加载分布的表达式，由 Singularity Function 表达式组成，用于 Beam 对象
    return self._load
    def applied_loads(self):
        """
        Returns a list of all loads applied on the beam object.
        Each load in the list is a tuple of form (value, start, order, end).

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point. Another pointload of magnitude 5 N
        is applied at same position.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(5, 2, -1)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 9*SingularityFunction(x, 2, -1)
        >>> b.applied_loads
        [(-3, 0, -2, None), (4, 2, -1, None), (5, 2, -1, None)]
        """
        return self._applied_loads



    def shear_force(self):
        """
        Returns a Singularity Function expression which represents
        the shear force curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.shear_force()
        8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
        """
        x = self.variable
        # 返回表示梁对象剪力曲线的奇异函数表达式
        return -integrate(self.load, x)
    def max_shear_force(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        
        # 计算剪力曲线
        shear_curve = self.shear_force()
        x = self.variable

        # 获取剪力函数的项
        terms = shear_curve.args
        singularity = []        # Points at which shear function changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of shear force
        shear_values = []   # List of values of shear force in each interval
        
        # 遍历奇异点，计算剪力的不同区间
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                # 计算剪力的斜率
                shear_slope = Piecewise((float("nan"), x<=singularity[i-1]),(self._load.rewrite(Piecewise), x<s), (float("nan"), True))
                points = solve(shear_slope, x)
                val = []
                for point in points:
                    val.append(abs(shear_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(shear_curve, x, singularity[i-1], '+')), abs(limit(shear_curve, x, s, '-'))]
                max_shear = max(val)
                shear_values.append(max_shear)
                intervals.append(points[val.index(max_shear)])
            # 如果在某个区间剪力为零或具有常数斜率，则 solve 函数会抛出 NotImplementedError
            except NotImplementedError:
                initial_shear = limit(shear_curve, x, singularity[i-1], '+')
                final_shear = limit(shear_curve, x, s, '-')
                # 如果剪力曲线具有恒定的斜率（即为直线）
                if shear_curve.subs(x, (singularity[i-1] + s)/2) == (initial_shear + final_shear)/2 and initial_shear != final_shear:
                    shear_values.extend([initial_shear, final_shear])
                    intervals.extend([singularity[i-1], s])
                else:    # 在整个区间内剪力曲线具有相同的值
                    shear_values.append(final_shear)
                    intervals.append(Interval(singularity[i-1], s))

        shear_values = list(map(abs, shear_values))
        maximum_shear = max(shear_values)
        point = intervals[shear_values.index(maximum_shear)]
        return (point, maximum_shear)
    def bending_moment(self):
        """
        Returns a Singularity Function expression which represents
        the bending moment curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.bending_moment()
        8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
        """
        x = self.variable  # 获取变量 x，通常用于表示横跨梁的位置坐标
        return integrate(self.shear_force(), x)  # 返回剪力函数关于 x 的积分，表示弯矩曲线的Singularity Function表达式
    def max_bmoment(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        
        bending_curve = self.bending_moment()   # 获取弯矩曲线

        x = self.variable   # 获取变量 x

        terms = bending_curve.args   # 弯矩曲线的项列表
        singularity = []   # 弯矩变化的点列表，即奇异点
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # 提取项中的 SingularityFunction
            singularity.append(term.args[1])   # 将奇异点添加到列表中
        singularity = list(set(singularity))   # 去重奇异点
        singularity.sort()   # 对奇异点进行排序

        intervals = []    # 弯矩离散值的区间列表
        moment_values = []   # 每个区间中弯矩值的列表
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                # 计算在当前区间内的弯矩斜率
                moment_slope = Piecewise(
                    (float("nan"), x <= singularity[i - 1]),
                    (self.shear_force().rewrite(Piecewise), x < s),
                    (float("nan"), True))
                points = solve(moment_slope, x)   # 求解斜率为零的点
                val = []
                for point in points:
                    val.append(abs(bending_curve.subs(x, point)))   # 计算在点处的绝对值弯矩
                points.extend([singularity[i-1], s])   # 添加区间的端点
                val += [abs(limit(bending_curve, x, singularity[i-1], '+')), abs(limit(bending_curve, x, s, '-'))]   # 添加端点处的绝对值弯矩
                max_moment = max(val)   # 获取最大的绝对值弯矩
                moment_values.append(max_moment)   # 将最大弯矩添加到列表中
                intervals.append(points[val.index(max_moment)])   # 将对应的区间端点添加到列表中

            # 如果在某个区间内弯矩为零或有恒定的斜率，则 solve 函数会抛出 NotImplementedError
            except NotImplementedError:
                initial_moment = limit(bending_curve, x, singularity[i-1], '+')   # 计算区间起点的极限弯矩
                final_moment = limit(bending_curve, x, s, '-')   # 计算区间终点的极限弯矩
                # 如果弯矩曲线具有恒定的斜率（即为一条直线）
                if bending_curve.subs(x, (singularity[i-1] + s)/2) == (initial_moment + final_moment)/2 and initial_moment != final_moment:
                    moment_values.extend([initial_moment, final_moment])   # 将起点和终点的弯矩添加到列表中
                    intervals.extend([singularity[i-1], s])   # 将区间端点添加到列表中
                else:    # 如果弯矩曲线在整个区间内具有相同的值
                    moment_values.append(final_moment)   # 将终点的弯矩添加到列表中
                    intervals.append(Interval(singularity[i-1], s))   # 将整个区间添加为 Interval 对象

        moment_values = list(map(abs, moment_values))   # 将所有弯矩值转换为绝对值
        maximum_moment = max(moment_values)   # 获取最大的绝对值弯矩
        point = intervals[moment_values.index(maximum_moment)]   # 获取对应的区间端点
        return (point, maximum_moment)   # 返回最大弯矩及其坐标
    def point_cflexure(self):
        """
        Returns a Set of point(s) with zero bending moment and
        where bending moment curve of the beam object changes
        its sign from negative to positive or vice versa.

        Examples
        ========
        There is is 10 meter long overhanging beam. There are
        two simple supports below the beam. One at the start
        and another one at a distance of 6 meters from the start.
        Point loads of magnitude 10KN and 20KN are applied at
        2 meters and 4 meters from start respectively. A Uniformly
        distribute load of magnitude of magnitude 3KN/m is also
        applied on top starting from 6 meters away from starting
        point till end.
        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(10, E, I)
        >>> b.apply_load(-4, 0, -1)
        >>> b.apply_load(-46, 6, -1)
        >>> b.apply_load(10, 2, -1)
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(3, 6, 0)
        >>> b.point_cflexure()
        [10/3]
        """

        # 计算不包含小于零的奇异函数在内的弯矩方程
        non_singular_bending_moment = sum(arg for arg in self.bending_moment().args if not arg.args[1].args[2] < 0)

        # 将弯矩曲线限制在梁的长度范围内
        moment_curve = Piecewise((float("nan"), self.variable<=0),
                (non_singular_bending_moment, self.variable<self.length),
                (float("nan"), True))
        
        try:
            # 解析弯矩曲线为 Piecewise 形式，求解其中弯矩为零的点
            points = solve(moment_curve.rewrite(Piecewise), self.variable,
                           domain=S.Reals)
        except NotImplementedError as e:
            if "An expression is already zero when" in str(e):
                raise NotImplementedError("This method cannot be used when a whole region of "
                                          "the bending moment line is equal to 0.")
            else:
                raise

        # 返回弯矩为零的点集合
        return points
    def slope(self):
        """
        Returns a Singularity Function expression which represents
        the slope the elastic curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.slope()
        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)
        """
        x = self.variable  # 获取自变量 x
        E = self.elastic_modulus  # 获取弹性模量 E
        I = self.second_moment  # 获取截面惯性矩 I

        if not self._boundary_conditions['slope']:  # 如果未设置斜率边界条件
            return diff(self.deflection(), x)  # 返回位移函数的导数

        if isinstance(I, Piecewise) and self._joined_beam:  # 如果截面惯性矩是分段函数并且为连接梁
            args = I.args  # 获取分段函数的参数列表
            slope = 0  # 初始化斜率为0
            prev_slope = 0  # 初始化上一个斜率为0
            prev_end = 0  # 初始化上一个段的结束点为0
            for i in range(len(args)):  # 遍历分段函数的参数
                if i != 0:
                    prev_end = args[i-1][1].args[1]  # 更新上一个段的结束点
                slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))  # 计算斜率值
                if i != len(args) - 1:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0) - \
                        (prev_slope + slope_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    # 更新斜率表达式
                else:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0)
                    # 更新斜率表达式
                prev_slope = slope_value.subs(x, args[i][1].args[1])  # 更新上一个斜率值
            return slope  # 返回计算出的斜率表达式

        C3 = Symbol('C3')  # 创建符号 C3
        slope_curve = -integrate(S.One/(E*I)*self.bending_moment(), x) + C3  # 计算弹性曲线的斜率

        bc_eqs = []  # 初始化边界条件方程列表
        for position, value in self._boundary_conditions['slope']:  # 遍历斜率边界条件
            eqs = slope_curve.subs(x, position) - value  # 计算每个边界条件方程
            bc_eqs.append(eqs)  # 将方程添加到列表中
        constants = list(linsolve(bc_eqs, C3))  # 解边界条件方程组，得到常数 C3
        slope_curve = slope_curve.subs({C3: constants[0][0]})  # 使用求解得到的常数替换 C3
        return slope_curve  # 返回计算出的斜率曲线表达式
    # 定义一个方法，计算梁的最大挠度点及其对应的挠度值
    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        in a Beam object.
        """

        # 创建一个分段函数，用于限定在梁长度范围内
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope(), self.variable<self.length),
                (float("nan"), True))

        # 求解斜率曲线的变量，获取可能的最大挠度点
        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        
        # 获取挠度曲线
        deflection_curve = self.deflection()
        
        # 计算在可能的最大挠度点处的挠度值
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        
        # 如果有计算出的挠度值，返回最大挠度点及其挠度值
        if len(deflections) != 0:
            max_def = max(deflections)
            return (points[deflections.index(max_def)], max_def)
        else:
            # 如果没有计算出的挠度值，返回 None
            return None

    # 定义一个方法，返回梁对象的剪切应力曲线表达式
    def shear_stress(self):
        """
        Returns an expression representing the Shear Stress
        curve of the Beam object.
        """
        return self.shear_force()/self._area
    def plot_shear_stress(self, subs=None):
        """
        Returns a plot of shear stress present in the beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters and area of cross section 2 square
        meters. A constant distributed load of 10 KN/m is applied from half of
        the beam till the end. There are two simple supports below the beam,
        one at the starting point and another at the ending point of the beam.
        A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6), 2)
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_stress()
            Plot object containing:
            [0]: cartesian line: 6875*SingularityFunction(x, 0, 0) - 2500*SingularityFunction(x, 2, 0)
            - 5000*SingularityFunction(x, 4, 1) + 15625*SingularityFunction(x, 8, 0)
            + 5000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """

        # Calculate shear stress distribution along the beam
        shear_stress = self.shear_stress()
        x = self.variable  # Symbol representing the position along the beam
        length = self.length  # Length of the beam

        if subs is None:
            subs = {}

        # Check if all symbols in shear stress expression have values provided
        for sym in shear_stress.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('value of %s was not passed.' % sym)

        # Substitute length if provided in subs
        if length in subs:
            length = subs[length]

        # Plot shear stress distribution
        # Returns a plot object showing shear stress variation
        return plot(shear_stress.subs(subs), (x, 0, length),
                    title='Shear Stress', xlabel=r'$\mathrm{x}$', ylabel=r'$\tau$',
                    line_color='r')
    def plot_shear_force(self, subs=None):
        """
        返回梁对象中的剪力图。

        Parameters
        ==========
        subs : dictionary
            包含符号作为键和对应值的 Python 字典。

        Examples
        ========
        有一根长度为8米的梁。从梁的中点到末端施加一个10 KN/m的均布载荷。
        梁下有两个简支，一个在起点，另一个在终点。还在梁的顶部距离起点4米处施加一个5 KN的集中力。
        取 E = 200 GPa 和 I = 400*(10**-6) 米**4。

        使用向下的力为正的符号约定。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_force()
            包含绘图对象:
            [0]: 笛卡尔线: 13750*SingularityFunction(x, 0, 0) - 5000*SingularityFunction(x, 2, 0)
            - 10000*SingularityFunction(x, 4, 1) + 31250*SingularityFunction(x, 8, 0)
            + 10000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """
        # 计算剪力图
        shear_force = self.shear_force()
        
        # 如果未提供 subs 参数，则设为空字典
        if subs is None:
            subs = {}
        
        # 验证剪力表达式中的符号是否在 subs 中定义
        for sym in shear_force.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        
        # 如果长度 self.length 在 subs 中有定义，则使用定义的值，否则使用默认长度 self.length
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        
        # 绘制剪力图，替换符号并设置标题、坐标轴标签和线条颜色
        return plot(shear_force.subs(subs), (self.variable, 0, length), title='Shear Force',
                    xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$', line_color='g')
    def plot_bending_moment(self, subs=None):
        """
        返回 Beam 对象中存在的弯矩的绘图。

        Parameters
        ==========
        subs : dictionary
            包含符号作为键和对应值的 Python 字典。

        Examples
        ========
        有一根长度为 8 米的梁。从梁的中点到末端施加一个10 KN/m的常分布载荷。在梁下有两个简支，一个在起点，另一个在终点。
        从梁顶部距起点 4 米处还施加一个 5 KN 的集中力。
        取 E = 200 GPa 和 I = 400*(10**-6) 米**4。

        使用向下的力为正的符号约定。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_bending_moment()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 1) - 5000*SingularityFunction(x, 2, 1)
            - 5000*SingularityFunction(x, 4, 2) + 31250*SingularityFunction(x, 8, 1)
            + 5000*SingularityFunction(x, 8, 2) for x over (0.0, 8.0)
        """
        
        # 计算弯矩
        bending_moment = self.bending_moment()
        
        # 如果 subs 未指定，则设为空字典
        if subs is None:
            subs = {}
        
        # 检查弯矩中的每个符号
        for sym in bending_moment.atoms(Symbol):
            # 如果符号是变量本身，则继续下一个符号
            if sym == self.variable:
                continue
            # 如果符号不在 subs 中，则引发值错误
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        
        # 如果长度在 subs 中，则使用 subs 中的长度；否则使用对象自身的长度
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        
        # 返回弯矩图
        return plot(bending_moment.subs(subs), (self.variable, 0, length), title='Bending Moment',
                    xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$', line_color='b')
    def plot_slope(self, subs=None):
        """
        Returns a plot for slope of deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_slope()
            Plot object containing:
            [0]: cartesian line: -8.59375e-5*SingularityFunction(x, 0, 2) + 3.125e-5*SingularityFunction(x, 2, 2)
            + 2.08333333333333e-5*SingularityFunction(x, 4, 3) - 0.0001953125*SingularityFunction(x, 8, 2)
            - 2.08333333333333e-5*SingularityFunction(x, 8, 3) + 0.00138541666666667 for x over (0.0, 8.0)
        """
        
        # 计算斜率
        slope = self.slope()
        
        # 如果未提供替代值字典，将其设为一个空字典
        if subs is None:
            subs = {}
        
        # 检查斜率表达式中的符号变量
        for sym in slope.atoms(Symbol):
            # 跳过主变量（self.variable）
            if sym == self.variable:
                continue
            # 如果在替代值字典中找不到当前符号变量，引发值错误异常
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' % sym)
        
        # 确定梁的长度替代值
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        
        # 返回斜率图表对象
        return plot(slope.subs(subs), (self.variable, 0, length), title='Slope',
                    xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$', line_color='m')
    def plot_deflection(self, subs=None):
        """
        返回 Beam 对象的挠曲线图。

        Parameters
        ==========
        subs : dictionary
            Python 字典，包含符号变量作为键和对应的值。

        Examples
        ========
        有一根长度为 8 米的梁。从梁的中点到末端施加了一个10 KN/m的均布载荷。
        梁下方有两个简支，一个在起点，另一个在终点。从梁顶部距离起点4米处还施加了一个5 KN的点载荷。
        取 E = 200 GPa 和 I = 400*(10**-6) meter**4。

        使用向下力为正的符号约定。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_deflection()
            Plot object containing:
            [0]: cartesian line: 0.00138541666666667*x - 2.86458333333333e-5*SingularityFunction(x, 0, 3)
            + 1.04166666666667e-5*SingularityFunction(x, 2, 3) + 5.20833333333333e-6*SingularityFunction(x, 4, 4)
            - 6.51041666666667e-5*SingularityFunction(x, 8, 3) - 5.20833333333333e-6*SingularityFunction(x, 8, 4)
            for x over (0.0, 8.0)
        """
        # 计算挠曲函数
        deflection = self.deflection()
        
        # 如果没有传入替换字典，则设为一个空字典
        if subs is None:
            subs = {}
        
        # 检查挠曲函数中的符号变量
        for sym in deflection.atoms(Symbol):
            # 如果符号变量是自变量，则跳过
            if sym == self.variable:
                continue
            # 如果替换字典中没有该符号变量，则抛出数值错误
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        
        # 获取梁的长度，如果在替换字典中有长度值，则使用该值，否则使用对象的长度属性
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        
        # 绘制挠曲线图并返回
        return plot(deflection.subs(subs), (self.variable, 0, length),
                    title='Deflection', xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                    line_color='r')
    def plot_loading_results(self, subs=None):
        """
        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object.

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> axes = b.plot_loading_results()
        """
        # 获取梁的长度
        length = self.length
        # 获取自变量（通常是 x）
        variable = self.variable
        # 如果未提供替换字典，则设为空字典
        if subs is None:
            subs = {}
        # 遍历计算出的挠度函数中的符号，确保每个符号都有相应的值传递进来
        for sym in self.deflection().atoms(Symbol):
            # 如果符号是自变量，则跳过
            if sym == self.variable:
                continue
            # 如果符号不在替换字典中，则抛出 ValueError
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        # 如果长度在替换字典中，则用替换字典中的值替换原来的长度
        if length in subs:
            length = subs[length]
        
        # 绘制四个图：剪力图、弯矩图、斜率图、挠度图，并设置各自的标题、坐标轴标签和线条颜色
        ax1 = plot(self.shear_force().subs(subs), (variable, 0, length),
                   title="Shear Force", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$',
                   line_color='g', show=False)
        ax2 = plot(self.bending_moment().subs(subs), (variable, 0, length),
                   title="Bending Moment", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$',
                   line_color='b', show=False)
        ax3 = plot(self.slope().subs(subs), (variable, 0, length),
                   title="Slope", xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$',
                   line_color='m', show=False)
        ax4 = plot(self.deflection().subs(subs), (variable, 0, length),
                   title="Deflection", xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                   line_color='r', show=False)

        # 返回包含四个子图的 PlotGrid 对象
        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)
    # 定义一个辅助函数，用于解决 I.L.D. 方程
    # 它接受一个值作为参数，用于计算剪力和弯矩方程
    def _solve_for_ild_equations(self, value):
        """
        Helper function for I.L.D. It takes the unsubstituted
        copy of the load equation and uses it to calculate shear force and bending
        moment equations.
        """
        # 获取变量 x 和 ild_variable 的值
        x = self.variable
        a = self.ild_variable
        # 创建一个新的加载方程，将给定的值 value 与 SingularityFunction(x, a, -1) 结合
        load = self._load + value * SingularityFunction(x, a, -1)
        # 计算加载方程的负积分，得到剪力
        shear_force = -integrate(load, x)
        # 计算剪力的积分，得到弯矩
        bending_moment = integrate(shear_force, x)

        # 返回剪力和弯矩
        return shear_force, bending_moment
    def solve_for_ild_shear(self, distance, value, *reactions):
        """
        根据移动荷载的影响线确定指定点处的剪力影响线图方程。

        Parameters
        ==========
        distance : Integer
            点距离梁起点的距离，用于确定方程
        value : Integer
            移动荷载的大小
        reactions :
            施加在梁上的反力。

        Warning
        =======
        当a = 0或a = l（梁的长度l）时，此方法可能会产生不正确的结果。

        Examples
        ========
        有一根长度为12米的梁。梁下有两个简支，一个在起点，另一个在距离8米处。
        计算距离4米处，受到1kN大小移动荷载影响时的剪力影响线图方程。

        使用向下力为正的符号约定。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1

        """

        x = self.variable  # 变量x，通常为位置变量
        l = self.length  # 梁的长度l
        a = self.ild_variable  # 影响线变量a

        shear_force, _ = self._solve_for_ild_equations(value)  # 解算出移动荷载导致的剪力曲线

        shear_curve1 = value - limit(shear_force, x, distance)  # 剪力曲线1的表达式
        shear_curve2 = (limit(shear_force, x, l) - limit(shear_force, x, distance)) - value  # 剪力曲线2的表达式

        for reaction in reactions:
            shear_curve1 = shear_curve1.subs(reaction,self._ild_reactions[reaction])
            shear_curve2 = shear_curve2.subs(reaction,self._ild_reactions[reaction])

        shear_eq = (shear_curve1 - (shear_curve1 - shear_curve2) * SingularityFunction(a, distance, 0)
                    - value * SingularityFunction(-a, 0, 0) + value * SingularityFunction(a, l, 0))

        self._ild_shear = shear_eq  # 将计算得到的剪力影响线方程存储在_ild_shear属性中
    def solve_for_ild_moment(self, distance, value, *reactions):
        """
        确定受移动荷载作用下指定点力矩的影响线图方程。

        Parameters
        ==========
        distance : Integer
            点距离梁起点的距离，需要确定方程的位置。
        value : Integer
            移动荷载的大小。
        reactions :
            应用在梁上的反力。

        Warning
        =======
        当替换 a = 0 或 a = l（梁的长度）时，此方法可能产生不正确的结果。

        Examples
        ========
        
        假设有一根长度为 12 米的梁。梁下方有两个简支，一个在起点，另一个在距离 8 米处。计算距离 4 米处，受 1kN 大小移动荷载作用下的力矩影响线图方程。

        使用向下力为正的符号约定。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        # 解决影响线图方程以获取力矩
        _, moment = self._solve_for_ild_equations(value)

        # 计算第一段力矩曲线
        moment_curve1 = value*(distance * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                               + SingularityFunction(a, distance, 1)) - limit(moment, x, distance)

        # 计算第二段力矩曲线
        moment_curve2 = (limit(moment, x, l)-limit(moment, x, distance)
                         - value * (l * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                                    + SingularityFunction(a, l, 1)))

        # 替换每个反力到对应的影响线图反力值
        for reaction in reactions:
            moment_curve1 = moment_curve1.subs(reaction, self._ild_reactions[reaction])
            moment_curve2 = moment_curve2.subs(reaction, self._ild_reactions[reaction])

        # 组合得到最终的力矩方程
        moment_eq = moment_curve1 - (moment_curve1 - moment_curve2) * SingularityFunction(a, distance, 0)

        self._ild_moment = moment_eq

    @doctest_depends_on(modules=('numpy',))
    def _is_load_negative(self, load):
        """判断载荷是正还是负，使用展开和必要时的执行。

        Returns
        =======
        True: 如果载荷为负
        False: 如果载荷为正
        None: 如果无法确定

        """
        # 检查是否已经直接给出了载荷的符号信息
        rv = load.is_negative
        # 如果载荷是原子或已经有符号信息，则直接返回结果
        if load.is_Atom or rv is not None:
            return rv
        # 否则，对载荷执行求值、展开操作，并检查是否为负
        return load.doit().expand().is_negative

    def _draw_supports(self, length, l):
        """绘制支撑物，根据给定的长度和l。

        Parameters
        ==========
        length : float
            结构长度的十分之一将用作支撑物高度。
        l : symbol or None
            用于替换位置参数的符号。

        Returns
        =======
        support_markers : list
            包含支撑标记信息的列表。
        support_rectangles : list
            包含支撑矩形信息的列表。

        """
        height = float(length / 10)  # 计算支撑物的高度

        support_markers = []  # 初始化支撑标记列表
        support_rectangles = []  # 初始化支撑矩形列表

        # 遍历已应用的支撑物列表
        for support in self._applied_supports:
            if l:
                pos = support[0].subs(l)  # 如果存在l，替换位置参数
            else:
                pos = support[0]  # 否则直接使用原始位置参数

            # 根据支撑类型进行不同的处理
            if support[1] == "pin":
                # 添加固定在位置pos的圆形支撑标记
                support_markers.append({'args': [pos, [0]], 'marker': 6, 'markersize': 13, 'color': "black"})
            elif support[1] == "roller":
                # 添加在位置pos处的滚动支撑标记
                support_markers.append({'args': [pos, [-height / 2.5]], 'marker': 'o', 'markersize': 11, 'color': "black"})
            elif support[1] == "fixed":
                # 根据位置pos添加固定支撑矩形
                if pos == 0:
                    # 在结构起点添加支撑矩形
                    support_rectangles.append({'xy': (0, -3 * height), 'width': -length / 20, 'height': 6 * height + height,
                                               'fill': False, 'hatch': '/////'})

                else:
                    # 在结构末端添加支撑矩形
                    support_rectangles.append({'xy': (length, -3 * height), 'width': length / 20,
                                               'height': 6 * height + height, 'fill': False, 'hatch': '/////'})
        
        # 返回支撑标记和支撑矩形的列表
        return support_markers, support_rectangles
class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')

    # Apply distributed load 'q' along y-axis at x=0 across the beam span
    >>> b.apply_load(q, 0, 0, dir="y")

    # Apply distributed moment 'm' along z-axis at x=0 across the beam span
    >>> b.apply_moment_load(m, 0, -1, dir="z")

    # Calculate and return shear force along beam axis
    >>> b.shear_force()
    [0, -q*x, 0]

    # Calculate and return bending moment along beam axis
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]

    # Set boundary conditions: zero slope at both ends of the beam
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]

    # Set boundary conditions: zero deflection at both ends of the beam
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]

    # Solve for slope and deflection of the beam
    >>> b.solve_slope_deflection()

    # Factor and simplify the slope equation
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]

    # Calculate deflection components dx, dy, dz
    >>> dx, dy, dz = b.deflection()

    # Collect and simplify deflection component dy
    >>> dy = collect(simplify(dy), x)

    # Check conditions
    >>> dx == dz == 0
    True

    # Check condition for dy
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """
    def __init__(self, length, elastic_modulus, shear_modulus, second_moment,
                 area, variable=Symbol('x')):
        """
        Initializes the class.

        Parameters
        ==========
        length : Sympifyable
            A Symbol or value representing the Beam's length.
        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material.
        shear_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of rigidity.
            It is a measure of rigidity of the Beam material.
        second_moment : Sympifyable or list
            A list of two elements having SymPy expressions representing the
            Beam's Second moment of area. First value represents Second moment
            across y-axis and second across z-axis.
            Single SymPy expression can be passed if both values are the same.
        area : Sympifyable
            A SymPy expression representing the Beam's cross-sectional area
            in a plane perpendicular to the length of the Beam.
        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope, and deflection
            curve. By default, it is set to ``Symbol('x')``.
        """
        # 调用父类的构造函数初始化长度、弹性模量、二阶矩、变量
        super().__init__(length, elastic_modulus, second_moment, variable)
        # 设置剪切模量
        self.shear_modulus = shear_modulus
        # 设置横截面积
        self.area = area
        # 初始化载荷向量为三个零元素的列表
        self._load_vector = [0, 0, 0]
        # 初始化弯矩载荷向量为三个零元素的列表
        self._moment_load_vector = [0, 0, 0]
        # 初始化扭矩字典为空字典
        self._torsion_moment = {}
        # 初始化载荷奇点为三个零元素的列表
        self._load_Singularity = [0, 0, 0]
        # 初始化斜率为三个零元素的列表
        self._slope = [0, 0, 0]
        # 初始化挠度为三个零元素的列表
        self._deflection = [0, 0, 0]
        # 初始化角度挠度为零
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        """Young's Modulus of the Beam. """
        # 返回剪切模量
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        # 将输入的剪切模量表达式符号化，并存储在私有变量中
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        # 返回横截面的二阶矩
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if isinstance(i, list):
            # 如果输入是列表，则将其中的每个元素符号化，并存储在私有变量中
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            # 否则，将输入的表达式符号化，并存储在私有变量中
            self._second_moment = sympify(i)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        # 返回横截面积
        return self._area

    @area.setter
    def area(self, a):
        # 将输入的横截面积表达式符号化，并存储在私有变量中
        self._area = sympify(a)

    @property
    def load_vector(self):
        """
        Returns a three element list representing the load vector.
        """
        # 返回载荷向量
        return self._load_vector

    @property
    def moment_load_vector(self):
        """
        Returns a three element list representing moment loads on Beam.
        """
        # 返回弯矩载荷向量
        return self._moment_load_vector
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has two keywords namely slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value). Further each value is a list corresponding to
        slope or deflection(s) values along three axes at that location.

        Examples
        ========
        There is a beam of length 4 meters. The slope at 0 should be 4 along
        the x-axis and 0 along others. At the other end of beam, deflection
        along all the three axes should be zero.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.bc_slope = [(0, (4, 0, 0))]
        >>> b.bc_deflection = [(4, [0, 0, 0])]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(4, [0, 0, 0])], 'shear_force': [], 'slope': [(0, (4, 0, 0))]}

        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.
        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``
        along y and z axis at ``0``.
        """
        return self._boundary_conditions

    # 返回斜率和挠度的边界条件字典
    def polar_moment(self):
        """
        Returns the polar moment of area of the beam
        about the X axis with respect to the centroid.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A = symbols('l, E, G, I, A')
        >>> b = Beam3D(l, E, G, I, A)
        >>> b.polar_moment()
        2*I
        >>> I1 = [9, 15]
        >>> b = Beam3D(l, E, G, I1, A)
        >>> b.polar_moment()
        24
        """
        # 如果 self.second_moment 不可迭代，则返回 2 * self.second_moment
        if not iterable(self.second_moment):
            return 2*self.second_moment
        # 否则返回 self.second_moment 的总和
        return sum(self.second_moment)
    def apply_load(self, value, start, order, dir="y"):
        """
        This method adds up the force load to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        dir : String
            Axis along which load is applied.
        order : Integer
            The order of the applied load.
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        """
        # 获取变量 x
        x = self.variable
        # 将 value、start、order 转换为 SymPy 可处理的对象
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        # 根据不同的方向 dir 处理力的加载
        if dir == "x":
            # 如果 order 不为 -1，则将 value 加到 _load_vector[0]
            if not order == -1:
                self._load_vector[0] += value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[0]
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)

        elif dir == "y":
            # 如果 order 不为 -1，则将 value 加到 _load_vector[1]
            if not order == -1:
                self._load_vector[1] += value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[1]
            self._load_Singularity[1] += value * SingularityFunction(x, start, order)

        else:
            # 如果 order 不为 -1，则将 value 加到 _load_vector[2]
            if not order == -1:
                self._load_vector[2] += value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[2]
            self._load_Singularity[2] += value * SingularityFunction(x, start, order)


    def apply_moment_load(self, value, start, order, dir="y"):
        """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
        # 获取变量 x
        x = self.variable
        # 将 value、start、order 转换为 SymPy 可处理的对象
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        # 根据不同的方向 dir 处理力矩的加载
        if dir == "x":
            # 如果 order 不为 -2，则将 value 加到 _moment_load_vector[0]
            if not order == -2:
                self._moment_load_vector[0] += value
            else:
                # 如果 start 在 _torsion_moment 的列表中，则将 value 加到 _torsion_moment[start]
                if start in list(self._torsion_moment):
                    self._torsion_moment[start] += value
                else:
                    # 否则将 value 设为 _torsion_moment[start]
                    self._torsion_moment[start] = value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[0]
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)

        elif dir == "y":
            # 如果 order 不为 -2，则将 value 加到 _moment_load_vector[1]
            if not order == -2:
                self._moment_load_vector[1] += value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[0]
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)

        else:
            # 如果 order 不为 -2，则将 value 加到 _moment_load_vector[2]
            if not order == -2:
                self._moment_load_vector[2] += value
            # 将 value*SingularityFunction(x, start, order) 加到 _load_Singularity[0]
            self._load_Singularity[0] += value * SingularityFunction(x, start, order)
    # 定义一个方法用于施加支持条件到结构上
    def apply_support(self, loc, type="fixed"):
        # 如果支持类型是固定或滚动支座
        if type in ("pin", "roller"):
            # 创建一个表示反应力的符号对象
            reaction_load = Symbol('R_'+str(loc))
            # 将反应力加入到反应力字典中
            self._reaction_loads[reaction_load] = reaction_load
            # 将位置和零位移添加到挠曲边界条件列表中
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            # 如果支持类型不是固定或滚动支座，定义反应力和弯矩的符号对象
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            # 将反应力和弯矩加入到反应力字典中
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            # 将位置和零位移添加到挠曲和斜率边界条件列表中
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    # 定义一个方法用于求解反应力
    def solve_for_reaction_loads(self, *reaction):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. It it supported by rollers at
        of its end. A constant distributed load of magnitude 8 N is applied
        from start till its end along y-axis. Another linear load having
        slope equal to 9 is applied along z-axis.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.apply_load(8, start=0, order=0, dir="y")
        >>> b.apply_load(9*x, start=0, order=0, dir="z")
        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
        >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
        >>> b.apply_load(R1, start=0, order=-1, dir="y")
        >>> b.apply_load(R2, start=30, order=-1, dir="y")
        >>> b.apply_load(R3, start=0, order=-1, dir="z")
        >>> b.apply_load(R4, start=30, order=-1, dir="z")
        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
        >>> b.reaction_loads
        {R1: -120, R2: -120, R3: -1350, R4: -2700}
        """
        # 获取变量和长度
        x = self.variable
        l = self.length
        q = self._load_Singularity
        # 计算剪力曲线和弯矩曲线
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        # 对每一个方向进行循环求解反应力
        for i in range(3):
            # 选择出现在剪力曲线或弯矩曲线中的反应力
            react = [r for r in reaction if (shear_curves[i].has(r) or moment_curves[i].has(r))]
            # 如果没有需要求解的反应力，继续下一个方向的求解
            if len(react) == 0:
                continue
            # 计算在该方向上的剪力和弯矩曲线的极限值
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            # 解线性方程组得到反应力的值
            sol = list((linsolve([shear_curve, moment_curve], react).args)[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            # 检查解得的反应力是否在其他方向上已经存在，并且值相同
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError("Ambiguous solution for %s in different directions." % key)
            # 更新反应力字典
            self._reaction_loads.update(sol_dict)
    def shear_force(self):
        """
        返回一个包含三个表达式的列表，表示梁对象沿着三个轴的剪力曲线。
        """
        x = self.variable  # 获取梁对象的变量
        q = self._load_vector  # 获取加载向量
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]  # 返回沿着三个轴的剪力表达式列表

    def axial_force(self):
        """
        返回梁对象内部轴向力的表达式。
        """
        return self.shear_force()[0]  # 返回第一个剪力表达式，即沿着 x 轴的剪力

    def shear_stress(self):
        """
        返回一个包含三个表达式的列表，表示梁对象沿着三个轴的剪应力曲线。
        """
        shear = self.shear_force()
        return [shear[0] / self._area, shear[1] / self._area, shear[2] / self._area]  # 返回沿着三个轴的剪应力表达式列表

    def axial_stress(self):
        """
        返回梁对象内部轴向应力的表达式。
        """
        return self.axial_force() / self._area  # 返回沿着 x 轴的轴向应力表达式

    def bending_moment(self):
        """
        返回一个包含三个表达式的列表，表示梁对象沿着三个轴的弯矩曲线。
        """
        x = self.variable  # 获取梁对象的变量
        m = self._moment_load_vector  # 获取弯矩加载向量
        shear = self.shear_force()  # 获取剪力

        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x),
                integrate(-m[2] - shear[1], x)]  # 返回沿着三个轴的弯矩表达式列表

    def torsional_moment(self):
        """
        返回梁对象内部扭矩的表达式。
        """
        return self.bending_moment()[0]  # 返回沿着 x 轴的扭矩表达式
    def solve_for_torsion(self):
        """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
        # 获取自变量 x
        x = self.variable
        # 初始化力矩总和
        sum_moments = 0
        # 遍历并累加所有扭矩点处的扭矩值
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        # 将扭矩点列表按顺序排列
        list(self._torsion_moment).sort()
        # 获取排列后的扭矩点列表
        pointsList = list(self._torsion_moment)
        # 初始化力矩图形，根据自变量 x 和第一个扭矩点来定义
        torque_diagram = Piecewise((sum_moments, x<=pointsList[0]), (0, x>=pointsList[0]))
        # 对剩余的扭矩点进行处理
        for i in range(len(pointsList))[1:]:
            # 更新力矩总和，减去前一个扭矩点处的扭矩值
            sum_moments -= self._torsion_moment[pointsList[i-1]]
            # 根据当前扭矩点和前后扭矩点的位置关系，更新力矩图形
            torque_diagram += Piecewise((0, x<=pointsList[i-1]), (sum_moments, x<=pointsList[i]), (0, x>=pointsList[i]))
        # 对力矩图形进行积分得到积分力矩图形
        integrated_torque_diagram = integrate(torque_diagram)
        # 计算角位移，通过除以剪切模量和极惯性矩的乘积
        self._angular_deflection =  integrated_torque_diagram/(self.shear_modulus*self.polar_moment())

    def slope(self):
        """
        Returns a three element list representing slope of deflection curve
        along all the three axes.
        """
        # 返回斜率列表
        return self._slope

    def deflection(self):
        """
        Returns a three element list representing deflection curve along all
        the three axes.
        """
        # 返回挠曲曲线列表
        return self._deflection

    def angular_deflection(self):
        """
        Returns a function in x depicting how the angular deflection, due to moments
        in the x-axis on the beam, varies with x.
        """
        # 返回描述角位移随 x 变化的函数
        return self._angular_deflection
    # 定义一个方法来绘制剪力图，根据指定的方向和替换值
    def _plot_shear_force(self, dir, subs=None):
        # 计算剪力
        shear_force = self.shear_force()

        # 根据方向选择对应的索引和颜色
        if dir == 'x':
            dir_num = 0  # X方向对应索引0
            color = 'r'  # 红色表示X方向

        elif dir == 'y':
            dir_num = 1  # Y方向对应索引1
            color = 'g'  # 绿色表示Y方向

        elif dir == 'z':
            dir_num = 2  # Z方向对应索引2
            color = 'b'  # 蓝色表示Z方向

        # 如果替换值subs为None，则设为空字典
        if subs is None:
            subs = {}

        # 检查剪力表达式中的符号，确保除了自变量和替换值之外的所有符号都有值传入
        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        # 如果长度在替换值中，则使用替换值中的长度值，否则使用对象自身的长度
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        # 绘制剪力图
        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show=False, 
                    title='Shear Force along %c direction' % dir,
                    xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{V(%c)}$' % dir, line_color=color)
    def plot_shear_force(self, dir="all", subs=None):
        """
        Returns a plot for Shear force along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear force plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_force()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x for x over (0.0, 20.0)

        """

        dir = dir.lower()  # 将方向参数转换为小写，便于统一处理

        # For shear force along x direction
        if dir == "x":
            # 调用内部函数 _plot_shear_force 处理 x 方向的剪力图
            Px = self._plot_shear_force('x', subs)
            # 显示生成的剪力图
            return Px.show()

        # For shear force along y direction
        elif dir == "y":
            # 调用内部函数 _plot_shear_force 处理 y 方向的剪力图
            Py = self._plot_shear_force('y', subs)
            # 显示生成的剪力图
            return Py.show()

        # For shear force along z direction
        elif dir == "z":
            # 调用内部函数 _plot_shear_force 处理 z 方向的剪力图
            Pz = self._plot_shear_force('z', subs)
            # 显示生成的剪力图
            return Pz.show()

        # For shear force along all direction
        else:
            # 分别调用内部函数 _plot_shear_force 处理 x、y、z 三个方向的剪力图
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            # 返回包含所有剪力图的 PlotGrid 对象
            return PlotGrid(3, 1, Px, Py, Pz)
    # 定义一个方法用于绘制弯矩图，参数包括方向（dir）和替换字典（subs）
    def _plot_bending_moment(self, dir, subs=None):
        # 获取弯矩数据
        bending_moment = self.bending_moment()

        # 根据方向选择对应的坐标轴编号和颜色
        if dir == 'x':
            dir_num = 0  # x 方向对应坐标轴编号 0
            color = 'g'   # 绿色

        elif dir == 'y':
            dir_num = 1  # y 方向对应坐标轴编号 1
            color = 'c'   # 青色

        elif dir == 'z':
            dir_num = 2  # z 方向对应坐标轴编号 2
            color = 'm'   # 洋红色

        # 如果替换字典 subs 为 None，则初始化为空字典
        if subs is None:
            subs = {}

        # 检查在弯矩表达式中的每个符号，确保除了自变量和替换字典中的符号外，其余符号均有传入值
        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        # 如果长度 self.length 在替换字典中，则使用替换字典中的值作为长度
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length  # 否则使用对象的长度属性值

        # 绘制弯矩图，替换自变量并设置坐标轴标签和标题
        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Bending Moment along %c direction' % dir,
                    xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{M(%c)}$' % dir, line_color=color)
        """
        Returns a plot for bending moment along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which bending moment plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_bending_moment()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)

        """

        # Convert direction to lowercase for case-insensitive comparison
        dir = dir.lower()

        # For bending moment along x direction
        if dir == "x":
            # Generate bending moment plot along x direction
            Px = self._plot_bending_moment('x', subs)
            return Px.show()

        # For bending moment along y direction
        elif dir == "y":
            # Generate bending moment plot along y direction
            Py = self._plot_bending_moment('y', subs)
            return Py.show()

        # For bending moment along z direction
        elif dir == "z":
            # Generate bending moment plot along z direction
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()

        # For bending moment along all directions
        else:
            # Generate bending moment plots along all three directions
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)
    # 定义一个方法来绘制函数斜率沿着指定方向的图像
    def _plot_slope(self, dir, subs=None):
        # 计算斜率
        slope = self.slope()

        # 根据指定的方向选择相应的参数
        if dir == 'x':
            dir_num = 0  # x方向对应的索引
            color = 'b'   # 绘图颜色为蓝色

        elif dir == 'y':
            dir_num = 1  # y方向对应的索引
            color = 'm'   # 绘图颜色为洋红色

        elif dir == 'z':
            dir_num = 2  # z方向对应的索引
            color = 'g'   # 绘图颜色为绿色

        # 如果未提供替换参数字典，则初始化为空字典
        if subs is None:
            subs = {}

        # 检查斜率表达式中的符号，并确保除了自变量和提供的替换参数外，其余符号都已赋值
        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        # 如果替换参数中包含了长度参数，则使用提供的值，否则使用默认长度
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        # 返回绘制的斜率图像对象
        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Slope along %c direction' % dir,
                    xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\theta(%c)}$' % dir, line_color=color)
    def plot_slope(self, dir="all", subs=None):
        """
        Returns a plot for Slope along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which Slope plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_slope()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)

        """

        # 将dir转换为小写，以便进行比较
        dir = dir.lower()

        # 如果dir指定了"x"方向
        if dir == "x":
            # 调用私有方法 `_plot_slope` 来绘制x方向的斜率图
            Px = self._plot_slope('x', subs)
            # 显示斜率图
            return Px.show()

        # 如果dir指定了"y"方向
        elif dir == "y":
            # 调用私有方法 `_plot_slope` 来绘制y方向的斜率图
            Py = self._plot_slope('y', subs)
            # 显示斜率图
            return Py.show()

        # 如果dir指定了"z"方向
        elif dir == "z":
            # 调用私有方法 `_plot_slope` 来绘制z方向的斜率图
            Pz = self._plot_slope('z', subs)
            # 显示斜率图
            return Pz.show()

        # 如果dir是"all"或者未知的方向
        else:
            # 分别调用私有方法 `_plot_slope` 绘制x、y、z方向的斜率图
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            # 返回一个包含所有斜率图的 PlotGrid 对象
            return PlotGrid(3, 1, Px, Py, Pz)
    # 定义一个方法用于绘制挠曲图，接受方向参数和替代变量字典作为可选参数
    def _plot_deflection(self, dir, subs=None):

        # 调用对象的挠曲计算方法，获取挠曲数组
        deflection = self.deflection()

        # 根据方向参数设置方向编号和绘图颜色
        if dir == 'x':
            dir_num = 0
            color = 'm'  # 使用紫色作为绘图颜色
        elif dir == 'y':
            dir_num = 1
            color = 'r'  # 使用红色作为绘图颜色
        elif dir == 'z':
            dir_num = 2
            color = 'c'  # 使用青色作为绘图颜色

        # 如果替代变量字典未提供，初始化为空字典
        if subs is None:
            subs = {}

        # 遍历挠曲数组中的符号对象，确保不包含自身的变量和未提供的符号变量
        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                # 如果符号不是自身的变量且未在替代变量字典中找到，则抛出数值错误异常
                raise ValueError('Value of %s was not passed.' % sym)

        # 如果长度在替代变量字典中，将其值赋给局部变量 length，否则使用对象的长度属性
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        # 返回挠曲图的绘制对象，替代变量后的挠曲曲线，绘制范围从 0 到指定长度
        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Deflection along %c direction' % dir,
                    xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\delta(%c)}$' % dir, line_color=color)
    def plot_deflection(self, dir="all", subs=None):
        """
        Returns a plot for Deflection along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which deflection plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_deflection()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)
        """

        # 将dir参数转换为小写以便统一处理
        dir = dir.lower()

        # 如果dir为"x"，绘制沿x方向的挠度图
        if dir == "x":
            Px = self._plot_deflection('x', subs)
            return Px.show()

        # 如果dir为"y"，绘制沿y方向的挠度图
        elif dir == "y":
            Py = self._plot_deflection('y', subs)
            return Py.show()

        # 如果dir为"z"，绘制沿z方向的挠度图
        elif dir == "z":
            Pz = self._plot_deflection('z', subs)
            return Pz.show()

        # 如果dir为"all"或其他未知值，绘制沿所有方向的挠度图
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)
    def plot_loading_results(self, dir='x', subs=None):
        """
        返回 Beam 对象沿指定方向的剪力、弯矩、斜率和挠度的子图。

        Parameters
        ==========

        dir : string (default : "x")
               指定需要绘制的方向。
               如果未指定方向，默认沿 x 轴显示图形。
        subs : dictionary
               Python 字典，包含符号作为键及其对应的值。

        Examples
        ========
        有一根长度为 20 米的梁。两端支承于滚动支座。沿 y 轴施加斜率为 12 的线性载荷。
        沿 z 轴从起点到终点施加幅值为 15 N 的恒定分布载荷。

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> subs = {E:40, G:21, I:100, A:25}
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_loading_results('y',subs)
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[3]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)

        """

        dir = dir.lower()  # 将方向参数转换为小写
        if subs is None:
            subs = {}

        # 调用私有方法，绘制剪力图
        ax1 = self._plot_shear_force(dir, subs)
        # 调用私有方法，绘制弯矩图
        ax2 = self._plot_bending_moment(dir, subs)
        # 调用私有方法，绘制斜率图
        ax3 = self._plot_slope(dir, subs)
        # 调用私有方法，绘制挠度图
        ax4 = self._plot_deflection(dir, subs)

        # 返回包含四个子图的 PlotGrid 对象
        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)
    # 定义一个方法来绘制剪切应力图像，根据指定的方向和可选的替代变量进行绘制

    shear_stress = self.shear_stress()
    # 计算并获取当前对象的剪切应力

    if dir == 'x':
        dir_num = 0
        color = 'r'
        # 如果指定方向为'x'，则设置方向编号为0，颜色为红色

    elif dir == 'y':
        dir_num = 1
        color = 'g'
        # 如果指定方向为'y'，则设置方向编号为1，颜色为绿色

    elif dir == 'z':
        dir_num = 2
        color = 'b'
        # 如果指定方向为'z'，则设置方向编号为2，颜色为蓝色

    if subs is None:
        subs = {}
        # 如果替代变量字典未提供，则初始化为空字典

    for sym in shear_stress[dir_num].atoms(Symbol):
        if sym != self.variable and sym not in subs:
            raise ValueError('Value of %s was not passed.' %sym)
        # 遍历剪切应力中指定方向的符号变量，如果变量不是对象的主变量且不在替代变量字典中，则引发值错误异常

    if self.length in subs:
        length = subs[self.length]
    else:
        length = self.length
        # 如果对象的长度在替代变量字典中，则使用替代变量中的长度值，否则使用对象的默认长度值

    return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show=False, title='Shear stress along %c direction' % dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\tau(%c)$' % dir, line_color=color)
    # 返回绘制的剪切应力图，替换符号变量为替代变量值，并设置图形标题、坐标轴标签和线条颜色
    def plot_shear_stress(self, dir="all", subs=None):
        """
        Returns a plot for Shear Stress along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear stress plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters and area of cross section 2 square
        meters. It is supported by rollers at both of its ends. A linear load having
        slope equal to 12 is applied along y-axis. A constant distributed load
        of magnitude 15 N is applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, 2, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_stress()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)

        """

        dir = dir.lower()  # 将方向参数转换为小写
        # 对于x方向的剪切应力
        if dir == "x":
            Px = self._plot_shear_stress('x', subs)  # 调用内部方法生成x方向剪切应力图
            return Px.show()  # 显示图像
        # 对于y方向的剪切应力
        elif dir == "y":
            Py = self._plot_shear_stress('y', subs)  # 调用内部方法生成y方向剪切应力图
            return Py.show()  # 显示图像
        # 对于z方向的剪切应力
        elif dir == "z":
            Pz = self._plot_shear_stress('z', subs)  # 调用内部方法生成z方向剪切应力图
            return Pz.show()  # 显示图像
        # 对于所有方向的剪切应力
        else:
            Px = self._plot_shear_stress('x', subs)  # 调用内部方法生成x方向剪切应力图
            Py = self._plot_shear_stress('y', subs)  # 调用内部方法生成y方向剪切应力图
            Pz = self._plot_shear_stress('z', subs)  # 调用内部方法生成z方向剪切应力图
            return PlotGrid(3, 1, Px, Py, Pz)  # 返回包含三个图的图形网格对象
    def _max_shear_force(self, dir):
        """
        Helper function for max_shear_force().
        """
        # 将方向参数转换为小写
        dir = dir.lower()

        # 根据方向参数确定对应的数字索引
        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        # 如果在指定方向上没有剪力数据，则返回零值元组
        if not self.shear_force()[dir_num]:
            return (0, 0)

        # 定义一个分段函数来限制变量的取值范围
        load_curve = Piecewise((float("nan"), self.variable <= 0),
                               (self._load_vector[dir_num], self.variable < self.length),
                               (float("nan"), True))

        # 解析分段函数，获取关键点
        points = solve(load_curve.rewrite(Piecewise), self.variable,
                       domain=S.Reals)
        points.append(0)
        points.append(self.length)

        # 获取在指定方向上的剪力曲线
        shear_curve = self.shear_force()[dir_num]

        # 计算关键点处的剪力值，并取绝对值
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))

        # 找到最大剪力值及其对应的点
        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """
        # 计算并返回在 x、y、z 三个方向上的最大剪力及其对应点的列表
        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear
    def _max_bending_moment(self, dir):
        """
        Helper function for max_bending_moment().
        Calculate the maximum bending moment and its corresponding position along a specific direction ('x', 'y', or 'z') in a Beam object.

        Parameters:
        dir : str
            Direction of interest ('x', 'y', or 'z').

        Returns:
        tuple
            Tuple containing the position and value of the maximum bending moment along the specified direction.
        """

        # Convert direction to lowercase for consistency
        dir = dir.lower()

        # Determine the numerical representation of the direction
        if dir == 'x':
            dir_num = 0
        elif dir == 'y':
            dir_num = 1
        elif dir == 'z':
            dir_num = 2

        # If there's no bending moment data for the specified direction, return (0, 0)
        if not self.bending_moment()[dir_num]:
            return (0, 0)

        # Define a piecewise function for the shear curve, restricting within the beam's length
        shear_curve = Piecewise((float("nan"), self.variable <= 0),
                                (self.shear_force()[dir_num], self.variable < self.length),
                                (float("nan"), True))

        # Solve the piecewise function to get critical points
        points = solve(shear_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)
        points.append(self.length)

        # Retrieve the bending moment curve for the specified direction
        bending_moment_curve = self.bending_moment()[dir_num]

        # Calculate bending moments at critical points and take absolute values
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))

        # Find the maximum bending moment and its corresponding position
        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        """
        Returns the points of maximum bending moments and their corresponding values along all directions ('x', 'y', 'z') in a Beam object.

        Requires that solve_for_reaction_loads() has been called before using this function.

        Returns:
        list
            List of tuples, each containing the position and value of the maximum bending moment for each direction.
        """

        max_bmoment = []
        # Calculate maximum bending moments for directions 'x', 'y', and 'z'
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment

    max_bmoment = max_bending_moment
    # 定义用于计算最大挠度的辅助函数，参数为方向（'x'、'y'、'z'）
    def _max_deflection(self, dir):
        """
        Helper function for max_Deflection()
        """

        # 将方向转换为小写以便进行比较
        dir = dir.lower()

        # 根据方向确定对应的数字索引
        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        # 如果指定方向的挠度为零，则返回 (0, 0)
        if not self.deflection()[dir_num]:
            return (0, 0)

        # 定义曲线，Piecewise 表示在不同区间采用不同的表达式
        slope_curve = Piecewise((float("nan"), self.variable <= 0),
                                (self.slope()[dir_num], self.variable < self.length),
                                (float("nan"), True))

        # 解方程 slope_curve.rewrite(Piecewise)，找到满足条件的变量 points
        points = solve(slope_curve.rewrite(Piecewise), self.variable, domain=S.Reals)
        points.append(0)  # 添加起点
        points.append(self._length)  # 添加长度

        # 获取指定方向上的挠度曲线
        deflection_curve = self.deflection()[dir_num]

        # 计算各个点处的挠度值，并取绝对值
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))

        # 找到最大挠度及其位置，返回位置和最大挠度值
        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)
    # 返回梁对象在各个方向上的最大挠度及其对应的挠度值的列表
    # 在调用该函数之前，必须先调用 solve_for_reaction_loads() 和 solve_slope_deflection()

    max_def = []
    # 计算梁在 x 方向上的最大挠度并添加到列表中
    max_def.append(self._max_deflection('x'))
    # 计算梁在 y 方向上的最大挠度并添加到列表中
    max_def.append(self._max_deflection('y'))
    # 计算梁在 z 方向上的最大挠度并添加到列表中
    max_def.append(self._max_deflection('z'))
    # 返回包含各个方向最大挠度及其对应挠度值的列表
    return max_def
```