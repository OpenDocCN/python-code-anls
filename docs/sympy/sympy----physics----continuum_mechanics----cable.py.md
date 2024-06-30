# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\cable.py`

```
"""
This module can be used to solve problems related
to 2D Cables.
"""

from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 SymPy 表达式
from sympy.core.symbol import Symbol, symbols  # 导入 Symbol 和 symbols 类，用于定义符号变量
from sympy import sin, cos, pi, atan, diff, Piecewise, solve, rad  # 导入数学函数和符号常量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.solvers.solveset import linsolve  # 导入线性方程组求解函数
from sympy.matrices import Matrix  # 导入矩阵处理模块
from sympy.plotting import plot  # 导入绘图函数

class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable  # 导入 Cable 类
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))  # 创建 Cable 对象 c，指定支持点坐标
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))  # 在点 P 处施加向下的负载
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))  # 在点 Q 处施加向下的负载
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}  # 显示施加的点负载
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}  # 显示施加点负载的位置
    """
    def __init__(self, support_1, support_2):
        """
        Initializes the class with given supports.

        Parameters
        ==========

        support_1 and support_2 are tuples of the form
        (label, x, y), where

        label : String or symbol
            The label of the support

        x : Sympifyable
            The x coordinate of the position of the support

        y : Sympifyable
            The y coordinate of the position of the support
        """
        # Initialize empty lists and dictionaries for various properties
        self._left_support = []
        self._right_support = []
        self._supports = {}
        self._support_labels = []
        self._loads = {"distributed": {}, "point_load": {}}
        self._loads_position = {}
        self._length = 0
        self._reaction_loads = {}
        self._tension = {}
        self._lowest_x_global = sympify(0)
        self._lowest_y_global = sympify(0)
        self._cable_eqn = None
        self._tension_func = None
        
        # Check if supports have the same label or are at the same location
        if support_1[0] == support_2[0]:
            raise ValueError("Supports can not have the same label")

        elif support_1[1] == support_2[1]:
            raise ValueError("Supports can not be at the same location")

        # Convert x, y coordinates to sympy objects and store supports
        x1 = sympify(support_1[1])
        y1 = sympify(support_1[2])
        self._supports[support_1[0]] = [x1, y1]

        x2 = sympify(support_2[1])
        y2 = sympify(support_2[2])
        self._supports[support_2[0]] = [x2, y2]

        # Determine and assign left and right supports based on x coordinates
        if support_1[1] < support_2[1]:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x2)
            self._right_support.append(y2)
            self._support_labels.append(support_1[0])
            self._support_labels.append(support_2[0])

        else:
            self._left_support.append(x2)
            self._left_support.append(y2)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.append(support_2[0])
            self._support_labels.append(support_1[0])

        # Initialize reaction loads for each support label
        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    @property
    def supports(self):
        """
        Returns the supports of the cable along with their
        positions.
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def loads(self):
        """
        Returns the magnitude and direction of the loads
        acting on the cable.
        """
        return self._loads

    @property
    def loads_position(self):
        """
        Returns the position of the point loads acting on the
        cable.
        """
        return self._loads_position
    @property
    def length(self):
        """
        Returns the length of the cable.
        """
        return self._length



    @property
    def reaction_loads(self):
        """
        Returns the reaction forces at the supports, which are
        initialized to 0.
        """
        return self._reaction_loads



    @property
    def tension(self):
        """
        Returns the tension developed in the cable due to the loads
        applied.
        """
        return self._tension



    def tension_at(self, x):
        """
        Returns the tension at a given value of x developed due to
        distributed load.

        Raises:
        ValueError: If 'distributed' load is not added or solve method not called.
                   If x is outside the range defined by the supports.
        """
        if 'distributed' not in self._tension.keys():
            raise ValueError("No distributed load added or solve method not called")

        if x > self._right_support[0] or x < self._left_support[0]:
            raise ValueError("The value of x should be between the two supports")

        A = self._tension['distributed']
        X = Symbol('X')

        return A.subs({X:(x-self._lowest_x_global)})



    def apply_length(self, length):
        """
        This method specifies the length of the cable

        Parameters
        ==========

        length : Sympifyable
            The length of the cable

        Raises:
        ValueError: If length is less than the distance between the supports.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_length(20)
        >>> c.length
        20
        """
        dist = ((self._left_support[0] - self._right_support[0])**2
                - (self._left_support[1] - self._right_support[1])**2)**(1/2)

        if length < dist:
            raise ValueError("length should not be less than the distance between the supports")

        self._length = length
    def change_support(self, label, new_support):
        """
        This method changes the mentioned support with a new support.

        Parameters
        ==========
        label: String or symbol
            The label of the support to be changed

        new_support: Tuple of the form (new_label, x, y)
            new_label: String or symbol
                The label of the new support

            x: Sympifyable
                The x-coordinate of the position of the new support.

            y: Sympifyable
                The y-coordinate of the position of the new support.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.supports
        {'A': [0, 10], 'B': [10, 10]}
        >>> c.change_support('B', ('C', 5, 6))
        >>> c.supports
        {'A': [0, 10], 'C': [5, 6]}
        """
        # 检查是否存在指定标签的支撑
        if label not in self._supports:
            raise ValueError("No support exists with the given label")

        # 找到待修改支撑在支撑标签列表中的索引
        i = self._support_labels.index(label)
        # 计算另一个支撑标签
        rem_label = self._support_labels[(i+1)%2]
        # 获取另一个支撑的坐标
        x1 = self._supports[rem_label][0]
        y1 = self._supports[rem_label][1]

        # 将新支撑的坐标转换为符号表达式
        x = sympify(new_support[1])
        y = sympify(new_support[2])

        # 检查新支撑是否会使任何负载超出范围
        for l in self._loads_position:
            if l[0] >= max(x, x1) or l[0] <= min(x, x1):
                raise ValueError("The change in support will throw an existing load out of range")

        # 移除旧支撑相关数据
        self._supports.pop(label)
        self._left_support.clear()
        self._right_support.clear()
        self._reaction_loads.clear()
        self._support_labels.remove(label)

        # 添加新支撑数据
        self._supports[new_support[0]] = [x, y]

        # 更新左右支撑列表和支撑标签列表
        if x1 < x:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x)
            self._right_support.append(y)
            self._support_labels.append(new_support[0])
        else:
            self._left_support.append(x)
            self._left_support.append(y)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.insert(0, new_support[0])

        # 为每个支撑标签生成对应的反力载荷
        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0
    def remove_loads(self, *args):
        """
        This methods removes the specified loads.

        Parameters
        ==========
        args: tuple
            Multiple label(s) of loads to be removed.

        Examples
        ========
        
        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.remove_loads('Z')
        >>> c.loads
        {'distributed': {}, 'point_load': {}}
        """
        # Iterate over each label in args
        for i in args:
            # Check if there are no loads in _loads_position dictionary
            if len(self._loads_position) == 0:
                # If label i is not in distributed loads, raise ValueError
                if i not in self._loads['distributed']:
                    raise ValueError("Error removing load " + i + ": no such load exists")
                else:
                    # Remove the load from distributed loads
                    self._loads['distributed'].pop(i)
            else:
                # If label i is not in point loads, raise ValueError
                if i not in self._loads['point_load']:
                    raise ValueError("Error removing load " + i + ": no such load exists")
                else:
                    # Remove the load from point loads and its position from _loads_position
                    self._loads['point_load'].pop(i)
                    self._loads_position.pop(i)
    def draw(self):
        """
        This method is used to obtain a plot for the specified cable with its supports,
        shape and loads.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.draw()
        >>> p  # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: Piecewise((10 - 1.37*x, x <= 2), (8.52 - 0.63*x, x <= 4), (2*x/3 + 10/3, x <= 10)) for x over (0.0, 10.0)
        ...
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 39.9955291375291*(0.0170706725844998*x - 1)**2 + 0.00447086247086247 for x over (0.0, 100.0)
        [1]: cartesian line: -7.49552913752915 for x over (0.0, 100.0)
        ...
        >>> p.show()
        """
        # Define the symbol for x
        x = Symbol("x")
        # Initialize an empty list to store annotations
        annotations = []
        # Draw support rectangles and store them
        support_rectangles = self._draw_supports()

        # Determine the minimum y-coordinate for plotting boundaries
        xy_min = min(self._left_support[0], self._lowest_y_global)
        # Determine the maximum y-coordinate for plotting boundaries
        xy_max = max(self._right_support[0], max(self._right_support[1], self._left_support[1]))
        # Calculate the maximum difference in coordinates
        max_diff = xy_max - xy_min

        # Check if point loads are applied
        if len(self._loads_position) != 0:
            # Draw the cable shape considering point loads
            self._cable_eqn = self._draw_cable(-1)
            # Add annotations for point loads
            annotations += self._draw_loads(-1)

        # Check if distributed loads are applied
        elif len(self._loads['distributed']) != 0:
            # Draw the cable shape considering uniformly distributed loads
            self._cable_eqn = self._draw_cable(0)
            # Add annotations for uniformly distributed loads
            annotations += self._draw_loads(0)

        # If no cable equation is determined, raise an error
        if not self._cable_eqn:
            raise ValueError("solve method not called and/or values provided for loads and supports not adequate")

        # Plot the cable shape with specified configurations
        cab_plot = plot(*self._cable_eqn, (x, self._left_support[0], self._right_support[0]),
                        xlim=(xy_min - 0.5 * max_diff, xy_max + 0.5 * max_diff),
                        ylim=(xy_min - 0.5 * max_diff, xy_max + 0.5 * max_diff),
                        rectangles=support_rectangles, show=False, annotations=annotations, axis=False)

        # Return the plot object
        return cab_plot
    # 定义一个方法用于绘制支撑的矩形
    def _draw_supports(self):
        # 创建一个空列表，用于存放支撑矩形的信息
        member_rectangles = []

        # 计算支撑的最小和最大坐标
        xy_min = min(self._left_support[0], self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1], self._left_support[1]))

        # 计算支撑的最大距离差
        max_diff = xy_max - xy_min

        # 计算支撑矩形的宽度
        supp_width = 0.075 * max_diff

        # 添加左侧支撑矩形的信息到列表中
        member_rectangles.append(
            {
                'xy': (self._left_support[0] - supp_width, self._left_support[1]),
                'width': supp_width,
                'height': supp_width,
                'color': 'brown',
                'fill': False
            }
        )

        # 添加右侧支撑矩形的信息到列表中
        member_rectangles.append(
            {
                'xy': (self._right_support[0], self._right_support[1]),
                'width': supp_width,
                'height': supp_width,
                'color': 'brown',
                'fill': False
            }
        )

        # 返回存放支撑矩形信息的列表
        return member_rectangles

    # 定义一个方法用于绘制缆绳
    def _draw_cable(self, order):
        # 计算支撑的最小和最大坐标
        xy_min = min(self._left_support[0], self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1], self._left_support[1]))

        # 计算支撑的最大距离差
        max_diff = xy_max - xy_min

        # 根据 order 参数判断是绘制什么类型的缆绳
        if order == -1:
            # 如果 order 为 -1，则生成线性函数表示的负荷分布
            x, y = symbols('x y')
            line_func = []

            # 根据负荷位置进行排序
            sorted_position = sorted(self._loads_position.items(), key=lambda item: item[1][0])

            # 生成每个负荷位置处的线性函数
            for i in range(len(sorted_position)):
                if i == 0:
                    y = ((sorted_position[i][1][1] - self._left_support[1]) * (x - self._left_support[0])) / (
                                sorted_position[i][1][0] - self._left_support[0]) + self._left_support[1]
                else:
                    y = ((sorted_position[i][1][1] - sorted_position[i - 1][1][1]) * (
                                x - sorted_position[i - 1][1][0])) / (
                                    sorted_position[i][1][0] - sorted_position[i - 1][1][0]) + sorted_position[i - 1][
                                1][1]
                line_func.append((y, x <= sorted_position[i][1][0]))

            # 添加最后一个支撑右侧的线性函数
            y = ((sorted_position[len(sorted_position) - 1][1][1] - self._right_support[1]) * (
                        x - self._right_support[0])) / (sorted_position[i][1][0] - self._right_support[0]) + self._right_support[1]
            line_func.append((y, x <= self._right_support[0]))

            # 返回线性函数列表的 Piecewise 对象
            return [Piecewise(*line_func)]

        elif order == 0:
            # 如果 order 为 0，则生成悬链线的方程和受力点高度
            x0 = self._lowest_x_global
            diff_force_height = max_diff * 0.075

            # 定义符号变量和抛物线方程
            a, c, x, y = symbols('a c x y')
            parabola_eqn = a * (x - x0) ** 2 + c - y

            # 定义支撑点的坐标
            points = [(self._left_support[0], self._left_support[1]), (self._right_support[0], self._right_support[1])]
            equations = []

            # 计算支撑点的抛物线方程
            for px, py in points:
                equations.append(parabola_eqn.subs({x: px, y: py}))

            # 解方程得到抛物线方程
            solution = solve(equations, (a, c))
            parabola_eqn = solution[a] * (x - x0) ** 2 + solution[c]

            # 返回抛物线方程和受力点高度
            return [parabola_eqn, self._lowest_y_global - diff_force_height]
    # 定义一个方法 `_draw_loads`，接受参数 `order`，用于绘制加载箭头
    def _draw_loads(self, order):
        # 计算最小的 x、y 值，作为箭头绘制的起点
        xy_min = min(self._left_support[0], self._lowest_y_global)
        # 计算最大的 x、y 值，作为箭头绘制的终点
        xy_max = max(self._right_support[0], max(self._right_support[1], self._left_support[1]))
        # 计算 x 轴方向的距离差
        max_diff = xy_max - xy_min
        
        # 如果 order 为 -1，则绘制点载荷箭头
        if (order == -1):
            # 计算箭头长度为 x 轴距离差的 10%
            arrow_length = max_diff * 0.1
            # 初始化箭头列表
            force_arrows = []
            # 遍历点载荷的键
            for key in self._loads['point_load']:
                # 添加箭头字典到列表，包括箭头位置和属性
                force_arrows.append(
                    {
                        'text': '',
                        'xy': (self._loads_position[key][0] + arrow_length * cos(rad(self._loads['point_load'][key][1])), \
                               self._loads_position[key][1] + arrow_length * sin(rad(self._loads['point_load'][key][1]))),
                        'xytext': (self._loads_position[key][0], self._loads_position[key][1]),
                        'arrowprops': {'width': 1, 'headlength': 3, 'headwidth': 3, 'facecolor': 'black'}
                    }
                )
                # 获取载荷大小
                mag = self._loads['point_load'][key][0]
                # 添加显示载荷大小的箭头字典到列表
                force_arrows.append(
                    {
                        'text': f'{mag}N',
                        'xy': (self._loads_position[key][0] + arrow_length * 1.6 * cos(rad(self._loads['point_load'][key][1])), \
                               self._loads_position[key][1] + arrow_length * 1.6 * sin(rad(self._loads['point_load'][key][1])))
                    }
                )
            # 返回绘制的箭头列表
            return force_arrows
        
        # 如果 order 为 0，则绘制分布载荷箭头
        elif (order == 0):
            # 定义符号 x
            x = symbols('x')
            # 初始化箭头列表
            force_arrows = []
            # 计算 x 值的取样点
            x_val = [self._left_support[0] + ((self._right_support[0] - self._left_support[0]) / 10) * i for i in range(1, 10)]
            # 遍历 x 值的取样点
            for i in x_val:
                # 添加箭头字典到列表，包括箭头位置和属性
                force_arrows.append(
                    {
                        'text': '',
                        'xytext': (
                            i,
                            self._cable_eqn[0].subs(x, i)
                        ),
                        'xy': (
                            i,
                            self._cable_eqn[1].subs(x, i)
                        ),
                        'arrowprops': {'width': 1, 'headlength': 3.5, 'headwidth': 3.5, 'facecolor': 'black'}
                    }
                )
            # 计算分布载荷的总大小
            mag = 0
            for key in self._loads['distributed']:
                mag += self._loads['distributed'][key]

            # 添加显示分布载荷大小的箭头字典到列表
            force_arrows.append(
                {
                    'text': f'{mag} N/m',
                    'xy': ((self._left_support[0] + self._right_support[0]) / 2, self._lowest_y_global - max_diff * 0.15)
                }
            )
            # 返回绘制的箭头列表
            return force_arrows
    def plot_tension(self):
        """
        Returns the diagram/plot of the tension generated in the cable at various points.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: Piecewise((8.91403453669861, x <= 2), (4.79150773600774, x <= 4), (19*sqrt(13)/10, x <= 10)) for x over (0.0, 10.0)
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: 36465.0*sqrt(0.00054335718671383*X**2 + 1) for X over (0.0, 100.0)
        >>> p.show()

        """
        # 检查是否有载荷位置的信息
        if len(self._loads_position) != 0:
            # 若存在载荷位置，则使用单点载荷张力函数绘制张力图
            x = symbols('x')
            tension_plot = plot(self._tension_func, (x, self._left_support[0], self._right_support[0]), show=False)
        else:
            # 若不存在载荷位置，则使用分布载荷张力函数绘制张力图
            X = symbols('X')
            tension_plot = plot(self._tension['distributed'], (X, self._left_support[0], self._right_support[0]), show=False)
        
        # 返回绘制的张力图对象
        return tension_plot
```