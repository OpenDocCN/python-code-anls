# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\color_scheme.py`

```
from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift

class ColorGradient:
    # 默认颜色和插值间隔
    colors = [0.4, 0.4, 0.4], [0.9, 0.9, 0.9]
    intervals = 0.0, 1.0

    def __init__(self, *args):
        # 根据参数个数不同初始化颜色和插值间隔
        if len(args) == 2:
            self.colors = list(args)
            self.intervals = [0.0, 1.0]
        elif len(args) > 0:
            if len(args) % 2 != 0:
                raise ValueError("len(args) should be even")
            self.colors = [args[i] for i in range(1, len(args), 2)]
            self.intervals = [args[i] for i in range(0, len(args), 2)]
        assert len(self.colors) == len(self.intervals)

    # 创建当前对象的深拷贝
    def copy(self):
        c = ColorGradient()
        c.colors = [e[::] for e in self.colors]
        c.intervals = self.intervals[::]
        return c

    # 查找给定值所在的插值区间索引
    def _find_interval(self, v):
        m = len(self.intervals)
        i = 0
        while i < m - 1 and self.intervals[i] <= v:
            i += 1
        return i

    # 在指定轴上进行插值
    def _interpolate_axis(self, axis, v):
        i = self._find_interval(v)
        v = rinterpolate(self.intervals[i - 1], self.intervals[i], v)
        return interpolate(self.colors[i - 1][axis], self.colors[i][axis], v)

    # 调用对象时，进行颜色插值计算
    def __call__(self, r, g, b):
        c = self._interpolate_axis
        return c(0, r), c(1, g), c(2, b)

# 默认颜色方案字典，定义在本文件末尾
default_color_schemes = {}

class ColorScheme:
    # 初始化颜色方案对象
    def __init__(self, *args, **kwargs):
        self.args = args
        self.f, self.gradient = None, ColorGradient()

        # 根据参数类型初始化函数或者使用默认颜色方案
        if len(args) == 1 and not isinstance(args[0], Basic) and callable(args[0]):
            self.f = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            if args[0] in default_color_schemes:
                cs = default_color_schemes[args[0]]
                self.f, self.gradient = cs.f, cs.gradient.copy()
            else:
                self.f = lambdify('x,y,z,u,v', args[0])
        else:
            self.f, self.gradient = self._interpret_args(args)
        
        # 测试颜色函数
        self._test_color_function()
        
        # 确保梯度对象是ColorGradient的实例
        if not isinstance(self.gradient, ColorGradient):
            raise ValueError("Color gradient not properly initialized. "
                             "(Not a ColorGradient instance.)")
    # 解析给定的参数列表，返回解析后的函数、梯度对象
    def _interpret_args(self, args):
        # 初始化函数为None，梯度为对象属性self.gradient
        f, gradient = None, self.gradient
        # 将参数args按照原子和列表进行排序
        atoms, lists = self._sort_args(args)
        # 从列表中弹出符号列表s
        s = self._pop_symbol_list(lists)
        # 填充符号列表中的变量
        s = self._fill_in_vars(s)

        # 准备用于 lambdify 失败时的错误消息
        f_str = ', '.join(str(fa) for fa in atoms)  # 将原子列表转换为字符串
        s_str = (str(sa) for sa in s)  # 将符号列表中每个元素转换为字符串生成器
        s_str = ', '.join(sa for sa in s_str if sa.find('unbound') < 0)  # 过滤掉包含'unbound'的字符串
        f_error = ValueError("Could not interpret arguments "
                             "%s as functions of %s." % (f_str, s_str))  # 构造错误消息对象

        # 尝试用 lambdify 解析参数
        if len(atoms) == 1:
            fv = atoms[0]
            try:
                f = lambdify(s, [fv, fv, fv])  # 尝试创建一个函数对象
            except TypeError:
                raise f_error  # 解析失败则抛出错误

        elif len(atoms) == 3:
            fr, fg, fb = atoms
            try:
                f = lambdify(s, [fr, fg, fb])  # 尝试创建一个函数对象
            except TypeError:
                raise f_error  # 解析失败则抛出错误

        else:
            raise ValueError("A ColorScheme must provide 1 or 3 "
                             "functions in x, y, z, u, and/or v.")  # 如果原子数不是1或3，则抛出错误

        # 尝试解析给定的颜色信息
        if len(lists) == 0:
            gargs = []  # 如果列表为空，则gargs为空列表

        elif len(lists) == 1:
            gargs = lists[0]  # 如果列表长度为1，则gargs为该列表

        elif len(lists) == 2:
            try:
                (r1, g1, b1), (r2, g2, b2) = lists  # 尝试从列表中解包两个颜色元组
            except TypeError:
                raise ValueError("If two color arguments are given, "
                                 "they must be given in the format "
                                 "(r1, g1, b1), (r2, g2, b2).")  # 解包失败则抛出错误
            gargs = lists  # 解包成功则gargs为lists

        elif len(lists) == 3:
            try:
                (r1, r2), (g1, g2), (b1, b2) = lists  # 尝试从列表中解包三个颜色元组
            except Exception:
                raise ValueError("If three color arguments are given, "
                                 "they must be given in the format "
                                 "(r1, r2), (g1, g2), (b1, b2). To create "
                                 "a multi-step gradient, use the syntax "
                                 "[0, colorStart, step1, color1, ..., 1, "
                                 "colorEnd].")  # 解包失败则抛出错误
            gargs = [[r1, g1, b1], [r2, g2, b2]]  # 解包成功则gargs为包含两个颜色列表的列表

        else:
            raise ValueError("Don't know what to do with collection "
                             "arguments %s." % (', '.join(str(l) for l in lists)))  # 如果列表长度不是0、1、2或3，则抛出错误

        # 如果gargs非空，则尝试用其初始化ColorGradient对象
        if gargs:
            try:
                gradient = ColorGradient(*gargs)  # 尝试创建ColorGradient对象
            except Exception as ex:
                raise ValueError(("Could not initialize a gradient "
                                  "with arguments %s. Inner "
                                  "exception: %s") % (gargs, str(ex)))  # 初始化失败则抛出错误信息

        return f, gradient  # 返回解析后的函数对象和梯度对象
    # 从给定的列表中弹出符号列表，符号列表中的每个元素都应该是 Symbol 类型
    def _pop_symbol_list(self, lists):
        # 初始化空列表用于存储符号列表
        symbol_lists = []
        # 遍历传入的列表中的每个元素
        for l in lists:
            # 标记是否所有元素都是 Symbol 类型
            mark = True
            # 检查当前列表 l 中的每个元素 s
            for s in l:
                # 如果 s 不为 None 并且不是 Symbol 类型，则标记为 False
                if s is not None and not isinstance(s, Symbol):
                    mark = False
                    break
            # 如果标记为 True，则将该列表从 lists 中移除并加入 symbol_lists 中
            if mark:
                lists.remove(l)
                symbol_lists.append(l)
        # 如果只有一个符号列表，则直接返回该列表
        if len(symbol_lists) == 1:
            return symbol_lists[0]
        # 如果没有符号列表，则返回空列表
        elif len(symbol_lists) == 0:
            return []
        else:
            # 如果存在多个符号列表，则抛出 ValueError 异常
            raise ValueError("Only one list of Symbols "
                             "can be given for a color scheme.")

    # 填充变量参数列表，如果没有显式给出变量，则使用默认值
    def _fill_in_vars(self, args):
        # 默认的变量列表
        defaults = symbols('x,y,z,u,v')
        # 错误：找不到要绘制的内容时抛出的异常
        v_error = ValueError("Could not find what to plot.")
        # 如果没有参数传入，则返回默认的变量列表
        if len(args) == 0:
            return defaults
        # 如果 args 不是 tuple 或 list 类型，则抛出 v_error 异常
        if not isinstance(args, (tuple, list)):
            raise v_error
        # 如果参数列表为空，则返回默认的变量列表
        if len(args) == 0:
            return defaults
        # 检查每个参数 s 是否是 Symbol 类型，若不是则抛出 v_error 异常
        for s in args:
            if s is not None and not isinstance(s, Symbol):
                raise v_error
        # 当显式给出变量时，任何未给出的变量都标记为 'unbound'，以免在表达式中意外使用
        vars = [Symbol('unbound%i' % (i)) for i in range(1, 6)]
        # 解释给定的变量
        # 当只有一个变量时，将其解释为 t
        if len(args) == 1:
            vars[3] = args[0]
        # 当有两个变量时，将它们解释为 u,v
        elif len(args) == 2:
            if args[0] is not None:
                vars[3] = args[0]
            if args[1] is not None:
                vars[4] = args[1]
        # 当有三个或更多的变量时
        elif len(args) >= 3:
            # 允许 x,y,z 中的一些变量未被赋值
            if args[0] is not None:
                vars[0] = args[0]
            if args[1] is not None:
                vars[1] = args[1]
            if args[2] is not None:
                vars[2] = args[2]
            # 若参数列表长度大于等于 4，则将第四个参数解释为 t
            if len(args) >= 4:
                vars[3] = args[3]
                # 若参数列表长度大于等于 5，则将第五个参数解释为 u,v
                if len(args) >= 5:
                    vars[4] = args[4]
        # 返回变量列表
        return vars

    # 对参数进行排序，分离出列表和原子元素
    def _sort_args(self, args):
        # 使用 sift 函数将 args 中的列表分离出来
        lists, atoms = sift(args,
            lambda a: isinstance(a, (tuple, list)), binary=True)
        # 返回分离出的原子元素和列表
        return atoms, lists

    # 测试颜色函数是否可调用，并满足预期的返回结果
    def _test_color_function(self):
        # 如果 self.f 不可调用，则抛出 ValueError 异常
        if not callable(self.f):
            raise ValueError("Color function is not callable.")
        try:
            # 调用 self.f 测试其返回结果
            result = self.f(0, 0, 0, 0, 0)
            # 如果返回结果长度不为 3，则抛出 ValueError 异常
            if len(result) != 3:
                raise ValueError("length should be equal to 3")
        except TypeError:
            # 如果 self.f 不接受 x,y,z,u,v 作为参数，则抛出 ValueError 异常
            raise ValueError("Color function needs to accept x,y,z,u,v, "
                             "as arguments even if it doesn't use all of them.")
        except AssertionError:
            # 如果 self.f 不返回 3 元组 (r,g,b)，则抛出 ValueError 异常
            raise ValueError("Color function needs to return 3-tuple r,g,b.")
        except Exception:
            pass  # 当颜色函数在给定参数下可能无效时，不抛出异常
    # 定义一个特殊方法 __call__()，用于将对象作为函数调用
    def __call__(self, x, y, z, u, v):
        try:
            # 调用对象的内部函数 f() 处理输入参数，并返回结果
            return self.f(x, y, z, u, v)
        except Exception:
            # 如果出现异常，返回 None
            return None

    # 将这种颜色方案应用到一组顶点上，顶点随单独变量 u 变化
    def apply_to_curve(self, verts, u_set, set_len=None, inc_pos=None):
        """
        Apply this color scheme to a
        set of vertices over a single
        independent variable u.
        """
        # 创建边界值字典，用于存储颜色分量的最小和最大值
        bounds = create_bounds()
        # 存储计算后的顶点颜色值
        cverts = []
        # 如果 set_len 参数是可调用的，则设置其长度为 u_set 的两倍
        if callable(set_len):
            set_len(len(u_set)*2)
        # 对每个顶点进行遍历
        for _u in range(len(u_set)):
            if verts[_u] is None:
                # 如果顶点为 None，则将其颜色值也设置为 None
                cverts.append(None)
            else:
                # 否则获取顶点的坐标和当前的 u 值
                x, y, z = verts[_u]
                u, v = u_set[_u], None
                # 调用当前对象作为函数，计算顶点的颜色值 c
                c = self(x, y, z, u, v)
                if c is not None:
                    # 如果颜色值不为 None，则将其转换为列表，并更新边界值
                    c = list(c)
                    update_bounds(bounds, c)
                # 将计算后的颜色值添加到 cverts 列表中
                cverts.append(c)
            # 如果 inc_pos 参数是可调用的，则调用它一次
            if callable(inc_pos):
                inc_pos()
        # 对计算后的颜色值进行缩放和应用渐变处理
        for _u in range(len(u_set)):
            if cverts[_u] is not None:
                for _c in range(3):
                    # 将颜色值从 [f_min, f_max] 缩放到 [0,1] 范围内
                    cverts[_u][_c] = rinterpolate(bounds[_c][0], bounds[_c][1],
                                                  cverts[_u][_c])
                # 应用渐变到颜色值
                cverts[_u] = self.gradient(*cverts[_u])
            # 如果 inc_pos 参数是可调用的，则调用它一次
            if callable(inc_pos):
                inc_pos()
        # 返回最终处理后的顶点颜色值列表
        return cverts
    def apply_to_surface(self, verts, u_set, v_set, set_len=None, inc_pos=None):
        """
        Apply this color scheme to a
        set of vertices over two
        independent variables u and v.
        """
        # 创建一个边界列表，用于存储颜色分量的最小值和最大值
        bounds = create_bounds()
        
        # 存储计算后的顶点颜色信息
        cverts = []
        
        # 如果set_len参数是可调用的函数，则调用它来设置总长度
        if callable(set_len):
            set_len(len(u_set) * len(v_set) * 2)
        
        # 遍历u和v集合，计算顶点的颜色，并更新边界值
        for _u in range(len(u_set)):
            column = []
            for _v in range(len(v_set)):
                # 如果顶点为空，则将空值添加到列中
                if verts[_u][_v] is None:
                    column.append(None)
                else:
                    # 提取顶点坐标和u、v值
                    x, y, z = verts[_u][_v]
                    u, v = u_set[_u], v_set[_v]
                    # 计算顶点的颜色值，并将其转换为列表形式
                    c = self(x, y, z, u, v)
                    if c is not None:
                        c = list(c)
                        # 更新边界值
                        update_bounds(bounds, c)
                    # 将颜色值添加到列中
                    column.append(c)
                
                # 如果inc_pos参数是可调用的函数，则调用它来指示进度增加
                if callable(inc_pos):
                    inc_pos()
            
            # 将列添加到顶点颜色列表中
            cverts.append(column)
        
        # 对颜色值进行缩放并应用渐变
        for _u in range(len(u_set)):
            for _v in range(len(v_set)):
                if cverts[_u][_v] is not None:
                    # 将每个颜色分量从[f_min, f_max]缩放到[0,1]
                    for _c in range(3):
                        cverts[_u][_v][_c] = rinterpolate(bounds[_c][0],
                                             bounds[_c][1], cverts[_u][_v][_c])
                    # 应用颜色渐变
                    cverts[_u][_v] = self.gradient(*cverts[_u][_v])
                
                # 如果inc_pos参数是可调用的函数，则调用它来指示进度增加
                if callable(inc_pos):
                    inc_pos()
        
        # 返回应用了颜色方案后的顶点颜色列表
        return cverts

    def str_base(self):
        # 返回基本参数的字符串表示，用逗号分隔
        return ", ".join(str(a) for a in self.args)

    def __repr__(self):
        # 返回对象的字符串表示，只包含基本参数的字符串
        return "%s" % (self.str_base())
# 定义符号变量 x, y, z, t, u, v，这些可能用于数学符号计算
x, y, z, t, u, v = symbols('x,y,z,t,u,v')

# 将新的颜色方案 'rainbow' 添加到默认颜色方案字典中，使用 ColorScheme 类创建
default_color_schemes['rainbow'] = ColorScheme(z, y, x)

# 将新的颜色方案 'zfade' 添加到默认颜色方案字典中，使用 ColorScheme 类创建
default_color_schemes['zfade'] = ColorScheme(z, (0.4, 0.4, 0.97),
                                             (0.97, 0.4, 0.4), (None, None, z))

# 将新的颜色方案 'zfade3' 添加到默认颜色方案字典中，使用 ColorScheme 类创建
default_color_schemes['zfade3'] = ColorScheme(z, (None, None, z),
                                              [0.00, (0.2, 0.2, 1.0),
                                               0.35, (0.2, 0.8, 0.4),
                                               0.50, (0.3, 0.9, 0.3),
                                               0.65, (0.4, 0.8, 0.2),
                                               1.00, (1.0, 0.2, 0.2)])

# 将新的颜色方案 'zfade4' 添加到默认颜色方案字典中，使用 ColorScheme 类创建
default_color_schemes['zfade4'] = ColorScheme(z, (None, None, z),
                                              [0.0, (0.3, 0.3, 1.0),
                                               0.30, (0.3, 1.0, 0.3),
                                               0.55, (0.95, 1.0, 0.2),
                                               0.65, (1.0, 0.95, 0.2),
                                               0.85, (1.0, 0.7, 0.2),
                                               1.0, (1.0, 0.3, 0.2)])
```