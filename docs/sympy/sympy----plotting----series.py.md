# `D:\src\scipysrc\sympy\sympy\plotting\series.py`

```
# 导入 Callable 抽象基类，用于声明可调用对象
# 从 sympy.calculus.util 模块导入 continuous_domain 函数
# 从 sympy.concrete 模块导入 Sum 和 Product 类
# 从 sympy.core.containers 模块导入 Tuple 类
# 从 sympy.core.expr 模块导入 Expr 类
# 从 sympy.core.function 模块导入 arity 函数
# 从 sympy.core.sorting 模块导入 default_sort_key 函数
# 从 sympy.core.symbol 模块导入 Symbol 类
# 从 sympy.functions 模块导入 atan2, zeta, frac, ceiling, floor, im 函数
# 从 sympy.core.relational 模块导入 Equality, GreaterThan, LessThan, Relational, Ne 类
# 从 sympy.core.sympify 模块导入 sympify 函数
# 从 sympy.external 模块导入 import_module 函数
# 从 sympy.logic.boolalg 模块导入 BooleanFunction 类
# 从 sympy.plotting.utils 模块导入 _get_free_symbols, extract_solution 函数
# 从 sympy.printing.latex 模块导入 latex 函数
# 从 sympy.printing.pycode 模块导入 PythonCodePrinter 类
# 从 sympy.printing.precedence 模块导入 precedence 函数
# 从 sympy.sets.sets 模块导入 Set, Interval, Union 类
# 从 sympy.simplify.simplify 模块导入 nsimplify 函数
# 从 sympy.utilities.exceptions 模块导入 sympy_deprecation_warning 函数
# 从 sympy.utilities.lambdify 模块导入 lambdify 函数
# 从当前包的 intervalmath 模块导入 interval 函数
# 导入 warnings 模块

class IntervalMathPrinter(PythonCodePrinter):
    """在 `plot_implicit` 中 `adaptive=True` 情况下使用的打印机，
    需要进行以下修改以支持区间算术模块。
    """
    def _print_And(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 按照默认排序键对表达式参数进行排序，并用 "&" 连接成字符串
        return " & ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        # 获取表达式的优先级
        PREC = precedence(expr)
        # 按照默认排序键对表达式参数进行排序，并用 "|" 连接成字符串
        return " | ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))


def _uniform_eval(f1, f2, *args, modules=None,
    force_real_eval=False, has_sum=False):
    """
    注意：这是一个实验性函数，可能会发生变化。
    请不要在您的代码中使用它。
    """
    # 导入 numpy 模块
    np = import_module('numpy')

    def wrapper_func(func, *args):
        # 尝试调用 func(*args)，返回其复数形式结果
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            # 处理除零错误和溢出错误，返回复数 NaN
            return complex(np.nan, np.nan)

    # 注意：np.vectorize 比 numpy 的向量化操作慢得多。
    # 然而，这些模块必须能够使用 mpmath 或 sympy 评估函数。
    # 将 wrapper_func 向量化，使用复数类型作为输出类型
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])
    # 定义一个函数 _eval_with_sympy，用于评估给定的数值函数
    def _eval_with_sympy(err=None):
        # 如果 f2 为 None，则抛出运行时异常，指示无法评估提供的数值函数
        if f2 is None:
            msg = "Impossible to evaluate the provided numerical function"
            # 如果没有特定的错误，添加一条消息到异常信息中
            if err is None:
                msg += "."
            else:
                # 如果存在错误，将错误类型和信息添加到异常信息中
                msg += "because the following exception was raised:\n"
                "{}: {}".format(type(err).__name__, err)
            raise RuntimeError(msg)
        
        # 如果存在错误，发出警告并尝试使用 SymPy 进行评估
        if err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
        
        # 使用封装函数 wrapper_func 对 f2 进行评估，并返回结果
        return wrapper_func(f2, *args)

    # 如果 modules 是 "sympy"，则使用 SymPy 进行评估，并返回结果
    if modules == "sympy":
        return _eval_with_sympy()

    # 否则，尝试使用封装函数 wrapper_func 对 f1 进行评估，捕获任何异常
    try:
        return wrapper_func(f1, *args)
    # 如果捕获到异常 err，则使用 _eval_with_sympy 函数进行评估，并返回结果
    except Exception as err:
        return _eval_with_sympy(err)
# 定义一个函数 `_adaptive_eval`，用于使用自适应算法评估函数 f(x)，并进行结果的后处理。
# 如果使用 SymPy 评估了一个符号表达式，可能会返回另一个包含加法等的符号表达式。
# 强制将结果评估为一个浮点数。
def _adaptive_eval(f, x):
    np = import_module('numpy')  # 导入 numpy 模块

    y = f(x)  # 调用函数 f 计算结果 y
    # 如果 y 是 SymPy 的表达式类型 Expr，并且不是一个数值，则对其进行数值评估
    if isinstance(y, Expr) and (not y.is_Number):
        y = y.evalf()
    y = complex(y)  # 将 y 转换为复数形式
    # 如果 y 的虚部大于 1e-08，则返回 NaN
    if y.imag > 1e-08:
        return np.nan
    return y.real  # 返回 y 的实部作为结果


# 定义一个函数 `_get_wrapper_for_expr`，根据返回类型 ret 返回相应的字符串格式化模板
def _get_wrapper_for_expr(ret):
    wrapper = "%s"  # 默认的字符串格式化模板
    # 根据返回类型 ret 选择不同的字符串格式化模板
    if ret == "real":
        wrapper = "re(%s)"
    elif ret == "imag":
        wrapper = "im(%s)"
    elif ret == "abs":
        wrapper = "abs(%s)"
    elif ret == "arg":
        wrapper = "arg(%s)"
    return wrapper  # 返回最终的字符串格式化模板


class BaseSeries:
    """数据对象的基类，包含用于绘图的数据。

    Notes
    =====

    后端应检查是否支持给定的数据系列类型。
    (例如，TextBackend 仅支持 LineOver1DRangeSeries)。
    后端的责任是知道如何使用给定的数据系列类。

    根据它们所呈现的 API (仅基于约定)，某些数据系列类被分组在一起
    (使用类属性如 is_2Dline)。
    后端不必使用该 API (例如，LineOver1DRangeSeries 属于 is_2Dline
    组，并且呈现 get_points 方法，但 TextBackend 不使用 get_points 方法)。

    BaseSeries
    """

    is_2Dline = False  # 标志，指示数据系列是否是二维线条
    # 一些后端期望：
    #   - get_points 返回 1D np.arrays list_x, list_y
    #   - get_color_array 返回 1D np.array (在 Line2DBaseSeries 中完成)
    #   以点的方式计算颜色，从 get_points 返回的点

    is_3Dline = False  # 标志，指示数据系列是否是三维线条
    # 一些后端期望：
    #   - get_points 返回 1D np.arrays list_x, list_y, list_y
    #   - get_color_array 返回 1D np.array (在 Line2DBaseSeries 中完成)
    #   以点的方式计算颜色，从 get_points 返回的点

    is_3Dsurface = False  # 标志，指示数据系列是否是三维表面
    # 一些后端期望：
    #   - get_meshes 返回 mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points 是 get_meshes 的别名

    is_contour = False  # 标志，指示数据系列是否是等高线
    # 一些后端期望：
    #   - get_meshes 返回 mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points 是 get_meshes 的别名

    is_implicit = False  # 标志，指示数据系列是否是隐式表达式
    # 一些后端期望：
    #   - get_meshes 返回 mesh_x (1D array), mesh_y(1D array), mesh_z (2D np.arrays)
    #   - get_points 是 get_meshes 的别名
    # 不同于 is_contour，因为后端中的颜色映射将不同

    is_interactive = False  # 标志，指示数据系列是否是交互式的，可以更新其数据。

    is_parametric = False  # 标志，指示数据系列是否是参数化的
    # 表示美学计算期望：
    #   - get_parameter_points 应返回一个或两个 np.array（1D 或 2D），用于计算美学

    is_generic = False
    # 表示通用用户提供的数值数据

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # 表示一个2D或3D向量数据系列

    _N = 100
    # 默认的离散化点数，用于均匀采样。每个子类可以设置自己的数量。

    def _block_lambda_functions(self, *exprs):
        """有些数据系列可用于绘制数值函数，而其他数据则不行。
        在 `__init__` 中执行此方法，以防止处理数值函数。
        """
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " 需要一个符号表达式。")

    def _check_fs(self):
        """ 检查是否有足够的参数和自由符号。
        """
        exprs, ranges = self.expr, self.ranges
        params, label = self.params, self.label
        exprs = exprs if hasattr(exprs, "__iter__") else [exprs]
        if any(callable(e) for e in exprs):
            return

        # 从表达式的自由符号中删除参数和范围中使用的符号
        fs = _get_free_symbols(exprs)
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 0:
            raise ValueError(
                "不兼容的表达式和参数。\n"
                + "表达式: {}\n".format(
                    (exprs, ranges, label) if ranges is not None else (exprs, label))
                + "参数: {}\n".format(params)
                + "请指定这些符号代表什么: {}\n".format(fs)
                + "它们是范围还是参数？"
            )

        # 验证所有符号是否已知（它们代表绘图范围或参数）
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                # 范围不能依赖于彼此，例如这种情况是不允许的：
                # (x, 0, y), (y, 0, 3)
                # (x, 0, y), (y, x + 2, 3)
                raise ValueError("范围符号不能包含在范围的最小值和最大值中。"
                    "收到的范围: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "在绘图范围中发现未知符号: %s。" % (r,) +
                    "以下是否是参数？ %s" % remaining_fs)
    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.

        Notes
        =====
        The old sympy.plotting used experimental_lambdify. It created one
        lambda function each time an evaluation was requested. If that failed,
        it went on to create a different lambda function and evaluated it,
        and so on.

        This new module changes strategy: it creates right away the default
        lambda function as well as the backup one. The reason is that the
        series could be interactive, hence the numerical function will be
        evaluated multiple times. So, let's create the functions just once.

        This approach works fine for the majority of cases, in which the
        symbolic expression is relatively short, hence the lambdification
        is fast. If the expression is very long, this approach takes twice
        the time to create the lambda functions. Be aware of that!
        """
        # Determine if self.expr is a single expression or a list of expressions
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        
        # Check if any expression in exprs is already callable (lambda function)
        if not any(callable(e) for e in exprs):
            # Extract free symbols from expressions
            fs = _get_free_symbols(exprs)
            # Sort symbols alphabetically by their name
            self._signature = sorted(fs, key=lambda t: t.name)

            # Initialize a list to store lambda functions for each expression
            self._functions = []
            for e in exprs:
                # Create two lambda functions for each expression:
                # 1. The default lambda function using specified modules.
                # 2. A backup lambda function using 'sympy' module with dummify=True.
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                self._functions.append([
                    lambdify(self._signature, e, modules=self.modules),
                    lambdify(self._signature, e, modules="sympy", dummify=True),
                ])
        else:
            # If any expression is already callable, assume they are stored in self.ranges
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            # Directly assign existing callable expressions to _functions
            self._functions = [(e, None) for e in exprs]

        # Handle the case where self.color_func is a symbolic expression
        if isinstance(self.color_func, Expr):
            # Convert self.color_func into a lambda function using self._signature
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        """If the value of a plotting range is a symbolic expression,
        substitute the parameters in order to get a numerical value.
        """
        # If plotting is not interactive, return t as a complex number
        if not self._interactive_ranges:
            return complex(t)
        # Substitute self.params into t to convert symbolic expression to numerical value
        return complex(t.subs(self.params))
    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization

        # Initialize empty lists to store symbols and discretizations
        discr_symbols = []
        discretizations = []

        # Iterate over each range to create a 1D discretization
        for i, r in enumerate(self.ranges):
            # Store the symbol corresponding to the range
            discr_symbols.append(r[0])

            # Update range values using a helper method and extract real parts if imaginary parts are zero
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end

            # Determine if integer discretization is needed based on conditions
            needs_integer_discr = self.only_integers or (r[0] in self._needs_to_be_int)

            # Compute the discretization for the current range
            d = BaseSeries._discretize(start, end, self.n[i],
                scale=self.scales[i],
                only_integers=needs_integer_discr)

            # Adjust the discretization based on additional conditions
            if ((not self._force_real_eval) and (not needs_integer_discr) and
                (d.dtype != "complex")):
                d = d + 1j * c_start.imag

            # Convert the discretization to integer type if needed
            if needs_integer_discr:
                d = d.astype(int)

            # Append the computed discretization to the list
            discretizations.append(d)

        # Call a helper method to create 2D or 3D discretized domains
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
        np = import_module('numpy')

        # Set the default indexing scheme for meshgrid
        # Determines the indexing scheme based on plot type requirements
        indexing = "xy"
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = "ij"

        # Generate meshgrid using numpy for the specified discretizations and indexing
        meshes = np.meshgrid(*discretizations, indexing=indexing)

        # Assign the generated meshgrid to self._discretized_domain as a dictionary
        self._discretized_domain = dict(zip(discr_symbols, meshes))
    def _evaluate(self, cast_to_real=True):
        """
        Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """

        np = import_module('numpy')  # 导入 NumPy 模块

        # create lambda functions
        if not self._functions:
            self._create_lambda_func()  # 如果没有定义 lambda 函数，则创建它们

        # create (or update) the discretized domain
        if (not self._discretized_domain) or self._interactive_ranges:
            self._create_discretized_domain()  # 创建（或更新）离散化域

        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.ranges]
        # 按照指定顺序获取离散化域，以确保返回顺序正确

        args = self._aggregate_args()  # 聚合参数

        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)  # 使用统一评估方法评估函数结果
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr[0])
            # 评估可能会产生 int/float 类型，需要进行形状校正

            # sometime the evaluation is performed over arrays of type object.
            # hence, `result` might be of type object, which don't work well
            # with numpy real and imag functions.
            r = r.astype(complex)
            results.append(r)  # 将结果添加到列表中

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
            # 如果需要转换为实数类型，则将离散化域的虚部丢弃

        return [*discr, *results]  # 返回离散化域和结果列表的组合

    def _aggregate_args(self):
        """
        Create a list of arguments to be passed to the lambda function,
        sorted accoring to self._signature.
        """
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(
                    int(self._params[s]) if s in self._needs_to_be_int else
                    self._params[s] if self._force_real_eval
                    else complex(self._params[s]))
                # 根据参数的类型需求，将参数转换为整数、实数或复数类型
            else:
                args.append(self._discretized_domain[s])
                # 如果参数不在参数字典中，则直接使用离散化域的值作为参数

        return args

    @property
    def expr(self):
        """Return the expression (or expressions) of the series."""
        return self._expr
        # 返回序列的表达式（或表达式）作为属性

    @expr.setter
    def expr(self, e):
        """Set the expression (or expressions) of the series."""
        # 检查参数 e 是否可迭代
        is_iter = hasattr(e, "__iter__")
        # 检查 e 是否为可调用对象，如果 e 是可迭代的，则检查其中的每个元素是否可调用
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        # 如果 e 是可调用的，则将其设置为表达式
        if is_callable:
            self._expr = e
        else:
            # 如果 e 不是可调用的，尝试将其转换为 sympy 的表达式
            self._expr = sympify(e) if not is_iter else Tuple(*e)

            # 查找表达式中所有的求和（Sum）和乘积（Product）操作的上界符号
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)

            # 定义一些 sympy 函数，它们在转换为 lambda 函数时，numpy 不支持复数类型参数
            pf = [ceiling, floor, atan2, frac, zeta]
            # 如果不强制使用实数进行评估，则检查表达式是否包含上述函数
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                # 如果强制使用实数评估，并且使用的是 numpy 模块，则发出警告
                if self._force_real_eval and ((self.modules is None) or
                    (isinstance(self.modules, str) and "numpy" in self.modules)):
                    funcs = [f for f, c in zip(pf, check_res) if c]
                    warnings.warn("NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            # 如果已定义自定义函数，则更新 lambda 函数
            if self._functions:
                self._create_lambda_func()

    @property
    def is_3D(self):
        """Check if the series represents a 3D plot."""
        # 返回当前系列是否是 3D 线条、表面或矢量的任意一种
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        """Check if the series represents a 2D or 3D line."""
        # 返回当前系列是否是 2D 或 3D 线条的任意一种
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        # 设置属性 prop 为 val，并根据 val 的类型决定是否设置 color_func
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        """Getter for line color."""
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        """Setter for line color."""
        # 调用 _line_surface_color 方法设置 _line_color 属性，并根据 val 的类型设置 color_func
        self._line_surface_color("_line_color", val)

    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points."""
        # 返回用于离散化的点的数量列表
        return self._n

    @n.setter
    def n(self, v):
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        The following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
        # Check if `v` is iterable; if not, assign `v` to the first element of self._n
        if not hasattr(v, "__iter__"):
            self._n[0] = v
        else:
            # Assign all elements of `v` to self._n, up to the length of `v`
            self._n[:len(v)] = v
        if self._discretized_domain:
            # If discretized domain exists, update it
            self._create_discretized_domain()

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        # Return the current parameters dictionary
        return self._params

    @params.setter
    def params(self, p):
        # Set the current parameters dictionary to `p`
        self._params = p

    def _post_init(self):
        # Convert self.expr to a list if it is not already iterable
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        # Check if any element in exprs is callable and self.params is not None
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # If any element in exprs is callable and self._label matches self.expr, set self.label to ""
        if any(callable(e) for e in exprs):
            if self._label == str(self.expr):
                self.label = ""

        # Check the 'fs' attribute
        self._check_fs()

        # If self.adaptive exists and is True and self.params is not None, issue a warning and set adaptive to False
        if hasattr(self, "adaptive") and self.adaptive and self.params:
            warnings.warn("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "adaptive evaluation. Automatically switched to "
                "adaptive=False.")
            self.adaptive = False

    @property
    def scales(self):
        # Return the current scales
        return self._scales

    @scales.setter
    def scales(self, v):
        # If `v` is a string, assign it to the first element of self._scales; otherwise, assign all elements of `v` to self._scales
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        # Return the current surface color
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        # Set the surface color using the private method _line_surface_color
        self._line_surface_color("_surface_color", val)

    @property
    def rendering_kw(self):
        # Return the current rendering keyword arguments
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        # 检查 kwargs 是否为字典类型，如果是则将其赋值给实例变量 _rendering_kw
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            # 如果不是字典类型，则将 _rendering_kw 设置为空字典
            self._rendering_kw = {}
            # 如果 kwargs 不为 None，则发出警告，并自动将 _rendering_kw 设置为空字典
            if kwargs is not None:
                warnings.warn(
                    "`rendering_kw` must be a dictionary, instead an "
                    "object of type %s was received. " % type(kwargs) +
                    "Automatically setting `rendering_kw` to an empty "
                    "dictionary")

    @staticmethod
    def _discretize(start, end, N, scale="linear", only_integers=False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
        np = import_module('numpy')

        # 如果 only_integers 参数为 True，则将 start 和 end 转换为整数，并重新计算 N
        if only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1

        # 根据 scale 参数选择线性或对数刻度，生成一维数组
        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_shape(a, b):
        """Convert ``a`` to a np.ndarray of the same shape of ``b``.

        Parameters
        ==========

        a : int, float, complex, np.ndarray
            Usually, this is the result of a numerical evaluation of a
            symbolic expression. Even if a discretized domain was used to
            evaluate the function, the result can be a scalar (int, float,
            complex). Think for example to ``expr = Float(2)`` and
            ``f = lambdify(x, expr)``. No matter the shape of the numerical
            array representing x, the result of the evaluation will be
            a single value.

        b : np.ndarray
            It represents the correct shape that ``a`` should have.

        Returns
        =======
        new_a : np.ndarray
            An array with the correct shape.
        """
        np = import_module('numpy')

        # 如果 a 不是 np.ndarray 类型，则转换为 np.ndarray 类型
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        
        # 如果 a 的形状不等于 b 的形状，则调整 a 的形状以匹配 b 的形状
        if a.shape != b.shape:
            # 如果 a 的形状为 ()，则将 a 扩展为与 b 相同的形状
            if a.shape == ():
                a = a * np.ones_like(b)
            else:
                a = a.reshape(b.shape)
        return a

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        # 抽象方法，子类需要实现此方法来计算并返回数值数据
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        # 根据给定的 latex 表示和包装器，将其包装在特定的字符中并返回
        return wrapper % label
    def get_label(self, use_latex=False, wrapper="$%s$"):
        """
        Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
            The label string based on the parameters provided.
        """
        if use_latex is False:
            # Return the stored string representation of the expression
            return self._label
        if self._label == str(self.expr):
            # Return the wrapped latex label if it matches the string representation
            return self._get_wrapped_label(self._latex_label, wrapper)
        # Return the stored latex label
        return self._latex_label

    @property
    def label(self):
        """
        Property method to retrieve the label.

        Returns
        =======
        str
            The label associated with this series.
        """
        return self.get_label()

    @label.setter
    def label(self, val):
        """
        Setter method to set the labels associated to this series.

        Parameters
        ==========
        val : str
            The new label value to be set.
        """
        # Set both _label and _latex_label attributes to the same new value
        self._label = self._latex_label = val

    @property
    def ranges(self):
        """
        Property method to retrieve the ranges.

        Returns
        =======
        list
            List of tuples representing ranges.
        """
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        """
        Setter method to set the ranges associated with this series.

        Parameters
        ==========
        val : list of tuples
            New ranges to be set. Each tuple should contain values to be sympified.
        """
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals
    # 对数值评估结果应用转换操作的方法

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        # 定义一个 lambda 函数 t，根据给定的 transform 函数对 x 进行转换，如果 transform 为 None，则返回 x 自身
        t = lambda x, transform: x if transform is None else transform(x)
        
        # 初始化 x, y, z 为 None
        x, y, z = None, None, None
        
        # 如果 args 的长度为 2，则将其解包给 x, y，并使用相应的 transform 函数进行转换
        if len(args) == 2:
            x, y = args
            return t(x, self._tx), t(y, self._ty)
        
        # 如果 args 的长度为 3，并且 self 是 Parametric2DLineSeries 类的实例，则将其解包给 x, y, u，并分别应用 transform 函数
        elif (len(args) == 3) and isinstance(self, Parametric2DLineSeries):
            x, y, u = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        
        # 如果 args 的长度为 3，则将其解包给 x, y, z，并分别应用 transform 函数
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        
        # 如果 args 的长度为 4，并且 self 是 Parametric3DLineSeries 类的实例，则将其解包给 x, y, z, u，并分别应用 transform 函数
        elif (len(args) == 4) and isinstance(self, Parametric3DLineSeries):
            x, y, z, u = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        
        # 如果 args 的长度为 4，则将其解包给 x, y, u, v，并分别应用 transform 函数，适用于二维向量图
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        
        # 如果 args 的长度为 5，并且 self 是 ParametricSurfaceSeries 类的实例，则将其解包给 x, y, z, u, v，并部分应用 transform 函数
        elif (len(args) == 5) and isinstance(self, ParametricSurfaceSeries):
            x, y, z, u, v = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        
        # 如果 args 的长度为 6，并且 is_3Dvector 为 True，则将其解包给 x, y, z, u, v, w，并分别应用 transform 函数，适用于三维向量图
        elif (len(args) == 6) and self.is_3Dvector: # 3D vector plot
            x, y, z, u, v, w = args
            return (
                t(x, self._tx), t(y, self._ty), t(z, self._tz),
                t(u, self._tx), t(v, self._ty), t(w, self._tz)
            )
        
        # 如果 args 的长度为 6，则将其解包给 x, y, _abs, _arg, img, colors，并应用部分的 transform 函数，适用于复杂图形
        elif len(args) == 6: # complex plot
            x, y, _abs, _arg, img, colors = args
            return (
                x, y, t(_abs, self._tz), _arg, img, colors)
        
        # 如果没有匹配到任何条件，则直接返回原始的 args
        return args

    # 返回一个字符串 s 的辅助方法，根据对象的属性 is_interactive，添加前缀和后缀信息
    def _str_helper(self, s):
        pre, post = "", ""
        
        # 如果对象具有 is_interactive 属性，则将前缀设置为 "interactive "，后缀为参数的字符串表示
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        
        # 返回带有前缀和后缀的字符串 s
        return pre + s + post
# 导入 numpy 模块
np = import_module('numpy')

# 定义一个基础类 Line2DBaseSeries，继承自 BaseSeries
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - 添加标签、步长和仅整数选项
    - 将 is_2Dline 设为 True
    - 定义 get_segments 和 get_color_array 方法
    """

    # 类属性，表示这是一个二维线条
    is_2Dline = True
    # 维度属性设为 2
    _dim = 2
    # 默认的数据点数量
    _N = 1000
    # 初始化函数，用于设置对象的各种属性
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置步骤属性，如果未提供则默认为 False
        self.steps = kwargs.get("steps", False)
        # 设置是否为点的标志属性，如果未提供则尝试使用 "point"，默认为 False
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        # 设置是否填充的标志属性，如果未提供则尝试使用 "fill"，默认为 True
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        # 设置是否自适应的标志属性，如果未提供则默认为 False
        self.adaptive = kwargs.get("adaptive", False)
        # 设置深度属性，如果未提供则默认为 12
        self.depth = kwargs.get('depth', 12)
        # 设置是否使用厘米单位的标志属性，如果未提供则默认为 False
        self.use_cm = kwargs.get("use_cm", False)
        # 设置颜色函数属性，如果未提供则默认为 None
        self.color_func = kwargs.get("color_func", None)
        # 设置线条颜色属性，如果未提供则默认为 None
        self.line_color = kwargs.get("line_color", None)
        # 设置是否检测极点的标志属性，如果未提供则默认为 False
        self.detect_poles = kwargs.get("detect_poles", False)
        # 设置 eps（极小值）属性，如果未提供则默认为 0.01
        self.eps = kwargs.get("eps", 0.01)
        # 设置是否极坐标的标志属性，如果未提供则尝试使用 "polar"，默认为 False
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        # 设置是否展开的标志属性，如果未提供则默认为 False
        self.unwrap = kwargs.get("unwrap", False)
        
        # 当 detect_poles="symbolic" 时，存储极点的位置信息，以便适当渲染
        self.poles_locations = []
        
        # 处理 exclude 属性，将其转换为列表并确保其每个元素为浮点数
        exclude = kwargs.get("exclude", [])
        if isinstance(exclude, Set):  # 如果 exclude 是集合类型，则从中提取解的元素，最多提取 100 个
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, "__iter__"):  # 如果 exclude 不可迭代，则转换为单元素列表
            exclude = [exclude]
        exclude = [float(e) for e in exclude]  # 将 exclude 中的元素转换为浮点数
        self.exclude = sorted(exclude)  # 对 exclude 列表进行排序赋值给对象属性
    # 定义一个方法，用于获取用于绘制线条的坐标数据
    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
        np = import_module('numpy')  # 导入 numpy 模块，并赋值给 np
        points = self._get_data_helper()  # 调用内部方法 _get_data_helper() 获取数据点

        # 检查是否为 LineOver1DRangeSeries 并且 detect_poles 属性为 "symbolic"
        if (isinstance(self, LineOver1DRangeSeries) and
            (self.detect_poles == "symbolic")):
            # 对符号表达式中的极点进行符号计算检测
            poles = _detect_poles_symbolic_helper(
                self.expr.subs(self.params), *self.ranges[0])
            # 将极点转换为浮点数数组，并应用可能的坐标变换
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = t(np.array(poles), self._tx)

        # 应用后处理转换
        points = self._apply_transform(*points)

        # 如果是 2D 线条并且需要检测极点
        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                # 对数值数据中的极点进行数值计算检测
                x, y = _detect_poles_numerical_helper(
                    x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)

        # 如果设置了 unwrap 属性
        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            # 如果是 2D 线条
            if self.is_2Dline:
                if len(points) == 2:
                    x, y = points
                    # 对 y 坐标进行相位展开处理
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    x, y, p = points
                    # 对 y 坐标进行相位展开处理
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)

        # 如果设置了 steps 属性为 True
        if self.steps is True:
            # 如果是 2D 线条
            if self.is_2Dline:
                x, y = points[0], points[1]
                # 沿 x 和 y 轴增加步进点
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            # 如果是 3D 线条
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)

        # 如果 exclude 列表中有元素
        if len(self.exclude) > 0:
            # 在数据点中插入排除点
            points = self._insert_exclusions(points)
        # 返回处理后的数据点
        return points
    # 定义一个方法 get_segments，用于获取线段的信息
    def get_segments(self):
        # 发出 SymPy 弃用警告，提醒使用者该方法已经被弃用，建议使用其他方法或者新的接口
        sympy_deprecation_warning(
            """
            The Line2DBaseSeries.get_segments() method is deprecated.

            Instead, use the MatplotlibBackend.get_segments() method, or use
            The get_points() or get_data() methods.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-get-segments")

        # 导入 numpy 模块
        np = import_module('numpy')
        # 调用当前类的 get_data 方法获取数据点
        points = type(self).get_data(self)
        # 将数据点转换为 numpy 的 masked array，并进行形状调整以便于处理线段
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        # 将相邻的点组合成线段，返回一个连接了所有线段的 masked array
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)
    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exlusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
        np = import_module("numpy")  # 导入 numpy 模块，用于数值计算
        points = list(points)  # 将输入的 points 转换为列表形式
        n = len(points)  # 获取 points 列表的长度
        # index of the x-coordinate (for 2d plots) or parameter (for 2d/3d
        # parametric plots)
        k = n - 1  # k 是 x 坐标（对于2D图）或参数（对于2D/3D参数化图）的索引
        if n == 2:
            k = 0  # 如果 points 的长度为2，则 k 设为0，对应于第一个坐标或参数
        # indeces of the other coordinates
        j_indeces = sorted(set(range(n)).difference([k]))  # 计算除了 k 以外的坐标索引集合
        # TODO: for now, I assume that numpy functions are going to succeed
        funcs = [f[0] for f in self._functions]  # 从 self._functions 中提取函数列表

        for e in self.exclude:  # 遍历每个需要排除的点
            res = points[k] - e >= 0  # 检查每个点与排除点的关系
            # if res contains both True and False, ie, if e is found
            if any(res) and any(~res):  # 如果 res 同时包含 True 和 False，即找到了 e
                idx = np.nanargmax(res)  # 找到最大值所在的索引，即最接近 e 的点
                # select the previous point with respect to e
                idx -= 1  # 选择 e 前面的一个点
                # TODO: what if points[k][idx]==e or points[k][idx+1]==e?

                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])  # 计算 e 与前一个点的距离
                    delta_post = abs(e - points[k][idx + 1])  # 计算 e 与后一个点的距离
                    delta = min(delta_prev, delta_post) / 100  # 计算一个小的增量
                    prev = e - delta  # 前一个新加的点
                    post = e + delta  # 后一个新加的点

                    # add points to the x-coord or the parameter
                    points[k] = np.concatenate(
                        (points[k][:idx], [prev, e, post], points[k][idx+1:]))
                    # 将新加的点加入到 x 坐标或参数中

                    # add points to the other coordinates
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))  # 计算其他坐标的值
                        c += 1
                        points[j] = np.concatenate(
                            (points[j][:idx], [values[0], np.nan, values[1]], points[j][idx+1:]))
                    # 将新加的点加入到其他坐标中

        return points  # 返回处理后的 points 列表

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])  # 尝试将起始值转换为指定类型
        except TypeError:
            return self.ranges[0][1]  # 如果转换失败，则直接返回起始值

    @property
    def end(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])  # 尝试将结束值转换为指定类型
        except TypeError:
            return self.ranges[0][2]  # 如果转换失败，则直接返回结束值
    # 获取 x 轴的比例尺
    def xscale(self):
        return self._scales[0]

    # 设置 x 轴的比例尺
    @xscale.setter
    def xscale(self, v):
        self.scales = v

    # 获取颜色数组
    def get_color_array(self):
        np = import_module('numpy')  # 导入 NumPy 模块
        c = self.line_color  # 获取线条颜色
        if hasattr(c, '__call__'):  # 检查颜色是否可调用（即函数）
            f = np.vectorize(c)  # 创建一个能够处理 NumPy 数组的函数
            nargs = arity(c)  # 获取函数的参数个数
            if nargs == 1 and self.is_parametric:  # 如果函数有一个参数且是参数化的
                x = self.get_parameter_points()  # 获取参数点
                return f(centers_of_segments(x))  # 对参数点进行处理并返回结果
            else:
                variables = list(map(centers_of_segments, self.get_points()))  # 获取数据点的中心
                if nargs == 1:
                    return f(variables[0])  # 对第一个变量进行处理并返回结果
                elif nargs == 2:
                    return f(*variables[:2])  # 对前两个变量进行处理并返回结果
                else:  # 只有在三维线条时才会执行（否则会引发错误）
                    return f(*variables)  # 对所有变量进行处理并返回结果
        else:
            return c * np.ones(self.nb_of_points)  # 返回颜色乘以点的数量的 NumPy 数组
class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        np = import_module('numpy')  # 导入NumPy模块

        # 检查传入的列表长度是否相同，如果不同则抛出ValueError异常
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(list_x)) +
                "and len(list_y) = {}".format(len(list_y))
            )

        self._block_lambda_functions(list_x, list_y)  # 调用内部方法处理lambda函数

        # 检查列表中是否包含符号表达式，若包含则需要提供params字典进行求值
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)  # 如果有符号表达式，使用Tuple封装列表
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)  # 转换列表为NumPy数组
            self.list_y = np.array(list_y, dtype=np.float64)

        self._expr = (self.list_x, self.list_y)  # 设置表达式为列表x和y的元组

        # 如果列表x和y中没有NumPy数组，则调用_check_fs方法
        if not any(isinstance(t, np.ndarray) for t in [self.list_x, self.list_y]):
            self._check_fs()

        # 设置是否为极坐标，并设置标签和渲染参数
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.label = label
        self.rendering_kw = kwargs.get("rendering_kw", {})

        # 如果使用色彩映射且颜色函数可调用，则设置为参数曲线
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError(
                    "%s don't support symbolic " % self.__class__.__name__ +
                    "expression for `color_func`."
                )

    def __str__(self):
        return "2D list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)  # 如果不是交互模式，直接返回处理后的数据

        np = import_module('numpy')  # 重新导入NumPy模块
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)  # 对lx列表中的符号表达式求值为浮点数数组
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)  # 对ly列表中的符号表达式求值为浮点数数组
        return self._eval_color_func_and_return(lx, ly)  # 返回处理后的数据

    def _eval_color_func_and_return(self, *data):
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]  # 如果使用色彩映射且颜色函数可调用，返回处理后的数据及颜色函数的评估结果
        return data


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""
    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        # 初始化函数，接收表达式、变量起始与结束值的元组、标签和其他关键字参数
        self.expr = expr if callable(expr) else sympify(expr)
        # 将表达式转换为符号对象（sympy expression），如果表达式不可调用则进行转换
        self._label = str(self.expr) if label is None else label
        # 设置标签，如果标签为空则使用表达式的字符串表示
        self._latex_label = latex(self.expr) if label is None else label
        # 将表达式转换为LaTeX格式的字符串，如果标签为空则使用标签
        self.ranges = [var_start_end]
        # 将变量的起始与结束值元组存入ranges列表
        self._cast = complex
        # 设置_cast属性为complex，用于类型转换
        # for complex-related data series, this determines what data to return
        # on the y-axis
        # 用于复数相关的数据系列，决定在y轴上返回什么数据

        self._return = kwargs.get("return", None)
        # 设置_return属性为关键字参数中的return值，如果没有则为None
        self._post_init()
        # 调用初始化后处理函数_post_init()

        if not self._interactive_ranges:
            # 如果不是交互式范围
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            # 注意：以下检查仅在绘图范围的最小值和最大值是数值时才可能进行
            start, end = [complex(t) for t in self.ranges[0][1:]]
            # 将ranges列表中第一个元素的起始与结束值元组的后两个值转换为复数
            if im(start) != im(end):
                # 如果起始值和结束值的虚部不相等
                raise ValueError(
                    "%s requires the imaginary " % self.__class__.__name__ +
                    "part of the start and end values of the range "
                    "to be the same.")
                # 抛出值错误，要求范围的起始和结束值的虚部相同

        if self.adaptive and self._return:
            # 如果开启了自适应并且有_return属性
            warnings.warn("The adaptive algorithm is unable to deal with "
                "complex numbers. Automatically switching to uniform meshing.")
            # 发出警告，自适应算法无法处理复数。自动切换到均匀网格化。
            self.adaptive = False
            # 将adaptive属性设置为False

    @property
    def nb_of_points(self):
        # 返回点的数量
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        # 设置点的数量
        self.n = v

    def __str__(self):
        # 返回对象的字符串表示
        def f(t):
            # 辅助函数，将复数转换为实数或保留复数的实部
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = "interactive " if self.is_interactive else ""
        # 如果是交互式，则添加前缀"interactive "
        post = ""
        if self.is_interactive:
            post = " and parameters " + str(tuple(self.params.keys()))
            # 如果是交互式，则添加后缀" and parameters "以及参数字典的键的字符串表示
        wrapper = _get_wrapper_for_expr(self._return)
        # 获取用于表达式的包装器函数
        return pre + "cartesian line: %s for %s over %s" % (
            wrapper % self.expr,
            str(self.var),
            str((f(self.start), f(self.end))),
        ) + post
        # 返回包含对象信息的字符串表示，包括表达式、变量、起始和结束值

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates
        """
        return self._get_data_helper()
        # 返回用于绘图的坐标列表，根据adaptive选项选择自适应算法或在提供的范围内均匀采样表达式
    def _adaptive_sampling(self):
        try:
            # 如果 self.expr 是可调用对象，则直接使用
            if callable(self.expr):
                f = self.expr
            else:
                # 否则，将 self.expr 转换为一个可调用的函数 f
                f = lambdify([self.var], self.expr, self.modules)
            # 使用 f 进行自适应采样，获取 x 和 y 的值
            x, y = self._adaptive_sampling_helper(f)
        except Exception as err:
            # 如果出现异常，给出警告并尝试使用 Sympy 进行表达式的评估
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            # 使用 Sympy 模块重新创建 f，并进行自适应采样
            f = lambdify([self.var], self.expr, "sympy")
            x, y = self._adaptive_sampling_helper(f)
        # 返回自适应采样得到的 x 和 y 值
        return x, y

    def _uniform_sampling(self):
        np = import_module('numpy')

        # 使用 _evaluate 方法获取 x 和 result
        x, result = self._evaluate()
        # 分别提取 result 的实部和虚部
        _re, _im = np.real(result), np.imag(result)
        # 根据 x 的形状对 _re 和 _im 进行形状修正
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        # 返回 x，_re 和 _im
        return x, _re, _im

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """
        np = import_module('numpy')
        # 如果开启了自适应采样且不仅限于整数值，则调用 _adaptive_sampling 方法
        if self.adaptive and (not self.only_integers):
            x, y = self._adaptive_sampling()
            # 返回 x 和 y 作为 numpy 数组列表
            return [np.array(t) for t in [x, y]]

        # 否则，调用 _uniform_sampling 方法进行均匀采样
        x, _re, _im = self._uniform_sampling()

        if self._return is None:
            # 如果 _return 为 None，则可能评估结果包含复数。在有非零虚部的情况下将对应的实部设为 NaN
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == "real":
            pass
        elif self._return == "imag":
            # 如果 _return 为 "imag"，则将 _re 设为 _im
            _re = _im
        elif self._return == "abs":
            # 如果 _return 为 "abs"，则将 _re 设为 result 的模长
            _re = np.sqrt(_re**2 + _im**2)
        elif self._return == "arg":
            # 如果 _return 为 "arg"，则将 _re 设为 result 的幅角
            _re = np.arctan2(_im, _re)
        else:
            # 如果 _return 不在预期的值中，则引发 ValueError 异常
            raise ValueError("`_return` not recognized. "
                "Received: %s" % self._return)

        # 返回 x 和处理后的 _re
        return x, _re
class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True  # 设定类属性is_parametric为True，表示这是一个参数化线的基础系列类

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label = str(self.var) if label is None else label  # 如果label为None，则使用self.var的字符串表示作为_label
        self._latex_label = latex(self.var) if label is None else label  # 如果label为None，则使用self.var的LaTeX表示作为_latex_label
        if (self.use_cm is False) and (self._label == str(self.var)):
            # 如果use_cm为False且_label等于self.var的字符串表示，则使用self.expr的字符串表示作为_label和_latex_label
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid suprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            # 如果表达式中有任何一个是可调用的函数，并且use_cm为False，且没有提供label，则将_label置为空字符串
            if self._label == str(self.expr):
                self._label = ""

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            np = import_module("numpy")  # 导入numpy模块
            coords = self._adaptive_sampling()  # 使用自适应采样算法获取坐标
            coords = [np.array(t) for t in coords]  # 将坐标转换为numpy数组形式
        else:
            coords = self._uniform_sampling()  # 使用均匀采样算法获取坐标

        if self.is_2Dline and self.is_polar:
            # when plot_polar is executed with polar_axis=True
            np = import_module('numpy')  # 导入numpy模块
            x, y, _ = coords  # 解包coords中的坐标x, y
            r = np.sqrt(x**2 + y**2)  # 计算极坐标中的半径r
            t = np.arctan2(y, x)  # 计算极坐标中的角度t
            coords = [t, r, coords[-1]]  # 更新coords为[t, r, 最后一个元素]

        if callable(self.color_func):  # 如果color_func是可调用的函数
            coords = list(coords)  # 将coords转换为列表
            coords[-1] = self.eval_color_func(*coords)  # 对最后一个元素使用eval_color_func进行颜色函数评估

        return coords  # 返回处理后的坐标数据
    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        # 导入 numpy 模块
        np = import_module('numpy')

        # 调用 _evaluate 方法获取结果
        results = self._evaluate()
        # 遍历结果列表
        for i, r in enumerate(results):
            # 将复数结果 r 拆分为实部和虚部
            _re, _im = np.real(r), np.imag(r)
            # 将非零虚部对应的实部设置为 NaN
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            # 更新结果列表中的第 i 项为处理后的实部 _re
            results[i] = _re

        # 返回处理后的结果，循环后的第一个元素被移到最后
        return [*results[1:], results[0]]

    def get_parameter_points(self):
        # 返回 get_data 方法的最后一个元素
        return self.get_data()[-1]

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates
            y : list
                List of y-coordinates
            z : list
                List of z-coordinates, only for 3D parametric line plot.
        """
        # 返回 _get_data_helper 方法返回的结果列表的前两项
        return self._get_data_helper()[:-1]

    @property
    def nb_of_points(self):
        # 返回属性 n 的第一个元素作为点的数量
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        # 设置属性 n 的第一个元素为新值 v
        self.n = v
### 2D Parametric Line Series
class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""

    is_2Dline = True  # 标志：这是一个二维线条系列

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)  # 存储 x 和 y 方向的表达式
        self.ranges = [var_start_end]  # 存储变量的起始和结束范围
        self._cast = float  # 类型转换函数，这里是 float
        self.use_cm = kwargs.get("use_cm", True)  # 是否使用厘米单位，默认为 True
        self._set_parametric_line_label(label)  # 设置线条的标签
        self._post_init()  # 执行初始化后的操作

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end))
        ))  # 返回线条的字符串表示，包括表达式和变量范围

    def _adaptive_sampling(self):
        try:
            if callable(self.expr_x) and callable(self.expr_y):
                f_x = self.expr_x
                f_y = self.expr_y
            else:
                f_x = lambdify([self.var], self.expr_x)
                f_y = lambdify([self.var], self.expr_y)
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)  # 执行自适应采样
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f_x = lambdify([self.var], self.expr_x, "sympy")
            f_y = lambdify([self.var], self.expr_y, "sympy")
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)  # 用 Sympy 再次尝试采样
        return x, y, p  # 返回采样结果

### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False  # 标志：这不是一个二维线条系列
    is_3Dline = True  # 标志：这是一个三维线条系列
    _dim = 3  # 维度数为 3

    def __init__(self):
        super().__init__()  # 调用父类的初始化函数


class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""

    is_2Dline = False  # 标志：这不是一个二维线条系列
    is_3Dline = True  # 标志：这是一个三维线条系列
    # 初始化方法，用于设置参数化线的表达式和范围
    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果expr_x不是可调用对象，则将其转换为SymPy表达式
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        # 如果expr_y不是可调用对象，则将其转换为SymPy表达式
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        # 如果expr_z不是可调用对象，则将其转换为SymPy表达式
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        # 将表达式存储在元组中
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        # 存储变量的起始和结束范围
        self.ranges = [var_start_end]
        # 将_cast属性设置为float类型
        self._cast = float
        # 是否使用厘米单位，默认为True
        self.use_cm = kwargs.get("use_cm", True)
        # 设置参数化线的标签
        self._set_parametric_line_label(label)
        # 执行初始化后的额外操作
        self._post_init()
        # TODO: remove this
        # 下面三行是为了待移除的功能，暂时设置为None
        self._xlim = None
        self._ylim = None
        self._zlim = None

    # 返回描述对象的字符串表示形式
    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))

    # 获取数据的方法，返回x、y、z坐标和参数化变量
    def get_data(self):
        # TODO: remove this
        # 导入numpy模块
        np = import_module("numpy")
        # 调用父类的get_data方法，获取x、y、z、p四个变量
        x, y, z, p = super().get_data()
        # 计算x、y、z各自的最小值和最大值，存储在_xlim、_ylim、_zlim属性中
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        # 返回x、y、z、p四个变量
        return x, y, z, p
### Surfaces
# 表面基础系列的类，继承自基础系列
class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    # 标志：表明这是一个3D表面
    is_3Dsurface = True

    # 初始化函数
    def __init__(self, *args, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 是否使用色彩映射，默认为False
        self.use_cm = kwargs.get("use_cm", False)
        # 是否极坐标系，默认为False
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        # 表面颜色，默认为None
        self.surface_color = kwargs.get("surface_color", None)
        # 颜色函数，默认为 lambda 函数，返回 z 值
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        # 如果 surface_color 是可调用对象，则将其作为颜色函数，并将 surface_color 设为 None
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    # 设置表面标签的方法
    def _set_surface_label(self, label):
        # 获取表达式
        exprs = self.expr
        # 如果标签为None，则将表达式转换为字符串形式
        self._label = str(exprs) if label is None else label
        # 如果标签为None，则将表达式转换为 LaTeX 形式
        self._latex_label = latex(exprs) if label is None else label
        # 如果表达式是 lambda 函数并且没有提供标签，则将标签置空，避免后端的意外
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self._label = ""
                self._latex_label = ""

    # 获取颜色数组的方法
    def get_color_array(self):
        # 导入 numpy 模块
        np = import_module('numpy')
        # 获取表面颜色
        c = self.surface_color
        # 如果表面颜色是可调用对象
        if isinstance(c, Callable):
            # 使用 numpy 的 vectorize 方法创建向量化函数
            f = np.vectorize(c)
            # 获取可调用对象的参数个数
            nargs = arity(c)
            # 如果是参数化的表面
            if self.is_parametric:
                # 获取参数网格的中心点
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                # 根据参数个数调用函数
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            # 如果不是参数化的表面
            variables = list(map(centers_of_faces, self.get_meshes()))
            # 根据参数个数调用函数
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        else:
            # 如果是 SurfaceOver2DRangeSeries 类型的表面，则返回表面颜色乘以长度较小的网格点数
            if isinstance(self, SurfaceOver2DRangeSeries):
                return c*np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
            else:
                return c*np.ones(min(self.nb_of_points_u, self.nb_of_points_v))


# 表示一个由 SymPy 表达式和2D范围组成的3D表面的类
class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""
    # 初始化函数，用于设置表达式、变量范围和标签
    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果表达式是可调用的则直接使用，否则将其转换为 sympy 表达式
        self.expr = expr if callable(expr) else sympify(expr)
        # 设置变量范围，var_start_end_x 是 x 范围，var_start_end_y 是 y 范围
        self.ranges = [var_start_end_x, var_start_end_y]
        # 设置表面图的标签
        self._set_surface_label(label)
        # 执行额外的初始化步骤
        self._post_init()
        # TODO: remove this
        # 设置 x 轴和 y 轴的限制范围
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    @property
    # 获取 x 变量的起始值
    def var_x(self):
        return self.ranges[0][0]

    @property
    # 获取 y 变量的起始值
    def var_y(self):
        return self.ranges[1][0]

    @property
    # 获取 x 范围的起始值（处理可能的类型错误）
    def start_x(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    # 获取 x 范围的结束值（处理可能的类型错误）
    def end_x(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    # 获取 y 范围的起始值（处理可能的类型错误）
    def start_y(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    # 获取 y 范围的结束值（处理可能的类型错误）
    def end_y(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    # 获取 x 方向上的点的数量
    def nb_of_points_x(self):
        return self.n[0]

    @nb_of_points_x.setter
    # 设置 x 方向上的点的数量
    def nb_of_points_x(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    # 获取 y 方向上的点的数量
    def nb_of_points_y(self):
        return self.n[1]

    @nb_of_points_y.setter
    # 设置 y 方向上的点的数量
    def nb_of_points_y(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    # 返回对象的字符串表示形式，包括表达式、变量和范围信息
    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        ))

    # 获取用于绘制表面的 x, y, z 坐标数据
    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        # 调用 get_data() 方法返回数据，用于后向兼容性
        return self.get_data()
    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray
            Discretized x-domain.
        mesh_y : np.ndarray
            Discretized y-domain.
        mesh_z : np.ndarray
            Results of the evaluation.
        """
        np = import_module('numpy')  # 导入 NumPy 模块

        results = self._evaluate()  # 调用对象内部方法 _evaluate() 获取评估结果

        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re  # 将复数部分不为零的值设为 NaN

        x, y, z = results  # 将结果拆分为 x, y, z 三个数组

        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)  # 若为极坐标且为三维表面图，根据极坐标转换计算 x, y

        # TODO: remove this
        self._zlim = (np.amin(z), np.amax(z))  # 设置对象内部属性 _zlim 为 z 的最小值和最大值的元组

        return self._apply_transform(x, y, z)  # 返回经过转换处理后的 x, y, z 数组
    # ParametricSurfaceSeries 类，继承自 SurfaceBaseSeries 类，用于表示由三个参数化 SymPy 表达式和一个范围组成的 3D 表面。

    is_parametric = True
    # 设置属性 is_parametric 为 True，表示这是一个参数化的表面。

    def __init__(self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs):
        # 构造函数，接受三个参数化表达式 expr_x, expr_y, expr_z 和两个变量范围 var_start_end_u, var_start_end_v，
        # 可选参数包括 label 和其他关键字参数。
        
        super().__init__(**kwargs)
        # 调用父类 SurfaceBaseSeries 的构造函数。

        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        # 如果 expr_x 是可调用的则直接使用，否则将其转换为 SymPy 表达式。

        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        # 如果 expr_y 是可调用的则直接使用，否则将其转换为 SymPy 表达式。

        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        # 如果 expr_z 是可调用的则直接使用，否则将其转换为 SymPy 表达式。

        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        # 将三个表达式组成的元组存储在 self.expr 中。

        self.ranges = [var_start_end_u, var_start_end_v]
        # 将变量范围列表 [var_start_end_u, var_start_end_v] 存储在 self.ranges 中。

        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)
        # 如果关键字参数中包含 "color_func"，则使用其值作为颜色函数，否则默认为返回 z 值的函数。

        self._set_surface_label(label)
        # 调用 _set_surface_label 方法设置表面的标签。

        self._post_init()
        # 调用 _post_init 方法完成初始化操作。

    @property
    def var_u(self):
        return self.ranges[0][0]
        # 返回变量 u 的起始值。

    @property
    def var_v(self):
        return self.ranges[1][0]
        # 返回变量 v 的起始值。

    @property
    def start_u(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]
        # 尝试将 u 的起始值转换为浮点数，如果类型错误则返回原始值。

    @property
    def end_u(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]
        # 尝试将 u 的结束值转换为浮点数，如果类型错误则返回原始值。

    @property
    def start_v(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]
        # 尝试将 v 的起始值转换为浮点数，如果类型错误则返回原始值。

    @property
    def end_v(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]
        # 尝试将 v 的结束值转换为浮点数，如果类型错误则返回原始值。

    @property
    def nb_of_points_u(self):
        return self.n[0]
        # 返回 u 方向上的点数。

    @nb_of_points_u.setter
    def nb_of_points_u(self, v):
        n = self.n
        self.n = [v, n[1:]]
        # 设置 u 方向上的点数。

    @property
    def nb_of_points_v(self):
        return self.n[1]
        # 返回 v 方向上的点数。

    @nb_of_points_v.setter
    def nb_of_points_v(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]
        # 设置 v 方向上的点数。

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        ))
        # 返回对象的字符串表示形式，描述参数化的笛卡尔表面及其参数信息。

    def get_parameter_meshes(self):
        return self.get_data()[3:]
        # 调用 get_data 方法获取数据并返回参数网格。

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()[:3]
        # 返回用于绘制表面的 x,y,z 坐标，为了向后兼容性考虑，建议使用 get_data 方法代替。
    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """
        np = import_module('numpy')  # 导入并命名 numpy 模块为 np

        results = self._evaluate()  # 调用对象的 _evaluate 方法获取计算结果

        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)  # 分别获取结果 r 的实部和虚部
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan  # 将非零虚部对应的实部置为 NaN
            results[i] = _re  # 更新 results 中第 i 个元素为处理后的实部数组

        # TODO: remove this  # 标记：后续需要移除这段代码
        x, y, z = results[2:]  # 获取 results 中的第 2、3、4 个元素作为 x, y, z
        self._xlim = (np.amin(x), np.amax(x))  # 计算 x 的最小值和最大值，保存到对象属性 _xlim
        self._ylim = (np.amin(y), np.amax(y))  # 计算 y 的最小值和最大值，保存到对象属性 _ylim
        self._zlim = (np.amin(z), np.amax(z))  # 计算 z 的最小值和最大值，保存到对象属性 _zlim

        return self._apply_transform(*results[2:], *results[:2])  # 调用对象的 _apply_transform 方法，传入 x, y, z, mesh_u, mesh_v，并返回结果
### Contours
# 定义一个轮廓系列，继承自SurfaceOver2DRangeSeries类
class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False  # 不是3D表面图
    is_contour = True  # 是轮廓图

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))  # 是否填充，默认为True
        self.show_clabels = kwargs.get("clabels", True)  # 是否显示轮廓线标签，默认为True

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        # 备注：轮廓图被plot_contour、plot_vector和plot_complex_vector使用。
        # 通过实现contour_kw，我们能够快速定位轮廓图。
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", {}))  # 渲染关键字参数，用于定制轮廓图的绘制


class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax[0]
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readibility. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    """
    is_generic = True  # 是通用数值数据系列

    def __init__(self, tp, *args, **kwargs):
        self.type = tp  # 类型
        self.args = args  # 参数
        self.rendering_kw = kwargs  # 渲染关键字参数

    def get_data(self):
        return self.args  # 返回数据


class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""

    is_implicit = True  # 是2D隐式图
    use_cm = False  # 不使用色彩映射
    _N = 100  # 默认N值为100
    # 初始化函数，接受表达式、变量范围、标签和其他关键字参数
    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 是否使用自适应采样，默认为 False
        self.adaptive = kwargs.get("adaptive", False)
        # 表达式本身
        self.expr = expr
        # 标签字符串，如果未提供则使用表达式的字符串表示
        self._label = str(expr) if label is None else label
        # LaTeX 格式的标签，如果未提供则转换表达式为 LaTeX 格式
        self._latex_label = latex(expr) if label is None else label
        # 变量范围的列表
        self.ranges = [var_start_end_x, var_start_end_y]
        # 变量 x 的名称及其起始和结束值
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        # 变量 y 的名称及其起始和结束值
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        # 线条颜色，可以从关键字参数中获取，否则为 None
        self._color = kwargs.get("color", kwargs.get("line_color", None))

        # 如果是交互式且自适应为 True，则抛出未实现错误
        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # 检查深度是否大于4或小于0
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        # 设置深度值
        self.depth = 4 + depth
        # 执行后续初始化操作
        self._post_init()

    # 获取表达式的属性方法
    @property
    def expr(self):
        # 如果使用自适应采样，则返回自适应表达式
        if self.adaptive:
            return self._adaptive_expr
        # 否则返回非自适应表达式
        return self._non_adaptive_expr

    # 设置表达式的属性方法
    @expr.setter
    def expr(self, expr):
        # 阻塞 Lambda 函数以及自适应评估所需
        self._block_lambda_functions(expr)
        # 检查表达式是否包含等式，并返回修正后的表达式及其等式状态
        expr, has_equality = self._has_equality(sympify(expr))
        # 设置自适应表达式及其等式状态
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self._label = str(expr)
        self._latex_label = latex(expr)

        # 如果表达式是布尔函数且非自适应，则自动设置为自适应，并发出警告
        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            self.adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(
                "The provided expression " + msg
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        # 如果表达式是布尔函数，则清除非自适应表达式及其等式状态
        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            # 否则，对表达式进行预处理以用于均匀网格评估
            expr, is_equality = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

    # 获取线条颜色的属性方法
    @property
    def line_color(self):
        return self._color

    # 设置线条颜色的属性方法
    @line_color.setter
    def line_color(self, v):
        self._color = v

    # 将 color 属性和 line_color 属性关联起来
    color = line_color
    def _has_equality(self, expr):
        # 判断表达式中是否包含等式（Equality）、大于（GreaterThan）或小于（LessThan）
        has_equality = False

        def arg_expand(bool_expr):
            """递归地展开布尔函数的参数"""
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # 检查参数列表中是否包含等式、大于或小于的关系表达式
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            # 如果表达式不是关系表达式，则将其视为等式和0的比较
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            # 如果表达式本身是等式、大于或小于的关系表达式，则直接判断为True
            has_equality = True

        return expr, has_equality

    def __str__(self):
        # 将数值转换为浮点数，如果表达式中不包含自由符号则直接转换
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            "Implicit expression: %s for %s over %s and %s over %s") % (
            str(self._adaptive_expr),
            str(self.var_x),
            str((f(self.start_x), f(self.end_x))),
            str(self.var_y),
            str((f(self.start_y), f(self.end_y))),
        )

    def get_data(self):
        """返回数值数据。

        返回
        =======

        如果使用 `adaptive=True` 进行评估，则返回：

        interval_list : list
            用于后处理并最终与Matplotlib的``fill``命令一起使用的边界矩形间隔列表。
        dummy : str
            包含``"fill"``的字符串。

        否则，返回2D numpy数组，用于与Matplotlib的``contour``或``contourf``命令一起使用：

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            指定使用的绘图命令的字符串，``"contour"``或``"contourf"``。
        """
        if self.adaptive:
            # 如果启用自适应模式，则获取自适应评估的数据
            data = self._adaptive_eval()
            if data is not None:
                return data

        # 否则返回网格的数据
        return self._get_meshes_grid()
    def _adaptive_eval(self):
        """
        References
        ==========

        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
        Mathematical Formulae with Two Free Variables.

        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
        Arithmetic. Master's thesis. University of Toronto, 1996
        """
        import sympy.plotting.intervalmath.lib_interval as li  # 导入 intervalmath 库中的 lib_interval 模块

        user_functions = {}  # 初始化用户定义的函数字典
        printer = IntervalMathPrinter({  # 创建 IntervalMathPrinter 打印器对象
            'fully_qualified_modules': False, 'inline': True,
            'allow_unknown_functions': True,
            'user_functions': user_functions})

        keys = [t for t in dir(li) if ("__" not in t) and (t not in ["import_module", "interval"])]  # 获取 lib_interval 模块中的非私有函数名列表
        vals = [getattr(li, k) for k in keys]  # 获取 lib_interval 模块中各函数的实际对象
        d = dict(zip(keys, vals))  # 将函数名与函数对象创建为字典
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)  # 使用 sympy lambdify 函数创建可调用的函数对象
        data = None  # 初始化数据变量为 None

        try:
            data = self._get_raster_interval(func)  # 尝试获取通过 func 计算的数据
        except NameError as err:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, as some functions are not yet implemented"
                " in the interval math module:\n\n"
                "NameError: %s\n\n" % err +
                "Proceeding with uniform meshing."
                )
            self.adaptive = False  # 若出现 NameError，则设置 adaptive 属性为 False
        except TypeError:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression. Using uniform meshing.")
            self.adaptive = False  # 若出现 TypeError，则设置 adaptive 属性为 False

        return data  # 返回计算得到的数据

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        np = import_module('numpy')  # 导入 numpy 模块

        xarray, yarray, z_grid = self._evaluate()  # 调用 _evaluate 方法获取 xarray、yarray 和 z_grid 数据
        _re, _im = np.real(z_grid), np.imag(z_grid)  # 计算 z_grid 的实部和虚部
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan  # 将不是接近零的虚部位置设置为 NaN
        if self._is_equality:
            return xarray, yarray, _re, 'contour'  # 如果是等式，则返回 xarray、yarray、_re 和 'contour'
        return xarray, yarray, _re, 'contourf'  # 如果不是等式，则返回 xarray、yarray、_re 和 'contourf'
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Whether the original expression was an Equality or not.
        """
        # 检查表达式是否为 Equality 类型，如果是，将其重写为左右两侧的差值
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        # 如果表达式是 Relational 类型，将其重写为左右两侧的差值
        elif isinstance(expr, Relational):
            expr = expr.gts - expr.lts
        # 如果不允许自适应，则抛出 NotImplementedError 异常
        elif not adaptive:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        # 返回重写后的表达式和是否为 Equality 的标志
        return expr, equality

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        # 如果 use_latex 参数为 False，直接返回内部保存的字符串标签
        if use_latex is False:
            return self._label
        # 如果标签与自适应表达式的字符串表示相同，返回经过包装的 LaTeX 标签
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        # 否则，返回 LaTeX 标签
        return self._latex_label
##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

# 计算线段中心点的函数
def centers_of_segments(array):
    # 导入numpy模块
    np = import_module('numpy')
    # 计算相邻点的均值，即线段的中心点
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


# 计算网格面中心点的函数
def centers_of_faces(array):
    # 导入numpy模块
    np = import_module('numpy')
    # 计算四个相邻点的均值，即网格面的中心点
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


# 检查三个点是否几乎共线的函数
def flat(x, y, z, eps=1e-3):
    # 导入numpy模块
    np = import_module('numpy')
    # 计算向量a和向量b，然后计算它们的点积和余弦值，判断是否几乎共线
    vector_a = (x - y).astype(float)
    vector_b = (z - y).astype(float)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


# 设置离散化点数的函数，兼容老版本的关键字参数
def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments ``n, n1, n2`` to
    specify the number of discretization points in one and two
    directions, while keeping back-compatibility with older keyword arguments
    like, ``nb_of_points, nb_of_points_*, points``.

    Parameters
    ==========

    kwargs : dict
        Dictionary of keyword arguments passed into a plotting function.
    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.
    """
    # 替换旧版本的关键字参数为新版本的关键字参数
    replace_old_keywords = {
        "nb_of_points": "n",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in replace_old_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)

    # 根据不同类型的图表对象调整关键字参数
    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
              Parametric3DLineSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n1"] = kwargs["n"][0]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries,
                ParametricSurfaceSeries, ImplicitSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    return kwargs
```