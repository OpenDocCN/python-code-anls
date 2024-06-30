# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_mode.py`

```
from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence


class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """

    ## Class-level attributes
    ## used to register and lookup
    ## plot modes. See PlotModeBase
    ## for descriptions and usage.

    i_vars, d_vars = '', ''
    intervals = []
    aliases = []
    is_default = False

    ## Draw is the only method here which
    ## is meant to be overridden in child
    ## classes, and PlotModeBase provides
    ## a base implementation.
    def draw(self):
        raise NotImplementedError()

    ## Everything else in this file has to
    ## do with registration and retrieval
    ## of plot modes. This is where I've
    ## hidden much of the ugliness of automatic
    ## plot mode divination...

    ## Plot mode registry data structures
    _mode_alias_list = []
    _mode_map = {
        1: {1: {}, 2: {}},
        2: {1: {}, 2: {}},
        3: {1: {}, 2: {}},
    }  # [d][i][alias_str]: class
    _mode_default_map = {
        1: {},
        2: {},
        3: {},
    }  # [d][i]: class
    _i_var_max, _d_var_max = 2, 3

    def __new__(cls, *args, **kwargs):
        """
        This is the function which interprets
        arguments given to Plot.__init__ and
        Plot.__setattr__. Returns an initialized
        instance of the appropriate child class.
        """

        newargs, newkwargs = PlotMode._extract_options(args, kwargs)
        mode_arg = newkwargs.get('mode', '')

        # Interpret the arguments
        d_vars, intervals = PlotMode._interpret_args(newargs)
        i_vars = PlotMode._find_i_vars(d_vars, intervals)
        i, d = max([len(i_vars), len(intervals)]), len(d_vars)

        # Find the appropriate mode
        subcls = PlotMode._get_mode(mode_arg, i, d)

        # Create the object
        o = object.__new__(subcls)

        # Do some setup for the mode instance
        o.d_vars = d_vars
        o._fill_i_vars(i_vars)
        o._fill_intervals(intervals)
        o.options = newkwargs

        return o

    @staticmethod
    def _extract_options(args, kwargs):
        """
        Extracts options from arguments
        passed to Plot.__init__ or
        Plot.__setattr__.
        """
        pass  # Implementation not shown

    @staticmethod
    def _interpret_args(args):
        """
        Interprets arguments passed
        to determine d_vars and intervals.
        """
        pass  # Implementation not shown

    @staticmethod
    def _find_i_vars(d_vars, intervals):
        """
        Determines i_vars based on
        d_vars and intervals.
        """
        pass  # Implementation not shown

    @classmethod
    def _get_mode(cls, mode_arg, i, d):
        """
        Determines the appropriate
        mode subclass based on mode_arg,
        i, and d.
        """
        pass  # Implementation not shown
    def _get_default_mode(i, d, i_vars=-1):
        # 如果未指定 i_vars，则默认与 i 相同
        if i_vars == -1:
            i_vars = i
        try:
            # 尝试从预设的 _mode_default_map 中获取默认的绘图模式
            return PlotMode._mode_default_map[d][i]
        except KeyError:
            # 如果未找到指定的模式，尝试在更高的 i_vars 计数中寻找支持给定 d 变量计数的模式，直到达到最大的 i_var 计数
            if i < PlotMode._i_var_max:
                return PlotMode._get_default_mode(i + 1, d, i_vars)
            else:
                # 如果超过最大 i_var 计数仍未找到，默认抛出数值错误异常
                raise ValueError(("Couldn't find a default mode "
                                  "for %i independent and %i "
                                  "dependent variables.") % (i_vars, d))

    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars=-1):
        # 如果未指定 i_vars，则默认与 i 相同
        if i_vars == -1:
            i_vars = i
        # 检查别名是否在已知的 _mode_alias_list 中
        if alias not in PlotMode._mode_alias_list:
            raise ValueError(("Couldn't find a mode called"
                              " %s. Known modes: %s.")
                             % (alias, ", ".join(PlotMode._mode_alias_list)))
        try:
            # 尝试从 _mode_map 中获取特定别名的绘图模式
            return PlotMode._mode_map[d][i][alias]
        except TypeError:
            # 如果未找到指定的模式，尝试在更高的 i_vars 计数中寻找支持给定 d 变量计数和别名的模式，直到达到最大的 i_var 计数
            if i < PlotMode._i_var_max:
                return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
            else:
                # 如果超过最大 i_var 计数仍未找到，默认抛出数值错误异常
                raise ValueError(("Couldn't find a %s mode "
                                  "for %i independent and %i "
                                  "dependent variables.")
                                 % (alias, i_vars, d))

    @classmethod
    def _register(cls):
        """
        Called once for each user-usable plot mode.
        For Cartesian2D, it is invoked after the
        class definition: Cartesian2D._register()
        """
        # 获取类名
        name = cls.__name__
        # 初始化模式
        cls._init_mode()

        try:
            # 获取 i_var_count 和 d_var_count
            i, d = cls.i_var_count, cls.d_var_count
            # 将模式添加到 _mode_map 下的所有给定别名中
            for a in cls.aliases:
                if a not in PlotMode._mode_alias_list:
                    # 追踪有效的别名，以便在 _get_mode 中快速识别无效的别名
                    PlotMode._mode_alias_list.append(a)
                PlotMode._mode_map[d][i][a] = cls
            if cls.is_default:
                # 如果此模式标记为该 d,i 组合的默认模式，则设置为默认模式
                PlotMode._mode_default_map[d][i] = cls

        except Exception as e:
            # 如果注册模式失败，抛出运行时异常
            raise RuntimeError(("Failed to register "
                              "plot mode %s. Reason: %s")
                               % (name, (str(e))))

    @classmethod
    def _init_mode(cls):
        """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """
        def symbols_list(symbol_str):
            return [Symbol(s) for s in symbol_str]

        # Convert the vars strs into
        # lists of symbols.
        cls.i_vars = symbols_list(cls.i_vars)
        cls.d_vars = symbols_list(cls.d_vars)

        # Var count is used often, calculate
        # it once here
        cls.i_var_count = len(cls.i_vars)
        cls.d_var_count = len(cls.d_vars)

        if cls.i_var_count > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, False))
        if cls.d_var_count > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, False))

        # Try to use first alias as primary_alias
        if len(cls.aliases) > 0:
            cls.primary_alias = cls.aliases[0]
        else:
            cls.primary_alias = cls.__name__

        di = cls.intervals
        if len(di) != cls.i_var_count:
            raise ValueError("Plot mode must provide a "
                             "default interval for each i_var.")
        for i in range(cls.i_var_count):
            # default intervals must be given [min,max,steps]
            # (no var, but they must be in the same order as i_vars)
            if len(di[i]) != 3:
                raise ValueError("length should be equal to 3")

            # Initialize an incomplete interval,
            # to later be filled with a var when
            # the mode is instantiated.
            di[i] = PlotInterval(None, *di[i])

        # To prevent people from using modes
        # without these required fields set up.
        cls._was_initialized = True

    _was_initialized = False

    ## Initializer Helper Methods

    @staticmethod
    def _find_i_vars(functions, intervals):
        """
        Find and return all independent variables (i_vars)
        based on given functions and intervals.
        """
        i_vars = []

        # First, collect i_vars in the
        # order they are given in any
        # intervals.
        for i in intervals:
            if i.v is None:
                continue
            elif i.v in i_vars:
                raise ValueError(("Multiple intervals given "
                                  "for %s.") % (str(i.v)))
            i_vars.append(i.v)

        # Then, find any remaining
        # i_vars in given functions
        # (aka d_vars)
        for f in functions:
            for a in f.free_symbols:
                if a not in i_vars:
                    i_vars.append(a)

        return i_vars

    def _fill_i_vars(self, i_vars):
        """
        Fill the instance's i_vars with the provided list.
        """
        # copy default i_vars
        self.i_vars = [Symbol(str(i)) for i in self.i_vars]
        # replace with given i_vars
        for i in range(len(i_vars)):
            self.i_vars[i] = i_vars[i]
    # 复制默认的区间列表，创建新的 PlotInterval 实例列表
    self.intervals = [PlotInterval(i) for i in self.intervals]
    
    # 用于跟踪已使用的 i_vars 列表
    v_used = []
    
    # 使用给定的 intervals 填充复制的默认 intervals
    for i in range(len(intervals)):
        # 从 intervals 中的每个元素填充到对应的 self.intervals 元素中
        self.intervals[i].fill_from(intervals[i])
        
        # 如果 self.intervals[i].v 不为空，则将其添加到 v_used 中
        if self.intervals[i].v is not None:
            v_used.append(self.intervals[i].v)
    
    # 查找任何孤立的 intervals，并为其分配 i_vars
    for i in range(len(self.intervals)):
        # 如果 self.intervals[i].v 为空
        if self.intervals[i].v is None:
            # 找到 self.i_vars 中未在 v_used 中使用的第一个值
            u = [v for v in self.i_vars if v not in v_used]
            
            # 如果找不到可用的 i_vars，则抛出 ValueError 异常
            if len(u) == 0:
                raise ValueError("length should not be equal to 0")
            
            # 将找到的第一个未使用的 i_vars 赋值给 self.intervals[i].v
            self.intervals[i].v = u[0]
            # 将已分配的 i_vars 添加到 v_used 中
            v_used.append(u[0])

@staticmethod
def _interpret_args(args):
    # 错误消息：如果 PlotInterval 被放在任何函数之前
    interval_wrong_order = "PlotInterval %s was given before any function(s)."
    # 错误消息：无法解释参数作为函数或区间
    interpret_error = "Could not interpret %s as a function or interval."

    functions, intervals = [], []
    
    # 如果 args[0] 是 GeometryEntity 类型
    if isinstance(args[0], GeometryEntity):
        # 将 args[0] 的任意点坐标添加到 functions 中
        for coords in list(args[0].arbitrary_point()):
            functions.append(coords)
        # 尝试解析 args[0] 的绘图区间并添加到 intervals 中
        intervals.append(PlotInterval.try_parse(args[0].plot_interval()))
    else:
        # 遍历 args 中的每个元素
        for a in args:
            # 尝试解析 a 为 PlotInterval 对象
            i = PlotInterval.try_parse(a)
            
            # 如果成功解析为 PlotInterval 对象
            if i is not None:
                # 如果 functions 列表为空，则抛出异常，因为区间应在函数之前
                if len(functions) == 0:
                    raise ValueError(interval_wrong_order % (str(i)))
                else:
                    # 否则将解析出的 PlotInterval 对象添加到 intervals 中
                    intervals.append(i)
            else:
                # 如果 a 是可迭代对象（不包括字符串），则抛出解释错误异常
                if is_sequence(a, include=str):
                    raise ValueError(interpret_error % (str(a)))
                
                # 尝试将 a 解析为符号表达式 f，并添加到 functions 列表中
                try:
                    f = sympify(a)
                    functions.append(f)
                except TypeError:
                    raise ValueError(interpret_error % str(a))

    # 返回解释出的 functions 和 intervals 列表
    return functions, intervals

@staticmethod
def _extract_options(args, kwargs):
    # 解析 args 中的字符串，并将其添加到 newkwargs 中
    newkwargs, newargs = {}, []
    for a in args:
        if isinstance(a, str):
            newkwargs = dict(newkwargs, **parse_option_string(a))
        else:
            # 如果不是字符串，则添加到 newargs 列表中
            newargs.append(a)
    
    # 将 kwargs 中的内容合并到 newkwargs 中
    newkwargs = dict(newkwargs, **kwargs)
    
    # 返回更新后的 newargs 和 newkwargs
    return newargs, newkwargs
# 定义一个函数，用于生成格式化的错误消息，根据两个布尔参数不同的值稍有不同
def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """
    # 如果 is_plotting 为 True，则错误消息中的操作为 "Plotting"
    if is_plotting:
        v = "Plotting"
    else:
        # 否则，错误消息中的操作为 "Registering plot modes"
        v = "Registering plot modes"
    
    # 根据 is_independent 的值选择相应的最大变量数和描述词
    if is_independent:
        # 如果 is_independent 为 True，则使用独立变量的最大数目和 "independent"
        n, s = PlotMode._i_var_max, "independent"
    else:
        # 否则，使用依赖变量的最大数目和 "dependent"
        n, s = PlotMode._d_var_max, "dependent"
    
    # 返回格式化后的错误消息，包括操作名称、最大变量数和描述词
    return ("%s with more than %i %s variables "
            "is not supported.") % (v, n, s)
```