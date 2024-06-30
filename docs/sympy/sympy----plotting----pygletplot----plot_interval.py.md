# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_interval.py`

```
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer

class PlotInterval:
    """
    Represents a plot interval with optional parameters: v, v_min, v_max, v_steps.
    """

    # Initialize instance variables to None
    _v, _v_min, _v_max, _v_steps = None, None, None, None

    # Decorator function to require all arguments are initialized
    def require_all_args(f):
        """
        Decorator function to check if all required attributes (_v, _v_min, _v_max, _v_steps) are set.
        Raises ValueError if any attribute is None.
        """
        def check(self, *args, **kwargs):
            for g in [self._v, self._v_min, self._v_max, self._v_steps]:
                if g is None:
                    raise ValueError("PlotInterval is incomplete.")
            return f(self, *args, **kwargs)
        return check

    def __init__(self, *args):
        """
        Initializes a PlotInterval object based on the given arguments.

        Args:
        - If one argument:
            - If it's a PlotInterval object, initializes from it.
            - If it's a string, attempts to evaluate it.
            - If it's a tuple or list, uses its elements as arguments.
        - If more than one argument or the argument is not a tuple, list, or string, raises ValueError.
        """
        if len(args) == 1:
            if isinstance(args[0], PlotInterval):
                self.fill_from(args[0])
                return
            elif isinstance(args[0], str):
                try:
                    args = eval(args[0])
                except TypeError:
                    s_eval_error = "Could not interpret string %s."
                    raise ValueError(s_eval_error % (args[0]))
            elif isinstance(args[0], (tuple, list)):
                args = args[0]
            else:
                raise ValueError("Not an interval.")
        
        if not isinstance(args, (tuple, list)) or len(args) > 4:
            f_error = "PlotInterval must be a tuple or list of length 4 or less."
            raise ValueError(f_error)

        args = list(args)
        if len(args) > 0 and (args[0] is None or isinstance(args[0], Symbol)):
            self.v = args.pop(0)
        if len(args) in [2, 3]:
            self.v_min = args.pop(0)
            self.v_max = args.pop(0)
            if len(args) == 1:
                self.v_steps = args.pop(0)
        elif len(args) == 1:
            self.v_steps = args.pop(0)

    def get_v(self):
        """
        Returns the value of v.
        """
        return self._v

    def set_v(self, v):
        """
        Sets the value of v.
        
        Args:
        - v: A SymPy Symbol object.
        
        Raises:
        - ValueError: If v is not a SymPy Symbol.
        """
        if v is None:
            self._v = None
            return
        if not isinstance(v, Symbol):
            raise ValueError("v must be a SymPy Symbol.")
        self._v = v

    def get_v_min(self):
        """
        Returns the minimum value of v (v_min).
        """
        return self._v_min

    def set_v_min(self, v_min):
        """
        Sets the minimum value of v (v_min).
        
        Args:
        - v_min: The value to set as v_min.
        
        Raises:
        - ValueError: If v_min cannot be interpreted as a number.
        """
        if v_min is None:
            self._v_min = None
            return
        try:
            self._v_min = sympify(v_min)
            float(self._v_min.evalf())
        except TypeError:
            raise ValueError("v_min could not be interpreted as a number.")

    def get_v_max(self):
        """
        Returns the maximum value of v (v_max).
        """
        return self._v_max

    def set_v_max(self, v_max):
        """
        Sets the maximum value of v (v_max).
        
        Args:
        - v_max: The value to set as v_max.
        
        Raises:
        - ValueError: If v_max cannot be interpreted as a number.
        """
        if v_max is None:
            self._v_max = None
            return
        try:
            self._v_max = sympify(v_max)
            float(self._v_max.evalf())
        except TypeError:
            raise ValueError("v_max could not be interpreted as a number.")

    def get_v_steps(self):
        """
        Returns the number of steps for v (v_steps).
        """
        return self._v_steps
    def set_v_steps(self, v_steps):
        # 设置 v_steps 属性的方法
        if v_steps is None:
            # 如果 v_steps 是 None，则将 _v_steps 属性设为 None 并返回
            self._v_steps = None
            return
        if isinstance(v_steps, int):
            # 如果 v_steps 是整数，则转换为 SymPy 的 Integer 类型
            v_steps = Integer(v_steps)
        elif not isinstance(v_steps, Integer):
            # 如果 v_steps 不是整数或 SymPy 的 Integer 类型，则抛出数值错误异常
            raise ValueError("v_steps must be an int or SymPy Integer.")
        if v_steps <= S.Zero:
            # 如果 v_steps 小于等于零，则抛出数值错误异常
            raise ValueError("v_steps must be positive.")
        # 设置 _v_steps 属性为 v_steps
        self._v_steps = v_steps

    @require_all_args
    def get_v_len(self):
        # 获取 v_len 属性的方法，返回 v_steps + 1
        return self.v_steps + 1

    v = property(get_v, set_v)
    v_min = property(get_v_min, set_v_min)
    v_max = property(get_v_max, set_v_max)
    v_steps = property(get_v_steps, set_v_steps)
    v_len = property(get_v_len)

    def fill_from(self, b):
        # 从另一个对象 b 填充当前对象的属性
        if b.v is not None:
            # 如果 b 的 v 属性不是 None，则设置当前对象的 v 属性为 b 的 v 属性
            self.v = b.v
        if b.v_min is not None:
            # 如果 b 的 v_min 属性不是 None，则设置当前对象的 v_min 属性为 b 的 v_min 属性
            self.v_min = b.v_min
        if b.v_max is not None:
            # 如果 b 的 v_max 属性不是 None，则设置当前对象的 v_max 属性为 b 的 v_max 属性
            self.v_max = b.v_max
        if b.v_steps is not None:
            # 如果 b 的 v_steps 属性不是 None，则设置当前对象的 v_steps 属性为 b 的 v_steps 属性
            self.v_steps = b.v_steps

    @staticmethod
    def try_parse(*args):
        """
        Returns a PlotInterval if args can be interpreted
        as such, otherwise None.
        """
        # 尝试解析参数 args 成为 PlotInterval 对象，如果成功则返回该对象，否则返回 None
        if len(args) == 1 and isinstance(args[0], PlotInterval):
            return args[0]
        try:
            return PlotInterval(*args)
        except ValueError:
            return None

    def _str_base(self):
        # 返回一个由 v, v_min, v_max, v_steps 组成的字符串，用逗号分隔
        return ",".join([str(self.v), str(self.v_min),
                         str(self.v_max), str(self.v_steps)])

    def __repr__(self):
        """
        A string representing the interval in class constructor form.
        """
        # 返回表示该对象的字符串，形式为类构造函数的形式
        return "PlotInterval(%s)" % (self._str_base())

    def __str__(self):
        """
        A string representing the interval in list form.
        """
        # 返回表示该对象的字符串，形式为列表的形式
        return "[%s]" % (self._str_base())

    @require_all_args
    def assert_complete(self):
        # 确保所有参数都被设置的方法，不做任何操作
        pass

    @require_all_args
    def vrange(self):
        """
        Yields v_steps+1 SymPy numbers ranging from
        v_min to v_max.
        """
        # 生成 v_steps+1 个 SymPy 数字，范围从 v_min 到 v_max
        d = (self.v_max - self.v_min) / self.v_steps
        for i in range(self.v_steps + 1):
            a = self.v_min + (d * Integer(i))
            yield a

    @require_all_args
    def vrange2(self):
        """
        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
        # 生成 v_steps 对 SymPy 数字的生成器，范围从 (v_min, v_min + step) 到 (v_max - step, v_max)
        d = (self.v_max - self.v_min) / self.v_steps
        a = self.v_min + (d * S.Zero)
        for i in range(self.v_steps):
            b = self.v_min + (d * Integer(i + 1))
            yield a, b
            a = b

    def frange(self):
        # 生成浮点数的生成器，从 vrange() 生成的 SymPy 数字中求其数值
        for i in self.vrange():
            yield float(i.evalf())
```