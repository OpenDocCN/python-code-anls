# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\Duration.py`

```py
"""Duration module."""

import functools  # 导入 functools 模块，用于函数操作的工具
import operator  # 导入 operator 模块，用于进行运算符操作

from matplotlib import _api  # 从 matplotlib 中导入 _api 模块

class Duration:
    """Class Duration in development."""

    allowed = ["ET", "UTC"]  # 允许的帧类型列表，包括 'ET' 和 'UTC'

    def __init__(self, frame, seconds):
        """
        Create a new Duration object.

        = ERROR CONDITIONS
        - If the input frame is not in the allowed list, an error is thrown.

        = INPUT VARIABLES
        - frame     The frame of the duration.  Must be 'ET' or 'UTC'
        - seconds   The number of seconds in the Duration.
        """
        _api.check_in_list(self.allowed, frame=frame)  # 检查输入的帧类型是否在允许的列表中
        self._frame = frame  # 设置对象的帧类型
        self._seconds = seconds  # 设置对象的秒数

    def frame(self):
        """Return the frame the duration is in."""
        return self._frame  # 返回对象所属的帧类型

    def __abs__(self):
        """Return the absolute value of the duration."""
        return Duration(self._frame, abs(self._seconds))  # 返回该持续时间对象的绝对值

    def __neg__(self):
        """Return the negative value of this Duration."""
        return Duration(self._frame, -self._seconds)  # 返回该持续时间对象的负值

    def seconds(self):
        """Return the number of seconds in the Duration."""
        return self._seconds  # 返回该持续时间对象的秒数

    def __bool__(self):
        """Check if the Duration is non-zero."""
        return self._seconds != 0  # 判断该持续时间对象是否非零

    def _cmp(self, op, rhs):
        """
        Check that *self* and *rhs* share frames; compare them using *op*.
        """
        self.checkSameFrame(rhs, "compare")  # 检查自身和rhs是否共享相同的帧类型
        return op(self._seconds, rhs._seconds)  # 使用给定的操作符op比较self和rhs的秒数值

    __eq__ = functools.partialmethod(_cmp, operator.eq)  # 定义相等操作符的特定方法
    __ne__ = functools.partialmethod(_cmp, operator.ne)  # 定义不等操作符的特定方法
    __lt__ = functools.partialmethod(_cmp, operator.lt)  # 定义小于操作符的特定方法
    __le__ = functools.partialmethod(_cmp, operator.le)  # 定义小于等于操作符的特定方法
    __gt__ = functools.partialmethod(_cmp, operator.gt)  # 定义大于操作符的特定方法
    __ge__ = functools.partialmethod(_cmp, operator.ge)  # 定义大于等于操作符的特定方法

    def __add__(self, rhs):
        """
        Add two Durations.

        = ERROR CONDITIONS
        - If the input rhs is not in the same frame, an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to add.

        = RETURN VALUE
        - Returns the sum of ourselves and the input Duration.
        """
        # 延迟加载由于循环依赖
        import matplotlib.testing.jpl_units as U  # 导入 matplotlib.testing.jpl_units 模块 as U

        if isinstance(rhs, U.Epoch):  # 如果rhs是Epoch类型的对象
            return rhs + self  # 调用rhs对象的__add__方法，并将self作为参数

        self.checkSameFrame(rhs, "add")  # 检查自身和rhs是否共享相同的帧类型
        return Duration(self._frame, self._seconds + rhs._seconds)  # 返回两个持续时间对象的和

    def __sub__(self, rhs):
        """
        Subtract two Durations.

        = ERROR CONDITIONS
        - If the input rhs is not in the same frame, an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to subtract.

        = RETURN VALUE
        - Returns the difference of ourselves and the input Duration.
        """
        self.checkSameFrame(rhs, "sub")  # 检查自身和rhs是否共享相同的帧类型
        return Duration(self._frame, self._seconds - rhs._seconds)  # 返回两个持续时间对象的差
    # 定义一个方法，用于实现乘法运算符重载，用于单位浮点数的缩放
    def __mul__(self, rhs):
        """
        Scale a UnitDbl by a value.

        = INPUT VARIABLES
        - rhs     The scalar to multiply by.

        = RETURN VALUE
        - Returns the scaled Duration.
        """
        # 返回一个新的 Duration 对象，其帧数与秒数乘以给定的标量值
        return Duration(self._frame, self._seconds * float(rhs))

    # 将 __mul__ 方法用于右乘运算符重载
    __rmul__ = __mul__

    # 定义一个方法，返回 Duration 对象的字符串表示形式
    def __str__(self):
        """Print the Duration."""
        return f"{self._seconds:g} {self._frame}"

    # 定义一个方法，返回 Duration 对象的可打印字符串表示形式
    def __repr__(self):
        """Print the Duration."""
        # 返回一个字符串，表示当前 Duration 对象的帧数和秒数
        return f"Duration('{self._frame}', {self._seconds:g})"

    # 定义一个方法，用于检查两个 Duration 对象的帧数是否相同
    def checkSameFrame(self, rhs, func):
        """
        Check to see if frames are the same.

        = ERROR CONDITIONS
        - If the frame of the rhs Duration is not the same as our frame,
          an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to check for the same frame
        - func    The name of the function doing the check.
        """
        # 如果当前 Duration 对象和 rhs 对象的帧数不相同，抛出 ValueError 异常
        if self._frame != rhs._frame:
            raise ValueError(
                f"Cannot {func} Durations with different frames.\n"
                f"LHS: {self._frame}\n"
                f"RHS: {rhs._frame}")
```