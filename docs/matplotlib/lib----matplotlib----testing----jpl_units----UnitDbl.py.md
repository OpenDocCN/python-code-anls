# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\UnitDbl.py`

```
"""UnitDbl module."""

import functools
import operator

from matplotlib import _api


class UnitDbl:
    """Class UnitDbl in development."""

    # Unit conversion table.  Small subset of the full one but enough
    # to test the required functions.  First field is a scale factor to
    # convert the input units to the units of the second field.  Only
    # units in this table are allowed.
    allowed = {
        "m": (0.001, "km"),   # 将米转换为千米的比例因子
        "km": (1, "km"),      # 千米到千米的比例因子（基准）
        "mile": (1.609344, "km"),  # 英里到千米的比例因子

        "rad": (1, "rad"),    # 弧度到弧度的比例因子（基准）
        "deg": (1.745329251994330e-02, "rad"),  # 度到弧度的比例因子

        "sec": (1, "sec"),    # 秒到秒的比例因子（基准）
        "min": (60.0, "sec"),  # 分钟到秒的比例因子
        "hour": (3600, "sec"),  # 小时到秒的比例因子
    }

    _types = {
        "km": "distance",     # 千米单位对应的类型是距离
        "rad": "angle",       # 弧度单位对应的类型是角度
        "sec": "time",        # 秒单位对应的类型是时间
    }

    def __init__(self, value, units):
        """
        Create a new UnitDbl object.

        Units are internally converted to km, rad, and sec.  The only
        valid inputs for units are [m, km, mile, rad, deg, sec, min, hour].

        The field UnitDbl.value will contain the converted value.  Use
        the convert() method to get a specific type of units back.

        = ERROR CONDITIONS
        - If the input units are not in the allowed list, an error is thrown.

        = INPUT VARIABLES
        - value     The numeric value of the UnitDbl.
        - units     The string name of the units the value is in.
        """
        # 获取单位对应的转换数据
        data = _api.check_getitem(self.allowed, units=units)
        # 根据转换数据将输入值转换为内部统一的单位值
        self._value = float(value * data[0])
        self._units = data[1]

    def convert(self, units):
        """
        Convert the UnitDbl to a specific set of units.

        = ERROR CONDITIONS
        - If the input units are not in the allowed list, an error is thrown.

        = INPUT VARIABLES
        - units     The string name of the units to convert to.

        = RETURN VALUE
        - Returns the value of the UnitDbl in the requested units as a floating
          point number.
        """
        # 如果当前单位和目标单位相同，直接返回当前值
        if self._units == units:
            return self._value
        # 获取目标单位对应的转换数据
        data = _api.check_getitem(self.allowed, units=units)
        # 检查是否能直接转换到目标单位，若不能则抛出异常
        if self._units != data[1]:
            raise ValueError(f"Error trying to convert to different units.\n"
                             f"    Invalid conversion requested.\n"
                             f"    UnitDbl: {self}\n"
                             f"    Units:   {units}\n")
        # 根据转换数据将当前值转换为目标单位的值并返回
        return self._value / data[0]

    def __abs__(self):
        """Return the absolute value of this UnitDbl."""
        # 返回当前值的绝对值对应的 UnitDbl 对象
        return UnitDbl(abs(self._value), self._units)

    def __neg__(self):
        """Return the negative value of this UnitDbl."""
        # 返回当前值的负数对应的 UnitDbl 对象
        return UnitDbl(-self._value, self._units)

    def __bool__(self):
        """Return the truth value of a UnitDbl."""
        # 返回当前值的布尔值
        return bool(self._value)
    def _cmp(self, op, rhs):
        """Check that *self* and *rhs* share units; compare them using *op*."""
        # 检查当前对象和右操作数是否具有相同的单位，用于比较它们的值
        self.checkSameUnits(rhs, "compare")
        # 返回使用指定操作符 op 对当前对象和右操作数值进行比较的结果
        return op(self._value, rhs._value)

    __eq__ = functools.partialmethod(_cmp, operator.eq)
    __ne__ = functools.partialmethod(_cmp, operator.ne)
    __lt__ = functools.partialmethod(_cmp, operator.lt)
    __le__ = functools.partialmethod(_cmp, operator.le)
    __gt__ = functools.partialmethod(_cmp, operator.gt)
    __ge__ = functools.partialmethod(_cmp, operator.ge)

    def _binop_unit_unit(self, op, rhs):
        """Check that *self* and *rhs* share units; combine them using *op*."""
        # 检查当前对象和右操作数是否具有相同的单位，用于使用指定操作符 op 合并它们的值
        self.checkSameUnits(rhs, op.__name__)
        # 返回一个新的 UnitDbl 对象，其值为当前对象值和右操作数值使用 op 操作后的结果，并且单位与当前对象相同
        return UnitDbl(op(self._value, rhs._value), self._units)

    __add__ = functools.partialmethod(_binop_unit_unit, operator.add)
    __sub__ = functools.partialmethod(_binop_unit_unit, operator.sub)

    def _binop_unit_scalar(self, op, scalar):
        """Combine *self* and *scalar* using *op*."""
        # 使用指定操作符 op 结合当前对象和标量值 scalar
        return UnitDbl(op(self._value, scalar), self._units)

    __mul__ = functools.partialmethod(_binop_unit_scalar, operator.mul)
    __rmul__ = functools.partialmethod(_binop_unit_scalar, operator.mul)

    def __str__(self):
        """Print the UnitDbl."""
        # 返回当前 UnitDbl 对象的可打印表示形式
        return f"{self._value:g} *{self._units}"

    def __repr__(self):
        """Print the UnitDbl."""
        # 返回当前 UnitDbl 对象的表达式形式
        return f"UnitDbl({self._value:g}, '{self._units}')"

    def type(self):
        """Return the type of UnitDbl data."""
        # 返回当前 UnitDbl 对象的数据类型
        return self._types[self._units]

    @staticmethod
    def range(start, stop, step=None):
        """
        Generate a range of UnitDbl objects.

        Similar to the Python range() method.  Returns the range [
        start, stop) at the requested step.  Each element will be a
        UnitDbl object.

        = INPUT VARIABLES
        - start     The starting value of the range.
        - stop      The stop value of the range.
        - step      Optional step to use.  If set to None, then a UnitDbl of
                      value 1 w/ the units of the start is used.

        = RETURN VALUE
        - Returns a list containing the requested UnitDbl values.
        """
        # 如果未提供步长 step，则使用与起始值 start 相同单位的 UnitDbl 对象作为步长
        if step is None:
            step = UnitDbl(1, start._units)

        elems = []

        i = 0
        while True:
            d = start + i * step
            # 如果当前值 d 大于或等于停止值 stop，则退出循环
            if d >= stop:
                break

            elems.append(d)
            i += 1

        # 返回包含所请求的 UnitDbl 值的列表
        return elems
    # 检查两个 UnitDbl 对象的单位是否相同。
    def checkSameUnits(self, rhs, func):
        """
        Check to see if units are the same.

        = ERROR CONDITIONS
        - If the units of the rhs UnitDbl are not the same as our units,
          an error is thrown.

        = INPUT VARIABLES
        - rhs     The UnitDbl to check for the same units
        - func    The name of the function doing the check.
        """
        # 如果当前对象 self 的单位与 rhs 对象的单位不同，抛出 ValueError 异常。
        if self._units != rhs._units:
            raise ValueError(f"Cannot {func} units of different types.\n"
                             f"LHS: {self._units}\n"
                             f"RHS: {rhs._units}")
```