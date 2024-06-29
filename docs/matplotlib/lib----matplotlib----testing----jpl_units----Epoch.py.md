# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\Epoch.py`

```py
"""Epoch module."""

import functools
import operator
import math
import datetime as DT

from matplotlib import _api
from matplotlib.dates import date2num


class Epoch:
    # Frame conversion offsets in seconds
    # t(TO) = t(FROM) + allowed[ FROM ][ TO ]
    allowed = {
        "ET": {
            "UTC": +64.1839,
            },
        "UTC": {
            "ET": -64.1839,
            },
        }

    def __init__(self, frame, sec=None, jd=None, daynum=None, dt=None):
        """
        Create a new Epoch object.

        Build an epoch 1 of 2 ways:

        Using seconds past a Julian date:
        #   Epoch('ET', sec=1e8, jd=2451545)

        or using a matplotlib day number
        #   Epoch('ET', daynum=730119.5)

        = ERROR CONDITIONS
        - If the input units are not in the allowed list, an error is thrown.

        = INPUT VARIABLES
        - frame     The frame of the epoch.  Must be 'ET' or 'UTC'
        - sec        The number of seconds past the input JD.
        - jd         The Julian date of the epoch.
        - daynum    The matplotlib day number of the epoch.
        - dt         A python datetime instance.
        """
        if ((sec is None and jd is not None) or
                (sec is not None and jd is None) or
                (daynum is not None and
                 (sec is not None or jd is not None)) or
                (daynum is None and dt is None and
                 (sec is None or jd is None)) or
                (daynum is not None and dt is not None) or
                (dt is not None and (sec is not None or jd is not None)) or
                ((dt is not None) and not isinstance(dt, DT.datetime))):
            raise ValueError(
                "Invalid inputs.  Must enter sec and jd together, "
                "daynum by itself, or dt (must be a python datetime).\n"
                "Sec = %s\n"
                "JD  = %s\n"
                "dnum= %s\n"
                "dt  = %s" % (sec, jd, daynum, dt))

        # Check if the specified frame is in the allowed list
        _api.check_in_list(self.allowed, frame=frame)
        self._frame = frame

        if dt is not None:
            # Convert the given datetime instance to a matplotlib day number
            daynum = date2num(dt)

        if daynum is not None:
            # Calculate Julian date from matplotlib day number
            jd = float(daynum) + 1721425.5
            self._jd = math.floor(jd)
            self._seconds = (jd - self._jd) * 86400.0

        else:
            # Initialize using seconds past Julian date
            self._seconds = float(sec)
            self._jd = float(jd)

            # Resolve seconds down to [ 0, 86400)
            deltaDays = math.floor(self._seconds / 86400)
            self._jd += deltaDays
            self._seconds -= deltaDays * 86400.0

    def convert(self, frame):
        """
        Convert the current Epoch object to a different frame.

        = INPUT VARIABLES
        - frame     The frame to convert to. Must be 'ET' or 'UTC'.
        """
        if self._frame == frame:
            return self

        # Calculate offset to convert from current frame to the specified frame
        offset = self.allowed[self._frame][frame]

        # Return a new Epoch object converted to the specified frame
        return Epoch(frame, self._seconds + offset, self._jd)

    def frame(self):
        """
        Get the frame of the current Epoch object.

        = RETURN VALUE
        - The frame of the Epoch object ('ET' or 'UTC').
        """
        return self._frame
    def julianDate(self, frame):
        # 将当前对象赋值给 t
        t = self
        # 如果给定的 frame 不等于当前对象的帧类型，则需要进行坐标转换
        if frame != self._frame:
            t = self.convert(frame)

        # 返回当前对象的儒略日期加上秒数转换成的小数
        return t._jd + t._seconds / 86400.0

    def secondsPast(self, frame, jd):
        # 将当前对象赋值给 t
        t = self
        # 如果给定的 frame 不等于当前对象的帧类型，则需要进行坐标转换
        if frame != self._frame:
            t = self.convert(frame)

        # 计算当前对象与指定儒略日期之间的秒数差，并返回结果
        delta = t._jd - jd
        return t._seconds + delta * 86400

    def _cmp(self, op, rhs):
        """Compare Epochs *self* and *rhs* using operator *op*."""
        # 将当前对象赋值给 t
        t = self
        # 如果当前对象的帧类型与 rhs 对象的帧类型不同，则需要进行坐标转换
        if self._frame != rhs._frame:
            t = self.convert(rhs._frame)
        # 比较当前对象的儒略日期和秒数与 rhs 对象的对应值，根据操作符 op 返回比较结果
        if t._jd != rhs._jd:
            return op(t._jd, rhs._jd)
        return op(t._seconds, rhs._seconds)

    # 以下是特殊方法的重载，通过 functools.partialmethod 将 _cmp 方法与不同的比较操作符关联起来
    __eq__ = functools.partialmethod(_cmp, operator.eq)
    __ne__ = functools.partialmethod(_cmp, operator.ne)
    __lt__ = functools.partialmethod(_cmp, operator.lt)
    __le__ = functools.partialmethod(_cmp, operator.le)
    __gt__ = functools.partialmethod(_cmp, operator.gt)
    __ge__ = functools.partialmethod(_cmp, operator.ge)

    def __add__(self, rhs):
        """
        Add a duration to an Epoch.

        = INPUT VARIABLES
        - rhs     The Epoch to subtract.

        = RETURN VALUE
        - Returns the difference of ourselves and the input Epoch.
        """
        # 将当前对象赋值给 t
        t = self
        # 如果当前对象的帧类型与 rhs 对象的帧类型不同，则需要进行坐标转换
        if self._frame != rhs.frame():
            t = self.convert(rhs._frame)

        # 计算当前对象的秒数加上 rhs 对象的秒数，并返回一个新的 Epoch 对象
        sec = t._seconds + rhs.seconds()

        return Epoch(t._frame, sec, t._jd)

    def __sub__(self, rhs):
        """
        Subtract two Epoch's or a Duration from an Epoch.

        Valid:
        Duration = Epoch - Epoch
        Epoch = Epoch - Duration

        = INPUT VARIABLES
        - rhs     The Epoch to subtract.

        = RETURN VALUE
        - Returns either the duration between to Epoch's or the a new
          Epoch that is the result of subtracting a duration from an epoch.
        """
        # 延迟加载 matplotlib.testing.jpl_units 模块，以处理循环依赖
        import matplotlib.testing.jpl_units as U

        # 如果 rhs 是 Duration 类型的对象，则执行 Epoch - Duration 的减法操作
        if isinstance(rhs, U.Duration):
            return self + -rhs

        # 将当前对象赋值给 t
        t = self
        # 如果当前对象的帧类型与 rhs 对象的帧类型不同，则需要进行坐标转换
        if self._frame != rhs._frame:
            t = self.convert(rhs._frame)

        # 计算当前对象的儒略日期和秒数与 rhs 对象的对应值的差值，返回一个新的 Duration 对象
        days = t._jd - rhs._jd
        sec = t._seconds - rhs._seconds

        return U.Duration(rhs._frame, days*86400 + sec)

    def __str__(self):
        """Print the Epoch."""
        # 返回当前 Epoch 对象的儒略日期和帧类型的字符串表示
        return f"{self.julianDate(self._frame):22.15e} {self._frame}"

    def __repr__(self):
        """Print the Epoch."""
        # 返回当前 Epoch 对象的字符串表示，调用 __str__ 方法
        return str(self)
    # 定义一个生成 Epoch 对象范围的函数
    def range(start, stop, step):
        """
        Generate a range of Epoch objects.

        Similar to the Python range() method.  Returns the range [
        start, stop) at the requested step.  Each element will be a
        Epoch object.

        = INPUT VARIABLES
        - start     The starting value of the range.
        - stop      The stop value of the range.
        - step      Step to use.

        = RETURN VALUE
        - Returns a list containing the requested Epoch values.
        """
        # 初始化空列表，用于存放生成的 Epoch 对象
        elems = []

        # 初始化计数器 i
        i = 0
        # 无限循环，生成指定范围内的 Epoch 对象
        while True:
            # 计算当前 Epoch 值
            d = start + i * step
            # 如果当前 Epoch 值超过或等于 stop，跳出循环
            if d >= stop:
                break

            # 将当前 Epoch 值加入结果列表
            elems.append(d)
            # 更新计数器
            i += 1

        # 返回生成的 Epoch 对象列表
        return elems
```