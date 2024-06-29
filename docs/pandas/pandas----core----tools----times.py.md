# `D:\src\scipysrc\pandas\pandas\core\tools\times.py`

```
# 从未来版本导入注解类型支持
from __future__ import annotations

# 导入 datetime 模块中的 datetime 和 time 类
from datetime import (
    datetime,
    time,
)

# 导入类型提示中的 TYPE_CHECKING
from typing import TYPE_CHECKING

# 导入第三方库 numpy，并用 np 作为别名
import numpy as np

# 导入 pandas 内部的 is_list_like 函数
from pandas._libs.lib import is_list_like

# 导入 pandas 核心数据类型模块中的 ABCIndex 和 ABCSeries 类
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

# 导入 pandas 核心数据类型模块中的 notna 函数
from pandas.core.dtypes.missing import notna

# 如果 TYPE_CHECKING 为真，则从 pandas._typing 模块导入 DateTimeErrorChoices 类型
if TYPE_CHECKING:
    from pandas._typing import DateTimeErrorChoices


def to_time(
    arg,
    format: str | None = None,
    infer_time_format: bool = False,
    errors: DateTimeErrorChoices = "raise",
):
    """
    Parse time strings to time objects using fixed strptime formats ("%H:%M",
    "%H%M", "%I:%M%p", "%I%M%p", "%H:%M:%S", "%H%M%S", "%I:%M:%S%p",
    "%I%M%S%p")

    Use infer_time_format if all the strings are in the same format to speed
    up conversion.

    Parameters
    ----------
    arg : string in time format, datetime.time, list, tuple, 1-d array,  Series
        表示时间格式的字符串，datetime.time 对象，列表，元组，一维数组或者 Series
    format : str, default None
        用于将 arg 转换为时间对象的格式。如果为 None，则使用固定的格式。
    infer_time_format: bool, default False
        根据第一个非 NaN 元素推断时间格式。如果所有字符串都是相同格式，这将加速转换。
    errors : {'raise', 'coerce'}, default 'raise'
        - 如果为 'raise'，则无效的解析会引发异常
        - 如果为 'coerce'，则无效的解析将被设置为 None

    Returns
    -------
    datetime.time
        返回解析后的时间对象
    """
    # 如果 errors 不是 'raise' 或 'coerce'，则抛出 ValueError 异常
    if errors not in ("raise", "coerce"):
        raise ValueError("errors must be one of 'raise', or 'coerce'.")
    # 定义一个函数 _convert_listlike，用于将参数 arg 转换为时间对象列表
    def _convert_listlike(arg, format):
        # 如果 arg 是 list 或 tuple 类型，则将其转换为 numpy 数组，dtype 为对象类型
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg, dtype="O")

        # 如果 arg 是具有 ndim 属性且 ndim 大于 1 的对象，则抛出 TypeError 异常
        elif getattr(arg, "ndim", 1) > 1:
            raise TypeError(
                "arg must be a string, datetime, list, tuple, 1-d array, or Series"
            )

        # 将 arg 转换为 numpy 数组，dtype 为对象类型
        arg = np.asarray(arg, dtype="O")

        # 如果启用了推断时间格式且 format 为 None，则尝试猜测数组 arg 的时间格式
        if infer_time_format and format is None:
            format = _guess_time_format_for_array(arg)

        # 初始化一个空列表 times 来存储时间对象或 None
        times: list[time | None] = []

        # 如果 format 不为 None，则尝试将每个元素转换为 datetime 对象的时间部分
        if format is not None:
            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    # 如果发生错误且 errors 为 "raise"，则抛出 ValueError 异常
                    if errors == "raise":
                        msg = (
                            f"Cannot convert {element} to a time with given "
                            f"format {format}"
                        )
                        raise ValueError(msg) from err
                    # 否则将 None 添加到 times 列表中
                    times.append(None)
        else:
            # 如果 format 为 None，则尝试多种时间格式来转换每个元素
            formats = _time_formats[:]
            format_found = False
            for element in arg:
                time_object = None
                try:
                    # 尝试使用 fromisoformat 方法将元素转换为 time 对象
                    time_object = time.fromisoformat(element)
                except (ValueError, TypeError):
                    # 如果无法使用 fromisoformat 方法，则尝试使用 datetime.strptime 方法
                    for time_format in formats:
                        try:
                            time_object = datetime.strptime(element, time_format).time()
                            # 如果尚未找到格式，则将找到的格式移到列表的最前面
                            if not format_found:
                                fmt = formats.pop(formats.index(time_format))
                                formats.insert(0, fmt)
                                format_found = True
                            break
                        except (ValueError, TypeError):
                            continue

                # 如果成功转换为时间对象，则将其添加到 times 列表中；否则根据 errors 处理
                if time_object is not None:
                    times.append(time_object)
                elif errors == "raise":
                    raise ValueError(f"Cannot convert arg {arg} to a time")
                else:
                    times.append(None)

        # 返回转换后的时间对象列表 times
        return times

    # 如果 arg 为 None，则直接返回 arg
    if arg is None:
        return arg
    # 如果 arg 是 time 对象，则直接返回 arg
    elif isinstance(arg, time):
        return arg
    # 如果 arg 是 pandas Series 对象，则将其值转换为时间对象列表，并返回新的 Series 对象
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    # 如果 arg 是 pandas Index 对象，则将其转换为时间对象列表，并返回转换后的结果
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, format)
    # 如果 arg 是类列表对象，则将其转换为时间对象列表，并返回转换后的结果
    elif is_list_like(arg):
        return _convert_listlike(arg, format)

    # 如果 arg 是单个元素，则将其转换为时间对象列表，并返回第一个元素
    return _convert_listlike(np.array([arg]), format)[0]
# 固定的时间格式列表，用于时间解析
_time_formats = [
    "%H:%M",         # 小时:分钟 格式，例如 14:30
    "%H%M",          # 小时分钟连续格式，例如 1430
    "%I:%M%p",       # 十二小时制小时:分钟AM/PM格式，例如 2:30PM
    "%I%M%p",        # 十二小时制小时分钟AM/PM连续格式，例如 230PM
    "%H:%M:%S",      # 小时:分钟:秒 格式，例如 14:30:00
    "%H%M%S",        # 小时分钟秒连续格式，例如 143000
    "%I:%M:%S%p",    # 十二小时制小时:分钟:秒AM/PM格式，例如 2:30:00PM
    "%I%M%S%p",      # 十二小时制小时分钟秒AM/PM连续格式，例如 230000PM
]

def _guess_time_format_for_array(arr):
    # 尝试根据第一个非 NaN 元素猜测时间格式
    non_nan_elements = notna(arr).nonzero()[0]  # 找出非 NaN 元素的索引
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]  # 获取第一个非 NaN 元素
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)  # 尝试使用各种时间格式解析元素
                return time_format  # 如果成功解析，返回该格式
            except ValueError:
                pass

    return None  # 如果无法解析任何格式，返回 None
```