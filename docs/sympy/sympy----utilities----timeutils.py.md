# `D:\src\scipysrc\sympy\sympy\utilities\timeutils.py`

```
"""
Simple tools for timing functions' execution, when IPython is not available.
"""

# 导入计时模块和数学模块
import timeit
import math

# 时间单位的缩放因子和单位名称
_scales = [1e0, 1e3, 1e6, 1e9]
_units = ['s', 'ms', '\N{GREEK SMALL LETTER MU}s', 'ns']

# 函数：测量函数执行时间
def timed(func, setup="pass", limit=None):
    """Adaptively measure execution time of a function."""
    # 创建计时器对象
    timer = timeit.Timer(func, setup=setup)
    repeat, number = 3, 1

    # 循环调整测试次数
    for i in range(1, 10):
        if timer.timeit(number) >= 0.2:
            break
        elif limit is not None and number >= limit:
            break
        else:
            number *= 10

    # 获取最小执行时间
    time = min(timer.repeat(repeat, number)) / number

    # 计算时间单位的序号
    if time > 0.0:
        order = min(-int(math.floor(math.log10(time)) // 3), 3)
    else:
        order = 3

    # 返回元组：测试次数、平均时间、调整后的时间、时间单位
    return (number, time, time * _scales[order], _units[order])


# 函数：内联递归算法的计时代码

def __do_timings():
    import os
    # 从环境变量获取用逗号分隔的定时任务名称列表
    res = os.getenv('SYMPY_TIMINGS', '')
    res = [x.strip() for x in res.split(',')]
    return set(res)

# 初始化全局变量 _do_timings
_do_timings = __do_timings()
_timestack = None


# 函数：打印时间栈
def _print_timestack(stack, level=1):
    # 打印时间信息：缩进、时间、函数名和单位
    print('-' * level, '%.2f %s%s' % (stack[2], stack[0], stack[3]))
    # 递归打印下层时间栈
    for s in stack[1]:
        _print_timestack(s, level + 1)


# 装饰器函数：计时装饰器
def timethis(name):
    def decorator(func):
        # 使用全局变量 _do_timings
        global _do_timings
        # 如果函数名称不在定时任务集合中，直接返回原函数
        if name not in _do_timings:
            return func

        # 装饰函数
        def wrapper(*args, **kwargs):
            from time import time
            global _timestack
            # 保存旧时间栈
            oldtimestack = _timestack
            # 设置当前时间栈：函数名、空列表、初始时间为0、传入参数
            _timestack = [func.__name__, [], 0, args]
            t1 = time()
            # 执行被装饰的函数
            r = func(*args, **kwargs)
            t2 = time()
            # 计算函数执行时间
            _timestack[2] = t2 - t1
            # 如果存在旧时间栈，则将当前时间栈添加到旧时间栈中；否则打印当前时间栈
            if oldtimestack is not None:
                oldtimestack[1].append(_timestack)
                _timestack = oldtimestack
            else:
                _print_timestack(_timestack)
                _timestack = None
            return r

        return wrapper

    return decorator
```