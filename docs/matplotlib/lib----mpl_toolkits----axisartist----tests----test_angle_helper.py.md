# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\tests\test_angle_helper.py`

```py
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 mpl_toolkits.axisartist.angle_helper 模块中导入以下函数和类
from mpl_toolkits.axisartist.angle_helper import (
    FormatterDMS, FormatterHMS, select_step, select_step24, select_step360)

# 正则表达式模式字符串，用于匹配角度表示中的度分秒格式
_MS_RE = (
    r'''\$  # Mathtext
        (
            # The sign sometimes appears on a 0 when a fraction is shown.
            # Check later that there's only one.
            (?P<degree_sign>-)?  # 度数的符号，可能为负号
            (?P<degree>[0-9.]+)  # 度数的数值部分
            {degree}  # 度数符号（稍后将被格式替换）
        )?
        (
            (?(degree)\\,)  # 如果度数也可见，则添加分隔符
            (?P<minute_sign>-)?  # 分钟的符号，可能为负号
            (?P<minute>[0-9.]+)  # 分钟的数值部分
            {minute}  # 分钟符号（稍后将被格式替换）
        )?
        (
            (?(minute)\\,)  # 如果分钟也可见，则添加分隔符
            (?P<second_sign>-)?  # 秒数的符号，可能为负号
            (?P<second>[0-9.]+)  # 秒数的数值部分
            {second}  # 秒数符号（稍后将被格式替换）
        )?
        \$  # Mathtext
    '''
)

# 编译生成度分秒格式的正则表达式对象，用于 FormatterDMS 类使用
DMS_RE = re.compile(_MS_RE.format(degree=re.escape(FormatterDMS.deg_mark),
                                  minute=re.escape(FormatterDMS.min_mark),
                                  second=re.escape(FormatterDMS.sec_mark)),
                    re.VERBOSE)

# 编译生成时分秒格式的正则表达式对象，用于 FormatterHMS 类使用
HMS_RE = re.compile(_MS_RE.format(degree=re.escape(FormatterHMS.deg_mark),
                                  minute=re.escape(FormatterHMS.min_mark),
                                  second=re.escape(FormatterHMS.sec_mark)),
                    re.VERBOSE)


# 将度分秒表示转换为浮点数表示
def dms2float(degrees, minutes=0, seconds=0):
    return degrees + minutes / 60.0 + seconds / 3600.0


# 使用 pytest 的 parametrize 装饰器来定义测试用例参数化
@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [
    # 第一个测试用例：不包含小时的情况
    ((-180, 180, 10), {'hour': False}, np.arange(-180, 181, 30), 1.0),
    # 第二个测试用例：包含小时的情况
    ((-12, 12, 10), {'hour': True}, np.arange(-12, 13, 2), 1.0)
])
# 测试函数，测试 select_step 函数
def test_select_step(args, kwargs, expected_levels, expected_factor):
    # 调用 select_step 函数，返回结果
    levels, n, factor = select_step(*args, **kwargs)

    # 使用 assert 断言检查结果是否符合预期
    assert n == len(levels)  # 检查 levels 的长度与 n 是否相等
    np.testing.assert_array_equal(levels, expected_levels)  # 检查 levels 是否与 expected_levels 相等
    assert factor == expected_factor  # 检查 factor 是否与 expected_factor 相等


# 使用 pytest 的 parametrize 装饰器来定义测试用例参数化
@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [
    # 第一个测试用例：不包含小时的情况
    ((-180, 180, 10), {}, np.arange(-180, 181, 30), 1.0),
    # 第二个测试用例：不包含小时的情况
    ((-12, 12, 10), {}, np.arange(-750, 751, 150), 60.0)
])
# 测试函数，测试 select_step24 函数
def test_select_step24(args, kwargs, expected_levels, expected_factor):
    # 调用 select_step24 函数，返回结果
    levels, n, factor = select_step24(*args, **kwargs)

    # 使用 assert 断言检查结果是否符合预期
    assert n == len(levels)  # 检查 levels 的长度与 n 是否相等
    np.testing.assert_array_equal(levels, expected_levels)  # 检查 levels 是否与 expected_levels 相等
    assert factor == expected_factor  # 检查 factor 是否与 expected_factor 相等


# 使用 pytest 的 parametrize 装饰器来定义测试用例参数化
@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [
    # 第一个测试用例：不包含小时的情况
    ((dms2float(20, 21.2), dms2float(21, 33.3), 5), {},
     np.arange(1215, 1306, 15), 60.0),
    # 第二个测试用例：不包含小时的情况
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {},
     np.arange(73820, 73835, 2), 3600.0),
    # 第三个测试用例：不包含小时的情况
    ((dms2float(20, 21.2), dms2float(20, 53.3), 5), {},
     np.arange(1220, 1256, 5), 60.0),
    # 第四个测试用例：不包含小时的情况
    ((21.2, 33.3, 5), {},
     np.arange(20, 35, 2), 1.0),
])
# 测试函数，测试 select_step360 函数
def test_select_step360(args, kwargs, expected_levels, expected_factor):
    # 调用 select_step360 函数，返回结果
    levels, n, factor = select_step360(*args, **kwargs)

    # 使用 assert 断言检查结果是否符合预期
    assert n == len(levels)  # 检查 levels 的长度与 n 是否相等
    np.testing.assert_array_equal(levels, expected_levels)  # 检查 levels 是否与 expected_levels 相等
    assert factor == expected_factor  # 检查 factor 是否与 expected_factor 相等
    # 创建包含经纬度和高度的元组，转换为浮点数表示
    ((dms2float(20, 21.2), dms2float(21, 33.3), 5), {},
     # 创建一个从1215到1305的间隔为15的整数数组
     np.arange(1215, 1306, 15), 60.0),
    # 创建包含经纬度和高度的元组，包括秒数，并转换为浮点数表示
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {},
     # 创建一个从73820到73834的间隔为2的整数数组
     np.arange(73820, 73835, 2), 3600.0),
    # 创建包含经纬度和高度的元组，包括秒数，并转换为浮点数表示
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=21.4), 5), {},
     # 创建一个从7382120到7382140的间隔为5的整数数组
     np.arange(7382120, 7382141, 5), 360000.0),
    # 测试阈值因子
    ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5),
     # 设置阈值因子为60
     {'threshold_factor': 60}, np.arange(12301, 12310), 600.0),
    # 创建包含经纬度和高度的元组，包括秒数，并转换为浮点数表示
    ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5),
     # 设置阈值因子为1
     {'threshold_factor': 1}, np.arange(20502, 20517, 2), 1000.0),
# 定义一个测试函数 test_select_step360，用于测试 select_step360 函数的行为
def test_select_step360(args, kwargs, expected_levels, expected_factor):
    # 调用 select_step360 函数，获取返回的 levels, n 和 factor
    levels, n, factor = select_step360(*args, **kwargs)

    # 断言 n 应该等于 levels 的长度
    assert n == len(levels)
    # 使用 NumPy 的函数检查 levels 是否与预期的 levels 数组相等
    np.testing.assert_array_equal(levels, expected_levels)
    # 断言 factor 应该等于预期的 factor
    assert factor == expected_factor


# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
@pytest.mark.parametrize('Formatter, regex',
                         [(FormatterDMS, DMS_RE),
                          (FormatterHMS, HMS_RE)],
                         ids=['Degree/Minute/Second', 'Hour/Minute/Second'])
@pytest.mark.parametrize('direction, factor, values', [
    ("left", 60, [0, -30, -60]),
    ("left", 600, [12301, 12302, 12303]),
    ("left", 3600, [0, -30, -60]),
    ("left", 36000, [738210, 738215, 738220]),
    ("left", 360000, [7382120, 7382125, 7382130]),
    ("left", 1., [45, 46, 47]),
    ("left", 10., [452, 453, 454]),
])
# 定义一个测试函数 test_formatters，用于测试 Formatter 类的格式化方法
def test_formatters(Formatter, regex, direction, factor, values):
    # 创建 Formatter 类的实例 fmt
    fmt = Formatter()
    # 调用格式化方法，获取结果 result
    result = fmt(direction, factor, values)

    # 初始化前一个度、分、秒为 None
    prev_degree = prev_minute = prev_second = None
    # 遍历结果 result 和 values 的元素，进行逐个检查
    for tick, value in zip(result, values):
        # 使用正则表达式 regex 匹配 tick
        m = regex.match(tick)
        # 断言匹配结果不为空，说明 tick 是预期的格式
        assert m is not None, f'{tick!r} is not an expected tick format.'

        # 计算 tick 中 'degree', 'minute', 'second' 的数量
        sign = sum(m.group(sign + '_sign') is not None
                   for sign in ('degree', 'minute', 'second'))
        # 断言 tick 中最多只能有一个元素有符号
        assert sign <= 1, f'Only one element of tick {tick!r} may have a sign.'
        # 如果没有符号，则 sign 等于 1；否则为 -1
        sign = 1 if sign == 0 else -1

        # 解析 tick 中的度、分、秒为浮点数
        degree = float(m.group('degree') or prev_degree or 0)
        minute = float(m.group('minute') or prev_minute or 0)
        second = float(m.group('second') or prev_second or 0)

        # 根据 Formatter 的类型计算预期的数值
        if Formatter == FormatterHMS:
            # 对于 FormatterHMS，将度数值映射到 24 小时范围
            expected_value = pytest.approx((value // 15) / factor)
        else:
            # 对于 FormatterDMS，直接计算度数值
            expected_value = pytest.approx(value / factor)
        
        # 断言解析后的度、分、秒转换成浮点数后与预期值相符
        assert sign * dms2float(degree, minute, second) == expected_value, \
            f'{tick!r} does not match expected tick value.'

        # 更新前一个度、分、秒的值
        prev_degree = degree
        prev_minute = minute
        prev_second = second
```