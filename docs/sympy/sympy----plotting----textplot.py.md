# `D:\src\scipysrc\sympy\sympy\plotting\textplot.py`

```
# 从 sympy 库中导入 Float 类和 Dummy 类
from sympy.core.numbers import Float
from sympy.core.symbol import Dummy
# 从 sympy.utilities.lambdify 导入 lambdify 函数
from sympy.utilities.lambdify import lambdify
# 导入 math 库
import math

# 检查浮点数 x 是否有效的函数
def is_valid(x):
    """Check if a floating point number is valid"""
    # 如果 x 是 None，则无效
    if x is None:
        return False
    # 如果 x 是复数，则无效
    if isinstance(x, complex):
        return False
    # 如果 x 是无穷大或者不是数字，则无效
    return not math.isinf(x) and not math.isnan(x)


# 将数组 y 重新缩放到整数值区间 [0, H-1] 中，对应值区间 [mi, ma]
def rescale(y, W, H, mi, ma):
    """Rescale the given array `y` to fit into the integer values
    between `0` and `H-1` for the values between ``mi`` and ``ma``.
    """
    # 新的缩放后的数组 y_new
    y_new = []

    # 计算范围的差值 norm 和偏移 offset
    norm = ma - mi
    offset = (ma + mi) / 2

    # 遍历输入数组 y 的元素
    for x in range(W):
        # 如果 y[x] 是有效的数值
        if is_valid(y[x]):
            # 标准化到 [-1, 1] 区间
            normalized = (y[x] - offset) / norm
            # 如果标准化后的值无效，则将 y_new 中对应位置设为 None
            if not is_valid(normalized):
                y_new.append(None)
            else:
                # 重新缩放并四舍五入到整数
                rescaled = Float((normalized*H + H/2) * (H-1)/H).round()
                rescaled = int(rescaled)
                y_new.append(rescaled)
        else:
            # 如果 y[x] 不是有效的数值，则将 y_new 中对应位置设为 None
            y_new.append(None)
    return y_new


# 生成均匀分布的数组
def linspace(start, stop, num):
    return [start + (stop - start) * x / (num-1) for x in range(num)]


# 生成文本形式的图形函数
def textplot_str(expr, a, b, W=55, H=21):
    """Generator for the lines of the plot"""
    # 获取表达式中的自由变量
    free = expr.free_symbols
    # 如果自由变量超过一个，抛出 ValueError 异常
    if len(free) > 1:
        raise ValueError(
            "The expression must have a single variable. (Got {})"
            .format(free))
    # 弹出唯一的自由变量，如果没有则创建一个 Dummy 变量
    x = free.pop() if free else Dummy()
    # 将表达式转换为可调用的函数
    f = lambdify([x], expr)

    # 如果 a 是复数且虚部为 0，则取其实部
    if isinstance(a, complex):
        if a.imag == 0:
            a = a.real
    # 如果 b 是复数且虚部为 0，则取其实部
    if isinstance(b, complex):
        if b.imag == 0:
            b = b.real

    # 将 a 和 b 转换为浮点数
    a = float(a)
    b = float(b)

    # 计算函数在区间 [a, b] 上的均匀分布
    x = linspace(a, b, W)
    y = []
    # 计算函数在每个点的值
    for val in x:
        try:
            y.append(f(val))
        # 捕获可能出现的异常，但具体要捕获哪些异常不明确
        except (ValueError, TypeError, ZeroDivisionError):
            y.append(None)

    # 根据有效值计算高度的范围
    y_valid = list(filter(is_valid, y))
    if y_valid:
        ma = max(y_valid)
        mi = min(y_valid)
        # 如果最大值和最小值相等，则扩展范围
        if ma == mi:
            if ma:
                mi, ma = sorted([0, 2*ma])
            else:
                mi, ma = -1, 1
    else:
        # 如果没有有效值，则使用默认范围
        mi, ma = -1, 1

    # 调整范围的精度并四舍五入
    y_range = ma - mi
    precision = math.floor(math.log10(y_range)) - 1
    precision *= -1
    mi = round(mi, precision)
    ma = round(ma, precision)

    # 对函数值进行重新缩放
    y = rescale(y, W, H, mi, ma)

    # 计算 y 的分段区间
    y_bins = linspace(mi, ma, H)

    # 绘制图形的边距
    margin = 7
    # 从最高行向最低行迭代
    for h in range(H - 1, -1, -1):
        # 创建一个长度为 W 的空白字符列表
        s = [' '] * W
        # 遍历每一列
        for i in range(W):
            # 如果当前列的 y 值等于当前行 h
            if y[i] == h:
                # 检查是否为斜杠字符
                if (i == 0 or y[i - 1] == h - 1) and (i == W - 1 or y[i + 1] == h + 1):
                    s[i] = '/'
                # 检查是否为反斜杠字符
                elif (i == 0 or y[i - 1] == h + 1) and (i == W - 1 or y[i + 1] == h - 1):
                    s[i] = '\\'
                else:
                    s[i] = '.'
        
        # 如果当前行 h 为最底部行
        if h == 0:
            # 将所有字符设置为下划线
            for i in range(W):
                s[i] = '_'
        
        # 打印 y 值
        if h in (0, H//2, H - 1):
            # 创建前缀，右对齐，不足部分填充空格
            prefix = ("%g" % y_bins[h]).rjust(margin)[:margin]
        else:
            # 创建空白前缀
            prefix = " " * margin
        
        # 将字符列表连接成字符串
        s = "".join(s)
        
        # 如果当前行 h 为中间行
        if h == H//2:
            # 将空格替换为横线
            s = s.replace(" ", "-")
        
        # 生成格式化后的输出行
        yield prefix + " |" + s
    
    # 打印 x 值
    bottom = " " * (margin + 2)
    # 左侧添加空白
    bottom += ("%g" % x[0]).ljust(W//2)
    # 如果列数为奇数，则补充额外空白
    if W % 2 == 1:
        bottom += ("%g" % x[W//2]).ljust(W//2)
    else:
        bottom += ("%g" % x[W//2]).ljust(W//2-1)
    # 右侧添加最后一个 x 值
    bottom += "%g" % x[-1]
    # 生成底部行的输出
    yield bottom
# 定义函数 textplot，用于在 ASCII 艺术风格中打印 SymPy 表达式 expr 的图形
def textplot(expr, a, b, W=55, H=21):
    # 文档字符串，解释函数的作用和用法示例
    r"""
    Print a crude ASCII art plot of the SymPy expression 'expr' (which
    should contain a single symbol, e.g. x or something else) over the
    interval [a, b].

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.plotting import textplot
    >>> t = Symbol('t')
    >>> textplot(sin(t)*t, 0, 15)
     14 |                                                  ...
        |                                                     .
        |                                                 .
        |                                                      .
        |                                                .
        |                            ...
        |                           /   .               .
        |                          /
        |                         /      .
        |                        .        .            .
    1.5 |----.......--------------------------------------------
        |....       \           .          .
        |            \         /                      .
        |             ..      /             .
        |               \    /                       .
        |                ....
        |                                    .
        |                                     .     .
        |
        |                                      .   .
    -11 |_______________________________________________________
         0                          7.5                        15
    """
    # 调用 textplot_str 函数生成 ASCII 图形的每一行，并打印出来
    for line in textplot_str(expr, a, b, W, H):
        print(line)
```