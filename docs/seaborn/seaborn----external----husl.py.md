# `D:\src\scipysrc\seaborn\seaborn\external\husl.py`

```
# 导入 operator 和 math 模块，operator 用于操作符函数，math 用于数学运算
import operator
import math

# 定义版本号
__version__ = "2.1.0"

# 转换矩阵 m 和其逆矩阵 m_inv，用于颜色空间转换
m = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
]

m_inv = [
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
]

# D65 光照条件下的参考值
refX = 0.95047
refY = 1.00000
refZ = 1.08883
refU = 0.19784
refV = 0.46834
lab_e = 0.008856
lab_k = 903.3


# 公共 API 函数

# 将 HUSL 颜色空间转换为 RGB 颜色空间
def husl_to_rgb(h, s, l):
    return lch_to_rgb(*husl_to_lch([h, s, l]))


# 将 HUSL 颜色空间转换为十六进制颜色表示
def husl_to_hex(h, s, l):
    return rgb_to_hex(husl_to_rgb(h, s, l))


# 将 RGB 颜色空间转换为 HUSL 颜色空间
def rgb_to_husl(r, g, b):
    return lch_to_husl(rgb_to_lch(r, g, b))


# 将十六进制颜色表示转换为 HUSL 颜色空间
def hex_to_husl(hex):
    return rgb_to_husl(*hex_to_rgb(hex))


# 将 HUSLp 颜色空间转换为 RGB 颜色空间
def huslp_to_rgb(h, s, l):
    return lch_to_rgb(*huslp_to_lch([h, s, l]))


# 将 HUSLp 颜色空间转换为十六进制颜色表示
def huslp_to_hex(h, s, l):
    return rgb_to_hex(huslp_to_rgb(h, s, l))


# 将 RGB 颜色空间转换为 HUSLp 颜色空间
def rgb_to_huslp(r, g, b):
    return lch_to_huslp(rgb_to_lch(r, g, b))


# 将十六进制颜色表示转换为 HUSLp 颜色空间
def hex_to_huslp(hex):
    return rgb_to_huslp(*hex_to_rgb(hex))


# 将 LCH 颜色空间转换为 RGB 颜色空间
def lch_to_rgb(l, c, h):
    return xyz_to_rgb(luv_to_xyz(lch_to_luv([l, c, h])))


# 将 RGB 颜色空间转换为 LCH 颜色空间
def rgb_to_lch(r, g, b):
    return luv_to_lch(xyz_to_luv(rgb_to_xyz([r, g, b])))


# 计算在给定亮度 L 和色相 H 下的最大饱和度
def max_chroma(L, H):
    hrad = math.radians(H)
    sinH = (math.sin(hrad))
    cosH = (math.cos(hrad))
    sub1 = (math.pow(L + 16, 3.0) / 1560896.0)
    sub2 = sub1 if sub1 > 0.008856 else (L / 903.3)
    result = float("inf")
    for row in m:
        m1 = row[0]
        m2 = row[1]
        m3 = row[2]
        top = ((0.99915 * m1 + 1.05122 * m2 + 1.14460 * m3) * sub2)
        rbottom = (0.86330 * m3 - 0.17266 * m2)
        lbottom = (0.12949 * m3 - 0.38848 * m1)
        bottom = (rbottom * sinH + lbottom * cosH) * sub2

        for t in (0.0, 1.0):
            C = (L * (top - 1.05122 * t) / (bottom + 0.17266 * sinH * t))
            if C > 0.0 and C < result:
                result = C
    return result


# 计算给定亮度 L 下的最大过去色彩饱和度
def max_chroma_pastel(L):
    H = math.degrees(_hrad_extremum(L))
    return max_chroma(L, H)


# 计算向量 a 和 b 的点积
def dot_product(a, b):
    return sum(map(operator.mul, a, b))


# 函数 f，根据 t 的值返回不同的结果
def f(t):
    if t > lab_e:
        return (math.pow(t, 1.0 / 3.0))
    else:
        # 如果 t 大于 0.008856，则应用非线性变换
        return (7.787 * t + 16.0 / 116.0)
def f_inv(t):
    # 如果 t 的立方大于 lab_e，返回 t 的立方
    if math.pow(t, 3.0) > lab_e:
        return (math.pow(t, 3.0))
    else:
        # 否则返回 (116.0 * t - 16.0) / lab_k
        return (116.0 * t - 16.0) / lab_k


def from_linear(c):
    # 如果 c 小于等于 0.0031308，应用线性变换
    if c <= 0.0031308:
        return 12.92 * c
    else:
        # 否则应用另一种线性变换
        return (1.055 * math.pow(c, 1.0 / 2.4) - 0.055)


def to_linear(c):
    # 设定参数 a 为 0.055
    a = 0.055

    # 如果 c 大于 0.04045，应用线性变换
    if c > 0.04045:
        return (math.pow((c + a) / (1.0 + a), 2.4))
    else:
        # 否则应用另一种线性变换
        return (c / 12.92)


def rgb_prepare(triple):
    # 初始化空列表 ret 用于存储处理后的 RGB 值
    ret = []
    # 遍历 RGB 三个通道的值
    for ch in triple:
        # 将每个通道的值四舍五入到三位小数
        ch = round(ch, 3)

        # 如果通道值 ch 超出合法范围 [-0.0001, 1.0001]，抛出异常
        if ch < -0.0001 or ch > 1.0001:
            raise Exception(f"Illegal RGB value {ch:f}")

        # 如果通道值 ch 小于 0，设为 0；如果大于 1，设为 1
        if ch < 0:
            ch = 0
        if ch > 1:
            ch = 1

        # 将处理后的通道值转换为整数并添加到 ret 列表中
        ret.append(int(round(ch * 255 + 0.001, 0)))

    return ret


def hex_to_rgb(hex):
    # 如果十六进制颜色值以 '#' 开头，去除 '#'
    if hex.startswith('#'):
        hex = hex[1:]
    # 将十六进制颜色值分别转换为 RGB 的浮点数表示
    r = int(hex[0:2], 16) / 255.0
    g = int(hex[2:4], 16) / 255.0
    b = int(hex[4:6], 16) / 255.0
    return [r, g, b]


def rgb_to_hex(triple):
    # 解构 RGB 三通道的值
    [r, g, b] = triple
    # 将 RGB 值转换为十六进制表示并返回
    return '#%02x%02x%02x' % tuple(rgb_prepare([r, g, b]))


def xyz_to_rgb(triple):
    # 使用矩阵 m 对 XYZ 色彩空间进行变换，并应用非线性变换返回 RGB 值
    xyz = map(lambda row: dot_product(row, triple), m)
    return list(map(from_linear, xyz))


def rgb_to_xyz(triple):
    # 应用线性变换将 RGB 转换为 XYZ 色彩空间的值
    rgbl = list(map(to_linear, triple))
    return list(map(lambda row: dot_product(row, rgbl), m_inv))


def xyz_to_luv(triple):
    # 解构 XYZ 三通道的值
    X, Y, Z = triple

    # 如果 XYZ 值全为 0，返回全零列表
    if X == Y == Z == 0.0:
        return [0.0, 0.0, 0.0]

    # 计算 L、U、V 的值
    varU = (4.0 * X) / (X + (15.0 * Y) + (3.0 * Z))
    varV = (9.0 * Y) / (X + (15.0 * Y) + (3.0 * Z))
    L = 116.0 * f(Y / refY) - 16.0

    # 如果 L 为 0，返回全零列表
    if L == 0.0:
        return [0.0, 0.0, 0.0]

    # 计算 U、V 的值
    U = 13.0 * L * (varU - refU)
    V = 13.0 * L * (varV - refV)

    return [L, U, V]


def luv_to_xyz(triple):
    # 解构 LUV 三通道的值
    L, U, V = triple

    # 如果 L 为 0，返回全零列表
    if L == 0:
        return [0.0, 0.0, 0.0]

    # 计算 varY、varU、varV 的值
    varY = f_inv((L + 16.0) / 116.0)
    varU = U / (13.0 * L) + refU
    varV = V / (13.0 * L) + refV

    # 计算 XYZ 的值
    Y = varY * refY
    X = 0.0 - (9.0 * Y * varU) / ((varU - 4.0) * varV - varU * varV)
    Z = (9.0 * Y - (15.0 * varV * Y) - (varV * X)) / (3.0 * varV)

    return [X, Y, Z]


def luv_to_lch(triple):
    # 解构 LUV 三通道的值
    L, U, V = triple

    # 计算 C、H 的值
    C = (math.pow(math.pow(U, 2) + math.pow(V, 2), (1.0 / 2.0)))
    hrad = (math.atan2(V, U))
    H = math.degrees(hrad)
    if H < 0.0:
        H = 360.0 + H

    return [L, C, H]


def lch_to_luv(triple):
    # 解构 LCH 三通道的值
    L, C, H = triple

    # 计算 H 的弧度值
    Hrad = math.radians(H)

    # 计算 U、V 的值
    U = (math.cos(Hrad) * C)
    V = (math.sin(Hrad) * C)

    return [L, U, V]


def husl_to_lch(triple):
    # 解构 HUSL 三通道的值
    H, S, L = triple

    # 如果 L 大于 99.9999999，返回最大饱和度的 LCH 值
    if L > 99.9999999:
        return [100, 0.0, H]
    # 如果 L 小于 0.00000001，返回全零列表
    if L < 0.00000001:
        return [0.0, 0.0, H]

    # 计算最大饱和度的 C 值
    mx = max_chroma(L, H)
    C = mx / 100.0 * S

    return [L, C, H]


def lch_to_husl(triple):
    # 待实现的函数，暂未提供具体实现
    # 从三元组 `triple` 中分别解包出 L, C, H 三个变量
    L, C, H = triple

    # 如果亮度 L 大于 99.9999999%，则返回饱和度为 0.0，亮度为 100.0 的颜色
    if L > 99.9999999:
        return [H, 0.0, 100.0]

    # 如果亮度 L 小于 0.00000001%，则返回饱和度为 0.0，亮度为 0.0 的颜色
    if L < 0.00000001:
        return [H, 0.0, 0.0]

    # 计算最大色度 (chroma) mx，调用 max_chroma 函数计算
    mx = max_chroma(L, H)

    # 计算饱和度 S，公式为 C / mx * 100.0
    S = C / mx * 100.0

    # 返回颜色的 HSL（色调、饱和度、亮度）表示，作为列表 [H, S, L]
    return [H, S, L]
# 将 HuslP（Hue, Saturation, Lightness Pastel）颜色空间的三元组转换为 LCH（Lightness, Chroma, Hue）颜色空间的三元组
def huslp_to_lch(triple):
    # 解构输入的三元组，分别获取 H, S, L 三个值
    H, S, L = triple

    # 如果 L 值接近或等于 100，则返回光谱颜色的极限值
    if L > 99.9999999:
        return [100, 0.0, H]
    # 如果 L 值接近或等于 0，则返回黑色的极限值
    if L < 0.00000001:
        return [0.0, 0.0, H]

    # 计算最大的过去色饱和度
    mx = max_chroma_pastel(L)
    # 计算在 LCH 颜色空间中的色度 C
    C = mx / 100.0 * S

    # 返回转换后的 LCH 颜色空间的三元组
    return [L, C, H]


# 将 LCH（Lightness, Chroma, Hue）颜色空间的三元组转换为 HuslP（Hue, Saturation, Lightness Pastel）颜色空间的三元组
def lch_to_huslp(triple):
    # 解构输入的三元组，分别获取 L, C, H 三个值
    L, C, H = triple

    # 如果 L 值接近或等于 100，则返回 HuslP 颜色空间的光谱色值
    if L > 99.9999999:
        return [H, 0.0, 100.0]
    # 如果 L 值接近或等于 0，则返回 HuslP 颜色空间的黑色值
    if L < 0.00000001:
        return [H, 0.0, 0.0]

    # 计算最大的过去色饱和度
    mx = max_chroma_pastel(L)
    # 计算在 HuslP 颜色空间中的饱和度 S
    S = C / mx * 100.0

    # 返回转换后的 HuslP 颜色空间的三元组
    return [H, S, L]
```