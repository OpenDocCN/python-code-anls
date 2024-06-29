# `D:\src\scipysrc\matplotlib\lib\matplotlib\hatch.py`

```
"""Contains classes for generating hatch patterns."""

# 导入必要的库
import numpy as np

# 导入 matplotlib 中的 _api 模块和 Path 类
from matplotlib import _api
from matplotlib.path import Path

# 定义一个基类用于生成阴影图案
class HatchPatternBase:
    """The base class for a hatch pattern."""
    pass

# 定义一个水平阴影图案类，继承自 HatchPatternBase
class HorizontalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        # 计算水平线的数量
        self.num_lines = int((hatch.count('-') + hatch.count('+')) * density)
        # 计算顶点的数量
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        # 在水平方向上均匀分布顶点
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = 0.0
        vertices[0::2, 1] = steps
        vertices[1::2, 0] = 1.0
        vertices[1::2, 1] = steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

# 定义一个垂直阴影图案类，继承自 HatchPatternBase
class VerticalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        # 计算垂直线的数量
        self.num_lines = int((hatch.count('|') + hatch.count('+')) * density)
        # 计算顶点的数量
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        # 在垂直方向上均匀分布顶点
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = steps
        vertices[0::2, 1] = 0.0
        vertices[1::2, 0] = steps
        vertices[1::2, 1] = 1.0
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

# 定义一个东北方向阴影图案类，继承自 HatchPatternBase
class NorthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        # 计算东北方向线的数量
        self.num_lines = int(
            (hatch.count('/') + hatch.count('x') + hatch.count('X')) * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        # 在东北方向上均匀分布顶点
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

# 定义一个东南方向阴影图案类，继承自 HatchPatternBase
class SouthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        # 计算东南方向线的数量
        self.num_lines = int(
            (hatch.count('\\') + hatch.count('x') + hatch.count('X'))
            * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        # 在东南方向上均匀分布顶点
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

# 定义一个形状类，继承自 HatchPatternBase
class Shapes(HatchPatternBase):
    filled = False
    # 初始化函数，接受两个参数：hatch（填充图案类型）、density（密度）
    def __init__(self, hatch, density):
        # 如果行数为0，则设置形状数和顶点数为0
        if self.num_rows == 0:
            self.num_shapes = 0
            self.num_vertices = 0
        else:
            # 计算形状数，基于行数，考虑奇偶行数的不同情况
            self.num_shapes = ((self.num_rows // 2 + 1) * (self.num_rows + 1) +
                               (self.num_rows // 2) * self.num_rows)
            # 计算顶点数，考虑形状数、顶点列表长度和是否填充的影响
            self.num_vertices = (self.num_shapes *
                                 len(self.shape_vertices) *
                                 (1 if self.filled else 2))

    # 设置顶点和代码的函数，接受顶点和代码作为参数
    def set_vertices_and_codes(self, vertices, codes):
        # 计算偏移量，根据行数
        offset = 1.0 / self.num_rows
        # 复制形状顶点列表，乘以偏移量和大小
        shape_vertices = self.shape_vertices * offset * self.size
        shape_codes = self.shape_codes
        # 如果不是填充状态，则在前向和后向连接形状顶点
        if not self.filled:
            shape_vertices = np.concatenate(
                [shape_vertices, shape_vertices[::-1] * 0.9])
            shape_codes = np.concatenate([shape_codes, shape_codes])
        vertices_parts = []
        codes_parts = []
        # 循环生成每一行的顶点和代码
        for row in range(self.num_rows + 1):
            if row % 2 == 0:
                cols = np.linspace(0, 1, self.num_rows + 1)
            else:
                cols = np.linspace(offset / 2, 1 - offset / 2, self.num_rows)
            row_pos = row * offset
            for col_pos in cols:
                vertices_parts.append(shape_vertices + [col_pos, row_pos])
                codes_parts.append(shape_codes)
        # 将顶点和代码部分连接成最终的顶点和代码数组
        np.concatenate(vertices_parts, out=vertices)
        np.concatenate(codes_parts, out=codes)
class Circles(Shapes):
    def __init__(self, hatch, density):
        # 创建一个单位圆的路径对象
        path = Path.unit_circle()
        # 设置当前对象的形状顶点为单位圆的顶点
        self.shape_vertices = path.vertices
        # 设置当前对象的形状代码为单位圆的代码
        self.shape_codes = path.codes
        # 调用父类的构造函数，传入填充图案和密度参数
        super().__init__(hatch, density)


class SmallCircles(Circles):
    size = 0.2

    def __init__(self, hatch, density):
        # 计算'o'在填充图案中出现的次数乘以密度，作为行数
        self.num_rows = (hatch.count('o')) * density
        # 调用父类构造函数，传入填充图案和密度参数
        super().__init__(hatch, density)


class LargeCircles(Circles):
    size = 0.35

    def __init__(self, hatch, density):
        # 计算'O'在填充图案中出现的次数乘以密度，作为行数
        self.num_rows = (hatch.count('O')) * density
        # 调用父类构造函数，传入填充图案和密度参数
        super().__init__(hatch, density)


class SmallFilledCircles(Circles):
    size = 0.1
    filled = True

    def __init__(self, hatch, density):
        # 计算'.'在填充图案中出现的次数乘以密度，作为行数
        self.num_rows = (hatch.count('.')) * density
        # 调用父类构造函数，传入填充图案和密度参数
        super().__init__(hatch, density)


class Stars(Shapes):
    size = 1.0 / 3.0
    filled = True

    def __init__(self, hatch, density):
        # 计算'*'在填充图案中出现的次数乘以密度，作为行数
        self.num_rows = (hatch.count('*')) * density
        # 创建一个五角星的路径对象
        path = Path.unit_regular_star(5)
        # 设置当前对象的形状顶点为五角星的顶点
        self.shape_vertices = path.vertices
        # 设置当前对象的形状代码为五角星的代码，全部为Path.LINETO
        self.shape_codes = np.full(len(self.shape_vertices), Path.LINETO,
                                   dtype=Path.code_type)
        # 将第一个顶点的代码设置为Path.MOVETO
        self.shape_codes[0] = Path.MOVETO
        # 调用父类构造函数，传入填充图案和密度参数
        super().__init__(hatch, density)

_hatch_types = [
    HorizontalHatch,
    VerticalHatch,
    NorthEastHatch,
    SouthEastHatch,
    SmallCircles,
    LargeCircles,
    SmallFilledCircles,
    Stars
    ]

def _validate_hatch_pattern(hatch):
    # 定义有效的填充图案字符集合
    valid_hatch_patterns = set(r'-+|/\xXoO.*')
    if hatch is not None:
        # 找出填充图案中不属于有效字符集合的字符
        invalids = set(hatch).difference(valid_hatch_patterns)
        if invalids:
            # 将有效字符集合和无效字符集合转换成字符串，抛出警告
            valid = ''.join(sorted(valid_hatch_patterns))
            invalids = ''.join(sorted(invalids))
            _api.warn_deprecated(
                '3.4',
                removal='3.11',  # 自定义填充图案之后的一个版本 (#20690)
                message=f'hatch must consist of a string of "{valid}" or '
                        'None, but found the following invalid values '
                        f'"{invalids}". Passing invalid values is deprecated '
                        'since %(since)s and will become an error %(removal)s.'
            )

def get_path(hatchpattern, density=6):
    """
    给定填充图案*hatchpattern*，生成在单位正方形中渲染填充图案的路径。*density* 是每单位正方形的线数。
    """
    density = int(density)

    # 为每种填充图案类型创建一个对象，并传入填充图案和密度参数
    patterns = [hatch_type(hatchpattern, density)
                for hatch_type in _hatch_types]
    # 计算所有图案顶点数的总和
    num_vertices = sum([pattern.num_vertices for pattern in patterns])

    if num_vertices == 0:
        # 如果总顶点数为0，则返回一个空的路径对象
        return Path(np.empty((0, 2)))

    # 创建一个空的顶点数组和代码数组
    vertices = np.empty((num_vertices, 2))
    codes = np.empty(num_vertices, Path.code_type)

    cursor = 0
    # 对于每个图案对象中的图案模式，依次处理
    for pattern in patterns:
        # 如果图案模式的顶点数不为零
        if pattern.num_vertices != 0:
            # 从顶点数组中切片出当前模式所需的顶点数据块
            vertices_chunk = vertices[cursor:cursor + pattern.num_vertices]
            # 从代码数组中切片出当前模式所需的代码数据块
            codes_chunk = codes[cursor:cursor + pattern.num_vertices]
            # 将切片后的顶点数据块和代码数据块设置到当前模式对象中
            pattern.set_vertices_and_codes(vertices_chunk, codes_chunk)
            # 更新游标位置，移动到下一个模式的起始位置
            cursor += pattern.num_vertices

    # 使用最终的顶点数组和代码数组创建并返回一个路径对象
    return Path(vertices, codes)
```