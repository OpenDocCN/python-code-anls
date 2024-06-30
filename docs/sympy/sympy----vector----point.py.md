# `D:\src\scipysrc\sympy\sympy\vector\point.py`

```
# 从 sympy 库导入基础类 Basic
from sympy.core.basic import Basic
# 从 sympy 库导入字符串类 Str
from sympy.core.symbol import Str
# 从 sympy.vector.vector 模块导入向量类 Vector
from sympy.vector.vector import Vector
# 从 sympy.vector.coordsysrect 模块导入三维直角坐标系类 CoordSys3D
from sympy.vector.coordsysrect import CoordSys3D
# 从 sympy.vector.functions 模块导入 _path 函数
from sympy.vector.functions import _path
# 从 sympy.core.cache 模块导入 cacheit 装饰器
from sympy.core.cache import cacheit


class Point(Basic):
    """
    Represents a point in 3-D space.
    """

    def __new__(cls, name, position=Vector.zero, parent_point=None):
        # 将名称转换为字符串
        name = str(name)
        # 检查参数有效性
        if not isinstance(position, Vector):
            raise TypeError(
                "position should be an instance of Vector, not %s" % type(
                    position))
        if (not isinstance(parent_point, Point) and
                parent_point is not None):
            raise TypeError(
                "parent_point should be an instance of Point, not %s" % type(
                    parent_point))
        # 调用父类的构造方法创建对象
        if parent_point is None:
            obj = super().__new__(cls, Str(name), position)
        else:
            obj = super().__new__(cls, Str(name), position, parent_point)
        # 设置对象的属性
        obj._name = name
        obj._pos = position
        if parent_point is None:
            obj._parent = None
            obj._root = obj
        else:
            obj._parent = parent_point
            obj._root = parent_point._root
        # 返回创建的对象
        return obj

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of this Point with respect to
        another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this Point wrt it is
            returned. If its an instance of CoordSyRect, the position
            wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> N.origin.position_wrt(p1)
        (-10)*N.i

        """

        # 检查参数是否为 Point 或 CoordSys3D 类型
        if (not isinstance(other, Point) and
                not isinstance(other, CoordSys3D)):
            raise TypeError(str(other) +
                            "is not a Point or CoordSys3D")
        # 如果 other 是 CoordSys3D 类型，则使用其原点
        if isinstance(other, CoordSys3D):
            other = other.origin
        # 处理特殊情况
        if other == self:
            return Vector.zero
        elif other == self._parent:
            return self._pos
        elif other._parent == self:
            return -1 * other._pos
        # 否则，使用点的树结构来计算位置向量
        rootindex, path = _path(self, other)
        result = Vector.zero
        for i in range(rootindex):
            result += path[i]._pos
        for i in range(rootindex + 1, len(path)):
            result -= path[i]._pos
        return result
    def locate_new(self, name, position):
        """
        返回一个位于给定位置的新点，相对于当前点。

        Parameters
        ==========

        name : str
            新点的名称

        position : Vector
            相对于当前点的位置向量

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p1.position_wrt(N.origin)
        10*N.i

        """
        # 创建并返回一个新的 Point 对象，其名称为 'name'，位置为 'position'，基于当前点 self
        return Point(name, position, self)

    def express_coordinates(self, coordinate_system):
        """
        返回该点相对于给定 CoordSys3D 实例原点的直角坐标。

        Parameters
        ==========

        coordinate_system : CoordSys3D
            用于表示该点坐标的坐标系。

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p2 = p1.locate_new('p2', 5 * N.j)
        >>> p2.express_coordinates(N)
        (10, 5, 0)

        """

        # 确定相对位置向量
        pos_vect = self.position_wrt(coordinate_system.origin)
        # 在给定的坐标系中表达该位置向量
        return tuple(pos_vect.to_matrix(coordinate_system))

    def _sympystr(self, printer):
        # 返回点的名称作为其在 SymPy 中的字符串表示
        return self._name
```