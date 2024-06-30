# `D:\src\scipysrc\sympy\sympy\geometry\__init__.py`

```
"""
A geometry module for the SymPy library. This module contains all of the
entities and functions needed to construct basic geometrical data and to
perform simple informational queries.

Usage:
======

Examples
========

"""
# 导入 SymPy 几何模块中的点类：Point, Point2D, Point3D
from sympy.geometry.point import Point, Point2D, Point3D
# 导入 SymPy 几何模块中的线类：Line, Ray, Segment, Line2D, Segment2D, Ray2D,
# Line3D, Segment3D, Ray3D
from sympy.geometry.line import Line, Ray, Segment, Line2D, Segment2D, Ray2D, \
    Line3D, Segment3D, Ray3D
# 导入 SymPy 几何模块中的平面类：Plane
from sympy.geometry.plane import Plane
# 导入 SymPy 几何模块中的椭圆和圆类：Ellipse, Circle
from sympy.geometry.ellipse import Ellipse, Circle
# 导入 SymPy 几何模块中的多边形类：Polygon, RegularPolygon, Triangle，以及角度单位转换函数：rad, deg
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle, rad, deg
# 导入 SymPy 几何模块中的一些实用函数：are_similar, centroid, convex_hull, idiff,
# intersection, closest_points, farthest_points
from sympy.geometry.util import are_similar, centroid, convex_hull, idiff, \
    intersection, closest_points, farthest_points
# 导入 SymPy 几何模块中的异常类：GeometryError
from sympy.geometry.exceptions import GeometryError
# 导入 SymPy 几何模块中的曲线类：Curve
from sympy.geometry.curve import Curve
# 导入 SymPy 几何模块中的抛物线类：Parabola
from sympy.geometry.parabola import Parabola

# 定义 __all__ 列表，用于指定模块中哪些对象可以被 from module import * 导入
__all__ = [
    'Point', 'Point2D', 'Point3D',

    'Line', 'Ray', 'Segment', 'Line2D', 'Segment2D', 'Ray2D', 'Line3D',
    'Segment3D', 'Ray3D',

    'Plane',

    'Ellipse', 'Circle',

    'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',

    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points',

    'GeometryError',

    'Curve',

    'Parabola',
]
```