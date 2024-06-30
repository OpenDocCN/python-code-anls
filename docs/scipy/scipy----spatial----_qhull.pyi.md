# `D:\src\scipysrc\scipy\scipy\spatial\_qhull.pyi`

```
'''
Static type checking stub file for scipy/spatial/qhull.pyx
'''

# 导入需要的模块
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import final

# 定义 QhullError 异常类，继承自 RuntimeError
class QhullError(RuntimeError):
    ...

# 使用 @final 装饰器标记 _Qhull 类，表示它是一个不可继承的最终类
@final
class _Qhull:
    # 定义一个只读的 Cython 属性，类似于属性的行为
    @property
    def ndim(self) -> int: ...

    # 类的属性定义
    mode_option: bytes
    options: bytes
    furthest_site: bool

    # _Qhull 类的初始化方法
    def __init__(
        self,
        mode_option: bytes,
        points: NDArray[np.float64],
        options: None | bytes = ...,
        required_options: None | bytes = ...,
        furthest_site: bool = ...,
        incremental: bool = ...,
        interior_point: None | NDArray[np.float64] = ...,
    ) -> None: ...

    # 检查活动状态的方法
    def check_active(self) -> None: ...

    # 关闭方法
    def close(self) -> None: ...

    # 获取点集的方法
    def get_points(self) -> NDArray[np.float64]: ...

    # 添加点集的方法
    def add_points(
        self,
        points: ArrayLike,
        interior_point: ArrayLike = ...
    ) -> None: ...

    # 获取抛物面转换的偏移和缩放比例的方法
    def get_paraboloid_shift_scale(self) -> tuple[float, float]: ...

    # 获取体积和面积的方法
    def volume_area(self) -> tuple[float, float]: ...

    # 三角化的方法
    def triangulate(self) -> None: ...

    # 获取简单形状面片数组的方法
    def get_simplex_facet_array(self) -> tuple[
        NDArray[np.intc],
        NDArray[np.intc],
        NDArray[np.float64],
        NDArray[np.intc],
        NDArray[np.intc],
    ]: ...

    # 获取凸壳点集的方法
    def get_hull_points(self) -> NDArray[np.float64]: ...

    # 获取凸壳面片的方法
    def get_hull_facets(self) -> tuple[
        list[list[int]],
        NDArray[np.float64],
    ]: ...

    # 获取 Voronoi 图的方法
    def get_voronoi_diagram(self) -> tuple[
        NDArray[np.float64],
        NDArray[np.intc],
        list[list[int]],
        list[list[int]],
        NDArray[np.intp],
    ]: ...

    # 获取二维极值的方法
    def get_extremes_2d(self) -> NDArray[np.intc]: ...

# 定义 _get_barycentric_transforms 函数
def _get_barycentric_transforms(
    points: NDArray[np.float64],
    simplices: NDArray[np.intc],
    eps: float
) -> NDArray[np.float64]: ...

# 定义 _QhullUser 类
class _QhullUser:
    # 类的属性定义
    ndim: int
    npoints: int
    min_bound: NDArray[np.float64]
    max_bound: NDArray[np.float64]

    # _QhullUser 类的初始化方法
    def __init__(self, qhull: _Qhull, incremental: bool = ...) -> None: ...

    # 关闭方法
    def close(self) -> None: ...

    # 更新方法
    def _update(self, qhull: _Qhull) -> None: ...

    # 添加点集的方法
    def _add_points(
        self,
        points: ArrayLike,
        restart: bool = ...,
        interior_point: ArrayLike = ...
    ) -> None: ...

# 继承自 _QhullUser 类的 Delaunay 类
class Delaunay(_QhullUser):
    # 类的属性定义
    furthest_site: bool
    paraboloid_scale: float
    paraboloid_shift: float
    simplices: NDArray[np.intc]
    neighbors: NDArray[np.intc]
    equations: NDArray[np.float64]
    coplanar: NDArray[np.intc]
    good: NDArray[np.intc]
    nsimplex: int
    vertices: NDArray[np.intc]

    # Delaunay 类的初始化方法
    def __init__(
        self,
        points: ArrayLike,
        furthest_site: bool = ...,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None: ...

    # 更新方法
    def _update(self, qhull: _Qhull) -> None: ...

    # 添加点集的方法
    def add_points(
        self,
        points: ArrayLike,
        restart: bool = ...
    ) -> None: ...
    # 定义一个属性，返回类型为 NDArray[np.float64] 的点集合
    @property
    def points(self) -> NDArray[np.float64]: ...
    
    # 定义一个属性，返回类型为 NDArray[np.float64] 的变换矩阵
    @property
    def transform(self) -> NDArray[np.float64]: ...
    
    # 定义一个属性，返回类型为 NDArray[np.intc] 的顶点到简单形式的映射
    @property
    def vertex_to_simplex(self) -> NDArray[np.intc]: ...
    
    # 定义一个属性，返回类型为元组，包含两个 NDArray[np.intc] 类型的顶点邻居映射
    @property
    def vertex_neighbor_vertices(self) -> tuple[
        NDArray[np.intc],
        NDArray[np.intc],
    ]: ...
    
    # 定义一个属性，返回类型为 NDArray[np.intc] 的凸包结果
    @property
    def convex_hull(self) -> NDArray[np.intc]: ...
    
    # 定义一个方法，查找简单形式中的一个点所属的简单形式
    def find_simplex(
        self,
        xi: ArrayLike,
        bruteforce: bool = ...,
        tol: float = ...
    ) -> NDArray[np.intc]: ...
    
    # 定义一个方法，计算一个点到平面的距离
    def plane_distance(self, xi: ArrayLike) -> NDArray[np.float64]: ...
    
    # 定义一个方法，将给定点集合进行映射到高维空间
    def lift_points(self, x: ArrayLike) -> NDArray[np.float64]: ...
def tsearch(tri: Delaunay, xi: ArrayLike) -> NDArray[np.intc]:
    ...

# 定义了一个函数 tsearch，用于在 Delaunay 三角剖分中搜索点 xi 的索引，返回一个整数数组

def _copy_docstr(dst: object, src: object) -> None:
    ...

# 定义了一个函数 _copy_docstr，用于将对象 src 的文档字符串复制到对象 dst 上，无返回值

class ConvexHull(_QhullUser):
    simplices: NDArray[np.intc]
    neighbors: NDArray[np.intc]
    equations: NDArray[np.float64]
    coplanar: NDArray[np.intc]
    good: None | NDArray[np.bool_]
    volume: float
    area: float
    nsimplex: int

    def __init__(
        self,
        points: ArrayLike,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None:
        ...

    # ConvexHull 类的构造函数，初始化凸包对象，接受点集 points、增量标志 incremental 和 qhull 选项

    def _update(self, qhull: _Qhull) -> None:
        ...

    # 更新凸包对象的私有方法 _update，接受一个 _Qhull 对象 qhull，无返回值

    def add_points(self, points: ArrayLike,
                   restart: bool = ...) -> None:
        ...

    # 向凸包对象添加点集的方法 add_points，接受点集 points 和重启标志 restart，无返回值

    @property
    def points(self) -> NDArray[np.float64]:
        ...

    # 返回凸包对象的属性 points，是一个浮点数数组

    @property
    def vertices(self) -> NDArray[np.intc]:
        ...

    # 返回凸包对象的属性 vertices，是一个整数数组

class Voronoi(_QhullUser):
    vertices: NDArray[np.float64]
    ridge_points: NDArray[np.intc]
    ridge_vertices: list[list[int]]
    regions: list[list[int]]
    point_region: NDArray[np.intp]
    furthest_site: bool

    def __init__(
        self,
        points: ArrayLike,
        furthest_site: bool = ...,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None:
        ...

    # Voronoi 类的构造函数，初始化 Voronoi 图对象，接受点集 points、最远站点标志 furthest_site、增量标志 incremental 和 qhull 选项

    def _update(self, qhull: _Qhull) -> None:
        ...

    # 更新 Voronoi 图对象的私有方法 _update，接受一个 _Qhull 对象 qhull，无返回值

    def add_points(
        self,
        points: ArrayLike,
        restart: bool = ...
    ) -> None:
        ...

    # 向 Voronoi 图对象添加点集的方法 add_points，接受点集 points 和重启标志 restart，无返回值

    @property
    def points(self) -> NDArray[np.float64]:
        ...

    # 返回 Voronoi 图对象的属性 points，是一个浮点数数组

    @property
    def ridge_dict(self) -> dict[tuple[int, int], list[int]]:
        ...

    # 返回 Voronoi 图对象的属性 ridge_dict，是一个从边对 (两个顶点索引) 到顶点列表的字典

class HalfspaceIntersection(_QhullUser):
    interior_point: NDArray[np.float64]
    dual_facets: list[list[int]]
    dual_equations: NDArray[np.float64]
    dual_points: NDArray[np.float64]
    dual_volume: float
    dual_area: float
    intersections: NDArray[np.float64]
    ndim: int
    nineq: int

    def __init__(
        self,
        halfspaces: ArrayLike,
        interior_point: ArrayLike,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None:
        ...

    # HalfspaceIntersection 类的构造函数，初始化半空间交点对象，接受半空间描述 halfspaces、内部点 interior_point、增量标志 incremental 和 qhull 选项

    def _update(self, qhull: _Qhull) -> None:
        ...

    # 更新半空间交点对象的私有方法 _update，接受一个 _Qhull 对象 qhull，无返回值

    def add_halfspaces(
        self,
        halfspaces: ArrayLike,
        restart: bool = ...
    ) -> None:
        ...

    # 向半空间交点对象添加半空间的方法 add_halfspaces，接受半空间描述 halfspaces 和重启标志 restart，无返回值

    @property
    def halfspaces(self) -> NDArray[np.float64]:
        ...

    # 返回半空间交点对象的属性 halfspaces，是一个浮点数数组

    @property
    def dual_vertices(self) -> NDArray[np.integer]:
        ...

    # 返回半空间交点对象的属性 dual_vertices，是一个整数数组
```