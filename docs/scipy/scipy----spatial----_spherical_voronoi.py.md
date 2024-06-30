# `D:\src\scipysrc\scipy\scipy\spatial\_spherical_voronoi.py`

```
"""
Spherical Voronoi Code

.. versionadded:: 0.18.0

"""
#
# Copyright (C)  Tyler Reddy, Ross Hemsley, Edd Edmondson,
#                Nikolai Nowaczyk, Joe Pitt-Francis, 2015.
#
# Distributed under the same BSD license as SciPy.
#

import numpy as np                        # 导入NumPy库
import scipy                              # 导入SciPy库
from . import _voronoi                    # 导入当前目录下的_voronoi模块
from scipy.spatial import cKDTree         # 导入SciPy的cKDTree类

__all__ = ['SphericalVoronoi']


def calculate_solid_angles(R):
    """Calculates the solid angles of plane triangles. Implements the method of
    Van Oosterom and Strackee [VanOosterom]_ with some modifications. Assumes
    that input points have unit norm."""
    # 计算平面三角形的立体角。使用Van Oosterom和Strackee的方法，并进行了一些修改。
    # 假设输入的点具有单位范数。

    # 计算三角形的顶点坐标行列式作为分子
    numerator = np.linalg.det(R)

    # 计算分母，使用内积和来计算
    denominator = 1 + (np.einsum('ij,ij->i', R[:, 0], R[:, 1]) +
                       np.einsum('ij,ij->i', R[:, 1], R[:, 2]) +
                       np.einsum('ij,ij->i', R[:, 2], R[:, 0]))

    return np.abs(2 * np.arctan2(numerator, denominator))


class SphericalVoronoi:
    """ Voronoi diagrams on the surface of a sphere.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points from which to construct a spherical
        Voronoi diagram.
    radius : float, optional
        Radius of the sphere (Default: 1)
    center : ndarray of floats, shape (ndim,)
        Center of sphere (Default: origin)
    threshold : float
        Threshold for detecting duplicate points and
        mismatches between points and sphere parameters.
        (Default: 1e-06)

    Attributes
    ----------
    points : double array of shape (npoints, ndim)
        the points in `ndim` dimensions to generate the Voronoi diagram from
    radius : double
        radius of the sphere
    center : double array of shape (ndim,)
        center of the sphere
    vertices : double array of shape (nvertices, ndim)
        Voronoi vertices corresponding to points
    regions : list of list of integers of shape (npoints, _ )
        the n-th entry is a list consisting of the indices
        of the vertices belonging to the n-th point in points

    Methods
    -------
    calculate_areas
        Calculates the areas of the Voronoi regions. For 2D point sets, the
        regions are circular arcs. The sum of the areas is ``2 * pi * radius``.
        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is ``4 * pi * radius**2``.

    Raises
    ------
    ValueError
        If there are duplicates in `points`.
        If the provided `radius` is not consistent with `points`.

    Notes
    -----
    The spherical Voronoi diagram algorithm proceeds as follows. The Convex
    Hull of the input points (generators) is calculated, and is equivalent to
    their Delaunay triangulation on the surface of the sphere [Caroli]_.

    """
    # SphericalVoronoi类，用于在球面上生成Voronoi图

    def __init__(self, points, radius=1, center=None, threshold=1e-06):
        """Initialize the SphericalVoronoi instance with the given points.

        Parameters
        ----------
        points : ndarray of floats, shape (npoints, ndim)
            Coordinates of points from which to construct a spherical
            Voronoi diagram.
        radius : float, optional
            Radius of the sphere (Default: 1)
        center : ndarray of floats, shape (ndim,)
            Center of sphere (Default: origin)
        threshold : float
            Threshold for detecting duplicate points and
            mismatches between points and sphere parameters.
            (Default: 1e-06)
        """
        self.points = np.asarray(points)        # 将输入点转换为NumPy数组
        self.radius = float(radius)             # 球的半径
        self.center = np.zeros_like(self.points[0]) if center is None else center  # 球的中心，默认为原点或用户指定
        self.threshold = threshold              # 阈值，用于检测重复点和点与球参数的不匹配

        if len(self.points) != len(set(map(tuple, self.points))):
            raise ValueError("Duplicate points are not allowed.")  # 检测是否有重复点，如果有则引发异常

        if not np.allclose(np.linalg.norm(self.points, axis=1), 1.0, atol=self.threshold):
            raise ValueError("Points should lie on the surface of the unit sphere.")  # 检测点是否在单位球面上，如果不在则引发异常

        # 计算 Voronoi 图的顶点和区域
        self.vertices, self.regions = _voronoi.compute_voronoi(self.points.tolist(), self.radius, self.center.tolist())

    def calculate_areas(self):
        """Calculates the areas of the Voronoi regions.

        Raises
        ------
        ValueError
            If the dimension of the Voronoi diagram is not 2 or 3.

        Returns
        -------
        areas : ndarray
            Array containing the areas of each Voronoi region.
        """
        # 计算 Voronoi 区域的面积
        if self.points.shape[1] == 2:
            # 对于2D点集，区域是圆弧，总面积为2 * pi * radius
            areas = np.pi * self.radius * self.radius * np.ones(len(self.regions))
        elif self.points.shape[1] == 3:
            # 对于3D点集，区域是球面多边形，总面积为4 * pi * radius^2
            areas = 2 * np.pi * self.radius * np.ones(len(self.regions))
        else:
            raise ValueError("Unsupported dimension for Voronoi diagram.")  # 不支持的维度，引发异常

        return areas

    # 添加其他方法和文档说明，可以继续在这里进行补充
    The Convex Hull neighbour information is then used to
    order the Voronoi region vertices around each generator. The latter
    approach is substantially less sensitive to floating point issues than
    angle-based methods of Voronoi region vertex sorting.

    Empirical assessment of spherical Voronoi algorithm performance suggests
    quadratic time complexity (loglinear is optimal, but algorithms are more
    challenging to implement).

    References
    ----------
    .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of
                points on or close to a sphere. Research Report RR-7004, 2009.

    .. [VanOosterom] Van Oosterom and Strackee. The solid angle of a plane
                     triangle. IEEE Transactions on Biomedical Engineering,
                     2, 1983, pp 125--126.

    See Also
    --------
    Voronoi : Conventional Voronoi diagrams in N dimensions.

    Examples
    --------
    Do some imports and take some points on a cube:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import SphericalVoronoi, geometric_slerp
    >>> from mpl_toolkits.mplot3d import proj3d
    >>> # set input data
    >>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
    ...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])

    Calculate the spherical Voronoi diagram:

    >>> radius = 1
    >>> center = np.array([0, 0, 0])
    >>> sv = SphericalVoronoi(points, radius, center)

    Generate plot:

    >>> # sort vertices (optional, helpful for plotting)
    >>> sv.sort_vertices_of_regions()
    >>> t_vals = np.linspace(0, 1, 2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # plot the unit sphere for reference (optional)
    >>> u = np.linspace(0, 2 * np.pi, 100)
    >>> v = np.linspace(0, np.pi, 100)
    >>> x = np.outer(np.cos(u), np.sin(v))
    >>> y = np.outer(np.sin(u), np.sin(v))
    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
    >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)
    >>> # plot generator points
    >>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    >>> # plot Voronoi vertices
    >>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
    ...                    c='g')
    >>> # indicate Voronoi regions (as Euclidean polygons)
    >>> for region in sv.regions:
    ...    n = len(region)
    ...    for i in range(n):
    ...        start = sv.vertices[region][i]
    ...        end = sv.vertices[region][(i + 1) % n]
    ...        result = geometric_slerp(start, end, t_vals)
    ...        ax.plot(result[..., 0],
    ...                result[..., 1],
    ...                result[..., 2],
    ...                c='k')
    >>> ax.azim = 10
    >>> ax.elev = 40
    >>> _ = ax.set_xticks([])
    >>> _ = ax.set_yticks([])
    >>> _ = ax.set_zticks([])
    >>> fig.set_size_inches(4, 4)
    >>> plt.show()

    """
    def __init__(self, points, radius=1, center=None, threshold=1e-06):
        # 如果 radius 参数为 None，则抛出数值错误异常，要求提供浮点数半径值
        if radius is None:
            raise ValueError('`radius` is `None`. '
                             'Please provide a floating point number '
                             '(i.e. `radius=1`).')

        # 将 radius 参数转换为浮点数并存储在实例属性 self.radius 中
        self.radius = float(radius)
        
        # 将 points 转换为 numpy 数组，并强制转换为 np.float64 类型，存储在 self.points 中
        self.points = np.array(points).astype(np.float64)
        
        # 获取点集的维度信息，并存储在 self._dim 中
        self._dim = self.points.shape[1]
        
        # 如果 center 参数为 None，则将 self.center 初始化为零向量；否则，将 center 转换为浮点数 numpy 数组并存储在 self.center 中
        if center is None:
            self.center = np.zeros(self._dim)
        else:
            self.center = np.array(center, dtype=float)

        # 检测输入点集是否退化（线性相关性检测）
        self._rank = np.linalg.matrix_rank(self.points - self.points[0],
                                           tol=threshold * self.radius)
        # 如果点集退化，抛出数值错误异常，指出点集的秩必须至少为 self._dim
        if self._rank < self._dim:
            raise ValueError(f"Rank of input points must be at least {self._dim}")

        # 使用 cKDTree 检测是否存在重复的生成器点
        if cKDTree(self.points).query_pairs(threshold * self.radius):
            raise ValueError("Duplicate generators present.")

        # 计算所有点到中心的距离，并检查距离是否与半径一致
        radii = np.linalg.norm(self.points - self.center, axis=1)
        max_discrepancy = np.abs(radii - self.radius).max()
        # 如果最大不一致距离超过阈值乘以半径，则抛出数值错误异常，指出半径与生成器不一致
        if max_discrepancy >= threshold * self.radius:
            raise ValueError("Radius inconsistent with generators.")

        # 调用 _calc_vertices_regions 方法计算 Voronoi 顶点和区域
        self._calc_vertices_regions()

    def _calc_vertices_regions(self):
        """
        Calculates the Voronoi vertices and regions of the generators stored
        in self.points. The vertices will be stored in self.vertices and the
        regions in self.regions.

        This algorithm was discussed at PyData London 2015 by
        Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk
        """
        # 计算点集的凸包
        conv = scipy.spatial.ConvexHull(self.points)
        
        # 通过凸包的面方程计算凸包的外接圆心，存储在 self.vertices 中
        # 对于 3D 输入，circumcenters 的形状为 (2N-4, 3)
        self.vertices = self.radius * conv.equations[:, :-1] + self.center
        
        # 存储凸包的单纯形（三角形）列表在 self._simplices 中
        self._simplices = conv.simplices
        
        # 根据三角化计算区域
        # 对于 3D 输入，simplex_indices 的形状为 (2N-4,)
        simplex_indices = np.arange(len(self._simplices))
        
        # 对于 3D 输入，tri_indices 的形状为 (6N-12,)
        tri_indices = np.column_stack([simplex_indices] * self._dim).ravel()
        
        # 对于 3D 输入，point_indices 的形状为 (6N-12,)
        point_indices = self._simplices.ravel()
        
        # 对于 3D 输入，indices 的形状为 (6N-12,)
        indices = np.argsort(point_indices, kind='mergesort')
        
        # 对于 3D 输入，flattened_groups 的形状为 (6N-12,)
        flattened_groups = tri_indices[indices].astype(np.intp)
        
        # intervals 的形状为 (N+1,)
        intervals = np.cumsum(np.bincount(point_indices + 1))
        
        # 将 flattened_groups 拆分为未排序区域的嵌套列表，存储在 self.regions 中
        groups = [list(flattened_groups[intervals[i]:intervals[i + 1]])
                  for i in range(len(intervals) - 1)]
        self.regions = groups
    def sort_vertices_of_regions(self):
        """
        Sort indices of the vertices to be (counter-)clockwise ordered.

        Raises
        ------
        TypeError
            If the points are not three-dimensional.

        Notes
        -----
        For each region in regions, it sorts the indices of the Voronoi
        vertices such that the resulting points are in a clockwise or
        counterclockwise order around the generator point.

        This is done as follows: Recall that the n-th region in regions
        surrounds the n-th generator in points and that the k-th
        Voronoi vertex in vertices is the circumcenter of the k-th triangle
        in self._simplices.  For each region n, we choose the first triangle
        (=Voronoi vertex) in self._simplices and a vertex of that triangle
        not equal to the center n. These determine a unique neighbor of that
        triangle, which is then chosen as the second triangle. The second
        triangle will have a unique vertex not equal to the current vertex or
        the center. This determines a unique neighbor of the second triangle,
        which is then chosen as the third triangle and so forth. We proceed
        through all the triangles (=Voronoi vertices) belonging to the
        generator in points and obtain a sorted version of the vertices
        of its surrounding region.
        """

        # 检查点集是否为三维，如果不是则抛出 TypeError 异常
        if self._dim != 3:
            raise TypeError("Only supported for three-dimensional point sets")
        
        # 调用 _voronoi 模块的方法，对每个区域中的顶点索引进行排序，
        # 以使结果顶点按照顺时针或逆时针的顺序围绕生成点排列
        _voronoi.sort_vertices_of_regions(self._simplices, self.regions)
    # 计算 3D 区域的表面积
    def _calculate_areas_3d(self):
        # 对区域的顶点进行排序
        self.sort_vertices_of_regions()
        # 计算每个区域的顶点数
        sizes = [len(region) for region in self.regions]
        # 计算区域顶点数的累积和
        csizes = np.cumsum(sizes)
        # 区域的总数
        num_regions = csizes[-1]

        # 创建由一个点和两个Voronoi顶点组成的三角形集合。
        # 每个三角形的顶点在排序后的区域列表中是相邻的。
        point_indices = [i for i, size in enumerate(sizes)
                         for j in range(size)]

        # 创建邻居数组 nbrs1，包含所有区域的顶点
        nbrs1 = np.array([r for region in self.regions for r in region])

        # 使用向量化的方式计算 nbrs2，其等效于：
        # np.array([r for region in self.regions for r in np.roll(region, 1)])
        nbrs2 = np.roll(nbrs1, 1)
        indices = np.roll(csizes, 1)
        indices[0] = 0
        nbrs2[indices] = nbrs1[csizes - 1]

        # 将点和顶点进行归一化处理
        pnormalized = (self.points - self.center) / self.radius
        vnormalized = (self.vertices - self.center) / self.radius

        # 创建完整的三角形集合并计算它们的固体角
        triangles = np.hstack([pnormalized[point_indices],
                               vnormalized[nbrs1],
                               vnormalized[nbrs2]
                               ]).reshape((num_regions, 3, 3))
        triangle_solid_angles = calculate_solid_angles(triangles)

        # 计算每个区域内三角形的固体角之和
        solid_angles = np.cumsum(triangle_solid_angles)[csizes - 1]
        solid_angles[1:] -= solid_angles[:-1]

        # 使用 A = omega * r**2 计算多边形的面积
        return solid_angles * self.radius**2

    # 计算 2D 区域的面积
    def _calculate_areas_2d(self):
        # 找到弧的起始点和结束点
        arcs = self.points[self._simplices] - self.center

        # 计算弧所对应的角度
        d = np.sum((arcs[:, 1] - arcs[:, 0]) ** 2, axis=1)
        theta = np.arccos(1 - (d / (2 * (self.radius ** 2))))

        # 使用 A = r * theta 计算区域的面积
        areas = self.radius * theta

        # 修正走错方向的弧（单半球输入）
        signs = np.sign(np.einsum('ij,ij->i', arcs[:, 0],
                                              self.vertices - self.center))
        indices = np.where(signs < 0)
        areas[indices] = 2 * np.pi * self.radius - areas[indices]
        return areas
    # 定义方法用于计算 Voronoi 区域的面积

    """Calculates the areas of the Voronoi regions.
    
    For 2D point sets, the regions are circular arcs. The sum of the areas
    is ``2 * pi * radius``.

    For 3D point sets, the regions are spherical polygons. The sum of the
    areas is ``4 * pi * radius**2``.

    .. versionadded:: 1.5.0

    Returns
    -------
    areas : double array of shape (npoints,)
        The areas of the Voronoi regions.
    """

    # 检查点集的维度，如果是二维则调用对应的计算方法
    if self._dim == 2:
        return self._calculate_areas_2d()
    # 如果是三维则调用三维的计算方法
    elif self._dim == 3:
        return self._calculate_areas_3d()
    # 如果维度不是二维或三维，则抛出类型错误异常
    else:
        raise TypeError("Only supported for 2D and 3D point sets")
```