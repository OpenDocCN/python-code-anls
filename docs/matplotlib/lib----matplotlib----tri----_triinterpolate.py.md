# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triinterpolate.py`

```py
"""
Interpolation inside triangular grids.
"""

import numpy as np  # 导入 NumPy 库

from matplotlib import _api  # 导入 Matplotlib 内部 API
from matplotlib.tri import Triangulation  # 导入 Triangulation 类
from matplotlib.tri._trifinder import TriFinder  # 导入 TriFinder 类
from matplotlib.tri._tritools import TriAnalyzer  # 导入 TriAnalyzer 类

__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')

class TriInterpolator:
    """
    Abstract base class for classes used to interpolate on a triangular grid.

    Derived classes implement the following methods:

    - ``__call__(x, y)``,
      where x, y are array-like point coordinates of the same shape, and
      that returns a masked array of the same shape containing the
      interpolated z-values.

    - ``gradient(x, y)``,
      where x, y are array-like point coordinates of the same
      shape, and that returns a list of 2 masked arrays of the same shape
      containing the 2 derivatives of the interpolator (derivatives of
      interpolated z values with respect to x and y).
    """

    def __init__(self, triangulation, z, trifinder=None):
        _api.check_isinstance(Triangulation, triangulation=triangulation)  # 检查 triangulation 是否为 Triangulation 类的实例
        self._triangulation = triangulation  # 初始化三角网格对象

        self._z = np.asarray(z)  # 将 z 转换为 NumPy 数组
        if self._z.shape != self._triangulation.x.shape:
            raise ValueError("z array must have same length as triangulation x"
                             " and y arrays")  # 如果 z 的形状与三角网格的 x 和 y 数组不匹配，则引发值错误异常

        _api.check_isinstance((TriFinder, None), trifinder=trifinder)  # 检查 trifinder 是否为 TriFinder 类或 None 类型
        self._trifinder = trifinder or self._triangulation.get_trifinder()  # 设置 trifinder，如果未提供，则使用三角网格的默认 trifinder

        # 默认的缩放因子为 1.0 （无缩放）
        # 缩放可用于某些插值，其中 x、y 的数量级影响插值器的定义。
        # 详细信息请参阅 _interpolate_multikeys 方法。
        self._unit_x = 1.0
        self._unit_y = 1.0

        # 默认的三角形重新编号为 None （无重新编号）
        # 如果在插值器内执行复杂计算，则可使用重新编号来避免不必要的计算。
        # 详细信息请参阅 _interpolate_multikeys 方法。
        self._tri_renum = None

    # __call__ and gradient docstrings are shared by all subclasses
    # (except, if needed, relevant additions).
    # However these methods are only implemented in subclasses to avoid
    # confusion in the documentation.
    _docstring__call__ = """
        Returns a masked array containing interpolated values at the specified
        (x, y) points.

        Parameters
        ----------
        x, y : array-like
            x and y coordinates of the same shape and any number of
            dimensions.

        Returns
        -------
        np.ma.array
            Masked array of the same shape as *x* and *y*; values corresponding
            to (*x*, *y*) points outside of the triangulation are masked out.

        """
    """
    Returns a list of 2 masked arrays containing interpolated derivatives
    at the specified (x, y) points.

    Parameters
    ----------
    x, y : array-like
        x and y coordinates of the same shape and any number of
        dimensions.

    Returns
    -------
    dzdx, dzdy : np.ma.array
        2 masked arrays of the same shape as *x* and *y*; values
        corresponding to (x, y) points outside of the triangulation
        are masked out.
        The first returned array contains the values of
        :math:`\frac{\partial z}{\partial x}` and the second those of
        :math:`\frac{\partial z}{\partial y}`.

    """

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        """
        Interpolate at points belonging to the triangulation
        (inside an unmasked triangles).

        Parameters
        ----------
        return_key : {'z', 'dzdx', 'dzdy'}
            The requested values (z or its derivatives).
        tri_index : 1D int array
            Valid triangle index (cannot be -1).
        x, y : 1D arrays, same shape as `tri_index`
            Valid locations where interpolation is requested.

        Returns
        -------
        1-d array
            Returned array of the same size as *tri_index*
        """
        raise NotImplementedError("TriInterpolator subclasses" +
                                  "should implement _interpolate_single_key!")
    """
    Cubic interpolator on a triangular grid.

    In one-dimension - on a segment - a cubic interpolating function is
    defined by the values of the function and its derivative at both ends.
    This is almost the same in 2D inside a triangle, except that the values
    of the function and its 2 derivatives have to be defined at each triangle
    node.

    The CubicTriInterpolator takes the value of the function at each node -
    provided by the user - and internally computes the value of the
    derivatives, resulting in a smooth interpolation.
    (As a special feature, the user can also impose the value of the
    """
    def __init__(self, triangulation, z, trifinder=None):
        """
        Constructor for CubicTriInterpolator.

        Parameters
        ----------
        triangulation : `~matplotlib.tri.Triangulation`
            The triangulation to interpolate over.
        z : (npoints,) array-like
            Array of values, defined at grid points, to interpolate between.
        trifinder : `~matplotlib.tri.TriFinder`, optional
            If this is not specified, the Triangulation's default TriFinder will
            be used by calling `.Triangulation.get_trifinder`.
        """
        super().__init__(triangulation, z, trifinder)

        # Calculate derivatives at each node to achieve smooth cubic interpolation.
        self._derivatives = self._triangulation.calculate_derivatives(self._z)

    def __call__(self, x, y):
        """
        Interpolate values at specified points (x, y).

        Parameters
        ----------
        x : float or array-like
            X-coordinates of the points to interpolate.
        y : float or array-like
            Y-coordinates of the points to interpolate.

        Returns
        -------
        array-like
            Interpolated values at points (x, y).
        """
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('z',))[0]
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        """
        Calculate gradients at specified points (x, y).

        Parameters
        ----------
        x : float or array-like
            X-coordinates of the points to calculate gradients.
        y : float or array-like
            Y-coordinates of the points to calculate gradients.

        Returns
        -------
        tuple
            Tuple of gradients (dzdx, dzdy) at points (x, y).
        """
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('dzdx', 'dzdy'))
    gradient.__doc__ = TriInterpolator._docstring__gradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        """
        Perform interpolation for a single return key at specified points.

        Parameters
        ----------
        return_key : str
            Key specifying what to interpolate ('z' for value, 'dzdx' for x-derivative,
            'dzdy' for y-derivative).
        tri_index : int
            Index of the triangle in which to perform interpolation.
        x : float or array-like
            X-coordinates of the points to interpolate.
        y : float or array-like
            Y-coordinates of the points to interpolate.

        Returns
        -------
        float or array-like
            Interpolated value or derivative at points (x, y).
        """
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        if return_key == 'z':
            return (self._plane_coefficients[tri_index, 0]*x +
                    self._plane_coefficients[tri_index, 1]*y +
                    self._plane_coefficients[tri_index, 2])
        elif return_key == 'dzdx':
            return self._plane_coefficients[tri_index, 0]
        else:  # 'dzdy'
            return self._plane_coefficients[tri_index, 1]
    # 用于进行三角网格上的三次样条插值，返回一个可调用对象和其梯度
    def CubicTriInterpolator(triangulation, z, kind='min_E', trifinder=None, dz=None):
        # 选择平滑算法以计算插值函数的导数，默认为'min_E'
        # - 'min_E': 最小化弯曲能量以计算每个节点的导数（默认）
        # - 'geom': 每个节点的导数被计算为相关三角形法线的加权平均，适用于速度优化（大网格）
        # - 'user': 用户提供参数*dz*，无需计算
        Parameters
        ----------
        triangulation : `~matplotlib.tri.Triangulation`
            用于插值的三角网格
        z : (npoints,) array-like
            在网格点上定义的插值数值数组
        kind : {'min_E', 'geom', 'user'}, optional
            平滑算法的选择，默认为'min_E'：
            
            - 如果为'min_E'（默认）：每个节点的导数被计算以最小化弯曲能量。
            - 如果为'geom'：每个节点的导数被计算为相关三角形法线的加权平均，用于速度优化（大网格）。
            - 如果为'user'：用户提供参数*dz*，无需进行计算。
        trifinder : `~matplotlib.tri.TriFinder`, optional
            如果未指定，则使用Triangulation的默认TriFinder，通过调用`.Triangulation.get_trifinder`获取。
        dz : tuple of array-likes (dzdx, dzdy), optional
            仅在*kind* ='user'时使用。在这种情况下，必须提供*dz*，形式为(dzdx, dzdy)，其中dzdx、dzdy与*z*具有相同的形状，
            是*triangulation*点处的插值一阶导数。
    
        Methods
        -------
        `__call__` (x, y) : 返回在(x, y)点处的插值数值。
        `gradient` (x, y) : 返回在(x, y)点处的插值导数。
    
        Notes
        -----
        这段注释有点技术性，详细描述了如何计算三次插值。
        
        插值基于*triangulation*网格的Clough-Tocher细分方案
        （更明确地说，网格的每个三角形将分成3个子三角形，并且在每个子三角形上，
        插值函数是2个坐标的三次多项式）。
        这种技术源自有限元方法（FEM）分析；所使用的元素是减少的Hsieh-Clough-Tocher（HCT）元素。
        其形状函数在[1]_中描述。
        组装的函数保证是C1光滑的，即连续且其一阶导数也连续
        （在三角形内部很容易证明，但在越过边界时也是如此）。
        
        在默认情况下（*kind* ='min_E'），插值最小化了由HCT元素形状函数生成的功能空间上的曲率能量
        - 以强加值但任意节点的导数。被最小化的功能是所谓的总曲率的积分
        （基于来自[2]_的算法 - PCG稀疏求解器）。
    """
    Initialize a class instance for solving problems related to triangular meshes with associated data.

    Parameters
    ----------
    triangulation : Triangulation object
        The triangulation structure defining the mesh.
    z : array_like
        Array of values associated with each node of the mesh.
    kind : {'min_E', 'geom', 'user'}, optional
        Specifies the type of problem to solve (default is 'min_E').
        - 'min_E': Minimize energy as defined by the integral.
        - 'geom': Use geometric approximation for speed.
        - 'user': Custom problem type specified by the user.
    trifinder : TriFinder instance, optional
        Object providing triangle search capabilities.
    dz : array_like, optional
        Additional data associated with nodes, if applicable.

    Notes
    -----
    This constructor initializes various attributes and computations required
    for solving the problem on the given mesh. It involves:
    - Loading the underlying C++ triangulation.
    - Compressing and renumbering triangles and nodes for efficiency.
    - Computing scale factors based on compressed coordinates.
    - Computing eccentricities and degrees of freedom estimations based on the
      chosen problem type.

    References
    ----------
    [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
        Hsieh-Clough-Tocher triangles, complete or reduced.",
        International Journal for Numerical Methods in Engineering,
        17(5):784 - 789. 2.01.
    [2] C.T. Kelley, "Iterative Methods for Optimization".
    """
    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
                 dz=None):
        super().__init__(triangulation, z, trifinder)

        # Loads the underlying C++ _triangulation.
        # (During loading, reordering of triangulation._triangles may occur so
        # that all final triangles are now anti-clockwise)
        self._triangulation.get_cpp_triangulation()

        # To build the stiffness matrix and avoid zero-energy spurious modes
        # we will only store internally the valid (unmasked) triangles and
        # the necessary (used) points coordinates.
        # 2 renumbering tables need to be computed and stored:
        #  - a triangle renum table in order to translate the result from a
        #    TriFinder instance into the internal stored triangle number.
        #  - a node renum table to overwrite the self._z values into the new
        #    (used) node numbering.
        tri_analyzer = TriAnalyzer(self._triangulation)
        (compressed_triangles, compressed_x, compressed_y, tri_renum,
         node_renum) = tri_analyzer._get_compressed_triangulation()
        self._triangles = compressed_triangles
        self._tri_renum = tri_renum

        # Taking into account the node renumbering in self._z:
        valid_node = (node_renum != -1)
        self._z[node_renum[valid_node]] = self._z[valid_node]

        # Computing scale factors
        self._unit_x = np.ptp(compressed_x)
        self._unit_y = np.ptp(compressed_y)
        self._pts = np.column_stack([compressed_x / self._unit_x,
                                     compressed_y / self._unit_y])

        # Computing triangle points
        self._tris_pts = self._pts[self._triangles]

        # Computing eccentricities
        self._eccs = self._compute_tri_eccentricities(self._tris_pts)

        # Computing dof estimations for HCT triangle shape function
        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
        self._dof = self._compute_dof(kind, dz=dz)

        # Loading HCT element
        self._ReferenceElement = _ReducedHCT_Element()
    def __call__(self, x, y):
        # 调用对象实例时的方法，通过调用_interpolate_multikeys方法获取插值结果中的'z'值
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('z',))[0]
    # 设置__call__方法的文档字符串为TriInterpolator._docstring__call__的内容
    __call__.__doc__ = TriInterpolator._docstring__call__

    def gradient(self, x, y):
        # 计算对象实例的梯度，通过调用_interpolate_multikeys方法获取插值结果中的'dzdx'和'dzdy'值
        return self._interpolate_multikeys(x, y, tri_index=None,
                                           return_keys=('dzdx', 'dzdy'))
    # 设置gradient方法的文档字符串为TriInterpolator._docstringgradient的内容
    gradient.__doc__ = TriInterpolator._docstringgradient

    def _interpolate_single_key(self, return_key, tri_index, x, y):
        # 检查return_key是否在['z', 'dzdx', 'dzdy']中
        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
        # 获取三角形索引对应的三角形顶点
        tris_pts = self._tris_pts[tri_index]
        # 获取alpha向量，用于计算插值权重
        alpha = self._get_alpha_vec(x, y, tris_pts)
        # 获取三角形索引对应的偏心率ecc
        ecc = self._eccs[tri_index]
        # 获取三角形索引对应的自由度dof，并扩展为二维数组
        dof = np.expand_dims(self._dof[tri_index], axis=1)
        if return_key == 'z':
            # 如果return_key为'z'，返回参考元素中的函数值
            return self._ReferenceElement.get_function_values(
                alpha, ecc, dof)
        else:  # 'dzdx', 'dzdy'
            # 计算Jacobian矩阵J
            J = self._get_jacobian(tris_pts)
            # 获取参考元素中的函数导数值
            dzdx = self._ReferenceElement.get_function_derivatives(
                alpha, J, ecc, dof)
            if return_key == 'dzdx':
                # 如果return_key为'dzdx'，返回导数值的x分量
                return dzdx[:, 0, 0]
            else:
                # 否则返回导数值的y分量
                return dzdx[:, 1, 0]

    def _compute_dof(self, kind, dz=None):
        """
        Compute and return nodal dofs according to kind.

        Parameters
        ----------
        kind : {'min_E', 'geom', 'user'}
            Choice of the _DOF_estimator subclass to estimate the gradient.
        dz : tuple of array-likes (dzdx, dzdy), optional
            Used only if *kind*=user; in this case passed to the
            :class:`_DOF_estimator_user`.

        Returns
        -------
        array-like, shape (npts, 2)
            Estimation of the gradient at triangulation nodes (stored as
            degree of freedoms of reduced-HCT triangle elements).
        """
        if kind == 'user':
            if dz is None:
                # 如果kind为'user'且dz为None，则引发数值错误
                raise ValueError("For a CubicTriInterpolator with "
                                 "*kind*='user', a valid *dz* "
                                 "argument is expected.")
            # 使用_DOE_estimator_user类并传入当前对象和dz来创建TE对象
            TE = _DOF_estimator_user(self, dz=dz)
        elif kind == 'geom':
            # 如果kind为'geom'，使用_DOE_estimator_geom类创建TE对象
            TE = _DOF_estimator_geom(self)
        else:  # 'min_E', checked in __init__
            # 否则（kind为'min_E'），使用_DOE_estimator_min_E类创建TE对象
            TE = _DOF_estimator_min_E(self)
        # 返回TE对象的compute_dof_from_df()方法计算的自由度
        return TE.compute_dof_from_df()
    @staticmethod
    def _get_alpha_vec(x, y, tris_pts):
        """
        Fast (vectorized) function to compute barycentric coordinates alpha.

        Parameters
        ----------
        x, y : array-like of dim 1 (shape (nx,))
            Coordinates of the points whose points barycentric coordinates are
            requested.
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 2 (shape (nx, 3))
            Barycentric coordinates of the points inside the containing
            triangles.
        """
        # Determine the number of dimensions in tris_pts excluding the last two axes
        ndim = tris_pts.ndim - 2

        # Compute vectors a and b as differences between triangle apex coordinates
        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]

        # Construct the transpose of [a, b] along the last axis
        abT = np.stack([a, b], axis=-1)

        # Transpose abT to match dimensions for matrix multiplication
        ab = _transpose_vectorized(abT)

        # Calculate vector OM as the difference between (x, y) and the first apex of each triangle
        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]

        # Compute the metric tensor as the dot product of ab and its transpose abT
        metric = ab @ abT

        # Handle colinear cases by computing the pseudo-inverse of metric
        # using the Moore-Penrose method, ensuring valid barycentric coordinates
        metric_inv = _pseudo_inv22sym_vectorized(metric)

        # Compute Covar as the dot product of ab and OM transposed
        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))

        # Calculate ksi by multiplying metric_inv with Covar
        ksi = metric_inv @ Covar

        # Calculate alpha matrix as a vectorized operation
        alpha = _to_matrix_vectorized([
            [1 - ksi[:, 0, 0] - ksi[:, 1, 0]],
            [ksi[:, 0, 0]],
            [ksi[:, 1, 0]]
        ])

        # Return the computed barycentric coordinates alpha
        return alpha
    def _compute_tri_eccentricities(tris_pts):
        """
        Compute triangle eccentricities.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the triangles apexes.

        Returns
        -------
        array like of dim 2 (shape: (nx, 3))
            The so-called eccentricity parameters [1] needed for HCT triangular
            element.
        """
        # 计算三角形的边长向量 a, b, c
        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
        
        # 矩阵转置函数，不使用 np.squeeze，因为当三角形只有一个时会有危险
        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
        
        # 注意：如果 dot_a, dot_b 或 dot_c 为零，这行将会产生警告，但我们不支持有重复顶点的三角形
        # 返回三角形的偏心率参数矩阵，用于 HCT 三角元素
        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
                                      [(dot_a-dot_c) / dot_b],
                                      [(dot_b-dot_a) / dot_c]])
# 定义一个类 `_ReducedHCT_Element`，用于实现带有显式形状函数的减少型 HCT 三角元素。

class _ReducedHCT_Element:
    """
    Implementation of reduced HCT triangular element with explicit shape
    functions.

    Computes z, dz, d2z and the element stiffness matrix for bending energy:
    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)

    *** Reference for the shape functions: ***
    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
        reduced.
        Michel Bernadou, Kamal Hassan
        International Journal for Numerical Methods in Engineering.
        17(5):784 - 789.  2.01

    *** Element description: ***
    9 dofs: z and dz given at 3 apex
    C1 (conform)

    """

    # 1) Loads matrices to generate shape functions as a function of
    #    triangle eccentricities - based on [1] p.11 '''
    
    # 定义三个用于生成形状函数的矩阵 M, M0, M1，基于文献 [1] 中的说明。
    M = np.array([
        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
    
    M0 = np.array([
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
    
    M1 = np.array([
        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
    # 定义矩阵 M2，描述了在梯度和Hessian矩阵的参考基础上旋转分量的旋转操作
    M2 = np.array([
        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])

    # 2) 加载矩阵，用于在三角形第一个顶点(a0)的参考基础上旋转梯度和Hessian向量的分量
    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
                          [ 0.,  1.], [-1., -1.],
                          [-1., -1.], [ 1.,  0.]])

    # 3) 加载高斯点及其权重，在P2元素的三个子三角形上进行精确积分 - 每个子三角形上3个点
    #    注意：由于二阶导数不连续，我们确实需要这9个点！
    n_gauss = 9
    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
                          [ 4./18., 13./18.,  1./18.],
                          [ 7./18.,  7./18.,  4./18.],
                          [ 1./18., 13./18.,  4./18.],
                          [ 1./18.,  4./18., 13./18.],
                          [ 4./18.,  7./18.,  7./18.],
                          [ 4./18.,  1./18., 13./18.],
                          [13./18.,  1./18.,  4./18.],
                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
    gauss_w = np.ones([9], dtype=np.float64) / 9.

    # 4) 曲率能量的刚度矩阵
    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])

    # 5) 加载矩阵，用于从三角形顶点0处的 tri_J 计算 DOF_rot
    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])
    def get_function_values(self, alpha, ecc, dofs):
        """
        Parameters
        ----------
        alpha : is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates,
        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities,
        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the N-array of interpolated function values.
        """
        # 计算每个三角形中最小的 barycentric coordinate 索引
        subtri = np.argmin(alpha, axis=1)[:, 0]
        # 将 alpha 和 ecc 按照最小索引循环移位
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        # 提取 ksi 的各个分量
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        # 计算各个分量的平方
        x_sq = x*x
        y_sq = y*y
        z_sq = z*z
        # 构建 V 矩阵
        V = _to_matrix_vectorized([
            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
        # 计算 M @ V 的乘积
        prod = self.M @ V
        # 加上 E[:, 0, 0] 乘以 self.M0 @ V 的标量乘积
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
        # 加上 E[:, 1, 0] 乘以 self.M1 @ V 的标量乘积
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
        # 加上 E[:, 2, 0] 乘以 self.M2 @ V 的标量乘积
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
        # 将 prod 按照 3*subtri 循环移位
        s = _roll_vectorized(prod, 3*subtri, axis=0)
        # 计算 dofs @ s 的乘积，并返回第一维度的结果
        return (dofs @ s)[:, 0, 0]
    def get_function_derivatives(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices of
        barycentric coordinates)
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
        eccentricities)
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
        in global coordinates at locations alpha, as a column-matrices of
        shape (N x 2 x 1).
        """
        # Identify the index of the smallest barycentric coordinate component
        subtri = np.argmin(alpha, axis=1)[:, 0]
        # Roll arrays alpha and ecc to align with the identified subtriangles
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        # Extract components of ksi
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        # Compute squares of x, y, and z
        x_sq = x*x
        y_sq = y*y
        z_sq = z*z
        # Compute the matrix dV using vectorized operations
        dV = _to_matrix_vectorized([
            [    -3.*x_sq,     -3.*x_sq],
            [     3.*y_sq,           0.],
            [          0.,      3.*z_sq],
            [     -2.*x*z, -2.*x*z+x_sq],
            [-2.*x*y+x_sq,      -2.*x*y],
            [ 2.*x*y-y_sq,        -y_sq],
            [      2.*y*z,         y_sq],
            [        z_sq,       2.*y*z],
            [       -z_sq,  2.*x*z-z_sq],
            [     x*z-y*z,      x*y-y*z]])
        # Transform dV back to the basis of the first apex of each triangle
        dV = dV @ _extract_submatrices(
            self.rotate_dV, subtri, block_size=2, axis=0)

        # Compute the product of M and dV
        prod = self.M @ dV
        # Add contributions scaled by eccentricities E[:, 0, 0], E[:, 1, 0], E[:, 2, 0]
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
        # Roll prod to align with the subtriangles and compute df/dksi
        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
        dfdksi = dofs @ dsdksi
        # Compute df/dx in global coordinates using the inverse of J
        J_inv = _safe_inv22_vectorized(J)
        dfdx = J_inv @ _transpose_vectorized(dfdksi)
        return dfdx
    def get_function_hessians(self, alpha, J, ecc, dofs):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the values of interpolated function 2nd-derivatives
        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
        as a column-matrices of shape (N x 3 x 1).
        """
        # Compute the Hessian of shape functions with respect to barycentric coordinates
        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
        
        # Compute the 2nd derivatives of the function using degrees of freedom and Hessian
        d2fdksi2 = dofs @ d2sdksi2
        
        # Rotate the derivatives to global coordinates using the Jacobian matrices
        H_rot = self.get_Hrot_from_J(J)
        d2fdx2 = d2fdksi2 @ H_rot
        
        # Return the transposed vectorized result
        return _transpose_vectorized(d2fdx2)

    def get_d2Sidksij2(self, alpha, ecc):
        """
        Parameters
        ----------
        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
        expressed in covariant coordinates in first apex basis.
        """
        # Determine the index of the smallest element in each row of alpha
        subtri = np.argmin(alpha, axis=1)[:, 0]
        
        # Roll alpha and ecc arrays to align with the found subtriangles
        ksi = _roll_vectorized(alpha, -subtri, axis=0)
        E = _roll_vectorized(ecc, -subtri, axis=0)
        
        # Extract components from rolled ksi
        x = ksi[:, 0, 0]
        y = ksi[:, 1, 0]
        z = ksi[:, 2, 0]
        
        # Construct the d2V matrix for second derivatives
        d2V = _to_matrix_vectorized([
            [     6.*x,      6.*x,      6.*x],
            [     6.*y,        0.,        0.],
            [       0.,      6.*z,        0.],
            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
            [2.*x-4.*y,        0.,     -2.*y],
            [     2.*z,        0.,      2.*y],
            [       0.,      2.*y,      2.*z],
            [       0., 2.*x-4.*z,     -2.*z],
            [    -2.*z,     -2.*y,     x-y-z]])
        
        # Rotate d2V matrix to the first apex basis
        d2V = d2V @ _extract_submatrices(self.rotate_d2V, subtri, block_size=3, axis=0)
        
        # Compute the product using M matrices and eccentricities
        prod = self.M @ d2V
        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
        
        # Roll back the product to align with original subtriangles
        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
        
        # Return the computed Hessian of shape functions
        return d2sdksi2
    def get_bending_matrices(self, J, ecc):
        """
        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities

        Returns
        -------
        Returns the element K matrices for bending energy expressed in
        GLOBAL nodal coordinates.
        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
        tri_J is needed to rotate dofs from local basis to global basis
        """
        n = np.size(ecc, 0)  # 获取数组 ecc 的第一维度大小（即 N）

        # 1) matrix to rotate dofs in global coordinates
        J1 = self.J0_to_J1 @ J  # 将 J 乘以 self.J0_to_J1，用于旋转全局坐标中的自由度
        J2 = self.J0_to_J2 @ J  # 将 J 乘以 self.J0_to_J2，用于旋转全局坐标中的自由度
        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)  # 创建一个 n x 9 x 9 的全零数组，用于存储旋转矩阵
        DOF_rot[:, 0, 0] = 1  # 对角线元素设为 1
        DOF_rot[:, 3, 3] = 1  # 对角线元素设为 1
        DOF_rot[:, 6, 6] = 1  # 对角线元素设为 1
        DOF_rot[:, 1:3, 1:3] = J  # 填充子矩阵
        DOF_rot[:, 4:6, 4:6] = J1  # 填充子矩阵
        DOF_rot[:, 7:9, 7:9] = J2  # 填充子矩阵

        # 2) matrix to rotate Hessian in global coordinates.
        H_rot, area = self.get_Hrot_from_J(J, return_area=True)  # 调用 get_Hrot_from_J 方法得到 H_rot 和 area

        # 3) Computes stiffness matrix
        # Gauss quadrature.
        K = np.zeros([n, 9, 9], dtype=np.float64)  # 创建一个 n x 9 x 9 的全零数组，用于存储刚度矩阵
        weights = self.gauss_w  # 获取高斯积分权重
        pts = self.gauss_pts  # 获取高斯积分点
        for igauss in range(self.n_gauss):
            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)  # 复制并重塑高斯积分点
            alpha = np.expand_dims(alpha, 2)
            weight = weights[igauss]  # 获取当前高斯积分点的权重
            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)  # 调用 get_d2Sidksij2 方法计算 d2Skdksi2
            d2Skdx2 = d2Skdksi2 @ H_rot  # 计算 d2Skdx2
            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))  # 计算刚度矩阵 K

        # 4) With nodal (not elem) dofs
        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot  # 使用节点自由度计算刚度矩阵 K

        # 5) Need the area to compute total element energy
        return _scalar_vectorized(area, K)  # 返回计算出的总元素能量
# Private classes used to compute the degree of freedom of each triangular
# element for the TriCubicInterpolator.
class _DOF_estimator:
    """
    Abstract base class for classes used to estimate a function's first
    derivatives, and deduce the dofs for a CubicTriInterpolator using a
    reduced HCT element formulation.

    Derived classes implement ``compute_df(self, **kwargs)``, returning
    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
    gradient coordinates.
    """

    def __init__(self, interpolator, **kwargs):
        # Ensure the interpolator is an instance of CubicTriInterpolator
        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
        
        # Initialize instance variables from the interpolator
        self._pts = interpolator._pts
        self._tris_pts = interpolator._tris_pts
        self.z = interpolator._z
        self._triangles = interpolator._triangles
        self._unit_x, self._unit_y = interpolator._unit_x, interpolator._unit_y
        
        # Compute dz using a method implemented in derived classes
        self.dz = self.compute_dz(**kwargs)
        
        # Compute degrees of freedom based on computed derivatives
        self.compute_dof_from_df()

    def compute_dz(self, **kwargs):
        # This method is meant to be implemented by derived classes
        raise NotImplementedError

    def compute_dof_from_df(self):
        """
        Compute reduced-HCT elements degrees of freedom, from the gradient.
        """
        # Compute the Jacobian matrix for the triangular elements
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        
        # Select relevant parts of z and dz for each triangle
        tri_z = self.z[self._triangles]
        tri_dz = self.dz[self._triangles]
        
        # Compute degrees of freedom vector for each triangle
        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
        return tri_dof

    @staticmethod
    def get_dof_vec(tri_z, tri_dz, J):
        """
        Compute the dof vector of a triangle, from the value of f, df and
        of the local Jacobian at each node.

        Parameters
        ----------
        tri_z : shape (3,) array
            f nodal values.
        tri_dz : shape (3, 2) array
            df/dx, df/dy nodal values.
        J
            Jacobian matrix in local basis of apex 0.

        Returns
        -------
        dof : shape (9,) array
            For each apex ``iapex``::

                dof[iapex*3+0] = f(Ai)
                dof[iapex*3+1] = df(Ai).(AiAi+)
                dof[iapex*3+2] = df(Ai).(AiAi-)
        """
        npt = tri_z.shape[0]
        dof = np.zeros([npt, 9], dtype=np.float64)
        
        # Transformations using predefined matrices
        J1 = _ReducedHCT_Element.J0_to_J1 @ J
        J2 = _ReducedHCT_Element.J0_to_J2 @ J

        # Calculate columns for the dof matrix
        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)

        # Combine columns into matrix form
        dfdksi = _to_matrix_vectorized([
            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]
        ])
        
        # Populate the dof array with computed values
        dof[:, 0:7:3] = tri_z
        dof[:, 1:8:3] = dfdksi[:, 0]
        dof[:, 2:9:3] = dfdksi[:, 1]
        return dof


class _DOF_estimator_user(_DOF_estimator):
    """dz is imposed by user; accounts for scaling if any."""
    # 定义一个方法 compute_dz，接受一个元组 dz 作为参数
    def compute_dz(self, dz):
        # 将元组 dz 拆分为 dzdx 和 dzdy
        (dzdx, dzdy) = dz
        # 将 dzdx 缩放为单位长度 self._unit_x
        dzdx = dzdx * self._unit_x
        # 将 dzdy 缩放为单位长度 self._unit_y
        dzdy = dzdy * self._unit_y
        # 将 dzdx 和 dzdy 堆叠成一个二维数组，并将其转置后返回
        return np.vstack([dzdx, dzdy]).T
class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""

    def compute_dz(self):
        """
        self.df is computed as weighted average of _triangles sharing a common
        node. On each triangle itri f is first assumed linear (= ~f), which
        allows to compute d~f[itri]
        Then the following approximation of df nodal values is then proposed:
            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
        The weighted coeff. w[itri] are proportional to the angle of the
        triangle itri at apex ipt
        """
        # Compute geometric weights for the elements
        el_geom_w = self.compute_geom_weights()
        # Compute gradients of geometric quantities
        el_geom_grad = self.compute_geom_grads()

        # Sum of weights coeffs
        w_node_sum = np.bincount(np.ravel(self._triangles),
                                 weights=np.ravel(el_geom_w))

        # Sum of weighted df = (dfx, dfy)
        dfx_el_w = np.empty_like(el_geom_w)
        dfy_el_w = np.empty_like(el_geom_w)
        for iapex in range(3):
            # Compute weighted dfx and dfy for each apex
            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
        dfx_node_sum = np.bincount(np.ravel(self._triangles),
                                   weights=np.ravel(dfx_el_w))
        dfy_node_sum = np.bincount(np.ravel(self._triangles),
                                   weights=np.ravel(dfy_el_w))

        # Estimation of df
        dfx_estim = dfx_node_sum/w_node_sum
        dfy_estim = dfy_node_sum/w_node_sum
        return np.vstack([dfx_estim, dfy_estim]).T

    def compute_geom_weights(self):
        """
        Build the (nelems, 3) weights coeffs of _triangles angles,
        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
        """
        weights = np.zeros([np.size(self._triangles, 0), 3])
        tris_pts = self._tris_pts
        for ipt in range(3):
            # Define triangle points
            p0 = tris_pts[:, ipt % 3, :]
            p1 = tris_pts[:, (ipt+1) % 3, :]
            p2 = tris_pts[:, (ipt-1) % 3, :]
            # Compute angles alpha1 and alpha2
            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
            # Compute angle normalized by np.pi
            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
            # Compute weight proportional to angle up to np.pi/2
            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
        return weights
    def compute_geom_grads(self):
        """
        Compute the (global) gradient component of f assumed linear (~f).
        returns array df of shape (nelems, 2)
        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
        """
        tris_pts = self._tris_pts  # 获取三角形顶点坐标
        tris_f = self.z[self._triangles]  # 获取三角形顶点处的函数值

        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]  # 计算第一个边向量 dM1
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]  # 计算第二个边向量 dM2
        dM = np.dstack([dM1, dM2])  # 将边向量堆叠成一个三维数组 dM

        # 处理最简单的共线情况：在这种情况下假设梯度为零。
        dM_inv = _safe_inv22_vectorized(dM)  # 使用向量化的方式计算 dM 的逆矩阵

        dZ1 = tris_f[:, 1] - tris_f[:, 0]  # 计算第一个函数值差值向量 dZ1
        dZ2 = tris_f[:, 2] - tris_f[:, 0]  # 计算第二个函数值差值向量 dZ2
        dZ = np.vstack([dZ1, dZ2]).T  # 将函数值差值向量堆叠成二维数组 dZ

        df = np.empty_like(dZ)  # 创建一个与 dZ 相同形状的空数组 df，用于存储计算得到的梯度

        # 使用 np.einsum 进行计算：可能是 ej,eji -> ej
        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]  # 计算 df 的第一列
        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]  # 计算 df 的第二列

        return df  # 返回计算得到的梯度数组 df
# 定义一个名为 _DOF_estimator_min_E 的类，继承自 _DOF_estimator_geom 类
class _DOF_estimator_min_E(_DOF_estimator_geom):
    """
    The 'smoothest' approximation, df is computed through global minimization
    of the bending energy:
      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
    """
    
    # 类的初始化方法，接受一个 Interpolator 参数
    def __init__(self, Interpolator):
        # 继承父类的 _eccs 属性
        self._eccs = Interpolator._eccs
        # 调用父类的初始化方法
        super().__init__(Interpolator)

    # 定义 compute_dz 方法，用于计算 dz
    def compute_dz(self):
        """
        Elliptic solver for bending energy minimization.
        Uses a dedicated 'toy' sparse Jacobi PCG solver.
        """
        # 初始猜测值，调用父类的 compute_dz 方法
        dz_init = super().compute_dz()
        Uf0 = np.ravel(dz_init)

        # 创建一个 _ReducedHCT_Element 类的实例作为参考元素
        reference_element = _ReducedHCT_Element()
        # 调用 CubicTriInterpolator 类的 _get_jacobian 方法，获取雅可比矩阵 J
        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
        # 获取椭圆度 eccs、三角形 triangles 和 z 坐标 Uc
        eccs = self._eccs
        triangles = self._triangles
        Uc = self.z[self._triangles]

        # 调用 reference_element 实例的 get_Kff_and_Ff 方法，获取刚度矩阵 Kff 和力向量 Ff
        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
            J, eccs, triangles, Uc)

        # 构建稀疏矩阵 Kff_coo 和解决最小化问题
        tol = 1.e-10
        n_dof = Ff.shape[0]
        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
                                     shape=(n_dof, n_dof))
        Kff_coo.compress_csc()
        # 调用 _cg 函数，执行共轭梯度法求解
        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
        
        # 如果共轭梯度法未收敛，返回 Uf0 和 Uf 中较好的猜测值
        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
        if err0 < err:
            # 可能会在此处发出警告
            _api.warn_external("In TriCubicInterpolator initialization, "
                               "PCG sparse solver did not converge after "
                               "1000 iterations. `geom` approximation is "
                               "used instead of `min_E`")
            Uf = Uf0

        # 从 Uf 构建 dz
        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
        dz[:, 0] = Uf[::2]
        dz[:, 1] = Uf[1::2]
        return dz


# 下面的私有类 _Sparse_Matrix_coo 和函数 _cg 提供了用于（对称）椭圆问题的 PCG 稀疏求解器
class _Sparse_Matrix_coo:
    # _Sparse_Matrix_coo 类的初始化方法
    def __init__(self, vals, rows, cols, shape):
        """
        Create a sparse matrix in coo format.
        *vals*: arrays of values of non-null entries of the matrix
        *rows*: int arrays of rows of non-null entries of the matrix
        *cols*: int arrays of cols of non-null entries of the matrix
        *shape*: 2-tuple (n, m) of matrix shape
        """
        self.n, self.m = shape
        self.vals = np.asarray(vals, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)
    def dot(self, V):
        """
        Dot product of self by a vector *V* in sparse-dense to dense format
        *V* dense vector of shape (self.m,).
        """
        # 断言向量 V 的形状必须与 self 的行数相同
        assert V.shape == (self.m,)
        # 使用 np.bincount 计算稀疏矩阵与稠密向量 V 的点积，并返回结果
        return np.bincount(self.rows,
                           weights=self.vals * V[self.cols],
                           minlength=self.m)

    def compress_csc(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csc format.
        """
        # 使用 np.unique 去除重复项，并返回索引
        _, unique, indices = np.unique(
            self.rows + self.n * self.cols,
            return_index=True, return_inverse=True)
        # 根据去重后的索引重新组织 rows, cols, vals，以压缩稀疏矩阵的数据
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def compress_csr(self):
        """
        Compress rows, cols, vals / summing duplicates. Sort for csr format.
        """
        # 使用 np.unique 去除重复项，并返回索引
        _, unique, indices = np.unique(
            self.m * self.rows + self.cols,
            return_index=True, return_inverse=True)
        # 根据去重后的索引重新组织 rows, cols, vals，以压缩稀疏矩阵的数据
        self.rows = self.rows[unique]
        self.cols = self.cols[unique]
        self.vals = np.bincount(indices, weights=self.vals)

    def to_dense(self):
        """
        Return a dense matrix representing self, mainly for debugging purposes.
        """
        # 创建一个全零的稠密矩阵，用于存储稀疏矩阵转换而来的密集矩阵
        ret = np.zeros([self.n, self.m], dtype=np.float64)
        nvals = self.vals.size
        # 遍历稀疏矩阵的每个元素，将对应位置的值加到密集矩阵上
        for i in range(nvals):
            ret[self.rows[i], self.cols[i]] += self.vals[i]
        return ret

    def __str__(self):
        # 转换当前对象为密集矩阵并返回其字符串表示形式
        return self.to_dense().__str__()

    @property
    def diag(self):
        """Return the (dense) vector of the diagonal elements."""
        # 找出稀疏矩阵中的对角线元素，并存储在密集向量 diag 中返回
        in_diag = (self.rows == self.cols)
        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
        diag[self.rows[in_diag]] = self.vals[in_diag]
        return diag
def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
    """
    Use Preconditioned Conjugate Gradient iteration to solve A x = b
    A simple Jacobi (diagonal) preconditioner is used.

    Parameters
    ----------
    A : _Sparse_Matrix_coo
        *A* must have been compressed before by compress_csc or
        compress_csr method.
    b : array
        Right hand side of the linear system.
    x0 : array, optional
        Starting guess for the solution. Defaults to the zero vector.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below tol. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after *maxiter*
        steps even if the specified tolerance has not been achieved. Defaults
        to 1000.

    Returns
    -------
    x : array
        The converged solution.
    err : float
        The absolute error np.linalg.norm(A.dot(x) - b)
    """
    n = b.size
    assert A.n == n  # Ensure A's number of rows matches b's size
    assert A.m == n  # Ensure A's number of columns matches b's size
    b_norm = np.linalg.norm(b)  # Compute norm of b for relative residual calculation

    # Jacobi pre-conditioner
    kvec = A.diag  # Extract diagonal elements of A as the preconditioner
    # For diag elem < 1e-6 we keep 1e-6.
    kvec = np.maximum(kvec, 1e-6)  # Ensure diagonal elements are at least 1e-6

    # Initial guess
    if x0 is None:
        x = np.zeros(n)  # Initialize x with zeros if no initial guess is provided
    else:
        x = x0  # Start with the provided initial guess

    r = b - A.dot(x)  # Compute initial residual
    w = r / kvec  # Apply Jacobi preconditioner to the residual

    p = np.zeros(n)  # Initialize conjugate direction vector
    beta = 0.0
    rho = np.dot(r, w)  # Compute initial residual inner product

    k = 0  # Iteration counter

    # Following C. T. Kelley
    while (np.sqrt(abs(rho)) > tol * b_norm) and (k < maxiter):
        p = w + beta * p
        z = A.dot(p)
        alpha = rho / np.dot(p, z)
        r = r - alpha * z
        w = r / kvec
        rhoold = rho
        rho = np.dot(r, w)
        x = x + alpha * p
        beta = rho / rhoold
        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
        k += 1

    err = np.linalg.norm(A.dot(x) - b)  # Compute final absolute error
    return x, err
# 函数用于对一组(2, 2)矩阵数组进行求逆操作，并对秩不足的矩阵返回零矩阵。
def _safe_inv22_vectorized(M):
    """
    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
    matrices.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    # 检查输入矩阵数组的形状是否符合要求
    _api.check_shape((None, 2, 2), M=M)
    
    # 创建一个与输入矩阵数组相同形状的空数组，用于存储逆矩阵结果
    M_inv = np.empty_like(M)
    
    # 计算矩阵的乘积和行列式
    prod1 = M[:, 0, 0] * M[:, 1, 1]
    delta = prod1 - M[:, 0, 1] * M[:, 1, 0]

    # 判断矩阵的秩是否大于设定的阈值
    rank2 = (np.abs(delta) > 1e-8 * np.abs(prod1))
    
    # 如果所有矩阵的秩都大于阈值，则执行正常流程
    if np.all(rank2):
        delta_inv = 1. / delta
    else:
        # 如果存在秩不足的矩阵，则将相应位置的逆矩阵设为零矩阵
        delta_inv = np.zeros(M.shape[0])
        delta_inv[rank2] = 1. / delta[rank2]

    # 计算逆矩阵的各个元素
    M_inv[:, 0, 0] = M[:, 1, 1] * delta_inv
    M_inv[:, 0, 1] = -M[:, 0, 1] * delta_inv
    M_inv[:, 1, 0] = -M[:, 1, 0] * delta_inv
    M_inv[:, 1, 1] = M[:, 0, 0] * delta_inv
    
    # 返回逆矩阵数组
    return M_inv


# 函数用于对一组(2, 2)对称矩阵进行求逆操作，并返回伪逆矩阵以处理秩不足的情况。
def _pseudo_inv22sym_vectorized(M):
    """
    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.

    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
    In case M is of rank 0, we return the null matrix.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    # 检查输入矩阵数组的形状是否符合要求
    _api.check_shape((None, 2, 2), M=M)
    
    # 创建一个与输入矩阵数组相同形状的空数组，用于存储逆矩阵结果
    M_inv = np.empty_like(M)
    
    # 计算矩阵的乘积和行列式
    prod1 = M[:, 0, 0] * M[:, 1, 1]
    delta = prod1 - M[:, 0, 1] * M[:, 1, 0]
    
    # 判断矩阵的秩是否大于设定的阈值
    rank2 = (np.abs(delta) > 1e-8 * np.abs(prod1))
    # 如果所有的元素在rank2中都为True，执行正常的优化流程
    if np.all(rank2):
        # 正常的优化流程
        M_inv[:, 0, 0] = M[:, 1, 1] / delta
        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
        M_inv[:, 1, 1] = M[:, 0, 0] / delta
    else:
        # 'Pathologic' flow，处理异常情况
        # 这里需要处理两个子情况
        # 1) 第一个子情况：rank2中的矩阵为rank 2：
        delta = delta[rank2]
        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
        # 2) 第二个子情况：rank-deficient矩阵，即rank 0和1的矩阵：
        rank01 = ~rank2
        tr = M[rank01, 0, 0] + M[rank01, 1, 1]  # 计算trace
        tr_zeros = (np.abs(tr) < 1.e-8)  # 找出接近零的trace元素
        sq_tr_inv = (1. - tr_zeros) / (tr**2 + tr_zeros)  # 计算trace的倒数
        # sq_tr_inv = 1. / tr**2  # 若没有接近零的trace元素，则简化为trace的倒数
        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv

    return M_inv
# 计算标量与矩阵之间的乘积，通过在标量维度上扩展矩阵来实现向量化操作
def _scalar_vectorized(scalar, M):
    return scalar[:, np.newaxis, np.newaxis]*M


# 对矩阵数组 *M* 进行转置操作
def _transpose_vectorized(M):
    return np.transpose(M, [0, 2, 1])


# 沿着指定轴（0：行，1：列）根据索引数组 *roll_indices* 将矩阵数组 *M* 进行滚动（循环移位）操作
def _roll_vectorized(M, roll_indices, axis):
    assert axis in [0, 1]
    ndim = M.ndim
    assert ndim == 3
    ndim_roll = roll_indices.ndim
    assert ndim_roll == 1
    sh = M.shape
    r, c = sh[-2:]
    assert sh[0] == roll_indices.shape[0]
    vec_indices = np.arange(sh[0], dtype=np.int32)

    # 构建滚动后的矩阵
    M_roll = np.empty_like(M)
    if axis == 0:
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
    else:  # axis == 1
        for ir in range(r):
            for ic in range(c):
                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
    return M_roll


# 从具有相同形状的各个 np.array 构建矩阵数组 *M*
def _to_matrix_vectorized(M):
    assert isinstance(M, (tuple, list))
    assert all(isinstance(item, (tuple, list)) for item in M)
    c_vec = np.asarray([len(item) for item in M])
    assert np.all(c_vec-c_vec[0] == 0)
    r = len(M)
    c = c_vec[0]
    M00 = np.asarray(M[0][0])
    dt = M00.dtype
    sh = [M00.shape[0], r, c]
    M_ret = np.empty(sh, dtype=dt)
    for irow in range(r):
        for icol in range(c):
            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
    return M_ret


# 根据参数 *block_indices* 和 *block_size* 从矩阵 *M* 中提取选定的块状子矩阵数组 *Mres*
def _extract_submatrices(M, block_indices, block_size, axis):
    assert block_indices.ndim == 1
    assert axis in [0, 1]

    r, c = M.shape
    if axis == 0:
        sh = [block_indices.shape[0], block_size, c]
    else:  # axis == 1
        sh = [block_indices.shape[0], r, block_size]

    dt = M.dtype
    M_res = np.empty(sh, dtype=dt)
    if axis == 0:
        for ir in range(block_size):
            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
    else:  # axis == 1
        for ic in range(block_size):
            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]

    return M_res
```