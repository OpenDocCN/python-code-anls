# `D:\src\scipysrc\scipy\scipy\optimize\_shgo_lib\_complex.py`

```
# 导入标准库和第三方库
"""Base classes for low memory simplicial complex structures."""
import copy  # 导入深拷贝模块
import logging  # 导入日志记录模块
import itertools  # 导入迭代工具模块
import decimal  # 导入高精度十进制模块
from functools import cache  # 从 functools 模块导入缓存装饰器

import numpy as np  # 导入 numpy 库

from ._vertex import (VertexCacheField, VertexCacheIndex)  # 从 _vertex 模块导入 VertexCacheField 和 VertexCacheIndex 类


class Complex:
    """
    Base class for a simplicial complex described as a cache of vertices
    together with their connections.

    Important methods:
        Domain triangulation:
                Complex.triangulate, Complex.split_generation
        Triangulating arbitrary points (must be traingulable,
            may exist outside domain):
                Complex.triangulate(sample_set)
        Converting another simplicial complex structure data type to the
            structure used in Complex (ex. OBJ wavefront)
                Complex.convert(datatype, data)

    Important objects:
        HC.V: The cache of vertices and their connection
        HC.H: Storage structure of all vertex groups

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the complex R^dim
    domain : list of tuples, optional
        The bounds [x_l, x_u]^dim of the hyperrectangle space
        ex. The default domain is the hyperrectangle [0, 1]^dim
        Note: The domain must be convex, non-convex spaces can be cut
              away from this domain using the non-linear
              g_cons functions to define any arbitrary domain
              (these domains may also be disconnected from each other)
    sfield :
        A scalar function defined in the associated domain f: R^dim --> R
    sfield_args : tuple
        Additional arguments to be passed to `sfield`
    vfield :
        A scalar function defined in the associated domain
                       f: R^dim --> R^m
                   (for example a gradient function of the scalar field)
    vfield_args : tuple
        Additional arguments to be passed to vfield
    symmetry : None or list
            Specify if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by up to
            O(n!) times in the fully symmetric case.

            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
             x_6 are symmetric to x_4, this can be specified to the solver as:

            symmetry = [0,  # Variable 1
                        0,  # symmetric to variable 1
                        0,  # symmetric to variable 1
                        3,  # Variable 4
                        3,  # symmetric to variable 4
                        3,  # symmetric to variable 4
                        ]
    """

    def __init__(self, dim, domain=None, sfield=None, sfield_args=(),
                 vfield=None, vfield_args=(), symmetry=None):
        """
        Initialize a simplicial complex with specified parameters.

        Parameters
        ----------
        dim : int
            Spatial dimensionality of the complex R^dim
        domain : list of tuples, optional
            The bounds [x_l, x_u]^dim of the hyperrectangle space
        sfield : callable, optional
            A scalar function defined in the associated domain f: R^dim --> R
        sfield_args : tuple, optional
            Additional arguments to be passed to `sfield`
        vfield : callable, optional
            A scalar function defined in the associated domain f: R^dim --> R^m
        vfield_args : tuple, optional
            Additional arguments to be passed to `vfield`
        symmetry : None or list, optional
            Specifies if the objective function contains symmetric variables
        """
        pass  # 初始化函数，暂无额外操作，保留

    def triangulate(self, sample_set):
        """
        Triangulate the simplicial complex based on the given sample set.

        Parameters
        ----------
        sample_set : iterable
            Iterable of points to be triangulated
        """
        pass  # 三角化方法，根据给定的样本集进行三角化，暂无具体实现，保留

    def split_generation(self):
        """
        Perform split generation for the simplicial complex.
        """
        pass  # 分裂生成方法，用于生成复杂结构的分裂，暂无具体实现，保留

    def convert(self, datatype, data):
        """
        Convert another simplicial complex structure data type to the
        structure used in Complex.

        Parameters
        ----------
        datatype : str
            Type of the data structure to convert from
        data : object
            Data object to convert
        """
        pass  # 转换方法，将另一种复杂结构数据类型转换为 Complex 使用的结构，暂无具体实现，保留
    # 约束条件的定义，可以是字典形式或字典序列形式，是一个可选参数
    constraints : dict or sequence of dict, optional
        Constraints definition.
    
        函数 ``R**n`` 的形式为：
    
            g(x) <= 0，作为 g : R^n -> R^m
            h(x) == 0，作为 h : R^n -> R^p
    
        每个约束以字典形式定义，包含以下字段：
    
            type : str
                约束类型：'eq' 表示等式约束，'ineq' 表示不等式约束。
            fun : callable
                定义约束的函数。
            jac : callable, optional
                `fun` 的雅可比矩阵（仅适用于 SLSQP）。
            args : sequence, optional
                传递给函数和雅可比矩阵的额外参数。
    
        等式约束意味着约束函数的结果应为零，而不等式约束意味着结果应为非负数。
    
    workers : int, optional
        使用 `multiprocessing.Pool <multiprocessing>` 并行计算字段函数。
    def __init__(self, dim, domain=None, sfield=None, sfield_args=(),
                 symmetry=None, constraints=None, workers=1):
        # 初始化函数，用于设置对象的各个属性
        self.dim = dim

        # Domains
        self.domain = domain
        if domain is None:
            self.bounds = [(0.0, 1.0), ] * dim  # 如果域未指定，则设定默认边界
        else:
            self.bounds = domain
        self.symmetry = symmetry  # 对称性设置

        # Field functions
        self.sfield = sfield  # 标量场函数
        self.sfield_args = sfield_args  # 标量场函数的参数

        # Process constraints
        # Constraints
        # 处理约束条件的设定
        if constraints is not None:
            self.min_cons = constraints  # 最小约束条件
            self.g_cons = []
            self.g_args = []
            if not isinstance(constraints, (tuple, list)):
                constraints = (constraints,)

            for cons in constraints:
                if cons['type'] in ('ineq'):  # 处理不等式类型的约束
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)  # 将约束函数列表转为元组
            self.g_args = tuple(self.g_args)  # 将约束函数的参数列表转为元组
        else:
            self.g_cons = None
            self.g_args = None

        # Homology properties
        self.gen = 0  # 生成代数
        self.perm_cycle = 0  # 排列周期

        # Every cell is stored in a list of its generation,
        # ex. the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is sub-generated it is removed from this list

        self.H = []  # 存储顶点组的数据结构列表

        # Cache of all vertices
        if (sfield is not None) or (self.g_cons is not None):
            # 如果存在标量场函数或约束条件，则初始化顶点缓存及相关的场缓存
            if sfield is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
            elif self.g_cons is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
        else:
            self.V = VertexCacheIndex()  # 否则初始化顶点索引缓存

        self.V_non_symm = []  # 非对称顶点列表
    def refine(self, n=1):
        if n is None:
            try:
                self.triangulated_vectors
                self.refine_all()
                return
            except AttributeError as ae:
                if str(ae) == "'Complex' object has no attribute " \
                              "'triangulated_vectors'":
                    self.triangulate(symmetry=self.symmetry)
                    return
                else:
                    raise

        nt = len(self.V.cache) + n  # 目标顶点总数
        # 外部 while 循环，在向复杂结构中添加额外的 `n` 个顶点之前一直迭代：
        while len(self.V.cache) < nt:  # while 循环 1
            try:  # try 块 1
                # 尝试访问 triangulated_vectors，只有在已执行初始三角剖分时才应该定义：
                self.triangulated_vectors
                # 尝试迭代当前生成器，如果不存在或已耗尽，则生成新的生成器
                try:  # try 块 2
                    next(self.rls)
                except (AttributeError, StopIteration, KeyError):
                    vp = self.triangulated_vectors[0]
                    self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                    next(self.rls)

            except (AttributeError, KeyError):
                # 如果尚未完成初始三角剖分，则开始/继续初始三角剖分，目标是 `nt` 个顶点，
                # 如果 `nt` 大于初始顶点数，则 `refine` 程序将返回到 try 块 1。
                self.triangulate(nt, self.symmetry)
        return

    def refine_all(self, centroids=True):
        """细化当前复杂结构的整个域。"""
        try:
            self.triangulated_vectors
            tvs = copy.copy(self.triangulated_vectors)
            for i, vp in enumerate(tvs):
                self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                for i in self.rls:
                    i
        except AttributeError as ae:
            if str(ae) == "'Complex' object has no attribute " \
                          "'triangulated_vectors'":
                self.triangulate(symmetry=self.symmetry, centroid=centroids)
            else:
                raise

        # 这将在由 self.triangulated_vectors 定义和生成的每个新子域中添加一个质心，并
        # 定义顶点！以完成三角剖分。
        return
    def refine_star(self, v):
        """Refine the star domain of a vertex `v`."""
        # 复制邻居列表以便迭代
        vnn = copy.copy(v.nn)
        v1nn = []
        d_v0v1_set = set()
        for v1 in vnn:
            v1nn.append(copy.copy(v1.nn))

        for v1, v1nn in zip(vnn, v1nn):
            # 计算 v1 的邻居与 v 的交集
            vnnu = v1nn.intersection(vnn)

            # 分割边 v.x 到 v1.x，并将结果添加到集合 d_v0v1_set
            d_v0v1 = self.split_edge(v.x, v1.x)
            for o_d_v0v1 in d_v0v1_set:
                d_v0v1.connect(o_d_v0v1)
            d_v0v1_set.add(d_v0v1)

            # 对于 v1 的每个邻居 v2，分割边 v1.x 到 v2.x，并将结果连接到 d_v0v1
            for v2 in vnnu:
                d_v1v2 = self.split_edge(v1.x, v2.x)
                d_v0v1.connect(d_v1v2)
        return

    @cache
    def split_edge(self, v1, v2):
        """Split the edge between vertices v1 and v2."""
        v1 = self.V[v1]
        v2 = self.V[v2]
        # 如果存在原始边，则断开它
        v1.disconnect(v2)

        # 计算边中心的顶点位置
        try:
            vct = (v2.x_a - v1.x_a) / 2.0 + v1.x_a
        except TypeError:  # 允许十进制运算
            vct = (v2.x_a - v1.x_a) / decimal.Decimal(2.0) + v1.x_a

        # 获取中心顶点对象，并连接到原始的两个顶点
        vc = self.V[tuple(vct)]
        vc.connect(v1)
        vc.connect(v2)
        return vc

    def vpool(self, origin, supremum):
        """Create a pool of vertices within the bounds defined by origin and supremum."""
        vot = tuple(origin)
        vst = tuple(supremum)
        # 如果顶点不存在，则初始化
        vo = self.V[vot]
        vs = self.V[vst]

        # 移除起点到终点的连接

        # 找到精细化超矩形的上下界
        bl = list(vot)
        bu = list(vst)
        for i, (voi, vsi) in enumerate(zip(vot, vst)):
            if bl[i] > vsi:
                bl[i] = vsi
            if bu[i] < voi:
                bu[i] = voi

        # 注意：这里使用集合和列表，因为不确定 numpy 数组在成千上万维情况下的扩展性。
        vn_pool = set()
        vn_pool.update(vo.nn)
        vn_pool.update(vs.nn)
        cvn_pool = copy.copy(vn_pool)
        for vn in cvn_pool:
            for i, xi in enumerate(vn.x):
                if bl[i] <= xi <= bu[i]:
                    pass
                else:
                    try:
                        vn_pool.remove(vn)
                    except KeyError:
                        pass  # 注意：并非所有的邻居都在初始池中
        return vn_pool
    # 如果当前对象的维度大于1，则执行以下代码块
    if self.dim > 1:
        # 对于每个简单形式中的每条边
        for s in simplices:
            # 生成当前简单形式中所有可能的边的组合
            edges = itertools.combinations(s, self.dim)
            # 对于每条边
            for e in edges:
                # 获取边的两个顶点，并将它们对应的顶点对象连接起来
                self.V[tuple(vertices[e[0]])].connect(
                    self.V[tuple(vertices[e[1]])])
    # 如果当前对象的维度不大于1，则执行以下代码块
    else:
        # 对于每条边
        for e in simplices:
            # 获取边的两个顶点，并将它们对应的顶点对象连接起来
            self.V[tuple(vertices[e[0]])].connect(
                self.V[tuple(vertices[e[1]])])
    # 函数结束，没有返回值
    return
    def connect_vertex_non_symm(self, v_x, near=None):
        """
        Adds a vertex at coords v_x to the complex that is not symmetric to the
        initial triangulation and sub-triangulation.

        If near is specified (for example; a star domain or collections of
        cells known to contain v) then only those simplices containd in near
        will be searched, this greatly speeds up the process.

        If near is not specified this method will search the entire simplicial
        complex structure.

        Parameters
        ----------
        v_x : tuple
            Coordinates of non-symmetric vertex
        near : set or list
            List of vertices, these are points near v to check for
        """
        # If near is not provided, initialize star to all vertices
        if near is None:
            star = self.V
        else:
            star = near
        
        # Check if the vertex v_x already exists in the cache
        if tuple(v_x) in self.V.cache:
            # If it exists, check if it's already marked as non-symmetric
            if self.V[v_x] in self.V_non_symm:
                pass  # If already non-symmetric, do nothing
            else:
                return  # If not non-symmetric, return without further action

        # Access the vertex object corresponding to v_x
        self.V[v_x]
        found_nn = False
        S_rows = []
        
        # Collect coordinates of vertices in star
        for v in star:
            S_rows.append(v.x)

        S_rows = np.array(S_rows)
        A = np.array(S_rows) - np.array(v_x)
        
        # Iterate through combinations of vertices to find simplices
        for s_i in itertools.combinations(range(S_rows.shape[0]),
                                          r=self.dim + 1):
            valid_simplex = True
            
            # Check if every pair of vertices in the simplex is connected
            for i in itertools.combinations(s_i, r=2):
                if ((self.V[tuple(S_rows[i[1]])] not in
                        self.V[tuple(S_rows[i[0]])].nn)
                    and (self.V[tuple(S_rows[i[0]])] not in
                         self.V[tuple(S_rows[i[1]])].nn)):
                    valid_simplex = False
                    break

            S = S_rows[tuple([s_i])]
            if valid_simplex:
                # Check if the simplex meets the degree condition
                if self.deg_simplex(S, proj=None):
                    valid_simplex = False

            # If s_i is a valid simplex, check if v_x is inside it
            if valid_simplex:
                A_j0 = A[tuple([s_i])]
                if self.in_simplex(S, v_x, A_j0):
                    found_nn = True
                    break  # Break out of the main loop, found target simplex

        # If a valid simplex containing v_x was found, connect it to the vertices
        if found_nn:
            for i in s_i:
                self.V[v_x].connect(self.V[tuple(S_rows[i])])
        
        # Store v_x in the list of non-symmetric vertices
        self.V_non_symm.append(self.V[v_x])
        
        # Return True if a successful connection was made, False otherwise
        return found_nn
    def in_simplex(self, S, v_x, A_j0=None):
        """Check if a vector v_x is in simplex `S`.

        Parameters
        ----------
        S : array_like
            Array containing simplex entries of vertices as rows
        v_x :
            A candidate vertex
        A_j0 : array, optional,
            Allows for A_j0 to be pre-calculated

        Returns
        -------
        res : boolean
            True if `v_x` is in `S`
        """
        # Calculate matrix A_11 by removing the first row from S and subtracting the first row
        A_11 = np.delete(S, 0, 0) - S[0]

        # Compute the sign of the determinant of A_11
        sign_det_A_11 = np.sign(np.linalg.det(A_11))

        # Handle the case where the determinant of A_11 is zero
        if sign_det_A_11 == 0:
            # NOTE: We keep the variable A_11, but we loop through A_jj
            # ind=
            # while sign_det_A_11 == 0:
            #    A_11 = np.delete(S, ind, 0) - S[ind]
            #    sign_det_A_11 = np.sign(np.linalg.det(A_11))

            # Assume a non-zero value for sign_det_A_11 to avoid zero determinant issue
            sign_det_A_11 = -1  # TODO: Choose another det of j instead?
            # TODO: Unlikely to work in many cases

        # If A_j0 is not provided, calculate it as S - v_x
        if A_j0 is None:
            A_j0 = S - v_x

        # Iterate over each dimension d of the simplex
        for d in range(self.dim + 1):
            # Compute the determinant of A_jj for current d
            det_A_jj = (-1)**d * sign_det_A_11

            # Compute the sign of the determinant of A_j0 with the d-th row deleted
            sign_det_A_j0 = np.sign(np.linalg.det(np.delete(A_j0, d, 0)))

            # Compare det_A_jj with sign_det_A_j0, if they are equal, continue; otherwise, return False
            if det_A_jj == sign_det_A_j0:
                continue
            else:
                return False

        # If all comparisons are equal, return True indicating v_x is in S
        return True

    def deg_simplex(self, S, proj=None):
        """Test a simplex S for degeneracy (linear dependence in R^dim).

        Parameters
        ----------
        S : np.array
            Simplex with rows as vertex vectors
        proj : array, optional,
            If the projection S[1:] - S[0] is already
            computed it can be added as an optional argument.
        """
        # Strategy: we test all combination of faces, if any of the
        # determinants are zero then the vectors lie on the same face and is
        # therefore linearly dependent in the space of R^dim
        
        # If proj is not provided, compute it as S[1:] - S[0]
        if proj is None:
            proj = S[1:] - S[0]

        # TODO: Is checking the projection of one vertex against faces of other
        #       vertices sufficient? Or do we need to check more vertices in
        #       dimensions higher than 2?
        # TODO: Literature seems to suggest using proj.T, but why is this
        #       needed?
        
        # Check if the determinant of proj is zero to determine degeneracy
        if np.linalg.det(proj) == 0.0:  # TODO: Replace with tolerance?
            return True  # Simplex is degenerate
        else:
            return False  # Simplex is not degenerate
```