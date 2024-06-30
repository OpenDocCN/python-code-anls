# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_special_sparse_arrays.py`

```
# 导入必要的库
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array

# 定义此模块的公共接口，仅包括'LaplacianNd'
__all__ = ['LaplacianNd']

# Sakurai and Mikota classes are intended for tests and benchmarks
# and explicitly not included in the public API of this module.

# LaplacianNd类继承自LinearOperator类，表示N维网格的拉普拉斯算子及其特征值/特征向量。
class LaplacianNd(LinearOperator):
    """
    The grid Laplacian in ``N`` dimensions and its eigenvalues/eigenvectors.

    Construct Laplacian on a uniform rectangular grid in `N` dimensions
    and output its eigenvalues and eigenvectors.
    The Laplacian ``L`` is square, negative definite, real symmetric array
    with signed integer entries and zeros otherwise.

    Parameters
    ----------
    grid_shape : tuple
        A tuple of integers of length ``N`` (corresponding to the dimension of
        the Lapacian), where each entry gives the size of that dimension. The
        Laplacian matrix is square of the size ``np.prod(grid_shape)``.
    boundary_conditions : {'neumann', 'dirichlet', 'periodic'}, optional
        The type of the boundary conditions on the boundaries of the grid.
        Valid values are ``'dirichlet'`` or ``'neumann'``(default) or
        ``'periodic'``.
    dtype : dtype
        Numerical type of the array. Default is ``np.int8``.

    Methods
    -------
    toarray()
        Construct a dense array from Laplacian data
    tosparse()
        Construct a sparse array from Laplacian data
    eigenvalues(m=None)
        Construct a 1D array of `m` largest (smallest in absolute value)
        eigenvalues of the Laplacian matrix in ascending order.
    eigenvectors(m=None):
        Construct the array with columns made of `m` eigenvectors (``float``)
        of the ``Nd`` Laplacian corresponding to the `m` ordered eigenvalues.

    .. versionadded:: 1.12.0

    Notes
    -----
    Compared to the MATLAB/Octave implementation [1] of 1-, 2-, and 3-D
    Laplacian, this code allows the arbitrary N-D case and the matrix-free
    callable option, but is currently limited to pure Dirichlet, Neumann or
    Periodic boundary conditions only.

    The Laplacian matrix of a graph (`scipy.sparse.csgraph.laplacian`) of a
    rectangular grid corresponds to the negative Laplacian with the Neumann
    conditions, i.e., ``boundary_conditions = 'neumann'``.

    All eigenvalues and eigenvectors of the discrete Laplacian operator for
    an ``N``-dimensional  regular grid of shape `grid_shape` with the grid
    step size ``h=1`` are analytically known [2].

    References
    ----------
    .. [1] https://github.com/lobpcg/blopex/blob/master/blopex_\
tools/matlab/laplacian/laplacian.m
    .. [2] "Eigenvalues and eigenvectors of the second derivative", Wikipedia
           https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors_\
of_the_second_derivative

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LaplacianNd
    >>> from scipy.sparse import diags, csgraph
    >>> from scipy.linalg import eigvalsh
    """
    pass
    # 定义一个六个网格点的一维拉普拉斯矩阵，使用稀疏的三对角形式表示邻接矩阵
    n = 6
    G = diags(np.ones(n - 1), 1, format='csr')
    # 计算函数形式的拉普拉斯矩阵 Lf，对称化为负图拉普拉斯矩阵
    Lf = csgraph.laplacian(G, symmetrized=True, form='function')
    # 定义网格形状为 (6,) 的拉普拉斯算子 lap，边界条件为 Neumann
    lap = LaplacianNd(grid_shape, boundary_conditions='neumann')
    # 检验 lap.matmat(np.eye(n)) 是否等于 -Lf(np.eye(n))
    np.array_equal(lap.matmat(np.eye(n)), -Lf(np.eye(n)))

    # 由于拉普拉斯矩阵的所有元素都是整数，'int8' 是存储矩阵表示的默认数据类型
    lap.tosparse()  # 将 lap 转换为稀疏对角矩阵
    lap.toarray()   # 将 lap 转换为密集数组表示
    np.array_equal(lap.matmat(np.eye(n)), lap.toarray())  # 检验稠密数组表示是否与 matmat 函数结果相等

    # 初始化具有周期边界条件的拉普拉斯算子 lap
    lap = LaplacianNd(grid_shape, boundary_conditions='periodic')
    # 计算拉普拉斯矩阵的特征值
    lap.eigenvalues()
    lap.eigenvalues()[-2:]  # 提取倒数第二和最后一个特征值
    lap.eigenvalues(2)      # 提取前两个特征值
    lap.eigenvectors(1)     # 计算第一个特征向量
    lap.eigenvectors(2)     # 计算前两个特征向量
    lap.eigenvectors()      # 计算所有特征向量

    # 在二维正则网格上展示二维拉普拉斯算子，网格形状为 (2, 3)
    grid_shape = (2, 3)
    n = np.prod(grid_shape)

    # 网格点的编号如下
    np.arange(n).reshape(grid_shape + (-1,))
    array([[[0],
            [1],
            [2]],
    
           [[3],
            [4],
            [5]]])



    Each of the boundary conditions ``'dirichlet'``, ``'periodic'``, and
    ``'neumann'`` is illustrated separately; with ``'dirichlet'``



    >>> lap = LaplacianNd(grid_shape, boundary_conditions='dirichlet')



    >>> lap.tosparse()



    <Compressed Sparse Row sparse array of dtype 'int8'
        with 20 stored elements and shape (6, 6)>



    >>> lap.toarray()



    array([[-4,  1,  0,  1,  0,  0],
           [ 1, -4,  1,  0,  1,  0],
           [ 0,  1, -4,  0,  0,  1],
           [ 1,  0,  0, -4,  1,  0],
           [ 0,  1,  0,  1, -4,  1],
           [ 0,  0,  1,  0,  1, -4]], dtype=int8)



    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())



    True



    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())



    True



    >>> lap.eigenvalues()



    array([-6.41421356, -5.        , -4.41421356, -3.58578644, -3.        ,
           -1.58578644])



    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))



    >>> np.allclose(lap.eigenvalues(), eigvals)



    True



    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))



    True



    with ``'periodic'``



    >>> lap = LaplacianNd(grid_shape, boundary_conditions='periodic')



    >>> lap.tosparse()



    <Compressed Sparse Row sparse array of dtype 'int8'
        with 24 stored elements and shape (6, 6)>



    >>> lap.toarray()



    array([[-4,  1,  1,  2,  0,  0],
           [ 1, -4,  1,  0,  2,  0],
           [ 1,  1, -4,  0,  0,  2],
           [ 2,  0,  0, -4,  1,  1],
           [ 0,  2,  0,  1, -4,  1],
           [ 0,  0,  2,  1,  1, -4]], dtype=int8)



    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())



    True



    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())



    True



    >>> lap.eigenvalues()



    array([-7., -7., -4., -3., -3.,  0.])



    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))



    >>> np.allclose(lap.eigenvalues(), eigvals)



    True



    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))



    True



    and with ``'neumann'``



    >>> lap = LaplacianNd(grid_shape, boundary_conditions='neumann')



    >>> lap.tosparse()



    <Compressed Sparse Row sparse array of dtype 'int8'
        with 20 stored elements and shape (6, 6)>



    >>> lap.toarray()



    array([[-2,  1,  0,  1,  0,  0],
           [ 1, -3,  1,  0,  1,  0],
           [ 0,  1, -2,  0,  0,  1],
           [ 1,  0,  0, -2,  1,  0],
           [ 0,  1,  0,  1, -3,  1],
           [ 0,  0,  1,  0,  1, -2]])



    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())



    True



    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())



    True



    >>> lap.eigenvalues()



    array([-5., -3., -3., -2., -1.,  0.])



    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))
    # 检查 LaplacianNd 对象的特征值是否与给定的特征值 `eigvals` 全部接近
    >>> np.allclose(lap.eigenvalues(), eigvals)
    True
    # 检查 LaplacianNd 对象的稀疏表示乘以其特征向量，与特征向量乘以特征值对角矩阵的结果是否全部接近
    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))
    True

    """

    # LaplacianNd 类的构造函数，初始化一个网格形状为 `grid_shape` 的 Laplacian 矩阵
    def __init__(self, grid_shape, *,
                 boundary_conditions='neumann',
                 dtype=np.int8):
        # 检查边界条件是否合法，只接受 'dirichlet', 'neumann', 'periodic' 三种选项
        if boundary_conditions not in ('dirichlet', 'neumann', 'periodic'):
            raise ValueError(
                f"Unknown value {boundary_conditions!r} is given for "
                "'boundary_conditions' parameter. The valid options are "
                "'dirichlet', 'periodic', and 'neumann' (default)."
            )

        self.grid_shape = grid_shape  # 将网格形状存储到实例变量中
        self.boundary_conditions = boundary_conditions  # 存储边界条件到实例变量中
        # 计算网格形状的总元素数，作为 Laplacian 矩阵的大小
        N = np.prod(grid_shape)
        # 调用父类的构造函数初始化 LaplacianNd 对象，设置数据类型和形状
        super().__init__(dtype=dtype, shape=(N, N))

    # 私有方法，用于处理特征值的排序和选择
    def _eigenvalue_ordering(self, m):
        """Compute `m` largest eigenvalues in each of the ``N`` directions,
        i.e., up to ``m * N`` total, order them and return `m` largest.
        """
        grid_shape = self.grid_shape
        if m is None:
            indices = np.indices(grid_shape)  # 获取网格形状的索引
            Leig = np.zeros(grid_shape)  # 初始化一个全零数组
        else:
            grid_shape_min = min(grid_shape,
                                 tuple(np.ones_like(grid_shape) * m))  # 取最小网格形状和 m 的元组
            indices = np.indices(grid_shape_min)  # 获取最小网格形状的索引
            Leig = np.zeros(grid_shape_min)  # 初始化一个全零数组

        # 根据不同的边界条件计算 Laplacian 矩阵的对角元素
        for j, n in zip(indices, grid_shape):
            if self.boundary_conditions == 'dirichlet':
                Leig += -4 * np.sin(np.pi * (j + 1) / (2 * (n + 1))) ** 2
            elif self.boundary_conditions == 'neumann':
                Leig += -4 * np.sin(np.pi * j / (2 * n)) ** 2
            else:  # boundary_conditions == 'periodic'
                Leig += -4 * np.sin(np.pi * np.floor((j + 1) / 2) / n) ** 2

        Leig_ravel = Leig.ravel()  # 将 Laplacian 矩阵展平成一维数组
        ind = np.argsort(Leig_ravel)  # 对展平后的 Laplacian 矩阵的元素进行排序，返回索引
        eigenvalues = Leig_ravel[ind]  # 根据排序的索引获取特征值
        if m is not None:
            eigenvalues = eigenvalues[-m:]  # 如果指定了 m，则返回最大的 m 个特征值
            ind = ind[-m:]  # 对应的索引也截取最大的 m 个

        return eigenvalues, ind  # 返回特征值数组和索引数组

    # 公共方法，返回请求的特征值数量
    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : float array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        eigenvalues, _ = self._eigenvalue_ordering(m)  # 调用私有方法获取特征值数组
        return eigenvalues  # 返回特征值数组
    def _ev1d(self, j, n):
        """Return 1 eigenvector in 1d with index `j`
        and number of grid points `n` where ``j < n``. 
        """
        if self.boundary_conditions == 'dirichlet':
            # Compute grid points for Dirichlet boundary conditions
            i = np.pi * (np.arange(n) + 1) / (n + 1)
            # Calculate the 1d eigenvector using sine function
            ev = np.sqrt(2. / (n + 1.)) * np.sin(i * (j + 1))
        elif self.boundary_conditions == 'neumann':
            # Compute grid points for Neumann boundary conditions
            i = np.pi * (np.arange(n) + 0.5) / n
            # Calculate the 1d eigenvector using cosine function
            ev = np.sqrt((1. if j == 0 else 2.) / n) * np.cos(i * j)
        else:  # boundary_conditions == 'periodic'
            if j == 0:
                # Special case for periodic boundary conditions
                ev = np.sqrt(1. / n) * np.ones(n)
            elif j + 1 == n and n % 2 == 0:
                # Special case for periodic boundary conditions and even n
                ev = np.sqrt(1. / n) * np.tile([1, -1], n//2)
            else:
                # Compute grid points for periodic boundary conditions
                i = 2. * np.pi * (np.arange(n) + 0.5) / n
                # Calculate the 1d eigenvector using cosine function
                ev = np.sqrt(2. / n) * np.cos(i * np.floor((j + 1) / 2))
        
        # Correct small values to exact zeros to handle round-off errors
        # Exact zeros are appropriate due to symmetry of eigenvectors
        ev[np.abs(ev) < np.finfo(np.float64).eps] = 0.
        return ev

    def _one_eve(self, k):
        """Return 1 eigenvector in Nd with multi-index `k`
        as a tensor product of the corresponding 1d eigenvectors. 
        """
        # Generate 1d eigenvectors for each index in multi-index `k`
        phi = [self._ev1d(j, n) for j, n in zip(k, self.grid_shape)]
        result = phi[0]
        # Compute tensor product of all 1d eigenvectors
        for phi in phi[1:]:
            result = np.tensordot(result, phi, axes=0)
        return np.asarray(result).ravel()

    def eigenvectors(self, m=None):
        """Return the requested number of eigenvectors for ordered eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of eigenvectors to return. If not provided,
            then all eigenvectors will be returned.
            
        Returns
        -------
        eigenvectors : float array
            An array with columns made of the requested `m` or all eigenvectors.
            The columns are ordered according to the `m` ordered eigenvalues. 
        """
        _, ind = self._eigenvalue_ordering(m)
        if m is None:
            grid_shape_min = self.grid_shape
        else:
            # Determine the minimum grid shape considering `m`
            grid_shape_min = min(self.grid_shape,
                                tuple(np.ones_like(self.grid_shape) * m))

        # Obtain indices for `m` ordered eigenvalues
        N_indices = np.unravel_index(ind, grid_shape_min)
        N_indices = [tuple(x) for x in zip(*N_indices)]
        # Generate eigenvectors corresponding to these indices
        eigenvectors_list = [self._one_eve(k) for k in N_indices]
        return np.column_stack(eigenvectors_list)
    def toarray(self):
        """
        Converts the Laplacian data to a dense array.

        Returns
        -------
        L : ndarray
            The shape is ``(N, N)`` where ``N = np.prod(grid_shape)``.

        """
        grid_shape = self.grid_shape  # 获取网格形状
        n = np.prod(grid_shape)  # 计算网格中元素的总数
        L = np.zeros([n, n], dtype=np.int8)  # 创建一个全零矩阵 L，数据类型为 int8
        # Scratch arrays
        L_i = np.empty_like(L)  # 创建一个与 L 相同形状的空数组 L_i
        Ltemp = np.empty_like(L)  # 创建一个与 L 相同形状的空数组 Ltemp

        for ind, dim in enumerate(grid_shape):
            # Start zeroing out L_i
            L_i[:] = 0  # 将 L_i 全部置零

            # Allocate the top left corner with the kernel of L_i
            # Einsum returns writable view of arrays
            np.einsum("ii->i", L_i[:dim, :dim])[:] = -2  # 在 L_i 的左上角分配 Laplacian 的核心部分
            np.einsum("ii->i", L_i[: dim - 1, 1:dim])[:] = 1  # 分配对角线右上方的元素
            np.einsum("ii->i", L_i[1:dim, : dim - 1])[:] = 1  # 分配对角线左下方的元素

            if self.boundary_conditions == 'neumann':
                L_i[0, 0] = -1  # Neumann 边界条件：左上角设置为 -1
                L_i[dim - 1, dim - 1] = -1  # Neumann 边界条件：右下角设置为 -1
            elif self.boundary_conditions == 'periodic':
                if dim > 1:
                    L_i[0, dim - 1] += 1  # 周期性边界条件：右上角设置为 +1
                    L_i[dim - 1, 0] += 1  # 周期性边界条件：左下角设置为 +1
                else:
                    L_i[0, 0] += 1  # 对于维度为 1 的情况，设置左上角为 +1

            # kron is too slow for large matrices hence the next two tricks
            # 1- kron(eye, mat) is block_diag(mat, mat, ...)
            # 2- kron(mat, eye) can be performed by 4d stride trick

            # 1-
            new_dim = dim
            # for block_diag we tile the top left portion on the diagonal
            if ind > 0:
                tiles = np.prod(grid_shape[:ind])
                for j in range(1, tiles):
                    L_i[j*dim:(j+1)*dim, j*dim:(j+1)*dim] = L_i[:dim, :dim]
                    new_dim += dim

            # 2-
            # we need the keep L_i, but reset the array
            Ltemp[:new_dim, :new_dim] = L_i[:new_dim, :new_dim]
            tiles = int(np.prod(grid_shape[ind+1:]))
            # Zero out the top left, the rest is already 0
            L_i[:new_dim, :new_dim] = 0
            idx = [x for x in range(tiles)]
            L_i.reshape(
                (new_dim, tiles,
                 new_dim, tiles)
                )[:, idx, :, idx] = Ltemp[:new_dim, :new_dim]

            L += L_i  # 将 L_i 的值累加到 L 中

        return L.astype(self.dtype)  # 返回结果 L，并将其类型转换为对象的指定类型
    def tosparse(self):
        """
        Constructs a sparse array from the Laplacian data. The returned sparse
        array format is dependent on the selected boundary conditions.

        Returns
        -------
        L : scipy.sparse.sparray
            The shape is ``(N, N)`` where ``N = np.prod(grid_shape)``.

        """
        # 获取网格形状的维度数
        N = len(self.grid_shape)
        # 计算网格总元素数
        p = np.prod(self.grid_shape)
        # 创建一个零稀疏矩阵，形状为 (p, p)，数据类型为 np.int8
        L = dia_array((p, p), dtype=np.int8)

        # 遍历每一个维度
        for i in range(N):
            # 获取当前维度的大小
            dim = self.grid_shape[i]
            # 创建一个 (3, dim) 的全1矩阵，数据类型为 np.int8
            data = np.ones([3, dim], dtype=np.int8)
            # 将第二行的所有元素乘以-2
            data[1, :] *= -2

            # 根据边界条件进行修改
            if self.boundary_conditions == 'neumann':
                # 如果是 Neumann 边界条件，调整第二行的首尾元素
                data[1, 0] = -1
                data[1, -1] = -1

            # 创建一个以 data 为对角线的稀疏矩阵，对角线偏移为 [-1, 0, 1]，形状为 (dim, dim)，数据类型为 np.int8
            L_i = dia_array((data, [-1, 0, 1]), shape=(dim, dim),
                            dtype=np.int8)

            # 如果是周期边界条件，添加额外的对角线项
            if self.boundary_conditions == 'periodic':
                t = dia_array((dim, dim), dtype=np.int8)
                t.setdiag([1], k=-dim+1)
                t.setdiag([1], k=dim-1)
                L_i += t

            # 根据当前维度之前的维度数量，使用 kron 函数扩展 L_i 的维度
            for j in range(i):
                L_i = kron(eye(self.grid_shape[j], dtype=np.int8), L_i)
            # 根据当前维度之后的维度数量，使用 kron 函数扩展 L_i 的维度
            for j in range(i + 1, N):
                L_i = kron(L_i, eye(self.grid_shape[j], dtype=np.int8))
            # 将扩展后的 L_i 加到 L 上
            L += L_i

        # 将结果转换为指定的数据类型并返回
        return L.astype(self.dtype)

    def _matvec(self, x):
        # 获取网格形状
        grid_shape = self.grid_shape
        # 获取网格维度数
        N = len(grid_shape)
        # 将输入向量 x 重塑为网格形状 + 最后一个维度为 -1 的数组
        X = x.reshape(grid_shape + (-1,))
        # 初始化 Y 为 -2 * N * X
        Y = -2 * N * X
        # 遍历每一个维度
        for i in range(N):
            # 在每个维度上对 X 进行向前和向后滚动操作，并累加到 Y 上
            Y += np.roll(X, 1, axis=i)
            Y += np.roll(X, -1, axis=i)
            # 如果是 Neumann 或 Dirichlet 边界条件，进行特定的边界值处理
            if self.boundary_conditions in ('neumann', 'dirichlet'):
                Y[(slice(None),)*i + (0,) + (slice(None),)*(N-i-1)] -= np.roll(X, 1, axis=i)[(slice(None),) * i + (0,) + (slice(None),) * (N-i-1)]
                Y[(slice(None),)*i + (-1,) + (slice(None),)*(N-i-1)] -= np.roll(X, -1, axis=i)[(slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)]

                # 如果是 Neumann 边界条件，额外处理中心元素
                if self.boundary_conditions == 'neumann':
                    Y[(slice(None),)*i + (0,) + (slice(None),)*(N-i-1)] += np.roll(X, 0, axis=i)[(slice(None),) * i + (0,) + (slice(None),) * (N-i-1)]
                    Y[(slice(None),)*i + (-1,) + (slice(None),)*(N-i-1)] += np.roll(X, 0, axis=i)[(slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)]

        # 将 Y 重塑为 (-1, X.shape[-1]) 的形状并返回
        return Y.reshape(-1, X.shape[-1])

    def _matmat(self, x):
        # 调用 _matvec 函数并返回结果
        return self._matvec(x)

    def _adjoint(self):
        # 返回自身，表示其共轭转置
        return self

    def _transpose(self):
        # 返回自身，表示其转置
        return self
class Sakurai(LinearOperator):
    """
    构造不同格式的Sakurai矩阵及其特征值。

    参考文献[1]_中描述的"Sakurai"矩阵：
    方阵、实对称正定、五对角线，
    主对角线为``[5, 6, 6, ..., 6, 6, 5]``，
    ``+1``和``-1``对角线为``-4``，
    ``+2``和``-2``对角线为``1``。
    其特征值已知为
    ``16. * np.power(np.cos(0.5 * k * np.pi / (n + 1)), 4)``。
    随着尺寸增加，矩阵的条件数变差。
    用于测试和基准测试稀疏特征值求解器，
    特别是那些利用其五对角结构的求解器。
    详见下面的注释。

    参数
    ----------
    n : int
        矩阵的大小。
    dtype : dtype
        数组的数值类型，默认为``np.int8``。

    方法
    -------
    toarray()
        从Laplacian数据构造密集数组
    tosparse()
        从Laplacian数据构造稀疏数组
    tobanded()
        Sakurai矩阵的带状对称矩阵格式，
        即（3，n）ndarray，带有3个上对角线，
        主对角线位于底部。
    eigenvalues
        Sakurai矩阵的所有特征值，按升序排序。

    注释
    -----
    参考文献[1]_介绍了矩阵对`A`和`B`的广义特征值问题，
    其中`A`是单位矩阵，因此我们将其转化为仅与矩阵`B`相关的特征值问题，
    该函数输出的矩阵可以用不同的格式表示及其特征值。

    .. versionadded:: 1.12.0

    引用
    ----------
    .. [1] T. Sakurai, H. Tadano, Y. Inadomi, and U. Nagashima,
       "A moment-based method for large-scale generalized
       eigenvalue problems",
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).

    示例
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import Sakurai
    >>> from scipy.linalg import eig_banded
    >>> n = 6
    >>> sak = Sakurai(n)

    由于所有矩阵条目都是小整数，“int8”是存储矩阵表示的默认dtype。

    >>> sak.toarray()
    array([[ 5, -4,  1,  0,  0,  0],
           [-4,  6, -4,  1,  0,  0],
           [ 1, -4,  6, -4,  1,  0],
           [ 0,  1, -4,  6, -4,  1],
           [ 0,  0,  1, -4,  6, -4],
           [ 0,  0,  0,  1, -4,  5]], dtype=int8)
    >>> sak.tobanded()
    array([[ 1,  1,  1,  1,  1,  1],
           [-4, -4, -4, -4, -4, -4],
           [ 5,  6,  6,  6,  6,  5]], dtype=int8)
    >>> sak.tosparse()
    <DIAgonal sparse array of dtype 'int8'
        with 24 stored elements (5 diagonals) and shape (6, 6)>
    >>> np.array_equal(sak.dot(np.eye(n)), sak.tosparse().toarray())
    True
    >>> sak.eigenvalues()
    array([0.03922866, 0.56703972, 2.41789479, 5.97822974,
           10.54287655, 14.45473055])
    >>> sak.eigenvalues(2)
    array([0.03922866, 0.56703972])

    The banded form can be used in scipy functions for banded matrices, e.g.,

    >>> e = eig_banded(sak.tobanded(), eigvals_only=True)
    >>> np.allclose(sak.eigenvalues, e, atol= n * n * n * np.finfo(float).eps)
    True

    """
    # 创建一个名为 SakuraiMatrix 的类
    def __init__(self, n, dtype=np.int8):
        # 初始化方法，设置矩阵的大小 n 和数据类型 dtype
        self.n = n
        self.dtype = dtype
        shape = (n, n)
        super().__init__(dtype, shape)

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.float64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        if m is None:
            m = self.n
        k = np.arange(self.n + 1 -m, self.n + 1)
        # 计算 Sakurai 矩阵的特征值
        return np.flip(16. * np.power(np.cos(0.5 * k * np.pi / (self.n + 1)), 4))

    def tobanded(self):
        """
        Construct the Sakurai matrix as a banded array.
        """
        # 构造成带状数组的 Sakurai 矩阵
        d0 = np.r_[5, 6 * np.ones(self.n - 2, dtype=self.dtype), 5]
        d1 = -4 * np.ones(self.n, dtype=self.dtype)
        d2 = np.ones(self.n, dtype=self.dtype)
        return np.array([d2, d1, d0]).astype(self.dtype)

    def tosparse(self):
        """
        Construct the Sakurai matrix is a sparse format.
        """
        from scipy.sparse import spdiags
        d = self.tobanded()
        # 使用 spdiags 函数构造稀疏格式的 Sakurai 矩阵
        # 带状格式中主对角线位于底部
        # spdiags 函数没有 dtype 参数，因此从带状数组继承 dtype
        return spdiags([d[0], d[1], d[2], d[1], d[0]], [-2, -1, 0, 1, 2],
                       self.n, self.n)

    def toarray(self):
        # 返回 Sakurai 矩阵的稀疏格式表示的密集数组形式
        return self.tosparse().toarray()
    
    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the 5-diagonal format.
        """
        # 实现对向量 x 的 Sakurai 矩阵乘法，无需实际构造或存储矩阵
        x = x.reshape(self.n, -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        sx = np.zeros_like(x, dtype=result_dtype)
        sx[0, :] = 5 * x[0, :] - 4 * x[1, :] + x[2, :]
        sx[-1, :] = 5 * x[-1, :] - 4 * x[-2, :] + x[-3, :]
        sx[1: -1, :] = (6 * x[1: -1, :] - 4 * (x[:-2, :] + x[2:, :])
                      + np.pad(x[:-3, :], ((1, 0), (0, 0)))
                      + np.pad(x[3:, :], ((0, 1), (0, 0))))
        return sx

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """        
        # 实现对矩阵 x 的 Sakurai 矩阵乘法，无需实际构造或存储矩阵本身，利用 _matvec(x) 支持 1D 和 2D 数组的特性
        return self._matvec(x)
    # 返回自身对象，用于伴随矩阵计算
    def _adjoint(self):
        return self

    # 返回自身对象，用于转置矩阵计算
    def _transpose(self):
        return self
class MikotaM(LinearOperator):
    """
    Construct a mass matrix in various formats of Mikota pair.

    The mass matrix `M` is square real diagonal
    positive definite with entries that are reciprocal to integers.

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (1, n) ndarray with the main diagonal.
    """
    def __init__(self, shape, dtype=np.float64):
        # 初始化函数，设置矩阵形状和数据类型，并调用父类的构造函数
        self.shape = shape
        self.dtype = dtype
        super().__init__(dtype, shape)

    def _diag(self):
        # 从对角线上的 1 / [1, ..., N+1] 构建矩阵;
        # 使用函数计算，避免重复代码和存储开销
        return (1. / np.arange(1, self.shape[0] + 1)).astype(self.dtype)

    def tobanded(self):
        # 返回矩阵的带状格式表示
        return self._diag()

    def tosparse(self):
        # 导入必要的库，并使用 diags 函数创建稀疏矩阵
        from scipy.sparse import diags
        return diags([self._diag()], [0], shape=self.shape, dtype=self.dtype)

    def toarray(self):
        # 将矩阵表示为密集数组
        return np.diag(self._diag()).astype(self.dtype)

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the diagonal format.
        """
        # 通过矩阵向量乘法计算矩阵与向量的乘积，利用对角线上的值
        x = x.reshape(self.shape[0], -1)
        return self._diag()[:, np.newaxis] * x

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """     
        # 利用 _matvec 方法实现矩阵乘法
        return self._matvec(x)

    def _adjoint(self):
        # 返回自身的共轭转置
        return self

    def _transpose(self):
        # 返回自身的转置
        return self


class MikotaK(LinearOperator):
    """
    Construct a stiffness matrix in various formats of Mikota pair.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite with integer entries. 

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.int32``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (2, n) ndarray with 2 upper diagonals
        placing the main diagonal at the bottom.
    """
    def __init__(self, shape, dtype=np.int32):
        # 初始化函数，设置矩阵的形状和数据类型，默认为 np.int32
        self.shape = shape
        self.dtype = dtype
        # 调用父类的初始化方法，传入数据类型和形状参数
        super().__init__(dtype, shape)
        # 为了避免重复计算，预先计算矩阵的对角线元素
        n = shape[0]
        # 计算主对角线的元素
        self._diag0 = np.arange(2 * n - 1, 0, -2, dtype=self.dtype)
        # 计算次对角线的元素
        self._diag1 = - np.arange(n - 1, 0, -1, dtype=self.dtype)

    def tobanded(self):
        # 将矩阵转换为带状矩阵表示
        return np.array([np.pad(self._diag1, (1, 0), 'constant'), self._diag0])

    def tosparse(self):
        # 将矩阵转换为稀疏矩阵表示，使用了 scipy.sparse.diags 函数
        from scipy.sparse import diags
        return diags([self._diag1, self._diag0, self._diag1], [-1, 0, 1],
                     shape=self.shape, dtype=self.dtype)

    def toarray(self):
        # 将稀疏矩阵表示转换为密集数组表示
        return self.tosparse().toarray()

    def _matvec(self, x):
        """
        构建无需构建或存储矩阵本身的可调用的带状矩阵向量乘法，
        使用其条目的知识和三对角格式。
        """
        x = x.reshape(self.shape[0], -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        kx = np.zeros_like(x, dtype=result_dtype)
        d1 = self._diag1
        d0 = self._diag0
        # 计算矩阵向量乘法的结果
        kx[0, :] = d0[0] * x[0, :] + d1[0] * x[1, :]
        kx[-1, :] = d1[-1] * x[-2, :] + d0[-1] * x[-1, :]
        kx[1: -1, :] = (d1[:-1, None] * x[: -2, :]
                        + d0[1: -1, None] * x[1: -1, :]
                        + d1[1:, None] * x[2:, :])
        return kx

    def _matmat(self, x):
        """
        构建无需构建或存储矩阵本身的可调用的矩阵-矩阵乘法，
        通过重用支持 1D 和 2D 数组 x 的 _matvec(x) 方法。
        """  
        return self._matvec(x)

    def _adjoint(self):
        # 返回自身的伴随（共轭转置）
        return self

    def _transpose(self):
        # 返回自身的转置
        return self
class MikotaPair:
    """
    Construct the Mikota pair of matrices in various formats and
    eigenvalues of the generalized eigenproblem with them.

    The Mikota pair of matrices [1, 2]_ models a vibration problem
    of a linear mass-spring system with the ends attached where
    the stiffness of the springs and the masses increase along
    the system length such that vibration frequencies are subsequent
    integers 1, 2, ..., `n` where `n` is the number of the masses. Thus,
    eigenvalues of the generalized eigenvalue problem for
    the matrix pair `K` and `M` where `K` is the system stiffness matrix
    and `M` is the system mass matrix are the squares of the integers,
    i.e., 1, 4, 9, ..., ``n * n``.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite. The mass matrix `M` is diagonal with diagonal
    entries 1, 1/2, 1/3, ...., ``1/n``. Both matrices get
    ill-conditioned with `n` growing.

    Parameters
    ----------
    n : int
        The size of the matrices of the Mikota pair.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Attributes
    ----------
    eigenvalues : 1D ndarray, ``np.uint64``
        All eigenvalues of the Mikota pair ordered ascending.

    Methods
    -------
    MikotaK()
        A `LinearOperator` custom object for the stiffness matrix.
    MikotaM()
        A `LinearOperator` custom object for the mass matrix.
    
    .. versionadded:: 1.12.0

    References
    ----------
    .. [1] J. Mikota, "Frequency tuning of chain structure multibody oscillators
       to place the natural frequencies at omega1 and N-1 integer multiples
       omega2,..., omegaN", Z. Angew. Math. Mech. 81 (2001), S2, S201-S202.
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).
    .. [2] Peter C. Muller and Metin Gurgoze,
       "Natural frequencies of a multi-degree-of-freedom vibration system",
       Proc. Appl. Math. Mech. 6, 319-320 (2006).
       http://dx.doi.org/10.1002/pamm.200610141.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
    >>> n = 6
    >>> mik = MikotaPair(n)
    >>> mik_k = mik.k
    >>> mik_m = mik.m
    >>> mik_k.toarray()
    array([[11., -5.,  0.,  0.,  0.,  0.],
           [-5.,  9., -4.,  0.,  0.,  0.],
           [ 0., -4.,  7., -3.,  0.,  0.],
           [ 0.,  0., -3.,  5., -2.,  0.],
           [ 0.,  0.,  0., -2.,  3., -1.],
           [ 0.,  0.,  0.,  0., -1.,  1.]])
    >>> mik_k.tobanded()
    array([[ 0., -5., -4., -3., -2., -1.],
           [11.,  9.,  7.,  5.,  3.,  1.]])
    >>> mik_m.tobanded()
    array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ,
        0.16666667])
    >>> mik_k.tosparse()
    <DIAgonal sparse array of dtype 'float64'
        with 20 stored elements (3 diagonals) and shape (6, 6)>
    >>> mik_m.tosparse()
    """
    
    def __init__(self, n, dtype=np.float64):
        # Initialize Mikota pair with size n and specified data type
        self.n = n
        self.dtype = dtype
        
        # Calculate eigenvalues of the Mikota pair
        self.eigenvalues = np.array([i * i for i in range(1, n + 1)], dtype=np.uint64)
    
    def MikotaK(self):
        # Return a `LinearOperator` custom object for the stiffness matrix K
        return LinearOperator(...)
    
    def MikotaM(self):
        # Return a `LinearOperator` custom object for the mass matrix M
        return LinearOperator(...)
    <DIAgonal sparse array of dtype 'float64'
        with 6 stored elements (1 diagonals) and shape (6, 6)>

这行代码是一个示例输出，显示一个稀疏对角矩阵的信息，不影响程序运行。


    >>> np.array_equal(mik_k(np.eye(n)), mik_k.toarray())

调用 `mik_k` 对象的方法 `np.eye(n)` 创建单位矩阵，并比较其与 `mik_k` 对象的稀疏矩阵表示之间是否相等。


    True

验证前一行代码返回的比较结果为真。


    >>> np.array_equal(mik_m(np.eye(n)), mik_m.toarray())

调用 `mik_m` 对象的方法 `np.eye(n)` 创建单位矩阵，并比较其与 `mik_m` 对象的稀疏矩阵表示之间是否相等。


    True

验证前一行代码返回的比较结果为真。


    >>> mik.eigenvalues()

调用 `mik` 对象的 `eigenvalues()` 方法，返回所有特征值。


    array([ 1,  4,  9, 16, 25, 36])  

显示 `mik.eigenvalues()` 方法返回的特征值数组。


    >>> mik.eigenvalues(2)

调用 `mik` 对象的 `eigenvalues()` 方法，请求返回最小的两个特征值。


    array([ 1,  4])

显示 `mik.eigenvalues(2)` 方法返回的特征值数组。


    """
    def __init__(self, n, dtype=np.float64):

定义 `Mikota` 类的构造函数，接受参数 `n` 和可选参数 `dtype`，默认为 `np.float64` 类型。


        self.n = n
        self.dtype = dtype
        self.shape = (n, n)

在 `Mikota` 类的实例中，设置 `n`、`dtype` 和 `shape` 属性。


        self.m = MikotaM(self.shape, self.dtype)
        self.k = MikotaK(self.shape, self.dtype)

创建 `MikotaM` 和 `MikotaK` 类的实例，并将其分别赋值给 `self.m` 和 `self.k` 属性。


    def eigenvalues(self, m=None):

定义 `eigenvalues` 方法，接受参数 `m`，用于返回特定数量的特征值。


        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.uint64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """

方法 `eigenvalues` 的文档字符串，解释了方法接受的参数和返回的特征值数组的类型和顺序。


        if m is None:
            m = self.n

如果参数 `m` 为 `None`，则将其设置为 `self.n`，即返回所有特征值。


        arange_plus1 = np.arange(1, m + 1, dtype=np.uint64)

创建一个从 1 到 `m`（包括 `m`）的整数数组，数据类型为 `np.uint64`。


        return arange_plus1 * arange_plus1

返回一个数组，其中包含了从 1 到 `m` 的整数的平方，作为特征值。
```