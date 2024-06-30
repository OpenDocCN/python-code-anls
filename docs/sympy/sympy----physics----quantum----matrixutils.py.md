# `D:\src\scipysrc\sympy\sympy\physics\quantum\matrixutils.py`

```
# 导入所需模块和类
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module

__all__ = [
    'numpy_ndarray',
    'scipy_sparse_matrix',
    'sympy_to_numpy',
    'sympy_to_scipy_sparse',
    'numpy_to_sympy',
    'scipy_sparse_to_sympy',
    'flatten_scalar',
    'matrix_dagger',
    'to_sympy',
    'to_numpy',
    'to_scipy_sparse',
    'matrix_tensor_product',
    'matrix_zeros'
]

# 定义条件导入的 numpy 和 scipy.sparse 的基类，用于 isinstance 测试

# 尝试导入 numpy 模块
np = import_module('numpy')
# 如果导入失败，定义一个空的类 numpy_ndarray
if not np:
    class numpy_ndarray:
        pass
else:
    # 如果成功导入，将 numpy.ndarray 赋给 numpy_ndarray
    numpy_ndarray = np.ndarray  # type: ignore

# 尝试导入 scipy 模块，设置从中导入 sparse 子模块
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
# 如果导入失败，定义一个空的类 scipy_sparse_matrix，并设置 sparse 为 None
if not scipy:
    class scipy_sparse_matrix:
        pass
    sparse = None
else:
    # 如果成功导入，将 scipy.sparse 赋给 sparse，并将其 spmatrix 类赋给 scipy_sparse_matrix
    sparse = scipy.sparse
    scipy_sparse_matrix = sparse.spmatrix  # type: ignore


def sympy_to_numpy(m, **options):
    """将 SymPy 矩阵/复数转换为 numpy 矩阵或标量。"""
    # 如果未成功导入 numpy 模块，抛出 ImportError 异常
    if not np:
        raise ImportError
    # 获取选项中的 dtype，默认为 'complex'
    dtype = options.get('dtype', 'complex')
    # 如果 m 是 MatrixBase 类型，则将其转换为 numpy 数组并指定 dtype
    if isinstance(m, MatrixBase):
        return np.array(m.tolist(), dtype=dtype)
    # 如果 m 是 Expr 类型，检查其是否是数字或复数，若是则转换为复数
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    # 若不是期望的类型，则抛出 TypeError 异常
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def sympy_to_scipy_sparse(m, **options):
    """将 SymPy 矩阵/复数转换为 scipy 稀疏矩阵。"""
    # 如果未成功导入 numpy 或 scipy.sparse 模块，抛出 ImportError 异常
    if not np or not sparse:
        raise ImportError
    # 获取选项中的 dtype，默认为 'complex'
    dtype = options.get('dtype', 'complex')
    # 如果 m 是 MatrixBase 类型，则先转换为 numpy 数组，再转换为 scipy 稀疏矩阵
    if isinstance(m, MatrixBase):
        return sparse.csr_matrix(np.array(m.tolist(), dtype=dtype))
    # 如果 m 是 Expr 类型，检查其是否是数字或复数，若是则转换为复数
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    # 若不是期望的类型，则抛出 TypeError 异常
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def scipy_sparse_to_sympy(m, **options):
    """将 scipy 稀疏矩阵转换为 SymPy 矩阵。"""
    return MatrixBase(m.todense())


def numpy_to_sympy(m, **options):
    """将 numpy 矩阵转换为 SymPy 矩阵。"""
    return MatrixBase(m)


def to_sympy(m, **options):
    """将 numpy/scipy.sparse 矩阵转换为 SymPy 矩阵。"""
    # 如果 m 已经是 MatrixBase 类型，则直接返回
    if isinstance(m, MatrixBase):
        return m
    # 如果 m 是 numpy_ndarray 类型，则转换为 SymPy 矩阵
    elif isinstance(m, numpy_ndarray):
        return numpy_to_sympy(m)
    # 如果 m 是 scipy_sparse_matrix 类型，则转换为 SymPy 矩阵
    elif isinstance(m, scipy_sparse_matrix):
        return scipy_sparse_to_sympy(m)
    # 如果 m 是 Expr 类型，则直接返回
    elif isinstance(m, Expr):
        return m
    # 若不是期望的类型，则抛出 TypeError 异常
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def to_numpy(m, **options):
    """将 sympy/scipy.sparse 矩阵转换为 numpy 矩阵。"""
    # 获取选项中的 dtype，默认为 'complex'
    dtype = options.get('dtype', 'complex')
    # 如果 m 是 MatrixBase 或 Expr 类型，则转换为 numpy 矩阵
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_numpy(m, dtype=dtype)
    # 如果 m 是 numpy 的 ndarray 类型，则直接返回 m
    elif isinstance(m, numpy_ndarray):
        return m
    # 如果 m 是 scipy 的稀疏矩阵类型，则将其转换为密集矩阵并返回
    elif isinstance(m, scipy_sparse_matrix):
        return m.todense()
    # 如果 m 不是预期的 sympy/numpy/scipy.sparse 矩阵类型，则抛出类型错误并显示 m 的类型信息
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)
# 将 sympy/numpy 矩阵转换为 scipy.sparse 矩阵
def to_scipy_sparse(m, **options):
    # 从 options 中获取 dtype，默认为复数类型
    dtype = options.get('dtype', 'complex')
    # 如果 m 是 sympy 的 MatrixBase 或 Expr 类型，则调用 sympy_to_scipy_sparse 函数进行转换
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_scipy_sparse(m, dtype=dtype)
    # 如果 m 是 numpy 的 ndarray 类型
    elif isinstance(m, numpy_ndarray):
        # 如果 sparse 模块未导入，则抛出 ImportError
        if not sparse:
            raise ImportError
        # 将 numpy 数组 m 转换为 scipy 的 csr_matrix 格式
        return sparse.csr_matrix(m)
    # 如果 m 已经是 scipy 的 sparse_matrix 类型，则直接返回 m
    elif isinstance(m, scipy_sparse_matrix):
        return m
    # 如果 m 不是期望的类型，则抛出 TypeError
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


# 将 1x1 矩阵展平为标量，保持更大的矩阵不变
def flatten_scalar(e):
    # 如果 e 是 sympy 的 MatrixBase 类型且形状为 (1, 1)，则将其转换为标量
    if isinstance(e, MatrixBase):
        if e.shape == (1, 1):
            e = e[0]
    # 如果 e 是 numpy 的 ndarray 或 scipy 的 sparse_matrix 类型且形状为 (1, 1)，则将其转换为复数标量
    if isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        if e.shape == (1, 1):
            e = complex(e[0, 0])
    return e


# 返回 sympy/numpy/scipy.sparse 矩阵的共轭转置
def matrix_dagger(e):
    # 如果 e 是 sympy 的 MatrixBase 类型，则返回其共轭转置
    if isinstance(e, MatrixBase):
        return e.H
    # 如果 e 是 numpy 的 ndarray 或 scipy 的 sparse_matrix 类型，则返回其共轭转置
    elif isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        return e.conjugate().transpose()
    # 如果 e 不是期望的类型，则抛出 TypeError
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % e)


# TODO: Move this into sympy.matricies.
# 计算一系列 SymPy 矩阵的 Kronecker 乘积
def _sympy_tensor_product(*matrices):
    """Compute the kronecker product of a sequence of SymPy Matrices.
    """
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product

    return matrix_kronecker_product(*matrices)


# numpy 版本的多个参数的张量积计算
def _numpy_tensor_product(*product):
    """numpy version of tensor product of multiple arguments."""
    if not np:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = np.kron(answer, item)
    return answer


# scipy.sparse 版本的多个参数的张量积计算
def _scipy_sparse_tensor_product(*product):
    """scipy.sparse version of tensor product of multiple arguments."""
    if not sparse:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = sparse.kron(answer, item)
    # 最终的矩阵将被相乘，因此使用 csr 是一个很好的最终稀疏格式
    return sparse.csr_matrix(answer)


# 计算 sympy/numpy/scipy.sparse 矩阵的矩阵张量积
def matrix_tensor_product(*product):
    if isinstance(product[0], MatrixBase):
        return _sympy_tensor_product(*product)
    elif isinstance(product[0], numpy_ndarray):
        return _numpy_tensor_product(*product)
    elif isinstance(product[0], scipy_sparse_matrix):
        return _scipy_sparse_tensor_product(*product)


# numpy 版本的复数单位矩阵
def _numpy_eye(n):
    """numpy version of complex eye."""
    if not np:
        raise ImportError
    return np.array(np.eye(n, dtype='complex'))


# scipy.sparse 版本的复数单位矩阵
def _scipy_sparse_eye(n):
    """scipy.sparse version of complex eye."""
    if not sparse:
        raise ImportError
    return sparse.eye(n, n, dtype='complex')


# 获取给定格式的单位矩阵和张量积版本
def matrix_eye(n, **options):
    """Get the version of eye and tensor_product for a given format."""
    format = options.get('format', 'sympy')
    # 如果格式为 'sympy'，返回一个 sympy 库中的单位矩阵
    if format == 'sympy':
        return eye(n)
    # 如果格式为 'numpy'，返回一个使用 numpy 库生成的单位矩阵
    elif format == 'numpy':
        return _numpy_eye(n)
    # 如果格式为 'scipy.sparse'，返回一个使用 scipy.sparse 库生成的稀疏单位矩阵
    elif format == 'scipy.sparse':
        return _scipy_sparse_eye(n)
    # 如果格式不是以上三种情况，抛出一个未实现的错误，显示无效的格式
    raise NotImplementedError('Invalid format: %r' % format)
# 使用 numpy 库创建一个给定大小的零矩阵，可选的数据类型由参数 dtype 指定，默认为 'float64'
def _numpy_zeros(m, n, **options):
    dtype = options.get('dtype', 'float64')
    # 如果没有导入 numpy 库，则抛出 ImportError 异常
    if not np:
        raise ImportError
    # 返回一个 m 行 n 列的零矩阵，数据类型为 dtype
    return np.zeros((m, n), dtype=dtype)


# 使用 scipy.sparse 库创建一个给定大小的零矩阵，稀疏矩阵格式由参数 spmatrix 指定，默认为 'csr'，数据类型由参数 dtype 指定，默认为 'float64'
def _scipy_sparse_zeros(m, n, **options):
    spmatrix = options.get('spmatrix', 'csr')
    dtype = options.get('dtype', 'float64')
    # 如果没有导入 scipy.sparse 库，则抛出 ImportError 异常
    if not sparse:
        raise ImportError
    # 根据 spmatrix 参数的值选择创建稀疏矩阵的方式
    if spmatrix == 'lil':
        return sparse.lil_matrix((m, n), dtype=dtype)
    elif spmatrix == 'csr':
        return sparse.csr_matrix((m, n), dtype=dtype)


# 根据指定的格式创建一个零矩阵
def matrix_zeros(m, n, **options):
    # 格式由参数 format 指定，默认为 'sympy'
    format = options.get('format', 'sympy')
    # 根据指定的格式选择创建对应的零矩阵
    if format == 'sympy':
        return zeros(m, n)
    elif format == 'numpy':
        return _numpy_zeros(m, n, **options)
    elif format == 'scipy.sparse':
        return _scipy_sparse_zeros(m, n, **options)
    # 如果指定的格式不在支持的列表中，则抛出 NotImplementedError 异常
    raise NotImplementedError('Invalid format: %r' % format)


# 将一个 numpy 零矩阵转换为零标量
def _numpy_matrix_to_zero(e):
    # 如果没有导入 numpy 库，则抛出 ImportError 异常
    if not np:
        raise ImportError
    # 创建一个与 e 相同形状的零矩阵
    test = np.zeros_like(e)
    # 如果 e 和 test 矩阵完全相等，则返回标量 0.0
    if np.allclose(e, test):
        return 0.0
    else:
        return e


# 将一个 scipy.sparse 零矩阵转换为零标量
def _scipy_sparse_matrix_to_zero(e):
    # 如果没有导入 numpy 库，则抛出 ImportError 异常
    if not np:
        raise ImportError
    # 将 scipy.sparse 矩阵 e 转换为密集矩阵 edense
    edense = e.todense()
    # 创建一个与 edense 相同形状的零矩阵
    test = np.zeros_like(edense)
    # 如果 edense 和 test 矩阵完全相等，则返回标量 0.0
    if np.allclose(edense, test):
        return 0.0
    else:
        return e


# 将一个零矩阵转换为零标量
def matrix_to_zero(e):
    # 如果 e 是 MatrixBase 类型，并且它与相同大小的零矩阵相等，则将 e 转换为符号 0
    if isinstance(e, MatrixBase):
        if zeros(*e.shape) == e:
            e = S.Zero
    # 如果 e 是 numpy 数组类型，则调用 _numpy_matrix_to_zero 函数将其转换为零标量
    elif isinstance(e, numpy_ndarray):
        e = _numpy_matrix_to_zero(e)
    # 如果 e 是 scipy.sparse 矩阵类型，则调用 _scipy_sparse_matrix_to_zero 函数将其转换为零标量
    elif isinstance(e, scipy_sparse_matrix):
        e = _scipy_sparse_matrix_to_zero(e)
    # 返回转换后的结果 e
    return e
```