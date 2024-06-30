# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\utils.py`

```
__docformat__ = "restructuredtext en"

__all__ = []

# 导入必要的库函数
from numpy import asanyarray, asarray, array, zeros

# 导入线性代数相关的接口和操作符
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
     IdentityOperator

# 定义类型转换规则的字典
_coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}

# 根据输入的类型进行类型转换
def coerce(x,y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x,y]

# 返回参数本身，用于标识函数
def id(x):
    return x

# 创建线性系统 Ax=b 的函数
def make_system(A, M, x0, b):
    """Make a linear system Ax=b

    Parameters
    ----------
    A : LinearOperator
        sparse or dense matrix (or any valid input to aslinearoperator)
    M : {LinearOperator, Nones}
        preconditioner
        sparse or dense matrix (or any valid input to aslinearoperator)
    x0 : {array_like, str, None}
        initial guess to iterative method.
        ``x0 = 'Mb'`` means using the nonzero initial guess ``M @ b``.
        Default is `None`, which means using the zero initial guess.
    b : array_like
        right hand side

    Returns
    -------
    (A, M, x, b, postprocess)
        A : LinearOperator
            matrix of the linear system
        M : LinearOperator
            preconditioner
        x : rank 1 ndarray
            initial guess
        b : rank 1 ndarray
            right hand side
        postprocess : function
            converts the solution vector to the appropriate
            type and dimensions (e.g. (N,1) matrix)

    """

    # 备份原始输入的 A
    A_ = A
    # 将 A 转换为线性操作符
    A = aslinearoperator(A)

    # 检查 A 是否为方阵
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')

    # 矩阵的维度
    N = A.shape[0]

    # 将 b 转换为数组
    b = asanyarray(b)

    # 检查 A 和 b 的形状是否兼容
    if not (b.shape == (N,1) or b.shape == (N,)):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                         'incompatible')

    # 将 b 的数据类型提升为双精度浮点数类型
    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    # 定义后处理函数，这里为直接返回结果 x
    def postprocess(x):
        return x

    # 获取 A 的数据类型，并将 b 转换为相同类型
    if hasattr(A,'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    # 将 b 转换为与 x 相同的数据类型
    b = asarray(b,dtype=xtype)  # make b the same type as x
    b = b.ravel()

    # 处理预条件器 M
    if M is None:
        # 检查 A 是否有 psolve 和 rpsolve 方法，如果没有，则使用 id 函数
        if hasattr(A_,'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_,'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        # 如果 psolve 和 rpsolve 都是 id 函数，则使用 IdentityOperator
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            # 否则创建 LinearOperator
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        # 如果不是字符串类型的初始值 x0，则将 M 转换为线性操作符
        M = aslinearoperator(M)
        # 检查 A 和 M 的形状是否相同，若不同则抛出数值错误
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    # 设置初始猜测
    if x0 is None:
        # 如果未指定初始值 x0，则创建一个元素类型为 xtype、长度为 N 的零向量
        x = zeros(N, dtype=xtype)
    elif isinstance(x0, str):
        # 如果 x0 是字符串类型
        if x0 == 'Mb':  # 如果 x0 等于 'Mb'，则使用非零的初始猜测 ``M @ b``
            # 复制向量 b 的内容
            bCopy = b.copy()
            # 计算 M @ b，作为初始猜测 x
            x = M.matvec(bCopy)
    else:
        # 如果 x0 不是 None 也不是字符串，则将其转换为 ndarray 类型的数组，并检查其形状是否与 A 的形状兼容
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            # 如果 x 的形状与 A 不兼容，则抛出数值错误
            raise ValueError(f'shapes of A {A.shape} and '
                             f'x0 {x.shape} are incompatible')
        # 将 x 展平为一维数组
        x = x.ravel()

    # 返回修正后的 A、M、x、b 以及后处理函数 postprocess
    return A, M, x, b, postprocess
```