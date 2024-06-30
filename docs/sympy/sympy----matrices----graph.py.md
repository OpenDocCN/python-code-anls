# `D:\src\scipysrc\sympy\sympy\matrices\graph.py`

```
from sympy.utilities.iterables import \
    flatten, connected_components, strongly_connected_components
from .exceptions import NonSquareMatrixError

def _connected_components(M):
    """Returns the list of connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])
    >>> A.connected_components()
    [[0, 3, 8], [1, 6, 7], [2, 4, 5]]

    Notes
    =====

    Even if any symbolic elements of the matrix can be indeterminate
    to be zero mathematically, this only takes the account of the
    structural aspect of the matrix, so they will considered to be
    nonzero.
    """
    if not M.is_square:
        raise NonSquareMatrixError

    # 获取图的顶点集合和边集合，顶点集合为矩阵的行数范围，边集合为非零元素的位置
    V = range(M.rows)
    E = sorted(M.todok().keys())
    # 返回由连接组件组成的列表
    return connected_components((V, E))


def _strongly_connected_components(M):
    """Returns the list of strongly connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])
    >>> A.strongly_connected_components()
    [[0, 4, 6], [2, 3, 7], [1, 5, 8]]
    """
    if not M.is_square:
        raise NonSquareMatrixError

    # 如果矩阵 M 包含属性 _rep，则使用更高效的方法 rep.scc() 进行计算
    rep = getattr(M, '_rep', None)
    if rep is not None:
        return rep.scc()

    # 否则，获取图的顶点集合和边集合，顶点集合为矩阵的行数范围，边集合为非零元素的位置
    V = range(M.rows)
    E = sorted(M.todok().keys())
    # 返回由强连接组件组成的列表
    return strongly_connected_components((V, E))


def _connected_components_decomposition(M):
    """Decomposes a square matrix into block diagonal form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Returns
    =======
    """
    P, B : PermutationMatrix, BlockDiagMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block diagonal matrix of
        the result of the permutation.

        If you would like to get the diagonal blocks from the
        BlockDiagMatrix, see
        :meth:`~sympy.matrices.expressions.blockmatrix.BlockDiagMatrix.get_diag_blocks`.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])

    >>> P, B = A.connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((1 3)(2 8 5 7 4 6))
    >>> pprint(B)
    [[66  68  67]                            ]
    [[          ]                            ]
    [[86  88  87]       0             0      ]
    [[          ]                            ]
    [[76  78  77]                            ]
    [                                        ]
    [              [55  54  53]              ]
    [              [          ]              ]
    [     0        [45  44  43]       0      ]
    [              [          ]              ]
    [              [35  34  33]              ]
    [                                        ]
    [                            [0   1   2 ]]
    [                            [          ]]
    [     0             0        [10  11  12]]
    [                            [          ]]
    [                            [20  21  22]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T*B*P == A
    True

    Notes
    =====

    This problem corresponds to the finding of the connected components
    of a graph, when a matrix is viewed as a weighted graph.
    """
    from sympy.combinatorics.permutations import Permutation  # 导入置换类
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix  # 导入块对角矩阵类
    from sympy.matrices.expressions.permutation import PermutationMatrix  # 导入置换矩阵类

    # 调用矩阵 M 的 connected_components 方法，获取其连接分量
    iblocks = M.connected_components()

    # 将连接分量 iblocks 展平，并创建置换对象 p
    p = Permutation(flatten(iblocks))
    # 创建置换矩阵 P
    P = PermutationMatrix(p)

    # 初始化空列表 blocks 用于存储块矩阵
    blocks = []
    # 遍历 iblocks 中的每个块索引 b
    for b in iblocks:
        # 将矩阵 M 的块 M[b, b] 添加到 blocks 列表中
        blocks.append(M[b, b])
    # 创建块对角矩阵 B，传入 blocks 列表作为参数
    B = BlockDiagMatrix(*blocks)
    # 返回置换矩阵 P 和块对角矩阵 B
    return P, B
# 将一个方阵分解为块三角形式，仅使用排列。
def _strongly_connected_components_decomposition(M, lower=True):
    """Decomposes a square matrix into block triangular form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Parameters
    ==========

    lower : bool
        Makes $B$ lower block triangular when ``True``.
        Otherwise, makes $B$ upper block triangular.

    Returns
    =======

    P, B : PermutationMatrix, BlockMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block triangular matrix of
        the result of the permutation.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])

    A lower block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((8)(1 4 3 2 6)(5 7))
    >>> pprint(B)
    [[44  43  45]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[34  33  35]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[54  53  55]   [0  0  0]     [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [62  61  60]  [66  68  67]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [82  81  80]  [86  88  87]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [72  71  70]  [76  78  77]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True

    An upper block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition(lower=False)
    >>> pprint(P)
    PermutationMatrix((0 1 5 7 4 3 2 8 6))
    >>> pprint(B)
    [[66  68  67]  [62  61  60]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[86  88  87]  [82  81  80]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[76  78  77]  [72  71  70]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [
        [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
        [                                        ]
        [ [0  0  0]     [0  0  0]    [44  43  45]]
        [ [       ]     [       ]    [          ]]
        [ [0  0  0]     [0  0  0]    [34  33  35]]
        [ [       ]     [       ]    [          ]]
        [ [0  0  0]     [0  0  0]    [54  53  55]]
    """

# 导入所需的库和模块
from sympy.combinatorics.permutations import Permutation
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.permutation import PermutationMatrix

# 定义函数，参数为M（假定为某种矩阵结构）和lower（布尔值，表示是否考虑下三角部分）
def function_name(M, lower=False):
    # 使用M的强连通分量计算iblocks
    iblocks = M.strongly_connected_components()
    # 如果lower为False，反转iblocks列表
    if not lower:
        iblocks = list(reversed(iblocks))
    
    # 根据iblocks中的元素创建置换p
    p = Permutation(flatten(iblocks))
    # 根据置换p创建置换矩阵P
    P = PermutationMatrix(p)

    # 初始化空列表rows用于存储行
    rows = []
    # 遍历iblocks中的元素a
    for a in iblocks:
        # 初始化空列表cols用于存储列
        cols = []
        # 再次遍历iblocks中的元素b
        for b in iblocks:
            # 将M中(a, b)位置的元素添加到cols列表中
            cols.append(M[a, b])
        # 将cols列表添加到rows列表中作为新行
        rows.append(cols)
    
    # 根据rows列表创建块矩阵B
    B = BlockMatrix(rows)
    
    # 返回置换矩阵P和块矩阵B作为结果
    return P, B
```