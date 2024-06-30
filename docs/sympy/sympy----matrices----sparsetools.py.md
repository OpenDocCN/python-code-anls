# `D:\src\scipysrc\sympy\sympy\matrices\sparsetools.py`

```
from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent

# 导入稀疏矩阵类 MutableSparseMatrix 作为 SparseMatrix
from .sparse import MutableSparseMatrix as SparseMatrix

# 将 Dictionary Of Keys (DOK) 格式的稀疏矩阵转换为 Compressed Sparse Row (CSR) 格式
def _doktocsr(dok):
    """Converts a sparse matrix to Compressed Sparse Row (CSR) format.

    Parameters
    ==========

    A : contains non-zero elements sorted by key (row, column)
    JA : JA[i] is the column corresponding to A[i]
    IA : IA[i] contains the index in A for the first non-zero element
        of row[i]. Thus IA[i+1] - IA[i] gives number of non-zero
        elements row[i]. The length of IA is always 1 more than the
        number of rows in the matrix.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _doktocsr
    >>> from sympy import SparseMatrix, diag
    >>> m = SparseMatrix(diag(1, 2, 3))
    >>> m[2, 0] = -1
    >>> _doktocsr(m)
    [[1, 2, -1, 3], [0, 1, 0, 2], [0, 1, 2, 4], [3, 3]]

    """
    # 从 DOK 格式中提取行、列、数据，并转换为列表格式
    row, JA, A = [list(i) for i in zip(*dok.row_list())]
    # 初始化 CSR 格式的行指针 IA
    IA = [0]*((row[0] if row else 0) + 1)
    # 遍历每行非零元素的数量，填充到 IA 中
    for i, r in enumerate(row):
        IA.extend([i]*(r - row[i - 1]))  # 如果 i = 0，则不会扩展列表
    # 将 IA 扩展到包含所有行，并指定其长度为行数加一
    IA.extend([len(A)]*(dok.rows - len(IA) + 1))
    # 返回 CSR 格式所需的 A、JA、IA、矩阵形状 shape
    shape = [dok.rows, dok.cols]
    return [A, JA, IA, shape]


def _csrtodok(csr):
    """Converts a CSR representation to DOK representation.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _csrtodok
    >>> _csrtodok([[5, 8, 3, 6], [0, 1, 2, 1], [0, 0, 2, 3, 4], [4, 3]])
    Matrix([
    [0, 0, 0],
    [5, 8, 0],
    [0, 0, 3],
    [0, 6, 0]])

    """
    # 初始化空的 DOK 格式矩阵 smat
    smat = {}
    A, JA, IA, shape = csr
    # 遍历 CSR 格式的 A、JA、IA，填充到 DOK 格式矩阵 smat 中
    for i in range(len(IA) - 1):
        indices = slice(IA[i], IA[i + 1])
        for l, m in zip(A[indices], JA[indices]):
            smat[i, m] = l
    # 返回转换后的 DOK 格式矩阵
    return SparseMatrix(*shape, smat)


def banded(*args, **kwargs):
    """Returns a SparseMatrix from the given dictionary describing
    the diagonals of the matrix. The keys are positive for upper
    diagonals and negative for those below the main diagonal. The
    values may be:

    * expressions or single-argument functions,

    * lists or tuples of values,

    * matrices

    Unless dimensions are given, the size of the returned matrix will
    be large enough to contain the largest non-zero value provided.

    kwargs
    ======

    rows : rows of the resulting matrix; computed if
           not given.

    cols : columns of the resulting matrix; computed if
           not given.

    Examples
    ========

    >>> from sympy import banded, ones, Matrix
    >>> from sympy.abc import x

    If explicit values are given in tuples,
    the matrix will autosize to contain all values, otherwise
    a single value is filled onto the entire diagonal:

    >>> banded({1: (1, 2, 3), -1: (4, 5, 6), 0: x})
    Matrix([
    [x, 1, 0, 0],
    [4, x, 2, 0],
    [0, 5, x, 3],
    [0, 0, 6, x]])

    A function accepting a single argument can be used to fill the
    """
    # 返回根据给定的对角线描述创建的稀疏矩阵 SparseMatrix
    pass  # 该函数在示例中未定义完整的返回内容，因此暂时不需要实现
    # 定义一个函数banded，用于创建带状矩阵。该函数可以根据指定的对角线和值创建一个矩阵。
    def banded(size, data=None, *, cols=None, rows=None):
        # 如果data参数为None，则将size解释为对角线和值的字典。
        if data is None:
            # 将size视为对角线和值的字典，并获取其长度。
            data = size
            size = max(data) + 1
    
        # 如果cols和rows都没有指定，则将cols设置为size，rows设置为size。
        if cols is None and rows is None:
            cols = rows = size
    
        # 初始化一个size行cols列的矩阵，所有元素初始化为0。
        m = zeros(rows, cols)
    
        # 遍历data中的每一个键值对。
        for k, v in data.items():
            # 如果v是函数，则根据对角线索引创建对角线上的元素。
            if callable(v):
                v = Matrix([v(i - k) for i in range(k, min(rows, cols))])
            # 如果v是一个数值或者一个矩阵，则将其放置在指定的对角线上。
            try:
                m.diagonal(k)[:] = v
            except:
                # 如果尝试在指定位置放置v时发生碰撞，抛出一个值错误。
                raise ValueError(f'collision at ({k}, {k})')
    
        # 返回创建的带状矩阵。
        return m
    try:
        # 检查参数个数是否为1、2或3，否则引发 TypeError 异常
        if len(args) not in (1, 2, 3):
            raise TypeError
        # 检查最后一个参数是否为字典类型，否则引发 TypeError 异常
        if not isinstance(args[-1], (dict, Dict)):
            raise TypeError
        # 根据参数个数处理不同的情况
        if len(args) == 1:
            # 如果参数个数为1，从 kwargs 中获取 rows 和 cols，然后转换为整数
            rows = kwargs.get('rows', None)
            cols = kwargs.get('cols', None)
            if rows is not None:
                rows = as_int(rows)
            if cols is not None:
                cols = as_int(cols)
        elif len(args) == 2:
            # 如果参数个数为2，将两个参数都转换为整数并赋值给 rows 和 cols
            rows = cols = as_int(args[0])
        else:
            # 如果参数个数为3，将前两个参数转换为整数分别赋值给 rows 和 cols
            rows, cols = map(as_int, args[:2])
        # 检查 args[-1] 中的所有键是否能转换为整数，否则引发 ValueError 异常
        _ = all(as_int(k) for k in args[-1])
    except (ValueError, TypeError):
        # 捕获 ValueError 或 TypeError 异常，抛出包含详细错误信息的 TypeError 异常
        raise TypeError(filldedent(
            '''unrecognized input to banded:
            expecting [[row,] col,] {int: value}'''))

    def rc(d):
        # 返回对角线起始的行列坐标
        r = -d if d < 0 else 0
        c = 0 if r else d
        return r, c

    smat = {}  # 初始化稀疏矩阵的字典
    undone = []  # 初始化未完成的任务列表
    tba = Dummy()  # 创建一个虚拟对象

    # 首先处理具有尺寸的对象
    for d, v in args[-1].items():
        r, c = rc(d)
        # 注意：只识别列表和元组，允许其他基本对象如 Tuple 进入矩阵
        if isinstance(v, (list, tuple)):
            extra = 0
            for i, vi in enumerate(v):
                i += extra
                if is_sequence(vi):
                    # 如果是序列，则转换为稀疏矩阵 SparseMatrix
                    vi = SparseMatrix(vi)
                    smat[r + i, c + i] = vi
                    extra += min(vi.shape) - 1
                else:
                    smat[r + i, c + i] = vi
        elif is_sequence(v):
            # 如果 v 是序列，则转换为稀疏矩阵 SparseMatrix
            v = SparseMatrix(v)
            rv, cv = v.shape
            if rows and cols:
                nr, xr = divmod(rows - r, rv)
                nc, xc = divmod(cols - c, cv)
                x = xr or xc
                do = min(nr, nc)
            elif rows:
                do, x = divmod(rows - r, rv)
            elif cols:
                do, x = divmod(cols - c, cv)
            else:
                do = 1
                x = 0
            if x:
                raise ValueError(filldedent('''
                    sequence does not fit an integral number of times
                    in the matrix'''))
            j = min(v.shape)
            for i in range(do):
                smat[r, c] = v
                r += j
                c += j
        elif v:
            smat[r, c] = tba
            undone.append((d, v))

    s = SparseMatrix(None, smat)  # 用于扩展矩阵
    smat = s.todok()  # 转换为 dok 格式的稀疏矩阵

    # 检查维度错误
    if rows is not None and rows < s.rows:
        raise ValueError('Designated rows %s < needed %s' % (rows, s.rows))
    if cols is not None and cols < s.cols:
        raise ValueError('Designated cols %s < needed %s' % (cols, s.cols))
    if rows is cols is None:
        rows = s.rows
        cols = s.cols
    elif rows is not None and cols is None:
        cols = max(rows, s.cols)
    # 如果列数不为空且行数为空，则将行数设为列数和当前稀疏矩阵行数的最大值
    elif cols is not None and rows is None:
        rows = max(cols, s.rows)

    # 定义一个更新函数，用于更新稀疏矩阵，并确保没有冲突
    def update(i, j, v):
        # 如果值 v 不为空
        if v:
            # 如果索引 (i, j) 已经在 smat 中，并且 smat[i, j] 不是 tba 或 v，则抛出冲突异常
            if (i, j) in smat and smat[i, j] not in (tba, v):
                raise ValueError('collision at %s' % ((i, j),))
            # 否则将值 v 更新到 smat[i, j] 中
            smat[i, j] = v

    # 如果存在 undone 列表
    if undone:
        # 遍历 undone 列表中的每对元素 (d, vi)
        for d, vi in undone:
            # 将 d 解析为行号 r 和列号 c
            r, c = rc(d)
            # 如果 vi 是可调用的，则将其作为函数 lambda _ 的结果 v
            v = vi if callable(vi) else lambda _: vi
            i = 0
            # 在 r 和 c 的基础上依次增加 i，直到超过 rows 或 cols
            while r + i < rows and c + i < cols:
                # 调用 update 函数更新 smat[r + i, c + i] 的值为 v(i)
                update(r + i, c + i, v(i))
                i += 1

    # 返回一个 SparseMatrix 对象，其行数为 rows，列数为 cols，稀疏矩阵数据为 smat
    return SparseMatrix(rows, cols, smat)
```