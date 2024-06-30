# `D:\src\scipysrc\sympy\sympy\matrices\expressions\slice.py`

```
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.functions.elementary.integers import floor

def normalize(i, parentsize):
    # 如果 i 是 slice 对象，则转换成元组
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    # 如果 i 不是元组、列表或 Tuple，则视为单个索引，根据 parentsize 进行调整
    if not isinstance(i, (tuple, list, Tuple)):
        # 如果索引 i 小于 0，则转换为正数索引
        if (i < 0) == True:
            i += parentsize
        i = (i, i+1, 1)
    i = list(i)
    # 如果元组长度为 2，则添加默认步长 1
    if len(i) == 2:
        i.append(1)
    start, stop, step = i
    start = start or 0
    # 如果 stop 是 None，则设置为 parentsize
    if stop is None:
        stop = parentsize
    # 如果 start 小于 0，则转换为正数索引
    if (start < 0) == True:
        start += parentsize
    # 如果 stop 小于 0，则转换为正数索引
    if (stop < 0) == True:
        stop += parentsize
    step = step or 1

    # 检查索引范围是否有效，否则抛出 IndexError 异常
    if ((stop - start) * step < 1) == True:
        raise IndexError()

    return (start, stop, step)

class MatrixSlice(MatrixExpr):
    """ A MatrixSlice of a Matrix Expression

    Examples
    ========

    >>> from sympy import MatrixSlice, ImmutableMatrix
    >>> M = ImmutableMatrix(4, 4, range(16))
    >>> M
    Matrix([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]])

    >>> B = MatrixSlice(M, (0, 2), (2, 4))
    >>> ImmutableMatrix(B)
    Matrix([
    [2, 3],
    [6, 7]])
    """
    # 父矩阵属性
    parent = property(lambda self: self.args[0])
    # 行切片属性
    rowslice = property(lambda self: self.args[1])
    # 列切片属性
    colslice = property(lambda self: self.args[2])

    def __new__(cls, parent, rowslice, colslice):
        # 标准化行和列切片
        rowslice = normalize(rowslice, parent.shape[0])
        colslice = normalize(colslice, parent.shape[1])
        # 如果切片不是长度为 3 的元组，则抛出 IndexError 异常
        if not (len(rowslice) == len(colslice) == 3):
            raise IndexError()
        # 检查切片是否在有效范围内，否则抛出 IndexError 异常
        if ((0 > rowslice[0]) == True or
            (parent.shape[0] < rowslice[1]) == True or
            (0 > colslice[0]) == True or
            (parent.shape[1] < colslice[1]) == True):
            raise IndexError()
        # 如果 parent 是 MatrixSlice 类型，则返回其切片
        if isinstance(parent, MatrixSlice):
            return mat_slice_of_slice(parent, rowslice, colslice)
        # 否则创建新的 MatrixSlice 对象
        return Basic.__new__(cls, parent, Tuple(*rowslice), Tuple(*colslice))

    @property
    def shape(self):
        # 计算切片后的矩阵形状
        rows = self.rowslice[1] - self.rowslice[0]
        rows = rows if self.rowslice[2] == 1 else floor(rows/self.rowslice[2])
        cols = self.colslice[1] - self.colslice[0]
        cols = cols if self.colslice[2] == 1 else floor(cols/self.colslice[2])
        return rows, cols

    def _entry(self, i, j, **kwargs):
        # 获取指定位置的元素
        return self.parent._entry(i*self.rowslice[2] + self.rowslice[0],
                                  j*self.colslice[2] + self.colslice[0],
                                  **kwargs)

    @property
    def on_diag(self):
        # 检查切片是否为对角线切片
        return self.rowslice == self.colslice


def slice_of_slice(s, t):
    # 计算两个切片的组合切片
    start1, stop1, step1 = s
    start2, stop2, step2 = t

    start = start1 + start2*step1
    step = step1 * step2
    stop = start1 + step1*stop2

    # 如果计算的停止位置超过范围，则抛出 IndexError 异常
    if stop > stop1:
        raise IndexError()

    return start, stop, step


def mat_slice_of_slice(parent, rowslice, colslice):
    # 返回父矩阵的组合切片
    """ Collapse nested matrix slices

    >>> from sympy import MatrixSymbol  # 导入 MatrixSymbol 类
    >>> X = MatrixSymbol('X', 10, 10)  # 创建一个 10x10 的 MatrixSymbol 对象 X
    >>> X[:, 1:5][5:8, :]  # 取 X 的列 1 到 5，然后再取行 5 到 8 的部分，返回 X[5:8, 1:5]
    X[5:8, 1:5]
    >>> X[1:9:2, 2:6][1:3, 2]  # 取 X 的行 1 到 9 步长为 2，列 2 到 6 的部分，再取行 1 到 3，列 2 的部分，返回 X[3:7:2, 4:5]
    X[3:7:2, 4:5]
    """
    row = slice_of_slice(parent.rowslice, rowslice)  # 使用 slice_of_slice 函数处理父对象的行切片和当前的行切片
    col = slice_of_slice(parent.colslice, colslice)  # 使用 slice_of_slice 函数处理父对象的列切片和当前的列切片
    return MatrixSlice(parent.parent, row, col)  # 返回一个新的 MatrixSlice 对象，包含父对象的信息和处理后的行列切片
```