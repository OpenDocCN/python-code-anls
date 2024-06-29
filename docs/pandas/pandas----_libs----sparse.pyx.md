# `D:\src\scipysrc\pandas\pandas\_libs\sparse.pyx`

```
# 导入Cython扩展模块
cimport cython

# 导入NumPy库，并命名为np
import numpy as np

# 导入Cython版本的NumPy库，命名为cnp
cimport numpy as cnp

# 从C标准库libc.math中导入常量INFINITY和NAN，重命名为INF和NaN
from libc.math cimport (
    INFINITY as INF,
    NAN as NaN,
)

# 从NumPy的Cython接口中导入特定数据类型和数组类型
from numpy cimport (
    float64_t,
    int8_t,
    int32_t,
    int64_t,
    ndarray,
    uint8_t,
)

# 导入NumPy C API
cnp.import_array()

# -----------------------------------------------------------------------------

# 定义一个Cython类SparseIndex，作为稀疏索引类型的抽象超类
cdef class SparseIndex:
    """
    Abstract superclass for sparse index types.
    """

    # 初始化方法，抛出未实现错误，表明该类是抽象类
    def __init__(self):
        raise NotImplementedError


# 定义一个Cython类IntIndex，继承自SparseIndex，用于保存精确整数稀疏索引信息
cdef class IntIndex(SparseIndex):
    """
    Object for holding exact integer sparse indexing information

    Parameters
    ----------
    length : integer
        长度参数，整数类型
    indices : array-like
        包含整数索引的类数组对象
    check_integrity : bool, default=True
        检查输入的完整性，默认为True
    """

    # 定义只读属性
    cdef readonly:
        Py_ssize_t length, npoints
        ndarray indices

    # 初始化方法，接受length（长度）、indices（索引数组）、check_integrity（是否检查完整性）
    def __init__(self, Py_ssize_t length, indices, bint check_integrity=True):
        # 设置长度属性
        self.length = length
        # 将indices转换为连续的NumPy数组，数据类型为np.int32
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        # 计算索引数组的长度
        self.npoints = len(self.indices)

        # 如果需要检查完整性，则调用check_integrity方法
        if check_integrity:
            self.check_integrity()

    # 序列化方法，用于对象的持久化
    def __reduce__(self):
        args = (self.length, self.indices)
        return IntIndex, args

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        output = "IntIndex\n"
        output += f"Indices: {repr(self.indices)}\n"
        return output

    # 返回索引数组的字节大小
    @property
    def nbytes(self) -> int:
        return self.indices.nbytes

    # 检查完整性方法，确保索引的严格递增、在合理范围内等
    cdef check_integrity(self):
        """
        Checks the following:

        - Indices are strictly ascending
        - Number of indices is at most self.length
        - Indices are at least 0 and at most the total length less one

        A ValueError is raised if any of these conditions is violated.
        """

        # 检查索引数量不超过长度
        if self.npoints > self.length:
            raise ValueError(
                f"Too many indices. Expected {self.length} but found {self.npoints}"
            )

        # 若索引序列为空，则返回
        if self.npoints == 0:
            return

        # 检查索引是否非负
        if self.indices.min() < 0:
            raise ValueError("No index can be less than zero")

        # 检查索引是否在合理范围内
        if self.indices.max() >= self.length:
            raise ValueError("All indices must be less than the length")

        # 检查索引是否严格递增
        monotonic = np.all(self.indices[:-1] < self.indices[1:])
        if not monotonic:
            raise ValueError("Indices must be strictly increasing")

    # 比较方法，判断两个IntIndex对象是否相等
    def equals(self, other: object) -> bool:
        if not isinstance(other, IntIndex):
            return False

        if self is other:
            return True

        same_length = self.length == other.length
        same_indices = np.array_equal(self.indices, other.indices)
        return same_length and same_indices

    # 返回空缺索引的数量
    @property
    def ngaps(self) -> int:
        return self.length - self.npoints

    # 返回自身对象
    cpdef to_int_index(self):
        return self
    # 转换当前对象为块索引对象
    def to_block_index(self):
        # 调用外部函数获取块的位置和长度信息
        locs, lens = get_blocks(self.indices)
        # 返回块索引对象，包括总长度、块的位置和长度信息
        return BlockIndex(self.length, locs, lens)

    # 计算当前稀疏索引对象与另一个稀疏索引对象的交集
    cpdef IntIndex intersect(self, SparseIndex y_):
        cdef:
            Py_ssize_t xi, yi = 0, result_indexer = 0
            int32_t xind
            ndarray[int32_t, ndim=1] xindices, yindices, new_indices
            IntIndex y

        # 将输入的y对象转换为整数索引对象
        y = y_.to_int_index()

        # 检查两个索引对象是否引用相同长度的基础数据
        if self.length != y.length:
            raise Exception("Indices must reference same underlying length")

        # 获取当前对象和y对象的索引数组
        xindices = self.indices
        yindices = y.indices

        # 创建一个新的整数数组来存储交集结果
        new_indices = np.empty(min(len(xindices), len(yindices)), dtype=np.int32)

        # 遍历当前对象的索引数组
        for xi in range(self.npoints):
            xind = xindices[xi]

            # 在y对象的索引数组中找到大于等于当前索引的位置
            while yi < y.npoints and yindices[yi] < xind:
                yi += 1

            # 如果y对象的索引已经遍历完毕，则结束
            if yi >= y.npoints:
                break

            # 如果当前索引与y对象的索引相等，则将其添加到交集结果中
            if yindices[yi] == xind:
                new_indices[result_indexer] = xind
                result_indexer += 1

        # 截取有效部分，并返回一个新的整数索引对象
        new_indices = new_indices[:result_indexer]
        return IntIndex(self.length, new_indices)

    # 计算当前稀疏索引对象与另一个稀疏索引对象的并集
    cpdef IntIndex make_union(self, SparseIndex y_):

        cdef:
            ndarray[int32_t, ndim=1] new_indices
            IntIndex y

        # 将输入的y对象转换为整数索引对象
        y = y_.to_int_index()

        # 检查两个索引对象是否引用相同长度的基础数据
        if self.length != y.length:
            raise ValueError("Indices must reference same underlying length")

        # 使用NumPy的union1d函数计算并集，并返回一个新的整数索引对象
        new_indices = np.union1d(self.indices, y.indices)
        return IntIndex(self.length, new_indices)

    # 查找给定索引位置的内部位置，如果存在则返回位置索引，否则返回-1
    @cython.wraparound(False)
    cpdef int32_t lookup(self, Py_ssize_t index):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            int32_t res
            ndarray[int32_t, ndim=1] inds

        # 获取当前对象的索引数组
        inds = self.indices

        # 如果索引数组为空，则直接返回-1
        if self.npoints == 0:
            return -1
        # 如果索引位置超出范围，则返回-1
        elif index < 0 or self.length <= index:
            return -1

        # 使用二分查找（searchsorted方法）在索引数组中查找给定的index
        res = inds.searchsorted(index)

        # 如果找到的位置超出了索引数组的长度，则返回-1
        if res == self.npoints:
            return -1
        # 如果找到的值与index相等，则返回找到的位置索引
        elif inds[res] == index:
            return res
        # 否则返回-1，表示未找到
        else:
            return -1

    # 设置禁用索引超出范围访问的装饰器
    @cython.wraparound(False)
    # 声明一个Cython函数，返回一个一维整数数组（ndarray[int32_t]）
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        # 声明变量n，表示传入索引数组indexer的长度
        cdef:
            Py_ssize_t n
            # 声明数组inds，用于存储self对象的索引
            ndarray[int32_t, ndim=1] inds
            # 声明数组mask，用于存储布尔值，表示是否在指定范围内的索引
            ndarray[uint8_t, ndim=1, cast=True] mask
            # 声明数组masked，存储符合条件的索引值
            ndarray[int32_t, ndim=1] masked
            # 声明数组res，用于存储搜索结果的索引值
            ndarray[int32_t, ndim=1] res
            # 声明数组results，用于存储最终的结果数组
            ndarray[int32_t, ndim=1] results

        # 获取索引数组indexer的长度
        n = len(indexer)
        # 创建一个dtype为int32的空数组results，长度与indexer相同，初始值为-1
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        # 如果self对象中没有点（npoints为0），直接返回结果数组results
        if self.npoints == 0:
            return results

        # 将self对象的索引数组赋值给inds
        inds = self.indices
        # 创建一个布尔类型的数组mask，表示indexer中的值是否在inds的最小值和最大值之间
        mask = (inds[0] <= indexer) & (indexer <= inds[len(inds) - 1])

        # 从indexer中筛选出符合条件的值，存入数组masked
        masked = indexer[mask]
        # 使用二分搜索在inds中查找masked中每个值的索引，转换为int32类型，存入数组res
        res = inds.searchsorted(masked).astype(np.int32)

        # 将未找到匹配的值（即搜索结果与masked值不匹配的情况）标记为-1
        res[inds[res] != masked] = -1
        # 将搜索结果res复制到results中，只覆盖mask为True的位置
        results[mask] = res
        # 返回最终的结果数组results
        return results
cpdef get_blocks(ndarray[int32_t, ndim=1] indices):
    cdef:
        Py_ssize_t i, npoints, result_indexer = 0  # 定义变量 i, npoints 和 result_indexer，分别表示循环索引、索引数量和结果索引器
        int32_t block, length = 1, cur, prev  # 定义整型变量 block, length, cur 和 prev，分别表示当前块号、长度、当前索引和前一个索引
        ndarray[int32_t, ndim=1] locs, lens  # 定义整型数组 locs 和 lens，用于存储块位置和长度

    npoints = len(indices)  # 获取索引数组的长度

    # 处理特殊的空情况
    if npoints == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)  # 返回空的 numpy 数组，表示没有块

    # 初始化块位置和长度的数组，大小为 npoints
    locs = np.empty(npoints, dtype=np.int32)
    lens = np.empty(npoints, dtype=np.int32)

    # 两次遍历算法是否更快？暂时未实现

    prev = block = indices[0]  # 初始化 prev 和 block 为索引数组的第一个元素
    for i in range(1, npoints):  # 循环处理索引数组中的每个索引
        cur = indices[i]  # 获取当前索引
        if cur - prev > 1:
            # 新块开始
            locs[result_indexer] = block  # 记录块的起始位置
            lens[result_indexer] = length  # 记录块的长度
            block = cur  # 更新当前块号为当前索引
            length = 1  # 重置长度为 1
            result_indexer += 1  # 更新结果索引器
        else:
            # 同一块内，增加长度
            length += 1  # 增加当前块的长度

        prev = cur  # 更新前一个索引为当前索引

    locs[result_indexer] = block  # 记录最后一个块的起始位置
    lens[result_indexer] = length  # 记录最后一个块的长度
    result_indexer += 1  # 更新结果索引器
    locs = locs[:result_indexer]  # 截取有效长度的块位置数组
    lens = lens[:result_indexer]  # 截取有效长度的块长度数组
    return locs, lens  # 返回块位置数组和块长度数组作为结果


# -----------------------------------------------------------------------------
# BlockIndex

cdef class BlockIndex(SparseIndex):
    """
    Object for holding block-based sparse indexing information

    Parameters
    ----------
    """

    cdef readonly:
        int32_t nblocks, npoints, length  # 声明只读整型变量 nblocks, npoints 和 length，分别表示块数、索引点数和总长度
        ndarray blocs, blengths  # 声明只读数组 blocs 和 blengths，分别存储块位置和块长度

    cdef:
        object __weakref__  # 声明 Python 弱引用对象，以便对象可以被序列化

        int32_t *locbuf  # 声明指向整型的指针 locbuf，用于块位置数据
        int32_t *lenbuf  # 声明指向整型的指针 lenbuf，用于块长度数据

    def __init__(self, length, blocs, blengths):
        """
        Initialize BlockIndex object with length, block positions, and block lengths.

        Parameters
        ----------
        length : int
            Total length covered by blocks.
        blocs : ndarray[int32_t, ndim=1]
            Array of block starting positions.
        blengths : ndarray[int32_t, ndim=1]
            Array of block lengths.
        """

        self.blocs = np.ascontiguousarray(blocs, dtype=np.int32)  # 将输入的块位置数组转换为连续的内存布局
        self.blengths = np.ascontiguousarray(blengths, dtype=np.int32)  # 将输入的块长度数组转换为连续的内存布局

        # 将块位置和块长度的数据指针设置为对应数组的数据起始位置
        self.locbuf = <int32_t*>self.blocs.data
        self.lenbuf = <int32_t*>self.blengths.data

        self.length = length  # 初始化对象的总长度
        self.nblocks = np.int32(len(self.blocs))  # 计算并初始化块的数量
        self.npoints = self.blengths.sum()  # 计算并初始化所有块的总长度

        self.check_integrity()  # 调用检查数据完整性的方法

    def __reduce__(self):
        """
        Method used for serialization.

        Returns
        -------
        tuple
            Returns a tuple containing the BlockIndex class and its initialization arguments.
        """
        args = (self.length, self.blocs, self.blengths)
        return BlockIndex, args  # 返回 BlockIndex 类及其初始化参数的元组，用于序列化对象

    def __repr__(self) -> str:
        """
        Return a string representation of the BlockIndex object.

        Returns
        -------
        str
            String representation of BlockIndex object.
        """
        output = "BlockIndex\n"
        output += f"Block locations: {repr(self.blocs)}\n"  # 添加块位置信息的字符串表示
        output += f"Block lengths: {repr(self.blengths)}"  # 添加块长度信息的字符串表示

        return output  # 返回完整的对象字符串表示

    @property
    def nbytes(self) -> int:
        """
        Property method to calculate the total memory usage in bytes.

        Returns
        -------
        int
            Total memory usage in bytes for the BlockIndex object.
        """
        return self.blocs.nbytes + self.blengths.nbytes  # 返回块位置和块长度数组的内存占用总和

    @property
    def ngaps(self) -> int:
        """
        Property method to calculate the number of gaps (gaps between blocks).

        Returns
        -------
        int
            Number of gaps between blocks.
        """
        return self.length - self.npoints  # 返回块之间的间隙数目
    # 定义一个Cython函数来检查块索引的完整性
    cdef check_integrity(self):
        """
        Check:
        - Locations are in ascending order
        - No overlapping blocks
        - Blocks do not start after the end of the index, nor extend beyond the end
        """
        # 声明变量和类型
        cdef:
            Py_ssize_t i  # 用于循环的索引变量
            ndarray[int32_t, ndim=1] blocs, blengths  # 块的起始位置和长度数组

        # 将实例变量赋值给本地变量
        blocs = self.blocs
        blengths = self.blengths

        # 检查块的数量和块长度数组的长度是否相同
        if len(blocs) != len(blengths):
            raise ValueError("block bound arrays must be same length")

        # 遍历每个块进行完整性检查
        for i in range(self.nblocks):
            # 检查位置是否按升序排列
            if i > 0:
                if blocs[i] <= blocs[i - 1]:
                    raise ValueError("Locations not in ascending order")

            # 检查块是否有重叠
            if i < self.nblocks - 1:
                if blocs[i] + blengths[i] > blocs[i + 1]:
                    raise ValueError(f"Block {i} overlaps")
            else:
                # 检查块是否超出索引的结束位置
                if blocs[i] + blengths[i] > self.length:
                    raise ValueError(f"Block {i} extends beyond end")

            # 检查是否存在长度为零的块
            if blengths[i] == 0:
                raise ValueError(f"Zero-length block {i}")

    # 比较当前对象与另一个对象是否相等
    def equals(self, other: object) -> bool:
        if not isinstance(other, BlockIndex):
            return False

        if self is other:
            return True

        # 比较长度和块数组是否相等
        same_length = self.length == other.length
        same_blocks = (np.array_equal(self.blocs, other.blocs) and
                       np.array_equal(self.blengths, other.blengths))
        return same_length and same_blocks

    # 返回当前对象自身，用于转换为块索引
    def to_block_index(self):
        return self

    # 转换为整数索引对象
    cpdef to_int_index(self):
        cdef:
            int32_t i = 0, j, b  # 循环中使用的整数变量
            int32_t offset  # 块的偏移量
            ndarray[int32_t, ndim=1] indices  # 索引数组

        # 创建一个空的整数索引数组
        indices = np.empty(self.npoints, dtype=np.int32)

        # 根据块的位置和长度填充整数索引数组
        for b in range(self.nblocks):
            offset = self.locbuf[b]

            for j in range(self.lenbuf[b]):
                indices[i] = offset + j
                i += 1

        # 返回创建的整数索引对象
        return IntIndex(self.length, indices)

    # 返回整数索引对象的indices属性
    @property
    def indices(self):
        return self.to_int_index().indices
    # 定义一个 cpdef 方法，用于计算两个 SparseIndex 对象的交集
    cpdef BlockIndex intersect(self, SparseIndex other):
        """
        Intersect two BlockIndex objects

        Returns
        -------
        BlockIndex
        """
        # 声明变量
        cdef:
            BlockIndex y  # 另一个 SparseIndex 对象
            ndarray[int32_t, ndim=1] xloc, xlen, yloc, ylen, out_bloc, out_blen  # 各种数组变量
            Py_ssize_t xi = 0, yi = 0, max_len, result_indexer = 0  # 下标和索引变量
            int32_t cur_loc, cur_length, diff  # 当前位置、长度和差异

        # 将另一个 SparseIndex 对象转换为 BlockIndex 对象
        y = other.to_block_index()

        # 检查两个索引对象的长度是否相等，若不相等则抛出异常
        if self.length != y.length:
            raise Exception("Indices must reference same underlying length")

        # 获取当前对象的块位置和块长度
        xloc = self.blocs
        xlen = self.blengths
        yloc = y.blocs
        ylen = y.blengths

        # 计算输出数组的最大长度限制
        max_len = min(self.length, y.length) // 2 + 1
        out_bloc = np.empty(max_len, dtype=np.int32)  # 输出块位置数组
        out_blen = np.empty(max_len, dtype=np.int32)  # 输出块长度数组

        # 开始迭代处理两个索引的块
        while True:
            # 如果其中一个索引对象的块已经遍历完，则退出循环
            if xi >= self.nblocks or yi >= y.nblocks:
                break

            # 对称地处理两个索引对象的块
            if xloc[xi] >= yloc[yi]:
                cur_loc = xloc[xi]
                diff = xloc[xi] - yloc[yi]

                if ylen[yi] <= diff:
                    # 如果 y 的块长度不足以覆盖当前块的差异，则跳过该块
                    yi += 1
                    continue

                if ylen[yi] - diff < xlen[xi]:
                    # 取 y 块的末尾部分，然后移动到下一个块
                    cur_length = ylen[yi] - diff
                    yi += 1
                else:
                    # 取 x 块的末尾部分
                    cur_length = xlen[xi]
                    xi += 1

            else:  # xloc[xi] < yloc[yi]
                cur_loc = yloc[yi]
                diff = yloc[yi] - xloc[xi]

                if xlen[xi] <= diff:
                    # 如果 x 的块长度不足以覆盖当前块的差异，则跳过该块
                    xi += 1
                    continue

                if xlen[xi] - diff < ylen[yi]:
                    # 取 x 块的末尾部分，然后移动到下一个块
                    cur_length = xlen[xi] - diff
                    xi += 1
                else:
                    # 取 y 块的末尾部分
                    cur_length = ylen[yi]
                    yi += 1

            # 将当前块的位置和长度保存到输出数组中
            out_bloc[result_indexer] = cur_loc
            out_blen[result_indexer] = cur_length
            result_indexer += 1

        # 截取实际使用的输出数组部分
        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        # 返回一个新的 BlockIndex 对象，表示两个索引对象的交集
        return BlockIndex(self.length, out_bloc, out_blen)
    # 定义一个 Cython 方法，用于将两个 BlockIndex 对象合并，生成一个新的 BlockIndex 对象
    cpdef BlockIndex make_union(self, SparseIndex y):
        """
        Combine together two BlockIndex objects, accepting indices if contained
        in one or the other

        Parameters
        ----------
        other : SparseIndex

        Notes
        -----
        union is a protected keyword in Cython, hence make_union

        Returns
        -------
        BlockIndex
        """
        # 调用 BlockUnion 类的实例化对象，传入当前对象 self 和 y 转换为 BlockIndex 后的对象，获取合并结果
        return BlockUnion(self, y.to_block_index()).result

    # 定义一个 Cython 方法，用于查找给定索引在 BlockIndex 对象中的位置
    cpdef Py_ssize_t lookup(self, Py_ssize_t index):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            Py_ssize_t i, cum_len
            ndarray[int32_t, ndim=1] locs, lens
        
        # 将 self 对象中的 blocs 和 blengths 赋值给 locs 和 lens
        locs = self.blocs
        lens = self.blengths

        # 若当前 BlockIndex 对象没有块（nblocks == 0），直接返回 -1
        if self.nblocks == 0:
            return -1
        # 若给定的 index 小于第一个块的起始位置 locs[0]，也返回 -1
        elif index < locs[0]:
            return -1

        # 初始化累计长度 cum_len
        cum_len = 0
        # 遍历所有块
        for i in range(self.nblocks):
            # 如果 index 在当前块的范围内，返回其在整个 BlockIndex 中的实际位置
            if index >= locs[i] and index < locs[i] + lens[i]:
                return cum_len + index - locs[i]
            # 更新累计长度
            cum_len += lens[i]

        # 如果 index 不在任何块的范围内，返回 -1
        return -1

    # 定义一个 Cython 方法，实现向量化的索引查找，返回一个 ndarray[int32_t]
    @cython.wraparound(False)
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        cdef:
            Py_ssize_t n, i, j, ind_val
            ndarray[int32_t, ndim=1] locs, lens
            ndarray[int32_t, ndim=1] results

        # 将 self 对象中的 blocs 和 blengths 赋值给 locs 和 lens
        locs = self.blocs
        lens = self.blengths

        # 获取 indexer 数组的长度 n
        n = len(indexer)
        # 创建一个与 indexer 大小相同的结果数组 results，初始值为 -1
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        # 如果当前 BlockIndex 对象中没有点（npoints == 0），直接返回结果数组
        if self.npoints == 0:
            return results

        # 遍历 indexer 数组
        for i in range(n):
            ind_val = indexer[i]
            # 如果 ind_val 小于 0 或者大于等于 self.length，直接跳过
            if not (ind_val < 0 or self.length <= ind_val):
                cum_len = 0
                # 遍历所有块
                for j in range(self.nblocks):
                    # 如果 ind_val 在当前块的范围内，更新 results[i] 的值为其在整个 BlockIndex 中的实际位置
                    if ind_val >= locs[j] and ind_val < locs[j] + lens[j]:
                        results[i] = cum_len + ind_val - locs[j]
                    # 更新累计长度
                    cum_len += lens[j]
        # 返回结果数组
        return results
# 定义一个 Cython 内部类 BlockMerge
@cython.internal
cdef class BlockMerge:
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """
    # 声明类变量和成员变量
    cdef:
        BlockIndex x, y, result  # 两个块索引对象和结果
        ndarray xstart, xlen, xend, ystart, ylen, yend  # 多个 ndarray 对象
        int32_t xi, yi  # 整型变量，块索引

    # 构造函数，初始化 BlockMerge 类
    def __init__(self, BlockIndex x, BlockIndex y):
        self.x = x  # 设置 x 属性为传入的 x 对象
        self.y = y  # 设置 y 属性为传入的 y 对象

        # 检查 x 和 y 对象的长度是否相同
        if x.length != y.length:
            raise Exception("Indices must reference same underlying length")

        # 设置起始和结束位置的 ndarray 对象
        self.xstart = self.x.blocs
        self.ystart = self.y.blocs
        self.xend = self.x.blocs + self.x.blengths
        self.yend = self.y.blocs + self.y.blengths

        # 初始化块索引
        self.xi = 0
        self.yi = 0

        # 调用内部函数 _make_merged_blocks() 来生成合并后的块
        self.result = self._make_merged_blocks()

    # 内部方法，用于生成合并后的块，但此处只有声明，具体实现未给出
    cdef _make_merged_blocks(self):
        raise NotImplementedError

    # 内部方法，用于设置当前的块索引
    cdef _set_current_indices(self, int32_t xi, int32_t yi, bint mode):
        if mode == 0:
            self.xi = xi
            self.yi = yi
        else:
            self.xi = yi
            self.yi = xi


# 定义一个 Cython 内部类 BlockUnion，继承自 BlockMerge 类
@cython.internal
cdef class BlockUnion(BlockMerge):
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """
    # 定义一个Cython函数，用于生成合并的块
    cdef _make_merged_blocks(self):
        # 定义变量和数组，指定它们的数据类型和维度
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart
            ndarray[int32_t, ndim=1] yend, out_bloc, out_blen
            int32_t nstart, nend
            Py_ssize_t max_len, result_indexer = 0

        # 将对象的属性赋值给本地变量
        xstart = self.xstart
        xend = self.xend
        ystart = self.ystart
        yend = self.yend

        # 计算最大长度，使用self.x和self.y的长度的一半加一
        max_len = min(self.x.length, self.y.length) // 2 + 1
        # 创建指定长度的空数组，用于存储输出的块索引和长度
        out_bloc = np.empty(max_len, dtype=np.int32)
        out_blen = np.empty(max_len, dtype=np.int32)

        # 进入主循环，直到处理完所有块或达到结束条件
        while True:
            # 如果self.xi和self.yi超出了各自块的数量，退出循环
            if self.xi >= self.x.nblocks and self.yi >= self.y.nblocks:
                break
            elif self.yi >= self.y.nblocks:
                # 如果self.yi超出了y的块数量，只处理x的块
                nstart = xstart[self.xi]
                nend = xend[self.xi]
                self.xi += 1
            elif self.xi >= self.x.nblocks:
                # 如果self.xi超出了x的块数量，只处理y的块
                nstart = ystart[self.yi]
                nend = yend[self.yi]
                self.yi += 1
            else:
                # 否则，根据块的起始位置找到新块的结束位置
                if xstart[self.xi] < ystart[self.yi]:
                    nstart = xstart[self.xi]
                    nend = self._find_next_block_end(0)
                else:
                    nstart = ystart[self.yi]
                    nend = self._find_next_block_end(1)

            # 将计算得到的块起始位置和长度存入输出数组中
            out_bloc[result_indexer] = nstart
            out_blen[result_indexer] = nend - nstart
            result_indexer += 1

        # 截取实际使用的部分，生成最终的块索引对象并返回
        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        return BlockIndex(self.x.length, out_bloc, out_blen)
    # 定义一个Cython函数，用于查找下一个数据块的结束位置，根据不同的模式来执行不同的逻辑
    cdef int32_t _find_next_block_end(self, bint mode) except -1:
        """
        Wow, this got complicated in a hurry

        mode 0: block started in index x
        mode 1: block started in index y
        """
        # 声明变量以存储各种数组和索引
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart, yend
            int32_t xi, yi, ynblocks, nend

        # 检查模式是否为有效值（0或1）
        if mode != 0 and mode != 1:
            raise Exception("Mode must be 0 or 1")

        # 根据模式选择对称的代码段
        if mode == 0:
            xstart = self.xstart
            xend = self.xend
            xi = self.xi

            ystart = self.ystart
            yend = self.yend
            yi = self.yi
            ynblocks = self.y.nblocks
        else:
            xstart = self.ystart
            xend = self.yend
            xi = self.yi

            ystart = self.xstart
            yend = self.xend
            yi = self.xi
            ynblocks = self.x.nblocks

        # 获取当前块的结束位置
        nend = xend[xi]

        # 如果当前块已经完成，则设置下一块的索引并返回当前块的结束位置
        if yi == ynblocks:
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        # 如果当前块的结束位置在下一个块的开始位置之前，则设置下一块的索引并返回当前块的结束位置
        elif nend < ystart[yi]:
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        else:
            # 否则，通过循环找到合适的块结束位置
            while yi < ynblocks and nend > yend[yi]:
                yi += 1

            # 设置下一块的索引
            self._set_current_indices(xi + 1, yi, mode)

            # 如果已经到达最后一个块，则返回当前块的结束位置
            if yi == ynblocks:
                return nend

            # 如果当前块的结束位置在下一个块的开始位置之前，则返回当前块的结束位置
            if nend < ystart[yi]:
                return nend
            else:
                # 合并块并继续搜索
                # 这也会处理块之间重叠的情况，并继续查找下一个块的结束位置
                return self._find_next_block_end(1 - mode)
# -----------------------------------------------------------------------------
# Sparse arithmetic

# 引入稀疏操作辅助文件
include "sparse_op_helper.pxi"

# -----------------------------------------------------------------------------
# SparseArray mask create operations

# 定义函数：根据对象数组创建掩码对象
def make_mask_object_ndarray(ndarray[object, ndim=1] arr, object fill_value):
    # 声明变量和数组长度
    cdef:
        object value
        Py_ssize_t i
        Py_ssize_t new_length = len(arr)
        ndarray[int8_t, ndim=1] mask  # 定义掩码数组，存储int8类型的数据

    # 初始化掩码数组，全部置为1
    mask = np.ones(new_length, dtype=np.int8)

    # 遍历对象数组
    for i in range(new_length):
        value = arr[i]
        # 检查当前值是否等于指定的填充值，并且类型与填充值相同
        if value == fill_value and type(value) is type(fill_value):
            # 如果是，则将掩码数组对应位置置为0
            mask[i] = 0

    # 返回掩码数组的视图，转换为布尔型
    return mask.view(dtype=bool)
```