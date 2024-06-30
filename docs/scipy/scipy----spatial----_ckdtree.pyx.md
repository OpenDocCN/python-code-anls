# `D:\src\scipysrc\scipy\scipy\spatial\_ckdtree.pyx`

```
# Copyright Anne M. Archibald 2008
# Additional contributions by Patrick Varilly and Sturla Molden 2012
# Revision by Sturla Molden 2015
# Balanced kd-tree construction written by Jake Vanderplas for scikit-learn
# Released under the scipy license

# cython: cpow=True

# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
import scipy.sparse  # 导入 SciPy 的稀疏矩阵模块
from scipy._lib._util import copy_if_needed  # 导入辅助函数 copy_if_needed

# 导入 C 扩展模块声明
cimport numpy as np

# 导入 CPython 的内存管理函数
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# 导入 C++ 的 vector 容器和 bool 类型
from libcpp.vector cimport vector
from libcpp cimport bool
# 导入 libc.math 中的数学函数和常量
from libc.math cimport isinf, INFINITY

# 导入 Cython 声明
cimport cython
# 导入标准库中的 os、threading 和 operator 模块
import os
import threading
import operator

# 导入 NumPy 的 C 扩展接口
np.import_array()

# 定义从 limits.h 导入的 LONG_MAX 常量
cdef extern from "<limits.h>":
    long LONG_MAX

# 定义 __all__ 列表，指定外部可访问的模块成员
__all__ = ['cKDTree']

# 定义 NPY_LIKELY 和 NPY_UNLIKELY 宏，用于优化条件分支
cdef extern from *:
    int NPY_LIKELY(int)
    int NPY_UNLIKELY(int)


# C++ implementations
# ===================

# 定义 ckdtree 节点结构体的声明
cdef extern from "ckdtree_decl.h":
    struct ckdtreenode:
        np.intp_t split_dim
        np.intp_t children
        np.float64_t split
        np.intp_t start_idx
        np.intp_t end_idx
        ckdtreenode *less
        ckdtreenode *greater
        np.intp_t _less
        np.intp_t _greater

    # 定义 ckdtree 结构体的声明
    struct ckdtree:
        vector[ckdtreenode]  *tree_buffer
        ckdtreenode   *ctree
        np.float64_t   *raw_data
        np.intp_t      n
        np.intp_t      m
        np.intp_t      leafsize
        np.float64_t   *raw_maxes
        np.float64_t   *raw_mins
        np.intp_t      *raw_indices
        np.float64_t   *raw_boxsize_data
        np.intp_t size

    # 外部的构建和查询方法声明
    int build_ckdtree(ckdtree *self,
                         np.intp_t start_idx,
                         np.intp_t end_idx,
                         np.float64_t *maxes,
                         np.float64_t *mins,
                         int _median,
                         int _compact) except + nogil

    int build_weights(ckdtree *self,
                         np.float64_t *node_weights,
                         np.float64_t *weights) except + nogil

    int query_knn(const ckdtree *self,
                     np.float64_t *dd,
                     np.intp_t    *ii,
                     const np.float64_t *xx,
                     const np.intp_t    n,
                     const np.intp_t    *k,
                     const np.intp_t    nk,
                     const np.intp_t    kmax,
                     const np.float64_t eps,
                     const np.float64_t p,
                     const np.float64_t distance_upper_bound) except + nogil

    int query_pairs(const ckdtree *self,
                       const np.float64_t r,
                       const np.float64_t p,
                       const np.float64_t eps,
                       vector[ordered_pair] *results) except + nogil
    # 计算未加权邻居数量的函数
    int count_neighbors_unweighted(const ckdtree *self,
                                   const ckdtree *other,
                                   np.intp_t     n_queries,
                                   np.float64_t  *real_r,
                                   np.intp_t     *results,
                                   const np.float64_t p,
                                   int cumulative) except + nogil
    
    # 计算加权邻居数量的函数
    int count_neighbors_weighted(const ckdtree *self,
                                 const ckdtree *other,
                                 np.float64_t  *self_weights,
                                 np.float64_t  *other_weights,
                                 np.float64_t  *self_node_weights,
                                 np.float64_t  *other_node_weights,
                                 np.intp_t     n_queries,
                                 np.float64_t  *real_r,
                                 np.float64_t  *results,
                                 const np.float64_t p,
                                 int cumulative) except + nogil
    
    # 查询球形范围内点的索引
    int query_ball_point(const ckdtree *self,
                         const np.float64_t *x,
                         const np.float64_t *r,
                         const np.float64_t p,
                         const np.float64_t eps,
                         const np.intp_t n_queries,
                         vector[np.intp_t] *results,
                         const bool return_length,
                         const bool sort_output) except + nogil
    
    # 查询球树内球形范围内点的索引
    int query_ball_tree(const ckdtree *self,
                        const ckdtree *other,
                        const np.float64_t r,
                        const np.float64_t p,
                        const np.float64_t eps,
                        vector[np.intp_t] *results) except + nogil
    
    # 计算稀疏距离矩阵的函数
    int sparse_distance_matrix(const ckdtree *self,
                               const ckdtree *other,
                               const np.float64_t p,
                               const np.float64_t max_distance,
                               vector[coo_entry] *results) except + nogil
# C++ helper functions
# ====================

# 从头文件 "coo_entries.h" 导入结构体 coo_entry
cdef extern from "coo_entries.h":

    struct coo_entry:
        np.intp_t i  # 定义 coo_entry 结构体的成员 i，为 np.intp_t 类型
        np.intp_t j  # 定义 coo_entry 结构体的成员 j，为 np.intp_t 类型
        np.float64_t v  # 定义 coo_entry 结构体的成员 v，为 np.float64_t 类型

# 从头文件 "ordered_pair.h" 导入结构体 ordered_pair
cdef extern from "ordered_pair.h":

    struct ordered_pair:
        np.intp_t i  # 定义 ordered_pair 结构体的成员 i，为 np.intp_t 类型
        np.intp_t j  # 定义 ordered_pair 结构体的成员 j，为 np.intp_t 类型

# coo_entry wrapper
# =================

# 定义 Python 类 coo_entries，用于包装 coo_entry 的操作
cdef class coo_entries:

    cdef:
        readonly object __array_interface__  # 只读对象 __array_interface__，用于与 NumPy 交互
        vector[coo_entry] *buf  # 使用 std::vector 包装的 coo_entry 结构体指针 buf

    # 构造函数 __cinit__
    def __cinit__(coo_entries self):
        self.buf = NULL  # 初始化 buf 为 NULL 指针

    # 初始化方法 __init__
    def __init__(coo_entries self):
        self.buf = new vector[coo_entry]()  # 使用 new 创建 coo_entry 结构体的 vector

    # 析构函数 __dealloc__
    def __dealloc__(coo_entries self):
        if self.buf != NULL:
            del self.buf  # 如果 buf 不为空，则删除 buf

    # 方法 ndarray，返回 NumPy 数组
    def ndarray(coo_entries self):
        cdef:
            coo_entry *pr  # 定义 coo_entry 指针 pr
            np.uintp_t uintptr  # uintptr 类型变量
            np.intp_t n  # 定义 n，为 np.intp_t 类型
        _dtype = [('i',np.intp),('j',np.intp),('v',np.float64)]  # 定义结构体的 dtype
        res_dtype = np.dtype(_dtype, align = True)  # 定义结果的 dtype
        n = <np.intp_t> self.buf.size()  # 获取 buf 中的元素数量
        if NPY_LIKELY(n > 0):  # 如果 n 大于 0
            pr = self.buf.data()  # 获取 buf 的数据指针
            uintptr = <np.uintp_t> (<void*> pr)  # 转换 pr 指针为 uintptr 类型
            dtype = np.dtype(np.uint8)  # 定义 dtype 为 np.uint8
            # 设置 __array_interface__ 属性，用于 NumPy 数组视图
            self.__array_interface__ = dict(
                data = (uintptr, False),  # 数据地址及是否所有者标志
                descr = dtype.descr,  # 数据描述
                shape = (n*sizeof(coo_entry),),  # 数组形状
                strides = (dtype.itemsize,),  # 数组步幅
                typestr = dtype.str,  # 数据类型字符串
                version = 3,  # 接口版本
            )
            return np.asarray(self).view(dtype=res_dtype)  # 返回 NumPy 数组视图
        else:
            return np.empty(shape=(0,), dtype=res_dtype)  # 返回空的 NumPy 数组

    # 方法 dict，返回 Python 字典
    def dict(coo_entries self):
        cdef:
            np.intp_t i, j, k, n  # 定义整数型变量 i, j, k, n
            np.float64_t v  # 定义浮点数变量 v
            coo_entry *pr  # 定义 coo_entry 指针 pr
            dict res_dict  # 定义结果字典 res_dict
        n = <np.intp_t> self.buf.size()  # 获取 buf 中的元素数量
        if NPY_LIKELY(n > 0):  # 如果 n 大于 0
            pr = self.buf.data()  # 获取 buf 的数据指针
            res_dict = dict()  # 初始化结果字典
            for k in range(n):  # 遍历 buf 中的元素
                i = pr[k].i  # 获取当前元素的 i 值
                j = pr[k].j  # 获取当前元素的 j 值
                v = pr[k].v  # 获取当前元素的 v 值
                res_dict[(i,j)] = v  # 将 (i, j) 对及对应的 v 值添加到结果字典中
            return res_dict  # 返回结果字典
        else:
            return {}  # 如果 buf 为空，则返回空字典

    # 方法 coo_matrix，返回 scipy 的 coo_matrix
    def coo_matrix(coo_entries self, m, n):
        res_arr = self.ndarray()  # 调用 ndarray 方法获取 NumPy 数组
        return scipy.sparse.coo_matrix(
                       (res_arr['v'], (res_arr['i'], res_arr['j'])),
                                       shape=(m, n))  # 根据数组创建 coo_matrix

    # 方法 dok_matrix，返回 scipy 的 dok_matrix
    def dok_matrix(coo_entries self, m, n):
        return self.coo_matrix(m,n).todok()  # 将 coo_matrix 转换为 dok_matrix


# ordered_pair wrapper
# ====================

# 定义 Python 类 ordered_pairs，用于包装 ordered_pair 的操作
cdef class ordered_pairs:

    cdef:
        readonly object __array_interface__  # 只读对象 __array_interface__，用于与 NumPy 交互
        vector[ordered_pair] *buf  # 使用 std::vector 包装的 ordered_pair 结构体指针 buf

    # 构造函数 __cinit__
    def __cinit__(ordered_pairs self):
        self.buf = NULL  # 初始化 buf 为 NULL 指针

    # 初始化方法 __init__
    def __init__(ordered_pairs self):
        self.buf = new vector[ordered_pair]()  # 使用 new 创建 ordered_pair 结构体的 vector

    # 析构函数 __dealloc__
    def __dealloc__(ordered_pairs self):
        if self.buf != NULL:
            del self.buf  # 如果 buf 不为空，则删除 buf
    # 定义一个 ndarray 方法，用于将 ordered_pairs 数据转换为 NumPy 数组
    def ndarray(ordered_pairs self):
        cdef:
            ordered_pair *pr  # 声明 ordered_pair 类型的指针 pr
            np.uintp_t uintptr  # 声明一个无符号整数类型 uintptr
            np.intp_t n  # 声明一个整数类型 n，用于存储 buf 的大小
        n = <np.intp_t> self.buf.size()  # 获取 buf 的大小，并将其转换为 np.intp_t 类型
        if NPY_LIKELY(n > 0):  # 如果 n 大于 0，则执行下面的代码块
            pr = self.buf.data()  # 获取 buf 的数据指针，赋值给 pr
            uintptr = <np.uintp_t> (<void*> pr)  # 将 pr 转换为 uintptr 类型
            dtype = np.dtype(np.intp)  # 创建一个 np.intp 类型的数据类型对象 dtype
            # 设置 __array_interface__ 属性为 NumPy 数组接口字典
            self.__array_interface__ = dict(
                data = (uintptr, False),  # 数据地址和是否是只读
                descr = dtype.descr,  # 数据描述符
                shape = (n, 2),  # 数组形状
                strides = (2 * dtype.itemsize, dtype.itemsize),  # 步长信息
                typestr = dtype.str,  # 数据类型字符串
                version = 3,  # 接口版本号
            )
            return np.asarray(self)  # 将 self 转换为 NumPy 数组并返回
        else:
            return np.empty(shape=(0, 2), dtype=np.intp)  # 如果 n <= 0，则返回一个空的 NumPy 数组
    
    # 定义一个 set 方法，用于将 ordered_pairs 数据转换为 set 类型
    def set(ordered_pairs self):
        cdef:
            ordered_pair *pair  # 声明 ordered_pair 类型的指针 pair
            np.intp_t i, n  # 声明整数类型的变量 i 和 n
            set results  # 声明一个 set 类型的变量 results，用于存储结果集合
        results = set()  # 初始化 results 为空集合
        pair = self.buf.data()  # 获取 buf 的数据指针，赋值给 pair
        n = <np.intp_t> self.buf.size()  # 获取 buf 的大小，并将其转换为 np.intp_t 类型
        # 遍历数据，将 ordered_pairs 的元素作为元组 (pair.i, pair.j) 添加到结果集合中
        for i in range(n):
            results.add((pair.i, pair.j))
            pair += 1  # 移动指针到下一个 ordered_pair 元素
        return results  # 返回填充了 ordered_pairs 数据的结果集合
# Tree structure exposed to Python
# ================================

# 定义 cKDTreeNode 类，用于表示 cKDTree 对象中的一个节点
cdef class cKDTreeNode:
    """
    class cKDTreeNode

    This class exposes a Python view of a node in the cKDTree object.

    All attributes are read-only.

    Attributes
    ----------
    level : int
        The depth of the node. 0 is the level of the root node.
    split_dim : int
        The dimension along which this node is split. If this value is -1
        the node is a leafnode in the kd-tree. Leafnodes are not split further
        and scanned by brute force.
    split : float
        The value used to separate split this node. Points with value >= split
        in the split_dim dimension are sorted to the 'greater' subnode
        whereas those with value < split are sorted to the 'lesser' subnode.
    children : int
        The number of data points sorted to this node.
    data_points : ndarray of float64
        An array with the data points sorted to this node.
    indices : ndarray of intp
        An array with the indices of the data points sorted to this node. The
        indices refer to the position in the data set used to construct the
        kd-tree.
    lesser : cKDTreeNode or None
        Subnode with the 'lesser' data points. This attribute is None for
        leafnodes.
    greater : cKDTreeNode or None
        Subnode with the 'greater' data points. This attribute is None for
        leafnodes.

    """
    
    # 定义 cKDTreeNode 类的属性和方法
    cdef:
        # 下面是只读属性
        readonly np.intp_t    level
        readonly np.intp_t    split_dim
        readonly np.intp_t    children
        readonly np.intp_t    start_idx
        readonly np.intp_t    end_idx
        readonly np.float64_t split
        # 可变属性
        np.ndarray            _data
        np.ndarray            _indices
        # 只读对象属性
        readonly object       lesser
        readonly object       greater
    
    # cKDTreeNode 类的初始化方法，用于设置节点的属性
    cdef void _setup(cKDTreeNode self, cKDTree parent, ckdtreenode *node, np.intp_t level) noexcept:
        cdef cKDTreeNode n1, n2
        self.level = level  # 设置节点的深度
        self.split_dim = node.split_dim  # 设置节点的分割维度
        self.children = node.children  # 设置节点包含数据点的数量
        self.split = node.split  # 设置节点的分割值
        self.start_idx = node.start_idx  # 设置节点数据在数据集中的起始索引
        self.end_idx = node.end_idx  # 设置节点数据在数据集中的结束索引
        self._data = parent.data  # 设置节点的数据数组
        self._indices = parent.indices  # 设置节点的索引数组
        
        # 如果当前节点是叶子节点（split_dim == -1），则没有子节点
        if self.split_dim == -1:
            self.lesser = None
            self.greater = None
        else:
            # 设置较小分支
            n1 = cKDTreeNode()
            n1._setup(parent, node=node.less, level=level + 1)
            self.lesser = n1
            # 设置较大分支
            n2 = cKDTreeNode()
            n2._setup(parent, node=node.greater, level=level + 1)
            self.greater = n2
    
    # 数据点属性的访问器，返回当前节点的数据点数组
    property data_points:
        def __get__(cKDTreeNode self):
            return self._data[self.indices,:]
    
    # 索引属性的访问器，返回当前节点的索引数组
    property indices:
        def __get__(cKDTreeNode self):
            cdef np.intp_t start, stop
            start = self.start_idx
            stop = self.end_idx
            return self._indices[start:stop]
cdef np.intp_t get_num_workers(workers: object, kwargs: dict) except -1:
    """Handle the workers argument"""
    # 如果 workers 参数为 None，则设为默认值 1
    if workers is None:
        workers = 1

    # 检查是否有未预期的关键字参数传入，如果有则抛出 TypeError 异常
    if len(kwargs) > 0:
        raise TypeError(
            f"Unexpected keyword argument{'s' if len(kwargs) > 1 else ''} "
            f"{kwargs}")

    # 将 workers 转换为整数类型，使用 operator.index 进行处理
    cdef np.intp_t n = operator.index(workers)

    # 如果 n 等于 -1，则根据系统 CPU 数量设置默认值
    if n == -1:
        num = os.cpu_count()
        if num is None:
            # 如果无法确定 CPU 数量，则抛出 NotImplementedError 异常
            raise NotImplementedError(
                'Cannot determine the number of cpus using os.cpu_count(), '
                'cannot use -1 for the number of workers')
        n = num
    # 如果 n 小于等于 0，则抛出 ValueError 异常，要求 workers 必须为 -1 或大于 0
    elif n <= 0:
        raise ValueError(f'Invalid number of workers {workers}, must be -1 or > 0')
    
    # 返回处理后的有效 workers 数量
    return n


# Main cKDTree class
# ==================

cdef class cKDTree:
    """
    cKDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
            balanced_tree=True, boxsize=None)

    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    .. note::
       `cKDTree` is functionally identical to `KDTree`. Prior to SciPy
       v1.6.0, `cKDTree` had better performance and slightly different
       functionality but now the two names exist only for
       backward-compatibility reasons. If compatibility with SciPy < 1.6 is not
       a concern, prefer `KDTree`.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force. Default: 16.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    """
    # boxsize 是一个数组或标量，可选参数
    # 应用 m-d 坐标环拓扑到 KD 树中。该拓扑由形如 x_i + n_i L_i 的形式生成，
    # 其中 n_i 是整数，L_i 是第 i 维的 boxsize。输入数据将被包装到 [0, L_i) 范围内。
    # 如果数据超出此范围，则引发 ValueError。

    Notes
    -----
    # 使用 Maneewongvatana 和 Mount 1999 年的算法描述的算法。
    # KD 树是一个二叉树，每个节点表示一个轴对齐超矩形。每个节点指定一个轴，
    # 并根据它们在该轴上的坐标值是否大于或小于特定值来分割点集。

    # 在构建过程中，采用“滑动中点”规则选择轴和分割点，以确保单元不会全部变得又长又细。

    # 可以查询树以获取任意给定点的 r 个最近邻居（可选择仅返回某些距离内的邻居）。
    # 还可以使用显著的效率提升查询 r 近似最近邻居。

    # 对于大维度（20 维已经很大），不要指望这比蛮力算法运行得快多少。
    # 高维度最近邻查询是计算机科学中一个重要的未解决问题。

    Attributes
    ----------
    # 数据点的 n 维度 m 的 ndarray，用于索引。
    # 除非需要产生连续的 double 数组，否则这个数组不会被复制。
    # 如果使用 ``copy_data=True`` 构建 KD 树，数据也会被复制。
    data : ndarray, shape (n,m)

    # 算法切换到蛮力方法的点数。
    leafsize : positive int

    # 单个数据点的维度 m。
    m : int

    # 数据点的数量 n。
    n : int

    # 数据点在每个维度上的最大值，形状为 (m,) 的 ndarray。
    maxes : ndarray, shape (m,)

    # 数据点在每个维度上的最小值，形状为 (m,) 的 ndarray。
    mins : ndarray, shape (m,)

    # cKDTree 中根节点的 Python 视图对象，这个属性在首次访问时动态创建。
    tree : object, class cKDTreeNode

    # 树中节点的数量。
    size : int
    # 定义 leafsize 属性的 getter 方法，返回 self.cself.leafsize 的值
    property leafsize:
        def __get__(self): return self.cself.leafsize

    # 定义 size 属性的 getter 方法，返回 self.cself.size 的值
    property size:
        def __get__(self): return self.cself.size

    # 定义 tree 属性的 getter 方法，用于在 Python 中查看树结构
    def __get__(cKDTree self):
        # 声明一些 Cython 的变量和结构体指针
        cdef cKDTreeNode n
        cdef ckdtree *cself = self.cself
        # 如果已经存在 Python 树对象则直接返回
        if self._python_tree is not None:
            return self._python_tree
        else:
            # 否则创建一个新的 cKDTreeNode 对象，设置其属性
            n = cKDTreeNode()
            n._setup(self, node=cself.ctree, level=0)
            self._python_tree = n
            return self._python_tree

    # cKDTree 对象的构造函数，在此处分配内存并初始化
    def __cinit__(cKDTree self):
        self.cself = <ckdtree *> PyMem_Malloc(sizeof(ckdtree))
        self.cself.tree_buffer = NULL
    def __init__(cKDTree self, data, np.intp_t leafsize=16, compact_nodes=True,
            copy_data=False, balanced_tree=True, boxsize=None):
        # 定义 Cython 变量和指针
        cdef:
            np.float64_t [::1] tmpmaxes, tmpmins  # 临时存储最大值和最小值的数组
            np.float64_t *ptmpmaxes  # 最大值数组的指针
            np.float64_t *ptmpmins  # 最小值数组的指针
            ckdtree *cself = self.cself  # Cython 的 cKDTree 实例指针
            int compact, median  # 布尔类型变量 compact 和 median

        self._python_tree = None  # 初始化 Python 树结构为 None

        if not copy_data:
            copy_data = copy_if_needed  # 如果不复制数据，则调用 copy_if_needed 函数
        data = np.array(data, order='C', copy=copy_data, dtype=np.float64)  # 将输入数据转换为 NumPy 数组

        if data.ndim != 2:
            raise ValueError("data must be of shape (n, m), where there are "
                             "n points of dimension m")  # 如果数据维度不是 2，则抛出 ValueError

        if not np.isfinite(data).all():
            raise ValueError("data must be finite, check for nan or inf values")  # 如果数据包含非有限值，则抛出 ValueError

        self.data = data  # 将处理后的数据赋给实例变量 self.data
        cself.n = data.shape[0]  # 将数据行数赋给 Cython 实例变量 cself.n
        cself.m = data.shape[1]  # 将数据列数赋给 Cython 实例变量 cself.m
        cself.leafsize = leafsize  # 将叶子节点大小赋给 Cython 实例变量 cself.leafsize

        if leafsize < 1:
            raise ValueError("leafsize must be at least 1")  # 如果叶子节点大小小于 1，则抛出 ValueError

        if boxsize is None:
            self.boxsize = None  # 如果未指定盒子大小，则将实例变量 self.boxsize 设为 None
            self.boxsize_data = None  # 将实例变量 self.boxsize_data 设为 None
        else:
            self.boxsize_data = np.empty(2 * self.m, dtype=np.float64)  # 创建大小为 2*m 的空数组 self.boxsize_data
            boxsize = broadcast_contiguous(boxsize, shape=(self.m,),
                                           dtype=np.float64)  # 将盒子大小广播成指定形状和数据类型的数组
            self.boxsize_data[:self.m] = boxsize  # 将盒子大小的前半部分赋给 self.boxsize_data
            self.boxsize_data[self.m:] = 0.5 * boxsize  # 将盒子大小的后半部分赋给 self.boxsize_data（每个值乘以 0.5）

            self.boxsize = boxsize  # 将盒子大小赋给实例变量 self.boxsize
            periodic_mask = self.boxsize > 0  # 创建一个布尔掩码，标识盒子大小大于零的部分
            if ((self.data >= self.boxsize[None, :])[:, periodic_mask]).any():
                raise ValueError("Some input data are greater than the size of the periodic box.")
                # 如果输入数据中有大于盒子大小的数据，则抛出 ValueError
            if ((self.data < 0)[:, periodic_mask]).any():
                raise ValueError("Negative input data are outside of the periodic box.")
                # 如果输入数据中有小于零的数据，则抛出 ValueError

        self.maxes = np.ascontiguousarray(
            np.amax(self.data, axis=0) if self.n > 0 else np.zeros(self.m),
            dtype=np.float64)  # 计算数据每列的最大值并转换为连续内存的 NumPy 数组
        self.mins = np.ascontiguousarray(
            np.amin(self.data, axis=0) if self.n > 0 else np.zeros(self.m),
            dtype=np.float64)  # 计算数据每列的最小值并转换为连续内存的 NumPy 数组
        self.indices = np.ascontiguousarray(np.arange(self.n, dtype=np.intp))  # 创建连续内存的索引数组

        self._pre_init()  # 调用预初始化方法

        compact = 1 if compact_nodes else 0  # 根据 compact_nodes 参数设置 compact 变量（1 或 0）
        median = 1 if balanced_tree else 0  # 根据 balanced_tree 参数设置 median 变量（1 或 0）

        cself.tree_buffer = new vector[ckdtreenode]()  # 分配内存给 Cython 实例的树缓冲区

        tmpmaxes = np.copy(self.maxes)  # 复制最大值数组到临时数组 tmpmaxes
        tmpmins = np.copy(self.mins)  # 复制最小值数组到临时数组 tmpmins

        ptmpmaxes = &tmpmaxes[0]  # 获取 tmpmaxes 的指针
        ptmpmins = &tmpmins[0]  # 获取 tmpmins 的指针
        with nogil:
            build_ckdtree(cself, 0, cself.n, ptmpmaxes, ptmpmins, median, compact)
            # 使用 Cython 的 nogil 上下文构建 ckdtree，传入必要的参数

        # 设置树结构指针
        self._post_init()  # 调用初始化后的方法
    cdef _pre_init(cKDTree self):
        cself = self.cself

        # finalize the pointers from array attributes
        # 从数组属性中获取指针并进行最终化处理

        cself.raw_data = <np.float64_t*> np.PyArray_DATA(self.data)
        cself.raw_maxes = <np.float64_t*> np.PyArray_DATA(self.maxes)
        cself.raw_mins = <np.float64_t*> np.PyArray_DATA(self.mins)
        cself.raw_indices = <np.intp_t*> np.PyArray_DATA(self.indices)

        if self.boxsize_data is not None:
            cself.raw_boxsize_data = <np.float64_t*> np.PyArray_DATA(self.boxsize_data)
        else:
            cself.raw_boxsize_data = NULL

    cdef _post_init(cKDTree self):
        cself = self.cself
        # finalize the tree points, this calls _post_init_traverse
        # 最终化树节点，这会调用 _post_init_traverse 方法

        cself.ctree = cself.tree_buffer.data()

        # set the size attribute after tree_buffer is built
        # 在构建完 tree_buffer 后设置 size 属性
        cself.size = cself.tree_buffer.size()

        self._post_init_traverse(cself.ctree)

    cdef _post_init_traverse(cKDTree self, ckdtreenode *node):
        cself = self.cself
        # recurse the tree and re-initialize
        # "less" and "greater" fields
        # 递归遍历树并重新初始化 "less" 和 "greater" 字段

        if node.split_dim == -1:
            # leafnode
            # 叶节点
            node.less = NULL
            node.greater = NULL
        else:
            node.less = cself.ctree + node._less
            node.greater = cself.ctree + node._greater
            self._post_init_traverse(node.less)
            self._post_init_traverse(node.greater)

    def __dealloc__(cKDTree self):
        cself = self.cself
        if cself.tree_buffer != NULL:
            del cself.tree_buffer
        PyMem_Free(cself)

    # -----
    # query
    # -----

    @cython.boundscheck(False)
    # ----------------
    # query_ball_point
    # ----------------

    # ---------------
    # query_ball_tree
    # ---------------

    # -----------
    # query_pairs
    # -----------
    def query_pairs(cKDTree self, np.float64_t r, np.float64_t p=2.,
                    np.float64_t eps=0, output_type='set'):
        """
        query_pairs(self, r, p=2., eps=0, output_type='set')

        Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  ``p`` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'set'

        Returns
        -------
        results : set or ndarray
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close. If output_type is 'ndarray', an ndarry is
            returned instead of a set.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import cKDTree
        >>> rng = np.random.default_rng()
        >>> points = rng.random((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = cKDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """

        # 声明一个变量 results，用于存储查询结果的有序对
        cdef ordered_pairs results

        # 调用 ordered_pairs 类来初始化 results
        results = ordered_pairs()

        # 使用 nogil 上下文，调用 C 函数 query_pairs，将结果存入 results.buf
        with nogil:
            query_pairs(self.cself, r, p, eps, results.buf)

        # 根据 output_type 返回结果，如果是 'set'，返回结果的集合形式；如果是 'ndarray'，返回结果的数组形式
        if output_type == 'set':
            return results.set()
        elif output_type == 'ndarray':
            return results.ndarray()
        else:
            # 如果 output_type 不是 'set' 或 'ndarray'，抛出 ValueError 异常
            raise ValueError("Invalid output type")
    def _build_weights(cKDTree self, object weights):
        """
        _build_weights(weights)

        Compute weights of nodes from weights of data points. This will sum
        up the total weight per node. This function is used internally.

        Parameters
        ----------
        weights : array_like
            weights of data points; must be the same length as the data points.
            currently only scalar weights are supported. Therefore the weights
            array must be 1 dimensional.

        Returns
        -------
        node_weights : array_like
            total weight for each KD-Tree node.

        """
        cdef:
            np.intp_t num_of_nodes  # 定义整数类型变量，存储节点数量
            np.float64_t [::1] node_weights  # 定义一维浮点数数组，存储节点权重
            np.float64_t [::1] proper_weights  # 定义一维浮点数数组，存储合适的权重
            np.float64_t *pnw  # 定义浮点数指针，指向节点权重数组
            np.float64_t *ppw  # 定义浮点数指针，指向合适的权重数组

        num_of_nodes = self.cself.tree_buffer.size();  # 获取KD树缓冲区的大小，即节点数量
        node_weights = np.empty(num_of_nodes, dtype=np.float64)  # 创建一个空的节点权重数组

        # FIXME: use templates to avoid the type conversion
        proper_weights = np.ascontiguousarray(weights, dtype=np.float64)  # 将权重数组转换为连续存储的浮点数数组

        if len(proper_weights) != self.n:
            raise ValueError('Number of weights differ from the number of data points')

        pnw = &node_weights[0]  # 获取节点权重数组的首地址
        ppw = &proper_weights[0]  # 获取合适权重数组的首地址

        with nogil:
            build_weights(self.cself, pnw, ppw)  # 调用Cython函数，计算节点权重

        return node_weights  # 返回节点权重数组

    # ---------------
    # count_neighbors
    # ---------------

    @cython.boundscheck(False)
    # ----------------------
    # sparse_distance_matrix
    # ----------------------

    # ----------------------
    # pickle
    # ----------------------

    def __getstate__(cKDTree self):
        cdef object state  # 定义对象类型变量state
        cdef ckdtree * cself = self.cself  # 定义指向CKDTree结构体的指针cself
        cdef np.intp_t size = cself.tree_buffer.size() * sizeof(ckdtreenode)  # 计算树缓冲区的总大小

        cdef np.ndarray tree = np.asarray(<char[:size]> <char*> cself.tree_buffer.data())  # 将树缓冲区数据转换为numpy数组

        state = (tree.copy(), self.data.copy(), self.n, self.m, self.leafsize,
                      self.maxes, self.mins, self.indices.copy(),
                      self.boxsize, self.boxsize_data)  # 构建状态元组，包括树的拷贝及其他相关数据的拷贝
        return state  # 返回状态元组

    def __setstate__(cKDTree self, state):
        cdef np.ndarray tree  # 定义numpy数组tree
        cdef ckdtree * cself = self.cself  # 定义指向CKDTree结构体的指针cself
        cdef np.ndarray mytree  # 定义numpy数组mytree

        # unpack the state
        (tree, self.data, self.cself.n, self.cself.m, self.cself.leafsize,
            self.maxes, self.mins, self.indices, self.boxsize, self.boxsize_data) = state  # 解包状态元组

        cself.tree_buffer = new vector[ckdtreenode]()  # 创建CKDTree节点的向量缓冲区
        cself.tree_buffer.resize(tree.size // sizeof(ckdtreenode))  # 调整缓冲区大小以容纳树数据

        mytree = np.asarray(<char[:tree.size]> <char*> cself.tree_buffer.data())  # 将树缓冲区数据转换为numpy数组

        # set raw pointers
        self._python_tree = None  # 初始化Python树为None
        self._pre_init()  # 执行预初始化操作

        # copy the tree data
        mytree[:] = tree  # 将解包后的树数据拷贝到树缓冲区

        # set up the tree structure pointers
        self._post_init()  # 执行后初始化操作
cdef _run_threads(_thread_func, np.intp_t n, np.intp_t n_jobs):
    # 限制并发线程数不超过可用任务数和指定的线程数之间的最小值
    n_jobs = min(n, n_jobs)
    
    # 如果需要并行处理
    if n_jobs > 1:
        # 计算每个线程处理的数据范围
        ranges = [(j * n // n_jobs, (j + 1) * n // n_jobs)
                        for j in range(n_jobs)]

        # 创建线程列表，每个线程执行给定的线程函数，并传入对应的数据范围
        threads = [threading.Thread(target=_thread_func,
                   args=(start, end))
                   for start, end in ranges]
        
        # 启动所有线程，并设置为守护线程
        for t in threads:
            t.daemon = True
            t.start()
        
        # 等待所有线程执行完成
        for t in threads:
            t.join()

    # 如果只有一个任务，则直接在当前线程上执行
    else:
        _thread_func(0, n)

cdef np.intp_t num_points(np.ndarray x, np.intp_t pdim) except -1:
    """返回数组 `x` 中的点数

    同时验证最后一个轴是否表示 `pdim` 维空间中单个点的组成部分
    """
    cdef np.intp_t i, n

    # 检查数组维度是否正确，最后一个轴长度应为 `pdim`
    if x.ndim == 0 or x.shape[x.ndim - 1] != pdim:
        raise ValueError("x must consist of vectors of length {} but "
                         "has shape {}".format(pdim, np.shape(x)))
    
    # 初始化点数为 1
    n = 1
    # 计算数组中除了最后一个轴外所有轴上的元素乘积，得到总点数
    for i in range(x.ndim - 1):
        n *= x.shape[i]
    return n

cdef np.ndarray broadcast_contiguous(object x, tuple shape, object dtype):
    """将 `x` 广播到 `shape`，并确保返回的数组是连续的，可能需要复制"""
    # 尝试避免复制操作，如果形状和类型都匹配，则直接返回连续数组视图
    try:
        if x.shape == shape:
            return np.ascontiguousarray(x, dtype)
    except AttributeError:
        pass
    
    # 如果无法避免复制，创建一个新的数组，广播 `x` 到指定的形状，并确保是连续的
    cdef np.ndarray ret = np.empty(shape, dtype)
    ret[...] = x
    return ret
```