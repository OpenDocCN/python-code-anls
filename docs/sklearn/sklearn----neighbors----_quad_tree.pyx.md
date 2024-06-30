# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_quad_tree.pyx`

```
# 导入所需的 CPython C 扩展模块
from cpython cimport Py_INCREF, PyObject, PyTypeObject

# 导入 libc.math 中的 fabsf 函数
from libc.math cimport fabsf
# 导入 libc.stdlib 中的 free 函数
from libc.stdlib cimport free
# 导入 libc.string 中的 memcpy 函数
from libc.string cimport memcpy
# 导入 libc.stdio 中的 printf 函数
from libc.stdio cimport printf
# 导入 libc.stdint 中的 SIZE_MAX 常量
from libc.stdint cimport SIZE_MAX

# 导入 ..tree._utils 中的 safe_realloc 函数
from ..tree._utils cimport safe_realloc

# 导入 NumPy 库并使用 cnp 别名，同时导入其 C 扩展部分
import numpy as np
cimport numpy as cnp
cnp.import_array()

# 从 numpy/arrayobject.h 中导入 PyArray_NewFromDescr 和 PyArray_SetBaseObject 函数
cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

# 创建 Cell 类型的 dummy 对象，并从中提取其 dtype 以构建 CELL_DTYPE
cdef Cell dummy
CELL_DTYPE = np.asarray(<Cell[:1]>(&dummy)).dtype

# 断言 CELL_DTYPE 的字节大小与 Cell 类型的大小相同
assert CELL_DTYPE.itemsize == sizeof(Cell)


# 定义 _QuadTree 类，表示基于数组的四叉树结构
cdef class _QuadTree:
    """Array-based representation of a QuadTree.

    This class is currently working for indexing 2D data (regular QuadTree) and
    for indexing 3D data (OcTree). It is planned to split the 2 implementations
    using `Cython.Tempita` to save some memory for QuadTree.

    Note that this code is currently internally used only by the Barnes-Hut
    method in `sklearn.manifold.TSNE`. It is planned to be refactored and
    generalized in the future to be compatible with nearest neighbors API of
    `sklearn.neighbors` with 2D and 3D data.
    """
    
    def __cinit__(self, int n_dimensions, int verbose):
        """Constructor."""
        # 初始化树的参数
        self.n_dimensions = n_dimensions  # 维度数
        self.verbose = verbose  # 是否输出详细信息
        self.n_cells_per_cell = <int> (2 ** self.n_dimensions)  # 每个单元包含的子单元数

        # 初始化内部结构
        self.max_depth = 0  # 树的最大深度
        self.cell_count = 0  # 单元格数量
        self.capacity = 0  # 容量
        self.n_points = 0  # 点的数量
        self.cells = NULL  # 指向单元格数组的指针，初始为空

    def __dealloc__(self):
        """Destructor."""
        # 释放所有内部结构占用的内存
        free(self.cells)

    @property
    def cumulative_size(self):
        """Property returning cumulative sizes of cells."""
        # 获取单元格数组的内存视图并返回累积大小属性
        cdef Cell[:] cell_mem_view = self._get_cell_ndarray()
        return cell_mem_view.base['cumulative_size'][:self.cell_count]

    @property
    def leafs(self):
        """Property returning leaf flags of cells."""
        # 获取单元格数组的内存视图并返回叶子标志属性
        cdef Cell[:] cell_mem_view = self._get_cell_ndarray()
        return cell_mem_view.base['is_leaf'][:self.cell_count]
    def build_tree(self, X):
        """Build a tree from an array of points X."""
        cdef:
            int i  # 定义整数变量 i，用于循环索引
            float32_t[3] pt  # 定义长度为 3 的浮点数数组 pt，存储点的坐标
            float32_t[3] min_bounds, max_bounds  # 定义长度为 3 的浮点数数组 min_bounds 和 max_bounds，存储最小和最大边界

        # validate X and prepare for query
        # X = check_array(X, dtype=float32_t, order='C')
        n_samples = X.shape[0]  # 获取样本数

        capacity = 100  # 设置容量为 100
        self._resize(capacity)  # 调整树的初始大小

        m = np.min(X, axis=0)  # 计算 X 中每个维度的最小值
        M = np.max(X, axis=0)  # 计算 X 中每个维度的最大值

        # Scale the maximum to get all points strictly in the tree bounding box
        # The 3 bounds are for positive, negative and small values
        M = np.maximum(M * (1. + 1e-3 * np.sign(M)), M + 1e-3)  # 扩展最大边界以确保所有点都在树的边界框内

        for i in range(self.n_dimensions):
            min_bounds[i] = m[i]  # 设置最小边界
            max_bounds[i] = M[i]  # 设置最大边界

            if self.verbose > 10:
                printf("[QuadTree] bounding box axis %i : [%f, %f]\n",
                       i, min_bounds[i], max_bounds[i])  # 打印每个维度的边界信息

        # Create the initial node with boundaries from the dataset
        self._init_root(min_bounds, max_bounds)  # 使用数据集的边界创建初始节点

        for i in range(n_samples):
            for j in range(self.n_dimensions):
                pt[j] = X[i, j]  # 将点的坐标复制到 pt 数组中
            self.insert_point(pt, i)  # 将点插入到四叉树中

        # Shrink the cells array to reduce memory usage
        self._resize(capacity=self.cell_count)  # 缩小单元格数组以减少内存使用
    # 定义一个Cython函数，在QuadTree中插入一个点
    cdef int insert_point(self, float32_t[3] point, intp_t point_index,
                          intp_t cell_id=0) except -1 nogil:
        """Insert a point in the QuadTree."""
        # 声明局部变量
        cdef int ax
        cdef intp_t selected_child
        # 获取指向给定cell_id的指针
        cdef Cell* cell = &self.cells[cell_id]
        # 获取当前cell中的点的数量
        cdef intp_t n_point = cell.cumulative_size

        # 如果设置了详细输出模式，则打印当前操作的深度信息
        if self.verbose > 10:
            printf("[QuadTree] Inserting depth %li\n", cell.depth)

        # 如果DEBUGFLAG开启，则验证点是否在正确的范围内
        if DEBUGFLAG:
            self._check_point_in_cell(point, cell)

        # 如果cell是一个空叶子节点，则将点插入其中
        if cell.cumulative_size == 0:
            cell.cumulative_size = 1
            self.n_points += 1
            # 将点的坐标复制到cell的重心位置
            for i in range(self.n_dimensions):
                cell.barycenter[i] = point[i]
            cell.point_index = point_index
            if self.verbose > 10:
                printf("[QuadTree] inserted point %li in cell %li\n",
                       point_index, cell_id)
            return cell_id

        # 如果cell不是叶子节点，则更新cell内部信息，并在选择的子节点中递归插入点
        if not cell.is_leaf:
            for ax in range(self.n_dimensions):
                # 使用加权平均更新重心位置
                cell.barycenter[ax] = (
                    n_point * cell.barycenter[ax] + point[ax]) / (n_point + 1)

            # 增加从该cell开始的子树的大小
            cell.cumulative_size += 1

            # 在正确的子树中插入点
            selected_child = self._select_child(point, cell)
            if self.verbose > 49:
                printf("[QuadTree] selected child %li\n", selected_child)
            # 如果无法选择有效的子节点，则创建一个新的子节点并插入点
            if selected_child == -1:
                self.n_points += 1
                return self._insert_point_in_new_child(point, cell, point_index)
            # 递归调用插入函数，将点插入选定的子节点中
            return self.insert_point(point, point_index, selected_child)

        # 最后，如果cell是一个叶子节点且已经插入了一个点，则根据点是否重复决定是分裂cell还是增加叶子的大小并返回
        if self._is_duplicate(point, cell.barycenter):
            if self.verbose > 10:
                printf("[QuadTree] found a duplicate!\n")
            cell.cumulative_size += 1
            self.n_points += 1
            return cell_id

        # 在叶子节点中，重心位置对应于唯一包含的点
        # 将点插入一个新的子节点中
        self._insert_point_in_new_child(cell.barycenter, cell, cell.point_index,
                                        cell.cumulative_size)
        # 递归调用插入函数，将点插入当前cell中
        return self.insert_point(point, point_index, cell_id)

    # XXX: This operation is not Thread safe
    # 定义一个Cython函数，在新的子节点中插入一个点
    cdef intp_t _insert_point_in_new_child(
        self, float32_t[3] point, Cell* cell, intp_t point_index, intp_t size=1
    ) noexcept nogil:
        """Create a child of cell which will contain point."""

        # Local variable definition
        cdef:
            intp_t cell_id, cell_child_id, parent_id              # Define integer variables for cell IDs
            float32_t[3] save_point                               # Define an array to save the current point
            float32_t width                                       # Define a float variable for width
            Cell* child                                           # Declare a pointer to Cell structure
            int i                                                 # Loop variable

        # If the maximal capacity of the Tree have been reached, double the capacity
        # We need to save the current cell id and the current point to retrieve them
        # in case the reallocation
        if self.cell_count + 1 > self.capacity:
            parent_id = cell.cell_id                              # Save current cell's ID
            for i in range(self.n_dimensions):
                save_point[i] = point[i]                          # Save current point coordinates
            self._resize(SIZE_MAX)                                # Resize the data structure
            cell = &self.cells[parent_id]                         # Update cell pointer after potential reallocation
            point = save_point                                    # Restore saved point coordinates

        # Get an empty cell and initialize it
        cell_id = self.cell_count                                 # Get the ID for the new cell
        self.cell_count += 1                                      # Increment cell count
        child = &self.cells[cell_id]                              # Point child to the new cell

        self._init_cell(child, cell.cell_id, cell.depth + 1)       # Initialize the child cell
        child.cell_id = cell_id                                   # Set child cell's ID

        # Set the cell as an inner cell of the Tree
        cell.is_leaf = False                                      # Mark current cell as non-leaf
        cell.point_index = -1                                     # Reset point index of current cell

        # Set the correct boundary for the cell, store the point in the cell
        # and compute its index in the children array.
        cell_child_id = 0                                         # Initialize child index
        for i in range(self.n_dimensions):
            cell_child_id *= 2                                    # Calculate child index based on dimensions
            if point[i] >= cell.center[i]:
                cell_child_id += 1                                # Determine the quadrant for the child
                child.min_bounds[i] = cell.center[i]              # Set child's min bound
                child.max_bounds[i] = cell.max_bounds[i]          # Set child's max bound
            else:
                child.min_bounds[i] = cell.min_bounds[i]          # Set child's min bound
                child.max_bounds[i] = cell.center[i]              # Set child's max bound
            child.center[i] = (child.min_bounds[i] + child.max_bounds[i]) / 2.  # Calculate child's center
            width = child.max_bounds[i] - child.min_bounds[i]      # Calculate width of the child

            child.barycenter[i] = point[i]                        # Store point in child's barycenter
            child.squared_max_width = max(child.squared_max_width, width*width)  # Update squared max width

        # Store the point info and the size to account for duplicated points
        child.point_index = point_index                            # Store point index in child
        child.cumulative_size = size                               # Store cumulative size in child

        # Store the child cell in the correct place in children
        cell.children[cell_child_id] = child.cell_id               # Assign child cell ID to parent's children array

        if DEBUGFLAG:
            # Assert that the point is in the right range
            self._check_point_in_cell(point, child)                # Check if the point lies within the cell's boundaries
        if self.verbose > 10:
            printf("[QuadTree] inserted point %li in new child %li\n",
                   point_index, cell_id)                           # Print debug message if verbosity level is high

        return cell_id                                            # Return the ID of the created cell

    cdef bint _is_duplicate(self, float32_t[3] point1, float32_t[3] point2) noexcept nogil:
        """Check if the two given points are equals."""
        cdef int i                                                # Declare loop variable
        cdef bint res = True                                      # Initialize result as true
        for i in range(self.n_dimensions):
            # Use EPSILON to avoid numerical error that would overgrow the tree
            res &= fabsf(point1[i] - point2[i]) <= EPSILON         # Compare each dimension of the two points
        return res                                                # Return whether points are considered duplicates
    # 选择包含给定查询点的单元格的子节点。
    cdef intp_t _select_child(self, float32_t[3] point, Cell* cell) noexcept nogil:
        """Select the child of cell which contains the given query point."""
        cdef:
            int i
            intp_t selected_child = 0

        for i in range(self.n_dimensions):
            # 通过比较点与单元格边界（使用预先计算的中心点）来选择正确的子单元格插入点。
            selected_child *= 2
            if point[i] >= cell.center[i]:
                selected_child += 1
        return cell.children[selected_child]

    # 初始化一个带有常量的单元格结构。
    cdef void _init_cell(self, Cell* cell, intp_t parent, intp_t depth) noexcept nogil:
        """Initialize a cell structure with some constants."""
        cell.parent = parent
        cell.is_leaf = True
        cell.depth = depth
        cell.squared_max_width = 0
        cell.cumulative_size = 0
        for i in range(self.n_cells_per_cell):
            cell.children[i] = SIZE_MAX

    # 使用给定的空间边界初始化根节点。
    cdef void _init_root(self, float32_t[3] min_bounds, float32_t[3] max_bounds
                         ) noexcept nogil:
        """Initialize the root node with the given space boundaries"""
        cdef:
            int i
            float32_t width
            Cell* root = &self.cells[0]

        self._init_cell(root, -1, 0)
        for i in range(self.n_dimensions):
            root.min_bounds[i] = min_bounds[i]
            root.max_bounds[i] = max_bounds[i]
            root.center[i] = (max_bounds[i] + min_bounds[i]) / 2.
            width = max_bounds[i] - min_bounds[i]
            root.squared_max_width = max(root.squared_max_width, width*width)
        root.cell_id = 0

        self.cell_count += 1
    # 定义一个 C 函数，用于检查给定点是否在指定的网格单元内
    cdef int _check_point_in_cell(self, float32_t[3] point, Cell* cell
                                  ) except -1 nogil:
        """Check that the given point is in the cell boundaries."""
        
        # 如果设置了详细输出级别 >= 50
        if self.verbose >= 50:
            # 如果是三维空间
            if self.n_dimensions == 3:
                # 打印详细信息，包括点的坐标及所在单元格的边界信息和大小
                printf("[QuadTree] Checking point (%f, %f, %f) in cell %li "
                       "([%f/%f, %f/%f, %f/%f], size %li)\n",
                       point[0], point[1], point[2], cell.cell_id,
                       cell.min_bounds[0], cell.max_bounds[0], cell.min_bounds[1],
                       cell.max_bounds[1], cell.min_bounds[2], cell.max_bounds[2],
                       cell.cumulative_size)
            else:
                # 打印详细信息，包括点的坐标及所在单元格的边界信息和大小（二维情况）
                printf("[QuadTree] Checking point (%f, %f) in cell %li "
                       "([%f/%f, %f/%f], size %li)\n",
                       point[0], point[1], cell.cell_id, cell.min_bounds[0],
                       cell.max_bounds[0], cell.min_bounds[1],
                       cell.max_bounds[1], cell.cumulative_size)

        # 遍历网格的维度
        for i in range(self.n_dimensions):
            # 如果点的坐标小于单元格的最小边界或者大于等于最大边界，则抛出数值错误异常
            if (cell.min_bounds[i] > point[i] or
                    cell.max_bounds[i] <= point[i]):
                with gil:
                    # 构造错误信息，指示点超出单元格边界的具体轴信息
                    msg = "[QuadTree] InsertionError: point out of cell "
                    msg += "boundary.\nAxis %li: cell [%f, %f]; point %f\n"
                    msg %= i, cell.min_bounds[i],  cell.max_bounds[i], point[i]
                    raise ValueError(msg)
    def _check_coherence(self):
        """Check the coherence of the cells of the tree.

        Check that the info stored in each cell is compatible with the info
        stored in descendent and sibling cells. Raise a ValueError if this
        fails.
        """
        # 遍历树中的每个单元格进行一致性检查
        for cell in self.cells[:self.cell_count]:
            # 检查插入点的重心是否在单元格边界内部
            self._check_point_in_cell(cell.barycenter, &cell)

            if not cell.is_leaf:
                # 计算子节点中的点数并与其累积大小进行比较
                n_points = 0
                for idx in range(self.n_cells_per_cell):
                    child_id = cell.children[idx]
                    if child_id != -1:
                        child = self.cells[child_id]
                        n_points += child.cumulative_size
                        assert child.cell_id == child_id, (
                            "Cell id not correctly initialized.")
                if n_points != cell.cumulative_size:
                    raise ValueError(
                        "Cell {} is incoherent. Size={} but found {} points "
                        "in children. ({})"
                        .format(cell.cell_id, cell.cumulative_size,
                                n_points, cell.children))

        # 确保树中的点数与根单元格中的累积大小相对应
        if self.n_points != self.cells[0].cumulative_size:
            raise ValueError(
                "QuadTree is incoherent. Size={} but found {} points "
                "in children."
                .format(self.n_points, self.cells[0].cumulative_size))

    def get_cell(self, point):
        """return the id of the cell containing the query point or raise
        ValueError if the point is not in the tree
        """
        # 定义一个查询点的数组并初始化
        cdef float32_t[3] query_pt
        cdef int i

        assert len(point) == self.n_dimensions, (
            "Query point should be a point in dimension {}."
            .format(self.n_dimensions))

        for i in range(self.n_dimensions):
            query_pt[i] = point[i]

        # 调用内部方法获取包含查询点的单元格的ID
        return self._get_cell(query_pt, 0)
    cdef int _get_cell(self, float32_t[3] point, intp_t cell_id=0
                       ) except -1 nogil:
        """guts of get_cell.

        Return the id of the cell containing the query point or raise ValueError
        if the point is not in the tree"""
        cdef:
            intp_t selected_child  # 声明选定的子节点的 ID
            Cell* cell = &self.cells[cell_id]  # 获取给定 cell_id 的 Cell 对象引用

        if cell.is_leaf:  # 如果当前 cell 是叶子节点
            if self._is_duplicate(cell.barycenter, point):  # 检查查询点是否与叶子节点的质心重复
                if self.verbose > 99:
                    printf("[QuadTree] Found point in cell: %li\n",
                           cell.cell_id)  # 打印找到查询点所在的 cell 的信息
                return cell_id  # 返回叶子节点的 ID
            with gil:
                raise ValueError("Query point not in the Tree.")  # 如果查询点不在树中，则引发 ValueError 异常

        selected_child = self._select_child(point, cell)  # 根据查询点选择子节点
        if selected_child > 0:  # 如果成功选择了一个子节点
            if self.verbose > 99:
                printf("[QuadTree] Selected_child: %li\n", selected_child)  # 打印所选子节点的信息
            return self._get_cell(point, selected_child)  # 递归调用以继续查找子节点中的查询点
        with gil:
            raise ValueError("Query point not in the Tree.")  # 如果无法选择有效的子节点，则引发 ValueError 异常

    # Pickling primitives

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (_QuadTree, (self.n_dimensions, self.verbose), self.__getstate__())  # 返回用于序列化对象的元组

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}  # 创建空字典以存储对象状态信息
        d["max_depth"] = self.max_depth  # 存储最大深度
        d["cell_count"] = self.cell_count  # 存储单元格计数
        d["capacity"] = self.capacity  # 存储容量
        d["n_points"] = self.n_points  # 存储点数
        d["cells"] = self._get_cell_ndarray().base  # 存储单元格数组的基本视图
        return d  # 返回状态字典

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]  # 恢复最大深度
        self.cell_count = d["cell_count"]  # 恢复单元格计数
        self.capacity = d["capacity"]  # 恢复容量
        self.n_points = d["n_points"]  # 恢复点数

        if 'cells' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')  # 如果状态中不包含单元格数组，则引发 ValueError 异常

        cell_ndarray = d['cells']  # 获取恢复的单元格数组

        if (cell_ndarray.ndim != 1 or
                cell_ndarray.dtype != CELL_DTYPE or
                not cell_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')  # 如果单元格数组的布局不符合预期，则引发 ValueError 异常

        self.capacity = cell_ndarray.shape[0]  # 更新容量
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)  # 调整树的大小以容纳更新的容量

        cdef Cell[:] cell_mem_view = cell_ndarray  # 创建单元格内存视图
        memcpy(
            pto=self.cells,  # 目标地址为 self.cells
            pfrom=&cell_mem_view[0],  # 源地址为 cell_mem_view 的第一个元素
            size=self.capacity * sizeof(Cell),  # 复制的字节数为单元格数组的容量乘以单个单元格的大小
        )

    # Array manipulation methods, to convert it to numpy or to resize
    # self.cells array
    # 定义一个 Cython 方法，返回包含 Cell 结构的 NumPy 结构化数组
    cdef Cell[:] _get_cell_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        # 定义数组形状
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.cell_count
        # 定义数组步长
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Cell)
        # 创建 Cell 类型的 NumPy 结构化数组
        cdef Cell[:] arr
        Py_INCREF(CELL_DTYPE)
        arr = PyArray_NewFromDescr(
            subtype=<PyTypeObject *> np.ndarray,
            descr=CELL_DTYPE,
            nd=1,
            dims=shape,
            strides=strides,
            data=<void*> self.cells,
            flags=cnp.NPY_ARRAY_DEFAULT,
            obj=None,
        )
        # 增加对 Tree 对象的引用计数
        Py_INCREF(self)
        # 设置 NumPy 数组的基础对象为当前对象
        if PyArray_SetBaseObject(arr.base, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array!")
        return arr

    # 定义一个 Cython 方法，用于调整内部数组的大小
    cdef int _resize(self, intp_t capacity) except -1 nogil:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 如果调用 _resize_c 方法失败，通过 GIL 机制抛出 MemoryError 异常
        if self._resize_c(capacity) != 0:
            # 只有在需要抛出异常时才获取全局解释器锁
            with gil:
                raise MemoryError()

    # 定义一个 Cython 方法，实现内部数组的大小调整
    cdef int _resize_c(self, intp_t capacity=SIZE_MAX) except -1 nogil:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 如果容量与当前容量相同且 cells 不为空，则返回 0
        if capacity == self.capacity and self.cells != NULL:
            return 0

        # 如果容量为 SIZE_MAX，则根据当前容量设置默认初始值或者将容量扩展为原来的两倍
        if <size_t> capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 9  # 默认初始值设为最小值
            else:
                capacity = 2 * self.capacity

        # 安全地重新分配内存给 cells
        safe_realloc(&self.cells, capacity)

        # 如果容量小于当前的 cell_count，调整计数器
        if capacity < self.cell_count:
            self.cell_count = capacity

        # 更新容量
        self.capacity = capacity
        return 0

    # 定义一个 Python 暴露的 Cython 方法，用于汇总数据
    def _py_summarize(self, float32_t[:] query_pt, float32_t[:, :] X, float angle):
        # Used for testing summarize
        cdef:
            float32_t[:] summary  # 汇总结果数组
            int n_samples  # 样本数量

        # 获取样本数量
        n_samples = X.shape[0]
        # 创建具有四倍样本数量空间的 float32_t 类型的 NumPy 数组
        summary = np.empty(4 * n_samples, dtype=np.float32)

        # 调用 summarize 方法进行数据汇总
        idx = self.summarize(&query_pt[0], &summary[0], angle * angle)
        return idx, summary
```