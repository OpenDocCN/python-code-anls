# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_quad_tree.pxd`

```
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Author: Olivier Grisel <olivier.grisel@ensta.fr>

# See quad_tree.pyx for details.

cimport numpy as cnp                      # 导入 C 数组的 numpy 包装
from ..utils._typedefs cimport float32_t, intp_t   # 导入自定义类型

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0                        # 定义一个枚举类型 DEBUGFLAG，用于条件编译调试信息

cdef float EPSILON = 1e-6                 # 定义浮点数 EPSILON，用于比较接近零的浮点数值

# XXX: Careful to not change the order of the arguments. It is important to
# have is_leaf and max_width consecutive as it permits to avoid padding by
# the compiler and keep the size coherent for both C and numpy data structures.
cdef struct Cell:
    # Base storage structure for cells in a QuadTree object
    # QuadTree 对象中用于存储单元格的基础结构

    # Tree structure
    intp_t parent                # Parent cell of this cell
    intp_t[8] children           # Array pointing to children of this cell

    # Cell description
    intp_t cell_id               # Id of the cell in the cells array in the Tree
    intp_t point_index           # Index of the point at this cell (only defined
                                 # in non empty leaf)
    bint is_leaf                 # Does this cell have children?
    float32_t squared_max_width  # Squared value of the maximum width w
    intp_t depth                 # Depth of the cell in the tree
    intp_t cumulative_size       # Number of points included in the subtree with
                                 # this cell as a root.

    # Internal constants
    float32_t[3] center          # Store the center for quick split of cells
    float32_t[3] barycenter      # Keep track of the center of mass of the cell

    # Cell boundaries
    float32_t[3] min_bounds      # Inferior boundaries of this cell (inclusive)
    float32_t[3] max_bounds      # Superior boundaries of this cell (exclusive)


cdef class _QuadTree:
    # The QuadTree object is a quad tree structure constructed by inserting
    # recursively points in the tree and splitting cells in 4 so that each
    # leaf cell contains at most one point.
    # This structure also handle 3D data, inserted in trees with 8 children
    # for each node.

    # Parameters of the tree
    cdef public int n_dimensions         # Number of dimensions in X
    cdef public int verbose              # Verbosity of the output
    cdef intp_t n_cells_per_cell         # Number of children per node. (2 ** n_dimension)

    # Tree inner structure
    cdef public intp_t max_depth         # Max depth of the tree
    cdef public intp_t cell_count        # Counter for node IDs
    cdef public intp_t capacity          # Capacity of tree, in terms of nodes
    cdef public intp_t n_points          # Total number of points
    cdef Cell* cells                     # Array of nodes

    # Point insertion methods
    cdef int insert_point(self, float32_t[3] point, intp_t point_index,
                          intp_t cell_id=*) except -1 nogil
                                          # 插入点的方法，将点插入到树中的指定单元格
    # 插入新子节点时确定插入点的位置，根据给定的点和当前节点，返回插入点的索引
    cdef intp_t _insert_point_in_new_child(self, float32_t[3] point, Cell* cell,
                                           intp_t point_index, intp_t size=*
                                           ) noexcept nogil

    # 选择子节点，根据给定的点和当前节点，返回选中子节点的索引
    cdef intp_t _select_child(self, float32_t[3] point, Cell* cell) noexcept nogil

    # 检查两个点是否重复，返回布尔值表示是否重复
    cdef bint _is_duplicate(self, float32_t[3] point1, float32_t[3] point2) noexcept nogil

    # 创建树结构相对于查询点的摘要信息
    cdef long summarize(self, float32_t[3] point, float32_t* results,
                        float squared_theta=*, intp_t cell_id=*, long idx=*
                        ) noexcept nogil

    # 内部单元格初始化方法
    cdef void _init_cell(self, Cell* cell, intp_t parent, intp_t depth) noexcept nogil
    cdef void _init_root(self, float32_t[3] min_bounds, float32_t[3] max_bounds
                         ) noexcept nogil

    # 私有方法

    # 检查点是否在给定的单元格内，返回布尔值或者 -1（异常情况）
    cdef int _check_point_in_cell(self, float32_t[3] point, Cell* cell
                                  ) except -1 nogil

    # 私有数组操作，用于管理 "cells" 数组

    # 调整数组大小，返回 -1 表示失败，否则返回 0 表示成功
    cdef int _resize(self, intp_t capacity) except -1 nogil

    # 调整 "cells" 数组的容量，返回 -1 表示失败，否则返回 0 表示成功
    cdef int _resize_c(self, intp_t capacity=*) except -1 nogil

    # 获取包含指定点的单元格索引，返回 -1 表示失败，否则返回单元格的索引
    cdef int _get_cell(self, float32_t[3] point, intp_t cell_id=*) except -1 nogil

    # 返回 "cells" 数组作为 Cell 结构的视图，用于内部使用
    cdef Cell[:] _get_cell_ndarray(self)
```