# `D:\src\scipysrc\scipy\scipy\spatial\_qhull.pyx`

```
# cython: cpow=True
# 设置 Cython 编译器选项，启用 cpow 支持

"""
Wrappers for Qhull triangulation, plus some additional N-D geometry utilities

.. versionadded:: 0.9

"""
# 包的简介和版本信息

#
# Copyright (C)  Pauli Virtanen, 2010.
# 版权声明，作者和年份信息

# Distributed under the same BSD license as Scipy.
# 采用与 Scipy 相同的 BSD 许可证分发

import numpy as np
# 导入 NumPy 库
cimport numpy as np
# 使用 C 语言级的导入 NumPy 库
cimport cython
# 使用 Cython 的 C 语言级别导入

from cpython.pythread cimport (
    PyThread_type_lock, PyThread_allocate_lock, PyThread_free_lock,
    PyThread_acquire_lock, PyThread_release_lock)
# 导入 Python 线程相关的 C 函数

from . cimport _qhull
# 导入当前目录下的 _qhull 模块
from . cimport setlist
# 导入当前目录下的 setlist 模块
from libc cimport stdlib
# C 库中导入 stdlib
from libc.math cimport NAN
# C 库中导入 NAN 常量
from scipy._lib.messagestream cimport MessageStream
# 从 SciPy 库中导入 MessageStream 类
from libc.stdio cimport FILE
# 从 C 库中导入 FILE 类

from scipy.linalg.cython_lapack cimport dgetrf, dgetrs, dgecon
# 从 SciPy 的 Cython Lapack 中导入特定函数

np.import_array()
# 导入 NumPy 库中的 import_array() 函数

__all__ = ['Delaunay', 'ConvexHull', 'QhullError', 'Voronoi', 'HalfspaceIntersection', 'tsearch']
# 定义模块中导出的公共接口列表

#------------------------------------------------------------------------------
# Qhull interface
#------------------------------------------------------------------------------

cdef extern from "stdio.h":
    extern void *stdin
    extern void *stderr
    extern void *stdout
# 使用 C 语言的 extern 导入标准输入、输出流

cdef extern from "math.h":
    double fabs(double x) nogil
    double sqrt(double x) nogil
# 使用 C 语言的 extern 导入 fabs 和 sqrt 函数

cdef extern from "setjmp.h" nogil:
    ctypedef struct jmp_buf:
        pass
    int setjmp(jmp_buf STATE) nogil
    void longjmp(jmp_buf STATE, int VALUE) nogil
# 使用 C 语言的 extern 导入 setjmp.h 中的结构体和函数

# Define the clockwise constant
# 定义顺时针常量
cdef extern from "qhull_src/src/user_r.h":
    cdef enum:
        qh_ORIENTclock
# 使用 C 语言的 extern 导入 qh_ORIENTclock 常量

cdef extern from "qhull_src/src/qset_r.h":
    ctypedef union setelemT:
        void *p
        int i

    ctypedef struct setT:
        int maxsize
        setelemT e[1]

    int qh_setsize(qhT *, setT *set) nogil
    void qh_setappend(qhT *, setT **setp, void *elem) nogil
# 使用 C 语言的 extern 导入 qset_r.h 中的数据结构和函数

cdef extern from "qhull_src/src/libqhull_r.h":
    ctypedef double realT
    ctypedef double coordT
    ctypedef double pointT
    ctypedef int boolT
    ctypedef unsigned int flagT

    ctypedef struct facetT:
        coordT offset
        coordT *center
        coordT *normal
        facetT *next
        facetT *previous
        unsigned id
        setT *vertices
        setT *neighbors
        setT *ridges
        setT *coplanarset
        flagT simplicial
        flagT flipped
        flagT upperdelaunay
        flagT toporient
        flagT good
        unsigned visitid

    ctypedef struct vertexT:
        vertexT *next
        vertexT *previous
        unsigned int id, visitid
        pointT *point
        setT *neighbors

    ctypedef struct ridgeT:
        setT *vertices
        facetT *top
        facetT *bottom
# 使用 C 语言的 extern 导入 libqhull_r.h 中的数据结构和类型定义
    # 定义了一个结构体 `qhT`，描述了 Qhull 算法的配置和状态
    ctypedef struct qhT:
        boolT DELAUNAY         # 是否计算 Delaunay 三角剖分
        boolT SCALElast        # 是否缩放最后一个点
        boolT KEEPcoplanar     # 是否保持共面点
        boolT MERGEexact       # 是否精确合并顶点
        boolT NOerrexit        # 是否禁用错误退出
        boolT PROJECTdelaunay  # 是否投影到 Delaunay 三角剖分
        boolT ATinfinity       # 是否处理无穷远点
        boolT UPPERdelaunay    # 是否使用上半球 Delaunay
        boolT hasTriangulation # 是否存在三角剖分
        boolT hasAreaVolume    # 是否存在面积和体积
        int normal_size         # 法向量大小
        char *qhull_command     # Qhull 命令
        facetT *facet_list      # 面列表
        facetT *facet_tail      # 面尾部
        vertexT *vertex_list    # 顶点列表
        vertexT *vertex_tail    # 顶点尾部
        int num_facets          # 面数量
        int num_visible         # 可见面数量
        int num_vertices        # 顶点数量
        int center_size         # 中心大小
        unsigned int facet_id   # 面的标识号
        int hull_dim            # 凸壳维度
        int num_points          # 点的数量
        pointT *first_point     # 第一个点
        pointT *input_points    # 输入点
        coordT* feasible_point  # 可行点
        realT last_low          # 最低值
        realT last_high         # 最高值
        realT last_newhigh      # 最新最高值
        realT max_outside       # 最大外部值
        realT MINoutside        # 最小外部值
        realT DISTround         # 距离舍入
        realT totvol            # 总体积
        realT totarea           # 总面积
        jmp_buf errexit         # 错误退出跳转缓冲区
        setT *other_points      # 其他点集
        unsigned int visit_id   # 访问标识号
        unsigned int vertex_visit  # 顶点访问号

    # 外部定义的 Qhull 常量
    extern int qh_PRINToff
    extern int qh_ALL

    # Qhull 初始化函数声明
    void qh_init_A(qhT *, void *inp, void *out, void *err, int argc, char **argv) nogil
    void qh_init_B(qhT *, realT *points, int numpoints, int dim, boolT ismalloc) nogil

    # Qhull 标志检查与初始化函数声明
    void qh_checkflags(qhT *, char *, char *) nogil
    void qh_initflags(qhT *, char *) nogil
    void qh_option(qhT *, char *, char*, char* ) nogil

    # Qhull 释放函数声明
    void qh_freeqhull(qhT *, boolT) nogil
    void qh_memfreeshort(qhT *, int *curlong, int *totlong) nogil

    # Qhull 主算法函数声明
    void qh_qhull(qhT *) nogil

    # Qhull 输出检查与生成函数声明
    void qh_check_output(qhT *) nogil
    void qh_produce_output(qhT *) nogil

    # Qhull 三角剖分函数声明
    void qh_triangulate(qhT *) nogil

    # Qhull 多边形检查与处理函数声明
    void qh_checkpolygon(qhT *) nogil

    # Qhull 寻找所有好的面函数声明
    void qh_findgood_all(qhT *, facetT *facetlist) nogil

    # Qhull 追加打印格式函数声明
    void qh_appendprint(qhT *, int format) nogil

    # Qhull 获取点和维度函数声明
    setT *qh_pointvertex(qhT *) nogil
    realT *qh_readpoints(qhT *, int* num, int *dim, boolT* ismalloc) nogil

    # Qhull 重置函数声明
    void qh_zero(qhT *, void *errfile) nogil

    # Qhull 获取点 ID 函数声明
    int qh_pointid(qhT *, pointT *point) nogil

    # Qhull 查找最近顶点函数声明
    vertexT *qh_nearvertex(qhT *, facetT *facet, pointT *point, double *dist) nogil

    # Qhull 添加点函数声明
    boolT qh_addpoint(qhT *, pointT *furthest, facetT *facet, boolT checkdist) nogil

    # Qhull 查找最佳面函数声明
    facetT *qh_findbestfacet(qhT *, pointT *point, boolT bestoutside,
                             realT *bestdist, boolT *isoutside) nogil

    # Qhull 设置 Delaunay 函数声明
    void qh_setdelaunay(qhT *, int dim, int count, pointT *points) nogil

    # Qhull 设置半空间函数声明
    coordT* qh_sethalfspace_all(qhT *, int dim, int count, coordT* halfspaces, pointT *feasible)
cdef extern from "qhull_misc.h":
    ctypedef int CBLAS_INT   # 定义 CBLAS_INT 类型，与头文件中的实际类型对应
    void qhull_misc_lib_check()   # 声明 qhull_misc_lib_check 函数

    int qh_new_qhull_scipy(qhT *, int dim, int numpoints, realT *points,
                           boolT ismalloc, char* qhull_cmd, void *outfile,
                           void *errfile, coordT* feaspoint) nogil
    # 声明 qh_new_qhull_scipy 函数，接受 Qhull 对象指针及其它参数，无需 GIL

cdef extern from "qhull_src/src/io_r.h":
    ctypedef enum qh_RIDGE:
        qh_RIDGEall
        qh_RIDGEinner
        qh_RIDGEouter
    # 定义 qh_RIDGE 枚举类型，表示 Qhull 的不同边类型

    ctypedef void printvridgeT(qhT *, void *fp, vertexT *vertex, vertexT *vertexA,
                               setT *centers, boolT unbounded)
    # 声明 printvridgeT 函数类型，用于打印 Voronoi 边缘

    int qh_eachvoronoi_all(qhT *, FILE *fp, void* printvridge,
                           boolT isUpper, qh_RIDGE innerouter,
                           boolT inorder) nogil
    # 声明 qh_eachvoronoi_all 函数，用于迭代 Voronoi 区域，无需 GIL

    void qh_order_vertexneighbors(qhT *, vertexT *vertex) nogil
    # 声明 qh_order_vertexneighbors 函数，用于对顶点邻居进行排序，无需 GIL

    int qh_compare_facetvisit(const void *p1, const void *p2) nogil
    # 声明 qh_compare_facetvisit 函数，用于比较面访问的顺序，无需 GIL

cdef extern from "qhull_src/src/geom_r.h":
    pointT *qh_facetcenter(qhT *, setT *vertices) nogil
    # 声明 qh_facetcenter 函数，计算面的中心点，无需 GIL

    double qh_getarea(qhT *, facetT *facetlist) nogil
    # 声明 qh_getarea 函数，计算面的面积，无需 GIL

cdef extern from "qhull_src/src/poly_r.h":
    void qh_check_maxout(qhT *) nogil
    # 声明 qh_check_maxout 函数，检查 Qhull 内存使用情况，无需 GIL

cdef extern from "qhull_src/src/mem_r.h":
    void qh_memfree(qhT *, void *object, int insize)
    # 声明 qh_memfree 函数，释放 Qhull 相关内存，指定内存大小

from libc.stdlib cimport qsort
# 从标准库中导入 qsort 函数

#------------------------------------------------------------------------------
# Qhull wrapper
#------------------------------------------------------------------------------

# 在导入时检查 Qhull 库的兼容性
qhull_misc_lib_check()

# 定义 QhullError 异常类，继承自 RuntimeError
class QhullError(RuntimeError):
    pass

@cython.final
cdef class _Qhull:
    # 注意，qhT 结构体是单独分配的，以免与 CRT（在 Windows 上）不兼容
    cdef qhT *_qh   # Qhull 对象指针

    cdef list _point_arrays   # 存储点数组的列表
    cdef list _dual_point_arrays   # 存储对偶点数组的列表
    cdef MessageStream _messages   # 消息流对象

    cdef public bytes options   # 公共属性，选项字节串
    cdef public bytes mode_option   # 公共属性，模式选项字节串
    cdef public object furthest_site   # 公共属性，最远点选项

    cdef readonly int ndim   # 只读属性，维度数
    cdef int numpoints, _is_delaunay, _is_halfspaces   # 点数、是否 Delaunay、是否半空间

    cdef np.ndarray _ridge_points   # 存储 ridge 点的 NumPy 数组

    cdef list _ridge_vertices   # 存储 ridge 顶点的列表
    cdef object _ridge_error   # ridge 错误对象
    cdef int _nridges   # ridge 数量

    cdef np.ndarray _ridge_equations   # 存储 ridge 方程的 NumPy 数组
    cdef PyThread_type_lock _lock   # 线程锁对象

    @cython.final
    cdef void acquire_lock(self):
        # 获取锁，如果获取失败则阻塞直到获取成功
        if not PyThread_acquire_lock(self._lock, 0):
            PyThread_acquire_lock(self._lock, 1)

    # 释放锁
    cdef void release_lock(self):
        PyThread_release_lock(self._lock)

    def check_active(self):
        # 检查 Qhull 实例是否为 NULL，如果是则抛出异常
        if self._qh == NULL:
            raise RuntimeError("Qhull instance is closed")

    @cython.final
    def __dealloc__(self):
        # 定义本地变量：当前已释放内存、总共需要释放的内存
        cdef int curlong, totlong

        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 如果 _qh 不为 NULL，则进行内存释放操作
            if self._qh != NULL:
                # 调用 qhull 库函数释放内存
                qh_freeqhull(self._qh, qh_ALL)
                # 获取已释放的短期内存和总内存，并更新 curlong 和 totlong
                qh_memfreeshort(self._qh, &curlong, &totlong)
                # 使用 stdlib 释放 self._qh 指向的内存块
                stdlib.free(self._qh)
                # 将 self._qh 置为 NULL，表示已释放
                self._qh = NULL

                # 如果 curlong 或 totlong 不为 0，则抛出异常
                if curlong != 0 or totlong != 0:
                    raise QhullError(
                        "qhull: did not free %d bytes (%d pieces)" %
                        (totlong, curlong))

            # 如果 _messages 不为 None，则调用其 close() 方法关闭资源
            if self._messages is not None:
                self._messages.close()
        finally:
            # 释放锁，确保释放后续操作可以继续
            self.release_lock()

        # 释放 Python 线程锁 self._lock
        PyThread_free_lock(self._lock)

    @cython.final
    def close(self):
        """
        Uninitialize this instance
        """
        # 注意：这段代码是从 __dealloc__() 直接复制粘贴过来的，请保持同步。
        # 必须直接在 __dealloc__() 中编写这段代码，因为否则生成的 C 代码会
        # 尝试调用 PyObject_GetAttrStr(self, "close")，这在 Pypy 上会导致崩溃。

        # 定义本地变量：当前已释放内存、总共需要释放的内存
        cdef int curlong, totlong

        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 如果 _qh 不为 NULL，则进行内存释放操作
            if self._qh != NULL:
                # 调用 qhull 库函数释放内存
                qh_freeqhull(self._qh, qh_ALL)
                # 获取已释放的短期内存和总内存，并更新 curlong 和 totlong
                qh_memfreeshort(self._qh, &curlong, &totlong)
                # 使用 stdlib 释放 self._qh 指向的内存块
                stdlib.free(self._qh)
                # 将 self._qh 置为 NULL，表示已释放
                self._qh = NULL

                # 如果 curlong 或 totlong 不为 0，则抛出异常
                if curlong != 0 or totlong != 0:
                    raise QhullError(
                        "qhull: did not free %d bytes (%d pieces)" %
                        (totlong, curlong))

            # 如果 _messages 不为 None，则调用其 close() 方法关闭资源
            if self._messages is not None:
                self._messages.close()
        finally:
            # 释放锁，确保释放后续操作可以继续
            self.release_lock()

    @cython.final
    def get_points(self):
        # 如果 _point_arrays 的长度为 1，则直接返回第一个数组
        if len(self._point_arrays) == 1:
            return self._point_arrays[0]
        else:
            # 否则，使用 numpy 的 concatenate 函数连接所有数组的前 ndim 列，并按行连接
            return np.concatenate(
                [x[:,:self.ndim] for x in self._point_arrays],
                axis=0)

    @cython.final
    # 定义一个方法用于向凸包数据结构添加点
    def add_points(self, points, interior_point=None):
        # 声明变量及数组指针
        cdef int j
        cdef realT *p
        cdef facetT *facet
        cdef double bestdist
        cdef boolT isoutside
        cdef np.ndarray arr

        # 获取锁，确保线程安全
        self.acquire_lock()

        try:
            # 检查凸包是否处于活动状态
            self.check_active()

            # 将输入的点集转换为 NumPy 数组
            points = np.asarray(points)
            # 检查输入点集的维度和大小是否符合预期
            if points.ndim!=2 or points.shape[1] != self._point_arrays[0].shape[1]:
                raise ValueError("invalid size for new points array")
            # 若输入点集为空，则直接返回
            if points.size == 0:
                return

            # 若凸包是 Delaunay 三角化的，则对点集进行适当的处理
            if self._is_delaunay:
                arr = np.empty((points.shape[0], self.ndim+1), dtype=np.double)
                arr[:,:-1] = points
            elif self._is_halfspaces:
                # 如果是半空间结构，则将半空间数据存储在 _point_arrays 中，
                # 将对偶点数据稍后存储在 _dual_points 中
                self._point_arrays.append(np.array(points, copy=True))
                # 计算半空间的距离，用于后续处理
                dists = points[:, :-1].dot(interior_point)+points[:, -1]
                arr = np.array(-points[:, :-1]/dists, dtype=np.double, order="C", copy=True)
            else:
                # 否则直接复制输入的点集数据
                arr = np.array(points, dtype=np.double, order="C", copy=True)

            # 清空消息列表
            self._messages.clear()

            try:
                # 非本地错误处理
                exitcode = setjmp(self._qh[0].errexit)
                if exitcode != 0:
                    raise QhullError(self._messages.get())
                self._qh[0].NOerrexit = 0

                # 将点添加到三角化结构中
                if self._is_delaunay:
                    # 将点提升到抛物面上
                    qh_setdelaunay(self._qh, arr.shape[1], arr.shape[0], <realT*>arr.data)

                # 获取数组的数据指针
                p = <realT*>arr.data

                # 遍历每个点，寻找最佳的外部面元
                for j in range(arr.shape[0]):
                    facet = qh_findbestfacet(self._qh, p, 0, &bestdist, &isoutside)
                    # 如果点在凸包外部，则尝试将其添加到凸包中
                    if isoutside:
                        if not qh_addpoint(self._qh, p, facet, 0):
                            break
                    else:
                        # 否则将点追加到“其他点”列表中，以维护点的标识
                        qh_setappend(self._qh, &self._qh[0].other_points, p)

                    # 更新指针位置以处理下一个点
                    p += arr.shape[1]

                # 检查是否超出最大输出限制
                qh_check_maxout(self._qh)
                self._qh[0].hasTriangulation = 0

                # 若是半空间结构，则将处理后的数据存储在 _dual_point_arrays 中
                if self._is_halfspaces:
                    self._dual_point_arrays.append(arr)
                else:
                    self._point_arrays.append(arr)
                # 更新点数统计
                self.numpoints += arr.shape[0]

                # 更新面元的可见性
                with nogil:
                    qh_findgood_all(self._qh, self._qh[0].facet_list)
            finally:
                # 恢复非错误状态
                self._qh[0].NOerrexit = 1
        finally:
            # 释放锁
            self.release_lock()

    # 声明 Cython 的 final 方法
    @cython.final
    def get_paraboloid_shift_scale(self):
        # 声明两个变量用于存储抛物面的尺度和位移
        cdef double paraboloid_scale
        cdef double paraboloid_shift

        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 检查对象是否处于活动状态
            self.check_active()

            # 计算抛物面的尺度和位移
            if self._qh[0].SCALElast:
                paraboloid_scale = self._qh[0].last_newhigh / (
                    self._qh[0].last_high - self._qh[0].last_low)
                paraboloid_shift = - self._qh[0].last_low * paraboloid_scale
            else:
                paraboloid_scale = 1.0
                paraboloid_shift = 0.0

            # 返回抛物面的尺度和位移
            return paraboloid_scale, paraboloid_shift
        finally:
            # 释放锁
            self.release_lock()

    @cython.final
    def volume_area(self):
        # 声明两个变量用于存储体积和面积
        cdef double volume
        cdef double area

        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 检查对象是否处于活动状态
            self.check_active()

            # 将区域体积计算结果存入对象属性
            self._qh.hasAreaVolume = 0
            # 在无GIL的环境下计算区域面积
            with nogil:
                qh_getarea(self._qh, self._qh[0].facet_list)

            # 从对象属性中获取总体积和总面积
            volume = self._qh[0].totvol
            area = self._qh[0].totarea

            # 返回总体积和总面积
            return volume, area
        finally:
            # 释放锁
            self.release_lock()

    @cython.final
    def triangulate(self):
        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 检查对象是否处于活动状态
            self.check_active()

            # 在无GIL的环境下进行三角化处理，消除非简单面
            with nogil:
                qh_triangulate(self._qh)
        finally:
            # 释放锁
            self.release_lock()

    @cython.final
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.final
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def get_hull_points(self):
        """Returns all points currently contained in Qhull.
        It is equivalent to retrieving the input in most cases, except in
        halfspace mode, where the points are in fact the points of the dual
        hull.

        Returns
        -------
        points: array of double, shape (nrpoints, ndim)
            The array of points contained in Qhull.

        """
        # 声明一些变量，包括点、索引、点数、点的维度
        cdef pointT *point
        cdef int i, j, numpoints, point_ndim
        cdef np.ndarray[np.npy_double, ndim=2] points

        # 获取锁，确保线程安全
        self.acquire_lock()

        try:
            # 检查对象是否处于活动状态
            self.check_active()

            # 获取点的维度
            point_ndim = self.ndim

            # 如果是半空间模式，则点的维度减一
            if self._is_halfspaces:
                point_ndim -= 1

            # 如果是德劳内模式，则点的维度加一
            if self._is_delaunay:
                point_ndim += 1

            # 获取点的数量
            numpoints = self._qh.num_points

            # 创建一个空数组来存放点
            points = np.empty((numpoints, point_ndim))

            # 在无GIL的环境下，遍历并复制点的数据
            with nogil:
                point = self._qh.first_point
                for i in range(numpoints):
                    for j in range(point_ndim):
                        points[i,j] = point[j]
                    point += self._qh.hull_dim

            # 返回包含在Qhull中的所有点的数组
            return points
        finally:
            # 释放锁
            self.release_lock()

    @cython.final
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def get_hull_facets(self):
        """Returns the facets contained in the current Qhull.
        This function does not assume that the hull is simplicial,
        meaning that facets will have different number of vertices.
        It is thus less efficient but more general than get_simplex_facet_array.

        Returns
        -------
        facets: list of lists of ints
            The indices of the vertices forming each facet.
        """
        # 声明 Cython 变量和类型
        cdef facetT *facet  # Cython 指针类型，指向 Qhull 的 facet 结构体
        cdef vertexT* vertex  # Cython 指针类型，指向 Qhull 的 vertex 结构体
        cdef int i, j, numfacets, facet_ndim  # 声明整型变量
        cdef np.ndarray[np.double_t, ndim=2] equations  # 声明 NumPy 二维数组 equations
        cdef list facets, facetsi  # 声明 Python 列表 facets 和 facetsi

        # 获取锁以防止并发访问
        self.acquire_lock()
        try:
            # 检查当前 Qhull 对象是否活跃
            self.check_active()

            # 获取当前 Qhull 的维度
            facet_ndim = self.ndim

            # 如果 Qhull 是半空间，则维度减一
            if self._is_halfspaces:
                facet_ndim -= 1

            # 如果 Qhull 是 Delaunay 三角剖分，则维度加一
            if self._is_delaunay:
                facet_ndim += 1

            # 计算非隐藏的面片数量
            numfacets = self._qh.num_facets - self._qh.num_visible

            # 获取 Qhull 中的面片列表
            facet = self._qh.facet_list

            # 创建一个空的 NumPy 数组来存储方程式
            equations = np.empty((numfacets, facet_ndim+1))

            # 初始化 facets 列表
            facets = []

            # 遍历 Qhull 中的每个面片
            i = 0
            while facet and facet.next:
                facetsi = []
                # 将面片的法向量和偏移保存到 equations 数组中
                for j in range(facet_ndim):
                    equations[i, j] = facet.normal[j]
                equations[i, facet_ndim] = facet.offset

                j = 0
                # 遍历当前面片的顶点
                vertex = <vertexT*>facet.vertices.e[0].p
                while vertex:
                    # 保存顶点的索引号
                    ipoint = qh_pointid(self._qh, vertex.point)
                    facetsi.append(ipoint)
                    j += 1
                    vertex = <vertexT*>facet.vertices.e[j].p

                # 将当前面片的顶点索引列表添加到 facets 列表中
                i += 1
                facets.append(facetsi)

                # 移动到下一个面片
                facet = facet.next

            # 返回 facets 列表和 equations 数组
            return facets, equations
        finally:
            # 释放锁
            self.release_lock()

    @cython.final
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.final
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def get_extremes_2d(_Qhull self):
        """
        Compute the extremal points in a 2-D convex hull, i.e. the
        vertices of the convex hull, ordered counterclockwise.

        See qhull/io.c:qh_printextremes_2d

        """
        # 获取锁，确保线程安全
        self.acquire_lock()
        try:
            # 检查对象是否处于活动状态
            self.check_active()

            # 如果是 Delaunay 四边形，抛出值错误异常
            if self._is_delaunay:
                raise ValueError("Cannot compute for Delaunay")

            # 初始化下一个极值索引
            nextremes = 0
            # 创建包含100个整数的数组，用于存储极值点
            extremes_arr = np.zeros(100, dtype=np.intc)
            extremes = extremes_arr

            # 增加访问 ID，标记访问过的顶点
            self._qh[0].visit_id += 1
            self._qh[0].vertex_visit += 1

            # 获取第一个面
            facet = self._qh[0].facet_list
            startfacet = facet
            while facet:
                # 检查面是否已经被访问过，如果是，抛出 QhullError 异常
                if facet.visitid == self._qh[0].visit_id:
                    raise QhullError("Qhull internal error: loop in facet list")

                # 根据面的方向选择顶点 A 和顶点 B
                if facet.toporient:
                    vertexA = <vertexT*>facet.vertices.e[0].p
                    vertexB = <vertexT*>facet.vertices.e[1].p
                    nextfacet = <facetT*>facet.neighbors.e[0].p
                else:
                    vertexB = <vertexT*>facet.vertices.e[0].p
                    vertexA = <vertexT*>facet.vertices.e[1].p
                    nextfacet = <facetT*>facet.neighbors.e[1].p

                # 如果极值数组即将超出容量，进行扩展
                if nextremes + 2 >= extremes.shape[0]:
                    extremes = None
                    # 数组安全扩展
                    extremes_arr.resize(2*extremes_arr.shape[0]+1, refcheck=False)
                    extremes = extremes_arr

                # 如果顶点 A 没有被访问过，记录其 ID
                if vertexA.visitid != self._qh[0].vertex_visit:
                    vertexA.visitid = self._qh[0].vertex_visit
                    extremes[nextremes] = qh_pointid(self._qh, vertexA.point)
                    nextremes += 1

                # 如果顶点 B 没有被访问过，记录其 ID
                if vertexB.visitid != self._qh[0].vertex_visit:
                    vertexB.visitid = self._qh[0].vertex_visit
                    extremes[nextremes] = qh_pointid(self._qh, vertexB.point)
                    nextremes += 1

                # 标记当前面已被访问过
                facet.visitid = self._qh[0].visit_id
                facet = nextfacet

                # 如果回到起始面，结束循环
                if facet == startfacet:
                    break

            # 清空极值数组，只保留有效部分
            extremes = None
            extremes_arr.resize(nextremes, refcheck=False)
            # 返回极值数组
            return extremes_arr
        finally:
            # 释放锁资源
            self.release_lock()
cdef void _visit_voronoi(qhT *_qh, FILE *ptr, vertexT *vertex, vertexT *vertexA,
                         setT *centers, boolT unbounded) noexcept:
    # 将传入的指针转换为 Qhull 对象
    cdef _Qhull qh = <_Qhull>ptr
    # 定义变量 point_1, point_2, ix
    cdef int point_1, point_2, ix
    # 定义空列表 cur_vertices
    cdef list cur_vertices

    # 如果存在 _ridge_error，直接返回
    if qh._ridge_error is not None:
        return

    # 如果当前的边数大于等于 _ridge_points 数组的行数，则尝试扩展 _ridge_points 数组
    if qh._nridges >= qh._ridge_points.shape[0]:
        try:
            # 保证能够安全地调整数组大小
            qh._ridge_points.resize(2*qh._nridges + 1, 2, refcheck=False)
        except Exception, e:
            # 如果调整大小时出错，记录错误并返回
            qh._ridge_error = e
            return

    # 记录边所连接的点
    point_1 = qh_pointid(_qh, vertex.point)
    point_2 = qh_pointid(_qh, vertexA.point)

    # 将边连接的两点信息记录在 _ridge_points 中
    p = <int*>qh._ridge_points.data
    p[2*qh._nridges + 0] = point_1
    p[2*qh._nridges + 1] = point_2

    # 记录组成边的 Voronoi 顶点
    cur_vertices = []
    for i in range(qh_setsize(_qh, centers)):
        ix = (<facetT*>centers.e[i].p).visitid - 1
        cur_vertices.append(ix)
    qh._ridge_vertices.append(cur_vertices)

    # 边数加一
    qh._nridges += 1

    return


cdef void qh_order_vertexneighbors_nd(qhT *qh, int nd, vertexT *vertex) noexcept:
    # 根据维度数选择不同的排序方式
    if nd == 3:
        qh_order_vertexneighbors(qh, vertex)
    elif nd >= 4:
        # 对顶点的邻居进行排序
        qsort(<facetT**>&vertex.neighbors.e[0].p, qh_setsize(qh, vertex.neighbors),
              sizeof(facetT*), qh_compare_facetvisit)


#------------------------------------------------------------------------------
# Barycentric coordinates
#------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _get_barycentric_transforms(np.ndarray[np.double_t, ndim=2] points,
                                np.ndarray[np.npy_int, ndim=2] simplices,
                                double eps):
    """
    Compute barycentric affine coordinate transformations for given
    simplices.

    Returns
    -------
    Tinvs : array, shape (nsimplex, ndim+1, ndim)
        Barycentric transforms for each simplex.

        Tinvs[i,:ndim,:ndim] contains inverse of the matrix ``T``,
        and Tinvs[i,ndim,:] contains the vector ``r_n`` (see below).

    Notes
    -----
    Barycentric transform from ``x`` to ``c`` is defined by::

        T c = x - r_n

    where the ``r_1, ..., r_n`` are the vertices of the simplex.
    The matrix ``T`` is defined by the condition::

        T e_j = r_j - r_n

    where ``e_j`` is the unit axis vector, e.g, ``e_2 = [0,1,0,0,...]``
    This implies that ``T_ij = (r_j - r_n)_i``.

    For the barycentric transforms, we need to compute the inverse
    matrix ``T^-1`` and store the vectors ``r_n`` for each vertex.
    These are stacked into the `Tinvs` returned.

    """
    cdef np.ndarray[np.double_t, ndim=2] T
    cdef np.ndarray[np.double_t, ndim=3] Tinvs
    cdef int isimplex
    cdef int i, j
    cdef CBLAS_INT n, nrhs, lda, ldb
    cdef CBLAS_INT info = 0
    cdef CBLAS_INT ipiv[NPY_MAXDIMS+1]
    # 函数体部分未提供，以下部分需要根据实际情况补充完整
    # 声明整型变量 ndim（维度数）和 nsimplex（简单形状数）
    cdef int ndim, nsimplex
    # 声明双精度浮点型变量 anorm 和 rcond，并初始化 rcond 为 0.0
    cdef double anorm
    cdef double rcond = 0.0
    # 声明双精度浮点型变量 rcond_limit
    cdef double rcond_limit

    # 声明大小为 4*NPY_MAXDIMS 的双精度浮点型数组 work 和大小为 NPY_MAXDIMS 的整型数组 iwork
    cdef double work[4*NPY_MAXDIMS]
    cdef CBLAS_INT iwork[NPY_MAXDIMS]

    # 获取 points 数组的第二个维度作为 ndim
    ndim = points.shape[1]
    # 获取 simplices 数组的第一个维度作为 nsimplex
    nsimplex = simplices.shape[0]

    # 创建一个空的 ndim x ndim 的双精度浮点型数组 T
    T = np.empty((ndim, ndim), dtype=np.double)
    # 创建一个大小为 nsimplex x (ndim+1) x ndim 的全零双精度浮点型数组 Tinvs
    Tinvs = np.zeros((nsimplex, ndim+1, ndim), dtype=np.double)

    # 设置 rcond_limit 的值为 1000 倍的 eps，用于设置最大逆条件数的阈值
    rcond_limit = 1000 * eps

    # 使用 nogil 上下文，允许不使用全局解释器锁进行并行计算
    with nogil:
        # 遍历每个简单形状
        for isimplex in range(nsimplex):
            # 遍历每个维度
            for i in range(ndim):
                # 将 points[simplices[isimplex, ndim], i] 赋给 Tinvs[isimplex, ndim, i]
                Tinvs[isimplex, ndim, i] = points[simplices[isimplex, ndim], i]
                # 遍历每个维度
                for j in range(ndim):
                    # 计算 T[i,j]，其值为 points[simplices[isimplex, j], i] - Tinvs[isimplex, ndim, i]
                    T[i, j] = (points[simplices[isimplex, j], i]
                               - Tinvs[isimplex, ndim, i])
                # 设置 Tinvs[isimplex, i, i] 为 1
                Tinvs[isimplex, i, i] = 1

            # 计算矩阵的 1-范数以估计条件数
            anorm = _matrix_norm1(ndim, <double*> T.data)

            # 进行 LU 分解
            n = ndim
            nrhs = ndim
            lda = ndim
            ldb = ndim
            dgetrf(&n, &n, <double*> T.data, &lda, ipiv, &info)

            # 检查条件数
            if info == 0:
                dgecon("1", &n, <double*> T.data, &lda, &anorm, &rcond,
                       work, iwork, &info)

                # 如果条件数小于 rcond_limit，则判定变换似乎是奇异的
                if rcond < rcond_limit:
                    info = 1

            # 计算变换
            if info == 0:
                dgetrs("N", &n, &nrhs, <double*> T.data, &lda, ipiv,
                       (<double*> Tinvs.data) + ndim*(ndim+1)*isimplex,
                       &ldb, &info)

            # 处理退化的简单形状
            if info != 0:
                # 将 Tinvs[isimplex, i, j] 设置为 NaN
                for i in range(ndim+1):
                    for j in range(ndim):
                        Tinvs[isimplex, i, j] = NAN

    # 返回 Tinvs 数组作为函数结果
    return Tinvs
# 禁用边界检查以优化性能，用于 Cython 函数的装饰器
@cython.boundscheck(False)
# 计算给定 Fortran 排序的方阵的 1-范数
cdef double _matrix_norm1(int n, double *a) noexcept nogil:
    """Compute the 1-norm of a square matrix given in in Fortran order"""
    # 初始化最大列和为 0，列和为 colsum
    cdef double maxsum = 0, colsum
    # 使用两个循环遍历矩阵
    cdef int i, j

    for j in range(n):
        colsum = 0
        for i in range(n):
            colsum += fabs(a[0])  # 计算当前列的绝对值和
            a += 1  # 移动到下一个元素
        if maxsum < colsum:
            maxsum = colsum  # 更新最大列和
    return maxsum  # 返回计算得到的最大列和

# 检查点是否位于简单形状内部，使用重心坐标法，如果在内部，填充重心坐标到 c
cdef int _barycentric_inside(int ndim, double *transform,
                             const double *x, double *c, double eps) noexcept nogil:
    """
    Check whether point is inside a simplex, using barycentric
    coordinates.  `c` will be filled with barycentric coordinates, if
    the point happens to be inside.

    """
    cdef int i, j
    c[ndim] = 1.0  # 设置最后一个重心坐标为 1.0

    for i in range(ndim):
        c[i] = 0  # 初始化当前重心坐标为 0
        for j in range(ndim):
            c[i] += transform[ndim*i + j] * (x[j] - transform[ndim*ndim + j])  # 计算重心坐标的分量
        c[ndim] -= c[i]  # 更新最后一个重心坐标

        if not (-eps <= c[i] <= 1 + eps):  # 检查重心坐标分量是否在指定的误差范围内
            return 0  # 如果不在范围内，返回 0
    if not (-eps <= c[ndim] <= 1 + eps):  # 检查最后一个重心坐标是否在指定的误差范围内
        return 0  # 如果不在范围内，返回 0
    return 1  # 如果点在简单形状内部，返回 1

# 计算单个重心坐标
cdef void _barycentric_coordinate_single(int ndim, double *transform,
                                         const double *x, double *c, int i) noexcept nogil:
    """
    Compute a single barycentric coordinate.

    Before the ndim+1'th coordinate can be computed, the other must have
    been computed earlier.

    """
    cdef int j

    if i == ndim:
        c[ndim] = 1.0  # 设置最后一个重心坐标为 1.0
        for j in range(ndim):
            c[ndim] -= c[j]  # 计算最后一个重心坐标的值
    else:
        c[i] = 0  # 初始化当前重心坐标为 0
        for j in range(ndim):
            c[i] += transform[ndim*i + j] * (x[j] - transform[ndim*ndim + j])  # 计算重心坐标的分量

# 计算重心坐标
cdef void _barycentric_coordinates(int ndim, double *transform,
                                   const double *x, double *c) noexcept nogil:
    """
    Compute barycentric coordinates.

    """
    cdef int i, j
    c[ndim] = 1.0  # 设置最后一个重心坐标为 1.0
    for i in range(ndim):
        c[i] = 0  # 初始化当前重心坐标为 0
        for j in range(ndim):
            c[i] += transform[ndim*i + j] * (x[j] - transform[ndim*ndim + j])  # 计算重心坐标的分量
        c[ndim] -= c[i]  # 更新最后一个重心坐标


#------------------------------------------------------------------------------
# N-D geometry
#------------------------------------------------------------------------------

# 将点提升到更高维度空间上
cdef void _lift_point(DelaunayInfo_t *d, const double *x, double *z) noexcept nogil:
    cdef int i
    z[d.ndim] = 0  # 初始化额外维度的坐标为 0
    for i in range(d.ndim):
        z[i] = x[i]  # 复制原始维度的坐标
        z[d.ndim] += x[i]**2  # 计算额外维度的坐标
    z[d.ndim] *= d.paraboloid_scale  # 缩放额外维度的坐标
    z[d.ndim] += d.paraboloid_shift  # 平移额外维度的坐标

# 计算点到超平面的距离
cdef double _distplane(DelaunayInfo_t *d, int isimplex, double *point) noexcept nogil:
    """
    qh_distplane
    """
    cdef double dist
    cdef int k
    dist = d.equations[isimplex*(d.ndim+2) + d.ndim+1]  # 初始化距离为平面截距
    for k in range(d.ndim+1):
        dist += d.equations[isimplex*(d.ndim+2) + k] * point[k]  # 计算距离的累积值
    return dist


#------------------------------------------------------------------------------
# Finding simplices
#------------------------------------------------------------------------------

# 在代码中标记为查找简单形状的部分
#------------------------------------------------------------------------------

cdef int _is_point_fully_outside(DelaunayInfo_t *d, const double *x,
                                 double eps) noexcept nogil:
    """
    判断点是否完全位于三角剖分的边界框外部。

    Args:
    - d: DelaunayInfo_t 结构体，包含三角剖分信息
    - x: double 类型的数组，表示待检测点的坐标
    - eps: double 类型，表示允许的误差范围

    Returns:
    - 1 如果点完全位于边界框外部，否则返回 0
    """

    cdef int i
    for i in range(d.ndim):
        if x[i] < d.min_bound[i] - eps or x[i] > d.max_bound[i] + eps:
            return 1
    return 0

cdef int _find_simplex_bruteforce(DelaunayInfo_t *d, double *c,
                                  const double *x, double eps,
                                  double eps_broad) noexcept nogil:
    """
    通过遍历所有单纯形来查找包含点 `x` 的单纯形。

    Args:
    - d: DelaunayInfo_t 结构体，包含三角剖分信息
    - c: double 类型的数组，用于存储找到的单纯形的重心坐标
    - x: double 类型的数组，表示待查找点的坐标
    - eps: double 类型，表示允许的误差范围
    - eps_broad: double 类型，用于宽松检查邻近单纯形的误差范围

    Returns:
    - 找到的单纯形的索引，如果未找到则返回 -1
    """

    cdef int inside, isimplex
    cdef int k, m, ineighbor
    cdef double *transform

    if _is_point_fully_outside(d, x, eps):
        return -1

    for isimplex in range(d.nsimplex):
        transform = d.transform + isimplex*d.ndim*(d.ndim+1)

        if transform[0] == transform[0]:
            # transform 有效（非 NaN）
            inside = _barycentric_inside(d.ndim, transform, x, c, eps)
            if inside:
                return isimplex
        else:
            # transform 无效（NaN，表示退化单纯形）

            # 通过检查邻居单纯形来替代内部检查，使用更大的 epsilon

            for k in range(d.ndim+1):
                ineighbor = d.neighbors[(d.ndim+1)*isimplex + k]
                if ineighbor == -1:
                    continue

                transform = d.transform + ineighbor*d.ndim*(d.ndim+1)
                if transform[0] != transform[0]:
                    # 另一个糟糕的单纯形
                    continue

                _barycentric_coordinates(d.ndim, transform, x, c)

                # 检查点是否（几乎）位于邻近单纯形内部
                inside = 1
                for m in range(d.ndim+1):
                    if d.neighbors[(d.ndim+1)*ineighbor + m] == isimplex:
                        # 允许朝 isimplex 方向的额外余地
                        if not (-eps_broad <= c[m] <= 1 + eps):
                            inside = 0
                            break
                    else:
                        # 正常检查
                        if not (-eps <= c[m] <= 1 + eps):
                            inside = 0
                            break
                if inside:
                    return ineighbor
    return -1
    cdef int k, m, ndim, inside, isimplex, cycle_k
    cdef double *transform

    # 获取数据维度
    ndim = d.ndim
    # 设置起始简单形式索引为第一个
    isimplex = start[0]

    # 如果起始简单形式索引小于0或大于等于简单形式数量，将其设为0
    if isimplex < 0 or isimplex >= d.nsimplex:
        isimplex = 0

    # 迭代次数的最大限制：应该足够大以保证算法通常成功，但小于nsimplex，
    # 以便在算法失败时，主要成本仍然来自暴力搜索。
    for cycle_k in range(1 + d.nsimplex//4):
        # 如果isimplex为-1，跳出循环
        if isimplex == -1:
            break

        # 计算变换矩阵的指针
        transform = d.transform + isimplex*ndim*(ndim+1)

        # 初始化inside为1，表示在简单形式内
        inside = 1
        for k in range(ndim+1):
            # 计算目标点在第k个顶点处的重心坐标
            _barycentric_coordinate_single(ndim, transform, x, c, k)

            if c[k] < -eps:
                # 目标点在第k个邻居的方向上！
                m = d.neighbors[(ndim+1)*isimplex + k]
                if m == -1:
                    # 点在网格外：中止搜索
                    start[0] = isimplex
                    return -1

                isimplex = m
                inside = -1
                break
            elif c[k] <= 1 + eps:
                # 我们在这个简单形式内
                pass
            else:
                # 我们在外面（或者坐标为nan；退化的简单形式）
                inside = 0

        if inside == -1:
            # 跳转到另一个简单形式
            continue
        elif inside == 1:
            # 找到了正确的简单形式！
            break
        else:
            # 完全失败（存在退化的简单形式）。退回到暴力搜索
            isimplex = _find_simplex_bruteforce(d, c, x, eps, eps_broad)
            break
    else:
        # 如果算法未收敛，则回退到蛮力搜索
        # 调用蛮力搜索函数来寻找简单形式
        isimplex = _find_simplex_bruteforce(d, c, x, eps, eps_broad)

    # 将找到的简单形式的索引存入起始位置数组中
    start[0] = isimplex
    # 返回找到的简单形式的索引
    return isimplex
    # 定义 C 函数 _find_simplex，寻找包含点 `x` 的简单形式
    # 参数说明：
    #   - d: DelaunayInfo_t 结构体指针，用于存储 Delaunay 三角化的信息
    #   - c: 双精度浮点数组，用于存储点 `x` 对应的坐标
    #   - x: 双精度浮点数组，表示要查找的点的坐标
    #   - start: 整型指针，表示查找的起始位置
    #   - eps: 双精度浮点数，控制精度
    #   - eps_broad: 双精度浮点数，控制广义精度
    # 返回值：
    #   - 整型，表示找到的简单形式的索引
    """
    Find simplex containing point `x` by walking the triangulation.

    Notes
    -----
    This algorithm is similar as used by ``qh_findbest``.  The idea
    is the following:

    1. Delaunay triangulation is a projection of the lower half of a convex
       hull, of points lifted on a paraboloid.

       Simplices in the triangulation == facets on the convex hull.

    2. If a point belongs to a given simplex in the triangulation,
       its image on the paraboloid is on the positive side of
       the corresponding facet.

    3. However, it is not necessarily the *only* such facet.

    4. Also, it is not necessarily the facet whose hyperplane distance
       to the point on the paraboloid is the largest.

    ..note::

        If I'm not mistaken, `qh_findbestfacet` finds a facet for
        which the plane distance is maximized -- so it doesn't always
        return the simplex containing the point given. For example:

        >>> p = np.array([(1 - 1e-4, 0.1)])
        >>> points = np.array([(0,0), (1, 1), (1, 0), (0.99189033, 0.37674127),
        ...                    (0.99440079, 0.45182168)], dtype=np.double)
        >>> tri = qhull.delaunay(points)
        >>> tri.simplices
        array([[4, 1, 0],
               [4, 2, 1],
               [3, 2, 0],
               [3, 4, 0],
               [3, 4, 2]])
        >>> dist = qhull.plane_distance(tri, p)
        >>> dist
        array([[-0.12231439,  0.00184863,  0.01049659, -0.04714842,
                0.00425905]])
        >>> tri.simplices[dist.argmax()]
        array([3, 2, 0]

        Now, the maximally positive-distant simplex is [3, 2, 0], although
        the simplex containing the point is [4, 2, 1].

    In this algorithm, we walk around the tessellation trying to locate
    a positive-distant facet. After finding one, we fall back to a
    directed search.

    """
    # 声明变量
    cdef int isimplex, k, ineigh
    cdef int ndim
    cdef double z[NPY_MAXDIMS+1]
    cdef double best_dist, dist
    cdef int changed

    # 如果点在外面，则返回 -1
    if _is_point_fully_outside(d, x, eps):
        return -1
    # 如果没有简单形式，则返回 -1
    if d.nsimplex <= 0:
        return -1

    # 设置维度
    ndim = d.ndim
    isimplex = start[0]

    # 如果起始索引不在有效范围内，则设置为 0
    if isimplex < 0 or isimplex >= d.nsimplex:
        isimplex = 0

    # 将点映射到抛物面上
    _lift_point(d, x, z)

    # 在三角剖分中寻找具有正平面距离的面
    best_dist = _distplane(d, isimplex, z)
    changed = 1
    # 当变量 changed 为真时执行循环，直到其为假
    while changed:
        # 如果 best_dist 大于 0，则退出循环
        if best_dist > 0:
            break
        # 将 changed 设为 0，表示未发生变化
        changed = 0
        # 遍历 ndim+1 个元素的范围
        for k in range(ndim+1):
            # 获取当前简单形状 isimplex 的第 k 个邻居
            ineigh = d.neighbors[(ndim+1)*isimplex + k]
            # 如果邻居索引为 -1，跳过本次循环
            if ineigh == -1:
                continue
            # 计算当前简单形状 isimplex 和其邻居 ineigh 之间的距离
            dist = _distplane(d, ineigh, z)

            # 注意：这里添加了 eps，否则该代码可能不会终止！
            # 编译器可能会使用 FPU 的扩展精度，导致 (dist > best_dist) 成立，
            # 但在存储到双倍大小后，dist == best_dist，导致循环无法终止

            # 如果距离 dist 大于 best_dist 加上 eps*(1 + fabs(best_dist))，
            # 则执行以下操作
            if dist > best_dist + eps*(1 + fabs(best_dist)):
                # 注意：这是有意为之：我们在循环的中间跳出，
                # 然后从下一个 k 继续循环。
                #
                # 这显然更有效地扫描不同的方向。我们不需要完全的精度，
                # 因为无论如何我们之后会进行定向搜索。
                # 将 isimplex 设置为邻居索引，更新 best_dist 和 changed 标志
                isimplex = ineigh
                best_dist = dist
                changed = 1

    # 现在应该在包含该点的简单形状附近，使用定向搜索定位它
    # 将起始点设置为 isimplex，然后调用 _find_simplex_directed 函数进行定向搜索
    start[0] = isimplex
    return _find_simplex_directed(d, c, x, start, eps, eps_broad)
#------------------------------------------------------------------------------
# Delaunay triangulation interface, for Python
#------------------------------------------------------------------------------

class _QhullUser:
    """
    Takes care of basic dealings with the Qhull objects
    """

    _qhull = None

    def __init__(self, qhull, incremental=False):
        # 初始化_QhullUser对象
        self._qhull = None
        try:
            # 尝试更新_QhullUser对象
            self._update(qhull)
            if incremental:
                # 如果是增量模式，则将传入的qhull对象直接赋给self._qhull
                self._qhull = qhull
        finally:
            # 最终操作，确保qhull对象被关闭
            if qhull is not self._qhull:
                qhull.close()

    def close(self):
        """
        close()

        Finish incremental processing.

        Call this to free resources taken up by Qhull, when using the
        incremental mode. After calling this, adding more points is no
        longer possible.
        """
        if self._qhull is not None:
            # 如果self._qhull不为None，则关闭qhull对象
            self._qhull.close()
            self._qhull = None

    def __del__(self):
        # 对象销毁时调用close方法释放资源
        self.close()

    def _update(self, qhull):
        # 更新_QhullUser对象的状态信息，包括获取点集合、维度、点数、最小边界和最大边界
        self._points = qhull.get_points()
        self.ndim = self._points.shape[1]
        self.npoints = self._points.shape[0]
        self.min_bound = self._points.min(axis=0)
        self.max_bound = self._points.max(axis=0)
    def _add_points(self, points, restart=False, interior_point=None):
        """
        add_points(points, restart=False)

        Process a set of additional new points.

        Parameters
        ----------
        points : ndarray
            New points to add. The dimensionality should match that of the
            initial points.
        restart : bool, optional
            Whether to restart processing from scratch, rather than
            adding points incrementally.
        interior_point : ndarray or None, optional
            Interior point for Qhull computation.

        Raises
        ------
        QhullError
            Raised when Qhull encounters an error condition, such as
            geometrical degeneracy when options to resolve are not enabled.

        See Also
        --------
        close

        Notes
        -----
        You need to specify ``incremental=True`` when constructing the
        object to be able to add points incrementally. Incremental addition
        of points is also not possible after `close` has been called.

        """
        # 检查是否处于增量模式，若不是则抛出运行时错误
        if self._qhull is None:
            raise RuntimeError("incremental mode not enabled or already closed")

        # 如果指定了 restart=True，则将当前点集与新点集合并
        if restart:
            points = np.concatenate([self._points, points], axis=0)
            # 使用新的点集创建一个新的 Qhull 对象
            qhull = _Qhull(self._qhull.mode_option, points,
                           options=self._qhull.options,
                           furthest_site=self._qhull.furthest_site,
                           incremental=True, interior_point=interior_point)
            try:
                # 更新当前对象的状态为新创建的 Qhull 对象
                self._update(qhull)
                self._qhull = qhull
            finally:
                # 如果新创建的 Qhull 对象与当前对象不同，则关闭新对象
                if qhull is not self._qhull:
                    qhull.close()
            return

        # 向当前 Qhull 对象添加新的点集
        self._qhull.add_points(points, interior_point)
        # 更新当前对象的状态为最新的 Qhull 对象
        self._update(self._qhull)
class Delaunay(_QhullUser):
    """
    Delaunay(points, furthest_site=False, incremental=False, qhull_options=None)

    Delaunay tessellation in N dimensions.

    .. versionadded:: 0.9

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to triangulate
    furthest_site : bool, optional
        Whether to compute a furthest-site Delaunay triangulation.
        Default: False

        .. versionadded:: 0.12.0
    incremental : bool, optional
        Allow adding new points incrementally. This takes up some additional
        resources.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual for
        details. Option "Qt" is always enabled.
        Default:"Qbb Qc Qz Qx Q12" for ndim > 4 and "Qbb Qc Qz Q12" otherwise.
        Incremental mode omits "Qz".

        .. versionadded:: 0.12.0

    Attributes
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    simplices : ndarray of ints, shape (nsimplex, ndim+1)
        Indices of the points forming the simplices in the triangulation.
        For 2-D, the points are oriented counterclockwise.
    neighbors : ndarray of ints, shape (nsimplex, ndim+1)
        Indices of neighbor simplices for each simplex.
        The kth neighbor is opposite to the kth vertex.
        For simplices at the boundary, -1 denotes no neighbor.
    equations : ndarray of double, shape (nsimplex, ndim+2)
        [normal, offset] forming the hyperplane equation of the facet
        on the paraboloid
        (see `Qhull documentation <http://www.qhull.org/>`__ for more).
    paraboloid_scale, paraboloid_shift : float
        Scale and shift for the extra paraboloid dimension
        (see `Qhull documentation <http://www.qhull.org/>`__ for more).
    transform : ndarray of double, shape (nsimplex, ndim+1, ndim)
        Affine transform from ``x`` to the barycentric coordinates ``c``.
        This is defined by::

            T c = x - r

        At vertex ``j``, ``c_j = 1`` and the other coordinates zero.

        For simplex ``i``, ``transform[i,:ndim,:ndim]`` contains
        inverse of the matrix ``T``, and ``transform[i,ndim,:]``
        contains the vector ``r``.

        If the simplex is degenerate or nearly degenerate, its
        barycentric transform contains NaNs.
    vertex_to_simplex : ndarray of int, shape (npoints,)
        Lookup array, from a vertex, to some simplex which it is a part of.
        If qhull option "Qc" was not specified, the list will contain -1
        for points that are not vertices of the tessellation.

    """

    def __init__(self, points, furthest_site=False, incremental=False, qhull_options=None):
        """
        Initialize Delaunay triangulation object.

        Parameters
        ----------
        points : ndarray of floats, shape (npoints, ndim)
            Coordinates of points to triangulate
        furthest_site : bool, optional
            Whether to compute a furthest-site Delaunay triangulation.
            Default: False

        incremental : bool, optional
            Allow adding new points incrementally. This takes up some additional
            resources.

        qhull_options : str, optional
            Additional options to pass to Qhull.
        """

        # 继承 QhullUser 类的初始化方法
        super(Delaunay, self).__init__()

        # 将输入点坐标存储到对象的 points 属性中
        self.points = points

        # 是否计算最远点 Delaunay 三角化
        self.furthest_site = furthest_site

        # 是否允许增量添加新点
        self.incremental = incremental

        # 设置 Qhull 选项，默认使用 Qt 选项
        self.qhull_options = qhull_options if qhull_options is not None else "Qt"

        # 初始化其他属性为空列表或数组
        self.simplices = None
        self.neighbors = None
        self.equations = None
        self.paraboloid_scale = None
        self.paraboloid_shift = None
        self.transform = None
        self.vertex_to_simplex = None

        # 调用 QhullUser 类的方法初始化
        self._update()  # 执行初始化更新操作

    def _update(self):
        """
        Perform Delaunay triangulation using Qhull.
        """
        # 调用 Qhull 库进行三角化，更新对象的相关属性
        pass
    # ndarray of int, shape (nfaces, ndim)
    # 表示形成点集凸包的面的顶点。
    # 该数组包含属于凸包的 (N-1) 维面的点的索引，这些面形成了三角剖分的凸包。

    .. note::
       通过Delaunay三角剖分计算凸包是低效的，并且容易受到数值不稳定性的影响。
       建议使用 `ConvexHull` 替代。

    # ndarray of int, shape (ncoplanar, 3)
    # 共面点的索引及其最近面和最近顶点的索引。
    # 由于数值精度问题，这些共面点未被包括在三角剖分中。

    # 如果未指定选项 "Qc"，则不会计算这个列表。

    .. versionadded:: 0.12.0

    # tuple of two ndarrays of int; (indptr, indices)
    # 每个顶点的邻近顶点。第 k 个顶点的邻近顶点索引为 ``indices[indptr[k]:indptr[k+1]]``。

    furthest_site
    # 如果这是一个最远点位三角剖分，则为True，否则为False。

    .. versionadded:: 1.4.0

    Raises
    ------
    QhullError
    # 当Qhull遇到错误条件时引发，例如几何退化但未启用解决选项时。

    ValueError
    # 如果输入的数组不兼容。

    Notes
    -----
    # 使用Qhull库进行网格化计算
    `Qhull library <http://www.qhull.org/>`__。

    .. note::
       除非传递选项 "QJ"，否则Qhull不能保证每个输入点都出现在Delaunay三角剖分中的顶点中。
       被省略的点列在 `coplanar` 属性中。

    Examples
    --------
    # 一组点的三角剖分：

    >>> import numpy as np
    >>> points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
    >>> from scipy.spatial import Delaunay
    >>> tri = Delaunay(points)

    # 我们可以绘制它：

    >>> import matplotlib.pyplot as plt
    >>> plt.triplot(points[:,0], points[:,1], tri.simplices)
    >>> plt.plot(points[:,0], points[:,1], 'o')
    >>> plt.show()

    # 形成三角剖分的两个三角形的点索引和坐标：

    >>> tri.simplices
    array([[2, 3, 0],                 # 可能会有所不同
           [3, 1, 0]], dtype=int32)

    # 请注意，根据舍入误差的情况，简单形式可能会与上述不同。

    >>> points[tri.simplices]
    array([[[ 1. ,  0. ],            # 可能会有所不同
            [ 1. ,  1. ],
            [ 0. ,  0. ]],
           [[ 1. ,  1. ],
            [ 0. ,  1.1],
            [ 0. ,  0. ]]])

    # 三角形1是三角形0的唯一邻居，并且它在三角形1的顶点1的对面：

    >>> tri.neighbors[1]
    array([-1,  0, -1], dtype=int32)
    >>> points[tri.simplices[1,1]]
    array([ 0. ,  1.1])



    # 创建一个包含两个元素的 NumPy 数组，元素分别为 0.0 和 1.1
    array([ 0. ,  1.1])



    We can find out which triangle points are in:



    # 我们可以找出哪些三角形包含了这些点：
    We can find out which triangle points are in:



    >>> p = np.array([(0.1, 0.2), (1.5, 0.5), (0.5, 1.05)])
    >>> tri.find_simplex(p)
    array([ 1, -1, 1], dtype=int32)



    # 创建一个包含三个点的 NumPy 数组 p，然后通过 tri.find_simplex(p) 找出每个点所在的简单形状的索引。
    >>> p = np.array([(0.1, 0.2), (1.5, 0.5), (0.5, 1.05)])
    >>> tri.find_simplex(p)
    array([ 1, -1, 1], dtype=int32)



    The returned integers in the array are the indices of the simplex the
    corresponding point is in. If -1 is returned, the point is in no simplex.
    Be aware that the shortcut in the following example only works correctly
    for valid points as invalid points result in -1 which is itself a valid
    index for the last simplex in the list.



    # 返回的整数数组中的整数是对应点所在的简单形状的索引。如果返回 -1，则表示该点不在任何简单形状中。
    # 需要注意，下面示例中的快捷方式仅对有效点有效，因为无效点会导致返回 -1，而 -1 本身是列表中最后一个简单形状的有效索引。
    The returned integers in the array are the indices of the simplex the
    corresponding point is in. If -1 is returned, the point is in no simplex.
    Be aware that the shortcut in the following example only works correctly
    for valid points as invalid points result in -1 which is itself a valid
    index for the last simplex in the list.



    >>> p_valids = np.array([(0.1, 0.2), (0.5, 1.05)])
    >>> tri.simplices[tri.find_simplex(p_valids)]
    array([[3, 1, 0],                 # may vary
           [3, 1, 0]], dtype=int32)



    # 创建一个包含两个有效点的 NumPy 数组 p_valids，然后通过 tri.find_simplex(p_valids) 找出每个有效点所在的简单形状的索引，再通过 tri.simplices 获取对应的简单形状。
    >>> p_valids = np.array([(0.1, 0.2), (0.5, 1.05)])
    >>> tri.simplices[tri.find_simplex(p_valids)]
    array([[3, 1, 0],                 # 可能会有所不同
           [3, 1, 0]], dtype=int32)



    We can also compute barycentric coordinates in triangle 1 for
    these points:



    # 我们还可以计算这些点在三角形 1 中的重心坐标：
    We can also compute barycentric coordinates in triangle 1 for
    these points:



    >>> b = tri.transform[1,:2].dot(np.transpose(p - tri.transform[1,2]))
    >>> np.c_[np.transpose(b), 1 - b.sum(axis=0)]
    array([[ 0.1       ,  0.09090909,  0.80909091],
           [ 1.5       , -0.90909091,  0.40909091],
           [ 0.5       ,  0.5       ,  0.        ]])



    # 计算三角形 1 中这些点的重心坐标：
    >>> b = tri.transform[1,:2].dot(np.transpose(p - tri.transform[1,2]))
    >>> np.c_[np.transpose(b), 1 - b.sum(axis=0)]
    array([[ 0.1       ,  0.09090909,  0.80909091],
           [ 1.5       , -0.90909091,  0.40909091],
           [ 0.5       ,  0.5       ,  0.        ]])



    The coordinates for the first point are all positive, meaning it
    is indeed inside the triangle. The third point is on an edge,
    hence its null third coordinate.



    # 第一个点的坐标全为正，这意味着它确实在三角形内。第三个点在边上，因此它的第三个坐标为零。
    The coordinates for the first point are all positive, meaning it
    is indeed inside the triangle. The third point is on an edge,
    hence its null third coordinate.



    """



    # 下面是一个类的定义，用于进行三角剖分和计算相关的几何操作。
    """



    def __init__(self, points, furthest_site=False, incremental=False,
                 qhull_options=None):



    # 初始化方法，用于创建一个 Delaunay 对象。
    def __init__(self, points, furthest_site=False, incremental=False,
                 qhull_options=None):



        if np.ma.isMaskedArray(points):
            raise ValueError('Input points cannot be a masked array')



        # 检查输入的点集是否是掩码数组，如果是则抛出 ValueError 异常。
        if np.ma.isMaskedArray(points):
            raise ValueError('Input points cannot be a masked array')



        points = np.ascontiguousarray(points, dtype=np.double)



        # 将输入的点集转换为连续的内存布局，并指定数据类型为双精度浮点数。
        points = np.ascontiguousarray(points, dtype=np.double)



        if points.ndim != 2:
            raise ValueError("Input points array must have 2 dimensions.")



        # 检查点集数组是否为二维数组，如果不是则抛出 ValueError 异常。
        if points.ndim != 2:
            raise ValueError("Input points array must have 2 dimensions.")



        if qhull_options is None:
            if not incremental:
                qhull_options = b"Qbb Qc Qz Q12"
            else:
                qhull_options = b"Qc"
            if points.shape[1] >= 5:
                qhull_options += b" Qx"
        else:
            qhull_options = qhull_options.encode('latin1')



        # 根据传入的参数设置 qhull_options，用于调用 Qhull 库进行几何计算。
        if qhull_options is None:
            if not incremental:
                qhull_options = b"Qbb Qc Qz Q12"
            else:
                qhull_options = b"Qc"
            if points.shape[1] >= 5:
                qhull_options += b" Qx"
        else:
            qhull_options = qhull_options.encode('latin1')



        # 调用 Qhull 库进行几何计算
        qhull = _Qhull(b"d", points, qhull_options, required_options=b"Qt",
                       furthest_site=furthest_site, incremental=incremental)



        # 调用 Qhull 库的用户接口进行初始化
        qhull = _Qhull(b"d", points, qhull_options, required_options=b"Qt",
                       furthest_site=furthest_site, incremental=incremental)



        # 初始化基类 _QhullUser
        _QhullUser.__init__(self, qhull, incremental=incremental)



        # 初始化基类 _QhullUser
        _QhullUser.__init__(self, qhull, incremental=incremental)



        self.furthest_site = furthest_site



        # 设置属性 furthest_site
        self.furthest_site = furthest_site
``
    def transform(self):
        """
        Affine transform from ``x`` to the barycentric coordinates ``c``.

        :type: *ndarray of double, shape (nsimplex, ndim+1, ndim)*

        This is defined by::

            T c = x - r

        At vertex ``j``, ``c_j = 1`` and the other coordinates zero.

        For simplex ``i``, ``transform[i,:ndim,:ndim]`` contains
        inverse of the matrix ``T``, and ``transform[i,ndim,:]``
        contains the vector ``r``.

        """
        # 如果尚未计算变换矩阵，调用 _get_barycentric_transforms 计算并存储
        if self._transform is None:
            self._transform = _get_barycentric_transforms(self.points,
                                                          self.simplices,
                                                          np.finfo(float).eps)
        return self._transform

    @property
    @cython.boundscheck(False)
    def vertex_to_simplex(self):
        """
        Lookup array, from a vertex, to some simplex which it is a part of.

        :type: *ndarray of int, shape (npoints,)*
        """
        # 如果尚未计算顶点到单形的映射关系
        if self._vertex_to_simplex is None:
            # 创建一个大小为 npoints 的空数组，填充为 -1
            self._vertex_to_simplex = np.empty((self.npoints,), dtype=np.intc)
            self._vertex_to_simplex.fill(-1)

            # 包括共面点
            self._vertex_to_simplex[self.coplanar[:,0]] = self.coplanar[:,2]

            # 包括其他点
            arr = self._vertex_to_simplex
            simplices = self.simplices

            nsimplex = self.nsimplex
            ndim = self.ndim

            # 使用 nogil 以释放全局解释器锁进行并行化处理
            with nogil:
                for isimplex in range(nsimplex):
                    for k in range(ndim+1):
                        ivertex = simplices[isimplex, k]
                        if arr[ivertex] == -1:
                            arr[ivertex] = isimplex

        return self._vertex_to_simplex

    @property
    @cython.boundscheck(False)
    # 定义一个方法，用于获取顶点及其相邻顶点的信息
    def vertex_neighbor_vertices(self):
        """
        Neighboring vertices of vertices.

        Tuple of two ndarrays of int: (indptr, indices). The indices of
        neighboring vertices of vertex `k` are
        ``indices[indptr[k]:indptr[k+1]]``.

        """
        # 声明变量 i, j, k 作为循环索引
        cdef int i, j, k
        # 声明变量 nsimplex, npoints, ndim 来存储相应的属性值
        cdef int nsimplex, npoints, ndim
        # 声明一个二维的 numpy 数组 simplices，存储复合体的简单形式信息
        cdef np.ndarray[np.npy_int, ndim=2] simplices
        # 声明一个 setlist.setlist_t 类型的 sets 变量，用于存储顶点集合

        # 如果未计算过顶点相邻顶点信息
        if self._vertex_neighbor_vertices is None:
            # 获取顶点维度、顶点数、简单形式数
            ndim = self.ndim
            npoints = self.npoints
            nsimplex = self.nsimplex
            simplices = self.simplices

            # 初始化 sets 变量，用于存储顶点集合信息
            setlist.init(&sets, npoints, ndim+1)

            try:
                # 使用 nogil 上下文，优化性能，禁止 GIL
                with nogil:
                    # 遍历每个简单形式
                    for i in range(nsimplex):
                        # 遍历当前简单形式的每个顶点
                        for j in range(ndim+1):
                            # 再次遍历当前简单形式的每个顶点
                            for k in range(ndim+1):
                                # 如果两个顶点不相同且未添加到集合中
                                if simplices[i,j] != simplices[i,k]:
                                    # 将两个顶点添加到集合中，若集合已满则抛出 MemoryError
                                    if setlist.add(&sets, simplices[i,j], simplices[i,k]):
                                        # 若添加失败，抛出 MemoryError 异常
                                        with gil:
                                            raise MemoryError

                # 将 sets 转换为稀疏矩阵格式，并存储到 _vertex_neighbor_vertices 属性中
                self._vertex_neighbor_vertices = setlist.tocsr(&sets)
            finally:
                # 释放 sets 占用的内存资源
                setlist.free(&sets)

        # 返回计算得到的顶点相邻顶点信息
        return self._vertex_neighbor_vertices

    # 定义一个属性，关闭索引边界检查，提高性能
    @property
    @cython.boundscheck(False)
    # 定义一个方法，计算点集的凸包
    def convex_hull(self):
        """
        Vertices of facets forming the convex hull of the point set.

        :type: *ndarray of int, shape (nfaces, ndim)*

        The array contains the indices of the points
        belonging to the (N-1)-dimensional facets that form the convex
        hull of the triangulation.

        .. note::

           Computing convex hulls via the Delaunay triangulation is
           inefficient and subject to increased numerical instability.
           Use `ConvexHull` instead.

        """
        # 声明一些变量
        cdef int isimplex, k, j, ndim, nsimplex, m, msize
        cdef object out
        cdef np.ndarray[np.npy_int, ndim=2] arr
        cdef np.ndarray[np.npy_int, ndim=2] neighbors
        cdef np.ndarray[np.npy_int, ndim=2] simplices

        # 将成员变量赋给本地变量，提高访问效率
        neighbors = self.neighbors
        simplices = self.simplices
        ndim = self.ndim
        nsimplex = self.nsimplex

        # 初始化输出数组的大小为10xndim
        msize = 10
        out = np.empty((msize, ndim), dtype=np.intc)
        arr = out

        m = 0
        # 遍历每一个简单形
        for isimplex in range(nsimplex):
            # 遍历每个简单形的每个顶点
            for k in range(ndim+1):
                # 如果邻接数组中的值为-1，表明当前顶点是凸壳的一部分
                if neighbors[isimplex,k] == -1:
                    # 将除了k之外的顶点索引复制到输出数组中
                    for j in range(ndim+1):
                        if j < k:
                            arr[m,j] = simplices[isimplex,j]
                        elif j > k:
                            arr[m,j-1] = simplices[isimplex,j]
                    m += 1

                    # 如果输出数组大小已满，进行扩展
                    if m >= msize:
                        arr = None
                        msize = 2*msize + 1
                        # 数组可以安全地调整大小
                        out.resize(msize, ndim, refcheck=False)
                        arr = out

        arr = None
        # 最终输出凸壳的顶点索引数组，将数组大小调整为实际需要的大小
        out.resize(m, ndim, refcheck=False)
        return out

    @cython.boundscheck(False)
    def find_simplex(self, xi, bruteforce=False, tol=None):
        """
        find_simplex(self, xi, bruteforce=False, tol=None)

        Find the simplices containing the given points.

        Parameters
        ----------
        tri : DelaunayInfo
            Delaunay triangulation
        xi : ndarray of double, shape (..., ndim)
            Points to locate
        bruteforce : bool, optional
            Whether to only perform a brute-force search
        tol : float, optional
            Tolerance allowed in the inside-triangle check.
            Default is ``100*eps``.

        Returns
        -------
        i : ndarray of int, same shape as `xi`
            Indices of simplices containing each point.
            Points outside the triangulation get the value -1.

        Notes
        -----
        This uses an algorithm adapted from Qhull's ``qh_findbestfacet``,
        which makes use of the connection between a convex hull and a
        Delaunay triangulation. After finding the simplex closest to
        the point in N+1 dimensions, the algorithm falls back to
        directed search in N dimensions.

        """

        # Define Cython specific types and variables
        cdef DelaunayInfo_t info
        cdef int isimplex
        cdef double c[NPY_MAXDIMS]
        cdef double eps, eps_broad
        cdef int start
        cdef int k
        cdef np.ndarray[np.double_t, ndim=2] x
        cdef np.ndarray[np.npy_int, ndim=1] out_

        # Convert xi to a numpy array if it is not already
        xi = np.asanyarray(xi)

        # Check if xi has the correct dimensionality
        if xi.shape[-1] != self.ndim:
            raise ValueError("wrong dimensionality in xi")

        # Reshape xi for processing
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])
        x = np.ascontiguousarray(xi.astype(np.double))
        x_shape = x.shape

        start = 0

        # Set tolerance eps based on input or default value
        if tol is None:
            eps = 100 * np.finfo(np.double).eps
        else:
            eps = tol
        eps_broad = sqrt(eps)

        # Initialize output array for indices of simplices
        out = np.zeros((xi.shape[0],), dtype=np.intc)
        out_ = out

        # Obtain Delaunay triangulation information
        _get_delaunay_info(&info, self, 1, 0, 0)

        # Perform simplex search based on bruteforce flag
        if bruteforce:
            # Brute-force search method
            with nogil:
                for k in range(x_shape[0]):
                    isimplex = _find_simplex_bruteforce(
                        &info, c,
                        <double*>x.data + info.ndim*k,
                        eps, eps_broad)
                    out_[k] = isimplex
        else:
            # Optimized search method
            with nogil:
                for k in range(x_shape[0]):
                    isimplex = _find_simplex(&info, c,
                                             <double*>x.data + info.ndim*k,
                                             &start, eps, eps_broad)
                    out_[k] = isimplex

        # Reshape output to match the original shape of xi
        return out.reshape(xi_shape[:-1])

    @cython.boundscheck(False)
    # 计算超平面到所有单纯形中点 `xi` 的距离

    cdef np.ndarray[np.double_t, ndim=2] x
    # 声明一个二维双精度浮点型数组 x
    cdef np.ndarray[np.double_t, ndim=2] out_
    # 声明一个二维双精度浮点型数组 out_
    cdef DelaunayInfo_t info
    # 声明一个 DelaunayInfo_t 类型的结构体 info
    cdef double z[NPY_MAXDIMS+1]
    # 声明一个长度为 NPY_MAXDIMS+1 的双精度浮点型数组 z
    cdef int i, j
    # 声明两个整数变量 i 和 j

    if xi.shape[-1] != self.ndim:
        # 如果 xi 的最后一个维度长度与三角剖分对象的维度不相同，则引发 ValueError 异常
        raise ValueError("xi has different dimensionality than "
                         "triangulation")

    xi_shape = xi.shape
    # 记录 xi 的形状
    xi = xi.reshape(-1, xi.shape[-1])
    # 将 xi 重塑为二维数组，行数为自动调整以匹配最后一个维度长度，列数为最后一个维度的长度
    x = np.ascontiguousarray(xi.astype(np.double))
    # 将 xi 转换为连续的双精度浮点类型数组 x
    x_shape = x.shape
    # 记录 x 的形状

    _get_delaunay_info(&info, self, 0, 0, 0)
    # 调用 C 函数 _get_delaunay_info，将 self 和 info 的地址作为参数传递给该函数

    out = np.zeros((x.shape[0], info.nsimplex), dtype=np.double)
    # 创建一个形状为 (x 行数, 单纯形数) 的双精度浮点型零数组 out
    out_ = out
    # 将 out 赋值给 out_

    with nogil:
        # 进入无 GIL 区域
        for i in range(x_shape[0]):
            # 循环遍历 x 的行数
            for j in range(info.nsimplex):
                # 循环遍历单纯形的数量
                _lift_point(&info, (<double*>x.data) + info.ndim*i, z)
                # 调用 C 函数 _lift_point，将 info、x 数据的双精度指针、以及 z 作为参数传递
                out_[i,j] = _distplane(&info, j, z)
                # 使用 C 函数 _distplane 计算超平面到单纯形 j 中点 z 的距离

    return out.reshape(xi_shape[:-1] + (self.nsimplex,))
    # 将 out 重塑为原始 xi 的形状，但是在最后加上单纯形的维度
# 返回点所在的简单形式的索引。该函数与 `Delaunay.find_simplex` 执行相同的操作。
def tsearch(tri, xi):
    """
    tsearch(tri, xi)

    根据给定的点找到包含这些点的简单形式。该函数与 `Delaunay.find_simplex` 执行相同的操作。

    Parameters
    ----------
    tri : DelaunayInfo
        Delaunay 三角化对象
    xi : ndarray of double, shape (..., ndim)
        要定位的点

    Returns
    -------
    i : ndarray of int, same shape as `xi`
        包含每个点的简单形式的索引。
        不在三角化内部的点将返回值 -1。

    See Also
    --------
    Delaunay.find_simplex

    Notes
    -----
    .. versionadded:: 0.9

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch
    >>> rng = np.random.default_rng()

    随机点的 Delaunay 三角化：

    >>> pts = rng.random((20, 2))
    >>> tri = Delaunay(pts)
    >>> _ = delaunay_plot_2d(tri)

    找到包含给定点集的简单形式：

    >>> loc = rng.uniform(0.2, 0.8, (5, 2))
    >>> s = tsearch(tri, loc)
    >>> plt.triplot(pts[:, 0], pts[:, 1], tri.simplices[s], 'b-', mask=s==-1)
    >>> plt.scatter(loc[:, 0], loc[:, 1], c='r', marker='x')
    >>> plt.show()

    """
    return tri.find_simplex(xi)

# 为 foo 设置文档字符串，使用 bar 的文档字符串，绕过 Cython 0.28 中的更改
# 参考 https://github.com/scipy/scipy/pull/8581
def _copy_docstr(dst, src):
    try:
        dst.__doc__ = src.__doc__
    except AttributeError:
        dst.__func__.__doc__ = src.__func__.__doc__

_copy_docstr(Delaunay.add_points, _QhullUser._add_points)

#------------------------------------------------------------------------------
# Delaunay 三角化接口，用于低级 C
#------------------------------------------------------------------------------

cdef int _get_delaunay_info(DelaunayInfo_t *info,
                            obj,
                            int compute_transform,
                            int compute_vertex_to_simplex,
                            int compute_vertex_neighbor_vertices) except -1:
    cdef np.ndarray[np.double_t, ndim=3] transform
    cdef np.ndarray[np.npy_int, ndim=1] vertex_to_simplex
    cdef np.ndarray[np.npy_int, ndim=1] vn_indices, vn_indptr
    cdef np.ndarray[np.double_t, ndim=2] points = obj.points
    cdef np.ndarray[np.npy_int, ndim=2] simplices = obj.simplices
    cdef np.ndarray[np.npy_int, ndim=2] neighbors = obj.neighbors
    cdef np.ndarray[np.double_t, ndim=2] equations = obj.equations
    cdef np.ndarray[np.double_t, ndim=1] min_bound = obj.min_bound
    cdef np.ndarray[np.double_t, ndim=1] max_bound = obj.max_bound

    info.ndim = points.shape[1]
    info.npoints = points.shape[0]
    info.nsimplex = simplices.shape[0]
    info.points = <double*>points.data
    info.simplices = <int*>simplices.data
    info.neighbors = <int*>neighbors.data
    info.equations = <double*>equations.data
    // 将对象 obj 中的 paraboloid_scale 赋值给 info 结构体的 paraboloid_scale 成员
    info.paraboloid_scale = obj.paraboloid_scale;
    // 将对象 obj 中的 paraboloid_shift 赋值给 info 结构体的 paraboloid_shift 成员
    info.paraboloid_shift = obj.paraboloid_shift;

    // 如果需要计算变换信息
    if (compute_transform) {
        // 获取对象 obj 中的 transform 对象
        transform = obj.transform;
        // 将 transform 数据的指针转换为 double* 类型，并赋值给 info 结构体的 transform 成员
        info.transform = <double*>transform.data;
    } else {
        // 如果不需要计算变换信息，则将 info 结构体的 transform 成员设为 NULL
        info.transform = NULL;
    }

    // 如果需要计算顶点到单纯形映射
    if (compute_vertex_to_simplex) {
        // 获取对象 obj 中的 vertex_to_simplex 对象
        vertex_to_simplex = obj.vertex_to_simplex;
        // 将 vertex_to_simplex 数据的指针转换为 int* 类型，并赋值给 info 结构体的 vertex_to_simplex 成员
        info.vertex_to_simplex = <int*>vertex_to_simplex.data;
    } else {
        // 如果不需要计算顶点到单纯形映射，则将 info 结构体的 vertex_to_simplex 成员设为 NULL
        info.vertex_to_simplex = NULL;
    }

    // 如果需要计算顶点的邻居顶点
    if (compute_vertex_neighbor_vertices) {
        // 获取对象 obj 中的 vertex_neighbor_vertices 对象
        vn_indptr, vn_indices = obj.vertex_neighbor_vertices;
        // 将 vn_indices 数据的指针转换为 int* 类型，并赋值给 info 结构体的 vertex_neighbors_indices 成员
        info.vertex_neighbors_indices = <int*>vn_indices.data;
        // 将 vn_indptr 数据的指针转换为 int* 类型，并赋值给 info 结构体的 vertex_neighbors_indptr 成员
        info.vertex_neighbors_indptr = <int*>vn_indptr.data;
    } else {
        // 如果不需要计算顶点的邻居顶点，则将 info 结构体的 vertex_neighbors_indices 和 vertex_neighbors_indptr 成员设为 NULL
        info.vertex_neighbors_indices = NULL;
        info.vertex_neighbors_indptr = NULL;
    }

    // 将 min_bound 数据的指针转换为 double* 类型，并赋值给 info 结构体的 min_bound 成员
    info.min_bound = <double*>min_bound.data;
    // 将 max_bound 数据的指针转换为 double* 类型，并赋值给 info 结构体的 max_bound 成员
    info.max_bound = <double*>max_bound.data;

    // 返回整数 0，表示函数执行成功
    return 0;
#------------------------------------------------------------------------------
# Convex hulls
#------------------------------------------------------------------------------

class ConvexHull(_QhullUser):
    """
    ConvexHull(points, incremental=False, qhull_options=None)

    Convex hulls in N dimensions.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to construct a convex hull from
    incremental : bool, optional
        Allow adding new points incrementally. This takes up some additional
        resources.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qx" for ndim > 4 and "" otherwise)
        Option "Qt" is always enabled.

    Attributes
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    vertices : ndarray of ints, shape (nvertices,)
        Indices of points forming the vertices of the convex hull.
        For 2-D convex hulls, the vertices are in counterclockwise order.
        For other dimensions, they are in input order.
    simplices : ndarray of ints, shape (nfacet, ndim)
        Indices of points forming the simplical facets of the convex hull.
    neighbors : ndarray of ints, shape (nfacet, ndim)
        Indices of neighbor facets for each facet.
        The kth neighbor is opposite to the kth vertex.
        -1 denotes no neighbor.
    equations : ndarray of double, shape (nfacet, ndim+1)
        [normal, offset] forming the hyperplane equation of the facet
        (see `Qhull documentation <http://www.qhull.org/>`__  for more).
    coplanar : ndarray of int, shape (ncoplanar, 3)
        Indices of coplanar points and the corresponding indices of
        the nearest facets and nearest vertex indices.  Coplanar
        points are input points which were *not* included in the
        triangulation due to numerical precision issues.

        If option "Qc" is not specified, this list is not computed.
    good : ndarray of bool or None
        A one-dimensional Boolean array indicating which facets are
        good. Used with options that compute good facets, e.g. QGn
        and QG-n. Good facets are defined as those that are
        visible (n) or invisible (-n) from point n, where
        n is the nth point in 'points'. The 'good' attribute may be
        used as an index into 'simplices' to return the good (visible)
        facets: simplices[good]. A facet is visible from the outside
        of the hull only, and neither coplanarity nor degeneracy count
        as cases of visibility.

        If a "QGn" or "QG-n" option is not specified, None is returned.

        .. versionadded:: 1.3.0
    area : float
        Surface area of the convex hull when input dimension > 2.
        When input `points` are 2-dimensional, this is the perimeter of the convex hull.

        .. versionadded:: 0.17.0
    """
    volume : float
        # 定义变量 volume，表示凸包的体积（当输入维度大于2时）或凸包的面积（当输入的点是二维时）。
        # 0.17.0 版本添加此功能。

    Raises
    ------
    QhullError
        # 当 Qhull 遇到错误条件时引发，如几何退化但未启用解决选项时。
    ValueError
        # 如果输入的数组不兼容，则引发此异常。

    Notes
    -----
    # 使用 Qhull 库计算凸包。
    # `Qhull library <http://www.qhull.org/>`__ 提供支持。

    Examples
    --------

    Convex hull of a random set of points:

    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))   # 30 random points in 2-D
    >>> hull = ConvexHull(points)

    Plot it:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(points[:,0], points[:,1], 'o')
    >>> for simplex in hull.simplices:
    ...     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    We could also have directly used the vertices of the hull, which
    for 2-D are guaranteed to be in counterclockwise order:

    >>> plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    >>> plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    >>> plt.show()

    Facets visible from a point:

    Create a square and add a point above the square.

    >>> generators = np.array([[0.2, 0.2],
    ...                        [0.2, 0.4],
    ...                        [0.4, 0.4],
    ...                        [0.4, 0.2],
    ...                        [0.3, 0.6]])

    Call ConvexHull with the QG option. QG4 means
    compute the portions of the hull not including
    point 4, indicating the facets that are visible
    from point 4.

    >>> hull = ConvexHull(points=generators,
    ...                   qhull_options='QG4')

    The "good" array indicates which facets are
    visible from point 4.

    >>> print(hull.simplices)
        [[1 0]
         [1 2]
         [3 0]
         [3 2]]
    >>> print(hull.good)
        [False  True False False]

    Now plot it, highlighting the visible facets.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1,1,1)
    >>> for visible_facet in hull.simplices[hull.good]:
    ...     ax.plot(hull.points[visible_facet, 0],
    ...             hull.points[visible_facet, 1],
    ...             color='violet',
    ...             lw=6)
    >>> convex_hull_plot_2d(hull, ax=ax)
        <Figure size 640x480 with 1 Axes> # may vary
    >>> plt.show()

    References
    ----------
    .. [Qhull] http://www.qhull.org/
    def __init__(self, points, incremental=False, qhull_options=None):
        # 检查输入的点集是否是掩码数组，如果是则引发异常
        if np.ma.isMaskedArray(points):
            raise ValueError('Input points cannot be a masked array')
        
        # 将输入的点集转换为连续存储的双精度浮点数数组
        points = np.ascontiguousarray(points, dtype=np.double)

        # 如果未提供 qhull_options，则设置为空字节串
        if qhull_options is None:
            qhull_options = b""
            # 如果点集的维度大于等于5，则添加 Qx 到 qhull_options 中
            if points.shape[1] >= 5:
                qhull_options += b"Qx"
        else:
            # 将 qhull_options 编码为 Latin-1 格式
            qhull_options = qhull_options.encode('latin1')

        # 运行 qhull 算法
        qhull = _Qhull(b"i", points, qhull_options, required_options=b"Qt",
                       incremental=incremental)
        
        # 调用父类 _QhullUser 的初始化方法
        _QhullUser.__init__(self, qhull, incremental=incremental)

    def _update(self, qhull):
        # 对 qhull 对象进行三角化处理
        qhull.triangulate()

        # 获取 qhull 对象中的简单xes, 邻居, 方程, 共面, 以及self.good
        self.simplices, self.neighbors, self.equations, self.coplanar, self.good = \
                       qhull.get_simplex_facet_array()

        # 如果 qhull.options 不为空，则创建一个选项集合并检查是否包含 QG 选项
        option_set = set()
        if qhull.options is not None:
            option_set.update(qhull.options.split())

        QG_option_present = 0
        for option in option_set:
            if b"QG" in option:
                QG_option_present += 1
                break

        # 如果没有找到 QG 选项，则将 self.good 设置为 None
        if not QG_option_present:
            self.good = None
        else:
            # 否则将 self.good 转换为布尔类型数组
            self.good = self.good.astype(bool)

        # 计算 qhull 对象的体积和面积
        self.volume, self.area = qhull.volume_area()

        # 如果 qhull 对象的维度为2，则获取二维极值点，否则设为 None
        if qhull.ndim == 2:
            self._vertices = qhull.get_extremes_2d()
        else:
            self._vertices = None

        # 设置 nsimplex 为 self.simplices 的行数
        self.nsimplex = self.simplices.shape[0]

        # 调用父类 _QhullUser 的 _update 方法
        _QhullUser._update(self, qhull)

    def add_points(self, points, restart=False):
        # 调用 _add_points 方法添加新的点集到当前对象中
        self._add_points(points, restart)

    @property
    def points(self):
        # 返回私有属性 _points，即当前对象的点集
        return self._points

    @property
    def vertices(self):
        # 如果 _vertices 为 None，则将其设为 simplices 的唯一值数组，并返回
        if self._vertices is None:
            self._vertices = np.unique(self.simplices)
        return self._vertices
# 将 ConvexHull.add_points 方法的文档字符串复制给 _QhullUser._add_points 方法
_copy_docstr(ConvexHull.add_points, _QhullUser._add_points)

#------------------------------------------------------------------------------
# Voronoi diagrams
#------------------------------------------------------------------------------

class Voronoi(_QhullUser):
    """
    Voronoi(points, furthest_site=False, incremental=False, qhull_options=None)

    Voronoi diagrams in N dimensions.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to construct a Voronoi diagram from
    furthest_site : bool, optional
        Whether to compute a furthest-site Voronoi diagram. Default: False
    incremental : bool, optional
        Allow adding new points incrementally. This takes up some additional
        resources.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qx" for ndim > 4 and
        "Qbb Qc Qz" otherwise. Incremental mode omits "Qz".)

    Attributes
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    vertices : ndarray of double, shape (nvertices, ndim)
        Coordinates of the Voronoi vertices.
    ridge_points : ndarray of ints, shape ``(nridges, 2)``
        Indices of the points between which each Voronoi ridge lies.
    ridge_vertices : list of list of ints, shape ``(nridges, *)``
        Indices of the Voronoi vertices forming each Voronoi ridge.
    regions : list of list of ints, shape ``(nregions, *)``
        Indices of the Voronoi vertices forming each Voronoi region.
        -1 indicates vertex outside the Voronoi diagram.
        When qhull option "Qz" was specified, an empty sublist
        represents the Voronoi region for a point at infinity that
        was added internally.
    point_region : array of ints, shape (npoints)
        Index of the Voronoi region for each input point.
        If qhull option "Qc" was not specified, the list will contain -1
        for points that are not associated with a Voronoi region.
        If qhull option "Qz" was specified, there will be one less
        element than the number of regions because an extra point
        at infinity is added internally to facilitate computation.
    furthest_site
        True if this was a furthest site triangulation and False if not.

        .. versionadded:: 1.4.0

    Raises
    ------
    QhullError
        Raised when Qhull encounters an error condition, such as
        geometrical degeneracy when options to resolve are not enabled.
    ValueError
        Raised if an incompatible array is given as input.

    Notes
    -----
    The Voronoi diagram is computed using the
    `Qhull library <http://www.qhull.org/>`__.

    Examples
    --------
    Voronoi diagram for a set of point:

    >>> import numpy as np
    >>> points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
    """
    初始化 Voronoi 图形对象，基于给定的点集。

    Parameters:
    - points: numpy 数组，包含 Voronoi 图形的顶点。
    - furthest_site: 布尔值，控制是否计算到所有点的最远的 Voronoi 图形。
    - incremental: 布尔值，控制是否增量计算。
    - qhull_options: 字符串，控制 Qhull 计算的选项。

    Raises:
    - ValueError: 如果输入的点集是一个掩码数组（masked array）。
                  如果输入的点集不是二维数组。

    Notes:
    - 如果未提供 qhull_options，则根据 incremental 的值设置默认选项。
    - 调用 _Qhull 类来运行 Qhull 算法计算 Voronoi 图形。

    """
    def __init__(self, points, furthest_site=False, incremental=False,
                 qhull_options=None):
        # 如果输入的 points 是掩码数组，则抛出错误
        if np.ma.isMaskedArray(points):
            raise ValueError('Input points cannot be a masked array')
        # 将 points 转换为双精度的连续数组
        points = np.ascontiguousarray(points, dtype=np.double)
        # 检查 points 是否为二维数组
        if points.ndim != 2:
            raise ValueError("Input points array must have 2 dimensions.")

        # 设置 Qhull 的选项
        if qhull_options is None:
            if not incremental:
                qhull_options = b"Qbb Qc Qz"
            else:
                qhull_options = b"Qc"
            if points.shape[1] >= 5:
                qhull_options += b" Qx"
        else:
            qhull_options = qhull_options.encode('latin1')

        # 运行 Qhull 算法
        qhull = _Qhull(b"v", points, qhull_options, furthest_site=furthest_site,
                       incremental=incremental)
        # 调用父类的初始化方法
        _QhullUser.__init__(self, qhull, incremental=incremental)

        # 设置对象的 furthest_site 属性
        self.furthest_site = furthest_site

    """
    更新 Voronoi 图形对象的数据。

    Parameters:
    - qhull: Qhull 对象，用于计算 Voronoi 图形的数据。

    Notes:
    - 更新对象的 vertices、ridge_points、ridge_vertices、regions 和 point_region 数据。
    - 清空对象的 _ridge_dict 缓存。

    """
    def _update(self, qhull):
        self.vertices, self.ridge_points, self.ridge_vertices, \
                       self.regions, self.point_region = \
                       qhull.get_voronoi_diagram()

        self._ridge_dict = None

        # 调用父类的 _update 方法
        _QhullUser._update(self, qhull)

    """
    向 Voronoi 图形对象添加新的点集。

    Parameters:
    - points: numpy 数组，要添加到 Voronoi 图形的新点。
    - restart: 布尔值，指示是否重新开始计算。

    """
    def add_points(self, points, restart=False):
        self._add_points(points, restart)

    """
    返回 Voronoi 图形对象的点集。

    Returns:
    - numpy 数组，包含 Voronoi 图形对象的点集。

    """
    @property
    def points(self):
        return self._points

    """
    返回 Voronoi 图形对象的 ridge 字典。

    Returns:
    - 字典，将 ridge_points 映射到 ridge_vertices。

    """
    @property
    def ridge_dict(self):
        # 如果 _ridge_dict 为空，则创建并缓存它
        if self._ridge_dict is None:
            self._ridge_dict = dict(zip(map(tuple, self.ridge_points.tolist()),
                                        self.ridge_vertices))
        return self._ridge_dict
# 将 Voronoi.add_points 的文档字符串复制给 _QhullUser._add_points
_copy_docstr(Voronoi.add_points, _QhullUser._add_points)

#------------------------------------------------------------------------------
# Halfspace Intersection
#------------------------------------------------------------------------------

# HalfspaceIntersection 类的定义，继承自 _QhullUser
class HalfspaceIntersection(_QhullUser):
    """
    HalfspaceIntersection(halfspaces, interior_point, incremental=False, qhull_options=None)

    Halfspace intersections in N dimensions.

    .. versionadded:: 0.19.0

    Parameters
    ----------
    halfspaces : ndarray of floats, shape (nineq, ndim+1)
        Stacked Inequalities of the form Ax + b <= 0 in format [A; b]
    interior_point : ndarray of floats, shape (ndim,)
        Point clearly inside the region defined by halfspaces. Also called a feasible
        point, it can be obtained by linear programming.
    incremental : bool, optional
        Allow adding new halfspaces incrementally. This takes up some additional
        resources.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qx" for ndim > 4 and "" otherwise)
        Option "H" is always enabled.

    Attributes
    ----------
    halfspaces : ndarray of double, shape (nineq, ndim+1)
        Input halfspaces.
    interior_point :ndarray of floats, shape (ndim,)
        Input interior point.
    intersections : ndarray of double, shape (ninter, ndim)
        Intersections of all halfspaces.
    dual_points : ndarray of double, shape (nineq, ndim)
        Dual points of the input halfspaces.
    dual_facets : list of lists of ints
        Indices of points forming the (non necessarily simplicial) facets of
        the dual convex hull.
    dual_vertices : ndarray of ints, shape (nvertices,)
        Indices of halfspaces forming the vertices of the dual convex hull.
        For 2-D convex hulls, the vertices are in counterclockwise order.
        For other dimensions, they are in input order.
    dual_equations : ndarray of double, shape (nfacet, ndim+1)
        [normal, offset] forming the hyperplane equation of the dual facet
        (see `Qhull documentation <http://www.qhull.org/>`__  for more).
    dual_area : float
        Area of the dual convex hull
    dual_volume : float
        Volume of the dual convex hull

    Raises
    ------
    QhullError
        Raised when Qhull encounters an error condition, such as
        geometrical degeneracy when options to resolve are not enabled.
    ValueError
        Raised if an incompatible array is given as input.

    Notes
    -----
    The intersections are computed using the
    `Qhull library <http://www.qhull.org/>`__.
    This reproduces the "qhalf" functionality of Qhull.

    Examples
    --------

    Halfspace intersection of planes forming some polygon

    >>> from scipy.spatial import HalfspaceIntersection
    >>> import numpy as np
    >>> halfspaces = np.array([[-1, 0., 0.],
    ...                        [0., -1., 0.],
    # Define a set of halfspaces represented as inequalities in the form Ax + b <= 0
    halfspaces = np.array([
        [ 1.,  1.,  1.],
        [ 2.,  1., -4.],
        [-0.5,  1., -2.]])
    # Define a feasible interior point within the convex polyhedron
    feasible_point = np.array([0.5, 0.5])
    # Compute the intersection points of the halfspaces at the feasible point
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    
    # Import matplotlib for plotting
    import matplotlib.pyplot as plt
    # Create a new figure and axis for the plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    # Set the limits for the x and y axes
    xlim, ylim = (-1, 3), (-1, 3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Generate a set of x values for plotting
    x = np.linspace(-1, 3, 100)
    # Define symbols and signs for plotting halfspaces
    symbols = ['-', '+', 'x', '*']
    signs = [0, 0, -1, -1]
    # Define the format for filling the regions between halfspaces
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
    
    # Iterate over each halfspace to plot and fill the corresponding regions
    for h, sym, sign in zip(halfspaces, symbols, signs):
        hlist = h.tolist()
        fmt["hatch"] = sym
        if h[1] == 0:
            # Plot a vertical line for the halfspace
            ax.axvline(-h[2] / h[0], label='{}x+{}y+{}=0'.format(*hlist))
            xi = np.linspace(xlim[sign], -h[2] / h[0], 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            # Plot the boundary line for the halfspace
            ax.plot(x, (-h[2] - h[0] * x) / h[1], label='{}x+{}y+{}=0'.format(*hlist))
            ax.fill_between(x, (-h[2] - h[0] * x) / h[1], ylim[sign], **fmt)
    
    # Plot the intersection points of the halfspaces
    x, y = zip(*hs.intersections)
    ax.plot(x, y, 'o', markersize=8)
    
    # Solve for the Chebyshev center using linear programming
    from scipy.optimize import linprog
    from matplotlib.patches import Circle
    
    # Compute the norms of the normal vectors of each halfspace
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    # Define the objective function coefficients for linear programming
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    # Formulate the constraints for linear programming
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    # Solve the linear program
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    x = res.x[:-1]
    y = res.x[-1]
    # Create a circle patch representing the largest inscribed hypersphere
    circle = Circle(x, radius=y, alpha=0.3)
    ax.add_patch(circle)
    
    # Display the legend and show the plot
    plt.legend(bbox_to_anchor=(1.6, 1.0))
    plt.show()
    
    # References for further reading and context about the Qhull library and convex optimization
    """
    References
    ----------
    .. [Qhull] http://www.qhull.org/
    .. [1] S. Boyd, L. Vandenberghe, Convex Optimization, available
           at http://stanford.edu/~boyd/cvxbook/
    """
    def __init__(self, halfspaces, interior_point,
                    incremental=False, qhull_options=None):
        # 检查输入的半空间是否为掩码数组，如果是则引发异常
        if np.ma.isMaskedArray(halfspaces):
            raise ValueError('Input halfspaces cannot be a masked array')
        # 检查输入的内部点是否为掩码数组，如果是则引发异常
        if np.ma.isMaskedArray(interior_point):
            raise ValueError('Input interior point cannot be a masked array')
        # 检查内部点的形状是否符合要求，即为 (ndim-1,) 形状
        if interior_point.shape != (halfspaces.shape[1]-1,):
            raise ValueError('Feasible point must be a (ndim-1,) array')
        # 将半空间数组转换为双精度的连续数组
        halfspaces = np.ascontiguousarray(halfspaces, dtype=np.double)
        # 将内部点数组转换为双精度的连续数组
        self.interior_point = np.ascontiguousarray(interior_point, dtype=np.double)

        # 如果未指定 Qhull 选项，则设置为空字节串
        if qhull_options is None:
            qhull_options = b""
            # 如果半空间数组的列数大于等于 6，则添加 Qhull 选项 "Qx"
            if halfspaces.shape[1] >= 6:
                qhull_options += b"Qx"
        else:
            # 将 Qhull 选项编码为 Latin-1 格式
            qhull_options = qhull_options.encode('latin1')

        # 运行 Qhull
        mode_option = "H"
        qhull = _Qhull(mode_option.encode(), halfspaces, qhull_options,
                       required_options=None, incremental=incremental,
                       interior_point=self.interior_point)

        # 调用父类 _QhullUser 的初始化方法
        _QhullUser.__init__(self, qhull, incremental=incremental)

    def _update(self, qhull):
        # 获取凸壳的双重面和双重方程组
        self.dual_facets, self.dual_equations = qhull.get_hull_facets()

        # 获取凸壳的双重点
        self.dual_points = qhull.get_hull_points()

        # 计算凸壳的体积和面积
        self.dual_volume, self.dual_area = qhull.volume_area()

        # 计算交点，使用内部点对双重方程进行处理
        self.intersections = self.dual_equations[:, :-1] / -self.dual_equations[:, -1:] + self.interior_point

        # 如果 Qhull 对象的维度为 2，则获取其在平面上的极值点
        if qhull.ndim == 2:
            self._vertices = qhull.get_extremes_2d()
        else:
            self._vertices = None

        # 调用父类 _QhullUser 的更新方法
        _QhullUser._update(self, qhull)

        # 设置对象的维度和不等式的数量
        self.ndim = self.halfspaces.shape[1] - 1
        self.nineq = self.halfspaces.shape[0]

    def add_halfspaces(self, halfspaces, restart=False):
        """
        add_halfspaces(halfspaces, restart=False)

        Process a set of additional new halfspaces.

        Parameters
        ----------
        halfspaces : ndarray
            New halfspaces to add. The dimensionality should match that of the
            initial halfspaces.
        restart : bool, optional
            Whether to restart processing from scratch, rather than
            adding halfspaces incrementally.

        Raises
        ------
        QhullError
            Raised when Qhull encounters an error condition, such as
            geometrical degeneracy when options to resolve are not enabled.

        See Also
        --------
        close

        Notes
        -----
        You need to specify ``incremental=True`` when constructing the
        object to be able to add halfspaces incrementally. Incremental addition
        of halfspaces is also not possible after `close` has been called.

        """
        # 调用内部方法 _add_points 处理新增的半空间
        self._add_points(halfspaces, restart, self.interior_point)

    @property
    def halfspaces(self):
        # 返回对象的半空间属性，即 _points
        return self._points

    @property
    # 定义一个方法 dual_vertices，用于获取对偶面的顶点集合
    def dual_vertices(self):
        # 如果当前对象的 _vertices 属性为空
        if self._vertices is None:
            # 将对象的 dual_facets 属性转换为 NumPy 数组，并去除重复元素，得到唯一的顶点集合
            self._vertices = np.unique(np.array(self.dual_facets))
        # 返回当前对象的 _vertices 属性，即对偶面的顶点集合
        return self._vertices
```