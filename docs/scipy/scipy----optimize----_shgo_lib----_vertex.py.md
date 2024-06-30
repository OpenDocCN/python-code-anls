# `D:\src\scipysrc\scipy\scipy\optimize\_shgo_lib\_vertex.py`

```
import collections
from abc import ABC, abstractmethod  # 导入 ABC 类和抽象方法装饰器

import numpy as np  # 导入 NumPy 库

from scipy._lib._util import MapWrapper  # 导入 MapWrapper 类


class VertexBase(ABC):
    """
    Base class for a vertex.
    """
    def __init__(self, x, nn=None, index=None):
        """
        Initiation of a vertex object.

        Parameters
        ----------
        x : tuple or vector
            The geometric location (domain).
        nn : list, optional
            Nearest neighbour list.
        index : int, optional
            Index of vertex.
        """
        self.x = x  # 设置顶点的几何位置 x
        self.hash = hash(self.x)  # 计算并保存顶点位置 x 的哈希值

        if nn is not None:
            self.nn = set(nn)  # 如果提供了最近邻列表 nn，则初始化为集合
        else:
            self.nn = set()  # 否则初始化为空集合

        self.index = index  # 设置顶点的索引号

    def __hash__(self):
        return self.hash  # 返回顶点对象的哈希值

    def __getattr__(self, item):
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute "
                                 f"'{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x)  # 如果请求属性为 'x_a'，则将顶点位置 x 转换为 NumPy 数组并返回
            return self.x_a

    @abstractmethod
    def connect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    def star(self):
        """Returns the star domain ``st(v)`` of the vertex.

        Parameters
        ----------
        v :
            The vertex ``v`` in ``st(v)``

        Returns
        -------
        st : set
            A set containing all the vertices in ``st(v)``
        """
        self.st = self.nn  # 将顶点的最近邻列表作为星形域 st(v) 的初始值
        self.st.add(self)  # 将顶点自身也加入星形域 st(v)
        return self.st


class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """
    def __init__(self, x, field=None, nn=None, index=None, field_args=(),
                 g_cons=None, g_cons_args=()):
        """
        Parameters
        ----------
        x : tuple,
            vector of vertex coordinates
        field : callable, optional
            a scalar field f: R^n --> R associated with the geometry
        nn : list, optional
            list of nearest neighbours
        index : int, optional
            index of the vertex
        field_args : tuple, optional
            additional arguments to be passed to field
        g_cons : callable, optional
            constraints on the vertex
        g_cons_args : tuple, optional
            additional arguments to be passed to g_cons

        """
        # 调用父类初始化方法，传递坐标、邻居列表和索引
        super().__init__(x, nn=nn, index=index)

        # 设定检查最小值标志，初始值为 True
        self.check_min = True
        # 设定检查最大值标志，初始值为 True
        self.check_max = True

    def connect(self, v):
        """Connects self to another vertex object v.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        # 如果 v 不是 self 且 v 不在 self 的邻居列表中，则添加 v 到邻居列表
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            # 重置检查最小值和最大值标志
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        # 如果 v 在邻居列表中，则从邻居列表中移除 v
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

            # 重置检查最小值和最大值标志
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its
           neighbours"""
        # 如果需要检查最小值，则计算当前顶点是否小于所有邻居顶点
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

    def maximiser(self):
        """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
        # 如果需要检查最大值，则计算当前顶点是否大于所有邻居顶点
        if self.check_max:
            self._max = all(self.f > v.f for v in self.nn)
            self.check_max = False

        return self._max
class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """

    def __init__(self, x, sfield=None, vfield=None, field_args=(),
                 vfield_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

        # 抛出未实现错误，表明该类仍在开发中
        raise NotImplementedError("This class is still a work in progress")


class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""
    def __init__(self):

        # 创建一个有序字典作为顶点缓存
        self.cache = collections.OrderedDict()
        # 初始化可行点计数器为0
        self.nfev = 0  # Feasible points
        # 初始化索引为-1
        self.index = -1

    def __iter__(self):
        # 迭代器方法，返回顶点缓存中每个顶点的值
        for v in self.cache:
            yield self.cache[v]
        return

    def size(self):
        """Returns the size of the vertex cache."""
        # 返回顶点缓存的大小（索引+1）
        return self.index + 1

    def print_out(self):
        # 打印顶点缓存的内容
        headlen = len(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        print(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()


class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        # 连接当前顶点和另一个顶点
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        # 断开当前顶点和另一个顶点的连接
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)


class VertexCacheIndex(VertexCacheBase):
    def __init__(self):
        """
        Class for a vertex cache for a simplicial complex without an associated
        field. Useful only for building and visualising a domain complex.

        Parameters
        ----------
        """
        super().__init__()
        # 设置顶点类型为VertexCube
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):
        try:
            # 尝试获取顶点缓存中的顶点
            return self.cache[x]
        except KeyError:
            # 如果顶点不存在，创建新的顶点并添加到缓存中
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging
            # is commented out
            # 将新生成的顶点添加到缓存中
            self.cache[x] = xval
            return self.cache[x]


class VertexCacheField(VertexCacheBase):
    # 这个类目前是空的，可能用于将来添加顶点场相关的功能
    # 初始化方法，用于设置顶点缓存和相关字段的计算
    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(),
                 workers=1):
        """
        Class for a vertex cache for a simplicial complex with an associated
        field.

        Parameters
        ----------
        field : callable
            Scalar or vector field callable.
        field_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            field function
        g_cons : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::
        g_cons_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            constraint functions
        workers : int  optional
            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
             functions in parallel.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 初始化索引为-1，用于顶点缓存中的索引
        self.index = -1
        # 设置顶点的标量场类型，默认为VertexScalarField
        self.Vertex = VertexScalarField
        # 设置字段计算函数
        self.field = field
        # 设置字段计算函数的额外参数
        self.field_args = field_args
        # 如果workers不为1，则使用FieldWrapper封装字段计算函数
        self.wfield = FieldWrapper(field, field_args)  # if workers is not 1

        # 设置约束条件
        self.g_cons = g_cons
        # 设置约束条件函数的额外参数
        self.g_cons_args = g_cons_args
        # 如果有约束条件，则使用ConstraintWrapper封装约束条件
        self.wgcons = ConstraintWrapper(g_cons, g_cons_args)
        # 初始化gpool为空集合，用于存储待处理的顶点
        self.gpool = set()  # A set of tuples to process for feasibility

        # Field processing objects
        # 初始化fpool为空集合，用于存储待处理的顶点
        self.fpool = set()  # A set of tuples to process for scalar function
        # 初始化sfc_lock为False，用于指示fpool是否非空
        self.sfc_lock = False  # True if self.fpool is non-Empty

        # 设置工作进程数
        self.workers = workers
        # 初始化_mapwrapper为MapWrapper对象，用于并行处理
        self._mapwrapper = MapWrapper(workers)

        # 根据工作进程数设置不同的处理方法
        if workers == 1:
            # 如果只有一个工作进程，使用proc_gpool处理gpool
            self.process_gpool = self.proc_gpool
            # 如果没有约束条件，则使用proc_fpool_nog处理fpool
            if g_cons is None:
                self.process_fpool = self.proc_fpool_nog
            else:
                # 否则使用proc_fpool_g处理fpool
                self.process_fpool = self.proc_fpool_g
        else:
            # 如果有多个工作进程，使用pproc_gpool处理gpool
            self.process_gpool = self.pproc_gpool
            # 如果没有约束条件，则使用pproc_fpool_nog处理fpool
            if g_cons is None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                # 否则使用pproc_fpool_g处理fpool
                self.process_fpool = self.pproc_fpool_g

    # 获取指定顶点的缓存值，如果不存在则创建新值
    def __getitem__(self, x, nn=None):
        try:
            # 尝试从缓存中返回顶点的值
            return self.cache[x]
        except KeyError:
            # 如果顶点不存在于缓存中，增加索引值
            self.index += 1
            # 创建新的顶点对象xval，并存储在缓存中
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index,
                               field_args=self.field_args,
                               g_cons=self.g_cons,
                               g_cons_args=self.g_cons_args)

            self.cache[x] = xval  # Define in cache
            self.gpool.add(xval)  # Add to pool for processing feasibility
            self.fpool.add(xval)  # Add to pool for processing field values
            return self.cache[x]

    # 序列化对象时调用，返回对象的状态信息
    def __getstate__(self):
        # 复制对象的__dict__属性
        self_dict = self.__dict__.copy()
        # 删除self_dict中的'pool'键
        del self_dict['pool']
        return self_dict

    # 处理顶点池中的任务，调用process_gpool和process_fpool方法
    def process_pools(self):
        if self.g_cons is not None:
            # 如果存在约束条件，则处理gpool
            self.process_gpool()
        # 处理fpool
        self.process_fpool()
        # 处理最小化程序
        self.proc_minimisers()
    def feasibility_check(self, v):
        # 设置可行性标志为 True
        v.feasible = True
        # 遍历所有约束条件
        for g, args in zip(self.g_cons, self.g_cons_args):
            # 约束条件可能返回多个值。
            # 如果任何一个约束条件小于 0，将顶点对象的目标函数值设置为无穷大，可行性标志设置为 False，并退出循环。
            if np.any(g(v.x_a, *args) < 0.0):
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        """计算顶点对象 `v` 的标量场值。

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
            顶点对象，可以是基类 VertexBase 或者标量场对象 VertexScalarField。
        """
        try:
            # 计算顶点对象的标量场值
            v.f = self.field(v.x_a, *self.field_args)
            # 增加函数评估次数计数器
            self.nfev += 1
        except AttributeError:
            # 如果属性错误，则将顶点对象的目标函数值设置为无穷大
            v.f = np.inf
            # 记录警告日志，说明在 x = self.x_a 处找不到字段函数
            # logging.warning(f"Field function not found at x = {self.x_a}")
        # 如果计算得到的标量场值为 NaN，则将其设置为无穷大
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        """处理所有约束条件。"""
        if self.g_cons is not None:
            # 遍历约束条件池中的所有顶点对象，并对其进行可行性检查
            for v in self.gpool:
                self.feasibility_check(v)
        # 清空约束条件池
        self.gpool = set()

    def pproc_gpool(self):
        """
        并行处理所有约束条件。
        """
        gpool_l = []
        # 构建约束条件池的顶点对象列表
        for v in self.gpool:
            gpool_l.append(v.x_a)

        # 使用并行映射方法调用约束条件函数
        G = self._mapwrapper(self.wgcons.gcons, gpool_l)
        # 将映射结果设置回顶点对象的可行性标志 v.feasible
        for v, g in zip(self.gpool, G):
            v.feasible = g  # 设置顶点对象属性 v.feasible = g（布尔值）

    def proc_fpool_g(self):
        """
        处理所有带有提供约束条件的字段函数。
        """
        for v in self.fpool:
            # 如果顶点对象的可行性标志为 True，则计算其标量场值
            if v.feasible:
                self.compute_sfield(v)
        # 清空字段函数池
        self.fpool = set()

    def proc_fpool_nog(self):
        """
        处理所有不带提供约束条件的字段函数。
        """
        for v in self.fpool:
            # 计算顶点对象的标量场值
            self.compute_sfield(v)
        # 清空字段函数池
        self.fpool = set()

    def pproc_fpool_g(self):
        """
        并行处理所有带有提供约束条件的字段函数。
        """
        self.wfield.func
        fpool_l = []
        # 构建字段函数池的顶点对象列表
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf

        # 使用并行映射方法调用字段函数
        F = self._mapwrapper(self.wfield.func, fpool_l)
        # 将映射结果设置回顶点对象的目标函数值 v.f
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # 设置顶点对象属性 v.f = f
            self.nfev += 1
        # 清空字段函数池
        self.fpool = set()

    def pproc_fpool_nog(self):
        """
        并行处理所有不带提供约束条件的字段函数。
        """
        self.wfield.func
        fpool_l = []
        # 构建字段函数池的顶点对象列表
        for v in self.fpool:
            fpool_l.append(v.x_a)

        # 使用并行映射方法调用字段函数
        F = self._mapwrapper(self.wfield.func, fpool_l)
        # 将映射结果设置回顶点对象的目标函数值 v.f
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # 设置顶点对象属性 v.f = f
            self.nfev += 1
        # 清空字段函数池
        self.fpool = set()
    # 定义一个方法 proc_minimisers，用于处理最小化器
    def proc_minimisers(self):
        """Check for minimisers."""
        # 对于对象自身的每个元素进行循环遍历
        for v in self:
            # 调用元素的 minimiser 方法，执行最小化操作
            v.minimiser()
            # 调用元素的 maximiser 方法，执行最大化操作
            v.maximiser()
class ConstraintWrapper:
    """封装约束条件以传递给 `multiprocessing.Pool` 的对象。"""
    
    def __init__(self, g_cons, g_cons_args):
        # 初始化方法，接受约束条件函数列表和参数列表
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        # 对给定的输入 v_x_a 检查所有约束条件是否满足
        vfeasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            # 约束条件可能返回多个值。
            if np.any(g(v_x_a, *args) < 0.0):
                vfeasible = False
                break
        return vfeasible


class FieldWrapper:
    """封装场域函数以传递给 `multiprocessing.Pool` 的对象。"""
    
    def __init__(self, field, field_args):
        # 初始化方法，接受场域函数和参数列表
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        # 对给定的输入 v_x_a 计算场域函数值
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except Exception:
            # 如果计算出现异常，则返回无穷大
            v_f = np.inf
        if np.isnan(v_f):
            # 如果计算结果是 NaN，则也返回无穷大
            v_f = np.inf

        return v_f
```