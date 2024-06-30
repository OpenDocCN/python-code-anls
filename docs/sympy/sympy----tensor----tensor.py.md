# `D:\src\scipysrc\sympy\sympy\tensor\tensor.py`

```
# 引入必要的模块和函数，包括类型注解、reduce 函数、prod 函数等
from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod

# 引入抽象基类模块和默认字典模块
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools

# 从 sympy 库中引入不同的类和函数
from sympy.core.numbers import (Integer, Rational)  # 导入整数和有理数类
from sympy.combinatorics import Permutation  # 导入排列组合类
from sympy.combinatorics.tensor_can import (  # 导入张量规范化相关函数
    get_symmetric_group_sgs,
    bsgs_direct_product,
    canonicalize,
    riemann_bsgs,
)
from sympy.core import Basic, Expr, sympify, Add, Mul, S  # 导入核心类
from sympy.core.cache import clear_cache  # 导入缓存清除函数
from sympy.core.containers import Tuple, Dict  # 导入元组和字典类
from sympy.core.function import WildFunction  # 导入野函数类
from sympy.core.sorting import default_sort_key  # 导入默认排序函数
from sympy.core.symbol import Symbol, symbols, Wild  # 导入符号类和符号创建函数
from sympy.core.sympify import CantSympify, _sympify  # 导入 sympify 相关函数
from sympy.core.operations import AssocOp  # 导入关联操作类
from sympy.external.gmpy import SYMPY_INTS  # 导入 gmpy 整数模块
from sympy.matrices import eye  # 导入单位矩阵函数
from sympy.utilities.exceptions import (  # 导入异常处理相关函数
    sympy_deprecation_warning,
    SymPyDeprecationWarning,
    ignore_warnings,
)
from sympy.utilities.decorator import memoize_property, deprecated  # 导入装饰器函数
from sympy.utilities.iterables import sift  # 导入筛选函数

# 函数用于发出关于数据属性的警告信息
def deprecate_data():
    sympy_deprecation_warning(
        """
        The data attribute of TensorIndexType is deprecated. Use The
        replace_with_arrays() method instead.
        """,
        deprecated_since_version="1.4",
        active_deprecations_target="deprecated-tensorindextype-attrs",
        stacklevel=4,
    )

# 函数用于发出关于函数评估的警告信息
def deprecate_fun_eval():
    pass  # 这里的函数体还没有提供，稍后可能会添加实现
    # 发出 SymPy 废弃警告消息，提醒用户 Tensor.fun_eval() 方法已被废弃，建议使用 Tensor.substitute_indices() 方法替代。
    sympy_deprecation_warning(
        """
        The Tensor.fun_eval() method is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        # 指定自版本 1.5 起废弃
        deprecated_since_version="1.5",
        # 指定当前废弃项的活跃警告目标为 "deprecated-tensor-fun-eval"
        active_deprecations_target="deprecated-tensor-fun-eval",
        # 指定警告发出时的堆栈层级为 4
        stacklevel=4,
    )
# 定义一个函数用于发出 Sympy 弃用警告
def deprecate_call():
    # 调用 sympy_deprecation_warning 函数，传入警告消息和相关参数
    sympy_deprecation_warning(
        """
        Calling a tensor like Tensor(*indices) is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        deprecated_since_version="1.5",  # 标明自版本 1.5 起被弃用
        active_deprecations_target="deprecated-tensor-fun-eval",  # 标明弃用目标为 deprecated-tensor-fun-eval
        stacklevel=4,  # 设置警告栈级别为 4
    )


class _IndexStructure(CantSympify):
    """
    This class handles the indices (free and dummy ones). It contains the
    algorithms to manage the dummy indices replacements and contractions of
    free indices under multiplications of tensor expressions, as well as stuff
    related to canonicalization sorting, getting the permutation of the
    expression and so on. It also includes tools to get the ``TensorIndex``
    objects corresponding to the given index structure.
    """

    def __init__(self, free, dum, index_types, indices, canon_bp=False):
        # 初始化 _IndexStructure 实例的属性
        self.free = free  # 自由指标列表
        self.dum = dum  # 哑指标列表
        self.index_types = index_types  # 指标类型列表
        self.indices = indices  # 指标列表
        self._ext_rank = len(self.free) + 2*len(self.dum)  # 扩展秩的计算公式
        self.dum.sort(key=lambda x: x[0])  # 根据第一个元素排序哑指标列表

    @staticmethod
    def from_indices(*indices):
        """
        Create a new ``_IndexStructure`` object from a list of ``indices``.

        Explanation
        ===========

        ``indices``     ``TensorIndex`` objects, the indices. Contractions are
                        detected upon construction.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure.from_indices(m0, m1, -m1, m3)
        _IndexStructure([(m0, 0), (m3, 3)], [(1, 2)], [Lorentz, Lorentz, Lorentz, Lorentz])
        """

        # 从指标列表创建新的 _IndexStructure 对象
        free, dum = _IndexStructure._free_dum_from_indices(*indices)  # 从指标列表中提取自由和哑指标
        index_types = [i.tensor_index_type for i in indices]  # 获取每个指标的类型
        indices = _IndexStructure._replace_dummy_names(indices, free, dum)  # 替换哑指标的名称
        return _IndexStructure(free, dum, index_types, indices)  # 返回创建的 _IndexStructure 对象

    @staticmethod
    def from_components_free_dum(components, free, dum):
        # 从组件、自由指标和哑指标创建 _IndexStructure 对象
        index_types = []
        for component in components:
            index_types.extend(component.index_types)
        indices = _IndexStructure.generate_indices_from_free_dum_index_types(free, dum, index_types)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def _free_dum_from_indices(*indices):
        """
        Convert ``indices`` into ``free``, ``dum`` for single component tensor.

        Explanation
        ===========

        ``free``     list of tuples ``(index, pos, 0)``,
                     where ``pos`` is the position of index in
                     the list of indices formed by the component tensors

        ``dum``      list of tuples ``(pos_contr, pos_cov, 0, 0)``

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, \
            _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure._free_dum_from_indices(m0, m1, -m1, m3)
        ([(m0, 0), (m3, 3)], [(1, 2)])
        """
        # 计算输入索引的数量
        n = len(indices)
        # 如果只有一个索引，返回该索引作为自由索引，没有哑指标
        if n == 1:
            return [(indices[0], 0)], []

        # 找到自由索引和哑指标的位置
        free = [True]*len(indices)  # 创建一个布尔列表，标记所有索引为自由
        index_dict = {}  # 用于跟踪每个索引名称和类型的字典
        dum = []  # 哑指标列表
        for i, index in enumerate(indices):
            name = index.name
            typ = index.tensor_index_type
            contr = index.is_up
            if (name, typ) in index_dict:
                # 找到一对哑指标
                is_contr, pos = index_dict[(name, typ)]
                # 检查一致性并更新自由列表
                if is_contr:
                    if contr:
                        raise ValueError('two equal contravariant indices in slots %d and %d' %(pos, i))
                    else:
                        free[pos] = False
                        free[i] = False
                else:
                    if contr:
                        free[pos] = False
                        free[i] = False
                    else:
                        raise ValueError('two equal covariant indices in slots %d and %d' %(pos, i))
                if contr:
                    dum.append((i, pos))
                else:
                    dum.append((pos, i))
            else:
                index_dict[(name, typ)] = index.is_up, i

        # 收集剩余的自由索引
        free = [(index, i) for i, index in enumerate(indices) if free[i]]
        free.sort()
        return free, dum

    def get_indices(self):
        """
        Get a list of indices, creating new tensor indices to complete dummy indices.
        """
        return self.indices[:]  # 返回对象中的索引列表的副本

    @staticmethod
    def generate_indices_from_free_dum_index_types(free, dum, index_types):
        # 初始化一个长度为 free + 2*dum 的列表，用于存放索引
        indices = [None]*(len(free)+2*len(dum))
        # 遍历自由索引，将索引位置与索引值对应存入 indices 列表
        for idx, pos in free:
            indices[pos] = idx

        # 获取生成虚拟索引名称的函数
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        # 遍历虚拟索引对，为每对虚拟索引生成 TensorIndex 对象，并放入 indices 列表
        for pos1, pos2 in dum:
            typ1 = index_types[pos1]
            indname = generate_dummy_name(typ1)
            indices[pos1] = TensorIndex(indname, typ1, True)
            indices[pos2] = TensorIndex(indname, typ1, False)

        # 返回替换虚拟索引名称后的 indices 列表
        return _IndexStructure._replace_dummy_names(indices, free, dum)

    @staticmethod
    def _get_generator_for_dummy_indices(free):
        # 使用 defaultdict 初始化计数器 cdt
        cdt = defaultdict(int)
        # 如果自由索引中的索引名称包含 dummy_name，则从比虚拟索引更高的索引值开始计数
        # 以避免名称冲突
        for indx, ipos in free:
            if indx.name.split('_')[0] == indx.tensor_index_type.dummy_name:
                cdt[indx.tensor_index_type] = max(cdt[indx.tensor_index_type], int(indx.name.split('_')[1]) + 1)

        # 定义生成虚拟索引名称的函数
        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        # 返回生成虚拟索引名称的函数
        return dummy_name_gen

    @staticmethod
    def _replace_dummy_names(indices, free, dum):
        # 根据第一个位置对虚拟索引对 dum 进行排序
        dum.sort(key=lambda x: x[0])
        # 复制 indices 列表，存入 new_indices
        new_indices = list(indices)
        # 断言 indices 列表的长度与 free + 2*dum 相等
        assert len(indices) == len(free) + 2*len(dum)
        # 获取生成虚拟索引名称的函数
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        # 遍历虚拟索引对 dum，为每对虚拟索引生成 TensorIndex 对象，并放入 new_indices 列表
        for ipos1, ipos2 in dum:
            typ1 = new_indices[ipos1].tensor_index_type
            indname = generate_dummy_name(typ1)
            new_indices[ipos1] = TensorIndex(indname, typ1, True)
            new_indices[ipos2] = TensorIndex(indname, typ1, False)
        # 返回替换虚拟索引名称后的 new_indices 列表
        return new_indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices.
        """
        # 根据索引位置对 self.free 列表进行排序，获取自由索引列表
        free = sorted(self.free, key=lambda x: x[1])
        # 返回自由索引列表中的索引部分
        return [i[0] for i in free]

    def __str__(self):
        # 返回一个描述性字符串，包含 self.free、self.dum 和 self.index_types 的信息
        return "_IndexStructure({}, {}, {})".format(self.free, self.dum, self.index_types)

    def __repr__(self):
        # 返回 __str__ 方法的结果
        return self.__str__()

    def _get_sorted_free_indices_for_canon(self):
        # 复制 self.free 列表，按照第一个元素排序，并返回排序后的列表
        sorted_free = self.free[:]
        sorted_free.sort(key=lambda x: x[0])
        return sorted_free

    def _get_sorted_dum_indices_for_canon(self):
        # 根据第一个位置对 self.dum 列表进行排序，并返回排序后的列表
        return sorted(self.dum, key=lambda x: x[0])

    def _get_lexicographically_sorted_index_types(self):
        # 获取 indices_canon_args 方法的返回结果中的排列顺序
        permutation = self.indices_canon_args()[0]
        # 初始化长度为 self._ext_rank 的 index_types 列表
        index_types = [None]*self._ext_rank
        # 遍历 self.index_types，按照排列顺序将其放入 index_types 列表
        for i, it in enumerate(self.index_types):
            index_types[permutation(i)] = it
        # 返回排列后的 index_types 列表
        return index_types
    # 获取按词典顺序排序的索引数组
    def _get_lexicographically_sorted_indices(self):
        # 调用 indices_canon_args 方法获取排列，取其第一个元素作为排列顺序
        permutation = self.indices_canon_args()[0]
        # 初始化一个长度为 self._ext_rank 的 None 列表作为结果容器
        indices = [None]*self._ext_rank
        # 遍历 self.indices 列表
        for i, it in enumerate(self.indices):
            # 使用排列 permutation 对 indices 进行重排
            indices[permutation(i)] = it
        # 返回排列后的 indices 结果
        return indices

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns a ``_IndexStructure`` instance corresponding to the permutation ``g``.

        Explanation
        ===========

        ``g``  permutation corresponding to the tensor in the representation
        used in canonicalization

        ``is_canon_bp``   if True, then ``g`` is the permutation
        corresponding to the canonical form of the tensor
        """
        # 获取按词典顺序排列的自由索引列表
        sorted_free = [i[0] for i in self._get_sorted_free_indices_for_canon()]
        # 获取按词典顺序排序的索引类型列表
        lex_index_types = self._get_lexicographically_sorted_index_types()
        # 获取按词典顺序排序的索引列表
        lex_indices = self._get_lexicographically_sorted_indices()
        # 计算自由索引的数量
        nfree = len(sorted_free)
        # 获取张量的秩
        rank = self._ext_rank
        # 初始化用于存放配对的虚指标的列表
        dum = [[None]*2 for i in range((rank - nfree)//2)]
        # 初始化自由索引列表
        free = []

        # 初始化索引类型列表和索引列表
        index_types = [None]*rank
        indices = [None]*rank
        # 遍历张量的每一个维度
        for i in range(rank):
            # 获取 g 中第 i 个位置的索引
            gi = g[i]
            # 根据 gi 获取对应的索引类型
            index_types[i] = lex_index_types[gi]
            # 根据 gi 获取对应的索引值
            indices[i] = lex_indices[gi]
            # 如果 gi 小于 nfree，说明是自由指标
            if gi < nfree:
                # 获取对应的自由索引
                ind = sorted_free[gi]
                # 断言索引类型与自由索引的张量索引类型相符
                assert index_types[i] == sorted_free[gi].tensor_index_type
                # 将自由索引及其位置添加到 free 列表中
                free.append((ind, i))
            else:
                # 计算虚指标的序号
                j = gi - nfree
                # 计算虚指标的分量号和协变/逆变性
                idum, cov = divmod(j, 2)
                # 根据协变/逆变性将虚指标添加到 dum 列表中的对应位置
                if cov:
                    dum[idum][1] = i
                else:
                    dum[idum][0] = i
        # 将 dum 列表中的列表转换为元组列表
        dum = [tuple(x) for x in dum]

        # 返回一个 _IndexStructure 实例，其中包含自由指标、虚指标、索引类型和索引
        return _IndexStructure(free, dum, index_types, indices)
    # 返回 ``(g, dummies, msym, v)``，即 ``canonicalize`` 的返回结果的各个部分
    def indices_canon_args(self):
        """
        Returns ``(g, dummies, msym, v)``, the entries of ``canonicalize``

        See ``canonicalize`` in ``tensor_can.py`` in combinatorics module.
        """
        # to be called after sorted_components
        from sympy.combinatorics.permutations import _af_new
        # 获取外部秩
        n = self._ext_rank
        # 初始化 g 列表，长度为 n+2，前 n 个元素为 None，最后两个元素为 n 和 n+1
        g = [None]*n + [n, n+1]

        # 将度规的对称性转换为 msym，调用 combinatorics 模块中的 .canonicalize() 方法
        def metric_symmetry_to_msym(metric):
            if metric is None:
                return None
            # 获取度规的对称性
            sym = metric.symmetry
            if sym == TensorSymmetry.fully_symmetric(2):
                return 0
            if sym == TensorSymmetry.fully_symmetric(-2):
                return 1
            return None

        # 有序的索引：首先是自由指标，按类型排序，然后是哑指标，按类型排序，逆变在前，协变在后
        # g[position in tensor] = position in ordered indices
        for i, (indx, ipos) in enumerate(self._get_sorted_free_indices_for_canon()):
            g[ipos] = i
        pos = len(self.free)
        j = len(self.free)
        dummies = []  # 存储哑指标的列表
        prev = None
        a = []  # 临时存储每个类型的哑指标的位置
        msym = []  # 存储度规对称性的列表
        # 遍历排序后的哑指标对
        for ipos1, ipos2 in self._get_sorted_dum_indices_for_canon():
            g[ipos1] = j
            g[ipos2] = j + 1
            j += 2
            typ = self.index_types[ipos1]
            # 如果当前类型不等于前一个类型，将之前的哑指标存入 dummies 中，并重新初始化 a 列表
            if typ != prev:
                if a:
                    dummies.append(a)
                a = [pos, pos + 1]
                prev = typ
                msym.append(metric_symmetry_to_msym(typ.metric))
            else:
                a.extend([pos, pos + 1])
            pos += 2
        # 将最后一组哑指标存入 dummies 中
        if a:
            dummies.append(a)

        # 调用 _af_new 函数生成结果并返回
        return _af_new(g), dummies, msym
# 定义一个函数，用于将给定的组件列表转换为规范化的参数形式
def components_canon_args(components):
    # 用于存储组件类型及其连续出现次数的列表
    numtyp = []
    # 用于记录前一个组件类型
    prev = None
    # 遍历组件列表
    for t in components:
        # 如果当前组件类型与前一个相同
        if t == prev:
            # 更新上一个组件类型的出现次数
            numtyp[-1][1] += 1
        else:
            # 更新前一个组件类型为当前类型，并记录出现次数为1
            prev = t
            numtyp.append([prev, 1])
    
    # 用于存储结果的列表
    v = []
    # 遍历 numtyp 列表中的每一对组件类型及其出现次数
    for h, n in numtyp:
        # 根据组件的 comm 属性确定通讯类型
        if h.comm in (0, 1):
            comm = h.comm
        else:
            # 如果 comm 不是 0 或 1，则调用 TensorManager.get_comm 方法获取通讯类型
            comm = TensorManager.get_comm(h.comm, h.comm)
        # 将组件的基础对称性、生成器、出现次数及通讯类型组成元组，加入结果列表 v
        v.append((h.symmetry.base, h.symmetry.generators, n, comm))
    
    # 返回结果列表 v
    return v


class _TensorDataLazyEvaluator(CantSympify):
    """
    EXPERIMENTAL: do not rely on this class, it may change without deprecation
    warnings in future versions of SymPy.

    Explanation
    ===========

    This object contains the logic to associate components data to a tensor
    expression. Components data are set via the ``.data`` property of tensor
    expressions, is stored inside this class as a mapping between the tensor
    expression and the ``ndarray``.

    Computations are executed lazily: whereas the tensor expressions can have
    contractions, tensor products, and additions, components data are not
    computed until they are accessed by reading the ``.data`` property
    associated to the tensor expression.
    """
    
    # 静态成员变量，用于存储替换字典，用于处理未知类型
    _substitutions_dict: dict[Any, Any] = {}
    # 静态成员变量，用于存储替换字典，用于处理 tensor product
    _substitutions_dict_tensmul: dict[Any, Any] = {}

    def __getitem__(self, key):
        # 获取 key 对应的数据
        dat = self._get(key)
        # 如果数据为 None，则返回 None
        if dat is None:
            return None

        # 导入 NDimArray 类
        from .array import NDimArray
        # 如果数据不是 NDimArray 类型，则直接返回数据
        if not isinstance(dat, NDimArray):
            return dat

        # 如果数据是标量，返回其值
        if dat.rank() == 0:
            return dat[()]
        # 如果数据是一维且只有一个元素，返回该元素
        elif dat.rank() == 1 and len(dat) == 1:
            return dat[0]
        # 否则返回数组本身
        return dat

    @staticmethod
    def data_contract_dum(ndarray_list, dum, ext_rank):
        # 导入 tensorproduct、tensorcontraction、MutableDenseNDimArray 类
        from .array import tensorproduct, tensorcontraction, MutableDenseNDimArray
        # 将 ndarray_list 中的每个 ndarray 转换为 MutableDenseNDimArray 类型的数组
        arrays = list(map(MutableDenseNDimArray, ndarray_list))
        # 计算 tensorproduct
        prodarr = tensorproduct(*arrays)
        # 计算 tensorcontraction
        return tensorcontraction(prodarr, *dum)

    def data_tensorhead_from_tensmul(self, data, tensmul, tensorhead):
        """
        This method is used when assigning components data to a ``TensMul``
        object, it converts components data to a fully contravariant ndarray,
        which is then stored according to the ``TensorHead`` key.
        """
        # 如果 data 为 None，则返回 None
        if data is None:
            return None

        # 调用 _correct_signature_from_indices 方法，根据 indices、free、dum 调整签名
        return self._correct_signature_from_indices(
            data,
            tensmul.get_indices(),
            tensmul.free,
            tensmul.dum,
            True)
    def data_from_tensor(self, tensor):
        """
        This method corrects the components data to the right signature
        (covariant/contravariant) using the metric associated with each
        ``TensorIndexType``.
        """
        # 获取张量的组件头部
        tensorhead = tensor.component

        # 如果张量头部数据为 None，则返回 None
        if tensorhead.data is None:
            return None

        # 调用私有方法 `_correct_signature_from_indices` 来修正张量数据的签名
        return self._correct_signature_from_indices(
            tensorhead.data,
            tensor.get_indices(),
            tensor.free,
            tensor.dum)

    def _assign_data_to_tensor_expr(self, key, data):
        # 如果 key 是 TensAdd 实例，则抛出 ValueError
        if isinstance(key, TensAdd):
            raise ValueError('cannot assign data to TensAdd')
        # 这里假定 key 是 TensMul 实例
        # 如果 key 的 components 不只有一个，则抛出 ValueError
        if len(key.components) != 1:
            raise ValueError('cannot assign data to TensMul with multiple components')
        # 获取 key 的第一个组件作为 tensorhead
        tensorhead = key.components[0]
        # 调用 data_tensorhead_from_tensmul 方法获取新的数据
        newdata = self.data_tensorhead_from_tensmul(data, key, tensorhead)
        return tensorhead, newdata

    def _check_permutations_on_data(self, tens, data):
        from .array import permutedims
        from .array.arrayop import Flatten

        # 根据 tens 的类型不同，获取相应的秩和对称性生成器
        if isinstance(tens, TensorHead):
            rank = tens.rank
            generators = tens.symmetry.generators
        elif isinstance(tens, Tensor):
            rank = tens.rank
            generators = tens.components[0].symmetry.generators
        elif isinstance(tens, TensorIndexType):
            rank = tens.metric.rank
            generators = tens.metric.symmetry.generators

        # 对于每个生成器，检查对数据进行置换后是否保持不变
        for gener in generators:
            # 签名变化为 +1 或 -1，取决于置换是否包含符号改变
            sign_change = +1 if (gener(rank) == rank) else -1
            data_swapped = data
            last_data = data
            permute_axes = list(map(gener, range(rank)))
            # 根据生成器的阶数进行置换
            for i in range(gener.order()-1):
                data_swapped = permutedims(data_swapped, permute_axes)
                # 如果差异数组中有非零值，则抛出 ValueError
                if any(Flatten(last_data - sign_change*data_swapped)):
                    raise ValueError("Component data symmetry structure error")
                last_data = data_swapped
    # 定义特殊方法 __setitem__，用于设置张量对象/表达式的组件数据。

    data = _TensorDataLazyEvaluator.parse_data(value)
    # 解析输入的数据值，将其转换为全逆变形式，并存储在相应的 TensorHead 对象中。

    self._check_permutations_on_data(key, data)
    # 对数据进行排列检查，以确保符合预期的排列规则。

    # 如果 key 不是 TensorHead 或 TensorIndexType 的实例，则需要将数据分配给张量表达式。
    if not isinstance(key, (TensorHead, TensorIndexType)):
        key, data = self._assign_data_to_tensor_expr(key, data)

    # 如果 key 是 TensorHead 的实例，则需要检查数据的维度和索引类型的匹配情况。
    if isinstance(key, TensorHead):
        for dim, indextype in zip(data.shape, key.index_types):
            # 如果索引类型没有关联的组件数据，则抛出 ValueError。
            if indextype.data is None:
                raise ValueError("index type {} has no components data"\
                " associated (needed to raise/lower index)".format(indextype))
            # 如果索引类型的维度不是数值类型，则继续下一个循环。
            if not indextype.dim.is_number:
                continue
            # 如果数据的维度与索引类型的维度不匹配，则抛出 ValueError。
            if dim != indextype.dim:
                raise ValueError("wrong dimension of ndarray")

    # 将 key 和对应的数据存储在 substitutions_dict 中。
    self._substitutions_dict[key] = data

def __delitem__(self, key):
    """
    Remove an item from the substitutions dictionary.

    Explanation
    ===========
    
    Removes the item associated with the given key from the substitutions
    dictionary _substitutions_dict.
    """
    del self._substitutions_dict[key]
    # 从 substitutions_dict 中删除指定的 key 对应的项。

def __contains__(self, key):
    """
    Check if an item exists in the substitutions dictionary.

    Explanation
    ===========
    
    Checks whether the given key exists in the substitutions dictionary
    _substitutions_dict and returns True if it does, otherwise False.
    """
    return key in self._substitutions_dict
    # 检查给定的 key 是否存在于 substitutions_dict 中，如果存在则返回 True，否则返回 False。
    def add_metric_data(self, metric, data):
        """
        Assign data to the ``metric`` tensor. The metric tensor behaves in an
        anomalous way when raising and lowering indices.
        """

        # 直接将数据分配给 `metric` 张量的 `TensorHead`：
        # `TensorHead` 的问题在于其度规是异常的，即升降指标意味着考虑度规或其逆，
        # 这对于其他张量来说并非如此。
        self._substitutions_dict_tensmul[metric, True, True] = data

        # 计算 `data` 的逆转置矩阵
        inverse_transpose = self.inverse_transpose_matrix(data)

        # 在对称空间中，转置与原始矩阵相同，
        # 完全协变度规张量是其逆转置，因此这段代码能够处理非对称度规。
        self._substitutions_dict_tensmul[metric, False, False] = inverse_transpose

        # 处理混合情况，如果度规是对称的，这些与单位矩阵相同。
        m = data.tomatrix()
        invt = inverse_transpose.tomatrix()
        self._substitutions_dict_tensmul[metric, True, False] = m * invt
        self._substitutions_dict_tensmul[metric, False, True] = invt * m

    @staticmethod
    def _flip_index_by_metric(data, metric, pos):
        """
        Rearrange indices of `data` according to the `metric` tensor.

        Args:
            data: Tensor data to be rearranged.
            metric: Metric tensor used for index rearrangement.
            pos: Position of the index to flip (0 or 1).

        Returns:
            Rearranged tensor data according to the metric tensor.
        """
        from .array import tensorproduct, tensorcontraction

        mdim = metric.rank()
        ddim = data.rank()

        if pos == 0:
            # 索引位置为0时，使用度规张量进行张量积和张量缩并
            data = tensorcontraction(
                tensorproduct(
                    metric,
                    data
                ),
                (1, mdim+pos)
            )
        else:
            # 索引位置非0时，使用度规张量进行张量积和张量缩并
            data = tensorcontraction(
                tensorproduct(
                    data,
                    metric
                ),
                (pos, ddim)
            )
        return data

    @staticmethod
    def inverse_matrix(ndarray):
        """
        Compute the inverse matrix of a given ndarray.

        Args:
            ndarray: Input ndarray for which the inverse matrix is computed.

        Returns:
            Inverse matrix of the input ndarray.
        """
        m = ndarray.tomatrix().inv()
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def inverse_transpose_matrix(ndarray):
        """
        Compute the inverse transpose matrix of a given ndarray.

        Args:
            ndarray: Input ndarray for which the inverse transpose matrix is computed.

        Returns:
            Inverse transpose matrix of the input ndarray.
        """
        m = ndarray.tomatrix().inv().T
        return _TensorDataLazyEvaluator.parse_data(m)
    def _correct_signature_from_indices(data, indices, free, dum, inverse=False):
        """
        Utility function to correct the values inside the components data
        ndarray according to whether indices are covariant or contravariant.

        It uses the metric matrix to lower values of covariant indices.
        """
        # 遍历给定的索引列表，并根据其协变性/逆变性改变数组中的值
        # 使用度量矩阵进行操作
        for i, indx in enumerate(indices):
            # 如果索引不是上指标且不是逆变的，则通过度量矩阵翻转索引
            if not indx.is_up and not inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(data, indx.tensor_index_type.data, i)
            # 如果索引不是上指标且是逆变的，则通过逆度量矩阵翻转索引
            elif not indx.is_up and inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(
                    data,
                    _TensorDataLazyEvaluator.inverse_matrix(indx.tensor_index_type.data),
                    i
                )
        return data

    @staticmethod
    def _sort_data_axes(old, new):
        from .array import permutedims

        # 复制旧数据以防止原始数据被修改
        new_data = old.data.copy()

        # 提取旧自由索引并新索引进行比较和排序
        old_free = [i[0] for i in old.free]
        new_free = [i[0] for i in new.free]

        # 对新自由索引进行遍历并匹配旧自由索引，若匹配成功则交换位置
        for i in range(len(new_free)):
            for j in range(i, len(old_free)):
                if old_free[j] == new_free[i]:
                    old_free[i], old_free[j] = old_free[j], old_free[i]
                    # 使用指定的排列顺序重新排列数据
                    new_data = permutedims(new_data, (i, j))
                    break
        return new_data

    @staticmethod
    def add_rearrange_tensmul_parts(new_tensmul, old_tensmul):
        def sorted_compo():
            # 返回根据新张量乘积和旧张量乘积排序的数据组件
            return _TensorDataLazyEvaluator._sort_data_axes(old_tensmul, new_tensmul)

        # 将排序后的数据组件添加到替换字典中
        _TensorDataLazyEvaluator._substitutions_dict[new_tensmul] = sorted_compo()

    @staticmethod
    def parse_data(data):
        """
        Transform ``data`` to array. The parameter ``data`` may
        contain data in various formats, e.g. nested lists, SymPy ``Matrix``,
        and so on.

        Examples
        ========

        >>> from sympy.tensor.tensor import _TensorDataLazyEvaluator
        >>> _TensorDataLazyEvaluator.parse_data([1, 3, -6, 12])
        [1, 3, -6, 12]

        >>> _TensorDataLazyEvaluator.parse_data([[1, 2], [4, 7]])
        [[1, 2], [4, 7]]
        """
        from .array import MutableDenseNDimArray

        # 如果数据不是 MutableDenseNDimArray 类型，则尝试转换为此类型
        if not isinstance(data, MutableDenseNDimArray):
            # 如果数据长度为2且第一个元素是可调用对象，则按给定参数创建 MutableDenseNDimArray
            if len(data) == 2 and hasattr(data[0], '__call__'):
                data = MutableDenseNDimArray(data[0], data[1])
            else:
                # 否则直接创建 MutableDenseNDimArray
                data = MutableDenseNDimArray(data)
        return data
# 创建一个空的 `_TensorDataLazyEvaluator` 实例，用于处理张量数据的惰性评估
_tensor_data_substitution_dict = _TensorDataLazyEvaluator()


class _TensorManager:
    """
    Class to manage tensor properties.

    Notes
    =====

    Tensors belong to tensor commutation groups; each group has a label
    ``comm``; there are predefined labels:

    ``0``   tensors commuting with any other tensor

    ``1``   tensors anticommuting among themselves

    ``2``   tensors not commuting, apart with those with ``comm=0``

    Other groups can be defined using ``set_comm``; tensors in those
    groups commute with those with ``comm=0``; by default they
    do not commute with any other group.
    """
    
    def __init__(self):
        # 初始化张量的交换属性
        self._comm_init()

    def _comm_init(self):
        # 初始化张量的交换属性字典
        self._comm = [{} for i in range(3)]
        
        # 设置默认的交换规则
        for i in range(3):
            self._comm[0][i] = 0
            self._comm[i][0] = 0
        
        # 设置特定的反交换规则
        self._comm[1][1] = 1
        self._comm[2][1] = None
        self._comm[1][2] = None
        
        # 符号到索引的映射
        self._comm_symbols2i = {0: 0, 1: 1, 2: 2}
        # 索引到符号的映射
        self._comm_i2symbol = {0: 0, 1: 1, 2: 2}

    @property
    def comm(self):
        # 返回张量的交换属性字典
        return self._comm

    def comm_symbols2i(self, i):
        """
        Get the commutation group number corresponding to ``i``.

        ``i`` can be a symbol or a number or a string.

        If ``i`` is not already defined its commutation group number
        is set.
        """
        if i not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[n][0] = 0
            self._comm[0][n] = 0
            self._comm_symbols2i[i] = n
            self._comm_i2symbol[n] = i
            return n
        return self._comm_symbols2i[i]

    def comm_i2symbol(self, i):
        """
        Returns the symbol corresponding to the commutation group number.
        """
        return self._comm_i2symbol[i]
    # 设置交换群组 `i, j` 的交换参数 `c`。

    Parameters
    ==========

    i, j : 表示交换群组的符号

    c  :  群组的交换数

    Notes
    =====

    `i, j` 可以是符号、字符串或数字，
    除了分别保留用于交换、反交换张量的 `0, 1` 和与任何其他组件都不交换的张量外。
    对于剩余的情况，请使用此方法设置交换规则；
    默认情况下 `c=None`。

    群组交换数 `c` 被分配给群组交换符号；
    它可以是：

    0        表示交换

    1        表示反交换

    None     表示没有交换属性

    Examples
    ========

    `G` 和 `GH` 不与自身交换，彼此之间交换；A 是交换的。

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorManager, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz')
    >>> i0,i1,i2,i3,i4 = tensor_indices('i0:5', Lorentz)
    >>> A = TensorHead('A', [Lorentz])
    >>> G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
    >>> GH = TensorHead('GH', [Lorentz], TensorSymmetry.no_symmetry(1), 'GHcomm')
    >>> TensorManager.set_comm('Gcomm', 'GHcomm', 0)
    >>> (GH(i1)*G(i0)).canon_bp()
    G(i0)*GH(i1)
    >>> (G(i1)*G(i0)).canon_bp()
    G(i1)*G(i0)
    >>> (G(i1)*A(i0)).canon_bp()
    A(i0)*G(i1)
    """

    # 如果 `c` 不是 (0, 1, None) 中的一个，则引发 ValueError 异常
    if c not in (0, 1, None):
        raise ValueError('`c` can assume only the values 0, 1 or None')

    # 将 `i` 和 `j` 转换为符号对象
    i = sympify(i)
    j = sympify(j)

    # 如果 `i` 不在 `_comm_symbols2i` 中，则添加新的群组
    if i not in self._comm_symbols2i:
        n = len(self._comm)
        self._comm.append({})
        self._comm[n][0] = 0
        self._comm[0][n] = 0
        self._comm_symbols2i[i] = n
        self._comm_i2symbol[n] = i
    
    # 如果 `j` 不在 `_comm_symbols2i` 中，则添加新的群组
    if j not in self._comm_symbols2i:
        n = len(self._comm)
        self._comm.append({})
        self._comm[0][n] = 0
        self._comm[n][0] = 0
        self._comm_symbols2i[j] = n
        self._comm_i2symbol[n] = j
    
    # 获取 `i` 和 `j` 对应的索引
    ni = self._comm_symbols2i[i]
    nj = self._comm_symbols2i[j]

    # 设置 `i` 和 `j` 之间的交换规则为 `c`
    self._comm[ni][nj] = c
    self._comm[nj][ni] = c

    """
    Cached sympy functions (e.g. expand) may have cached the results of
    expressions involving tensors, but those results may not be valid after
    changing the commutation properties. To stay on the safe side, we clear
    the cache of all functions.
    """
    # 清除所有函数的缓存，因为张量的交换属性可能会影响之前缓存的结果
    clear_cache()
    # 设置符号对 ``(i, j)`` 的交换群编号 ``c``。
    def set_comms(self, *args):
        """
        Set the commutation group numbers ``c`` for symbols ``i, j``.

        Parameters
        ==========

        args : sequence of ``(i, j, c)``
        """
        # 遍历参数列表，对每个 ``(i, j, c)`` 调用 set_comm 方法进行设置
        for i, j, c in args:
            self.set_comm(i, j, c)

    # 返回符号对 ``(i, j)`` 的交换参数
    def get_comm(self, i, j):
        """
        Return the commutation parameter for commutation group numbers ``i, j``

        see ``_TensorManager.set_comm``
        """
        # 返回 _comm 中索引为 i 的字典中键为 j 的值，若不存在则返回 0（当 i 或 j 为 0 时）
        return self._comm[i].get(j, 0 if i == 0 or j == 0 else None)

    # 清空 TensorManager 的状态
    def clear(self):
        """
        Clear the TensorManager.
        """
        # 调用 _comm_init 方法来清空状态
        self._comm_init()
TensorManager = _TensorManager()



# 创建一个 `_TensorManager` 的实例并将其赋给 `TensorManager` 变量



class TensorIndexType(Basic):
    """
    A TensorIndexType is characterized by its name and its metric.

    Parameters
    ==========

    name : name of the tensor type
    dummy_name : name of the head of dummy indices
    dim : dimension, it can be a symbol or an integer or ``None``
    eps_dim : dimension of the epsilon tensor
    metric_symmetry : integer that denotes metric symmetry or ``None`` for no metric
    metric_name : string with the name of the metric tensor

    Attributes
    ==========

    ``metric`` : the metric tensor
    ``delta`` : ``Kronecker delta``
    ``epsilon`` : the ``Levi-Civita epsilon`` tensor
    ``data`` : (deprecated) a property to add ``ndarray`` values, to work in a specified basis.

    Notes
    =====

    The possible values of the ``metric_symmetry`` parameter are:

        ``1``   :   metric tensor is fully symmetric
        ``0``   :   metric tensor possesses no index symmetry
        ``-1``  :   metric tensor is fully antisymmetric
        ``None``:   there is no metric tensor (metric equals to ``None``)

    The metric is assumed to be symmetric by default. It can also be set
    to a custom tensor by the ``.set_metric()`` method.

    If there is a metric the metric is used to raise and lower indices.

    In the case of non-symmetric metric, the following raising and
    lowering conventions will be adopted:

    ``psi(a) = g(a, b)*psi(-b); chi(-a) = chi(b)*g(-b, -a)``

    From these it is easy to find:

    ``g(-a, b) = delta(-a, b)``

    where ``delta(-a, b) = delta(b, -a)`` is the ``Kronecker delta``
    (see ``TensorIndex`` for the conventions on indices).
    For antisymmetric metrics there is also the following equality:

    ``g(a, -b) = -delta(a, -b)``

    If there is no metric it is not possible to raise or lower indices;
    e.g. the index of the defining representation of ``SU(N)``
    is 'covariant' and the conjugate representation is
    'contravariant'; for ``N > 2`` they are linearly independent.

    ``eps_dim`` is by default equal to ``dim``, if the latter is an integer;
    else it can be assigned (for use in naive dimensional regularization);
    if ``eps_dim`` is not an integer ``epsilon`` is ``None``.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> Lorentz.metric
    metric(Lorentz,Lorentz)
    """



# 定义了一个 `TensorIndexType` 类，它继承自 `Basic` 类，表示张量索引类型
# 该类具有多个参数和属性，用于描述张量类型的特征和性质，包括名称、度量、维度等
# 类中包含详细的文档字符串，解释了各个参数和属性的含义、使用方法及示例
    def __new__(cls, name, dummy_name=None, dim=None, eps_dim=None,
                metric_symmetry=1, metric_name='metric', **kwargs):
        # 如果传入参数 kwargs 中包含 'dummy_fmt'，则发出警告并使用其作为 dummy_name
        if 'dummy_fmt' in kwargs:
            dummy_fmt = kwargs['dummy_fmt']
            sympy_deprecation_warning(
                f"""
                The dummy_fmt keyword to TensorIndexType is deprecated. Use
                dummy_name={dummy_fmt} instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-dummy-fmt",
            )
            dummy_name = dummy_fmt

        # 如果 name 是字符串类型，则将其转换为符号类型
        if isinstance(name, str):
            name = Symbol(name)

        # 如果 dummy_name 为 None，则将其设置为 name 的首字母作为符号类型
        if dummy_name is None:
            dummy_name = str(name)[0]
        if isinstance(dummy_name, str):
            dummy_name = Symbol(dummy_name)

        # 如果 dim 为 None，则创建一个以 dummy_name 为名称的符号类型 dim
        # 否则，将 dim 转换为 SymPy 的表达式
        if dim is None:
            dim = Symbol("dim_" + dummy_name.name)
        else:
            dim = sympify(dim)

        # 如果 eps_dim 为 None，则将其设置为 dim
        # 否则，将 eps_dim 转换为 SymPy 的表达式
        if eps_dim is None:
            eps_dim = dim
        else:
            eps_dim = sympify(eps_dim)

        # 将 metric_symmetry 转换为 SymPy 的表达式
        metric_symmetry = sympify(metric_symmetry)

        # 如果 metric_name 是字符串类型，则将其转换为符号类型
        if isinstance(metric_name, str):
            metric_name = Symbol(metric_name)

        # 如果 kwargs 中包含 'metric' 关键字，则发出警告
        if 'metric' in kwargs:
            SymPyDeprecationWarning(
                """
                The 'metric' keyword argument to TensorIndexType is
                deprecated. Use the 'metric_symmetry' keyword argument or the
                TensorIndexType.set_metric() method instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-metric",
            )
            # 获取 'metric' 对应的值
            metric = kwargs.get('metric')
            if metric is not None:
                # 如果 metric 是 True、False、0 或 1 中的一个，则设置 metric_name 为 'metric'
                # 否则，将 metric_name 设置为 metric 的名称
                if metric in (True, False, 0, 1):
                    metric_name = 'metric'
                    #metric_antisym = metric
                else:
                    metric_name = metric.name
                    #metric_antisym = metric.antisym

                # 如果 metric 是 True，则 metric_symmetry 设置为 -1
                # 否则，设置为 1
                if metric:
                    metric_symmetry = -1
                else:
                    metric_symmetry = 1

        # 使用父类 Basic 的 __new__ 方法创建一个新的对象
        obj = Basic.__new__(cls, name, dummy_name, dim, eps_dim,
                            metric_symmetry, metric_name)

        # 初始化一个属性 _autogenerated，值为空列表
        obj._autogenerated = []
        return obj

    @property
    def name(self):
        # 返回对象的第一个参数 name 的名称
        return self.args[0].name

    @property
    def dummy_name(self):
        # 返回对象的第二个参数 dummy_name 的名称
        return self.args[1].name

    @property
    def dim(self):
        # 返回对象的第三个参数 dim
        return self.args[2]

    @property
    def eps_dim(self):
        # 返回对象的第四个参数 eps_dim
        return self.args[3]

    @memoize_property
    # 计算度量
    def metric(self):
        # 获取度量的对称性和名称参数
        metric_symmetry = self.args[4]
        metric_name = self.args[5]
        # 如果度量对称性为 None，则返回 None
        if metric_symmetry is None:
            return None

        # 根据度量对称性参数选择对称性类型
        if metric_symmetry == 0:
            symmetry = TensorSymmetry.no_symmetry(2)
        elif metric_symmetry == 1:
            symmetry = TensorSymmetry.fully_symmetric(2)
        elif metric_symmetry == -1:
            symmetry = TensorSymmetry.fully_symmetric(-2)

        # 返回一个 TensorHead 对象，表示该度量
        return TensorHead(metric_name, [self]*2, symmetry)

    @memoize_property
    # 返回一个指定的 TensorHead 对象，表示 Kronecker δ
    def delta(self):
        return TensorHead('KD', [self]*2, TensorSymmetry.fully_symmetric(2))

    @memoize_property
    # 返回一个指定的 TensorHead 对象，表示 Levi-Civita 符号 ε
    def epsilon(self):
        # 如果 eps_dim 不是整数类型，则返回 None
        if not isinstance(self.eps_dim, (SYMPY_INTS, Integer)):
            return None
        # 根据 eps_dim 的值创建对应的对称性类型
        symmetry = TensorSymmetry.fully_symmetric(-self.eps_dim)
        # 返回一个 TensorHead 对象，表示 Levi-Civita 符号 ε
        return TensorHead('Eps', [self]*self.eps_dim, symmetry)

    # 设置度量张量的值
    def set_metric(self, tensor):
        self._metric = tensor

    # 定义小于运算符，比较 Tensor 对象的名称
    def __lt__(self, other):
        return self.name < other.name

    # 定义字符串表示形式，返回 Tensor 对象的名称
    def __str__(self):
        return self.name

    # 将 __repr__ 方法与 __str__ 方法相同
    __repr__ = __str__

    # 以下所有代码已被弃用

    # 返回存储在 Tensor 对象中的数据，通过调用 deprecate_data() 方法告知弃用
    @property
    def data(self):
        deprecate_data()
        # 忽略 SymPy 的弃用警告，返回与当前 Tensor 对象相关的数据字典
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    # 设置存储在 Tensor 对象中的数据，通过调用 deprecate_data() 方法告知弃用
    @data.setter
    def data(self, data):
        deprecate_data()
        # 导入 MutableDenseNDimArray 类，用于处理数组转换
        from .array import MutableDenseNDimArray

        # 解析传入的数据
        data = _TensorDataLazyEvaluator.parse_data(data)
        # 检查数据的秩是否大于 2
        if data.rank() > 2:
            raise ValueError("data have to be of rank 1 (diagonal metric) or 2.")
        # 如果数据的秩为 1，则转换为对角线形式的 2D 数组
        if data.rank() == 1:
            if self.dim.is_number:
                nda_dim = data.shape[0]
                if nda_dim != self.dim:
                    raise ValueError("Dimension mismatch")

            dim = data.shape[0]
            newndarray = MutableDenseNDimArray.zeros(dim, dim)
            for i, val in enumerate(data):
                newndarray[i, i] = val
            data = newndarray
        # 获取数据的维度信息
        dim1, dim2 = data.shape
        # 检查数据是否为非方阵
        if dim1 != dim2:
            raise ValueError("Non-square matrix tensor.")
        # 如果 Tensor 对象具有定义的维度，则检查与数据维度的匹配情况
        if self.dim.is_number:
            if self.dim != dim1:
                raise ValueError("Dimension mismatch")
        # 更新存储 Tensor 数据的全局字典
        _tensor_data_substitution_dict[self] = data
        # 将数据添加到度量对象中
        _tensor_data_substitution_dict.add_metric_data(self.metric, data)
        # 忽略 SymPy 的弃用警告，获取 Kronecker δ
        with ignore_warnings(SymPyDeprecationWarning):
            delta = self.get_kronecker_delta()
        # 创建 TensorIndex 对象 i1 和 i2
        i1 = TensorIndex('i1', self)
        i2 = TensorIndex('i2', self)
        # 忽略 SymPy 的弃用警告，使用单位矩阵更新数据
        with ignore_warnings(SymPyDeprecationWarning):
            delta(i1, -i2).data = _TensorDataLazyEvaluator.parse_data(eye(dim1))

    # 定义数据删除器，告知数据操作已被弃用
    @data.deleter
    # 调用 deprecate_data() 函数，标记该函数已废弃
    deprecate_data()
    
    # 忽略 SymPyDeprecationWarning 警告的上下文执行以下代码块
    with ignore_warnings(SymPyDeprecationWarning):
        # 如果 self 存在于 _tensor_data_substitution_dict 中，则从字典中删除 self 对应的条目
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]
        
        # 如果 self.metric 存在于 _tensor_data_substitution_dict 中，则从字典中删除 self.metric 对应的条目
        if self.metric in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self.metric]

@deprecated(
    """
    The TensorIndexType.get_kronecker_delta() method is deprecated. Use
    the TensorIndexType.delta attribute instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorindextype-methods",
)
def get_kronecker_delta(self):
    # 创建一个关于对称群 (symmetric group) S_2 的 TensorSymmetry 对象
    sym2 = TensorSymmetry(get_symmetric_group_sgs(2))
    
    # 创建一个名为 'KD' 的 TensorHead 对象，表示 Kronecker delta 符号
    delta = TensorHead('KD', [self]*2, sym2)
    
    # 返回 delta 对象
    return delta

@deprecated(
    """
    The TensorIndexType.get_epsilon() method is deprecated. Use
    the TensorIndexType.epsilon attribute instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorindextype-methods",
)
def get_epsilon(self):
    # 如果 self._eps_dim 不是 SYMPY_INTS 或 Integer 类型，则返回 None
    if not isinstance(self._eps_dim, (SYMPY_INTS, Integer)):
        return None
    
    # 根据 self._eps_dim 创建关于对称群的 TensorSymmetry 对象
    sym = TensorSymmetry(get_symmetric_group_sgs(self._eps_dim, 1))
    
    # 创建一个名为 'Eps' 的 TensorHead 对象，表示 Levi-Civita 符号
    epsilon = TensorHead('Eps', [self]*self._eps_dim, sym)
    
    # 返回 epsilon 对象
    return epsilon

def _components_data_full_destroy(self):
    """
    EXPERIMENTAL: do not rely on this API method.

    This destroys components data associated to the ``TensorIndexType``, if
    any, specifically:

    * metric tensor data
    * Kronecker tensor data
    """
    # 如果 self 存在于 _tensor_data_substitution_dict 中，则从字典中删除 self 对应的条目
    if self in _tensor_data_substitution_dict:
        del _tensor_data_substitution_dict[self]

    # 定义一个内部函数 delete_tensmul_data，用于删除与 TensMul 相关的数据
    def delete_tensmul_data(key):
        # 如果 key 存在于 _tensor_data_substitution_dict._substitutions_dict_tensmul 中，则从字典中删除 key 对应的条目
        if key in _tensor_data_substitution_dict._substitutions_dict_tensmul:
            del _tensor_data_substitution_dict._substitutions_dict_tensmul[key]

    # 删除与 metric 张量相关的数据：
    delete_tensmul_data((self.metric, True, True))
    delete_tensmul_data((self.metric, True, False))
    delete_tensmul_data((self.metric, False, True))
    delete_tensmul_data((self.metric, False, False))

    # 获取 Kronecker delta 对象并尝试删除与之相关的数据条目：
    delta = self.get_kronecker_delta()
    if delta in _tensor_data_substitution_dict:
        del _tensor_data_substitution_dict[delta]
class TensorIndex(Basic):
    """
    Represents a tensor index

    Parameters
    ==========

    name : name of the index, or ``True`` if you want it to be automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)

    Attributes
    ==========

    ``name``
    ``tensor_index_type``
    ``is_up``

    Notes
    =====

    Tensor indices are contracted with the Einstein summation convention.

    An index can be in contravariant or in covariant form; in the latter
    case it is represented prepending a ``-`` to the index name. Adding
    ``-`` to a covariant (is_up=False) index makes it contravariant.

    Dummy indices have a name with head given by
    ``tensor_inde_type.dummy_name`` with underscore and a number.

    Similar to ``symbols`` multiple contravariant indices can be created
    at once using ``tensor_indices(s, typ)``, where ``s`` is a string
    of names.


    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> mu = TensorIndex('mu', Lorentz, is_up=False)
    >>> nu, rho = tensor_indices('nu, rho', Lorentz)
    >>> A = TensorHead('A', [Lorentz, Lorentz])
    >>> A(mu, nu)
    A(-mu, nu)
    >>> A(-mu, -rho)
    A(mu, -rho)
    >>> A(mu, -mu)
    A(-L_0, L_0)
    """

    def __new__(cls, name, tensor_index_type, is_up=True):
        # 根据传入的参数，创建一个新的 TensorIndex 实例
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            # 如果 name 为 True，则自动分配一个名称
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")

        # 确保 is_up 参数为布尔值
        is_up = sympify(is_up)
        # 调用父类的构造方法创建实例
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up)

    @property
    def name(self):
        # 返回当前 TensorIndex 实例的名称
        return self.args[0].name

    @property
    def tensor_index_type(self):
        # 返回当前 TensorIndex 实例的索引类型
        return self.args[1]

    @property
    def is_up(self):
        # 返回当前 TensorIndex 实例的升降标志
        return self.args[2]

    def _print(self):
        # 根据 TensorIndex 实例的升降标志，打印对应的字符串表示
        s = self.name
        if not self.is_up:
            s = '-%s' % s
        return s

    def __lt__(self, other):
        # 实现 TensorIndex 实例的小于比较，依据索引类型和名称
        return ((self.tensor_index_type, self.name) <
                (other.tensor_index_type, other.name))

    def __neg__(self):
        # 返回当前 TensorIndex 实例的相反指标
        t1 = TensorIndex(self.name, self.tensor_index_type,
                         (not self.is_up))
        return t1


def tensor_indices(s, typ):
    """
    Returns list of tensor indices given their names and their types.

    Parameters
    ==========

    s : string of comma separated names of indices

    typ : ``TensorIndexType`` of the indices

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    """
    # 根据输入的字符串 s 和类型 typ，创建一组张量指标对象，并返回列表
    pass  # 此函数还未实现具体功能，暂时使用 pass 占位符
    # 如果输入参数 s 是一个字符串，则将其解析为符号列表，并提取符号的名称
    if isinstance(s, str):
        # 使用 sympy 库的 symbols 函数创建一个符号列表，并提取每个符号的名称
        a = [x.name for x in symbols(s, seq=True)]
    else:
        # 如果输入参数不是字符串，则抛出数值错误异常
        raise ValueError('expecting a string')
    
    # 使用列表推导式，将每个符号名称转换为对应的 TensorIndex 对象，并放入列表 tilist 中
    tilist = [TensorIndex(i, typ) for i in a]
    
    # 如果列表 tilist 中只有一个元素，则直接返回该元素
    if len(tilist) == 1:
        return tilist[0]
    
    # 否则，返回整个列表 tilist
    return tilist
    """
    Monoterm symmetry of a tensor (i.e. any symmetric or anti-symmetric
    index permutation). For the relevant terminology see ``tensor_can.py``
    section of the combinatorics module.

    Parameters
    ==========

    bsgs : tuple ``(base, sgs)`` BSGS of the symmetry of the tensor

    Attributes
    ==========

    ``base`` : base of the BSGS
    ``generators`` : generators of the BSGS
    ``rank`` : rank of the tensor

    Notes
    =====

    A tensor can have an arbitrary monoterm symmetry provided by its BSGS.
    Multiterm symmetries, like the cyclic symmetry of the Riemann tensor
    (i.e., Bianchi identity), are not covered. See combinatorics module for
    information on how to generate BSGS for a general index permutation group.
    Simple symmetries can be generated using built-in methods.

    See Also
    ========

    sympy.combinatorics.tensor_can.get_symmetric_group_sgs

    Examples
    ========

    Define a symmetric tensor of rank 2

    >>> from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> sym = TensorSymmetry(get_symmetric_group_sgs(2))
    >>> T = TensorHead('T', [Lorentz]*2, sym)

    Note, that the same can also be done using built-in TensorSymmetry methods

    >>> sym2 = TensorSymmetry.fully_symmetric(2)
    >>> sym == sym2
    True
    """
    def __new__(cls, *args, **kw_args):
        if len(args) == 1:
            base, generators = args[0]
        elif len(args) == 2:
            base, generators = args
        else:
            raise TypeError("bsgs required, either two separate parameters or one tuple")

        if not isinstance(base, Tuple):
            base = Tuple(*base)
        if not isinstance(generators, Tuple):
            generators = Tuple(*generators)

        return Basic.__new__(cls, base, generators, **kw_args)

    @property
    def base(self):
        """
        Return the base of the BSGS (Base and Strong Generating Set) of the tensor symmetry.

        This represents the base elements used in the BSGS algorithm.
        """
        return self.args[0]

    @property
    def generators(self):
        """
        Return the generators of the BSGS (Base and Strong Generating Set) of the tensor symmetry.

        These generators define the symmetry group of the tensor.
        """
        return self.args[1]

    @property
    def rank(self):
        """
        Return the rank of the tensor.

        It is derived from the size of the first generator minus 2.
        """
        return self.generators[0].size - 2

    @classmethod
    def fully_symmetric(cls, rank):
        """
        Returns a fully symmetric (antisymmetric if ``rank``<0)
        TensorSymmetry object for ``abs(rank)`` indices.

        Constructs a TensorSymmetry object for a fully symmetric or antisymmetric tensor
        based on the given rank.
        """
        if rank > 0:
            bsgs = get_symmetric_group_sgs(rank, False)
        elif rank < 0:
            bsgs = get_symmetric_group_sgs(-rank, True)
        elif rank == 0:
            bsgs = ([], [Permutation(1)])
        return TensorSymmetry(bsgs)

    @classmethod
    def direct_product(cls, *args):
        """
        Returns a TensorSymmetry object that is being a direct product of
        fully (anti-)symmetric index permutation groups.

        Notes
        =====

        Some examples for different values of ``(*args)``:
        ``(1)``         vector, equivalent to ``TensorSymmetry.fully_symmetric(1)``
        ``(2)``         tensor with 2 symmetric indices, equivalent to ``.fully_symmetric(2)``
        ``(-2)``        tensor with 2 antisymmetric indices, equivalent to ``.fully_symmetric(-2)``
        ``(2, -2)``     tensor with the first 2 indices commuting and the last 2 anticommuting
        ``(1, 1, 1)``   tensor with 3 indices without any symmetry
        """
        # Initialize base as an empty list and sgs with a list containing a single Permutation object
        base, sgs = [], [Permutation(1)]
        # Iterate through each argument provided
        for arg in args:
            # Check if arg is positive
            if arg > 0:
                # Get symmetric group structure for positive arg (commutative)
                bsgs2 = get_symmetric_group_sgs(arg, False)
            # Check if arg is negative
            elif arg < 0:
                # Get symmetric group structure for negative arg (anticommutative)
                bsgs2 = get_symmetric_group_sgs(-arg, True)
            else:
                # Skip arg if it's zero
                continue
            # Compute direct product of base and sgs with bsgs2
            base, sgs = bsgs_direct_product(base, sgs, *bsgs2)

        # Return a TensorSymmetry object initialized with computed base and sgs
        return TensorSymmetry(base, sgs)

    @classmethod
    def riemann(cls):
        """
        Returns a monotorem symmetry of the Riemann tensor
        """
        # Return a TensorSymmetry object initialized with riemann_bsgs
        return TensorSymmetry(riemann_bsgs)

    @classmethod
    def no_symmetry(cls, rank):
        """
        TensorSymmetry object for ``rank`` indices with no symmetry
        """
        # Return a TensorSymmetry object initialized with an empty base and a single Permutation object
        return TensorSymmetry([], [Permutation(rank+1)])
@deprecated(
    """
    The tensorsymmetry() function is deprecated. Use the TensorSymmetry
    constructor instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorsymmetry",
)
def tensorsymmetry(*args):
    """
    Returns a ``TensorSymmetry`` object. This method is deprecated, use
    ``TensorSymmetry.direct_product()`` or ``.riemann()`` instead.

    Explanation
    ===========

    One can represent a tensor with any monoterm slot symmetry group
    using a BSGS.

    ``args`` can be a BSGS
    ``args[0]``    base
    ``args[1]``    sgs

    Usually tensors are in (direct products of) representations
    of the symmetric group;
    ``args`` can be a list of lists representing the shapes of Young tableaux

    Notes
    =====

    For instance:
    ``[[1]]``       vector
    ``[[1]*n]``     symmetric tensor of rank ``n``
    ``[[n]]``       antisymmetric tensor of rank ``n``
    ``[[2, 2]]``    monoterm slot symmetry of the Riemann tensor
    ``[[1],[1]]``   vector*vector
    ``[[2],[1],[1]`` (antisymmetric tensor)*vector*vector

    Notice that with the shape ``[2, 2]`` we associate only the monoterm
    symmetries of the Riemann tensor; this is an abuse of notation,
    since the shape ``[2, 2]`` corresponds usually to the irreducible
    representation characterized by the monoterm symmetries and by the
    cyclic symmetry.
    """
    from sympy.combinatorics import Permutation

    # 定义内部函数tableau2bsgs，将Young图形式转换为BSGS对象
    def tableau2bsgs(a):
        if len(a) == 1:
            # 对称向量的情况
            n = a[0]
            bsgs = get_symmetric_group_sgs(n, 1)
        else:
            if all(x == 1 for x in a):
                # 对称张量的情况
                n = len(a)
                bsgs = get_symmetric_group_sgs(n)
            elif a == [2, 2]:
                # Riemann张量的单项槽对称性的情况
                bsgs = riemann_bsgs
            else:
                raise NotImplementedError
        return bsgs

    # 如果没有参数传入，则返回一个空的TensorSymmetry对象
    if not args:
        return TensorSymmetry(Tuple(), Tuple(Permutation(1)))

    # 如果参数长度为2且第二个参数的第一个元素是Permutation对象，则返回一个TensorSymmetry对象
    if len(args) == 2 and isinstance(args[1][0], Permutation):
        return TensorSymmetry(args)

    # 否则，根据参数调用tableau2bsgs函数，构建base和sgs，然后合并这些结果
    base, sgs = tableau2bsgs(args[0])
    for a in args[1:]:
        basex, sgsx = tableau2bsgs(a)
        base, sgs = bsgs_direct_product(base, sgs, basex, sgsx)
    return TensorSymmetry(Tuple(base, sgs))


@deprecated(
    "TensorType is deprecated. Use tensor_heads() instead.",
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensortype",
)
class TensorType(Basic):
    """
    Class of tensor types. Deprecated, use tensor_heads() instead.

    Parameters
    ==========

    index_types : list of ``TensorIndexType`` of the tensor indices
    symmetry : ``TensorSymmetry`` of the tensor

    Attributes
    ==========

    ``index_types``
    ``symmetry``
    ``types`` : list of ``TensorIndexType`` without repetitions
    """

    # 标记类的交换性为False
    is_commutative = False
    # 重载类的特殊方法 __new__，用于创建新的 TensorType 对象
    def __new__(cls, index_types, symmetry, **kw_args):
        # 断言对称性对象的秩（rank）与索引类型列表的长度相等
        assert symmetry.rank == len(index_types)
        # 调用父类 Basic 的 __new__ 方法创建对象
        obj = Basic.__new__(cls, Tuple(*index_types), symmetry, **kw_args)
        # 返回创建的对象
        return obj

    # 定义属性 index_types，返回对象的第一个参数，即索引类型的元组
    @property
    def index_types(self):
        return self.args[0]

    # 定义属性 symmetry，返回对象的第二个参数，即对称性对象
    @property
    def symmetry(self):
        return self.args[1]

    # 定义属性 types，返回索引类型的排序列表，按名称排序
    def types(self):
        return sorted(set(self.index_types), key=lambda x: x.name)

    # 定义特殊方法 __str__，返回描述对象的字符串表示
    def __str__(self):
        return 'TensorType(%s)' % ([str(x) for x in self.index_types])

    # 定义特殊方法 __call__，根据参数 s 和 comm 返回 TensorHead 对象或对象列表
    def __call__(self, s, comm=0):
        """
        Return a TensorHead object or a list of TensorHead objects.

        Parameters
        ==========

        s : name or string of names.

        comm : Commutation group.

        see ``_TensorManager.set_comm``
        """
        # 如果 s 是字符串，则创建符号名称列表
        if isinstance(s, str):
            names = [x.name for x in symbols(s, seq=True)]
        else:
            # 如果 s 不是字符串，引发 ValueError 异常
            raise ValueError('expecting a string')
        # 如果符号名称列表只有一个元素，返回单个 TensorHead 对象
        if len(names) == 1:
            return TensorHead(names[0], self.index_types, self.symmetry, comm)
        else:
            # 否则返回多个 TensorHead 对象组成的列表
            return [TensorHead(name, self.index_types, self.symmetry, comm) for name in names]
# 使用装饰器标记函数为已弃用，并提供替代方法及相关信息
@deprecated(
    """
    The tensorhead() function is deprecated. Use tensor_heads() instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorhead",
)
def tensorhead(name, typ, sym=None, comm=0):
    """
    Function generating tensorhead(s). This method is deprecated,
    use TensorHead constructor or tensor_heads() instead.

    Parameters
    ==========

    name : name or sequence of names (as in ``symbols``)

    typ :  index types

    sym :  same as ``*args`` in ``tensorsymmetry``

    comm : commutation group number
    see ``_TensorManager.set_comm``
    """
    # 如果未提供 sym 参数，则初始化为适当的默认对称性
    if sym is None:
        sym = [[1] for i in range(len(typ))]
    # 忽略 SymPy 的弃用警告，调用 tensorsymmetry 函数处理对称性
    with ignore_warnings(SymPyDeprecationWarning):
        sym = tensorsymmetry(*sym)
    # 返回 TensorHead 对象的实例化结果，传递参数 name, typ, sym, comm
    return TensorHead(name, typ, sym, comm)


class TensorHead(Basic):
    """
    Tensor head of the tensor.

    Parameters
    ==========

    name : name of the tensor
    index_types : list of TensorIndexType
    symmetry : TensorSymmetry of the tensor
    comm : commutation group number

    Attributes
    ==========

    ``name``
    ``index_types``
    ``rank`` : total number of indices
    ``symmetry``
    ``comm`` : commutation group

    Notes
    =====

    Similar to ``symbols`` multiple TensorHeads can be created using
    ``tensorhead(s, typ, sym=None, comm=0)`` function, where ``s``
    is the string of names and ``sym`` is the monoterm tensor symmetry
    (see ``tensorsymmetry``).

    A ``TensorHead`` belongs to a commutation group, defined by a
    symbol on number ``comm`` (see ``_TensorManager.set_comm``);
    tensors in a commutation group have the same commutation properties;
    by default ``comm`` is ``0``, the group of the commuting tensors.

    Examples
    ========

    Define a fully antisymmetric tensor of rank 2:

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> asym2 = TensorSymmetry.fully_symmetric(-2)
    >>> A = TensorHead('A', [Lorentz, Lorentz], asym2)

    Examples with ndarray values, the components data assigned to the
    ``TensorHead`` object are assumed to be in a fully-contravariant
    representation. In case it is necessary to assign components data which
    represents the values of a non-fully covariant tensor, see the other
    examples.

    >>> from sympy.tensor.tensor import tensor_indices
    >>> from sympy import diag
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i0, i1 = tensor_indices('i0:2', Lorentz)

    Specify a replacement dictionary to keep track of the arrays to use for
    replacements in the tensorial expression. The ``TensorIndexType`` is
    associated to the metric used for contractions (in fully covariant form):

    >>> repl = {Lorentz: diag(1, -1, -1, -1)}

    Let's see some examples of working with components with the electromagnetic
    tensor:

    >>> from sympy import symbols
    # 导入必要的符号和张量操作库
    >>> Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
    >>> c = symbols('c', positive=True)

    # 定义反对称张量 `F`
    >>> F = TensorHead('F', [Lorentz, Lorentz], asym2)

    # 更新替换字典，包含用于替换的矩阵
    >>> repl.update({F(-i0, -i1): [
    ... [0, Ex/c, Ey/c, Ez/c],
    ... [-Ex/c, 0, -Bz, By],
    ... [-Ey/c, Bz, 0, -Bx],
    ... [-Ez/c, -By, Bx, 0]]})

    # 检索电磁张量的逆变形式
    >>> F(i0, i1).replace_with_arrays(repl, [i0, i1])
    [[0, -E_x/c, -E_y/c, -E_z/c], [E_x/c, 0, -B_z, B_y], [E_y/c, B_z, 0, -B_x], [E_z/c, -B_y, B_x, 0]]

    # 检索电磁张量的混合逆变协变形式
    >>> F(i0, -i1).replace_with_arrays(repl, [i0, -i1])
    [[0, E_x/c, E_y/c, E_z/c], [E_x/c, 0, B_z, -B_y], [E_y/c, -B_z, 0, B_x], [E_z/c, B_y, -B_x, 0]]

    # 表示粒子的能量-动量张量
    >>> from sympy import symbols
    >>> P = TensorHead('P', [Lorentz], TensorSymmetry.no_symmetry(1))
    >>> E, px, py, pz = symbols('E p_x p_y p_z', positive=True)
    >>> repl.update({P(i0): [E, px, py, pz]})

    # 检索能量-动量张量的逆变和协变分量
    >>> P(i0).replace_with_arrays(repl, [i0])
    [E, p_x, p_y, p_z]
    >>> P(-i0).replace_with_arrays(repl, [-i0])
    [E, -p_x, -p_y, -p_z]

    # 计算张量自身的缩并
    >>> expr = P(i0)*P(-i0)
    >>> expr.replace_with_arrays(repl, [])
    E**2 - p_x**2 - p_y**2 - p_z**2
    """
    # 设置张量是否可交换的属性为 False
    is_commutative = False

    # 构造函数，初始化张量名称、索引类型、对称性和交换信息
    def __new__(cls, name, index_types, symmetry=None, comm=0):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError("invalid name")

        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)

        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), symmetry, sympify(comm))
        return obj

    # 返回张量的名称属性
    @property
    def name(self):
        return self.args[0].name

    # 返回张量的索引类型列表
    @property
    def index_types(self):
        return list(self.args[1])

    # 返回张量的对称性
    @property
    def symmetry(self):
        return self.args[2]

    # 返回张量的交换信息
    @property
    def comm(self):
        return TensorManager.comm_symbols2i(self.args[3])

    # 返回张量的秩（索引的数量）
    @property
    def rank(self):
        return len(self.index_types)

    # 比较函数，按名称和索引类型排序张量
    def __lt__(self, other):
        return (self.name, self.index_types) < (other.name, other.index_types)

    # 判断张量与另一个张量是否交换
    def commutes_with(self, other):
        """
        如果 `self` 和 `other` 交换则返回 `0`，如果反交换则返回 `1`。

        如果 `self` 和 `other` 既不交换也不反交换，则返回 `None`。
        """
        r = TensorManager.get_comm(self.comm, other.comm)
        return r
    # 返回格式化的字符串，包含对象的名称和索引类型的字符串表示
    def _print(self):
        return '%s(%s)' %(self.name, ','.join([str(x) for x in self.index_types]))

    # 实现对象的可调用接口，返回一个包含指定索引的张量对象
    def __call__(self, *indices, **kw_args):
        """
        Returns a tensor with indices.

        Explanation
        ===========

        There is a special behavior in case of indices denoted by ``True``,
        they are considered auto-matrix indices, their slots are automatically
        filled, and confer to the tensor the behavior of a matrix or vector
        upon multiplication with another tensor containing auto-matrix indices
        of the same ``TensorIndexType``. This means indices get summed over the
        same way as in matrix multiplication. For matrix behavior, define two
        auto-matrix indices, for vector behavior define just one.

        Indices can also be strings, in which case the attribute
        ``index_types`` is used to convert them to proper ``TensorIndex``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, TensorHead
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b = tensor_indices('a,b', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
        >>> t = A(a, -b)
        >>> t
        A(a, -b)

        """

        updated_indices = []
        # 遍历输入的索引，根据其类型进行处理
        for idx, typ in zip(indices, self.index_types):
            if isinstance(idx, str):
                # 如果索引是字符串，清除空格并创建相应的 TensorIndex 对象
                idx = idx.strip().replace(" ", "")
                if idx.startswith('-'):
                    updated_indices.append(TensorIndex(idx[1:], typ,
                                           is_up=False))
                else:
                    updated_indices.append(TensorIndex(idx, typ))
            else:
                updated_indices.append(idx)

        # 添加未处理的额外索引到更新后的索引列表中
        updated_indices += indices[len(updated_indices):]

        # 使用更新后的索引创建 Tensor 对象，并返回计算结果
        tensor = Tensor(self, updated_indices, **kw_args)
        return tensor.doit()

    # 以下所有内容已经过时

    # 定义对象的指数运算操作
    def __pow__(self, other):
        deprecate_data()  # 调用数据已过时的警告
        with ignore_warnings(SymPyDeprecationWarning):  # 忽略 SymPy 弃用警告
            if self.data is None:
                raise ValueError("No power on abstract tensors.")
            from .array import tensorproduct, tensorcontraction
            metrics = [_.data for _ in self.index_types]

            marray = self.data
            marraydim = marray.rank()
            # 应用张量积和张量缩并操作
            for metric in metrics:
                marray = tensorproduct(marray, metric, marray)
                marray = tensorcontraction(marray, (0, marraydim), (marraydim+1, marraydim+2))

            # 返回指数运算后的结果
            return marray ** (other * S.Half)

    # 返回对象的数据属性值
    @property
    def data(self):
        deprecate_data()  # 调用数据已过时的警告
        with ignore_warnings(SymPyDeprecationWarning):  # 忽略 SymPy 弃用警告
            return _tensor_data_substitution_dict[self]

    # 设置对象的数据属性值
    @data.setter
    def data(self, data):
        deprecate_data()  # 调用数据已过时的警告
        with ignore_warnings(SymPyDeprecationWarning):  # 忽略 SymPy 弃用警告
            _tensor_data_substitution_dict[self] = data
    # 使用 `data` 装饰器定义一个 `data` 属性的删除器方法，用于删除对象的数据
    @data.deleter
    def data(self):
        # 弃用 `data` 属性，并触发相关处理
        deprecate_data()
        # 如果对象存在于 `_tensor_data_substitution_dict` 中，则从中删除
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

    # 定义 `__iter__` 方法，使对象可迭代
    def __iter__(self):
        # 弃用 `data` 属性，并触发相关处理
        deprecate_data()
        # 使用 `ignore_warnings` 上下文管理器，忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 返回 `data` 属性的迭代器
            return self.data.__iter__()

    # 定义 `_components_data_full_destroy` 方法，用于完全销毁与 `TensorHead` 对象关联的组件数据
    def _components_data_full_destroy(self):
        """
        实验性方法：不要依赖此 API 方法。

        销毁与 ``TensorHead`` 对象关联的组件数据，检查并销毁已附加的组件数据。
        """
        # 不要对 Kronecker 张量进行垃圾回收（应由 ``TensorIndexType`` 的垃圾回收完成）
        deprecate_data()
        # 如果对象的名称是 "KD"，则直接返回，不执行后续操作
        if self.name == "KD":
            return

        # 仅应由 `TensorHead` 析构函数删除附加到张量的数据。如果 `TensorHead` 被删除，
        # 这意味着没有任何地方还有该张量的实例。
        if self in _tensor_data_substitution_dict:
            # 如果对象存在于 `_tensor_data_substitution_dict` 中，则从中删除
            del _tensor_data_substitution_dict[self]
# 定义函数 `tensor_heads`，从字符串 `s` 中生成一系列 TensorHead 对象
def tensor_heads(s, index_types, symmetry=None, comm=0):
    """
    Returns a sequence of TensorHeads from a string `s`
    """
    # 如果输入的 `s` 是字符串，使用 `symbols` 函数获取其中所有符号的名称
    if isinstance(s, str):
        names = [x.name for x in symbols(s, seq=True)]
    else:
        # 如果 `s` 不是字符串，抛出 ValueError 异常
        raise ValueError('expecting a string')

    # 使用列表推导式，为每个名称创建一个 TensorHead 对象，存储在 thlist 中
    thlist = [TensorHead(name, index_types, symmetry, comm) for name in names]
    # 如果 thlist 中只有一个对象，直接返回该对象
    if len(thlist) == 1:
        return thlist[0]
    # 否则返回 thlist 列表
    return thlist


class TensExpr(Expr, ABC):
    """
    Abstract base class for tensor expressions

    Notes
    =====

    A tensor expression is an expression formed by tensors;
    currently the sums of tensors are distributed.

    A ``TensExpr`` can be a ``TensAdd`` or a ``TensMul``.

    ``TensMul`` objects are formed by products of component tensors,
    and include a coefficient, which is a SymPy expression.


    In the internal representation contracted indices are represented
    by ``(ipos1, ipos2, icomp1, icomp2)``, where ``icomp1`` is the position
    of the component tensor with contravariant index, ``ipos1`` is the
    slot which the index occupies in that component tensor.

    Contracted indices are therefore nameless in the internal representation.
    """

    _op_priority = 12.0  # 运算符优先级设为 12.0
    is_commutative = False  # 不可交换

    # 定义负号操作符重载方法，返回自身乘以 -1
    def __neg__(self):
        return self * S.NegativeOne

    # 定义绝对值操作符重载方法，但未实现，抛出 NotImplementedError 异常
    def __abs__(self):
        raise NotImplementedError

    # 定义加法操作符重载方法，返回两个表达式的和
    def __add__(self, other):
        return TensAdd(self, other).doit()

    # 定义右加法操作符重载方法，返回另一个表达式加上自身的和
    def __radd__(self, other):
        return TensAdd(other, self).doit()

    # 定义减法操作符重载方法，返回两个表达式的差
    def __sub__(self, other):
        return TensAdd(self, -other).doit()

    # 定义右减法操作符重载方法，返回另一个表达式减去自身的差
    def __rsub__(self, other):
        return TensAdd(other, -self).doit()

    # 定义乘法操作符重载方法，返回两个张量的乘积
    def __mul__(self, other):
        """
        Multiply two tensors using Einstein summation convention.

        Explanation
        ===========

        If the two tensors have an index in common, one contravariant
        and the other covariant, in their product the indices are summed

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t1 = p(m0)
        >>> t2 = q(-m0)
        >>> t1*t2
        p(L_0)*q(-L_0)
        """
        return TensMul(self, other).doit()

    # 定义右乘法操作符重载方法，返回另一个表达式乘以自身的乘积
    def __rmul__(self, other):
        return TensMul(other, self).doit()

    # 定义除法操作符重载方法，实现张量除以数值的操作
    def __truediv__(self, other):
        other = _sympify(other)
        if isinstance(other, TensExpr):
            raise ValueError('cannot divide by a tensor')
        return TensMul(self, S.One / other).doit()

    # 定义右除法操作符重载方法，不允许张量除以其他张量
    def __rtruediv__(self, other):
        raise ValueError('cannot divide by a tensor')
    # 警告：此方法已被弃用
    def __pow__(self, other):
        # 调用弃用数据函数
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 如果数据为空，则抛出数值错误
            if self.data is None:
                raise ValueError("No power without ndarray data.")
            # 导入张量积和张量收缩方法
            from .array import tensorproduct, tensorcontraction
            # 获取自由指标
            free = self.free
            # 获取多维数组数据
            marray = self.data
            # 获取多维数组的秩
            mdim = marray.rank()
            # 对每个度量进行张量收缩操作
            for metric in free:
                marray = tensorcontraction(
                    tensorproduct(
                        marray,
                        metric[0].tensor_index_type.data,
                        marray),
                    (0, mdim), (mdim+1, mdim+2)
                )
            # 返回数组的乘幂运算结果
            return marray ** (other * S.Half)

    # 此方法未实现
    def __rpow__(self, other):
        raise NotImplementedError

    # 抽象属性：无系数
    @property
    @abstractmethod
    def nocoeff(self):
        raise NotImplementedError("abstract method")

    # 抽象属性：系数
    @property
    @abstractmethod
    def coeff(self):
        raise NotImplementedError("abstract method")

    # 抽象方法：获取指标
    @abstractmethod
    def get_indices(self):
        raise NotImplementedError("abstract method")

    # 抽象方法：获取自由指标
    @abstractmethod
    def get_free_indices(self) -> list[TensorIndex]:
        raise NotImplementedError("abstract method")

    # 抽象方法：替换指标
    @abstractmethod
    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        raise NotImplementedError("abstract method")

    # 函数评估方法：替换指标并返回结果
    def fun_eval(self, *index_tuples):
        # 警告：此方法已被弃用
        deprecate_fun_eval()
        # 调用替换指标方法返回结果
        return self.substitute_indices(*index_tuples)

    # 获取矩阵表示，如果数据可用且维度不超过2，则返回矩阵
    def get_matrix(self):
        """
        DEPRECATED: do not use.

        Returns ndarray components data as a matrix, if components data are
        available and ndarray dimension does not exceed 2.
        """
        # 导入矩阵类
        from sympy.matrices.dense import Matrix
        # 警告：此方法已被弃用
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 如果秩在1到2之间
            if 0 < self.rank <= 2:
                # 获取数组的行数
                rows = self.data.shape[0]
                # 如果秩为2，则获取数组的列数；否则列数为1
                columns = self.data.shape[1] if self.rank == 2 else 1
                # 初始化矩阵列表
                if self.rank == 2:
                    mat_list = [] * rows  # 错误修复：应该为 mat_list = [[] for _ in range(rows)]
                    # 逐行逐列获取数组元素
                    for i in range(rows):
                        mat_list.append([])
                        for j in range(columns):
                            mat_list[i].append(self[i, j])
                else:
                    mat_list = [None] * rows
                    for i in range(rows):
                        mat_list[i] = self[i]
                # 返回矩阵对象
                return Matrix(mat_list)
            else:
                # 抛出未实现错误
                raise NotImplementedError(
                    "missing multidimensional reduction to matrix.")

    # 静态方法：获取指标排列的置换
    @staticmethod
    def _get_indices_permutation(indices1, indices2):
        return [indices1.index(i) for i in indices2]

    # 扩展方法：使用 _expand 函数扩展并执行
    def expand(self, **hints):
        return _expand(self, **hints).doit()

    # 内部方法：直接返回自身
    def _expand(self, **kwargs):
        return self
    # 返回当前 TensExpr 对象中所有自由指标的集合
    def _get_free_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_free_indices_set())
        return indset

    # 返回当前 TensExpr 对象中所有虚拟指标的集合
    def _get_dummy_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_dummy_indices_set())
        return indset

    # 返回当前 TensExpr 对象中所有指标的集合
    def _get_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_indices_set())
        return indset

    # 返回当前 TensExpr 对象中所有虚拟指标的迭代器
    @property
    def _iterate_dummy_indices(self):
        dummy_set = self._get_dummy_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in dummy_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    # 返回当前 TensExpr 对象中所有自由指标的迭代器
    @property
    def _iterate_free_indices(self):
        free_set = self._get_free_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in free_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    # 返回当前 TensExpr 对象中所有指标的迭代器
    @property
    def _iterate_indices(self):
        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    # 使用度量张量对数组进行收缩和置换，并返回结果
    @staticmethod
    def _contract_and_permute_with_metric(metric, array, pos, dim):
        # TODO: add possibility of metric after (spinors)
        from .array import tensorcontraction, tensorproduct, permutedims

        # 对数组进行张量收缩和张量积操作，再按照指定的维度进行维度置换
        array = tensorcontraction(tensorproduct(metric, array), (1, 2+pos))
        # 构造用于置换维度的置换数组
        permu = list(range(dim))
        permu[0], permu[pos] = permu[pos], permu[0]
        # 返回置换后的数组
        return permutedims(array, permu)
    # 定义一个函数，用于将一个数组的自由指标与另一个张量的指标匹配
    def _match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict):
        # 导入 permutedims 函数，用于对数组进行指标重新排序
        from .array import permutedims

        # 获取第一个自由指标列表中每个指标的类型
        index_types1 = [i.tensor_index_type for i in free_ind1]

        # 检查是否需要修正指标的变化：
        pos2up = []  # 用于存储需要提升的指标位置
        pos2down = []  # 用于存储需要降低的指标位置
        free2remaining = free_ind2[:]  # 复制第二个自由指标列表
        for pos1, index1 in enumerate(free_ind1):
            if index1 in free2remaining:  # 如果第二个列表中存在匹配的指标
                pos2 = free2remaining.index(index1)
                free2remaining[pos2] = None
                continue
            if -index1 in free2remaining:  # 如果存在反向的匹配指标
                pos2 = free2remaining.index(-index1)
                free2remaining[pos2] = None
                free_ind2[pos2] = index1  # 更新第二个列表中的指标
                if index1.is_up:
                    pos2up.append(pos2)  # 记录需要提升的指标位置
                else:
                    pos2down.append(pos2)  # 记录需要降低的指标位置
            else:
                index2 = free2remaining[pos1]
                if index2 is None:
                    raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))
                free2remaining[pos1] = None
                free_ind2[pos1] = index1
                if index1.is_up ^ index2.is_up:
                    if index1.is_up:
                        pos2up.append(pos1)
                    else:
                        pos2down.append(pos1)

        # 检查是否存在不兼容的指标
        if len(set(free_ind1) & set(free_ind2)) < len(free_ind1):
            raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))

        # 提升指标：
        for pos in pos2up:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            metric_inverse = _TensorDataLazyEvaluator.inverse_matrix(metric)
            array = TensExpr._contract_and_permute_with_metric(metric_inverse, array, pos, len(free_ind1))

        # 降低指标：
        for pos in pos2down:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            array = TensExpr._contract_and_permute_with_metric(metric, array, pos, len(free_ind1))

        # 如果还有剩余的自由指标，根据指标的排列顺序重新排序数组
        if free_ind1:
            permutation = TensExpr._get_indices_permutation(free_ind2, free_ind1)
            array = permutedims(array, permutation)

        # 如果数组具有 rank 属性且其 rank 为 0，则将其转换为标量值
        if hasattr(array, "rank") and array.rank() == 0:
            array = array[()]

        return free_ind2, array
    def _expand_partial_derivative(self):
        # 将 _expand_partial_derivative() 简单地委托给其参数，以展开可能找到的 PartialDerivative
        return self.func(*[
                    a._expand_partial_derivative()
                    if isinstance(a, TensExpr) else a
                    for a in self.args])

    def _matches_simple(self, expr, repl_dict=None, old=False):
        """
        Matches assuming there are no wild objects in self.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # 如果 expr 不是 TensExpr 类型
        if not isinstance(expr, TensExpr):
            # 如果 self 有自由指标，但 expr 没有
            if len(self.get_free_indices()) > 0:
                # 返回 None，表示无法匹配
                return None
        else:
            # 如果 self 和 expr 都不含通配符，并且它们的自由指标不相同，则无法匹配
            if set(self.get_free_indices()) != set(expr.get_free_indices()):
                # 返回 None，表示无法匹配
                return None

        # 如果 canon_bp(self - expr) 等于零，说明表达式相等，返回 repl_dict
        if canon_bp(self - expr) == S.Zero:
            return repl_dict
        else:
            # 否则返回 None，表示无法匹配
            return None
    """
    Sum of tensors.

    Parameters
    ==========

    free_args : list of the free indices

    Attributes
    ==========

    ``args`` : tuple of addends
    ``rank`` : rank of the tensor
    ``free_args`` : list of the free indices in sorted order

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_heads, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b = tensor_indices('a,b', Lorentz)
    >>> p, q = tensor_heads('p,q', [Lorentz])
    >>> t = p(a) + q(a); t
    p(a) + q(a)

    Examples with components data added to the tensor expression:

    >>> from sympy import symbols, diag
    >>> x, y, z, t = symbols("x y z t")
    >>> repl = {}
    >>> repl[Lorentz] = diag(1, -1, -1, -1)
    >>> repl[p(a)] = [1, 2, 3, 4]
    >>> repl[q(a)] = [x, y, z, t]

    The following are: 2**2 - 3**2 - 2**2 - 7**2 ==> -58

    >>> expr = p(a) + q(a)
    >>> expr.replace_with_arrays(repl, [a])
    [x + 1, y + 2, z + 3, t + 4]
    """

    def __new__(cls, *args, **kw_args):
        # 将所有参数转化为 Sympy 的表达式对象
        args = [_sympify(x) for x in args if x]
        # 对参数进行展平处理
        args = TensAdd._tensAdd_flatten(args)
        # 根据默认排序键对参数进行排序
        args.sort(key=default_sort_key)
        if not args:
            return S.Zero
        if len(args) == 1:
            return args[0]

        # 调用基类的构造方法，返回一个新的 TensAdd 对象
        return Basic.__new__(cls, *args, **kw_args)

    @property
    def coeff(self):
        # 返回标量系数 1
        return S.One

    @property
    def nocoeff(self):
        # 返回自身，表示没有乘以额外的系数
        return self

    def get_free_indices(self) -> list[TensorIndex]:
        # 返回自由指标的列表
        return self.free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # 用给定的替换字典替换张量表达式中的指标
        newargs = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args]
        return self.func(*newargs)

    @memoize_property
    def rank(self):
        # 如果第一个参数是 TensExpr 对象，则返回其秩，否则返回 0
        if isinstance(self.args[0], TensExpr):
            return self.args[0].rank
        else:
            return 0

    @memoize_property
    def free_args(self):
        # 如果第一个参数是 TensExpr 对象，则返回其自由指标列表，否则返回空列表
        if isinstance(self.args[0], TensExpr):
            return self.args[0].free_args
        else:
            return []

    @memoize_property
    def free_indices(self):
        # 如果第一个参数是 TensExpr 对象，则返回其自由指标集合，否则返回空集合
        if isinstance(self.args[0], TensExpr):
            return self.args[0].get_free_indices()
        else:
            return set()
    def doit(self, **hints):
        # 从 hints 字典中获取 deep 键的值，默认为 True
        deep = hints.get('deep', True)
        # 如果 deep 为 True，则递归调用 self.args 中每个参数的 doit 方法
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args

        # 过滤掉 args 中的零值参数 S.Zero
        args = [arg for arg in args if arg != S.Zero]

        # 如果 args 为空列表，返回零 S.Zero
        if len(args) == 0:
            return S.Zero
        # 如果 args 只包含一个元素，直接返回该元素
        elif len(args) == 1:
            return args[0]

        # 检查所有加数是否具有相同的索引结构
        TensAdd._tensAdd_check(args)

        # 收集出现多次且系数不同的项
        args = TensAdd._tensAdd_collect_terms(args)

        # 定义排序关键字函数 sort_key
        def sort_key(t):
            # 如果 t 不是 TensExpr 类型，返回空元组
            if not isinstance(t, TensExpr):
                return [], [], []
            # 如果 t 具有 "_index_structure" 属性和 "components" 属性，则调用 get_index_structure 函数获取索引结构
            if hasattr(t, "_index_structure") and hasattr(t, "components"):
                x = get_index_structure(t)
                return t.components, x.free, x.dum
            return [], [], []

        # 根据 sort_key 对 args 列表进行排序
        args.sort(key=sort_key)

        # 如果 args 为空列表，返回零 S.Zero
        if not args:
            return S.Zero
        # 如果 args 只包含一个元素，直接返回该元素
        if len(args) == 1:
            return args[0]

        # 使用 self.func 构造一个新的对象 obj，并返回该对象
        obj = self.func(*args)
        return obj

    @staticmethod
    def _tensAdd_flatten(args):
        # 展开 TensAdd 对象，将非张量项转换为张量项
        a = []
        for x in args:
            if isinstance(x, (Add, TensAdd)):
                a.extend(list(x.args))
            else:
                a.append(x)
        # 过滤掉系数为零的项，并返回处理后的列表 args
        args = [x for x in a if x.coeff]
        return args

    @staticmethod
    def _tensAdd_check(args):
        # 检查所有加数的自由指标是否相同

        # 获取表达式 x 的自由指标集合
        def get_indices_set(x: Expr) -> set[TensorIndex]:
            if isinstance(x, TensExpr):
                return set(x.get_free_indices())
            return set()

        # 获取 args 中第一个元素的自由指标集合
        indices0 = get_indices_set(args[0])
        # 获取除第一个元素外所有元素的自由指标集合，并存储在 list_indices 列表中
        list_indices = [get_indices_set(arg) for arg in args[1:]]
        # 如果 list_indices 中任意一个集合与 indices0 不相等，抛出 ValueError 异常
        if not all(x == indices0 for x in list_indices):
            raise ValueError('all tensors must have the same indices')

    @staticmethod
    def _tensAdd_collect_terms(args):
        # 用于收集 TensMul 项，这些项最多只有系数不同
        terms_dict = defaultdict(list)
        scalars = S.Zero
        if isinstance(args[0], TensExpr):
            free_indices = set(args[0].get_free_indices())
        else:
            free_indices = set()

        for arg in args:
            if not isinstance(arg, TensExpr):
                # 如果参数不是张量表达式，检查自由指标是否为空，然后累加到scalars
                if free_indices != set():
                    raise ValueError("wrong valence")
                scalars += arg
                continue
            # 如果参数是张量表达式，检查其自由指标是否与第一个参数一致
            if free_indices != set(arg.get_free_indices()):
                raise ValueError("wrong valence")
            # TODO: 这部分不是系数的部分是什么？
            # 需要类似于 .as_coeff_Mul() 的实现
            # 将没有系数的项(arg.nocoeff)按照系数(arg.coeff)进行分组
            terms_dict[arg.nocoeff].append(arg.coeff)

        # 对于每组项，创建新的 TensMul 对象，系数为该组的和，并尝试求值
        new_args = [TensMul(Add(*coeff), t).doit() for t, coeff in terms_dict.items() if Add(*coeff) != 0]
        if isinstance(scalars, Add):
            # 如果scalars是一个加法表达式，将其参数加入到new_args中
            new_args = list(scalars.args) + new_args
        elif scalars != 0:
            # 如果scalars不为零，将其作为新的第一个参数加入new_args中
            new_args = [scalars] + new_args
        return new_args

    def get_indices(self):
        # 获取所有参数的索引，确保没有重复
        indices = []
        for arg in self.args:
            indices.extend([i for i in get_indices(arg) if i not in indices])
        return indices

    def _expand(self, **hints):
        # 对 TensAdd 对象进行展开，返回展开后的结果
        return TensAdd(*[_expand(i, **hints) for i in self.args])

    def __call__(self, *indices):
        # 对象被调用时的行为，已弃用
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        # 检查传入的索引是否与自由参数的张量类型一致
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        # 如果传入的索引与自由参数一致，返回对象自身
        if indices == free_args:
            return self
        # 将自由参数与传入的索引组成元组，并替换原对象的索引生成新对象
        index_tuples = list(zip(free_args, indices))
        a = [x.func(*x.substitute_indices(*index_tuples).args) for x in self.args]
        # 对生成的新 TensAdd 对象进行求值并返回结果
        res = TensAdd(*a).doit()
        return res

    def canon_bp(self):
        """
        使用 Butler-Portugal 算法对具有单项对称性的对象进行规范化。
        """
        # 对当前对象进行展开并规范化
        expr = self.expand()
        args = [canon_bp(x) for x in expr.args]
        res = TensAdd(*args).doit()
        return res

    def equals(self, other):
        # 判断当前对象与另一个对象是否相等
        other = _sympify(other)
        if isinstance(other, TensMul) and other.coeff == 0:
            return all(x.coeff == 0 for x in self.args)
        if isinstance(other, TensExpr):
            if self.rank != other.rank:
                return False
        if isinstance(other, TensAdd):
            # 如果是 TensAdd 对象，检查其参数集合是否相同
            if set(self.args) != set(other.args):
                return False
            else:
                return True
        # 如果不是上述情况，计算两个对象的差并判断是否为零
        t = self - other
        if not isinstance(t, TensExpr):
            return t == 0
        else:
            if isinstance(t, TensMul):
                return t.coeff == 0
            else:
                return all(x.coeff == 0 for x in t.args)
    # 根据索引获取元素的方法，触发数据弃用警告
    def __getitem__(self, item):
        # 调用数据弃用函数
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 返回索引为 item 的数据
            return self.data[item]

    # 使用指定的 delta 对张量进行缩并运算
    def contract_delta(self, delta):
        # 对每个参数进行 delta 缩并运算
        args = [x.contract_delta(delta) for x in self.args]
        # 对结果进行完全展开并规范化
        t = TensAdd(*args).doit()
        # 规范化缩并结果
        return canon_bp(t)

    # 使用指定的度量张量 g 进行度量收缩运算
    def contract_metric(self, g):
        """
        使用度量张量 ``g`` 进行指标升降操作。

        Parameters
        ==========

        g :  度量张量

        contract_all : 如果为 True，则消除所有被缩并的 ``g`` 张量

        Notes
        =====

        请参考 ``TensorIndexType`` 文档字符串了解缩并约定
        """
        # 对每个参数进行度量张量收缩运算
        args = [contract_metric(x, g) for x in self.args]
        # 对结果进行完全展开并规范化
        t = TensAdd(*args).doit()
        # 规范化缩并结果
        return canon_bp(t)

    # 替换张量表达式中的所有索引
    def substitute_indices(self, *index_tuples):
        # 对每个参数进行索引替换操作
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        # 对结果进行完全展开并规范化
        return TensAdd(*new_args).doit()

    # 将张量表达式转换为字符串输出
    def _print(self):
        # 将每个参数转换为字符串并存储在列表中
        a = []
        args = self.args
        for x in args:
            a.append(str(x))
        # 将所有参数字符串连接起来，使用 ' + ' 分隔
        s = ' + '.join(a)
        # 替换 '+ -' 为 '- '，并返回结果字符串
        s = s.replace('+ -', '- ')
        return s

    # 从替换字典中提取数据
    def _extract_data(self, replacement_dict):
        from sympy.tensor.array import Array, permutedims
        # 提取每个参数的数据
        args_indices, arrays = zip(*[
            arg._extract_data(replacement_dict) if
            isinstance(arg, TensExpr) else ([], arg) for arg in self.args
        ])
        # 将非张量表达式参数转换为数组
        arrays = [Array(i) for i in arrays]
        # 参考索引为第一个参数的索引
        ref_indices = args_indices[0]
        # 对每个参数执行维度排列
        for i in range(1, len(args_indices)):
            indices = args_indices[i]
            array = arrays[i]
            permutation = TensMul._get_indices_permutation(indices, ref_indices)
            arrays[i] = permutedims(array, permutation)
        # 返回参考索引和所有数组求和后的结果
        return ref_indices, sum(arrays, Array.zeros(*array.shape))

    # 返回张量的数据属性
    @property
    def data(self):
        # 触发数据弃用警告
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 返回对应张量表达式扩展后的数据字典
            return _tensor_data_substitution_dict[self.expand()]

    # 设置张量的数据属性
    @data.setter
    def data(self, data):
        # 触发数据弃用警告
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 将数据存储到对应张量的替换字典中
            _tensor_data_substitution_dict[self] = data

    # 删除张量的数据属性
    @data.deleter
    def data(self):
        # 触发数据弃用警告
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告
        with ignore_warnings(SymPyDeprecationWarning):
            # 如果张量存在于替换字典中，则删除其数据
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]

    # 返回张量的迭代器
    def __iter__(self):
        # 触发数据弃用警告
        deprecate_data()
        # 如果数据为空，则抛出 ValueError 异常
        if not self.data:
            raise ValueError("No iteration on abstract tensors")
        # 返回数据的展平迭代器
        return self.data.flatten().__iter__()

    # 重写方法，将表达式重写为 Indexed 对象
    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        return Add.fromiter(args)
    # 定义一个方法 `_eval_partial_derivative`，用于计算偏导数
    def _eval_partial_derivative(self, s):
        # 初始化一个空列表，用于存储偏导数表达式的各个部分
        list_addends = []
        # 遍历表达式中的每个元素
        for a in self.args:
            # 如果当前元素是一个张量表达式，则递归计算其关于变量 s 的偏导数
            if isinstance(a, TensExpr):
                list_addends.append(a._eval_partial_derivative(s))
            # 如果 s 可以作为符号参与微分，则计算当前元素关于 s 的导数
            elif s._diff_wrt:
                list_addends.append(a._eval_derivative(s))

        # 将计算得到的各部分重新组合成一个新的表达式，并返回结果
        return self.func(*list_addends)
class Tensor(TensExpr):
    """
    Base tensor class, i.e. this represents a tensor, the single unit to be
    put into an expression.

    Explanation
    ===========

    This object is usually created from a ``TensorHead``, by attaching indices
    to it. Indices preceded by a minus sign are considered contravariant,
    otherwise covariant.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
    >>> Lorentz = TensorIndexType("Lorentz", dummy_name="L")
    >>> mu, nu = tensor_indices('mu nu', Lorentz)
    >>> A = TensorHead("A", [Lorentz, Lorentz])
    >>> A(mu, -nu)
    A(mu, -nu)
    >>> A(mu, -mu)
    A(L_0, -L_0)

    It is also possible to use symbols instead of inidices (appropriate indices
    are then generated automatically).

    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> A(x, mu)
    A(x, mu)
    >>> A(x, -x)
    A(L_0, -L_0)

    """

    is_commutative = False

    _index_structure = None  # type: _IndexStructure
    args: tuple[TensorHead, Tuple]

    def __new__(cls, tensor_head, indices, *, is_canon_bp=False, **kw_args):
        # 解析索引，确保正确的索引结构
        indices = cls._parse_indices(tensor_head, indices)
        # 创建新的 Tensor 对象
        obj = Basic.__new__(cls, tensor_head, Tuple(*indices), **kw_args)
        # 从索引中构建索引结构对象
        obj._index_structure = _IndexStructure.from_indices(*indices)
        # 设置自由和哑指标
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        # 初始化系数为 1
        obj._coeff = S.One
        # 设置无系数形式
        obj._nocoeff = obj
        # 设置组件和主组件
        obj._component = tensor_head
        obj._components = [tensor_head]
        # 如果张量头的秩与索引数不符，则引发 ValueError 异常
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        # 是否为标准基底点 (is_canon_bp) 设置为假
        obj.is_canon_bp = is_canon_bp
        # 构建索引映射
        obj._index_map = Tensor._build_index_map(indices, obj._index_structure)
        return obj

    @property
    def free(self):
        # 返回自由指标
        return self._free

    @property
    def dum(self):
        # 返回哑指标
        return self._dum

    @property
    def ext_rank(self):
        # 返回扩展秩
        return self._ext_rank

    @property
    def coeff(self):
        # 返回系数
        return self._coeff

    @property
    def nocoeff(self):
        # 返回无系数形式
        return self._nocoeff

    @property
    def component(self):
        # 返回组件
        return self._component

    @property
    def components(self):
        # 返回所有组件
        return self._components

    @property
    def head(self):
        # 返回头部
        return self.args[0]

    @property
    def indices(self):
        # 返回所有索引
        return self.args[1]

    @property
    def free_indices(self):
        # 返回自由索引的集合
        return set(self._index_structure.get_free_indices())

    @property
    def index_types(self):
        # 返回索引类型
        return self.head.index_types

    @property
    def rank(self):
        # 返回自由索引的数量，即秩
        return len(self.free_indices)

    @staticmethod
    def _build_index_map(indices, index_structure):
        # 构建索引映射字典
        index_map = {}
        for idx in indices:
            index_map[idx] = (indices.index(idx),)
        return index_map
    def doit(self, **hints):
        # 调用 TensMul 类的静态方法 _tensMul_contract_indices，处理当前对象 self，并返回结果 args, indices, free, dum
        args, indices, free, dum = TensMul._tensMul_contract_indices([self])
        # 返回处理结果中的第一个元素
        return args[0]

    @staticmethod
    def _parse_indices(tensor_head, indices):
        # 检查 indices 是否为 tuple、list 或 Tuple 类型之一，否则抛出类型错误异常
        if not isinstance(indices, (tuple, list, Tuple)):
            raise TypeError("indices should be an array, got %s" % type(indices))
        # 将 indices 转换为列表形式
        indices = list(indices)
        # 遍历索引列表，根据元素类型进行相应处理
        for i, index in enumerate(indices):
            if isinstance(index, Symbol):
                # 如果索引是 Symbol 类型，则用 TensorIndex 类重新包装，并标记为 covariant=True
                indices[i] = TensorIndex(index, tensor_head.index_types[i], True)
            elif isinstance(index, Mul):
                # 如果索引是 Mul 类型，则尝试解析其系数和乘积项，并据此判断是否需要转换为 TensorIndex
                c, e = index.as_coeff_Mul()
                if c == -1 and isinstance(e, Symbol):
                    indices[i] = TensorIndex(e, tensor_head.index_types[i], False)
                else:
                    raise ValueError("index not understood: %s" % index)
            elif not isinstance(index, TensorIndex):
                # 如果索引不是 TensorIndex 类型，则抛出类型错误异常
                raise TypeError("wrong type for index: %s is %s" % (index, type(index)))
        # 返回处理后的索引列表
        return indices

    def _set_new_index_structure(self, im, is_canon_bp=False):
        # 获取索引映射对象 im 的索引列表
        indices = im.get_indices()
        # 调用 _set_indices 方法，传入 indices 作为参数，并返回结果
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        # 如果 indices 的长度与对象的 ext_rank 属性不匹配，则抛出值错误异常
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        # 调用对象的 func 方法，传入 self.args[0]、indices 和 is_canon_bp，并调用其 doit 方法，返回结果
        return self.func(self.args[0], indices, is_canon_bp=is_canon_bp).doit()

    def _get_free_indices_set(self):
        # 获取对象的 _index_structure 属性中的自由索引集合，并返回
        return {i[0] for i in self._index_structure.free}

    def _get_dummy_indices_set(self):
        # 获取对象的 _index_structure 属性中的虚拟索引位置集合，构成虚拟索引集合，并返回
        dummy_pos = set(itertools.chain(*self._index_structure.dum))
        return {idx for i, idx in enumerate(self.args[1]) if i in dummy_pos}

    def _get_indices_set(self):
        # 获取对象的 args[1] 属性的参数集合，并返回为集合形式
        return set(self.args[1].args)

    @property
    def free_in_args(self):
        # 返回对象的 free 属性中每个元素的第一个元素构成的列表
        return [(ind, pos, 0) for ind, pos in self.free]

    @property
    def dum_in_args(self):
        # 返回对象的 dum 属性中每个元素的两个元素构成的列表
        return [(p1, p2, 0, 0) for p1, p2 in self.dum]

    @property
    def free_args(self):
        # 返回对象的 free 属性中第一个元素构成的排序列表
        return sorted([x[0] for x in self.free])

    def commutes_with(self, other):
        """
        检查当前对象与另一个 TensExpr 对象 other 的交换性：
        :param other: 另一个 TensExpr 对象
        :return:
            0  表示交换
            1  表示反交换
            None  既不交换也不反交换
        """
        # 如果 other 不是 TensExpr 类型，则返回 0
        if not isinstance(other, TensExpr):
            return 0
        # 如果 other 是 Tensor 类型，则检查当前对象的 component 属性与 other 的 component 属性的交换性
        elif isinstance(other, Tensor):
            return self.component.commutes_with(other.component)
        # 其他情况抛出未实现错误
        return NotImplementedError

    def perm2tensor(self, g, is_canon_bp=False):
        """
        返回与置换 g 对应的张量。

        进一步细节参见 TIDS 类中同名方法。
        """
        # 调用 perm2tensor 函数，传入 self、g 和 is_canon_bp 参数，并返回结果
        return perm2tensor(self, g, is_canon_bp)
    def canon_bp(self):
        # 如果已经是标准形式，直接返回自身
        if self.is_canon_bp:
            return self
        # 对表达式进行展开
        expr = self.expand()
        # 获取指数结构的规范参数
        g, dummies, msym = expr._index_structure.indices_canon_args()
        # 获取组件的规范参数
        v = components_canon_args([expr.component])
        # 规范化操作
        can = canonicalize(g, dummies, msym, *v)
        # 如果规范化结果为零，则返回零
        if can == 0:
            return S.Zero
        # 将排列转换为张量
        tensor = self.perm2tensor(can, True)
        return tensor

    def split(self):
        # 返回一个包含自身的列表，用于表示分割后的结果
        return [self]

    def _expand(self, **kwargs):
        # 返回自身，表示不进行任何展开操作
        return self

    def sorted_components(self):
        # 返回自身，表示不进行任何排序操作
        return self

    def get_indices(self) -> list[TensorIndex]:
        """
        获取张量的索引列表。
        """
        return list(self.args[1])

    def get_free_indices(self) -> list[TensorIndex]:
        """
        获取自由索引的列表。
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        """
        使用指定的替换字典替换张量的索引。
        """
        # TODO: 可以通过仅交换索引而不访问整个表达式树来优化此过程：
        return self.xreplace(repl)

    def as_base_exp(self):
        # 返回自身作为基础和指数1的元组
        return self, S.One

    def substitute_indices(self, *index_tuples):
        """
        根据 ``index_tuples`` 替换张量的自由索引。

        ``index_tuples`` 是 ``(旧索引, 新索引)`` 组成的元组列表。

        例子
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(i, k)*B(-k, -j); t
        A(i, L_0)*B(-L_0, -j)
        >>> t.substitute_indices((i, k),(-j, l))
        A(k, L_0)*B(-L_0, l)
        """
        indices = []
        for index in self.indices:
            for ind_old, ind_new in index_tuples:
                if (index.name == ind_old.name and index.tensor_index_type ==
                                                   ind_old.tensor_index_type):
                    if index.is_up == ind_old.is_up:
                        indices.append(ind_new)
                    else:
                        indices.append(-ind_new)
                    break
            else:
                indices.append(index)
        return self.head(*indices)
    def _get_symmetrized_forms(self):
        """
        Return a list giving all possible permutations of self that are allowed by its symmetries.
        返回一个列表，包含所有由该张量的对称性允许的自身可能的排列组合。
        """
        comp = self.component
        # 获取张量的组件
        gens = comp.symmetry.generators
        # 获取张量的对称性生成器
        rank = comp.rank
        # 获取张量的秩

        old_perms = None
        new_perms = {self}
        while new_perms != old_perms:
            old_perms = new_perms.copy()
            for tens in old_perms:
                for gen in gens:
                    inds = tens.get_indices()
                    per = [gen.apply(i) for i in range(0,rank)]
                    # 应用生成器获取索引的排列
                    sign = (-1)**(gen.apply(rank) - rank)
                    # 计算排列的符号
                    ind_map = dict(zip(inds, [inds[i] for i in per]))
                    # 构建索引映射字典
                    new_perms.add( sign * tens._replace_indices(ind_map) )
                    # 添加新的对称形式到集合中

        return new_perms
        # 返回所有可能的对称形式的集合

    def matches(self, expr, repl_dict=None, old=False):
        """
        Return a dictionary mapping indices of self to indices of expr if they match under symmetries.
        如果在对称性下匹配，返回一个字典，将self的索引映射到expr的索引。
        """
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # simple checks
        if self == expr:
            return repl_dict
            # 如果self与expr相等，则返回替换字典
        if not isinstance(expr, Tensor):
            return None
            # 如果expr不是Tensor的实例，则返回None
        if self.head != expr.head:
            return None
            # 如果self的头部与expr的头部不相同，则返回None

        # Now consider all index symmetries of expr, and see if any of them allow a match.
        # 现在考虑expr的所有索引对称性，看看它们是否允许匹配。
        for new_expr in expr._get_symmetrized_forms():
            m = self._matches(new_expr, repl_dict, old=old)
            # 对expr的所有对称形式进行匹配
            if m is not None:
                repl_dict.update(m)
                # 更新替换字典
                return repl_dict

        return None
        # 如果没有找到匹配的对称形式，则返回None

    def _matches(self, expr, repl_dict=None, old=False):
        """
        This does not account for index symmetries of expr
        这个方法不考虑expr的索引对称性。
        """
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # simple checks
        if self == expr:
            return repl_dict
            # 如果self与expr相等，则返回替换字典
        if not isinstance(expr, Tensor):
            return None
            # 如果expr不是Tensor的实例，则返回None
        if self.head != expr.head:
            return None
            # 如果self的头部与expr的头部不相同，则返回None

        s_indices = self.get_indices()
        e_indices = expr.get_indices()

        if len(s_indices) != len(e_indices):
            return None
            # 如果张量的索引数不相等，则返回None

        for i in range(len(s_indices)):
            s_ind = s_indices[i]
            m = s_ind.matches(e_indices[i])
            # 对每个索引进行匹配
            if m is None:
                return None
                # 如果匹配失败，则返回None
            elif -s_ind in repl_dict.keys() and -repl_dict[-s_ind] != m[s_ind]:
                return None
                # 如果已经存在相反的索引，并且它们不匹配，则返回None
            else:
                repl_dict.update(m)
                # 更新替换字典

        return repl_dict
        # 返回更新后的替换字典
    # 当实例被调用时执行的方法
    def __call__(self, *indices):
        # 调用过时的函数来提醒
        deprecate_call()
        # 获取自由指标
        free_args = self.free_args
        # 将传入的索引参数转换为列表
        indices = list(indices)
        # 检查传入的索引类型与自由指标的类型是否一致
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        # 如果传入的索引与自由指标相同，则返回自身
        if indices == free_args:
            return self
        # 使用自由指标和传入的索引替换张量表达式中的指标
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # 重建对象，确保所有缩并的指标被识别为哑指标，但仅当存在缩并的指标时。
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    # TODO: 将此方法放入TensExpr类中？
    def __iter__(self):
        # 调用过时的数据方法来提醒
        deprecate_data()
        # 忽略SymPy的过时警告
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    # TODO: 将此方法放入TensExpr类中？
    def __getitem__(self, item):
        # 调用过时的数据方法来提醒
        deprecate_data()
        # 忽略SymPy的过时警告
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    # 从替换字典中提取数据
    def _extract_data(self, replacement_dict):
        # 导入Array类
        from .array import Array
        # 遍历替换字典中的键值对
        for k, v in replacement_dict.items():
            # 如果键是Tensor类型且与当前对象的第一个参数相同
            if isinstance(k, Tensor) and k.args[0] == self.args[0]:
                # 将当前对象标记为other，并将v标记为array
                other = k
                array = v
                break
        else:
            # 如果未找到匹配的替换项，则引发值错误异常
            raise ValueError("%s not found in %s" % (self, replacement_dict))

        # TODO: 效率低下，应该仅在根级别执行此操作：
        # 使用Array类重新构建替换字典中的每个项
        replacement_dict = {k: Array(v) for k, v in replacement_dict.items()}
        # 将array标记为Array对象
        array = Array(array)

        # 获取当前对象的哑指标和other的哑指标
        dum1 = self.dum
        dum2 = other.dum

        # 如果dum2的长度大于0，则执行以下操作
        if len(dum2) > 0:
            for pair in dum2:
                # 如果pair不在dum1中，则引发未实现错误异常，包含other的缩并
                if pair not in dum1:
                    raise NotImplementedError("%s with contractions is not implemented" % other)
            # 从dum1中移除dum2中的元素
            dum1 = [pair for pair in dum1 if pair not in dum2]
        
        # 如果dum1的长度大于0，则执行以下操作
        if len(dum1) > 0:
            # 获取当前对象和other的索引
            indices1 = self.get_indices()
            indices2 = other.get_indices()
            repl = {}
            # 遍历dum1中的每一对索引
            for p1, p2 in dum1:
                # 将other的索引p2替换为-self的索引p1
                repl[indices2[p2]] = -indices2[p1]
                # 如果indices1和indices2中的位置p1或p2的is_up属性不相同
                for pos in (p1, p2):
                    if indices1[pos].is_up ^ indices2[pos].is_up:
                        metric = replacement_dict[indices1[pos].tensor_index_type]
                        if indices1[pos].is_up:
                            metric = _TensorDataLazyEvaluator.inverse_matrix(metric)
                        array = self._contract_and_permute_with_metric(metric, array, pos, len(indices2))
            # 使用repl替换other并完成操作
            other = other.xreplace(repl).doit()
            # 使用数据合同哑指标的_LazyEvaluator方法，对array执行操作
            array = _TensorDataLazyEvaluator.data_contract_dum([array], dum1, len(indices2))

        # 获取当前对象和other的自由指标
        free_ind1 = self.get_free_indices()
        free_ind2 = other.get_free_indices()

        # 将匹配的索引与other张量进行操作
        return self._match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict)
    # 返回私有属性的属性函数，标记为已弃用
    @property
    def data(self):
        # 调用函数来弃用数据
        deprecate_data()
        # 忽略 SymPyDeprecationWarning 警告并返回数据替换字典中对应的值
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    # 设置私有属性的属性函数，标记为已弃用
    @data.setter
    def data(self, data):
        # 调用函数来弃用数据
        deprecate_data()
        # TODO: 检查数据与张量属性的兼容性
        with ignore_warnings(SymPyDeprecationWarning):
            # 设置数据替换字典中对应的值为给定数据
            _tensor_data_substitution_dict[self] = data

    # 删除私有属性的属性函数，标记为已弃用
    @data.deleter
    def data(self):
        # 调用函数来弃用数据
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            # 如果张量存在于数据替换字典中，则删除对应项
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]
            # 如果张量的度规存在于数据替换字典中，则删除对应项
            if self.metric in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self.metric]

    # 私有方法，用于打印张量的字符串表示
    def _print(self):
        # 将张量的指标转换为字符串列表
        indices = [str(ind) for ind in self.indices]
        # 获取张量的组件
        component = self.component
        # 如果组件的秩大于0，返回格式化后的字符串表示
        if component.rank > 0:
            return ('%s(%s)' % (component.name, ', '.join(indices)))
        # 否则，仅返回组件的名称
        else:
            return ('%s' % component.name)

    # 判断当前张量与另一个张量是否相等
    def equals(self, other):
        # 如果另一个张量为0，则比较当前张量的系数是否为0
        if other == 0:
            return self.coeff == 0
        # 将另一个张量转换为 Sympy 对象
        other = _sympify(other)
        # 如果另一个对象不是张量表达式，则断言当前张量没有组件，并返回比较结果
        if not isinstance(other, TensExpr):
            assert not self.components
            return S.One == other

        # 内部函数，用于获取当前张量的比较对象
        def _get_compar_comp(self):
            # 规范化当前张量并返回其系数、组件、自由指标和哑指标的元组
            t = self.canon_bp()
            r = (t.coeff, tuple(t.components), \
                    tuple(sorted(t.free)), tuple(sorted(t.dum)))
            return r

        # 比较当前张量与另一个张量的比较对象是否相等
        return _get_compar_comp(self) == _get_compar_comp(other)

    # 将当前张量与给定的度规收缩
    def contract_metric(self, g):
        # 如果当前张量的组件与度规不同，则直接返回当前张量
        if self.component != g:
            return self
        # 如果存在自由指标，则直接返回当前张量
        if len(self.free) != 0:
            return self

        # 根据度规的对称性确定反对称性
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError
        sign = S.One
        typ = g.index_types[0]

        # 如果不存在反对称性，则计算张量的度规
        if not antisym:
            # g(i, -i)
            sign = sign*typ.dim
        else:
            # g(i, -i)
            sign = sign*typ.dim

            dp0, dp1 = self.dum[0]
            if dp0 < dp1:
                # g(i, -i) = -D with antisymmetric metric
                sign = -sign

        return sign

    # 使用度规进行 delta 收缩
    def contract_delta(self, metric):
        return self.contract_metric(metric)

    # 将当前对象重写为 Indexed 对象
    def _eval_rewrite_as_Indexed(self, tens, indices, **kwargs):
        from sympy.tensor.indexed import Indexed
        # TODO: 将 .args[0] 替换为 .name：
        # 获取当前对象的指标符号列表
        index_symbols = [i.args[0] for i in self.get_indices()]
        # 创建并返回 Indexed 对象
        expr = Indexed(tens.args[0], *index_symbols)
        return self._check_add_Sum(expr, index_symbols)
    # 定义一个方法用于计算偏导数，返回表达式对象
    def _eval_partial_derivative(self, s):  # type: (Tensor) -> Expr
        # 如果输入参数不是 Tensor 对象，则返回零表达式
        if not isinstance(s, Tensor):
            return S.Zero
        else:
            # 若头部不相同，则返回零表达式
            if self.head != s.head:
                return S.Zero

            # 如果头部相同，提供 delta 和/或度量乘积
            # 对于适当张量中每个自由指标对
            # 假定自由指标顺序正确
            # 派生后，导数中的逆变指标变成协变指标，反之亦然

            kronecker_delta_list = [1]

            # 遍历并比较自身和目标张量的自由指标对
            for (count, (iself, iother)) in enumerate(zip(self.get_free_indices(), s.get_free_indices())):
                # 如果自身和目标的指标类型不兼容，则引发 ValueError
                if iself.tensor_index_type != iother.tensor_index_type:
                    raise ValueError("index types not compatible")
                else:
                    tensor_index_type = iself.tensor_index_type
                    tensor_metric = tensor_index_type.metric
                    dummy = TensorIndex("d_" + str(count), tensor_index_type,
                                        is_up=iself.is_up)
                    # 根据指标是否升降处理 Kronecker δ
                    if iself.is_up == iother.is_up:
                        kroneckerdelta = tensor_index_type.delta(iself, -iother)
                    else:
                        kroneckerdelta = (
                            TensMul(tensor_metric(iself, dummy),
                                    tensor_index_type.delta(-dummy, -iother))
                        )
                    kronecker_delta_list.append(kroneckerdelta)

            # 返回从 kronecker_delta_list 构建的 TensMul 表达式，并进行必要的计算
            return TensMul.fromiter(kronecker_delta_list).doit()
            # doit 方法用于相应地重命名虚指标
# 定义一个类 `TensMul`，继承自 `TensExpr` 和 `AssocOp`
class TensMul(TensExpr, AssocOp):
    """
    Product of tensors.

    Parameters
    ==========

    coeff : SymPy coefficient of the tensor
    args

    Attributes
    ==========

    ``components`` : list of ``TensorHead`` of the component tensors
    ``types`` : list of nonrepeated ``TensorIndexType``
    ``free`` : list of ``(ind, ipos, icomp)``, see Notes
    ``dum`` : list of ``(ipos1, ipos2, icomp1, icomp2)``, see Notes
    ``ext_rank`` : rank of the tensor counting the dummy indices
    ``rank`` : rank of the tensor
    ``coeff`` : SymPy coefficient of the tensor
    ``free_args`` : list of the free indices in sorted order
    ``is_canon_bp`` : ``True`` if the tensor in in canonical form

    Notes
    =====

    ``args[0]``   list of ``TensorHead`` of the component tensors.

    ``args[1]``   list of ``(ind, ipos, icomp)``
    where ``ind`` is a free index, ``ipos`` is the slot position
    of ``ind`` in the ``icomp``-th component tensor.

    ``args[2]`` list of tuples representing dummy indices.
    ``(ipos1, ipos2, icomp1, icomp2)`` indicates that the contravariant
    dummy index is the ``ipos1``-th slot position in the ``icomp1``-th
    component tensor; the corresponding covariant index is
    in the ``ipos2`` slot position in the ``icomp2``-th component tensor.

    """

    # 类变量，用于表示张量乘积的单位元
    identity = S.One

    # 类变量，用于存储索引结构信息，初始化为 None
    _index_structure = None  # type: _IndexStructure
    # 重载 __new__ 方法，用于创建新对象实例
    def __new__(cls, *args, **kw_args):
        # 获取关键字参数中的 is_canon_bp，如果不存在则默认为 False
        is_canon_bp = kw_args.get('is_canon_bp', False)
        # 对所有位置参数应用 _sympify 函数，并转换为列表
        args = list(map(_sympify, args))

        """
        如果某个参数中的内部虚指标与其他参数的自由指标冲突，需要重命名这些内部虚指标。
        """
        # 获取所有参数的自由指标并放入列表中
        free = [get_free_indices(arg) for arg in args]
        # 将列表中的所有集合合并成一个集合，以扁平化处理
        free = set(itertools.chain(*free)) #flatten free
        # 创建一个空列表，用于存储处理后的参数
        newargs = []
        # 遍历所有参数
        for arg in args:
            # 获取当前参数的虚指标，并放入集合 dum_this 中
            dum_this = set(get_dummy_indices(arg))
            # 获取已处理参数列表 newargs 中所有参数的虚指标，放入列表 dum_other 中
            dum_other = [get_dummy_indices(a) for a in newargs]
            # 将列表 dum_other 中的所有集合合并成一个集合，以扁平化处理
            dum_other = set(itertools.chain(*dum_other)) #flatten dum_other
            # 获取当前参数的自由指标，并放入集合 free_this 中
            free_this = set(get_free_indices(arg))
            # 如果当前参数的虚指标与整体的自由指标集合有交集
            if len(dum_this.intersection(free)) > 0:
                # 将当前参数中的指标重命名，排除 free_this、free 和 dum_other 中的指标
                exclude = free_this.union(free, dum_other)
                newarg = TensMul._dedupe_indices(arg, exclude)
            else:
                # 如果没有冲突，直接使用原始参数
                newarg = arg
            # 将处理后的参数加入到新参数列表中
            newargs.append(newarg)

        # 将参数列表更新为处理后的新参数列表
        args = newargs

        # 将所有参数扁平化处理为一个列表
        args = [i for arg in args for i in (arg.args if isinstance(arg, (TensMul, Mul)) else [arg])]

        # 调用 _tensMul_contract_indices 方法，处理参数，返回处理后的 args、indices、free 和 dum
        args, indices, free, dum = TensMul._tensMul_contract_indices(args, replace_indices=False)

        # 创建 _IndexStructure 对象，用于存储索引相关的数据结构
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        # 使用父类 TensExpr 的 __new__ 方法创建对象实例
        obj = TensExpr.__new__(cls, *args)
        # 设置对象实例的索引相关属性
        obj._indices = indices
        obj._index_types = index_types[:]
        obj._index_structure = index_structure
        obj._free = index_structure.free[:]
        obj._dum = index_structure.dum[:]
        obj._free_indices = {x[0] for x in obj.free}
        obj._rank = len(obj.free)
        obj._ext_rank = len(obj._index_structure.free) + 2 * len(obj._index_structure.dum)
        obj._coeff = S.One
        obj._is_canon_bp = is_canon_bp
        # 返回创建的对象实例
        return obj

    # 定义 index_types 属性，返回对象实例的索引类型列表
    index_types = property(lambda self: self._index_types)
    # 定义 free 属性，返回对象实例的自由指标列表
    free = property(lambda self: self._free)
    # 定义 dum 属性，返回对象实例的虚指标列表
    dum = property(lambda self: self._dum)
    # 定义 free_indices 属性，返回对象实例的自由指标集合
    free_indices = property(lambda self: self._free_indices)
    # 定义 rank 属性，返回对象实例的秩（自由指标的数量）
    rank = property(lambda self: self._rank)
    # 定义 ext_rank 属性，返回对象实例的扩展秩（自由指标数量加上两倍的虚指标数量）
    ext_rank = property(lambda self: self._ext_rank)
    def _indices_to_free_dum(args_indices):
        # 初始化空字典，用于存储自由指标到其在 `args_indices` 中位置的映射关系
        free2pos1 = {}
        # 初始化空字典，用于存储自由指标到其在整个表达式中位置的映射关系
        free2pos2 = {}
        # 初始化空列表，用于存储虚指标的数据
        dummy_data = []
        # 初始化空列表，用于存储所有指标对象
        indices = []

        # 对整个表达式中的指标位置进行计数
        pos2 = 0

        # 遍历 `args_indices` 中的每个位置 `pos1` 和其对应的指标列表 `arg_indices`
        for pos1, arg_indices in enumerate(args_indices):

            # 遍历当前位置 `arg_indices` 中的每个指标 `index`
            for index in arg_indices:
                # 检查指标是否为 `TensorIndex` 类型，如果不是则抛出类型错误
                if not isinstance(index, TensorIndex):
                    raise TypeError("expected TensorIndex")
                
                # 检查指标是否为虚指标
                if -index in free2pos1:
                    # 检测到虚指标：找到对应的自由指标位置和整个表达式中的位置
                    other_pos1 = free2pos1.pop(-index)
                    other_pos2 = free2pos2.pop(-index)
                    # 根据指标的上下位置，添加到虚指标数据列表中
                    if index.is_up:
                        dummy_data.append((index, pos1, other_pos1, pos2, other_pos2))
                    else:
                        dummy_data.append((-index, other_pos1, pos1, other_pos2, pos2))
                    # 将指标对象添加到指标列表中
                    indices.append(index)
                elif index in free2pos1:
                    # 如果指标已经在自由指标映射中，则抛出重复指标错误
                    raise ValueError("Repeated index: %s" % index)
                else:
                    # 将自由指标和其位置映射到 `free2pos1` 和 `free2pos2` 中
                    free2pos1[index] = pos1
                    free2pos2[index] = pos2
                    # 将指标对象添加到指标列表中
                    indices.append(index)
                
                # 更新整个表达式中的指标位置计数
                pos2 += 1

        # 将 `free2pos2` 字典转换为列表形式
        free = list(free2pos2.items())
        # 获取所有自由指标的名称列表
        free_names = [i.name for i in free2pos2.keys()]

        # 根据虚指标的整体位置排序 `dummy_data` 列表，并返回四个组成部分
        dummy_data.sort(key=lambda x: x[3])
        return indices, free, free_names, dummy_data
    def _tensMul_contract_indices(args, replace_indices=True):
        replacements = [{} for _ in args]

        # 初始化一个计数器字典，用于统计每种张量索引类型的虚拟索引数目
        cdt = defaultdict(int)

        # 获取每个参数的索引信息
        args_indices = [get_indices(arg) for arg in args]

        # 将索引信息转换成自由指标、虚拟指标和其它信息
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)

        # 定义一个生成虚拟指标名称的函数
        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        # 如果需要替换索引
        if replace_indices:
            # 遍历每个需要替换的旧索引及其位置信息
            for old_index, pos1cov, pos1contra, pos2cov, pos2contra in dummy_data:
                index_type = old_index.tensor_index_type
                # 生成新的虚拟指标名称，确保其在自由指标名称中不存在
                while True:
                    dummy_name = dummy_name_gen(index_type)
                    if dummy_name not in free_names:
                        break
                # 创建新的虚拟指标对象，并更新替换字典及索引列表
                dummy = old_index.func(dummy_name, index_type, *old_index.args[2:])
                replacements[pos1cov][old_index] = dummy
                replacements[pos1contra][-old_index] = -dummy
                indices[pos2cov] = dummy
                indices[pos2contra] = -dummy

            # 更新参数列表中的索引替换结果
            args = [
                arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg
                for arg, repl in zip(args, replacements)]

            """
            由于替换索引可能改变索引的顺序（例如，如果某个参数是 TensAdd，替换一个索引可能改变项的排序顺序，从而改变其 get_indices() 方法返回的索引顺序）。
            为了保险起见，我们重新计算这些量。
            """
            # 重新计算参数的索引信息
            args_indices = [get_indices(arg) for arg in args]
            indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)

        # 将虚拟指标数据转换为虚拟指标对象列表
        dum = TensMul._dummy_data_to_dum(dummy_data)

        # 返回更新后的参数列表、索引、自由指标和虚拟指标
        return args, indices, free, dum

    @staticmethod
    def _get_components_from_args(args):
        """
        获取通过彼此相乘得到的具有相同 TIDS 的 Tensor 对象列表。
        """
        components = []
        for arg in args:
            # 如果参数不是 Tensor 表达式，则跳过
            if not isinstance(arg, TensExpr):
                continue
            # 如果参数是 TensAdd 对象，则跳过
            if isinstance(arg, TensAdd):
                continue
            # 将参数的 components 属性（Tensor 对象列表）加入结果列表
            components.extend(arg.components)
        # 返回组件列表
        return components

    @staticmethod
    def _rebuild_tensors_list(args, index_structure):
        # 获取索引结构的索引列表
        indices = index_structure.get_indices()

        # 遍历参数列表，并根据每个参数的属性创建新的 Tensor 对象
        ind_pos = 0
        for i, arg in enumerate(args):
            # 如果参数不是 Tensor 表达式，则跳过
            if not isinstance(arg, TensExpr):
                continue
            # 记录前一个位置和当前位置，创建新的 Tensor 对象并替换原参数
            prev_pos = ind_pos
            ind_pos += arg.ext_rank
            args[i] = Tensor(arg.component, indices[prev_pos:ind_pos])
    # 定义一个实例方法 `doit`，接受关键字参数 `hints`
    def doit(self, **hints):
        # 从当前对象中获取 `_is_canon_bp` 方法引用
        is_canon_bp = self._is_canon_bp
        # 从 `hints` 中获取 `deep` 参数，默认为 `True`
        deep = hints.get('deep', True)
        
        # 如果 `deep` 为真，则递归调用每个 `args` 中的 `doit` 方法
        if deep:
            args = [arg.doit(**hints) for arg in self.args]

            """
            There may now be conflicts between dummy indices of different args
            (each arg's doit method does not have any information about which
            dummy indices are already used in the other args), so we
            deduplicate them.
            """
            # 创建一个字典 `rule`，将每个 `arg` 映射到其对应的递归结果 `args`
            rule = dict(zip(self.args, args))
            # 调用对象的 `_dedupe_indices_in_rule` 方法，处理 `rule` 中的重复索引
            rule = self._dedupe_indices_in_rule(rule)
            # 更新 `args`，使每个 `arg` 指向 `rule` 中的对应值
            args = [rule[a] for a in self.args]

        else:
            # 如果 `deep` 为假，则直接使用 `self.args` 作为 `args`
            args = self.args

        # 过滤掉 `args` 中等于 `self.identity` 的元素
        args = [arg for arg in args if arg != self.identity]

        # 提取非张量系数：
        coeff = reduce(lambda a, b: a*b, [arg for arg in args if not isinstance(arg, TensExpr)], S.One)
        # 过滤掉 `args` 中非张量表达式的元素，保留张量表达式
        args = [arg for arg in args if isinstance(arg, TensExpr)]

        # 如果 `args` 中没有张量表达式，则返回 `coeff`
        if len(args) == 0:
            return coeff

        # 如果 `coeff` 不等于 `self.identity`，则将 `coeff` 插入到 `args` 的开头
        if coeff != self.identity:
            args = [coeff] + args
        # 如果 `coeff` 等于 0，则返回 `S.Zero`
        if coeff == 0:
            return S.Zero

        # 如果 `args` 只有一个元素，则直接返回该元素
        if len(args) == 1:
            return args[0]

        # 调用 `TensMul._tensMul_contract_indices` 处理 `args`，返回处理后的 `args`, `indices`, `free`, `dum`
        args, indices, free, dum = TensMul._tensMul_contract_indices(args)

        # 准备索引数据：
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        # 创建一个 `self.func` 类的实例 `obj`，传入 `args` 作为参数
        obj = self.func(*args)
        # 设置 `obj` 的索引类型属性
        obj._index_types = index_types
        # 设置 `obj` 的索引结构属性
        obj._index_structure = index_structure
        # 设置 `obj` 的外部秩属性
        obj._ext_rank = len(obj._index_structure.free) + 2*len(obj._index_structure.dum)
        # 设置 `obj` 的系数属性
        obj._coeff = coeff
        # 设置 `obj` 的 `_is_canon_bp` 属性
        obj._is_canon_bp = is_canon_bp
        # 返回 `obj`
        return obj

    # TODO: this method should be private
    # TODO: should this method be renamed _from_components_free_dum ?
    # 定义一个静态方法 `from_data`，接受 `coeff`, `components`, `free`, `dum` 等参数
    @staticmethod
    def from_data(coeff, components, free, dum, **kw_args):
        # 使用 `TensMul._get_tensors_from_components_free_dum` 获取 `components`, `free`, `dum` 对应的张量，并调用 `doit()` 方法
        return TensMul(coeff, *TensMul._get_tensors_from_components_free_dum(components, free, dum), **kw_args).doit()

    # 定义一个静态方法 `_get_tensors_from_components_free_dum`，接受 `components`, `free`, `dum` 参数
    @staticmethod
    def _get_tensors_from_components_free_dum(components, free, dum):
        """
        Get a list of ``Tensor`` objects by distributing ``free`` and ``dum`` indices on the ``components``.
        """
        # 调用 `_IndexStructure.from_components_free_dum` 处理 `components`, `free`, `dum`，得到索引结构
        index_structure = _IndexStructure.from_components_free_dum(components, free, dum)
        # 获取索引列表
        indices = index_structure.get_indices()
        # 预先分配张量列表
        tensors = [None for i in components]  # pre-allocate list

        # 在组件上分配索引以构建张量列表：
        ind_pos = 0
        for i, component in enumerate(components):
            prev_pos = ind_pos
            ind_pos += component.rank
            tensors[i] = Tensor(component, indices[prev_pos:ind_pos])
        # 返回张量列表
        return tensors

    # 定义一个实例方法 `_get_free_indices_set`，返回自由索引的集合
    def _get_free_indices_set(self):
        return {i[0] for i in self.free}
    # 返回一个集合，包含所有虚指标的位置
    def _get_dummy_indices_set(self):
        # 使用 itertools.chain 将 self.dum 中的所有元素展开成一个单层列表，然后转换为集合
        dummy_pos = set(itertools.chain(*self.dum))
        return {idx for i, idx in enumerate(self._index_structure.get_indices()) if i in dummy_pos}

    # 返回一个列表，其中每个元素表示在参数列表中的位置偏移
    def _get_position_offset_for_indices(self):
        # 创建一个长度为 self.ext_rank 的 None 列表
        arg_offset = [None for i in range(self.ext_rank)]
        counter = 0
        for arg in self.args:
            if not isinstance(arg, TensExpr):
                continue
            # 将每个 TensExpr 对象的 ext_rank 添加到对应位置的偏移列表中
            for j in range(arg.ext_rank):
                arg_offset[j + counter] = counter
            counter += arg.ext_rank
        return arg_offset

    # 返回自由参数的排序列表
    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    # 返回通过参数获取的组件列表
    @property
    def components(self):
        return self._get_components_from_args(self.args)

    # 返回自由参数中每个自由指标的位置信息的列表
    @property
    def free_in_args(self):
        # 获取位置偏移量和参数位置信息
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        # 返回一个列表，每个元素包含指标、位置偏移后的位置和参数位置
        return [(ind, pos-arg_offset[pos], argpos[pos]) for (ind, pos) in self.free]

    # 返回系数
    @property
    def coeff(self):
        # 返回不是 TensExpr 实例的所有参数的乘积
        # return Mul.fromiter([c for c in self.args if not isinstance(c, TensExpr)])
        return self._coeff

    # 返回没有系数的表达式
    @property
    def nocoeff(self):
        # 返回一个表达式，其中包含所有 TensExpr 实例的乘积，并进行计算
        return self.func(*[t for t in self.args if isinstance(t, TensExpr)]).doit()

    # 返回参数中每个虚指标的位置信息的列表
    @property
    def dum_in_args(self):
        # 获取位置偏移量和参数位置信息
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        # 返回一个列表，每个元素包含两个虚指标的位置偏移后的位置和参数位置
        return [(p1-arg_offset[p1], p2-arg_offset[p2], argpos[p1], argpos[p2]) for p1, p2 in self.dum]

    # 检查当前 Tensor 对象是否等于另一个对象
    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        # 将 other 转换为 sympy 对象
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            # 断言没有组件
            assert not self.components
            return self.coeff == other
        # 比较规范形式
        return self.canon_bp() == other.canon_bp()

    # 返回张量的指标列表
    def get_indices(self):
        """
        返回张量的指标列表。

        解释
        ===========

        指标按照它们在组件张量中出现的顺序列出。
        虚指标被赋予一个不会与自由指标名称冲突的名称。

        示例
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_indices()
        [L_0, -L_0, m2]
        """
        return self._indices
    def get_free_indices(self) -> list[TensorIndex]:
        """
        Returns the list of free indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_free_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_free_indices()
        [m2]
        """
        # 调用内部方法 `_index_structure` 获取自由指标列表
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        """
        Replace indices in the tensor expression.

        Explanation
        ===========

        This method replaces indices in the tensor expression with new ones
        according to the provided dictionary.

        """
        # 使用提供的替换字典 `repl` 替换张量表达式中的指标
        return self.func(*[arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args])

    def split(self):
        """
        Returns a list of tensors, whose product is ``self``.

        Explanation
        ===========

        This method splits the tensor expression into a list of tensors,
        each representing a component of the original tensor expression.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(a,b)*B(-b,c)
        >>> t
        A(a, L_0)*B(-L_0, c)
        >>> t.split()
        [A(a, L_0), B(-L_0, c)]
        """
        # 如果没有子项，直接返回当前列表中的自身作为单一张量的列表
        if self.args == ():
            return [self]
        splitp = []
        res = 1
        # 遍历每个子项，将张量作为单独的成分添加到 `splitp` 中
        for arg in self.args:
            if isinstance(arg, Tensor):
                splitp.append(res*arg)
                res = 1
            else:
                res *= arg
        return splitp

    def _expand(self, **hints):
        """
        Expand the tensor expression.

        Explanation
        ===========

        This method is a temporary solution and aims to expand the tensor
        expression using the provided hints.

        """
        # 扩展每个参数并形成它们的笛卡尔积，最终生成一个 `TensAdd` 实例
        args = [_expand(arg, **hints) for arg in self.args]
        args1 = [arg.args if isinstance(arg, (Add, TensAdd)) else (arg,) for arg in args]
        return TensAdd(*[
            TensMul(*i) for i in itertools.product(*args1)]
        )

    def __neg__(self):
        """
        Negate the tensor expression.

        Explanation
        ===========

        This method negates the tensor expression by multiplying it with `-1`.

        """
        # 返回当前张量表达式的相反数
        return TensMul(S.NegativeOne, self, is_canon_bp=self._is_canon_bp).doit()

    def __getitem__(self, item):
        """
        Get item from tensor data.

        Explanation
        ===========

        This method retrieves an item from the tensor's data using the provided
        index `item`. It raises a deprecation warning.

        """
        # 弃用警告：此方法即将弃用
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]
    def _get_args_for_traditional_printer(self):
        # 复制参数列表，以防对原参数列表进行修改
        args = list(self.args)
        # 如果系数可以提取负号
        if self.coeff.could_extract_minus_sign():
            # 对于类似于 "-A(a)" 的表达式
            sign = "-"
            # 如果参数列表的第一个元素是 S.NegativeOne
            if args[0] == S.NegativeOne:
                # 移除第一个元素
                args = args[1:]
            else:
                # 将第一个元素变为其相反数
                args[0] = -args[0]
        else:
            sign = ""
        # 返回负号和处理后的参数列表
        return sign, args

    def _sort_args_for_sorted_components(self):
        """
        Returns the ``args`` sorted according to the components commutation
        properties.

        Explanation
        ===========

        The sorting is done taking into account the commutation group
        of the component tensors.
        """
        # 从参数列表中筛选出张量表达式
        cv = [arg for arg in self.args if isinstance(arg, TensExpr)]
        sign = 1
        n = len(cv) - 1
        for i in range(n):
            for j in range(n, i, -1):
                # 检查 cv[j-1] 和 cv[j] 的交换性
                c = cv[j-1].commutes_with(cv[j])
                # 如果 `c` 是 `None`，既不交换也不反交换，跳过处理：
                if c not in (0, 1):
                    continue
                # 对组件的索引类型按名称排序
                typ1 = sorted(set(cv[j-1].component.index_types), key=lambda x: x.name)
                typ2 = sorted(set(cv[j].component.index_types), key=lambda x: x.name)
                # 根据排序后的索引类型和组件名称比较进行交换
                if (typ1, cv[j-1].component.name) > (typ2, cv[j].component.name):
                    cv[j-1], cv[j] = cv[j], cv[j-1]
                    # 如果 `c` 为 1，反交换，修改符号
                    if c:
                        sign = -sign

        # 计算最终的系数
        coeff = sign * self.coeff
        # 如果系数不为 1，则将其作为第一个元素返回
        if coeff != 1:
            return [coeff] + cv
        # 否则，只返回排序后的参数列表
        return cv

    def sorted_components(self):
        """
        Returns a tensor product with sorted components.
        """
        # 对排序后的组件进行张量乘积，并进行必要的计算
        return TensMul(*self._sort_args_for_sorted_components()).doit()

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``

        For further details, see the method in ``TIDS`` with the same name.
        """
        # 调用 perm2tensor 函数，返回给定置换 ``g`` 对应的张量
        return perm2tensor(self, g, is_canon_bp=is_canon_bp)
    def canon_bp(self):
        """
        使用Butler-Portugal算法对具有单项对称性的张量进行规范化。

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        >>> t = A(m0,-m1)*A(m1,-m0)
        >>> t.canon_bp()
        -A(L_0, L_1)*A(-L_0, -L_1)
        >>> t = A(m0,-m1)*A(m1,-m2)*A(m2,-m0)
        >>> t.canon_bp()
        0
        """
        # 如果已经是Butler-Portugal规范化的，直接返回自身
        if self._is_canon_bp:
            return self
        # 对表达式进行展开
        expr = self.expand()
        # 如果是TensAdd类型，则递归调用canon_bp()
        if isinstance(expr, TensAdd):
            return expr.canon_bp()
        # 如果没有组件，则直接返回表达式
        if not expr.components:
            return expr
        # 对组件进行排序
        t = expr.sorted_components()
        # 获取索引结构的参数
        g, dummies, msym = t._index_structure.indices_canon_args()
        # 获取组件的参数并规范化
        v = components_canon_args(t.components)
        can = canonicalize(g, dummies, msym, *v)
        # 如果规范化结果为0，则返回S.Zero
        if can == 0:
            return S.Zero
        # 将规范化后的结果转换为张量乘积
        tmul = t.perm2tensor(can, True)
        return tmul

    def contract_delta(self, delta):
        # 使用delta缩并度量
        t = self.contract_metric(delta)
        return t

    def _get_indices_to_args_pos(self):
        """
        返回一个字典，映射索引位置到TensMul的参数编号。
        """
        pos_map = {}
        pos_counter = 0
        for arg_i, arg in enumerate(self.args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            for i in range(arg.ext_rank):
                pos_map[pos_counter] = arg_i
                pos_counter += 1
        return pos_map

    def _set_new_index_structure(self, im, is_canon_bp=False):
        # 设置新的索引结构
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        # 设置张量表达式的索引
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        args = list(self.args)[:]
        pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            ext_rank = arg.ext_rank
            args[i] = arg._set_indices(*indices[pos:pos+ext_rank])
            pos += ext_rank
        return TensMul(*args, is_canon_bp=is_canon_bp).doit()

    @staticmethod
    def _index_replacement_for_contract_metric(args, free, dum):
        # 对于缩并度量，进行索引替换
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
    # 替换张量表达式中的指标，返回替换后的新表达式
    def substitute_indices(self, *index_tuples):
        # 初始化一个空列表用于存储替换后的参数
        new_args = []
        # 遍历当前张量表达式的参数列表
        for arg in self.args:
            # 如果参数是张量表达式，则递归调用 substitute_indices 方法进行指标替换
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            # 将处理后的参数添加到新的参数列表中
            new_args.append(arg)
        # 创建一个新的 TensMul 对象，并调用 doit 方法处理结果
        return TensMul(*new_args).doit()

    # 调用对象，用给定的指标进行替换
    def __call__(self, *indices):
        # 弃用的调用警告
        deprecate_call()
        # 获取自由指标列表
        free_args = self.free_args
        # 将输入的指标转换为列表
        indices = list(indices)
        # 检查输入的指标类型与自由指标类型是否兼容
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        # 如果输入的指标与自由指标相同，则返回当前对象
        if indices == free_args:
            return self
        # 使用输入的指标替换自由指标，并返回结果
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # 如果存在缩并指标，则重新构建对象以确保所有缩并指标被识别为虚指标
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        # 否则直接返回替换后的对象
        return t

    # 提取数据方法，从当前张量表达式中提取数据
    def _extract_data(self, replacement_dict):
        # 提取参数中张量表达式的数据
        args_indices, arrays = zip(*[arg._extract_data(replacement_dict) for arg in self.args if isinstance(arg, TensExpr)])
        # 计算非张量表达式参数的乘积系数
        coeff = reduce(operator.mul, [a for a in self.args if not isinstance(a, TensExpr)], S.One)
        # 将指标转换为自由和虚指标
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
        # 将虚指标数据转换为虚指标
        dum = TensMul._dummy_data_to_dum(dummy_data)
        # 获取当前张量表达式的外部秩
        ext_rank = self.ext_rank
        # 按照索引位置对自由指标进行排序
        free.sort(key=lambda x: x[1])
        # 提取自由指标的索引
        free_indices = [i[0] for i in free]
        # 返回自由指标和通过数据缩并得到的结果
        return free_indices, coeff*_TensorDataLazyEvaluator.data_contract_dum(arrays, dum, ext_rank)

    # 获取数据的属性方法，获取当前张量表达式的数据
    @property
    def data(self):
        # 弃用的数据属性调用警告
        deprecate_data()
        # 使用忽略警告上下文管理器获取数据
        with ignore_warnings(SymPyDeprecationWarning):
            # 扩展当前对象并获取其数据替换字典中的数据
            dat = _tensor_data_substitution_dict[self.expand()]
        # 返回获取的数据
        return dat

    # 设置数据的属性方法，尝试设置当前张量表达式的数据
    @data.setter
    def data(self, data):
        # 弃用的数据属性调用警告
        deprecate_data()
        # 抛出错误，不允许设置张量表达式的组件数据
        raise ValueError("Not possible to set component data to a tensor expression")

    # 删除数据的属性方法，尝试删除当前张量表达式的数据
    @data.deleter
    def data(self):
        # 弃用的数据属性调用警告
        deprecate_data()
        # 抛出错误，不允许删除张量表达式的组件数据
        raise ValueError("Not possible to delete component data to a tensor expression")

    # 迭代器方法，迭代当前张量表达式的数据
    def __iter__(self):
        # 弃用的数据属性调用警告
        deprecate_data()
        # 使用忽略警告上下文管理器获取数据
        with ignore_warnings(SymPyDeprecationWarning):
            # 如果数据为空，则抛出错误
            if self.data is None:
                raise ValueError("No iteration on abstract tensors")
            # 否则返回数据的迭代器
            return self.data.__iter__()

    # 静态方法，用于...
    def _dedupe_indices(new, exclude):
        """
        exclude: set
        new: TensExpr
        
        如果 ``new`` 中有任何在 ``exclude`` 中的虚拟指标，返回一个替换了这些指标的新版本。
        如果不需要替换任何指标，返回 None。
        """
        exclude = set(exclude)
        dums_new = set(get_dummy_indices(new))  # 获取 new 中的虚拟指标
        free_new = set(get_free_indices(new))   # 获取 new 中的自由指标
        
        conflicts = dums_new.intersection(exclude)  # 找到与 exclude 中重复的虚拟指标
        if len(conflicts) == 0:
            return None
        
        """
        ``exclude_for_gen`` 将会传递给 ``_IndexStructure._get_generator_for_dummy_indices()``。
        由于后者不会使用索引位置，我们在这里将其设置为 ``None``。
        """
        exclude.update(dums_new)
        exclude.update(free_new)
        exclude_for_gen = [(i, None) for i in exclude]  # 为生成器准备排除的指标
        gen = _IndexStructure._get_generator_for_dummy_indices(exclude_for_gen)  # 获取用于虚拟指标生成器
        repl = {}
        for d in conflicts:
            if -d in repl.keys():
                continue
            newname = gen(d.tensor_index_type)  # 生成新的虚拟指标名称
            new_d = d.func(newname, *d.args[1:])  # 创建替换后的虚拟指标
            repl[d] = new_d
            repl[-d] = -new_d
        
        if len(repl) == 0:
            return None
        
        new_renamed = new._replace_indices(repl)  # 使用 repl 替换 new 的指标
        return new_renamed
    
    def _dedupe_indices_in_rule(self, rule):
        """
        rule: dict
        
        对 rule 中的所有值应用 TensMul._dedupe_indices。
        """
        index_rules = {k:v for k,v in rule.items() if isinstance(k, TensorIndex)}  # 筛选出规则中的张量指标
        other_rules = {k:v for k,v in rule.items() if k not in index_rules.keys()}  # 其余规则
        exclude = set(self.get_indices())  # 获取当前对象的指标集合
        
        newrule = {}
        newrule.update(index_rules)
        exclude.update(index_rules.keys())
        exclude.update(index_rules.values())
        for old, new in other_rules.items():
            new_renamed = TensMul._dedupe_indices(new, exclude)  # 对非指标规则应用去重函数
            if old == new or new_renamed is None:
                newrule[old] = new
            else:
                newrule[old] = new_renamed
                exclude.update(get_indices(new_renamed))  # 更新排除集合以包含新指标
        return newrule
    
    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        from sympy.concrete.summations import Sum
        index_symbols = [i.args[0] for i in self.get_indices()]  # 获取对象的索引符号
        args = [arg.args[0] if isinstance(arg, Sum) else arg for arg in args]  # 提取参数中的索引符号
        expr = Mul.fromiter(args)  # 创建一个从参数中生成的乘积表达式
        return self._check_add_Sum(expr, index_symbols)  # 检查并添加求和符号到表达式中的索引符号
    # 定义一个方法，用于计算部分导数
    def _eval_partial_derivative(self, s):
        # 创建一个空列表，用于存储导数项
        terms = []
        # 遍历自身的参数列表
        for i, arg in enumerate(self.args):
            # 检查参数是否为 TensExpr 类型的实例（张量表达式）
            if isinstance(arg, TensExpr):
                # 如果是张量表达式，递归调用其 _eval_partial_derivative 方法计算偏导数
                d = arg._eval_partial_derivative(s)
            else:
                # 如果不是张量表达式
                # 如果 s 不是一个符号（symbol），则不调用 diff 方法
                if s._diff_wrt:
                    # 调用 arg 的 _eval_derivative 方法计算关于 s 的导数
                    d = arg._eval_derivative(s)
                else:
                    # 否则将导数设置为零
                    d = S.Zero
            # 如果计算得到的导数不为零，则将其加入到 terms 列表中
            if d:
                terms.append(TensMul.fromiter(self.args[:i] + (d,) + self.args[i + 1:]))
        # 从 terms 列表中创建一个 TensAdd 对象，表示所有导数项的和
        return TensAdd.fromiter(terms)


    # 定义一个方法，用于检查自身与给定表达式的匹配情况
    def matches(self, expr, repl_dict=None, old=False):
        # 将输入的 expr 转换为 Sympy 符号
        expr = sympify(expr)

        # 如果 repl_dict 为 None，则初始化为空字典，否则复制 repl_dict
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # 检查所有参数是否为非交换张量，并且其 comm 属性为零
        commute = all(arg.component.comm == 0 for arg in expr.args if isinstance(arg, Tensor))
        # 如果所有张量都是非交换的，则调用 _matches_commutative 方法进行匹配
        if commute:
            return self._matches_commutative(expr, repl_dict, old)
        else:
            # 否则抛出未实现异常，因为非交换张量的匹配尚未实现
            raise NotImplementedError("Tensor matching not implemented for non-commuting tensors")
class TensorElement(TensExpr):
    """
    Tensor with evaluated components.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> i, j, k = symbols("i j k")
    >>> A = TensorHead("A", [L, L], TensorSymmetry.fully_symmetric(2))
    >>> A(i, j).get_free_indices()
    [i, j]

    If we want to set component ``i`` to a specific value, use the
    ``TensorElement`` class:

    >>> from sympy.tensor.tensor import TensorElement
    >>> te = TensorElement(A(i, j), {i: 2})

    As index ``i`` has been accessed (``{i: 2}`` is the evaluation of its 3rd
    element), the free indices will only contain ``j``:

    >>> te.get_free_indices()
    [j]
    """

    def __new__(cls, expr, index_map):
        if not isinstance(expr, Tensor):
            # If the expression `expr` is not a `Tensor`, attempt to remap its components.
            if not isinstance(expr, TensExpr):
                # Raise an error if `expr` is neither a `Tensor` nor a `TensExpr`.
                raise TypeError("%s is not a tensor expression" % expr)
            # Recursively construct a new `TensorElement` for each argument in `expr`.
            return expr.func(*[TensorElement(arg, index_map) for arg in expr.args])
        # Get the free indices of the tensor expression `expr`.
        expr_free_indices = expr.get_free_indices()
        # Create a mapping from index names to their corresponding `TensorIndex` objects.
        name_translation = {i.args[0]: i for i in expr_free_indices}
        # Remap the indices in `index_map` according to the `name_translation`.
        index_map = {name_translation.get(index, index): value for index, value in index_map.items()}
        # Filter `index_map` to retain only indices that exist in `expr_free_indices`.
        index_map = {index: value for index, value in index_map.items() if index in expr_free_indices}
        # If no indices are left in `index_map`, return the original expression `expr`.
        if len(index_map) == 0:
            return expr
        # Calculate the free indices after applying `index_map`.
        free_indices = [i for i in expr_free_indices if i not in index_map.keys()]
        # Convert `index_map` to a `Dict` object.
        index_map = Dict(index_map)
        # Create a new `TensExpr` object using the current class (`cls`), `expr`, and `index_map`.
        obj = TensExpr.__new__(cls, expr, index_map)
        # Store the computed `free_indices` in the object.
        obj._free_indices = free_indices
        return obj

    @property
    def free(self):
        # Return a list of tuples containing free indices and their enumeration.
        return [(index, i) for i, index in enumerate(self.get_free_indices())]

    @property
    def dum(self):
        # TODO: inherit dummies from expr, but currently returns an empty list.
        return []

    @property
    def expr(self):
        # Return the underlying expression stored in `_args[0]`.
        return self._args[0]

    @property
    def index_map(self):
        # Return the index mapping stored in `_args[1]`.
        return self._args[1]

    @property
    def coeff(self):
        # Return the coefficient associated with the tensor element, which is `1` (S.One).
        return S.One

    @property
    def nocoeff(self):
        # Return the tensor element itself, as there is no coefficient to be removed.
        return self

    def get_free_indices(self):
        # Return the list of free indices stored in `_free_indices`.
        return self._free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # TODO: Implement a more efficient replacement of indices using `xreplace`.
        return self.xreplace(repl)

    def get_indices(self):
        # Return the free indices of the tensor element.
        return self.get_free_indices()

    def _extract_data(self, replacement_dict):
        # Extract data from the tensor expression `expr` after applying `replacement_dict`.
        ret_indices, array = self.expr._extract_data(replacement_dict)
        # Retrieve the index mapping stored in `index_map`.
        index_map = self.index_map
        # Create a tuple of slices using `index_map` for indexing `array`.
        slice_tuple = tuple(index_map.get(i, slice(None)) for i in ret_indices)
        # Remove indices from `ret_indices` that are present in `index_map`.
        ret_indices = [i for i in ret_indices if i not in index_map]
        # Slice `array` using `slice_tuple`.
        array = array.__getitem__(slice_tuple)
        return ret_indices, array


class WildTensorHead(TensorHead):
    """
    A wild object that is used to create ``WildTensor`` instances

    Explanation
    ===========

    Examples
    ========
    """
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    导入所需的类：TensorHead, TensorIndex, WildTensorHead, TensorIndexType

    >>> R3 = TensorIndexType('R3', dim=3)
    创建一个名为 'R3'，维度为 3 的张量索引类型对象

    >>> p = TensorIndex('p', R3)
    创建一个名为 'p' 的张量索引对象，属于索引类型 'R3'

    >>> q = TensorIndex('q', R3)
    创建一个名为 'q' 的张量索引对象，属于索引类型 'R3'

    A WildTensorHead can be created without specifying a ``TensorIndexType``
    可以创建一个没有指定 ``TensorIndexType`` 的 WildTensorHead 对象

    >>> W = WildTensorHead("W")
    创建一个名为 'W' 的 WildTensorHead 对象

    Calling it with a ``TensorIndex`` creates a ``WildTensor`` instance.
    使用一个 ``TensorIndex`` 调用它会创建一个 ``WildTensor`` 实例。

    >>> type(W(p))
    返回调用 W(p) 后的对象类型，应为 <class 'sympy.tensor.tensor.WildTensor'>

    The ``TensorIndexType`` is automatically detected from the index that is passed
    从传递的索引自动检测到 ``TensorIndexType``

    >>> W(p).component
    返回 W(p) 的组件，应为 W(R3)

    Calling it with no indices returns an object that can match tensors with any number of indices.
    不使用索引调用它会返回一个可以匹配任意数量索引的对象。

    >>> K = TensorHead('K', [R3])
    创建一个名为 'K' 的张量头对象，索引类型为 [R3]

    >>> Q = TensorHead('Q', [R3, R3])
    创建一个名为 'Q' 的张量头对象，索引类型为 [R3, R3]

    >>> W().matches(K(p))
    调用 W() 并匹配 K(p)，返回匹配结果字典 {W: K(p)}

    >>> W().matches(Q(p,q))
    调用 W() 并匹配 Q(p,q)，返回匹配结果字典 {W: Q(p, q)}

    If you want to ignore the order of indices while matching, pass ``unordered_indices=True``.
    如果想要在匹配时忽略索引的顺序，请传递 ``unordered_indices=True``。

    >>> U = WildTensorHead("U", unordered_indices=True)
    创建一个名为 'U' 的 WildTensorHead 对象，指定忽略索引顺序为 True

    >>> W(p,q).matches(Q(q,p))
    调用 W(p,q) 并匹配 Q(q,p)，无返回结果

    >>> U(p,q).matches(Q(q,p))
    调用 U(p,q) 并匹配 Q(q,p)，返回匹配结果字典 {U(R3,R3): _WildTensExpr(Q(q, p))}

    Parameters
    ==========
    name : name of the tensor
    张量的名称

    unordered_indices : whether the order of the indices matters for matching
        (default: False)
    索引顺序是否影响匹配的布尔值，默认为 False

    See also
    ========
    ``WildTensor``
    ``TensorHead``
    参见：``WildTensor`` 和 ``TensorHead``

    """
    定义一个新类 __new__ 方法，用于创建 WildTensorHead 对象
    参数：
    name : 张量的名称
    index_types : 张量的索引类型列表，默认为 None
    symmetry : 张量的对称性，默认为 None
    comm : 张量的交换性，默认为 0
    unordered_indices : 是否忽略索引顺序进行匹配，默认为 False

    if isinstance(name, str):
        name_symbol = Symbol(name)
    如果 name 是字符串类型，则将其转换为符号类型

    elif isinstance(name, Symbol):
        name_symbol = name
    否则，如果 name 已经是符号类型，则直接使用

    else:
        raise ValueError("invalid name")
    否则，抛出值错误异常，表示名称无效

    if index_types is None:
        index_types = []
    如果 index_types 是 None，则将其设为一个空列表

    if symmetry is None:
        symmetry = TensorSymmetry.no_symmetry(len(index_types))
    如果 symmetry 是 None，则使用 TensorSymmetry.no_symmetry 函数创建一个对称性对象，长度为 index_types 的长度

    else:
        assert symmetry.rank == len(index_types)
        否则，确保对称性的秩与 index_types 的长度相等

    if symmetry != TensorSymmetry.no_symmetry(len(index_types)):
        raise NotImplementedError("Wild matching based on symmetry is not implemented.")
    如果对称性不等于 TensorSymmetry.no_symmetry(len(index_types))，则抛出未实现错误，表示基于对称性的 Wild 匹配尚未实现

    obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), sympify(symmetry), sympify(comm), sympify(unordered_indices))
    使用 Basic.__new__ 方法创建一个新的对象，传递名称符号、索引类型元组、对称性、交换性和索引顺序是否无序的布尔值

    return obj
    返回创建的对象

    @property
    def unordered_indices(self):
    定义一个 unordered_indices 属性方法

    return self.args[4]
    返回对象的第五个参数，即索引顺序是否无序的布尔值

    def __call__(self, *indices, **kwargs):
    定义一个 __call__ 方法，用于对象的调用

    tensor = WildTensor(self, indices, **kwargs)
    创建一个 WildTensor 对象，传递当前 WildTensorHead 实例、索引元组和其他关键字参数

    return tensor.doit()
    返回 WildTensor 对象调用 doit() 方法后的结果
class WildTensor(Tensor):
    """
    A wild object which matches ``Tensor`` instances

    Explanation
    ===========
    This is instantiated by attaching indices to a ``WildTensorHead`` instance.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])

    Matching also takes the indices into account
    >>> W(p).matches(K(p))
    {W(R3): _WildTensExpr(K(p))}
    >>> W(p).matches(K(q))
    >>> W(p).matches(K(-p))

    If you want to match objects with any number of indices, just use a ``WildTensor`` with no indices.
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    See Also
    ========
    ``WildTensorHead``
    ``Tensor``

    """
    def __new__(cls, tensor_head, indices, **kw_args):
        # Check if the user intends to create a canonical object
        is_canon_bp = kw_args.pop("is_canon_bp", False)

        # If the provided tensor_head is a TensorHead instance (not WildTensorHead), return a normal Tensor
        if tensor_head.func == TensorHead:
            """
            If someone tried to call WildTensor by supplying a TensorHead (not a WildTensorHead), return a normal tensor instead. This is helpful when using subs on an expression to replace occurrences of a WildTensorHead with a TensorHead.
            """
            return Tensor(tensor_head, indices, is_canon_bp=is_canon_bp, **kw_args)
        # If the provided tensor_head is a _WildTensExpr instance, return it with given indices
        elif tensor_head.func == _WildTensExpr:
            return tensor_head(*indices)

        # Parse indices and create corresponding index types
        indices = cls._parse_indices(tensor_head, indices)
        index_types = [ind.tensor_index_type for ind in indices]
        
        # Create a new TensorHead instance with updated properties
        tensor_head = tensor_head.func(
            tensor_head.name,
            index_types,
            symmetry=None,
            comm=tensor_head.comm,
            unordered_indices=tensor_head.unordered_indices,
            )

        # Instantiate a Basic object with tensor_head and indices
        obj = Basic.__new__(cls, tensor_head, Tuple(*indices))

        # Initialize various properties of the WildTensor object
        obj.name = tensor_head.name
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]

        # Validate the number of indices against the rank of tensor_head
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        
        obj.is_canon_bp = is_canon_bp
        obj._index_map = obj._build_index_map(indices, obj._index_structure)

        return obj
    # 判断给定表达式是否为 TensExpr 类型或者是否为数值 1，如果不是则返回 None
    def matches(self, expr, repl_dict=None, old=False):
        # 如果 expr 不是 TensExpr 类型并且不等于数值 1，则返回 None
        if not isinstance(expr, TensExpr) and expr != S(1):
            return None

        # 如果 repl_dict 为 None，则将其初始化为空字典；否则复制 repl_dict
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # 如果 self.indices 的长度大于 0
        if len(self.indices) > 0:
            # 如果 expr 没有 get_free_indices 方法，则返回 None
            if not hasattr(expr, "get_free_indices"):
                return None
            
            # 获取表达式的自由指标
            expr_indices = expr.get_free_indices()
            
            # 如果表达式的自由指标数量不等于 self.indices 的长度，则返回 None
            if len(expr_indices) != len(self.indices):
                return None
            
            # 如果 WildTensor 允许无序指标
            if self._component.unordered_indices:
                # 忽略顺序匹配指标，更新 repl_dict
                m = self._match_indices_ignoring_order(expr)
                if m is None:
                    return None
                else:
                    repl_dict.update(m)
            else:
                # 逐一匹配指标
                for i in range(len(expr_indices)):
                    m = self.indices[i].matches(expr_indices[i])
                    if m is None:
                        return None
                    else:
                        repl_dict.update(m)

            # 将 WildTensor 的组件映射到 _WildTensExpr(expr)
            repl_dict[self.component] = _WildTensExpr(expr)
        else:
            # 如果没有传递指标给 WildTensor，则它可以匹配任意数量的指标的张量
            repl_dict[self] = expr

        # 返回更新后的 repl_dict
        return repl_dict
    def _match_indices_ignoring_order(self, expr, repl_dict=None, old=False):
        """
        Helper method for matches. Checks if the indices of self and expr
        match disregarding index ordering.
        """
        # 如果 repl_dict 为 None，则将其初始化为空字典
        if repl_dict is None:
            repl_dict = {}
        else:
            # 复制 repl_dict，以确保不修改原始输入的字典
            repl_dict = repl_dict.copy()

        # 定义一个内部函数 siftkey，用于根据索引类型分类
        def siftkey(ind):
            # 如果索引是 WildTensorIndex 类型
            if isinstance(ind, WildTensorIndex):
                # 如果忽略上下文的匹配，则返回 "wild, updown"
                if ind.ignore_updown:
                    return "wild, updown"
                # 否则返回 "wild"
                else:
                    return "wild"
            # 对于非 WildTensorIndex 类型的索引，返回 "nonwild"
            else:
                return "nonwild"

        # 使用 sift 函数对 self.indices 进行分类处理
        indices_sifted = sift(self.indices, siftkey)

        # 存储已匹配的索引
        matched_indices = []
        # 获取表达式 expr 的索引
        expr_indices_remaining = expr.get_indices()

        # 遍历非通配符索引（"nonwild" 类型）
        for ind in indices_sifted["nonwild"]:
            matched_this_ind = False
            # 遍历剩余的表达式索引
            for e_ind in expr_indices_remaining:
                # 如果表达式索引已经在已匹配列表中，则跳过
                if e_ind in matched_indices:
                    continue
                # 尝试匹配 self 索引 ind 和表达式索引 e_ind
                m = ind.matches(e_ind)
                if m is not None:
                    matched_this_ind = True
                    # 更新 repl_dict
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            # 如果未成功匹配，则返回 None
            if not matched_this_ind:
                return None

        # 过滤掉已匹配的表达式索引
        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]

        # 遍历通配符索引（"wild" 类型）
        for ind in indices_sifted["wild"]:
            matched_this_ind = False
            # 遍历剩余的表达式索引
            for e_ind in expr_indices_remaining:
                # 尝试匹配 self 索引 ind 和表达式索引 e_ind
                m = ind.matches(e_ind)
                if m is not None:
                    # 检查是否存在替换的冲突
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    # 更新 repl_dict
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            # 如果未成功匹配，则返回 None
            if not matched_this_ind:
                return None

        # 过滤掉已匹配的表达式索引
        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]

        # 遍历带有忽略上下文的通配符索引（"wild, updown" 类型）
        for ind in indices_sifted["wild, updown"]:
            matched_this_ind = False
            # 遍历剩余的表达式索引
            for e_ind in expr_indices_remaining:
                # 尝试匹配 self 索引 ind 和表达式索引 e_ind
                m = ind.matches(e_ind)
                if m is not None:
                    # 检查是否存在替换的冲突
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    # 更新 repl_dict
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            # 如果未成功匹配，则返回 None
            if not matched_this_ind:
                return None

        # 如果已匹配的索引数量小于 self.indices 的总数，则返回 None
        if len(matched_indices) < len(self.indices):
            return None
        else:
            # 否则返回更新后的 repl_dict
            return repl_dict
class WildTensorIndex(TensorIndex):
    """
    A wild object that matches TensorIndex instances.
    
    Examples
    ========
    >>> from sympy.tensor.tensor import TensorIndex, TensorIndexType, WildTensorIndex
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex("p", R3)
    
    By default, covariant indices only match with covariant indices (and
    similarly for contravariant)
    
    >>> q = WildTensorIndex("q", R3)
    >>> (q).matches(p)
    {q: p}
    >>> (q).matches(-p)
    
    If you want matching to ignore whether the index is co/contra-variant, set
    ignore_updown=True
    
    >>> r = WildTensorIndex("r", R3, ignore_updown=True)
    >>> (r).matches(-p)
    {r: -p}
    >>> (r).matches(p)
    {r: p}
    
    Parameters
    ==========
    name : name of the index (string), or ``True`` if you want it to be
        automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)
    ignore_updown : bool, Whether this should match both co- and contra-variant
        indices (default:False)
    """
    
    def __new__(cls, name, tensor_index_type, is_up=True, ignore_updown=False):
        # 创建符号对象，用于表示索引名称
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            # 自动生成索引名称并创建符号对象
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")
        
        # 将is_up和ignore_updown参数转换为符号表达式
        is_up = sympify(is_up)
        ignore_updown = sympify(ignore_updown)
        # 调用基类的构造函数创建实例
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up, ignore_updown)

    @property
    def ignore_updown(self):
        # 返回ignore_updown属性的值
        return self.args[3]

    def __neg__(self):
        # 创建一个新的WildTensorIndex对象，表示相反的变量（协变或逆变）
        t1 = WildTensorIndex(self.name, self.tensor_index_type,
                             (not self.is_up), self.ignore_updown)
        return t1

    def matches(self, expr, repl_dict=None, old=False):
        # 检查表达式是否是TensorIndex的实例
        if not isinstance(expr, TensorIndex):
            return None
        # 检查索引类型是否匹配
        if self.tensor_index_type != expr.tensor_index_type:
            return None
        # 如果不忽略协变/逆变性质，则检查是否匹配
        if not self.ignore_updown:
            if self.is_up != expr.is_up:
                return None
        
        # 创建一个替换字典，将当前对象映射到给定的表达式
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        
        repl_dict[self] = expr
        return repl_dict


class _WildTensExpr(Basic):
    """
    INTERNAL USE ONLY
    
    This is an object that helps with replacement of WildTensors in expressions.
    When this object is set as the tensor_head of a WildTensor, it replaces the
    WildTensor by a TensExpr (passed when initializing this object).
    
    Examples
    ========
    >>> from sympy.tensor.tensor import WildTensorHead, TensorIndex, TensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    """
    # 创建一个名为 q 的 TensorIndex 对象，关联到 R3 空间
    >>> q = TensorIndex('q', R3)
    # 创建一个名为 K 的 TensorHead 对象，表示一个张量头部，其指标为 R3
    >>> K = TensorHead('K', [R3])
    # 打印表达式 ( K(p) ).replace( W(p), W(q)*W(-q)*W(p) )
    >>> print( ( K(p) ).replace( W(p), W(q)*W(-q)*W(p) ) )
    # 打印结果 K(R_0)*K(-R_0)*K(p)

    """
    # _WildTensExpr 类的初始化方法，接受一个 TensExpr 类型的表达式作为参数
    def __init__(self, expr):
        # 如果传入的表达式不是 TensExpr 类型，则抛出类型错误
        if not isinstance(expr, TensExpr):
            raise TypeError("_WildTensExpr expects a TensExpr as argument")
        # 将传入的表达式存储在实例变量 expr 中
        self.expr = expr

    # 使实例对象能够像函数一样被调用，接受一系列索引作为参数
    def __call__(self, *indices):
        # 使用表达式对象的 get_free_indices 方法获取其自由索引，并与传入的索引构成映射关系，返回索引替换后的新表达式
        return self.expr._replace_indices(dict(zip(self.expr.get_free_indices(), indices)))

    # 定义负号运算，返回表达式乘以负一的结果
    def __neg__(self):
        return self.func(self.expr*S.NegativeOne)

    # 定义绝对值运算，但抛出未实现错误
    def __abs__(self):
        raise NotImplementedError

    # 定义加法运算，要求另一个操作数与当前对象具有相同的类型，返回两个表达式相加后的结果
    def __add__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(self.expr+other.expr)

    # 定义反向加法运算，要求另一个操作数与当前对象具有相同的类型，返回两个表达式相加后的结果
    def __radd__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(other.expr+self.expr)

    # 定义减法运算，返回当前对象加上另一个对象的负数的结果
    def __sub__(self, other):
        return self + (-other)

    # 定义反向减法运算，返回另一个对象加上当前对象的负数的结果
    def __rsub__(self, other):
        return other + (-self)

    # 定义乘法运算，但抛出未实现错误
    def __mul__(self, other):
        raise NotImplementedError

    # 定义反向乘法运算，但抛出未实现错误
    def __rmul__(self, other):
        raise NotImplementedError

    # 定义真除法运算，但抛出未实现错误
    def __truediv__(self, other):
        raise NotImplementedError

    # 定义反向真除法运算，但抛出未实现错误
    def __rtruediv__(self, other):
        raise NotImplementedError

    # 定义乘幂运算，但抛出未实现错误
    def __pow__(self, other):
        raise NotImplementedError

    # 定义反向乘幂运算，但抛出未实现错误
    def __rpow__(self, other):
        raise NotImplementedError
# 根据给定的对象 p，如果它是 TensExpr 类型的，则调用它的 canon_bp 方法，否则直接返回 p
def canon_bp(p):
    """
    Butler-Portugal canonicalization. See ``tensor_can.py`` from the
    combinatorics module for the details.
    """
    if isinstance(p, TensExpr):
        return p.canon_bp()
    return p


# 对输入的多个张量对象进行乘积计算，返回它们的乘积
def tensor_mul(*a):
    """
    product of tensors
    """
    if not a:
        return TensMul.from_data(S.One, [], [], [])
    # 取第一个张量对象作为初始值 t
    t = a[0]
    # 遍历剩余的张量对象，依次与 t 相乘
    for tx in a[1:]:
        t = t * tx
    return t


# 将给定的 Riemann 张量 t_r 替换为等价表达式，满足循环恒等式
def riemann_cyclic_replace(t_r):
    """
    replace Riemann tensor with an equivalent expression

    ``R(m,n,p,q) -> 2/3*R(m,n,p,q) - 1/3*R(m,q,n,p) + 1/3*R(m,p,n,q)``
    """
    # 按照自由指标的第二个元素排序
    free = sorted(t_r.free, key=lambda x: x[1])
    m, n, p, q = [x[0] for x in free]
    # 计算替换后的各个部分
    t0 = t_r * Rational(2, 3)
    t1 = -t_r.substitute_indices((m, m), (n, q), (p, n), (q, p)) * Rational(1, 3)
    t2 = t_r.substitute_indices((m, m), (n, p), (p, n), (q, q)) * Rational(1, 3)
    t3 = t0 + t1 + t2
    return t3


# 将输入的张量对象 t2 中的每个 Riemann 张量替换为满足循环恒等式的等价表达式
def riemann_cyclic(t2):
    """
    Replace each Riemann tensor with an equivalent expression
    satisfying the cyclic identity.

    This trick is discussed in the reference guide to Cadabra.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, riemann_cyclic, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
    >>> R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    >>> t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    >>> riemann_cyclic(t)
    0
    """
    # 对 t2 进行展开
    t2 = t2.expand()
    if isinstance(t2, (TensMul, Tensor)):
        args = [t2]
    else:
        args = t2.args
    # 将每个张量对象的每个成分拆分为基础张量的乘积，并对每个乘积中的 Riemann 张量进行替换
    a1 = [x.split() for x in args]
    a2 = [[riemann_cyclic_replace(tx) for tx in y] for y in a1]
    # 将替换后的结果重新组合成张量对象
    a3 = [tensor_mul(*v) for v in a2]
    # 对结果进行简化
    t3 = TensAdd(*a3).doit()
    if not t3:
        return t3
    else:
        return canon_bp(t3)


# 返回一个索引类型为 index_type 的张量的线条位置列表 lines、迹线列表 traces 和其余张量元素 rest
def get_lines(ex, index_type):
    """
    Returns ``(lines, traces, rest)`` for an index type,
    where ``lines`` is the list of list of positions of a matrix line,
    ``traces`` is the list of list of traced matrix lines,
    ``rest`` is the rest of the elements of the tensor.
    """
    # 定义一个内部函数 _join_lines，用于合并输入列表 a 中符合特定条件的元素
    def _join_lines(a):
        i = 0
        # 遍历列表 a 中的元素
        while i < len(a):
            x = a[i]
            xend = x[-1]
            xstart = x[0]
            hit = True
            # 在当前元素 x 的末尾或开头找到匹配的元素，进行合并操作，直到没有匹配项为止
            while hit:
                hit = False
                # 遍历列表 a 中当前元素之后的元素
                for j in range(i + 1, len(a)):
                    if j >= len(a):
                        break
                    # 若下一个元素的开头与当前元素的末尾匹配，则将其合并到当前元素末尾
                    if a[j][0] == xend:
                        hit = True
                        x.extend(a[j][1:])
                        xend = x[-1]
                        a.pop(j)
                        continue
                    # 若下一个元素的末尾与当前元素的开头匹配，则将其合并到当前元素开头
                    if a[j][0] == xstart:
                        hit = True
                        a[i] = reversed(a[j][1:]) + x
                        x = a[i]
                        xstart = a[i][0]
                        a.pop(j)
                        continue
                    # 若下一个元素的末尾与当前元素的末尾匹配，则将其逆序合并到当前元素末尾
                    if a[j][-1] == xend:
                        hit = True
                        x.extend(reversed(a[j][:-1]))
                        xend = x[-1]
                        a.pop(j)
                        continue
                    # 若下一个元素的开头与当前元素的末尾匹配，则将其逆序合并到当前元素开头
                    if a[j][-1] == xstart:
                        hit = True
                        a[i] = a[j][:-1] + x
                        x = a[i]
                        xstart = x[0]
                        a.pop(j)
                        continue
            i += 1
        return a

    # 从 ex 对象中获取参数列表
    arguments = ex.args
    # 初始化一个空字典 dt
    dt = {}
    # 遍历 ex.args 中的每个元素 c
    for c in ex.args:
        # 如果 c 不是 TensExpr 类型的对象，则跳过当前循环
        if not isinstance(c, TensExpr):
            continue
        # 如果 dt 字典中已经包含了 c，则跳过当前循环
        if c in dt:
            continue
        # 获取 c 对象的索引类型列表
        index_types = c.index_types
        # 初始化一个空列表 a
        a = []
        # 遍历索引类型列表 index_types 中的每个元素
        for i in range(len(index_types)):
            # 如果当前索引类型与指定的索引类型 index_type 相同，则将索引位置添加到列表 a 中
            if index_types[i] is index_type:
                a.append(i)
        # 如果列表 a 的长度超过了 2，则抛出 ValueError 异常
        if len(a) > 2:
            raise ValueError('at most two indices of type %s allowed' % index_type)
        # 如果列表 a 的长度为 2，则将 c 对象作为键，a 列表作为值存入 dt 字典中
        if len(a) == 2:
            dt[c] = a

    # 初始化空列表 lines、traces 和 traces1
    lines = []
    traces = []
    traces1 = []

    # TODO: 是否需要添加 dum_to_components_map 的实现？
    for p0, p1, c0, c1 in ex.dum_in_args:
        # 遍历 ex.dum_in_args 中的每个元组，元组包含四个值 p0, p1, c0, c1
        if arguments[c0] not in dt:
            # 如果 arguments[c0] 不在字典 dt 的键中，则跳过当前循环
            continue
        if c0 == c1:
            # 如果 c0 等于 c1，则将 [c0] 添加到 traces 列表中，并跳过当前循环
            traces.append([c0])
            continue
        # 获取 arguments[c0] 和 arguments[c1] 对应的值并命名为 ta0 和 ta1
        ta0 = dt[arguments[c0]]
        ta1 = dt[arguments[c1]]
        if p0 not in ta0:
            # 如果 p0 不在 ta0 中，则跳过当前循环
            continue
        if ta0.index(p0) == ta1.index(p1):
            # 如果 ta0 中 p0 的索引等于 ta1 中 p1 的索引，则抛出 NotImplementedError 异常
            # 处理这种情况可以添加一个置换标志
            raise NotImplementedError
        # 根据 p0 是否等于 ta0[1] 选择 b0, b1 的顺序
        ta0 = dt[arguments[c0]]
        b0, b1 = (c0, c1) if p0 == ta0[1] else (c1, c0)
        # 复制 lines 到 lines1
        lines1 = lines[:]
        for line in lines:
            if line[-1] == b0:
                if line[0] == b1:
                    # 如果 line 的最后一个元素等于 b0，且第一个元素等于 b1
                    # 找到 line 中最小值的索引并将其添加到 traces 中
                    n = line.index(min(line))
                    traces1.append(line)
                    # 将 line 分成两部分并重新排序，添加到 traces 中
                    traces.append(line[n:] + line[:n])
                else:
                    # 否则将 b1 添加到 line 中
                    line.append(b1)
                break
            elif line[0] == b1:
                # 如果 line 的第一个元素等于 b1，则在 line 的开头插入 b0
                line.insert(0, b0)
                break
        else:
            # 如果未找到符合条件的 line，则将 [b0, b1] 添加到 lines1 中
            lines1.append([b0, b1])

        # 从 lines1 中移除在 traces1 中已经存在的 lines，并将结果赋给 lines
        lines = [x for x in lines1 if x not in traces1]
        # 将 lines 中的线路连接起来
        lines = _join_lines(lines)
    # 初始化 rest 列表
    rest = []
    # 遍历 lines 中的每个 line
    for line in lines:
        # 遍历 line 中的每个元素并将其添加到 rest 中
        for y in line:
            rest.append(y)
    # 遍历 traces 中的每个 line
    for line in traces:
        # 遍历 line 中的每个元素并将其添加到 rest 中
        for y in line:
            rest.append(y)
    # 从 arguments 的长度中找出不在 rest 中的索引，并赋给 rest
    rest = [x for x in range(len(arguments)) if x not in rest]

    # 返回 lines, traces, rest 三个列表
    return lines, traces, rest
# 如果输入的对象不是 TensExpr 类型，则返回空元组
def get_free_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    # 调用对象的 get_free_indices 方法获取自由指标
    return t.get_free_indices()


# 如果输入的对象不是 TensExpr 类型，则返回空元组
def get_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    # 调用对象的 get_indices 方法获取所有指标
    return t.get_indices()


# 如果输入的对象不是 TensExpr 类型，则返回空列表；否则返回所有虚拟指标列表中不在自由指标列表中的指标
def get_dummy_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    # 获取所有指标和自由指标列表
    inds = t.get_indices()
    free = t.get_free_indices()
    # 返回非自由指标的虚拟指标列表
    return [i for i in inds if i not in free]


# 如果输入对象是 TensExpr 类型，则返回其 _index_structure 属性；否则返回一个空的 _IndexStructure 对象
def get_index_structure(t):
    if isinstance(t, TensExpr):
        return t._index_structure
    return _IndexStructure([], [], [], [])


# 根据输入对象的类型返回相应的系数
def get_coeff(t):
    if isinstance(t, Tensor):
        return S.One
    if isinstance(t, TensMul):
        return t.coeff
    if isinstance(t, TensExpr):
        # 如果输入对象是 TensExpr 类型，则引发 ValueError 异常，表示该表达式没有相关的系数
        raise ValueError("no coefficient associated to this tensor expression")
    return t


# 如果输入对象是 TensExpr 类型，则调用其 contract_metric 方法进行度规收缩；否则返回输入对象本身
def contract_metric(t, g):
    if isinstance(t, TensExpr):
        return t.contract_metric(g)
    return t


# 如果输入对象不是 TensExpr 类型，则直接返回输入对象
# 如果输入对象是 Tensor 或 TensMul 类型，则调用 perm2tensor 方法返回一个新的张量，根据给定的排列 g 进行排列
# 如果 g 的最后一个元素不等于 g 长度减一，则返回新张量的负数
def perm2tensor(t, g, is_canon_bp=False):
    """
    Returns the tensor corresponding to the permutation ``g``

    For further details, see the method in ``TIDS`` with the same name.
    """
    if not isinstance(t, TensExpr):
        return t
    elif isinstance(t, (Tensor, TensMul)):
        nim = get_index_structure(t).perm2tensor(g, is_canon_bp=is_canon_bp)
        res = t._set_new_index_structure(nim, is_canon_bp=is_canon_bp)
        if g[-1] != len(g) - 1:
            return -res
        return res
    # 如果 t 是 TensExpr 类型且不是 Tensor 或 TensMul 类型，则引发 NotImplementedError
    raise NotImplementedError()


# 如果输入对象不是 TensExpr 类型，则直接返回输入对象
# 否则调用输入对象的 substitute_indices 方法替换指标
def substitute_indices(t, *index_tuples):
    if not isinstance(t, TensExpr):
        return t
    return t.substitute_indices(*index_tuples)


# 如果输入表达式是 TensExpr 类型，则调用其 _expand 方法进行展开
# 否则调用表达式的 expand 方法进行展开
def _expand(expr, **kwargs):
    if isinstance(expr, TensExpr):
        return expr._expand(**kwargs)
    else:
        return expr.expand(**kwargs)


# 返回输入表达式中所有的通配符对象的列表
def _get_wilds(expr):
    return list(expr.atoms(Wild, WildFunction, WildTensor, WildTensorIndex, WildTensorHead))


# 返回一个函数，该函数对表达式进行处理以确保其属于指定的 TensExpr 类型
def get_postprocessor(cls):
    def _postprocessor(expr):
        tens_class = {Mul: TensMul, Add: TensAdd}[cls]
        # 如果表达式的任何参数是 TensExpr 类型，则返回一个新的 tens_class 类型的对象
        if any(isinstance(a, TensExpr) for a in expr.args):
            return tens_class(*expr.args)
        else:
            return expr

    return _postprocessor

# 将 TensExpr 类型的构造后处理映射到指定的函数列表，例如 Mul 对应的 get_postprocessor(Mul)
Basic._constructor_postprocessor_mapping[TensExpr] = {
    "Mul": [get_postprocessor(Mul)],
}
```