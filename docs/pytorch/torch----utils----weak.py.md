# `.\pytorch\torch\utils\weak.py`

```
# mypy: allow-untyped-defs
# 导入 __future__ 模块中的 annotations 特性，允许未定义类型的函数签名
from __future__ import annotations

# 导入 weakref 模块，用于处理弱引用
import weakref
# 从 weakref 模块导入 ref 函数，用于创建对象的弱引用
from weakref import ref
# 导入 _weakrefset 模块中的 _IterationGuard 类型，类型注释时忽略属性定义
from _weakrefset import _IterationGuard  # type: ignore[attr-defined]
# 从 collections.abc 模块中导入 MutableMapping 和 Mapping 类型，用于定义映射类型
from collections.abc import MutableMapping, Mapping
# 导入 torch 模块中的 Tensor 类型
from torch import Tensor
# 导入 collections.abc 模块并命名为 _collections_abc，用于访问抽象集合类
import collections.abc as _collections_abc

# 将 ref 函数重命名为 WeakRef，简化后续代码中的使用
WeakRef = ref

# 定义模块中公开的符号列表
__all__ = ['TensorWeakRef', 'WeakIdRef', 'WeakIdKeyDictionary', 'WeakTensorKeyDictionary']


# This file defines a variant of WeakKeyDictionary that overrides the hashing
# behavior of the key to use object identity, rather than the builtin
# __eq__/__hash__ functions.  This is useful for Tensor weak keys, as their
# __eq__ implementation return a Tensor (elementwise equality), which means
# you can't use them directly with the WeakKeyDictionary in standard library.
#
# Our implementation strategy is to create a wrapper weak key object, which we
# use as a key in a stock Python dictionary.  This is similar to how weakref
# implements WeakKeyDictionary, but instead of using weakref.ref as the
# wrapper, we use a custom wrapper that has different __eq__ and __hash__
# behavior.  Note that we subsequently store this weak key directly in an
# ORDINARY dictionary, since the newly constructed WeakIdKey's only use would
# be a dictionary so it would have no strong references.  Ensuring that
# only live WeakIdKeys are in the map is handled by putting finalizers on the
# original key object.

# 文件定义了一个弱引用字典的变体，重写了键的哈希行为，使用对象标识而不是内建的 __eq__/__hash__ 函数。
# 这对于 Tensor 弱引用键非常有用，因为它们的 __eq__ 实现返回一个 Tensor（逐元素比较相等性），
# 这意味着你不能直接将它们与标准库中的 WeakKeyDictionary 一起使用。
#
# 我们的实现策略是创建一个包装的弱引用键对象，作为普通 Python 字典中的键使用。
# 这类似于 weakref 实现 WeakKeyDictionary，但我们使用的是具有不同 __eq__ 和 __hash__ 行为的自定义包装器。
# 注意，随后我们直接将这个弱引用键存储在普通字典中，因为新构建的 WeakIdKey 的唯一用途就是作为字典键，
# 所以它不会有强引用。确保映射中只有活动的 WeakIdKey 是通过在原始键对象上放置终结器来处理的。


# It is simpler to implement this with composition, but if we want to
# directly reuse the callback mechanism on weakref, we need the weakref
# and the key to be exactly the same object.  Reusing the callback mechanism
# minimizes the divergence between our implementation and Lib/weakref.py
#
# NB: Prefer using this when working with weakrefs of Tensors; e.g., do
# WeakIdRef(tensor) rather than weakref.ref(tensor); it handles a number of
# easy to get wrong cases transparently for you.

# 使用组合方式更简单实现，但如果我们想直接重用 weakref 上的回调机制，
# 我们需要 weakref 和键对象确切地是同一个对象。
# 重用回调机制可以最小化我们的实现与 Lib/weakref.py 之间的差异。

# 注意：在处理 Tensor 的弱引用时最好使用这个类；例如，使用 WeakIdRef(tensor) 而不是 weakref.ref(tensor)；
# 它会透明地处理一些容易出错的情况。
class WeakIdRef(weakref.ref):
    __slots__ = ['_id']

    def __init__(self, key, callback=None):
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = id(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, '_fix_weakref'):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id
    def __eq__(self, other):
        # 实现自定义对象的相等性比较方法
        # 一个看似吸引人但错误的替代实现是仅测试存储的 _ids 是否匹配。这可能导致ABA问题，例如：
        #
        #   a1 = A()
        #   w1 = WeakIdRef(a1)
        #   del a1
        #   a2 = A()  # 假设它获得了与 a1 相同的ID
        #   w2 = WeakIdRef(a2)
        #   print(w1 == w2)
        #
        # 这应该返回 False，因为 a1 和 a2 是无关的（而且 a1 已经失效）
        
        # 获取自身对象和其他对象的引用
        a = self()
        b = other()
        
        # 如果两个对象都不是 None，则直接比较它们的引用
        if a is not None and b is not None:
            return a is b
        
        # 如果有任一对象是 None，则比较对象自身的引用
        return self is other
# 这个类与 WeakIdRef 类似，但是相等性检查使用 hash() 而不是 id。
class _WeakHashRef(weakref.ref):
    __slots__ = ['_id']

    def __init__(self, key, callback=None):
        # 不同于标准的 weakref，它保留了原始对象的哈希语义，
        # 但是延迟了哈希调用，直到用户尝试对 weakref 进行哈希操作。
        # 在这里我们可以急切地缓存键的 id，因为我们确定这绝对是哈希方法。
        self._id = hash(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        # 特殊逻辑用于 Tensor PyObject 的复活
        if hasattr(r, '_fix_weakref'):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        # 使用哈希相等性来确定引用的相等性。
        # ScriptObject 实现了 __hash__ 方法返回封装的 IValue 的 id，
        # 因此这相当于进行身份比较。
        a = self()
        b = other()
        if a is not None and b is not None:
            return hash(a) == hash(b)
        return self is other

# 这段代码直接改编自 cpython/Lib/weakref.py
class WeakIdKeyDictionary(MutableMapping):
    def __init__(self, dict=None, ref_type=WeakIdRef):  # CHANGED
        self.data = {}  # 初始化一个空字典用于存储数据

        self.ref_type = ref_type  # CHANGED

        def remove(k, selfref=ref(self)):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass
        self._remove = remove  # 设置私有方法 _remove 用于移除键值对
        # 用于存储将要删除的死掉的 weakref（键）
        self._pending_removals = []
        self._iterating = set()  # 用集合存储正在迭代的键，防止并发修改
        self._dirty_len = False  # 表示字典长度是否已经被修改过
        if dict is not None:
            self.update(dict)  # 如果提供了初始字典，则更新到当前字典中

    def _commit_removals(self):
        # 注意：在修改字典之前我们不需要调用这个方法，
        # 因为死掉的 weakref 永远不会等于活的 weakref，即使它们引用相等的对象。
        # 然而，这意味着键可能已经被删除了。
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self):
        d = self.data
        # 清理掉已经不在字典中的键
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False
    def __delitem__(self, key):
        self._dirty_len = True  # 设置标志位，表示长度需要重新计算
        del self.data[self.ref_type(key)]  # 删除指定键对应的条目

    def __getitem__(self, key):
        return self.data[self.ref_type(key)]  # 获取指定键对应的值

    def __len__(self):
        if self._dirty_len and self._pending_removals:
            # 如果需要重新计算长度并且有待移除的键，执行清理操作以移除已显式删除的键
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)  # 返回当前有效键的数量

    def __repr__(self):
        return f"<{self.__class__.__name__} at {id(self):#x}>"  # 返回对象的字符串表示形式

    def __setitem__(self, key, value):
        self.data[self.ref_type(key, self._remove)] = value  # 设置指定键对应的值

    def copy(self):
        new = WeakIdKeyDictionary()  # 创建一个新的 WeakIdKeyDictionary 实例
        with _IterationGuard(self):  # 确保在迭代期间不会修改对象
            for key, value in self.data.items():
                o = key()  # 获取键所引用的对象
                if o is not None:
                    new[o] = value  # 将键-值对复制到新实例中
        return new  # 返回复制后的新实例

    __copy__ = copy  # 将 copy 方法赋值给 __copy__ 方法

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = self.__class__()  # 创建当前类的新实例
        with _IterationGuard(self):  # 确保在迭代期间不会修改对象
            for key, value in self.data.items():
                o = key()  # 获取键所引用的对象
                if o is not None:
                    new[o] = deepcopy(value, memo)  # 深度复制键-值对到新实例中
        return new  # 返回深度复制后的新实例

    def get(self, key, default=None):
        return self.data.get(self.ref_type(key), default)  # 获取指定键对应的值，如果键不存在则返回默认值

    def __contains__(self, key):
        try:
            wr = self.ref_type(key)  # 获取键对应的弱引用对象
        except TypeError:
            return False  # 如果无法获取弱引用，则键不存在
        return wr in self.data  # 判断弱引用是否在数据集合中

    def items(self):
        with _IterationGuard(self):  # 确保在迭代期间不会修改对象
            for wr, value in self.data.items():
                key = wr()  # 获取弱引用所指向的对象
                if key is not None:
                    yield key, value  # 返回键-值对的生成器

    def keys(self):
        with _IterationGuard(self):  # 确保在迭代期间不会修改对象
            for wr in self.data:
                obj = wr()  # 获取弱引用所指向的对象
                if obj is not None:
                    yield obj  # 返回对象的生成器作为键

    __iter__ = keys  # 将 keys 方法赋值给 __iter__ 方法

    def values(self):
        with _IterationGuard(self):  # 确保在迭代期间不会修改对象
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value  # 返回值的生成器

    def keyrefs(self):
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)  # 返回数据中所有弱引用对象的列表

    def popitem(self):
        self._dirty_len = True  # 设置标志位，表示长度需要重新计算
        while True:
            key, value = self.data.popitem()  # 弹出并返回数据中的一对键-值
            o = key()  # 获取键所引用的对象
            if o is not None:
                return o, value  # 如果对象存在，则返回对象和对应的值

    def pop(self, key, *args):
        self._dirty_len = True  # 设置标志位，表示长度需要重新计算
        return self.data.pop(self.ref_type(key), *args)  # 弹出并返回指定键对应的值，如果键不存在则返回默认值或引发 KeyError
    # 设置键值对，如果键不存在则插入默认值
    def setdefault(self, key, default=None):
        return self.data.setdefault(self.ref_type(key, self._remove), default)  # 调用引用类型处理方法后插入或获取默认值

    # 更新映射对象的内容
    def update(self, dict=None, **kwargs):
        d = self.data  # 获取当前映射对象的数据
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)  # 如果传入的不是映射类型，则转换为字典类型
            for key, value in dict.items():
                d[self.ref_type(key, self._remove)] = value  # 调用引用类型处理方法后更新键值对
        if len(kwargs):
            self.update(kwargs)  # 递归调用自身，更新剩余的关键字参数

    # 实现按位或运算符，更新当前映射对象和另一个映射对象的内容
    def __ior__(self, other):
        self.update(other)
        return self

    # 实现按位或运算符，返回一个新的映射对象，包含当前对象和另一个映射对象的内容
    def __or__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()  # 复制当前映射对象
            c.update(other)  # 更新新对象的内容
            return c
        return NotImplemented

    # 实现反向按位或运算符，返回一个新的映射对象，包含另一个映射对象和当前对象的内容
    def __ror__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()  # 创建当前类的新实例
            c.update(other)  # 更新新对象的内容
            c.update(self)  # 更新当前对象的内容
            return c
        return NotImplemented

    # 实现相等比较运算符，比较映射对象中键的标识是否相等
    # 默认映射相等性测试键是否相等，但我们希望测试键的标识是否相等
    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {id(k): v for k, v in other.items()}
# Convenience alias，为 WeakTensorKeyDictionary 提供方便的别名，使其等同于 WeakIdKeyDictionary
WeakTensorKeyDictionary = WeakIdKeyDictionary

class TensorWeakRef:
    """Wrapper around a weak ref of a Tensor that handles the _fix_weakref() call required when unwrapping a Tensor weakref."""
    
    ref: WeakRef[Tensor]  # 弱引用，指向一个 Tensor 对象的弱引用
    
    def __init__(self, tensor: Tensor):
        assert isinstance(tensor, Tensor)
        self.ref = weakref.ref(tensor)  # 创建对 Tensor 对象的弱引用

    def __call__(self):
        out = self.ref()  # 获取弱引用指向的 Tensor 对象
        if out is None:
            return out
        assert isinstance(out, Tensor)
        # TODO, add _fix_weakref type binding
        out._fix_weakref()  # type: ignore[attr-defined]，调用 Tensor 对象的 _fix_weakref 方法（类型注释告知类型检查忽略该属性的定义）
        return out
```