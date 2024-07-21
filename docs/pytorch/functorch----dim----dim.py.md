# `.\pytorch\functorch\dim\dim.py`

```
python
# 导入dis模块，用于反汇编Python字节码
import dis
# 导入inspect模块，用于检查和分析Python对象
import inspect

# 导入dataclass装饰器，用于定义数据类
from dataclasses import dataclass
# 导入Union类型，用于声明一个变量可以是多种类型中的一种
from typing import Union

# 从当前包中导入DimList模块
from . import DimList

# 全局变量_vmap_levels，用于存储维度映射的级别信息
_vmap_levels = []


# 使用dataclass装饰器定义LevelInfo类，表示维度的信息
@dataclass
class LevelInfo:
    level: int  # 维度级别
    alive: bool = True  # 维度是否活跃


# 定义Dim类，表示一个维度对象
class Dim:
    def __init__(self, name: str, size: Union[None, int] = None):
        self.name = name  # 维度的名称
        self._size = None  # 维度的大小，默认为None
        self._vmap_level = None  # 维度映射的级别，默认为None
        if size is not None:
            self.size = size  # 如果指定了大小，则调用size属性设置大小

    def __del__(self):
        # 析构函数，用于在对象销毁时处理维度映射的级别
        if self._vmap_level is not None:
            _vmap_active_levels[self._vmap_stack].alive = False  # 设置对应维度映射级别为不活跃
            while (
                not _vmap_levels[-1].alive
                and current_level() == _vmap_levels[-1].level
            ):
                _vmap_decrement_nesting()  # 减少维度映射的嵌套级别
                _vmap_levels.pop()  # 移除最后一个维度映射信息

    @property
    def size(self):
        assert self.is_bound  # 断言维度是否已绑定
        return self._size  # 返回维度的大小

    @size.setter
    def size(self, size: int):
        from . import DimensionBindError  # 导入维度绑定错误类

        if self._size is None:
            # 如果维度大小为None，则设置维度大小，并增加维度映射的嵌套级别
            self._size = size
            self._vmap_level = _vmap_increment_nesting(size, "same")
            self._vmap_stack = len(_vmap_levels)
            _vmap_levels.append(LevelInfo(self._vmap_level))

        elif self._size != size:
            # 如果维度大小不为None且与指定大小不同，则抛出维度绑定错误
            raise DimensionBindError(
                f"Dim '{self}' previously bound to a dimension of size {self._size} cannot bind to a dimension of size {size}"
            )

    @property
    def is_bound(self):
        return self._size is not None  # 返回维度是否已绑定的布尔值

    def __repr__(self):
        return self.name  # 返回维度对象的名称


# 定义extract_name函数，用于从指令实例中提取名称
def extract_name(inst):
    assert inst.opname == "STORE_FAST" or inst.opname == "STORE_NAME"  # 断言操作名称为"STORE_FAST"或"STORE_NAME"
    return inst.argval  # 返回指令的参数值作为名称


_cache = {}  # 定义一个空的缓存字典


# 定义dims函数，用于处理维度列表
def dims(lists=0):
    frame = inspect.currentframe()  # 获取当前帧对象
    assert frame is not None  # 断言帧对象不为None
    calling_frame = frame.f_back  # 获取调用帧对象
    assert calling_frame is not None  # 断
    # 如果缓存中没有这个键（key），则执行以下操作
    if key not in _cache:
        # 计算指令列表中间的位置
        first = lasti // 2 + 1
        # 获取当前函数的字节码指令列表
        instructions = list(dis.get_instructions(calling_frame.f_code))
        # 取出中间位置的指令对象
        unpack = instructions[first]

        # 检查中间位置的指令操作码是否为 STORE_FAST 或 STORE_NAME
        if unpack.opname == "STORE_FAST" or unpack.opname == "STORE_NAME":
            # 如果是单个维度，而不是列表
            name = unpack.argval
            # 根据列表数目决定使用 Dim 还是 DimList 构造函数
            ctor = Dim if lists == 0 else DimList
            # 将一个匿名函数存入缓存，该函数会创建并返回一个 Dim 或 DimList 对象
            _cache[key] = lambda: ctor(name=name)
        else:
            # 如果中间位置的指令操作码是 UNPACK_SEQUENCE
            assert unpack.opname == "UNPACK_SEQUENCE"
            # 获取解包的元素个数
            ndims = unpack.argval
            # 提取出解包后的每个元素的名称
            names = tuple(
                extract_name(instructions[first + 1 + i]) for i in range(ndims)
            )
            # 计算第一个列表开始的位置
            first_list = len(names) - lists
            # 将一个匿名函数存入缓存，该函数会创建并返回一个元组，其中包含 Dim 或 DimList 对象
            _cache[key] = lambda: tuple(
                Dim(n) if i < first_list else DimList(name=n)
                for i, n in enumerate(names)
            )
    # 返回缓存中对应键的值，即创建的 Dim 或 DimList 对象
    return _cache[key]()
# 定义一个函数 `_dim_set`，用于处理维度信息的设置
def _dim_set(positional, arg):
    # 定义内部函数 `convert`，用于将参数转换为 `Dim` 对象
    def convert(a):
        # 如果参数已经是 `Dim` 类型，则直接返回
        if isinstance(a, Dim):
            return a
        else:
            # 否则，断言参数是整数，并从 `positional` 中获取对应的 `Dim` 对象
            assert isinstance(a, int)
            return positional[a]

    # 如果 `arg` 参数为 `None`，直接返回 `positional`
    if arg is None:
        return positional
    # 如果 `arg` 不是 `Dim` 或整数类型，则将其元素逐个转换为 `Dim` 对象并返回元组
    elif not isinstance(arg, (Dim, int)):
        return tuple(convert(a) for a in arg)
    else:
        # 否则，将 `arg` 转换为 `Dim` 对象并返回单元素元组
        return (convert(arg),)
```