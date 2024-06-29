# `D:\src\scipysrc\numpy\numpy\random\bit_generator.pyi`

```py
import abc  # 导入抽象基类模块
from threading import Lock  # 导入线程锁对象
from collections.abc import Callable, Mapping, Sequence  # 导入抽象基类的Callable、Mapping、Sequence
from typing import (  # 导入类型提示相关模块
    Any,
    NamedTuple,
    TypedDict,
    TypeVar,
    overload,
    Literal,
)

from numpy import dtype, uint32, uint64  # 导入numpy的dtype、uint32和uint64类型
from numpy._typing import (  # 导入numpy的类型提示模块
    NDArray,
    _ArrayLikeInt_co,
    _ShapeLike,
    _SupportsDType,
    _UInt32Codes,
    _UInt64Codes,
)

_T = TypeVar("_T")  # 定义泛型变量_T

_DTypeLikeUint32 = (  # 定义_DTypeLikeUint32类型，支持uint32类型
    dtype[uint32]
    | _SupportsDType[dtype[uint32]]
    | type[uint32]
    | _UInt32Codes
)
_DTypeLikeUint64 = (  # 定义_DTypeLikeUint64类型，支持uint64类型
    dtype[uint64]
    | _SupportsDType[dtype[uint64]]
    | type[uint64]
    | _UInt64Codes
)

class _SeedSeqState(TypedDict):  # 定义名为_SeedSeqState的字典类型
    entropy: None | int | Sequence[int]  # 键为entropy，值可以为None、int或int序列
    spawn_key: tuple[int, ...]  # 键为spawn_key，值为int类型的元组
    pool_size: int  # 键为pool_size，值为int类型
    n_children_spawned: int  # 键为n_children_spawned，值为int类型

class _Interface(NamedTuple):  # 定义名为_Interface的命名元组
    state_address: Any  # 元组字段state_address的类型为Any
    state: Any  # 元组字段state的类型为Any
    next_uint64: Any  # 元组字段next_uint64的类型为Any
    next_uint32: Any  # 元组字段next_uint32的类型为Any
    next_double: Any  # 元组字段next_double的类型为Any
    bit_generator: Any  # 元组字段bit_generator的类型为Any

class ISeedSequence(abc.ABC):  # 定义名为ISeedSequence的抽象基类
    @abc.abstractmethod
    def generate_state(  # 抽象方法generate_state，生成状态
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...  # 参数n_words为int类型，dtype为_DTypeLikeUint32或_DTypeLikeUint64类型
    ) -> NDArray[uint32 | uint64]: ...  # 返回类型为NDArray[uint32 | uint64]

class ISpawnableSeedSequence(ISeedSequence):  # 定义名为ISpawnableSeedSequence的抽象基类，继承自ISeedSequence
    @abc.abstractmethod
    def spawn(self: _T, n_children: int) -> list[_T]: ...  # 抽象方法spawn，生成_T类型的列表，参数n_children为int类型

class SeedlessSeedSequence(ISpawnableSeedSequence):  # 定义名为SeedlessSeedSequence的类，继承自ISpawnableSeedSequence
    def generate_state(  # 实现generate_state方法，生成状态
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...  # 参数n_words为int类型，dtype为_DTypeLikeUint32或_DTypeLikeUint64类型
    ) -> NDArray[uint32 | uint64]: ...  # 返回类型为NDArray[uint32 | uint64]

    def spawn(self: _T, n_children: int) -> list[_T]: ...  # 实现spawn方法，生成_T类型的列表，参数n_children为int类型

class SeedSequence(ISpawnableSeedSequence):  # 定义名为SeedSequence的类，继承自ISpawnableSeedSequence
    entropy: None | int | Sequence[int]  # 属性entropy，可以为None、int或int序列
    spawn_key: tuple[int, ...]  # 属性spawn_key，为int类型的元组
    pool_size: int  # 属性pool_size，为int类型
    n_children_spawned: int  # 属性n_children_spawned，为int类型
    pool: NDArray[uint32]  # 属性pool，类型为NDArray[uint32]

    def __init__(  # 构造方法__init__
        self,
        entropy: None | int | Sequence[int] | _ArrayLikeInt_co = ...,  # 参数entropy，可以为None、int、int序列或_ArrayLikeInt_co类型
        *,
        spawn_key: Sequence[int] = ...,  # 关键字参数spawn_key，类型为int类型的序列
        pool_size: int = ...,  # 关键字参数pool_size，类型为int
        n_children_spawned: int = ...,  # 关键字参数n_children_spawned，类型为int
    ) -> None: ...  # 返回类型为None

    def __repr__(self) -> str: ...  # 定义__repr__方法，返回类型为str

    @property
    def state(  # 属性state，返回类型为_SeedSeqState
        self,
    ) -> _SeedSeqState: ...

    def generate_state(  # 实现generate_state方法，生成状态
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...  # 参数n_words为int类型，dtype为_DTypeLikeUint32或_DTypeLikeUint64类型
    ) -> NDArray[uint32 | uint64]: ...

    def spawn(self, n_children: int) -> list[SeedSequence]: ...  # 实现spawn方法，生成SeedSequence类型的列表，参数n_children为int类型

class BitGenerator(abc.ABC):  # 定义名为BitGenerator的抽象基类
    lock: Lock  # 属性lock，类型为Lock对象

    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...  # 构造方法__init__

    def __getstate__(self) -> tuple[dict[str, Any], ISeedSequence]: ...  # 定义__getstate__方法，返回类型为元组

    def __setstate__(  # 定义__setstate__方法
            self, state_seed_seq: dict[str, Any] | tuple[dict[str, Any], ISeedSequence]  # 参数state_seed_seq，类型为dict[str, Any]或元组
    ) -> None: ...

    def __reduce__(  # 定义__reduce__方法
        self,
    ) -> tuple[  # 返回类型为元组
        Callable[[str], BitGenerator],  # 第一个元素为函数类型，接受str参数，返回BitGenerator对象
        tuple[str],  # 第二个元素为元组类型，包含一个str类型的元素
        tuple[dict[str, Any], ISeedSequence]  # 第三个元素为元组类型，包含一个dict[str, Any]和一个ISeedSequence类型的元素
    ]: ...

    @abc.abstractmethod
    @property
    def state(self) -> Mapping[str, Any]: ...  # 抽象属性state，返回类型为Mapping[str, Any]

    @state.setter
    def state(self, value: Mapping[str, Any]) -> None: ...  # state的setter方法，参数value为Mapping[str, Any]类型，返回类型为None

    @property
    def seed_seq(self) -> ISeedSequence: ...  # 属性seed_seq，返回类型为ISeedSequence
    # 定义一个方法 spawn，用于生成指定数量的子生成器
    def spawn(self, n_children: int) -> list[BitGenerator]: ...
    
    # 重载定义 random_raw 方法，返回类型为 int，用于特定情况下的随机数生成
    @overload
    def random_raw(self, size: None = ..., output: Literal[True] = ...) -> int: ...
    
    # 重载定义 random_raw 方法，返回类型为 NDArray[uint64]，用于特定情况下的随机数生成
    @overload
    def random_raw(self, size: _ShapeLike = ..., output: Literal[True] = ...) -> NDArray[uint64]: ...
    
    # 重载定义 random_raw 方法，返回类型为 None，用于特定情况下的随机数生成
    @overload
    def random_raw(self, size: None | _ShapeLike = ..., output: Literal[False] = ...) -> None: ...
    
    # 定义一个 _benchmark 方法，用于执行性能基准测试，接受计数器和方法名作为参数
    def _benchmark(self, cnt: int, method: str = ...) -> None: ...
    
    # 定义一个属性方法 ctypes，返回类型为 _Interface，用于访问 ctypes 接口
    @property
    def ctypes(self) -> _Interface: ...
    
    # 定义一个属性方法 cffi，返回类型为 _Interface，用于访问 cffi 接口
    @property
    def cffi(self) -> _Interface: ...
```