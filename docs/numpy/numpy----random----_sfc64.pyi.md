# `D:\src\scipysrc\numpy\numpy\random\_sfc64.pyi`

```py
# 引入必要的类型定义
from typing import TypedDict

# 引入必要的数据类型和类
from numpy import uint64
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy._typing import NDArray, _ArrayLikeInt_co

# 定义 SFC64 内部状态的字典类型
class _SFC64Internal(TypedDict):
    state: NDArray[uint64]

# 定义 SFC64 状态的字典类型
class _SFC64State(TypedDict):
    bit_generator: str  # 字符串类型的比特生成器名称
    state: _SFC64Internal  # SFC64 内部状态的字典
    has_uint32: int  # 整数，标志是否具有 uint32 数值
    uinteger: int  # 整数，保存 uinteger 值

# SFC64 类继承自 BitGenerator 类
class SFC64(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
        # 初始化函数，接受种子参数，可以是 None、_ArrayLikeInt_co 类型或 SeedSequence 类型

    @property
    def state(
        self,
    ) -> _SFC64State: ...
        # state 属性的 getter 方法，返回 _SFC64State 类型的对象

    @state.setter
    def state(
        self,
        value: _SFC64State,
    ) -> None: ...
        # state 属性的 setter 方法，接受 _SFC64State 类型的对象作为参数，无返回值
```