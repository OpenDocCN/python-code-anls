# `D:\src\scipysrc\numpy\numpy\random\_pcg64.pyi`

```py
# 导入必要的类型引用
from typing import TypedDict

# 导入需要的类和函数
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy._typing import _ArrayLikeInt_co

# 定义 PCG64 内部状态的类型字典
class _PCG64Internal(TypedDict):
    state: int  # PCG64 内部状态值
    inc: int    # PCG64 内部增量值

# 定义 PCG64 完整状态的类型字典
class _PCG64State(TypedDict):
    bit_generator: str      # 字符串类型的位生成器名称
    state: _PCG64Internal   # PCG64 内部状态字典
    has_uint32: int         # 是否有无符号整数类型标记
    uinteger: int           # 无符号整数值

# PCG64 类，继承自 BitGenerator 类
class PCG64(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    def jumped(self, jumps: int = ...) -> PCG64: ...
    
    @property
    def state(
        self,
    ) -> _PCG64State: ...
    
    @state.setter
    def state(
        self,
        value: _PCG64State,
    ) -> None: ...
    
    def advance(self, delta: int) -> PCG64: ...
    
# PCG64DXSM 类，继承自 BitGenerator 类
class PCG64DXSM(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    def jumped(self, jumps: int = ...) -> PCG64DXSM: ...
    
    @property
    def state(
        self,
    ) -> _PCG64State: ...
    
    @state.setter
    def state(
        self,
        value: _PCG64State,
    ) -> None: ...
    
    def advance(self, delta: int) -> PCG64DXSM: ...
```