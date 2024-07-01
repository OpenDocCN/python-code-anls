# `.\numpy\numpy\random\_mt19937.pyi`

```py
# 导入必要的类型定义
from typing import TypedDict

# 导入必要的数据类型和函数
from numpy import uint32
from numpy.typing import NDArray
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy._typing import _ArrayLikeInt_co

# 定义用于存储 MT19937 内部状态的 TypedDict
class _MT19937Internal(TypedDict):
    key: NDArray[uint32]  # MT19937 算法的内部状态向量
    pos: int  # 当前状态向量的位置

# 定义用于存储 MT19937 完整状态的 TypedDict
class _MT19937State(TypedDict):
    bit_generator: str  # 使用的位生成器名称
    state: _MT19937Internal  # MT19937 的内部状态

# MT19937 类，继承自 BitGenerator
class MT19937(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None:
        ...

    def _legacy_seeding(self, seed: _ArrayLikeInt_co) -> None:
        ...

    def jumped(self, jumps: int = ...) -> MT19937:
        ...

    @property
    def state(self) -> _MT19937State:
        ...

    @state.setter
    def state(self, value: _MT19937State) -> None:
        ...
```