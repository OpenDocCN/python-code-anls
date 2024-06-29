# `D:\src\scipysrc\numpy\numpy\random\_philox.pyi`

```py
from typing import TypedDict  # 导入 TypedDict 类型提示

from numpy import uint64  # 从 numpy 库中导入 uint64 类型
from numpy.typing import NDArray  # 导入 NDArray 泛型类型
from numpy.random.bit_generator import BitGenerator, SeedSequence  # 从 numpy.random.bit_generator 导入 BitGenerator 和 SeedSequence 类型
from numpy._typing import _ArrayLikeInt_co  # 导入 _ArrayLikeInt_co 私有类型

class _PhiloxInternal(TypedDict):
    counter: NDArray[uint64]  # 定义 _PhiloxInternal 类型的 counter 属性为 uint64 类型的 NDArray
    key: NDArray[uint64]  # 定义 _PhiloxInternal 类型的 key 属性为 uint64 类型的 NDArray

class _PhiloxState(TypedDict):
    bit_generator: str  # 定义 _PhiloxState 类型的 bit_generator 属性为字符串类型
    state: _PhiloxInternal  # 定义 _PhiloxState 类型的 state 属性为 _PhiloxInternal 类型
    buffer: NDArray[uint64]  # 定义 _PhiloxState 类型的 buffer 属性为 uint64 类型的 NDArray
    buffer_pos: int  # 定义 _PhiloxState 类型的 buffer_pos 属性为整数类型
    has_uint32: int  # 定义 _PhiloxState 类型的 has_uint32 属性为整数类型
    uinteger: int  # 定义 _PhiloxState 类型的 uinteger 属性为整数类型

class Philox(BitGenerator):
    def __init__(
        self,
        seed: None | _ArrayLikeInt_co | SeedSequence = ...,  # 构造函数，初始化 seed 参数，可接受 None、_ArrayLikeInt_co 或 SeedSequence 类型，默认为 ...
        counter: None | _ArrayLikeInt_co = ...,  # 构造函数，初始化 counter 参数，可接受 None 或 _ArrayLikeInt_co 类型，默认为 ...
        key: None | _ArrayLikeInt_co = ...,  # 构造函数，初始化 key 参数，可接受 None 或 _ArrayLikeInt_co 类型，默认为 ...
    ) -> None: ...  # 构造函数返回 None，无具体实现

    @property
    def state(  # 定义状态属性 state 的 getter 方法
        self,
    ) -> _PhiloxState: ...  # 返回 _PhiloxState 类型的对象，无具体实现

    @state.setter
    def state(  # 定义状态属性 state 的 setter 方法
        self,
        value: _PhiloxState,  # 接受 _PhiloxState 类型的 value 参数
    ) -> None: ...  # setter 方法返回 None，无具体实现

    def jumped(self, jumps: int = ...) -> Philox: ...  # 定义 jumped 方法，接受整数类型的 jumps 参数，默认为 ...，返回 Philox 类型的对象，无具体实现

    def advance(self, delta: int) -> Philox: ...  # 定义 advance 方法，接受整数类型的 delta 参数，返回 Philox 类型的对象，无具体实现
```