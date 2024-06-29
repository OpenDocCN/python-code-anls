# `D:\src\scipysrc\numpy\numpy\typing\tests\data\misc\extended_precision.pyi`

```py
import sys
# 导入系统模块

import numpy as np
# 导入 NumPy 库，并将其命名为 np
from numpy._typing import _80Bit, _96Bit, _128Bit, _256Bit
# 从 NumPy 的 _typing 模块导入特定的位数类型定义

if sys.version_info >= (3, 11):
    # 如果 Python 版本大于等于 3.11，使用标准库中的 assert_type
    from typing import assert_type
else:
    # 否则使用 typing_extensions 中的 assert_type
    from typing_extensions import assert_type

assert_type(np.uint128(), np.unsignedinteger[_128Bit])
# 断言 np.uint128() 的类型为 np.unsignedinteger[_128Bit]

assert_type(np.uint256(), np.unsignedinteger[_256Bit])
# 断言 np.uint256() 的类型为 np.unsignedinteger[_256Bit]

assert_type(np.int128(), np.signedinteger[_128Bit])
# 断言 np.int128() 的类型为 np.signedinteger[_128Bit]

assert_type(np.int256(), np.signedinteger[_256Bit])
# 断言 np.int256() 的类型为 np.signedinteger[_256Bit]

assert_type(np.float80(), np.floating[_80Bit])
# 断言 np.float80() 的类型为 np.floating[_80Bit]

assert_type(np.float96(), np.floating[_96Bit])
# 断言 np.float96() 的类型为 np.floating[_96Bit]

assert_type(np.float128(), np.floating[_128Bit])
# 断言 np.float128() 的类型为 np.floating[_128Bit]

assert_type(np.float256(), np.floating[_256Bit])
# 断言 np.float256() 的类型为 np.floating[_256Bit]

assert_type(np.complex160(), np.complexfloating[_80Bit, _80Bit])
# 断言 np.complex160() 的类型为 np.complexfloating[_80Bit, _80Bit]

assert_type(np.complex192(), np.complexfloating[_96Bit, _96Bit])
# 断言 np.complex192() 的类型为 np.complexfloating[_96Bit, _96Bit]

assert_type(np.complex256(), np.complexfloating[_128Bit, _128Bit])
# 断言 np.complex256() 的类型为 np.complexfloating[_128Bit, _128Bit]

assert_type(np.complex512(), np.complexfloating[_256Bit, _256Bit])
# 断言 np.complex512() 的类型为 np.complexfloating[_256Bit, _256Bit]
```