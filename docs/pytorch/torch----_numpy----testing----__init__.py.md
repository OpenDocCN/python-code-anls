# `.\pytorch\torch\_numpy\testing\__init__.py`

```
# 忽略类型检查错误，适用于mypy工具
# 从本地的utils模块中导入多个函数和变量
from .utils import (
    _gen_alignment_data,           # 导入_gen_alignment_data函数
    assert_,                       # 导入assert_函数
    assert_allclose,               # 导入assert_allclose函数
    assert_almost_equal,           # 导入assert_almost_equal函数
    assert_array_almost_equal,     # 导入assert_array_almost_equal函数
    assert_array_equal,            # 导入assert_array_equal函数
    assert_array_less,             # 导入assert_array_less函数
    assert_equal,                  # 导入assert_equal函数
    assert_raises_regex,           # 导入assert_raises_regex函数
    assert_warns,                  # 导入assert_warns函数
    HAS_REFCOUNT,                  # 导入HAS_REFCOUNT变量
    IS_WASM,                       # 导入IS_WASM变量
    suppress_warnings              # 导入suppress_warnings函数
)

# from .testing import assert_allclose    # FIXME
# 从本地的testing模块中导入assert_allclose函数，但此行代码被注释掉并标记为待修复问题
```