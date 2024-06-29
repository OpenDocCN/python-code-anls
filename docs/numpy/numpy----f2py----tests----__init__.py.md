# `.\numpy\numpy\f2py\tests\__init__.py`

```py
# 从 numpy.testing 模块中导入 IS_WASM 和 IS_EDITABLE 常量
from numpy.testing import IS_WASM, IS_EDITABLE
# 从 pytest 模块中导入 pytest 函数或类

# 如果 IS_WASM 常量为真，则跳过当前测试，并提供相应的消息
if IS_WASM:
    pytest.skip(
        "WASM/Pyodide does not use or support Fortran",
        allow_module_level=True
    )

# 如果 IS_EDITABLE 常量为真，则跳过当前测试，并提供相应的消息
if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )
```