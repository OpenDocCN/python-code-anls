# `.\numpy\numpy\_pyinstaller\tests\__init__.py`

```
# 从 numpy.testing 模块导入 IS_WASM 和 IS_EDITABLE 常量
from numpy.testing import IS_WASM, IS_EDITABLE
# 导入 pytest 模块，用于测试框架

# 如果 IS_WASM 常量为真，跳过当前测试并显示相应信息
if IS_WASM:
    pytest.skip(
        "WASM/Pyodide does not use or support Fortran",
        allow_module_level=True
    )

# 如果 IS_EDITABLE 常量为真，跳过当前测试并显示相应信息
if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )
```