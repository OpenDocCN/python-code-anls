# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\modules.pyi`

```py
import numpy as np  # 导入 NumPy 库，使用 np 作为别名

np.testing.bob  # E: Module has no attribute，尝试访问 np.testing 模块中的 bob 属性，但该属性不存在
np.bob  # E: Module has no attribute，尝试访问 np 模块中的 bob 属性，但该属性不存在

# Stdlib modules in the namespace by accident
np.warnings  # E: Module has no attribute，尝试访问 np 模块中的 warnings 模块，但该模块不存在
np.sys  # E: Module has no attribute，尝试访问 np 模块中的 sys 模块，但该模块不存在
np.os  # E: Module "numpy" does not explicitly export，尝试访问 np 模块中的 os 模块，NumPy 不显式导出该模块
np.math  # E: Module has no attribute，尝试访问 np 模块中的 math 模块，但该模块不存在

# Public sub-modules that are not imported to their parent module by default;
# e.g. one must first execute `import numpy.lib.recfunctions`
np.lib.recfunctions  # E: Module has no attribute，尝试访问 np.lib 模块中的 recfunctions 模块，但该模块不存在

np.__NUMPY_SETUP__  # E: Module has no attribute，尝试访问 np 模块中的 __NUMPY_SETUP__ 属性，但该属性不存在
np.__deprecated_attrs__  # E: Module has no attribute，尝试访问 np 模块中的 __deprecated_attrs__ 属性，但该属性不存在
np.__expired_functions__  # E: Module has no attribute，尝试访问 np 模块中的 __expired_functions__ 属性，但该属性不存在
```