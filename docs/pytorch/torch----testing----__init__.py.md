# `.\pytorch\torch\testing\__init__.py`

```
# 导入 torch._C 模块中的 FileCheck 别名为 FileCheck
# （注意：通常不建议直接从内部模块导入，而应该从公共接口导入）
from torch._C import FileCheck as FileCheck

# 从当前包中导入 _utils 模块
from . import _utils

# 从当前包中导入 _comparison 模块中的 assert_allclose 函数和 assert_close 别名为 assert_close
# （注意：在导入时使用 as 关键字给函数取别名）
from ._comparison import assert_allclose, assert_close as assert_close

# 从当前包中导入 _creation 模块中的 make_tensor 函数
from ._creation import make_tensor as make_tensor
```