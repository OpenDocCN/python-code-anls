# `.\pytorch\torch\testing\_internal\optests\__init__.py`

```
# 忽略 mypy 的类型检查错误

# 导入 make_fx 模块中的 make_fx_check 函数
from .make_fx import make_fx_check

# 导入 aot_autograd 模块中的 aot_autograd_check 和 _test_aot_autograd_forwards_backwards_helper 函数
from .aot_autograd import aot_autograd_check, _test_aot_autograd_forwards_backwards_helper

# 导入 fake_tensor 模块中的 fake_check 函数
from .fake_tensor import fake_check

# 导入 autograd_registration 模块中的 autograd_registration_check 函数
from .autograd_registration import autograd_registration_check

# 导入 generate_tests 模块中的 generate_opcheck_tests, opcheck, OpCheckError, dontGenerateOpCheckTests, is_inside_opcheck_mode 函数或类
from .generate_tests import generate_opcheck_tests, opcheck, OpCheckError, dontGenerateOpCheckTests, is_inside_opcheck_mode
```