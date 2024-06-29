# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_getattr.py`

```
from importlib import import_module  # 导入模块动态加载函数 import_module
from pkgutil import walk_packages  # 导入包工具模块的 walk_packages 函数

import matplotlib  # 导入 matplotlib 库
import pytest  # 导入 pytest 测试框架

# 获取所有 matplotlib 子模块的名称，
# 排除单元测试和私有模块。
module_names = [
    m.name
    for m in walk_packages(
        path=matplotlib.__path__, prefix=f'{matplotlib.__name__}.'
    )
    if not m.name.startswith(__package__)  # 排除当前包名下的子模块
    and not any(x.startswith('_') for x in m.name.split('.'))  # 排除私有模块
]


@pytest.mark.parametrize('module_name', module_names)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::ImportWarning')
def test_getattr(module_name):
    """
    Test that __getattr__ methods raise AttributeError for unknown keys.
    See #20822, #20855.
    """
    try:
        module = import_module(module_name)  # 动态导入当前模块名对应的模块
    except (ImportError, RuntimeError, OSError) as e:
        # 如果因缺少依赖项而无法导入模块，则跳过测试
        pytest.skip(f'Cannot import {module_name} due to {e}')

    key = 'THIS_SYMBOL_SHOULD_NOT_EXIST'
    if hasattr(module, key):  # 如果模块存在指定的属性名
        delattr(module, key)  # 删除模块中的指定属性名
```