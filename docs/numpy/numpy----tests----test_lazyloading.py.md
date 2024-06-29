# `.\numpy\numpy\tests\test_lazyloading.py`

```
# 导入系统模块 sys
import sys
# 导入 importlib 库，用于动态加载模块
import importlib
# 从 importlib.util 中导入 LazyLoader、find_spec、module_from_spec 函数
from importlib.util import LazyLoader, find_spec, module_from_spec
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 使用 pytest 的装饰器，忽略 "The NumPy module was reloaded" 的警告
@pytest.mark.filterwarnings("ignore:The NumPy module was reloaded")
def test_lazy_load():
    # 删除 sys.modules 中的 "numpy" 模块，模拟重新加载 numpy
    old_numpy = sys.modules.pop("numpy")

    # 创建一个字典，用于保存以 "numpy." 开头的所有模块
    numpy_modules = {}
    # 遍历 sys.modules 的副本，将以 "numpy." 开头的模块从 sys.modules 中移除并保存到 numpy_modules 中
    for mod_name, mod in list(sys.modules.items()):
        if mod_name[:6] == "numpy.":
            numpy_modules[mod_name] = mod
            sys.modules.pop(mod_name)

    try:
        # 查找 numpy 模块的规范对象
        spec = find_spec("numpy")
        # 根据规范对象创建模块对象
        module = module_from_spec(spec)
        # 将新创建的模块对象添加到 sys.modules 中
        sys.modules["numpy"] = module
        # 创建 LazyLoader 对象，使用其加载器加载模块
        loader = LazyLoader(spec.loader)
        loader.exec_module(module)
        # 将加载后的 numpy 模块赋值给变量 np
        np = module

        # 测试导入子包 numpy.lib.recfunctions
        from numpy.lib import recfunctions

        # 测试触发导入 numpy 包
        np.ndarray

    finally:
        # 如果 old_numpy 不为空，将其重新添加到 sys.modules 中
        if old_numpy:
            sys.modules["numpy"] = old_numpy
            # 将之前移除的 numpy 模块及其子模块重新添加到 sys.modules 中
            sys.modules.update(numpy_modules)
```