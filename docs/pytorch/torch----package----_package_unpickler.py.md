# `.\pytorch\torch\package\_package_unpickler.py`

```py
# mypy: allow-untyped-defs
# 导入兼容的 pickle 模块
import _compat_pickle
# 导入标准的 pickle 模块
import pickle

# 从当前包的 importer 模块中导入 Importer 类
from .importer import Importer

# 定义一个自定义的 unpickler 类，继承自 pickle._Unpickler
class PackageUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    """Package-aware unpickler.

    This behaves the same as a normal unpickler, except it uses `importer` to
    find any global names that it encounters while unpickling.
    """

    # 初始化方法，接受一个 importer 实例作为参数
    def __init__(self, importer: Importer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._importer = importer

    # 查找类的方法，用于根据模块名和类名查找相应的类
    def find_class(self, module, name):
        # 若协议版本小于 3 并且需要修复导入
        if self.proto < 3 and self.fix_imports:  # type: ignore[attr-defined]
            # 检查是否有名字映射，用于兼容旧版本的 pickle
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            # 检查是否有模块映射，用于兼容旧版本的 pickle
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        # 使用 importer 实例的 import_module 方法导入指定模块
        mod = self._importer.import_module(module)
        # 返回模块中的指定类
        return getattr(mod, name)
```