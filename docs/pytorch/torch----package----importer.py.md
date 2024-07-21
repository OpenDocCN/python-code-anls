# `.\pytorch\torch\package\importer.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
    _getattribute,  # 导入 pickle 模块的 _getattribute 函数
    _Pickler,  # 导入 pickle 模块的 _Pickler 类
    whichmodule as _pickle_whichmodule,  # 导入 pickle 模块的 whichmodule 函数并重命名为 _pickle_whichmodule
)
from types import ModuleType  # 导入 ModuleType 类型
from typing import Any, Dict, List, Optional, Tuple  # 导入类型提示需要的类型

from ._mangling import demangle, get_mangle_prefix, is_mangled  # 导入局部模块中的函数和类

__all__ = ["ObjNotFoundError", "ObjMismatchError", "Importer", "OrderedImporter"]  # 设置在 from 模块中导入的公共名称列表


class ObjNotFoundError(Exception):
    """Raised when an importer cannot find an object by searching for its name."""

    pass


class ObjMismatchError(Exception):
    """Raised when an importer found a different object with the same name as the user-provided one."""

    pass


class Importer(ABC):
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """

    modules: Dict[str, ModuleType]  # 模块名称到 ModuleType 对象的映射

    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """
        pass

    def whichmodule(self, obj: Any, name: str) -> str:
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """
        module_name = getattr(obj, "__module__", None)  # 获取对象所属的模块名称
        if module_name is not None:
            return module_name

        # 通过遍历 self.modules 字典，查找对象所属的模块
        # 注意：为了避免动态导入触发其他模块的导入，对 self.modules 的副本进行迭代
        for module_name, module in self.modules.copy().items():
            if (
                module_name == "__main__"
                or module_name == "__mp_main__"  # 排除特定的模块名称
                or module is None
            ):
                continue
            try:
                if _getattribute(module, name)[0] is obj:  # 调用 pickle 模块的 _getattribute 函数查找对象
                    return module_name
            except AttributeError:
                pass

        return "__main__"  # 如果未找到，返回默认的模块名称
class _SysImporter(Importer):
    """An importer that implements the default behavior of Python."""

    def import_module(self, module_name: str):
        # 使用 importlib 模块导入指定名称的模块
        return importlib.import_module(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        # 调用 _pickle_whichmodule 函数来确定对象所在的模块
        return _pickle_whichmodule(obj, name)


sys_importer = _SysImporter()


class OrderedImporter(Importer):
    """A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    """

    def __init__(self, *args):
        self._importers: List[Importer] = list(args)

    def _is_torchpackage_dummy(self, module):
        """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImportere will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
        # 检查是否为 torch.package 中的空 PackageNode 模块
        if not getattr(module, "__torch_package__", False):
            return False
        if not hasattr(module, "__path__"):
            return False
        if not hasattr(module, "__file__"):
            return True
        return module.__file__ is None

    def import_module(self, module_name: str) -> ModuleType:
        # 依次尝试每个导入器，直到成功导入模块或者抛出异常
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, Importer):
                raise TypeError(
                    f"{importer} is not a Importer. "
                    "All importers in OrderedImporter must inherit from Importer."
                )
            try:
                # 使用当前导入器尝试导入指定名称的模块
                module = importer.import_module(module_name)
                # 检查导入的模块是否为 torch.package 中的空 PackageNode
                if self._is_torchpackage_dummy(module):
                    continue  # 如果是空的 PackageNode，则继续尝试下一个导入器
                return module  # 返回成功导入的模块
            except ModuleNotFoundError as err:
                last_err = err

        if last_err is not None:
            raise last_err  # 如果所有导入器均未成功导入模块，则抛出最后一个异常
        else:
            raise ModuleNotFoundError(module_name)  # 如果没有异常抛出，说明模块未找到

    def whichmodule(self, obj: Any, name: str) -> str:
        # 遍历所有导入器，确定对象所在的模块名称
        for importer in self._importers:
            module_name = importer.whichmodule(obj, name)
            if module_name != "__main__":
                return module_name

        return "__main__"  # 如果在所有导入器中都未找到对象所在的模块，则默认为 "__main__"
```