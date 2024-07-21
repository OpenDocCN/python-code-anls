# `.\pytorch\torch\ao\nn\intrinsic\__init__.py`

```
# mypy: allow-untyped-defs
# 从当前包的 modules 子模块中导入所有内容，禁止检查 F403 错误
from .modules import *  # noqa: F403
# 从当前包的 modules 子模块中导入 _FusedModule 类
from .modules.fused import _FusedModule  # noqa: F403

# 将下列变量列入模块的导出列表，使其在 from package import * 时可见
__all__ = [
    'ConvBn1d',
    'ConvBn2d',
    'ConvBn3d',
    'ConvBnReLU1d',
    'ConvBnReLU2d',
    'ConvBnReLU3d',
    'ConvReLU1d',
    'ConvReLU2d',
    'ConvReLU3d',
    'LinearReLU',
    'BNReLU2d',
    'BNReLU3d',
    'LinearBn1d',
    'LinearLeakyReLU',
    'LinearTanh',
    'ConvAdd2d',
    'ConvAddReLU2d',
]

# 我们将所有子包暴露给最终用户。
# 由于可能存在互相依赖的情况，我们希望避免循环导入，
# 因此按照 https://peps.python.org/pep-0562/ 实现延迟加载的版本
def __getattr__(name):
    if name in __all__:
        import importlib
        # 动态导入当前包中的子模块，并返回
        return importlib.import_module("." + name, __name__)
    # 抛出错误，指明模块没有指定的属性
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```