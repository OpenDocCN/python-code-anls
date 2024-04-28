# `.\models\deprecated\tapex\__init__.py`

```
# 版权声明和许可协议
# 版权声明和许可协议信息
# 类型检查模块
# 导入惰性模块
# 导入结构，包括了 tokenization_tapex 模块中的 TapexTokenizer 类
_import_structure = {"tokenization_tapex": ["TapexTokenizer"]} # 导入结构，包括了 tokenization_tapex 模块中的 TapexTokenizer 类

# 如果是类型检查
if TYPE_CHECKING:
    from .tokenization_tapex import TapexTokenizer # 从 tokenization_tapex 模块中导入 TapexTokenizer 类
# 如果不是类型检查
else:
    import sys # 导入 sys 模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure) # 将当前模块注册为惰性模块
```