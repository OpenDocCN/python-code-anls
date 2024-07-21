# `.\pytorch\torch\package\_mangling.py`

```
# mypy: allow-untyped-defs
"""Import mangling.
See mangling.md for details.
"""
# 导入正则表达式模块
import re

# 全局变量，用于生成唯一标识符
_mangle_index = 0


class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self):
        # 引用全局变量 _mangle_index
        global _mangle_index
        # 将全局变量的值赋给实例变量 _mangle_index
        self._mangle_index = _mangle_index
        # 全局变量值加一，用于下一个实例
        _mangle_index += 1
        # 创建一个唯一的父级标识符，用尖括号包围以避免与真实模块混淆
        self._mangle_parent = f"<torch_package_{self._mangle_index}>"

    def mangle(self, name) -> str:
        # 断言名称长度不为零
        assert len(name) != 0
        # 返回经过编码的名称，包括父级标识符和名称本身
        return self._mangle_parent + "." + name

    def demangle(self, mangled: str) -> str:
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
        # 如果编码后的名称以当前实例的父级标识符开头
        if mangled.startswith(self._mangle_parent + "."):
            # 返回去除父级标识符后的原始名称
            return mangled.partition(".")[2]

        # 如果不是编码过的名称，则直接返回原名称
        return mangled

    def parent_name(self):
        # 返回当前实例的父级标识符
        return self._mangle_parent


def is_mangled(name: str) -> bool:
    # 判断名称是否是经过编码的，以特定格式开始和结束
    return bool(re.match(r"<torch_package_\d+>", name))


def demangle(name: str) -> str:
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
    # 如果名称是经过编码的
    if is_mangled(name):
        first, sep, last = name.partition(".")
        # 如果只有基本编码前缀，如 '<torch_package_0>'，则返回空字符串
        return last if len(sep) != 0 else ""
    # 如果不是经过编码的名称，直接返回原名称
    return name


def get_mangle_prefix(name: str) -> str:
    # 如果名称是经过编码的，返回编码前缀
    return name.partition(".")[0] if is_mangled(name) else name
```