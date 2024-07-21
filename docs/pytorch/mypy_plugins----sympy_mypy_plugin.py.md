# `.\pytorch\mypy_plugins\sympy_mypy_plugin.py`

```py
# 导入所需模块和类
from mypy.plugin import Plugin
from mypy.plugins.common import add_attribute_to_class
from mypy.types import NoneType, UnionType


# 定义自定义的插件类 SympyPlugin，继承自 Plugin
class SympyPlugin(Plugin):
    
    # 重写 get_base_class_hook 方法，用于处理特定基类的钩子
    def get_base_class_hook(self, fullname: str):
        # 如果基类名称为 "sympy.core.basic.Basic"
        if fullname == "sympy.core.basic.Basic":
            # 返回 add_assumptions 函数作为钩子处理函数
            return add_assumptions
        # 如果基类名称不匹配，则返回 None，表示不处理
        return None


# 定义函数 add_assumptions，用于为特定类添加属性
def add_assumptions(ctx) -> None:
    # 预设的假设列表，描述了数学对象可能具有的特性
    assumptions = [
        "hermitian", "prime", "noninteger", "negative", "antihermitian",
        "infinite", "finite", "irrational", "extended_positive", "nonpositive",
        "odd", "algebraic", "integer", "rational", "extended_real", "nonnegative",
        "transcendental", "extended_nonzero", "extended_negative", "composite",
        "complex", "imaginary", "nonzero", "zero", "even", "positive", "polar",
        "extended_nonpositive", "extended_nonnegative", "real", "commutative",
    ]
    
    # 遍历假设列表，为上下文中的类添加对应属性
    for a in assumptions:
        add_attribute_to_class(
            ctx.api,  # 使用的 Mypy API 对象
            ctx.cls,  # 当前类的上下文
            f"is_{a}",  # 添加的属性名称，格式为 is_{假设名}
            UnionType([ctx.api.named_type("builtins.bool"), NoneType()]),  # 属性类型为布尔值或 NoneType
        )


# 插件函数，返回 SympyPlugin 类作为 Mypy 插件
def plugin(version: str):
    return SympyPlugin
```