# `.\pytorch\torch\utils\backcompat\__init__.py`

```
# 导入必要的模块并允许未类型化的定义（mypy: allow-untyped-defs）
from torch._C import _set_backcompat_broadcast_warn    # 导入设置广播警告的函数
from torch._C import _get_backcompat_broadcast_warn    # 导入获取广播警告状态的函数
from torch._C import _set_backcompat_keepdim_warn      # 导入设置保持维度警告的函数
from torch._C import _get_backcompat_keepdim_warn      # 导入获取保持维度警告状态的函数

# 定义一个Warning类，用于处理警告相关的设置和获取
class Warning:
    def __init__(self, setter, getter):
        self.setter = setter  # 设置警告状态的函数
        self.getter = getter  # 获取警告状态的函数

    def set_enabled(self, value):
        self.setter(value)   # 设置警告状态为给定的值

    def get_enabled(self):
        return self.getter()  # 获取当前警告状态的值

    enabled = property(get_enabled, set_enabled)  # 将获取和设置警告状态的方法绑定到enabled属性上

# 创建广播警告对象
broadcast_warning = Warning(_set_backcompat_broadcast_warn, _get_backcompat_broadcast_warn)
# 创建保持维度警告对象
keepdim_warning = Warning(_set_backcompat_keepdim_warn, _get_backcompat_keepdim_warn)
```