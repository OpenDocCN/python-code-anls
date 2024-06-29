# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_fontconfig_pattern.py`

```py
# 导入 pytest 库，用于编写和运行测试
import pytest

# 从 matplotlib.font_manager 模块导入 FontProperties 类
from matplotlib.font_manager import FontProperties

# 定义 FontProperties 对象的属性列表，用于检查一致性
keys = [
    "get_family",
    "get_style",
    "get_variant",
    "get_weight",
    "get_size",
    ]


# 定义测试函数 test_fontconfig_pattern
def test_fontconfig_pattern():
    """Test converting a FontProperties to string then back."""

    # 默认情况下的测试
    test = "defaults "
    f1 = FontProperties()  # 创建默认的 FontProperties 对象
    s = str(f1)  # 将 FontProperties 对象转换为字符串

    f2 = FontProperties(s)  # 使用字符串创建新的 FontProperties 对象
    for k in keys:
        assert getattr(f1, k)() == getattr(f2, k)(), test + k

    # 基本输入的测试
    test = "basic "
    f1 = FontProperties(family="serif", size=20, style="italic")  # 指定属性创建 FontProperties 对象
    s = str(f1)  # 将 FontProperties 对象转换为字符串

    f2 = FontProperties(s)  # 使用字符串创建新的 FontProperties 对象
    for k in keys:
        assert getattr(f1, k)() == getattr(f2, k)(), test + k

    # 完整输入的测试
    test = "full "
    f1 = FontProperties(family="sans-serif", size=24, weight="bold",
                        style="oblique", variant="small-caps",
                        stretch="expanded")  # 指定多个属性创建 FontProperties 对象
    s = str(f1)  # 将 FontProperties 对象转换为字符串

    f2 = FontProperties(s)  # 使用字符串创建新的 FontProperties 对象
    for k in keys:
        assert getattr(f1, k)() == getattr(f2, k)(), test + k


# 定义测试函数 test_fontconfig_str
def test_fontconfig_str():
    """Test FontProperties string conversions for correctness."""

    # 从实际字体配置规格中获取已知的良好字符串，并根据 MPL 的默认值进行修改

    # 通过检查发现的默认值
    test = "defaults "
    s = ("sans\\-serif:style=normal:variant=normal:weight=normal"
         ":stretch=normal:size=12.0")  # 字符串表示的字体属性
    font = FontProperties(s)  # 使用字符串创建 FontProperties 对象
    right = FontProperties()  # 创建默认的 FontProperties 对象
    for k in keys:
        assert getattr(font, k)() == getattr(right, k)(), test + k

    # 完整输入的测试
    test = "full "
    s = ("serif-24:style=oblique:variant=small-caps:weight=bold"
         ":stretch=expanded")  # 字符串表示的字体属性
    font = FontProperties(s)  # 使用字符串创建 FontProperties 对象
    right = FontProperties(family="serif", size=24, weight="bold",
                           style="oblique", variant="small-caps",
                           stretch="expanded")  # 指定属性创建 FontProperties 对象
    for k in keys:
        assert getattr(font, k)() == getattr(right, k)(), test + k


# 定义测试函数 test_fontconfig_unknown_constant
def test_fontconfig_unknown_constant():
    """Test FontProperties initialization with an unknown constant."""

    # 使用 pytest 检查是否会引发 ValueError 异常，并匹配 "ParseException"
    with pytest.raises(ValueError, match="ParseException"):
        FontProperties(":unknown")
```