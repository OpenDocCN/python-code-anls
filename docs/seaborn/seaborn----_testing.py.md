# `D:\src\scipysrc\seaborn\seaborn\_testing.py`

```
import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba
from numpy.testing import assert_array_equal

# 定义可以比较的属性列表
USE_PROPS = [
    "alpha",
    "edgecolor",
    "facecolor",
    "fill",
    "hatch",
    "height",
    "linestyle",
    "linewidth",
    "paths",
    "xy",
    "xydata",
    "sizes",
    "zorder",
]

# 比较两个艺术家对象列表是否相等
def assert_artists_equal(list1, list2):

    # 断言两个列表长度相等
    assert len(list1) == len(list2)
    
    # 遍历两个列表中的每一对艺术家对象
    for a1, a2 in zip(list1, list2):
        # 断言两个对象的类型相同
        assert a1.__class__ == a2.__class__
        
        # 获取艺术家对象的属性字典
        prop1 = a1.properties()
        prop2 = a2.properties()
        
        # 遍历需要比较的属性
        for key in USE_PROPS:
            if key not in prop1:
                continue
            v1 = prop1[key]
            v2 = prop2[key]
            
            # 对路径属性进行特殊处理，比较路径的顶点和代码
            if key == "paths":
                for p1, p2 in zip(v1, v2):
                    assert_array_equal(p1.vertices, p2.vertices)
                    assert_array_equal(p1.codes, p2.codes)
            # 对颜色属性进行特殊处理，转换为 RGBA 格式后比较
            elif key == "color":
                v1 = mpl.colors.to_rgba(v1)
                v2 = mpl.colors.to_rgba(v2)
                assert v1 == v2
            # 如果是 NumPy 数组，直接比较数组内容
            elif isinstance(v1, np.ndarray):
                assert_array_equal(v1, v2)
            # 否则，直接比较值
            else:
                assert v1 == v2

# 比较两个图例对象是否相等
def assert_legends_equal(leg1, leg2):

    # 断言两个图例对象的标题文本相等
    assert leg1.get_title().get_text() == leg2.get_title().get_text()
    
    # 遍历两个图例对象中的每一个文本对象，断言文本内容相等
    for t1, t2 in zip(leg1.get_texts(), leg2.get_texts()):
        assert t1.get_text() == t2.get_text()

    # 比较图例中的艺术家对象列表
    assert_artists_equal(
        leg1.get_patches(), leg2.get_patches(),
    )
    assert_artists_equal(
        leg1.get_lines(), leg2.get_lines(),
    )

# 比较两个图表对象是否相等
def assert_plots_equal(ax1, ax2, labels=True):

    # 比较图表中的补丁对象、线条对象和集合对象
    assert_artists_equal(ax1.patches, ax2.patches)
    assert_artists_equal(ax1.lines, ax2.lines)
    assert_artists_equal(ax1.collections, ax2.collections)

    # 如果 labels 参数为 True，比较轴对象的标签文本
    if labels:
        assert ax1.get_xlabel() == ax2.get_xlabel()
        assert ax1.get_ylabel() == ax2.get_ylabel()

# 比较两个颜色对象是否相等
def assert_colors_equal(a, b, check_alpha=True):

    # 处理可能的数组输入
    def handle_array(x):
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = np.unique(x, axis=0).squeeze()
            if x.ndim > 1:
                raise ValueError("Color arrays must be 1 dimensional")
        return x

    # 处理输入颜色对象
    a = handle_array(a)
    b = handle_array(b)

    # 根据 check_alpha 参数选择转换函数
    f = to_rgba if check_alpha else to_rgb
    
    # 断言转换后的颜色对象相等
    assert f(a) == f(b)
```