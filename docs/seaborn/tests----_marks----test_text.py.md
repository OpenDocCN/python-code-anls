# `D:\src\scipysrc\seaborn\tests\_marks\test_text.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from matplotlib.colors import to_rgba  # 导入 matplotlib 中的颜色转换函数 to_rgba
from matplotlib.text import Text as MPLText  # 导入 matplotlib 中的文本对象 Text，并重命名为 MPLText

from numpy.testing import assert_array_almost_equal  # 导入 NumPy 测试模块中的数组几乎相等断言函数

from seaborn._core.plot import Plot  # 导入 seaborn 中的绘图对象 Plot
from seaborn._marks.text import Text  # 导入 seaborn 中的文本对象 Text


class TestText:

    def get_texts(self, ax):
        if ax.texts:
            return list(ax.texts)  # 如果轴上存在文本对象，则返回文本对象列表
        else:
            # 兼容 matplotlib < 3.5 的情况下，返回所有艺术家中的文本对象
            return [a for a in ax.artists if isinstance(a, MPLText)]

    def test_simple(self):

        x = y = [1, 2, 3]  # 定义 x 和 y 的值都为 [1, 2, 3]
        s = list("abc")  # 将字符串 "abc" 转换为字符列表

        p = Plot(x, y, text=s).add(Text()).plot()  # 创建 Plot 对象并添加 Text 对象，绘制图形
        ax = p._figure.axes[0]  # 获取图形的第一个轴
        for i, text in enumerate(self.get_texts(ax)):  # 遍历轴上的文本对象
            x_, y_ = text.get_position()  # 获取文本对象的位置坐标
            assert x_ == x[i]  # 断言文本对象的横坐标与预期值相等
            assert y_ == y[i]  # 断言文本对象的纵坐标与预期值相等
            assert text.get_text() == s[i]  # 断言文本对象的文本内容与预期值相等
            assert text.get_horizontalalignment() == "center"  # 断言文本对象的水平对齐方式为 "center"
            assert text.get_verticalalignment() == "center_baseline"  # 断言文本对象的垂直对齐方式为 "center_baseline"

    def test_set_properties(self):

        x = y = [1, 2, 3]  # 定义 x 和 y 的值都为 [1, 2, 3]
        s = list("abc")  # 将字符串 "abc" 转换为字符列表
        color = "red"  # 定义文本颜色为红色
        alpha = .6  # 定义文本透明度为 0.6
        fontsize = 6  # 定义文本字号为 6
        valign = "bottom"  # 定义文本的垂直对齐方式为底部对齐

        m = Text(color=color, alpha=alpha, fontsize=fontsize, valign=valign)  # 创建 Text 对象 m，并设置属性
        p = Plot(x, y, text=s).add(m).plot()  # 创建 Plot 对象并添加 Text 对象 m，绘制图形
        ax = p._figure.axes[0]  # 获取图形的第一个轴
        for i, text in enumerate(self.get_texts(ax)):  # 遍历轴上的文本对象
            assert text.get_text() == s[i]  # 断言文本对象的文本内容与预期值相等
            assert text.get_color() == to_rgba(m.color, m.alpha)  # 断言文本对象的颜色与预期值经过 RGBA 转换后相等
            assert text.get_fontsize() == m.fontsize  # 断言文本对象的字号与预期值相等
            assert text.get_verticalalignment() == m.valign  # 断言文本对象的垂直对齐方式与预期值相等

    def test_mapped_properties(self):

        x = y = [1, 2, 3]  # 定义 x 和 y 的值都为 [1, 2, 3]
        s = list("abc")  # 将字符串 "abc" 转换为字符列表
        color = list("aab")  # 定义文本颜色列表
        fontsize = [1, 2, 4]  # 定义文本字号列表

        p = Plot(x, y, color=color, fontsize=fontsize, text=s).add(Text()).plot()  # 创建 Plot 对象并添加 Text 对象，绘制图形
        ax = p._figure.axes[0]  # 获取图形的第一个轴
        texts = self.get_texts(ax)  # 获取轴上的所有文本对象
        assert texts[0].get_color() == texts[1].get_color()  # 断言第一个和第二个文本对象的颜色相等
        assert texts[0].get_color() != texts[2].get_color()  # 断言第一个和第三个文本对象的颜色不相等
        assert texts[0].get_fontsize() < texts[1].get_fontsize() < texts[2].get_fontsize()  # 断言文本对象的字号大小顺序正确

    def test_mapped_alignment(self):

        x = [1, 2]  # 定义 x 的值为 [1, 2]
        p = Plot(x=x, y=x, halign=x, valign=x, text=x).add(Text()).plot()  # 创建 Plot 对象并添加 Text 对象，绘制图形
        ax = p._figure.axes[0]  # 获取图形的第一个轴
        t1, t2 = self.get_texts(ax)  # 获取轴上的前两个文本对象
        assert t1.get_horizontalalignment() == "left"  # 断言第一个文本对象的水平对齐方式为 "left"
        assert t2.get_horizontalalignment() == "right"  # 断言第二个文本对象的水平对齐方式为 "right"
        assert t1.get_verticalalignment() == "top"  # 断言第一个文本对象的垂直对齐方式为 "top"
        assert t2.get_verticalalignment() == "bottom"  # 断言第二个文本对象的垂直对齐方式为 "bottom"

    def test_identity_fontsize(self):

        x = y = [1, 2, 3]  # 定义 x 和 y 的值都为 [1, 2, 3]
        s = list("abc")  # 将字符串 "abc" 转换为字符列表
        fs = [5, 8, 12]  # 定义文本字号列表
        p = Plot(x, y, text=s, fontsize=fs).add(Text()).scale(fontsize=None).plot()  # 创建 Plot 对象并添加 Text 对象，绘制图形
        ax = p._figure.axes[0]  # 获取图形的第一个轴
        for i, text in enumerate(self.get_texts(ax)):  # 遍历轴上的文本对象
            assert text.get_fontsize() == fs[i]  # 断言文本对象的字号与预期值相等
    # 测试偏移量居中的情况
    def test_offset_centered(self):
        # 设置 x 和 y 都为 [1, 2, 3]
        x = y = [1, 2, 3]
        # 将字符串 "abc" 转换为列表
        s = list("abc")
        # 创建 Plot 对象，并添加文本 s，绘制图形
        p = Plot(x, y, text=s).add(Text()).plot()
        # 获取第一个坐标轴对象
        ax = p._figure.axes[0]
        # 获取坐标变换矩阵
        ax_trans = ax.transData.get_matrix()
        # 对于坐标轴中的每个文本对象
        for text in self.get_texts(ax):
            # 断言文本对象的变换矩阵与坐标轴的变换矩阵近似相等
            assert_array_almost_equal(text.get_transform().get_matrix(), ax_trans)

    # 测试偏移量垂直对齐的情况
    def test_offset_valign(self):
        # 设置 x 和 y 都为 [1, 2, 3]
        x = y = [1, 2, 3]
        # 将字符串 "abc" 转换为列表
        s = list("abc")
        # 创建垂直对齐在底部、字体大小为 5、偏移量为 0.1 的文本对象
        m = Text(valign="bottom", fontsize=5, offset=.1)
        # 创建 Plot 对象，并添加文本 s 和 m，绘制图形
        p = Plot(x, y, text=s).add(m).plot()
        # 获取第一个坐标轴对象
        ax = p._figure.axes[0]
        # 创建预期的偏移矩阵，维度为 (3, 3)，初始化为零矩阵
        expected_shift_matrix = np.zeros((3, 3))
        # 根据偏移量计算预期的偏移矩阵值
        expected_shift_matrix[1, -1] = m.offset * ax.figure.dpi / 72
        # 获取坐标变换矩阵
        ax_trans = ax.transData.get_matrix()
        # 对于坐标轴中的每个文本对象
        for text in self.get_texts(ax):
            # 计算文本对象的变换矩阵与坐标轴的变换矩阵的差值矩阵
            shift_matrix = text.get_transform().get_matrix() - ax_trans
            # 断言差值矩阵与预期的偏移矩阵近似相等
            assert_array_almost_equal(shift_matrix, expected_shift_matrix)

    # 测试偏移量水平对齐的情况
    def test_offset_halign(self):
        # 设置 x 和 y 都为 [1, 2, 3]
        x = y = [1, 2, 3]
        # 将字符串 "abc" 转换为列表
        s = list("abc")
        # 创建水平对齐在右侧、字体大小为 10、偏移量为 0.5 的文本对象
        m = Text(halign="right", fontsize=10, offset=.5)
        # 创建 Plot 对象，并添加文本 s 和 m，绘制图形
        p = Plot(x, y, text=s).add(m).plot()
        # 获取第一个坐标轴对象
        ax = p._figure.axes[0]
        # 创建预期的偏移矩阵，维度为 (3, 3)，初始化为零矩阵
        expected_shift_matrix = np.zeros((3, 3))
        # 根据偏移量计算预期的偏移矩阵值
        expected_shift_matrix[0, -1] = -m.offset * ax.figure.dpi / 72
        # 获取坐标变换矩阵
        ax_trans = ax.transData.get_matrix()
        # 对于坐标轴中的每个文本对象
        for text in self.get_texts(ax):
            # 计算文本对象的变换矩阵与坐标轴的变换矩阵的差值矩阵
            shift_matrix = text.get_transform().get_matrix() - ax_trans
            # 断言差值矩阵与预期的偏移矩阵近似相等
            assert_array_almost_equal(shift_matrix, expected_shift_matrix)
```