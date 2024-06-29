# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\style.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入以下模块
    DataFrame,       # 数据帧（DataFrame）
    IndexSlice,      # 多级索引切片（IndexSlice）
)


class Render:
    params = [[12, 24, 36], [12, 120]]  # 参数化测试的参数列表，分别为列数和行数
    param_names = ["cols", "rows"]       # 参数化测试的参数名称列表

    def setup(self, cols, rows):
        self.df = DataFrame(  # 创建一个 DataFrame 对象
            np.random.randn(rows, cols),  # 使用随机数填充的指定行列的数据
            columns=[f"float_{i+1}" for i in range(cols)],  # 列名为 float_1, float_2, ...
            index=[f"row_{i+1}" for i in range(rows)],      # 行名为 row_1, row_2, ...
        )

    def time_apply_render(self, cols, rows):
        self._style_apply()      # 调用内部方法，应用样式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def peakmem_apply_render(self, cols, rows):
        self._style_apply()      # 调用内部方法，应用样式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def time_classes_render(self, cols, rows):
        self._style_classes()    # 调用内部方法，应用类样式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def peakmem_classes_render(self, cols, rows):
        self._style_classes()    # 调用内部方法，应用类样式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def time_tooltips_render(self, cols, rows):
        self._style_tooltips()   # 调用内部方法，设置工具提示
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def peakmem_tooltips_render(self, cols, rows):
        self._style_tooltips()   # 调用内部方法，设置工具提示
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def time_format_render(self, cols, rows):
        self._style_format()     # 调用内部方法，应用格式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def peakmem_format_render(self, cols, rows):
        self._style_format()     # 调用内部方法，应用格式
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def time_apply_format_hide_render(self, cols, rows):
        self._style_apply_format_hide()  # 调用内部方法，应用样式并隐藏
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def peakmem_apply_format_hide_render(self, cols, rows):
        self._style_apply_format_hide()  # 调用内部方法，应用样式并隐藏
        self.st._render_html(True, True)  # 渲染 HTML 输出

    def _style_apply(self):
        def _apply_func(s):
            return [
                "background-color: lightcyan" if s.name == "row_1" else "" for v in s
            ]  # 根据条件应用样式，如果行名为 "row_1" 则设置背景颜色为淡青色

        self.st = self.df.style.apply(_apply_func, axis=1)  # 应用样式函数到 DataFrame 的每一行

    def _style_classes(self):
        classes = self.df.map(lambda v: ("cls-1" if v > 0 else ""))  # 根据条件映射为类名
        classes.index, classes.columns = self.df.index, self.df.columns
        self.st = self.df.style.set_td_classes(classes)  # 设置单元格类样式

    def _style_format(self):
        ic = int(len(self.df.columns) / 4 * 3)  # 计算列索引范围
        ir = int(len(self.df.index) / 4 * 3)   # 计算行索引范围
        self.st = self.df.style.format(
            "{:,.3f}",
            subset=IndexSlice["row_1" : f"row_{ir}", "float_1" : f"float_{ic}"],  # 对指定范围应用格式化字符串
        )

    def _style_apply_format_hide(self):
        self.st = self.df.style.map(lambda v: "color: red;")  # 映射样式设置为红色文本
        self.st.format("{:.3f}")  # 格式化数字为三位小数
        self.st.hide(self.st.index[1:], axis=0)  # 隐藏除第一行外的所有行
        self.st.hide(self.st.columns[1:], axis=1)  # 隐藏除第一列外的所有列

    def _style_tooltips(self):
        ttips = DataFrame("abc", index=self.df.index[::2], columns=self.df.columns[::2])  # 创建工具提示内容的 DataFrame
        self.st = self.df.style.set_tooltips(ttips)  # 设置工具提示
        self.st.hide(self.st.index[12:], axis=0)  # 隐藏除前12行外的所有行
        self.st.hide(self.st.columns[12:], axis=1)  # 隐藏除前12列外的所有列
```