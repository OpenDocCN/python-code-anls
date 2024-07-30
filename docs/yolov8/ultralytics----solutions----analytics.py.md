# `.\yolov8\ultralytics\solutions\analytics.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入循环迭代工具模块，用于创建迭代器
from itertools import cycle

# 导入OpenCV库，用于图像处理
import cv2
# 导入matplotlib.pyplot模块，用于绘制图表
import matplotlib.pyplot as plt
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入matplotlib的FigureCanvas类，用于绘制图形
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# 导入matplotlib的Figure类，用于创建图形对象
from matplotlib.figure import Figure


class Analytics:
    """一个用于创建和更新各种类型图表（线形图、柱状图、饼图、面积图）的类，用于视觉分析。"""

    def __init__(
        self,
        type,
        writer,
        im0_shape,
        title="ultralytics",
        x_label="x",
        y_label="y",
        bg_color="white",
        fg_color="black",
        line_color="yellow",
        line_width=2,
        points_width=10,
        fontsize=13,
        view_img=False,
        save_img=True,
        max_points=50,
    def update_area(self, frame_number, counts_dict):
        """
        Update the area graph with new data for multiple classes.

        Args:
            frame_number (int): The current frame number.
            counts_dict (dict): Dictionary with class names as keys and counts as values.
        """

        # 初始化 x_data 为空数组
        x_data = np.array([])
        # 初始化 y_data_dict，使用 counts_dict 的键创建对应的空数组作为值
        y_data_dict = {key: np.array([]) for key in counts_dict.keys()}

        # 如果图形已经存在线条
        if self.ax.lines:
            # 获取第一条线的 x 轴数据
            x_data = self.ax.lines[0].get_xdata()
            # 遍历每条线并更新对应类别的 y 轴数据
            for line, key in zip(self.ax.lines, counts_dict.keys()):
                y_data_dict[key] = line.get_ydata()

        # 将当前帧数添加到 x_data 中
        x_data = np.append(x_data, float(frame_number))
        max_length = len(x_data)

        # 遍历每个类别的数据
        for key in counts_dict.keys():
            # 将新的计数值添加到对应类别的 y 数据中
            y_data_dict[key] = np.append(y_data_dict[key], float(counts_dict[key]))
            # 如果某个类别的 y 数据长度小于 max_length，则用常数填充
            if len(y_data_dict[key]) < max_length:
                y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])), "constant")

        # 如果 x_data 的长度超过了 max_points，则移除最旧的点
        if len(x_data) > self.max_points:
            x_data = x_data[1:]
            for key in counts_dict.keys():
                y_data_dict[key] = y_data_dict[key][1:]

        # 清空当前图形
        self.ax.clear()

        # 设置颜色循环使用的颜色列表
        colors = ["#E1FF25", "#0BDBEB", "#FF64DA", "#111F68", "#042AFF"]
        # 创建一个颜色循环迭代器
        color_cycle = cycle(colors)

        # 遍历每个类别及其对应的 y 数据
        for key, y_data in y_data_dict.items():
            # 获取下一个颜色
            color = next(color_cycle)
            # 填充区域图形
            self.ax.fill_between(x_data, y_data, color=color, alpha=0.6)
            # 绘制线条并设置线条属性
            self.ax.plot(
                x_data,
                y_data,
                color=color,
                linewidth=self.line_width,
                marker="o",
                markersize=self.points_width,
                label=f"{key} Data Points",
            )

        # 设置图形标题、x 轴标签和 y 轴标签的属性
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)
        
        # 设置图例的位置、字体大小、背景颜色和边框颜色
        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.fg_color)

        # 设置图例文本的颜色为前景色
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        # 绘制更新后的图形
        self.canvas.draw()
        # 将画布转换为 RGBA 缓冲区数组
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        # 将图像数据写入并显示
        self.write_and_display(im0)
    def update_line(self, frame_number, total_counts):
        """
        Update the line graph with new data.

        Args:
            frame_number (int): The current frame number.
            total_counts (int): The total counts to plot.
        """

        # 获取当前线图的 x 和 y 数据
        x_data = self.line.get_xdata()
        y_data = self.line.get_ydata()

        # 将新的 frame_number 和 total_counts 添加到 x_data 和 y_data 中
        x_data = np.append(x_data, float(frame_number))
        y_data = np.append(y_data, float(total_counts))

        # 更新线图的数据
        self.line.set_data(x_data, y_data)

        # 重新计算坐标轴限制
        self.ax.relim()

        # 自动调整视图范围
        self.ax.autoscale_view()

        # 重新绘制画布
        self.canvas.draw()

        # 将画布转换为 RGBA 缓冲区图像
        im0 = np.array(self.canvas.renderer.buffer_rgba())

        # 将图像写入并显示
        self.write_and_display(im0)

    def update_multiple_lines(self, counts_dict, labels_list, frame_number):
        """
        Update the line graph with multiple classes.

        Args:
            counts_dict (int): Dictionary include each class counts.
            labels_list (int): list include each classes names.
            frame_number (int): The current frame number.
        """
        # 发出警告，多条线的显示不受支持，将正常存储输出！
        warnings.warn("Display is not supported for multiple lines, output will be stored normally!")

        # 遍历所有标签
        for obj in labels_list:
            # 如果标签不在已存在的线图对象中，则创建新的线图对象
            if obj not in self.lines:
                (line,) = self.ax.plot([], [], label=obj, marker="o", markersize=self.points_width)
                self.lines[obj] = line

            # 获取当前标签对应的线图对象的 x 和 y 数据
            x_data = self.lines[obj].get_xdata()
            y_data = self.lines[obj].get_ydata()

            # 如果数据点超过最大点数限制，则删除最早的数据点
            if len(x_data) >= self.max_points:
                x_data = np.delete(x_data, 0)
                y_data = np.delete(y_data, 0)

            # 将新的 frame_number 和对应类别的 counts 添加到 x_data 和 y_data 中
            x_data = np.append(x_data, float(frame_number))
            y_data = np.append(y_data, float(counts_dict.get(obj, 0)))

            # 更新当前标签对应的线图对象的数据
            self.lines[obj].set_data(x_data, y_data)

        # 重新计算坐标轴限制
        self.ax.relim()

        # 自动调整视图范围
        self.ax.autoscale_view()

        # 添加图例
        self.ax.legend()

        # 重新绘制画布
        self.canvas.draw()

        # 将画布转换为 RGBA 缓冲区图像
        im0 = np.array(self.canvas.renderer.buffer_rgba())

        # 多条线的视图暂不支持，将 view_img 设置为 False
        self.view_img = False  # for multiple line view_img not supported yet, coming soon!

        # 将图像写入并显示
        self.write_and_display(im0)

    def write_and_display(self, im0):
        """
        Write and display the line graph
        Args:
            im0 (ndarray): Image for processing
        """
        # 转换图像格式从 RGBA 到 BGR
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)

        # 如果 view_img 为 True，则显示图像
        cv2.imshow(self.title, im0) if self.view_img else None

        # 如果 save_img 为 True，则写入图像
        self.writer.write(im0) if self.save_img else None
    def update_bar(self, count_dict):
        """
        Update the bar graph with new data.

        Args:
            count_dict (dict): Dictionary containing the count data to plot.
        """

        # 清空当前图形并设置背景颜色
        self.ax.clear()
        self.ax.set_facecolor(self.bg_color)
        
        # 获取标签和计数数据
        labels = list(count_dict.keys())
        counts = list(count_dict.values())

        # 将标签映射到颜色
        for label in labels:
            if label not in self.color_mapping:
                self.color_mapping[label] = next(self.color_cycle)

        colors = [self.color_mapping[label] for label in labels]

        # 使用颜色绘制柱状图
        bars = self.ax.bar(labels, counts, color=colors)
        
        # 在柱状图上方显示数值
        for bar, count in zip(bars, counts):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                color=self.fg_color,
            )

        # 显示和保存更新后的图形
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        im0 = np.asarray(buf)
        self.write_and_display(im0)

    def update_pie(self, classes_dict):
        """
        Update the pie chart with new data.

        Args:
            classes_dict (dict): Dictionary containing the class data to plot.
        """

        # 更新饼图数据
        labels = list(classes_dict.keys())
        sizes = list(classes_dict.values())
        total = sum(sizes)
        percentages = [size / total * 100 for size in sizes]
        start_angle = 90
        
        # 清空当前图形
        self.ax.clear()

        # 创建饼图，并设置起始角度及文本颜色
        wedges, autotexts = self.ax.pie(sizes, autopct=None, startangle=start_angle, textprops={"color": self.fg_color})

        # 构建带百分比的图例标签
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # 调整布局以适应图例
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.75)

        # 显示和保存更新后的饼图
        im0 = self.fig.canvas.draw()
        im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.write_and_display(im0)
# 如果脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 创建一个 Analytics 对象，设置参数为 "line"，writer 为 None，im0_shape 为 None
    Analytics("line", writer=None, im0_shape=None)
```