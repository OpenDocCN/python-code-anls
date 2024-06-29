# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_datetime.py`

```py
import datetime  # 导入datetime模块，用于处理日期和时间
import numpy as np  # 导入numpy库，用于数值计算

import pytest  # 导入pytest，用于编写和运行测试用例

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于绘图
import matplotlib as mpl  # 导入matplotlib库的主模块，用于配置绘图参数


class TestDatetimePlotting:
    @mpl.style.context("default")
    def test_annotate(self):
        mpl.rcParams["date.converter"] = 'concise'  # 设置matplotlib的日期转换器为'concise'
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, layout="constrained")  # 创建包含4个子图的图形对象

        start_date = datetime.datetime(2023, 10, 1)  # 定义起始日期为2023年10月1日
        dates = [start_date + datetime.timedelta(days=i) for i in range(31)]  # 生成包含31天日期的列表
        data = list(range(1, 32))  # 生成从1到31的整数列表
        test_text = "Test Text"  # 定义测试用文本字符串

        ax1.plot(dates, data)  # 在第一个子图上绘制日期与数据的折线图
        ax1.annotate(text=test_text, xy=(dates[15], data[15]))  # 在第一个子图上添加注释
        ax2.plot(data, dates)  # 在第二个子图上绘制数据与日期的折线图
        ax2.annotate(text=test_text, xy=(data[5], dates[26]))  # 在第二个子图上添加注释
        ax3.plot(dates, dates)  # 在第三个子图上绘制日期与日期的折线图
        ax3.annotate(text=test_text, xy=(dates[15], dates[3]))  # 在第三个子图上添加注释
        ax4.plot(dates, dates)  # 在第四个子图上绘制日期与日期的折线图
        ax4.annotate(text=test_text, xy=(dates[5], dates[30]),  # 在第四个子图上添加带箭头的注释
                        xytext=(dates[1], dates[7]), arrowprops=dict(facecolor='red'))

    @pytest.mark.xfail(reason="Test for arrow not written yet")
    @mpl.style.context("default")
    def test_arrow(self):
        fig, ax = plt.subplots()  # 创建一个新的图形对象和一个子图对象
        ax.arrow(...)  # 绘制箭头，此处省略具体细节

    @mpl.style.context("default")
    def test_axhline(self):
        mpl.rcParams["date.converter"] = 'concise'  # 设置matplotlib的日期转换器为'concise'
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')  # 创建包含3个子图的图形对象
        ax1.set_ylim(bottom=datetime.datetime(2020, 4, 1),  # 设置第一个子图的y轴范围
                     top=datetime.datetime(2020, 8, 1))
        ax2.set_ylim(bottom=np.datetime64('2005-01-01'),  # 设置第二个子图的y轴范围
                     top=np.datetime64('2005-04-01'))
        ax3.set_ylim(bottom=datetime.datetime(2023, 9, 1),  # 设置第三个子图的y轴范围
                     top=datetime.datetime(2023, 11, 1))
        ax1.axhline(y=datetime.datetime(2020, 6, 3), xmin=0.5, xmax=0.7)  # 在第一个子图上绘制水平线
        ax2.axhline(np.datetime64('2005-02-25T03:30'), xmin=0.1, xmax=0.9)  # 在第二个子图上绘制水平线
        ax3.axhline(y=datetime.datetime(2023, 10, 24), xmin=0.4, xmax=0.7)  # 在第三个子图上绘制水平线
    def test_axhspan(self):
        # 设置 Matplotlib 全局参数，用于日期转换
        mpl.rcParams["date.converter"] = 'concise'

        # 创建起始日期对象
        start_date = datetime.datetime(2023, 1, 1)
        # 生成日期列表，从起始日期开始，逐日增加，共计31天
        dates = [start_date + datetime.timedelta(days=i) for i in range(31)]
        # 生成对应的数字列表，从1到31
        numbers = list(range(1, 32))

        # 创建包含三个子图的图形对象，设置子图之间的布局约束和尺寸
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                            constrained_layout=True,
                                            figsize=(10, 12))

        # 在第一个子图上绘制日期与数字的关系图，使用圆点标记和蓝色线条
        ax1.plot(dates, numbers, marker='o', color='blue')
        # 遍历每两天的索引，为每段区域添加水平跨度，使用绿色填充和半透明效果
        for i in range(0, 31, 2):
            ax1.axhspan(ymin=i+1, ymax=i+2, facecolor='green', alpha=0.5)
        ax1.set_title('Datetime vs. Number')  # 设置子图标题
        ax1.set_xlabel('Date')  # 设置X轴标签
        ax1.set_ylabel('Number')  # 设置Y轴标签

        # 在第二个子图上绘制数字与日期的关系图，使用圆点标记和蓝色线条
        ax2.plot(numbers, dates, marker='o', color='blue')
        # 遍历每两天的索引，为每段区域添加水平跨度，使用绿色填充和半透明效果
        for i in range(0, 31, 2):
            ymin = start_date + datetime.timedelta(days=i)
            ymax = ymin + datetime.timedelta(days=1)
            ax2.axhspan(ymin=ymin, ymax=ymax, facecolor='green', alpha=0.5)
        ax2.set_title('Number vs. Datetime')  # 设置子图标题
        ax2.set_xlabel('Number')  # 设置X轴标签
        ax2.set_ylabel('Date')  # 设置Y轴标签

        # 在第三个子图上绘制日期与日期的关系图，使用圆点标记和蓝色线条
        ax3.plot(dates, dates, marker='o', color='blue')
        # 遍历每两天的索引，为每段区域添加水平跨度，使用绿色填充和半透明效果
        for i in range(0, 31, 2):
            ymin = start_date + datetime.timedelta(days=i)
            ymax = ymin + datetime.timedelta(days=1)
            ax3.axhspan(ymin=ymin, ymax=ymax, facecolor='green', alpha=0.5)
        ax3.set_title('Datetime vs. Datetime')  # 设置子图标题
        ax3.set_xlabel('Date')  # 设置X轴标签
        ax3.set_ylabel('Date')  # 设置Y轴标签

    @pytest.mark.xfail(reason="Test for axline not written yet")
    @mpl.style.context("default")
    def test_axline(self):
        # 创建包含单个子图的图形对象
        fig, ax = plt.subplots()
        ax.axline(...)  # 在图中绘制直线，具体细节尚未编写

    @mpl.style.context("default")
    def test_axvline(self):
        # 设置 Matplotlib 全局参数，用于日期转换
        mpl.rcParams["date.converter"] = 'concise'
        # 创建包含三个子图的图形对象，使用约束布局
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
        # 在第一个子图上设置X轴范围，限定在指定日期范围内
        ax1.set_xlim(left=datetime.datetime(2020, 4, 1),
                     right=datetime.datetime(2020, 8, 1))
        # 在第二个子图上设置X轴范围，限定在指定日期范围内
        ax2.set_xlim(left=np.datetime64('2005-01-01'),
                     right=np.datetime64('2005-04-01'))
        # 在第三个子图上设置X轴范围，限定在指定日期范围内
        ax3.set_xlim(left=datetime.datetime(2023, 9, 1),
                     right=datetime.datetime(2023, 11, 1))
        # 在第一个子图上绘制垂直线，标记在指定日期上，指定垂直位置范围
        ax1.axvline(x=datetime.datetime(2020, 6, 3), ymin=0.5, ymax=0.7)
        # 在第二个子图上绘制垂直线，标记在指定日期上，指定垂直位置范围
        ax2.axvline(np.datetime64('2005-02-25T03:30'), ymin=0.1, ymax=0.9)
        # 在第三个子图上绘制垂直线，标记在指定日期上，指定垂直位置范围
        ax3.axvline(x=datetime.datetime(2023, 10, 24), ymin=0.4, ymax=0.7)
    def test_axvspan(self):
        # 设置 Matplotlib 全局参数，用于日期转换器
        mpl.rcParams["date.converter"] = 'concise'

        # 定义起始日期
        start_date = datetime.datetime(2023, 1, 1)
        # 生成日期列表，从起始日期开始，每天增加一天，共31天
        dates = [start_date + datetime.timedelta(days=i) for i in range(31)]
        # 生成与日期对应的数字列表，从1到31
        numbers = list(range(1, 32))

        # 创建一个包含3个子图的图形对象，使用 constrained_layout 参数布局，指定尺寸为 (10, 12)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                            constrained_layout=True,
                                            figsize=(10, 12))

        # 在第一个子图 ax1 上绘制日期与数字的关系图，设置标记为圆点，颜色为蓝色
        ax1.plot(dates, numbers, marker='o', color='blue')
        # 遍历日期范围，每隔2天添加一个带有红色半透明背景的竖直区域
        for i in range(0, 31, 2):
            xmin = start_date + datetime.timedelta(days=i)
            xmax = xmin + datetime.timedelta(days=1)
            ax1.axvspan(xmin=xmin, xmax=xmax, facecolor='red', alpha=0.5)
        # 设置第一个子图的标题、x轴标签和y轴标签
        ax1.set_title('Datetime vs. Number')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number')

        # 在第二个子图 ax2 上绘制数字与日期的关系图，设置标记为圆点，颜色为蓝色
        ax2.plot(numbers, dates, marker='o', color='blue')
        # 遍历数字范围，每隔2个数字添加一个带有红色半透明背景的竖直区域
        for i in range(0, 31, 2):
            ax2.axvspan(xmin=i+1, xmax=i+2, facecolor='red', alpha=0.5)
        # 设置第二个子图的标题、x轴标签和y轴标签
        ax2.set_title('Number vs. Datetime')
        ax2.set_xlabel('Number')
        ax2.set_ylabel('Date')

        # 在第三个子图 ax3 上绘制日期与日期的关系图，设置标记为圆点，颜色为蓝色
        ax3.plot(dates, dates, marker='o', color='blue')
        # 遍历日期范围，每隔2天添加一个带有红色半透明背景的竖直区域
        for i in range(0, 31, 2):
            xmin = start_date + datetime.timedelta(days=i)
            xmax = xmin + datetime.timedelta(days=1)
            ax3.axvspan(xmin=xmin, xmax=xmax, facecolor='red', alpha=0.5)
        # 设置第三个子图的标题、x轴标签和y轴标签
        ax3.set_title('Datetime vs. Datetime')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Date')

    @mpl.style.context("default")
    def test_bar(self):
        # 设置 Matplotlib 全局参数，用于日期转换器
        mpl.rcParams["date.converter"] = "concise"

        # 创建一个包含2个子图的图形对象，使用 layout 参数布局
        fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained")

        # 定义日期数组和对应的数值范围
        x_dates = np.array([
            datetime.datetime(2020, 6, 30),
            datetime.datetime(2020, 7, 22),
            datetime.datetime(2020, 8, 3),
            datetime.datetime(2020, 9, 14),
        ], dtype=np.datetime64)
        x_ranges = [8800, 2600, 8500, 7400]

        # 定义起始日期 x
        x = np.datetime64(datetime.datetime(2020, 6, 1))
        # 在第一个子图 ax1 上绘制柱状图，使用日期作为 x 轴，数值范围作为高度，柱宽为4天
        ax1.bar(x_dates, x_ranges, width=np.timedelta64(4, "D"))
        # 在第二个子图 ax2 上绘制柱状图，使用索引作为 x 轴，日期间隔与起始日期的差作为底部，高度为 x_dates 日期
        ax2.bar(np.arange(4), x_dates - x, bottom=x)

    @mpl.style.context("default")
    def test_bar_label(self):
        # 生成包含日期的示例数据列表
        date_list = [datetime.datetime(2023, 1, 1) +
                     datetime.timedelta(days=i) for i in range(5)]
        # 示例数值列表
        values = [10, 20, 15, 25, 30]

        # 创建一个包含1个子图的图形对象，设置尺寸为 (10, 8)，使用 layout 参数布局
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), layout='constrained')
        # 绘制柱状图，使用日期列表作为 x 轴，values 作为高度
        bars = ax.bar(date_list, values)

        # 使用 bar_label 方法为柱状图添加标签，标签类型为边缘，颜色为黑色
        ax.bar_label(bars, labels=[f'{val}%' for val in values],
                     label_type='edge', color='black')
    def test_barbs(self):
        # 设置绘图参数，指定日期格式转换器为 'concise'
        plt.rcParams["date.converter"] = 'concise'

        # 设置起始日期为 2022 年 2 月 8 日 22 点，创建包含 12 个时间点的日期列表
        start_date = datetime.datetime(2022, 2, 8, 22)
        dates = [start_date + datetime.timedelta(hours=i) for i in range(12)]

        # 生成一个包含 12 个元素的正弦波数列
        numbers = np.sin(np.linspace(0, 2 * np.pi, 12))

        # 创建长度为 12 的数组，每个元素为 10
        u = np.ones(12) * 10
        # 创建长度为 12 的数组，从 0 到 110，步长为 10
        v = np.arange(0, 120, 10)

        # 创建包含两个子图的画布，尺寸为 12x6
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # 在第一个子图中绘制风羽图(barbs)，用日期作为 x 轴，numbers 作为 y 轴
        # u 和 v 分别为风向和风速，长度为 7
        axes[0].barbs(dates, numbers, u, v, length=7)
        # 设置第一个子图标题
        axes[0].set_title('Datetime vs. Numeric Data')
        # 设置第一个子图 x 轴标签
        axes[0].set_xlabel('Datetime')
        # 设置第一个子图 y 轴标签
        axes[0].set_ylabel('Numeric Data')

        # 在第二个子图中绘制风羽图(barbs)，用 numbers 作为 x 轴，dates 作为 y 轴
        # u 和 v 分别为风向和风速，长度为 7
        axes[1].barbs(numbers, dates, u, v, length=7)
        # 设置第二个子图标题
        axes[1].set_title('Numeric vs. Datetime Data')
        # 设置第二个子图 x 轴标签
        axes[1].set_xlabel('Numeric Data')
        # 设置第二个子图 y 轴标签
        axes[1].set_ylabel('Datetime')

    @mpl.style.context("default")
    def test_barh(self):
        # 设置绘图参数，指定日期格式转换器为 'concise'
        mpl.rcParams["date.converter"] = 'concise'
        # 创建包含两个子图的画布，布局方式为 constrained
        fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')

        # 定义包含四个日期的 numpy 数组
        birth_date = np.array([datetime.datetime(2020, 4, 10),
                               datetime.datetime(2020, 5, 30),
                               datetime.datetime(2020, 10, 12),
                               datetime.datetime(2020, 11, 15)])
        # 设置年份起始日期和结束日期
        year_start = datetime.datetime(2020, 1, 1)
        year_end = datetime.datetime(2020, 12, 31)
        # 定义包含四个年龄的列表
        age = [21, 53, 20, 24]

        # 设置第一个子图 x 轴标签
        ax1.set_xlabel('Age')
        # 设置第一个子图 y 轴标签
        ax1.set_ylabel('Birth Date')
        # 绘制水平条形图，用年龄作为宽度，高度为每个出生日期加上 10 天
        ax1.barh(birth_date, width=age, height=datetime.timedelta(days=10))

        # 设置第二个子图 x 轴范围
        ax2.set_xlim(left=year_start, right=year_end)
        # 设置第二个子图 x 轴标签
        ax2.set_xlabel('Birth Date')
        # 设置第二个子图 y 轴标签
        ax2.set_ylabel('Order of Birth Dates')
        # 绘制水平条形图，y 轴为 0 到 3，宽度为每个出生日期与年初的时间差，起始位置为年初
        ax2.barh(np.arange(4), birth_date-year_start, left=year_start)

    @pytest.mark.xfail(reason="Test for boxplot not written yet")
    @mpl.style.context("default")
    def test_boxplot(self):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots()
        # 还未实现箱线图的测试，故此处代码未完成

    @mpl.style.context("default")
    def test_broken_barh(self):
        # 设置绘图参数，指定日期格式转换器为 'concise'
        mpl.rcParams["date.converter"] = 'concise'
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots()

        # 在坐标轴上绘制带有间隔的水平条形图
        ax.broken_barh([(datetime.datetime(2023, 1, 4), datetime.timedelta(days=2)),
                        (datetime.datetime(2023, 1, 8), datetime.timedelta(days=3))],
                        (10, 9), facecolors='tab:blue')
        ax.broken_barh([(datetime.datetime(2023, 1, 2), datetime.timedelta(days=1)),
                        (datetime.datetime(2023, 1, 4), datetime.timedelta(days=4))],
                        (20, 9), facecolors=('tab:red'))
    def test_bxp(self):
        # 设置 Matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = 'concise'
        # 创建一个图形和轴对象
        fig, ax = plt.subplots()
        # 准备包含日期时间数据的字典列表
        data = [{
            "med": datetime.datetime(2020, 1, 15),
            "q1": datetime.datetime(2020, 1, 10),
            "q3": datetime.datetime(2020, 1, 20),
            "whislo": datetime.datetime(2020, 1, 5),
            "whishi": datetime.datetime(2020, 1, 25),
            "fliers": [
                datetime.datetime(2020, 1, 3),
                datetime.datetime(2020, 1, 27)
            ]
        }]
        # 绘制箱线图，设置水平方向绘制
        ax.bxp(data, orientation='horizontal')
        # 设置 x 轴主要刻度的日期格式为 "%Y-%m-%d"
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
        # 设置图形标题
        ax.set_title('Box plot with datetime data')

    @pytest.mark.xfail(reason="Test for clabel not written yet")
    @mpl.style.context("default")
    def test_clabel(self):
        # 创建一个图形和轴对象
        fig, ax = plt.subplots()
        # 在轴上添加标签，此处省略了具体参数

    @mpl.style.context("default")
    def test_contour(self):
        # 设置 Matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = "concise"
        # 定义范围阈值
        range_threshold = 10
        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 生成日期时间数组
        x_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        y_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        # 生成范围数组
        x_ranges = np.array(range(1, range_threshold))
        y_ranges = np.array(range(1, range_threshold))

        # 创建日期时间的网格
        X_dates, Y_dates = np.meshgrid(x_dates, y_dates)
        # 创建范围的网格
        X_ranges, Y_ranges = np.meshgrid(x_ranges, y_ranges)

        # 计算 Z 值，这里使用简单的余弦和正弦函数
        Z_ranges = np.cos(X_ranges / 4) + np.sin(Y_ranges / 4)

        # 在每个子图上绘制等高线图
        ax1.contour(X_dates, Y_dates, Z_ranges)
        ax2.contour(X_dates, Y_ranges, Z_ranges)
        ax3.contour(X_ranges, Y_dates, Z_ranges)

    @mpl.style.context("default")
    def test_contourf(self):
        # 设置 Matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = "concise"
        # 定义范围阈值
        range_threshold = 10
        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 生成日期时间数组
        x_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        y_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        # 生成范围数组
        x_ranges = np.array(range(1, range_threshold))
        y_ranges = np.array(range(1, range_threshold))

        # 创建日期时间的网格
        X_dates, Y_dates = np.meshgrid(x_dates, y_dates)
        # 创建范围的网格
        X_ranges, Y_ranges = np.meshgrid(x_ranges, y_ranges)

        # 计算 Z 值，这里使用简单的余弦和正弦函数
        Z_ranges = np.cos(X_ranges / 4) + np.sin(Y_ranges / 4)

        # 在每个子图上绘制填充等高线图
        ax1.contourf(X_dates, Y_dates, Z_ranges)
        ax2.contourf(X_dates, Y_ranges, Z_ranges)
        ax3.contourf(X_ranges, Y_dates, Z_ranges)
    def test_errorbar(self):
        # 设置 Matplotlib 参数，用于日期转换
        mpl.rcParams["date.converter"] = "concise"
        
        # 创建包含四个子图的 Figure 对象，并约束布局
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, layout="constrained")
        
        # 定义日期范围和起始日期
        limit = 7
        start_date = datetime.datetime(2023, 1, 1)

        # 创建 x 和 y 的日期数组
        x_dates = np.array([datetime.datetime(2023, 10, d) for d in range(1, limit)])
        y_dates = np.array([datetime.datetime(2023, 10, d) for d in range(1, limit)])
        
        # 定义 x 和 y 的日期误差
        x_date_error = datetime.timedelta(days=1)
        y_date_error = datetime.timedelta(days=1)

        # 创建 x 和 y 的数值数组
        x_values = list(range(1, limit))
        y_values = list(range(1, limit))
        
        # 定义 x 和 y 的数值误差
        x_value_error = 0.5
        y_value_error = 0.5

        # 在 ax1 中绘制误差线图，包括 y 方向的误差
        ax1.errorbar(x_dates, y_values,
                     yerr=y_value_error,
                     capsize=10,
                     barsabove=True,
                     label='Data')
        
        # 在 ax2 中绘制误差线图，包括 x 和 y 方向的误差，指定每隔 (1, 2) 个数据点显示误差线
        ax2.errorbar(x_values, y_dates,
                     xerr=x_value_error, yerr=y_date_error,
                     errorevery=(1, 2),
                     fmt='-o', label='Data')
        
        # 在 ax3 中绘制误差线图，包括 x 和 y 方向的误差，显示左边界和底部左边界
        ax3.errorbar(x_dates, y_dates,
                     xerr=x_date_error, yerr=y_date_error,
                     lolims=True, xlolims=True,
                     label='Data')
        
        # 在 ax4 中绘制误差线图，包括 x 和 y 方向的误差，显示上边界和右边界
        ax4.errorbar(x_dates, y_values,
                     xerr=x_date_error, yerr=y_value_error,
                     uplims=True, xuplims=True,
                     label='Data')

    @mpl.style.context("default")
    def test_eventplot(self):
        # 设置 Matplotlib 参数，用于日期转换
        mpl.rcParams["date.converter"] = "concise"

        # 创建包含三个子图的 Figure 对象，并约束布局
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 创建第一个子图的事件图 x_dates1
        x_dates1 = np.array([datetime.datetime(2020, 6, 30),
                             datetime.datetime(2020, 7, 22),
                             datetime.datetime(2020, 8, 3),
                             datetime.datetime(2020, 9, 14),],
                            dtype=np.datetime64,
                            )

        ax1.eventplot(x_dates1)

        # 设置随机种子
        np.random.seed(19680801)

        # 定义起始日期和结束日期，并计算日期范围
        start_date = datetime.datetime(2020, 7, 1)
        end_date = datetime.datetime(2020, 10, 15)
        date_range = end_date - start_date

        # 创建随机生成的日期数组
        dates1 = start_date + np.random.rand(30) * date_range
        dates2 = start_date + np.random.rand(10) * date_range
        dates3 = start_date + np.random.rand(50) * date_range

        # 定义用于事件图的颜色、线偏移和线长度
        colors1 = ['C1', 'C2', 'C3']
        lineoffsets1 = np.array([1, 6, 8])
        linelengths1 = [5, 2, 3]

        # 在 ax2 中绘制事件图，指定不同颜色、线偏移和线长度的事件
        ax2.eventplot([dates1, dates2, dates3],
                      colors=colors1,
                      lineoffsets=lineoffsets1,
                      linelengths=linelengths1)

        # 定义用于第三个子图的线偏移，使用日期数组
        lineoffsets2 = np.array([
            datetime.datetime(2020, 7, 1),
            datetime.datetime(2020, 7, 15),
            datetime.datetime(2020, 8, 1)
        ], dtype=np.datetime64)

        # 在 ax3 中绘制事件图，指定不同颜色和线偏移的事件
        ax3.eventplot([dates1, dates2, dates3],
                      colors=colors1,
                      lineoffsets=lineoffsets2,
                      linelengths=linelengths1)
    # 定义一个测试方法，用于测试填充区域绘图
    def test_fill(self):
        # 设置 matplotlib 的全局参数，指定日期转换器为 concise
        mpl.rcParams["date.converter"] = "concise"
        # 创建包含四个子图的图形对象，布局为 constrained
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, layout="constrained")

        # 设置随机数种子
        np.random.seed(19680801)

        # 初始化 x 轴的基准日期
        x_base_date = datetime.datetime(2023, 1, 1)
        x_dates = [x_base_date]
        # 生成随机的日期序列作为 x 轴数据点
        for _ in range(1, 5):
            x_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            x_dates.append(x_base_date)

        # 初始化 y 轴的基准日期
        y_base_date = datetime.datetime(2023, 1, 1)
        y_dates = [y_base_date]
        # 生成随机的日期序列作为 y 轴数据点
        for _ in range(1, 5):
            y_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            y_dates.append(y_base_date)

        # 生成随机的 x 和 y 值作为填充区域的数据点
        x_values = np.random.rand(5) * 5
        y_values = np.random.rand(5) * 5 - 2

        # 在子图 ax1 到 ax4 上绘制填充区域图形
        ax1.fill(x_dates, y_values)
        ax2.fill(x_values, y_dates)
        ax3.fill(x_values, y_values)
        ax4.fill(x_dates, y_dates)

    # 定义一个测试方法，用于测试填充区域之间的绘图
    @mpl.style.context("default")
    def test_fill_between(self):
        # 设置 matplotlib 的全局参数，指定日期转换器为 concise
        mpl.rcParams["date.converter"] = "concise"
        # 设置随机数种子
        np.random.seed(19680801)

        # 初始化 y 轴的基准日期
        y_base_date = datetime.datetime(2023, 1, 1)
        y_dates1 = [y_base_date]
        # 生成随机的日期序列作为 y 轴数据点
        for i in range(1, 10):
            y_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            y_dates1.append(y_base_date)

        # 继续使用相同的基准日期生成另一个随机的日期序列作为 y 轴数据点
        y_dates2 = [y_base_date]
        for i in range(1, 10):
            y_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            y_dates2.append(y_base_date)

        # 生成随机的 x 值作为填充区域的数据点，并排序
        x_values = np.random.rand(10) * 10
        x_values.sort()

        # 生成随机的 y 值作为填充区域的数据点，并排序
        y_values1 = np.random.rand(10) * 10
        y_values2 = y_values1 + np.random.rand(10) * 10
        y_values1.sort()
        y_values2.sort()

        # 初始化 x 轴的基准日期
        x_base_date = datetime.datetime(2023, 1, 1)
        x_dates = [x_base_date]
        # 生成随机的日期序列作为 x 轴数据点
        for i in range(1, 10):
            x_base_date += datetime.timedelta(days=np.random.randint(1, 10))
            x_dates.append(x_base_date)

        # 创建包含三个子图的图形对象，布局为 constrained
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 在子图 ax1 到 ax3 上绘制填充区域之间的图形
        ax1.fill_between(x_values, y_dates1, y_dates2)
        ax2.fill_between(x_dates, y_values1, y_values2)
        ax3.fill_between(x_dates, y_dates1, y_dates2)
    def test_fill_betweenx(self):
        # 设置 matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = "concise"
        # 设置随机种子，以便生成可重复的随机数
        np.random.seed(19680801)

        # 定义基础日期
        x_base_date = datetime.datetime(2023, 1, 1)
        # 初始化第一个日期列表，并以基础日期开始
        x_dates1 = [x_base_date]
        # 生成随机增量的日期列表
        for i in range(1, 10):
            x_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            x_dates1.append(x_base_date)

        # 初始化第二个日期列表，以同一基础日期开始
        x_dates2 = [x_base_date]
        # 再次生成随机增量的日期列表
        for i in range(1, 10):
            x_base_date += datetime.timedelta(days=np.random.randint(1, 5))
            x_dates2.append(x_base_date)

        # 生成随机的 y 值，并按升序排序
        y_values = np.random.rand(10) * 10
        y_values.sort()

        # 生成随机的 x 值，并按升序排序
        x_values1 = np.random.rand(10) * 10
        x_values2 = x_values1 + np.random.rand(10) * 10
        x_values1.sort()
        x_values2.sort()

        # 定义基础日期
        y_base_date = datetime.datetime(2023, 1, 1)
        # 初始化日期列表，并以基础日期开始
        y_dates = [y_base_date]
        # 生成随机增量的日期列表
        for i in range(1, 10):
            y_base_date += datetime.timedelta(days=np.random.randint(1, 10))
            y_dates.append(y_base_date)

        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout="constrained")

        # 在第一个子图中绘制填充区域（x轴固定），使用给定的 y 值和日期范围
        ax1.fill_betweenx(y_values, x_dates1, x_dates2)
        # 在第二个子图中绘制填充区域（y轴固定），使用给定的 x 值和日期范围
        ax2.fill_betweenx(y_dates, x_values1, x_values2)
        # 在第三个子图中绘制填充区域（y轴固定），使用给定的日期范围和日期范围
        ax3.fill_betweenx(y_dates, x_dates1, x_dates2)

    @pytest.mark.xfail(reason="Test for hexbin not written yet")
    @mpl.style.context("default")
    def test_hexbin(self):
        # 创建图形对象和坐标轴对象
        fig, ax = plt.subplots()
        # 调用 hexbin 函数，绘制六边形图（但未实现具体代码）

    @mpl.style.context("default")
    def test_hist(self):
        # 设置 matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = 'concise'

        # 定义起始日期和时间间隔
        start_date = datetime.datetime(2023, 10, 1)
        time_delta = datetime.timedelta(days=1)

        # 生成随机整数值列表
        values1 = np.random.randint(1, 10, 30)
        values2 = np.random.randint(1, 10, 30)
        values3 = np.random.randint(1, 10, 30)

        # 生成日期范围作为直方图的 bins 边界
        bin_edges = [start_date + i * time_delta for i in range(31)]

        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)

        # 在第一个子图中绘制直方图，使用给定的日期范围和权重值
        ax1.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=10,
            weights=values1
        )
        # 在第二个子图中绘制直方图，使用给定的日期范围和权重值
        ax2.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=10,
            weights=values2
        )
        # 在第三个子图中绘制直方图，使用给定的日期范围和权重值
        ax3.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=10,
            weights=values3
        )

        # 创建包含三个子图的图形对象
        fig, (ax4, ax5, ax6) = plt.subplots(3, 1, constrained_layout=True)

        # 在第一个子图中绘制直方图，使用给定的日期范围和权重值
        ax4.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=bin_edges,
            weights=values1
        )
        # 在第二个子图中绘制直方图，使用给定的日期范围和权重值
        ax5.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=bin_edges,
            weights=values2
        )
        # 在第三个子图中绘制直方图，使用给定的日期范围和权重值
        ax6.hist(
            [start_date + i * time_delta for i in range(30)],
            bins=bin_edges,
            weights=values3
        )
    def test_hist2d(self):
        # 创建一个包含图和轴的子图对象
        fig, ax = plt.subplots()
        # 在轴上绘制二维直方图
        ax.hist2d(...)

    @mpl.style.context("default")
    def test_hlines(self):
        # 设置日期转换器为'concise'
        mpl.rcParams["date.converter"] = 'concise'
        # 创建一个包含 2x4 布局的子图对象
        fig, axs = plt.subplots(2, 4, layout='constrained')
        # 定义一组日期字符串
        dateStrs = ['2023-03-08',
                    '2023-04-09',
                    '2023-05-13',
                    '2023-07-28',
                    '2023-12-24']
        # 生成日期对象列表
        dates = [datetime.datetime(2023, m*2, 10) for m in range(1, 6)]
        # 生成日期起始列表
        date_start = [datetime.datetime(2023, 6, d) for d in range(5, 30, 5)]
        # 生成日期结束列表
        date_end = [datetime.datetime(2023, 7, d) for d in range(5, 30, 5)]
        # 将日期字符串转换为numpy日期对象
        npDates = [np.datetime64(s) for s in dateStrs]
        # 在指定轴上绘制水平线
        axs[0, 0].hlines(y=dates,
                         xmin=[0.1, 0.2, 0.3, 0.4, 0.5],
                         xmax=[0.5, 0.6, 0.7, 0.8, 0.9])
        axs[0, 1].hlines(dates,
                         xmin=datetime.datetime(2020, 5, 10),
                         xmax=datetime.datetime(2020, 5, 31))
        axs[0, 2].hlines(dates,
                         xmin=date_start,
                         xmax=date_end)
        axs[0, 3].hlines(dates,
                         xmin=0.45,
                         xmax=0.65)
        axs[1, 0].hlines(y=npDates,
                         xmin=[0.5, 0.6, 0.7, 0.8, 0.9],
                         xmax=[0.1, 0.2, 0.3, 0.4, 0.5])
        axs[1, 2].hlines(y=npDates,
                         xmin=date_start,
                         xmax=date_end)
        axs[1, 1].hlines(npDates,
                         xmin=datetime.datetime(2020, 5, 10),
                         xmax=datetime.datetime(2020, 5, 31))
        axs[1, 3].hlines(npDates,
                         xmin=0.45,
                         xmax=0.65)

    @mpl.style.context("default")
    def test_imshow(self):
        # 创建一个包含图和轴的子图对象
        fig, ax = plt.subplots()
        # 创建一个对角矩阵
        a = np.diag(range(5))
        # 定义起始和结束日期时间对象
        dt_start = datetime.datetime(2010, 11, 1)
        dt_end = datetime.datetime(2010, 11, 11)
        # 定义图像的范围
        extent = (dt_start, dt_end, dt_start, dt_end)
        # 在轴上显示图像
        ax.imshow(a, extent=extent)
        # 设置x轴标签旋转90度
        ax.tick_params(axis="x", labelrotation=90)

    @pytest.mark.xfail(reason="Test for loglog not written yet")
    @mpl.style.context("default")
    def test_loglog(self):
        # 创建一个包含图和轴的子图对象
        fig, ax = plt.subplots()
        # 在轴上绘制对数-对数图
        ax.loglog(...)

    @mpl.style.context("default")
    def test_matshow(self):
        # 创建一个对角矩阵
        a = np.diag(range(5))
        # 定义起始和结束日期时间对象
        dt_start = datetime.datetime(1980, 4, 15)
        dt_end = datetime.datetime(2020, 11, 11)
        # 定义图像的范围
        extent = (dt_start, dt_end, dt_start, dt_end)
        # 创建一个包含图和轴的子图对象
        fig, ax = plt.subplots()
        # 在轴上显示矩阵图
        ax.matshow(a, extent=extent)
        # 旋转x轴标签至90度
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    @pytest.mark.xfail(reason="Test for pcolor not written yet")
    @mpl.style.context("default")
    def test_pcolor(self):
        # 创建一个包含图和轴的子图对象
        fig, ax = plt.subplots()
        # 在轴上绘制伪彩图
        ax.pcolor(...)

    @pytest.mark.xfail(reason="Test for pcolorfast not written yet")
    @mpl.style.context("default")
    # 定义一个测试函数，用于测试 pcolorfast 方法
    def test_pcolorfast(self):
        # 创建一个包含单个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 调用轴对象的 pcolorfast 方法，用于快速绘制彩色阵列图

    # 标记为预期失败的测试函数，因为尚未编写 pcolormesh 的测试
    @pytest.mark.xfail(reason="Test for pcolormesh not written yet")
    # 应用默认风格的上下文管理器，用于测试 pcolormesh 方法
    @mpl.style.context("default")
    def test_pcolormesh(self):
        # 创建一个包含单个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 调用轴对象的 pcolormesh 方法，用于绘制伪彩色图

    # 应用默认风格的上下文管理器，用于测试 plot 方法
    @mpl.style.context("default")
    def test_plot(self):
        # 配置 Matplotlib，将日期转换器设置为 'concise'
        mpl.rcParams["date.converter"] = 'concise'
        N = 6
        # 创建一个包含三个子图的图形和轴对象数组，布局为 constrained
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
        # 创建日期时间数组 x，包含从 2023 年 9 月 1 日到 9 月 6 日的日期时间对象
        x = np.array([datetime.datetime(2023, 9, n) for n in range(1, N)])
        # 在第一个子图上绘制 x 与 range(1, N) 的图形
        ax1.plot(x, range(1, N))
        # 在第二个子图上绘制 range(1, N) 与 x 的图形
        ax2.plot(range(1, N), x)
        # 在第三个子图上绘制 x 与 x 的图形
        ax3.plot(x, x)

    # 应用默认风格的上下文管理器，用于测试 plot_date 方法
    @mpl.style.context("default")
    def test_plot_date(self):
        # 配置 Matplotlib，将日期转换器设置为 'concise'
        mpl.rcParams["date.converter"] = "concise"
        range_threshold = 10
        # 创建一个包含三个子图的图形和轴对象数组，布局为 constrained
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 创建日期时间数组 x_dates 和 y_dates，分别包含从 2023 年 10 月 1 日到 10 月 9 日的日期时间对象
        x_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        y_dates = np.array(
            [datetime.datetime(2023, 10, delta) for delta in range(1, range_threshold)]
        )
        # 创建整数数组 x_ranges 和 y_ranges，分别包含从 1 到 9 的整数
        x_ranges = np.array(range(1, range_threshold))
        y_ranges = np.array(range(1, range_threshold))

        # 使用 Matplotlib 的警告管理器监控 Matplotlib 的过时警告
        with pytest.warns(mpl.MatplotlibDeprecationWarning):
            # 在第一个子图上绘制日期时间数组 x_dates 与 y_dates 的散点图
            ax1.plot_date(x_dates, y_dates)
            # 在第二个子图上绘制日期时间数组 x_dates 与 y_ranges 的散点图
            ax2.plot_date(x_dates, y_ranges)
            # 在第三个子图上绘制整数数组 x_ranges 与 y_dates 的散点图

    # 标记为预期失败的测试函数，因为尚未编写 quiver 的测试
    @pytest.mark.xfail(reason="Test for quiver not written yet")
    # 应用默认风格的上下文管理器，用于测试 quiver 方法
    @mpl.style.context("default")
    def test_quiver(self):
        # 创建一个包含单个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 调用轴对象的 quiver 方法，用于绘制矢量场图

    # 应用默认风格的上下文管理器，用于测试 scatter 方法
    @mpl.style.context("default")
    def test_scatter(self):
        # 配置 Matplotlib，将日期转换器设置为 'concise'
        mpl.rcParams["date.converter"] = 'concise'
        # 设置基准日期时间为 2005 年 2 月 1 日
        base = datetime.datetime(2005, 2, 1)
        # 创建日期时间数组 dates，包含从基准日期时间开始每隔两小时的日期时间对象，共 10 个
        dates = [base + datetime.timedelta(hours=(2 * i)) for i in range(10)]
        N = len(dates)
        # 设置随机数种子，保证可重现性
        np.random.seed(19680801)
        # 创建随机数数组 y，表示累积标准正态分布的值
        y = np.cumsum(np.random.randn(N))
        # 创建一个包含三个子图的图形和轴对象数组，布局为 constrained，设置图形大小为 6x6
        fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))
        # 在第一个子图上绘制日期时间数组 dates 与 y 的散点图
        axs[0].scatter(dates, y)
        # 调整第一个子图的 x 轴刻度标签旋转为 40 度，并设置水平对齐方式为右对齐
        for label in axs[0].get_xticklabels():
            label.set_rotation(40)
            label.set_horizontalalignment('right')
        # 在第二个子图上绘制 y 与 dates 的散点图
        axs[1].scatter(y, dates)
        # 在第三个子图上绘制日期时间数组 dates 与 dates 的散点图
        axs[2].scatter(dates, dates)
        # 调整第三个子图的 x 轴刻度标签旋转为 40 度，并设置水平对齐方式为右对齐

    # 标记为预期失败的测试函数，因为尚未编写 semilogx 的测试
    @pytest.mark.xfail(reason="Test for semilogx not written yet")
    # 应用默认风格的上下文管理器，用于测试 semilogx 方法
    @mpl.style.context("default")
    def test_semilogx(self):
        # 创建一个包含单个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 调用轴对象的 semilogx 方法，用于绘制 x 轴为对数坐标的图形

    # 标记为预期失败的测试函数，因为尚未编写 semilogy 的测试
    @pytest.mark.xfail(reason="Test for semilogy not written yet")
    # 应用默认风格的上下文管理器，用于测试 semilogy 方法
    @mpl.style.context("default")
    def test_semilogy(self):
        # 创建一个包含单个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 调用轴对象的 semilogy 方法，用于绘制 y 轴为对数坐标的图形

    # 应用默认风格的上下文管理器
    @mpl.style.context("default")
    @mpl.style.context("default")
    def test_stackplot(self):
        # 设置日期转换器为'concise'格式
        mpl.rcParams["date.converter"] = 'concise'
        # 创建一个大小为10的数组，并复制成4行1列的矩阵
        N = 10
        stacked_nums = np.tile(np.arange(1, N), (4, 1))
        # 创建包含10个日期对象的数组，从2020年1月1日开始，每年增加1年
        dates = np.array([datetime.datetime(2020 + i, 1, 1) for i in range(N - 1)])

        # 创建一个包含子图的图形对象
        fig, ax = plt.subplots(layout='constrained')
        # 在子图上绘制堆叠区域图
        ax.stackplot(dates, stacked_nums)

    @mpl.style.context("default")
    def test_stairs(self):
        # 设置日期转换器为'concise'格式
        mpl.rcParams["date.converter"] = 'concise'

        # 设置开始日期和时间增量
        start_date = datetime.datetime(2023, 12, 1)
        time_delta = datetime.timedelta(days=1)
        # 设置基准日期
        baseline_date = datetime.datetime(1980, 1, 1)

        # 创建日期边缘列表
        bin_edges = [start_date + i * time_delta for i in range(31)]
        edge_int = np.arange(31)
        # 生成随机数据
        np.random.seed(123456)
        values1 = np.random.randint(1, 100, 30)
        values2 = [start_date + datetime.timedelta(days=int(i))
                   for i in np.random.randint(1, 10000, 30)]
        values3 = [start_date + datetime.timedelta(days=int(i))
                   for i in np.random.randint(-10000, 10000, 30)]

        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
        # 在第一个子图上绘制阶梯图
        ax1.stairs(values1, edges=bin_edges)
        # 在第二个子图上绘制阶梯图
        ax2.stairs(values2, edges=edge_int, baseline=baseline_date)
        # 在第三个子图上绘制阶梯图
        ax3.stairs(values3, edges=bin_edges, baseline=baseline_date)

    @mpl.style.context("default")
    def test_stem(self):
        # 设置日期转换器为'concise'格式
        mpl.rcParams["date.converter"] = "concise"

        # 创建包含六个子图的图形对象
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, layout="constrained")

        # 设置限制值、上限和下限日期
        limit_value = 10
        above = datetime.datetime(2023, 9, 18)
        below = datetime.datetime(2023, 11, 18)

        # 创建范围数组
        x_ranges = np.arange(1, limit_value)
        y_ranges = np.arange(1, limit_value)

        # 创建日期数组
        x_dates = np.array(
            [datetime.datetime(2023, 10, n) for n in range(1, limit_value)]
        )
        y_dates = np.array(
            [datetime.datetime(2023, 10, n) for n in range(1, limit_value)]
        )

        # 在各个子图上绘制茎图
        ax1.stem(x_dates, y_dates, bottom=above)
        ax2.stem(x_dates, y_ranges, bottom=5)
        ax3.stem(x_ranges, y_dates, bottom=below)

        ax4.stem(x_ranges, y_dates, orientation="horizontal", bottom=above)
        ax5.stem(x_dates, y_ranges, orientation="horizontal", bottom=5)
        ax6.stem(x_ranges, y_dates, orientation="horizontal", bottom=below)

    @mpl.style.context("default")
    def test_step(self):
        # 设置日期转换器为'concise'格式
        mpl.rcParams["date.converter"] = "concise"
        N = 6
        # 创建包含三个子图的图形对象
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
        # 创建日期数组
        x = np.array([datetime.datetime(2023, 9, n) for n in range(1, N)])
        # 在各个子图上绘制步阶图
        ax1.step(x, range(1, N))
        ax2.step(range(1, N), x)
        ax3.step(x, x)

    @pytest.mark.xfail(reason="Test for streamplot not written yet")
    @mpl.style.context("default")
    def test_streamplot(self):
        # 创建图形对象和轴对象
        fig, ax = plt.subplots()
        # 待补充：绘制流场图
        ax.streamplot(...)

    @mpl.style.context("default")
    # 定义测试方法，用于测试文本绘制功能
    def test_text(self):
        # 设定 Matplotlib 参数，指定日期转换器为简洁模式
        mpl.rcParams["date.converter"] = 'concise'
        # 创建包含三个子图的画布，使用约束布局
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")

        # 设定限制值
        limit_value = 10
        # 设定字体属性
        font_properties = {'family': 'serif', 'size': 12, 'weight': 'bold'}
        # 设定测试日期
        test_date = datetime.datetime(2023, 10, 1)

        # 创建 x 轴数据，使用 numpy 数组
        x_data = np.array(range(1, limit_value))
        # 创建 y 轴数据，使用 numpy 数组
        y_data = np.array(range(1, limit_value))

        # 创建 x 轴日期数据，使用 numpy 数组
        x_dates = np.array(
            [datetime.datetime(2023, 10, n) for n in range(1, limit_value)]
        )
        # 创建 y 轴日期数据，使用 numpy 数组
        y_dates = np.array(
            [datetime.datetime(2023, 10, n) for n in range(1, limit_value)]
        )

        # 在 ax1 子图上绘制日期与数据的折线图
        ax1.plot(x_dates, y_data)
        # 在 ax1 子图上插入文本，指定插入位置与文本内容，使用指定的字体属性
        ax1.text(test_date, 5, "Inserted Text", **font_properties)

        # 在 ax2 子图上绘制数据与日期的折线图
        ax2.plot(x_data, y_dates)
        # 在 ax2 子图上插入文本，指定插入位置与文本内容，使用指定的字体属性
        ax2.text(7, test_date, "Inserted Text", **font_properties)

        # 在 ax3 子图上绘制日期与日期的折线图
        ax3.plot(x_dates, y_dates)
        # 在 ax3 子图上插入文本，指定插入位置与文本内容，使用指定的字体属性
        ax3.text(test_date, test_date, "Inserted Text", **font_properties)

    # 标记为预期失败的测试方法，用于 tricontour 绘图功能的测试
    @pytest.mark.xfail(reason="Test for tricontour not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_tricontour(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制三角剖分轮廓
        ax.tricontour(...)

    # 标记为预期失败的测试方法，用于 tricontourf 绘图功能的测试
    @pytest.mark.xfail(reason="Test for tricontourf not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_tricontourf(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制三角剖分填充轮廓
        ax.tricontourf(...)

    # 标记为预期失败的测试方法，用于 tripcolor 绘图功能的测试
    @pytest.mark.xfail(reason="Test for tripcolor not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_tripcolor(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制三角形网格颜色填充
        ax.tripcolor(...)

    # 标记为预期失败的测试方法，用于 triplot 绘图功能的测试
    @pytest.mark.xfail(reason="Test for triplot not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_triplot(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制三角形网格
        ax.triplot(...)

    # 标记为预期失败的测试方法，用于 violin 绘图功能的测试
    @pytest.mark.xfail(reason="Test for violin not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_violin(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制小提琴图
        ax.violin(...)

    # 标记为预期失败的测试方法，用于 violinplot 绘图功能的测试
    @pytest.mark.xfail(reason="Test for violinplot not written yet")
    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    def test_violinplot(self):
        # 创建包含一个子图的画布
        fig, ax = plt.subplots()
        # 绘制小提琴图
        ax.violinplot(...)

    # 设定 Matplotlib 样式上下文为默认样式
    @mpl.style.context("default")
    # 定义一个测试方法 test_vlines，用于测试垂直线绘制功能
    def test_vlines(self):
        # 设置 matplotlib 的参数，指定日期转换器为 'concise'
        mpl.rcParams["date.converter"] = 'concise'
        
        # 创建一个包含三个子图的图形对象，布局为约束布局
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
        
        # 在第一个子图 ax1 上设置 x 轴的显示范围为 2023 年 1 月 1 日到 2023 年 6 月 30 日
        ax1.set_xlim(left=datetime.datetime(2023, 1, 1),
                     right=datetime.datetime(2023, 6, 30))
        
        # 在 ax1 上绘制垂直线，分别位于指定的日期，每条线的 y 范围从 0 到指定值
        ax1.vlines(x=[datetime.datetime(2023, 2, 10),
                      datetime.datetime(2023, 5, 18),
                      datetime.datetime(2023, 6, 6)],
                   ymin=[0, 0.25, 0.5],
                   ymax=[0.25, 0.5, 0.75])
        
        # 在第二个子图 ax2 上设置 x 轴的显示范围为 0 到 0.5
        ax2.set_xlim(left=0,
                     right=0.5)
        
        # 在 ax2 上绘制垂直线，分别位于指定的 x 坐标，每条线的 y 范围从指定日期到指定日期
        ax2.vlines(x=[0.3, 0.35],
                   ymin=[np.datetime64('2023-03-20'), np.datetime64('2023-03-31')],
                   ymax=[np.datetime64('2023-05-01'), np.datetime64('2023-05-16')])
        
        # 在第三个子图 ax3 上设置 x 轴的显示范围为 2023 年 7 月 1 日到 2023 年 12 月 31 日
        ax3.set_xlim(left=datetime.datetime(2023, 7, 1),
                     right=datetime.datetime(2023, 12, 31))
        
        # 在 ax3 上绘制垂直线，分别位于指定的日期，每条线的 y 范围从指定日期到指定日期
        ax3.vlines(x=[datetime.datetime(2023, 9, 1), datetime.datetime(2023, 12, 10)],
                   ymin=datetime.datetime(2023, 1, 15),
                   ymax=datetime.datetime(2023, 1, 30))
```