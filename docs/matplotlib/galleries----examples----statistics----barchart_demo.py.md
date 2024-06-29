# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\barchart_demo.py`

```py
"""
===================================
Percentiles as horizontal bar chart
===================================

Bar charts are useful for visualizing counts, or summary statistics
with error bars. Also see the :doc:`/gallery/lines_bars_and_markers/barchart`
or the :doc:`/gallery/lines_bars_and_markers/barh` example for simpler versions
of those features.

This example comes from an application in which grade school gym
teachers wanted to be able to show parents how their child did across
a handful of fitness tests, and importantly, relative to how other
children did. To extract the plotting code for demo purposes, we'll
just make up some data for little Johnny Doe.
"""

from collections import namedtuple  # 导入 namedtuple 模块
import matplotlib.pyplot as plt    # 导入 matplotlib.pyplot 模块
import numpy as np                  # 导入 numpy 模块

Student = namedtuple('Student', ['name', 'grade', 'gender'])  # 定义名为 Student 的命名元组，包含字段 'name', 'grade', 'gender'
Score = namedtuple('Score', ['value', 'unit', 'percentile'])  # 定义名为 Score 的命名元组，包含字段 'value', 'unit', 'percentile'


def to_ordinal(num):
    """Convert an integer to an ordinal string, e.g. 2 -> '2nd'."""
    suffixes = {str(i): v
                for i, v in enumerate(['th', 'st', 'nd', 'rd', 'th',
                                       'th', 'th', 'th', 'th', 'th'])}  # 构建后缀字典，用于表示序数后缀
    v = str(num)
    # special case early teens
    if v in {'11', '12', '13'}:  # 对于 '11', '12', '13' 这几个特殊情况，直接返回其后缀为 'th'
        return v + 'th'
    return v + suffixes[v[-1]]   # 其他情况根据个位数数字选择对应后缀


def format_score(score):
    """
    Create score labels for the right y-axis as the test name followed by the
    measurement unit (if any), split over two lines.
    """
    return f'{score.value}\n{score.unit}' if score.unit else str(score.value)  # 格式化分数标签，如果有单位则显示在第二行


def plot_student_results(student, scores_by_test, cohort_size):
    fig, ax1 = plt.subplots(figsize=(9, 7), layout='constrained')  # 创建一个 9x7 英寸大小的子图，约束布局为 'constrained'
    fig.canvas.manager.set_window_title('Eldorado K-8 Fitness Chart')  # 设置图窗口标题为 'Eldorado K-8 Fitness Chart'

    ax1.set_title(student.name)  # 设置子图标题为学生姓名
    ax1.set_xlabel(
        'Percentile Ranking Across {grade} Grade {gender}s\n'
        'Cohort Size: {cohort_size}'.format(
            grade=to_ordinal(student.grade),  # 设置 x 轴标签，显示学生所在年级和性别，以及班级规模
            gender=student.gender.title(),
            cohort_size=cohort_size))

    test_names = list(scores_by_test.keys())      # 获取所有测试项目名称
    percentiles = [score.percentile for score in scores_by_test.values()]  # 获取所有测试项目的百分位数

    rects = ax1.barh(test_names, percentiles, align='center', height=0.5)  # 创建水平条形图，显示测试项目名称和对应的百分位数
    # Partition the percentile values to be able to draw large numbers in
    # white within the bar, and small numbers in black outside the bar.
    large_percentiles = [to_ordinal(p) if p > 40 else '' for p in percentiles]  # 将大于 40 的百分位数转换为序数字符串，用于在条形图内部显示白色文本
    small_percentiles = [to_ordinal(p) if p <= 40 else '' for p in percentiles]  # 将小于等于 40 的百分位数转换为序数字符串，用于在条形图外部显示黑色文本
    ax1.bar_label(rects, small_percentiles,
                  padding=5, color='black', fontweight='bold')  # 在条形图外部显示小百分位数文本，黑色加粗字体
    ax1.bar_label(rects, large_percentiles,
                  padding=-32, color='white', fontweight='bold')  # 在条形图内部显示大百分位数文本，白色加粗字体

    ax1.set_xlim([0, 100])  # 设置 x 轴范围为 0 到 100
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # 设置 x 轴刻度
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)  # 设置 x 轴网格线样式
    ax1.axvline(50, color='grey', alpha=0.25)  # 绘制一条垂直于 x 轴的灰色虚线，表示中位数位置
    # 创建一个新的坐标轴ax2，作为ax1的双Y轴
    ax2 = ax1.twinx()
    # 设置ax2的y轴限制与ax1相同，确保刻度线对齐
    ax2.set_ylim(ax1.get_ylim())
    # 设置ax2的刻度位置和标签，使用scores_by_test中的值生成标签
    ax2.set_yticks(
        np.arange(len(scores_by_test)),
        labels=[format_score(score) for score in scores_by_test.values()])
    # 设置ax2的y轴标签文本
    ax2.set_ylabel('Test Scores')
student = Student(name='Johnny Doe', grade=2, gender='Boy')
# 创建一个名为 student 的学生对象，包括姓名、年级和性别信息

scores_by_test = {
    'Pacer Test': Score(7, 'laps', percentile=37),
    # 为"Pacer Test"测试创建一个 Score 对象，包括分数、单位和百分位数信息
    'Flexed Arm\n Hang': Score(48, 'sec', percentile=95),
    # 为"Flexed Arm\n Hang"测试创建一个 Score 对象，包括分数、单位和百分位数信息
    'Mile Run': Score('12:52', 'min:sec', percentile=73),
    # 为"Mile Run"测试创建一个 Score 对象，包括分数、单位和百分位数信息
    'Agility': Score(17, 'sec', percentile=60),
    # 为"Agility"测试创建一个 Score 对象，包括分数、单位和百分位数信息
    'Push Ups': Score(14, '', percentile=16),
    # 为"Push Ups"测试创建一个 Score 对象，包括分数、单位和百分位数信息
}

plot_student_results(student, scores_by_test, cohort_size=62)
# 调用 plot_student_results 函数绘制学生的测试成绩图表，包括给定的学生对象和测试分数数据，以及班级规模信息
plt.show()
# 显示绘制出的图表



# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
#    - `matplotlib.axes.Axes.twinx` / `matplotlib.pyplot.twinx`
```