# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\timeline.py`

```py
"""
===============================================
Creating a timeline with lines, dates, and text
===============================================

How to create a simple timeline using Matplotlib release dates.

Timelines can be created with a collection of dates and text. In this example,
we show how to create a simple timeline using the dates for recent releases
of Matplotlib. First, we'll pull the data from GitHub.
"""

# 导入需要的库和模块
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates

try:
    # 尝试从 https://api.github.com/repos/matplotlib/matplotlib/releases 获取 Matplotlib 发布版本和日期
    import json
    import urllib.request

    url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
    url += '?per_page=100'
    data = json.loads(urllib.request.urlopen(url, timeout=1).read().decode())

    dates = []
    releases = []
    # 遍历获取的数据，提取非候选版本和非测试版本的发布日期和版本号
    for item in data:
        if 'rc' not in item['tag_name'] and 'b' not in item['tag_name']:
            dates.append(item['published_at'].split("T")[0])
            releases.append(item['tag_name'].lstrip("v"))

except Exception:
    # 如果上述获取数据的过程出现问题，例如没有网络连接，使用以下列表作为备选方案
    releases = ['2.2.4', '3.0.3', '3.0.2', '3.0.1', '3.0.0', '2.2.3',
                '2.2.2', '2.2.1', '2.2.0', '2.1.2', '2.1.1', '2.1.0',
                '2.0.2', '2.0.1', '2.0.0', '1.5.3', '1.5.2', '1.5.1',
                '1.5.0', '1.4.3', '1.4.2', '1.4.1', '1.4.0']
    dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
             '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
             '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
             '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
             '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
             '2014-10-26', '2014-10-18', '2014-08-26']

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]  # 将日期字符串转换为日期对象
dates, releases = zip(*sorted(zip(dates, releases)))  # 按日期升序排序版本和日期

# %%
# 接下来，我们将创建一个带有不同级别的柄图，以区分紧邻的事件。我们在基线上添加标记以强调时间线的一维性质。
#
# 对于每个事件，我们通过 `~.Axes.annotate` 添加一个文本标签，该标签偏移一定单位的点数以与事件线的末端对齐。
#
# 注意，Matplotlib 会自动处理日期时间输入。

# 选择一些好看的级别：在顶部和底部交替放置次要版本，对于修复版本逐渐缩短柄长度。
levels = []
major_minor_releases = sorted({release[:3] for release in releases})
for release in releases:
    major_minor = release[:3]
    bugfix = int(release[4])
    h = 1 + 0.8 * (5 - bugfix)
    level = h if major_minor_releases.index(major_minor) % 2 == 0 else -h
    levels.append(level)

# 创建图和坐标轴。
fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
ax.set(title="Matplotlib release dates")

# 绘制垂直的线段。
ax.vlines(dates, 0, levels,
          color=[("tab:red", 1 if release.endswith(".0") else .5)
                 for release in releases])
# 绘制基线。
ax.axhline(0, c="black")
# 在基线上标记点。
minor_dates = [date for date, release in zip(dates, releases) if release[-1] == '0']
bugfix_dates = [date for date, release in zip(dates, releases) if release[-1] != '0']
ax.plot(bugfix_dates, np.zeros_like(bugfix_dates), "ko", mfc="white")
ax.plot(minor_dates, np.zeros_like(minor_dates), "ko", mfc="tab:red")

# 标注线段上的文字信息。
for date, level, release in zip(dates, levels, releases):
    ax.annotate(release, xy=(date, level),
                xytext=(-3, np.sign(level)*3), textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="bold" if release.endswith(".0") else "normal",
                bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

ax.yaxis.set(major_locator=mdates.YearLocator(),
             major_formatter=mdates.DateFormatter("%Y"))

# 移除 y 轴及部分轴线。
ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

ax.margins(y=0.1)
plt.show()
```