# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\mri_with_eeg.py`

```py
"""
============
MRI with EEG
============

Displays a set of subplots with an MRI image, its intensity
histogram and some EEG traces.

.. redirect-from:: /gallery/specialty_plots/mri_demo
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy库，用于数值计算

import matplotlib.cbook as cbook  # 导入matplotlib的cbook模块，用于一些工具函数

# 使用plt.subplot_mosaic创建具有特定布局的子图
fig, axd = plt.subplot_mosaic(
    [["image", "density"],
     ["EEG", "EEG"]],
    layout="constrained",
    # "image"子图将包含一个方形图像。通过调整width_ratios使得没有多余的水平或垂直边距。
    width_ratios=[1.05, 2],
)

# 加载MRI数据（256x256的16位整数）
with cbook.get_sample_data('s1045.ima.gz') as dfile:
    im = np.frombuffer(dfile.read(), np.uint16).reshape((256, 256))

# 在"image"子图中绘制MRI图像
axd["image"].imshow(im, cmap="gray")  # 使用灰度色彩映射绘制MRI图像
axd["image"].axis('off')  # 关闭图像的坐标轴显示

# 绘制MRI强度的直方图
im = im[im.nonzero()]  # 忽略背景部分的数据
axd["density"].hist(im, bins=np.arange(0, 2**16+1, 512))  # 绘制直方图，512个区间
axd["density"].set(xlabel='Intensity (a.u.)', xlim=(0, 2**16),  # 设置x轴标签、限制和y轴标签
                   ylabel='MRI density', yticks=[])
axd["density"].minorticks_on()  # 打开次要刻度线

# 加载EEG数据
n_samples, n_rows = 800, 4  # 定义样本数和行数
with cbook.get_sample_data('eeg.dat') as eegfile:
    data = np.fromfile(eegfile, dtype=float).reshape((n_samples, n_rows))
t = 10 * np.arange(n_samples) / n_samples  # 时间向量，从0到10秒

# 绘制EEG图
axd["EEG"].set_xlabel('Time (s)')  # 设置x轴标签为时间（秒）
axd["EEG"].set_xlim(0, 10)  # 设置x轴范围
dy = (data.min() - data.max()) * 0.7  # 计算y轴范围，稍微拥挤一点
axd["EEG"].set_ylim(-dy, n_rows * dy)  # 设置y轴范围
axd["EEG"].set_yticks([0, dy, 2*dy, 3*dy], labels=['PG3', 'PG5', 'PG7', 'PG9'])  # 设置y轴刻度和标签

# 遍历数据列，绘制EEG图
for i, data_col in enumerate(data.T):
    axd["EEG"].plot(t, data_col + i*dy, color="C0")  # 使用颜色C0绘制每列的数据

plt.show()  # 显示绘制的图形
```