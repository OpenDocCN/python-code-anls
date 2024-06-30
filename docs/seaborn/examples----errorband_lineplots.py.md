# `D:\src\scipysrc\seaborn\examples\errorband_lineplots.py`

```
"""
Timeseries plot with error bands
================================

_thumb: .48, .45

"""
# 导入 seaborn 库并设置风格为 'darkgrid'
import seaborn as sns
sns.set_theme(style="darkgrid")

# 使用 seaborn 提供的示例数据集 'fmri'，加载长格式数据
fmri = sns.load_dataset("fmri")

# 绘制不同事件和区域的时间序列图
sns.lineplot(x="timepoint", y="signal",  # 横轴为时间点，纵轴为信号强度
             hue="region", style="event",  # 根据区域和事件样式区分曲线
             data=fmri)  # 使用加载的 fmri 数据集进行绘图
```