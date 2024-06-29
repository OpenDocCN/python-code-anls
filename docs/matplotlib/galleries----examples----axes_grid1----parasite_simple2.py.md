# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\parasite_simple2.py`

```
"""
================
Parasite Simple2
================

"""
# 导入matplotlib的pyplot模块，并用plt作为别名
import matplotlib.pyplot as plt

# 导入matplotlib的transforms模块中的mtransforms
import matplotlib.transforms as mtransforms
# 从mpl_toolkits.axes_grid1.parasite_axes模块中导入HostAxes类
from mpl_toolkits.axes_grid1.parasite_axes import HostAxes

# 观测数据的列表，每项包括星号编号、距离、距离误差、速度、速度误差
obs = [["01_S1", 3.88, 0.14, 1970, 63],
       ["01_S4", 5.6, 0.82, 1622, 150],
       ["02_S1", 2.4, 0.54, 1570, 40],
       ["03_S1", 4.1, 0.62, 2380, 170]]


# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中添加一个新的子图ax_kms，使用HostAxes作为坐标系类型，设置宽高比为1
ax_kms = fig.add_subplot(axes_class=HostAxes, aspect=1)

# 根据给定的公式计算角 proper motion("/yr) 转换为距离为2.3kpc时的线速度(km/s)
pm_to_kms = 1./206265.*2300*3.085e18/3.15e7/1.e5

# 创建一个辅助转换对象aux_trans，通过仿射变换在x方向上缩放pm_to_kms倍，并应用于ax_kms上
aux_trans = mtransforms.Affine2D().scale(pm_to_kms, 1.)
ax_pm = ax_kms.twin(aux_trans)

# 遍历观测数据obs中的每一项，计算速度v和误差ve，绘制在ax_kms上的误差条形图
for n, ds, dse, w, we in obs:
    time = ((2007 + (10. + 4/30.)/12) - 1988.5)
    v = ds / time * pm_to_kms
    ve = dse / time * pm_to_kms
    ax_kms.errorbar([v], [w], xerr=[ve], yerr=[we], color="k")

# 设置ax_kms和ax_pm的坐标轴标签
ax_kms.axis["bottom"].set_label("Linear velocity at 2.3 kpc [km/s]")
ax_kms.axis["left"].set_label("FWHM [km/s]")
ax_pm.axis["top"].set_label(r"Proper Motion [$''$/yr]")
# 显示ax_pm的顶部坐标轴标签
ax_pm.axis["top"].label.set_visible(True)
# 隐藏ax_pm的右侧主刻度标签
ax_pm.axis["right"].major_ticklabels.set_visible(False)

# 设置ax_kms的x和y轴限制
ax_kms.set_xlim(950, 3700)
ax_kms.set_ylim(950, 3100)
# ax_pm的x和y轴限制将自动调整

# 显示绘图结果
plt.show()
```