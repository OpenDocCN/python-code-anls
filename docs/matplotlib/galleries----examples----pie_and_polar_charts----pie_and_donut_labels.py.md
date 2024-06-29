# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\pie_and_donut_labels.py`

```
# %%
# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# 创建一个图形和子图，设置纵横比为“相等”
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

# 定义一个食谱列表
recipe = ["375 g flour",
          "75 g sugar",
          "250 g butter",
          "300 g berries"]

# 提取数据（以克为单位）和成分名称
data = [float(x.split()[0]) for x in recipe]  # 从食谱中提取每个成分的重量数据
ingredients = [x.split()[-1] for x in recipe]  # 提取每个成分的名称

# 定义一个函数，用于在饼图上显示百分比和绝对值
def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d} g)"

# 创建饼图，并获取返回的对象
wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

# 添加图例到饼图，设置标题和位置
ax.legend(wedges, ingredients,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# 设置自动标签的大小和字体加粗
plt.setp(autotexts, size=8, weight="bold")

# 设置图形标题
ax.set_title("Matplotlib bakery: A pie")

# 展示图形
plt.show()

# %%
# 现在是画甜甜圈的时候了。从甜甜圈的配方开始，将数据转换为数字（将每个鸡蛋转换为50克），然后直接绘制饼图。
# 嘿，等等，这不是甜甜圈吗？嗯，正如我们在这里看到的，甜甜圈其实就是一个饼图，其楔形部分的宽度不同于半径。
# 这是通过“wedgeprops”参数完成的。

# 准备绘制甜甜圈
wedges, _, _ = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

# 标记每个楔形的标签
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

# 循环处理每个楔形，为其添加注释
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(ingredients[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

# 设置图形标题
ax.set_title("Matplotlib bakery: A donut")

# 展示图形
plt.show()
# 创建一个新的图形和轴对象，并设置图形的大小为 6x3，同时设置子图的纵横比为“等比例”
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

# 定义一个配方列表和相应的数据列表，用于描述甜甜圈的配方及其重量
recipe = ["225 g flour",
          "90 g sugar",
          "1 egg",
          "60 g butter",
          "100 ml milk",
          "1/2 package of yeast"]

data = [225, 90, 50, 60, 100, 5]

# 使用给定的数据列表绘制甜甜圈图，并设置每个饼块的宽度为0.5，起始角度为-40度
wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

# 定义一个文本框的属性，用于注解框的外观设置
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

# 定义箭头的属性，包括箭头风格和连接风格，并设置垂直对齐方式为中心
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

# 遍历甜甜圈的每个饼块和对应的文本，确定注解的位置和属性
for i, p in enumerate(wedges):
    # 计算每个饼块的角度中心
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    # 根据x的正负确定文本的水平对齐方式
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    # 根据角度设置连接风格
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    # 在指定位置和文本的相对位置创建注解
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

# 设置图表的标题
ax.set_title("Matplotlib bakery: A donut")

# 显示图表
plt.show()
```