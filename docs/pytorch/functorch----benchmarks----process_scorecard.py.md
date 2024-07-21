# `.\pytorch\functorch\benchmarks\process_scorecard.py`

```
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 pandas 库，用于数据操作和分析
import pandas

# 使用 pandas 读取名为 "perf.csv" 的 CSV 文件，将其加载到 DataFrame 中
df = pandas.read_csv("perf.csv")

# 从 DataFrame 中提取出唯一的操作符列表
ops = pandas.unique(df["operator"])
# 计算操作符的数量
nops = len(ops)

# 根据 "operator" 和 "shape" 列，以及 "fuser" 列的值创建数据透视表
pivot_op_shape = df.pivot_table(values="time", index=["operator", "shape"], columns=["fuser"])
# 计算速度提升比例，转置后再转置回来以便后续使用
pivot_speedups = (pivot_op_shape.T / pivot_op_shape["eager"]).T

# 设置图形参数，指定图形的尺寸为 (20, 100)
plt.rcParams["figure.figsize"] = (20, 100)
# 创建包含 nops 个子图的图形对象，调整子图之间的垂直间距
fig, axs = plt.subplots(nops)
plt.subplots_adjust(hspace=0.5)

# 遍历操作符列表，并在每个子图上绘制柱状图
for idx, op in enumerate(ops):
    # 从透视表中提取特定操作符的速度提升数据
    op_speedups = pivot_speedups.T[op].T
    # 在当前子图 axs[idx] 上绘制柱状图，限定 y 轴范围在 (0, 5)，标签旋转角度为 45 度
    op_speedups.plot(ax=axs[idx], kind="bar", ylim=(0, 5), rot=45)
    # 设置当前子图的标题为操作符名称
    axs[idx].set_title(op)
    # 设置当前子图的 x 轴标签为空字符串，避免重复显示
    axs[idx].set_xlabel("")

# 将生成的图形保存为 SVG 文件 "scorecard.svg"
plt.savefig("scorecard.svg")
```