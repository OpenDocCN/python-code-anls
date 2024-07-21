# `.\pytorch\benchmarks\fuser\plot_speedups.py`

```py
# 导入 pandas 库，用于数据处理和分析
import pandas

# 从名为 "perf.csv" 的 CSV 文件中读取数据，存储到 DataFrame 对象 df 中
df = pandas.read_csv("perf.csv")

# 获取 df 中 "operator" 列中唯一的操作符列表
ops = pandas.unique(df["operator"])

# 计算操作符列表的长度
nops = len(ops)

# 创建数据透视表，计算每个操作符和形状的组合下的时间数据，并以不同的融合器为列
pivot_op_shape = df.pivot_table(
    values="time", index=["operator", "shape"], columns=["fuser"]
)

# 计算性能提升比例，将每个操作符和形状的性能与 "eager" 方式的性能比较，得到速度提升比例
pivot_speedups = (pivot_op_shape.T / pivot_op_shape["eager"]).T

# 导入 matplotlib.pyplot 库，用于绘制图形
import matplotlib.pyplot as plt

# 设置绘图的默认图形大小为 (20, 100)
plt.rcParams["figure.figsize"] = (20, 100)

# 创建包含 nops 个子图的图形布局，调整子图之间的垂直间距
fig, axs = plt.subplots(nops)
plt.subplots_adjust(hspace=0.5)

# 遍历操作符列表 ops，并绘制每个操作符对应的性能提升比例条形图
for idx, op in enumerate(ops):
    # 获取特定操作符 op 的性能提升比例数据
    op_speedups = pivot_speedups.T[op].T
    
    # 在 axs[idx] 上绘制操作符 op 的条形图，设置 y 轴范围为 (0, 2)，x 轴标签旋转 45 度
    op_speedups.plot(ax=axs[idx], kind="bar", ylim=(0, 2), rot=45)
    
    # 设置子图 axs[idx] 的标题为操作符 op
    axs[idx].set_title(op)
    
    # 设置子图 axs[idx] 的 x 轴标签为空字符串，避免重复显示
    axs[idx].set_xlabel("")

# 将绘制好的图形保存为 "perf.png" 文件
plt.savefig("perf.png")
```