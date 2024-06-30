# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_polynomial_interpolation.py`

```
# 创建多项式特征和样条插值，用于拟合训练数据点并展示插值效果

# 设置线宽
lw = 2
# 创建图形和轴对象
fig, ax = plt.subplots()
# 设置绘图循环颜色
ax.set_prop_cycle(
    color=["black", "teal", "yellowgreen", "gold", "darkorange", "tomato"]
)
# 绘制原始函数曲线
ax.plot(x_plot, f(x_plot), linewidth=lw, label="ground truth")

# 绘制训练点
ax.scatter(x_train, y_train, label="training points")

# 对于不同的多项式阶数进行循环
for degree in [3, 4, 5]:
    # 使用 PolynomialFeatures 对象和 Ridge 对象创建一个机器学习管道模型，其中多项式特征的阶数由变量 degree 指定，正则化参数 alpha 设置为 1e-3
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    
    # 使用训练数据集 X_train 和对应的目标值 y_train 来训练模型
    model.fit(X_train, y_train)
    
    # 使用训练好的模型对 X_plot 进行预测，得到预测值 y_plot
    y_plot = model.predict(X_plot)
    
    # 在当前的坐标轴上绘制 x_plot 和 y_plot，使用 label 指定标签为当前多项式阶数的描述
    ax.plot(x_plot, y_plot, label=f"degree {degree}")
# 创建一个管道模型，包含 B-spline 特征转换器和 Ridge 回归器，用于拟合训练数据
model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model.fit(X_train, y_train)

# 使用模型进行预测得到预测值
y_plot = model.predict(X_plot)

# 在图形上绘制预测结果的曲线，并设置标签为 "B-spline"
ax.plot(x_plot, y_plot, label="B-spline")

# 在图形上添加图例，位置在底部中心
ax.legend(loc="lower center")

# 设置 y 轴的显示范围为 -20 到 10
ax.set_ylim(-20, 10)

# 显示图形
plt.show()

# %%
# 这里清楚地展示了高阶多项式可以更好地拟合数据。但是，同时，过高阶的多项式可能展示出
# 不期望的振荡行为，并且在超出拟合数据范围进行外推时尤为危险。这就是 B-spline
# 的优势所在。它们通常能与多项式一样很好地拟合数据，并且表现出非常平滑的行为。
# 它们还有很好的选项来控制外推行为，默认情况下会使用常数进行外推。请注意，通常你会
# 增加节点数而保持 `degree=3` 不变。
#
# 为了更好地理解生成的特征基函数，我们分别绘制两个转换器的所有列。

fig, axes = plt.subplots(ncols=2, figsize=(16, 5))

# 创建并拟合多项式特征转换器，绘制其转换后的结果
pft = PolynomialFeatures(degree=3).fit(X_train)
axes[0].plot(x_plot, pft.transform(X_plot))
axes[0].legend(axes[0].lines, [f"degree {n}" for n in range(4)])
axes[0].set_title("PolynomialFeatures")

# 创建并拟合 B-spline 特征转换器，绘制其转换后的结果
splt = SplineTransformer(n_knots=4, degree=3).fit(X_train)
axes[1].plot(x_plot, splt.transform(X_plot))
axes[1].legend(axes[1].lines, [f"spline {n}" for n in range(6)])
axes[1].set_title("SplineTransformer")

# 绘制 B-spline 的节点位置
knots = splt.bsplines_[0].t
axes[1].vlines(knots[3:-3], ymin=0, ymax=0.8, linestyles="dashed")

# 显示图形
plt.show()

# %%
# 在左图中，我们可以看到对应于简单单项式的线，从 `x**0` 到 `x**3`。在右图中，
# 我们可以看到 `degree=3` 的六个 B-spline 基函数，以及在 `fit` 过程中选择的四个
# 节点位置。请注意，在拟合区间的左右各有 `degree` 个额外的节点。这些节点是为了
# 技术原因而存在的，因此我们不展示它们。每个基函数具有局部支持，并且在拟合范围
# 之外继续为常数。这种外推行为可以通过参数 `extrapolation` 进行更改。

# %%
# 周期性样条
# ----------------
# 在前面的示例中，我们看到了多项式和样条在超出训练观测范围进行外推时的局限性。
# 在某些情况下，例如季节效应，我们期望底层信号的周期性延续。这种效果可以通过
# 使用周期性样条来建模，它们在第一个和最后一个节点处具有相等的函数值和导数。
# 在下面的例子中，我们展示了如何利用周期性样条，在提供周期性附加信息的情况下，
# 在训练数据范围内外提供更好的拟合效果。样条的周期是第一个和最后一个节点之间的
# 距离，我们在这里手动指定。
#
# 周期样条插值对于自然周期特征（如一年中的日期）也非常有用，因为边界结点处的平滑性可以防止转换值的跳跃（例如从12月31日到1月1日）。对于这种自然周期特征或更一般的已知周期特征，建议通过手动设置结点将此信息明确传递给 `SplineTransformer`。

# %%
def g(x):
    """被周期样条插值逼近的函数。"""
    return np.sin(x) - 0.7 * np.cos(x * 3)

# 使用训练数据 x_train 计算函数 g 的值
y_train = g(x_train)

# 将测试数据向未来扩展：
x_plot_ext = np.linspace(-1, 21, 200)
X_plot_ext = x_plot_ext[:, np.newaxis]

lw = 2
fig, ax = plt.subplots()
ax.set_prop_cycle(color=["black", "tomato", "teal"])
ax.plot(x_plot_ext, g(x_plot_ext), linewidth=lw, label="ground truth")  # 绘制原始函数曲线
ax.scatter(x_train, y_train, label="training points")  # 绘制训练数据点

# 针对不同的变换器和标签组合进行迭代绘图
for transformer, label in [
    (SplineTransformer(degree=3, n_knots=10), "spline"),  # 使用默认结点数创建样条变换器
    (
        SplineTransformer(
            degree=3,
            knots=np.linspace(0, 2 * np.pi, 10)[:, None],
            extrapolation="periodic",
        ),
        "periodic spline",  # 使用周期外推创建样条变换器
    ),
]:
    model = make_pipeline(transformer, Ridge(alpha=1e-3))  # 创建管道模型
    model.fit(X_train, y_train)  # 对训练数据进行拟合
    y_plot_ext = model.predict(X_plot_ext)  # 预测扩展后的测试数据
    ax.plot(x_plot_ext, y_plot_ext, label=label)  # 绘制预测结果曲线

ax.legend()  # 显示图例
fig.show()

# %% 再次绘制基础样条曲线。
fig, ax = plt.subplots()
knots = np.linspace(0, 2 * np.pi, 4)
# 创建周期外推的样条变换器，并在训练数据上进行拟合
splt = SplineTransformer(knots=knots[:, None], degree=3, extrapolation="periodic").fit(
    X_train
)
ax.plot(x_plot_ext, splt.transform(X_plot_ext))  # 绘制转换后的样条曲线
ax.legend(ax.lines, [f"spline {n}" for n in range(3)])  # 显示图例
plt.show()
```