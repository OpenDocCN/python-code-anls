# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_antialiasing.py`

```
# %%
# 首先生成一个 450x450 像素的图像，其中包含不同频率的内容
N = 450
x = np.arange(N) / N - 0.5
y = np.arange(N) / N - 0.5
# 创建一个全为1的数组
aa = np.ones((N, N))
# 每隔一行将数组值设置为-1，用于创建高频效果
aa[::2, :] = -1

# 创建坐标网格
X, Y = np.meshgrid(x, y)
# 计算距离中心的径向距离
R = np.sqrt(X**2 + Y**2)
f0 = 5
k = 100
# 计算根据径向距离生成的复杂波形
a = np.sin(np.pi * 2 * (f0 * R + k * R**2 / 2))
# 将左侧区域设为-1或1，以创建具有高对比度的区域
a[:int(N / 2), :][R[:int(N / 2), :] < 0.4] = -1
a[:int(N / 2), :][R[:int(N / 2), :] < 0.3] = 1
# 将生成的波形复制到整体图像的一部分
aa[:, int(N / 3):] = a[:, int(N / 3):]
# 将最终的图像赋值给变量a
a = aa

# %%
# 以下图像是从450个数据像素减少到125像素或250像素（取决于显示器分辨率）。
# 'nearest'插值产生的莫尔纹图案是由于高频数据的子采样。
# 'antialiased'插值的图像仍然有一些莫尔纹，但大大减少了。
#
# 在'data'插值和'rgba'插值之间存在显著差异。
# 图像左侧的红色和蓝色交替条纹在子采样时变得更加明显。
# 在'data'空间（默认情况下）进行抗锯齿滤波使得条纹接近白色，
# 因为-1和+1的平均值是零，在这个色彩映射中零代表白色。
#
# 相反，在'rgba'空间进行抗锯齿处理时，红色和蓝色在视觉上混合形成紫色。
# 这种行为更像是典型的图像处理软件包，但请注意，紫色不在原始色彩映射中，
# 因此不再可能将单个像素反转为其数据值。

fig, axs = plt.subplots(2, 2, figsize=(5, 6), layout='constrained')
# 在第一个子图中显示图像a，使用'nearest'插值和'RdBu_r'颜色映射
axs[0, 0].imshow(a, interpolation='nearest', cmap='RdBu_r')
# 设置第一个子图的显示范围
axs[0, 0].set_xlim(100, 200)
axs[0, 0].set_ylim(275, 175)
# 设置第一个子图的标题
axs[0, 0].set_title('Zoom')
# 对 axs.flat[1:] 进行迭代，ax 为当前迭代的子图对象，interp 为插值方法，space 为插值阶段
for ax, interp, space in zip(axs.flat[1:],
                             ['nearest', 'antialiased', 'antialiased'],
                             ['data', 'data', 'rgba']):
    # 在当前子图 ax 上显示图像 a，使用指定的插值方法 interp，指定插值阶段 space，并使用 'RdBu_r' 颜色映射
    ax.imshow(a, interpolation=interp, interpolation_stage=space,
              cmap='RdBu_r')
    # 设置子图标题，展示当前的插值方法 interp 和插值阶段 space
    ax.set_title(f"interpolation='{interp}'\nspace='{space}'")
# 显示所有子图
plt.show()

# %%
# 即使使用 'nearest' 插值方法对图像进行上采样，当上采样因子不是整数时，可能会产生 Moiré 图案。
# 下面的图像将 500 个数据像素上采样到 530 个渲染像素。你可能会注意到一个由于额外的 24 个像素而产生的
# 30 条线状伪影。由于插值是 'nearest'，这些像素与相邻行的像素相同，因此会在局部拉伸图像，使其看起来畸形。
fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.imshow(a, interpolation='nearest', cmap='gray')
ax.set_title("upsampled by factor a 1.048, interpolation='nearest'")
plt.show()

# %%
# 更好的抗锯齿算法可以减少这种效应：
fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.imshow(a, interpolation='antialiased', cmap='gray')
ax.set_title("upsampled by factor a 1.048, interpolation='antialiased'")
plt.show()

# %%
# 除了默认的 'hanning' 抗锯齿外，`~.Axes.imshow` 支持多种不同的插值算法，这些算法在处理模式时可能效果更好或更差。
fig, axs = plt.subplots(1, 2, figsize=(7, 4), layout='constrained')
for ax, interp in zip(axs, ['hanning', 'lanczos']):
    # 在当前子图 ax 上显示图像 a，使用指定的插值方法 interp，灰度颜色映射
    ax.imshow(a, interpolation=interp, cmap='gray')
    # 设置子图标题，展示当前的插值方法 interp
    ax.set_title(f"interpolation='{interp}'")
plt.show()

# %%
#
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.imshow`
```