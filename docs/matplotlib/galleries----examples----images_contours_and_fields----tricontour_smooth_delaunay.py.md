# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\tricontour_smooth_delaunay.py`

```py
# ----------------------------------------------------------------------------
# Analytical test function
# ----------------------------------------------------------------------------
# 定义一个代表实验结果的解析函数
def experiment_res(x, y):
    # 对输入的 x 坐标进行线性变换
    x = 2 * x
    # 计算第一个圆的半径和角度
    r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    # 计算第二个圆的半径和角度
    r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    # 计算最终的函数结果 z
    z = (4 * (np.exp((r1/10)**2) - 1) * 30 * np.cos(3 * theta1) +
         (np.exp((r2/10)**2) - 1) * 30 * np.cos(5 * theta2) +
         2 * (x**2 + y**2))
    # 返回归一化的结果
    return (np.max(z) - z) / (np.max(z) - np.min(z))

# ----------------------------------------------------------------------------
# Generating the initial data test points and triangulation for the demo
# ----------------------------------------------------------------------------
# 用户参数用于生成测试点数据

# 测试点的数量，取值范围为 3 到 5000（subdiv=3 时）
n_test = 200

# 初始网格的递归分割次数，用于平滑绘图
subdiv = 3

# 初始三角网格中无效三角形的比例，设置为 0 表示没有遮罩
init_mask_frac = 0.0

# 最小圆比率，低于此比率的边界三角形将被遮罩
min_circle_ratio = .01

# 随机生成测试点
random_gen = np.random.RandomState(seed=19680801)
x_test = random_gen.uniform(-1., 1., size=n_test)
y_test = random_gen.uniform(-1., 1., size=n_test)
z_test = experiment_res(x_test, y_test)

# 使用 Delaunay 三角化生成初始网格
# 创建一个 Triangulation 对象，使用 x_test 和 y_test 作为输入
tri = Triangulation(x_test, y_test)
# 获取三角形的数量
ntri = tri.triangles.shape[0]

# 初始化一个用于标记无效数据的布尔类型数组
mask_init = np.zeros(ntri, dtype=bool)
# 随机选择一部分三角形进行遮蔽（掩盖）
masked_tri = random_gen.randint(0, ntri, int(ntri * init_mask_frac))
# 将选中的三角形在 mask_init 中标记为 True
mask_init[masked_tri] = True
# 将标记好的 mask_init 应用于 Triangulation 对象
tri.set_mask(mask_init)

# ----------------------------------------------------------------------------
# 改进三角网格以便绘制高分辨率图形：移除平坦的三角形
# ----------------------------------------------------------------------------
# 获取三角形的平坦面掩码
mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
# 将获取的掩码应用于 Triangulation 对象
tri.set_mask(mask)

# 对数据进行细化处理
refiner = UniformTriRefiner(tri)
tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)

# 用于比较的理论结果
z_expected = experiment_res(tri_refi.x, tri_refi.y)

# 用于演示：加载用于绘制的“平坦”三角形
flat_tri = Triangulation(x_test, y_test)
flat_tri.set_mask(~mask)

# ----------------------------------------------------------------------------
# 现在开始绘图
# ----------------------------------------------------------------------------
# 用户绘图选项
plot_tri = True          # 绘制基础三角网格
plot_masked_tri = True   # 绘制被排除的过于平坦的三角形
plot_refi_tri = False    # 绘制细化后的三角网格
plot_expected = False    # 绘制理论函数值以进行比较

# 设置等高线的水平值
levels = np.arange(0., 1., 0.025)

# 创建图形和坐标轴
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_title("Filtering a Delaunay mesh\n"
             "(application to high-resolution tricontouring)")

# 1) 绘制细化后数据的等高线：
ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap='Blues',
              linewidths=[2.0, 0.5, 1.0, 0.5])
# 2) 绘制理论数据的等高线（虚线）：
if plot_expected:
    ax.tricontour(tri_refi, z_expected, levels=levels, cmap='Blues',
                  linestyles='--')
# 3) 绘制插值操作后的细网格：
if plot_refi_tri:
    ax.triplot(tri_refi, color='0.97')
# 4) 绘制初始的“粗略”三角网格：
if plot_tri:
    ax.triplot(tri, color='0.7')
# 5) 绘制来自朴素 Delaunay 三角化的未验证三角形：
if plot_masked_tri:
    ax.triplot(flat_tri, color='red')

plt.show()
```