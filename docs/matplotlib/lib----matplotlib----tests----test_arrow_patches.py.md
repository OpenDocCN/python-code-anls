# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_arrow_patches.py`

```
import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches

# 定义一个函数，用于在指定的坐标系中绘制箭头
def draw_arrow(ax, t, r):
    ax.annotate('', xy=(0.5, 0.5 + r), xytext=(0.5, 0.5), size=30,
                arrowprops=dict(arrowstyle=t,
                                fc="b", ec='k'))

# 使用 matplotlib 的 image_comparison 装饰器来比较图片，并设置容忍度
@image_comparison(['fancyarrow_test_image'],
                  tol=0.012 if platform.machine() == 'arm64' else 0)
def test_fancyarrow():
    # 添加了一个零以测试在 issue 3930 中描述的除零错误
    r = [0.4, 0.3, 0.2, 0.1, 0]
    t = ["fancy", "simple", mpatches.ArrowStyle.Fancy()]

    # 创建一个多子图的图形对象
    fig, axs = plt.subplots(len(t), len(r), squeeze=False,
                            figsize=(8, 4.5), subplot_kw=dict(aspect=1))

    # 在每个子图中绘制箭头
    for i_r, r1 in enumerate(r):
        for i_t, t1 in enumerate(t):
            ax = axs[i_t, i_r]
            draw_arrow(ax, t1, r1)
            ax.tick_params(labelleft=False, labelbottom=False)

# 使用 matplotlib 的 image_comparison 装饰器来比较图片
@image_comparison(['boxarrow_test_image.png'])
def test_boxarrow():

    # 获取所有的 BoxStyle 样式
    styles = mpatches.BoxStyle.get_styles()

    n = len(styles)
    spacing = 1.2

    # 计算图形的高度
    figheight = (n * spacing + .5)
    fig = plt.figure(figsize=(4 / 1.5, figheight / 1.5))

    fontsize = 0.3 * 72

    # 在图中按顺序绘制每种样式的文本框
    for i, stylename in enumerate(sorted(styles)):
        fig.text(0.5, ((n - i) * spacing - 0.5)/figheight, stylename,
                 ha="center",
                 size=fontsize,
                 transform=fig.transFigure,
                 bbox=dict(boxstyle=stylename, fc="w", ec="k"))

# 定义一个准备 FancyArrowPatch 的辅助函数
def __prepare_fancyarrow_dpi_cor_test():
    """
    Convenience function that prepares and returns a FancyArrowPatch. It aims
    at being used to test that the size of the arrow head does not depend on
    the DPI value of the exported picture.

    NB: this function *is not* a test in itself!
    """
    # 创建一个图形对象和坐标系，并添加 FancyArrowPatch
    fig2 = plt.figure("fancyarrow_dpi_cor_test", figsize=(4, 3), dpi=50)
    ax = fig2.add_subplot()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.add_patch(mpatches.FancyArrowPatch(posA=(0.3, 0.4), posB=(0.8, 0.6),
                                          lw=3, arrowstyle='->',
                                          mutation_scale=100))
    return fig2

# 使用 matplotlib 的 image_comparison 装饰器来比较图片，设置 DPI 为 100
@image_comparison(['fancyarrow_dpi_cor_100dpi.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.02,
                  savefig_kwarg=dict(dpi=100))
def test_fancyarrow_dpi_cor_100dpi():
    """
    Check the export of a FancyArrowPatch @ 100 DPI. FancyArrowPatch is
    instantiated through a dedicated function because another similar test
    checks a similar export but with a different DPI value.

    Remark: test only a rasterized format.
    """

    __prepare_fancyarrow_dpi_cor_test()

# 使用 matplotlib 的 image_comparison 装饰器来比较图片，设置 DPI 为 200
@image_comparison(['fancyarrow_dpi_cor_200dpi.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.02,
                  savefig_kwarg=dict(dpi=200))
def test_fancyarrow_dpi_cor_200dpi():
    """
    Check the export of a FancyArrowPatch @ 200 DPI. FancyArrowPatch is
    instantiated through a dedicated function because another similar test
    checks a similar export but with a different DPI value.

    Remark: test only a rasterized format.
    """

    __prepare_fancyarrow_dpi_cor_test()
    """
    As test_fancyarrow_dpi_cor_100dpi, but exports @ 200 DPI. The relative size
    of the arrow head should be the same.
    """

    # 调用一个准备函数来设置测试环境，此函数名为__prepare_fancyarrow_dpi_cor_test
    __prepare_fancyarrow_dpi_cor_test()
# 使用装饰器创建图像比较测试函数，比较生成的图像与预期的基准图像是否一致
@image_comparison(['fancyarrow_dash.png'], remove_text=True, style='default')
def test_fancyarrow_dash():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建第一个箭头对象
    e = mpatches.FancyArrowPatch((0, 0), (0.5, 0.5),
                                 arrowstyle='-|>',  # 设置箭头样式为 '-|>'
                                 connectionstyle='angle3,angleA=0,angleB=90',  # 设置连接样式
                                 mutation_scale=10.0,  # 突变比例
                                 linewidth=2,  # 线宽
                                 linestyle='dashed',  # 线型为虚线
                                 color='k')  # 颜色为黑色
    # 创建第二个箭头对象
    e2 = mpatches.FancyArrowPatch((0, 0), (0.5, 0.5),
                                  arrowstyle='-|>',  # 设置箭头样式为 '-|>'
                                  connectionstyle='angle3',  # 设置连接样式
                                  mutation_scale=10.0,  # 突变比例
                                  linewidth=2,  # 线宽
                                  linestyle='dotted',  # 线型为点线
                                  color='k')  # 颜色为黑色
    # 将箭头对象添加到坐标轴
    ax.add_patch(e)
    ax.add_patch(e2)


# 使用装饰器创建图像比较测试函数，比较生成的图像与预期的基准图像是否一致
@image_comparison(['arrow_styles.png'], style='mpl20', remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.02)
def test_arrow_styles():
    # 获取所有箭头样式
    styles = mpatches.ArrowStyle.get_styles()

    # 箭头样式数量
    n = len(styles)
    # 创建新的图形和坐标轴对象，设置图形尺寸
    fig, ax = plt.subplots(figsize=(8, 8))
    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n)
    # 调整图形边界
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 遍历每种箭头样式并在图形上绘制
    for i, stylename in enumerate(sorted(styles)):
        # 创建箭头对象
        patch = mpatches.FancyArrowPatch((0.1 + (i % 2)*0.05, i),
                                         (0.45 + (i % 2)*0.05, i),
                                         arrowstyle=stylename,  # 箭头样式
                                         mutation_scale=25)  # 突变比例
        # 将箭头对象添加到坐标轴
        ax.add_patch(patch)

    # 额外绘制特定角度的箭头样式
    for i, stylename in enumerate([']-[', ']-', '-[', '|-|']):
        style = stylename
        # 根据箭头样式特定条件添加角度参数
        if stylename[0] != '-':
            style += ',angleA=ANGLE'
        if stylename[-1] != '-':
            style += ',angleB=ANGLE'

        # 遍历每种角度并绘制箭头
        for j, angle in enumerate([-30, 60]):
            arrowstyle = style.replace('ANGLE', str(angle))
            # 创建箭头对象
            patch = mpatches.FancyArrowPatch((0.55, 2*i + j), (0.9, 2*i + j),
                                             arrowstyle=arrowstyle,  # 箭头样式
                                             mutation_scale=25)  # 突变比例
            # 将箭头对象添加到坐标轴
            ax.add_patch(patch)


# 使用装饰器创建图像比较测试函数，比较生成的图像与预期的基准图像是否一致
@image_comparison(['connection_styles.png'], style='mpl20', remove_text=True,
                  tol=0.013 if platform.machine() == 'arm64' else 0)
def test_connection_styles():
    # 获取所有连接样式
    styles = mpatches.ConnectionStyle.get_styles()

    # 连接样式数量
    n = len(styles)
    # 创建新的图形和坐标轴对象，设置图形尺寸
    fig, ax = plt.subplots(figsize=(6, 10))
    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n)

    # 遍历每种连接样式并在图形上绘制
    for i, stylename in enumerate(sorted(styles)):
        # 创建箭头对象
        patch = mpatches.FancyArrowPatch((0.1, i), (0.8, i + 0.5),
                                         arrowstyle="->",  # 箭头样式
                                         connectionstyle=stylename,  # 连接样式
                                         mutation_scale=25)  # 突变比例
        # 将箭头对象添加到坐标轴
        ax.add_patch(patch)


# 定义一个测试函数，创建一个具有特定角度的连接样式对象
def test_invalid_intersection():
    # 创建一个具有特定角度的连接样式对象
    conn_style_1 = mpatches.ConnectionStyle.Angle3(angleA=20, angleB=200)
    # 创建一个具有指定连接风格的箭头补丁，连接起始点 (.2, .2) 和结束点 (.5, .5)
    p1 = mpatches.FancyArrowPatch((.2, .2), (.5, .5),
                                  connectionstyle=conn_style_1)
    # 使用 pytest 模块断言抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 将箭头补丁 p1 添加到当前坐标轴上
        plt.gca().add_patch(p1)
    
    # 创建一个具有另一种指定连接风格的箭头补丁，连接起始点 (.2, .2) 和结束点 (.5, .5)
    conn_style_2 = mpatches.ConnectionStyle.Angle3(angleA=20, angleB=199.9)
    p2 = mpatches.FancyArrowPatch((.2, .2), (.5, .5),
                                  connectionstyle=conn_style_2)
    # 将箭头补丁 p2 添加到当前坐标轴上
    plt.gca().add_patch(p2)
```