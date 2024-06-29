# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_bbox_tight.py`

```
# 导入所需模块和函数
from io import BytesIO  # 导入字节流处理模块
import platform  # 导入平台信息模块

import numpy as np  # 导入NumPy库

# 导入Matplotlib相关模块和函数
from matplotlib.testing.decorators import image_comparison  # 导入图片比较装饰器
import matplotlib.pyplot as plt  # 导入绘图库
import matplotlib.path as mpath  # 导入路径处理模块
import matplotlib.patches as mpatches  # 导入图形绘制模块
from matplotlib.ticker import FuncFormatter  # 导入自定义格式化刻度模块


@image_comparison(['bbox_inches_tight'], remove_text=True,
                  savefig_kwarg={'bbox_inches': 'tight'})
def test_bbox_inches_tight():
    #: Test that a figure saved using bbox_inches='tight' is clipped correctly
    data = [[66386, 174296, 75131, 577908, 32015],
            [58230, 381139, 78045, 99308, 160454],
            [89135, 80552, 152558, 497981, 603535],
            [78415, 81858, 150656, 193263, 69638],
            [139361, 331509, 343164, 781380, 52269]]

    col_labels = row_labels = [''] * 5

    rows = len(data)
    ind = np.arange(len(col_labels)) + 0.3  # 柱状图组的x坐标位置
    cell_text = []
    width = 0.4  # 柱状图的宽度
    yoff = np.zeros(len(col_labels))  # 堆叠柱状图的底部值
    # 创建图形和轴对象
    fig, ax = plt.subplots(1, 1)
    for row in range(rows):
        ax.bar(ind, data[row], width, bottom=yoff, align='edge', color='b')  # 绘制堆叠柱状图
        yoff = yoff + data[row]  # 更新堆叠的底部值
        cell_text.append([''])  # 添加表格的单元格内容
    plt.xticks([])  # 不显示x轴刻度
    plt.xlim(0, 5)  # 设置x轴范围
    plt.legend([''] * 5, loc=(1.2, 0.2))  # 添加图例并设置位置
    fig.legend([''] * 5, bbox_to_anchor=(0, 0.2), loc='lower left')  # 在图形上添加另一个图例
    # 在轴底部添加表格
    cell_text.reverse()
    plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
              loc='bottom')  # 创建并添加表格到底部


@image_comparison(['bbox_inches_tight_suptile_legend'],
                  savefig_kwarg={'bbox_inches': 'tight'},
                  tol=0.02 if platform.machine() == 'arm64' else 0)
def test_bbox_inches_tight_suptile_legend():
    plt.plot(np.arange(10), label='a straight line')  # 绘制简单的折线图并添加标签
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left')  # 添加图例并设置位置
    plt.title('Axis title')  # 设置图的标题
    plt.suptitle('Figure title')  # 设置图形的总标题

    # 添加一个超长的y轴刻度以确保bbox正确计算
    def y_formatter(y, pos):
        if int(y) == 4:
            return 'The number 4'
        else:
            return str(y)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))  # 设置y轴刻度格式化函数

    plt.xlabel('X axis')  # 设置x轴标签


@image_comparison(['bbox_inches_tight_suptile_non_default.png'],
                  savefig_kwarg={'bbox_inches': 'tight'},
                  tol=0.1)  # 由于只测试裁剪，因此使用大容差
def test_bbox_inches_tight_suptitle_non_default():
    fig, ax = plt.subplots()  # 创建图形和轴对象
    fig.suptitle('Booo', x=0.5, y=1.1)  # 设置总标题的位置和内容


@image_comparison(['bbox_inches_tight_layout.png'], remove_text=True,
                  style='mpl20',
                  savefig_kwarg=dict(bbox_inches='tight', pad_inches='layout'))
def test_bbox_inches_tight_layout_constrained():
    fig, ax = plt.subplots(layout='constrained')  # 创建受约束的布局的图形和轴对象
    fig.get_layout_engine().set(h_pad=0.5)  # 设置布局引擎的垂直间距
    ax.set_aspect('equal')  # 设置轴的纵横比


def test_bbox_inches_tight_layout_notconstrained(tmp_path):
    # 此函数没有图像比较，因此不需要注释该函数内部的代码
    # 创建一个新的图形对象和一个坐标轴对象
    fig, ax = plt.subplots()
    
    # 将当前图形保存为文件，文件格式为 PNG，保存路径为 tmp_path / 'foo.png'
    # bbox_inches='tight' 表示将图的边界框紧密包裹在图形内容周围
    # pad_inches='layout' 参数在没有使用约束/压缩布局时应被忽略，这是一个烟雾测试，用于确认在此情况下 savefig 没有错误。
    fig.savefig(tmp_path / 'foo.png', bbox_inches='tight', pad_inches='layout')
# 使用 @image_comparison 装饰器比较生成的图像与参考图像
@image_comparison(['bbox_inches_tight_clipping'],
                  remove_text=True, savefig_kwarg={'bbox_inches': 'tight'})
def test_bbox_inches_tight_clipping():
    # 测试散点图的 bbox 裁剪，并在 patch 上进行路径裁剪以生成紧凑的 bbox
    plt.scatter(np.arange(10), np.arange(10))
    ax = plt.gca()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    # 创建一个大的矩形，并使用路径进行裁剪
    patch = mpatches.Rectangle([-50, -50], 100, 100,
                               transform=ax.transData,
                               facecolor='blue', alpha=0.5)

    # 创建一个路径，并对其进行缩放
    path = mpath.Path.unit_regular_star(5).deepcopy()
    path.vertices *= 0.25
    # 设置 patch 的裁剪路径和变换
    patch.set_clip_path(path, transform=ax.transAxes)
    # 将 patch 添加到当前图形的艺术家列表中
    plt.gcf().artists.append(patch)


@image_comparison(['bbox_inches_tight_raster'],
                  remove_text=True, savefig_kwarg={'bbox_inches': 'tight'})
def test_bbox_inches_tight_raster():
    """测试使用 tight_layout 进行栅格化"""
    fig, ax = plt.subplots()
    # 在坐标系中绘制一条曲线，并进行栅格化处理
    ax.plot([1.0, 2.0], rasterized=True)


def test_only_on_non_finite_bbox():
    fig, ax = plt.subplots()
    # 在坐标系中添加一个注释，y 值为 NaN
    ax.annotate("", xy=(0, float('nan')))
    ax.set_axis_off()
    # 只需测试保存时不会出错
    fig.savefig(BytesIO(), bbox_inches='tight', format='png')


def test_tight_pcolorfast():
    fig, ax = plt.subplots()
    # 在坐标系中绘制快速色彩图，并设置 y 轴限制
    ax.pcolorfast(np.arange(4).reshape((2, 2)))
    ax.set(ylim=(0, .1))
    buf = BytesIO()
    # 保存图形，并使用 tight 布局
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    height, width, _ = plt.imread(buf).shape
    # 以前，bbox 包括被坐标轴裁剪的图像区域，给定 y 轴限制为 (0, 0.1) 会导致非常高的图像
    assert width > height


def test_noop_tight_bbox():
    from PIL import Image
    x_size, y_size = (10, 7)
    dpi = 100
    # 从头开始创建合适大小的图形
    fig = plt.figure(frameon=False, dpi=dpi, figsize=(x_size/dpi, y_size/dpi))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    data = np.arange(x_size * y_size).reshape(y_size, x_size)
    # 在坐标系中显示数据，使用栅格化处理
    ax.imshow(data, rasterized=True)

    # 当包含栅格化的艺术家时，混合模式渲染器会进行额外的 bbox 调整。
    # 这应该是一个无操作，并且不会影响下一次保存。
    fig.savefig(BytesIO(), bbox_inches='tight', pad_inches=0, format='pdf')

    out = BytesIO()
    fig.savefig(out, bbox_inches='tight', pad_inches=0)
    out.seek(0)
    im = np.asarray(Image.open(out))
    assert (im[:, :, 3] == 255).all()  # Alpha 通道全为 255
    assert not (im[:, :, :3] == 255).all()  # RGB 颜色并非全部为 255
    assert im.shape == (7, 10, 4)  # 图像形状为 (7, 10, 4)


@image_comparison(['bbox_inches_fixed_aspect'], extensions=['png'],
                  remove_text=True, savefig_kwarg={'bbox_inches': 'tight'})
def test_bbox_inches_fixed_aspect():
    # 空函数，用于测试特定的固定纵横比 bbox
    # 使用 plt.rc_context 上下文管理器来设置特定的绘图参数，确保布局约束为 True
    with plt.rc_context({'figure.constrained_layout.use': True}):
        # 创建一个新的图形对象和一个轴对象
        fig, ax = plt.subplots()
        # 在轴上绘制一条线，坐标点为 (0,0) 和 (1,1)
        ax.plot([0, 1])
        # 设置 x 轴的显示范围为 0 到 1
        ax.set_xlim(0, 1)
        # 设置绘图区域的纵横比为相等
        ax.set_aspect('equal')
```