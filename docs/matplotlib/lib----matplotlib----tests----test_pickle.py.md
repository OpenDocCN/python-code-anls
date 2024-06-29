# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_pickle.py`

```
from io import BytesIO  # 导入 BytesIO 类，用于创建二进制数据的内存缓冲区
import ast  # 导入 ast 模块，用于处理抽象语法树
import pickle  # 导入 pickle 模块，用于序列化 Python 对象
import pickletools  # 导入 pickletools 模块，用于分析和调试 pickle 数据流

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import matplotlib as mpl  # 导入 Matplotlib 库的顶层包
from matplotlib import cm  # 导入 Matplotlib 的 colormap 模块
from matplotlib.testing import subprocess_run_helper  # 导入 Matplotlib 测试辅助模块
from matplotlib.testing.decorators import check_figures_equal  # 导入 Matplotlib 测试装饰器
from matplotlib.dates import rrulewrapper  # 导入 Matplotlib 的日期处理模块
from matplotlib.lines import VertexSelector  # 导入 Matplotlib 的顶点选择器
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，用于绘图
import matplotlib.transforms as mtransforms  # 导入 Matplotlib 的变换模块
import matplotlib.figure as mfigure  # 导入 Matplotlib 的 figure 模块
from mpl_toolkits.axes_grid1 import axes_divider, parasite_axes  # 导入 Matplotlib 的坐标轴网格分割器和寄生坐标轴

def test_simple():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 将图形对象序列化并存储到一个字节流中
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)

    # 在图形上创建一个子图对象
    ax = plt.subplot(121)
    # 将子图对象序列化并存储到一个字节流中
    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    # 在极坐标上创建一个子图对象
    ax = plt.axes(projection='polar')
    # 绘制一条直线，并添加图例
    plt.plot(np.arange(10), label='foobar')
    plt.legend()
    # 将极坐标子图对象序列化并存储到一个字节流中
    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

#    ax = plt.subplot(121, projection='hammer')
#    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    # 创建一个新的 Matplotlib 图形对象
    plt.figure()
    # 绘制柱状图并获取当前的坐标轴对象
    plt.bar(x=np.arange(10), height=np.arange(10))
    # 将当前坐标轴对象序列化并存储到一个字节流中
    pickle.dump(plt.gca(), BytesIO(), pickle.HIGHEST_PROTOCOL)

    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 创建一个新的坐标轴对象
    ax = plt.axes()
    # 绘制折线图
    plt.plot(np.arange(10))
    # 设置坐标轴的纵向比例为对数尺度
    ax.set_yscale('log')
    # 将图形对象序列化并存储到一个字节流中
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


def _generate_complete_test_figure(fig_ref):
    # 设置参考图形对象的尺寸
    fig_ref.set_size_inches((10, 6))
    # 设定当前图形为参考图形对象
    plt.figure(fig_ref)

    # 设置总标题
    plt.suptitle('Can you fit any more in a figure?')

    # 创建一些任意数据
    x, y = np.arange(8), np.arange(10)
    data = u = v = np.linspace(0, 10, 80).reshape(10, 8)
    v = np.sin(v * -0.6)

    # 确保列表也能正确序列化
    plt.subplot(3, 3, 1)
    plt.plot(list(range(10)))
    plt.ylabel("hello")

    plt.subplot(3, 3, 2)
    plt.contourf(data, hatches=['//', 'ooo'])
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.pcolormesh(data)

    plt.subplot(3, 3, 4)
    plt.imshow(data)
    plt.ylabel("hello\nworld!")

    plt.subplot(3, 3, 5)
    plt.pcolor(data)

    ax = plt.subplot(3, 3, 6)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.streamplot(x, y, u, v)

    ax = plt.subplot(3, 3, 7)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.quiver(x, y, u, v)

    plt.subplot(3, 3, 8)
    plt.scatter(x, x ** 2, label='$x^2$')
    plt.legend(loc='upper left')

    plt.subplot(3, 3, 9)
    plt.errorbar(x, x * -0.5, xerr=0.2, yerr=0.4, label='$-.5 x$')
    plt.legend(draggable=True)

    # 确保子图的父子关系正常
    subfigs = fig_ref.subfigures(2)
    subfigs[0].subplots(1, 2)
    subfigs[1].subplots(1, 2)

    fig_ref.align_ylabels()  # 测试处理 _align_label_groups 组合器的功能


@mpl.style.context("default")
@check_figures_equal(extensions=["png"])
def test_complete(fig_test, fig_ref):
    _generate_complete_test_figure(fig_ref)
    # 绘图完成后，测试其是否可以被序列化
    pkl = pickle.dumps(fig_ref, pickle.HIGHEST_PROTOCOL)
    # FigureCanvasAgg 可以被序列化，通常 GUI 画布不能被序列化，但应该可以
    # 检查反序列化后的对象中是否存在 FigureCanvasAgg 的引用，确保与 GUI 工具包无关，并使用 Agg 运行测试
    assert "FigureCanvasAgg" not in [arg for op, arg, pos in pickletools.genops(pkl)]
    # 反序列化 pickle 数据，加载为 Python 对象
    loaded = pickle.loads(pkl)
    # 绘制加载后对象的画布
    loaded.canvas.draw()

    # 将测试图形的尺寸设置为加载对象的尺寸
    fig_test.set_size_inches(loaded.get_size_inches())
    # 将加载对象的画布渲染为 RGBA 缓冲图像，并添加到测试图形上
    fig_test.figimage(loaded.canvas.renderer.buffer_rgba())

    # 关闭已加载的图形对象
    plt.close(loaded)
def _pickle_load_subprocess():
    # 导入必要的库
    import os
    import pickle

    # 从环境变量中获取 pickle 文件路径
    path = os.environ['PICKLE_FILE_PATH']

    # 打开 pickle 文件进行反序列化
    with open(path, 'rb') as blob:
        fig = pickle.load(blob)

    # 打印反序列化后的对象的字节表示
    print(str(pickle.dumps(fig)))


@mpl.style.context("default")
@check_figures_equal(extensions=['png'])
def test_pickle_load_from_subprocess(fig_test, fig_ref, tmp_path):
    # 生成完整的测试图形
    _generate_complete_test_figure(fig_ref)

    # 创建临时 pickle 文件路径
    fp = tmp_path / 'sinus.pickle'
    assert not fp.exists()

    # 将参考图形序列化并写入 pickle 文件
    with fp.open('wb') as file:
        pickle.dump(fig_ref, file, pickle.HIGHEST_PROTOCOL)
    assert fp.exists()

    # 在子进程中运行 _pickle_load_subprocess 函数
    proc = subprocess_run_helper(
        _pickle_load_subprocess,
        timeout=60,
        extra_env={'PICKLE_FILE_PATH': str(fp), 'MPLBACKEND': 'Agg'}
    )

    # 从子进程的输出中反序列化加载的图形对象
    loaded_fig = pickle.loads(ast.literal_eval(proc.stdout))

    # 绘制加载后的图形
    loaded_fig.canvas.draw()

    # 设置测试用图形的大小与加载的图形相同
    fig_test.set_size_inches(loaded_fig.get_size_inches())
    fig_test.figimage(loaded_fig.canvas.renderer.buffer_rgba())

    # 关闭加载的图形
    plt.close(loaded_fig)


def test_gcf():
    # 创建一个带有标签的图形对象
    fig = plt.figure("a label")

    # 将图形对象序列化为字节流
    buf = BytesIO()
    pickle.dump(fig, buf, pickle.HIGHEST_PROTOCOL)

    # 关闭所有图形
    plt.close("all")

    # 断言没有任何图形存在
    assert plt._pylab_helpers.Gcf.figs == {}

    # 从字节流中反序列化加载图形对象
    fig = pickle.loads(buf.getbuffer())

    # 断言图形管理器再次存在
    assert plt._pylab_helpers.Gcf.figs != {}

    # 断言加载的图形对象的标签为 "a label"
    assert fig.get_label() == "a label"


def test_no_pyplot():
    # 测试不使用 pyplot 创建的图形对象的可序列化性
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    # 创建一个 Figure 对象
    fig = mfigure.Figure()

    # 创建 FigureCanvasPdf 对象
    _ = FigureCanvasPdf(fig)

    # 向图形对象添加子图并绘制
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])

    # 将图形对象序列化并写入字节流中
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


def test_renderer():
    # 测试渲染器对象的序列化
    from matplotlib.backends.backend_agg import RendererAgg

    # 创建 RendererAgg 对象并将其序列化
    renderer = RendererAgg(10, 20, 30)
    pickle.dump(renderer, BytesIO())


def test_image():
    # 在 v1.4.0 之前，Image 对象在绘制后会缓存不可序列化的数据
    from matplotlib.backends.backend_agg import new_figure_manager

    # 创建新的 FigureManager 对象
    manager = new_figure_manager(1000)
    fig = manager.canvas.figure

    # 向图形对象添加子图并绘制
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.arange(12).reshape(3, 4))

    # 手动绘制画布
    manager.canvas.draw()

    # 将图形对象序列化并写入字节流中
    pickle.dump(fig, BytesIO())


def test_polar():
    # 创建一个极坐标子图
    plt.subplot(polar=True)

    # 获取当前图形对象
    fig = plt.gcf()

    # 将图形对象序列化为字节流
    pf = pickle.dumps(fig)

    # 从序列化的字节流中反序列化加载图形对象
    pickle.loads(pf)

    # 绘制图形对象
    plt.draw()


class TransformBlob:
    # 这里需要添加类的详细说明和方法实现
    pass
    def __init__(self):
        # 创建一个身份变换对象
        self.identity = mtransforms.IdentityTransform()
        # 创建另一个身份变换对象
        self.identity2 = mtransforms.IdentityTransform()
        # 强制使用复合变换对象
        self.composite = mtransforms.CompositeGenericTransform(
            self.identity,
            self.identity2)
        # 创建一个 TransformWrapper 对象，检查父到子的链接
        self.wrapper = mtransforms.TransformWrapper(self.composite)
        # 创建另一个复合变换对象，检查子到父的链接
        self.composite2 = mtransforms.CompositeGenericTransform(
            self.wrapper,
            self.identity)
def test_transform():
    # 创建 TransformBlob 对象
    obj = TransformBlob()
    # 将对象序列化为字节流
    pf = pickle.dumps(obj)
    # 删除原始对象的引用
    del obj

    # 从序列化的字节流中反序列化对象
    obj = pickle.loads(pf)
    # 检查 TransformWrapper 的父子链接
    assert obj.wrapper._child == obj.composite
    # 检查 TransformWrapper 的子父链接
    assert [v() for v in obj.wrapper._parents.values()] == [obj.composite2]
    # 检查输入和输出维度是否设置如预期
    assert obj.wrapper.input_dims == obj.composite.input_dims
    assert obj.wrapper.output_dims == obj.composite.output_dims


def test_rrulewrapper():
    # 创建 rrulewrapper 对象
    r = rrulewrapper(2)
    try:
        # 尝试序列化和反序列化对象以测试递归
        pickle.loads(pickle.dumps(r))
    except RecursionError:
        print('rrulewrapper pickling test failed')
        raise


def test_shared():
    # 创建包含共享 X 轴的子图
    fig, axs = plt.subplots(2, sharex=True)
    # 序列化和反序列化图形对象
    fig = pickle.loads(pickle.dumps(fig))
    # 设置第一个子图的 X 轴限制
    fig.axes[0].set_xlim(10, 20)
    # 断言第二个子图的 X 轴限制是否符合预期
    assert fig.axes[1].get_xlim() == (10, 20)


def test_inset_and_secondary():
    # 创建带插图和次要 X 轴的子图
    fig, ax = plt.subplots()
    ax.inset_axes([.1, .1, .3, .3])
    ax.secondary_xaxis("top", functions=(np.square, np.sqrt))
    # 序列化和反序列化图形对象
    pickle.loads(pickle.dumps(fig))


@pytest.mark.parametrize("cmap", cm._colormaps.values())
def test_cmap(cmap):
    # 序列化颜色映射对象
    pickle.dumps(cmap)


def test_unpickle_canvas():
    # 创建 Figure 对象
    fig = mfigure.Figure()
    assert fig.canvas is not None
    # 将 Figure 对象序列化到字节流中
    out = BytesIO()
    pickle.dump(fig, out)
    out.seek(0)
    # 从字节流中反序列化 Figure 对象
    fig2 = pickle.load(out)
    # 断言反序列化后的 Figure 对象仍然有 canvas 属性
    assert fig2.canvas is not None


def test_mpl_toolkits():
    # 创建 HostAxes 对象
    ax = parasite_axes.host_axes([0, 0, 1, 1])
    axes_divider.make_axes_area_auto_adjustable(ax)
    # 序列化和反序列化 HostAxes 对象
    assert type(pickle.loads(pickle.dumps(ax))) == parasite_axes.HostAxes


def test_standard_norm():
    # 断言标准归一化对象的序列化和反序列化后类型是否正确
    assert type(pickle.loads(pickle.dumps(mpl.colors.LogNorm()))) == mpl.colors.LogNorm


def test_dynamic_norm():
    # 创建动态归一化对象
    logit_norm_instance = mpl.colors.make_norm_from_scale(
        mpl.scale.LogitScale, mpl.colors.Normalize)()
    # 断言动态归一化对象的序列化和反序列化后类型是否正确
    assert type(pickle.loads(pickle.dumps(logit_norm_instance))) == type(logit_norm_instance)


def test_vertexselector():
    # 创建包含选择点功能的折线图
    line, = plt.plot([0, 1], picker=True)
    # 序列化和反序列化 VertexSelector 对象
    pickle.loads(pickle.dumps(VertexSelector(line)))


def test_cycler():
    # 创建具有指定颜色循环的子图
    ax = plt.figure().add_subplot()
    ax.set_prop_cycle(c=["c", "m", "y", "k"])
    ax.plot([1, 2])
    # 序列化和反序列化子图对象
    ax = pickle.loads(pickle.dumps(ax))
    # 获取第一条线的颜色并断言其是否为预期值
    l, = ax.plot([3, 4])
    assert l.get_color() == "m"
```