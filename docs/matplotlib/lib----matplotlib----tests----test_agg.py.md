# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_agg.py`

```
import io  # 导入io模块，用于处理字节流

import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_array_almost_equal  # 导入NumPy测试模块，用于数组比较
from PIL import Image, TiffTags  # 导入PIL库中的Image和TiffTags模块
import pytest  # 导入pytest测试框架


from matplotlib import (  # 导入matplotlib库中的多个模块
    collections, patheffects, pyplot as plt, transforms as mtransforms,
    rcParams, rc_context)
from matplotlib.backends.backend_agg import RendererAgg  # 导入matplotlib中用于渲染的Agg后端
from matplotlib.figure import Figure  # 导入matplotlib中的Figure类
from matplotlib.image import imread  # 导入matplotlib中的imread函数，用于读取图像文件
from matplotlib.path import Path  # 导入matplotlib中的Path类，用于定义图形路径
from matplotlib.testing.decorators import image_comparison  # 导入matplotlib测试装饰器
from matplotlib.transforms import IdentityTransform  # 导入matplotlib中的IdentityTransform类


def test_repeated_save_with_alpha():
    # We want an image which has a background color of bluish green, with an
    # alpha of 0.25.
    
    fig = Figure([1, 0.4])  # 创建一个Figure对象，指定大小为[1, 0.4]
    fig.set_facecolor((0, 1, 0.4))  # 设置Figure的背景色为绿色调的蓝绿色
    fig.patch.set_alpha(0.25)  # 设置Figure的背景透明度为0.25

    # The target color is fig.patch.get_facecolor()

    buf = io.BytesIO()  # 创建一个字节流缓冲区对象

    fig.savefig(buf,  # 将Figure对象保存为图像文件到缓冲区
                facecolor=fig.get_facecolor(),  # 指定保存图像的背景色为当前Figure的背景色
                edgecolor='none')

    # Save the figure again to check that the
    # colors don't bleed from the previous renderer.
    buf.seek(0)  # 将缓冲区指针移动到起始位置
    fig.savefig(buf,  # 再次将Figure对象保存为图像文件到缓冲区
                facecolor=fig.get_facecolor(),  # 继续指定保存图像的背景色为当前Figure的背景色
                edgecolor='none')

    # Check the first pixel has the desired color & alpha
    # (approx: 0, 1.0, 0.4, 0.25)
    buf.seek(0)  # 将缓冲区指针移动到起始位置
    assert_array_almost_equal(tuple(imread(buf)[0, 0]),  # 读取缓冲区中的图像数据，并检查第一个像素的颜色和透明度
                              (0.0, 1.0, 0.4, 0.250),
                              decimal=3)


def test_large_single_path_collection():
    buff = io.BytesIO()  # 创建一个字节流缓冲区对象

    # Generates a too-large single path in a path collection that
    # would cause a segfault if the draw_markers optimization is
    # applied.
    f, ax = plt.subplots()  # 创建一个包含Figure和Axes对象的图形窗口
    collection = collections.PathCollection(  # 创建一个路径集合对象
        [Path([[-10, 5], [10, 5], [10, -5], [-10, -5], [-10, 5]])])  # 定义一个包含单个路径的路径集合
    ax.add_artist(collection)  # 将路径集合添加到Axes对象中
    ax.set_xlim(10**-3, 1)  # 设置X轴的显示范围
    plt.savefig(buff)  # 将当前图形保存为图像文件到缓冲区


def test_marker_with_nan():
    # This creates a marker with nans in it, which was segfaulting the
    # Agg backend (see #3722)
    fig, ax = plt.subplots(1)  # 创建包含一个Axes对象的图形窗口
    steps = 1000  # 定义步数
    data = np.arange(steps)  # 生成一个数组，包含指定步数的元素
    ax.semilogx(data)  # 绘制半对数X轴上的数据
    ax.fill_between(data, data*0.8, data*1.2)  # 用给定的数据填充两个水平曲线之间的区域
    buf = io.BytesIO()  # 创建一个字节流缓冲区对象
    fig.savefig(buf, format='png')  # 将当前图形保存为PNG格式的图像文件到缓冲区


def test_long_path():
    buff = io.BytesIO()  # 创建一个字节流缓冲区对象
    fig = Figure()  # 创建一个Figure对象
    ax = fig.subplots()  # 在Figure对象上创建一个Axes对象
    points = np.ones(100_000)  # 创建一个包含大量元素的数组
    points[::2] *= -1  # 修改数组中每隔一个元素的值为其相反数
    ax.plot(points)  # 绘制给定数据的折线图
    fig.savefig(buff, format='png')  # 将当前图形保存为PNG格式的图像文件到缓冲区


@image_comparison(['agg_filter.png'], remove_text=True)
def test_agg_filter():
    def smooth1d(x, window_len):
        # copied from https://scipy-cookbook.readthedocs.io/
        s = np.r_[
            2*x[0] - x[window_len:1:-1], x, 2*x[-1] - x[-1:-window_len:-1]]
        w = np.hanning(window_len)
        y = np.convolve(w/w.sum(), s, mode='same')
        return y[window_len-1:-window_len+1]

    def smooth2d(A, sigma=3):
        window_len = max(int(sigma), 3) * 2 + 1
        A = np.apply_along_axis(smooth1d, 0, A, window_len)
        A = np.apply_along_axis(smooth1d, 1, A, window_len)
        return A
    class BaseFilter:
    
        def get_pad(self, dpi):
            return 0
        # process_image 方法应由子类重写
        def process_image(self, padded_src, dpi):
            raise NotImplementedError("Should be overridden by subclasses")
        
        # 将类实例作为可调用对象使用，处理图像并返回结果
        def __call__(self, im, dpi):
            # 调用 get_pad 方法获取填充值
            pad = self.get_pad(dpi)
            # 对输入图像进行填充
            padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)],
                                "constant")
            # 调用 process_image 方法处理填充后的图像
            tgt_image = self.process_image(padded_src, dpi)
            # 返回处理后的目标图像及其相对于原始图像的偏移量
            return tgt_image, -pad, -pad

    class OffsetFilter(BaseFilter):
    
        # 初始化 OffsetFilter 类，设置偏移量
        def __init__(self, offsets=(0, 0)):
            self.offsets = offsets
        
        # 根据 DPI 计算填充值
        def get_pad(self, dpi):
            return int(max(self.offsets) / 72 * dpi)
        
        # 处理图像的方法，根据 x 和 y 偏移量对图像进行滚动
        def process_image(self, padded_src, dpi):
            ox, oy = self.offsets
            a1 = np.roll(padded_src, int(ox / 72 * dpi), axis=1)
            a2 = np.roll(a1, -int(oy / 72 * dpi), axis=0)
            return a2

    class GaussianFilter(BaseFilter):
        """Simple Gaussian filter."""
    
        # 初始化 GaussianFilter 类，设置 sigma、alpha 和 color
        def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
            self.sigma = sigma
            self.alpha = alpha
            self.color = color
        
        # 根据 DPI 计算填充值
        def get_pad(self, dpi):
            return int(self.sigma*3 / 72 * dpi)
        
        # 处理图像的方法，创建一个带有高斯模糊效果的图像
        def process_image(self, padded_src, dpi):
            tgt_image = np.empty_like(padded_src)
            tgt_image[:, :, :3] = self.color
            tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha,
                                          self.sigma / 72 * dpi)
            return tgt_image

    class DropShadowFilter(BaseFilter):
    
        # 初始化 DropShadowFilter 类，设置 sigma、alpha、color 和 offsets
        def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
            self.gauss_filter = GaussianFilter(sigma, alpha, color)
            self.offset_filter = OffsetFilter(offsets)
        
        # 根据 DPI 计算填充值
        def get_pad(self, dpi):
            return max(self.gauss_filter.get_pad(dpi),
                       self.offset_filter.get_pad(dpi))
        
        # 处理图像的方法，首先应用高斯模糊，然后应用偏移滤镜
        def process_image(self, padded_src, dpi):
            t1 = self.gauss_filter.process_image(padded_src, dpi)
            t2 = self.offset_filter.process_image(t1, dpi)
            return t2

    fig, ax = plt.subplots()

    # 绘制线条
    line1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-",
                     mec="b", mfc="w", lw=5, mew=3, ms=10, label="Line 1")
    line2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "ro-",
                     mec="r", mfc="w", lw=5, mew=3, ms=10, label="Line 1")

    # 创建一个 DropShadowFilter 实例，使用 sigma 值为 4
    gauss = DropShadowFilter(4)
    for line in [line1, line2]:
        # 针对给定的两条线，绘制具有略微偏移的阴影。

        # 获取线条的 x 和 y 数据
        xx = line.get_xdata()
        yy = line.get_ydata()

        # 在图形上绘制阴影线条
        shadow, = ax.plot(xx, yy)
        # 从原始线条更新阴影线条的属性
        shadow.update_from(line)

        # 创建偏移变换
        transform = mtransforms.offset_copy(line.get_transform(), ax.figure,
                                            x=4.0, y=-6.0, units='points')
        shadow.set_transform(transform)

        # 调整阴影线条的绘制顺序，使其在原始线条下方绘制
        shadow.set_zorder(line.get_zorder() - 0.5)
        # 设置阴影线条的聚合过滤器为高斯
        shadow.set_agg_filter(gauss)
        # 设置阴影线条为光栅化，以支持混合模式渲染器
        shadow.set_rasterized(True)

    # 设置 x 和 y 轴的显示范围为 0 到 1
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)

    # 隐藏 x 和 y 轴
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
def test_too_large_image():
    # 创建一个大尺寸的图形对象
    fig = plt.figure(figsize=(300, 1000))
    # 创建一个字节流缓冲区
    buff = io.BytesIO()
    # 断言保存超出限制的图形会引发 ValueError 异常
    with pytest.raises(ValueError):
        fig.savefig(buff)


def test_chunksize():
    x = range(200)

    # 测试不使用 chunksize 的情况
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x))
    fig.canvas.draw()

    # 测试使用 chunksize 的情况
    fig, ax = plt.subplots()
    rcParams['agg.path.chunksize'] = 105
    ax.plot(x, np.sin(x))
    fig.canvas.draw()


@pytest.mark.backend('Agg')
def test_jpeg_dpi():
    # 检查 JPG 文件中 DPI 设置是否正确
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi=200)
    im = Image.open(buf)
    assert im.info['dpi'] == (200, 200)


def test_pil_kwargs_png():
    from PIL.PngImagePlugin import PngInfo
    buf = io.BytesIO()
    pnginfo = PngInfo()
    pnginfo.add_text("Software", "test")
    # 保存 PNG 格式图像，传递自定义 PIL 参数
    plt.figure().savefig(buf, format="png", pil_kwargs={"pnginfo": pnginfo})
    im = Image.open(buf)
    assert im.info["Software"] == "test"


def test_pil_kwargs_tiff():
    buf = io.BytesIO()
    pil_kwargs = {"description": "test image"}
    # 保存 TIFF 格式图像，传递自定义 PIL 参数
    plt.figure().savefig(buf, format="tiff", pil_kwargs=pil_kwargs)
    im = Image.open(buf)
    tags = {TiffTags.TAGS_V2[k].name: v for k, v in im.tag_v2.items()}
    assert tags["ImageDescription"] == "test image"


def test_pil_kwargs_webp():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf_small = io.BytesIO()
    pil_kwargs_low = {"quality": 1}
    # 保存低质量的 WebP 格式图像，传递自定义 PIL 参数
    plt.savefig(buf_small, format="webp", pil_kwargs=pil_kwargs_low)
    assert len(pil_kwargs_low) == 1
    buf_large = io.BytesIO()
    pil_kwargs_high = {"quality": 100}
    # 保存高质量的 WebP 格式图像，传递自定义 PIL 参数
    plt.savefig(buf_large, format="webp", pil_kwargs=pil_kwargs_high)
    assert len(pil_kwargs_high) == 1
    # 检查高质量图像文件大小比低质量图像大
    assert buf_large.getbuffer().nbytes > buf_small.getbuffer().nbytes


def test_webp_alpha():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    # 保存带有透明通道的 WebP 格式图像
    plt.savefig(buf, format="webp", transparent=True)
    im = Image.open(buf)
    assert im.mode == "RGBA"


def test_draw_path_collection_error_handling():
    fig, ax = plt.subplots()
    ax.scatter([1], [1]).set_paths(Path([(0, 1), (2, 3)]))
    # 断言绘制错误路径集合会引发 TypeError 异常
    with pytest.raises(TypeError):
        fig.canvas.draw()


def test_chunksize_fails():
    # 注意：这个测试函数覆盖了多个独立的测试场景，因为每个场景需要约2GB内存，
    #       我们不希望并行测试执行器意外同时运行多个这样的测试。

    N = 100_000
    dpi = 500
    w = 5*dpi
    h = 6*dpi

    # 创建一个跨越整个 w-h 矩形的 Path
    x = np.linspace(0, w, N)
    y = np.ones(N) * h
    y[::2] = 0
    path = Path(np.vstack((x, y)).T)
    # 实际上禁用路径简化（但保持其“打开”状态）
    path.simplify_threshold = 0

    # 设置最小的 GraphicsContext 来绘制 Path
    ra = RendererAgg(w, h, dpi)
    gc = ra.new_gc()
    gc.set_linewidth(1)
    gc.set_foreground('r')
    # 设置填充图案为斜线'/'
    gc.set_hatch('/')
    # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
    with pytest.raises(OverflowError, match='cannot split hatched path'):
        # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径和变换
        ra.draw_path(gc, path, IdentityTransform())
    # 清除填充图案设置
    gc.set_hatch(None)

    # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
    with pytest.raises(OverflowError, match='cannot split filled path'):
        # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径、变换和填充颜色(红色)
        ra.draw_path(gc, path, IdentityTransform(), (1, 0, 0))

    # 设置 agg.path.chunksize 参数为 0，禁用路径分块处理
    with rc_context({'agg.path.chunksize': 0}):
        # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
        with pytest.raises(OverflowError, match='Please set'):
            # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径和变换
            ra.draw_path(gc, path, IdentityTransform())

    # 设置 agg.path.chunksize 参数为 1,000,000，确保路径分块大小足够大
    with rc_context({'agg.path.chunksize': 1_000_000}):
        # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
        with pytest.raises(OverflowError, match='Please reduce'):
            # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径和变换
            ra.draw_path(gc, path, IdentityTransform())

    # 设置 agg.path.chunksize 参数为 90,000，足够小使得尝试进行路径分块，但仍然会失败
    with rc_context({'agg.path.chunksize': 90_000}):
        # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
        with pytest.raises(OverflowError, match='Please reduce'):
            # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径和变换
            ra.draw_path(gc, path, IdentityTransform())

    # 设置路径的 should_simplify 属性为 False
    path.should_simplify = False
    # 使用 pytest 检测是否会抛出 OverflowError，并匹配特定错误信息
    with pytest.raises(OverflowError, match="should_simplify is False"):
        # 调用 ra.draw_path 方法绘制路径，使用给定的绘图上下文、路径和变换
        ra.draw_path(gc, path, IdentityTransform())
# 定义一个函数用于测试非元组类型的 rgbaFace 参数
def test_non_tuple_rgbaface():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 向图形对象添加一个带有 3D 投影的子图
    fig.add_subplot(projection="3d").scatter(
        [0, 1, 2], [0, 1, 2], path_effects=[patheffects.Stroke(linewidth=4)])
    # 绘制图形对象的画布
    fig.canvas.draw()
```