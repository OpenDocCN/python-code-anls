# `D:\src\scipysrc\matplotlib\galleries\examples\misc\demo_agg_filter.py`

```
"""
==========
AGG filter
==========

Most pixel-based backends in Matplotlib use `Anti-Grain Geometry (AGG)`_ for
rendering. You can modify the rendering of Artists by applying a filter via
`.Artist.set_agg_filter`.

.. _Anti-Grain Geometry (AGG): http://agg.sourceforge.net/antigrain.com
"""

# 导入 matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 导入需要使用的模块和类
from matplotlib.artist import Artist
import matplotlib.cm as cm
from matplotlib.colors import LightSource
import matplotlib.transforms as mtransforms


# 定义一维平滑函数
def smooth1d(x, window_len):
    # 从 https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html 复制而来
    s = np.r_[2*x[0] - x[window_len:1:-1], x, 2*x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


# 定义二维平滑函数
def smooth2d(A, sigma=3):
    window_len = max(int(sigma), 3) * 2 + 1
    A = np.apply_along_axis(smooth1d, 0, A, window_len)
    A = np.apply_along_axis(smooth1d, 1, A, window_len)
    return A


# 定义基础滤镜类
class BaseFilter:

    def get_pad(self, dpi):
        return 0

    def process_image(self, padded_src, dpi):
        raise NotImplementedError("Should be overridden by subclasses")

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], "constant")
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


# 定义偏移滤镜类，继承自 BaseFilter
class OffsetFilter(BaseFilter):

    def __init__(self, offsets=(0, 0)):
        self.offsets = offsets

    def get_pad(self, dpi):
        return int(max(self.offsets) / 72 * dpi)

    def process_image(self, padded_src, dpi):
        ox, oy = self.offsets
        a1 = np.roll(padded_src, int(ox / 72 * dpi), axis=1)
        a2 = np.roll(a1, -int(oy / 72 * dpi), axis=0)
        return a2


# 定义高斯滤镜类，继承自 BaseFilter
class GaussianFilter(BaseFilter):
    """Simple Gaussian filter."""

    def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
        self.sigma = sigma
        self.alpha = alpha
        self.color = color

    def get_pad(self, dpi):
        return int(self.sigma*3 / 72 * dpi)

    def process_image(self, padded_src, dpi):
        tgt_image = np.empty_like(padded_src)
        tgt_image[:, :, :3] = self.color
        tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha,
                                      self.sigma / 72 * dpi)
        return tgt_image


# 定义阴影滤镜类，继承自 BaseFilter
class DropShadowFilter(BaseFilter):

    def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
        self.gauss_filter = GaussianFilter(sigma, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi),
                   self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        t2 = self.offset_filter.process_image(t1, dpi)
        return t2


# 定义光滤镜类，继承自 BaseFilter
class LightFilter(BaseFilter):
    pass
    """Apply LightSource filter"""

    # 初始化方法，设置高斯滤波器和光源对象，并接收参数 sigma 和 fraction
    def __init__(self, sigma, fraction=1):
        """
        Parameters
        ----------
        sigma : float
            高斯滤波器的标准差
        fraction: number, default: 1
            增加或减少山影的对比度。
            参见 `matplotlib.colors.LightSource`
        """
        # 创建高斯滤波器对象，使用给定的 sigma 和默认的 alpha=1 参数
        self.gauss_filter = GaussianFilter(sigma, alpha=1)
        # 创建光源对象，用于处理影像的光照效果
        self.light_source = LightSource()
        # 设置 fraction 属性，用于调整光照的对比度
        self.fraction = fraction

    # 获取填充后的影像，使用高斯滤波器对象的方法
    def get_pad(self, dpi):
        return self.gauss_filter.get_pad(dpi)

    # 处理影像的方法，返回处理后的 RGBA 影像
    def process_image(self, padded_src, dpi):
        # 使用高斯滤波器处理填充后的源影像
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        # 获取处理后影像的高程值
        elevation = t1[:, :, 3]
        # 获取填充后影像的 RGB 通道
        rgb = padded_src[:, :, :3]
        # 获取填充后影像的 Alpha 通道
        alpha = padded_src[:, :, 3:]
        # 使用光源对象为 RGB 影像添加光照效果，根据高程和 fraction 参数调整对比度，采用叠加混合模式
        rgb2 = self.light_source.shade_rgb(rgb, elevation,
                                           fraction=self.fraction,
                                           blend_mode="overlay")
        # 将处理后的 RGB 影像和 Alpha 通道合并，返回完整的 RGBA 影像
        return np.concatenate([rgb2, alpha], -1)
class GrowFilter(BaseFilter):
    """Enlarge the area."""

    def __init__(self, pixels, color=(1, 1, 1)):
        # 初始化函数，接受像素和颜色参数
        self.pixels = pixels  # 设定像素扩展值
        self.color = color    # 设定颜色值

    def __call__(self, im, dpi):
        # 在调用对象时执行的方法，用于处理图像和分辨率参数
        alpha = np.pad(im[..., 3], self.pixels, "constant")  # 扩展 alpha 通道
        alpha2 = np.clip(smooth2d(alpha, self.pixels / 72 * dpi) * 5, 0, 1)  # 平滑处理 alpha 通道并进行剪切
        new_im = np.empty((*alpha2.shape, 4))  # 创建新的图像数组
        new_im[:, :, :3] = self.color  # 设定新图像的 RGB 颜色通道
        new_im[:, :, 3] = alpha2  # 设定新图像的 alpha 通道
        offsetx, offsety = -self.pixels, -self.pixels  # 设定偏移值
        return new_im, offsetx, offsety  # 返回处理后的图像和偏移值


class FilteredArtistList(Artist):
    """A simple container to filter multiple artists at once."""

    def __init__(self, artist_list, filter):
        # 初始化函数，接受艺术家列表和过滤器
        super().__init__()  # 调用父类初始化方法
        self._artist_list = artist_list  # 存储艺术家列表
        self._filter = filter  # 存储过滤器对象

    def draw(self, renderer):
        # 绘制函数，接受渲染器参数
        renderer.start_rasterizing()  # 开始光栅化
        renderer.start_filter()  # 开始过滤
        for a in self._artist_list:
            a.draw(renderer)  # 绘制每个艺术家对象
        renderer.stop_filter(self._filter)  # 停止过滤器
        renderer.stop_rasterizing()  # 停止光栅化


def filtered_text(ax):
    # mostly copied from contour_demo.py

    # prepare image
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)  # 生成 X 轴坐标数组
    y = np.arange(-2.0, 2.0, delta)  # 生成 Y 轴坐标数组
    X, Y = np.meshgrid(x, y)  # 生成网格坐标
    Z1 = np.exp(-X**2 - Y**2)  # 计算第一个高斯函数
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  # 计算第二个高斯函数
    Z = (Z1 - Z2) * 2  # 生成合成的 Z 值

    # draw
    ax.imshow(Z, interpolation='bilinear', origin='lower',
              cmap=cm.gray, extent=(-3, 3, -2, 2), aspect='auto')  # 绘制图像
    levels = np.arange(-1.2, 1.6, 0.2)  # 设定等高线级别
    CS = ax.contour(Z, levels,
                    origin='lower',
                    linewidths=2,
                    extent=(-3, 3, -2, 2))  # 绘制等高线

    # contour label
    cl = ax.clabel(CS, levels[1::2],  # 标记每第二个等高线
                   inline=True,
                   fmt='%1.1f',
                   fontsize=11)  # 设定标签格式和字体大小

    # change clabel color to black
    from matplotlib.patheffects import Normal
    for t in cl:
        t.set_color("k")  # 设置标签颜色为黑色
        # to force TextPath (i.e., same font in all backends)
        t.set_path_effects([Normal()])  # 设置路径效果为正常

    # Add white glows to improve visibility of labels.
    white_glows = FilteredArtistList(cl, GrowFilter(3))  # 创建白色辉光效果对象
    ax.add_artist(white_glows)  # 将白色辉光效果对象添加到图表中
    white_glows.set_zorder(cl[0].get_zorder() - 0.1)  # 设置辉光效果对象的层次顺序

    ax.xaxis.set_visible(False)  # 隐藏 X 轴
    ax.yaxis.set_visible(False)  # 隐藏 Y 轴


def drop_shadow_line(ax):
    # copied from examples/misc/svg_filter_line.py

    # draw lines
    l1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-")  # 绘制蓝色线条
    l2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "ro-")  # 绘制红色线条

    gauss = DropShadowFilter(4)  # 创建高斯模糊滤镜对象
    # 对于每个线对象 l1 和 l2 执行以下操作：

    # 获取线对象的 x 数据和 y 数据
    xx = l.get_xdata()
    yy = l.get_ydata()

    # 在图表 ax 上绘制阴影，使用相同的数据稍微偏移
    shadow, = ax.plot(xx, yy)

    # 从原始线对象 l 复制属性到阴影对象 shadow
    shadow.update_from(l)

    # 创建偏移后的变换对象
    transform = mtransforms.offset_copy(l.get_transform(), ax.figure,
                                        x=4.0, y=-6.0, units='points')

    # 设置阴影对象的变换
    shadow.set_transform(transform)

    # 调整阴影线的 zorder，使其绘制在原始线的下方
    shadow.set_zorder(l.get_zorder() - 0.5)

    # 设置阴影对象的汇聚过滤器为 gauss
    shadow.set_agg_filter(gauss)

    # 将阴影对象设为矢量化，以支持混合模式渲染器
    shadow.set_rasterized(True)

# 设置图表 ax 的 x 和 y 轴范围为 0 到 1
ax.set_xlim(0., 1.)
ax.set_ylim(0., 1.)

# 隐藏图表 ax 的 x 和 y 轴
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
def drop_shadow_patches(ax):
    # 从 barchart_demo.py 复制的函数，用于在柱状图上添加阴影效果

    N = 5  # 组数
    group1_means = [20, 35, 30, 35, 27]  # 第一组的平均值列表

    ind = np.arange(N)  # 柱状图组的 x 坐标位置
    width = 0.35  # 柱状图的宽度

    # 在 ax 上绘制第一组柱状图，设置颜色为红色，边缘颜色为白色，边缘线宽度为2
    rects1 = ax.bar(ind, group1_means, width, color='r', ec="w", lw=2)

    group2_means = [25, 32, 34, 20, 25]  # 第二组的平均值列表
    # 在 ax 上绘制第二组柱状图，位置偏移 width+0.1，设置颜色为黄色，边缘颜色为白色，边缘线宽度为2
    rects2 = ax.bar(ind + width + 0.1, group2_means, width,
                    color='y', ec="w", lw=2)

    # 创建一个 DropShadowFilter 对象，添加到包含 rects1 和 rects2 的 FilteredArtistList 中
    drop = DropShadowFilter(5, offsets=(1, 1))
    shadow = FilteredArtistList(rects1 + rects2, drop)
    ax.add_artist(shadow)
    # 设置阴影的层级，使其低于 rects1 的柱状图
    shadow.set_zorder(rects1[0].get_zorder() - 0.1)

    ax.set_ylim(0, 40)  # 设置 y 轴的范围为 0 到 40

    ax.xaxis.set_visible(False)  # 隐藏 x 轴刻度和标签
    ax.yaxis.set_visible(False)  # 隐藏 y 轴刻度和标签


def light_filter_pie(ax):
    fracs = [15, 30, 45, 10]  # 饼图各部分的比例
    explode = (0.1, 0.2, 0.1, 0.1)  # 各部分的爆炸程度

    pies = ax.pie(fracs, explode=explode)  # 在 ax 上绘制饼图，根据 explode 参数进行爆炸

    light_filter = LightFilter(9)  # 创建一个 LightFilter 对象，参数为 9
    # 为每个饼图部分设置聚合过滤器为 light_filter
    for p in pies[0]:
        p.set_agg_filter(light_filter)
        p.set_rasterized(True)  # 支持混合模式渲染的光栅化设置
        p.set(ec="none", lw=2)  # 设置饼图部分的边缘颜色为无，线宽为2

    # 创建一个 DropShadowFilter 对象，添加到 pies[0] 中的 FilteredArtistList 中
    gauss = DropShadowFilter(9, offsets=(3, -4), alpha=0.7)
    shadow = FilteredArtistList(pies[0], gauss)
    ax.add_artist(shadow)
    # 设置阴影的层级，使其低于 pies[0] 的第一个饼图部分
    shadow.set_zorder(pies[0][0].get_zorder() - 0.1)


if __name__ == "__main__":
    # 在主程序中创建一个 2x2 的图表布局
    fig, axs = plt.subplots(2, 2)

    filtered_text(axs[0, 0])  # 调用 filtered_text 函数，处理 axs 的第一行第一列
    drop_shadow_line(axs[0, 1])  # 调用 drop_shadow_line 函数，处理 axs 的第一行第二列
    drop_shadow_patches(axs[1, 0])  # 调用 drop_shadow_patches 函数，处理 axs 的第二行第一列
    light_filter_pie(axs[1, 1])  # 调用 light_filter_pie 函数，处理 axs 的第二行第二列
    axs[1, 1].set_frame_on(True)  # 在 axs 的第二行第二列的子图上设置框架可见

    plt.show()  # 显示图表
```