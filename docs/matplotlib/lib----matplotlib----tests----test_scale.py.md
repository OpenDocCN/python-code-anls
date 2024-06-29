# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_scale.py`

```py
# 导入copy模块，用于深复制对象
import copy

# 导入matplotlib的pyplot模块，并引入所需的比例尺和转换器
import matplotlib.pyplot as plt
from matplotlib.scale import (
    AsinhScale, AsinhTransform,  # 引入反双曲正弦比例尺及其转换器
    LogTransform, InvertedLogTransform,  # 引入对数比例尺及其反转换器
    SymmetricalLogTransform)  # 引入对称对数比例尺及其转换器
import matplotlib.scale as mscale  # 引入matplotlib的比例尺模块
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation  # 引入反双曲正弦定位器和科学记数对数格式化器
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 引入用于检查图形是否相等的装饰器和图像对比的装饰器

# 导入numpy模块及其测试工具
import numpy as np
from numpy.testing import assert_allclose  # 引入用于全面比较数组是否接近的函数
import io  # 导入io模块
import pytest  # 导入pytest测试框架


@check_figures_equal()
def test_log_scales(fig_test, fig_ref):
    # 在测试图中添加子图，设置y轴为对数尺度，x轴为对称对数尺度
    ax_test = fig_test.add_subplot(122, yscale='log', xscale='symlog')
    ax_test.axvline(24.1)  # 在测试子图中添加垂直于x轴的线
    ax_test.axhline(24.1)  # 在测试子图中添加水平于y轴的线
    xlim = ax_test.get_xlim()  # 获取测试子图的x轴限制范围
    ylim = ax_test.get_ylim()  # 获取测试子图的y轴限制范围
    ax_ref = fig_ref.add_subplot(122, yscale='log', xscale='symlog')
    ax_ref.set(xlim=xlim, ylim=ylim)  # 在参考图中设置相同的x和y轴限制
    ax_ref.plot([24.1, 24.1], ylim, 'b')  # 在参考图中绘制垂直于x轴的蓝色线
    ax_ref.plot(xlim, [24.1, 24.1], 'b')  # 在参考图中绘制水平于y轴的蓝色线


def test_symlog_mask_nan():
    # 使用转换器来验证正向和反向转换是否正常工作，并且它们是否遵守nan和/或掩码。
    slt = SymmetricalLogTransform(10, 2, 1)  # 创建对称对数转换对象
    slti = slt.inverted()  # 获取其反向转换对象

    x = np.arange(-1.5, 5, 0.5)  # 创建一个numpy数组
    out = slti.transform_non_affine(slt.transform_non_affine(x))  # 执行转换的往返验证
    assert_allclose(out, x)  # 检查转换结果是否接近原始输入
    assert type(out) is type(x)  # 检查输出类型是否与输入类型相同

    x[4] = np.nan  # 将数组中的第五个元素设置为NaN
    out = slti.transform_non_affine(slt.transform_non_affine(x))  # 再次执行转换的往返验证
    assert_allclose(out, x)  # 检查转换结果是否接近包含NaN的输入
    assert type(out) is type(x)  # 检查输出类型是否与输入类型相同

    x = np.ma.array(x)  # 创建一个带掩码的numpy数组
    out = slti.transform_non_affine(slt.transform_non_affine(x))  # 再次执行转换的往返验证
    assert_allclose(out, x)  # 检查转换结果是否接近带掩码的输入
    assert type(out) is type(x)  # 检查输出类型是否与输入类型相同

    x[3] = np.ma.masked  # 将数组中的第四个元素掩码化
    out = slti.transform_non_affine(slt.transform_non_affine(x))  # 再次执行转换的往返验证
    assert_allclose(out, x)  # 检查转换结果是否接近掩码化的输入
    assert type(out) is type(x)  # 检查输出类型是否与输入类型相同


@image_comparison(['logit_scales.png'], remove_text=True)
def test_logit_scales():
    fig, ax = plt.subplots()

    # 典型的logit消光曲线
    x = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.997, 0.999])
    y = 1.0 / x

    ax.plot(x, y)  # 在图中绘制消光曲线
    ax.set_xscale('logit')  # 设置x轴为logit比例尺
    ax.grid(True)  # 打开网格显示
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())  # 获取紧凑的边界框
    assert np.isfinite(bbox.x0)  # 检查边界框的x0坐标是否有限
    assert np.isfinite(bbox.y0)  # 检查边界框的y0坐标是否有限


def test_log_scatter():
    """Issue #1799"""
    fig, ax = plt.subplots(1)

    x = np.arange(10)
    y = np.arange(10) - 1

    ax.scatter(x, y)  # 在子图中绘制散点图

    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')  # 将图保存为PDF格式

    buf = io.BytesIO()
    fig.savefig(buf, format='eps')  # 将图保存为EPS格式

    buf = io.BytesIO()
    fig.savefig(buf, format='svg')  # 将图保存为SVG格式


def test_logscale_subs():
    fig, ax = plt.subplots()
    ax.set_yscale('log', subs=np.array([2, 3, 4]))  # 设置y轴为对数尺度，并指定辅助刻度为2、3、4
    fig.canvas.draw()  # 强制绘制图形


@image_comparison(['logscale_mask.png'], remove_text=True)
def test_logscale_mask():
    # 检查在对数尺度上零值是否被正确屏蔽。
    # 参见GitHub问题8045
    xs = np.linspace(0, 50, 1001)  # 创建一个线性间隔的数组

    fig, ax = plt.subplots()
    ax.plot(np.exp(-xs**2))  # 在子图中绘制指数函数的负二次方
    fig.canvas.draw()  # 强制绘制图形
    # 设置 Y 轴的比例为对数（log）尺度
    ax.set(yscale="log")
def test_extra_kwargs_raise():
    # 创建一个包含图形和轴的 subplot
    fig, ax = plt.subplots()

    # 针对不同的缩放方式进行循环测试
    for scale in ['linear', 'log', 'symlog']:
        # 使用 pytest 检查设置 y 轴缩放时传入额外参数会引发 TypeError 异常
        with pytest.raises(TypeError):
            ax.set_yscale(scale, foo='mask')


def test_logscale_invert_transform():
    # 创建一个包含图形和轴的 subplot
    fig, ax = plt.subplots()

    # 设置 y 轴为对数缩放
    ax.set_yscale('log')

    # 获取数据到轴的转换
    tform = (ax.transAxes + ax.transData.inverted()).inverted()

    # 直接测试对数转换的反转
    inverted_transform = LogTransform(base=2).inverted()
    assert isinstance(inverted_transform, InvertedLogTransform)
    assert inverted_transform.base == 2


def test_logscale_transform_repr():
    # 创建一个包含图形和轴的 subplot
    fig, ax = plt.subplots()

    # 设置 y 轴为对数缩放
    ax.set_yscale('log')

    # 打印数据转换的字符串表示
    repr(ax.transData)

    # 打印对数变换的字符串表示，以基数 10，非正数值设为 'clip'
    repr(LogTransform(10, nonpositive='clip'))


@image_comparison(['logscale_nonpos_values.png'],
                  remove_text=True, tol=0.02, style='mpl20')
def test_logscale_nonpos_values():
    # 设置随机数种子
    np.random.seed(19680801)

    # 生成随机数据
    xs = np.random.normal(size=int(1e3))

    # 创建一个 2x2 的图形，每个子图包含一个轴对象
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # 在第一个子图上绘制直方图，设置 y 轴为对数缩放
    ax1.hist(xs, range=(-5, 5), bins=10)
    ax1.set_yscale('log')

    # 在第二个子图上绘制直方图，设置 y 轴为对数缩放，非正数值设为 'mask'
    ax2.hist(xs, range=(-5, 5), bins=10)
    ax2.set_yscale('log', nonpositive='mask')

    # 在第三个子图上绘制填充区域图，设置 y 轴为对数缩放
    xdata = np.arange(0, 10, 0.01)
    ydata = np.exp(-xdata)
    edata = 0.2*(10-xdata)*np.cos(5*xdata)*np.exp(-xdata)
    ax3.fill_between(xdata, ydata - edata, ydata + edata)
    ax3.set_yscale('log')

    # 在第四个子图上绘制误差线图，设置 x 和 y 轴为对数缩放
    x = np.logspace(-1, 1)
    y = x ** 3
    yerr = x**2
    ax4.errorbar(x, y, yerr=yerr)
    ax4.set_yscale('log')
    ax4.set_xscale('log')


def test_invalid_log_lims():
    # 检查无效的对数缩放限制是否被忽略
    fig, ax = plt.subplots()
    ax.scatter(range(0, 4), range(0, 4))

    # 设置 x 轴为对数缩放
    ax.set_xscale('log')
    original_xlim = ax.get_xlim()

    # 测试设置无效的左限制是否会引发 UserWarning
    with pytest.warns(UserWarning):
        ax.set_xlim(left=0)
    assert ax.get_xlim() == original_xlim

    # 测试设置无效的右限制是否会引发 UserWarning
    with pytest.warns(UserWarning):
        ax.set_xlim(right=-1)
    assert ax.get_xlim() == original_xlim

    # 设置 y 轴为对数缩放
    ax.set_yscale('log')
    original_ylim = ax.get_ylim()

    # 测试设置无效的下限制是否会引发 UserWarning
    with pytest.warns(UserWarning):
        ax.set_ylim(bottom=0)
    assert ax.get_ylim() == original_ylim

    # 测试设置无效的上限制是否会引发 UserWarning
    with pytest.warns(UserWarning):
        ax.set_ylim(top=-1)
    assert ax.get_ylim() == original_ylim


@image_comparison(['function_scales.png'], remove_text=True, style='mpl20')
def test_function_scale():
    # 定义一个反向函数
    def inverse(x):
        return x**2

    # 定义一个正向函数
    def forward(x):
        return x**(1/2)

    # 创建一个包含图形和轴的 subplot
    fig, ax = plt.subplots()

    # 绘制 x 和 y 的图形
    x = np.arange(1, 1000)
    ax.plot(x, x)

    # 设置 x 轴为自定义函数缩放，指定正向和反向函数
    ax.set_xscale('function', functions=(forward, inverse))
    ax.set_xlim(1, 1000)


def test_pass_scale():
    # 测试传递缩放对象是否有效
    fig, ax = plt.subplots()

    # 创建一个对数缩放的对象并设置为 x 轴的缩放
    scale = mscale.LogScale(axis=None)
    ax.set_xscale(scale)

    # 创建一个对数缩放的对象并设置为 y 轴的缩放
    scale = mscale.LogScale(axis=None)
    ax.set_yscale(scale)

    # 断言 x 轴和 y 轴的缩放类型为对数缩放
    assert ax.xaxis.get_scale() == 'log'
    assert ax.yaxis.get_scale() == 'log'


def test_scale_deepcopy():
    # 创建一个基于对数的缩放对象，并指定为 x 轴的缩放
    sc = mscale.LogScale(axis='x', base=10)
    # 使用深拷贝创建变量 sc2，确保两个对象在内存中完全独立
    sc2 = copy.deepcopy(sc)
    # 断言两个对象的变换转换为字符串后相同，用于验证深拷贝是否成功复制了对象的状态
    assert str(sc.get_transform()) == str(sc2.get_transform())
    # 断言两个对象的 _transform 属性不是同一个对象，以验证深拷贝确实创建了对象及其属性的完全副本
    assert sc._transform is not sc2._transform
class TestAsinhScale:
    def test_transforms(self):
        a0 = 17.0  # 设置初始值 a0 为 17.0
        a = np.linspace(-50, 50, 100)  # 创建一个包含 100 个元素的数组 a，范围从 -50 到 50

        forward = AsinhTransform(a0)  # 使用给定的 a0 创建 AsinhTransform 对象
        inverse = forward.inverted()  # 创建 forward 的反转对象 inverse
        invinv = inverse.inverted()  # 创建 inverse 的反转对象 invinv

        a_forward = forward.transform_non_affine(a)  # 对数组 a 进行非仿射变换，并保存结果到 a_forward
        a_inverted = inverse.transform_non_affine(a_forward)  # 对 a_forward 进行反向非仿射变换，保存到 a_inverted
        assert_allclose(a_inverted, a)  # 断言 a_inverted 应该与原始数组 a 很接近

        a_invinv = invinv.transform_non_affine(a)  # 对数组 a 进行 invinv 的非仿射变换
        assert_allclose(a_invinv, a0 * np.arcsinh(a / a0))  # 断言 a_invinv 应该与 a0 * arcsinh(a / a0) 很接近

    def test_init(self):
        fig, ax = plt.subplots()  # 创建一个图形对象和一个坐标轴对象

        s = AsinhScale(axis=None, linear_width=23.0)  # 创建一个 AsinhScale 实例 s，设置 linear_width 为 23.0
        assert s.linear_width == 23  # 断言 s 的 linear_width 属性应该为 23
        assert s._base == 10  # 断言 s 的 _base 属性应该为 10
        assert s._subs == (2, 5)  # 断言 s 的 _subs 属性应该为元组 (2, 5)

        tx = s.get_transform()  # 获得 s 的变换对象 tx
        assert isinstance(tx, AsinhTransform)  # 断言 tx 应该是 AsinhTransform 的实例
        assert tx.linear_width == s.linear_width  # 断言 tx 的 linear_width 属性与 s 的相同

    def test_base_init(self):
        fig, ax = plt.subplots()  # 创建一个图形对象和一个坐标轴对象

        s3 = AsinhScale(axis=None, base=3)  # 创建一个 AsinhScale 实例 s3，设置 base 为 3
        assert s3._base == 3  # 断言 s3 的 _base 属性应该为 3
        assert s3._subs == (2,)  # 断言 s3 的 _subs 属性应该为元组 (2,)

        s7 = AsinhScale(axis=None, base=7, subs=(2, 4))  # 创建一个 AsinhScale 实例 s7，设置 base 为 7，subs 为 (2, 4)
        assert s7._base == 7  # 断言 s7 的 _base 属性应该为 7
        assert s7._subs == (2, 4)  # 断言 s7 的 _subs 属性应该为元组 (2, 4)

    def test_fmtloc(self):
        class DummyAxis:
            def __init__(self):
                self.fields = {}

            def set(self, **kwargs):
                self.fields.update(**kwargs)

            def set_major_formatter(self, f):
                self.fields['major_formatter'] = f

        ax0 = DummyAxis()  # 创建一个虚拟的坐标轴对象 ax0
        s0 = AsinhScale(axis=ax0, base=0)  # 创建一个 AsinhScale 实例 s0，设置 base 为 0，关联到 ax0
        s0.set_default_locators_and_formatters(ax0)  # 设置 ax0 的默认定位器和格式化器
        assert isinstance(ax0.fields['major_locator'], AsinhLocator)  # 断言 ax0 的 major_locator 是 AsinhLocator 的实例
        assert isinstance(ax0.fields['major_formatter'], str)  # 断言 ax0 的 major_formatter 是字符串类型

        ax5 = DummyAxis()  # 创建另一个虚拟的坐标轴对象 ax5
        s7 = AsinhScale(axis=ax5, base=5)  # 创建一个 AsinhScale 实例 s7，设置 base 为 5，关联到 ax5
        s7.set_default_locators_and_formatters(ax5)  # 设置 ax5 的默认定位器和格式化器
        assert isinstance(ax5.fields['major_locator'], AsinhLocator)  # 断言 ax5 的 major_locator 是 AsinhLocator 的实例
        assert isinstance(ax5.fields['major_formatter'],
                          LogFormatterSciNotation)  # 断言 ax5 的 major_formatter 是 LogFormatterSciNotation 的实例

    def test_bad_scale(self):
        fig, ax = plt.subplots()  # 创建一个图形对象和一个坐标轴对象

        with pytest.raises(ValueError):  # 使用 pytest 断言抛出 ValueError 异常
            AsinhScale(axis=None, linear_width=0)  # 尝试创建 linear_width 为 0 的 AsinhScale 实例
        with pytest.raises(ValueError):  # 使用 pytest 断言抛出 ValueError 异常
            AsinhScale(axis=None, linear_width=-1)  # 尝试创建 linear_width 为 -1 的 AsinhScale 实例
        s0 = AsinhScale(axis=None, )  # 创建一个默认参数的 AsinhScale 实例 s0
        s1 = AsinhScale(axis=None, linear_width=3.0)  # 创建一个 linear_width 为 3.0 的 AsinhScale 实例
```