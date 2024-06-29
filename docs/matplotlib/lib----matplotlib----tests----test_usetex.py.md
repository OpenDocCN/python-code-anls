# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_usetex.py`

```py
@pytest.mark.parametrize("fontsize", [8, 10, 12])
def test_minus_no_descent(fontsize):
    # 使用 pytest 的 parametrize 装饰器来定义测试参数化，测试不同的 fontsize 值
    # 这个测试函数用于验证 DviFont._height_depth_of 方法中对减号下降特殊处理的正确性
    mpl.style.use("mpl20")
    # 应用样式表 "mpl20"
    mpl.rcParams['font.size'] = fontsize
    # 设置全局字体大小为测试参数中的 fontsize
    heights = {}
    # 初始化一个空字典 heights
    fig = plt.figure()
    # 创建一个新的 Figure 对象
    # 遍历包含元组的列表，元组是用于设置数学表达式的参数
    for vals in [(1,), (-1,), (-1, 1)]:
        # 清空图形对象，准备绘制新内容
        fig.clear()
        # 对每个元组中的值进行循环，将数学表达式插入图形
        for x in vals:
            fig.text(.5, .5, f"${x}$", usetex=True)
        # 刷新画布，使得新的文本内容可见
        fig.canvas.draw()
        # 下面的代码用于计算非完全空白像素行的数量
        heights[vals] = ((np.array(fig.canvas.buffer_rgba())[..., 0] != 255)
                         .any(axis=1).sum())
    # 使用断言检查所有高度值是否相等
    assert len({*heights.values()}) == 1
@pytest.mark.parametrize('pkg', ['xcolor', 'chemformula'])
def test_usetex_packages(pkg):
    if not _has_tex_package(pkg):
        pytest.skip(f'{pkg} is not available')
    # 设置 matplotlib 的 rc 参数，启用 LaTeX 渲染
    mpl.rcParams['text.usetex'] = True

    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加文本对象
    text = fig.text(0.5, 0.5, "Some text 0123456789")
    # 绘制图形的画布
    fig.canvas.draw()

    # 设置 LaTeX 渲染时的 preamble，加载指定的 LaTeX 宏包
    mpl.rcParams['text.latex.preamble'] = (
        r'\PassOptionsToPackage{dvipsnames}{xcolor}\usepackage{%s}' % pkg)
    # 创建另一个新的图形对象
    fig = plt.figure()
    # 在新图形上添加文本对象
    text2 = fig.text(0.5, 0.5, "Some text 0123456789")
    # 绘制新图形的画布
    fig.canvas.draw()
    # 检查两个文本对象的边界框是否相等
    np.testing.assert_array_equal(text2.get_window_extent(),
                                  text.get_window_extent())


@pytest.mark.parametrize(
    "preamble",
    [r"\usepackage[full]{textcomp}", r"\usepackage{underscore}"],
)
def test_latex_pkg_already_loaded(preamble):
    # 设置 matplotlib 的 rc 参数，加载指定的 LaTeX 宏包
    plt.rcParams["text.latex.preamble"] = preamble
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加文本对象，使用 LaTeX 渲染
    fig.text(.5, .5, "hello, world", usetex=True)
    # 绘制图形的画布
    fig.canvas.draw()


def test_usetex_with_underscore():
    # 设置 matplotlib 的 rc 参数，启用 LaTeX 渲染
    plt.rcParams["text.usetex"] = True
    # 创建一个数据帧
    df = {"a_b": range(5)[::-1], "c": range(5)}
    # 创建一个包含子图的图形对象
    fig, ax = plt.subplots()
    # 在子图上绘制数据
    ax.plot("c", "a_b", data=df)
    # 添加图例
    ax.legend()
    # 在子图上添加文本对象，使用 LaTeX 渲染
    ax.text(0, 0, "foo_bar", usetex=True)
    # 绘制图形的画布
    plt.draw()


@pytest.mark.flaky(reruns=3)  # Tends to hit a TeX cache lock on AppVeyor.
@pytest.mark.parametrize("fmt", ["pdf", "svg"])
def test_missing_psfont(fmt, monkeypatch):
    """如果 TeX 字体缺少 Type-1 等效项，则会引发错误"""
    # 使用 monkeypatch 修改 PsfontsMap 的行为，模拟缺少 Type-1 等效项的情况
    monkeypatch.setattr(
        dviread.PsfontsMap, '__getitem__',
        lambda self, k: dviread.PsFont(
            texname=b'texfont', psname=b'Some Font',
            effects=None, encoding=None, filename=None))
    # 设置 matplotlib 的 rc 参数，启用 LaTeX 渲染
    mpl.rcParams['text.usetex'] = True
    # 创建一个包含子图的图形对象
    fig, ax = plt.subplots()
    # 在子图上添加文本对象
    ax.text(0.5, 0.5, 'hello')
    # 使用 pytest.raises 检查保存图形时是否会引发 ValueError
    with TemporaryFile() as tmpfile, pytest.raises(ValueError):
        fig.savefig(tmpfile, format=fmt)


try:
    _old_gs_version = mpl._get_executable_info('gs').version < parse_version('9.55')
except mpl.ExecutableNotFoundError:
    _old_gs_version = True


@image_comparison(baseline_images=['rotation'], extensions=['eps', 'pdf', 'png', 'svg'],
                  style='mpl20', tol=3.91 if _old_gs_version else 0)
def test_rotation():
    # 设置 matplotlib 的 rc 参数，启用 LaTeX 渲染
    mpl.rcParams['text.usetex'] = True

    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个新的坐标轴到图形中
    ax = fig.add_axes([0, 0, 1, 1])
    # 设置坐标轴的属性：无边框，指定范围和刻度
    ax.set(xlim=[-0.5, 5], xticks=[], ylim=[-0.5, 3], yticks=[], frame_on=False)

    # 定义一些文本和其对应的基线位置
    text = {val: val[0] for val in ['top', 'center', 'bottom', 'left', 'right']}
    text['baseline'] = 'B'
    text['center_baseline'] = 'C'
    # 外层循环，枚举垂直对齐方式
    for i, va in enumerate(['top', 'center', 'bottom', 'baseline', 'center_baseline']):
        # 中层循环，枚举水平对齐方式
        for j, ha in enumerate(['left', 'center', 'right']):
            # 内层循环，枚举角度
            for k, angle in enumerate([0, 90, 180, 270]):
                # 将 k 除以 2，更新 k 的值
                k //= 2
                # 计算点的 x 坐标
                x = i + k / 2
                # 计算点的 y 坐标
                y = j + k / 2
                # 在坐标 (x, y) 处绘制十字形符号，颜色使用 C{k}，大小为 20，边框宽度为 0.5
                ax.plot(x, y, '+', c=f'C{k}', markersize=20, markeredgewidth=0.5)
                # 在坐标 (x, y) 处绘制文本，文本内容为数学模式下的字符串，包含垂直对齐方式、水平对齐方式和角度信息
                ax.text(x, y, f"$\\mathrm{{My {text[ha]}{text[va]} {angle}}}$",
                        rotation=angle, horizontalalignment=ha, verticalalignment=va)
```