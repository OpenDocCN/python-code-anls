# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_image.py`

```
    "img_size, fig_size, interpolation",
    [
        ((100, 100), (1, 1), 'nearest'),
        ((100, 100), (10, 10), 'bilinear'),
        ((100, 100), (20, 20), 'bicubic'),
    ]
)
@image_comparison(['figimage', 'figimage10x10', 'figimage20x20'],
                  extensions=['png', 'pdf'])
def test_figimage_size_interpolation(img_size, fig_size, interpolation):
    fig = plt.figure(figsize=fig_size, dpi=100)
    x, y = np.ix_(np.arange(img_size[0]) / img_size[0],
                  np.arange(img_size[1]) / img_size[1])
    z = np.sin(x**2 + y**2 - x*y)
    c = np.sin(20*x**2 + 50*y**2)
    img = z + c/5

    fig.figimage(img, xo=0, yo=0, origin='lower', interpolation=interpolation)
    fig.figimage(img[::-1, :], xo=0, yo=100, origin='lower', interpolation=interpolation)
    fig.figimage(img[:, ::-1], xo=100, yo=0, origin='lower', interpolation=interpolation)
    fig.figimage(img[::-1, ::-1], xo=100, yo=100, origin='lower', interpolation=interpolation)


注释：


@pytest.mark.parametrize(
    "img_size, fig_size, interpolation",
    [
        ((100, 100), (1, 1), 'nearest'),     # 参数化测试，使用最近邻插值方法，小图像，小尺寸的图形
        ((100, 100), (10, 10), 'bilinear'),  # 参数化测试，使用双线性插值方法，中等大小图像，中等尺寸的图形
        ((100, 100), (20, 20), 'bicubic'),   # 参数化测试，使用双三次插值方法，大图像，大尺寸的图形
    ]
)
@image_comparison(['figimage', 'figimage10x10', 'figimage20x20'],
                  extensions=['png', 'pdf'])
def test_figimage_size_interpolation(img_size, fig_size, interpolation):
    fig = plt.figure(figsize=fig_size, dpi=100)  # 创建指定尺寸和 DPI 的图形对象
    x, y = np.ix_(np.arange(img_size[0]) / img_size[0],
                  np.arange(img_size[1]) / img_size[1])  # 生成网格坐标
    z = np.sin(x**2 + y**2 - x*y)  # 计算数据矩阵
    c = np.sin(20*x**2 + 50*y**2)  # 计算辅助数据
    img = z + c/5  # 合成图像数据

    # 在图形对象上绘制图像，设置插值方法、原点位置和偏移量
    fig.figimage(img, xo=0, yo=0, origin='lower', interpolation=interpolation)
    fig.figimage(img[::-1, :], xo=0, yo=100, origin='lower', interpolation=interpolation)
    fig.figimage(img[:, ::-1], xo=100, yo=0, origin='lower', interpolation=interpolation)
    fig.figimage(img[::-1, ::-1], xo=100, yo=100, origin='lower', interpolation=interpolation)
    # 定义一个包含元组的列表，每个元组包含三个元素：第一个是整数，第二个可以是整数或浮点数，第三个是字符串。
    [(5, 2, "hanning"),  # 数据大于图形大小。
     (5, 5, "nearest"),  # 精确重采样。
     (5, 10, "nearest"),  # 倍数采样。
     (3, 2.9, "hanning"),  # 小于3倍的上采样。
     (3, 9.1, "nearest"),  # 大于3倍的上采样。
     ])
@check_figures_equal(extensions=['png'])
# 使用装饰器检查两个图形是否相等，并指定要生成的图像格式为 PNG
def test_imshow_antialiased(fig_test, fig_ref,
                            img_size, fig_size, interpolation):
    # 设置随机数种子
    np.random.seed(19680801)
    # 获取当前保存图像的 DPI 设置
    dpi = plt.rcParams["savefig.dpi"]
    # 创建一个随机数组，大小为 dpi * img_size 的正方形
    A = np.random.rand(int(dpi * img_size), int(dpi * img_size))
    # 针对 fig_test 和 fig_ref，设置图像的尺寸为 fig_size
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(fig_size, fig_size)
    # 在 fig_test 上创建子图 ax，并设置其位置为整个图像区域
    ax = fig_test.subplots()
    ax.set_position([0, 0, 1, 1])
    # 在 ax 上显示 A 数组，使用 'antialiased' 插值方法
    ax.imshow(A, interpolation='antialiased')
    # 在 fig_ref 上创建子图 ax，并设置其位置为整个图像区域
    ax = fig_ref.subplots()
    ax.set_position([0, 0, 1, 1])
    # 在 ax 上显示 A 数组，使用给定的插值方法 interpolation
    ax.imshow(A, interpolation=interpolation)


@check_figures_equal(extensions=['png'])
# 使用装饰器检查两个图形是否相等，并指定要生成的图像格式为 PNG
def test_imshow_zoom(fig_test, fig_ref):
    # 应该小于 3 倍放大，因此应该使用最近邻插值...
    np.random.seed(19680801)
    # 获取当前保存图像的 DPI 设置
    dpi = plt.rcParams["savefig.dpi"]
    # 创建一个随机数组，大小为 dpi * 3 的正方形
    A = np.random.rand(int(dpi * 3), int(dpi * 3))
    # 针对 fig_test 和 fig_ref，设置图像的尺寸为 2.9 英寸
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(2.9, 2.9)
    # 在 fig_test 上创建子图 ax，并显示 A 数组，使用 'antialiased' 插值方法
    ax = fig_test.subplots()
    ax.imshow(A, interpolation='antialiased')
    # 设置图像的 x 和 y 轴限制为 [10, 20]
    ax.set_xlim([10, 20])
    ax.set_ylim([10, 20])
    # 在 fig_ref 上创建子图 ax，并显示 A 数组，使用 'nearest' 插值方法
    ax = fig_ref.subplots()
    ax.imshow(A, interpolation='nearest')
    # 设置图像的 x 和 y 轴限制为 [10, 20]
    ax.set_xlim([10, 20])
    ax.set_ylim([10, 20])


@check_figures_equal()
# 使用装饰器检查两个图形是否相等，并使用默认的扩展名
def test_imshow_pil(fig_test, fig_ref):
    # 使用默认样式
    style.use("default")
    # 设置 PNG 和 TIFF 图像的路径
    png_path = Path(__file__).parent / "baseline_images/pngsuite/basn3p04.png"
    tiff_path = Path(__file__).parent / "baseline_images/test_image/uint16.tif"
    # 在 fig_test 上创建包含两个子图的 axs 数组
    axs = fig_test.subplots(2)
    # 在 axs[0] 上显示 PNG 图像
    axs[0].imshow(Image.open(png_path))
    # 在 axs[1] 上显示 TIFF 图像
    axs[1].imshow(Image.open(tiff_path))
    # 在 fig_ref 上创建包含两个子图的 axs 数组
    axs = fig_ref.subplots(2)
    # 在 axs[0] 上显示读取的 PNG 图像
    axs[0].imshow(plt.imread(png_path))
    # 在 axs[1] 上显示读取的 TIFF 图像
    axs[1].imshow(plt.imread(tiff_path))


def test_imread_pil_uint16():
    # 读取 uint16 格式的图像
    img = plt.imread(os.path.join(os.path.dirname(__file__),
                     'baseline_images', 'test_image', 'uint16.tif'))
    # 断言图像的数据类型为 np.uint16
    assert img.dtype == np.uint16
    # 断言图像所有像素值的总和为指定值
    assert np.sum(img) == 134184960


def test_imread_fspath():
    # 使用路径对象读取图像
    img = plt.imread(
        Path(__file__).parent / 'baseline_images/test_image/uint16.tif')
    # 断言图像的数据类型为 np.uint16
    assert img.dtype == np.uint16
    # 断言图像所有像素值的总和为指定值
    assert np.sum(img) == 134184960


@pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg", "tiff"])
# 使用参数化测试来测试不同格式的图像保存
def test_imsave(fmt):
    # 检查是否有 alpha 通道
    has_alpha = fmt not in ["jpg", "jpeg"]

    # 用户可以指定输出图像的逻辑 DPI，但实际上并不会添加额外的像素，只是用于元数据
    # 所以我们测试传统情况（dpi == 1）和新情况（dpi == 100），并读取生成的 PNG 文件，确保数据完全相同。

    np.random.seed(1)
    # 选择 1856 像素的高度是因为以 100 DPI 保存图像，使用 Pillow 提供的格式会导致四舍五入误差，最终图像形状为 1855。
    data = np.random.rand(1856, 2)

    buff_dpi1 = io.BytesIO()
    # 保存数据数组为 PNG 格式，dpi 为 1
    plt.imsave(buff_dpi1, data, format=fmt, dpi=1)

    buff_dpi100 = io.BytesIO()
    # 将图像数据保存为指定格式到内存缓冲区 buff_dpi100，设置 DPI 为 100
    plt.imsave(buff_dpi100, data, format=fmt, dpi=100)
    
    # 将内存缓冲区 buff_dpi1 的指针位置移动到起始位置
    buff_dpi1.seek(0)
    # 从内存缓冲区 buff_dpi1 中读取图像数据并解码，使用指定的格式 fmt
    arr_dpi1 = plt.imread(buff_dpi1, format=fmt)
    
    # 将内存缓冲区 buff_dpi100 的指针位置移动到起始位置
    buff_dpi100.seek(0)
    # 从内存缓冲区 buff_dpi100 中读取图像数据并解码，使用指定的格式 fmt
    arr_dpi100 = plt.imread(buff_dpi100, format=fmt)
    
    # 断言：验证 arr_dpi1 的形状是否为 (1856, 2, 3 + has_alpha)
    assert arr_dpi1.shape == (1856, 2, 3 + has_alpha)
    # 断言：验证 arr_dpi100 的形状是否为 (1856, 2, 3 + has_alpha)
    assert arr_dpi100.shape == (1856, 2, 3 + has_alpha)
    
    # 断言：验证 arr_dpi1 和 arr_dpi100 是否完全相等
    assert_array_equal(arr_dpi1, arr_dpi100)
@pytest.mark.parametrize("origin", ["upper", "lower"])
# 使用 pytest 提供的参数化装饰器，定义测试函数 test_imsave_rgba_origin，测试不同参数下的行为
def test_imsave_rgba_origin(origin):
    # 创建一个字节流对象 buf
    buf = io.BytesIO()
    # 创建一个 10x10x4 的全零数组，数据类型为 uint8
    result = np.zeros((10, 10, 4), dtype='uint8')
    # 使用 matplotlib 的 imsave 函数将 result 数组保存为 PNG 格式到 buf 中，传入 origin 参数
    mimage.imsave(buf, arr=result, format="png", origin=origin)


@pytest.mark.parametrize("fmt", ["png", "pdf", "ps", "eps", "svg"])
# 使用 pytest 提供的参数化装饰器，定义测试函数 test_imsave_fspath，测试不同格式保存到文件路径的行为
def test_imsave_fspath(fmt):
    # 使用 matplotlib 的 imsave 函数将一个 2x2 的数组保存为指定格式（如 PNG、PDF 等）到 os.devnull
    plt.imsave(Path(os.devnull), np.array([[0, 1]]), format=fmt)


def test_imsave_color_alpha():
    # 测试 matplotlib 的 imsave 函数处理三维数组（包含颜色和透明度信息）的能力
    # 使用随机种子设定随机数生成器
    np.random.seed(1)

    # 循环测试 'lower' 和 'upper' 两种 origin 参数下的行为
    for origin in ['lower', 'upper']:
        # 创建一个随机数据的 16x16x4 数组
        data = np.random.rand(16, 16, 4)
        # 创建一个字节流对象 buff
        buff = io.BytesIO()
        # 使用 matplotlib 的 imsave 函数将 data 数组保存为 PNG 格式到 buff 中，传入 origin 参数
        plt.imsave(buff, data, origin=origin, format="png")

        # 将 buff 的读取指针移到开头
        buff.seek(0)
        # 使用 matplotlib 的 imread 函数读取 buff 中的数据为 arr_buf
        arr_buf = plt.imread(buff)

        # 重新将浮点数数据转换为 uint8 类型，以便比较数据的精度
        data = (255 * data).astype('uint8')
        # 如果 origin 为 'lower'，则反转 data 数组的顺序
        if origin == 'lower':
            data = data[::-1]
        # 将 arr_buf 数组也转换为 uint8 类型
        arr_buf = (255 * arr_buf).astype('uint8')

        # 使用 numpy 的 assert_array_equal 函数断言 data 和 arr_buf 数组内容相等
        assert_array_equal(data, arr_buf)


def test_imsave_pil_kwargs_png():
    # 测试 matplotlib 的 imsave 函数使用 pil_kwargs 参数保存 PNG 图像的行为
    from PIL.PngImagePlugin import PngInfo
    # 创建一个字节流对象 buf
    buf = io.BytesIO()
    # 创建一个 PngInfo 对象，添加 "Software" 文本信息
    pnginfo = PngInfo()
    pnginfo.add_text("Software", "test")
    # 使用 matplotlib 的 imsave 函数将一个 2x2 的数组保存为 PNG 格式到 buf 中，传入 pil_kwargs 参数
    plt.imsave(buf, [[0, 1], [2, 3]],
               format="png", pil_kwargs={"pnginfo": pnginfo})
    # 使用 PIL 的 Image 类打开 buf 中的图像为 im 对象
    im = Image.open(buf)
    # 断言 im 对象的信息中包含 "Software" 且为 "test"
    assert im.info["Software"] == "test"


def test_imsave_pil_kwargs_tiff():
    # 测试 matplotlib 的 imsave 函数使用 pil_kwargs 参数保存 TIFF 图像的行为
    from PIL.TiffTags import TAGS_V2 as TAGS
    # 创建一个字节流对象 buf
    buf = io.BytesIO()
    # 创建一个包含描述信息的 pil_kwargs 字典
    pil_kwargs = {"description": "test image"}
    # 使用 matplotlib 的 imsave 函数将一个 2x2 的数组保存为 TIFF 格式到 buf 中，传入 pil_kwargs 参数
    plt.imsave(buf, [[0, 1], [2, 3]], format="tiff", pil_kwargs=pil_kwargs)
    # 断言 pil_kwargs 字典中只有一个键值对
    assert len(pil_kwargs) == 1
    # 使用 PIL 的 Image 类打开 buf 中的图像为 im 对象
    im = Image.open(buf)
    # 将 im 对象的标签转换为字典 tags
    tags = {TAGS[k].name: v for k, v in im.tag_v2.items()}
    # 断言 tags 字典中的 "ImageDescription" 值为 "test image"
    assert tags["ImageDescription"] == "test image"


@image_comparison(['image_alpha'], remove_text=True)
# 使用 image_comparison 装饰器，比较测试函数 test_image_alpha 生成的图像和参考图像 'image_alpha'
def test_image_alpha():
    np.random.seed(0)
    Z = np.random.rand(6, 6)

    # 创建一个包含三个子图的图像 fig 和相应的轴对象 ax1, ax2, ax3
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # 在 ax1 上显示 Z 数据，设置 alpha 参数为 1.0，插值方式为 'none'
    ax1.imshow(Z, alpha=1.0, interpolation='none')
    # 在 ax2 上显示 Z 数据，设置 alpha 参数为 0.5，插值方式为 'none'
    ax2.imshow(Z, alpha=0.5, interpolation='none')
    # 在 ax3 上显示 Z 数据，设置 alpha 参数为 0.5，插值方式为 'nearest'


@mpl.style.context('mpl20')
@check_figures_equal(extensions=['png'])
# 使用 mpl.style.context 和 check_figures_equal 装饰器，测试函数 test_imshow_alpha 的行为
def test_imshow_alpha(fig_test, fig_ref):
    np.random.seed(19680801)

    # 创建一个随机的 6x6x3 的 RGB 浮点数组 rgbf 和其对应的 uint8 数组 rgbu
    rgbf = np.random.rand(6, 6, 3)
    rgbu = np.uint8(rgbf * 255)
    # 创建一个包含四个子图的测试图像 fig_test 和参考图像 fig_ref
    ((ax0, ax1), (ax2, ax3)) = fig_test.subplots(2, 2)
    # 在 ax0 上显示 rgbf 数据，设置 alpha 参数为 0.5
    ax0.imshow(rgbf, alpha=0.5)
    # 在 ax1 上显示 rgbf 数据，设置 alpha 参数为 0.75
    ax1.imshow(rgbf, alpha=0.75)
    # 在 ax2 上显示 rgbu 数据，设置 alpha 参数为 0.5
    ax2.imshow(rgbu, alpha=0.5)
    # 在 ax3 上显示 rgbu 数据，设置 alpha 参数为 0.75

    # 创建一个包含 alpha 通道的 rgbaf 和 rgbau 数组
    rgbaf = np.concatenate((rgbf, np.ones((6, 6, 1))), axis=2)
    rgbau = np.concatenate((rgbu, np.full((6, 6, 1), 255, np.uint8)), axis=2)
    # 创建一个包含四个子图的参考图像 fig_ref
    ((ax0, ax1), (ax2, ax3)) = fig_ref.subplots(2, 2)
    # 将rgbaf数组的所有像素的alpha通道值设为0.5
    rgbaf[:, :, 3] = 0.5
    # 在ax0图形化界面中显示rgbaf数组
    ax0.imshow(rgbaf)
    # 将rgbaf数组的所有像素的alpha通道值设为0.75
    rgbaf[:, :, 3] = 0.75
    # 在ax1图形化界面中显示更新后的rgbaf数组
    ax1.imshow(rgbaf)
    # 将rgbau数组的所有像素的alpha通道值设为127
    rgbau[:, :, 3] = 127
    # 在ax2图形化界面中显示rgbau数组
    ax2.imshow(rgbau)
    # 将rgbau数组的所有像素的alpha通道值设为191
    rgbau[:, :, 3] = 191
    # 在ax3图形化界面中显示更新后的rgbau数组
    ax3.imshow(rgbau)
def test_cursor_data():
    # 导入 MouseEvent 类
    from matplotlib.backend_bases import MouseEvent

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 在坐标轴上显示像素值为 0 到 99 的图像
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='upper')

    # 设置测试点的坐标
    x, y = 4, 4

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 44
    assert im.get_cursor_data(event) == 44

    # 测试图像外的点
    # 检查问题 #4957
    x, y = 10.1, 4

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 None
    assert im.get_cursor_data(event) is None

    # 注释部分：这段代码被注释掉，用于进一步测试在翻转坐标轴后的情况
    # x, y = 0.1, -0.1
    # xdisp, ydisp = ax.transData.transform([x, y])
    # event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    # z = im.get_cursor_data(event)
    # assert z is None, "Did not get None, got %d" % z

    # 清空坐标轴内容
    ax.clear()

    # 在翻转坐标轴后重新显示图像
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='lower')

    # 设置测试点的坐标
    x, y = 4, 4

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 44
    assert im.get_cursor_data(event) == 44

    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 在指定范围内显示图像
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 0.5, 0, 0.5])

    # 设置测试点的坐标
    x, y = 0.25, 0.25

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 55
    assert im.get_cursor_data(event) == 55

    # 测试图像外的点
    # 检查问题 #4957
    x, y = 0.75, 0.25

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 None
    assert im.get_cursor_data(event) is None

    # 设置测试点的坐标
    x, y = 0.01, -0.01

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 None
    assert im.get_cursor_data(event) is None

    # 添加额外的变换到图像艺术家上
    trans = Affine2D().scale(2).rotate(0.5)
    im = ax.imshow(np.arange(100).reshape(10, 10),
                   transform=trans + ax.transData)

    # 设置测试点的坐标
    x, y = 3, 10

    # 转换坐标点到显示坐标系中的位置
    xdisp, ydisp = ax.transData.transform([x, y])

    # 创建 MouseEvent 对象，模拟鼠标移动事件
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)

    # 断言获取的光标数据是否为 44
    assert im.get_cursor_data(event) == 44


@pytest.mark.parametrize("xy, data", [
    # 用于参数化测试，验证不同点的预期数据
    [[0.5, 0.5], 0 + 0],
    [[0.5, 1.5], 0 + 1],
    [[4.5, 0.5], 16 + 0],
    [[8.5, 0.5], 16 + 0],
    [[9.5, 2.5], 81 + 4],
    [[-1, 0.5], None],
    [[0.5, -1], None],
    ]
)
def test_cursor_data_nonuniform(xy, data):
    # 导入 MouseEvent 类
    from matplotlib.backend_bases import MouseEvent

    # 创建非线性的 x 值集合
    x = np.array([0, 1, 4, 9, 16])

    # 创建线性的 y 值集合
    y = np.array([0, 1, 2, 3, 4])

    # 创建图像数据 z，代表点的平方和
    z = x[np.newaxis, :]**2 + y[:, np.newaxis]**2
    # 创建一个包含单个子图的图形对象和轴对象
    fig, ax = plt.subplots()
    
    # 创建一个非均匀图像对象，并设置其数据和范围
    im = NonUniformImage(ax, extent=(x.min(), x.max(), y.min(), y.max()))
    im.set_data(x, y, z)  # 设置非均匀图像对象的数据
    
    # 将非均匀图像对象添加到轴对象中显示
    ax.add_image(im)
    
    # 设置轴对象的 X 轴和 Y 轴的显示范围，设置较低的最小限制以便在图像外部测试光标
    ax.set_xlim(x.min() - 2, x.max())
    ax.set_ylim(y.min() - 2, y.max())
    
    # 将数据点的坐标转换为轴坐标系中的显示坐标
    xdisp, ydisp = ax.transData.transform(xy)
    
    # 创建模拟鼠标事件对象，并传递给它转换后的坐标数据
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    
    # 断言当前非均匀图像对象在给定事件下的光标数据等于预期的数据，用于测试目的
    assert im.get_cursor_data(event) == data, (im.get_cursor_data(event), data)
@pytest.mark.parametrize(
    "data, text", [
        ([[10001, 10000]], "[10001.000]"),  # 参数化测试数据：测试数据和预期文本结果
        ([[.123, .987]], "[0.123]"),       # 参数化测试数据：测试数据和预期文本结果
        ([[np.nan, 1, 2]], "[]"),          # 参数化测试数据：测试数据和预期文本结果
        ([[1, 1+1e-15]], "[1.0000000000000000]"),  # 参数化测试数据：测试数据和预期文本结果
        ([[-1, -1]], "[-1.0000000000000000]"),     # 参数化测试数据：测试数据和预期文本结果
    ])
def test_format_cursor_data(data, text):
    # 导入必要的模块，准备测试环境
    from matplotlib.backend_bases import MouseEvent

    # 创建图形和轴
    fig, ax = plt.subplots()
    # 在轴上显示数据
    im = ax.imshow(data)

    # 获取数据坐标转换后的显示位置
    xdisp, ydisp = ax.transData.transform([0, 0])
    # 创建鼠标事件对象
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    # 断言图像的格式化光标数据与预期文本结果一致
    assert im.format_cursor_data(im.get_cursor_data(event)) == text


@image_comparison(['image_clip'], style='mpl20')
def test_image_clip():
    # 准备测试环境，创建图形和轴
    d = [[1, 2], [3, 4]]
    fig, ax = plt.subplots()
    # 在轴上显示图像数据
    im = ax.imshow(d)

    # 创建圆形裁剪路径对象
    patch = patches.Circle((0, 0), radius=1, transform=ax.transData)
    # 设置图像的裁剪路径
    im.set_clip_path(patch)


@image_comparison(['image_cliprect'], style='mpl20')
def test_image_cliprect():
    # 准备测试环境，创建图形和轴
    fig, ax = plt.subplots()
    d = [[1, 2], [3, 4]]
    # 在轴上显示图像数据，并设置显示范围
    im = ax.imshow(d, extent=(0, 5, 0, 5))

    # 创建矩形裁剪路径对象
    rect = patches.Rectangle(
        xy=(1, 1), width=2, height=2, transform=im.axes.transData)
    # 设置图像的裁剪路径
    im.set_clip_path(rect)


@check_figures_equal(extensions=['png'])
def test_imshow_10_10_1(fig_test, fig_ref):
    # 测试 imshow 函数对于 10x10x1 数据的显示，确保结果一致
    arr = np.arange(100).reshape((10, 10, 1))
    # 在参考图形中创建子图
    ax = fig_ref.subplots()
    ax.imshow(arr[:, :, 0], interpolation="bilinear", extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # 在测试图形中创建子图
    ax = fig_test.subplots()
    ax.imshow(arr, interpolation="bilinear", extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)


def test_imshow_10_10_2():
    # 测试 imshow 函数对于 10x10x2 数据的行为，期望抛出 TypeError 异常
    fig, ax = plt.subplots()
    arr = np.arange(200).reshape((10, 10, 2))
    with pytest.raises(TypeError):
        ax.imshow(arr)


def test_imshow_10_10_5():
    # 测试 imshow 函数对于 10x10x5 数据的行为，期望抛出 TypeError 异常
    fig, ax = plt.subplots()
    arr = np.arange(500).reshape((10, 10, 5))
    with pytest.raises(TypeError):
        ax.imshow(arr)


@image_comparison(['no_interpolation_origin'], remove_text=True)
def test_no_interpolation_origin():
    # 测试 imshow 函数的原点参数设置，生成对比图像
    fig, axs = plt.subplots(2)
    # 在第一个子图中显示数据，设置原点在底部
    axs[0].imshow(np.arange(100).reshape((2, 50)), origin="lower",
                  interpolation='none')
    # 在第二个子图中显示数据，不进行插值
    axs[1].imshow(np.arange(100).reshape((2, 50)), interpolation='none')


@image_comparison(['image_shift'], remove_text=True, extensions=['pdf', 'svg'])
def test_image_shift():
    # 测试 imshow 函数的图像位移效果，生成对比图像
    imgData = [[1 / x + 1 / y for x in range(1, 100)] for y in range(1, 100)]
    tMin = 734717.945208
    tMax = 734717.946366

    fig, ax = plt.subplots()
    ax.imshow(imgData, norm=colors.LogNorm(), interpolation='none',
              extent=(tMin, tMax, 1, 100))
    ax.set_aspect('auto')


def test_image_edges():
    # 测试 imshow 函数对于边缘情况的处理
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    # 创建二维数据，并设置其在坐标轴上的显示范围和属性
    data = np.tile(np.arange(12), 15).reshape(20, 9)
    im = ax.imshow(data, origin='upper', extent=[-10, 10, -10, 10],
                   interpolation='none', cmap='gray')

    # 设置坐标轴的 x 和 y 轴限制
    x = y = 2
    ax.set_xlim([-x, x])
    # 设置图形的 y 轴范围为 [-y, y]
    ax.set_ylim([-y, y])

    # 清除 x 轴和 y 轴的刻度标签
    ax.set_xticks([])
    ax.set_yticks([])

    # 创建一个字节流对象 buf
    buf = io.BytesIO()

    # 将图形保存到字节流 buf 中，背景色设置为绿色 (0, 1, 0)
    fig.savefig(buf, facecolor=(0, 1, 0))

    # 将 buf 的读取位置移动到开头
    buf.seek(0)

    # 从 buf 中读取图像数据并存入变量 im
    im = plt.imread(buf)

    # 计算图像第一列像素的 RGB 通道值之和
    r, g, b, a = sum(im[:, 0])

    # 计算图像最后一列像素的 RGB 通道值之和
    r, g, b, a = sum(im[:, -1])

    # 断言 g 不等于 100，如果等于则抛出异常并显示消息
    assert g != 100, 'Expected a non-green edge - but sadly, it was.'
@image_comparison(['image_composite_background'],
                  remove_text=True, style='mpl20')
def test_image_composite_background():
    # 创建一个包含图形和坐标轴的新图形对象
    fig, ax = plt.subplots()
    # 创建一个4x3的数组
    arr = np.arange(12).reshape(4, 3)
    # 在坐标轴上显示数组 arr，使用给定的范围进行拉伸
    ax.imshow(arr, extent=[0, 2, 15, 0])
    # 在同一个坐标轴上再次显示数组 arr，不同的范围进行拉伸
    ax.imshow(arr, extent=[4, 6, 15, 0])
    # 设置坐标轴背景颜色为半透明红色
    ax.set_facecolor((1, 0, 0, 0.5))
    # 设置坐标轴 x 轴的显示范围
    ax.set_xlim([0, 12])


@image_comparison(['image_composite_alpha'], remove_text=True)
def test_image_composite_alpha():
    """
    Tests that the alpha value is recognized and correctly applied in the
    process of compositing images together.
    """
    # 创建一个包含图形和坐标轴的新图形对象
    fig, ax = plt.subplots()
    # 创建一个大小为 (11, 21, 4) 的全零数组
    arr = np.zeros((11, 21, 4))
    arr[:, :, 0] = 1  # 将数组的第一个通道设为1
    # 将数组的第四个通道设为一个渐变的 alpha 值
    arr[:, :, 3] = np.concatenate(
        (np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))
    # 创建一个大小为 (21, 11, 4) 的全零数组
    arr2 = np.zeros((21, 11, 4))
    arr2[:, :, 0] = 1  # 将数组的第一个通道设为1
    arr2[:, :, 1] = 1  # 将数组的第二个通道设为1
    # 将数组的第四个通道设为一个渐变的 alpha 值，通过增加维度实现
    arr2[:, :, 3] = np.concatenate(
        (np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))[:, np.newaxis]
    # 在坐标轴上显示数组 arr，使用给定的范围进行拉伸，并设置透明度
    ax.imshow(arr, extent=[1, 2, 5, 0], alpha=0.3)
    # 在同一坐标轴上再次显示数组 arr，使用不同的范围进行拉伸，并设置透明度
    ax.imshow(arr, extent=[2, 3, 5, 0], alpha=0.6)
    # 在同一坐标轴上显示数组 arr，使用给定的范围进行拉伸
    ax.imshow(arr, extent=[3, 4, 5, 0])
    # 在同一坐标轴上显示数组 arr2，使用给定的范围进行拉伸
    ax.imshow(arr2, extent=[0, 5, 1, 2])
    # 在同一坐标轴上再次显示数组 arr2，使用不同的范围进行拉伸，并设置透明度
    ax.imshow(arr2, extent=[0, 5, 2, 3], alpha=0.6)
    # 在同一坐标轴上再次显示数组 arr2，使用不同的范围进行拉伸，并设置透明度
    ax.imshow(arr2, extent=[0, 5, 3, 4], alpha=0.3)
    # 设置坐标轴背景颜色为绿色
    ax.set_facecolor((0, 0.5, 0, 1))
    # 设置坐标轴的 x 轴显示范围
    ax.set_xlim([0, 5])
    # 设置坐标轴的 y 轴显示范围
    ax.set_ylim([5, 0])


@check_figures_equal(extensions=["pdf"])
def test_clip_path_disables_compositing(fig_test, fig_ref):
    # 创建一个 3x3 的数组
    t = np.arange(9).reshape((3, 3))
    # 对于每个图形（测试和参考），添加一个子图
    for fig in [fig_test, fig_ref]:
        ax = fig.add_subplot()
        # 在子图上显示数组 t，并使用裁剪路径来限制显示区域
        ax.imshow(t, clip_path=(mpl.path.Path([(0, 0), (0, 1), (1, 0)]),
                                ax.transData))
        # 在同一子图上再次显示数组 t，并使用不同的裁剪路径来限制显示区域
        ax.imshow(t, clip_path=(mpl.path.Path([(1, 1), (1, 2), (2, 1)]),
                                ax.transData))
    # 设置参考图的属性，禁止图形的混合（compositing）
    fig_ref.suppressComposite = True


@image_comparison(['rasterize_10dpi'],
                  extensions=['pdf', 'svg'], remove_text=True, style='mpl20')
def test_rasterize_dpi():
    # This test should check rasterized rendering with high output resolution.
    # It plots a rasterized line and a normal image with imshow.  So it will
    # catch when images end up in the wrong place in case of non-standard dpi
    # setting.  Instead of high-res rasterization I use low-res.  Therefore
    # the fact that the resolution is non-standard is easily checked by
    # image_comparison.
    # 创建一个 2x2 的数组
    img = np.asarray([[1, 2], [3, 4]])

    # 创建一个包含三个子图的图形对象，指定尺寸为 (3, 1)
    fig, axs = plt.subplots(1, 3, figsize=(3, 1))

    # 在第一个子图上显示数组 img
    axs[0].imshow(img)

    # 在第二个子图上绘制一条线，并进行光栅化处理，以低分辨率显示
    axs[1].plot([0, 1], [0, 1], linewidth=20., rasterized=True)
    axs[1].set(xlim=(0, 1), ylim=(-1, 2))

    # 在第三个子图上绘制一条线，不进行光栅化处理
    axs[2].plot([0, 1], [0, 1], linewidth=20.)
    axs[2].set(xlim=(0, 1), ylim=(-1, 2))

    # 设置每个子图的 x 和 y 轴刻度为空
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        # 隐藏子图的轴线
        ax.spines[:].set_visible(False)

    # 设置全局参数，保存图形时的分辨率为 10dpi
    rcParams['savefig.dpi'] = 10
@image_comparison(['bbox_image_inverted'], remove_text=True, style='mpl20')
def test_bbox_image_inverted():
    # 创建一个用于生成图像以供 BboxImage 使用的示例
    image = np.arange(100).reshape((10, 10))

    # 创建一个图形和轴对象
    fig, ax = plt.subplots()

    # 创建一个 BboxImage 对象，用给定的变换后的边界框（转换为数据坐标系）作为基础
    bbox_im = BboxImage(
        TransformedBbox(Bbox([[100, 100], [0, 0]]), ax.transData),
        interpolation='nearest')
    bbox_im.set_data(image)  # 设置图像数据
    bbox_im.set_clip_on(False)  # 设置不裁剪图像
    ax.set_xlim(0, 100)  # 设置 X 轴范围
    ax.set_ylim(0, 100)  # 设置 Y 轴范围
    ax.add_artist(bbox_im)  # 将 BboxImage 添加到轴上

    # 创建一个单位矩阵图像
    image = np.identity(10)

    # 创建第二个 BboxImage 对象，用给定的变换后的边界框（转换为图形坐标系）作为基础
    bbox_im = BboxImage(TransformedBbox(Bbox([[0.1, 0.2], [0.3, 0.25]]),
                                        ax.figure.transFigure),
                        interpolation='nearest')
    bbox_im.set_data(image)  # 设置图像数据
    bbox_im.set_clip_on(False)  # 设置不裁剪图像
    ax.add_artist(bbox_im)  # 将第二个 BboxImage 添加到轴上


def test_get_window_extent_for_AxisImage():
    # 创建一个已知大小（1000x1000 像素）的图形，放置一个图像对象，并检查 get_window_extent()
    # 返回正确的边界框值（以像素为单位）。

    im = np.array([[0.25, 0.75, 1.0, 0.75], [0.1, 0.65, 0.5, 0.4],
                   [0.6, 0.3, 0.0, 0.2], [0.7, 0.9, 0.4, 0.6]])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(
        im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest')

    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)

    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(
        im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest',
        transform=ax.transAxes)

    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)

    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])


@image_comparison(['zoom_and_clip_upper_origin.png'],
                  remove_text=True, style='mpl20')
def test_zoom_and_clip_upper_origin():
    # 创建一个 10x10 的图像数组
    image = np.arange(100)
    image = image.reshape((10, 10))

    # 创建一个图形和轴对象
    fig, ax = plt.subplots()

    # 在轴上显示图像，并设置 Y 轴的限制为 2.0 到 -0.5，X 轴的限制为 -0.5 到 2.0
    ax.imshow(image)
    ax.set_ylim(2.0, -0.5)
    ax.set_xlim(-0.5, 2.0)


def test_nonuniformimage_setcmap():
    # 获取当前图形的当前轴对象
    ax = plt.gca()

    # 创建一个 NonUniformImage 对象
    im = NonUniformImage(ax)

    # 设置颜色映射为 'Blues'
    im.set_cmap('Blues')


def test_nonuniformimage_setnorm():
    # 获取当前图形的当前轴对象
    ax = plt.gca()

    # 创建一个 NonUniformImage 对象
    im = NonUniformImage(ax)

    # 设置归一化方式为默认的 Normalize() 对象
    im.set_norm(plt.Normalize())


def test_jpeg_2d():
    # 烟雾测试，验证模式为 'L' 的 PIL 图像是否正常工作。
    imd = np.ones((10, 10), dtype='uint8')
    for i in range(10):
        imd[i, :] = np.linspace(0.0, 1.0, 10) * 255
    im = Image.new('L', (10, 10))
    im.putdata(imd.flatten())
    fig, ax = plt.subplots()

    # 在轴上显示 PIL 图像
    ax.imshow(im)


def test_jpeg_alpha():
    plt.figure(figsize=(1, 1), dpi=300)
    # 创建一个全黑的图像，并在其上创建从 0 到 1 的渐变
    # 创建一个300x300像素的RGBA图像，初始值为全零（黑色），alpha通道从完全透明到完全不透明线性变化
    im = np.zeros((300, 300, 4), dtype=float)
    im[..., 3] = np.linspace(0.0, 1.0, 300)
    
    # 在当前图形窗口中显示im图像
    plt.figimage(im)
    
    # 创建一个字节流对象用于保存图形数据
    buff = io.BytesIO()
    # 将当前图形保存为JPEG格式到buff中，背景色为红色，分辨率为300dpi
    plt.savefig(buff, facecolor="red", format='jpg', dpi=300)
    
    # 将字节流的读写指针移动到起始位置
    buff.seek(0)
    # 从字节流中打开图像
    image = Image.open(buff)
    
    # 获取图像中使用的颜色数量，应该包含256种灰度色阶
    num_colors = len(image.getcolors(256))
    # 断言颜色数量在175到210之间，确保图像包含了预期的灰度色阶数量
    assert 175 <= num_colors <= 210
    
    # 断言图像最左上角像素的颜色是红色，确保完全透明的部分是红色的
    corner_pixel = image.getpixel((0, 0))
    assert corner_pixel == (254, 0, 0)
def test_axesimage_setdata():
    ax = plt.gca()  # 获取当前图的轴对象
    im = AxesImage(ax)  # 创建一个基于轴对象的AxesImage对象
    z = np.arange(12, dtype=float).reshape((4, 3))  # 创建一个12个元素的浮点数数组，并重塑为4x3的数组
    im.set_data(z)  # 设置AxesImage对象的数据为z
    z[0, 0] = 9.9  # 修改z数组的第一个元素为9.9
    assert im._A[0, 0] == 0, 'value changed'  # 断言检查AxesImage对象的_A属性的第一个元素是否为0，用于检测数值是否发生变化


def test_figureimage_setdata():
    fig = plt.gcf()  # 获取当前图形对象
    im = FigureImage(fig)  # 创建一个基于图形对象的FigureImage对象
    z = np.arange(12, dtype=float).reshape((4, 3))  # 创建一个12个元素的浮点数数组，并重塑为4x3的数组
    im.set_data(z)  # 设置FigureImage对象的数据为z
    z[0, 0] = 9.9  # 修改z数组的第一个元素为9.9
    assert im._A[0, 0] == 0, 'value changed'  # 断言检查FigureImage对象的_A属性的第一个元素是否为0，用于检测数值是否发生变化


@pytest.mark.parametrize(
    "image_cls,x,y,a", [
        (NonUniformImage,
         np.arange(3.), np.arange(4.), np.arange(12.).reshape((4, 3))),  # 参数化测试数据：NonUniformImage类的对象及其参数
        (PcolorImage,
         np.arange(3.), np.arange(4.), np.arange(6.).reshape((3, 2))),  # 参数化测试数据：PcolorImage类的对象及其参数
    ])
def test_setdata_xya(image_cls, x, y, a):
    ax = plt.gca()  # 获取当前图的轴对象
    im = image_cls(ax)  # 根据给定的image_cls类创建一个对象，传入轴对象ax
    im.set_data(x, y, a)  # 设置对象的数据为给定的x、y、a参数
    x[0] = y[0] = a[0, 0] = 9.9  # 修改x、y、a的第一个元素为9.9
    assert im._A[0, 0] == im._Ax[0] == im._Ay[0] == 0, 'value changed'  # 断言检查对象的_A、_Ax、_Ay属性的第一个元素是否都为0，用于检测数值是否发生变化
    im.set_data(x, y, a.reshape((*a.shape, -1)))  # 重新设置对象的数据为reshape后的a数组，进行一次简单的测试。


def test_minimized_rasterized():
    # This ensures that the rasterized content in the colorbars is
    # only as thick as the colorbar, and doesn't extend to other parts
    # of the image.  See #5814.  While the original bug exists only
    # in Postscript, the best way to detect it is to generate SVG
    # and then parse the output to make sure the two colorbar images
    # are the same size.
    from xml.etree import ElementTree  # 导入XML解析库ElementTree

    np.random.seed(0)  # 使用种子0初始化随机数生成器
    data = np.random.rand(10, 10)  # 创建一个10x10的随机浮点数数组

    fig, ax = plt.subplots(1, 2)  # 创建一个包含2个子图的图形对象
    p1 = ax[0].pcolormesh(data)  # 在第一个子图上绘制一个伪彩色图像
    p2 = ax[1].pcolormesh(data)  # 在第二个子图上绘制一个伪彩色图像

    plt.colorbar(p1, ax=ax[0])  # 在第一个子图上添加颜色条
    plt.colorbar(p2, ax=ax[1])  # 在第二个子图上添加颜色条

    buff = io.BytesIO()  # 创建一个字节流缓冲区
    plt.savefig(buff, format='svg')  # 将图形保存为SVG格式到缓冲区

    buff = io.BytesIO(buff.getvalue())  # 重新使用缓冲区的内容创建一个新的字节流对象
    tree = ElementTree.parse(buff)  # 使用ElementTree解析SVG内容
    width = None
    for image in tree.iter('image'):  # 迭代所有的SVG图像元素
        if width is None:
            width = image['width']  # 如果width为None，则设置为当前图像元素的宽度属性
        else:
            if image['width'] != width:  # 如果发现有图像元素的宽度与第一个不同，则断言失败
                assert False


def test_load_from_url():
    path = Path(__file__).parent / "baseline_images/pngsuite/basn3p04.png"  # 指定文件路径
    url = ('file:'
           + ('///' if sys.platform == 'win32' else '')  # 根据操作系统决定URL前缀
           + path.resolve().as_posix())  # 获取绝对路径并转换为URL格式

    with pytest.raises(ValueError, match="Please open the URL"):  # 使用pytest断言检查是否抛出指定异常并包含特定消息
        plt.imread(url)  # 尝试从指定URL读取图像

    with urllib.request.urlopen(url) as file:  # 打开指定URL的文件流
        plt.imread(file)  # 从打开的文件流中读取图像


@image_comparison(['log_scale_image'], remove_text=True)
def test_log_scale_image():
    Z = np.zeros((10, 10))  # 创建一个10x10的零数组
    Z[::2] = 1  # 将每隔一行一列的元素设为1

    fig, ax = plt.subplots()  # 创建一个图形对象和轴对象
    ax.imshow(Z, extent=[1, 100, 1, 100], cmap='viridis', vmax=1, vmin=-1,  # 在轴对象上显示Z数组的图像，并设置一些参数
              aspect='auto')
    ax.set(yscale='log')  # 设置轴的y轴为对数刻度


@image_comparison(['rotate_image'], remove_text=True)
def test_rotate_image():
    delta = 0.25  # 定义步长delta
    x = y = np.arange(-3.0, 3.0, delta)  # 创建x和y数组，范围为-3到3，步长为delta
    X, Y = np.meshgrid(x, y)  # 创建网格坐标
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)  # 创建高斯分布数组Z1
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))  # 创建高斯分布数组Z2
    Z = Z2 - Z1  # 计算高斯分布数组Z的差值
    # 创建一个包含单个子图的图形对象和一个坐标轴对象
    fig, ax1 = plt.subplots(1, 1)
    # 在坐标轴上显示图像，并指定插值方式、颜色映射、坐标原点位置、坐标范围和是否裁剪
    im1 = ax1.imshow(Z, interpolation='none', cmap='viridis',
                     origin='lower',
                     extent=[-2, 4, -3, 2], clip_on=True)

    # 创建一个仿射变换对象，将其与坐标轴的数据变换组合，实现图像的旋转
    trans_data2 = Affine2D().rotate_deg(30) + ax1.transData
    # 设置图像使用新的仿射变换
    im1.set_transform(trans_data2)

    # 获取图像显示区域的四个边界坐标
    x1, x2, y1, y2 = im1.get_extent()

    # 在坐标轴上绘制红色虚线框，框选图像的显示区域
    ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r--", lw=3,
             transform=trans_data2)

    # 设置坐标轴的显示范围
    ax1.set_xlim(2, 5)
    ax1.set_ylim(0, 4)
def test_image_preserve_size():
    # 创建一个字节流对象
    buff = io.BytesIO()

    # 创建一个大小为 (481, 321) 的全零 NumPy 数组作为图像数据
    im = np.zeros((481, 321))
    # 将图像数据保存为 PNG 格式到字节流中
    plt.imsave(buff, im, format="png")

    # 将字节流指针位置移动到开头
    buff.seek(0)
    # 从字节流中读取图像数据
    img = plt.imread(buff)

    # 断言读取的图像数据形状与原始图像数据形状一致
    assert img.shape[:2] == im.shape


def test_image_preserve_size2():
    # 设置矩阵的维度
    n = 7
    # 创建一个 n x n 的单位矩阵
    data = np.identity(n, float)

    # 创建一个指定尺寸和无边框的图像对象
    fig = plt.figure(figsize=(n, n), frameon=False)
    # 在图像对象中添加一个轴，并设置其位置
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    # 关闭轴的显示
    ax.set_axis_off()
    # 在轴上显示数据，使用最近邻插值，原点在左下角，自动调整纵横比
    ax.imshow(data, interpolation='nearest', origin='lower', aspect='auto')
    # 创建一个字节流对象
    buff = io.BytesIO()
    # 将图像保存为 PNG 格式到字节流中，设置 DPI 为 1
    fig.savefig(buff, dpi=1)

    # 将字节流指针位置移动到开头
    buff.seek(0)
    # 从字节流中读取图像数据
    img = plt.imread(buff)

    # 断言读取的图像数据形状为 (7, 7, 4)
    assert img.shape == (7, 7, 4)

    # 断言读取的图像数据第一个通道与反转的单位矩阵的第一个通道一致
    assert_array_equal(np.asarray(img[:, :, 0], bool),
                       np.identity(n, bool)[::-1])


@image_comparison(['mask_image_over_under.png'], remove_text=True, tol=1.0)
def test_mask_image_over_under():
    # 在重新生成此测试图像时移除此行
    plt.rcParams['pcolormesh.snap'] = False

    # 设置间隔
    delta = 0.025
    # 创建坐标轴点
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    # 创建两个高斯函数的差值作为 Z 数据
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = 10*(Z2 - Z1)  # 差值高斯函数

    # 创建调色板，定义颜色映射和异常值处理
    palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')
    # 创建掩码数组 Zm，根据条件将 Z 数据进行屏蔽
    Zm = np.ma.masked_where(Z > 1.2, Z)
    # 创建包含两个子图的图像对象
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 在第一个子图上显示屏蔽后的 Zm 数据，使用双线性插值
    im = ax1.imshow(Zm, interpolation='bilinear',
                    cmap=palette,
                    norm=colors.Normalize(vmin=-1.0, vmax=1.0, clip=False),
                    origin='lower', extent=[-3, 3, -3, 3])
    ax1.set_title('Green=low, Red=high, Blue=bad')
    # 在第一个子图上添加水平方向的颜色条
    fig.colorbar(im, extend='both', orientation='horizontal',
                 ax=ax1, aspect=10)

    # 在第二个子图上显示屏蔽后的 Zm 数据，使用最近邻插值和边界标准化
    im = ax2.imshow(Zm, interpolation='nearest',
                    cmap=palette,
                    norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                             ncolors=256, clip=False),
                    origin='lower', extent=[-3, 3, -3, 3])
    ax2.set_title('With BoundaryNorm')
    # 在第二个子图上添加水平方向的颜色条
    fig.colorbar(im, extend='both', spacing='proportional',
                 orientation='horizontal', ax=ax2, aspect=10)


@image_comparison(['mask_image'], remove_text=True)
def test_mask_image():
    # 以两种方式测试掩码图像：使用 NaN 和使用掩码数组

    # 创建包含两个子图的图像对象
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 创建全为 1 的 5x5 数组 A，并将其部分置为 NaN
    A = np.ones((5, 5))
    A[1:2, 1:2] = np.nan
    # 在第一个子图上显示数组 A，使用最近邻插值
    ax1.imshow(A, interpolation='nearest')

    # 创建全为 0 的 5x5 布尔数组 A，并将其部分置为 True
    A = np.zeros((5, 5), dtype=bool)
    A[1:2, 1:2] = True
    # 创建掩码数组 A，使用 16 位无符号整数，并将其转换为掩码数组
    A = np.ma.masked_array(np.ones((5, 5), dtype=np.uint16), A)
    # 在第二个子图上显示掩码数组 A，使用最近邻插值
    ax2.imshow(A, interpolation='nearest')


def test_mask_image_all():
    # 测试完全掩码图像的行为，不会发出警告
    data = np.full((2, 2), np.nan)
    # 创建图像对象和轴对象
    fig, ax = plt.subplots()
    # 在轴上显示数据
    ax.imshow(data)
    # 绘图后强制更新画布，可能会发出警告
    fig.canvas.draw_idle()
    # 创建一个包含整数0到9的NumPy数组
    x = np.arange(10)
    # 使用meshgrid函数生成两个二维数组X和Y，这些数组用于表示平面上的网格点坐标
    X, Y = np.meshgrid(x, x)
    # 计算数组Z中每个点到中心点(5, 5)的欧几里德距离
    Z = np.hypot(X - 5, Y - 5)
    
    # 创建一个包含两个子图(ax1和ax2)的图形对象
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 定义kwargs字典，包含imshow函数的参数：图像原点在左下角、插值方法为最近邻插值、使用'viridis'颜色映射
    kwargs = dict(origin="lower", interpolation='nearest', cmap='viridis')
    
    # 在第一个子图(ax1)上显示Z数组的浮点数表示，使用kwargs中定义的参数
    ax1.imshow(Z.astype('<f8'), **kwargs)
    # 在第二个子图(ax2)上显示Z数组的浮点数表示，使用kwargs中定义的参数
    ax2.imshow(Z.astype('>f8'), **kwargs)
# 使用 image_comparison 装饰器比较函数生成的图像是否与参考图像匹配
@image_comparison(['imshow_masked_interpolation'],
                  tol=0 if platform.machine() == 'x86_64' else 0.01,
                  remove_text=True, style='mpl20')
def test_imshow_masked_interpolation():

    # 选择颜色映射为 'viridis'，并设置极值颜色为红色（over='r'）、蓝色（under='b'）和黑色（bad='k'）
    cmap = mpl.colormaps['viridis'].with_extremes(over='r', under='b', bad='k')

    N = 20
    # 使用 Normalize 类将数据归一化到范围 [0, N*N-1]
    n = colors.Normalize(vmin=0, vmax=N*N-1)

    # 创建一个 N*N 的浮点型数组，元素值从 0 开始递增
    data = np.arange(N*N, dtype=float).reshape(N, N)

    # 将特定位置设为 -1，这会导致高阶插值出现异常振铃
    data[5, 5] = -1
    # 这会导致高阶插值出现异常振铃
    data[15, 5] = 1e5

    # data[3, 3] = np.nan

    # 将特定位置设为无穷大
    data[15, 15] = np.inf

    # 创建一个与 data 形状相同的布尔掩码数组
    mask = np.zeros_like(data).astype('bool')
    mask[5, 15] = True

    # 使用掩码创建掩码数组，用于标记数据中被遮罩的值
    data = np.ma.masked_array(data, mask)

    # 创建包含 3 行 6 列子图的图像，并获取子图数组
    fig, ax_grid = plt.subplots(3, 6)
    # 获取插值方法列表并排序
    interps = sorted(mimage._interpd_)
    # 移除插值方法列表中的 'antialiased'
    interps.remove('antialiased')

    # 遍历插值方法和对应的子图，并在每个子图上绘制经过插值处理的图像
    for interp, ax in zip(interps, ax_grid.ravel()):
        ax.set_title(interp)  # 设置子图标题为插值方法名称
        ax.imshow(data, norm=n, cmap=cmap, interpolation=interp)  # 在子图上绘制图像，使用指定的插值方法
        ax.axis('off')  # 关闭子图坐标轴


# 检查 imshow 函数在显示包含 NaN 值的数组时是否会发出警告
def test_imshow_no_warn_invalid():
    plt.imshow([[1, 2], [3, np.nan]])  # 检查是否有警告被发出。


# 使用 pytest.mark.parametrize 装饰器执行多组测试参数，测试不同类型的数据在显示时是否正确裁剪到有效范围内
@pytest.mark.parametrize(
    'dtype', [np.dtype(s) for s in 'u2 u4 i2 i4 i8 f4 f8'.split()])
def test_imshow_clips_rgb_to_valid_range(dtype):
    # 创建指定类型的数据数组
    arr = np.arange(300, dtype=dtype).reshape((10, 10, 3))
    if dtype.kind != 'u':
        arr -= 10  # 如果数据类型不是无符号整数类型，则减去 10
    too_low = arr < 0  # 找出低于 0 的元素索引
    too_high = arr > 255  # 找出高于 255 的元素索引
    if dtype.kind == 'f':
        arr = arr / 255  # 如果数据类型是浮点数类型，则将数组归一化到 [0, 1] 范围内
    _, ax = plt.subplots()
    out = ax.imshow(arr).get_array()  # 在子图上绘制数组并获取绘制结果
    assert (out[too_low] == 0).all()  # 断言所有低于 0 的元素被裁剪为 0
    if dtype.kind == 'f':
        assert (out[too_high] == 1).all()  # 断言所有高于 255 的元素被裁剪为 1
        assert out.dtype.kind == 'f'  # 断言输出数组数据类型为浮点数
    else:
        assert (out[too_high] == 255).all()  # 断言所有高于 255 的元素被裁剪为 255
        assert out.dtype == np.uint8  # 断言输出数组数据类型为无符号整数类型


# 使用 image_comparison 装饰器比较函数生成的图像是否与参考图像匹配
@image_comparison(['imshow_flatfield.png'], remove_text=True, style='mpl20')
def test_imshow_flatfield():
    fig, ax = plt.subplots()
    im = ax.imshow(np.ones((5, 5)), interpolation='nearest')
    im.set_clim(.5, 1.5)  # 设置图像的颜色限制范围为 [0.5, 1.5]


# 使用 image_comparison 装饰器比较函数生成的图像是否与参考图像匹配
@image_comparison(['imshow_bignumbers.png'], remove_text=True, style='mpl20')
def test_imshow_bignumbers():
    rcParams['image.interpolation'] = 'nearest'
    # 在整数数组中放置一个大数值不应该破坏分辨率的动态范围
    fig, ax = plt.subplots()
    img = np.array([[1, 2, 1e12], [3, 1, 4]], dtype=np.uint64)
    pc = ax.imshow(img)
    pc.set_clim(0, 5)  # 设置图像的颜色限制范围为 [0, 5]


# 使用 image_comparison 装饰器比较函数生成的图像是否与参考图像匹配
@image_comparison(['imshow_bignumbers_real.png'],
                  remove_text=True, style='mpl20')
def test_imshow_bignumbers_real():
    rcParams['image.interpolation'] = 'nearest'
    # 在浮点数数组中放置一个大数值不应该破坏分辨率的动态范围
    fig, ax = plt.subplots()
    img = np.array([[2., 1., 1.e22], [4., 1., 3.]])
    pc = ax.imshow(img)
    pc.set_clim(0, 5)  # 设置图像的颜色限制范围为 [0, 5]


# 使用 pytest.mark.parametrize 装饰器执行多组测试参数，测试不同的 Normalize 函数实例
@pytest.mark.parametrize(
    "make_norm",
    [colors.Normalize,
     colors.LogNorm,
     lambda: colors.SymLogNorm(1),
     lambda: colors.PowerNorm(1)])
def test_empty_imshow(make_norm):
    # 测试 imshow 函数在不同的 Normalize 函数实例下是否正常工作
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 使用 pytest 中的 warns 上下文管理器捕获 UserWarning，当匹配特定字符串时发出警告
    with pytest.warns(UserWarning, match="Attempting to set identical low and high xlims"):
        # 在轴上显示一个空的图像，使用自定义的归一化函数
        im = ax.imshow([[]], norm=make_norm())
    # 设置图像的显示范围为 [-5, 5, -5, 5]
    im.set_extent([-5, 5, -5, 5])
    # 绘制图形上下文的画布
    fig.canvas.draw()
    
    # 使用 pytest 中的 raises 上下文管理器捕获 RuntimeError 异常
    with pytest.raises(RuntimeError):
        # 创建图像以备在给定的画布渲染器上绘制
        im.make_image(fig.canvas.get_renderer())
def test_imshow_float16():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 在轴上显示一个 dtype 为 np.float16 的全零矩阵
    ax.imshow(np.zeros((3, 3), dtype=np.float16))
    # 确保绘图不会导致崩溃。
    fig.canvas.draw()


def test_imshow_float128():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 在轴上显示一个 dtype 为 np.longdouble 的全零矩阵
    ax.imshow(np.zeros((3, 3), dtype=np.longdouble))
    with (ExitStack() if np.can_cast(np.longdouble, np.float64, "equiv")
          else pytest.warns(UserWarning)):
        # 确保绘图不会导致崩溃。
        fig.canvas.draw()


def test_imshow_bool():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 在轴上显示一个布尔类型的数组
    ax.imshow(np.array([[True, False], [False, True]], dtype=bool))


def test_full_invalid():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 在轴上显示一个全是 NaN 的矩阵
    ax.imshow(np.full((10, 10), np.nan))
    # 确保绘图不会导致崩溃。
    fig.canvas.draw()


@pytest.mark.parametrize("fmt,counted",
                         [("ps", b" colorimage"), ("svg", b"<image")])
@pytest.mark.parametrize("composite_image,count", [(True, 1), (False, 2)])
def test_composite(fmt, counted, composite_image, count):
    # 测试可以保存带有合成图像和不带合成图像的图形。
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)

    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    # 在轴上显示图像 Z，并指定其范围
    ax.imshow(Z, extent=[0, 1, 0, 1])
    ax.imshow(Z[::-1], extent=[2, 3, 0, 1])
    # 设置图像参数中的合成图像属性
    plt.rcParams['image.composite_image'] = composite_image
    buf = io.BytesIO()
    # 将图形保存为指定格式的图像文件
    fig.savefig(buf, format=fmt)
    # 断言保存的图像文件中包含特定的计数项
    assert buf.getvalue().count(counted) == count


def test_relim():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 在轴上显示一个矩阵 [[0]]
    ax.imshow([[0]], extent=(0, 1, 0, 1))
    # 重新计算轴的数据限制
    ax.relim()
    # 自动调整轴的显示范围
    ax.autoscale()
    # 断言轴的 x 和 y 轴限制是否为 (0, 1)
    assert ax.get_xlim() == ax.get_ylim() == (0, 1)


def test_unclipped():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 设置轴的不显示
    ax.set_axis_off()
    # 在轴上显示一个灰度图像，不进行裁剪
    im = ax.imshow([[0, 0], [0, 0]], aspect="auto", extent=(-10, 10, -10, 10),
                   cmap='gray', clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    # 断言未裁剪的图像是否填充整个图形并且是黑色
    assert (np.array(fig.canvas.buffer_rgba())[..., :3] == 0).all()


def test_respects_bbox():
    # 创建一个包含 2 个轴的图形
    fig, axs = plt.subplots(2)
    for ax in axs:
        # 设置轴不显示
        ax.set_axis_off()
    # 在 axs[1] 上显示一个图像，并根据 axs[0] 的 bbox 设置裁剪框
    im = axs[1].imshow([[0, 1], [2, 3]], aspect="auto", extent=(0, 1, 0, 1))
    im.set_clip_path(None)
    buf_before = io.BytesIO()
    # 将图形保存为 rgba 格式的图像文件
    fig.savefig(buf_before, format="rgba")
    # 断言保存前的图像文件是否全白
    assert {*buf_before.getvalue()} == {0xff}  # All white.
    # 移动 axs[1] 的限制范围
    axs[1].set(ylim=(-1, 0))
    buf_after = io.BytesIO()
    # 再次保存图形为 rgba 格式的图像文件
    fig.savefig(buf_after, format="rgba")
    # 断言保存后的图像文件和保存前的不同
    assert buf_before.getvalue() != buf_after.getvalue()  # Not all white.


def test_image_cursor_formatting():
    # 创建一个包含图形和轴的新子图
    fig, ax = plt.subplots()
    # 创建一个用于调用 format_cursor_data 的虚拟图像
    im = ax.imshow(np.zeros((4, 4)))

    data = np.ma.masked_array([0], mask=[True])
    # 断言：验证给定的数据在调用 format_cursor_data 函数后返回的结果是否为 '[]'
    assert im.format_cursor_data(data) == '[]'
    
    # 创建一个带有单个未屏蔽元素的屏蔽数组，并断言调用 format_cursor_data 函数后返回的结果是否为 '[0]'
    data = np.ma.masked_array([0], mask=[False])
    assert im.format_cursor_data(data) == '[0]'
    
    # 将 data 设为 NaN（Not a Number），并断言调用 format_cursor_data 函数后返回的结果是否为 '[nan]'
    data = np.nan
    assert im.format_cursor_data(data) == '[nan]'
# 使用装饰器 @check_figures_equal() 注册的测试函数，用于比较两个图像对象的像素数据是否相等
@check_figures_equal()
def test_image_array_alpha(fig_test, fig_ref):
    """Per-pixel alpha channel test."""
    # 创建一个从0到1的等差数列
    x = np.linspace(0, 1)
    # 创建二维网格
    xx, yy = np.meshgrid(x, x)
    
    # 根据给定的函数生成二维数组 zz
    zz = np.exp(- 3 * ((xx - 0.5) ** 2) + (yy - 0.7 ** 2))
    # 计算每个像素点的 alpha 值，使其归一化
    alpha = zz / zz.max()
    
    # 获取 'viridis' 颜色映射
    cmap = mpl.colormaps['viridis']
    # 在 fig_test 中添加子图
    ax = fig_test.add_subplot()
    # 显示 zz 数组的图像，设置透明度为 alpha，颜色映射为 cmap，插值方式为最近邻
    ax.imshow(zz, alpha=alpha, cmap=cmap, interpolation='nearest')

    # 在 fig_ref 中添加子图
    ax = fig_ref.add_subplot()
    # 使用颜色映射 cmap 根据归一化后的 zz 数组生成 RGBA 数组 rgba
    rgba = cmap(colors.Normalize()(zz))
    # 将 RGBA 数组的 alpha 通道设置为 alpha 数组的值
    rgba[..., -1] = alpha
    # 显示 rgba 数组的图像，插值方式为最近邻
    ax.imshow(rgba, interpolation='nearest')


# 测试函数，验证 plt.imshow() 函数对 alpha 参数的类型检查
def test_image_array_alpha_validation():
    # 使用 pytest 检测是否会抛出 TypeError 异常，异常信息需包含 "alpha must be a float, two-d"
    with pytest.raises(TypeError, match="alpha must be a float, two-d"):
        plt.imshow(np.zeros((2, 2)), alpha=[1, 1])


# 使用 mpl.style.context('mpl20') 设置的风格上下文装饰器，对图形进行风格设置
@mpl.style.context('mpl20')
def test_exact_vmin():
    # 复制 'autumn_r' 颜色映射
    cmap = copy(mpl.colormaps["autumn_r"])
    # 设置 'under' 颜色为 "lightgrey"
    cmap.set_under(color="lightgrey")
    
    # 创建图形对象 fig，确保图像宽度为 190 像素，dpi 为 100
    fig = plt.figure(figsize=(1.9, 0.1), dpi=100)
    # 在图形上添加坐标轴，位置为整个图形区域
    ax = fig.add_axes([0, 0, 1, 1])

    # 创建一个特定数据的 numpy 数组
    data = np.array(
        [[-1, -1, -1, 0, 0, 0, 0, 43, 79, 95, 66, 1, -1, -1, -1, 0, 0, 0, 34]],
        dtype=float,
    )

    # 在坐标轴 ax 上显示数据的图像，设置长宽比为 "auto"，颜色映射为 cmap，最小值 vmin=0，最大值 vmax=100
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    # 关闭坐标轴显示
    ax.axis("off")
    # 绘制图形对象 fig 的画布
    fig.canvas.draw()

    # 从图像中获取 RGBA 数据切片
    from_image = im.make_image(fig.canvas.renderer)[0][0]
    # 将输入数据扩展为 190 长并通过 norm / cmap 处理
    direct_computation = (
        im.cmap(im.norm((data * ([[1]] * 10)).T.ravel())) * 255
    ).astype(int)

    # 检查 RGBA 值是否完全相同
    assert np.all(from_image == direct_computation)


# 使用 image_comparison 装饰器注册的测试函数，对图像的布局进行测试，比较生成的图像与参考图像的差异
@image_comparison(['image_placement'], extensions=['svg', 'pdf'],
                  remove_text=True, style='mpl20')
def test_image_placement():
    """
    The red box should line up exactly with the outside of the image.
    """
    # 创建图形对象 fig 和坐标轴 ax
    fig, ax = plt.subplots()
    # 在坐标轴上绘制红色盒子，指定顶点坐标和边界线宽度
    ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color='r', lw=0.1)
    # 设置随机种子，确保生成相同的随机数据
    np.random.seed(19680801)
    # 在坐标轴上显示指定大小和范围的随机数据图像，颜色映射为 'Blues'
    ax.imshow(np.random.randn(16, 16), cmap='Blues', extent=(0, 1, 0, 1),
              interpolation='none', vmin=-1, vmax=1)
    # 设置坐标轴的 x 和 y 范围
    ax.set_xlim(-0.1, 1+0.1)
    ax.set_ylim(-0.1, 1+0.1)


# 一个基本的 ndarray 子类，实现了数量的概念
# 没有实现完整的单位系统或所有数量的数学运算。
# 只实现了足够的内容来测试 ndarray 的处理。
class QuantityND(np.ndarray):
    def __new__(cls, input_array, units):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        self.units = getattr(obj, "units", None)

    def __getitem__(self, item):
        units = getattr(self, "units", None)
        ret = super().__getitem__(item)
        if isinstance(ret, QuantityND) or units is not None:
            ret = QuantityND(ret, units)
        return ret
    # 实现数组对象对通用函数（ufunc）的特殊方法，以处理各种数学运算
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 获取指定的ufunc方法，如add、subtract等
        func = getattr(ufunc, method)
        # 如果在kwargs中指定了输出(out)，则返回NotImplemented
        if "out" in kwargs:
            return NotImplemented
        # 如果只有一个输入数组
        if len(inputs) == 1:
            i0 = inputs[0]
            # 获取第一个输入数组的单位，如果没有定义单位则默认为"dimensionless"
            unit = getattr(i0, "units", "dimensionless")
            # 对输入数组进行转换为ndarray，并调用ufunc的函数进行计算
            out_arr = func(np.asarray(i0), **kwargs)
        # 如果有两个输入数组
        elif len(inputs) == 2:
            i0 = inputs[0]
            i1 = inputs[1]
            # 获取两个输入数组的单位，如果其中一个没有定义单位则默认为"dimensionless"
            u0 = getattr(i0, "units", "dimensionless")
            u1 = getattr(i1, "units", "dimensionless")
            # 处理单位的兼容性
            u0 = u1 if u0 is None else u0
            u1 = u0 if u1 is None else u1
            # 根据ufunc类型确定输出的单位
            if ufunc in [np.add, np.subtract]:
                # 对于加法和减法，要求两个输入数组的单位必须相同，否则抛出ValueError
                if u0 != u1:
                    raise ValueError
                unit = u0
            elif ufunc == np.multiply:
                # 对于乘法，输出单位为两个输入单位的乘积形式
                unit = f"{u0}*{u1}"
            elif ufunc == np.divide:
                # 对于除法，输出单位为两个输入单位的比率形式
                unit = f"{u0}/({u1})"
            elif ufunc in (np.greater, np.greater_equal,
                           np.equal, np.not_equal,
                           np.less, np.less_equal):
                # 对比较操作，输出为无单位的布尔值
                unit = None
            else:
                # 对于其他类型的ufunc，返回NotImplemented
                return NotImplemented
            # 调用ufunc函数计算两个输入数组的结果
            out_arr = func(i0.view(np.ndarray), i1.view(np.ndarray), **kwargs)
        else:
            # 如果输入的数组个数不是1或2，则返回NotImplemented
            return NotImplemented
        # 根据计算结果的单位情况进行处理，如果没有单位则转换为普通ndarray，否则创建带单位的QuantityND对象
        if unit is None:
            out_arr = np.array(out_arr)
        else:
            out_arr = QuantityND(out_arr, unit)
        # 返回计算结果数组
        return out_arr

    # 定义属性v，返回当前对象的ndarray视图
    @property
    def v(self):
        return self.view(np.ndarray)
def test_quantitynd():
    # 创建一个 QuantityND 对象，数据为 [1, 2]，单位为 "m"
    q = QuantityND([1, 2], "m")
    # 将 QuantityND 对象 q 解包为 q0 和 q1
    q0, q1 = q[:]
    # 断言：q 的值与 [1, 2] 相等
    assert np.all(q.v == np.asarray([1, 2]))
    # 断言：q 的单位为 "m"
    assert q.units == "m"
    # 断言：q0 和 q1 的和为 [3]
    assert np.all((q0 + q1).v == np.asarray([3]))
    # 断言：q0 与 q1 的乘积的单位为 "m*m"
    assert (q0 * q1).units == "m*m"
    # 断言：q1 除以 q0 的单位为 "m/(m)"
    assert (q1 / q0).units == "m/(m)"
    # 断言：使用 pytest 应该会抛出 ValueError 异常
    with pytest.raises(ValueError):
        q0 + QuantityND(1, "s")


def test_imshow_quantitynd():
    # 创建一个 QuantityND 对象 arr，数据为全 1 的 2x2 数组，单位为 "m"
    arr = QuantityND(np.ones((2, 2)), "m")
    # 创建一个图形 fig 和坐标轴 ax
    fig, ax = plt.subplots()
    # 在坐标轴上显示 arr
    ax.imshow(arr)
    # 执行绘图操作不应该引发异常
    fig.canvas.draw()


@check_figures_equal(extensions=['png'])
def test_norm_change(fig_test, fig_ref):
    # 创建一个全 1 的 5x5 浮点数数据数组 data
    data = np.full((5, 5), 1, dtype=np.float64)
    # 将部分数据设为 -1
    data[0:2, :] = -1

    # 创建一个不带掩码的掩码数组 masked_data
    masked_data = np.ma.array(data, mask=False)
    # 将部分数据设为掩码
    masked_data.mask[0:2, 0:2] = True

    # 获取 "viridis" 颜色映射并设置极值
    cmap = mpl.colormaps['viridis'].with_extremes(under='w')

    # 创建测试图形 fig_test 的坐标轴 ax，并在上面显示 data
    ax = fig_test.subplots()
    im = ax.imshow(data, norm=colors.LogNorm(vmin=0.5, vmax=1),
                   extent=(0, 5, 0, 5), interpolation='nearest', cmap=cmap)
    # 设置 im 的归一化方式为指定范围内的 Normalize 对象
    im.set_norm(colors.Normalize(vmin=-2, vmax=2))
    # 在坐标轴上显示 masked_data
    im = ax.imshow(masked_data, norm=colors.LogNorm(vmin=0.5, vmax=1),
                   extent=(5, 10, 5, 10), interpolation='nearest', cmap=cmap)
    # 设置 im 的归一化方式为指定范围内的 Normalize 对象
    im.set_norm(colors.Normalize(vmin=-2, vmax=2))
    # 设置坐标轴的限制
    ax.set(xlim=(0, 10), ylim=(0, 10))

    # 创建参考图形 fig_ref 的坐标轴 ax，并在上面显示 data 和 masked_data
    ax = fig_ref.subplots()
    ax.imshow(data, norm=colors.Normalize(vmin=-2, vmax=2),
              extent=(0, 5, 0, 5), interpolation='nearest', cmap=cmap)
    ax.imshow(masked_data, norm=colors.Normalize(vmin=-2, vmax=2),
              extent=(5, 10, 5, 10), interpolation='nearest', cmap=cmap)
    # 设置坐标轴的限制
    ax.set(xlim=(0, 10), ylim=(0, 10))


@pytest.mark.parametrize('x', [-1, 1])
@check_figures_equal(extensions=['png'])
def test_huge_range_log(fig_test, fig_ref, x):
    # parametrize over bad lognorm -1 values and large range 1 -> 1e20
    # 创建一个全为 x 的 5x5 浮点数数据数组 data
    data = np.full((5, 5), x, dtype=np.float64)
    # 将部分数据设为 1E20
    data[0:2, :] = 1E20

    # 创建测试图形 fig_test 的坐标轴 ax，并在上面显示 data
    ax = fig_test.subplots()
    ax.imshow(data, norm=colors.LogNorm(vmin=1, vmax=data.max()),
              interpolation='nearest', cmap='viridis')

    # 创建一个全为 x 的 5x5 浮点数数据数组 data
    data = np.full((5, 5), x, dtype=np.float64)
    data[0:2, :] = 1000

    # 创建参考图形 fig_ref 的坐标轴 ax，并在上面显示 data
    cmap = mpl.colormaps['viridis'].with_extremes(under='w')
    ax = fig_ref.subplots()
    ax.imshow(data, norm=colors.Normalize(vmin=1, vmax=data.max()),
              interpolation='nearest', cmap=cmap)


@check_figures_equal()
def test_spy_box(fig_test, fig_ref):
    # 设置测试和参考的坐标轴
    ax_test = fig_test.subplots(1, 3)
    ax_ref = fig_ref.subplots(1, 3)

    # 设置绘图数据和标题
    plot_data = (
        [[1, 1], [1, 1]],
        [[0, 0], [0, 0]],
        [[0, 1], [1, 0]],
    )
    plot_titles = ["ones", "zeros", "mixed"]
    # 对于给定的数据和标题列表，使用循环枚举每个元素
    for i, (z, title) in enumerate(zip(plot_data, plot_titles)):
        # 设置当前子图 ax_test[i] 的标题为给定的标题
        ax_test[i].set_title(title)
        # 在当前子图 ax_test[i] 上绘制稀疏矩阵 z 的视图
        ax_test[i].spy(z)
        # 设置当前子图 ax_ref[i] 的标题为给定的标题
        ax_ref[i].set_title(title)
        # 在当前子图 ax_ref[i] 上显示矩阵 z 的图像，使用最近邻插值，灰度颜色映射
        ax_ref[i].imshow(z, interpolation='nearest',
                            aspect='equal', origin='upper', cmap='Greys',
                            vmin=0, vmax=1)
        # 设置当前子图 ax_ref[i] 的 x 轴范围
        ax_ref[i].set_xlim(-0.5, 1.5)
        # 设置当前子图 ax_ref[i] 的 y 轴范围
        ax_ref[i].set_ylim(1.5, -0.5)
        # 设置当前子图 ax_ref[i] 的 x 轴刻度位置在顶部
        ax_ref[i].xaxis.tick_top()
        # 设置当前子图 ax_ref[i] 的标题向上移动到指定位置
        ax_ref[i].title.set_y(1.05)
        # 设置当前子图 ax_ref[i] 的 x 轴刻度线在上下都显示
        ax_ref[i].xaxis.set_ticks_position('both')
        # 设置当前子图 ax_ref[i] 的 x 轴主刻度定位器为最大刻度定位器，使用指定的步长
        ax_ref[i].xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
        )
        # 设置当前子图 ax_ref[i] 的 y 轴主刻度定位器为最大刻度定位器，使用指定的步长
        ax_ref[i].yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
        )
@image_comparison(["nonuniform_and_pcolor.png"], style="mpl20")
def test_nonuniform_and_pcolor():
    # 创建一个大小为 3x3 的图形，并共享坐标轴
    axs = plt.figure(figsize=(3, 3)).subplots(3, sharex=True, sharey=True)
    # 遍历每个子图和插值方式
    for ax, interpolation in zip(axs, ["nearest", "bilinear"]):
        # 在每个子图上创建非均匀图像对象
        im = NonUniformImage(ax, interpolation=interpolation)
        # 设置图像数据为指定的 x, y, z 值
        im.set_data(np.arange(3) ** 2, np.arange(3) ** 2,
                    np.arange(9).reshape((3, 3)))
        # 将非均匀图像添加到子图中
        ax.add_image(im)
    # 在最后一个子图上绘制快速着色，使用 PcolorImage
    axs[2].pcolorfast(
        np.arange(4) ** 2, np.arange(4) ** 2, np.arange(9).reshape((3, 3)))
    # 设置每个子图的坐标轴关闭
    for ax in axs:
        ax.set_axis_off()
        # 针对非均匀图像，注意其可能会超出边界，而 PColorImage 不会。
        ax.set(xlim=(0, 10))


@image_comparison(["nonuniform_logscale.png"], style="mpl20")
def test_nonuniform_logscale():
    # 创建包含 3 个列的子图布局
    _, axs = plt.subplots(ncols=3, nrows=1)
    # 遍历每个子图
    for i in range(3):
        ax = axs[i]
        # 在每个子图上创建非均匀图像对象
        im = NonUniformImage(ax)
        # 设置图像数据为指定的 x, y, z 值
        im.set_data(np.arange(1, 4) ** 2, np.arange(1, 4) ** 2,
                    np.arange(9).reshape((3, 3)))
        # 设置子图的 x 和 y 轴限制
        ax.set_xlim(1, 16)
        ax.set_ylim(1, 16)
        # 设置子图的盒子纵横比
        ax.set_box_aspect(1)
        # 如果是第二个子图，设置 x 和 y 轴为对数尺度（基数为2）
        if i == 1:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
        # 如果是第三个子图，设置 x 和 y 轴为对数尺度（基数为4）
        if i == 2:
            ax.set_xscale("log", base=4)
            ax.set_yscale("log", base=4)
        # 将非均匀图像添加到子图中
        ax.add_image(im)


@image_comparison(
    ['rgba_antialias.png'], style='mpl20', remove_text=True,
    tol=0 if platform.machine() == 'x86_64' else 0.007)
def test_rgba_antialias():
    # 创建一个包含 2x2 子图的图形，具有指定的尺寸和共享属性
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5), sharex=False,
                            sharey=False, constrained_layout=True)
    N = 250
    aa = np.ones((N, N))
    aa[::2, :] = -1

    x = np.arange(N) / N - 0.5
    y = np.arange(N) / N - 0.5

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    f0 = 10
    k = 75
    # 创建锯齿状同心圆
    a = np.sin(np.pi * 2 * (f0 * R + k * R**2 / 2))

    # 在左侧创建条纹
    a[:int(N/2), :][R[:int(N/2), :] < 0.4] = -1
    a[:int(N/2), :][R[:int(N/2), :] < 0.3] = 1
    aa[:, int(N/2):] = a[:, int(N/2):]

    # 设置一些超限和 NaN 值
    aa[20:50, 20:50] = np.nan
    aa[70:90, 70:90] = 1e6
    aa[70:90, 20:30] = -1e6
    aa[70:90, 195:215] = 1e6
    aa[20:30, 195:215] = -1e6

    # 复制彩色映射，并设置上下限值
    cmap = copy(plt.cm.RdBu_r)
    cmap.set_over('yellow')
    cmap.set_under('cyan')

    # 将子图数组展平
    axs = axs.flatten()
    # 在第一个子图上显示锯齿效果，使用最近邻插值，指定颜色映射和范围
    axs[0].imshow(aa, interpolation='nearest', cmap=cmap, vmin=-1.2, vmax=1.2)
    axs[0].set_xlim([N/2-25, N/2+25])
    axs[0].set_ylim([N/2+50, N/2-10])

    # 在第二个子图上显示锯齿效果，使用最近邻插值，指定颜色映射和范围
    axs[1].imshow(aa, interpolation='nearest', cmap=cmap, vmin=-1.2, vmax=1.2)

    # 在第三个子图上显示锯齿效果，使用数据抗锯齿插值，指定颜色映射和范围
    axs[2].imshow(aa, interpolation='antialiased', interpolation_stage='data',
                  cmap=cmap, vmin=-1.2, vmax=1.2)

    # 在第四个子图上显示 RGBA 抗锯齿效果
    # 注意：边界处显示紫色，并且交替的红蓝条纹变为白色。
    # 在第四个轴上显示图像
    axs[3].imshow(aa, interpolation='antialiased', interpolation_stage='rgba',
                  cmap=cmap, vmin=-1.2, vmax=1.2)
    # 使用抗锯齿插值方式 ('antialiased') 和 RGBA 插值阶段 ('rgba') 显示图像
    # cmap 参数指定颜色映射
    # vmin 和 vmax 参数指定显示数据的值范围
# 定义测试函数，用于测试 rc_interpolation_stage 功能
def test_rc_interpolation_stage():
    # 遍历两种值："data" 和 "rgba"
    for val in ["data", "rgba"]:
        # 创建临时的 rc_context 上下文管理器，设置 image.interpolation_stage 为 val
        with mpl.rc_context({"image.interpolation_stage": val}):
            # 断言调用 plt.imshow([[1, 2]]).get_interpolation_stage() 返回的值等于 val
            assert plt.imshow([[1, 2]]).get_interpolation_stage() == val
    
    # 遍历三种不允许的值："DATA", "foo", None
    for val in ["DATA", "foo", None]:
        # 使用 pytest.raises 检查是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 将 mpl.rcParams["image.interpolation_stage"] 设为 val
            mpl.rcParams["image.interpolation_stage"] = val


# 标记为忽略特定警告，用于测试大尺寸图像处理功能
@pytest.mark.filterwarnings(r'ignore:Data with more than .* '
                            'cannot be accurately displayed')
# 参数化测试：origin 参数取值 'upper' 和 'lower'
@pytest.mark.parametrize('origin', ['upper', 'lower'])
# 参数化测试：dim, size, msg 取值分别为 ['row', 2**23, r'2\*\*23 columns'] 和 ['col', 2**24, r'2\*\*24 rows']
@pytest.mark.parametrize(
    'dim, size, msg', [['row', 2**23, r'2\*\*23 columns'],
                       ['col', 2**24, r'2\*\*24 rows']])
# 使用装饰器检查图形是否相等，输出文件格式为 'png'
@check_figures_equal(extensions=('png', ))
def test_large_image(fig_test, fig_ref, dim, size, msg, origin):
    # 检查 Matplotlib 是否能够正确处理太大的图像（用于 AGG 渲染引擎）
    # 参见问题编号 #19276。目前只修复了 PNG 输出，但未修复 PDF 或 SVG 输出。
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    # 创建一个大小为 (1, size+2) 的零矩阵
    array = np.zeros((1, size + 2))
    # 在矩阵右半部分赋值为 1
    array[:, array.size // 2:] = 1
    # 如果 dim 为 'col'，则转置矩阵
    if dim == 'col':
        array = array.T
    # 在 ax_test 上绘制图像，设置 vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1), interpolation='none', origin=origin
    im = ax_test.imshow(array, vmin=0, vmax=1,
                        aspect='auto', extent=(0, 1, 0, 1),
                        interpolation='none',
                        origin=origin)

    # 使用 pytest.warns 检查是否发出 UserWarning 警告，匹配特定的警告消息
    with pytest.warns(UserWarning,
                      match=f'Data with more than {msg} cannot be '
                      'accurately displayed.'):
        # 绘制 fig_test 的画布
        fig_test.canvas.draw()

    # 创建一个大小为 (1, 2) 的零矩阵
    array = np.zeros((1, 2))
    # 在矩阵第二列赋值为 1
    array[:, 1] = 1
    # 如果 dim 为 'col'，则转置矩阵
    if dim == 'col':
        array = array.T
    # 在 ax_ref 上绘制图像，设置 vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1), interpolation='none', origin=origin
    im = ax_ref.imshow(array, vmin=0, vmax=1, aspect='auto',
                       extent=(0, 1, 0, 1),
                       interpolation='none',
                       origin=origin)


# 使用装饰器检查图形是否相等，输出文件格式为 'png'
@check_figures_equal(extensions=["png"])
def test_str_norms(fig_test, fig_ref):
    # 创建一个 10x10 的随机数矩阵 t，数值范围在 0.1 到 0.9 之间
    t = np.random.rand(10, 10) * .8 + .1  # between 0 and 1
    # 在 fig_test 上创建包含 5 个子图的坐标轴数组 axts
    axts = fig_test.subplots(1, 5)
    # 在 axts[0] 上绘制 t，使用 norm="log"
    axts[0].imshow(t, norm="log")
    # 在 axts[1] 上绘制 t，使用 norm="log"，设置 vmin=0.2
    axts[1].imshow(t, norm="log", vmin=.2)
    # 在 axts[2] 上绘制 t，使用 norm="symlog"
    axts[2].imshow(t, norm="symlog")
    # 在 axts[3] 上绘制 t，使用 norm="symlog"，设置 vmin=0.3, vmax=0.7
    axts[3].imshow(t, norm="symlog", vmin=.3, vmax=.7)
    # 在 axts[4] 上绘制 t，使用 norm="logit"，设置 vmin=0.3, vmax=0.7
    axts[4].imshow(t, norm="logit", vmin=.3, vmax=.7)
    
    # 在 fig_ref 上创建包含 5 个子图的坐标轴数组 axrs
    axrs = fig_ref.subplots(1, 5)
    # 在 axrs[0] 上绘制 t，使用 norm=colors.LogNorm()
    axrs[0].imshow(t, norm=colors.LogNorm())
    # 在 axrs[1] 上绘制 t，使用 norm=colors.LogNorm(vmin=0.2)
    axrs[1].imshow(t, norm=colors.LogNorm(vmin=.2))
    # 在 axrs[2] 上绘制 t，使用 norm=colors.SymLogNorm(linthresh=2)
    axrs[2].imshow(t, norm=colors.SymLogNorm(linthresh=2))
    # 在 axrs[3] 上绘制 t，使用 norm=colors.SymLogNorm(linthresh=2, vmin=0.3, vmax=0.7)
    axrs[3].imshow(t, norm=colors.SymLogNorm(linthresh=2, vmin=.3, vmax=.7))
    # 在 axrs[4] 上绘制 t，使用 norm="logit"，设置 clim=(0.3, 0.7)
    axrs[4].imshow(t, norm="logit", clim=(.3, .7))
    
    # 断言 axts[0] 的第一个图像的 norm 属性的类型为 colors.LogNorm
    assert type(axts[0].images[0].norm) is colors.LogNorm  # Exactly that class
    # 使用 pytest.raises 检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 在 axts[0] 上绘制 t，使用 norm="foobar"，引发异常
        axts[0].imshow(t, norm="foobar")


# 测试函数，用于测试 _resample_valid_output 函数
def test__resample_valid_output():
    # 创建 functools.partial 对象 resample，调用 mpl._image.resample 函数并传入 Affine2D() 作为 transform 参数
    resample = functools.partial(mpl._image.resample, transform=Affine2D())
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 TypeError 异常，并匹配特定的错误信息
    with pytest.raises(TypeError, match="incompatible function arguments"):
        resample(np.zeros((9, 9)), None)
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="different dimensionalities"):
        resample(np.zeros((9, 9)), np.zeros((9, 9, 4)))
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="different dimensionalities"):
        resample(np.zeros((9, 9, 4)), np.zeros((9, 9)))
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="3D input array must be RGBA"):
        resample(np.zeros((9, 9, 3)), np.zeros((9, 9, 4)))
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="3D output array must be RGBA"):
        resample(np.zeros((9, 9, 4)), np.zeros((9, 9, 3)))
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="mismatched types"):
        resample(np.zeros((9, 9), np.uint8), np.zeros((9, 9)))
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="must be C-contiguous"):
        resample(np.zeros((9, 9)), np.zeros((9, 9)).T)

    # 创建一个形状为 (9, 9) 的全零数组 out，并将其设为不可写
    out = np.zeros((9, 9))
    out.flags.writeable = False
    
    # 使用 pytest 来测试 resample 函数是否能够正确抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="Output array must be writeable"):
        resample(np.zeros((9, 9)), out)
# 定义测试函数，用于测试 AxesImage 类的 get_shape 方法
def test_axesimage_get_shape():
    # 获取当前轴对象
    ax = plt.gca()
    # 创建一个 AxesImage 对象，用于测试
    im = AxesImage(ax)
    # 使用 pytest 检查是否会引发 RuntimeError 异常，并验证异常信息
    with pytest.raises(RuntimeError, match="You must first set the image array"):
        im.get_shape()
    # 创建一个 4x3 的浮点数数组 z
    z = np.arange(12, dtype=float).reshape((4, 3))
    # 设置 AxesImage 对象的数据
    im.set_data(z)
    # 断言获取的形状为 (4, 3)
    assert im.get_shape() == (4, 3)
    # 断言获取的大小与形状相同
    assert im.get_size() == im.get_shape()


# 定义测试函数，验证非 transdata 形式的图像不会改变纵横比
def test_non_transdata_image_does_not_touch_aspect():
    # 在一个新的图表中添加子图
    ax = plt.figure().add_subplot()
    # 创建一个2x2的数组 im
    im = np.arange(4).reshape((2, 2))
    # 在轴上显示图像，使用 ax.transAxes 进行变换
    ax.imshow(im, transform=ax.transAxes)
    # 断言轴的纵横比为 "auto"
    assert ax.get_aspect() == "auto"
    # 使用 Affine2D().scale(2) + ax.transData 变换后再次显示图像
    ax.imshow(im, transform=Affine2D().scale(2) + ax.transData)
    # 断言轴的纵横比为 1
    assert ax.get_aspect() == 1
    # 在轴上显示图像，指定 aspect 为 2
    ax.imshow(im, transform=ax.transAxes, aspect=2)
    # 断言轴的纵横比为 2
    assert ax.get_aspect() == 2


# 使用参数化测试，测试不同数据类型和维度的图像重采样
@pytest.mark.parametrize(
    'dtype',
    ('float64', 'float32', 'int16', 'uint16', 'int8', 'uint8'),
)
@pytest.mark.parametrize('ndim', (2, 3))
def test_resample_dtypes(dtype, ndim):
    # 解决 Issue 28448：在 C++ 图像重采样中，不正确的 dtype 比较可能引发 ValueError
    # 使用默认随机数生成器创建 rng 对象
    rng = np.random.default_rng(4181)
    # 根据 ndim 创建对应形状的随机数数组，将其转换为指定的 dtype
    shape = (2, 2) if ndim == 2 else (2, 2, 3)
    data = rng.uniform(size=shape).astype(np.dtype(dtype, copy=True))
    # 创建图表和轴对象
    fig, ax = plt.subplots()
    # 在轴上显示数据
    axes_image = ax.imshow(data)
    # 在修复前，以下代码对某些 dtype 可能会引发 ValueError
    axes_image.make_image(None)[0]
```