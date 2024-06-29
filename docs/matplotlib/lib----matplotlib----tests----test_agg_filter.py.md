# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_agg_filter.py`

```
@image_comparison(baseline_images=['agg_filter_alpha'],
                  extensions=['png', 'pdf'])
def test_agg_filter_alpha():
    # 移除此行，当重新生成测试图像时
    plt.rcParams['pcolormesh.snap'] = False

    # 创建一个新的绘图区域
    ax = plt.axes()

    # 创建一个网格
    x, y = np.mgrid[0:7, 0:8]

    # 根据网格数据计算出数据数组
    data = x**2 - y**2

    # 在绘图区域上绘制一个伪彩色网格图
    mesh = ax.pcolormesh(data, cmap='Reds', zorder=5)

    # 定义一个修改 alpha 值的函数
    def manual_alpha(im, dpi):
        im[:, :, 3] *= 0.6
        print('CALLED')
        return im, 0, 0

    # 设置使用自定义 alpha 函数来处理绘图网格
    # 注意：这种方法与在网格本身上设置 alpha 不同。目前，网格被绘制为独立的补丁，
    # 我们可以看到各个颜色块之间的细边。参考 Stack Overflow 上的问题：
    # https://stackoverflow.com/q/20678817/
    mesh.set_agg_filter(manual_alpha)

    # 目前我们必须在 PDF 后端启用栅格化，才能使这一设置生效。
    mesh.set_rasterized(True)

    # 在绘图区域上绘制一个简单的折线图
    ax.plot([0, 4, 7], [1, 3, 8])
```