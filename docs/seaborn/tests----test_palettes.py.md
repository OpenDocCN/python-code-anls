# `D:\src\scipysrc\seaborn\tests\test_palettes.py`

```
import colorsys  # 导入colorsys模块，提供了与颜色系统相关的转换函数
import numpy as np  # 导入NumPy库，用于支持数组和矩阵运算
import matplotlib as mpl  # 导入Matplotlib库，用于绘图和数据可视化

import pytest  # 导入pytest测试框架，用于编写和运行测试
import numpy.testing as npt  # 导入NumPy的测试模块，用于数组比较

from seaborn import palettes, utils, rcmod  # 从seaborn库中导入调色板、实用工具和rc设置模块
from seaborn.external import husl  # 从seaborn的external子模块中导入husl模块
from seaborn._compat import get_colormap  # 从seaborn的_compat模块中导入get_colormap函数
from seaborn.colors import xkcd_rgb, crayons  # 从seaborn的colors模块中导入xkcd_rgb和crayons函数


class TestColorPalettes:  # 定义测试类TestColorPalettes

    def test_current_palette(self):  # 定义测试当前调色板的方法

        pal = palettes.color_palette(["red", "blue", "green"])  # 生成一个颜色调色板
        rcmod.set_palette(pal)  # 设置当前绘图环境的颜色调色板为pal
        assert pal == utils.get_color_cycle()  # 断言当前颜色调色板与获取的颜色循环一致
        rcmod.set()  # 恢复默认的rc设置

    def test_palette_context(self):  # 定义测试调色板上下文的方法

        default_pal = palettes.color_palette()  # 获取默认的颜色调色板
        context_pal = palettes.color_palette("muted")  # 获取"muted"调色板

        with palettes.color_palette(context_pal):  # 在调色板上下文中
            assert utils.get_color_cycle() == context_pal  # 断言当前颜色循环与context_pal一致

        assert utils.get_color_cycle() == default_pal  # 断言恢复后的颜色循环与默认调色板一致

    def test_big_palette_context(self):  # 定义测试大型调色板上下文的方法

        original_pal = palettes.color_palette("deep", n_colors=8)  # 获取"deep"调色板，指定颜色数为8
        context_pal = palettes.color_palette("husl", 10)  # 获取"husl"调色板，指定颜色数为10

        rcmod.set_palette(original_pal)  # 设置当前绘图环境的颜色调色板为original_pal
        with palettes.color_palette(context_pal, 10):  # 在调色板上下文中，指定颜色数为10
            assert utils.get_color_cycle() == context_pal  # 断言当前颜色循环与context_pal一致

        assert utils.get_color_cycle() == original_pal  # 断言恢复后的颜色循环与original_pal一致

        # Reset default
        rcmod.set()  # 恢复默认的rc设置

    def test_palette_size(self):  # 定义测试调色板大小的方法

        pal = palettes.color_palette("deep")  # 获取"deep"调色板
        assert len(pal) == palettes.QUAL_PALETTE_SIZES["deep"]  # 断言调色板长度与预定义的大小相等

        pal = palettes.color_palette("pastel6")  # 获取"pastel6"调色板
        assert len(pal) == palettes.QUAL_PALETTE_SIZES["pastel6"]  # 断言调色板长度与预定义的大小相等

        pal = palettes.color_palette("Set3")  # 获取"Set3"调色板
        assert len(pal) == palettes.QUAL_PALETTE_SIZES["Set3"]  # 断言调色板长度与预定义的大小相等

        pal = palettes.color_palette("husl")  # 获取"husl"调色板
        assert len(pal) == 6  # 断言调色板长度为6

        pal = palettes.color_palette("Greens")  # 获取"Greens"调色板
        assert len(pal) == 6  # 断言调色板长度为6

    def test_seaborn_palettes(self):  # 定义测试seaborn调色板的方法

        pals = "deep", "muted", "pastel", "bright", "dark", "colorblind"  # 定义调色板名称列表
        for name in pals:
            full = palettes.color_palette(name, 10).as_hex()  # 获取指定名称和颜色数的完整调色板，并转换为十六进制表示
            short = palettes.color_palette(name + "6", 6).as_hex()  # 获取指定名称和颜色数的简略调色板，并转换为十六进制表示
            b, _, g, r, m, _, _, _, y, c = full  # 解包完整调色板的颜色值
            assert [b, g, r, m, y, c] == list(short)  # 断言简略调色板中的颜色与完整调色板中的相应颜色一致

    def test_hls_palette(self):  # 定义测试hls调色板的方法

        pal1 = palettes.hls_palette()  # 获取默认的hls调色板
        pal2 = palettes.color_palette("hls")  # 获取"hls"调色板
        npt.assert_array_equal(pal1, pal2)  # 使用NumPy测试模块断言两个调色板数组相等

        cmap1 = palettes.hls_palette(as_cmap=True)  # 获取作为颜色映射的hls调色板
        cmap2 = palettes.color_palette("hls", as_cmap=True)  # 获取作为颜色映射的"hls"调色板
        npt.assert_array_equal(cmap1([.2, .8]), cmap2([.2, .8]))  # 使用NumPy测试模块断言两个颜色映射函数的输出数组相等

    def test_husl_palette(self):  # 定义测试husl调色板的方法

        pal1 = palettes.husl_palette()  # 获取默认的husl调色板
        pal2 = palettes.color_palette("husl")  # 获取"husl"调色板
        npt.assert_array_equal(pal1, pal2)  # 使用NumPy测试模块断言两个调色板数组相等

        cmap1 = palettes.husl_palette(as_cmap=True)  # 获取作为颜色映射的husl调色板
        cmap2 = palettes.color_palette("husl", as_cmap=True)  # 获取作为颜色映射的"husl"调色板
        npt.assert_array_equal(cmap1([.2, .8]), cmap2([.2, .8]))  # 使用NumPy测试模块断言两个颜色映射函数的输出数组相等
    # 测试 matplotlib 风格的调色板生成函数
    def test_mpl_palette(self):
        
        # 使用 palettes.mpl_palette() 获取调色板 "Reds" 的颜色列表
        pal1 = palettes.mpl_palette("Reds")
        # 使用 palettes.color_palette() 同样获取调色板 "Reds" 的颜色列表
        pal2 = palettes.color_palette("Reds")
        # 断言两个调色板的颜色列表是否相等
        npt.assert_array_equal(pal1, pal2)
        
        # 获取调色板 "Reds" 对应的 colormap，用自定义函数 get_colormap()
        cmap1 = get_colormap("Reds")
        # 使用 palettes.mpl_palette(as_cmap=True) 获取 "Reds" 的 colormap
        cmap2 = palettes.mpl_palette("Reds", as_cmap=True)
        # 使用 palettes.color_palette(as_cmap=True) 获取 "Reds" 的 colormap
        cmap3 = palettes.color_palette("Reds", as_cmap=True)
        # 断言三种方式获取的 colormap 是否相等
        npt.assert_array_equal(cmap1, cmap2)
        npt.assert_array_equal(cmap1, cmap3)

    # 测试 matplotlib 风格的暗调色板生成函数
    def test_mpl_dark_palette(self):
        
        # 使用 palettes.mpl_palette() 获取 "Blues_d" 的颜色列表
        mpl_pal1 = palettes.mpl_palette("Blues_d")
        # 使用 palettes.color_palette() 同样获取 "Blues_d" 的颜色列表
        mpl_pal2 = palettes.color_palette("Blues_d")
        # 断言两个调色板的颜色列表是否相等
        npt.assert_array_equal(mpl_pal1, mpl_pal2)
        
        # 使用 palettes.mpl_palette() 获取 "Blues_r_d" 的颜色列表
        mpl_pal1 = palettes.mpl_palette("Blues_r_d")
        # 使用 palettes.color_palette() 同样获取 "Blues_r_d" 的颜色列表
        mpl_pal2 = palettes.color_palette("Blues_r_d")
        # 断言两个调色板的颜色列表是否相等
        npt.assert_array_equal(mpl_pal1, mpl_pal2)

    # 测试使用错误的调色板名称抛出 ValueError 异常
    def test_bad_palette_name(self):
        
        # 使用不存在的调色板名称 "IAmNotAPalette" 来调用 palettes.color_palette()
        with pytest.raises(ValueError):
            palettes.color_palette("IAmNotAPalette")

    # 测试使用不推荐的调色板名称抛出 ValueError 异常
    def test_terrible_palette_name(self):
        
        # 使用不推荐使用的调色板名称 "jet" 来调用 palettes.color_palette()
        with pytest.raises(ValueError):
            palettes.color_palette("jet")

    # 测试使用错误的颜色列表抛出 ValueError 异常
    def test_bad_palette_colors(self):
        
        # 包含非法颜色 "iamnotacolor" 的颜色列表作为参数调用 palettes.color_palette()
        pal = ["red", "blue", "iamnotacolor"]
        # 断言调用时是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            palettes.color_palette(pal)

    # 测试调色板生成时的饱和度设置
    def test_palette_desat(self):
        
        # 使用 husl_palette() 生成长度为 6 的调色板
        pal1 = palettes.husl_palette(6)
        # 对 husl_palette() 生成的调色板进行降低饱和度处理
        pal1 = [utils.desaturate(c, .5) for c in pal1]
        # 使用 color_palette() 生成 husl 风格的调色板并指定饱和度为 0.5
        pal2 = palettes.color_palette("husl", desat=.5)
        # 断言两个调色板是否相等
        npt.assert_array_equal(pal1, pal2)

    # 测试生成的调色板是否为元组列表
    def test_palette_is_list_of_tuples(self):
        
        # 创建包含字符串颜色名称的数组作为输入
        pal_in = np.array(["red", "blue", "green"])
        # 调用 color_palette() 生成长度为 3 的调色板
        pal_out = palettes.color_palette(pal_in, 3)
        
        # 断言输出调色板是否为列表类型
        assert isinstance(pal_out, list)
        # 断言输出调色板中的第一个颜色是否为元组
        assert isinstance(pal_out[0], tuple)
        # 断言元组中的第一个值是否为浮点数
        assert isinstance(pal_out[0][0], float)
        # 断言元组的长度是否为 3
        assert len(pal_out[0]) == 3

    # 测试调色板的循环特性
    def test_palette_cycles(self):
        
        # 获取 "deep6" 调色板的颜色列表
        deep = palettes.color_palette("deep6")
        # 获取将 "deep6" 调色板重复两次的颜色列表
        double_deep = palettes.color_palette("deep6", 12)
        # 断言重复两次的调色板是否等于单次调色板的两倍长度
        assert double_deep == deep + deep

    # 测试 HLS 颜色空间中的调色板生成
    def test_hls_values(self):
        
        # 使用 hls_palette() 生成长度为 6，色调 h 为 0 的调色板
        pal1 = palettes.hls_palette(6, h=0)
        # 使用 hls_palette() 生成长度为 6，色调 h 为 0.5 的调色板
        pal2 = palettes.hls_palette(6, h=.5)
        # 将 pal2 调色板列表循环左移 3 个位置
        pal2 = pal2[3:] + pal2[:3]
        # 断言两个调色板是否近似相等
        npt.assert_array_almost_equal(pal1, pal2)
        
        # 生成亮度 l 为 0.2 的长度为 5 的 HLS 调色板
        pal_dark = palettes.hls_palette(5, l=.2)  # noqa
        # 生成亮度 l 为 0.8 的长度为 5 的 HLS 调色板
        pal_bright = palettes.hls_palette(5, l=.8)  # noqa
        # 断言深色调色板的颜色总和小于亮色调色板的颜色总和
        npt.assert_array_less(list(map(sum, pal_dark)),
                              list(map(sum, pal_bright)))

        # 生成饱和度 s 为 0.1 的长度为 5 的 HLS 调色板
        pal_flat = palettes.hls_palette(5, s=.1)
        # 生成饱和度 s 为 0.9 的长度为 5 的 HLS 调色板
        pal_bold = palettes.hls_palette(5, s=.9)
        # 断言低饱和度调色板的标准差小于高饱和度调色板的标准差
        npt.assert_array_less(list(map(np.std, pal_flat)),
                              list(map(np.std, pal_bold)))
    # 测试 HUSL 调色板的值是否正确
    def test_husl_values(self):

        # 使用 HUSL 调色板生成 6 种颜色，hue 参数为 0
        pal1 = palettes.husl_palette(6, h=0)
        # 使用 HUSL 调色板生成另外 6 种颜色，hue 参数为 0.5
        pal2 = palettes.husl_palette(6, h=.5)
        # 将 pal2 数组的前三个元素移至末尾，以测试循环性质
        pal2 = pal2[3:] + pal2[:3]
        # 断言 pal1 和经过旋转的 pal2 数组近乎相等
        npt.assert_array_almost_equal(pal1, pal2)

        # 使用较暗的 HUSL 调色板生成 5 种颜色，亮度参数为 0.2
        pal_dark = palettes.husl_palette(5, l=.2)  # noqa
        # 使用较亮的 HUSL 调色板生成 5 种颜色，亮度参数为 0.8
        pal_bright = palettes.husl_palette(5, l=.8)  # noqa
        # 断言较暗色调板的颜色总和小于较亮色调板的颜色总和
        npt.assert_array_less(list(map(sum, pal_dark)),
                              list(map(sum, pal_bright)))

        # 使用较平的 HUSL 调色板生成 5 种颜色，饱和度参数为 0.1
        pal_flat = palettes.husl_palette(5, s=.1)
        # 使用较鲜明的 HUSL 调色板生成 5 种颜色，饱和度参数为 0.9
        pal_bold = palettes.husl_palette(5, s=.9)
        # 断言较平调色板的标准差小于较鲜明调色板的标准差
        npt.assert_array_less(list(map(np.std, pal_flat)),
                              list(map(np.std, pal_bold)))

    # 测试 cbrewer 调色板的质量
    def test_cbrewer_qual(self):

        # 使用 matplotlib 中的 "Set1" 调色板生成 4 种颜色
        pal_short = palettes.mpl_palette("Set1", 4)
        # 使用 matplotlib 中的 "Set1" 调色板生成 6 种颜色
        pal_long = palettes.mpl_palette("Set1", 6)
        # 断言 pal_short 与 pal_long 的前四个颜色相等
        assert pal_short == pal_long[:4]

        # 使用 matplotlib 中的 "Set2" 调色板生成 8 种颜色
        pal_full = palettes.mpl_palette("Set2", 8)
        # 使用 matplotlib 中的 "Set2" 调色板生成 10 种颜色
        pal_long = palettes.mpl_palette("Set2", 10)
        # 断言 pal_full 与 pal_long 的前八个颜色相等
        assert pal_full == pal_long[:8]

    # 测试 matplotlib 中调色板颜色反转
    def test_mpl_reversal(self):

        # 使用 "BuPu" 调色板生成 6 种颜色
        pal_forward = palettes.mpl_palette("BuPu", 6)
        # 使用 "BuPu_r" 调色板生成 6 种颜色，反转顺序
        pal_reverse = palettes.mpl_palette("BuPu_r", 6)
        # 断言 pal_forward 与反转后的 pal_reverse 数组近乎相等
        npt.assert_array_almost_equal(pal_forward, pal_reverse[::-1])

    # 测试从 HLS 转换为 RGB 的功能
    def test_rgb_from_hls(self):

        # 定义 HLS 颜色
        color = .5, .8, .4
        # 使用 palettes._color_to_rgb 将 HLS 颜色转换为 RGB
        rgb_got = palettes._color_to_rgb(color, "hls")
        # 使用 colorsys.hls_to_rgb 将 HLS 颜色转换为 RGB 作为期望结果
        rgb_want = colorsys.hls_to_rgb(*color)
        # 断言转换后的 RGB 值与期望值相等
        assert rgb_got == rgb_want

    # 测试从 HUSL 转换为 RGB 的功能
    def test_rgb_from_husl(self):

        # 定义 HUSL 颜色
        color = 120, 50, 40
        # 使用 palettes._color_to_rgb 将 HUSL 颜色转换为 RGB
        rgb_got = palettes._color_to_rgb(color, "husl")
        # 使用 husl.husl_to_rgb 将 HUSL 颜色转换为 RGB 作为期望结果
        rgb_want = tuple(husl.husl_to_rgb(*color))
        # 断言转换后的 RGB 值与期望值相等
        assert rgb_got == rgb_want

        # 遍历 0 到 359 的色调值，测试每个值转换为 RGB 后的范围
        for h in range(0, 360):
            # 定义 HUSL 颜色
            color = h, 100, 100
            # 使用 palettes._color_to_rgb 将 HUSL 颜色转换为 RGB
            rgb = palettes._color_to_rgb(color, "husl")
            # 断言 RGB 值的最小值大于等于 0，最大值小于等于 1
            assert min(rgb) >= 0
            assert max(rgb) <= 1

    # 测试从 XKCD 颜色名转换为 RGB 的功能
    def test_rgb_from_xkcd(self):

        # 定义 XKCD 颜色名
        color = "dull red"
        # 使用 palettes._color_to_rgb 将 XKCD 颜色名转换为 RGB
        rgb_got = palettes._color_to_rgb(color, "xkcd")
        # 使用 mpl.colors.to_rgb 将 XKCD 颜色名转换为 RGB 作为期望结果
        rgb_want = mpl.colors.to_rgb(xkcd_rgb[color])
        # 断言转换后的 RGB 值与期望值相等
        assert rgb_got == rgb_want
    # 定义测试函数 `test_light_palette`
    def test_light_palette(self):

        # 设定颜色调色板的颜色为红色，并生成包含 n 个颜色的正向调色板
        pal_forward = palettes.light_palette("red", n)
        # 设定颜色调色板的颜色为红色，并生成包含 n 个颜色的反向调色板
        pal_reverse = palettes.light_palette("red", n, reverse=True)
        # 断言正向调色板与反向调色板翻转后相等
        assert np.allclose(pal_forward, pal_reverse[::-1])

        # 将字符串表示的红色转换为 RGB 形式
        red = mpl.colors.colorConverter.to_rgb("red")
        # 断言正向调色板的最后一个颜色与红色的 RGB 值相等
        assert pal_forward[-1] == red

        # 使用字符串描述的方法生成红色的正向调色板，并断言第四个颜色与手动生成的相同
        pal_f_from_string = palettes.color_palette("light:red", n)
        assert pal_forward[3] == pal_f_from_string[3]

        # 使用字符串描述的方法生成红色的反向调色板，并断言第四个颜色与手动生成的相同
        pal_r_from_string = palettes.color_palette("light:red_r", n)
        assert pal_reverse[3] == pal_r_from_string[3]

        # 生成蓝色的线性分段色彩映射，并断言其类型为 LinearSegmentedColormap
        pal_cmap = palettes.light_palette("blue", as_cmap=True)
        assert isinstance(pal_cmap, mpl.colors.LinearSegmentedColormap)

        # 使用字符串描述的方法生成蓝色的线性分段色彩映射，并断言在指定点处映射值相等
        pal_cmap_from_string = palettes.color_palette("light:blue", as_cmap=True)
        assert pal_cmap(.8) == pal_cmap_from_string(.8)

        # 生成蓝色的反向线性分段色彩映射，并断言在指定点处映射值相等
        pal_cmap = palettes.light_palette("blue", as_cmap=True, reverse=True)
        pal_cmap_from_string = palettes.color_palette("light:blue_r", as_cmap=True)
        assert pal_cmap(.8) == pal_cmap_from_string(.8)

    # 定义测试函数 `test_dark_palette`
    def test_dark_palette(self):

        n = 4
        # 设定颜色调色板的颜色为红色，并生成包含 n 个颜色的正向调色板
        pal_forward = palettes.dark_palette("red", n)
        # 设定颜色调色板的颜色为红色，并生成包含 n 个颜色的反向调色板
        pal_reverse = palettes.dark_palette("red", n, reverse=True)
        # 断言正向调色板与反向调色板翻转后相等
        assert np.allclose(pal_forward, pal_reverse[::-1])

        # 将字符串表示的红色转换为 RGB 形式
        red = mpl.colors.colorConverter.to_rgb("red")
        # 断言正向调色板的最后一个颜色与红色的 RGB 值相等
        assert pal_forward[-1] == red

        # 使用字符串描述的方法生成红色的正向调色板，并断言第四个颜色与手动生成的相同
        pal_f_from_string = palettes.color_palette("dark:red", n)
        assert pal_forward[3] == pal_f_from_string[3]

        # 使用字符串描述的方法生成红色的反向调色板，并断言第四个颜色与手动生成的相同
        pal_r_from_string = palettes.color_palette("dark:red_r", n)
        assert pal_reverse[3] == pal_r_from_string[3]

        # 生成蓝色的线性分段色彩映射，并断言其类型为 LinearSegmentedColormap
        pal_cmap = palettes.dark_palette("blue", as_cmap=True)
        assert isinstance(pal_cmap, mpl.colors.LinearSegmentedColormap)

        # 使用字符串描述的方法生成蓝色的线性分段色彩映射，并断言在指定点处映射值相等
        pal_cmap_from_string = palettes.color_palette("dark:blue", as_cmap=True)
        assert pal_cmap(.8) == pal_cmap_from_string(.8)

        # 生成蓝色的反向线性分段色彩映射，并断言在指定点处映射值相等
        pal_cmap = palettes.dark_palette("blue", as_cmap=True, reverse=True)
        pal_cmap_from_string = palettes.color_palette("dark:blue_r", as_cmap=True)
        assert pal_cmap(.8) == pal_cmap_from_string(.8)

    # 定义测试函数 `test_diverging_palette`
    def test_diverging_palette(self):

        # 定义色相值 h_neg 和 h_pos，饱和度 sat，亮度 lum
        h_neg, h_pos = 100, 200
        sat, lum = 70, 50
        args = h_neg, h_pos, sat, lum

        n = 12
        # 使用给定的参数生成一个分散调色板
        pal = palettes.diverging_palette(*args, n=n)
        # 使用 HUSL 输入生成负方向的亮色调色板
        neg_pal = palettes.light_palette((h_neg, sat, lum), int(n // 2),
                                         input="husl")
        # 使用 HUSL 输入生成正方向的亮色调色板
        pos_pal = palettes.light_palette((h_pos, sat, lum), int(n // 2),
                                         input="husl")
        # 断言生成的分散调色板长度为 n
        assert len(pal) == n
        # 断言分散调色板的第一个颜色与负向亮色调色板的最后一个颜色相等
        assert pal[0] == neg_pal[-1]
        # 断言分散调色板的最后一个颜色与正向亮色调色板的最后一个颜色相等
        assert pal[-1] == pos_pal[-1]

        # 使用给定参数生成一个以“dark”为中心的分散调色板
        pal_dark = palettes.diverging_palette(*args, n=n, center="dark")
        # 断言分散调色板与以“dark”为中心的分散调色板的中间颜色的均值比较
        assert np.mean(pal[int(n / 2)]) > np.mean(pal_dark[int(n / 2)])

        # 生成一个分散调色板的线性分段色彩映射，并断言其类型为 LinearSegmentedColormap
        pal_cmap = palettes.diverging_palette(*args, as_cmap=True)
        assert isinstance(pal_cmap, mpl.colors.LinearSegmentedColormap)
    # 测试混合调色板功能

    # 定义颜色列表
    colors = ["red", "yellow", "white"]
    # 使用给定颜色列表创建混合调色板，返回一个 matplotlib 的色彩映射对象
    pal_cmap = palettes.blend_palette(colors, as_cmap=True)
    # 断言返回的调色板对象是 LinearSegmentedColormap 类的实例
    assert isinstance(pal_cmap, mpl.colors.LinearSegmentedColormap)

    # 重新定义颜色列表
    colors = ["red", "blue"]
    # 使用给定颜色列表创建混合调色板
    pal = palettes.blend_palette(colors)
    # 构建混合调色板的字符串表示
    pal_str = "blend:" + ",".join(colors)
    # 使用调色板字符串创建调色板对象
    pal_from_str = palettes.color_palette(pal_str)
    # 断言两个调色板对象相等
    assert pal == pal_from_str

    # 测试 Cubehelix 调色板与 Matplotlib 的一致性

    # 在 [0, 1] 范围内生成均匀间隔的数据点
    x = np.linspace(0, 1, 8)
    # 使用 Matplotlib 的 Cubehelix 函数生成调色板，并转换为列表
    mpl_pal = mpl.cm.cubehelix(x)[:, :3].tolist()

    # 使用 Seaborn 提供的 Cubehelix 调色板函数生成调色板
    sns_pal = palettes.cubehelix_palette(8, start=0.5, rot=-1.5, hue=1,
                                         dark=0, light=1, reverse=True)

    # 断言两个调色板列表相等
    assert sns_pal == mpl_pal

    # 测试生成指定颜色数目的 Cubehelix 调色板

    # 对于每个指定的颜色数目 n
    for n in [3, 5, 8]:
        # 使用 Seaborn 的 Cubehelix 调色板函数生成调色板
        pal = palettes.cubehelix_palette(n)
        # 断言生成的调色板长度为 n
        assert len(pal) == n

    # 测试 Cubehelix 调色板的反转功能

    # 使用默认参数生成正向的 Cubehelix 调色板
    pal_forward = palettes.cubehelix_palette()
    # 使用 reverse=True 参数生成反向的 Cubehelix 调色板
    pal_reverse = palettes.cubehelix_palette(reverse=True)
    # 断言反向调色板列表与正向调色板列表的逆序相等
    assert pal_forward == pal_reverse[::-1]

    # 测试生成 Cubehelix 色彩映射

    # 使用 as_cmap=True 参数生成 Cubehelix 色彩映射对象
    cmap = palettes.cubehelix_palette(as_cmap=True)
    # 断言生成的色彩映射对象是 ListedColormap 类的实例
    assert isinstance(cmap, mpl.colors.ListedColormap)
    # 生成标准的 Cubehelix 调色板
    pal = palettes.cubehelix_palette()
    # 在 [0, 1] 范围内生成均匀间隔的数据点
    x = np.linspace(0, 1, 6)
    # 断言通过色彩映射对象生成的颜色与标准调色板的颜色相等
    npt.assert_array_equal(cmap(x)[:, :3], pal)

    # 使用 as_cmap=True 和 reverse=True 参数生成反向的 Cubehelix 色彩映射对象
    cmap_rev = palettes.cubehelix_palette(as_cmap=True, reverse=True)
    # 在 [0, 1] 范围内生成均匀间隔的数据点
    x = np.linspace(0, 1, 6)
    # 使用正向和反向的色彩映射对象生成调色板颜色列表
    pal_forward = cmap(x).tolist()
    pal_reverse = cmap_rev(x[::-1]).tolist()
    # 断言反向生成的调色板颜色列表与正向生成的调色板颜色列表相等
    assert pal_forward == pal_reverse

    # 测试 Cubehelix 调色板字符串代码生成

    # 定义颜色调色板和 Cubehelix 调色板函数的别名
    color_palette = palettes.color_palette
    cubehelix_palette = palettes.cubehelix_palette

    # 使用调色板字符串生成 Cubehelix 调色板
    pal1 = color_palette("ch:", 8)
    pal2 = color_palette(cubehelix_palette(8))
    # 断言两个调色板对象相等
    assert pal1 == pal2

    # 使用带参数的调色板字符串生成 Cubehelix 调色板
    pal1 = color_palette("ch:.5, -.25,hue = .5,light=.75", 8)
    pal2 = color_palette(cubehelix_palette(8, .5, -.25, hue=.5, light=.75))
    # 断言两个调色板对象相等
    assert pal1 == pal2

    # 使用带参数的调色板字符串生成 Cubehelix 调色板
    pal1 = color_palette("ch:h=1,r=.5", 9)
    pal2 = color_palette(cubehelix_palette(9, hue=1, rot=.5))
    # 断言两个调色板对象相等
    assert pal1 == pal2

    # 使用反向参数的调色板字符串生成 Cubehelix 调色板
    pal1 = color_palette("ch:_r", 6)
    pal2 = color_palette(cubehelix_palette(6, reverse=True))
    # 断言两个调色板对象相等
    assert pal1 == pal2

    # 使用反向参数的调色板字符串生成反向的 Cubehelix 色彩映射对象
    pal1 = color_palette("ch:_r", as_cmap=True)
    pal2 = cubehelix_palette(6, reverse=True, as_cmap=True)
    # 断言通过两种方式生成的色彩映射对象在中间值处返回的颜色相等
    assert pal1(.5) == pal2(.5)

    # 测试 xkcd 调色板

    # 从 xkcd_rgb 字典中选择一些颜色名称
    names = list(xkcd_rgb.keys())[10:15]
    # 使用 xkcd 调色板函数生成颜色列表
    colors = palettes.xkcd_palette(names)
    # 对比每个颜色的 RGB 转换为十六进制值，确保与 xkcd_rgb 中的值相等
    for name, color in zip(names, colors):
        as_hex = mpl.colors.rgb2hex(color)
        assert as_hex == xkcd_rgb[name]
    # 测试调色板功能中的颜色匹配
    def test_crayon_palette(self):
        # 选择从调色板中取出的颜色名列表的子集
        names = list(crayons.keys())[10:15]
        # 使用选定的颜色名生成调色板
        colors = palettes.crayon_palette(names)
        # 遍历颜色名和对应的颜色值
        for name, color in zip(names, colors):
            # 将颜色转换为十六进制表示
            as_hex = mpl.colors.rgb2hex(color)
            # 断言颜色的十六进制表示与预期的调色板颜色匹配
            assert as_hex == crayons[name].lower()

    # 测试颜色编码设置
    def test_color_codes(self):
        # 设置颜色编码为深色模式
        palettes.set_color_codes("deep")
        # 获取深色模式下的颜色调色板及其扩展
        colors = palettes.color_palette("deep6") + [".1"]
        # 遍历颜色编码及其对应的颜色值
        for code, color in zip("bgrmyck", colors):
            # 将颜色转换为 RGB 格式
            rgb_want = mpl.colors.colorConverter.to_rgb(color)
            rgb_got = mpl.colors.colorConverter.to_rgb(code)
            # 断言预期的 RGB 值与实际获取的 RGB 值相等
            assert rgb_want == rgb_got
        # 重置颜色编码设置
        palettes.set_color_codes("reset")

        # 断言设置不支持的颜色编码会抛出 ValueError 异常
        with pytest.raises(ValueError):
            palettes.set_color_codes("Set1")

    # 测试将调色板颜色转换为十六进制表示
    def test_as_hex(self):
        # 获取调色板中的颜色列表
        pal = palettes.color_palette("deep")
        # 遍历 RGB 格式和对应的十六进制表示
        for rgb, hex in zip(pal, pal.as_hex()):
            # 断言 RGB 格式转换为十六进制后与原始十六进制表示相同
            assert mpl.colors.rgb2hex(rgb) == hex

    # 测试保持调色板长度不变
    def test_preserved_palette_length(self):
        # 获取设置为指定长度的调色板
        pal_in = palettes.color_palette("Set1", 10)
        # 复制调色板以确保长度不变
        pal_out = palettes.color_palette(pal_in)
        # 断言输入调色板与输出调色板相等
        assert pal_in == pal_out

    # 测试调色板的 HTML 表示
    def test_html_repr(self):
        # 获取调色板对象
        pal = palettes.color_palette()
        # 获取调色板的 HTML 表示
        html = pal._repr_html_()
        # 断言调色板中的十六进制颜色表示在 HTML 中出现
        for color in pal.as_hex():
            assert color in html

    # 测试颜色映射显示补丁
    def test_colormap_display_patch(self):
        # 获取原始的 Colormap 类的 HTML 和 PNG 表示方法
        orig_repr_png = getattr(mpl.colors.Colormap, "_repr_png_", None)
        orig_repr_html = getattr(mpl.colors.Colormap, "_repr_html_", None)

        try:
            # 打补丁以支持调色板的 HTML 表示
            palettes._patch_colormap_display()
            # 获取一个预定义的颜色映射对象
            cmap = mpl.cm.Reds
            # 断言颜色映射的 HTML 表示以特定字符串开头
            assert cmap._repr_html_().startswith('<img alt="Reds')
        finally:
            # 恢复原始的 HTML 和 PNG 表示方法
            if orig_repr_png is not None:
                mpl.colors.Colormap._repr_png_ = orig_repr_png
            if orig_repr_html is not None:
                mpl.colors.Colormap._repr_html_ = orig_repr_html
```