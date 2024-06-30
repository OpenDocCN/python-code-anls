# `D:\src\scipysrc\scipy\scipy\signal\tests\test_windows.py`

```
import numpy as np  # 导入NumPy库，用于科学计算
from numpy import array  # 导入array函数，用于创建NumPy数组
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose,
                           assert_equal, assert_, assert_array_less,
                           suppress_warnings)  # 导入测试相关的函数和类

from pytest import raises as assert_raises  # 导入raises函数并重命名为assert_raises

from scipy.fft import fft  # 导入fft函数，用于快速傅里叶变换
from scipy.signal import windows, get_window, resample  # 导入信号处理相关的函数和窗函数

window_funcs = [  # 定义窗函数列表
    ('boxcar', ()),  # 矩形窗函数
    ('triang', ()),  # 三角窗函数
    ('parzen', ()),  # Parzen窗函数
    ('bohman', ()),  # Bohman窗函数
    ('blackman', ()),  # Blackman窗函数
    ('nuttall', ()),  # Nuttall窗函数
    ('blackmanharris', ()),  # Blackman-Harris窗函数
    ('flattop', ()),  # Flat top窗函数
    ('bartlett', ()),  # Bartlett窗函数
    ('barthann', ()),  # Bartlett-Hann窗函数
    ('hamming', ()),  # Hamming窗函数
    ('kaiser', (1,)),  # Kaiser窗函数
    ('dpss', (2,)),  # DPSS（Slepian）窗函数
    ('gaussian', (0.5,)),  # 高斯窗函数
    ('general_gaussian', (1.5, 2)),  # 广义高斯窗函数
    ('chebwin', (1,)),  # Chebyshev窗函数
    ('cosine', ()),  # 余弦窗函数
    ('hann', ()),  # Hann窗函数
    ('exponential', ()),  # 指数衰减窗函数
    ('taylor', ()),  # Taylor窗函数
    ('tukey', (0.5,)),  # Tukey窗函数
    ('lanczos', ()),  # Lanczos窗函数
]


class TestBartHann:  # 定义测试类TestBartHann

    def test_basic(self):  # 测试基本功能
        assert_allclose(windows.barthann(6, sym=True),  # 断言函数输出与预期结果相近
                        [0, 0.35857354213752, 0.8794264578624801,
                         0.8794264578624801, 0.3585735421375199, 0],
                        rtol=1e-15, atol=1e-15)  # 设置断言的相对和绝对误差

        assert_allclose(windows.barthann(7),  # 断言函数输出与预期结果相近
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0],
                        rtol=1e-15, atol=1e-15)  # 设置断言的相对和绝对误差

        assert_allclose(windows.barthann(6, False),  # 断言函数输出与预期结果相近
                        [0, 0.27, 0.73, 1.0, 0.73, 0.27],
                        rtol=1e-15, atol=1e-15)  # 设置断言的相对和绝对误差


class TestBartlett:  # 定义测试类TestBartlett

    def test_basic(self):  # 测试基本功能
        assert_allclose(windows.bartlett(6),  # 断言函数输出与预期结果相近
                        [0, 0.4, 0.8, 0.8, 0.4, 0])  # 设置预期结果

        assert_allclose(windows.bartlett(7),  # 断言函数输出与预期结果相近
                        [0, 1/3, 2/3, 1.0, 2/3, 1/3, 0])  # 设置预期结果

        assert_allclose(windows.bartlett(6, False),  # 断言函数输出与预期结果相近
                        [0, 1/3, 2/3, 1.0, 2/3, 1/3])  # 设置预期结果


class TestBlackman:  # 定义测试类TestBlackman

    def test_basic(self):  # 测试基本功能
        assert_allclose(windows.blackman(6, sym=False),  # 断言函数输出与预期结果相近
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13], atol=1e-14)  # 设置预期结果和允许的绝对误差

        assert_allclose(windows.blackman(7, sym=False),  # 断言函数输出与预期结果相近
                        [0, 0.09045342435412804, 0.4591829575459636,
                         0.9203636180999081, 0.9203636180999081,
                         0.4591829575459636, 0.09045342435412804], atol=1e-8)  # 设置预期结果和允许的绝对误差

        assert_allclose(windows.blackman(6),  # 断言函数输出与预期结果相近
                        [0, 0.2007701432625305, 0.8492298567374694,
                         0.8492298567374694, 0.2007701432625305, 0],
                        atol=1e-14)  # 设置预期结果和允许的绝对误差

        assert_allclose(windows.blackman(7, True),  # 断言函数输出与预期结果相近
                        [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14)  # 设置预期结果和允许的绝对误差


class TestBlackmanHarris:  # 定义测试类TestBlackmanHarris
    # 定义测试方法，用于测试 windows 模块中的 blackmanharris 函数的不同参数组合
    def test_basic(self):
        # 断言 blackmanharris 函数生成的窗口函数与预期值非常接近
        assert_allclose(windows.blackmanharris(6, False),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645])
        # 断言 blackmanharris 函数生成的窗口函数与预期值非常接近，带有非对称参数
        assert_allclose(windows.blackmanharris(7, sym=False),
                        [6.0e-05, 0.03339172347815117, 0.332833504298565,
                         0.8893697722232837, 0.8893697722232838,
                         0.3328335042985652, 0.03339172347815122])
        # 断言 blackmanharris 函数生成的窗口函数与预期值非常接近，使用默认的对称参数
        assert_allclose(windows.blackmanharris(6),
                        [6.0e-05, 0.1030114893456638, 0.7938335106543362,
                         0.7938335106543364, 0.1030114893456638, 6.0e-05])
        # 断言 blackmanharris 函数生成的窗口函数与预期值非常接近，带有对称参数
        assert_allclose(windows.blackmanharris(7, sym=True),
                        [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645,
                         6.0e-05])
class TestTaylor:

    def test_normalized(self):
        """Tests windows of small length that are normalized to 1. See the
        documentation for the Taylor window for more information on
        normalization.
        """
        # 测试长度较小的窗口，确保其被归一化为1。详见Taylor窗口的文档以获取更多归一化信息。
        assert_allclose(windows.taylor(1, 2, 15), 1.0)
        assert_allclose(
            windows.taylor(5, 2, 15),
            np.array([0.75803341, 0.90757699, 1.0, 0.90757699, 0.75803341])
        )
        assert_allclose(
            windows.taylor(6, 2, 15),
            np.array([
                0.7504082, 0.86624416, 0.98208011, 0.98208011, 0.86624416,
                0.7504082
            ])
        )

    def test_non_normalized(self):
        """Test windows of small length that are not normalized to 1. See
        the documentation for the Taylor window for more information on
        normalization.
        """
        # 测试长度较小的窗口，确保其未归一化为1。详见Taylor窗口的文档以获取更多归一化信息。
        assert_allclose(
            windows.taylor(5, 2, 15, norm=False),
            np.array([
                0.87508054, 1.04771499, 1.15440894, 1.04771499, 0.87508054
            ])
        )
        assert_allclose(
            windows.taylor(6, 2, 15, norm=False),
            np.array([
                0.86627793, 1.0, 1.13372207, 1.13372207, 1.0, 0.86627793
            ])
        )

    def test_correctness(self):
        """This test ensures the correctness of the implemented Taylor
        Windowing function. A Taylor Window of 1024 points is created, its FFT
        is taken, and the Peak Sidelobe Level (PSLL) and 3dB and 18dB bandwidth
        are found and checked.

        A publication from Sandia National Laboratories was used as reference
        for the correctness values [1]_.

        References
        -----
        .. [1] Armin Doerry, "Catalog of Window Taper Functions for
               Sidelobe Control", 2017.
               https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
        """
        # 创建1024点的Taylor窗口，进行FFT操作，并计算其峰值旁瓣电平（PSLL）、3dB和18dB带宽，并进行检查。
        # 使用Sandia国家实验室的出版物作为正确性数值的参考。
        M_win = 1024
        N_fft = 131072
        # 由于从科学出版物获得的值不进行归一化，设置norm=False以确保正确性。
        # 归一化会改变旁瓣电平达到的期望值。
        w = windows.taylor(M_win, nbar=4, sll=35, norm=False, sym=False)
        f = fft(w, N_fft)
        spec = 20 * np.log10(np.abs(f / np.amax(f)))

        first_zero = np.argmax(np.diff(spec) > 0)

        PSLL = np.amax(spec[first_zero:-first_zero])

        BW_3dB = 2*np.argmax(spec <= -3.0102999566398121) / N_fft * M_win
        BW_18dB = 2*np.argmax(spec <= -18.061799739838872) / N_fft * M_win

        assert_allclose(PSLL, -35.1672, atol=1)
        assert_allclose(BW_3dB, 1.1822, atol=0.1)
        assert_allclose(BW_18dB, 2.6112, atol=0.1)
    # 定义测试方法 test_basic，用于测试 windows 模块中的 bohman 函数
    def test_basic(self):
        # 断言 bohman(6) 的返回值与预期结果接近
        assert_allclose(windows.bohman(6),
                        [0, 0.1791238937062839, 0.8343114522576858,
                         0.8343114522576858, 0.1791238937062838, 0])
        # 断言 bohman(7, sym=True) 的返回值与预期结果接近
        assert_allclose(windows.bohman(7, sym=True),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293, 0])
        # 断言 bohman(6, False) 的返回值与预期结果接近
        assert_allclose(windows.bohman(6, False),
                        [0, 0.1089977810442293, 0.6089977810442293, 1.0,
                         0.6089977810442295, 0.1089977810442293])
class TestBoxcar:

    def test_basic(self):
        # 检查窗口函数 boxcar 在不同参数下的输出是否符合预期
        assert_allclose(windows.boxcar(6), [1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(7), [1, 1, 1, 1, 1, 1, 1])
        assert_allclose(windows.boxcar(6, False), [1, 1, 1, 1, 1, 1])


cheb_odd_true = array([0.200938, 0.107729, 0.134941, 0.165348,
                       0.198891, 0.235450, 0.274846, 0.316836,
                       0.361119, 0.407338, 0.455079, 0.503883,
                       0.553248, 0.602637, 0.651489, 0.699227,
                       0.745266, 0.789028, 0.829947, 0.867485,
                       0.901138, 0.930448, 0.955010, 0.974482,
                       0.988591, 0.997138, 1.000000, 0.997138,
                       0.988591, 0.974482, 0.955010, 0.930448,
                       0.901138, 0.867485, 0.829947, 0.789028,
                       0.745266, 0.699227, 0.651489, 0.602637,
                       0.553248, 0.503883, 0.455079, 0.407338,
                       0.361119, 0.316836, 0.274846, 0.235450,
                       0.198891, 0.165348, 0.134941, 0.107729,
                       0.200938])

cheb_even_true = array([0.203894, 0.107279, 0.133904,
                        0.163608, 0.196338, 0.231986,
                        0.270385, 0.311313, 0.354493,
                        0.399594, 0.446233, 0.493983,
                        0.542378, 0.590916, 0.639071,
                        0.686302, 0.732055, 0.775783,
                        0.816944, 0.855021, 0.889525,
                        0.920006, 0.946060, 0.967339,
                        0.983557, 0.994494, 1.000000,
                        1.000000, 0.994494, 0.983557,
                        0.967339, 0.946060, 0.920006,
                        0.889525, 0.855021, 0.816944,
                        0.775783, 0.732055, 0.686302,
                        0.639071, 0.590916, 0.542378,
                        0.493983, 0.446233, 0.399594,
                        0.354493, 0.311313, 0.270385,
                        0.231986, 0.196338, 0.163608,
                        0.133904, 0.107279, 0.203894])


class TestChebWin:
    # 空类定义，用于测试 Chebyshev 窗口函数，下面将会添加测试方法
    # 测试基本的窗函数生成是否正确
    def test_basic(self):
        # 使用 suppress_warnings 上下文管理器来抑制特定类型的警告
        with suppress_warnings() as sup:
            # 过滤掉 UserWarning 类型的警告消息 "This window is not suitable"
            sup.filter(UserWarning, "This window is not suitable")
            # 检查生成的 Chebyshev 窗函数（阶数为 6，窗长为 100）是否符合预期
            assert_allclose(windows.chebwin(6, 100),
                            [0.1046401879356917, 0.5075781475823447, 1.0, 1.0,
                             0.5075781475823447, 0.1046401879356917])
            # 检查生成的 Chebyshev 窗函数（阶数为 7，窗长为 100）是否符合预期
            assert_allclose(windows.chebwin(7, 100),
                            [0.05650405062850233, 0.316608530648474,
                             0.7601208123539079, 1.0, 0.7601208123539079,
                             0.316608530648474, 0.05650405062850233])
            # 检查生成的 Chebyshev 窗函数（阶数为 6，窗长为 10）是否符合预期
            assert_allclose(windows.chebwin(6, 10),
                            [1.0, 0.6071201674458373, 0.6808391469897297,
                             0.6808391469897297, 0.6071201674458373, 1.0])
            # 检查生成的 Chebyshev 窗函数（阶数为 7，窗长为 10）是否符合预期
            assert_allclose(windows.chebwin(7, 10),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651, 1.0])
            # 检查生成的 Chebyshev 窗函数（阶数为 6，窗长为 10，不对称）是否符合预期
            assert_allclose(windows.chebwin(6, 10, False),
                            [1.0, 0.5190521247588651, 0.5864059018130382,
                             0.6101519801307441, 0.5864059018130382,
                             0.5190521247588651])

    # 测试高衰减的奇数阶 Chebyshev 窗函数生成是否正确
    def test_cheb_odd_high_attenuation(self):
        with suppress_warnings() as sup:
            # 过滤掉 UserWarning 类型的警告消息 "This window is not suitable"
            sup.filter(UserWarning, "This window is not suitable")
            # 生成高衰减的奇数阶 Chebyshev 窗函数（阶数为 53，衰减为 -40dB）
            cheb_odd = windows.chebwin(53, at=-40)
        # 检查生成的窗函数是否与预期的真实值（cheb_odd_true）几乎相等
        assert_array_almost_equal(cheb_odd, cheb_odd_true, decimal=4)

    # 测试高衰减的偶数阶 Chebyshev 窗函数生成是否正确
    def test_cheb_even_high_attenuation(self):
        with suppress_warnings() as sup:
            # 过滤掉 UserWarning 类型的警告消息 "This window is not suitable"
            sup.filter(UserWarning, "This window is not suitable")
            # 生成高衰减的偶数阶 Chebyshev 窗函数（阶数为 54，衰减为 40dB）
            cheb_even = windows.chebwin(54, at=40)
        # 检查生成的窗函数是否与预期的真实值（cheb_even_true）几乎相等
        assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)

    # 测试低衰减的奇数阶 Chebyshev 窗函数生成是否正确
    def test_cheb_odd_low_attenuation(self):
        # 预期的低衰减奇数阶 Chebyshev 窗函数真实值
        cheb_odd_low_at_true = array([1.000000, 0.519052, 0.586405,
                                      0.610151, 0.586405, 0.519052,
                                      1.000000])
        with suppress_warnings() as sup:
            # 过滤掉 UserWarning 类型的警告消息 "This window is not suitable"
            sup.filter(UserWarning, "This window is not suitable")
            # 生成低衰减的奇数阶 Chebyshev 窗函数（阶数为 7，衰减为 10dB）
            cheb_odd = windows.chebwin(7, at=10)
        # 检查生成的窗函数是否与预期的真实值几乎相等
        assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)

    # 测试低衰减的偶数阶 Chebyshev 窗函数生成是否正确
    def test_cheb_even_low_attenuation(self):
        # 预期的低衰减偶数阶 Chebyshev 窗函数真实值
        cheb_even_low_at_true = array([1.000000, 0.451924, 0.51027,
                                       0.541338, 0.541338, 0.51027,
                                       0.451924, 1.000000])
        with suppress_warnings() as sup:
            # 过滤掉 UserWarning 类型的警告消息 "This window is not suitable"
            sup.filter(UserWarning, "This window is not suitable")
            # 生成低衰减的偶数阶 Chebyshev 窗函数（阶数为 8，衰减为 -10dB）
            cheb_even = windows.chebwin(8, at=-10)
        # 检查生成的窗函数是否与预期的真实值几乎相等
        assert_array_almost_equal(cheb_even, cheb_even_low_at_true, decimal=4)
# 定义一个包含指数窗口函数参数及其对应结果的字典
exponential_data = {
    (4, None, 0.2, False):  # 参数组合：(4, None, 0.2, False)，对应的结果数组
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03]),
    (4, None, 0.2, True):  # 参数组合：(4, None, 0.2, True)，对应的结果数组
        array([0.00055308437014783, 0.0820849986238988,
               0.0820849986238988, 0.00055308437014783]),
    (4, None, 1.0, False):  # 参数组合：(4, None, 1.0, False)，对应的结果数组
        array([0.1353352832366127, 0.36787944117144233, 1.,
               0.36787944117144233]),
    (4, None, 1.0, True):  # 参数组合：(4, None, 1.0, True)，对应的结果数组
        array([0.22313016014842982, 0.60653065971263342,
               0.60653065971263342, 0.22313016014842982]),
    (4, 2, 0.2, False):  # 参数组合：(4, 2, 0.2, False)，对应的结果数组
        array([4.53999297624848542e-05, 6.73794699908546700e-03,
               1.00000000000000000e+00, 6.73794699908546700e-03]),
    (4, 2, 0.2, True):  # 参数组合：(4, 2, 0.2, True)，结果为 None
        None,
    (4, 2, 1.0, False):  # 参数组合：(4, 2, 1.0, False)，对应的结果数组
        array([0.1353352832366127, 0.36787944117144233, 1.,
               0.36787944117144233]),
    (4, 2, 1.0, True):  # 参数组合：(4, 2, 1.0, True)，结果为 None
        None,
    (5, None, 0.2, True):  # 参数组合：(5, None, 0.2, True)，对应的结果数组
        array([4.53999297624848542e-05,
               6.73794699908546700e-03, 1.00000000000000000e+00,
               6.73794699908546700e-03, 4.53999297624848542e-05]),
    (5, None, 1.0, True):  # 参数组合：(5, None, 1.0, True)，对应的结果数组
        array([0.1353352832366127, 0.36787944117144233, 1.,
               0.36787944117144233, 0.1353352832366127]),
    (5, 2, 0.2, True):  # 参数组合：(5, 2, 0.2, True)，结果为 None
        None,
    (5, 2, 1.0, True):  # 参数组合：(5, 2, 1.0, True)，结果为 None
        None
}

def test_exponential():
    # 遍历指数窗口函数数据字典中的每个项
    for k, v in exponential_data.items():
        if v is None:
            # 如果结果为 None，则断言应该抛出 ValueError 异常
            assert_raises(ValueError, windows.exponential, *k)
        else:
            # 否则，调用指数窗口函数，与预期结果进行数值比较
            win = windows.exponential(*k)
            assert_allclose(win, v, rtol=1e-14)

class TestFlatTop:

    def test_basic(self):
        # 断言对称和非对称情况下生成的 FlatTop 窗口函数的数值与预期值匹配
        assert_allclose(windows.flattop(6, sym=False),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156])
        assert_allclose(windows.flattop(7, sym=False),
                        [-0.000421051, -0.03684078115492348,
                         0.01070371671615342, 0.7808739149387698,
                         0.7808739149387698, 0.01070371671615342,
                         -0.03684078115492348])
        assert_allclose(windows.flattop(6),
                        [-0.000421051, -0.0677142520762119, 0.6068721525762117,
                         0.6068721525762117, -0.0677142520762119,
                         -0.000421051])
        assert_allclose(windows.flattop(7, True),
                        [-0.000421051, -0.051263156, 0.19821053, 1.0,
                         0.19821053, -0.051263156, -0.000421051])

class TestGaussian:
    # 定义测试方法 test_basic(self)，用于测试 windows 模块中的 gaussian 函数
    def test_basic(self):
        # 断言 gaussian 函数计算结果与期望值的近似程度
        assert_allclose(windows.gaussian(6, 1.0),
                        [0.04393693362340742, 0.3246524673583497,
                         0.8824969025845955, 0.8824969025845955,
                         0.3246524673583497, 0.04393693362340742])
        # 断言 gaussian 函数计算结果与期望值的近似程度
        assert_allclose(windows.gaussian(7, 1.2),
                        [0.04393693362340742, 0.2493522087772962,
                         0.7066482778577162, 1.0, 0.7066482778577162,
                         0.2493522087772962, 0.04393693362340742])
        # 断言 gaussian 函数计算结果与期望值的近似程度
        assert_allclose(windows.gaussian(7, 3),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081, 0.6065306597126334])
        # 断言 gaussian 函数计算结果与期望值的近似程度，关闭对称性
        assert_allclose(windows.gaussian(6, 3, False),
                        [0.6065306597126334, 0.8007374029168081,
                         0.9459594689067654, 1.0, 0.9459594689067654,
                         0.8007374029168081])
class TestGeneralCosine:
    # 定义测试类 TestGeneralCosine

    def test_basic(self):
        # 定义基本测试方法 test_basic
        assert_allclose(windows.general_cosine(5, [0.5, 0.3, 0.2]),
                        [0.4, 0.3, 1, 0.3, 0.4])
        # 调用 windows 模块的 general_cosine 函数进行测试，期望结果为 [0.4, 0.3, 1, 0.3, 0.4]

        assert_allclose(windows.general_cosine(4, [0.5, 0.3, 0.2], sym=False),
                        [0.4, 0.3, 1, 0.3])
        # 调用 windows 模块的 general_cosine 函数进行测试，不对称情况下期望结果为 [0.4, 0.3, 1, 0.3]


class TestGeneralHamming:
    # 定义测试类 TestGeneralHamming

    def test_basic(self):
        # 定义基本测试方法 test_basic
        assert_allclose(windows.general_hamming(5, 0.7),
                        [0.4, 0.7, 1.0, 0.7, 0.4])
        # 调用 windows 模块的 general_hamming 函数进行测试，期望结果为 [0.4, 0.7, 1.0, 0.7, 0.4]

        assert_allclose(windows.general_hamming(5, 0.75, sym=False),
                        [0.5, 0.6727457514, 0.9522542486,
                         0.9522542486, 0.6727457514])
        # 调用 windows 模块的 general_hamming 函数进行测试，不对称情况下期望结果为 [0.5, 0.6727457514, 0.9522542486, 0.9522542486, 0.6727457514]

        assert_allclose(windows.general_hamming(6, 0.75, sym=True),
                        [0.5, 0.6727457514, 0.9522542486,
                        0.9522542486, 0.6727457514, 0.5])
        # 调用 windows 模块的 general_hamming 函数进行测试，对称情况下期望结果为 [0.5, 0.6727457514, 0.9522542486, 0.9522542486, 0.6727457514, 0.5]


class TestHamming:
    # 定义测试类 TestHamming

    def test_basic(self):
        # 定义基本测试方法 test_basic
        assert_allclose(windows.hamming(6, False),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31])
        # 调用 windows 模块的 hamming 函数进行测试，不对称情况下期望结果为 [0.08, 0.31, 0.77, 1.0, 0.77, 0.31]

        assert_allclose(windows.hamming(7, sym=False),
                        [0.08, 0.2531946911449826, 0.6423596296199047,
                         0.9544456792351128, 0.9544456792351128,
                         0.6423596296199047, 0.2531946911449826])
        # 调用 windows 模块的 hamming 函数进行测试，不对称情况下期望结果为 [0.08, 0.2531946911449826, 0.6423596296199047, 0.9544456792351128, 0.9544456792351128, 0.6423596296199047, 0.2531946911449826]

        assert_allclose(windows.hamming(6),
                        [0.08, 0.3978521825875242, 0.9121478174124757,
                         0.9121478174124757, 0.3978521825875242, 0.08])
        # 调用 windows 模块的 hamming 函数进行测试，期望结果为 [0.08, 0.3978521825875242, 0.9121478174124757, 0.9121478174124757, 0.3978521825875242, 0.08]

        assert_allclose(windows.hamming(7, sym=True),
                        [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])
        # 调用 windows 模块的 hamming 函数进行测试，对称情况下期望结果为 [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08]


class TestHann:
    # 定义测试类 TestHann

    def test_basic(self):
        # 定义基本测试方法 test_basic
        assert_allclose(windows.hann(6, sym=False),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25],
                        rtol=1e-15, atol=1e-15)
        # 调用 windows 模块的 hann 函数进行测试，不对称情况下期望结果为 [0, 0.25, 0.75, 1.0, 0.75, 0.25]

        assert_allclose(windows.hann(7, sym=False),
                        [0, 0.1882550990706332, 0.6112604669781572,
                         0.9504844339512095, 0.9504844339512095,
                         0.6112604669781572, 0.1882550990706332],
                        rtol=1e-15, atol=1e-15)
        # 调用 windows 模块的 hann 函数进行测试，不对称情况下期望结果为 [0, 0.1882550990706332, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781572, 0.1882550990706332]

        assert_allclose(windows.hann(6, True),
                        [0, 0.3454915028125263, 0.9045084971874737,
                         0.9045084971874737, 0.3454915028125263, 0],
                        rtol=1e-15, atol=1e-15)
        # 调用 windows 模块的 hann 函数进行测试，对称情况下期望结果为 [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737, 0.3454915028125263, 0]

        assert_allclose(windows.hann(7),
                        [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0],
                        rtol=1e-15, atol=1e-15)
        # 调用 windows 模块的 hann 函数进行测试，期望结果为 [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0]


class TestKaiser:
    # 定义测试类 TestKaiser，此处暂无代码，略过
    # 定义测试方法 test_basic，用于测试 windows 模块中 kaiser 函数的不同参数情况
    def test_basic(self):
        # 断言调用 windows 模块中的 kaiser 函数，检查返回结果是否与预期接近
        assert_allclose(windows.kaiser(6, 0.5),
                        [0.9403061933191572, 0.9782962393705389,
                         0.9975765035372042, 0.9975765035372042,
                         0.9782962393705389, 0.9403061933191572])
        assert_allclose(windows.kaiser(7, 0.5),
                        [0.9403061933191572, 0.9732402256999829,
                         0.9932754654413773, 1.0, 0.9932754654413773,
                         0.9732402256999829, 0.9403061933191572])
        assert_allclose(windows.kaiser(6, 2.7),
                        [0.2603047507678832, 0.6648106293528054,
                         0.9582099802511439, 0.9582099802511439,
                         0.6648106293528054, 0.2603047507678832])
        assert_allclose(windows.kaiser(7, 2.7),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844, 0.2603047507678832])
        # 测试带有参数 'False' 的 kaiser 函数调用，期望返回结果是否与预期接近
        assert_allclose(windows.kaiser(6, 2.7, False),
                        [0.2603047507678832, 0.5985765418119844,
                         0.8868495172060835, 1.0, 0.8868495172060835,
                         0.5985765418119844])
class TestKaiserBesselDerived:

    def test_basic(self):
        M = 100
        # 调用 windows 模块的 kaiser_bessel_derived 函数，生成长度为 M 的 Kaiser-Bessel Derived 窗口
        w = windows.kaiser_bessel_derived(M, beta=4.0)
        # 调用 windows 模块的 get_window 函数，生成长度为 M 的 Kaiser-Bessel Derived 窗口
        w2 = windows.get_window(('kaiser bessel derived', 4.0),
                                M, fftbins=False)
        # 使用 assert_allclose 检查两个窗口 w 和 w2 的近似相等性
        assert_allclose(w, w2)

        # 检查 Princen-Bradley 条件是否成立，即窗口前半部分和后半部分的平方和是否为 1
        assert_allclose(w[:M // 2] ** 2 + w[-M // 2:] ** 2, 1.)

        # 测试从其他实现中得到的实际值
        # M = 2 时的值为 sqrt(2) / 2
        assert_allclose(windows.kaiser_bessel_derived(2, beta=np.pi / 2)[:1],
                        np.sqrt(2) / 2)

        # M = 4 时的值为 0.518562710536, 0.855039598640
        assert_allclose(windows.kaiser_bessel_derived(4, beta=np.pi / 2)[:2],
                        [0.518562710536, 0.855039598640])

        # M = 6 时的值为 0.436168993154, 0.707106781187, 0.899864772847
        assert_allclose(windows.kaiser_bessel_derived(6, beta=np.pi / 2)[:3],
                        [0.436168993154, 0.707106781187, 0.899864772847])

    def test_exceptions(self):
        M = 100
        # 断言窗口长度为奇数时会引发 ValueError
        msg = ("Kaiser-Bessel Derived windows are only defined for even "
               "number of points")
        with assert_raises(ValueError, match=msg):
            windows.kaiser_bessel_derived(M + 1, beta=4.)

        # 断言非对称设置时会引发 ValueError
        msg = ("Kaiser-Bessel Derived windows are only defined for "
               "symmetric shapes")
        with assert_raises(ValueError, match=msg):
            windows.kaiser_bessel_derived(M + 1, beta=4., sym=False)


class TestNuttall:

    def test_basic(self):
        # 检查 Nuttall 窗口在不同参数下的实际值是否符合预期
        assert_allclose(windows.nuttall(6, sym=False),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345])
        assert_allclose(windows.nuttall(7, sym=False),
                        [0.0003628, 0.03777576895352025, 0.3427276199688195,
                         0.8918518610776603, 0.8918518610776603,
                         0.3427276199688196, 0.0377757689535203])
        assert_allclose(windows.nuttall(6),
                        [0.0003628, 0.1105152530498718, 0.7982580969501282,
                         0.7982580969501283, 0.1105152530498719, 0.0003628])
        assert_allclose(windows.nuttall(7, True),
                        [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
                         0.0613345, 0.0003628])


class TestParzen:
    # 待补充
    # 定义一个单元测试方法，用于测试 windows 模块中的 parzen 函数的基本功能
    def test_basic(self):
        # 断言调用 windows 模块中的 parzen 函数，检查其返回值是否接近给定的列表
        assert_allclose(windows.parzen(6),
                        [0.009259259259259254, 0.25, 0.8611111111111112,
                         0.8611111111111112, 0.25, 0.009259259259259254])
        # 断言调用 windows 模块中的 parzen 函数，带有 sym=True 参数，检查返回值是否接近给定的列表
        assert_allclose(windows.parzen(7, sym=True),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616, 0.00583090379008747])
        # 断言调用 windows 模块中的 parzen 函数，带有 False 参数，检查返回值是否接近给定的列表
        assert_allclose(windows.parzen(6, False),
                        [0.00583090379008747, 0.1574344023323616,
                         0.6501457725947521, 1.0, 0.6501457725947521,
                         0.1574344023323616])
class TestTriang:

    def test_basic(self):
        # 断言检查窗口函数triang的输出是否符合预期值
        assert_allclose(windows.triang(6, True),
                        [1/6, 1/2, 5/6, 5/6, 1/2, 1/6])
        # 断言检查窗口函数triang的输出是否符合预期值
        assert_allclose(windows.triang(7),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4])
        # 断言检查窗口函数triang的输出是否符合预期值
        assert_allclose(windows.triang(6, sym=False),
                        [1/4, 1/2, 3/4, 1, 3/4, 1/2])


tukey_data = {
    (4, 0.5, True): array([0.0, 1.0, 1.0, 0.0]),
    (4, 0.9, True): array([0.0, 0.84312081893436686,
                           0.84312081893436686, 0.0]),
    (4, 1.0, True): array([0.0, 0.75, 0.75, 0.0]),
    (4, 0.5, False): array([0.0, 1.0, 1.0, 1.0]),
    (4, 0.9, False): array([0.0, 0.58682408883346526,
                            1.0, 0.58682408883346526]),
    (4, 1.0, False): array([0.0, 0.5, 1.0, 0.5]),
    (5, 0.0, True): array([1.0, 1.0, 1.0, 1.0, 1.0]),
    (5, 0.8, True): array([0.0, 0.69134171618254492,
                           1.0, 0.69134171618254492, 0.0]),
    (5, 1.0, True): array([0.0, 0.5, 1.0, 0.5, 0.0]),

    (6, 0): [1, 1, 1, 1, 1, 1],
    (7, 0): [1, 1, 1, 1, 1, 1, 1],
    (6, .25): [0, 1, 1, 1, 1, 0],
    (7, .25): [0, 1, 1, 1, 1, 1, 0],
    (6,): [0, 0.9045084971874737, 1.0, 1.0, 0.9045084971874735, 0],
    (7,): [0, 0.75, 1.0, 1.0, 1.0, 0.75, 0],
    (6, .75): [0, 0.5522642316338269, 1.0, 1.0, 0.5522642316338267, 0],
    (7, .75): [0, 0.4131759111665348, 0.9698463103929542, 1.0,
               0.9698463103929542, 0.4131759111665347, 0],
    (6, 1): [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737,
             0.3454915028125263, 0],
    (7, 1): [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0],
}


class TestTukey:

    def test_basic(self):
        # 对每个 tukey_data 中的键值对进行测试
        for k, v in tukey_data.items():
            if v is None:
                # 如果预期输出是 None，断言应引发 ValueError
                assert_raises(ValueError, windows.tukey, *k)
            else:
                # 否则，使用 windows.tukey 函数计算结果并与预期输出 v 进行比较
                win = windows.tukey(*k)
                assert_allclose(win, v, rtol=1e-15, atol=1e-15)

    def test_extremes(self):
        # 检查 alpha 的极端值是否对应于 boxcar 和 hann 窗口函数
        tuk0 = windows.tukey(100, 0)
        box0 = windows.boxcar(100)
        assert_array_almost_equal(tuk0, box0)

        tuk1 = windows.tukey(100, 1)
        han1 = windows.hann(100)
        assert_array_almost_equal(tuk1, han1)


dpss_data = {
    # MATLAB 中的所有值:
    # * (3, 1.4, 3) 的第一个 tap sign-flipped
    # * (5, 1.5, 5) 的第三个 tap sign-flipped
    (4, 0.1, 2): ([[0.497943898, 0.502047681, 0.502047681, 0.497943898], [0.670487993, 0.224601537, -0.224601537, -0.670487993]], [0.197961815, 0.002035474]),  # noqa: E501
    (3, 1.4, 3): ([[0.410233151, 0.814504464, 0.410233151], [0.707106781, 0.0, -0.707106781], [0.575941629, -0.580157287, 0.575941629]], [0.999998093, 0.998067480, 0.801934426]),  # noqa: E501
}
    (5, 1.5, 5): ([[0.1745071052, 0.4956749177, 0.669109327, 0.495674917, 0.174507105], [0.4399493348, 0.553574369, 0.0, -0.553574369, -0.439949334], [0.631452756, 0.073280238, -0.437943884, 0.073280238, 0.631452756], [0.553574369, -0.439949334, 0.0, 0.439949334, -0.553574369], [0.266110290, -0.498935248, 0.600414741, -0.498935248, 0.266110290147157]], [0.999728571, 0.983706916, 0.768457889, 0.234159338, 0.013947282907567]),  # noqa: E501



# 定义一个包含三个元素的元组
# - 第一个元素是一个包含五个列表的列表，每个列表包含五个浮点数
# - 第二个元素是一个包含五个浮点数的列表
# - 最后一个元素是一个包含五个浮点数的列表
# 这个元组的结构用于表示复杂的数学数据结构
    }

# 测试 DPSS 窗函数的相关类
class TestDPSS:

    # 测试基本功能
    def test_basic(self):
        # 对预设的数据进行测试
        for k, v in dpss_data.items():
            # 调用 dpss 函数生成窗函数和比率
            win, ratios = windows.dpss(*k, return_ratios=True)
            # 检查窗函数是否接近预期值，允许的绝对误差为 1e-7
            assert_allclose(win, v[0], atol=1e-7, err_msg=k)
            # 检查比率是否接近预期值，允许的相对误差为 1e-5 和绝对误差为 1e-7
            assert_allclose(ratios, v[1], rtol=1e-5, atol=1e-7, err_msg=k)

    # 测试单位值处理
    def test_unity(self):
        # 测试单位值处理（gh-2221）
        for M in range(1, 21):
            # 使用默认设置，生成修正的 DPSS 窗函数
            win = windows.dpss(M, M / 2.1)
            # 预期为奇数时为 1，偶数时为 0
            expected = M % 2
            assert_equal(np.isclose(win, 1.).sum(), expected,
                         err_msg=f'{win}')
            # 使用子采样延迟模式生成修正的 DPSS 窗函数
            win_sub = windows.dpss(M, M / 2.1, norm='subsample')
            if M > 2:
                # 当 M > 2 时，子采样不产生变化
                assert_equal(np.isclose(win_sub, 1.).sum(), expected,
                             err_msg=f'{win_sub}')
                # 检查两种模式下的窗函数是否在 3% 范围内一致
                assert_allclose(win, win_sub, rtol=0.03)
            # 使用 L2 范数生成修正的 DPSS 窗函数
            win_2 = windows.dpss(M, M / 2.1, norm=2)
            # 当 M 为 1 时为 1，否则为 0
            expected = 1 if M == 1 else 0
            assert_equal(np.isclose(win_2, 1.).sum(), expected,
                         err_msg=f'{win_2}')

    # 测试极端情况
    def test_extremes(self):
        # 测试 alpha 的极端值
        lam = windows.dpss(31, 6, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 7, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)
        lam = windows.dpss(31, 8, 4, return_ratios=True)[1]
        assert_array_almost_equal(lam, 1.)

    # 测试退化情况
    def test_degenerate(self):
        # 测试异常情况
        assert_raises(ValueError, windows.dpss, 4, 1.5, -1)  # Bad Kmax
        assert_raises(ValueError, windows.dpss, 4, 1.5, -5)
        assert_raises(TypeError, windows.dpss, 4, 1.5, 1.1)
        assert_raises(ValueError, windows.dpss, 3, 1.5, 3)  # NW must be < N/2.
        assert_raises(ValueError, windows.dpss, 3, -1, 3)  # NW must be pos
        assert_raises(ValueError, windows.dpss, 3, 0, 3)
        assert_raises(ValueError, windows.dpss, -1, 1, 3)  # negative M

# 测试 Lanczos 窗函数的相关类
class TestLanczos:
    def test_basic(self):
        # 分析结果：
        # sinc(x) = sinc(-x)
        # sinc(pi) = 0, sinc(0) = 1
        # 在WolframAlpha上手动计算：
        # sinc(2 pi / 3) = 0.413496672
        # sinc(pi / 3) = 0.826993343
        # sinc(3 pi / 5) = 0.504551152
        # sinc(pi / 5) = 0.935489284
        # 使用assert_allclose函数检查非对称Lanczos窗口的结果是否准确
        assert_allclose(windows.lanczos(6, sym=False),
                        [0., 0.413496672,
                         0.826993343, 1., 0.826993343,
                         0.413496672],
                        atol=1e-9)
        # 使用assert_allclose函数检查对称Lanczos窗口的结果是否准确
        assert_allclose(windows.lanczos(6),
                        [0., 0.504551152,
                         0.935489284, 0.935489284,
                         0.504551152, 0.],
                        atol=1e-9)
        # 使用assert_allclose函数检查对称Lanczos窗口的结果是否准确
        assert_allclose(windows.lanczos(7, sym=True),
                        [0., 0.413496672,
                         0.826993343, 1., 0.826993343,
                         0.413496672, 0.],
                        atol=1e-9)

    def test_array_size(self):
        # 对于数组大小的测试，验证Lanczos窗口函数生成的数组长度是否正确
        for n in [0, 10, 11]:
            # 使用assert_equal函数检查非对称Lanczos窗口的数组长度
            assert_equal(len(windows.lanczos(n, sym=False)), n)
            # 使用assert_equal函数检查对称Lanczos窗口的数组长度
            assert_equal(len(windows.lanczos(n, sym=True)), n)
class TestGetWindow:

    def test_boxcar(self):
        # 调用 windows 模块的 get_window 函数，获取长度为 12 的 boxcar 窗口
        w = windows.get_window('boxcar', 12)
        # 断言 w 数组的每个元素与全为 1 的数组相等
        assert_array_equal(w, np.ones_like(w))

        # 使用长度为 1 的元组作为参数调用 get_window 函数
        w = windows.get_window(('boxcar',), 16)
        # 断言 w 数组的每个元素与全为 1 的数组相等
        assert_array_equal(w, np.ones_like(w))

    def test_cheb_odd(self):
        # 使用 suppress_warnings 上下文管理器，过滤 UserWarning 类型的警告信息
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            # 调用 get_window 函数获取长度为 53 的 chebwin 窗口，不使用 FFT bins
            w = windows.get_window(('chebwin', -40), 53, fftbins=False)
        # 断言 w 数组与预设的 cheb_odd_true 数组几乎相等，精确度为小数点后四位
        assert_array_almost_equal(w, cheb_odd_true, decimal=4)

    def test_cheb_even(self):
        # 使用 suppress_warnings 上下文管理器，过滤 UserWarning 类型的警告信息
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "This window is not suitable")
            # 调用 get_window 函数获取长度为 54 的 chebwin 窗口，不使用 FFT bins
            w = windows.get_window(('chebwin', 40), 54, fftbins=False)
        # 断言 w 数组与预设的 cheb_even_true 数组几乎相等，精确度为小数点后四位
        assert_array_almost_equal(w, cheb_even_true, decimal=4)

    def test_dpss(self):
        # 调用 get_window 函数获取长度为 64 的 dpss 窗口，不使用 FFT bins
        win1 = windows.get_window(('dpss', 3), 64, fftbins=False)
        # 调用 dpss 函数获取长度为 64 的 dpss 窗口
        win2 = windows.dpss(64, 3)
        # 断言 win1 和 win2 数组几乎相等，精确度为小数点后四位
        assert_array_almost_equal(win1, win2, decimal=4)

    def test_kaiser_float(self):
        # 调用 get_window 函数获取长度为 64 的 kaiser 窗口，beta 参数为 7.2
        win1 = windows.get_window(7.2, 64)
        # 调用 kaiser 函数获取长度为 64 的 kaiser 窗口，beta 参数为 7.2，不进行归一化
        win2 = windows.kaiser(64, 7.2, False)
        # 断言 win1 和 win2 数组几乎相等
        assert_allclose(win1, win2)

    def test_invalid_inputs(self):
        # 断言调用 get_window 函数时传入集合对象 'hann' 会引发 ValueError 异常
        assert_raises(ValueError, windows.get_window, set('hann'), 8)

        # 断言调用 get_window 函数时传入未知窗口类型 'broken' 会引发 ValueError 异常
        assert_raises(ValueError, windows.get_window, 'broken', 4)

    def test_array_as_window(self):
        # github 问题 3603
        osfactor = 128
        sig = np.arange(128)

        # 调用 get_window 函数获取长度为 osfactor//2 的 kaiser 窗口，beta 参数为 8.0
        win = windows.get_window(('kaiser', 8.0), osfactor // 2)
        # 使用 assert_raises 断言 resample 函数在使用 win 作为窗口参数时会引发 ValueError 异常
        with assert_raises(ValueError, match='must have the same length'):
            resample(sig, len(sig) * osfactor, window=win)

    def test_general_cosine(self):
        # 断言调用 get_window 函数生成 'general_cosine' 窗口，并与预设数组比较是否几乎相等
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4),
                        [0.4, 0.3, 1, 0.3])
        # 断言调用 get_window 函数生成 'general_cosine' 窗口，不使用 FFT bins，并与预设数组比较是否几乎相等
        assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4,
                                   fftbins=False),
                        [0.4, 0.55, 0.55, 0.4])

    def test_general_hamming(self):
        # 断言调用 get_window 函数生成 'general_hamming' 窗口，并与预设数组比较是否几乎相等
        assert_allclose(get_window(('general_hamming', 0.7), 5),
                        [0.4, 0.6072949, 0.9427051, 0.9427051, 0.6072949])
        # 断言调用 get_window 函数生成 'general_hamming' 窗口，不使用 FFT bins，并与预设数组比较是否几乎相等
        assert_allclose(get_window(('general_hamming', 0.7), 5, fftbins=False),
                        [0.4, 0.7, 1.0, 0.7, 0.4])

    def test_lanczos(self):
        # 断言调用 get_window 函数生成 'lanczos' 窗口，并与预设数组比较是否几乎相等
        assert_allclose(get_window('lanczos', 6),
                        [0., 0.413496672, 0.826993343, 1., 0.826993343,
                         0.413496672], atol=1e-9)
        # 断言调用 get_window 函数生成 'lanczos' 窗口，不使用 FFT bins，并与预设数组比较是否几乎相等
        assert_allclose(get_window('lanczos', 6, fftbins=False),
                        [0., 0.504551152, 0.935489284, 0.935489284,
                         0.504551152, 0.], atol=1e-9)
        # 断言调用 get_window 函数生成 'lanczos' 窗口与 'sinc' 窗口相等
        assert_allclose(get_window('lanczos', 6), get_window('sinc', 6))


def test_windowfunc_basics():
    # 这是一个空函数，没有任何操作
    for window_name, params in window_funcs:
        # 根据窗口函数名称从 windows 模块中获取相应的窗口函数对象
        window = getattr(windows, window_name)
        
        # 使用 suppress_warnings 上下文管理器来捕获 UserWarning，避免其打印到控制台
        with suppress_warnings() as sup:
            # 过滤特定的 UserWarning 提示消息
            sup.filter(UserWarning, "This window is not suitable")
            
            # 检查对称性，分别用于奇数和偶数长度的窗口函数
            w1 = window(8, *params, sym=True)
            w2 = window(7, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            w1 = window(9, *params, sym=True)
            w2 = window(8, *params, sym=False)
            assert_array_almost_equal(w1[:-1], w2)

            # 检查窗口函数执行后输出长度的正确性
            assert_equal(len(window(6, *params, sym=True)), 6)
            assert_equal(len(window(6, *params, sym=False)), 6)
            assert_equal(len(window(7, *params, sym=True)), 7)
            assert_equal(len(window(7, *params, sym=False)), 7)

            # 检查无效长度的处理
            assert_raises(ValueError, window, 5.5, *params)
            assert_raises(ValueError, window, -7, *params)

            # 检查退化情况
            assert_array_equal(window(0, *params, sym=True), [])
            assert_array_equal(window(0, *params, sym=False), [])
            assert_array_equal(window(1, *params, sym=True), [1])
            assert_array_equal(window(1, *params, sym=False), [1])

            # 检查数据类型为浮点型
            assert_(window(0, *params, sym=True).dtype == 'float')
            assert_(window(0, *params, sym=False).dtype == 'float')
            assert_(window(1, *params, sym=True).dtype == 'float')
            assert_(window(1, *params, sym=False).dtype == 'float')
            assert_(window(6, *params, sym=True).dtype == 'float')
            assert_(window(6, *params, sym=False).dtype == 'float')

            # 检查归一化
            assert_array_less(window(10, *params, sym=True), 1.01)
            assert_array_less(window(10, *params, sym=False), 1.01)
            assert_array_less(window(9, *params, sym=True), 1.01)
            assert_array_less(window(9, *params, sym=False), 1.01)

            # 检查 DFT-even 谱在奇数和偶数情况下的纯实性
            assert_allclose(fft(window(10, *params, sym=False)).imag,
                            0, atol=1e-14)
            assert_allclose(fft(window(11, *params, sym=False)).imag,
                            0, atol=1e-14)
# 测试需要参数的窗口函数
def test_needs_params():
    # 遍历需要参数的窗口函数列表
    for winstr in ['kaiser', 'ksr', 'kaiser_bessel_derived', 'kbd',
                   'gaussian', 'gauss', 'gss',
                   'general gaussian', 'general_gaussian',
                   'general gauss', 'general_gauss', 'ggs',
                   'dss', 'dpss', 'general cosine', 'general_cosine',
                   'chebwin', 'cheb', 'general hamming', 'general_hamming',
                   ]:
        # 断言调用获取窗口函数的行为会抛出 ValueError 异常，参数为窗口函数名称和窗口长度
        assert_raises(ValueError, get_window, winstr, 7)


# 测试不需要参数的窗口函数
def test_not_needs_params():
    # 遍历不需要参数的窗口函数列表
    for winstr in ['barthann',
                   'bartlett',
                   'blackman',
                   'blackmanharris',
                   'bohman',
                   'boxcar',
                   'cosine',
                   'flattop',
                   'hamming',
                   'nuttall',
                   'parzen',
                   'taylor',
                   'exponential',
                   'poisson',
                   'tukey',
                   'tuk',
                   'triangle',
                   'lanczos',
                   'sinc',
                   ]:
        # 调用获取窗口函数，参数为窗口函数名称和窗口长度，赋值给变量 win
        win = get_window(winstr, 7)
        # 断言获取的窗口函数的长度为 7
        assert_equal(len(win), 7)


# 测试对称性窗口函数
def test_symmetric():

    # 对于每个窗口函数 windows.lanczos
    for win in [windows.lanczos]:
        # 获取窗口函数 w，参数为采样点数 4096
        w = win(4096)
        # 计算 w 与其反转后的差的最大绝对值，应为 0.0
        error = np.max(np.abs(w-np.flip(w)))
        # 断言 error 等于 0.0
        assert_equal(error, 0.0)

        # 重复上述步骤，但采样点数为 4097
        w = win(4097)
        error = np.max(np.abs(w-np.flip(w)))
        assert_equal(error, 0.0)
```