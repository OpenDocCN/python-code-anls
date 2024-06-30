# `D:\src\scipysrc\scipy\scipy\signal\tests\test_dltisys.py`

```
# 导入需要的库和模块
import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_, assert_almost_equal,
                           suppress_warnings)
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
                          StateSpace, TransferFunction, ZerosPolesGain,
                          dfreqresp, dbode, BadCoefficients)

# 定义测试类 TestDLTI
class TestDLTI:

    # 定义测试函数 test_dstep
    def test_dstep(self):

        # 设定系统的状态空间参数
        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # 因为 b.shape[1] == 3，dstep 应返回三个结果向量的元组
        yout_step_truth = (np.asarray([0.0, 0.04, 0.052, 0.0404, 0.00956,
                                       -0.036324, -0.093318, -0.15782348,
                                       -0.226628324, -0.2969374948]),
                           np.asarray([-0.1, -0.075, -0.058, -0.04815,
                                       -0.04453, -0.0461895, -0.0521812,
                                       -0.061588875, -0.073549579,
                                       -0.08727047595]),
                           np.asarray([0.0, -0.01, -0.013, -0.0101, -0.00239,
                                       0.009081, 0.0233295, 0.03945587,
                                       0.056657081, 0.0742343737]))

        # 调用 dstep 函数进行模拟，并获取结果时间向量 tout 和响应向量 yout
        tout, yout = dstep((a, b, c, d, dt), n=10)

        # 断言结果向量 yout 的长度为 3
        assert_equal(len(yout), 3)

        # 循环检查每一个结果向量的长度和内容是否与预期一致
        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_step_truth[i])

        # 使用传递给 dstep 的其他两种输入（tf, zpk），检查它们的工作情况
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfstep = np.asarray([0.0, 1.0, 0.0])
        tout, yout = dstep(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)

        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dstep(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)

        # 对于连续时间系统，确认引发 AttributeError 错误
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dstep, system)
    # 定义一个测试用例，验证 dimpulse 函数的输出是否正确
    def test_dimpulse(self):

        # 定义系统的状态空间矩阵
        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        # 定义输入矩阵
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        # 定义输出矩阵
        c = np.asarray([[0.1, 0.3]])
        # 定义直通矩阵
        d = np.asarray([[0.0, -0.1, 0.0]])
        # 定义采样时间间隔
        dt = 0.5

        # 期望的 dimpulse 输出结果
        yout_imp_truth = (np.asarray([0.0, 0.04, 0.012, -0.0116, -0.03084,
                                      -0.045884, -0.056994, -0.06450548,
                                      -0.068804844, -0.0703091708]),
                          np.asarray([-0.1, 0.025, 0.017, 0.00985, 0.00362,
                                      -0.0016595, -0.0059917, -0.009407675,
                                      -0.011960704, -0.01372089695]),
                          np.asarray([0.0, -0.01, -0.003, 0.0029, 0.00771,
                                      0.011471, 0.0142485, 0.01612637,
                                      0.017201211, 0.0175772927]))

        # 调用 dimpulse 函数，得到实际输出结果
        tout, yout = dimpulse((a, b, c, d, dt), n=10)

        # 断言实际输出结果的长度与期望输出结果相同
        assert_equal(len(yout), 3)

        # 遍历实际输出结果，断言每个向量的长度与期望输出结果相同，并检查其值接近期望值
        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_imp_truth[i])

        # 检查其他两种输入情况 (tf, zpk) 的工作情况
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfimpulse = np.asarray([0.0, 1.0, -1.0])

        # 调用 dimpulse 函数，断言实际输出结果的长度为1，并检查其值接近期望值
        tout, yout = dimpulse(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)

        # 将 tf 输入转换为 zpk 输入，并调用 dimpulse 函数，断言实际输出结果的长度为1，并检查其值接近期望值
        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dimpulse(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)

        # 对于连续时间系统，断言 dimpulse 函数引发 AttributeError 异常
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dimpulse, system)

    # 定义一个测试用例，验证 dlsim 函数在简单情况下的输出是否正确
    def test_dlsim_trivial(self):
        a = np.array([[0.0]])
        b = np.array([[0.0]])
        c = np.array([[0.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        
        # 调用 dlsim 函数，得到实际输出结果
        tout, yout, xout = dlsim((a, b, c, d, 1), u)

        # 断言输出结果与预期结果相等
        assert_array_equal(tout, np.arange(float(n)))
        assert_array_equal(yout, np.zeros((n, 1)))
        assert_array_equal(xout, np.zeros((n, 1)))

    # 定义一个测试用例，验证 dlsim 函数在简单 1 维情况下的输出是否正确
    def test_dlsim_simple1d(self):
        a = np.array([[0.5]])
        b = np.array([[0.0]])
        c = np.array([[1.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        
        # 调用 dlsim 函数，得到实际输出结果
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)

        # 断言时间输出结果与预期相等
        assert_array_equal(tout, np.arange(float(n)))
        # 断言输出结果与预期结果接近
        expected = (0.5 ** np.arange(float(n))).reshape(-1, 1)
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)
    # 定义一个测试方法，用于测试简单二维系统的离散状态空间模型
    def test_dlsim_simple2d(self):
        # 设置系统参数
        lambda1 = 0.5
        lambda2 = 0.25
        # 系统的状态矩阵
        a = np.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        # 输入矩阵
        b = np.array([[0.0],
                      [0.0]])
        # 输出矩阵
        c = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        # 前馈矩阵
        d = np.array([[0.0],
                      [0.0]])
        # 离散步数
        n = 5
        # 初始输入向量
        u = np.zeros(n).reshape(-1, 1)
        # 调用离散状态空间模型计算
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
        # 断言输出时间向量与期望的时间向量一致
        assert_array_equal(tout, np.arange(float(n)))
        # 断言输出结果与解析解一致
        expected = (np.array([lambda1, lambda2]) **
                                np.arange(float(n)).reshape(-1, 1))
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)

    # 定义另一个测试方法，用于测试更多步骤和冲击响应的离散状态空间模型
    def test_more_step_and_impulse(self):
        # 设置系统参数
        lambda1 = 0.5
        lambda2 = 0.75
        # 系统的状态矩阵
        a = np.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        # 输入矩阵
        b = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        # 输出矩阵
        c = np.array([[1.0, 1.0]])
        # 前馈矩阵
        d = np.array([[0.0, 0.0]])

        # 离散步数
        n = 10

        # 检查阶跃响应
        ts, ys = dstep((a, b, c, d, 1), n=n)

        # 创建精确的阶跃响应
        stp0 = (1.0 / (1 - lambda1)) * (1.0 - lambda1 ** np.arange(n))
        stp1 = (1.0 / (1 - lambda2)) * (1.0 - lambda2 ** np.arange(n))

        assert_allclose(ys[0][:, 0], stp0)
        assert_allclose(ys[1][:, 0], stp1)

        # 检查带有初始条件的冲击响应
        x0 = np.array([1.0, 1.0])
        ti, yi = dimpulse((a, b, c, d, 1), n=n, x0=x0)

        # 创建精确的冲击响应
        imp = (np.array([lambda1, lambda2]) **
                            np.arange(-1, n + 1).reshape(-1, 1))
        imp[0, :] = 0.0
        # 冲击响应的解析解
        y0 = imp[:n, 0] + np.dot(imp[1:n + 1, :], x0)
        y1 = imp[:n, 1] + np.dot(imp[1:n + 1, :], x0)

        assert_allclose(yi[0][:, 0], y0)
        assert_allclose(yi[1][:, 0], y1)

        # 检查当 dt=0.1，n=3 时，是否得到了3个时间值
        system = ([1.0], [1.0, -0.5], 0.1)
        t, (y,) = dstep(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1.0, 1.5]])
        t, (y,) = dimpulse(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1, 0.5]])
class TestDlti:
    def test_dlti_instantiation(self):
        # Test that lti can be instantiated.

        dt = 0.05
        # TransferFunction
        s = dlti([1], [-1], dt=dt)
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # ZerosPolesGain
        s = dlti(np.array([]), np.array([-1]), 1, dt=dt)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # StateSpace
        s = dlti([1], [-1], 1, 3, dt=dt)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # Number of inputs
        assert_raises(ValueError, dlti, 1)
        assert_raises(ValueError, dlti, 1, 1, 1, 1, 1)

class TestStateSpaceDisc:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        StateSpace(1, 1, 1, 1, dt=dt)
        StateSpace([1], [2], [3], [4], dt=dt)
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
                   np.array([[1, 0]]), np.array([[0]]), dt=dt)
        StateSpace(1, 1, 1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = StateSpace(1, 2, 3, 4, dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_tf() and to_zpk()

        # Getters
        s = StateSpace(1, 1, 1, 1, dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])

class TestTransferFunction:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        TransferFunction(1, 1, dt=dt)
        TransferFunction([1], [2], dt=dt)
        TransferFunction(np.array([1]), np.array([2]), dt=dt)
        TransferFunction(1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(TransferFunction(s) is not s)
        assert_(s.to_tf() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_ss() and to_zpk()

        # Getters
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])

class TestZerosPolesGain:
    # Tests for ZerosPolesGain class are not provided here.
    # Depending on the actual implementation, additional tests may be required.
    def test_initialization(self):
        # 检查所有初始化是否正常工作
        dt = 0.05  # 设定时间步长为0.05秒
        ZerosPolesGain(1, 1, 1, dt=dt)  # 创建 ZerosPolesGain 对象，传入参数为1, 1, 1，并指定时间步长
        ZerosPolesGain([1], [2], 1, dt=dt)  # 创建 ZerosPolesGain 对象，传入参数为 [1], [2], 1，并指定时间步长
        ZerosPolesGain(np.array([1]), np.array([2]), 1, dt=dt)  # 创建 ZerosPolesGain 对象，传入参数为 np.array([1]), np.array([2]), 1，并指定时间步长
        ZerosPolesGain(1, 1, 1, dt=True)  # 创建 ZerosPolesGain 对象，传入参数为1, 1, 1，并指定时间步长为True（可能是个错误）

    def test_conversion(self):
        # 检查转换函数是否正常工作
        s = ZerosPolesGain(1, 2, 3, dt=0.05)  # 创建 ZerosPolesGain 对象，传入参数为1, 2, 3，并指定时间步长为0.05秒
        assert_(isinstance(s.to_ss(), StateSpace))  # 断言对象 s 转换为 StateSpace 类型
        assert_(isinstance(s.to_tf(), TransferFunction))  # 断言对象 s 转换为 TransferFunction 类型
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))  # 断言对象 s 转换为 ZerosPolesGain 类型

        # 确保复制功能正常工作
        assert_(ZerosPolesGain(s) is not s)  # 断言创建的 ZerosPolesGain 对象与 s 不是同一个对象
        assert_(s.to_zpk() is not s)  # 断言对象 s 转换为 ZerosPolesGain 类型后不是同一个对象
class Test_dfreqresp:

    def test_manual(self):
        # Test dfreqresp() real part calculation (manual sanity check).
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        # 创建一个一阶低通滤波器系统：H(z) = 1 / (z - 0.2)
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        # 指定频率数组
        w = [0.1, 1, 10]
        # 调用 dfreqresp 函数计算系统的频率响应
        w, H = dfreqresp(system, w=w)

        # test real
        # 测试实部
        expected_re = [1.2383, 0.4130, -0.7553]
        assert_almost_equal(H.real, expected_re, decimal=4)

        # test imag
        # 测试虚部
        expected_im = [-0.1555, -1.0214, 0.3955]
        assert_almost_equal(H.imag, expected_im, decimal=4)

    def test_auto(self):
        # Test dfreqresp() real part calculation.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        # 创建一个一阶低通滤波器系统：H(z) = 1 / (z - 0.2)
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        # 指定频率数组
        w = [0.1, 1, 10, 100]
        # 调用 dfreqresp 函数计算系统的频率响应
        w, H = dfreqresp(system, w=w)
        # 计算复频率响应
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)

        # test real
        # 测试实部
        expected_re = y.real
        assert_almost_equal(H.real, expected_re)

        # test imag
        # 测试虚部
        expected_im = y.imag
        assert_almost_equal(H.imag, expected_im)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        # 期望的频率范围是从 0.01 到 10
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        n = 10
        expected_w = np.linspace(0, np.pi, 10, endpoint=False)
        # 调用 dfreqresp 函数计算系统的频率响应，并检查频率数组
        w, H = dfreqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        # 创建一个具有零极点的系统：H(s) = 1 / s
        system = TransferFunction([1], [1, -1], dt=0.1)

        with suppress_warnings() as sup:
            # 忽略特定警告信息
            sup.filter(RuntimeWarning, message="divide by zero")
            sup.filter(RuntimeWarning, message="invalid value encountered")
            # 调用 dfreqresp 函数计算系统的频率响应，并获取频率和响应
            w, H = dfreqresp(system, n=2)
        assert_equal(w[0], 0.)  # a fail would give not-a-number

    def test_error(self):
        # Raise an error for continuous-time systems
        # 创建一个连续时间系统对象
        system = lti([1], [1, 1])
        # 断言调用 dfreqresp 函数会引发 AttributeError 异常
        assert_raises(AttributeError, dfreqresp, system)

    def test_from_state_space(self):
        # H(z) = 2 / z^3 - 0.5 * z^2
        # 创建一个离散状态空间系统对象
        system_TF = dlti([2], [1, -0.5, 0, 0])

        A = np.array([[0.5, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
        B = np.array([[1, 0, 0]]).T
        C = np.array([[0, 0, 2]])
        D = 0

        # 创建另一个离散状态空间系统对象
        system_SS = dlti(A, B, C, D)
        # 指定频率数组
        w = 10.0**np.arange(-3,0,.5)
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            # 调用 dfreqresp 函数计算系统的频率响应
            w1, H1 = dfreqresp(system_TF, w=w)
            w2, H2 = dfreqresp(system_SS, w=w)

        # 断言两个系统的频率响应近似相等
        assert_almost_equal(H1, H2)
    # 定义一个测试方法，用于测试从零极点增益形式（ZPK）转换为传递函数（TF）的过程
    def test_from_zpk(self):
        # 定义一个一阶低通滤波器的 ZPK 形式：H(s) = 0.3 / (s - 0.2)
        system_ZPK = dlti([], [0.2], 0.3)
        # 定义相同低通滤波器的传递函数（TF）形式：H(s) = 0.3 / (s - 0.2)
        system_TF = dlti(0.3, [1, -0.2])
        # 定义频率数组
        w = [0.1, 1, 10, 100]
        # 计算 ZPK 形式滤波器的频率响应
        w1, H1 = dfreqresp(system_ZPK, w=w)
        # 计算 TF 形式滤波器的频率响应
        w2, H2 = dfreqresp(system_TF, w=w)
        # 断言两种形式的频率响应近似相等
        assert_almost_equal(H1, H2)
class Test_bode:

    def test_manual(self):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),

        # 定义采样间隔
        dt = 0.1
        # 创建传递函数对象表示1阶低通滤波器 H(s) = 0.3 / (z - 0.2)，使用给定的采样间隔
        system = TransferFunction(0.3, [1, -0.2], dt=dt)
        # 定义测试频率
        w = [0.1, 0.5, 1, np.pi]
        # 调用 dbode() 函数计算系统的频率响应特性
        w2, mag, phase = dbode(system, w=w)

        # Test mag
        # 预期的幅度响应
        expected_mag = [-8.5329, -8.8396, -9.6162, -12.0412]
        # 断言实际计算得到的幅度响应与预期值接近
        assert_almost_equal(mag, expected_mag, decimal=4)

        # Test phase
        # 预期的相位响应
        expected_phase = [-7.1575, -35.2814, -67.9809, -180.0000]
        # 断言实际计算得到的相位响应与预期值接近
        assert_almost_equal(phase, expected_phase, decimal=4)

        # Test frequency
        # 断言计算得到的频率与预期的频率相等
        assert_equal(np.array(w) / dt, w2)

    def test_auto(self):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        # 创建传递函数对象表示1阶低通滤波器 H(s) = 0.3 / (z - 0.2)，使用默认的采样间隔0.1
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        # 定义测试频率
        w = np.array([0.1, 0.5, 1, np.pi])
        # 调用 dbode() 函数计算系统的频率响应特性
        w2, mag, phase = dbode(system, w=w)
        # 计算复数频率响应
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)

        # Test mag
        # 计算预期的幅度响应
        expected_mag = 20.0 * np.log10(abs(y))
        # 断言实际计算得到的幅度响应与预期值接近
        assert_almost_equal(mag, expected_mag)

        # Test phase
        # 计算预期的相位响应
        expected_phase = np.rad2deg(np.angle(y))
        # 断言实际计算得到的相位响应与预期值接近
        assert_almost_equal(phase, expected_phase)

    def test_range(self):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        # 定义采样间隔
        dt = 0.1
        # 创建传递函数对象表示1阶低通滤波器 H(s) = 0.3 / (z - 0.2)，使用默认的采样间隔0.1
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        # 定义测试点数量
        n = 10
        # 期望的频率范围是从0.01到10
        expected_w = np.linspace(0, np.pi, n, endpoint=False) / dt
        # 调用 dbode() 函数计算系统的频率响应特性
        w, mag, phase = dbode(system, n=n)
        # 断言计算得到的频率范围与期望的频率范围接近
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        # 创建传递函数对象表示具有零点在零处的积分器系统 H(s) = 1 / s，使用默认的采样间隔0.1
        system = TransferFunction([1], [1, -1], dt=0.1)

        # 使用 suppress_warnings() 上下文管理器，忽略特定的警告信息
        with suppress_warnings() as sup:
            # 过滤 "divide by zero" 的运行时警告
            sup.filter(RuntimeWarning, message="divide by zero")
            # 过滤 "invalid value encountered" 的运行时警告
            sup.filter(RuntimeWarning, message="invalid value encountered")
            # 调用 dbode() 函数计算系统的频率响应特性
            w, mag, phase = dbode(system, n=2)
        # 断言计算得到的第一个频率为0
        assert_equal(w[0], 0.)  # a fail would give not-a-number

    def test_imaginary(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        # 创建传递函数对象表示具有纯虚数极点的系统 H(s) = 1 / (s^2 + 100)，使用默认的采样间隔0.1
        system = TransferFunction([1], [1, 0, 100], dt=0.1)
        # 调用 dbode() 函数计算系统的频率响应特性
        dbode(system, n=2)

    def test_error(self):
        # Raise an error for continuous-time systems
        # 创建具有连续时间系统的传递函数对象，这里故意使用了一个假设上不存在的对象 lti
        system = lti([1], [1, 1])
        # 断言调用 dbode() 函数时会引发 AttributeError 异常
        assert_raises(AttributeError, dbode, system)
    def test_full(self):
        # Numerator and denominator same order
        num = [2, 3, 4]  # 定义分子
        den = [5, 6, 7]  # 定义分母
        num2, den2 = TransferFunction._z_to_zinv(num, den)  # 调用函数将 z 转换为 zinv
        assert_equal(num, num2)  # 断言分子转换后的结果与原始分子相同
        assert_equal(den, den2)  # 断言分母转换后的结果与原始分母相同

        num2, den2 = TransferFunction._zinv_to_z(num, den)  # 再次调用函数将 zinv 转换为 z
        assert_equal(num, num2)  # 断言分子转换后的结果与原始分子相同
        assert_equal(den, den2)  # 断言分母转换后的结果与原始分母相同

    def test_numerator(self):
        # Numerator lower order than denominator
        num = [2, 3]  # 定义较低阶的分子
        den = [5, 6, 7]  # 定义较高阶的分母
        num2, den2 = TransferFunction._z_to_zinv(num, den)  # 调用函数将 z 转换为 zinv
        assert_equal([0, 2, 3], num2)  # 断言分子转换后的结果为在高阶分子前添加了零
        assert_equal(den, den2)  # 断言分母转换后的结果与原始分母相同

        num2, den2 = TransferFunction._zinv_to_z(num, den)  # 再次调用函数将 zinv 转换为 z
        assert_equal([2, 3, 0], num2)  # 断言分子转换后的结果为在低阶分子后添加了零
        assert_equal(den, den2)  # 断言分母转换后的结果与原始分母相同

    def test_denominator(self):
        # Numerator higher order than denominator
        num = [2, 3, 4]  # 定义较高阶的分子
        den = [5, 6]  # 定义较低阶的分母
        num2, den2 = TransferFunction._z_to_zinv(num, den)  # 调用函数将 z 转换为 zinv
        assert_equal(num, num2)  # 断言分子转换后的结果与原始分子相同
        assert_equal([0, 5, 6], den2)  # 断言分母转换后的结果为在高阶分母前添加了零

        num2, den2 = TransferFunction._zinv_to_z(num, den)  # 再次调用函数将 zinv 转换为 z
        assert_equal(num, num2)  # 断言分子转换后的结果与原始分子相同
        assert_equal([5, 6, 0], den2)  # 断言分母转换后的结果为在低阶分母后添加了零
```