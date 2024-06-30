# `D:\src\scipysrc\scipy\scipy\signal\tests\test_cont2discrete.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import \  # 导入 NumPy 测试模块的部分函数
                          assert_array_almost_equal, assert_almost_equal, \
                          assert_allclose, assert_equal

import pytest  # 导入 pytest 库，用于编写和运行测试
from scipy.signal import cont2discrete as c2d  # 导入信号处理模块中的函数
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti  # 导入更多信号处理相关函数
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep  # 继续导入信号处理函数

# 作者信息和编写日期
# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# March 29, 2011

class TestC2D:
    def test_zoh(self):
        ac = np.eye(2)  # 创建一个2x2的单位矩阵 ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组 bc，元素均为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组 cc
        dc = np.array([[0.0], [0.0], [-0.33]])  # 创建一个3x1的数组 dc

        ad_truth = 1.648721270700128 * np.eye(2)  # 预期的 ad 矩阵
        bd_truth = np.full((2, 1), 0.324360635350064)  # 预期的 bd 矩阵
        # 在离散情况下，cc 和 dd 应该与它们的连续时间对应部分相等
        dt_requested = 0.5  # 所请求的采样周期

        # 调用 cont2discrete 函数，将连续时间系统转换为离散时间系统，使用“zoh”方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='zoh')

        # 检查计算得到的 ad 是否与预期值相等
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cc, cd)
        assert_array_almost_equal(dc, dd)
        assert_almost_equal(dt_requested, dt)

    def test_foh(self):
        ac = np.eye(2)  # 创建一个2x2的单位矩阵 ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组 bc，元素均为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组 cc
        dc = np.array([[0.0], [0.0], [-0.33]])  # 创建一个3x1的数组 dc

        # 使用 Matlab 验证得到的真实值
        ad_truth = 1.648721270700128 * np.eye(2)
        bd_truth = np.full((2, 1), 0.420839287058789)
        cd_truth = cc
        dd_truth = np.array([[0.260262223725224],
                             [0.297442541400256],
                             [-0.144098411624840]])
        dt_requested = 0.5

        # 调用 cont2discrete 函数，将连续时间系统转换为离散时间系统，使用“foh”方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='foh')

        # 检查计算得到的 ad 是否与预期值相等
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_impulse(self):
        ac = np.eye(2)  # 创建一个2x2的单位矩阵 ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组 bc，元素均为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组 cc
        dc = np.array([[0.0], [0.0], [0.0]])  # 创建一个3x1的数组 dc

        # 使用 Matlab 验证得到的真实值
        ad_truth = 1.648721270700128 * np.eye(2)
        bd_truth = np.full((2, 1), 0.412180317675032)
        cd_truth = cc
        dd_truth = np.array([[0.4375], [0.5], [0.3125]])
        dt_requested = 0.5

        # 调用 cont2discrete 函数，将连续时间系统转换为离散时间系统，使用“impulse”方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='impulse')

        # 检查计算得到的 ad 是否与预期值相等
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)
    def test_gbt(self):
        # 定义测试用例的初始条件
        ac = np.eye(2)  # 创建一个2x2的单位矩阵ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组bc，每个元素为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组cc
        dc = np.array([[0.0], [0.0], [-0.33]])  # 创建一个3x1的数组dc

        dt_requested = 0.5  # 定义所请求的采样周期dt
        alpha = 1.0 / 3.0  # 定义用于计算的alpha值

        ad_truth = 1.6 * np.eye(2)  # 创建一个2x2的数组ad_truth，每个元素为1.6
        bd_truth = np.full((2, 1), 0.3)  # 创建一个2x1的数组bd_truth，每个元素为0.3
        cd_truth = np.array([[0.9, 1.2],
                             [1.2, 1.2],
                             [1.2, 0.3]])  # 创建一个3x2的数组cd_truth
        dd_truth = np.array([[0.175],
                             [0.2],
                             [-0.205]])  # 创建一个3x1的数组dd_truth

        # 调用c2d函数将连续时间系统转换为离散时间系统，使用gbt方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='gbt', alpha=alpha)

        # 断言转换后的离散时间系统与预期值相近
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)

    def test_euler(self):
        # 定义测试用例的初始条件
        ac = np.eye(2)  # 创建一个2x2的单位矩阵ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组bc，每个元素为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组cc
        dc = np.array([[0.0], [0.0], [-0.33]])  # 创建一个3x1的数组dc

        dt_requested = 0.5  # 定义所请求的采样周期dt

        ad_truth = 1.5 * np.eye(2)  # 创建一个2x2的数组ad_truth，每个元素为1.5
        bd_truth = np.full((2, 1), 0.25)  # 创建一个2x1的数组bd_truth，每个元素为0.25
        cd_truth = np.array([[0.75, 1.0],
                             [1.0, 1.0],
                             [1.0, 0.25]])  # 创建一个3x2的数组cd_truth，与cc相同
        dd_truth = dc  # 将dd_truth设置为dc

        # 调用c2d函数将连续时间系统转换为离散时间系统，使用euler方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='euler')

        # 断言转换后的离散时间系统与预期值相近
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_backward_diff(self):
        # 定义测试用例的初始条件
        ac = np.eye(2)  # 创建一个2x2的单位矩阵ac
        bc = np.full((2, 1), 0.5)  # 创建一个2x1的数组bc，每个元素为0.5
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])  # 创建一个3x2的数组cc
        dc = np.array([[0.0], [0.0], [-0.33]])  # 创建一个3x1的数组dc

        dt_requested = 0.5  # 定义所请求的采样周期dt

        ad_truth = 2.0 * np.eye(2)  # 创建一个2x2的数组ad_truth，每个元素为2.0
        bd_truth = np.full((2, 1), 0.5)  # 创建一个2x1的数组bd_truth，每个元素为0.5
        cd_truth = np.array([[1.5, 2.0],
                             [2.0, 2.0],
                             [2.0, 0.5]])  # 创建一个3x2的数组cd_truth
        dd_truth = np.array([[0.875],
                             [1.0],
                             [0.295]])  # 创建一个3x1的数组dd_truth

        # 调用c2d函数将连续时间系统转换为离散时间系统，使用backward_diff方法
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='backward_diff')

        # 断言转换后的离散时间系统与预期值相近
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
    # 定义测试方法，用于测试 bilinear 方法的离散化效果
    def test_bilinear(self):
        # 创建连续系统的状态空间矩阵 ac
        ac = np.eye(2)
        # 创建连续系统的输入到状态矩阵 bc
        bc = np.full((2, 1), 0.5)
        # 创建连续系统的输出到状态矩阵 cc
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        # 创建连续系统的状态到输出矩阵 dc
        dc = np.array([[0.0], [0.0], [-0.33]])

        # 请求的离散化时间步长
        dt_requested = 0.5

        # 期望的离散化状态空间矩阵 ad_truth
        ad_truth = (5.0 / 3.0) * np.eye(2)
        # 期望的离散化输入到状态矩阵 bd_truth
        bd_truth = np.full((2, 1), 1.0 / 3.0)
        # 期望的离散化状态到输出矩阵 cd_truth
        cd_truth = np.array([[1.0, 4.0 / 3.0],
                             [4.0 / 3.0, 4.0 / 3.0],
                             [4.0 / 3.0, 1.0 / 3.0]])
        # 期望的离散化输出到输出矩阵 dd_truth
        dd_truth = np.array([[0.291666666666667],
                             [1.0 / 3.0],
                             [-0.121666666666667]])

        # 调用 c2d 函数执行 bilinear 方法离散化
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='bilinear')

        # 断言结果与期望值的近似性
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

        # 同样的连续系统，但改变采样率

        # 更新期望的离散化状态空间矩阵 ad_truth
        ad_truth = 1.4 * np.eye(2)
        # 更新期望的离散化输入到状态矩阵 bd_truth
        bd_truth = np.full((2, 1), 0.2)
        # 更新期望的离散化状态到输出矩阵 cd_truth
        cd_truth = np.array([[0.9, 1.2], [1.2, 1.2], [1.2, 0.3]])
        # 更新期望的离散化输出到输出矩阵 dd_truth
        dd_truth = np.array([[0.175], [0.2], [-0.205]])

        # 更新请求的离散化时间步长
        dt_requested = 1.0 / 3.0

        # 再次调用 c2d 函数执行 bilinear 方法离散化
        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='bilinear')

        # 断言结果与更新后的期望值的近似性
        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    # 定义测试方法，用于测试传递函数的离散化效果
    def test_transferfunction(self):
        # 创建连续系统的传递函数的分子系数 numc
        numc = np.array([0.25, 0.25, 0.5])
        # 创建连续系统的传递函数的分母系数 denc
        denc = np.array([0.75, 0.75, 1.0])

        # 期望的离散化传递函数的分子系数 numd
        numd = np.array([[1.0 / 3.0, -0.427419169438754, 0.221654141101125]])
        # 期望的离散化传递函数的分母系数 dend
        dend = np.array([1.0, -1.351394049721225, 0.606530659712634])

        # 请求的离散化时间步长
        dt_requested = 0.5

        # 调用 c2d 函数执行 zoh 方法离散化传递函数
        num, den, dt = c2d((numc, denc), dt_requested, method='zoh')

        # 断言结果与期望值的近似性
        assert_array_almost_equal(numd, num)
        assert_array_almost_equal(dend, den)
        assert_almost_equal(dt_requested, dt)

    # 定义测试方法，用于测试零极点增益的离散化效果
    def test_zerospolesgain(self):
        # 创建连续系统的零点 zeros_c、极点 poles_c 和增益 k_c
        zeros_c = np.array([0.5, -0.5])
        poles_c = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
        k_c = 1.0

        # 期望的离散化系统的零点 zeros_d、极点 poles_d 和增益 k_d
        zeros_d = [1.23371727305860, 0.735356894461267]
        poles_d = [0.938148335039729 + 0.346233593780536j,
                   0.938148335039729 - 0.346233593780536j]
        k_d = 1.0

        # 请求的离散化时间步长
        dt_requested = 0.5

        # 调用 c2d 函数执行 zoh 方法离散化零极点增益
        zeros, poles, k, dt = c2d((zeros_c, poles_c, k_c), dt_requested,
                                  method='zoh')

        # 断言结果与期望值的近似性
        assert_array_almost_equal(zeros_d, zeros)
        assert_array_almost_equal(polls_d, poles)
        assert_almost_equal(k_d, k)
        assert_almost_equal(dt_requested, dt)
    def test_gbt_with_sio_tf_and_zpk(self):
        """Test method='gbt' with alpha=0.25 for tf and zpk cases."""
        # State space coefficients for the continuous SIO system.
        A = -1.0
        B = 1.0
        C = 1.0
        D = 0.5

        # The continuous transfer function coefficients.
        cnum, cden = ss2tf(A, B, C, D)
        # Convert state space representation to transfer function (tf) representation

        # Continuous zpk representation
        cz, cp, ck = ss2zpk(A, B, C, D)
        # Convert state space representation to zero-pole-gain (zpk) representation

        h = 1.0
        alpha = 0.25

        # Explicit formulas, in the scalar case.
        Ad = (1 + (1 - alpha) * h * A) / (1 - alpha * h * A)
        Bd = h * B / (1 - alpha * h * A)
        Cd = C / (1 - alpha * h * A)
        Dd = D + alpha * C * Bd
        # Discretization formulas using the given alpha and sampling period h

        # Convert the explicit solution to tf
        dnum, dden = ss2tf(Ad, Bd, Cd, Dd)
        # Convert discretized state space solution to transfer function

        # Compute the discrete tf using cont2discrete.
        c2dnum, c2dden, dt = c2d((cnum, cden), h, method='gbt', alpha=alpha)
        # Compute discrete transfer function using generalized bilinear transformation (gbt)

        assert_allclose(dnum, c2dnum)
        assert_allclose(dden, c2dden)
        # Assert that the coefficients of the computed discrete tf match the expected values

        # Convert explicit solution to zpk.
        dz, dp, dk = ss2zpk(Ad, Bd, Cd, Dd)
        # Convert discretized state space solution to zpk representation

        # Compute the discrete zpk using cont2discrete.
        c2dz, c2dp, c2dk, dt = c2d((cz, cp, ck), h, method='gbt', alpha=alpha)
        # Compute discrete zpk using generalized bilinear transformation (gbt)

        assert_allclose(dz, c2dz)
        assert_allclose(dp, c2dp)
        assert_allclose(dk, c2dk)
        # Assert that the coefficients of the computed discrete zpk match the expected values

    def test_discrete_approx(self):
        """
        Test that the solution to the discrete approximation of a continuous
        system actually approximates the solution to the continuous system.
        This is an indirect test of the correctness of the implementation
        of cont2discrete.
        """

        def u(t):
            return np.sin(2.5 * t)

        a = np.array([[-0.01]])
        b = np.array([[1.0]])
        c = np.array([[1.0]])
        d = np.array([[0.2]])
        x0 = 1.0

        t = np.linspace(0, 10.0, 101)
        dt = t[1] - t[0]
        u1 = u(t)

        # Use lsim to compute the solution to the continuous system.
        t, yout, xout = lsim((a, b, c, d), T=t, U=u1, X0=x0)
        # Simulate continuous system response using lsim

        # Convert the continuous system to a discrete approximation.
        dsys = c2d((a, b, c, d), dt, method='bilinear')
        # Convert continuous system to discrete using bilinear transformation

        # Use dlsim with the pairwise averaged input to compute the output
        # of the discrete system.
        u2 = 0.5 * (u1[:-1] + u1[1:])
        t2 = t[:-1]
        td2, yd2, xd2 = dlsim(dsys, u=u2.reshape(-1, 1), t=t2, x0=x0)
        # Simulate discrete system response using dlsim with averaged input

        # ymid is the average of consecutive terms of the "exact" output
        # computed by lsim2.  This is what the discrete approximation
        # actually approximates.
        ymid = 0.5 * (yout[:-1] + yout[1:])
        # Compute the midpoint average of the continuous system output

        assert_allclose(yd2.ravel(), ymid, rtol=1e-4)
        # Assert that the simulated discrete system output approximates the midpoint average of continuous output

    def test_simo_tf(self):
        # See gh-5753
        tf = ([[1, 0], [1, 1]], [1, 1])
        num, den, dt = c2d(tf, 0.01)
        # Convert given transfer function to discrete using specified sampling period

        assert_equal(dt, 0.01)  # sanity check
        assert_allclose(den, [1, -0.990404983], rtol=1e-3)
        assert_allclose(num, [[1, -1], [1, -0.99004983]], rtol=1e-3)
        # Assert that the computed discrete transfer function coefficients match expected values
    # 定义一个测试方法，用于测试多输出的情况
    def test_multioutput(self):
        ts = 0.01  # 时间步长，单位为秒

        # 定义一个连续时间系统的传递函数 tf = ([[1, -3], [1, 5]], [1, 1])
        # 将其离散化，得到离散时间系统的分子、分母以及离散化时间步长
        num, den, dt = c2d(tf, ts)

        # 将传递函数 tf 的第一个分子元素独立出来，重新构造传递函数 tf1
        # 然后对 tf1 进行离散化，得到其分子、分母以及离散化时间步长
        tf1 = (tf[0][0], tf[1])
        num1, den1, dt1 = c2d(tf1, ts)

        # 将传递函数 tf 的第二个分子元素独立出来，重新构造传递函数 tf2
        # 然后对 tf2 进行离散化，得到其分子、分母以及离散化时间步长
        tf2 = (tf[0][1], tf[1])
        num2, den2, dt2 = c2d(tf2, ts)

        # 进行一些基本的检查

        # 检查离散化的时间步长 dt、dt1、dt2 是否相等
        assert_equal(dt, dt1)
        assert_equal(dt, dt2)

        # 检查通过组合 num1 和 num2 得到的分子数组是否与 num 近似相等
        assert_allclose(num, np.vstack((num1, num2)), rtol=1e-13)

        # 对于单输入系统，分母数组应该不会像分子数组一样是多维的
        # 因此，检查 den、den1、den2 是否近似相等
        assert_allclose(den, den1, rtol=1e-13)
        assert_allclose(den, den2, rtol=1e-13)
class TestC2dLti:
    def test_c2d_ss(self):
        # StateSpace
        A = np.array([[-0.3, 0.1], [0.2, -0.7]])  # 系统矩阵 A
        B = np.array([[0], [1]])  # 输入矩阵 B
        C = np.array([[1, 0]])  # 输出矩阵 C
        D = 0  # 直接传递项 D

        A_res = np.array([[0.985136404135682, 0.004876671474795],  # 预期的离散化后的系统矩阵 A
                          [0.009753342949590, 0.965629718236502]])
        B_res = np.array([[0.000122937599964], [0.049135527547844]])  # 预期的离散化后的输入矩阵 B

        sys_ssc = lti(A, B, C, D)  # 创建连续时间系统
        sys_ssd = sys_ssc.to_discrete(0.05)  # 将系统离散化，采样时间为 0.05

        assert_allclose(sys_ssd.A, A_res)  # 断言离散化后的系统矩阵 A 是否与预期相近
        assert_allclose(sys_ssd.B, B_res)  # 断言离散化后的输入矩阵 B 是否与预期相近
        assert_allclose(sys_ssd.C, C)  # 断言输出矩阵 C 是否与预期相近
        assert_allclose(sys_ssd.D, D)  # 断言直接传递项 D 是否与预期相近

    def test_c2d_tf(self):
        sys = lti([0.5, 0.3], [1.0, 0.4])  # 创建传递函数系统
        sys = sys.to_discrete(0.005)  # 将系统离散化，采样时间为 0.005

        # Matlab results
        num_res = np.array([0.5, -0.485149004980066])  # 预期的离散化后的传递函数分子系数
        den_res = np.array([1.0, -0.980198673306755])  # 预期的离散化后的传递函数分母系数

        # Somehow a lot of numerical errors
        assert_allclose(sys.den, den_res, atol=0.02)  # 断言离散化后的传递函数分母系数是否与预期相近，允许数值误差为 0.02
        assert_allclose(sys.num, num_res, atol=0.02)  # 断言离散化后的传递函数分子系数是否与预期相近，允许数值误差为 0.02


class TestC2dInvariants:
    # Some test cases for checking the invariances.
    # Array of triplets: (system, sample time, number of samples)
    cases = [
        (tf2ss([1, 1], [1, 1.5, 1]), 0.25, 10),  # 测试样例：传递函数到状态空间的转换，采样时间为 0.25，样本数为 10
        (tf2ss([1, 2], [1, 1.5, 3, 1]), 0.5, 10),  # 测试样例：传递函数到状态空间的转换，采样时间为 0.5，样本数为 10
        (tf2ss(0.1, [1, 1, 2, 1]), 0.5, 10),  # 测试样例：传递函数到状态空间的转换，采样时间为 0.5，样本数为 10
    ]

    # Check that systems discretized with the impulse-invariant
    # method really hold the invariant
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_impulse_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont = impulse(sys, T=time)  # 连续时间系统的阶跃响应
        _, yout_disc = dimpulse(c2d(sys, sample_time, method='impulse'),
                                n=len(time))  # 离散时间系统的阶跃响应
        assert_allclose(sample_time * yout_cont.ravel(), yout_disc[0].ravel())  # 断言离散化后的系统响应与预期相近

    # Step invariant should hold for ZOH discretized systems
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_step_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont = step(sys, T=time)  # 连续时间系统的阶跃响应
        _, yout_disc = dstep(c2d(sys, sample_time, method='zoh'), n=len(time))  # ZOH 方法离散化的阶跃响应
        assert_allclose(yout_cont.ravel(), yout_disc[0].ravel())  # 断言离散化后的系统响应与预期相近

    # Linear invariant should hold for FOH discretized systems
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_linear_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont, _ = lsim(sys, T=time, U=time)  # 连续时间系统的线性响应
        _, yout_disc, _ = dlsim(c2d(sys, sample_time, method='foh'), u=time)  # FOH 方法离散化的线性响应
        assert_allclose(yout_cont.ravel(), yout_disc.ravel())  # 断言离散化后的系统响应与预期相近
```