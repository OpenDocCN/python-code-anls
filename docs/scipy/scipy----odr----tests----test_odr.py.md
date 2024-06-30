# `D:\src\scipysrc\scipy\scipy\odr\tests\test_odr.py`

```
# 导入必要的库：用于创建临时文件和目录的模块、操作系统功能、以及数学运算的 numpy 库和其内部的 pi 常数。
import tempfile
import shutil
import os

import numpy as np
from numpy import pi
# numpy.testing 用于进行数值测试，包含多个断言函数，如 assert_array_almost_equal 等。
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_warns,
                           assert_allclose)
# 使用 pytest 进行测试，并导入 raises 函数作为 assert_raises 的别名。
import pytest
from pytest import raises as assert_raises
# 导入 scipy.odr 中的各种对象和函数，用于正交距离回归分析。
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
                       multilinear, exponential, unilinear, quadratic,
                       polynomial)


class TestODR:
    # 对 'x' 的错误数据情况进行测试
    def test_bad_data(self):
        # 检查在数据值为 2 和误差为 1 时是否会引发 ValueError 异常
        assert_raises(ValueError, Data, 2, 1)
        assert_raises(ValueError, RealData, 2, 1)

    # 对 'x' 的空数据情况进行测试
    def empty_data_func(self, B, x):
        # 返回 B[0]*x + B[1] 的线性函数结果
        return B[0]*x + B[1]

    def test_empty_data(self):
        beta0 = [0.02, 0.0]
        # 创建线性模型对象
        linear = Model(self.empty_data_func)

        # 创建空的 Data 对象和 RealData 对象
        empty_dat = Data([], [])
        # 检查在使用空数据时是否会引发 OdrWarning 警告
        assert_warns(OdrWarning, ODR,
                     empty_dat, linear, beta0=beta0)

        empty_dat = RealData([], [])
        # 检查在使用空数据时是否会引发 OdrWarning 警告
        assert_warns(OdrWarning, ODR,
                     empty_dat, linear, beta0=beta0)

    # 显式的例子
    def explicit_fcn(self, B, x):
        # 计算显式函数的结果并返回
        ret = B[0] + B[1] * np.power(np.exp(B[2]*x) - 1.0, 2)
        return ret

    def explicit_fjd(self, B, x):
        # 计算显式函数关于参数 B 的导数并返回
        eBx = np.exp(B[2]*x)
        ret = B[1] * 2.0 * (eBx-1.0) * B[2] * eBx
        return ret

    def explicit_fjb(self, B, x):
        # 计算显式函数关于参数 B 的雅可比矩阵并返回
        eBx = np.exp(B[2]*x)
        res = np.vstack([np.ones(x.shape[-1]),
                         np.power(eBx-1.0, 2),
                         B[1]*2.0*(eBx-1.0)*eBx*x])
        return res
    # 定义一个测试函数，用于测试显式模型
    def test_explicit(self):
        # 创建显式模型对象，传入显式函数及其导数，以及元数据信息
        explicit_mod = Model(
            self.explicit_fcn,  # 显式函数
            fjacb=self.explicit_fjb,  # 显式函数的雅可比矩阵
            fjacd=self.explicit_fjd,  # 显式函数的二阶导数
            meta=dict(name='Sample Explicit Model',  # 模型的名称
                      ref='ODRPACK UG, pg. 39'),  # 模型的参考文献
        )
        # 创建显式模型的数据对象，包括 x 和 y 值
        explicit_dat = Data([0.,0.,5.,7.,7.5,10.,16.,26.,30.,34.,34.5,100.],
                        [1265.,1263.6,1258.,1254.,1253.,1249.8,1237.,1218.,1220.6,
                         1213.8,1215.5,1212.])
        # 创建 ODR 对象，传入数据、模型和初始参数
        explicit_odr = ODR(explicit_dat, explicit_mod, beta0=[1500.0, -50.0, -0.1],
                       ifixx=[0,0,1,1,1,1,1,1,1,1,1,0])
        # 设置 ODR 对象的作业，这里设置二阶导数的计算
        explicit_odr.set_job(deriv=2)
        # 设置 ODR 对象的打印选项，这里设置初始化、迭代和最终打印均为 0
        explicit_odr.set_iprint(init=0, iter=0, final=0)

        # 运行 ODR 拟合过程，得到结果
        out = explicit_odr.run()
        
        # 断言拟合参数 out.beta 与预期值的近似程度
        assert_array_almost_equal(
            out.beta,
            np.array([1.2646548050648876e+03, -5.4018409956678255e+01,
                -8.7849712165253724e-02]),
        )
        # 断言拟合参数的标准误差 out.sd_beta 与预期值的近似程度
        assert_array_almost_equal(
            out.sd_beta,
            np.array([1.0349270280543437, 1.583997785262061, 0.0063321988657267]),
        )
        # 断言拟合参数的协方差 out.cov_beta 与预期值的近似程度
        assert_array_almost_equal(
            out.cov_beta,
            np.array([[4.4949592379003039e-01, -3.7421976890364739e-01,
                 -8.0978217468468912e-04],
               [-3.7421976890364739e-01, 1.0529686462751804e+00,
                 -1.9453521827942002e-03],
               [-8.0978217468468912e-04, -1.9453521827942002e-03,
                  1.6827336938454476e-05]]),
        )

    # 隐式示例

    # 定义一个隐式函数，计算 B 和 x 的关系
    def implicit_fcn(self, B, x):
        return (B[2]*np.power(x[0]-B[0], 2) +
                2.0*B[3]*(x[0]-B[0])*(x[1]-B[1]) +
                B[4]*np.power(x[1]-B[1], 2) - 1.0)
    # 定义一个测试函数，用于隐式模型的测试
    def test_implicit(self):
        # 创建一个模型对象，指定隐式函数，设置隐式参数为1，并添加元数据
        implicit_mod = Model(
            self.implicit_fcn,
            implicit=1,
            meta=dict(name='Sample Implicit Model',
                      ref='ODRPACK UG, pg. 49'),
        )
        # 创建隐式模型的数据对象，包括两个变量的数据数组和一个标志参数
        implicit_dat = Data([
            [0.5,1.2,1.6,1.86,2.12,2.36,2.44,2.36,2.06,1.74,1.34,0.9,-0.28,
             -0.78,-1.36,-1.9,-2.5,-2.88,-3.18,-3.44],
            [-0.12,-0.6,-1.,-1.4,-2.54,-3.36,-4.,-4.75,-5.25,-5.64,-5.97,-6.32,
             -6.44,-6.44,-6.41,-6.25,-5.88,-5.5,-5.24,-4.86]],
            1,
        )
        # 创建隐式模型的最小二乘法对象，指定数据、模型和初值猜测向量
        implicit_odr = ODR(implicit_dat, implicit_mod,
            beta0=[-1.0, -3.0, 0.09, 0.02, 0.08])

        # 运行隐式模型的最小二乘法，返回结果对象
        out = implicit_odr.run()
        
        # 断言结果的 beta 参数数组近似于给定的数值数组
        assert_array_almost_equal(
            out.beta,
            np.array([-0.9993809167281279, -2.9310484652026476, 0.0875730502693354,
                0.0162299708984738, 0.0797537982976416]),
        )
        
        # 断言结果的 beta 标准差数组近似于给定的数值数组
        assert_array_almost_equal(
            out.sd_beta,
            np.array([0.1113840353364371, 0.1097673310686467, 0.0041060738314314,
                0.0027500347539902, 0.0034962501532468]),
        )
        
        # 断言结果的 beta 协方差矩阵近似于给定的数值数组
        assert_allclose(
            out.cov_beta,
            np.array([[2.1089274602333052e+00, -1.9437686411979040e+00,
                  7.0263550868344446e-02, -4.7175267373474862e-02,
                  5.2515575927380355e-02],
               [-1.9437686411979040e+00, 2.0481509222414456e+00,
                 -6.1600515853057307e-02, 4.6268827806232933e-02,
                 -5.8822307501391467e-02],
               [7.0263550868344446e-02, -6.1600515853057307e-02,
                  2.8659542561579308e-03, -1.4628662260014491e-03,
                  1.4528860663055824e-03],
               [-4.7175267373474862e-02, 4.6268827806232933e-02,
                 -1.4628662260014491e-03, 1.2855592885514335e-03,
                 -1.2692942951415293e-03],
               [5.2515575927380355e-02, -5.8822307501391467e-02,
                  1.4528860663055824e-03, -1.2692942951415293e-03,
                  2.0778813389755596e-03]]),
            rtol=1e-6, atol=2e-6,
        )

    # 定义一个多变量函数，接受参数 B 和变量 x 作为输入
    def multi_fcn(self, B, x):
        # 如果 x 中有任何一个值小于 0，则抛出 OdrStop 异常
        if (x < 0.0).any():
            raise OdrStop
        # 根据 B 的部分元素计算一系列中间变量
        theta = pi*B[3]/2.
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        omega = np.power(2.*pi*x*np.exp(-B[2]), B[3])
        phi = np.arctan2((omega*stheta), (1.0 + omega*ctheta))
        r = (B[0] - B[1]) * np.power(np.sqrt(np.power(1.0 + omega*ctheta, 2) +
             np.power(omega*stheta, 2)), -B[4])
        # 构建并返回一个由两行组成的矩阵，表示多变量函数的结果
        ret = np.vstack([B[1] + r*np.cos(B[4]*phi),
                         r*np.sin(B[4]*phi)])
        return ret

    # 定义一个 Pearson 函数，接受参数 B 和变量 x 作为输入
    # 函数的返回值是 B[0] + B[1]*x
    # 出处：K. Pearson, Philosophical Magazine, 2, 559 (1901)
    def pearson_fcn(self, B, x):
        return B[0] + B[1]*x
    # 定义一个测试函数 test_pearson，用于测试皮尔逊相关系数计算
    def test_pearson(self):
        # 构造输入数据点的 x 坐标数组
        p_x = np.array([0., .9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
        # 构造输入数据点的 y 坐标数组
        p_y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
        # 构造 x 坐标的标准误差数组
        p_sx = np.array([.03, .03, .04, .035, .07, .11, .13, .22, .74, 1.])
        # 构造 y 坐标的标准误差数组
        p_sy = np.array([1., .74, .5, .35, .22, .22, .12, .12, .1, .04])

        # 创建 RealData 对象，用于拟合数据，包括数据点坐标和标准误差
        p_dat = RealData(p_x, p_y, sx=p_sx, sy=p_sy)

        # 反转数据以测试结果的不变性
        pr_dat = RealData(p_y, p_x, sx=p_sy, sy=p_sx)

        # 创建一个 Model 对象，使用自定义的皮尔逊相关函数作为模型
        p_mod = Model(self.pearson_fcn, meta=dict(name='Uni-linear Fit'))

        # 使用 ODR 方法创建一个拟合实例，传入数据、模型和初始参数
        p_odr = ODR(p_dat, p_mod, beta0=[1., 1.])
        pr_odr = ODR(pr_dat, p_mod, beta0=[1., 1.])

        # 运行拟合并获取结果
        out = p_odr.run()
        # 断言拟合参数 beta 的数值近似等于给定值
        assert_array_almost_equal(
            out.beta,
            np.array([5.4767400299231674, -0.4796082367610305]),
        )
        # 断言拟合参数 beta 的标准偏差近似等于给定值
        assert_array_almost_equal(
            out.sd_beta,
            np.array([0.3590121690702467, 0.0706291186037444]),
        )
        # 断言拟合参数 beta 的协方差矩阵近似等于给定值
        assert_array_almost_equal(
            out.cov_beta,
            np.array([[0.0854275622946333, -0.0161807025443155],
               [-0.0161807025443155, 0.003306337993922]]),
        )

        # 对反转数据进行相同的拟合和断言操作
        rout = pr_odr.run()
        assert_array_almost_equal(
            rout.beta,
            np.array([11.4192022410781231, -2.0850374506165474]),
        )
        assert_array_almost_equal(
            rout.sd_beta,
            np.array([0.9820231665657161, 0.3070515616198911]),
        )
        assert_array_almost_equal(
            rout.cov_beta,
            np.array([[0.6391799462548782, -0.1955657291119177],
               [-0.1955657291119177, 0.0624888159223392]]),
        )

    # 定义一个 Lorentz 峰函数，用于计算 Lorentz 峰的数学表达式
    # 数据来源于我参与过的本科物理实验室之一
    def lorentz(self, beta, x):
        # 计算 Lorentz 峰函数的值，参数包括 beta 和 x
        return (beta[0] * beta[1] * beta[2] / np.sqrt(np.power(x * x -
            beta[2] * beta[2], 2.0) + np.power(beta[1] * x, 2.0)))
    # 定义一个测试方法 test_lorentz，用于测试 Lorentz 模型拟合
    def test_lorentz(self):
        # 创建一个长度为 18 的 NumPy 数组，元素均为 0.29
        l_sy = np.array([.29]*18)
        # 创建一个包含 18 个浮点数的 NumPy 数组，表示 Lorentz 模型的 sx 参数
        l_sx = np.array([.000972971,.000948268,.000707632,.000706679,
            .000706074, .000703918,.000698955,.000456856,
            .000455207,.000662717,.000654619,.000652694,
            .000000859202,.00106589,.00106378,.00125483, .00140818,.00241839])

        # 创建 RealData 对象 l_dat，用于存储实际数据
        l_dat = RealData(
            [3.9094, 3.85945, 3.84976, 3.84716, 3.84551, 3.83964, 3.82608,
             3.78847, 3.78163, 3.72558, 3.70274, 3.6973, 3.67373, 3.65982,
             3.6562, 3.62498, 3.55525, 3.41886],
            [652, 910.5, 984, 1000, 1007.5, 1053, 1160.5, 1409.5, 1430, 1122,
             957.5, 920, 777.5, 709.5, 698, 578.5, 418.5, 275.5],
            sx=l_sx,
            sy=l_sy,
        )
        
        # 创建 Lorentz 模型对象 l_mod
        l_mod = Model(self.lorentz, meta=dict(name='Lorentz Peak'))
        
        # 使用 ODR 方法创建一个拟合任务对象 l_odr
        l_odr = ODR(l_dat, l_mod, beta0=(1000., .1, 3.8))

        # 运行拟合任务，并将结果存储在 out 中
        out = l_odr.run()
        
        # 断言拟合结果的 beta 值与预期值的近似相等
        assert_array_almost_equal(
            out.beta,
            np.array([1.4306780846149925e+03, 1.3390509034538309e-01,
                 3.7798193600109009e+00]),
        )
        
        # 断言拟合结果的 beta 标准差 sd_beta 与预期值的近似相等
        assert_array_almost_equal(
            out.sd_beta,
            np.array([7.3621186811330963e-01, 3.5068899941471650e-04,
                 2.4451209281408992e-04]),
        )
        
        # 断言拟合结果的 beta 协方差 cov_beta 与预期值的近似相等
        assert_array_almost_equal(
            out.cov_beta,
            np.array([[2.4714409064597873e-01, -6.9067261911110836e-05,
                 -3.1236953270424990e-05],
               [-6.9067261911110836e-05, 5.6077531517333009e-08,
                  3.6133261832722601e-08],
               [-3.1236953270424990e-05, 3.6133261832722601e-08,
                  2.7261220025171730e-08]]),
        )

    # 定义一个测试方法 test_ticket_1253，用于测试线性模型的拟合
    def test_ticket_1253(self):
        # 定义一个线性函数 linear，接受参数 c 和 x
        def linear(c, x):
            return c[0]*x+c[1]

        # 设置线性函数的参数 c
        c = [2.0, 3.0]
        # 生成一个 0 到 10 的等间距数组作为 x
        x = np.linspace(0, 10)
        # 根据线性函数计算 y
        y = linear(c, x)

        # 创建线性模型对象 model
        model = Model(linear)
        # 创建 Data 对象 data，存储 x 和 y 数据
        data = Data(x, y, wd=1.0, we=1.0)
        # 创建 ODR 拟合任务对象 job，使用 beta0 初值进行拟合
        job = ODR(data, model, beta0=[1.0, 1.0])
        # 运行拟合任务，并将结果存储在 result 中
        result = job.run()
        # 断言拟合结果的 info 属性值为 2
        assert_equal(result.info, 2)

    # 验证 GitHub 问题修复 #9140

    # 定义一个测试方法 test_ifixx，用于测试带有 ifixx 参数的 ODR 拟合任务
    def test_ifixx(self):
        # 定义两个输入数据数组 x1 和 x2
        x1 = [-2.01, -0.99, -0.001, 1.02, 1.98]
        x2 = [3.98, 1.01, 0.001, 0.998, 4.01]
        # 创建一个 fix 数组，用于指定哪些数据需要固定
        fix = np.vstack((np.zeros_like(x1, dtype=int), np.ones_like(x2, dtype=int)))
        # 创建 Data 对象 data，将 x1 和 x2 组成的数组作为输入
        data = Data(np.vstack((x1, x2)), y=1, fix=fix)
        # 创建隐式模型 model，使用 lambda 表达式定义
        model = Model(lambda beta, x: x[1, :] - beta[0] * x[0, :]**2., implicit=True)

        # 创建 ODR 拟合任务对象 odr1，使用 beta0 初值进行拟合
        odr1 = ODR(data, model, beta0=np.array([1.]))
        # 运行拟合任务，并将结果存储在 sol1 中
        sol1 = odr1.run()
        # 创建带有 ifixx 参数的 ODR 拟合任务对象 odr2，使用 fix 数组指定固定数据
        odr2 = ODR(data, model, beta0=np.array([1.]), ifixx=fix)
        # 运行带有 ifixx 参数的拟合任务，并将结果存储在 sol2 中
        sol2 = odr2.run()
        # 断言两次拟合的 beta 值近似相等
        assert_equal(sol1.beta, sol2.beta)

    # 验证 GitHub 问题修复 #11800 in #11802
    def test_ticket_11800(self):
        # 定义真实的 beta 参数数组
        beta_true = np.array([1.0, 2.3, 1.1, -1.0, 1.3, 0.5])
        # 定义测量次数
        nr_measurements = 10

        # 定义 x 的标准偏差
        std_dev_x = 0.01
        # 定义 x 的测量误差数组
        x_error = np.array([[0.00063445, 0.00515731, 0.00162719, 0.01022866,
            -0.01624845, 0.00482652, 0.00275988, -0.00714734, -0.00929201, -0.00687301],
            [-0.00831623, -0.00821211, -0.00203459, 0.00938266, -0.00701829,
            0.0032169, 0.00259194, -0.00581017, -0.0030283, 0.01014164]])

        # 定义 y 的标准偏差
        std_dev_y = 0.05
        # 定义 y 的测量误差数组
        y_error = np.array([[0.05275304, 0.04519563, -0.07524086, 0.03575642,
            0.04745194, 0.03806645, 0.07061601, -0.00753604, -0.02592543, -0.02394929],
            [0.03632366, 0.06642266, 0.08373122, 0.03988822, -0.0092536,
            -0.03750469, -0.03198903, 0.01642066, 0.01293648, -0.05627085]])

        # 定义解决方案的 beta 参数数组
        beta_solution = np.array([
            2.62920235756665876536e+00, -1.26608484996299608838e+02,
            1.29703572775403074502e+02, -1.88560985401185465804e+00,
            7.83834160771274923718e+01, -7.64124076838087091801e+01])

        # 定义模型的函数和雅可比矩阵
        def func(beta, x):
            # 计算 y0 和 y1
            y0 = beta[0] + beta[1] * x[0, :] + beta[2] * x[1, :]
            y1 = beta[3] + beta[4] * x[0, :] + beta[5] * x[1, :]

            return np.vstack((y0, y1))

        def df_dbeta_odr(beta, x):
            # 计算 beta 对应的雅可比矩阵
            nr_meas = np.shape(x)[1]
            zeros = np.zeros(nr_meas)
            ones = np.ones(nr_meas)

            dy0 = np.array([ones, x[0, :], x[1, :], zeros, zeros, zeros])
            dy1 = np.array([zeros, zeros, zeros, ones, x[0, :], x[1, :]])

            return np.stack((dy0, dy1))

        def df_dx_odr(beta, x):
            # 计算 x 对应的雅可比矩阵
            nr_meas = np.shape(x)[1]
            ones = np.ones(nr_meas)

            dy0 = np.array([beta[1] * ones, beta[2] * ones])
            dy1 = np.array([beta[4] * ones, beta[5] * ones])
            return np.stack((dy0, dy1))

        # 使用带误差的独立和依赖变量进行测量
        x0_true = np.linspace(1, 10, nr_measurements)
        x1_true = np.linspace(1, 10, nr_measurements)
        x_true = np.array([x0_true, x1_true])

        y_true = func(beta_true, x_true)

        x_meas = x_true + x_error
        y_meas = y_true + y_error

        # 估计模型的参数
        model_f = Model(func, fjacb=df_dbeta_odr, fjacd=df_dx_odr)

        data = RealData(x_meas, y_meas, sx=std_dev_x, sy=std_dev_y)

        odr_obj = ODR(data, model_f, beta0=0.9 * beta_true, maxit=100)
        #odr_obj.set_iprint(init=2, iter=0, iter_step=1, final=1)
        # 设置 ODR 对象的作业模式为 deriv=3
        odr_obj.set_job(deriv=3)

        # 运行 ODR 拟合
        odr_out = odr_obj.run()

        # 检查结果
        assert_equal(odr_out.info, 1)
        assert_array_almost_equal(odr_out.beta, beta_solution)
    # 定义一个测试多元线性模型的方法
    def test_multilinear_model(self):
        # 生成一个从0到5的等间距数组作为自变量
        x = np.linspace(0.0, 5.0)
        # 计算对应的因变量，这里使用线性关系 y = 10.0 + 5.0 * x
        y = 10.0 + 5.0 * x
        # 创建数据对象，包括自变量 x 和因变量 y
        data = Data(x, y)
        # 使用 multilinear 模型创建 ODR 对象
        odr_obj = ODR(data, multilinear)
        # 运行拟合过程
        output = odr_obj.run()
        # 断言拟合结果的系数是否接近于预期值 [10.0, 5.0]
        assert_array_almost_equal(output.beta, [10.0, 5.0])

    # 定义一个测试指数模型的方法
    def test_exponential_model(self):
        # 生成一个从0到5的等间距数组作为自变量
        x = np.linspace(0.0, 5.0)
        # 计算对应的因变量，这里使用指数关系 y = -10.0 + exp(0.5*x)
        y = -10.0 + np.exp(0.5*x)
        # 创建数据对象，包括自变量 x 和因变量 y
        data = Data(x, y)
        # 使用 exponential 模型创建 ODR 对象
        odr_obj = ODR(data, exponential)
        # 运行拟合过程
        output = odr_obj.run()
        # 断言拟合结果的系数是否接近于预期值 [-10.0, 0.5]
        assert_array_almost_equal(output.beta, [-10.0, 0.5])

    # 定义一个测试多项式模型的方法
    def test_polynomial_model(self):
        # 生成一个从0到5的等间距数组作为自变量
        x = np.linspace(0.0, 5.0)
        # 计算对应的因变量，这里使用三次多项式 y = 1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3
        y = 1.0 + 2.0 * x + 3.0 * x ** 2 + 4.0 * x ** 3
        # 创建三次多项式模型
        poly_model = polynomial(3)
        # 创建数据对象，包括自变量 x 和因变量 y
        data = Data(x, y)
        # 使用指定的多项式模型创建 ODR 对象
        odr_obj = ODR(data, poly_model)
        # 运行拟合过程
        output = odr_obj.run()
        # 断言拟合结果的系数是否接近于预期值 [1.0, 2.0, 3.0, 4.0]
        assert_array_almost_equal(output.beta, [1.0, 2.0, 3.0, 4.0])

    # 定义一个测试一元线性模型的方法
    def test_unilinear_model(self):
        # 生成一个从0到5的等间距数组作为自变量
        x = np.linspace(0.0, 5.0)
        # 计算对应的因变量，这里使用一元线性关系 y = 1.0 * x + 2.0
        y = 1.0 * x + 2.0
        # 创建数据对象，包括自变量 x 和因变量 y
        data = Data(x, y)
        # 使用 unilinear 模型创建 ODR 对象
        odr_obj = ODR(data, unilinear)
        # 运行拟合过程
        output = odr_obj.run()
        # 断言拟合结果的系数是否接近于预期值 [1.0, 2.0]
        assert_array_almost_equal(output.beta, [1.0, 2.0])

    # 定义一个测试二次模型的方法
    def test_quadratic_model(self):
        # 生成一个从0到5的等间距数组作为自变量
        x = np.linspace(0.0, 5.0)
        # 计算对应的因变量，这里使用二次模型 y = 1.0 * x^2 + 2.0 * x + 3.0
        y = 1.0 * x ** 2 + 2.0 * x + 3.0
        # 创建数据对象，包括自变量 x 和因变量 y
        data = Data(x, y)
        # 使用 quadratic 模型创建 ODR 对象
        odr_obj = ODR(data, quadratic)
        # 运行拟合过程
        output = odr_obj.run()
        # 断言拟合结果的系数是否接近于预期值 [1.0, 2.0, 3.0]
        assert_array_almost_equal(output.beta, [1.0, 2.0, 3.0])

    # 定义一个测试工作指数的方法
    def test_work_ind(self):

        # 定义一个线性函数，用于拟合
        def func(par, x):
            b0, b1 = par
            return b0 + b1 * x

        # 生成一些数据
        n_data = 4
        x = np.arange(n_data)
        # 根据 x 的奇偶性生成对应的 y 数据
        y = np.where(x % 2, x + 0.1, x - 0.1)
        # 设置 x 和 y 的误差
        x_err = np.full(n_data, 0.1)
        y_err = np.full(n_data, 0.1)

        # 创建线性模型
        linear_model = Model(func)
        # 创建实际数据对象，包括自变量 x, 因变量 y, 以及它们的误差
        real_data = RealData(x, y, sx=x_err, sy=y_err)
        # 使用线性模型创建 ODR 对象，指定初始参数 beta0=[0.4, 0.4]
        odr_obj = ODR(real_data, linear_model, beta0=[0.4, 0.4])
        # 设置拟合类型为 0
        odr_obj.set_job(fit_type=0)
        # 运行拟合过程
        out = odr_obj.run()

        # 获取工作指数结果中的标准差
        sd_ind = out.work_ind['sd']
        # 断言工作数组中的一部分是否与标准差数组 sd_beta 接近
        assert_array_almost_equal(out.sd_beta,
                                  out.work[sd_ind:sd_ind + len(out.sd_beta)])

    # 使用 pytest.mark.skipif 标记该测试，当条件为真时跳过执行，并附加原因说明
    @pytest.mark.skipif(True, reason="Fortran I/O prone to crashing so better "
                                     "not to run this test, see gh-13127")
    def test_output_file_overwrite(self):
        """
        Verify fix for gh-1892
        """
        # 定义一个简单的线性函数，计算 b[0] + b[1] * x
        def func(b, x):
            return b[0] + b[1] * x

        # 创建一个 Model 对象，使用 func 函数作为模型
        p = Model(func)
        # 创建一个包含数据的 Data 对象，x 为 [0, 1, ..., 9]，y 为 [0, 12, ..., 108]
        data = Data(np.arange(10), 12 * np.arange(10))
        # 创建临时目录用于存储临时文件
        tmp_dir = tempfile.mkdtemp()
        # 设置错误文件和报告文件的路径
        error_file_path = os.path.join(tmp_dir, "error.dat")
        report_file_path = os.path.join(tmp_dir, "report.dat")
        try:
            # 运行 ODR 对象，将结果保存到错误文件和报告文件
            ODR(data, p, beta0=[0.1, 13], errfile=error_file_path,
                rptfile=report_file_path).run()
            # 使用 overwrite=True 参数再次运行 ODR 对象，覆盖之前的输出文件
            ODR(data, p, beta0=[0.1, 13], errfile=error_file_path,
                rptfile=report_file_path, overwrite=True).run()
        finally:
            # 清理临时目录，删除生成的临时文件
            shutil.rmtree(tmp_dir)

    def test_odr_model_default_meta(self):
        # 定义一个简单的线性函数，计算 b[0] + b[1] * x
        def func(b, x):
            return b[0] + b[1] * x

        # 创建一个 Model 对象，使用 func 函数作为模型
        p = Model(func)
        # 设置模型的元数据
        p.set_meta(name='Sample Model Meta', ref='ODRPACK')
        # 断言模型的元数据是否正确设置
        assert_equal(p.meta, {'name': 'Sample Model Meta', 'ref': 'ODRPACK'})

    def test_work_array_del_init(self):
        """
        Verify fix for gh-18739 where del_init=1 fails.
        """
        # 定义一个简单的线性函数，计算 b[0] + b[1] * x
        def func(b, x):
            return b[0] + b[1] * x

        # 生成一些数据
        n_data = 4
        x = np.arange(n_data)
        y = np.where(x % 2, x + 0.1, x - 0.1)
        x_err = np.full(n_data, 0.1)
        y_err = np.full(n_data, 0.1)

        # 创建一个线性模型对象
        linear_model = Model(func)
        # 使用不同的 RealData 对象进行测试
        rd0 = RealData(x, y, sx=x_err, sy=y_err)
        rd1 = RealData(x, y, sx=x_err, sy=0.1)
        rd2 = RealData(x, y, sx=x_err, sy=[0.1])
        rd3 = RealData(x, y, sx=x_err, sy=np.full((1, n_data), 0.1))
        rd4 = RealData(x, y, sx=x_err, covy=[[0.01]])
        rd5 = RealData(x, y, sx=x_err, covy=np.full((1, 1, n_data), 0.01))
        for rd in [rd0, rd1, rd2, rd3, rd4, rd5]:
            # 创建 ODR 对象并设置特定参数，运行模型拟合
            odr_obj = ODR(rd, linear_model, beta0=[0.4, 0.4],
                          delta0=np.full(n_data, -0.1))
            odr_obj.set_job(fit_type=0, del_init=1)
            # 确保运行不会引发异常
            odr_obj.run()
```