# `.\pytorch\test\nn\test_init.py`

```
# 导入必要的库和模块
# Owner(s): ["module: nn"]
import math  # 导入数学函数库
import random  # 导入随机数生成模块
import string  # 导入字符串处理模块
import unittest  # 导入单元测试模块
from functools import reduce  # 导入函数工具模块中的reduce函数
from operator import mul  # 导入操作符模块中的mul函数

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块
import torch.nn.init as init  # 导入PyTorch初始化模块

from torch.testing._internal.common_utils import (  # 从内部测试工具中导入函数和类
    run_tests,
    skipIfNoLapack,
    skipIfTorchDynamo,
    slowTest,
    TEST_SCIPY,
    TestCase,
)

if TEST_SCIPY:
    from scipy import stats  # 如果设置了TEST_SCIPY标志，导入scipy中的统计模块

class TestNNInit(TestCase):  # 定义测试类TestNNInit，继承自unittest中的TestCase类
    def setUp(self):  # 设置测试前的准备工作
        super().setUp()  # 调用父类的setUp方法
        random.seed(123)  # 设定随机数种子为123，保证测试的可重复性

    def _is_normal(self, tensor, mean, std):  # 定义检查张量是否符合正态分布的方法
        samples = tensor.view(-1).tolist()  # 将张量展平并转换为列表
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]  # 使用Kolmogorov-Smirnov检验检查样本是否符合正态分布
        return p_value > 0.0001  # 返回是否通过检验的布尔值

    def _is_trunc_normal(self, tensor, mean, std, a, b):  # 定义检查张量是否符合截断正态分布的方法
        # scipy的截断正态分布适用于从N(0, 1)中抽取的数据，因此需要将数据转换后使用scipy进行检验
        z_samples = (tensor.view(-1) - mean) / std  # 标准化张量数据
        z_samples = z_samples.tolist()  # 转换为列表
        a0 = (a - mean) / std  # 标准化截断范围下界
        b0 = (b - mean) / std  # 标准化截断范围上界
        p_value = stats.kstest(z_samples, "truncnorm", args=(a0, b0))[1]  # 使用Kolmogorov-Smirnov检验检查样本是否符合截断正态分布
        return p_value > 0.0001  # 返回是否通过检验的布尔值

    def _is_uniform(self, tensor, a, b):  # 定义检查张量是否符合均匀分布的方法
        samples = tensor.view(-1).tolist()  # 将张量展平并转换为列表
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]  # 使用Kolmogorov-Smirnov检验检查样本是否符合均匀分布
        return p_value > 0.0001  # 返回是否通过检验的布尔值

    def _create_random_nd_tensor(self, dims, size_min, size_max):  # 定义创建随机多维张量的方法
        size = [random.randint(size_min, size_max) for _ in range(dims)]  # 随机生成各维度大小的列表
        tensor = torch.zeros(size)  # 根据大小创建全零张量
        return tensor  # 返回创建的张量

    def _random_float(self, a, b):  # 定义生成指定范围内随机浮点数的方法
        return (b - a) * random.random() + a  # 返回指定范围内的随机浮点数

    def test_calculate_gain_linear(self):  # 定义测试线性激活函数gain计算的方法
        for fn in [  # 遍历线性操作函数列表
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose2d",
            "conv_transpose2d",
            "conv_transpose3d",
        ]:
            gain = init.calculate_gain(fn)  # 计算激活函数的gain
            self.assertEqual(gain, 1)  # 断言gain的值为1

    def test_calculate_gain_nonlinear(self):  # 定义测试非线性激活函数gain计算的方法
        for fn in ["sigmoid", "tanh", "relu", "leaky_relu"]:  # 遍历非线性激活函数列表
            gain = init.calculate_gain(fn)  # 计算激活函数的gain
            if fn == "sigmoid":
                self.assertEqual(gain, 1)  # 断言sigmoid函数的gain为1
            elif fn == "tanh":  # tanh函数的gain为5/3
                self.assertEqual(gain, 1.6666666666666667)
            elif fn == "relu":  # relu函数的gain为sqrt(2)
                self.assertEqual(gain, 1.4142135623730951)
            elif fn == "leaky_relu":  # leaky_relu函数的gain为sqrt(2 / (1 + slope^2))
                self.assertEqual(gain, 1.4141428569978354)
            elif fn == "selu":  # selu函数的gain为0.75
                self.assertEqual(gain, 0.75)
    def test_calculate_gain_leaky_relu(self):
        # 针对不同参数测试初始化"leaky_relu"的增益值计算
        for param in [None, 0, 0.01, 10]:
            # 调用初始化模块的函数计算"leaky_relu"的增益值
            gain = init.calculate_gain("leaky_relu", param)
            if param is None:  # 默认斜率为0.01时的预期增益
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 0:  # 没有斜率时，与普通ReLU相同的增益
                self.assertEqual(gain, 1.4142135623730951)
            elif param == 0.01:
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 10:
                self.assertEqual(gain, 0.14071950894605836)

    def test_calculate_gain_leaky_relu_only_accepts_numbers(self):
        # 针对非数值参数测试"leaky_relu"初始化函数抛出值错误异常
        for param in [True, [1], {"a": "b"}]:
            with self.assertRaises(ValueError):
                init.calculate_gain("leaky_relu", param)

    def test_calculate_gain_only_accepts_valid_nonlinearities(self):
        # 针对不支持的非线性函数名称长度进行测试，确保抛出值错误异常
        for n in [2, 5, 25]:
            # 生成指定长度的随机字符串，用作非线性函数名称
            random_string = "".join(
                [random.choice(string.ascii_lowercase) for i in range(n)]
            )
            with self.assertRaises(ValueError):
                init.calculate_gain(random_string)

    @unittest.skipIf(not TEST_SCIPY, "Scipy未安装。")
    @skipIfTorchDynamo("在Dynamo环境下scipy.kstest失败")
    def test_uniform(self):
        # 针对不同维度的输入张量测试均匀分布初始化
        for dims in [1, 2, 4]:
            # 创建指定维度和大小范围的随机张量
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            # 随机生成均匀分布的参数a和b
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            # 调用初始化函数对输入张量进行均匀分布初始化
            init.uniform_(input_tensor, a=a, b=b)
            # 断言初始化结果符合预期的均匀分布
            assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy未安装。")
    @skipIfTorchDynamo("在Dynamo环境下scipy.kstest失败")
    def test_normal(self):
        # 针对不同维度的输入张量测试正态分布初始化
        for dims in [1, 2, 4]:
            # 创建指定维度和大小范围的随机张量
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            # 随机生成正态分布的均值和标准差
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            # 调用初始化函数对输入张量进行正态分布初始化
            init.normal_(input_tensor, mean=mean, std=std)
            # 断言初始化结果符合预期的正态分布
            assert self._is_normal(input_tensor, mean, std)

    @unittest.skipIf(not TEST_SCIPY, "Scipy未安装。")
    @skipIfTorchDynamo("在Dynamo环境下scipy.kstest失败")
    def test_trunc_normal(self):
        # 针对不同维度的输入张量测试截断正态分布初始化
        for dims in [1, 2, 4]:
            # 创建指定维度和大小范围的随机张量
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            # 随机生成截断正态分布的均值、标准差、下界a和上界b
            mean = self._random_float(-3, 3)
            std = self._random_float(0.01, 1)
            a = self._random_float(mean - 2 * std, mean)
            b = self._random_float(mean, mean + 2 * std)
            # 调用初始化函数对输入张量进行截断正态分布初始化
            init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)
            # 断言初始化结果符合预期的截断正态分布
            assert self._is_trunc_normal(input_tensor, mean, std, a, b)
    # 测试截断正态分布生成器
    def test_trunc_normal_generator(self):
        # 创建一个新的随机数生成器对象
        gen = torch.Generator()
        # 设置生成器的种子为42
        gen.manual_seed(42)
        # 创建一个形状为(5,)的空张量
        input_tensor = torch.empty(5)
        # 使用截断正态分布初始化输入张量
        init.trunc_normal_(input_tensor, generator=gen)

        # 创建一个形状为(5,)的空张量作为参考
        ref = torch.empty(5)
        # 设置全局的随机种子为42
        torch.manual_seed(42)
        # 使用截断正态分布初始化参考张量
        init.trunc_normal_(ref)

        # 断言输入张量与参考张量相等
        self.assertEqual(input_tensor, ref)
        # 断言输入张量是否符合指定的截断正态分布属性
        assert self._is_trunc_normal(input_tensor, mean=0, std=1, a=0, b=1)

    # 测试常数初始化函数
    def test_constant(self):
        # 遍历不同维度的情况
        for dims in [1, 2, 4]:
            # 创建一个随机形状的张量作为输入张量
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            # 生成一个随机的浮点数作为常数值
            val = self._random_float(1, 10)
            # 使用常数初始化函数初始化输入张量
            init.constant_(input_tensor, val)

            # 断言输入张量是否等于填充了常数值的克隆张量
            self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    # 测试全1和全0初始化函数
    def test_ones_and_zeros(self):
        # 遍历全1和全0初始化函数及其对应的初始值
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            # 遍历不同维度的情况
            for dims in [1, 2, 4]:
                # 创建一个随机形状的张量作为输入张量
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=1, size_max=5
                )
                # 使用全1或全0初始化函数初始化输入张量
                init_fn_(input_tensor)

                # 断言输入张量是否等于填充了对应值的克隆张量
                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    # 测试单位矩阵初始化函数
    def test_eye(self):
        # 创建一个随机形状的2维张量作为输入张量
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5)
        # 使用单位矩阵初始化函数初始化输入张量
        init.eye_(input_tensor)

        # 检查每个单元素
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    assert input_tensor[i][j] == 1
                else:
                    assert input_tensor[i][j] == 0

    # 测试单位矩阵初始化函数对于非2维输入的处理
    def test_eye_only_works_on_2d_inputs(self):
        # 遍历不同维度的情况
        for dims in [1, 3]:
            # 使用自定义异常检查是否抛出异常
            with self.assertRaises(ValueError):
                # 创建一个随机形状的张量作为输入张量
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                # 尝试使用单位矩阵初始化函数
                init.eye_(tensor)

    # 测试Dirac delta初始化函数的属性
    def test_dirac_properties(self):
        # 遍历不同的维数和组数
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # 准备随机大小的张量，确保符合组数的要求
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(
                    1, 5 * groups
                )  # 与a*groups相同的范围，但所有范围都允许
                # 确保第一个维度可以被组数整除
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims])

                # 使用Dirac delta初始化函数初始化张量
                init.dirac_(input_tensor, groups)

                # 计算输出和输入通道数
                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # 检查非零元素的数量是否等于最小维度数（每个组）
                assert torch.nonzero(input_tensor).size(0) == min_d * groups
                # 检查值的总和是否等于最小维度数乘以组数（可能存在精度问题，因此使用assertEqual）
                self.assertEqual(input_tensor.sum(), min_d * groups)
    # 测试函数：验证 `dirac_` 初始化方法仅适用于 3、4、5 维输入
    def test_dirac_only_works_on_3_4_5d_inputs(self):
        # 对于维度为 1、2、6 的情况，期望引发 ValueError 异常
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                # 创建一个随机的 N 维张量，并调用 `dirac_` 初始化方法
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.dirac_(tensor)

    # 测试函数：验证 `xavier_uniform_` 初始化方法在输入小于 2 维时会引发错误
    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        # 对于维度为 0、1 的情况
        for dims in [0, 1]:
            # 创建一个随机的 N 维张量
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                # 调用 `xavier_uniform_` 初始化方法
                init.xavier_uniform_(tensor)

    # 测试函数：验证 `xavier_normal_` 初始化方法在输入小于 2 维时会引发错误
    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        # 对于维度为 0、1 的情况
        for dims in [0, 1]:
            # 创建一个随机的 N 维张量
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                # 调用 `xavier_normal_` 初始化方法
                init.xavier_normal_(tensor)

    # 装饰器标记的测试函数：验证 `xavier_uniform_` 初始化方法的正确性
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @slowTest
    def test_xavier_uniform(self):
        # 对于是否使用增益和维度为 2、4 的情况
        for use_gain in [True, False]:
            for dims in [2, 4]:
                # 创建一个随机的 N 维张量
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                # 如果使用增益，随机生成增益值，并调用 `xavier_uniform_` 初始化方法
                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    # 否则，调用 `xavier_uniform_` 初始化方法
                    init.xavier_uniform_(input_tensor)

                # 计算输入张量的 fan_in 和 fan_out
                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                # 计算预期的标准差和边界
                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                bounds = expected_std * math.sqrt(3)
                # 断言输入张量在 [-bounds, bounds] 范围内是否均匀分布
                assert self._is_uniform(input_tensor, -bounds, bounds)

    # 装饰器标记的测试函数：验证 `xavier_normal_` 初始化方法的正确性
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_xavier_normal(self):
        # 对于是否使用增益和维度为 2、4 的情况
        for use_gain in [True, False]:
            for dims in [2, 4]:
                # 创建一个随机的 N 维张量
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                # 如果使用增益，随机生成增益值，并调用 `xavier_normal_` 初始化方法
                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    # 否则，调用 `xavier_normal_` 初始化方法
                    init.xavier_normal_(input_tensor)

                # 计算输入张量的 fan_in 和 fan_out
                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                # 计算预期的标准差
                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                # 断言输入张量是否服从均值为 0、标准差为 expected_std 的正态分布
                assert self._is_normal(input_tensor, 0, expected_std)
    # 定义测试方法：测试当输入维度小于2时，kaiming_uniform 初始化引发 ValueError 异常
    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        # 对每个维度（0维和1维）进行测试
        for dims in [0, 1]:
            # 使用断言检查是否引发 ValueError 异常
            with self.assertRaises(ValueError):
                # 创建一个随机的小于2维的张量，并尝试用 kaiming_uniform_ 初始化
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_uniform_(tensor)

    # 定义测试方法：测试当输入维度小于2时，kaiming_normal 初始化引发 ValueError 异常
    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        # 对每个维度（0维和1维）进行测试
        for dims in [0, 1]:
            # 使用断言检查是否引发 ValueError 异常
            with self.assertRaises(ValueError):
                # 创建一个随机的小于2维的张量，并尝试用 kaiming_normal_ 初始化
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_normal_(tensor)

    # 定义测试方法：测试对于零元素张量的 kaiming_uniform_ 初始化会引发警告
    def test_kaiming_uniform_warning_on_0element_tensor(self):
        # 创建一个空的张量（0 行 1 列）
        tensor = torch.empty(0, 1)
        # 使用 assertWarnsRegex 检查是否引发 UserWarning 并包含指定的警告信息
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            # 调用 kaiming_uniform_ 对空张量进行初始化
            _ = init.kaiming_uniform_(tensor)

    # 定义测试方法：测试对于零元素张量的 kaiming_normal_ 初始化会引发警告
    def test_kaiming_normal_warning_on_0element_tensor(self):
        # 创建一个空的张量（0 行 1 列）
        tensor = torch.empty(0, 1)
        # 使用 assertWarnsRegex 检查是否引发 UserWarning 并包含指定的警告信息
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            # 调用 kaiming_normal_ 对空张量进行初始化
            _ = init.kaiming_normal_(tensor)

    # 定义测试方法：测试 kaiming_uniform_ 初始化在不同条件下的正确性
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_kaiming_uniform(self):
        # 对每个条件进行组合测试
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    # 创建一个随机的 n 维张量
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    # 根据 use_a 和 mode 的值调用 kaiming_uniform_ 进行初始化
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

                    # 计算 fan_in 和 fan_out
                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    # 根据 mode 设置 n
                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    # 计算预期的标准差和边界
                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    
                    # 使用自定义的 _is_uniform 方法检查张量是否在指定范围内均匀分布
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    # 这部分代码可能还有其他内容，需要查看完整代码以提供完整注释
    # 使用 Kaiming 正态分布初始化张量的测试函数
    def test_kaiming_normal(self):
        # 遍历 use_a 变量的 True 和 False 两种取值
        for use_a in [True, False]:
            # 遍历 dims 变量的取值为 2 和 4
            for dims in [2, 4]:
                # 遍历 mode 变量的取值为 "fan_in" 和 "fan_out"
                for mode in ["fan_in", "fan_out"]:
                    # 创建一个随机形状的张量作为输入张量
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    # 根据 use_a 的取值决定是否生成随机的 a 值
                    if use_a:
                        a = self._random_float(0.1, 2)
                        # 使用 Kaiming 正态分布初始化张量
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        # 使用 Kaiming 正态分布初始化张量
                        init.kaiming_normal_(input_tensor, mode=mode)

                    # 计算输入张量的 fan_in 和 fan_out
                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    # 如果输入张量的维度大于 2，更新 fan_in 和 fan_out
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    # 根据 mode 的取值确定 n 的值
                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    # 计算预期的标准差
                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    # 断言输入张量是否服从正态分布
                    assert self._is_normal(input_tensor, 0, expected_std)

    # 测试 sparse_ 函数在非 2 维输入上是否会抛出 ValueError 异常
    def test_sparse_only_works_on_2d_inputs(self):
        # 遍历 dims 变量的取值为 1 和 3
        for dims in [1, 3]:
            # 使用断言检查是否会抛出 ValueError 异常
            with self.assertRaises(ValueError):
                # 创建一个随机形状的张量作为输入张量
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                # 使用 sparse_ 函数初始化稀疏张量，预期会抛出异常
                init.sparse_(tensor, self._random_float(0.1, 0.9))

    # 跳过测试，条件是没有找到 Lapack
    @skipIfNoLapack
    # 定义一个测试方法，用于测试正交初始化函数的行为
    def test_orthogonal(self):
        # 对于是否使用增益参数进行循环测试
        for use_gain in [True, False]:
            # 对于不同的张量大小进行循环测试
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                # 创建指定大小的零张量
                input_tensor = torch.zeros(tensor_size)
                # 设置默认增益值为1.0
                gain = 1.0

                # 如果指定使用增益参数，则随机生成一个增益值
                if use_gain:
                    gain = self._random_float(0.1, 2)
                    # 对输入张量进行正交初始化，使用指定的增益值
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    # 对输入张量进行正交初始化，使用默认的增益值
                    init.orthogonal_(input_tensor)

                # 计算张量的行数和列数
                rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                # 将输入张量展平为二维矩阵
                flattened_tensor = input_tensor.view(rows, cols)

                # 根据行数和列数的关系选择不同的测试方式
                if rows > cols:
                    # 断言：转置后的张量与其自身的乘积应接近单位矩阵乘以增益值的平方
                    self.assertEqual(
                        torch.mm(flattened_tensor.t(), flattened_tensor),
                        torch.eye(cols) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )
                else:
                    # 断言：张量与其转置的乘积应接近单位矩阵乘以增益值的平方
                    self.assertEqual(
                        torch.mm(flattened_tensor, flattened_tensor.t()),
                        torch.eye(rows) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )

    # 定义一个测试方法，用于测试函数的弃用提示
    def test_deprecation(self):
        # 创建一个 3x3 的随机张量
        x = torch.randn(3, 3)

        # 定义一个嵌套函数，调用将被弃用的初始化函数
        def fn():
            init.normal(x)

        # 使用断言检查是否产生了弃用警告，并验证警告消息的内容
        with self.assertWarnsRegex(
            FutureWarning,
            "deprecated",
            msg="methods not suffixed with underscore should be deprecated",
        ):
            fn()
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行时，执行以下代码块
    run_tests()
```