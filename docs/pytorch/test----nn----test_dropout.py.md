# `.\pytorch\test\nn\test_dropout.py`

```py
# Owner(s): ["module: nn"]
# 导入必要的模块和库：itertools 用于迭代操作，random 用于生成随机数，unittest 用于单元测试
import itertools
import random
import unittest
from itertools import product

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入用于测试的特定 CUDA 和设备类型相关模块
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    expectedFailureXLA,
    instantiate_device_type_tests,
)

# 导入用于测试的公共神经网络模块和工具函数
from torch.testing._internal.common_nn import freeze_rng_state, NNTestCase

# 导入用于测试的参数化测试和运行测试的工具函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    set_default_dtype,
    TEST_PRIVATEUSE1,
)

# 定义测试类 TestDropoutNN，继承自 NNTestCase
class TestDropoutNN(NNTestCase):
    # 开启 CUDA 内存泄漏检查
    _do_cuda_memory_leak_check = True
    # 使用非默认 CUDA 流
    _do_cuda_non_default_stream = True

    # 测试 AlphaDropout 的方法
    def _test_alpha_dropout(self, cls, input):
        # 计算输入张量的均值和标准差
        mean = input.mean()
        std = input.std()

        # 对不同的丢弃概率进行测试
        for p in [0.2, 0.5, 0.8]:
            # 实例化 AlphaDropout 模块
            module = cls(p)
            # 对输入张量进行操作：分离出张量、克隆、需要梯度
            input_var = input.detach().clone().requires_grad_()
            # 应用 AlphaDropout 模块
            output = module(input_var)
            # 断言输出的均值接近于输入的均值
            self.assertLess(abs(output.data.mean() - mean), 0.1)
            # 断言输出的标准差接近于输入的标准差
            self.assertLess(abs(output.data.std() - std), 0.1)
            # 对输出进行反向传播
            output.backward(input)

    # 测试 AlphaDropout 模块
    def test_AlphaDropout(self):
        # 生成零均值和单位标准差的随机张量
        input = torch.randn(5000)
        self._test_alpha_dropout(nn.AlphaDropout, input)

    # 测试 FeatureAlphaDropout 模块
    def test_FeatureAlphaDropout(self):
        # 随机生成批次、宽度、高度和深度的尺寸
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        # 生成符合指定尺寸的随机输入张量
        input = torch.randn(num_features, b, d, w, h)
        self._test_alpha_dropout(nn.FeatureAlphaDropout, input)

        # 对于无批次维度的情况，生成具有指定尺寸的随机输入张量
        input = torch.randn(50, 20, 64, 64)
        self._test_alpha_dropout(nn.FeatureAlphaDropout, input)

    # 在 CUDA 或 PRIVATEUSE1 可用时执行测试，否则跳过
    @unittest.skipIf(
        not (TEST_CUDA or TEST_PRIVATEUSE1), "CUDA and PRIVATEUSE1 unavailable"
    )
    def test_native_dropout_corner_case(self):
        # 根据当前环境选择设备
        if TEST_CUDA:
            device = "cuda"
        elif TEST_PRIVATEUSE1:
            device = torch._C._get_privateuse1_backend_name()
        
        # 对于每种训练状态和丢弃概率的组合，以及每种当前设备的情况，执行测试
        for train in [True, False]:
            for p in [0.0, 1.0]:
                for current_device in [device, "cpu"]:
                    # 在指定设备上生成随机张量，并要求梯度
                    x = torch.randn(5).to(device=current_device).requires_grad_()
                    # 分离张量并保留梯度
                    x_ref = x.detach().requires_grad_()
                    # 调用本地的 dropout 函数
                    o = torch.native_dropout(x, p, train)[0]
                    # 使用 PyTorch 的 dropout 函数
                    o_ref = torch.dropout(x_ref, p, train)
                    # 断言两个张量相等
                    assert o.equal(o_ref)
                    # 断言梯度相等
                    assert x.grad.equal(x_ref.grad)
    # 定义测试函数，用于测试不合法的 dropout 概率参数是否会引发 ValueError 异常
    def test_invalid_dropout_p(self):
        # 创建一个包含单个元素的张量 v，其值为 1
        v = torch.ones(1)
        # 测试 nn.Dropout 类构造时传入负数概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout(-0.1))
        # 测试 nn.Dropout 类构造时传入超出范围的概率（大于1）是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout(1.1))
        # 测试 nn.Dropout1d 类构造时传入负数概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout1d(-0.1))
        # 测试 nn.Dropout1d 类构造时传入超出范围的概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout1d(1.1))
        # 测试 nn.Dropout2d 类构造时传入负数概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout2d(-0.1))
        # 测试 nn.Dropout2d 类构造时传入超出范围的概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout2d(1.1))
        # 测试 nn.Dropout3d 类构造时传入负数概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout3d(-0.1))
        # 测试 nn.Dropout3d 类构造时传入超出范围的概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: nn.Dropout3d(1.1))
        # 测试 F.dropout 函数调用时传入负数概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: F.dropout(v, -0.1))
        # 测试 F.dropout 函数调用时传入超出范围的概率是否引发 ValueError 异常
        self.assertRaises(ValueError, lambda: F.dropout(v, 1.1))
# 定义一个测试类 TestDropoutNNDeviceType，继承自 NNTestCase
class TestDropoutNNDeviceType(NNTestCase):

    # 定义测试函数 _test_dropout，用于测试 dropout 功能
    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        # 设定 dropout 概率为 0.2
        p = 0.2
        # 将输入数据转移到指定设备上，并填充为 (1 - p)
        input = input.to(device).fill_(1 - p)

        # 实例化一个 dropout 模块
        module = cls(p)
        # 克隆输入数据，并声明需要梯度计算
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        # 将输入数据传递给 dropout 模块，得到输出
        output = module(input_var)
        # 断言输出在指定内存格式下是连续的
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        # 断言输出数据的均值接近于 (1 - p)
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        # 对输出进行反向传播
        output.backward(input)
        # 断言输入数据的梯度在指定内存格式下是连续的
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        # 断言输入数据的梯度均值接近于 (1 - p)
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # 以 inplace=True 和 inplace=False 分别实例化 dropout 模块并进入评估模式，验证评估模式对结果是否有影响
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            # 断言在评估模式下，模块对输入的操作不改变其值
            self.assertEqual(input, module(input))

        # 验证 __repr__() 和 str() 方法是否能够正常调用
        module.__repr__()
        str(module)

    # 定义测试函数 _test_dropout_discontiguous，测试在不同内存格式下 dropout 是否能正确保留数据布局和数值
    def _test_dropout_discontiguous(
        self, cls, device, memory_format=torch.contiguous_format
    ):
        # 在这个测试中，验证 dropout 在不同内存格式下能否正确保留布局和数据
        # 检查当 dropout 概率接近于 0 时，输出值与输入值是否相同
        close_to_zero_p = 1e-10  # 应该接近于零但不等于零，因为 p=0 时会采取不同的路径
        for p in [0, close_to_zero_p]:
            # 创建一个全为 1 的张量作为输入
            inp = torch.ones(2, 3, 3, 3, device=device)
            # 在指定内存格式下创建一个不连续的张量，复制输入数据并取偶数位置的元素
            inp_discontiguous = torch.empty(
                2, 3, 3, 6, device=device, memory_format=memory_format
            )[..., ::2]
            inp_discontiguous.copy_(inp)
            # 实例化一个 dropout 模块
            mod = cls(p=p)
            # 对不连续的输入进行 dropout 操作，得到输出
            out = mod(inp_discontiguous)
            # 如果 p 不为 0，则检查输出在指定内存格式下是否是连续的
            if p != 0:
                self.assertTrue(out.is_contiguous(memory_format=memory_format))
            # 检查输入与输出张量是否相等
            self.assertEqual(inp_discontiguous, out)
    # 测试函数：测试在给定类和设备下的 Dropout 层的行为是否符合预期
    def _test_dropout_stride_mean_preserve(self, cls, device):
        # 辅助函数：反转排列顺序，返回排列的反向映射
        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2], d[3])

        # 创建一个形状为 (2, 3, 4, 5) 的全一张量，并将其移动位置的组合
        inp = torch.ones(2, 3, 4, 5, device=device)
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        # 遍历所有的排列方式
        for perm in itertools.permutations((0, 1, 2, 3), r=4):
            # 遍历所有的平移方式
            for shift in shifts:
                # 遍历不同的丢弃概率
                for p in [1e-10, 0.3, 0.5, 0.7]:
                    # 使用给定的丢弃概率创建一个丢弃层对象
                    mod = cls(p=p)
                    # 对输入进行排列和连续化操作，并且再次反转回原始排列
                    permuted_inp = (
                        inp.permute(perm).contiguous().permute(invert_perm(perm))
                    )
                    # 根据指定的平移裁剪输入张量
                    permuted_inp = permuted_inp[shift[0] :, shift[1] :, :, :]
                    # 应用丢弃层到裁剪后的输入上
                    out = mod(permuted_inp)

                    # 断言输出张量按照排列顺序是连续的
                    self.assertTrue(out.permute(perm).is_contiguous())
                    # 断言输入和输出的均值在指定的误差范围内相等
                    self.assertEqual(inp.mean(), out.mean(), rtol=0.5, atol=0.5)
                    # 根据丢弃概率判断输出是否等于输入
                    if p == 1e-10:
                        self.assertEqual(permuted_inp, out)
                    else:
                        self.assertNotEqual(permuted_inp, out)

    # 测试函数：测试 Dropout 层在给定设备上的行为
    def test_Dropout(self, device):
        # 创建一个形状为 (1000,) 的空张量作为输入
        input = torch.empty(1000)
        # 测试标准 Dropout 行为
        self._test_dropout(nn.Dropout, device, input)

        # 测试在不连续内存情况下的 Dropout 行为
        self._test_dropout_discontiguous(nn.Dropout, device)
        # 测试在通道优先内存格式下的 Dropout 行为
        self._test_dropout_discontiguous(
            nn.Dropout, device, memory_format=torch.channels_last
        )

        # 测试 Dropout 层在不同条件下对步长和均值的保持行为
        self._test_dropout_stride_mean_preserve(nn.Dropout, device)

        # 如果设备类型是 "cuda" 或 "cpu"，将输入张量转换为 bf16 类型后再次测试 Dropout 行为
        if self.device_type == "cuda" or self.device_type == "cpu":
            input = input.bfloat16()
            self._test_dropout(nn.Dropout, device, input)

    # 辅助测试函数：测试没有批处理维度的 Dropout 行为
    def _test_dropoutNd_no_batch(self, dropout, input):
        # 克隆输入张量
        input_clone = input.clone()
        # 使用冻结的随机数状态计算没有批处理维度的 Dropout 结果
        with freeze_rng_state():
            res_no_batch = dropout(input)

        # 使用冻结的随机数状态计算有批处理维度的 Dropout 结果，并且去掉额外的批处理维度
        with freeze_rng_state():
            res_batched = dropout(input_clone.unsqueeze(0)).squeeze(0)

        # 断言没有批处理维度和有批处理维度的结果相等
        self.assertEqual(res_no_batch, res_batched)

    # 辅助测试函数：测试通道为零的 Dropout 行为
    def _test_dropoutNd_channel_zero(self, dropout, input):
        # 验证通道中零的数量是 0 或通道中元素的数量
        shape = input.shape
        B = shape[0]
        C = shape[1]
        channel_numel = torch.tensor(shape[2:]).prod()
        result = dropout(input)

        # 对每个批次和通道组合进行验证
        for b, c in product(range(B), range(C)):
            self.assertTrue(result[b, c].count_nonzero() in (0, channel_numel))

    # 预期失败的 XLA 测试：XLA 不支持冻结随机数状态的功能
    @expectedFailureXLA  # 似乎 XLA 不支持冻结随机数状态
    # 定义名为 test_Dropout1d 的测试函数，接受 device 参数
    def test_Dropout1d(self, device):
        # 使用 torch.set_default_dtype 设置默认张量类型为双精度浮点数
        with set_default_dtype(torch.double):
            # 随机生成 N、C、L 分别为 10 到 15 之间的整数
            N, C, L = (
                random.randint(10, 15),
                random.randint(10, 15),
                random.randint(10, 15),
            )
            # 创建一个大小为 (N, C, L) 的空张量 input
            input = torch.empty(N, C, L)
            # 调用 self._test_dropout 方法，测试 nn.Dropout1d 的功能，传入 input 张量和设备类型 device

            # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并检查错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Expected 2D or 3D input, but received a 4D input"
            ):
                # 使用 nn.Dropout1d 对 4D 输入进行测试，期望抛出异常

            # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并检查错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Expected 2D or 3D input, but received a 1D input"
            ):
                # 使用 nn.Dropout1d 对 1D 输入进行测试，期望抛出异常

            # 创建一个大小为 (50, 2) 的随机张量 input，并测试 nn.Dropout1d 的功能，不考虑批次维度
            input = torch.rand(50, 2, device=device)
            self._test_dropoutNd_no_batch(nn.Dropout1d(p=0.5), input)
            # 测试带 inplace=True 参数的 nn.Dropout1d 的功能，不考虑批次维度
            self._test_dropoutNd_no_batch(nn.Dropout1d(p=0.5, inplace=True), input)

            # 创建一个大小为 (10, 4, 2) 的全一张量 input，并测试 nn.Dropout1d 的功能，检查通道是否完全丢弃
            input = torch.ones(10, 4, 2, device=device)
            self._test_dropoutNd_channel_zero(nn.Dropout1d(p=0.5), input)
            # 测试带 inplace=True 参数的 nn.Dropout1d 的功能，检查通道是否完全丢弃
            self._test_dropoutNd_channel_zero(nn.Dropout1d(p=0.5, inplace=True), input)

    # 标记此处的测试为预期失败，由于 XLA 不遵守 freeze_rng_state
    @expectedFailureXLA  # seems like freeze_rng_state is not honoured by XLA
    # 定义一个名为 test_Dropout2d 的测试函数，参数包括设备类型 device
    def test_Dropout2d(self, device):
        # 随机生成 batch size b、宽度 w、高度 h
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        # 设置特征数为 1000，创建一个空的张量 input，维度为 (num_features, b, w, h)
        num_features = 1000
        input = torch.empty(num_features, b, w, h)

        # 调用 self._test_dropout 方法测试 nn.Dropout2d，传入设备和 input 张量
        self._test_dropout(nn.Dropout2d, device, input)
        
        # 调用 self._test_dropout 方法测试 nn.Dropout2d，并指定 memory_format 为 torch.channels_last
        self._test_dropout(
            nn.Dropout2d, device, input, memory_format=torch.channels_last
        )

        # 调用 self._test_dropout_discontiguous 方法测试 nn.Dropout2d
        self._test_dropout_discontiguous(nn.Dropout2d, device)
        
        # 调用 self._test_dropout_discontiguous 方法测试 nn.Dropout2d，并指定 memory_format 为 torch.channels_last
        self._test_dropout_discontiguous(
            nn.Dropout2d, device, memory_format=torch.channels_last
        )

        # 使用 assertWarnsRegex 检查是否会收到 UserWarning，警告信息为 "Received a 5-D input to dropout2d"
        with self.assertWarnsRegex(UserWarning, "Received a 5-D input to dropout2d"):
            # 调用 nn.Dropout2d(p=0.5) 对 5 维的随机张量进行操作，传入设备参数 device
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, 2, 2, device=device))

        # 使用 assertWarnsRegex 检查是否会收到 UserWarning，警告信息为 "Received a 2-D input to dropout2d"
        with self.assertWarnsRegex(UserWarning, "Received a 2-D input to dropout2d"):
            # 调用 nn.Dropout2d(p=0.5) 对 2 维的随机张量进行操作，传入设备参数 device
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, device=device))

        # 下面的代码段是一个 TODO 注释，当前不支持无 batch 维度的输入
        # 暂时会按照历史的 dropout1d 行为处理 3 维输入
        # 详细信息参见 https://github.com/pytorch/pytorch/issues/77081

        # 对于 3 维的输入进行测试，因为当前不支持无 batch 维度的输入，这些行被注释掉
        # input = torch.rand(50, 2, 2, device=device)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5), input)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5, inplace=True), input)

        # 使用 assertWarnsRegex 检查是否会收到 UserWarning，警告信息为 "assuming that channel-wise 1D dropout behavior is desired"
        with self.assertWarnsRegex(
            UserWarning, "assuming that channel-wise 1D dropout behavior is desired"
        ):
            # 调用 nn.Dropout2d(p=0.5) 对 4 维的随机张量进行操作，传入设备参数 device
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, device=device))

        # 检查是否完整地丢弃通道
        input = torch.ones(10, 4, 2, 2, device=device)
        # 调用 self._test_dropoutNd_channel_zero 方法测试 nn.Dropout2d，传入 p=0.5 和 input 张量
        self._test_dropoutNd_channel_zero(nn.Dropout2d(p=0.5), input)
        # 调用 self._test_dropoutNd_channel_zero 方法测试 nn.Dropout2d，传入 p=0.5、inplace=True 和 input 张量
        self._test_dropoutNd_channel_zero(nn.Dropout2d(p=0.5, inplace=True), input)
    # 定义名为 test_Dropout3d 的测试方法，接受一个设备参数 device
    def test_Dropout3d(self, device):
        # 随机生成 b, w, h, d 四个整数，范围在 [1, 5] 之间
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        # 设置 num_features 为 1000
        num_features = 1000
        # 创建一个空的 Tensor，形状为 (num_features, b, d, w, h)，并赋值给 input
        input = torch.empty(num_features, b, d, w, h)
        # 调用 self._test_dropout 方法，测试 nn.Dropout3d 的效果，传入设备和输入数据 input
        self._test_dropout(nn.Dropout3d, device, input)

        # 调用 self._test_dropout_discontiguous 方法，测试不连续的 Tensor 对 nn.Dropout3d 的影响
        self._test_dropout_discontiguous(nn.Dropout3d, device)
        # 再次调用 self._test_dropout_discontiguous 方法，这次使用内存格式为 torch.channels_last
        self._test_dropout_discontiguous(
            nn.Dropout3d, device, memory_format=torch.channels_last
        )

        # 使用 assertWarnsRegex 检查是否会收到 UserWarning，提示输入到 dropout3d 的维度为 6-D
        with self.assertWarnsRegex(UserWarning, "Received a 6-D input to dropout3d"):
            # 调用 nn.Dropout3d(p=0.5)，输入一个形状为 (1, 2, 2, 2, 2, 2) 的随机 Tensor，并指定设备
            nn.Dropout3d(p=0.5)(torch.rand(1, 2, 2, 2, 2, 2, device=device))

        # 使用 assertWarnsRegex 检查是否会收到 UserWarning，提示输入到 dropout3d 的维度为 3-D
        with self.assertWarnsRegex(UserWarning, "Received a 3-D input to dropout3d"):
            # 调用 nn.Dropout3d(p=0.5)，输入一个形状为 (1, 2, 2) 的随机 Tensor，并指定设备
            nn.Dropout3d(p=0.5)(torch.rand(1, 2, 2, device=device))

        # 对没有批量维度的输入进行测试，创建一个形状为 (50, 2, 2, 2) 的随机 Tensor，并指定设备
        input = torch.rand(50, 2, 2, 2, device=device)
        # 调用 self._test_dropoutNd_no_batch 方法，测试不带批量维度的情况下 nn.Dropout3d 的效果
        self._test_dropoutNd_no_batch(nn.Dropout3d(p=0.5), input)
        # 再次调用 self._test_dropoutNd_no_batch 方法，测试 inplace=True 模式下的效果
        self._test_dropoutNd_no_batch(nn.Dropout3d(p=0.5, inplace=True), input)

        # 检查是否完整的通道被丢弃，创建一个形状为 (10, 4, 2, 2, 2) 的全 1 Tensor，并指定设备
        input = torch.ones(10, 4, 2, 2, 2, device=device)
        # 调用 self._test_dropoutNd_channel_zero 方法，检查通道被完全丢弃时 nn.Dropout3d 的效果
        self._test_dropoutNd_channel_zero(nn.Dropout3d(p=0.5), input)
        # 再次调用 self._test_dropoutNd_channel_zero 方法，测试 inplace=True 模式下的效果
        self._test_dropoutNd_channel_zero(nn.Dropout3d(p=0.5, inplace=True), input)

    # 定义名为 test_empty_dropout 的测试方法，接受一个设备参数 device
    def test_empty_dropout(self, device):
        # 创建一个空的 Tensor x，并将其移动到指定设备
        x = torch.tensor([]).to(device)
        # 对空的 Tensor x 应用 nn.functional.dropout 函数
        out = torch.nn.functional.dropout(x)
        # 使用 assertEqual 检查输出 out 的形状与输入 x 的形状相同
        self.assertEqual(out.size(), x.size())
# 调用函数，实例化与设备类型相关的测试类 TestDropoutNNDeviceType，并将其添加到全局命名空间中
instantiate_device_type_tests(TestDropoutNNDeviceType, globals())

# 实例化参数化测试类 TestDropoutNN，用于测试不同参数组合下的功能
instantiate_parametrized_tests(TestDropoutNN)

# 检查当前脚本是否作为主程序运行，如果是，则执行测试
if __name__ == "__main__":
    run_tests()
```