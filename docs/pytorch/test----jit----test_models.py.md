# `.\pytorch\test\jit\test_models.py`

```
# Owner(s): ["oncall: jit"]

# 导入标准库模块
import os
import sys
import unittest

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    ProfilingMode,
    set_default_dtype,
)

# 将 test/ 目录添加到系统路径，以便使其中的辅助文件可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import slowTest, suppress_warnings
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA

# 如果直接运行此文件，则抛出运行时错误，提示使用正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 尝试导入 torchvision，标记是否成功导入
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False

# 根据 torchvision 是否导入成功，决定是否跳过相关测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 定义一个简单的神经网络模型，继承自 nn.Module
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络结构：两个卷积层、两个线性层和一个 Dropout 层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # 定义前向传播方法
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 定义一个测试类，继承自 JitTestCase
class TestModels(JitTestCase):

    # 静态方法：测试 DCGAN 模型
    @staticmethod
    def test_dcgan_models(self):
        # 注意：如果使用 float 类型运行，可能会因低精度而失败
        with set_default_dtype(torch.double):
            self._test_dcgan_models(self, device="cpu")

    # 如果支持 CUDA，则测试 DCGAN 模型在 CUDA 上的运行情况
    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_dcgan_models_cuda(self):
        # 注意：如果使用 float 类型运行，可能会因低精度而失败
        with set_default_dtype(torch.double):
            # 注意：CUDA 模块上的 export_import 不工作 (#11480)
            self._test_dcgan_models(self, device="cuda", check_export_import=False)

    # 静态方法：测试神经风格转换模型
    @staticmethod
    @slowTest
    def test_neural_style(self):
        self._test_neural_style(self, device="cpu")

    # 如果支持 CUDA，则测试神经风格转换模型在 CUDA 上的运行情况
    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_neural_style_cuda(self):
        # 注意：CUDA 模块上的 export_import 不工作 (#11480)
        self._test_neural_style(self, device="cuda", check_export_import=False)

    # 如果使用的是 ProfilingMode.LEGACY 执行器，则跳过测试
    @unittest.skipIf(
        GRAPH_EXECUTOR == ProfilingMode.LEGACY, "Bug found in deprecated executor"
    )
    @staticmethod
    def _test_mnist(self, device, check_export_import=True):
        # 在进行性能分析测试时启用性能模式
        with enable_profiling_mode_for_profiling_tests():
            # 调用checkTrace方法检查MnistNet的跟踪情况：
            # 1. 创建MnistNet实例并移至指定设备，设置为评估模式
            # 2. 使用随机数据调用该网络
            # 3. 可选地导出和导入模型参数
            self.checkTrace(
                MnistNet().to(device).eval(),
                (torch.rand(5, 1, 28, 28, device=device),),
                export_import=check_export_import,
            )

    def test_mnist(self):
        # 执行针对CPU的MNIST测试
        self._test_mnist(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_cuda(self):
        # 在CUDA环境中执行MNIST测试，但不检查导出和导入的功能
        # 原因是CUDA模块上的导出和导入不可用（Issue #11480）
        self._test_mnist(self, device="cuda", check_export_import=False)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_training_leaks_no_memory_cuda(self):
        # 创建一个MnistNet实例并移至CUDA设备
        net = MnistNet().cuda()
        # 使用torch.jit.trace方法对网络进行跟踪，不检查跟踪的一致性
        traced_net = torch.jit.trace(
            net, [torch.randn(5, 1, 28, 28, device="cuda")], check_trace=False
        )

        def train(iters):
            for _ in range(iters):
                # 生成一些虚拟数据
                inp = torch.randn(5, 1, 28, 28, device="cuda")
                out = traced_net(inp)

                # 计算虚拟损失
                out.sum().backward()

                # 清空梯度
                traced_net.zero_grad()

        # 设置参数的.grad字段，以确保它们不会被报告为内存泄漏
        train(1)

        # 在没有CUDA张量泄漏的情况下进行测试
        with self.assertLeaksNoCudaTensors():
            train(5)

    @staticmethod
    def _test_reinforcement_learning(self, device, test_export_import=True):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.affine1 = nn.Linear(4, 128)
                self.affine2 = nn.Linear(128, 2)

            def forward(self, x):
                x = F.relu(self.affine1(x))
                action_scores = self.affine2(x)
                return F.softmax(action_scores, dim=1)

        with enable_profiling_mode_for_profiling_tests():
            # 调用checkTrace方法检查Policy的跟踪情况：
            # 1. 创建Policy实例并移至指定设备
            # 2. 使用随机数据调用该策略
            # 3. 可选地导出和导入模型参数
            self.checkTrace(
                Policy().to(device),
                (torch.rand(1, 4, device=device),),
                export_import=test_export_import,
            )

    def test_reinforcement_learning(self):
        # 执行针对CPU的强化学习策略测试
        self._test_reinforcement_learning(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_reinforcement_learning_cuda(self):
        # 在CUDA环境中执行强化学习策略测试，但不检查导出和导入的功能
        # 原因是CUDA模块上的导出和导入不可用（Issue #11480）
        self._test_reinforcement_learning(self, device="cuda", test_export_import=False)

    @staticmethod
    @slowTest
    def test_snli(self):
        # 执行针对CPU的SNLI测试
        self._test_snli(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_snli_cuda(self):
        # 在CUDA环境中执行SNLI测试，但不检查导出和导入的功能
        # 原因是CUDA模块上的导出和导入不可用（Issue #11480）
        self._test_snli(self, device="cuda", check_export_import=False)
    def _test_super_resolution(self, device, check_export_import=True):
        # 定义一个名为 Net 的神经网络模型类
        class Net(nn.Module):
            def __init__(self, upscale_factor):
                super().__init__()
                # 定义神经网络的各层及其参数
                self.relu = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
                self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
                self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
                self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
                self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            def forward(self, x):
                # 定义前向传播过程
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.pixel_shuffle(self.conv4(x))
                return x

        # 创建一个 Net 类的实例 net，传入 upscale_factor=4，并将其移动到指定设备上
        net = Net(upscale_factor=4).to(device)
        # 调用自定义函数 checkTrace 来验证网络，传入随机生成的张量作为输入，支持导出和导入操作
        self.checkTrace(
            net,
            (torch.rand(5, 1, 32, 32, device=device),),
            export_import=check_export_import,
        )

    @slowTest
    def test_super_resolution(self):
        # 调用 _test_super_resolution 函数，设备为 CPU，进行超分辨率测试
        self._test_super_resolution(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_super_resolution_cuda(self):
        # 在 CUDA 可用的情况下，调用 _test_super_resolution 函数，设备为 CUDA，不支持导出和导入操作的测试
        # XXX: export_import 在 CUDA 模块上不起作用 (#11480)
        self._test_super_resolution(self, device="cuda", check_export_import=False)

    @suppress_warnings
    def test_time_sequence_prediction(self):
        class Sequence(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义第一个 LSTM 细胞，输入维度为1，输出维度为51
                self.lstm1 = nn.LSTMCell(1, 51)
                # 定义第二个 LSTM 细胞，输入维度和输出维度均为51
                self.lstm2 = nn.LSTMCell(51, 51)
                # 定义一个线性层，将51维度的输出映射到1维度
                self.linear = nn.Linear(51, 1)

            @torch.jit.script_method
            def forward(self, input):
                # TODO: add future as input with default val
                # see https://github.com/pytorch/pytorch/issues/8724
                # 初始化一个空的张量 outputs，形状为(3, 0)
                outputs = torch.empty((3, 0))
                # 初始化第一个 LSTM 细胞的隐藏状态 h_t 和细胞状态 c_t，均为全零，形状为(3, 51)
                h_t = torch.zeros((3, 51))
                c_t = torch.zeros((3, 51))
                # 初始化第二个 LSTM 细胞的隐藏状态 h_t2 和细胞状态 c_t2，均为全零，形状为(3, 51)
                h_t2 = torch.zeros((3, 51))
                c_t2 = torch.zeros((3, 51))

                # 初始化输出，形状为[3, 51]
                output = torch.zeros([3, 51])
                # 设置未来预测的步数为2
                future = 2

                # TODO: chunk call should appear as the for loop iterable
                # We hard-code it to 4 for now.
                # 对输入进行分块，这里暂时硬编码为4个块
                a, b, c, d = input.chunk(input.size(1), dim=1)
                # 遍历每个输入块 input_t
                for input_t in (a, b, c, d):
                    # 应用第一个 LSTM 细胞
                    h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                    # 应用第二个 LSTM 细胞
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    # 使用线性层进行预测
                    output = self.linear(h_t2)
                    # 将当前输出拼接到 outputs 中
                    outputs = torch.cat((outputs, output), 1)
                # 如果需要预测未来
                for _ in range(future):
                    # 继续预测未来，类似上面的过程
                    h_t, c_t = self.lstm1(output, (h_t, c_t))
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    output = self.linear(h_t2)
                    outputs = torch.cat((outputs, output), 1)
                # 返回所有预测结果
                return outputs

        class Traced(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 Sequence 类的实例
                self.seq = Sequence()

            def forward(self, input):
                # 调用 Sequence 类的 forward 方法
                return self.seq.forward(input)

        # disabled due to a jitter issues that will be fixed by using load/store in the compiler
        # 禁用了由于编译器问题引起的抖动，将通过在编译器中使用加载/存储来解决
        with torch._jit_internal._disable_emit_hooks():
            # TODO: toggle export_import once above issues are fixed
            # 检查追踪结果是否正确，传入 Traced 类的实例和一个随机的输入张量
            self.checkTrace(Traced(), (torch.rand(3, 4),), export_import=False)

    @staticmethod
    # 定义测试用例 `_test_vae`，用于测试变分自编码器 (VAE)
    def _test_vae(self, device, check_export_import=True):
        # 定义变分自编码器类 VAE，继承自 nn.Module
        class VAE(nn.Module):
            # 初始化函数，定义神经网络结构
            def __init__(self):
                super().__init__()

                # 定义全连接层，输入784维，输出400维
                self.fc1 = nn.Linear(784, 400)
                # 均值网络层，输入400维，输出20维
                self.fc21 = nn.Linear(400, 20)
                # 方差网络层，输入400维，输出20维
                self.fc22 = nn.Linear(400, 20)
                # 解码网络层，输入20维，输出400维
                self.fc3 = nn.Linear(20, 400)
                # 输出层，输入400维，输出784维
                self.fc4 = nn.Linear(400, 784)

            # 编码函数，接受输入 x，输出均值 mu 和对数方差 logvar
            def encode(self, x):
                h1 = F.relu(self.fc1(x))
                return self.fc21(h1), self.fc22(h1)

            # 重参数化函数，根据均值 mu 和对数方差 logvar 生成随机采样结果
            def reparameterize(self, mu, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return eps.mul(std).add_(mu)
                else:
                    return mu

            # 解码函数，接受隐变量 z，返回重建的输出
            def decode(self, z):
                h3 = F.relu(self.fc3(z))
                return torch.sigmoid(self.fc4(h3))

            # 前向传播函数，对输入 x 进行编码、重参数化和解码操作，返回重建输出、均值和对数方差
            def forward(self, x):
                mu, logvar = self.encode(x.view(-1, 784))
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        # 启用测试模式进行性能分析
        with enable_profiling_mode_for_profiling_tests():
            # 调用 checkTrace 方法检查 VAE 模型的跟踪情况
            # 使用 torch.rand 生成输入数据，并转移到指定设备上
            # export_import 参数控制是否导出导入模型
            self.checkTrace(
                VAE().to(device).eval(),
                (torch.rand(128, 1, 28, 28, device=device),),
                export_import=check_export_import,
            )

    # 测试 CPU 上的变分自编码器
    def test_vae(self):
        self._test_vae(self, device="cpu")

    # 在 CUDA 上测试变分自编码器
    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_vae_cuda(self):
        # XXX: 在 CUDA 模块上禁用 export_import，因为其功能不可用 (#11480)
        self._test_vae(self, device="cuda", check_export_import=False)

    # 测试脚本模块跟踪 ResNet18 模型
    @slowTest
    @skipIfNoTorchVision
    def test_script_module_trace_resnet18(self):
        # 创建输入张量 x，尺寸为 1x3x224x224，全为1
        x = torch.ones(1, 3, 224, 224)
        # 对 torchvision 中的 ResNet18 模型进行跟踪
        m_orig = torch.jit.trace(
            torchvision.models.resnet18(), torch.ones(1, 3, 224, 224)
        )
        # 获取 ResNet18 模型的导出导入复制
        m_import = self.getExportImportCopy(m_orig)

        # 创建需要计算梯度的输入张量，大小为 1x3x224x224，随机初始化
        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        # 原始模型计算输出
        output_orig = m_orig(input)
        # 对原始模型输出进行求和并反向传播
        output_orig.sum().backward()
        # 复制梯度信息
        grad_orig = input.grad.clone()
        # 清零输入张量的梯度
        input.grad.zero_()

        # 导入模型计算输出
        output_import = m_import(input)
        # 对导入模型输出进行求和并反向传播
        output_import.sum().backward()
        # 复制导入模型的梯度信息
        grad_import = input.grad.clone()

        # 断言原始模型输出与导入模型输出相等
        self.assertEqual(output_orig, output_import)
        # 断言原始模型梯度与导入模型梯度相等
        self.assertEqual(grad_orig, grad_import)

    # 测试 AlexNet 模型
    @slowTest
    @skipIfNoTorchVision
    @skipIfNoTorchVision
    def test_alexnet(self):
        # 创建输入张量 x，尺寸为 1x3x224x224，全为1
        x = torch.ones(1, 3, 224, 224)
        # 创建 torchvision 中的 AlexNet 模型实例
        model = torchvision.models.AlexNet()
        # 使用 torch.random.fork_rng 创建一个随机数生成器环境
        with torch.random.fork_rng(devices=[]):
            # 获取模型的计算图和输入输出
            g, outputs, inputs = torch.jit._get_trace_graph(
                model, x, return_inputs=True
            )
        # 运行优化器 pass "cse"
        self.run_pass("cse", g)
        # 从计算图创建函数
        m = self.createFunctionFromGraph(g)
        with torch.random.fork_rng(devices=[]):
            # 断言模型的输出与预期输出相等
            self.assertEqual(outputs, m(*inputs))
```