# `.\pytorch\test\optim\test_swa_utils.py`

```py
# Owner(s): ["module: optimizer"]

# 导入 itertools 和 pickle 模块
import itertools
import pickle

# 导入 PyTorch 库
import torch
# 从 torch.optim.swa_utils 中导入所需的函数和类
from torch.optim.swa_utils import (
    AveragedModel,
    get_ema_multi_avg_fn,
    get_swa_multi_avg_fn,
    update_bn,
)
# 从 torch.testing._internal.common_utils 中导入测试相关的函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    TestCase,
)

# load_tests 函数来自 common_utils，用于在 sandcastle 上自动过滤测试用例以进行分片。此行消除了 flake 警告
load_tests = load_tests


class TestSWAUtils(TestCase):
    # 定义一个用于测试的简单 DNN 模型
    class SWATestDNN(torch.nn.Module):
        def __init__(self, input_features):
            super().__init__()
            self.n_features = 100
            self.fc1 = torch.nn.Linear(input_features, self.n_features)
            self.bn = torch.nn.BatchNorm1d(self.n_features)

        def compute_preactivation(self, x):
            return self.fc1(x)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn(x)
            return x

    # 定义一个用于测试的简单 CNN 模型
    class SWATestCNN(torch.nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.n_features = 10
            self.conv1 = torch.nn.Conv2d(
                input_channels, self.n_features, kernel_size=3, padding=1
            )
            self.bn = torch.nn.BatchNorm2d(self.n_features, momentum=0.3)

        def compute_preactivation(self, x):
            return self.conv1(x)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn(x)
            return x

    # 测试函数，用于验证 AveragedModel 的工作方式
    def _test_averaged_model(self, net_device, swa_device, ema):
        # 定义一个简单的神经网络模型
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
        ).to(net_device)

        # 运行平均步骤，得到平均后的参数和模型
        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, swa_device, ema)

        # 遍历比较每个参数的平均值和 SWA 模型参数
        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            # 断言两者相等
            self.assertEqual(p_avg, p_swa)
            # 检查 AveragedModel 是否在正确的设备上
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p_avg.device == net_device)
        
        # 检查平均后的模型的设备是否正确
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)
    # 运行平均步骤的私有方法，用于计算模型参数的加权平均
    def _run_averaged_steps(self, dnn, swa_device, ema):
        # 指数移动平均的衰减率
        ema_decay = 0.999
        # 如果选择使用指数移动平均（EMA），则创建一个基于EMA的平均模型
        if ema:
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
        else:
            # 否则创建一个基于SWA（Stochastic Weight Averaging）的平均模型
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_swa_multi_avg_fn()
            )

        # 初始化一个与模型参数相同大小的零张量列表，用于存储平均后的参数
        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]

        # 更新次数
        n_updates = 10
        # 执行指定次数的更新步骤
        for i in range(n_updates):
            # 遍历模型参数和平均参数的对应项
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                # 断开梯度计算，添加正态分布随机数到模型参数
                p.detach().add_(torch.randn_like(p))
                # 如果使用EMA，更新平均参数
                if ema:
                    p_avg += (
                        p.detach()
                        * ema_decay ** (n_updates - i - 1)
                        * ((1 - ema_decay) if i > 0 else 1.0)
                    )
                else:
                    # 如果使用SWA，更新平均参数
                    p_avg += p.detach() / n_updates
            # 更新平均模型的参数
            averaged_dnn.update_parameters(dnn)

        # 返回平均后的参数和平均模型
        return averaged_params, averaged_dnn

    # 使用不同设备测试平均模型的方法，包括CPU和GPU
    @parametrize("ema", [True, False])
    def test_averaged_model_all_devices(self, ema):
        # CPU设备
        cpu = torch.device("cpu")
        # 在CPU上测试平均模型
        self._test_averaged_model(cpu, cpu, ema)
        # 如果CUDA可用，测试在CUDA设备上的平均模型
        if torch.cuda.is_available():
            cuda = torch.device(0)
            self._test_averaged_model(cuda, cpu, ema)
            self._test_averaged_model(cpu, cuda, ema)
            self._test_averaged_model(cuda, cuda, ema)

    # 在混合设备上测试平均模型的方法，如CPU和GPU的混合
    @parametrize("ema", [True, False])
    def test_averaged_model_mixed_device(self, ema):
        # 如果CUDA不可用，则直接返回
        if not torch.cuda.is_available():
            return
        # 创建包含卷积和线性层的序列模型
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        # 将卷积层移动到CUDA设备
        dnn[0].cuda()
        # 将线性层移动到CPU设备
        dnn[1].cpu()

        # 执行平均步骤，返回平均后的参数和平均模型
        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, None, ema)

        # 检查每个平均参数和平均模型的参数是否相等
        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # 检查平均模型参数是否位于正确的设备上
            self.assertTrue(p_avg.device == p_swa.device)

    # 测试平均模型状态字典的方法
    def test_averaged_model_state_dict(self):
        # 创建包含卷积和线性层的序列模型
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        # 创建一个基于SWA的平均模型
        averaged_dnn = AveragedModel(dnn)
        # 创建另一个基于SWA的平均模型
        averaged_dnn2 = AveragedModel(dnn)
        # 更新次数
        n_updates = 10
        # 执行指定次数的更新步骤
        for i in range(n_updates):
            # 遍历模型参数，添加正态分布随机数
            for p in dnn.parameters():
                p.detach().add_(torch.randn_like(p))
            # 更新平均模型的参数
            averaged_dnn.update_parameters(dnn)
        # 使用状态字典加载第一个平均模型的状态到第二个平均模型
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        # 检查每个平均模型的参数是否相等
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            self.assertEqual(p_swa, p_swa2)
        # 检查平均模型的平均次数是否相等
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)
    # 定义一个测试方法，测试默认的平均函数是否可被 pickle 序列化
    def test_averaged_model_default_avg_fn_picklable(self):
        # 创建一个包含卷积、批归一化和线性层的神经网络模型
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5),
            torch.nn.Linear(5, 5),
        )
        # 使用 AveragedModel 封装神经网络模型
        averaged_dnn = AveragedModel(dnn)
        # 对封装后的模型进行 pickle 序列化
        pickle.dumps(averaged_dnn)

    @parametrize("use_multi_avg_fn", [True, False])
    @parametrize("use_buffers", [True, False])
    # 定义测试方法，测试带有指数移动平均函数和使用缓冲区标志的 AveragedModel
    def test_averaged_model_exponential(self, use_multi_avg_fn, use_buffers):
        # 创建一个包含卷积、批归一化和线性层的神经网络模型
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Linear(5, 10),
        )
        # 设置指数移动平均的衰减率
        decay = 0.9

        # 根据 use_multi_avg_fn 参数选择使用的多平均函数
        if use_multi_avg_fn:
            averaged_dnn = AveragedModel(
                dnn, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=use_buffers
            )
        else:
            # 定义自定义的平均函数 avg_fn
            def avg_fn(p_avg, p, n_avg):
                return decay * p_avg + (1 - decay) * p

            # 使用自定义的平均函数创建 AveragedModel
            averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn, use_buffers=use_buffers)

        # 根据 use_buffers 参数选择要操作的参数列表
        if use_buffers:
            dnn_params = list(itertools.chain(dnn.parameters(), dnn.buffers()))
        else:
            dnn_params = list(dnn.parameters())

        # 初始化平均参数列表，与神经网络模型参数相同形状的零张量
        averaged_params = [
            torch.zeros_like(param)
            for param in dnn_params
            if param.size() != torch.Size([])
        ]

        # 执行多次参数更新
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            # 遍历神经网络模型的参数和对应的平均参数
            for p, p_avg in zip(dnn_params, averaged_params):
                # 跳过标量参数
                if p.size() == torch.Size([]):
                    continue
                # 对参数应用随机扰动
                p.detach().add_(torch.randn_like(p))
                # 根据衰减率更新平均参数
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * decay + p * (1 - decay)).clone()
                    )
            # 更新 AveragedModel 的参数
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        # 根据 use_buffers 参数验证平均参数与 AveragedModel 的参数是否一致
        if use_buffers:
            for p_avg, p_swa in zip(
                averaged_params,
                itertools.chain(
                    averaged_dnn.module.parameters(), averaged_dnn.module.buffers()
                ),
            ):
                self.assertEqual(p_avg, p_swa)
        else:
            for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
                self.assertEqual(p_avg, p_swa)
            for b_avg, b_swa in zip(dnn.buffers(), averaged_dnn.module.buffers()):
                self.assertEqual(b_avg, b_swa)
    # 测试用例：更新批归一化层的参数
    def _test_update_bn(self, dnn, dl_x, dl_xy, cuda):
        # 初始化总预激活和预激活平方和为零向量
        preactivation_sum = torch.zeros(dnn.n_features)
        preactivation_squared_sum = torch.zeros(dnn.n_features)
        # 如果使用 CUDA，则将张量移动到 GPU 上
        if cuda:
            preactivation_sum = preactivation_sum.cuda()
            preactivation_squared_sum = preactivation_squared_sum.cuda()
        # 总样本数量初始化为零
        total_num = 0
        # 遍历数据加载器 dl_x 中的每个批次 x
        for x in dl_x:
            # 从批次中获取数据 x
            x = x[0]
            # 如果使用 CUDA，则将数据 x 移动到 GPU 上
            if cuda:
                x = x.cuda()

            # 对神经网络进行前向传播计算
            dnn.forward(x)
            # 计算预激活值
            preactivations = dnn.compute_preactivation(x)
            # 如果预激活张量的维度为四维，则进行维度转换
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            # 将预激活张量展平为二维张量
            preactivations = preactivations.contiguous().view(-1, dnn.n_features)
            # 更新总样本数量
            total_num += preactivations.shape[0]

            # 更新总预激活和预激活平方和
            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations**2, dim=0)

        # 计算预激活均值和方差
        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean**2

        # 更新批归一化层参数
        update_bn(dl_xy, dnn, device=x.device)
        # 断言批归一化层的 running_mean 和 running_var 是否与预期值相等
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        # 定义函数：重置批归一化层的 running_mean 和 running_var
        def _reset_bn(module):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        # 重置批归一化层参数，并再次运行 update_bn
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=x.device)
        # 再次断言批归一化层的 running_mean 和 running_var 是否与预期值相等
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        # 使用 dl_x 数据加载器而不是 dl_xy 运行 update_bn
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=x.device)
        # 断言批归一化层的 running_mean 和 running_var 是否与预期值相等
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

    # 测试函数：测试对全连接神经网络应用批归一化
    def test_update_bn_dnn(self):
        objects, input_features = 100, 5
        # 创建输入张量 x 和目标张量 y
        x = torch.rand(objects, input_features)
        y = torch.rand(objects)
        # 创建数据集对象 ds_x 和 ds_xy
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        # 创建数据加载器 dl_x 和 dl_xy
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        # 创建全连接神经网络对象 dnn，并设置为训练模式
        dnn = self.SWATestDNN(input_features=input_features)
        dnn.train()
        # 测试在不使用 CUDA 的情况下运行 _test_update_bn 函数
        self._test_update_bn(dnn, dl_x, dl_xy, False)
        # 如果 CUDA 可用，则测试在使用 CUDA 的情况下运行 _test_update_bn 函数
        if torch.cuda.is_available():
            dnn = self.SWATestDNN(input_features=input_features)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, True)
        # 断言神经网络对象仍处于训练模式
        self.assertTrue(dnn.training)
    # 定义一个测试方法，用于测试更新卷积神经网络和 BatchNorm2d 的行为
    def test_update_bn_cnn(self):
        # 设定对象数量为 100
        objects = 100
        # 输入通道数为 3
        input_channels = 3
        # 图像高度和宽度设定为 5
        height, width = 5, 5
        # 生成随机张量 x，形状为 (objects, input_channels, height, width)
        x = torch.rand(objects, input_channels, height, width)
        # 生成随机张量 y，形状为 (objects,)
        y = torch.rand(objects)
        # 创建数据集 ds_x，包含张量 x
        ds_x = torch.utils.data.TensorDataset(x)
        # 创建数据集 ds_xy，包含张量 x 和 y
        ds_xy = torch.utils.data.TensorDataset(x, y)
        # 创建数据加载器 dl_x，从数据集 ds_x 加载数据，批量大小为 5，打乱数据顺序
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        # 创建数据加载器 dl_xy，从数据集 ds_xy 加载数据，批量大小为 5，打乱数据顺序
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        # 创建一个 SWA 测试用的 CNN，设定输入通道数为 input_channels
        cnn = self.SWATestCNN(input_channels=input_channels)
        # 将 CNN 设置为训练模式
        cnn.train()
        # 调用 _test_update_bn 方法，测试更新 BatchNorm 后的行为，传入 cnn, dl_x, dl_xy 和 False
        self._test_update_bn(cnn, dl_x, dl_xy, False)
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 创建一个新的 SWA 测试用的 CNN，设定输入通道数为 input_channels
            cnn = self.SWATestCNN(input_channels=input_channels)
            # 将 CNN 设置为训练模式
            cnn.train()
            # 将 CNN 移动到 CUDA 设备上，并设置为训练模式
            self._test_update_bn(cnn.cuda(), dl_x, dl_xy, True)
        # 断言 cnn 当前为训练模式
        self.assertTrue(cnn.training)

    # 定义一个测试方法，用于测试 BatchNorm 更新和 eval 模式下的动量是否保持不变
    def test_bn_update_eval_momentum(self):
        # 设定对象数量为 100
        objects = 100
        # 输入通道数为 3
        input_channels = 3
        # 图像高度和宽度设定为 5
        height, width = 5, 5
        # 生成随机张量 x，形状为 (objects, input_channels, height, width)
        x = torch.rand(objects, input_channels, height, width)
        # 创建数据集 ds_x，包含张量 x
        ds_x = torch.utils.data.TensorDataset(x)
        # 创建数据加载器 dl_x，从数据集 ds_x 加载数据，批量大小为 5，打乱数据顺序
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        # 创建一个 SWA 测试用的 CNN，设定输入通道数为 input_channels
        cnn = self.SWATestCNN(input_channels=input_channels)
        # 将 CNN 设置为评估模式
        cnn.eval()
        # 调用 update_bn 函数，更新 BatchNorm，传入 dl_x 和 cnn
        update_bn(dl_x, cnn)
        # 断言 cnn 当前不为训练模式
        self.assertFalse(cnn.training)

        # 断言评估模式下 BatchNorm 的动量是否为 0.3
        self.assertEqual(cnn.bn.momentum, 0.3)
# 实例化参数化测试用例，使用 TestSWAUtils 类来创建测试实例
instantiate_parametrized_tests(TestSWAUtils)

# 如果当前脚本被直接执行，则输出警告信息，建议通过 test/test_optim.py 来运行这些测试
if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
```