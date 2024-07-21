# `.\pytorch\test\quantization\core\experimental\test_adaround_eager.py`

```py
# Owner(s): ["oncall: speech_infra"]

# 导入所需的库和模块
import copy  # 导入深拷贝模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch.ao.quantization.experimental.adaround_optimization import (
    AdaptiveRoundingOptimizer,  # 导入自适应舍入优化器
)

from torch.nn import functional as F  # 导入函数别名
from torch.quantization.observer import MinMaxObserver  # 导入最小-最大观察器
from torch.testing._internal.common_quantization import QuantizationTestCase  # 导入量化测试用例基类


# 定义用于获取模型前后数据的回调函数
def forward_wrapper(fetcher):
    def forward(module, input, output):
        fetcher.append(input[0].detach())  # 将输入数据的第一个元素添加到fetcher中
        fetcher.append(output.detach())  # 将输出数据添加到fetcher中

    return forward


# 定义测试Adaround功能的测试类，继承自QuantizationTestCase
class TestAdaround(QuantizationTestCase):
    # 定义用于模型前向传播的回调函数，记录模型对数据的处理过程
    def feedforawrd_callback(
        self,
        model,
        data,
    ) -> None:
        model(data)

    # 定义带包装器的模型前向传播回调函数，使用自定义的包装器包裹模型
    def feedforawrd_callback_with_wrapper(self, model, data, wrapper) -> None:
        wrapper(model, data)

    # 运行Adaround优化过程的方法，返回优化后的模型
    def run_adaround(self, model, img_data, wrapper=None):
        # 创建AdaptiveRoundingOptimizer对象，配置参数并运行Adaround优化
        adaround_optimizer = AdaptiveRoundingOptimizer(
            model,
            self.feedforawrd_callback
            if wrapper is None
            else self.feedforawrd_callback_with_wrapper,
            forward_wrapper,
            img_data,
            max_iter=100,  # 最大迭代次数设为100
            batch_size=10,  # 批处理大小设为10
            feed_forward_wrapper=wrapper,  # 若存在包装器，则传入
        )
        adarounded_model = adaround_optimizer.run_adaround()  # 执行Adaround优化过程
        return adarounded_model  # 返回优化后的模型

    # 获取带有硬件仿真量化的模型的方法
    def get_fake_quant(self, model):
        hard_fake_quant_model = copy.deepcopy(model)  # 深拷贝原始模型
        for _, module in hard_fake_quant_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # 创建MinMaxObserver对象，观察权重的最小和最大值
                weight_observer = MinMaxObserver(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                )
                weight_observer(module.weight)  # 对权重进行观察
                scale, zero_point = weight_observer.calculate_qparams()  # 计算量化参数
                # 使用仿真量化创建fake_quant_module
                fake_quant_module = torch.fake_quantize_per_tensor_affine(
                    module.weight,
                    scale=scale,
                    zero_point=zero_point,
                    quant_min=-128,
                    quant_max=127,
                )
                module.weight.data.copy_(fake_quant_module)  # 将仿真量化后的结果复制回权重数据
        return hard_fake_quant_model  # 返回带有仿真量化的模型副本

    # 获取用于模型前向传播的包装器的方法
    def get_feed_forward_wrapper(self):
        # 定义用于包装模型的前向传播的类
        class FeedForwardWrapper(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, model, sample):
                return model(sample)  # 执行模型的前向传播

        wrapper_module = FeedForwardWrapper()  # 创建包装器对象
        return wrapper_module  # 返回包装器对象
    def test_linear_chain(self):
        # 定义一个简单的线性模型类
        class LinearChain(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义三个线性层，输入输出维度分别为 (3, 4), (4, 5), (5, 6)
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                # 模型的前向传播，依次通过三个线性层
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        # 创建一个浮点数版本的线性模型
        float_model = LinearChain()
        # 生成一个包含随机数据的列表作为输入
        img_data = [torch.rand(10, 3, dtype=torch.float) for _ in range(50)]
        # 对浮点数版本的模型进行自动环绕运行
        adarounded_model = self.run_adaround(
            float_model, img_data, self.get_feed_forward_wrapper()
        )
        # 获取一个使用伪量化的模型
        fq_model = self.get_fake_quant(float_model)
        # 生成一个随机输入张量
        rand_input = torch.rand(10, 3)
        # 使用 torch.no_grad() 上下文管理器来禁用梯度计算
        with torch.no_grad():
            # 对自动环绕模型进行推理
            ada_out = adarounded_model(rand_input)
            # 对伪量化模型进行推理
            fq_out = fq_model(rand_input)
            # 对浮点数版本的模型进行推理
            float_out = float_model(rand_input)
            # 计算自动环绕模型输出与浮点数版本模型输出之间的均方误差损失
            ada_loss = F.mse_loss(ada_out, float_out)
            # 计算伪量化模型输出与浮点数版本模型输出之间的均方误差损失
            fq_loss = F.mse_loss(fq_out, float_out)
            # 断言自动环绕模型的损失比伪量化模型的损失更低
            self.assertTrue(ada_loss.item() < fq_loss.item())

    def test_conv_chain(self):
        # 定义一个简单的卷积模型类
        class ConvChain(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义三个卷积层，输入输出通道和卷积核大小均为 (3, 4, 5x5), (4, 5, 5x5), (5, 6, 5x5)
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                # 模型的前向传播，依次通过三个卷积层
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x

        # 创建一个浮点数版本的卷积模型
        float_model = ConvChain()
        # 生成一个包含随机数据的列表作为输入
        img_data = [torch.rand(10, 3, 125, 125, dtype=torch.float) for _ in range(50)]
        # 对浮点数版本的模型进行自动环绕运行
        adarounded_model = self.run_adaround(float_model, img_data)
        # 获取一个使用伪量化的模型
        fq_model = self.get_fake_quant(float_model)
        # 生成一个随机输入张量
        rand_input = torch.rand(10, 3, 256, 256)
        # 使用 torch.no_grad() 上下文管理器来禁用梯度计算
        with torch.no_grad():
            # 对自动环绕模型进行推理
            ada_out = adarounded_model(rand_input)
            # 对伪量化模型进行推理
            fq_out = fq_model(rand_input)
            # 对浮点数版本的模型进行推理
            float_out = float_model(rand_input)
            # 计算自动环绕模型输出与浮点数版本模型输出之间的均方误差损失
            ada_loss = F.mse_loss(ada_out, float_out)
            # 计算伪量化模型输出与浮点数版本模型输出之间的均方误差损失
            fq_loss = F.mse_loss(fq_out, float_out)
            # 断言自动环绕模型的损失比伪量化模型的损失更低
            self.assertTrue(ada_loss.item() < fq_loss.item())
```