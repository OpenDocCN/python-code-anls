# `.\pytorch\test\quantization\eager\test_bias_correction_eager.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.testing._internal.common_quantization import QuantizationTestCase  # 导入量化测试用例
from torch.testing._internal.common_quantization import skipIfNoFBGEMM  # 导入用于跳过没有FBGEMM的测试的装饰器

from torch.ao.quantization import default_qconfig  # 导入默认量化配置
from torch.ao.quantization import QuantWrapper  # 导入量化封装器
import torch.ao.ns._numeric_suite as ns  # 导入数值套件模块

from torch.ao.quantization._correct_bias import (  # 导入用于正确偏置的模块
    _supported_modules,
    _supported_modules_quantized,
    bias_correction,
    get_module,
    get_param,
    parent_child_names
)

import copy  # 导入复制模块


class TestBiasCorrectionEager(QuantizationTestCase):
    def compute_sqnr(self, x, y):
        '''计算信噪比（SQNR），使用PyTorch张量x和y计算'''
        Ps = torch.norm(x)  # 计算张量x的范数Ps
        Pn = torch.norm(x - y)  # 计算张量x和y之间差值的范数Pn
        return 20 * torch.log10(Ps / Pn)  # 返回信噪比（dB）

    def correct_artificial_bias_quantize(self, float_model, img_data):
        '''添加人工偏置并测试偏置在修正后是否保持不变。
           此测试用例更改量化子模块的偏置'''
        artificial_model = copy.deepcopy(float_model)  # 深度复制浮点模型以创建人工模型副本
        artificial_model.qconfig = default_qconfig  # 设置人工模型的量化配置为默认配置
        torch.ao.quantization.prepare(artificial_model, inplace=True)  # 在原地准备人工模型以进行量化
        for data in img_data:
            artificial_model(data[0])  # 对图像数据应用人工模型
        torch.ao.quantization.convert(artificial_model, inplace=True)  # 在原地将人工模型转换为量化模型

        # 手动更改偏置
        for name, submodule in artificial_model.named_modules():
            if type(submodule) in _supported_modules:  # 如果子模块类型支持偏置修改
                x = get_param(submodule, 'bias')  # 获取子模块的偏置参数x
                weight = get_param(submodule, 'weight')  # 获取子模块的权重参数
                if x is not None:
                    submodule.set_weight_bias(weight, x.data * 3)  # 设置子模块的权重和偏置（偏置乘以3）

        bias_correction(float_model, artificial_model, img_data, target_modules=_supported_modules_quantized)
        # 对浮点模型和人工模型进行偏置修正，针对量化支持的模块

        # 去除影子模块
        for name, submodule in artificial_model.named_modules():
            if isinstance(submodule, ns.Shadow):  # 如果子模块是影子模块
                parent_name, child_name = parent_child_names(name)  # 获取父模块和子模块名称
                parent = get_module(artificial_model, parent_name)  # 获取父模块
                parent._modules[child_name] = submodule.orig_module  # 将子模块的原始模块替换回父模块

        # 检查修正后的偏置
        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in _supported_modules_quantized:  # 如果人工子模块的类型在量化支持的模块中
                submodule = get_module(float_model, name)  # 获取浮点模型中的对应模块
                float_bias = get_param(submodule, 'bias')  # 获取浮点模型中的偏置
                artificial_bias = get_param(artificial_submodule, 'bias')  # 获取人工模型中的偏置

                # 断言：确保修正后的偏置信噪比大于30dB，否则打印错误信息
                self.assertTrue(self.compute_sqnr(float_bias, artificial_bias) > 30,
                                "Correcting quantized bias produced too much noise, sqnr score too low")

    @skipIfNoFBGEMM  # 如果没有FBGEMM，则跳过此测试
    # 定义测试线性层链的方法
    def test_linear_chain(self):
        # 定义线性层链模型
        class LinearChain(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 第一个线性层，输入维度为3，输出维度为4
                self.linear1 = nn.Linear(3, 4)
                # 第二个线性层，输入维度为4，输出维度为5
                self.linear2 = nn.Linear(4, 5)
                # 第三个线性层，输入维度为5，输出维度为6
                self.linear3 = nn.Linear(5, 6)

            # 前向传播方法
            def forward(self, x):
                # 应用第一个线性层
                x = self.linear1(x)
                # 应用第二个线性层
                x = self.linear2(x)
                # 应用第三个线性层
                x = self.linear3(x)
                return x
        # 创建一个QuantWrapper对象，包裹线性层链模型
        float_model = QuantWrapper(LinearChain())
        # 生成包含随机数据和随机标签的图像数据列表，共50个
        img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        # 调用self.correct_artificial_bias_quantize方法，对float_model和img_data进行量化处理的正确人工偏差处理

    # 根据FBGEMM库的可用性，跳过测试卷积层链的方法
    @skipIfNoFBGEMM
    def test_conv_chain(self):
        # 定义卷积层链模型
        class ConvChain(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 第一个卷积层，输入通道数为3，输出通道数为4，卷积核大小为5x5
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                # 第二个卷积层，输入通道数为4，输出通道数为5，卷积核大小为5x5
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                # 第三个卷积层，输入通道数为5，输出通道数为6，卷积核大小为5x5
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            # 前向传播方法
            def forward(self, x):
                # 应用第一个卷积层
                x = self.conv2d1(x)
                # 应用第二个卷积层
                x = self.conv2d2(x)
                # 应用第三个卷积层
                x = self.conv2d3(x)
                return x
        # 创建一个QuantWrapper对象，包裹卷积层链模型
        float_model = QuantWrapper(ConvChain())
        # 生成包含随机数据和随机标签的图像数据列表，共50个
        img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        # 调用self.correct_artificial_bias_quantize方法，对float_model和img_data进行量化处理的正确人工偏差处理
```