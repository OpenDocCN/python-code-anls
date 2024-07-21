# `.\pytorch\test\quantization\eager\test_equalize_eager.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

from torch.testing._internal.common_quantization import QuantizationTestCase  # 导入量化测试用例
from torch.ao.quantization.fuse_modules import fuse_modules  # 导入模块融合函数

import torch.ao.quantization._equalize as _equalize  # 导入权重均衡模块

import copy  # 导入拷贝模块

class TestEqualizeEager(QuantizationTestCase):
    def checkChannelsEqualized(self, tensor1, tensor2, output_axis, input_axis):
        ''' Checks the channel ranges of tensor1, tensor2 are the same,
        which is an indication that equalization has been applied correctly
        '''
        output_channel_tensor1 = _equalize.channel_range(tensor1, output_axis)  # 获取tensor1在输出轴上的通道范围
        input_channel_tensor2 = _equalize.channel_range(tensor2, input_axis)  # 获取tensor2在输入轴上的通道范围

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output
        self.assertEqual(output_channel_tensor1, input_channel_tensor2)  # 断言tensor1的输出通道范围与tensor2的输入通道范围相等

    def getModule(self, model, name):
        ''' Given the name is a submodule to a model, return the submodule
        '''
        curr = model  # 当前模型
        name = name.split('.')  # 将名称按点号拆分成列表
        for subname in name:
            curr = curr._modules[subname]  # 逐级获取子模块
        return curr  # 返回指定名称的子模块

    def test_cross_layer_equalization(self):
        ''' applies _equalize.cross_layer_equalization on two modules and checks
        to make sure channels ranges are equivalent
        '''
        module1 = nn.Conv2d(3, 4, 2)  # 创建一个2D卷积模块，输入通道数3，输出通道数4，卷积核大小2x2
        module2 = nn.Linear(4, 4)  # 创建一个线性模块，输入特征数4，输出特征数4

        module1_output_channel_axis = 0  # module1的输出通道轴
        module2_input_channel_axis = 1  # module2的输入通道轴

        _equalize.cross_layer_equalization(module1, module2)  # 应用跨层均衡函数_equalize.cross_layer_equalization

        mod_tensor1, mod_tensor2 = module1.weight, module2.weight  # 获取module1和module2的权重张量

        self.checkChannelsEqualized(mod_tensor1, mod_tensor2, module1_output_channel_axis, module2_input_channel_axis)  # 调用通道均衡检查函数

    def test_converged(self):
        ''' Sanity checks on _equalize.converged working
        identical modules should return true
        modules with high difference in weights should return false
        '''
        module1 = nn.Linear(3, 3)  # 创建一个线性模块，输入特征数3，输出特征数3
        module2 = nn.Linear(3, 3)  # 创建另一个线性模块，输入特征数3，输出特征数3

        module1.weight = nn.parameter.Parameter(torch.ones(module1.weight.size()))  # 设置module1的权重为全1张量
        module2.weight = nn.parameter.Parameter(torch.zeros(module1.weight.size()))  # 设置module2的权重为全0张量

        # input is a dictionary
        dictionary_1 = {'linear1': module1}  # 创建模块1的字典
        dictionary_2 = {'linear1': module2}  # 创建模块2的字典
        self.assertTrue(_equalize.converged(dictionary_1, dictionary_1, 1e-6))  # 断言模块1与自身的收敛性为真
        self.assertFalse(_equalize.converged(dictionary_1, dictionary_2, 1e-6))  # 断言模块1与模块2的收敛性为假
    # 定义一个测试方法，用于测试 _equalize.equalize 方法处理多个模块作为输入的能力
    # 然后通过比较相同输入情况下模型均衡和不均衡版本的输出来检查函数的正确性
    def test_equalize(self):
        # 定义一个简单的神经网络模型类 ChainModule
        class ChainModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 4)   # 第一个线性层，输入维度为3，输出维度为4
                self.linear2 = nn.Linear(4, 5)   # 第二个线性层，输入维度为4，输出维度为5
                self.linear3 = nn.Linear(5, 6)   # 第三个线性层，输入维度为5，输出维度为6

            def forward(self, x):
                x = self.linear1(x)   # 通过第一层线性层
                x = self.linear2(x)   # 通过第二层线性层
                x = self.linear3(x)   # 通过第三层线性层
                return x

        # 创建两个 ChainModule 类的实例
        chain1 = ChainModule()
        chain2 = copy.deepcopy(chain1)

        # 使用 _equalize.equalize 方法均衡 chain1 模型的指定线性层
        _equalize.equalize(chain1, [['linear1', 'linear2'], ['linear2', 'linear3']], 1e-6)

        # 获取经过均衡后的 chain1 模型的各个线性层
        linear1 = self.getModule(chain1, 'linear1')
        linear2 = self.getModule(chain1, 'linear2')
        linear3 = self.getModule(chain1, 'linear3')

        # 检查第一层和第二层线性层的权重是否均衡
        self.checkChannelsEqualized(linear1.weight, linear2.weight, 0, 1)
        # 检查第二层和第三层线性层的权重是否均衡
        self.checkChannelsEqualized(linear2.weight, linear3.weight, 0, 1)

        # 创建一个随机输入张量
        input = torch.randn(20, 3)
        # 断言经过均衡和未经均衡的 chain1 模型在相同输入下的输出应相等
        self.assertEqual(chain1(input), chain2(input))
    def test_equalize_fused_convrelu(self):
        '''
        检查是否支持融合 ConvReLU2d 模型的急切模式均衡化

        创建一个具有 3 个 ConvReLU2d 的模型。接下来，将 conv2d 和 relu 层融合在一起，
        并对相邻的 conv2d 层应用跨层均衡化。最后，确保通道已经均衡化，并且在相同输入的情况下，
        均衡化和非均衡化版本的模型输出相同。
        '''
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义三个 ConvReLU2d 层，每个层都使用了 3 个输入通道和 3 个输出通道的 1x1 卷积核
                self.conv1 = nn.Conv2d(3, 3, 1).to(dtype=torch.float)
                self.relu1 = nn.ReLU(inplace=False).to(dtype=torch.float)
                self.conv2 = nn.Conv2d(3, 3, 1).to(dtype=torch.float)
                self.relu2 = nn.ReLU(inplace=False).to(dtype=torch.float)
                self.conv3 = nn.Conv2d(3, 3, 1).to(dtype=torch.float)
                self.relu3 = nn.ReLU(inplace=False).to(dtype=torch.float)

            def forward(self, x):
                # 模型的前向传播过程，依次经过 conv1、relu1、conv2、relu2、conv3、relu3 层
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.relu2(x)
                x = self.conv3(x)
                x = self.relu3(x)
                return x

        model = M()

        # 将模型中指定的层进行融合，生成融合后的模型 fused_model1 和深度复制 fused_model2
        fused_model1 = fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3']])
        fused_model2 = copy.deepcopy(fused_model1)

        # 对融合后的模型进行通道均衡化，具体是对 conv1 和 conv2，以及 conv2 和 conv3 之间的卷积核进行均衡化
        _equalize.equalize(fused_model1, [['conv1', 'conv2'], ['conv2', 'conv3']], 1e-6)

        # 获取均衡化后的模型中的 conv1、conv2、conv3 层的权重
        conv1 = self.getModule(fused_model1, 'conv1')[0]
        conv2 = self.getModule(fused_model1, 'conv2')[0]
        conv3 = self.getModule(fused_model1, 'conv3')[0]

        # 检查 conv1 和 conv2 以及 conv2 和 conv3 之间的通道是否均衡化
        self.checkChannelsEqualized(conv1.weight, conv2.weight, 0, 1)
        self.checkChannelsEqualized(conv2.weight, conv3.weight, 0, 1)

        # 创建输入张量，检查融合后的模型与未均衡化的副本以及原始模型在相同输入下的输出是否相等
        input = torch.randn(3, 3, 1, 1)
        self.assertEqual(fused_model1(input), fused_model2(input))
        self.assertEqual(fused_model1(input), model(input))
        def test_equalize_fused_linearrelu(self):
            ''' Checks to see if eager mode equalization supports fused
            LinearReLU models

            A model with 3 LinearReLU is constructed. Next, the linear and relu
            layers are fused together and adjacent linear layers have cross-layer
            equalization applied. Finally, we ensure that the channels have been
            equalized and that the equalized and unequalized versions of the model
            yield the same output given the same input
            '''
            # 定义一个简单的模型类，包含三个 LinearReLU 结构的网络层
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(3, 4)
                    self.relu1 = nn.ReLU(inplace=False).to(dtype=torch.float)
                    self.linear2 = nn.Linear(4, 5)
                    self.relu2 = nn.ReLU(inplace=False).to(dtype=torch.float)
                    self.linear3 = nn.Linear(5, 6)
                    self.relu3 = nn.ReLU(inplace=False).to(dtype=torch.float)

                def forward(self, x):
                    # 模型前向传播过程，依次通过线性层和ReLU激活层
                    x = self.linear1(x)
                    x = self.relu1(x)
                    x = self.linear2(x)
                    x = self.relu2(x)
                    x = self.linear3(x)
                    x = self.relu3(x)
                    return x

            # 创建模型实例
            model = M()

            # 融合模型中指定的层，形成新的融合模型
            fused_model1 = fuse_modules(model, [['linear1', 'relu1'], ['linear2', 'relu2'], ['linear3', 'relu3']])
            # 深度复制融合后的模型，用于后续对比
            fused_model2 = copy.deepcopy(fused_model1)

            # 对融合后的模型进行跨层均衡化处理
            _equalize.equalize(fused_model1, [['linear1', 'linear2'], ['linear2', 'linear3']], 1e-6)

            # 获取融合后模型中的各线性层
            linear1 = self.getModule(fused_model1, 'linear1')[0]
            linear2 = self.getModule(fused_model1, 'linear2')[0]
            linear3 = self.getModule(fused_model1, 'linear3')[0]

            # 检查线性层的通道是否均衡化
            self.checkChannelsEqualized(linear1.weight, linear2.weight, 0, 1)
            self.checkChannelsEqualized(linear2.weight, linear3.weight, 0, 1)

            # 创建随机输入
            input = torch.randn(20, 3)
            # 确保均衡化后的模型和未均衡化的模型以及原始模型在相同输入下输出相同
            self.assertEqual(fused_model1(input), fused_model2(input))
            self.assertEqual(fused_model1(input), model(input))
```