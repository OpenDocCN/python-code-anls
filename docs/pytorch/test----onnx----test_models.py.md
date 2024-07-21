# `.\pytorch\test\onnx\test_models.py`

```
# Owner(s): ["module: onnx"]

# 引入单元测试模块
import unittest

# 引入 PyTorch 测试通用模块
import pytorch_test_common

# 从模型定义中导入相关模型和常量
from model_defs.dcgan import _netD, _netG, bsz, imgsz, nz, weights_init
from model_defs.emb_seq import EmbeddingNetwork1, EmbeddingNetwork2
from model_defs.mnist import MNIST
from model_defs.op_test import ConcatNet, DummyNet, FakeQuantNet, PermuteNet, PReluNet
from model_defs.squeezenet import SqueezeNet
from model_defs.srresnet import SRResNet
from model_defs.super_resolution import SuperResolutionNet

# 从 PyTorch 测试通用模块中导入函数和装饰器
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion, skipScriptTest

# 从 torchvision.models 中导入模型
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet1_0
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18

# 从 verify 模块中导入 verify 函数
from verify import verify

# 导入 Torch 相关模块
import torch
from torch.ao import quantization
from torch.autograd import Variable
from torch.onnx import OperatorExportTypes
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack

# 检查是否支持 CUDA，根据支持情况定义 toC 函数
if torch.cuda.is_available():

    def toC(x):
        return x.cuda()

else:

    def toC(x):
        return x

# 定义批大小常量
BATCH_SIZE = 2

# 定义测试类 TestModels，继承自 pytorch_test_common.ExportTestCase
class TestModels(pytorch_test_common.ExportTestCase):
    opset_version = 9  # Caffe2 不支持默认的 opset 版本
    keep_initializers_as_inputs = False

    # 定义导出测试函数 exportTest，接受模型、输入、相对误差和绝对误差作为参数
    def exportTest(self, model, inputs, rtol=1e-2, atol=1e-7, **kwargs):
        # 导入 Caffe2 的 ONNX 后端
        import caffe2.python.onnx.backend as backend
        
        # 设置模型导出模式为 EVAL
        with torch.onnx.select_model_mode_for_export(
            model, torch.onnx.TrainingMode.EVAL
        ):
            # 使用 torch.onnx.utils._trace 对模型进行追踪
            graph = torch.onnx.utils._trace(model, inputs, OperatorExportTypes.ONNX)
            
            # 使用 torch._C._jit_pass_lint 对图进行静态分析
            torch._C._jit_pass_lint(graph)
            
            # 调用 verify 函数，验证模型导出结果
            verify(
                model,
                inputs,
                backend,
                rtol=rtol,
                atol=atol,
                opset_version=self.opset_version,
            )

    # 定义测试函数 test_ops，测试 DummyNet 模型的导出
    def test_ops(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(DummyNet()), toC(x))

    # 定义测试函数 test_prelu，测试 PReluNet 模型的导出
    def test_prelu(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(PReluNet(), x)

    # 使用 skipScriptTest 装饰器修饰的测试函数 test_concat，测试 ConcatNet 模型的导出
    @skipScriptTest()
    def test_concat(self):
        input_a = Variable(torch.randn(BATCH_SIZE, 3))
        input_b = Variable(torch.randn(BATCH_SIZE, 3))
        inputs = ((toC(input_a), toC(input_b)),)
        self.exportTest(toC(ConcatNet()), inputs)

    # 定义测试函数 test_permute，测试 PermuteNet 模型的导出
    def test_permute(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 10, 12))
        self.exportTest(PermuteNet(), x)

    # 使用 skipScriptTest 装饰器修饰的测试函数 test_concat 后续内容省略
    @skipScriptTest()
    # 定义测试方法，用于测试 EmbeddingNetwork1 模型的序列化导出功能
    def test_embedding_sequential_1(self):
        # 创建一个形状为 (BATCH_SIZE, 3) 的随机整数张量 x
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        # 调用 exportTest 方法，将 EmbeddingNetwork1 模型和 x 作为参数传入进行导出测试
        self.exportTest(EmbeddingNetwork1(), x)

    # 跳过脚本测试的装饰器，定义测试方法，用于测试 EmbeddingNetwork2 模型的序列化导出功能
    @skipScriptTest()
    def test_embedding_sequential_2(self):
        # 创建一个形状为 (BATCH_SIZE, 3) 的随机整数张量 x
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        # 调用 exportTest 方法，将 EmbeddingNetwork2 模型和 x 作为参数传入进行导出测试
        self.exportTest(EmbeddingNetwork2(), x)

    # 标记为跳过单元测试的装饰器，定义测试方法，用于测试 SRResNet 模型的序列化导出功能
    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        # 创建一个形状为 (1, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(1, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 SRResNet 模型和 x 作为参数传入进行导出测试
        self.exportTest(
            toC(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)), toC(x)
        )

    # 标记为跳过 LAPACK 不存在的装饰器，定义测试方法，用于测试 SuperResolutionNet 模型的序列化导出功能
    @skipIfNoLapack
    def test_super_resolution(self):
        # 创建一个形状为 (BATCH_SIZE, 1, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 SuperResolutionNet 模型和 x 作为参数传入进行导出测试，并设置绝对容差为 1e-6
        self.exportTest(toC(SuperResolutionNet(upscale_factor=3)), toC(x), atol=1e-6)

    # 定义测试方法，用于测试 AlexNet 模型的序列化导出功能
    def test_alexnet(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 alexnet 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(alexnet()), toC(x))

    # 定义测试方法，用于测试 MNIST 模型的序列化导出功能
    def test_mnist(self):
        # 创建一个形状为 (BATCH_SIZE, 1, 28, 28) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0))
        # 调用 exportTest 方法，将 MNIST 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(MNIST()), toC(x))

    # 标记为跳过单元测试的装饰器，定义测试方法，用于测试 VGG16 模型的序列化导出功能
    @unittest.skip("This model takes too much memory")
    def test_vgg16(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 vgg16 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(vgg16()), toC(x))

    # 标记为跳过单元测试的装饰器，定义测试方法，用于测试 VGG16_bn 模型的序列化导出功能
    @unittest.skip("This model takes too much memory")
    def test_vgg16_bn(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 vgg16_bn 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(vgg16_bn()), toC(x))

    # 标记为跳过单元测试的装饰器，定义测试方法，用于测试 VGG19 模型的序列化导出功能
    @unittest.skip("This model takes too much memory")
    def test_vgg19(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 vgg19 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(vgg19()), toC(x))

    # 标记为跳过单元测试的装饰器，定义测试方法，用于测试 VGG19_bn 模型的序列化导出功能
    @unittest.skip("This model takes too much memory")
    def test_vgg19_bn(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 vgg19_bn 模型和 x 作为参数传入进行导出测试
        self.exportTest(toC(vgg19_bn()), toC(x))

    # 定义测试方法，用于测试 ResNet50 模型的序列化导出功能
    def test_resnet(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 224, 224) 的随机正态分布张量 x，并填充为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 调用 exportTest 方法，将 resnet50 模型和 x 作为参数传入进行导出测试，并设置绝对容差为 1e-6
        self.exportTest(toC(resnet50()), toC(x), atol=1e-6)

    # 定义测试方法，用于测试 Inception v3 模型的序列化导出功能
    # 此测试不稳定，偶尔会出现单个元素的不匹配
    def test_inception(self):
        # 创建一个形状为 (BATCH_SIZE, 3, 299, 299) 的随机正态分布张量 x
        x = Variable(torch.randn(BATCH_SIZE, 3, 299, 299))
        # 调用 exportTest 方法，将 inception_v3 模型和 x 作为参数传入进行导出测试，并设置可接受误差百分比为 0.01
        self.exportTest(toC(inception_v3()), toC(x), acceptable_error_percentage=0.01)
    def test_squeezenet(self):
        # 测试 SqueezeNet：具有与 AlexNet 相当的准确率，但参数少 50 倍，模型大小小于 0.5MB
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 使用 SqueezeNet 版本 1.1 初始化模型
        sqnet_v1_0 = SqueezeNet(version=1.1)
        # 将模型和输入数据导出进行测试
        self.exportTest(toC(sqnet_v1_0), toC(x))

        # SqueezeNet 1.1 的计算量比 SqueezeNet 1.0 小 2.4 倍，参数略少
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 使用 SqueezeNet 版本 1.1 初始化模型
        sqnet_v1_1 = SqueezeNet(version=1.1)
        # 将模型和输入数据导出进行测试
        self.exportTest(toC(sqnet_v1_1), toC(x))

    def test_densenet(self):
        # 测试 DenseNet-121 模型
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 使用 DenseNet-121 初始化模型
        self.exportTest(toC(densenet121()), toC(x), rtol=1e-2, atol=1e-5)

    @skipScriptTest()
    def test_dcgan_netD(self):
        # 测试 DCGAN 中的 Discriminator 网络
        netD = _netD(1)
        netD.apply(weights_init)
        # 创建随机输入数据
        input = Variable(torch.empty(bsz, 3, imgsz, imgsz).normal_(0, 1))
        # 将网络和输入数据导出进行测试
        self.exportTest(toC(netD), toC(input))

    @skipScriptTest()
    def test_dcgan_netG(self):
        # 测试 DCGAN 中的 Generator 网络
        netG = _netG(1)
        netG.apply(weights_init)
        # 创建随机输入数据
        input = Variable(torch.empty(bsz, nz, 1, 1).normal_(0, 1))
        # 将网络和输入数据导出进行测试
        self.exportTest(toC(netG), toC(input))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quant(self):
        # 测试 FakeQuantNet 模型
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 将 FakeQuantNet 模型和输入数据导出进行测试
        self.exportTest(toC(FakeQuantNet()), toC(x))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_resnet_pertensor(self):
        # 测试量化 ResNet50 模型
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        qat_resnet50 = resnet50()

        # 使用 per tensor 方式量化权重，per channel 支持将在 opset 13 中引入
        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_fake_quant,
        )
        # 准备模型进行量化训练
        quantization.prepare_qat(qat_resnet50, inplace=True)
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        _ = qat_resnet50(x)
        # 计算每个 FakeQuantize 模块的量化参数
        for module in qat_resnet50.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        # 将量化后的模型和输入数据导出进行测试
        self.exportTest(toC(qat_resnet50), toC(x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_resnet_per_channel(self):
        # 量化 ResNet50 模型
        # 创建输入张量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        # 实例化一个 ResNet50 模型
        qat_resnet50 = resnet50()

        # 配置量化参数，使用默认的激活量化函数和逐通道权重量化函数
        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        # 准备 QAT（Quantization Aware Training）模型，原地操作
        quantization.prepare_qat(qat_resnet50, inplace=True)
        # 启用观察者
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        # 启用假量化
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        # 对输入 x 进行前向传播，获取输出，不保存结果
        _ = qat_resnet50(x)
        # 遍历 QAT 模型中的每个模块
        for module in qat_resnet50.modules():
            # 如果当前模块是 FakeQuantize 模块
            if isinstance(module, quantization.FakeQuantize):
                # 计算量化参数
                module.calculate_qparams()
        # 禁用观察者
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        # 导出测试，将 QAT ResNet50 模型和输入 x 转换为 C 代码
        self.exportTest(toC(qat_resnet50), toC(x))

    @skipScriptTest(skip_before_opset_version=15, reason="None type in outputs")
    def test_googlenet(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 GoogLeNet 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(googlenet()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mnasnet(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 MnasNet1.0 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(mnasnet1_0()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mobilenet(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 MobileNetV2 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(mobilenet_v2()), toC(x), rtol=1e-3, atol=1e-5)

    @skipScriptTest()  # prim_data
    def test_shufflenet(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 ShuffleNetV2_x1.0 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(shufflenet_v2_x1_0()), toC(x), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_fcn(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 FCN-ResNet101 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(
            toC(fcn_resnet101(weights=None, weights_backbone=None)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_deeplab(self):
        # 创建输入变量 x，形状为 BATCH_SIZE * 3 * 224 * 224，填充值为 1.0
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        # 导出测试，将 DeepLabV3-ResNet101 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(
            toC(deeplabv3_resnet101(weights=None, weights_backbone=None)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_r3d_18_video(self):
        # 创建输入变量 x，形状为 1 * 3 * 4 * 112 * 112，填充值为 1.0
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        # 导出测试，将 R3D-18 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(r3d_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mc3_18_video(self):
        # 创建输入变量 x，形状为 1 * 3 * 4 * 112 * 112，填充值为 1.0
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        # 导出测试，将 MC3-18 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(mc3_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_r2plus1d_18_video(self):
        # 创建输入变量 x，形状为 1 * 3 * 4 * 112 * 112，填充值为 1.0
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        # 导出测试，将 R(2+1)D-18 模型和输入 x 转换为 C 代码，设置相对和绝对容差
        self.exportTest(toC(r2plus1d_18()), toC(x), rtol=1e-3, atol=1e-5)
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于执行测试用例
    common_utils.run_tests()
```