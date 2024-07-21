# `.\pytorch\test\ao\sparsity\test_structured_sparsifier.py`

```py
# 导入必要的模块和库
import copy  # 导入深拷贝函数
import logging  # 导入日志记录模块
import random  # 导入随机数生成模块

import torch  # 导入PyTorch深度学习框架
from torch import nn  # 导入神经网络模块
from torch.ao.pruning._experimental.pruner import (  # 导入实验性修剪器模块
    BaseStructuredSparsifier,  # 基础结构稀疏化修剪器
    FakeStructuredSparsity,  # 虚拟结构稀疏性
    FPGMPruner,  # FPGM修剪器
    LSTMSaliencyPruner,  # LSTM显著性修剪器
    SaliencyPruner,  # 显著性修剪器
)
from torch.nn.utils import parametrize  # 导入参数化工具函数
from torch.testing._internal.common_pruning import (  # 导入内部常用修剪测试模块
    Conv2dActivation,  # 二维卷积激活层
    Conv2dBias,  # 二维卷积偏置
    Conv2dPadBias,  # 带填充的二维卷积偏置
    Conv2dPool,  # 二维卷积池化
    Conv2dPoolFlatten,  # 二维卷积池化展平
    Conv2dPoolFlattenFunctional,  # 二维卷积池化展平函数式
    LinearActivation,  # 线性激活层
    LinearActivationFunctional,  # 线性激活函数式
    LinearBias,  # 线性偏置
    LSTMLayerNormLinearModel,  # LSTM层归一化线性模型
    LSTMLinearModel,  # LSTM线性模型
    rows_are_subset,  # 行是子集
    SimpleConv2d,  # 简单二维卷积
    SimpleLinear,  # 简单线性层
)

from torch.testing._internal.common_utils import skipIfTorchDynamo, TestCase  # 导入测试工具和测试用例类

# 配置日志格式和级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# 定义设备列表，包括CPU和可能的CUDA设备
DEVICES = {
    torch.device("cpu"),  # CPU设备
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),  # 如果CUDA可用，则选择CUDA设备，否则选择CPU设备
}


class SimplePruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        # 更新掩码，将第二个位置的掩码值设置为False
        getattr(module.parametrizations, tensor_name)[0].mask[1] = False


class ImplementedPruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        """Prunes 1/3 of the weight output channels, so resulting module has 33.3% pruning"""
        # 对权重输出通道的1/3进行修剪，使得模块具有33.3%的修剪效果
        num_rows = len(module.parametrizations[tensor_name][0].mask)
        prune = random.sample(list(range(num_rows)), num_rows // 3)
        module.parametrizations[tensor_name][0].mask[prune] = False


class BottomHalfLSTMPruner(BaseStructuredSparsifier):
    """
    Pruner that will remove the bottom half of the rows.
    This is primarily meant for testing purposes
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # 移除行的下半部分的修剪器，主要用于测试目的
        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = p.mask
                masks = torch.split(mask, len(mask) // 4)
                for small in masks:
                    num = len(small)
                    small[num // 2 :] = False
                new_mask = torch.cat(masks)
                mask.data = new_mask.data


class TestSaliencyPruner(TestCase):
    pass  # 测试显著性修剪器的测试用例
    # 定义测试函数，测试 SaliencyPruner 类的 update_mask 方法
    def test_saliency_pruner_update_mask(self):
        """Test that we prune out the row with the lowest saliency (first row)"""
        # 创建一个简单的线性模型
        model = SimpleLinear()
        # 使用 torch.no_grad() 上下文管理器，设置模型的第一个线性层的权重
        with torch.no_grad():
            model.linear1.weight = nn.Parameter(
                torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
            )
        
        # 设置剪枝配置，指定要剪枝的张量和稀疏度水平
        pruning_config = [{"tensor_fqn": "linear1.weight", "sparsity_level": 0.5}]
        # 创建 SaliencyPruner 实例
        pruner = SaliencyPruner({})

        # 准备模型和剪枝配置
        pruner.prepare(model, pruning_config)
        # 启用更新掩码的选项
        pruner.enable_mask_update = True
        # 执行一步更新
        pruner.step()
        # 进行剪枝操作
        pruned_model = pruner.prune()

        # 期望的剪枝后的权重
        expected = torch.Tensor([[3, 3, 3, 3], [4, 4, 4, 4]])
        # 获取实际剪枝后的权重
        pruned = pruned_model.linear1.weight

        # 断言剪枝后的权重与期望的形状和数值接近
        assert expected.shape == pruned.shape
        assert torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all()

    # 定义测试函数，测试 LSTM 模型的 SaliencyPruner 类的 update_mask 方法
    def test_lstm_saliency_pruner_update_mask(self):
        # 创建一个具有 LSTM 层的线性模型
        model = LSTMLinearModel(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=1,
        )

        # 手动设置 LSTM 层的权重
        manual_weights = torch.Tensor(
            [[1, 1], [2, 2], [2, 2], [1, 1], [-1, -1], [-2, -2], [-2, -2], [-1, -1]]
        )

        # 使用 torch.no_grad() 上下文管理器，设置 LSTM 层的权重和偏置
        with torch.no_grad():
            model.lstm.weight_ih_l0 = nn.Parameter(manual_weights)
            model.lstm.weight_hh_l0 = nn.Parameter(torch.Tensor(manual_weights))
            model.lstm.bias_ih_l0 = nn.Parameter(manual_weights[:, 0])
            model.lstm.bias_hh_l0 = nn.Parameter(manual_weights[:, 0])

        # 设置剪枝配置，指定要剪枝的张量
        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]
        # 创建 LSTMSaliencyPruner 实例
        fx_pruner = LSTMSaliencyPruner({"sparsity_level": 0.5})
        # 准备模型和剪枝配置
        fx_pruner.prepare(model, config)
        # 启用更新掩码的选项
        fx_pruner.enable_mask_update = True
        # 执行一步更新
        fx_pruner.step()

        # 设置模型为评估模式
        model.eval()
        # 进行剪枝操作
        pruned_model = fx_pruner.prune()
        # 设置剪枝后模型为评估模式
        pruned_model.eval()

        # 确保原模型和剪枝后模型都能运行
        lstm_input = torch.ones((1, 2))
        model(lstm_input)
        pruned_model(lstm_input)

        # 断言最低显著性行已被剪枝
        expected = torch.Tensor([[2, 2], [2, 2], [-2, -2], [-2, -2]])
        pruned = model.lstm.weight_ih_l0
        assert expected.shape == pruned.shape
        assert torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all()

        expected = torch.Tensor([[2], [2], [-2], [-2]])
        pruned = model.lstm.weight_hh_l0
        assert expected.shape == pruned.shape
        assert torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all()

        expected = torch.Tensor([2, 2, -2, -2])
        for pruned in [model.lstm.bias_ih_l0, model.lstm.bias_hh_l0]:
            assert expected.shape == pruned.shape
            assert torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all()
# 定义一个名为 TestBaseStructuredSparsifier 的测试类，继承自 TestCase
class TestBaseStructuredSparsifier(TestCase):

    # 检查修剪器在执行前是否已准备就绪
    def _check_pruner_prepared(self, model, pruner, device):
        # 遍历修剪器的组配置
        for config in pruner.groups:
            # 获取当前组的模块
            module = config["module"]
            # 断言模块的权重张量所在设备类型与指定设备类型相同
            assert module.weight.device.type == device.type
            # 检查状态字典中是否包含掩码
            assert config["tensor_fqn"] in pruner.state
            # 检查模块是否被参数化，并且是否具有 parametrizations 属性
            assert parametrize.is_parametrized(module)
            assert hasattr(module, "parametrizations")
            # 假设这是第一个/唯一的参数化
            assert type(module.parametrizations.weight[0]) == FakeStructuredSparsity

    # 检查修剪器在执行步骤前的有效性
    def _check_pruner_valid_before_step(self, model, pruner, device):
        # 遍历修剪器的组配置
        for config in pruner.groups:
            modules = []
            # 如果模块是元组，则扩展模块列表
            if type(config["module"]) is tuple:
                modules.extend(config["module"])
            else:
                module = config["module"]
                modules.append(module)
            # 遍历每个模块
            for module in modules:
                # 断言模块的权重张量所在设备类型与指定设备类型相同
                assert module.weight.device.type == device.type
                # 断言权重参数化掩码的数据类型为 torch.bool
                assert module.parametrizations.weight[0].mask.dtype == torch.bool

    # 检查修剪器在执行步骤后的有效性
    def _check_pruner_valid_after_step(self, model, pruner, mask, device):
        # 遍历修剪器的组配置
        for config in pruner.groups:
            modules = []
            # 如果模块是元组，则扩展模块列表
            if type(config["module"]) is tuple:
                modules.extend(config["module"])
            else:
                module = config["module"]
                modules.append(module)
            # 遍历每个模块
            for module in modules:
                # 断言模块的权重张量所在设备类型与指定设备类型相同
                assert module.weight.device.type == device.type
                # 计算权重参数化掩码的总元素数
                total = module.parametrizations.weight[0].mask.numel()
                # 断言非零掩码的数量等于总元素数减去指定的 mask 数量
                assert (
                    module.parametrizations.weight[0].mask.count_nonzero()
                    == total - mask
                )

    # 在特定设备上测试构造函数
    def _test_constructor_on_device(self, model, device):
        # 断言在构造 BaseStructuredSparsifier 时会抛出 TypeError 异常，且异常信息包含 "BaseStructuredSparsifier.*update_mask"
        self.assertRaisesRegex(
            TypeError,
            "BaseStructuredSparsifier.*update_mask",
            BaseStructuredSparsifier,
        )
        # 在指定设备上深度复制模型
        model1 = copy.deepcopy(model).to(device)
        # 创建 SimplePruner 实例
        pruner = SimplePruner(None)
        # 准备修剪器和模型
        pruner.prepare(model1, None)
        # 启用掩码更新
        pruner.enable_mask_update = True
        # 遍历修剪器的组配置
        for g in pruner.groups:
            # 获取当前组的模块
            module = g["module"]
            # 断言模块的权重张量所在设备类型与指定设备类型相同
            assert module.weight.device.type == device.type
        # 断言修剪器组的数量为 5
        assert len(pruner.groups) == 5
        # 执行一步修剪
        pruner.step()
        # 在设备上可以实例化带有配置的模型
        model2 = copy.deepcopy(model).to(device)
        # 创建带有配置的 SimplePruner 实例
        pruner = SimplePruner({"test": 3})
        # 准备修剪器和模型
        pruner.prepare(model2, [{"tensor_fqn": "seq.0.weight"}])
        # 断言修剪器组的数量为 1
        assert len(pruner.groups) == 1
        # 断言修剪器组的模块全名为 "seq.0"
        assert pruner.groups[0]["module_fqn"] == "seq.0"
        # 断言修剪器组包含 "test" 字段，并且其值为 3
        assert "test" in pruner.groups[0]
        assert pruner.groups[0]["test"] == 3

    # 测试构造函数
    def test_constructor(self):
        # 创建 SimpleLinear 模型实例
        model = SimpleLinear()
        # 遍历不同的设备
        for device in DEVICES:
            # 在设备上测试构造函数
            self._test_constructor_on_device(model, torch.device(device))
    # 在给定设备上准备线性模型，并进行测试准备过程
    def _test_prepare_linear_on_device(self, model, device):
        # 深拷贝模型并移动到指定设备
        model = copy.deepcopy(model).to(device)
        # 创建输入张量，全为1，形状为(128, 7)，在指定设备上
        x = torch.ones(128, 7, device=device)
        # 创建简单的修剪器对象
        pruner = SimplePruner(None)
        # 准备模型和修剪器，未指定额外配置
        pruner.prepare(model, None)
        # 检查模型和修剪器是否准备就绪
        self._check_pruner_prepared(model, pruner, device)
        # 断言模型在给定输入下的输出形状是否符合预期
        assert model(x).shape == (128, 10)

    # 测试不同线性模型在各设备上的准备过程
    def test_prepare_linear(self):
        # 待测试的线性模型列表，包括无偏置和有偏置的模型
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]
        # 遍历所有设备
        for device in DEVICES:
            # 遍历所有模型
            for model in models:
                # 在指定设备上进行线性模型准备测试
                self._test_prepare_linear_on_device(model, torch.device(device))

    # 在给定设备上准备二维卷积模型，并进行测试准备过程
    def _test_prepare_conv2d_on_device(self, model, expected_shape, config, device):
        # 创建输入张量，全为1，形状为(1, 1, 28, 28)，在指定设备上
        x = torch.ones((1, 1, 28, 28), device=device)
        # 创建简单的修剪器对象
        pruner = SimplePruner(None)
        # 准备模型和修剪器，使用给定配置
        pruner.prepare(model, config)
        # 检查模型和修剪器是否准备就绪
        self._check_pruner_prepared(model, pruner, device)
        # 断言模型在给定输入下的输出形状是否符合预期
        assert model(x).shape == expected_shape

    # 测试不同二维卷积模型在各设备上的准备过程
    def test_prepare_conv2d(self):
        # 待测试的二维卷积模型列表
        models = [
            SimpleConv2d(),
            Conv2dBias(),
            Conv2dActivation(),
            Conv2dPadBias(),
            Conv2dPool(),
        ]
        # 每个模型对应的预期输出形状
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        # 针对每个设备和模型形状配置进行测试
        configs = [None, None, None, None, None]
        for device in DEVICES:
            for model, shape, config in zip(models, shapes, configs):
                # 在指定设备上进行二维卷积模型准备测试
                model = model.to(device)
                self._test_prepare_conv2d_on_device(
                    model, shape, config, torch.device(device)
                )

    # 在给定设备上执行线性模型的步骤，并进行测试
    def _test_step_linear_on_device(self, model, device):
        # 将模型移动到指定设备
        model = model.to(device)
        # 创建输入张量，全为1，形状为(7, 7)，在指定设备上
        x = torch.ones(7, 7, device=device)
        # 创建简单的修剪器对象
        pruner = SimplePruner(None)
        # 准备模型和修剪器，未指定额外配置
        pruner.prepare(model, None)
        # 启用修剪掩码更新
        pruner.enable_mask_update = True
        # 检查步骤前模型和修剪器的有效性
        self._check_pruner_valid_before_step(model, pruner, device)
        # 执行修剪步骤
        pruner.step()
        # 检查步骤后模型和修剪器的有效性，预期步数为1
        self._check_pruner_valid_after_step(model, pruner, 1, device)

    # 测试不同线性模型在各设备上的步骤执行过程
    def test_step_linear(self):
        # 待测试的线性模型列表
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]
        # 遍历所有设备
        for device in DEVICES:
            # 遍历所有模型
            for model in models:
                # 在指定设备上进行线性模型步骤测试
                self._test_step_linear_on_device(model, torch.device(device))

    # 在给定设备上执行二维卷积模型的步骤，并进行测试
    def _test_step_conv2d_on_device(self, model, expected_shape, config, device):
        # 将模型移动到指定设备
        model = model.to(device)
        # 创建输入张量，全为1，形状为(1, 1, 28, 28)，在指定设备上
        x = torch.ones((1, 1, 28, 28), device=device)
        # 创建简单的修剪器对象
        pruner = SimplePruner(None)
        # 准备模型和修剪器，使用给定配置
        pruner.prepare(model, config)
        # 启用修剪掩码更新
        pruner.enable_mask_update = True
        # 检查步骤前模型和修剪器的有效性
        self._check_pruner_valid_before_step(model, pruner, device)
        # 执行修剪步骤
        pruner.step()
        # 检查步骤后模型和修剪器的有效性，预期步数为1
        self._check_pruner_valid_after_step(model, pruner, 1, device)
        # 断言模型在给定输入下的输出形状是否符合预期
        assert model(x).shape == expected_shape
    # 如果 TorchDynamo 无法正常工作，则跳过此测试
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_step_conv2d(self):
        # 定义一组测试模型
        models = [
            SimpleConv2d(),        # 使用 SimpleConv2d 类创建模型
            Conv2dBias(),          # 使用 Conv2dBias 类创建模型
            Conv2dActivation(),    # 使用 Conv2dActivation 类创建模型
            Conv2dPadBias(),       # 使用 Conv2dPadBias 类创建模型
            Conv2dPool(),          # 使用 Conv2dPool 类创建模型
        ]
        # 每个模型对应的输入形状
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        # 每个模型对应的配置
        configs = [None, None, None, None, None]
        # 遍历设备列表进行测试
        for device in DEVICES:
            # 遍历模型、输入形状和配置，并调用 _test_step_conv2d_on_device 方法进行测试
            for model, shape, config in zip(models, shapes, configs):
                self._test_step_conv2d_on_device(
                    model, shape, config, torch.device(device)
                )
    
    # 检查模型是否在剪枝前准备好
    def _check_pruner_prepared(self, model, pruner, device):
        # 遍历剪枝器的组配置
        for config in pruner.groups:
            module = config["module"]
            # 断言模块没有 parametrizations 属性
            assert not hasattr(module, "parametrizations")
            # 断言模块没有 mask 属性
            assert not hasattr(module, "mask")
    
    # 在指定设备上测试线性模型
    def _test_linear_on_device(
        self, model, config, expected_shape, device, also_prune_bias
    ):
        # 将模型移动到指定设备上
        model = model.to(device)
        # 设置模型为评估模式
        model.eval()
        # 计算模型参数的总数
        num_original_params = sum(p.numel() for p in model.parameters())
        # 创建输入张量 x，全为 1
        x = torch.ones(128, 7, device=device)
    
        # 创建一个实现了的剪枝器对象
        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        # 准备剪枝器，传入模型和配置
        pruner.prepare(model, config)
        # 启用 mask 更新
        pruner.enable_mask_update = True
        # 执行剪枝器的一步操作
        pruner.step()
    
        # 计算模型对输入 x 的预期输出
        y_expected = model(x)
    
        # 断言预期输出的形状符合预期
        assert y_expected.shape == (128, 10)
        # 检查模型在剪枝前的准备情况
        self._check_pruner_prepared(model, pruner, device)
    
        # 执行剪枝操作
        pruned = pruner.prune()
        # 计算剪枝后模型对输入 x 的输出
        y_pruned = pruned(x)
        # 计算剪枝后模型的参数总数
        num_pruned_params = sum(p.numel() for p in pruned.parameters())
    
        # 断言剪枝后输出的形状符合预期形状
        assert y_pruned.shape == expected_shape
        # 检查模型在剪枝后的状态
        self._check_pruner_pruned(model, pruner, device)
        # 如果剪枝后输出形状与预期形状相同，则比较预期输出和剪枝后输出的接近程度
        if y_pruned.shape == y_expected.shape:
            assert torch.isclose(y_expected, y_pruned, rtol=1e-05, atol=1e-07).all()
            # 断言剪枝后的参数数量少于剪枝前的参数数量
            assert num_pruned_params < num_original_params
    def test_prune_linear_linear(self):
        r"""test pruning linear-> linear modules"""
        # 初始化空列表存储配置和形状
        configs, shapes = [], []
        # 添加第一个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        # 添加第二个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "linear1.weight"},
            ]
        )
        shapes.append((128, 10))

        # 添加第三个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))
        
        # 遍历设备列表和是否剪枝偏置的标志
        for device in DEVICES:
            for also_prune_bias in [True, False]:
                # 遍历配置和形状，执行线性层测试
                for config, shape in zip(configs, shapes):
                    self._test_linear_on_device(
                        SimpleLinear(),
                        config,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_linear_bias_linear(self):
        # linear(bias) -> linear(no bias)
        # 初始化空列表存储配置和形状
        configs, shapes = [], []
        # 添加第一个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(bias) -> linear(bias)
        # 添加第二个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.3.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(no bias) -> linear(bias)
        # 添加第三个配置和形状
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        # 遍历设备列表和是否剪枝偏置的标志
        for device in DEVICES:
            for also_prune_bias in [True, False]:
                # 遍历配置和形状，执行线性层带偏置测试
                for config, shape in zip(configs, shapes):
                    self._test_linear_on_device(
                        LinearBias(),
                        config,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )
    # 定义测试方法，用于验证线性激活后的线性层裁剪
    def test_prune_linear_activation_linear(self):
        # 配置裁剪的参数，包括各个线性层的权重参数的完整路径
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.2.weight"},
            {"tensor_fqn": "seq.4.weight"},
            {"tensor_fqn": "linear1.weight"},
        ]
        # 定义线性层输出形状
        shape = (128, 10)

        # 遍历设备列表进行测试
        for device in DEVICES:
            # 遍历是否同时裁剪偏置的选项进行测试
            for also_prune_bias in [True, False]:
                # 测试使用 nn.Modules 的版本
                self._test_linear_on_device(
                    LinearActivation(),  # 使用线性激活的线性层
                    config,              # 裁剪配置
                    shape,               # 输出形状
                    torch.device(device),  # 当前设备
                    also_prune_bias,     # 是否同时裁剪偏置
                )
                # 测试使用函数式版本的线性激活
                self._test_linear_on_device(
                    LinearActivationFunctional(),  # 使用函数式线性激活
                    config,                        # 裁剪配置
                    shape,                         # 输出形状
                    torch.device(device),          # 当前设备
                    also_prune_bias,               # 是否同时裁剪偏置
                )

    # 定义测试卷积层在特定设备上的方法
    def _test_conv2d_on_device(
        self, model, config, x, expected_shape, device, also_prune_bias
    ):
        # 将模型移动到指定设备上
        model = model.to(device)
        # 计算模型原始参数数量
        num_original_params = sum(p.numel() for p in model.parameters())
        # 将模型设置为评估模式
        model.eval()

        # 创建裁剪器对象，并进行模型准备与掩码更新
        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()

        # 计算模型在输入 x 上的预期输出
        y_expected = model(x)
        assert y_expected.shape == expected_shape

        # 检查裁剪器是否成功准备好模型
        self._check_pruner_prepared(model, pruner, device)

        # 执行融合步骤，返回裁剪后的模型
        pruned = pruner.prune()
        y_pruned = pruned(x)
        # 计算裁剪后的模型参数数量
        num_pruned_params = sum(p.numel() for p in pruned.parameters())

        # 断言裁剪后的模型输出形状与预期形状相同
        assert y_pruned.shape == expected_shape
        # 检查裁剪器是否成功裁剪模型
        self._check_pruner_pruned(model, pruner, device)
        # 如果裁剪后的输出与预期输出形状相同，则进行数值比较
        if y_pruned.shape == y_expected.shape:
            # TODO 这个 rtol 可能有点高，需要确认是否有特定原因导致失败
            # 断言裁剪后的输出与预期输出在数值上相近
            assert torch.isclose(
                y_expected,
                y_pruned,
                rtol=1e-3,
                atol=1e-3,
            ).all(), f"fail for {type(model)}"
            # 当所有层都有填充且无法裁剪时，断言裁剪后的参数数量不大于原始参数数量
            assert num_pruned_params <= num_original_params
    def test_prune_conv2d_conv2d(self):
        # 初始化空列表用于存储配置和形状信息
        configs, shapes = [], []
        # 在顺序块内部添加配置
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        # 添加形状信息
        shapes.append((1, 52, 20, 20))
        
        # 在顺序块间进行修剪
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        # 添加形状信息
        shapes.append((1, 52, 20, 20))

        # 对于每个设备进行迭代测试
        for device in DEVICES:
            # 创建一个形状为 (1, 1, 28, 28) 的张量 x，并将其移动到指定的设备上
            x = torch.ones((1, 1, 28, 28), device=device)
            # 对于每个 also_prune_bias 进行迭代
            for also_prune_bias in [True, False]:
                # 对于每个配置和形状的组合进行测试
                for config, shape in zip(configs, shapes):
                    # 调用 _test_conv2d_on_device 方法，传入 SimpleConv2d 类的实例，
                    # 配置信息 config，输入张量 x，期望的输出形状 shape，
                    # 设备信息 torch.device(device)，以及是否也修剪偏置 also_prune_bias
                    self._test_conv2d_on_device(
                        SimpleConv2d(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_conv2d_bias_conv2d(self):
        # Conv2d 带有偏置且无激活函数
        configs, shapes = [], []
        
        # conv2d（带偏置） -> conv2d（带偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        # 添加形状信息
        shapes.append((1, 52, 18, 18))

        # conv2d（无偏置） -> conv2d（带偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        # 添加形状信息
        shapes.append((1, 52, 18, 18))

        # conv2d（带偏置） -> conv2d（无偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        # 添加形状信息
        shapes.append((1, 52, 18, 18))

        # 对于每个设备进行迭代测试
        for device in DEVICES:
            # 创建一个形状为 (1, 1, 28, 28) 的张量 x，并将其移动到指定的设备上
            x = torch.ones((1, 1, 28, 28), device=device)
            # 对于每个 also_prune_bias 进行迭代
            for also_prune_bias in [True, False]:
                # 对于每个配置和形状的组合进行测试
                for config, shape in zip(configs, shapes):
                    # 调用 _test_conv2d_on_device 方法，传入 Conv2dBias 类的实例，
                    # 配置信息 config，输入张量 x，期望的输出形状 shape，
                    # 设备信息 torch.device(device)，以及是否也修剪偏置 also_prune_bias
                    self._test_conv2d_on_device(
                        Conv2dBias(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )
    def test_prune_conv2d_activation_conv2d(self):
        # 定义一个测试方法，用于测试带激活函数的卷积层组合
        # Conv2d with Activation and no Bias
        # 使用空列表存储配置和形状信息
        configs, shapes = [], []

        # conv2d(no bias) -> activation -> conv2d(no bias)
        # 第一种配置：卷积层（无偏置） -> 激活函数 -> 卷积层（无偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activation -> conv2d(bias)
        # 第二种配置：卷积层（有偏置） -> 激活函数 -> 卷积层（有偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activation -> conv2d(no bias)
        # 第三种配置：卷积层（有偏置） -> 激活函数 -> 卷积层（无偏置）
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(no bias) -> activation -> conv2d(bias)
        # 第四种配置：卷积层（无偏置） -> 激活函数 -> 卷积层（有偏置）
        configs.append(
            [
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # 针对不同设备进行测试
        for device in DEVICES:
            # 创建一个输入张量
            x = torch.ones((1, 1, 28, 28), device=device)
            # 针对是否同时修剪偏置的选项进行测试
            for also_prune_bias in [True, False]:
                # 遍历每种配置和形状，执行卷积层测试
                for config, shape in zip(configs, shapes):
                    self._test_conv2d_on_device(
                        Conv2dActivation(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )
    # 测试带有填充层的 Conv2d
    def test_prune_conv2d_padding_conv2d(self):
        # Conv2d with Padded layers after Bias layers
        # 存储不同配置和形状的列表
        configs, shapes = [], []

        # conv(padded, bias) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(no bias, no pad) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(padded, bias) -> conv ( no bias ,no pad)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(pad, bias) -> conv(no pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.6.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(no pad, bias) -> conv(pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.8.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # 在不同设备上进行测试
        for device in DEVICES:
            # 创建一个输入张量
            x = torch.ones((1, 1, 28, 28), device=device)
            # 循环测试不同的配置和形状
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    # 调用内部函数 _test_conv2d_on_device 进行 Conv2d 测试
                    self._test_conv2d_on_device(
                        Conv2dPadBias(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    # 测试带有池化层的 Conv2d
    def test_prune_conv2d_pool_conv2d(self):
        # Conv2d with Pooling layers
        # 定义单个配置和形状
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        shape = (1, 52, 3, 3)

        # 在不同设备上进行测试
        for device in DEVICES:
            # 创建一个输入张量
            x = torch.ones((1, 1, 28, 28), device=device)
            # 循环测试是否同时修剪偏置
            for also_prune_bias in [True, False]:
                # 调用内部函数 _test_conv2d_on_device 进行 Conv2d 测试
                self._test_conv2d_on_device(
                    Conv2dPool(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_complex_conv2d(self):
        """Test fusion for models that contain Conv2d & Linear modules.
        Currently supports: Conv2d-Pool2d-Flatten-Linear, Skip-add"""
        
        # 定义配置列表，指定需要测试的模型权重名称
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        
        # 定义输入数据的形状
        shape = (1, 13)

        # 遍历设备列表，测试每个设备上的模型行为
        for device in DEVICES:
            # 创建输入张量 x，全为1，形状为(1, 1, 28, 28)，指定设备为当前遍历的设备
            x = torch.ones((1, 1, 28, 28), device=device)
            
            # 对每种 also_prune_bias 取值进行测试
            for also_prune_bias in [True, False]:
                # 调用 _test_conv2d_on_device 方法，测试 Conv2dPoolFlattenFunctional 模型
                self._test_conv2d_on_device(
                    Conv2dPoolFlattenFunctional(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )
                
                # 调用 _test_conv2d_on_device 方法，测试 Conv2dPoolFlatten 模型
                self._test_conv2d_on_device(
                    Conv2dPoolFlatten(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_lstm_linear_multiple_layer(self):
        """
        Test fusion support for LSTM(multi-layer) -> Linear
        """
        
        # 创建 LSTMLinearModel 模型实例，指定输入维度、隐藏层维度、输出维度和层数
        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=2,
        )

        # 定义配置列表，指定需要测试的 LSTM 模型的权重名称
        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
            {"tensor_fqn": "lstm.weight_ih_l1"},
            {"tensor_fqn": "lstm.weight_hh_l1"},
        ]

        # 创建输入数据 lstm_input，全为1，形状为(1, 8)
        lstm_input = torch.ones((1, 8))
        
        # 创建 BottomHalfLSTMPruner 实例，指定稀疏性水平为 0.5，并准备对模型和配置进行修剪
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)

        # 启用掩码更新
        fx_pruner.enable_mask_update = True
        
        # 执行修剪步骤
        fx_pruner.step()

        # 将模型设置为评估模式，并传入输入数据 lstm_input 进行前向计算
        model.eval()
        _, _ = model(lstm_input)
        
        # 获取修剪后的模型
        pruned_model = fx_pruner.prune()
        
        # 将修剪后的模型设置为评估模式，并传入输入数据 lstm_input 进行前向计算
        pruned_model.eval()
        _, _ = pruned_model(lstm_input)

        # 获取模型的预期参数字典
        expected_params = dict(model.named_parameters())
        
        # 遍历模型的每个参数，确保其名称存在于预期参数字典中
        for name, param in model.named_parameters():
            assert name in expected_params
            
            # 由于零元素可能会影响数值比较，这里通过子集关系验证修剪后的权重是否是旧模型权重的子集
            assert rows_are_subset(param, expected_params[name])
            
            # 删除预期参数字典中已验证的参数名称
            del expected_params[name]

        # 断言预期参数字典中不再存在任何键
        assert len(expected_params) == 0
    def test_prune_lstm_linear_single_layer(self):
        """
        Test fusion support for LSTM (single-layer) -> Linear
        """
        # 创建单层 LSTM -> Linear 的模型，指定输入维度为8，隐藏层维度为8，输出维度为8，层数为1
        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        )

        # 定义需要剪枝的参数配置，这里包括 LSTM 的权重
        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]

        # 创建输入张量，全为1
        lstm_input = torch.ones((1, 8))
        
        # 创建 BottomHalfLSTMPruner 实例，设定稀疏度为0.5
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        # 准备剪枝，传入模型和参数配置
        fx_pruner.prepare(model, config)
        # 允许更新掩码
        fx_pruner.enable_mask_update = True
        # 执行剪枝步骤
        fx_pruner.step()
        # 将模型设置为评估模式
        model.eval()

        # 获取模型的预期输出和 LSTM 层的预期输出
        out_expected, lstm_out_expected = model(lstm_input)
        # 执行剪枝操作，得到剪枝后的模型
        pruned_model = fx_pruner.prune()
        # 将剪枝后的模型设置为评估模式
        pruned_model.eval()
        # 获取剪枝后的模型的输出和 LSTM 层的输出
        out_pruned, lstm_out_pruned = pruned_model(lstm_input)
        r, c = lstm_out_expected.size()

        # 由于存在零值和缺失元素导致数值结果不同，无法直接比较 y_expected 和 y_pruned
        # 因此，我们检查剪枝后的输出的前半部分是否与预期输出的前半部分接近
        assert torch.isclose(
            lstm_out_expected[:, : c // 2], lstm_out_pruned, rtol=1e-05, atol=1e-07
        ).all()
        # 同时检查线性层的输出形状是否相同，这表示我们已正确调整了线性层的列数
        assert out_expected.shape == out_pruned.shape

    def test_prune_lstm_layernorm_linear_multiple_layer(self):
        """
        Test fusion support for LSTM(multi-layer) -> Linear
        """
        # 创建多层 LSTM -> Linear 的模型，指定输入维度为8，隐藏层维度为8，输出维度为8，层数为2
        model = LSTMLayerNormLinearModel(
            input_dim=8,
            output_dim=8,
            hidden_dim=8,
            num_layers=2,
        )

        # 定义需要剪枝的参数配置，这里包括两个 LSTM 层的权重
        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
            {"tensor_fqn": "lstm.weight_ih_l1"},
            {"tensor_fqn": "lstm.weight_hh_l1"},
        ]

        # 创建输入张量，全为1
        lstm_input = torch.ones((1, 8))
        
        # 创建 BottomHalfLSTMPruner 实例，设定稀疏度为0.5
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        # 准备剪枝，传入模型和参数配置
        fx_pruner.prepare(model, config)

        # 允许更新掩码
        fx_pruner.enable_mask_update = True
        # 执行剪枝步骤
        fx_pruner.step()

        # 将模型设置为评估模式
        model.eval()
        # 获取模型的输出，忽略中间变量
        _, _ = model(lstm_input)
        # 执行剪枝操作，得到剪枝后的模型
        pruned_model = fx_pruner.prune()
        # 将剪枝后的模型设置为评估模式
        pruned_model.eval()
        # 获取剪枝后的模型的输出，忽略中间变量
        _, _ = pruned_model(lstm_input)

        # 获取预期的参数字典
        expected_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            assert name in expected_params
            # 由于 0 元素可能影响数值结果，无法直接比较 y_expected 和 y_pruned
            # 因此，我们检查新 LSTM 的权重是否是旧 LSTM 权重的子集
            assert rows_are_subset(param, expected_params[name])
            del expected_params[name]

        # 确保没有删除任何键
        assert len(expected_params) == 0
    # 定义测试函数，用于测试 LSTM（单层）到线性层的融合支持
    def test_prune_lstm_layernorm_linear_single_layer(self):
        """
        Test fusion support for LSTM (single-layer) -> Linear
        """
        # 创建一个 LSTMLinearModel 实例，指定输入维度为8，隐藏层维度为8，输出维度为8，层数为1
        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        )

        # 配置文件列表，指定需要裁剪的张量的全限定名
        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]

        # 创建一个大小为 (1, 8) 的全1张量作为 LSTM 的输入
        lstm_input = torch.ones((1, 8))
        
        # 创建一个 BottomHalfLSTMPruner 实例，设置稀疏度水平为0.5，并准备裁剪模型和配置
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)
        
        # 启用掩码更新
        fx_pruner.enable_mask_update = True
        
        # 执行一步裁剪操作
        fx_pruner.step()
        
        # 将模型设置为评估模式
        model.eval()

        # 使用未裁剪的模型计算预期输出及 LSTM 输出
        out_expected, lstm_out_expected = model(lstm_input)
        
        # 执行裁剪操作，得到裁剪后的模型
        pruned_model = fx_pruner.prune()
        
        # 将裁剪后的模型设置为评估模式
        pruned_model.eval()
        
        # 使用裁剪后的模型计算输出及 LSTM 输出
        out_pruned, lstm_out_pruned = pruned_model(lstm_input)
        
        # 获取 LSTM 输出的行数和列数
        r, c = lstm_out_expected.size()

        # 由于裁剪后的输出可能包含零值或缺失元素，无法直接比较 y_expected == y_pruned
        # 因此使用 torch.isclose 进行数值接近性比较，检查裁剪后的结果的前半部分是否与未裁剪部分接近
        assert torch.isclose(
            lstm_out_expected[:, : c // 2], lstm_out_pruned, rtol=1e-05, atol=1e-07
        ).all()
        
        # 还要检查裁剪后的线性层输出形状是否与未裁剪时相同，以确保我们正确调整了线性层的列数
        assert out_expected.shape == out_pruned.shape
# 定义一个测试用例类 TestFPGMPruner，用于测试实现论文 `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration`
class TestFPGMPruner(TestCase):
    """
    Test case for the implementation of paper:
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.
    """

    # 定义一个简单的卷积神经网络模型 SimpleConvFPGM，继承自 nn.Module
    class SimpleConvFPGM(nn.Module):
        def __init__(self):
            super().__init__()
            # 第一个卷积层，输入通道数为1，输出通道数为3，卷积核大小为3x3，padding为1，无偏置
            self.conv2d1 = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False
            )
            # 手动设置滤波器权重，用于演示目的
            """
            Three filters' weight are manually set to values 3.0, 2.0, and 0.1.
            Different from the norm-based decision that prunes filter with value 0.1,
            FPGM will prune the one with value 2.0.
            """
            weights = torch.tensor([3.0, 2.0, 0.1])  # 每个滤波器的权重
            weights = weights[:, None, None, None]  # 进行广播操作
            # 将 conv2d1 的权重数据复制为每个滤波器的权重值
            self.conv2d1.weight.data.copy_(
                torch.ones(self.conv2d1.weight.shape) * weights
            )

            # 第二个卷积层
            self.conv2d2 = nn.Conv2d(
                in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False
            )
            weights = torch.tensor([6.0, 7.0, 0.4, 0.5])
            weights = weights[:, None, None, None]
            # 将 conv2d2 的权重数据复制为每个滤波器的权重值
            self.conv2d2.weight.data.copy_(
                torch.ones(self.conv2d2.weight.shape) * weights
            )

        # 前向传播函数
        def forward(self, x):
            x = self.conv2d1(x)  # 使用 conv2d1 进行卷积操作
            x = self.conv2d2(x)  # 使用 conv2d2 进行卷积操作
            return x
    def test_compute_distance(self, device="cpu"):
        """Test the distance computation function"""
        # 创建一个 SimpleConvFPGM 模型，并将其移动到指定设备上
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        # 创建一个 FPGMPruner 对象，设置剪枝阈值为 0.3
        pruner = FPGMPruner(0.3)
        # 使用 FPGMPruner 对象计算给定卷积层权重的距离
        dist_conv1 = pruner._compute_distance(model.conv2d1.weight)

        # 使用 torch.cdist 计算扁平化后的滤波器之间的距离矩阵
        flattened_filters = torch.Tensor(
            [
                [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                [2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000],
                [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
            ]
        )

        """
        期望的距离矩阵应该具有以下值:
            [0.0000, 3.0000, 8.7000],
            [3.0000, 0.0000, 5.7000],
            [8.7000, 5.7000, 0.0000],
        因此距离应该是:
            [11.7000, 8.7000, 14.4000]
        """
        # 使用 torch.cdist 计算期望的距离矩阵
        expected_dist_matrix_conv1 = torch.cdist(flattened_filters, flattened_filters, p=2)
        # 计算每行距离矩阵的绝对值之和，得到期望的距离值
        expected_dist_conv1 = torch.sum(torch.abs(expected_dist_matrix_conv1), 1)
        # 断言实际计算得到的距离与期望的距离非常接近
        assert torch.isclose(dist_conv1, expected_dist_conv1, rtol=1e-05, atol=1e-07).all()
    # 定义单层网络层蒂拉蒂
    def _test_update_mask_on_single_layer(self, expected_conv1, device):
        """Test that pruning is conducted based on the pair-wise distance measurement instead of absolute norm value"""
        # 使用简单的卷积网络模型进行测试
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        # 创建输入张量 x，全为1
        x = torch.ones((1, 1, 32, 32), device=device)
        # 创建 FPGMPruner 实例，并设置剪枝率为 0.3
        pruner = FPGMPruner(0.3)
        # 配置要剪枝的层信息
        config = [{"tensor_fqn": "conv2d1.weight"}]
        # 准备模型和剪枝配置
        pruner.prepare(model, config)
        # 启用掩码更新
        pruner.enable_mask_update = True
        # 执行一步剪枝操作
        pruner.step()
        # 断言：确保未剪枝掉最小范数的滤波器
        assert (
            pruner.groups[0]["module"].parametrizations.weight[0].mask[-1].item()
            is not False
        ), "do not prune the least-norm filter"

        # 合并剪枝步骤后的模型
        pruned_model = pruner.prune()

        # 对剪枝后的模型进行前向传播
        pruned_y = pruned_model(x)
        # 断言：检查输出张量形状
        expected_conv1 = expected_conv1.to(device)
        assert pruned_y.shape == (1, 4, 32, 32)
        # 断言：检查 conv2d1 权重的形状
        assert pruned_model.conv2d1.weight.shape == expected_conv1.shape
        # 断言：检查 conv2d2 权重的形状，确保输入通道被剪枝
        assert pruned_model.conv2d2.weight.shape == (
            4,
            2,
            3,
            3,
        ), "conv2d2 should have input channel pruned"
        # 断言：检查权重值是否近似相等
        assert torch.isclose(
            pruned_model.conv2d1.weight, expected_conv1, rtol=1e-05, atol=1e-07
        ).all()

    # 定义多层网络层蒂拉蒂
    def _test_update_mask_on_multiple_layer(
        self, expected_conv1, expected_conv2, device
    ):
        # 第二种设置
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        x = torch.ones((1, 1, 32, 32), device=device)
        pruner = FPGMPruner(0.3)
        # 配置要剪枝的层信息，包括剪枝稀疏度
        config = [
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight", "sparsity_level": 0.5},
        ]
        # 准备模型和剪枝配置
        pruner.prepare(model, config)
        # 启用掩码更新
        pruner.enable_mask_update = True
        # 执行一步剪枝操作
        pruner.step()
        # 获取两个最小范数滤波器的掩码
        mask1 = pruner.groups[0]["module"].parametrizations.weight[0].mask[-1]
        mask2 = pruner.groups[0]["module"].parametrizations.weight[0].mask[-2]
        # 断言：确保至少一个最小范数滤波器未被剪枝
        assert (
            mask1.item() is not False or mask2.item() is not False
        ), "Do not prune all least-norm filters"

        # 合并剪枝步骤后的模型
        pruned_model = pruner.prune()
        pruned_y = pruned_model(x)
        # 断言：检查输出张量形状
        expected_conv1 = expected_conv1.to(device)
        expected_conv2 = expected_conv2.to(device)
        assert pruned_y.shape == (1, 2, 32, 32)
        # 断言：检查 conv2d1 权重的形状
        assert pruned_model.conv2d1.weight.shape == expected_conv1.shape
        # 断言：检查 conv2d2 权重的形状
        assert pruned_model.conv2d2.weight.shape == expected_conv2.shape
        # 断言：检查权重值是否近似相等
        assert torch.isclose(
            pruned_model.conv2d1.weight, expected_conv1, rtol=1e-05, atol=1e-07
        ).all()
        assert torch.isclose(
            pruned_model.conv2d2.weight, expected_conv2, rtol=1e-05, atol=1e-07
        ).all()
    # 定义一个测试方法，用于测试更新掩码的功能
    def test_update_mask(self):
        # 创建一个张量，表示权重 [3.0, 0.1]
        weights = torch.tensor([3.0, 0.1])
        # 创建一个期望的卷积核1，形状为 (2, 1, 3, 3)，每个元素是对应权重的倍数
        expected_conv1 = torch.ones((2, 1, 3, 3)) * weights[:, None, None, None]

        # 更新权重张量为 [7.0, 0.4]
        weights = torch.tensor([7.0, 0.4])
        # 创建一个期望的卷积核2，形状为 (2, 2, 3, 3)，每个元素是对应权重的倍数
        expected_conv2 = torch.ones((2, 2, 3, 3)) * weights[:, None, None, None]

        # 遍历设备列表 DEVICES，对每个设备执行以下两个测试方法
        for device in DEVICES:
            # 在单个层上测试更新掩码，验证期望的卷积核1
            self._test_update_mask_on_single_layer(expected_conv1, device)
            # 在多层上测试更新掩码，验证期望的卷积核1和卷积核2
            self._test_update_mask_on_multiple_layer(
                expected_conv1, expected_conv2, device
            )
```