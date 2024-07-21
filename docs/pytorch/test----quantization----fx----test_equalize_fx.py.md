# `.\pytorch\test\quantization\fx\test_equalize_fx.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口
import torch.ao.nn.intrinsic.quantized as nniq  # 导入 PyTorch AO 中的量化内部操作模块
import torch.ao.nn.quantized as nnq  # 导入 PyTorch AO 中的量化模块
from torch.ao.quantization import default_qconfig  # 导入默认的量化配置
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver  # 导入量化观察器类
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx  # 导入量化函数
from torch.ao.quantization.fx._equalize import (  # 导入平衡化相关的函数和配置
    _InputEqualizationObserver,
    _WeightEqualizationObserver,
    calculate_equalization_scale,
    default_equalization_qconfig,
    _convert_equalization_ref,
    get_layer_sqnr_dict,
    get_equalization_qconfig_dict,
)

from torch.testing._internal.common_quantization import (  # 导入量化测试的通用模型和工具
    NodeSpec as ns,
    QuantizationTestCase,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    LinearAddModel,
    SingleLayerFunctionalLinearModel,
    TwoLayerFunctionalLinearModel,
    FunctionalLinearAddModel,
    ConvModel,
    TwoLayerConvModel,
    SingleLayerFunctionalConvModel,
    TwoLayerFunctionalConvModel,
    skipIfNoFBGEMM,
    LinearReluModel,
    LinearReluLinearModel,
    LinearReluAddModel,
    FunctionalLinearReluModel,
    FunctionalLinearReluLinearModel,
    ConvReluModel,
    ConvReluConvModel,
    ConvReluAddModel,
    FunctionalConvReluModel,
    FunctionalConvReluConvModel,
)

# Standard Libraries
import copy  # 导入拷贝库
import numpy as np  # 导入 NumPy 库

# Testing utils
from hypothesis import given  # 导入 hypothesis 中的 given 函数
from hypothesis import strategies as st  # 导入 hypothesis 中的策略模块


default_qconfig_dict = {"": default_qconfig}  # 默认量化配置字典

specific_qconfig_dict = {  # 特定对象量化配置字典，包含不同对象类型和对应的量化配置
    "": None,
    "object_type": [
        (nn.Linear, default_qconfig),
        (F.linear, default_qconfig),
        (nn.ReLU, default_qconfig),
        (F.relu, default_qconfig),
        (nn.Conv2d, default_qconfig),
        (F.conv2d, default_qconfig)
    ]
}

default_equalization_qconfig_dict = {  # 默认的平衡化配置字典
    "": None,
    "object_type": [
        (nn.Linear, default_equalization_qconfig),
        (F.linear, default_equalization_qconfig),
        (nn.ReLU, default_equalization_qconfig),
        (F.relu, default_equalization_qconfig),
        (nn.Conv2d, default_equalization_qconfig),
        (F.conv2d, default_equalization_qconfig)
    ]
}


class TestEqualizeFx(QuantizationTestCase):
    def channel_minmax(self, input, axis=1):
        ''' Finds the min/max of inputs associated with a specific channel
        '''
        size_of_tensor_dim = input.ndim  # 获取张量的维度数
        axis_list = list(range(size_of_tensor_dim))  # 创建维度索引列表
        axis_list.remove(axis)  # 移除指定轴
        axis_list.sort(reverse=True)  # 逆序排序轴列表

        mins = input.copy()  # 拷贝输入张量作为最小值张量
        maxs = input.copy()  # 拷贝输入张量作为最大值张量
        for a in axis_list:  # 遍历轴列表
            mins = mins.min(a)  # 沿指定轴计算最小值
            maxs = maxs.max(a)  # 沿指定轴计算最大值

        return (mins, maxs)  # 返回最小值和最大值元组
    # 使用 `@given` 装饰器定义一个参数化测试的测试用例生成函数，使用了多个参数。
    @given(ndim=st.sampled_from((2, 3, 4, 5)),  # 参数 `ndim` 可以取 2、3、4、5 中的一个值
           input_qdtype=st.sampled_from((torch.qint8, torch.quint8)),  # 参数 `input_qdtype` 可以取 torch.qint8 或 torch.quint8 中的一个值
           input_qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),  # 参数 `input_qscheme` 可以取 torch.per_tensor_affine 或 torch.per_tensor_symmetric 中的一个值
           weight_qdtype=st.sampled_from((torch.qint8, torch.quint8)),  # 参数 `weight_qdtype` 可以取 torch.qint8 或 torch.quint8 中的一个值
           weight_qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric,  # 参数 `weight_qscheme` 可以取 torch.per_channel_affine、torch.per_channel_symmetric、torch.per_channel_affine_float_qparams 中的一个值
                                           torch.per_channel_affine_float_qparams)))
    def test_input_weight_equalization_prepare(self):
        """ Tests that graphs created after prepare_fx is as expected
        """

        # 定义单层神经网络模型节点发生次数字典
        single_nn_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,
            ns.call_module(MinMaxObserver): 2,
        }

        # 定义两层神经网络模型节点发生次数字典
        two_nn_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 3,
        }

        # 定义单层函数层模型节点发生次数字典
        single_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,
            ns.call_module(_WeightEqualizationObserver): 1,
            ns.call_module(MinMaxObserver): 3,
        }

        # 定义两层函数层模型节点发生次数字典
        two_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(_WeightEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 5,
        }

        # 定义函数层和卷积层模型节点发生次数字典
        fp_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(_WeightEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 6,
        }

        # 测试用例列表，包含不同模型和预期节点发生次数的元组
        tests = [(SingleLayerLinearModel, single_nn_layer_node_occurrence),
                 (TwoLayerLinearModel, two_nn_layer_node_occurrence),
                 (TwoLayerFunctionalLinearModel, two_F_layer_node_occurrence),
                 (FunctionalLinearAddModel, fp_F_layer_node_occurrence),
                 (LinearReluModel, single_nn_layer_node_occurrence),
                 (LinearReluLinearModel, two_nn_layer_node_occurrence),
                 (FunctionalLinearReluModel, single_F_layer_node_occurrence),
                 (FunctionalLinearReluLinearModel, two_F_layer_node_occurrence),
                 (ConvModel, single_nn_layer_node_occurrence),
                 (TwoLayerConvModel, two_nn_layer_node_occurrence),
                 (TwoLayerFunctionalConvModel, two_F_layer_node_occurrence),
                 (ConvReluModel, single_nn_layer_node_occurrence),
                 (ConvReluConvModel, two_nn_layer_node_occurrence),
                 (FunctionalConvReluModel, single_F_layer_node_occurrence),
                 (FunctionalConvReluConvModel, two_F_layer_node_occurrence)]

        # 遍历测试用例列表
        for (M, node_occurrence) in tests:
            # 实例化模型并设为评估模式
            m = M().eval()
            # 获取模型的示例输入
            example_inputs = m.get_example_inputs()
            # 准备模型以进行量化，返回准备后的图模块
            prepared = prepare_fx(
                m,
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            # 检查图模块的节点以确认预期的节点发生次数
            self.checkGraphModuleNodes(prepared, expected_node_occurrence=node_occurrence)
    def test_input_weight_equalization_branching(self):
        """ Tests that graphs containing branches are prepared correctly.
        Specifically, equalization observers should not be inserted in front of
        branches in which both initial layers in the branches plan to be
        quantized.
        """

        # Tests that we do not add an equalization observer due to both initial
        # nodes in the branch containing layers that need to be equalized.
        # Note that this should print out 2 warning messages for not being able
        # to equalize layers linear1 and linear1 because it is part of a branch
        class TestBranchingWithoutEqualizationModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Define two linear layers, each with input size 5 and output size 5
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)

            def forward(self, x):
                # Apply the first linear layer to input x
                y = self.linear1(x)
                # Apply the second linear layer to input x
                z = self.linear2(x)
                # Return the element-wise sum of y and z
                return torch.add(y, z)

        # Define expected node occurrence for certain observer types
        no_eq_branching_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 0,  # Expect 0 instances of _InputEqualizationObserver
            ns.call_module(MinMaxObserver): 3,  # Expect 3 instances of MinMaxObserver
        }

        # Instantiate the test model and prepare it for evaluation
        m = TestBranchingWithoutEqualizationModel().eval()
        example_inputs = (torch.rand(1, 5),)
        # Prepare the model for quantization with specific configurations and default equalization settings
        prepared = prepare_fx(
            m, specific_qconfig_dict, example_inputs=example_inputs,
            _equalization_config=default_equalization_qconfig_dict)
        # Check if the prepared model's node occurrences match the expected ones
        self.checkGraphModuleNodes(prepared, expected_node_occurrence=no_eq_branching_node_occurrence)

        # Tests that we will add an equalization observer because there is only
        # one initial node in the branch that needs to be equalized
        class TestBranchingWithEqualizationModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Define a linear layer with input size 5 and output size 5
                self.linear1 = nn.Linear(5, 5)

            def forward(self, x):
                # Apply the linear layer to input x
                y = self.linear1(x)
                # Add a constant value of 5 element-wise to x
                z = torch.add(x, 5)
                # Return the element-wise sum of y and z
                return torch.add(y, z)

        # Define expected node occurrence for certain observer types
        eq_branching_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,  # Expect 1 instance of _InputEqualizationObserver
            ns.call_module(MinMaxObserver): 2,  # Expect 2 instances of MinMaxObserver
        }

        # Instantiate the test model and prepare it for evaluation
        m = TestBranchingWithEqualizationModel().eval()
        example_inputs = (torch.randn(1, 5),)
        # Prepare the model for quantization with specific configurations and default equalization settings
        prepared = prepare_fx(
            m, specific_qconfig_dict, example_inputs=example_inputs,
            _equalization_config=default_equalization_qconfig_dict)
        # Check if the prepared model's node occurrences match the expected ones
        self.checkGraphModuleNodes(prepared, expected_node_occurrence=eq_branching_node_occurrence)

    @skipIfNoFBGEMM
    def test_input_weight_equalization_convert(self):
        """ Tests that the modified model for equalization (before quantization)
        returns the same output as the original model
        """
        # 定义测试用例列表，每个元素包含模型类和维度
        tests = [(SingleLayerLinearModel, 2), (LinearAddModel, 2), (TwoLayerLinearModel, 2),
                 (SingleLayerFunctionalLinearModel, 2), (FunctionalLinearAddModel, 2),
                 (TwoLayerFunctionalLinearModel, 2),
                 (LinearReluModel, 2), (LinearReluLinearModel, 2), (LinearReluAddModel, 2),
                 (FunctionalLinearReluModel, 2), (FunctionalLinearReluLinearModel, 2),
                 (ConvModel, 4), (TwoLayerConvModel, 4), (SingleLayerFunctionalConvModel, 4),
                 (TwoLayerFunctionalConvModel, 4),
                 (ConvReluModel, 4), (ConvReluConvModel, 4), (ConvReluAddModel, 4),
                 (FunctionalConvReluModel, 4), (FunctionalConvReluConvModel, 4)]

        # 遍历每个测试用例
        for (M, ndim) in tests:
            # 创建模型实例并设置为评估模式
            m = M().eval()

            # 根据不同的维度生成随机输入张量
            if ndim == 2:
                x = torch.rand((5, 5))
            elif ndim == 4:
                x = torch.rand((16, 3, 224, 224))

            example_inputs = (x,)
            # 深拷贝模型并进行准备，使用特定的量化配置字典和示例输入
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict
            )
            # 获取预测输出
            output = prepared(x)

            # 获取参考的均衡化后转换
            convert_ref = _convert_equalization_ref(prepared)
            # 计算参考均衡化后转换的输出
            convert_ref_output = convert_ref(x)

            # 重新准备原始模型
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            # 对原始模型进行预测
            prepared(x)
            # 转换为 FX 格式，用于编译检查
            convert_fx(prepared)  # Check if compile
            # 断言预测输出与参考转换输出相等
            self.assertEqual(output, convert_ref_output)

    def calculate_equalization_scale_ref(self, x, w):
        """ Calculates the equalization scale based on the input and weight
        """
        # 计算输入张量 x 的最小值和最大值
        min_inputs = x.min(axis=0)
        max_inputs = x.max(axis=0)

        # 计算权重矩阵 w 每列的最小值和最大值
        min_weights_col = w.min(axis=0)
        max_weights_col = w.max(axis=0)

        # 计算均衡化比例尺度
        equalization_scale = np.sqrt((max_weights_col - min_weights_col) /
                                     (max_inputs - min_inputs))
        return equalization_scale

    def get_expected_eq_scales(self, model, x):
        """ For each module in the graph, we want to calculate the equalization
        scale at that point. This only works for models containing single or
        connected linear layers.
        """
        # 存储预期的均衡化比例尺度列表
        exp_eq_scales = []
        # 遍历模型的每个子模块
        for _, module in model.named_children():
            # 获取权重和偏置，并转换为 numpy 数组
            weight = module.weight.detach().numpy()
            bias = module.bias.detach().numpy()

            # 计算当前模块的均衡化比例尺度
            eq_scale = self.calculate_equalization_scale_ref(x, weight)
            # 将结果添加到预期均衡化比例尺度列表中
            exp_eq_scales.append(eq_scale)

            # 更新输入 x，模拟前向传播
            x = x @ weight.T + bias

        return exp_eq_scales
    # 定义一个测试方法，用于测试输入权重均衡后的均衡比例是否符合预期
    def test_input_weight_equalization_equalization_scales(self):
        """ After applying the equalization functions, check if the equalization
        scales are the expected values
        """
        
        # 定义要测试的模型列表
        tests = [SingleLayerLinearModel, TwoLayerLinearModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        # 生成一个 5x5 的随机张量作为输入
        x = torch.rand((5, 5))
        
        # 遍历每个测试模型
        for M in tests:
            # 创建模型实例，并设为评估模式
            m = M().eval()
            
            # 获取预期的均衡比例
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())

            # 准备模型以进行量化
            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            
            # 执行模型准备
            prepared(*example_inputs)
            
            # 转换均衡的参考值
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)

            # 计数器初始化为0
            counter = 0
            
            # 遍历转换后的图中的每个节点
            for node in convert_ref.graph.nodes:
                # 如果节点的名称中包含 'equalization_scale'，且操作为 'get_attr'
                if 'equalization_scale' in node.name and node.op == 'get_attr':
                    # 检查均衡比例是否与预期相符
                    self.assertEqual(convert_ref.get_buffer(str(node.target)).reshape(-1), exp_eq_scales[counter])
                    counter += 1

    # 定义一个方法，用于计算预期的权重和偏置值
    def get_expected_weights_bias(self, model, x, exp_eq_scales):
        """ For each module in the graph, we want to calculate the expected
        scaled weight and bias values. This only works for models containing
        single or connected linear layers.
        """
        
        # 初始化预期权重和偏置列表
        exp_weights = []
        exp_bias = []
        
        # 遍历模型的每个子模块
        for i, (_, module) in enumerate(model.named_children()):
            # 获取模块的权重和偏置，并转为 NumPy 数组
            weight = module.weight.detach().numpy()
            bias = module.bias.detach().numpy()

            # 根据均衡比例对权重进行缩放
            scaled_weight = weight * np.reciprocal(exp_eq_scales[i])
            scaled_bias = bias
            
            # 如果不是最后一层，进一步缩放权重和偏置
            if i + 1 < len(exp_eq_scales):
                scaled_weight = (scaled_weight.T * exp_eq_scales[i + 1]).T
                scaled_bias = (scaled_bias.T * exp_eq_scales[i + 1]).T

            # 将缩放后的权重和偏置添加到预期列表中
            exp_weights.append(scaled_weight)
            exp_bias.append(scaled_bias)

            # 更新输入，用于下一层的计算
            x = x @ weight.T + bias

        # 返回预期的权重和偏置列表
        return exp_weights, exp_bias
    # 定义一个测试方法，用于检查在应用均衡化函数后，权重和偏置是否符合预期
    def test_input_weight_equalization_weights_bias(self):
        """ After applying the equalization functions check if the weights and
        biases are as expected
        """

        # 定义要测试的模型类列表
        tests = [SingleLayerLinearModel, TwoLayerLinearModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        # 生成一个 5x5 的随机张量作为输入数据
        x = torch.rand((5, 5))
        for M in tests:
            # 实例化一个模型并设置为评估模式
            m = M().eval()
            # 获取预期的均衡化比例尺度
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())
            # 获取预期的权重和偏置
            exp_weights, exp_bias = self.get_expected_weights_bias(m, x.detach().numpy(), exp_eq_scales)

            # 准备模型进行均衡化
            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(x)
            # 将均衡化后的模型转换为参考模型
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)

            # 获取转换后模型中的所有模块
            modules = dict(convert_ref.named_modules(remove_duplicate=False))
            counter = 0
            # 遍历转换后模型的所有节点
            for node in convert_ref.graph.nodes:
                # 如果节点是调用模块并且目标模块是 nn.Linear 类型
                if node.op == 'call_module' and isinstance(modules[str(node.target)], nn.Linear):
                    # 断言模块的权重与预期的权重相等
                    self.assertEqual(modules[str(node.target)].weight, exp_weights[counter])
                    # 断言模块的偏置与预期的偏置相等
                    self.assertEqual(modules[str(node.target)].bias, exp_bias[counter])
                    counter += 1

    # 根据模型、输入、预期的均衡化比例尺度、预期的权重和偏置计算预期的输入激活值范围
    def get_expected_inp_act_vals(self, model, x, exp_eq_scales, exp_weights, exp_bias):
        """ For each module in the graph, we want to calculate the expected
        min/max values for every input activation node. This only works for
        models containing only single or connected linear layers.
        """
        # 根据均衡化比例尺度调整输入数据
        x = x * exp_eq_scales[0]

        # 初始化存储预期输入激活值范围的列表
        exp_inp_activation_vals = []
        for i, _ in enumerate(model.named_children()):
            # 计算每个输入激活节点的预期最小和最大值，并存储到列表中
            exp_inp_activation_vals.append((x.min(), x.max()))
            x = x @ exp_weights[i].T + exp_bias[i]

        # 添加最终输出层的预期输入激活值范围到列表中
        exp_inp_activation_vals.append((x.min(), x.max()))
        return exp_inp_activation_vals

    # 根据预期的权重计算预期的权重激活值范围
    def get_expected_weight_act_vals(self, exp_weights):
        """ For each module in the graph, we want to calculate the expected
        min/max values for every weight activation node. This is assuming that
        the weight observers are all MinMaxObservers.
        """

        # 初始化存储预期权重激活值范围的列表
        exp_weight_activation_vals = []
        for w in exp_weights:
            # 计算每个权重激活节点的预期最小和最大值，并存储到列表中
            exp_weight_activation_vals.append((w.min(), w.max()))

        return exp_weight_activation_vals
    def test_input_weight_equalization_activation_values(self):
        """
        After applying the equalization functions check if the input
        observer's min/max values are as expected
        """
        
        # 测试用例列表，包含不同的模型类
        tests = [SingleLayerLinearModel, TwoLayerLinearModel, SingleLayerFunctionalLinearModel]

        # 创建一个随机张量作为输入
        x = torch.rand((5, 5))
        
        # 设置随机种子为0，保证可复现性
        torch.manual_seed(0)
        
        # 遍历每个模型类
        for M in tests:
            # 创建模型实例，并设置为评估模式
            m = M().eval()
            
            # 获取期望的均衡因子
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())
            
            # 获取期望的权重和偏置
            exp_weights, exp_bias = self.get_expected_weights_bias(m, x.detach().numpy(), exp_eq_scales)
            
            # 获取期望的输入激活值
            exp_inp_act_vals = self.get_expected_inp_act_vals(m, x, exp_eq_scales, exp_weights, exp_bias)
            
            # 获取期望的权重激活值
            exp_weight_act_vals = self.get_expected_weight_act_vals(exp_weights)
            
            # 准备模型用于量化
            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(x)
            
            # 转换为均衡化参考模型
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)
            
            # 构建模块字典以便于查找
            modules = dict(convert_ref.named_modules(remove_duplicate=False))
            
            # 初始化输入和权重计数器
            inp_counter = 0
            weight_counter = 0
            
            # 遍历转换后的图中的节点
            for node in convert_ref.graph.nodes:
                users = list(node.users)
                
                # 检查节点是否为调用模块，并且模块是MinMaxObserver类型
                if node.op == 'call_module' and isinstance(modules[str(node.target)], MinMaxObserver):
                    # 检查节点是否只有一个用户，并且用户是torch.nn.functional.linear函数，且参数匹配
                    if len(users) == 1 and users[0].target == torch.nn.functional.linear and users[0].args[1] == node:
                        # 检查权重激活层的最小/最大值
                        exp_min_val, exp_max_val = exp_weight_act_vals[weight_counter]
                        self.assertEqual(modules[str(node.target)].min_val, exp_min_val)
                        self.assertEqual(modules[str(node.target)].max_val, exp_max_val)
                        weight_counter += 1
                    else:
                        # 检查输入激活层的最小/最大值
                        exp_min_val, exp_max_val = exp_inp_act_vals[inp_counter]
                        self.assertEqual(modules[str(node.target)].min_val, exp_min_val)
                        self.assertEqual(modules[str(node.target)].max_val, exp_max_val)
                        inp_counter += 1
    # 定义一个方法，用于检查原始模型和均衡化模型的图结构是否相同
    def check_orig_and_eq_graphs(self, orig_model, eq_model):
        """ Given a non-equalized model and an equalized model, check that the
        graphs are structured in the same way, except the equalized model has
        additional 'equalization_scale' and 'mul' nodes.
        """
        # 初始化原始模型节点索引、节点列表和命名模块字典
        orig_idx = 0
        orig_nodes = list(orig_model.graph.nodes)
        orig_modules = dict(orig_model.named_modules(remove_duplicate=False))

        # 初始化均衡化模型节点索引、节点列表和命名模块字典
        eq_idx = 0
        eq_nodes = list(eq_model.graph.nodes)
        eq_modules = dict(eq_model.named_modules(remove_duplicate=False))

        # 遍历原始模型和均衡化模型的节点列表
        while orig_idx < len(orig_nodes) and eq_idx < len(eq_nodes):
            # 如果当前均衡化模型节点包含 'equalization_scale' 和 'mul'，则跳过这两个节点
            if 'equalization_scale' in eq_nodes[eq_idx].name and 'mul' in eq_nodes[eq_idx + 1].name:
                eq_idx += 2
                continue
            # 如果原始模型节点和均衡化模型节点的操作类型不同，返回 False
            elif orig_nodes[orig_idx].op != eq_nodes[eq_idx].op:
                return False
            # 如果原始模型节点和均衡化模型节点的操作类型为 'call_module'
            elif orig_nodes[orig_idx].op == 'call_module':
                # 检查 call_module 类型是否相同（例如 nn.Linear, MinMaxObserver）
                orig_node = orig_nodes[orig_idx]
                eq_node = eq_nodes[eq_idx]
                if type(orig_modules[orig_node.target]) is not type(eq_modules[eq_node.target]):
                    return False
            # 如果原始模型节点和均衡化模型节点的操作类型为 'call_function'
            elif orig_nodes[orig_idx].op == 'call_function':
                # 检查 call_function 是否相同（例如 F.linear）
                orig_node = orig_nodes[orig_idx]
                eq_node = eq_nodes[eq_idx]
                if orig_node.target != eq_node.target:
                    return False

            # 均衡化模型节点索引和原始模型节点索引均加一
            eq_idx += 1
            orig_idx += 1

        # 若所有比较都通过，则返回 True
        return True

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_input_weight_equalization_results(self):
        """ Tests that for small models, the results of quantized models that
        have been equalized are very close to models that have not been equalized.
        """

        # 定义测试用例，包括不同的小模型
        tests = [SingleLayerLinearModel, TwoLayerLinearModel, LinearAddModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        # 创建一个随机的输入张量
        x = torch.rand((5, 5))
        for M in tests:
            m = M().eval()

            # 无均衡化的情况
            example_inputs = (x,)

            # 准备模型以进行量化仿真，使用指定的量化配置字典和示例输入
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config={})
            prepared(x)

            # 将准备好的模型转换为量化模型，检查是否能够编译
            quantized = convert_fx(prepared)
            quantized_output = quantized(x)

            # 带有均衡化的情况
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict
            )
            prepared(x)

            # 将均衡化后的模型转换为量化模型，检查是否能够编译
            equalized_and_quantized = convert_fx(prepared)
            equalized_and_quantized_output = equalized_and_quantized(x)

            # 断言量化输出与均衡化并量化输出在给定误差下相等
            self.assertEqual(quantized_output, equalized_and_quantized_output, rtol=1e-5, atol=0.1)

    @skipIfNoFBGEMM


这段代码是一个用于测试输入权重均衡化结果的测试函数。它包含了对多个小型模型进行测试，分别比较了经过均衡化和未经均衡化的量化模型的输出结果是否非常接近。
    def test_selective_equalization(self):
        """ Tests that we are able to run numeric suite on the equalized model
        and construct a valid equalization_config equalizing only the top
        4 layers with the highest quantization errors.
        """
        
        # 设置随机种子为1，确保结果可复现
        torch.manual_seed(1)

        # 定义一个简单的神经网络模型
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bot = torch.nn.Sequential(torch.nn.Linear(5, 5))  # 底部神经网络层
                self.top = torch.nn.Sequential(torch.nn.Linear(5, 5))  # 顶部神经网络层

            def forward(self, x):
                x = self.bot(x)  # 应用底部层
                x = torch.add(x, 5)  # 加上常数5
                x = self.top(x)  # 应用顶部层
                return x

        # 创建并评估浮点数模型
        float_model = M().eval()

        # 硬编码以确保顶层具有更高的量化误差
        x = torch.tensor([[0.0642, 0.7824, 0.4255, 0.7106, 0.5957],
                          [0.8373, 0.8851, 0.8229, 0.0212, 0.8987],
                          [0.9077, 0.7538, 0.4530, 0.5772, 0.1376],
                          [0.0690, 0.9002, 0.7998, 0.2768, 0.8985],
                          [0.0282, 0.5068, 0.6725, 0.1829, 0.5480]])

        # 对浮点数模型进行量化
        example_inputs = (x,)
        prepared_model = prepare_fx(
            copy.deepcopy(float_model),
            specific_qconfig_dict,
            example_inputs=example_inputs
        )
        prepared_model(x)
        quantized_model = convert_fx(copy.deepcopy(prepared_model))

        # 获取浮点数模型和量化模型之间的SQNR（信号失真噪声比）
        layer_to_sqnr_dict = get_layer_sqnr_dict(copy.deepcopy(prepared_model), quantized_model, x)

        # 构建equalization_qconfig_dict，以均衡具有最高量化误差的层
        selective_equalization_qconfig_dict = get_equalization_qconfig_dict(layer_to_sqnr_dict, 1)

        # 创建选择性均衡的模型
        prepared_model = prepare_fx(
            copy.deepcopy(float_model),
            specific_qconfig_dict,
            example_inputs=example_inputs,
            _equalization_config=selective_equalization_qconfig_dict,
        )
        prepared_model(x)
        equalized_model = convert_fx(prepared_model)

        # 预定义的节点列表，用于检查图中节点的顺序
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.add),
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        # 检查图中节点的顺序是否符合预期
        self.checkGraphModuleNodes(equalized_model, expected_node_list=node_list)
```