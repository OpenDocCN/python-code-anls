# `.\pytorch\test\ao\sparsity\test_activation_sparsifier.py`

```
# 导入所需的库和模块
import copy  # 导入copy模块，用于深拷贝对象
import logging  # 导入logging模块，用于日志记录
from typing import List  # 导入List类型提示，用于类型注解

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口
from torch.ao.pruning._experimental.activation_sparsifier.activation_sparsifier import (
    ActivationSparsifier,  # 从特定路径导入ActivationSparsifier类
)
from torch.ao.pruning.sparsifier.utils import module_to_fqn  # 从sparsifier.utils模块导入module_to_fqn函数
from torch.testing._internal.common_utils import skipIfTorchDynamo, TestCase  # 导入测试相关的辅助函数和类

# 配置日志记录格式和级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义神经网络的各个层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 第一个卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # 第二个卷积层
        self.identity1 = nn.Identity()  # 标识层1，不做任何操作
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层1，池化核为2x2

        self.linear1 = nn.Linear(4608, 128)  # 第一个全连接层，输入维度4608，输出维度128
        self.identity2 = nn.Identity()  # 标识层2，不做任何操作
        self.linear2 = nn.Linear(128, 10)  # 第二个全连接层，输入维度128，输出维度10

    def forward(self, x):
        # 定义前向传播过程
        out = self.conv1(x)  # 第一个卷积操作
        out = self.conv2(out)  # 第二个卷积操作
        out = self.identity1(out)  # 标识层1操作
        out = self.max_pool1(out)  # 最大池化操作

        batch_size = x.shape[0]  # 获取输入张量的批次大小
        out = out.reshape(batch_size, -1)  # 将输出展平为(batch_size, -1)的形状

        out = F.relu(self.identity2(self.linear1(out)))  # 第一个全连接层后接ReLU激活函数和标识层2
        out = self.linear2(out)  # 第二个全连接层

        return out  # 返回最终输出


class TestActivationSparsifier(TestCase):
    def _check_constructor(self, activation_sparsifier, model, defaults, sparse_config):
        """Helper function to check if the model, defaults and sparse_config are loaded correctly
        in the activation sparsifier
        """
        sparsifier_defaults = activation_sparsifier.defaults  # 获取激活稀疏化器的默认配置
        combined_defaults = {**defaults, "sparse_config": sparse_config}  # 组合默认配置和稀疏配置参数

        # 验证组合后的配置项数量不超过激活稀疏化器的默认配置项数量
        assert len(combined_defaults) <= len(activation_sparsifier.defaults)

        for key, config in sparsifier_defaults.items():
            # 检查所有组合配置的键是否存在于激活稀疏化器的默认配置中，并且对应的值相等
            assert config == combined_defaults.get(key, None)

    def _check_register_layer(
        self, activation_sparsifier, defaults, sparse_config, layer_args_list
        """Checks if layers in the model`
    ):
        """Checks if layers in the model are correctly mapped to their arguments.

        Args:
            activation_sparsifier (sparsifier object):
                The activation sparsifier object being tested.

            defaults (Dict):
                All default configurations (except sparse_config).

            sparse_config (Dict):
                Default sparse configuration passed to the sparsifier.

            layer_args_list (list of tuples):
                Each entry corresponds to the layer arguments:
                - First entry in the tuple: All arguments except sparse_config.
                - Second entry in the tuple: sparse_config.

        """
        # check args
        # Retrieve data groups from activation sparsifier
        data_groups = activation_sparsifier.data_groups
        # Ensure number of data groups matches number of layer argument lists
        assert len(data_groups) == len(layer_args_list)
        # Iterate over layer argument lists
        for layer_args in layer_args_list:
            layer_arg, sparse_config_layer = layer_args

            # check sparse config
            # Deep copy sparse_config and update with sparse_config_layer
            sparse_config_actual = copy.deepcopy(sparse_config)
            sparse_config_actual.update(sparse_config_layer)

            # Get fully qualified name of layer using activation_sparsifier model
            name = module_to_fqn(activation_sparsifier.model, layer_arg["layer"])

            # Assert sparse_config matches expected sparse_config_actual
            assert data_groups[name]["sparse_config"] == sparse_config_actual

            # assert the rest
            # Deep copy defaults and update with layer_arg, remove "layer" key
            other_config_actual = copy.deepcopy(defaults)
            other_config_actual.update(layer_arg)
            other_config_actual.pop("layer")

            # Assert each key-value pair in other_config_actual against data_groups[name]
            for key, value in other_config_actual.items():
                assert key in data_groups[name]
                assert value == data_groups[name][key]

            # Ensure get_mask raises ValueError for the current layer name
            with self.assertRaises(ValueError):
                activation_sparsifier.get_mask(name=name)
    def _check_pre_forward_hook(self, activation_sparsifier, data_list):
        """Registering a layer attaches a pre-forward hook to that layer. This function
        checks if the pre-forward hook works as expected. Specifically, checks if the
        input is aggregated correctly.

        Basically, asserts that the aggregate of input activations is the same as what was
        computed in the sparsifier.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_list (list of torch tensors)
                data input to the model attached to the sparsifier

        """
        # can only check for the first layer
        # 获取第一个数据聚合结果
        data_agg_actual = data_list[0]
        # 获取模型
        model = activation_sparsifier.model
        # 获取层名称
        layer_name = module_to_fqn(model, model.conv1)
        # 获取聚合函数
        agg_fn = activation_sparsifier.data_groups[layer_name]["aggregate_fn"]

        # 对于数据列表中的每个数据，进行聚合操作
        for i in range(1, len(data_list)):
            data_agg_actual = agg_fn(data_agg_actual, data_list[i])

        # 断言确保数据组中包含"data"字段，并且与聚合后的数据一致
        assert "data" in activation_sparsifier.data_groups[layer_name]
        assert torch.all(
            activation_sparsifier.data_groups[layer_name]["data"] == data_agg_actual
        )

        # 返回聚合后的数据
        return data_agg_actual

    def _check_step(self, activation_sparsifier, data_agg_actual):
        """Checks if .step() works as expected. Specifically, checks if the mask is computed correctly.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_agg_actual (torch tensor)
                aggregated torch tensor

        """
        # 获取模型
        model = activation_sparsifier.model
        # 获取层名称
        layer_name = module_to_fqn(model, model.conv1)
        # 断言确保层名称不为空
        assert layer_name is not None

        # 获取数据缩减函数
        reduce_fn = activation_sparsifier.data_groups[layer_name]["reduce_fn"]

        # 使用缩减函数对聚合数据进行缩减操作
        data_reduce_actual = reduce_fn(data_agg_actual)
        # 获取掩码函数和稀疏配置
        mask_fn = activation_sparsifier.data_groups[layer_name]["mask_fn"]
        sparse_config = activation_sparsifier.data_groups[layer_name]["sparse_config"]
        # 计算实际的掩码
        mask_actual = mask_fn(data_reduce_actual, **sparse_config)

        # 获取模型计算的掩码
        mask_model = activation_sparsifier.get_mask(layer_name)

        # 断言确保模型计算的掩码与实际计算的掩码一致
        assert torch.all(mask_model == mask_actual)

        # 断言确保数据组中的每个配置中都不包含"data"字段
        for config in activation_sparsifier.data_groups.values():
            assert "data" not in config
    def _check_squash_mask(self, activation_sparsifier, data):
        """Makes sure that squash_mask() works as usual. Specifically, checks
        if the sparsifier hook is attached correctly.
        This is achieved by only looking at the identity layers and making sure that
        the output == layer(input * mask).

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data (torch tensor)
                dummy batched data
        """

        # create a forward hook for checking output == layer(input * mask)
        def check_output(name):
            # Retrieve the mask and features for the current layer
            mask = activation_sparsifier.get_mask(name)
            features = activation_sparsifier.data_groups[name].get("features")
            feature_dim = activation_sparsifier.data_groups[name].get("feature_dim")

            # Define the hook function
            def hook(module, input, output):
                input_data = input[0]
                if features is None:
                    # Check if output equals layer(input * mask)
                    assert torch.all(mask * input_data == output)
                else:
                    # Iterate over each feature and verify output equality
                    for feature_idx in range(0, len(features)):
                        feature = torch.Tensor(
                            [features[feature_idx]], device=input_data.device
                        ).long()
                        inp_data_feature = torch.index_select(
                            input_data, feature_dim, feature
                        )
                        out_data_feature = torch.index_select(
                            output, feature_dim, feature
                        )

                        assert torch.all(
                            mask[feature_idx] * inp_data_feature == out_data_feature
                        )

            return hook

        # Register forward hooks for identity layers
        for name, config in activation_sparsifier.data_groups.items():
            if "identity" in name:
                config["layer"].register_forward_hook(check_output(name))

        # Execute the model with the provided data
        activation_sparsifier.model(data)


注释：
    # 检查状态字典的加载和恢复是否按预期工作
    """Checks if loading and restoring of state_dict() works as expected.
    Basically, dumps the state of the sparsifier and loads it in the other sparsifier
    and checks if all the configuration are in line.

    This function is called at various times in the workflow to makes sure that the sparsifier
    can be dumped and restored at any point in time.
    """
    # 获取 sparsifier1 的状态字典
    state_dict = sparsifier1.state_dict()

    # 创建一个新的模型对象
    new_model = Model()

    # 创建一个空的新 sparsifier2
    sparsifier2 = ActivationSparsifier(new_model)

    # 断言 sparsifier2 的默认设置与 sparsifier1 不同
    assert sparsifier2.defaults != sparsifier1.defaults

    # 断言 sparsifier2 的数据组数量与 sparsifier1 不同
    assert len(sparsifier2.data_groups) != len(sparsifier1.data_groups)

    # 使用 sparsifier1 的状态字典加载 sparsifier2
    sparsifier2.load_state_dict(state_dict)

    # 断言 sparsifier2 的默认设置与 sparsifier1 相同
    assert sparsifier2.defaults == sparsifier1.defaults

    # 检查每个状态字典中的条目
    for name, state in sparsifier2.state.items():
        # 断言状态字典中的名称在 sparsifier1 的状态字典中存在
        assert name in sparsifier1.state
        mask1 = sparsifier1.state[name]["mask"]
        mask2 = state["mask"]

        # 如果 mask1 为空，则要求 mask2 也为空
        if mask1 is None:
            assert mask2 is None
        else:
            # 断言 mask1 和 mask2 的类型相同
            assert type(mask1) == type(mask2)
            # 如果 mask1 是列表，则逐个比较列表中的元素
            if isinstance(mask1, list):
                assert len(mask1) == len(mask2)
                for idx in range(len(mask1)):
                    assert torch.all(mask1[idx] == mask2[idx])
            else:
                # 否则比较两个张量是否完全相同
                assert torch.all(mask1 == mask2)

    # 确保状态字典中的 mask 存储为稀疏张量
    for state in state_dict["state"].values():
        mask = state["mask"]
        if mask is not None:
            # 如果 mask 是列表，则逐个检查是否为稀疏张量
            if isinstance(mask, list):
                for idx in range(len(mask)):
                    assert mask[idx].is_sparse
            else:
                # 否则检查是否为稀疏张量
                assert mask.is_sparse

    # 比较 sparsifier1 和 sparsifier2 的数据组配置
    dg1, dg2 = sparsifier1.data_groups, sparsifier2.data_groups

    for layer_name, config in dg1.items():
        # 断言层名称在 dg2 中存在
        assert layer_name in dg2

        # 排除 "hook" 和 "layer" 键，比较两个配置字典
        config1 = {
            key: value
            for key, value in config.items()
            if key not in ["hook", "layer"]
        }
        config2 = {
            key: value
            for key, value in dg2[layer_name].items()
            if key not in ["hook", "layer"]
        }

        # 断言两个配置字典相等
        assert config1 == config2
```