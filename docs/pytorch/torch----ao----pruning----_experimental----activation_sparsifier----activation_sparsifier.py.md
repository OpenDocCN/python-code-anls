# `.\pytorch\torch\ao\pruning\_experimental\activation_sparsifier\activation_sparsifier.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类型声明
from typing import Any, Dict, List, Optional
import torch  # 导入 PyTorch 模块
from collections import defaultdict  # 导入 defaultdict 类
from torch import nn  # 导入 PyTorch 的神经网络模块
import copy  # 导入 copy 模块，用于复制对象
from ...sparsifier.utils import fqn_to_module, module_to_fqn  # 导入自定义的工具函数
import warnings  # 导入警告模块

__all__ = ['ActivationSparsifier']

# 定义 ActivationSparsifier 类
class ActivationSparsifier:
    r"""
    The Activation sparsifier class aims to sparsify/prune activations in a neural
    network. The idea is to attach the sparsifier to a layer (or layers) and it
    zeroes out the activations based on the mask_fn (or sparsification function)
    input by the user.
    The mask_fn is applied once all the inputs are aggregated and reduced i.e.
    mask = mask_fn(reduce_fn(aggregate_fn(activations)))

    Note::
        The sparsification mask is computed on the input **before it goes through the attached layer**.

    Args:
        model (nn.Module):
            The model whose layers will be sparsified. The layers that needs to be
            sparsified should be added separately using the register_layer() function
        aggregate_fn (Optional, Callable):
            default aggregate_fn that is used if not specified while registering the layer.
            specifies how inputs should be aggregated over time.
            The aggregate_fn should usually take 2 torch tensors and return the aggregated tensor.
            Example
                def add_agg_fn(tensor1, tensor2):  return tensor1 + tensor2
        reduce_fn (Optional, Callable):
            default reduce_fn that is used if not specified while registering the layer.
            reduce_fn will be called on the aggregated tensor i.e. the tensor obtained after
            calling agg_fn() on all inputs.
            Example
                def mean_reduce_fn(agg_tensor):    return agg_tensor.mean(dim=0)
        mask_fn (Optional, Callable):
            default mask_fn that is used to create the sparsification mask using the tensor obtained after
            calling the reduce_fn(). This is used by default if a custom one is passed in the
            register_layer().
            Note that the mask_fn() definition should contain the sparse arguments that is passed in sparse_config
            arguments.
        features (Optional, list):
            default selected features to sparsify.
            If this is non-empty, then the mask_fn will be applied for each feature of the input.
            For example,
                mask = [mask_fn(reduce_fn(aggregated_fn(input[feature])) for feature in features]
        feature_dim (Optional, int):
            default dimension of input features. Again, features along this dim will be chosen
            for sparsification.
        sparse_config (Dict):
            Default configuration for the mask_fn. This config will be passed
            with the mask_fn()
    """
    def __init__(self, model: nn.Module, aggregate_fn=None, reduce_fn=None, mask_fn=None,
                 features=None, feature_dim=None, **sparse_config):
        # 初始化函数，设置对象的初始属性
        self.model = model  # 设置模型属性
        self.defaults: Dict[str, Any] = defaultdict()  # 使用默认字典存储默认配置
        self.defaults['sparse_config'] = sparse_config  # 存储稀疏配置参数

        # functions
        self.defaults['aggregate_fn'] = aggregate_fn  # 存储聚合函数
        self.defaults['reduce_fn'] = reduce_fn  # 存储减少函数
        self.defaults['mask_fn'] = mask_fn  # 存储掩码函数

        # default feature and feature_dim
        self.defaults['features'] = features  # 存储特征
        self.defaults['feature_dim'] = feature_dim  # 存储特征维度

        self.data_groups: Dict[str, Dict] = defaultdict(dict)  # 存储注册层的相关信息，每个层的信息存储在字典中

        self.state: Dict[str, Any] = defaultdict(dict)  # 存储状态信息，每个层的掩码存储在字典中

    @staticmethod
    def _safe_rail_checks(args):
        """确保一些函数和属性未被错误传递
        """

        # 如果 features 不为 None，则 feature_dim 不能为 None
        features, feature_dim = args['features'], args['feature_dim']
        if features is not None:
            assert feature_dim is not None, "need feature dim to select features"  # 断言确保需要特征维度来选择特征

        # 所有 *_fn 应该是可调用的函数
        fn_keys = ['aggregate_fn', 'reduce_fn', 'mask_fn']
        for key in fn_keys:
            fn = args[key]
            assert callable(fn), 'function should be callable'  # 断言确保函数是可调用的
    def _aggregate_hook(self, name):
        """Returns hook that computes aggregate of activations passing through.
        """

        # 获取特征维度和特征数据以及聚合函数
        feature_dim = self.data_groups[name]['feature_dim']
        features = self.data_groups[name]['features']
        agg_fn = self.data_groups[name]['aggregate_fn']

        def hook(module, input) -> None:
            input_data = input[0]

            data = self.data_groups[name].get('data')  # 获取已聚合数据
            if features is None:
                # 如果没有关联特征，数据不应为列表
                if data is None:
                    data = torch.zeros_like(input_data)  # 创建与输入数据相同形状的零张量
                    self.state[name]['mask'] = torch.ones_like(input_data)  # 创建与输入数据相同形状的全一张量
                out_data = agg_fn(data, input_data)  # 聚合输入数据和已有数据
            else:
                # 数据应为列表（针对每个特征聚合）
                if data is None:
                    out_data = [0 for _ in range(0, len(features))]  # 创建一个列表以保存聚合后的数据
                    self.state[name]['mask'] = [0 for _ in range(0, len(features))]  # 创建一个列表以保存掩码信息
                else:
                    out_data = data  # 使用已有数据列表

                # 对每个特征计算聚合
                for feature_idx in range(len(features)):
                    # 每个特征可能是列表或标量，转换为 torch 张量
                    feature_tensor = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                    data_feature = torch.index_select(input_data, feature_dim, feature_tensor)  # 根据特征维度索引数据
                    if data is None:
                        curr_data = torch.zeros_like(data_feature)  # 创建一个与数据特征相同形状的零张量
                        self.state[name]['mask'][feature_idx] = torch.ones_like(data_feature)  # 设置掩码为全一张量
                    else:
                        curr_data = data[feature_idx]  # 获取当前特征已有的数据
                    out_data[feature_idx] = agg_fn(curr_data, data_feature)  # 聚合当前特征的数据和已有数据
            self.data_groups[name]['data'] = out_data  # 更新数据组中的已聚合数据
        return hook
    # 注册一个用于稀疏化的层。layer参数应属于self.model的一部分。
    # 具体来说，注册一个前向钩子到该层。钩子将应用aggregate_fn并存储聚合的激活值作为每一步的输入。
    def register_layer(self, layer: nn.Module, aggregate_fn=None, reduce_fn=None,
                       mask_fn=None, features=None, feature_dim=None, **sparse_config):
        r"""
        Registers a layer for sparsification. The layer should be part of self.model.
        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn
        and store the aggregated activations that is input over each step.

        Note::
            - There is no need to pass in the name of the layer as it is automatically computed as per
              the fqn convention.

            - All the functions (fn) passed as argument will be called at a dim, feature level.
        """
        # 计算层的全限定名（fqn）作为注册的标识符
        name = module_to_fqn(self.model, layer)
        # 如果未找到层，则引发断言错误
        assert name is not None, "layer not found in the model"  # satisfy mypy

        # 如果层已经在data_groups中，则取消注册该层
        if name in self.data_groups:  # unregister layer if already present
            # 发出警告并重新注册层到新配置
            warnings.warn("layer already attached to the sparsifier, deregistering the layer and registering with new config")
            self.unregister_layer(name=name)

        # 深拷贝self.defaults以确保本地参数独立于全局配置
        local_args = copy.deepcopy(self.defaults)
        
        # 更新本地参数字典，包括传入的非空参数
        update_dict = {
            'aggregate_fn': aggregate_fn,
            'reduce_fn': reduce_fn,
            'mask_fn': mask_fn,
            'features': features,
            'feature_dim': feature_dim,
            'layer': layer
        }
        local_args.update((arg, val) for arg, val in update_dict.items() if val is not None)
        
        # 更新sparse_config到sparse_config中
        local_args['sparse_config'].update(sparse_config)

        # 执行安全的边界检查
        self._safe_rail_checks(local_args)

        # 将本地参数注册到data_groups字典中，使用层的名称作为键
        self.data_groups[name] = local_args

        # 注册一个前向钩子到层，用于聚合操作
        agg_hook = layer.register_forward_pre_hook(self._aggregate_hook(name=name))

        # 初始化状态字典中该层对应的mask为None，模型前向传播时将会创建mask
        self.state[name]['mask'] = None  # mask will be created when model forward is called.

        # 将聚合钩子（agg_hook）附加到data_groups中对应层的钩子（hook）字段
        self.data_groups[name]['hook'] = agg_hook

        # 为了序列化的目的，指示是否附加了aggregate_hook
        self.data_groups[name]['hook_state'] = "aggregate"  # aggregate hook is attached
    def get_mask(self, name: Optional[str] = None, layer: Optional[nn.Module] = None):
        """
        Returns mask associated to the layer.

        The mask is
            - a torch tensor if features for that layer is None.
            - a list of torch tensors for each feature, otherwise

        Note::
            The shape of the mask is unknown until model.forward() is applied.
            Hence, if get_mask() is called before model.forward(), an
            error will be raised.
        """
        assert name is not None or layer is not None, "Need at least name or layer obj to retrieve mask"

        if name is None:
            assert layer is not None
            name = module_to_fqn(self.model, layer)
            assert name is not None, "layer not found in the specified model"

        if name not in self.state:
            raise ValueError("Error: layer with the given name not found")

        # Retrieve the mask associated with the layer name from the internal state
        mask = self.state[name].get('mask', None)

        if mask is None:
            raise ValueError("Error: shape unknown, call layer() routine at least once to infer mask")
        return mask

    def unregister_layer(self, name):
        """Detaches the sparsifier from the layer
        """

        # Detach any hooks attached to the layer specified by `name`
        self.data_groups[name]['hook'].remove()

        # Remove the layer state from the internal state dict
        self.state.pop(name)

        # Remove the layer data group from the data groups dictionary
        self.data_groups.pop(name)

    def step(self):
        """Internally calls the update_mask() function for each layer
        """
        with torch.no_grad():
            # Iterate over each registered layer and update its mask
            for name, configs in self.data_groups.items():
                data = configs['data']
                self.update_mask(name, data, configs)

                # Reset the accumulated data for the current layer
                self.data_groups[name].pop('data')  # reset the accumulated data

    def update_mask(self, name, data, configs):
        """
        Called for each registered layer and does the following-
            1. apply reduce_fn on the aggregated activations
            2. use mask_fn to compute the sparsification mask

        Note:
            the reduce_fn and mask_fn is called for each feature, dim over the data
        """
        # Retrieve the current mask associated with the layer name
        mask = self.get_mask(name)
        
        # Extract sparse configuration and features from the provided configs
        sparse_config = configs['sparse_config']
        features = configs['features']
        reduce_fn = configs['reduce_fn']
        mask_fn = configs['mask_fn']

        if features is None:
            # If features is None, apply reduce_fn directly on the data and compute the mask
            data = reduce_fn(data)
            mask.data = mask_fn(data, **sparse_config)
        else:
            # If features is a list, iterate over each feature index and compute masks individually
            for feature_idx in range(len(features)):
                data_feature = reduce_fn(data[feature_idx])
                mask[feature_idx].data = mask_fn(data_feature, **sparse_config)
    def _sparsify_hook(self, name):
        """Returns hook that applies sparsification mask to input entering the attached layer
        """
        # 获取名为name的层的稀疏化掩码
        mask = self.get_mask(name)
        # 获取名为name的数据组的特征
        features = self.data_groups[name]['features']
        # 获取名为name的数据组的特征维度
        feature_dim = self.data_groups[name]['feature_dim']

        def hook(module, input):
            # 获取输入数据
            input_data = input[0]
            if features is None:
                # 如果特征为空，应用到所有特征
                return input_data * mask
            else:
                # 针对每个特征应用掩码
                for feature_idx in range(0, len(features)):
                    # 创建张量表示特征，并移到对应设备
                    feature = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                    # 通过索引选择特定特征的数据，并应用稀疏化掩码
                    sparsified = torch.index_select(input_data, feature_dim, feature) * mask[feature_idx]
                    # 将稀疏化后的数据复制回原数据中对应位置
                    input_data.index_copy_(feature_dim, feature, sparsified)
                return input_data
        return hook

    def squash_mask(self, attach_sparsify_hook=True, **kwargs):
        """
        Unregisters aggregate hook that was applied earlier and registers sparsification hooks if
        attach_sparsify_hook = True.
        """
        for name, configs in self.data_groups.items():
            # 取消注册先前应用的聚合钩子
            configs['hook'].remove()
            # 移除钩子对象
            configs.pop('hook')
            # 设置钩子状态为空
            self.data_groups[name]['hook_state'] = "None"
            if attach_sparsify_hook:
                # 注册名为name的层的稀疏化钩子
                configs['hook'] = configs['layer'].register_forward_pre_hook(self._sparsify_hook(name))
            # 设置钩子状态为稀疏化
            configs['hook_state'] = "sparsify"  # signals that sparsify hook is now attached

    def _get_serializable_data_groups(self):
        """Exclude hook and layer from the config keys before serializing

        TODO: Might have to treat functions (reduce_fn, mask_fn etc) in a different manner while serializing.
              For time-being, functions are treated the same way as other attributes
        """
        # 创建默认字典用于存储数据组
        data_groups: Dict[str, Any] = defaultdict()
        # 遍历每个数据组
        for name, config in self.data_groups.items():
            # 从配置中排除钩子和层，以便在序列化之前进行处理
            new_config = {key: value for key, value in config.items() if key not in ['hook', 'layer']}
            # 将处理后的配置存入数据组中
            data_groups[name] = new_config
        return data_groups
    def _convert_mask(self, states_dict, sparse_coo=True):
        r"""Converts the mask to sparse coo or dense depending on the `sparse_coo` argument.
        If `sparse_coo=True`, then the mask is stored as sparse coo else dense tensor
        """
        # 深拷贝输入的状态字典，以免修改原始数据
        states = copy.deepcopy(states_dict)
        # 遍历状态字典中的每一个状态
        for state in states.values():
            # 如果当前状态的'mask'不为None
            if state['mask'] is not None:
                # 如果'mask'是一个列表
                if isinstance(state['mask'], List):
                    # 遍历列表中的每一个元素
                    for idx in range(len(state['mask'])):
                        # 根据 sparse_coo 参数选择将当前元素转换为稀疏 COO 格式还是密集张量格式
                        if sparse_coo:
                            state['mask'][idx] = state['mask'][idx].to_sparse_coo()
                        else:
                            state['mask'][idx] = state['mask'][idx].to_dense()
                else:
                    # 如果'mask'不是列表，直接根据 sparse_coo 参数选择转换为稀疏 COO 格式或密集张量格式
                    if sparse_coo:
                        state['mask'] = state['mask'].to_sparse_coo()
                    else:
                        state['mask'] = state['mask'].to_dense()
        # 返回转换后的状态字典
        return states

    def state_dict(self) -> Dict[str, Any]:
        r"""Returns the state of the sparsifier as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a dictionary containing all config information for each
            layer
        * defaults - the default config while creating the constructor
        """
        # 获取可序列化的数据组
        data_groups = self._get_serializable_data_groups()
        # 将当前对象的状态转换为包含稀疏化掩码的状态字典
        state = self._convert_mask(self.state)
        # 返回包含状态字典、数据组信息和默认配置的字典
        return {
            'state': state,
            'data_groups': data_groups,
            'defaults': self.defaults
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary that to which the current sparsifier needs to be restored to
        """
        # 从状态字典中获取状态、数据组信息和默认配置
        state = state_dict['state']
        data_groups, defaults = state_dict['data_groups'], state_dict['defaults']
        # 调用内部方法设置当前对象的状态为给定的状态、数据组信息和默认配置
        self.__set_state__({'state': state, 'data_groups': data_groups, 'defaults': defaults})

    def __get_state__(self) -> Dict[str, Any]:
        # 获取可序列化的数据组
        data_groups = self._get_serializable_data_groups()
        # 将当前对象的状态转换为包含稀疏化掩码的状态字典
        state = self._convert_mask(self.state)
        # 返回包含默认配置、状态字典和数据组信息的字典
        return {
            'defaults': self.defaults,
            'state': state,
            'data_groups': data_groups,
        }
    # 重写对象状态的方法，将给定的状态转换为稠密张量，更新对象的状态
    def __set_state__(self, state: Dict[str, Any]) -> None:
        state['state'] = self._convert_mask(state['state'], sparse_coo=False)  # 将掩码转换为稠密张量
        self.__dict__.update(state)

        # 需要将层和钩子信息附加到数据组中
        for name, config in self.data_groups.items():
            # 获取对应的层对象
            layer = fqn_to_module(self.model, name)
            assert layer is not None  # 确保层对象不为空，满足类型检查

            # 如果配置中指定了钩子状态为 "aggregate"，则注册聚合钩子
            if "hook_state" in config and config['hook_state'] == "aggregate":
                hook = layer.register_forward_pre_hook(self._aggregate_hook(name))

            # 如果配置中指定了钩子状态为 "sparsify"，则注册稀疏化钩子
            elif "hook_state" in config and config["hook_state"] == "sparsify":
                hook = layer.register_forward_pre_hook(self._sparsify_hook(name))

            config['layer'] = layer  # 将层对象存入配置字典中
            config['hook'] = hook  # 将钩子对象存入配置字典中，类型标注可能未定义

    # 返回对象的字符串表示形式，包括数据组的详细信息
    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('  # 类名作为开头
        for name, config in self.data_groups.items():
            format_string += '\n'
            format_string += '\tData Group\n'  # 数据组的标识
            format_string += f'\t    name: {name}\n'  # 数据组名称
            for key in sorted(config.keys()):
                # 排除不需要在字符串表示中展示的键
                if key in ['data', 'hook', 'reduce_fn', 'mask_fn', 'aggregate_fn']:
                    continue
                format_string += f'\t    {key}: {config[key]}\n'  # 输出每个配置项的键值对
        format_string += ')'  # 字符串结尾
        return format_string
```