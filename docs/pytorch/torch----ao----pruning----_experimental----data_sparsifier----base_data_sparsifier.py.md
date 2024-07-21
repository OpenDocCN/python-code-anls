# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\base_data_sparsifier.py`

```
# 引入必要的模块和库
import abc  # 抽象基类模块
import torch  # PyTorch深度学习库
from typing import Optional, Tuple, List, Any, Dict  # 类型提示相关
from ...sparsifier import base_sparsifier  # 导入基础稀疏化器
from collections import defaultdict  # 默认字典，用于初始化数据结构
from torch import nn  # PyTorch神经网络模块
import copy  # 复制对象相关操作
from ...sparsifier import utils  # 稀疏化相关工具函数
from torch.nn.utils import parametrize  # PyTorch参数化工具
import sys  # 系统相关操作
import warnings  # 警告处理模块

if not sys.warnoptions:
    # 在训练循环中使用时，抑制重复的警告信息
    warnings.simplefilter("once")

__all__ = ['BaseDataSparsifier']  # 当前模块导出的所有符号名称

EMBEDDING_TYPES = {
    nn.Embedding,  # 嵌入层
    nn.EmbeddingBag,  # 嵌入袋
}

SUPPORTED_TYPES = {
    torch.Tensor,  # PyTorch张量
    nn.Parameter,  # PyTorch参数
    *EMBEDDING_TYPES,  # 所有嵌入类型的组合
}


class _Container(nn.Module):
    pass  # 空的子类，用作基类的占位符


class BaseDataSparsifier(base_sparsifier.BaseSparsifier):
    r"""
    所有数据稀疏化器的基类。
    这个抽象类接受原始的 torch 张量 / 嵌入 / 嵌入袋（参见上面的 SUPPORTED_TYPES）用于稀疏化准备。
    在这种情况下，掩码（和参数化）由类而不是用户拥有。
    具体来说，类内的容器对象维护输入数据的掩码和参数化。

    Args:
        data_list (list of tuples)
            要稀疏化的 (name, data) 元组列表。查看 SUPPORTED_TYPES 以获取数据类型。
            在内部，一个容器模块处理数据的稀疏化。

        defaults (dict)
            将默认配置附加到配置中。只更新 `config` 中不存在的键。

    Example::
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list=data_list, **defaults)  # 某个继承 BaseDataSparsifier 的稀疏化器
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 和 tensor_2 的稀疏程度为 0.7，而 tensor_3 的稀疏程度为 0.3
    """
    def __init__(self, data_list: Optional[List[Tuple[str, Any]]] = None, **defaults):
        super().__init__(defaults=defaults)

        self._container = _Container()  # 初始化容器对象

        self.data_groups: Dict[str, Dict] = defaultdict(dict)  # 数据组的字典，初始化为空字典
        if data_list is not None:
            # 将带有默认配置的数据添加到这里
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self):
        raise NotImplementedError("this function is undefined for this class")  # 抽象方法，子类需要实现

    def _extract_weight(self, data):
        # 提取权重参数而不是底层数据
        if type(data) in [torch.Tensor, nn.Parameter]:
            return data
        elif type(data) in EMBEDDING_TYPES:
            return data.weight
    def add_data(self, name: str, data, reuse_mask=True, **config):
        r""" Configures and parametrizes the internal container model with name and data.

        **Note**:
            1. If the data with name already exists, it replaces the data.
            2. While replacing, the old mask is reused when `reuse_mask=True`
            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.
            4. By default, the config of the replaced data is used as config for the replacing data, unless something
               is specified in the config dictionary.
        """
        assert type(data) in SUPPORTED_TYPES, \
            "specified data type not supported at the moment"

        # Deep copy the default configuration and update it with any provided config
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)

        # Extract the weight from the provided data
        weight = self._extract_weight(data)

        # Initialize mask with ones, or reuse existing mask if specified
        mask = local_args.get('mask', torch.ones_like(weight))

        # Determine the parametrization class to use, defaulting to FakeSparsity
        param_class = local_args.get('parametrization', utils.FakeSparsity)

        if name in self.state:
            # Warn about replacing existing data with the same name
            warnings.warn("Replacing existing data of the same name. - Did you mean a different name?")

            # Retrieve and update configuration from existing data
            old_args = self.data_groups[name]
            local_args = copy.deepcopy(old_args)
            local_args.update(config)

            if reuse_mask:
                # If reuse_mask=True, verify new data shape matches existing data
                current_data = self.get_data(name=name)
                assert weight.shape == current_data.shape, \
                    "to retain the old mask, the shape of the new data must be the same as the previous one"
                mask = self.get_mask(name=name)  # Reuse existing mask instead of creating a new one

            # Remove existing data entry
            self._delete_data(name=name)

        # Register the weight tensor in the internal container
        self._container.register_buffer(name=name, tensor=weight)

        # Apply parametrization to the registered tensor
        parametrize.register_parametrization(self._container, name, param_class(mask))

        # Update the state and configuration records
        self.state[name]['mask'] = mask
        self.data_groups[name] = local_args

        # Return the named tensor from the internal container
        return getattr(self._container, name)


    def get_data(self, name: str, return_original: bool = True):
        r"""Returns weight tensor (or data)

        Args:
            - name: name of the data to be returned
    # 将给定状态(states)的掩码(mask)转换为稀疏COO张量或密集张量，取决于sparse_coo参数。
    def _convert_mask(self, states, sparse_coo=True):
        # 深度复制状态(states)，以免直接修改原始数据
        states = copy.deepcopy(states)
        # 遍历states中的每一个状态
        for state in states.values():
            # 如果sparse_coo为True，则将掩码(mask)转换为稀疏COO张量
            if sparse_coo:
                state['mask'] = state['mask'].to_sparse_coo()
            else:
                # 否则将掩码(mask)转换为密集张量
                state['mask'] = state['mask'].to_dense()

        # 返回转换后的states
        return states

    # 返回优化器的状态作为一个字典
    def state_dict(self):
        # 调用_convert_mask方法将self.state中的掩码转换为相应的稀疏或密集张量形式
        state = self._convert_mask(self.state)
        # 返回包含以下内容的字典：
        # - state: 包含名称到掩码映射的字典
        # - data_groups: 包含所有稀疏性配置组的列表，键名指定数据的名称
        # - _container: 内部容器模型的状态字典，用于稀疏化
        return {
            'state': state,
            'data_groups': self.data_groups,
            '_container': self._container.state_dict()
        }

    # 从给定的states、data_groups和container_state_dict中加载容器的状态
    def _load_container_from_state(self, states, data_groups, container_state_dict):
        # 遍历每一个状态及其对应的名称
        for name, state in states.items():
            # 获取与当前名称对应的配置名称
            config_name = data_groups.get(name, None)
            # 如果配置名称为None，则抛出运行时错误
            if config_name is None:
                raise RuntimeError(f"Error loading {name}")

            # 检查是否已对具有此名称的数据进行了参数化，如果是，则进行参数化处理，否则只添加属性到容器中
            parametrized_name = f'parametrizations.{name}.original'
            parametrized = False
            data = container_state_dict.get(name, None)
            # 如果容器状态字典中存在当前名称的数据
            if name in container_state_dict:
                # 可能已删除此处的参数化
                data = container_state_dict.get(name)

            # 如果在容器状态字典中存在parametrized_name
            elif parametrized_name in container_state_dict:
                # 权重已参数化
                data = container_state_dict.get(parametrized_name)
                parametrized = True

            else:
                # 否则抛出运行时错误
                raise RuntimeError(f"Error loading {name}")

            # 将数据作为缓冲区注册到容器中
            self._container.register_buffer(name=name, tensor=data)

            # 如果已参数化，则注册参数
            if parametrized:
                # 获取状态中的掩码(mask)，默认为与data相同形状的全1张量
                mask = state.get('mask', torch.ones_like(data))
                # 获取parametrization类，用于参数化
                param_class = data_groups.get('parametrization', utils.FakeSparsity)  # 一旦修复了utils的public_api，此处将更改！
                # 在_container中注册参数化
                parametrize.register_parametrization(self._container, name, param_class(mask))
    def load_state_dict(self, state_dict, strict=True):
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary that to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased.
        """
        # 深拷贝状态字典中的'state'项
        states = copy.deepcopy(state_dict['state'])
        # 深拷贝状态字典中的'data_groups'项
        data_groups = copy.deepcopy(state_dict['data_groups'])
        # 深拷贝状态字典中的'_container'项
        container_state_dict = copy.deepcopy(state_dict['_container'])

        # 将'states'中的稀疏 COO 掩码转换为稠密形式
        states = self._convert_mask(states, sparse_coo=False)

        # 如果 strict 为 True，则重置容器对象
        if strict:
            # 重置 _container 对象为新的 _Container 实例
            self._container = _Container()

        # 从状态中加载容器的内容
        self._load_container_from_state(states, data_groups, container_state_dict)

        # 如果 strict 不为 True，则更新当前状态和数据组
        if not strict:
            # 更新 states 和 self.state 的内容
            states.update(self.state)
            # 更新 data_groups 和 self.data_groups 的内容
            data_groups.update(self.data_groups)

        # 设置当前对象的状态
        self.__setstate__({'state': states, 'data_groups': data_groups})

    def __setstate__(self, state):
        # 如果状态中包含 '_container'，则加载模型
        if '_container' in state:
            # 弹出 '_container' 并加载到 container_dict
            container_dict = state.pop('_container')
            # 重置 _container 对象为新的 _Container 实例
            self._container = _Container()
            # 将'state'中的稀疏 COO 掩码转换为稠密形式
            state['state'] = self._convert_mask(state['state'], sparse_coo=False)
            # 从状态中加载容器的内容
            self._load_container_from_state(state['state'], state['data_groups'], container_dict)

        # 更新对象的字典形式状态
        self.__dict__.update(state)

    def __getstate__(self):
        # 将 self.state 转换为稀疏形式
        state = self._convert_mask(self.state)
        return {
            'defaults': self.defaults,
            'state': state,
            'data_groups': self.data_groups,
            '_container': self._container.state_dict()
        }

    def __repr__(self):
        # 创建对象的字符串表示形式
        format_string = self.__class__.__name__ + ' ('
        for name, sparse_args in self.data_groups.items():
            format_string += '\n'
            format_string += '\tData Group\n'
            format_string += f'\t    name: {name}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'data':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def get_mask(self, name: str):
        # 获取给定名称的掩码数据
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]['mask']
    # 压缩稀疏掩码到相应的张量中。还可以接受要压缩掩码的字符串列表。如果没有提供，则压缩所有键的掩码。
    # kwargs:
    # * names: 要压缩掩码的字符串列表
    # * sparsified: 如果为True - 在压缩之前应用掩码；如果为False - 在压缩之前不应用掩码
    def squash_mask(self, *args, leave_parametrized=True, names=None, **kwargs):
        if names is None:
            names = list(self.data_groups.keys())
        for name in names:
            # 使用 parametrize 模块移除参数化信息，可能保留参数化，具体取决于 leave_parametrized 参数
            parametrize.remove_parametrizations(self._container, name, leave_parametrized=leave_parametrized)

    def step(self):
        if not self.enable_mask_update:
            return
        # 使用 torch.no_grad() 上下文管理器禁用梯度计算
        with torch.no_grad():
            # 遍历数据组字典中的每个数据名称和配置
            for name, config in self.data_groups.items():
                # 获取非稀疏化的数据
                data = self.get_data(name)
                # 更新掩码，根据配置参数更新给定数据的掩码
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):
        # 抽象方法，用于在子类中实现，更新指定名称的数据的掩码
        pass

    def _delete_data(self, name):
        """从稀疏化器中分离某些数据。

        Args:
            name (str)
                要从稀疏化器中删除的数据的名称

        Note:
            当前为私有方法。在替换同名数据时作为辅助函数使用。
        """
        # 删除数据名称对应的掩码，同时不应用掩码
        self.squash_mask(names=[name], leave_parametrized=False)
        # 删除属性中对应名称的数据
        delattr(self._container, name)
        # 从状态中删除名称对应的数据状态
        self.state.pop(name)
        # 从数据组中删除名称对应的数据配置
        self.data_groups.pop(name)
```