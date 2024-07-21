# `.\pytorch\torch\ao\pruning\sparsifier\base_sparsifier.py`

```py
# mypy: allow-untyped-defs
# 引入抽象基类模块
import abc
# 复制对象模块
import copy
# 默认字典模块
from collections import defaultdict
# 类型提示相关模块
from typing import Any, Dict, Optional, Set, Tuple, List, Type

# 引入PyTorch模块
import torch
# 引入PyTorch神经网络模块
from torch import nn
# 引入PyTorch神经网络工具箱中的参数化相关模块
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import type_before_parametrizations

# 从当前目录下的utils模块中导入以下函数和类
from .utils import (
    module_contains_param,
    swap_module,
    FakeSparsity,
    get_arg_info_from_tensor_fqn,
    module_to_fqn,
)

# 定义公开接口列表，包含BaseSparsifier类名
__all__ = ["BaseSparsifier"]

# 定义支持的神经网络模块集合，仅包含nn.Linear
SUPPORTED_MODULES = {nn.Linear}

# 不包含在状态字典中的键列表
KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]

# 定义公开接口列表，包含BaseSparsifier类名
__all__ = ["BaseSparsifier"]


# TODO update desc with new config args
# BaseSparsifier基类，继承自抽象基类abc.ABC
class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    Example::

        >>> # xdoctest: +SKIP("Can't instantiate abstract class BaseSparsifier with abstract method update_mask")
        >>> config = [{'tensor_fqn': 'layer1.weight', 'tensor_fqn': 'linear2.weight2', 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """

    # 构造方法，初始化默认配置
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        super().__init__()
        # 设置默认配置，如果未提供则为空字典
        self.defaults: Dict[str, Any] = defaults or {}

        # 状态字典，使用默认字典初始化
        self.state: Dict[str, Dict] = defaultdict(dict)
        # 分组列表，初始化为空列表
        self.groups: List[Dict[str, Any]] = []
        # 启用掩码更新，默认为True
        self.enable_mask_update = True

    # 获取对象状态的方法，返回包含默认配置、状态字典和分组列表的字典
    def __getstate__(self) -> Dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "groups": self.groups,
        }

    # 设置对象状态的方法，根据传入的状态字典更新对象的属性
    def __setstate__(self, state: Dict[str, Dict[str, Any]]) -> None:
        self.__dict__.update(state)

    # 返回对象的字符串表示形式，显示类名及其分组的详细信息
    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, sparse_args in enumerate(self.groups):
            module = sparse_args["module"]
            format_string += "\n"
            format_string += f"\tGroup {i}\n"
            format_string += f"\t    module: {module}\n"
            for key in sorted(sparse_args.keys()):
                if key == "module":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string
    # 返回优化器当前状态的字典表示
    def state_dict(self) -> Dict[str, Any]:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * groups - a list containing all sparsity configuration groups
            with the key 'tensor_fqn' specifying the path to the sparsified tensor within a model

        TODO: Need a clean way of loading the state of the "prepared" module
        """

        # 创建包含所有组稀疏性配置的列表，其中每个组都排除 KEYS_NOT_IN_STATE_DICT 中指定的键
        groups: List[Dict[str, Any]] = [
            dict(
                filter(
                    lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT,
                    mg.items(),
                )
            )
            for mg in self.groups
        ]

        # 返回优化器的状态字典，包含状态和组信息
        return {
            "state": self.state,
            "groups": groups,
        }

    # 从给定的状态字典加载状态到优化器中
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        # 深拷贝状态字典中的 groups 列表
        groups = copy.deepcopy(state_dict["groups"])
        # 获取状态字典中的 state 对象
        states = state_dict["state"]
        # 遍历状态字典中的每个 tensor_fqn 和对应的状态信息 s
        for tensor_fqn, s in states.items():
            # 从 tensor_fqn 获取相关参数信息
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            # 获取参数所属模块
            module = arg_info["module"]
            # 获取张量名称
            tensor_name = arg_info["tensor_name"]
            # 如果 strict 为 True 且模块为 None，则引发运行时错误
            if strict and module is None:
                raise RuntimeError(f"Error loading {tensor_fqn} into the model")

            # 检查模块的参数化是否包含 FakeSparsity 类型的参数化，如果没有，则注册一个默认的 FakeSparsity 参数化
            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)

            # 如果状态信息 s 中存在 mask，则将其赋给参数化 p 的 mask 属性
            if s.get("mask", None) is not None:
                mask = s.pop("mask")
                p.mask = mask

            # 更新 groups 中与当前 tensor_fqn 对应的 mg 字典，将其扩展为 arg_info
            for mg in groups:
                if mg["tensor_fqn"] == tensor_fqn:
                    mg.update(arg_info)

        # 调用 __setstate__ 方法来设置优化器的状态和组信息
        self.__setstate__({"state": states, "groups": groups})

    # 从给定的模型创建配置信息列表
    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: Set[Type] = SUPPORTED_MODULES,
    ) -> None:
        # 初始化配置信息列表为空
        self.config = []
        # 使用栈来遍历模型及其子模块
        stack = [model]
        while stack:
            # 弹出栈顶的模块
            module = stack.pop()
            # 遍历模块的每个子模块
            for name, child in module.named_children():
                # 如果子模块的类型在 SUPPORTED_MODULES 集合中，则添加其权重张量路径到配置信息列表中
                if type(child) in SUPPORTED_MODULES:
                    module_fqn = module_to_fqn(model, child)
                    assert isinstance(module_fqn, str)  # for mypy
                    self.config.append({"tensor_fqn": module_fqn + ".weight"})
                else:
                    # 否则，将子模块压入栈中继续遍历其子模块
                    stack.append(child)
    def prepare(self, model, config):
        r"""Prepares a model, by adding the parametrizations.

        Note::

            The model is modified inplace. If you need to preserve the original
            model, use copy.deepcopy.
        """
        self.model = model  # 将传入的模型赋值给实例变量self.model
        self.config = config  # 将传入的配置赋值给实例变量self.config

        # If no config -- try getting all the supported layers
        if self.config is None:
            self.make_config_from_model(model)  # 如果未提供配置，从模型中生成配置信息

        # TODO: Remove the configuration by reference ('module')
        for module_config in self.config:
            assert isinstance(module_config, dict), (
                "config elements should be dicts not modules i.e.:"
                "[{`tensor_fqn`: `foo.bar.weight`}, {`tensor_fqn`: ... }, ...]"
            )

            assert isinstance(self.defaults, Dict)  # for mypy
            local_args = copy.deepcopy(self.defaults)  # 深拷贝默认参数
            local_args.update(module_config)  # 更新默认参数以及模块特定的配置

            tensor_fqn = local_args.get("tensor_fqn", None)  # 获取tensor_fqn，用于定位模块的特定张量
            assert tensor_fqn is not None, (
                "tensor_fqn is a required argument in the sparsity config which"
                "replaces previous `module` and [module]`fqn` arguments"
            )

            # populate all information from tensor_fqn
            info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)  # 根据tensor_fqn获取参数信息

            # check that whatever was put into local_args agrees with what was obtained
            # from tensor_fqn
            for key in info_from_tensor_fqn.keys():
                if key in local_args:
                    assert (
                        info_from_tensor_fqn[key] == local_args[key]
                        or (
                            key == "tensor_fqn"
                            and "." + info_from_tensor_fqn[key] == local_args[key]
                        )
                        # info_from_tensor_fqn will chop leading '.' from tensor_fqn so ignore that
                    ), (
                        f"Given both `{key}` and `tensor_fqn` in the config, it is expected them to agree!"
                    )

            local_args.update(info_from_tensor_fqn)  # 更新参数信息
            self.groups.append(local_args)  # 将更新后的参数添加到self.groups列表中
        self._prepare()  # 调用_prepare方法，继续准备工作

    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight"""
        for config in self.groups:
            module = config["module"]  # 获取模块
            tensor_name = config["tensor_name"]  # 获取张量名称
            parametrization = config.get("parametrization", FakeSparsity)  # 获取参数化函数，默认为FakeSparsity
            mask = config.get("mask", torch.ones_like(getattr(module, tensor_name)))  # 获取掩码，默认为全1掩码
            self.state[config["tensor_fqn"]]["mask"] = mask  # 将掩码存储到self.state中
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )  # 注册参数化方法到模块的张量上
    # 定义一个实例方法，用于压缩模型中的参数掩码
    def squash_mask(
        self,
        params_to_keep: Optional[Tuple[str, ...]] = None,
        params_to_keep_per_layer: Optional[Dict[str, Tuple[str, ...]]] = None,
        *args,
        **kwargs,
    ):
        # 方法用于将模块结构从一个模块类型转换为另一个模块类型，根据给定的映射表进行转换
        def convert(
            self,
            module: nn.Module,
            mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
            inplace: bool = False,
            parameterization: Type[nn.Module] = FakeSparsity,
        ):
            r"""Converts submodules in input module to a different module according to `mapping`
            by calling `from_dense` method on the target module class
            Args:
                module: input module
                mapping: a dictionary that maps from source module type to target
                    module type, can be overwritten to allow swapping user defined
                    Modules
                inplace: carry out model transformations in-place, the original module
                    is mutated
            """
            # 如果未提供映射表，则抛出未实现的错误信息
            if mapping is None:
                raise NotImplementedError("Need to auto generate mapping ")
            # 如果不是原地操作，则使用深拷贝复制模块
            if not inplace:
                module = copy.deepcopy(module)

            reassign = {}
            # 遍历模块的每个子模块
            for name, mod in module.named_children():
                # 如果当前子模块包含参数化信息，并且其类型在映射表中
                if (
                    module_contains_param(mod, parameterization)
                    and type_before_parametrizations(mod) in mapping
                ):
                    # 使用映射表中的目标类型替换当前子模块
                    reassign[name] = swap_module(mod, mapping)
                else:
                    # 递归调用 convert 方法转换当前子模块
                    reassign[name] = self.convert(
                        mod,
                        mapping=mapping,
                        inplace=True,
                        parameterization=parameterization,
                    )

            # 将替换后的子模块重新赋值给模块的 _modules 属性
            for key, value in reassign.items():
                module._modules[key] = value

            return module

        # 实例方法，用于处理步骤，更新模型参数掩码
        def step(self, use_path: bool = True) -> None:
            # 如果禁用参数掩码更新，则直接返回
            if not self.enable_mask_update:
                return
            # 使用 torch.no_grad 上下文管理器，禁止梯度计算
            with torch.no_grad():
                # 遍历 self.groups 中的配置信息，逐个更新参数掩码
                for config in self.groups:
                    self.update_mask(**config)

        # 抽象方法声明，要求子类实现，用于更新模型的参数掩码
        @abc.abstractmethod
        def update_mask(self, module: nn.Module, tensor_name: str, **kwargs):
            pass
```