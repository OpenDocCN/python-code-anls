# `.\pytorch\torch\ao\quantization\quantize.py`

```py
# mypy: allow-untyped-defs
# 引入需要的模块和库
import copy  # 导入 copy 模块，用于复制对象
import itertools  # 导入 itertools 模块，用于迭代器操作
import warnings  # 导入 warnings 模块，用于警告处理
import inspect  # 导入 inspect 模块，用于获取对象信息
import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.ao.nn.quantized as nnq  # 导入 PyTorch AO 模块下的量化神经网络子模块
from torch.ao.nn.intrinsic import _FusedModule  # 从 PyTorch AO 模块导入 _FusedModule 类

from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,  # 导入获取动态量化模块映射的函数
    get_default_static_quant_module_mappings,  # 导入获取静态量化模块映射的函数
    get_default_static_quant_reference_module_mappings,  # 导入获取静态量化参考模块映射的函数
    get_default_qat_module_mappings,  # 导入获取量化感知训练模块映射的函数
    get_default_qconfig_propagation_list,  # 导入获取默认量化配置传播列表的函数
    no_observer_set,  # 导入空观察者集合对象
    _has_special_act_post_process,  # 导入检查是否具有特殊激活后处理的函数
    _get_special_act_post_process,  # 导入获取特殊激活后处理对象的函数
)
from .utils import get_qparam_dict, has_no_children_ignoring_parametrizations  # 从当前包导入自定义工具函数
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper  # 从 PyTorch AO 量化存根模块导入解量化存根和量化包装器类
from torch.ao.quantization.qconfig import (
    _add_module_to_qconfig_obs_ctr,  # 导入添加模块到量化配置观察计数器的函数
    default_dynamic_qconfig,  # 导入默认动态量化配置对象
    float16_dynamic_qconfig,  # 导入浮点16位动态量化配置对象
    float_qparams_weight_only_qconfig,  # 导入仅包含权重浮点量化参数的配置对象
    float_qparams_weight_only_qconfig_4bit,  # 导入4位浮点量化参数的配置对象
    _activation_is_memoryless,  # 导入检查激活是否无记忆性的函数
)
from torch.nn.utils.parametrize import type_before_parametrizations  # 从 PyTorch 神经网络工具模块导入在参数化之前类型的函数
from torch.ao.quantization.observer import _is_activation_post_process  # 从 PyTorch AO 量化观察者模块导入检查是否为激活后处理的函数

# TODO remove this once BC is no longer required to avoid a SEV
from torch.ao.quantization.observer import (   # noqa: F401
    _is_activation_post_process as is_activation_post_process  # 导入并重命名激活后处理检查函数
)

__all__ = [  # 模块可导出的公共对象列表
    "get_default_custom_config_dict",  # 默认自定义配置字典获取函数
    "propagate_qconfig_",  # 量化配置传播函数
    "add_quant_dequant",  # 添加量化和解量化函数
    "prepare",  # 准备函数
    "quantize",  # 量化函数
    "quantize_dynamic",  # 动态量化函数
    "prepare_qat",  # 准备量化感知训练函数
    "quantize_qat",  # 量化感知训练函数
    "convert",  # 转换函数
    "swap_module",  # 模块交换函数
]

_DEFAULT_CUSTOM_CONFIG_DICT = {  # 默认自定义配置字典
    'float_to_observed_custom_module_class': {  # 浮点到观察到的自定义模块类映射
        nn.LSTM: nn.quantizable.LSTM,  # LSTM 到量化 LSTM 的映射
        nn.MultiheadAttention: nn.quantizable.MultiheadAttention,  # 多头注意力机制到量化多头注意力机制的映射
    },
    'observed_to_quantized_custom_module_class': {  # 观察到量化的自定义模块类映射
        nn.quantizable.LSTM: nn.quantized.LSTM,  # 量化 LSTM 到量化后 LSTM 的映射
        nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention,  # 量化多头注意力机制到量化后多头注意力机制的映射
    }
}

def get_default_custom_config_dict():  # 获取默认自定义配置字典的函数
    r"""Defines the default custom config dict.
    """
    return _DEFAULT_CUSTOM_CONFIG_DICT

def _propagate_qconfig_helper(module, qconfig_dict,
                              qconfig_parent=None, prefix='', prepare_custom_config_dict=None):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module  # 输入的模块
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration  # 映射子模块名称到量化配置的字典
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module  # 父模块的量化配置，当当前模块没有指定配置时会回退到此配置
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict  # 当前模块的对应前缀，在 qconfig_dict 中用作键
        prepare_custom_config_dict: dictionary for custom handling of modules
                                    see docs for :func:`~torch.ao.quantization.prepare_fx`  # 用于自定义处理模块的字典

    Return:
        None, module is modified inplace with qconfig attached  # 返回空，模块被就地修改以附加量化配置
    """
    # 根据模块类型获取对应的量化配置，若不存在则使用父级的量化配置
    module_qconfig = qconfig_dict.get(type_before_parametrizations(module), qconfig_parent)
    # 根据指定前缀获取对应的量化配置，若不存在则继续使用之前的量化配置
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    # 尝试从模块的属性中获取量化配置，若不存在则保持之前的量化配置
    module_qconfig = getattr(module, 'qconfig', module_qconfig)

    # 检验模块的量化配置是否有效，如不符合将抛出错误
    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)

    # 将模块加入到具备设备检查的量化配置中
    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(module_qconfig, module)
    # 将带有设备检查的量化配置赋值给模块的 qconfig 属性
    module.qconfig = qconfig_with_device_check

    # 遍历模块的每个子模块，为每个子模块传播量化配置
    for name, child in module.named_children():
        # 如果有前缀，则将当前子模块名加上前缀作为新的模块前缀
        module_prefix = prefix + '.' + name if prefix else name
        # 如果存在自定义准备配置字典，并且当前子模块名或类型不在非可追踪模块列表中，则传播量化配置
        if prepare_custom_config_dict is None or not (
            name in prepare_custom_config_dict.get("non_traceable_module_name", [])
            or type(child) in prepare_custom_config_dict.get("non_traceable_module_class", [])
        ):
            # 递归调用辅助函数，传播量化配置到当前子模块
            _propagate_qconfig_helper(
                child, qconfig_dict, qconfig_with_device_check, module_prefix
            )
# 传播量化配置到模块层级并在每个叶子模块上分配 `qconfig` 属性
def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    # 如果未提供 qconfig_dict，则初始化为空字典
    if qconfig_dict is None:
        qconfig_dict = {}
    # 如果未提供 prepare_custom_config_dict，则初始化为空字典
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    # 调用辅助函数来执行量化配置的传播
    _propagate_qconfig_helper(module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)

# 前向钩子，调用输出的观察器
def _observer_forward_hook(self, input, output):
    return self.activation_post_process(output)

# 前向预钩子，调用输入的观察器
def _observer_forward_pre_hook(self, input):
    return self.activation_post_process(input[0])

# 注册激活后处理钩子，用于模块上的激活后处理
def _register_activation_post_process_hook(module, pre_hook=False):
    # 断言模块上已经附加了 'activation_post_process' 属性
    assert hasattr(module, 'activation_post_process'), \
        'Expect activation_post_process attribute already attached to the module'
    # 根据 pre_hook 参数注册前向预钩子或前向钩子
    if pre_hook:
        handle = module.register_forward_pre_hook(
            _observer_forward_pre_hook, prepend=True
        )
    else:
        handle = module.register_forward_hook(
            _observer_forward_hook, prepend=True
        )

# 为模块的叶子子模块添加观察器
def _add_observer_(module, qconfig_propagation_list=None, non_leaf_module_list=None, device=None, custom_module_class_mapping=None):
    # 如果未提供 qconfig_propagation_list，则使用默认的量化配置传播列表
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    # 如果未提供 custom_module_class_mapping，则初始化为空字典
    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    # 在添加观察器时尊重设备亲和性
    # 如果设备为 None，则获取模块的唯一设备列表
    devices = _get_unique_devices_(module)
    # 断言设备数量不超过一个，否则抛出异常
    assert len(devices) <= 1, (
        f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
    )
    # 如果设备列表不为空，则选择第一个设备作为当前设备
    device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        # 根据配置获取激活函数后处理模块
        activation = qconfig.activation() if special_act_post_process is None else special_act_post_process()
        # 如果设备不为 None，则将激活函数后处理模块移动到指定设备
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        # 判断模块是否需要被观察（是否具有 qconfig 属性并且不为 None）
        return hasattr(m, 'qconfig') and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        """ Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        # 不为 DeQuantStub 类型的模块添加激活函数后处理模块
        if needs_observation(m) and not isinstance(m, DeQuantStub):
            # 将激活函数后处理模块添加到模块中
            m.add_module('activation_post_process', get_activation_post_process(
                m.qconfig, device, special_act_post_process))
            # 将观察者注册为钩子列表的第一个条目
            # 所有后向钩子都会在观察者之后执行，然后才执行转换操作
            _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))
    # 遍历模块中的每个子模块，获取子模块的名称和对象
    for name, child in module.named_children():
        # TODO remove Dropout special after codebase stable
        # 如果子模块的类型是 nn.Dropout，跳过处理
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        # 如果子模块的类型是 nnq.FloatFunctional 或 nnq.QFunctional
        elif issubclass(type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)):
            # 如果需要观察该子模块
            if needs_observation(child):
                # 确保该 functional 类有预定义的 `activation_post_process`
                assert hasattr(child, "activation_post_process"), (
                    f"functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`"
                )
                # 设置子模块的 activation_post_process 属性
                child.activation_post_process = get_activation_post_process(child.qconfig, device)
        # 如果子模块是 _FusedModule 类的实例
        elif isinstance(child, _FusedModule):
            # 如果需要观察该子模块，插入 activation_post_process
            if needs_observation(child):
                insert_activation_post_process(child)
        # 如果非叶子模块列表非空，并且子模块的类型在非叶子模块列表中
        elif non_leaf_module_list is not None and type_before_parametrizations(child) in non_leaf_module_list:
            # 如果需要观察该子模块，插入 activation_post_process
            if needs_observation(child):
                insert_activation_post_process(child)
        # 如果子模块有特殊的 activation_post_process
        elif _has_special_act_post_process(child):
            # 获取特殊的 activation_post_process
            special_act_post_process = _get_special_act_post_process(child)
            # 向子模块插入 activation_post_process
            insert_activation_post_process(child, special_act_post_process)
        # 如果需要观察该子模块，并且该子模块的类型在自定义模块类映射中
        elif needs_observation(child) and type_before_parametrizations(child) in custom_module_class_mapping:
            # 使用 float 模型转换方法从浮点数模型创建观察子模块
            observed_child = custom_module_class_mapping[type_before_parametrizations(child)].from_float(child)
            # 将观察后的子模块设置回原始模块中
            setattr(module, name, observed_child)
            # 如果自定义模块类不在无观察器集合中，插入 activation_post_process
            if custom_module_class_mapping[type_before_parametrizations(child)] not in no_observer_set():
                insert_activation_post_process(observed_child)
        else:
            # 为子模块添加观察器
            _add_observer_(child, qconfig_propagation_list, non_leaf_module_list, device, custom_module_class_mapping)

    # 仅为叶子节点插入观察器，注意这个观察器用于模块的输出，在输入上 QuantStub 将会观察
    if has_no_children_ignoring_parametrizations(module) and not isinstance(module, torch.nn.Sequential) \
       and type_before_parametrizations(module) in qconfig_propagation_list:
        # 向模块插入 activation_post_process
        insert_activation_post_process(module)

    # 这是 AdaRound eager 模式的特殊情况
    # AdaRound 包含 weight_fake_quant，需要从 API 转换并传播给 leaf node
    # 对于包含多个子节点的假设看起来有些幼稚，因此加入 AdaRound 的例外情况
    if hasattr(module, "weight_fake_quant") and not isinstance(module, torch.nn.Sequential) \
       and type_before_parametrizations(module) in qconfig_propagation_list:
        # 向模块插入 activation_post_process
        insert_activation_post_process(module)
def _get_unique_devices_(module):
    # 返回一个集合，包含所有参数的设备类型和所有缓冲区的设备类型
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}

def add_quant_dequant(module):
    r"""如果具有有效的qconfig，则将叶子子模块包装在QuantWrapper中
    注意，此函数会就地修改模块的子模块，并且可能返回一个包装输入模块的新模块。

    Args:
        module: 具有所有叶子模块的qconfig属性的输入模块，我们希望量化它们

    Return:
        可能是就地修改后的模块，其中子模块基于qconfig被包装在`QuantWrapper`中，或者是一个新的`QuantWrapper`模块，它包装了输入模块；
        后一种情况仅在输入模块是叶子模块且我们希望量化它时发生。
    """
    # 如果模块没有子模块并且具有'qconfig'属性且qconfig不为空
    if has_no_children_ignoring_parametrizations(module) and hasattr(module, 'qconfig') and module.qconfig:
        # 返回一个QuantWrapper对象，将输入模块包装起来
        return QuantWrapper(module)

    # 遍历模块的所有子模块
    for name, child in module.named_children():
        # 递归调用add_quant_dequant函数，修改子模块并更新到父模块中
        module._modules[name] = add_quant_dequant(child)
    return module

def prepare(model, inplace=False, allow_list=None,
            observer_non_leaf_module_list=None,
            prepare_custom_config_dict=None):
    r"""为量化校准或量化感知训练准备模型的副本。

    量化配置应事先分配给每个子模块的`.qconfig`属性。

    模型将附加观察器或假量化模块，并传播qconfig。

    Args:
        `model`: 要就地修改的输入模型
        `inplace`: 是否就地执行模型转换，原始模块会发生变异
        `allow_list`: 可量化模块的列表
        `observer_non_leaf_module_list`: 我们希望为其添加观察器的非叶子模块列表
        `prepare_custom_config_dict`: prepare函数的自定义配置字典

    .. code-block:: python

       # prepare_custom_config_dict的示例:
       prepare_custom_config_dict = {
           # 用户将手动定义相应的观察模块类，具有from_float类方法，将浮点自定义模块转换为观察自定义模块
           "float_to_observed_custom_module_class": {
               CustomModule: ObservedCustomModule
           }
        }

    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare")
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = prepare_custom_config_dict.get("float_to_observed_custom_module_class", {})

    # 如果不是就地操作，则对模型进行深拷贝
    if not inplace:
        model = copy.deepcopy(model)

    # TODO: 移除allow_list
    qconfig_propagation_list = allow_list
    if allow_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    
    # 传播qconfig到模型中的所有子模块
    propagate_qconfig_(model, qconfig_dict=None)
    # 检查常见的 API 使用错误
    # 遍历模型的所有模块，检查是否有任何模块具有 'qconfig' 属性且不为 None
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        # 如果没有任何子模块应用了 qconfig，请发出警告
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")

    # 将观察器添加到模型中
    _add_observer_(
        model, qconfig_propagation_list, observer_non_leaf_module_list,
        custom_module_class_mapping=custom_module_class_mapping)
    
    # 返回修改后的模型
    return model
# 清除模块中的 activation_post_process 相关设置，防止用户误用
def _remove_activation_post_process(module):
    # 如果模块具有 activation_post_process 属性，并且该属性是有效的激活后处理函数
    if hasattr(module, 'activation_post_process') and \
       _is_activation_post_process(module.activation_post_process):
        # 删除 activation_post_process 属性
        delattr(module, 'activation_post_process')

    # 移除 activation_post_process 的前后钩子函数
    def remove_hooks(pre_hook=False):
        # 根据 pre_hook 标志选择钩子映射
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        # 根据 pre_hook 选择观察者钩子函数
        observer_hook = _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        # 要移除的处理句柄 ID 集合
        handle_ids_to_remove = set()
        # 遍历钩子映射，查找观察者钩子函数
        for handle_id, hook_fn in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        # 移除找到的处理句柄
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

    # 移除前向预处理的钩子函数
    remove_hooks(pre_hook=True)
    # 移除后向预处理的钩子函数
    remove_hooks(pre_hook=False)

# 重命名为更一般化的名称，清除模块中的 qconfig 设置，以便可以传播新的 qconfig
def _remove_qconfig(module):
    r"""清除模块中遗留的 qconfig，以便可以传播新的 qconfig。

    Args:
        module: 待清理的模块
    """
    # 递归地对模块的子模块应用 _remove_qconfig 函数
    for child in module.children():
        _remove_qconfig(child)

    # 如果模块具有 "qconfig" 属性，则删除该属性
    if hasattr(module, "qconfig"):
        del module.qconfig

    # 清除模块中的 activation_post_process 相关设置
    _remove_activation_post_process(module)

# 对输入的浮点模型进行后训练静态量化
def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    r"""对输入的浮点模型进行后训练静态量化。

    首先准备模型进行校准，然后调用 `run_fn` 来运行校准步骤，之后将模型转换为量化模型。

    Args:
        model: 输入的浮点模型
        run_fn: 用于校准准备模型的校准函数
        run_args: `run_fn` 的位置参数
        inplace: 是否原地进行模型转换，即是否对原模块进行变异
        mapping: 原始模块类型与量化后模块类型之间的对应关系

    Return:
        量化后的模型。
    """
    # 记录 API 使用情况
    torch._C._log_api_usage_once("quantization_api.quantize.quantize")
    # 如果未提供 mapping，则使用默认的静态量化模块映射
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    # 如果不是原地操作，则深拷贝一份模型
    if not inplace:
        model = copy.deepcopy(model)
    # 将模型设为评估模式
    model.eval()
    # 准备模型进行量化
    prepare(model, inplace=True)
    # 运行校准函数，校准模型
    run_fn(model, *run_args)
    # 将模型转换为量化模型，使用给定的映射
    convert(model, mapping, inplace=True)
    # 返回量化后的模型
    return model

# 将浮点模型转换为动态（仅权重）量化模型
def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8,
                     mapping=None, inplace=False):
    r"""将浮点模型转换为动态（仅权重）量化模型。

    替换指定模块为动态仅权重量化版本，并输出量化后的模型。

    对于最简单的使用方式，请提供 `dtype` 参数，可以是 float16 或 qint8。默认情况下，仅对具有大权重尺寸的层进行仅权重量化 - 例如线性层和各种 RNN 变体。

    Args:
        model: 输入的浮点模型
        qconfig_spec: 量化配置的规范
        dtype: 量化后的数据类型，默认为 torch.qint8
        mapping: 原始模块类型与量化后模块类型之间的对应关系
        inplace: 是否原地进行模型转换，即是否对原模块进行变异
    """
    """
    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
            输入的模型
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfig instances.
            - 一个字典，将子模块的名称或类型映射到量化配置，qconfig适用于给定模块的所有子模块，除非为子模块指定了qconfig（当子模块已经有qconfig属性时）。字典条目需要是QConfig实例。

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width
            - 一组类型和/或子模块名称，用于应用动态量化，此时使用`dtype`参数来指定位宽度

        inplace: carry out model transformations in-place, the original module is mutated
            在原地执行模型转换，修改原始模块
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced
            将子模块的类型映射到相应的动态量化版本的类型，用于替换子模块
    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_dynamic")
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {
                nn.Linear : default_dynamic_qconfig,
                nn.LSTM : default_dynamic_qconfig,
                nn.GRU : default_dynamic_qconfig,
                nn.LSTMCell : default_dynamic_qconfig,
                nn.RNNCell : default_dynamic_qconfig,
                nn.GRUCell : default_dynamic_qconfig,
            }
        elif dtype == torch.float16:
            qconfig_spec = {
                nn.Linear : float16_dynamic_qconfig,
                nn.LSTM : float16_dynamic_qconfig,
                nn.GRU : float16_dynamic_qconfig,
                nn.LSTMCell : float16_dynamic_qconfig,
                nn.RNNCell : float16_dynamic_qconfig,
                nn.GRUCell : float16_dynamic_qconfig,
            }
        elif dtype == torch.quint8:
            qconfig_spec = {
                nn.EmbeddingBag : float_qparams_weight_only_qconfig,
                nn.Embedding : float_qparams_weight_only_qconfig,
            }
        elif dtype == torch.quint4x2:
            qconfig_spec = {
                nn.EmbeddingBag : float_qparams_weight_only_qconfig_4bit,
            }
        else:
            raise ValueError(
                f"Don't know how to quantize with default settings for {dtype}. Provide full qconfig please")
    # 如果 qconfig_spec 是 set 类型，则根据 dtype 设置默认的量化配置
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            # 如果数据类型是 torch.qint8，使用默认的动态量化配置
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            # 如果数据类型是 torch.float16，使用 float16 的动态量化配置
            default_qconfig = float16_dynamic_qconfig
        elif dtype is torch.quint8:
            # 如果数据类型是 torch.quint8，使用仅量化参数权重的配置
            default_qconfig = float_qparams_weight_only_qconfig
        elif dtype is torch.quint4x2:
            # 如果数据类型是 torch.quint4x2，使用 4 位的仅量化参数权重的配置
            default_qconfig = float_qparams_weight_only_qconfig_4bit
        else:
            # 如果指定了未知的 dtype，抛出运行时异常
            raise RuntimeError('Unknown dtype specified for quantize_dynamic: ', str(dtype))
        # 使用 itertools.repeat 将默认的量化配置映射到 qconfig_spec 中的每个元素
        qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))

    # 如果 mapping 为 None，则获取默认的动态量化模块映射
    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()

    # 如果不是原地操作（inplace=False），则深拷贝模型
    if not inplace:
        model = copy.deepcopy(model)
    
    # 将模型设为评估模式
    model.eval()

    # 将量化配置应用于模型
    propagate_qconfig_(model, qconfig_spec)

    # 对模型进行转换以应用量化
    convert(model, mapping, inplace=True)

    # 返回量化后的模型
    return model
def prepare_qat(model, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    # 记录 API 使用情况到 PyTorch 事件日志
    torch._C._log_api_usage_once("quantization_api.quantize.prepare_qat")
    
    # 断言模型处于训练模式
    assert model.training, "prepare_qat only works on models in training mode"
    
    # 如果未提供映射关系，则使用默认的量化感知训练模块映射
    if mapping is None:
        mapping = get_default_qat_module_mappings()

    # 如果不是在原地操作，创建模型的深层副本
    if not inplace:
        model = copy.deepcopy(model)

    # 传播量化配置至模型的各个子模块
    propagate_qconfig_(model, qconfig_dict=None)
    
    # 将模型转换为量化版本，使用提供的映射关系，并在原地进行转换
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    
    # 准备模型以进行后续量化感知训练，设置观察器以跟踪非叶子模块
    prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
    
    # 返回准备好的模型
    return model

def quantize_qat(model, run_fn, run_args, inplace=False):
    r"""Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
                function that simply runs the prepared model or a training
                loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
    # 记录 API 使用情况到 PyTorch 事件日志
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_qat")
    
    # 如果不是在原地操作，创建模型的深层副本
    if not inplace:
        model = copy.deepcopy(model)
    
    # 设置模型处于训练模式
    model.train()
    
    # 准备模型以进行量化感知训练，修改模型本身
    prepare_qat(model, inplace=True)
    
    # 执行指定的运行函数来评估准备好的模型
    run_fn(model, *run_args)
    
    # 将模型转换为量化版本，并在原地进行转换
    convert(model, inplace=True)
    
    # 返回量化后的模型
    return model

def convert(
        module, mapping=None, inplace=False, remove_qconfig=True,
        is_reference=False, convert_custom_config_dict=None,
        use_precomputed_fake_quant=False):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class. And remove qconfig at the
    end if remove_qconfig is set to True.

    Args:
        `module`: prepared and calibrated module
        `mapping`: a dictionary that maps from source module type to target
                   module type, can be overwritten to allow swapping user defined
                   Modules
        `inplace`: carry out model transformations in-place, the original module
                   is mutated
        `convert_custom_config_dict`: custom configuration dictionary for convert function
        `use_precomputed_fake_quant`: a flag to enable use of precomputed fake quant
    """
    # 此函数的详细文档注释见函数定义处
    # Example of convert_custom_config_dict:
    # convert_custom_config_dict 是一个字典，用于配置自定义模块的量化转换
    convert_custom_config_dict = {
        # 用户将手动定义对应的量化模块类，该类具有 from_observed 类方法，用于将观察到的自定义模块转换为量化自定义模块
        # ObservedCustomModule 类对应 QuantizedCustomModule 类
        "observed_to_quantized_custom_module_class": {
            ObservedCustomModule: QuantizedCustomModule
        }
    }
    
    """
    torch._C._log_api_usage_once("quantization_api.quantize.convert")
    在调用一次之后记录 PyTorch 的 API 使用情况，用于量化 API 的转换操作
    """
    if not inplace:
        # 如果不是原地操作，深拷贝模块
        module = copy.deepcopy(module)
    # 调用 _convert 函数，进行模块的量化转换操作
    _convert(
        module, mapping, inplace=True, is_reference=is_reference,
        convert_custom_config_dict=convert_custom_config_dict,
        use_precomputed_fake_quant=use_precomputed_fake_quant)
    if remove_qconfig:
        # 如果需要移除量化配置信息，则调用 _remove_qconfig 函数
        _remove_qconfig(module)
    # 返回转换后的模块
    return module
# 将输入模块中的子模块按照给定的映射转换为不同的模块，调用目标模块类的 `from_float` 方法来实现转换

def _convert(
        module, mapping=None, inplace=False,
        is_reference=False, convert_custom_config_dict=None,
        use_precomputed_fake_quant=False):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class

    Args:
        module: input module  输入的模块
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules  映射字典，将源模块类型映射到目标模块类型，可以被覆盖以允许交换用户定义的模块
        inplace: carry out model transformations in-place, the original module
                 is mutated  是否原地操作模型转换，会改变原始模块
        is_reference: a flag to enable quantized reference module  启用量化参考模块的标志
        use_precomputed_fake_quant: a flag to enable use of precomputed fake quant  启用预计算的伪量化使用标志

    """
    if mapping is None:
        # 如果映射为空，则根据是否参考来获取默认的静态量化参考模块映射或默认的静态量化模块映射
        mapping = get_default_static_quant_reference_module_mappings() if is_reference \
            else get_default_static_quant_module_mappings()
    if convert_custom_config_dict is None:
        # 如果自定义配置字典为空，则获取默认的自定义配置字典
        convert_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = convert_custom_config_dict.get("observed_to_quantized_custom_module_class", {})

    if not inplace:
        # 如果不是原地操作，则深拷贝模块
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        # 对于每个子模块，如果不是融合模块且其类型不在自定义模块映射中，则递归调用 _convert 进行转换
        if not isinstance(mod, _FusedModule) and \
           type_before_parametrizations(mod) not in custom_module_class_mapping:
            _convert(mod, mapping, True,  # inplace
                     is_reference, convert_custom_config_dict,
                     use_precomputed_fake_quant=use_precomputed_fake_quant)
        # 对模块进行交换，得到新的模块
        reassign[name] = swap_module(mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant)

    for key, value in reassign.items():
        # 更新模块的子模块
        module._modules[key] = value

    return module

def swap_module(mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant=False):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module  输入的模块
        mapping: a dictionary that maps from nn module to nnq module  映射字典，将 nn 模块映射到 nnq 模块
        custom_module_class_mapping: mapping of observed custom module classes to quantized versions  观察到的自定义模块类到量化版本的映射
        use_precomputed_fake_quant: a flag to enable precomputed fake quantization  启用预计算的伪量化标志

    Return:
        The corresponding quantized module of `mod`  `mod` 对应的量化模块
    """
    new_mod = mod
    # 检查模块是否具有属性 'qconfig' 并且该属性不为 None
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        # 标记是否已经进行了模块替换
        swapped = False
        
        # 如果模块在参数化之前的类型在自定义模块类映射中
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            # 根据自定义映射将模块转换为观察模式的新模块
            new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_observed(mod)
            swapped = True
        
        # 如果模块在参数化之前的类型在标准映射中
        elif type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            
            # 如果量化模块具有 '_IS_REFERENCE' 属性并且为真
            if hasattr(qmod, '_IS_REFERENCE') and qmod._IS_REFERENCE:
                # 断言模块的 qconfig 不为 None
                assert mod.qconfig is not None
                # 获取权重后处理函数
                weight_post_process = mod.qconfig.weight()
                # 将权重数据应用到权重后处理函数中
                weight_post_process(mod.weight)
                # 获取量化参数字典
                weight_qparams = get_qparam_dict(weight_post_process)
                # 将模块转换为浮点数表示到量化表示的新模块
                new_mod = qmod.from_float(mod, weight_qparams)
            else:
                # 检查量化模块的 from_float 方法的参数签名
                sig = inspect.signature(qmod.from_float)
                # 如果方法支持 use_precomputed_fake_quant 参数
                if 'use_precomputed_fake_quant' in sig.parameters:
                    # 使用预先计算的伪量化参数将模块转换为量化表示的新模块
                    new_mod = qmod.from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
                else:
                    # 将模块转换为量化表示的新模块（默认情况）
                    new_mod = qmod.from_float(mod)
            swapped = True
        
        # 如果进行了模块替换
        if swapped:
            # 保留模块的前向预处理钩子，它们将在量化输入时被调用
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            
            # 保留模块的后向前向钩子，但排除 _observer_forward_hook
            # 在转换后，它们将与量化输出一起工作
            for hook_fn in mod._forward_hooks.values():
                if hook_fn is not _observer_forward_hook:
                    new_mod.register_forward_hook(hook_fn)
            
            # 在替换模块时保留设备关联性
            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1, (
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
            # 获取模块的设备（CPU 或单设备 CUDA）
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                # 将新模块移动到相同的设备上
                new_mod.to(device)
    
    # 返回替换后的模块
    return new_mod
# 递归函数，用于遍历模块并将所有观察器保存到字典中
def _get_observer_dict(mod, target_dict, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """
    # 定义一个函数，用于获取当前模块的前缀，如果有的话
    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + '.'

    # 如果模块具有 'activation_post_process' 属性，则将其加入目标字典
    if hasattr(mod, 'activation_post_process'):
        target_dict[get_prefix(prefix) + 'activation_post_process'] = mod.activation_post_process
    # 遍历模块的所有子模块
    for name, child in mod.named_children():
        # 计算当前子模块的完整前缀
        module_prefix = get_prefix(prefix) + name if prefix else name
        # 递归调用 _get_observer_dict 函数，处理当前子模块
        _get_observer_dict(child, target_dict, module_prefix)
```