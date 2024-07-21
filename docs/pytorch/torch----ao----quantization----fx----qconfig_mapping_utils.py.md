# `.\pytorch\torch\ao\quantization\fx\qconfig_mapping_utils.py`

```py
# 导入必要的模块和函数，包括torch、re、defaultdict、OrderedDict等
# mypy: allow-untyped-defs 用于指示mypy允许未标记类型的函数
import torch
import re
from collections import defaultdict, OrderedDict
from typing import Callable, Any, Dict, Tuple, Set, List, Union
# 导入torch.quantization模块中的QConfig相关内容
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
# 导入torch.quantization.observer模块中的_is_activation_post_process函数
from torch.ao.quantization.observer import (
    _is_activation_post_process,
)
# 导入torch.quantization.backend_config模块中的BackendConfig和DTypeConfig类
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
)
# 导入torch.quantization.backend_config.utils模块中的get_module_to_qat_module函数
from torch.ao.quantization.backend_config.utils import (
    get_module_to_qat_module,
)

# 导入torch.fx模块中的GraphModule类和Graph类
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
)
# 导入torch.ao.nn.intrinsic模块中的_FusedModule类
from torch.ao.nn.intrinsic import _FusedModule

# 导入上层目录中的_utils模块中的_parent_name和get_qconfig_dtypes函数
from ..utils import (
    _parent_name,
    get_qconfig_dtypes,
)
# 导入上层目录中的qconfig_mapping模块中的相关变量和类
from ..qconfig_mapping import (
    _OBJECT_TYPE_DICT_KEY,
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    QConfigMapping,
)

# 定义一个空列表，用于指示模块中公开的变量名
__all__: List[str] = []


def _maybe_adjust_qconfig_for_module_name_object_type_order(
    qconfig_mapping: QConfigMapping,
    cur_module_path: str,
    cur_object_type: Callable,
    cur_object_type_idx: int,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    """
    根据模块名称、对象类型和对象类型索引可能调整QConfig，以便按顺序排序。
    """
    # 遍历QConfigMapping中保存的模块名称、对象类型和索引对应的QConfig字典
    for (module_name, object_type, index), qconfig in qconfig_mapping.module_name_object_type_order_qconfigs.items():
        # 如果找到匹配的模块名称、对象类型和索引，则返回对应的QConfig
        if (
            (module_name == cur_module_path) and
            (object_type == cur_object_type) and
            (index == cur_object_type_idx)
        ):
            return qconfig
    # 如果没有找到匹配项，则返回默认的QConfig
    return fallback_qconfig


def _update_qconfig_for_fusion(model: GraphModule, qconfig_mapping: QConfigMapping):
    """
    更新QConfigMapping以考虑如LinearReLU等融合模块。
    这假设QConfigMapping的属性已经转换为OrderedDict。
    """
    # 获取QConfigMapping中的对象类型到QConfig的字典
    object_type_dict = qconfig_mapping.object_type_qconfigs
    # 如果对象类型字典为空，则直接返回QConfigMapping
    if len(object_type_dict) == 0:
        return qconfig_mapping

    # 获取模型中所有模块的命名字典
    modules = dict(model.named_modules())
    # 遍历模型的图中的每个节点
    for node in model.graph.nodes:
        # 检查节点操作是否为 'call_module'，且目标模块在给定的模块集合中
        if node.op == 'call_module' and node.target in modules:
            # 获取可能的融合模块
            maybe_fused_module = modules[str(node.target)]
            # 如果不是 _FusedModule 类型，则跳过
            if not isinstance(maybe_fused_module, _FusedModule):
                continue
            
            # 获取融合模块中的操作列表
            ops = list(maybe_fused_module._modules.values())
            # 获取第一个操作的类型对应的 qconfig
            fused_qconfig = object_type_dict.get(type(ops[0]), None)

            # 如果融合模块中的其他操作与第一个操作的 qconfig 不一致，则抛出错误
            for op in ops[1:]:
                if not qconfig_equals(object_type_dict.get(type(op), None), fused_qconfig):
                    raise LookupError(
                        "During fusion, we need to specify the same " +
                        f"qconfigs for all module types in {type(maybe_fused_module)} " +
                        f"offending type: {type(op)}")

            # 如果融合模块的 qconfig 不为 None，则将融合模块类型与其 qconfig 关联起来
            if fused_qconfig is not None:
                object_type_dict[type(maybe_fused_module)] = fused_qconfig
# 生成节点名称到量化配置的映射字典
def _generate_node_name_to_qconfig(
        root: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
        input_graph: Graph,
        qconfig_mapping: QConfigMapping,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> Dict[str, QConfigAny]:
    # 获取全局量化配置
    global_qconfig = qconfig_mapping.global_qconfig
    # 初始化节点名称到量化配置的空字典
    node_name_to_qconfig = {}

    # 用于记录子模块到对象类型（函数）到当前索引的字典
    # 示例:
    #
    #   {'foo.bar': {F.linear: 0, F.conv2d: 1, ...}, ...}
    #
    # 意味着在子模块 'foo.bar' 中，我们已经看到了0次 F.linear 和 1次 F.conv2d 的调用。
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = \
        defaultdict(lambda: defaultdict(int))
    
    # 返回节点名称到量化配置的空字典
    return node_name_to_qconfig


# 检查给定的配置字典是否具有正确的键
def _check_is_valid_config_dict(config_dict: Any, allowed_keys: Set[str], dict_name: str) -> None:
    r""" 检查给定的 config_dict 是否具有正确的键

    Args:
      `config_dict`: 要检查键的字典
    """
    for k in config_dict.keys():
        if k not in allowed_keys:
            raise ValueError(
                'Expected ' + dict_name + ' to have the following keys: ' +
                str(allowed_keys) + '. But found \'' + k +
                '\' instead.')


# 比较和准备转换量化配置映射
def _compare_prepare_convert_qconfig_mappings(
        prepare_qconfig_mapping: QConfigMapping,
        convert_qconfig_mapping: QConfigMapping):
    r""" 比较在转换中传递的 qconfig_mapping 与准备中的并检查其值

    Args:
      `prepare_qconfig_mapping`: 准备量化步骤的配置
      `convert_qconfig_mapping`: 转换量化步骤的配置
    """
    # 断言预处理和转换配置中的全局量化配置相同
    assert qconfig_equals(prepare_qconfig_mapping.global_qconfig, convert_qconfig_mapping.global_qconfig), \
        "Expected global qconfigs to be the same in the prepare and convert quantization configs"
    
    # 准备和转换配置中的对象类型、模块名称和模块名称正则表达式量化配置字典
    prepare_dicts: List[OrderedDict] = [
        prepare_qconfig_mapping.object_type_qconfigs,
        prepare_qconfig_mapping.module_name_qconfigs,
        prepare_qconfig_mapping.module_name_regex_qconfigs,
    ]
    convert_dicts: List[OrderedDict] = [
        convert_qconfig_mapping.object_type_qconfigs,
        convert_qconfig_mapping.module_name_qconfigs,
        convert_qconfig_mapping.module_name_regex_qconfigs,
    ]
    dict_names = [_OBJECT_TYPE_DICT_KEY, _MODULE_NAME_DICT_KEY, _MODULE_NAME_REGEX_DICT_KEY]
    
    # 检查每个预处理字典中的键是否存在于转换字典中，并比较其量化配置
    for i in range(len(prepare_dicts)):
        for name in prepare_dicts[i].keys():
            assert name in convert_dicts[i], f"Missing key {dict_names[i]} {name} in convert QConfigMapping \
                when it was present in prepare"
            assert convert_dicts[i][name] is None \
                or qconfig_equals(prepare_dicts[i][name], convert_dicts[i][name]), \
                f"Expected convert QConfigMapping to have the same qconfig as prepare for key {dict_names[i]} {name}; \
                prepare: {prepare_dicts[i][name]}; convert: {convert_dicts[i][name]}"
# 检查给定的量化配置是否受到指定数据类型配置列表支持
def _is_qconfig_supported_by_dtype_configs(qconfig: QConfig, dtype_configs: List[DTypeConfig]):
    # 遍历数据类型配置列表中的每个配置
    for dtype_config in dtype_configs:
        # 获取是否为动态配置，如果未指定则默认为 False
        is_dynamic = dtype_config.is_dynamic
        if is_dynamic is None:
            is_dynamic = False
        # 获取输入数据类型，如果未指定则默认为 torch.float
        input_dtype = dtype_config.input_dtype or torch.float
        # 获取权重数据类型，如果未指定则默认为 torch.float
        weight_dtype = dtype_config.weight_dtype or torch.float
        # 获取偏置数据类型，如果未指定则默认为 torch.float
        bias_dtype = dtype_config.bias_dtype or torch.float
        # 获取输出数据类型，如果未指定则默认为 torch.float
        output_dtype = dtype_config.output_dtype or torch.float
        # 获取量化配置的激活数据类型、权重数据类型及是否为动态输入激活的配置
        qconfig_activation_dtype, qconfig_weight_dtype, qconfig_input_act_is_dynamic = \
            get_qconfig_dtypes(qconfig)
        # 根据条件确定量化配置的偏置数据类型
        qconfig_bias_dtype = torch.float16 \
            if (
                qconfig_activation_dtype == torch.float16
                and qconfig_weight_dtype == torch.float16
                and not is_dynamic
            ) else torch.float

        # 根据是否为动态配置来判断是否匹配
        if is_dynamic:
            is_match = qconfig_input_act_is_dynamic and \
                input_dtype == qconfig_activation_dtype and \
                output_dtype == torch.float and \
                weight_dtype == qconfig_weight_dtype
        else:
            is_match = input_dtype == qconfig_activation_dtype and \
                output_dtype == qconfig_activation_dtype and \
                weight_dtype == qconfig_weight_dtype and \
                bias_dtype == qconfig_bias_dtype
        # 如果匹配，则返回 True
        if is_match:
            return True
    # 如果没有任何匹配，则返回 False
    return False

# 获取对象类型的量化配置，如果找不到则返回默认的量化配置
def _get_object_type_qconfig(
        qconfig_mapping: QConfigMapping,
        object_type: Union[Callable, str],
        fallback_qconfig: QConfigAny) -> QConfigAny:
    return qconfig_mapping.object_type_qconfigs.get(object_type, fallback_qconfig)

# 根据模块名的正则表达式模式获取对应的量化配置，如果没有匹配则返回默认的量化配置
def _get_module_name_regex_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    for regex_pattern, qconfig in qconfig_mapping.module_name_regex_qconfigs.items():
        # 如果模块名与正则表达式模式匹配，则返回对应的量化配置
        if re.match(regex_pattern, module_name):
            # 第一个匹配成功的优先返回
            return qconfig
    # 如果没有任何匹配，则返回默认的量化配置
    return fallback_qconfig

# 根据模块名获取对应的量化配置，如果模块名为空则返回默认的量化配置，如果没有找到则递归地向上查找父模块的量化配置
def _get_module_name_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    if module_name == '':
        # 如果模块名为空，则返回默认的量化配置
        return fallback_qconfig
    # 如果在映射中找到了模块名对应的量化配置，则返回该配置
    if module_name in qconfig_mapping.module_name_qconfigs:
        return qconfig_mapping.module_name_qconfigs[module_name]
    else:
        # 否则，获取父模块的名称，并递归地继续查找量化配置
        parent, _ = _parent_name(module_name)
        return _get_module_name_qconfig(qconfig_mapping, parent, fallback_qconfig)

# 可能根据模块类型或名称调整量化配置，优先获取模块名称的量化配置，其次是正则表达式匹配的量化配置，然后是全局量化配置
def _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_name, global_qconfig):
    # 获取模块类型对应的量化配置，如果找不到则使用全局的量化配置
    module_type_qconfig = _get_object_type_qconfig(
        qconfig_mapping, module_type, global_qconfig)
    # 获取模块名称对应的正则表达式匹配的量化配置，如果没有匹配则使用模块类型的量化配置
    module_name_regex_qconfig = _get_module_name_regex_qconfig(
        qconfig_mapping, module_name, module_type_qconfig)
    # 调用函数 _get_module_name_qconfig，根据给定的 qconfig_mapping、module_name 和 module_name_regex_qconfig 参数获取模块名及其对应的量化配置
    module_name_qconfig = _get_module_name_qconfig(
        qconfig_mapping, module_name, module_name_regex_qconfig)
    # 返回从 _get_module_name_qconfig 函数得到的模块名及其对应的量化配置
    return module_name_qconfig
def _get_flattened_qconfig_dict(qconfig_mapping: QConfigMapping) -> Dict[Union[Callable, str], QConfigAny]:
    """ 
    将全局的、对象类型和模块名称的量化配置展平到同一个 qconfig_dict 中，
    以便 propagate_qconfig_ 函数可以使用。

    "module_name_regex" 目前不支持 propagate_qconfig_，因此在此忽略，
    但可以以后修复。

    例如：
    输入: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    输出: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    # 初始化一个空的展平字典，包含全局 qconfig
    flattened: Dict[Union[Callable, str], QConfigAny] = {"": qconfig_mapping.global_qconfig}
    
    # 遍历对象类型的 qconfig 映射，将每个对象和对应的 qconfig 添加到展平字典中
    for obj, qconfig in qconfig_mapping.object_type_qconfigs.items():
        flattened[obj] = qconfig
    
    # 遍历模块名称的 qconfig 映射，将每个模块名称和对应的 qconfig 添加到展平字典中
    for obj, qconfig in qconfig_mapping.module_name_qconfigs.items():
        flattened[obj] = qconfig
    
    # 返回展平后的 qconfig 字典
    return flattened


def _update_qconfig_for_qat(
        qconfig_mapping: QConfigMapping,
        backend_config: BackendConfig):
    """
    更新 qconfig_mapping，以考虑在 QAT 过程中的模块交换。
    在 QAT 过程中，我们将 nn.Module 类型的模块交换为相应的 nn.qat.modules 类型。
    """
    # 获取模块到 QAT 模块的映射
    module_to_qat_module_class = get_module_to_qat_module(backend_config)
    
    # 复制对象类型字典，避免在迭代时修改原始字典
    object_type_dict = qconfig_mapping.object_type_qconfigs
    new_object_type_dict = object_type_dict.copy()
    
    # 遍历复制后的对象类型字典
    for k, v in new_object_type_dict.items():
        # 如果对象在映射中存在对应的 QAT 模块，则更新对象类型字典中的对应项
        if k in module_to_qat_module_class:
            object_type_dict[module_to_qat_module_class[k]] = v
```