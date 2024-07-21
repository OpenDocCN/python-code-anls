# `.\pytorch\torch\ao\ns\fx\weight_utils.py`

```
# 导入 PyTorch 库中所需的模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq

# 引入 quantized 操作符
toq = torch.ops.quantized

# 导入 Torch 的 GraphModule 和 Node 类
from torch.fx import GraphModule
from torch.fx.graph import Node

# 从本地 utils 模块中导入一些函数
from .utils import (
    get_target_type_str,
    getattr_from_fqn,
    return_first_non_observer_node,
)

# 导入自定义类型
from .ns_types import (
    NSSingleResultValuesType,
    NSSingleResultType,
)

# 导入类型提示
from typing import List, Optional, Dict, Callable

# 从 nn.Module 中分离模块的权重张量，并返回分离后的结果
def mod_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod.weight.detach()  # type: ignore[operator]

# 从 nn.Module 中分离第一个模块的权重张量，并返回分离后的结果
def mod_0_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod[0].weight.detach()  # type: ignore[index]

# 从 nn.Module 中获取权重和偏置的第一个元素，并返回结果
def mod_weight_bias_0(mod: nn.Module) -> torch.Tensor:
    return mod._weight_bias()[0]  # type: ignore[operator]

# 从 nn.Module 中获取 LSTM 模块的权重张量列表，并返回结果
def get_lstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for idx, param_name in enumerate(mod._flat_weights_names):  # type: ignore[arg-type]
        if 'weight_ih_l' in param_name or 'weight_hh_l' in param_name:
            param_value = mod._flat_weights[idx].detach()  # type: ignore[index]
            res.append(param_value)
    return res

# 从 nn.Module 中获取量化 LSTM 模块的权重张量列表，并返回结果
def get_qlstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for weight_value in mod._all_weight_values:  # type: ignore[union-attr]
        # 访问量化 LSTM 模块的权重张量的内部状态，并获取特定位置的值
        res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
        res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
    return res

# 从 nn.Module 中获取卷积模块的权重张量，并返回结果
def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    if (
        isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    ):
        return mod.weight.detach()
    elif (
        isinstance(mod, (nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d))
    ):
        return mod[0].weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]

# 从 nn.Module 中获取线性模块的权重张量，并返回结果
def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor:
    if isinstance(mod, nn.Linear):
        return mod.weight.detach()
    elif isinstance(mod, nni.LinearReLU):
        return mod[0].weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]

# 从 nn.Module 中获取 LSTM 模块的权重张量列表，并返回结果
def get_lstm_mod_weights(mod: nn.Module) -> List[torch.Tensor]:
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.LSTM):
        res = []
        for idx, param_name in enumerate(mod._flat_weights_names):
            if 'weight_ih_l' in param_name or 'weight_hh_l' in param_name:
                param_value = mod._flat_weights[idx].detach()
                res.append(param_value)
        return res
    else:
        # 断言模块类型是 nnqd.LSTM 的实例，如果不是则抛出异常
        assert isinstance(mod, nnqd.LSTM), f"type {type(mod)} not handled yet"
        # 初始化结果列表
        res = []
        # 遍历模块中的所有权重值
        for weight_value in mod._all_weight_values:
            # 获取权重值的第一个元素的第五层的第一个元素的状态，然后获取其第一个元素的第一个值，并添加到结果列表
            res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
            # 获取权重值的第一个元素的第五层的第二个元素的状态，然后获取其第一个元素的第一个值，并添加到结果列表
            res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
        # 返回结果列表
        return res
# 获取卷积操作的权重张量
def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # 从节点的参数中找到权重参数节点
    weight_arg_node = node.args[1]
    assert isinstance(weight_arg_node, Node)
    # 获取不包含观察者节点的权重节点
    weight_node = return_first_non_observer_node(weight_arg_node, gm)
    assert isinstance(weight_node, Node)
    assert weight_node.op == 'get_attr'
    # 从GraphModule中获取权重张量
    weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
    # 返回权重张量的副本（不带梯度）
    return weight.detach()

# 获取量化卷积操作的权重张量
def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # 量化卷积状态位于第二个参数
    qconv_state_node = node.args[1]
    assert isinstance(qconv_state_node, Node)
    assert qconv_state_node.op == 'get_attr'
    # 从GraphModule中获取量化卷积状态对象
    qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)  # type: ignore[arg-type]
    # 返回量化卷积的权重张量
    return qconv_state_obj.weight()

# 获取线性操作的权重张量
def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # 从权重参数开始向后遍历，考虑任何观察者
    # 支持的模式有：
    # weight -> obs -> linear
    # weight -> to(torch.float16) -> dequantize -> linear
    linear_second_arg = node.args[1]
    assert isinstance(linear_second_arg, Node)

    if linear_second_arg.op == 'call_module':
        # weight -> obs -> linear 的情况
        weight_arg_node = node.args[1]
        assert isinstance(weight_arg_node, Node)
        weight_node = weight_arg_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        # 从GraphModule中获取权重张量
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        # 返回权重张量的副本（不带梯度）
        return weight.detach()
    elif linear_second_arg.op == 'call_method':
        # weight -> to(torch.float16) -> dequantize -> linear 的情况
        assert linear_second_arg.op == 'call_method'
        dequant_node = node.args[1]
        assert isinstance(dequant_node, Node)
        to_fp16_node = dequant_node.args[0]
        assert isinstance(to_fp16_node, Node)
        # 提取目标数据类型，以便在返回之前进行类型转换
        target_dtype = to_fp16_node.args[1]
        weight_node = to_fp16_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        # 从GraphModule中获取权重张量
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        # 返回带有目标数据类型转换的权重张量的副本（不带梯度）
        return weight.detach().to(target_dtype)
    else:
        assert linear_second_arg.op == 'get_attr'
        # 直接从GraphModule中获取权重张量
        weight = getattr_from_fqn(gm, linear_second_arg.target)  # type: ignore[arg-type]
        # 返回权重张量的副本（不带梯度）
        return weight.detach()

# 获取量化线性操作的权重张量
def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # 打包的权重位于第二个参数
    packed_weight_node = node.args[1]
    assert isinstance(packed_weight_node, Node)
    assert packed_weight_node.op == 'get_attr'
    # 从GraphModule中获取打包的权重对象
    packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore[arg-type]
    # TODO（未来改进）：为什么 packed_weight.unpack() 不起作用？
    # 从打包的权重中提取权重张量和偏置，并忽略名称信息
    (weight, _bias), _name = packed_weight.__getstate__()
    # 返回weight变量的值作为函数的返回结果
    return weight
# 定义函数，返回一个字典，将操作类型映射到对应的权重提取函数
def get_op_to_type_to_weight_extraction_fn() -> Dict[str, Dict[Callable, Callable]]:
    # 初始化操作类型到权重提取函数的映射字典
    op_to_type_to_weight_extraction_fn: Dict[str, Dict[Callable, Callable]] = {
        'call_module': {
            # 对应 nn.Conv1d 操作，使用 mod_weight_detach 函数提取权重
            nn.Conv1d: mod_weight_detach,
            # 对应 nni.ConvReLU1d 操作，使用 mod_0_weight_detach 函数提取权重
            nni.ConvReLU1d: mod_0_weight_detach,
            # 对应 nnq.Conv1d 操作，使用 mod_weight_bias_0 函数提取权重
            nnq.Conv1d: mod_weight_bias_0,
            # 对应 nnqat.Conv1d 操作，使用 mod_weight_detach 函数提取权重
            nnqat.Conv1d: mod_weight_detach,
            # 对应 nniqat.ConvBn1d 操作，使用 mod_weight_detach 函数提取权重
            nniqat.ConvBn1d: mod_weight_detach,
            # 对应 nniqat.ConvBnReLU1d 操作，使用 mod_weight_detach 函数提取权重
            nniqat.ConvBnReLU1d: mod_weight_detach,
            # 对应 nniqat.ConvReLU1d 操作，使用 mod_weight_detach 函数提取权重
            nniqat.ConvReLU1d: mod_weight_detach,
            # 对应 nniq.ConvReLU1d 操作，使用 mod_weight_bias_0 函数提取权重
            nniq.ConvReLU1d: mod_weight_bias_0,
            # Conv2d 相关操作
            nn.Conv2d: mod_weight_detach,
            nni.ConvReLU2d: mod_0_weight_detach,
            nnq.Conv2d: mod_weight_bias_0,
            nnqat.Conv2d: mod_weight_detach,
            nniqat.ConvBn2d: mod_weight_detach,
            nniqat.ConvBnReLU2d: mod_weight_detach,
            nniqat.ConvReLU2d: mod_weight_detach,
            nniq.ConvReLU2d: mod_weight_bias_0,
            # Conv3d 相关操作
            nn.Conv3d: mod_weight_detach,
            nni.ConvReLU3d: mod_0_weight_detach,
            nnq.Conv3d: mod_weight_bias_0,
            nnqat.Conv3d: mod_weight_detach,
            nniqat.ConvBn3d: mod_weight_detach,
            nniqat.ConvBnReLU3d: mod_weight_detach,
            nniqat.ConvReLU3d: mod_weight_detach,
            nniq.ConvReLU3d: mod_weight_bias_0,
            # Linear 相关操作
            nn.Linear: mod_weight_detach,
            nnq.Linear: mod_weight_bias_0,
            nni.LinearReLU: mod_0_weight_detach,
            nniq.LinearReLU: mod_weight_bias_0,
            nnqat.Linear: mod_weight_detach,
            nnqd.Linear: mod_weight_bias_0,
            nniqat.LinearReLU: mod_weight_detach,
            nniqat.LinearBn1d: mod_weight_detach,
            nn.modules.linear.NonDynamicallyQuantizableLinear: mod_weight_detach,
            # LSTM 相关操作
            nn.LSTM: get_lstm_weight,
            nnqd.LSTM: get_qlstm_weight,
        },
        'call_function': {
            # Conv 相关函数调用
            F.conv1d: get_conv_fun_weight,
            F.conv2d: get_conv_fun_weight,
            F.conv3d: get_conv_fun_weight,
            toq.conv1d: get_qconv_fun_weight,
            toq.conv2d: get_qconv_fun_weight,
            toq.conv3d: get_qconv_fun_weight,
            toq.conv1d_relu: get_qconv_fun_weight,
            toq.conv2d_relu: get_qconv_fun_weight,
            toq.conv3d_relu: get_qconv_fun_weight,
            # Linear 相关函数调用
            F.linear: get_linear_fun_weight,
            toq.linear: get_qlinear_fun_weight,
            toq.linear_relu: get_qlinear_fun_weight,
        },
    }

    # 返回操作类型到权重提取函数的映射字典
    return op_to_type_to_weight_extraction_fn


# 定义函数，从节点中提取权重
def extract_weight_from_node(
    node: Node,
    gm: GraphModule,
    op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]] = None,
) -> Optional[NSSingleResultType]:
    # 结果类型为权重值
    res_type = NSSingleResultValuesType.WEIGHT.value

    # Not all graphmodules have _node_name_to_scope, so only fill it
    # 并非所有的图模块都有 _node_name_to_scope 属性，所以仅填充它
    # 初始化完全限定名称为 None
    fqn = None
    # 如果图模型对象 gm 具有属性 '_node_name_to_scope'，则获取节点名称对应的作用域完全限定名称
    if hasattr(gm, '_node_name_to_scope'):
        fqn = gm._node_name_to_scope[node.name][0]  # type: ignore[index]

    # 如果没有给定操作到类型到权重提取函数的映射，则获取默认的映射
    if op_to_type_to_weight_extraction_fn is None:
        op_to_type_to_weight_extraction_fn = get_op_to_type_to_weight_extraction_fn()

    # 获取目标节点的类型字符串作为参考节点类型
    ref_node_type = get_target_type_str(node, gm)
    # 对于权重提取，前一个节点的类型总是相同的
    prev_node_type = ref_node_type

    # 如果节点的操作是 'call_function'
    if node.op == 'call_function':
        # 获取 'call_function' 类型的权重提取函数映射
        function_mapping = op_to_type_to_weight_extraction_fn['call_function']
        # 遍历映射中的每个目标函数类型及其权重提取函数
        for target_fn_type, weight_extraction_fn in function_mapping.items():
            # 如果当前节点的目标函数类型与映射中的目标函数类型匹配
            if node.target == target_fn_type:
                # 调用权重提取函数，获取权重值
                weight = weight_extraction_fn(node, gm)
                # 返回包含权重信息的字典
                return {
                    'type': res_type,
                    'values': [weight],
                    'prev_node_name': node.name,
                    'prev_node_target_type': prev_node_type,
                    'ref_node_name': node.name,
                    'ref_node_target_type': ref_node_type,
                    'index_within_arg': 0,
                    'index_of_arg': 0,
                    'fqn': fqn,
                }

    # 如果节点的操作是 'call_module'
    elif node.op == 'call_module':
        # 对于 'call_module'，需要查找模块以进行类型检查
        assert isinstance(node.target, str)
        # 从完全限定名称中获取模块对象
        mod = getattr_from_fqn(gm, node.target)
        # 获取 'call_module' 类型的权重提取函数映射
        module_mapping = op_to_type_to_weight_extraction_fn['call_module']
        # 遍历映射中的每个目标模块类型及其权重提取函数
        for target_mod_type, weight_extraction_fn in module_mapping.items():
            # 如果当前模块的类型与映射中的目标模块类型匹配
            if type(mod) == target_mod_type:
                # 调用权重提取函数，获取权重值
                weight = weight_extraction_fn(mod)
                # 返回包含权重信息的字典
                return {
                    'type': res_type,
                    'values': [weight],
                    'prev_node_name': node.name,
                    'prev_node_target_type': prev_node_type,
                    'ref_node_name': node.name,
                    'ref_node_target_type': ref_node_type,
                    'index_within_arg': 0,
                    'index_of_arg': 0,
                    'fqn': fqn,
                }

    # 如果以上条件都不满足，则返回 None
    return None
```