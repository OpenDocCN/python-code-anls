# `.\pytorch\torch\ao\ns\fx\utils.py`

```
# mypy: allow-untyped-defs
# 引入枚举和操作符模块
import enum
import operator

# 引入PyTorch相关模块
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq

# 定义torch.ops.quantized的简称
toq = torch.ops.quantized

# 引入类型提示相关模块
from typing import Tuple, Callable, Dict, Set, List, Optional, Union

# 引入torch.fx相关模块
from torch.fx import GraphModule
from torch.fx.graph import Node

# 引入量化相关模块和工具函数
from torch.ao.quantization import (
    ObserverBase,
    FakeQuantizeBase,
)
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process

# 引入本地的类型定义模块
from .ns_types import NSNodeTargetType, NSResultsType

# TODO: 考虑在未来的PR中删除这个枚举并直接使用torch的类型定义。这可能会有些棘手，因为它们并非一一对应。
class NodeInputOrOutputType(enum.Enum):
    FP32 = enum.auto()  # torch.float
    INT8 = enum.auto()  # torch.qint8 or torch.quint8
    FP16 = enum.auto()  # torch.float16
    UNKNOWN = enum.auto()  # 无法确定输入/输出数据类型
    # TODO: 在未来的PR中，虽然这些函数可以支持多种数据类型，但为了数值调试的目的，我们想获取模型中实际使用的数据类型。可能需要一些数据类型传播来估计这一点。
    FP32_OR_INT8 = enum.auto()  # 可能是torch.float或torch.quint8或torch.qint8
    # TODO: 在未来的PR中，考虑动态量化、虚假量化等

# 获取节点的第一个输入和输出的数据类型
def get_node_first_input_and_output_type(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Tuple[NodeInputOrOutputType, NodeInputOrOutputType]:

    # TODO: 清理此部分代码
    # 从节点类型到输入输出类型映射字典中获取各种类型
    FUNS_IO_TYPE_FP32 = node_type_to_io_type_map["funs_io_type_fp32"]
    FUNS_IO_TYPE_FP16 = node_type_to_io_type_map["funs_io_type_fp16"]
    FUNS_IO_TYPE_INT8 = node_type_to_io_type_map["funs_io_type_int8"]
    FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["funs_io_type_fp32_or_int8"]
    MODS_IO_TYPE_FP32 = node_type_to_io_type_map["mods_io_type_fp32"]
    MODS_IO_TYPE_INT8 = node_type_to_io_type_map["mods_io_type_int8"]
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]
    METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["meths_io_type_fp32_or_int8"]
    # 如果节点操作为函数调用
    if node.op == "call_function":
        # 如果函数目标在FP32类型函数集合中
        if node.target in FUNS_IO_TYPE_FP32:
            # 返回FP32作为输入输出类型
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        # 如果函数目标在FP16类型函数集合中
        if node.target in FUNS_IO_TYPE_FP16:
            # 返回FP16作为输入输出类型
            return (NodeInputOrOutputType.FP16, NodeInputOrOutputType.FP16)
        # 如果函数目标在INT8类型函数集合中
        elif node.target in FUNS_IO_TYPE_INT8:
            # 返回INT8作为输入输出类型
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        # 如果函数目标在同时支持FP32或INT8类型函数集合中
        elif node.target in FUNS_IO_TYPE_FP32_OR_INT8:
            # 获取函数调用的第一个标准化输入
            first_arg = get_normalized_nth_input(node, gm, 0)
            # 断言第一个参数是一个节点对象
            assert isinstance(first_arg, Node)
            # 获取第一个输入节点的输入类型和输出类型
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            # 返回第一个输入节点的输出类型作为输入输出类型
            return (prev_node_output_type, prev_node_output_type)
        # 如果函数目标不在以上已知类型的函数集合中
        else:
            # 返回未知类型作为输入输出类型
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    # 如果节点操作为模块调用
    elif node.op == "call_module":
        # 断言节点操作为模块调用
        assert node.op == "call_module"
        # 断言节点目标是一个字符串
        assert isinstance(node.target, str)
        # 从全限定名获取模块对象
        mod = getattr_from_fqn(gm, node.target)
        # 检查模块是否是已知支持FP32或INT8类型输入模块之一
        is_known_fp32_or_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8  # type: ignore[arg-type]
        )
        # 如果模块是日志记录器、观察器或伪量化基类的实例，或者是已知支持FP32或INT8类型输入模块之一
        if (
            isinstance(mod, (logger_cls, ObserverBase, FakeQuantizeBase))  # type: ignore[arg-type]
            or is_known_fp32_or_int8_input_module
        ):
            # 获取函数调用的第一个标准化输入
            first_arg = get_normalized_nth_input(node, gm, 0)
            # 断言第一个参数是一个节点对象
            assert isinstance(first_arg, Node)
            # 获取第一个输入节点的输入类型和输出类型
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            # 返回第一个输入节点的输出类型作为输入输出类型
            return (prev_node_output_type, prev_node_output_type)
        
        # 检查模块是否是已知支持FP32类型输入模块之一
        is_known_fp32_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32  # type: ignore[arg-type]
        )
        # 检查模块是否是已知支持INT8类型输入模块之一
        is_known_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_INT8  # type: ignore[arg-type]
        )
        # 如果模块是已知支持FP32类型输入模块之一
        if is_known_fp32_input_module:
            # 返回FP32作为输入输出类型
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        # 如果模块是已知支持INT8类型输入模块之一
        elif is_known_int8_input_module:
            # 返回INT8作为输入输出类型
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        # 如果模块不是以上已知类型的模块
        else:
            # 返回未知类型作为输入输出类型
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    # 如果节点操作是 "call_method"
    elif node.op == "call_method":
        # 如果调用的方法是 "dequantize"
        if node.target == "dequantize":
            # "dequantize" 是一个特殊节点，因为它允许多种输入类型。
            # 因此，我们查找前一个节点的输出类型，并将其作为此节点实例的输入类型返回。
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, NodeInputOrOutputType.FP32)

        # 如果调用的方法是 "to"
        elif node.target == "to":
            # "to" 是一个特殊节点，因为它允许多种输入类型。
            # 因此，我们查找前一个节点的输出类型，并将其作为此节点实例的输入类型返回。
            # 我们还查找 "to" 的目标并返回正确的输出类型。
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )

            # 获取当前节点的数据类型目标
            cur_node_dtype_target = get_normalized_nth_input(node, gm, 1)
            assert (
                cur_node_dtype_target is torch.float16
            ), f"{cur_node_dtype_target} handling needs to be added"

            return (prev_node_output_type, NodeInputOrOutputType.FP16)

        # 如果调用的方法在 METHS_IO_TYPE_FP32_OR_INT8 列表中
        elif node.target in METHS_IO_TYPE_FP32_OR_INT8:
            # 获取第一个参数
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)

        # 如果调用的方法不在已知的特殊方法中，则返回未知类型
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    
    # 如果节点操作不是 "call_method"，则返回未知类型
    else:
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
# 获取节点的输入量化参数 (scale, zero_point)，如果能够从图中推断出来的话
def get_node_input_qparams(
    node: Node,
    gm: GraphModule,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Optional[Tuple[Union[torch.Tensor, float], Union[torch.Tensor, int]]]:
    """
    Returns the qparams (scale, zero_point) of the first input to `node`,
    if they can be inferred from the graph.
    """
    # 获取规范化后的节点输入中的第一个节点
    prev_node = get_normalized_nth_input(node, gm, 0)

    # 如果前驱节点不是一个节点对象，则返回空
    if not isinstance(prev_node, Node):
        return None

    # 从映射中获取节点类型到 I/O 类型映射的子集
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]

    # 从函数参数中获取 scale 和 zero_point 的节点对象
    def _get_scale_zp_from_function_args(node, gm, scale_arg_idx, zp_arg_idx):
        scale_node = get_normalized_nth_input(node, gm, scale_arg_idx)
        zp_node = get_normalized_nth_input(node, gm, zp_arg_idx)
        # 断言确保节点对象是有效的，并且目标是字符串
        assert isinstance(scale_node, Node) and isinstance(scale_node.target, str)
        assert isinstance(zp_node, Node) and isinstance(zp_node.target, str)
        # 从完全限定名称 (FQN) 获取 scale 和 zero_point 对象
        scale_obj = getattr_from_fqn(gm, scale_node.target)
        zp_obj = getattr_from_fqn(gm, zp_node.target)
        return (scale_obj, zp_obj)

    # 如果前驱节点的操作是 "call_function"
    if prev_node.op == "call_function":

        # 如果调用的函数是 torch.quantize_per_tensor，则直接读取参数
        if prev_node.target == torch.quantize_per_tensor:
            return _get_scale_zp_from_function_args(prev_node, gm, 1, 2)
        # 如果调用的函数在 (toq.add, toq.add_relu, toq.mul, toq.mul_relu) 中，则读取参数
        elif prev_node.target in (toq.add, toq.add_relu, toq.mul, toq.mul_relu):
            return _get_scale_zp_from_function_args(prev_node, gm, 2, 3)

        # 其他情况返回空
        return None
        # TODO(future PR): 处理更多功能性操作
        # TODO(future PR): 处理从输入继承 qparams 的功能操作
    # 如果前一个节点操作是调用模块
    elif prev_node.op == "call_module":
        # 获取模块的类型
        assert isinstance(prev_node.target, str)
        module_obj = getattr_from_fqn(gm, prev_node.target)
        
        # 检查模块类型是否属于量化或特定类型
        if isinstance(
            module_obj,
            (
                nnq.Linear,
                nnq.Conv1d,
                nnq.Conv2d,
                nniq.ConvReLU2d,
                nnq.Conv3d,
                nnq.BatchNorm2d,
                nnq.BatchNorm3d,
                nnq.ConvTranspose1d,
                nnq.ConvTranspose2d,
                nnq.ELU,
                nnq.GroupNorm,
                nnq.InstanceNorm1d,
                nnq.InstanceNorm2d,
                nnq.InstanceNorm3d,
                nnq.LayerNorm,
                nnq.Hardswish,
                nnq.LeakyReLU,
                nnq.ReLU6,
                nniq.BNReLU2d,
                nniq.BNReLU3d,
                nniq.ConvReLU1d,
                nniq.ConvReLU2d,
                nniq.ConvReLU3d,
                nniq.LinearReLU,
            ),
        ):
            # 返回模块的缩放因子和零点偏移量
            return (module_obj.scale, module_obj.zero_point)  # type: ignore[return-value]

        # 检查是否为已知的 FP32 或 INT8 输入模块类型
        is_known_fp32_or_int8_input_module = any(
            isinstance(module_obj, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8  # type: ignore[arg-type]
        )
        if is_known_fp32_or_int8_input_module:
            # 获取节点输入的量化参数
            return get_node_input_qparams(prev_node, gm, node_type_to_io_type_map)

    # 如果条件不满足，返回空值
    return None
# 如果节点不是观察者，则直接返回该节点；如果节点是观察者，则向上遍历图并返回第一个不是观察者的父节点。
def return_first_non_observer_node(
    node: Node,
    gm: GraphModule,
) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    if node.op == "call_module":
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if _is_activation_post_process(node_obj):
            assert len(node.args) == 1
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            # code duplication intended, not worth refactoring
            assert isinstance(node.target, str)
            node_obj = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(node_obj):
                assert len(node.args) == 1
                assert isinstance(node.args[0], Node)
                node = node.args[0]
    return node


# 假设所有非参数的参数都在前面。返回节点所期望的非参数参数的数量。
def get_number_of_non_param_args(
    node: Node,
    gm: GraphModule,
) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
    if node.op == "call_module":
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if isinstance(node_obj, nn.LSTM):
            return 2

    # 默认返回1
    return 1


# 返回应将日志器附加到的节点参数的索引列表，如果启用了输入日志记录。
def get_arg_indices_of_inputs_to_log(node: Node) -> List[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
    if len(node.args) == 0:
        return []
    if node.op == "call_function" and (
        # TODO(future PR): use relationship map instead of hardcoding
        node.target in (torch.add, torch.ops.quantized.add, operator.add)
        or node.target in (torch.mul, torch.ops.quantized.mul, operator.mul)
    ):
        result = []
        for i in range(2):
            if type(node.args[i]) == Node:
                result.append(i)
        return result
    return [0]


# 返回指向此节点的函数或模块类型的字符串表示，对于其他节点类型返回 ''。
def get_target_type_str(node: Node, gm: GraphModule) -> str:
    """
    Returns a string representation of the type of the function or module
    pointed to by this node, or '' for other node types.
    """
    target_type = ""
    if node.op in ("call_function", "call_method"):
        target_type = torch.typename(node.target)
    # 如果节点操作为调用模块
    elif node.op == "call_module":
        # 断言节点的目标是字符串类型
        assert isinstance(node.target, str)
        # 从全限定名中获取目标模块对象
        target_mod = getattr_from_fqn(gm, node.target)
        # 获取目标模块的类型名称
        target_type = torch.typename(target_mod)
    # 返回目标模块的类型名称
    return target_type
def rekey_logger_info_on_node_name_of_model(
    results: NSResultsType,
    model_name: str,
) -> NSResultsType:
    """
    Rekeys the layer name of a results dictionary to use node names
    from `model_name`.

    For example, transforms

        {'base_op_1_0': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    into

        {'linear1': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    Note: we cannot use these node names directly because they are not
    guaranteed to be consistent across models. This is why we extract
    the results first and rekey afterwards.
    """
    # Initialize an empty dictionary to store the rekeyed results
    new_results = {}

    # Iterate through the original results dictionary
    for old_layer_name, result_type_to_results in results.items():
        new_layer_name = None
        # Iterate through nested dictionaries within the results
        for model_name_to_results in result_type_to_results.values():
            for cur_model_name, list_of_results in model_name_to_results.items():
                # Check if the current model name matches the specified model_name
                if cur_model_name == model_name:
                    assert len(list_of_results) > 0
                    # Extract the node name from the first result
                    new_layer_name = list_of_results[0]["ref_node_name"]
                else:
                    continue
        # Assign the rekeyed results to the new_results dictionary
        if new_layer_name is not None:
            new_results[new_layer_name] = result_type_to_results
        else:
            new_results[old_layer_name] = result_type_to_results

    # Return the rekeyed results dictionary
    return new_results


def maybe_add_missing_fqns(results: NSResultsType) -> None:
    """
    If `fqn` entries are filled in for one of the models in `results`, copies
    them over to any models which do not have them filled out.

    A common use case benefitting from this is comparing a model prepared by
    quantization to a quantized model. In this case, the model prepared by
    quantization would have `fqn` entries, and the quantized model would not.
    """

    # Initialize a variable to store the model name with fqn entries
    model_name_with_fqns = None

    # Iterate through the nested dictionaries in results
    for result_type_to_results in results.values():
        for model_name_to_results in result_type_to_results.values():
            for model_name, model_results in model_name_to_results.items():
                # Check if there are results for the current model
                if len(model_results) > 0:
                    # Check if the first result has the 'fqn' entry filled
                    if model_results[0]["fqn"] is not None:
                        model_name_with_fqns = model_name
                        break
            if model_name_with_fqns:
                break
        if model_name_with_fqns:
            break

    # If a model with 'fqn' entries is found, copy these entries to other models
    if model_name_with_fqns:
        for result_type_to_results in results.values():
            for model_name_to_results in result_type_to_results.values():
                ref_model_results = model_name_to_results[model_name_with_fqns]
                for model_name, model_results in model_name_to_results.items():
                    if model_name == model_name_with_fqns:
                        continue
                    for i in range(len(model_results)):
                        # Copy 'fqn' from the reference model results
                        fqn = ref_model_results[i]["fqn"]
                        model_results[i]["fqn"] = fqn
    # 定义一个内部函数 inner，接受任意数量的位置参数和关键字参数
    def inner(*args, **kwargs):
        # 解包位置参数 args，取第一个、第二个参数到 a0 和 a1，其余参数到 a_other 中
        a0, a1, *a_other = args

        # 如果 a0 和 a1 均为元组或列表类型，则递归处理它们的元素
        if (isinstance(a0, tuple) and isinstance(a1, tuple)) or (
            isinstance(a0, list) and isinstance(a1, list)
        ):
            # 初始化结果列表
            results = []
            # 使用 zip 函数迭代处理 a0 和 a1 的每一对元素
            for el0, el1 in zip(a0, a1):
                # 创建新的参数元组 new_args，包括当前元素和其余位置参数 a_other
                new_args = (el0, el1, *a_other)
                # 递归调用 inner 函数，将结果添加到 results 中
                results.append(inner(*new_args, **kwargs))
            # 返回处理后的结果列表
            return results

        # 如果 a0 和 a1 均为 torch.Tensor 类型，则进行额外的处理
        elif isinstance(a0, torch.Tensor) and isinstance(a1, torch.Tensor):
            # 如果 a0 是量化的 tensor，则去量化化处理
            if a0.is_quantized:
                a0 = a0.dequantize()
            # 如果 a1 是量化的 tensor，则去量化化处理
            if a1.is_quantized:
                a1 = a1.dequantize()

        # 仅处理 float 类型的参数
        # 如果 a0 或 a1 的数据类型不是 torch.float，则返回 None
        if a0.dtype != torch.float or a1.dtype != torch.float:
            return None

        # 创建新的参数元组 new_args，包括当前参数和其余位置参数 a_other
        new_args = (a0, a1, *a_other)
        # 调用外部传入的函数 f，使用新参数和关键字参数 kwargs
        return f(*new_args, **kwargs)

    # 返回内部函数 inner 的引用
    return inner
# 对函数进行装饰，可能会对前两个张量参数进行反量化处理并处理元组
@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算 `x` 和 `y` 之间的信噪比（SQNR）。

    Args:
        x: 张量或张量元组
        y: 张量或张量元组

    Return:
        float 或浮点数元组
    """
    # 计算信号功率 Ps
    Ps = torch.norm(x)
    # 计算噪声功率 Pn
    Pn = torch.norm(x - y)
    # 计算并返回信噪比（SQNR）
    return 20 * torch.log10(Ps / Pn)


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_normalized_l2_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算 `x` 和 `y` 之间的归一化 L2 误差。

    Args:
        x: 张量或张量元组
        y: 张量或张量元组

    Return:
        float 或浮点数元组
    """
    # 计算归一化 L2 误差并返回
    return torch.sqrt(((x - y) ** 2).sum() / (x ** 2).sum())


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算 `x` 和 `y` 之间的余弦相似度。

    Args:
        x: 张量或张量元组
        y: 张量或张量元组

    Return:
        float 或浮点数元组
    """
    # 对卷积操作，量化权重的形状比 fp32 权重多一个维度。匹配形状以启用余弦相似度比较。
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    # 计算余弦相似度并返回
    return torch.nn.functional.cosine_similarity(x, y)

def op_type_supports_shadowing(node: Node) -> bool:
    """
    检查节点操作是否支持阴影处理。

    Args:
        node: 节点对象

    Returns:
        bool，如果支持阴影处理则返回 True，否则返回 False
    """
    if node.op == 'call_function':
        if node.target in (torch.add, torch.mul, operator.add, operator.mul, torch.cat, torch.stack):
            # 多个张量输入的操作暂不支持阴影处理
            return False
    return True

def get_normalized_nth_input(node: Node, gm: GraphModule, idx: int) -> Node:
    """
    给定一个节点，获取该节点的第 n 个输入，尽可能地进行参数和关键字参数的标准化处理。

    Args:
        node: 节点对象
        gm: 图模块
        idx: 输入索引

    Returns:
        Node: 节点的第 n 个输入
    """
    try:
        # 标准化参数和关键字参数到尽可能只使用关键字参数
        norm_args_and_kwargs = node.normalized_arguments(
            gm, normalize_to_only_use_kwargs=True)
        if norm_args_and_kwargs is not None:
            norm_args, norm_kwargs = norm_args_and_kwargs
            assert len(norm_args) + len(norm_kwargs) > idx
            if idx < len(norm_args):
                return norm_args[idx]
            else:
                # 注意：在 Python 3.7+ 中字典是有序的
                return list(norm_kwargs.values())[idx]
        else:
            assert len(node.args) + len(node.kwargs) > idx
            if idx < len(node.args):
                return node.args[idx]  # type: ignore[return-value]
            else:
                kwargs_idx = idx + len(node.args)
                return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
    except Exception as e:
        # 如果出现异常，记录并返回空节点
        return Node()
    # 捕获 RuntimeError 异常
    except RuntimeError:
        # 当节点参数规范化需要类型提示才能继续时会引发此 RuntimeError，
        # 例如对于 torch.add，其中第一个、第二个或两个参数可能是张量
        assert len(node.args) + len(node.kwargs) > idx
        
        # 如果 idx 小于 node.args 的长度，则返回对应位置的参数值
        if idx < len(node.args):
            return node.args[idx]  # type: ignore[return-value]
        else:
            # 否则，计算关键字参数的索引并返回对应位置的值
            kwargs_idx = idx + len(node.args)
            return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
```