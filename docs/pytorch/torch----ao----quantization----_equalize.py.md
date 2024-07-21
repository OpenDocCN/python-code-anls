# `.\pytorch\torch\ao\quantization\_equalize.py`

```
"""
# 添加类型提示 'mypy: allow-untyped-defs'
import torch  # 导入 PyTorch 库
import copy  # 导入复制模块
from typing import Dict, Any  # 导入类型提示

__all__ = [  # 列出公开的函数和变量
    "set_module_weight",
    "set_module_bias",
    "has_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "get_name_by_module",
    "cross_layer_equalization",
    "process_paired_modules_list_to_name",
    "expand_groups_in_paired_modules_list",
    "equalize",
    "converged",
]

# 支持的普通模块类型
_supported_types = {torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d}
# 支持的内置模块类型
_supported_intrinsic_types = {torch.ao.nn.intrinsic.ConvReLU2d, torch.ao.nn.intrinsic.LinearReLU, torch.ao.nn.intrinsic.ConvReLU1d}
# 所有支持的模块类型的集合
_all_supported_types = _supported_types.union(_supported_intrinsic_types)

# 设置模块权重的函数
def set_module_weight(module, weight) -> None:
    if type(module) in _supported_types:  # 如果模块类型是支持的普通类型之一
        module.weight = torch.nn.Parameter(weight)  # 设置模块的权重
    else:
        module[0].weight = torch.nn.Parameter(weight)  # 否则，设置模块列表中第一个模块的权重

# 设置模块偏置的函数
def set_module_bias(module, bias) -> None:
    if type(module) in _supported_types:  # 如果模块类型是支持的普通类型之一
        module.bias = torch.nn.Parameter(bias)  # 设置模块的偏置
    else:
        module[0].bias = torch.nn.Parameter(bias)  # 否则，设置模块列表中第一个模块的偏置

# 检查模块是否有偏置的函数
def has_bias(module) -> bool:
    if type(module) in _supported_types:  # 如果模块类型是支持的普通类型之一
        return module.bias is not None  # 返回模块是否有偏置
    else:
        return module[0].bias is not None  # 返回模块列表中第一个模块是否有偏置

# 获取模块权重的函数
def get_module_weight(module):
    if type(module) in _supported_types:  # 如果模块类型是支持的普通类型之一
        return module.weight  # 返回模块的权重
    else:
        return module[0].weight  # 返回模块列表中第一个模块的权重

# 获取模块偏置的函数
def get_module_bias(module):
    if type(module) in _supported_types:  # 如果模块类型是支持的普通类型之一
        return module.bias  # 返回模块的偏置
    else:
        return module[0].bias  # 返回模块列表中第一个模块的偏置

# 在指定的轴上应用 'torch.max' 函数的函数
def max_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.max' over the given axes."""
    axis_list.sort(reverse=True)  # 对轴列表进行倒序排序
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)  # 应用 'torch.max' 函数到指定轴
    return input  # 返回结果

# 在指定的轴上应用 'torch.min' 函数的函数
def min_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.min' over the given axes."""
    axis_list.sort(reverse=True)  # 对轴列表进行倒序排序
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)  # 应用 'torch.min' 函数到指定轴
    return input  # 返回结果

# 查找特定通道权重范围的函数
def channel_range(input, axis=0):
    """Find the range of weights associated with a specific channel."""
    size_of_tensor_dim = input.ndim  # 获取张量的维度数
    axis_list = list(range(size_of_tensor_dim))  # 创建轴索引列表
    axis_list.remove(axis)  # 移除指定的轴索引

    mins = min_over_ndim(input, axis_list)  # 计算在其他轴上的最小值
    maxs = max_over_ndim(input, axis_list)  # 计算在其他轴上的最大值

    assert mins.size(0) == input.size(axis), "Dimensions of resultant channel range does not match size of requested axis"  # 断言确保结果的维度与请求的轴大小匹配
    return maxs - mins  # 返回权重范围

# 根据模块获取模块在模型中的名称的函数
def get_name_by_module(model, module):
    """Get the name of a module within a model.

    Args:
        model: a model (nn.module) that equalization is to be applied on
        module: a module within the model

    Returns:
        name: the name of the module within the model
    """
    for name, m in model.named_modules():  # 遍历模型中的所有模块
        if m is module:  # 如果找到了目标模块
            return name  # 返回模块的名称
    raise ValueError("module is not in the model")  # 抛出异常，模块不在模型中

# 交叉层均衡化的函数
def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
"""
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors, scale the weights so that the ranges of
    the output channels of the first tensor equal the ranges of the input
    channels of the second tensor.
    """
    # 检查 module1 和 module2 的类型是否被支持，如果不支持则抛出 ValueError 异常
    if type(module1) not in _all_supported_types or type(module2) not in _all_supported_types:
        raise ValueError("module type not supported:", type(module1), " ", type(module2))

    # 检查 module1 是否有偏置
    conv1_has_bias = has_bias(module1)
    bias = None

    # 获取 module1 和 module2 的权重
    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    # 检查权重的输出通道数和输入通道数是否匹配，如果不匹配则抛出 TypeError 异常
    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError("Number of output channels of first arg do not match \
        number input channels of second arg")

    # 如果 module1 有偏置，则获取偏置值
    if conv1_has_bias:
        bias = get_module_bias(module1)

    # 计算 module1 和 module2 的权重的通道范围
    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # 增加一个微小的值，避免除以零
    weight2_range += 1e-9

    # 计算应用的缩放因子
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    # 如果 module1 有偏置，则对偏置进行缩放
    if conv1_has_bias:
        bias = bias * inverse_scaling_factors

    # 将 scaling_factors 和 inverse_scaling_factors 调整为 1 维张量，以便广播应用
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    # 对权重进行缩放
    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    # 更新 module1 和 module2 的权重
    set_module_weight(module1, weight1)
    if conv1_has_bias:
        set_module_bias(module1, bias)
    set_module_weight(module2, weight2)
    # 将给定模型中的模块名对应列表处理成模块名的列表
    paired_modules_list = process_paired_modules_list_to_name(model, paired_modules_list)

    # 如果不是原地操作，对模型进行深拷贝
    if not inplace:
        model = copy.deepcopy(model)

    # 将模块名对应列表中的模块组扩展为只包含两个模块的组
    paired_modules_list = expand_groups_in_paired_modules_list(paired_modules_list)

    # 创建一个字典，用于存储模块名到对应模块的映射关系
    name_to_module: Dict[str, torch.nn.Module] = {}

    # 创建一个字典，用于存储上一轮迭代中的模块名到对应模块的映射关系
    previous_name_to_module: Dict[str, Any] = {}

    # 创建一个集合，用于存储所有模块的名称
    name_set = {name for pair in paired_modules_list for name in pair}
    # 遍历模型中所有命名的模块及其对应的模块对象
    for name, module in model.named_modules():
        # 如果模块名存在于指定的名称集合中
        if name in name_set:
            # 将该模块名与模块对象添加到名称到模块的映射字典中
            name_to_module[name] = module
            # 将之前的名称到模块的映射字典中该模块名对应的值设为None
            previous_name_to_module[name] = None
    
    # 循环直到模型收敛于给定的阈值
    while not converged(name_to_module, previous_name_to_module, threshold):
        # 遍历成对的模块列表
        for pair in paired_modules_list:
            # 使用深拷贝复制当前模块名对应的模块对象到之前的名称到模块的映射字典中
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            # 对成对的模块执行跨层均衡化操作
            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    # 返回经过处理后的模型对象
    return model
def converged(curr_modules, prev_modules, threshold=1e-4):
    """Test whether modules are converged to a specified threshold.

    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary

    """
    # 检查当前模块和前一模块的名称是否一致，如果不一致则引发 ValueError
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError("The keys to the given mappings must have the same set of names of modules")

    # 初始化总和范数为零的张量
    summed_norms = torch.tensor(0.)

    # 如果前一模块中存在值为 None 的情况，则返回 False
    if None in prev_modules.values():
        return False

    # 遍历当前模块的每个名称
    for name in curr_modules.keys():
        # 获取当前模块和前一模块的权重
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])

        # 计算当前模块和前一模块权重的差异
        difference = curr_weight.sub(prev_weight)

        # 计算差异的范数并累加到总和范数
        summed_norms += torch.norm(difference)

    # 返回总和范数是否小于给定阈值的布尔值
    return bool(summed_norms < threshold)
```