# `.\pytorch\torch\utils\mobile_optimizer.py`

```py
# mypy: allow-untyped-defs
"""This module contains utility method for mobile model optimization and lint."""

import torch  # 导入PyTorch库
from enum import Enum  # 导入枚举类型Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType  # 导入MobileOptimizerType类型别名
from typing import Optional, Set, List, AnyStr  # 导入类型提示所需的类

class LintCode(Enum):  # 定义LintCode枚举类
    BUNDLED_INPUT = 1  # 枚举项：BUNDLED_INPUT
    REQUIRES_GRAD = 2  # 枚举项：REQUIRES_GRAD
    DROPOUT = 3  # 枚举项：DROPOUT
    BATCHNORM = 4  # 枚举项：BATCHNORM

def optimize_for_mobile(
        script_module: torch.jit.ScriptModule,  # 参数：torch脚本模块
        optimization_blocklist: Optional[Set[MobileOptimizerType]] = None,  # 参数：优化阻止列表（可选）
        preserved_methods: Optional[List[AnyStr]] = None,  # 参数：保留方法列表（可选）
        backend: str = 'CPU') -> torch.jit.RecursiveScriptModule:
    """
    Optimize a torch script module for mobile deployment.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.
            Torch脚本模块的实例，类型为ScriptModule。
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
            MobileOptimizerType类型的集合。如果未传递集合，则优化方法将运行所有优化器过程；
            否则，优化方法将运行不包含在optimization_blocklist中的优化过程。
        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
            freeze_module调用时需要保留的方法列表。
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
            运行结果模型所使用的设备类型（默认为CPU，也可以是Vulkan或Metal）。

    Returns:
        A new optimized torch script module
        返回一个新的优化过的torch脚本模块
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    if optimization_blocklist is None:
        optimization_blocklist = set()  # 如果未提供优化阻止列表，默认为空集合

    if preserved_methods is None:
        preserved_methods = []  # 如果未提供保留方法列表，默认为空列表

    # Convert potential byte arrays into strings (if there is any) to pass type checking
    # Here we use a new name as assigning it back to preserved_methods will invoke
    # mypy errors (i.e. List[AnyStr] = List[str])
    preserved_methods_str: List[str] = [str(method) for method in preserved_methods]
    # 将潜在的字节数组转换为字符串（如果有的话），以通过类型检查

    bundled_inputs_attributes = _get_bundled_inputs_preserved_attributes(script_module, preserved_methods_str)
    # 获取绑定输入的保留属性

    if all(hasattr(script_module, method) for method in bundled_inputs_attributes):
        preserved_methods_str = list(set(preserved_methods_str + bundled_inputs_attributes))
    # 如果所有的属性都存在于script_module中，则将它们添加到保留方法列表中

    non_exist_methods = []
    for method in preserved_methods_str:
        if not hasattr(script_module, method):
            non_exist_methods.append(method)
    if non_exist_methods:
        raise AttributeError(
            f"The following methods to preserve do not exist in script_module: {', '.join(non_exist_methods)}")
    # 检查是否所有的保留方法都存在于script_module中，否则引发AttributeError异常

    backend = backend.lower()  # 将backend参数转换为小写
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
        # 对脚本模块进行移动端优化处理，并返回优化后的模块
    elif backend == 'vulkan':
        # 如果指定的后端是 Vulkan，调用 Vulkan 优化函数对脚本模块进行优化
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
    elif backend == 'metal':
        # 如果指定的后端是 Metal，调用 Metal 优化函数对脚本模块进行优化
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods_str)
    else:
        # 如果指定的后端不是 'CPU', 'Vulkan' 或 'Metal' 中的任何一个，抛出类型错误异常
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")

    # 返回经过递归包装的优化后的 C++ 模块
    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)
# 生成给定 torch 脚本模块的 lint 列表
def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Generate a list of lints for a given torch script module.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
    # 检查传入的参数是否为 torch.jit.ScriptModule 类型，若不是则抛出类型错误
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    # 初始化 lint 列表
    lint_list = []

    # 检查脚本模块是否具有 "_generate_bundled_inputs_for_forward" 属性，若没有则添加相应 lint 提示
    if not hasattr(script_module, "_generate_bundled_inputs_for_forward"):
        lint_list.append({"name": LintCode.BUNDLED_INPUT.name, "message": "No bundled input for forward, please add bundled inputs "
                          "before saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs."})

    # 遍历脚本模块的参数，若参数需要梯度则添加相应 lint 提示
    for name, param in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({"name": LintCode.REQUIRES_GRAD.name, "message": f"Param {name} requires grad, "
                             "please set torch.no_grad() to reduce memory usage and improve computation speed during "
                             "inference phase."})

    # 获取脚本模块中的操作名列表
    op_names = torch.jit.export_opnames(script_module)
    # 遍历操作名列表，若包含 "dropout" 则添加相应 lint 提示，若包含 "batch_norm" 则也添加相应 lint 提示
    for op_name in op_names:
        if "dropout" in op_name:
            lint_list.append({"name": LintCode.DROPOUT.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout "
                              "operator."})
        if "batch_norm" in op_name:
            lint_list.append({"name": LintCode.BATCHNORM.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm "
                              "operator."})

    # 返回 lint 列表
    return lint_list

# 获取脚本模块中保留的捆绑输入属性列表
def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: List[str]) -> List[str]:

    bundled_inputs_attributes = []

    # 如果脚本模块具有 'get_all_bundled_inputs' 属性，则添加到捆绑输入属性列表中
    if hasattr(script_module, 'get_all_bundled_inputs'):
        bundled_inputs_attributes.append('get_all_bundled_inputs')
        bundled_inputs_attributes.append('get_num_bundled_inputs')

    # 对于引入多个函数捆绑输入的变更后的模块中的捆绑输入，在这里处理
    # 检查 script_module 是否有 get_bundled_inputs_functions_and_info 方法
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        # 将 'get_bundled_inputs_functions_and_info' 添加到 bundled_inputs_attributes 列表中
        bundled_inputs_attributes.append('get_bundled_inputs_functions_and_info')
        
        # 调用 script_module 的 get_bundled_inputs_functions_and_info 方法，获取所有信息
        all_info = script_module.get_bundled_inputs_functions_and_info()
        
        # 遍历 all_info 中的每个 function_name
        for function_name in all_info:
            # 如果 function_name 不在 preserved_methods 中，则将其添加到 bundled_inputs_attributes 列表中
            if function_name not in preserved_methods:
                bundled_inputs_attributes.append(function_name)
            
            # 向 bundled_inputs_attributes 列表中添加特定的字符串
            bundled_inputs_attributes.append("get_all_bundled_inputs_for_" + function_name)
            bundled_inputs_attributes.append("_bundled_inputs_deflated_" + function_name)

    # 返回存储了所有相关信息的 bundled_inputs_attributes 列表
    return bundled_inputs_attributes
```