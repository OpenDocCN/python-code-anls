# `.\pytorch\torch\ao\quantization\fuse_modules.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import copy  # 导入copy模块，用于对象的深拷贝操作

import torch.nn as nn  # 导入PyTorch的神经网络模块

from torch.ao.quantization.fuser_method_mappings import get_fuser_method  # 从torch.ao.quantization.fuser_method_mappings模块中导入get_fuser_method函数
# 为了向后兼容性
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn  # noqa: F401
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn_relu  # noqa: F401
# 从torch.nn.utils.parametrize模块中导入type_before_parametrizations函数
from torch.nn.utils.parametrize import type_before_parametrizations

from typing import List, Optional  # 导入类型提示List和Optional

__all__ = [
    "fuse_known_modules",
    "fuse_modules",
    "fuse_modules_qat",
]

# Generalization of getattr
# 获取模块的通用函数，根据子模块的键名从模型中获取对应的模块
def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')  # 将子模块键名按点号分割成列表
    cur_mod = model  # 当前模块初始化为输入的模型对象
    for s in tokens:
        cur_mod = getattr(cur_mod, s)  # 根据当前的键名获取模块对象
    return cur_mod

# Generalization of setattr
# 设置模块的通用函数，根据子模块的键名将指定的模块设置到模型中
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')  # 将子模块键名按点号分割成列表
    sub_tokens = tokens[:-1]  # 子模块键名列表除了最后一个元素外的所有元素
    cur_mod = model  # 当前模块初始化为输入的模型对象
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)  # 根据当前的键名获取模块对象

    setattr(cur_mod, tokens[-1], module)  # 将指定的模块设置到最后一个子模块键名对应的模块对象中

def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    r"""Return a list of known fuse modules.

    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, bn
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    types = tuple(type_before_parametrizations(m) for m in mod_list)  # 获取每个模块的类型并组成元组
    fuser_method = get_fuser_method(types, additional_fuser_method_mapping)  # 获取模块融合的方法
    if fuser_method is None:
        raise NotImplementedError(f"Cannot fuse modules: {types}")  # 如果没有找到合适的融合方法，则抛出NotImplementedError异常
    new_mod : List[Optional[nn.Module]] = [None] * len(mod_list)  # 初始化一个空的模块列表，用于存储融合后的模块结果
    fused = fuser_method(is_qat, *mod_list)  # 调用融合方法进行模块融合
    # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
    # 将基础模块的前向预处理钩子移动到融合后的模块中
    for pre_hook_fn in mod_list[0]._forward_pre_hooks.values():
        fused.register_forward_pre_hook(pre_hook_fn)
    mod_list[0]._forward_pre_hooks.clear()  # 清空第一个模块的前向预处理钩子
    # 将最后一个模块的后向钩子移动到融合后的模块中
    for hook_fn in mod_list[-1]._forward_hooks.values():
        fused.register_forward_hook(hook_fn)
    mod_list[-1]._forward_hooks.clear()  # 清空最后一个模块的后向钩子
    new_mod[0] = fused  # 将融合后的模块放入结果列表的第一个位置

    for i in range(1, len(mod_list)):
        identity = nn.Identity()  # 创建一个nn.Identity()对象
        identity.training = mod_list[0].training  # 将训练状态设置为第一个模块的训练状态
        new_mod[i] = identity  # 将Identity对象放入结果列表对应位置

    return new_mod  # 返回融合后的模块列表

def _fuse_modules_helper(model, modules_to_fuse, is_qat, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})  # 获取额外的模块融合方法映射
    mod_list = []  # 初始化一个空的模块列表
    # 对于待融合的模块列表中的每个模块，通过调用 _get_module 函数获取模块并添加到 mod_list 列表中
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))

    # 使用指定的融合函数 fuser_func 对 mod_list 中的模块进行融合，传入是否量化标志 is_qat 和额外的融合方法映射 additional_fuser_method_mapping
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

    # 将融合后的新模块列表 new_mod_list 中的模块替换回原始模型 model 中相应位置的模块
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])
# 将一组模块融合成单个模块的函数
def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    r"""Fuse a list of modules into a single module.

    # 只融合以下顺序的模块序列：
    # conv, bn
    # conv, bn, relu
    # conv, relu
    # linear, relu
    # bn, relu
    # 其他所有序列保持不变。
    # 对于这些序列，在列表中用融合后的模块替换第一个项目，
    # 其余模块用身份模块替换。

    Args:
        model: 包含要融合模块的模型
        modules_to_fuse: 要融合的模块名称列表的列表。如果只有一个要融合的模块列表，也可以是字符串列表。
        inplace: 布尔值，指定是否在模型上进行原位融合，默认情况下返回一个新模型。
        fuser_func: 接受模块列表并输出相同长度的融合模块列表的函数。
                    例如，fuser_func([convModule, BNModule]) 返回列表 [ConvBNModule, nn.Identity()]。
                    默认为 torch.ao.quantization.fuse_known_modules。
        fuse_custom_config_dict: 融合的自定义配置。

    .. code-block:: python

       # fuse_custom_config_dict 的示例
       fuse_custom_config_dict = {
           # 额外的融合方法映射
           "additional_fuser_method_mapping": {
               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
           },
       }

    Returns:
        融合后的模型。如果 inplace=True，则创建一个新副本。

    Examples::

            >>> # xdoctest: +SKIP
            >>> m = M().eval()
            >>> # m 是包含以下子模块的模块
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = M().eval()
            >>> # 或者直接提供要融合的单个模块列表
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    # 调用 _fuse_modules 函数，用于融合模块
    return _fuse_modules(
        # 将指定的模型作为参数传递给 _fuse_modules 函数
        model,
        # 指定要融合的模块列表作为参数传递给 _fuse_modules 函数
        modules_to_fuse,
        # 指定是否为量化训练（QAT），此处设为 False
        is_qat=False,
        # 指定是否原地操作（inplace），此处使用函数参数指定的值
        inplace=inplace,
        # 指定用于融合的自定义函数，作为参数传递给 _fuse_modules 函数
        fuser_func=fuser_func,
        # 指定用于融合的自定义配置字典，作为参数传递给 _fuse_modules 函数
        fuse_custom_config_dict=fuse_custom_config_dict
    )
# 使用 QAT 版本的 `fuse_modules` 函数，用于模型中指定模块的融合操作
def fuse_modules_qat(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    """QAT version for `fuse_modules`."""
    # 调用内部函数 `_fuse_modules` 进行模块融合操作
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=True,  # 指定为量化感知训练（Quantization Aware Training）模式
        inplace=inplace,  # 控制是否在原地修改模型
        fuser_func=fuser_func,  # 指定用于融合的函数，默认为 `fuse_known_modules`
        fuse_custom_config_dict=fuse_custom_config_dict  # 自定义的融合配置字典，用于进一步控制融合过程
    )
```