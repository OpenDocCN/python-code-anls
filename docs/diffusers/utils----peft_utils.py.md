# `.\diffusers\utils\peft_utils.py`

```py
# 版权声明，声明本文件的版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 在 Apache 许可证 2.0（“许可证”）下授权；
# 除非遵循该许可证，否则不得使用本文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是“按原样”提供的，
# 不提供任何形式的保证或条件，无论是明示还是暗示的。
# 有关许可证所管辖的权限和限制的具体信息，请参阅许可证。
"""
PEFT 工具：与 peft 库相关的工具
"""

# 导入 collections 模块以便于使用集合相关的功能
import collections
# 导入 importlib 模块以支持动态导入
import importlib
# 从 typing 导入 Optional 类型以用于类型注释
from typing import Optional

# 导入 version 模块以处理版本相关的功能
from packaging import version

# 从当前包导入 utils，检查 peft 和 torch 库是否可用
from .import_utils import is_peft_available, is_torch_available

# 如果 torch 库可用，则导入 torch 模块
if is_torch_available():
    import torch

# 定义函数以递归地替换模型中的 LoraLayer 实例
def recurse_remove_peft_layers(model):
    r"""
    递归替换模型中所有 `LoraLayer` 的实例为相应的新层。
    """
    # 从 peft.tuners.tuners_utils 导入 BaseTunerLayer 类
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 初始化一个标志，指示是否存在基础层模式
    has_base_layer_pattern = False
    # 遍历模型中的所有模块
    for module in model.modules():
        # 检查模块是否为 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 如果模块有名为 "base_layer" 的属性，则更新标志
            has_base_layer_pattern = hasattr(module, "base_layer")
            break

    # 如果存在基础层模式
    if has_base_layer_pattern:
        # 从 peft.utils 导入 _get_submodules 函数
        from peft.utils import _get_submodules

        # 获取所有不包含 "lora" 的模块名称列表
        key_list = [key for key, _ in model.named_modules() if "lora" not in key]
        # 遍历所有模块名称
        for key in key_list:
            try:
                # 获取当前模块的父级、目标和目标名称
                parent, target, target_name = _get_submodules(model, key)
            except AttributeError:
                # 如果发生属性错误，则继续下一个模块
                continue
            # 如果目标具有 "base_layer" 属性
            if hasattr(target, "base_layer"):
                # 用目标的基础层替换父模块中的目标
                setattr(parent, target_name, target.get_base_layer())
    else:
        # 处理与 PEFT <= 0.6.2 的向后兼容性
        # TODO: 一旦不再支持该 PEFT 版本，可以移除此代码
        from peft.tuners.lora import LoraLayer  # 导入 LoraLayer 模块

        # 遍历模型的所有子模块
        for name, module in model.named_children():
            # 如果子模块有子模块，则递归进入处理
            if len(list(module.children())) > 0:
                ## 复合模块，递归处理其内部的层
                recurse_remove_peft_layers(module)

            module_replaced = False  # 初始化标志，表示模块是否被替换

            # 检查当前模块是否为 LoraLayer 且为线性层
            if isinstance(module, LoraLayer) and isinstance(module, torch.nn.Linear):
                # 创建新的线性层，保留输入和输出特征及偏置信息
                new_module = torch.nn.Linear(
                    module.in_features,  # 输入特征数量
                    module.out_features,  # 输出特征数量
                    bias=module.bias is not None,  # 是否使用偏置
                ).to(module.weight.device)  # 将新模块移动到当前模块权重的设备上
                new_module.weight = module.weight  # 复制权重
                if module.bias is not None:
                    new_module.bias = module.bias  # 复制偏置

                module_replaced = True  # 标记模块已被替换
            # 检查当前模块是否为 LoraLayer 且为卷积层
            elif isinstance(module, LoraLayer) and isinstance(module, torch.nn.Conv2d):
                # 创建新的卷积层，保留卷积参数
                new_module = torch.nn.Conv2d(
                    module.in_channels,  # 输入通道数
                    module.out_channels,  # 输出通道数
                    module.kernel_size,  # 卷积核大小
                    module.stride,  # 步幅
                    module.padding,  # 填充
                    module.dilation,  # 膨胀
                    module.groups,  # 分组卷积
                ).to(module.weight.device)  # 将新模块移动到当前模块权重的设备上

                new_module.weight = module.weight  # 复制权重
                if module.bias is not None:
                    new_module.bias = module.bias  # 复制偏置

                module_replaced = True  # 标记模块已被替换

            # 如果模块被替换，则更新模型
            if module_replaced:
                setattr(model, name, new_module)  # 更新模型中的模块
                del module  # 删除旧模块

                # 如果可用，则清空 CUDA 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 释放 CUDA 内存
    return model  # 返回更新后的模型
# 定义一个函数来调整模型的 LoRA 层的权重
def scale_lora_layers(model, weight):
    """
    调整模型的 LoRA 层的权重。

    参数:
        model (`torch.nn.Module`):
            需要调整的模型。
        weight (`float`):
            要分配给 LoRA 层的权重。
    """
    # 从 peft.tuners.tuners_utils 导入 BaseTunerLayer 类
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 如果权重为 1.0，直接返回，不做任何调整
    if weight == 1.0:
        return

    # 遍历模型的所有模块
    for module in model.modules():
        # 检查当前模块是否是 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 调整当前 LoRA 层的权重
            module.scale_layer(weight)


# 定义一个函数来移除先前给定的 LoRA 层的权重
def unscale_lora_layers(model, weight: Optional[float] = None):
    """
    移除模型的 LoRA 层的权重。

    参数:
        model (`torch.nn.Module`):
            需要调整的模型。
        weight (`float`, *可选*):
            要分配给 LoRA 层的权重。如果未传入权重，将重新初始化 LoRA 层的权重。如果传入 0.0，将以正确的值重新初始化权重。
    """
    # 从 peft.tuners.tuners_utils 导入 BaseTunerLayer 类
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 如果权重为 1.0，直接返回，不做任何调整
    if weight == 1.0:
        return

    # 遍历模型的所有模块
    for module in model.modules():
        # 检查当前模块是否是 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 如果传入了权重且权重不为 0
            if weight is not None and weight != 0:
                # 移除当前 LoRA 层的权重
                module.unscale_layer(weight)
            # 如果传入的权重为 0
            elif weight is not None and weight == 0:
                # 遍历当前模块的所有活动适配器
                for adapter_name in module.active_adapters:
                    # 如果权重为 0，则将权重重置为原始值
                    module.set_scale(adapter_name, 1.0)


# 定义一个函数来获取 PEFT 的参数
def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    # 初始化排名模式和 alpha 模式的字典
    rank_pattern = {}
    alpha_pattern = {}
    # 获取 rank_dict 的第一个值作为 lora_alpha
    r = lora_alpha = list(rank_dict.values())[0]

    # 如果 rank_dict 中的值不全相同
    if len(set(rank_dict.values())) > 1:
        # 获取出现次数最多的 rank
        r = collections.Counter(rank_dict.values()).most_common()[0][0]

        # 对于排名与最常见排名不同的模块，将其添加到 rank_pattern 中
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        # 提取模块名称，去掉 ".lora_B." 后的部分，并保存到 rank_pattern 中
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}
    # 检查网络 alpha 字典是否不为 None 且包含元素
    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        # 检查网络 alpha 字典中是否有超过一个不同的值
        if len(set(network_alpha_dict.values())) > 1:
            # 获取出现次数最多的 alpha
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

            # 对于与出现次数最多的 alpha 不同的模块，将其添加到 `alpha_pattern`
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            # 如果是 UNet 模型
            if is_unet:
                # 处理模块名称，去掉 ".lora_A." 和 ".alpha"
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                # 处理模块名称，去掉 ".down." 后的部分
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            # 如果只有一个不同的 alpha，直接取出该值
            lora_alpha = set(network_alpha_dict.values()).pop()

    # 获取不包含 Diffusers 特定的层名称，去重
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})
    # 检查 peft_state_dict 中是否有使用 "lora_magnitude_vector"
    use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)

    # 构建 lora 配置参数字典
    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
        "use_dora": use_dora,
    }
    # 返回 lora 配置参数字典
    return lora_config_kwargs
# 获取模型中适配器的名称，根据 BaseTunerLayer 的数量返回
def get_adapter_name(model):
    # 从 PEFT 库中导入基础调优层
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 遍历模型的所有模块
    for module in model.modules():
        # 检查模块是否为 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 返回适配器的名称，格式为 "default_" 加上适配器数量
            return f"default_{len(module.r)}"
    # 如果没有找到适配器，返回默认名称 "default_0"
    return "default_0"


# 设置模型的适配层，可启用或禁用
def set_adapter_layers(model, enabled=True):
    # 从 PEFT 库中导入基础调优层
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 遍历模型的所有模块
    for module in model.modules():
        # 检查模块是否为 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 检查模块是否具备 enable_adapters 方法
            if hasattr(module, "enable_adapters"):
                # 调用 enable_adapters 方法启用或禁用适配器
                module.enable_adapters(enabled=enabled)
            else:
                # 通过禁用状态设置 disable_adapters 属性
                module.disable_adapters = not enabled


# 删除模型中的适配层
def delete_adapter_layers(model, adapter_name):
    # 从 PEFT 库中导入基础调优层
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 遍历模型的所有模块
    for module in model.modules():
        # 检查模块是否为 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 检查模块是否具备 delete_adapter 方法
            if hasattr(module, "delete_adapter"):
                # 调用 delete_adapter 方法删除指定适配器
                module.delete_adapter(adapter_name)
            else:
                # 抛出错误，提示 PEFT 版本不兼容
                raise ValueError(
                    "The version of PEFT you are using is not compatible, please use a version that is greater than 0.6.1"
                )

    # 检查模型是否已加载 PEFT 配置
    if getattr(model, "_hf_peft_config_loaded", False) and hasattr(model, "peft_config"):
        # 从配置中删除适配器
        model.peft_config.pop(adapter_name, None)
        # 如果所有适配器都已删除，删除配置并重置标志
        if len(model.peft_config) == 0:
            del model.peft_config
            model._hf_peft_config_loaded = None


# 设置适配器的权重并激活它们
def set_weights_and_activate_adapters(model, adapter_names, weights):
    # 从 PEFT 库中导入基础调优层
    from peft.tuners.tuners_utils import BaseTunerLayer

    # 定义获取模块权重的内部函数
    def get_module_weight(weight_for_adapter, module_name):
        # 如果权重不是字典，直接返回该权重
        if not isinstance(weight_for_adapter, dict):
            return weight_for_adapter

        # 遍历权重字典，查找对应模块的权重
        for layer_name, weight_ in weight_for_adapter.items():
            if layer_name in module_name:
                return weight_

        # 分割模块名称为部分
        parts = module_name.split(".")
        # 生成关键字以获取块权重
        key = f"{parts[0]}.{parts[1]}.attentions.{parts[3]}"
        block_weight = weight_for_adapter.get(key, 1.0)

        return block_weight

    # 遍历每个适配器，使其激活并设置对应的缩放权重
    for adapter_name, weight in zip(adapter_names, weights):
        for module_name, module in model.named_modules():
            # 检查模块是否为 BaseTunerLayer 的实例
            if isinstance(module, BaseTunerLayer):
                # 检查模块是否具备 set_adapter 方法，以兼容旧版本
                if hasattr(module, "set_adapter"):
                    # 设置适配器名称
                    module.set_adapter(adapter_name)
                else:
                    # 设置当前激活的适配器名称
                    module.active_adapter = adapter_name
                # 设置适配器的缩放权重
                module.set_scale(adapter_name, get_module_weight(weight, module_name))

    # 设置多个激活的适配器
    # 遍历模型中的所有模块
        for module in model.modules():
            # 检查当前模块是否为 BaseTunerLayer 的实例
            if isinstance(module, BaseTunerLayer):
                # 为了与以前的 PEFT 版本保持向后兼容
                if hasattr(module, "set_adapter"):
                    # 如果模块具有 set_adapter 方法，则调用该方法并传入适配器名称
                    module.set_adapter(adapter_names)
                else:
                    # 如果没有 set_adapter 方法，则直接设置 active_adapter 属性
                    module.active_adapter = adapter_names
# 检查 PEFT 的版本是否兼容
def check_peft_version(min_version: str) -> None:
    # 文档字符串，说明该函数的作用和参数
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    # 检查 PEFT 是否可用，若不可用则抛出异常
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    # 获取当前 PEFT 版本并与最小版本进行比较
    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) > version.parse(min_version)

    # 若版本不兼容，则抛出异常并提示用户使用兼容版本
    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )
```