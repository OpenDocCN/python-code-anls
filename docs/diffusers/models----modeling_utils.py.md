# `.\diffusers\models\modeling_utils.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # HuggingFace Inc. 团队的版权声明
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.  # NVIDIA 的版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 指定此文件使用 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 使用此文件需要遵循许可证的规定
# You may obtain a copy of the License at  # 可以在以下网址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的具体链接
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 否则按 "现状" 基础分发软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何形式的担保或条件
# See the License for the specific language governing permissions and  # 参见许可证了解特定权限和限制
# limitations under the License.  # 以及许可证下的限制

import inspect  # 导入 inspect 模块，用于获取对象的信息
import itertools  # 导入 itertools 模块，提供高效的迭代器
import json  # 导入 json 模块，用于 JSON 数据的解析和生成
import os  # 导入 os 模块，提供与操作系统交互的功能
import re  # 导入 re 模块，提供正则表达式操作
from collections import OrderedDict  # 从 collections 导入有序字典
from functools import partial  # 从 functools 导入部分函数应用工具
from pathlib import Path  # 从 pathlib 导入路径处理工具
from typing import Any, Callable, List, Optional, Tuple, Union  # 导入类型注解支持

import safetensors  # 导入 safetensors 库，处理安全的张量
import torch  # 导入 PyTorch 库
from huggingface_hub import create_repo, split_torch_state_dict_into_shards  # 从 huggingface_hub 导入相关功能
from huggingface_hub.utils import validate_hf_hub_args  # 导入验证 Hugging Face Hub 参数的工具
from torch import Tensor, nn  # 从 torch 导入 Tensor 和神经网络模块

from .. import __version__  # 从父级模块导入当前版本
from ..utils import (  # 从父级模块的 utils 导入多个工具
    CONFIG_NAME,  # 配置文件名常量
    FLAX_WEIGHTS_NAME,  # Flax 权重文件名常量
    SAFE_WEIGHTS_INDEX_NAME,  # 安全权重索引文件名常量
    SAFETENSORS_WEIGHTS_NAME,  # Safetensors 权重文件名常量
    WEIGHTS_INDEX_NAME,  # 权重索引文件名常量
    WEIGHTS_NAME,  # 权重文件名常量
    _add_variant,  # 导入添加变体的工具
    _get_checkpoint_shard_files,  # 导入获取检查点分片文件的工具
    _get_model_file,  # 导入获取模型文件的工具
    deprecate,  # 导入弃用标记的工具
    is_accelerate_available,  # 导入检测加速库可用性的工具
    is_torch_version,  # 导入检测 PyTorch 版本的工具
    logging,  # 导入日志记录工具
)
from ..utils.hub_utils import (  # 从父级模块的 hub_utils 导入多个工具
    PushToHubMixin,  # 导入用于推送到 Hub 的混合类
    load_or_create_model_card,  # 导入加载或创建模型卡的工具
    populate_model_card,  # 导入填充模型卡的工具
)
from .model_loading_utils import (  # 从当前包的 model_loading_utils 导入多个工具
    _determine_device_map,  # 导入确定设备映射的工具
    _fetch_index_file,  # 导入获取索引文件的工具
    _load_state_dict_into_model,  # 导入将状态字典加载到模型中的工具
    load_model_dict_into_meta,  # 导入将模型字典加载到元数据中的工具
    load_state_dict,  # 导入加载状态字典的工具
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")  # 编译正则表达式，用于匹配分片文件名

if is_torch_version(">=", "1.9.0"):  # 检查当前 PyTorch 版本是否大于等于 1.9.0
    _LOW_CPU_MEM_USAGE_DEFAULT = True  # 设置低 CPU 内存使用默认值为 True
else:  # 如果 PyTorch 版本小于 1.9.0
    _LOW_CPU_MEM_USAGE_DEFAULT = False  # 设置低 CPU 内存使用默认值为 False

if is_accelerate_available():  # 检查加速库是否可用
    import accelerate  # 如果可用，则导入 accelerate 库

def get_parameter_device(parameter: torch.nn.Module) -> torch.device:  # 定义获取模型参数设备的函数
    try:  # 尝试执行以下代码
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())  # 合并模型参数和缓冲区
        return next(parameters_and_buffers).device  # 返回第一个参数或缓冲区的设备
    except StopIteration:  # 如果没有参数和缓冲区
        # For torch.nn.DataParallel compatibility in PyTorch 1.5  # 为兼容 PyTorch 1.5 的 DataParallel

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:  # 定义查找张量属性的内部函数
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]  # 获取模块中所有张量属性
            return tuples  # 返回张量属性的列表

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)  # 获取模型的命名成员生成器
        first_tuple = next(gen)  # 获取生成器中的第一个元组
        return first_tuple[1].device  # 返回第一个张量的设备

def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:  # 定义获取模型参数数据类型的函数
    try:  # 尝试执行以下代码
        params = tuple(parameter.parameters())  # 将模型参数转换为元组
        if len(params) > 0:  # 如果参数数量大于零
            return params[0].dtype  # 返回第一个参数的数据类型

        buffers = tuple(parameter.buffers())  # 将缓冲区转换为元组
        if len(buffers) > 0:  # 如果缓冲区数量大于零
            return buffers[0].dtype  # 返回第一个缓冲区的数据类型
    # 捕获 StopIteration 异常，处理迭代器停止的情况
    except StopIteration:
        # 为了兼容 PyTorch 1.5 中的 torch.nn.DataParallel

        # 定义一个函数，用于查找模块中所有的张量属性，返回属性名和张量的元组列表
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            # 生成一个元组列表，包含模块中所有张量属性的名称和对应的张量
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            # 返回元组列表
            return tuples

        # 使用指定的函数获取模块的命名成员生成器
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取生成器中的第一个元组
        first_tuple = next(gen)
        # 返回第一个张量的 dtype（数据类型）
        return first_tuple[1].dtype
# 定义一个模型混合类，继承自 PyTorch 的 nn.Module 和 PushToHubMixin
class ModelMixin(torch.nn.Module, PushToHubMixin):
    r"""
    所有模型的基类。

    [`ModelMixin`] 负责存储模型配置，并提供加载、下载和保存模型的方法。

        - **config_name** ([`str`]) -- 保存模型时的文件名，调用 [`~models.ModelMixin.save_pretrained`]。
    """

    # 配置名称，作为模型保存时的文件名
    config_name = CONFIG_NAME
    # 自动保存的参数列表
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    # 是否支持梯度检查点
    _supports_gradient_checkpointing = False
    # 加载时忽略的意外键
    _keys_to_ignore_on_load_unexpected = None
    # 不分割的模块
    _no_split_modules = None

    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 重写 getattr 方法以优雅地弃用直接访问配置属性
    def __getattr__(self, name: str) -> Any:
        """重写 `getattr` 的唯一原因是优雅地弃用直接访问配置属性。
        参见 https://github.com/huggingface/diffusers/pull/3129 需要在这里重写
        __getattr__，以免触发 `torch.nn.Module` 的 __getattr__：
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        # 检查属性是否在内部字典中，并且是否存在于内部字典的属性中
        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        # 检查属性是否在当前实例的字典中
        is_attribute = name in self.__dict__

        # 如果属性在配置中且不在实例字典中，显示弃用警告
        if is_in_config and not is_attribute:
            # 构建弃用消息
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."
            # 调用弃用函数显示警告
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False, stacklevel=3)
            # 返回内部字典中的属性值
            return self._internal_dict[name]

        # 调用 PyTorch 的原始 __getattr__ 方法
        return super().__getattr__(name)

    # 定义一个只读属性，检查是否启用了梯度检查点
    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        检查该模型是否启用了梯度检查点。
        """
        # 遍历模型中的所有模块，检查是否有启用梯度检查点的模块
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    # 启用梯度检查点的方法
    def enable_gradient_checkpointing(self) -> None:
        """
        启用当前模型的梯度检查点（在其他框架中可能称为 *激活检查点* 或
        *检查点激活*）。
        """
        # 检查当前模型是否支持梯度检查点
        if not self._supports_gradient_checkpointing:
            # 如果不支持，抛出值错误
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        # 应用设置，启用梯度检查点
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    # 禁用梯度检查点的方法
    def disable_gradient_checkpointing(self) -> None:
        """
        禁用当前模型的梯度检查点（在其他框架中可能称为 *激活检查点* 或
        *检查点激活*）。
        """
        # 检查当前模型是否支持梯度检查点
        if self._supports_gradient_checkpointing:
            # 应用设置，禁用梯度检查点
            self.apply(partial(self._set_gradient_checkpointing, value=False))
    # 定义一个设置 npu flash attention 开关的方法，接收布尔值 valid
    def set_use_npu_flash_attention(self, valid: bool) -> None:
        r""" 
        设置 npu flash attention 的开关。
        """
    
        # 定义一个递归设置 npu flash attention 的内部方法，接收一个模块
        def fn_recursive_set_npu_flash_attention(module: torch.nn.Module):
            # 如果模块有设置 npu flash attention 的方法，则调用它
            if hasattr(module, "set_use_npu_flash_attention"):
                module.set_use_npu_flash_attention(valid)
    
            # 递归遍历模块的所有子模块
            for child in module.children():
                fn_recursive_set_npu_flash_attention(child)
    
        # 遍历当前对象的所有子模块
        for module in self.children():
            # 如果子模块是一个 torch.nn.Module 类型，则调用递归方法
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_npu_flash_attention(module)
    
    # 定义一个启用 npu flash attention 的方法
    def enable_npu_flash_attention(self) -> None:
        r""" 
        启用来自 torch_npu 的 npu flash attention。
        """
        # 调用设置方法，将开关置为 True
        self.set_use_npu_flash_attention(True)
    
    # 定义一个禁用 npu flash attention 的方法
    def disable_npu_flash_attention(self) -> None:
        r""" 
        禁用来自 torch_npu 的 npu flash attention。
        """
        # 调用设置方法，将开关置为 False
        self.set_use_npu_flash_attention(False)
    
    # 定义一个设置内存高效注意力的 xformers 方法，接收布尔值 valid 和可选的注意力操作
    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # 递归遍历所有子模块。
        # 任何暴露 set_use_memory_efficient_attention_xformers 方法的子模块都会接收到消息
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            # 如果模块有设置内存高效注意力的方法，则调用它
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)
    
            # 递归遍历模块的所有子模块
            for child in module.children():
                fn_recursive_set_mem_eff(child)
    
        # 遍历当前对象的所有子模块
        for module in self.children():
            # 如果子模块是一个 torch.nn.Module 类型，则调用递归方法
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)
    # 启用来自 xFormers 的内存高效注意力
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None) -> None:
            # 文档字符串，描述该方法的功能和使用示例
            r"""
            Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
    
            When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
            inference. Speed up during training is not guaranteed.
    
            <Tip warning={true}>
    
            ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
            precedent.
    
            </Tip>
    
            Parameters:
                attention_op (`Callable`, *optional*):
                    Override the default `None` operator for use as `op` argument to the
                    [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                    function of xFormers.
    
            Examples:
    
            ```py
            >>> import torch
            >>> from diffusers import UNet2DConditionModel
            >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
    
            >>> model = UNet2DConditionModel.from_pretrained(
            ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float16
            ... )
            >>> model = model.to("cuda")
            >>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            ```py
            """
            # 设置使用 xFormers 的内存高效注意力，传入可选的注意力操作
            self.set_use_memory_efficient_attention_xformers(True, attention_op)
    
        # 禁用来自 xFormers 的内存高效注意力
        def disable_xformers_memory_efficient_attention(self) -> None:
            # 文档字符串，描述该方法的功能
            r"""
            Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
            """
            # 设置不使用 xFormers 的内存高效注意力
            self.set_use_memory_efficient_attention_xformers(False)
    
        # 保存预训练模型的方法
        def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            save_function: Optional[Callable] = None,
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            max_shard_size: Union[int, str] = "10GB",
            push_to_hub: bool = False,
            **kwargs,
        @classmethod
        # 类方法，加载预训练模型
        @validate_hf_hub_args
        @classmethod
        def _load_pretrained_model(
            cls,
            model,
            state_dict: OrderedDict,
            resolved_archive_file,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            ignore_mismatched_sizes: bool = False,
        @classmethod
        # 获取对象的构造函数签名参数
        def _get_signature_keys(cls, obj):
            # 获取构造函数的参数字典
            parameters = inspect.signature(obj.__init__).parameters
            # 提取必需的参数
            required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
            # 提取可选参数
            optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
            # 计算期望的模块，排除 'self'
            expected_modules = set(required_parameters.keys()) - {"self"}
    
            return expected_modules, optional_parameters
    
        # 从 transformers 的 modeling_utils.py 修改而来
    # 定义一个私有方法，用于获取在使用 device_map 时不应拆分的模块
    def _get_no_split_modules(self, device_map: str):
        """
        获取模型中在使用 device_map 时不应拆分的模块。我们遍历模块以获取底层的 `_no_split_modules`。
    
        参数:
            device_map (`str`):
                设备映射值。选项包括 ["auto", "balanced", "balanced_low_0", "sequential"]
    
        返回:
            `List[str]`: 不应拆分的模块列表
        """
        # 初始化一个集合，用于存储不应拆分的模块
        _no_split_modules = set()
        # 将当前对象添加到待检查的模块列表中
        modules_to_check = [self]
        # 当待检查模块列表不为空时继续循环
        while len(modules_to_check) > 0:
            # 从待检查列表中弹出最后一个模块
            module = modules_to_check.pop(-1)
            # 如果模块不在不应拆分的模块集合中，检查其子模块
            if module.__class__.__name__ not in _no_split_modules:
                # 如果模块是 ModelMixin 的实例
                if isinstance(module, ModelMixin):
                    # 如果模块的 `_no_split_modules` 属性为 None，抛出异常
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    # 否则，将模块的不应拆分模块添加到集合中
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                # 将当前模块的所有子模块添加到待检查列表中
                modules_to_check += list(module.children())
        # 返回不应拆分模块的列表
        return list(_no_split_modules)
    
    # 定义一个属性，用于获取模块所在的设备
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: 模块所在的设备（假设所有模块参数在同一设备上）。
        """
        # 调用函数获取当前对象的参数设备
        return get_parameter_device(self)
    
    # 定义一个属性，用于获取模块的数据类型
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: 模块的数据类型（假设所有模块参数具有相同的数据类型）。
        """
        # 调用函数获取当前对象的参数数据类型
        return get_parameter_dtype(self)
    # 定义一个方法，用于获取模块中的参数数量
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        获取模块中（可训练或非嵌入）参数的数量。
    
        参数：
            only_trainable (`bool`, *可选*, 默认为 `False`):
                是否仅返回可训练参数的数量。
            exclude_embeddings (`bool`, *可选*, 默认为 `False`):
                是否仅返回非嵌入参数的数量。
    
        返回：
            `int`: 参数的数量。
    
        示例：
    
        ```py
        from diffusers import UNet2DConditionModel
    
        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        unet.num_parameters(only_trainable=True)
        859520964
        ```py
        """
    
        # 如果排除嵌入参数
        if exclude_embeddings:
            # 获取所有嵌入层的参数名
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            # 筛选出非嵌入参数
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            # 返回所有非嵌入参数的数量（可训练或非可训练）
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            # 返回所有参数的数量（可训练或非可训练）
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
    # 定义一个方法，用于转换过时的注意力块
    def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) -> None:
        # 初始化一个列表，用于存储过时注意力块的路径
        deprecated_attention_block_paths = []

        # 定义一个递归函数，用于查找过时的注意力块
        def recursive_find_attn_block(name, module):
            # 检查当前模块是否是过时的注意力块
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                # 将找到的模块名称添加到路径列表中
                deprecated_attention_block_paths.append(name)

            # 遍历模块的子模块
            for sub_name, sub_module in module.named_children():
                # 形成完整的子模块名称
                sub_name = sub_name if name == "" else f"{name}.{sub_name}"
                # 递归查找子模块
                recursive_find_attn_block(sub_name, sub_module)

        # 从当前对象开始递归查找过时的注意力块
        recursive_find_attn_block("", self)

        # 注意：需要检查过时参数是否在状态字典中
        # 因为可能加载的是已经转换过的状态字典

        # 遍历所有找到的过时注意力块路径
        for path in deprecated_attention_block_paths:
            # group_norm 路径保持不变

            # 将 query 参数转换为 to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(f"{path}.query.bias")

            # 将 key 参数转换为 to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(f"{path}.key.bias")

            # 将 value 参数转换为 to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(f"{path}.value.bias")

            # 将 proj_attn 参数转换为 to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(f"{path}.proj_attn.bias")
    # 将当前对象的注意力模块转换为已弃用的注意力块
    def _temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        # 初始化一个列表，用于存储已弃用的注意力块模块
        deprecated_attention_block_modules = []
    
        # 定义递归函数以查找注意力块模块
        def recursive_find_attn_block(module):
            # 检查模块是否为已弃用的注意力块
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                # 将找到的模块添加到列表中
                deprecated_attention_block_modules.append(module)
    
            # 遍历子模块并递归调用
            for sub_module in module.children():
                recursive_find_attn_block(sub_module)
    
        # 从当前对象开始递归查找
        recursive_find_attn_block(self)
    
        # 遍历所有已弃用的注意力块模块
        for module in deprecated_attention_block_modules:
            # 将新属性赋值给相应的旧属性
            module.query = module.to_q
            module.key = module.to_k
            module.value = module.to_v
            module.proj_attn = module.to_out[0]
    
            # 删除旧属性以确保所有权重都加载到新属性中
            del module.to_q
            del module.to_k
            del module.to_v
            del module.to_out
    
    # 将已弃用的注意力块模块恢复为当前对象的注意力模块
    def _undo_temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        # 初始化一个列表，用于存储已弃用的注意力块模块
        deprecated_attention_block_modules = []
    
        # 定义递归函数以查找注意力块模块
        def recursive_find_attn_block(module) -> None:
            # 检查模块是否为已弃用的注意力块
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                # 将找到的模块添加到列表中
                deprecated_attention_block_modules.append(module)
    
            # 遍历子模块并递归调用
            for sub_module in module.children():
                recursive_find_attn_block(sub_module)
    
        # 从当前对象开始递归查找
        recursive_find_attn_block(self)
    
        # 遍历所有已弃用的注意力块模块
        for module in deprecated_attention_block_modules:
            # 将旧属性赋值给相应的新属性
            module.to_q = module.query
            module.to_k = module.key
            module.to_v = module.value
            module.to_out = nn.ModuleList([module.proj_attn, nn.Dropout(module.dropout)])
    
            # 删除新属性以恢复旧的模块结构
            del module.query
            del module.key
            del module.value
            del module.proj_attn
# 定义一个继承自 ModelMixin 的类，用于处理从旧类到特定管道类的映射
class LegacyModelMixin(ModelMixin):
    r"""
    一个 `ModelMixin` 的子类，用于从旧类（如 `Transformer2DModel`）解析到更具体的管道类（如 `DiTTransformer2DModel`）的类映射。
    """

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # 为了避免依赖导入问题
        from .model_loading_utils import _fetch_remapped_cls_from_config

        # 创建 kwargs 的副本，以避免对后续调用中的关键字参数造成影响
        kwargs_copy = kwargs.copy()

        # 从 kwargs 中提取 cache_dir 参数，若未提供则为 None
        cache_dir = kwargs.pop("cache_dir", None)
        # 从 kwargs 中提取 force_download 参数，默认为 False
        force_download = kwargs.pop("force_download", False)
        # 从 kwargs 中提取 proxies 参数，默认为 None
        proxies = kwargs.pop("proxies", None)
        # 从 kwargs 中提取 local_files_only 参数，默认为 None
        local_files_only = kwargs.pop("local_files_only", None)
        # 从 kwargs 中提取 token 参数，默认为 None
        token = kwargs.pop("token", None)
        # 从 kwargs 中提取 revision 参数，默认为 None
        revision = kwargs.pop("revision", None)
        # 从 kwargs 中提取 subfolder 参数，默认为 None
        subfolder = kwargs.pop("subfolder", None)

        # 如果未提供配置，则将配置路径设置为预训练模型名称或路径
        config_path = pretrained_model_name_or_path

        # 设置用户代理信息
        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # 加载配置
        config, _, _ = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        # 解析类的映射
        remapped_class = _fetch_remapped_cls_from_config(config, cls)

        # 返回映射后的类的 from_pretrained 方法调用
        return remapped_class.from_pretrained(pretrained_model_name_or_path, **kwargs_copy)
```