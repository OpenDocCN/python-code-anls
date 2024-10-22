# `.\diffusers\models\model_loading_utils.py`

```py
# 指定编码为 UTF-8
# coding=utf-8
# 版权声明，表明此文件的版权归 HuggingFace Inc. 团队所有
# Copyright 2024 The HuggingFace Inc. team.
# 版权声明，表明此文件的版权归 NVIDIA CORPORATION 所有
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# 根据 Apache 许可证第 2.0 版进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 使用此文件必须遵守许可证
# you may not use this file except in compliance with the License.
# 可以在此处获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在 "AS IS" 基础上分发
# Unless required by applicable law or agreed to in writing, software
# 不提供任何明示或暗示的担保或条件
# distributed under the License is distributed on an "AS IS" BASIS,
# 查看许可证以获取特定权限和限制的详细信息
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入标准库中的 importlib 模块
import importlib
# 导入 inspect 模块，用于检查对象
import inspect
# 导入操作系统模块
import os
# 从 collections 导入 OrderedDict，用于保持字典的顺序
from collections import OrderedDict
# 从 pathlib 导入 Path，处理文件路径
from pathlib import Path
# 导入 List、Optional 和 Union 类型提示
from typing import List, Optional, Union

# 导入 safetensors 模块
import safetensors
# 导入 PyTorch 库
import torch
# 从 huggingface_hub.utils 导入 EntryNotFoundError 异常
from huggingface_hub.utils import EntryNotFoundError

# 从 utils 模块中导入常量和函数
from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    WEIGHTS_INDEX_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    logging,
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义类重映射字典，将旧类名映射到新类名
_CLASS_REMAPPING_DICT = {
    "Transformer2DModel": {
        "ada_norm_zero": "DiTTransformer2DModel",
        "ada_norm_single": "PixArtTransformer2DModel",
    }
}

# 如果可用，导入加速库的相关功能
if is_accelerate_available():
    from accelerate import infer_auto_device_map
    from accelerate.utils import get_balanced_memory, get_max_memory, set_module_tensor_to_device

# 根据模型和设备映射确定设备映射
# Adapted from `transformers` (see modeling_utils.py)
def _determine_device_map(model: torch.nn.Module, device_map, max_memory, torch_dtype):
    # 如果 device_map 是字符串，获取不拆分模块
    if isinstance(device_map, str):
        no_split_modules = model._get_no_split_modules(device_map)
        device_map_kwargs = {"no_split_module_classes": no_split_modules}

        # 如果 device_map 不是 "sequential"，计算平衡内存
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                dtype=torch_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        # 否则获取最大内存
        else:
            max_memory = get_max_memory(max_memory)

        # 更新 device_map 参数并推断设备映射
        device_map_kwargs["max_memory"] = max_memory
        device_map = infer_auto_device_map(model, dtype=torch_dtype, **device_map_kwargs)

    # 返回最终的设备映射
    return device_map

# 从配置中获取重映射的类
def _fetch_remapped_cls_from_config(config, old_class):
    # 获取旧类的名称
    previous_class_name = old_class.__name__
    # 根据配置中的 norm_type 查找重映射的类名
    remapped_class_name = _CLASS_REMAPPING_DICT.get(previous_class_name).get(config["norm_type"], None)

    # 详细信息：
    # https://github.com/huggingface/diffusers/pull/7647#discussion_r1621344818
    # 如果 remapped_class_name 存在
        if remapped_class_name:
            # 加载 diffusers 库以导入兼容的原始调度器
            diffusers_library = importlib.import_module(__name__.split(".")[0])
            # 从 diffusers 库中获取 remapped_class_name 指定的类
            remapped_class = getattr(diffusers_library, remapped_class_name)
            # 记录日志，说明类对象正在更改，因之前的类将在未来版本中弃用
            logger.info(
                f"Changing class object to be of `{remapped_class_name}` type from `{previous_class_name}` type."
                f"This is because `{previous_class_name}` is scheduled to be deprecated in a future version. Note that this"
                " DOESN'T affect the final results."
            )
            # 返回映射后的类
            return remapped_class
        else:
            # 如果没有 remapped_class_name，返回旧类
            return old_class
# 定义一个函数，用于加载检查点文件，返回格式化的错误信息（如有）
def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    读取检查点文件，如果出现错误，则返回正确格式的错误信息。
    """
    try:
        # 获取检查点文件名的扩展名
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        # 如果文件扩展名是 SAFETENSORS_FILE_EXTENSION，则使用 safetensors 加载文件
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
        else:
            # 检查 PyTorch 版本，如果大于等于 1.13，则设置 weights_only 参数
            weights_only_kwarg = {"weights_only": True} if is_torch_version(">=", "1.13") else {}
            # 加载检查点文件，并将模型权重映射到 CPU
            return torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
            )
    except Exception as e:
        try:
            # 尝试打开检查点文件
            with open(checkpoint_file) as f:
                # 检查文件是否以 "version" 开头，以确定是否缺少 git-lfs
                if f.read().startswith("version"):
                    raise OSError(
                        "您似乎克隆了一个没有安装 git-lfs 的库。请安装 "
                        "git-lfs 并在克隆的文件夹中运行 `git lfs install` 以及 `git lfs pull`。"
                    )
                else:
                    # 如果文件不存在，抛出 ValueError
                    raise ValueError(
                        f"无法找到加载此预训练模型所需的文件 {checkpoint_file}。请确保已正确保存模型。"
                    ) from e
        except (UnicodeDecodeError, ValueError):
            # 如果读取文件时出现错误，抛出 OSError
            raise OSError(
                f"无法从检查点文件加载权重 '{checkpoint_file}' " f"在 '{checkpoint_file}'。"
            )


# 定义一个函数，将模型状态字典加载到元数据中
def load_model_dict_into_meta(
    model,
    state_dict: OrderedDict,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    model_name_or_path: Optional[str] = None,
) -> List[str]:
    # 如果未提供设备，则默认使用 CPU
    device = device or torch.device("cpu")
    # 如果未提供数据类型，则默认使用 float32
    dtype = dtype or torch.float32

    # 检查 set_module_tensor_to_device 函数是否接受 dtype 参数
    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())

    # 初始化一个列表以存储意外的键
    unexpected_keys = []
    # 获取模型的空状态字典
    empty_state_dict = model.state_dict()
    # 遍历状态字典中的每个参数名称和对应的参数值
    for param_name, param in state_dict.items():
        # 如果参数名称不在空状态字典中，则记录为意外的键
        if param_name not in empty_state_dict:
            unexpected_keys.append(param_name)
            continue  # 跳过本次循环，继续下一个参数

        # 检查空状态字典中对应参数的形状是否与当前参数的形状匹配
        if empty_state_dict[param_name].shape != param.shape:
            # 如果模型路径存在，则格式化字符串以包含模型路径
            model_name_or_path_str = f"{model_name_or_path} " if model_name_or_path is not None else ""
            # 抛出值错误，提示参数形状不匹配，并给出解决方案和参考链接
            raise ValueError(
                f"Cannot load {model_name_or_path_str}because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example."
            )

        # 如果接受数据类型，则将参数设置到模型的指定设备上，并指定数据类型
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, device, value=param, dtype=dtype)
        else:
            # 如果不接受数据类型，则仅将参数设置到模型的指定设备上
            set_module_tensor_to_device(model, param_name, device, value=param)
    # 返回意外的键列表
    return unexpected_keys
# 定义一个函数，将状态字典加载到模型中，并返回错误信息列表
def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # 如果需要，从 PyTorch 的 state_dict 转换旧格式到新格式
    # 复制 state_dict，以便 _load_from_state_dict 可以对其进行修改
    state_dict = state_dict.copy()
    # 用于存储加载过程中的错误信息
    error_msgs = []

    # PyTorch 的 `_load_from_state_dict` 不会复制模块子孙中的参数
    # 所以我们需要递归地应用这个函数
    def load(module: torch.nn.Module, prefix: str = ""):
        # 准备参数，调用模块的 `_load_from_state_dict` 方法
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        # 遍历模块的所有子模块
        for name, child in module._modules.items():
            # 如果子模块存在，递归加载
            if child is not None:
                load(child, prefix + name + ".")

    # 初始调用加载模型
    load(model_to_load)

    # 返回所有错误信息
    return error_msgs


# 定义一个函数，获取索引文件的路径
def _fetch_index_file(
    is_local,
    pretrained_model_name_or_path,
    subfolder,
    use_safetensors,
    cache_dir,
    variant,
    force_download,
    proxies,
    local_files_only,
    token,
    revision,
    user_agent,
    commit_hash,
):
    # 如果是本地文件
    if is_local:
        # 构造索引文件的路径
        index_file = Path(
            pretrained_model_name_or_path,
            subfolder or "",  # 如果子文件夹为空，则使用空字符串
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        )
    else:
        # 构造索引文件在远程仓库中的路径
        index_file_in_repo = Path(
            subfolder or "",  # 如果子文件夹为空，则使用空字符串
            _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME, variant),
        ).as_posix()  # 转换为 POSIX 路径格式
        try:
            # 获取模型文件的路径
            index_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=index_file_in_repo,  # 指定权重文件名
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=None,  # 子文件夹为 None
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            # 将返回的路径转换为 Path 对象
            index_file = Path(index_file)
        except (EntryNotFoundError, EnvironmentError):
            # 如果找不到文件或发生环境错误，将索引文件设置为 None
            index_file = None

    # 返回索引文件的路径
    return index_file
```