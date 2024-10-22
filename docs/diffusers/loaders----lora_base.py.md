# `.\diffusers\loaders\lora_base.py`

```py
# 版权声明，指明版权持有者及其权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 版本授权该文件，用户需遵守该授权
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在符合该许可证的情况下使用此文件
# 许可证副本可在以下地址获取
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则根据许可证分发的软件是按“原样”提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入标准库中的复制模块
import copy
# 导入检查模块以获取对象的签名和源代码
import inspect
# 导入操作系统模块以处理文件和目录
import os
# 从路径库导入 Path 类以方便路径操作
from pathlib import Path
# 从类型库导入所需的类型注解
from typing import Callable, Dict, List, Optional, Union

# 导入 safetensors 库以处理安全张量
import safetensors
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 huggingface_hub 导入模型信息获取函数
from huggingface_hub import model_info
# 从 huggingface_hub 导入常量以指示离线模式
from huggingface_hub.constants import HF_HUB_OFFLINE

# 从父模块导入模型混合相关的工具和函数
from ..models.modeling_utils import ModelMixin, load_state_dict
# 从工具模块导入多个实用函数和常量
from ..utils import (
    USE_PEFT_BACKEND,              # 是否使用 PEFT 后端的标志
    _get_model_file,               # 获取模型文件的函数
    delete_adapter_layers,         # 删除适配器层的函数
    deprecate,                     # 用于标记弃用功能的函数
    is_accelerate_available,       # 检查 accelerate 是否可用的函数
    is_peft_available,             # 检查 PEFT 是否可用的函数
    is_transformers_available,     # 检查 transformers 是否可用的函数
    logging,                       # 日志模块
    recurse_remove_peft_layers,    # 递归删除 PEFT 层的函数
    set_adapter_layers,            # 设置适配器层的函数
    set_weights_and_activate_adapters, # 设置权重并激活适配器的函数
)

# 如果 transformers 可用，则导入 PreTrainedModel 类
if is_transformers_available():
    from transformers import PreTrainedModel

# 如果 PEFT 可用，则导入 BaseTunerLayer 类
if is_peft_available():
    from peft.tuners.tuners_utils import BaseTunerLayer

# 如果 accelerate 可用，则导入相关的钩子
if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

# 创建一个日志记录器实例，用于当前模块
logger = logging.get_logger(__name__)

# 定义一个函数以融合文本编码器的 LoRA
def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
    """
    融合文本编码器的 LoRA。

    参数：
        text_encoder (`torch.nn.Module`):
            要设置适配器层的文本编码器模块。如果为 `None`，则会尝试获取 `text_encoder`
            属性。
        lora_scale (`float`, defaults to 1.0):
            控制 LoRA 参数对输出的影响程度。
        safe_fusing (`bool`, defaults to `False`):
            是否在融合之前检查融合的权重是否为 NaN 值，如果为 NaN 则不进行融合。
        adapter_names (`List[str]` 或 `str`):
            要使用的适配器名称列表。
    """
    # 定义合并参数字典，包含安全合并选项
    merge_kwargs = {"safe_merge": safe_fusing}
    # 遍历文本编码器中的所有模块
    for module in text_encoder.modules():
        # 检查当前模块是否是 BaseTunerLayer 类型
        if isinstance(module, BaseTunerLayer):
            # 如果 lora_scale 不是 1.0，则对当前模块进行缩放
            if lora_scale != 1.0:
                module.scale_layer(lora_scale)

            # 为了与之前的 PEFT 版本兼容，检查 `merge` 方法的签名
            # 以查看是否支持 `adapter_names` 参数
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            # 如果 `adapter_names` 参数被支持，则将其添加到合并参数中
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            # 如果不支持 `adapter_names` 且其值不为 None，则抛出错误
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    # 抛出的错误信息，提示用户升级 PEFT 版本
                    "The `adapter_names` argument is not supported with your PEFT version. "
                    "Please upgrade to the latest version of PEFT. `pip install -U peft`"
                )

            # 调用模块的 merge 方法，使用合并参数进行合并
            module.merge(**merge_kwargs)
# 解锁文本编码器的 LoRA 层
def unfuse_text_encoder_lora(text_encoder):
    """
    解锁文本编码器的 LoRA 层。

    参数：
        text_encoder (`torch.nn.Module`):
            要设置适配器层的文本编码器模块。如果为 `None`，将尝试获取 `text_encoder` 属性。
    """
    # 遍历文本编码器中的所有模块
    for module in text_encoder.modules():
        # 检查当前模块是否是 BaseTunerLayer 的实例
        if isinstance(module, BaseTunerLayer):
            # 对于符合条件的模块，解除合并操作
            module.unmerge()


# 设置文本编码器的适配器层
def set_adapters_for_text_encoder(
    adapter_names: Union[List[str], str],
    text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
    text_encoder_weights: Optional[Union[float, List[float], List[None]]] = None,
):
    """
    设置文本编码器的适配器层。

    参数：
        adapter_names (`List[str]` 或 `str`):
            要使用的适配器名称。
        text_encoder (`torch.nn.Module`, *可选*):
            要设置适配器层的文本编码器模块。如果为 `None`，将尝试获取 `text_encoder` 属性。
        text_encoder_weights (`List[float]`, *可选*):
            要用于文本编码器的权重。如果为 `None`，则所有适配器的权重均设置为 `1.0`。
    """
    # 如果文本编码器为 None，抛出错误
    if text_encoder is None:
        raise ValueError(
            "管道没有默认的 `pipe.text_encoder` 类。请确保传递一个 `text_encoder`。"
        )

    # 处理适配器权重的函数
    def process_weights(adapter_names, weights):
        # 将权重扩展为列表，确保每个适配器都有一个权重
        # 例如，对于 2 个适配器:  7 -> [7,7] ; [3, None] -> [3, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        # 检查适配器名称与权重列表的长度是否相等
        if len(adapter_names) != len(weights):
            raise ValueError(
                f"适配器名称的长度 {len(adapter_names)} 不等于权重的长度 {len(weights)}"
            )

        # 将 None 值设置为默认值 1.0
        # 例如: [7,7] -> [7,7] ; [3, None] -> [3,1]
        weights = [w if w is not None else 1.0 for w in weights]

        return weights

    # 如果适配器名称是字符串，则将其转为列表
    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
    # 处理适配器权重
    text_encoder_weights = process_weights(adapter_names, text_encoder_weights)
    # 设置权重并激活适配器
    set_weights_and_activate_adapters(text_encoder, adapter_names, text_encoder_weights)


# 禁用文本编码器的 LoRA 层
def disable_lora_for_text_encoder(text_encoder: Optional["PreTrainedModel"] = None):
    """
    禁用文本编码器的 LoRA 层。

    参数：
        text_encoder (`torch.nn.Module`, *可选*):
            要禁用 LoRA 层的文本编码器模块。如果为 `None`，将尝试获取 `text_encoder` 属性。
    """
    # 如果文本编码器为 None，抛出错误
    if text_encoder is None:
        raise ValueError("未找到文本编码器。")
    # 设置适配器层为禁用状态
    set_adapter_layers(text_encoder, enabled=False)


# 启用文本编码器的 LoRA 层
def enable_lora_for_text_encoder(text_encoder: Optional["PreTrainedModel"] = None):
    """
    启用文本编码器的 LoRA 层。
    # 函数参数文档字符串，说明 text_encoder 的作用和类型
        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                # 可选参数，文本编码器模块，用于启用 LoRA 层。如果为 `None`，将尝试获取 `text_encoder`
                attribute.
        """
        # 如果未提供文本编码器，则抛出错误
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        # 调用函数以启用适配器层
        set_adapter_layers(text_encoder, enabled=True)
# 移除文本编码器的猴子补丁
def _remove_text_encoder_monkey_patch(text_encoder):
    # 递归移除 PEFT 层
    recurse_remove_peft_layers(text_encoder)
    # 如果 text_encoder 有 peft_config 属性且不为 None
    if getattr(text_encoder, "peft_config", None) is not None:
        # 删除 peft_config 属性
        del text_encoder.peft_config
        # 将 hf_peft_config_loaded 设置为 None
        text_encoder._hf_peft_config_loaded = None


class LoraBaseMixin:
    """处理 LoRA 的实用类。"""

    # 可加载的 LoRA 模块列表
    _lora_loadable_modules = []
    # 融合 LoRA 的数量
    num_fused_loras = 0

    # 加载 LoRA 权重的未实现方法
    def load_lora_weights(self, **kwargs):
        raise NotImplementedError("`load_lora_weights()` is not implemented.")

    # 保存 LoRA 权重的未实现方法
    @classmethod
    def save_lora_weights(cls, **kwargs):
        raise NotImplementedError("`save_lora_weights()` not implemented.")

    # 获取 LoRA 状态字典的未实现方法
    @classmethod
    def lora_state_dict(cls, **kwargs):
        raise NotImplementedError("`lora_state_dict()` is not implemented.")

    # 可选地禁用管道的离线加载
    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        """
        可选地移除已离线加载到 CPU 的管道。

        Args:
            _pipeline (`DiffusionPipeline`):
                要禁用离线加载的管道。

        Returns:
            tuple:
                指示 `is_model_cpu_offload` 或 `is_sequential_cpu_offload` 是否为 True 的元组。
        """
        # 初始化模型和序列 CPU 离线标志
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        # 如果管道不为 None 且没有 hf_device_map
        if _pipeline is not None and _pipeline.hf_device_map is None:
            # 遍历管道的组件
            for _, component in _pipeline.components.items():
                # 如果组件是 nn.Module 且有 _hf_hook 属性
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    # 判断模型是否已经离线加载到 CPU
                    if not is_model_cpu_offload:
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    # 判断是否序列化离线加载
                    if not is_sequential_cpu_offload:
                        is_sequential_cpu_offload = (
                            isinstance(component._hf_hook, AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                    # 记录检测到的加速钩子信息
                    logger.info(
                        "检测到加速钩子。由于您已调用 `load_lora_weights()`，之前的钩子将首先被移除。然后将加载 LoRA 参数并再次应用钩子。"
                    )
                    # 从模块中移除钩子
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        # 返回模型和序列 CPU 离线状态
        return (is_model_cpu_offload, is_sequential_cpu_offload)

    # 获取状态字典的方法，参数尚未完全列出
    @classmethod
    def _fetch_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict,
        weight_name,
        use_safetensors,
        local_files_only,
        cache_dir,
        force_download,
        proxies,
        token,
        revision,
        subfolder,
        user_agent,
        allow_pickle,
    ):
        # 从当前模块导入 LORA_WEIGHT_NAME 和 LORA_WEIGHT_NAME_SAFE
        from .lora_pipeline import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE

        # 初始化模型文件为 None
        model_file = None
        # 检查传入的模型参数是否为字典
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # 如果使用 safetensors 且权重名称为空，或者权重名称以 .safetensors 结尾
            # 则尝试加载 .safetensors 权重
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    # 放宽加载检查以提高推理 API 的友好性
                    # 有时无法自动确定 `weight_name`
                    if weight_name is None:
                        # 获取最佳猜测的权重名称
                        weight_name = cls._best_guess_weight_name(
                            pretrained_model_name_or_path_or_dict,
                            file_extension=".safetensors",  # 指定文件扩展名为 .safetensors
                            local_files_only=local_files_only,  # 仅限本地文件
                        )
                    # 获取模型文件
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,  # 使用安全的权重名称
                        cache_dir=cache_dir,  # 指定缓存目录
                        force_download=force_download,  # 是否强制下载
                        proxies=proxies,  # 代理设置
                        local_files_only=local_files_only,  # 仅限本地文件
                        token=token,  # 认证令牌
                        revision=revision,  # 版本信息
                        subfolder=subfolder,  # 子文件夹
                        user_agent=user_agent,  # 用户代理信息
                    )
                    # 从模型文件加载状态字典到 CPU 设备
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except (IOError, safetensors.SafetensorError) as e:
                    # 如果不允许使用 pickle，则抛出异常
                    if not allow_pickle:
                        raise e
                    # 尝试加载非 safetensors 权重
                    model_file = None
                    pass  # 忽略异常并继续执行

            # 如果模型文件仍然为 None
            if model_file is None:
                # 如果权重名称为空，获取最佳猜测的权重名称
                if weight_name is None:
                    weight_name = cls._best_guess_weight_name(
                        pretrained_model_name_or_path_or_dict,  # 使用给定的参数
                        file_extension=".bin",  # 指定文件扩展名为 .bin
                        local_files_only=local_files_only  # 仅限本地文件
                    )
                # 获取模型文件
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,  # 使用常规权重名称
                    cache_dir=cache_dir,  # 指定缓存目录
                    force_download=force_download,  # 是否强制下载
                    proxies=proxies,  # 代理设置
                    local_files_only=local_files_only,  # 仅限本地文件
                    token=token,  # 认证令牌
                    revision=revision,  # 版本信息
                    subfolder=subfolder,  # 子文件夹
                    user_agent=user_agent,  # 用户代理信息
                )
                # 从模型文件加载状态字典
                state_dict = load_state_dict(model_file)
        else:
            # 如果传入的是字典，则直接将其赋值给状态字典
            state_dict = pretrained_model_name_or_path_or_dict

        # 返回加载的状态字典
        return state_dict

    # 定义类方法的装饰器
    @classmethod
    # 获取最佳权重名称的方法，支持多种输入形式
        def _best_guess_weight_name(
            # 类参数，预训练模型名称或路径或字典，文件扩展名，是否仅使用本地文件
            cls, pretrained_model_name_or_path_or_dict, file_extension=".safetensors", local_files_only=False
        ):
            # 从lora_pipeline模块导入权重名称常量
            from .lora_pipeline import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE
    
            # 如果是本地文件模式或离线模式，抛出错误
            if local_files_only or HF_HUB_OFFLINE:
                raise ValueError("When using the offline mode, you must specify a `weight_name`.")
    
            # 初始化目标文件列表
            targeted_files = []
    
            # 如果输入是文件，直接返回
            if os.path.isfile(pretrained_model_name_or_path_or_dict):
                return
            # 如果输入是目录，列出符合扩展名的文件
            elif os.path.isdir(pretrained_model_name_or_path_or_dict):
                targeted_files = [
                    f for f in os.listdir(pretrained_model_name_or_path_or_dict) if f.endswith(file_extension)
                ]
            # 否则从模型信息中获取文件列表
            else:
                files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
                targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]
            # 如果没有找到目标文件，直接返回
            if len(targeted_files) == 0:
                return
    
            # 定义不允许的子字符串
            unallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
            # 过滤掉包含不允许子字符串的文件
            targeted_files = list(
                filter(lambda x: all(substring not in x for substring in unallowed_substrings), targeted_files)
            )
    
            # 如果找到以LORA_WEIGHT_NAME结尾的文件，仅保留这些文件
            if any(f.endswith(LORA_WEIGHT_NAME) for f in targeted_files):
                targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME), targeted_files))
            # 否则如果找到以LORA_WEIGHT_NAME_SAFE结尾的文件，保留这些
            elif any(f.endswith(LORA_WEIGHT_NAME_SAFE) for f in targeted_files):
                targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME_SAFE), targeted_files))
    
            # 如果找到多个目标文件，抛出错误
            if len(targeted_files) > 1:
                raise ValueError(
                    f"Provided path contains more than one weights file in the {file_extension} format. Either specify `weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  {pretrained_model_name_or_path_or_dict}."
                )
            # 选择第一个目标文件作为权重名称
            weight_name = targeted_files[0]
            # 返回权重名称
            return weight_name
    
        # 卸载LoRA权重的方法
        def unload_lora_weights(self):
            """
            卸载LoRA参数的方法。
    
            示例：
    
            ```python
            >>> # 假设`pipeline`已经加载了LoRA参数。
            >>> pipeline.unload_lora_weights()
            >>> ...
            ```py
            """
            # 如果未使用PEFT后端，抛出错误
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for this method.")
    
            # 遍历可加载LoRA模块
            for component in self._lora_loadable_modules:
                # 获取相应的模型
                model = getattr(self, component, None)
                # 如果模型存在
                if model is not None:
                    # 如果模型是ModelMixin的子类，卸载LoRA
                    if issubclass(model.__class__, ModelMixin):
                        model.unload_lora()
                    # 如果模型是PreTrainedModel的子类，移除文本编码器的猴子补丁
                    elif issubclass(model.__class__, PreTrainedModel):
                        _remove_text_encoder_monkey_patch(model)
    # 定义一个方法，融合 LoRA 参数
        def fuse_lora(
            self,  # 方法所属的类实例
            components: List[str] = [],  # 要融合的组件列表，默认为空列表
            lora_scale: float = 1.0,  # LoRA 的缩放因子，默认为1.0
            safe_fusing: bool = False,  # 是否安全融合的标志，默认为 False
            adapter_names: Optional[List[str]] = None,  # 可选的适配器名称列表
            **kwargs,  # 额外的关键字参数
        ):
            # 定义一个方法，反融合 LoRA 参数
            def unfuse_lora(self, components: List[str] = [], **kwargs):
                r"""  # 文档字符串，描述该方法的作用
                Reverses the effect of  # 反转 fuse_lora 方法的效果
                [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).
    
                <Tip warning={true}>  # 提示框，表示这是一个实验性 API
                This is an experimental API.  # 说明该 API 是实验性质的
                </Tip>
    
                Args:  # 参数说明
                    components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.  # 要反融合 LoRA 的组件列表
                    unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.  # 是否反融合 UNet 的 LoRA 参数
                    unfuse_text_encoder (`bool`, defaults to `True`):  # 是否反融合文本编码器的 LoRA 参数
                        Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the  # 如果文本编码器没有打补丁，则无效
                        LoRA parameters then it won't have any effect.  # 如果没有效果，则不会反融合
                """
                # 检查关键字参数中是否包含 unfuse_unet
                if "unfuse_unet" in kwargs:
                    depr_message = "Passing `unfuse_unet` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_unet` will be removed in a future version."  # 过时的警告消息
                    deprecate(  # 调用 deprecate 函数
                        "unfuse_unet",  # 被弃用的参数名
                        "1.0.0",  # 被弃用的版本
                        depr_message,  # 过时消息
                    )
                # 检查关键字参数中是否包含 unfuse_transformer
                if "unfuse_transformer" in kwargs:
                    depr_message = "Passing `unfuse_transformer` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_transformer` will be removed in a future version."  # 过时的警告消息
                    deprecate(  # 调用 deprecate 函数
                        "unfuse_transformer",  # 被弃用的参数名
                        "1.0.0",  # 被弃用的版本
                        depr_message,  # 过时消息
                    )
                # 检查关键字参数中是否包含 unfuse_text_encoder
                if "unfuse_text_encoder" in kwargs:
                    depr_message = "Passing `unfuse_text_encoder` to `unfuse_lora()` is deprecated and will be ignored. Please use the `components` argument. `unfuse_text_encoder` will be removed in a future version."  # 过时的警告消息
                    deprecate(  # 调用 deprecate 函数
                        "unfuse_text_encoder",  # 被弃用的参数名
                        "1.0.0",  # 被弃用的版本
                        depr_message,  # 过时消息
                    )
    
                # 如果组件列表为空，则抛出异常
                if len(components) == 0:
                    raise ValueError("`components` cannot be an empty list.")  # 抛出 ValueError，说明组件列表不能为空
    
                # 遍历组件列表中的每个组件
                for fuse_component in components:
                    # 如果组件不在可加载的 LoRA 模块中，抛出异常
                    if fuse_component not in self._lora_loadable_modules:
                        raise ValueError(f"{fuse_component} is not found in {self._lora_loadable_modules=}.")  # 抛出 ValueError，说明组件未找到
    
                    # 获取当前组件的模型
                    model = getattr(self, fuse_component, None)  # 从当前实例获取组件的模型
                    # 如果模型存在
                    if model is not None:
                        # 检查模型是否是 ModelMixin 或 PreTrainedModel 的子类
                        if issubclass(model.__class__, (ModelMixin, PreTrainedModel)):
                            # 遍历模型中的每个模块
                            for module in model.modules():
                                # 如果模块是 BaseTunerLayer 的实例
                                if isinstance(module, BaseTunerLayer):
                                    module.unmerge()  # 调用 unmerge 方法，反融合 LoRA 参数
    
                # 将融合的 LoRA 数量减少1
                self.num_fused_loras -= 1  # 更新已融合 LoRA 的数量
    
        # 定义一个方法，设置适配器
        def set_adapters(
            self,  # 方法所属的类实例
            adapter_names: Union[List[str], str],  # 适配器名称，可以是列表或字符串
            adapter_weights: Optional[Union[float, Dict, List[float], List[Dict]]] = None,  # 可选的适配器权重
    ):
        # 将 adapter_names 转换为列表，如果它是字符串则单独包装成列表
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # 深拷贝 adapter_weights，防止修改原始数据
        adapter_weights = copy.deepcopy(adapter_weights)

        # 如果 adapter_weights 不是列表，则将其扩展为与 adapter_names 相同长度的列表
        if not isinstance(adapter_weights, list):
            adapter_weights = [adapter_weights] * len(adapter_names)

        # 检查 adapter_names 和 adapter_weights 的长度是否一致，若不一致则抛出错误
        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}"
            )

        # 获取所有适配器的列表，返回一个字典，示例：{"unet": ["adapter1", "adapter2"], "text_encoder": ["adapter2"]}
        list_adapters = self.get_list_adapters()  # eg {"unet": ["adapter1", "adapter2"], "text_encoder": ["adapter2"]}
        
        # 获取所有适配器的集合，例如：{"adapter1", "adapter2"}
        all_adapters = {
            adapter for adapters in list_adapters.values() for adapter in adapters
        }  # eg ["adapter1", "adapter2"]
        
        # 生成一个字典，键为适配器，值为适配器所对应的部分
        invert_list_adapters = {
            adapter: [part for part, adapters in list_adapters.items() if adapter in adapters]
            for adapter in all_adapters
        }  # eg {"adapter1": ["unet"], "adapter2": ["unet", "text_encoder"]}

        # 初始化一个空字典，用于存放分解后的权重
        _component_adapter_weights = {}
        
        # 遍历可加载的模块
        for component in self._lora_loadable_modules:
            # 动态获取模块的实例
            model = getattr(self, component)

            # 将适配器名称与权重一一对应
            for adapter_name, weights in zip(adapter_names, adapter_weights):
                # 如果权重是字典，尝试从中获取特定组件的权重
                if isinstance(weights, dict):
                    component_adapter_weights = weights.pop(component, None)

                    # 如果权重存在但模型中没有该组件，记录警告
                    if component_adapter_weights is not None and not hasattr(self, component):
                        logger.warning(
                            f"Lora weight dict contains {component} weights but will be ignored because pipeline does not have {component}."
                        )

                    # 如果权重存在但适配器中不包含该组件，记录警告
                    if component_adapter_weights is not None and component not in invert_list_adapters[adapter_name]:
                        logger.warning(
                            (
                                f"Lora weight dict for adapter '{adapter_name}' contains {component},"
                                f"but this will be ignored because {adapter_name} does not contain weights for {component}."
                                f"Valid parts for {adapter_name} are: {invert_list_adapters[adapter_name]}."
                            )
                        )

                else:
                    # 如果权重不是字典，直接使用权重
                    component_adapter_weights = weights

                # 确保组件权重字典中有该组件的列表，如果没有则初始化为空列表
                _component_adapter_weights.setdefault(component, [])
                # 将组件的权重添加到对应的列表中
                _component_adapter_weights[component].append(component_adapter_weights)

            # 如果模型是 ModelMixin 的子类，设置适配器
            if issubclass(model.__class__, ModelMixin):
                model.set_adapters(adapter_names, _component_adapter_weights[component])
            # 如果模型是 PreTrainedModel 的子类，设置文本编码器的适配器
            elif issubclass(model.__class__, PreTrainedModel):
                set_adapters_for_text_encoder(adapter_names, model, _component_adapter_weights[component])
    # 定义一个禁用 LoRA 的方法
        def disable_lora(self):
            # 检查是否使用 PEFT 后端，若不使用则抛出错误
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for this method.")
    
            # 遍历可加载 LoRA 的模块
            for component in self._lora_loadable_modules:
                # 获取当前组件对应的模型
                model = getattr(self, component, None)
                # 如果模型存在
                if model is not None:
                    # 如果模型是 ModelMixin 的子类，禁用其 LoRA
                    if issubclass(model.__class__, ModelMixin):
                        model.disable_lora()
                    # 如果模型是 PreTrainedModel 的子类，调用相应的禁用方法
                    elif issubclass(model.__class__, PreTrainedModel):
                        disable_lora_for_text_encoder(model)
    
    # 定义一个启用 LoRA 的方法
        def enable_lora(self):
            # 检查是否使用 PEFT 后端，若不使用则抛出错误
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for this method.")
    
            # 遍历可加载 LoRA 的模块
            for component in self._lora_loadable_modules:
                # 获取当前组件对应的模型
                model = getattr(self, component, None)
                # 如果模型存在
                if model is not None:
                    # 如果模型是 ModelMixin 的子类，启用其 LoRA
                    if issubclass(model.__class__, ModelMixin):
                        model.enable_lora()
                    # 如果模型是 PreTrainedModel 的子类，调用相应的启用方法
                    elif issubclass(model.__class__, PreTrainedModel):
                        enable_lora_for_text_encoder(model)
    
    # 定义一个删除适配器的函数
        def delete_adapters(self, adapter_names: Union[List[str], str]):
            """
            Args:
            Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).
                adapter_names (`Union[List[str], str]`):
                    The names of the adapter to delete. Can be a single string or a list of strings
            """
            # 检查是否使用 PEFT 后端，若不使用则抛出错误
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for this method.")
    
            # 如果 adapter_names 是字符串，则转换为列表
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
    
            # 遍历可加载 LoRA 的模块
            for component in self._lora_loadable_modules:
                # 获取当前组件对应的模型
                model = getattr(self, component, None)
                # 如果模型存在
                if model is not None:
                    # 如果模型是 ModelMixin 的子类，删除适配器
                    if issubclass(model.__class__, ModelMixin):
                        model.delete_adapters(adapter_names)
                    # 如果模型是 PreTrainedModel 的子类，逐个删除适配器层
                    elif issubclass(model.__class__, PreTrainedModel):
                        for adapter_name in adapter_names:
                            delete_adapter_layers(model, adapter_name)
    # 定义获取当前活动适配器的函数，返回类型为字符串列表
    def get_active_adapters(self) -> List[str]:
        # 函数说明：获取当前活动适配器的列表，包含使用示例
        """
        Gets the list of the current active adapters.
    
        Example:
    
        ```python
        from diffusers import DiffusionPipeline
    
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
        ).to("cuda")
        pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipeline.get_active_adapters()
        ```py
        """
        # 检查是否启用了 PEFT 后端，未启用则抛出异常
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )
    
        # 初始化活动适配器列表
        active_adapters = []
    
        # 遍历可加载的 LORA 模块
        for component in self._lora_loadable_modules:
            # 获取当前组件的模型，如果不存在则为 None
            model = getattr(self, component, None)
            # 检查模型是否存在且是 ModelMixin 的子类
            if model is not None and issubclass(model.__class__, ModelMixin):
                # 遍历模型的所有模块
                for module in model.modules():
                    # 检查模块是否为 BaseTunerLayer 的实例
                    if isinstance(module, BaseTunerLayer):
                        # 获取活动适配器并赋值
                        active_adapters = module.active_adapters
                        break
    
        # 返回活动适配器列表
        return active_adapters
    
    # 定义获取当前所有可用适配器列表的函数，返回类型为字典
    def get_list_adapters(self) -> Dict[str, List[str]]:
        # 函数说明：获取当前管道中所有可用适配器的列表
        """
        Gets the current list of all available adapters in the pipeline.
        """
        # 检查是否启用了 PEFT 后端，未启用则抛出异常
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )
    
        # 初始化适配器集合字典
        set_adapters = {}
    
        # 遍历可加载的 LORA 模块
        for component in self._lora_loadable_modules:
            # 获取当前组件的模型，如果不存在则为 None
            model = getattr(self, component, None)
            # 检查模型是否存在且是 ModelMixin 或 PreTrainedModel 的子类，并具有 peft_config 属性
            if (
                model is not None
                and issubclass(model.__class__, (ModelMixin, PreTrainedModel))
                and hasattr(model, "peft_config")
            ):
                # 将适配器配置的键列表存入字典
                set_adapters[component] = list(model.peft_config.keys())
    
        # 返回适配器集合字典
        return set_adapters
    # 定义一个方法，用于将指定的 LoRA 适配器移动到目标设备
    def set_lora_device(self, adapter_names: List[str], device: Union[torch.device, str, int]) -> None:
        """
        将 `adapter_names` 中列出的 LoRA 适配器移动到目标设备。此方法用于在加载多个适配器时将 LoRA 移动到 CPU，以释放一些 GPU 内存。

        Args:
            adapter_names (`List[str]`):
                要发送到设备的适配器列表。
            device (`Union[torch.device, str, int]`):
                适配器要发送到的设备，可以是 torch 设备、字符串或整数。
        """
        # 检查是否使用 PEFT 后端，如果没有则抛出错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # 遍历可加载 LoRA 模块的组件
        for component in self._lora_loadable_modules:
            # 获取当前组件的模型，如果没有则为 None
            model = getattr(self, component, None)
            # 如果模型存在，则继续处理
            if model is not None:
                # 遍历模型的所有模块
                for module in model.modules():
                    # 检查模块是否是 BaseTunerLayer 的实例
                    if isinstance(module, BaseTunerLayer):
                        # 遍历适配器名称列表
                        for adapter_name in adapter_names:
                            # 将 lora_A 适配器移动到指定设备
                            module.lora_A[adapter_name].to(device)
                            # 将 lora_B 适配器移动到指定设备
                            module.lora_B[adapter_name].to(device)
                            # 如果模块有 lora_magnitude_vector 属性并且不为 None
                            if hasattr(module, "lora_magnitude_vector") and module.lora_magnitude_vector is not None:
                                # 将 lora_magnitude_vector 中的适配器移动到指定设备，并重新赋值
                                module.lora_magnitude_vector[adapter_name] = module.lora_magnitude_vector[
                                    adapter_name
                                ].to(device)

    # 定义一个静态方法，用于打包层的权重
    @staticmethod
    def pack_weights(layers, prefix):
        # 获取层的状态字典，如果层是 nn.Module 则调用其 state_dict() 方法
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
        # 将权重和模块名称组合成一个新的字典
        layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
        # 返回新的权重字典
        return layers_state_dict

    # 定义一个静态方法，用于写入 LoRA 层的权重
    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, torch.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
    ):
        # 从本地模块导入 LORA_WEIGHT_NAME 和 LORA_WEIGHT_NAME_SAFE 常量
        from .lora_pipeline import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE

        # 检查提供的路径是否为文件，如果是则记录错误并返回
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # 如果没有提供保存函数
        if save_function is None:
            # 根据是否使用安全序列化来定义保存函数
            if safe_serialization:
                # 定义一个保存函数，使用 safetensors 库保存文件，带有元数据
                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})
            else:
                # 如果不使用安全序列化，使用 PyTorch 自带的保存函数
                save_function = torch.save

        # 创建保存目录，如果目录已存在则不报错
        os.makedirs(save_directory, exist_ok=True)

        # 如果没有提供权重名称，根据安全序列化的设置选择默认名称
        if weight_name is None:
            if safe_serialization:
                # 使用安全权重名称
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                # 使用普通权重名称
                weight_name = LORA_WEIGHT_NAME

        # 构造保存文件的完整路径
        save_path = Path(save_directory, weight_name).as_posix()
        # 调用保存函数，将状态字典保存到指定路径
        save_function(state_dict, save_path)
        # 记录模型权重保存成功的信息
        logger.info(f"Model weights saved in {save_path}")

    @property
    # 定义属性函数，返回 lora_scale 的值，可以在运行时由管道设置
    def lora_scale(self) -> float:
        # 如果 _lora_scale 未被设置，返回默认值 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0
```