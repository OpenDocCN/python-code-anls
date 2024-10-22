# `.\diffusers\pipelines\pipeline_utils.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 版权声明，表明文件归 HuggingFace Inc. 团队所有
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.  # 版权声明，表明文件归 NVIDIA CORPORATION 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 指明此文件的许可证为 Apache 2.0 版本
# you may not use this file except in compliance with the License.  # 指出必须遵循许可证才能使用此文件
# You may obtain a copy of the License at  # 提供获取许可证的方式
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 指向许可证的 URL
#
# Unless required by applicable law or agreed to in writing, software  # 指出软件在没有明确同意或适用法律时按“原样”提供
# distributed under the License is distributed on an "AS IS" BASIS,  # 声明不提供任何形式的保证
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 没有任何形式的明示或暗示的担保
# See the License for the specific language governing permissions and  # 建议查看许可证以获取权限和限制的具体信息
# limitations under the License.  # 指出许可证下的限制
import fnmatch  # 导入 fnmatch 模块，用于文件名匹配
import importlib  # 导入 importlib 模块，用于动态导入模块
import inspect  # 导入 inspect 模块，用于获取对象的信息
import os  # 导入 os 模块，用于操作系统相关功能
import re  # 导入 re 模块，用于正则表达式匹配
import sys  # 导入 sys 模块，用于访问 Python 解释器的变量和函数
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器，用于简化类的定义
from pathlib import Path  # 从 pathlib 导入 Path 类，用于路径操作
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin  # 导入类型提示相关的工具

import numpy as np  # 导入 NumPy 库并简写为 np，用于数值计算
import PIL.Image  # 导入 PIL 的 Image 模块，用于图像处理
import requests  # 导入 requests 库，用于发送 HTTP 请求
import torch  # 导入 PyTorch 库，用于深度学习
from huggingface_hub import (  # 从 huggingface_hub 导入多个功能
    ModelCard,  # 导入 ModelCard 类，用于处理模型卡
    create_repo,  # 导入 create_repo 函数，用于创建模型仓库
    hf_hub_download,  # 导入 hf_hub_download 函数，用于从 Hugging Face Hub 下载文件
    model_info,  # 导入 model_info 函数，用于获取模型信息
    snapshot_download,  # 导入 snapshot_download 函数，用于下载快照
)
from huggingface_hub.utils import OfflineModeIsEnabled, validate_hf_hub_args  # 导入帮助函数用于验证参数和检查离线模式
from packaging import version  # 从 packaging 导入 version 模块，用于版本比较
from requests.exceptions import HTTPError  # 从 requests.exceptions 导入 HTTPError，用于处理 HTTP 错误
from tqdm.auto import tqdm  # 从 tqdm.auto 导入 tqdm，用于显示进度条

from .. import __version__  # 从当前模块导入版本号
from ..configuration_utils import ConfigMixin  # 从上级模块导入 ConfigMixin 类，用于配置混入
from ..models import AutoencoderKL  # 从上级模块导入 AutoencoderKL 模型
from ..models.attention_processor import FusedAttnProcessor2_0  # 从上级模块导入 FusedAttnProcessor2_0 类
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin  # 从上级模块导入常量和 ModelMixin 类
from ..schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME  # 从上级模块导入调度器配置名称
from ..utils import (  # 从上级模块导入多个工具函数和常量
    CONFIG_NAME,  # 配置文件名
    DEPRECATED_REVISION_ARGS,  # 已弃用的修订参数
    BaseOutput,  # 基础输出类
    PushToHubMixin,  # 推送到 Hub 的混入类
    deprecate,  # 用于标记弃用的函数
    is_accelerate_available,  # 检查 accelerate 库是否可用的函数
    is_accelerate_version,  # 检查 accelerate 版本的函数
    is_torch_npu_available,  # 检查 PyTorch NPU 是否可用的函数
    is_torch_version,  # 检查 PyTorch 版本的函数
    logging,  # 日志记录模块
    numpy_to_pil,  # NumPy 数组转换为 PIL 图像的函数
)
from ..utils.hub_utils import load_or_create_model_card, populate_model_card  # 从上级模块导入处理模型卡的函数
from ..utils.torch_utils import is_compiled_module  # 从上级模块导入检查模块是否已编译的函数


if is_torch_npu_available():  # 如果 PyTorch NPU 可用
    import torch_npu  # 导入 torch_npu 模块，提供对 NPU 的支持 # noqa: F401  # noqa: F401 表示忽略未使用的导入警告


from .pipeline_loading_utils import (  # 从当前包导入多个加载管道相关的工具
    ALL_IMPORTABLE_CLASSES,  # 所有可导入的类
    CONNECTED_PIPES_KEYS,  # 连接管道的键
    CUSTOM_PIPELINE_FILE_NAME,  # 自定义管道文件名
    LOADABLE_CLASSES,  # 可加载的类
    _fetch_class_library_tuple,  # 获取类库元组的私有函数
    _get_custom_pipeline_class,  # 获取自定义管道类的私有函数
    _get_final_device_map,  # 获取最终设备映射的私有函数
    _get_pipeline_class,  # 获取管道类的私有函数
    _unwrap_model,  # 解包模型的私有函数
    is_safetensors_compatible,  # 检查是否兼容 SafeTensors 的函数
    load_sub_model,  # 加载子模型的函数
    maybe_raise_or_warn,  # 可能抛出警告或错误的函数
    variant_compatible_siblings,  # 检查变体兼容的兄弟类的函数
    warn_deprecated_model_variant,  # 发出关于模型变体弃用的警告的函数
)


if is_accelerate_available():  # 如果 accelerate 库可用
    import accelerate  # 导入 accelerate 库，提供加速功能


LIBRARIES = []  # 初始化空列表，用于存储库
for library in LOADABLE_CLASSES:  # 遍历可加载的类
    LIBRARIES.append(library)  # 将每个库添加到 LIBRARIES 列表中

SUPPORTED_DEVICE_MAP = ["balanced"]  # 定义支持的设备映射，使用平衡策略

logger = logging.get_logger(__name__)  # 创建一个与当前模块同名的日志记录器


@dataclass  # 使用 dataclass 装饰器定义一个数据类
class ImagePipelineOutput(BaseOutput):  # 定义图像管道输出类，继承自 BaseOutput
    """
    Output class for image pipelines.  # 图像管道的输出类

    Args:  # 参数说明
        images (`List[PIL.Image.Image]` or `np.ndarray`)  # images 参数，接受 PIL 图像列表或 NumPy 数组
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.  # 说明该参数可以是图像列表或具有特定形状的 NumPy 数组
    """
    # 定义一个变量 images，它可以是一个 PIL 图像对象列表或一个 NumPy 数组
    images: Union[List[PIL.Image.Image], np.ndarray]
# 定义音频管道输出的数据类，继承自 BaseOutput
@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    音频管道的输出类。

    参数:
        audios (`np.ndarray`)
            一个形状为 `(batch_size, num_channels, sample_rate)` 的 NumPy 数组，表示去噪后的音频样本列表。
    """

    # 存储音频样本的 NumPy 数组
    audios: np.ndarray


# 定义扩散管道的基类，继承自 ConfigMixin 和 PushToHubMixin
class DiffusionPipeline(ConfigMixin, PushToHubMixin):
    r"""
    所有管道的基类。

    [`DiffusionPipeline`] 存储所有扩散管道的组件（模型、调度器和处理器），并提供加载、下载和保存模型的方法。它还包含以下方法：

        - 将所有 PyTorch 模块移动到您选择的设备
        - 启用/禁用去噪迭代的进度条

    类属性：

        - **config_name** (`str`) -- 存储扩散管道所有组件类和模块名称的配置文件名。
        - **_optional_components** (`List[str]`) -- 所有可选组件的列表，这些组件在管道功能上并不是必需的（应由子类重写）。
    """

    # 配置文件名称，默认值为 "model_index.json"
    config_name = "model_index.json"
    # 模型 CPU 卸载序列，初始值为 None
    model_cpu_offload_seq = None
    # Hugging Face 设备映射，初始值为 None
    hf_device_map = None
    # 可选组件列表，初始化为空
    _optional_components = []
    # 不参与 CPU 卸载的组件列表，初始化为空
    _exclude_from_cpu_offload = []
    # 是否加载连接的管道，初始化为 False
    _load_connected_pipes = False
    # 是否为 ONNX 格式，初始化为 False
    _is_onnx = False

    # 注册模块的方法，接收任意关键字参数
    def register_modules(self, **kwargs):
        # 遍历关键字参数中的模块
        for name, module in kwargs.items():
            # 检索库
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                # 如果模块为 None，注册字典设置为 None
                register_dict = {name: (None, None)}
            else:
                # 获取库和类名的元组
                library, class_name = _fetch_class_library_tuple(module)
                # 注册字典设置为库和类名元组
                register_dict = {name: (library, class_name)}

            # 保存模型索引配置
            self.register_to_config(**register_dict)

            # 设置模型
            setattr(self, name, module)

    # 自定义属性设置方法
    def __setattr__(self, name: str, value: Any):
        # 检查属性是否在实例字典中且在配置中存在
        if name in self.__dict__ and hasattr(self.config, name):
            # 如果名称在配置中存在，则需要覆盖配置
            if isinstance(getattr(self.config, name), (tuple, list)):
                # 如果值不为 None 且配置中存在有效值
                if value is not None and self.config[name][0] is not None:
                    # 获取类库元组
                    class_library_tuple = _fetch_class_library_tuple(value)
                else:
                    # 否则设置为 None
                    class_library_tuple = (None, None)

                # 注册到配置中
                self.register_to_config(**{name: class_library_tuple})
            else:
                # 直接注册到配置中
                self.register_to_config(**{name: value})

        # 调用父类的设置属性方法
        super().__setattr__(name, value)

    # 保存预训练模型的方法
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    @property
    # 定义一个方法，返回当前使用的设备类型
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        # 获取当前实例的模块名和其他相关信息
        module_names, _ = self._get_signature_keys(self)
        # 根据模块名获取实例中对应的模块对象，若不存在则为 None
        modules = [getattr(self, n, None) for n in module_names]
        # 过滤出类型为 torch.nn.Module 的模块
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        # 遍历所有模块
        for module in modules:
            # 返回第一个模块的设备类型
            return module.device

        # 如果没有模块，默认返回 CPU 设备
        return torch.device("cpu")

    # 定义一个只读属性，返回当前使用的数据类型
    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        # 获取当前实例的模块名和其他相关信息
        module_names, _ = self._get_signature_keys(self)
        # 根据模块名获取实例中对应的模块对象，若不存在则为 None
        modules = [getattr(self, n, None) for n in module_names]
        # 过滤出类型为 torch.nn.Module 的模块
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        # 遍历所有模块
        for module in modules:
            # 返回第一个模块的数据类型
            return module.dtype

        # 如果没有模块，默认返回 float32 数据类型
        return torch.float32

    # 定义一个类方法，返回模型的名称或路径
    @classmethod
    @validate_hf_hub_args
    @property
    def name_or_path(self) -> str:
        # 从配置中获取名称或路径，若不存在则为 None
        return getattr(self.config, "_name_or_path", None)

    # 定义一个只读属性，返回执行设备
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        # 遍历组件字典中的每个模型
        for name, model in self.components.items():
            # 如果不是 nn.Module 或者在排除列表中，则跳过
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            # 如果模型没有 HF hook，返回当前设备
            if not hasattr(model, "_hf_hook"):
                return self.device
            # 遍历模型中的所有模块
            for module in model.modules():
                # 检查模块是否有执行设备信息
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    # 返回找到的执行设备
                    return torch.device(module._hf_hook.execution_device)
        # 如果没有找到，返回当前设备
        return self.device

    # 定义一个方法，用于移除所有注册的 hook
    def remove_all_hooks(self):
        r"""
        Removes all hooks that were added when using `enable_sequential_cpu_offload` or `enable_model_cpu_offload`.
        """
        # 遍历组件字典中的每个模型
        for _, model in self.components.items():
            # 如果是 nn.Module 且有 HF hook，则移除 hook
            if isinstance(model, torch.nn.Module) and hasattr(model, "_hf_hook"):
                accelerate.hooks.remove_hook_from_module(model, recurse=True)
        # 清空所有 hooks 列表
        self._all_hooks = []
    # 定义一个可能释放模型钩子的函数
        def maybe_free_model_hooks(self):
            r"""
            该函数卸载所有组件，移除通过 `enable_model_cpu_offload` 添加的模型钩子，然后再次应用它们。
            如果模型未被卸载，该函数无操作。确保将此函数添加到管道的 `__call__` 函数末尾，以便在应用 enable_model_cpu_offload 时正确工作。
            """
            # 检查是否没有钩子被添加，如果没有，什么都不做
            if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
                # `enable_model_cpu_offload` 尚未被调用，因此静默返回
                return
    
            # 确保模型的状态与调用之前一致
            self.enable_model_cpu_offload(device=getattr(self, "_offload_device", "cuda"))
    
    # 定义一个重置设备映射的函数
        def reset_device_map(self):
            r"""
            将设备映射（如果存在）重置为 None。
            """
            # 如果设备映射已经是 None，什么都不做
            if self.hf_device_map is None:
                return
            else:
                # 移除所有钩子
                self.remove_all_hooks()
                # 遍历组件，将每个 torch.nn.Module 移动到 CPU
                for name, component in self.components.items():
                    if isinstance(component, torch.nn.Module):
                        component.to("cpu")
                # 将设备映射设置为 None
                self.hf_device_map = None
    
    # 定义一个类方法以获取签名键
        @classmethod
        @validate_hf_hub_args
        @classmethod
        def _get_signature_keys(cls, obj):
            # 获取对象初始化方法的参数
            parameters = inspect.signature(obj.__init__).parameters
            # 获取所需参数（没有默认值的）
            required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
            # 获取可选参数（有默认值的）
            optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
            # 预期模块为所需参数的键集，排除 "self"
            expected_modules = set(required_parameters.keys()) - {"self"}
    
            # 将可选参数名转换为列表
            optional_names = list(optional_parameters)
            # 遍历可选参数，如果在可选组件中，则添加到预期模块并从可选参数中移除
            for name in optional_names:
                if name in cls._optional_components:
                    expected_modules.add(name)
                    optional_parameters.remove(name)
    
            # 返回预期模块和可选参数
            return expected_modules, optional_parameters
    
    # 定义一个类方法以获取签名类型
        @classmethod
        def _get_signature_types(cls):
            # 初始化一个字典以存储签名类型
            signature_types = {}
            # 遍历初始化方法的参数，获取每个参数的注解
            for k, v in inspect.signature(cls.__init__).parameters.items():
                # 如果参数注解是类，存储该注解
                if inspect.isclass(v.annotation):
                    signature_types[k] = (v.annotation,)
                # 如果参数注解是联合类型，获取所有类型
                elif get_origin(v.annotation) == Union:
                    signature_types[k] = get_args(v.annotation)
                # 如果无法获取类型注解，记录警告
                else:
                    logger.warning(f"cannot get type annotation for Parameter {k} of {cls}.")
            # 返回签名类型字典
            return signature_types
    
    # 定义一个属性
        @property
    # 定义一个方法，返回一个字典，包含初始化管道所需的所有模块
    def components(self) -> Dict[str, Any]:
        r"""  # 文档字符串，描述方法的功能和返回值
        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the modules needed to initialize the pipeline.

        Examples:

        ```py
        >>> from diffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```py
        """
        # 获取预期模块和可选参数的签名
        expected_modules, optional_parameters = self._get_signature_keys(self)
        # 构建一个字典，包含所有必要的组件
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        # 检查组件的键是否与预期模块匹配
        if set(components.keys()) != expected_modules:
            # 如果不匹配，抛出错误，说明初始化有误
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        # 返回构建的组件字典
        return components

    # 定义一个静态方法，将 NumPy 图像或图像批次转换为 PIL 图像
    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        # 调用外部函数进行转换
        return numpy_to_pil(images)

    # 定义一个方法，用于创建进度条
    def progress_bar(self, iterable=None, total=None):
        # 检查是否已定义进度条配置，如果没有则初始化为空字典
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        # 如果已经定义，则检查其类型是否为字典
        elif not isinstance(self._progress_bar_config, dict):
            # 如果类型不匹配，抛出错误
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        # 如果提供了可迭代对象，则返回一个带进度条的可迭代对象
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        # 如果提供了总数，则返回一个总数为 total 的进度条
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        # 如果两个都没有提供，抛出错误
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    # 定义一个方法，用于设置进度条的配置
    def set_progress_bar_config(self, **kwargs):
        # 将传入的参数存储到进度条配置中
        self._progress_bar_config = kwargs
    # 定义一个启用 xFormers 内存高效注意力的方法，支持可选的注意力操作
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            r"""
            启用来自 [xFormers](https://facebookresearch.github.io/xformers/) 的内存高效注意力。启用此选项后，
            你应该会观察到较低的 GPU 内存使用率，并在推理过程中可能加速。训练期间的加速不保证。
    
            <Tip warning={true}>
    
            ⚠️ 当同时启用内存高效注意力和切片注意力时，内存高效注意力优先。
    
            </Tip>
    
            参数:
                attention_op (`Callable`, *可选*):
                    用于覆盖默认的 `None` 操作符，以用作 xFormers 的
                    [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                    函数的 `op` 参数。
    
            示例:
    
            ```py
            >>> import torch
            >>> from diffusers import DiffusionPipeline
            >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
    
            >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
            >>> pipe = pipe.to("cuda")
            >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            >>> # 针对 Flash Attention 使用 VAE 时不接受注意力形状的解决方法
            >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            ```py
            """
            # 调用设置内存高效注意力的函数，并将标志设为 True 和传入的注意力操作
            self.set_use_memory_efficient_attention_xformers(True, attention_op)
    
    # 定义一个禁用 xFormers 内存高效注意力的方法
    def disable_xformers_memory_efficient_attention(self):
            r"""
            禁用来自 [xFormers](https://facebookresearch.github.io/xformers/) 的内存高效注意力。
            """
            # 调用设置内存高效注意力的函数，并将标志设为 False
            self.set_use_memory_efficient_attention_xformers(False)
    
    # 定义一个设置内存高效注意力的函数，接受有效标志和可选注意力操作
    def set_use_memory_efficient_attention_xformers(
            self, valid: bool, attention_op: Optional[Callable] = None
        ) -> None:
            # 递归遍历所有子模块
            # 任何暴露 set_use_memory_efficient_attention_xformers 方法的子模块将接收此消息
            def fn_recursive_set_mem_eff(module: torch.nn.Module):
                # 检查模块是否有设置内存高效注意力的方法，如果有则调用
                if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                    module.set_use_memory_efficient_attention_xformers(valid, attention_op)
    
                # 递归处理所有子模块
                for child in module.children():
                    fn_recursive_set_mem_eff(child)
    
            # 获取当前对象的模块名称及其签名
            module_names, _ = self._get_signature_keys(self)
            # 获取所有子模块，过滤出 torch.nn.Module 类型的模块
            modules = [getattr(self, n, None) for n in module_names]
            modules = [m for m in modules if isinstance(m, torch.nn.Module)]
    
            # 对每个模块调用递归设置函数
            for module in modules:
                fn_recursive_set_mem_eff(module)
    # 定义一个方法来启用切片注意力计算，默认为“auto”
        def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
            r"""
            启用切片注意力计算。当启用此选项时，注意力模块将输入张量分成多个切片
            以分步骤计算注意力。对于多个注意力头，计算将在每个头上顺序执行。
            这有助于节省一些内存，换取略微降低的速度。
    
            <Tip warning={true}>
    
            ⚠️ 如果您已经在使用来自 PyTorch 2.0 的 `scaled_dot_product_attention` (SDPA) 或 xFormers，
            请勿启用注意力切片。这些注意力计算已经非常节省内存，因此您不需要启用
            此功能。如果在 SDPA 或 xFormers 中启用注意力切片，可能会导致严重的性能下降！
    
            </Tip>
    
            参数:
                slice_size (`str` 或 `int`, *可选*, 默认为 `"auto"`):
                    当为 `"auto"` 时，将输入分为两个注意力头进行计算。
                    如果为 `"max"`，则通过一次只运行一个切片来保存最大内存。
                    如果提供一个数字，使用 `attention_head_dim // slice_size` 个切片。
                    在这种情况下，`attention_head_dim` 必须是 `slice_size` 的倍数。
    
            示例:
    
            ```py
            >>> import torch
            >>> from diffusers import StableDiffusionPipeline
    
            >>> pipe = StableDiffusionPipeline.from_pretrained(
            ...     "runwayml/stable-diffusion-v1-5",
            ...     torch_dtype=torch.float16,
            ...     use_safetensors=True,
            ... )
    
            >>> prompt = "a photo of an astronaut riding a horse on mars"
            >>> pipe.enable_attention_slicing()
            >>> image = pipe(prompt).images[0]
            ```
            """
            # 调用设置切片的方法，传入切片大小
            self.set_attention_slice(slice_size)
    
        # 定义一个方法来禁用切片注意力计算
        def disable_attention_slicing(self):
            r"""
            禁用切片注意力计算。如果之前调用过 `enable_attention_slicing`，则注意力
            将在一步中计算。
            """
            # 将切片大小设置为 `None` 以禁用 `attention slicing`
            self.enable_attention_slicing(None)
    
        # 定义一个方法来设置切片大小
        def set_attention_slice(self, slice_size: Optional[int]):
            # 获取当前类的签名键和模块名称
            module_names, _ = self._get_signature_keys(self)
            # 获取当前类的所有模块
            modules = [getattr(self, n, None) for n in module_names]
            # 过滤出具有 `set_attention_slice` 方法的 PyTorch 模块
            modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attention_slice")]
    
            # 遍历所有模块并设置切片大小
            for module in modules:
                module.set_attention_slice(slice_size)
    
        # 类方法的定义开始
        @classmethod
# 定义一个混合类，用于处理具有 VAE 和 UNet 的扩散管道（主要用于稳定扩散 LDM）
class StableDiffusionMixin:
    r""" 
    帮助 DiffusionPipeline 使用 VAE 和 UNet（主要用于 LDM，如稳定扩散）
    """

    # 启用切片 VAE 解码的功能
    def enable_vae_slicing(self):
        r"""
        启用切片 VAE 解码。当启用此选项时，VAE 将输入张量分割为切片
        以分几步计算解码。这对于节省内存和允许更大的批处理大小很有用。
        """
        # 调用 VAE 的方法以启用切片
        self.vae.enable_slicing()

    # 禁用切片 VAE 解码的功能
    def disable_vae_slicing(self):
        r"""
        禁用切片 VAE 解码。如果之前启用了 `enable_vae_slicing`，此方法将恢复到
        一步计算解码。
        """
        # 调用 VAE 的方法以禁用切片
        self.vae.disable_slicing()

    # 启用平铺 VAE 解码的功能
    def enable_vae_tiling(self):
        r"""
        启用平铺 VAE 解码。当启用此选项时，VAE 将输入张量分割为块
        以分几步计算解码和编码。这对于节省大量内存并允许处理更大图像很有用。
        """
        # 调用 VAE 的方法以启用平铺
        self.vae.enable_tiling()

    # 禁用平铺 VAE 解码的功能
    def disable_vae_tiling(self):
        r"""
        禁用平铺 VAE 解码。如果之前启用了 `enable_vae_tiling`，此方法将恢复到
        一步计算解码。
        """
        # 调用 VAE 的方法以禁用平铺
        self.vae.disable_tiling()

    # 启用 FreeU 机制，使用指定的缩放因子
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""启用 FreeU 机制，如 https://arxiv.org/abs/2309.11497 所述。

        缩放因子后缀表示应用它们的阶段。

        请参考 [官方库](https://github.com/ChenyangSi/FreeU) 以获取已知适用于不同管道（如
        稳定扩散 v1、v2 和稳定扩散 XL）组合的值。

        Args:
            s1 (`float`):
                第一阶段的缩放因子，用于减轻跳过特征的贡献，以缓解增强去噪过程中的
                “过平滑效应”。
            s2 (`float`):
                第二阶段的缩放因子，用于减轻跳过特征的贡献，以缓解增强去噪过程中的
                “过平滑效应”。
            b1 (`float`): 第一阶段的缩放因子，用于放大骨干特征的贡献。
            b2 (`float`): 第二阶段的缩放因子，用于放大骨干特征的贡献。
        """
        # 检查当前对象是否具有 `unet` 属性
        if not hasattr(self, "unet"):
            # 如果没有，则抛出值错误
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        # 调用 UNet 的方法以启用 FreeU，传递缩放因子
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # 禁用 FreeU 机制
    def disable_freeu(self):
        """禁用 FreeU 机制（如果已启用）。"""
        # 调用 UNet 的方法以禁用 FreeU
        self.unet.disable_freeu()
    # 定义融合 QKV 投影的方法，默认启用 UNet 和 VAE
        def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
            """
            启用融合 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）被融合。
            对于交叉注意力模块，键和值投影矩阵被融合。
    
            <Tip warning={true}>
    
            此 API 为 🧪 实验性。
    
            </Tip>
    
            参数:
                unet (`bool`, 默认值为 `True`): 是否在 UNet 上应用融合。
                vae (`bool`, 默认值为 `True`): 是否在 VAE 上应用融合。
            """
            # 初始化 UNet 和 VAE 的融合状态为 False
            self.fusing_unet = False
            self.fusing_vae = False
    
            # 如果启用 UNet 融合
            if unet:
                # 设置 UNet 融合状态为 True
                self.fusing_unet = True
                # 调用 UNet 的 QKV 融合方法
                self.unet.fuse_qkv_projections()
                # 设置 UNet 的注意力处理器为融合版本
                self.unet.set_attn_processor(FusedAttnProcessor2_0())
    
            # 如果启用 VAE 融合
            if vae:
                # 检查 VAE 是否为 AutoencoderKL 类型
                if not isinstance(self.vae, AutoencoderKL):
                    # 抛出异常提示不支持的 VAE 类型
                    raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")
    
                # 设置 VAE 融合状态为 True
                self.fusing_vae = True
                # 调用 VAE 的 QKV 融合方法
                self.vae.fuse_qkv_projections()
                # 设置 VAE 的注意力处理器为融合版本
                self.vae.set_attn_processor(FusedAttnProcessor2_0())
    
        # 定义取消 QKV 投影融合的方法，默认启用 UNet 和 VAE
        def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
            """如果启用了 QKV 投影融合，则禁用它。
    
            <Tip warning={true}>
    
            此 API 为 🧪 实验性。
    
            </Tip>
    
            参数:
                unet (`bool`, 默认值为 `True`): 是否在 UNet 上应用融合。
                vae (`bool`, 默认值为 `True`): 是否在 VAE 上应用融合。
    
            """
            # 如果启用 UNet 解融合
            if unet:
                # 检查 UNet 是否已经融合
                if not self.fusing_unet:
                    # 如果没有融合，记录警告信息
                    logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
                else:
                    # 调用 UNet 的解融合方法
                    self.unet.unfuse_qkv_projections()
                    # 设置 UNet 融合状态为 False
                    self.fusing_unet = False
    
            # 如果启用 VAE 解融合
            if vae:
                # 检查 VAE 是否已经融合
                if not self.fusing_vae:
                    # 如果没有融合，记录警告信息
                    logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
                else:
                    # 调用 VAE 的解融合方法
                    self.vae.unfuse_qkv_projections()
                    # 设置 VAE 融合状态为 False
                    self.fusing_vae = False
```