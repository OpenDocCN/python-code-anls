# `.\diffusers\loaders\lora_pipeline.py`

```
# 版权声明，表明该代码的版权所有者及其权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证的规定使用该文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵循许可证的情况下，才能使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，否则根据许可证分发的软件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证以获取有关权限和限制的具体条款
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入操作系统模块
import os
# 从 typing 模块导入类型提示相关的工具
from typing import Callable, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 huggingface_hub.utils 导入验证 Hugging Face Hub 参数的函数
from huggingface_hub.utils import validate_hf_hub_args

# 从 utils 模块中导入多个工具函数和常量
from ..utils import (
    USE_PEFT_BACKEND,  # 用于指示是否使用 PEFT 后端的常量
    convert_state_dict_to_diffusers,  # 转换状态字典到 Diffusers 格式的函数
    convert_state_dict_to_peft,  # 转换状态字典到 PEFT 格式的函数
    convert_unet_state_dict_to_peft,  # 将 UNet 状态字典转换为 PEFT 格式的函数
    deprecate,  # 用于标记过时函数的装饰器
    get_adapter_name,  # 获取适配器名称的函数
    get_peft_kwargs,  # 获取 PEFT 关键字参数的函数
    is_peft_version,  # 检查是否为 PEFT 版本的函数
    is_transformers_available,  # 检查 Transformers 库是否可用的函数
    logging,  # 日志记录工具
    scale_lora_layers,  # 调整 LoRA 层规模的函数
)
# 从 lora_base 模块导入 LoraBaseMixin 类
from .lora_base import LoraBaseMixin
# 从 lora_conversion_utils 模块导入两个用于转换的函数
from .lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers, _maybe_map_sgm_blocks_to_diffusers

# 如果 Transformers 库可用，则导入相关的模块
if is_transformers_available():
    from ..models.lora import text_encoder_attn_modules, text_encoder_mlp_modules

# 创建一个日志记录器，用于记录本模块的日志信息
logger = logging.get_logger(__name__)

# 定义一些常量，表示不同组件的名称
TEXT_ENCODER_NAME = "text_encoder"  # 文本编码器的名称
UNET_NAME = "unet"  # UNet 模型的名称
TRANSFORMER_NAME = "transformer"  # Transformer 模型的名称

# 定义 LoRA 权重文件的名称
LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"  # 二进制格式的权重文件名
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"  # 安全格式的权重文件名


# 定义一个类，用于加载 LoRA 层到稳定扩散模型中
class StableDiffusionLoraLoaderMixin(LoraBaseMixin):
    r"""
    将 LoRA 层加载到稳定扩散模型 [`UNet2DConditionModel`] 和
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 中。
    """

    # 可加载的 LoRA 模块列表
    _lora_loadable_modules = ["unet", "text_encoder"]
    unet_name = UNET_NAME  # UNet 模型的名称
    text_encoder_name = TEXT_ENCODER_NAME  # 文本编码器的名称

    # 定义加载 LoRA 权重的方法
    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
    ):
        """
        加载指定的 LoRA 权重到 `self.unet` 和 `self.text_encoder` 中。

        所有关键字参数将转发给 `self.lora_state_dict`。

        详情请参阅 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`]，了解如何加载状态字典。

        详情请参阅 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`]，了解如何将状态字典加载到 `self.unet` 中。

        详情请参阅 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder`]，了解如何将状态字典加载到 `self.text_encoder` 中。

        参数:
            pretrained_model_name_or_path_or_dict (`str` 或 `os.PathLike` 或 `dict`):
                详情请参阅 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`]。
            kwargs (`dict`, *可选*):
                详情请参阅 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`]。
            adapter_name (`str`, *可选*):
                用于引用加载的适配器模型的适配器名称。如果未指定，将使用 `default_{i}`，其中 i 是加载的适配器总数。
        """
        # 检查是否使用 PEFT 后端，如果未使用则引发错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # 如果传入的是字典，则复制一份而不是就地修改
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # 首先，确保检查点是兼容的，并且可以成功加载
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        # 检查状态字典中的所有键是否包含 "lora" 或 "dora_scale"
        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        # 如果格式不正确，则引发错误
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        # 将 LoRA 权重加载到 UNet 中
        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet,
            adapter_name=adapter_name,
            _pipeline=self,
        )
        # 将 LoRA 权重加载到文本编码器中
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=getattr(self, self.text_encoder_name)
            if not hasattr(self, "text_encoder")
            else self.text_encoder,
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    # 类方法，用于验证 HF Hub 参数
    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    @classmethod
    # 定义一个类方法，用于将 LoRA 层加载到 UNet 模型中
    def load_lora_into_unet(cls, state_dict, network_alphas, unet, adapter_name=None, _pipeline=None):
        """
        将 `state_dict` 中指定的 LoRA 层加载到 `unet` 中。
    
        参数：
            state_dict (`dict`):
                包含 LoRA 层参数的标准状态字典。键可以直接索引到 unet，或者以额外的 `unet` 前缀标识，以区分文本编码器的 LoRA 层。
            network_alphas (`Dict[str, float]`):
                用于稳定学习和防止下溢的网络 alpha 值。此值与 kohya-ss 训练脚本中的 `--network_alpha` 选项含义相同。参考[此链接](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning)。
            unet (`UNet2DConditionModel`):
                用于加载 LoRA 层的 UNet 模型。
            adapter_name (`str`, *可选*):
                用于引用加载的适配器模型的适配器名称。如果未指定，将使用 `default_{i}`，其中 i 是加载的适配器总数。
        """
        # 检查是否使用 PEFT 后端，如果未使用则引发错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
    
        # 检查序列化格式是否为新格式，`state_dict` 的键是否以 `cls.unet_name` 和/或 `cls.text_encoder_name` 为前缀
        keys = list(state_dict.keys())
        only_text_encoder = all(key.startswith(cls.text_encoder_name) for key in keys)
        if not only_text_encoder:
            # 加载与 UNet 对应的层
            logger.info(f"Loading {cls.unet_name}.")
            # 调用 UNet 的加载方法，传入状态字典和其他参数
            unet.load_attn_procs(
                state_dict, network_alphas=network_alphas, adapter_name=adapter_name, _pipeline=_pipeline
            )
    
    # 定义一个类方法，用于将 LoRA 层加载到文本编码器中
    @classmethod
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        adapter_name=None,
        _pipeline=None,
    ):
        # 方法定义，具体实现未提供
        pass
    
    # 定义一个类方法，用于保存 LoRA 权重
    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        # 方法定义，具体实现未提供
        pass
    ):
        r"""  # 文档字符串，描述函数的作用和参数
        Save the LoRA parameters corresponding to the UNet and text encoder.  # 保存与 UNet 和文本编码器相对应的 LoRA 参数

        Arguments:  # 参数说明
            save_directory (`str` or `os.PathLike`):  # 保存目录的类型说明
                Directory to save LoRA parameters to. Will be created if it doesn't exist.  # 保存 LoRA 参数的目录，如果不存在则创建
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):  # UNet 的 LoRA 层状态字典
                State dict of the LoRA layers corresponding to the `unet`.  # 与 `unet` 相对应的 LoRA 层的状态字典
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):  # 文本编码器的 LoRA 层状态字典
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text  # 与 `text_encoder` 相对应的 LoRA 层状态字典，必须显式传递
                encoder LoRA state dict because it comes from 🤗 Transformers.  # 因为它来自 🤗 Transformers
            is_main_process (`bool`, *optional*, defaults to `True`):  # 主要进程的布尔值，可选，默认值为 True
                Whether the process calling this is the main process or not. Useful during distributed training and you  # 调用此函数的进程是否为主进程，在分布式训练中很有用
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main  # 在这种情况下，只在主进程上设置 `is_main_process=True` 以避免竞争条件
                process to avoid race conditions.  # 避免竞争条件
            save_function (`Callable`):  # 保存函数的类型说明
                The function to use to save the state dictionary. Useful during distributed training when you need to  # 用于保存状态字典的函数，在分布式训练中很有用
                replace `torch.save` with another method. Can be configured with the environment variable  # 可以通过环境变量配置
                `DIFFUSERS_SAVE_MODE`.  # `DIFFUSERS_SAVE_MODE`
            safe_serialization (`bool`, *optional*, defaults to `True`):  # 安全序列化的布尔值，可选，默认值为 True
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.  # 是否使用 `safetensors` 或传统的 PyTorch 方法 `pickle` 保存模型
        """  # 文档字符串结束
        state_dict = {}  # 初始化一个空的状态字典

        if not (unet_lora_layers or text_encoder_lora_layers):  # 检查是否至少有一个 LoRA 层
            raise ValueError("You must pass at least one of `unet_lora_layers` and `text_encoder_lora_layers`.")  # 如果没有，抛出错误

        if unet_lora_layers:  # 如果存在 UNet 的 LoRA 层
            state_dict.update(cls.pack_weights(unet_lora_layers, cls.unet_name))  # 更新状态字典，打包 UNet 权重

        if text_encoder_lora_layers:  # 如果存在文本编码器的 LoRA 层
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, cls.text_encoder_name))  # 更新状态字典，打包文本编码器权重

        # Save the model  # 保存模型的注释
        cls.write_lora_layers(  # 调用类方法保存 LoRA 层
            state_dict=state_dict,  # 状态字典参数
            save_directory=save_directory,  # 保存目录参数
            is_main_process=is_main_process,  # 主要进程参数
            weight_name=weight_name,  # 权重名称参数
            save_function=save_function,  # 保存函数参数
            safe_serialization=safe_serialization,  # 安全序列化参数
        )  # 方法调用结束

    def fuse_lora(  # 定义 fuse_lora 方法
        self,  # 实例方法的 self 参数
        components: List[str] = ["unet", "text_encoder"],  # 组件列表，默认包含 UNet 和文本编码器
        lora_scale: float = 1.0,  # LoRA 缩放因子，默认值为 1.0
        safe_fusing: bool = False,  # 安全融合的布尔值，默认值为 False
        adapter_names: Optional[List[str]] = None,  # 适配器名称的可选列表，默认值为 None
        **kwargs,  # 接收额外的关键字参数
    ):
        r"""  # 开始文档字符串，描述该方法的功能和用法
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.  # 将 LoRA 参数融合到对应块的原始参数中

        <Tip warning={true}>  # 开始警告提示框
        This is an experimental API.  # 说明这是一个实验性 API
        </Tip>  # 结束警告提示框

        Args:  # 开始参数说明
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.  # 可注入 LoRA 的组件列表
            lora_scale (`float`, defaults to 1.0):  # LoRA 参数对输出影响的比例
                Controls how much to influence the outputs with the LoRA parameters.  # 控制 LoRA 参数对输出的影响程度
            safe_fusing (`bool`, defaults to `False`):  # 是否在融合前检查权重是否为 NaN
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.  # 如果值为 NaN 则不进行融合
            adapter_names (`List[str]`, *optional*):  # 可选的适配器名称
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.  # 如果未传入，默认融合所有活动适配器

        Example:  # 示例部分的开始
        ```py  # Python 代码块开始
        from diffusers import DiffusionPipeline  # 导入 DiffusionPipeline 模块
        import torch  # 导入 PyTorch 库

        pipeline = DiffusionPipeline.from_pretrained(  # 从预训练模型创建管道
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16  # 使用 float16 类型的模型
        ).to("cuda")  # 将管道移动到 GPU
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")  # 加载 LoRA 权重
        pipeline.fuse_lora(lora_scale=0.7)  # 融合 LoRA，影响比例为 0.7
        ```  # Python 代码块结束
        """
        super().fuse_lora(  # 调用父类的 fuse_lora 方法
            components=components, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names  # 将参数传递给父类方法
        )

    def unfuse_lora(self, components: List[str] = ["unet", "text_encoder"], **kwargs):  # 定义 unfuse_lora 方法，带有默认组件
        r"""  # 开始文档字符串，描述该方法的功能和用法
        Reverses the effect of  # 反转 fuse_lora 方法的效果
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).  # 提供 fuse_lora 的链接

        <Tip warning={true}>  # 开始警告提示框
        This is an experimental API.  # 说明这是一个实验性 API
        </Tip>  # 结束警告提示框

        Args:  # 开始参数说明
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.  # 可注入 LoRA 的组件列表，用于反融合
            unfuse_unet (`bool`, defaults to `True`):  # 是否反融合 UNet 的 LoRA 参数
                Whether to unfuse the UNet LoRA parameters.  # 反融合 UNet LoRA 参数的选项
            unfuse_text_encoder (`bool`, defaults to `True`):  # 是否反融合文本编码器的 LoRA 参数
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the  # 反融合文本编码器的 LoRA 参数的选项
                LoRA parameters then it won't have any effect.  # 如果文本编码器未被修改，则不会有任何效果
        """  # 结束文档字符串
        super().unfuse_lora(components=components)  # 调用父类的 unfuse_lora 方法，并传递组件参数
# 定义一个类，混合自 LoraBaseMixin，用于加载 LoRA 层到 Stable Diffusion XL
class StableDiffusionXLLoraLoaderMixin(LoraBaseMixin):
    r"""
    将 LoRA 层加载到 Stable Diffusion XL 的 [`UNet2DConditionModel`]、
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 和
    [`CLIPTextModelWithProjection`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection) 中。
    """

    # 定义可以加载 LoRA 的模块名列表
    _lora_loadable_modules = ["unet", "text_encoder", "text_encoder_2"]
    # 指定 UNET 的名称
    unet_name = UNET_NAME
    # 指定文本编码器的名称
    text_encoder_name = TEXT_ENCODER_NAME

    # 定义一个加载 LoRA 权重的方法
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
    @classmethod
    # 验证传入的 Hugging Face Hub 参数
    @validate_hf_hub_args
    # 定义一个类方法，获取 LoRA 状态字典
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
    @classmethod
    # 定义一个类方法，用于将 LoRA 加载到 UNET 中
    # 定义一个类方法，用于将 LoRA 层加载到 UNet 模型中
    def load_lora_into_unet(cls, state_dict, network_alphas, unet, adapter_name=None, _pipeline=None):
        # 文档字符串，描述方法的作用和参数
        """
        This will load the LoRA layers specified in `state_dict` into `unet`.
    
        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow. This value has the
                same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
                link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        # 检查是否启用 PEFT 后端，若未启用则抛出异常
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
    
        # 获取 state_dict 中的所有键
        keys = list(state_dict.keys())
        # 检查所有键是否都以 text_encoder_name 开头
        only_text_encoder = all(key.startswith(cls.text_encoder_name) for key in keys)
        # 如果不是仅有文本编码器
        if not only_text_encoder:
            # 记录正在加载的 UNet 名称
            logger.info(f"Loading {cls.unet_name}.")
            # 加载与 UNet 对应的层
            unet.load_attn_procs(
                state_dict, network_alphas=network_alphas, adapter_name=adapter_name, _pipeline=_pipeline
            )
    
        @classmethod
        # 从 diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder 复制的方法
        def load_lora_into_text_encoder(
            cls,
            state_dict,
            network_alphas,
            text_encoder,
            prefix=None,
            lora_scale=1.0,
            adapter_name=None,
            _pipeline=None,
        @classmethod
        # 定义一个类方法，用于保存 LoRA 权重
        def save_lora_weights(
            cls,
            save_directory: Union[str, os.PathLike],
            unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            is_main_process: bool = True,
            weight_name: str = None,
            save_function: Callable = None,
            safe_serialization: bool = True,
    ):
        r"""
        # 文档字符串，描述保存 UNet 和文本编码器对应的 LoRA 参数的功能

        Arguments:
            # 保存 LoRA 参数的目录，若不存在则创建
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            # UNet 对应的 LoRA 层的状态字典
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            # 文本编码器对应的 LoRA 层的状态字典，必须显式传入
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            # 第二个文本编码器对应的 LoRA 层的状态字典，必须显式传入
            text_encoder_2_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder_2`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            # 表示调用此函数的进程是否为主进程，主要用于分布式训练
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            # 保存状态字典的函数，分布式训练时可替换 `torch.save`
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            # 是否使用 safetensors 保存模型，默认为 True
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        # 初始化状态字典，用于存储 LoRA 参数
        state_dict = {}

        # 如果没有传入任何 LoRA 层，则抛出异常
        if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                # 报告至少需要传入一个 LoRA 层
                "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
            )

        # 如果有 UNet 的 LoRA 层，则打包并更新状态字典
        if unet_lora_layers:
            state_dict.update(cls.pack_weights(unet_lora_layers, "unet"))

        # 如果有文本编码器的 LoRA 层，则打包并更新状态字典
        if text_encoder_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, "text_encoder"))

        # 如果有第二个文本编码器的 LoRA 层，则打包并更新状态字典
        if text_encoder_2_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        # 写入 LoRA 层参数，调用保存函数
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )
    # 定义一个方法 fuse_lora，用于将 LoRA 参数融合到相应模块的原始参数中
    def fuse_lora(
        # 方法参数：可注入 LoRA 的组件列表，默认为 ["unet", "text_encoder", "text_encoder_2"]
        self,
        components: List[str] = ["unet", "text_encoder", "text_encoder_2"],
        # LoRA 权重影响输出的程度，默认为 1.0
        lora_scale: float = 1.0,
        # 是否在融合前检查权重是否为 NaN，默认为 False
        safe_fusing: bool = False,
        # 可选参数，指定用于融合的适配器名称
        adapter_names: Optional[List[str]] = None,
        # 允许传入额外的关键字参数
        **kwargs,
    ):
        r"""
        将 LoRA 参数融合到相应模块的原始参数中。

        <Tip warning={true}>

        这是一个实验性 API。

        </Tip>

        Args:
            components: (`List[str]`): 需要融合 LoRA 的组件列表。
            lora_scale (`float`, defaults to 1.0):
                控制 LoRA 参数对输出的影响程度。
            safe_fusing (`bool`, defaults to `False`):
                在融合前检查权重是否为 NaN 的开关。
            adapter_names (`List[str]`, *optional*):
                用于融合的适配器名称。如果未传入，则将融合所有活动适配器。

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        # 加载 LoRA 权重
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        # 融合 LoRA 参数，影响程度为 0.7
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        # 调用父类的 fuse_lora 方法，传入组件、LoRA 权重、检查 NaN 的选项和适配器名称
        super().fuse_lora(
            components=components, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
        )

    # 定义一个方法 unfuse_lora，用于逆转 LoRA 参数的融合效果
    def unfuse_lora(self, components: List[str] = ["unet", "text_encoder", "text_encoder_2"], **kwargs):
        r"""
        逆转
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora) 的效果。

        <Tip warning={true}>

        这是一个实验性 API。

        </Tip>

        Args:
            components (`List[str]`): 需要从中解融合 LoRA 的组件列表。
            unfuse_unet (`bool`, defaults to `True`): 是否解融合 UNet 的 LoRA 参数。
            unfuse_text_encoder (`bool`, defaults to `True`):
                是否解融合文本编码器的 LoRA 参数。如果文本编码器没有被 LoRA 参数修补，则不会有任何效果。
        """
        # 调用父类的 unfuse_lora 方法，传入组件和其他参数
        super().unfuse_lora(components=components)
# 定义一个混合类 SD3LoraLoaderMixin，继承自 LoraBaseMixin
class SD3LoraLoaderMixin(LoraBaseMixin):
    r"""
    加载 LoRA 层到 [`SD3Transformer2DModel`]、
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) 和
    [`CLIPTextModelWithProjection`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection)。

    特定于 [`StableDiffusion3Pipeline`]。
    """

    # 可加载 LoRA 的模块列表
    _lora_loadable_modules = ["transformer", "text_encoder", "text_encoder_2"]
    # 转换器名称，使用预定义的常量
    transformer_name = TRANSFORMER_NAME
    # 文本编码器名称，使用预定义的常量
    text_encoder_name = TEXT_ENCODER_NAME

    # 类方法，验证 Hugging Face Hub 参数
    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        # 加载 LoRA 权重的方法，接收模型名称或路径或字典
        def load_lora_weights(
            self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
        ):
            # 类方法，从 diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin 中复制的加载文本编码器的方法
            @classmethod
            def load_lora_into_text_encoder(
                cls,
                state_dict,
                network_alphas,
                text_encoder,
                prefix=None,
                lora_scale=1.0,
                adapter_name=None,
                _pipeline=None,
            ):
                # 类方法，保存 LoRA 权重到指定目录
                def save_lora_weights(
                    cls,
                    save_directory: Union[str, os.PathLike],
                    transformer_lora_layers: Dict[str, torch.nn.Module] = None,
                    text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
                    text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
                    is_main_process: bool = True,
                    weight_name: str = None,
                    save_function: Callable = None,
                    safe_serialization: bool = True,
                ):
    ):
        r"""
        保存与 UNet 和文本编码器对应的 LoRA 参数。

        参数：
            save_directory (`str` 或 `os.PathLike`):
                保存 LoRA 参数的目录。如果不存在，将创建该目录。
            transformer_lora_layers (`Dict[str, torch.nn.Module]` 或 `Dict[str, torch.Tensor]`):
                与 `transformer` 相关的 LoRA 层的状态字典。
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` 或 `Dict[str, torch.Tensor]`):
                与 `text_encoder` 相关的 LoRA 层的状态字典。必须显式传递文本编码器的 LoRA 状态字典，因为它来自 🤗 Transformers。
            text_encoder_2_lora_layers (`Dict[str, torch.nn.Module]` 或 `Dict[str, torch.Tensor]`):
                与 `text_encoder_2` 相关的 LoRA 层的状态字典。必须显式传递文本编码器的 LoRA 状态字典，因为它来自 🤗 Transformers。
            is_main_process (`bool`, *可选*, 默认值为 `True`):
                调用此函数的进程是否为主进程。在分布式训练期间非常有用，您需要在所有进程上调用此函数。在这种情况下，只有在主进程上设置 `is_main_process=True` 以避免竞争条件。
            save_function (`Callable`):
                用于保存状态字典的函数。在分布式训练时，当您需要将 `torch.save` 替换为其他方法时非常有用。可以通过环境变量 `DIFFUSERS_SAVE_MODE` 进行配置。
            safe_serialization (`bool`, *可选*, 默认值为 `True`):
                是否使用 `safetensors` 保存模型，或使用传统的 PyTorch 方法 `pickle`。
        """
        # 初始化一个空字典，用于存储状态字典
        state_dict = {}

        # 检查是否至少传递了一个 LoRA 层的状态字典，如果没有则引发错误
        if not (transformer_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                "必须至少传递一个 `transformer_lora_layers`、`text_encoder_lora_layers` 或 `text_encoder_2_lora_layers`。"
            )

        # 如果传递了 transformer_lora_layers，则将其打包并更新状态字典
        if transformer_lora_layers:
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

        # 如果传递了 text_encoder_lora_layers，则将其打包并更新状态字典
        if text_encoder_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, "text_encoder"))

        # 如果传递了 text_encoder_2_lora_layers，则将其打包并更新状态字典
        if text_encoder_2_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        # 保存模型
        cls.write_lora_layers(
            state_dict=state_dict,  # 要保存的状态字典
            save_directory=save_directory,  # 保存目录
            is_main_process=is_main_process,  # 主进程标志
            weight_name=weight_name,  # 权重名称
            save_function=save_function,  # 保存函数
            safe_serialization=safe_serialization,  # 安全序列化标志
        )
    # 定义一个方法，用于将 LoRA 参数融合到原始参数中
    def fuse_lora(
        # 方法的参数列表
        self,
        # 可选组件列表，默认包括 "transformer"、"text_encoder" 和 "text_encoder_2"
        components: List[str] = ["transformer", "text_encoder", "text_encoder_2"],
        # LoRA 参数影响输出的比例，默认为 1.0
        lora_scale: float = 1.0,
        # 安全融合标志，默认为 False
        safe_fusing: bool = False,
        # 可选适配器名称列表，默认为 None
        adapter_names: Optional[List[str]] = None,
        # 其他关键字参数
        **kwargs,
    ):
        # 方法文档字符串，描述该方法的功能和参数
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        # 调用父类的方法进行 LoRA 参数融合，传递相关参数
        super().fuse_lora(
            components=components, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
        )

    # 定义一个方法，用于将 LoRA 参数从组件中移除
    def unfuse_lora(self, components: List[str] = ["transformer", "text_encoder", "text_encoder_2"], **kwargs):
        # 方法文档字符串，描述该方法的功能和参数
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        # 调用父类的方法进行 LoRA 参数移除，传递组件参数
        super().unfuse_lora(components=components)
# 定义一个混合类，用于加载 LoRA 层，继承自 LoraBaseMixin
class FluxLoraLoaderMixin(LoraBaseMixin):
    r"""
    加载 LoRA 层到 [`FluxTransformer2DModel`] 和 [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)。
    
    特定于 [`StableDiffusion3Pipeline`]。
    """

    # 可加载的 LoRA 模块名称列表
    _lora_loadable_modules = ["transformer", "text_encoder"]
    # Transformer 的名称
    transformer_name = TRANSFORMER_NAME
    # 文本编码器的名称
    text_encoder_name = TEXT_ENCODER_NAME

    # 类方法，验证 Hugging Face Hub 参数，并获取 LoRA 状态字典
    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        # 预训练模型的名称、路径或字典，类型可以是字符串或字典
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        # 是否返回 alpha 值，默认为 False
        return_alphas: bool = False,
        # 其他关键字参数
        **kwargs,
    ):
        # 方法体缺失，需实现具体逻辑
        pass

    # 实例方法，加载 LoRA 权重
    def load_lora_weights(
        self, 
        # 预训练模型的名称、路径或字典，类型可以是字符串或字典
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], 
        # 可选的适配器名称
        adapter_name=None, 
        # 其他关键字参数
        **kwargs
    ):
        # 方法体缺失，需实现具体逻辑
        pass
    ):
        """
        加载指定的 LoRA 权重到 `self.transformer` 和 `self.text_encoder`。

        所有关键字参数会转发给 `self.lora_state_dict`。

        详见 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] 如何加载状态字典。

        详见 [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] 如何将状态字典加载到 `self.transformer`。

        参数：
            pretrained_model_name_or_path_or_dict (`str` 或 `os.PathLike` 或 `dict`):
                详见 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`]。
            kwargs (`dict`, *可选*):
                详见 [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`]。
            adapter_name (`str`, *可选*):
                用于引用加载的适配器模型的名称。如果未指定，将使用
                `default_{i}`，其中 i 是加载的适配器总数。
        """
        # 检查是否启用 PEFT 后端，若未启用则抛出错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # 如果传入的是字典，则复制它以避免就地修改
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # 首先，确保检查点是兼容的，并可以成功加载
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, return_alphas=True, **kwargs
        )

        # 验证状态字典的格式是否正确，确保包含 "lora" 或 "dora_scale"
        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        # 将状态字典加载到 transformer 中
        self.load_lora_into_transformer(
            state_dict,
            network_alphas=network_alphas,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
        )

        # 从状态字典中提取与 text_encoder 相关的部分
        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        # 如果提取的字典不为空，则加载到 text_encoder 中
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
            )

    @classmethod
    @classmethod
    # 从 diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder 复制的
    # 定义一个类方法，用于将 Lora 模型加载到文本编码器中
        def load_lora_into_text_encoder(
            cls,  # 类本身
            state_dict,  # 状态字典，包含模型权重
            network_alphas,  # 网络中的缩放因子
            text_encoder,  # 文本编码器实例
            prefix=None,  # 可选的前缀，用于命名
            lora_scale=1.0,  # Lora 缩放因子，默认为 1.0
            adapter_name=None,  # 可选的适配器名称
            _pipeline=None,  # 可选的管道参数，用于进一步处理
        @classmethod  # 指定这是一个类方法
        # 从 diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.save_lora_weights 拷贝而来，将 unet 替换为 transformer
        def save_lora_weights(
            cls,  # 类本身
            save_directory: Union[str, os.PathLike],  # 保存权重的目录
            transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,  # transformer 的 Lora 层
            text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,  # 文本编码器的 Lora 层
            is_main_process: bool = True,  # 标识当前是否为主进程
            weight_name: str = None,  # 权重文件的名称
            save_function: Callable = None,  # 自定义保存函数
            safe_serialization: bool = True,  # 是否安全序列化，默认为 True
    ):
        r"""  # 定义文档字符串，描述此函数的功能及参数
        Save the LoRA parameters corresponding to the UNet and text encoder.  # 描述保存LoRA参数的功能

        Arguments:  # 开始列出函数的参数
            save_directory (`str` or `os.PathLike`):  # 参数：保存LoRA参数的目录，类型为字符串或路径类
                Directory to save LoRA parameters to. Will be created if it doesn't exist.  # 描述：如果目录不存在，将创建该目录
            transformer_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):  # 参数：与transformer对应的LoRA层的状态字典
                State dict of the LoRA layers corresponding to the `transformer`.  # 描述：说明参数的作用
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):  # 参数：与text_encoder对应的LoRA层的状态字典
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text  # 描述：说明此参数必须提供，来自🤗 Transformers
                encoder LoRA state dict because it comes from 🤗 Transformers.  # 继续描述参数的来源
            is_main_process (`bool`, *optional*, defaults to `True`):  # 参数：指示当前进程是否为主进程，类型为布尔值
                Whether the process calling this is the main process or not. Useful during distributed training and you  # 描述：用于分布式训练时判断主进程
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main  # 进一步说明如何使用此参数
                process to avoid race conditions.  # 描述：避免竞争条件
            save_function (`Callable`):  # 参数：用于保存状态字典的函数，类型为可调用对象
                The function to use to save the state dictionary. Useful during distributed training when you need to  # 描述：在分布式训练中，可能需要替换默认的保存方法
                replace `torch.save` with another method. Can be configured with the environment variable  # 说明如何配置此参数
                `DIFFUSERS_SAVE_MODE`.  # 提供环境变量名称
            safe_serialization (`bool`, *optional*, defaults to `True`):  # 参数：指示是否使用安全序列化保存模型，类型为布尔值
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.  # 描述：选择保存模型的方式
        """
        state_dict = {}  # 初始化一个空字典，用于存储状态字典

        if not (transformer_lora_layers or text_encoder_lora_layers):  # 检查是否至少有一个LoRA层字典传入
            raise ValueError("You must pass at least one of `transformer_lora_layers` and `text_encoder_lora_layers`.")  # 如果没有，抛出异常

        if transformer_lora_layers:  # 如果存在transformer的LoRA层字典
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))  # 打包LoRA权重并更新状态字典

        if text_encoder_lora_layers:  # 如果存在text_encoder的LoRA层字典
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, cls.text_encoder_name))  # 打包LoRA权重并更新状态字典

        # Save the model  # 保存模型的注释
        cls.write_lora_layers(  # 调用类方法以写入LoRA层
            state_dict=state_dict,  # 传入状态字典
            save_directory=save_directory,  # 传入保存目录
            is_main_process=is_main_process,  # 传入主进程标志
            weight_name=weight_name,  # 传入权重名称
            save_function=save_function,  # 传入保存函数
            safe_serialization=safe_serialization,  # 传入安全序列化标志
        )

    # Copied from diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora with unet->transformer  # 注释说明此方法的来源和修改
    def fuse_lora(  # 定义fuse_lora方法
        self,  # 方法的第一个参数是实例自身
        components: List[str] = ["transformer", "text_encoder"],  # 参数：要融合的组件列表，默认包含transformer和text_encoder
        lora_scale: float = 1.0,  # 参数：LoRA的缩放因子，默认值为1.0
        safe_fusing: bool = False,  # 参数：指示是否安全融合，默认值为False
        adapter_names: Optional[List[str]] = None,  # 参数：可选的适配器名称列表，默认为None
        **kwargs,  # 可接收其他关键字参数
    ):
        r""" 
        # 文档字符串，说明此函数的作用和用法
        
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.
        # 将 LoRA 参数融合到对应块的原始参数中

        <Tip warning={true}>
        # 警告提示，说明这是一个实验性 API

        This is an experimental API.
        # 这是一项实验性 API

        </Tip>

        Args:
            components: (`List[str]`): 
            # 参数说明，接受一个字符串列表，表示要融合 LoRA 的组件
            
            lora_scale (`float`, defaults to 1.0):
            # 参数说明，控制 LoRA 参数对输出的影响程度
            
                Controls how much to influence the outputs with the LoRA parameters.
                # 控制 LoRA 参数对输出的影响程度
            
            safe_fusing (`bool`, defaults to `False`):
            # 参数说明，是否在融合之前检查权重中是否有 NaN 值
            
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
                # 是否在融合之前检查权重的 NaN 值，如果存在则不进行融合
            
            adapter_names (`List[str]`, *optional*):
            # 参数说明，可选的适配器名称列表，用于融合
            
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.
                # 用于融合的适配器名称列表，如果未传入，则将融合所有活动适配器

        Example:
        # 示例代码，展示如何使用该 API

        ```py
        from diffusers import DiffusionPipeline
        # 导入 DiffusionPipeline 类
        
        import torch
        # 导入 PyTorch 库

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        # 从预训练模型创建管道，并将其移动到 CUDA 设备上
        
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        # 加载 LoRA 权重到管道中
        
        pipeline.fuse_lora(lora_scale=0.7)
        # 融合 LoRA 参数，设置影响程度为 0.7
        ```
        """
        super().fuse_lora(
            # 调用父类的 fuse_lora 方法，将相关参数传递给它
            components=components, lora_scale=lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names
        )

    def unfuse_lora(self, components: List[str] = ["transformer", "text_encoder"], **kwargs):
        r"""
        # 方法文档字符串，说明此方法的作用和用法
        
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).
        # 反转 fuse_lora 方法的效果

        <Tip warning={true}>
        # 警告提示，说明这是一个实验性 API

        This is an experimental API.
        # 这是一项实验性 API

        </Tip>

        Args:
            components (`List[str]`): 
            # 参数说明，接受一个字符串列表，表示要从中解除 LoRA 的组件
            
            List of LoRA-injectable components to unfuse LoRA from.
            # 要从中解除 LoRA 的组件列表
        """
        super().unfuse_lora(components=components)
        # 调用父类的 unfuse_lora 方法，将相关参数传递给它
# 这里我们从 `StableDiffusionLoraLoaderMixin` 子类化，因为 Amused 最初依赖于该类提供 LoRA 支持
class AmusedLoraLoaderMixin(StableDiffusionLoraLoaderMixin):
    # 可加载的 LoRA 模块列表
    _lora_loadable_modules = ["transformer", "text_encoder"]
    # 定义变换器的名称
    transformer_name = TRANSFORMER_NAME
    # 定义文本编码器的名称
    text_encoder_name = TEXT_ENCODER_NAME

    @classmethod
    @classmethod
    # 从 diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin 中复制的方法，用于将 LoRA 加载到文本编码器中
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        adapter_name=None,
        _pipeline=None,
    @classmethod
    # 定义保存 LoRA 权重的方法
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,
        transformer_lora_layers: Dict[str, torch.nn.Module] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    # 定义一个方法，保存与 UNet 和文本编码器对应的 LoRA 参数
        ):
            r""" 
            保存与 UNet 和文本编码器对应的 LoRA 参数。
    
            参数：
                save_directory (`str` 或 `os.PathLike`):
                    保存 LoRA 参数的目录。如果目录不存在，将被创建。
                unet_lora_layers (`Dict[str, torch.nn.Module]` 或 `Dict[str, torch.Tensor]`):
                    与 `unet` 相关的 LoRA 层的状态字典。
                text_encoder_lora_layers (`Dict[str, torch.nn.Module]` 或 `Dict[str, torch.Tensor]`):
                    与 `text_encoder` 相关的 LoRA 层的状态字典。必须明确传递文本编码器的 LoRA 状态字典，因为它来自 🤗 Transformers。
                is_main_process (`bool`, *可选*, 默认值为 `True`):
                    调用此函数的过程是否为主过程。在分布式训练期间，您需要在所有进程上调用此函数。在这种情况下，只有在主过程中将 `is_main_process=True`，以避免竞争条件。
                save_function (`Callable`):
                    用于保存状态字典的函数。在分布式训练时，需要用其他方法替换 `torch.save`。可以通过环境变量 `DIFFUSERS_SAVE_MODE` 进行配置。
                safe_serialization (`bool`, *可选*, 默认值为 `True`):
                    是否使用 `safetensors` 或传统的 PyTorch 方式通过 `pickle` 保存模型。
            """
            # 初始化状态字典，用于存储 LoRA 参数
            state_dict = {}
    
            # 检查至少传递一个 LoRA 层的状态字典
            if not (transformer_lora_layers or text_encoder_lora_layers):
                # 如果两个都没有，抛出错误
                raise ValueError("You must pass at least one of `transformer_lora_layers` or `text_encoder_lora_layers`.")
    
            # 如果有 transformer LoRA 层，更新状态字典
            if transformer_lora_layers:
                state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))
    
            # 如果有文本编码器 LoRA 层，更新状态字典
            if text_encoder_lora_layers:
                state_dict.update(cls.pack_weights(text_encoder_lora_layers, cls.text_encoder_name))
    
            # 保存模型的过程
            cls.write_lora_layers(
                # 传入状态字典
                state_dict=state_dict,
                # 保存目录
                save_directory=save_directory,
                # 是否为主进程
                is_main_process=is_main_process,
                # 权重名称
                weight_name=weight_name,
                # 保存函数
                save_function=save_function,
                # 是否使用安全序列化
                safe_serialization=safe_serialization,
            )
# 定义一个名为 LoraLoaderMixin 的类，继承自 StableDiffusionLoraLoaderMixin
class LoraLoaderMixin(StableDiffusionLoraLoaderMixin):
    # 初始化方法，接收可变位置和关键字参数
    def __init__(self, *args, **kwargs):
        # 设置弃用警告信息，提示用户该类将在未来版本中移除
        deprecation_message = "LoraLoaderMixin is deprecated and this will be removed in a future version. Please use `StableDiffusionLoraLoaderMixin`, instead."
        # 调用 deprecate 函数，记录该类的弃用信息
        deprecate("LoraLoaderMixin", "1.0.0", deprecation_message)
        # 调用父类的初始化方法，传递位置和关键字参数
        super().__init__(*args, **kwargs)
```