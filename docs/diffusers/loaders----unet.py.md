# `.\diffusers\loaders\unet.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 在许可证下分发是按“原样”基础进行的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证下特定语言管理权限和
# 限制的信息，请参阅许可证。
import os  # 导入操作系统模块，提供与操作系统交互的功能
from collections import defaultdict  # 导入默认字典，用于处理键不存在时的默认值
from contextlib import nullcontext  # 导入空上下文管理器，提供不做任何操作的上下文
from pathlib import Path  # 导入路径库，便于处理文件路径
from typing import Callable, Dict, Union  # 导入类型提示，提供函数、字典和联合类型的支持

import safetensors  # 导入 safetensors 库，处理安全张量
import torch  # 导入 PyTorch 库，深度学习框架
import torch.nn.functional as F  # 导入 PyTorch 的神经网络功能模块
from huggingface_hub.utils import validate_hf_hub_args  # 导入用于验证 Hugging Face Hub 参数的工具
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..models.embeddings import (  # 从父级目录导入嵌入模型
    ImageProjection,  # 导入图像投影类
    IPAdapterFaceIDImageProjection,  # 导入人脸识别图像投影类
    IPAdapterFaceIDPlusImageProjection,  # 导入增强的人脸识别图像投影类
    IPAdapterFullImageProjection,  # 导入完整图像投影类
    IPAdapterPlusImageProjection,  # 导入增强图像投影类
    MultiIPAdapterImageProjection,  # 导入多种图像投影类
)
from ..models.modeling_utils import load_model_dict_into_meta, load_state_dict  # 导入模型加载工具
from ..utils import (  # 从父级目录导入工具模块
    USE_PEFT_BACKEND,  # 导入使用 PEFT 后端的标志
    _get_model_file,  # 导入获取模型文件的函数
    convert_unet_state_dict_to_peft,  # 导入转换 UNet 状态字典到 PEFT 的函数
    get_adapter_name,  # 导入获取适配器名称的函数
    get_peft_kwargs,  # 导入获取 PEFT 参数的函数
    is_accelerate_available,  # 导入检查加速可用性的函数
    is_peft_version,  # 导入检查 PEFT 版本的函数
    is_torch_version,  # 导入检查 PyTorch 版本的函数
    logging,  # 导入日志模块
)
from .lora_pipeline import LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE, TEXT_ENCODER_NAME, UNET_NAME  # 从当前目录导入 LoRA 权重名称和模型名称
from .utils import AttnProcsLayers  # 从当前目录导入注意力处理层

if is_accelerate_available():  # 检查是否可以使用加速功能
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module  # 导入加速库的钩子函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"  # 定义自定义扩散权重的文件名
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"  # 定义安全自定义扩散权重的文件名


class UNet2DConditionLoadersMixin:  # 定义一个混合类，用于加载 LoRA 层
    """
    将 LoRA 层加载到 [`UNet2DCondtionModel`] 中。
    """  # 类文档字符串，说明该类的作用

    text_encoder_name = TEXT_ENCODER_NAME  # 定义文本编码器名称
    unet_name = UNET_NAME  # 定义 UNet 名称

    @validate_hf_hub_args  # 使用装饰器验证 Hugging Face Hub 参数
    # 定义处理自定义扩散的方法，接收状态字典作为参数
    def _process_custom_diffusion(self, state_dict):
        # 从模块中导入自定义扩散注意力处理器
        from ..models.attention_processor import CustomDiffusionAttnProcessor

        # 初始化空字典，用于存储注意力处理器
        attn_processors = {}
        # 使用 defaultdict 初始化一个字典，用于分组自定义扩散数据
        custom_diffusion_grouped_dict = defaultdict(dict)
        
        # 遍历状态字典中的每一项
        for key, value in state_dict.items():
            # 如果当前值为空，设置分组字典的对应键为空字典
            if len(value) == 0:
                custom_diffusion_grouped_dict[key] = {}
            else:
                # 如果键中包含"to_out"，则提取相应的处理器键和子键
                if "to_out" in key:
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                else:
                    # 否则，按另一种方式提取处理器键和子键
                    attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                # 将值存储到分组字典中
                custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

        # 遍历分组字典中的每一项
        for key, value_dict in custom_diffusion_grouped_dict.items():
            # 如果值字典为空，初始化自定义扩散注意力处理器
            if len(value_dict) == 0:
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                )
            else:
                # 获取交叉注意力维度
                cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                # 获取隐藏层大小
                hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                # 判断是否训练 q 输出
                train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                # 初始化自定义扩散注意力处理器并传入参数
                attn_processors[key] = CustomDiffusionAttnProcessor(
                    train_kv=True,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
                # 加载状态字典到注意力处理器
                attn_processors[key].load_state_dict(value_dict)

        # 返回注意力处理器的字典
        return attn_processors

    # 类方法的装饰器
    @classmethod
    # 从 diffusers.loaders.lora_base.LoraBaseMixin 中复制的方法，用于选择性禁用卸载功能
    # 定义一个类方法，用于选择性地禁用模型的 CPU 离线
    def _optionally_disable_offloading(cls, _pipeline):
        """
        可选地移除离线处理，如果管道已经被顺序离线到 CPU。

        Args:
            _pipeline (`DiffusionPipeline`):
                需要禁用离线处理的管道。

        Returns:
            tuple:
                一个元组，指示 `is_model_cpu_offload` 或 `is_sequential_cpu_offload` 是否为 True。
        """
        # 初始化模型 CPU 离线标志为 False
        is_model_cpu_offload = False
        # 初始化顺序 CPU 离线标志为 False
        is_sequential_cpu_offload = False

        # 如果管道不为 None 且 hf_device_map 为 None
        if _pipeline is not None and _pipeline.hf_device_map is None:
            # 遍历管道中的每个组件
            for _, component in _pipeline.components.items():
                # 检查组件是否为 nn.Module 类型并且具有 _hf_hook 属性
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    # 如果模型尚未 CPU 离线
                    if not is_model_cpu_offload:
                        # 检查组件的 _hf_hook 是否为 CpuOffload 类型
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    # 如果顺序离线尚未设置
                    if not is_sequential_cpu_offload:
                        # 检查 _hf_hook 是否为 AlignDevicesHook 类型，或者其 hooks 属性的第一个元素是否为 AlignDevicesHook
                        is_sequential_cpu_offload = (
                            isinstance(component._hf_hook, AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                    # 记录信息，指示检测到加速钩子并即将移除之前的钩子
                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    # 从组件中移除钩子，是否递归取决于顺序离线标志
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        # 返回 CPU 离线标志的元组
        return (is_model_cpu_offload, is_sequential_cpu_offload)

    # 定义保存注意力处理器的方法
    def save_attn_procs(
        # 保存目录，支持字符串或路径对象
        save_directory: Union[str, os.PathLike],
        # 主进程标志，默认值为 True
        is_main_process: bool = True,
        # 权重名称，默认值为 None
        weight_name: str = None,
        # 保存功能，默认值为 None
        save_function: Callable = None,
        # 安全序列化标志，默认值为 True
        safe_serialization: bool = True,
        # 其他关键字参数
        **kwargs,
    ):
        # 定义获取自定义扩散状态字典的方法
        def _get_custom_diffusion_state_dict(self):
            # 从模型中导入自定义注意力处理器
            from ..models.attention_processor import (
                CustomDiffusionAttnProcessor,
                CustomDiffusionAttnProcessor2_0,
                CustomDiffusionXFormersAttnProcessor,
            )

            # 创建要保存的注意力处理器层
            model_to_save = AttnProcsLayers(
                {
                    # 过滤出类型为自定义注意力处理器的项
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(
                        x,
                        (
                            CustomDiffusionAttnProcessor,
                            CustomDiffusionAttnProcessor2_0,
                            CustomDiffusionXFormersAttnProcessor,
                        ),
                    )
                }
            )
            # 获取模型的状态字典
            state_dict = model_to_save.state_dict()
            # 遍历注意力处理器
            for name, attn in self.attn_processors.items():
                # 如果当前注意力处理器的状态字典为空
                if len(attn.state_dict()) == 0:
                    # 在状态字典中为该名称添加空字典
                    state_dict[name] = {}

            # 返回状态字典
            return state_dict
    # 加载 IP 适配器权重的私有方法
        def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
            # 检查 state_dicts 是否为列表，如果不是则转换为列表
            if not isinstance(state_dicts, list):
                state_dicts = [state_dicts]
    
            # 如果已有编码器隐藏投影且配置为文本投影，则赋值给文本编码器隐藏投影
            if (
                self.encoder_hid_proj is not None
                and self.config.encoder_hid_dim_type == "text_proj"
                and not hasattr(self, "text_encoder_hid_proj")
            ):
                self.text_encoder_hid_proj = self.encoder_hid_proj
    
            # 在加载 IP 适配器权重后将 encoder_hid_proj 设置为 None
            self.encoder_hid_proj = None
    
            # 将 IP 适配器的注意力处理器转换为 Diffusers 格式
            attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
            # 设置注意力处理器
            self.set_attn_processor(attn_procs)
    
            # 转换 IP 适配器图像投影层为 Diffusers 格式
            image_projection_layers = []
            # 遍历每个 state_dict，转换图像投影层
            for state_dict in state_dicts:
                image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                    state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
                )
                # 将转换后的图像投影层添加到列表中
                image_projection_layers.append(image_projection_layer)
    
            # 创建多重 IP 适配器图像投影并赋值给 encoder_hid_proj
            self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
            # 更新编码器隐藏维度类型为图像投影
            self.config.encoder_hid_dim_type = "ip_image_proj"
    
            # 将模型转移到指定的数据类型和设备
            self.to(dtype=self.dtype, device=self.device)
    # 加载 IP 适配器的 LoRA 权重，返回包含这些权重的字典
    def _load_ip_adapter_loras(self, state_dicts):
        # 初始化空字典以存储 LoRA 权重
        lora_dicts = {}
        # 遍历注意力处理器的键值对，获取索引和名称
        for key_id, name in enumerate(self.attn_processors.keys()):
            # 遍历每个状态字典
            for i, state_dict in enumerate(state_dicts):
                # 检查当前状态字典中是否包含特定的 LoRA 权重
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    # 如果该索引不在字典中，则初始化为空字典
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    # 更新字典，添加 'to_k_lora.down.weight' 的权重
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    # 更新字典，添加 'to_q_lora.down.weight' 的权重
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    # 更新字典，添加 'to_v_lora.down.weight' 的权重
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    # 更新字典，添加 'to_out_lora.down.weight' 的权重
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    # 更新字典，添加 'to_k_lora.up.weight' 的权重
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    # 更新字典，添加 'to_q_lora.up.weight' 的权重
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    # 更新字典，添加 'to_v_lora.up.weight' 的权重
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    # 更新字典，添加 'to_out_lora.up.weight' 的权重
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )
        # 返回包含所有 LoRA 权重的字典
        return lora_dicts
```