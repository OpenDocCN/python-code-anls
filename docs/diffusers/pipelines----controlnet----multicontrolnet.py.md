# `.\diffusers\pipelines\controlnet\multicontrolnet.py`

```py
# 导入操作系统模块
import os
# 从 typing 模块导入类型注解
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 torch 模块导入神经网络相关功能
from torch import nn

# 从上级目录导入 ControlNetModel 和 ControlNetOutput
from ...models.controlnet import ControlNetModel, ControlNetOutput
# 从上级目录导入 ModelMixin 类
from ...models.modeling_utils import ModelMixin
# 从上级目录导入 logging 工具
from ...utils import logging

# 创建一个日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)

# 定义 MultiControlNetModel 类，继承自 ModelMixin
class MultiControlNetModel(ModelMixin):
    r"""
    多个 `ControlNetModel` 的包装类，用于 Multi-ControlNet

    该模块是多个 `ControlNetModel` 实例的包装器。`forward()` API 设计为与 `ControlNetModel` 兼容。

    参数:
        controlnets (`List[ControlNetModel]`):
            在去噪过程中为 unet 提供额外的条件。必须将多个 `ControlNetModel` 作为列表设置。
    """

    # 初始化方法，接收一个 ControlNetModel 的列表或元组
    def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]):
        # 调用父类的初始化方法
        super().__init__()
        # 将控制网模型保存到模块列表中
        self.nets = nn.ModuleList(controlnets)

    # 前向传播方法，处理输入数据
    def forward(
        self,
        sample: torch.Tensor,  # 输入样本
        timestep: Union[torch.Tensor, float, int],  # 当前时间步
        encoder_hidden_states: torch.Tensor,  # 编码器的隐藏状态
        controlnet_cond: List[torch.tensor],  # 控制网络的条件
        conditioning_scale: List[float],  # 条件缩放因子
        class_labels: Optional[torch.Tensor] = None,  # 可选的类标签
        timestep_cond: Optional[torch.Tensor] = None,  # 可选的时间步条件
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,  # 可选的附加条件参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        guess_mode: bool = False,  # 是否使用猜测模式
        return_dict: bool = True,  # 是否返回字典格式的输出
    ) -> Union[ControlNetOutput, Tuple]:  # 返回类型可以是 ControlNetOutput 或元组
        # 遍历每个控制网络条件和缩放因子
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            # 调用控制网络进行前向传播，获取下采样和中间样本
            down_samples, mid_sample = controlnet(
                sample=sample,  # 输入样本
                timestep=timestep,  # 当前时间步
                encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                controlnet_cond=image,  # 控制网络条件
                conditioning_scale=scale,  # 条件缩放
                class_labels=class_labels,  # 类标签
                timestep_cond=timestep_cond,  # 时间步条件
                attention_mask=attention_mask,  # 注意力掩码
                added_cond_kwargs=added_cond_kwargs,  # 附加条件参数
                cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                guess_mode=guess_mode,  # 猜测模式
                return_dict=return_dict,  # 返回格式
            )

            # 合并样本
            if i == 0:  # 如果是第一个控制网络
                # 初始化下采样和中间样本
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:  # 如果不是第一个控制网络
                # 将当前下采样样本与之前的样本合并
                down_block_res_samples = [
                    samples_prev + samples_curr  # 累加下采样样本
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                # 累加中间样本
                mid_block_res_sample += mid_sample

        # 返回合并后的下采样样本和中间样本
        return down_block_res_samples, mid_block_res_sample
    # 定义一个方法，用于将模型及其配置文件保存到指定目录
    def save_pretrained(
        self,  # 代表类实例
        save_directory: Union[str, os.PathLike],  # 保存目录，可以是字符串或路径类型
        is_main_process: bool = True,  # 指示当前进程是否为主进程，默认为 True
        save_function: Callable = None,  # 自定义保存函数，默认为 None
        safe_serialization: bool = True,  # 是否使用安全序列化方式，默认为 True
        variant: Optional[str] = None,  # 可选参数，指定保存权重的格式
    ):
        """
        保存模型及其配置文件到指定目录，以便可以使用
        `[`~pipelines.controlnet.MultiControlNetModel.from_pretrained`]` 类方法重新加载。

        参数：
            save_directory (`str` 或 `os.PathLike`):
                要保存的目录，如果不存在则会创建。
            is_main_process (`bool`, *可选*, 默认为 `True`):
                调用此方法的进程是否为主进程，适用于分布式训练，避免竞争条件。
            save_function (`Callable`):
                用于保存状态字典的函数，适用于分布式训练。
            safe_serialization (`bool`, *可选*, 默认为 `True`):
                是否使用 `safetensors` 保存模型，或使用传统的 PyTorch 方法。
            variant (`str`, *可选*):
                如果指定，权重将以 pytorch_model.<variant>.bin 格式保存。
        """
        # 遍历网络模型列表，并获取索引
        for idx, controlnet in enumerate(self.nets):
            # 确定后缀名，如果是第一个模型则无后缀
            suffix = "" if idx == 0 else f"_{idx}"
            # 调用每个控制网络的保存方法，传入相关参数
            controlnet.save_pretrained(
                save_directory + suffix,  # 结合目录和后缀形成完整的保存路径
                is_main_process=is_main_process,  # 传递主进程标识
                save_function=save_function,  # 传递保存函数
                safe_serialization=safe_serialization,  # 传递序列化方式
                variant=variant,  # 传递权重格式
            )
```