# `.\diffusers\pipelines\pag\pag_utils.py`

```py
# 版权所有 2024 HuggingFace 团队。所有权利保留。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有约定，软件
# 在“按原样”基础上分发，没有任何形式的保证或条件，
# 无论是明示还是暗示的。
# 有关许可证具体条款和条件，请参见许可文件。
import re  # 导入正则表达式模块，用于字符串模式匹配
from typing import Dict, List, Tuple, Union  # 从 typing 导入类型提示，用于增强代码可读性和类型检查

import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 从 PyTorch 导入神经网络模块，用于构建模型

from ...models.attention_processor import (  # 从相对路径导入注意力处理相关类
    Attention,  # 导入注意力机制类
    AttentionProcessor,  # 导入注意力处理器类
    PAGCFGIdentitySelfAttnProcessor2_0,  # 导入特定版本的身份自注意力处理器
    PAGIdentitySelfAttnProcessor2_0,  # 导入另一个特定版本的身份自注意力处理器
)
from ...utils import logging  # 从相对路径导入日志记录工具

logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器实例，pylint 禁用无效名称检查

class PAGMixin:  # 定义一个混合类，用于实现 Pertubed Attention Guidance 功能
    r"""Mixin class for [Pertubed Attention Guidance](https://arxiv.org/abs/2403.17377v1)."""  # 文档字符串，描述该混合类的目的及其引用
    # 定义设置 PAG 注意力处理器的私有方法，接收应用的层和分类器自由引导的标志
    def _set_pag_attn_processor(self, pag_applied_layers, do_classifier_free_guidance):
        r"""
        设置 PAG 层的注意力处理器。
        """
        # 获取当前对象的 PAG 注意力处理器列表
        pag_attn_processors = self._pag_attn_processors
        # 检查是否已设置 PAG 注意力处理器，如果没有，则抛出异常
        if pag_attn_processors is None:
            raise ValueError(
                "No PAG attention processors have been set. Set the attention processors by calling `set_pag_applied_layers` and passing the relevant parameters."
            )
    
        # 根据是否使用分类器自由引导选择对应的 PAG 注意力处理器
        pag_attn_proc = pag_attn_processors[0] if do_classifier_free_guidance else pag_attn_processors[1]
    
        # 检查当前对象是否具有 unet 属性
        if hasattr(self, "unet"):
            # 如果有，设置模型为 unet
            model: nn.Module = self.unet
        else:
            # 如果没有，设置模型为 transformer
            model: nn.Module = self.transformer
    
        # 定义一个检查模块是否为自注意力模块的函数
        def is_self_attn(module: nn.Module) -> bool:
            r"""
            根据模块名称检查它是否是自注意力模块。
            """
            # 判断模块是否为 Attention 类型且不是交叉注意力模块
            return isinstance(module, Attention) and not module.is_cross_attention
    
        # 定义一个检查是否为假积分匹配的函数
        def is_fake_integral_match(layer_id, name):
            # 获取层 ID 和名称的最后部分
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            # 检查层 ID 和名称是否都是数字并且相等
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name
    
        # 遍历应用的 PAG 层
        for layer_id in pag_applied_layers:
            # 为每个 PAG 层输入，找到在 unet 模型中对应的自注意力层
            target_modules = []
    
            # 遍历模型中的所有命名模块
            for name, module in model.named_modules():
                # 确定以下简单情况：
                #   (1) 存在自注意力层
                #   (2) 模块名称是否与 PAG 层 ID 部分匹配
                #   (3) 确保如果层 ID 以数字结尾则不是假积分匹配
                #       例如，blocks.1 和 blocks.10 应该可区分，如果 layer_id="blocks.1"
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                ):
                    # 记录调试信息，显示应用 PAG 到的层
                    logger.debug(f"Applying PAG to layer: {name}")
                    # 将匹配的模块添加到目标模块列表
                    target_modules.append(module)
    
            # 如果未找到任何目标模块，则抛出异常
            if len(target_modules) == 0:
                raise ValueError(f"Cannot find PAG layer to set attention processor for: {layer_id}")
    
            # 将选定的 PAG 注意力处理器分配给目标模块
            for module in target_modules:
                module.processor = pag_attn_proc
    
    # 定义获取在时间步 `t` 的扰动注意力引导的缩放因子的私有方法
    def _get_pag_scale(self, t):
        r"""
        获取时间步 `t` 的扰动注意力引导的缩放因子。
        """
    
        # 检查是否进行自适应缩放
        if self.do_pag_adaptive_scaling:
            # 计算信号缩放因子
            signal_scale = self.pag_scale - self.pag_adaptive_scale * (1000 - t)
            # 如果信号缩放小于 0，则设置为 0
            if signal_scale < 0:
                signal_scale = 0
            # 返回计算后的信号缩放因子
            return signal_scale
        else:
            # 否则直接返回预设的 PAG 缩放因子
            return self.pag_scale
    # 定义应用扰动注意力引导的函数，更新噪声预测
    def _apply_perturbed_attention_guidance(self, noise_pred, do_classifier_free_guidance, guidance_scale, t):
        r"""
        应用扰动注意力引导到噪声预测中。

        参数：
            noise_pred (torch.Tensor): 噪声预测张量。
            do_classifier_free_guidance (bool): 是否应用无分类器引导。
            guidance_scale (float): 引导项的缩放因子。
            t (int): 当前时间步。

        返回：
            torch.Tensor: 应用扰动注意力引导后更新的噪声预测张量。
        """
        # 获取当前时间步的引导缩放因子
        pag_scale = self._get_pag_scale(t)
        # 如果需要应用无分类器引导
        if do_classifier_free_guidance:
            # 将噪声预测张量分割为三个部分：无条件、文本和扰动
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
            # 更新噪声预测，结合无条件、文本和扰动信息
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            # 将噪声预测张量分割为两部分：文本和扰动
            noise_pred_text, noise_pred_perturb = noise_pred.chunk(2)
            # 更新噪声预测，仅结合文本和扰动信息
            noise_pred = noise_pred_text + pag_scale * (noise_pred_text - noise_pred_perturb)
        # 返回更新后的噪声预测
        return noise_pred

    # 定义准备扰动注意力引导的函数
    def _prepare_perturbed_attention_guidance(self, cond, uncond, do_classifier_free_guidance):
        """
        为 PAG 模型准备扰动注意力引导。

        参数：
            cond (torch.Tensor): 条件输入张量。
            uncond (torch.Tensor): 无条件输入张量。
            do_classifier_free_guidance (bool): 表示是否执行无分类器引导的标志。

        返回：
            torch.Tensor: 准备好的扰动注意力引导张量。
        """

        # 将条件输入张量在维度 0 上重复两次
        cond = torch.cat([cond] * 2, dim=0)

        # 如果需要应用无分类器引导
        if do_classifier_free_guidance:
            # 将无条件输入张量与条件张量在维度 0 上连接
            cond = torch.cat([uncond, cond], dim=0)
        # 返回准备好的条件张量
        return cond

    # 定义设置应用扰动注意力引导层的函数
    def set_pag_applied_layers(
        self,
        pag_applied_layers: Union[str, List[str]],  # 指定应用的层，可以是字符串或字符串列表
        pag_attn_processors: Tuple[AttentionProcessor, AttentionProcessor] = (  # 设置默认的注意力处理器
            PAGCFGIdentitySelfAttnProcessor2_0(),  # 第一个注意力处理器的实例
            PAGIdentitySelfAttnProcessor2_0(),  # 第二个注意力处理器的实例
        ),
    ):
        r""" 
        设置自注意力层以应用PAG。 如果输入无效，则引发ValueError。
        
        参数：
            pag_applied_layers (`str` 或 `List[str]`):
                一个或多个字符串标识层名称，或用于匹配多个层的简单正则表达式，PAG将应用于这些层。预期用法有几种：
                  - 单层指定为 - "blocks.{layer_index}"
                  - 多层作为列表 - ["blocks.{layers_index_1}", "blocks.{layer_index_2}", ...]
                  - 多层作为块名称 - "mid"
                  - 多层作为正则表达式 - "blocks.({layer_index_1}|{layer_index_2})"
            pag_attn_processors:
                (`Tuple[AttentionProcessor, AttentionProcessor]`, 默认值为 `(PAGCFGIdentitySelfAttnProcessor2_0(),
                PAGIdentitySelfAttnProcessor2_0())`): 一个包含两个注意力处理器的元组。第一个注意力
                处理器用于启用分类器无关指导的PAG（条件和无条件）。第二个
                注意力处理器用于禁用CFG的PAG（仅无条件）。
        """

        # 检查实例是否具有属性"_pag_attn_processors"，如果没有则将其设置为None
        if not hasattr(self, "_pag_attn_processors"):
            self._pag_attn_processors = None

        # 如果输入的pag_applied_layers不是列表，则将其转换为单元素列表
        if not isinstance(pag_applied_layers, list):
            pag_applied_layers = [pag_applied_layers]
        
        # 如果pag_attn_processors不为None，则检查其类型和长度
        if pag_attn_processors is not None:
            if not isinstance(pag_attn_processors, tuple) or len(pag_attn_processors) != 2:
                # 如果不满足条件，则引发ValueError
                raise ValueError("Expected a tuple of two attention processors")

        # 遍历pag_applied_layers中的每个元素，检查它们是否都是字符串类型
        for i in range(len(pag_applied_layers)):
            if not isinstance(pag_applied_layers[i], str):
                # 如果类型不匹配，则引发ValueError并输出类型信息
                raise ValueError(
                    f"Expected either a string or a list of string but got type {type(pag_applied_layers[i])}"
                )

        # 将有效的pag_applied_layers和pag_attn_processors存储到实例属性中
        self.pag_applied_layers = pag_applied_layers
        self._pag_attn_processors = pag_attn_processors

    @property
    def pag_scale(self) -> float:
        r"""获取扰动注意力引导的缩放因子。"""
        # 返回实例的_pag_scale属性
        return self._pag_scale

    @property
    def pag_adaptive_scale(self) -> float:
        r"""获取扰动注意力引导的自适应缩放因子。"""
        # 返回实例的_pag_adaptive_scale属性
        return self._pag_adaptive_scale

    @property
    def do_pag_adaptive_scaling(self) -> bool:
        r"""检查是否启用扰动注意力引导的自适应缩放。"""
        # 检查_pag_adaptive_scale和_pag_scale是否大于0，并且pag_applied_layers的长度大于0
        return self._pag_adaptive_scale > 0 and self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    def do_perturbed_attention_guidance(self) -> bool:
        r"""检查是否启用扰动注意力引导。"""
        # 检查_pag_scale是否大于0，并且pag_applied_layers的长度大于0
        return self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    # 定义一个方法，用于获取 PAG 注意力处理器
    def pag_attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的 PAG 注意力处理器：一个字典包含模型中使用的所有 PAG 注意力处理器
            以层的名称作为键。
        """
    
        # 检查 PAG 注意力处理器是否为 None，如果是，则返回空字典
        if self._pag_attn_processors is None:
            return {}
    
        # 创建一个集合，包含所有有效的注意力处理器类
        valid_attn_processors = {x.__class__ for x in self._pag_attn_processors}
    
        # 初始化一个字典，用于存储处理器
        processors = {}
        # 通过检查是否存在 'unet' 属性来决定使用哪个去噪模块
        # 如果存在，则使用 self.unet
        if hasattr(self, "unet"):
            denoiser_module = self.unet
        # 如果 'unet' 属性不存在，则检查 'transformer' 属性
        elif hasattr(self, "transformer"):
            denoiser_module = self.transformer
        # 如果两者都不存在，则引发错误
        else:
            raise ValueError("No denoiser module found.")
    
        # 遍历去噪模块中的注意力处理器
        for name, proc in denoiser_module.attn_processors.items():
            # 如果当前处理器类在有效处理器集合中，则将其添加到处理器字典中
            if proc.__class__ in valid_attn_processors:
                processors[name] = proc
    
        # 返回找到的处理器字典
        return processors
```