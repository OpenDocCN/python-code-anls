# `.\cogview3-finetune\sat\diffusion.py`

```
# 导入数学库以进行数学运算
import math
# 从 typing 模块导入类型提示相关的类
from typing import Any, Dict, List, Tuple, Union

# 导入 PyTorch 库及其 nn 模块
import torch
from torch import nn
# 导入 PyTorch 的功能模块
import torch.nn.functional as F

# 从 sgm.modules 导入未指定条件的配置
from sgm.modules import UNCONDITIONAL_CONFIG
# 从 sgm.modules.diffusionmodules.wrappers 导入 OPENAIUNETWRAPPER 类
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
# 从 sgm.util 导入默认值获取、字符串到对象的转换、配置实例化函数
from sgm.util import default, get_obj_from_str, instantiate_from_config


# 定义 SATDiffusionEngine 类，继承自 nn.Module
class SATDiffusionEngine(nn.Module):
    # 使用装饰器禁用梯度计算
    @torch.no_grad()
    # 定义解码第一阶段的方法
    def decode_first_stage(self, z):
        # 根据缩放因子调整 z 的值
        z = 1.0 / self.scale_factor * z
        # 获取每次解码的样本数量，使用默认值处理
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        # 计算所需的轮数以解码所有样本
        n_rounds = math.ceil(z.shape[0] / n_samples)
        # 创建一个空列表以存储输出
        all_out = []
        # 在自动混合精度的上下文中运行
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            # 遍历每一轮
            for n in range(n_rounds):
                # 解码当前样本批次
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples])
                # 将输出添加到输出列表中
                all_out.append(out)
        # 将所有输出在第0维拼接
        out = torch.cat(all_out, dim=0)
        # 返回拼接后的输出
        return out

    # 使用装饰器禁用梯度计算
    @torch.no_grad()
    # 定义编码第一阶段的方法
    def encode_first_stage(self, x):
        # 获取每次编码的样本数量，使用默认值处理
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        # 计算所需的轮数以编码所有样本
        n_rounds = math.ceil(x.shape[0] / n_samples)
        # 创建一个空列表以存储输出
        all_out = []
        # 在自动混合精度的上下文中运行
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            # 遍历每一轮
            for n in range(n_rounds):
                # 编码当前样本批次
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                # 将输出添加到输出列表中
                all_out.append(out)
        # 将所有输出在第0维拼接
        z = torch.cat(all_out, dim=0)
        # 根据缩放因子调整 z 的值
        z = self.scale_factor * z
        # 返回编码后的结果
        return z

    # 定义前向传播的方法
    def forward(self, x, batch, **kwargs):
        # 计算损失
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        # 计算损失的均值
        loss_mean = loss.mean()
        # 创建一个字典以存储损失
        loss_dict = {"loss": loss_mean}
        # 返回损失均值和损失字典
        return loss_mean, loss_dict

    # 定义共享步骤的方法
    def shared_step(self, batch: Dict) -> Any:
        # 从批次中获取输入
        x = self.get_input(batch)
        # 检查学习率缩放因子是否为 None
        if self.lr_scale is not None:
            # 对输入进行下采样
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            # 对下采样后的输入进行上采样
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            # 编码下采样后的输入
            lr_z = self.encode_first_stage(lr_x)
            # 将编码结果存入批次
            batch["lr_input"] = lr_z
        # 编码原始输入
        x = self.encode_first_stage(x)
        # batch["global_step"] = self.global_step  # 这行被注释掉，可能是为了调试或保留未来的扩展
        # 计算损失和损失字典
        loss, loss_dict = self(x, batch)
        # 返回损失和损失字典
        return loss, loss_dict

    # 使用装饰器禁用梯度计算
    @torch.no_grad()
    # 定义采样的方法，包含条件和可选参数
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        target_size=None,
        **kwargs,
    ):
        # 生成形状为 (batch_size, *shape) 的随机正态分布张量，并转换为 float32 类型，移动到指定设备
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)

        # 检查是否提供 target_size
        if target_size is not None:
            # 定义 denoiser 函数，包含 target_size 参数
            denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
                self.model, input, sigma, c, target_size=target_size, **additional_model_inputs
            )
        else:
            # 定义 denoiser 函数，不包含 target_size 参数
            denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
                self.model, input, sigma, c, **additional_model_inputs
            )
        # 使用采样器生成样本
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        # 检查样本是否为列表
        if isinstance(samples, list):
            # 遍历样本列表，将每个样本转换为指定数据类型
            for i in range(len(samples)):
                samples[i] = samples[i].to(self.dtype)
        else:
            # 将样本转换为指定数据类型
            samples = samples.to(self.dtype)
        # 返回生成的样本
        return samples

    @torch.no_grad()  # 禁用梯度计算以节省内存和提高速度
    def sample_relay(
        self,
        image: torch.Tensor,  # 输入图像张量
        cond: Dict,  # 条件字典
        uc: Union[Dict, None] = None,  # 可选的未条件字典
        batch_size: int = 16,  # 批量大小，默认为16
        shape: Union[None, Tuple, List] = None,  # 输出形状，可选
        **kwargs,  # 其他可选参数
    ):
        # 生成形状为 (batch_size, *shape) 的随机正态分布张量，转换为指定数据类型，移动到指定设备
        randn = torch.randn(batch_size, *shape).to(self.dtype).to(self.device)
        # 定义 denoiser 函数
        denoiser = lambda input, sigma, c, **additional_model_inputs: self.denoiser(
            self.model, input, sigma, c, **additional_model_inputs
        )
        # 使用采样器生成样本
        samples = self.sampler(denoiser, image, randn, cond, uc=uc)
        # 检查样本是否为列表
        if isinstance(samples, list):
            # 遍历样本列表，将每个样本转换为指定数据类型
            for i in range(len(samples)):
                samples[i] = samples[i].to(self.dtype)
        else:
            # 将样本转换为指定数据类型
            samples = samples.to(self.dtype)
        # 返回生成的样本
        return samples
```