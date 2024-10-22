# `.\diffusers\pipelines\deprecated\score_sde_ve\pipeline_score_sde_ve.py`

```py
# 版权声明，表明此代码的版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 可以通过以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是以“按现状”基础提供的，不提供任何形式的保证或条件。
# 请参见许可证了解特定语言所涉及的权限和
# 限制条款。

# 从 typing 模块导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从指定路径导入 UNet2DModel 模型
from ....models import UNet2DModel
# 从指定路径导入 ScoreSdeVeScheduler 调度器
from ....schedulers import ScoreSdeVeScheduler
# 从指定路径导入 randn_tensor 函数
from ....utils.torch_utils import randn_tensor
# 从指定路径导入 DiffusionPipeline 和 ImagePipelineOutput
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 定义 ScoreSdeVePipeline 类，继承自 DiffusionPipeline
class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    用于无条件图像生成的管道。

    此模型继承自 [`DiffusionPipeline`]. 请查看超类文档以了解所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        unet ([`UNet2DModel`]):
            一个用于去噪编码图像的 `UNet2DModel`。
        scheduler ([`ScoreSdeVeScheduler`]):
            一个与 `unet` 结合使用以去噪编码图像的 `ScoreSdeVeScheduler`。
    """

    # 定义 unet 和 scheduler 属性，分别为 UNet2DModel 和 ScoreSdeVeScheduler 类型
    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    # 初始化方法，接收 unet 和 scheduler 作为参数
    def __init__(self, unet: UNet2DModel, scheduler: ScoreSdeVeScheduler):
        # 调用父类的初始化方法
        super().__init__()
        # 注册 unet 和 scheduler 模块
        self.register_modules(unet=unet, scheduler=scheduler)

    # 使用 @torch.no_grad() 装饰器，指示在调用时不计算梯度
    @torch.no_grad()
    def __call__(
        # 设置默认批大小为 1
        batch_size: int = 1,
        # 设置默认推理步骤数为 2000
        num_inference_steps: int = 2000,
        # 可选生成器，支持单个或多个 torch.Generator 实例
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选输出类型，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 可选返回字典，默认为 True
        return_dict: bool = True,
        # 接收额外的关键字参数
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r""" 
        生成的管道调用函数。

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                要生成的图像数量。
            generator (`torch.Generator`, `optional`):
                用于生成确定性结果的 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)。
            output_type (`str`, `optional`, defaults to `"pil"`):
                生成图像的输出格式。可以选择 `PIL.Image` 或 `np.array`。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`ImagePipelineOutput`] 而不是普通元组。

        Returns:
            [`~pipelines.ImagePipelineOutput`] 或 `tuple`:
                如果 `return_dict` 为 `True`，则返回 [`~pipelines.ImagePipelineOutput`]，否则返回一个元组，元组的第一个元素是生成图像的列表。
        """

        # 获取 UNet 模型配置中的样本大小
        img_size = self.unet.config.sample_size
        # 定义输入张量的形状
        shape = (batch_size, 3, img_size, img_size)

        # 赋值 UNet 模型
        model = self.unet

        # 生成随机张量并乘以初始噪声标准差
        sample = randn_tensor(shape, generator=generator) * self.scheduler.init_noise_sigma
        # 将样本移动到指定设备
        sample = sample.to(self.device)

        # 设置调度器的时间步
        self.scheduler.set_timesteps(num_inference_steps)
        # 设置调度器的 sigma 值
        self.scheduler.set_sigmas(num_inference_steps)

        # 遍历调度器时间步的进度条
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # 创建 sigma_t，形状为 (batch_size,)
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=self.device)

            # 修正步骤
            for _ in range(self.scheduler.config.correct_steps):
                # 获取模型输出
                model_output = self.unet(sample, sigma_t).sample
                # 更新样本
                sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample

            # 预测步骤
            model_output = model(sample, sigma_t).sample
            # 更新输出
            output = self.scheduler.step_pred(model_output, t, sample, generator=generator)

            # 获取上一个样本及其均值
            sample, sample_mean = output.prev_sample, output.prev_sample_mean

        # 限制样本均值在 0 到 1 之间
        sample = sample_mean.clamp(0, 1)
        # 转换样本格式为 numpy 数组
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        # 根据输出类型转换为 PIL 图像
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        # 如果不返回字典，则返回样本元组
        if not return_dict:
            return (sample,)

        # 返回图像管道输出
        return ImagePipelineOutput(images=sample)
```