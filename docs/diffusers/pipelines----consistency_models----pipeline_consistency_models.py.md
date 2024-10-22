# `.\diffusers\pipelines\consistency_models\pipeline_consistency_models.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件是“按原样”提供的，
# 不附带任何形式的保证或条件，无论是明示还是暗示。
# 请参阅许可证以获取有关权限和限制的具体信息。

from typing import Callable, List, Optional, Union  # 导入类型注解，用于函数签名和变量类型标注

import torch  # 导入 PyTorch 库，供后续深度学习模型使用

from ...models import UNet2DModel  # 从模型模块导入 UNet2DModel 类
from ...schedulers import CMStochasticIterativeScheduler  # 从调度模块导入 CMStochasticIterativeScheduler 类
from ...utils import (  # 从工具模块导入多个工具函数和类
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从管道工具模块导入 DiffusionPipeline 和 ImagePipelineOutput 类

logger = logging.get_logger(__name__)  # 创建日志记录器，记录当前模块的日志信息

EXAMPLE_DOC_STRING = """  # 示例文档字符串，提供用法示例
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库

        >>> from diffusers import ConsistencyModelPipeline  # 从 diffusers 导入 ConsistencyModelPipeline 类

        >>> device = "cuda"  # 设置设备为 CUDA（GPU）
        >>> # 加载 cd_imagenet64_l2 检查点。
        >>> model_id_or_path = "openai/diffusers-cd_imagenet64_l2"  # 指定模型 ID 或路径
        >>> pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)  # 从预训练模型加载管道
        >>> pipe.to(device)  # 将管道移到指定设备上

        >>> # 单步采样
        >>> image = pipe(num_inference_steps=1).images[0]  # 使用单步推理生成图像
        >>> image.save("cd_imagenet64_l2_onestep_sample.png")  # 保存生成的图像

        >>> # 单步采样，类条件图像生成
        >>> # ImageNet-64 类标签 145 对应于国王企鹅
        >>> image = pipe(num_inference_steps=1, class_labels=145).images[0]  # 生成特定类的图像
        >>> image.save("cd_imagenet64_l2_onestep_sample_penguin.png")  # 保存生成的图像

        >>> # 多步采样，类条件图像生成
        >>> # 可以显式指定时间步，以下时间步来自原始 GitHub 仓库：
        >>> # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
        >>> image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]  # 生成特定类的多步图像
        >>> image.save("cd_imagenet64_l2_multistep_sample_penguin.png")  # 保存生成的图像
        ```py
"""

class ConsistencyModelPipeline(DiffusionPipeline):  # 定义 ConsistencyModelPipeline 类，继承自 DiffusionPipeline
    r"""  # 类的文档字符串，描述其功能
    Pipeline for unconditional or class-conditional image generation.  # 描述此管道用于无条件或类条件图像生成

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods  # 说明此模型继承自 DiffusionPipeline，并建议查看超类文档以了解通用方法
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).  # 说明所实现的方法，包括下载、保存和在特定设备上运行等
    # 函数参数说明
    Args:
        unet ([`UNet2DModel`]):  # 传入一个 UNet2DModel 对象，用于对编码后的图像潜变量去噪。
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):  # 传入一个调度器，结合 unet 用于去噪，当前仅与 CMStochasticIterativeScheduler 兼容。
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            compatible with [`CMStochasticIterativeScheduler`].
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "unet"

    # 构造函数，初始化类的实例
    def __init__(self, unet: UNet2DModel, scheduler: CMStochasticIterativeScheduler) -> None:
        # 调用父类的构造函数
        super().__init__()

        # 注册 unet 和 scheduler 模块
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        # 初始化安全检查器为 None
        self.safety_checker = None

    # 准备潜变量的函数
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        # 定义潜变量的形状
        shape = (batch_size, num_channels, height, width)
        # 检查生成器列表的长度是否与批量大小匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有传入潜变量，则生成随机潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 将潜变量移动到指定设备并转换数据类型
            latents = latents.to(device=device, dtype=dtype)

        # 根据调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    # 后处理图像的函数，遵循 diffusers.VaeImageProcessor.postprocess
    def postprocess_image(self, sample: torch.Tensor, output_type: str = "pil"):
        # 检查输出类型是否合法
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(
                f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )

        # 等同于 diffusers.VaeImageProcessor.denormalize
        sample = (sample / 2 + 0.5).clamp(0, 1)  # 将样本值归一化到 [0, 1] 范围内
        if output_type == "pt":  # 如果输出类型为 pt，直接返回样本
            return sample

        # 等同于 diffusers.VaeImageProcessor.pt_to_numpy
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()  # 转换为 NumPy 数组
        if output_type == "np":  # 如果输出类型为 np，返回样本
            return sample

        # 如果输出类型必须为 'pil'
        sample = self.numpy_to_pil(sample)  # 将 NumPy 数组转换为 PIL 图像
        return sample  # 返回最终的图像
    # 准备类别标签，根据给定的批大小和设备，将类别标签转换为张量
    def prepare_class_labels(self, batch_size, device, class_labels=None):
        # 检查 UNet 配置中类别嵌入的数量是否不为 None
        if self.unet.config.num_class_embeds is not None:
            # 如果 class_labels 是一个列表，将其转换为整型张量
            if isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, dtype=torch.int)
            # 如果 class_labels 是一个整数，确保批大小为 1，并将其转换为张量
            elif isinstance(class_labels, int):
                assert batch_size == 1, "Batch size must be 1 if classes is an int"
                class_labels = torch.tensor([class_labels], dtype=torch.int)
            # 如果 class_labels 为 None，随机生成 batch_size 个类别标签
            elif class_labels is None:
                # 随机生成 batch_size 类别标签
                # TODO: 应该在这里使用生成器吗？randn_tensor 的整数等价物未在 ...utils 中公开
                class_labels = torch.randint(0, self.unet.config.num_class_embeds, size=(batch_size,))
            # 将类别标签移动到指定的设备上
            class_labels = class_labels.to(device)
        else:
            # 如果没有类别嵌入，类别标签设为 None
            class_labels = None
        # 返回处理后的类别标签
        return class_labels

    # 检查输入参数的有效性
    def check_inputs(self, num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps):
        # 确保提供了 num_inference_steps 或 timesteps 其中之一
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        # 如果同时提供了 num_inference_steps 和 timesteps，发出警告
        if num_inference_steps is not None and timesteps is not None:
            logger.warning(
                f"Both `num_inference_steps`: {num_inference_steps} and `timesteps`: {timesteps} are supplied;"
                " `timesteps` will be used over `num_inference_steps`."
            )

        # 如果 latents 不为 None，检查其形状是否符合预期
        if latents is not None:
            expected_shape = (batch_size, 3, img_size, img_size)
            # 如果 latents 的形状不符合预期，则抛出错误
            if latents.shape != expected_shape:
                raise ValueError(f"The shape of latents is {latents.shape} but is expected to be {expected_shape}.")

        # 检查 callback_steps 是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 装饰器，禁止梯度计算，提供调用示例文档字符串
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义默认参数和类型注解，初始化调用方法
        batch_size: int = 1,
        class_labels: Optional[Union[torch.Tensor, List[int], int]] = None,
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
```