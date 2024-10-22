# `.\diffusers\pipelines\deepfloyd_if\pipeline_if_inpainting.py`

```py
# 导入 HTML 处理库
import html
# 导入检查对象信息的库
import inspect
# 导入正则表达式库
import re
# 导入 urllib 的解析模块并重命名为 ul
import urllib.parse as ul
# 从 typing 导入多种类型提示工具
from typing import Any, Callable, Dict, List, Optional, Union

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入图像处理库 PIL
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 导入 CLIP 图像处理器、T5 编码模型和 T5 分词器
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

# 从 loaders 导入 StableDiffusionLoraLoaderMixin 类
from ...loaders import StableDiffusionLoraLoaderMixin
# 从 models 导入 UNet2DConditionModel 类
from ...models import UNet2DConditionModel
# 从 schedulers 导入 DDPMScheduler 类
from ...schedulers import DDPMScheduler
# 从 utils 导入多个实用工具
from ...utils import (
    BACKENDS_MAPPING,          # 后端映射
    PIL_INTERPOLATION,        # PIL 插值方法
    is_bs4_available,         # 检查 bs4 可用性
    is_ftfy_available,        # 检查 ftfy 可用性
    logging,                  # 日志记录工具
    replace_example_docstring, # 替换示例文档字符串的工具
)
# 从 utils.torch_utils 导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline
# 从 pipeline_output 导入 IFPipelineOutput 类
from .pipeline_output import IFPipelineOutput
# 从 safety_checker 导入 IFSafetyChecker 类
from .safety_checker import IFSafetyChecker
# 从 watermark 导入 IFWatermarker 类
from .watermark import IFWatermarker

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 如果 bs4 可用，则导入 BeautifulSoup 类
if is_bs4_available():
    from bs4 import BeautifulSoup

# 如果 ftfy 可用，则导入 ftfy 库
if is_ftfy_available():
    import ftfy

# 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.resize 复制的 resize 函数
def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    # 获取输入图像的宽度和高度
    w, h = images.size

    # 计算宽高比
    coef = w / h

    # 将宽度和高度都设置为目标大小
    w, h = img_size, img_size

    # 根据宽高比调整宽度或高度
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)  # 调整宽度为最接近的8的倍数
    else:
        h = int(round(img_size / 8 / coef) * 8)  # 调整高度为最接近的8的倍数

    # 调整图像大小，使用双三次插值法
    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)

    # 返回调整后的图像
    return images

# 示例文档字符串
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
```py 
    # 示例代码说明如何使用图像修复和超分辨率模型
        Examples:
            ```py
            # 导入所需的库和模块
            >>> from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
            >>> from diffusers.utils import pt_to_pil
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            >>> from io import BytesIO
    
            # 定义图像的URL
            >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
            # 发送GET请求获取图像
            >>> response = requests.get(url)
            # 打开图像并转换为RGB格式
            >>> original_image = Image.open(BytesIO(response.content)).convert("RGB")
            # 将原始图像赋值给变量
            >>> original_image = original_image
    
            # 定义掩膜图像的URL
            >>> url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
            # 发送GET请求获取掩膜图像
            >>> response = requests.get(url)
            # 打开掩膜图像
            >>> mask_image = Image.open(BytesIO(response.content))
            # 将掩膜图像赋值给变量
            >>> mask_image = mask_image
    
            # 从预训练模型加载图像修复管道
            >>> pipe = IFInpaintingPipeline.from_pretrained(
            ...     "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
            ... )
            # 启用模型的CPU卸载功能以节省内存
            >>> pipe.enable_model_cpu_offload()
    
            # 定义图像修复的提示
            >>> prompt = "blue sunglasses"
            # 对提示进行编码，生成提示嵌入和负面嵌入
            >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
    
            # 使用管道进行图像修复
            >>> image = pipe(
            ...     image=original_image,
            ...     mask_image=mask_image,
            ...     prompt_embeds=prompt_embeds,
            ...     negative_prompt_embeds=negative_embeds,
            ...     output_type="pt",
            ... ).images
    
            # 保存中间图像为文件
            >>> # save intermediate image
            >>> pil_image = pt_to_pil(image)
            >>> pil_image[0].save("./if_stage_I.png")
    
            # 从预训练模型加载超分辨率管道
            >>> super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
            ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
            ... )
            # 启用模型的CPU卸载功能
            >>> super_res_1_pipe.enable_model_cpu_offload()
    
            # 使用超分辨率管道处理图像
            >>> image = super_res_1_pipe(
            ...     image=image,
            ...     mask_image=mask_image,
            ...     original_image=original_image,
            ...     prompt_embeds=prompt_embeds,
            ...     negative_prompt_embeds=negative_embeds,
            ... ).images
            # 保存最终图像为文件
            >>> image[0].save("./if_stage_II.png")
            ```py
"""
# 文档字符串，通常用于描述类的用途和功能
class IFInpaintingPipeline(DiffusionPipeline, StableDiffusionLoraLoaderMixin):
    # 定义一个类，继承自 DiffusionPipeline 和 StableDiffusionLoraLoaderMixin

    tokenizer: T5Tokenizer
    # 声明一个属性 tokenizer，类型为 T5Tokenizer

    text_encoder: T5EncoderModel
    # 声明一个属性 text_encoder，类型为 T5EncoderModel

    unet: UNet2DConditionModel
    # 声明一个属性 unet，类型为 UNet2DConditionModel

    scheduler: DDPMScheduler
    # 声明一个属性 scheduler，类型为 DDPMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    # 声明一个可选属性 feature_extractor，类型为 CLIPImageProcessor

    safety_checker: Optional[IFSafetyChecker]
    # 声明一个可选属性 safety_checker，类型为 IFSafetyChecker

    watermarker: Optional[IFWatermarker]
    # 声明一个可选属性 watermarker，类型为 IFWatermarker

    # 使用正则表达式编译不良标点符号的模式
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    # 定义一个可选组件列表，包含不同的组件名称
    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "text_encoder->unet"
    # 定义需要从 CPU 卸载中排除的组件列表
    _exclude_from_cpu_offload = ["watermarker"]

    # 初始化方法，设置类的属性
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 如果安全检查器为 None 且需要安全检查器，发出警告
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 如果安全检查器不为 None 且特征提取器为 None，抛出错误
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 注册模块，包括所有必要的组件
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        # 注册配置，指定是否需要安全检查器
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    # 禁用梯度计算，通常用于推理阶段以节省内存
    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.encode_prompt
    # 定义一个方法用于编码输入提示
        def encode_prompt(
            self,
            # 输入的提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 是否进行无分类器自由引导，默认为 True
            do_classifier_free_guidance: bool = True,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 指定设备，可选，默认为 None
            device: Optional[torch.device] = None,
            # 可选的负面提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 可选的提示嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 可选的负面提示嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 是否清理提示，默认为 False
            clean_caption: bool = False,
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.run_safety_checker 复制
        def run_safety_checker(self, image, device, dtype):
            # 如果存在安全检查器
            if self.safety_checker is not None:
                # 将图像转换为 PIL 格式并提取特征，返回张量，移动到指定设备
                safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
                # 使用安全检查器检测图像，返回处理后的图像及检测结果
                image, nsfw_detected, watermark_detected = self.safety_checker(
                    images=image,
                    # 获取安全检查器输入的像素值并转换数据类型
                    clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
                )
            else:
                # 如果没有安全检查器，设置检测结果为 None
                nsfw_detected = None
                watermark_detected = None
    
            # 返回处理后的图像及检测结果
            return image, nsfw_detected, watermark_detected
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_extra_step_kwargs 复制
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为并非所有调度器的签名相同
            # eta (η) 仅在 DDIMScheduler 中使用，其他调度器将忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 值应在 [0, 1] 之间
    
            # 检查调度器是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化额外参数字典
            extra_step_kwargs = {}
            # 如果接受 eta，将其添加到额外参数字典
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，将其添加到额外参数字典
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外参数字典
            return extra_step_kwargs
    
        # 定义一个方法用于检查输入参数
        def check_inputs(
            self,
            # 输入的提示
            prompt,
            # 输入的图像
            image,
            # 输入的掩码图像
            mask_image,
            # 批处理大小
            batch_size,
            # 回调步骤
            callback_steps,
            # 可选的负面提示
            negative_prompt=None,
            # 可选的提示嵌入
            prompt_embeds=None,
            # 可选的负面提示嵌入
            negative_prompt_embeds=None,
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing 复制
    # 定义文本预处理函数，接受文本和清理标题标志
        def _text_preprocessing(self, text, clean_caption=False):
            # 如果设置清理标题且未安装 bs4，则记录警告并将标志设置为 False
            if clean_caption and not is_bs4_available():
                logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果设置清理标题且未安装 ftfy，则记录警告并将标志设置为 False
            if clean_caption and not is_ftfy_available():
                logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
                logger.warning("Setting `clean_caption` to False...")
                clean_caption = False
    
            # 如果输入文本不是元组或列表，则将其转换为列表
            if not isinstance(text, (tuple, list)):
                text = [text]
    
            # 定义内部处理函数，清理或标准化文本
            def process(text: str):
                if clean_caption:
                    text = self._clean_caption(text)  # 清理标题
                    text = self._clean_caption(text)  # 再次清理标题
                else:
                    text = text.lower().strip()  # 转小写并去除首尾空格
                return text
    
            # 返回处理后的文本列表
            return [process(t) for t in text]
    
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption 复制
        # 从 diffusers.pipelines.deepfloyd_if.pipeline_if_img2img.IFImg2ImgPipeline.preprocess_image 复制
        def preprocess_image(self, image: PIL.Image.Image) -> torch.Tensor:
            # 如果输入不是列表，则将其转换为列表
            if not isinstance(image, list):
                image = [image]
    
            # 定义将 numpy 数组转换为 PyTorch 张量的内部函数
            def numpy_to_pt(images):
                if images.ndim == 3:
                    images = images[..., None]  # 如果是 3 维，增加一个维度
    
                images = torch.from_numpy(images.transpose(0, 3, 1, 2))  # 转换并调整维度
                return images
    
            # 如果第一个元素是 PIL 图像
            if isinstance(image[0], PIL.Image.Image):
                new_image = []  # 初始化新图像列表
    
                # 遍历图像列表进行处理
                for image_ in image:
                    image_ = image_.convert("RGB")  # 转换为 RGB 格式
                    image_ = resize(image_, self.unet.config.sample_size)  # 调整大小
                    image_ = np.array(image_)  # 转换为 numpy 数组
                    image_ = image_.astype(np.float32)  # 转换为浮点型
                    image_ = image_ / 127.5 - 1  # 归一化到 [-1, 1]
                    new_image.append(image_)  # 添加到新图像列表
    
                image = new_image  # 更新图像为新列表
    
                image = np.stack(image, axis=0)  # 将列表转换为 numpy 数组
                image = numpy_to_pt(image)  # 转换为 PyTorch 张量
    
            # 如果第一个元素是 numpy 数组
            elif isinstance(image[0], np.ndarray):
                # 根据维度进行拼接或堆叠
                image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
                image = numpy_to_pt(image)  # 转换为 PyTorch 张量
    
            # 如果第一个元素是 PyTorch 张量
            elif isinstance(image[0], torch.Tensor):
                # 根据维度进行拼接或堆叠
                image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
    
            # 返回处理后的图像
            return image
    # 处理掩码图像，返回一个张量
        def preprocess_mask_image(self, mask_image) -> torch.Tensor:
            # 检查输入是否为列表，如果不是则转换为单元素列表
            if not isinstance(mask_image, list):
                mask_image = [mask_image]
    
            # 检查第一个元素是否为张量
            if isinstance(mask_image[0], torch.Tensor):
                # 如果是四维张量则在第0轴上拼接，否则堆叠
                mask_image = torch.cat(mask_image, axis=0) if mask_image[0].ndim == 4 else torch.stack(mask_image, axis=0)
    
                # 如果是二维张量，添加批次和通道维度
                if mask_image.ndim == 2:
                    mask_image = mask_image.unsqueeze(0).unsqueeze(0)
                # 如果是三维张量且批次大小为1，添加批次维度
                elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
                    mask_image = mask_image.unsqueeze(0)
                # 如果是三维张量且批次大小不为1，添加通道维度
                elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
                    mask_image = mask_image.unsqueeze(1)
    
                # 将小于0.5的值设为0
                mask_image[mask_image < 0.5] = 0
                # 将大于等于0.5的值设为1
                mask_image[mask_image >= 0.5] = 1
    
            # 检查第一个元素是否为PIL图像
            elif isinstance(mask_image[0], PIL.Image.Image):
                new_mask_image = []
    
                # 遍历每个掩码图像进行处理
                for mask_image_ in mask_image:
                    # 将图像转换为灰度模式
                    mask_image_ = mask_image_.convert("L")
                    # 调整图像大小
                    mask_image_ = resize(mask_image_, self.unet.config.sample_size)
                    # 转换为numpy数组
                    mask_image_ = np.array(mask_image_)
                    # 添加批次和通道维度
                    mask_image_ = mask_image_[None, None, :]
                    new_mask_image.append(mask_image_)
    
                # 将所有处理后的掩码合并
                mask_image = new_mask_image
    
                # 在第0轴上拼接所有掩码图像
                mask_image = np.concatenate(mask_image, axis=0)
                # 将像素值缩放到[0, 1]
                mask_image = mask_image.astype(np.float32) / 255.0
                # 将小于0.5的值设为0
                mask_image[mask_image < 0.5] = 0
                # 将大于等于0.5的值设为1
                mask_image[mask_image >= 0.5] = 1
                # 转换为PyTorch张量
                mask_image = torch.from_numpy(mask_image)
    
            # 检查第一个元素是否为numpy数组
            elif isinstance(mask_image[0], np.ndarray):
                # 在第0轴上拼接所有掩码图像
                mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
    
                # 将小于0.5的值设为0
                mask_image[mask_image < 0.5] = 0
                # 将大于等于0.5的值设为1
                mask_image[mask_image >= 0.5] = 1
                # 转换为PyTorch张量
                mask_image = torch.from_numpy(mask_image)
    
            # 返回处理后的掩码图像
            return mask_image
    
        # 从diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps复制的函数
        def get_timesteps(self, num_inference_steps, strength):
            # 根据初始化时间步长计算原始时间步长
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            # 计算时间步长的起始索引
            t_start = max(num_inference_steps - init_timestep, 0)
            # 获取调度器中相应的时间步长
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            # 如果调度器具有设置起始索引的方法，则调用
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
    
            # 返回时间步长和剩余的推理步骤
            return timesteps, num_inference_steps - t_start
    
        # 准备中间图像的函数
        def prepare_intermediate_images(
            self, image, timestep, batch_size, num_images_per_prompt, dtype, device, mask_image, generator=None
    ):
        # 获取输入图像的批大小、通道数、高度和宽度
        image_batch_size, channels, height, width = image.shape

        # 根据每个提示所需生成的图像数量调整批大小
        batch_size = batch_size * num_images_per_prompt

        # 定义图像的形状，包括调整后的批大小和通道、高度、宽度
        shape = (batch_size, channels, height, width)

        # 检查生成器是否为列表且其长度与批大小不匹配
        if isinstance(generator, list) and len(generator) != batch_size:
            # 如果生成器的长度不等于批大小，抛出值错误
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 生成随机噪声张量，形状与输入图像匹配
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 重复输入图像，生成所需的图像数量
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
        # 将噪声添加到图像上，得到带噪声的图像
        noised_image = self.scheduler.add_noise(image, noise, timestep)

        # 根据掩膜图像合成原始图像和带噪声的图像
        image = (1 - mask_image) * image + mask_image * noised_image

        # 返回处理后的图像
        return image

    # 禁用梯度计算以节省内存和计算资源
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 提示字符串或字符串列表，默认值为 None
        prompt: Union[str, List[str]] = None,
        # 输入图像，可以是多种格式，包括 PIL 图像、张量、数组等，默认值为 None
        image: Union[
            PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
        ] = None,
        # 掩膜图像，可以是多种格式，默认值为 None
        mask_image: Union[
            PIL.Image.Image, torch.Tensor, np.ndarray, List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]
        ] = None,
        # 强度参数，影响图像生成的程度，默认值为 1.0
        strength: float = 1.0,
        # 推理步骤数量，控制生成过程的迭代次数，默认值为 50
        num_inference_steps: int = 50,
        # 时间步列表，控制噪声添加的时刻，默认值为 None
        timesteps: List[int] = None,
        # 引导比例，影响生成图像与提示的相关性，默认值为 7.0
        guidance_scale: float = 7.0,
        # 负提示，指定不希望出现的提示，可以是字符串或字符串列表，默认值为 None
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示生成的图像数量，默认值为 1
        num_images_per_prompt: Optional[int] = 1,
        # η 参数，控制随机性，默认值为 0.0
        eta: float = 0.0,
        # 生成器，可以是单个生成器或生成器列表，默认值为 None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 提示嵌入，预先计算的提示表示，默认值为 None
        prompt_embeds: Optional[torch.Tensor] = None,
        # 负提示嵌入，预先计算的负提示表示，默认值为 None
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # 输出类型，指定返回的图像格式，默认值为 "pil"
        output_type: Optional[str] = "pil",
        # 是否返回字典形式的结果，默认值为 True
        return_dict: bool = True,
        # 回调函数，接受当前步骤和生成图像的函数，默认值为 None
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调步骤，控制回调函数调用的频率，默认值为 1
        callback_steps: int = 1,
        # 是否清理提示，默认值为 True
        clean_caption: bool = True,
        # 交叉注意力的额外参数，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
```