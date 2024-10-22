# `.\diffusers\pipelines\controlnet_hunyuandit\pipeline_hunyuandit_controlnet.py`

```py
# 版权声明，指明文件的版权归 HunyuanDiT 和 HuggingFace 团队所有
# 本文件在 Apache 2.0 许可证下授权使用
# 除非遵循许可证，否则不能使用此文件
# 许可证的副本可以在以下网址获取
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律规定或书面协议另有约定，否则软件在"按现状"基础上提供，不附带任何明示或暗示的保证
# 查看许可证以了解特定语言的权限和限制

# 导入用于获取函数信息的 inspect 模块
import inspect
# 导入类型提示所需的类型
from typing import Callable, Dict, List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 从 transformers 库导入相关模型和分词器
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

# 从 diffusers 库导入 StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

# 导入多管道回调类
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# 导入图像处理类
from ...image_processor import PipelineImageInput, VaeImageProcessor
# 导入自动编码器和模型
from ...models import AutoencoderKL, HunyuanDiT2DControlNetModel, HunyuanDiT2DModel, HunyuanDiT2DMultiControlNetModel
# 导入 2D 旋转位置嵌入函数
from ...models.embeddings import get_2d_rotary_pos_embed
# 导入稳定扩散安全检查器
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 导入扩散调度器
from ...schedulers import DDPMScheduler
# 导入实用工具函数
from ...utils import (
    is_torch_xla_available,  # 检查是否可用 XLA
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 替换示例文档字符串的工具
)
# 导入 PyTorch 相关的随机张量函数
from ...utils.torch_utils import randn_tensor
# 导入扩散管道工具类
from ..pipeline_utils import DiffusionPipeline

# 检查是否可用 XLA，并根据结果导入相应模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # 导入 XLA 核心模型

    XLA_AVAILABLE = True  # 设置 XLA 可用标志为 True
else:
    XLA_AVAILABLE = False  # 设置 XLA 可用标志为 False

# 创建一个日志记录器实例，记录当前模块的日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，用于说明使用方法
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
    # 示例代码展示如何使用 HunyuanDiT 进行图像生成
        Examples:
            ```py
            # 从 diffusers 库导入所需的模型和管道
            from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
            # 导入 PyTorch 库
            import torch
    
            # 从预训练模型加载 HunyuanDiT2DControlNetModel，并指定数据类型为 float16
            controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", torch_dtype=torch.float16
            )
    
            # 从预训练模型加载 HunyuanDiTControlNetPipeline，传入 controlnet 和数据类型
            pipe = HunyuanDiTControlNetPipeline.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16
            )
            # 将管道移动到 CUDA 设备以加速处理
            pipe.to("cuda")
    
            # 从 diffusers.utils 导入加载图像的工具
            from diffusers.utils import load_image
    
            # 从指定 URL 加载条件图像
            cond_image = load_image(
                "https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true"
            )
    
            ## HunyuanDiT 支持英语和中文提示，因此也可以使用英文提示
            # 定义图像生成的提示内容，描述夜晚的场景
            prompt = "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
            # prompt="At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
            # 使用提示、图像尺寸、条件图像和推理步骤生成图像，并获取生成的第一张图像
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                control_image=cond_image,
                num_inference_steps=50,
            ).images[0]
            ```  
"""
# 文档字符串，通常用于描述模块或类的功能
"""

# 定义一个标准宽高比的 NumPy 数组
STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)

# 定义一个标准尺寸的列表，每个比例对应不同的宽高组合
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]

# 根据标准尺寸计算每个形状的面积，并将结果存储在 NumPy 数组中
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]

# 定义一个支持的尺寸列表，包含不同的宽高组合
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]

# 定义一个函数，用于将目标宽高映射到标准形状
def map_to_standard_shapes(target_width, target_height):
    # 计算目标宽高比
    target_ratio = target_width / target_height
    # 找到与目标宽高比最接近的标准宽高比的索引
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    # 找到与目标面积最接近的标准形状的索引
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    # 获取对应的标准宽和高
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    # 返回标准宽和高
    return width, height

# 定义一个函数，用于计算源图像的缩放裁剪区域以适应目标大小
def get_resize_crop_region_for_grid(src, tgt_size):
    # 获取目标尺寸的高度和宽度
    th = tw = tgt_size
    # 获取源图像的高度和宽度
    h, w = src

    # 计算源图像的宽高比
    r = h / w

    # 根据宽高比决定缩放方式
    # 如果高度大于宽度
    if r > 1:
        # 将目标高度作为缩放高度
        resize_height = th
        # 根据高度缩放计算对应的宽度
        resize_width = int(round(th / h * w))
    else:
        # 否则，将目标宽度作为缩放宽度
        resize_width = tw
        # 根据宽度缩放计算对应的高度
        resize_height = int(round(tw / w * h))

    # 计算裁剪区域的顶部和左边位置
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    # 返回裁剪区域的起始和结束坐标
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg 复制的函数
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    根据 `guidance_rescale` 对 `noise_cfg` 进行重新缩放。基于论文[Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)中的发现。见第3.4节
    """
    # 计算噪声预测文本的标准差
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # 计算噪声配置的标准差
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 重新缩放来自引导的结果（修复过度曝光问题）
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # 按照引导缩放因子与原始引导结果进行混合，以避免生成“单调”的图像
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    # 返回重新缩放后的噪声配置
    return noise_cfg

# 定义 HunyuanDiT 控制网络管道类，继承自 DiffusionPipeline
class HunyuanDiTControlNetPipeline(DiffusionPipeline):
    r"""
    使用 HunyuanDiT 进行英语/中文到图像生成的管道。

    该模型继承自 [`DiffusionPipeline`]. 请查看超类文档以获取库为所有管道实现的通用方法
    （例如下载或保存，在特定设备上运行等）。

    HunyuanDiT 使用两个文本编码器：[mT5](https://huggingface.co/google/mt5-base) 和 [双语 CLIP](自行微调)
    """
    # 参数说明
    Args:
        vae ([`AutoencoderKL`]):  # 变分自编码器模型，用于将图像编码和解码为潜在表示，这里使用'sdxl-vae-fp16-fix'
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. We use
            `sdxl-vae-fp16-fix`.
        text_encoder (Optional[`~transformers.BertModel`, `~transformers.CLIPTextModel`]):  # 冻结的文本编码器，使用CLIP模型
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)). 
            HunyuanDiT uses a fine-tuned [bilingual CLIP].
        tokenizer (Optional[`~transformers.BertTokenizer`, `~transformers.CLIPTokenizer`]):  # 文本标记化器，可以是BertTokenizer或CLIPTokenizer
            A `BertTokenizer` or `CLIPTokenizer` to tokenize text.
        transformer ([`HunyuanDiT2DModel`]):  # HunyuanDiT模型，由腾讯Hunyuan设计
            The HunyuanDiT model designed by Tencent Hunyuan.
        text_encoder_2 (`T5EncoderModel`):  # mT5嵌入模型，特别是't5-v1_1-xxl'
            The mT5 embedder. Specifically, it is 't5-v1_1-xxl'.
        tokenizer_2 (`MT5Tokenizer`):  # mT5嵌入模型的标记化器
            The tokenizer for the mT5 embedder.
        scheduler ([`DDPMScheduler`]):  # 调度器，用于与HunyuanDiT结合，去噪编码的图像潜在表示
            A scheduler to be used in combination with HunyuanDiT to denoise the encoded image latents.
        controlnet ([`HunyuanDiT2DControlNetModel`] or `List[HunyuanDiT2DControlNetModel]` or [`HunyuanDiT2DControlNetModel`]):  # 提供额外的条件信息以辅助去噪过程
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    # 定义模型在CPU上卸载的顺序
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    # 可选组件列表，可能会在初始化中使用
    _optional_components = [
        "safety_checker",  # 安全检查器
        "feature_extractor",  # 特征提取器
        "text_encoder_2",  # 第二个文本编码器
        "tokenizer_2",  # 第二个标记化器
        "text_encoder",  # 第一个文本编码器
        "tokenizer",  # 第一个标记化器
    ]
    # 从CPU卸载中排除的组件
    _exclude_from_cpu_offload = ["safety_checker"]  # 不允许卸载安全检查器
    # 回调张量输入的列表，用于传递给模型
    _callback_tensor_inputs = [
        "latents",  # 潜在变量
        "prompt_embeds",  # 提示的嵌入表示
        "negative_prompt_embeds",  # 负提示的嵌入表示
        "prompt_embeds_2",  # 第二个提示的嵌入表示
        "negative_prompt_embeds_2",  # 第二个负提示的嵌入表示
    ]

    # 初始化方法定义，接收多个参数以构造模型
    def __init__(
        self,
        vae: AutoencoderKL,  # 变分自编码器模型
        text_encoder: BertModel,  # 文本编码器
        tokenizer: BertTokenizer,  # 文本标记化器
        transformer: HunyuanDiT2DModel,  # HunyuanDiT模型
        scheduler: DDPMScheduler,  # 调度器
        safety_checker: StableDiffusionSafetyChecker,  # 安全检查器
        feature_extractor: CLIPImageProcessor,  # 特征提取器
        controlnet: Union[  # 控制网络，可以是单个或多个模型
            HunyuanDiT2DControlNetModel,
            List[HunyuanDiT2DControlNetModel],
            Tuple[HunyuanDiT2DControlNetModel],
            HunyuanDiT2DMultiControlNetModel,
        ],
        text_encoder_2=T5EncoderModel,  # 第二个文本编码器，默认使用T5模型
        tokenizer_2=MT5Tokenizer,  # 第二个标记化器，默认使用MT5标记化器
        requires_safety_checker: bool = True,  # 是否需要安全检查器，默认是True
    # 初始化父类
    ):
        super().__init__()

        # 注册多个模块，提供必要的组件以供使用
        self.register_modules(
            vae=vae,  # 注册变分自编码器
            text_encoder=text_encoder,  # 注册文本编码器
            tokenizer=tokenizer,  # 注册分词器
            tokenizer_2=tokenizer_2,  # 注册第二个分词器
            transformer=transformer,  # 注册变换器
            scheduler=scheduler,  # 注册调度器
            safety_checker=safety_checker,  # 注册安全检查器
            feature_extractor=feature_extractor,  # 注册特征提取器
            text_encoder_2=text_encoder_2,  # 注册第二个文本编码器
            controlnet=controlnet,  # 注册控制网络
        )

        # 检查安全检查器是否为 None 并且需要使用安全检查器
        if safety_checker is None and requires_safety_checker:
            # 记录警告信息，提醒用户禁用安全检查器的后果
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        # 检查安全检查器不为 None 且特征提取器为 None
        if safety_checker is not None and feature_extractor is None:
            # 抛出错误，提醒用户必须定义特征提取器
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # 计算 VAE 的缩放因子，如果存在 VAE 配置则使用其通道数量，否则默认为 8
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # 初始化图像处理器，传入 VAE 缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 注册到配置中，指明是否需要安全检查器
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # 设置默认样本大小，根据变换器配置或默认为 128
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    # 从其他模块复制的方法，用于编码提示
    def encode_prompt(
        self,
        prompt: str,  # 输入的提示文本
        device: torch.device = None,  # 设备参数，指定在哪个设备上处理
        dtype: torch.dtype = None,  # 数据类型参数，指定张量的数据类型
        num_images_per_prompt: int = 1,  # 每个提示生成的图像数量
        do_classifier_free_guidance: bool = True,  # 是否执行无分类器的引导
        negative_prompt: Optional[str] = None,  # 可选的负面提示文本
        prompt_embeds: Optional[torch.Tensor] = None,  # 可选的提示嵌入张量
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 可选的负面提示嵌入张量
        prompt_attention_mask: Optional[torch.Tensor] = None,  # 可选的提示注意力掩码
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # 可选的负面提示注意力掩码
        max_sequence_length: Optional[int] = None,  # 可选的最大序列长度
        text_encoder_index: int = 0,  # 文本编码器索引，默认值为 0
    # 从其他模块复制的方法，用于运行安全检查器
    # 定义运行安全检查器的方法，接收图像、设备和数据类型作为参数
    def run_safety_checker(self, image, device, dtype):
        # 如果安全检查器未定义，设置无敏感内容标志为 None
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            # 如果输入图像是张量，则后处理为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 如果输入图像不是张量，则将其转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 使用特征提取器处理输入图像并将其转换为指定设备上的张量
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 调用安全检查器，返回处理后的图像和无敏感内容标志
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        # 返回处理后的图像和无敏感内容标志
        return image, has_nsfw_concept
    
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs 复制
        def prepare_extra_step_kwargs(self, generator, eta):
            # 为调度器步骤准备额外参数，因为不是所有调度器的参数签名相同
            # eta（η）仅在 DDIMScheduler 中使用，其他调度器将忽略
            # eta 在 DDIM 论文中的对应关系：https://arxiv.org/abs/2010.02502
            # 其值应在 [0, 1] 之间
    
            # 检查调度器的步骤是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            # 如果接受 eta，则将其添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果接受 generator，则将其添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回额外参数字典
            return extra_step_kwargs
    
        # 从 diffusers.pipelines.hunyuandit.pipeline_hunyuandit.HunyuanDiTPipeline.check_inputs 复制
        def check_inputs(
            self,
            prompt,
            height,
            width,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            prompt_embeds_2=None,
            negative_prompt_embeds_2=None,
            prompt_attention_mask_2=None,
            negative_prompt_attention_mask_2=None,
            callback_on_step_end_tensor_inputs=None,
        # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制
    # 准备潜在变量的函数，接收多个参数以配置潜在变量的形状和生成方式
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # 根据批大小、通道数、高度和宽度计算潜在变量的形状
            shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(width) // self.vae_scale_factor,
            )
            # 检查生成器列表的长度是否与批大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不匹配，抛出一个值错误
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果潜在变量为 None，则生成随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 如果已提供潜在变量，将其移动到指定设备
                latents = latents.to(device)
    
            # 按调度器要求的标准差缩放初始噪声
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 准备图像的函数，从外部调用
        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            # 检查图像是否为张量，如果是则不处理
            if isinstance(image, torch.Tensor):
                pass
            else:
                # 否则对图像进行预处理，调整为指定的高度和宽度
                image = self.image_processor.preprocess(image, height=height, width=width)
    
            # 获取图像的批大小
            image_batch_size = image.shape[0]
    
            # 如果图像批大小为1，则重复次数为批大小
            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # 否则图像批大小与提示批大小相同
                repeat_by = num_images_per_prompt
    
            # 沿着维度0重复图像
            image = image.repeat_interleave(repeat_by, dim=0)
    
            # 将图像移动到指定设备，并转换为指定数据类型
            image = image.to(device=device, dtype=dtype)
    
            # 如果启用了无分类器自由引导，并且未启用猜测模式，则将图像复制两次
            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)
    
            # 返回处理后的图像
            return image
    
        # 获取指导比例的属性
        @property
        def guidance_scale(self):
            # 返回当前的指导比例
            return self._guidance_scale
    
        # 获取指导重标定的属性
        @property
        def guidance_rescale(self):
            # 返回当前的指导重标定值
            return self._guidance_rescale
    
        # 此属性定义了类似于论文中指导权重的定义
        @property
        def do_classifier_free_guidance(self):
            # 如果指导比例大于1，则启用无分类器自由引导
            return self._guidance_scale > 1
    
        # 获取时间步数的属性
        @property
        def num_timesteps(self):
            # 返回当前的时间步数
            return self._num_timesteps
    
        # 获取中断状态的属性
        @property
        def interrupt(self):
            # 返回当前中断状态
            return self._interrupt
    
        # 在不计算梯度的情况下运行，替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法
        def __call__(
            # 提示内容，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 输出图像的高度
            height: Optional[int] = None,
            # 输出图像的宽度
            width: Optional[int] = None,
            # 推理步骤的数量，默认为 50
            num_inference_steps: Optional[int] = 50,
            # 引导比例，默认为 5.0
            guidance_scale: Optional[float] = 5.0,
            # 控制图像输入，默认为 None
            control_image: PipelineImageInput = None,
            # 控制网条件比例，可以是单一值或值列表，默认为 1.0
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            # 负提示内容，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: Optional[int] = 1,
            # 用于生成的随机性，默认为 0.0
            eta: Optional[float] = 0.0,
            # 随机数生成器，可以是单个或列表，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 提示的嵌入，默认为 None
            prompt_embeds: Optional[torch.Tensor] = None,
            # 第二组提示的嵌入，默认为 None
            prompt_embeds_2: Optional[torch.Tensor] = None,
            # 负提示的嵌入，默认为 None
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 第二组负提示的嵌入，默认为 None
            negative_prompt_embeds_2: Optional[torch.Tensor] = None,
            # 提示的注意力掩码，默认为 None
            prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二组提示的注意力掩码，默认为 None
            prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 负提示的注意力掩码，默认为 None
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            # 第二组负提示的注意力掩码，默认为 None
            negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式，默认为 True
            return_dict: bool = True,
            # 在步骤结束时的回调函数
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            # 回调时的张量输入列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 引导重标定，默认为 0.0
            guidance_rescale: float = 0.0,
            # 原始图像大小，默认为 (1024, 1024)
            original_size: Optional[Tuple[int, int]] = (1024, 1024),
            # 目标图像大小，默认为 None
            target_size: Optional[Tuple[int, int]] = None,
            # 裁剪坐标，默认为 (0, 0)
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            # 是否使用分辨率分箱，默认为 True
            use_resolution_binning: bool = True,
```