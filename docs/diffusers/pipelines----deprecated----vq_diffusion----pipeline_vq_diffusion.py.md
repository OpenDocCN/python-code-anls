# `.\diffusers\pipelines\deprecated\vq_diffusion\pipeline_vq_diffusion.py`

```py
# 版权所有 2024 Microsoft 和 The HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按“原样”基础分发的，没有任何明示或暗示的担保或条件。
# 有关许可证下特定语言的权限和限制，请参阅许可证。

# 从 typing 模块导入所需的类型
from typing import Callable, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 相关模型和分词器
from transformers import CLIPTextModel, CLIPTokenizer

# 从本地模块导入配置和模型相关的类
from ....configuration_utils import ConfigMixin, register_to_config
from ....models import ModelMixin, Transformer2DModel, VQModel
from ....schedulers import VQDiffusionScheduler
from ....utils import logging
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个 logger 实例，用于记录日志信息，禁用 pylint 的命名检查
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):
    """
    存储用于分类器自由采样的学习文本嵌入的实用类
    """

    @register_to_config
    # 初始化方法，接受可学习标志和可选的隐藏层大小及长度
    def __init__(self, learnable: bool, hidden_size: Optional[int] = None, length: Optional[int] = None):
        # 调用父类初始化方法
        super().__init__()

        # 设置可学习标志
        self.learnable = learnable

        # 如果可学习，检查隐藏层大小和长度是否被设置
        if self.learnable:
            # 确保在可学习时隐藏层大小不为空
            assert hidden_size is not None, "learnable=True requires `hidden_size` to be set"
            # 确保在可学习时长度不为空
            assert length is not None, "learnable=True requires `length` to be set"

            # 创建一个形状为 (length, hidden_size) 的全零张量作为嵌入
            embeddings = torch.zeros(length, hidden_size)
        else:
            # 如果不可学习，嵌入设为 None
            embeddings = None

        # 将嵌入转换为可学习参数
        self.embeddings = torch.nn.Parameter(embeddings)

class VQDiffusionPipeline(DiffusionPipeline):
    r"""
    使用 VQ Diffusion 进行文本到图像生成的管道。

    此模型继承自 [`DiffusionPipeline`]。查看超类文档以获取所有管道实现的通用方法
    （下载、保存、在特定设备上运行等）。

    参数：
        vqvae ([`VQModel`]):
            用于编码和解码图像到潜在表示的向量量化变分自编码器（VAE）模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器（[clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)）。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于分词的 `CLIPTokenizer`。
        transformer ([`Transformer2DModel`]):
            用于去噪编码图像潜在的条件 `Transformer2DModel`。
        scheduler ([`VQDiffusionScheduler`]):
            用于与 `transformer` 一起去噪编码图像潜在的调度器。
    """

    # 定义类属性 vqvae，类型为 VQModel
    vqvae: VQModel
    # 定义类属性 text_encoder，类型为 CLIPTextModel
    text_encoder: CLIPTextModel
    # 定义类属性 tokenizer，类型为 CLIPTokenizer
    tokenizer: CLIPTokenizer
    # 定义一个包含多个模型和调度器的类
        transformer: Transformer2DModel  # 2D 变换器模型
        learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings  # 学习的分类器自由采样嵌入
        scheduler: VQDiffusionScheduler  # VQ 扩散调度器
    
        # 初始化方法，接受多个模型和调度器作为参数
        def __init__(
            self,
            vqvae: VQModel,  # VQ-VAE 模型
            text_encoder: CLIPTextModel,  # 文本编码器模型
            tokenizer: CLIPTokenizer,  # 分词器
            transformer: Transformer2DModel,  # 2D 变换器模型
            scheduler: VQDiffusionScheduler,  # VQ 扩散调度器
            learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings,  # 学习的分类器自由采样嵌入
        ):
            super().__init__()  # 调用父类的初始化方法
    
            # 注册多个模块，使其在模型中可用
            self.register_modules(
                vqvae=vqvae,  # 注册 VQ-VAE 模型
                transformer=transformer,  # 注册 2D 变换器
                text_encoder=text_encoder,  # 注册文本编码器
                tokenizer=tokenizer,  # 注册分词器
                scheduler=scheduler,  # 注册调度器
                learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,  # 注册自由采样嵌入
            )
    
        # 禁用梯度计算，优化内存使用
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]],  # 输入的提示，可以是字符串或字符串列表
            num_inference_steps: int = 100,  # 推理步骤数，默认为100
            guidance_scale: float = 5.0,  # 引导比例，默认为5.0
            truncation_rate: float = 1.0,  # 截断比例，默认为1.0
            num_images_per_prompt: int = 1,  # 每个提示生成的图像数量，默认为1
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器，默认为None
            latents: Optional[torch.Tensor] = None,  # 潜在变量，默认为None
            output_type: Optional[str] = "pil",  # 输出类型，默认为"pil"
            return_dict: bool = True,  # 是否返回字典，默认为True
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,  # 回调函数，默认为None
            callback_steps: int = 1,  # 回调步骤，默认为1
        # 定义一个截断方法，用于调整概率分布
        def truncate(self, log_p_x_0: torch.Tensor, truncation_rate: float) -> torch.Tensor:
            """
            截断 `log_p_x_0`，使每列向量的总累积概率等于 `truncation_rate`
            低于该比例的概率将被设置为零。
            """
            # 对 log 概率进行排序，并获取索引
            sorted_log_p_x_0, indices = torch.sort(log_p_x_0, 1, descending=True)
            # 计算排序后概率的指数值
            sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
            # 创建掩码，标记哪些概率的累积和低于截断比例
            keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate
    
            # 确保至少保留最大概率
            all_true = torch.full_like(keep_mask[:, 0:1, :], True)  # 创建全为 True 的张量
            keep_mask = torch.cat((all_true, keep_mask), dim=1)  # 在掩码前添加全 True 的行
            keep_mask = keep_mask[:, :-1, :]  # 移除最后一列
    
            # 根据原始索引排序掩码
            keep_mask = keep_mask.gather(1, indices.argsort(1))
    
            rv = log_p_x_0.clone()  # 复制输入的 log 概率
    
            rv[~keep_mask] = -torch.inf  # 将未保留的概率设置为负无穷（即 log(0)）
    
            return rv  # 返回调整后的概率
```