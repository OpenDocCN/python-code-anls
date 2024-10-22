# `.\diffusers\pipelines\auto_pipeline.py`

```py
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权声明，指明代码版权所有者为 HuggingFace Inc. 团队
# Copyright 2024 The HuggingFace Inc. team.
#
# 指明该文件遵循 Apache License 2.0，用户需遵循该许可证的条款
# Licensed under the Apache License, Version 2.0 (the "License");
# 用户不得在不遵循许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 用户可以在以下地址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果没有适用的法律规定或书面协议，软件在 "AS IS" 基础上分发，不提供任何保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 指明许可证的详细信息，包括权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 collections 模块导入有序字典类
from collections import OrderedDict

# 从 huggingface_hub.utils 导入参数验证工具
from huggingface_hub.utils import validate_hf_hub_args

# 从 configuration_utils 导入配置混合器
from ..configuration_utils import ConfigMixin
# 从 utils 导入句子分割可用性检查工具
from ..utils import is_sentencepiece_available
# 从 aura_flow 模块导入 AuraFlow 管道类
from .aura_flow import AuraFlowPipeline
# 从 controlnet 模块导入多种稳定扩散控制网络管道
from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)
# 从 deepfloyd_if 模块导入多种图像处理管道
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
# 从 flux 模块导入 Flux 管道类
from .flux import FluxPipeline
# 从 hunyuandit 模块导入 HunyuanDiT 管道类
from .hunyuandit import HunyuanDiTPipeline
# 从 kandinsky 模块导入多种 Kandinsky 管道
from .kandinsky import (
    KandinskyCombinedPipeline,
    KandinskyImg2ImgCombinedPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintCombinedPipeline,
    KandinskyInpaintPipeline,
    KandinskyPipeline,
)
# 从 kandinsky2_2 模块导入多种 Kandinsky V2.2 管道
from .kandinsky2_2 import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintCombinedPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
)
# 从 kandinsky3 模块导入 Kandinsky 3 的图像处理管道
from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
# 从 latent_consistency_models 模块导入潜在一致性模型管道
from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
# 从 pag 模块导入多种 PAG 管道
from .pag import (
    HunyuanDiTPAGPipeline,
    PixArtSigmaPAGPipeline,
    StableDiffusion3PAGPipeline,
    StableDiffusionControlNetPAGPipeline,
    StableDiffusionPAGPipeline,
    StableDiffusionXLControlNetPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusionXLPAGInpaintPipeline,
    StableDiffusionXLPAGPipeline,
)
# 从 pixart_alpha 模块导入 PixArt Alpha 和 Sigma 管道
from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
# 从 stable_cascade 模块导入多种稳定级联管道
from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline
# 从 stable_diffusion 模块导入多种稳定扩散管道
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
# 从 stable_diffusion_3 模块导入多种稳定扩散 3 管道
from .stable_diffusion_3 import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
)
# 从 stable_diffusion_xl 模块导入多种稳定扩散 XL 管道
from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
# 从 wuerstchen 模块导入多种 Wuerstchen 管道
from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline


# 创建一个有序字典，用于映射自动文本到图像的管道
AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    # 创建一个包含模型名称和对应管道类的列表
        [
            # 定义稳定扩散模型及其管道
            ("stable-diffusion", StableDiffusionPipeline),
            # 定义稳定扩散 XL 模型及其管道
            ("stable-diffusion-xl", StableDiffusionXLPipeline),
            # 定义稳定扩散 3 模型及其管道
            ("stable-diffusion-3", StableDiffusion3Pipeline),
            # 定义稳定扩散 3 PAG 模型及其管道
            ("stable-diffusion-3-pag", StableDiffusion3PAGPipeline),
            # 定义 IF 模型及其管道
            ("if", IFPipeline),
            # 定义 Hunyuan 模型及其管道
            ("hunyuan", HunyuanDiTPipeline),
            # 定义 Hunyuan PAG 模型及其管道
            ("hunyuan-pag", HunyuanDiTPAGPipeline),
            # 定义 Kandinsky 组合模型及其管道
            ("kandinsky", KandinskyCombinedPipeline),
            # 定义 Kandinsky 2.2 组合模型及其管道
            ("kandinsky22", KandinskyV22CombinedPipeline),
            # 定义 Kandinsky 3 模型及其管道
            ("kandinsky3", Kandinsky3Pipeline),
            # 定义稳定扩散控制网模型及其管道
            ("stable-diffusion-controlnet", StableDiffusionControlNetPipeline),
            # 定义稳定扩散 XL 控制网模型及其管道
            ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetPipeline),
            # 定义 Wuerstchen 组合模型及其管道
            ("wuerstchen", WuerstchenCombinedPipeline),
            # 定义稳定级联组合模型及其管道
            ("cascade", StableCascadeCombinedPipeline),
            # 定义潜在一致性模型及其管道
            ("lcm", LatentConsistencyModelPipeline),
            # 定义 PixArt Alpha 模型及其管道
            ("pixart-alpha", PixArtAlphaPipeline),
            # 定义 PixArt Sigma 模型及其管道
            ("pixart-sigma", PixArtSigmaPipeline),
            # 定义稳定扩散 PAG 模型及其管道
            ("stable-diffusion-pag", StableDiffusionPAGPipeline),
            # 定义稳定扩散控制网 PAG 模型及其管道
            ("stable-diffusion-controlnet-pag", StableDiffusionControlNetPAGPipeline),
            # 定义稳定扩散 XL PAG 模型及其管道
            ("stable-diffusion-xl-pag", StableDiffusionXLPAGPipeline),
            # 定义稳定扩散 XL 控制网 PAG 模型及其管道
            ("stable-diffusion-xl-controlnet-pag", StableDiffusionXLControlNetPAGPipeline),
            # 定义 PixArt Sigma PAG 模型及其管道
            ("pixart-sigma-pag", PixArtSigmaPAGPipeline),
            # 定义 AuraFlow 模型及其管道
            ("auraflow", AuraFlowPipeline),
            # 定义 Flux 模型及其管道
            ("flux", FluxPipeline),
        ]
# 定义用于图像到图像转换的管道映射，使用有序字典确保顺序
AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        # 映射每种图像到图像转换管道
        ("stable-diffusion", StableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", StableDiffusionXLImg2ImgPipeline),
        ("stable-diffusion-3", StableDiffusion3Img2ImgPipeline),
        ("if", IFImg2ImgPipeline),
        ("kandinsky", KandinskyImg2ImgCombinedPipeline),
        ("kandinsky22", KandinskyV22Img2ImgCombinedPipeline),
        ("kandinsky3", Kandinsky3Img2ImgPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetImg2ImgPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetImg2ImgPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGImg2ImgPipeline),
        ("lcm", LatentConsistencyModelImg2ImgPipeline),
    ]
)

# 定义用于图像修复的管道映射，同样使用有序字典
AUTO_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        # 映射每种图像修复管道
        ("stable-diffusion", StableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", StableDiffusionXLInpaintPipeline),
        ("stable-diffusion-3", StableDiffusion3InpaintPipeline),
        ("if", IFInpaintingPipeline),
        ("kandinsky", KandinskyInpaintCombinedPipeline),
        ("kandinsky22", KandinskyV22InpaintCombinedPipeline),
        ("stable-diffusion-controlnet", StableDiffusionControlNetInpaintPipeline),
        ("stable-diffusion-xl-controlnet", StableDiffusionXLControlNetInpaintPipeline),
        ("stable-diffusion-xl-pag", StableDiffusionXLPAGInpaintPipeline),
    ]
)

# 定义用于文本到图像解码器的管道映射
_AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        # 映射每种文本到图像解码器管道
        ("kandinsky", KandinskyPipeline),
        ("kandinsky22", KandinskyV22Pipeline),
        ("wuerstchen", WuerstchenDecoderPipeline),
        ("cascade", StableCascadeDecoderPipeline),
    ]
)

# 定义用于图像到图像解码器的管道映射
_AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        # 映射每种图像到图像解码器管道
        ("kandinsky", KandinskyImg2ImgPipeline),
        ("kandinsky22", KandinskyV22Img2ImgPipeline),
    ]
)

# 定义用于图像修复解码器的管道映射
_AUTO_INPAINT_DECODER_PIPELINES_MAPPING = OrderedDict(
    [
        # 映射每种图像修复解码器管道
        ("kandinsky", KandinskyInpaintPipeline),
        ("kandinsky22", KandinskyV22InpaintPipeline),
    ]
)

# 检查是否可用 sentencepiece 库
if is_sentencepiece_available():
    # 从模块中导入所需的管道类
    from .kolors import KolorsPipeline
    from .pag import KolorsPAGPipeline

    # 将 Kolors 管道添加到文本到图像管道映射
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsPipeline
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING["kolors-pag"] = KolorsPAGPipeline
    # 将 Kolors 管道添加到图像到图像管道映射
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["kolors"] = KolorsPipeline

# 定义支持的任务映射，包含各种管道映射
SUPPORTED_TASKS_MAPPINGS = [
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING,
    _AUTO_INPAINT_DECODER_PIPELINES_MAPPING,
]

# 定义函数以获取连接的管道，参数为管道类
def _get_connected_pipeline(pipeline_cls):
    # 当前连接的管道只能从解码器管道加载
    # 检查 pipeline_cls 是否在自动文本到图像解码器管道映射的值中
        if pipeline_cls in _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING.values():
            # 获取与给定 pipeline_cls 名称对应的任务类，不存在时不抛出错误
            return _get_task_class(
                AUTO_TEXT2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
            )
        # 检查 pipeline_cls 是否在自动图像到图像解码器管道映射的值中
        if pipeline_cls in _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING.values():
            # 获取与给定 pipeline_cls 名称对应的任务类，不存在时不抛出错误
            return _get_task_class(
                AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False
            )
        # 检查 pipeline_cls 是否在自动修复解码器管道映射的值中
        if pipeline_cls in _AUTO_INPAINT_DECODER_PIPELINES_MAPPING.values():
            # 获取与给定 pipeline_cls 名称对应的任务类，不存在时不抛出错误
            return _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, pipeline_cls.__name__, throw_error_if_not_exist=False)
# 根据映射获取任务类，如果不存在则抛出异常（默认抛出）
def _get_task_class(mapping, pipeline_class_name, throw_error_if_not_exist: bool = True):
    # 定义内部函数，用于根据管道类名获取模型名称
    def get_model(pipeline_class_name):
        # 遍历所有支持的任务映射
        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            # 遍历每个任务映射中的模型名称和管道
            for model_name, pipeline in task_mapping.items():
                # 如果管道名称与提供的类名匹配，返回模型名称
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    # 调用内部函数获取模型名称
    model_name = get_model(pipeline_class_name)

    # 如果找到了模型名称
    if model_name is not None:
        # 从映射中获取相应的任务类
        task_class = mapping.get(model_name, None)
        # 如果找到了任务类，返回该类
        if task_class is not None:
            return task_class

    # 如果模型不存在且需要抛出错误，抛出 ValueError
    if throw_error_if_not_exist:
        raise ValueError(f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name}")

# 定义一个文本到图像的自动管道类，继承自 ConfigMixin
class AutoPipelineForText2Image(ConfigMixin):
    r"""

    [`AutoPipelineForText2Image`] 是一个通用管道类，用于实例化文本到图像的管道类。
    特定的基础管道类将通过 [`~AutoPipelineForText2Image.from_pretrained`] 或
    [`~AutoPipelineForText2Image.from_pipe`] 方法自动选择。

    此类不能通过 `__init__()` 实例化（会抛出错误）。

    类属性：

        - **config_name** (`str`) -- 存储所有扩散管道组件的类和模块名称的配置文件名。

    """
    # 配置文件名称，指向模型索引
    config_name = "model_index.json"

    # 初始化方法，禁止直接实例化
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    # 类方法，用于验证 HF Hub 参数
    @classmethod
    @validate_hf_hub_args
    @classmethod
# 定义一个图像到图像的自动管道类，继承自 ConfigMixin
class AutoPipelineForImage2Image(ConfigMixin):
    r"""

    [`AutoPipelineForImage2Image`] 是一个通用管道类，用于实例化图像到图像的管道类。
    特定的基础管道类将通过 [`~AutoPipelineForImage2Image.from_pretrained`] 或
    [`~AutoPipelineForImage2Image.from_pipe`] 方法自动选择。

    此类不能通过 `__init__()` 实例化（会抛出错误）。

    类属性：

        - **config_name** (`str`) -- 存储所有扩散管道组件的类和模块名称的配置文件名。

    """
    # 配置文件名称，指向模型索引
    config_name = "model_index.json"

    # 初始化方法，禁止直接实例化
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    # 类方法，用于验证 HF Hub 参数
    @classmethod
    @validate_hf_hub_args
    @classmethod
# 定义一个图像修复的自动管道类，继承自 ConfigMixin
class AutoPipelineForInpainting(ConfigMixin):
    r"""

    [`AutoPipelineForInpainting`] 是一个通用管道类，用于实例化图像修复的管道类。该
    # 自动选择特定的基础管道类，可以通过 `from_pretrained` 或 `from_pipe` 方法实现
        specific underlying pipeline class is automatically selected from either the
        # 无法通过 `__init__()` 方法实例化该类（会抛出错误）
        [`~AutoPipelineForInpainting.from_pretrained`] or [`~AutoPipelineForInpainting.from_pipe`] methods.
    
        # 类属性：
        # - **config_name** (`str`) -- 存储所有扩散管道组件类和模块名称的配置文件名
        This class cannot be instantiated using `__init__()` (throws an error).
    
        # 配置文件名，指向模型索引的 JSON 文件
        config_name = "model_index.json"
    
        # 初始化方法，接受任意数量的位置和关键字参数
        def __init__(self, *args, **kwargs):
            # 抛出环境错误，指示使用特定的方法实例化该类
            raise EnvironmentError(
                f"{self.__class__.__name__} is designed to be instantiated "
                f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
                f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
            )
    
        # 类方法装饰器，表明该方法是属于类而不是实例的
        @classmethod
        @validate_hf_hub_args
        # 再次标记该方法为类方法
        @classmethod
```