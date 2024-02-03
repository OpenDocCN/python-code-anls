# `stable-diffusion-webui\modules\sd_models_xl.py`

```
# 导入未来版本的注释语法
from __future__ import annotations

# 导入 torch 库
import torch

# 导入 sgm.models.diffusion 模块
import sgm.models.diffusion

# 导入 sgm.modules.diffusionmodules.denoiser_scaling 模块
import sgm.modules.diffusionmodules.denoiser_scaling

# 导入 sgm.modules.diffusionmodules.discretizer 模块
import sgm.modules.diffusionmodules.discretizer

# 从 modules 模块中导入 devices, shared, prompt_parser
from modules import devices, shared, prompt_parser

# 定义函数 get_learned_conditioning，接受 self 和 batch 作为参数
def get_learned_conditioning(self: sgm.models.diffusion.DiffusionEngine, batch: prompt_parser.SdConditioning | list[str]):
    # 遍历 self.conditioner.embedders 中的 embedder，并将其 ucg_rate 设置为 0.0
    for embedder in self.conditioner.embedders:
        embedder.ucg_rate = 0.0

    # 获取 batch 中的 width 和 height，如果不存在则使用默认值 1024
    width = getattr(batch, 'width', 1024)
    height = getattr(batch, 'height', 1024)

    # 获取 batch 中的 is_negative_prompt 和 aesthetic_score
    is_negative_prompt = getattr(batch, 'is_negative_prompt', False)
    aesthetic_score = shared.opts.sdxl_refiner_low_aesthetic_score if is_negative_prompt else shared.opts.sdxl_refiner_high_aesthetic_score

    # 定义 devices_args 字典，包含设备和数据类型信息
    devices_args = dict(device=devices.device, dtype=devices.dtype)

    # 定义 sdxl_conds 字典，包含条件信息
    sdxl_conds = {
        "txt": batch,
        "original_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "crop_coords_top_left": torch.tensor([shared.opts.sdxl_crop_top, shared.opts.sdxl_crop_left], **devices_args).repeat(len(batch), 1),
        "target_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "aesthetic_score": torch.tensor([aesthetic_score], **devices_args).repeat(len(batch), 1),
    }

    # 如果 is_negative_prompt 为 True 且 batch 中所有元素为空字符串，则 force_zero_negative_prompt 为 True
    force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in batch)

    # 调用 self.conditioner 函数，传入 sdxl_conds 和 force_zero_embeddings 参数
    c = self.conditioner(sdxl_conds, force_zero_embeddings=['txt'] if force_zero_negative_prompt else [])

    # 返回结果 c
    return c

# 定义函数 apply_model，接受 self, x, t, cond 作为参数
def apply_model(self: sgm.models.diffusion.DiffusionEngine, x, t, cond):
    # 调用 self.model 函数，传入 x, t, cond 参数
    return self.model(x, t, cond)

# 定义函数 get_first_stage_encoding，接受 self 和 x 作为参数，用于兼容性
def get_first_stage_encoding(self, x):  
    # 返回输入 x
    return x

# 将 get_learned_conditioning 函数赋值给 sgm.models.diffusion.DiffusionEngine 类的 get_learned_conditioning 方法
sgm.models.diffusion.DiffusionEngine.get_learned_conditioning = get_learned_conditioning

# 将 apply_model 函数赋值给 sgm.models.diffusion.DiffusionEngine 类的 apply_model 方法
sgm.models.diffusion.DiffusionEngine.apply_model = apply_model
# 将 DiffusionEngine 类的 get_first_stage_encoding 方法赋值给 GeneralConditioner 类的 get_first_stage_encoding 方法
sgm.models.diffusion.DiffusionEngine.get_first_stage_encoding = get_first_stage_encoding

# 定义一个方法，用于对初始文本进行编码
def encode_embedding_init_text(self: sgm.modules.GeneralConditioner, init_text, nvpt):
    # 初始化结果列表
    res = []

    # 遍历 embedders 列表中具有 encode_embedding_init_text 方法的 embedder 对象
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'encode_embedding_init_text')]:
        # 调用 embedder 对象的 encode_embedding_init_text 方法对初始文本进行编码
        encoded = embedder.encode_embedding_init_text(init_text, nvpt)
        # 将编码结果添加到结果列表中
        res.append(encoded)

    # 沿着指定维度拼接结果列表中的张量
    return torch.cat(res, dim=1)

# 定义一个方法，用于对文本进行分词
def tokenize(self: sgm.modules.GeneralConditioner, texts):
    # 遍历 embedders 列表中具有 tokenize 方法的 embedder 对象
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'tokenize')]:
        # 调用 embedder 对象的 tokenize 方法对文本进行分词
        return embedder.tokenize(texts)

    # 如果没有可用的分词器，则抛出 AssertionError 异常
    raise AssertionError('no tokenizer available')

# 定义一个方法，用于处理文本
def process_texts(self, texts):
    # 遍历 embedders 列表中具有 process_texts 方法的 embedder 对象
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'process_texts')]:
        # 调用 embedder 对象的 process_texts 方法处理文本
        return embedder.process_texts(texts)

# 定义一个方法，用于获取目标提示的令牌数量
def get_target_prompt_token_count(self, token_count):
    # 遍历 embedders 列表中具有 get_target_prompt_token_count 方法的 embedder 对象
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'get_target_prompt_token_count')]:
        # 调用 embedder 对象的 get_target_prompt_token_count 方法获取目标提示的令牌数量
        return embedder.get_target_prompt_token_count(token_count)

# 将 encode_embedding_init_text、tokenize、process_texts 和 get_target_prompt_token_count 方法赋值给 GeneralConditioner 类
sgm.modules.GeneralConditioner.encode_embedding_init_text = encode_embedding_init_text
sgm.modules.GeneralConditioner.tokenize = tokenize
sgm.modules.GeneralConditioner.process_texts = process_texts
sgm.modules.GeneralConditioner.get_target_prompt_token_count = get_target_prompt_token_count

# 定义一个方法，用于扩展 SDXL 模型的参数，使其更像 SD1.5 模型
def extend_sdxl(model):
    """this adds a bunch of parameters to make SDXL model look a bit more like SD1.5 to the rest of the codebase."""

    # 获取模型中扩散模型的参数类型，并赋值给模型的扩散模型
    dtype = next(model.model.diffusion_model.parameters()).dtype
    model.model.diffusion_model.dtype = dtype
    # 设置模型的 conditioning_key 为 'crossattn'
    model.model.conditioning_key = 'crossattn'
    # 设置模型的 cond_stage_key 为 'txt'
    model.cond_stage_key = 'txt'
    # model.cond_stage_model 将在 sd_hijack 中设置
    # 根据 model.denoiser.scaling 的类型判断 parameterization 的取值，如果是 VScaling 类型则为 "v"，否则为 "eps"
    model.parameterization = "v" if isinstance(model.denoiser.scaling, sgm.modules.diffusionmodules.denoiser_scaling.VScaling) else "eps"

    # 创建 LegacyDDPMDiscretization 的实例对象 discretization
    discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
    # 将 discretization 中的 alphas_cumprod 转换为 torch 数组，并指定设备和数据类型，赋值给 model.alphas_cumprod
    model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=dtype)

    # 将一个空的 torch.nn.Module 对象赋值给 model.conditioner.wrapped
    model.conditioner.wrapped = torch.nn.Module()
# 将sgm.modules.attention.print的值设置为shared.ldm_print
sgm.modules.attention.print = shared.ldm_print
# 将sgm.modules.diffusionmodules.model.print的值设置为shared.ldm_print
sgm.modules.diffusionmodules.model.print = shared.ldm_print
# 将sgm.modules.diffusionmodules.openaimodel.print的值设置为shared.ldm_print
sgm.modules.diffusionmodules.openaimodel.print = shared.ldm_print
# 将sgm.modules.encoders.modules.print的值设置为shared.ldm_print
sgm.modules.encoders.modules.print = shared.ldm_print

# 设置sgm.modules.attention.SDP_IS_AVAILABLE为True，表示可用
sgm.modules.attention.SDP_IS_AVAILABLE = True
# 设置sgm.modules.attention.XFORMERS_IS_AVAILABLE为False，表示不可用
sgm.modules.attention.XFORMERS_IS_AVAILABLE = False
```