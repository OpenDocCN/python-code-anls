# `.\diffusers\loaders\__init__.py`

```py
# 引入类型检查支持
from typing import TYPE_CHECKING

# 从上级目录导入工具函数和常量
from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, deprecate
from ..utils.import_utils import is_peft_available, is_torch_available, is_transformers_available


# 定义获取文本编码器 LORA 状态字典的函数
def text_encoder_lora_state_dict(text_encoder):
    # 发出关于函数即将弃用的警告
    deprecate(
        "text_encoder_load_state_dict in `models`",
        "0.27.0",
        "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
    )
    # 初始化状态字典
    state_dict = {}

    # 遍历文本编码器的注意力模块
    for name, module in text_encoder_attn_modules(text_encoder):
        # 获取 q_proj 线性层的状态字典并更新状态字典
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        # 获取 k_proj 线性层的状态字典并更新状态字典
        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        # 获取 v_proj 线性层的状态字典并更新状态字典
        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        # 获取 out_proj 线性层的状态字典并更新状态字典
        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    # 返回构建的状态字典
    return state_dict


# 检查是否可用 Transformers 库
if is_transformers_available():

    # 定义获取文本编码器注意力模块的函数
    def text_encoder_attn_modules(text_encoder):
        # 发出关于函数即将弃用的警告
        deprecate(
            "text_encoder_attn_modules in `models`",
            "0.27.0",
            "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",
        )
        # 从 Transformers 导入相关模型
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        # 初始化注意力模块列表
        attn_modules = []

        # 检查文本编码器的类型并获取相应的注意力模块
        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))
        else:
            # 如果不认识的编码器类型，抛出错误
            raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

        # 返回注意力模块列表
        return attn_modules


# 初始化导入结构字典
_import_structure = {}

# 检查是否可用 PyTorch 库
if is_torch_available():
    # 更新导入结构以包含单文件模型
    _import_structure["single_file_model"] = ["FromOriginalModelMixin"]

    # 更新导入结构以包含 UNet
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    # 更新导入结构以包含工具函数
    _import_structure["utils"] = ["AttnProcsLayers"]
    # 检查是否可以使用 transformers 库
        if is_transformers_available():
            # 将 "single_file" 模块的导入结构更新为包含 FromSingleFileMixin 类
            _import_structure["single_file"] = ["FromSingleFileMixin"]
            # 将 "lora_pipeline" 模块的导入结构更新为包含多个 LoraLoaderMixin 类
            _import_structure["lora_pipeline"] = [
                "AmusedLoraLoaderMixin",  # 包含 AmusedLoraLoaderMixin 类
                "StableDiffusionLoraLoaderMixin",  # 包含 StableDiffusionLoraLoaderMixin 类
                "SD3LoraLoaderMixin",  # 包含 SD3LoraLoaderMixin 类
                "StableDiffusionXLLoraLoaderMixin",  # 包含 StableDiffusionXLLoraLoaderMixin 类
                "LoraLoaderMixin",  # 包含 LoraLoaderMixin 类
                "FluxLoraLoaderMixin",  # 包含 FluxLoraLoaderMixin 类
            ]
            # 将 "textual_inversion" 模块的导入结构更新为包含 TextualInversionLoaderMixin 类
            _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
            # 将 "ip_adapter" 模块的导入结构更新为包含 IPAdapterMixin 类
            _import_structure["ip_adapter"] = ["IPAdapterMixin"]
# 将 "peft" 模块的结构信息映射到包含的类
_import_structure["peft"] = ["PeftAdapterMixin"]

# 检查是否在类型检查模式或慢导入标志被设置
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 检查 PyTorch 是否可用
    if is_torch_available():
        # 从单文件模型导入混合类
        from .single_file_model import FromOriginalModelMixin
        # 从 UNet 导入混合类
        from .unet import UNet2DConditionLoadersMixin
        # 从工具模块导入 Attention 处理层类
        from .utils import AttnProcsLayers

        # 检查 Transformers 是否可用
        if is_transformers_available():
            # 从 IP 适配器模块导入混合类
            from .ip_adapter import IPAdapterMixin
            # 从 Lora 管道模块导入多个加载器混合类
            from .lora_pipeline import (
                AmusedLoraLoaderMixin,
                FluxLoraLoaderMixin,
                LoraLoaderMixin,
                SD3LoraLoaderMixin,
                StableDiffusionLoraLoaderMixin,
                StableDiffusionXLLoraLoaderMixin,
            )
            # 从单文件模块导入混合类
            from .single_file import FromSingleFileMixin
            # 从文本反演模块导入混合类
            from .textual_inversion import TextualInversionLoaderMixin

    # 从 PEFT 模块导入 PeftAdapterMixin 类
    from .peft import PeftAdapterMixin
else:
    # 如果不在类型检查或慢导入状态，导入系统模块
    import sys

    # 使用懒加载模块，将当前模块替换为 _LazyModule 实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```