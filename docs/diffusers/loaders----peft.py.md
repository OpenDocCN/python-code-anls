# `.\diffusers\loaders\peft.py`

```py
# 指定文件的编码格式为 UTF-8
# coding=utf-8
# 版权声明，表示此代码归 2024 The HuggingFace Inc. 团队所有
# Copyright 2024 The HuggingFace Inc. team.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则不得使用此文件。
# 可在以下网址获取许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则软件在“按原样”基础上分发，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证下具体权利和限制的更多信息，请查看许可证。
import inspect  # 导入 inspect 模块，用于获取对象的信息
from functools import partial  # 从 functools 模块导入 partial 函数，用于部分应用
from typing import Dict, List, Optional, Union  # 导入类型注解，便于类型提示

from ..utils import (  # 从父目录的 utils 模块导入多个工具函数
    MIN_PEFT_VERSION,  # 最小 PEFT 版本常量
    USE_PEFT_BACKEND,  # 使用 PEFT 后端的标志
    check_peft_version,  # 检查 PEFT 版本的函数
    delete_adapter_layers,  # 删除适配器层的函数
    is_peft_available,  # 检查 PEFT 是否可用的函数
    set_adapter_layers,  # 设置适配器层的函数
    set_weights_and_activate_adapters,  # 设置权重并激活适配器的函数
)
from .unet_loader_utils import _maybe_expand_lora_scales  # 从当前目录的 unet_loader_utils 模块导入函数

# 定义适配器缩放函数映射字典，以模型名称为键，缩放函数为值
_SET_ADAPTER_SCALE_FN_MAPPING = {
    "UNet2DConditionModel": _maybe_expand_lora_scales,  # UNet2DConditionModel 使用特定缩放函数
    "UNetMotionModel": _maybe_expand_lora_scales,  # UNetMotionModel 使用特定缩放函数
    "SD3Transformer2DModel": lambda model_cls, weights: weights,  # SD3Transformer2DModel 直接返回权重
    "FluxTransformer2DModel": lambda model_cls, weights: weights,  # FluxTransformer2DModel 直接返回权重
}

class PeftAdapterMixin:  # 定义 PeftAdapterMixin 类
    """
    包含用于加载和使用适配器权重的所有函数，该函数在 PEFT 库中受支持。有关适配器的更多详细信息以及如何将其注入基础模型，请查阅 PEFT
    [文档](https://huggingface.co/docs/peft/index)。

    安装最新版本的 PEFT，并使用此混入以：

    - 在模型中附加新适配器。
    - 附加多个适配器并逐步激活/停用它们。
    - 激活/停用模型中的所有适配器。
    - 获取活动适配器的列表。
    """

    _hf_peft_config_loaded = False  # 初始化标志，指示 PEFT 配置是否已加载

    def set_adapters(  # 定义设置适配器的方法
        self,
        adapter_names: Union[List[str], str],  # 适配器名称，可以是字符串或字符串列表
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,  # 可选的权重参数
    ):
        """
        设置当前活跃的适配器，以便在 UNet 中使用。

        参数：
            adapter_names (`List[str]` 或 `str`):
                要使用的适配器名称。
            adapter_weights (`Union[List[float], float]`, *可选*):
                与 UNet 一起使用的适配器权重。如果为 `None`，则所有适配器的权重设置为 `1.0`。

        示例：

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```py
        """
        # 检查是否启用 PEFT 后端，如果未启用则引发错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        # 如果 adapter_names 是字符串，则将其转换为列表
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # 将权重扩展为列表，每个适配器一个条目
        # 例如对于 2 个适配器:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        # 检查适配器名称和权重的长度是否匹配
        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # 将 None 值设置为默认的 1.0
        # 例如: [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # 扩展权重以适配特定适配器的要求
        # 例如: [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
        weights = scale_expansion_fn(self, weights)

        # 设置权重并激活适配器
        set_weights_and_activate_adapters(self, adapter_names, weights)
    # 定义一个添加适配器的方法，接受适配器配置和可选的适配器名称
    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        r"""
        向当前模型添加一个新的适配器用于训练。如果未传递适配器名称，将为适配器分配默认名称，以遵循 PEFT 库的约定。

        如果您不熟悉适配器和 PEFT 方法，建议您查看 PEFT 的
        [文档](https://huggingface.co/docs/peft)。

        参数:
            adapter_config (`[~peft.PeftConfig]`):
                要添加的适配器的配置；支持的适配器包括非前缀调整和适应提示方法。
            adapter_name (`str`, *可选*, 默认为 `"default"`):
                要添加的适配器名称。如果未传递名称，将为适配器分配默认名称。
        """
        # 检查 PEFT 的版本是否符合最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果 PEFT 不可用，则引发 ImportError 异常
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        # 从 peft 模块导入 PeftConfig 和 inject_adapter_in_model
        from peft import PeftConfig, inject_adapter_in_model

        # 如果尚未加载 HF PEFT 配置，则设置标志为 True
        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        # 如果适配器名称已存在，则引发 ValueError 异常
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        # 检查 adapter_config 是否为 PeftConfig 的实例
        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # 将适配器配置的基础模型名称或路径设置为 None，因为加载逻辑由 load_lora_layers 或 StableDiffusionLoraLoaderMixin 处理
        adapter_config.base_model_name_or_path = None
        # 将适配器注入到模型中
        inject_adapter_in_model(adapter_config, self, adapter_name)
        # 设置适配器名称
        self.set_adapter(adapter_name)
    # 设置特定适配器，强制模型仅使用该适配器并禁用其他适配器
    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        设置特定适配器，强制模型仅使用该适配器并禁用其他适配器。
    
        如果您不熟悉适配器和 PEFT 方法，我们邀请您阅读 PEFT 的更多信息
        [文档](https://huggingface.co/docs/peft)。
    
        参数：
            adapter_name (Union[str, List[str]])):
                要设置的适配器名称或适配器名称列表（如果是单个适配器）。
        """
        # 检查 PEFT 版本是否符合最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
    
        # 如果尚未加载 HF PEFT 配置，则抛出错误
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
    
        # 如果适配器名称是字符串，则将其转换为列表
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]
    
        # 计算缺失的适配器名称
        missing = set(adapter_name) - set(self.peft_config)
        # 如果有缺失的适配器，则抛出错误
        if len(missing) > 0:
            raise ValueError(
                f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                f" current loaded adapters are: {list(self.peft_config.keys())}"
            )
    
        # 从 peft.tuners.tuners_utils 导入 BaseTunerLayer
        from peft.tuners.tuners_utils import BaseTunerLayer
    
        # 初始化适配器设置标志
        _adapters_has_been_set = False
    
        # 遍历模型中命名的模块
        for _, module in self.named_modules():
            # 如果模块是 BaseTunerLayer 的实例
            if isinstance(module, BaseTunerLayer):
                # 如果模块具有 set_adapter 方法，则调用该方法
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # 如果没有 set_adapter 方法且适配器名称列表不为1，抛出错误
                elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                    raise ValueError(
                        "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                        " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                    )
                # 否则，将活动适配器设置为适配器名称
                else:
                    module.active_adapter = adapter_name
                # 标记适配器已设置
                _adapters_has_been_set = True
    
        # 如果没有成功设置适配器，则抛出错误
        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
    # 定义一个方法来禁用模型中所有附加的适配器，仅使用基础模型进行推理
    def disable_adapters(self) -> None:
        r"""
        禁用所有附加到模型的适配器，并回退到仅使用基础模型进行推理。
    
        如果您对适配器和 PEFT 方法不熟悉，我们邀请您在 PEFT
        [文档](https://huggingface.co/docs/peft) 中了解更多信息。
        """
        # 检查 PEFT 版本，确保满足最低版本要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
    
        # 如果没有加载 HF PEFT 配置，则抛出异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
    
        # 从 tuners_utils 导入基础调优层
        from peft.tuners.tuners_utils import BaseTunerLayer
    
        # 遍历模型中所有命名的模块
        for _, module in self.named_modules():
            # 如果模块是基础调优层的实例
            if isinstance(module, BaseTunerLayer):
                # 如果模块具有 enable_adapters 属性，则禁用适配器
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # 支持旧版 PEFT
                    module.disable_adapters = True
    
    # 定义一个方法来启用附加到模型的适配器
    def enable_adapters(self) -> None:
        """
        启用附加到模型的适配器。模型使用 `self.active_adapters()` 检索要启用的适配器列表。
    
        如果您对适配器和 PEFT 方法不熟悉，我们邀请您在 PEFT
        [文档](https://huggingface.co/docs/peft) 中了解更多信息。
        """
        # 检查 PEFT 版本，确保满足最低版本要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
    
        # 如果没有加载 HF PEFT 配置，则抛出异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
    
        # 从 tuners_utils 导入基础调优层
        from peft.tuners.tuners_utils import BaseTunerLayer
    
        # 遍历模型中所有命名的模块
        for _, module in self.named_modules():
            # 如果模块是基础调优层的实例
            if isinstance(module, BaseTunerLayer):
                # 如果模块具有 enable_adapters 属性，则启用适配器
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # 支持旧版 PEFT
                    module.disable_adapters = False
    
    # 定义一个方法来获取当前活动的适配器列表
    def active_adapters(self) -> List[str]:
        """
        获取模型当前活动的适配器列表。
    
        如果您对适配器和 PEFT 方法不熟悉，我们邀请您在 PEFT
        [文档](https://huggingface.co/docs/peft) 中了解更多信息。
        """
        # 检查 PEFT 版本，确保满足最低版本要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
    
        # 如果 PEFT 不可用，则抛出导入错误
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")
    
        # 如果没有加载 HF PEFT 配置，则抛出异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
    
        # 从 tuners_utils 导入基础调优层
        from peft.tuners.tuners_utils import BaseTunerLayer
    
        # 遍历模型中所有命名的模块
        for _, module in self.named_modules():
            # 如果模块是基础调优层的实例
            if isinstance(module, BaseTunerLayer):
                # 返回活动适配器
                return module.active_adapter
    # 定义融合 LoRA 的方法，允许传入比例和安全融合标志
        def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
            # 检查是否使用 PEFT 后端
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for `fuse_lora()`.")
    
            # 设置 LoRA 的比例
            self.lora_scale = lora_scale
            # 设置安全融合标志
            self._safe_fusing = safe_fusing
            # 应用融合方法，部分应用于指定的适配器名称
            self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))
    
        # 定义具体的 LoRA 融合应用方法
        def _fuse_lora_apply(self, module, adapter_names=None):
            # 从 PEFT 库导入基础调优层
            from peft.tuners.tuners_utils import BaseTunerLayer
    
            # 设置合并参数，包含安全合并标志
            merge_kwargs = {"safe_merge": self._safe_fusing}
    
            # 检查模块是否为基础调优层
            if isinstance(module, BaseTunerLayer):
                # 如果 LoRA 比例不为 1.0，则调整比例
                if self.lora_scale != 1.0:
                    module.scale_layer(self.lora_scale)
    
                # 检查合并方法是否支持适配器名称参数
                supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
                if "adapter_names" in supported_merge_kwargs:
                    merge_kwargs["adapter_names"] = adapter_names
                elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                    # 抛出错误提示 PEFT 版本不支持适配器名称
                    raise ValueError(
                        "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                        " to the latest version of PEFT. `pip install -U peft`"
                    )
    
                # 调用合并方法，传入合并参数
                module.merge(**merge_kwargs)
    
        # 定义解除 LoRA 融合的方法
        def unfuse_lora(self):
            # 检查是否使用 PEFT 后端
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for `unfuse_lora()`.")
            # 应用解除融合方法
            self.apply(self._unfuse_lora_apply)
    
        # 定义具体的 LoRA 解除融合应用方法
        def _unfuse_lora_apply(self, module):
            # 从 PEFT 库导入基础调优层
            from peft.tuners.tuners_utils import BaseTunerLayer
    
            # 检查模块是否为基础调优层
            if isinstance(module, BaseTunerLayer):
                # 调用解除合并方法
                module.unmerge()
    
        # 定义卸载 LoRA 的方法
        def unload_lora(self):
            # 检查是否使用 PEFT 后端
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for `unload_lora()`.")
    
            # 从工具库导入递归删除 PEFT 层的方法
            from ..utils import recurse_remove_peft_layers
    
            # 递归删除 PEFT 层
            recurse_remove_peft_layers(self)
            # 如果存在 PEFT 配置，则删除该属性
            if hasattr(self, "peft_config"):
                del self.peft_config
    
        # 定义禁用 LoRA 的方法
        def disable_lora(self):
            """
            禁用底层模型的活动 LoRA 层。
    
            示例：
    
            ```py
            from diffusers import AutoPipelineForText2Image
            import torch
    
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
            ).to("cuda")
            pipeline.load_lora_weights(
                "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
            )
            pipeline.disable_lora()
            ```py
            """
            # 检查是否使用 PEFT 后端
            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for this method.")
            # 设置适配器层为禁用状态
            set_adapter_layers(self, enabled=False)
    # 启用底层模型的活动 LoRA 层
    def enable_lora(self):
        # 检查是否启用了 PEFT 后端，若未启用则抛出错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        # 设置适配器层为启用状态
        set_adapter_layers(self, enabled=True)
    
    # 删除底层模型的适配器 LoRA 层
    def delete_adapters(self, adapter_names: Union[List[str], str]):
        # 删除适配器的 LoRA 层的说明
        """
        Delete an adapter's LoRA layers from the underlying model.
    
        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.
    
        Example:
        ...
        """
        # 检查是否启用了 PEFT 后端，若未启用则抛出错误
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
    
        # 如果传入的是单个适配器名称，将其转换为列表
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
    
        # 遍历所有适配器名称，逐个删除
        for adapter_name in adapter_names:
            # 删除指定适配器的层
            delete_adapter_layers(self, adapter_name)
    
            # 从配置中也删除相应的适配器
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)
```