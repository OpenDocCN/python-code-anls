# `.\transformers\integrations\peft.py`

```
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..utils import (
    check_peft_version,
    find_adapter_config_file,
    is_accelerate_available,
    is_peft_available,
    is_torch_available,
    logging,
)

# 如果 Accelerate 可用，则导入相关模块和函数
if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map

# 支持的最低 PEFT 版本
MIN_PEFT_VERSION = "0.5.0"

# 如果是类型检查环境
if TYPE_CHECKING:
    # 如果 Torch 可用，则导入 Torch 模块
    if is_torch_available():
        import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# PEFT 适配器混合类
class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods are all non-prefix tuning methods. Below is the list of supported PEFT methods
    that anyone can load, train and run with this mixin class:
    - Low Rank Adapters (LoRA): https://huggingface.co/docs/peft/conceptual_guides/lora
    - IA3: https://huggingface.co/docs/peft/conceptual_guides/ia3
    - AdaLora: https://arxiv.org/abs/2303.10512

    Other PEFT models such as prompt tuning, prompt learning are out of scope as these adapters are not "injectable"
    into a torch module. For using these methods, please refer to the usage guide of PEFT library.

    With this mixin, if the correct PEFT version is installed, it is possible to:

    - Load an adapter stored on a local path or in a remote Hub repository, and inject it in the model
    - Attach new adapters in the model and train them with Trainer or by your own.
    - Attach multiple adapters and iteratively activate / deactivate them
    - Activate / deactivate all adapters from the model.
    - Get the `state_dict` of the active adapter.
    """

    # PEFT 配置是否已加载的标志
    _hf_peft_config_loaded = False
    def load_adapter(
        self,
        peft_model_id: Optional[str] = None,  # 加载适配器的方法，支持指定 PEFT 模型 ID
        adapter_name: Optional[str] = None,   # 适配器的名称，默认为 None
        revision: Optional[str] = None,       # 适配器的修订版本，默认为 None
        token: Optional[str] = None,          # 访问 PEFT API 所需的 token，默认为 None
        device_map: Optional[str] = "auto",   # 设备映射方式，默认为 "auto"
        max_memory: Optional[str] = None,     # 最大内存限制，默认为 None
        offload_folder: Optional[str] = None, # 离线加载的文件夹路径，默认为 None
        offload_index: Optional[int] = None,  # 离线加载的索引，默认为 None
        peft_config: Dict[str, Any] = None,   # PEFT 配置信息，默认为 None
        adapter_state_dict: Optional[Dict[str, "torch.Tensor"]] = None,  # 适配器状态字典，默认为 None
        adapter_kwargs: Optional[Dict[str, Any]] = None,  # 适配器关键字参数，默认为 None
    def add_adapter(self, adapter_config, adapter_name: Optional[str] = None) -> None:  # 添加适配器的方法
        r"""
        如果您对适配器和 PEFT 方法不熟悉，我们建议您在 PEFT 官方文档中阅读更多信息：https://huggingface.co/docs/peft

        在当前模型中添加一个全新的适配器以进行训练。如果未传递适配器名称，则会为适配器分配一个默认名称，以遵循 PEFT 库的约定（在 PEFT 中我们使用 "default" 作为默认适配器名称）。

        Args:
            adapter_config (`~peft.PeftConfig`):  # 适配器的配置信息，支持非前缀调整和适应提示方法
                要添加的适配器的配置，支持的适配器为非前缀调整和适应提示方法
            adapter_name (`str`, *optional*, defaults to `"default"`):  # 要添加的适配器的名称，默认为 "default"
                要添加的适配器的名称。如果未传递名称，则会为适配器分配一个默认名称。
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)  # 检查 PEFT 的版本是否符合要求

        from peft import PeftConfig, inject_adapter_in_model  # 导入 PEFT 配置和将适配器注入模型的方法

        adapter_name = adapter_name or "default"  # 如果未指定适配器名称，则使用默认名称 "default"

        if not self._hf_peft_config_loaded:  # 如果未加载 PEFT 配置
            self._hf_peft_config_loaded = True  # 设置 PEFT 配置已加载
        elif adapter_name in self.peft_config:  # 如果适配器名称已经存在于 PEFT 配置中
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")  # 抛出值错误

        if not isinstance(adapter_config, PeftConfig):  # 如果适配器配置不是 PeftConfig 的实例
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )  # 抛出值错误

        # 检索模型的名称或路径，也可以使用 self.config._name_or_path
        # 但为了与 PEFT 中的操作保持一致：https://github.com/huggingface/peft/blob/6e783780ca9df3a623992cc4d1d665001232eae0/src/peft/mapping.py#L100
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)  # 设置适配器配置的基础模型名称或路径
        inject_adapter_in_model(adapter_config, self, adapter_name)  # 将适配器注入模型

        self.set_adapter(adapter_name)  # 设置适配器
    def set_adapter(self, adapter_name: Union[List[str], str]) -> None:
        """
        设置特定的适配器，强制模型使用该适配器并禁用其他适配器。

        Args:
            adapter_name (`Union[List[str], str]`):
                要设置的适配器的名称。也可以是一个字符串列表，以设置多个适配器。
        """
        # 检查 PEFT 版本是否符合要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
        # 如果没有加载 PEFT 配置，则抛出异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        # 如果 adapter_name 是列表
        elif isinstance(adapter_name, list):
            # 找出缺失的适配器
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        # 如果 adapter_name 不在 peft_config 中
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        # 导入必要的模块
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # 对于旧版本 PEFT 的向后兼容性
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        # 如果没有成功设置适配器，则抛出异常
        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
    # 禁用模型上的所有适配器，使其只使用基础模型进行推理
    def disable_adapters(self) -> None:
        r"""
        如果您对适配器和 PEFT 方法不熟悉，我们邀请您在 PEFT 官方文档中阅读更多信息：https://huggingface.co/docs/peft

        禁用连接到模型的所有适配器。这将导致仅使用基础模型进行推理。
        """
        # 检查 PEFT 版本是否符合要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果没有加载 HF PEFT 配置，则引发值错误
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入必要的模块和类
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            # 如果模块是 BaseTunerLayer 或 ModulesToSaveWrapper 的实例
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # 最近版本的 PEFT 需要调用 `enable_adapters` 而不是 `disable_adapters`
                if hasattr(module, "enable_adapters"):
                    # 调用模块的 enable_adapters 方法，禁用适配器
                    module.enable_adapters(enabled=False)
                else:
                    # 直接将模块的 disable_adapters 属性设置为 True
                    module.disable_adapters = True

    # 启用模型上的适配器，模型将使用 `self.active_adapter()`
    def enable_adapters(self) -> None:
        """
        如果您对适配器和 PEFT 方法不熟悉，我们邀请您在 PEFT 官方文档中阅读更多信息：https://huggingface.co/docs/peft

        启用连接到模型的适配器。模型将使用 `self.active_adapter()`
        """
        # 检查 PEFT 版本是否符合要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果没有加载 HF PEFT 配置，则引发值错误
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入必要的模块和类
        from peft.tuners.tuners_utils import BaseTunerLayer

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            # 如果模块是 BaseTunerLayer 的实例
            if isinstance(module, BaseTunerLayer):
                # 最近版本的 PEFT 需要调用 `enable_adapters` 而不是 `disable_adapters`
                if hasattr(module, "enable_adapters"):
                    # 调用模块的 enable_adapters 方法，启用适配器
                    module.enable_adapters(enabled=True)
                else:
                    # 直接将模块的 disable_adapters 属性设置为 False
                    module.disable_adapters = False
    # 获取当前模型的活动适配器
    def active_adapters(self) -> List[str]:
        """
        如果您对适配器和 PEFT 方法不熟悉，请阅读 PEFT 官方文档获取更多信息：https://huggingface.co/docs/peft
        
        获取模型的当前活动适配器。在多适配器推理情况下（将多个适配器组合进行推理），返回所有活动适配器的列表，以便用户可以相应地处理它们。

        对于旧版本的 PEFT（不支持多适配器推理），`module.active_adapter` 将返回单个字符串。
        """
        # 检查 PEFT 版本是否满足最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 检查是否已安装 PEFT
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        # 检查是否加载了 PEFT 配置
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入 PEFT 调谐器工具类
        from peft.tuners.tuners_utils import BaseTunerLayer

        # 遍历模型的命名模块
        for _, module in self.named_modules():
            # 检查模块是否是 PEFT 调谐器层
            if isinstance(module, BaseTunerLayer):
                # 获取活动适配器
                active_adapters = module.active_adapter
                break

        # 对于旧版本的 PEFT
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        return active_adapters

    # 获取活动适配器（已弃用）
    def active_adapter(self) -> str:
        # 发出警告，该方法已弃用，并将在未来版本中移除
        warnings.warn(
            "The `active_adapter` method is deprecated and will be removed in a future version.", FutureWarning
        )

        # 返回活动适配器列表的第一个元素
        return self.active_adapters()[0]

    # 获取适配器状态字典
    def get_adapter_state_dict(self, adapter_name: Optional[str] = None) -> dict:
        """
        如果您对适配器和 PEFT 方法不熟悉，请阅读 PEFT 官方文档获取更多信息：https://huggingface.co/docs/peft
        
        获取适配器状态字典，该字典应仅包含指定适配器名称适配器的权重张量。如果未传递适配器名称，则使用活动适配器。

        Args:
            adapter_name (`str`, *optional*):
                要从中获取状态字典的适配器名称。如果未传递名称，则使用活动适配器。
        """
        # 检查 PEFT 版本是否满足最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 检查是否加载了 PEFT 配置
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入获取 PEFT 模型状态字典的函数
        from peft import get_peft_model_state_dict

        # 如果未传递适配器名称，则使用活动适配器
        if adapter_name is None:
            adapter_name = self.active_adapter()

        # 获取指定适配器名称的适配器状态字典
        adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
        return adapter_state_dict

    # 分发加速模型
    def _dispatch_accelerate_model(
        self,
        device_map: str,
        max_memory: Optional[int] = None,
        offload_folder: Optional[str] = None,
        offload_index: Optional[int] = None,
    ) -> None:
        """
        Optional re-dispatch the model and attach new hooks to the model in case the model has been loaded with
        accelerate (i.e. with `device_map=xxx`)

        Args:
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_index (`int`, *optional*):
                The offload_index argument to be passed to `accelerate.dispatch_model` method.
        """
        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = self._no_split_modules

        # Check if device_map is not "sequential"
        if device_map != "sequential":
            # Calculate balanced memory allocation based on device_map
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        
        # If device_map is a string, infer the auto device map
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        
        # Dispatch the model with specified device_map and offload_folder
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_folder,
            **dispatch_model_kwargs,
        )
```