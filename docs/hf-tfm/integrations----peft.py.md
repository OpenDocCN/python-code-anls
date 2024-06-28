# `.\integrations\peft.py`

```py
# 导入所需的模块和函数
import inspect  # 导入 inspect 模块，用于检查和分析 Python 对象的属性和结构
import warnings  # 导入 warnings 模块，用于管理警告信息
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union  # 从 typing 模块导入类型提示相关的类型

from ..utils import (
    check_peft_version,  # 导入 check_peft_version 函数，用于检查 PEFT 版本
    find_adapter_config_file,  # 导入 find_adapter_config_file 函数，用于查找适配器配置文件
    is_accelerate_available,  # 导入 is_accelerate_available 函数，用于检查是否可用 accelerate 库
    is_peft_available,  # 导入 is_peft_available 函数，用于检查是否可用 PEFT 库
    is_torch_available,  # 导入 is_torch_available 函数，用于检查是否可用 torch 库
    logging,  # 导入 logging 模块，用于记录日志
)


if is_accelerate_available():  # 如果 accelerate 库可用，则执行以下导入
    from accelerate import dispatch_model  # 导入 dispatch_model 函数，用于调度模型
    from accelerate.utils import get_balanced_memory, infer_auto_device_map  # 导入 get_balanced_memory 和 infer_auto_device_map 函数，用于内存管理和设备映射推断

# PEFT 集成所需的最低版本
MIN_PEFT_VERSION = "0.5.0"

if TYPE_CHECKING:  # 如果是类型检查阶段，则执行以下导入
    if is_torch_available():  # 如果 torch 库可用，则导入 torch 库
        import torch  # 导入 torch 库

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class PeftAdapterMixin:
    """
    包含加载和使用 PEFT 库支持的适配器权重的所有函数的类。有关适配器及如何在基于 Transformer 的模型中注入它们的详细信息，
    请参阅 PEFT 库的文档: https://huggingface.co/docs/peft/index

    当前支持的 PEFT 方法是所有非前缀调整方法。以下是可以使用此混合类加载、训练和运行的支持的 PEFT 方法列表：
    - Low Rank Adapters (LoRA): https://huggingface.co/docs/peft/conceptual_guides/lora
    - IA3: https://huggingface.co/docs/peft/conceptual_guides/ia3
    - AdaLora: https://arxiv.org/abs/2303.10512

    其他 PEFT 模型，如提示调整、提示学习等因其适配器无法“注入”到 torch 模块而不在讨论范围内。要使用这些方法，请参阅 PEFT 库的使用指南。

    使用此混合类，如果安装了正确的 PEFT 版本，可以：
    - 加载存储在本地路径或远程 Hub 存储库中的适配器，并将其注入模型中
    - 在模型中附加新的适配器，并使用 Trainer 或自己的方法进行训练
    - 附加多个适配器并迭代地激活/停用它们
    - 激活/停用模型中的所有适配器
    - 获取激活适配器的 `state_dict`
    """

    _hf_peft_config_loaded = False  # 初始配置标志，指示 PEFT 配置是否已加载
   python
    def load_adapter(
        self,
        peft_model_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        device_map: Optional[str] = "auto",
        max_memory: Optional[str] = None,
        offload_folder: Optional[str] = None,
        offload_index: Optional[int] = None,
        peft_config: Dict[str, Any] = None,
        adapter_state_dict: Optional[Dict[str, "torch.Tensor"]] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Args:
            adapter_config (`~peft.PeftConfig`):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        # 检查 PEFT 版本是否符合最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 导入 PEFT 配置和适配器注入函数
        from peft import PeftConfig, inject_adapter_in_model

        # 如果未提供适配器名称，则使用默认名称 "default"
        adapter_name = adapter_name or "default"

        # 如果 PEFT 配置未加载，则标记为已加载
        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        # 如果同名适配器已存在，则抛出 ValueError 异常
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        # 如果 adapter_config 不是 PeftConfig 的实例，则抛出 ValueError 异常
        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # 获取模型的名称或路径，以保持与 PEFT 中的一致性
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)

        # 将适配器注入到模型中
        inject_adapter_in_model(adapter_config, self, adapter_name)

        # 设置当前模型的适配器
        self.set_adapter(adapter_name)
    def set_adapter(self, adapter_name: Union[List[str], str]) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            adapter_name (`Union[List[str], str]`):
                The name of the adapter to set. Can be also a list of strings to set multiple adapters.
        """
        # 检查 PEFT 的最小版本要求
        check_peft_version(min_version=MIN_PEFT_VERSION)
        
        # 如果尚未加载 PEFT 配置，则引发 ValueError
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        
        # 如果 adapter_name 是一个列表，检查列表中的适配器是否存在于当前配置中
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        
        # 如果 adapter_name 不在当前配置中，引发 ValueError
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        # 导入 PEFT 中必要的模块
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        # 标记是否成功设置了适配器
        _adapters_has_been_set = False

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            # 如果模块是 BaseTunerLayer 或 ModulesToSaveWrapper 的实例
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # 对于兼容旧版 PEFT 的情况，检查是否有 set_adapter 方法
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    # 否则直接设置 active_adapter 属性
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        # 如果没有成功设置适配器，引发 ValueError
        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
    def disable_adapters(self) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        # 检查 PEFT 版本是否符合要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果 PEFT 配置未加载，则抛出数值错误异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入必要的 PEFT 模块
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            # 检查模块是否属于 BaseTunerLayer 或 ModulesToSaveWrapper 类型
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # 如果模块具有 enable_adapters 方法，则调用以禁用适配器
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # 否则，将模块的 disable_adapters 属性设置为 True
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        # 检查 PEFT 版本是否符合要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果 PEFT 配置未加载，则抛出数值错误异常
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入必要的 PEFT 模块
        from peft.tuners.tuners_utils import BaseTunerLayer

        # 遍历模型的所有模块
        for _, module in self.named_modules():
            # 检查模块是否属于 BaseTunerLayer 类型
            if isinstance(module, BaseTunerLayer):
                # 如果模块具有 enable_adapters 方法，则调用以启用适配器
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # 否则，将模块的 disable_adapters 属性设置为 False
                    module.disable_adapters = False
    def active_adapters(self) -> List[str]:
        """
        获取当前模型的活跃适配器列表。如果进行多适配器推理（结合多个适配器进行推理），返回所有活跃适配器的列表，以便用户可以相应处理。

        对于之前版本的 PEFT（不支持多适配器推理），`module.active_adapter` 将返回一个单独的字符串。
        """
        # 检查 PEFT 版本是否符合最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 检查 PEFT 是否可用，如果不可用则抛出 ImportError
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        # 如果没有加载 PEFT 配置，抛出 ValueError
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入 PEFT 的 BaseTunerLayer
        from peft.tuners.tuners_utils import BaseTunerLayer

        # 遍历模型的所有子模块，查找 BaseTunerLayer 类型的模块，获取其活跃适配器
        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapter
                break

        # 对于之前的 PEFT 版本，确保 active_adapters 是列表类型
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        # 返回活跃适配器列表
        return active_adapters

    def active_adapter(self) -> str:
        """
        警告：`active_adapter` 方法已弃用，并将在未来版本中移除。
        """
        # 发出警告：方法已弃用
        warnings.warn(
            "The `active_adapter` method is deprecated and will be removed in a future version.", FutureWarning
        )

        # 返回当前活跃适配器列表中的第一个适配器
        return self.active_adapters()[0]

    def get_adapter_state_dict(self, adapter_name: Optional[str] = None) -> dict:
        """
        获取适配器的状态字典，该字典应仅包含指定适配器名称的权重张量。如果未传适配器名称，则使用活跃适配器。

        Args:
            adapter_name (`str`, *optional*):
                要获取状态字典的适配器名称。如果未传适配器名称，则使用活跃适配器。
        """
        # 检查 PEFT 版本是否符合最低要求
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # 如果没有加载 PEFT 配置，抛出 ValueError
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        # 导入 PEFT 的 get_peft_model_state_dict 函数
        from peft import get_peft_model_state_dict

        # 如果未传适配器名称，使用当前的活跃适配器
        if adapter_name is None:
            adapter_name = self.active_adapter()

        # 获取指定适配器名称的状态字典
        adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
        return adapter_state_dict

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
        # Prepare arguments for dispatching the model
        dispatch_model_kwargs = {}

        # Safety checker for previous `accelerate` versions
        # Check if `offload_index` is supported by the `dispatch_model` function
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        # Get the list of module classes that should not be split during dispatch
        no_split_module_classes = self._no_split_modules

        # Calculate balanced memory allocation if device_map is not "sequential"
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )

        # Infer an automatic device_map if device_map is a string
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )

        # Dispatch the model with the specified parameters
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_folder,
            **dispatch_model_kwargs,
        )
```