# `.\diffusers\loaders\single_file.py`

```py
# 版权所有 2024 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 按“原样”分发，没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言的权限和
# 限制。
import importlib  # 导入用于动态导入模块的库
import inspect  # 导入用于检查对象的库
import os  # 导入与操作系统交互的库

import torch  # 导入 PyTorch 库
from huggingface_hub import snapshot_download  # 从 Hugging Face Hub 下载快照
from huggingface_hub.utils import LocalEntryNotFoundError, validate_hf_hub_args  # 导入特定异常和验证函数
from packaging import version  # 导入用于处理版本的库

from ..utils import deprecate, is_transformers_available, logging  # 从上级模块导入工具函数
from .single_file_utils import (  # 从当前模块导入多个单文件相关的工具函数和类
    SingleFileComponentError,
    _is_legacy_scheduler_kwargs,
    _is_model_weights_in_cached_folder,
    _legacy_load_clip_tokenizer,
    _legacy_load_safety_checker,
    _legacy_load_scheduler,
    create_diffusers_clip_model_from_ldm,
    create_diffusers_t5_model_from_checkpoint,
    fetch_diffusers_config,
    fetch_original_config,
    is_clip_model_in_single_file,
    is_t5_in_single_file,
    load_single_file_checkpoint,
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 旧版行为。`from_single_file` 不会加载安全检查器，除非明确提供
SINGLE_FILE_OPTIONAL_COMPONENTS = ["safety_checker"]

if is_transformers_available():  # 检查 transformers 库是否可用
    import transformers  # 导入 transformers 库
    from transformers import PreTrainedModel, PreTrainedTokenizer  # 导入预训练模型和分词器类


def load_single_file_sub_model(  # 定义加载单文件子模型的函数
    library_name,  # 库名称
    class_name,  # 类名称
    name,  # 模型名称
    checkpoint,  # 检查点
    pipelines,  # 管道
    is_pipeline_module,  # 是否为管道模块
    cached_model_config_path,  # 缓存模型配置路径
    original_config=None,  # 原始配置（可选）
    local_files_only=False,  # 是否仅使用本地文件
    torch_dtype=None,  # PyTorch 数据类型
    is_legacy_loading=False,  # 是否为旧版加载
    **kwargs,  # 其他参数
):
    if is_pipeline_module:  # 如果是管道模块
        pipeline_module = getattr(pipelines, library_name)  # 从管道中获取指定库的模块
        class_obj = getattr(pipeline_module, class_name)  # 获取指定类
    else:  # 否则从库中导入
        library = importlib.import_module(library_name)  # 动态导入库
        class_obj = getattr(library, class_name)  # 获取指定类

    if is_transformers_available():  # 检查 transformers 库是否可用
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)  # 解析 transformers 版本
    else:  # 如果不可用
        transformers_version = "N/A"  # 设置版本为不可用

    is_transformers_model = (  # 检查是否为 transformers 模型
        is_transformers_available()  # transformers 可用
        and issubclass(class_obj, PreTrainedModel)  # 是预训练模型的子类
        and transformers_version >= version.parse("4.20.0")  # 版本不低于 4.20.0
    )
    is_tokenizer = (  # 检查是否为分词器
        is_transformers_available()  # transformers 可用
        and issubclass(class_obj, PreTrainedTokenizer)  # 是预训练分词器的子类
        and transformers_version >= version.parse("4.20.0")  # 版本不低于 4.20.0
    )

    diffusers_module = importlib.import_module(__name__.split(".")[0])  # 动态导入当前模块的上级模块
    is_diffusers_single_file_model = issubclass(class_obj, diffusers_module.FromOriginalModelMixin)  # 检查是否为 diffusers 单文件模型的子类
    # 检查类对象是否是 diffusers_module.ModelMixin 的子类
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    # 检查类对象是否是 diffusers_module.SchedulerMixin 的子类
    is_diffusers_scheduler = issubclass(class_obj, diffusers_module.SchedulerMixin)
    
    # 如果是单文件模型
    if is_diffusers_single_file_model:
        # 获取类对象的 from_single_file 方法
        load_method = getattr(class_obj, "from_single_file")
    
        # 如果提供了 original_config，则不能同时使用 cached_model_config_path
        if original_config:
            # 忽略加载缓存的模型配置路径
            cached_model_config_path = None
    
        # 调用 from_single_file 方法加载子模型
        loaded_sub_model = load_method(
            pretrained_model_link_or_path_or_dict=checkpoint,  # 加载预训练模型链接或路径或字典
            original_config=original_config,  # 原始配置
            config=cached_model_config_path,  # 缓存的模型配置路径
            subfolder=name,  # 子文件夹名称
            torch_dtype=torch_dtype,  # Torch 数据类型
            local_files_only=local_files_only,  # 仅加载本地文件
            **kwargs,  # 其他参数
        )
    
    # 如果是 transformers 模型且是单文件中的 CLIP 模型
    elif is_transformers_model and is_clip_model_in_single_file(class_obj, checkpoint):
        # 从 LDM 创建 diffusers CLIP 模型
        loaded_sub_model = create_diffusers_clip_model_from_ldm(
            class_obj,  # 类对象
            checkpoint=checkpoint,  # 检查点
            config=cached_model_config_path,  # 缓存的模型配置路径
            subfolder=name,  # 子文件夹名称
            torch_dtype=torch_dtype,  # Torch 数据类型
            local_files_only=local_files_only,  # 仅加载本地文件
            is_legacy_loading=is_legacy_loading,  # 是否为遗留加载
        )
    
    # 如果是 transformers 模型且检查点是单文件中的 T5 模型
    elif is_transformers_model and is_t5_in_single_file(checkpoint):
        # 从检查点创建 diffusers T5 模型
        loaded_sub_model = create_diffusers_t5_model_from_checkpoint(
            class_obj,  # 类对象
            checkpoint=checkpoint,  # 检查点
            config=cached_model_config_path,  # 缓存的模型配置路径
            subfolder=name,  # 子文件夹名称
            torch_dtype=torch_dtype,  # Torch 数据类型
            local_files_only=local_files_only,  # 仅加载本地文件
        )
    
    # 如果是 tokenizer 并且在遗留加载状态
    elif is_tokenizer and is_legacy_loading:
        # 从检查点加载遗留 CLIP tokenizer
        loaded_sub_model = _legacy_load_clip_tokenizer(
            class_obj,  # 类对象
            checkpoint=checkpoint,  # 检查点
            config=cached_model_config_path,  # 缓存的模型配置路径
            local_files_only=local_files_only  # 仅加载本地文件
        )
    
    # 如果是 diffusers scheduler 且处于遗留加载状态或参数为遗留 scheduler 的关键字
    elif is_diffusers_scheduler and (is_legacy_loading or _is_legacy_scheduler_kwargs(kwargs)):
        # 加载遗留调度器
        loaded_sub_model = _legacy_load_scheduler(
            class_obj,  # 类对象
            checkpoint=checkpoint,  # 检查点
            component_name=name,  # 组件名称
            original_config=original_config,  # 原始配置
            **kwargs  # 其他参数
        )
    else:  # 处理非预期条件的情况
        # 检查 class_obj 是否具有 from_pretrained 方法
        if not hasattr(class_obj, "from_pretrained"):
            # 如果没有，抛出值错误，提示加载方法不支持
            raise ValueError(
                (
                    f"The component {class_obj.__name__} cannot be loaded as it does not seem to have"
                    " a supported loading method."
                )
            )

        loading_kwargs = {}  # 初始化加载参数的字典
        # 更新加载参数字典，添加预训练模型路径和其他配置
        loading_kwargs.update(
            {
                "pretrained_model_name_or_path": cached_model_config_path,  # 预训练模型路径
                "subfolder": name,  # 子文件夹名称
                "local_files_only": local_files_only,  # 仅加载本地文件的标志
            }
        )

        # Schedulers 和 Tokenizers 不使用 torch_dtype
        # 因此跳过将其传递给这些对象
        if issubclass(class_obj, torch.nn.Module):  # 检查 class_obj 是否是 torch.nn.Module 的子类
            loading_kwargs.update({"torch_dtype": torch_dtype})  # 如果是，添加 torch_dtype 到加载参数

        # 检查是否为 diffusers 或 transformers 模型
        if is_diffusers_model or is_transformers_model:
            # 检查权重文件是否存在于缓存文件夹中
            if not _is_model_weights_in_cached_folder(cached_model_config_path, name):
                # 如果权重缺失，抛出错误
                raise SingleFileComponentError(
                    f"Failed to load {class_name}. Weights for this component appear to be missing in the checkpoint."
                )

        # 获取 class_obj 的 from_pretrained 方法
        load_method = getattr(class_obj, "from_pretrained")
        # 调用 from_pretrained 方法并传入加载参数，加载子模型
        loaded_sub_model = load_method(**loading_kwargs)

    return loaded_sub_model  # 返回加载的子模型
# 映射组件类型到配置字典
def _map_component_types_to_config_dict(component_types):
    # 导入当前模块的主模块
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    # 初始化配置字典
    config_dict = {}
    # 从组件类型中移除 'self' 键
    component_types.pop("self", None)

    # 检查 transformers 库是否可用
    if is_transformers_available():
        # 解析 transformers 版本的基版本
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        # 如果不可用，则版本设置为 "N/A"
        transformers_version = "N/A"

    # 遍历组件名称和对应的值
    for component_name, component_value in component_types.items():
        # 检查组件值是否为 diffusers 模型的子类
        is_diffusers_model = issubclass(component_value[0], diffusers_module.ModelMixin)
        # 检查组件值是否为 KarrasDiffusionSchedulers 枚举
        is_scheduler_enum = component_value[0].__name__ == "KarrasDiffusionSchedulers"
        # 检查组件值是否为调度器的子类
        is_scheduler = issubclass(component_value[0], diffusers_module.SchedulerMixin)

        # 检查组件值是否为 transformers 模型
        is_transformers_model = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedModel)
            and transformers_version >= version.parse("4.20.0")
        )
        # 检查组件值是否为 transformers 分词器
        is_transformers_tokenizer = (
            is_transformers_available()
            and issubclass(component_value[0], PreTrainedTokenizer)
            and transformers_version >= version.parse("4.20.0")
        )

        # 如果是 diffusers 模型且不在单文件可选组件中
        if is_diffusers_model and component_name not in SINGLE_FILE_OPTIONAL_COMPONENTS:
            # 将组件名称和模型名称添加到配置字典
            config_dict[component_name] = ["diffusers", component_value[0].__name__]

        # 如果是调度器枚举或调度器
        elif is_scheduler_enum or is_scheduler:
            # 如果是调度器枚举，默认使用 DDIMScheduler
            if is_scheduler_enum:
                # 因为无法从 hub 获取调度器配置，默认使用 DDIMScheduler
                config_dict[component_name] = ["diffusers", "DDIMScheduler"]

            # 如果是调度器
            elif is_scheduler:
                config_dict[component_name] = ["diffusers", component_value[0].__name__]

        # 如果是 transformers 模型或分词器且不在单文件可选组件中
        elif (
            is_transformers_model or is_transformers_tokenizer
        ) and component_name not in SINGLE_FILE_OPTIONAL_COMPONENTS:
            # 将组件名称和模型名称添加到配置字典
            config_dict[component_name] = ["transformers", component_value[0].__name__]

        # 否则设置为 None
        else:
            config_dict[component_name] = [None, None]

    # 返回配置字典
    return config_dict


# 推断管道配置字典
def _infer_pipeline_config_dict(pipeline_class):
    # 获取管道类初始化方法的参数
    parameters = inspect.signature(pipeline_class.__init__).parameters
    # 收集所有必需的参数
    required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
    # 获取管道类的组件类型
    component_types = pipeline_class._get_signature_types()

    # 忽略非必需参数的组件类型
    component_types = {k: v for k, v in component_types.items() if k in required_parameters}
    # 映射组件类型到配置字典
    config_dict = _map_component_types_to_config_dict(component_types)

    # 返回配置字典
    return config_dict


# 从 hub 下载 diffusers 模型配置
def _download_diffusers_model_config_from_hub(
    pretrained_model_name_or_path,
    cache_dir,
    revision,
    proxies,
    force_download=None,
    local_files_only=None,
    token=None,
):
    # 定义允许的文件模式
    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]
    # 下载预训练模型的快照，并将其缓存到指定目录
        cached_model_path = snapshot_download(
            # 指定要下载的预训练模型名称或路径
            pretrained_model_name_or_path,
            # 指定缓存目录
            cache_dir=cache_dir,
            # 指定版本修订
            revision=revision,
            # 代理设置
            proxies=proxies,
            # 是否强制下载，即使缓存中已有
            force_download=force_download,
            # 是否仅使用本地文件
            local_files_only=local_files_only,
            # 访问令牌
            token=token,
            # 允许的文件模式
            allow_patterns=allow_patterns,
        )
    
    # 返回缓存模型的路径
        return cached_model_path
# 定义一个名为 FromSingleFileMixin 的类
class FromSingleFileMixin:
    """
    加载以 `.ckpt` 格式保存的模型权重到 [`DiffusionPipeline`] 中。
    """

    # 定义一个类方法，通常用于从类中直接调用
    @classmethod
    # 装饰器，用于验证与 Hugging Face Hub 相关的参数
    @validate_hf_hub_args
```