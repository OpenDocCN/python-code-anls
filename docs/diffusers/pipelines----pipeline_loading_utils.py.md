# `.\diffusers\pipelines\pipeline_loading_utils.py`

```py
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权声明，说明版权归 HuggingFace Inc. 团队所有
# Copyright 2024 The HuggingFace Inc. team.
#
# 根据 Apache 许可证第 2.0 版进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非遵守许可证，否则不得使用本文件
# you may not use this file except in compliance with the License.
# 可以通过以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则软件在"按原样"基础上分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件，包括明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的库
import importlib
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 导入 PyTorch 库
import torch
# 从 huggingface_hub 导入模型信息函数
from huggingface_hub import model_info
# 导入 Hugging Face Hub 的参数验证工具
from huggingface_hub.utils import validate_hf_hub_args
# 导入版本管理工具
from packaging import version

# 导入当前模块的版本信息
from .. import __version__
# 从 utils 模块导入一些工具函数和常量
from ..utils import (
    FLAX_WEIGHTS_NAME,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_class_from_dynamic_module,
    is_accelerate_available,
    is_peft_available,
    is_transformers_available,
    logging,
)
# 从 torch_utils 模块导入编译模块检查函数
from ..utils.torch_utils import is_compiled_module

# 检查 transformers 库是否可用
if is_transformers_available():
    # 导入 transformers 库
    import transformers
    # 从 transformers 中导入预训练模型基类
    from transformers import PreTrainedModel
    # 导入 transformers 中的权重名称常量
    from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME
    from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME
    from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

# 检查 accelerate 库是否可用
if is_accelerate_available():
    # 导入 accelerate 库
    import accelerate
    # 从 accelerate 中导入模型调度函数
    from accelerate import dispatch_model
    # 导入用于从模块中移除钩子的工具
    from accelerate.hooks import remove_hook_from_module
    # 从 accelerate 中导入计算模块大小和获取最大内存的工具
    from accelerate.utils import compute_module_sizes, get_max_memory

# 定义加载模型时使用的索引文件名
INDEX_FILE = "diffusion_pytorch_model.bin"
# 定义自定义管道文件名
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
# 定义虚拟模块的文件夹路径
DUMMY_MODULES_FOLDER = "diffusers.utils"
# 定义 transformers 虚拟模块的文件夹路径
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"
# 定义连接管道的关键字列表
CONNECTED_PIPES_KEYS = ["prior"]

# 创建一个记录器实例，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义可加载类的字典，映射库名到相应的类和方法
LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
}

# 初始化一个空字典，用于存储所有可导入的类
ALL_IMPORTABLE_CLASSES = {}
# 遍历 LOADABLE_CLASSES 字典中的每个库
for library in LOADABLE_CLASSES:
    # 将指定库中的可加载类更新到所有可导入的类集合中
        ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])
# 检查文件名是否与 safetensors 兼容，返回布尔值
def is_safetensors_compatible(filenames, variant=None, passed_components=None) -> bool:
    """
    检查 safetensors 兼容性：
    - 默认情况下，所有模型使用默认 pytorch 序列化保存，因此我们使用默认 pytorch 文件列表来了解所需的 safetensors 文件。
    - 仅当每个默认 pytorch 文件都有匹配的 safetensors 文件时，模型才与 safetensors 兼容。

    将默认 pytorch 序列化文件名转换为 safetensors 序列化文件名：
    - 对于来自 diffusers 库的模型，仅需将 ".bin" 扩展名替换为 ".safetensors"
    - 对于来自 transformers 库的模型，文件名从 "pytorch_model" 更改为 "model"，并将 ".bin" 扩展名替换为 ".safetensors"
    """
    # 初始化一个空列表，用于存储默认 pytorch 文件名
    pt_filenames = []

    # 初始化一个空集合，用于存储 safetensors 文件名
    sf_filenames = set()

    # 如果未传递组件，则将其设置为空列表
    passed_components = passed_components or []

    # 遍历输入的文件名
    for filename in filenames:
        # 分离文件名和扩展名
        _, extension = os.path.splitext(filename)

        # 如果文件在传递的组件中，跳过处理
        if len(filename.split("/")) == 2 and filename.split("/")[0] in passed_components:
            continue

        # 如果扩展名为 .bin，则添加到 pytorch 文件名列表中
        if extension == ".bin":
            pt_filenames.append(os.path.normpath(filename))
        # 如果扩展名为 .safetensors，则添加到 safetensors 文件名集合中
        elif extension == ".safetensors":
            sf_filenames.add(os.path.normpath(filename))

    # 遍历所有默认 pytorch 文件名
    for filename in pt_filenames:
        # 拆分路径和文件名
        path, filename = os.path.split(filename)
        filename, extension = os.path.splitext(filename)

        # 如果文件名以 "pytorch_model" 开头，则进行替换
        if filename.startswith("pytorch_model"):
            filename = filename.replace("pytorch_model", "model")
        else:
            filename = filename

        # 构建预期的 safetensors 文件名
        expected_sf_filename = os.path.normpath(os.path.join(path, filename))
        expected_sf_filename = f"{expected_sf_filename}.safetensors"
        # 检查预期的 safetensors 文件名是否在集合中
        if expected_sf_filename not in sf_filenames:
            logger.warning(f"{expected_sf_filename} not found")
            return False

    # 如果所有检查通过，返回 True
    return True


# 检查文件名是否与 variant 兼容，返回文件路径列表或字符串
def variant_compatible_siblings(filenames, variant=None) -> Union[List[os.PathLike], str]:
    # 定义一个权重文件名的列表
    weight_names = [
        WEIGHTS_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME,
        ONNX_WEIGHTS_NAME,
        ONNX_EXTERNAL_WEIGHTS_NAME,
    ]

    # 如果 transformers 可用，添加更多权重文件名
    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME, TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]

    # 从权重文件名中提取前缀
    weight_prefixes = [w.split(".")[0] for w in weight_names]
    # 从权重文件名中提取后缀
    weight_suffixs = [w.split(".")[-1] for w in weight_names]
    # 定义 transformers 索引格式的正则表达式
    transformers_index_format = r"\d{5}-of-\d{5}"
    # 如果 variant 不为 None，表示需要处理变体文件
    if variant is not None:
        # 定义一个正则表达式，匹配带有变体的权重文件名
        variant_file_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({variant}|{variant}-{transformers_index_format})\.({'|'.join(weight_suffixs)})$"
        )
        # 定义一个正则表达式，匹配带有变体的索引文件名
        variant_index_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.{variant}\.json$"
        )

    # 定义一个正则表达式，匹配不带变体的权重文件名
    non_variant_file_re = re.compile(
        rf"({'|'.join(weight_prefixes)})(-{transformers_index_format})?\.({'|'.join(weight_suffixs)})$"
    )
    # 定义一个正则表达式，匹配不带变体的索引文件名
    non_variant_index_re = re.compile(rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.json")

    # 如果 variant 不为 None，获取所有变体权重和索引文件名
    if variant is not None:
        variant_weights = {f for f in filenames if variant_file_re.match(f.split("/")[-1]) is not None}
        variant_indexes = {f for f in filenames if variant_index_re.match(f.split("/")[-1]) is not None}
        # 合并变体权重和索引文件名
        variant_filenames = variant_weights | variant_indexes
    else:
        # 如果没有变体，则变体文件名集合为空
        variant_filenames = set()

    # 获取所有不带变体的权重文件名
    non_variant_weights = {f for f in filenames if non_variant_file_re.match(f.split("/")[-1]) is not None}
    # 获取所有不带变体的索引文件名
    non_variant_indexes = {f for f in filenames if non_variant_index_re.match(f.split("/")[-1]) is not None}
    # 合并不带变体的权重和索引文件名
    non_variant_filenames = non_variant_weights | non_variant_indexes

    # 默认情况下使用所有变体文件名
    usable_filenames = set(variant_filenames)

    # 定义一个函数，将文件名转换为对应的变体文件名
    def convert_to_variant(filename):
        # 如果文件名中包含 'index'，则替换为带变体的索引文件名
        if "index" in filename:
            variant_filename = filename.replace("index", f"index.{variant}")
        # 如果文件名符合特定格式，则转换为带变体的文件名
        elif re.compile(f"^(.*?){transformers_index_format}").match(filename) is not None:
            variant_filename = f"{filename.split('-')[0]}.{variant}-{'-'.join(filename.split('-')[1:])}"
        # 否则默认按变体格式修改文件名
        else:
            variant_filename = f"{filename.split('.')[0]}.{variant}.{filename.split('.')[1]}"
        # 返回变体文件名
        return variant_filename

    # 遍历所有不带变体的文件名
    for f in non_variant_filenames:
        # 转换为对应的变体文件名
        variant_filename = convert_to_variant(f)
        # 如果该变体文件名不在可用文件名集合中，则添加
        if variant_filename not in usable_filenames:
            usable_filenames.add(f)

    # 返回可用文件名和变体文件名的集合
    return usable_filenames, variant_filenames
# 装饰器，用于验证 Hugging Face Hub 参数
@validate_hf_hub_args
# 定义一个函数，用于发出关于过时模型变体的警告
def warn_deprecated_model_variant(pretrained_model_name_or_path, token, variant, revision, model_filenames):
    # 获取模型信息，包括预训练模型的路径和其他参数
    info = model_info(
        pretrained_model_name_or_path,
        token=token,
        revision=None,
    )
    # 从模型信息中提取所有文件名
    filenames = {sibling.rfilename for sibling in info.siblings}
    # 获取与指定变体兼容的模型文件名
    comp_model_filenames, _ = variant_compatible_siblings(filenames, variant=revision)
    # 去掉文件名中的版本信息，生成新文件名列表
    comp_model_filenames = [".".join(f.split(".")[:1] + f.split(".")[2:]) for f in comp_model_filenames]

    # 检查给定的模型文件名是否是兼容文件名的子集
    if set(model_filenames).issubset(set(comp_model_filenames)):
        # 发出关于通过修订加载模型变体的警告
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{revision}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
            FutureWarning,
        )
    else:
        # 发出关于不正确加载模型变体的警告，并请求用户报告缺失文件的问题
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added.",
            FutureWarning,
        )


# 定义一个函数，用于解包模型
def _unwrap_model(model):
    """Unwraps a model."""
    # 检查模型是否为编译模块，如果是则解包
    if is_compiled_module(model):
        model = model._orig_mod

    # 检查 PEFT 是否可用
    if is_peft_available():
        from peft import PeftModel

        # 如果模型是 PeftModel 类型，则解包至基础模型
        if isinstance(model, PeftModel):
            model = model.base_model.model

    # 返回解包后的模型
    return model


# 定义一个简单的帮助函数，用于在不正确模块时抛出或发出警告
def maybe_raise_or_warn(
    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
):
    """Simple helper method to raise or warn in case incorrect module has been passed"""
    # 如果当前模块不是管道模块
        if not is_pipeline_module:
            # 动态导入指定的库
            library = importlib.import_module(library_name)
            # 获取库中指定名称的类对象
            class_obj = getattr(library, class_name)
            # 遍历可导入类，构建文件名到类对象的字典
            class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
    
            # 初始化期望的类对象为 None
            expected_class_obj = None
            # 遍历类候选者，检查是否与目标类兼容
            for class_name, class_candidate in class_candidates.items():
                # 如果候选类不为 None 且是目标类的子类
                if class_candidate is not None and issubclass(class_obj, class_candidate):
                    # 将其设置为期望的类对象
                    expected_class_obj = class_candidate
    
            # Dynamo 将原始模型包装在一个私有类中。
            # 没有找到公共 API 获取原始类。
            # 从传入的类对象中获取子模型
            sub_model = passed_class_obj[name]
            # 解包子模型，获取原始模型
            unwrapped_sub_model = _unwrap_model(sub_model)
            # 获取解包后模型的类
            model_cls = unwrapped_sub_model.__class__
    
            # 检查解包模型类是否是期望类的子类
            if not issubclass(model_cls, expected_class_obj):
                # 如果不是，抛出值错误，指明类型不匹配
                raise ValueError(
                    f"{passed_class_obj[name]} is of type: {model_cls}, but should be" f" {expected_class_obj}"
                )
        else:
            # 如果是管道模块，记录警告信息
            logger.warning(
                f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                " has the correct type"
            )
# 定义一个获取类对象和候选类的辅助方法
def get_class_obj_and_candidates(
    # 传入库名、类名、可导入的类、管道、是否为管道模块、组件名及缓存目录
    library_name, class_name, importable_classes, pipelines, is_pipeline_module, component_name=None, cache_dir=None
):
    """简单的辅助方法来检索模块的类对象以及潜在的父类对象"""
    # 构建组件文件夹路径
    component_folder = os.path.join(cache_dir, component_name)

    # 如果是管道模块
    if is_pipeline_module:
        # 获取指定库名的管道模块
        pipeline_module = getattr(pipelines, library_name)

        # 获取指定类对象
        class_obj = getattr(pipeline_module, class_name)
        # 创建类候选字典，键为可导入类名，值为类对象
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    # 如果组件文件存在
    elif os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # 从动态模块加载自定义组件
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        # 创建类候选字典
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        # 否则从库中导入
        library = importlib.import_module(library_name)

        # 获取指定类对象
        class_obj = getattr(library, class_name)
        # 创建类候选字典
        class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

    # 返回类对象及候选类字典
    return class_obj, class_candidates


# 定义一个获取自定义管道类的辅助方法
def _get_custom_pipeline_class(
    # 传入自定义管道及其相关参数
    custom_pipeline,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    # 如果自定义管道是一个文件
    if custom_pipeline.endswith(".py"):
        path = Path(custom_pipeline)
        # 分解为文件夹和文件名
        file_name = path.name
        custom_pipeline = path.parent.absolute()
    # 如果提供了仓库 ID
    elif repo_id is not None:
        file_name = f"{custom_pipeline}.py"
        custom_pipeline = repo_id
    else:
        # 默认文件名
        file_name = CUSTOM_PIPELINE_FILE_NAME

    # 如果提供了仓库 ID 和修订版本
    if repo_id is not None and hub_revision is not None:
        # 从 Hub 加载管道代码时，确保覆盖修订版本
        revision = hub_revision

    # 返回从动态模块获取类
    return get_class_from_dynamic_module(
        custom_pipeline,
        module_file=file_name,
        class_name=class_name,
        cache_dir=cache_dir,
        revision=revision,
    )


# 定义一个获取管道类的辅助方法
def _get_pipeline_class(
    # 传入类对象及其他参数
    class_obj,
    config=None,
    load_connected_pipeline=False,
    custom_pipeline=None,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    # 如果提供了自定义管道
    if custom_pipeline is not None:
        # 调用获取自定义管道类的方法
        return _get_custom_pipeline_class(
            custom_pipeline,
            repo_id=repo_id,
            hub_revision=hub_revision,
            class_name=class_name,
            cache_dir=cache_dir,
            revision=revision,
        )

    # 如果类对象不是 DiffusionPipeline
    if class_obj.__name__ != "DiffusionPipeline":
        # 直接返回类对象
        return class_obj

    # 导入 diffusers 模块
    diffusers_module = importlib.import_module(class_obj.__module__.split(".")[0])
    # 获取类名，如果未提供则从配置中获取
    class_name = class_name or config["_class_name"]
    # 如果类名不存在，抛出错误
    if not class_name:
        raise ValueError(
            "在配置文件中找不到类名。请确保传入正确的 `class_name`。"
        )
    # 如果类名以 "Flax" 开头，则去掉前四个字符，否则保持原样
        class_name = class_name[4:] if class_name.startswith("Flax") else class_name
    
        # 从 diffusers_module 动态获取指定名称的类
        pipeline_cls = getattr(diffusers_module, class_name)
    
        # 如果需要加载连接的管道
        if load_connected_pipeline:
            # 从 auto_pipeline 导入获取连接管道的函数
            from .auto_pipeline import _get_connected_pipeline
    
            # 获取与管道类相关联的连接管道类
            connected_pipeline_cls = _get_connected_pipeline(pipeline_cls)
            # 如果找到了连接的管道类
            if connected_pipeline_cls is not None:
                # 记录加载的连接管道类的信息
                logger.info(
                    f"Loading connected pipeline {connected_pipeline_cls.__name__} instead of {pipeline_cls.__name__} as specified via `load_connected_pipeline=True`"
                )
            else:
                # 记录没有找到连接管道类的信息
                logger.info(f"{pipeline_cls.__name__} has no connected pipeline class. Loading {pipeline_cls.__name__}.")
    
            # 使用找到的连接管道类或保持原管道类
            pipeline_cls = connected_pipeline_cls or pipeline_cls
    
        # 返回最终的管道类
        return pipeline_cls
# 加载一个空模型的函数，接受多种参数
def _load_empty_model(
    library_name: str,  # 库的名称
    class_name: str,  # 类的名称
    importable_classes: List[Any],  # 可导入的类列表
    pipelines: Any,  # 管道相关信息
    is_pipeline_module: bool,  # 是否为管道模块的布尔值
    name: str,  # 模型名称
    torch_dtype: Union[str, torch.dtype],  # Torch 数据类型
    cached_folder: Union[str, os.PathLike],  # 缓存文件夹路径
    **kwargs,  # 其他额外参数
):
    # 检索类对象
    class_obj, _ = get_class_obj_and_candidates(
        library_name,  # 库的名称
        class_name,  # 类的名称
        importable_classes,  # 可导入的类列表
        pipelines,  # 管道相关信息
        is_pipeline_module,  # 是否为管道模块的布尔值
        component_name=name,  # 组件名称
        cache_dir=cached_folder,  # 缓存目录
    )

    # 检查是否可用 transformers 库
    if is_transformers_available():
        # 解析 transformers 的版本号
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        # 如果不可用，版本设置为 "N/A"
        transformers_version = "N/A"

    # 确定库的类型
    is_transformers_model = (
        is_transformers_available()  # 检查 transformers 库是否可用
        and issubclass(class_obj, PreTrainedModel)  # 检查类是否为 PreTrainedModel 的子类
        and transformers_version >= version.parse("4.20.0")  # 检查版本号是否符合要求
    )
    # 导入 diffusers 模块
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    # 检查类是否为 diffusers 模型的子类
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    model = None  # 初始化模型为 None
    config_path = cached_folder  # 设置配置路径为缓存文件夹
    # 设置用户代理信息
    user_agent = {
        "diffusers": __version__,  # 当前 diffusers 的版本
        "file_type": "model",  # 文件类型为模型
        "framework": "pytorch",  # 框架为 PyTorch
    }

    # 如果是 diffusers 模型
    if is_diffusers_model:
        # 加载配置，然后在元信息上加载模型
        config, unused_kwargs, commit_hash = class_obj.load_config(
            os.path.join(config_path, name),  # 配置文件路径
            cache_dir=cached_folder,  # 缓存目录
            return_unused_kwargs=True,  # 返回未使用的关键字参数
            return_commit_hash=True,  # 返回提交哈希值
            force_download=kwargs.pop("force_download", False),  # 强制下载标志
            proxies=kwargs.pop("proxies", None),  # 代理设置
            local_files_only=kwargs.pop("local_files_only", False),  # 仅使用本地文件标志
            token=kwargs.pop("token", None),  # 认证令牌
            revision=kwargs.pop("revision", None),  # 版本修订信息
            subfolder=kwargs.pop("subfolder", None),  # 子文件夹
            user_agent=user_agent,  # 用户代理信息
        )
        # 初始化空权重
        with accelerate.init_empty_weights():
            model = class_obj.from_config(config, **unused_kwargs)  # 从配置中创建模型
    # 如果是 transformers 模型
    elif is_transformers_model:
        config_class = getattr(class_obj, "config_class", None)  # 获取配置类
        # 如果配置类为空，抛出错误
        if config_class is None:
            raise ValueError("`config_class` cannot be None. Please double-check the model.")

        # 从预训练配置加载模型配置
        config = config_class.from_pretrained(
            cached_folder,  # 缓存文件夹
            subfolder=name,  # 子文件夹
            force_download=kwargs.pop("force_download", False),  # 强制下载标志
            proxies=kwargs.pop("proxies", None),  # 代理设置
            local_files_only=kwargs.pop("local_files_only", False),  # 仅使用本地文件标志
            token=kwargs.pop("token", None),  # 认证令牌
            revision=kwargs.pop("revision", None),  # 版本修订信息
            user_agent=user_agent,  # 用户代理信息
        )
        # 初始化空权重
        with accelerate.init_empty_weights():
            model = class_obj(config)  # 使用配置创建模型

    # 如果模型已创建
    if model is not None:
        model = model.to(dtype=torch_dtype)  # 将模型转换为指定的数据类型
    return model  # 返回加载的模型
    # 定义一个包含模块大小的字典，键为模块名称，值为模块大小（以浮点数表示）
        module_sizes: Dict[str, float], 
        # 定义一个包含设备内存的字典，键为设备名称，值为设备内存（以浮点数表示）
        device_memory: Dict[str, float], 
        # 定义设备映射策略的字符串，默认值为 "balanced"
        device_mapping_strategy: str = "balanced"
):
    # 获取设备内存字典的所有设备 ID，并转换为列表
    device_ids = list(device_memory.keys())
    # 创建一个设备循环列表，包含设备 ID 的正序和反序
    device_cycle = device_ids + device_ids[::-1]
    # 复制设备内存字典，以避免修改原始字典
    device_memory = device_memory.copy()

    # 初始化设备 ID 和组件的映射字典
    device_id_component_mapping = {}
    # 当前设备索引，初始化为 0
    current_device_index = 0
    # 遍历模块大小字典
    for component in module_sizes:
        # 根据当前索引获取对应的设备 ID
        device_id = device_cycle[current_device_index % len(device_cycle)]
        # 获取当前组件所需的内存大小
        component_memory = module_sizes[component]
        # 获取当前设备的可用内存
        curr_device_memory = device_memory[device_id]

        # 如果 GPU 的内存不足以容纳当前组件，则将其转移到 CPU
        if component_memory > curr_device_memory:
            device_id_component_mapping["cpu"] = [component]
        else:
            # 如果设备 ID 不在映射中，则初始化该设备的组件列表
            if device_id not in device_id_component_mapping:
                device_id_component_mapping[device_id] = [component]
            else:
                # 如果设备 ID 已存在，则将组件添加到该设备的组件列表
                device_id_component_mapping[device_id].append(component)

            # 更新设备的剩余内存
            device_memory[device_id] -= component_memory
            # 移动到下一个设备索引
            current_device_index += 1

    # 返回设备 ID 到组件的映射字典
    return device_id_component_mapping


def _get_final_device_map(device_map, pipeline_class, passed_class_obj, init_dict, library, max_memory, **kwargs):
    # 为了避免循环导入问题，导入管道模块
    from diffusers import pipelines

    # 从关键字参数中获取 torch 数据类型，默认值为 torch.float32
    torch_dtype = kwargs.get("torch_dtype", torch.float32)

    # 在一个元设备上加载管道中的每个模块，以便推导出设备映射
    init_empty_modules = {}
    # 遍历初始化字典中的每个名称及其对应的库名称和类名称
        for name, (library_name, class_name) in init_dict.items():
            # 如果类名以 "Flax" 开头，抛出不支持的错误
            if class_name.startswith("Flax"):
                raise ValueError("Flax pipelines are not supported with `device_map`.")
    
            # 定义所有可导入的类
            is_pipeline_module = hasattr(pipelines, library_name)  # 检查 pipelines 是否有对应的库名称
            importable_classes = ALL_IMPORTABLE_CLASSES  # 获取所有可导入类的集合
            loaded_sub_model = None  # 初始化已加载的子模型为 None
    
            # 使用传入的子模型或从库名称加载类名称
            if name in passed_class_obj:  # 如果传入的类对象中有该名称
                # 如果模型在管道模块中，则从管道加载模型
                # 检查传入的类对象是否具有正确的父类
                maybe_raise_or_warn(
                    library_name,
                    library,
                    class_name,
                    importable_classes,
                    passed_class_obj,
                    name,
                    is_pipeline_module,
                )
                with accelerate.init_empty_weights():  # 在初始化空权重的上下文中
                    loaded_sub_model = passed_class_obj[name]  # 从传入的对象中加载模型
    
            else:
                # 加载空模型
                loaded_sub_model = _load_empty_model(
                    library_name=library_name,  # 库名称
                    class_name=class_name,  # 类名称
                    importable_classes=importable_classes,  # 可导入类集合
                    pipelines=pipelines,  # 管道
                    is_pipeline_module=is_pipeline_module,  # 是否为管道模块
                    pipeline_class=pipeline_class,  # 管道类
                    name=name,  # 名称
                    torch_dtype=torch_dtype,  # torch 数据类型
                    cached_folder=kwargs.get("cached_folder", None),  # 缓存文件夹
                    force_download=kwargs.get("force_download", None),  # 强制下载标志
                    proxies=kwargs.get("proxies", None),  # 代理设置
                    local_files_only=kwargs.get("local_files_only", None),  # 仅限本地文件标志
                    token=kwargs.get("token", None),  # 访问令牌
                    revision=kwargs.get("revision", None),  # 版本修订号
                )
    
            # 如果已加载子模型不为 None，将其添加到初始化空模块中
            if loaded_sub_model is not None:
                init_empty_modules[name] = loaded_sub_model
    
        # 确定设备映射
        # 获取一个按大小排序的字典，用于映射模型级组件
        # 到其大小。
        module_sizes = {
            module_name: compute_module_sizes(module, dtype=torch_dtype)[""]  # 计算每个模块的大小
            for module_name, module in init_empty_modules.items()  # 遍历初始化空模块中的每个模块
            if isinstance(module, torch.nn.Module)  # 仅考虑 PyTorch 模块
        }
        # 对模块大小字典进行排序，按值降序排列
        module_sizes = dict(sorted(module_sizes.items(), key=lambda item: item[1], reverse=True))
    
        # 获取每个设备（仅限 GPU）的最大可用内存
        max_memory = get_max_memory(max_memory)  # 获取最大内存信息
        # 对最大内存字典进行排序，按值降序排列
        max_memory = dict(sorted(max_memory.items(), key=lambda item: item[1], reverse=True))
        # 从最大内存字典中移除 CPU 条目
        max_memory = {k: v for k, v in max_memory.items() if k != "cpu"}
    
        # 获取一个字典，用于将模型级组件映射到基于最大内存和模型大小的可用设备
        final_device_map = None  # 初始化最终设备映射为 None
    # 检查最大内存的长度是否大于 0
        if len(max_memory) > 0:
            # 分配组件到设备，并返回设备与组件的映射
            device_id_component_mapping = _assign_components_to_devices(
                module_sizes, max_memory, device_mapping_strategy=device_map
            )
    
            # 初始化最终设备映射字典
            final_device_map = {}
            # 遍历设备 ID 和其对应的组件
            for device_id, components in device_id_component_mapping.items():
                # 遍历每个组件，将其映射到设备 ID
                for component in components:
                    final_device_map[component] = device_id
    
        # 返回最终的设备映射字典
        return final_device_map
# 定义一个加载子模型的辅助方法，接受多个参数来配置模型加载
def load_sub_model(
    # 模型所在库的名称
    library_name: str,
    # 模型类的名称
    class_name: str,
    # 可导入类的列表
    importable_classes: List[Any],
    # 管道相关参数
    pipelines: Any,
    # 是否为管道模块的标志
    is_pipeline_module: bool,
    # 管道类
    pipeline_class: Any,
    # 指定的 torch 数据类型
    torch_dtype: torch.dtype,
    # 提供者参数
    provider: Any,
    # 会话选项
    sess_options: Any,
    # 设备映射，可能是字典或字符串
    device_map: Optional[Union[Dict[str, torch.device], str]],
    # 最大内存使用配置
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]],
    # 离线文件夹路径
    offload_folder: Optional[Union[str, os.PathLike]],
    # 是否离线保存状态字典
    offload_state_dict: bool,
    # 模型变体的字典
    model_variants: Dict[str, str],
    # 模型名称
    name: str,
    # 是否从 Flax 框架加载
    from_flax: bool,
    # 模型变体名称
    variant: str,
    # 是否使用低 CPU 内存使用模式
    low_cpu_mem_usage: bool,
    # 缓存文件夹路径
    cached_folder: Union[str, os.PathLike],
):
    """从指定库和类名加载模块 `name` 的辅助方法"""

    # 获取类对象及候选类列表
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # 获取加载方法名称
    for class_name, class_candidate in class_candidates.items():
        # 如果候选类不为 None，且 class_obj 是其子类
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            # 从可导入类中获取加载方法名称
            load_method_name = importable_classes[class_name][1]

    # 如果加载方法名称为 None，说明是一个虚拟模块 -> 抛出错误
    if load_method_name is None:
        none_module = class_obj.__module__
        # 检查模块路径是否属于虚拟模块
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
            TRANSFORMERS_DUMMY_MODULES_FOLDER
        )
        # 如果是虚拟模块，调用 class_obj 以获得友好的错误信息
        if is_dummy_path and "dummy" in none_module:
            class_obj()

        # 抛出值错误，说明没有定义任何加载方法
        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    # 获取加载方法
    load_method = getattr(class_obj, load_method_name)

    # 为加载方法添加关键字参数
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    loading_kwargs = {}
    # 如果类是 PyTorch 模块，则添加 torch_dtype 参数
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    # 如果类是 OnnxRuntimeModel，则添加 provider 和 sess_options 参数
    if issubclass(class_obj, diffusers_module.OnnxRuntimeModel):
        loading_kwargs["provider"] = provider
        loading_kwargs["sess_options"] = sess_options

    # 检查类是否为 diffusers 模型
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    # 检查 transformers 是否可用
    if is_transformers_available():
        # 获取 transformers 的版本
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        # 如果不可用，则版本标记为 "N/A"
        transformers_version = "N/A"

    # 检查类是否为 transformers 模型，并且版本满足要求
    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # 加载 transformers 模型时，如果 device_map 为 None，则权重将被初始化，而不是 diffusers.
    # 为了加快默认加载速度，设置 `low_cpu_mem_usage=low_cpu_mem_usage` 标志，默认为 `True`。
    # 确保权重不会被初始化，从而显著加快加载速度。
    if is_diffusers_model or is_transformers_model:
        # 将设备映射添加到加载参数中
        loading_kwargs["device_map"] = device_map
        # 设置最大内存使用限制
        loading_kwargs["max_memory"] = max_memory
        # 设置用于卸载的文件夹
        loading_kwargs["offload_folder"] = offload_folder
        # 设置卸载状态字典的标志
        loading_kwargs["offload_state_dict"] = offload_state_dict
        # 从模型变体中获取对应的变体
        loading_kwargs["variant"] = model_variants.pop(name, None)

        if from_flax:
            # 如果模型来自 Flax，设置标志
            loading_kwargs["from_flax"] = True

        # 以下内容可以在 `transformers` 的最低版本高于 4.27 时删除
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            # 如果版本不符合要求，抛出导入错误
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            # 如果变体为空，从加载参数中移除变体
            loading_kwargs.pop("variant")

        # 如果来自 Flax 并且模型是变换器模型，无法使用 `low_cpu_mem_usage` 加载
        if not (from_flax and is_transformers_model):
            # 设置低 CPU 内存使用标志
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            # 否则将标志设置为 False
            loading_kwargs["low_cpu_mem_usage"] = False

    # 检查模块是否在子目录中
    if os.path.isdir(os.path.join(cached_folder, name)):
        # 从子目录加载模型
        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
    else:
        # 否则从根目录加载模型
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    if isinstance(loaded_sub_model, torch.nn.Module) and isinstance(device_map, dict):
        # 移除模型中的钩子
        remove_hook_from_module(loaded_sub_model, recurse=True)
        # 检查是否需要将模型卸载到 CPU
        needs_offloading_to_cpu = device_map[""] == "cpu"

        if needs_offloading_to_cpu:
            # 如果需要卸载，将模型分发到 CPU
            dispatch_model(
                loaded_sub_model,
                state_dict=loaded_sub_model.state_dict(),
                device_map=device_map,
                force_hooks=True,
                main_device=0,
            )
        else:
            # 否则正常分发模型
            dispatch_model(loaded_sub_model, device_map=device_map, force_hooks=True)

    # 返回加载的子模型
    return loaded_sub_model
# 根据传入模块获取类库名称和类名的元组
def _fetch_class_library_tuple(module):
    # 在这里导入模块以避免循环导入问题
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    # 获取 diffusers_module 中的 pipelines 属性
    pipelines = getattr(diffusers_module, "pipelines")

    # 从原始模块注册配置，而不是从动态编译的模块
    not_compiled_module = _unwrap_model(module)
    # 获取模块的库名称
    library = not_compiled_module.__module__.split(".")[0]

    # 检查该模块是否为管道模块
    module_path_items = not_compiled_module.__module__.split(".")
    # 获取模块路径倒数第二项，作为管道目录
    pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

    # 获取模块路径，并检查是否为管道模块
    path = not_compiled_module.__module__.split(".")
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

    # 如果库名称不在 LOADABLE_CLASSES 中，则认为它是自定义模块
    # 或者如果是管道模块，则库名称设为模块名称
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in LOADABLE_CLASSES:
        library = not_compiled_module.__module__

    # 获取类名
    class_name = not_compiled_module.__class__.__name__

    # 返回库名称和类名的元组
    return (library, class_name)
```