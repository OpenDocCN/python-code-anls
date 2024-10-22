# `.\diffusers\utils\dynamic_modules_utils.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 版权声明，指明版权所有者
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 指明许可证类型
# you may not use this file except in compliance with the License.  # 说明使用条件
# You may obtain a copy of the License at  # 提供获取许可证的链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的具体链接
#
# Unless required by applicable law or agreed to in writing, software  # 说明免责条款
# distributed under the License is distributed on an "AS IS" BASIS,  # 指出软件按原样分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 声明无任何保证或条件
# See the License for the specific language governing permissions and  # 提示查看许可证以获取权限说明
# limitations under the License.  # 说明许可证的限制
"""Utilities to dynamically load objects from the Hub."""  # 模块描述，说明功能

import importlib  # 导入模块以动态导入其他模块
import inspect  # 导入模块以检查对象的类型和属性
import json  # 导入模块以处理 JSON 数据
import os  # 导入模块以进行操作系统相关的操作
import re  # 导入模块以进行正则表达式匹配
import shutil  # 导入模块以进行文件和目录操作
import sys  # 导入模块以访问解释器使用的变量和函数
from pathlib import Path  # 从路径模块导入 Path 类以处理文件路径
from typing import Dict, Optional, Union  # 从 typing 模块导入类型注解
from urllib import request  # 导入模块以进行 URL 请求

from huggingface_hub import hf_hub_download, model_info  # 从 huggingface_hub 导入特定功能
from huggingface_hub.utils import RevisionNotFoundError, validate_hf_hub_args  # 导入错误和验证函数
from packaging import version  # 导入版本管理模块

from .. import __version__  # 从上级模块导入当前版本
from . import DIFFUSERS_DYNAMIC_MODULE_NAME, HF_MODULES_CACHE, logging  # 导入当前包中的常量和日志模块


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name  # 初始化日志记录器
# See https://huggingface.co/datasets/diffusers/community-pipelines-mirror  # 提供社区管道镜像的链接
COMMUNITY_PIPELINES_MIRROR_ID = "diffusers/community-pipelines-mirror"  # 定义社区管道镜像的 ID


def get_diffusers_versions():  # 定义获取 diffusers 版本的函数
    url = "https://pypi.org/pypi/diffusers/json"  # 设置获取版本信息的 URL
    releases = json.loads(request.urlopen(url).read())["releases"].keys()  # 请求 URL 并解析 JSON，获取版本键
    return sorted(releases, key=lambda x: version.Version(x))  # 返回按版本排序的版本列表


def init_hf_modules():  # 定义初始化 HF 模块的函数
    """
    Creates the cache directory for modules with an init, and adds it to the Python path.
    """  # 函数说明，创建缓存目录并添加到 Python 路径
    # This function has already been executed if HF_MODULES_CACHE already is in the Python path.  # 如果缓存目录已在路径中，直接返回
    if HF_MODULES_CACHE in sys.path:  # 检查缓存目录是否在 Python 路径中
        return  # 如果在，则退出函数

    sys.path.append(HF_MODULES_CACHE)  # 将缓存目录添加到 Python 路径
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)  # 创建缓存目录，如果已存在则不报错
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"  # 定义缓存目录中初始化文件的路径
    if not init_path.exists():  # 检查初始化文件是否存在
        init_path.touch()  # 如果不存在，则创建初始化文件


def create_dynamic_module(name: Union[str, os.PathLike]):  # 定义创建动态模块的函数，接受字符串或路径对象
    """
    Creates a dynamic module in the cache directory for modules.
    """  # 函数说明，创建动态模块
    init_hf_modules()  # 调用初始化函数以确保缓存目录存在
    dynamic_module_path = Path(HF_MODULES_CACHE) / name  # 定义动态模块的路径
    # If the parent module does not exist yet, recursively create it.  # 如果父模块不存在，则递归创建
    if not dynamic_module_path.parent.exists():  # 检查父模块路径是否存在
        create_dynamic_module(dynamic_module_path.parent)  # 如果不存在，则递归调用创建函数
    os.makedirs(dynamic_module_path, exist_ok=True)  # 创建动态模块目录，如果已存在则不报错
    init_path = dynamic_module_path / "__init__.py"  # 定义动态模块中初始化文件的路径
    if not init_path.exists():  # 检查初始化文件是否存在
        init_path.touch()  # 如果不存在，则创建初始化文件


def get_relative_imports(module_file):  # 定义获取相对导入的函数，接受模块文件路径
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.  # 函数说明，接受字符串或路径对象作为参数
    """
    with open(module_file, "r", encoding="utf-8") as f:  # 以 UTF-8 编码打开模块文件
        content = f.read()  # 读取文件内容

    # Imports of the form `import .xxx`  # 说明以下是相对导入的处理
    # 使用正则表达式查找以相对导入形式书写的模块名
        relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
        # 查找以相对导入形式书写的具体从属模块名
        relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
        # 将结果去重，确保唯一性
        return list(set(relative_imports))
# 获取给定模块所需的所有文件列表，包括相对导入的文件
def get_relative_import_files(module_file):
    # 初始化一个标志，用于控制递归循环
    no_change = False
    # 存储待检查的文件列表，初始为传入的模块文件
    files_to_check = [module_file]
    # 存储所有找到的相对导入文件
    all_relative_imports = []

    # 递归遍历所有相对导入文件
    while not no_change:
        # 存储新发现的导入文件
        new_imports = []
        # 遍历待检查的文件列表
        for f in files_to_check:
            # 获取当前文件的相对导入，并添加到新导入列表中
            new_imports.extend(get_relative_imports(f))

        # 获取当前模块文件的目录路径
        module_path = Path(module_file).parent
        # 将新导入的模块文件转为绝对路径
        new_import_files = [str(module_path / m) for m in new_imports]
        # 过滤掉已经找到的导入文件
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]
        # 更新待检查的文件列表，添加 .py 后缀
        files_to_check = [f"{f}.py" for f in new_import_files]

        # 检查是否有新导入文件，如果没有，则结束循环
        no_change = len(new_import_files) == 0
        # 将当前待检查文件加入所有相对导入列表
        all_relative_imports.extend(files_to_check)

    # 返回所有找到的相对导入文件
    return all_relative_imports


# 检查当前 Python 环境是否包含文件中导入的所有库
def check_imports(filename):
    # 以 UTF-8 编码打开指定文件并读取内容
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # 正则表达式查找 `import xxx` 形式的导入
    imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # 正则表达式查找 `from xxx import yyy` 形式的导入
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # 仅保留顶级模块，过滤掉相对导入
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]

    # 去重并确保导入模块的唯一性
    imports = list(set(imports))
    # 存储缺失的包列表
    missing_packages = []
    # 遍历每个导入模块并尝试导入
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            # 如果导入失败，记录缺失的包
            missing_packages.append(imp)

    # 如果有缺失的包，抛出 ImportError 异常并提示用户
    if len(missing_packages) > 0:
        raise ImportError(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    # 返回文件的所有相对导入文件
    return get_relative_imports(filename)


# 从模块缓存中导入指定的类
def get_class_in_module(class_name, module_path):
    # 将模块路径中的分隔符替换为点，以便导入
    module_path = module_path.replace(os.path.sep, ".")
    # 导入指定模块
    module = importlib.import_module(module_path)

    # 如果类名为空，查找管道类
    if class_name is None:
        return find_pipeline_class(module)
    # 返回指定类的引用
    return getattr(module, class_name)


# 获取继承自 `DiffusionPipeline` 的管道类
def find_pipeline_class(loaded_module):
    # 从上级导入 DiffusionPipeline 类
    from ..pipelines import DiffusionPipeline

    # 获取加载模块中所有的类成员
    cls_members = dict(inspect.getmembers(loaded_module, inspect.isclass))

    # 初始化管道类变量
    pipeline_class = None
    # 遍历 cls_members 字典中的每个类名及其对应的类
    for cls_name, cls in cls_members.items():
        # 检查类名不是 DiffusionPipeline 的名称，且是其子类，且模块不是 diffusers
        if (
            cls_name != DiffusionPipeline.__name__
            and issubclass(cls, DiffusionPipeline)
            and cls.__module__.split(".")[0] != "diffusers"
        ):
            # 如果已经找到一个管道类，则抛出值错误，表示发现多个类
            if pipeline_class is not None:
                raise ValueError(
                    # 错误信息，包含找到的多个类的信息
                    f"Multiple classes that inherit from {DiffusionPipeline.__name__} have been found:"
                    f" {pipeline_class.__name__}, and {cls_name}. Please make sure to define only one in"
                    f" {loaded_module}."
                )
            # 记录找到的管道类
            pipeline_class = cls

    # 返回找到的管道类
    return pipeline_class
# 装饰器，用于验证传入的参数是否符合预期
@validate_hf_hub_args
# 定义获取缓存模块文件的函数，接受多个参数
def get_cached_module_file(
    # 预训练模型名称或路径，可以是字符串或路径类型
    pretrained_model_name_or_path: Union[str, os.PathLike],
    # 模块文件名称，字符串类型
    module_file: str,
    # 缓存目录，可选参数，路径类型或字符串
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    # 强制下载标志，可选参数，布尔类型，默认为 False
    force_download: bool = False,
    # 代理服务器字典，可选参数
    proxies: Optional[Dict[str, str]] = None,
    # 授权令牌，可选参数，可以是布尔或字符串类型
    token: Optional[Union[bool, str]] = None,
    # 版本修订信息，可选参数，字符串类型
    revision: Optional[str] = None,
    # 仅使用本地文件标志，可选参数，布尔类型，默认为 False
    local_files_only: bool = False,
):
    """
    准备从本地文件夹或远程仓库下载模块，并返回其在缓存中的路径。
    
    参数:
        pretrained_model_name_or_path (`str` 或 `os.PathLike`):
            可以是预训练模型配置的模型 ID 或包含配置文件的目录路径。
        module_file (`str`):
            包含要查找的类的模块文件名称。
        cache_dir (`str` 或 `os.PathLike`, *可选*):
            下载的预训练模型配置的缓存目录路径。
        force_download (`bool`, *可选*, 默认值为 `False`):
            是否强制重新下载配置文件并覆盖已存在的缓存版本。
        proxies (`Dict[str, str]`, *可选*):
            代理服务器字典，用于每个请求。
        token (`str` 或 *bool*, *可选*):
            用作远程文件的 HTTP 授权令牌。
        revision (`str`, *可选*, 默认值为 `"main"`):
            要使用的特定模型版本。
        local_files_only (`bool`, *可选*, 默认值为 `False`):
            如果为 `True`，仅尝试从本地文件加载配置。
    
    返回:
        `str`: 模块在缓存中的路径。
    """
    # 下载并缓存来自 `pretrained_model_name_or_path` 的 module_file，或获取本地文件
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)  # 将预训练模型路径转换为字符串

    module_file_or_url = os.path.join(pretrained_model_name_or_path, module_file)  # 组合路径和文件名形成完整路径

    if os.path.isfile(module_file_or_url):  # 检查该路径是否指向一个文件
        resolved_module_file = module_file_or_url  # 如果是文件，保存该路径
        submodule = "local"  # 标记为本地文件
    elif pretrained_model_name_or_path.count("/") == 0:  # 如果路径中没有斜杠，表示这是一个模型名称而非路径
        available_versions = get_diffusers_versions()  # 获取可用版本列表
        # 去掉 ".dev0" 部分
        latest_version = "v" + ".".join(__version__.split(".")[:3])  # 获取最新的版本号

        # 获取匹配的 GitHub 版本
        if revision is None:  # 如果没有指定修订版本
            revision = latest_version if latest_version[1:] in available_versions else "main"  # 默认选择最新版本或主分支
            logger.info(f"Defaulting to latest_version: {revision}.")  # 记录默认选择的版本
        elif revision in available_versions:  # 如果指定版本在可用列表中
            revision = f"v{revision}"  # 格式化版本号
        elif revision == "main":  # 如果指定版本为主分支
            revision = revision  # 保持不变
        else:  # 如果指定版本不在可用版本中
            raise ValueError(
                f"`custom_revision`: {revision} does not exist. Please make sure to choose one of"
                f" {', '.join(available_versions + ['main'])}."  # 提示可用版本
            )

        try:
            resolved_module_file = hf_hub_download(  # 从 Hugging Face Hub 下载指定文件
                repo_id=COMMUNITY_PIPELINES_MIRROR_ID,  # 设定资源库 ID
                repo_type="dataset",  # 指定资源类型为数据集
                filename=f"{revision}/{pretrained_model_name_or_path}.py",  # 构造文件名
                cache_dir=cache_dir,  # 设置缓存目录
                force_download=force_download,  # 决定是否强制下载
                proxies=proxies,  # 设置代理
                local_files_only=local_files_only,  # 是否只考虑本地文件
            )
            submodule = "git"  # 标记为从 GitHub 下载的文件
            module_file = pretrained_model_name_or_path + ".py"  # 更新模块文件名
        except RevisionNotFoundError as e:  # 捕获未找到修订版本的异常
            raise EnvironmentError(
                f"Revision '{revision}' not found in the community pipelines mirror. Check available revisions on"
                " https://huggingface.co/datasets/diffusers/community-pipelines-mirror/tree/main."
                " If you don't find the revision you are looking for, please open an issue on https://github.com/huggingface/diffusers/issues."
            ) from e  # 抛出环境错误并提供信息
        except EnvironmentError:  # 捕获环境错误
            logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")  # 记录错误
            raise  # 重新抛出异常
    else:  # 如果不是文件
        try:
            # 从 URL 加载或从缓存加载（如果已缓存）
            resolved_module_file = hf_hub_download(  # 从 Hugging Face Hub 下载文件
                pretrained_model_name_or_path,  # 使用预训练模型名称作为资源路径
                module_file,  # 指定模块文件名
                cache_dir=cache_dir,  # 设置缓存目录
                force_download=force_download,  # 决定是否强制下载
                proxies=proxies,  # 设置代理
                local_files_only=local_files_only,  # 是否只考虑本地文件
                token=token,  # 传递身份验证令牌
            )
            submodule = os.path.join("local", "--".join(pretrained_model_name_or_path.split("/")))  # 构造本地子模块路径
        except EnvironmentError:  # 捕获环境错误
            logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")  # 记录错误
            raise  # 重新抛出异常
    # 检查环境中是否具备所有所需的模块
    modules_needed = check_imports(resolved_module_file)

    # 将模块移动到我们的缓存动态模块中
    full_submodule = DIFFUSERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)  # 创建动态模块
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule  # 构建子模块路径
    if submodule == "local" or submodule == "git":  # 检查子模块类型
        # 始终复制本地文件（可以通过哈希判断是否有变化）
        # 复制的目的是避免将过多文件夹放入 sys.path
        shutil.copy(resolved_module_file, submodule_path / module_file)  # 复制模块文件到子模块路径
        for module_needed in modules_needed:  # 遍历所需的模块
            if len(module_needed.split(".")) == 2:  # 检查模块名称是否有两部分
                module_needed = "/".join(module_needed.split("."))  # 将模块名称转换为路径格式
                module_folder = module_needed.split("/")[0]  # 获取模块文件夹名
                if not os.path.exists(submodule_path / module_folder):  # 检查文件夹是否存在
                    os.makedirs(submodule_path / module_folder)  # 创建文件夹
            module_needed = f"{module_needed}.py"  # 添加 .py 后缀
            shutil.copy(os.path.join(pretrained_model_name_or_path, module_needed), submodule_path / module_needed)  # 复制所需模块文件
    else:
        # 获取提交哈希值
        # TODO: 未来将从 etag 获取此信息，而不是这里
        commit_hash = model_info(pretrained_model_name_or_path, revision=revision, token=token).sha  # 获取模型的提交哈希

        # 模块文件将放置在带有 git 哈希的子文件夹中，以便实现版本控制
        submodule_path = submodule_path / commit_hash  # 更新子模块路径以包含哈希
        full_submodule = full_submodule + os.path.sep + commit_hash  # 更新完整子模块名称
        create_dynamic_module(full_submodule)  # 创建新的动态模块

        if not (submodule_path / module_file).exists():  # 检查模块文件是否已存在
            if len(module_file.split("/")) == 2:  # 检查模块文件路径是否包含两个部分
                module_folder = module_file.split("/")[0]  # 获取模块文件夹名
                if not os.path.exists(submodule_path / module_folder):  # 检查文件夹是否存在
                    os.makedirs(submodule_path / module_folder)  # 创建文件夹
            shutil.copy(resolved_module_file, submodule_path / module_file)  # 复制模块文件

        # 确保每个相对文件都存在
        for module_needed in modules_needed:  # 遍历所需模块
            if len(module_needed.split(".")) == 2:  # 检查模块名称是否有两部分
                module_needed = "/".join(module_needed.split("."))  # 将模块名称转换为路径格式
            if not (submodule_path / module_needed).exists():  # 检查模块文件是否存在
                get_cached_module_file(  # 获取缓存的模块文件
                    pretrained_model_name_or_path,
                    f"{module_needed}.py",  # 模块文件名
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
    return os.path.join(full_submodule, module_file)  # 返回完整子模块路径及模块文件名
# 装饰器，用于验证传入的参数是否符合预期
@validate_hf_hub_args
def get_class_from_dynamic_module(
    # 预训练模型的名称或路径，可以是字符串或路径类型
    pretrained_model_name_or_path: Union[str, os.PathLike],
    # 模块文件的名称，包含要查找的类
    module_file: str,
    # 要导入的类的名称，默认为 None
    class_name: Optional[str] = None,
    # 缓存目录的路径，默认为 None
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    # 是否强制下载配置文件，默认为 False
    force_download: bool = False,
    # 代理服务器的字典，默认为 None
    proxies: Optional[Dict[str, str]] = None,
    # 用于远程文件的 HTTP 认证令牌，默认为 None
    token: Optional[Union[bool, str]] = None,
    # 具体的模型版本，默认为 "main"
    revision: Optional[str] = None,
    # 是否仅加载本地文件，默认为 False
    local_files_only: bool = False,
    # 其他可选的关键字参数
    **kwargs,
):
    """
    从模块文件中提取一个类，该模块文件可以位于本地文件夹或模型的仓库中。

    <Tip warning={true}>

    调用此函数将执行在本地找到的模块文件或从 Hub 下载的代码。
    因此，仅应在可信的仓库上调用。

    </Tip>

    Args:
        # 预训练模型的名称或路径，可以是 huggingface.co 上的模型 id 或本地目录路径
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            可以是字符串，表示在 huggingface.co 上托管的预训练模型配置的模型 id。
            有效的模型 id 可以位于根目录下，例如 `bert-base-uncased`，
            或者在用户或组织名称下命名，例如 `dbmdz/bert-base-german-cased`。
            也可以是一个目录的路径，包含使用 [`~PreTrainedTokenizer.save_pretrained`] 方法保存的配置文件，
            例如 `./my_model_directory/`。

        # 模块文件的名称，包含要查找的类
        module_file (`str`):
            包含要查找的类的模块文件的名称。
        # 类的名称，默认为 None
        class_name (`str`):
            要导入的类的名称。
        # 缓存目录的路径，默认为 None
        cache_dir (`str` or `os.PathLike`, *optional*):
            下载的预训练模型配置应缓存的目录路径，
            如果不使用标准缓存的话。
        # 是否强制下载配置文件，默认为 False
        force_download (`bool`, *optional*, defaults to `False`):
            是否强制重新下载配置文件，并覆盖已存在的缓存版本。
        # 代理服务器的字典，默认为 None
        proxies (`Dict[str, str]`, *optional*):
            以协议或端点为基础使用的代理服务器字典，例如 `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`。代理将在每个请求中使用。
        # 用于远程文件的 HTTP 认证令牌，默认为 None
        token (`str` or `bool`, *optional*):
            用作远程文件的 HTTP bearer 认证的令牌。
            如果为 True，将使用在运行 `transformers-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        # 具体的模型版本，默认为 "main"
        revision (`str`, *optional*, defaults to `"main"`):
            使用的特定模型版本。可以是分支名、标签名或提交 ID，
            因为我们使用基于 git 的系统在 huggingface.co 上存储模型和其他工件，
            所以 `revision` 可以是 git 允许的任何标识符。
        # 是否仅加载本地文件，默认为 False
        local_files_only (`bool`, *optional*, defaults to `False`):
            如果为 True，仅尝试从本地文件加载标记器配置。

    <Tip>
    # 如果未登录（`huggingface-cli login`），可以通过 `token` 参数传递令牌，以便使用私有或
    # [受限模型](https://huggingface.co/docs/hub/models-gated#gated-models)。
        
    # 返回值：
    #     `type`: 从模块动态导入的类。
    
    # 示例：
    
    # ```python
    # 从 huggingface.co 下载模块 `modeling.py`，缓存并提取类 `MyBertModel`。
    cls = get_class_from_dynamic_module("sgugger/my-bert-model", "modeling.py", "MyBertModel")
    # ```py"""
    # 最后，我们获取新创建模块中的类
    final_module = get_cached_module_file(
        # 获取预训练模型的名称或路径
        pretrained_model_name_or_path,
        # 模块文件名称
        module_file,
        # 缓存目录
        cache_dir=cache_dir,
        # 强制下载标志
        force_download=force_download,
        # 代理设置
        proxies=proxies,
        # 访问令牌
        token=token,
        # 版本控制
        revision=revision,
        # 仅本地文件标志
        local_files_only=local_files_only,
    )
    # 从最终模块中获取类名，去掉 ".py" 后缀
    return get_class_in_module(class_name, final_module.replace(".py", ""))
```