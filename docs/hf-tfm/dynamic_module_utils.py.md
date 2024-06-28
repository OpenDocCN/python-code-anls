# `.\dynamic_module_utils.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to dynamically load objects from the Hub."""
import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import try_to_load_from_cache

from .utils import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    cached_file,
    extract_commit_hash,
    is_offline_mode,
    logging,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def init_hf_modules():
    """
    Creates the cache directory for modules with an init, and adds it to the Python path.
    """
    # 如果 HF_MODULES_CACHE 已经在 Python 路径中，说明函数已经执行过，直接返回
    if HF_MODULES_CACHE in sys.path:
        return

    # 将 HF_MODULES_CACHE 加入到 Python 路径中
    sys.path.append(HF_MODULES_CACHE)
    # 创建 HF_MODULES_CACHE 目录，如果目录已存在则不做操作
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    # 在 HF_MODULES_CACHE 目录下创建 __init__.py 文件，如果文件已存在则不做操作
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        # 清除 importlib 缓存，使得新创建的模块可以被正确加载
        importlib.invalidate_caches()


def create_dynamic_module(name: Union[str, os.PathLike]):
    """
    Creates a dynamic module in the cache directory for modules.

    Args:
        name (`str` or `os.PathLike`):
            The name of the dynamic module to create.
    """
    # 初始化 HF 模块，确保 HF 模块缓存目录存在并在 Python 路径中
    init_hf_modules()
    # 获取动态模块的完整路径
    dynamic_module_path = (Path(HF_MODULES_CACHE) / name).resolve()
    # 如果父目录不存在，则递归创建
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    # 创建动态模块的目录，如果目录已存在则不做操作
    os.makedirs(dynamic_module_path, exist_ok=True)
    # 在动态模块目录下创建 __init__.py 文件，如果文件已存在则不做操作
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        # 清除 importlib 缓存，确保新创建的模块可以被正确加载
        importlib.invalidate_caches()


def get_relative_imports(module_file: Union[str, os.PathLike]) -> List[str]:
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of relative imports in the module.
    """
    # 使用 `utf-8` 编码打开指定文件 `module_file` 并读取其内容
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找内容中形如 `import .xxx` 的相对导入语句，并将结果存入 `relative_imports`
    relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # 查找内容中形如 `from .xxx import yyy` 的相对导入语句，并将结果追加到 `relative_imports`
    relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # 将 `relative_imports` 列表转换为集合，以去除重复项，然后再转换回列表形式
    return list(set(relative_imports))
def get_relative_import_files(module_file: Union[str, os.PathLike]) -> List[str]:
    """
    Get the list of all files that are needed for a given module. Note that this function recurses through the relative
    imports (if a imports b and b imports c, it will return module files for b and c).

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of all relative imports a given module needs (recursively), which will give us the list
        of module files a given module needs.
    """
    no_change = False  # 标志变量，用于检测是否有新的相对导入被找到
    files_to_check = [module_file]  # 初始时待检查的文件列表，从传入的模块文件开始
    all_relative_imports = []  # 存储所有找到的相对导入模块文件的列表

    # Let's recurse through all relative imports
    while not no_change:  # 进入循环，直到没有新的相对导入被找到为止
        new_imports = []
        for f in files_to_check:
            new_imports.extend(get_relative_imports(f))  # 递归获取当前文件 f 的相对导入

        module_path = Path(module_file).parent  # 获取传入模块文件的父目录路径
        new_import_files = [str(module_path / m) for m in new_imports]  # 构建新的相对导入文件列表
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]  # 去重，确保不重复添加
        files_to_check = [f"{f}.py" for f in new_import_files]  # 将新的相对导入文件名列表加上 '.py' 后缀，准备下一轮检查

        no_change = len(new_import_files) == 0  # 如果没有新的相对导入被找到，则结束循环
        all_relative_imports.extend(files_to_check)  # 将新找到的相对导入文件列表加入到总列表中

    return all_relative_imports  # 返回所有找到的相对导入文件列表


def get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Extracts all the libraries (not relative imports this time) that are imported in a file.

    Args:
        filename (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `List[str]`: The list of all packages required to use the input module.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()  # 读取文件内容

    # filter out try/except block so in custom code we can have try/except imports
    content = re.sub(r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL)

    # Imports of the form `import xxx`
    imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]  # 提取导入的顶级模块名称
    return list(set(imports))  # 返回去重后的模块名称列表


def check_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Check if the current Python environment contains all the libraries that are imported in a file. Will raise if a
    library is missing.

    Args:
        filename (`str` or `os.PathLike`): The module file to check.

    Returns:
        `List[str]`: The list of relative imports in the file.
    """
    imports = get_imports(filename)  # 获取文件中所有的非相对导入模块名称
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)  # 尝试导入模块，如果失败则捕获 ImportError
        except ImportError:
            missing_packages.append(imp)  # 将缺失的模块名称加入到缺失列表中
    # 检查缺失的包列表是否有内容
    if len(missing_packages) > 0:
        # 如果有缺失的包，则抛出 ImportError 异常，提示用户缺少哪些包
        raise ImportError(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    # 如果没有缺失的包，返回模块文件的相对导入路径列表
    return get_relative_imports(filename)
# 从指定的模块文件中获取指定名称的类对象

def get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    # 标准化模块路径，替换路径分隔符和去掉.py后缀，生成模块名
    name = os.path.normpath(module_path).replace(".py", "").replace(os.path.sep, ".")
    # 构建模块文件的完整路径
    module_path = str(Path(HF_MODULES_CACHE) / module_path)
    # 使用 SourceFileLoader 加载模块文件并返回模块对象
    module = importlib.machinery.SourceFileLoader(name, module_path).load_module()
    # 从加载的模块中获取指定名称的类对象并返回
    return getattr(module, class_name)


def get_cached_module_file(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> str:
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.
    """
    # 从参数中弹出并获取 `use_auth_token`，用于兼容旧的参数命名
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 `use_auth_token` 参数被指定了，则发出警告并提示将在 Transformers v5 版本中移除
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 `token` 参数也被指定了，则引发 ValueError，因为不能同时设置 `token` 和 `use_auth_token`
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 `token` 参数设置为 `use_auth_token` 的值，以实现向后兼容性
        token = use_auth_token
    # 如果处于离线模式且不限制只使用本地文件，则强制设置 local_files_only=True
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # 将 pretrained_model_name_or_path 转换为字符串类型
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    # 检查 pretrained_model_name_or_path 是否为本地目录
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        # 如果是本地目录，则 submodule 为该目录的基本名称
        submodule = os.path.basename(pretrained_model_name_or_path)
    else:
        # 如果不是本地目录，则将 pretrained_model_name_or_path 中的 '/' 替换为系统路径分隔符
        submodule = pretrained_model_name_or_path.replace("/", os.path.sep)
        # 尝试从缓存中加载模块文件
        cached_module = try_to_load_from_cache(
            pretrained_model_name_or_path, module_file, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )

    # 用于存储新添加的文件
    new_files = []
    try:
        # 尝试从 URL 或缓存中加载模块文件
        resolved_module_file = cached_file(
            pretrained_model_name_or_path,
            module_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            repo_type=repo_type,
            _commit_hash=_commit_hash,
        )
        # 如果不是本地模式且缓存的模块文件与解析的模块文件不同，则将模块文件添加到 new_files 列表中
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)

    # 如果发生环境错误，则记录错误信息并抛出异常
    except EnvironmentError:
        logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
        raise

    # 检查当前环境中是否存在所需的模块
    modules_needed = check_imports(resolved_module_file)

    # 将模块移动到缓存的动态模块中
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule

    # 如果 submodule 是 pretrained_model_name_or_path 的基本名称
    if submodule == os.path.basename(pretrained_model_name_or_path):
        # 为避免在 sys.path 中添加过多文件夹，将本地文件复制到 submodule_path 中
        # 当文件是新的或自上次复制以来已更改时执行复制操作
        if not (submodule_path / module_file).exists() or not filecmp.cmp(
            resolved_module_file, str(submodule_path / module_file)
        ):
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()

        # 复制所需的模块文件到 submodule_path 中
        for module_needed in modules_needed:
            module_needed = f"{module_needed}.py"
            module_needed_file = os.path.join(pretrained_model_name_or_path, module_needed)
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(
                module_needed_file, str(submodule_path / module_needed)
            ):
                shutil.copy(module_needed_file, submodule_path / module_needed)
                importlib.invalidate_caches()
    else:
        # 提取提交哈希值
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)

        # 模块文件将被放置在具有存储库 git 哈希的子文件夹中，以便进行版本控制。
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        create_dynamic_module(full_submodule)

        # 如果子模块路径下的模块文件不存在，则复制已解析的模块文件
        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()

        # 确保我们也有每个相对的文件
        for module_needed in modules_needed:
            # 如果子模块路径下的模块文件不存在，则获取缓存的模块文件
            if not (submodule_path / f"{module_needed}.py").exists():
                get_cached_module_file(
                    pretrained_model_name_or_path,
                    f"{module_needed}.py",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                )
                new_files.append(f"{module_needed}.py")

    # 如果有新的文件被下载并且没有指定 revision，则生成警告消息
    if len(new_files) > 0 and revision is None:
        new_files = "\n".join([f"- {f}" for f in new_files])
        repo_type_str = "" if repo_type is None else f"{repo_type}s/"
        url = f"https://huggingface.co/{repo_type_str}{pretrained_model_name_or_path}"
        logger.warning(
            f"A new version of the following files was downloaded from {url}:\n{new_files}"
            "\n. Make sure to double-check they do not contain any added malicious code. To avoid downloading new "
            "versions of the code file, you can pin a revision."
        )

    # 返回完整的子模块路径和模块文件名
    return os.path.join(full_submodule, module_file)
# 从动态模块中获取指定类的定义
def get_class_from_dynamic_module(
    # 类的完整引用路径，例如 "module.submodule.ClassName"
    class_reference: str,
    # 预训练模型的名称或路径，可以是字符串或路径对象
    pretrained_model_name_or_path: Union[str, os.PathLike],
    # 缓存目录的路径，可选参数，默认为 None
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    # 是否强制重新下载模型文件，默认为 False
    force_download: bool = False,
    # 是否恢复之前中断的下载，默认为 False
    resume_download: bool = False,
    # 可选的代理设置，字典类型，用于网络请求
    proxies: Optional[Dict[str, str]] = None,
    # 访问模型所需的令牌，可以是布尔值或字符串，可选
    token: Optional[Union[bool, str]] = None,
    # 模型所在仓库的版本号或标签，可选
    revision: Optional[str] = None,
    # 是否仅使用本地已有文件，默认为 False
    local_files_only: bool = False,
    # 仓库类型，例如 git、hg 等，可选
    repo_type: Optional[str] = None,
    # 代码的特定版本号，可选
    code_revision: Optional[str] = None,
    # 其他参数作为关键字参数传递，用于模块初始化
    **kwargs,
) -> typing.Type:
    """
    从本地文件夹或模型仓库中提取一个类的定义。

    <Tip warning={true}>

    调用此函数将执行本地或从 Hub 下载的模块文件中的代码。因此，应仅在可信任的仓库中调用。

    </Tip>
    # 加载指定类的配置和模型数据
    Args:
        class_reference (`str`):
            要加载的类的完整名称，包括其模块和可选的存储库。
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            可以是以下之一：

            - 字符串，表示在 huggingface.co 模型仓库中预训练模型配置的 *模型 ID*。
            - 目录路径，包含使用 [`~PreTrainedTokenizer.save_pretrained`] 方法保存的配置文件，例如 `./my_model_directory/`。

            当 `class_reference` 没有指定其他存储库时使用。
        module_file (`str`):
            包含要查找的类的模块文件名。
        class_name (`str`):
            要在模块中导入的类的名称。
        cache_dir (`str` or `os.PathLike`, *optional*):
            下载预训练模型配置时应该缓存的目录路径，如果不想使用标准缓存。
        force_download (`bool`, *optional*, defaults to `False`):
            是否强制下载配置文件，并覆盖已存在的缓存版本。
        resume_download (`bool`, *optional*, defaults to `False`):
            是否删除未完全接收的文件。如果存在这样的文件，则尝试恢复下载。
        proxies (`Dict[str, str]`, *optional*):
            使用的代理服务器字典，按协议或端点分组，例如 `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。
            代理服务器会在每个请求上使用。
        token (`str` or `bool`, *optional*):
            用作远程文件的 HTTP Bearer 授权令牌。如果是 `True`，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        revision (`str`, *optional*, defaults to `"main"`):
            要使用的特定模型版本。可以是分支名称、标签名称或提交 ID。由于我们在 huggingface.co 上使用基于 Git 的系统存储模型和其他工件，因此 `revision` 可以是 Git 允许的任何标识符。
        local_files_only (`bool`, *optional*, defaults to `False`):
            如果为 `True`，将仅尝试从本地文件加载 tokenizer 配置。
        repo_type (`str`, *optional*):
            指定存储库类型（在下载时特别有用，例如从空间下载）。
        code_revision (`str`, *optional*, defaults to `"main"`):
            在 Hub 上使用的代码的特定版本。如果代码存储在与模型其余部分不同的存储库中，可以是分支名称、标签名称或提交 ID。由于我们在 huggingface.co 上使用基于 Git 的系统存储模型和其他工件，因此 `revision` 可以是 Git 允许的任何标识符。
    Passing `token=True` is required when you want to use a private model.



    </Tip>



    Returns:
        `typing.Type`: The class, dynamically imported from the module.



    Examples:

    ```python
    # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("modeling.MyBertModel", "sgugger/my-bert-model")

    # Download module `modeling.py` from a given repo and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("sgugger/my-bert-model--modeling.MyBertModel", "sgugger/another-bert-model")
    ```"""



    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token



    # Catch the name of the repo if it's specified in `class_reference`
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path
    module_file, class_name = class_reference.split(".")



    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision
    # And lastly we get the class inside our newly created module
    final_module = get_cached_module_file(
        repo_id,
        module_file + ".py",
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=code_revision,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )
    return get_class_in_module(class_name, final_module)
def custom_object_save(obj: Any, folder: Union[str, os.PathLike], config: Optional[Dict] = None) -> List[str]:
    """
    Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder. Optionally
    adds the proper fields in a config.

    Args:
        obj (`Any`): The object for which to save the module files.
        folder (`str` or `os.PathLike`): The folder where to save.
        config (`PretrainedConfig` or dictionary, `optional`):
            A config in which to register the auto_map corresponding to this custom object.

    Returns:
        `List[str]`: The list of files saved.
    """
    # Check if the object is defined in the '__main__' module; issue a warning if true and return.
    if obj.__module__ == "__main__":
        logger.warning(
            f"We can't save the code defining {obj} in {folder} as it's been defined in __main__. You should put "
            "this code in a separate module so we can include it in the saved folder and make it easier to share via "
            "the Hub."
        )
        return

    def _set_auto_map_in_config(_config):
        # Get the module name where the object's class is defined.
        module_name = obj.__class__.__module__
        # Extract the last module name from the full module path.
        last_module = module_name.split(".")[-1]
        # Construct the full name of the object's class.
        full_name = f"{last_module}.{obj.__class__.__name__}"

        # Special handling for tokenizers
        if "Tokenizer" in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith("Fast"):
                # For fast tokenizers, capture the fast tokenizer class and check for a slow tokenizer attribute.
                fast_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
                if getattr(obj, "slow_tokenizer_class", None) is not None:
                    slow_tokenizer = getattr(obj, "slow_tokenizer_class")
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split(".")[-1]
                    slow_tokenizer_class = f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
            else:
                # For slow tokenizers, only record the slow tokenizer class.
                slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"

            # Assign both tokenizer classes to full_name.
            full_name = (slow_tokenizer_class, fast_tokenizer_class)

        # Update the auto_map in the provided config.
        if isinstance(_config, dict):
            auto_map = _config.get("auto_map", {})
            auto_map[obj._auto_class] = full_name
            _config["auto_map"] = auto_map
        elif getattr(_config, "auto_map", None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class: full_name}

    # Add object class to the config auto_map based on the type of config provided.
    if isinstance(config, (list, tuple)):
        for cfg in config:
            _set_auto_map_in_config(cfg)
    elif config is not None:
        _set_auto_map_in_config(config)

    result = []
    # Get the file path of the module where the object's class is defined.
    object_file = sys.modules[obj.__module__].__file__
    # 构建目标文件路径，将对象文件复制到目标路径中
    dest_file = Path(folder) / (Path(object_file).name)
    shutil.copy(object_file, dest_file)
    result.append(dest_file)

    # 递归获取对象文件的所有相对导入文件，并确保它们也被复制到目标路径中
    for needed_file in get_relative_import_files(object_file):
        # 构建相对导入文件的目标路径，复制文件到目标路径中
        dest_file = Path(folder) / (Path(needed_file).name)
        shutil.copy(needed_file, dest_file)
        result.append(dest_file)

    # 返回复制操作完成后的结果列表
    return result
# 定义一个处理超时错误的函数，当超时发生时抛出 ValueError 异常
def _raise_timeout_error(signum, frame):
    raise ValueError(
        "Loading this model requires you to execute custom code contained in the model repository on your local "
        "machine. Please set the option `trust_remote_code=True` to permit loading of this model."
    )

# 设定远程代码加载超时时间为 15 秒
TIME_OUT_REMOTE_CODE = 15

# 解析是否信任远程代码的函数，根据不同情况返回信任标志
def resolve_trust_remote_code(trust_remote_code, model_name, has_local_code, has_remote_code):
    # 如果未设置信任远程代码选项
    if trust_remote_code is None:
        # 如果本地存在代码，则默认不信任远程代码
        if has_local_code:
            trust_remote_code = False
        # 如果存在远程代码且设置了正超时时间，则尝试获取用户输入以决定是否信任远程代码
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            try:
                # 设置信号处理函数为 _raise_timeout_error，并启动超时定时器
                signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                # 在用户未作出决定之前循环提示
                while trust_remote_code is None:
                    answer = input(
                        f"The repository for {model_name} contains custom code which must be executed to correctly "
                        f"load the model. You can inspect the repository content at https://hf.co/{model_name}.\n"
                        f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    # 根据用户输入确定是否信任远程代码
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                # 取消超时定时器
                signal.alarm(0)
            except Exception:
                # 捕获可能出现的异常（如操作系统不支持 signal.SIGALRM）
                raise ValueError(
                    f"The repository for {model_name} contains custom code which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{model_name}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )
        # 对于存在远程代码但超时时间设置为 0 的情况，抛出超时错误
        elif has_remote_code:
            _raise_timeout_error(None, None)

    # 如果存在远程代码但本地没有代码且用户不信任远程代码，则抛出 ValueError 异常
    if has_remote_code and not has_local_code and not trust_remote_code:
        raise ValueError(
            f"Loading {model_name} requires you to execute the configuration file in that"
            " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
            " set the option `trust_remote_code=True` to remove this error."
        )

    # 返回最终的信任远程代码标志
    return trust_remote_code
```