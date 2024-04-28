# `.\transformers\dynamic_module_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""从 Hub 动态加载对象的实用程序。"""
# 导入必要的库
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

# 从 huggingface_hub 模块中导入 try_to_load_from_cache 函数
from huggingface_hub import try_to_load_from_cache

# 从 utils 模块中导入一些函数和变量
from .utils import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    cached_file,
    extract_commit_hash,
    is_offline_mode,
    logging,
)

# 获取 logger 对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 初始化 HF 模块
def init_hf_modules():
    """
    创建具有 init 的模块缓存目录，并将其添加到 Python 路径中。
    """
    # 如果 HF_MODULES_CACHE 已经在 Python 路径中，则说明此函数已经执行过了
    if HF_MODULES_CACHE in sys.path:
        return

    # 将 HF_MODULES_CACHE 添加到 Python 路径中
    sys.path.append(HF_MODULES_CACHE)
    # 如果 HF_MODULES_CACHE 目录不存在，则创建它
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    # 创建 __init__.py 文件并使其生效
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        importlib.invalidate_caches()

# 创建动态模块
def create_dynamic_module(name: Union[str, os.PathLike]):
    """
    在模块缓存目录中创建动态模块。

    Args:
        name (`str` or `os.PathLike`):
            要创建的动态模块的名称。
    """
    # 初始化 HF 模块
    init_hf_modules()
    # 解析动态模块的路径
    dynamic_module_path = (Path(HF_MODULES_CACHE) / name).resolve()
    # 如果父模块尚不存在，则递归创建它
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    # 如果动态模块目录不存在，则创建它
    os.makedirs(dynamic_module_path, exist_ok=True)
    # 创建 __init__.py 文件并使其生效
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        importlib.invalidate_caches()

# 获取相对导入
def get_relative_imports(module_file: Union[str, os.PathLike]) -> List[str]:
    """
    获取模块文件中相对导入的模块列表。

    Args:
        module_file (`str` or `os.PathLike`): 要检查的模块文件。

    Returns:
        `List[str]`: 模块中的相对导入列表。
    """
    # 打开指定文件，以只读方式读取内容，并指定编码为 UTF-8
    with open(module_file, "r", encoding="utf-8") as f:
        # 读取文件的全部内容
        content = f.read()

    # 使用正则表达式查找形如 `import .xxx` 的相对导入语句，并返回匹配结果列表
    relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # 使用正则表达式查找形如 `from .xxx import yyy` 的相对导入语句，并返回匹配结果列表
    relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # 将列表转换为集合去重，再转换为列表，确保结果中每个相对导入只出现一次
    return list(set(relative_imports))
# 获取给定模块所需的所有文件列表。注意，此函数会递归处理相对导入（如果 a 导入了 b，而 b 导入了 c，则会返回 b 和 c 的模块文件）。
def get_relative_import_files(module_file: Union[str, os.PathLike]) -> List[str]:
    # 初始化标志，用于控制循环
    no_change = False
    # 初始待检查文件列表，包含给定模块文件
    files_to_check = [module_file]
    # 初始化相对导入列表
    all_relative_imports = []

    # 开始递归处理所有相对导入
    while not no_change:
        # 存储新的相对导入
        new_imports = []
        # 遍历待检查文件列表
        for f in files_to_check:
            # 获取当前文件的相对导入列表
            new_imports.extend(get_relative_imports(f))

        # 获取给定模块文件的路径
        module_path = Path(module_file).parent
        # 构建新的导入文件列表，将模块文件的路径与相对导入合并
        new_import_files = [str(module_path / m) for m in new_imports]
        # 过滤掉已经添加过的相对导入文件
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]
        # 更新待检查文件列表，将文件名转换成.py文件路径
        files_to_check = [f"{f}.py" for f in new_import_files]

        # 如果没有新的相对导入文件，则标志无变化
        no_change = len(new_import_files) == 0
        # 将新的相对导入文件列表添加到所有相对导入列表中
        all_relative_imports.extend(files_to_check)

    # 返回所有相对导入文件列表
    return all_relative_imports


# 提取在文件中导入的所有库（不包括相对导入）。
def get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    # 读取文件内容
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # 过滤掉try/except块，以便在自定义代码中使用try/except导入
    content = re.sub(r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL)

    # 导入形式为 `import xxx` 的表达式
    imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # 导入形式为 `from xxx import yyy` 的表达式
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # 仅保留顶层模块
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
    return list(set(imports))


# 检查当前 Python 环境是否包含文件中导入的所有库。如果缺少库，则会引发异常。
def check_imports(filename: Union[str, os.PathLike]) -> List[str]:
    # 获取文件中导入的所有库
    imports = get_imports(filename)
    # 存储缺少的库
    missing_packages = []
    # 遍历导入的库
    for imp in imports:
        try:
            # 尝试导入库
            importlib.import_module(imp)
        except ImportError:
            # 如果导入失败，则将库添加到缺少的库列表中
            missing_packages.append(imp)
    # 检查缺失的包是否存在，如果存在缺失的包，则抛出 ImportError 异常
    if len(missing_packages) > 0:
        raise ImportError(
            # 异常消息，列出缺失的包
            "This modeling file requires the following packages that were not found in your environment: "
            # 使用逗号分隔缺失的包名
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    # 返回指定文件中相对导入的列表
    return get_relative_imports(filename)
# 导入所需的模块
def get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    # 将路径分隔符替换为模块路径分隔符
    module_path = module_path.replace(os.path.sep, ".")
    # 导入指定路径的模块
    module = importlib.import_module(module_path)
    # 返回指定类的引用
    return getattr(module, class_name)


# 准备下载模块文件
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
    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    # Check if the deprecated argument 'use_auth_token' is provided and remove it from the dictionary
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 不为 None，则发出警告，提示该参数将在 Transformers 的 v5 版本中移除，建议使用 token 参数
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了 token 参数，则抛出数值错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 参数设置为 use_auth_token
        token = use_auth_token

    # 如果处于离线模式且未设置 local_files_only 为 True，则强制设置 local_files_only 为 True
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # 从 repo `pretrained_model_name_or_path` 下载并缓存 module_file，如果是本地文件则直接获取
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        submodule = os.path.basename(pretrained_model_name_or_path)
    else:
        submodule = pretrained_model_name_or_path.replace("/", os.path.sep)
        cached_module = try_to_load_from_cache(
            pretrained_model_name_or_path, module_file, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )

    new_files = []
    try:
        # 从 URL 或缓存中加载已缓存的模块文件
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
        # 如果不是本地文件且缓存的模块文件与解析后的模块文件不同，则将模块文件添加到 new_files 列表中
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)

    except EnvironmentError:
        # 如果无法在 pretrained_model_name_or_path 中找到 module_file，则记录错误并抛出异常
        logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
        raise

    # 检查我们的环境中是否具有所有所需的模块
    modules_needed = check_imports(resolved_module_file)

    # 现在我们将模块��动到我们的缓存动态模块中
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    # 检查 submodule 是否与预训练模型名称或路径的基本名称相同
    if submodule == os.path.basename(pretrained_model_name_or_path):
        # 为了避免在 sys.path 中放置太多文件夹，我们复制本地文件。当文件是新的或自上次复制以来已更改时，执行此复制操作。
        if not (submodule_path / module_file).exists() or not filecmp.cmp(
            resolved_module_file, str(submodule_path / module_file)
        ):
            # 复制解析后的模块文件到 submodule_path/module_file
            shutil.copy(resolved_module_file, submodule_path / module_file)
            # 使导入模块的缓存无效
            importlib.invalidate_caches()
        # 遍历所需的模块
        for module_needed in modules_needed:
            module_needed = f"{module_needed}.py"
            module_needed_file = os.path.join(pretrained_model_name_or_path, module_needed)
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(
                module_needed_file, str(submodule_path / module_needed)
            ):
                # 复制所需的模块文件到 submodule_path/module_needed
                shutil.copy(module_needed_file, submodule_path / module_needed)
                # 使导入模块的缓存无效
                importlib.invalidate_caches()
    else:
        # 获取提交哈希
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)

        # 模块文件最终将被放置在具有存储库 git 哈希的子文件夹中。这样我们就可以获得版本控制的好处。
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        # 创建动态模块
        create_dynamic_module(full_submodule)

        if not (submodule_path / module_file).exists():
            # 复制解析后的模块文件到 submodule_path/module_file
            shutil.copy(resolved_module_file, submodule_path / module_file)
            # 使导入模块的缓存无效
            importlib.invalidate_caches()
        # 确保我们也有每个相对文件
        for module_needed in modules_needed:
            if not (submodule_path / f"{module_needed}.py").exists():
                # 获取缓存的模块文件
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

    if len(new_files) > 0 and revision is None:
        # 将新文件列表转换为字符串
        new_files = "\n".join([f"- {f}" for f in new_files])
        repo_type_str = "" if repo_type is None else f"{repo_type}s/"
        url = f"https://huggingface.co/{repo_type_str}{pretrained_model_name_or_path}"
        # 发出警告，指出从 url 下载了以下文件的新版本
        logger.warning(
            f"A new version of the following files was downloaded from {url}:\n{new_files}"
            "\n. Make sure to double-check they do not contain any added malicious code. To avoid downloading new "
            "versions of the code file, you can pin a revision."
        )

    # 返回完整的子模块路径和模块文件
    return os.path.join(full_submodule, module_file)
def get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> typing.Type:
    """
    Extracts a class from a module file, present in the local folder or repository of a model.

    <Tip warning={true}>

    Calling this function will execute the code in the module file found locally or downloaded from the Hub. It should
    therefore only be called on trusted repos.

    </Tip>

    <Tip>

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
    # Check if `use_auth_token` is provided and issue a warning
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # Check if both `token` and `use_auth_token` are specified
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # Extract the repo ID if specified in `class_reference`
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path
    module_file, class_name = class_reference.split(".")

    # Set `code_revision` if not provided and `pretrained_model_name_or_path` matches `repo_id`
    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision
    # Get the module file from cache or download it
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
    # Return the class extracted from the module
    return get_class_in_module(class_name, final_module.replace(".py", ""))


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
    # Check if the object is defined in __main__ module
    if obj.__module__ == "__main__":
        # Log a warning message if the object is defined in __main__ module
        logger.warning(
            f"We can't save the code defining {obj} in {folder} as it's been defined in __main__. You should put "
            "this code in a separate module so we can include it in the saved folder and make it easier to share via "
            "the Hub."
        )
        # Return if object is defined in __main__ module
        return

    # Function to set auto_map in config
    def _set_auto_map_in_config(_config):
        # Get the module name of the object
        module_name = obj.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{obj.__class__.__name__}"
        # Special handling for tokenizers
        if "Tokenizer" in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith("Fast"):
                # Fast tokenizer: we have the fast tokenizer class and we may have the slow one has an attribute.
                fast_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
                if getattr(obj, "slow_tokenizer_class", None) is not None:
                    slow_tokenizer = getattr(obj, "slow_tokenizer_class")
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split(".")[-1]
                    slow_tokenizer_class = f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
            else:
                # Slow tokenizer: no way to have the fast class
                slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"

            full_name = (slow_tokenizer_class, fast_tokenizer_class)

        # Update auto_map in config based on config type
        if isinstance(_config, dict):
            auto_map = _config.get("auto_map", {})
            auto_map[obj._auto_class] = full_name
            _config["auto_map"] = auto_map
        elif getattr(_config, "auto_map", None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class: full_name}

    # Add object class to the config auto_map
    if isinstance(config, (list, tuple)):
        # Iterate over multiple configs
        for cfg in config:
            _set_auto_map_in_config(cfg)
    elif config is not None:
        # Set auto_map in single config
        _set_auto_map_in_config(config)

    result = []
    # Copy module file to the output folder.
    object_file = sys.modules[obj.__module__].__file__
    dest_file = Path(folder) / (Path(object_file).name)
    # Copy the object file to the destination folder
    shutil.copy(object_file, dest_file)
    result.append(dest_file)
    # 递归地收集所有相对导入的文件，并确保它们也被复制
    for needed_file in get_relative_import_files(object_file):
        # 将需要的文件复制到目标文件夹中
        dest_file = Path(folder) / (Path(needed_file).name)
        shutil.copy(needed_file, dest_file)
        # 将目标文件添加到结果列表中
        result.append(dest_file)

    # 返回结果列表
    return result
# 定义一个函数，用于在超时时引发数值错误
def _raise_timeout_error(signum, frame):
    raise ValueError(
        "Loading this model requires you to execute custom code contained in the model repository on your local "
        "machine. Please set the option `trust_remote_code=True` to permit loading of this model."
    )

# 设置远程代码信任的超时时间为15秒
TIME_OUT_REMOTE_CODE = 15

# 解析远程代码信任选项，根据不同情况返回是否信任远程代码
def resolve_trust_remote_code(trust_remote_code, model_name, has_local_code, has_remote_code):
    # 如果未设置远程代码信任选项
    if trust_remote_code is None:
        # 如果存在本地代码，则默认不信任远程代码
        if has_local_code:
            trust_remote_code = False
        # 如果存在远程代码且超时时间大于0
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            try:
                # 设置信号处理函数为_raise_timeout_error，并启动定时器
                signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                # 在信任远程代码选项未确定的情况下循环询问用户
                while trust_remote_code is None:
                    answer = input(
                        f"The repository for {model_name} contains custom code which must be executed to correctly "
                        f"load the model. You can inspect the repository content at https://hf.co/{model_name}.\n"
                        f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                # 取消定时器
                signal.alarm(0)
            except Exception:
                # 捕获异常，处理不支持signal.SIGALRM的操作系统
                raise ValueError(
                    f"The repository for {model_name} contains custom code which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{model_name}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )
        # 如果存在远程代码但超时时间为0
        elif has_remote_code:
            # 对于将超时时间设置为0的CI
            _raise_timeout_error(None, None)

    # 如果存在远程代码且不存在本地代码且未信任远程代码，则引发数值错误
    if has_remote_code and not has_local_code and not trust_remote_code:
        raise ValueError(
            f"Loading {model_name} requires you to execute the configuration file in that"
            " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
            " set the option `trust_remote_code=True` to remove this error."
        )

    # 返回最终的远程代码信任选项
    return trust_remote_code
```