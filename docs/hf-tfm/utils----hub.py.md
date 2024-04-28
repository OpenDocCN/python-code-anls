# `.\transformers\utils\hub.py`

```
# 版权声明和许可信息
# 版权声明和许可信息，指定了代码的版权和许可信息
#
# 导入所需的库和模块
# 导入所需的库和模块，包括各种工具和功能
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4

import huggingface_hub
import requests
from huggingface_hub import (
    _CACHED_NO_EXIST,
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    constants,
    create_branch,
    create_commit,
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    build_hf_headers,
    hf_raise_for_status,
    send_telemetry,
)
from requests.exceptions import HTTPError

# 导入自定义的模块和函数
# 导入自定义的模块和函数，包括版本信息和日志记录功能
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)
from .logging import tqdm

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 检查是否处于离线模式
_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False

def is_offline_mode():
    return _is_offline_mode

# 设置 Torch 缓存目录
torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
# 设置默认缓存路径
default_cache_path = constants.default_cache_path
# 设置旧的默认缓存路径
old_default_cache_path = os.path.join(torch_cache_home, "transformers")

# 确定默认缓存目录
# 确定默认缓存目录，考虑了许多遗留环境变量以确保向后兼容性
# 最佳设置缓存路径的方式是使用环境变量 HF_HOME
# 在代码中，使用 `HF_HUB_CACHE` 作为默认缓存路径
# 此变量由库设置，并保证设置为正确的值
#
# TODO: 为 v5 版本进行清理？
# 设置 PyTorch 预训练 BERT 缓存路径，默认为 HF_HUB_CACHE
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
# 设置 PyTorch Transformers 缓存路径，默认为 PYTORCH_PRETRAINED_BERT_CACHE
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
# 设置 Transformers 缓存路径，默认为 PYTORCH_TRANSFORMERS_CACHE
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

# 如果旧的默认缓存路径存在且新的缓存路径不存在且没有设置相关环境变量，则进行一次性移动
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(constants.HF_HUB_CACHE)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    # 发出警告，提示缓存路径已更改
    logger.warning(
        "In Transformers v4.22.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/hub'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/hub' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    # 将旧的默认缓存路径移动到新的缓存路径
    shutil.move(old_default_cache_path, constants.HF_HUB_CACHE)

# 设置 HF 模块缓存路径，默认为 HF_HOME 下的 modules 文件夹
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
# 设置 Transformers 动态模块名称
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
# 生成会话 ID
SESSION_ID = uuid4().hex

# 对旧的环境变量进行弃用警告
for key in ("PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE"):
    if os.getenv(key) is not None:
        warnings.warn(
            f"Using `{key}` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.",
            FutureWarning,
        )

# 设置 S3 存储桶前缀和 CloudFront 分发前缀
S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

# 检查是否处于暂存模式
_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
# 设置默认端点
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"

# 解析 HUGGINGFACE_CO_RESOLVE_ENDPOINT 环境��量
HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
# 设置 Hugging Face CO 前缀
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
# 设置 Hugging Face CO 示例遥测
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"

# 判断是否为远程 URL
def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

# 待移除的 TODO 注释，用于标记待处理事项
# 待移除的 TODO 注释，用于标记待处理事项
# 标记该方法已过时，不支持新的缓存系统，建议从 './examples/research_projects/visual_bert/utils.py' 中移除
@_deprecate_method(version="4.39.0", message="This method is outdated and does not support the new cache system.")
def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    返回一个表示本地缓存的模型二进制文件的元组列表。每个元组的形状为 `(model_url, etag, size_MB)`。只有以 *.bin* 结尾的 URL 才会被添加。

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            要在其中搜索模型的缓存目录。如果未设置，将默认使用 transformers 缓存。

    Returns:
        List[Tuple]: 元组列表，每个元组的形状为 `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    使用基本信息格式化用户代理字符串。
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    # 检查是否安装了 TensorFlow
    if is_tf_available():
        # 如果安装了 TensorFlow，则在用户代理中添加 TensorFlow 版本信息
        ua += f"; tensorflow/{_tf_version}"
    
    # 检查是否禁用了 Hub 的遥测功能
    if constants.HF_HUB_DISABLE_TELEMETRY:
        # 如果禁用了遥测功能，则在用户代理中添加相应信息
        return ua + "; telemetry/off"
    
    # 检查是否在 SageMaker 上运行训练任务
    if is_training_run_on_sagemaker():
        # 如果在 SageMaker 上运行训练任务，则在用户代理中添加 SageMaker 相关信息
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    
    # 检查是否在 CI 环境中
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        # 如果在 CI 环境中，则在用户代理中添加相应信息
        ua += "; is_ci/true
    
    # 检查用户代理是否为字典类型
    if isinstance(user_agent, dict):
        # 如果用户代理是字典类型，则将字典中的键值对添加到用户代理中
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    
    # 检查用户代理是否为字符串类型
    elif isinstance(user_agent, str):
        # 如果用户代理是字符串类型，则直接添加到用户代理中
        ua += "; " + user_agent
    
    # 返回最终的用户代理信息
    return ua
# 从已解析的文件名中提取提交哈希值，用于缓存文件
def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    # 如果已解析的文件名为空或提交哈希值不为空，则返回提交哈希值
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    # 将解析的文件名转换为 POSIX 格式的字符串
    resolved_file = str(Path(resolved_file).as_posix())
    # 在解析的文件名中搜索包含"snapshots/([^/]+)/"的部分
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    # 如果搜索结果为空，则返回 None
    if search is None:
        return None
    # 提取搜索结果中的提交哈希值
    commit_hash = search.groups()[0]
    # 如果提取的提交哈希值符合正则表达式 REGEX_COMMIT_HASH，则返回该值，否则返回 None
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None

# 尝试在本地文件夹和仓库中定位文件，如果需要则下载并缓存
def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]] = None,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
    """
    # 定义函数参数说明
    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
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
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).
    
    <Tip>
    
    Passing `token=True` is required when you want to use a private model.
    
    </Tip>
    
    # 返回值说明
    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).
    
    # 示例
    Examples:
    
    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```
    """
    # 获取旧版本参数中的"use_auth_token"值
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 不为 None，则发出警告，提示该参数将在 Transformers 的 v5 版本中移除，建议使用 token 替代
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 不为 None，则抛出数值错误，提示只能设置一个参数 token
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 设置为 use_auth_token
        token = use_auth_token

    # 私有参数
    #     _raise_exceptions_for_missing_entries: 如果为 False，则不会因缺少条目而引发异常，而是返回 None
    #     _raise_exceptions_for_connection_errors: 如果为 False，则不会因连接错误而引发异常，而是返回 None
    #     _commit_hash: 在链式调用多个文件时传递，如果这些文件在此提交哈希中缓存，则避免调用缓存中的 head 和 get
    if is_offline_mode() and not local_files_only:
        # 如果处于离线模式且不仅限于本地文件，则强制将 local_files_only 设置为 True
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        # 如果 subfolder 为 None，则将其设置为空字符串
        subfolder = ""

    # 将 path_or_repo_id 转换为字符串
    path_or_repo_id = str(path_or_repo_id)
    # 将 subfolder 和 filename 组合成完整的文件名
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        # 如果 path_or_repo_id 是目录，则解析文件路径
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            # 如果解析的文件不存在
            if _raise_exceptions_for_missing_entries:
                # 如果 _raise_exceptions_for_missing_entries 为 True，则抛出环境错误，提示文件不存在
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
                )
            else:
                # 如果 _raise_exceptions_for_missing_entries 为 False，则返回 None
                return None
        # 返回解析的文件路径
        return resolved_file

    if cache_dir is None:
        # 如果 cache_dir 为 None，则将其设置为 TRANSFORMERS_CACHE
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        # 如果 cache_dir 是 Path 对象，则将其转换为字符串
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # 如果 _commit_hash 不为 None 且不强制下载
        # 如果文件在该提交哈希下已缓存，则直接返回
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                # 如果文件不存在且 _raise_exceptions_for_missing_entries 为 True，则抛出环境错误
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    # 根据用户代理设置 HTTP 用户代理
    user_agent = http_user_agent(user_agent)
    try:
        # 尝试从 URL 或缓存加载文件，如果已经缓存则直接使用缓存
        resolved_file = hf_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        # 如果尝试访问的是受限制的 repo，则抛出环境错误
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to request access at "
            f"https://huggingface.co/{path_or_repo_id} and pass a token having permission to this repo either "
            "by logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        # 如果指定的 repo 不存在，则抛出环境错误
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        # 如果指定的 revision 不存在，则抛出环境错误
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        # 尝试从缓存加载文件（可能不是最新的）
        resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        # 如果不为缺失条目或连接错误抛出异常，则返回 None
        if not _raise_exceptions_for_missing_entries or not _raise_exceptions_for_connection_errors:
            return None
        # 如果无法连接到 Hugging Face 的解析端点加载文件，也找不到缓存文件，并且 path_or_repo_id 不是包含名为 full_filename 的文件的目录路径，则抛出环境错误
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ) from e
    # 处理 EntryNotFoundError 异常
    except EntryNotFoundError as e:
        # 如果不抛出异常以表示缺少条目，则返回 None
        if not _raise_exceptions_for_missing_entries:
            return None
        # 如果未指定修订版本，则设置为 "main"
        if revision is None:
            revision = "main"
        # 抛出环境错误，指示指定的文件不存在
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
        ) from e
    # 处理 HTTPError 异常
    except HTTPError as err:
        # 首先尝试从缓存中加载文件（可能不是最新的）
        resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
        # 如果成功从缓存中加载文件，则返回该文件
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        # 如果不抛出连接错误异常，则返回 None
        if not _raise_exceptions_for_connection_errors:
            return None
        # 抛出环境错误，指示在加载文件时发生特定的连接错误
        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
    # 处理 HFValidationError 异常
    except HFValidationError as e:
        # 抛出环境错误，指示 path_or_model_id 参数不正确
        raise EnvironmentError(
            f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a local folder or the repo_id of a model on the Hub."
        ) from e
    # 返回解析后的文件
    return resolved_file
# TODO: 废弃 `get_file_from_repo` 或以不同方式记录？
#       文档字符串与 `cached_repo` 完全相同，但行为略有不同。如果文件丢失或连接错误，`cached_repo` 将返回 None，而 `get_file_from_repo` 将引发错误。
#       我认为我们应该只保留一个方法，并有一个单一的 `raise_error` 参数（待讨论）。
def get_file_from_repo(
    # 文件路径或存储库的路径
    path_or_repo: Union[str, os.PathLike],
    # 文件名
    filename: str,
    # 缓存目录，默认为 None
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    # 是否强制下载，默认为 False
    force_download: bool = False,
    # 是否恢复下载，默认为 False
    resume_download: bool = False,
    # 代理设置，默认为 None
    proxies: Optional[Dict[str, str]] = None,
    # 令牌，默认为 None
    token: Optional[Union[bool, str]] = None,
    # 版本，默认为 None
    revision: Optional[str] = None,
    # 仅本地文件，默认为 False
    local_files_only: bool = False,
    # 子文件夹，默认为空字符串
    subfolder: str = "",
    # **deprecated_kwargs 参数，已弃用
    **deprecated_kwargs,
):
    """
    尝试在本地文件夹和存储库中定位文件，如有必要则下载并缓存。
    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
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
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("xlm-roberta-base", "tokenizer_config.json")
    ```
    """
    # Check if the deprecated argument 'use_auth_token' is provided and assign it to 'use_auth_token' variable
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 不为 None，则发出警告，提示该参数已被弃用，并将在 Transformers 的 v5 版本中移除，建议使用 token 参数代替
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 参数也不为 None，则抛出数值错误，提示只能设置一个参数，要么是 token，要么是 use_auth_token
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 参数设置为 use_auth_token 参数的值
        token = use_auth_token

    # 调用 cached_file 函数，传入各个参数，并返回结果
    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
# 下载给定 URL 的内容到临时文件中。此函数不适合在多个进程中使用。它的唯一用途是允许下载配置/模型的过时行为，而不是使用 Hub。

def download_url(url, proxies=None):
    # 发出警告，提示使用 `from_pretrained` 与文件的 URL（这里是 {url}）已被弃用，在 Transformers 的 v5 版本中将不再支持。应该将文件托管在 Hub（hf.co）上，并使用存储库 ID。请注意，这与缓存系统不兼容（每次执行都会下载文件），也不兼容多个进程（每个进程将在不同的临时文件中下载文件）。
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    # 创建临时文件并返回文件描述符和文件名
    tmp_fd, tmp_file = tempfile.mkstemp()
    # 使用二进制模式打开文件描述符，准备写入下载的内容
    with os.fdopen(tmp_fd, "wb") as f:
        # 使用 HTTP GET 请求下载 URL 的内容到临时文件中
        http_get(url, f, proxies=proxies)
    # 返回临时文件的位置
    return tmp_file


# 检查存储库是否包含给定文件，而无需下载它。适用于远程存储库和本地文件夹。

def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    **deprecated_kwargs,
):
    # 弹出并获取 `use_auth_token` 参数的值
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 `use_auth_token` 参数不为空，则发出警告，并建议使用 `token` 参数
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 `token` 参数不为空，则引发 ValueError
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 `use_auth_token` 参数的值赋给 `token` 参数
        token = use_auth_token

    # 如果路径是一个目录，则检查目录中是否存在指定的文件
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    # 构建 HF Hub URL，用于获取文件的 URL
    url = hf_hub_url(path_or_repo, filename=filename, revision=revision)
    # 构建 HTTP 请求头
    headers = build_hf_headers(token=token, user_agent=http_user_agent())

    # 发送 HEAD 请求到 URL，检查文件是否存在
    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        # 检查请求的状态码是否正常
        hf_raise_for_status(r)
        # 如果状态码正常，则返回 True，表示文件存在
        return True
    # 捕获 GatedRepoError 异常
    except GatedRepoError as e:
        # 记录错误信息
        logger.error(e)
        # 抛出 EnvironmentError 异常，提供相关信息和建议
        raise EnvironmentError(
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by "
            "logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    # 捕获 RepositoryNotFoundError 异常
    except RepositoryNotFoundError as e:
        # 记录错误信息
        logger.error(e)
        # 抛出 EnvironmentError 异常，提供相关信息
        raise EnvironmentError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.")
    # 捕获 RevisionNotFoundError 异常
    except RevisionNotFoundError as e:
        # 记录错误信息
        logger.error(e)
        # 抛出 EnvironmentError 异常，提供相关信息
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    # 捕获 requests.HTTPError 异常
    except requests.HTTPError:
        # 返回 False，表示未找到条目或连接错误
        # 逻辑上也包括 EntryNotFoundError
        return False
    class PushToHubMixin:
        """
        A Mixin containing the functionality to push a model or tokenizer to the hub.
        """

        def _create_repo(
            self,
            repo_id: str,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            repo_url: Optional[str] = None,
            organization: Optional[str] = None,
        ) -> str:
            """
            Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
            the token.
            """
            # 如果传入了 repo_url 参数，则发出警告并使用 repo_url 替换 repo_id
            if repo_url is not None:
                warnings.warn(
                    "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                    "instead."
                )
                if repo_id is not None:
                    raise ValueError(
                        "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                    )
                repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
            # 如果传入了 organization 参数，则发出警告并将 organization 添加到 repo_id 中
            if organization is not None:
                warnings.warn(
                    "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                    "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
                )
                if not repo_id.startswith(organization):
                    if "/" in repo_id:
                        repo_id = repo_id.split("/")[-1]
                    repo_id = f"{organization}/{repo_id}"

            # 调用 create_repo 函数创建仓库，并返回仓库的 URL
            url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
            return url.repo_id

        def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
            """
            Returns the list of files with their last modification timestamp.
            """
            # 返回工作目录中文件的最后修改时间戳字典
            return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

        def _upload_modified_files(
            self,
            working_dir: Union[str, os.PathLike],
            repo_id: str,
            files_timestamps: Dict[str, float],
            commit_message: Optional[str] = None,
            token: Optional[Union[bool, str]] = None,
            create_pr: bool = False,
            revision: str = None,
            commit_description: str = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        # 如果没有指定提交消息，则根据类名自动生成
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        # 获取所有修改过的文件
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]

        # 过滤出实际文件和文件夹（根目录）
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        operations = []
        # 上传独立文件
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # 遍历文件夹中的各个文件
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        # 如果指定了修订版本，则创建分支
        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)

        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        # 创建提交
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
# 发送示例遥测数据，用于跟踪示例的使用情况
def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    # 如果处于离线模式，则直接返回
    if is_offline_mode():
        return

    # 构建遥测数据字典
    data = {"example": example_name, "framework": framework}
    # 遍历示例参数
    for args in example_args:
        # 将参数转换为字典形式
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        # 提取模型名称
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # 过滤掉本地路径
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        # 提取数据集名称
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # 从示例名称中提取脚本名称
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    # 在后台发送遥测数据
    send_telemetry(
        topic="examples", library_name="transformers", library_version=__version__, user_agent=http_user_agent(data)
    )


# 将以字符串表示的大小（如 `"5MB"`）转换为整数（以字节为单位）
def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    # 如果大小已经是整数，则直接返回
    if isinstance(size, int):
        return size
    # 根据单位转换大小为字节
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    # 如果大小格式不正确，则抛出异常
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


# 获取检查点分片文件
def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    # 是否恢复下载，默认为 False
    resume_download=False,
    # 是否仅使用本地文件，默认为 False
    local_files_only=False,
    # 访问令牌，初始值为 None
    token=None,
    # 用户代理，初始值为 None
    user_agent=None,
    # 版本号，初始值为 None
    revision=None,
    # 子文件夹名称，初始值为空字符串
    subfolder="",
    # 提交哈希值，初始值为 None
    _commit_hash=None,
    # 其他已废弃的参数，以字典形式传入
    **deprecated_kwargs,
# 定义一个函数，用于下载和缓存分片检查点的所有分片，如果`pretrained_model_name_or_path`是Hub上的模型ID，则返回所有分片的路径列表以及一些元数据
def _load_sharded_checkpoint(
    pretrained_model_name_or_path, index_filename, cache_dir=None, force_download=False, resume_download=False, proxies=None, local_files_only=False, use_auth_token=None, **deprecated_kwargs
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json

    # 从deprecated_kwargs中弹出"use_auth_token"参数
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果use_auth_token不为None，则发出警告
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果token也不为None，则引发ValueError异常
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # 如果index_filename不是文件，则引发ValueError异常
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    # 读取index_filename中的内容并解析为JSON格式
    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    # 对分片文件名进行排序并去重
    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # 首先处理本地文件夹
    if os.path.isdir(pretrained_model_name_or_path):
        # 将分片文件名与文件夹路径拼接成完整路径
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # 在这个阶段，pretrained_model_name_or_path是Hub上的模型标识符
    cached_filenames = []
    # 检查模型是否已经缓存。我们只尝试最后一个检查点，这���该涵盖大多数情况（如果中断下载）
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    # 如果last_shard为None或force_download为True，则显示进度条
    show_progress_bar = last_shard is None or force_download
    # 遍历分片文件名列表，并显示下载进度条
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # 从URL加载缓存文件
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        # 在获取索引时，已经处理过 RepositoryNotFoundError 和 RevisionNotFoundError，因此这里不需要再捕获它们
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        # 将缓存的文件名添加到列表中
        cached_filenames.append(cached_filename)

    # 返回缓存的文件名列表和分片的元数据
    return cached_filenames, sharded_metadata
# 下面的所有内容是用于旧缓存格式和新缓存格式之间的转换。

# 获取所有缓存文件的列表，包括相应的元数据
def get_all_cached_files(cache_dir=None):
    """
    返回所有具有适当元数据的缓存文件列表。
    """
    如果未提供缓存目录，则使用默认目录 TRANSFORMERS_CACHE
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = str(cache_dir)
    如果缓存目录不存在，则返回一个空列表
    if not os.path.isdir(cache_dir):
        return []

    cached_files = []
    遍历缓存目录中的所有文件
    for file in os.listdir(cache_dir):
        meta_path = os.path.join(cache_dir, f"{file}.json")
        如果对应的元数据文件不存在，则跳过
        if not os.path.isfile(meta_path):
            continue

        读取元数据文件并加载 JSON 数据
        with open(meta_path, encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
            提取 URL 和 ETag，并存储在字典中
            url = metadata["url"]
            etag = metadata["etag"].replace('"', "")
            cached_files.append({"file": file, "url": url, "etag": etag})

    返回缓存文件列表
    return cached_files


# 从 URL 提取仓库名、版本和文件名
def extract_info_from_url(url):
    """
    从 URL 中提取仓库名、版本和文件名。
    """
    使用正则表达式从 URL 中匹配提取仓库名、版本和文件名
    search = re.search(r"^https://huggingface\.co/(.*)/resolve/([^/]*)/(.*)$", url)
    如果匹配失败，则返回 None
    if search is None:
        return None
    分组提取仓库、版本和文件名，并存储在字典中返回
    repo, revision, filename = search.groups()
    cache_repo = "--".join(["models"] + repo.split("/"))
    return {"repo": cache_repo, "revision": revision, "filename": filename}


# 创建并附上标签的模型卡
def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    """
    创建或加载现有模型卡并对其进行标记。

    参数:
        repo_id（`str`）：
            模型卡所在的 repo_id。
        tags（`List[str]`，*可选*）：
            要添加到模型卡中的标签列表。
        token（`str`，*可选*）：
            认证令牌，使用`huggingface_hub.HfApi.login`方法获取。将默认使用存储的令牌。
        ignore_metadata_errors（`str`）：
            如果为True，解析元数据部分时将忽略错误。在此过程中可能会丢失一些信息。请自行承担风险。
    """
    尝试：
        # 检查��程仓库上是否存在模型卡
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    除了EntryNotFoundError异常以外:
        # 否则，从模板创建简单的模型卡
        model_description = "这是 🤗 Transformers 模型的模型卡，已推送至 Hub。此模型卡已自动生成。"
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    如果提供了标签：
        遍历每个标签，如果标签不在模型卡中则添加
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    返回模型卡
    return model_card


# 清理文件
def clean_files_for(file):
    """
    如果存在，删除文件、文件.json 和文件.lock。
    """
    # 遍历文件列表，包括原始文件名、添加 .json 后缀的文件名和添加 .lock 后缀的文件名
    for f in [file, f"{file}.json", f"{file}.lock"]:
        # 检查当前文件是否存在
        if os.path.isfile(f):
            # 如果文件存在，删除该文件
            os.remove(f)
# 将文件移动到新的缓存目录，遵循新的 Hugging Face Hub 缓存组织方式
def move_to_new_cache(file, repo, filename, revision, etag, commit_hash):
    # 创建目标仓库目录，如果不存在则创建
    os.makedirs(repo, exist_ok=True)

    # refs 目录
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    # 如果 revision 和 commit_hash 不同，写入 revision 对应的 commit_hash 到 refs 目录下
    if revision != commit_hash:
        ref_path = os.path.join(repo, "refs", revision)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

    # blobs 目录
    os.makedirs(os.path.join(repo, "blobs"), exist_ok=True)
    # 将文件移动到 blobs 目录下，并以 etag 命名
    blob_path = os.path.join(repo, "blobs", etag)
    shutil.move(file, blob_path)

    # snapshots 目录
    os.makedirs(os.path.join(repo, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(repo, "snapshots", commit_hash), exist_ok=True)
    # 在 snapshots 目录下创建指向 blobs 目录下文件的相对符号链接
    pointer_path = os.path.join(repo, "snapshots", commit_hash, filename)
    huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
    # 清理文件
    clean_files_for(file)


# 移动缓存
def move_cache(cache_dir=None, new_cache_dir=None, token=None):
    # 如果未提供新缓存目录，则使用默认 TRANSFORMERS_CACHE
    if new_cache_dir is None:
        new_cache_dir = TRANSFORMERS_CACHE
    # 如果未提供缓存目录，则尝试迁移旧缓存
    if cache_dir is None:
        # 旧缓存目录在 .cache/huggingface/transformers
        old_cache = Path(TRANSFORMERS_CACHE).parent / "transformers"
        # 如果旧缓存目录存在，则使用它，否则使用新缓存目录
        if os.path.isdir(str(old_cache)):
            cache_dir = str(old_cache)
        else:
            cache_dir = new_cache_dir
    # 获取所有缓存文件列表
    cached_files = get_all_cached_files(cache_dir=cache_dir)
    # 输出日志，显示将移动文件的数量
    logger.info(f"Moving {len(cached_files)} files to the new cache system")

    hub_metadata = {}
    # 遍历所有缓存文件
    for file_info in tqdm(cached_files):
        # 移除 url 字段，将其用作字典键
        url = file_info.pop("url")
        # 如果 url 不在 hub_metadata 中，则尝试获取其元数据
        if url not in hub_metadata:
            try:
                hub_metadata[url] = get_hf_file_metadata(url, token=token)
            # 若获取失败，则跳过该文件
            except requests.HTTPError:
                continue

        etag, commit_hash = hub_metadata[url].etag, hub_metadata[url].commit_hash
        # 如果 etag 或 commit_hash 为 None，则跳过该文件
        if etag is None or commit_hash is None:
            continue

        # 如果缓存文件的 etag 与元数据中的 etag 不同，则说明文件已过期，直接清理并跳过
        if file_info["etag"] != etag:
            clean_files_for(os.path.join(cache_dir, file_info["file"]))
            continue

        # 提取 url 中的信息
        url_info = extract_info_from_url(url)
        # 如果提取失败，则跳过该文件
        if url_info is None:
            continue

        # 构建目标仓库目录路径
        repo = os.path.join(new_cache_dir, url_info["repo"])
        # 调用 move_to_new_cache 函数移动文件
        move_to_new_cache(
            file=os.path.join(cache_dir, file_info["file"]),
            repo=repo,
            filename=url_info["filename"],
            revision=url_info["revision"],
            etag=etag,
            commit_hash=commit_hash,
        )


# 表示正在进行的推送的内部类，用于跟踪推送过程中的状态
class PushInProgress:
    """
    Internal class to keep track of a push in progress (which might contain multiple `Future` jobs).
    """

    def __init__(self, jobs: Optional[futures.Future] = None) -> None:
        # 初始化推送任务列表
        self.jobs = [] if jobs is None else jobs

    # 检查推送是否完成
    def is_done(self):
        return all(job.done() for job in self.jobs)
    # 等待直到所有任务完成
    def wait_until_done(self):
        # 使用futures模块中的wait函数等待任务列表中的任务完成
        futures.wait(self.jobs)

    # 取消任务
    def cancel(self) -> None:
        # 将任务列表中未开始的任务取消，并移除已取消或已完成的任务
        self.jobs = [
            job
            for job in self.jobs
            # 如果任务没有被开始则取消，并从列表中移除取消或完成的任务
            if not (job.cancel() or job.done())
        ]
# 定义缓存版本文件的路径
cache_version_file = os.path.join(TRANSFORMERS_CACHE, "version.txt")
# 如果缓存版本文件不存在，则将缓存版本设置为0
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    # 否则，从缓存版本文件中读取缓存版本号
    with open(cache_version_file) as f:
        try:
            cache_version = int(f.read())
        except ValueError:
            # 如果无法解析版本号，则将缓存版本设置为0
            cache_version = 0

# 检查缓存目录是否存在且非空
cache_is_not_empty = os.path.isdir(TRANSFORMERS_CACHE) and len(os.listdir(TRANSFORMERS_CACHE)) > 0

# 如果缓存版本小于1且缓存目录非空，则执行缓存迁移操作
if cache_version < 1 and cache_is_not_empty:
    if is_offline_mode():
        # 在离线模式下警告用户缓存需要更新
        logger.warning(
            "You are offline and the cache for model files in Transformers v4.22.0 has been updated while your local "
            "cache seems to be the one of a previous version. It is very likely that all your calls to any "
            "`from_pretrained()` method will fail. Remove the offline mode and enable internet connection to have "
            "your cache be updated automatically, then you can go back to offline mode."
        )
    else:
        # 提示用户缓存已更新，并开始迁移操作
        logger.warning(
            "The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a "
            "one-time only operation. You can interrupt this and resume the migration later on by calling "
            "`transformers.utils.move_cache()`."
        )
    try:
        if TRANSFORMERS_CACHE != constants.HF_HUB_CACHE:
            # 如果用户自定义了缓存路径，则使用自定义的缓存路径进行迁移
            move_cache(TRANSFORMERS_CACHE, TRANSFORMERS_CACHE)
        else:
            # 否则使用默认的缓存路径进行迁移
            move_cache()
    except Exception as e:
        # 如果迁移过程中出现异常，记录异常信息
        trace = "\n".join(traceback.format_tb(e.__traceback__))
        logger.error(
            f"There was a problem when trying to move your cache:\n\n{trace}\n{e.__class__.__name__}: {e}\n\nPlease "
            "file an issue at https://github.com/huggingface/transformers/issues/new/choose and copy paste this whole "
            "message and we will do our best to help."
        )

# 如果缓存版本小于1，则将缓存版本更新为1，并创建缓存目录
if cache_version < 1:
    try:
        os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        # 如果无法创建缓存目录或写入版本文件，则提示用户设置可写的缓存目录
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({TRANSFORMERS_CACHE}). You should set "
            "the environment variable TRANSFORMERS_CACHE to a writable directory."
        )
```