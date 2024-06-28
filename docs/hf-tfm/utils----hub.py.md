# `.\utils\hub.py`

```py
# 标准版权声明，声明此代码版权归 HuggingFace 团队所有
#
# 根据 Apache License, Version 2.0 许可证进行许可，除非符合许可证要求，否则不得使用此文件
#
# 导入必要的库和模块
import json  # 导入处理 JSON 的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
import shutil  # 导入文件操作相关模块
import sys  # 导入系统相关的模块
import tempfile  # 导入临时文件目录相关模块
import traceback  # 导入追踪异常的模块
import warnings  # 导入警告处理模块
from concurrent import futures  # 导入并发处理模块
from pathlib import Path  # 导入处理路径相关功能的模块
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块
from urllib.parse import urlparse  # 导入处理 URL 解析的模块
from uuid import uuid4  # 导入生成 UUID 的模块

import huggingface_hub  # 导入 HuggingFace Hub 库
import requests  # 导入处理 HTTP 请求的模块
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
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get  # 导入文件下载相关模块
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
from huggingface_hub.utils._deprecation import _deprecate_method  # 导入废弃方法相关模块
from requests.exceptions import HTTPError  # 导入处理 HTTP 错误的模块

from . import __version__, logging  # 导入当前模块的版本和日志模块
from .generic import working_or_temp_dir  # 导入通用功能的临时工作目录处理模块
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)  # 导入导入相关的实用工具模块

from .logging import tqdm  # 导入显示进度条的日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为 logger，禁止 pylint 提示

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    # 返回当前是否处于离线模式的布尔值
    return _is_offline_mode


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
default_cache_path = constants.default_cache_path  # 获取默认缓存路径
old_default_cache_path = os.path.join(torch_cache_home, "transformers")

# 确定默认缓存目录。为了向后兼容性，考虑了大量遗留环境变量。
# 最佳设置缓存路径的方式是使用环境变量 HF_HOME。详情请查看文档页：https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables.
#
# 在代码中，使用 `HF_HUB_CACHE` 作为默认缓存路径。这个变量由库设置，保证设置为正确的值。
#
# TODO: 为 v5 版本进行清理？
# 设置用于缓存预训练 BERT 模型的环境变量，如果未设置则使用默认值 constants.HF_HUB_CACHE
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
# 设置用于缓存 PyTorch Transformers 的环境变量，如果未设置则使用上面设置的 PYTORCH_PRETRAINED_BERT_CACHE
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
# 设置用于缓存 Transformers 的环境变量，如果未设置则使用上面设置的 PYTORCH_TRANSFORMERS_CACHE
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

# 如果旧的默认缓存路径存在，并且新的缓存路径 constants.HF_HUB_CACHE 不存在，并且没有设置相关环境变量，则执行一次性迁移
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(constants.HF_HUB_CACHE)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    # 发出警告说明在 Transformers v4.22.0 中，默认的模型下载缓存路径从 '~/.cache/torch/transformers' 变更为 '~/.cache/huggingface/hub'
    # 由于当前 '~/.cache/torch/transformers' 存在且未被覆盖，所以执行移动操作到 '~/.cache/huggingface/hub'，避免重复下载已缓存的模型
    logger.warning(
        "In Transformers v4.22.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/hub'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/hub' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    # 执行实际的文件移动操作
    shutil.move(old_default_cache_path, constants.HF_HUB_CACHE)

# 设置用于缓存 HF 模块的环境变量，默认路径为 constants.HF_HOME 下的 modules 文件夹
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
# 定义 Transformers 动态模块的名称
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
# 生成一个新的会话 ID，用于标识当前会话的唯一性
SESSION_ID = uuid4().hex

# 对旧的环境变量进行弃用警告
for key in ("PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE"):
    if os.getenv(key) is not None:
        # 发出警告，提示使用这些环境变量已经被弃用，建议在 Transformers v5 中使用 `HF_HOME` 替代
        warnings.warn(
            f"Using `{key}` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.",
            FutureWarning,
        )

# 定义 Hugging Face 模型存储在 S3 桶中的前缀
S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
# 定义 Hugging Face CDN 的分发前缀
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

# 检查是否处于 Hugging Face CO 的测试阶段，根据环境变量 HUGGINGFACE_CO_STAGING 来判断
_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
# 根据是否处于测试阶段选择默认的 API 终端地址
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"

# 设置用于解析模型的终端地址，默认为 _default_endpoint
HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
# 如果设置了环境变量 HUGGINGFACE_CO_RESOLVE_ENDPOINT，则发出警告提示使用 HF_ENDPOINT 替代
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    # 使用环境变量 HF_ENDPOINT 的值来更新 HUGGINGFACE_CO_RESOLVE_ENDPOINT
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
# 使用环境变量 HF_ENDPOINT 的值来更新 HUGGINGFACE_CO_RESOLVE_ENDPOINT
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
# 构建 Hugging Face CO 的模型解析路径模板
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
# 构建用于上报示例数据的 Telemetry API 地址
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"


def _get_cache_file_to_return(
    path_or_repo_id: str, full_filename: str, cache_dir: Union[str, Path, None] = None, revision: Optional[str] = None
):
    # 尝试查找缓存文件，如果存在且未过期，则返回该文件（未完成的部分）
    # 定义函数：尝试从缓存中加载文件并返回路径，若找不到或不存在，则返回None
    def try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=None, revision=None):
        # 调用尝试从缓存中直接加载文件路径的内部函数(未提供，需根据实际情况补充实现)
        resolved_file = try_to_load_from_cache_inner(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
        # 若返回的路径不为None且不是缓存不存在的标志(_CACHED_NO_EXIST)，说明找到了正确路径
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            # 返回找到的文件路径
            return resolved_file
        # 如果上述条件均不满足，说明未找到文件或存在异常情况，返回None
        return None
    
    
    # 注意事项中未提供try_to_load_from_cache_inner的具体实现，以下是假设实现供参考
    
    
    def try_to_load_from_cache_inner(path_or_repo_id, full_filename, cache_dir=None, revision=None):
        # 数据处理逻辑：尝试使用给定的repo_id和full_filename从缓存目录cache_dir中加载文件路径
        cache_file_path = None
        if cache_dir is not None:
            # 构建缓存文件路径
            cache_file_path = os.path.join(cache_dir, full_filename)
            # 检查缓存文件是否存在
            if os.path.exists(cache_file_path):
                # 省略读取缓存、验证等具体逻辑，假设通过验证后返回有效文件路径
                # 这里返回一个伪有效路径，代表实际应从缓存中加载文件路径的具体处理
                # 在实际应用中应增强逻辑以正确处理缓存文件的读取和验证
                return os.fspath(cache_file_path)
            else:
                # 文件不存在于缓存目录，不返回任何值表示未找到缓存文件
                return None
        else:
            # 缓存目录未指定，返回None表示不尝试从缓存加载文件
            return None
# 检查给定的 URL 或文件名是否是远程 URL
def is_remote_url(url_or_filename):
    # 解析 URL 或文件名，获取其组成部分
    parsed = urlparse(url_or_filename)
    # 判断解析结果中的 scheme 是否为 http 或 https
    return parsed.scheme in ("http", "https")


# TODO: 在完全弃用后删除此函数
# TODO? 同时也要从 './examples/research_projects/lxmert/utils.py' 中移除
# TODO? 同时也要从 './examples/research_projects/visual_bert/utils.py' 中移除
@_deprecate_method(version="4.39.0", message="This method is outdated and does not support the new cache system.")
def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    返回一个列表，表示本地缓存的模型二进制文件。每个元组的形式为 `(model_url, etag, size_MB)`。
    只有以 *.bin* 结尾的 URL 文件名会被添加到列表中。

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            要在其中搜索模型的缓存目录。如果未设置，将默认使用 transformers 的缓存目录。

    Returns:
        List[Tuple]: 包含 `(model_url, etag, size_MB)` 形式的元组列表
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    # 如果缓存目录不存在，返回空列表
    if not os.path.isdir(cache_dir):
        return []

    cached_models = []
    # 遍历缓存目录中的文件
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            # 打开元数据文件并加载 JSON 数据
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                # 如果 URL 以 .bin 结尾，计算文件大小并添加到列表中
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def define_sagemaker_information():
    try:
        # 获取当前实例的容器元数据
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        # 如果获取失败，设置为 None
        dlc_container_used = None
        dlc_tag = None

    # 解析 SageMaker 框架的参数
    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    # 检查是否启用了 SageMaker 的分布式训练
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    # 从环境变量中提取账户 ID
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    # 构建包含 SageMaker 相关信息的字典对象
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
    # 格式化用户代理字符串，包含请求的基本信息
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if constants.HF_HUB_DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    返回格式化后的用户代理字符串
    """
# 从已解析的文件名中提取提交哈希值，并用于缓存文件。
def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    # 如果 resolved_file 为 None 或者 commit_hash 不为 None，则直接返回 commit_hash
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    
    # 将 resolved_file 转换为标准的 POSIX 路径字符串
    resolved_file = str(Path(resolved_file).as_posix())
    
    # 使用正则表达式在 resolved_file 中搜索匹配 'snapshots/([^/]+)/' 的内容
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    
    # 如果未找到匹配项，则返回 None
    if search is None:
        return None
    
    # 从搜索结果中获取第一个捕获组，即提取的 commit_hash
    commit_hash = search.groups()[0]
    
    # 如果提取的 commit_hash 符合预期的格式（由 REGEX_COMMIT_HASH 定义），则返回 commit_hash，否则返回 None
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


# 尝试在本地文件夹或存储库中定位文件，如果必要则下载并缓存它。
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
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
    """
    # 获取 deprecated_kwargs 字典中的 use_auth_token 键对应的值，并将其从字典中移除
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 参数不为 None，则发出警告信息，说明该参数已弃用，并将在 Transformers 版本 v5 中移除。建议使用 `token` 参数替代。
    # 引发 FutureWarning 警告。
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了 token 参数和 use_auth_token 参数，则抛出 ValueError 异常。
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 token 参数设置为 use_auth_token 参数的值。
        token = use_auth_token

    # Private arguments
    #     _raise_exceptions_for_gated_repo: if False, do not raise an exception for gated repo error but return
    #         None.
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.

    # 如果处于离线模式且 local_files_only 参数为 False，则设置 local_files_only 参数为 True，并输出相应的日志信息。
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # 如果 subfolder 参数为 None，则将其设置为空字符串。
    if subfolder is None:
        subfolder = ""

    # 将 path_or_repo_id 参数转换为字符串。
    path_or_repo_id = str(path_or_repo_id)
    # 将 subfolder 和 filename 参数拼接成完整的文件路径。
    full_filename = os.path.join(subfolder, filename)

    # 如果 path_or_repo_id 参数指定的路径是一个目录，则解析文件路径并检查文件是否存在。
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        # 如果解析后的文件路径不是一个文件，并且 _raise_exceptions_for_missing_entries 参数为 True，则抛出 EnvironmentError 异常。
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
                )
            # 如果 _raise_exceptions_for_missing_entries 参数为 False，则返回 None。
            else:
                return None
        # 返回解析后的文件路径。
        return resolved_file

    # 如果 cache_dir 参数为 None，则将其设置为 TRANSFORMERS_CACHE 变量的值。
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE

    # 如果 cache_dir 参数是 Path 对象，则将其转换为字符串。
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    # 如果 _commit_hash 参数不为 None 并且 force_download 参数为 False，则尝试从缓存中加载文件。
    if _commit_hash is not None and not force_download:
        # 如果文件在指定的 _commit_hash 下被缓存，则直接返回该文件。
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )
        # 如果成功加载文件，则根据情况返回解析后的文件路径、None 或抛出异常。
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    # 调用 http_user_agent 函数来处理 user_agent 参数。
    user_agent = http_user_agent(user_agent)
    try:
        # 尝试从 URL 或缓存加载文件
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
        # 如果遇到受限制的仓库错误，则尝试从缓存中获取文件以返回
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        # 如果已获取文件或不应为受限制的仓库错误引发异常，则返回解析的文件
        if resolved_file is not None or not _raise_exceptions_for_gated_repo:
            return resolved_file
        # 否则，引发环境错误并显示详细信息
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to have access to it at "
            f"https://huggingface.co/{path_or_repo_id}.\n{str(e)}"
        ) from e
    except RepositoryNotFoundError as e:
        # 如果仓库未找到，则引发环境错误并显示详细信息
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        # 如果找不到指定的版本号，则引发环境错误并显示详细信息
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        # 如果本地条目未找到，则尝试从缓存获取文件以返回
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        # 如果已获取文件或不应为丢失条目或连接错误引发异常，则返回解析的文件
        if (
            resolved_file is not None
            or not _raise_exceptions_for_missing_entries
            or not _raise_exceptions_for_connection_errors
        ):
            return resolved_file
        # 否则，引发环境错误并显示详细信息
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ) from e
    # 处理 EntryNotFoundError 异常，如果设置了不抛出缺失条目异常，则返回 None
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        # 如果未指定修订版本，则默认为 "main"
        if revision is None:
            revision = "main"
        # 抛出环境错误，指示指定的路径或 repo_id 中不存在指定的完整文件名
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
        ) from e
    # 处理 HTTPError 异常
    except HTTPError as err:
        # 尝试获取缓存中已解决的文件，如果存在或设置了不抛出连接错误异常，则返回该文件
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_connection_errors:
            return resolved_file
        # 抛出环境错误，指示加载指定路径或 repo_id 时发生特定的连接错误
        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
    # 处理 HFValidationError 异常
    except HFValidationError as e:
        # 抛出环境错误，指示路径或模型 ID 的提供方式不正确，应提供本地文件夹的路径或 Hub 上模型的 repo_id
        raise EnvironmentError(
            f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a local folder or the repo_id of a model on the Hub."
        ) from e
    # 返回已解决的文件（如果有）
    return resolved_file
# TODO: deprecate `get_file_from_repo` or document it differently?
#       Docstring is exactly the same as `cached_repo` but behavior is slightly different. If file is missing or if
#       there is a connection error, `cached_repo` will return None while `get_file_from_repo` will raise an error.
#       IMO we should keep only 1 method and have a single `raise_error` argument (to be discussed).
# 定义了一个函数 `get_file_from_repo`，用于从本地文件夹或仓库中获取文件，并在需要时下载和缓存它。
def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],  # 参数1: 文件路径或仓库位置，可以是字符串或PathLike对象
    filename: str,                          # 参数2: 文件名，表示需要获取的文件名
    cache_dir: Optional[Union[str, os.PathLike]] = None,  # 参数3: 缓存目录的路径，可选，默认为None
    force_download: bool = False,           # 参数4: 是否强制下载文件，默认为False
    resume_download: bool = False,          # 参数5: 是否继续下载（即断点续传），默认为False
    proxies: Optional[Dict[str, str]] = None,  # 参数6: 代理设置，可选，默认为None
    token: Optional[Union[bool, str]] = None,   # 参数7: 访问令牌，可选，默认为None
    revision: Optional[str] = None,         # 参数8: 仓库的版本或分支，可选，默认为None
    local_files_only: bool = False,          # 参数9: 是否只使用本地文件，不从仓库下载，默认为False
    subfolder: str = "",                    # 参数10: 仓库中的子文件夹路径，默认为空字符串
    **deprecated_kwargs,                    # 其他已废弃的关键字参数将被收集到deprecated_kwargs中
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
    """
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

    ```
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("google-bert/bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("FacebookAI/xlm-roberta-base", "tokenizer_config.json")
    ```
    """
    # Check for deprecated argument and assign its value to use_auth_token
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    # 如果 use_auth_token 参数不为 None，则发出警告，指出该参数将在 Transformers 的 v5 版本中被移除，建议使用 token 参数。
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果 token 参数也不为 None，则抛出 ValueError，因为不能同时指定 `token` 和 `use_auth_token`。
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 use_auth_token 的值赋给 token 参数
        token = use_auth_token

    # 调用 cached_file 函数，传入各种参数，并返回其结果
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
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
# 下载指定的 URL 对应的文件到临时文件中
def download_url(url, proxies=None):
    # 发出警告，提醒使用者此函数即将在 Transformers v5 版本中被移除
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    # 创建临时文件并返回其文件名
    tmp_fd, tmp_file = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "wb") as f:
        # 使用 HTTP GET 方法下载 URL 对应的文件到临时文件中
        http_get(url, f, proxies=proxies)
    # 返回临时文件的路径
    return tmp_file


# 检查指定的仓库或路径是否包含指定的文件，支持远程仓库和本地文件夹
def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    **deprecated_kwargs,
):
    # 如果使用的是已弃用的参数 `use_auth_token`，则发出警告并将其转换到 `token` 参数
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # 如果路径为一个目录，则直接检查目录下是否有指定的文件
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    # 构建 Hub 的 URL，并获取相应的 headers
    url = hf_hub_url(path_or_repo, filename=filename, revision=revision)
    headers = build_hf_headers(token=token, user_agent=http_user_agent())

    # 发送 HEAD 请求到指定的 URL，检查文件是否存在
    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        # 检查请求的状态，如果正常则返回 True，否则引发异常
        hf_raise_for_status(r)
        return True
    # 如果捕获到 GatedRepoError 异常，则记录错误信息并抛出 EnvironmentError 异常
    except GatedRepoError as e:
        logger.error(e)
        raise EnvironmentError(
            # 指定路径或资源 {path_or_repo} 是一个受保护的仓库。请确保在 'https://huggingface.co/{path_or_repo}' 请求访问权限，
            # 并通过 `huggingface-cli login` 登录或通过 `token=<your_token>` 传递具有访问权限的令牌。
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by "
            "logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    # 如果捕获到 RepositoryNotFoundError 异常，则记录错误信息并抛出 EnvironmentError 异常
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            # 指定路径或资源 {path_or_repo} 不是本地文件夹或 'https://hf.co' 上的有效仓库名称。
            f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'."
        )
    # 如果捕获到 RevisionNotFoundError 异常，则记录错误信息并抛出 EnvironmentError 异常
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            # 指定的 {revision} 不是此模型名称存在的有效 git 标识符（分支名称、标签名称或提交 ID）。
            # 查看 'https://huggingface.co/{path_or_repo}' 上模型页面获取可用的修订版本。
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    # 如果捕获到 requests.HTTPError 异常，则返回 False，处理 EntryNotFoundError 和所有连接错误
    except requests.HTTPError:
        return False
    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """
        创建仓库（如果需要），清理使用了已弃用参数 `repo_url` 和 `organization` 的 `repo_id`，并获取 token。
        """
        # 如果指定了 repo_url 参数，则发出警告并处理 repo_id
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                )
            # 根据约定的终结点，修改 repo_id
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        
        # 如果指定了 organization 参数，则发出警告并调整 repo_id
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            # 如果 repo_id 不以 organization 开头，则进行调整
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"

        # 调用 create_repo 函数创建仓库，并返回 repo_id
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        返回工作目录下文件及其最后修改时间戳的字典。
        """
        # 遍历工作目录中的文件，获取它们的最后修改时间戳
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
        上传修改过的文件到指定的仓库，并支持创建 Pull Request 功能。
        """
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """

        # Determine the commit message if not provided explicitly
        if commit_message is None:
            # Set default commit messages based on the class name
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

        # Identify modified files based on timestamps and existence in `working_dir`
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]

        # Filter for actual files and folders at the root level of `working_dir`
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        operations = []

        # Upload individual files or files within directories
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # Upload files within the directory individually
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                # Upload standalone file
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        # Optionally create a new branch if `revision` is specified
        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)

        # Log the files being uploaded
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")

        # Create and return a commit with specified parameters
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
# 发送示例的遥测数据，用于跟踪示例的使用情况
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

    # 准备遥测数据的基本信息
    data = {"example": example_name, "framework": framework}

    # 遍历传入的每组参数
    for args in example_args:
        # 将参数对象转换为字典，过滤掉以"_"开头的私有属性，并且值不为None的项
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        
        # 如果参数字典中包含模型名或路径
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # 如果模型名不是一个目录，则记录模型名
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        
        # 如果参数字典中包含数据集名
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # 从示例名中提取脚本名
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            # 构建数据集名，由脚本名和任务名组成
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    # 在后台发送遥测数据
    send_telemetry(
        topic="examples", library_name="transformers", library_version=__version__, user_agent=http_user_agent(data)
    )


# 将文件大小转换为整数表示的字节数
def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    # 如果大小已经是整数类型，则直接返回
    if isinstance(size, int):
        return size
    
    # 根据大小字符串的单位后缀进行转换
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
    
    # 若无法识别大小的格式，则抛出错误
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


# 获取检查点分片文件列表
def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,  # 是否继续下载，默认为False，表示不继续下载
    local_files_only=False,  # 是否仅使用本地文件，默认为False，表示不仅使用本地文件
    token=None,  # 访问令牌，通常为None，表示未指定特定的访问令牌
    user_agent=None,  # 用户代理信息，通常为None，表示未指定特定的用户代理
    revision=None,  # 版本号，通常为None，表示未指定特定的版本号
    subfolder="",  # 子文件夹路径，默认为空字符串，表示没有特定的子文件夹
    _commit_hash=None,  # 提交哈希值，通常为None，表示未指定特定的提交哈希值
    **deprecated_kwargs,  # 其他过时的关键字参数，通过**kwargs收集
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    # 导入 json 模块
    import json

    # 处理已弃用的参数 `use_auth_token`
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        # 引发未来警告，提醒用户 `use_auth_token` 参数将在 Transformers v5 版本中删除
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        # 如果同时指定了 `use_auth_token` 和 `token`，则引发值错误
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        # 将 `use_auth_token` 赋值给 `token`
        token = use_auth_token

    # 如果指定的 `index_filename` 不是文件，则引发值错误
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    # 读取 `index_filename` 文件内容并解析为 JSON 格式，赋值给 `index` 变量
    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    # 获取所有分片文件名并排序，去重
    shard_filenames = sorted(set(index["weight_map"].values()))
    # 获取分片元数据
    sharded_metadata = index["metadata"]
    # 添加额外的检查点键到 `sharded_metadata` 中
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    # 复制 `weight_map` 到 `sharded_metadata` 中的 `weight_map` 键
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # 首先处理本地文件夹的情况
    if os.path.isdir(pretrained_model_name_or_path):
        # 构建所有分片文件的完整路径
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        # 返回分片文件名列表和分片元数据
        return shard_filenames, sharded_metadata

    # 如果代码执行到这里，说明 `pretrained_model_name_or_path` 是 Hub 上的模型标识符
    cached_filenames = []
    # 检查模型是否已经缓存。我们只尝试最后一个检查点，这应该涵盖大多数下载情况（如果下载被中断）。
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    # 如果 `last_shard` 为 None 或者强制下载被设置，则显示进度条
    show_progress_bar = last_shard is None or force_download
    # 遍历每个分片文件名列表，显示下载进度条，如果禁用了进度条则不显示
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # 尝试从URL加载文件到缓存，并返回缓存后的文件名
            cached_filename = cached_file(
                pretrained_model_name_or_path,  # 预训练模型名称或路径
                shard_filename,                 # 当前分片文件名
                cache_dir=cache_dir,            # 缓存目录路径
                force_download=force_download,  # 是否强制重新下载
                proxies=proxies,                # 代理设置
                resume_download=resume_download,  # 是否从上次中断处继续下载
                local_files_only=local_files_only,  # 仅使用本地文件
                token=token,                    # 访问令牌
                user_agent=user_agent,          # 用户代理
                revision=revision,              # 版本号
                subfolder=subfolder,            # 子文件夹
                _commit_hash=_commit_hash,      # 提交哈希值
            )
        # 已经在获取索引时处理了 RepositoryNotFoundError 和 RevisionNotFoundError，因此这里不需要捕获它们
        except EntryNotFoundError:
            # 如果缓存中找不到指定的文件，则抛出环境错误异常
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            # 如果无法连接到指定的下载端点，抛出环境错误异常，提示检查网络连接后重试
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        # 将缓存后的文件名添加到缓存文件名列表中
        cached_filenames.append(cached_filename)

    # 返回缓存文件名列表和分片元数据
    return cached_filenames, sharded_metadata
# 所有以下代码都是用于旧缓存格式和新缓存格式之间的转换。

# 返回所有已缓存文件的列表，包括相应的元数据
def get_all_cached_files(cache_dir=None):
    # 如果未指定缓存目录，则使用默认的 TRANSFORMERS_CACHE
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        # 将缓存目录转换为字符串形式
        cache_dir = str(cache_dir)
    # 如果缓存目录不存在，则返回空列表
    if not os.path.isdir(cache_dir):
        return []

    # 初始化一个空列表，用于存储已缓存的文件及其元数据
    cached_files = []
    # 遍历缓存目录下的所有文件
    for file in os.listdir(cache_dir):
        # 构建元数据文件的路径
        meta_path = os.path.join(cache_dir, f"{file}.json")
        # 如果元数据文件不存在，则跳过当前文件
        if not os.path.isfile(meta_path):
            continue

        # 打开元数据文件并加载其中的 JSON 数据
        with open(meta_path, encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
            # 提取 URL 和 ETag，并移除双引号
            url = metadata["url"]
            etag = metadata["etag"].replace('"', "")
            # 将文件名、URL 和 ETag 组成字典，加入到 cached_files 列表中
            cached_files.append({"file": file, "url": url, "etag": etag})

    # 返回所有已缓存文件及其元数据的列表
    return cached_files


# 从 URL 中提取仓库名、版本和文件名
def extract_info_from_url(url):
    # 使用正则表达式从 URL 中提取仓库名、版本和文件名
    search = re.search(r"^https://huggingface\.co/(.*)/resolve/([^/]*)/(.*)$", url)
    # 如果未能匹配到结果，则返回 None
    if search is None:
        return None
    # 提取仓库名、版本和文件名，并拼接成字典返回
    repo, revision, filename = search.groups()
    cache_repo = "--".join(["models"] + repo.split("/"))
    return {"repo": cache_repo, "revision": revision, "filename": filename}


# 创建或加载现有的模型卡，并添加标签
def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    try:
        # 尝试从远程仓库加载模型卡
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    except EntryNotFoundError:
        # 如果未找到模型卡，则从模板创建一个简单的模型卡
        model_description = "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    # 如果提供了标签列表，则逐个添加到模型卡的标签中
    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    # 返回创建或加载的模型卡对象
    return model_card


# 删除指定文件及其相关的文件（如文件的元数据文件和锁文件），如果存在的话
def clean_files_for(file):
    pass  # 这个函数定义了一个清理文件的方法，但在这里没有实现具体功能，只是占位用的
    # 对文件和其关联的 .json 和 .lock 文件进行循环操作
    for f in [file, f"{file}.json", f"{file}.lock"]:
        # 检查当前路径下是否存在指定的文件
        if os.path.isfile(f):
            # 如果存在，删除该文件
            os.remove(f)
# 创建一个函数，将文件移动到新的缓存组织中，按照新的 huggingface hub 缓存组织规则操作
def move_to_new_cache(file, repo, filename, revision, etag, commit_hash):
    # 确保目标 repo 目录存在，如果不存在则创建
    os.makedirs(repo, exist_ok=True)

    # refs 目录
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    # 如果 revision 和 commit_hash 不相同，将 commit_hash 写入到相应的 ref 文件中
    if revision != commit_hash:
        ref_path = os.path.join(repo, "refs", revision)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

    # blobs 目录
    os.makedirs(os.path.join(repo, "blobs"), exist_ok=True)
    # 将文件移动到 blobs 目录下的以 etag 命名的文件中
    blob_path = os.path.join(repo, "blobs", etag)
    shutil.move(file, blob_path)

    # snapshots 目录
    os.makedirs(os.path.join(repo, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(repo, "snapshots", commit_hash), exist_ok=True)
    # 在 snapshots 目录下的 commit_hash 子目录中创建文件名为 filename 的指针文件
    pointer_path = os.path.join(repo, "snapshots", commit_hash, filename)
    # 使用 huggingface_hub.file_download._create_relative_symlink 创建相对符号链接
    huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
    # 清理原始文件
    clean_files_for(file)


# 创建一个函数，用于迁移缓存
def move_cache(cache_dir=None, new_cache_dir=None, token=None):
    # 如果未提供 new_cache_dir，则使用默认 TRANSFORMERS_CACHE
    if new_cache_dir is None:
        new_cache_dir = TRANSFORMERS_CACHE
    # 如果未提供 cache_dir，则尝试从旧的缓存目录 .cache/huggingface/transformers 中获取
    if cache_dir is None:
        old_cache = Path(TRANSFORMERS_CACHE).parent / "transformers"
        if os.path.isdir(str(old_cache)):
            cache_dir = str(old_cache)
        else:
            cache_dir = new_cache_dir
    # 获取所有缓存文件列表
    cached_files = get_all_cached_files(cache_dir=cache_dir)
    # 记录迁移过程中的日志信息
    logger.info(f"Moving {len(cached_files)} files to the new cache system")

    # 存储 hub 元数据的字典
    hub_metadata = {}
    # 遍历所有缓存文件
    for file_info in tqdm(cached_files):
        # 获取文件的 URL，并移除文件信息中的 URL 键
        url = file_info.pop("url")
        # 如果 URL 不在 hub_metadata 中，则尝试获取文件的元数据信息并存储
        if url not in hub_metadata:
            try:
                hub_metadata[url] = get_hf_file_metadata(url, token=token)
            except requests.HTTPError:
                continue

        # 获取文件的 etag 和 commit_hash
        etag, commit_hash = hub_metadata[url].etag, hub_metadata[url].commit_hash
        # 如果 etag 或 commit_hash 为空，则跳过当前文件
        if etag is None or commit_hash is None:
            continue

        # 如果文件信息中的 etag 与当前文件的 etag 不同，清理当前文件并跳过
        if file_info["etag"] != etag:
            # 缓存文件不是最新版本，清理该文件，因为将会下载新版本
            clean_files_for(os.path.join(cache_dir, file_info["file"]))
            continue

        # 从 URL 中提取信息
        url_info = extract_info_from_url(url)
        # 如果无法从 URL 中提取信息，则跳过当前文件
        if url_info is None:
            # 不是来自 huggingface.co 的文件
            continue

        # 构建目标 repo 的路径
        repo = os.path.join(new_cache_dir, url_info["repo"])
        # 调用 move_to_new_cache 函数，将文件移动到新缓存中
        move_to_new_cache(
            file=os.path.join(cache_dir, file_info["file"]),
            repo=repo,
            filename=url_info["filename"],
            revision=url_info["revision"],
            etag=etag,
            commit_hash=commit_hash,
        )


# PushInProgress 类，用于跟踪进行中的推送（可能包含多个 Future 任务）
class PushInProgress:
    """
    Internal class to keep track of a push in progress (which might contain multiple `Future` jobs).
    """

    def __init__(self, jobs: Optional[futures.Future] = None) -> None:
        # 初始化 jobs 列表，默认为空列表
        self.jobs = [] if jobs is None else jobs

    # 检查所有任务是否完成
    def is_done(self):
        # 返回所有任务是否都已完成的布尔值
        return all(job.done() for job in self.jobs)
    # 等待所有任务完成
    def wait_until_done(self):
        # 使用 futures 模块等待所有任务完成
        futures.wait(self.jobs)

    # 取消所有未开始或已取消/已完成的任务
    def cancel(self) -> None:
        self.jobs = [
            job
            for job in self.jobs
            # 如果任务还未开始，则取消该任务；同时移除已取消或已完成的任务
            if not (job.cancel() or job.done())
        ]
# 拼接得到缓存版本文件的路径
cache_version_file = os.path.join(TRANSFORMERS_CACHE, "version.txt")

# 检查缓存版本文件是否存在，如果不存在则默认缓存版本为0
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    # 如果文件存在，则读取其中的内容并尝试转换为整数，如果无法转换则默认缓存版本为0
    with open(cache_version_file) as f:
        try:
            cache_version = int(f.read())
        except ValueError:
            cache_version = 0

# 检查 TRANSFORMERS_CACHE 目录是否存在且不为空
cache_is_not_empty = os.path.isdir(TRANSFORMERS_CACHE) and len(os.listdir(TRANSFORMERS_CACHE)) > 0

# 如果缓存版本小于1且 TRANSFORMERS_CACHE 目录不为空
if cache_version < 1 and cache_is_not_empty:
    # 如果处于离线模式，则记录警告信息
    if is_offline_mode():
        logger.warning(
            "You are offline and the cache for model files in Transformers v4.22.0 has been updated while your local "
            "cache seems to be the one of a previous version. It is very likely that all your calls to any "
            "`from_pretrained()` method will fail. Remove the offline mode and enable internet connection to have "
            "your cache be updated automatically, then you can go back to offline mode."
        )
    else:
        # 否则记录警告信息，说明模型文件的缓存已更新
        logger.warning(
            "The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a "
            "one-time only operation. You can interrupt this and resume the migration later on by calling "
            "`transformers.utils.move_cache()`."
        )

    try:
        # 尝试迁移缓存，根据是否设置了自定义的缓存路径
        if TRANSFORMERS_CACHE != constants.HF_HUB_CACHE:
            # 用户设置了某些环境变量以自定义缓存存储
            move_cache(TRANSFORMERS_CACHE, TRANSFORMERS_CACHE)
        else:
            # 否则执行默认的缓存迁移
            move_cache()
    except Exception as e:
        # 如果迁移过程中发生异常，记录错误信息和堆栈追踪
        trace = "\n".join(traceback.format_tb(e.__traceback__))
        logger.error(
            f"There was a problem when trying to move your cache:\n\n{trace}\n{e.__class__.__name__}: {e}\n\nPlease "
            "file an issue at https://github.com/huggingface/transformers/issues/new/choose and copy paste this whole "
            "message and we will do our best to help."
        )

# 如果缓存版本小于1，则尝试创建 TRANSFORMERS_CACHE 目录，并在其中写入缓存版本号为"1"
if cache_version < 1:
    try:
        os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        # 如果创建过程中发生异常，则记录警告信息
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({TRANSFORMERS_CACHE}). You should set "
            "the environment variable TRANSFORMERS_CACHE to a writable directory."
        )
```