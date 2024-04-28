# `.\transformers\utils\peft_utils.py`

```
# 导入必要的模块
import importlib
import os
from typing import Dict, Optional, Union

# 导入版本控制模块
from packaging import version

# 导入自定义模块
from .hub import cached_file
from .import_utils import is_peft_available

# 定义常量，适配器配置文件名和权重文件名
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"

# 查找适配器配置文件的函数
def find_adapter_config_file(
    model_id: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    _commit_hash: Optional[str] = None,
) -> Optional[str]:
    r"""
    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path of the adapter
    config file if it is, None otherwise.
    Args:
        model_id (`str`):
            要查找的模型的标识符，可以是本地路径，也可以是 Hub 上存储库的 id。
        cache_dir (`str` or `os.PathLike`, *optional*):
            下载预训练模型配置文件时应缓存的目录路径，如果不想使用标准缓存。
        force_download (`bool`, *optional*, defaults to `False`):
            是否强制重新下载配置文件并覆盖已存在的缓存版本。
        resume_download (`bool`, *optional*, defaults to `False`):
            是否删除接收不完整的文件。如果存在这样的文件，则尝试恢复下载。
        proxies (`Dict[str, str]`, *optional*):
            要使用的代理服务器字典，按协议或端点划分，例如，`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`。
            代理服务器将在每个请求上使用。
        token (`str` or *bool*, *optional*):
            用作远程文件的 HTTP bearer 授权的令牌。如果为 `True`，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        revision (`str`, *optional*, defaults to `"main"`):
            要使用的特定模型版本。它可以是分支名称、标签名称或提交 id，因为我们在 huggingface.co 上使用基于 git 的系统存储模型和其他工件，所以 `revision` 可以是 git 允许的任何标识符。

            <Tip>

            要测试您在 Hub 上提交的拉取请求，可以传递 `revision="refs/pr/<pr_number>"。

            </Tip>

        local_files_only (`bool`, *optional*, defaults to `False`):
            如果为 `True`，将仅尝试从本地文件加载 tokenizer 配置。
        subfolder (`str`, *optional*, defaults to `""`):
            如果相关文件位于 huggingface.co 上模型存储库的子文件夹中，可以在此处指定文件夹名称。
    """
    # 适配器缓存文件名初始化为 None
    adapter_cached_filename = None
    # 如果模型标识符为 None，则返回 None
    if model_id is None:
        return None
    # 如果模型标识符是一个目录
    elif os.path.isdir(model_id):
        # 列出目录中的文件
        list_remote_files = os.listdir(model_id)
        # 如果适配器配置文件在列表中
        if ADAPTER_CONFIG_NAME in list_remote_files:
            # 设置适配器缓存文件名为适配器配置文件的路径
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    else:
        # 否则，从缓存或远程下载适配器配置文件
        adapter_cached_filename = cached_file(
            model_id,
            ADAPTER_CONFIG_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            subfolder=subfolder,
            _commit_hash=_commit_hash,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

    # 返回适配器缓存文件名
    return adapter_cached_filename
# 检查 PEFT 的版本是否兼容
def check_peft_version(min_version: str) -> None:
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    # 检查是否已安装 PEFT
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    # 检查 PEFT 的版本是否兼容
    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version)

    # 如果版本不兼容，则抛出异常
    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )
```