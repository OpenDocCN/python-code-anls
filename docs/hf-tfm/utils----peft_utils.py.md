# `.\utils\peft_utils.py`

```py
# 导入模块和库
import importlib  # 用于动态导入模块
import os  # 系统操作相关功能
from typing import Dict, Optional, Union  # 引入类型提示

# 从外部导入相关模块和函数
from packaging import version  # 版本管理
from .hub import cached_file  # 从本地导入 cached_file 函数
from .import_utils import is_peft_available  # 从本地导入 is_peft_available 函数

# 定义常量：适配器配置文件名、适配器模型权重文件名、安全张量适配器模型权重文件名
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"


def find_adapter_config_file(
    model_id: str,  # 模型标识符
    cache_dir: Optional[Union[str, os.PathLike]] = None,  # 缓存目录，默认为 None
    force_download: bool = False,  # 是否强制下载，默认为 False
    resume_download: bool = False,  # 是否恢复下载，默认为 False
    proxies: Optional[Dict[str, str]] = None,  # 代理设置，默认为 None
    token: Optional[Union[bool, str]] = None,  # 访问令牌，默认为 None
    revision: Optional[str] = None,  # 版本标识符，默认为 None
    local_files_only: bool = False,  # 仅使用本地文件，默认为 False
    subfolder: str = "",  # 子文件夹，默认为空字符串
    _commit_hash: Optional[str] = None,  # 提交哈希值，默认为 None
) -> Optional[str]:
    r"""
    简单检查存储在 Hub 或本地的模型是否为适配器模型，如果是，则返回适配器配置文件的路径，否则返回 None。
    """
    # 初始化一个变量用于存储适配器配置文件的本地路径，默认为 None
    adapter_cached_filename = None
    # 如果 model_id 为空，则直接返回 None
    if model_id is None:
        return None
    # 如果 model_id 是一个目录路径
    elif os.path.isdir(model_id):
        # 获取该目录下的所有文件列表
        list_remote_files = os.listdir(model_id)
        # 检查 ADAPTER_CONFIG_NAME 是否在文件列表中
        if ADAPTER_CONFIG_NAME in list_remote_files:
            # 如果找到了适配器配置文件，设置适配器配置文件的本地路径
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    # 否则，从缓存中获取适配器配置文件的文件名
    adapter_cached_filename = cached_file(
        model_id,  # 模型ID
        ADAPTER_CONFIG_NAME,  # 适配器配置文件名
        cache_dir=cache_dir,  # 缓存目录
        force_download=force_download,  # 是否强制下载
        resume_download=resume_download,  # 是否恢复下载
        proxies=proxies,  # 代理设置
        token=token,  # 访问令牌
        revision=revision,  # 版本号
        local_files_only=local_files_only,  # 仅本地文件
        subfolder=subfolder,  # 子文件夹
        _commit_hash=_commit_hash,  # 提交哈希值
        _raise_exceptions_for_gated_repo=False,  # 是否为受限仓库抛出异常
        _raise_exceptions_for_missing_entries=False,  # 是否为缺失条目抛出异常
        _raise_exceptions_for_connection_errors=False,  # 是否为连接错误抛出异常
    )

# 返回适配器配置文件的文件名
return adapter_cached_filename
def check_peft_version(min_version: str) -> None:
    r"""
    检查 PEFT 的版本是否兼容。

    Args:
        min_version (`str`):
            要检查兼容性的 PEFT 版本。
    """
    # 检查 PEFT 是否可用
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    # 获取当前 PEFT 的版本并与指定的最小版本比较
    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version)

    # 如果当前 PEFT 版本不兼容，则抛出异常
    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )
```