# `.\DB-GPT-src\dbgpt\util\dbgpts\base.py`

```py
import hashlib
import logging
import os
import sys
from pathlib import Path

# 设置日志记录器，使用当前模块的名称
logger = logging.getLogger(__name__)

# 获取绝对根路径的上一级目录
_ABS_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 设置默认的 dbgpts 目录为用户主目录下的 .dbgpts 目录
DEFAULT_DBGPTS_DIR = Path.home() / ".dbgpts"
# 获取 DBGPTS_HOME 环境变量，如果不存在则使用默认的 .dbgpts 目录
DBGPTS_HOME = os.getenv("DBGPTS_HOME", str(DEFAULT_DBGPTS_DIR))
# 设置 DBGPTS_REPO_HOME 环境变量，如果不存在则使用默认的 .dbgpts/repos 目录
DBGPTS_REPO_HOME = os.getenv("DBGPTS_REPO_HOME", str(DEFAULT_DBGPTS_DIR / "repos"))

# 默认的仓库名称到 URL 映射
DEFAULT_REPO_MAP = {
    "eosphoros/dbgpts": "https://github.com/eosphoros-ai/dbgpts.git",
    "fangyinc/dbgpts": "https://github.com/fangyinc/dbgpts.git",
}

# 默认的包名称列表
DEFAULT_PACKAGES = ["agents", "apps", "operators", "workflow", "resources"]
# 默认的包类型列表
DEFAULT_PACKAGE_TYPES = ["agent", "app", "operator", "flow", "resource"]
# 安装元数据文件名
INSTALL_METADATA_FILE = "install_metadata.toml"
# DBGPTS 元数据文件名
DBGPTS_METADATA_FILE = "dbgpts.toml"

# 类型到包名的映射
TYPE_TO_PACKAGE = {
    "agent": "agents",
    "app": "apps",
    "operator": "operators",
    "flow": "workflow",
    "resource": "resources",
}


def _get_env_sig() -> str:
    """获取当前 Python 环境的唯一签名"""
    # 获取 Python 解释器路径，并结合绝对根路径生成环境路径
    py_path = os.path.join(os.path.dirname(sys.executable), "python")
    env_path = f"{_ABS_ROOT_PATH}_{py_path}"
    # 计算路径的 MD5 哈希值作为环境签名
    md5_hash = hashlib.md5()
    md5_hash.update(env_path.encode("utf-8"))
    return md5_hash.hexdigest()


def _print_path(path: str | Path) -> str:
    """打印路径，将用户主目录替换为 '~'"""
    str_path = str(path)
    if str_path.startswith(str(Path.home())):
        str_path = str_path.replace(str(Path.home()), "~")
    return str_path


def get_repo_path(repo: str) -> str:
    """获取仓库的路径

    Args:
        repo (str): 仓库的名称，格式为 "组织名/仓库名"

    Returns:
        str: 仓库的路径
    """
    repo_group, repo_name = repo.split("/")
    return str(Path(DBGPTS_REPO_HOME) / repo_group / repo_name)


# 获取当前环境的签名
ENV_SIG = _get_env_sig()
# 安装目录为 DBGPTS_HOME/packages/ENV_SIG
INSTALL_DIR = Path(DBGPTS_HOME) / "packages" / ENV_SIG

# 确保 DBGPTS_REPO_HOME 目录存在，如果不存在则创建
os.makedirs(DBGPTS_REPO_HOME, exist_ok=True)
# 确保 INSTALL_DIR 目录存在，如果不存在则创建
os.makedirs(INSTALL_DIR, exist_ok=True)
```