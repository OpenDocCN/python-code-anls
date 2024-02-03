# `.\AutoGPT\autogpts\autogpt\scripts\install_plugin_deps.py`

```py
import logging
import os
import subprocess
import sys
import zipfile
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)

# 定义一个函数，用于安装插件的依赖项
def install_plugin_dependencies():
    """
    Installs dependencies for all plugins in the plugins dir.

    Args:
        None

    Returns:
        None
    """
    # 获取插件目录路径，默认为"plugins"
    plugins_dir = Path(os.getenv("PLUGINS_DIR", "plugins"))

    logger.debug("Checking for dependencies in zipped plugins...")

    # 安装基于 ZIP 文件的插件
    for plugin_archive in plugins_dir.glob("*.zip"):
        logger.debug(f"Checking for requirements in '{plugin_archive}'...")
        # 打开 ZIP 文件
        with zipfile.ZipFile(str(plugin_archive), "r") as zfile:
            # 如果 ZIP 文件为空，则跳过
            if not zfile.namelist():
                continue

            # 假设列表中的第一个条目将是最低公共目录中的条目
            first_entry = zfile.namelist()[0]
            basedir = first_entry.rsplit("/", 1)[0] if "/" in first_entry else ""
            logger.debug(f"Looking for requirements.txt in '{basedir}'")

            basereqs = os.path.join(basedir, "requirements.txt")
            try:
                # 解压缩 requirements.txt 文件到插件目录
                extracted = zfile.extract(basereqs, path=plugins_dir)
            except KeyError as e:
                logger.debug(e.args[0])
                continue

            logger.debug(f"Installing dependencies from '{basereqs}'...")
            # 使用 pip 安装依赖项
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", extracted]
            )
            # 删除已解压的文件
            os.remove(extracted)
            # 删除基础目录
            os.rmdir(os.path.join(plugins_dir, basedir))

    logger.debug("Checking for dependencies in other plugin folders...")

    # 安装基于目录的插件
    for requirements_file in glob(f"{plugins_dir}/*/requirements.txt"):
        logger.debug(f"Installing dependencies from '{requirements_file}'...")
        # 使用 pip 安装依赖项
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            stdout=subprocess.DEVNULL,
        )
    # 记录调试信息，表示已完成安装插件的依赖项
    logger.debug("Finished installing plugin dependencies")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 调用函数安装插件依赖
    install_plugin_dependencies()
```