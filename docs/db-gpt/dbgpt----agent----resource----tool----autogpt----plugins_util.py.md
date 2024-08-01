# `.\DB-GPT-src\dbgpt\agent\resource\tool\autogpt\plugins_util.py`

```py
"""Load plugins from a directory or a zip file.

This module provides utility functions to load auto_gpt plugins from a directory or a
zip file.
"""

import datetime  # 导入处理日期和时间的模块
import glob  # 导入用于文件名匹配的模块
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入处理路径的模块
from typing import TYPE_CHECKING, List, Optional  # 导入类型提示相关的功能

if TYPE_CHECKING:
    from auto_gpt_plugin_template import AutoGPTPluginTemplate  # 导入自动生成插件模板类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例


def inspect_zip_for_modules(zip_path: str, debug: bool = False) -> list[str]:
    """Load the AutoGPTPluginTemplate from a zip file.

    Loader zip plugin file. Native support Auto_gpt_plugin

    Args:
    zip_path (str): Path to the zipfile.
    debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
    list[str]: The list of module names found or empty list if none were found.
    """
    import zipfile  # 导入处理 ZIP 文件的模块

    result = []  # 初始化一个空列表，用于存储找到的模块名
    with zipfile.ZipFile(zip_path, "r") as zfile:  # 打开 ZIP 文件进行读取
        for name in zfile.namelist():  # 遍历 ZIP 文件中的文件名列表
            if name.endswith("__init__.py") and not name.startswith("__MACOSX"):  # 判断文件名是否以 '__init__.py' 结尾且不以 '__MACOSX' 开头
                logger.debug(f"Found module '{name}' in the zipfile at: {name}")  # 记录找到的模块名和其位置
                result.append(name)  # 将符合条件的模块名添加到结果列表中
    if len(result) == 0:  # 如果未找到符合条件的模块名
        logger.debug(f"Module '__init__.py' not found in the zipfile @ {zip_path}.")  # 记录未找到 '__init__.py' 模块
    return result  # 返回找到的模块名列表或空列表


def write_dict_to_json_file(data: dict, file_path: str) -> None:
    """Write a dictionary to a JSON file.

    Args:
        data (dict): Dictionary to write.
        file_path (str): Path to the file.
    """
    with open(file_path, "w") as file:  # 打开文件进行写入
        json.dump(data, file, indent=4)  # 将字典数据写入 JSON 文件，格式化缩进为 4


def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create a directory if it does not exist.

    Args:
        directory_path (str): Path to the directory.
    Returns:
        bool: True if the directory was created, else False.
    """
    if not os.path.exists(directory_path):  # 如果目录不存在
        try:
            os.makedirs(directory_path)  # 创建多层目录
            logger.debug(f"Created directory: {directory_path}")  # 记录创建的目录信息
            return True  # 返回创建成功
        except OSError as e:
            logger.warn(f"Error creating directory {directory_path}: {e}")  # 记录创建目录时的错误信息
            return False  # 返回创建失败
    else:
        logger.info(f"Directory {directory_path} already exists")  # 记录目录已存在的信息
        return True  # 返回已存在


def scan_plugin_file(file_path, debug: bool = False) -> List["AutoGPTPluginTemplate"]:
    """Scan a plugin file and load the plugins."""
    from zipimport import zipimporter  # 导入 ZIP 文件导入器

    logger.info(f"__scan_plugin_file:{file_path},{debug}")  # 记录插件文件扫描开始的信息
    loaded_plugins = []  # 初始化一个空列表，用于存储加载的插件对象
    # 如果模块列表通过检查 ZIP 文件中的模块，并且指定了文件路径和调试选项
    if moduleList := inspect_zip_for_modules(str(file_path), debug):
        # 遍历模块列表中的每个模块
        for module in moduleList:
            # 创建 Path 对象表示插件文件的路径
            plugin = Path(file_path)
            # 创建 Path 对象表示当前处理的模块路径，忽略类型检查
            module = Path(module)  # type: ignore
            # 记录调试信息，显示插件文件和当前加载的模块
            logger.debug(f"Plugin: {plugin} Module: {module}")
            # 使用 zipimporter 加载插件文件作为 ZIP 导入器对象
            zipped_package = zipimporter(str(plugin))
            # 从 ZIP 导入器中加载指定模块
            zipped_module = zipped_package.load_module(
                str(module.parent)  # type: ignore
            )
            # 遍历加载的模块中的属性名列表
            for key in dir(zipped_module):
                # 如果属性名以双下划线开头，则跳过
                if key.startswith("__"):
                    continue
                # 获取当前属性对应的模块对象
                a_module = getattr(zipped_module, key)
                # 获取当前模块对象的属性名列表
                a_keys = dir(a_module)
                # 如果属性名列表中包含 '_abc_impl'，且模块名不是 'AutoGPTPluginTemplate'
                # 和 denylist_allowlist_check(a_module.__name__, cfg) 相关的注释已被注释掉
                if (
                    "_abc_impl" in a_keys
                    and a_module.__name__ != "AutoGPTPluginTemplate"
                ):
                    # 将实例化的模块对象添加到已加载插件列表中
                    loaded_plugins.append(a_module())
    # 返回所有已加载的插件对象列表
    return loaded_plugins
def scan_plugins(
    plugins_file_path: str, file_name: str = "", debug: bool = False
) -> List["AutoGPTPluginTemplate"]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        plugins_file_path (str): Path to the plugins directory.
        file_name (str, optional): Specific plugin file name to load. Defaults to "".
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[AutoGPTPluginTemplate]: List of loaded plugins.
    """
    # Initialize an empty list to store loaded plugins
    loaded_plugins = []
    
    # Convert the plugins file path to a Path object
    plugins_path = Path(plugins_file_path)
    
    # Check if a specific file name is provided
    if file_name:
        # If file_name is provided, create a Path object for the plugin file
        plugin_path = Path(plugins_path, file_name)
        # Scan the specified plugin file and add loaded plugins to the list
        loaded_plugins = scan_plugin_file(plugin_path)
    else:
        # If no specific file name is provided, iterate through all .zip files in the plugins directory
        for plugin_path in plugins_path.glob("*.zip"):
            # Scan each plugin file and extend the loaded_plugins list
            loaded_plugins.extend(scan_plugin_file(plugin_path))

    # If any plugins are loaded, log the number of plugins found
    if loaded_plugins:
        logger.info(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    
    # Log details of each loaded plugin
    for plugin in loaded_plugins:
        logger.info(f"{plugin._name}: {plugin._version} - {plugin._description}")
    
    # Return the list of loaded plugins
    return loaded_plugins


def update_from_git(
    download_path: str,
    github_repo: str = "",
    branch_name: str = "main",
    authorization: Optional[str] = None,
):
    """Update plugins from a git repository.

    Args:
        download_path (str): Path to download the plugins.
        github_repo (str, optional): URL of the GitHub repository. Defaults to "".
        branch_name (str, optional): Branch name to download from. Defaults to "main".
        authorization (Optional[str], optional): Authorization token for GitHub API. Defaults to None.

    Returns:
        str: Name of the downloaded plugin repository.
    
    Raises:
        ValueError: If the GitHub repository address is incorrect or if plugin download fails.
    """
    import requests
    
    # Ensure the download directory exists
    os.makedirs(download_path, exist_ok=True)
    
    # Check if a GitHub repository URL is provided
    if github_repo:
        # Validate if the URL is a GitHub repository
        if github_repo.index("github.com") <= 0:
            raise ValueError("Not a correct Github repository address！" + github_repo)
        
        # Remove ".git" suffix from the GitHub repository URL
        github_repo = github_repo.replace(".git", "")
        
        # Construct the URL to download the repository as a zip archive
        url = github_repo + "/archive/refs/heads/" + branch_name + ".zip"
        
        # Extract the name of the repository from the GitHub URL
        plugin_repo_name = github_repo.strip("/").split("/")[-1]
    else:
        # If no GitHub repository URL is provided, use a default URL for a specific repository
        url = "https://github.com/eosphoros-ai/DB-GPT-Plugins/archive/refs/heads/main.zip"
        
        # Assign a default name for the repository
        plugin_repo_name = "DB-GPT-Plugins"
    
    try:
        # Create a persistent HTTP session
        session = requests.Session()
        headers = {}
        
        # Include authorization token in headers if provided
        if authorization and len(authorization) > 0:
            headers = {"Authorization": authorization}
        
        # Send a GET request to download the repository zip archive
        response = session.get(
            url,
            headers=headers,
        )

        # If the download request is successful (status code 200)
        if response.status_code == 200:
            # Convert the download path to a Path object
            plugins_path_path = Path(download_path)
            
            # Clean up any existing files matching the repository name pattern in the download path
            files = glob.glob(os.path.join(plugins_path_path, f"{plugin_repo_name}*"))
            for file in files:
                os.remove(file)
            
            # Generate a unique file name for the downloaded zip archive
            now = datetime.datetime.now()
            time_str = now.strftime("%Y%m%d%H%M%S")
            file_name = (
                f"{plugins_path_path}/{plugin_repo_name}-{branch_name}-{time_str}.zip"
            )
            
            # Write the downloaded content to the specified file
            with open(file_name, "wb") as f:
                f.write(response.content)
            
            # Return the name of the downloaded plugin repository
            return plugin_repo_name
        else:
            # Log an error message if plugin update fails
            logger.error(f"Update plugins failed，response code：{response.status_code}")
            raise ValueError(f"Download plugin failed: {response.status_code}")
    
    # Handle any exceptions that occur during the update process
    except Exception as e:
        logger.error("update plugins from git exception!" + str(e))
        raise ValueError("download plugin exception!", e)
# 从Git仓库中获取文件到本地路径
def __fetch_from_git(local_path, git_url):
    # 导入git模块
    import git

    # 记录日志，指示从Git中获取插件到本地路径
    logger.info("fetch plugins from git to local path:{}", local_path)
    
    # 确保本地路径存在，如果不存在则创建
    os.makedirs(local_path, exist_ok=True)
    
    # 使用本地路径初始化git仓库对象
    repo = git.Repo(local_path)
    
    # 如果本地路径已经是一个git仓库，则执行远程仓库的拉取操作
    if repo.is_repo():
        repo.remotes.origin.pull()
    # 如果本地路径不是git仓库，则进行克隆操作从指定的git_url克隆仓库到本地路径
    else:
        git.Repo.clone_from(git_url, local_path)

    # 如果需要验证仓库头部有效性，可以取消下面的注释
    # if repo.head.is_valid():
    # clone成功，获取插件信息
```