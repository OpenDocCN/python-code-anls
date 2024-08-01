# `.\DB-GPT-src\dbgpt\util\dbgpts\repo.py`

```py
# 导入必要的模块和库
import functools  # functools 模块提供了一些高阶函数，如缓存
import os  # os 模块提供了与操作系统交互的功能
import shutil  # shutil 模块提供了高级的文件操作功能
import subprocess  # subprocess 模块允许你在单独的进程中执行新的程序
from pathlib import Path  # pathlib 模块提供了面向对象的文件系统路径操作
from typing import List, Tuple  # typing 模块支持类型提示功能

from rich.table import Table  # 导入 rich 模块中的 Table 类，用于创建表格

from ..console import CliLogger  # 从相对路径中导入 CliLogger 类
from ..i18n_utils import _  # 导入国际化函数 _
from .base import (  # 从当前包的 base 模块导入多个常量
    DBGPTS_METADATA_FILE,
    DBGPTS_REPO_HOME,
    DEFAULT_PACKAGES,
    DEFAULT_REPO_MAP,
    INSTALL_DIR,
    INSTALL_METADATA_FILE,
    _print_path,
)
from .loader import _load_package_from_path  # 从当前包的 loader 模块导入 _load_package_from_path 函数

cl = CliLogger()  # 创建一个 CliLogger 实例对象

_DEFAULT_REPO = "eosphoros/dbgpts"  # 定义默认的仓库路径

@functools.cache
def list_repos() -> List[str]:
    """列出所有的仓库

    Returns:
        List[str]: 仓库列表
    """
    repos = set()  # 使用集合存储仓库名，确保唯一性
    for repo in os.listdir(DBGPTS_REPO_HOME):  # 遍历仓库主目录下的所有文件和文件夹
        full_path = os.path.join(DBGPTS_REPO_HOME, repo)  # 获取完整的仓库路径
        if os.path.isdir(full_path):  # 如果是目录
            for sub_repo in os.listdir(full_path):  # 遍历仓库下的子目录
                if os.path.isdir(os.path.join(full_path, sub_repo)):  # 如果子目录也是目录
                    repos.add(f"{repo}/{sub_repo}")  # 将仓库名添加到集合中，格式为 "主目录/子目录"
    repos.add(_DEFAULT_REPO)  # 添加默认仓库到集合中
    return sorted(list(repos))  # 返回排序后的仓库列表

def _get_repo_path(repo: str) -> Path:
    """获取仓库路径

    Args:
        repo (str): 仓库名，格式为 "主目录/子目录"

    Returns:
        Path: 仓库的完整路径
    """
    repo_arr = repo.split("/")  # 根据 "/" 分割仓库名
    if len(repo_arr) != 2:  # 如果分割后的仓库名不是两部分
        cl.error(
            f"Invalid repo name '{repo}', repo name must split by '/', "
            f"eg.(eosphoros/dbgpts).",
            exit_code=1,
        )  # 输出错误信息并退出程序
    return Path(DBGPTS_REPO_HOME) / repo_arr[0] / repo_arr[1]  # 返回拼接后的仓库路径对象

def _list_repos_details() -> List[Tuple[str, str]]:
    """列出所有仓库的详细信息

    Returns:
        List[Tuple[str, str]]: 包含仓库名和完整路径的元组列表
    """
    repos = list_repos()  # 获取所有仓库列表
    results = []  # 初始化空列表用于存储结果
    for repo in repos:  # 遍历每个仓库
        repo_arr = repo.split("/")  # 根据 "/" 分割仓库名
        repo_group, repo_name = repo_arr  # 获取仓库主目录和子目录名
        full_path = os.path.join(DBGPTS_REPO_HOME, repo_group, repo_name)  # 拼接完整的仓库路径
        results.append((repo, full_path))  # 将仓库名和路径组成的元组添加到结果列表中
    return results  # 返回包含仓库名和完整路径的元组列表

def _print_repos():
    """打印所有仓库信息"""
    repos = _list_repos_details()  # 获取所有仓库的详细信息
    repos.sort(key=lambda x: (x[0], x[1]))  # 按仓库名和路径排序
    table = Table(title=_("Repos"))  # 创建一个标题为 "Repos" 的表格对象
    table.add_column(_("Repository"), justify="right", style="cyan", no_wrap=True)  # 添加名为 "Repository" 的列，右对齐，青色字体
    table.add_column(_("Path"), justify="right", style="green")  # 添加名为 "Path" 的列，右对齐，绿色字体
    for repo, full_path in repos:  # 遍历所有仓库的详细信息
        full_path = _print_path(full_path)  # 调用 _print_path 函数打印路径
        table.add_row(repo, full_path)  # 将仓库名和路径添加为表格的一行
    cl.print(table)  # 使用 CliLogger 实例打印表格

def _install_default_repos_if_no_repos():
    """如果没有仓库存在，则安装默认的仓库"""
    has_repos = False  # 初始化标志变量，表示是否存在仓库
    for repo, full_path in _list_repos_details():  # 遍历所有仓库的详细信息
        if os.path.exists(full_path):  # 如果仓库路径存在
            has_repos = True  # 更新标志变量为 True
            break  # 中断循环
    if not has_repos:  # 如果不存在仓库
        repo_url = DEFAULT_REPO_MAP[_DEFAULT_REPO]  # 获取默认仓库的 URL
        cl.info(
            f"No repos found, installing default repos {_DEFAULT_REPO} from {repo_url}"
        )  # 输出信息，指示安装默认仓库
        add_repo(_DEFAULT_REPO, repo_url)  # 调用 add_repo 函数安装默认仓库

def add_repo(repo: str, repo_url: str, branch: str | None = None):
    """添加一个新仓库

    Args:
        repo (str): 仓库名
        repo_url (str): 仓库的 URL
        branch (str): 仓库的分支名
    """
    exist_repos = list_repos()  # 获取已存在的仓库列表
    if repo in exist_repos and repo_url not in DEFAULT_REPO_MAP.values():  # 如果仓库名已存在且 URL 不在默认仓库映射中
        cl.error(f"The repo '{repo}' already exists.", exit_code=1)  # 输出错误信息并退出程序
    # 将 repo 字符串按 '/' 分割成列表
    repo_arr = repo.split("/")

    # 如果 repo_arr 长度不为 2，则说明 repo 格式不正确，输出错误信息并退出程序
    if len(repo_arr) != 2:
        cl.error(
            f"Invalid repo name '{repo}', repo name must split by '/', "
            "eg.(eosphoros/dbgpts).",
            exit_code=1,
        )

    # 取 repo_arr 列表中的第二个元素作为 repo_name
    repo_name = repo_arr[1]

    # 根据 DBGPTS_REPO_HOME 和 repo_arr[0] 拼接出 repo_group_dir，用于存放同一组 repo 的目录
    repo_group_dir = os.path.join(DBGPTS_REPO_HOME, repo_arr[0])

    # 如果 repo_group_dir 目录不存在，则创建它，存在则不做任何操作
    os.makedirs(repo_group_dir, exist_ok=True)

    # 如果 repo_url 以 "http" 或 "git" 开头，表示是一个 URL，调用 clone_repo 函数克隆 repo
    if repo_url.startswith("http") or repo_url.startswith("git"):
        clone_repo(repo, repo_group_dir, repo_name, repo_url, branch)
    
    # 否则，如果 repo_url 是一个存在的目录路径，则创建软链接到 repo_group_dir 中
    elif os.path.isdir(repo_url):
        os.symlink(repo_url, os.path.join(repo_group_dir, repo_name))
def remove_repo(repo: str):
    """Remove the specified repo

    Args:
        repo (str): The name of the repo
    """
    # 获取指定 repo 的路径
    repo_path = _get_repo_path(repo)
    
    # 如果 repo 路径不存在，报错并退出
    if not os.path.exists(repo_path):
        cl.error(f"The repo '{repo}' does not exist.", exit_code=1)
    
    # 如果 repo 路径是一个符号链接，直接删除链接
    if os.path.islink(repo_path):
        os.unlink(repo_path)
    else:
        # 否则递归删除整个目录
        shutil.rmtree(repo_path)
    
    # 显示删除成功的消息
    cl.info(f"Repo '{repo}' removed successfully.")


def clone_repo(
    repo: str,
    repo_group_dir: str,
    repo_name: str,
    repo_url: str,
    branch: str | None = None,
):
    """Clone the specified repo

    Args:
        repo (str): The name of the repo
        repo_group_dir (str): The directory of the repo group
        repo_name (str): The name of the repo
        repo_url (str): The URL of the repo
        branch (str): The branch of the repo
    """
    # 切换到 repo 所在的目录
    os.chdir(repo_group_dir)
    
    # 构建 git clone 命令
    clone_command = ["git", "clone", repo_url, repo_name]

    # 如果指定了 branch，则将其添加到 clone 命令中
    if branch:
        clone_command += ["-b", branch]

    # 执行 git clone 命令
    subprocess.run(clone_command, check=True)
    
    # 显示克隆成功的消息，包含 branch 信息
    if branch:
        cl.info(
            f"Repo '{repo}' cloned from {repo_url} with branch '{branch}' successfully."
        )
    else:
        # 显示克隆成功的消息
        cl.info(f"Repo '{repo}' cloned from {repo_url} successfully.")


def update_repo(repo: str):
    """Update the specified repo

    Args:
        repo (str): The name of the repo
    """
    # 显示更新 repo 的信息
    cl.info(f"Updating repo '{repo}'...")
    
    # 获取 repo 的完整路径
    repo_path = os.path.join(DBGPTS_REPO_HOME, repo)
    
    # 如果 repo 路径不存在
    if not os.path.exists(repo_path):
        # 如果 repo 存在于默认 repo 映射中，则添加并重新检查路径
        if repo in DEFAULT_REPO_MAP:
            add_repo(repo, DEFAULT_REPO_MAP[repo])
            if not os.path.exists(repo_path):
                cl.error(f"The repo '{repo}' does not exist.", exit_code=1)
        else:
            # 否则报错并退出
            cl.error(f"The repo '{repo}' does not exist.", exit_code=1)
    
    # 切换到 repo 的目录
    os.chdir(repo_path)
    
    # 如果目录中没有 .git 文件夹，显示提示信息并返回
    if not os.path.exists(".git"):
        cl.info(f"Repo '{repo}' is not a git repository.")
        return
    
    # 执行 git pull 命令来更新 repo
    cl.info(f"Updating repo '{repo}'...")
    subprocess.run(["git", "pull"], check=False)


def install(
    name: str,
    repo: str | None = None,
    with_update: bool = True,
):
    """Install the specified dbgpt from the specified repo

    Args:
        name (str): The name of the dbgpt
        repo (str): The name of the repo
        with_update (bool): Whether to update the repo before installing
    """
    # 检查并获取 dbgpt 的信息，包括 repo 信息
    repo_info = check_with_retry(name, repo, with_update=with_update, is_first=True)
    
    # 如果未找到 dbgpt 信息，报错并退出
    if not repo_info:
        cl.error(f"The specified dbgpt '{name}' does not exist.", exit_code=1)
    
    # 解包 repo_info 元组，获取 repo 名称和 dbgpt 路径
    repo, dbgpt_path = repo_info
    
    # 复制并安装 dbgpt
    _copy_and_install(repo, name, dbgpt_path)


def uninstall(name: str):
    """Uninstall the specified dbgpt

    Args:
        name (str): The name of the dbgpt
    """
    # 获取 dbgpt 的安装路径
    install_path = INSTALL_DIR / name
    
    # 如果安装路径不存在，报错并退出
    if not install_path.exists():
        cl.error(f"The dbgpt '{name}' has not been installed yet.", exit_code=1)
    
    # 切换到安装路径目录
    os.chdir(install_path)
    # 使用 subprocess 模块运行命令来卸载指定名称的包，"-y" 参数表示自动确认
    subprocess.run(["pip", "uninstall", name, "-y"], check=True)
    # 使用 shutil 模块的 rmtree 函数删除指定路径下的所有文件和文件夹
    shutil.rmtree(install_path)
    # 使用 cl 对象的 info 方法输出调试信息，提示正在卸载指定名称的 dbgpt 包
    cl.info(f"Uninstalling dbgpt '{name}'...")
# 复制并安装指定 dbgpt 包到指定的安装路径
def _copy_and_install(repo: str, name: str, package_path: Path):
    # 如果指定的包路径不存在，则报错
    if not package_path.exists():
        cl.error(
            f"The specified dbgpt '{name}' does not exist in the {repo} tap.",
            exit_code=1,
        )
    # 设置安装路径
    install_path = INSTALL_DIR / name
    # 如果安装路径已经存在，则报错
    if install_path.exists():
        cl.error(
            f"The dbgpt '{name}' has already been installed"
            f"({_print_path(install_path)}).",
            exit_code=1,
        )
    try:
        # 复制整个包到安装路径
        shutil.copytree(package_path, install_path)
        # 提示安装 dbgpt 包
        cl.info(f"Installing dbgpts '{name}' from {repo}...")
        # 切换工作目录到安装路径
        os.chdir(install_path)
        # 使用 subprocess 运行 poetry 构建包
        subprocess.run(["poetry", "build"], check=True)
        # 查找安装路径下的 wheel 文件
        wheel_files = list(install_path.glob("dist/*.whl"))
        # 如果找不到 wheel 文件，则报错
        if not wheel_files:
            cl.error("No wheel file found after building the package.", exit_code=1)
        # 获取第一个 wheel 文件
        wheel_file = wheel_files[0]
        # 提示安装 wheel 文件
        cl.info(f"Installing dbgpts '{name}' wheel file {_print_path(wheel_file)}...")
        # 使用 pip 安装 wheel 文件
        subprocess.run(["pip", "install", str(wheel_file)], check=True)
        # 写入安装元数据
        _write_install_metadata(name, repo, install_path)
        # 提示安装成功
        cl.success(f"Installed dbgpts at {_print_path(install_path)}.")
        cl.success(f"dbgpts '{name}' installed successfully.")
    except Exception as e:
        # 如果安装路径存在，则删除安装路径
        if install_path.exists():
            shutil.rmtree(install_path)
        # 抛出异常
        raise e


# 写入安装的元数据到指定的路径
def _write_install_metadata(name: str, repo: str, install_path: Path):
    import tomlkit

    # 定义安装的元数据
    install_metadata = {
        "name": name,
        "repo": repo,
    }
    # 使用 tomlkit 将元数据写入文件
    with open(install_path / INSTALL_METADATA_FILE, "w") as f:
        tomlkit.dump(install_metadata, f)


# 检查指定的 dbgpt 包，可选择重试
def check_with_retry(
    name: str,
    spec_repo: str | None = None,
    with_update: bool = False,
    is_first: bool = False,
) -> Tuple[str, Path] | None:
    """Check the specified dbgpt with retry.

    Args:
        name (str): The name of the dbgpt
        spec_repo (str): The name of the repo
        with_update (bool): Whether to update the repo before installing
        is_first (bool): Whether it's the first time to check the dbgpt
    Returns:
        Tuple[str, Path] | None: The repo and the path of the dbgpt
    """
    # 获取所有的仓库详情
    repos = _list_repos_details()
    # 如果指定了特定的仓库名，则过滤出该仓库
    if spec_repo:
        repos = list(filter(lambda x: x[0] == spec_repo, repos))
        # 如果过滤后的仓库列表为空，则报错
        if not repos:
            cl.error(f"The specified repo '{spec_repo}' does not exist.", exit_code=1)
    # 如果是第一次检查并且需要更新，则更新所有仓库
    if is_first and with_update:
        for repo in repos:
            update_repo(repo[0])
    # 遍历所有仓库
    for repo in repos:
        # 获取仓库的路径
        repo_path = Path(repo[1])
        # 遍历默认的包名列表
        for package in DEFAULT_PACKAGES:
            # 组合 dbgpt 的路径和元数据文件的路径
            dbgpt_path = repo_path / package / name
            dbgpt_metadata_path = dbgpt_path / DBGPTS_METADATA_FILE
            # 如果 dbgpt 存在且是一个目录，并且包含元数据文件，则返回仓库名和 dbgpt 的路径
            if (
                dbgpt_path.exists()
                and dbgpt_path.is_dir()
                and dbgpt_metadata_path.exists()
            ):
                return repo[0], dbgpt_path
    # 如果 is_first 参数为 True，则进行重试检查操作
    if is_first:
        # 调用 check_with_retry 函数，传入指定的参数和选项
        return check_with_retry(
            name, spec_repo, with_update=with_update, is_first=False
        )
    # 如果 is_first 参数不为 True，则返回 None
    return None
def list_repo_apps(repo: str | None = None, with_update: bool = True):
    """List all installed dbgpts"""
    # 获取所有已安装 dbgpt 的详细信息
    repos = _list_repos_details()
    
    # 如果指定了 repo 参数，则筛选出指定的仓库信息
    if repo:
        repos = list(filter(lambda x: x[0] == repo, repos))
        # 如果筛选结果为空列表，则报错并退出程序
        if not repos:
            cl.error(f"The specified repo '{repo}' does not exist.", exit_code=1)
    
    # 如果 with_update 参数为 True，则更新每个仓库的 dbgpt
    if with_update:
        for repo in repos:
            update_repo(repo[0])
    
    # 创建一个标题为 "dbgpts In All Repos" 的表格对象
    table = Table(title=_("dbgpts In All Repos"))

    # 添加表格列：Repository，Type，Name
    table.add_column(_("Repository"), justify="right", style="cyan", no_wrap=True)
    table.add_column(_("Type"), style="magenta")
    table.add_column(_("Name"), justify="right", style="green")

    # 初始化数据列表
    data = []
    
    # 遍历每个仓库的信息
    for repo in repos:
        repo_path = Path(repo[1])
        
        # 遍历默认包的列表
        for package in DEFAULT_PACKAGES:
            dbgpt_path = repo_path / package
            
            # 遍历调试点路径下的每个应用
            for app in os.listdir(dbgpt_path):
                dbgpt_metadata_path = dbgpt_path / app / DBGPTS_METADATA_FILE
                
                # 如果调试点路径存在且是一个目录，并且调试点元数据文件存在，则将其添加到数据列表中
                if (
                    dbgpt_path.exists()
                    and dbgpt_path.is_dir()
                    and dbgpt_metadata_path.exists()
                ):
                    data.append((repo[0], package, app))
    
    # 根据仓库名称、包名称和应用名称对数据列表进行排序
    data.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # 将排序后的数据添加到表格中
    for repo, package, app in data:
        table.add_row(repo, package, app)
    
    # 打印表格
    cl.print(table)


def list_installed_apps():
    """List all installed dbgpts"""
    # 从安装目录加载所有的包信息
    packages = _load_package_from_path(INSTALL_DIR)
    
    # 创建一个标题为 "Installed dbgpts" 的表格对象
    table = Table(title=_("Installed dbgpts"))

    # 添加表格列：Name，Type，Repository，Path
    table.add_column(_("Name"), justify="right", style="cyan", no_wrap=True)
    table.add_column(_("Type"), style="blue")
    table.add_column(_("Repository"), style="magenta")
    table.add_column(_("Path"), justify="right", style="green")

    # 根据包名称、包类型和仓库名称对包信息列表进行排序
    packages.sort(key=lambda x: (x.package, x.package_type, x.repo))
    
    # 遍历每个包信息，将其路径格式化并添加到表格中
    for package in packages:
        str_path = package.root
        str_path = _print_path(str_path)
        table.add_row(package.package, package.package_type, package.repo, str_path)
    
    # 打印表格
    cl.print(table)
```