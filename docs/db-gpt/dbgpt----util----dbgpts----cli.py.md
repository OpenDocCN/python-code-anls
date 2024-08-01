# `.\DB-GPT-src\dbgpt\util\dbgpts\cli.py`

```py
    type=str,
    default=None,
    required=False,
    help="The branch to install from",
)
@click.option(
    "-r",
    "--repo",
    type=str,
    default=None,
    required=False,
    help="The repository to install the dbgpts from",
)
@click.option(
    "-U",
    "--update",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Whether to update the repo",
)
@functools.wraps(func)
def wrapper(*args, **kwargs):
    return func(*args, **kwargs)



@click.command(name="install")
@add_add_common_options
@click.argument("names", type=str, nargs=-1)
def install(repo: str | None, update: bool, names: list[str]):
    """Install your dbgpts(operators,agents,workflows or apps)"""
    from .repo import _install_default_repos_if_no_repos, install

    # 检查 Poetry 是否已安装
    check_poetry_installed()
    # 如果没有默认的仓库，则安装默认仓库
    _install_default_repos_if_no_repos()
    # 遍历安装指定的 dbgpts
    for name in names:
        install(name, repo, with_update=update)



@click.command(name="uninstall")
@click.argument("names", type=str, nargs=-1)
def uninstall(names: list[str]):
    """Uninstall your dbgpts(operators,agents,workflows or apps)"""
    from .repo import uninstall

    # 遍历卸载指定的 dbgpts
    for name in names:
        uninstall(name)



@click.command(name="list-remote")
@add_add_common_options
def list_all_apps(
    repo: str | None,
    update: bool,
):
    """List all available dbgpts"""
    from .repo import _install_default_repos_if_no_repos, list_repo_apps

    # 如果没有默认的仓库，则安装默认仓库
    _install_default_repos_if_no_repos()
    # 列出仓库中的所有 dbgpts
    list_repo_apps(repo, with_update=update)



@click.command(name="list")
def list_installed_apps():
    """List all installed dbgpts"""
    from .repo import list_installed_apps

    # 列出所有已安装的 dbgpts
    list_installed_apps()



@click.command(name="list")
def list_repos():
    """List all repos"""
    from .repo import _print_repos

    # 列出所有仓库
    _print_repos()



@click.command(name="add")
@add_tap_options
@click.option(
    "-b",
    "--branch",
    type=str,
    default=None,
    required=False,
    help="The branch to install from",
)
    type=str,  # 参数类型为字符串
    default=None,  # 默认值为None
    required=False,  # 参数非必需
    help="The branch of the repository(Just for git repo)",  # 参数帮助信息，用于说明该参数是用来指定仓库分支的（仅适用于Git仓库）
@click.option(
    "--url",
    type=str,
    required=True,
    help="The URL of the repo",
)
@click.option(
    "--url",
    type=str,
    required=True,
    help="The URL of the repo",
)
@click.option(
    "-r",
    "--repo",
    type=str,
    default=None,
    required=False,
    help="The repository to update(Default: all repos)",
)
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="The name you want to give to the dbgpt",
)
@click.option(
    "-l",
    "--label",
    type=str,
    default=None,
    required=False,
    help="The label of the dbgpt",
)
@click.option(
    "-d",
    "--description",
    type=str,
    default=None,
    required=False,
    help="The description of the dbgpt",
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(DEFAULT_PACKAGE_TYPES),
    default="flow",
    required=False,
    help="The type of the dbgpt",
)
@click.option(
    "--definition_type",
    type=click.Choice(["json", "python"]),
    default="json",
    required=False,
    help="The definition type of the dbgpt",
)
@click.option(
    "-C",
    "--directory",
    type=str,
    default=None,
    required=False,
    help="The working directory of the dbgpt(defaults to the current directory).",
)
def new_dbgpts(
    name: str,
    label: str | None,
    description: str | None,
    type: str,
    definition_type: str,
    directory: str | None,
):
    """New a dbgpts module structure"""
    如果标签未提供：
        默认使用名称设置标签，将 "-" 和 "_" 替换为空格，并转换为首字母大写的标题形式
        label = click.prompt(
            "Please input the label of the dbgpt", default=default_label
        )
    如果描述未提供：
        使用 click 提示用户输入 dbgpt 的描述信息
        description = click.prompt(
            "Please input the description of the dbgpt", default=""
        )
    如果工作目录未提供：
        默认将工作目录设置为当前工作目录的绝对路径
        directory = click.prompt(
            "Please input the working directory of the dbgpt",
            default=str(Path.cwd()),
            type=click.Path(exists=True, file_okay=False, dir_okay=True),
        )

    检查 Poetry 是否已安装
    导入 create_template 函数用于创建 dbgpt 模板
    create_template(name, label, description, type, definition_type, directory)
```