# `.\DB-GPT-src\dbgpt\cli\cli_scripts.py`

```py
# 导入必要的模块
import copy  # 导入 copy 模块，用于深拷贝对象
import logging  # 导入 logging 模块，用于记录日志

import click  # 导入 click 模块，用于创建命令行界面

# 配置 logging 模块，设置日志级别为 WARNING，编码为 UTF-8，日志格式为指定格式
logging.basicConfig(
    level=logging.WARNING,
    encoding="utf-8",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("dbgpt_cli")  # 创建名为 "dbgpt_cli" 的 logger 对象


@click.group()  # 创建命令行组
@click.option(
    "--log-level",  # 命令行选项 --log-level，用于设置日志级别
    required=False,
    type=str,
    default="warn",
    help="Log level",  # 帮助信息，指定日志级别
)
@click.version_option()  # 添加版本选项
def cli(log_level: str):
    logger.setLevel(logging.getLevelName(log_level.upper()))  # 设置 logger 对象的日志级别


# 函数用于向命令行组添加新的命令别名
def add_command_alias(command, name: str, hidden: bool = False, parent_group=None):
    if not parent_group:
        parent_group = cli  # 如果没有指定父命令行组，默认使用全局命令行组 cli
    new_command = copy.deepcopy(command)  # 深拷贝命令对象
    new_command.hidden = hidden  # 设置命令是否隐藏
    parent_group.add_command(new_command, name=name)  # 添加命令别名到父命令行组


@click.group()  # 定义一个命令行组
def start():
    """Start specific server."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def stop():
    """Start specific server."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def install():
    """Install dependencies, plugins, etc."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def db():
    """Manage your metadata database and your datasources."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def new():
    """New a template."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def app():
    """Manage your apps(dbgpts)."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def repo():
    """The repository to install the dbgpts from."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def run():
    """Run your dbgpts."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


@click.group()  # 定义一个命令行组
def net():
    """Net tools."""  # 命令行组的简要说明
    pass  # 空操作，因为命令行组本身不执行具体操作


stop_all_func_list = []  # 创建一个空列表，用于存储所有停止函数的引用


@click.command(name="all")  # 定义一个命令，命名为 "all"
def stop_all():
    """Stop all servers"""  # 命令的简要说明
    for stop_func in stop_all_func_list:  # 遍历所有的停止函数引用
        stop_func()  # 调用每个停止函数


cli.add_command(start)  # 将 start 命令行组添加到全局命令行组 cli
cli.add_command(stop)  # 将 stop 命令行组添加到全局命令行组 cli
cli.add_command(install)  # 将 install 命令行组添加到全局命令行组 cli
cli.add_command(db)  # 将 db 命令行组添加到全局命令行组 cli
cli.add_command(new)  # 将 new 命令行组添加到全局命令行组 cli
cli.add_command(app)  # 将 app 命令行组添加到全局命令行组 cli
cli.add_command(repo)  # 将 repo 命令行组添加到全局命令行组 cli
cli.add_command(run)  # 将 run 命令行组添加到全局命令行组 cli
cli.add_command(net)  # 将 net 命令行组添加到全局命令行组 cli

add_command_alias(stop_all, name="all", parent_group=stop)  # 向 stop 命令行组添加 "all" 命令别名

try:
    from dbgpt.model.cli import (  # 尝试导入 dbgpt.model.cli 模块的命令行接口
        _stop_all_model_server,  # 导入停止所有模型服务器的函数
        model_cli_group,  # 导入模型命令行组
        start_apiserver,  # 导入启动 API 服务器的函数
        start_model_controller,  # 导入启动模型控制器的函数
        start_model_worker,  # 导入启动模型工作器的函数
        stop_apiserver,  # 导入停止 API 服务器的函数
        stop_model_controller,  # 导入停止模型控制器的函数
        stop_model_worker,  # 导入停止模型工作器的函数
    )

    add_command_alias(model_cli_group, name="model", parent_group=cli)  # 向全局命令行组 cli 添加 "model" 命令别名
    add_command_alias(start_model_controller, name="controller", parent_group=start)  # 向 start 命令行组添加 "controller" 命令别名
    add_command_alias(start_model_worker, name="worker", parent_group=start)  # 向 start 命令行组添加 "worker" 命令别名
    add_command_alias(start_apiserver, name="apiserver", parent_group=start)  # 向 start 命令行组添加 "apiserver" 命令别名

    add_command_alias(stop_model_controller, name="controller", parent_group=stop)  # 向 stop 命令行组添加 "controller" 命令别名
    add_command_alias(stop_model_worker, name="worker", parent_group=stop)  # 向 stop 命令行组添加 "worker" 命令别名
    add_command_alias(stop_apiserver, name="apiserver", parent_group=stop)  # 向 stop 命令行组添加 "apiserver" 命令别名
    stop_all_func_list.append(_stop_all_model_server)  # 将停止所有模型服务器的函数引用添加到列表中

except ImportError as e:
    logging.warning(f"Integrating dbgpt model command line tool failed: {e}")  # 如果导入失败，记录警告日志

try:
    from dbgpt.app._cli import (  # 尝试导入 dbgpt.app._cli 模块的命令行接口
        _stop_all_dbgpt_server,  # 导入停止所有 dbgpt 服务器的函数
        migration,  # 导入迁移函数
    # 添加一个命令别名，将 start_webserver 添加到名为 "webserver" 的命令组中
    add_command_alias(start_webserver, name="webserver", parent_group=start)
    # 添加一个命令别名，将 stop_webserver 添加到名为 "webserver" 的命令组中
    add_command_alias(stop_webserver, name="webserver", parent_group=stop)
    # 添加迁移命令的别名，将 migration 命令添加到名为 "migration" 的数据库命令组中
    add_command_alias(migration, name="migration", parent_group=db)
    # 将 _stop_all_dbgpt_server 函数添加到 stop_all_func_list 列表中，用于后续调用
    stop_all_func_list.append(_stop_all_dbgpt_server)
# 尝试导入 dbgpt webserver 命令行工具的命令组
try:
    # 导入 knowledge_cli_group 命令组
    from dbgpt.app.knowledge._cli.knowledge_cli import knowledge_cli_group

    # 将 knowledge_cli_group 命令组添加为 'knowledge' 别名，加入到父命令组 cli 中
    add_command_alias(knowledge_cli_group, name="knowledge", parent_group=cli)
# 如果导入失败，则捕获 ImportError 异常
except ImportError as e:
    # 记录警告日志，显示导入失败的具体错误信息
    logging.warning(f"Integrating dbgpt knowledge command line tool failed: {e}")

# 类似地，尝试导入 dbgpt trace 命令行工具的命令组，并添加为 'trace' 别名到父命令组 cli 中
try:
    from dbgpt.util.tracer.tracer_cli import trace_cli_group

    add_command_alias(trace_cli_group, name="trace", parent_group=cli)
except ImportError as e:
    logging.warning(f"Integrating dbgpt trace command line tool failed: {e}")

# 尝试导入 dbgpt serve 命令行工具的 serve 函数，并添加为 'serve' 别名到父命令组 new 中
try:
    from dbgpt.serve.utils.cli import serve

    add_command_alias(serve, name="serve", parent_group=new)
except ImportError as e:
    logging.warning(f"Integrating dbgpt serve command line tool failed: {e}")

# 尝试导入 dbgpt dbgpts 命令行工具的各个函数，并分别添加为相应的别名到不同的父命令组中
try:
    from dbgpt.util.dbgpts.cli import add_repo
    from dbgpt.util.dbgpts.cli import install as app_install
    from dbgpt.util.dbgpts.cli import list_all_apps as app_list_remote
    from dbgpt.util.dbgpts.cli import (
        list_installed_apps,
        list_repos,
        new_dbgpts,
        remove_repo,
    )
    from dbgpt.util.dbgpts.cli import uninstall as app_uninstall
    from dbgpt.util.dbgpts.cli import update_repo

    # 添加各个命令的别名到相应的父命令组中
    add_command_alias(list_repos, name="list", parent_group=repo)
    add_command_alias(add_repo, name="add", parent_group=repo)
    add_command_alias(remove_repo, name="remove", parent_group=repo)
    add_command_alias(update_repo, name="update", parent_group=repo)
    add_command_alias(app_install, name="install", parent_group=app)
    add_command_alias(app_uninstall, name="uninstall", parent_group=app)
    add_command_alias(app_list_remote, name="list-remote", parent_group=app)
    add_command_alias(list_installed_apps, name="list", parent_group=app)
    add_command_alias(new_dbgpts, name="app", parent_group=new)
except ImportError as e:
    logging.warning(f"Integrating dbgpt dbgpts command line tool failed: {e}")

# 尝试导入 dbgpt client 命令行工具的 flow 函数，并添加为 'flow' 别名到父命令组 run 中
try:
    from dbgpt.client._cli import flow as run_flow

    add_command_alias(run_flow, name="flow", parent_group=run)
except ImportError as e:
    logging.warning(f"Integrating dbgpt client command line tool failed: {e}")

# 尝试导入 dbgpt net 命令行工具的 start_forward 函数，并添加为 'forward' 别名到父命令组 net 中
try:
    from dbgpt.util.network._cli import start_forward

    add_command_alias(start_forward, name="forward", parent_group=net)
except ImportError as e:
    logging.warning(f"Integrating dbgpt net command line tool failed: {e}")

# 定义主函数入口，执行父命令组 cli 的操作并返回结果
def main():
    return cli()

# 如果当前脚本作为主程序运行，则调用主函数入口
if __name__ == "__main__":
    main()
```