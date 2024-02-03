# `.\AutoGPT\benchmark\agbenchmark\utils\dependencies\__init__.py`

```py
"""
A module that provides the pytest hooks for this plugin.

The logic itself is in main.py.
"""

# 导入警告模块
import warnings
# 导入类型提示模块
from typing import Any, Callable, Optional
# 导入 pytest 模块
import pytest
# 导入 pytest 的配置参数解析模块
from _pytest.config.argparsing import OptionGroup, Parser
# 导入 pytest 的节点模块
from _pytest.nodes import Item
# 导入自定义的 DependencyManager 类
from .main import DependencyManager

# 创建一个空列表 managers 用于存储 DependencyManager 对象
managers: list[DependencyManager] = []

# 定义一个字典，包含处理依赖问题的动作
DEPENDENCY_PROBLEM_ACTIONS: dict[str, Callable[[str], None] | None] = {
    "run": None,
    "skip": lambda m: pytest.skip(m),
    "fail": lambda m: pytest.fail(m, False),
    "warning": lambda m: warnings.warn(m),
}

# 定义一个函数，用于添加配置文件和命令行选项
def _add_ini_and_option(
    parser: Any,
    group: OptionGroup,
    name: str,
    help: str,
    default: str | bool | int,
    **kwargs: Any,
) -> None:
    """Add an option to both the ini file as well as the command line flags, with the latter overriding the former."""
    # 向配置文件中添加选项
    parser.addini(
        name,
        help + " This overrides the similarly named option from the config.",
        default=default,
    )
    # 向命令行选项中添加选项
    group.addoption(f'--{name.replace("_", "-")}', help=help, default=None, **kwargs)

# 定义一个函数，用于获取配置文件或命令行选项的值
def _get_ini_or_option(
    config: Any, name: str, choices: Optional[list[str]]
) -> str | None:
    """Get an option from either the ini file or the command line flags, the latter taking precedence."""
    # 从配置文件中获取选项的值
    value = config.getini(name)
    # 如果值不在选项列表中，则抛出异常
    if value is not None and choices is not None and value not in choices:
        raise ValueError(
            f'Invalid ini value for {name}, choose from {", ".join(choices)}'
        )
    # 返回命令行选项的值，如果不存在则返回配置文件中的值
    return config.getoption(name) or value

# 定义 pytest_addoption 函数，用于添加命令行选项
def pytest_addoption(parser: Parser) -> None:
    # 获取当前所有选项字符串
    current_options = []
    for action in parser._anonymous.options:
        current_options += action._short_opts + action._long_opts

    for group in parser._groups:
        for action in group.options:
            current_options += action._short_opts + action._long_opts

    # 获取 depends 组
    group = parser.getgroup("depends")

    # 添加一个标志，用于列出所有名称及其解析的测试
    # 如果当前选项中没有"--list-dependency-names"，则添加一个选项到测试组中
    if "--list-dependency-names" not in current_options:
        group.addoption(
            "--list-dependency-names",
            action="store_true",
            default=False,
            help=(
                "List all non-nodeid dependency names + the tests they resolve to. "
                "Will also list all nodeid dependency names when verbosity is high enough."
            ),
        )

    # 添加一个标志来列出所有测试的所有（已解析的）依赖项 + 无法解析的名称
    if "--list-processed-dependencies" not in current_options:
        group.addoption(
            "--list-processed-dependencies",
            action="store_true",
            default=False,
            help="List all dependencies of all tests as a list of nodeids + the names that could not be resolved.",
        )

    # 添加一个ini选项和标志，选择对于失败的依赖项要采取的操作
    if "--failed-dependency-action" not in current_options:
        _add_ini_and_option(
            parser,
            group,
            name="failed_dependency_action",
            help=(
                "The action to take when a test has dependencies that failed. "
                'Use "run" to run the test anyway, "skip" to skip the test, and "fail" to fail the test.'
            ),
            default="skip",
            choices=DEPENDENCY_PROBLEM_ACTIONS.keys(),
        )

    # 添加一个ini选项和标志，选择对于无法解析的依赖项要采取的操作
    if "--missing-dependency-action" not in current_options:
        _add_ini_and_option(
            parser,
            group,
            name="missing_dependency_action",
            help=(
                "The action to take when a test has dependencies that cannot be found within the current scope. "
                'Use "run" to run the test anyway, "skip" to skip the test, and "fail" to fail the test.'
            ),
            default="warning",
            choices=DEPENDENCY_PROBLEM_ACTIONS.keys(),
        )
# 配置 pytest，设置依赖管理器并将其添加到全局管理器列表中
def pytest_configure(config: Any) -> None:
    manager = DependencyManager()
    managers.append(manager)

    # 设置处理依赖问题的方式
    manager.options["failed_dependency_action"] = _get_ini_or_option(
        config,
        "failed_dependency_action",
        list(DEPENDENCY_PROBLEM_ACTIONS.keys()),
    )
    manager.options["missing_dependency_action"] = _get_ini_or_option(
        config,
        "missing_dependency_action",
        list(DEPENDENCY_PROBLEM_ACTIONS.keys()),
    )

    # 注册标记
    config.addinivalue_line(
        "markers",
        "depends(name='name', on=['other_name']): marks depencies between tests.",
    )


# 修改 pytest 收集到的测试项
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: Any, items: list[Item]) -> None:
    manager = managers[-1]

    # 将找到的测试项注册到管理器中
    manager.items = items

    # 如果请求显示额外信息
    if config.getoption("list_dependency_names"):
        verbose = config.getoption("verbose") > 1
        manager.print_name_map(verbose)
    if config.getoption("list_processed_dependencies"):
        color = config.getoption("color")
        manager.print_processed_dependencies(color)

    # 重新排序测试项，使测试在其依赖之后运行
    items[:] = manager.sorted_items


# 运行测试并生成报告
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item) -> Any:
    manager = managers[-1]

    # 运行测试步骤
    outcome = yield

    # 将结果存储在管理器中
    manager.register_result(item, outcome.get_result())


# 调用测试项
def pytest_runtest_call(item: Item) -> None:
    manager = managers[-1]

    # 处理缺失的依赖项
    missing_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["missing_dependency_action"]
    ]
    missing = manager.get_missing(item)
    # 如果存在缺失的依赖项且有处理函数，则执行缺失依赖项的处理函数
    if missing_dependency_action and missing:
        missing_dependency_action(
            f'{item.nodeid} depends on {", ".join(missing)}, which was not found'
        )

    # 获取失败的依赖项处理函数
    failed_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["failed_dependency_action"]
    ]
    # 获取失败的依赖项
    failed = manager.get_failed(item)
    # 如果存在失败的依赖项且有处理函数，则执行失败依赖项的处理函数
    if failed_dependency_action and failed:
        failed_dependency_action(f'{item.nodeid} depends on {", ".join(failed)}')
# 在 pytest 结束时执行的函数，用于弹出 managers 列表的最后一个元素
def pytest_unconfigure() -> None:
    managers.pop()
```