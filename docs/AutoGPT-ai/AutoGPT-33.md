# AutoGPT源码解析 33

# `benchmark/agbenchmark/utils/dependencies/main.py`

这段代码是一个pytest模块，用于管理测试中依赖关系。它通过提供在__init__.py文件中的方法来实现这一逻辑。这些方法在pytest钩子中使用，如颜色、网络和代码自动覆盖率等。模块的主要功能是初始化pytest钩子中的参数，并为用户提供可扩展的API。


```py
"""
A module to manage dependencies between pytest tests.

This module provides the methods implementing the main logic. These are used in the pytest hooks that are in
__init__.py.
"""

import collections
import json
import os
from typing import Any, Generator

import colorama
import networkx
from _pytest.nodes import Item

```

该代码的作用是创建一个名为 `TestResult` 的类，用于跟踪单个测试的结果。该类有两个方法：`__init__` 和 `register_result`。

`__init__` 方法用于初始化每个测试的步骤和结果。该方法创建一个名为 `self.nodeid` 的属性，用于存储测试的节点的ID，以及一个名为 `self.results` 的属性，用于存储每个步骤的结果。

`register_result` 方法用于将测试结果注册到 `self.results` 属性中。它接收一个结果对象 `result`，并在 `self.STEPS` 列表中检查 `result.when` 是否存在于 `self.results` 中。如果是，就创建一个名为 `result.outcome` 的属性，并将其添加到 `self.results` 中。如果不是，就创建一个名为 `None` 的属性，并将其添加到 `self.results` 中。

该类的 `__init__` 方法还定义了一个 `self.STEPS` 列表，该列表包含了所有的步骤。该列表中的每个步骤都有一个默认的 `when` 值 `None`，用于指示该步骤是否应该在结果中报告。如果步骤 `when` 在 `self.STEPS` 中不存在，那么该步骤的结果不会被报告。

该类的 `register_result` 方法还定义了一个 `self.GOOD_OUTCOMES` 列表，该列表包含了该测试中期望得到的好结果。如果测试的所有步骤中至少有一个步骤的结果不属于 `self.GOOD_OUTCOMES`，那么该函数就会抛出一个 `ValueError` 异常。


```py
from .constants import MARKER_KWARG_DEPENDENCIES, MARKER_NAME
from .graphs import graph_interactive_network
from .util import clean_nodeid, get_absolute_nodeid, get_markers, get_name


class TestResult(object):
    """Keeps track of the results of a single test."""

    STEPS = ["setup", "call", "teardown"]
    GOOD_OUTCOMES = ["passed"]

    def __init__(self, nodeid: str) -> None:
        """Create a new instance for a test with a given node id."""
        self.nodeid = nodeid
        self.results: dict[str, Any] = {}

    def register_result(self, result: Any) -> None:
        """Register a result of this test."""
        if result.when not in self.STEPS:
            raise ValueError(
                f"Received result for unknown step {result.when} of test {self.nodeid}"
            )
        if result.when in self.results:
            raise AttributeError(
                f"Received multiple results for step {result.when} of test {self.nodeid}"
            )
        self.results[result.when] = result.outcome

    @property
    def success(self) -> bool:
        """Whether the entire test was successful."""
        return all(
            self.results.get(step, None) in self.GOOD_OUTCOMES for step in self.STEPS
        )


```

这段代码定义了一个名为 `TestDependencies` 的类，用于表示单个测试中已经解决或未解决的依赖关系。

在 `__init__` 方法中，创建了一个新的实例，其中包含一个 `Item` 类型的参数 `item`，一个 `DependencyManager` 类型的参数 `manager`，以及一些内部变量。

在 `__init__` 方法中，首先将 `item` 的 `nodeid` 值转换为绝对路径名，以便在 `get_absolute_nodeid` 方法中使用。然后，创建了一个 `set` 集合来存储该测试中所有的已解决依赖关系，并创建了一个 `set` 集合来存储所有尚未解决的依赖关系。

接着，使用 `get_markers` 函数从 `item` 中提取出所有已知标记，并使用解析器将解析出的标记名称存储到一个 `set` 集合中。然后，遍历 `markers` 中的每个标记，尝试从 `marker.kwargs.get(MARKER_KWARG_DEPENDENCIES, [])` 列表中提取出该标记的依赖关系。如果该依赖关系的名称在 `manager.name_to_nodeids` 映射中，则将其添加到 `dependencies` 集合中。否则，将其添加到 `unresolved` 集合中。

最后，将 `item` 作为参数传递给 `__init__` 方法，并在 `__init__` 方法返回后返回自己。


```py
class TestDependencies(object):
    """Information about the resolved dependencies of a single test."""

    def __init__(self, item: Item, manager: "DependencyManager") -> None:
        """Create a new instance for a given test."""
        self.nodeid = clean_nodeid(item.nodeid)
        self.dependencies = set()
        self.unresolved = set()

        markers = get_markers(item, MARKER_NAME)
        dependencies = [
            dep
            for marker in markers
            for dep in marker.kwargs.get(MARKER_KWARG_DEPENDENCIES, [])
        ]
        for dependency in dependencies:
            # If the name is not known, try to make it absolute (ie file::[class::]method)
            if dependency not in manager.name_to_nodeids:
                absolute_dependency = get_absolute_nodeid(dependency, self.nodeid)
                if absolute_dependency in manager.name_to_nodeids:
                    dependency = absolute_dependency

            # Add all items matching the name
            if dependency in manager.name_to_nodeids:
                for nodeid in manager.name_to_nodeids[dependency]:
                    self.dependencies.add(nodeid)
            else:
                self.unresolved.add(dependency)


```

This appears to be a Python class for a machine learning challenge, specifically the "Process Control" task. It contains methods for registering the results of a test, getting a list of failed dependencies, and getting a list of missing dependencies.

The `register_result` method takes an item, a result, and is used to register the result with the given item. The result is expected to be a dict that contains information about the test, such as the node ID, the result, and any additional data that can be used to track the test.

The `get_failed` method takes an item and returns a list of all unfulfilled dependencies for the given item. This is done by traversing the dependency tree and checking the dependencies of each node, and returning any dependencies that have not been satisfied.

The `get_missing` method is similar to `get_failed`, but it returns a list of all missing dependencies for the given item. This is done by traversing the dependency tree and checking the unresolved dependencies of each node, and returning any dependencies that have not been satisfied.


```py
class DependencyManager(object):
    """Keep track of tests, their names and their dependencies."""

    def __init__(self) -> None:
        """Create a new DependencyManager."""
        self.options: dict[str, Any] = {}
        self._items: list[Item] | None = None
        self._name_to_nodeids: Any = None
        self._nodeid_to_item: Any = None
        self._results: Any = None

    @property
    def items(self) -> list[Item]:
        """The collected tests that are managed by this instance."""
        if self._items is None:
            raise AttributeError("The items attribute has not been set yet")
        return self._items

    @items.setter
    def items(self, items: list[Item]) -> None:
        if self._items is not None:
            raise AttributeError("The items attribute has already been set")
        self._items = items

        self._name_to_nodeids = collections.defaultdict(list)
        self._nodeid_to_item = {}
        self._results = {}
        self._dependencies = {}

        for item in items:
            nodeid = clean_nodeid(item.nodeid)
            # Add the mapping from nodeid to the test item
            self._nodeid_to_item[nodeid] = item
            # Add the mappings from all names to the node id
            name = get_name(item)
            self._name_to_nodeids[name].append(nodeid)
            # Create the object that will contain the results of this test
            self._results[nodeid] = TestResult(clean_nodeid(item.nodeid))

        # Don't allow using unknown keys on the name_to_nodeids mapping
        self._name_to_nodeids.default_factory = None

        for item in items:
            nodeid = clean_nodeid(item.nodeid)
            # Process the dependencies of this test
            # This uses the mappings created in the previous loop, and can thus not be merged into that loop
            self._dependencies[nodeid] = TestDependencies(item, self)

    @property
    def name_to_nodeids(self) -> dict[str, list[str]]:
        """A mapping from names to matching node id(s)."""
        assert self.items is not None
        return self._name_to_nodeids

    @property
    def nodeid_to_item(self) -> dict[str, Item]:
        """A mapping from node ids to test items."""
        assert self.items is not None
        return self._nodeid_to_item

    @property
    def results(self) -> dict[str, TestResult]:
        """The results of the tests."""
        assert self.items is not None
        return self._results

    @property
    def dependencies(self) -> dict[str, TestDependencies]:
        """The dependencies of the tests."""
        assert self.items is not None
        return self._dependencies

    def print_name_map(self, verbose: bool = False) -> None:
        """Print a human-readable version of the name -> test mapping."""
        print("Available dependency names:")
        for name, nodeids in sorted(self.name_to_nodeids.items(), key=lambda x: x[0]):
            if len(nodeids) == 1:
                if name == nodeids[0]:
                    # This is just the base name, only print this when verbose
                    if verbose:
                        print(f"  {name}")
                else:
                    # Name refers to a single node id, so use the short format
                    print(f"  {name} -> {nodeids[0]}")
            else:
                # Name refers to multiple node ids, so use the long format
                print(f"  {name} ->")
                for nodeid in sorted(nodeids):
                    print(f"    {nodeid}")

    def print_processed_dependencies(self, colors: bool = False) -> None:
        """Print a human-readable list of the processed dependencies."""
        missing = "MISSING"
        if colors:
            missing = f"{colorama.Fore.RED}{missing}{colorama.Fore.RESET}"
            colorama.init()
        try:
            print("Dependencies:")
            for nodeid, info in sorted(self.dependencies.items(), key=lambda x: x[0]):
                descriptions = []
                for dependency in info.dependencies:
                    descriptions.append(dependency)
                for dependency in info.unresolved:
                    descriptions.append(f"{dependency} ({missing})")
                if descriptions:
                    print(f"  {nodeid} depends on")
                    for description in sorted(descriptions):
                        print(f"    {description}")
        finally:
            if colors:
                colorama.deinit()

    @property
    def sorted_items(self) -> Generator:
        """Get a sorted list of tests where all tests are sorted after their dependencies."""
        # Build a directed graph for sorting
        build_skill_tree = os.getenv("BUILD_SKILL_TREE")
        BUILD_SKILL_TREE = (
            build_skill_tree.lower() == "true" if build_skill_tree else False
        )
        dag = networkx.DiGraph()

        # Insert all items as nodes, to prevent items that have no dependencies and are not dependencies themselves from
        # being lost
        dag.add_nodes_from(self.items)

        # Insert edges for all the dependencies
        for item in self.items:
            nodeid = clean_nodeid(item.nodeid)
            for dependency in self.dependencies[nodeid].dependencies:
                dag.add_edge(self.nodeid_to_item[dependency], item)

        labels = {}
        for item in self.items:
            try:
                with open(item.cls.CHALLENGE_LOCATION) as f:
                    data = json.load(f)
            except:
                data = {}

            node_name = get_name(item)
            data["name"] = node_name
            labels[item] = data

        # only build the tree if it's specified in the env and is a whole run
        if BUILD_SKILL_TREE:
            # graph_spring_layout(dag, labels)
            graph_interactive_network(dag, labels, html_graph_path="")

        # Sort based on the dependencies
        return networkx.topological_sort(dag)

    def register_result(self, item: Item, result: Any) -> None:
        """Register a result of a test."""
        nodeid = clean_nodeid(item.nodeid)
        self.results[nodeid].register_result(result)

    def get_failed(self, item: Item) -> Any:
        """Get a list of unfulfilled dependencies for a test."""
        nodeid = clean_nodeid(item.nodeid)
        failed = []
        for dependency in self.dependencies[nodeid].dependencies:
            result = self.results[dependency]
            if not result.success:
                failed.append(dependency)
        return failed

    def get_missing(self, item: Item) -> Any:
        """Get a list of missing dependencies for a test."""
        nodeid = clean_nodeid(item.nodeid)
        return self.dependencies[nodeid].unresolved

```

# `benchmark/agbenchmark/utils/dependencies/util.py`

这段代码定义了一些工具函数来处理测试用例的标识符。"clean_nodeid()"函数从给定的参数中删除任何多余的":"。这个函数主要用于清理测试用例标识符中的多余部分，使其更具可读性。


```py
""" Utility functions to process the identifiers of tests. """
import re
from typing import Iterator

from _pytest.mark.structures import Mark
from _pytest.nodes import Item

from .constants import MARKER_KWARG_ID, MARKER_NAME

REGEX_PARAMETERS = re.compile(r"\[.+\]$")


def clean_nodeid(nodeid: str) -> str:
    """
    Remove any superfluous ::() from a node id.

    >>> clean_nodeid('test_file.py::TestClass::()::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::test')
    'test_file.py::test'
    """
    return nodeid.replace("::()::", "::")


```



This code defines two functions that take in a `nodeid` string and a `scope` string, and returns a transformed `nodeid` string.

The first function `strip_nodeid_parameters()` takes the `nodeid` string as input and returns the stripped parameters from the `nodeid`. It does this by removing the `RegEX_PARAMETERS` regular expression pattern, which would match the `nodeid` string.

The second function `get_absolute_nodeid()` takes the `nodeid` string and the `scope` string as input, and returns the transformed absolute `nodeid`. It does this by splitting the `nodeid` string into its component parts, and then using the `RegEX_PARAMETERS` regular expression pattern to remove the `::` references to the file and class, if present. It then joins the parts back into a single string, and if the `scope` string is present, it adds the file scope to the base nodeid. Finally, it calls the `clean_nodeid()` function, which is responsible for cleaning up the nodeid string.

Both functions are designed to handle a potentially complex nodeid string that includes both class and file references, and convert it into a simpler form that can be easily understood by humans.


```py
def strip_nodeid_parameters(nodeid: str) -> str:
    """
    Strip parameters from a node id.

    >>> strip_nodeid_parameters('test_file.py::TestClass::test[foo]')
    'test_file.py::TestClass::test'
    >>> strip_nodeid_parameters('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    """
    return REGEX_PARAMETERS.sub("", nodeid)


def get_absolute_nodeid(nodeid: str, scope: str) -> str:
    """
    Transform a possibly relative node id to an absolute one using the scope in which it is used.

    >>> scope = 'test_file.py::TestClass::test'
    >>> get_absolute_nodeid('test2', scope)
    'test_file.py::TestClass::test2'
    >>> get_absolute_nodeid('TestClass2::test2', scope)
    'test_file.py::TestClass2::test2'
    >>> get_absolute_nodeid('test_file2.py::TestClass2::test2', scope)
    'test_file2.py::TestClass2::test2'
    """
    parts = nodeid.split("::")
    # Completely relative (test_name), so add the full current scope (either file::class or file)
    if len(parts) == 1:
        base_nodeid = scope.rsplit("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    # Contains some scope already (Class::test_name), so only add the current file scope
    elif "." not in parts[0]:
        base_nodeid = scope.split("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    return clean_nodeid(nodeid)


```

这段代码定义了一个名为 `get_name` 的函数，它接收一个名为 `item` 的测试项参数，并返回该测试项的名称。

该函数首先定义了一个空字符串 `name`，用于存储测试项的名称。

接着，函数使用 `get_markers` 函数来获取与测试项相关的标记物(也就是 `MARKER_NAME` 参数)。

然后，函数遍历 `markers` 列表中的每个标记物，并检查它是否包含一个名为 `MARKER_KWARG_ID` 的参数。如果是，函数将该参数的值存储在 `name` 变量中。

最后，函数通过调用 `name` 变量来输出测试项的名称。


```py
def get_name(item: Item) -> str:
    """
    Get all names for a test.

    This will use the following methods to determine the name of the test:
        - If given, the custom name(s) passed to the keyword argument name on the marker
    """
    name = ""

    # Custom name
    markers = get_markers(item, MARKER_NAME)
    for marker in markers:
        if MARKER_KWARG_ID in marker.kwargs:
            name = marker.kwargs[MARKER_KWARG_ID]

    return name


```

这段代码定义了一个名为 `get_markers` 的函数，它接收两个参数 `item` 和 `name`。函数返回一个名为 `Iterator[Mark]` 的迭代器类型，其中 `Mark` 是自定义的标记类，包含一个名为 `name` 的属性，用于指定要查找的标记名称。

函数的主要逻辑是遍历 `item` 对象中的所有标记，并检查每个标记是否具有与传递给函数的标记名称相同的名称。如果是，函数将返回该标记对象，通过 `yield` 语句将其输出。

因此，该函数的作用是获取具有指定名称的标记对象，用于指定一个给定物品对象的标记名称。


```py
def get_markers(item: Item, name: str) -> Iterator[Mark]:
    """Get all markers with the given name for a given item."""
    for marker in item.iter_markers():
        if marker.name == name:
            yield marker

```

# `benchmark/agbenchmark/utils/dependencies/__init__.py`

这段代码是一个pytest插件的配置模块，提供了用于测试的功能。它的逻辑都在main.py中。具体来说，它是一个包含两个函数的模块，一个是warnings.py中的一个函数，另一个是在这个插件的配置中定义的。

这个插件的作用是允许在测试函数中使用第三方库或模块时，通过导入相应的包或模块来禁止警告信息的发送。通过这种方式，可以更轻松地编写测试用例，同时避免在测试过程中意外的警告信息对测试的影响。

具体来说，这个插件需要定义一个选项组(OptionGroup)，该选项组用于指定允许使用的警告。然后，在测试函数中调用deprecate函数时，插件会检查该函数是否使用了警告，如果是，则调用一个特定的函数进行处理，从而允许该测试继续进行。

此外，插件还提供了一个option_group参数，用于指定包含所有允许使用的警告的选项组。通过这种方式，用户可以更轻松地管理允许使用的警告，同时仍然可以利用警告的优势来提高测试的准确性。


```py
"""
A module that provides the pytest hooks for this plugin.

The logic itself is in main.py.
"""

import warnings
from typing import Any, Callable, Optional

import pytest
from _pytest.config.argparsing import OptionGroup, Parser
from _pytest.nodes import Item

from .main import DependencyManager

```

这段代码定义了一个名为 `managers` 的列表类型，其初始值为空列表 `[]`。

接下来定义了一个名为 `DEPENDENCY_PROBLEM_ACTIONS` 的字典类型，其中包含 5 个键值对，每个键都是一个字符串类型的函数，它们的作用是在依赖关系问题出现时执行相应的操作。具体来说，这些函数分别是：

- `"run"`：执行依赖关系问题的运行操作
- `"skip"`：延迟执行 `run` 操作，但不会输出任何错误信息
- `"fail"`：执行 `"run"` 操作，并输出错误信息
- `"warning"`：执行 `"fail"` 或 `"skip"` 操作，但不会输出警告信息

此外，还有一段注释，指出如何调用这些选项。


```py
managers: list[DependencyManager] = []


DEPENDENCY_PROBLEM_ACTIONS: dict[str, Callable[[str], None] | None] = {
    "run": None,
    "skip": lambda m: pytest.skip(m),
    "fail": lambda m: pytest.fail(m, False),
    "warning": lambda m: warnings.warn(m),
}


def _add_ini_and_option(
    parser: Any,
    group: OptionGroup,
    name: str,
    help: str,
    default: str | bool | int,
    **kwargs: Any,
) -> None:
    """Add an option to both the ini file as well as the command line flags, with the latter overriding the former."""
    parser.addini(
        name,
        help + " This overrides the similarly named option from the config.",
        default=default,
    )
    group.addoption(f'--{name.replace("_", "-")}', help=help, default=None, **kwargs)


```

The `current_options` list is being modified to include the new `--list-dependency-names` and `--list-processed-dependencies` options.

The `--list-dependency-names` option will list all non-nodeid dependency names for each test, as well as the tests they resolve to.

The `--list-processed-dependencies` option will list all dependencies of all tests as a list of nodeids, along with the names that could not be resolved.

The `failed_dependency_action` and `missing_dependency_action` options have been added as ini options and flags to choose the action to take for failed and missing dependencies, respectively. These options are available when the `--failed-dependency-action` and `--missing-dependency-action` options are not specified in the `current_options` list.


```py
def _get_ini_or_option(
    config: Any, name: str, choices: Optional[list[str]]
) -> str | None:
    """Get an option from either the ini file or the command line flags, the latter taking precedence."""
    value = config.getini(name)
    if value is not None and choices is not None and value not in choices:
        raise ValueError(
            f'Invalid ini value for {name}, choose from {", ".join(choices)}'
        )
    return config.getoption(name) or value


def pytest_addoption(parser: Parser) -> None:
    # get all current option strings
    current_options = []
    for action in parser._anonymous.options:
        current_options += action._short_opts + action._long_opts

    for group in parser._groups:
        for action in group.options:
            current_options += action._short_opts + action._long_opts

    group = parser.getgroup("depends")

    # Add a flag to list all names + the tests they resolve to
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

    # Add a flag to list all (resolved) dependencies for all tests + unresolvable names
    if "--list-processed-dependencies" not in current_options:
        group.addoption(
            "--list-processed-dependencies",
            action="store_true",
            default=False,
            help="List all dependencies of all tests as a list of nodeids + the names that could not be resolved.",
        )

    # Add an ini option + flag to choose the action to take for failed dependencies
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

    # Add an ini option + flag to choose the action to take for unresolved dependencies
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


```

这段代码是一个pytest库中的配置函数，名为`pytest_configure`，其作用是在测试开始时对测试进行一些设置，包括设置如何处理与依赖关系问题的选项，以及注册一个标记来指示测试依赖于哪些依赖项。

具体来说，代码首先创建了一个名为`DependencyManager`的依赖管理器对象，并将其添加到`managers`列表中。接下来，代码使用`_get_ini_or_option`函数从配置中读取有关依赖关系的选项，并将其存储在`manager.options`中。这个选项是一个元组，其中包含两个键，分别是`failed_dependency_action`和`missing_dependency_action`，它们分别指定在依赖关系失败时应该执行的选项。如果这些选项不存在，则默认执行DEPENDENCY_PROBLEM_ACTIONS中定义的第一个选项。

接下来，代码注册了一个标记，用于指示测试依赖于哪些依赖项。这个标记通过在`config.addinivalue_line`函数中传递一个包含两个参数的字符串来实现。第一个参数是一个字符串，指定标记中包含哪些选项。第二个参数是一个包含两个元素的列表，指定每个选项在标记中的行为。在这里，标记将告知pytest运行时忽略与测试依赖于的依赖项的选项，而不是执行DEPENDENCY_PROBLEM_ACTIONS中定义的第一个选项。


```py
def pytest_configure(config: Any) -> None:
    manager = DependencyManager()
    managers.append(manager)

    # Setup the handling of problems with dependencies
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

    # Register marker
    config.addinivalue_line(
        "markers",
        "depends(name='name', on=['other_name']): marks depencies between tests.",
    )


```

这段代码是一个 Pytest 测试代码钩子，它的作用是在测试执行时修改Manager对象的物品列表，并将注册的测试在Manager上。

具体来说，代码中首先使用 `Any` 类型来表示要修改的Manager对象，然后使用 `manager[-1]` 获取到当前Manager对象。接着，使用列表推导式 `items = [Item]` 创建一个待修改的列表，将列表中的所有Item对象赋值给Manager对象中的 `items` 属性。

接下来，代码中使用 `trylast=True` 参数，使得在每次获取Manager对象时，尝试使用该Manager对象去获取Manager对象的最后一种方式(即通过 `keys()` 方法获取)。这样做是为了确保在每次获取Manager对象时，获取到的Manager对象都相同，从而保证每次测试运行时，Manager对象的物品列表都是一致的。

然后，代码中使用 Manager对象的方法 `items = items` 将列表中的所有Item对象复制到Manager对象中的 `items` 属性中，这样就完成了对Manager对象中物品列表的修改。

接着，代码中使用 `if config.getoption("list_dependency_names"):` 判断是否需要输出测试依赖关系的名称，如果需要输出，则使用 `print_name_map` 方法输出测试依赖关系的名称，并在后面加上 `verbose` 参数，表示输出更加详细的信息。如果不需要输出，则跳过该判断，继续执行下一次测试。

最后，代码中使用 `if config.getoption("list_processed_dependencies"):` 判断是否需要输出测试处理依赖关系的名称，如果需要输出，则使用 `print_processed_dependencies` 方法输出测试处理依赖关系的名称，并在后面加上 `color` 参数，表示输出更加详细的信息，其中 `color` 参数是一个字符串，用于指定颜色。如果不需要输出，则跳过该判断，继续执行下一次测试。

整个函数的作用是，在测试执行时修改Manager对象的物品列表，并将注册的测试在Manager上，如果需要输出测试依赖关系或处理依赖关系的名称，则根据配置参数进行相应的操作。


```py
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: Any, items: list[Item]) -> None:
    manager = managers[-1]

    # Register the founds tests on the manager
    manager.items = items

    # Show the extra information if requested
    if config.getoption("list_dependency_names"):
        verbose = config.getoption("verbose") > 1
        manager.print_name_map(verbose)
    if config.getoption("list_processed_dependencies"):
        color = config.getoption("color")
        manager.print_processed_dependencies(color)

    # Reorder the items so that tests run after their dependencies
    items[:] = manager.sorted_items


```

这段代码是用来进行 pytest 测试的 runtests 函数。runtests 是 pytest 提供的一个用于运行测试的扩展，可以允许您使用 Python 编写测试函数。

这段代码的具体作用如下：

1. 在测试函数内定义了两个 hook：tryfirst 和 hookwrapper。tryfirst 表示在测试函数内捕获 try-except 语句，而 hookwrapper 表示在测试函数内创建一个 hook 对象，该对象将传递给 pytest 的 tryfirst 选项。
2. 在 hookwrapper 方法内，创建了一个 manager 对象，它是 pytest 的一个管理器（manager）。manager 对象管理着一系列测试函数，包括 register_result 和 get_failed 等。
3. 在 pytest_runtest_call 函数内，处理了测试函数的 missing dependencies。具体来说，如果测试函数依赖于某个组件，而该组件尚未被测试，会执行特定的 action。这段代码定义了两个 action：missing_dependency_action 和 failed_dependency_action。如果测试函数依赖于某些组件，而这些组件尚未被测试，将会执行 missing_dependency_action。如果测试函数依赖于某些组件，而这些组件已经失败，将会执行 failed_dependency_action。
4. 在 pytest_runtest_call 函数内，通过注册结果和失败的结果到 manager 对象中，实现了注册结果的功能。
5. 在 pytest_runtest_makereport 函数内，运行了测试函数，并返回了测试的运行结果。该结果将作为 pytest 的实验报告的一部分输出。


```py
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item) -> Any:
    manager = managers[-1]

    # Run the step
    outcome = yield

    # Store the result on the manager
    manager.register_result(item, outcome.get_result())


def pytest_runtest_call(item: Item) -> None:
    manager = managers[-1]

    # Handle missing dependencies
    missing_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["missing_dependency_action"]
    ]
    missing = manager.get_missing(item)
    if missing_dependency_action and missing:
        missing_dependency_action(
            f'{item.nodeid} depends on {", ".join(missing)}, which was not found'
        )

    # Check whether all dependencies succeeded
    failed_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["failed_dependency_action"]
    ]
    failed = manager.get_failed(item)
    if failed_dependency_action and failed:
        failed_dependency_action(f'{item.nodeid} depends on {", ".join(failed)}')


```

这段代码是一个Python测试框架pytest的函数，被称为“声明作用域”函数(声明函数)。

在函数内部，首先使用managers.pop()这个函数，将其返回值赋给一个名为变量p的局部变量。

然后，将p赋值为一个名为“managers”的匿名函数，这个匿名函数调用了pytest_unconfigure()函数，并将返回值p作为参数传入。

最后，将p赋值为一个名为“None”的变量，用于确保函数代码在退出函数时不会留下任何未清理的资源。


```py
def pytest_unconfigure() -> None:
    managers.pop()

```

# `benchmark/backend/__init__.py`

很抱歉，我无法不输出源代码，因为我是一个 AI 语言模型，不会在输出时包含任何可执行的代码。我只能解释代码的作用，如果您可以提供代码，我会尽力解释其作用。


```py

```

# agbenchmark-frontend

Frontend for https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks

Objectively know how well your agent is performing in categories like code, retrieval, memory, and safety.

Save time and money while doing it through smart dependencies. Best part? It's all automated.


# `benchmark/frontend/tailwind.config.ts`

这段代码是一个 TypeScript 配置文件，定义了一个可以使用 Tailwind CSS 样式的应用程序。

具体来说，该配置文件做了以下几件事情：

1. 导入了 "tailwindcss" 包，这是用于创建自定义 Tailwind CSS 样式的库。

2. 在应用程序的根目录下定义了一个内容(content)选项，它指定了哪些文件可以使用 Tailwind CSS 样式。在这个例子中，定义的内容选项是 `"./src/**/*.{js,ts,jsx,tsx}"` ，这意味着所有的 `.js`、`.ts`、`.jsx` 和 `.tsx` 文件都可以使用 Tailwind CSS 样式。

3. 定义了一个主题(theme)，其中包括了一些自定义配置，例如扩展(extend)和插件(plugins)。在这里，使用了 `{}` 语法来扩展默认的 Tailwind CSS 主题，添加了一些自定义主题配置。

4. 返回了一个 `Config` 对象，这个对象是 Tailwind CSS 配置文件的标准格式，包含了应用程序的所有配置选项。


```py
import { type Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;

```

# `benchmark/frontend/src/server/db.ts`

这段代码使用了JavaScript的解构语法，将变量 `globalForPrisma` 的行为赋值给一个名为 `prisma` 的变量。

具体来说，代码首先引入了名为 `PrismaClient` 的类，然后使用 `import { env } from "~/env.mjs"` 导入了一个名为 `env` 的环境变量对象。接着，在 `const globalForPrisma = ...` 这一行中，将 `globalThis`（即 `window this`，因为 `globalThis` 是全局对象，而 `window this` 是在全局作用域内绑定在 `window` 对象上的一个副本）的行为赋值给 `prisma`。这里的 `...`（三个星号）表示将 `globalThis` 的行为展开到一个临时变量中，然后赋值给 `prisma`。

接下来，代码判断了环境变量 `env.NODE_ENV` 的值是否为 `"production"`。如果是，那么代码将 `globalForPrisma.prisma` 的值设置为调用 `PrismaClient` 构造函数时传入的第一个参数，即 `new PrismaClient`。否则，如果 `env.NODE_ENV` 不是 `"production"`，则将 `globalForPrisma.prisma` 的值设置为调用 `PrismaClient` 构造函数时传入的第二个参数，即 `PrismaClient`。这样，无论当前运行的环境是什么，`prisma` 的值都将是一个类同于 `globalForPrisma` 的 `PrismaClient` 实例。


```py
import { PrismaClient } from "@prisma/client";
import { env } from "~/env.mjs";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log:
      env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"],
  });

if (env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

```

# `benchmark/reports/json_to_base_64.py`

这段代码的作用是：

1. 导入 `base64` 和 `json` 库。
2. 加载一个名为 `secrets.json` 的 JSON 文件中的数据。
3. 将 JSON 数据转换为字符串。
4. 将字符串转换为字节序列。
5. 将字节序列转换为 Base64 字符串。
6. 将 Base64 字符串返回，以便稍后将其解码为原始 JSON 数据。


```py
import base64
import json

# Load JSON data from a file
with open("secrets.json", "r") as f:
    data = json.load(f)

# Convert the JSON object into a string
json_string = json.dumps(data)

# Encode the string into bytes
json_bytes = json_string.encode("utf-8")

# Convert the bytes to a base64 string
base64_string = base64.b64encode(json_bytes).decode("utf-8")

```

这行代码使用了 Python 的 `print()` 函数来输出一个 Base64 编码的字符串。

具体地说，`print(base64_string)` 将一个名为 `base64_string` 的字符串作为参数传递给 `print()` 函数，这个字符串会被转换成Base64编码形式，然后输出一个字符串，该字符串将包含原始字符串的 Base64 编码。

例如，如果 `base64_string` 的内容是 `"Hello, World!"`，则 `print(base64_string)` 将输出 `"bZGradeturyW6hJvG11F8bJHqNOPd8Q0`，这是该字符串的 Base64 编码形式。


```py
print(base64_string)

```

# `benchmark/reports/match_records.py`

这段代码的作用是定义了一个名为 "Metrics" 的类，该类包含了一些可以报告给测试套件的度量指标。

具体来说，该类定义了一个 "difficulty" 字段，它可以是任何类型（包括字符串和数字），用于表示测试的难度水平。该类还定义了一个 "success" 字段，它是一个布尔值，表示测试是否成功。

另外，该类定义了一个 "success\_percent" 字段，它是一个浮点数，表示成功测试的比例。如果 "success" 为 True，则该字段将被填充为 1.0，否则将被填充为 0.0。

此外，该类定义了一个 "run\_time" 字段，它是一个字符串，用于表示测试运行的时间。如果 "run\_time" 字段没有被提供，则该报告将不包含任何度量指标。

最后，该类定义了一个 "fail\_reason" 字段，它是一个字符串，用于表示测试失败的原因。如果 "fail\_reason" 字段没有被提供，则该报告将不包含任何度量指标。

该类的定义作为对一个 Report 的补充，可以用于生成测试套件的度量指标。


```py
import glob
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from pydantic import BaseModel, Field

# from agbenchmark.reports.processing.report_types import Report, SuiteTest


class Metrics(BaseModel):
    difficulty: str
    success: bool
    success_percent: float = Field(..., alias="success_%")
    run_time: Optional[str] = None
    fail_reason: Optional[str] = None
    attempted: Optional[bool] = None


```

这段代码定义了两个类，一个是`MetricsOverall`，另一个是`Test`。`MetricsOverall`类有两个方法，一个是`run_time`，另一个是`highest_difficulty`。`Test`类也有两个方法，一个是`is_regression`，另一个是`description`。`MetricsOverall`类有一个`percentage`属性，它是可选的，并且没有默认值。`Test`类有一个`description`属性，它是字符串类型的，用于描述测试的内容。`MetricsOverall`类有一个`metrics`属性，它是一个`Metrics`对象，其中`Metrics`类可能包含了一些计算指标，比如运行时间、难度级别等。`Test`类有一个`category`属性，它是一个列表，用于指定测试所属的类别，比如操作系统、浏览器等。`Test`类还有一个`task`属性，它是可选的，用于指定要执行的任务，比如上传文件、访问网页等。`Test`类还有一个`reached_cutoff`属性，它是可选的，用于指定是否达到了某个设置的阈值，比如倒计时、计数器等。


```py
class MetricsOverall(BaseModel):
    run_time: str
    highest_difficulty: str
    percentage: Optional[float] = None


class Test(BaseModel):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None


```

这段代码定义了两个类，一个是 SuiteTest，另一个是 Report。其中 SuiteTest 是继承自 BaseModel 的类，而 Report 是继承自 BaseModel 的类。这两个类都是为了在测试过程中报告测试的运行情况而设计的。

具体来说，SuiteTest 类中定义了一些变量，包括数据文件路径、测试指标、测试用例、测试分类和测试任务等，用于在测试过程中记录和报告测试的运行情况。而 Report 类中定义了一些变量，包括测试指令、测试完成时间、基准测试的运行时间、测试指标和测试用例等，用于在测试结束后记录和报告测试的运行情况。

这些类的实例都位于一个应用程序中，可以在应用程序启动时通过命令行或 API 进行调用，用于测试应用程序的各个功能。


```py
class SuiteTest(BaseModel):
    data_path: str
    metrics: MetricsOverall
    tests: Dict[str, Test]
    category: Optional[List[str]] = None
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None


class Report(BaseModel):
    command: str
    completion_time: str
    benchmark_start_time: str
    metrics: MetricsOverall
    tests: Dict[str, Union[Test, SuiteTest]]
    config: Dict[str, str | dict[str, str]]


```

This code appears to be a Python script that creates a Pandas DataFrame with information about a suite of tests, including the test name, attempt number, category, task, success percentage, difficulty, and run time. It appears to be part of a larger software testing framework, and it is using the data generated by another script.

The script first sets the `challenge` variable to the name of the test, and then loops through each test in the suite. For each test, it creates a dictionary with information about the test, including the test name, attempt number, category, task, success percentage, difficulty, and run time. It then appends this dictionary to a list of dictionaries that is being passed to the `pd.DataFrame` constructor.

The final result is a DataFrame that contains information about each test in the suite, including the test name, attempt number, category, task, success percentage, difficulty, and run time.

It should be noted that this code snippet does not provide enough information to understand the full context of the software testing framework it is part of, and it is recommended to carefully read and modify the code if you are going to use it for any real-world testing purposes.


```py
def get_reports():
    # Initialize an empty list to store the report data
    report_data = []

    # Get the current working directory
    current_dir = os.getcwd()

    # Check if the current directory ends with 'reports'
    if current_dir.endswith("reports"):
        reports_dir = "/"
    else:
        reports_dir = "reports"

    # Iterate over all agent directories in the reports directory
    for agent_name in os.listdir(reports_dir):
        if agent_name is None:
            continue
        agent_dir = os.path.join(reports_dir, agent_name)

        # Check if the item is a directory (an agent directory)
        if os.path.isdir(agent_dir):
            # Construct the path to the report.json file
            # Get all directories and files, but note that this will also include any file, not just directories.
            run_dirs = glob.glob(os.path.join(agent_dir, "*"))

            # Get all json files starting with 'file'
            # old_report_files = glob.glob(os.path.join(agent_dir, "file*.json"))

            # For each run directory, add the report.json to the end
            # Only include the path if it's actually a directory
            report_files = [
                os.path.join(run_dir, "report.json")
                for run_dir in run_dirs
                if os.path.isdir(run_dir)
            ]
            # old_report_files already contains the full paths, so no need to join again
            # report_files = report_files + old_report_files
            for report_file in report_files:
                # Check if the report.json file exists
                if os.path.isfile(report_file):
                    # Open the report.json file
                    with open(report_file, "r") as f:
                        # Load the JSON data from the file
                        json_data = json.load(f)
                        print(f"Processing {report_file}")
                        report = Report.parse_obj(json_data)

                        for test_name, test_data in report.tests.items():
                            test_json = {
                                "agent": agent_name.lower(),
                                "benchmark_start_time": report.benchmark_start_time,
                            }

                            if isinstance(test_data, SuiteTest):
                                if (
                                    test_data.category
                                ):  # this means it's a same task test
                                    test_json["challenge"] = test_name
                                    test_json["attempted"] = test_data.tests[
                                        list(test_data.tests.keys())[0]
                                    ].metrics.attempted
                                    test_json["categories"] = ", ".join(
                                        test_data.category
                                    )
                                    test_json["task"] = test_data.task
                                    test_json["success"] = test_data.metrics.percentage
                                    test_json[
                                        "difficulty"
                                    ] = test_data.metrics.highest_difficulty
                                    test_json[
                                        "success_%"
                                    ] = test_data.metrics.percentage
                                    test_json["run_time"] = test_data.metrics.run_time
                                    test_json["is_regression"] = test_data.tests[
                                        list(test_data.tests.keys())[0]
                                    ].is_regression
                                else:  # separate tasks in 1 suite
                                    for (
                                        suite_test_name,
                                        suite_data,
                                    ) in test_data.tests.items():
                                        test_json["challenge"] = suite_test_name
                                        test_json[
                                            "attempted"
                                        ] = suite_data.metrics.attempted
                                        test_json["categories"] = ", ".join(
                                            suite_data.category
                                        )
                                        test_json["task"] = suite_data.task
                                        test_json["success"] = (
                                            100.0 if suite_data.metrics.success else 0
                                        )
                                        test_json[
                                            "difficulty"
                                        ] = suite_data.metrics.difficulty
                                        test_json[
                                            "success_%"
                                        ] = suite_data.metrics.success_percentage
                                        test_json[
                                            "run_time"
                                        ] = suite_data.metrics.run_time
                                        test_json[
                                            "is_regression"
                                        ] = suite_data.is_regression

                            else:
                                test_json["challenge"] = test_name
                                test_json["attempted"] = test_data.metrics.attempted
                                test_json["categories"] = ", ".join(test_data.category)
                                test_json["task"] = test_data.task
                                test_json["success"] = (
                                    100.0 if test_data.metrics.success else 0
                                )
                                test_json["difficulty"] = test_data.metrics.difficulty
                                test_json[
                                    "success_%"
                                ] = test_data.metrics.success_percentage
                                test_json["run_time"] = test_data.metrics.run_time
                                test_json["is_regression"] = test_data.is_regression

                            report_data.append(test_json)

    return pd.DataFrame(report_data)


```

This is a Python function that appears to fetch record data from an endpoint using the Helicone Search API. The function takes in two parameters, `query` and `size`, and returns a pandas DataFrame containing the requested data.

The function first sends a GET request to the endpoint with the `query` parameter and the `size` parameter. The query parameter is then passed as a variable to the `heliconeRequest` function, which appears to handle the request and return the response.

The function then extracts the request and response data from the response and stores it in a dictionary. The request data is stored in a `request_body` dictionary, while the response data is stored in a `response` dictionary.

The function then iterates through the request and response data and stores the properties of each item in a dictionary. The properties include the `createdAt`, `agent`, `costUSD`, `job_id`, `challenge`, `benchmark_start_time`, `prompt`, `response`, and `model` properties of each item.

The function then returns the data as a pandas DataFrame.

Note that the function does not handle errors that may occur with the API or the Helicone Search API.


```py
def get_helicone_data():
    helicone_api_key = os.getenv("HELICONE_API_KEY")

    url = "https://www.helicone.ai/api/graphql"
    # Replace <KEY> with your personal access key
    transport = AIOHTTPTransport(
        url=url, headers={"authorization": f"Bearer {helicone_api_key}"}
    )

    client = Client(transport=transport, fetch_schema_from_transport=True)

    SIZE = 250

    i = 0

    data = []
    print("Fetching data from Helicone")
    while True:
        query = gql(
            """
            query ExampleQuery($limit: Int, $offset: Int){
                heliconeRequest(
                    limit: $limit
                    offset: $offset
                ) {
                    costUSD
                    prompt
                    properties{
                        name
                        value
                    }
                    
                    requestBody
                    response
                    createdAt

                }

                }
        """
        )
        print(f"Fetching {i * SIZE} to {(i + 1) * SIZE} records")
        try:
            result = client.execute(
                query, variable_values={"limit": SIZE, "offset": i * SIZE}
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            result = None

        i += 1

        if result:
            for item in result["heliconeRequest"]:
                properties = {
                    prop["name"]: prop["value"] for prop in item["properties"]
                }
                data.append(
                    {
                        "createdAt": item["createdAt"],
                        "agent": properties.get("agent"),
                        "costUSD": item["costUSD"],
                        "job_id": properties.get("job_id"),
                        "challenge": properties.get("challenge"),
                        "benchmark_start_time": properties.get("benchmark_start_time"),
                        "prompt": item["prompt"],
                        "response": item["response"],
                        "model": item["requestBody"].get("model"),
                        "request": item["requestBody"].get("messages"),
                    }
                )

        if not result or (len(result["heliconeRequest"]) == 0):
            print("No more results")
            break

    df = pd.DataFrame(data)
    # Drop rows where agent is None
    df = df.dropna(subset=["agent"])

    # Convert the remaining agent names to lowercase
    df["agent"] = df["agent"].str.lower()

    return df


```

这段代码的作用是判断两个文件是否存在，如果存在，则读取它们的内容并存储到两个DataFrame中，最后将它们保存到本地文件系统。如果不存在，则从外部API获取需要的数据，并将获取到的数据存储到两个DataFrame中，最后将它们保存到本地文件系统。

`os.path.exists()`函数用于检查文件是否存在，如果文件存在，则返回True，否则返回False。

`reports_df = pd.read_pickle("raw_reports.pkl")`和`helicone_df = pd.read_pickle("raw_helicone.pkl")`用于读取保存为pickle格式的两个DataFrame。

`try_formats(date_str)`函数用于将传入的日期字符串尝试不同的格式，如果当前格式无法转换，则返回None。这个函数的作用是帮助在将来的代码中处理日期和时间类型。


```py
if os.path.exists("raw_reports.pkl") and os.path.exists("raw_helicone.pkl"):
    reports_df = pd.read_pickle("raw_reports.pkl")
    helicone_df = pd.read_pickle("raw_helicone.pkl")
else:
    reports_df = get_reports()
    reports_df.to_pickle("raw_reports.pkl")
    helicone_df = get_helicone_data()
    helicone_df.to_pickle("raw_helicone.pkl")


def try_formats(date_str):
    formats = ["%Y-%m-%d-%H:%M", "%Y-%m-%dT%H:%M:%S%z"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return None


```

这段代码的作用是对于两个DataFrame helicone_df 和 reports_df，分别对它们的基准测试开始时间(benchmark_start_time)和创建时间(createdAt)列进行转换为datetime类型。

具体来说，对于benchmark_start_time列，首先应用一个名为try_formats的函数，该函数尝试将字符串转换为datetime格式，如果转换成功则将结果保存回原来的列中。如果尝试失败，则返回原始值，不会对原始值进行修改。然后对结果进行应用utc=True参数，使得结果基于协调世界时(UTC)的时间戳。最后，对结果应用pd.to_datetime函数，将结果转换为datetime类型，单位为ms，原名为createdAt。

对于createdAt列，首先对它应用pd.to_datetime函数，将结果转换为datetime类型，单位为ms，原名为createdAt。然后对结果应用unit="ms"参数，使得结果的单位是毫秒。最后，对结果应用origin="unix"参数，将结果的原始值设置为unix时间戳，以便与benchmark_start_time列中的时间戳类型进行匹配。

通过这些转换，可以使得原始的基准测试开始时间和创建时间数据类型与datetime类型完全匹配，可以进行日期时间相关的操作。


```py
helicone_df["benchmark_start_time"] = pd.to_datetime(
    helicone_df["benchmark_start_time"].apply(try_formats), utc=True
)
helicone_df = helicone_df.dropna(subset=["benchmark_start_time"])
helicone_df["createdAt"] = pd.to_datetime(
    helicone_df["createdAt"], unit="ms", origin="unix"
)
reports_df["benchmark_start_time"] = pd.to_datetime(
    reports_df["benchmark_start_time"].apply(try_formats), utc=True
)
reports_df = reports_df.dropna(subset=["benchmark_start_time"])

assert pd.api.types.is_datetime64_any_dtype(
    helicone_df["benchmark_start_time"]
), "benchmark_start_time in helicone_df is not datetime"
```

这段代码的作用是检查一个名为"benchmark_start_time"的列是否为 datetime64 类型，如果不是，则输出 "benchmark_start_time in reports_df is not datetime64类型"。接着，将 "benchmark_start_time" 列中的值替换为 "report_time"，从而使得两个列的名称更加一致。最后，使用 pandas 的 merge_asof 函数，将一个名为 "helicone_df" 的 DataFrame 与 "reports_df" 中的 "benchmark_start_time" 列进行源数据分析，并将其结果保存到一个新的 DataFrame 中。


```py
assert pd.api.types.is_datetime64_any_dtype(
    reports_df["benchmark_start_time"]
), "benchmark_start_time in reports_df is not datetime"

reports_df["report_time"] = reports_df["benchmark_start_time"]

# df = pd.merge_asof(
#     helicone_df.sort_values("benchmark_start_time"),
#     reports_df.sort_values("benchmark_start_time"),
#     left_on="benchmark_start_time",
#     right_on="benchmark_start_time",
#     by=["agent", "challenge"],
#     direction="backward",
# )

```

这段代码的作用是使用 Pandas 的 `merge` 函数将两个数据框 `helicone_df` 和 `reports_df` 按照 `benchmark_start_time`、`agent` 和 `challenge` 列进行内连接，并将结果保存到一个新的数据框 `df` 中。

`how="inner"` 表示只保留内连接后的结果，即只连接 `helicone_df` 和 `reports_df` 中具有相同行索引的行，而不是将它们连接成一个更大的合成数据框。

最后，代码导入了 `df` 数据框，并将其保存为名为 `df.pkl` 的文件，同时输出了一些数据框的信息，包括数据框中所有的数据和索引。如果要将数据框重新加载，可以使用 `pd.read_pickle` 函数将其从 .pkl 文件中读取回来。


```py
df = pd.merge(
    helicone_df,
    reports_df,
    on=["benchmark_start_time", "agent", "challenge"],
    how="inner",
)

df.to_pickle("df.pkl")
print(df.info())
print("Data saved to df.pkl")
print("To load the data use: df = pd.read_pickle('df.pkl')")

```

# `benchmark/reports/send_to_googledrive.py`

这段代码的作用是：

1. 从本地文件系统中读取一个名为 "base64String.txt" 的文件内容，并将其转换为Base64编码的字符串。
2. 将得到的Base64编码的字符串进行解码，得到原始数据。
3. 将原始数据（可能是一个JSON对象）存储为json格式的数据。
4. 将json数据存储到 "data.json" 文件中。
5. 使用GSPREAD库访问一个特定的Google Sheets文件，并读取其中的数据。
6. 将读取的数据与之前存储的json数据进行重合，生成新的DataFrame。
7. 最后，将新的DataFrame存储为 "output.df" 文件。


```py
import base64
import json
import os
import re
from datetime import datetime, timedelta

import gspread
import pandas as pd
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

# Load environment variables from .env file
load_dotenv()

# Get the base64 string from the environment variable
```

这段代码的作用是获取一个名为 "GDRIVE_BASE64" 的环境变量，如果该变量不存在，则抛出一个 "The GDRIVE_BASE64 environment variable is not set" 的异常。然后，它将获取到的 "GDRIVE_BASE64" 环境变量作为 base64 编码的字符串，并将其解码为字节。接着，它将该字节字符串转换为字符串，并尝试将其解码为 JSON 对象。最后，它定义了一个名为 "creds_info" 的 JSON 对象的 "base_directory" 属性，但没有对其进行任何其他操作。


```py
base64_creds = os.getenv("GDRIVE_BASE64")

if base64_creds is None:
    raise ValueError("The GDRIVE_BASE64 environment variable is not set")

# Decode the base64 string into bytes
creds_bytes = base64.b64decode(base64_creds)

# Convert the bytes into a string
creds_string = creds_bytes.decode("utf-8")

# Parse the string into a JSON object
creds_info = json.loads(creds_string)

# Define the base directory containing JSON files
```

这段代码的作用是设置一个名为“base_dir”的变量为报告文件夹（“reports”），然后获取当前工作目录（即当前目录的路径）。接下来，它检查当前目录是否以“reports”结尾，如果不是，则将“base_dir”变量设置为当前目录。然后，它将创建一个名为“rows”的列表来存储每个数据行。


```py
base_dir = "reports"

# Get the current working directory
current_dir = os.getcwd()

# Check if the current directory ends with 'reports'
if current_dir.endswith("reports"):
    base_dir = "/"
else:
    base_dir = "reports"

# Create a list to store each row of data
rows = []


```

This appears to be a Python function that takes in some information about a test, such as the test name, data path, and difficulty, and outputs a table of metrics about the test, such as run time, cost, and attempts. It appears to be doing this for a CI/CD pipeline and also includes information about the benchmarks and the Git commit corresponding to the benchmark. It also appears to be handling the case where a test has multiple answers and outputs the first one it finds.


```py
def process_test(
    test_name: str, test_info: dict, agent_name: str, common_data: dict
) -> None:
    """Recursive function to process test data."""
    parts = test_name.split("_", 1)  # Split by underscore only once
    test_suite = parts[0] if len(parts) > 1 else None

    # transform array into string with | as separator
    separator = "|"
    categories = separator.join(
        test_info.get("category", []),
    )

    row = {
        "Agent": agent_name,
        "Command": common_data.get("command", ""),
        "Completion Time": common_data.get("completion_time", ""),
        "Benchmark Start Time": common_data.get("benchmark_start_time", ""),
        "Total Run Time": common_data.get("metrics", {}).get("run_time", ""),
        "Highest Difficulty": common_data.get("metrics", {}).get(
            "highest_difficulty", ""
        ),
        "Workspace": common_data.get("config", {}).get("workspace", ""),
        "Test Name": test_name,
        "Data Path": test_info.get("data_path", ""),
        "Is Regression": test_info.get("is_regression", ""),
        "Difficulty": test_info.get("metrics", {}).get("difficulty", ""),
        "Success": test_info.get("metrics", {}).get("success", ""),
        "Success %": test_info.get("metrics", {}).get("success_%", ""),
        "Non mock success %": test_info.get("metrics", {}).get(
            "non_mock_success_%", ""
        ),
        "Run Time": test_info.get("metrics", {}).get("run_time", ""),
        "Benchmark Git Commit Sha": common_data.get("benchmark_git_commit_sha", None),
        "Agent Git Commit Sha": common_data.get("agent_git_commit_sha", None),
        "Cost": test_info.get("metrics", {}).get("cost", ""),
        "Attempted": test_info.get("metrics", {}).get("attempted", ""),
        "Test Suite": test_suite,
        "Category": categories,
        "Task": test_info.get("task", ""),
        "Answer": test_info.get("answer", ""),
        "Description": test_info.get("description", ""),
        "Fail Reason": test_info.get("metrics", {}).get("fail_reason", ""),
        "Reached Cutoff": test_info.get("reached_cutoff", ""),
    }

    rows.append(row)

    # Check for nested tests and process them if present
    nested_tests = test_info.get("tests")
    if nested_tests:
        for nested_test_name, nested_test_info in nested_tests.items():
            process_test(nested_test_name, nested_test_info, agent_name, common_data)


```

This script appears to be processing benchmark reports for a software agent. It uses the `report_folder` structure to organize the test results, where each test is measured by a `report.json` file.

The script first checks if the `report_folder_path` is a directory. If it's not, it creates it. Then, it loops through each test in the `report_folder` and runs the `process_test` function for each test.

The `process_test` function takes as input the test name, test information, and the path to the agent directory. It then processes the test by calling itself with the appropriate arguments. The function may perform any necessary transformations or calculations on the test data before passing it to `process_test`.


```py
# Usage:


# Loop over each directory in the base directory
for agent_dir in os.listdir(base_dir):
    agent_dir_path = os.path.join(base_dir, agent_dir)

    # Ensure the agent_dir_path is a directory
    if os.path.isdir(agent_dir_path):
        # Loop over each sub-directory in the agent directory (e.g., "folder49_07-28-03-53")
        for report_folder in os.listdir(agent_dir_path):
            report_folder_path = os.path.join(agent_dir_path, report_folder)

            # Ensure the report_folder_path is a directory
            if os.path.isdir(report_folder_path):
                # Check for a file named "report.json" in the sub-directory
                report_path = os.path.join(report_folder_path, "report.json")

                if os.path.exists(report_path):
                    # Load the JSON data from the file
                    with open(report_path, "r") as f:
                        data = json.load(f)
                    benchmark_start_time = data.get("benchmark_start_time", "")

                    # Check if benchmark_start_time complies with the required format
                    pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")
                    if not pattern.fullmatch(benchmark_start_time):
                        continue  # Skip processing this report if the date is not in the correct format

                    # Parse the benchmark_start_time to a datetime object
                    benchmark_datetime = datetime.strptime(
                        benchmark_start_time, "%Y-%m-%dT%H:%M:%S+00:00"
                    )

                    # Check if benchmark_start_time is older than 3 days
                    current_datetime = datetime.utcnow()
                    if current_datetime - benchmark_datetime > timedelta(days=3):
                        continue  # Skip processing this report if it's more than 3 days old

                    # Loop through each test
                    for test_name, test_info in data["tests"].items():
                        process_test(test_name, test_info, agent_dir, data)

```

这段代码的作用是将一个包含行数据的列表（rows）转换成一个DataFrame数据框。首先，定义了一个scope变量，它包含两个URL，用于访问Google Sheets和Google Drive的服务。接着，定义了一个creds变量，它包含一个ServiceAccountCredentials对象，该对象的值从名为creds_info的JSON文件中读取。然后，使用gspread库从该ServiceAccountCredentials对象中获取授权。最后，将获取的授权与client变量进行关联，以便授权访问客户端sheet。


```py
# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows)

# Define the scope
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Add your service account credentials
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)

# Authorize the clientsheet
client = gspread.authorize(creds)

```

这段代码的作用是获取一个名为"benchmark-{branch_name}"的Google Sheets的实例，然后将该实例的第一个工作表的数据存储在变量`df.values`中。接下来，该数据被转换为列表形式，以便将其上传到Google Sheets。最后，第一个工作表被清空，以便在将来的第一次使用时能够重新开始。


```py
# Get the instance of the Spreadsheet
branch_name = os.getenv("GITHUB_REF_NAME")
sheet = client.open(f"benchmark-{branch_name}")

# Get the first sheet of the Spreadsheet
sheet_instance = sheet.get_worksheet(0)

# Convert dataframe to list of lists for uploading to Google Sheets
values = df.values.tolist()

# Prepend the header to the values list
values.insert(0, df.columns.tolist())

# Clear the existing values in the worksheet
sheet_instance.clear()

```

这段代码的主要作用是更新工作表（worksheet）中的数据，具体解释如下：

1. `sheet_instance`：获取并引用工作簿（workbook）实例。
2. `append_rows(values)`：将所传的参数`values`添加到工作表中。这个参数可以理解为一个数组，里面可以包含多个元素，每个元素可以是任何类型（如数值、文本等）。
3. `sheet_instance.append_rows(values)`：调用自定义函数`append_rows`，将`values`数组添加到工作表中。

总的来说，这段代码的作用是将一个包含新值的数组`values`添加到工作表中。


```py
# Update the worksheet with the new values
sheet_instance.append_rows(values)

```