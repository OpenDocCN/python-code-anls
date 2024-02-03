# `.\AutoGPT\benchmark\agbenchmark\utils\dependencies\main.py`

```py
"""
A module to manage dependencies between pytest tests.

This module provides the methods implementing the main logic. These are used in the pytest hooks that are in
__init__.py.
"""

# 导入必要的模块
import collections
import json
import os
from typing import Any, Generator

import colorama
import networkx
from _pytest.nodes import Item

# 导入自定义常量
from .constants import MARKER_KWARG_DEPENDENCIES, MARKER_NAME
# 导入自定义模块
from .graphs import graph_interactive_network
from .util import clean_nodeid, get_absolute_nodeid, get_markers, get_name

# 定义一个类来跟踪单个测试的结果
class TestResult(object):
    """Keeps track of the results of a single test."""

    # 定义测试的步骤和成功的结果
    STEPS = ["setup", "call", "teardown"]
    GOOD_OUTCOMES = ["passed"]

    def __init__(self, nodeid: str) -> None:
        """Create a new instance for a test with a given node id."""
        # 初始化测试结果实例
        self.nodeid = nodeid
        self.results: dict[str, Any] = {}

    def register_result(self, result: Any) -> None:
        """Register a result of this test."""
        # 注册测试结果
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
        # 检查测试是否完全成功
        return all(
            self.results.get(step, None) in self.GOOD_OUTCOMES for step in self.STEPS
        )

# 定义一个类来存储单个测试的解析依赖信息
class TestDependencies(object):
    """Information about the resolved dependencies of a single test."""
    def __init__(self, item: Item, manager: "DependencyManager") -> None:
        """初始化函数，为给定的测试创建一个新实例。"""
        # 清理节点 ID，将其赋值给实例的节点 ID
        self.nodeid = clean_nodeid(item.nodeid)
        # 初始化依赖和未解决依赖的集合
        self.dependencies = set()
        self.unresolved = set()

        # 获取测试项中的标记
        markers = get_markers(item, MARKER_NAME)
        # 从标记中提取依赖项
        dependencies = [
            dep
            for marker in markers
            for dep in marker.kwargs.get(MARKER_KWARG_DEPENDENCIES, [])
        ]
        # 遍历依赖项列表
        for dependency in dependencies:
            # 如果依赖项的名称未知，则尝试将其转换为绝对路径形式（例如 file::[class::]method）
            if dependency not in manager.name_to_nodeids:
                absolute_dependency = get_absolute_nodeid(dependency, self.nodeid)
                # 如果转换后的绝对路径在管理器中存在，则更新依赖项
                if absolute_dependency in manager.name_to_nodeids:
                    dependency = absolute_dependency

            # 添加所有匹配名称的项目
            if dependency in manager.name_to_nodeids:
                for nodeid in manager.name_to_nodeids[dependency]:
                    self.dependencies.add(nodeid)
            else:
                # 将未解决的依赖项添加到未解决集合中
                self.unresolved.add(dependency)
class DependencyManager(object):
    """Keep track of tests, their names and their dependencies."""

    def __init__(self) -> None:
        """Create a new DependencyManager."""
        # 初始化 DependencyManager 类
        self.options: dict[str, Any] = {}
        self._items: list[Item] | None = None
        self._name_to_nodeids: Any = None
        self._nodeid_to_item: Any = None
        self._results: Any = None

    @property
    def items(self) -> list[Item]:
        """The collected tests that are managed by this instance."""
        # 返回由该实例管理的收集的测试
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
            # 将 nodeid 映射到测试项
            self._nodeid_to_item[nodeid] = item
            # 将所有名称映射到 node id
            name = get_name(item)
            self._name_to_nodeids[name].append(nodeid)
            # 创建包含此测试结果的对象
            self._results[nodeid] = TestResult(clean_nodeid(item.nodeid))

        # 不允许在 name_to_nodeids 映射中使用未知键
        self._name_to_nodeids.default_factory = None

        for item in items:
            nodeid = clean_nodeid(item.nodeid)
            # 处理此测试的依赖关系
            # 这使用了前面循环中创建的映射，因此无法合并到该循环中
            self._dependencies[nodeid] = TestDependencies(item, self)

    @property
    def name_to_nodeids(self) -> dict[str, list[str]]:
        """返回名称到匹配节点ID(s)的映射。"""
        # 断言 self.items 不为空
        assert self.items is not None
        # 返回名称到节点ID(s)的映射
        return self._name_to_nodeids

    @property
    def nodeid_to_item(self) -> dict[str, Item]:
        """返回节点ID到测试项的映射。"""
        # 断言 self.items 不为空
        assert self.items is not None
        # 返回节点ID到测试项的映射
        return self._nodeid_to_item

    @property
    def results(self) -> dict[str, TestResult]:
        """测试结果的映射。"""
        # 断言 self.items 不为空
        assert self.items is not None
        # 返回测试结果的映射
        return self._results

    @property
    def dependencies(self) -> dict[str, TestDependencies]:
        """测试的依赖关系。"""
        # 断言 self.items 不为空
        assert self.items is not None
        # 返回测试的依赖关系
        return self._dependencies

    def print_name_map(self, verbose: bool = False) -> None:
        """打印名称到测试映射的人类可读版本。"""
        # 打印可用的依赖名称
        print("Available dependency names:")
        # 遍历名称到节点ID(s)的映射，按名称排序
        for name, nodeids in sorted(self.name_to_nodeids.items(), key=lambda x: x[0]):
            if len(nodeids) == 1:
                if name == nodeids[0]:
                    # 这只是基本名称，只有在 verbose 为真时才打印
                    if verbose:
                        print(f"  {name}")
                else:
                    # 名称指向单个节点ID，因此使用短格式
                    print(f"  {name} -> {nodeids[0]}")
            else:
                # 名称指向多个节点ID，因此使用长格式
                print(f"  {name} ->")
                for nodeid in sorted(nodeids):
                    print(f"    {nodeid}")
    # 定义一个方法，用于打印已处理的依赖关系的人类可读列表
    def print_processed_dependencies(self, colors: bool = False) -> None:
        """Print a human-readable list of the processed dependencies."""
        # 定义一个字符串常量，表示缺失的依赖
        missing = "MISSING"
        # 如果启用了颜色选项，则将缺失的依赖文本设置为红色
        if colors:
            missing = f"{colorama.Fore.RED}{missing}{colorama.Fore.RESET}"
            # 初始化 colorama 模块，用于在终端中显示颜色
            colorama.init()
        try:
            # 打印标题
            print("Dependencies:")
            # 遍历已处理的依赖关系字典，按键排序
            for nodeid, info in sorted(self.dependencies.items(), key=lambda x: x[0]):
                descriptions = []
                # 遍历依赖关系中的依赖项，添加到描述列表中
                for dependency in info.dependencies:
                    descriptions.append(dependency)
                # 遍历未解决的依赖项，将其添加到描述列表中，并标记为缺失
                for dependency in info.unresolved:
                    descriptions.append(f"{dependency} ({missing})")
                # 如果存在描述列表
                if descriptions:
                    # 打印节点 ID 和依赖关系
                    print(f"  {nodeid} depends on")
                    # 遍历并打印描述列表中的内容
                    for description in sorted(descriptions):
                        print(f"    {description}")
        finally:
            # 如果启用了颜色选项，则关闭 colorama 模块
            if colors:
                colorama.deinit()

    # 定义一个属性
    @property
    # 定义一个生成器函数，返回一个按照测试依赖关系排序后的测试列表
    def sorted_items(self) -> Generator:
        """Get a sorted list of tests where all tests are sorted after their dependencies."""
        # 根据环境变量 BUILD_SKILL_TREE 的值来确定是否构建技能树
        build_skill_tree = os.getenv("BUILD_SKILL_TREE")
        BUILD_SKILL_TREE = (
            build_skill_tree.lower() == "true" if build_skill_tree else False
        )
        # 创建一个有向图用于排序
        dag = networkx.DiGraph()

        # 将所有测试项作为节点插入，以防止没有依赖关系且不是依赖项的测试项丢失
        dag.add_nodes_from(self.items)

        # 为所有依赖关系插入边
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

        # 只有在环境变量中指定构建技能树且是整个运行时才构建树
        if BUILD_SKILL_TREE:
            # 调用函数构建交互式网络图
            graph_interactive_network(dag, labels, html_graph_path="")

        # 根据依赖关系排序
        return networkx.topological_sort(dag)

    # 注册测试结果
    def register_result(self, item: Item, result: Any) -> None:
        """Register a result of a test."""
        nodeid = clean_nodeid(item.nodeid)
        self.results[nodeid].register_result(result)
    # 获取测试项的未满足依赖项列表
    def get_failed(self, item: Item) -> Any:
        """Get a list of unfulfilled dependencies for a test."""
        # 清理测试项的节点ID
        nodeid = clean_nodeid(item.nodeid)
        # 初始化一个空列表用于存放未满足的依赖项
        failed = []
        # 遍历测试项的依赖项
        for dependency in self.dependencies[nodeid].dependencies:
            # 获取依赖项的执行结果
            result = self.results[dependency]
            # 如果依赖项执行结果不成功，则将其添加到未满足依赖项列表中
            if not result.success:
                failed.append(dependency)
        # 返回未满足依赖项列表
        return failed

    # 获取测试项的缺失依赖项列表
    def get_missing(self, item: Item) -> Any:
        """Get a list of missing dependencies for a test."""
        # 清理测试项的节点ID
        nodeid = clean_nodeid(item.nodeid)
        # 返回测试项的未解决依赖项列表
        return self.dependencies[nodeid].unresolved
```