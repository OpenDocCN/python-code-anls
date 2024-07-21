# `.\pytorch\test\conftest.py`

```
# 导入必要的模块和库
import copy  # 导入深拷贝模块
import functools  # 导入函数工具模块
import json  # 导入 JSON 处理模块
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import sys  # 导入系统模块
import xml.etree.ElementTree as ET  # 导入 XML 解析模块 ElementTree，并重命名为 ET
from collections import defaultdict  # 导入默认字典模块
from types import MethodType  # 导入方法类型模块
from typing import Any, List, Optional, TYPE_CHECKING, Union  # 导入类型检查相关模块

import pytest  # 导入 pytest 测试框架
from _pytest.config import Config, filename_arg  # 导入 pytest 配置模块中的 Config 类和 filename_arg 函数
from _pytest.config.argparsing import Parser  # 导入 pytest 配置模块中的 Parser 类
from _pytest.junitxml import _NodeReporter, bin_xml_escape, LogXML  # 导入 pytest junitxml 模块中的相关内容
from _pytest.python import Module  # 导入 pytest python 模块中的 Module 类
from _pytest.reports import TestReport  # 导入 pytest 报告模块中的 TestReport 类
from _pytest.stash import StashKey  # 导入 pytest 存储键模块中的 StashKey 类
from _pytest.terminal import _get_raw_skip_reason  # 导入 pytest 终端模块中的 _get_raw_skip_reason 函数
from pytest_shard_custom import pytest_addoptions as shard_addoptions, PytestShardPlugin  # 导入 pytest 自定义模块中的相关内容

if TYPE_CHECKING:
    from _pytest._code.code import ReprFileLocation  # 如果是类型检查阶段，则导入特定的代码位置模块

# 从 pytest-shard-custom 中获取并初始化 XML 日志重新运行的关键字存储对象
xml_key = StashKey["LogXMLReruns"]()
# 定义步骤当前缓存目录
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


def pytest_addoption(parser: Parser) -> None:
    # 获取 "general" 组并添加选项
    group = parser.getgroup("general")
    group.addoption(
        "--scs",
        action="store",
        default=None,
        dest="stepcurrent_skip",
    )
    group.addoption(
        "--sc",
        action="store",
        default=None,
        dest="stepcurrent",
    )

    # 添加 "--use-main-module" 选项
    parser.addoption("--use-main-module", action="store_true")

    # 获取 "terminal reporting" 组并添加选项
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-xml-reruns",
        action="store",
        dest="xmlpath_reruns",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-xml-reruns"),
        default=None,
        help="create junit-xml style report file at given path.",
    )
    group.addoption(
        "--junit-prefix-reruns",
        action="store",
        metavar="str",
        default=None,
        help="prepend prefix to classnames in junit-xml output",
    )

    # 添加初始化选项以配置 junit-xml 重新运行的测试套件名称
    parser.addini(
        "junit_suite_name_reruns", "Test suite name for JUnit report", default="pytest"
    )

    # 添加初始化选项以配置 junit-xml 重新运行的日志记录方式
    parser.addini(
        "junit_logging_reruns",
        "Write captured log messages to JUnit report: "
        "one of no|log|system-out|system-err|out-err|all",
        default="no",
    )

    # 添加初始化选项以配置 junit-xml 重新运行的传递测试日志信息
    parser.addini(
        "junit_log_passing_tests_reruns",
        "Capture log information for passing tests to JUnit report: ",
        type="bool",
        default=True,
    )

    # 添加初始化选项以配置 junit-xml 重新运行的测试持续时间报告方式
    parser.addini(
        "junit_duration_report_reruns",
        "Duration time to report: one of total|call",
        default="total",
    )

    # 添加初始化选项以配置 junit-xml 重新运行的 XML 输出模式
    parser.addini(
        "junit_family_reruns",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit2",
    )

    # 调用 pytest-shard-custom 中的添加选项方法，添加自定义选项
    shard_addoptions(parser)


def pytest_configure(config: Config) -> None:
    xmlpath = config.option.xmlpath_reruns
    # 配置阶段，如果存在 --junit-xml-reruns 选项，设置 XML 日志路径，避免在 worker 节点（xdist）上打开 XML 日志
    if xmlpath:
        # 防止在 worker 节点（xdist）上打开 XML 日志
        pass
    # 如果 xmlpath 不为空且 config 没有 "workerinput" 属性
    if xmlpath and not hasattr(config, "workerinput"):
        # 从配置中获取 junit_family_reruns 的值
        junit_family = config.getini("junit_family_reruns")
        # 将 LogXMLReruns 对象存储到 config.stash 中，键为 xml_key
        config.stash[xml_key] = LogXMLReruns(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name_reruns"),
            config.getini("junit_logging_reruns"),
            config.getini("junit_duration_report_reruns"),
            junit_family,
            config.getini("junit_log_passing_tests_reruns"),
        )
        # 注册 config.stash[xml_key] 到 pluginmanager 中
        config.pluginmanager.register(config.stash[xml_key])

    # 如果 config.getoption("stepcurrent_skip") 返回 True
    if config.getoption("stepcurrent_skip"):
        # 将 config.getoption("stepcurrent_skip") 的值赋给 config.option.stepcurrent
        config.option.stepcurrent = config.getoption("stepcurrent_skip")

    # 如果 config.getoption("stepcurrent") 返回 True
    if config.getoption("stepcurrent"):
        # 注册 StepcurrentPlugin(config) 到 pluginmanager 中，使用别名 "stepcurrentplugin"
        config.pluginmanager.register(StepcurrentPlugin(config), "stepcurrentplugin")

    # 如果 config.getoption("num_shards") 返回 True
    if config.getoption("num_shards"):
        # 注册 PytestShardPlugin(config) 到 pluginmanager 中，使用别名 "pytestshardplugin"
        config.pluginmanager.register(PytestShardPlugin(config), "pytestshardplugin")
def pytest_unconfigure(config: Config) -> None:
    # 从配置中获取 XML 数据，如果没有则为 None
    xml = config.stash.get(xml_key, None)
    # 如果存在 XML 数据
    if xml:
        # 从配置中删除 XML 数据
        del config.stash[xml_key]
        # 注销 XML 插件
        config.pluginmanager.unregister(xml)


class _NodeReporterReruns(_NodeReporter):
    def _prepare_content(self, content: str, header: str) -> str:
        # 准备报告内容，这里简单地返回传入的内容
        return content

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        # 如果内容为空，则直接返回
        if content == "":
            return
        # 创建一个 XML 元素标签
        tag = ET.Element(jheader)
        # 对内容进行 XML 转义并设置为标签的文本内容
        tag.text = bin_xml_escape(content)
        # 将标签添加到报告中
        self.append(tag)

    def append_skipped(self, report: TestReport) -> None:
        # 引用自以下链接
        # https://github.com/pytest-dev/pytest/blob/2178ee86d7c1ee93748cfb46540a6e40b4761f2d/src/_pytest/junitxml.py#L236C6-L236C6
        # 修改以转义 XML 不支持的字符在跳过原因中。其他内容应保持不变。
        
        # 如果报告有 wasxfail 属性
        if hasattr(report, "wasxfail"):
            # 调用父类方法以减少可能的差异
            super().append_skipped(report)
        else:
            # 断言报告的 longrepr 是一个元组
            assert isinstance(report.longrepr, tuple)
            # 解构元组获取文件名、行号和跳过原因
            filename, lineno, skipreason = report.longrepr
            # 如果跳过原因以 "Skipped: " 开头，则去除这部分前缀
            if skipreason.startswith("Skipped: "):
                skipreason = skipreason[9:]
            # 构建详细信息字符串
            details = f"{filename}:{lineno}: {skipreason}"

            # 创建一个 skipped 的 XML 元素，设置类型和消息，并对消息进行 XML 转义
            skipped = ET.Element(
                "skipped", type="pytest.skip", message=bin_xml_escape(skipreason)
            )
            # 将详细信息进行 XML 转义并设置为 skipped 元素的文本内容
            skipped.text = bin_xml_escape(details)
            # 将 skipped 元素添加到报告中
            self.append(skipped)
            # 写入捕获的输出
            self.write_captured_output(report)


class LogXMLReruns(LogXML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_rerun(self, reporter: _NodeReporter, report: TestReport) -> None:
        # 如果报告有 wasxfail 属性
        if hasattr(report, "wasxfail"):
            # 添加一个简单的 "skipped" 标签，说明是 xfail 标记的测试意外通过
            reporter._add_simple("skipped", "xfail-marked test passes unexpectedly")
        else:
            # 断言报告的 longrepr 不为空
            assert report.longrepr is not None
            # 获取报告的 reprcrash 属性，这里是可选的文件位置表示
            reprcrash: Optional[ReprFileLocation] = getattr(
                report.longrepr, "reprcrash", None
            )
            # 如果存在 reprcrash 属性，则使用其消息作为消息内容
            if reprcrash is not None:
                message = reprcrash.message
            else:
                # 否则，将 longrepr 转换为字符串作为消息内容
                message = str(report.longrepr)
            # 对消息内容进行 XML 转义
            message = bin_xml_escape(message)
            # 添加一个 "rerun" 标签，消息内容和报告的 longrepr 作为子元素
            reporter._add_simple("rerun", message, str(report.longrepr))

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        # 调用父类方法处理报告
        super().pytest_runtest_logreport(report)
        # 如果报告的结果为 "rerun"
        if report.outcome == "rerun":
            # 打开测试用例的报告
            reporter = self._opentestcase(report)
            # 添加 rerun 信息到报告中
            self.append_rerun(reporter, report)
        # 如果报告的结果为 "skipped"
        if report.outcome == "skipped":
            # 如果 longrepr 是元组形式
            if isinstance(report.longrepr, tuple):
                # 获取文件路径、行号和原因，构建新的跳过原因
                fspath, lineno, reason = report.longrepr
                reason = f"{report.nodeid}: {_get_raw_skip_reason(report)}"
                report.longrepr = (fspath, lineno, reason)
    # 定义一个方法 `node_reporter`，接受一个参数 `report`，类型可以是 `TestReport` 或者 `str`，返回 `_NodeReporterReruns` 类型的对象
    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporterReruns:
        # 获取 report 的 nodeid 属性，如果 report 是 TestReport 对象，则直接获取 nodeid，否则 nodeid 就是 report 本身
        nodeid: Union[str, TestReport] = getattr(report, "nodeid", report)
        
        # 获取 report 的 node 属性，通常用于处理 xdist 的报告顺序
        workernode = getattr(report, "node", None)
        
        # 组成一个元组作为字典的键，用于唯一标识一个报告节点
        key = nodeid, workernode
        
        # 如果 key 已经在 self.node_reporters 中存在，则直接返回对应的 _NodeReporterReruns 对象
        if key in self.node_reporters:
            # TODO: breaks for --dist=each
            return self.node_reporters[key]
        
        # 如果 key 不存在，则创建一个新的 _NodeReporterReruns 对象
        reporter = _NodeReporterReruns(nodeid, self)
        
        # 将新创建的 reporter 存储到 self.node_reporters 字典中，以 key 作为索引
        self.node_reporters[key] = reporter
        
        # 将 reporter 添加到 self.node_reporters_ordered 列表中，用于保持报告的顺序
        self.node_reporters_ordered.append(reporter)
        
        # 返回新创建的 reporter 对象
        return reporter
# 模仿 pytest 的 terminal.py 中的 summary_failures 函数
# 使用 hookwrapper 和 tryfirst 确保此函数在 pytest 之前运行
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # 如果配置中指定不使用堆栈跟踪样式，则不打印堆栈跟踪
    if terminalreporter.config.option.tbstyle != "no":
        # 获取所有 "rerun" 类型的测试报告
        reports = terminalreporter.getreports("rerun")
        if reports:
            # 在输出中添加分隔线和 "RERUNS" 标题
            terminalreporter.write_sep("=", "RERUNS")
            # 根据配置中的堆栈跟踪样式输出相应的信息
            if terminalreporter.config.option.tbstyle == "line":
                for rep in reports:
                    line = terminalreporter._getcrashline(rep)
                    terminalreporter.write_line(line)
            else:
                for rep in reports:
                    msg = terminalreporter._getfailureheadline(rep)
                    # 输出带有红色和加粗效果的报告标题
                    terminalreporter.write_sep("_", msg, red=True, bold=True)
                    # 输出报告的总结信息
                    terminalreporter._outrep_summary(rep)
                    # 处理测试结束后的清理部分
                    terminalreporter._handle_teardown_sections(rep.nodeid)
    yield


# 使用 tryfirst 确保此钩子在 pytest 之前运行
@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, path, parent) -> Module:
    # 如果配置中有指定 "--use-main-module" 选项，则使用主模块
    if parent.config.getoption("--use-main-module"):
        # 创建一个模块对象
        mod = Module.from_parent(parent, path=module_path)
        # 将模块对象的 _getobj 方法设置为 lambda 表达式，用于获取 "__main__" 模块
        mod._getobj = MethodType(lambda x: sys.modules["__main__"], mod)
        return mod


# 使用 hookwrapper 包装此钩子函数
@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    # 将结果传递给 pluggy_result
    pluggy_result = yield
    # 如果报告不是 pytest.TestReport 类型，则直接返回
    if not isinstance(report, pytest.TestReport):
        return
    # 从 pluggy_result 中获取结果信息
    outcome, letter, verbose = pluggy_result.get_result()
    # 如果需要详细输出
    if verbose:
        # 强制设置结果，包括测试执行时间
        pluggy_result.force_result(
            (outcome, letter, f"{verbose} [{report.duration:.4f}s]")
        )


# 使用 trylast 确保此钩子在 pytest 之后运行
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: List[Any]) -> None:
    """
    当重新运行禁用的测试时使用此钩子函数，以便在收集测试时清除所有跳过的测试，
    而不是多次运行和跳过它们。这可以避免在控制台和 XML 输出中生成大量无用信息。
    因此，我们希望在收集测试时最后运行此函数。
    """
    # 检查环境变量以确定是否重新运行禁用的测试
    rerun_disabled_tests = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
    if not rerun_disabled_tests:
        return

    # 编译用于匹配禁用测试的正则表达式
    disabled_regex = re.compile(r"(?P<test_name>.+)\s+\([^\.]+\.(?P<test_class>.+)\)")
    disabled_tests = defaultdict(set)

    # 检查环境变量和文件是否已设置
    disabled_tests_file = os.getenv("DISABLED_TESTS_FILE", "")
    if not disabled_tests_file or not os.path.exists(disabled_tests_file):
        return

    # 从文件中加载禁用的测试信息并进行处理
    with open(disabled_tests_file) as fp:
        for disabled_test in json.load(fp):
            m = disabled_regex.match(disabled_test)
            if m:
                test_name = m["test_name"]
                test_class = m["test_class"]
                disabled_tests[test_class].add(test_name)
    # 创建一个空列表，用于存储筛选后的测试项
    filtered_items = []

    # 遍历所有测试项
    for item in items:
        # 获取当前测试项的名称
        test_name = item.name
        # 获取当前测试项所属的测试类名称
        test_class = item.parent.name

        # 如果当前测试类不在禁用测试集合中，或者当前测试项不在该测试类的禁用测试列表中，则跳过该测试项
        if (
            test_class not in disabled_tests
            or test_name not in disabled_tests[test_class]
        ):
            continue

        # 复制当前测试项，以便对副本进行修改而不影响原始测试项
        cpy = copy.copy(item)
        # 初始化复制后的测试项的请求状态
        cpy._initrequest()

        # 将复制后的测试项添加到筛选后的列表中
        filtered_items.append(cpy)

    # 清空原始的测试项列表
    items.clear()
    # 将筛选后的测试项列表重新添加到原始的测试项列表中
    # 注意：这里直接编辑 items 列表，以便将更改反映回 pytest
    items.extend(filtered_items)
# 定义 StepcurrentPlugin 类，用于实现保存当前运行测试而非上次失败测试的功能
class StepcurrentPlugin:
    # 初始化方法，接受一个 Config 对象作为参数
    def __init__(self, config: Config) -> None:
        self.config = config
        self.report_status = ""  # 初始化报告状态为空字符串
        assert config.cache is not None  # 断言确保 config.cache 不为空
        self.cache: pytest.Cache = config.cache  # 将 config.cache 赋值给 self.cache
        # 设置缓存目录为 STEPCURRENT_CACHE_DIR 下的一个子目录，子目录名由 config.getoption('stepcurrent') 提供
        self.directory = f"{STEPCURRENT_CACHE_DIR}/{config.getoption('stepcurrent')}"
        # 从缓存中获取上次运行的测试节点 ID，如果没有则为 None
        self.lastrun: Optional[str] = self.cache.get(self.directory, None)
        self.initial_val = self.lastrun  # 将初始值设为上次运行的测试节点 ID
        self.skip: bool = config.getoption("stepcurrent_skip")  # 根据配置获取是否跳过的设置

    # pytest 集合修改钩子，修改测试集合中的项目
    def pytest_collection_modifyitems(self, config: Config, items: List[Any]) -> None:
        # 如果没有找到上次运行的测试节点 ID，则设置报告状态为 "Cannot find last run test, not skipping" 并返回
        if not self.lastrun:
            self.report_status = "Cannot find last run test, not skipping"
            return
        
        # 检查所有项目节点，直到找到与上次运行的节点 ID 匹配的项目
        failed_index = None
        for index, item in enumerate(items):
            if item.nodeid == self.lastrun:
                failed_index = index
                # 如果需要跳过，则将 failed_index 增加 1
                if self.skip:
                    failed_index += 1
                break
        
        # 如果未找到之前失败的测试项目，则设置报告状态为 "previously run test not found, not skipping."
        if failed_index is None:
            self.report_status = "previously run test not found, not skipping."
        else:
            # 否则设置报告状态为 "skipping {failed_index} already run items."
            self.report_status = f"skipping {failed_index} already run items."
            # 将已选择的项目从 items 列表中移除，并通知 pytest_deselected 钩子
            deselected = items[:failed_index]
            del items[:failed_index]
            config.hook.pytest_deselected(items=deselected)

    # pytest 集合完成报告钩子
    def pytest_report_collectionfinish(self) -> Optional[str]:
        # 如果 verbose 大于等于 0 并且 report_status 不为空，则返回带有报告状态的字符串
        if self.config.getoption("verbose") >= 0 and self.report_status:
            return f"stepcurrent: {self.report_status}"
        return None

    # pytest 运行测试协议钩子，记录最后运行的测试节点 ID 并将其保存到缓存中
    def pytest_runtest_protocol(self, item, nextitem) -> None:
        self.lastrun = item.nodeid
        self.cache.set(self.directory, self.lastrun)

    # pytest 会话结束钩子，如果退出状态为 0，则将初始值重新保存到缓存中
    def pytest_sessionfinish(self, session, exitstatus):
        if exitstatus == 0:
            self.cache.set(self.directory, self.initial_val)
```