# `D:\src\scipysrc\matplotlib\doc\sphinxext\missing_references.py`

```
"""
Thispython
"""
This is a sphinx extension to freeze your broken reference problems
when using ``nitpicky = True``.

The basic operation is:

1. Add this extension to your ``conf.py`` extensions.
2. Add ``missing_references_write_json = True`` to your ``conf.py``
3. Run sphinx-build. It will generate ``missing-references.json``
    next to your ``conf.py``.
4. Remove ``missing_references_write_json = True`` from your
    ``conf.py`` (or set it to ``False``)
5. Run sphinx-build again, and ``nitpick_ignore`` will
    contain all of the previously failed references.

"""

from collections import defaultdict  # Importing defaultdict for efficient handling of missing references data
import json  # Importing json for JSON serialization of missing references
import logging  # Importing logging for capturing and handling log messages
from pathlib import Path  # Importing Path for handling filesystem paths

from docutils.utils import get_source_line  # Importing get_source_line for obtaining source code line information
from docutils import nodes  # Importing nodes for representing structured document elements
from sphinx.util import logging as sphinx_logging  # Importing sphinx_logging for Sphinx-specific logging utilities

import matplotlib  # Importing matplotlib for plotting capabilities

logger = sphinx_logging.getLogger(__name__)  # Initializing logger for this module


class MissingReferenceFilter(logging.Filter):
    """
    A logging filter designed to record missing reference warning messages
    for use by this extension
    """
    def __init__(self, app):
        self.app = app
        super().__init__()

    def _record_reference(self, record):
        if not (getattr(record, 'type', '') == 'ref' and
                isinstance(getattr(record, 'location', None), nodes.Node)):
            return

        if not hasattr(self.app.env, "missing_references_warnings"):
            self.app.env.missing_references_warnings = defaultdict(set)

        record_missing_reference(self.app,
                                 self.app.env.missing_references_warnings,
                                 record.location)

    def filter(self, record):
        self._record_reference(record)
        return True


def record_missing_reference(app, record, node):
    """
    Record missing reference details based on the provided node information.

    :param app: Sphinx application object
    :param record: Data structure to record missing references
    :param node: Node representing the missing reference
    """
    domain = node["refdomain"]
    typ = node["reftype"]
    target = node["reftarget"]
    location = get_location(node, app)

    domain_type = f"{domain}:{typ}"

    record[(domain_type, target)].add(location)


def record_missing_reference_handler(app, env, node, contnode):
    """
    Handler function to record missing references during Sphinx build.

    :param app: Sphinx application object
    :param env: Sphinx environment object
    :param node: Node representing the missing reference
    :param contnode: Containing node
    """
    if not app.config.missing_references_enabled:
        # no-op when we are disabled.
        return

    if not hasattr(env, "missing_references_events"):
        env.missing_references_events = defaultdict(set)

    record_missing_reference(app, env.missing_references_events, node)


def get_location(node, app):
    """
    Retrieve the source location of a node within the Sphinx application.

    :param node: Node representing a document element
    :param app: Sphinx application object
    :return: String representation of the source location
    """
    # Implementation details for obtaining source location
    # Usually returns "path/to/file:linenumber", "<external>", or "<unknown>"
    pass  # Placeholder for actual implementation
    """
    返回与给定节点相关的源文件路径和行号的字符串表示。

    Parameters:
        node (object): 表示代码文档节点的对象。

    Returns:
        str: 表示源文件路径、文档字符串（如果有）、行号的字符串。

    Notes:
        如果无法定位原始源文件（通常是因为扩展向 sphinx 解析引擎注入了文本），
        则返回一个默认的未知路径和空行号的字符串。
    """
    # 获取给定节点的源文件路径和行号
    source, line = get_source_line(node)

    if source:
        # 'source' 可能形如 '/some/path:docstring of some.api'，但是冒号在 Windows 下是被禁止的，
        # 在 POSIX 系统中可以通过，这里处理这种情况
        if ':docstring of' in source:
            path, *post = source.rpartition(':docstring of')
            post = ''.join(post)
        else:
            path = source
            post = ''
        # 我们相对于文档目录的父目录定位引用，对于 matplotlib，这将是 matplotlib 仓库的根目录。
        # 当 matplotlib 不是可编辑安装时，会出现奇怪的情况，但我们无法完全从中恢复。
        basepath = Path(app.srcdir).parent.resolve()

        fullpath = Path(path).resolve()

        try:
            # 尝试获取相对于基本路径的相对路径
            path = fullpath.relative_to(basepath)
        except ValueError:
            # 有时文档直接包含来自已安装模块的文档字符串，我们将其记录为 '<external>'，
            # 以便与模块安装位置无关。
            path = Path("<external>") / fullpath.name

        # 确保所有报告的路径都是 POSIX 格式，以确保在 Windows 上生成的文档中生成相同的警告。
        path = path.as_posix()

    else:
        # 如果未找到源文件，则将路径设置为 "<unknown>"
        path = "<unknown>"
        post = ''
    if not line:
        line = ""

    # 返回格式化后的表示路径、文档字符串（如果有）、行号的字符串
    return f"{path}{post}:{line}"
# 截取位置字符串中第一个冒号之前的部分，用于简化位置比较
def _truncate_location(location):
    return location.split(":", 1)[0]


# 根据应用程序配置检查是否应发出未使用的或缺失引用警告
def _warn_unused_missing_references(app):
    # 如果未设置忽略警告标志，则直接返回
    if not app.config.missing_references_warn_unused_ignores:
        return

    # 获取 matplotlib 库安装路径的父目录的父目录的解析路径
    basepath = Path(matplotlib.__file__).parent.parent.parent.resolve()
    # 获取应用程序源目录的解析路径
    srcpath = Path(app.srcdir).parent.resolve()

    # 如果 matplotlib 库安装路径与应用程序源目录不相等，则直接返回
    if basepath != srcpath:
        return

    # 获取环境中被忽略的引用字典 {(domain_type, target): locations}
    references_ignored = getattr(
        app.env, 'missing_references_ignored_references', {})
    # 获取环境中的引用事件字典 {(domain_type, target): locations}
    references_events = getattr(app.env, 'missing_references_events', {})

    # 遍历被忽略的引用字典
    for (domain_type, target), locations in references_ignored.items():
        # 获取引用事件中缺失引用的位置列表，仅截取位置字符串的第一个冒号之前的部分
        missing_reference_locations = [
            _truncate_location(location)
            for location in references_events.get((domain_type, target), [])]

        # 遍历被忽略的引用的位置列表
        for ignored_reference_location in locations:
            # 获取被忽略引用位置的截断形式
            short_location = _truncate_location(ignored_reference_location)
            # 如果被忽略引用位置的截断形式不在缺失引用的位置列表中，则发出警告
            if short_location not in missing_reference_locations:
                msg = (f"Reference {domain_type} {target} for "
                       f"{ignored_reference_location} can be removed"
                       f" from {app.config.missing_references_filename}."
                       " It is no longer a missing reference in the docs.")
                # 记录警告信息，包括位置和类型
                logger.warning(msg,
                               location=ignored_reference_location,
                               type='ref',
                               subtype=domain_type)


# 在 sphinx 构建结束时调用，检查现有 JSON 文件中所有行是否仍然必要
# 如果配置值 missing_references_write_json 设置为 True，则将缺失引用写入新的 JSON 文件中
def save_missing_references_handler(app, exc):
    # 如果未启用缺失引用功能，则直接返回
    if not app.config.missing_references_enabled:
        return

    # 发出未使用或缺失引用的警告
    _warn_unused_missing_references(app)

    # 构造 JSON 文件的路径
    json_path = Path(app.confdir) / app.config.missing_references_filename

    # 获取环境中的缺失引用警告字典
    references_warnings = getattr(app.env, 'missing_references_warnings', {})

    # 如果配置允许写入 JSON，则将缺失引用写入指定的 JSON 文件中
    if app.config.missing_references_write_json:
        _write_missing_references_json(references_warnings, json_path)


# 将被忽略的引用转换为 JSON 可序列化的格式
# 从 {(domain_type, target): locations} 转换为 {domain_type: {target: locations}}
def _write_missing_references_json(records, json_path):
    pass
    # 对记录和键进行排序可以减少在重新生成 missing_references.json 时出现的不必要的大型差异。
    transformed_records = defaultdict(dict)
    
    # 遍历 records 中的条目，按指定的顺序进行处理和排序
    for (domain_type, target), paths in records.items():
        # 将路径列表按字母顺序排序，并存储到 transformed_records 字典中的适当位置
        transformed_records[domain_type][target] = sorted(paths)
    
    # 打开 json_path 文件以供写入，准备将 transformed_records 写入其中
    with json_path.open("w") as stream:
        # 将 transformed_records 写入到 stream（文件流）中，按键进行排序，并且每行缩进2个空格
        json.dump(transformed_records, stream, sort_keys=True, indent=2)
def _read_missing_references_json(json_path):
    """
    Convert from the JSON file to the form used internally by this
    extension.

    The JSON file is stored as ``{domain_type: {target: [locations,]}}``
    since JSON can't store dictionary keys which are tuples. We convert
    this back to ``{(domain_type, target):[locations]}`` for internal use.

    """
    # 打开指定路径的 JSON 文件并加载数据
    with json_path.open("r") as stream:
        data = json.load(stream)

    # 初始化一个空字典用于存储被忽略的引用
    ignored_references = {}
    # 遍历 JSON 数据中的每一个域类型及其对应的目标和位置
    for domain_type, targets in data.items():
        for target, locations in targets.items():
            # 将域类型和目标组合为元组作为字典的键，位置作为值
            ignored_references[(domain_type, target)] = locations
    # 返回整理后的被忽略引用字典
    return ignored_references


def prepare_missing_references_handler(app):
    """
    Handler called to initialize this extension once the configuration
    is ready.

    Reads the missing references file and populates ``nitpick_ignore`` if
    appropriate.
    """
    # 如果未启用缺失引用功能，则直接返回
    if not app.config.missing_references_enabled:
        return

    # 获取 Sphinx 日志记录器
    sphinx_logger = logging.getLogger('sphinx')
    # 创建一个缺失引用过滤器对象
    missing_reference_filter = MissingReferenceFilter(app)
    # 遍历日志处理器，找到适合的警告流处理器并添加过滤器
    for handler in sphinx_logger.handlers:
        if (isinstance(handler, sphinx_logging.WarningStreamHandler)
                and missing_reference_filter not in handler.filters):

            # 这必须是第一个过滤器，因为后续的过滤器会丢弃节点信息，导致无法唯一标识引用
            handler.filters.insert(0, missing_reference_filter)

    # 初始化一个空字典，用于存储被忽略的引用
    app.env.missing_references_ignored_references = {}

    # 构建 JSON 文件的路径
    json_path = Path(app.confdir) / app.config.missing_references_filename
    # 如果 JSON 文件不存在，则直接返回
    if not json_path.exists():
        return

    # 读取 JSON 文件中的被忽略引用数据
    ignored_references = _read_missing_references_json(json_path)

    # 将读取的被忽略引用数据存储到应用环境变量中
    app.env.missing_references_ignored_references = ignored_references

    # 如果不需要写入 JSON 文件，则将所有已知的被忽略引用添加到 nitpick_ignore 中
    if not app.config.missing_references_write_json:
        # 由于 Sphinx v6.2 中 nitpick_ignore 可能是列表、集合或元组，默认为集合。为了版本一致性，将其转换为列表。
        app.config.nitpick_ignore = list(app.config.nitpick_ignore)
        # 扩展 nitpick_ignore，添加所有被忽略引用的键
        app.config.nitpick_ignore.extend(ignored_references.keys())


def setup(app):
    app.add_config_value("missing_references_enabled", True, "env")
    app.add_config_value("missing_references_write_json", False, "env")
    app.add_config_value("missing_references_warn_unused_ignores", True, "env")
    app.add_config_value("missing_references_filename",
                         "missing-references.json", "env")

    # 在 builder-inited 事件发生时调用 prepare_missing_references_handler 处理函数
    app.connect("builder-inited", prepare_missing_references_handler)
    app.connect("missing-reference", record_missing_reference_handler)
    # 将名为 "build-finished" 的信号与函数 save_missing_references_handler 绑定
    app.connect("build-finished", save_missing_references_handler)

    # 返回一个字典，表明该函数的并行读取安全性为 True
    return {'parallel_read_safe': True}
```