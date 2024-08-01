# `.\DB-GPT-src\dbgpt\util\tracer\tracer_cli.py`

```py
# 导入必要的库
import glob  # 文件路径模式匹配
import json  # JSON 数据处理
import logging  # 日志记录
import os  # 操作系统功能
from datetime import datetime  # 处理日期时间
from typing import Callable, Dict, Iterable  # 类型提示

import click  # 命令行参数解析库

from dbgpt.configs.model_config import LOGDIR  # 导入日志文件目录
from dbgpt.util.tracer import SpanType, SpanTypeRunName  # 导入跟踪相关功能

# 设置日志记录器
logger = logging.getLogger("dbgpt_cli")

# 定义默认的日志文件模式
_DEFAULT_FILE_PATTERN = os.path.join(LOGDIR, "dbgpt*.jsonl")

# 定义命令组 'trace'
@click.group("trace")
def trace_cli_group():
    """Analyze and visualize trace spans."""
    pass

# 定义 'list' 命令
@trace_cli_group.command()
@click.option(
    "--trace_id",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Specify the trace ID to list",
)
@click.option(
    "--span_id",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Specify the Span ID to list.",
)
@click.option(
    "--span_type",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Specify the Span Type to list.",
)
@click.option(
    "--parent_span_id",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Specify the Parent Span ID to list.",
)
@click.option(
    "--search",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Search trace_id, span_id, parent_span_id, operation_name or content in metadata.",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    default=20,
    help="Limit the number of recent span displayed.",
)
@click.option(
    "--start_time",
    type=str,
    help='Filter by start time. Format: "YYYY-MM-DD HH:MM:SS.mmm"',
)
@click.option(
    "--end_time", type=str, help='Filter by end time. Format: "YYYY-MM-DD HH:MM:SS.mmm"'
)
@click.option(
    "--desc",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Whether to use reverse sorting. By default, sorting is based on start time.",
)
@click.option(
    "--output",
    required=False,
    type=click.Choice(["text", "html", "csv", "latex", "json"]),
    default="text",
    help="The output format",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True, readable=True))
def list(
    trace_id: str,
    span_id: str,
    span_type: str,
    parent_span_id: str,
    search: str,
    limit: int,
    start_time: str,
    end_time: str,
    desc: bool,
    output: str,
    files=None,
):
    """List your trace spans"""
    from prettytable import PrettyTable  # 导入用于美化输出的 PrettyTable 库

    # 如果未显式指定文件，则使用默认模式来获取文件列表
    spans = read_spans_from_files(files)

    if trace_id:
        # 根据 trace_id 过滤 spans
        spans = filter(lambda s: s["trace_id"] == trace_id, spans)
    if span_id:
        # 根据 span_id 过滤 spans
        spans = filter(lambda s: s["span_id"] == span_id, spans)
    if span_type:
        # 根据 span_type 过滤 spans
        spans = filter(lambda s: s["span_type"] == span_type, spans)
    if parent_span_id:
        # 根据 parent_span_id 过滤 spans
        spans = filter(lambda s: s["parent_span_id"] == parent_span_id, spans)
    # 根据开始和结束时间过滤 spans
    # 如果指定了起始时间，则解析起始时间为 datetime 对象
    if start_time:
        start_dt = _parse_datetime(start_time)
        # 过滤 spans，只保留起始时间大于等于 start_dt 的记录
        spans = filter(
            lambda span: _parse_datetime(span["start_time"]) >= start_dt, spans
        )

    # 如果指定了结束时间，则解析结束时间为 datetime 对象
    if end_time:
        end_dt = _parse_datetime(end_time)
        # 过滤 spans，只保留起始时间小于等于 end_dt 的记录
        spans = filter(
            lambda span: _parse_datetime(span["start_time"]) <= end_dt, spans
        )

    # 如果指定了搜索条件 search，则使用 _new_search_span_func(search) 进行过滤
    if search:
        spans = filter(_new_search_span_func(search), spans)

    # 根据 span 的 start_time 属性对 spans 进行排序
    spans = sorted(
        spans, key=lambda span: _parse_datetime(span["start_time"]), reverse=desc
    )[:limit]

    # 创建一个 PrettyTable 对象，定义表格的列
    table = PrettyTable(
        ["Trace ID", "Span ID", "Operation Name", "Conversation UID"],
    )

    # 遍历 spans，为每个 span 生成一行数据并添加到表格中
    for sp in spans:
        conv_uid = None
        # 检查 span 中是否有 metadata，并且 metadata 是一个字典
        if "metadata" in sp and isinstance(sp["metadata"], dict):
            metadata = sp["metadata"]
            # 获取 metadata 中的 conv_uid，如果不存在则为 None
            conv_uid = metadata.get("conv_uid")
        
        # 向表格中添加一行数据，包括 trace_id, span_id, operation_name 和 conv_uid
        table.add_row(
            [
                sp.get("trace_id"),
                sp.get("span_id"),
                # sp.get("parent_span_id"),  # 注释掉的一行，未使用
                sp.get("operation_name"),
                conv_uid,
            ]
        )
    
    # 根据输出格式 output（可能是 "json" 或其它），生成表格的格式化字符串
    out_kwargs = {"ensure_ascii": False} if output == "json" else {}
    # 打印表格的格式化字符串到标准输出
    print(table.get_formatted_string(out_format=output, **out_kwargs))
@trace_cli_group.command()
@click.option(
    "--trace_id",
    required=True,
    type=str,
    help="Specify the trace ID to list",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True, readable=True))
def tree(trace_id: str, files):
    """Display trace links as a tree"""
    # 调用内部函数 _view_trace_hierarchy，获取跟踪层次结构
    hierarchy = _view_trace_hierarchy(trace_id, files)
    # 如果没有层次结构信息，则打印空消息并返回
    if not hierarchy:
        _print_empty_message(files)
        return
    # 打印跟踪层次结构
    _print_trace_hierarchy(hierarchy)


@trace_cli_group.command()
@click.option(
    "--trace_id",
    required=False,
    type=str,
    default=None,
    help="Specify the trace ID to analyze. If None, show latest conversation details",
)
@click.option(
    "--tree",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Display trace spans as a tree",
)
@click.option(
    "--hide_conv",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Hide your conversation details",
)
@click.option(
    "--hide_run_params",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Hide run params",
)
@click.option(
    "--output",
    required=False,
    type=click.Choice(["text", "html", "csv", "latex", "json"]),
    default="text",
    help="The output format",
)
@click.argument("files", nargs=-1, type=click.Path(exists=False, readable=True))
def chat(
    trace_id: str,
    tree: bool,
    hide_conv: bool,
    hide_run_params: bool,
    output: str,
    files,
):
    """Show conversation details"""
    from prettytable import PrettyTable

    # 从文件中读取 spans 数据
    spans = read_spans_from_files(files)

    # 按开始时间排序 spans
    spans = sorted(
        spans, key=lambda span: _parse_datetime(span["start_time"]), reverse=True
    )
    # 过滤掉空的 spans 列表
    spans = [sp for sp in spans]
    if not spans:
        # 如果 spans 为空，则打印空消息并返回
        _print_empty_message(files)
        return

    # 用于存储服务 spans 的字典
    service_spans = {}
    # 存储 SpanTypeRunName 的服务名集合
    service_names = set(SpanTypeRunName.values())
    # 用于记录找到的 trace_id
    found_trace_id = None

    # 遍历 spans
    for sp in spans:
        span_type = sp["span_type"]
        metadata = sp.get("metadata")

        # 如果 span_type 是 SpanType.RUN，并且 metadata 中包含 run_service
        if span_type == SpanType.RUN and metadata and "run_service" in metadata:
            service_name = metadata["run_service"]
            # 复制 span 到 service_spans 中的对应服务名
            service_spans[service_name] = sp.copy()
            # 如果找到的服务名集合等于 service_names，并且已找到 trace_id，则结束循环
            if set(service_spans.keys()) == service_names and found_trace_id:
                break

        # 如果 span_type 是 SpanType.CHAT，并且尚未找到 trace_id
        elif span_type == SpanType.CHAT and not found_trace_id:
            # 如果 trace_id 为 None，则标记找到的 trace_id 是当前 span 的 trace_id
            if not trace_id:
                found_trace_id = sp["trace_id"]
            # 如果 trace_id 存在且与当前 span 的 trace_id 匹配，则标记找到的 trace_id
            if trace_id and trace_id == sp["trace_id"]:
                found_trace_id = trace_id

    # 存储服务表格的字典
    service_tables = {}
    # 存储系统信息表格的字典
    system_infos_table = {}
    # 输出参数设置为 ensure_ascii=False 如果输出格式是 json
    out_kwargs = {"ensure_ascii": False} if output == "json" else {}
    # 遍历服务名和服务跨度的字典项
    for service_name, sp in service_spans.items():
        # 获取当前服务跨度的元数据
        metadata = sp["metadata"]
        # 创建一个新的漂亮表格，用于显示服务的配置键和配置值，表格标题为服务名
        table = PrettyTable(["Config Key", "Config Value"], title=service_name)
        # 遍历元数据中的参数字典，将每个参数键值对添加到漂亮表格中
        for k, v in metadata["params"].items():
            table.add_row([k, v])
        # 将服务名和对应的配置表格存入服务表格字典
        service_tables[service_name] = table
        # 获取可能存在的系统信息
        sys_infos = metadata.get("sys_infos")
        # 如果存在系统信息并且是字典类型，则创建一个显示系统配置的漂亮表格
        if sys_infos and isinstance(sys_infos, dict):
            sys_table = PrettyTable(
                ["System Config Key", "System Config Value"],
                title=f"{service_name} System information",
            )
            # 将每个系统配置键值对添加到系统信息表格中
            for k, v in sys_infos.items():
                sys_table.add_row([k, v])
            # 将服务名和对应的系统信息表格存入系统信息表格字典
            system_infos_table[service_name] = sys_table

    # 如果不隐藏运行参数信息
    if not hide_run_params:
        # 横向合并指定服务的表格，创建第一个合并表格
        merged_table1 = merge_tables_horizontally(
            [
                service_tables.get(SpanTypeRunName.WEBSERVER.value),
                service_tables.get(SpanTypeRunName.EMBEDDING_MODEL.value),
            ]
        )
        # 横向合并指定服务的表格，创建第二个合并表格
        merged_table2 = merge_tables_horizontally(
            [
                service_tables.get(SpanTypeRunName.MODEL_WORKER.value),
                service_tables.get(SpanTypeRunName.WORKER_MANAGER.value),
            ]
        )
        # 获取指定服务的系统信息表格
        sys_table = system_infos_table.get(SpanTypeRunName.WORKER_MANAGER.value)
        # 如果系统信息表格存在
        if system_infos_table:
            # 遍历系统信息表格字典，将第一个找到的系统信息表格赋值给 sys_table
            for k, v in system_infos_table.items():
                sys_table = v
                break
        # 如果输出格式为文本
        if output == "text":
            # 打印第一个合并表格
            print(merged_table1)
            # 打印第二个合并表格
            print(merged_table2)
        else:
            # 否则，遍历服务表格字典，打印每个服务表格的格式化字符串
            for service_name, table in service_tables.items():
                print(table.get_formatted_string(out_format=output, **out_kwargs))
        # 如果存在系统信息表格，则打印系统信息表格的格式化字符串
        if sys_table:
            print(sys_table.get_formatted_string(out_format=output, **out_kwargs))

    # 如果未找到追踪 ID，则打印相应消息并返回
    if not found_trace_id:
        print(f"Can't found conversation with trace_id: {trace_id}")
        return
    # 使用找到的追踪 ID 获取对应的跨度列表
    trace_id = found_trace_id
    trace_spans = [span for span in spans if span["trace_id"] == trace_id]
    # 将跨度列表反转
    trace_spans = [s for s in reversed(trace_spans)]
    # 构建追踪层次结构
    hierarchy = _build_trace_hierarchy(trace_spans)
    # 如果需要显示树形结构
    if tree:
        # 打印追踪树的标题及追踪 ID
        print(f"\nInvoke Trace Tree(trace_id: {trace_id}):\n")
        # 打印追踪层次结构
        _print_trace_hierarchy(hierarchy)

    # 如果隐藏会话，则直接返回
    if hide_conv:
        return

    # 从层次结构中获取有序的追踪跨度
    trace_spans = _get_ordered_trace_from(hierarchy)
    # 创建一个漂亮表格，用于显示聊天追踪的键和值，表格标题为 "Chat Trace Details"
    table = PrettyTable(["Key", "Value Value"], title="Chat Trace Details")
    # 如果输出格式为文本，则将长文本拆分显示
    split_long_text = output == "text"

    # 打印聊天追踪详细信息的格式化字符串
    print(table.get_formatted_string(out_format=output, **out_kwargs))
# 从文件中读取跟踪的片段信息，返回一个生成器，每次生成一个字典
def read_spans_from_files(files=None) -> Iterable[Dict]:
    """
    Reads spans from multiple files based on the provided file paths.
    根据提供的文件路径读取多个文件中的跟踪片段信息。
    """
    # 如果未提供文件列表，则使用默认的文件路径模式
    if not files:
        files = [_DEFAULT_FILE_PATTERN]

    # 遍历文件列表中的每个文件路径
    for filepath in files:
        # 使用glob模块匹配文件路径，返回符合条件的文件名列表
        for filename in glob.glob(filepath):
            # 打开文件进行读取
            with open(filename, "r") as file:
                # 逐行读取文件内容，并将每行解析为JSON格式的字典对象，使用生成器返回
                for line in file:
                    yield json.loads(line)


# 打印未找到跟踪片段记录的消息
def _print_empty_message(files=None):
    """
    Print an empty message indicating no trace span records found in the specified tracer files.
    打印一条空消息，指示在指定的跟踪器文件中未找到跟踪片段记录。
    """
    # 如果未提供文件列表，则使用默认的文件路径模式
    if not files:
        files = [_DEFAULT_FILE_PATTERN]
    # 将文件列表转换为逗号分隔的字符串，并打印提示消息
    file_names = ",".join(files)
    print(f"No trace span records found in your tracer files: {file_names}")


# 创建一个用于搜索跟踪片段的函数
def _new_search_span_func(search: str):
    """
    Create a function to search for trace spans containing a specific string.
    创建一个用于搜索包含特定字符串的跟踪片段的函数。
    """
    def func(span: Dict) -> bool:
        # 提取跟踪片段中的关键信息到一个列表中
        items = [span["trace_id"], span["span_id"], span["parent_span_id"]]
        # 如果跟踪片段中包含操作名，则加入到列表中
        if "operation_name" in span:
            items.append(span["operation_name"])
        # 如果跟踪片段中包含元数据，并且是字典类型，则将所有键和值加入到列表中
        if "metadata" in span:
            metadata = span["metadata"]
            if isinstance(metadata, dict):
                for k, v in metadata.items():
                    items.append(k)
                    items.append(v)
        # 返回是否在列表中找到包含搜索字符串的项
        return any(search in str(item) for item in items if item)

    return func


# 解析日期时间字符串为日期时间对象
def _parse_datetime(dt_str):
    """
    Parse a datetime string to a datetime object.
    将日期时间字符串解析为日期时间对象。
    """
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")


# 构建跟踪层级结构
def _build_trace_hierarchy(spans, parent_span_id=None, indent=0):
    """
    Build a hierarchical trace structure based on spans.
    根据跟踪片段构建一个层级结构的跟踪结构。
    """
    # 获取当前层级的跟踪片段
    current_level_spans = [
        span
        for span in spans
        if span["parent_span_id"] == parent_span_id and span["end_time"] is None
    ]

    hierarchy = []

    # 遍历当前层级的每个起始跟踪片段
    for start_span in current_level_spans:
        # 查找匹配的结束跟踪片段
        end_span = next(
            (
                span
                for span in spans
                if span["span_id"] == start_span["span_id"]
                and span["end_time"] is not None
            ),
            None,
        )
        # 创建跟踪条目
        entry = {
            "operation_name": start_span["operation_name"],
            "parent_span_id": start_span["parent_span_id"],
            "span_id": start_span["span_id"],
            "start_time": start_span["start_time"],
            "end_time": start_span["end_time"],
            "metadata": start_span["metadata"],
            "children": _build_trace_hierarchy(
                spans, start_span["span_id"], indent + 1
            ),
        }
        hierarchy.append(entry)

        # 如果存在结束跟踪片段，则添加结束跟踪条目
        if end_span:
            entry_end = {
                "operation_name": end_span["operation_name"],
                "parent_span_id": end_span["parent_span_id"],
                "span_id": end_span["span_id"],
                "start_time": end_span["start_time"],
                "end_time": end_span["end_time"],
                "metadata": end_span["metadata"],
                "children": [],
            }
            hierarchy.append(entry_end)

    return hierarchy


# 查看跟踪层级结构的函数（尚未实现完整）
def _view_trace_hierarchy(trace_id, files=None):
    # 从给定的文件列表中读取所有跟踪信息
    spans = read_spans_from_files(files)
    # 从所有跟踪信息中筛选出具有指定 trace_id 的跟踪信息
    trace_spans = [span for span in spans if span["trace_id"] == trace_id]
    # 如果没有找到符合条件的跟踪信息，返回 None
    if not trace_spans:
        return None
    # 根据筛选出的跟踪信息构建跟踪层次结构
    hierarchy = _build_trace_hierarchy(trace_spans)
    # 返回构建好的跟踪层次结构
    return hierarchy
# 打印给定的操作层次结构，包括每个操作的名称、开始时间和结束时间
def _print_trace_hierarchy(hierarchy, indent=0):
    """Print link hierarchy"""
    # 遍历层次结构中的每个条目
    for entry in hierarchy:
        # 打印当前条目的操作名称、开始时间和结束时间，缩进量由参数指定
        print(
            "  " * indent
            + f"Operation: {entry['operation_name']} (Start: {entry['start_time']}, End: {entry['end_time']})"
        )
        # 递归打印当前条目的子节点层次结构，缩进增加
        _print_trace_hierarchy(entry["children"], indent + 1)


# 从给定的操作层次结构中获取按顺序排列的所有追踪数据
def _get_ordered_trace_from(hierarchy):
    traces = []

    def func(items):
        # 递归函数，将每个条目及其子节点追加到 traces 列表中
        for item in items:
            traces.append(item)
            func(item["children"])

    # 调用递归函数处理整个层次结构
    func(hierarchy)
    # 返回按顺序排列的所有追踪数据列表
    return traces


# 打印服务跟踪信息，根据不同的服务名分组打印
def _print(service_spans: Dict):
    # 遍历预定义的服务名列表
    for names in [
        [SpanTypeRunName.WEBSERVER.name, SpanTypeRunName.EMBEDDING_MODEL],
        [SpanTypeRunName.WORKER_MANAGER.name, SpanTypeRunName.MODEL_WORKER],
    ]:
        pass  # 这里没有实际操作，保留占位符以后可能添加功能


# 将多个表水平合并成一个表格
def merge_tables_horizontally(tables):
    from prettytable import PrettyTable

    # 如果 tables 列表为空，则返回 None
    if not tables:
        return None

    # 去除 tables 中为空的表格
    tables = [t for t in tables if t]
    # 如果处理后 tables 仍为空，则返回 None
    if not tables:
        return None

    # 计算所有表格中行数的最大值
    max_rows = max(len(table._rows) for table in tables)

    # 创建一个新的 PrettyTable 对象作为合并后的表格
    merged_table = PrettyTable()

    # 创建一个新的字段名列表，用于合并后的表头
    new_field_names = []
    for table in tables:
        new_field_names.extend(
            [
                f"{name} ({table.title})" if table.title else f"{name}"
                for name in table.field_names
            ]
        )

    # 设置合并后表格的字段名
    merged_table.field_names = new_field_names

    # 填充合并后表格的行数据
    for i in range(max_rows):
        merged_row = []
        for table in tables:
            if i < len(table._rows):
                merged_row.extend(table._rows[i])
            else:
                # 对于较短的表格，在行不足时填充空单元格
                merged_row.extend([""] * len(table.field_names))
        merged_table.add_row(merged_row)

    # 返回合并后的 PrettyTable 对象
    return merged_table


# 根据终端宽度将字符串分割为子字符串
def split_string_by_terminal_width(s, split=True, max_len=None, sp="\n"):
    """
    Split a string into substrings based on the current terminal width.

    Parameters:
    - s: the input string
    """
    # 如果 split 参数为 False，则直接返回原始字符串 s
    if not split:
        return s
    # 如果未提供 max_len，则尝试获取当前终端宽度的80%作为 max_len
    if not max_len:
        try:
            max_len = int(os.get_terminal_size().columns * 0.8)
        except OSError:
            # 如果无法获取终端大小，则默认 max_len 为 100
            max_len = 100
    # 按照指定的 max_len 将字符串 s 分割成多行，每行末尾添加 sp 参数指定的分隔符
    return sp.join([s[i : i + max_len] for i in range(0, len(s), max_len)])
```