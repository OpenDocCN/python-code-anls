# `.\pytorch\torch\_inductor\metrics.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations

import csv  # 导入处理 CSV 文件的模块
import dataclasses  # 导入用于数据类的模块
import inspect  # 导入用于检查对象的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
from dataclasses import dataclass  # 导入数据类装饰器
from functools import lru_cache  # 导入用于缓存函数调用结果的装饰器

from typing import Dict, List, Set, Tuple, TYPE_CHECKING, Union  # 导入类型提示相关的模块

from torch._inductor import config  # 导入 Torch 模块的配置功能
from torch._inductor.utils import get_benchmark_name  # 导入获取基准名称的功能

# Prevent circular import
if TYPE_CHECKING:
    from torch._inductor.scheduler import (  # 导入 Torch 调度器相关模块，避免循环导入
        BaseSchedulerNode,
        ExternKernelSchedulerNode,
        NopKernelSchedulerNode,
        SchedulerNode,
    )

# counter for tracking how many kernels have been generated
generated_kernel_count = 0  # 记录已生成的内核数量
generated_cpp_vec_kernel_count = 0  # 记录已生成的 C++ 向量内核数量
num_bytes_accessed = 0  # 记录已访问的字节数
nodes_num_elem: List[  # 记录节点元素数量的列表
    Tuple[
        Union[NopKernelSchedulerNode, SchedulerNode, ExternKernelSchedulerNode],  # 节点类型
        int,  # 元素数量
    ]
] = []
node_runtimes: List[Tuple[BaseSchedulerNode, float]] = []  # 记录节点运行时间的列表

# counters for tracking fusions
ir_nodes_pre_fusion = 0  # 记录融合前的 IR 节点数量

# counters for tracking to_dtype inserted
cpp_to_dtype_count = 0  # 记录插入 to_dtype 的次数

# The length counts the number of outer loop fusions.
# Each element counts the number of inner kernels in each outer loop fusion.
cpp_outer_loop_fused_inner_counts: List[int] = []  # 记录外层循环融合中内核数量的列表

num_comprehensive_padding = 0  # 记录全面填充数量
num_matches_for_scatter_upon_const_tensor = 0  # 记录对常量张量的分散匹配数量


# reset all counters
def reset():
    global generated_kernel_count  # 使用全局变量
    global generated_cpp_vec_kernel_count  # 使用全局变量
    global num_bytes_accessed, nodes_num_elem  # 使用全局变量
    global ir_nodes_pre_fusion  # 使用全局变量
    global cpp_to_dtype_count  # 使用全局变量
    global cpp_outer_loop_fused_inner_counts  # 使用全局变量
    global num_comprehensive_padding  # 使用全局变量
    global num_matches_for_scatter_upon_const_tensor  # 使用全局变量

    generated_kernel_count = 0  # 重置已生成的内核数量
    generated_cpp_vec_kernel_count = 0  # 重置已生成的 C++ 向量内核数量
    num_bytes_accessed = 0  # 重置已访问的字节数
    nodes_num_elem.clear()  # 清空节点元素数量列表
    node_runtimes.clear()  # 清空节点运行时间列表
    ir_nodes_pre_fusion = 0  # 重置融合前的 IR 节点数量
    cpp_to_dtype_count = 0  # 重置插入 to_dtype 的次数
    cpp_outer_loop_fused_inner_counts.clear()  # 清空外层循环融合中内核数量列表
    num_comprehensive_padding = 0  # 重置全面填充数量
    num_matches_for_scatter_upon_const_tensor = 0  # 重置对常量张量的分散匹配数量


@dataclass
class CachedMetricsDeltas:
    """
    The subset of metrics we want update across cache hits, e.g., the
    FxGraphCache.
    """
    generated_kernel_count: int  # 记录已生成的内核数量的数据类字段
    generated_cpp_vec_kernel_count: int  # 记录已生成的 C++ 向量内核数量的数据类字段
    ir_nodes_pre_fusion: int  # 记录融合前的 IR 节点数量的数据类字段
    cpp_to_dtype_count: int  # 记录插入 to_dtype 的次数的数据类字段
    num_bytes_accessed: int  # 记录已访问的字节数的数据类字段


def get_metric_fields():
    return [field.name for field in dataclasses.fields(CachedMetricsDeltas)]  # 返回 CachedMetricsDeltas 类的字段名列表


class CachedMetricsHelper:
    """
    A helper class to help calculate and apply counter deltas for those
    metrics we want to save with cache entries (e.g., FxGraphCache) and
    apply on a cache hit.
    """

    def __init__(self):
        self.cached_metrics = {}  # 初始化缓存的指标字典
        for metric in get_metric_fields():  # 遍历所有指标字段
            self.cached_metrics[metric] = globals()[metric]  # 将全局变量中的指标值存入缓存的指标字典
    # 定义一个方法获取指标变化的字典，并返回 CachedMetricsDeltas 对象
    def get_deltas(self) -> CachedMetricsDeltas:
        # 初始化一个空字典用于存储指标的变化量
        delta_metrics = {}
        # 遍历获取所有指标字段
        for metric in get_metric_fields():
            # 计算每个指标的变化量，并存入 delta_metrics 字典中
            delta_metrics[metric] = globals()[metric] - self.cached_metrics[metric]

        # 使用 delta_metrics 字典创建 CachedMetricsDeltas 对象，并返回
        return CachedMetricsDeltas(**delta_metrics)

    # 定义一个静态方法，将给定的 CachedMetricsDeltas 对象应用到全局指标中
    @staticmethod
    def apply_deltas(delta: CachedMetricsDeltas):
        # 遍历所有指标字段
        for metric in get_metric_fields():
            # 将 delta 中对应的指标变化量加到全局变量 globals()[metric] 上
            globals()[metric] += getattr(delta, metric)
# 定义全局变量 REGISTERED_METRIC_TABLES，用于存储所有已注册的 MetricTable 对象，键为表名，值为 MetricTable 实例
REGISTERED_METRIC_TABLES: Dict[str, MetricTable] = {}

# 定义数据类 MetricTable，用于存储表格名称、列名以及行数等信息
@dataclass
class MetricTable:
    table_name: str  # 表格名称
    column_names: List[str]  # 列名列表

    num_rows_added: int = 0  # 添加的行数，默认为0

    # 添加行的方法，接受一个返回字典的函数作为参数
    def add_row(self, row_fn):
        # 如果表格名称不在启用的度量表列表中，直接返回
        if self.table_name not in enabled_metric_tables():
            return
        
        # 通过 row_fn 函数获取行数据字典
        row_dict = row_fn()
        
        # 断言列名数与行数据字典的键数相同
        assert len(self.column_names) == len(row_dict), f"{len(self.column_names)} v.s. {len(row_dict)}"
        
        # 断言列名集合与行数据字典的键集合相同
        assert set(self.column_names) == set(row_dict.keys()), f"{set(self.column_names)} v.s. {set(row_dict.keys())}"
        
        # 构建行数据列表，包括基准名称和各列数据
        row = [
            get_benchmark_name(),
        ]
        row += [row_dict[column_name] for column_name in self.column_names]
        
        # 将行数据写入表格
        self._write_row(row)

    # 返回输出文件名，格式为 metric_table_{table_name}.csv
    def output_filename(self):
        return f"metric_table_{self.table_name}.csv"

    # 写入表头的方法，包括模型名称和列名
    def write_header(self):
        filename = self.output_filename()
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(["model_name"] + self.column_names)

    # 写入行数据的私有方法
    def _write_row(self, row):
        filename = self.output_filename()
        
        # 如果尚未添加过行且文件不存在，则先写入表头
        if self.num_rows_added == 0 and not os.path.exists(filename):
            self.write_header()
        
        # 增加已添加行数
        self.num_rows_added += 1
        
        # 处理行数据中的浮点数或空值，并写入文件
        for idx, orig_val in enumerate(row):
            if isinstance(orig_val, float):
                new_val = f"{orig_val:.6f}"
            elif orig_val is None:
                new_val = ""
            else:
                new_val = orig_val
            row[idx] = new_val
        
        # 追加行数据到文件
        with open(filename, "a") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(row)

    # 静态方法，用于注册表格，创建并存储 MetricTable 实例
    @staticmethod
    def register_table(name, column_names):
        table = MetricTable(name, column_names)
        REGISTERED_METRIC_TABLES[name] = table


# 注册名为 "slow_fusion" 的表格，包含多个性能指标的列名
MetricTable.register_table(
    "slow_fusion",
    [
        "kernel1_path",
        "kernel1_latency",
        "kernel2_path",
        "kernel2_latency",
        "fused_kernel_path",
        "fused_kernel_latency",
        "slow_down_ratio",
    ]
)

# 注册名为 "graph_stats" 的表格，包含图形统计数据的列名
MetricTable.register_table(
    "graph_stats",
    [
        "graph_id",
        "num_nodes_before_fusion",
        "num_nodes_after_fusion",
    ]
)

# 注册名为 "persistent_red_perf" 的表格，包含持久化约简性能指标的列名
MetricTable.register_table(
    "persistent_red_perf",
    [
        "kernel1_name",
        "kernel2_name",
        "kernel1_latency",
        "kernel2_latency",
        "size_hints",
        "reduction_hint",
        "speedup",
    ]
)

# 注册名为 "fusion_failure_due_to_indexing_mismatch" 的表格，包含因索引不匹配导致融合失败的列名
MetricTable.register_table(
    "fusion_failure_due_to_indexing_mismatch",
    [
        "pre_grad_graph_id",
        "post_grad_graph_id",
        "node1_name",
        "node2_name",
        "node1_debug_str",
        "node2_debug_str",
        "common_buffer_names",
        "failure_reason",
    ]
)
# 注册表格用于记录点对点/归约内核的元数据。例如，模型名称、内核路径、元素数量、约简提示等。
MetricTable.register_table(
    "kernel_metadata",
    [
        "kernel_name",           # 内核名称
        "kernel_path",           # 内核路径
        "kernel_category",       # 内核类别，如点对点/归约/foreach等
        "size_hints",            # 大小提示
        "reduction_hint",        # 约简提示
        "line_of_code",          # 代码行数
        "num_load",              # 载入次数
        "num_store",             # 存储次数
        "num_for_loop",          # for循环次数
        "num_atomic_add",        # 原子加次数
        "num_args",              # 参数数量
        # xyz numel可能与size_hints不同，因为size_hints会向上取最近的2的幂。
        # Inductor kernel将在内核代码中使用xyz numel进行静态形状内核的烧录。
        # 记录它们有助于找到归约的不对齐形状。
        "xnumel",                # xnumel
        "ynumel",                # ynumel
        "rnumel",                # rnumel
        "kernel_args_num_gb",    # 内核参数数量（全局）
    ],
)


def _parse_kernel_fn_code(kernel_module_code):
    """
    kernel_module_code是包含内核函数代码的Python模块。
    kernel function是用@triton.jit注释的正确的Triton内核函数。
    """
    from .codecache import PyCodeCache
    from .wrapper_benchmark import get_triton_kernel

    # 加载内核模块代码
    mod = PyCodeCache.load(kernel_module_code)
    # 获取Triton内核函数
    kernel = get_triton_kernel(mod)
    # kernel是一个CachingAutotune；kernel.fn是JITFunction；
    # kernel.fn.fn是被triton.jit装饰的函数
    return inspect.getsource(kernel.fn.fn)


def _parse_kernel_line_of_code(proper_kernel_fn_code):
    """
    返回内核代码的行数，不包括装饰器。
    """
    return len(proper_kernel_fn_code.splitlines())


def _parse_size_hints(kernel_module_code, kernel_category):
    if kernel_category == "foreach":
        # foreach内核没有size_hints
        return None
    # 从kernel_module_code中解析size_hints
    m = re.search(r"size_hints=(\[[0-9, ]*\]),", kernel_module_code)
    assert m, "size_hints缺失！"
    return m.group(1)


def _parse_reduction_hint(kernel_category, kernel_module_code):
    if kernel_category not in ("reduction", "persistent_reduction"):
        return None
    # 从kernel_module_code中解析reduction_hint
    m = re.search(r"reduction_hint=ReductionHint\.(\w*),", kernel_module_code)
    assert m, "内核源代码中未找到reduction_hint！"
    return m.group(1)


def _count_pattern(proper_kernel_fn_code, pattern):
    # 统计proper_kernel_fn_code中pattern出现的次数
    return proper_kernel_fn_code.count(pattern)


def _count_args(proper_kernel_fn_code):
    def_line = proper_kernel_fn_code.splitlines()[0]
    assert def_line.startswith("def ")
    start_idx = def_line.index("(")
    end_idx = def_line.index("):")
    decl_csv = def_line[start_idx + 1 : end_idx]
    comps = decl_csv.split(",")
    return len(comps)


def _parse_proper_kernel_fn_code(kernel_fn_code):
    """
    跳过装饰器，返回适当的内核函数代码。
    """
    start_pos = kernel_fn_code.index("def ")
    return kernel_fn_code[start_pos:]


def _parse_numel(proper_kernel_fn_code, numel_arg_name):
    # 从proper_kernel_fn_code中解析numel_arg_name对应的数值
    m = re.search(f"{numel_arg_name} = ([\\d]+)", proper_kernel_fn_code)
    # 如果匹配对象 m 不为 None，则返回捕获组 1 的整数形式
    if m:
        return int(m.group(1))
    # 如果 m 为 None，则返回 None
    else:
        return None
# 解析内核函数代码中的 kernel_num_gb 字段，返回其值作为浮点数
def _parse_kernel_args_num_gb(kernel_fn_code, kernel_category):
    # 使用正则表达式搜索 kernel_num_gb 字段并提取其值
    m = re.search(r".kernel_num_gb.:\s*([0-9.]+)", kernel_fn_code)
    if m:
        return float(m.group(1))
    else:
        # 如果 kernel_num_gb 字段缺失，返回 None
        """
        There are a few cases that kernel_num_gdb field can be missing:
        1. the field will be missing if config.benchmark_kernel and
           config.profile_bandwidth are false
        2. even if config.benchmark_kernel or config.profile_bandwidth is true.
           foreach kernel does not have kernel_num_gb field in the metadata
        """
        return None


# 记录内核元数据的实用函数，从内核源代码中解析元数据
def log_kernel_metadata(kernel_name, kernel_path, kernel_module_code):
    # 导入获取内核类别的函数
    from .wrapper_benchmark import get_kernel_category_by_source_code

    # 获取内核类别
    kernel_category = get_kernel_category_by_source_code(kernel_module_code)
    
    # 解析缩减提示和大小提示
    reduction_hint = _parse_reduction_hint(kernel_category, kernel_module_code)
    size_hints = _parse_size_hints(kernel_module_code, kernel_category)
    
    # 解析内核函数代码
    kernel_fn_code = _parse_kernel_fn_code(kernel_module_code)
    
    # 解析修正后的内核函数代码
    proper_kernel_fn_code = _parse_proper_kernel_fn_code(kernel_fn_code)
    
    # 解析内核函数代码的行数（不包括装饰器）
    kernel_line_of_code = _parse_kernel_line_of_code(proper_kernel_fn_code)
    
    # 获取度量表并添加行，记录内核元数据
    get_metric_table("kernel_metadata").add_row(
        lambda: {
            "kernel_name": kernel_name,
            "kernel_path": kernel_path,
            "kernel_category": kernel_category,
            "size_hints": size_hints,
            "reduction_hint": reduction_hint,
            "line_of_code": kernel_line_of_code,
            "num_load": _count_pattern(proper_kernel_fn_code, "tl.load"),
            "num_store": _count_pattern(proper_kernel_fn_code, "tl.store"),
            "num_for_loop": _count_pattern(proper_kernel_fn_code, "for "),
            "num_atomic_add": _count_pattern(proper_kernel_fn_code, "tl.atomic_add"),
            "num_args": _count_args(proper_kernel_fn_code),
            "xnumel": _parse_numel(proper_kernel_fn_code, "xnumel"),
            "ynumel": _parse_numel(proper_kernel_fn_code, "ynumel"),
            "rnumel": _parse_numel(proper_kernel_fn_code, "rnumel"),
            # 解析内核函数的 kernel_args_num_gb 字段
            "kernel_args_num_gb": _parse_kernel_args_num_gb(kernel_fn_code, kernel_category),
        }
    )


# 清理旧的日志文件函数，用于在基准测试脚本运行时清理旧的日志文件
def purge_old_log_files():
    """
    Purge the old log file at the beginning when the benchmark script runs.
    Should do it in the parent process rather than the child processes running
    each individual model.
    """
    # 遍历REGISTERED_METRIC_TABLES字典，获取每个注册表的名称(name)和对应的表对象(table)
    for name, table in REGISTERED_METRIC_TABLES.items():
        # 检查当前表名是否在启用的度量表列表中
        if name in enabled_metric_tables():
            # 获取当前表对象的输出文件名
            filename = table.output_filename()
            # 如果该文件名已存在，则删除该文件
            if os.path.exists(filename):
                os.unlink(filename)

            # 调用表对象的写头部信息的方法
            table.write_header()
# 使用 functools 模块中的 lru_cache 装饰器，将该函数结果缓存，提高性能
@lru_cache
# 定义一个函数，返回启用的度量表名称集合
def enabled_metric_tables() -> Set[str]:
    # 从配置中获取启用的度量表名称字符串
    config_str = config.enabled_metric_tables

    # 初始化一个空集合来存储启用的度量表名称
    enabled = set()

    # 遍历配置字符串中以逗号分隔的每个名称
    for name in config_str.split(","):
        # 去除名称两端的空白字符
        name = name.strip()
        # 如果名称为空则跳过当前循环
        if not name:
            continue
        # 断言当前名称在已注册的度量表集合中，如果不在则抛出异常
        assert (
            name in REGISTERED_METRIC_TABLES
        ), f"Metric table name {name} is not registered"
        # 将当前名称添加到启用的度量表名称集合中
        enabled.add(name)

    # 返回包含所有启用度量表名称的集合
    return enabled


# 定义一个函数，检查给定名称的度量表是否已启用
def is_metric_table_enabled(name):
    # 返回判断给定名称是否在已启用度量表名称集合中的布尔值
    return name in enabled_metric_tables()


# 定义一个函数，获取给定名称的度量表对象
def get_metric_table(name):
    # 断言给定名称在已注册的度量表集合中，如果不存在则抛出异常
    assert name in REGISTERED_METRIC_TABLES, f"Metric table {name} is not defined"
    # 返回已注册度量表集合中对应名称的度量表对象
    return REGISTERED_METRIC_TABLES[name]
```