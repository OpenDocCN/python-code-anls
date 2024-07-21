# `.\pytorch\torch\_inductor\debug.py`

```py
# mypy: allow-untyped-defs
import collections  # 导入collections模块，用于命名元组等数据结构
import contextlib  # 导入contextlib模块，用于创建上下文管理器和处理上下文相关的异常
import dataclasses  # 导入dataclasses模块，用于创建和操作数据类
import functools  # 导入functools模块，用于高阶函数操作，如LRU缓存
import itertools  # 导入itertools模块，用于高效的迭代工具
import logging  # 导入logging模块，用于记录日志信息
import os  # 导入os模块，用于与操作系统进行交互
import os.path  # 导入os.path模块，用于处理文件和目录路径
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import pstats  # 导入pstats模块，用于处理分析统计数据
import shutil  # 导入shutil模块，用于高级文件操作
import subprocess  # 导入subprocess模块，用于创建新的进程，执行外部命令
from typing import Any, Dict, List, Optional  # 导入类型提示相关的类和方法
from unittest.mock import patch  # 导入patch函数，用于在测试中模拟对象

import torch  # 导入PyTorch深度学习库

from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled  # 导入编译相关函数
from torch import fx as fx  # 导入PyTorch FX子模块

from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug  # 导入AOT编译相关函数
from torch._dynamo.utils import get_debug_dir  # 导入调试目录获取函数
from torch.fx.graph_module import GraphModule  # 导入GraphModule类
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata  # 导入张量元数据提取相关函数和类
from torch.fx.passes.tools_common import legalize_graph  # 导入合法化图函数
from torch.utils._pytree import tree_map  # 导入树结构映射函数

from . import config, ir  # 导入本地模块config和ir（需忽略F811错误）

from .scheduler import (  # 从本地模块导入调度器相关类
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
)
from .virtualized import V  # 从本地模块导入虚拟化相关类V

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例

SchedulerNodeList = List[Any]  # 定义SchedulerNodeList为Any类型的列表
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])  # 定义BufMeta为命名元组，包含name和n_origin两个字段
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]  # 定义GRAPHVIZ_COMMAND_SCALABLE为图形化命令列表

@functools.lru_cache(None)
def has_dot() -> bool:
    """
    Check if 'dot' executable is available.
    """
    try:
        subprocess.check_output(["which", "dot"], stderr=subprocess.PIPE)  # 尝试执行shell命令查找'dot'命令
        return True  # 如果找到'dot'命令，返回True
    except subprocess.SubprocessError:
        return False  # 如果未找到'dot'命令，返回False

def draw_buffers(nodes: List[BaseSchedulerNode], print_graph=False, fname=None):
    """
    Draw a graph in fname.svg.
    """
    if not has_dot():  # 如果没有安装'dot'，输出警告信息并返回
        log.warning("draw_buffers() requires `graphviz` package")
        return

    if fname is None:  # 如果未指定文件名，使用当前编译图的默认文件名
        fname = get_graph_being_compiled()

    graph = create_fx_from_snodes(nodes)  # 创建FX图形对象

    for node in graph.nodes:  # 遍历图中的节点
        if "fusion_meta" not in node.meta:  # 如果节点的元数据中没有'fusion_meta'信息，跳过
            continue
        group = node.meta["fusion_meta"].group  # 获取节点的融合组信息
        if isinstance(group, tuple):  # 如果组信息是元组
            if isinstance(group[1], int):
                group = (group[1],)
            else:
                group = group[1]

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):  # 如果节点是计算缓冲区类型
            dtype = node.data.dtype  # 获取节点数据的数据类型

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)  # 创建张量元数据对象
        node.meta["tensor_meta"] = metadata  # 将张量元数据对象存储在节点的元数据中

    if print_graph:  # 如果指定打印图形
        print(graph)  # 打印图形对象

    gm = GraphModule({}, graph)  # 创建图模块对象
    legalize_graph(gm)  # 合法化图形对象
    gm.graph.lint()  # 对图形进行检查
    draw_graph(  # 绘制图形
        gm, fname, clear_meta=False, dot_graph_shape=config.trace.dot_graph_shape
    )

def create_fx_from_snodes(snodes: List[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name):
        def func1(*args):
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])  # 定义融合元数据命名元组

    buf_to_fx_node = {}  # 创建空字典，用于映射缓冲区到FX节点
    graph = torch.fx.Graph()  # 创建FX图形对象
    # 初始化第一个节点为空
    first_node = None

    # 初始化输出节点列表为空
    outputs = []

    # 初始化组变量为 None
    group: Any = None

    # 遍历每个 SchedulerNode
    # 为每个 Buffer 和 Kernel 创建一个 call_function 节点
    for snode in snodes:
        # 根据节点类型设置 node_type 和 group
        if snode.is_extern():
            node_type = "extern"
            group = node_type
        elif snode.is_template():
            node_type = "template"
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")

        # 获取融合的名称
        fused_name = torch._inductor.utils.get_fused_kernel_name(
            snode.get_nodes(), "original_aten"
        )
        func_name = f"{node_type}: {fused_name}"

        # 获取假的函数对象
        node_func = get_fake_func(func_name)

        # 初始化关键字参数为空字典
        kwargs = {}

        # 如果节点有 get_device 方法，设置设备参数
        if hasattr(snode, "get_device"):
            kwargs = {"device": snode.get_device()}

        # 创建 call_function 节点
        fx_node = graph.call_function(node_func, args=(), kwargs=kwargs)

        # 判断节点是否为输出节点，如果是则添加到 outputs 列表中
        def in_output(snode):
            if isinstance(snode, FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(isinstance(user.node, OutputNode) for user in snode.users)

        if in_output(snode):
            outputs.append(fx_node)

        # 设置节点名称
        name = snode.get_name()
        fx_node.name = name

        # 设置节点的元数据 fusion_meta
        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        # 如果节点是融合节点，将其子节点映射到 fx_node
        if isinstance(snode, FusedSchedulerNode):
            for x in snode.snodes:
                buf_to_fx_node[x.get_name()] = fx_node

        # 将节点映射到 buf_to_fx_node
        buf_to_fx_node[name] = fx_node

        # 如果是第一个节点，则设置为当前节点
        if first_node is None:
            first_node = fx_node

    # 创建节点之间的边
    for snode in snodes:
        # 获取节点名称和依赖
        name = snode.get_name()
        deps = snode.read_writes.reads

        # 获取对应的 fx_node
        fx_node = buf_to_fx_node[name]

        # 创建新的参数列表
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                # 如果依赖节点不存在于 buf_to_fx_node 中，创建一个占位符节点
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)

        # 更新节点的参数列表
        fx_node.args = tuple(new_args)

    # 设置图的输出节点
    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))

    # 返回构建的图
    return graph
# 定义函数，用于更新原始 FX 节点名称到缓冲区名称的映射关系
def update_orig_fx_node_name_to_buf_name(
    nodes: SchedulerNodeList,
    node_name_to_buf_name: Dict[str, str],
    parent_buf_name: Optional[str] = None,
    n_origins: int = 0,
):
    # 如果节点列表为空，则直接返回
    if nodes is None:
        return

    # 遍历节点列表中的每个节点
    for node in nodes:
        # 获取当前节点的名称作为缓冲区名称
        buf_name = node.get_name()
        # 获取当前节点的子节点列表
        children_nodes = node.get_nodes()

        # 如果子节点列表不为空且长度大于1，则递归调用更新函数
        if children_nodes is not None and len(children_nodes) > 1:
            update_orig_fx_node_name_to_buf_name(
                children_nodes,
                node_name_to_buf_name,
                buf_name if parent_buf_name is None else parent_buf_name,
            )
            continue
        else:
            # 如果子节点列表长度为1且唯一的子节点是当前节点本身，则断言成立
            assert len(children_nodes) == 1 and children_nodes[0] == node

        # 获取当前节点对应的 IR 节点
        ir_node = node.node

        # 如果 IR 节点为空或者其 origins 属性为空，则继续下一个节点
        if ir_node is None or ir_node.origins is None:
            continue

        # 遍历 IR 节点的 origins
        for origin in ir_node.origins:
            node_name = origin.name
            # 当 node_name 不在 node_name_to_buf_name 中时，将其映射为当前 buf_name 或者 parent_buf_name
            if node_name not in node_name_to_buf_name:
                node_name_to_buf_name[node_name] = (
                    buf_name if parent_buf_name is None else parent_buf_name
                )


# 定义函数，根据节点名称到缓冲区名称的映射，生成缓冲区名称到节点元数据的映射
def get_node_name_to_buf_meta(node_name_to_buf_name: Dict[str, str]):
    # 创建一个空字典，用于记录每个缓冲区名称对应的节点数目
    buf_name_to_n_node = {}
    
    # 遍历节点名称到缓冲区名称的映射字典
    for node_name, buf_name in node_name_to_buf_name.items():
        # 如果当前 buf_name 不在 buf_name_to_n_node 中，则初始化为一个集合
        if buf_name not in buf_name_to_n_node:
            buf_name_to_n_node[buf_name] = {node_name}
        else:
            # 如果已经存在，则添加节点名称到集合中
            buf_name_to_n_node[buf_name].add(node_name)

    # 创建一个空字典，用于记录每个节点名称到缓冲区元数据的映射
    node_name_to_buf_meta = {}
    
    # 再次遍历节点名称到缓冲区名称的映射字典
    for node_name, buf_name in node_name_to_buf_name.items():
        # 获取当前缓冲区名称对应的节点数目
        n_node = len(buf_name_to_n_node[buf_name])
        # 将节点名称映射为包含缓冲区名称和节点数目的 BufMeta 对象，并存入结果字典
        node_name_to_buf_meta[node_name] = BufMeta(buf_name, n_node)
    
    return node_name_to_buf_meta


# 定义函数，将 SchedulerNodeList 中的节点映射到 GM 图中的节点元数据
def annotate_orig_fx_with_snodes(
    gm: torch.fx.GraphModule, snodes: SchedulerNodeList
) -> None:
    # 创建一个空字典，用于存储节点名称到缓冲区名称的映射关系
    node_name_to_buf_name: Dict[str, str] = {}
    
    # 更新节点名称到缓冲区名称的映射关系
    update_orig_fx_node_name_to_buf_name(snodes, node_name_to_buf_name)
    
    # 如果映射字典为空，则直接返回
    if node_name_to_buf_name is None:
        return
    
    # 获取节点名称到缓冲区元数据的映射关系
    node_name_to_buf_meta = get_node_name_to_buf_meta(node_name_to_buf_name)
    
    # 遍历 GM 图中的每个节点
    for node in gm.graph.nodes:
        # 如果当前节点的名称存在于节点名称到缓冲区元数据的映射中，则设置当前节点的 meta 属性为对应的缓冲区元数据
        if node.name in node_name_to_buf_meta:
            node.meta["buf_meta"] = node_name_to_buf_meta.get(node.name)


# 定义上下文管理器，用于启用 AOT 编译日志记录
@contextlib.contextmanager
def enable_aot_logging():
    # 检查是否启用编译调试
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    # 导入必要的模块
    import torch._functorch.aot_autograd

    # 获取 AOT 自动求导模块的日志记录器
    log = logging.getLogger(torch._functorch.aot_autograd.__name__)

    # 创建一个上下文管理器堆栈
    stack = contextlib.ExitStack()

    # 如果未启用编译调试，则直接返回
    if not compile_debug:
        try:
            yield
        finally:
            stack.close()
        return

    # 否则，启用所有图形的记录到文件功能，设置文件记录器的日志级别为 DEBUG
    # 并在此处记录详细的实现过程
    # 使用上下文管理器进入 patch 模式，设置 functorch.compile.config.debug_partitioner 为 True
    stack.enter_context(patch("functorch.compile.config.debug_partitioner", True))

    # 获取调试目录，并确保目录存在
    path = os.path.join(get_debug_dir(), "torchinductor")
    os.makedirs(path, exist_ok=True)

    # 创建一个文件处理器，用于记录调试日志到文件
    fh = logging.FileHandler(
        os.path.join(
            path,
            f"aot_{get_aot_graph_name()}_debug.log",
        )
    )
    fh.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别为 DEBUG
    fh.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
    )
    # 将文件处理器添加到全局日志记录器 log 中
    log.addHandler(fh)
    try:
        # 执行 yield 语句，即执行被装饰函数或上下文中的代码
        yield
    finally:
        # 在 finally 块中，移除文件处理器，关闭文件流
        log.removeHandler(fh)
        stack.close()
class DebugContext:
    # 使用 itertools.count() 创建一个计数器，用于生成唯一的调试目录编号
    _counter = itertools.count()

    # 静态方法 wrap，接受一个函数 fn 作为参数，并返回一个装饰器
    @staticmethod
    def wrap(fn):
        # 内部函数 inner 包装了传入的函数 fn，并在其内部使用 DebugContext 上下文
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            # 进入 DebugContext 上下文
            with DebugContext():
                # 调用传入的函数 fn，并返回其结果
                return fn(*args, **kwargs)

        # 返回内部函数 inner，作为包装后的函数
        return wrap_compiler_debug(inner, compiler_name="inductor")

    # 静态方法 create_debug_dir，用于创建调试目录
    @staticmethod
    def create_debug_dir(folder_name: str) -> Optional[str]:
        # 获取调试目录的根路径
        debug_dir = config.trace.debug_dir or get_debug_dir()
        # 使用 DebugContext._counter 计数器生成唯一编号
        for n in DebugContext._counter:
            # 组装调试目录的完整路径
            dirname = os.path.join(
                debug_dir,
                "torchinductor",
                f"{folder_name}.{n}",
            )
            # 如果目录不存在，则创建该目录，并返回其路径
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                return dirname
        # 如果所有可能的调试目录都已存在，则返回 None
        return None

    # DebugContext 的初始化方法
    def __init__(self):
        # 初始化属性：_prof 为 None，_path 为 None，_stack 为一个上下文管理器
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    # 方法 copy，用于复制调试文件到新路径
    def copy(self, new_path: str):
        # 如果 _path 为 None，则直接返回
        if not self._path:
            return
        # 确保新路径以 ".debug" 结尾
        assert new_path.endswith(".debug"), new_path
        # 导入 FileLock
        from filelock import FileLock

        try:
            # 使用 FileLock 锁定新路径，确保线程安全地复制文件
            with FileLock(f"{new_path}.lock"):
                # 如果新路径已存在，则先删除
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                # 复制整个 _path 到新路径
                shutil.copytree(self._path, new_path)
        except OSError:
            # 如果复制过程中发生 OSError，则记录警告日志
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )

    # 方法 fopen，打开位于 _path 下的文件，返回文件对象
    def fopen(self, filename: str, write_mode: str = "w", *args, **kwargs):
        # 确保 _path 不为 None
        assert self._path
        # 打开 _path 下的文件，并返回文件对象
        return open(os.path.join(self._path, filename), write_mode, *args, **kwargs)

    # 上下文管理器 fopen_context，打开位于 _path 下的文件，并作为上下文返回文件对象
    @contextlib.contextmanager
    def fopen_context(self, filename: str, write_mode: str = "w", *args, **kwargs):
        # 确保 _path 不为 None
        assert self._path
        # 打开 _path 下的文件，作为上下文，并返回文件对象
        with open(os.path.join(self._path, filename), write_mode, *args, **kwargs) as f:
            yield f

    # 方法 filename，返回位于 _path 下的文件名
    def filename(self, suffix: str):
        # 确保 _path 不为 None
        assert self._path
        # 返回 _path 加上给定后缀 suffix 的文件名
        return os.path.join(self._path, suffix)

    # 方法 upload_tar，用于将调试文件夹打包为 tar.gz 文件并上传
    def upload_tar(self):
        # 如果 config.trace.upload_tar 不为 None，则执行打包上传操作
        if config.trace.upload_tar is not None:
            # 确保 _path 不为 None
            assert self._path
            # 导入 tarfile
            import tarfile

            # 构造 tar.gz 文件的路径
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            # 创建 tar.gz 文件，并将 _path 目录及其内容添加到其中
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            # 调用 config.trace.upload_tar 函数上传 tar.gz 文件
            config.trace.upload_tar(tar_file)
    # 进入上下文管理器时调用的方法，用于设置调试环境和日志捕获
    def __enter__(self):
        # 如果配置为调试模式
        if config.debug:
            # 获取名为"torch._dynamo"的日志记录器对象
            log = logging.getLogger("torch._dynamo")
            # 保存当前日志级别
            prev_level = log.level
            # 设置日志级别为DEBUG
            log.setLevel(logging.DEBUG)

            # 定义回调函数，用于在离开上下文时恢复之前的日志级别
            def reset_log_level(level):
                log.setLevel(level)

            # 将回调函数添加到堆栈中
            self._stack.callback(reset_log_level, prev_level)

        # 设置调试处理程序
        self._stack.enter_context(V.set_debug_handler(self))

        # 如果未启用跟踪，直接返回
        if not config.trace.enabled:
            return

        # 创建调试目录并设置路径
        self._path = self.create_debug_dir(get_aot_graph_name())

        # 如果配置为输出调试日志
        if config.trace.debug_log:
            # 设置捕获"debug.log"文件中的日志信息
            self._setup_log_capture("debug.log", logging.DEBUG)
        # 如果配置为输出信息日志
        if config.trace.info_log:
            # 设置捕获"info.log"文件中的日志信息
            self._setup_log_capture("info.log", logging.INFO)

    # 设置日志捕获的具体方法，用于配置日志记录器和文件处理器
    def _setup_log_capture(self, filename: str, level: int):
        # 获取名为"torch._inductor"的日志记录器对象
        log = logging.getLogger("torch._inductor")
        # 打开文件并将其作为上下文管理器的一部分
        fd = self._stack.enter_context(self.fopen(filename))
        # 创建流处理器并设置日志级别
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        # 设置日志格式
        ch.setFormatter(
            logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
        )
        # 将流处理器添加到日志记录器
        log.addHandler(ch)
        # 设置日志记录器的级别为当前级别和指定级别的最小值
        log.setLevel(min(log.level, level))
        # 在离开上下文时移除流处理器
        self._stack.callback(log.removeHandler, ch)

    # 退出上下文管理器时调用的方法，用于保存性能分析数据和上传调试结果
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果启用了性能分析
        if self._prof:
            # 禁用性能分析器并保存数据
            self._prof.disable()
            self._save_profile_data()

        # 如果存在调试路径
        if self._path:
            # 上传调试结果到远程服务器
            self.upload_tar()
            # 记录警告日志，指示调试路径和正在编译的图形名称
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        # 关闭堆栈上下文管理器
        self._stack.close()

    # 保存性能分析数据的方法，将数据写入文件并打印统计信息
    def _save_profile_data(self):
        # 断言性能分析器对象存在
        assert self._prof
        # 将性能数据保存到"compile.prof"文件中
        self._prof.dump_stats(self.filename("compile.prof"))
        # 使用文件对象创建性能统计对象
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            # 剥离文件路径信息
            stats.strip_dirs()
            # 根据累计运行时间排序并打印前100项统计信息
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            # 根据总运行时间排序并打印前100项统计信息
            stats.sort_stats("tottime")
            stats.print_stats(100)

    # 获取属性的方法，用于动态返回调试格式化对象或忽略异常
    def __getattr__(self, name):
        # 如果启用了跟踪并且配置项为真
        if config.trace.enabled and getattr(config.trace, name):
            try:
                # 返回由DebugFormatter创建的属性对象
                return getattr(DebugFormatter(self), name)
            except Exception:
                # 记录警告日志，指示在调试代码中忽略异常
                log.warning("Ignoring exception in debug code", exc_info=True)
        else:
            # 定义一个忽略一切的空函数
            def ignored(*args, **kwargs):
                pass

            # 返回忽略函数
            return ignored
class DebugFormatter:
    # DebugFormatter 类用于格式化调试信息输出

    def __init__(self, handler):
        # 初始化方法，接收一个 handler 对象作为参数
        self.fopen = handler.fopen
        self.fopen_context = handler.fopen_context
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        # 生成 FX 图并保存为可运行的 Python 脚本
        with self.fopen("fx_graph_runnable.py") as fd:
            save_graph_repro(fd, gm, inputs, "inductor")

        # 将 FX 图输出为可读的 Python 脚本
        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ):
        # 将转换后的 FX 图输出为可读的 Python 脚本
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList):
        # 将节点列表输出为预融合的 IR 格式文本
        self._write_ir("ir_pre_fusion.txt", nodes)

    def ir_post_fusion(self, nodes: SchedulerNodeList):
        # 将节点列表输出为后融合的 IR 格式文本
        self._write_ir("ir_post_fusion.txt", nodes)

    def _write_ir(self, filename: str, nodes: SchedulerNodeList):
        # 内部方法，将节点列表写入指定文件中作为 IR 格式
        with self.fopen(filename) as fd:
            log.info("Writing debug ir to  %s", fd.name)
            for node in nodes:
                fd.write(node.debug_str())
                fd.write("\n\n\n")

    def graph_diagram(self, nodes: SchedulerNodeList):
        # 绘制调度节点列表的图示，并保存为 SVG 文件
        draw_buffers(nodes, fname=self.filename("graph_diagram.svg"))

    def draw_orig_fx_graph(self, gm: torch.fx.GraphModule, nodes: SchedulerNodeList):
        # 绘制原始的 FX 图，并保存为 SVG 文件
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(
            gm,
            fname=self.filename("orig_fx_graph_diagram.svg"),
            clear_meta=False,
            prog=GRAPHVIZ_COMMAND_SCALABLE,
            parse_stack_trace=True,
            dot_graph_shape=config.trace.dot_graph_shape,
        )

    def output_code(self, filename):
        # 复制指定文件到输出文件名处
        shutil.copy(filename, self.filename("output_code.py"))

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: List[ir.IRNode],
        timings: Dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
    ):
        # 记录自动调优结果的日志信息
        pass  # (方法体未完整提供，此处仅占位)
    # 定义处理张量的函数，将张量转换为其元数据对象以便序列化
    def handle_tensor(x):
        """
        Pickle FakeTensor will result in error:
        AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'

        Convert all Tensor to metadata. This may also makes pickle faster.
        """
        # 如果输入是 torch.Tensor 类型，则转换为其元数据对象
        if isinstance(x, torch.Tensor):
            return TensorMetadataHolder(_extract_tensor_metadata(x), x.device)
        else:
            # 其它类型直接返回
            return x

    # 对参数和关键字参数应用 handle_tensor 函数，转换其中的张量
    args_to_save, kwargs_to_save = tree_map(handle_tensor, (args, kwargs))

    # 定义文件名和路径以保存编译后的函数及其参数
    fn_name = "compile_fx_inner"
    path = f"{folder}/{fn_name}_{next(save_args_cnt)}.pkl"

    # 使用二进制写模式打开文件，准备将数据序列化并保存到文件中
    with open(path, "wb") as f:
        pickle.dump((args_to_save, kwargs_to_save), f)

    # 如果日志级别为 DEBUG，则生成详细信息消息
    if log.isEnabledFor(logging.DEBUG):
        message = f"""
# 将消息打印到标准输出，而不是使用日志调试功能。使用日志调试功能会在每行消息前添加消息前缀，
# 使得代码片段难以复制。
# 尽管代码已经通过检查日志级别来进行保护，这并不是什么大问题。
print(message)

# 从指定路径加载参数并运行 compile_fx_inner 函数
def load_args_and_run_compile_fx_inner(path: str):
    # 导入 compile_fx_inner 函数
    from torch._inductor.compile_fx import compile_fx_inner

    # 使用二进制读取打开路径指定的文件
    with open(path, "rb") as f:
        # 从文件中反序列化出参数和关键字参数
        args, kwargs = pickle.load(f)

    # 定义处理张量的函数，用于替换 TensorMetadataHolder 对象为随机生成的张量数据
    def handle_tensor(x):
        # 如果 x 是 TensorMetadataHolder 类型的实例
        if isinstance(x, TensorMetadataHolder):
            # 返回使用指定的形状、步长、数据类型和设备生成的随机张量数据
            return torch._dynamo.testing.rand_strided(
                x.tensor_metadata.shape,
                x.tensor_metadata.stride,
                x.tensor_metadata.dtype,
                x.device,
            )
        else:
            # 否则直接返回 x
            return x

    # 创建一个 FakeTensorMode 上下文，允许非 Fake 输入
    fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
    
    # 在 fake_mode 上下文中，设置 config.patch("save_args", False) 来禁用 save_args 选项
    with fake_mode, config.patch("save_args", False):
        # 对 args 和 kwargs 中的每个元素应用 handle_tensor 函数
        args, kwargs = tree_map(handle_tensor, (args, kwargs))
        # 调用 compile_fx_inner 函数，传入处理后的参数和关键字参数，并返回其结果
        return compile_fx_inner(*args, **kwargs)
```