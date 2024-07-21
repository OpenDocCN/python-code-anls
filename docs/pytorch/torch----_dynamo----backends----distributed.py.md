# `.\pytorch\torch\_dynamo\backends\distributed.py`

```
# 忽略类型检查错误，由于某些特定情况，mypy检查时会忽略错误
# 导入日志模块，用于记录程序运行时的信息
import logging
# 导入异常堆栈追踪模块，用于记录和输出异常的详细信息
import traceback
# 导入用于定义数据类的装饰器，用于定义具有默认字段值和自动生成方法的类
from dataclasses import dataclass, field
# 导入类型提示模块，用于指定变量、函数参数和返回值的类型
from typing import Any, List, Optional
# 导入模拟测试模块，用于模拟对象和函数的行为
from unittest import mock

# 导入PyTorch深度学习框架
import torch
# 从torch模块中导入fx子模块，用于构建和修改图的工具
from torch import fx
# 导入torch._dynamo.output_graph模块中的GraphCompileReason类
from torch._dynamo.output_graph import GraphCompileReason
# 导入torch._dynamo.utils模块中的函数和类，用于深度拷贝和检测假张量
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
# 导入torch._logging模块中的trace_structured函数，用于结构化记录信息
from torch._logging import trace_structured
# 从torch.fx.node模块导入Node类，用于表示图中的节点

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)
# 获取一个用于记录分布式数据并行（DDP）图形信息的专用日志记录器对象
ddp_graph_log = torch._logging.getArtifactLogger(__name__, "ddp_graphs")


def args_str(args):
    # 一个用于调试的辅助函数，返回给定参数的字符串表示
    if torch.is_tensor(args):
        return f"T[{args.shape}]"
    elif isinstance(args, tuple):
        return f"tuple({', '.join([args_str(x) for x in args])})"
    elif isinstance(args, list):
        return f"list({', '.join([args_str(x) for x in args])})"
    else:
        return str(args)


@dataclass
class Bucket:
    # 桶的大小，默认为0
    size: int = 0
    # 参数列表，默认为空列表
    params: List[str] = field(default_factory=list)
    # 节点列表，默认为空列表
    nodes: List[fx.Node] = field(default_factory=list)

    # 仅用于单元测试的参数列表
    param_ids: List = field(default_factory=list)

    # 用于记录为了日志目的而扩展的桶的数量
    opcount_increased_to_capture_external_output: int = 0
    # 在操作计数增加前桶的参数大小
    paramsize_before_opcount_increase: int = 0


def bucket_has_external_output(bucket: Bucket) -> bool:
    nodes_in_bucket = set()
    # 我们希望按照反向顺序迭代，但幸运的是bucket.nodes列表已经是反向创建的
    # 所以我们不需要在这里反转它
    for node in bucket.nodes:
        # 假设节点的操作不是输出，因为这些已在原始迭代中被过滤掉
        nodes_in_bucket.add(node)
        for user in node.users:
            if user not in nodes_in_bucket:
                return True
    return False


def pretty_print_buckets(buckets: List[Bucket], bucket_bytes_cap: int):
    headers = ("Index", "Size (b)", "Param Names")
    rows = []
    extended_buckets = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            for param in bucket.params[1:]:
                rows.append((None, None, param))
        if bucket.opcount_increased_to_capture_external_output > 0:
            extended_buckets.append(
                (
                    idx,
                    bucket.opcount_increased_to_capture_external_output,
                    bucket.size - bucket.paramsize_before_opcount_increase,
                )
            )
    # 如果 rows 的长度大于零，则记录优化器使用的桶容量和创建的桶的数量
    if len(rows):
        log.info(
            "\nDDPOptimizer used bucket cap %s and created %d buckets. Enable debug logs for detailed bucket info.",
            bucket_bytes_cap,
            len(buckets),
        )

        # 如果 extended_buckets 的长度大于零，则记录警告信息，说明某些桶已超出请求的容量限制以确保每个子图都有输出节点
        if len(extended_buckets):
            log.warning(
                "Some buckets were extended beyond their requested parameter capacities"
                " in order to ensure each subgraph has an output node, required for fx graph partitioning."
                " This can be the case when a subgraph would have only contained nodes performing inplace mutation,"
                " and returning no logical outputs. This should not be a problem, unless it results in too few graph"
                " partitions for optimal DDP performance."
            )

        # 尝试导入 tabulate 库，用于将桶分配信息以表格形式显示
        try:
            from tabulate import tabulate

            # 记录调试信息，显示 DDPOptimizer 的桶分配情况
            log.debug(
                "\nDDPOptimizer produced the following bucket assignments:\n%s",
                tabulate(rows, headers=headers, tablefmt="simple_grid"),
            )

            # 如果 extended_buckets 的长度大于零，则记录警告信息，显示扩展的桶以确保每个子图有输出节点的详细信息
            if len(extended_buckets):
                log.warning(
                    "DDPOptimizer extended these buckets to ensure per-subgraph output nodes:\n%s",
                    tabulate(
                        extended_buckets,
                        headers=("Index", "Extra Ops", "Extra Param Size (b)"),
                        tablefmt="simple_grid",
                    ),
                )
        except ImportError:
            # 若导入 tabulate 失败，则记录警告信息，提示用户安装 tabulate 库以显示 DDPOptimizer 的调试信息
            log.debug(
                "Please `pip install tabulate` in order to display ddp bucket sizes and diagnostic information."
            )
    else:
        # 若 rows 的长度为零，则记录调试信息，说明 DDPOptimizer 未捕获任何参数且未分割该图
        log.debug("DDPOptimizer captured no parameters and did not split this graph.")
# 检查图中是否存在高阶操作
def has_higher_order_op(gm):
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点操作是否为 "get_attr"
        if node.op == "get_attr":
            # 获取节点对应的属性
            maybe_param = getattr(gm, node.target)
            # 如果属性是 torch.fx.GraphModule 类型，则存在高阶操作
            if isinstance(maybe_param, torch.fx.GraphModule):
                return True
    # 如果没有找到高阶操作，则返回 False
    return False


# 使用用户提供的编译器编译每个分区子模块
class SubmodCompiler(torch.fx.interpreter.Interpreter):
    def __init__(self, module, compiler, fake_mode):
        super().__init__(module)
        self.compiler = compiler  # 存储用户提供的编译器
        self.fake_mode = fake_mode  # 存储是否处于 fake 模式

    def compile_submod(self, input_mod, args, kwargs):
        """
        编译子模块，
        使用一个包装器确保其输出始终是一个元组，
        这是 AotAutograd 基于编译器所需的
        """
        assert len(kwargs) == 0, "We assume only args for these modules"

        class WrapperModule(torch.nn.Module):
            def __init__(self, submod, unwrap_singleton_tuple):
                super().__init__()
                self.submod = submod  # 存储子模块
                self.unwrap_singleton_tuple = unwrap_singleton_tuple  # 存储是否解开单例元组的标志

            def forward(self, *args):
                x = self.submod(*args)
                # TODO(whc)
                # 由于某些原因，如果我将一个节点拆分到一个子模块中，则需要进行 isinstance 检查
                # 即使我在这些情况下据说将输出包装在元组中，真正编译的模块仍然返回张量
                if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                    return x[0]  # 如果需要解开单例元组并且 x 是元组或列表，则返回第一个元素
                return x  # 否则返回 x

        unwrap_singleton_tuple = False
        for sn in input_mod.graph.nodes:
            if sn.op == "output":
                if not isinstance(sn.args[0], tuple):
                    unwrap_singleton_tuple = True
                    sn.args = (sn.args,)

        input_mod.recompile()  # 重新编译子图
        input_mod.compile_subgraph_reason = GraphCompileReason(
            "DDPOptimizer intentional graph-break (See Note [DDPOptimizer])."
            " Set `torch._dynamo.config.optimize_ddp = False` to disable.",
            [
                # 几乎无法获取真实的堆栈跟踪，并且相当冗长。
                traceback.FrameSummary(__file__, 0, DDPOptimizer),
            ],
        )

        # 创建包装器模块
        wrapper = WrapperModule(
            self.compiler(input_mod, args),  # 使用编译器编译输入模块和参数
            unwrap_singleton_tuple,  # 传递是否解开单例元组的标志
        )
        return wrapper

    # 注意：
    #
    # 当前分布式实现在处理 fake 张量时可能会有些混乱。
    # 这些代码路径在运行时和编译时都是共享的。fake_mode 的存在，读取自 fake 张量输入，
    # 决定了我们的操作方式。
    #
    # 一些需要注意的事情：
    #
    # 1）我们使用一个真实模块来调用 `compile_submod`。其输出将被存储
    # 在图上通过 `self.module.add_submodule(n.target, compiled_submod_real)` 添加子模块。
    #
    # 2) 当运行一个以 call_module 为目标的节点时，如果我们处于 fake_mode 下，我们会对从
    # self.fetch_attr(n.target) 获取到的模块进行伪装。无论是否处于 fake_mode，我们都会执行它。
    #
    # 3) 编译时应始终存在伪张量。
    #
    # 4) 在运行时，伪张量永远不应存在。
    #
    # 5) 我们最终得到一种编译模式，该模式接受一个真实的子模块和伪张量，以匹配 aot_autograd 预期的行为。参见注释: [Fake Modules and AOTAutograd]
# 定义一个名为 DDPOptimizer 的类，用于在使用 DistributedDataParallel (DDP) 包装模型时进行优化。
class DDPOptimizer:
    """Note [DDPOptimizer]
    DDPOptimizer applies when dynamo compiles models wrapped in DistributedDataParallel (DDP),
    breaking the dynamo graph into chunks to compile separately, with the breaks aligning to
    the boundaries of gradient-allreduce buckets chosen by DDP.

    Background/Motivation
     - DDP uses allreduce collectives to synchronize partial gradients computed on different workers
     - DDP groups gradient allreduces into 'buckets' to optimize communication efficiency of all-reduce
     - Parameters grouped into buckets are assumed to be adjacent in time, so they become ready
       at around the same time during backward and thus can share the same allreduce efficiently
     - Allreduces must overlap with backward compute for optimal training performance
     - DDP schedules allreduces using 'hooks' fired from the c++ autograd engine in pytorch, which
       operates when individual grads become 'ready'
     - Dynamo+AOTAutograd produces a single fused graph that runs 'atomically' from the perspective of the
       autograd engine, such that all gradients become 'ready' at the same time.  Hooks fire after the whole
       fused backward function executes, preventing any overlap of compute and communication

    Algorithm
     - DDPOptimizer starts off with an FX graph traced by dynamo which represents forward.  It can traverse
       this graph in reverse order to determine the true order that gradients will become ready during backward.
     - Parameter sizes are counted in reverse order, up to a bucket size limit, at which point a new bucket is started
       and a graph break introduced
     - Each of the subgraphs is compiled by the compiler provided to dynamo by the user, and then fused back together
       into an outer module that is returned to the user
    """
    # 分布式数据并行优化器 DDPOptimizer 的初始化函数
    """
    Notes
     - It would be better to enforce (by adding an API to DDP) that the bucket splits chosen here are used by DDP,
       and that DDP does not need to detect or optimize bucket order by observing execution at runtime, as it does
       in eager.
     - If Dynamo can't capture a whole graph for the portion of the model wrapped by DDP, this algorithm will currently
       produce splits that do not necessarily align with the buckets used by DDP.  This should result in performance
       degradation approaching the baseline case where graph-splits are not used, but not worse.
     - If the backend compiler fails to compile a single subgraph, it will execute eagerly despite the rest of the
       subgraphs being compiled
     - DDP has a 'parameters_and_buffers_to_ignore' field, which DDPOptimizer attempts to honor by reading markers
       left by DDP on individual parameters.  In cases where other transformations, such as reparameterization, are
       also used, the ignore markers could be lost.  If DDPOptimizer fails to ignore a parameter ignored by DDP,
       it is not catastrophic but could impact performance by choosing sub-optimal bucket splits.
     - DDPOptimizer always ignores all buffers, regardless of their ignore flag, since buffers do not require gradients,
       and therefore aren't allreduced by DDP.  (They are broadcast during forward, but this is not covered by
       DDPOptimizer)
    
    Debugging
     - Generally, it is easiest to debug DDPOptimizer in a single process program, using pdb.
     - In many cases, the log messages are helpful (they show bucket size assignments)-
       just set TORCH_LOGS env to include any of 'dynamo', 'distributed', or 'dist_ddp'.
     - See `benchmarks/dynamo/distributed.py` for a simple harness that will run a toy model or a torchbench model
       in a single process (or with torchrun, in multiple processes)
    
    Args:
        bucket_bytes_cap (int): Controls the size of buckets, in bytes, used to determine graphbreaks.  Should be
            set to match the equivalent parameter on the original DDP module.
    
        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.
    
        first_bucket_cap (int): Controls the size of the first bucket.  Should match DDP's first bucket cap.  DDP
            special-cases the first bucket size since it is sometimes optimal to start a small allreduce early.
    """
    def __init__(
        self,
        bucket_bytes_cap: int,
        backend_compile_fn,
        first_bucket_cap: Optional[int] = None,
    ):
        # 如果指定了首个桶的容量，则使用指定的值
        if first_bucket_cap is not None:
            self.first_bucket_cap = first_bucket_cap
        # 如果 Torch 分布式可用，使用默认的首个桶容量（来自 C10D 库）
        elif torch.distributed.is_available():
            self.first_bucket_cap = torch.distributed._DEFAULT_FIRST_BUCKET_BYTES
        # 否则使用传入的桶容量
        else:
            self.first_bucket_cap = bucket_bytes_cap

        # 设置桶的总字节容量
        self.bucket_bytes_cap = bucket_bytes_cap
        # 断言首个桶的容量不大于总桶容量，以确保尽快初始化通信
        assert (
            self.first_bucket_cap <= self.bucket_bytes_cap
        ), "First bucket should be smaller/equal to other buckets to get comms warmed up ASAP"

        # 设置后端编译函数
        self.backend_compile_fn = backend_compile_fn

    # 检查参数是否被忽略
    def _ignore_parameter(self, parameter):
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    # 将参数添加到指定桶中
    def add_param(self, bucket, param, name):
        # 增加桶的大小，计算参数的字节大小并加到桶的 size 中
        bucket.size += param.untyped_storage().nbytes()
        # 记录参数的名字到桶的 params 列表中
        bucket.params.append(name)
        # 记录参数的 ID 到桶的 param_ids 列表中
        bucket.param_ids.append(id(param))

    # 将模块的参数添加到指定桶中
    def add_module_params_to_bucket(self, mod, bucket, processed_modules, prefix):
        # 将当前模块标记为已处理
        processed_modules.add(mod)
        # 遍历模块中的命名参数
        for name, param in mod.named_parameters():
            # 如果参数需要梯度并且未被忽略，则添加到指定桶中
            if param.requires_grad and not self._ignore_parameter(param):
                self.add_param(bucket, param, f"{prefix}_{name}")

    # 将参数节点添加到指定桶中
    def add_param_args(self, bucket, node):
        # 遍历节点的参数
        for arg in node.args:
            # 如果参数不是 Torch 的 FX 节点，则跳过
            if not isinstance(arg, torch.fx.node.Node):
                continue
            # 如果参数操作不是 "placeholder"，则跳过
            if arg.op != "placeholder":
                continue
            # 获取参数的示例值
            param = arg.meta["example_value"]
            # 如果参数是需要梯度的参数且未被忽略，则添加到指定桶中
            if (
                isinstance(param, torch.nn.Parameter)
                and param.requires_grad
                and not self._ignore_parameter(param)
            ):
                self.add_param(bucket, param, arg.target)
```