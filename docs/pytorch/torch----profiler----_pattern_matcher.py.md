# `.\pytorch\torch\profiler\_pattern_matcher.py`

```
# mypy: allow-untyped-defs
# 导入所需的模块
import json  # 导入 JSON 模块
import math  # 导入数学模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
from typing import Dict, List, Optional, Set  # 导入类型提示模块

import torch  # 导入 PyTorch 模块
import torch.utils.benchmark as benchmark  # 导入 PyTorch 的 benchmark 模块
from torch._C._profiler import (  # 导入 PyTorch C++ 层的 profiler 模块
    _EventType,
    _ExtraFields_PyCall,
    _ExtraFields_PyCCall,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
)
from torch.profiler import profile  # 导入 PyTorch 的 profiler 模块
from torch.profiler._utils import (  # 导入 PyTorch profiler 模块的辅助工具函数
    index_of_first_match,
    traverse_bfs,
    traverse_dfs,
)


class Pattern:
    """
    Base class for all patterns, subclass this class and implement match()
    to define custom patterns.

    In subclass, define description and skip property.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        # 初始化 Pattern 类的实例
        self.prof = prof  # 设置 profiler 对象
        self.should_benchmark = should_benchmark  # 是否应进行基准测试的标志
        self.name = "Please specify a name for pattern"  # 模式名称，默认值
        self.description = "Please specify a description for pattern"  # 模式描述，默认值
        self.url = ""  # 相关 URL，默认为空字符串
        # 断言检查 profiler 对象和 kineto_results 不为空
        assert prof.profiler is not None and prof.profiler.kineto_results is not None
        # 获取实验性事件树
        self.event_tree = prof.profiler.kineto_results.experimental_event_tree()
        self.tid_root: Dict[int, List[_ProfilerEvent]] = {}
        # 将事件按线程 ID 分类存储在 tid_root 字典中
        for event in self.event_tree:
            self.tid_root.setdefault(event.start_tid, []).append(event)

    @property
    def skip(self):
        # skip 属性，用于指示是否跳过当前模式的匹配
        return False

    def report(self, event: _ProfilerEvent):
        # 生成事件的报告消息
        msg = (
            f"{self.description}\n[Source Code Location] {source_code_location(event)}"
        )
        return msg

    def eventTreeTraversal(self):
        """
        Traverse the event tree and yield all events.
        Override this method in subclass to customize the traversal.
        """
        # 遍历事件树并生成所有事件
        yield from traverse_dfs(self.event_tree)

    def summary(self, events: List[_ProfilerEvent]):
        # 生成模式匹配的摘要信息
        default_summary = f"{self.name}: {len(events)} events matched."
        if self.should_benchmark:
            # 如果需要进行基准测试，返回基准测试的摘要信息
            return (
                self.benchmark_summary(events)
                if hasattr(self, "benchmark")  # 检查是否定义了 benchmark 方法
                else default_summary
            )
        return default_summary
    def benchmark_summary(self, events: List[_ProfilerEvent]):
        def format_time(time_ns: int):
            unit_lst = ["ns", "us", "ms"]
            for unit in unit_lst:
                if time_ns < 1000:
                    return f"{time_ns:.2f} {unit}"
                time_ns //= 1000
            return f"{time_ns:.2f} s"

        assert hasattr(self, "benchmark"), "Please implement benchmark()"
        shapes_factor_map = self.benchmark(events)  # type: ignore[attr-defined]
        original_time = sum(event.duration_time_ns for event in events)
        new_time = sum(
            shapes_factor_map[input_shapes(event)] * event.duration_time_ns
            for event in events
        )
        return (
            f"{self.name}: {len(events)} events matched. "
            f"Total Estimated Speedup: {format_time(original_time - new_time)} ({round(original_time/new_time, 2)}X)"
        )


        """
        Return True if the event matches the pattern.
        This method should be overriden in subclass.
        """
        raise NotImplementedError


        if self.skip:
            return []
        matched_events = []
        for event in self.eventTreeTraversal():
            if self.match(event):
                matched_events.append(event)
        return matched_events


        while event.parent:
            event = event.parent
        return event


        if event.parent:
            children = event.parent.children
        else:
            children = self.tid_root[event.start_tid]
        index = children.index(event)
        return children[:index], children[index + 1 :]


        _, next_events = self.siblings_of(event)
        return next_events[0] if next_events else None


        prev_events, _ = self.siblings_of(event)
        return prev_events[-1] if prev_events else None


        if not event:
            return None
        while event.parent and not predicate(event):
            event = event.parent
        return event
# Patterns

# 定义一个名字匹配模式的类，继承自Pattern类
class NamePattern(Pattern):
    def __init__(self, prof: profile, name: str, should_benchmark: bool = False):
        # 调用父类构造函数初始化
        super().__init__(prof, should_benchmark)
        # 设置描述信息，表示匹配到名称事件
        self.description = f"Matched Name Event: {name}"
        # 设置名称属性
        self.name = name

    # 匹配方法，判断事件名称是否包含指定名称
    def match(self, event: _ProfilerEvent):
        return re.search(self.name, event.name) is not None


# 定义一个额外的CUDA复制模式的类，继承自Pattern类
class ExtraCUDACopyPattern(Pattern):
    """
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to GPU.
    example: torch.zeros((100, 100)).to("cuda")

    Pattern:
    build-in method                 |build-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        # 调用父类构造函数初始化
        super().__init__(prof, should_benchmark)
        # 设置模式名称
        self.name = "Extra CUDA Copy Pattern"
        # 设置描述信息，说明匹配到CPU张量并立即移动到GPU
        self.description = "Filled a CPU tensor and immediately moved it to GPU. Please initialize it on GPU."
        # 设置链接到PyTorch文档的URL
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device"
        # 初始化操作集合，包含了在CPU上填充张量的操作
        self.init_ops = {
            "aten::fill_",
            "aten::zero_",
            "aten::normal_",
            "aten::uniform_",
        }

    # skip属性的getter方法，根据记录的堆栈信息和形状信息判断是否跳过
    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes
    # 判断是否匹配特定的事件，用于分析和过滤事件流
    def match(self, event):
        # TODO: We should also check tensor identities
        # 检查事件名称是否为 "aten::to"，如果不是则返回 False
        if event.name != "aten::to":
            return False
        
        # 将当前事件保存为 to_event
        to_event = event
        
        # 如果当前事件没有子事件，则返回 False
        if not event.children:
            return False
        
        # 获取当前事件的最后一个子事件
        event = event.children[-1]
        
        # 检查子事件的名称是否为 "aten::_to_copy"，如果不是则返回 False
        if event.name != "aten::_to_copy":
            return False
        
        # 如果当前事件的最后一个子事件没有子事件，则返回 False
        if not event.children:
            return False
        
        # 获取当前事件的最后一个子事件的最后一个子事件
        event = event.children[-1]
        
        # 检查子事件的名称是否为 "aten::copy_"，如果不是则返回 False
        if event.name != "aten::copy_":
            return False
        
        # 检查 aten::copy_ 的前两个参数的数据类型是否相同
        dtypes = input_dtypes(event)
        
        # 如果参数个数小于 2，则返回 False
        if len(dtypes) < 2:
            return False
        
        # 如果第一个参数为 None 或者前两个参数的数据类型不相同，则返回 False
        if dtypes[0] is None or dtypes[0] != dtypes[1]:
            return False
        
        # 将事件回溯到最初的 to_event
        event = to_event
        
        # 上升一级
        event = event.parent
        
        # 如果事件为 None，则返回 False
        if event is None:
            return False
        
        # 在前一个叶子节点中检查是否存在 aten::fill_
        event = self.prev_of(event)
        
        # 如果前一个叶子节点为 None，则返回 False
        if event is None:
            return False
        
        # 循环向下查找子事件，直到找到最后一个子事件
        while event.children:
            event = event.children[-1]
            
            # 如果事件名称在 init_ops 中，则返回 True
            # 特例：aten::zero_ 是 fill_ 未调用的优化情况
            if event.name in self.init_ops:
                return True
        
        # 检查最终的事件名称是否在 init_ops 中，如果在则返回 True，否则返回 False
        return event.name in self.init_ops
        
        # TODO: Check if tensor is reused

    # 对一组事件进行基准测试，返回形状因子映射
    def benchmark(self, events: List[_ProfilerEvent]):
        # 初始化形状因子映射，每个形状对应初始值为 0.0
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        
        # 遍历形状因子映射中的每个形状
        for shape in shapes_factor_map:
            size = shape[0]
            
            # 创建一个计时器，测量 torch.ones(size).to("cuda") 的平均执行时间
            to_timer = benchmark.Timer(
                stmt='torch.ones(size).to("cuda")', globals={"size": size}
            )
            
            # 创建一个计时器，测量 torch.ones(size, device="cuda") 的平均执行时间
            de_timer = benchmark.Timer(
                stmt='torch.ones(size, device="cuda")', globals={"size": size}
            )
            
            # 计算 to_timer 和 de_timer 的平均执行时间比例，作为形状因子的值
            to_time = to_timer.timeit(10).mean
            de_time = de_timer.timeit(10).mean
            shapes_factor_map[shape] = de_time / to_time
        
        # 返回形状因子映射
        return shapes_factor_map
class ForLoopIndexingPattern(Pattern):
    """
    This pattern identifies if we use a for loop to index a tensor that
    can be vectorized.
    example:
    tensor = torch.empty((100, 100))
    for i in range(100):
        tensor[i] = i

    Pattern:
    aten::select | ... | aten::select | ... (Repeat)

    Algorithm:
    We start at node aten::select, and we check if we can find this alternating patterns.
    We also keep a dictionary to avoid duplicate match in the for loop.
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "For Loop Indexing Pattern"  # 设定模式名称为 "For Loop Indexing Pattern"
        self.description = "For loop indexing detected. Vectorization recommended."  # 描述检测到的情况为使用了 for 循环进行索引，建议进行向量化处理
        self.visited: Set[int] = set()  # 初始化一个集合，用于记录已访问过的事件 ID

    def eventTreeTraversal(self):
        """
        We need to use BFS traversal order to avoid duplicate match.
        """
        yield from traverse_bfs(self.event_tree)  # 使用广度优先搜索遍历事件树，并返回遍历结果

    def match(self, event: _ProfilerEvent):
        if event.name != "aten::select":  # 如果事件名称不是 "aten::select"，则不匹配当前模式，返回 False
            return False
        if event.id in self.visited:  # 如果事件 ID 已经在已访问集合中，表示已经处理过，返回 False
            return False
        repeat_count = 1  # 初始化重复计数为 1
        _, next = self.siblings_of(event)  # 获取当前事件的兄弟事件列表
        if len(next) <= 1:  # 如果兄弟事件列表长度小于等于 1，返回 False
            return False

        # Custom event list matching
        def same_ops(list1, list2):
            if len(list1) != len(list2):  # 如果两个列表长度不相等，返回 False
                return False
            for op1, op2 in zip(list1, list2):  # 遍历两个列表，比较每个操作的名称是否相同
                if op1.name != op2.name:  # 如果有操作的名称不相同，返回 False
                    return False
            return True

        # Record the ops between two aten::select
        next_select_idx = index_of_first_match(next, lambda e: e.name == "aten::select")  # 找到下一个 "aten::select" 的位置索引
        if next_select_idx is None:  # 如果找不到下一个 "aten::select"，返回 False
            return False
        indexing_ops = [event] + next[:next_select_idx]  # 记录两个 "aten::select" 之间的操作列表
        next = next[len(indexing_ops) - 1 :]  # 更新剩余的事件列表
        for i in range(0, len(next), len(indexing_ops)):
            if same_ops(indexing_ops, next[i : i + len(indexing_ops)]):  # 如果剩余的列表中有与操作列表相同的部分
                repeat_count += 1  # 增加重复计数
                self.visited.add(next[i].id)  # 将当前事件 ID 添加到已访问集合中
            else:
                break  # 否则退出循环
        return repeat_count >= 10  # 返回是否重复次数大于等于 10


class FP32MatMulPattern(Pattern):
    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "FP32 MatMul Pattern"  # 设定模式名称为 "FP32 MatMul Pattern"
        self.description = (
            "You are currently using GPU that supports TF32. "
            "Please enable TF32 by setting 'torch.backends.cuda.matmul.allow_tf32 = True'"
        )  # 描述当前情况是使用了支持 TF32 的 GPU，建议通过设置 'torch.backends.cuda.matmul.allow_tf32 = True' 来启用 TF32
        self.url = "https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"  # 相关文档链接地址

    @property
    def skip(self):
        if torch.version.hip is not None:  # 如果当前使用的是 HIP 版本的 Torch，则不跳过
            has_tf32 = False
        else:
            # Anything less than sm_80 is not Ampere which doesn't support TF32
            has_tf32 = all(int(arch[3:]) >= 80 for arch in torch.cuda.get_arch_list())  # 检查所有 CUDA 架构版本是否都大于等于 80
        return has_tf32 is False or super().skip or not self.prof.record_shapes  # 如果没有 TF32 支持或者需要跳过，或者不记录形状信息，则返回 True
    # 定义一个方法用于匹配给定的事件是否符合特定条件
    def match(self, event: _ProfilerEvent):
        # 如果事件的标签不是 TorchOp，直接返回 False
        if event.tag != _EventType.TorchOp:
            return False
        # 确保事件的额外字段属于 _ExtraFields_TorchOp 类型
        assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
        # 如果事件的名称是 "aten::mm"
        if event.name == "aten::mm":
            # 检查额外字段中的 allow_tf32_cublas 是否为 False，如果是则返回 True
            if event.extra_fields.allow_tf32_cublas is False:
                return True
        # 默认返回 False
        return False

    # 定义一个方法用于报告描述信息，直接返回类的描述信息
    def report(self, event: _ProfilerEvent):
        return self.description

    # 定义一个方法用于对一组事件进行基准测试
    def benchmark(self, events: List[_ProfilerEvent]):
        # 创建一个字典，将每个输入形状映射到初始值为 0.0 的因子
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        # 遍历形状因子映射中的每个形状
        for shape in shapes_factor_map:
            # 在 CUDA 设备上生成随机张量 matrixA 和 matrixB，数据类型为 float32
            matrixA = torch.randn(shape[0], device="cuda", dtype=torch.float32)
            matrixB = torch.randn(shape[1], device="cuda", dtype=torch.float32)
            # 创建一个针对 fp32 运行时间的计时器
            fp32_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            # 创建一个针对 tf32 运行时间的计时器，并设置相应的运行环境
            tf32_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                setup="torch.backends.cuda.matmul.allow_tf32 = True",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            # 禁用 tf32 运算
            torch.backends.cuda.matmul.allow_tf32 = False
            # 计算 fp32 运行时间的平均值
            fp32_time = fp32_timer.timeit(10).mean
            # 计算 tf32 运行时间的平均值
            tf32_time = tf32_timer.timeit(10).mean
            # 将形状因子映射中当前形状的值设置为 tf32 时间与 fp32 时间之比
            shapes_factor_map[shape] = tf32_time / fp32_time
        # 返回形状因子映射
        return shapes_factor_map
# 定义一个继承自Pattern类的优化器单张量模式类，用于识别是否使用了优化器的单张量版本
class OptimizerSingleTensorPattern(Pattern):
    """
    This pattern identifies if we are using the single-tensor version of an optimizer.
    example:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    By adding foreach=True to enable multi-tensor optimizer, we can gain speedup when
    the kernels are relatively small.

    Pattern:
    XXXXX: _single_tenser_<OPTIMIZER_NAME>

    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Optimizer Single Tensor Pattern"  # 设定模式名称
        self.optimizers_with_foreach = ["adam", "sgd", "adamw"]  # 包含支持foreach=True的优化器列表
        self.description = (
            "Deteced optimizer running with single tensor implementation. "  # 检测到使用单张量实现的优化器
            "Please enable multi tensor implementation by passing 'foreach=True' into optimizer."  # 请通过传入'foreach=True'来启用多张量实现
        )
        self.url = ""  # 相关文档链接为空字符串

    # 匹配函数，检查事件是否以"_single_tensor_<优化器名称>"结尾，表示使用了单张量版本的优化器
    def match(self, event: _ProfilerEvent):
        for optimizer in self.optimizers_with_foreach:
            if event.name.endswith(f"_single_tensor_{optimizer}"):
                return True
        return False


# 定义一个继承自Pattern类的同步数据加载器模式类，用于识别DataLoader是否同步加载数据
class SynchronizedDataLoaderPattern(Pattern):
    """
    This pattern identifies if we are using num_workers=0 in DataLoader.
    example:
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    Add num_workers=N to the arguments. N depends on system configuration.

    Pattern:
    dataloader.py(...): __iter__
        dataloader.py(...): _get_iterator
            NOT dataloader.py(...): check_worker_number_rationality

    Algorithm:
    If we don't see check_worker_number_rationality call in the dataloader __iter__,
    It is not an asynchronous dataloader.

    """

    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        self.name = "Synchronized DataLoader Pattern"  # 设定模式名称
        self.description = (
            "Detected DataLoader running with synchronized implementation. "  # 检测到使用同步实现的DataLoader
            "Please enable asynchronous dataloading by setting num_workers > 0 when initializing DataLoader."  # 请在初始化DataLoader时设置num_workers > 0以启用异步数据加载
        )
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#enable-async-data-loading-and-augmentation"  # 相关文档链接，指导如何启用异步数据加载和增强
        )
    # 定义一个方法，用于匹配给定的事件对象是否符合特定条件
    def match(self, event: _ProfilerEvent):
        # 定义一个内部函数，用于检查给定的函数名是否属于数据加载器相关函数
        def is_dataloader_function(name: str, function_name: str):
            # 判断给定的函数名是否以指定路径开始且以给定的函数名结尾
            return name.startswith(
                os.path.join("torch", "utils", "data", "dataloader.py")
            ) and name.endswith(function_name)

        # TODO: fixme! 由于函数名的生命周期问题，当事件是 PyCall 时，这个字段可能指向一个已释放的字符串。
        # 因此，在这种情况下，静默跳过以解除测试阻塞。
        try:
            # 尝试访问事件对象的 name 属性，检测是否会引发 UnicodeDecodeError 异常
            event.name
        except UnicodeDecodeError:
            # 如果捕获到异常，则返回 False
            return False

        # 如果事件的名称不是 "__iter__" 开头的数据加载器函数，则返回 False
        if not is_dataloader_function(event.name, "__iter__"):
            return False
        # 如果事件没有子节点，则返回 False
        if not event.children:
            return False
        # 将事件对象更新为其第一个子节点
        event = event.children[0]
        # 如果子节点的名称不是 "_get_iterator" 开头的数据加载器函数，则返回 False
        if not is_dataloader_function(event.name, "_get_iterator"):
            return False
        # 如果子节点没有子节点，则返回 False
        if not event.children:
            return False
        # 将事件对象更新为其第一个子节点
        event = event.children[0]
        # 返回子节点的名称不是 "check_worker_number_rationality" 开头的数据加载器函数的结果的逆
        return not is_dataloader_function(event.name, "check_worker_number_rationality")
        # TODO: 我们还应该检查加载器是否成为瓶颈。
# 定义一个继承自 Pattern 的类，用于检测在 zero_grad 中是否未将 grad 设置为 None 的模式
class GradNotSetToNonePattern(Pattern):
    """
    This pattern identifies if we are not setting grad to None in zero_grad.
    example:
    optimizer.zero_grad()
    By setting set_to_none=True, we can gain speedup

    Pattern:
    XXXXX: _zero_grad
        NOT aten::zeros
            aten::zero_

    aten::zero_ is called on each parameter in the model.
    We also want to make sure it is not called by aten::zeros.

    Algorithm:
    String match
    """

    # 初始化方法，接受 prof 和 should_benchmark 两个参数
    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        # 模式的名称
        self.name = "Gradient Set To Zero Instead of None Pattern"
        # 描述信息，提醒用户将 zero_grad 中的 grad 设置为 None
        self.description = (
            "Detected gradient set to zero instead of None. "
            "Please add 'set_to_none=True' when calling zero_grad()."
        )
        # 相关链接，指导用户如何进行优化
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#disable-gradient-calculation-for-validation-or-inference"
        )

    # 匹配方法，判断是否匹配到设置 grad 为 None 的模式
    def match(self, event: _ProfilerEvent):
        # 如果事件名称不以 ": zero_grad" 结尾，则不匹配
        if not event.name.endswith(": zero_grad"):
            return False
        # 如果事件没有子事件，则不匹配
        if not event.children:
            return False

        # 遍历所有子事件
        for sub_event in traverse_dfs(event.children):
            # 如果子事件名称为 "aten::zero_"，并且其父事件名称不为 "aten::zeros"，则匹配成功
            if (
                sub_event.name == "aten::zero_"
                and sub_event.parent.name != "aten::zeros"
            ):
                return True
        # TODO: We should also check if the optimizer's numerical behavior will change.
        return False


# 定义一个继承自 Pattern 的类，用于检测在 Conv2d 后是否启用了 BatchNorm2d 中的偏置项
class Conv2dBiasFollowedByBatchNorm2dPattern(Pattern):
    """
    This pattern identifies if we are enabling bias in Conv2d which is followed by BatchNorm2d.
    Bias doesn't do anything when followed by batchnorm.
    Pattern:
    nn.Module: Conv2d            | nn.Module: BatchNorm2d
        ...
            aten::conv2d AND dtype of third argument is not null
    The third argument is the bias
    Algorithm:
    String match
    """

    # 初始化方法，接受 prof 和 should_benchmark 两个参数
    def __init__(self, prof: profile, should_benchmark: bool = False):
        super().__init__(prof, should_benchmark)
        # 模式的名称
        self.name = "Enabling Bias in Conv2d Followed By BatchNorm Pattern"
        # 描述信息，提醒用户在 Conv2d 中禁用偏置项
        self.description = "Detected bias enabled in Conv2d that is followed by BatchNorm2d. Please set 'bias=False' in Conv2d."
        # 相关链接，指导用户如何进行优化
        self.url = (
            "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html"
            "#disable-bias-for-convolutions-directly-followed-by-a-batch-norm"
        )

    # skip 属性，用于判断是否跳过该模式的检测
    @property
    def skip(self):
        return self.prof.record_shapes is False or super().skip
    # 匹配器方法，用于检查事件是否为 "aten::conv2d"
    def match(self, event: _ProfilerEvent):
        # 如果事件名不是 "aten::conv2d"，返回 False
        if event.name != "aten::conv2d":
            return False
        # 如果输入数据类型的数量小于 3，或者第三个输入数据类型为 None，则返回 False
        if len(input_dtypes(event)) < 3 or input_dtypes(event)[2] is None:
            return False
        # 如果以上条件均不满足，表示 bias=True
        # 查找事件直到找到名称以 "nn.Module: Conv2d" 开头的父级事件
        event = self.go_up_until(
            event, lambda e: e.name.startswith("nn.Module: Conv2d")
        )
        # 如果未找到符合条件的父级事件，返回 False
        if not event:
            return False
        # 获取找到的父级事件的下一个事件
        event = self.next_of(event)
        # 如果未找到下一个事件，返回 False
        if not event:
            return False
        # 检查下一个事件的名称是否以 "nn.Module: BatchNorm2d" 开头
        return event.name.startswith("nn.Module: BatchNorm2d")
class MatMulDimInFP16Pattern(Pattern):
    # 继承自 Pattern 类的矩阵乘法维度不对齐模式检测类
    def __init__(self, prof: profile, should_benchmark: bool = False):
        # 初始化方法，接受一个 profile 对象和一个布尔值 should_benchmark
        super().__init__(prof, should_benchmark)
        # 调用父类的初始化方法
        self.name = "Matrix Multiplication Dimension Not Aligned Pattern"
        # 模式名称为“矩阵乘法维度不对齐模式”
        self.description = "Detected matmul with dimension not aligned. Please use matmul with aligned dimension."
        # 模式描述，指示检测到乘法操作的维度不对齐
        self.url = "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-mixed-precision-and-amp"
        # 相关链接，指向 PyTorch 官方文档中的优化建议页面

    @property
    def skip(self):
        # 跳过属性的计算方法
        return not self.prof.with_stack or not self.prof.record_shapes
        # 若 profile 对象未启用堆栈信息或形状记录，则跳过

    def match(self, event: _ProfilerEvent):
        # 匹配方法，用于判断是否符合模式的条件
        def mutiple_of(shapes, multiple):
            # 内部辅助函数，检查所有形状的最后两个维度是否均为 multiple 的倍数
            return all(dim % multiple == 0 for shape in shapes for dim in shape[-2:])

        if event.name not in ("aten::mm", "aten::bmm", "aten::addmm"):
            # 如果事件名称不在支持的矩阵乘法操作中，则不匹配
            return False
        if not input_dtypes(event):
            # 如果事件的输入数据类型未知，则不匹配
            return False
        arg_dtype = input_dtypes(event)[0]
        if arg_dtype in (torch.bfloat16, torch.half) and not mutiple_of(
            input_shapes(event), 8
        ):
            # 如果输入数据类型为 torch.bfloat16 或 torch.half，并且维度不对齐于 8，则匹配
            return True
        return False

    def benchmark(self, events: List[_ProfilerEvent]):
        # 基准测试方法，对事件列表进行性能测试
        def closest_multiple(shapes, multiple):
            # 内部辅助函数，找到最接近的 multiple 的倍数
            return [multiple * math.ceil(shape / multiple) for shape in shapes]

        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        # 初始化形状因子映射，用于记录每种形状的因子

        for shape in shapes_factor_map:
            # 遍历形状因子映射中的每种形状
            matrixA = torch.randn(shape[0], device="cuda", dtype=torch.float16)
            matrixB = torch.randn(shape[1], device="cuda", dtype=torch.float16)
            # 随机生成两个 CUDA 设备上的 float16 类型的矩阵
            not_aligned_dim_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            # 创建未对齐维度的计时器对象，测量 torch.mm 的运行时间
            matrixA = torch.randn(
                closest_multiple(shape[0], 8), device="cuda", dtype=torch.float16
            )
            matrixB = torch.randn(
                closest_multiple(shape[1], 8), device="cuda", dtype=torch.float16
            )
            # 调整维度为最接近 8 的倍数的矩阵
            aligned_dim_timer = benchmark.Timer(
                stmt="torch.mm(matrixA, matrixB)",
                globals={"matrixA": matrixA, "matrixB": matrixB},
            )
            # 创建对齐维度的计时器对象，测量 torch.mm 的运行时间
            not_aligned_dim_time = not_aligned_dim_timer.timeit(10).mean
            aligned_dim_time = aligned_dim_timer.timeit(10).mean
            # 进行 10 次平均运行时间的测量
            shapes_factor_map[shape] = aligned_dim_time / not_aligned_dim_time
            # 计算并记录形状的性能因子

        return shapes_factor_map
        # 返回形状因子映射

def source_code_location(event: Optional[_ProfilerEvent]):
    # 源代码位置检索函数，根据事件追溯调用链来确定代码位置
    while event:
        if event.tag == _EventType.PyCall or event.tag == _EventType.PyCCall:
            assert isinstance(
                event.extra_fields, (_ExtraFields_PyCall, _ExtraFields_PyCCall)
            )
            # 断言事件的额外字段类型为 PyCall 或 PyCCall
            if not event.extra_fields.caller.file_name.startswith("torch" + os.sep):
                # 如果调用者的文件名不以 "torch/" 开头，则返回调用位置信息
                return f"{event.extra_fields.caller.file_name}:{event.extra_fields.caller.line_number}"
        event = event.parent
        # 获取父级事件，继续向上追溯

    return "No source code location found"
    # 如果未找到源代码位置，则返回默认信息
# 定义函数input_shapes，接收一个_ProfilerEvent对象作为参数
def input_shapes(event: _ProfilerEvent):
    # 断言event.extra_fields是_ExtraFields_TorchOp类型的实例
    assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
    # 返回一个元组，元组的每个元素是event.extra_fields.inputs中每个元素的sizes属性，如果没有则返回空元组
    return tuple(tuple(getattr(i, "sizes", ())) for i in event.extra_fields.inputs)


# 定义函数input_dtypes，接收一个_ProfilerEvent对象作为参数
def input_dtypes(event: _ProfilerEvent):
    # 断言event.extra_fields是_ExtraFields_TorchOp类型的实例
    assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
    # 返回一个元组，元组的每个元素是event.extra_fields.inputs中每个元素的dtype属性，如果没有则返回None
    return tuple(getattr(i, "dtype", None) for i in event.extra_fields.inputs)


# 定义函数report_all_anti_patterns，接收多个参数：
# prof: 分析器对象
# should_benchmark: 是否应该进行基准测试，默认为False
# print_enable: 是否打印报告信息，默认为True
# json_report_dir: 可选参数，JSON报告存储目录，默认为None
def report_all_anti_patterns(
    prof,
    should_benchmark: bool = False,
    print_enable: bool = True,
    json_report_dir: Optional[str] = None,
):
    # 初始化报告字典
    report_dict: Dict = {}
    # 初始化反模式列表
    anti_patterns = [
        ExtraCUDACopyPattern(prof, should_benchmark),
        # ForLoopIndexingPattern(prof, should_benchmark),
        FP32MatMulPattern(prof, should_benchmark),
        OptimizerSingleTensorPattern(prof, should_benchmark),
        SynchronizedDataLoaderPattern(prof, should_benchmark),
        GradNotSetToNonePattern(prof, should_benchmark),
        Conv2dBiasFollowedByBatchNorm2dPattern(prof, should_benchmark),
        MatMulDimInFP16Pattern(prof, should_benchmark),
    ]
    # 初始化已报告的集合
    reported = set()
    # 初始化总结列表
    summaries = []
    # 初始化消息列表，用于存储打印报告的信息
    message_list = [f"{'-'*40}TorchTidy Report{'-'*40}"]
    message_list.append("Matched Events:")

    # 遍历反模式列表
    for anti_pattern in anti_patterns:
        # 获得匹配的事件列表
        matched_events = anti_pattern.matched_events()
        # 如果没有匹配的事件，跳过当前反模式
        if not matched_events:
            continue
        # 生成反模式的总结信息，并添加到总结列表中
        summaries.append(anti_pattern.summary(matched_events))
        # 遍历每个匹配的事件
        for event in matched_events:
            # 生成报告消息
            report_msg = anti_pattern.report(event)
            # 如果报告消息不在已报告的集合中，将其添加到消息列表和已报告集合中
            if report_msg not in reported:
                message_list.append(report_msg)
                reported.add(report_msg)
                # 获取源代码位置和行号信息
                src_location, line_no = source_code_location(event).split(":")
                # 在报告字典中添加信息
                report_dict.setdefault(src_location, []).append(
                    {
                        "line_number": int(line_no),
                        "name": anti_pattern.name,
                        "url": anti_pattern.url,
                        "message": anti_pattern.description,
                    }
                )

    # 如果提供了json_report_dir参数
    if json_report_dir is not None:
        # 拼接JSON报告文件路径
        json_report_path = os.path.join(json_report_dir, "torchtidy_report.json")
        # 如果JSON报告文件已存在，读取并更新现有报告
        if os.path.exists(json_report_path):
            with open(json_report_path) as f:
                exisiting_report = json.load(f)
                exisiting_report.update(report_dict)
                report_dict = exisiting_report
        # 将报告字典写入JSON文件
        with open(json_report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

    # 添加总结信息到消息列表末尾
    message_list.append("Summary:")
    message_list += summaries
    message_list.append(f"{'-'*40}TorchTidy Report{'-'*40}")
    # 如果print_enable为True，则打印消息列表中的所有信息
    if print_enable:
        print("\n".join(message_list))
```