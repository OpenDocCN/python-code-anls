# `.\pytorch\torch\distributed\_tools\memory_tracker.py`

```py
# mypy: allow-untyped-defs
import operator  # 导入 operator 模块，用于操作符相关功能
import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认值字典
from itertools import chain  # 导入 chain 函数，用于串联迭代器
from typing import Any, Callable, Dict, List, no_type_check, Sequence, TYPE_CHECKING  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块

from torch.utils._python_dispatch import TorchDispatchMode  # 导入 TorchDispatchMode 类


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle  # 类型检查时导入 RemovableHandle 类


BYTES_PER_MB = 1024 * 1024.0  # 定义常量 BYTES_PER_MB，表示每兆字节的字节数


class MemoryProfileDispatchMode(TorchDispatchMode):
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""

    def __init__(self, memory_tracker) -> None:
        self.memory_tracker = memory_tracker  # 初始化 MemoryProfileDispatchMode 实例的 memory_tracker 属性

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)  # 调用给定函数 func，并传入 args 和 kwargs 参数
        if func == torch.ops.aten.detach.default:
            return rs  # 如果函数是 torch.ops.aten.detach.default，则直接返回 rs
        
        func_name: str = (
            self.memory_tracker._cur_module_name
            + "."
            + func.__name__
            + "_"
            + str(self.memory_tracker._operator_names[func.__name__])
        )  # 构造函数名称字符串，包括当前模块名、函数名和操作数统计索引

        self.memory_tracker._operator_names[func.__name__] = (
            self.memory_tracker._operator_names[func.__name__] + 1
        )  # 更新函数名对应的操作数统计索引

        self.memory_tracker._record_memory_stats(func_name)  # 调用 MemoryTracker 的 _record_memory_stats 方法记录内存统计信息

        return rs  # 返回函数执行结果


class MemoryTracker:
    """
    Collect and plot the memory stats at operator level.

    Includes ``memories_allocated``, ``memories_active`` and ``memories_reserved``.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> # xdoctest: +SKIP(failing)
        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """

    def __init__(self) -> None:
        torch._C._log_api_usage_once("torch.distributed.memory_tracker")  # 记录 API 使用情况到日志
        self._hooks: List[RemovableHandle] = []  # 初始化 _hooks 属性为空列表，用于存放可移除的钩子
        self._operator_names: Dict[str, int] = defaultdict(int)  # 初始化 _operator_names 属性为默认值为整数的字典
        self.memories_allocated: Dict[int, Dict[str, float]] = defaultdict()  # 初始化 memories_allocated 属性为空的字典
        self.memories_active: Dict[int, Dict[str, float]] = defaultdict()  # 初始化 memories_active 属性为空的字典
        self.memories_reserved: Dict[int, Dict[str, float]] = defaultdict()  # 初始化 memories_reserved 属性为空的字典
        self._markers: Dict[str, int] = defaultdict(int)  # 初始化 _markers 属性为空的字典
        self._cur_module_name: str = ""  # 初始化 _cur_module_name 属性为空字符串
        self._op_index: int = 0  # 初始化 _op_index 属性为整数 0
        self._num_cuda_retries: int = 0  # 初始化 _num_cuda_retries 属性为整数 0

    @no_type_check
    def start_monitor(self, root_module: nn.Module) -> None:
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``.

        This enables operator level memory stats can be tracked during module runtime.
        """
        # 清除当前状态，准备开始监视
        self._clear_state()
        # 将根模块标记为根节点以便跟踪
        root_module.__setattr__("_memory_tracker_is_root", True)
        # 遍历所有模块，注册钩子
        for name, m in root_module.named_modules():
            if m is not root_module:
                # 将非根模块标记为非根节点
                m.__setattr__("_memory_tracker_is_root", False)
            # 如果模块名包含特定字符串，则跳过不支持钩子的模块
            if ".fused_proxy_grouped_embedding_bag" in name:
                continue
            # 注册前向预钩子和后向钩子
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            # 暂时移除后向钩子，因为它无法处理不规则张量
            # h3 = m.register_backward_hook(self._create_backward_hook(name))
            # 将新注册的钩子添加到列表中
            self._hooks.extend([h1, h2])
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        # 确保当前没有处于监视模式
        assert getattr(self, "profile_mode", None) is None
        # 进入内存分析调度模式
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    @no_type_check
    def stop(self) -> None:
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop tracking memory stats at operator level.

        Get some aggregated stats when the memory_tracker() is enabled, like cuda ``num_alloc_retries``.
        """
        # 获取CUDA内存统计信息中的重试次数
        self._num_cuda_retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)

        # 移除所有注册的钩子
        for h in self._hooks:
            h.remove()
        # 清空钩子列表
        self._hooks.clear()
        # 确保当前在监视模式下
        assert getattr(self, "profile_mode", None) is not None
        # 退出内存分析调度模式
        self.profile_mode.__exit__(None, None, None)
        # 清空监视模式
        self.profile_mode = None

    @no_type_check
    # 定义一个方法 summary，用于打印出生成内存最多的顶级操作符

    def summary(self, top: int = 20) -> None:
        """
        Print out the top operators that generate the most memories.

        The number of the top operators can be configured.
        """
        
        # 创建一个默认值为浮点型的字典 op_diff，用于存储操作符及其生成的内存差异
        op_diff: Dict[str, float] = defaultdict(float)
        
        # 获取第一个操作符的名称和已分配内存
        op_name, previous_allocated_memory = self.memories_allocated[0]
        
        # 遍历从第二个到倒数第二个操作符的索引
        for i in range(1, self._op_index):
            # 获取当前操作符的名称和已分配内存
            op_name, current_allocated_memory = self.memories_allocated[i]
            
            # 计算当前操作符的内存差异并存储到 op_diff 中
            op_diff[op_name] = current_allocated_memory - previous_allocated_memory
            
            # 更新 previous_allocated_memory 为当前操作符的已分配内存，为下一个迭代做准备
            previous_allocated_memory = current_allocated_memory

        # 打印分隔线
        print("------------------------------------------------")
        
        # 打印 CUDA 重试的次数
        print(f"The number of cuda retries are: {self._num_cuda_retries}")
        
        # 打印生成内存最多的前 top 个操作符
        print(f"Top {top} ops that generates memory are:")
        
        # 对 op_diff 中的操作符按照生成内存降序排序，并打印出前 top 个操作符及其生成的内存
        for k, v in sorted(op_diff.items(), key=operator.itemgetter(1), reverse=True)[:top]:
            print(f"{k}: {v}MB")
        
        # 打印分隔线
        print("------------------------------------------------")

    # 定义一个方法 show_traces，用于展示内存跟踪图像
    @no_type_check
    def show_traces(self, path: str = "") -> None:
        # 导入 matplotlib 库
        import matplotlib.pyplot as plt

        # 定义一个内部函数 _plot_figure，用于绘制图像
        def _plot_figure(x, y_values, labels):
            # 计算 y 轴值的最小值和最大值
            min_val = min(list(chain(*y_values))) * 0.999
            max_val = max(list(chain(*y_values))) * 1.001
            
            # 创建一个新的图像
            plt.figure()
            
            # 对每组 y 值进行绘制
            for y, label in zip(y_values, labels):
                plt.plot(x, y, label=label)
            
            # 设置 x 轴和 y 轴的标签
            plt.xlabel("# Operator Calls")
            plt.ylabel("Memory (MB)")
            
            # 添加图例
            plt.legend()
            
            # 对 self._markers 中的每个标记进行处理
            for marker_name, marker in self._markers.items():
                if marker_name == "fw_bw_boundary":
                    # 如果是 "fw_bw_boundary" 标记，绘制红色垂直线
                    plt.plot([marker, marker], [min_val, max_val], "r", lw=2, label=marker_name)
                else:
                    # 否则，绘制黑色实线
                    plt.plot([marker, marker], [min_val, max_val], "k-", lw=2, label=marker_name)

        # 如果给定了路径参数，则加载路径对应的数据
        if path != "":
            self.load(path)

        # 获取 self.memories_allocated、self.memories_active 和 self.memories_reserved 的内存数据
        y_1 = [gb for (name, gb) in self.memories_allocated.values()]
        y_2 = [gb for (name, gb) in self.memories_active.values()]
        y_3 = [gb for (name, gb) in self.memories_reserved.values()]
        
        # 创建 x 轴数据
        x = list(range(len(y_1)))
        
        # 绘制图像：绘制所有内存数据的图像
        _plot_figure(
            x,
            [list(y_1), list(y_2), list(y_3)],
            ["allocated_memory", "active_memory", "reserved_memory"],
        )
        
        # 绘制图像：仅绘制 allocated_memory 的图像
        _plot_figure(x, [list(y_1)], ["allocated_memory"])
        
        # 绘制图像：仅绘制 active_memory 的图像
        _plot_figure(x, [list(y_2)], ["active_memory"])
        
        # 绘制图像：仅绘制 reserved_memory 的图像
        _plot_figure(x, [list(y_3)], ["reserved_memory"])
    # 保存内存统计信息到指定路径，使用 pickle 序列化数据
    def save_stats(self, path: str) -> None:
        stats = {
            "memories_allocated": self.memories_allocated,
            "memories_active": self.memories_active,
            "memories_reserved": self.memories_reserved,
            "markers": self._markers,
            "num_alloc_retries": self._num_cuda_retries,
        }

        # 打开文件并使用 pickle 将统计信息写入文件
        with open(path, "wb") as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    # 从指定路径加载序列化的内存统计信息
    def load(self, path: str) -> None:
        # 打开文件并使用 pickle 加载内存统计信息
        with open(path, "rb") as f:
            stats = pickle.load(f)

        # 将加载的统计信息分配给对象的对应属性
        self.memories_allocated = stats["memories_allocated"]
        self.memories_active = stats["memories_active"]
        self.memories_reserved = stats["memories_reserved"]
        self._markers = stats["markers"]
        self._num_cuda_retries = stats["num_alloc_retries"]

    # 创建前向传播的钩子函数，用于在前向传播开始时插入标记
    def _create_pre_forward_hook(self, name: str) -> Callable:
        """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            # 设置当前模块名称为给定名称加上 '.forward' 后缀
            self._cur_module_name = f"{name}.forward"
            # 如果模块具有 _memory_tracker_is_root 属性且为 True，则插入 'fw_start' 标记
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_start")

        return _pre_forward_hook

    # 创建后向传播的钩子函数，用于在前向传播和后向传播之间的边界插入 'fw_bw_boundary' 标记
    def _create_post_forward_hook(self, name: str) -> Callable:
        """Insert the marker 'fw_bw_boundary' at the boundary of forward and backward pass."""

        def _post_forward_hook(
            module: nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ) -> None:
            # 如果模块具有 _memory_tracker_is_root 属性且为 True，则插入 'fw_bw_boundary' 标记
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_bw_boundary")

        return _post_forward_hook

    # 创建后向传播的钩子函数，用于在反向传播时设置当前模块名称
    def _create_backward_hook(self, name: str) -> Callable:
        """Insert the current module name with backward prefix for the operator name."""

        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            # 设置当前模块名称为给定名称加上 '.backward' 后缀
            self._cur_module_name = f"{name}.backward"

        return _backward_hook

    # Decorator: no_type_check decorator is used here, but its implementation is not provided in this snippet.
    def _record_memory_stats(self, fn_name: str) -> None:
        """
        记录当前分配的内存、当前活跃的内存和当前保留的内存。

        内存统计字典以 `self._op_index` 作为索引。
        """
        # 获取当前分配的内存并转换为 MB
        memory_allocated: float = torch.cuda.memory_allocated() / BYTES_PER_MB
        # 获取当前保留的内存并转换为 MB
        memory_reserved: float = torch.cuda.memory_reserved() / BYTES_PER_MB
        # 获取当前活跃的内存并转换为 MB
        memory_active: float = (
            torch.cuda.memory_stats().get("active_bytes.all.current", 0) / BYTES_PER_MB
        )
        # 将函数名和分配的内存记录到字典中
        self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
        # 将函数名和保留的内存记录到字典中
        self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
        # 将函数名和活跃的内存记录到字典中
        self.memories_active[self._op_index] = (fn_name, memory_active)
        # 自增操作索引 `_op_index`
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        """
        设置标记点的 x 轴数值。

        根据已记录内存数据的条目数为标记值。
        """
        # 获取已记录内存数据的条目数作为标记值
        marker_val = len(self.memories_allocated.values())
        # 将标记名称和标记值添加到 `_markers` 字典中
        self._markers[marker_name] = marker_val

    def _clear_state(self) -> None:
        """
        在调用 `start_monitor()` 时清除状态。
        """
        # 清空操作符名称列表
        self._operator_names.clear()
        # 清空分配内存记录字典
        self.memories_allocated.clear()
        # 清空活跃内存记录字典
        self.memories_active.clear()
        # 清空保留内存记录字典
        self.memories_reserved.clear()
        # 清空标记字典
        self._markers.clear()
        # 清空当前模块名称
        self._cur_module_name = ""
        # 重置操作索引 `_op_index`
        self._op_index = 0
        # 重置 CUDA 重试次数
        self._num_cuda_retries = 0
```