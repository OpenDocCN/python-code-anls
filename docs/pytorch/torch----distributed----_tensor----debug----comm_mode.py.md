# `.\pytorch\torch\distributed\_tensor\debug\comm_mode.py`

```
# 设置 mypy 选项以允许未类型化的函数定义
# 导入正则表达式模块
import re
# 导入默认字典模块
from collections import defaultdict
# 导入类型提示模块中的类型和字典
from typing import Any, Dict

# 导入 PyTorch 框架
import torch
# 从 torch.autograd.graph 模块中导入 register_multi_grad_hook 函数
from torch.autograd.graph import register_multi_grad_hook
# 从 torch.distributed._tensor.api 模块中导入 DTensor 类
from torch.distributed._tensor.api import DTensor
# 从 torch.nn.modules.module 模块中导入 register_module_forward_hook 和 register_module_forward_pre_hook 函数
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
# 从 torch.utils._python_dispatch 模块中导入 TorchDispatchMode 类
from torch.utils._python_dispatch import TorchDispatchMode
# 从 torch.utils._pytree 模块中导入 tree_flatten 函数
from torch.utils._pytree import tree_flatten
# 从 torch.utils.module_tracker 模块中导入 ModuleTracker 类
from torch.utils.module_tracker import ModuleTracker

# 导入 C10D 模块的功能操作符
funcol_native = torch.ops._c10d_functional
funcol_py = torch.ops.c10d_functional
funcol_autograd = torch.ops._c10d_functional_autograd
c10d_ops = torch.ops.c10d

# 定义本地到 Python 版本的功能映射字典
NATIVE_TO_PY_MAPPING = {
    funcol_native.all_gather_into_tensor: funcol_py.all_gather_into_tensor,
    funcol_native.all_gather_into_tensor_coalesced: funcol_py.all_gather_into_tensor_coalesced,
    funcol_native.all_reduce: funcol_py.all_reduce,
    funcol_native.all_reduce_coalesced: funcol_py.all_reduce_coalesced,
    funcol_native.all_to_all_single: funcol_py.all_to_all_single,
    funcol_native.broadcast: funcol_py.broadcast,
    funcol_native.reduce_scatter_tensor: funcol_py.reduce_scatter_tensor,
    funcol_native.reduce_scatter_tensor_coalesced: funcol_py.reduce_scatter_tensor_coalesced,
    funcol_autograd.all_to_all_single: funcol_py.all_to_all_single,
}

# 定义 C10D 集体操作的集合
c10d_collective_ops = {
    c10d_ops._allgather_base_,
    c10d_ops._reduce_scatter_base_,
    c10d_ops.allgather_,
    c10d_ops.allgather_coalesced_,
    c10d_ops.allgather_into_tensor_coalesced_,
    c10d_ops.allreduce_,
    c10d_ops.allreduce_coalesced_,
    c10d_ops.alltoall_,
    c10d_ops.alltoall_base_,
    c10d_ops.broadcast_,
    c10d_ops.gather_,
    c10d_ops.scatter_,
    c10d_ops.reduce_,
    c10d_ops.reduce_scatter_,
    c10d_ops.reduce_scatter_tensor_coalesced_,
}

# 定义 CommModeModuleTracker 类，继承自 ModuleTracker 类
class CommModeModuleTracker(ModuleTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the
    parameters and sharding information of a model at a module-level
    """

    # 初始化函数，扩展父类的功能
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化模块深度字典
        self.module_depth_dict = {}
        # 初始化模块参数字典
        self.module_parameters_dict = {}
        # 初始化分片信息字典
        self.sharding_dict = {}
        # 初始化名称为空字符串
        self.name = ""
    def _fw_pre_hook(self, mod, input):
        """
        This function is called before the forward pass of a module. It
        collects the parameters and sharding information of a module and
        stores it in a dictionary.
        """
        # 获取当前模块的名称
        self.name = super()._get_mod_name(mod)

        # 记录当前模块在模块树中的深度
        self.module_depth_dict[self.name] = len(self.parents)
        
        # 将当前子模块添加到模块追踪器的父类中
        super()._get_append_fn(self.name, False)()

        # 将输入展平成参数列表
        args, _ = tree_flatten(input)
        
        # 选择出是Tensor且需要梯度的参数
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        
        # 如果存在需要梯度的参数，注册多梯度钩子
        if tensors:
            register_multi_grad_hook(tensors, super()._get_pop_fn(self.name, True))

        # 遍历当前模块的参数
        for param_name, param in mod.named_parameters(recurse=False):
            # 如果当前模块不在模块参数字典中，创建它
            if self.name not in self.module_parameters_dict:
                self.module_parameters_dict[self.name] = {}

            # 存储参数数据到模块参数字典
            self.module_parameters_dict[self.name][param_name] = param.data

            # 如果参数数据是DTensor类型，存储分片信息到分片字典中
            if isinstance(param.data, DTensor):
                key_name = self.name + "." + param_name
                self.sharding_dict[key_name] = param.data.placements

    def __enter__(self):
        # 清空模块参数字典、分片字典和模块深度字典
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self.module_depth_dict.clear()
        # 将全局深度设置为0
        self.module_depth_dict["Global"] = 0
        # 注册模块前向预处理钩子和后向处理钩子
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(super()._fw_post_hook)

    def __exit__(self, *args):
        # 退出时调用父类的退出方法
        super().__exit__(*args)

    def print_paramater_info(self):
        # 打印模块参数字典
        print(self.module_parameters_dict)

    def print_sharding_info(self):
        # 打印分片字典的内容
        for key, value in self.sharding_dict.items():
            print(key + ": " + str(value))
class CommDebugMode(TorchDispatchMode):
    """
    ``CommDebugMode`` is a context manager that counts the number of
    functional collectives within its context. It does this using a
    ``TorchDispatchMode``.

    NOTE: this mode only works for functional collective atm and the
    distributed_c10d collectives are not supported yet.

    Example usage

    .. code-block:: python

        mod = ...
        comm_mode = CommDebugMode()
        with comm_mode:
            mod.sum().backward()

    """

    def __init__(self):
        # Initialize a dictionary to count communication occurrences
        self.comm_counts: Dict[Any, int] = defaultdict(int)
        # Initialize a dictionary to store module-specific communication counts
        self.comm_module_counts = {}
        # Initialize a set to register native and Python operation names
        for native_op, py_op in NATIVE_TO_PY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(py_op)

        # Add specific operation to the registry
        self.comm_registry.add(torch.ops._dtensor.shard_dim_alltoall)
        # Initialize an instance of CommModeModuleTracker
        self.advanced_module_tracker = CommModeModuleTracker()

    def generate_module_tracing_table(self):
        """
        Inspired by flop counter, generates a detailed table displaying collective tracing
        information on a module level
        """
        # Initialize an empty string to accumulate the tracing table
        table = ""
        # Iterate over module depth dictionary
        for fqn in self.advanced_module_tracker.module_depth_dict:
            # Calculate indentation based on module depth
            indent = "  " * (self.advanced_module_tracker.module_depth_dict[fqn])
            # Add fully qualified name to the table
            table += f"{indent}{fqn}\n"

            # Print all collectives in the submodule if available
            if fqn in self.comm_module_counts:
                for collective, count in self.comm_module_counts[fqn].items():
                    # Calculate indentation for collectives
                    collective_indent = "  " * (
                        (self.advanced_module_tracker.module_depth_dict[fqn]) + 1
                    )
                    # Highlight collective information in yellow
                    table += (
                        f"\033[1;33m{collective_indent}*{collective}: {count}\033[0m\n"
                    )

        return table

    def get_total_counts(self) -> int:
        # Return the sum of all communication counts
        return sum(self.comm_counts.values())

    def get_comm_counts(self) -> Dict[Any, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
        return self.comm_counts

    def get_comm_module_counts(self) -> Dict[str, Dict[Any, int]]:
        """
        Returns the communication counts at a module level as a dictionary.
        """
        return self.comm_module_counts

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        # Return module parameter information
        return self.advanced_module_tracker.module_parameters_dict

    def get_sharding_info(self) -> Dict[str, Dict[str, Any]]:
        # Return sharding information
        return self.advanced_module_tracker.sharding_dict

    def __enter__(self):
        # Clear communication counts and module counts
        self.comm_counts.clear()
        self.comm_module_counts.clear()
        # Call superclass __enter__ method
        super().__enter__()
        # Call __enter__ method of advanced_module_tracker
        self.advanced_module_tracker.__enter__()
        return self

    def __exit__(self, *args):
        # Call __exit__ method of advanced_module_tracker
        self.advanced_module_tracker.__exit__()
        # Call superclass __exit__ method
        super().__exit__(*args)
    # 将模块跟踪表格写入文件，用于日志记录
    def log_module_tracing_table_to_file(self):
        # ansi_escape 用于移除表格中的 ANSI 转义序列，以使终端输出更易读

        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        # 生成模块跟踪表格，并移除其中的 ANSI 转义序列
        table = ansi_escape.sub("", self.generate_module_tracing_table())

        # 打开文件 "output.txt"，以写模式写入处理后的表格内容
        with open("output.txt", "w") as log_file:
            log_file.write(table)

    # 打印参数信息，委托给 advanced_module_tracker 对象的 print_paramater_info 方法
    def print_paramater_info(self):
        self.advanced_module_tracker.print_paramater_info()

    # 打印分片信息，委托给 advanced_module_tracker 对象的 print_sharding_info 方法
    def print_sharding_info(self):
        self.advanced_module_tracker.print_sharding_info()

    # Torch 分发方法，用于处理函数的分发
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 当 types 中包含 DTensor 类型时，返回 NotImplemented，以便 DTensor 可以优先处理
        if any(t == DTensor for t in types):
            return NotImplemented
        
        # 如果 kwargs 为 None，则设置为空字典
        kwargs = kwargs if kwargs else {}
        
        # 执行函数并获取返回值
        out = func(*args, **kwargs)
        # 获取函数的重载包
        func_packet = func._overloadpacket

        # 如果 func_packet 在 comm_registry 或者 c10d_collective_ops 中
        if func_packet in self.comm_registry or func_packet in c10d_collective_ops:
            # 如果 func_packet 在 NATIVE_TO_PY_MAPPING 中，则映射到对应的 Python 原生函数包
            if func_packet in NATIVE_TO_PY_MAPPING:
                func_packet = NATIVE_TO_PY_MAPPING[func_packet]

            # 将 func_packet 的计数加一，用于统计通信操作的发生次数
            self.comm_counts[func_packet] += 1

            # 如果当前模块的名称不在 comm_module_counts 中，则添加，并初始化为 defaultdict(int)
            if self.advanced_module_tracker.name not in self.comm_module_counts:
                self.comm_module_counts[self.advanced_module_tracker.name] = defaultdict(int)
            
            # 将 func_packet 的计数加一，记录在当前模块中发生的通信操作次数
            self.comm_module_counts[self.advanced_module_tracker.name][func_packet] += 1

            # 遍历当前模块的所有父模块
            for par in self.advanced_module_tracker.parents:
                # 确保不重复计数当前子模块的通信操作
                if par != self.advanced_module_tracker.name:
                    # 如果父模块不在 comm_module_counts 中，则添加，并初始化为 defaultdict(int)
                    if par not in self.comm_module_counts:
                        self.comm_module_counts[par] = defaultdict(int)
                    # 将 func_packet 的计数加一，记录在父模块中发生的通信操作次数
                    self.comm_module_counts[par][func_packet] += 1

        # 返回函数执行的结果 out
        return out
```