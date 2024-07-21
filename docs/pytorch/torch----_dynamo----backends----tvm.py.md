# `.\pytorch\torch\_dynamo\backends\tvm.py`

```py
# 忽略类型检查中的错误，这可能是因为某些类型在类型检查时无法识别
import functools  # 导入 functools 模块，用于高阶函数和操作函数对象的工具
import importlib  # 导入 importlib 模块，用于动态加载模块
import logging  # 导入 logging 模块，用于记录日志消息
import os  # 导入 os 模块，提供与操作系统交互的功能
import sys  # 导入 sys 模块，提供与 Python 解释器交互的功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
from types import MappingProxyType  # 从 types 模块导入 MappingProxyType 类型，提供不可变映射类型的支持
from typing import Optional  # 从 typing 模块导入 Optional 类型，表示可选类型的支持

import torch  # 导入 PyTorch 模块
from .common import device_from_inputs, fake_tensor_unsupported  # 从当前包导入指定函数

from .registry import register_backend  # 从当前包导入 register_backend 函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@register_backend  # 使用注册装饰器标记下面的函数作为后端注册
@fake_tensor_unsupported  # 使用装饰器标记函数不支持伪张量
def tvm(
    gm,  # 函数 gm 的输入参数
    example_inputs,  # 示例输入的参数
    *,
    options: Optional[MappingProxyType] = MappingProxyType(
        {"scheduler": None, "trials": 20000, "opt_level": 3}
    ),  # options 参数，默认为不可变映射类型，包含调度器、试验次数和优化级别的配置
):
    import tvm  # 导入 tvm 模块，用于 TVM 框架的功能
    from tvm import relay  # 从 tvm 模块导入 relay 子模块，用于中间表示的定义
    from tvm.contrib import graph_executor  # 从 tvm.contrib 模块导入 graph_executor，用于图执行器的功能

    jit_mod = torch.jit.trace(gm, example_inputs)  # 使用 PyTorch 的 jit 跟踪创建模块
    device = device_from_inputs(example_inputs)  # 根据示例输入获取设备信息
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]  # 创建输入形状的列表
    example_outputs = gm(*example_inputs)  # 执行 gm 函数得到示例输出

    if len(example_outputs) == 0:  # 如果示例输出为空
        log.warning("Explicitly fall back to eager due to zero output")  # 记录警告消息
        return gm.forward  # 返回 gm 函数的前向方法

    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)  # 使用 PyTorch 转换为 TVM 的中间表示和参数

    if device.type == "cuda":  # 如果设备类型是 CUDA
        dev = tvm.cuda(device.index)  # 创建 CUDA 设备
        target = tvm.target.cuda()  # 设置目标为 CUDA
    else:  # 否则
        dev = tvm.cpu(0)  # 创建 CPU 设备
        target = tvm.target.Target(llvm_target())  # 设置目标为 LLVM 目标

    scheduler = options.get("scheduler", None)  # 获取调度器选项
    if scheduler is None:  # 如果调度器未指定
        scheduler = os.environ.get("TVM_SCHEDULER", None)  # 从环境变量获取调度器

    trials = options.get("trials", 20000)  # 获取试验次数选项，默认为 20000
    opt_level = options.get("opt_level", 3)  # 获取优化级别选项，默认为 3

    if scheduler == "auto_scheduler":  # 如果调度器是自动调度器
        from tvm import auto_scheduler  # 导入 TVM 自动调度器模块

        log_file = tempfile.NamedTemporaryFile()  # 创建一个命名的临时文件

        if not os.path.exists(log_file):  # 如果日志文件不存在
            tasks, task_weights = auto_scheduler.extract_tasks(  # 提取自动调度器任务和权重
                mod["main"], params, target
            )
            for task in tasks:  # 遍历任务
                print(task.compute_dag)  # 打印任务的计算 DAG
            else:
                print("No tasks")  # 打印无任务

            if len(tasks) != 0:  # 如果有任务
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)  # 创建任务调度器
                if not os.path.exists(log_file):  # 如果日志文件不存在
                    assert trials > 0  # 断言试验次数大于 0
                    tune_option = auto_scheduler.TuningOptions(  # 创建调整选项
                        num_measure_trials=trials,  # 设置测量试验次数
                        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],  # 设置测量回调
                        early_stopping=2000,  # 设置早期停止次数
                    )
                    try:
                        tuner.tune(tune_option)  # 执行调整过程
                    except Exception:  # 捕获异常
                        if os.path.exists(log_file):  # 如果日志文件存在
                            os.unlink(log_file)  # 删除日志文件
                        raise  # 抛出异常

        with auto_scheduler.ApplyHistoryBest(log_file):  # 应用历史最佳记录
            with tvm.transform.PassContext(  # 使用转换上下文
                opt_level=opt_level,  # 设置优化级别
                config={"relay.backend.use_auto_scheduler": True},  # 配置使用自动调度器
            ):
                lib = relay.build(mod, target=target, params=params)  # 使用中间表示构建库
    elif scheduler == "meta_schedule":
        # 导入 meta_schedule 模块
        from tvm import meta_schedule as ms

        # 使用临时目录作为工作目录
        with tempfile.TemporaryDirectory() as work_dir:
            if device.type != "cuda":
                # meta_schedule 需要指定 num-cores
                # 这里使用最大核心数
                target = tvm.target.Target(
                    f"{llvm_target()} --num-cores {ms.utils.cpu_count(logical=False)}"
                )
            # TODO(shingjan): 这可以通过 tvm.contrib.torch.optimize_torch 替换
            # 一旦 USE_PT_TVMDSOOP 在 TVM 中更新并默认打开。
            assert trials > 0
            # 使用 meta_schedule 进行自动调优
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=trials,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
                opt_level=opt_level,
            )
            # 编译 Relay 模块
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
                opt_level=opt_level,
            )
    elif scheduler == "default" or not scheduler:
        # 如果调度器是 "default" 或者未指定，不进行自动调优
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)
    else:
        # 抛出未实现错误，说明该调度选项在 torchdynamo 的 TVM 后端中无效或未实现
        raise NotImplementedError(
            "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
            "There are three available options: default, auto_scheduler and meta_schedule."
        )

    # 根据编译好的库文件创建 GraphModule 实例
    m = graph_executor.GraphModule(lib["default"](dev))

    def to_torch_tensor(nd_tensor):
        """将 NDArray 转换为 torch.tensor 的辅助函数."""
        if nd_tensor.dtype == "bool":
            # DLPack 不支持布尔类型，无法通过 torch.utils.dlpack.from_pack 处理。
            # 通过 numpy 进行转换，尽管会引入额外的数据复制开销。
            return torch.from_numpy(nd_tensor.numpy())
        return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

    def to_tvm_tensor(torch_tensor):
        """将 torch.tensor 转换为 NDArray 的辅助函数."""
        if torch_tensor.dtype == torch.bool:
            # 同样的原因，回退到 numpy 转换，可能会引入数据复制开销。
            return tvm.nd.array(torch_tensor.cpu().numpy())
        return tvm.nd.from_dlpack(torch_tensor)
    # 定义一个函数 exec_tvm，用于执行 TVM 模型推理
    def exec_tvm(*i_args):
        # 将所有输入参数转换为连续的张量
        args = [a.contiguous() for a in i_args]
        # 获取模型输入的形状信息和输入名称
        shape_info, _ = m.get_input_info()
        # 创建一个集合，包含模型中所有活跃的输入名称
        active_inputs = {name for name, _ in shape_info.items()}
        # 遍历所有输入参数
        for idx, arg in enumerate(args, 0):
            # 检查参数是否为非零维度
            if arg.dim() != 0:
                # 如果参数需要梯度，将其分离（detach）
                if arg.requires_grad:
                    arg = arg.detach()
                # 根据参数索引创建输入名称
                inp_name = f"inp_{idx}"
                # 如果输入名称不在活跃输入集合中，记录警告信息并跳过
                if inp_name not in active_inputs:
                    log.warning(
                        "input %s skipped as not found in tvm's runtime library",
                        inp_name,
                    )
                    continue
                # 将参数转换为 TVM 张量，并设置为模型的输入
                m.set_input(
                    inp_name,
                    to_tvm_tensor(arg),
                )
        # 运行 TVM 模型推理
        m.run()
        # 返回所有输出的 Torch 张量列表
        return [to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())]

    # 返回定义好的 exec_tvm 函数
    return exec_tvm
# 使用 functools.partial 函数创建一个 tvm_meta_schedule 函数，将 tvm 函数的 scheduler 参数设为 "meta_schedule"
tvm_meta_schedule = functools.partial(tvm, scheduler="meta_schedule")

# 使用 functools.partial 函数创建一个 tvm_auto_scheduler 函数，将 tvm 函数的 scheduler 参数设为 "auto_scheduler"
tvm_auto_scheduler = functools.partial(tvm, scheduler="auto_scheduler")

# 检查是否存在 tvm 模块
def has_tvm():
    try:
        # 尝试导入 tvm 模块
        importlib.import_module("tvm")
        return True
    except ImportError:
        # 如果导入失败，则返回 False
        return False

# 使用 functools.lru_cache(None) 装饰器定义 llvm_target 函数，根据系统平台返回相应的 LLVM 目标字符串
@functools.lru_cache(None)
def llvm_target():
    if sys.platform == "linux":
        # 读取 /proc/cpuinfo 文件内容
        cpuinfo = open("/proc/cpuinfo").read()
        if "avx512" in cpuinfo:
            # 如果 cpuinfo 中包含 avx512，则返回适用于 skylake-avx512 架构的 LLVM 目标字符串
            return "llvm -mcpu=skylake-avx512"
        elif "avx2" in cpuinfo:
            # 如果 cpuinfo 中包含 avx2，则返回适用于 core-avx2 架构的 LLVM 目标字符串
            return "llvm -mcpu=core-avx2"
    # 默认情况下返回通用的 LLVM 目标字符串
    return "llvm"
```