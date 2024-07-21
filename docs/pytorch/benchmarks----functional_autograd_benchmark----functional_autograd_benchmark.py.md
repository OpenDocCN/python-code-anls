# `.\pytorch\benchmarks\functional_autograd_benchmark\functional_autograd_benchmark.py`

```py
import time  # 导入time模块，用于时间相关操作
from argparse import ArgumentParser  # 导入ArgumentParser类，用于解析命令行参数
from collections import defaultdict  # 导入defaultdict类，创建默认值字典
from typing import Any, Callable, List, NamedTuple  # 导入类型提示相关类和函数

import torch  # 导入PyTorch深度学习库
from torch.autograd import functional  # 导入functional模块，用于自动求导函数

try:
    import functorch as ft  # 尝试导入functorch库，用于自动微分增强
    has_functorch = True  # 设置标志，表示functorch已导入
    print(f"Found functorch: {ft.__version__}")  # 打印functorch版本信息
except ImportError:
    has_functorch = False  # 设置标志，表示functorch导入失败

import audio_text_models  # 导入音频文本模型
import ppl_models  # 导入概率编程模型
import vision_models  # 导入视觉模型

from utils import (  # 从utils模块导入多个函数和类型
    GetterType, InputsType, TimingResultType, to_markdown_table, VType
)


def get_task_func(task: str) -> Callable:
    def hessian_fwdrev(model, inp, strict=None):
        return functional.hessian(  # 计算模型的Hessian矩阵，使用forward-mode
            model,
            inp,
            strict=False,
            vectorize=True,
            outer_jacobian_strategy="forward-mode",
        )

    def hessian_revrev(model, inp, strict=None):
        return functional.hessian(  # 计算模型的Hessian矩阵，使用reverse-mode
            model, inp, strict=False, vectorize=True
        )

    def jacfwd(model, inp, strict=None):
        return functional.jacobian(  # 计算模型的Jacobian矩阵，使用forward-mode
            model, inp, strict=False, vectorize=True, strategy="forward-mode"
        )

    def jacrev(model, inp, strict=None):
        return functional.jacobian(  # 计算模型的Jacobian矩阵，使用reverse-mode
            model, inp, strict=False, vectorize=True
        )

    if task == "hessian_fwdrev":  # 根据任务名返回对应的函数
        return hessian_fwdrev
    elif task == "hessian_revrev":
        return hessian_revrev
    elif task == "jacfwd":
        return jacfwd
    elif task == "jacrev":
        return jacrev
    else:
        return getattr(functional, task)  # 返回functional模块中对应的函数


def get_task_functorch(task: str) -> Callable:
    @torch.no_grad()
    def vjp(model, inp, v=None, strict=None):
        assert v is not None  # 确保传入的向量v不为None
        out, vjpfunc = ft.vjp(model, *inp)  # 计算模型在输入inp处的向量-雅可比积，使用functorch
        return out, vjpfunc(v)  # 返回计算结果

    @torch.no_grad()
    def jvp(model, inp, v=None, strict=None):
        assert v is not None  # 确保传入的向量v不为None
        return ft.jvp(model, inp, v)  # 计算模型在输入inp处的雅可比向量积，使用functorch

    @torch.no_grad()
    def vhp(model, inp, v=None, strict=None):
        assert v is not None  # 确保传入的向量v不为None
        argnums = tuple(range(len(inp)))
        _, vjpfunc, aux = ft.vjp(  # 计算模型在输入inp处的向量-哈森积，使用functorch
            ft.grad_and_value(model, argnums), *inp, has_aux=True
        )
        return aux, vjpfunc(v)  # 返回计算结果

    @torch.no_grad()
    def hvp(model, inp, v=None, strict=None):
        assert v is not None  # 确保传入的向量v不为None
        argnums = tuple(range(len(inp)))
        _, hvp_out, aux = ft.jvp(  # 计算模型在输入inp处的哈森向量积，使用functorch
            ft.grad_and_value(model, argnums), inp, v, has_aux=True
        )
        return aux, hvp_out  # 返回计算结果

    @torch.no_grad()
    def jacfwd(model, inp, v=None, strict=None):
        argnums = tuple(range(len(inp)))
        return ft.jacfwd(model, argnums)(*inp)  # 计算模型的Jacobian矩阵，使用forward-mode，使用functorch

    @torch.no_grad()
    def jacrev(model, inp, v=None, strict=None):
        argnums = tuple(range(len(inp)))
        return ft.jacrev(model, argnums)(*inp)  # 计算模型的Jacobian矩阵，使用reverse-mode，使用functorch

    @torch.no_grad()
    def hessian(model, inp, v=None, strict=None):
        argnums = tuple(range(len(inp)))
        return ft.hessian(model, argnums=argnums)(*inp)  # 计算模型的Hessian矩阵，使用functorch

    @torch.no_grad()
    # 计算函数的前向和反向混合 Hessian 矩阵
    def hessian_fwdrev(model, inp, v=None, strict=None):
        # 生成参数索引元组
        argnums = tuple(range(len(inp)))
        # 使用前向自动求导计算模型的反向自动求导的 Jacobian 矩阵，并对其再次进行前向自动求导
        return ft.jacfwd(ft.jacrev(model, argnums=argnums), argnums=argnums)(*inp)

    # 使用 PyTorch 的无梯度上下文装饰器
    @torch.no_grad()
    # 计算函数的反向自动求导的 Hessian 矩阵
    def hessian_revrev(model, inp, v=None, strict=None):
        # 生成参数索引元组
        argnums = tuple(range(len(inp)))
        # 对模型的反向自动求导的 Jacobian 矩阵再次进行反向自动求导
        return ft.jacrev(ft.jacrev(model, argnums=argnums), argnums=argnums)(*inp)

    # 如果任务函数在局部变量中存在，则返回该函数
    if task in locals():
        return locals()[task]
    # 如果任务是计算 Jacobian 矩阵但没有 vectorize=False 的等价方法，则引发运行时错误
    elif task == "jacobian":
        raise RuntimeError(
            "functorch has no equivalent of autograd.functional.jacobian with vectorize=False yet"
        )
    # 如果任务是未支持的任务，则引发运行时错误并显示任务名称
    else:
        raise RuntimeError(f"Unsupported task: {task}")
# 定义不需要双向传播的快速任务列表
FAST_TASKS_NO_DOUBLE_BACK = [
    "vjp",
]

# 定义包含双向传播任务的快速任务列表
FAST_TASKS = FAST_TASKS_NO_DOUBLE_BACK + [
    "vhp",
    "jvp",
]

# 包含所有非向量化任务的列表，包括快速任务和"hvp", "jacobian", "hessian"
ALL_TASKS_NON_VECTORIZED = FAST_TASKS + ["hvp", "jacobian", "hessian"]

# 双向传播任务的列表
DOUBLE_BACKWARD_TASKS = ["jvp", "hvp", "vhp", "hessian"]

# 向量化任务的列表
VECTORIZED_TASKS = ["hessian_fwdrev", "hessian_revrev", "jacfwd", "jacrev"]

# 所有任务的列表，包括所有非向量化任务和向量化任务
ALL_TASKS = ALL_TASKS_NON_VECTORIZED + VECTORIZED_TASKS


# 模型定义，包括：
# - name: 模型名称字符串
# - getter: 获取模型的函数，输入为模型运行的设备，返回前向函数和作为输入的参数（张量）
# - tasks: 建议使用该模型运行的任务列表
# - unsupported: 该模型无法运行的任务列表
class ModelDef(NamedTuple):
    name: str
    getter: GetterType
    tasks: List[str]
    unsupported: List[str]


# 模型列表，每个元素是一个 ModelDef 实例，包含不同的模型定义
MODELS = [
    ModelDef("resnet18", vision_models.get_resnet18, FAST_TASKS, []),
    ModelDef("fcn_resnet", vision_models.get_fcn_resnet, FAST_TASKS, []),
    ModelDef("detr", vision_models.get_detr, FAST_TASKS, []),
    ModelDef("ppl_simple_reg", ppl_models.get_simple_regression, ALL_TASKS, []),
    ModelDef("ppl_robust_reg", ppl_models.get_robust_regression, ALL_TASKS, []),
    ModelDef("wav2letter", audio_text_models.get_wav2letter, FAST_TASKS, []),
    ModelDef(
        "deepspeech",
        audio_text_models.get_deepspeech,
        FAST_TASKS_NO_DOUBLE_BACK,
        DOUBLE_BACKWARD_TASKS,
    ),
    ModelDef("transformer", audio_text_models.get_transformer, FAST_TASKS, []),
    ModelDef("multiheadattn", audio_text_models.get_multiheadattn, FAST_TASKS, []),
]


# 获取模型的 V 值的函数定义，输入为模型、输入和任务类型，输出为 V 类型的结果
def get_v_for(model: Callable, inp: InputsType, task: str) -> VType:
    v: VType

    # 根据任务类型选择不同的处理逻辑生成 V 值
    if task in ["vjp"]:
        out = model(*inp)
        v = torch.rand_like(out)
    elif task in ["jvp", "hvp", "vhp"]:
        if isinstance(inp, tuple):
            v = tuple(torch.rand_like(i) for i in inp)
        else:
            v = torch.rand_like(inp)
    else:
        v = None

    return v


# 运行一次任务的函数定义，输入为模型、输入、任务类型、V 值和其他关键字参数，无返回值
def run_once(model: Callable, inp: InputsType, task: str, v: VType, **kwargs) -> None:
    func = get_task_func(task)

    # 如果有 V 值，则使用严格模式调用任务函数
    if v is not None:
        res = func(model, inp, v=v, strict=True)
    else:
        res = func(model, inp, strict=True)


# 使用 FunctoRch 运行一次任务的函数定义，输入为模型、输入、任务类型、V 值以及可选的一致性检查参数，无返回值
def run_once_functorch(
    model: Callable, inp: InputsType, task: str, v: VType, maybe_check_consistency=False
) -> None:
    func = get_task_functorch(task)

    # 如果有 V 值，则使用严格模式调用任务函数
    if v is not None:
        res = func(model, inp, v=v, strict=True)
    else:
        res = func(model, inp, strict=True)
    # 如果需要检查一致性
    if maybe_check_consistency:
        # 获取任务函数
        af_func = get_task_func(task)
        
        # 如果输入参数 v 不为 None
        if v is not None:
            # 使用任务函数计算期望结果，设置 strict=True 来进行严格比较
            expected = af_func(model, inp, v=v, strict=True)
        else:
            # 使用任务函数计算期望结果，设置 strict=True 来进行严格比较
            expected = af_func(model, inp, strict=True)
        
        # 根据任务类型设置绝对误差容限
        atol = 1e-2 if task == "vhp" else 5e-3
        
        # 使用 PyTorch 的测试工具 assert_close 进行结果比较
        torch.testing.assert_close(
            res,                           # 实际结果
            expected,                      # 期望结果
            rtol=1e-5,                     # 相对误差容限
            atol=atol,                     # 绝对误差容限
            msg=f"Consistency fail for task '{task}'",  # 失败时的错误消息
        )
# 定义一个函数 run_model，用于运行模型并返回每次运行的时间列表
def run_model(
    model_getter: GetterType, args: Any, task: str, run_once_fn: Callable = run_once
) -> List[float]:
    # 根据参数中的 gpu 值选择设备，-1 为 CPU
    if args.gpu == -1:
        device = torch.device("cpu")

        # 定义一个空操作函数 noop
        def noop():
            pass

        # 当 GPU 为 -1 时，同步操作设为 noop 函数
        do_sync = noop
    else:
        # 根据参数中的 gpu 值选择对应的 cuda 设备
        device = torch.device(f"cuda:{args.gpu}")
        # 获取 cuda 同步函数
        do_sync = torch.cuda.synchronize

    # 获取模型和输入数据
    model, inp = model_getter(device)

    # 获取用于当前任务的 v 值
    v = get_v_for(model, inp, task)

    # 预热阶段
    # maybe_check_consistency=True 用于检查 functorch 和 autograd.functional 之间的一致性，
    # 仅在 run_once_functorch 函数中进行检查
    run_once_fn(model, inp, task, v, maybe_check_consistency=True)

    # 记录每次运行的时间
    elapsed = []
    for it in range(args.num_iters):
        # 执行同步操作
        do_sync()
        # 记录开始时间
        start = time.time()
        # 运行一次模型
        run_once_fn(model, inp, task, v)
        # 再次执行同步操作
        do_sync()
        # 计算运行时间并添加到列表中
        elapsed.append(time.time() - start)

    # 返回时间列表
    return elapsed


# 主函数入口
def main():
    # 创建参数解析器
    parser = ArgumentParser("Main script to benchmark functional API of the autograd.")
    # 添加命令行参数
    parser.add_argument(
        "--output", type=str, default="", help="Text file where to write the output"
    )
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument(
        "--gpu",
        type=int,
        default=-2,
        help="GPU to use, -1 for CPU and -2 for auto-detect",
    )
    parser.add_argument(
        "--run-slow-tasks", action="store_true", help="Run even the slow tasks"
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default="",
        help="Only run the models in this filter",
    )
    parser.add_argument(
        "--task-filter", type=str, default="", help="Only run the tasks in this filter"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=10,
        help="Number of concurrent threads to use when running on cpu",
    )
    parser.add_argument("--seed", type=int, default=0, help="The random seed to use.")
    # 解析命令行参数
    args = parser.parse_args()

    # 初始化结果字典，用于存储运行时间
    results: TimingResultType = defaultdict(defaultdict)
    # 设置 PyTorch 的线程数
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    # 设置随机种子，如果可用的话自动设置 cuda 的随机种子
    torch.manual_seed(args.seed)

    # 如果 gpu 参数为 -2，则自动检测是否有可用的 cuda 设备
    if args.gpu == -2:
        args.gpu = 0 if torch.cuda.is_available() else -1
    # 遍历模型列表，每个元素包含模型名、模型获取函数、推荐任务列表和不支持任务列表
    for name, model_getter, recommended_tasks, unsupported_tasks in MODELS:
        # 如果命令行参数中指定了模型过滤器，并且当前模型名不在过滤器列表中，则跳过当前模型
        if args.model_filter and name not in args.model_filter:
            continue
        # 根据是否运行慢速任务来决定使用推荐任务列表或者全部任务列表
        tasks = ALL_TASKS if args.run_slow_tasks else recommended_tasks
        # 遍历当前模型的任务列表
        for task in tasks:
            # 如果当前任务在不支持任务列表中，则跳过当前任务
            if task in unsupported_tasks:
                continue
            # 如果命令行参数中指定了任务过滤器，并且当前任务不在过滤器列表中，则跳过当前任务
            if args.task_filter and task not in args.task_filter:
                continue
            # 运行模型，并记录运行时间
            runtimes = run_model(model_getter, args, task)

            # 将运行时间转换为 PyTorch 的张量
            runtimes = torch.tensor(runtimes)
            # 计算平均值和方差
            mean, var = runtimes.mean(), runtimes.var()
            # 将结果存入结果字典
            results[name][task] = (mean.item(), var.item())
            # 打印模型在任务上的结果，包括平均时间和方差
            print(f"Results for model {name} on task {task}: {mean}s (var: {var})")

            # 如果存在 Functorch 模块
            if has_functorch:
                try:
                    # 使用 Functorch 运行模型，并记录运行时间
                    runtimes = run_model(
                        model_getter, args, task, run_once_fn=run_once_functorch
                    )
                except RuntimeError as e:
                    # 捕获异常情况，打印出错信息
                    print(
                        f"Failed model using Functorch: {name}, task: {task}, Error message: \n\t",
                        e,
                    )
                    continue

                # 将运行时间转换为 PyTorch 的张量
                runtimes = torch.tensor(runtimes)
                # 计算平均值和方差
                mean, var = runtimes.mean(), runtimes.var()
                # 将结果存入结果字典，包含 Functorch 标记
                results[name][f"functorch {task}"] = (mean.item(), var.item())
                # 打印使用 Functorch 的模型在任务上的结果，包括平均时间和方差
                print(
                    f"Results for model {name} on task {task} using Functorch: {mean}s (var: {var})"
                )

    # 如果命令行参数指定了输出文件
    if args.output:
        # 将结果字典转换为 Markdown 格式的表格，并写入指定的输出文件
        with open(args.output, "w") as f:
            f.write(to_markdown_table(results))
# 如果这个脚本被直接执行（而不是被导入为模块），那么执行下面的代码
if __name__ == "__main__":
    # 调用名为 main 的函数，作为程序的入口点
    main()
```