# `.\pytorch\benchmarks\dynamo\microbenchmarks\operatorbench.py`

```py
# 指定脚本的解释器为 Python 3
#!/usr/bin/env python3

# 导入需要使用的库和模块
import click  # 用于创建命令行界面
import numpy as np  # 用于数值计算
from operator_inp_utils import OperatorInputsLoader  # 导入自定义模块 OperatorInputsLoader

import torch  # 导入 PyTorch 库

# 导入 PyTorch 内部模块和函数
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._dynamo.testing import same
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import lowerings
from torch._inductor.utils import gen_gm_and_inputs
from torch.utils._pytree import tree_map_only

# 将 torch.ops.aten 赋值给变量 aten，用于简化调用
aten = torch.ops.aten


# 定义函数 compute_speedups，用于计算多个模型在给定输入下的加速比
def compute_speedups(
    operator, models, example_inputs, repeats, accuracy_checking=False, device="cuda"
):
    # 计算期望输出，以第一个模型的输出为标准
    expected = models[0](*example_inputs)
    
    # 如果开启了精度检查
    if accuracy_checking:
        # 对每个模型进行精度检查
        for model in models[1:]:
            actual = model(*example_inputs)
            # 检查实际输出是否与期望输出相同，使用余弦相似度和 NaN 相等性检查
            try:
                same(actual, expected, cos_similarity=True, equal_nan=True)
            except AssertionError as e:
                print(e)
                print(f"Accuracy check failed: {operator}")
                # 打印精度检查失败的信息及相关计算结果
                print((expected[0] - actual[0]).abs().max())

    # 初始化计时器数组，用于记录多次运行的时间
    timings = np.zeros((repeats, len(models)), np.float64)
    # 多次重复运行
    for rep in range(repeats):
        # 交替运行不同模型，以处理频率缩放和负载变化
        for m, model in enumerate(models):
            if device == "cuda":
                # 如果设备为 CUDA，导入 triton 库并执行模型
                import triton

                model(*example_inputs)

                # 使用 triton.testing.do_bench() 清除 L2 缓存以隐藏 CPU 启动时间的延迟
                # 同时进行 CUDA 同步
                timings[rep, m] = triton.testing.do_bench(
                    lambda: model(*example_inputs)
                )
            else:
                # 如果设备不是 CUDA，导入 torch._inductor.utils 中的 timed 函数进行计时
                from torch._inductor.utils import timed

                timings[rep, m] = timed(model, example_inputs)
    # 返回每个模型运行时间的中位数
    return np.median(timings, axis=0)


# 定义函数 strip_overloads，用于修改图节点的目标，以去除重载
def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        # 如果节点的目标是 OpOverload 类型的对象，则将目标改为其 overloadpacket 属性
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    # 重新编译修改后的 gm
    gm.recompile()


# 定义函数 convert_to_jit，将图模块转换为 JIT 模块（即 TorchScript）
def convert_to_jit(gm, gm_args):
    # 去除重载
    strip_overloads(gm)
    try:
        # 尝试将 gm 转换为 TorchScript 的脚本模块
        return torch.jit.script(gm)
    except Exception:
        pass
    # 如果转换失败，则回退到使用 TorchScript 的追踪功能
    return torch.jit.trace(gm, gm_args)


# 定义函数 to_channels_last，将张量转换为 channels_last 内存格式
def to_channels_last(ten):
    return ten if ten.ndim != 4 else ten.to(memory_format=torch.channels_last)


# 定义函数 microbenchmark，用于微基准测试
def microbenchmark(
    operator, args, kwargs, dtype, accuracy_checking, repeats, measure_nvfuser, device
):
    # 生成 gm（图模块）及其输入
    gm, gm_args = gen_gm_and_inputs(operator, args, kwargs)
    
    # 注册内置的 torch.ops.aten.convolution_backward.default 函数为 "aten::convolution_backward"
    torch.jit._builtins._register_builtin(
        torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
    )
    # 如果设备是 "cuda"，执行以下操作
    if device == "cuda":
        # 使用 cudagraphs_inner 函数对计算图 gm 和参数 gm_args 进行即时图计算，
        # 不复制输出和输入
        cudagraphs_eager = cudagraphs_inner(
            gm, gm_args, copy_outputs=False, copy_inputs=False
        )
        # 编译计算图 gm，并生成编译后的函数对象 compiled_fn
        compiled_fn = compile_fx(gm, gm_args)
        # 使用 cudagraphs_inner 函数对编译后的函数对象 compiled_fn 和参数 gm_args 进行即时图计算，
        # 不复制输出和输入
        cudagraphs_compiled = cudagraphs_inner(
            compiled_fn, gm_args, copy_outputs=False, copy_inputs=False
        )
        # 将即时计算图和编译后的即时计算图存入列表 compiled
        compiled = [cudagraphs_eager, cudagraphs_compiled]
    else:
        # 如果设备不是 "cuda"，仅编译计算图 gm，生成编译后的函数对象 compiled_fn
        compiled_fn = compile_fx(gm, gm_args)
        # 将原始计算图 gm 和编译后的函数对象存入列表 compiled
        compiled = [gm, compiled_fn]

    # 如果需要测量 nvfuser 性能
    if measure_nvfuser:
        # 将计算图 gm 转换为 JIT 模块 g
        g = convert_to_jit(gm, gm_args)
        # 使用 cudagraphs_inner 函数对 JIT 模块 g 和参数 gm_args 进行即时图计算，
        # 不复制输出和输入
        cudagraphs_jit = cudagraphs_inner(
            g, gm_args, copy_outputs=False, copy_inputs=False
        )
        # 将 JIT 模块的即时计算图加入列表 compiled
        compiled += [cudagraphs_jit]

    # 如果需要进行准确性检查
    if accuracy_checking:
        # 设置重复运行次数为 1
        repeats = 1

    # 计算不同计算图下的运行速度中位数，并返回结果
    medians = compute_speedups(
        operator, compiled, gm_args, repeats, accuracy_checking, device
    )
    return medians
# 定义一个函数用于跳过不需要进行性能基准测试的运算符
def skip_operator(operator):
    # 列出不需要进行测试的运算符字符串
    nyi_strings = (
        "aten.gather.default",
        "nll_loss",
        "aten.index",
        "aten.scatter_",
        "masked_fill_.Scalar",
    )

    # 如果运算符包含在不需要进行测试的字符串中，则跳过该运算符
    if any(nyi_string in str(operator) for nyi_string in nyi_strings):
        # 输出跳过的信息及原因
        print(f"Skipping {operator}, input generator nyi")
        return True

    # 如果是非计算类的运算符，则跳过
    if operator == torch.ops.aten._unsafe_view.default:
        # 输出跳过的信息及原因
        print(f"Skipping {operator}, non compute operator")
        return True

    # 对于注册到 OpOverload 或 OpOverloadPacket 的运算符，都需要进行性能基准测试
    op_impls = [operator]
    if isinstance(operator, torch._ops.OpOverload):
        op_impls.append(operator.overloadpacket)

    # TODO - 跳过需要使用回退方案进行基准测试的运算符，有些运算符同时存在多种降低策略
    # 因此从运算符本身无法明确它将使用哪种降低策略。
    
    # 如果所有实现都不在 decompositions 和 lowerings 中，则跳过该运算符
    if all(op not in decompositions and op not in lowerings for op in op_impls):
        # 输出跳过的信息及原因
        print(f"Skipping {operator}, no inductor impl")
        return True

    # 如果运算符名称中包含"convolution"，则跳过该运算符
    if "convolution" in str(operator):
        return True

    # 其他情况不跳过该运算符
    return False


# 命令行入口函数，用于设置基准测试的各项参数
@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
@click.option("--max-samples", help="max samples per op", default=15)
@click.option("--accuracy-checking", help="check accuracy", default=False)
@click.option(
    "--repeats", help="how many times to repeat for perf measurement", default=3
)
@click.option(
    "--measure-nvfuser", help="default we only measure inductor", default=False
)
@click.option("--device", help="cpu or cuda", default="cuda")
@click.option("--inp-file", help="use custom input file instead of suite", default=None)
@click.option("--start-idx", help="specify start index of samples", default=0)
@click.option(
    "--channels-last", help="force inputs to channels last", is_flag=True, default=False
)
def benchmark(
    suite,
    op,
    dtype,
    max_samples,
    accuracy_checking,
    repeats,
    measure_nvfuser,
    device,
    inp_file,
    start_idx,
    channels_last,
):
    # 如果指定了自定义的输入文件，则使用 OperatorInputsLoader 加载器加载该文件
    if inp_file is not None:
        loader = OperatorInputsLoader(inp_file)
    else:
        # 否则根据 suite 的不同选择相应的默认加载器
        assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
        if suite == "timm":
            loader = OperatorInputsLoader.get_timm_loader()
        elif suite == "huggingface":
            loader = OperatorInputsLoader.get_huggingface_loader()
        else:
            loader = OperatorInputsLoader.get_torchbench_loader()

    # 断言 dtype 必须是 "float16" 或 "float32"
    assert dtype in ("float16", "float32"), f"got {dtype}"
    # 如果操作为 "all"，则生成文件名格式为 timmings_{suite}_{op.replace('.', '_')}{dtype}.txt
    if op == "all":
        filename = f"timings_{suite}_{op.replace('.', '_')}{dtype}.txt"
        # 打开文件以追加模式写入
        f = open(filename, "a")

    # 根据 dtype 的取值，确定 torch 的数据类型为 float16 或 float32
    dtype = torch.float16 if dtype == "float16" else torch.float32

    # 如果操作为 "all"，获取所有操作符；否则，将字符串 op 转换为操作符列表
    if op == "all":
        ops = loader.get_all_ops()
    else:
        ops = [eval(op)]

    # 将 max_samples 加上 start_idx 以获取最大样本数
    max_samples = max_samples + start_idx

    # 遍历每个操作符
    for operator in ops:
        # 如果应跳过该操作符，则继续下一个操作符
        if skip_operator(operator):
            continue

        # 打印当前正在运行的操作符名称
        print(f"Running {operator}")

        # 获取用于当前操作符的输入生成器 inp_gen
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype, device=device)

        # 初始化用于存储执行时间的列表
        timings = []

        # 循环执行最多 max_samples 次（但不超过 1000000 次）
        for i in range(min(max_samples, 1000000)):
            try:
                # 从输入生成器中获取下一个输入对 inps
                inps = next(inp_gen)
                if inps is None:
                    break
                # 如果当前循环次数 i 小于 start_idx，则跳过本次循环
                if i < start_idx:
                    continue
                # 打印当前迭代的次数 i
                print(f"Iter {i}")
                # 解包输入对 inps 到 args 和 kwargs
                args, kwargs = inps
                # 如果 channels_last 为 True，则将 args 和 kwargs 转换为通道位置优先格式
                if channels_last:
                    args, kwargs = tree_map_only(
                        torch.Tensor, to_channels_last, (args, kwargs)
                    )

            except StopIteration:
                break

            try:
                # 运行微基准测试，并将执行时间添加到 timings 列表中
                timings.append(
                    microbenchmark(
                        operator,
                        args,
                        kwargs,
                        dtype,
                        accuracy_checking,
                        repeats,
                        measure_nvfuser,
                        device,
                    )
                )
            except Exception as e:
                # 如果发生异常，打印异常信息和操作符名称
                print(f"error {operator}")
                print(e)
                # 可以将下面的 raise e 注释掉以避免阻止其他测试
                # raise e

        # 如果 timings 列表为空，则继续下一个操作符的循环
        if not timings:
            continue

        # 将 timings 转换为 Torch 张量，并取其转置
        timings = torch.tensor(timings).T

        # 定义一个包含 0.2、0.5、0.8 分位数的 Torch 张量 q
        q = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)

        # 构造输出字符串 output，包括操作符名称和 Inductor 的速度提升信息
        output = f"{operator}:\nInductor Speedups : {(torch.quantile(timings[0] / timings[1], q)).tolist()}\n"

        # 如果 measure_nvfuser 为 True，则添加 NVFUSER 的速度提升信息到 output 中
        if measure_nvfuser:
            output += f"NVFUSER Speedups :{(torch.quantile(timings[0] / timings[2], q)).tolist()}\n"

        # 如果操作为 "all"，将 output 写入文件 f
        if op == "all":
            f.write(output)

        # 打印 output 到控制台
        print(output)

    # 如果操作为 "all"，关闭文件 f
    if op == "all":
        f.close()
# 如果这个脚本被直接执行（而不是被导入作为模块），则执行 benchmark 函数
if __name__ == "__main__":
    benchmark()
```