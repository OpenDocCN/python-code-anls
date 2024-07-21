# `.\pytorch\benchmarks\tensorexpr\__main__.py`

```py
import argparse  # 导入 argparse 模块，用于解析命令行参数
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import os  # 导入 os 模块，提供了许多与操作系统交互的功能

# 以下为从不同模块导入特定符号的注释，所有导入均为 F401 类型，表示未使用警告（不应删除）
from . import (
    attention,     # 导入注意力机制相关功能
    benchmark,     # 导入基准测试相关功能
    broadcast,     # 导入广播相关功能
    concat,        # 导入拼接相关功能
    elementwise,   # 导入逐元素操作相关功能
    matmul,        # 导入矩阵乘法相关功能
    reduction,     # 导入降维操作相关功能
    rnn_eltwise,   # 导入循环神经网络逐元素操作相关功能
    softmax,       # 导入 softmax 操作相关功能
    swish,         # 导入 Swish 激活函数相关功能
    tensor_engine, # 导入张量引擎相关功能
)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Benchmark operators in specific shapes.
Works only with Python3.\n A few examples:
  * benchmark.py: runs all the default configs with all the benchmarks.
  * benchmark.py reduce: runs all the default configs with all benchmark with a prefix 'reduce'
  * benchmark.py layernorm_fwd_cpu_128_32_128_128: run a particular benchmark in that config""",
    )
    parser.add_argument(
        "benchmark_names",
        type=str,
        default=None,
        nargs="*",
        help="name of the benchmark to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu,cuda",
        help="a comma separated list of device names",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd,both",
        help="a comma separated list of running modes",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="a comma separated list of Data Types: {float32[default], float16}",
    )
    parser.add_argument(
        "--input-iter",
        type=str,
        default=None,
        help="a comma separated list of Tensor dimensions that includes a start, \
              stop, and increment that can be constant or a power of 2 \
              {start:stop:inc,start:stop:pow2}",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="pt",
        help="the underlying tensor engine. only pt for now",
    )
    parser.add_argument(
        "--jit-mode",
        "--jit_mode",
        type=str,
        default="trace",
        help="the jit mode to use: one of {trace, none}",
    )
    parser.add_argument(
        "--cuda-pointwise-loop-levels",
        "--cuda_pointwise_loop_levels",
        type=int,
        default=None,
        help="num of loop levesl for Cuda pointwise operations: 2 or 3",
    )
    parser.add_argument(
        "--cuda-pointwise-block-count",
        "--cuda_pointwise_block_count",
        type=int,
        default=None,
        help="num of block for Cuda pointwise operations",
    )
    parser.add_argument(
        "--cuda-pointwise-block-size",
        "--cuda_pointwise_block_size",
        type=int,
        default=None,
        help="num of blocks for Cuda pointwise operations",
    )
    # 添加一个名为 "--cuda-fuser" 的命令行参数，用于选择 CUDA 融合后端
    parser.add_argument(
        "--cuda-fuser",
        "--cuda_fuser",
        type=str,
        default="te",
        help="The Cuda fuser backend to use: one of {te, nvf, old, none}",
    )
    
    # 添加一个名为 "--output" 的命令行参数，指定基准运行的输出格式，默认为 stdout
    parser.add_argument(
        "--output",
        type=str,
        default="stdout",
        help="The output format of the benchmark run {stdout[default], json}",
    )
    
    # 添加一个名为 "--print-ir" 的命令行参数，如果设置则打印融合的 IR 图
    parser.add_argument(
        "--print-ir",
        action="store_true",
        help="Print the IR graph of the Fusion.",
    )
    
    # 添加一个名为 "--print-kernel" 的命令行参数，如果设置则打印生成的内核
    parser.add_argument(
        "--print-kernel",
        action="store_true",
        help="Print generated kernel(s).",
    )
    
    # 添加一个名为 "--no-dynamic-shape" 的命令行参数，如果设置则禁用动态基准测试中的形状随机化
    parser.add_argument(
        "--no-dynamic-shape",
        action="store_true",
        help="Disable shape randomization in dynamic benchmarks.",
    )
    
    # 添加一个名为 "--cpu-fusion" 的命令行参数，如果设置则启用 CPU 融合
    parser.add_argument(
        "--cpu-fusion",
        "--cpu_fusion",
        default=False,
        action="store_true",
        help="Enable CPU fusion.",
    )
    
    # 添加一个名为 "--cat-wo-conditionals" 的命令行参数，如果设置则启用无条件的 CAT
    parser.add_argument(
        "--cat-wo-conditionals",
        "--cat_wo_conditionals",
        default=False,
        action="store_true",
        help="Enable CAT wo conditionals.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 根据 "--cuda-fuser" 参数的值进行不同的设置
    if args.cuda_fuser == "te":
        import torch
        
        # 设置 Torch 的 JIT 执行器为性能分析模式
        torch._C._jit_set_profiling_executor(True)
        # 启用 Torch 的 Texpr 融合器
        torch._C._jit_set_texpr_fuser_enabled(True)
        # 覆盖 GPU 上的融合能力判断
        torch._C._jit_override_can_fuse_on_gpu(True)
        # 获取图执行器的优化设置
        torch._C._get_graph_executor_optimize(True)
    elif args.cuda_fuser == "old":
        import torch
        
        # 禁用 Torch 的 JIT 执行器性能分析模式
        torch._C._jit_set_profiling_executor(False)
        # 禁用 Torch 的 Texpr 融合器
        torch._C._jit_set_texpr_fuser_enabled(False)
        # 覆盖 GPU 上的融合能力判断
        torch._C._jit_override_can_fuse_on_gpu(True)
    elif args.cuda_fuser == "nvf":
        import torch
        
        # 设置 Torch 的 JIT 执行器为性能分析模式
        torch._C._jit_set_profiling_executor(True)
        # 禁用 Torch 的 Texpr 融合器
        torch._C._jit_set_texpr_fuser_enabled(False)
        # 启用 Torch 的 NVFuser
        torch._C._jit_set_nvfuser_enabled(True)
        # 获取图执行器的优化设置
        torch._C._get_graph_executor_optimize(True)
    else:
        # 如果未定义的融合器类型，则抛出 ValueError
        raise ValueError(f"Undefined fuser: {args.cuda_fuser}")

    # 根据 "--cpu-fusion" 参数的设置，选择是否启用 CPU 融合
    if args.cpu_fusion:
        import torch
        
        # 覆盖 CPU 上的融合能力判断
        torch._C._jit_override_can_fuse_on_cpu(True)
    else:
        import torch
        
        # 禁用 CPU 上的融合能力判断
        torch._C._jit_override_can_fuse_on_cpu(False)

    # 根据 "--cat-wo-conditionals" 参数的设置，选择是否启用无条件的 CAT
    if args.cat_wo_conditionals:
        import torch
        
        # 启用无条件的 CAT
        torch._C._jit_cat_wo_conditionals(True)
    else:
        import torch
        
        # 禁用无条件的 CAT
        torch._C._jit_cat_wo_conditionals(False)

    # 定义一个函数 set_global_threads，用于设置全局线程数环境变量
    def set_global_threads(num_threads):
        import os
        
        # 设置并发执行线程数环境变量
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["TVM_NUM_THREADS"] = str(num_threads)
        os.environ["NNC_NUM_THREADS"] = str(num_threads)

    # 将 args.device 按逗号分割成设备列表
    devices = args.device.split(",")
    # 如果设备列表中包含 'gpu'，则替换为 'cuda'
    devices = ["cuda" if device == "gpu" else device for device in devices]
    
    # 初始化 cpu_count 变量为 0
    cpu_count = 0
    # 遍历设备列表，并获取索引和每个设备的值
    for index, device in enumerate(devices):
        # 检查设备名是否以"cpu"开头
        if device.startswith("cpu"):
            # 如果是CPU设备，增加CPU计数器
            cpu_count += 1
            # 如果CPU计数超过1，抛出数值错误异常
            if cpu_count > 1:
                raise ValueError(
                    "more than one CPU device is not allowed: %d" % (cpu_count)
                )
            # 如果设备名为"cpu"，跳过本次循环
            if device == "cpu":
                continue
            # 从设备名中提取线程数字符串
            num_threads_str = device[3:]
            try:
                # 尝试将线程数字符串转换为整数
                num_threads = int(num_threads_str)
                # 设置全局线程数
                set_global_threads(num_threads)
                # 将设备名改为"cpu"
                devices[index] = "cpu"
            except ValueError:
                # 如果转换失败，继续下一个循环
                continue

    # 将命令行参数中的模式按逗号分隔成列表
    modes = args.mode.split(",")

    # 将命令行参数中的数据类型按逗号分隔成列表，并转换为torch库中的数据类型
    datatypes = args.dtype.split(",")
    for index, dtype in enumerate(datatypes):
        datatypes[index] = getattr(torch, dtype)
        # 如果转换后的数据类型不存在，抛出属性错误异常
        if not datatypes[index]:
            raise AttributeError(f"DataType: {dtype} is not valid!")

    # 设置张量计算引擎的模式
    tensor_engine.set_engine_mode(args.engine)

    # 定义一个运行默认配置的函数，遍历模式、设备、数据类型和配置的组合
    def run_default_configs(bench_cls, allow_skip=True):
        for mode, device, dtype, config in itertools.product(
            modes, devices, datatypes, bench_cls.default_configs()
        ):
            # 创建基准测试对象，使用给定的模式、设备、数据类型和配置
            bench = bench_cls(mode, device, dtype, *config)
            # 设置基准测试对象的输出类型为命令行参数中指定的输出类型
            bench.output_type = args.output
            # 设置基准测试对象的即时编译模式为命令行参数中指定的即时编译模式
            bench.jit_mode = args.jit_mode
            # 如果当前基准测试对象不受支持，根据allow_skip参数决定是否跳过或抛出数值错误异常
            if not bench.is_supported():
                if allow_skip:
                    continue
                else:
                    raise ValueError(
                        f"attempted to run an unsupported benchmark: {bench.desc()}"
                    )
            # 运行当前基准测试对象的运行方法，使用命令行参数传递额外的参数
            bench.run(args)
    # 定义一个函数，用于执行带有输入迭代器的基准测试
    def run_with_input_iter(bench_cls, input_iter, allow_skip=True):
        # 将输入迭代器按逗号分割成维度规格列表
        tensor_dim_specs = input_iter.split(",")
        # 将每个维度规格再按冒号分割，形成列表的列表
        tensor_dim_specs = [dim.split(":") for dim in tensor_dim_specs]

        # 存储所有可能的配置
        configs = []
        # 遍历每个维度规格
        for start, stop, inc in tensor_dim_specs:
            dim_list = []
            # 根据增量类型生成维度列表
            if inc == "pow2":
                curr = int(start)
                while curr <= int(stop):
                    dim_list.append(curr)
                    curr <<= 1  # 使用位移操作计算下一个2的幂
            elif inc == "pow2+1":
                curr = int(start)
                while curr <= int(stop):
                    dim_list.append(curr)
                    curr -= 1
                    curr <<= 1
                    curr += 1  # 使用位移和加法计算下一个2的幂加1
            else:
                dim_list = list(range(int(start), int(stop) + int(inc), int(inc)))
            configs.append(dim_list)

        # 获取所有配置的笛卡尔积
        configs = itertools.product(*configs)

        # 遍历所有模式、设备、数据类型和配置的组合
        for mode, device, dtype, config in itertools.product(
            modes, devices, datatypes, list(configs)
        ):
            # 根据给定的类实例化基准测试对象
            bench = bench_cls(mode, device, dtype, *config)
            # 设置基准测试对象的输出类型
            bench.output_type = args.output
            # 设置基准测试对象的即时编译模式
            bench.jit_mode = args.jit_mode
            # 如果当前配置不受支持
            if not bench.is_supported():
                # 如果允许跳过不支持的基准测试，则继续下一个配置
                if allow_skip:
                    continue
                else:
                    # 否则抛出值错误异常，说明尝试运行一个不支持的基准测试
                    raise ValueError(
                        f"attempted to run an unsupported benchmark: {bench.desc()}"
                    )
            # 运行基准测试
            bench.run(args)

    # 获取所有基准测试类的列表
    benchmark_classes = benchmark.benchmark_classes
    # 如果没有指定特定的基准测试名称
    if not args.benchmark_names:
        # 默认情况下运行所有基准测试
        for benchmark_cls in benchmark_classes:
            run_default_configs(benchmark_cls, allow_skip=True)
    else:
        # 对于每个指定的基准测试名称
        for name in args.benchmark_names:
            # 如果名称是某个基准测试类的前缀，则运行该类的所有基准测试
            match_class_name = False
            for bench_cls in benchmark_classes:
                if name in bench_cls.module():
                    match_class_name = True
                    # 如果指定了输入迭代器，并且基准测试类支持输入迭代器，则使用输入迭代器运行基准测试
                    if (args.input_iter is not None) and bench_cls.input_iterable():
                        run_with_input_iter(bench_cls, args.input_iter, allow_skip=True)
                    else:
                        # 否则，如果指定了输入迭代器，则显示警告信息
                        if args.input_iter is not None:
                            print(
                                f"WARNING: Incompatible benchmark class called with input_iter arg: {name}"
                            )
                        # 运行默认配置的基准测试
                        run_default_configs(bench_cls, allow_skip=True)

            # 如果匹配到了类名，则继续处理下一个基准测试名称
            if match_class_name:
                continue

            # 如果不是类模块，则解析配置并调用相应的基准测试
            match_class_name = False
            for bench_cls in benchmark_classes:
                cls_module = bench_cls.module()
                if name.startswith(cls_module):
                    match_class_name = True
                    # 确保配置名称格式正确
                    if name[len(cls_module)] != "_":
                        raise ValueError(f"invalid name: {name}")
                    # 解析配置字符串
                    config_str = name[(len(cls_module) + 1) :]
                    config = config_str.split("_")
                    # 确保配置格式正确
                    if len(config) < 2:
                        raise ValueError(f"invalid config: {config}")
                    mode, device = config[0:2]
                    # TODO: 确保支持虚拟设备如 'cpu1' 和 'cpu4'
                    if mode not in ["fwd", "both"]:
                        raise ValueError(f"invalid mode: {mode}")
                    # 尝试将配置中的数值项转换为整数
                    for i, entry in enumerate(config):
                        try:
                            value = int(entry)
                            config[i] = value
                        except ValueError:
                            pass
                    # TODO: 在配置中输出数据类型，并从字符串中解析回来
                    # 根据解析的配置创建基准测试对象
                    bench = bench_cls(config[0], config[1], torch.float32, *config[2:])
                    bench.jit_mode = args.jit_mode
                    bench.output_type = args.output
                    # 运行基准测试
                    bench.run(args)

            # 如果没有匹配到类名，则抛出异常，显示可用的基准测试类名
            if not match_class_name:
                available_classes = ", ".join(
                    [bench_cls.module() for bench_cls in benchmark_classes]
                )
                raise ValueError(
                    f"invalid name: {name}\nAvailable benchmark classes:\n{available_classes}"
                )
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 main() 函数
if __name__ == "__main__":
    main()
```