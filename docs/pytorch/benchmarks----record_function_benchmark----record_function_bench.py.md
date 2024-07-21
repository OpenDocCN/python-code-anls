# `.\pytorch\benchmarks\record_function_benchmark\record_function_bench.py`

```py
# 导入命令行参数解析模块
import argparse
# 导入系统模块
import sys

# 导入自定义模块，用于创建 LSTM 模型
from benchmarks.fastrnns.factory import lstm_creator

# 导入 torchvision 中的 resnet50 模型
from torchvision.models import resnet50

# 导入 PyTorch 相关模块
import torch
import torch.utils.benchmark as benchmark_utils

# 准备用于创建 JIT 编译 LSTM 模型的函数
def prepare_lstm_jit(bench_args):
    # 调用 lstm_creator 函数创建 LSTM 模型定义
    model_def = lstm_creator(
        script=True,
        seqLength=bench_args.lstmSeqLength,
        numLayers=bench_args.lstmNumLayers,
        inputSize=bench_args.lstmInputSize,
        hiddenSize=bench_args.lstmHiddenSize,
        miniBatch=bench_args.lstmMiniBatch,
        device="cpu",
    )
    # 返回 LSTM 模型的输入和前向传播函数
    return model_def.inputs, model_def.forward

# 准备用于创建 JIT 编译 ResNet50 模型的函数
def prepare_resnet50_jit(bench_args):
    # 创建 ResNet50 模型实例
    model = resnet50()
    # 创建随机输入数据
    inputs = (torch.randn(32, 3, 224, 224),)
    # 对 ResNet50 模型进行 JIT 编译
    model = torch.jit.trace(model, inputs)
    # 返回输入数据和 JIT 编译后的 ResNet50 模型
    return inputs, model

# 定义模型名称与模型创建函数的映射关系
MODELS = {
    "resnet50_jit": prepare_resnet50_jit,
    "lstm_jit": prepare_lstm_jit,
}

# 定义用于多线程测试的线程数量列表
NUM_THREADS = [1, 2, 4, 8, 16, 32]

# 运行性能测试函数
def run_bench(model_names, bench_args):
    # 存储每个模型运行结果的列表
    results = []
    # 遍历每个模型名称
    for model_name in model_names:
        # 根据模型名称获取对应的模型创建函数
        model_creator = MODELS[model_name]
        # 调用模型创建函数创建模型及其输入数据
        inputs, model = model_creator(bench_args)

        # 打印当前正在测试的模型名称
        print("Benchmarking RecordFunction overhead for", model_name)
        # 执行预热操作
        print("Running warmup...", end=" ")
        sys.stdout.flush()
        # 进行预热运行
        for _ in range(bench_args.warmup):
            model(*inputs)
        print("finished")

        # 遍历不同线程数量
        for num_threads in NUM_THREADS:
            # 遍历是否开启 RecordFunction 的选项
            for with_rec_fn in [True, False]:
                # 根据 with_rec_fn 设置是否启用 RecordFunction
                torch.autograd._enable_record_function(with_rec_fn)
                torch.autograd._clear_callbacks()
                if with_rec_fn:
                    torch.autograd._set_empty_test_observer(True, 0.0001)

                # 打印当前测试的配置信息
                print(
                    "Running {} RecordFunction, num threads {} ...".format(
                        "with" if with_rec_fn else "without", num_threads
                    ),
                    end=" ",
                )
                sys.stdout.flush()
                # 创建性能计时器对象
                timer = benchmark_utils.Timer(
                    stmt="model(*inputs)",
                    globals={"model": model, "inputs": inputs},
                    description=model_name,
                    label="Record function overhead",
                    sub_label=f"with{'' if with_rec_fn else 'out'}_rec_fn, num_threads {num_threads}",
                    num_threads=num_threads,
                )
                # 运行性能测试并获取结果
                result = timer.blocked_autorange(
                    min_run_time=bench_args.timer_min_run_time
                )
                print("finished")
                # 打印性能测试结果
                print(result)
                sys.stdout.flush()
                # 将结果添加到结果列表中
                results.append(result)

    # 创建性能对比对象
    comparison = benchmark_utils.Compare(results)
    # 对比结果修剪到合适的有效数字
    comparison.trim_significant_figures()
    # 高亮显示任何潜在的警告信息
    comparison.highlight_warnings()
    # 打印性能对比结果
    comparison.print()

# 程序入口点，如果作为主程序运行则执行以下操作
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Benchmark RecordFunction overhead for ResNet and LSTM models"
    )
    # 添加一个参数，用于指定要运行的模型列表，默认为["lstm_jit"]
    parser.add_argument(
        "--models",
        nargs="*",
        default=["lstm_jit"],
        help="What model to run: " + str(MODELS.keys()),
    )

    # 添加一个参数，指定LSTM模型的序列长度，默认为100
    parser.add_argument("--lstmSeqLength", default="100", type=int)
    
    # 添加一个参数，指定LSTM模型的层数，默认为1
    parser.add_argument("--lstmNumLayers", default="1", type=int)
    
    # 添加一个参数，指定LSTM模型的输入尺寸，默认为512
    parser.add_argument("--lstmInputSize", default="512", type=int)
    
    # 添加一个参数，指定LSTM模型的隐藏单元尺寸，默认为512
    parser.add_argument("--lstmHiddenSize", default="512", type=int)
    
    # 添加一个参数，指定LSTM模型的迷你批次大小，默认为64
    parser.add_argument("--lstmMiniBatch", default="64", type=int)
    
    # 添加一个参数，指定预热次数，默认为2
    parser.add_argument("--warmup", default="2", type=int)
    
    # 添加一个参数，指定循环次数，默认为50
    parser.add_argument("--nloops", default="50", type=int)
    
    # 添加一个参数，指定计时器的最小运行时间，默认为120秒
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=120, type=int
    )

    # 解析命令行参数，并将结果保存在args变量中
    args = parser.parse_args()

    # 如果未提供--models参数，则使用所有可用的模型列表
    models = args.models or MODELS.keys()

    # 确保每个指定的模型在MODELS字典中存在
    for model in models:
        assert model in MODELS

    # 调用run_bench函数，传递模型列表和命令行参数args
    run_bench(models, args)
```