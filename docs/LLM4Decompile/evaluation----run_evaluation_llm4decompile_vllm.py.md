# `.\LLM4Decompile\evaluation\run_evaluation_llm4decompile_vllm.py`

```py
# 导入子进程模块，用于执行外部命令
import subprocess
# 导入异步 I/O 模块，用于异步操作
import asyncio
# 导入 transformers 库中的自动分词器
from transformers import AutoTokenizer
# 导入操作系统模块
import os
# 导入 JSON 模块，用于处理 JSON 格式数据
import json
# 导入日志记录模块 Loguru 中的日志对象
from loguru import logger
# 导入异常回溯模块，用于异常时的详细信息记录
import traceback
# 导入命令行参数解析模块
from argparse import ArgumentParser
# 导入路径操作模块
from pathlib import Path
# 导入系统相关模块
import sys
# 导入进度条模块
from tqdm import tqdm
# 导入多进程模块
import multiprocessing
# 导入临时文件模块
import tempfile
# 导入自定义模块 vllm 中的 LLM 类和 SamplingParams 类
from vllm import LLM, SamplingParams

# 将日志输出到标准输出，设置日志格式
logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")
# 设置环境变量以启用分词器的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# 定义命令行参数解析函数，返回 ArgumentParser 对象
def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    # 添加模型路径参数
    parser.add_argument("--model_path", type=str)
    # 添加 GPU 数量参数，默认为 8
    parser.add_argument("--gpus", type=int, default=8)
    # 添加最大序列数量参数，默认为 8
    parser.add_argument("--max_num_seqs", type=int, default=8)
    # 添加 GPU 内存利用率参数，默认为 0.82
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    # 添加生成温度参数，默认为 0
    parser.add_argument("--temperature", type=float, default=0)
    # 添加最大总标记数参数，默认为 8192
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    # 添加最大新增标记数参数，默认为 512
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # 添加重复次数参数，默认为 1
    parser.add_argument("--repeat", type=int, default=1)
    # 添加测试集路径参数
    parser.add_argument("--testset_path", type=str)
    # 添加输出路径参数，默认为 None
    parser.add_argument("--output_path", type=str, default=None)
    # 添加工作进程数量参数，默认为 16
    parser.add_argument("--num_workers", type=int, default=16)
    # 解析并返回命令行参数
    return parser.parse_args()


# 定义评估函数，参数为包含需要评估的 C 代码的字典 params
def evaluate_func(params):
    # 从 params 中获取 C 函数、C 测试、反编译后的 C 函数字符串
    c_func, c_test, c_func_decompile = (
        params["c_func"],
        params["c_test"],
        params["c_func_decompile"],
    )

    # 设置超时时间为 10 秒
    timeout = 10
    # 初始化编译标志为 0
    flag_compile = 0
    # 初始化运行标志为 0
    flag_run = 0
    # 初始化包含 C 函数的头文件字符串
    c_include = ""

    # 遍历 C 函数字符串的每一行
    for line in c_func.split("\n"):
        # 如果当前行包含 "#include" 字符串
        if "#include" in line:
            # 将该行添加到 c_include 中
            c_include += line + "\n"
            # 在 c_func 中移除该行
            c_func = c_func.replace(line, "")

    # 遍历 C 测试字符串的每一行
    for line in c_test.split("\n"):
        # 如果当前行包含 "#include" 字符串
        if "#include" in line:
            # 将该行添加到 c_include 中
            c_include += line + "\n"
            # 在 c_test 中移除该行
            c_test = c_test.replace(line, "")

    # 组合包含了头文件、反编译函数和测试函数的完整 C 代码
    c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test
    # 组合只包含头文件和反编译函数的 C 代码
    c_onlyfunc = c_include + "\n" + c_func_decompile
    # 使用临时目录作为上下文管理器，确保代码执行完毕后临时文件被清理
    with tempfile.TemporaryDirectory() as temp_dir:
        # 获取当前进程的 PID
        pid = os.getpid()
        # 组合文件名：combine_<pid>.c 和 combine_<pid>，用于存放合并后的 C 代码和可执行文件
        c_file = os.path.join(temp_dir, f"combine_{pid}.c")
        executable = os.path.join(temp_dir, f"combine_{pid}")
        # 组合文件名：onlyfunc_<pid>.c 和 onlyfunc_<pid>，用于存放仅包含函数的 C 代码和对应的可执行文件
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")

        # 将合并后的 C 代码写入 combine_<pid>.c 文件
        with open(c_file, "w") as f:
            f.write(c_combine)
        # 将仅包含函数的 C 代码写入 onlyfunc_<pid>.c 文件
        with open(c_file_onlyfunc, "w") as f:
            f.write(c_onlyfunc)

        # 编译仅包含函数的 C 代码为汇编文件 executable_onlyfunc，链接 math 库
        compile_command = [
            "gcc",
            "-S",
            c_file_onlyfunc,
            "-o",
            executable_onlyfunc,
            "-lm",
        ]
        try:
            # 执行编译命令，检查并设置编译成功标志为 1
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except:
            # 发生异常时返回当前编译状态和运行状态
            return flag_compile, flag_run

        # 编译包含全部代码的 C 文件为可执行文件 executable，链接 math 库
        compile_command = ["gcc", c_file, "-o", executable, "-lm"]
        try:
            # 执行编译命令，检查并设置编译成功标志为 1
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except:
            # 发生异常时返回当前编译状态和运行状态
            return flag_compile, flag_run

        # 运行已编译的可执行文件
        run_command = [executable]
        try:
            # 运行命令，捕获输出，设置运行成功标志为 1
            process = subprocess.run(
                run_command, capture_output=True, text=True, timeout=timeout, check=True
            )
            flag_run = 1
        except:
            # 发生异常时，如果 process 对象存在，则终止进程并等待，然后返回编译状态和运行状态
            if "process" in locals() and process:
                process.kill()
                process.wait()
            return flag_compile, flag_run

    # 最终返回编译状态和运行状态
    return flag_compile, flag_run
# 计算各个优化选项的编译通过率和运行通过率的平均值
def decompile_pass_rate(testsets, gen_results_repeat, opts, args):
    # 存储所有统计数据的列表
    all_stats = []

    # 遍历生成结果的重复次数
    for gen_index, gen_results in enumerate(gen_results_repeat):
        # 使用多进程池并行处理任务
        with multiprocessing.Pool(args.num_workers) as pool:
            # 准备要并行评估的任务列表
            tasks = [
                {
                    "c_func": testset["c_func"],                  # C 函数名
                    "c_test": testset["c_test"],                  # C 测试数据
                    "c_func_decompile": output[0],                # 反编译得到的 C 函数代码
                }
                for testset, output in zip(testsets, gen_results)
            ]

            # 使用进度条展示并行任务的进度，并收集评估结果
            eval_results = list(tqdm(pool.imap(evaluate_func, tasks), total=len(tasks)))

        # 终止并关闭进程池
        pool.terminate()
        pool.join()

        # 初始化统计数据字典，记录每种优化选项的编译通过次数、运行通过次数和总数
        stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
        # 遍历测试集、生成结果和评估结果，更新统计数据
        for idx, (testset, output, flag) in enumerate(
            tqdm(
                zip(testsets, gen_results, eval_results),
                total=len(testsets),
                desc="Evaluating",
            )
        ):
            c_func_decompile = output[0]                       # 反编译得到的 C 函数代码
            c_func = testset["c_func"]                         # C 函数名
            c_test = testset["c_test"]                         # C 测试数据

            flag_compile, flag_run = flag[0], flag[1]          # 编译通过标志、运行通过标志
            opt = testset["type"]                              # 优化选项

            stats[opt]["total"] += 1                           # 总数加一
            if flag_compile:
                stats[opt]["compile"] += 1                     # 如果编译通过，编译通过次数加一
            if flag_run:
                stats[opt]["run"] += 1                         # 如果运行通过，运行通过次数加一

        # 将当前优化选项的统计数据添加到总统计列表
        all_stats.append(stats)

    # 计算各个优化选项的平均编译通过率和运行通过率
    avg_stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
    for stats in all_stats:
        for opt in opts:
            avg_stats[opt]["compile"] += stats[opt]["compile"]   # 累加编译通过次数
            avg_stats[opt]["run"] += stats[opt]["run"]           # 累加运行通过次数
            avg_stats[opt]["total"] += stats[opt]["total"]       # 累加总数

    # 计算平均值
    for opt in opts:
        avg_stats[opt]["compile"] /= len(gen_results_repeat)      # 平均编译通过率
        avg_stats[opt]["run"] /= len(gen_results_repeat)          # 平均运行通过率
        avg_stats[opt]["total"] /= len(gen_results_repeat)        # 平均总数

    # 输出每种优化选项的平均编译通过率和运行通过率
    for opt, data in avg_stats.items():
        compile_rate = data["compile"] / data["total"] if data["total"] > 0 else 0
        run_rate = data["run"] / data["total"] if data["total"] > 0 else 0
        print(
            f"Optimization {opt}: Compile Rate: {compile_rate:.4f}, Run Rate: {run_rate:.4f}"
        )

    # 返回成功状态码
    return 0


# 运行评估管道，检查模型路径的有效性，若无效则记录错误并返回失败状态码
def run_eval_pipeline(args: ArgumentParser) -> int:
    model_path = Path(args.model_path)                         # 模型路径
    if not model_path.exists() or not model_path.is_dir():     # 如果路径不存在或不是目录
        logger.error(f"Invalid model {model_path}")             # 记录错误日志
        return -1                                               # 返回失败状态码
    # 尝试加载测试集文件，以 JSON 格式读取
    try:
        testsets = json.load(open(args.testset_path, "r"))
        # 打印加载的测试集数量信息
        logger.info(f"Loaded testset with {len(testsets)} cases")
        # 从预训练模型路径加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 停止生成的标记序列
        stop_sequences = [tokenizer.eos_token]

        # 设置选项字典，不同优化等级对应不同的注释
        opts = {
            "O0": "# This is the assembly code:\n",
            "O1": "# This is the assembly code:\n",
            "O2": "# This is the assembly code:\n",
            "O3": "# This is the assembly code:\n",
        }

        # 设置输入内容的后续注释
        after = "\n# What is the source code?\n"
        inputs = []
        # 遍历测试集，构建输入内容
        for testset in testsets:
            # 获取输入的汇编提示
            input_asm_prompt = testset["input_asm_prompt"]
            # 获取测试集类型对应的选项注释，构建完整的输入内容
            opt = testset["type"]
            prompt = opts[opt] + input_asm_prompt + after
            inputs.append(prompt)

        # 初始化长文本生成模型
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.gpus,
            max_model_len=args.max_total_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            stop=stop_sequences,
        )

        # 重复生成结果列表
        gen_results_repeat = []
        # 打印循环次数信息
        logger.info(f"The exp will loop for {args.repeat} times....")
        # 执行循环生成
        for i in range(args.repeat):
            logger.info(f"The {i+1} loop...")
            # 生成结果
            gen_results = llm.generate(inputs, sampling_params)
            # 提取生成结果文本
            gen_results = [[output.outputs[0].text] for output in gen_results]
            # 将结果添加到重复生成结果列表中
            gen_results_repeat.append(gen_results)

    # 捕获异常并记录错误信息
    except Exception as e:
        logger.error(e)
        # 打印异常堆栈信息
        traceback.print_exc()
        # 返回错误码
        return -1

    # 构建保存数据列表
    save_data = []
    # 将生成结果与测试集对应，并保存到保存数据列表中
    for testset, res in zip(testsets, gen_results_repeat[0]):
        testset["output"] = res[0]
        save_data.append(testset)

    # 如果指定了输出路径，则将保存数据列表写入到指定文件中
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=True)

    # 计算反编译通过率
    ret = decompile_pass_rate(testsets, gen_results_repeat, opts, args)
    # 返回反编译通过率
    return ret
# 主程序入口点，程序从这里开始执行
def main():
    # 解析命令行参数，获取程序运行所需的参数配置
    args = parse_args()
    # 运行评估管道，并获取返回值
    ret = run_eval_pipeline(args)
    # 使用返回值作为退出码，退出程序
    sys.exit(ret)

# 如果当前脚本被作为主程序执行，则调用主函数 main()
if __name__ == "__main__":
    main()
```