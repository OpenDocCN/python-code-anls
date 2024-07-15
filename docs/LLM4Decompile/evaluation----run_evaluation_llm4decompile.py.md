# `.\LLM4Decompile\evaluation\run_evaluation_llm4decompile.py`

```py
# 导入必要的子模块和库
import subprocess  # 子进程管理模块，用于执行外部命令
import asyncio  # 异步编程库，用于异步执行任务
from transformers import AutoTokenizer  # 从transformers库中导入AutoTokenizer类
import os  # 操作系统相关功能的标准库
import json  # 处理JSON格式数据的标准库
from loguru import logger  # 强大的日志库Loguru
import traceback  # 提供异常堆栈追踪功能的标准库
from argparse import ArgumentParser  # 命令行参数解析模块
from pathlib import Path  # 操作路径的对象导向的标准库
import sys  # 提供对Python解释器相关功能的访问
from tqdm import tqdm  # 进度条显示库，用于显示任务进度
from server.text_generation import TextGenerationServer, TextGenerationClient  # 导入文本生成服务器和客户端类
import tempfile  # 创建临时文件和目录的标准库
import multiprocessing  # 多进程处理模块

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")  # 将日志输出到标准输出，设置日志格式

# 解析命令行参数并返回解析结果
def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)  # 模型路径参数
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型参数，默认为bfloat16
    parser.add_argument("--port", type=int, default=8080)  # 端口号参数，默认为8080
    parser.add_argument("--max_input_len", type=int, default=8192)  # 最大输入长度参数，默认为8192
    parser.add_argument("--max_total_tokens", type=int, default=8800)  # 总令牌数上限参数，默认为8800
    parser.add_argument("--max_batch_prefill_tokens", type=int, default=72000)  # 批次预填充令牌数上限参数，默认为72000
    parser.add_argument("--num_shards", type=int, default=4)  # 分片数参数，默认为4
    parser.add_argument("--max_new_tokens", type=int, default=512)  # 最大新令牌数参数，默认为512
    parser.add_argument("--repeat", type=int, default=1)  # 重复次数参数，默认为1
    parser.add_argument("--testset_path", type=str)  # 测试集路径参数
    parser.add_argument("--num_workers", type=int, default=16)  # 工作进程数参数，默认为16
    return parser.parse_args()

# 定义评估函数，接收参数字典params作为输入
def evaluate_func(params):
    # 从params字典中获取C函数、测试用例和反编译后的C函数字符串
    c_func, c_test, c_func_decompile = (
        params["c_func"],
        params["c_test"],
        params["c_func_decompile"],
    )

    timeout = 10  # 设置超时时间为10秒
    flag_compile = 0  # 编译标志位初始化为0
    flag_run = 0  # 运行标志位初始化为0
    c_include = ""  # C语言头文件字符串初始化为空
    # 遍历C函数字符串中的每一行
    for line in c_func.split("\n"):
        if "#include" in line:  # 如果当前行包含#include
            c_include += line + "\n"  # 将该行添加到头文件字符串中
            c_func = c_func.replace(line, "")  # 在C函数字符串中删除该行
    # 遍历测试用例字符串中的每一行
    for line in c_test.split("\n"):
        if "#include" in line:  # 如果当前行包含#include
            c_include += line + "\n"  # 将该行添加到头文件字符串中
            c_test = c_test.replace(line, "")  # 在测试用例字符串中删除该行
    # 拼接头文件字符串、反编译后的C函数字符串和测试用例字符串，形成组合的C代码字符串
    c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test
    # 拼接头文件字符串和反编译后的C函数字符串，形成只包含函数的C代码字符串
    c_onlyfunc = c_include + "\n" + c_func_decompile
    # 使用临时目录作为工作空间，自动管理临时目录的生命周期
    with tempfile.TemporaryDirectory() as temp_dir:
        # 获取当前进程的 PID
        pid = os.getpid()
        # 组合出要写入的合并后 C 代码文件路径
        c_file = os.path.join(temp_dir, f"combine_{pid}.c")
        # 组合出生成的可执行文件路径
        executable = os.path.join(temp_dir, f"combine_{pid}")
        # 组合出只含有函数的 C 代码文件路径
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        # 组合出生成的只含有函数的可执行文件路径
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")

        # 将合并后的 C 代码写入文件
        with open(c_file, "w") as f:
            f.write(c_combine)
        # 将只含有函数的 C 代码写入文件
        with open(c_file_onlyfunc, "w") as f:
            f.write(c_onlyfunc)

        # 编译只含有函数的 C 代码为汇编语言文件
        compile_command = [
            "gcc",
            "-S",
            c_file_onlyfunc,
            "-o",
            executable_onlyfunc,
            "-lm",
        ]
        try:
            # 执行编译命令，检查并设定超时时间
            subprocess.run(compile_command, check=True, timeout=timeout)
            # 标记编译成功
            flag_compile = 1
        except:
            # 发生异常则返回编译和运行标志
            return flag_compile, flag_run

        # 编译合并后的 C 代码为可执行文件
        compile_command = ["gcc", c_file, "-o", executable, "-lm"]
        try:
            # 执行编译命令，检查并设定超时时间
            subprocess.run(compile_command, check=True, timeout=timeout)
            # 标记编译成功
            flag_compile = 1
        except:
            # 发生异常则返回编译和运行标志
            return flag_compile, flag_run

        # 运行生成的可执行文件
        run_command = [executable]
        try:
            # 执行运行命令，捕获输出并设定超时时间，检查运行结果
            process = subprocess.run(
                run_command, capture_output=True, text=True, timeout=timeout, check=True
            )
            # 标记运行成功
            flag_run = 1
        except:
            # 如果程序已启动，则终止它并等待其结束
            if 'process' in locals() and process:
                process.kill()
                process.wait()
            # 返回编译和运行标志
            return flag_compile, flag_run

    # 返回最终的编译和运行标志
    return flag_compile, flag_run
# 定义一个函数，用于计算并返回解析通过率的统计信息
def decompile_pass_rate(testsets, gen_results_repeat, opts, args):
    # 存储所有生成结果的统计信息
    all_stats = []

    # 遍历每一次生成的结果
    for gen_index, gen_results in enumerate(gen_results_repeat):
        # 使用多进程池来并行处理任务
        with multiprocessing.Pool(args.num_workers) as pool:
            # 准备任务列表，每个任务包含要评估的函数及其结果
            tasks = [
                {
                    "c_func": testset["c_func"],              # 待评估的 C 函数
                    "c_test": testset["c_test"],              # 用于测试的 C 代码
                    "c_func_decompile": output[0],            # 解析得到的 C 函数
                }
                for testset, output in zip(testsets, gen_results)
            ]

            # 使用 tqdm 来展示任务处理进度，并且评估每个任务的结果
            eval_results = list(tqdm(pool.imap(evaluate_func, tasks), total=len(tasks)))

        # 终止并等待所有子进程完成
        pool.terminate()
        pool.join()
        
        # 初始化当前优化选项的统计信息
        stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
        
        # 遍历每个测试集及其生成结果和评估结果，计算统计信息
        for idx, (testset, output, flag) in enumerate(
            tqdm(
                zip(testsets, gen_results, eval_results),
                total=len(testsets),
                desc="Evaluating",
            )
        ):
            # 获取当前测试集的解析后的 C 函数及相关信息
            c_func_decompile = output[0]
            c_func = testset["c_func"]         # 待评估的原始 C 函数
            c_test = testset["c_test"]         # 用于测试的 C 代码

            # 获取评估结果中的编译及运行标志
            flag_compile, flag_run = flag[0], flag[1]
            opt = testset["type"]               # 当前测试集的优化选项

            # 更新当前优化选项的总数统计
            stats[opt]["total"] += 1
            # 如果编译成功，增加编译成功次数统计
            if flag_compile:
                stats[opt]["compile"] += 1
            # 如果运行成功，增加运行成功次数统计
            if flag_run:
                stats[opt]["run"] += 1

        # 将当前优化选项的统计信息添加到总统计列表中
        all_stats.append(stats)

    # 计算平均统计信息
    avg_stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in opts}
    for stats in all_stats:
        for opt in opts:
            avg_stats[opt]["compile"] += stats[opt]["compile"]
            avg_stats[opt]["run"] += stats[opt]["run"]
            avg_stats[opt]["total"] += stats[opt]["total"]

    for opt in opts:
        # 计算平均编译通过率和运行通过率，并输出结果
        avg_stats[opt]["compile"] /= len(gen_results_repeat)
        avg_stats[opt]["run"] /= len(gen_results_repeat)
        avg_stats[opt]["total"] /= len(gen_results_repeat)

    for opt, data in avg_stats.items():
        compile_rate = data["compile"] / data["total"] if data["total"] > 0 else 0
        run_rate = data["run"] / data["total"] if data["total"] > 0 else 0
        # 输出每个优化选项的编译通过率和运行通过率
        print(
            f"Optimization {opt}: Compile Rate: {compile_rate:.4f}, Run Rate: {run_rate:.4f}"
        )

    # 返回状态码 0，表示成功完成函数执行
    return 0


# 定义一个函数，用于运行评估流水线，参数为命令行参数解析器对象，返回执行结果状态码
def run_eval_pipeline(args: ArgumentParser) -> int:
    # 将命令行参数中的模型路径转为 Path 对象
    model_path = Path(args.model_path)
    # 检查模型路径是否存在且为目录，若不是则记录错误并返回 -1
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Invalid model {model_path}")
        return -1
    try:
        # 试图从文件中加载测试集数据，并转换为 Python 对象
        testsets = json.load(open(args.testset_path, "r"))
        # 记录日志，显示加载了多少个测试用例
        logger.info(f"Loaded testset with {len(testsets)} cases")
        # 根据模型路径加载对应的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 设置停止生成的序列，这里只包含一个终止符
        stop_sequences = [tokenizer.eos_token]
    
        # 不同优化级别对应的提示信息
        opts = {
            "O0": "# This is the assembly code with O0 optimization:\n",
            "O1": "# This is the assembly code with O1 optimization:\n",
            "O2": "# This is the assembly code with O2 optimization:\n",
            "O3": "# This is the assembly code with O3 optimization:\n",
        }
    
        # 每个输入后面追加的附加信息
        after = "\n# What is the source code?\n"
    
        # 存储所有的输入生成提示信息的列表
        inputs = []
    
        # 遍历所有的测试集
        for testset in testsets:
            # 获取当前测试集的汇编输入提示信息
            input_asm_prompt = testset["input_asm_prompt"]
            # 获取当前测试集的优化级别
            opt = testset["type"]
            # 组合优化级别、汇编输入和附加信息，形成完整的生成提示信息
            prompt = opts[opt] + input_asm_prompt + after
            # 将生成的提示信息添加到输入列表中
            inputs.append(prompt)
    
        # 初始化文本生成服务器
        text_gen_server = TextGenerationServer(
            str(model_path),
            args.port,
            args.dtype,
            args.max_input_len,
            args.max_total_tokens,
            args.max_batch_prefill_tokens,
            args.num_shards,
        )
    
        # 初始化文本生成客户端
        text_gen_client = TextGenerationClient(
            port=args.port, stop_sequences=stop_sequences
        )
    
        # 存储生成的结果列表
        gen_results_repeat = []
        # 记录日志，显示实验将循环多少次
        logger.info(f"The exp will loop for {args.repeat} times....")
        # 根据指定次数执行循环
        for i in range(args.repeat):
            # 记录当前循环次数
            logger.info(f"The {i+1} loop...")
            # 获取当前循环的事件循环对象
            loop = asyncio.get_event_loop()
            # 设置当前循环的事件循环对象
            asyncio.set_event_loop(loop)
            # 执行生成代码结果的异步操作，并获取结果
            gen_results = loop.run_until_complete(
                text_gen_client.generate_code_results(
                    inputs, args.max_new_tokens, num_outputs=1
                )
            )
            # 将生成的结果添加到重复生成结果列表中
            gen_results_repeat.append(gen_results)
    
    except Exception as e:
        # 如果发生异常，记录错误日志
        logger.error(e)
        # 打印异常堆栈信息
        traceback.print_exc()
        # 返回错误状态码
        return -1
    
    # 调用函数计算反编译通过率，并返回结果
    ret = decompile_pass_rate(testsets, gen_results_repeat, opts, args)
    return ret
# 主函数入口点
def main():
    # 解析命令行参数
    args = parse_args()
    # 运行评估管道，并获取返回值
    ret = run_eval_pipeline(args)
    # 使用返回值退出程序，返回值通常用于指示程序执行状态
    sys.exit(ret)


# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```