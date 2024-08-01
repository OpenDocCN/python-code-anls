# `.\DB-GPT-src\dbgpt\util\benchmarks\llm\llm_benchmarks.py`

```py
import argparse
import asyncio
import csv
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List

from dbgpt.configs.model_config import LLM_MODEL_CONFIG, ROOT_PATH
from dbgpt.core import ModelInferenceMetrics, ModelOutput
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.model.cluster.worker.manager import (
    WorkerManager,
    initialize_worker_manager_in_client,
    run_worker_manager,
)

model_name = "vicuna-7b-v1.5"
model_path = LLM_MODEL_CONFIG[model_name]
# or vllm
model_type = "huggingface"

controller_addr = "http://127.0.0.1:5670"

result_csv_file = None

parallel_nums = [1, 2, 4, 16, 32]
# parallel_nums = [1, 2, 4]


def get_result_csv_file() -> str:
    """
    返回结果 CSV 文件的路径
    """
    return os.path.join(
        ROOT_PATH, f"pilot/data/{model_name}_{model_type}_benchmarks_llm.csv"
    )


input_lens = [64, 64]
output_lens = [256, 512]


prompt_file_map = {
    "11k": os.path.join(
        ROOT_PATH, "docker/examples/benchmarks/benchmarks_llm_11k_prompt.txt"
    )
}

METRICS_HEADERS = [
    # Params
    "model_name",
    "gpu_nums",
    "parallel_nums",
    "input_length",
    "output_length",
    # Merge parallel result
    "test_time_cost_ms",
    "test_total_tokens",
    # avg_test_speed_per_second: (tokens / s), test_total_tokens / (test_time_cost_ms / 1000.0)
    "avg_test_speed_per_second(tokens/s)",
    # avg_first_token_latency_ms: sum(first_token_time_ms) / parallel_nums
    "avg_first_token_latency_ms",
    # avg_latency_ms: sum(end_time_ms - start_time_ms) / parallel_nums
    "avg_latency_ms",
    "gpu_mem(GiB)",
    # Detail for each task
    "start_time_ms",
    "end_time_ms",
    "current_time_ms",
    "first_token_time_ms",
    "first_completion_time_ms",
    "first_completion_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "speed_per_second",
]


def read_prompt_from_file(file_key: str) -> str:
    """
    从文件中读取提示文本并返回
    """
    full_path = prompt_file_map[file_key]
    with open(full_path, "r+", encoding="utf-8") as f:
        return f.read()


def build_param(
    input_len: int,
    output_len: int,
    user_input: str,
    system_prompt: str = None,
) -> Dict:
    """
    构建参数字典用于模型推断
    """
    hist = []
    if system_prompt is not None:
        hist.append(
            ModelMessage(role=ModelMessageRoleType.SYSTEM, content=system_prompt)
        )
    hist.append(ModelMessage(role=ModelMessageRoleType.HUMAN, content=user_input))
    hist = list(h.dict() for h in hist)
    context_len = input_len + output_len + 2
    params = {
        "prompt": user_input,
        "messages": hist,
        "model": model_name,
        "echo": False,
        "max_new_tokens": output_len,
        "context_len": context_len,
    }
    return params


async def run_batch(
    wh: WorkerManager,
    input_len: int,
    output_len: int,
    parallel_num: int,
    output_file: str,
):
    """
    运行批处理任务的异步函数
    """
    tasks = []
    prompt = read_prompt_from_file("11k")
    # 如果模型类型是 "vllm"
    if model_type == "vllm":
        # 设置最大输入字符串长度为输入长度
        max_input_str_len = input_len
        # 如果模型名中包含 "baichuan"
        if "baichuan" in model_name:
            # 将最大输入字符串长度扩展为原来的两倍
            max_input_str_len *= 2
        # 从提示文本中截取后面的 max_input_str_len 字符
        prompt = prompt[-max_input_str_len:]

    # 预热操作
    # 构建参数，包括输入长度、输出长度、提示文本和系统提示
    params = build_param(input_len, output_len, prompt, system_prompt="")
    # 异步生成文本
    await wh.generate(params)

    # 并行生成多个文本
    tasks = []
    for _ in range(parallel_num):
        # 构建参数，包括输入长度、输出长度、提示文本和系统提示
        params = build_param(input_len, output_len, prompt, system_prompt="")
        # 将生成文本任务添加到任务列表中
        tasks.append(wh.generate(params))

    # 输出运行基准信息，包括模型名称、输入长度、输出长度、并行数，并将结果保存到指定文件中
    print(
        f"Begin run benchmarks, model name: {model_name}, input_len: {input_len}, output_len: {output_len}, parallel_num: {parallel_num}, save result to {output_file}"
    )

    # 记录开始时间（毫秒）
    start_time_ms = time.time_ns() // 1_000_000
    # 并行等待所有任务完成，并获取结果
    results: List[ModelOutput] = await asyncio.gather(*tasks)
    # 记录结束时间（毫秒）
    end_time_ms = time.time_ns() // 1_000_000

    # 计算测试运行时间消耗（毫秒）
    test_time_cost_ms = end_time_ms - start_time_ms
    # 初始化测试总 tokens 数
    test_total_tokens = 0
    # 初始化第一个 token 的延迟时间（毫秒）
    first_token_latency_ms = 0
    # 初始化平均延迟时间（毫秒）
    latency_ms = 0
    # 初始化 GPU 数量
    gpu_nums = 0
    # 初始化平均 GPU 内存（GiB）
    avg_gpu_mem = 0
    # 初始化结果行列表
    rows = []

    # 遍历每个生成的结果
    for r in results:
        # 获取结果的指标数据
        metrics = r.metrics
        # 如果指标数据是字典类型，则转换为 ModelInferenceMetrics 对象
        if isinstance(metrics, dict):
            metrics = ModelInferenceMetrics(**metrics)
        # 打印结果
        print(r)
        # 累加测试总 tokens 数
        test_total_tokens += metrics.total_tokens
        # 累加第一个 token 的延迟时间（毫秒）减去开始时间（毫秒）
        first_token_latency_ms += metrics.first_token_time_ms - metrics.start_time_ms
        # 累加结束时间（毫秒）减去开始时间（毫秒）作为延迟时间（毫秒）
        latency_ms += metrics.end_time_ms - metrics.start_time_ms
        # 将指标数据转换为字典形式的行数据
        row_data = metrics.to_dict()
        # 删除行数据中的 collect_index 字段
        del row_data["collect_index"]
        # 如果行数据包含 avg_gpu_infos
        if "avg_gpu_infos" in row_data:
            # 获取 avg_gpu_infos 字段数据
            avg_gpu_infos = row_data["avg_gpu_infos"]
            # 计算 GPU 数量
            gpu_nums = len(avg_gpu_infos)
            # 计算平均 GPU 内存（GiB）
            avg_gpu_mem = (
                sum(i["allocated_memory_gb"] for i in avg_gpu_infos) / gpu_nums
            )
            # 删除行数据中的 avg_gpu_infos 和 current_gpu_infos 字段
            del row_data["avg_gpu_infos"]
            del row_data["current_gpu_infos"]
        # 将行数据添加到行列表中
        rows.append(row_data)

    # 计算平均测试速度（每秒 tokens 数）
    avg_test_speed_per_second = test_total_tokens / (test_time_cost_ms / 1000.0)
    # 计算平均第一个 token 的延迟时间（毫秒）
    avg_first_token_latency_ms = first_token_latency_ms / len(results)
    # 计算平均延迟时间（毫秒）
    avg_latency_ms = latency_ms / len(results)

    # 将结果写入 CSV 文件
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        # 创建 CSV 字典写入器
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADERS)
        # 如果文件位置指针在开头
        if f.tell() == 0:
            # 写入 CSV 头部信息
            writer.writeheader()
        # 遍历行数据列表
        for row in rows:
            # 添加模型名称、并行数、输入长度、输出长度、测试时间消耗等信息到行数据中
            row["model_name"] = model_name
            row["parallel_nums"] = parallel_num
            row["input_length"] = input_len
            row["output_length"] = output_len
            row["test_time_cost_ms"] = test_time_cost_ms
            row["test_total_tokens"] = test_total_tokens
            row["avg_test_speed_per_second(tokens/s)"] = avg_test_speed_per_second
            row["avg_first_token_latency_ms"] = avg_first_token_latency_ms
            row["avg_latency_ms"] = avg_latency_ms
            row["gpu_nums"] = gpu_nums
            row["gpu_mem(GiB)"] = avg_gpu_mem
            # 将行数据写入 CSV 文件
            writer.writerow(row)
    # 打印带有格式化字符串的消息，展示输入长度、输出长度、并行数和输出文件的信息
    print(
        f"input_len: {input_len}, output_len: {output_len}, parallel_num: {parallel_num}, save result to {output_file}"
    )
# 异步函数，运行模型的批处理任务
async def run_model(wh: WorkerManager) -> None:
    # 全局变量，存储结果的 CSV 文件名
    global result_csv_file
    # 如果结果 CSV 文件名为空，则获取默认的结果 CSV 文件名
    if not result_csv_file:
        result_csv_file = get_result_csv_file()
    # 如果结果 CSV 文件存在，则重命名为备份文件
    if os.path.exists(result_csv_file):
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d")
        os.rename(result_csv_file, f"{result_csv_file}.bak_{now_str}.csv")
    # 遍历并执行每个并行数和输入输出长度的组合
    for parallel_num in parallel_nums:
        for input_len, output_len in zip(input_lens, output_lens):
            try:
                # 异步执行批处理任务
                await run_batch(
                    wh, input_len, output_len, parallel_num, result_csv_file
                )
            except Exception:
                # 捕获异常并记录错误日志
                msg = traceback.format_exc()
                logging.error(
                    f"Run benchmarks error, input_len: {input_len}, output_len: {output_len}, parallel_num: {parallel_num}, error message: {msg}"
                )
                # 如果是 CUDA 内存溢出异常，则终止程序
                if "torch.cuda.OutOfMemoryError" in msg:
                    return
    # 正常退出程序
    sys.exit(0)


# 启动语言模型环境的函数
def startup_llm_env():
    # 导入并创建 FastAPI 应用
    from dbgpt.util.fastapi import create_app

    app = create_app()
    # 在客户端初始化工作管理器
    initialize_worker_manager_in_client(
        app=app,
        model_name=model_name,
        model_path=model_path,
        run_locally=False,
        controller_addr=controller_addr,
        local_port=6000,
        start_listener=run_model,  # 开始监听运行模型的函数
    )


# 连接到远程模型的函数
def connect_to_remote_model():
    # 启动语言模型环境
    startup_llm_env()


# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="huggingface")
    parser.add_argument("--result_csv_file", type=str, default=None)
    parser.add_argument("--input_lens", type=str, default="8,8,256,1024")
    parser.add_argument("--output_lens", type=str, default="256,512,1024,1024")
    parser.add_argument("--parallel_nums", type=str, default="1,2,4,16,32")
    parser.add_argument(
        "--remote_model", type=bool, default=False, help="Connect to remote model"
    )
    parser.add_argument("--controller_addr", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--limit_model_concurrency", type=int, default=200)

    args = parser.parse_args()
    print(f"args: {args}")
    # 设置模型名称
    model_name = args.model_name
    # 设置模型路径，默认从配置中获取
    model_path = args.model_path or LLM_MODEL_CONFIG[model_name]
    # 设置结果 CSV 文件名
    result_csv_file = args.result_csv_file
    # 解析输入长度列表
    input_lens = [int(i) for i in args.input_lens.strip().split(",")]
    # 解析输出长度列表
    output_lens = [int(i) for i in args.output_lens.strip().split(",")]
    # 解析并行数列表
    parallel_nums = [int(i) for i in args.parallel_nums.strip().split(",")]
    # 是否连接到远程模型的标志
    remote_model = args.remote_model
    # 控制器地址
    controller_addr = args.controller_addr
    # 限制模型并发数
    limit_model_concurrency = args.limit_model_concurrency
    # 模型类型，默认为 HuggingFace 模型
    model_type = args.model_type
    # 如果输入长度与输出长度列表长度不一致，抛出数值错误异常
    if len(input_lens) != len(output_lens):
        raise ValueError("input_lens size must equal output_lens size")
    # 如果 remote_model 为真，则连接到远程模型并运行基准测试
    if remote_model:
        connect_to_remote_model()
    # 否则，启动工作管理器并运行基准测试
    else:
        run_worker_manager(
            # 设置模型名称
            model_name=model_name,
            # 设置模型路径
            model_path=model_path,
            # 设置启动监听器为 run_model 函数
            start_listener=run_model,
            # 设置模型并发限制
            limit_model_concurrency=limit_model_concurrency,
            # 设置模型类型
            model_type=model_type,
        )
```