# `.\LLM4Decompile\evaluation\server\text_generation.py`

```
import asyncio  # 引入 asyncio 异步编程库
import numpy as np  # 引入 numpy 库，用于数值计算
import os  # 引入 os 库，用于与操作系统交互
import random  # 引入 random 库，用于生成随机数
import socket  # 引入 socket 库，用于网络通信
import subprocess  # 引入 subprocess 库，用于创建和管理子进程
import time  # 引入 time 库，用于时间相关操作

from loguru import logger  # 从 loguru 库中引入 logger 对象，用于日志记录
from text_generation import AsyncClient  # 从 text_generation 模块中引入 AsyncClient 类
from tqdm import tqdm  # 从 tqdm 库中引入 tqdm 进度条
from typing import List  # 从 typing 模块中引入 List 类型


class TextGenerationServer:
    def __init__(
        self,
        model_id: str,
        port: int,
        dtype: str,
        max_input_len: int,
        max_total_tokens: int,
        max_batch_prefill_tokens: int,
        num_shards: int,
    ):
        # 生成一个随机的 master 端口号
        master_port = random.randint(10_000, 20_000)
        
        # 组装启动参数列表
        args = [
            "text-generation-launcher",
            "--model-id",
            model_id,
            "--port",
            str(port),
            "--master-port",
            str(master_port),
        ]
        # 添加其他参数到参数列表
        args.extend(["--num-shard", str(num_shards)])
        args.extend(["--dtype", dtype])
        args.extend(["--max-input-length", str(max_input_len)])
        args.extend(["--max-total-tokens", str(max_total_tokens)])
        args.extend(["--max-batch-prefill-tokens", str(max_batch_prefill_tokens)])

        # 打印日志记录启动命令
        logger.info(" ".join(args))
        
        # 启动子进程，并将标准输出重定向到 DEVNULL（即丢弃输出）
        self.launcher = subprocess.Popen(args, stdout=subprocess.DEVNULL)
        
        # 打印等待信息，等待文本生成服务器启动
        logger.info("Waiting for text generation server to start...")

        # 检查 web 服务器是否准备就绪
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8080), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

        # 循环检查 web 服务器是否准备就绪，每次间隔 10 秒
        while not webserver_ready():
            time.sleep(10)
        
        # 打印信息，表示文本生成服务器已经准备好
        logger.info("Text generation webserver ready")

    def __del__(self):
        # 在对象销毁时，终止启动的子进程
        self.launcher.terminate()
        self.launcher.wait()


class TextGenerationClient:
    def __init__(self, port, stop_sequences: List[str]):
        # 创建 AsyncClient 客户端对象，连接到指定端口的文本生成服务器
        self.client = AsyncClient(f"http://127.0.0.1:{port}", timeout=9999)
        
        # 存储停止生成文本的序列列表
        self.stop_sequences = stop_sequences

    async def generate(
        self,
        input: str,
        max_new_tokens: int,
        do_sample: bool,
        pbar: tqdm,
        **kwargs,
    ) -> str:
        try:
            # 如果指定进行采样，则获取采样参数，否则使用默认参数
            if do_sample:
                top_p = kwargs.get("top_p", 0.95)
                temperature = kwargs.get("temperature", 0.8)
                # 调用客户端生成函数，传入参数并生成文本
                output = await self.client.generate(
                    input,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=self.stop_sequences,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                # 调用客户端生成函数，传入参数并生成文本（无采样参数）
                output = await self.client.generate(
                    input,
                    max_new_tokens=max_new_tokens,
                    stop_sequences=self.stop_sequences,
                    do_sample=do_sample,
                )
            # 从生成的文本中移除停止序列
            generated_text = output.generated_text
            for stop_sequence in self.stop_sequences:
                generated_text = generated_text.replace(stop_sequence, "")

        except Exception as e:
            # 发生异常时，记录错误并返回空文本
            generated_text = ""
            logger.error(e)
        # 更新进度条
        pbar.update()
        # 返回生成的文本
        return generated_text

    async def generate_code_results(
        self,
        inputs: List[str],
        max_new_tokens: int,
        num_outputs: int,
        task_size: int = 50,
        **kwargs,
    ) -> np.array:
        # 使用进度条显示代码生成结果的获取进度
        with tqdm(
            total=len(inputs * num_outputs), desc="Fetching code generation results"
        ) as pbar:
            results = []
            # 如果 max_new_tokens 小于等于 0，则设为 32
            max_new_tokens = max_new_tokens if max_new_tokens > 0 else 32
            # 判断是否进行多次输出采样
            do_sample = num_outputs > 1
            # 创建请求列表，重复输入以获取多个输出
            requests = [input for input in inputs for _ in range(num_outputs)]
            # 根据任务大小分批处理请求
            for i in range(0, len(requests), task_size):
                tasks = []
                # 对每个请求创建异步任务
                for input in requests[i : i + task_size]:
                    task = asyncio.ensure_future(
                        self.generate(input, max_new_tokens, do_sample, pbar, **kwargs)
                    )
                    tasks.append(task)
                # 并发执行任务，并收集结果
                for result in await asyncio.gather(*tasks):
                    results.append(result)
            # 将结果转换成 numpy 数组，形状为 (len(inputs), num_outputs)
            results = np.array(results).reshape(len(inputs), num_outputs)
        # 返回生成的结果数组
        return results
```