# `.\chatglm4-finetune\composite_demo\src\clients\vllm.py`

```
"""
vLLM client.  # vLLM 客户端的说明
Please install [vLLM](https://github.com/vllm-project/vllm) according to its
installation guide before running this client.  # 提示用户在运行客户端前安装 vLLM
"""

import time  # 导入时间模块，用于时间相关操作
from collections.abc import Generator  # 从 collections 模块导入 Generator 类型

from transformers import AutoTokenizer  # 从 transformers 模块导入自动分词器
from vllm import SamplingParams, LLMEngine, EngineArgs  # 从 vllm 模块导入相关类

from client import Client, process_input, process_response  # 从 client 模块导入 Client 类和处理函数
from conversation import Conversation  # 从 conversation 模块导入 Conversation 类


class VLLMClient(Client):  # 定义 VLLMClient 类，继承自 Client 类
    def __init__(self, model_path: str):  # 初始化方法，接收模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(  # 创建分词器实例，从预训练模型加载
            model_path, trust_remote_code=True  # 指定模型路径，信任远程代码
        )
        self.engine_args = EngineArgs(  # 创建引擎参数对象
            model=model_path,  # 设置模型路径
            tensor_parallel_size=1,  # 设置张量并行大小为 1
            dtype="bfloat16",  # 指定数据类型为 bfloat16，适用于高性能计算
            trust_remote_code=True,  # 信任远程代码
            gpu_memory_utilization=0.6,  # 设置 GPU 内存利用率为 60%
            enforce_eager=True,  # 强制使用即时执行
            worker_use_ray=False,  # 设置不使用 Ray 进行工作管理
        )
        self.engine = LLMEngine.from_engine_args(self.engine_args)  # 从引擎参数创建 LLM 引擎实例

    def generate_stream(  # 定义生成流的方法
        self, tools: list[dict], history: list[Conversation], **parameters  # 接收工具列表、对话历史和其他参数
    ) -> Generator[tuple[str | dict, list[dict]]]:  # 返回生成器，产生元组类型的输出
        chat_history = process_input(history, tools)  # 处理输入，将历史记录与工具结合
        model_inputs = self.tokenizer.apply_chat_template(  # 应用聊天模板生成模型输入
            chat_history, add_generation_prompt=True, tokenize=False  # 设置生成提示并禁用分词
        )
        parameters["max_tokens"] = parameters.pop("max_new_tokens")  # 将 max_new_tokens 转换为 max_tokens
        params_dict = {  # 创建参数字典
            "n": 1,  # 设置生成样本数量为 1
            "best_of": 1,  # 设置最佳选择数量为 1
            "top_p": 1,  # 设置 nucleus 采样的阈值为 1
            "top_k": -1,  # 设置 top-k 采样为禁用状态
            "use_beam_search": False,  # 禁用束搜索
            "length_penalty": 1,  # 设置长度惩罚为 1
            "early_stopping": False,  # 禁用提前停止
            "stop_token_ids": [151329, 151336, 151338],  # 设置停止标记的 ID 列表
            "ignore_eos": False,  # 不忽略结束标记
            "logprobs": None,  # 不记录概率日志
            "prompt_logprobs": None,  # 不记录提示的概率日志
        }
        params_dict.update(parameters)  # 更新参数字典，加入其他传入参数
        sampling_params = SamplingParams(**params_dict)  # 创建采样参数实例

        self.engine.add_request(  # 向引擎添加请求
            request_id=str(time.time()), inputs=model_inputs, params=sampling_params  # 设置请求 ID 和参数
        )
        while self.engine.has_unfinished_requests():  # 当引擎有未完成的请求时
            request_outputs = self.engine.step()  # 执行一步，获取请求输出
            for request_output in request_outputs:  # 遍历每个请求输出
                yield process_response(request_output.outputs[0].text, chat_history)  # 处理输出并生成响应
```