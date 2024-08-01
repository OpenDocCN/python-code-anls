# `.\DB-GPT-src\dbgpt\model\llm_out\vllm_llm.py`

```py
import os  # 导入操作系统相关模块
from typing import Dict  # 导入类型提示模块

from vllm import AsyncLLMEngine  # 导入异步语言模型引擎
from vllm.sampling_params import SamplingParams  # 导入采样参数
from vllm.utils import random_uuid  # 导入生成随机 UUID 的工具函数

_IS_BENCHMARK = os.getenv("DB_GPT_MODEL_BENCHMARK", "False").lower() == "true"  # 检查是否为基准测试模式

async def generate_stream(
    model: AsyncLLMEngine, tokenizer, params: Dict, device: str, context_len: int
):
    """
    Adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/vllm_worker.py
    """
    prompt = params["prompt"]  # 从参数中获取提示文本
    request_id = params.pop("request_id") if "request_id" in params else random_uuid()  # 获取请求 ID 或生成随机 UUID
    temperature = float(params.get("temperature", 1.0))  # 获取温度参数，默认为 1.0
    top_p = float(params.get("top_p", 1.0))  # 获取 top-p 参数，默认为 1.0
    max_new_tokens = int(params.get("max_new_tokens", 2048))  # 获取最大生成 token 数，默认为 2048
    echo = bool(params.get("echo", True))  # 获取回显参数，默认为 True
    stop_str = params.get("stop", None)  # 获取停止条件字符串

    stop_token_ids = params.get("stop_token_ids", None) or []  # 获取停止 token ID 列表或为空列表
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)  # 如果存在 EOS token ID，则添加到停止 token IDs 中

    # 处理 stop_str
    stop = set()
    if isinstance(stop_str, str) and stop_str != "":
        stop.add(stop_str)  # 如果 stop_str 是字符串且非空，加入到停止条件集合中
    elif isinstance(stop_str, list) and stop_str != []:
        stop.update(stop_str)  # 如果 stop_str 是列表且非空，加入到停止条件集合中

    for tid in stop_token_ids:
        if tid is not None:
            stop.add(tokenizer.decode(tid))  # 将每个停止 token ID 解码后加入到停止条件集合中

    # 设置生成参数
    top_p = max(top_p, 1e-5)  # 确保 top-p 不小于 1e-5
    if temperature <= 1e-5:
        top_p = 1.0  # 如果温度小于等于 1e-5，设定 top-p 为 1.0
    gen_params = {
        "stop": list(stop),  # 停止条件集合转换为列表
        "ignore_eos": False,  # 是否忽略 EOS 标记，默认为 False
    }

    prompt_token_ids = None
    if _IS_BENCHMARK:
        gen_params["stop"] = []  # 如果是基准测试模式，清空停止条件
        gen_params["ignore_eos"] = True  # 在基准测试模式下忽略 EOS 标记
        prompt_len = context_len - max_new_tokens - 2
        prompt_token_ids = tokenizer([prompt]).input_ids[0]  # 根据提示文本获取 token IDs
        prompt_token_ids = prompt_token_ids[-prompt_len:]  # 截取指定长度的 token IDs
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        use_beam_search=False,
        max_tokens=max_new_tokens,
        **gen_params  # 传递生成参数到采样参数对象中
    )

    results_generator = model.generate(prompt, sampling_params, request_id)  # 使用模型生成结果的生成器
    # 异步迭代器，逐个获取结果生成器中的输出
    async for request_output in results_generator:
        # 获取每个请求输出的提示文本
        prompt = request_output.prompt
        
        # 根据需求选择是否在输出文本前加上提示文本
        if echo:
            # 如果需要回显，则将每个输出的文本与提示文本连接起来
            text_outputs = [prompt + output.text for output in request_output.outputs]
        else:
            # 否则直接使用输出的文本
            text_outputs = [output.text for output in request_output.outputs]
        
        # 将所有输出文本连接成一个字符串，以空格分隔
        text_outputs = " ".join(text_outputs)

        # 计算提示文本的 token 数量
        prompt_tokens = len(request_output.prompt_token_ids)
        
        # 计算所有输出的 token 总数
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )
        
        # 计算总的 token 数量，包括提示文本和所有输出
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        # 确定完成原因，如果只有一个输出，使用其完成原因；如果多个，收集所有完成原因
        finish_reason = (
            request_output.outputs[0].finish_reason
            if len(request_output.outputs) == 1
            else [output.finish_reason for output in request_output.outputs]
        )
        
        # 返回生成器的下一个结果，包括文本输出、错误代码、使用情况和完成原因
        yield {
            "text": text_outputs,
            "error_code": 0,
            "usage": usage,
            "finish_reason": finish_reason,
        }
```