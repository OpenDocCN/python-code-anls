# `.\DB-GPT-src\dbgpt\util\benchmarks\llm\fastchat_benchmarks_inference.py`

```py
"""
Adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py.
For benchmarks.

"""
import gc  # 导入垃圾回收模块，用于释放不再使用的内存空间
from typing import Dict, Iterable  # 导入类型提示模块，用于声明函数参数和返回类型

import torch  # 导入PyTorch库，用于深度学习任务
from fastchat.utils import get_context_length, is_partial_stop, is_sentence_complete  # 导入自定义工具函数
from transformers.generation.logits_process import (  # 导入处理生成模型logits的相关模块
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()  # 创建一个空的logits处理器列表

    # 根据参数设置添加不同的logits处理器
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))  # 添加温度调整处理器
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))  # 添加重复惩罚处理器
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))  # 添加Top-p采样处理器
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))  # 添加Top-k采样处理器

    return processor_list  # 返回处理器列表


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 1,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device  # 如果模型有device属性，则使用模型的device

    # 读取参数
    prompt = params["prompt"]  # 获取生成任务的提示文本
    len_prompt = len(prompt)  # 获取提示文本的长度
    temperature = float(params.get("temperature", 1.0))  # 获取温度参数，默认为1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # 获取重复惩罚参数，默认为1.0
    top_p = float(params.get("top_p", 1.0))  # 获取Top-p采样参数，默认为1.0
    top_k = int(params.get("top_k", -1))  # 获取Top-k采样参数，默认为-1（禁用）
    max_new_tokens = int(params.get("max_new_tokens", 256))  # 获取最大生成token数，默认为256
    logprobs = params.get("logprobs", None)  # 获取是否输出logprobs的参数
    echo = bool(params.get("echo", True))  # 获取是否输出echo的参数，默认为True
    stop_str = params.get("stop", None)  # 获取停止生成的字符串
    stop_token_ids = params.get("stop_token_ids", None) or []  # 获取停止生成的token ID列表，默认为空列表
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)  # 如果结束标记不在停止token ID列表中，添加到列表中

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )  # 准备logits处理器

    input_ids = tokenizer(prompt).input_ids  # 使用tokenizer处理提示文本并获取输入token ID

    if model.config.is_encoder_decoder:
        max_src_len = context_len  # 如果模型是编码-解码结构，设置最大源文本长度为context_len
    else:  # 否则（单编码器模型），进行截断处理
        max_src_len = context_len - max_new_tokens - 1  # 计算截断后的最大源文本长度

    input_ids = input_ids[-max_src_len:]  # 截取输入token ID，保留最后max_src_len个token
    output_ids = list(input_ids)  # 初始化输出token ID列表为输入token ID的拷贝
    input_echo_len = len(input_ids)  # 计算输入token的长度

    # 初始化停止生成的token ID列表和停止字符串为None
    stop_token_ids = []
    stop_str = None
    # 如果模型配置为编码-解码模型
    if model.config.is_encoder_decoder:
        # 如果 logprobs 不为 None，则需要支持编码-解码模型的 logprobs，暂时未实现此功能
        if logprobs is not None:  # FIXME: Support logprobs for encoder-decoder models.
            raise NotImplementedError
        
        # 使用输入的 input_ids，通过模型的 encoder 获取 encoder 输出
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        
        # 设置生成过程的起始 token id
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    
    # 如果不是编码-解码模型
    else:
        # 直接使用 input_ids 作为起始 token id
        start_ids = torch.as_tensor([input_ids], device=device)

    # 初始化过去的 key values 和输出为 None
    past_key_values = out = None
    
    # 第一个 token 没有对应的 logprobs
    token_logprobs = [None]  # The first token has no logprobs.
    
    # 设置句子中断标志为 False
    sent_interrupt = False
    
    # 完成原因初始为 None
    finish_reason = None
    
    # 如果 stopped 为 True，则设置完成原因为 "stop"
    if stopped:
        finish_reason = "stop"
    
    # 生成器函数返回一个字典，包含输出文本、logprobs、使用信息和完成原因
    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # 清理过去的 key values 和输出
    del past_key_values, out
    
    # 执行垃圾回收
    gc.collect()
    
    # 清空 CUDA 缓存
    torch.cuda.empty_cache()
    
    # 如果设备为 "xpu"，清空 XPU 缓存
    if device == "xpu":
        torch.xpu.empty_cache()
    
    # 如果设备为 "npu"，清空 NPU 缓存
    if device == "npu":
        torch.npu.empty_cache()
```