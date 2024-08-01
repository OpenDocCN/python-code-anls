# `.\DB-GPT-src\dbgpt\model\llm\inference.py`

```py
"""
Fork from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py

This code file will be deprecated in the future. 
We have integrated fastchat. For details, see: dbgpt/model/model_adapter.py
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
from typing import Dict, Iterable

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from dbgpt.model.utils.llm_utils import is_partial_stop, is_sentence_complete


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Prepare a list of logits processors based on provided parameters.

    Args:
        temperature (float): Temperature value for temperature warping.
        repetition_penalty (float): Repetition penalty value.
        top_p (float): Top-p value for nucleus sampling.
        top_k (int): Top-k value for top-k sampling.

    Returns:
        LogitsProcessorList: List of initialized logits processors.
    """
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    """
    Generate tokens sequentially using the provided model and tokenizer.

    Args:
        model: The pretrained model to generate text.
        tokenizer: Tokenizer corresponding to the model.
        params (Dict): Parameters dict containing generation settings.
        device (str): Device ('cpu' or 'cuda') on which to run inference.
        context_len (int): Length of context to consider for generation.
        stream_interval (int, optional): Interval for streaming inference. Defaults to 2.
        judge_sent_end (bool, optional): Whether to judge sentence completion. Defaults to False.
    """
    # Read parameters
    prompt = params["prompt"]
    print(f"Prompt of model: \n{prompt}")
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    # Finish stream event, which contains finish reason
    # 如果生成的新 token 数量达到最大限制减一，则完成原因为长度
    if i == max_new_tokens - 1:
        finish_reason = "length"
    # 如果模型停止生成新 token，则完成原因为停止
    elif stopped:
        finish_reason = "stop"
    # 否则完成原因为空
    else:
        finish_reason = None
    # 生成输出
    yield output
    # 清理内存
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
```