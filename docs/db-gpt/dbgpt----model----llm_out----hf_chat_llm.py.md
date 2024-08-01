# `.\DB-GPT-src\dbgpt\model\llm_out\hf_chat_llm.py`

```py
import logging  # 导入日志模块
from threading import Thread  # 导入线程模块

import torch  # 导入PyTorch库
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # 导入transformers库中的相关模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@torch.inference_mode()  # 使用PyTorch的推断模式装饰器
def huggingface_chat_generate_stream(
    model: AutoModelForCausalLM,  # 输入参数：预训练语言模型
    tokenizer: AutoTokenizer,  # 输入参数：tokenizer对象
    params,  # 输入参数：其他相关参数
    device,  # 输入参数：指定设备
    context_len=4096,  # 输入参数：上下文长度，默认为4096
):
    prompt = params["prompt"]  # 从参数中获取提示语
    temperature = float(params.get("temperature", 0.7))  # 获取温度参数，默认为0.7
    top_p = float(params.get("top_p", 1.0))  # 获取top-p采样参数，默认为1.0
    echo = params.get("echo", False)  # 获取回显参数，默认为False
    max_new_tokens = int(params.get("max_new_tokens", 2048))  # 获取生成的最大新token数，默认为2048
    stop_token_ids = params.get("stop_token_ids", [])  # 获取停止token的ID列表，默认为空列表
    do_sample = params.get("do_sample", True)  # 获取是否进行采样的参数，默认为True
    custom_stop_words = params.get("custom_stop_words", [])  # 获取自定义停止词列表，默认为空列表

    input_ids = tokenizer(prompt).input_ids  # 使用tokenizer处理提示语，获取输入的token IDs
    # input_ids = input_ids.to(device)  # （被注释掉的代码）将输入的token IDs移动到指定的设备

    if model.config.is_encoder_decoder:
        max_src_len = context_len  # 如果模型是编码-解码结构，最大源长度为context_len
    else:  # 否则，进行截断操作
        max_src_len = context_len - max_new_tokens - 1
    input_ids = input_ids[-max_src_len:]  # 从后截取指定长度的input_ids
    input_echo_len = len(input_ids)  # 计算截取后的input_ids的长度
    input_ids = torch.as_tensor([input_ids], device=device)  # 将截取后的input_ids转换为Tensor，并移动到指定设备上

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=not echo, skip_special_tokens=True
    )  # 创建TextIteratorStreamer对象，用于生成文本流

    base_kwargs = {
        "max_length": context_len,  # 设置生成文本的最大长度
        "temperature": temperature,  # 设置温度参数
        "streamer": streamer,  # 设置文本流生成器
        "top_p": top_p,  # 设置top-p参数
    }

    if stop_token_ids:
        base_kwargs["eos_token_id"] = stop_token_ids  # 如果有停止token的ID列表，则将其加入生成参数中
    if do_sample is not None:
        base_kwargs["do_sample"] = do_sample  # 如果有指定是否进行采样的参数，则将其加入生成参数中

    logger.info(
        f"Predict with parameters: {base_kwargs}\ncustom_stop_words: {custom_stop_words}"
    )  # 记录日志，打印生成的参数和自定义停止词列表

    generate_kwargs = {"input_ids": input_ids, **base_kwargs}  # 组合生成参数
    thread = Thread(target=model.generate, kwargs=generate_kwargs)  # 创建线程，使用模型进行文本生成
    thread.start()  # 启动线程

    out = ""  # 初始化输出变量
    for new_text in streamer:  # 遍历文本流
        out += new_text  # 将生成的新文本追加到输出变量中
        if custom_stop_words:  # 如果存在自定义停止词列表
            for stop_word in custom_stop_words:  # 遍历停止词列表
                if out.endswith(stop_word):  # 如果输出的文本以停止词结尾
                    out = out[: -len(stop_word)]  # 则从输出文本中移除该停止词
        yield out  # 生成当前的输出文本
```