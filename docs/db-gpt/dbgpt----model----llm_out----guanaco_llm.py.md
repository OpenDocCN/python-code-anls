# `.\DB-GPT-src\dbgpt\model\llm_out\guanaco_llm.py`

```py
from threading import Thread

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


def guanaco_generate_output(model, tokenizer, params, device, context_len=2048):
    """Fork from: https://github.com/KohakuBlueleaf/guanaco-lora/blob/main/generate.py"""

    # 打印参数 params
    print(params)
    # 获取停止标记，默认为 "###"
    stop = params.get("stop", "###")
    # 获取提示信息
    prompt = params["prompt"]
    # 将查询消息设为提示信息
    query = prompt
    # 打印查询消息
    print("Query Message: ", query)

    # 使用分词器对查询消息进行编码，返回 PyTorch 张量的输入 ID
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    # 将输入 ID 移动到模型所在的设备上
    input_ids = input_ids.to(model.device)

    # 创建文本迭代流对象
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    # 定义停止标记的 token ID 列表
    stop_token_ids = [0]

    # 定义停止标记的策略类 StopOnTokens
    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            # 遍历停止标记的 token ID 列表
            for stop_id in stop_token_ids:
                # 如果输入 ID 的最后一个 token 是停止标记之一，则返回 True
                if input_ids[0][-1] == stop_id:
                    return True
            # 否则返回 False
            return False

    # 实例化停止标记策略对象
    stop = StopOnTokens()

    # 定义生成新文本的参数字典 generate_kwargs
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=True,
        top_k=1,
        streamer=streamer,
        repetition_penalty=1.7,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    # 创建并启动一个线程，使用模型生成新文本
    t1 = Thread(target=model.generate, kwargs=generate_kwargs)
    t1.start()

    # 生成器对象 generator，用于生成文本
    generator = model.generate(**generate_kwargs)
    # 遍历生成器，逐行处理生成的文本
    for output in generator:
        # 解码生成的文本
        decoded_output = tokenizer.decode(output)
        # 如果生成的文本的最后一个 token 是 EOS 标记，则终止生成
        if output[-1] in [tokenizer.eos_token_id]:
            break
        # 根据特定分隔符切分生成的文本，获取响应部分，并去除首尾空格
        out = decoded_output.split("### Response:")[-1].strip()

        # 返回生成的响应部分
        yield out


def guanaco_generate_stream(model, tokenizer, params, device, context_len=2048):
    """Fork from: https://github.com/KohakuBlueleaf/guanaco-lora/blob/main/generate.py"""

    # 设置 BOS（Beginning of Sequence）标记的 token ID 为 1
    tokenizer.bos_token_id = 1
    # 打印参数 params
    print(params)
    # 获取停止标记，默认为 "###"
    stop = params.get("stop", "###")
    # 获取提示信息
    prompt = params["prompt"]
    # 获取最大新 token 数，默认为 512
    max_new_tokens = params.get("max_new_tokens", 512)
    # 获取温度参数，默认为 1.0
    temerature = params.get("temperature", 1.0)

    # 将查询消息设为提示信息
    query = prompt
    # 打印查询消息
    print("Query Message: ", query)

    # 使用分词器对查询消息进行编码，返回 PyTorch 张量的输入 ID
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    # 将输入 ID 移动到模型所在的设备上
    input_ids = input_ids.to(model.device)

    # 创建文本迭代流对象
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    # 再次将 BOS（Beginning of Sequence）标记的 token ID 重新设置为 1
    tokenizer.bos_token_id = 1
    # 定义停止标记的 token ID 列表
    stop_token_ids = [0]

    # 定义停止标记的策略类 StopOnTokens
    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            # 遍历停止标记的 token ID 列表
            for stop_id in stop_token_ids:
                # 如果输入 ID 的最后一个 token 是停止标记之一，则返回 True
                if input_ids[-1][-1] == stop_id:
                    return True
            # 否则返回 False
            return False

    # 实例化停止标记策略对象
    stop = StopOnTokens()
    # 创建生成文本的参数字典，包括输入的 token IDs、生成的最大新 token 数量、温度参数、允许采样、top_k 设置、数据流处理器、重复惩罚项、停止条件列表
    generate_kwargs = dict(
        input_ids=input_ids,                    # 输入的 token IDs
        max_new_tokens=max_new_tokens,           # 生成的最大新 token 数量
        temperature=temerature,                 # 温度参数
        do_sample=True,                         # 允许采样
        top_k=1,                                # top_k 设置
        streamer=streamer,                      # 数据流处理器
        repetition_penalty=1.7,                 # 重复惩罚项
        stopping_criteria=StoppingCriteriaList([stop]),  # 停止条件列表
    )
    
    # 使用模型生成文本，传入之前创建的参数字典
    model.generate(**generate_kwargs)
    
    # 初始化输出字符串
    out = ""
    # 遍历数据流处理器中的新生成文本
    for new_text in streamer:
        out += new_text  # 将新生成的文本追加到输出字符串
        yield out        # 通过生成器逐步产出累积的输出文本
```