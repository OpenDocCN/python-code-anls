# `.\DB-GPT-src\dbgpt\model\llm_out\falcon_llm.py`

```py
# 从 threading 模块中导入 Thread 类，用于多线程处理
from threading import Thread

# 导入 torch 库，用于深度学习模型和张量操作
import torch

# 从 transformers 库中导入 StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer 类
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# 定义函数 falcon_generate_output，生成文本输出
def falcon_generate_output(model, tokenizer, params, device, context_len=2048):
    """Fork from: https://github.com/KohakuBlueleaf/guanaco-lora/blob/main/generate.py"""
    
    # 设定 tokenizer 的 bos_token_id 为 1
    tokenizer.bos_token_id = 1
    
    # 打印参数 params
    print(params)
    
    # 获取参数中的 "stop" 键对应的值，默认为 "###"
    stop = params.get("stop", "###")
    
    # 获取参数中的 "prompt" 键对应的值作为提示信息
    prompt = params["prompt"]
    
    # 将提示信息赋值给 query 变量
    query = prompt
    
    # 打印查询消息
    print("Query Message: ", query)
    
    # 使用 tokenizer 对查询文本进行编码，返回的是一个 PyTorch 张量
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    
    # 将 input_ids 移动到指定的模型设备上
    input_ids = input_ids.to(model.device)
    
    # 创建 TextIteratorStreamer 对象，用于迭代生成文本
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    
    # 再次将 tokenizer 的 bos_token_id 设置为 1
    tokenizer.bos_token_id = 1
    
    # 设置停止生成文本的 token id 列表，这里仅包含一个值为 0 的 token id
    stop_token_ids = [0]
    
    # 定义一个自定义的停止条件类 StopOnTokens
    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            # 遍历停止生成的 token id 列表
            for stop_id in stop_token_ids:
                # 如果当前生成的 token 是停止 token，则返回 True
                if input_ids[0][-1] == stop_id:
                    return True
            # 否则返回 False，继续生成
            return False
    
    # 实例化 StopOnTokens 类
    stop = StopOnTokens()
    
    # 定义生成文本的参数字典 generate_kwargs
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
    
    # 创建一个线程 t，调用 model.generate 方法生成文本，传入 generate_kwargs 参数
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # 启动线程
    
    # 初始化输出字符串
    out = ""
    
    # 使用 streamer 迭代生成的新文本
    for new_text in streamer:
        out += new_text  # 将新生成的文本添加到输出字符串中
        yield out  # 使用生成器返回当前的输出字符串
```