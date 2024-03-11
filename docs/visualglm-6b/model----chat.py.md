# `.\VisualGLM-6B\model\chat.py`

```
# -*- encoding: utf-8 -*-
'''
@File    :   chat.py
@Time    :   2023/05/08 19:10:08
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# 导入所需的库
import os
import sys
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, BaseStrategy

from .blip2 import BlipImageEvalProcessor

# 定义函数，用于获取 GLM 模型中的 masks 和 position ids
def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    '''GLM model, different from GPT.
    Args:
        seq: torch.IntTensor, [seq_len]
        mask_position: int, the position of the masked place.
        context_length: int, the length of context.
    Returns:
        tokens: torch.IntTensor, [1, seq_len]
        attention_mask: torch.FloatTensor, [1, seq_len, seq_len]
        position_ids: torch.IntTensor, [2, seq_len]
    '''
    # 将输入序列转换为形状为 [1, seq_len] 的张量
    tokens = seq.unsqueeze(0)

    # 创建注意力遮罩张量，形状为 [1, seq_len, seq_len]
    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    # 创建 2D 位置 id 张量
    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

# 定义函数，用于处理响应文本
def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    # 定义标点符号替换规则
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    # 遍历 punkts 列表中的每个元素
    for item in punkts:
        # 使用正则表达式替换 response 中匹配到的中文字符前面的 item[0] 为 item[1]
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        # 使用正则表达式替换 response 中匹配到的中文字符后面的 item[0] 为 item[1]
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    # 返回替换后的 response
    return response
def process_image(text, image=None):
    '''Process image in text.
    Args:
        text: str, text.
        image: Optional, image path / url / PIL image.
    '''
    # 寻找文本中最后一个"<img>"的位置
    image_position = text.rfind("<img>") + 5
    # 如果没有找到"<img>"，返回原文本、位置和空图片
    if image_position < 5:
        return text, image_position, None
    # 使用正则表达式提取<img></img>中的路径
    image_path = re.findall(r"<img>(.*?)</img>", text)
    image_path = image_path[-1] if image_path[-1] else None
    # 如果存在图片路径
    if image_path is not None:
        # 断言图片和图片路径不能同时存在
        assert image is None, "image and image_path cannot be both not None."
        # 替换文本中的图片路径为空
        text = text.replace(image_path, "")
        image_path = image_path.strip()
        # 如果是URL路径
        if image_path.startswith("http"):
            # 从URL获取图片内容
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        # 如果是本地路径
        else:
            # 打开本地图片
            image = Image.open(image_path)
    # 如果存在图片且为PIL图片对象
    if image is not None and isinstance(image, Image.Image):
        # 图像处理
        processor = BlipImageEvalProcessor(224)
        image = processor(image.convert('RGB'))
        image = image.unsqueeze(0)
    return text, image_position, image


def chat(image_path, model, tokenizer, 
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 1024, top_p=0.7, top_k=30, temperature=0.95, repetition_penalty=1.2,
        invalid_slices=[], english=False
        ):
    if not history:
        history = []
    if image_path or image is not None:
        prompt = "<img>{}</img>".format(image_path if image_path else "")
    else:
        prompt = ""
    if english:
        for i, (old_query, response) in enumerate(history):
            prompt += "Q:{}\nA:{}\n".format(old_query, response)
        prompt += "Q:{}\nA:".format(query)
    else:
        for i, (old_query, response) in enumerate(history):
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)
    # ---------------
    # tokenizer, this is an example of huggingface tokenizer.
    # input str, output['input_ids'] = tensor([[tokenized str, gmask, sop]])
    # 处理图像，返回处理后的文本、图像位置和 torch 图像
    prompt, image_position, torch_image = process_image(prompt, image=image)
    # 如果 torch 图像不为空，则将其转换为 float16 类型，并移到模型参数所在的设备上
    if torch_image is not None:
        torch_image = torch_image.to(torch.float16).to(next(model.parameters()).device)
    # 如果图像位置小于5，则表示没有图像
    if image_position < 5:
        # 使用 tokenizer 对文本进行编码，返回输入张量
        inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
        pre_image = 0
    else:
        # 对文本进行分段编码
        input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
        input1 = [tokenizer.pad_token_id] * model.image_length
        input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
        # 将分段编码合并为一个输入序列
        inputs = sum([input0, input1, input2], [])
        inputs = torch.tensor(tokenizer.build_inputs_with_special_tokens(inputs)).to(model.parameters().__next__().device)
        pre_image = len(input0)
    # ---------------
    # 接下来，我们手动设置格式以保持灵活性。
    # 计算 mask 位置和上下文长度
    mask_position = len(inputs) - 2
    context_length = len(inputs) - 1 # 所有 sop 之前的内容
    # 部分应用函数，用于获取 mask 和位置 id
    get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=context_length)
    # 将输入序列与填充的部分拼接为一个序列
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    # ---------------
    # 创建策略对象，用于生成文本
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    # 调用 filling_sequence 函数填充序列，获取生成的输出
    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=strategy,
        pre_image=pre_image,
        image=torch_image,
    )[0] # drop memory
    
    # ---------------
    # 从 inference_glm.py 移植过来，比聊天模式更通用
    # 剪切掉-1并将生成的内容填充回序列中
    if type(output) is not list:
        output_list = output.tolist()
    else:
        output_list = output
    for i in range(len(output_list)):
        output = output_list[i]
        if type(output) is not list:
            output = output.tolist()
        try:
            unfinished = output.index(-1)
        except ValueError:
            unfinished = len(output)
        if output[unfinished - 1] == tokenizer.eos_token_id:
            unfinished -= 1
        bog = output.index(tokenizer.bos_token_id)
        output_list[i] = output[:mask_position] + output[bog + 1:unfinished] + output[mask_position + 1:bog]
    # ---------------

    # 将生成的输出解码为文本
    response = tokenizer.decode(output_list[0])
    # 根据语言选择不同的分隔符
    sep = 'A:' if english else '答：'
    # 处理生成的回复文本
    response = process_response(response).split(sep)[-1].strip()
    # 将查询和回复添加到历史记录中
    history = history + [(query, response)]
    # 返回回复、历史记录和图像
    return response, history, torch_image
```