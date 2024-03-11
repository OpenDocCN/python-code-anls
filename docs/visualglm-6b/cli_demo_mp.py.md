# `.\VisualGLM-6B\cli_demo_mp.py`

```py
# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

from model import VisualGLMModel, chat
from finetune_visualglm import FineTuneVisualGLMModel
from sat.model import AutoModel

# 主函数
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", help='Chinese prompt for the first round')
    parser.add_argument("--prompt_en", type=str, default="Describe the image.", help='English prompt for the first round')
    # 解析命令行参数
    args = parser.parse_args()

    # load model
    # 从预训练模型加载模型和模型参数
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
    ), overwrite_args={'model_parallel_size': 2})
    # 设置模型为评估模式
    model = model.eval()

    # 如果指定了量化参数
    if args.quant:
        # 对模型进行量化
        quantize(model.transformer, args.quant)
        # 如果有可用的 GPU，则将模型移动到 GPU 上
        if torch.cuda.is_available():
            model = model.cuda()

    # 为模型添加自回归 mixin
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # 图像路径
    image_path = 'fewshot-data/meme.png'
    # 根据语言选择查询内容
    query = args.prompt_en if args.english else args.prompt_zh
    # 历史对话记录初始化为空
    history = None
    # 缓存图像初始化为空
    cache_image = None
    # 聊天函数调用，传入图像路径、模型、分词器、查询内容等参数
    response, history, cache_image = chat(
        image_path, 
        model, 
        tokenizer,
        query, 
        history=history, 
        image=cache_image, 
        max_length=args.max_length, 
        top_p=args.top_p, 
        temperature=args.temperature,
        top_k=args.top_k,
        english=args.english,
        invalid_slices=[slice(63823, 130000)] if args.english else []
        )
    # 根据语言选择分隔符
    sep = 'A:' if args.english else '答：'
    # 打印回复内容
    print(response.split(sep)[-1].strip())
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```