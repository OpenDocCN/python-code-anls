# `.\VisualGLM-6B\cli_demo.py`

```py
# -*- encoding: utf-8 -*- 
# 设置文件编码为 utf-8

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
    # 添加参数：总序列的最大长度
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    # 添加参数：nucleus 抽样的 top p
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    # 添加参数：top k 抽样的 top k
    parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
    # 添加参数：抽样的温度
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    # 添加参数：只输出英文
    parser.add_argument("--english", action='store_true', help='only output English')
    # 添加参数：量化位数
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    # 添加参数：预训练模型
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    # 添加参数：中文提示
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", help='Chinese prompt for the first round')
    # 添加参数：英文提示
    parser.add_argument("--prompt_en", type=str, default="Describe the image.", help='English prompt for the first round')
    # 解析参数
    args = parser.parse_args()

    # 加载模型
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
    ))
    # 设置模型为评估模式
    model = model.eval()

    # 如果指定了量化位数
    if args.quant:
        # 对模型进行量化
        quantize(model, args.quant)
        # 如果有可用的 CUDA 设备，则将模型移动到 CUDA
        if torch.cuda.is_available():
            model = model.cuda()

    # 为模型添加自回归 mixin
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    # 从预训练模型 "THUDM/chatglm-6b" 中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # 如果不是英文模式
    if not args.english:
        # 打印欢迎信息，提示用户如何操作
        print('欢迎使用 VisualGLM-6B 模型，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
    else:
        # 如果是英文模式，打印欢迎信息，提示用户如何操作
        print('Welcome to VisualGLM-6B model. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 进入无限循环
        while True:
            # 初始化历史记录和缓存图像
            history = None
            cache_image = None
            # 根据语言选择提示信息，要求用户输入图像路径或 URL
            if not args.english:
                image_path = input("请输入图像路径或URL（回车进入纯文本对话）： ")
            else:
                image_path = input("Please enter the image path or URL (press Enter for plain text conversation): ")

            # 如果用户输入'stop'，则跳出循环
            if image_path == 'stop':
                break
            # 如果输入的图像路径不为空
            if len(image_path) > 0:
                # 根据语言选择提示信息，要求用户输入对话内容
                query = args.prompt_en if args.english else args.prompt_zh
            else:
                # 如果输入的图像路径为空
                if not args.english:
                    query = input("用户：")
                else:
                    query = input("User: ")
            # 进入内部循环
            while True:
                # 如果用户输入'clear'，则跳出内部循环
                if query == "clear":
                    break
                # 如果用户输入'stop'，则退出程序
                if query == "stop":
                    sys.exit(0)
                try:
                    # 调用 chat 函数进行对话生成
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
                except Exception as e:
                    # 捕获异常并打印错误信息
                    print(e)
                    break
                # 根据语言选择分隔符，打印生成的回复
                sep = 'A:' if args.english else '答：'
                print("VisualGLM-6B："+response.split(sep)[-1].strip())
                # 重置图像路径为 None
                image_path = None
                # 根据语言选择提示信息，要求用户输入下一轮对话内容
                if not args.english:
                    query = input("用户：")
                else:
                    query = input("User: ")
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```