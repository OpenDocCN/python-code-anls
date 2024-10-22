# `.\chatglm4-finetune\intel_device_demo\itrex\itrex_cli_demo.py`

```py
"""
该脚本创建一个命令行接口（CLI）演示，使用 transformers 后端，适用于 glm-4-9b 模型，结合 Intel® Extension for Transformers
"""

# 导入操作系统相关模块
import os
# 获取环境变量 'MODEL_PATH' 的值，如果不存在则使用默认值 'THUDM/glm-4-9b-chat'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

# 导入 PyTorch 库
import torch
# 从 threading 模块导入 Thread 类
from threading import Thread
# 从 intel_extension_for_transformers 导入 AutoModelForCausalLM 类
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
# 从 transformers 模块导入必要的类
from transformers import TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria, AutoTokenizer


# 定义停止条件类，继承自 StoppingCriteria
class StopOnTokens(StoppingCriteria):
    # 重写 __call__ 方法，检查是否需要停止生成
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 定义停止的 token ID 列表
        stop_ids = [151329, 151336, 151338]
        # 遍历停止 ID 列表
        for stop_id in stop_ids:
            # 如果当前输入的最后一个 token ID 是停止 ID，则返回 True
            if input_ids[0][-1] == stop_id:
                return True
        # 如果没有匹配的停止 ID，则返回 False
        return False


# 初始化模型和分词器的函数
def initialize_model_and_tokenizer():
    # 从预训练模型路径加载分词器，信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 从预训练模型路径加载 causal language model，指定设备为 CPU，信任远程代码，并以 4bit 模式加载
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",  # 使用 Intel CPU 进行推理
        trust_remote_code=True,
        load_in_4bit=True
    )
    # 返回加载的分词器和模型
    return tokenizer, model


# 获取用户输入的函数
def get_user_input():
    # 提示用户输入并返回输入内容
    return input("\nUser: ")


# 主函数
def main():
    # 初始化模型和分词器
    tokenizer, model = initialize_model_and_tokenizer()

    # 初始化历史记录列表
    history = []
    # 设置最大生成长度
    max_length = 100
    # 设置 top-p 取样参数
    top_p = 0.9
    # 设置温度参数
    temperature = 0.8
    # 实例化停止条件对象
    stop = StopOnTokens()

    # 打印欢迎信息
    print("Welcome to the CLI chat. Type your messages below.")
    # 无限循环，直到用户选择退出
    while True:
        # 获取用户输入
        user_input = get_user_input()
        # 检查用户输入是否为退出指令
        if user_input.lower() in ["exit", "quit"]:
            break
        # 将用户输入添加到历史记录中，模型响应初始化为空
        history.append([user_input, ""])

        # 初始化消息列表，用于存储用户和模型的对话内容
        messages = []
        # 遍历历史记录，获取用户和模型的消息
        for idx, (user_msg, model_msg) in enumerate(history):
            # 如果是最新的用户消息且没有模型消息，添加用户消息到消息列表
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            # 如果用户消息存在，添加到消息列表
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            # 如果模型消息存在，添加到消息列表
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        # 应用聊天模板处理消息，并返回模型输入的张量
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # 添加生成提示
            tokenize=True,                # 对内容进行分词
            return_tensors="pt"          # 返回 PyTorch 张量
        )

        # 创建一个文本迭代流处理器，用于流式生成输出
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,          # 使用的分词器
            timeout=60,                   # 超时设置为60秒
            skip_prompt=True,             # 跳过提示
            skip_special_tokens=True      # 跳过特殊标记
        )

        # 设置生成模型的参数
        generate_kwargs = {
            "input_ids": model_inputs,    # 输入的模型张量
            "streamer": streamer,          # 使用的流处理器
            "max_new_tokens": max_length,  # 生成的最大新标记数量
            "do_sample": True,             # 启用采样
            "top_p": top_p,                # 样本筛选阈值
            "temperature": temperature,     # 温度参数控制生成随机性
            "stopping_criteria": StoppingCriteriaList([stop]),  # 停止生成的条件
            "repetition_penalty": 1.2,     # 重复惩罚系数
            "eos_token_id": model.config.eos_token_id,  # 结束标记的 ID
        }

        # 创建一个线程来生成模型的输出
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        # 启动线程
        t.start()
        # 打印助手的提示，保持在同一行
        print("Assistant:", end="", flush=True)
        # 从流中获取新生成的标记并打印
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)  # 打印新标记
                history[-1][1] += new_token  # 将新标记添加到最新的历史模型消息

        # 去掉最新模型消息的前后空白
        history[-1][1] = history[-1][1].strip()
# 当脚本作为主程序运行时
if __name__ == "__main__":
    # 调用 main 函数
    main()
```