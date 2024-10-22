# `.\chatglm4-finetune\intel_device_demo\openvino\openvino_cli_demo.py`

```py
# 导入 argparse 库用于处理命令行参数
import argparse
# 从 typing 导入 List 和 Tuple 类型提示
from typing import List, Tuple
# 从 threading 导入 Thread 用于多线程
from threading import Thread
# 导入 PyTorch 库
import torch
# 从 optimum.intel.openvino 导入模型类
from optimum.intel.openvino import OVModelForCausalLM
# 从 transformers 导入所需的类和函数
from transformers import (AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

# 定义停止条件类
class StopOnTokens(StoppingCriteria):
    # 初始化类，传入要停止的 token ID
    def __init__(self, token_ids):
        self.token_ids = token_ids

    # 重载 __call__ 方法，实现停止条件逻辑
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # 遍历每个停止 ID
        for stop_id in self.token_ids:
            # 检查当前输入的最后一个 token 是否为停止 ID
            if input_ids[0][-1] == stop_id:
                return True  # 如果是，则返回 True
        return False  # 否则返回 False


# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(add_help=False)
    # 添加帮助参数
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    # 添加模型路径参数
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. model path')
    # 添加最大序列长度参数
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=256,
                        required=False,
                        type=int,
                        help='Required. maximun length of output')
    # 添加设备参数
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='Required. device for inference')
    # 解析命令行参数
    args = parser.parse_args()
    # 获取模型路径
    model_dir = args.model_path

    # 配置 OpenVINO 参数
    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}

    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)

    # 打印模型编译信息
    print("====Compiling model====")
    # 从预训练模型加载 OpenVINO 模型
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    # 创建文本迭代流处理器
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    # 初始化停止 token 列表
    stop_tokens = [StopOnTokens([151329, 151336, 151338])]
    # 定义一个函数，将对话历史转换为模型输入格式
    def convert_history_to_token(history: List[Tuple[str, str]]):
        # 初始化一个空的消息列表，用于存储用户和助手的消息
        messages = []
        # 遍历历史记录中的每一条消息，索引为 idx
        for idx, (user_msg, model_msg) in enumerate(history):
            # 如果是最后一条记录且助手消息为空，添加用户消息并终止循环
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            # 如果用户消息不为空，添加到消息列表中
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            # 如果助手消息不为空，添加到消息列表中
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        # 将消息列表转换为模型输入格式，并添加生成提示
        model_inputs = tokenizer.apply_chat_template(messages,
                                                     add_generation_prompt=True,
                                                     tokenize=True,
                                                     return_tensors="pt")
        # 返回模型输入
        return model_inputs

    # 初始化历史记录为空列表
    history = []
    # 输出对话开始的提示信息
    print("====Starting conversation====")
    # 无限循环以持续对话
    while True:
        # 获取用户输入
        input_text = input("用户: ")
        # 如果用户输入为 'stop'，则终止循环
        if input_text.lower() == 'stop':
            break

        # 如果用户输入为 'clear'，则清空对话历史
        if input_text.lower() == 'clear':
            history = []
            print("AI助手: 对话历史已清空")
            continue

        # 输出助手的提示，准备生成回复
        print("GLM-4-9B-OpenVINO:", end=" ")
        # 将当前用户输入添加到历史记录中
        history = history + [[input_text, ""]]
        # 将对话历史转换为模型输入
        model_inputs = convert_history_to_token(history)
        # 构造生成模型所需的参数字典
        generate_kwargs = dict(
            input_ids=model_inputs,  # 输入的 ID
            max_new_tokens=args.max_sequence_length,  # 最大生成的 token 数量
            temperature=0.1,  # 生成的温度控制
            do_sample=True,  # 启用采样
            top_p=1.0,  # Nucleus 采样参数
            top_k=50,  # Top-k 采样参数
            repetition_penalty=1.1,  # 重复惩罚参数
            streamer=streamer,  # 流式输出的处理对象
            stopping_criteria=StoppingCriteriaList(stop_tokens)  # 停止生成的标准
        )

        # 创建一个线程，用于生成模型的回复
        t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        t1.start()  # 启动线程

        # 初始化部分文本为空字符串
        partial_text = ""
        # 遍历流式输出生成的文本
        for new_text in streamer:
            new_text = new_text  # 接收新生成的文本
            print(new_text, end="", flush=True)  # 输出新文本
            partial_text += new_text  # 累加到部分文本中
        print("\n")  # 输出换行
        # 更新历史记录中最后一条消息的助手回复
        history[-1][1] = partial_text
```