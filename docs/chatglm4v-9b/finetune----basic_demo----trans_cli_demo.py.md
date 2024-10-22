# `.\chatglm4-finetune\basic_demo\trans_cli_demo.py`

```
"""
# 该脚本创建一个 CLI 演示，使用 transformers 后端的 glm-4-9b 模型，
# 允许用户通过命令行接口与模型进行交互。

# 用法：
# - 运行脚本以启动 CLI 演示。
# - 通过输入问题与模型进行互动，并接收响应。

# 注意：该脚本包含处理 markdown 到纯文本转换的修改，
# 确保 CLI 接口正确显示格式化文本。

# 如果使用闪存注意力，您应该安装 flash-attn 并在模型加载中添加 attn_implementation="flash_attention_2"。
"""

# 导入操作系统模块
import os
# 导入 PyTorch 库
import torch
# 从 threading 模块导入 Thread 类
from threading import Thread
# 从 transformers 模块导入相关类
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel

# 从环境变量获取模型路径，默认值为 'THUDM/glm-4-9b-chat'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

## 如果使用 peft 模型。
# def load_model_and_tokenizer(model_dir, trust_remote_code: bool = True):
#     # 检查适配器配置文件是否存在
#     if (model_dir / 'adapter_config.json').exists():
#         # 从预训练模型加载模型，设置为自动设备映射
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         # 获取基本模型名称
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path
#     else:
#         # 从预训练模型加载模型，设置为自动设备映射
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         # 将模型目录作为 tokenizer 目录
#         tokenizer_dir = model_dir
#     # 从预训练 tokenizer 目录加载 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
#     )
#     # 返回模型和 tokenizer
#     return model, tokenizer

# 从预训练模型路径加载 tokenizer，允许远程代码
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 从预训练模型路径加载模型，允许远程代码，设置为自动设备映射并评估模式
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2", # 使用闪存注意力
    # torch_dtype=torch.bfloat16, # 使用闪存注意力必须使用 bfloat16 或 float16
    device_map="auto").eval()

# 定义 StopOnTokens 类，继承自 StoppingCriteria
class StopOnTokens(StoppingCriteria):
    # 定义调用方法，检查是否满足停止条件
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取模型的结束标记 ID
        stop_ids = model.config.eos_token_id
        # 遍历停止 ID 列表
        for stop_id in stop_ids:
            # 如果最后一个输入 ID 等于停止 ID，则返回 True
            if input_ids[0][-1] == stop_id:
                return True
        # 否则返回 False
        return False

# 如果该脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 初始化历史记录列表
    history = []
    # 设置最大输入长度
    max_length = 8192
    # 设置 top_p 参数
    top_p = 0.8
    # 设置温度参数
    temperature = 0.6
    # 实例化停止条件类
    stop = StopOnTokens()

    # 打印欢迎信息
    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    # 无限循环，直到用户输入退出指令
        while True:
            # 获取用户输入
            user_input = input("\nYou: ")
            # 如果用户输入为“退出”或“结束”，则跳出循环
            if user_input.lower() in ["exit", "quit"]:
                break
            # 将用户输入添加到历史记录中，模型回复先留空
            history.append([user_input, ""])
    
            # 初始化消息列表
            messages = []
            # 遍历历史记录，获取用户和模型消息
            for idx, (user_msg, model_msg) in enumerate(history):
                # 如果是最后一条消息且模型回复为空，添加用户消息
                if idx == len(history) - 1 and not model_msg:
                    messages.append({"role": "user", "content": user_msg})
                    break
                # 如果用户消息存在，添加到消息列表
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                # 如果模型消息存在，添加到消息列表
                if model_msg:
                    messages.append({"role": "assistant", "content": model_msg})
            # 将消息列表转换为模型输入格式
            model_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # 添加生成提示
                tokenize=True,  # 启用分词
                return_tensors="pt"  # 返回 PyTorch 张量格式
            ).to(model.device)  # 移动到模型设备
            # 创建文本流处理器
            streamer = TextIteratorStreamer(
                tokenizer=tokenizer,  # 使用的分词器
                timeout=60,  # 设置超时时间
                skip_prompt=True,  # 跳过提示
                skip_special_tokens=True  # 跳过特殊标记
            )
            # 设置生成参数
            generate_kwargs = {
                "input_ids": model_inputs,  # 模型输入 ID
                "streamer": streamer,  # 文本流处理器
                "max_new_tokens": max_length,  # 生成的最大新 token 数量
                "do_sample": True,  # 启用采样
                "top_p": top_p,  # 采样阈值
                "temperature": temperature,  # 温度参数
                "stopping_criteria": StoppingCriteriaList([stop]),  # 停止标准
                "repetition_penalty": 1.2,  # 重复惩罚
                "eos_token_id": model.config.eos_token_id,  # 结束 token ID
            }
            # 创建一个新线程来生成模型输出
            t = Thread(target=model.generate, kwargs=generate_kwargs)
            t.start()  # 启动线程
            # 打印模型输出提示
            print("GLM-4:", end="", flush=True)
            # 从流处理器中获取新生成的 token
            for new_token in streamer:
                if new_token:
                    # 打印生成的 token
                    print(new_token, end="", flush=True)
                    # 将新 token 添加到最后一条历史记录的模型回复中
                    history[-1][1] += new_token
    
            # 清理最后一条模型回复的前后空格
            history[-1][1] = history[-1][1].strip()
```