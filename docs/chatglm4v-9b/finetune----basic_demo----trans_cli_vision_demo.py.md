# `.\chatglm4-finetune\basic_demo\trans_cli_vision_demo.py`

```py
"""
# 此脚本创建一个命令行界面（CLI）演示，使用 transformers 后端，针对 glm-4v-9b 模型，
# 允许用户通过命令行界面与模型进行交互。

# 用法：
# - 运行脚本以启动 CLI 演示。
# - 通过输入问题与模型互动并接收响应。

# 注意：脚本包括一个处理 markdown 转换为纯文本的修改，
# 确保 CLI 界面正确显示格式化文本。
"""

# 导入操作系统模块
import os
# 导入 PyTorch 库
import torch
# 从 threading 模块导入 Thread 类
from threading import Thread
# 从 transformers 库导入相关类和函数
from transformers import (
    AutoTokenizer,  # 自动加载预训练的分词器
    StoppingCriteria,  # 停止标准类
    StoppingCriteriaList,  # 停止标准列表类
    TextIteratorStreamer,  # 文本迭代流处理器
    AutoModel,  # 自动加载预训练模型
    BitsAndBytesConfig  # 位和字节配置类
)

# 从 PIL 导入图像处理库
from PIL import Image

# 从环境变量中获取模型路径，默认值为 'THUDM/glm-4v-9b'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4v-9b')

# 从预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,  # 指定模型路径
    trust_remote_code=True,  # 信任远程代码
    encode_special_tokens=True  # 编码特殊标记
)

# 从预训练模型加载模型，并配置相关参数
model = AutoModel.from_pretrained(
    MODEL_PATH,  # 指定模型路径
    trust_remote_code=True,  # 信任远程代码
    # attn_implementation="flash_attention_2",  # 使用闪电注意力（被注释掉）
    torch_dtype=torch.bfloat16,  # 设置张量数据类型为 bfloat16
    device_map="auto",  # 自动选择设备映射
).eval()  # 将模型设置为评估模式


## 针对 INT4 推理的配置
# model = AutoModel.from_pretrained(
#     MODEL_PATH,  # 指定模型路径
#     trust_remote_code=True,  # 信任远程代码
#     quantization_config=BitsAndBytesConfig(load_in_4bit=True),  # 配置为 4 位量化
#     torch_dtype=torch.bfloat16,  # 设置张量数据类型为 bfloat16
#     low_cpu_mem_usage=True  # 低 CPU 内存使用模式
# ).eval()  # 将模型设置为评估模式

# 定义停止标准类
class StopOnTokens(StoppingCriteria):
    # 重写 __call__ 方法以定义停止条件
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id  # 获取模型的结束标记 ID
        for stop_id in stop_ids:  # 遍历结束标记 ID
            if input_ids[0][-1] == stop_id:  # 检查输入的最后一个 ID 是否为结束标记
                return True  # 如果是，返回 True 表示停止
        return False  # 否则返回 False 表示继续

# 主程序入口
if __name__ == "__main__":
    history = []  # 初始化对话历史列表
    max_length = 1024  # 设置最大输入长度
    top_p = 0.8  # 设置 top-p 采样值
    temperature = 0.6  # 设置温度控制生成文本的随机性
    stop = StopOnTokens()  # 创建停止标准实例
    uploaded = False  # 初始化上传状态
    image = None  # 初始化图像变量
    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")  # 打印欢迎信息
    image_path = input("Image Path:")  # 提示用户输入图像路径
    try:
        image = Image.open(image_path).convert("RGB")  # 尝试打开图像并转换为 RGB 模式
    except:  # 捕获任何异常
        print("Invalid image path. Continuing with text conversation.")  # 打印错误信息并继续文本对话
    # 无限循环，直到用户选择退出
    while True:
        # 获取用户输入
        user_input = input("\nYou: ")
        # 如果用户输入是 "exit" 或 "quit"，则退出循环
        if user_input.lower() in ["exit", "quit"]:
            break
        # 将用户输入添加到历史记录中，助手消息暂时为空
        history.append([user_input, ""])

        # 初始化消息列表
        messages = []
        # 遍历历史记录，获取用户和助手的消息
        for idx, (user_msg, model_msg) in enumerate(history):
            # 如果是最后一条用户消息且助手消息为空
            if idx == len(history) - 1 and not model_msg:
                # 添加用户消息到消息列表
                messages.append({"role": "user", "content": user_msg})
                # 如果有图片且尚未上传，则将图片添加到消息中
                if image and not uploaded:
                    messages[-1].update({"image": image})
                    # 标记图片已上传
                    uploaded = True
                # 结束当前循环
                break
            # 如果用户消息存在，添加到消息列表
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            # 如果助手消息存在，添加到消息列表
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        # 使用 tokenizer 处理消息，准备模型输入
        model_inputs = tokenizer.apply_chat_template(
            messages,
            # 添加生成提示
            add_generation_prompt=True,
            # 启用分词
            tokenize=True,
            # 返回 PyTorch 张量
            return_tensors="pt",
            # 返回字典格式
            return_dict=True
        ).to(next(model.parameters()).device)  # 将输入移动到模型所在的设备
        # 创建一个文本流迭代器，用于生成文本
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            # 设置超时
            timeout=60,
            # 跳过提示
            skip_prompt=True,
            # 跳过特殊字符
            skip_special_tokens=True
        )
        # 准备生成的参数
        generate_kwargs = {
            # 合并模型输入参数
            **model_inputs,
            # 指定使用流式生成
            "streamer": streamer,
            # 设置最大生成的 token 数量
            "max_new_tokens": max_length,
            # 启用采样
            "do_sample": True,
            # 设置 nucleus 采样的概率阈值
            "top_p": top_p,
            # 设置生成温度
            "temperature": temperature,
            # 设置停止条件
            "stopping_criteria": StoppingCriteriaList([stop]),
            # 设置重复惩罚
            "repetition_penalty": 1.2,
            # 指定结束 token 的 ID
            "eos_token_id": [151329, 151336, 151338],
        }
        # 创建并启动线程以生成文本
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        # 打印生成模型的提示
        print("GLM-4V:", end="", flush=True)
        # 从流中读取新生成的 token
        for new_token in streamer:
            # 如果新 token 存在
            if new_token:
                # 打印新 token
                print(new_token, end="", flush=True)
                # 将新 token 添加到历史记录的助手消息中
                history[-1][1] += new_token

        # 清理助手消息，去掉首尾空白
        history[-1][1] = history[-1][1].strip()
```