# `.\VisualGLM-6B\cli_demo_hf.py`

```
# 导入所需的库
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import torch

# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
# 从预训练模型中加载模型，并转换为半精度浮点数，移动到 GPU 上
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
# 设置模型为评估模式
model = model.eval()

# 获取操作系统名称
os_name = platform.system()
# 根据操作系统名称设置清屏命令
clear_command = 'cls' if os_name == 'Windows' else 'clear'
# 初始化停止流标志
stop_stream = False

# 构建对话提示信息
def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nVisualGLM-6B：{response}"
    return prompt

# 信号处理函数，用于处理中断信号
def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

# 主函数入口
def main():
    global stop_stream
    # 无限循环，用于持续对话
    while True:
        # 初始化对话历史
        history = []
        # 初始化提示信息
        prefix = "欢迎使用 VisualGLM-6B 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        # 打印提示信息
        print(prefix)
        # 获取用户输入的图片路径
        image_path = input("\n请输入图片路径：")
        # 如果用户输入为 "stop"，则终止程序
        if image_path == "stop":
            break
        # 更新提示信息
        prefix = prefix + "\n" + image_path
        # 初始化查询内容
        query = "描述这张图片。"
        # 无限循环，用于持续对话
        while True:
            # 初始化计数器
            count = 0
            # 禁用梯度计算
            with torch.no_grad():
                # 遍历模型的对话流程
                for response, history in model.stream_chat(tokenizer, image_path, query, history=history):
                    # 如果需要停止对话流程
                    if stop_stream:
                        stop_stream = False
                        break
                    else:
                        count += 1
                        # 每8次打印一次对话历史和提示信息
                        if count % 8 == 0:
                            os.system(clear_command)
                            print(build_prompt(history, prefix), flush=True)
                            signal.signal(signal.SIGINT, signal_handler)
            # 清空屏幕
            os.system(clear_command)
            # 打印对话历史和提示信息
            print(build_prompt(history, prefix), flush=True)
            # 获取用户输入的查询内容
            query = input("\n用户：")
            # 如果用户输入为 "clear"，则清空对话历史并跳出当前循环
            if query.strip() == "clear":
                break
            # 如果用户输入为 "stop"，则设置停止对话流程的标志并退出程序
            if query.strip() == "stop":
                stop_stream = True
                exit(0)
            # 如果用户输入为 "clear"，则清空对话历史并跳出当前循环
            # if query.strip() == "clear":
            #     history = []
            #     os.system(clear_command)
            #     print(prefix)
            #     continue
# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```