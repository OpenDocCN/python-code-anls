# `.\VisualGLM-6B\web_demo_hf.py`

```py
# 导入所需的库
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import torch

# 重写 Chatbot 类的 postprocess 方法
def postprocess(self, y):
    # 如果返回结果为空，则返回空列表
    if y is None:
        return []
    # 遍历返回结果，将消息和响应转换为 HTML 格式
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

# 将重写的 postprocess 方法应用到 Chatbot 类
gr.Chatbot.postprocess = postprocess

# 解析文本，将文本转换为 HTML 格式
def parse_text(text):
    # 根据换行符分割文本
    lines = text.split("\n")
    # 过滤空行
    lines = [line for line in lines if line != ""]
    count = 0
    # 遍历文本行
    for i, line in enumerate(lines):
        # 检查是否包含代码块标记
        if "```" in line:
            count += 1
            items = line.split('`')
            # 根据奇偶数判断代码块的开始和结束
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    # 替换特殊字符为 HTML 实体
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    # 将处理后的文本行拼接为完整文本
    text = "".join(lines)
    return text

# 预测函数，处理输入文本和图片路径
def predict(input, image_path, chatbot, max_length, top_p, temperature, history):
    # 如果图片路径为空，返回提示信息
    if image_path is None:
        return [(input, "图片不能为空。请重新上传图片并重试。")], []
    # 将输入文本转换为 HTML 格式，并添加到对话历史中
    chatbot.append((parse_text(input), ""))
    # 使用 torch.no_grad() 上下文管理器，确保在此范围内不进行梯度计算
    with torch.no_grad():
        # 遍历模型的对话流方法，生成对话回复和历史记录
        for response, history in model.stream_chat(tokenizer, image_path, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
            # 将输入和回复解析成文本，并更新最新的对话记录
            chatbot[-1] = (parse_text(input), parse_text(response))
            # 生成对话和历史记录的元组
            yield chatbot, history
# 预测新图片的描述
def predict_new_image(image_path, chatbot, max_length, top_p, temperature):
    # 初始化输入和历史记录
    input, history = "描述这张图片。", []
    chatbot.append((parse_text(input), ""))
    # 禁止梯度计算
    with torch.no_grad():
        # 使用模型流式聊天生成响应
        for response, history in model.stream_chat(tokenizer, image_path, input, history, max_length=max_length,
                                               top_p=top_p,
                                               temperature=temperature):
            # 更新最后一个对话记录
            chatbot[-1] = (parse_text(input), parse_text(response))
            # 生成聊天记录和历史记录
            yield chatbot, history

# 重置用户输入
def reset_user_input():
    return gr.update(value='')

# 重置状态
def reset_state():
    return None, [], []

# 描述
DESCRIPTION = '''<h1 align="center"><a href="https://github.com/THUDM/VisualGLM-6B">VisualGLM</a></h1>'''
# 维护通知
MAINTENANCE_NOTICE = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
# 注意事项
NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM-6B">https://github.com/THUDM/VisualGLM-6B</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'

# 主函数
def main(args):
    global model, tokenizer
    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
    # 根据量化参数加载模型
    if args.quant in [4, 8]:
        model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).quantize(args.quant).half().cuda()
    else:
        model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
    # 设置模型为评估模式
    model = model.eval()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    main(args)
```