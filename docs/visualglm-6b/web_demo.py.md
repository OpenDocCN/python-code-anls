# `.\VisualGLM-6B\web_demo.py`

```
#!/usr/bin/env python

# 导入必要的库
import gradio as gr
from PIL import Image
import os
import json
from model import is_chinese, get_infer_setting, generate_input, chat
import torch

# 根据输入文本和图片生成对话文本
def generate_text_with_image(input_text, image, history=[], request_data=dict(), is_zh=True):
    # 设置生成文本的参数
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    # 生成输入数据
    input_data = generate_input(input_text, image, history, input_para, image_is_encoded=False)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    
    # 使用模型生成对话文本
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
    return answer

# 请求模型生成对话文本
def request_model(input_text, temperature, top_p, image_prompt, result_previous):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][1] == "":
            del result_text[i]
    print(f"history {result_text}")

    # 判断输入文本是否为中文
    is_zh = is_chinese(input_text)
    
    # 处理图片输入
    if image_prompt is None:
        if is_zh:
            result_text.append((input_text, '图片为空！请上传图片并重试。'))
        else:
            result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
        return input_text, result_text
    elif input_text == "":
        result_text.append((input_text, 'Text empty! Please enter text and retry.'))
        return "", result_text                

    # 打开图片文件
    image = Image.open(image_prompt)
    # 尝试生成包含文本和图片的结果
    try:
        answer = generate_text_with_image(input_text, image, result_text.copy(), request_para, is_zh)
    # 捕获异常并打印错误信息
    except Exception as e:
        print(f"error: {e}")
        # 根据语言添加超时提示到结果文本中
        if is_zh:
            result_text.append((input_text, '超时！请稍等几分钟再重试。'))
        else:
            result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        # 返回空字符串和更新后的结果文本
        return "", result_text

    # 将输入文本和生成的答案添加到结果文本中
    result_text.append((input_text, answer))
    # 打印结果文本
    print(result_text)
    # 返回空字符串和更新后的结果文本
    return "", result_text
# 描述信息，包含链接到 GitHub 仓库
DESCRIPTION = '''# <a href="https://github.com/THUDM/VisualGLM-6B">VisualGLM</a>'''

# 维护提示信息，包含两条提示
MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
MAINTENANCE_NOTICE2 = '提示1: 如果应用报了“Something went wrong, connection error out”的错误，请关闭代理并重试。\n提示2: 如果你上传了很大的图片，比如10MB大小，那将需要一些时间来上传和处理，请耐心等待。'

# 注意事项信息，包含链接到 GitHub 仓库
NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM-6B">https://github.com/THUDM/VisualGLM-6B</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'

# 清除函数，返回空字符串、包含一条消息的列表和空值
def clear_fn(value):
    return "", [("", "Hi, What do you want to know about this image?")], None

# 清除函数2，返回包含一条消息的列表
def clear_fn2(value):
    return [("", "Hi, What do you want to know about this image?")]

# 主函数，关闭所有图形窗口，获取推断设置并赋值给全局变量 model 和 tokenizer
def main(args):
    gr.close_all()
    global model, tokenizer
    model, tokenizer = get_infer_setting(gpu_device=0, quant=args.quant)
    # 创建一个包含样式表的 Blocks 对象
    with gr.Blocks(css='style.css') as demo:
        # 在 Blocks 中添加 Markdown 组件，显示描述信息
        gr.Markdown(DESCRIPTION)
        # 创建一个 Row 区块
        with gr.Row():
            # 创建一个 Column 区块，设置比例为 4.5
            with gr.Column(scale=4.5):
                # 创建一个 Group 区块
                with gr.Group():
                    # 创建一个 Textbox 组件，用于输入文本提示
                    input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                    # 创建一个 Row 区块
                    with gr.Row():
                        # 创建一个按钮组件，用于生成结果
                        run_button = gr.Button('Generate')
                        # 创建一个按钮组件，用于清除输入
                        clear_button = gr.Button('Clear')

                    # 创建一个 Image 组件，用于上传图片提示
                    image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                # 创建一个 Row 区块
                with gr.Row():
                    # 创建一个滑块组件，用于调整温度
                    temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                    # 创建一个滑块组件，用于调整 Top P
                    top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                # 创建一个 Group 区块
                with gr.Group():
                    # 创建一个 Row 区块
                    with gr.Row():
                        # 创建一个 Markdown 组件，显示维护通知信息
                        maintenance_notice = gr.Markdown(MAINTENANCE_NOTICE1)
            # 创建一个 Column 区块，设置比例为 5.5
            with gr.Column(scale=5.5):
                # 创建一个 Chatbot 组件，用于显示多轮对话历史
                result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about this image?")]).style(height=550)

        # 在 Blocks 中添加 Markdown 组件，显示注意事项
        gr.Markdown(NOTES)

        # 打印 gr 库的版本号
        print(gr.__version__)
        # 绑定按钮点击事件，调用 request_model 函数
        run_button.click(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        # 绑定文本框提交事件，调用 request_model 函数
        input_text.submit(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        # 绑定清除按钮点击事件，调用 clear_fn 函数
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        # 绑定图片上传事件，调用 clear_fn2 函数
        image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
        # 绑定清除图片事件，调用 clear_fn2 函数
        image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])

        # 打印 gr 库的版本号
        print(gr.__version__)

    # 设置 Blocks 对象的并发计数为 10
    demo.queue(concurrency_count=10)
    # 启动 Blocks 对象，分享参数为 args.share
    demo.launch(share=args.share)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == '__main__':
    # 导入 argparse 模块，用于解析命令行参数
    import argparse
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数 "--quant"，可选值为 8 或 4，类型为整数，默认为 None
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    # 添加命令行参数 "--share"，如果存在则设置为 True
    parser.add_argument("--share", action="store_true")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用 main 函数，并传入解析后的参数
    main(args)
```