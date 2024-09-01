# `.\flux\demo_gr.py`

```py
# 导入操作系统相关模块
import os
# 导入时间相关模块
import time
# 从 io 模块导入 BytesIO 类
from io import BytesIO
# 导入 UUID 生成模块
import uuid

# 导入 PyTorch 库
import torch
# 导入 Gradio 库
import gradio as gr
# 导入 NumPy 库
import numpy as np
# 从 einops 模块导入 rearrange 函数
from einops import rearrange
# 从 PIL 库导入 Image 和 ExifTags
from PIL import Image, ExifTags
# 从 transformers 库导入 pipeline 函数
from transformers import pipeline

# 从 flux.cli 模块导入 SamplingOptions 类
from flux.cli import SamplingOptions
# 从 flux.sampling 模块导入多个函数
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
# 从 flux.util 模块导入多个函数
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

# 设置 NSFW (不适宜工作) 图像的分类阈值
NSFW_THRESHOLD = 0.85

# 定义获取模型的函数
def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    # 加载 T5 模型，长度限制根据是否为 schnell 模型决定
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    # 加载 CLIP 模型
    clip = load_clip(device)
    # 加载流动模型，根据是否卸载来决定使用 CPU 还是设备
    model = load_flow_model(name, device="cpu" if offload else device)
    # 加载自编码器模型，同样根据是否卸载来决定使用 CPU 还是设备
    ae = load_ae(name, device="cpu" if offload else device)
    # 创建 NSFW 分类器管道
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    # 返回加载的模型和分类器
    return model, ae, t5, clip, nsfw_classifier

# 定义 FluxGenerator 类
class FluxGenerator:
    # 类的初始化函数
    def __init__(self, model_name: str, device: str, offload: bool):
        # 将设备字符串转换为 torch.device 对象
        self.device = torch.device(device)
        # 是否卸载的标志
        self.offload = offload
        # 模型名称
        self.model_name = model_name
        # 判断是否为 schnell 模型
        self.is_schnell = model_name == "flux-schnell"
        # 获取模型及相关组件
        self.model, self.ae, self.t5, self.clip, self.nsfw_classifier = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            is_schnell=self.is_schnell,
        )

    # 使用 torch 的推理模式生成图像
    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        guidance,
        seed,
        prompt,
        init_image=None,
        image2image_strength=0.0,
        add_sampling_metadata=True,
    # 定义创建演示的函数
def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    # 初始化 FluxGenerator 对象
    generator = FluxGenerator(model_name, device, offload)
    # 判断是否为 schnell 模型
    is_schnell = model_name == "flux-schnell"
    # 创建一个 Gradio 应用的 UI 布局
    with gr.Blocks() as demo:
        # 添加标题 Markdown 文本，显示模型名称
        gr.Markdown(f"# Flux Image Generation Demo - Model: {model_name}")
        
        # 创建一行布局
        with gr.Row():
            # 创建一列布局
            with gr.Column():
                # 创建一个文本框用于输入提示
                prompt = gr.Textbox(label="Prompt", value="a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture")
                # 创建一个复选框用于选择是否启用图像到图像转换
                do_img2img = gr.Checkbox(label="Image to Image", value=False, interactive=not is_schnell)
                # 创建一个隐藏的图像输入框
                init_image = gr.Image(label="Input Image", visible=False)
                # 创建一个隐藏的滑块，用于调整图像到图像转换的强度
                image2image_strength = gr.Slider(0.0, 1.0, 0.8, step=0.1, label="Noising strength", visible=False)
                
                # 创建一个可折叠的高级选项区域
                with gr.Accordion("Advanced Options", open=False):
                    # 创建滑块用于设置图像宽度
                    width = gr.Slider(128, 8192, 1360, step=16, label="Width")
                    # 创建滑块用于设置图像高度
                    height = gr.Slider(128, 8192, 768, step=16, label="Height")
                    # 创建滑块用于设置步骤数，根据是否快速模式设置初始值
                    num_steps = gr.Slider(1, 50, 4 if is_schnell else 50, step=1, label="Number of steps")
                    # 创建滑块用于设置指导强度
                    guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Guidance", interactive=not is_schnell)
                    # 创建一个文本框用于输入种子值
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")
                    # 创建一个复选框用于选择是否将采样参数添加到元数据
                    add_sampling_metadata = gr.Checkbox(label="Add sampling parameters to metadata?", value=True)
                
                # 创建一个生成按钮
                generate_btn = gr.Button("Generate")
            
            # 创建另一列布局
            with gr.Column():
                # 创建一个图像框用于显示生成的图像
                output_image = gr.Image(label="Generated Image")
                # 创建一个数字框用于显示使用的种子
                seed_output = gr.Number(label="Used Seed")
                # 创建一个文本框用于显示警告信息
                warning_text = gr.Textbox(label="Warning", visible=False)
                # 创建一个文件框用于下载高分辨率图像
                download_btn = gr.File(label="Download full-resolution")

        # 定义一个函数，用于更新图像到图像转换的可见性
        def update_img2img(do_img2img):
            return {
                init_image: gr.update(visible=do_img2img),
                image2image_strength: gr.update(visible=do_img2img),
            }

        # 当复选框状态变化时，调用更新函数
        do_img2img.change(update_img2img, do_img2img, [init_image, image2image_strength])

        # 设置生成按钮的点击事件，调用生成图像的函数并设置输入和输出
        generate_btn.click(
            fn=generator.generate_image,
            inputs=[width, height, num_steps, guidance, seed, prompt, init_image, image2image_strength, add_sampling_metadata],
            outputs=[output_image, seed_output, download_btn, warning_text],
        )

    # 返回创建的 Gradio 应用布局
    return demo
# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 导入 argparse 模块用于处理命令行参数
    import argparse
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="Flux")
    # 添加 --name 参数，指定模型名称，默认值为 "flux-schnell"，并限制选择范围
    parser.add_argument("--name", type=str, default="flux-schnell", choices=list(configs.keys()), help="Model name")
    # 添加 --device 参数，指定设备，默认值为 "cuda"（如果有 GPU 可用），否则为 "cpu"
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    # 添加 --offload 参数，标志位，指示是否在不使用时将模型移到 CPU
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    # 添加 --share 参数，标志位，指示是否创建一个公共链接以共享演示
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    # 解析命令行参数，并将结果存储在 args 对象中
    args = parser.parse_args()

    # 使用解析出的参数创建 demo 对象
    demo = create_demo(args.name, args.device, args.offload)
    # 启动 demo，是否共享由 --share 参数决定
    demo.launch(share=args.share)
```