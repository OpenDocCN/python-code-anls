# `.\flux\demo_st.py`

```py
# 导入操作系统相关功能
import os
# 导入正则表达式处理功能
import re
# 导入时间处理功能
import time
# 从 glob 模块导入 iglob，用于生成匹配特定模式的文件路径
from glob import iglob
# 从 io 模块导入 BytesIO，用于处理字节流
from io import BytesIO

# 导入 streamlit 库，用于创建 Web 应用
import streamlit as st
# 导入 PyTorch 库，用于深度学习模型
import torch
# 从 einops 库导入 rearrange，用于张量的重排
from einops import rearrange
# 从 fire 库导入 Fire，用于将命令行参数绑定到函数
from fire import Fire
# 从 PIL 库导入 ExifTags 和 Image，用于图像处理
from PIL import ExifTags, Image
# 从 st_keyup 库导入 st_keyup，用于捕捉键盘事件
from st_keyup import st_keyup
# 从 torchvision 库导入 transforms，用于图像转换
from torchvision import transforms
# 从 transformers 库导入 pipeline，用于各种预训练模型的管道
from transformers import pipeline

# 设置 NSFW 内容的阈值
NSFW_THRESHOLD = 0.85


# 使用 Streamlit 缓存模型加载函数的结果，以提高性能
@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    # 加载 T5 模型，最大长度取决于是否使用 Schnell 模式
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    # 加载 CLIP 模型
    clip = load_clip(device)
    # 加载流模型，设备可能是 CPU 或 GPU
    model = load_flow_model(name, device="cpu" if offload else device)
    # 加载自动编码器模型，设备可能是 CPU 或 GPU
    ae = load_ae(name, device="cpu" if offload else device)
    # 加载 NSFW 分类器，用于图像内容检测
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    # 返回模型、自动编码器、T5、CLIP 和 NSFW 分类器
    return model, ae, t5, clip, nsfw_classifier


# 获取用户上传的图像，返回处理后的张量
def get_image() -> torch.Tensor | None:
    # 允许用户上传 JPG、JPEG 或 PNG 格式的图像
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"])
    # 如果没有上传图像，返回 None
    if image is None:
        return None
    # 打开图像文件并转换为 RGB 模式
    image = Image.open(image).convert("RGB")

    # 定义图像转换操作，将图像转为张量，并进行归一化
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    # 应用转换，将图像处理为张量，并增加一个维度
    img: torch.Tensor = transform(image)
    return img[None, ...]


# 主函数，用于运行应用逻辑
@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    # 根据用户选择的设备创建 PyTorch 设备对象
    torch_device = torch.device(device)
    # 获取配置中的模型名称列表
    names = list(configs.keys())
    # 让用户选择要加载的模型
    name = st.selectbox("Which model to load?", names)
    # 如果未选择模型或未勾选加载模型的复选框，则返回
    if name is None or not st.checkbox("Load model", False):
        return

    # 判断是否使用 Schnell 模式
    is_schnell = name == "flux-schnell"
    # 获取所需的模型和分类器
    model, ae, t5, clip, nsfw_classifier = get_models(
        name,
        device=torch_device,
        offload=offload,
        is_schnell=is_schnell,
    )

    # 判断是否执行图像到图像的转换
    do_img2img = (
        st.checkbox(
            "Image to Image",
            False,
            disabled=is_schnell,
            help="Partially noise an image and denoise again to get variations.\n\nOnly works for flux-dev",
        )
        and not is_schnell
    )
    # 如果需要图像到图像转换
    if do_img2img:
        # 获取用户上传的图像
        init_image = get_image()
        # 如果没有上传图像，显示警告信息
        if init_image is None:
            st.warning("Please add an image to do image to image")
        # 让用户输入噪声强度
        image2image_strength = st.number_input("Noising strength", min_value=0.0, max_value=1.0, value=0.8)
        # 如果上传了图像，显示图像尺寸
        if init_image is not None:
            h, w = init_image.shape[-2:]
            st.write(f"Got image of size {w}x{h} ({h*w/1e6:.2f}MP)")
        # 让用户选择是否调整图像大小
        resize_img = st.checkbox("Resize image", False) or init_image is None
    else:
        # 如果不进行图像到图像转换，初始化图像和图像调整标志
        init_image = None
        resize_img = True
        image2image_strength = 0.0

    # 允许进行打包和转换到潜在空间
    # 根据用户输入的宽度值计算实际宽度，确保宽度为16的倍数
    width = int(
        16 * (st.number_input("Width", min_value=128, value=1360, step=16, disabled=not resize_img) // 16)
    )
    # 根据用户输入的高度值计算实际高度，确保高度为16的倍数
    height = int(
        16 * (st.number_input("Height", min_value=128, value=768, step=16, disabled=not resize_img) // 16)
    )
    # 根据用户输入的步数值设置步数，默认值为4（如果是"schnell"模式），否则为50
    num_steps = int(st.number_input("Number of steps", min_value=1, value=(4 if is_schnell else 50)))
    # 根据用户输入的引导值设置引导参数，默认为3.5，"schnell"模式下禁用此输入
    guidance = float(st.number_input("Guidance", min_value=1.0, value=3.5, disabled=is_schnell))
    # 根据用户输入的种子值设置种子，"schnell"模式下禁用此输入
    seed_str = st.text_input("Seed", disabled=is_schnell)
    # 如果种子值是有效的十进制数，则将其转换为整数；否则，设置种子为None，并显示提示信息
    if seed_str.isdecimal():
        seed = int(seed_str)
    else:
        st.info("No seed set, set to positive integer to enable")
        seed = None
    # 根据用户选择是否保存样本，设置保存样本的选项
    save_samples = st.checkbox("Save samples?", not is_schnell)
    # 根据用户选择是否将采样参数添加到元数据中，设置此选项
    add_sampling_metadata = st.checkbox("Add sampling parameters to metadata?", True)

    # 默认提示文本，用于生成图像
    default_prompt = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    )
    # 获取用户输入的提示文本，默认值为default_prompt，并设置300毫秒的防抖延迟
    prompt = st_keyup("Enter a prompt", value=default_prompt, debounce=300, key="interactive_text")

    # 构造输出文件名的路径，并检查输出目录是否存在
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        # 如果输出目录不存在，则创建目录，并初始化索引为0
        os.makedirs(output_dir)
        idx = 0
    else:
        # 如果输出目录存在，获取所有匹配的文件名，并计算下一个可用的索引
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # 创建一个 PyTorch 随机数生成器对象
    rng = torch.Generator(device="cpu")

    # 如果 session_state 中没有“seed”项，则初始化种子
    if "seed" not in st.session_state:
        st.session_state.seed = rng.seed()

    # 定义增加种子值的函数
    def increment_counter():
        st.session_state.seed += 1

    # 定义减少种子值的函数（种子值不能小于0）
    def decrement_counter():
        if st.session_state.seed > 0:
            st.session_state.seed -= 1

    # 创建一个采样选项对象，用于后续处理
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    # 如果应用名为“flux-schnell”，则显示带有按钮的列来增加或减少种子值
    if name == "flux-schnell":
        cols = st.columns([5, 1, 1, 5])
        with cols[1]:
            st.button("↩", on_click=increment_counter)
        with cols[2]:
            st.button("↪", on_click=decrement_counter)
    # 获取会话状态中的样本（如果存在），并显示图像及其相关信息
    samples = st.session_state.get("samples", None)
    if samples is not None:
        st.image(samples["img"], caption=samples["prompt"])
        st.download_button(
            "Download full-resolution",
            samples["bytes"],
            file_name="generated.jpg",
            mime="image/jpg",
        )
        st.write(f"Seed: {samples['seed']}")
# 定义应用程序入口函数
def app():
    # 调用 Fire 函数并传入 main 作为参数
    Fire(main)


# 如果脚本是主程序（而不是被导入），则执行 app() 函数
if __name__ == "__main__":
    app()
```