# `.\cogvideo-finetune\tools\replicate\predict_i2v.py`

```py
# Cog 的预测接口 ⚙️
# https://cog.run/python

# 导入必要的库
import os  # 用于操作系统功能
import subprocess  # 用于执行子进程命令
import time  # 用于时间相关操作
import torch  # 用于深度学习库
from diffusers import CogVideoXImageToVideoPipeline  # 导入视频生成管道
from diffusers.utils import export_to_video, load_image  # 导入工具函数
from cog import BasePredictor, Input, Path  # 导入 Cog 的基础预测器和输入处理

# 定义模型缓存目录
MODEL_CACHE = "model_cache_i2v"
# 定义模型下载的 URL
MODEL_URL = (
    f"https://weights.replicate.delivery/default/THUDM/CogVideo/{MODEL_CACHE}.tar"
)
# 设置环境变量以离线模式运行
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 禁用数据集在线下载
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 禁用变换器在线下载
os.environ["HF_HOME"] = MODEL_CACHE  # 设置 Hugging Face 的缓存目录
os.environ["TORCH_HOME"] = MODEL_CACHE  # 设置 PyTorch 的缓存目录
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE  # 设置数据集缓存目录
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE  # 设置变换器缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE  # 设置 Hugging Face Hub 缓存目录

# 定义下载模型权重的函数
def download_weights(url, dest):
    start = time.time()  # 记录开始时间
    print("downloading url: ", url)  # 输出下载 URL
    print("downloading to: ", dest)  # 输出下载目标路径
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)  # 调用 pget 下载模型权重
    print("downloading took: ", time.time() - start)  # 输出下载所用时间

# 定义预测类
class Predictor(BasePredictor):
    def setup(self) -> None:
        """将模型加载到内存中以提高多个预测的效率"""

        # 如果模型缓存目录不存在，则下载模型权重
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # 使用预训练模型初始化管道
        # model_id: THUDM/CogVideoX-5b-I2V
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_CACHE, torch_dtype=torch.bfloat16  # 使用 bfloat16 数据类型
        ).to("cuda")  # 将模型转移到 GPU

        self.pipe.enable_model_cpu_offload()  # 启用模型 CPU 离线处理
        self.pipe.vae.enable_tiling()  # 启用 VAE 的平铺处理

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="Starry sky slowly rotating."
        ),  # 输入提示的默认值
        image: Path = Input(description="Input image"),  # 输入图像路径
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),  # 去噪步骤数量的输入
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),  # 分类无关引导的比例输入
        num_frames: int = Input(
            description="Number of frames for the output video", default=49
        ),  # 输出视频的帧数输入
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),  # 随机种子的输入
    ) -> Path:
        """对模型进行单次预测"""

        # 如果没有提供种子，则生成一个随机种子
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")  # 生成随机种子
        print(f"Using seed: {seed}")  # 输出使用的种子

        img = load_image(image=str(image))  # 加载输入图像

        # 调用管道进行视频生成
        video = self.pipe(
            prompt=prompt,  # 输入提示
            image=img,  # 输入图像
            num_videos_per_prompt=1,  # 每个提示生成一个视频
            num_inference_steps=num_inference_steps,  # 去噪步骤数量
            num_frames=num_frames,  # 输出视频帧数
            guidance_scale=guidance_scale,  # 分类无关引导比例
            generator=torch.Generator(device="cuda").manual_seed(seed),  # 随机数生成器
        ).frames[0]  # 获取生成的视频帧

        out_path = "/tmp/out.mp4"  # 设置输出视频的路径

        export_to_video(video, out_path, fps=8)  # 导出视频到指定路径
        return Path(out_path)  # 返回输出视频路径
```