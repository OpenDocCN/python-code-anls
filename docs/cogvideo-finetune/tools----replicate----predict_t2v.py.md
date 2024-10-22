# `.\cogvideo-finetune\tools\replicate\predict_t2v.py`

```py
# Cog的预测接口 ⚙️
# https://cog.run/python

# 导入必要的库和模块
import os  # 用于与操作系统交互
import subprocess  # 用于执行子进程
import time  # 用于时间管理
import torch  # 深度学习框架
from diffusers import CogVideoXPipeline  # 导入CogVideoXPipeline类
from diffusers.utils import export_to_video  # 导入视频导出工具
from cog import BasePredictor, Input, Path  # 导入Cognition框架的基础类和输入类

MODEL_CACHE = "model_cache"  # 定义模型缓存目录
MODEL_URL = (  # 定义模型权重下载URL
    f"https://weights.replicate.delivery/default/THUDM/CogVideo/{MODEL_CACHE}.tar"
)
# 设置环境变量，强制使用离线模式以避免下载模型
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE  # 设置Hugging Face的根目录
os.environ["TORCH_HOME"] = MODEL_CACHE  # 设置PyTorch的根目录
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE  # 设置数据集缓存目录
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE  # 设置变换器缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE  # 设置Hugging Face Hub缓存目录

# 定义下载权重的函数
def download_weights(url, dest):
    start = time.time()  # 记录开始时间
    print("downloading url: ", url)  # 输出下载URL
    print("downloading to: ", dest)  # 输出下载目标路径
    # 使用子进程命令下载文件
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # 输出下载所需时间

# 定义预测器类，继承自BasePredictor
class Predictor(BasePredictor):
    def setup(self) -> None:
        """将模型加载到内存中，以提高多次预测的效率"""

        # 检查模型缓存目录是否存在
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)  # 如果不存在，下载模型权重

        # 加载指定的模型
        # model_id: THUDM/CogVideoX-5b
        self.pipe = CogVideoXPipeline.from_pretrained(
            MODEL_CACHE,  # 使用缓存的模型
            torch_dtype=torch.bfloat16,  # 设置模型的数据类型
        ).to("cuda")  # 将模型移动到GPU

        self.pipe.enable_model_cpu_offload()  # 启用CPU卸载以优化内存使用
        self.pipe.vae.enable_tiling()  # 启用VAE的分块处理

    # 定义预测方法
    def predict(
        self,
        prompt: str = Input(  # 输入提示，描述生成内容
            description="Input prompt",
            default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.",
        ),
        num_inference_steps: int = Input(  # 输入去噪步骤数量
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(  # 输入无分类指导的比例
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        num_frames: int = Input(  # 输入输出视频的帧数
            description="Number of frames for the output video", default=49
        ),
        seed: int = Input(  # 输入随机种子，留空以随机化
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    # 预测模型的单次运行，返回生成的视频路径
    ) -> Path:
        # 文档字符串，说明函数的功能
        """Run a single prediction on the model"""
    
        # 如果没有提供种子，则随机生成一个种子
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        # 输出当前使用的种子
        print(f"Using seed: {seed}")
    
        # 调用模型管道生成视频，使用提供的参数
        video = self.pipe(
            # 传入的提示文本
            prompt=prompt,
            # 每个提示生成一个视频
            num_videos_per_prompt=1,
            # 推理步骤的数量
            num_inference_steps=num_inference_steps,
            # 视频帧数
            num_frames=num_frames,
            # 指导比例
            guidance_scale=guidance_scale,
            # 设定随机数生成器，使用指定的种子
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]  # 取得生成的视频的第一帧
    
        # 设置视频输出路径
        out_path = "/tmp/out.mp4"
    
        # 将生成的视频导出为 MP4 文件，帧率为8
        export_to_video(video, out_path, fps=8)
        # 返回输出路径的 Path 对象
        return Path(out_path)
```