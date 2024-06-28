# `.\models\git\convert_git_to_pytorch.py`

```py
# 设置脚本的编码格式为 UTF-8
# 版权声明，声明代码归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0
# 获取命令行参数解析器
import argparse
# 导入路径处理模块 Path
from pathlib import Path

# 导入 numpy 库，用于科学计算
import numpy as np
# 导入 requests 库，用于发送 HTTP 请求
import requests
# 导入 PyTorch 深度学习库
import torch
# 从 huggingface_hub 库中导入 hf_hub_download 函数
from huggingface_hub import hf_hub_download
# 导入 PIL 库中的 Image 模块，用于图像处理
from PIL import Image
# 从 torchvision.transforms 模块导入图像预处理函数
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

# 从 transformers 库中导入相关模块和类
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GitConfig,
    GitForCausalLM,
    GitProcessor,
    GitVisionConfig,
    VideoMAEImageProcessor,
)
# 从 transformers.utils 模块中导入 logging 模块
from transformers.utils import logging

# 设置日志输出级别为 INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 定义函数，根据模型名称获取 GitConfig 对象
def get_git_config(model_name):
    # 根据模型名称设置图像大小
    if "base" in model_name and "vqa" in model_name:
        image_size = 480
    elif "large" in model_name and "vqa" in model_name:
        image_size = 420
    else:
        image_size = 224

    # 创建 GitVisionConfig 对象，设置图像大小
    vision_config = GitVisionConfig(image_size=image_size)

    # 如果模型名称中包含 "large"，则设置更大的模型参数
    if "large" in model_name:
        vision_config.patch_size = 14
        vision_config.hidden_size = 1024
        vision_config.intermediate_size = 4096
        vision_config.num_hidden_layers = 24
        vision_config.num_attention_heads = 16

    # 根据模型名称判断是否处理视频
    is_video = "vatex" in model_name or "msrvtt" in model_name
    # 如果处理视频，则设置 num_image_with_embedding 为 6，否则为 None
    num_image_with_embedding = 6 if is_video else None
    # 创建 GitConfig 对象，包含视觉配置和图像嵌入数量
    config = GitConfig(vision_config=vision_config.to_dict(), num_image_with_embedding=num_image_with_embedding)

    return config, image_size, is_video


# 定义函数，创建用于重命名的键列表
def create_rename_keys(config, prefix=""):
    rename_keys = []

    # 图像编码器部分的键重命名
    # ftm: off
    rename_keys.append(
        (f"{prefix}image_encoder.class_embedding", "git.image_encoder.vision_model.embeddings.class_embedding")
    )
    rename_keys.append(
        (
            f"{prefix}image_encoder.positional_embedding",
            "git.image_encoder.vision_model.embeddings.position_embedding.weight",
        )
    )
    rename_keys.append(
        (f"{prefix}image_encoder.conv1.weight", "git.image_encoder.vision_model.embeddings.patch_embedding.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_pre.weight", "git.image_encoder.vision_model.pre_layrnorm.weight"))
    rename_keys.append((f"{prefix}image_encoder.ln_pre.bias", "git.image_encoder.vision_model.pre_layrnorm.bias"))
    rename_keys.append(
        (f"{prefix}image_encoder.ln_post.weight", "git.image_encoder.vision_model.post_layernorm.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_post.bias", "git.image_encoder.vision_model.post_layernorm.bias"))
    # 将旧的键和新的键对添加到 rename_keys 列表中，用于重命名权重和偏置项

    # fmt: on
    rename_keys.append((f"{prefix}image_encoder.proj", "git.image_encoder.visual_projection.weight"))
    # 将旧的键和新的键对添加到 rename_keys 列表中，用于重命名视觉投影的权重

    # fmt: off
    for i in range(config.vision_config.num_hidden_layers):
        # 对于每一个视觉编码器的层，依次添加权重和偏置项的重命名对
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias"))
    # fmt: on

    # text decoder
    # fmt: off
    rename_keys.append((f"{prefix}textual.embedding.words.weight", "git.embeddings.word_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.embedding.positions.weight", "git.embeddings.position_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.weight", "git.visual_projection.visual_projection.0.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.bias", "git.visual_projection.visual_projection.0.bias"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.weight", "git.visual_projection.visual_projection.1.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.bias", "git.visual_projection.visual_projection.1.bias"))
    # 将文本解码器相关的旧的键和新的键对添加到 rename_keys 列表中，用于重命名文本嵌入和视觉投影的权重和偏置项
    # 将需要重命名的键值对添加到 rename_keys 列表中
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.weight", "git.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.bias", "git.embeddings.LayerNorm.bias"))
    rename_keys.append((f"{prefix}textual.output.weight", "output.weight"))
    rename_keys.append((f"{prefix}textual.output.bias", "output.bias"))
    
    # 遍历配置中指定的隐藏层数量，生成对应的键值对并添加到 rename_keys 中
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.weight", f"git.encoder.layer.{i}.attention.self.query.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.bias", f"git.encoder.layer.{i}.attention.self.query.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.weight", f"git.encoder.layer.{i}.attention.self.key.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.bias", f"git.encoder.layer.{i}.attention.self.key.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.weight", f"git.encoder.layer.{i}.attention.self.value.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.bias", f"git.encoder.layer.{i}.attention.self.value.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.weight", f"git.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.bias", f"git.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.weight", f"git.encoder.layer.{i}.attention.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.bias", f"git.encoder.layer.{i}.attention.output.LayerNorm.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.weight", f"git.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.bias", f"git.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.weight", f"git.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.bias", f"git.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.weight", f"git.encoder.layer.{i}.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.bias", f"git.encoder.layer.{i}.output.LayerNorm.bias"))
    # fmt: on
    # 如果配置中指定了嵌入图像的数量，则执行以下操作
    if config.num_image_with_embedding is not None:
        # 将以下键值对添加到重命名键列表中，用于重命名图像临时嵌入的索引
        rename_keys.append(("img_temperal_embedding.0", "git.img_temperal_embedding.0"))
        rename_keys.append(("img_temperal_embedding.1", "git.img_temperal_embedding.1"))
        rename_keys.append(("img_temperal_embedding.2", "git.img_temperal_embedding.2"))
        rename_keys.append(("img_temperal_embedding.3", "git.img_temperal_embedding.3"))
        rename_keys.append(("img_temperal_embedding.4", "git.img_temperal_embedding.4"))
        rename_keys.append(("img_temperal_embedding.5", "git.img_temperal_embedding.5"))

    # 返回更新后的重命名键列表
    return rename_keys
# 从字典中移除旧键，将其对应的值保存到变量val中
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 如果新键中包含特定字符串，则对值进行转置操作
    dct[new] = val.T if "image_encoder.visual_projection" in new else val


# 从状态字典中读取查询、键和值，并添加到指定位置的新键名下
def read_in_q_k_v(state_dict, config, prefix=""):
    # 获取隐藏层的大小
    dim = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        # 读取注意力机制中的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
        # 将查询、键和值的投影加入到状态字典中
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:dim, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:dim]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[dim:dim*2, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[dim:dim*2]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-dim:, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-dim:]


# 根据模型名称准备图像数据
def prepare_img(model_name):
    if "textvqa" in model_name:
        # 如果模型名称包含"textvqa"，则下载并打开示例图像文件
        filepath = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        image = Image.open(filepath).convert("RGB")
    else:
        # 否则，从指定的 URL 下载图像文件
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    return image


# 准备视频数据，使用decord库进行视频处理
def prepare_video():
    from decord import VideoReader, cpu

    # 设置随机数种子以保证可重现性
    np.random.seed(0)

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.

        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.

        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        # 计算需要采样的帧的数量
        converted_len = int(clip_len * frame_sample_rate)
        # 在视频长度内随机选择结束帧索引
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        # 生成均匀分布的帧索引列表，并限制在视频长度范围内
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    # 从指定的 HF Hub 仓库下载视频数据集中的特定文件，此处下载的文件是 "eating_spaghetti.mp4"
    file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
    
    # 使用 VideoReader 类读取视频文件，设置线程数为 1，在 CPU 0 上执行
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    
    # 将视频读取器定位到视频的起始位置
    videoreader.seek(0)
    
    # 通过 sample_frame_indices 函数从视频中随机抽取 6 个帧的索引
    # clip_len=6 表示要抽取 6 个帧
    # frame_sample_rate=4 表示每隔 4 个帧抽取一次
    # seg_len=len(videoreader) 获取视频的总帧数，作为抽取帧的范围
    indices = sample_frame_indices(clip_len=6, frame_sample_rate=4, seg_len=len(videoreader))
    
    # 从 videoreader 中获取指定 indices 的帧数据，返回一个 numpy 数组
    video = videoreader.get_batch(indices).asnumpy()
    
    # 返回抽取的视频帧数据
    return video
# 声明一个装饰器，用于指示在函数执行过程中不需要计算梯度
@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """

    # 定义不同模型名称对应的预训练模型下载链接
    model_name_to_url = {
        "git-base": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt",
        "git-base-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_COCO/snapshot/model.pt",
        "git-base-textcaps": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTCAPS/snapshot/model.pt",
        "git-base-vqav2": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VQAv2/snapshot/model.pt",
        "git-base-textvqa": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTVQA/snapshot/model.pt",  # todo
        "git-base-vatex": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VATEX/snapshot/model.pt",
        "git-base-msrvtt-qa": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_MSRVTT_QA/snapshot/model.pt"
        ),
        "git-large": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE/snapshot/model.pt",
        "git-large-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_COCO/snapshot/model.pt",
        "git-large-textcaps": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTCAPS/snapshot/model.pt"
        ),
        "git-large-vqav2": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VQAv2/snapshot/model.pt",
        "git-large-textvqa": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTVQA/snapshot/model.pt",
        "git-large-vatex": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VATEX/snapshot/model.pt",
        "git-large-msrvtt-qa": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_MSRVTT_QA/snapshot/model.pt"
        ),
        "git-large-r": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R/snapshot/model.pt",
        "git-large-r-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_COCO/snapshot/model.pt",
        "git-large-r-textcaps": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_TEXTCAPS/snapshot/model.pt"
        ),
    }

    # 定义不同模型名称对应的本地路径
    model_name_to_path = {
        "git-large": "/Users/nielsrogge/Documents/GIT/git_large_model.pt",
        "git-large-coco": "/Users/nielsrogge/Documents/GIT/git_large_coco_model.pt",
        "git-large-textcaps": "/Users/nielsrogge/Documents/GIT/git_large_textcaps_model.pt",
        "git-large-vqav2": "/Users/nielsrogge/Documents/GIT/git_large_vqav2_model.pt",
        "git-large-textvqa": "/Users/nielsrogge/Documents/GIT/git_large_textvqa_model.pt",
    }

    # 根据模型名称获取相应的 GIT 配置，图像尺寸和是否为视频
    config, image_size, is_video = get_git_config(model_name)
    # 检查模型名称中是否包含"large"，且不是视频模型，且不是"large-r"模型
    if "large" in model_name and not is_video and "large-r" not in model_name:
        # 如果是大模型，从本地加载预训练权重
        checkpoint_path = model_name_to_path[model_name]
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    else:
        # 否则，从指定的 URL 加载预训练权重
        checkpoint_url = model_name_to_url[model_name]
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)["model"]

    # 根据模型名称确定键名前缀是否为"module."
    prefix = "module." if model_name == "git-base" else ""
    # 创建重命名键名的映射列表
    rename_keys = create_rename_keys(config, prefix=prefix)
    # 对预训练权重中的键名进行重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取输入、查询和值的权重
    read_in_q_k_v(state_dict, config, prefix=prefix)

    # 加载 HuggingFace 模型
    model = GitForCausalLM(config)
    # 加载模型权重，允许缺少键名和不期待的键名
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # 断言确实缺少的键名和意外的键名
    assert missing_keys == ["git.embeddings.position_ids", "git.image_encoder.vision_model.embeddings.position_ids"]
    assert unexpected_keys == ["git.image_encoder.visual_projection.weight"]

    # 验证处理结果
    # 根据是否为视频选择不同的图像处理器
    image_processor = (
        VideoMAEImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
        if is_video
        else CLIPImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
    )
    # 根据模型类型选择适当的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased", model_input_names=["input_ids", "attention_mask"]
    )
    # 创建 GitProcessor 对象，用于处理文本和图像输入
    processor = GitProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if is_video:
        # 准备视频并处理像素值
        video = prepare_video()
        pixel_values = processor(images=list(video), return_tensors="pt").pixel_values
    else:
        # 准备图像并进行图像转换
        image = prepare_img(model_name)
        image_transforms = Compose(
            [
                Resize(image_size, interpolation=Image.BICUBIC),
                CenterCrop(image_size),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        # 对原始图像应用转换并获取像素值张量
        original_pixel_values = image_transforms(image).unsqueeze(0)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # 断言处理后的像素值与原始像素值接近
        assert torch.allclose(pixel_values, original_pixel_values)

    # 创建输入张量
    input_ids = torch.tensor([[101]])
    # 使用模型生成输出
    outputs = model(input_ids, pixel_values=pixel_values)
    logits = outputs.logits
    print("Logits:", logits[0, -1, :3])

    # 根据模型名称选择预期的切片 logits
    if model_name == "git-base":
        expected_slice_logits = torch.tensor([-1.2832, -1.2835, -1.2840])
    elif model_name == "git-base-coco":
        expected_slice_logits = torch.tensor([-0.9925, -0.9930, -0.9935])
    # 如果模型名称为 "git-base-textcaps"，设置预期的输出 logits
    elif model_name == "git-base-textcaps":
        expected_slice_logits = torch.tensor([-1.2980, -1.2983, -1.2985])
    # 如果模型名称为 "git-base-vqav2"，设置预期的输出 logits
    elif model_name == "git-base-vqav2":
        expected_slice_logits = torch.tensor([-0.8570, -0.8568, -0.8561])
    # 如果模型名称为 "git-base-textvqa"，设置预期的输出 logits
    elif model_name == "git-base-textvqa":
        expected_slice_logits = torch.tensor([-1.4085, -1.4083, -1.4082])
    # 如果模型名称为 "git-base-vatex"，设置预期的输出 logits
    elif model_name == "git-base-vatex":
        expected_slice_logits = torch.tensor([-1.3451, -1.3447, -1.3447])
    # 如果模型名称为 "git-base-msrvtt-qa"，设置预期的输出 logits
    elif model_name == "git-base-msrvtt-qa":
        expected_slice_logits = torch.tensor([-0.8554, -0.8550, -0.8540])
    # 如果模型名称为 "git-large"，设置预期的输出 logits
    elif model_name == "git-large":
        expected_slice_logits = torch.tensor([-1.1708, -1.1707, -1.1705])
    # 如果模型名称为 "git-large-coco"，设置预期的输出 logits
    elif model_name == "git-large-coco":
        expected_slice_logits = torch.tensor([-1.0425, -1.0423, -1.0422])
    # 如果模型名称为 "git-large-textcaps"，设置预期的输出 logits
    elif model_name == "git-large-textcaps":
        expected_slice_logits = torch.tensor([-1.2705, -1.2708, -1.2706])
    # 如果模型名称为 "git-large-vqav2"，设置预期的输出 logits
    elif model_name == "git-large-vqav2":
        expected_slice_logits = torch.tensor([-0.7042, -0.7043, -0.7043])
    # 如果模型名称为 "git-large-textvqa"，设置预期的输出 logits
    elif model_name == "git-large-textvqa":
        expected_slice_logits = torch.tensor([-0.8590, -0.8592, -0.8590])
    # 如果模型名称为 "git-large-vatex"，设置预期的输出 logits
    elif model_name == "git-large-vatex":
        expected_slice_logits = torch.tensor([-1.0113, -1.0114, -1.0113])
    # 如果模型名称为 "git-large-msrvtt-qa"，设置预期的输出 logits
    elif model_name == "git-large-msrvtt-qa":
        expected_slice_logits = torch.tensor([0.0130, 0.0134, 0.0131])
    # 如果模型名称为 "git-large-r"，设置预期的输出 logits
    elif model_name == "git-large-r":
        expected_slice_logits = torch.tensor([-1.1283, -1.1285, -1.1286])
    # 如果模型名称为 "git-large-r-coco"，设置预期的输出 logits
    elif model_name == "git-large-r-coco":
        expected_slice_logits = torch.tensor([-0.9641, -0.9641, -0.9641])
    # 如果模型名称为 "git-large-r-textcaps"，设置预期的输出 logits
    elif model_name == "git-large-r-textcaps":
        expected_slice_logits = torch.tensor([-1.1121, -1.1120, -1.1124])

    # 断言检查模型输出 logits 的正确性
    assert torch.allclose(logits[0, -1, :3], expected_slice_logits, atol=1e-4)
    # 输出提示信息
    print("Looks ok!")

    # 根据模型名称设置不同的提示语句
    prompt = ""
    if "textvqa" in model_name:
        prompt = "what does the front of the bus say at the top?"
    elif "msrvtt-qa" in model_name:
        prompt = "what does the woman eat?"
    elif "vqa" in model_name:
        prompt = "what are the cats doing?"

    # 使用分词器处理提示语句，生成输入的 token IDs
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    # 在 token IDs 前添加特殊 token 的 ID
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    # 将输入 token IDs 转换成张量并增加一个维度
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # 输出生成标题的提示信息
    print("Generating caption...")
    # 使用模型生成标题
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    # 打印生成的标题，跳过特殊 token 的解码
    print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 确保路径存在或创建路径
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 输出保存模型和处理器的信息
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)
    # 如果 push_to_hub 为 True，则执行以下操作
    if push_to_hub:
        # 打印推送模型和处理器到 hub 的信息，包括模型名称
        print(f"Pushing model and processor of {model_name} to the hub...")
        # 调用 model 对象的 push_to_hub 方法，将模型推送到 Microsoft 的 hub 中
        model.push_to_hub(f"microsoft/{model_name}")
        # 调用 processor 对象的 push_to_hub 方法，将处理器推送到 Microsoft 的 hub 中
        processor.push_to_hub(f"microsoft/{model_name}")
if __name__ == "__main__":
    # 如果作为主程序运行，执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="git-base",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    # 添加一个必需的参数：模型的名称，如果未提供则默认为"git-base"，类型为字符串

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个参数：PyTorch 模型输出目录的路径，如果未提供则为None，默认类型为字符串

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )
    # 添加一个参数：是否将模型推送到 hub 上，采用布尔标志方式

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 对象中

    convert_git_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_git_checkpoint，传入解析后的参数：模型名称、输出目录路径、是否推送到 hub
```