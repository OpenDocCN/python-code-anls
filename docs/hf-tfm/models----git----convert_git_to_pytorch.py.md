# `.\models\git\convert_git_to_pytorch.py`

```
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache License, Version 2.0 的规定，您可以在遵守许可证的前提下使用此文件
# 有关许可证的详细信息，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或在书面形式上同意，否则根据许可证分发的软件均在 "AS IS" 基础上分发，
# 不提供任何明示或暗示的保证或条件。有关特定语言的许可证，请参阅许可证
# 在许可证规定的限制下进行权限
"""从原始存储库中转换 GIT 检查点。

链接: https://github.com/microsoft/GenerativeImage2Text/tree/main"""

# 导入所需的软件包
import argparse
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GitConfig,
    GitForCausalLM,
    GitProcessor,
    GitVisionConfig,
    VideoMAEImageProcessor,
)
from transformers.utils import logging

# 设置日志显示级别
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 获取 git 配置信息
def get_git_config(model_name):
    if "base" in model_name and "vqa" in model_name:
        image_size = 480
    elif "large" in model_name and "vqa" in model_name:
        image_size = 420
    else:
        image_size = 224

    vision_config = GitVisionConfig(image_size=image_size)

    if "large" in model_name:
        vision_config.patch_size = 14
        vision_config.hidden_size = 1024
        vision_config.intermediate_size = 4096
        vision_config.num_hidden_layers = 24
        vision_config.num_attention_heads = 16

    is_video = "vatex" in model_name or "msrvtt" in model_name
    num_image_with_embedding = 6 if is_video else None
    config = GitConfig(vision_config=vision_config.to_dict(), num_image_with_embedding=num_image_with_embedding)

    return config, image_size, is_video

# 创建要重命名的所有键的列表（左侧是原始名称，右侧是我们的名称）
def create_rename_keys(config, prefix=""):
    rename_keys = []

    # 图像编码器
    # ftm: 关闭
    rename_keys.append((f"{prefix}image_encoder.class_embedding", "git.image_encoder.vision_model.embeddings.class_embedding"))
    rename_keys.append((f"{prefix}image_encoder.positional_embedding", "git.image_encoder.vision_model.embeddings.position_embedding.weight"))
    rename_keys.append((f"{prefix}image_encoder.conv1.weight", "git.image_encoder.vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append((f"{prefix}image_encoder.ln_pre.weight", "git.image_encoder.vision_model.pre_layrnorm.weight"))
    rename_keys.append((f"{prefix}image_encoder.ln_pre.bias", "git.image_encoder.vision_model.pre_layrnorm.bias"))
    rename_keys.append(
        (f"{prefix}image_encoder.ln_post.weight", "git.image_encoder.vision_model.post_layernorm.weight")
    )
    # 将图像编码器的ln_post.weight键重命名为git.image_encoder.vision_model.post_layernorm.weight，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}image_encoder.ln_post.bias", "git.image_encoder.vision_model.post_layernorm.bias"))
    # 将图像编码器的ln_post.bias键重命名为git.image_encoder.vision_model.post_layernorm.bias，并添加到rename_keys列表中
    # fmt: on
    rename_keys.append((f"{prefix}image_encoder.proj", "git.image_encoder.visual_projection.weight"))
    # 将图像编码器的proj键重命名为git.image_encoder.visual_projection.weight，并添加到rename_keys列表中

    # fmt: off
    for i in range(config.vision_config.num_hidden_layers):
        # 遍历图像编码器层：输出投影，2个前馈神经网络和2个layernorms
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        # 将图像编码器的transformer.resblocks.{i}.attn.out_proj.weight键重命名为git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        # 将图像编码器的transformer.resblocks.{i}.attn.out_proj.bias键重命名为git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 将图像编码器的transformer.resblocks.{i}.ln_1.weight键重命名为git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 将图像编码器的transformer.resblocks.{i}.ln_1.bias键重命名为git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 将图像编码器的transformer.resblocks.{i}.mlp.c_fc.weight键重命名为git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 将图像编码器的transformer.resblocks.{i}.mlp.c_fc.bias键重命名为git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 将图像编码器的transformer.resblocks.{i}.mlp.c_proj.weight键重命名为git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        # 将图像编码器的transformer.resblocks.{i}.mlp.c_proj.bias键重命名为git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 将图像编码器的transformer.resblocks.{i}.ln_2.weight键重命名为git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight，并添加到rename_keys列表中
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 将图像编码器的transformer.resblocks.{i}.ln_2.bias键重命名为git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias，并添加到rename_keys列表中
    # fmt: on

    # text decoder
    # fmt: off
    rename_keys.append((f"{prefix}textual.embedding.words.weight", "git.embeddings.word_embeddings.weight"))
    # 将文本解码器的textual.embedding.words.weight键重命名为git.embeddings.word_embeddings.weight，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}textual.embedding.positions.weight", "git.embeddings.position_embeddings.weight"))
    # 将文本解码器的textual.embedding.positions.weight键重命名为git.embeddings.position_embeddings.weight，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}textual.visual_projection.0.weight", "git.visual_projection.visual_projection.0.weight"))
    # 将文本解码器的textual.visual_projection.0.weight键重命名为git.visual_projection.visual_projection.0.weight，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}textual.visual_projection.0.bias", "git.visual_projection.visual_projection.0.bias"))
    # 将文本解码器的textual.visual_projection.0.bias键重命名为git.visual_projection.visual_projection.0.bias，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}textual.visual_projection.1.weight", "git.visual_projection.visual_projection.1.weight"))
    # 将文本解码器的textual.visual_projection.1.weight键重命名为git.visual_projection.visual_projection.1.weight，并添加到rename_keys列表中
    rename_keys.append((f"{prefix}textual.visual_projection.1.bias", "git.visual_projection.visual_projection.1.bias"))
    # 将文本解码器的textual.visual_projection.1.bias键重命名为git.visual_projection.visual_projection.1.bias，并添加到rename_keys列表中
    # 添加需要重命名的键值对到 rename_keys 列表中，用于后续替换模型参数的名称
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.weight", "git.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.bias", "git.embeddings.LayerNorm.bias"))
    rename_keys.append((f"{prefix}textual.output.weight", "output.weight"))
    rename_keys.append((f"{prefix}textual.output.bias", "output.bias"))
    # 遍历模型的每一个隐藏层，重命名对应的参数键名
    for i in range(config.num_hidden_layers):
        # 重命名注意力机制中的查询参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.weight", f"git.encoder.layer.{i}.attention.self.query.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.bias", f"git.encoder.layer.{i}.attention.self.query.bias"))
        # 重命名注意力机制中的键参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.weight", f"git.encoder.layer.{i}.attention.self.key.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.bias", f"git.encoder.layer.{i}.attention.self.key.bias"))
        # 重命名注意力机制中的值参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.weight", f"git.encoder.layer.{i}.attention.self.value.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.bias", f"git.encoder.layer.{i}.attention.self.value.bias"))
        # 重命名注意力机制中的输出参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.weight", f"git.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.bias", f"git.encoder.layer.{i}.attention.output.dense.bias"))
        # 重命名注意力机制中的输出层规范化参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.weight", f"git.encoder.layer.{i}.attention.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.bias", f"git.encoder.layer.{i}.attention.output.LayerNorm.bias"))
        # 重命名隐藏层前馈网络中的中间层参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.weight", f"git.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.bias", f"git.encoder.layer.{i}.intermediate.dense.bias"))
        # 重命名隐藏层前馈网络中的输出层参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.weight", f"git.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.bias", f"git.encoder.layer.{i}.output.dense.bias"))
        # 重命名隐藏层输出的规范化参数权重
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.weight", f"git.encoder.layer.{i}.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.bias", f"git.encoder.layer.{i}.output.LayerNorm.bias"))
    # 结束 fmt 格式化
    # fmt: on
    # 如果配置中指定了嵌入图像的数量，则重命名键以匹配特定格式
    if config.num_image_with_embedding is not None:
        # 将键重命名为匹配特定格式的新键
        rename_keys.append(("img_temperal_embedding.0", "git.img_temperal_embedding.0"))
        rename_keys.append(("img_temperal_embedding.1", "git.img_temperal_embedding.1"))
        rename_keys.append(("img_temperal_embedding.2", "git.img_temperal_embedding.2"))
        rename_keys.append(("img_temperal_embedding.3", "git.img_temperal_embedding.3"))
        rename_keys.append(("img_temperal_embedding.4", "git.img_temperal_embedding.4"))
        rename_keys.append(("img_temperal_embedding.5", "git.img_temperal_embedding.5"))
    # 返回重命名后的键列表
    return rename_keys
# 重命名字典中的键值对
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 如果新键中包含特定字符串，则将值的转置存入新键，否则存入原值
    dct[new] = val.T if "image_encoder.visual_projection" in new else val


# 从状态字典中读取查询、键和值
def read_in_q_k_v(state_dict, config, prefix=""):
    # 获取隐藏层的维度
    dim = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        # 读取输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
        # 将查询、键、值分别添加到状态字典中
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:dim, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:dim]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[dim : dim * 2, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[dim : dim * 2]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-dim:, :]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-dim:]


# 准备图像，根据模型名获取图像
def prepare_img(model_name):
    # 如果模型名中包含"textvqa"，则从数据集中获取图像，否则从URL中获取图像
    if "textvqa" in model_name:
        filepath = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        image = Image.open(filepath).convert("RGB")
    else:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    return image


# 准备视频
def prepare_video():
    from decord import VideoReader, cpu

    # 为了能够重现结果，设置随机种子
    np.random.seed(0)

    # 从视频中抽样一定数量的帧索引
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
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    # 视频片段包含300帧（30 FPS时长10秒）
    # 从指定的资源库下载视频文件，文件名为"eating_spaghetti.mp4"，资源库类型为数据集
    file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
    # 创建视频读取器对象，使用单个线程，运行在CPU 0上
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    # 从视频的起始位置开始读取帧
    videoreader.seek(0)
    # 从视频中抽样6个帧，帧采样率为4，视频长度为len(videoreader)，得到抽样的帧索引列表
    indices = sample_frame_indices(clip_len=6, frame_sample_rate=4, seg_len=len(videoreader))
    # 根据抽样的帧索引列表，从视频中获取对应的帧数据，转换为NumPy数组形式
    video = videoreader.get_batch(indices).asnumpy()

    # 返回视频数据
    return video
# 导入 torch 模块中的 no_grad 装饰器，用于禁用梯度计算
@torch.no_grad()
# 定义函数用于将模型的权重复制/粘贴/调整到我们的 GIT 结构中
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """

    # 模型名与其对应的 URL 映射关系
    model_name_to_url = {
        "git-base": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt",
        # 其他模型名与 URL 映射 ...
    }

    # 模型名与其对应的本地文件路径映射关系
    model_name_to_path = {
        "git-large": "/Users/nielsrogge/Documents/GIT/git_large_model.pt",
        # 其他模型名与本地文件路径映射 ...
    }

    # 根据模型名获取对应的 GIT 配置、图像大小和是否为视频
    config, image_size, is_video = get_git_config(model_name)
    # 如果模型名包含"large"且不是视频，并且不是"large-r"，则执行以下操作
    if "large" in model_name and not is_video and "large-r" not in model_name:
        # 打印提示信息，大型检查点下载时间过长
        checkpoint_path = model_name_to_path[model_name]
        # 使用torch.load加载检查点文件，并指定映射位置为"cpu"，提取其中的"model"数据
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 否则，执行以下操作
    else:
        # 从模型名称到URL的映射中获取检查点URL
        checkpoint_url = model_name_to_url[model_name]
        # 使用torch.hub.load_state_dict_from_url从URL加载状态字典，并指定映射位置为"cpu"，文件名为model_name，提取其中的"model"数据
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)["model"]
    # 如果模型名为"git-base"，则指定前缀为"module."
    prefix = "module." if model_name == "git-base" else ""
    # 创建重命名键列表
    rename_keys = create_rename_keys(config, prefix=prefix)
    # 对state_dict进行重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取Q、K、V值
    read_in_q_k_v(state_dict, config, prefix=prefix)
    
    # 加载HuggingFace模型
    # 使用GitForCausalLM类实例化模型
    model = GitForCausalLM(config)
    # 加载状态字典，并指定非严格模式，返回缺失键和意外键列表
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 设置模型为评估模式
    model.eval()
    
    # 打印缺失键和意外键
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    # 断言缺失键与指定列表相等
    assert missing_keys == ["git.embeddings.position_ids", "git.image_encoder.vision_model.embeddings.position_ids"]
    # 断言意外键与指定列表相等
    assert unexpected_keys == ["git.image_encoder.visual_projection.weight"]
    
    # 验证结果
    # 如果是视频，则实例化VideoMAEImageProcessor类，否则实例化CLIPImageProcessor类，并指定图像大小和裁剪大小
    image_processor = (
        VideoMAEImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
        if is_video
        else CLIPImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
    )
    # 从"bert-base-uncased"模型实例化标记器，指定输入名称列表
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_input_names=["input_ids", "attention_mask"])
    # 使用GitProcessor类实例化处理器，传入标记器和图像处理器
    processor = GitProcessor(tokenizer=tokenizer, image_processor=image_processor)
    
    # 如果是视频，则准备视频数据，并通过processor处理后获取像素值
    if is_video:
        video = prepare_video()
        pixel_values = processor(images=list(video), return_tensors="pt").pixel_values
    # 否则，准备图像数据，进行预处理后获取原始像素值
    else:
        image = prepare_img(model_name)
        image_transforms = Compose(
            [
                Resize(image_size, interpolation=Image.BICUBIC),
                CenterCrop(image_size),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        original_pixel_values = image_transforms(image).unsqueeze(0)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
        # 断言处理后的像素值与原始像素值接近
        assert torch.allclose(pixel_values, original_pixel_values)
    
    # 构建输入标识
    input_ids = torch.tensor([[101]])
    # 使用模型预测，并传入像素值
    outputs = model(input_ids, pixel_values=pixel_values)
    # 从输出中提取logits
    logits = outputs.logits
    # 打印logits的前三个值
    print("Logits:", logits[0, -1, :3])
    
    # 根据模型名设置预期的切片logits值
    if model_name == "git-base":
        expected_slice_logits = torch.tensor([-1.2832, -1.2835, -1.2840])
    elif model_name == "git-base-coco":
        expected_slice_logits = torch.tensor([-0.9925, -0.9930, -0.9935])
    elif model_name == "git-base-textcaps":
        expected_slice_logits = torch.tensor([-1.2980, -1.2983, -1.2985])
    # 根据模型名称选择对应的预期切片logits
    elif model_name == "git-base-vqav2":
        expected_slice_logits = torch.tensor([-0.8570, -0.8568, -0.8561])
    elif model_name == "git-base-textvqa":
        expected_slice_logits = torch.tensor([-1.4085, -1.4083, -1.4082])
    elif model_name == "git-base-vatex":
        expected_slice_logits = torch.tensor([-1.3451, -1.3447, -1.3447])
    elif model_name == "git-base-msrvtt-qa":
        expected_slice_logits = torch.tensor([-0.8554, -0.8550, -0.8540])
    elif model_name == "git-large":
        expected_slice_logits = torch.tensor([-1.1708, -1.1707, -1.1705])
    elif model_name == "git-large-coco":
        expected_slice_logits = torch.tensor([-1.0425, -1.0423, -1.0422])
    elif model_name == "git-large-textcaps":
        expected_slice_logits = torch.tensor([-1.2705, -1.2708, -1.2706])
    elif model_name == "git-large-vqav2":
        expected_slice_logits = torch.tensor([-0.7042, -0.7043, -0.7043])
    elif model_name == "git-large-textvqa":
        expected_slice_logits = torch.tensor([-0.8590, -0.8592, -0.8590])
    elif model_name == "git-large-vatex":
        expected_slice_logits = torch.tensor([-1.0113, -1.0114, -1.0113])
    elif model_name == "git-large-msrvtt-qa":
        expected_slice_logits = torch.tensor([0.0130, 0.0134, 0.0131])
    elif model_name == "git-large-r":
        expected_slice_logits = torch.tensor([-1.1283, -1.1285, -1.1286])
    elif model_name == "git-large-r-coco":
        expected_slice_logits = torch.tensor([-0.9641, -0.9641, -0.9641])
    elif model_name == "git-large-r-textcaps":
        expected_slice_logits = torch.tensor([-1.1121, -1.1120, -1.1124])

    # 验证预测的最后三个logits是否与预期的slice logits接近
    assert torch.allclose(logits[0, -1, :3], expected_slice_logits, atol=1e-4)
    # 打印检查消息
    print("Looks ok!")

    # 根据模型名称选择对应的提示语句
    prompt = ""
    if "textvqa" in model_name:
        prompt = "what does the front of the bus say at the top?"
    elif "msrvtt-qa" in model_name:
        prompt = "what does the woman eat?"
    elif "vqa" in model_name:
        prompt = "what are the cats doing?"
    # 使用分词器处理提示语句，添加特殊标记，生成input_ids
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # 打印生成字幕的消息
    print("Generating caption...")
    # 生成字幕
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    # 打印生成的字幕
    print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))

    # 如果指定了pytorch_dump_folder_path，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了push_to_hub，则将模型和处理器推送到hub
    if push_to_hub:
        print(f"Pushing model and processor of {model_name} to the hub...")
        model.push_to_hub(f"microsoft/{model_name}")
        processor.push_to_hub(f"microsoft/{model_name}")
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数：模型名称
    parser.add_argument(
        "--model_name",
        default="git-base",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    # 添加必需参数：输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加可选参数：是否将模型推送到 Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数 convert_git_checkpoint，将给定的模型名称、PyTorch 模型目录路径和是否推送到 Hub 作为参数
    convert_git_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```