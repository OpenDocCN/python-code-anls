# `.\transformers\models\clvp\convert_clvp_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明
"""
Weights conversion script for CLVP
"""

# 导入所需模块
import argparse
import os

import torch
from huggingface_hub import hf_hub_download

from transformers import ClvpConfig, ClvpModelForConditionalGeneration

# 定义模型的下载链接
_MODELS = {
    "clvp": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth",
    "decoder": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/autoregressive.pth",
}

# 设置维度和子维度
dim = 1024
sub_dim = dim // 16

# 定义 CLVP 编码器权重映射
CLVP_ENCODERS_MAPPING = {
    "text_transformer.transformer.attn_layers": "text_encoder_model",
    "speech_transformer.transformer.attn_layers": "speech_encoder_model",
    "text_transformer.transformer.norm": "text_encoder_model.final_layer_norm",
    "speech_transformer.transformer.norm": "speech_encoder_model.final_layer_norm",
    "to_text_latent": "text_encoder_model.projection",
    "to_speech_latent": "speech_encoder_model.projection",
    "text_emb": "text_encoder_model.token_embedding",
    "speech_emb": "speech_encoder_model.token_embedding",
    "1.wrap.net.0": "mlp.fc1",
    "1.wrap.net.3": "mlp.fc2",
    "1.wrap": "self_attn",
    "to_out": "out_proj",
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "temperature": "logit_scale",
}

# 定义 CLVP 解码器权重映射
CLVP_DECODER_MAPPING = {
    "conditioning_encoder.init": "conditioning_encoder.mel_conv",
    "conditioning_encoder.attn": "conditioning_encoder.mel_attn_blocks",
    "mel_attn_blocks": "group_norms",
    ".norm.weight": ".weight",
    ".norm.bias": ".bias",
    "text_embedding": "conditioning_encoder.text_token_embedding",
    "text_pos_embedding.emb": "conditioning_encoder.text_position_embedding",
    "final_norm": "speech_decoder_model.final_norm",
    "mel_head": "speech_decoder_model.lm_head",
    "gpt.ln_f": "speech_decoder_model.model.decoder.layer_norm",
    "mel_embedding": "speech_decoder_model.model.decoder.input_embeds_layer",
    "mel_pos_embedding.emb": "speech_decoder_model.model.decoder.position_embeds_layer",
    "gpt.h": "speech_decoder_model.model.decoder.layers",
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
}

# 更新索引
def update_index(present_index):
    if present_index % 2 == 0:
        return int(present_index / 2)
    else:
        return int((present_index - 1) / 2)

# 转换编码器权重函数
def convert_encoder_weights(original_weights):
    converted_weights = {}
    # 获取原始权重字典的按键排序后的列表
    original_weights_keys = sorted(original_weights.keys())
    # 遍历原始权重字典的按键列表
    for original_key in original_weights_keys:
        # 将更新后的键初始化为原始键
        updated_key = original_key
        # 检查是否为 input_rmsnorm.weight 和 post_attention_rmsnorm.weight 的权重
        if "0.0.g" in updated_key:
            # 获取当前索引号
            present_index = updated_key.split(".")[4]
            # 如果索引号为偶数，更新键为 input_rmsnorm.weight
            if int(present_index) % 2 == 0:
                updated_key = updated_key.replace("0.0.g", "input_rmsnorm.weight")
            # 如果索引号为奇数，更新键为 post_attention_rmsnorm.weight
            else:
                updated_key = updated_key.replace("0.0.g", "post_attention_rmsnorm.weight")

        # 检查是否为 transformer.attn_layers.layers 的权重
        if "transformer.attn_layers.layers" in updated_key:
            # 获取当前索引号
            present_index = updated_key.split(".")[4]
            # 更新索引号
            updated_index = update_index(int(present_index))
            # 更新键中的索引号
            updated_key = updated_key.replace(
                f"transformer.attn_layers.layers.{present_index}", f"transformer.attn_layers.layers.{updated_index}"
            )

        # 替换键中的特定字符串，根据 CLVP_ENCODERS_MAPPING 字典进行映射
        for k, v in CLVP_ENCODERS_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)

        # 将更新后的键值对存入转换后的权重字典，并从原始权重字典中删除原始键值对
        converted_weights[updated_key] = original_weights.pop(original_key)

    # 返回转换后的权重字典
    return converted_weights
# 将原始权重转换为新的权重格式
def convert_decoder_weights(original_weights):
    # 创建一个空字典用于存储转换后的权重
    converted_weights = {}
    # 获取原始权重字典中的键，并按字母顺序排序
    original_weights_keys = sorted(original_weights.keys())
    # 返回转换后的权重字典
    return converted_weights


# 下载指定 URL 的文件到指定目录
def _download(url: str, root: str):
    # 从 URL 中提取仓库 ID 和文件名
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-2]}/{url.split('/')[-1]}"
    # 使用 Hugging Face Hub 下载文件到指定路径
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


# 将 CLVP 模型的检查点权重转换为 PyTorch 格式
def convert_clvp_weights(checkpoint_path, pytorch_dump_folder_path):
    # 创建一个空字典用于存储转换后的检查点权重
    converted_checkpoint = {}

    # 遍历所有模型及其对应的 URL
    for each_model_name, each_model_url in _MODELS.items():
        # 构建每个模型对应的本地文件路径
        each_model_path = os.path.join(checkpoint_path, each_model_url.split("/")[-1])
        # 如果本地文件不存在，则下载对应模型文件
        if not os.path.exists(each_model_path):
            print(f"\n{each_model_name} was not found! Downloading it to {each_model_path}")
            _download(url=each_model_url, root=each_model_path)

        # 根据模型名加载对应的检查点文件
        if each_model_name == "clvp":
            clvp_checkpoint = torch.load(each_model_path, map_location="cpu")
        else:
            decoder_checkpoint = torch.load(each_model_path, map_location="cpu")

    # 将 CLVP 模型和解码器模型的权重转换为新的权重格式，并合并到转换后的检查点字典中
    converted_checkpoint.update(**convert_encoder_weights(clvp_checkpoint))
    converted_checkpoint.update(**convert_decoder_weights(decoder_checkpoint))

    # 从预训练配置创建 CLVP 模型配置
    config = ClvpConfig.from_pretrained("susnato/clvp_dev")
    # 使用 CLVP 模型配置创建 CLVP 模型实例
    model = ClvpModelForConditionalGeneration(config)

    # 加载转换后的检查点权重到 CLVP 模型
    model.load_state_dict(converted_checkpoint, strict=True)
    # 将模型保存为 PyTorch 模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved at {pytorch_dump_folder_path}!")


# 主程序入口
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数：检查点文件夹路径
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    # 添加可选参数：PyTorch 模型输出文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 转换 CLVP 模型的检查点权重为 PyTorch 格式
    convert_clvp_weights(args.checkpoint_path, args.pytorch_dump_folder_path)
```