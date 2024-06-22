# `.\transformers\models\timesformer\convert_timesformer_to_pytorch.py`

```py
# 设置脚本文件的编码格式为 UTF-8
# 版权声明
# 依据 Apache License, Version 2.0 许可证，除非符合许可证要求，否则您不得使用此文件
# 您可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，没有任何形式的保证或条件，无论是明示的还是暗示的
# 请参见许可证以获取特定语言规定的权限和限制

# 导入所需的库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 格式的数据

import gdown  # 用于从 Google Drive 下载文件
import numpy as np  # 用于进行数值计算
import torch  # 用于构建深度学习模型
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载文件

from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEImageProcessor  # 导入深度学习模型相关的类


# 获取 TimeSformer 模型的配置信息
def get_timesformer_config(model_name):
    # 创建一个 TimeSformer 的配置对象
    config = TimesformerConfig()

    # 如果模型名称中包含 "large"，则设置 num_frames 为 96
    if "large" in model_name:
        config.num_frames = 96

    # 如果模型名称中包含 "hr"，则设置 num_frames 为 16，image_size 为 448
    if "hr" in model_name:
        config.num_frames = 16
        config.image_size = 448

    # 根据模型名称设置不同的 num_labels 和加载对应的标签文件
    repo_id = "huggingface/label-files"
    if "k400" in model_name:
        config.num_labels = 400
        filename = "kinetics400-id2label.json"
    elif "k600" in model_name:
        config.num_labels = 600
        filename = "kinetics600-id2label.json"
    elif "ssv2" in model_name:
        config.num_labels = 174
        filename = "something-something-v2-id2label.json"
    else:
        raise ValueError("Model name should either contain 'k400', 'k600' or 'ssv2'.")

    # 加载 id 到 label 的映射关系，并转换为字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# 重命名模型中的 key
def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "timesformer.embeddings.cls_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "timesformer.embeddings.position_embeddings")
    if "time_embed" in name:
        name = name.replace("time_embed", "timesformer.embeddings.time_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "timesformer.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "timesformer.embeddings.norm")
    if "blocks" in name:
        name = name.replace("blocks", "timesformer.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name and "temporal" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name and "temporal" not in name:
        name = name.replace("attn", "attention.attention")
    # 如果文件名中包含 "temporal_norm1"，则替换为 "temporal_layernorm"
    if "temporal_norm1" in name:
        name = name.replace("temporal_norm1", "temporal_layernorm")
    # 如果文件名中包含 "temporal_attn.proj"，则替换为 "temporal_attention.output.dense"
    if "temporal_attn.proj" in name:
        name = name.replace("temporal_attn", "temporal_attention.output.dense")
    # 如果文件名中包含 "temporal_fc"，则替换为 "temporal_dense"
    if "temporal_fc" in name:
        name = name.replace("temporal_fc", "temporal_dense")
    # 如果文件名中包含 "norm1" 但不包含 "temporal"，则替换为 "layernorm_before"
    if "norm1" in name and "temporal" not in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果文件名中包含 "norm2"，则替换为 "layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果文件名中包含 "mlp.fc1"，则替换为 "intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果文件名中包含 "mlp.fc2"，则替换为 "output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果文件名中包含 "norm.weight"，且不包含 "fc" 和 "temporal"，则替换为 "timesformer.layernorm.weight"
    if "norm.weight" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.weight", "timesformer.layernorm.weight")
    # 如果文件名中包含 "norm.bias"，且不包含 "fc" 和 "temporal"，则替换为 "timesformer.layernorm.bias"
    if "norm.bias" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.bias", "timesformer.layernorm.bias")
    # 如果文件名中包含 "head"，则替换为 "classifier"
    if "head" in name:
        name = name.replace("head", "classifier")

    # 返回替换后的文件名
    return name
``` 
# 将给定的 state_dict 转换为适合特定模型配置的新 state_dict
def convert_state_dict(orig_state_dict, config):
    # 使用 .copy() 复制原始 state_dict 的键列表，以便在迭代时修改原始 state_dict
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值，从原始 state_dict 中删除该键
        val = orig_state_dict.pop(key)

        # 检查键是否以 "model." 开头，若是则去除该前缀
        if key.startswith("model."):
            key = key.replace("model.", "")

        # 检查键中是否包含 "qkv"，如果包含，则进行特定处理
        if "qkv" in key:
            # 拆分键名，提取层编号
            key_split = key.split(".")
            layer_num = int(key_split[1])
            prefix = "timesformer.encoder.layer."
            
            # 根据键中是否包含 "temporal" 构建后缀
            if "temporal" in key:
                postfix = ".temporal_attention.attention.qkv."
            else:
                postfix = ".attention.attention.qkv."

            # 根据键中是否包含 "weight" 决定新键的格式，并赋值
            if "weight" in key:
                orig_state_dict[f"{prefix}{layer_num}{postfix}weight"] = val
            else:
                orig_state_dict[f"{prefix}{layer_num}{postfix}bias"] = val
        else:
            # 对于其他键，应用重命名函数并更新 state_dict
            orig_state_dict[rename_key(key)] = val

    # 返回转换后的 state_dict
    return orig_state_dict


# 准备视频数据以供模型验证
# 我们将使用吃意大利面的视频进行验证
# 使用的帧索引：[164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    # 从 HF Hub 下载数据集，此处是吃意大利面视频的 numpy 数据
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    # 加载视频数据
    video = np.load(file)
    return list(video)


# 将 Timesformer 模型的检查点转换为 PyTorch 模型的权重
def convert_timesformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    # 获取特定模型配置
    config = get_timesformer_config(model_name)

    # 创建 Timesformer 模型
    model = TimesformerForVideoClassification(config)

    # 下载原始检查点，该检查点托管在 Google Drive 上
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    # 加载检查点文件并处理 state_dict
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    elif "module" in files:
        state_dict = files["module"]
    else:
        state_dict = files["model_state"]
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载模型权重
    model.load_state_dict(new_state_dict)
    model.eval()

    # 创建视频处理器，处理模型输入数据
    image_processor = VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    # 准备视频数据
    video = prepare_video()
    # 处理视频数据，准备模型输入
    inputs = image_processor(video[:8], return_tensors="pt")

    # 使用模型进行推理
    outputs = model(**inputs)
    logits = outputs.logits

    # 定义模型名称列表，包含了不同数据集训练的 Timesformer 模型名称
    model_names = [
        "timesformer-base-finetuned-k400",
        "timesformer-large-finetuned-k400",
        "timesformer-hr-finetuned-k400",
        "timesformer-base-finetuned-k600",
        "timesformer-large-finetuned-k600",
        "timesformer-hr-finetuned-k600",
        "timesformer-base-finetuned-ssv2",
        "timesformer-large-finetuned-ssv2",
        "timesformer-hr-finetuned-ssv2",
    ]

    # 注意：logits 使用图像均值和标准差为 [0.5, 0.5, 0.5] 的情况进行了测试
    # 如果模型名为指定的值，则设置期望形状和切片
    if model_name == "timesformer-base-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.3016, -0.7713, -0.4205])
    elif model_name == "timesformer-base-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([-0.7267, -0.7466, 3.2404])
    elif model_name == "timesformer-base-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.9059, 0.6433, -3.1457])
    elif model_name == "timesformer-large-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-hr-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9617, -3.7311, -3.7708])
    elif model_name == "timesformer-hr-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([2.5273, 0.7127, 1.8848])
    elif model_name == "timesformer-hr-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-3.6756, -0.7513, 0.7180])
    else:
        raise ValueError(f"Model name not supported. Should be one of {model_names}")
    
    # 校验logits
    assert logits.shape == expected_shape
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    print("Logits ok!")
    
    # 如果pytorch_dump_folder_path不为空，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
    
    # 如果push_to_hub为True，则推送到hub
    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"fcakyon/{model_name}")
# 检查当前模块是否作为主程序运行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=17yvuYp9L4mn-HpIcK5Zo6K3UoOy1kA5l&export=download",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="timesformer-base-finetuned-k400", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 TimesFormer 模型检查点转换为 PyTorch 模型
    convert_timesformer_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub
    )
```