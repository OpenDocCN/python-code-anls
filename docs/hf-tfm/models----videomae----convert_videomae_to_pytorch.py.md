# `.\models\videomae\convert_videomae_to_pytorch.py`

```py
# coding=utf-8
# 声明文件编码格式为 UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明

# Licensed under the Apache License, Version 2.0 (the "License");
# 依据 Apache License, Version 2.0 授权许可

# you may not use this file except in compliance with the License.
# 除非符合 Apache License, Version 2.0 的授权许可，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 在适用法律要求或书面同意的情况下，依据“原样”提供，软件分发

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论是明示还是暗示的保证或条件

# See the License for the specific language governing permissions and
# 详细了解许可证可参阅特定的语言和权限
# limitations under the License.
# 在许可证下的限制

"""Convert VideoMAE checkpoints from the original repository: https://github.com/MCG-NJU/VideoMAE"""
# 文档字符串，指明代码用途是将 VideoMAE 检查点从原始仓库转换过来

import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 数据处理模块

import gdown  # 导入 gdown 用于下载工具
import numpy as np  # 导入 NumPy 模块
import torch  # 导入 PyTorch 模块
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 导入模型下载函数

from transformers import (  # 导入 transformers 模块中的多个类
    VideoMAEConfig,  # VideoMAE 模型配置类
    VideoMAEForPreTraining,  # 用于预训练的 VideoMAE 模型类
    VideoMAEForVideoClassification,  # 用于视频分类的 VideoMAE 模型类
    VideoMAEImageProcessor,  # VideoMAE 图像处理器类
)


def get_videomae_config(model_name):
    # 获取 VideoMAE 模型配置的函数定义，参数为模型名称
    config = VideoMAEConfig()  # 创建 VideoMAEConfig 实例

    set_architecture_configs(model_name, config)  # 调用设置架构配置的函数

    if "finetuned" not in model_name:
        # 如果模型名称中不包含 "finetuned"
        config.use_mean_pooling = False  # 禁用平均池化

    if "finetuned" in model_name:
        # 如果模型名称中包含 "finetuned"
        repo_id = "huggingface/label-files"  # 设置仓库 ID
        if "kinetics" in model_name:
            # 如果模型名称中包含 "kinetics"
            config.num_labels = 400  # 设置标签数量为 400
            filename = "kinetics400-id2label.json"  # 设置文件名
        elif "ssv2" in model_name:
            # 如果模型名称中包含 "ssv2"
            config.num_labels = 174  # 设置标签数量为 174
            filename = "something-something-v2-id2label.json"  # 设置文件名
        else:
            # 如果模型名称既不包含 "kinetics" 也不包含 "ssv2"
            raise ValueError("Model name should either contain 'kinetics' or 'ssv2' in case it's fine-tuned.")
            # 抛出数值错误，要求模型名称中应包含 'kinetics' 或 'ssv2'，以表明其是否进行了微调
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 使用 huggingface_hub 下载并加载标签文件内容到 id2label 字典中
        id2label = {int(k): v for k, v in id2label.items()}  # 将键转换为整数类型
        config.id2label = id2label  # 设置配置对象的 id2label 属性
        config.label2id = {v: k for k, v in id2label.items()}  # 设置配置对象的 label2id 属性

    return config  # 返回配置对象


def set_architecture_configs(model_name, config):
    # 设置架构配置的函数定义，参数为模型名称和配置对象
    if "small" in model_name:
        # 如果模型名称中包含 "small"
        config.hidden_size = 384  # 设置隐藏层大小为 384
        config.intermediate_size = 1536  # 设置中间层大小为 1536
        config.num_hidden_layers = 12  # 设置隐藏层层数为 12
        config.num_attention_heads = 16  # 设置注意力头数为 16
        config.decoder_num_hidden_layers = 12  # 设置解码器隐藏层层数为 12
        config.decoder_num_attention_heads = 3  # 设置解码器注意力头数为 3
        config.decoder_hidden_size = 192  # 设置解码器隐藏层大小为 192
        config.decoder_intermediate_size = 768  # 设置解码器中间层大小为 768
    elif "large" in model_name:
        # 如果模型名称中包含 "large"
        config.hidden_size = 1024  # 设置隐藏层大小为 1024
        config.intermediate_size = 4096  # 设置中间层大小为 4096
        config.num_hidden_layers = 24  # 设置隐藏层层数为 24
        config.num_attention_heads = 16  # 设置注意力头数为 16
        config.decoder_num_hidden_layers = 12  # 设置解码器隐藏层层数为 12
        config.decoder_num_attention_heads = 8  # 设置解码器注意力头数为 8
        config.decoder_hidden_size = 512  # 设置解码器隐藏层大小为 512
        config.decoder_intermediate_size = 2048  # 设置解码器中间层大小为 2048
    # 如果模型名中包含 "huge"
    elif "huge" in model_name:
        # 设置隐藏层大小为 1280
        config.hidden_size = 1280
        # 设置中间层大小为 5120
        config.intermediate_size = 5120
        # 设置隐藏层的数量为 32
        config.num_hidden_layers = 32
        # 设置注意力头的数量为 16
        config.num_attention_heads = 16
        # 设置解码器隐藏层的数量为 12
        config.decoder_num_hidden_layers = 12
        # 设置解码器注意力头的数量为 8
        config.decoder_num_attention_heads = 8
        # 设置解码器隐藏层大小为 640
        config.decoder_hidden_size = 640
        # 设置解码器中间层大小为 2560
        config.decoder_intermediate_size = 2560
    # 如果模型名中不包含 "base"
    elif "base" not in model_name:
        # 抛出数值错误，提示模型名应包含 "small", "base", "large", 或 "huge"
        raise ValueError('Model name should include either "small", "base", "large", or "huge"')
# 定义一个函数用于重命名给定的键名
def rename_key(name):
    # 如果键名中包含 "encoder."，则替换为空字符串
    if "encoder." in name:
        name = name.replace("encoder.", "")
    # 如果键名中包含 "cls_token"，则替换为 "videomae.embeddings.cls_token"
    if "cls_token" in name:
        name = name.replace("cls_token", "videomae.embeddings.cls_token")
    # 如果键名中包含 "decoder_pos_embed"，则替换为 "decoder.decoder_pos_embed"
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    # 如果键名中包含 "pos_embed" 且不包含 "decoder"，则替换为 "videomae.embeddings.position_embeddings"
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "videomae.embeddings.position_embeddings")
    # 如果键名中包含 "patch_embed.proj"，则替换为 "videomae.embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "videomae.embeddings.patch_embeddings.projection")
    # 如果键名中包含 "patch_embed.norm"，则替换为 "videomae.embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "videomae.embeddings.norm")
    # 如果键名中包含 "decoder.blocks"，则替换为 "decoder.decoder_layers"
    if "decoder.blocks" in name:
        name = name.replace("decoder.blocks", "decoder.decoder_layers")
    # 如果键名中包含 "blocks"，则替换为 "videomae.encoder.layer"
    if "blocks" in name:
        name = name.replace("blocks", "videomae.encoder.layer")
    # 如果键名中包含 "attn.proj"，则替换为 "attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    # 如果键名中包含 "attn" 且不包含 "bias"，则替换为 "attention.self"
    if "attn" in name and "bias" not in name:
        name = name.replace("attn", "attention.self")
    # 如果键名中包含 "attn"，则替换为 "attention.attention"
    if "attn" in name:
        name = name.replace("attn", "attention.attention")
    # 如果键名中包含 "norm1"，则替换为 "layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果键名中包含 "norm2"，则替换为 "layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果键名中包含 "mlp.fc1"，则替换为 "intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果键名中包含 "mlp.fc2"，则替换为 "output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果键名中包含 "decoder_embed"，则替换为 "decoder.decoder_embed"
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    # 如果键名中包含 "decoder_norm"，则替换为 "decoder.decoder_norm"
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    # 如果键名中包含 "decoder_pred"，则替换为 "decoder.decoder_pred"
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    # 如果键名中包含 "norm.weight" 且不包含 "decoder" 和 "fc"，则替换为 "videomae.layernorm.weight"
    if "norm.weight" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.weight", "videomae.layernorm.weight")
    # 如果键名中包含 "norm.bias" 且不包含 "decoder" 和 "fc"，则替换为 "videomae.layernorm.bias"
    if "norm.bias" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.bias", "videomae.layernorm.bias")
    # 如果键名中包含 "head" 且不包含 "decoder"，则替换为 "classifier"
    if "head" in name and "decoder" not in name:
        name = name.replace("head", "classifier")

    # 返回处理后的键名
    return name
    # 遍历原始状态字典的键的副本
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键以"encoder."开头，则移除该前缀
        if key.startswith("encoder."):
            key = key.replace("encoder.", "")

        # 如果键中包含"qkv"
        if "qkv" in key:
            # 根据"."分割键
            key_split = key.split(".")
            # 如果键以"decoder.blocks"开头
            if key.startswith("decoder.blocks"):
                # 设置维度和层号
                dim = config.decoder_hidden_size
                layer_num = int(key_split[2])
                prefix = "decoder.decoder_layers."
                # 如果键包含"weight"
                if "weight" in key:
                    # 更新原始状态字典，替换成特定格式的键和对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                # 设置维度和层号
                dim = config.hidden_size
                layer_num = int(key_split[1])
                prefix = "videomae.encoder.layer."
                # 如果键包含"weight"
                if "weight" in key:
                    # 更新原始状态字典，替换成特定格式的键和对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
        else:
            # 对键进行重命名处理并更新原始状态字典
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict
# 我们将在吃意大利面视频上验证我们的结果
# 使用的帧索引：[164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    # 从指定的数据集仓库下载名为 'eating_spaghetti.npy' 的文件
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    # 加载.npy文件中的视频数据
    video = np.load(file)
    return list(video)


def convert_videomae_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    # 获取VideoMAE模型配置
    config = get_videomae_config(model_name)

    if "finetuned" in model_name:
        # 如果模型名中包含'finetuned'，则使用VideoMAEForVideoClassification进行初始化
        model = VideoMAEForVideoClassification(config)
    else:
        # 否则使用VideoMAEForPreTraining进行初始化
        model = VideoMAEForPreTraining(config)

    # 下载托管在Google Drive上的原始检查点
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    # 加载检查点文件并映射到CPU
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    else:
        state_dict = files["module"]
    # 转换检查点的状态字典
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载新状态字典到模型中
    model.load_state_dict(new_state_dict)
    # 设置模型为评估模式
    model.eval()

    # 使用图像处理器VideoMAEImageProcessor进行视频帧的预处理
    image_processor = VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    # 准备视频数据，转换为PyTorch张量列表
    video = prepare_video()
    inputs = image_processor(video, return_tensors="pt")

    # 如果模型名中不包含'finetuned'
    if "finetuned" not in model_name:
        # 从指定的数据集仓库下载名为 'bool_masked_pos.pt' 的本地文件
        local_path = hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
        # 加载本地文件到inputs字典中的 'bool_masked_pos' 键
        inputs["bool_masked_pos"] = torch.load(local_path)

    # 使用模型处理inputs，得到输出结果
    outputs = model(**inputs)
    logits = outputs.logits

    # 定义不同模型名称对应的预期输出形状和切片
    model_names = [
        "videomae-small-finetuned-kinetics",
        "videomae-small-finetuned-ssv2",
        # Kinetics-400检查点（short = 仅预训练800个周期，而不是1600个周期）
        "videomae-base-short",
        "videomae-base-short-finetuned-kinetics",
        "videomae-base",
        "videomae-base-finetuned-kinetics",
        "videomae-large",
        "videomae-large-finetuned-kinetics",
        "videomae-huge-finetuned-kinetics",
        # Something-Something-v2检查点（short = 仅预训练800个周期，而不是2400个周期）
        "videomae-base-short-ssv2",
        "videomae-base-short-finetuned-ssv2",
        "videomae-base-ssv2",
        "videomae-base-finetuned-ssv2",
    ]

    # 注意：logits使用的图像均值和标准差分别为[0.5, 0.5, 0.5]和[0.5, 0.5, 0.5]进行了测试
    if model_name == "videomae-small-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9291, -0.4061, -0.9307])
    elif model_name == "videomae-small-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.2671, -0.4689, -0.8235])
    elif model_name == "videomae-base":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7739, 0.7968, 0.7089], [0.6701, 0.7487, 0.6209], [0.4287, 0.5158, 0.4773]])
    elif model_name == "videomae-base-short":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7994, 0.9612, 0.8508], [0.7401, 0.8958, 0.8302], [0.5862, 0.7468, 0.7325]])
        # 对于这个模型，我们验证了归一化和非归一化目标的损失
        expected_loss = torch.tensor([0.5142]) if config.norm_pix_loss else torch.tensor([0.6469])
    elif model_name == "videomae-large":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7149, 0.7997, 0.6966], [0.6768, 0.7869, 0.6948], [0.5139, 0.6221, 0.5605]])
    elif model_name == "videomae-large-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.0771, 0.0011, -0.3625])
    elif model_name == "videomae-huge-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.2433, 0.1632, -0.4894])
    elif model_name == "videomae-base-short-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.6588, 0.0990, -0.2493])
    elif model_name == "videomae-base-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.3669, -0.0688, -0.2421])
    elif model_name == "videomae-base-short-ssv2":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.4712, 0.5296, 0.5786], [0.2278, 0.2729, 0.4026], [0.0352, 0.0730, 0.2506]])
    elif model_name == "videomae-base-short-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.0537, -0.1539, -0.3266])
    elif model_name == "videomae-base-ssv2":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.8131, 0.8727, 0.8546], [0.7366, 0.9377, 0.8870], [0.5935, 0.8874, 0.8564]])
    elif model_name == "videomae-base-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.1961, -0.8337, -0.6389])
    else:
        raise ValueError(f"Model name not supported. Should be one of {model_names}")

    # 验证输出的形状是否符合预期
    assert logits.shape == expected_shape
    # 如果模型名称包含“finetuned”，则验证前三个输出值是否接近预期切片值
    if "finetuned" in model_name:
        assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    else:
        print("Logits:", logits[0, :3, :3])
        assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)
    print("Logits ok!")

    # 如果适用，验证损失值
    if model_name == "videomae-base-short":
        loss = outputs.loss
        assert torch.allclose(loss, expected_loss, atol=1e-4)
        print("Loss ok!")

    # 如果指定了 PyTorch 模型保存路径，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
    # 如果 push_to_hub 为真，则执行下面的代码块
    if push_to_hub:
        # 打印信息：正在推送到hub...
        print("Pushing to the hub...")
        # 调用 model 对象的 push_to_hub 方法，将模型推送到指定的 hub
        model.push_to_hub(model_name, organization="nielsr")
if __name__ == "__main__":
    # 如果脚本直接运行而非被导入，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=aa3276eb-fb7e-482a-adec-dc7171df14c4",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    # 添加必需的参数：原始 PyTorch 检查点的下载链接

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/Users/nielsrogge/Documents/VideoMAE/Test",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加必需的参数：输出 PyTorch 模型的目录路径

    parser.add_argument("--model_name", default="videomae-base", type=str, help="Name of the model.")
    # 添加参数：模型的名称，默认为 "videomae-base"

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加参数：是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数并返回一个命名空间

    convert_videomae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
    # 调用函数 convert_videomae_checkpoint，传递解析后的参数进行模型检查点转换
```