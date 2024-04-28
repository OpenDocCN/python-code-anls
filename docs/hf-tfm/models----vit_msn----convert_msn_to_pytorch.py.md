# `.\transformers\models\vit_msn\convert_msn_to_pytorch.py`

```
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache License, Version 2.0 许可证使用此文件
# 如果不遵守许可证，则不得使用此文件
# 可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得根据许可证分发软件
# 根据许可证的具体语言，以“AS IS”基础分发软件，无论是否有担保或条件，明示或暗示
# 请参阅许可证以了解权限和限制
"""从原始存储库：https://github.com/facebookresearch/msn 转换 ViT MSN 检查点"""

# 导入必要的包
import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# 禁用梯度
torch.set_grad_enabled(False)

# 列出需要重命名的所有键（原始名称在左边，我们的名称在右边）
def create_rename_keys(config, base_model=False):
    # 初始化需要重命名的键列表
    rename_keys = []
    # 遍历隐藏层的数量
    for i in range(config.num_hidden_layers):
        # 编码器层：输出投影，2 个前馈神经网络和 2 个层归一化
        rename_keys.append((f"module.blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"module.blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"module.blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append((f"module.blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"module.blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # 投影层 + 位置嵌入
    rename_keys.extend(
        [
            ("module.cls_token", "vit.embeddings.cls_token"),
            ("module.patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("module.patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("module.pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )
    # 如果存在基础模型
    if base_model:
        # 添加layernorm和pooler的键到要重命名的键的列表中
        rename_keys.extend(
            [
                ("module.norm.weight", "layernorm.weight"),
                ("module.norm.bias", "layernorm.bias"),
            ]
        )
    
        # 如果只有基础模型，我们应该从所有以"vit"开头的键中移除"vit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 添加layernorm和分类头的键到要重命名的键的列表中
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )
    
    # 返回重命名后的键列表
    return rename_keys
# 将 encoder 层中的矩阵分裂成查询、键和值
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历所有 encoder 层
    for i in range(config.num_hidden_layers):
        # 如果是基础模型，前缀为空
        if base_model:
            prefix = ""
        # 否则前缀为 "vit."
        else:
            prefix = "vit."
        # 读取输入映射层的权重和偏置（在 timm 中，这是一个单一的矩阵加偏置）
        in_proj_weight = state_dict.pop(f"module.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"module.blocks.{i}.attn.qkv.bias")
        # 接下来将查询、键和值（按此顺序）添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 移除分类头
def remove_classification_head_(state_dict):
    # 需要忽略的键
    ignore_keys = ["head.weight", "head.bias"]
    # 从状态字典中删除这些键
    for k in ignore_keys:
        state_dict.pop(k, None)


# 移除预测头
def remove_projection_head(state_dict):
    # 预测头在 MSN 的自监督预训练中使用，
    # 对于下游任务不需要
    ignore_keys = [
        "module.fc.fc1.weight",
        "module.fc.fc1.bias",
        "module.fc.bn1.weight",
        "module.fc.bn1.bias",
        "module.fc.bn1.running_mean",
        "module.fc.bn1.running_var",
        "module.fc.bn1.num_batches_tracked",
        "module.fc.fc2.weight",
        "module.fc.fc2.bias",
        "module.fc.bn2.weight",
        "module.fc.bn2.bias",
        "module.fc.bn2.running_mean",
        "module.fc.bn2.running_var",
        "module.fc.bn2.num_batches_tracked",
        "module.fc.fc3.weight",
        "module.fc.fc3.bias",
    ]
    # 从状态字典中删除这些键
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名键
def rename_key(dct, old, new):
    # 获取旧键的值，并将其用新键存入字典
    val = dct.pop(old)
    dct[new] = val


# 转换 ViT-MSN 检查点
def convert_vit_msn_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    # 创建 ViTMSNConfig 对象
    config = ViTMSNConfig()
    config.num_labels = 1000

    # 从 HuggingFace Hub 下载 ImageNet-1K 标签文件
    repo_id = "datasets/huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    # 如果检查点 URL 中包含 "s16"，则设置配置的隐藏大小为 384，中间大小为 1536，注意力头数为 6
    if "s16" in checkpoint_url:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_attention_heads = 6
    # 如果检查点 URL 中包含 "l16"，则设置配置的隐藏大小为 1024，中间大小为 4096，隐藏层数为 24，注意力头数为 16，隐藏层丢失概率为 0.1
    elif "l16" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    # 如果检查点 URL 中包含 "b4"，则设置配置的补丁大小为 4
    elif "b4" in checkpoint_url:
        config.patch_size = 4
    # 如果检查点 URL 中包含 "l7"，则设置配置的补丁大小为 7，隐藏大小为 1024，中间大小为 4096，隐藏层数为 24，注意力头数为 16，隐藏层丢失概率为 0.1
    elif "l7" in checkpoint_url:
        config.patch_size = 7
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    
    # 创建 ViTMSNModel 模型，使用配置参数
    model = ViTMSNModel(config)
    
    # 从给定的 URL 加载状态字典，仅加载目标编码器，将其映射到 CPU 上
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["target_encoder"]
    
    # 创建 ViTImageProcessor 对象，设置图像大小为配置中指定的大小，同时使用默认的图像均值和标准差
    image_processor = ViTImageProcessor(size=config.image_size)
    
    # 移除投影头部
    remove_projection_head(state_dict)
    # 创建重命名键
    rename_keys = create_rename_keys(config, base_model=True)
    
    # 遍历重命名键，将状态字典中的键从源重命名为目标
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取 Q、K、V
    read_in_q_k_v(state_dict, config, base_model=True)
    
    # 加载状态字典到模型
    model.load_state_dict(state_dict)
    # 将模型设置为评估模式
    model.eval()
    
    # 设置图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # 使用请求获取图像流，并打开图像
    image = Image.open(requests.get(url, stream=True).raw)
    # 创建 ViTImageProcessor 对象，设置图像大小为配置中指定的大小，并使用默认的图像均值和标准差
    image_processor = ViTImageProcessor(
        size=config.image_size, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD
    )
    # 使用图像处理器处理图像，返回 PyTorch 张量
    inputs = image_processor(images=image, return_tensors="pt")
    
    # 正向传播
    # 设置随机种子
    torch.manual_seed(2)
    # 将输入传递给模型，得到输出
    outputs = model(**inputs)
    # 获取最后一个隐藏状态
    last_hidden_state = outputs.last_hidden_state
    
    # 下面的 Colab 笔记本用于生成这些输出：
    # https://colab.research.google.com/gist/sayakpaul/3672419a04f5997827503fd84079bdd1/scratchpad.ipynb
    # 如果检查点 URL 中包含 "s16"，则预期的切片是特定的张量
    if "s16" in checkpoint_url:
        expected_slice = torch.tensor([[-1.0915, -1.4876, -1.1809]])
    # 如果检查点 URL 中包含 "b16"，则预期的切片是特定的张量
    elif "b16" in checkpoint_url:
        expected_slice = torch.tensor([[14.2889, -18.9045, 11.7281]])
    # 如果检查点 URL 中包含 "l16"，则预期的切片是特定的张量
    elif "l16" in checkpoint_url:
        expected_slice = torch.tensor([[41.5028, -22.8681, 45.6475]])
    # 如果检查点 URL 中包含 "b4"，则预期的切片是特定的张量
    elif "b4" in checkpoint_url:
        expected_slice = torch.tensor([[-4.3868, 5.2932, -0.4137]])
    # 如果上述情况都不满足，则预期的切片是特定的张量
    else:
        expected_slice = torch.tensor([[-0.1792, -0.6465, 2.4263]])
    
    # 验证 logits，确保最后一个隐藏状态的前三个元素与预期切片在给定的容差范围内匹配
    assert torch.allclose(last_hidden_state[:, 0, :3], expected_slice, atol=1e-4)
    
    # 打印保存模型的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    
    # 打印保存图像处理器的路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    # 添加必需的参数
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 从命令行读取参数
    args = parser.parse_args()
    # 调用函数，将参数传递给函数
    convert_vit_msn_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```