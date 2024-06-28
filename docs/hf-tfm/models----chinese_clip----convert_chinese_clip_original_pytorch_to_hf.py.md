# `.\models\chinese_clip\convert_chinese_clip_original_pytorch_to_hf.py`

```
# 设置文件编码为 UTF-8，确保代码中的中文字符可以正确处理
# 版权声明和许可条款，指明代码的版权归属和使用许可
# 导入命令行参数解析模块
import argparse

# 导入 PyTorch 深度学习框架
import torch

# 导入 Transformers 库中的中文 CLIP 模型配置和模型类
from transformers import ChineseCLIPConfig, ChineseCLIPModel

# 定义函数：复制注意力层参数
def copy_attn_layer(hf_attn_layer, pt_weights, prefix):
    # 将 PyTorch 权重参数按照注意力头进行分块
    q_proj, k_proj, v_proj = pt_weights[f"{prefix}.in_proj_weight"].chunk(3, dim=0)
    # 分离偏置项
    q_proj_bias, k_proj_bias, v_proj_bias = pt_weights[f"{prefix}.in_proj_bias"].chunk(3, dim=0)

    # 复制权重和偏置到 HF 注意力层对象
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    # 复制输出投影层的权重和偏置
    out_proj_weights = pt_weights[f"{prefix}.out_proj.weight"]
    out_proj_bias = pt_weights[f"{prefix}.out_proj.bias"]
    hf_attn_layer.out_proj.weight.data = out_proj_weights
    hf_attn_layer.out_proj.bias.data = out_proj_bias

# 定义函数：复制 MLP 层参数
def copy_mlp(hf_mlp, pt_weights, prefix):
    # 复制线性变换层的权重和偏置
    copy_linear(hf_mlp.fc1, pt_weights, f"{prefix}.c_fc")
    copy_linear(hf_mlp.fc2, pt_weights, f"{prefix}.c_proj")

# 定义函数：复制线性变换层参数
def copy_linear(hf_linear, pt_weights, prefix):
    # 复制权重和偏置到 HF 线性层对象
    hf_linear.weight.data = pt_weights[f"{prefix}.weight"].data
    hf_linear.bias.data = pt_weights[f"{prefix}.bias"].data

# 定义函数：复制整个层的参数
def copy_layer(hf_layer, pt_weights, prefix):
    # 复制层归一化层的参数
    copy_linear(hf_layer.layer_norm1, pt_weights, f"{prefix}.ln_1")
    copy_linear(hf_layer.layer_norm2, pt_weights, f"{prefix}.ln_2")

    # 复制 MLP 层参数
    copy_mlp(hf_layer.mlp, pt_weights, f"{prefix}.mlp")

    # 复制注意力层参数
    copy_attn_layer(hf_layer.self_attn, pt_weights, f"{prefix}.attn")

# 定义函数：复制多层的参数
def copy_layers(hf_layers, pt_weights, prefix):
    # 遍历 HF 模型的每一层并复制参数
    for layer_id, hf_layer in enumerate(hf_layers):
        copy_layer(hf_layer, pt_weights, f"{prefix}.{layer_id}")

# 定义函数：复制文本模型和投影参数
def copy_text_model_and_projection(hf_model, pt_weights):
    # 复制文本投影层的权重，并转置数据
    hf_model.text_projection.weight.data = pt_weights["text_projection"].data.T

    # 遍历 HF 文本模型的每个参数并复制对应的 PyTorch 参数
    for name, param in hf_model.text_model.named_parameters():
        param.data = pt_weights[f"bert.{name}"].data

# 定义函数：复制视觉模型和投影参数
def copy_vision_model_and_projection(hf_model, pt_weights):
    # 复制视觉投影层的权重，并转置数据
    hf_model.visual_projection.weight.data = pt_weights["visual.proj"].data.T

    # 复制视觉模型的层归一化层参数
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_weights, "visual.ln_pre")
    # 将 PyTorch 模型的后层归一化权重复制到 HF 模型的视觉模型的后层归一化
    copy_linear(hf_model.vision_model.post_layernorm, pt_weights, "visual.ln_post")

    # 复制嵌入层权重
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_weights["visual.conv1.weight"].data
    hf_model.vision_model.embeddings.class_embedding.data = pt_weights["visual.class_embedding"].data
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_weights["visual.positional_embedding"].data

    # 复制编码器层
    copy_layers(hf_model.vision_model.encoder.layers, pt_weights, "visual.transformer.resblocks")
@torch.no_grad()
def convert_chinese_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    # 确保传入的配置路径不为空，用于加载对应模型大小的 ChineseCLIP 配置
    assert config_path is not None, "Please specify the ChineseCLIP model config of the corresponding model size."
    # 从预训练的配置文件中加载 ChineseCLIPConfig
    config = ChineseCLIPConfig.from_pretrained(config_path)

    # 创建 ChineseCLIPModel 实例并设置为评估模式
    hf_model = ChineseCLIPModel(config).eval()

    # 使用 torch.load 加载模型权重，指定在 CPU 上加载
    pt_weights = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    # 将模型权重字典中的键名处理为非多 GPU 情况下的模型名称格式
    pt_weights = {(name[7:] if name.startswith("module.") else name): value for name, value in pt_weights.items()}

    # 复制文本模型和投影层的权重到 hf_model
    copy_text_model_and_projection(hf_model, pt_weights)
    # 复制视觉模型和投影层的权重到 hf_model
    copy_vision_model_and_projection(hf_model, pt_weights)
    # 设置 hf_model 的 logit_scale 数据为 pt_weights 中的 logit_scale 数据
    hf_model.logit_scale.data = pt_weights["logit_scale"].data

    # 将转换后的模型保存到指定的 PyTorch 转储文件夹路径
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加输出的 PyTorch 模型文件夹路径参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output folder storing converted hf PyTorch model.",
    )
    # 添加原始 GitHub 格式 ChineseCLIP 检查点路径参数
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to original github format ChineseCLIP checkpoint."
    )
    # 添加必需的 hf 配置文件路径参数，用于模型转换
    parser.add_argument(
        "--config_path", default=None, required=True, type=str, help="Path to hf config.json of model to convert."
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行 ChineseCLIP 检查点的转换
    convert_chinese_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
    # 打印转换完成的提示信息
    print("The conversion is finished!")
```