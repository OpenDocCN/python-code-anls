# `.\transformers\models\chinese_clip\convert_chinese_clip_original_pytorch_to_hf.py`

```py
# 导入所需模块
import argparse  # 解析命令行参数的模块
import torch  # PyTorch 深度学习框架
from transformers import ChineseCLIPConfig, ChineseCLIPModel  # 导入中文 CLIP 模型相关的配置和模型

# 复制自注意力层（self-attention layer）的权重到 HF（Hugging Face）模型
def copy_attn_layer(hf_attn_layer, pt_weights, prefix):
    # 将 PyTorch 权重拆分成查询（query）、键（key）和值（value）的投影矩阵
    q_proj, k_proj, v_proj = pt_weights[f"{prefix}.in_proj_weight"].chunk(3, dim=0)
    # 将 PyTorch 权重拆分成查询、键和值的偏置项
    q_proj_bias, k_proj_bias, v_proj_bias = pt_weights[f"{prefix}.in_proj_bias"].chunk(3, dim=0)

    # 获取输出投影层的权重和偏置项
    out_proj_weights = pt_weights[f"{prefix}.out_proj.weight"]
    out_proj_bias = pt_weights[f"{prefix}.out_proj.bias"]

    # 将查询投影矩阵赋值给 HF 模型中的查询投影矩阵
    hf_attn_layer.q_proj.weight.data = q_proj
    # 将查询偏置项赋值给 HF 模型中的查询偏置项
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    # 将键投影矩阵赋值给 HF 模型中的键投影矩阵
    hf_attn_layer.k_proj.weight.data = k_proj
    # 将键偏置项赋值给 HF 模型中的键偏置项
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    # 将值投影矩阵赋值给 HF 模型中的值投影矩阵
    hf_attn_layer.v_proj.weight.data = v_proj
    # 将值偏置项赋值给 HF 模型中的值偏置项
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    # 将输出投影层的权重赋值给 HF 模型中的输出投影层的权重
    hf_attn_layer.out_proj.weight.data = out_proj_weights
    # 将输出投影层的偏置项赋值给 HF 模型中的输出投影层的偏置项
    hf_attn_layer.out_proj.bias.data = out_proj_bias

# 复制 MLP（多层感知机）层的权重到 HF 模型
def copy_mlp(hf_mlp, pt_weights, prefix):
    # 复制第一个全连接层的权重和偏置项
    copy_linear(hf_mlp.fc1, pt_weights, f"{prefix}.c_fc")
    # 复制第二个全连接层的权重和偏置项
    copy_linear(hf_mlp.fc2, pt_weights, f"{prefix}.c_proj")

# 复制线性层的权重到 HF 模型
def copy_linear(hf_linear, pt_weights, prefix):
    # 复制权重
    hf_linear.weight.data = pt_weights[f"{prefix}.weight"].data
    # 复制偏置项
    hf_linear.bias.data = pt_weights[f"{prefix}.bias"].data

# 复制一个层的权重到 HF 模型
def copy_layer(hf_layer, pt_weights, prefix):
    # 复制层归一化层的权重和偏置项
    copy_linear(hf_layer.layer_norm1, pt_weights, f"{prefix}.ln_1")
    copy_linear(hf_layer.layer_norm2, pt_weights, f"{prefix}.ln_2")

    # 复制 MLP 层的权重
    copy_mlp(hf_layer.mlp, pt_weights, f"{prefix}.mlp")

    # 复制自注意力层的权重
    copy_attn_layer(hf_layer.self_attn, pt_weights, f"{prefix}.attn")

# 复制多个层的权重到 HF 模型
def copy_layers(hf_layers, pt_weights, prefix):
    # 遍历所有层
    for layer_id, hf_layer in enumerate(hf_layers):
        # 复制单个层的权重
        copy_layer(hf_layer, pt_weights, f"{prefix}.{layer_id}")

# 复制文本模型和投影层的权重到 HF 模型
def copy_text_model_and_projection(hf_model, pt_weights):
    # 复制投影层的权重
    hf_model.text_projection.weight.data = pt_weights["text_projection"].data.T

    # 复制文本编码器的权重
    for name, param in hf_model.text_model.named_parameters():
        param.data = pt_weights[f"bert.{name}"].data

# 复制视觉模型和投影层的权重到 HF 模型
def copy_vision_model_and_projection(hf_model, pt_weights):
    # 复制投影层的权重
    hf_model.visual_projection.weight.data = pt_weights["visual.proj"].data.T

    # 复制层归一化层的权重
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_weights, "visual.ln_pre")
    # 复制线性层的权重到 HF 模型的视觉模型的后层归一化层
    copy_linear(hf_model.vision_model.post_layernorm, pt_weights, "visual.ln_post")

    # 复制嵌入层的权重
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_weights["visual.conv1.weight"].data
    hf_model.vision_model.embeddings.class_embedding.data = pt_weights["visual.class_embedding"].data
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_weights["visual.positional_embedding"].data

    # 复制编码器层的权重
    copy_layers(hf_model.vision_model.encoder.layers, pt_weights, "visual.transformer.resblocks")
# 使用torch.no_grad()装饰器，确保在模型推断时不会进行梯度计算
@torch.no_grad()
def convert_chinese_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    将模型的权重转换到transformers设计中。
    """

    # 断言config_path不为None，若为None则抛出异常，提示需要指定对应模型尺寸的ChineseCLIP模型配置文件
    assert config_path is not None, "Please specify the ChineseCLIP model config of the corresponding model size."
    # 从预训练模型配置文件中加载ChineseCLIP配置
    config = ChineseCLIPConfig.from_pretrained(config_path)

    # 创建ChineseCLIP模型并设置为评估模式
    hf_model = ChineseCLIPModel(config).eval()

    # 加载PyTorch格式的模型权重，指定CPU上的位置
    pt_weights = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    # 将模型权重的键值对的键名中去掉前缀"module."，如果有的话，并重新组成字典
    pt_weights = {(name[7:] if name.startswith("module.") else name): value for name, value in pt_weights.items()}

    # 复制文本模型及投影层
    copy_text_model_and_projection(hf_model, pt_weights)
    # 复制视觉模型及投影层
    copy_vision_model_and_projection(hf_model, pt_weights)
    # 设置ChineseCLIP模型的logit_scale属性为加载的模型权重中的"logit_scale"属性的数据
    hf_model.logit_scale.data = pt_weights["logit_scale"].data

    # 将转换后的ChineseCLIP模型保存到指定路径
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定转换后的hf PyTorch模型的输出文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output folder storing converted hf PyTorch model.",
    )
    # 添加命令行参数，指定原始github格式的ChineseCLIP检查点文件路径
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to original github format ChineseCLIP checkpoint."
    )
    # 添加命令行参数，指定要转换的模型的hf配置文件config.json路径
    parser.add_argument(
        "--config_path", default=None, required=True, type=str, help="Path to hf config.json of model to convert."
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用convert_chinese_clip_checkpoint函数进行ChineseCLIP模型的转换
    convert_chinese_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
    # 打印转换完成的提示信息
    print("The conversion is finished!")
```