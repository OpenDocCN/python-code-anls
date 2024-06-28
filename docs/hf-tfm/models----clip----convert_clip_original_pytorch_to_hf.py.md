# `.\models\clip\convert_clip_original_pytorch_to_hf.py`

```
# 引入 argparse 模块，用于处理命令行参数
import argparse

# 引入 PyTorch 库
import torch
# 从 clip 模块中导入 load 函数
from clip import load
# 从 transformers 库中导入 CLIPConfig 和 CLIPModel 类
from transformers import CLIPConfig, CLIPModel


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    # 将 pt_attn_layer.in_proj_weight 按行分割成 q_proj, k_proj, v_proj 三部分
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    # 将 pt_attn_layer.in_proj_bias 按行分割成 q_proj_bias, k_proj_bias, v_proj_bias 三部分
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    # 设置 hf_attn_layer 的权重和偏置
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    # 设置 hf_attn_layer 的输出投影权重和偏置
    hf_attn_layer.out_proj.weight = pt_attn_layer.out_proj.weight
    hf_attn_layer.out_proj.bias = pt_attn_layer.out_proj.bias


def copy_mlp(hf_mlp, pt_mlp):
    # 复制 pt_mlp 中的全连接层参数到 hf_mlp 中的 fc1 和 fc2 层
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    # 复制权重和偏置
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias


def copy_layer(hf_layer, pt_layer):
    # 复制层归一化
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # 复制 MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # 复制注意力层
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    # 遍历并复制每个层
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # 复制嵌入层权重
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # 复制最终层归一化
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # 复制隐藏层
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # 复制文本投影层
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T

    # 复制文本编码器
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model_and_projection(hf_model, pt_model):
    # 复制视觉投影层
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T

    # 复制层归一化
    copy_linear(hf_model.visual_model.layer_norm, pt_model.visual.ln)


# 以上是对给定代码的详细注释
    # 将 hf_model 的预层标准化层复制到 pt_model 的视觉模型的前标准化层
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)

    # 将 hf_model 的后层标准化层复制到 pt_model 的视觉模型的后标准化层
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # 复制嵌入层的权重
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    # 复制嵌入层的类别嵌入
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    # 复制嵌入层的位置编码权重
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data

    # 复制编码器的层
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)
@torch.no_grad()
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置路径，则从预训练模型加载配置
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    else:
        # 否则使用默认配置创建 CLIPConfig 对象，设置投影维度为512，文本和视觉配置为空字典
        config = CLIPConfig(projection_dim=512, text_config={}, vision_config={})

    # 创建并设置为评估模式的 HF 模型对象
    hf_model = CLIPModel(config).eval()

    # 加载 PyTorch 模型，返回模型和其他元数据
    pt_model, _ = load(checkpoint_path, device="cpu", jit=False)
    # 将 PyTorch 模型设置为评估模式
    pt_model = pt_model.eval()

    # 复制文本模型和投影
    copy_text_model_and_projection(hf_model, pt_model)
    # 复制视觉模型和投影
    copy_vison_model_and_projection(hf_model, pt_model)
    # 将 HF 模型的 logit_scale 属性设置为与 PT 模型相同的值
    hf_model.logit_scale = pt_model.logit_scale

    # 创建输入的示例数据
    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    # 使用 HF 模型进行推理，返回结果字典
    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    # 提取 HF 模型的图像 logit
    hf_logits_per_image = hf_outputs.logits_per_image
    # 提取 HF 模型的文本 logit
    hf_logits_per_text = hf_outputs.logits_per_text
    # 使用 PT 模型进行推理，返回图像和文本 logit
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

    # 断言 HF 模型的图像 logit 与 PT 模型的图像 logit 接近（误差不超过1e-3）
    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    # 断言 HF 模型的文本 logit 与 PT 模型的文本 logit 接近（误差不超过1e-3）
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3)

    # 将转换后的 HF 模型保存到指定路径
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    # 调用转换函数，传入命令行参数
    convert_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```