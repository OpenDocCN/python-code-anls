# `.\transformers\models\clip\convert_clip_original_pytorch_to_hf.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“原样”分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的库
import argparse
import torch
from clip import load
from transformers import CLIPConfig, CLIPModel

# 复制注意力层
def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)
    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias
    hf_attn_layer.out_proj.weight = out_proj_weights
    hf_attn_layer.out_proj.bias = out_proj_bias

# 复制 MLP 层
def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)

# 复制线性层
def copy_linear(hf_linear, pt_linear):
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias

# 复制层
def copy_layer(hf_layer, pt_layer):
    # 复制层归一化
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)
    # 复制 MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)
    # 复制注意力层
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)

# 复制多个层
def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)

# 复制编码器
def copy_encoder(hf_encoder, pt_model):
    # 复制嵌入
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding
    # 复制层归一化
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)
    # 复制隐藏层
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)

# 复制文本模型和投影
def copy_text_model_and_projection(hf_model, pt_model):
    # 复制投影
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T
    # 复制文本编码器
    copy_encoder(hf_model.text_model, pt_model)

# 复制视觉模型和投影
def copy_vison_model_and_projection(hf_model, pt_model):
    # 复制投影
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T
    # 复制层归一化
    # 复制预层归一化层的权重和偏置到 PyTorch 模型的视觉模型的预归一化层
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    # 复制后层归一化层的权重和偏置到 PyTorch 模型的视觉模型的后归一化层
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)
    
    # 复制嵌入层的权重和偏置到 PyTorch 模型的视觉模型的嵌入层
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    # 复制类别嵌入到 PyTorch 模型的视觉模型的嵌入层
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    # 复制位置嵌入层的权重到 PyTorch 模型的视觉模型的嵌入层
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data
    
    # 复制编码器的层到 PyTorch 模型的视觉模型的 Transformer 模块的残差块
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)
# 导入 torch 库中的 no_grad 函数
@torch.no_grad()
# 定义函数 convert_clip_checkpoint，用于将模型的权重转换到 transformers 设计中
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则使用 CLIPConfig 类从预训练配置中加载配置
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    # 否则创建一个新的 CLIPConfig 对象，设置 projection_dim 为 512，text_config 和 vision_config 为空字典
    else:
        config = CLIPConfig(projection_dim=512, text_config={}, vision_config={})

    # 创建一个 CLIPModel 对象，并设置为评估模式
    hf_model = CLIPModel(config).eval()

    # 加载 PyTorch 模型和权重，设备为 CPU，jit 模式为 False
    pt_model, _ = load(checkpoint_path, device="cpu", jit=False)
    pt_model = pt_model.eval()

    # 复制文本模型和投影层
    copy_text_model_and_projection(hf_model, pt_model)
    # 复制视觉模型和投影层
    copy_vison_model_and_projection(hf_model, pt_model)
    # 将 hf_model 的 logit_scale 设置为 pt_model 的 logit_scale

    # 创建一个输入张量 input_ids，值为从 0 到 77，维度为 (1, 77)
    input_ids = torch.arange(0, 77).unsqueeze(0)
    # 创建一个随机张量 pixel_values，维度为 (1, 3, 224, 224)
    pixel_values = torch.randn(1, 3, 224, 224)

    # 使用 hf_model 进行推理，传入 input_ids 和 pixel_values，返回结果字典
    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    # 获取 hf_outputs 中的 logits_per_image 和 logits_per_text
    hf_logits_per_image = hf_outputs.logits_per_image
    hf_logits_per_text = hf_outputs.logits_per_text
    # 使用 pt_model 进行推理，传入 pixel_values 和 input_ids，返回 logits_per_image 和 logits_per_text
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

    # 断言 hf_logits_per_image 与 pt_logits_per_image 的值在 1e-3 的误差范围内相等
    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    # 断言 hf_logits_per_text 与 pt_logits_per_text 的值在 1e-3 的误差范围内相等

    # 将 hf_model 的权重保存到指定路径 pytorch_dump_folder_path
    hf_model.save_pretrained(pytorch_dump_folder_path)


# 如果当前脚本被直接执行，则解析命令行参数并调用 convert_clip_checkpoint 函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    # 调用 convert_clip_checkpoint 函数，传入命令行参数中的路径信息
    convert_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```