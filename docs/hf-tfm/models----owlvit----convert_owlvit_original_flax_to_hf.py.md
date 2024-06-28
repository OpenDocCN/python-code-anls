# `.\models\owlvit\convert_owlvit_original_flax_to_hf.py`

```
# 设定脚本编码格式为UTF-8，确保支持中文等非ASCII字符
# 版权声明，声明该代码的版权归The HuggingFace Inc.团队所有，保留所有权利
#
# 根据Apache许可证2.0版，除非符合许可证的条款，否则禁止使用本文件
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"按原样"提供的，没有任何形式的担保或条件
# 您可以查看许可证了解具体的法律条款和条件

"""从原始仓库中转换OWL-ViT检查点。URL:
https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit"""

import argparse  # 导入解析命令行参数的模块
import collections  # 导入用于操作集合的模块

import jax  # 导入用于自动求导的数值计算库JAX
import jax.numpy as jnp  # 导入JAX中的数学运算模块并重命名为jnp
import torch  # 导入PyTorch深度学习库
import torch.nn as nn  # 导入PyTorch中的神经网络模块
from clip.model import CLIP  # 从CLIP模块中导入CLIP模型
from flax.training import checkpoints  # 导入用于处理检查点的flax训练模块
from huggingface_hub import Repository  # 从Hugging Face Hub中导入Repository类

from transformers import (  # 从transformers库中导入以下模块
    CLIPTokenizer,  # 导入用于CLIP模型的分词器
    OwlViTConfig,  # 导入OWL-ViT模型的配置类
    OwlViTForObjectDetection,  # 导入用于物体检测的OWL-ViT模型
    OwlViTImageProcessor,  # 导入用于图像处理的OWL-ViT模型
    OwlViTModel,  # 导入OWL-ViT模型
    OwlViTProcessor,  # 导入OWL-ViT模型的处理器
)

CONFIGS = {
    "vit_b32": {  # vit_b32配置
        "embed_dim": 512,  # 嵌入维度
        "image_resolution": 768,  # 图像分辨率
        "context_length": 16,  # 上下文长度
        "vocab_size": 49408,  # 词汇表大小
        "vision_layers": 12,  # 视觉层数量
        "vision_width": 768,  # 视觉宽度
        "vision_patch_size": 32,  # 视觉补丁大小
        "transformer_width": 512,  # 转换器宽度
        "transformer_heads": 8,  # 转换器头部数量
        "transformer_layers": 12,  # 转换器层数
    },
    "vit_b16": {  # vit_b16配置
        "embed_dim": 512,  # 嵌入维度
        "image_resolution": 768,  # 图像分辨率
        "context_length": 16,  # 上下文长度
        "vocab_size": 49408,  # 词汇表大小
        "vision_layers": 12,  # 视觉层数量
        "vision_width": 768,  # 视觉宽度
        "vision_patch_size": 16,  # 视觉补丁大小
        "transformer_width": 512,  # 转换器宽度
        "transformer_heads": 8,  # 转换器头部数量
        "transformer_layers": 12,  # 转换器层数
    },
    "vit_l14": {  # vit_l14配置
        "embed_dim": 768,  # 嵌入维度
        "image_resolution": 840,  # 图像分辨率
        "context_length": 16,  # 上下文长度
        "vocab_size": 49408,  # 词汇表大小
        "vision_layers": 24,  # 视觉层数量
        "vision_width": 1024,  # 视觉宽度
        "vision_patch_size": 14,  # 视觉补丁大小
        "transformer_width": 768,  # 转换器宽度
        "transformer_heads": 12,  # 转换器头部数量
        "transformer_layers": 12,  # 转换器层数
    },
}


def flatten_nested_dict(params, parent_key="", sep="/"):
    """将嵌套字典展开为扁平化字典

    Args:
        params (dict): 要展开的嵌套字典
        parent_key (str, optional): 父键名. Defaults to "".
        sep (str, optional): 键之间的分隔符. Defaults to "/".

    Returns:
        dict: 扁平化后的字典
    """
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_f32(params):
    """将参数中的bfloat16类型转换为float32类型

    Args:
        params (any): 待转换的参数

    Returns:
        any: 转换后的参数
    """
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, params)


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    """复制注意力层参数

    Args:
        hf_attn_layer (torch.nn.Module): Hugging Face模型中的注意力层
        pt_attn_layer (torch.nn.Module): PyTorch模型中的注意力层
    """
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias
    # 设置自注意力层的查询投影权重数据为给定的张量 q_proj
    hf_attn_layer.q_proj.weight.data = q_proj
    # 设置自注意力层的查询投影偏置数据为给定的张量 q_proj_bias
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    
    # 设置自注意力层的键投影权重数据为给定的张量 k_proj
    hf_attn_layer.k_proj.weight.data = k_proj
    # 设置自注意力层的键投影偏置数据为给定的张量 k_proj_bias
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    
    # 设置自注意力层的值投影权重数据为给定的张量 v_proj
    hf_attn_layer.v_proj.weight.data = v_proj
    # 设置自注意力层的值投影偏置数据为给定的张量 v_proj_bias
    hf_attn_layer.v_proj.bias.data = v_proj_bias
    
    # 设置自注意力层的输出投影权重数据为给定的张量 out_proj_weights
    hf_attn_layer.out_proj.weight = out_proj_weights
    # 设置自注意力层的输出投影偏置数据为给定的张量 out_proj_bias
    hf_attn_layer.out_proj.bias = out_proj_bias
def copy_mlp(hf_mlp, pt_mlp):
    # 复制多层感知机的线性层
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    # 复制线性层的权重和偏置
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias


def copy_layer(hf_layer, pt_layer):
    # 复制层的归一化层
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # 复制多层感知机
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # 复制注意力层
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    # 遍历并复制每一层
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # 复制编码器的嵌入层
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # 复制最终层归一化
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # 复制隐藏层
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # 复制文本投影
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T

    # 复制文本编码器
    copy_encoder(hf_model.text_model, pt_model)


def copy_vision_model_and_projection(hf_model, pt_model):
    # 复制视觉投影
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T

    # 复制视觉模型的归一化层
    copy_linear(hf_model.vision_model.pre_layernorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # 复制视觉模型的嵌入层
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data

    # 复制视觉模型的编码器
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


def copy_class_merge_token(hf_model, flax_params):
    # 扁平化嵌套字典的类合并标记参数
    flax_class_token_params = flatten_nested_dict(flax_params["backbone"]["merged_class_token"])

    # 将参数转换为PyTorch张量并复制到层归一化的权重和偏置
    weight = torch.from_numpy(flax_class_token_params["scale"])
    bias = torch.from_numpy(flax_class_token_params["bias"])
    hf_model.layer_norm.weight = nn.Parameter(weight)
    hf_model.layer_norm.bias = nn.Parameter(bias)


def copy_class_box_heads(hf_model, flax_params):
    pt_params = hf_model.state_dict()
    new_params = {}

    # 将Flax类预测头参数重命名为PyTorch HF
    flax_class_params = flatten_nested_dict(flax_params["class_head"])
    # 遍历flax_class_params字典，其中包含Flax模型的类别头参数
    for flax_key, v in flax_class_params.items():
        # 将flax_key中的斜杠替换为点号，以匹配PyTorch参数命名风格
        torch_key = flax_key.replace("/", ".")
        # 替换".kernel"为".weight"，调整命名以匹配PyTorch的权重命名
        torch_key = torch_key.replace(".kernel", ".weight")
        # 将"Dense_0"替换为"dense0"，调整命名以匹配PyTorch的命名约定
        torch_key = torch_key.replace("Dense_0", "dense0")
        # 将调整后的参数名加上"class_head."前缀，以表示这些参数属于分类头部
        torch_key = "class_head." + torch_key

        # 如果参数名中包含"weight"且v的维度为2，则将v转置
        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        # 使用torch.from_numpy(v)创建一个PyTorch的参数对象，并保存到new_params字典中
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # 重命名盒预测的Flax参数到PyTorch HF
    # 将obj_box_head中的Flax参数展平为一个字典
    flax_box_params = flatten_nested_dict(flax_params["obj_box_head"])

    # 遍历flax_box_params字典，其中包含盒预测头部的Flax参数
    for flax_key, v in flax_box_params.items():
        # 将flax_key中的斜杠替换为点号，以匹配PyTorch参数命名风格
        torch_key = flax_key.replace("/", ".")
        # 替换".kernel"为".weight"，调整命名以匹配PyTorch的权重命名
        torch_key = torch_key.replace(".kernel", ".weight")
        # 替换下划线为空字符串，将所有字符转为小写，调整命名以匹配PyTorch的命名约定
        torch_key = torch_key.replace("_", "").lower()
        # 将调整后的参数名加上"box_head."前缀，以表示这些参数属于盒预测头部
        torch_key = "box_head." + torch_key

        # 如果参数名中包含"weight"且v的维度为2，则将v转置
        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        # 使用torch.from_numpy(v)创建一个PyTorch的参数对象，并保存到new_params字典中
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # 将调整后的参数复制到PyTorch的模型参数中
    for name, param in new_params.items():
        # 如果new_params中的参数名在pt_params中存在，则将其复制到pt_params中
        if name in pt_params.keys():
            pt_params[name].copy_(param)
# 将 Flax CLIP 模型的注意力参数复制到 Hugging Face PyTorch 模型的对应位置
def copy_flax_attn_params(hf_backbone, flax_attn_params):
    # 遍历 Flax 模型的注意力参数字典
    for k, v in flax_attn_params.items():
        # 如果键名以 "transformer" 开头，则替换为对应的 PyTorch 键名
        if k.startswith("transformer"):
            torch_key = k.replace("transformer.resblocks", "text_model.encoder.layers")
        else:
            torch_key = k.replace("visual.transformer.resblocks", "vision_model.encoder.layers")

        # 将键名中的 "attn" 替换为 "self_attn"
        torch_key = torch_key.replace("attn", "self_attn")
        # 将键名中的 "key" 替换为 "k_proj"
        torch_key = torch_key.replace("key", "k_proj")
        # 将键名中的 "value" 替换为 "v_proj"
        torch_key = torch_key.replace("value", "v_proj")
        # 将键名中的 "query" 替换为 "q_proj"
        torch_key = torch_key.replace("query", "q_proj")
        # 将键名中的 "out" 替换为 "out_proj"
        torch_key = torch_key.replace("out", "out_proj")

        # 如果键名包含 "bias" 并且值的维度为 2，则将值进行形状变换
        if "bias" in torch_key and v.ndim == 2:
            shape = v.shape[0] * v.shape[1]
            v = v.reshape(shape)

        # 如果键名包含 "weight" 并且包含 "out"，则将值进行形状变换和转置
        if "weight" in torch_key and "out" in torch_key:
            shape = (v.shape[0] * v.shape[1], v.shape[2])
            v = v.reshape(shape).T

        # 如果键名包含 "weight" 但不包含 "out"，则将值进行形状变换和转置
        if "weight" in torch_key and "out" not in torch_key:
            shape = (v.shape[0], v.shape[1] * v.shape[2])
            v = v.reshape(shape).T

        # 将 NumPy 数组转换为 PyTorch 张量，并复制到 Hugging Face PyTorch 模型的对应位置
        v = torch.from_numpy(v)
        hf_backbone.state_dict()[torch_key].copy_(v)


# 将 Flax CLIP 模型的注意力层参数转换为适合 Hugging Face PyTorch 模型的参数格式
def _convert_attn_layers(params):
    new_params = {}
    processed_attn_layers = []

    # 遍历参数字典
    for k, v in params.items():
        # 如果键名中包含 "attn."
        if "attn." in k:
            # 提取基础键名
            base = k[: k.rindex("attn.") + 5]
            # 如果基础键名已经处理过，则跳过
            if base in processed_attn_layers:
                continue

            # 将基础键名加入已处理列表
            processed_attn_layers.append(base)
            # 获取维度信息
            dim = params[base + "out.weight"].shape[-1]
            # 转换权重参数并进行转置，存入新参数字典
            new_params[base + "out_proj.weight"] = params[base + "out.weight"].reshape(dim, dim).T
            # 复制偏置参数到新参数字典
            new_params[base + "out_proj.bias"] = params[base + "out.bias"]
        else:
            # 直接复制非注意力层参数到新参数字典
            new_params[k] = v
    return new_params


# 将 Flax CLIP 模型的参数转换为适合 Hugging Face PyTorch CLIP 模型的参数格式
def convert_clip_backbone(flax_params, torch_config):
    # 使用给定的 PyTorch 配置创建 CLIP 模型
    torch_model = CLIP(**torch_config)
    # 将模型设为评估模式
    torch_model.eval()
    # 获取 PyTorch CLIP 模型的状态字典
    torch_clip_params = torch_model.state_dict()

    # 将嵌套字典展平为一级键值对
    flax_clip_params = flatten_nested_dict(flax_params["backbone"]["clip"])
    # 初始化新的 PyTorch 参数字典
    new_torch_params = {}
    # 遍历 flax_clip_params 字典的键值对
    for flax_key, v in flax_clip_params.items():
        # 将 flax 的键名替换为符合 PyTorch 命名规范的格式
        torch_key = flax_key.replace("/", ".")
        # 进一步替换特定的文本处理层的命名格式
        torch_key = torch_key.replace("text.token_embedding.embedding", "token_embedding.kernel")

        # 如果 torch_key 以指定的文本处理模块开头，则删除开头的部分
        if (
            torch_key.startswith("text.transformer")
            or torch_key.startswith("text.text_projection")
            or torch_key.startswith("text.ln_final")
            or torch_key.startswith("text.positional_embedding")
        ):
            torch_key = torch_key[5:]

        # 进一步替换其他特定的模块命名格式
        torch_key = torch_key.replace("text_projection.kernel", "text_projection")
        torch_key = torch_key.replace("visual.proj.kernel", "visual.proj")
        torch_key = torch_key.replace(".scale", ".weight")
        torch_key = torch_key.replace(".kernel", ".weight")

        # 如果 torch_key 包含 "conv" 或者 "downsample.0.weight"，进行张量维度转置
        if "conv" in torch_key or "downsample.0.weight" in torch_key:
            v = v.transpose(3, 2, 0, 1)

        # 如果 torch_key 包含 "weight"，且张量维度为二维，并且不是嵌入层，进行转置
        elif "weight" in torch_key and v.ndim == 2 and "embedding" not in torch_key:
            # 全连接层进行转置，嵌入层不转置
            v = v.T

        # 将处理后的键值对存入 new_torch_params 字典
        new_torch_params[torch_key] = v

    # 调用 _convert_attn_layers 函数将注意力层参数进行转换
    attn_params = _convert_attn_layers(new_torch_params)
    # 将转换后的注意力层参数更新到 new_torch_params 字典中
    new_torch_params.update(attn_params)
    # 清空 attn_params 字典
    attn_params = {}

    # 将 flax CLIP 骨干网络参数复制到 PyTorch 参数中
    for name, param in new_torch_params.items():
        # 如果参数名在 torch_clip_params 的键中存在
        if name in torch_clip_params.keys():
            # 将 new_torch_params 中的 NumPy 数组转换为 PyTorch 张量，并复制给 torch_clip_params
            new_param = torch.from_numpy(new_torch_params[name])
            torch_clip_params[name].copy_(new_param)
        else:
            # 将未复制的参数存入 attn_params 字典中
            attn_params[name] = param

    # 返回更新后的 PyTorch 参数、模型及注意力参数
    return torch_clip_params, torch_model, attn_params
@torch.no_grad()
def convert_owlvit_checkpoint(pt_backbone, flax_params, attn_params, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 创建一个本地仓库对象，克隆或加载现有的PyTorch模型存储路径
    repo = Repository(pytorch_dump_folder_path, clone_from=f"google/{pytorch_dump_folder_path}")
    # 执行git pull操作，更新本地仓库内容
    repo.git_pull()

    # 如果提供了配置文件路径，则从预训练模型配置中加载OwlViTConfig
    if config_path is not None:
        config = OwlViTConfig.from_pretrained(config_path)
    else:
        # 否则创建一个空的OwlViTConfig对象
        config = OwlViTConfig()

    # 初始化一个评估模式的OwlViTModel和OwlViTForObjectDetection模型
    hf_backbone = OwlViTModel(config).eval()
    hf_model = OwlViTForObjectDetection(config).eval()

    # 复制文本模型和投影层到hf_backbone
    copy_text_model_and_projection(hf_backbone, pt_backbone)
    # 复制视觉模型和投影层到hf_backbone
    copy_vision_model_and_projection(hf_backbone, pt_backbone)
    # 将pt_backbone的logit_scale属性复制到hf_backbone的logit_scale属性
    hf_backbone.logit_scale = pt_backbone.logit_scale
    # 复制Flax的注意力参数到hf_backbone
    copy_flax_attn_params(hf_backbone, attn_params)

    # 将hf_backbone设置为hf_model的OwlViT模块
    hf_model.owlvit = hf_backbone
    # 复制flax_params中的类合并令牌到hf_model
    copy_class_merge_token(hf_model, flax_params)
    # 复制flax_params中的类盒头到hf_model
    copy_class_box_heads(hf_model, flax_params)

    # 保存转换后的HF模型到本地仓库目录
    hf_model.save_pretrained(repo.local_dir)

    # 初始化图像处理器，使用指定的图像大小和裁剪大小
    image_processor = OwlViTImageProcessor(
        size=config.vision_config.image_size, crop_size=config.vision_config.image_size
    )
    # 初始化CLIPTokenizer，从预训练模型"openai/clip-vit-base-patch32"加载，并设置pad_token和model_max_length
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!", model_max_length=16)

    # 初始化OwlViTProcessor，传入image_processor和tokenizer作为参数
    processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # 将image_processor保存到本地仓库目录
    image_processor.save_pretrained(repo.local_dir)
    # 将processor保存到本地仓库目录
    processor.save_pretrained(repo.local_dir)

    # 向git仓库添加修改
    repo.git_add()
    # 提交修改，并添加描述信息"Upload model and processor"
    repo.git_commit("Upload model and processor")
    # 推送本地仓库内容到远程仓库

    repo.git_push()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--owlvit_version",
        default=None,
        type=str,
        required=True,
        help="OWL-ViT model name [clip_b16, clip_b32, clip_l14].",
    )
    parser.add_argument(
        "--owlvit_checkpoint", default=None, type=str, required=True, help="Path to flax model checkpoint."
    )
    parser.add_argument("--hf_config", default=None, type=str, required=True, help="Path to HF model config.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default="hf_model", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    # 初始化PyTorch CLIP模型
    model_name = args.owlvit_version
    if model_name == "clip_b16":
        torch_config = CONFIGS["vit_b16"]
    elif model_name == "clip_b32":
        torch_config = CONFIGS["vit_b32"]
    elif model_name == "clip_l14":
        torch_config = CONFIGS["vit_l14"]

    # 从检查点中加载变量，并将参数转换为float-32
    variables = checkpoints.restore_checkpoint(args.owlvit_checkpoint, target=None)["optimizer"]["target"]
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    del variables

    # 转换CLIP的backbone
    # 调用函数 convert_clip_backbone 将 flax_params 转换为 pt_backbone_params、clip_pt 和 attn_params
    pt_backbone_params, clip_pt, attn_params = convert_clip_backbone(flax_params, torch_config)

    # 调用函数 convert_owlvit_checkpoint，将 clip_pt、flax_params 和 attn_params 转换为 PyTorch 模型的检查点
    # 将结果保存到指定的路径 args.pytorch_dump_folder_path，并传入额外的配置参数 args.hf_config
    convert_owlvit_checkpoint(clip_pt, flax_params, attn_params, args.pytorch_dump_folder_path, args.hf_config)
```