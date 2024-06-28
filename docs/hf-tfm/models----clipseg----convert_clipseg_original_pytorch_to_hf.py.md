# `.\models\clipseg\convert_clipseg_original_pytorch_to_hf.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，该代码由 HuggingFace Inc. 团队版权所有
#
# 根据 Apache 许可证 2.0 版本发布，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”分发的，不提供任何形式的明示或暗示保证
# 请参阅许可证了解具体语言和限制
#

"""从原始存储库转换 CLIPSeg 检查点。URL: https://github.com/timojl/clipseg."""

# 导入所需的库
import argparse  # 解析命令行参数

import requests  # 发送 HTTP 请求
import torch  # PyTorch 库
from PIL import Image  # Python Imaging Library，处理图像

from transformers import (  # 导入 Transformers 库中的相关模块和类
    CLIPSegConfig,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    CLIPSegTextConfig,
    CLIPSegVisionConfig,
    CLIPTokenizer,
    ViTImageProcessor,
)

# 定义函数，根据模型名称获取对应的 CLIPSegConfig 配置对象
def get_clipseg_config(model_name):
    # 创建 CLIPSegTextConfig 对象
    text_config = CLIPSegTextConfig()
    # 创建 CLIPSegVisionConfig 对象，并指定 patch 大小为 16
    vision_config = CLIPSegVisionConfig(patch_size=16)

    # 根据模型名称确定是否使用复杂的转置卷积
    use_complex_transposed_convolution = True if "refined" in model_name else False
    # 根据模型名称确定降维大小
    reduce_dim = 16 if "rd16" in model_name else 64

    # 创建 CLIPSegConfig 对象，从 text_config 和 vision_config 创建配置
    config = CLIPSegConfig.from_text_vision_configs(
        text_config,
        vision_config,
        use_complex_transposed_convolution=use_complex_transposed_convolution,
        reduce_dim=reduce_dim,
    )
    return config

# 定义函数，重命名键名以匹配转换后的 CLIPSeg 模型
def rename_key(name):
    # 更新前缀
    if "clip_model" in name:
        name = name.replace("clip_model", "clip")
    if "transformer" in name:
        if "visual" in name:
            name = name.replace("visual.transformer", "vision_model")
        else:
            name = name.replace("transformer", "text_model")
    if "resblocks" in name:
        name = name.replace("resblocks", "encoder.layers")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "attn" in name and "self" not in name:
        name = name.replace("attn", "self_attn")
    # 文本编码器
    if "token_embedding" in name:
        name = name.replace("token_embedding", "text_model.embeddings.token_embedding")
    if "positional_embedding" in name and "visual" not in name:
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # 视觉编码器
    if "visual.class_embedding" in name:
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    # 检查字符串"name"是否包含"visual.positional_embedding"
    if "visual.positional_embedding" in name:
        # 如果包含，则用"vision_model.embeddings.position_embedding.weight"替换它
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    
    # 检查字符串"name"是否包含"visual.ln_pre"
    if "visual.ln_pre" in name:
        # 如果包含，则用"vision_model.pre_layrnorm"替换它
        name = name.replace("visual.ln_pre", "vision_model.pre_layrnorm")
    
    # 检查字符串"name"是否包含"visual.ln_post"
    if "visual.ln_post" in name:
        # 如果包含，则用"vision_model.post_layernorm"替换它
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    
    # 检查字符串"name"是否包含"visual.proj"
    if "visual.proj" in name:
        # 如果包含，则用"visual_projection.weight"替换它
        name = name.replace("visual.proj", "visual_projection.weight")
    
    # 检查字符串"name"是否包含"text_projection"
    if "text_projection" in name:
        # 如果包含，则用"text_projection.weight"替换它
        name = name.replace("text_projection", "text_projection.weight")
    
    # 检查字符串"name"是否包含"trans_conv"
    if "trans_conv" in name:
        # 如果包含，则用"transposed_convolution"替换它
        name = name.replace("trans_conv", "transposed_convolution")
    
    # 如果字符串"name"包含"film_mul"、"film_add"、"reduce"或"transposed_convolution"中的任意一个
    if "film_mul" in name or "film_add" in name or "reduce" in name or "transposed_convolution" in name:
        # 替换"name"为"decoder." + name
        name = "decoder." + name
    
    # 检查字符串"name"是否包含"blocks"
    if "blocks" in name:
        # 如果包含，则用"decoder.layers"替换它
        name = name.replace("blocks", "decoder.layers")
    
    # 检查字符串"name"是否包含"linear1"
    if "linear1" in name:
        # 如果包含，则用"mlp.fc1"替换它
        name = name.replace("linear1", "mlp.fc1")
    
    # 检查字符串"name"是否包含"linear2"
    if "linear2" in name:
        # 如果包含，则用"mlp.fc2"替换它
        name = name.replace("linear2", "mlp.fc2")
    
    # 检查字符串"name"是否包含"norm1"且不包含"layer_"
    if "norm1" in name and "layer_" not in name:
        # 如果满足条件，则用"layer_norm1"替换它
        name = name.replace("norm1", "layer_norm1")
    
    # 检查字符串"name"是否包含"norm2"且不包含"layer_"
    if "norm2" in name and "layer_" not in name:
        # 如果满足条件，则用"layer_norm2"替换它
        name = name.replace("norm2", "layer_norm2")
    
    # 返回修改后的"name"
    return name
# 将原始状态字典转换为适合新模型的格式
def convert_state_dict(orig_state_dict, config):
    # 使用 .copy() 创建原始字典的副本，以便安全地迭代和修改
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键以 "clip_model" 开头并且包含 "attn.in_proj"，则进行下列操作
        if key.startswith("clip_model") and "attn.in_proj" in key:
            # 按 "." 分割键名
            key_split = key.split(".")
            # 根据键名中是否含有 "visual" 选择相应的处理
            if "visual" in key:
                # 提取层编号和隐藏层大小
                layer_num = int(key_split[4])
                dim = config.vision_config.hidden_size
                prefix = "vision_model"
            else:
                layer_num = int(key_split[3])
                dim = config.text_config.hidden_size
                prefix = "text_model"

            # 根据键名中是否含有 "weight"，更新对应的原始状态字典
            if "weight" in key:
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        
        # 如果键包含 "self_attn" 但不含 "out_proj"，则进行下列操作
        elif "self_attn" in key and "out_proj" not in key:
            # 按 "." 分割键名
            key_split = key.split(".")
            # 提取层编号和降维大小
            layer_num = int(key_split[1])
            dim = config.reduce_dim
            # 根据键名中是否含有 "weight"，更新对应的原始状态字典
            if "weight" in key:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        
        # 否则，对当前键进行重命名并将值更新到原始状态字典中
        else:
            new_name = rename_key(key)
            # 如果新键名中含有 "visual_projection" 或 "text_projection"，则对值进行转置
            if "visual_projection" in new_name or "text_projection" in new_name:
                val = val.T
            orig_state_dict[new_name] = val

    # 返回转换后的原始状态字典
    return orig_state_dict
    # 使用 state_dict 的副本遍历所有键
    for key in state_dict.copy().keys():
        # 如果键以 "model" 开头，则从 state_dict 中删除该键
        if key.startswith("model"):
            state_dict.pop(key, None)

    # 重命名一些键值
    state_dict = convert_state_dict(state_dict, config)
    # 加载经过转换后的 state_dict 到模型中，允许部分不严格匹配
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # 检查缺失的键是否符合预期
    if missing_keys != ["clip.text_model.embeddings.position_ids", "clip.vision_model.embeddings.position_ids"]:
        raise ValueError("Missing keys that are not expected: {}".format(missing_keys))
    # 检查意外的键是否符合预期
    if unexpected_keys != ["decoder.reduce.weight", "decoder.reduce.bias"]:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    # 创建图像处理器和文本处理器
    image_processor = ViTImageProcessor(size=352)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPSegProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 准备图像和文本输入
    image = prepare_img()
    text = ["a glass", "something to fill", "wood", "a jar"]

    # 使用处理器处理文本和图像输入，进行填充并返回 PyTorch 张量
    inputs = processor(text=text, images=[image] * len(text), padding="max_length", return_tensors="pt")

    # 使用无梯度计算环境执行模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 验证输出的特定值是否符合预期
    expected_conditional = torch.tensor([0.1110, -0.1882, 0.1645])
    expected_pooled_output = torch.tensor([0.2692, -0.7197, -0.1328])
    if model_name == "clipseg-rd64-refined":
        expected_masks_slice = torch.tensor(
            [[-10.0407, -9.9431, -10.2646], [-9.9751, -9.7064, -9.9586], [-9.6891, -9.5645, -9.9618]]
        )
    elif model_name == "clipseg-rd64":
        expected_masks_slice = torch.tensor(
            [[-7.2877, -7.2711, -7.2463], [-7.2652, -7.2780, -7.2520], [-7.2239, -7.2204, -7.2001]]
        )
    elif model_name == "clipseg-rd16":
        expected_masks_slice = torch.tensor(
            [[-6.3955, -6.4055, -6.4151], [-6.3911, -6.4033, -6.4100], [-6.3474, -6.3702, -6.3762]]
        )
    else:
        # 如果模型名称不受支持，则引发 ValueError
        raise ValueError(f"Model name {model_name} not supported.")

    # 使用 allclose 函数验证张量是否在给定的容差内相等
    assert torch.allclose(outputs.logits[0, :3, :3], expected_masks_slice, atol=1e-3)
    assert torch.allclose(outputs.conditional_embeddings[0, :3], expected_conditional, atol=1e-3)
    assert torch.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=1e-3)
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了 push_to_hub，则将模型和处理器推送到 Hub
    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to the hub")
        model.push_to_hub(f"CIDAS/{model_name}")
        processor.push_to_hub(f"CIDAS/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default="clipseg-rd64",
        type=str,
        choices=["clipseg-rd16", "clipseg-rd64", "clipseg-rd64-refined"],
        help=(
            "Name of the model. Supported models are: clipseg-rd64, clipseg-rd16 and clipseg-rd64-refined (rd meaning"
            " reduce dimension)"
        ),
    )

    # 可选参数：原始检查点路径
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/CLIPSeg/clip_plus_rd64-uni.pth",
        type=str,
        help=(
            "Path to the original checkpoint. Note that the script assumes that the checkpoint includes both CLIP and"
            " the decoder weights."
        ),
    )

    # 可选参数：输出 PyTorch 模型目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 可选参数：是否推送模型到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，转换 CLIPSeg 检查点
    convert_clipseg_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)


这段代码是一个命令行工具的入口点，使用 argparse 模块解析命令行参数，并调用 `convert_clipseg_checkpoint` 函数进行处理。
```