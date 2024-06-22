# `.\models\deta\convert_deta_swin_to_pytorch.py`

```py
# 设置编码格式为utf-8
# 版权声明，许可证信息，以及规则
# 导入需要的库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image
from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor, SwinConfig
from transformers.utils import logging

# 设置日志的显示级别
logging.set_verbosity_info()
# 获取当前脚本的日志对象
logger = logging.get_logger(__name__)

# 创建函数用于获取DETA模型的配置信息
def get_deta_config(model_name):
    # 设置SwinTransformer的配置
    backbone_config = SwinConfig(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=12,
        out_features=["stage2", "stage3", "stage4"],
    )

    # 设置DETA的配置信息
    config = DetaConfig(
        backbone_config=backbone_config,
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # 设置标签信息
    repo_id = "huggingface/label-files"
    if "o365" in model_name:
        num_labels = 366
        filename = "object365-id2label.json"
    else:
        num_labels = 91
        filename = "coco-detection-id2label.json"
    # 设置DETA的标签数量
    config.num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config

# 创建函数用于重命名模型参数(原始名称和新名称)
def create_rename_keys(config):
    rename_keys = []

    # stem
    # fmt: off
    # 以下是待重命名的参数
    rename_keys.append(("backbone.0.body.patch_embed.proj.weight", "model.backbone.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.0.body.patch_embed.proj.bias", "model.backbone.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.0.body.patch_embed.norm.weight", "model.backbone.model.embeddings.norm.weight"))
    rename_keys.append(("backbone.0.body.patch_embed.norm.bias", "model.backbone.model.embeddings.norm.bias"))
    # stages
    # 对于每一层骨干结构的深度遍历
    for i in range(len(config.backbone_config.depths)):
        # 遍历当前层的所有块
        for j in range(config.backbone_config.depths[i]):
            # 添加每个块的第一个层规范化权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm1.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            # 添加每个块的第一个层规范化偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm1.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            # 添加相对位置偏移表重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.relative_position_bias_table", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            # 添加相对位置索引重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.relative_position_index", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            # 添加注意力输出投影权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.proj.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            # 添加注意力输出投影偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.proj.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            # 添加第二个层规范化权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm2.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            # 添加第二个层规范化偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm2.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            # 添加 MLP 第一层权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc1.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            # 添加 MLP 第一层偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc1.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            # 添加 MLP 第二层权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc2.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            # 添加 MLP 第二层偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc2.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.output.dense.bias"))
    
        # 检查当前层是否需要重命名下采样的参数
        if i < 3:
            # 添加下采样 reduction 权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.reduction.weight", f"model.backbone.model.encoder.layers.{i}.downsample.reduction.weight"))
            # 添加下采样规范化权重重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.norm.weight", f"model.backbone.model.encoder.layers.{i}.downsample.norm.weight"))
            # 添加下采样规范化偏置重命名
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.norm.bias", f"model.backbone.model.encoder.layers.{i}.downsample.norm.bias"))
    
    # 添加整体骨干结构的第一个层规范化权重重命名
    rename_keys.append(("backbone.0.body.norm1.weight", "model.backbone.model.hidden_states_norms.stage2.weight"))
    # 添加整体骨干结构的第一个层规范化偏置重命名
    rename_keys.append(("backbone.0.body.norm1.bias", "model.backbone.model.hidden_states_norms.stage2.bias"))
    # 添加键值对到重命名键列表中，将旧键名和新键名作为元组添加到列表中
    rename_keys.append(("backbone.0.body.norm2.weight", "model.backbone.model.hidden_states_norms.stage3.weight"))
    rename_keys.append(("backbone.0.body.norm2.bias", "model.backbone.model.hidden_states_norms.stage3.bias"))
    rename_keys.append(("backbone.0.body.norm3.weight", "model.backbone.model.hidden_states_norms.stage4.weight"))
    rename_keys.append(("backbone.0.body.norm3.bias", "model.backbone.model.hidden_states_norms.stage4.bias"))

    # 循环遍历 transformer 编码器的层，处理每一层的权重和偏置
    for i in range(config.encoder_layers):
        # 添加键值对到重命名键列表中，使用格式化字符串添加旧键名和新键名
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        # ... 其他类似操作
        # 一直到最后一层的操作。
    # 对模型中的多层解码器进行重命名操作，将指定层的参数键名替换为新的命名
    for i in range(config.decoder_layers):
        # 将跨注意力机制中的权重键名重命名为编码器注意力机制中的对应键名
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.weight", f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.bias", f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.weight", f"model.decoder.layers.{i}.encoder_attn.value_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.bias", f"model.decoder.layers.{i}.encoder_attn.value_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.weight", f"model.decoder.layers.{i}.encoder_attn.output_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.bias", f"model.decoder.layers.{i}.encoder_attn.output_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"model.decoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"model.decoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias"))
    
    # 恢复格式设置
    # 返回重命名后的键名列表
    return rename_keys
# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将该值对应的新键加入字典
    dct[new] = val

# 从给定的状态字典和骨干配置中读取 Swin Transformer 模型的查询、键和值
def read_in_swin_q_k_v(state_dict, backbone_config):
    # 计算每个编码器层级的特征数
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # 读取原始实现中每个编码器层级矩阵的查询、键和值的权重和偏置
            in_proj_weight = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.bias")
            # 添加新的查询、键和值权重和偏置到状态字典中
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[dim : dim * 2, :]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[dim : dim * 2]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[-dim :, :]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on

# 从给定的状态字典和配置中读取解码器的查询、键和值
def read_in_decoder_q_k_v(state_dict, config):
    # 解码器的隐藏层大小
    hidden_size = config.d_model
    for i in range(config.decoder_layers):
        # 读取解码器自注意力的输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 添加新的查询、键和值权重和偏置到状态字典中
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]
# 准备一张图片，用于验证模型转换结果
def prepare_img():
    # 图片链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用请求获取图片的原始数据流，并打开为图片对象
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
# 转换模型权重到我们的 DETA 结构
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETA structure.
    """

    # 加载配置
    config = get_deta_config(model_name)

    # 加载原始状态字典
    if model_name == "deta-swin-large":
        checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename="adet_swin_ft.pth")
    elif model_name == "deta-swin-large-o365":
        checkpoint_path = hf_hub_download(repo_id="jozhang97/deta-swin-l-o365", filename="deta_swin_pt_o365.pth")
    else:
        raise ValueError(f"Model name {model_name} not supported")

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 打印原始状态字典的名称和形状
    for name, param in state_dict.items():
        print(name, param.shape)

    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    read_in_decoder_q_k_v(state_dict, config)

    # 修正一些前缀
    for key in state_dict.copy().keys():
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        if "input_proj" in key:
            val = state_dict.pop(key)
            state_dict["model." + key] = val
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer", "model")] = val

    # 最后，创建 HuggingFace 模型并加载状态字典
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 加载图片处理器
    processor = DetaImageProcessor(format="coco_detection")

    # 在图片上验证我们的转换
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))

    # 验证 logits
    print("Logits:", outputs.logits[0, :3, :3])
    print("Boxes:", outputs.pred_boxes[0, :3, :3])
    if model_name == "deta-swin-large":
        expected_logits = torch.tensor(
            [[-7.6308, -2.8485, -5.3737], [-7.2037, -4.5505, -4.8027], [-7.2943, -4.2611, -4.6617]]
        )
        expected_boxes = torch.tensor([[0.4987, 0.4969, 0.9999], [0.2549, 0.5498, 0.4805], [0.5498, 0.2757, 0.0569]])
```  
    # 如果模型名为 "deta-swin-large-o365"，则设置期望的 logits 和 boxes
    elif model_name == "deta-swin-large-o365":
        expected_logits = torch.tensor(
            [[-8.0122, -3.5720, -4.9717], [-8.1547, -3.6886, -4.6389], [-7.6610, -3.6194, -5.0134]]
        )
        expected_boxes = torch.tensor([[0.2523, 0.5549, 0.4881], [0.7715, 0.4149, 0.4601], [0.5503, 0.2753, 0.0575]])
    # 检查模型的输出 logits 和 boxes 是否与期望值的所有元素近似相等
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    # 打印一条消息表明一切正常
    print("Everything ok!")

    # 如果提供了 pytorch_dump_folder_path
    if pytorch_dump_folder_path:
        # 保存模型和处理器到指定路径
        logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存处理器到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型和处理器推送到 hub
    if push_to_hub:
        # 打印一条消息表明正在将模型和处理器推送到 hub
        print("Pushing model and processor to hub...")
        # 将模型推送到 hub
        model.push_to_hub(f"jozhang97/{model_name}")
        # 将处理器推送到 hub
        processor.push_to_hub(f"jozhang97/{model_name}")
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加模型名称参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-swin-large",
        choices=["deta-swin-large", "deta-swin-large-o365"],
        help="Name of the model you'd like to convert.",
    )
    # 添加输出 PyTorch 模型的文件夹路径参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    # 添加是否将转换后的模型推送到 🤗 hub 的参数
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 DETA 检查点转换为 PyTorch 模型
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```