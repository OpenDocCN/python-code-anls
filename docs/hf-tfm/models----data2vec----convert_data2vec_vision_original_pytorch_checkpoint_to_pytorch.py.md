# `.\models\data2vec\convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py`

```
#!/usr/bin/env python3
# 在Unix系统中，告诉操作系统使用Python3解释器执行脚本

import argparse
# 导入argparse模块，用于解析命令行参数和生成帮助文档

import json
# 导入json模块，用于处理JSON格式的数据

import torch
# 导入torch模块，PyTorch深度学习库

from huggingface_hub import hf_hub_download
# 从huggingface_hub模块导入hf_hub_download函数

from PIL import Image
# 从PIL模块导入Image类，用于处理图片

from timm.models import create_model
# 从timm.models模块导入create_model函数，用于创建模型

from transformers import (
    BeitImageProcessor,
    Data2VecVisionConfig,
    Data2VecVisionForImageClassification,
    Data2VecVisionModel,
)
# 从transformers模块导入BeitImageProcessor、Data2VecVisionConfig、Data2VecVisionForImageClassification、Data2VecVisionModel类

def create_rename_keys(config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec."):
    # 创建函数create_rename_keys，根据提供的参数创建重命名键列表
    prefix = "backbone." if is_semantic else ""
    # 如果是语义模型，前缀为"backbone."，否则为空字符串

    rename_keys = []
    # 创建空列表rename_keys，用于存储重命名的键值对
    for i in range(config.num_hidden_layers):
        # 遍历config.num_hidden_layers次
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm1.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_before.weight")
        )
        # 将(权重名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_before.bias"))
        # 将(偏置名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.weight")
        )
        # 将(权重名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.bias")
        )
        # 将(偏置名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm2.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_after.weight")
        )
        # 将(权重名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_after.bias"))
        # 将(偏置名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.weight", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.weight")
        )
        # 将(权重名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.bias", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.bias")
        )
        # 将(偏置名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"{hf_prefix}encoder.layer.{i}.output.dense.weight"))
        # 将(权重名称, 新名称)的键值对添加到rename_keys列表中
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"{hf_prefix}encoder.layer.{i}.output.dense.bias"))
        # 将(偏置名称, 新名称)的键值对添加到rename_keys列表中

    # projection layer + position embeddings
    rename_keys.extend(
        [
            (f"{prefix}cls_token", f"{hf_prefix}embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", f"{hf_prefix}embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", f"{hf_prefix}embeddings.patch_embeddings.projection.bias"),
        ]
    )
    # 扩展rename_keys列表，添加投影层和位置嵌入的重命名键值对
    # 如果有 LM 头部
    if has_lm_head:
        # 将指定的键添加到重命名列表中
        rename_keys.extend(
            [
                ("mask_token", f"{hf_prefix}embeddings.mask_token"),  # 重命名 mask token
                (
                    "rel_pos_bias.relative_position_bias_table",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_bias_table",
                ),  # 重命名相对位置偏置表
                (
                    "rel_pos_bias.relative_position_index",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_index",
                ),  # 重命名相对位置索引
                ("norm.weight", "layernorm.weight"),  # 重命名权重
                ("norm.bias", "layernorm.bias"),  # 重命名偏置
            ]
        )
    # 如果是语义
    elif is_semantic:
        # 将指定的键添加到重命名列表中
        rename_keys.extend(
            [
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),  # 重命名卷积层权重
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),  # 重命名卷积层偏置
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),  # 重命名辅助卷积层权重
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),  # 重命名辅助卷积层偏置
            ]
        )
    # 如果不是语义也不是 LM 头部
    else:
        # 将指定的键添加到重命名列表中
        rename_keys.extend(
            [
                ("fc_norm.weight", f"{hf_prefix}pooler.layernorm.weight"),  # 重命名权重
                ("fc_norm.bias", f"{hf_prefix}pooler.layernorm.bias"),  # 重命名偏置
                ("head.weight", "classifier.weight"),  # 重命名头部权重
                ("head.bias", "classifier.bias"),  # 重命名头部偏置
            ]
        )

    # 返回重命名列表
    return rename_keys
# 从 state_dict 中读取 qkv 权重和偏置，用于构建注意力机制的查询、键和值
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec_vision."):
    # 遍历所有隐藏层
    for i in range(config.num_hidden_layers):
        # 如果是语义模型，则使用 "backbone." 前缀
        prefix = "backbone." if is_semantic else ""
        
        # 读取注意力机制的查询权重、查询偏置和值偏置
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")
        
        # 更新 state_dict 中的注意力查询、键和值权重和偏置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[config.hidden_size : config.hidden_size * 2, :]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.bias"] = v_bias
        
        # 读取 gamma_1 和 gamma_2，更新 state_dict 中的 lambda_1 和 lambda_2
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")
        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_2"] = gamma_2
        
        # 如果没有 LM 头部，则读取相对位置偏置表格和索引，更新 state_dict 中的相对位置偏置相关内容
        if not has_lm_head:
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")
            state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"] = table
            state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"] = index


# 获取命令行参数
def get_args():
    # 创建解析器
    parser = argparse.ArgumentParser("Convert Data2VecVision to HF for image classification and pretraining", add_help=False)
    # 添加命令行参数
    parser.add_argument("--hf_checkpoint_name", type=str)
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--beit_checkpoint", default="", help="beit checkpoint")
    
    # 解析并返回参数
    return parser.parse_args()


# 加载 BEiT 模型
def load_beit_model(args, is_finetuned, is_large):
    # 加载给定模型的状态字典
    def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
        # 用来存储未找到的键
        missing_keys = []
        # 用来存储意外的键
        unexpected_keys = []
        # 用来存储错误消息
        error_msgs = []
        # 复制state_dict以便_load_from_state_dict可以修改它
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # 从state_dict加载模块
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # 加载模型和前缀
        load(model, prefix=prefix)

        # 警告未找到的键
        warn_missing_keys = []
        # 忽略的键
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split("|"):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            # 打印未从预训练模型中初始化的模型权重
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            # 打印未在预训练模型中使用的权重
            print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            # 打印从预训练模型中未初始化的被忽略权重
            print(
                "Ignored weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, ignore_missing_keys
                )
            )
        if len(error_msgs) > 0:
            print("\n".join(error_msgs))

    # 模型的一些参数设置
    model_kwargs = {
        "pretrained": False,
        "use_shared_rel_pos_bias": True,
        "use_abs_pos_emb": False,
        "init_values": 0.1,
    }

    # 如果是微调
    if is_finetuned:
        model_kwargs.update(
            {
                "num_classes": 1000,
                "use_mean_pooling": True,
                "init_scale": 0.001,
                "use_rel_pos_bias": True,
            }
        )

    # 创建模型并根据参数更新模型类型
    model = create_model(
        "beit_large_patch16_224" if is_large else "beit_base_patch16_224",
        **model_kwargs,
    )
    patch_size = model.patch_embed.patch_size
    # 设置窗口大小
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    checkpoint = torch.load(args.beit_checkpoint, map_location="cpu")

    # 加载检查点
    print(f"Load ckpt from {args.beit_checkpoint}")
    checkpoint_model = None
    # 遍历模型关键字列表，检查是否存在模型相关的关键字（"model"或"module"）
    for model_key in ("model", "module"):
        # 如果存在指定的模型关键字
        if model_key in checkpoint:
            # 获取该关键字对应的模型参数
            checkpoint_model = checkpoint[model_key]
            # 打印加载状态字典的消息，指明加载的模型关键字
            print(f"Load state_dict by model_key = {model_key}")
            # 中断循环，仅加载第一个匹配的模型关键字对应的状态字典
            break

    # 获取状态字典中的所有键
    all_keys = list(checkpoint_model.keys())
    # 遍历所有键
    for key in all_keys:
        # 如果键中包含"relative_position_index"
        if "relative_position_index" in key:
            # 从状态字典中移除该键
            checkpoint_model.pop(key)

        # 如果键中包含"relative_position_bias_table"
        if "relative_position_bias_table" in key:
            # 获取相对位置偏置表
            rel_pos_bias = checkpoint_model[key]
            # 获取源位置数和注意力头数
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            # 获取目标位置数和目标模型状态字典中的键的大小
            dst_num_pos, _ = model.state_dict()[key].size()
            # 获取目标补丁形状
            dst_patch_shape = model.patch_embed.patch_shape
            # 如果目标补丁的宽和高不相等
            if dst_patch_shape[0] != dst_patch_shape[1]:
                # 抛出未实现错误
                raise NotImplementedError()

    # 加载模型的状态字典
    load_state_dict(model, checkpoint_model, prefix="")

    # 返回加载后的模型
    return model
# 定义主函数
def main():
    # 获取命令行参数
    args = get_args()

    # 检查是否已微调
    is_finetuned = "ft1k" in args.hf_checkpoint_name
    # 检查模型是否为大模型
    is_large = "large" in args.hf_checkpoint_name

    # 若已微调
    if is_finetuned:
        # 需要将 Beit 的 data2vec_vision 转换为 HF 模型，需要将 modeling_finetune.py 复制到当前文件夹
        import modeling_finetune  # noqa: F401
    # 若未微调
    else:
        # 需要将 Beit 的 data2vec_vision 转换为 HF 模型，需要将 modeling_cyclical.py 复制到当前文件夹
        # 注意：目前我们只转换了下游模型而不是完整的预训练模型。这意味着在集成测试中，需要在以下行后添加 `return x`：
        # https://github.com/facebookresearch/data2vec_vision/blob/af9a36349aaed59ae66e69b5dabeef2d62fdc5da/beit/modeling_cyclical.py#L197
        import modeling_cyclical  # noqa: F401

    # 1. 创建模型配置
    config = Data2VecVisionConfig()
    if is_finetuned:
        # 配置微调的参数
        config.use_relative_position_bias = True
        config.use_shared_relative_position_bias = False
        config.use_mean_pooling = True
        config.num_labels = 1000

        # 加载 imagenet-1k-id2label.json 文件
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        # 配置预训练的参数
        config.use_relative_position_bias = False
        config.use_shared_relative_position_bias = True
        config.use_mean_pooling = False

    # 如果模型为大模型
    if is_large:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # 2. 加载 Beit 模型
    orig_model = load_beit_model(args, is_finetuned, is_large)
    orig_model.eval()

    # 3. 前向传播 Beit 模型
    image_processor = BeitImageProcessor(size=config.image_size, do_center_crop=False)
    image = Image.open("../../../../tests/fixtures/tests_samples/COCO/000000039769.png")
    encoding = image_processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    orig_args = (pixel_values,) if is_finetuned else (pixel_values, None)
    with torch.no_grad():
        orig_model_output = orig_model(*orig_args)

    # 4. 加载 HF Data2VecVision 模型
    if is_finetuned:
        hf_model = Data2VecVisionForImageClassification(config)
        hf_model.eval()
        has_lm_head = False
        hf_prefix = "data2vec_vision."
    else:
        hf_model = Data2VecVisionModel(config)
        hf_model.eval()
        has_lm_head = True
        hf_prefix = ""
    # 创建需要重命名的键值对列表，根据配置、HF 前缀和 LM 头信息
    rename_keys = create_rename_keys(config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    # 获取原始模型的状态字典
    state_dict = orig_model.state_dict()
    # 遍历重命名键值对列表，将原始模型状态字典中相应的键值对进行替换
    for src, dest in rename_keys:
        val = state_dict.pop(src)
        state_dict[dest] = val

    # 通过状态字典更新 HF QKV 模型
    read_in_q_k_v(state_dict, config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    # 加载模型状态字典到 HF 模型，strict=False 表示允许缺失或者多余的键
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    print("HF missing", missing_keys)
    print("HF unexpected_keys", unexpected_keys)

    # 5. 对 HF Data2VecVision 模型进行前向推断
    with torch.no_grad():
        hf_model_output = hf_model(pixel_values)

    # 根据是否 Fine-tuned 来选择 HF 模型输出是 logits 还是最后隐藏层状态
    hf_output = hf_model_output.logits if is_finetuned else hf_model_output.last_hidden_state

    # 6. 比较 HF 模型输出与原始模型输出的最大绝对差异
    max_absolute_diff = torch.max(torch.abs(hf_output - orig_model_output)).item()

    print(f"max_absolute_diff = {max_absolute_diff}")
    # 检查 HF 模型输出是否与原始模型输出近似相等，atol=1e-3 表示误差容限
    success = torch.allclose(hf_output, orig_model_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    # 7. 保存模型
    print(f"Saving to {args.hf_checkpoint_name}")
    hf_model.save_pretrained(args.hf_checkpoint_name)
    image_processor.save_pretrained(args.hf_checkpoint_name)
if __name__ == "__main__":
    # 如果当前模块被直接执行，调用主函数
    main()
    # 运行以下命令以转换检查点文件
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./pretrained_base.pt \
    #          --hf_checkpoint_name "./data2vec-vision-base"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./finetuned_base.pt \
    #          --hf_checkpoint_name "./data2vec-vision-base-ft1k"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./pretrained_large.pt \
    #          --hf_checkpoint_name "./data2vec-vision-large"
    #  python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #          --beit_checkpoint ./finetuned_large.pt \
    #          --hf_checkpoint_name "./data2vec-vision-large-ft1k"
```