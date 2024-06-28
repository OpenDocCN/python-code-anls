# `.\models\data2vec\convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py`

```py
#!/usr/bin/env python3
import argparse  # 导入命令行参数解析库
import json  # 导入 JSON 操作库

import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入 Hugging Face Hub 下载函数
from PIL import Image  # 导入 PIL 图像处理库
from timm.models import create_model  # 导入 timm 模型创建函数

from transformers import (  # 导入 transformers 库中的以下模块
    BeitImageProcessor,  # Beit 图像处理器
    Data2VecVisionConfig,  # Data2Vec 视觉配置类
    Data2VecVisionForImageClassification,  # Data2Vec 图像分类模型
    Data2VecVisionModel,  # Data2Vec 视觉模型
)


def create_rename_keys(config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec."):
    prefix = "backbone." if is_semantic else ""  # 根据是否语义化设置前缀

    rename_keys = []  # 初始化重命名键列表
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        # 编码器层：输出投影、2个前馈神经网络和2个层归一化
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm1.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_before.weight")
        )  # 添加权重归一化前的重命名键
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_before.bias"))  # 添加偏置归一化前的重命名键
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.weight")
        )  # 添加注意力投影层权重的重命名键
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"{hf_prefix}encoder.layer.{i}.attention.output.dense.bias")
        )  # 添加注意力投影层偏置的重命名键
        rename_keys.append(
            (f"{prefix}blocks.{i}.norm2.weight", f"{hf_prefix}encoder.layer.{i}.layernorm_after.weight")
        )  # 添加权重归一化后的重命名键
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"{hf_prefix}encoder.layer.{i}.layernorm_after.bias"))  # 添加偏置归一化后的重命名键
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.weight", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.weight")
        )  # 添加中间层第一个全连接层权重的重命名键
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.bias", f"{hf_prefix}encoder.layer.{i}.intermediate.dense.bias")
        )  # 添加中间层第一个全连接层偏置的重命名键
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"{hf_prefix}encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"{hf_prefix}encoder.layer.{i}.output.dense.bias"))  # 添加中间层第二个全连接层偏置的重命名键

    # projection layer + position embeddings
    # 投影层 + 位置嵌入
    rename_keys.extend(
        [
            (f"{prefix}cls_token", f"{hf_prefix}embeddings.cls_token"),  # 添加类别标记的重命名键
            (f"{prefix}patch_embed.proj.weight", f"{hf_prefix}embeddings.patch_embeddings.projection.weight"),  # 添加投影层权重的重命名键
            (f"{prefix}patch_embed.proj.bias", f"{hf_prefix}embeddings.patch_embeddings.projection.bias"),  # 添加投影层偏置的重命名键
        ]
    )
    # 如果具有语言模型头部
    if has_lm_head:
        # 将以下键值对添加到重命名列表，用于重命名模型的不同部分
        rename_keys.extend(
            [
                ("mask_token", f"{hf_prefix}embeddings.mask_token"),  # 重命名掩码标记
                (
                    "rel_pos_bias.relative_position_bias_table",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_bias_table",  # 重命名相对位置偏置表
                ),
                (
                    "rel_pos_bias.relative_position_index",
                    f"{hf_prefix}encoder.relative_position_bias.relative_position_index",  # 重命名相对位置索引
                ),
                ("norm.weight", "layernorm.weight"),  # 重命名归一化层权重
                ("norm.bias", "layernorm.bias"),  # 重命名归一化层偏置
            ]
        )
    # 如果是语义任务
    elif is_semantic:
        # 将以下键值对添加到重命名列表，用于语义分割分类头部的重命名
        rename_keys.extend(
            [
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),  # 重命名解码头部卷积层权重
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),  # 重命名解码头部卷积层偏置
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),  # 重命名辅助头部卷积层权重
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),  # 重命名辅助头部卷积层偏置
            ]
        )
    else:
        # 将以下键值对添加到重命名列表，用于常规的分类任务头部重命名
        rename_keys.extend(
            [
                ("fc_norm.weight", f"{hf_prefix}pooler.layernorm.weight"),  # 重命名全连接层归一化层权重
                ("fc_norm.bias", f"{hf_prefix}pooler.layernorm.bias"),  # 重命名全连接层归一化层偏置
                ("head.weight", "classifier.weight"),  # 重命名分类头部权重
                ("head.bias", "classifier.bias"),  # 重命名分类头部偏置
            ]
        )
    
    return rename_keys  # 返回包含所有重命名键值对的列表
# 读取输入的状态字典，根据配置和条件重新组织其内容
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False, hf_prefix="data2vec_vision."):
    # 遍历配置中指定数量的隐藏层
    for i in range(config.num_hidden_layers):
        # 根据语义和前缀确定当前层的前缀
        prefix = "backbone." if is_semantic else ""

        # 读取并移除当前层注意力机制的查询、键和值的权重
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        # 将查询权重放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        # 将查询偏置放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        # 将键权重放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        # 将值权重放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        # 将值偏置放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # 读取并移除当前层的 gamma_1 和 gamma_2
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        # 将 gamma_1 放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_1"] = gamma_1
        # 将 gamma_2 放入预定义的位置
        state_dict[f"{hf_prefix}encoder.layer.{i}.lambda_2"] = gamma_2

        # 如果没有语言模型头部，处理相对位置偏置表和索引
        if not has_lm_head:
            # 移除当前层的相对位置偏置表和索引
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")

            # 将相对位置偏置表放入预定义的位置
            state_dict[
                f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"
            ] = table
            # 将相对位置索引放入预定义的位置
            state_dict[
                f"{hf_prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"
            ] = index


# 获取命令行参数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        "Convert Data2VecVision to HF for image classification and pretraining", add_help=False
    )
    # 添加命令行参数：HF 检查点名称
    parser.add_argument("--hf_checkpoint_name", type=str)
    # 添加命令行参数：输入图像大小，默认为 224
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    # 添加命令行参数：BEiT 检查点路径，默认为空字符串
    parser.add_argument("--beit_checkpoint", default="", help="beit checkpoint")

    # 解析并返回命令行参数
    return parser.parse_args()


# 加载 BEiT 模型
def load_beit_model(args, is_finetuned, is_large):
    # 加载模型的状态字典，用于模型权重初始化
    def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
        # 用于存储找不到的键的列表
        missing_keys = []
        # 用于存储意外的键的列表
        unexpected_keys = []
        # 用于存储错误消息的列表
        error_msgs = []

        # 复制 state_dict 以便 _load_from_state_dict 可以修改它
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # 递归加载模型的每个模块的状态字典
        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix=prefix)

        # 根据指定的 ignore_missing 规则筛选出需要警告的缺失键和需要忽略的键
        warn_missing_keys = []
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

        # 更新 missing_keys 为 warn_missing_keys
        missing_keys = warn_missing_keys

        # 输出模型权重未初始化的警告信息
        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        # 输出未使用的预训练模型权重的信息
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        # 输出被忽略的模型权重未初始化的信息
        if len(ignore_missing_keys) > 0:
            print(
                "Ignored weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, ignore_missing_keys
                )
            )
        # 输出加载模型过程中的错误消息
        if len(error_msgs) > 0:
            print("\n".join(error_msgs))

    # 定义模型的关键字参数字典
    model_kwargs = {
        "pretrained": False,
        "use_shared_rel_pos_bias": True,
        "use_abs_pos_emb": False,
        "init_values": 0.1,
    }

    # 如果是微调过的模型，更新模型关键字参数字典
    if is_finetuned:
        model_kwargs.update(
            {
                "num_classes": 1000,
                "use_mean_pooling": True,
                "init_scale": 0.001,
                "use_rel_pos_bias": True,
            }
        )

    # 创建指定配置的模型实例
    model = create_model(
        "beit_large_patch16_224" if is_large else "beit_base_patch16_224",
        **model_kwargs,
    )
    # 获取模型的补丁嵌入层的补丁大小
    patch_size = model.patch_embed.patch_size
    # 计算窗口大小
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    # 加载 PyTorch 模型检查点
    checkpoint = torch.load(args.beit_checkpoint, map_location="cpu")

    # 打印加载检查点的信息
    print(f"Load ckpt from {args.beit_checkpoint}")
    # 初始化检查点模型
    checkpoint_model = None
    # 遍历指定的模型关键字列表，检查检查点中是否存在该关键字
    for model_key in ("model", "module"):
        # 如果找到了指定的模型关键字
        if model_key in checkpoint:
            # 从检查点中获取相应模型的状态字典
            checkpoint_model = checkpoint[model_key]
            # 打印加载状态字典的消息，指定加载的模型关键字
            print(f"Load state_dict by model_key = {model_key}")
            # 中断循环，已找到并加载了状态字典
            break

    # 获取所有状态字典键的列表
    all_keys = list(checkpoint_model.keys())
    # 遍历所有状态字典的键
    for key in all_keys:
        # 如果键包含"relative_position_index"字符串
        if "relative_position_index" in key:
            # 从状态字典中移除该键及其对应的值
            checkpoint_model.pop(key)

        # 如果键包含"relative_position_bias_table"字符串
        if "relative_position_bias_table" in key:
            # 获取相对位置偏置表的值
            rel_pos_bias = checkpoint_model[key]
            # 获取源和目标模型中的位置数量及注意力头数
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            # 检查目标模型的补丁形状是否为方形，若不是则抛出未实现的错误
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()

    # 使用加载的状态字典更新模型的参数
    load_state_dict(model, checkpoint_model, prefix="")

    # 返回更新后的模型
    return model
def main():
    # 获取命令行参数
    args = get_args()

    # 检查是否进行了微调
    is_finetuned = "ft1k" in args.hf_checkpoint_name
    # 检查模型是否为大模型
    is_large = "large" in args.hf_checkpoint_name

    if is_finetuned:
        # 如果进行了微调，导入微调模型的代码
        # 你需要将 https://github.com/facebookresearch/data2vec_vision/blob/main/beit/modeling_finetune.py
        # 复制到当前文件夹中。
        import modeling_finetune  # noqa: F401
    else:
        # 如果没有进行微调，导入周期性模型的代码
        # 你需要将 https://github.com/facebookresearch/data2vec_vision/blob/main/beit/modeling_cyclical.py
        # 复制到当前文件夹中。
        # 注意：目前我们只转换了下游模型而不是完整的预训练模型。这意味着在集成测试中，你需要在以下行之后添加 `return x`：
        # https://github.com/facebookresearch/data2vec_vision/blob/af9a36349aaed59ae66e69b5dabeef2d62fdc5da/beit/modeling_cyclical.py#L197
        import modeling_cyclical  # noqa: F401

    # 1. 创建模型配置
    config = Data2VecVisionConfig()
    if is_finetuned:
        # 如果进行了微调，设置特定的配置选项
        config.use_relative_position_bias = True
        config.use_shared_relative_position_bias = False
        config.use_mean_pooling = True
        config.num_labels = 1000

        # 下载并加载 ImageNet 类标签映射
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        # 如果没有进行微调，设置默认的配置选项
        config.use_relative_position_bias = False
        config.use_shared_relative_position_bias = True
        config.use_mean_pooling = False

    if is_large:
        # 如果模型是大模型，设置大模型特有的配置选项
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
        # 如果进行了微调，使用 Image Classification 的配置创建 HF Data2VecVision 模型
        hf_model = Data2VecVisionForImageClassification(config)
        hf_model.eval()
        has_lm_head = False
        hf_prefix = "data2vec_vision."
    else:
        # 如果没有进行微调，创建标准 HF Data2VecVision 模型
        hf_model = Data2VecVisionModel(config)
        hf_model.eval()
        has_lm_head = True
        hf_prefix = ""
    # 使用配置和前缀生成重命名键列表
    rename_keys = create_rename_keys(config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    # 获取原始模型的状态字典
    state_dict = orig_model.state_dict()
    # 根据重命名键，更新状态字典中的键名
    for src, dest in rename_keys:
        val = state_dict.pop(src)  # 移除原始键，并获取对应的数值
        state_dict[dest] = val  # 将数值与新的键名关联起来

    # 将更新后的状态字典读入查询-键-值功能
    read_in_q_k_v(state_dict, config, hf_prefix=hf_prefix, has_lm_head=has_lm_head)
    # 加载状态字典到 HF 模型中，允许缺失的键
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    print("HF missing", missing_keys)  # 打印缺失的键列表
    print("HF unexpected_keys", unexpected_keys)  # 打印意外的键列表

    # 5. Forward HF Data2VecVision model
    # 使用 torch.no_grad() 上下文，前向传播 HF 模型，计算像素值的输出
    with torch.no_grad():
        hf_model_output = hf_model(pixel_values)

    # 如果是微调状态，选择 logits；否则选择最后的隐藏状态
    hf_output = hf_model_output.logits if is_finetuned else hf_model_output.last_hidden_state

    # 6. Compare
    # 计算 HF 输出与原始模型输出的最大绝对差值
    max_absolute_diff = torch.max(torch.abs(hf_output - orig_model_output)).item()

    print(f"max_absolute_diff = {max_absolute_diff}")  # 打印最大绝对差值
    # 检查 HF 输出与原始模型输出是否接近，指定绝对容差
    success = torch.allclose(hf_output, orig_model_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")  # 打印比较结果
    if not success:
        raise Exception("Something went wRoNg")  # 如果输出不接近，抛出异常

    # 7. Save
    print(f"Saving to {args.hf_checkpoint_name}")  # 打印保存路径
    hf_model.save_pretrained(args.hf_checkpoint_name)  # 将 HF 模型保存到指定路径
    image_processor.save_pretrained(args.hf_checkpoint_name)  # 将图像处理器保存到同一路径
# 如果该脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
    # 运行以下命令将检查点转换为 PyTorch 格式：
    # python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #         --beit_checkpoint ./pretrained_base.pt \
    #         --hf_checkpoint_name "./data2vec-vision-base"
    # python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #         --beit_checkpoint ./finetuned_base.pt \
    #         --hf_checkpoint_name "./data2vec-vision-base-ft1k"
    # python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #         --beit_checkpoint ./pretrained_large.pt \
    #         --hf_checkpoint_name "./data2vec-vision-large"
    # python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
    #         --beit_checkpoint ./finetuned_large.pt \
    #         --hf_checkpoint_name "./data2vec-vision-large-ft1k"
```