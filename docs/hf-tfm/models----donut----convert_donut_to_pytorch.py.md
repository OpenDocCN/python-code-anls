# `.\models\donut\convert_donut_to_pytorch.py`

```py
# 设置脚本的编码格式为UTF-8，确保支持中文等Unicode字符
# 版权声明，声明使用Apache License Version 2.0许可证
# 可以在遵守许可证条件的前提下使用此文件
"""Convert Donut checkpoints using the original `donut-python` library. URL: https://github.com/clovaai/donut"""

# 导入命令行参数解析模块
import argparse

# 导入PyTorch库
import torch
# 导入datasets模块中的load_dataset函数，用于加载数据集
from datasets import load_dataset
# 导入donut模块中的DonutModel类
from donut import DonutModel

# 导入transformers库中的多个模块和类
from transformers import (
    DonutImageProcessor,
    DonutProcessor,
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizerFast,
)


# 定义函数，根据给定模型获取相关的配置信息
def get_configs(model):
    # 获取原始模型的配置信息
    original_config = model.config

    # 创建编码器的配置信息对象DonutSwinConfig
    encoder_config = DonutSwinConfig(
        image_size=original_config.input_size,  # 使用原始模型的输入尺寸作为图像尺寸
        patch_size=4,  # 指定图像块的大小为4
        depths=original_config.encoder_layer,  # 使用原始模型的编码器层数
        num_heads=[4, 8, 16, 32],  # 设定多头注意力机制的头数分别为4, 8, 16, 32
        window_size=original_config.window_size,  # 使用原始模型的窗口大小
        embed_dim=128,  # 设定嵌入维度为128
    )

    # 创建解码器的配置信息对象MBartConfig
    decoder_config = MBartConfig(
        is_decoder=True,  # 设置为解码器
        is_encoder_decoder=False,  # 不是编码器-解码器模型
        add_cross_attention=True,  # 添加交叉注意力
        decoder_layers=original_config.decoder_layer,  # 使用原始模型的解码器层数
        max_position_embeddings=original_config.max_position_embeddings,  # 使用原始模型的最大位置嵌入数
        vocab_size=len(
            model.decoder.tokenizer
        ),  # 设定词汇表大小为解码器的词汇量，XLMRobertaTokenizer添加了一些特殊标记，请查看hub上的repo（added_tokens.json）
        scale_embedding=True,  # 缩放嵌入
        add_final_layer_norm=True,  # 添加最终的层归一化
    )

    # 返回编码器和解码器的配置信息
    return encoder_config, decoder_config


# 定义函数，根据给定的名字对模型的键进行重命名处理
def rename_key(name):
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")  # 将名字中的"encoder.model"替换为"encoder"
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")  # 将名字中的"decoder.model"替换为"decoder"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")  # 将名字中的"patch_embed.proj"替换为"embeddings.patch_embeddings.projection"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")  # 将名字中的"patch_embed.norm"替换为"embeddings.norm"
    # 如果名称以 "encoder" 开头，则进行以下操作
    if name.startswith("encoder"):
        # 如果名称中包含 "layers"
        if "layers" in name:
            # 在名称前添加 "encoder."
            name = "encoder." + name
        # 如果名称中包含 "attn.proj"
        if "attn.proj" in name:
            # 将 "attn.proj" 替换为 "attention.output.dense"
            name = name.replace("attn.proj", "attention.output.dense")
        # 如果名称中包含 "attn" 且不包含 "mask"
        if "attn" in name and "mask" not in name:
            # 将 "attn" 替换为 "attention.self"
            name = name.replace("attn", "attention.self")
        # 如果名称中包含 "norm1"
        if "norm1" in name:
            # 将 "norm1" 替换为 "layernorm_before"
            name = name.replace("norm1", "layernorm_before")
        # 如果名称中包含 "norm2"
        if "norm2" in name:
            # 将 "norm2" 替换为 "layernorm_after"
            name = name.replace("norm2", "layernorm_after")
        # 如果名称中包含 "mlp.fc1"
        if "mlp.fc1" in name:
            # 将 "mlp.fc1" 替换为 "intermediate.dense"
            name = name.replace("mlp.fc1", "intermediate.dense")
        # 如果名称中包含 "mlp.fc2"
        if "mlp.fc2" in name:
            # 将 "mlp.fc2" 替换为 "output.dense"
            name = name.replace("mlp.fc2", "output.dense")

        # 如果名称是 "encoder.norm.weight"
        if name == "encoder.norm.weight":
            # 将名称替换为 "encoder.layernorm.weight"
            name = "encoder.layernorm.weight"
        # 如果名称是 "encoder.norm.bias"
        if name == "encoder.norm.bias":
            # 将名称替换为 "encoder.layernorm.bias"
            name = "encoder.layernorm.bias"

    # 返回修改后的名称
    return name
# 将给定的原始状态字典按键值进行迭代复制，以避免在迭代时修改字典结构
def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "qkv"
        if "qkv" in key:
            # 根据 "." 分割键名
            key_split = key.split(".")
            # 解析层号和块号
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            # 计算注意力机制的维度
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 根据键名中是否包含 "weight" 分别处理权重和偏置
            if "weight" in key:
                # 更新 query、key、value 的权重
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新 query、key、value 的偏置
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        # 如果键名中包含 "attn_mask" 或者是指定的其他键名
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            # HuggingFace 实现中不使用 attn_mask 缓冲区
            # 模型不使用编码器的最终 LayerNorms
            pass
        else:
            # 对于其余的键名，应用重命名函数，并保留其原始值
            orig_state_dict[rename_key(key)] = val

    # 返回处理后的原始状态字典
    return orig_state_dict


# 将 Donut 模型检查点转换为 HuggingFace 模型
def convert_donut_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # 加载原始模型
    original_model = DonutModel.from_pretrained(model_name).eval()

    # 加载 HuggingFace 模型
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 转换状态字典中的键值结构
    new_state_dict = convert_state_dict(state_dict, model)
    # 加载转换后的状态字典到 HuggingFace 模型
    model.load_state_dict(new_state_dict)

    # 在扫描文档上验证结果
    dataset = load_dataset("hf-internal-testing/example-documents")
    image = dataset["test"][0]["image"].convert("RGB")

    # 从模型名称加载 XLM-Roberta 分词器
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name, from_slow=True)
    # 创建 Donut 图像处理器，根据原始模型配置设定
    image_processor = DonutImageProcessor(
        do_align_long_axis=original_model.config.align_long_axis, size=original_model.config.input_size[::-1]
    )
    # 创建 Donut 处理器，整合图像处理器和分词器
    processor = DonutProcessor(image_processor, tokenizer)
    # 处理图像并获取像素值张量
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-docvqa"
    if model_name == "naver-clova-ix/donut-base-finetuned-docvqa":
        # 设置任务提示，包含用户输入的问题
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        # 设置问题内容为 "When is the coffee break?"
        question = "When is the coffee break?"
        # 替换任务提示中的占位符 {user_input} 为实际问题内容
        task_prompt = task_prompt.replace("{user_input}", question)
    
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-rvlcdip"
    elif model_name == "naver-clova-ix/donut-base-finetuned-rvlcdip":
        # 设置任务提示为 "<s_rvlcdip>"
        task_prompt = "<s_rvlcdip>"
    
    # 检查模型名称是否为以下任一
    elif model_name in [
        "naver-clova-ix/donut-base-finetuned-cord-v1",
        "naver-clova-ix/donut-base-finetuned-cord-v1-2560",
    ]:
        # 设置任务提示为 "<s_cord>"
        task_prompt = "<s_cord>"
    
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-cord-v2"
    elif model_name == "naver-clova-ix/donut-base-finetuned-cord-v2":
        # 设置任务提示为 "s_cord-v2>"
        task_prompt = "s_cord-v2>"
    
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-zhtrainticket"
    elif model_name == "naver-clova-ix/donut-base-finetuned-zhtrainticket":
        # 设置任务提示为 "<s_zhtrainticket>"
        task_prompt = "<s_zhtrainticket>"
    
    # 检查模型名称是否为以下任一
    elif model_name in ["naver-clova-ix/donut-proto", "naver-clova-ix/donut-base"]:
        # 如果以上条件均不满足，使用随机任务提示 "hello world"
        task_prompt = "hello world"
    
    else:
        # 如果模型名称不在支持列表中，抛出数值错误
        raise ValueError("Model name not supported")
    
    # 使用原始模型的解码器的标记器(tokenizer)处理任务提示，返回输入 ID（input_ids）张量
    prompt_tensors = original_model.decoder.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    # 使用原始模型的编码器的模型.patch_embed方法对像素值进行嵌入处理，获取原始补丁嵌入
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    # 使用当前模型的编码器的嵌入方法对像素值进行嵌入处理，获取当前模型的补丁嵌入和其他信息
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    # 断言原始补丁嵌入与当前模型的补丁嵌入在指定的误差范围内相似
    assert torch.allclose(original_patch_embed, patch_embeddings, atol=1e-3)

    # 验证编码器的隐藏状态是否相似
    original_last_hidden_state = original_model.encoder(pixel_values)
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    # 断言原始模型的最后隐藏状态与当前模型的最后隐藏状态在指定的误差范围内相似
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=1e-2)

    # 验证解码器的隐藏状态是否相似
    original_logits = original_model(pixel_values, prompt_tensors, None).logits
    logits = model(pixel_values, decoder_input_ids=prompt_tensors).logits
    # 断言原始模型的输出 logits 与当前模型的输出 logits 在指定的误差范围内相似
    assert torch.allclose(original_logits, logits, atol=1e-3)
    
    # 如果指定了 PyTorch 导出文件夹路径，则保存模型和处理器到该路径
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    
    # 如果指定了推送到 Hub，则将模型和处理器推送到 Hub
    if push_to_hub:
        # 使用模型名称的最后一部分推送到 Hub
        model.push_to_hub("nielsr/" + model_name.split("/")[-1], commit_message="Update model")
        processor.push_to_hub("nielsr/" + model_name.split("/")[-1], commit_message="Update model")
if __name__ == "__main__":
    # 如果这个模块被直接运行而非导入，则执行以下代码块

    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="naver-clova-ix/donut-base-finetuned-docvqa",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )

    # 添加可选的参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    # 添加标志参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行模型转换，传入解析后的参数
    convert_donut_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```