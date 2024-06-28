# `.\models\nougat\convert_nougat_to_hf.py`

```
# 设置编码为 UTF-8，确保可以正确处理中文等特殊字符
# 版权声明，指明本代码的版权归属于 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本使用本代码，详细条款可参考许可证链接
"""Convert Nougat checkpoints using the original `nougat` library. URL:
https://github.com/facebookresearch/nougat/tree/main"""
# 导入 argparse 模块，用于处理命令行参数
import argparse

# 导入 torch 库
import torch
# 导入 hf_hub_download 函数，用于从 Hugging Face Hub 下载模型
from huggingface_hub import hf_hub_download
# 导入 NougatModel 类，用于加载 Nougat 模型
from nougat import NougatModel
# 导入 rasterize_paper 函数，用于将数据转换为光栅化的图像数据
from nougat.dataset.rasterize import rasterize_paper
# 导入 get_checkpoint 函数，用于获取检查点
from nougat.utils.checkpoint import get_checkpoint
# 导入 Image 类，用于处理图像
from PIL import Image
# 导入 transformers 库的多个类和函数
from transformers import (
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    NougatImageProcessor,
    NougatProcessor,
    NougatTokenizerFast,
    VisionEncoderDecoderModel,
)

# 定义函数 get_configs，用于根据给定模型获取编码器和解码器的配置
def get_configs(model):
    # 获取原始模型的配置
    original_config = model.config

    # 定义编码器的配置，使用 DonutSwinConfig 类
    encoder_config = DonutSwinConfig(
        image_size=original_config.input_size,
        patch_size=4,
        depths=original_config.encoder_layer,
        num_heads=[4, 8, 16, 32],
        window_size=original_config.window_size,
        embed_dim=128,
    )
    
    # 定义解码器的配置，使用 MBartConfig 类
    decoder_config = MBartConfig(
        is_decoder=True,
        is_encoder_decoder=False,
        add_cross_attention=True,
        decoder_layers=original_config.decoder_layer,
        max_position_embeddings=original_config.max_position_embeddings,
        vocab_size=len(
            model.decoder.tokenizer
        ),  # 根据模型的解码器的 tokenizer 获得词汇表大小
        scale_embedding=True,
        add_final_layer_norm=True,
        tie_word_embeddings=False,
    )

    # 返回编码器和解码器的配置
    return encoder_config, decoder_config

# 定义函数 rename_key，用于重命名模型中的特定键名，以便与 PyTorch 模型兼容
# 这些名称更改主要用于适应不同框架的不同命名习惯
def rename_key(name):
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 检查字符串 name 是否以 "encoder" 开头
    if name.startswith("encoder"):
        # 如果字符串 name 中包含 "layers"
        if "layers" in name:
            # 在字符串 name 前加上 "encoder."
            name = "encoder." + name
        # 如果字符串 name 中包含 "attn.proj"
        if "attn.proj" in name:
            # 将字符串 name 中的 "attn.proj" 替换为 "attention.output.dense"
            name = name.replace("attn.proj", "attention.output.dense")
        # 如果字符串 name 中包含 "attn" 且不包含 "mask"
        if "attn" in name and "mask" not in name:
            # 将字符串 name 中的 "attn" 替换为 "attention.self"
            name = name.replace("attn", "attention.self")
        # 如果字符串 name 中包含 "norm1"
        if "norm1" in name:
            # 将字符串 name 中的 "norm1" 替换为 "layernorm_before"
            name = name.replace("norm1", "layernorm_before")
        # 如果字符串 name 中包含 "norm2"
        if "norm2" in name:
            # 将字符串 name 中的 "norm2" 替换为 "layernorm_after"
            name = name.replace("norm2", "layernorm_after")
        # 如果字符串 name 中包含 "mlp.fc1"
        if "mlp.fc1" in name:
            # 将字符串 name 中的 "mlp.fc1" 替换为 "intermediate.dense"
            name = name.replace("mlp.fc1", "intermediate.dense")
        # 如果字符串 name 中包含 "mlp.fc2"
        if "mlp.fc2" in name:
            # 将字符串 name 中的 "mlp.fc2" 替换为 "output.dense"
            name = name.replace("mlp.fc2", "output.dense")

        # 如果字符串 name 等于 "encoder.norm.weight"
        if name == "encoder.norm.weight":
            # 将字符串 name 替换为 "encoder.layernorm.weight"
            name = "encoder.layernorm.weight"
        # 如果字符串 name 等于 "encoder.norm.bias"
        if name == "encoder.norm.bias":
            # 将字符串 name 替换为 "encoder.layernorm.bias"
            name = "encoder.layernorm.bias"

    # 返回经过处理的字符串 name
    return name
# 从 transformers.models.donut.convert_donut_to_pytorch.convert_state_dict 复制的函数，用于将原始状态字典转换为 PyTorch 模型的状态字典
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的副本的键
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "qkv"
        if "qkv" in key:
            # 拆分键名为列表
            key_split = key.split(".")
            # 获取层编号和块编号
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            # 获取当前自注意力的维度
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键名包含 "weight"
            if "weight" in key:
                # 更新状态字典，设置查询权重的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                # 更新状态字典，设置键权重的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                # 更新状态字典，设置值权重的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新状态字典，设置查询偏置的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                # 更新状态字典，设置键偏置的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                # 更新状态字典，设置值偏置的新键值对
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        # 如果键名中包含 "attn_mask" 或者是特定的键名列表
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            # HuggingFace 实现不使用 attn_mask 缓冲区，且模型不使用编码器的最终 LayerNorm
            # 跳过处理这些键
            pass
        else:
            # 使用自定义的函数处理键名，然后更新状态字典
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict


# 根据模型标签和可能的 PyTorch 导出文件夹路径，以及是否推送到 Hub，转换 Nougat 检查点
def convert_nougat_checkpoint(model_tag, pytorch_dump_folder_path=None, push_to_hub=False):
    # 获取检查点路径
    checkpoint_path = get_checkpoint(None, model_tag)
    # 从预训练模型路径加载原始模型
    original_model = NougatModel.from_pretrained(checkpoint_path)
    # 将原始模型设置为评估模式
    original_model.eval()

    # 加载 HuggingFace 模型的编码器和解码器配置
    encoder_config, decoder_config = get_configs(original_model)
    # 创建 DonutSwinModel 编码器和 MBartForCausalLM 解码器
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    # 创建 VisionEncoderDecoderModel 模型，设置编码器和解码器
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    # 将模型设置为评估模式
    model.eval()

    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 将原始状态字典转换为新的状态字典
    new_state_dict = convert_state_dict(state_dict, model)
    # 使用新的状态字典加载模型的参数
    model.load_state_dict(new_state_dict)

    # 在 PDF 上验证结果
    filepath = hf_hub_download(repo_id="ysharma/nougat", filename="input/nougat.pdf", repo_type="space")
    # 将 PDF 渲染为图像，并返回 PIL 图像列表
    images = rasterize_paper(pdf=filepath, return_pil=True)
    # 打开第一张图像
    image = Image.open(images[0])

    # 加载 NougatTokenizerFast，设置 tokenizer 文件路径和填充标记
    tokenizer_file = checkpoint_path / "tokenizer.json"
    tokenizer = NougatTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.pad_token = "<pad>"
    # 设置 tokenizer 的特殊符号
    tokenizer.bos_token = "<s>"  # 开始符号
    tokenizer.eos_token = "</s>"  # 结束符号
    tokenizer.unk_token = "<unk>"  # 未知符号
    # 设置 tokenizer 的最大模型长度为原始模型的最大长度
    tokenizer.model_max_length = original_model.config.max_length

    # 创建图像处理器对象，配置对齐长轴和大小
    size = {"height": original_model.config.input_size[0], "width": original_model.config.input_size[1]}
    image_processor = NougatImageProcessor(
        do_align_long_axis=original_model.config.align_long_axis,
        size=size,
    )
    # 创建处理器对象，整合图像处理器和 tokenizer
    processor = NougatProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 验证像素值
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # 准备输入图像的像素值并展开为张量
    original_pixel_values = original_model.encoder.prepare_input(image).unsqueeze(0)

    # 断言检查原始像素值与处理后的像素值是否相等
    assert torch.allclose(original_pixel_values, pixel_values)

    # 验证补丁嵌入
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    # 计算模型的补丁嵌入和补丁嵌入器的结果
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    # 断言检查原始补丁嵌入和模型补丁嵌入是否相等
    assert torch.allclose(original_patch_embed, patch_embeddings)

    # 验证编码器隐藏状态
    original_last_hidden_state = original_model.encoder(pixel_values)
    # 计算模型的编码器最后隐藏状态
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    # 断言检查原始编码器隐藏状态和模型编码器隐藏状态是否相等，容忍度为 1e-2
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=1e-2)

    # 注意：原始模型在解码器的嵌入中不使用绑定权重
    # 检查原始模型和当前模型的解码器嵌入权重是否相等，容忍度为 1e-3
    original_embeddings = original_model.decoder.model.model.decoder.embed_tokens
    embeddings = model.decoder.model.decoder.embed_tokens
    assert torch.allclose(original_embeddings.weight, embeddings.weight, atol=1e-3)

    # 验证解码器隐藏状态
    prompt = "hello world"
    # 使用原始模型的 tokenizer 对提示进行编码
    decoder_input_ids = original_model.decoder.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    # 计算原始模型和当前模型的 logits
    original_logits = original_model(
        image_tensors=pixel_values, decoder_input_ids=decoder_input_ids, attention_mask=decoder_attention_mask
    ).logits
    logits = model(
        pixel_values,
        decoder_input_ids=decoder_input_ids[:, :-1],
        decoder_attention_mask=decoder_attention_mask[:, :-1],
    ).logits
    # 断言检查原始 logits 和当前 logits 是否相等，容忍度为 1e-3
    assert torch.allclose(original_logits, logits, atol=1e-3)

    # 验证生成结果
    outputs = model.generate(
        pixel_values,
        min_length=1,
        max_length=30,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[
            [tokenizer.unk_token_id],
        ],
        return_dict_in_generate=True,
        do_sample=False,
    )
    # 解码生成的文本并跳过特殊符号
    generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    # 如果模型版本标签是 "0.1.0-base"，则验证生成的文本是否符合预期
    if model_tag == "0.1.0-base":
        expected_generation = "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lblec"
    # 如果模型标签为 "0.1.0-small"，设置期望生成的文本内容
    elif model_tag == "0.1.0-small":
        expected_generation = (
            "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lble"
        )
    else:
        # 如果模型标签不是已知的版本，则抛出值错误异常
        raise ValueError(f"Unexpected model tag: {model_tag}")

    # 断言生成的文本与期望的生成文本相等，用于验证生成结果是否符合预期
    assert generated == expected_generation
    # 打印确认信息，表示生成的文本符合预期
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型和处理器到指定路径的消息
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 定义模型标签到 Hub 仓库名称的映射
        tag_to_name = {"0.1.0-base": "nougat-base", "0.1.0-small": "nougat-small"}
        # 获取当前模型标签对应的 Hub 仓库名称
        model_name = tag_to_name[model_tag]

        # 将模型推送到 Facebook Hub 中对应的仓库
        model.push_to_hub(f"facebook/{model_name}")
        # 将处理器推送到 Facebook Hub 中对应的仓库
        processor.push_to_hub(f"facebook/{model_name}")
if __name__ == "__main__":
    # 如果作为主程序执行，则开始执行以下代码

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_tag",
        default="0.1.0-base",
        required=False,
        type=str,
        choices=["0.1.0-base", "0.1.0-small"],
        help="Tag of the original model you'd like to convert.",
    )
    # 添加一个必需的参数 --model_tag，用于指定要转换的原始模型的标签

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个参数 --pytorch_dump_folder_path，用于指定输出的 PyTorch 模型目录的路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )
    # 添加一个参数 --push_to_hub，是一个布尔标志，用于指示是否将转换后的模型和处理器推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_nougat_checkpoint(args.model_tag, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_nougat_checkpoint，传入解析后的参数 args 中的相关信息作为参数
```