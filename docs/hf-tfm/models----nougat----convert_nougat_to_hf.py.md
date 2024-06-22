# `.\transformers\models\nougat\convert_nougat_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据Apache许可证2.0版的规定，您不得使用此文件，除非遵守该许可证。
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件，没有任何明示或暗示的担保或条件。
# 请查看特定语言所适用的特定语言，以及许可证下所限制的条件。
"""使用原始`nougat`库转换Nougat检查点。URL：https://github.com/facebookresearch/nougat/tree/main"""

# 导入必要的库
import argparse
import torch
from huggingface_hub import hf_hub_download
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.checkpoint import get_checkpoint
from PIL import Image
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

# 定义一个函数，用于获取编码器和解码器的配置
def get_configs(model):
    # 获取原始模型配置
    original_config = model.config
    # 设置编码器的配置
    encoder_config = DonutSwinConfig(
        image_size=original_config.input_size,
        patch_size=4,
        depths=original_config.encoder_layer,
        num_heads=[4, 8, 16, 32],
        window_size=original_config.window_size,
        embed_dim=128,
    )
    # 设置解码器的配置
    decoder_config = MBartConfig(
        is_decoder=True,
        is_encoder_decoder=False,
        add_cross_attention=True,
        decoder_layers=original_config.decoder_layer,
        max_position_embeddings=original_config.max_position_embeddings,
        vocab_size=len(
            model.decoder.tokenizer
        ),  # 词汇表大小为解码器的标记数，见hub上的repo(added_tokens.json)添加了几个特殊标记
        scale_embedding=True,
        add_final_layer_norm=True,
        tie_word_embeddings=False,
    )
    # 返回编码器和解码器配置
    return encoder_config, decoder_config

# 从transformers.models.donut.convert_donut_to_pytorch.rename_key中复制的函数
# 用于重命名键名
def rename_key(name):
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 检查变量名是否以"encoder"开头
    if name.startswith("encoder"):
        # 如果变量名包含"layers"，在前面添加"encoder."
        if "layers" in name:
            name = "encoder." + name
        # 如果变量名包含"attn.proj"，替换为"attention.output.dense"
        if "attn.proj" in name:
            name = name.replace("attn.proj", "attention.output.dense")
        # 如果变量名包含"attn"但不包含"mask"，替换为"attention.self"
        if "attn" in name and "mask" not in name:
            name = name.replace("attn", "attention.self")
        # 如果变量名包含"norm1"，替换为"layernorm_before"
        if "norm1" in name:
            name = name.replace("norm1", "layernorm_before")
        # 如果变量名包含"norm2"，替换为"layernorm_after"
        if "norm2" in name:
            name = name.replace("norm2", "layernorm_after")
        # 如果变量名包含"mlp.fc1"，替换为"intermediate.dense"
        if "mlp.fc1" in name:
            name = name.replace("mlp.fc1", "intermediate.dense")
        # 如果变量名包含"mlp.fc2"，替换为"output.dense"
        if "mlp.fc2" in name:
            name = name.replace("mlp.fc2", "output.dense")

        # 如果变量名为"encoder.norm.weight"，替换为"encoder.layernorm.weight"
        if name == "encoder.norm.weight":
            name = "encoder.layernorm.weight"
        # 如果变量名为"encoder.norm.bias"，替换为"encoder.layernorm.bias"
        if name == "encoder.norm.bias":
            name = "encoder.layernorm.bias"

    # 返回修改后的变量名
    return name
# 从transformers.models.donut.convert_donut_to_pytorch.convert_state_dict中复制过来的函数
def convert_state_dict(orig_state_dict, model):
    # 对原始状态字典的键进行深拷贝并遍历
    for key in orig_state_dict.copy().keys():
        # 弹出原始状态字典中的值
        val = orig_state_dict.pop(key)

        # 如果键中包含"qkv"
        if "qkv" in key:
            # 通过"."分割键名获取层号和块号
            key_split = key.split(".")
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            # 获取编码器中特定位置的注意力模块的维度
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键中包含"weight"
            if "weight" in key:
                # 将值按维度分割并赋给相应的查询、键、值权重
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
                # 将值按维度分割并赋给相应的查询、键、值偏置
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        # 如果键中包含"attn_mask"或键为"encoder.model.norm.weight"或"encoder.model.norm.bias"
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            # HuggingFace实现不使用attn_mask缓冲区，模型不使用最终的LayerNorms进行编码器
            pass
        else:
            # 转换键并赋予新的值
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_nougat_checkpoint(model_tag, pytorch_dump_folder_path=None, push_to_hub=False):
    # 加载原始模型
    checkpoint_path = get_checkpoint(None, model_tag)
    original_model = NougatModel.from_pretrained(checkpoint_path)
    original_model.eval()

    # 加载HuggingFace模型
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 转换状态字典并加载到模型中
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # 在PDF上验证结果
    filepath = hf_hub_download(repo_id="ysharma/nougat", filename="input/nougat.pdf", repo_type="space")
    images = rasterize_paper(pdf=filepath, return_pil=True)
    image = Image.open(images[0])

    # 加载tokenizer
    tokenizer_file = checkpoint_path / "tokenizer.json"
    tokenizer = NougatTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.pad_token = "<pad>"
    # 设置tokenizer的特殊token
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    # 设置tokenizer的模型最大长度
    tokenizer.model_max_length = original_model.config.max_length
    
    # 创建图像处理器
    size = {"height": original_model.config.input_size[0], "width": original_model.config.input_size[1]}
    image_processor = NougatImageProcessor(
        do_align_long_axis=original_model.config.align_long_axis,
        size=size,
    )
    # 创建NougatProcessor，用于图像和文本处理
    processor = NougatProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # 验证像素值
    pixel_values = processor(image, return_tensors="pt").pixel_values
    original_pixel_values = original_model.encoder.prepare_input(image).unsqueeze(0)
    assert torch.allclose(original_pixel_values, pixel_values)
    
    # 验证补丁嵌入
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    assert torch.allclose(original_patch_embed, patch_embeddings)
    
    # 验证编码器的隐藏状态
    original_last_hidden_state = original_model.encoder(pixel_values)
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=1e-2)
    
    # 验证译码器的嵌入
    original_embeddings = original_model.decoder.model.model.decoder.embed_tokens
    embeddings = model.decoder.model.decoder.embed_tokens
    assert torch.allclose(original_embeddings.weight, embeddings.weight, atol=1e-3)
    
    # 验证译码器的隐藏状态
    prompt = "hello world"
    decoder_input_ids = original_model.decoder.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    original_logits = original_model(
        image_tensors=pixel_values, decoder_input_ids=decoder_input_ids, attention_mask=decoder_attention_mask
    ).logits
    logits = model(
        pixel_values,
        decoder_input_ids=decoder_input_ids[:, :-1],
        decoder_attention_mask=decoder_attention_mask[:, :-1],
    ).logits
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
    generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    
    # 如果模型标记为"0.1.0-base"，则设置期望的生成结果
    if model_tag == "0.1.0-base":
        expected_generation = "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lblec"
    elif model_tag == "0.1.0-small":
        # 将预期生成的文本内容赋值给变量 expected_generation
        expected_generation = (
            "# Nougat: Neural Optical Understanding for Academic Documents\n\nLukas Blecher\n\nCorrespondence to: lble"
        )
    else:
        # 如果 model_tag 不是 "0.1.0-small"，则抛出 ValueError 异常
        raise ValueError(f"Unexpected model tag: {model_tag}")

    # 断言生成的文本与预期生成的文本相同
    assert generated == expected_generation
    # 打印输出 "Looks ok!"
    print("Looks ok!")

    # 如果 pytorch_dump_folder_path 不为 None
    if pytorch_dump_folder_path is not None:
        # 打印输出保存模型和处理器的路径
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果 push_to_hub 为 True
    if push_to_hub:
        # 定义一个字典，将 model_tag 映射为相应的模型名
        tag_to_name = {"0.1.0-base": "nougat-base", "0.1.0-small": "nougat-small"}
        # 获取 model_tag 对应的模型名
        model_name = tag_to_name[model_tag]

        # 将模型推送到模型中心的 facebook/{model_name}
        model.push_to_hub(f"facebook/{model_name}")
        # 将处理器推送到模型中心的 facebook/{model_name}
        processor.push_to_hub(f"facebook/{model_name}")
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_tag",
        default="0.1.0-base",
        required=False,
        type=str,
        choices=["0.1.0-base", "0.1.0-small"],
        help="Tag of the original model you'd like to convert.",
    )
    # 添加参数：输出 PyTorch 模型目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加参数：是否将转换后的模型和处理器推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入参数
    convert_nougat_checkpoint(args.model_tag, args.pytorch_dump_folder_path, args.push_to_hub)
```