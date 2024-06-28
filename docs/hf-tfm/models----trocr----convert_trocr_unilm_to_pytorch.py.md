# `.\models\trocr\convert_trocr_unilm_to_pytorch.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明，说明代码的版权归 The HuggingFace Inc. team. 所有
# 根据 Apache 许可证 2.0 版本使用此文件，详细信息可查阅许可证
"""从 unilm 代码库转换 TrOCR 检查点。"""


# 导入必要的库和模块
import argparse  # 用于处理命令行参数
from pathlib import Path  # 用于处理文件路径

import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 深度学习框架
from PIL import Image  # Python 图像处理库

# 导入 Transformers 库中的相关模块和类
from transformers import (
    RobertaTokenizer,  # RoBERTa 模型的分词器
    TrOCRConfig,  # TrOCR 模型的配置类
    TrOCRForCausalLM,  # TrOCR 用于有因果语言建模的模型类
    TrOCRProcessor,  # 处理 TrOCR 模型的数据处理器类
    VisionEncoderDecoderModel,  # 视觉编码器解码器模型
    ViTConfig,  # Vision Transformer (ViT) 的配置类
    ViTImageProcessor,  # 处理 ViT 模型输入图像的类
    ViTModel,  # Vision Transformer (ViT) 模型类
)
from transformers.utils import logging  # Transformers 库中的日志记录工具


logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# 定义函数：创建需要重命名的键值对列表（原始名称在左边，我们的名称在右边）
def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []  # 初始化空列表，用于存储重命名键值对
    for i in range(encoder_config.num_hidden_layers):
        # encoder 层的配置：输出投影，2 个前馈神经网络和 2 个层归一化
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    # encoder 的 cls token、位置嵌入和 patch 嵌入
    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )



# 将一组元组添加到 rename_keys 列表中，用于重新命名模型的特定参数路径
rename_keys.extend(
    [
        ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),  # 将 encoder.deit.cls_token 重命名为 encoder.embeddings.cls_token
        ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),  # 将 encoder.deit.pos_embed 重命名为 encoder.embeddings.position_embeddings
        ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),  # 将 encoder.deit.patch_embed.proj.weight 重命名为 encoder.embeddings.patch_embeddings.projection.weight
        ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),  # 将 encoder.deit.patch_embed.proj.bias 重命名为 encoder.embeddings.patch_embeddings.projection.bias
        ("encoder.deit.norm.weight", "encoder.layernorm.weight"),  # 将 encoder.deit.norm.weight 重命名为 encoder.layernorm.weight
        ("encoder.deit.norm.bias", "encoder.layernorm.bias"),  # 将 encoder.deit.norm.bias 重命名为 encoder.layernorm.bias
    ]
)

return rename_keys
# 将每个编码器层的权重矩阵分割为查询、键和值
def read_in_q_k_v(state_dict, encoder_config):
    # 遍历编码器的每一层
    for i in range(encoder_config.num_hidden_layers):
        # 提取查询、键和值的权重（没有偏置）
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        # 将权重分配给查询的权重矩阵
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        # 将权重分配给键的权重矩阵
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        # 将权重分配给值的权重矩阵
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]


# 将字典中的一个键名改为另一个键名
def rename_key(dct, old, new):
    # 弹出旧键名对应的值
    val = dct.pop(old)
    # 将该值插入到新的键名下
    dct[new] = val


# 在IAM手写数据库的图像上验证结果
def prepare_img(checkpoint_url):
    if "handwritten" in checkpoint_url:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
        # 下面的url是一些备用的图像链接，供验证使用
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-12.jpg" # have
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-10.jpg" # let
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"  #
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122.jpg"
    elif "printed" in checkpoint_url or "stage1" in checkpoint_url:
        # 如果是打印体或者是stage1的检查点，使用另一个图像链接
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    # 使用请求获取图像的原始数据流并将其转换为RGB格式的图像
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    将模型的权重复制/粘贴/调整到我们的VisionEncoderDecoderModel结构中。
    """
    # 根据checkpoint_url定义编码器和解码器的配置
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    decoder_config = TrOCRConfig()

    # 根据checkpoint_url选择架构的大小
    if "base" in checkpoint_url:
        decoder_config.encoder_hidden_size = 768
    elif "large" in checkpoint_url:
        # 使用ViT-large编码器
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16
        decoder_config.encoder_hidden_size = 1024
    else:
        # 如果checkpoint_url不包含'base'或'large'，则引发错误
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")

    # 对于large-printed + stage1的检查点，使用正弦位置嵌入，之后没有layernorm
    # 如果 checkpoint_url 中包含 "large-printed" 或者 "stage1"
    if "large-printed" in checkpoint_url or "stage1" in checkpoint_url:
        # 设置解码器配置的一些属性
        decoder_config.tie_word_embeddings = False
        decoder_config.activation_function = "relu"
        decoder_config.max_position_embeddings = 1024
        decoder_config.scale_embedding = True
        decoder_config.use_learned_position_embeddings = False
        decoder_config.layernorm_embedding = False

    # 加载 HuggingFace 模型，创建编码器和解码器
    encoder = ViTModel(encoder_config, add_pooling_layer=False)
    decoder = TrOCRForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # 加载原始模型的状态字典，并进行一些键名的重命名
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["model"]

    # 创建一个用于重命名键的列表，并应用到状态字典上
    rename_keys = create_rename_keys(encoder_config, decoder_config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, encoder_config)

    # 移除不需要的参数
    del state_dict["encoder.deit.head.weight"]
    del state_dict["encoder.deit.head.bias"]
    del state_dict["decoder.version"]

    # 对解码器键名添加前缀
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val
        else:
            state_dict[key] = val

    # 加载状态字典到模型中
    model.load_state_dict(state_dict)

    # 在图像上进行输出检查
    image_processor = ViTImageProcessor(size=encoder_config.image_size)
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
    processor = TrOCRProcessor(image_processor, tokenizer)

    # 准备图像并获取像素值
    pixel_values = processor(images=prepare_img(checkpoint_url), return_tensors="pt").pixel_values

    # 验证模型的输出
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits

    # 预期输出的形状应为 [1, 1, 50265]
    expected_shape = torch.Size([1, 1, 50265])

    # 根据 checkpoint_url 的不同类型设置预期的输出切片
    if "trocr-base-handwritten" in checkpoint_url:
        expected_slice = torch.tensor(
            [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
        )
    elif "trocr-large-handwritten" in checkpoint_url:
        expected_slice = torch.tensor(
            [-2.6437, -1.3129, -2.2596, -5.3455, 6.3539, 1.7604, 5.4991, 1.4702, 5.6113, 2.0170]
        )
    elif "trocr-base-printed" in checkpoint_url:
        expected_slice = torch.tensor(
            [-5.6816, -5.8388, 1.1398, -6.9034, 6.8505, -2.4393, 1.2284, -1.0232, -1.9661, -3.9210]
        )
    elif "trocr-large-printed" in checkpoint_url:
        expected_slice = torch.tensor(
            [-6.0162, -7.0959, 4.4155, -5.1063, 7.0468, -3.1631, 2.6466, -0.3081, -0.8106, -1.7535]
        )
    # 如果 checkpoint_url 中不包含 "stage1" 字符串，则执行以下断言
    if "stage1" not in checkpoint_url:
        # 检查 logits 的形状是否符合预期形状
        assert logits.shape == expected_shape, "Shape of logits not as expected"
        # 检查 logits 的前10个元素是否与预期的切片（expected_slice）非常接近，容差为 1e-3
        assert torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-3), "First elements of logits not as expected"

    # 根据给定的路径创建一个目录，如果目录已存在则不执行任何操作
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印消息，指示正在将模型保存到指定路径中
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 调用模型对象的 save_pretrained 方法，将模型保存到指定路径中
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印消息，指示正在将处理器保存到指定路径中
    print(f"Saving processor to {pytorch_dump_folder_path}")
    # 调用处理器对象的 save_pretrained 方法，将处理器保存到指定路径中
    processor.save_pretrained(pytorch_dump_folder_path)
# 如果这个脚本被直接运行（而不是被作为模块导入），则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：--checkpoint_url
    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )

    # 添加命令行参数：--pytorch_dump_folder_path
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the folder to output PyTorch model."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_tr_ocr_checkpoint，传递解析后的参数 checkpoint_url 和 pytorch_dump_folder_path
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```