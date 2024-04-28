# `.\transformers\models\trocr\convert_trocr_unilm_to_pytorch.py`

```
# 设置代码编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本发布
# 在遵守许可证的情况下才能使用本文件
# 可以在下面链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非依法要求或书面同意，否则将根据许可证分发的软件基于"原样"分发
# 没有任何明示或默示的担保或条件，包括但不限于隐含的担保或条件
# 请查看许可证以获取权限和限制的详细信息

"""Convert TrOCR checkpoints from the unilm repository."""

# 导入必要的包
import argparse
from pathlib import Path
import requests
import torch
from PIL import Image

# 导入 transformers 库相关内容
from transformers import (
    RobertaTokenizer,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
)
from transformers.utils import logging

# 设置日志级别为 info，并创建日志记录器
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 创建需要重命名的键值对列表（原始名称在左，我们的名称在右）
def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        # 编码器层：输出投影、2个前馈神经网络和2层 layernorm
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

    # 编码���的 cls 标记、位置嵌入和补丁嵌入
    # 扩展重命名键列表，将每对旧键和新键添加到列表中
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
    
    # 返回扩展后的重命名键列表
    return rename_keys
# 将每个编码器层的矩阵拆分为查询、键和值
def read_in_q_k_v(state_dict, encoder_config):
    # 遍历编码器的每一层
    for i in range(encoder_config.num_hidden_layers):
        # 获取查询、键和值的权重，不包含偏置
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        # 将权重分别赋给查询、键和值
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]


# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧的键值对，并放入新的键值对
    val = dct.pop(old)
    dct[new] = val


# 在 IAM 手写数据库的图像上验证我们的结果
def prepare_img(checkpoint_url):
    # 根据 checkpoint_url 的内容选择对应图像
    if "handwritten" in checkpoint_url:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-12.jpg" # have
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-10.jpg" # let
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"  #
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122.jpg"
    elif "printed" in checkpoint_url or "stage1" in checkpoint_url:
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    # 从 URL 中获取图像并转换为 RGB 格式
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im


# 转换 TR-OCR 模型检查点
@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型权重以适应我们的 VisionEncoderDecoderModel 结构。
    """
    # 基于 checkpoint_url 定义编码器和解码器配置
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    decoder_config = TrOCRConfig()

    # 根据 checkpoint_url 获取架构的尺寸
    if "base" in checkpoint_url:
        decoder_config.encoder_hidden_size = 768
    elif "large" in checkpoint_url:
        # 使用 ViT 大型编码器
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16
        decoder_config.encoder_hidden_size = 1024
    else:
        raise ValueError("应在检查点 URL 中找到 'base' 或 'large'")

    # 大印刷 + stage1 检查点使用正弦位置嵌入，之后没有层归一化
    # 检查 checkpoint_url 中是否包含特定字符串，根据条件设定 decoder_config 的一些属性
    if "large-printed" in checkpoint_url or "stage1" in checkpoint_url:
        decoder_config.tie_word_embeddings = False  # 不共享词嵌入层参数
        decoder_config.activation_function = "relu"  # 激活函数设为 ReLU
        decoder_config.max_position_embeddings = 1024  # 最大位置嵌入长度设为 1024
        decoder_config.scale_embedding = True  # 缩放嵌入层参数
        decoder_config.use_learned_position_embeddings = False  # 不使用学习的位置嵌入
        decoder_config.layernorm_embedding = False  # 不对嵌入进行层归一化处理

    # 加载 HuggingFace 模型
    encoder = ViTModel(encoder_config, add_pooling_layer=False)  # 初始化 ViT 编码器模型
    decoder = TrOCRForCausalLM(decoder_config)  # 初始化 TrOCR 解码器模型
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)  # 初始化视觉编码器-解码器模型
    model.eval()  # 将模型设置为评估模式

    # 加载原始模型的状态字典，并重命名一些键
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["model"]

    rename_keys = create_rename_keys(encoder_config, decoder_config)  # 根据编码器和解码器配置创建重命名键列表
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)  # 重命名状态字典中的键
    read_in_q_k_v(state_dict, encoder_config)  # 将 Q、K、V 参数读入状态字典中

    # 移除不需要的参数
    del state_dict["encoder.deit.head.weight"]  # 移除编码器头部权重参数
    del state_dict["encoder.deit.head.bias"]  # 移除编码器头部偏置参数
    del state_dict["decoder.version"]  # 移除解码器版本参数

    # 给解码器键添加前缀
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val  # 给解码器键添加前缀"decoder.model."
        else:
            state_dict[key] = val

    # 加载状态字典
    model.load_state_dict(state_dict)

    # 在图像上检查输出
    image_processor = ViTImageProcessor(size=encoder_config.image_size)  # 初始化 ViT 图像处理器
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")  # 使用 RoBERTa-large 分词器
    processor = TrOCRProcessor(image_processor, tokenizer)  # 初始化 TrOCR 处理器

    pixel_values = processor(images=prepare_img(checkpoint_url), return_tensors="pt").pixel_values  # 准备图像并获取像素值

    # 验证 logits
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])  # 初始化解码器输入 ID
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)  # 获取模型输出
    logits = outputs.logits  # 获取模型输出的 logits

    # 期望的 logits 形状
    expected_shape = torch.Size([1, 1, 50265])
    # 根据 checkpoint_url 设定预期的 logits 切片
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
    # 如果检查点URL中不包含"stage1"，则执行以下断言
    if "stage1" not in checkpoint_url:
        # 检查 logits 的形状是否符合预期
        assert logits.shape == expected_shape, "Shape of logits not as expected"
        # 检查 logits 的前10个元素是否与期望的切片接近
        assert torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-3), "First elements of logits not as expected"

    # 创建用于保存模型的文件夹，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定文件夹中
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存处理器的路径
    print(f"Saving processor to {pytorch_dump_folder_path}")
    # 将处理器保存到指定文件夹中
    processor.save_pretrained(pytorch_dump_folder_path)
# 如果该脚本作为主程序运行
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数，用于指定 checkpoint_url，默认为指定的 URL
    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    # 添加命令行参数，用于指定 pytorch_dump_folder_path，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数 convert_tr_ocr_checkpoint，传入 checkpoint_url 和 pytorch_dump_folder_path 参数
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```