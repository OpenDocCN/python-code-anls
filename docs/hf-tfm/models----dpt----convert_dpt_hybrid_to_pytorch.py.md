# `.\models\dpt\convert_dpt_hybrid_to_pytorch.py`

```py
# 设置脚本的编码格式为 UTF-8
# 版权声明，引用的 HuggingFace Inc. 的团队
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。查看许可证以获取具体语言的权限和限制
"""从原始存储库中转换 DPT 检查点。URL：https://github.com/isl-org/DPT"""


# 导入必要的库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
from pathlib import Path  # 用于处理文件路径

import requests  # 发送 HTTP 请求
import torch  # PyTorch 深度学习库
from huggingface_hub import cached_download, hf_hub_url  # 从 Hugging Face Hub 下载模型和数据
from PIL import Image  # Python Imaging Library，用于图像处理

# 导入 DPT 模型和相关工具
from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
from transformers.utils import logging  # 导入日志记录工具


# 设置日志输出级别为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# 根据给定的 checkpoint_url 返回相应的 DPTConfig 对象和预期的输出形状
def get_dpt_config(checkpoint_url):
    config = DPTConfig(embedding_type="hybrid")

    # 根据 URL 中是否包含 "large" 来设置不同的配置
    if "large" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.backbone_out_indices = [5, 11, 17, 23]
        config.neck_hidden_sizes = [256, 512, 1024, 1024]
        expected_shape = (1, 384, 384)

    # 根据 URL 中是否包含 "nyu" 或 "midas" 来设置不同的配置
    if "nyu" or "midas" in checkpoint_url:
        config.hidden_size = 768
        config.reassemble_factors = [1, 1, 1, 0.5]
        config.neck_hidden_sizes = [256, 512, 768, 768]
        config.num_labels = 150
        config.patch_size = 16
        expected_shape = (1, 384, 384)
        config.use_batch_norm_in_fusion_residual = False
        config.readout_type = "project"

    # 根据 URL 中是否包含 "ade" 来设置不同的配置
    if "ade" in checkpoint_url:
        config.use_batch_norm_in_fusion_residual = True
        config.hidden_size = 768
        config.reassemble_stage = [1, 1, 1, 0.5]
        config.num_labels = 150
        config.patch_size = 16
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        # 下载并加载 ADE20K 的标签映射文件
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        expected_shape = [1, 150, 480, 480]

    return config, expected_shape


# 移除 state_dict 中特定的键
def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名键名，将 "pretrained.model" 替换为 "dpt.encoder"
def rename_key(name):
    if (
        "pretrained.model" in name
        and "cls_token" not in name
        and "pos_embed" not in name
        and "patch_embed" not in name
    ):
        name = name.replace("pretrained.model", "dpt.encoder")
    # 检查字符串 "pretrained.model" 是否在变量 name 中
    if "pretrained.model" in name:
        # 将字符串 "pretrained.model" 替换为 "dpt.embeddings"
        name = name.replace("pretrained.model", "dpt.embeddings")

    # 检查字符串 "patch_embed" 是否在变量 name 中
    if "patch_embed" in name:
        # 将字符串 "patch_embed" 替换为空字符串 ""
        name = name.replace("patch_embed", "")

    # 检查字符串 "pos_embed" 是否在变量 name 中
    if "pos_embed" in name:
        # 将字符串 "pos_embed" 替换为 "position_embeddings"
        name = name.replace("pos_embed", "position_embeddings")

    # 检查字符串 "attn.proj" 是否在变量 name 中
    if "attn.proj" in name:
        # 将字符串 "attn.proj" 替换为 "attention.output.dense"
        name = name.replace("attn.proj", "attention.output.dense")

    # 检查字符串 "proj" 是否在变量 name 中，并且 "project" 不在 name 中
    if "proj" in name and "project" not in name:
        # 将字符串 "proj" 替换为 "projection"
        name = name.replace("proj", "projection")

    # 检查字符串 "blocks" 是否在变量 name 中
    if "blocks" in name:
        # 将字符串 "blocks" 替换为 "layer"
        name = name.replace("blocks", "layer")

    # 检查字符串 "mlp.fc1" 是否在变量 name 中
    if "mlp.fc1" in name:
        # 将字符串 "mlp.fc1" 替换为 "intermediate.dense"
        name = name.replace("mlp.fc1", "intermediate.dense")

    # 检查字符串 "mlp.fc2" 是否在变量 name 中
    if "mlp.fc2" in name:
        # 将字符串 "mlp.fc2" 替换为 "output.dense"
        name = name.replace("mlp.fc2", "output.dense")

    # 检查字符串 "norm1" 是否在变量 name 中，并且 "backbone" 不在 name 中
    if "norm1" in name and "backbone" not in name:
        # 将字符串 "norm1" 替换为 "layernorm_before"
        name = name.replace("norm1", "layernorm_before")

    # 检查字符串 "norm2" 是否在变量 name 中，并且 "backbone" 不在 name 中
    if "norm2" in name and "backbone" not in name:
        # 将字符串 "norm2" 替换为 "layernorm_after"
        name = name.replace("norm2", "layernorm_after")

    # 检查字符串 "scratch.output_conv" 是否在变量 name 中
    if "scratch.output_conv" in name:
        # 将字符串 "scratch.output_conv" 替换为 "head"
        name = name.replace("scratch.output_conv", "head")

    # 检查字符串 "scratch" 是否在变量 name 中
    if "scratch" in name:
        # 将字符串 "scratch" 替换为 "neck"
        name = name.replace("scratch", "neck")

    # 检查字符串 "layer1_rn" 是否在变量 name 中
    if "layer1_rn" in name:
        # 将字符串 "layer1_rn" 替换为 "convs.0"
        name = name.replace("layer1_rn", "convs.0")

    # 检查字符串 "layer2_rn" 是否在变量 name 中
    if "layer2_rn" in name:
        # 将字符串 "layer2_rn" 替换为 "convs.1"
        name = name.replace("layer2_rn", "convs.1")

    # 检查字符串 "layer3_rn" 是否在变量 name 中
    if "layer3_rn" in name:
        # 将字符串 "layer3_rn" 替换为 "convs.2"
        name = name.replace("layer3_rn", "convs.2")

    # 检查字符串 "layer4_rn" 是否在变量 name 中
    if "layer4_rn" in name:
        # 将字符串 "layer4_rn" 替换为 "convs.3"
        name = name.replace("layer4_rn", "convs.3")

    # 检查字符串 "refinenet" 是否在变量 name 中
    if "refinenet" in name:
        # 获取 refinenet 后面的数字索引
        layer_idx = int(name[len("neck.refinenet"): len("neck.refinenet") + 1])
        # 根据索引映射替换字符串，例如 refinenet4 替换为 fusion_stage.layers.0
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx-4)}")

    # 检查字符串 "out_conv" 是否在变量 name 中
    if "out_conv" in name:
        # 将字符串 "out_conv" 替换为 "projection"
        name = name.replace("out_conv", "projection")

    # 检查字符串 "resConfUnit1" 是否在变量 name 中
    if "resConfUnit1" in name:
        # 将字符串 "resConfUnit1" 替换为 "residual_layer1"
        name = name.replace("resConfUnit1", "residual_layer1")

    # 检查字符串 "resConfUnit2" 是否在变量 name 中
    if "resConfUnit2" in name:
        # 将字符串 "resConfUnit2" 替换为 "residual_layer2"
        name = name.replace("resConfUnit2", "residual_layer2")

    # 检查字符串 "conv1" 是否在变量 name 中
    if "conv1" in name:
        # 将字符串 "conv1" 替换为 "convolution1"
        name = name.replace("conv1", "convolution1")

    # 检查字符串 "conv2" 是否在变量 name 中
    if "conv2" in name:
        # 将字符串 "conv2" 替换为 "convolution2"
        name = name.replace("conv2", "convolution2")

    # 检查字符串 "pretrained.act_postprocess1.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess1.0.project.0" in name:
        # 将字符串 "pretrained.act_postprocess1.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.0.0"
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")

    # 检查字符串 "pretrained.act_postprocess2.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess2.0.project.0" in name:
        # 将字符串 "pretrained.act_postprocess2.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.1.0"
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")

    # 检查字符串 "pretrained.act_postprocess3.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess3.0.project.0" in name:
        # 将字符串 "pretrained.act_postprocess3.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.2.0"
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")

    # 检查字符串 "pretrained.act_postprocess4.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess4.0.project.0" in name:
        # 将字符串 "pretrained.act_postprocess4.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.3.0"
        name = name.replace("pretrained.act_postprocess4.0.project.0", "neck.reassemble_stage.readout_projects.3.0")

    # resize blocks
    # 检查字符串 "pretrained.act_postprocess1.3" 是否在变量 name 中
    if "pretrained.act_postprocess1.3" in name:
        # 将字符串 "pretrained.act_postprocess1.3" 替换为 "neck.reassemble_stage.layers.0.projection"
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    # 检查字符串 "pretrained.act_postprocess1.4" 是否在变量 name 中
    if "pretrained.act_postprocess1.4" in name:
        # 将字符串 "pretrained.act_postprocess1.4" 替换为 "neck.reassemble_stage.layers.0.resize"
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    # 检查字符串 "pretrained.act_postprocess2.3" 是否在变量 name 中
    if "pretrained.act_postprocess2.3" in name:
        # 将字符串 "pretrained.act_postprocess2.3" 替换为 "neck.reassemble_stage.layers.1.projection"
        name = name.replace("pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    # 检查字符串 "pretrained.act_postprocess2.4" 是否在变量 name 中
    if "pretrained.act_postprocess2.4" in name:
        # 将字符串 "pretrained.act_postprocess2.4" 替换为 "neck.reassemble_stage.layers.1.resize"
        name = name.replace("pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    # 检查字符串 "pretrained.act_postprocess3.3" 是否在变量 name 中
    if "pretrained.act_postprocess3.3" in name:
        # 将字符串 "pretrained.act_postprocess3.3" 替换为 "neck.reassemble_stage.layers.2.projection"
        name = name.replace("pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    # 检查字符串 "pretrained.act_postprocess4.3" 是否在变量 name 中
    if "pretrained.act_postprocess4.3" in name:
        # 将字符串 "pretrained.act_postprocess4.3" 替换为 "neck.reassemble_stage.layers.3.projection"
        name = name.replace("pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")
    # 检查字符串 "pretrained.act_postprocess4.4" 是否在变量 name 中
    if "pretrained.act_postprocess4.4" in name:
        # 将字符串 "pretrained.act_postprocess4.4" 替换为 "neck.reassemble_stage.layers.3.resize"
        name = name.replace("pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")
    # 检查字符串 "pretrained" 是否在变量 name 中
    if "pretrained" in name:
        # 将字符串 "pretrained" 替换为 "dpt"
        name = name.replace("pretrained", "dpt")
    # 检查字符串 "bn" 是否在变量 name 中
    if "bn" in name:
        # 将字符串 "bn" 替换为 "batch_norm"
        name = name.replace("bn", "batch_norm")
    # 检查字符串 "head" 是否在变量 name 中
    if "head" in name:
        # 将字符串 "head" 替换为 "head.head"
        name = name.replace("head", "head.head")
    # 检查字符串 "encoder.norm" 是否在变量 name 中
    if "encoder.norm" in name:
        # 将字符串 "encoder.norm" 替换为 "layernorm"
        name = name.replace("encoder.norm", "layernorm")
    # 检查字符串 "auxlayer" 是否在变量 name 中
    if "auxlayer" in name:
        # 将字符串 "auxlayer" 替换为 "auxiliary_head.head"
        name = name.replace("auxlayer", "auxiliary_head.head")
    # 检查字符串 "backbone" 是否在变量 name 中
    if "backbone" in name:
        # 将字符串 "backbone" 替换为 "backbone.bit.encoder"
        name = name.replace("backbone", "backbone.bit.encoder")

    # 检查字符串 ".." 是否在变量 name 中
    if ".." in name:
        # 将字符串 ".." 替换为 "."
        name = name.replace("..", ".")

    # 检查字符串 "stem.conv" 是否在变量 name 中
    if "stem.conv" in name:
        # 将字符串 "stem.conv" 替换为 "bit.embedder.convolution"
        name = name.replace("stem.conv", "bit.embedder.convolution")
    # 检查字符串 "blocks" 是否在变量 name 中
    if "blocks" in name:
        # 将字符串 "blocks" 替换为 "layers"
        name = name.replace("blocks", "layers")
    # 检查字符串 "convolution" 和 "backbone" 是否在变量 name 中
    if "convolution" in name and "backbone" in name:
        # 将字符串 "convolution" 替换为 "conv"
        name = name.replace("convolution", "conv")
    # 检查字符串 "layer" 和 "backbone" 是否在变量 name 中
    if "layer" in name and "backbone" in name:
        # 将字符串 "layer" 替换为 "layers"
        name = name.replace("layer", "layers")
    # 检查字符串 "backbone.bit.encoder.bit" 是否在变量 name 中
    if "backbone.bit.encoder.bit" in name:
        # 将字符串 "backbone.bit.encoder.bit" 替换为 "backbone.bit"
        name = name.replace("backbone.bit.encoder.bit", "backbone.bit")
    # 检查字符串 "embedder.conv" 是否在变量 name 中
    if "embedder.conv" in name:
        # 将字符串 "embedder.conv" 替换为 "embedder.convolution"
        name = name.replace("embedder.conv", "embedder.convolution")
    # 检查字符串 "backbone.bit.encoder.stem.norm" 是否在变量 name 中
    if "backbone.bit.encoder.stem.norm" in name:
        # 将字符串 "backbone.bit.encoder.stem.norm" 替换为 "backbone.bit.embedder.norm"
        name = name.replace("backbone.bit.encoder.stem.norm", "backbone.bit.embedder.norm")
    # 返回处理后的字符串 name
    return name
# 将每个编码器层的权重矩阵分解为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config):
    # 遍历每个编码器层
    for i in range(config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在timm中，这是一个单独的矩阵加偏置）
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # 将查询、键、值依次添加到状态字典中
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests获取图片的原始流，并由PIL库打开为图像对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name, show_prediction):
    """
    复制/粘贴/调整模型权重到我们的DPT结构中。
    """

    # 根据URL定义DPT配置
    config, expected_shape = get_dpt_config(checkpoint_url)
    # 从URL加载原始state_dict
    state_dict = torch.load(checkpoint_url, map_location="cpu")
    # 移除特定的键
    remove_ignore_keys_(state_dict)
    # 重命名键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 读取qkv矩阵
    read_in_q_k_v(state_dict, config)

    # 加载HuggingFace模型
    model = DPTForSemanticSegmentation(config) if "ade" in checkpoint_url else DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 在图片上进行输出检查
    size = 480 if "ade" in checkpoint_url else 384
    image_processor = DPTImageProcessor(size=size)

    # 准备图像
    image = prepare_img()
    encoding = image_processor(image, return_tensors="pt")

    # 前向传播
    outputs = model(**encoding).logits if "ade" in checkpoint_url else model(**encoding).predicted_depth
    # 如果需要展示预测结果
    if show_prediction:
        # 对模型输出进行插值，使其与原始图像大小一致，使用双三次插值，不对齐角落
        prediction = (
            torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=(image.size[1], image.size[0]),
                mode="bicubic",
                align_corners=False,
            )
            # 去除插值后的张量的单维度，将其转移到 CPU 上，并转换为 NumPy 数组
            .squeeze()
            .cpu()
            .numpy()
        )

        # 将 NumPy 数组转换为图像并显示
        Image.fromarray((prediction / prediction.max()) * 255).show()

    # 如果有指定的 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 创建保存模型的文件夹（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印模型保存的路径
        print(f"Saving model to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印图像处理器保存的路径
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 将图像处理器保存到指定路径
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 推送模型到指定 Hub 仓库
        model.push_to_hub("ybelkada/dpt-hybrid-midas")
        # 推送图像处理器到指定 Hub 仓库
        image_processor.push_to_hub("ybelkada/dpt-hybrid-midas")
if __name__ == "__main__":
    # 如果这个脚本是作为主程序运行

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    # 添加一个必需的参数 --checkpoint_url，用于指定原始 DPT 模型的下载链接

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    # 添加一个可选的参数 --pytorch_dump_folder_path，用于指定输出的 PyTorch 模型存储目录的路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    # 添加一个布尔参数 --push_to_hub，用于指示是否将转换后的模型推送到模型中心（hub）

    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        help="Name of the model, in case you're pushing to the hub.",
    )
    # 添加一个参数 --model_name，用于指定模型的名称，如果将其推送到模型中心（hub）

    parser.add_argument(
        "--show_prediction",
        action="store_true",
    )
    # 添加一个布尔参数 --show_prediction，用于指示是否显示模型的预测结果

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 变量中

    convert_dpt_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name, args.show_prediction
    )
    # 调用 convert_dpt_checkpoint 函数，传递命令行参数中解析得到的参数
```