# `.\models\dpt\convert_dpt_hybrid_to_pytorch.py`

```
# 设置文件编码为 utf-8
# 版权信息
# 根据 Apache License, Version 2.0 可以在指定网址获得许可证
# 除非依据适用法律或书面同意，根据许可证分发的软件均基于“原样”分发，不附带任何形式的担保或条件，无论是明示的还是隐含的
# 查看许可证以了解特定语言的详细权限和限制

"""从原始存储库转换 DPT 检查点。URL: https://github.com/isl-org/DPT"""

# 导入所需的库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image
from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
from transformers.utils import logging

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)

# 根据检查点 URL 获取 DPT 配置
def get_dpt_config(checkpoint_url):
    # 创建一个 DPTConfig 对象，设置 embedding_type 为 "hybrid"
    config = DPTConfig(embedding_type="hybrid")

    if "large" in checkpoint_url:
        # 如果 URL 中包含 "large"，则重新配置 config 对象
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.backbone_out_indices = [5, 11, 17, 23]
        config.neck_hidden_sizes = [256, 512, 1024, 1024]
        expected_shape = (1, 384, 384)

    if "nyu" or "midas" in checkpoint_url:
        # 如果 URL 中包含 "nyu" 或者 "midas"，则重新配置 config 对象
        config.hidden_size = 768
        config.reassemble_factors = [1, 1, 1, 0.5]
        config.neck_hidden_sizes = [256, 512, 768, 768]
        config.num_labels = 150
        config.patch_size = 16
        expected_shape = (1, 384, 384)
        config.use_batch_norm_in_fusion_residual = False
        config.readout_type = "project"

    if "ade" in checkpoint_url:
        # 如果 URL 中包含 "ade"，则重新配置 config 对象
        config.use_batch_norm_in_fusion_residual = True
        config.hidden_size = 768
        config.reassemble_stage = [1, 1, 1, 0.5]
        config.num_labels = 150
        config.patch_size = 16
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        # 加载并解析 ID 到标签的映射关系
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        expected_shape = [1, 150, 480, 480]

    return config, expected_shape

# 在状态字典中移除��定的键
def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 重命名键
def rename_key(name):
    if (
        "pretrained.model" in name
        and "cls_token" not in name
        and "pos_embed" not in name
        and "patch_embed" not in name
    ):
        name = name.replace("pretrained.model", "dpt.encoder")
    # 如果文件名中包含"pretrained.model"，则将其替换为"dpt.embeddings"
    if "pretrained.model" in name:
        name = name.replace("pretrained.model", "dpt.embeddings")
    
    # 如果文件名中包含"patch_embed"，则将其替换为空字符串
    if "patch_embed" in name:
        name = name.replace("patch_embed", "")
    
    # 如果文件名中包含"pos_embed"，则将其替换为"position_embeddings"
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    
    # 如果文件名中包含"attn.proj"，则将其替换为"attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    
    # 如果文件名中包含"proj"但不包含"project"，则将其替换为"projection"
    if "proj" in name and "project" not in name:
        name = name.replace("proj", "projection")
    
    # 如果文件名中包含"blocks"，则将其替换为"layer"
    if "blocks" in name:
        name = name.replace("blocks", "layer")
    
    # 如果文件名中包含"mlp.fc1"，则将其替换为"intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    
    # 如果文件名中包含"mlp.fc2"，则将其替换为"output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    
    # 如果文件名中包含"norm1"但不包含"backbone"，则将其替换为"layernorm_before"
    if "norm1" in name and "backbone" not in name:
        name = name.replace("norm1", "layernorm_before")
    
    # 如果文件名中包含"norm2"但不包含"backbone"，则将其替换为"layernorm_after"
    if "norm2" in name and "backbone" not in name:
        name = name.replace("norm2", "layernorm_after")
    
    # 如果文件名中包含"scratch.output_conv"，则将其替换为"head"
    if "scratch.output_conv" in name:
        name = name.replace("scratch.output_conv", "head")
    
    # 如果文件名中包含"scratch"，则将其替换为"neck"
    if "scratch" in name:
        name = name.replace("scratch", "neck")
    
    # 如果文件名中包含"layer1_rn"，则将其替换为"convs.0"
    if "layer1_rn" in name:
        name = name.replace("layer1_rn", "convs.0")
    
    # 如果文件名中包含"layer2_rn"，则将其替换为"convs.1"
    if "layer2_rn" in name:
        name = name.replace("layer2_rn", "convs.1")
    
    # 如果文件名中包含"layer3_rn"，则将其替换为"convs.2"
    if "layer3_rn" in name:
        name = name.replace("layer3_rn", "convs.2")
    
    # 如果文件名中包含"layer4_rn"，则将其替换为"convs.3"
    if "layer4_rn" in name:
        name = name.replace("layer4_rn", "convs.3")
    
    # 如果文件名中包含"refinenet"，则执行以下操作
    if "refinenet" in name:
        # 获取refinenet层的索引号
        layer_idx = int(name[len("neck.refinenet") : len("neck.refinenet") + 1])
        # 将refinenet层的索引号映射为fusion_stage.layers的索引号
        # 例如，4映射为0，3映射为1，2映射为2，1映射为3
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx-4)}")
    
    # 如果文件名中包含"out_conv"，则将其替换为"projection"
    if "out_conv" in name:
        name = name.replace("out_conv", "projection")
    
    # 如果文件名中包含"resConfUnit1"，则将其替换为"residual_layer1"
    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")
    
    # 如果文件名中包含"resConfUnit2"，则将其替换为"residual_layer2"
    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")
    
    # 如果文件名中包含"conv1"，则将其替换为"convolution1"
    if "conv1" in name:
        name = name.replace("conv1", "convolution1")
    
    # 如果文件名中包含"conv2"，则将其替换为"convolution2"
    if "conv2" in name:
        name = name.replace("conv2", "convolution2")
    
    # 如果文件名中包含"pretrained.act_postprocess1.0.project.0"，则将其替换为"neck.reassemble_stage.readout_projects.0.0"
    if "pretrained.act_postprocess1.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")
    
    # 如果文件名中包含"pretrained.act_postprocess2.0.project.0"，则将其替换为"neck.reassemble_stage.readout_projects.1.0"
    if "pretrained.act_postprocess2.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")
    
    # 如果文件名中包含"pretrained.act_postprocess3.0.project.0"，则将其替换为"neck.reassemble_stage.readout_projects.2.0"
    if "pretrained.act_postprocess3.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")
    
    # 如果文件名中包含"pretrained.act_postprocess4.0.project.0"，则将其替换为"neck.reassemble_stage.readout_projects.3.0"
    if "pretrained.act_postprocess4.0.project.0" in name:
        name = name.replace("pre
    # 检查名字中是否包含"pretrained.act_postprocess1.3"，如果是则替换成"neck.reassemble_stage.layers.0.projection"
    if "pretrained.act_postprocess1.3" in name:
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    # 检查名字中是否包含"pretrained.act_postprocess1.4"，如果是则替换成"neck.reassemble_stage.layers.0.resize"
    if "pretrained.act_postprocess1.4" in name:
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    # ... 后续的 if 语句依次类推，是一系列的字符串替换操作
    # 对替换操作后的名字进行清理，去掉多余的".."
    if ".." in name:
        name = name.replace("..", ".")
    # 对替换操作后的名字进行进一步的清理和规范化，以符合特定的命名约定
    # 最终返回处理后的名字
    return name
# we split up the matrix of each encoder layer into queries, keys and values
# 根据每个编码器层的矩阵分割成查询、键和数值

def read_in_q_k_v(state_dict, config):
    # iterate through the number of hidden layers in the configuration
    # 在配置中的隐藏层数范围内进行迭代
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        # 读取输入投影层的权重和偏置（在timm中，这是一个单矩阵和偏置）
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        # 接下来，按顺序将查询、键和数值添加到状态字典中
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[config.hidden_size : config.hidden_size * 2, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[config.hidden_size : config.hidden_size * 2]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]

# We will verify our results on an image of cute cats
# 我们将在一幅可爱猫咪的图像上验证我们的结果
def prepare_img():
    # specify image URL
    # 指定图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # read image data from the URL and open it as an image
    # 从 URL 读取图像数据并将其作为图像打开
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name, show_prediction):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration based on URL
    # 根据 URL 定义 DPT 配置
    config, expected_shape = get_dpt_config(checkpoint_url)
    # load original state_dict from URL
    # 从 URL 加载原始的状态字典
    state_dict = torch.load(checkpoint_url, map_location="cpu")
    # remove certain keys
    # 移除特定的键
    remove_ignore_keys_(state_dict)
    # rename keys
    # 重命名键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # read in qkv matrices
    # 读取 qkv 矩阵
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    # 加载 HuggingFace 模型
    model = DPTForSemanticSegmentation(config) if "ade" in checkpoint_url else DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Check outputs on an image
    # 在图像上检查输出
    size = 480 if "ade" in checkpoint_url else 384
    image_processor = DPTImageProcessor(size=size)

    image = prepare_img()
    encoding = image_processor(image, return_tensors="pt")

    # forward pass
    # 前向传播
    outputs = model(**encoding).logits if "ade" in checkpoint_url else model(**encoding).predicted_depth
    # 如果需要展示模型预测结果
    if show_prediction:
        # 对模型输出进行插值操作，将其调整为与原始图像相同大小的预测结果
        prediction = (
            torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=(image.size[1], image.size[0]),
                mode="bicubic",
                align_corners=False,
            )
            # 去除多余的维度并将数据转移到 CPU 上
            .squeeze()
            .cpu()
            .numpy()
        )
        
        # 将预测结果转换为图像并展示
        Image.fromarray((prediction / prediction.max()) * 255).show()

    # 如果指定了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 如果指定的路径不存在，则创建新的文件夹
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印提示信息，保存模型到指定路径
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印提示信息，保存图像处理器到指定路径
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 Hub 上
    if push_to_hub:
        # 将模型推送到指定的 Hub 上
        model.push_to_hub("ybelkada/dpt-hybrid-midas")
        # 将图像处理器推送到指定的 Hub 上
        image_processor.push_to_hub("ybelkada/dpt-hybrid-midas")
# 如果代码执行在主程序中
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_url",  # 参数名称
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",  # 默认值
        type=str,  # 参数类型
        help="URL of the original DPT checkpoint you'd like to convert.",  # 参数描述
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 参数名称
        default=None,  # 默认值
        type=str,  # 参数类型
        required=False,  # 是否必需
        help="Path to the output PyTorch model directory.",  # 参数描述
    )
    parser.add_argument(
        "--push_to_hub",  # 参数名称
        action="store_true",  # 动作类型
    )
    parser.add_argument(
        "--model_name",  # 参数名称
        default="dpt-large",  # 默认值
        type=str,  # 参数类型
        help="Name of the model, in case you're pushing to the hub.",  # 参数描述
    )
    parser.add_argument(
        "--show_prediction",  # 参数名称
        action="store_true",  # 动作类型
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，转换 DPT checkpoint
    convert_dpt_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name, args.show_prediction
    )
```