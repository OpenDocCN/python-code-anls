# `.\models\dpt\convert_dpt_to_pytorch.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明，指明此代码的版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何形式的担保或条件，包括但不限于适销性的担保或适用于特定目的的担保
# 请查阅许可证以获取具体的法律条款和限制条件
"""从原始代码库中转换 DPT 模型的检查点。URL: https://github.com/isl-org/DPT"""

# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import json  # 处理 JSON 格式数据的库
from pathlib import Path  # 处理文件路径的类

import requests  # 发送 HTTP 请求的库
import torch  # PyTorch 深度学习库
from huggingface_hub import cached_download, hf_hub_url  # 使用 HF Hub 的函数
from PIL import Image  # Python 图像处理库

# 导入 DPT 模型相关的类和函数
from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
from transformers.utils import logging  # 导入日志记录工具

# 设置日志记录的详细程度为信息级别
logging.set_verbosity_info()
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_dpt_config(checkpoint_url):
    config = DPTConfig()  # 创建 DPTConfig 实例

    # 根据检查点 URL 中的关键词调整配置
    if "large" in checkpoint_url:
        # 调整大模型的配置参数
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.backbone_out_indices = [5, 11, 17, 23]
        config.neck_hidden_sizes = [256, 512, 1024, 1024]
        expected_shape = (1, 384, 384)  # 预期输入形状为 (batch_size, height, width)

    if "ade" in checkpoint_url:
        # 根据检查点 URL 中的关键词调整配置，这里针对 ADE 模型的配置调整
        config.use_batch_norm_in_fusion_residual = True

        config.num_labels = 150  # ADE 模型的标签数目为 150
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        # 从 HF Hub 下载 ADE 模型的标签映射文件并加载为字典
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        id2label = {int(k): v for k, v in id2label.items()}  # 将键转换为整数类型
        config.id2label = id2label  # 设置 ID 到标签的映射
        config.label2id = {v: k for k, v in id2label.items()}  # 设置标签到 ID 的映射
        expected_shape = [1, 150, 480, 480]  # 预期输入形状为 (batch_size, num_labels, height, width)

    return config, expected_shape  # 返回配置对象和预期输入形状信息


def remove_ignore_keys_(state_dict):
    # 移除状态字典中指定的键
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    # 根据名称的特定规则重命名键名
    if (
        "pretrained.model" in name
        and "cls_token" not in name
        and "pos_embed" not in name
        and "patch_embed" not in name
    ):
        name = name.replace("pretrained.model", "dpt.encoder")
    if "pretrained.model" in name:
        name = name.replace("pretrained.model", "dpt.embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "patch_embeddings")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "proj" in name and "project" not in name:
        name = name.replace("proj", "projection")
    # 检查字符串 "blocks" 是否在变量 name 中
    if "blocks" in name:
        # 将变量 name 中的字符串 "blocks" 替换为 "layer"
        name = name.replace("blocks", "layer")
    
    # 检查字符串 "mlp.fc1" 是否在变量 name 中
    if "mlp.fc1" in name:
        # 将变量 name 中的字符串 "mlp.fc1" 替换为 "intermediate.dense"
        name = name.replace("mlp.fc1", "intermediate.dense")
    
    # 检查字符串 "mlp.fc2" 是否在变量 name 中
    if "mlp.fc2" in name:
        # 将变量 name 中的字符串 "mlp.fc2" 替换为 "output.dense"
        name = name.replace("mlp.fc2", "output.dense")
    
    # 检查字符串 "norm1" 是否在变量 name 中
    if "norm1" in name:
        # 将变量 name 中的字符串 "norm1" 替换为 "layernorm_before"
        name = name.replace("norm1", "layernorm_before")
    
    # 检查字符串 "norm2" 是否在变量 name 中
    if "norm2" in name:
        # 将变量 name 中的字符串 "norm2" 替换为 "layernorm_after"
        name = name.replace("norm2", "layernorm_after")
    
    # 检查字符串 "scratch.output_conv" 是否在变量 name 中
    if "scratch.output_conv" in name:
        # 将变量 name 中的字符串 "scratch.output_conv" 替换为 "head"
        name = name.replace("scratch.output_conv", "head")
    
    # 检查字符串 "scratch" 是否在变量 name 中
    if "scratch" in name:
        # 将变量 name 中的字符串 "scratch" 替换为 "neck"
        name = name.replace("scratch", "neck")
    
    # 检查字符串 "layer1_rn" 是否在变量 name 中
    if "layer1_rn" in name:
        # 将变量 name 中的字符串 "layer1_rn" 替换为 "convs.0"
        name = name.replace("layer1_rn", "convs.0")
    
    # 检查字符串 "layer2_rn" 是否在变量 name 中
    if "layer2_rn" in name:
        # 将变量 name 中的字符串 "layer2_rn" 替换为 "convs.1"
        name = name.replace("layer2_rn", "convs.1")
    
    # 检查字符串 "layer3_rn" 是否在变量 name 中
    if "layer3_rn" in name:
        # 将变量 name 中的字符串 "layer3_rn" 替换为 "convs.2"
        name = name.replace("layer3_rn", "convs.2")
    
    # 检查字符串 "layer4_rn" 是否在变量 name 中
    if "layer4_rn" in name:
        # 将变量 name 中的字符串 "layer4_rn" 替换为 "convs.3"
        name = name.replace("layer4_rn", "convs.3")
    
    # 检查字符串 "refinenet" 是否在变量 name 中
    if "refinenet" in name:
        # 提取 refinenet 后的数字，计算新的索引并替换字符串
        layer_idx = int(name[len("neck.refinenet") : len("neck.refinenet") + 1])
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx-4)}")
    
    # 检查字符串 "out_conv" 是否在变量 name 中
    if "out_conv" in name:
        # 将变量 name 中的字符串 "out_conv" 替换为 "projection"
        name = name.replace("out_conv", "projection")
    
    # 检查字符串 "resConfUnit1" 是否在变量 name 中
    if "resConfUnit1" in name:
        # 将变量 name 中的字符串 "resConfUnit1" 替换为 "residual_layer1"
        name = name.replace("resConfUnit1", "residual_layer1")
    
    # 检查字符串 "resConfUnit2" 是否在变量 name 中
    if "resConfUnit2" in name:
        # 将变量 name 中的字符串 "resConfUnit2" 替换为 "residual_layer2"
        name = name.replace("resConfUnit2", "residual_layer2")
    
    # 检查字符串 "conv1" 是否在变量 name 中
    if "conv1" in name:
        # 将变量 name 中的字符串 "conv1" 替换为 "convolution1"
        name = name.replace("conv1", "convolution1")
    
    # 检查字符串 "conv2" 是否在变量 name 中
    if "conv2" in name:
        # 将变量 name 中的字符串 "conv2" 替换为 "convolution2"
        name = name.replace("conv2", "convolution2")
    
    # 检查字符串 "pretrained.act_postprocess1.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess1.0.project.0" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess1.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.0.0"
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")
    
    # 检查字符串 "pretrained.act_postprocess2.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess2.0.project.0" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess2.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.1.0"
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")
    
    # 检查字符串 "pretrained.act_postprocess3.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess3.0.project.0" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess3.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.2.0"
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")
    
    # 检查字符串 "pretrained.act_postprocess4.0.project.0" 是否在变量 name 中
    if "pretrained.act_postprocess4.0.project.0" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess4.0.project.0" 替换为 "neck.reassemble_stage.readout_projects.3.0"
        name = name.replace("pretrained.act_postprocess4.0.project.0", "neck.reassemble_stage.readout_projects.3.0")
    
    # 检查字符串 "pretrained.act_postprocess1.3" 是否在变量 name 中
    if "pretrained.act_postprocess1.3" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess1.3" 替换为 "neck.reassemble_stage.layers.0.projection"
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    
    # 检查字符串 "pretrained.act_postprocess1.4" 是否在变量 name 中
    if "pretrained.act_postprocess1.4" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess1.4" 替换为 "neck.reassemble_stage.layers.0.resize"
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    
    # 检查字符串 "pretrained.act_postprocess2.3" 是否在变量 name 中
    if "pretrained.act_postprocess2.3" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess2.3" 替换为 "neck.reassemble_stage.layers.1.projection"
        name = name.replace("pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    
    # 检查字符串 "pretrained.act_postprocess2.4" 是否在变量 name 中
    if "pretrained.act_postprocess2.4" in name:
        # 将变量 name 中的字符串 "pretrained.act_postprocess2.4" 替换为 "neck.reassemble_stage.layers.1
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "pretrained.act_postprocess3.3" in name:
        name = name.replace("pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "pretrained.act_postprocess4.3" in name:
        name = name.replace("pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "pretrained.act_postprocess4.4" in name:
        name = name.replace("pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "pretrained" in name:
        name = name.replace("pretrained", "dpt")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "bn" in name:
        name = name.replace("bn", "batch_norm")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "head" in name:
        name = name.replace("head", "head.head")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "encoder.norm" in name:
        name = name.replace("encoder.norm", "layernorm")
    # 检查名称中是否包含特定字符串，然后替换成指定的新字符串
    if "auxlayer" in name:
        name = name.replace("auxlayer", "auxiliary_head.head")

    # 返回经过所有替换操作后的名称
    return name
# 将每个编码器层的权重矩阵分割为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config):
    # 遍历每个隐藏层
    for i in range(config.num_hidden_layers):
        # 读取输入投影层的权重和偏置（在timm中，这是单个矩阵和偏置）
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        
        # 添加查询（query）、键（key）和值（value）到状态字典中，顺序为查询、键、值
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


# 在一张可爱猫咪的图像上准备我们的结果验证
def prepare_img():
    # 图片URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests获取图像原始流，并打开为PIL图像
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name):
    """
    复制/粘贴/调整模型权重到我们的DPT结构。
    """

    # 基于URL定义DPT配置
    config, expected_shape = get_dpt_config(checkpoint_url)
    # 从URL加载原始的state_dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 移除特定的键
    remove_ignore_keys_(state_dict)
    # 重命名键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 读取qkv矩阵
    read_in_q_k_v(state_dict, config)

    # 根据URL加载HuggingFace模型
    model = DPTForSemanticSegmentation(config) if "ade" in checkpoint_url else DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 在一张图像上检查输出
    size = 480 if "ade" in checkpoint_url else 384
    image_processor = DPTImageProcessor(size=size)

    image = prepare_img()
    encoding = image_processor(image, return_tensors="pt")

    # 前向传播
    outputs = model(**encoding).logits if "ade" in checkpoint_url else model(**encoding).predicted_depth

    # 断言Logits
    expected_slice = torch.tensor([[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]])
    # 如果 checkpoint_url 字符串中包含 "ade"，则定义预期的切片张量
    if "ade" in checkpoint_url:
        expected_slice = torch.tensor([[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]])
    
    # 断言输出张量的形状与预期形状相等
    assert outputs.shape == torch.Size(expected_shape)
    
    # 断言输出张量的部分内容与预期的切片张量在数值上相近
    assert (
        torch.allclose(outputs[0, 0, :3, :3], expected_slice, atol=1e-4)
        if "ade" in checkpoint_url
        else torch.allclose(outputs[0, :3, :3], expected_slice)
    )
    
    # 打印信息表明一切正常
    print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 将图像处理器保存到指定路径
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 hub
    if push_to_hub:
        print("Pushing model to hub...")
        # 将模型推送到指定的 hub 仓库
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将图像处理器推送到指定的 hub 仓库
        image_processor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
if __name__ == "__main__":
    # 如果作为主程序执行，则开始解析命令行参数
    parser = argparse.ArgumentParser()
    
    # 添加必需的参数：checkpoint_url
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    
    # 添加可选的参数：pytorch_dump_folder_path
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    
    # 添加开关参数：push_to_hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    
    # 添加可选的参数：model_name
    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        required=False,
        help="Name of the model, in case you're pushing to the hub.",
    )

    # 解析命令行参数并将其存储在 args 对象中
    args = parser.parse_args()
    
    # 调用函数 convert_dpt_checkpoint，传入命令行参数
    convert_dpt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name)
```