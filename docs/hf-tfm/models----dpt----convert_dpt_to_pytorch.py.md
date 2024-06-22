# `.\models\dpt\convert_dpt_to_pytorch.py`

```py
# 设置文件编码
# 版权声明
# 根据 Apache 许可证 2.0 进行授权，除非符合许可证，否则不得使用此文件
# 可以在以下网址获得该许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面协议同意，否则依照 "原样" 分发软件
# 没有任何明示或暗示的担保或条件，详情见许可证
# 在许可下许可的特定语言管理权限和限制
"""从原始代码库中转换 DPT 检查点。URL: https://github.com/isl-org/DPT"""

# 导入所需的库和模块
import argparse
import json
from pathlib import Path
import requests
import torch
# 从 huggingface_hub 模块中导入 cached_download, hf_hub_url 方法
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image
# 从 transformers 模块中导入 DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor 类
from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
# 从 transformers 模块中导入 logging 模块
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)


# 根据检查点 URL 获取 DPT 配置信息
def get_dpt_config(checkpoint_url):
    # 创建 DPTConfig 对象
    config = DPTConfig()

    if "large" in checkpoint_url:
        # 设置大型模型的隐藏层大小、中间层大小等参数
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.backbone_out_indices = [5, 11, 17, 23]
        config.neck_hidden_sizes = [256, 512, 1024, 1024]
        # 设置期望的形状
        expected_shape = (1, 384, 384)

    if "ade" in checkpoint_url:
        # 针对 ADE 的配置设置
        config.use_batch_norm_in_fusion_residual = True
        config.num_labels = 150
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        # 从 Hub 上下载 ADE 标签文件
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        # 设置 id2label 和 label2id 属性
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        # 设置期望的形状
        expected_shape = [1, 150, 480, 480]

    # 返回配置信息和期望形状
    return config, expected_shape


# 移除状态字典中指定的键
def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命��键
def rename_key(name):
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
    # 如果文件名中包含"blocks"，将其替换为"layer"
    if "blocks" in name:
        name = name.replace("blocks", "layer")
    # 如果文件名中包含"mlp.fc1"，将其替换为"intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果文件名中包含"mlp.fc2"，将其替换为"output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果文件名中包含"norm1"，将其替换为"layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果文件名中包含"norm2"，将其替换为"layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果文件名中包含"scratch.output_conv"，将其替换为"head"
    if "scratch.output_conv" in name:
        name = name.replace("scratch.output_conv", "head")
    # 如果文件名中包含"scratch"，将其替换为"neck"
    if "scratch" in name:
        name = name.replace("scratch", "neck")
    # 如果文件名中包含"layer1_rn"，将其替换为"convs.0"
    if "layer1_rn" in name:
        name = name.replace("layer1_rn", "convs.0")
    # 如果文件名中包含"layer2_rn"，将其替换为"convs.1"
    if "layer2_rn" in name:
        name = name.replace("layer2_rn", "convs.1")
    # 如果文件名中包含"layer3_rn"，将其替换为"convs.2"
    if "layer3_rn" in name:
        name = name.replace("layer3_rn", "convs.2")
    # 如果文件名中包含"layer4_rn"，将其替换为"convs.3"
    if "layer4_rn" in name:
        name = name.replace("layer4_rn", "convs.3")
    # 如果文件名中包含"refinenet"，根据特定规则替换为其他字符串
    if "refinenet" in name:
        # 获取refinenet后面的数字，然后根据规则替换为其他字符串
        layer_idx = int(name[len("neck.refinenet") : len("neck.refinenet") + 1])
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx-4)}")
    # 如果文件名中包含"out_conv"，将其替换为"projection"
    if "out_conv" in name:
        name = name.replace("out_conv", "projection")
    # 如果文件名中包含"resConfUnit1"，将其替换为"residual_layer1"
    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")
    # 如果文件名中包含"resConfUnit2"，将其替换为"residual_layer2"
    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")
    # 如果文件名中包含"conv1"，将其替换为"convolution1"
    if "conv1" in name:
        name = name.replace("conv1", "convolution1")
    # 如果文件名中包含"conv2"，将其替换为"convolution2"
    if "conv2" in name:
        name = name.replace("conv2", "convolution2")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess1.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess2.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess3.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess4.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess4.0.project.0", "neck.reassemble_stage.readout_projects.3.0")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess1.3" in name:
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess1.4" in name:
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess2.3" in name:
        name = name.replace("pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    # 如果文件名中包含特定字符串，根据规则替换为其他字符串
    if "pretrained.act_postprocess2.4" in name:
        name = name.replace("pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "pretrained.act_postprocess3.3" in name:
        name = name.replace("pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "pretrained.act_postprocess4.3" in name:
        name = name.replace("pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "pretrained.act_postprocess4.4" in name:
        name = name.replace("pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "pretrained" in name:
        name = name.replace("pretrained", "dpt")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "bn" in name:
        name = name.replace("bn", "batch_norm")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "head" in name:
        name = name.replace("head", "head.head")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "encoder.norm" in name:
        name = name.replace("encoder.norm", "layernorm")
    # 检查是否包含特定字符串，替换为对应的目标字符串
    if "auxlayer" in name:
        name = name.replace("auxlayer", "auxiliary_head.head")

    # 返回替换后的字符串
    return name
# 将每个编码器层的矩阵划分为查询(query)、键(keys)和值(values)
def read_in_q_k_v(state_dict, config):
    # 对于每个编码器层
    for i in range(config.num_hidden_layers):
        # 读取输入投影层的权重和偏置项（在timm中，这是一个矩阵和偏置项的组合）
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # 接下来，按照顺序向状态字典中添加查询(query)、键(keys)和值(values)
        # 查询(query)对应的是in_proj_weight矩阵的前config.hidden_size行，偏置项对应的是in_proj_bias的前config.hidden_size个元素
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        # 键(keys)对应的是in_proj_weight矩阵的从config.hidden_size到config.hidden_size * 2行，偏置项对应的是in_proj_bias的从config.hidden_size到config.hidden_size * 2个元素
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        # 值(values)对应的是in_proj_weight矩阵的从-config.hidden_size到最后一行，偏置项对应的是in_proj_bias的从-config.hidden_size到最后一个元素
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 我们将在一张可爱的猫的图片上验证结果
def prepare_img():
    # 图片URL地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests库发送GET请求，获取图片的二进制数据，然后使用PIL库打开图片
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 不计算梯度
@torch.no_grad()
# 将DPT来源的检查点模型的权重复制/粘贴/调整到我们的DPT结构中
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # 根据URL定义DPT配置和期望形状
    config, expected_shape = get_dpt_config(checkpoint_url)
    # 从URL加载原始的state_dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 移除特定的key
    remove_ignore_keys_(state_dict)
    # 重命名key
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 读取qkv矩阵
    read_in_q_k_v(state_dict, config)

    # 加载HuggingFace模型
    # 如果checkpoint_url中包含"ade"字符，则加载DPTForSemanticSegmentation模型，否则加载DPTForDepthEstimation模型
    model = DPTForSemanticSegmentation(config) if "ade" in checkpoint_url else DPTForDepthEstimation(config)
    # 加载权重到模型中
    model.load_state_dict(state_dict)
    # 设置为评估模式
    model.eval()

    # 在一张图片上检查输出结果
    size = 480 if "ade" in checkpoint_url else 384
    image_processor = DPTImageProcessor(size=size)

    image = prepare_img()
    # 将图片编码为对应模型所需的输入格式（PyTorch Tensor）
    encoding = image_processor(image, return_tensors="pt")

    # 模型的前向传播
    # 如果checkpoint_url中包含"ade"字符，则返回logits，否则返回predicted_depth
    outputs = model(**encoding).logits if "ade" in checkpoint_url else model(**encoding).predicted_depth

    # 断言输出的logits结果
    expected_slice = torch.tensor([[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]])
    # 如果checkpoint_url中包含"ade"，则定义一个预期的切片张量
        if "ade" in checkpoint_url:
            expected_slice = torch.tensor([[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]])
        # 断言输出的形状与预期形状相同
        assert outputs.shape == torch.Size(expected_shape)
        # 如果"ade"在checkpoint_url中，则比较指定切片与预期切片是否接近，否则比较输出和预期切片是否接近
        assert (
            torch.allclose(outputs[0, 0, :3, :3], expected_slice, atol=1e-4)
            if "ade" in checkpoint_url
            else torch.allclose(outputs[0, :3, :3], expected_slice)
        )
        # 打印"Looks ok!"
        print("Looks ok!")
    
    # 如果pytorch_dump_folder_path不为空
        if pytorch_dump_folder_path is not None:
            # 创建路径pytorch_dump_folder_path，如果存在则忽略
            Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
            # 打印"Saving model to {pytorch_dump_folder_path}"，保存模型到指定路径
            print(f"Saving model to {pytorch_dump_folder_path}")
            model.save_pretrained(pytorch_dump_folder_path)
            # 打印"Saving image processor to {pytorch_dump_folder_path}"，保存图像处理器到指定路径
            print(f"Saving image processor to {pytorch_dump_folder_path}")
    
    # 如果push_to_hub为真
        if push_to_hub:
            # 打印"Pushing model to hub..."，将模型推送到hub
            print("Pushing model to hub...")
            model.push_to_hub(
                repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
                organization="nielsr",
                commit_message="Add model",
                use_temp_dir=True,
            )
            # 将图像处理器推送到hub
            image_processor.push_to_hub(
                repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
                organization="nielsr",
                commit_message="Add image processor",
                use_temp_dir=True,
            )
# 如果当前脚本被直接执行，而非被导入其他模块，那么执行以下代码
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        required=False,
        help="Name of the model, in case you're pushing to the hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_dpt_checkpoint函数，传入解析后的参数
    convert_dpt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name)
```