# `.\models\deformable_detr\convert_deformable_detr_to_pytorch.py`

```
def read_in_q_k_v(state_dict):
    # 读取变量 state_dict 中的键，遍历每个键
    for key in state_dict.keys():
        # 检查键名是否包含 'cross_attn'，表示交叉注意力
        if "cross_attn" in key:
            # 如果是交叉注意力层，将键名中的 'cross_attn' 替换为 'encoder_attn'，表示编码器注意力
            state_dict[key.replace("cross_attn", "encoder_attn")] = state_dict.pop(key)
        # 检查键名是否包含 'query_embed'，表示查询嵌入
        elif "query_embed" in key:
            # 如果是查询嵌入，将键名中的 'query_embed' 替换为 'query_position_embeddings'，表示查询位置嵌入
            state_dict[key.replace("query_embed", "query_position_embeddings")] = state_dict.pop(key)
        # 检查键名是否包含 'linear2'，表示第二个线性层
        elif "linear2" in key:
            # 如果是第二个线性层，将键名中的 'linear2' 替换为 'fc2'，表示全连接层2
            state_dict[key.replace("linear2", "fc2")] = state_dict.pop(key)
        # 检查键名是否包含 'linear1'，表示第一个线性层
        elif "linear1" in key:
            # 如果是第一个线性层，将键名中的 'linear1' 替换为 'fc1'，表示全连接层1
            state_dict[key.replace("linear1", "fc1")] = state_dict.pop(key)
        # 检查键名是否包含 'norm3'，表示第三个归一化层
        elif "norm3" in key:
            # 如果是第三个归一化层，将键名中的 'norm3' 替换为 'final_layer_norm'，表示最终归一化层
            state_dict[key.replace("norm3", "final_layer_norm")] = state_dict.pop(key)
        # 检查键名是否包含 'norm2'，表示第二个归一化层
        elif "norm2" in key:
            # 如果是第二个归一化层，检查是否属于编码器
            if "encoder" in key:
                # 如果是编码器的归一化层，将键名中的 'norm2' 替换为 'final_layer_norm'，表示最终归一化层
                state_dict[key.replace("norm2", "final_layer_norm")] = state_dict.pop(key)
            else:
                # 如果不是编码器的归一化层，将键名中的 'norm2' 替换为 'self_attn_layer_norm'，表示自注意力归一化层
                state_dict[key.replace("norm2", "self_attn_layer_norm")] = state_dict.pop(key)
        # 检查键名是否包含 'norm1'，表示第一个归一化层
        elif "norm1" in key:
            # 如果是第一个归一化层，检查是否属于编码器
            if "encoder" in key:
                # 如果是编码器的归一化层，将键名中的 'norm1' 替换为 'self_attn_layer_norm'，表示自注意力归一化层
                state_dict[key.replace("norm1", "self_attn_layer_norm")] = state_dict.pop(key)
            else:
                # 如果不是编码器的归一化层，将键名中的 'norm1' 替换为 'encoder_attn_layer_norm'，表示编码器注意力归一化层
                state_dict[key.replace("norm1", "encoder_attn_layer_norm")] = state_dict.pop(key)
        # 检查键名是否包含 'transformer'，表示变换器
        elif "transformer" in key:
            # 如果是变换器层，将键名中的 'transformer' 替换为空字符串
            state_dict[key.replace("transformer.", "")] = state_dict.pop(key)
        # 检查键名是否包含 'backbone.0.body'，表示骨干网络
        elif "backbone.0.body" in key:
            # 如果是骨干网络，将键名中的 'backbone.0.body' 替换为 'backbone.conv_encoder.model'，表示卷积编码器模型
            state_dict[key.replace("backbone.0.body", "backbone.conv_encoder.model")] = state_dict.pop(key)
    # 返回处理后的 state_dict
    return state_dict
    # 遍历范围为0到5的整数，表示要处理解码器中的每一层
    for i in range(6):
        # 弹出存储在状态字典中的自注意力层的输入投影层的权重和偏置项
        in_proj_weight = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将权重按照特定规则分配给查询、键和值的投影层
        # 查询投影层权重：取in_proj_weight的前256行（对应查询部分），所有列
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        # 查询投影层偏置：取in_proj_bias的前256个元素（对应查询部分）
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        # 键投影层权重：取in_proj_weight的第256到511行（对应键部分），所有列
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        # 键投影层偏置：取in_proj_bias的第256到511个元素（对应键部分）
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        # 值投影层权重：取in_proj_weight的最后256行（对应值部分），所有列
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        # 值投影层偏置：取in_proj_bias的最后256个元素（对应值部分）
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
# 导入所需模块
from PIL import Image
import requests
import torch
import json
import logging

# 准备待处理的图像
def prepare_img():
    # 图像地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 模块获取图像，并封装成 PIL.Image 对象
    im = Image.open(requests.get(url, stream=True).raw)

    return im

# 用于模型权重转换的函数
@torch.no_grad()
def convert_deformable_detr_checkpoint(
    checkpoint_path,  # 源模型的路径
    single_scale,  # 是否为单尺度模型
    dilation,  # 是否使用空洞卷积
    with_box_refine,  # 是否进行边界框微调
    two_stage,  # 是否进行两阶段目标检测
    pytorch_dump_folder_path,  # 导出转换后模型的路径
    push_to_hub  # 是否推送模型到 HuggingFace Hub
):
    """
    将模型的权重复制/粘贴/调整以适应我们的 Deformable DETR 结构。
    """

    # 加载默认配置
    config = DeformableDetrConfig()
    # 设置配置属性
    if single_scale:
        config.num_feature_levels = 1
    config.dilation = dilation
    config.with_box_refine = with_box_refine
    config.two_stage = two_stage
    # 设置标签
    config.num_labels = 91
    # 从 HuggingFace Hub 下载标签文件，并加载为 id 到 label 的映射关系
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # 加载图像处理器
    image_processor = DeformableDetrImageProcessor(format="coco_detection")

    # 准备图像
    img = prepare_img()
    # 使用图像处理器将图像编码为 Tensor
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info("Converting model...")

    # 加载原始模型的权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 重命名键
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 对查询、键和值矩阵进行特殊处理
    read_in_q_k_v(state_dict)
    # 需要为基础模型的每个键添加前缀，因为头模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    # 创建 HuggingFace 模型并加载权重
    model = DeformableDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # ���动模型到 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 验证模型是否转换成功
    outputs = model(pixel_values.to(device))

    # 预期的 logits 和 boxes 的值
    expected_logits = torch.tensor(
        [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
    )
    expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]])
    
    # 如果使用单一尺度并且有扩张操作
    if single_scale and dilation:
        # 预期的logits值
        expected_logits = torch.tensor(
            [[-8.9652, -4.1074, -5.6635], [-9.0596, -4.9447, -6.6075], [-10.1178, -4.5275, -6.2671]]
        )
        # 预期的盒子坐标值
        expected_boxes = torch.tensor([[0.7665, 0.4130, 0.4769], [0.8364, 0.1841, 0.3391], [0.6261, 0.3895, 0.7978]])

    # 如果需要盒子细化
    if with_box_refine:
        # 预期的logits值
        expected_logits = torch.tensor(
            [[-8.8895, -5.4187, -6.8153], [-8.4706, -6.1668, -7.6184], [-9.0042, -5.5359, -6.9141]]
        )
        # 预期的盒子坐标值
        expected_boxes = torch.tensor([[0.7828, 0.2208, 0.4323], [0.0892, 0.5996, 0.1319], [0.5524, 0.6389, 0.8914]])

    # 如果需要盒子细化且是两阶段操作
    if with_box_refine and two_stage:
        # 预期的logits值
        expected_logits = torch.tensor(
            [[-6.7108, -4.3213, -6.3777], [-8.9014, -6.1799, -6.7240], [-6.9315, -4.4735, -6.2298]]
        )
        # 预期的盒子坐标值
        expected_boxes = torch.tensor([[0.2583, 0.5499, 0.4683], [0.7652, 0.9068, 0.4882], [0.5490, 0.2763, 0.0564]])

    # 打印logits的部分数据内容
    print("Logits:", outputs.logits[0, :3, :3])

    # 断言输出的logits值和预期的logits值在给定容差内相等
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    # 断言输出的盒子坐标值和预期的盒子坐标值在给定容差内相等
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

    # 打印提示信息
    print("Everything ok!")

    # 保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 推送到Hub
    if push_to_hub:
        model_name = "deformable-detr"
        model_name += "-single-scale" if single_scale else ""
        model_name += "-dc5" if dilation else ""
        model_name += "-with-box-refine" if with_box_refine else ""
        model_name += "-two-stage" if two_stage else ""
        print("Pushing model to hub...")
        model.push_to_hub(repo_path_or_name=model_name, organization="nielsr", commit_message="Add model")
# 如果当前脚本是直接执行的主脚本，而不是被导入的模块，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个命令行参数，用于指定要转换的 PyTorch 模型的检查点路径
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/deformable_detr/r50_deformable_detr-checkpoint.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    
    # 添加一个命令行参数，用于设置 config.num_features_levels = 1
    parser.add_argument("--single_scale", action="store_true", help="Whether to set config.num_features_levels = 1.")
    
    # 添加一个命令行参数，用于设置 config.dilation=True
    parser.add_argument("--dilation", action="store_true", help="Whether to set config.dilation=True.")
    
    # 添加一个命令行参数，用于设置 config.with_box_refine=True
    parser.add_argument("--with_box_refine", action="store_true", help="Whether to set config.with_box_refine=True.")
    
    # 添加一个命令行参数，用于设置 config.two_stage=True
    parser.add_argument("--two_stage", action="store_true", help="Whether to set config.two_stage=True.")
    
    # 添加一个命令行参数，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to output PyTorch model.",
    )
    
    # 添加一个命令行参数，用于是否将转换后的模型推送到 🤗 hub 上
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    
    # 调用 convert_deformable_detr_checkpoint 函数，将解析得到的参数传递给它
    convert_deformable_detr_checkpoint(
        args.checkpoint_path,
        args.single_scale,
        args.dilation,
        args.with_box_refine,
        args.two_stage,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
```