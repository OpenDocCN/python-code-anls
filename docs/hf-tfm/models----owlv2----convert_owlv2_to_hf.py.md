# `.\transformers\models\owlv2\convert_owlv2_to_hf.py`

```
# 设置编码为 UTF-8
# 版权声明及许可证信息
# 这段代码的功能是从原始仓库中转换 OWLv2 检查点
# 请参考以下链接获取更多信息：https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
import argparse  # 导入解析命令行参数的模块
import collections  # 导入 collections 模块
import os  # 导入操作系统相关功能的模块

import jax  # 导入 JAX 库
import jax.numpy as jnp  # 导入 JAX 库中的 NumPy 接口
import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库
from flax.training import checkpoints  # 导入 flax 库中的 checkpoints 模块
from huggingface_hub import hf_hub_download  # 导入 huggingface_hub 库中的 hf_hub_download 函数
from PIL import Image  # 导入 PIL 库中的 Image 模块

from transformers import (  # 从 transformers 库中导入以下模块和类
    CLIPTokenizer,  # 导入 CLIPTokenizer 类
    Owlv2Config,  # 导入 Owlv2Config 类
    Owlv2ForObjectDetection,  # 导入 Owlv2ForObjectDetection 类
    Owlv2ImageProcessor,  # 导入 Owlv2ImageProcessor 类
    Owlv2Processor,  # 导入 Owlv2Processor 类
    Owlv2TextConfig,  # 导入 Owlv2TextConfig 类
    Owlv2VisionConfig,  # 导入 Owlv2VisionConfig 类
)
from transformers.utils import logging  # 从 transformers 库中导入 logging 模块

# 设置日志输出级别为 info
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)


def get_owlv2_config(model_name):
    # 根据模型名获取 OWLv2 配置信息
    if "large" in model_name:
        # 如果模型名中包含 "large"
        # 设置大型模型的参数
        image_size = 1008
        patch_size = 14
        vision_hidden_size = 1024
        vision_intermediate_size = 4096
        vision_num_hidden_layers = 24
        vision_num_attention_heads = 16
        projection_dim = 768
        text_hidden_size = 768
        text_intermediate_size = 3072
        text_num_attention_heads = 12
        text_num_hidden_layers = 12
    else:
        # 如果模型名中不包含 "large"
        # 设置小型模型的参数
        image_size = 960
        patch_size = 16
        vision_hidden_size = 768
        vision_intermediate_size = 3072
        vision_num_hidden_layers = 12
        vision_num_attention_heads = 12
        projection_dim = 512
        text_hidden_size = 512
        text_intermediate_size = 2048
        text_num_attention_heads = 8
        text_num_hidden_layers = 12

    # 构建视觉配置对象
    vision_config = Owlv2VisionConfig(
        patch_size=patch_size,
        image_size=image_size,
        hidden_size=vision_hidden_size,
        num_hidden_layers=vision_num_hidden_layers,
        intermediate_size=vision_intermediate_size,
        num_attention_heads=vision_num_attention_heads,
    )
    # 构建文本配置对象
    text_config = Owlv2TextConfig(
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
    )

    # 构建 OWLv2 配置对象
    config = Owlv2Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=projection_dim,
    )

    return config


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []  # 存储扁平化后的字典项
```  
    # 遍历参数字典中的键值对
    for k, v in params.items():
        # 如果有父键存在，则拼接新键，并使用指定的分隔符
        new_key = parent_key + sep + k if parent_key else k

        # 如果值是可变映射类型（字典），则递归展开该嵌套字典
        if isinstance(v, collections.MutableMapping):
            # 将展开后的键值对列表扩展到items中
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            # 如果值不是可变映射类型，则将新键值对添加到items中
            items.append((new_key, v))
    
    # 返回展开后的结果字典
    return dict(items)
# 在这里我们列出所有需要重命名的键（原始名称在左边，我们的名称在右边）
def create_rename_keys(config, model_name):
    # 创建一个空列表，用于存储重命名的键值对
    rename_keys = []

    # 格式化关闭（关闭 black 格式化）
    # CLIP 视觉编码器
    # 将原始键值对应的新名称添加到重命名列表中
    rename_keys.append(("backbone/clip/visual/class_embedding", "owlv2.vision_model.embeddings.class_embedding"))
    rename_keys.append(("backbone/clip/visual/conv1/kernel", "owlv2.vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/positional_embedding", "owlv2.vision_model.embeddings.position_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/scale", "owlv2.vision_model.pre_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/bias", "owlv2.vision_model.pre_layernorm.bias"))

    rename_keys.append(("backbone/clip/visual/ln_post/scale", "owlv2.vision_model.post_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_post/bias", "owlv2.vision_model.post_layernorm.bias"))

    # CLIP 文本编码器
    rename_keys.append(("backbone/clip/text/token_embedding/embedding", "owlv2.text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("backbone/clip/text/positional_embedding", "owlv2.text_model.embeddings.position_embedding.weight"))

    rename_keys.append(("backbone/clip/text/ln_final/scale", "owlv2.text_model.final_layer_norm.weight"))
    rename_keys.append(("backbone/clip/text/ln_final/bias", "owlv2.text_model.final_layer_norm.bias"))

    # 对数尺度
    rename_keys.append(("backbone/clip/logit_scale", "owlv2.logit_scale"))

    # 投影头
    rename_keys.append(("backbone/clip/text/text_projection/kernel", "owlv2.text_projection.weight"))

    # 类别和框头
    rename_keys.append(("backbone/merged_class_token/scale", "layer_norm.weight"))
    rename_keys.append(("backbone/merged_class_token/bias", "layer_norm.bias"))
    rename_keys.append(("class_head/Dense_0/kernel", "class_head.dense0.weight"))
    rename_keys.append(("class_head/Dense_0/bias", "class_head.dense0.bias"))
    rename_keys.append(("class_head/logit_shift/kernel", "class_head.logit_shift.weight"))
    rename_keys.append(("class_head/logit_scale/kernel", "class_head.logit_scale.weight"))
    rename_keys.append(("class_head/logit_scale/bias", "class_head.logit_scale.bias"))
    rename_keys.append(("class_head/logit_shift/bias", "class_head.logit_shift.bias"))
    rename_keys.append(("obj_box_head/Dense_0/kernel", "box_head.dense0.weight"))
    rename_keys.append(("obj_box_head/Dense_0/bias", "box_head.dense0.bias"))
    rename_keys.append(("obj_box_head/Dense_1/kernel", "box_head.dense1.weight"))
    rename_keys.append(("obj_box_head/Dense_1/bias", "box_head.dense1.bias"))
    rename_keys.append(("obj_box_head/Dense_2/kernel", "box_head.dense2.weight"))
    rename_keys.append(("obj_box_head/Dense_2/bias", "box_head.dense2.bias"))

    # 目标头（仅适用于 v2）

    # 返回重命名的键列表
    return rename_keys
    # 如果模型名称中包含字符串 "v2"，则执行以下操作
    if "v2" in model_name:
        # 将指定键值对添加到 rename_keys 列表中，用于重命名模型参数
        rename_keys.append(("objectness_head/Dense_0/kernel", "objectness_head.dense0.weight"))
        rename_keys.append(("objectness_head/Dense_0/bias", "objectness_head.dense0.bias"))
        rename_keys.append(("objectness_head/Dense_1/kernel", "objectness_head.dense1.weight"))
        rename_keys.append(("objectness_head/Dense_1/bias", "objectness_head.dense1.bias"))
        rename_keys.append(("objectness_head/Dense_2/kernel", "objectness_head.dense2.weight"))
        rename_keys.append(("objectness_head/Dense_2/bias", "objectness_head.dense2.bias"))
    
    # 返回 rename_keys 列表，其中包含用于重命名模型参数的键值对
    return rename_keys
# 重命名字典中的键，并根据配置进行相应的形状调整
def rename_and_reshape_key(dct, old, new, config):
    # 弹出旧键的值
    val = dct.pop(old)

    # 如果新键包含 "out_proj"、"v_proj"、"k_proj"、"q_proj"，并且包含 "vision"，则调整值的形状
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    # 如果新键包含 "out_proj"、"v_proj"、"k_proj"、"q_proj"，并且包含 "text"，则调整值的形状
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    # 如果新键包含 "patch_embedding"，则转置值的形状
    if "patch_embedding" in new:
        print("Reshaping patch embedding... for", new)
        val = val.transpose(3, 2, 0, 1)
    # 如果新键以 "weight" 结尾，并且不包含 "position_embedding" 和 "token_embedding"，则转置值的形状
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    # 如果新键以 "bias" 结尾，则调整值的形状
    if new.endswith("bias"):
        val = val.reshape(-1)

    # 将调整后的值转换为 Torch 张量，并存入字典中
    dct[new] = torch.from_numpy(np.array(val))


# 禁用 Torch 的梯度追踪
@torch.no_grad()
# 将权重从 Flax 格式转换为 OWL-ViT 结构
def convert_owlv2_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our OWL-ViT structure.
    """
    # 获取 OWL-ViT 模型的配置
    config = get_owlv2_config(model_name)

    # 获取检查点中的变量
    variables = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    variables = variables["params"] if "v2" in model_name else variables["optimizer"]["target"]
    # 将 Flax 参数转换为 PyTorch 参数
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    state_dict = flatten_nested_dict(flax_params)

    # 生成重命名键列表
    rename_keys = create_rename_keys(config, model_name)
    # 遍历重命名键列表，对状态字典中的键进行重命名和形状调整
    for src, dest in rename_keys:
        rename_and_reshape_key(state_dict, src, dest, config)

    # 加载 HuggingFace 模型
    model = Owlv2ForObjectDetection(config)
    # 加载模型参数，严格模式关闭，允许缺失键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 断言确实缺失的键为 ["owlv2.visual_projection.weight"]
    assert missing_keys == ["owlv2.visual_projection.weight"]
    # 断言没有意外的键
    assert unexpected_keys == []
    # 设置模型为评估模式
    model.eval()

    # 初始化图像处理器
    size = {"height": config.vision_config.image_size, "width": config.vision_config.image_size}
    image_processor = Owlv2ImageProcessor(size=size)
    # 初始化分词器
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!", model_max_length=16)
    # 初始化处理器
    processor = Owlv2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # 验证像素值和输入 ID
    # 下载并加载原始像素值
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="owlvit_pixel_values_960.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath).permute(0, 3, 1, 2)

    # 下载并加载原始输入 ID
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="owlv2_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath).squeeze()

    # 加载并处理示例图像
    filepath = hf_hub_download(repo_id="adirik/OWL-ViT", repo_type="space", filename="assets/astronaut.png")
    image = Image.open(filepath)
    # 定义图像中的文本
    texts = [["face", "rocket", "nasa badge", "star-spangled banner"]]
    # 使用processor处理输入文本和图像数据，返回PyTorch张量
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # 如果模型名称中不包含"large"
    if "large" not in model_name:
        # 断言处理后的像素值与原始像素值十分接近
        assert torch.allclose(inputs.pixel_values, original_pixel_values.float(), atol=1e-6)
    # 断言处理后的输入标识与原始输入标识十分接近
    assert torch.allclose(inputs.input_ids[:4, :], original_input_ids[:4, :], atol=1e-6)

    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型处理输入
        outputs = model(**inputs)
        # 获取模型输出的logits
        logits = outputs.logits
        # 获取预测框的坐标
        pred_boxes = outputs.pred_boxes
        # 获取目标检测的置信度logits

    else:
        # 如果不进入上面的if条件语句，则输出提示信息
        print("Model converted without verifying logits")

    # 如果指定了模型保存路径
    if pytorch_dump_folder_path is not None:
        # 输出提示信息
        print("Saving model and processor locally...")
        # 如果保存路径不存在则创建文件夹
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 保存processor到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要push到hub
    if push_to_hub:
        # 输出提示信息
        print(f"Pushing {model_name} to the hub...")
        # 将模型推送到hub
        model.push_to_hub(f"google/{model_name}")
        # 将processor推送到hub
        processor.push_to_hub(f"google/{model_name}")
# 如果代码作为独立脚本运行
if __name__ == "__main__":
    # 创建命令行参数解析器实例
    parser = argparse.ArgumentParser()

    # 添加必需的参数
    # 模型名称
    parser.add_argument(
        "--model_name",
        default="owlv2-base-patch16",
        choices=[
            "owlv2-base-patch16",
            "owlv2-base-patch16-finetuned",
            "owlv2-base-patch16-ensemble",
            "owlv2-large-patch14",
            "owlv2-large-patch14-finetuned",
            "owlv2-large-patch14-ensemble",
        ],
        type=str,
        help="Name of the Owlv2 model you'd like to convert from FLAX to PyTorch."
    )
    # Flax模型原始检查点的路径
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Flax checkpoint."
    )
    # 输出PyTorch模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory."
    )
    # 是否验证logits
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Path to the output PyTorch model directory."
    )
    # 推送模型和图像预处理器到hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数，将Flax模型转换为PyTorch模型
    convert_owlv2_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
```