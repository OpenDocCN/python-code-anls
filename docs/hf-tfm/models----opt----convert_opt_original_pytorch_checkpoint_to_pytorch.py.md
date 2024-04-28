# `.\transformers\models\opt\convert_opt_original_pytorch_checkpoint_to_pytorch.py`

```
# 导入所需模块和库
import argparse  # 用于解析命令行参数
from pathlib import Path  # 用于处理文件路径

import torch  # PyTorch 深度学习框架

# 从 transformers 模块中导入 OPTConfig 和 OPTModel 类
from transformers import OPTConfig, OPTModel
# 从 transformers.utils 模块中导入 logging 模块
from transformers.utils import logging

# 设置日志输出级别为 INFO
logging.set_verbosity_info()
# 获取当前模块的 logger
logger = logging.get_logger(__name__)

# 定义加载检查点的函数，参数为检查点路径
def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    # 使用 torch.load() 加载检查点，指定 map_location 为 "cpu"
    sd = torch.load(checkpoint_path, map_location="cpu")
    # 如果检查点字典中包含 "model" 键，则重新加载检查点
    if "model" in sd.keys():
        sd = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 弹出不必要的权重
    keys_to_delete = [
        "decoder.version",
        "decoder.output_projection.weight",
    ]
    for key in keys_to_delete:
        # 如果键在字典中，则弹出该键
        if key in sd:
            sd.pop(key)

    # 重命名键
    keys_to_rename = {
        "decoder.project_in_dim.weight": "decoder.project_in.weight",
        "decoder.project_out_dim.weight": "decoder.project_out.weight",
        "decoder.layer_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.layer_norm.bias": "decoder.final_layer_norm.bias",
    }
    for old_key, new_key in keys_to_rename.items():
        # 如果旧键在字典中，则将其重命名为新键
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

    # 获取字典中的所有键
    keys = list(sd.keys())
    for key in keys:
        # 如果键包含 ".qkv_proj."，则进行以下处理
        if ".qkv_proj." in key:
            # 获取值
            value = sd[key]
            # 将 QKV 分离为单独的 Q、K、V
            q_name = key.replace(".qkv_proj.", ".q_proj.")
            k_name = key.replace(".qkv_proj.", ".k_proj.")
            v_name = key.replace(".qkv_proj.", ".v_proj.")

            # 获取深度
            depth = value.shape[0]
            assert depth % 3 == 0
            # 分割成 Q、K、V
            k, v, q = torch.split(value, depth // 3, dim=0)

            # 将分割后的值赋给新键
            sd[q_name] = q
            sd[k_name] = k
            sd[v_name] = v
            # 删除原始键
            del sd[key]

    # 返回处理后的字典
    return sd


# 装饰器，用于执行不记录梯度的函数
@torch.no_grad()
# 定义将 OPT 检查点转换为 PyTorch 检查点的函数，参数为 OPT 检查点路径、转换后的 PyTorch 检查点保存路径、配置（可选）
def convert_opt_checkpoint(checkpoint_path, pytorch_dump_folder_path, config=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 加载检查点字典
    state_dict = load_checkpoint(checkpoint_path)

    # 如果配置不为空，则从预训练配置加载配置，否则使用默认配置
    if config is not None:
        config = OPTConfig.from_pretrained(config)
    else:
        config = OPTConfig()

    # 创建 OPTModel 模型对象，将其转换为半精度浮点数，并设置为评估模式
    model = OPTModel(config).half().eval()
    # 载入模型的状态字典
    model.load_state_dict(state_dict)
    
    # 检查结果，如果目录不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
# 检查脚本是否作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )
    # 添加可选参数：输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加可选参数：定义 HF 配置
    parser.add_argument("--hf_config", default=None, type=str, help="Define HF config.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 fairseq 格式的检查点转换为 PyTorch 格式
    convert_opt_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, config=args.hf_config)
```