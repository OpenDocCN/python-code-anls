# `.\cogvideo-finetune\tools\export_sat_lora_weight.py`

```py
# 导入所需的类型和库
from typing import Any, Dict
import torch 
import argparse 
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.models.modeling_utils import load_state_dict

# 定义函数，获取状态字典，输入为一个字典，输出为一个字典
def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 初始化状态字典为输入字典
    state_dict = saved_dict
    # 如果字典中包含"model"键，更新状态字典为"model"对应的值
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    # 如果字典中包含"module"键，更新状态字典为"module"对应的值
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    # 如果字典中包含"state_dict"键，更新状态字典为"state_dict"对应的值
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    # 返回最终的状态字典
    return state_dict

# 定义LORA键重命名的字典
LORA_KEYS_RENAME = {
    'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
    'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
    'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
    'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
    'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
    'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
    'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
    'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight'
}

# 定义前缀键和相关常量
PREFIX_KEY = "model.diffusion_model."
SAT_UNIT_KEY = "layers"
LORA_PREFIX_KEY = "transformer_blocks"

# 导出LORA权重的函数，输入为检查点路径和保存目录
def export_lora_weight(ckpt_path,lora_save_directory):
    # 加载检查点并获取合并后的状态字典
    merge_original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))

    # 初始化LORA状态字典
    lora_state_dict = {}
    # 遍历合并后的状态字典的所有键
    for key in list(merge_original_state_dict.keys()):
        # 获取新键，去掉前缀
        new_key = key[len(PREFIX_KEY) :]
        # 遍历LORA键重命名字典
        for special_key, lora_keys in LORA_KEYS_RENAME.items():
            # 如果新键以特殊键结尾，则进行替换
            if new_key.endswith(special_key):
                new_key = new_key.replace(special_key, lora_keys)
                new_key = new_key.replace(SAT_UNIT_KEY, LORA_PREFIX_KEY)
                # 将替换后的键及其对应值添加到LORA状态字典
                lora_state_dict[new_key] = merge_original_state_dict[key]

    # 检查LORA状态字典的长度是否为240
    if len(lora_state_dict) != 240:
        raise ValueError("lora_state_dict length is not 240")

    # 获取LORA状态字典的所有键
    lora_state_dict.keys()

    # 调用LoraBaseMixin的写入LORA层函数，保存权重
    LoraBaseMixin.write_lora_layers(
        state_dict=lora_state_dict,
        save_directory=lora_save_directory,
        is_main_process=True,
        weight_name=None,
        save_function=None,
        safe_serialization=True
    )

# 定义获取命令行参数的函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加原始检查点路径参数
    parser.add_argument(
        "--sat_pt_path", type=str, required=True, help="Path to original sat transformer checkpoint"
    )
    # 添加LORA保存目录参数
    parser.add_argument("--lora_save_directory", type=str, required=True, help="Path where converted lora should be saved") 
    # 返回解析后的参数
    return parser.parse_args()

# 主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用导出LORA权重的函数
    export_lora_weight(args.sat_pt_path, args.lora_save_directory)
```