# `.\models\nystromformer\convert_nystromformer_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置脚本的编码格式为 UTF-8
# Copyright 2022 The HuggingFace Inc. team.
#
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则依法分发的软件
# 在"原样"基础上分发，不带任何明示或暗示的担保或条件
# 请参阅许可证以获取详细的法律条款
"""从原始存储库转换 Nystromformer 检查点"""

# 导入参数解析库
import argparse

# 导入 PyTorch 库
import torch

# 从 transformers 库中导入 NystromformerConfig 和 NystromformerForMaskedLM 类
from transformers import NystromformerConfig, NystromformerForMaskedLM


# 定义函数：重命名键名以匹配新模型的结构
def rename_key(orig_key):
    # 如果键名中包含 "model"，则替换为 ""
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    # 如果键名中包含 "norm1"，则替换为 "attention.output.LayerNorm"
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    # 如果键名中包含 "norm2"，则替换为 "output.LayerNorm"
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    # 如果键名中包含 "norm"，则替换为 "LayerNorm"
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    # 如果键名中包含 "transformer"，则根据层编号重构为 "encoder.layer.X"
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    # 如果键名中包含 "mha.attn"，则替换为 "attention.self"
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    # 如果键名中包含 "mha"，则替换为 "attention"
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    # 如果键名中包含 "W_q"，则替换为 "self.query"
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    # 如果键名中包含 "W_k"，则替换为 "self.key"
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    # 如果键名中包含 "W_v"，则替换为 "self.value"
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    # 如果键名中包含 "ff1"，则替换为 "intermediate.dense"
    if "ff1" in orig_key:
        orig_key = orig_key.replace("ff1", "intermediate.dense")
    # 如果键名中包含 "ff2"，则替换为 "output.dense"
    if "ff2" in orig_key:
        orig_key = orig_key.replace("ff2", "output.dense")
    # 如果键名中包含 "ff"，则替换为 "output.dense"
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    # 如果键名中包含 "mlm_class"，则替换为 "cls.predictions.decoder"
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    # 如果键名中包含 "mlm"，则替换为 "cls.predictions.transform"
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    # 如果键名中不包含 "cls"，则添加前缀 "nystromformer."
    if "cls" not in orig_key:
        orig_key = "nystromformer." + orig_key

    return orig_key


# 定义函数：帮助转换检查点，调整键名以匹配新模型的结构
def convert_checkpoint_helper(config, orig_state_dict):
    # 遍历原始状态字典的键名副本
    for key in orig_state_dict.copy().keys():
        # 弹出当前键名的值
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "pooler"、"sen_class" 或 "conv.bias"，则跳过处理
        if ("pooler" in key) or ("sen_class" in key) or ("conv.bias" in key):
            continue
        else:
            # 否则，根据定义的函数重命名键名，并使用原始值更新字典
            orig_state_dict[rename_key(key)] = val

    # 将原始状态字典中的 "cls.predictions.bias" 键名设为 "cls.predictions.decoder.bias" 的值
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    # 设置 "nystromformer.embeddings.position_ids" 键名为一系列从 2 开始的数字，形状为 (1, max_position_embeddings)
    orig_state_dict["nystromformer.embeddings.position_ids"] = (
        torch.arange(config.max_position_embeddings).expand((1, -1)) + 2
    )

    return orig_state_dict
# 定义函数，用于将 Nystromformer 模型的检查点转换为 PyTorch 模型
def convert_nystromformer_checkpoint(checkpoint_path, nystromformer_config_file, pytorch_dump_path):
    # 加载原始检查点的状态字典，使用CPU进行计算
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    # 根据给定的 Nystromformer 配置文件创建配置对象
    config = NystromformerConfig.from_json_file(nystromformer_config_file)
    # 根据配置创建一个 NystromformerForMaskedLM 模型对象
    model = NystromformerForMaskedLM(config)

    # 调用辅助函数将原始状态字典转换为适合新模型的状态字典
    new_state_dict = convert_checkpoint_helper(config, orig_state_dict)

    # 加载新的模型状态字典到模型中
    model.load_state_dict(new_state_dict)
    # 设置模型为评估模式
    model.eval()
    # 将模型保存为 PyTorch 模型到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印转换成功信息，显示保存的模型路径
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to Nystromformer pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for Nystromformer model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入解析后的参数
    convert_nystromformer_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```