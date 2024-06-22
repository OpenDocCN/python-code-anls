# `.\transformers\models\nystromformer\convert_nystromformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# 指定文件编码为 UTF-8
# Copyright 2022 Hugging Face Inc.团队。
#
# 根据 Apache License, Version 2.0 (许可证) 授权许可。
# 除非符合许可证规定，否则不得使用本文件。
# 可在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按照“按原样”提供软件，
# 不提供任何形式的明示或暗示保证。
# 查看许可证以了解有关许可权限和
# 限制的信息。

"""将 Nystromformer 检查点从原始存储库转换。"""

import argparse

import torch

# 导入 transformers 库中的 NystromformerConfig 和 NystromformerForMaskedLM 类
from transformers import NystromformerConfig, NystromformerForMaskedLM


# 重命名原始键以适应新格式
def rename_key(orig_key):
    # 如果键中包含"model"，则删除"model."部分
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    # 如果键中包含"norm1"，则替换为"attention.output.LayerNorm"
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    # 如果键中包含"norm2"，则替换为"output.LayerNorm"
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    # 如果键中包含"norm"，则替换为"LayerNorm"
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    # 如果键中包含"transformer"，则将"transformer_{layer_num}"替换为"encoder.layer.{layer_num}"
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    # 如果键中包含"mha.attn"，则替换为"attention.self"
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    # 如果键中包含"mha"，则替换为"attention"
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    # 如果键中包含"W_q"，则替换为"self.query"
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    # 如果键中包含"W_k"，则替换为"self.key"
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    # 如果键中包含"W_v"，则替换为"self.value"
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    # 如果键中包含"ff1"，则替换为"intermediate.dense"
    if "ff1" in orig_key:
        orig_key = orig_key.replace("ff1", "intermediate.dense")
    # 如果键中包含"ff2"，则替换为"output.dense"
    if "ff2" in orig_key:
        orig_key = orig_key.replace("ff2", "output.dense")
    # 如果键中包含"ff"，则替换为"output.dense"
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    # 如果键中包含"mlm_class"，则替换为"cls.predictions.decoder"
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm_class", "cls.predictions.decoder")
    # 如果键中包含"mlm"，则替换为"cls.predictions.transform"
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    # 如果键中不包含"cls"，则在前面添加"nystromformer."
    if "cls" not in orig_key:
        orig_key = "nystromformer." + orig_key

    return orig_key


# 辅助函数，用于转换检查点
def convert_checkpoint_helper(config, orig_state_dict):
    # 遍历原始状态字典中的键
    for key in orig_state_dict.copy().keys():
        # 获取键对应的值
        val = orig_state_dict.pop(key)

        # 跳过不需要处理的键
        if ("pooler" in key) or ("sen_class" in key) or ("conv.bias" in key):
            continue
        else:
            # 将重命名的键与原始值一起存储回状态字典中
            orig_state_dict[rename_key(key)] = val

    # 设置 "cls.predictions.bias" 键的值为 "cls.predictions.decoder.bias"
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    # 设置"nystromformer.embeddings.position_ids"键的值为配置的最大位置嵌入
    orig_state_dict["nystromformer.embeddings.position_ids"] = (
        torch.arange(config.max_position_embeddings).expand((1, -1)) + 2
    )

    return orig_state_dict
# 将 Nystromformer 模型的检查点转换为 PyTorch 模型
def convert_nystromformer_checkpoint(checkpoint_path, nystromformer_config_file, pytorch_dump_path):
    # 从指定路径加载检查点的原始状态词典
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    # 从指定的 JSON 文件加载 Nystromformer 配置
    config = NystromformerConfig.from_json_file(nystromformer_config_file)
    # 根据配置创建 NystromformerForMaskedLM 模型
    model = NystromformerForMaskedLM(config)

    # 将原始状态词典转换为新的状态词典
    new_state_dict = convert_checkpoint_helper(config, orig_state_dict)

    # 加载新的状态词典到模型中
    model.load_state_dict(new_state_dict)
    # 设置模型为评估模式
    model.eval()
    # 保存模型的参数到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印转换成功的消息，并显示模型保存的路径
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
    # 调用转换函数
    convert_nystromformer_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```