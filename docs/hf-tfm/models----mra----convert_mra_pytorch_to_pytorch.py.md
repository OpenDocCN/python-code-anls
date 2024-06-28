# `.\models\mra\convert_mra_pytorch_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import torch  # PyTorch 深度学习框架

# 从transformers库中导入MraConfig和MraForMaskedLM类
from transformers import MraConfig, MraForMaskedLM

# 定义函数：重命名原始键名
def rename_key(orig_key):
    # 替换包含 "model" 的键名为去除 "model." 后的内容
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    # 替换包含 "norm1" 的键名为 "attention.output.LayerNorm"
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    # 替换包含 "norm2" 的键名为 "output.LayerNorm"
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    # 替换包含 "norm" 的键名为 "LayerNorm"
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    # 替换包含 "transformer" 的键名为 "encoder.layer."
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    # 替换包含 "mha.attn" 的键名为 "attention.self"
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    # 替换包含 "mha" 的键名为 "attention"
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    # 替换包含 "W_q" 的键名为 "self.query"
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    # 替换包含 "W_k" 的键名为 "self.key"
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    # 替换包含 "W_v" 的键名为 "self.value"
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    # 替换包含 "ff.0" 的键名为 "intermediate.dense"
    if "ff.0" in orig_key:
        orig_key = orig_key.replace("ff.0", "intermediate.dense")
    # 替换包含 "ff.2" 的键名为 "output.dense"
    if "ff.2" in orig_key:
        orig_key = orig_key.replace("ff.2", "output.dense")
    # 替换包含 "ff" 的键名为 "output.dense"
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    # 替换包含 "mlm_class" 的键名为 "cls.predictions.decoder"
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    # 替换包含 "mlm" 的键名为 "cls.predictions.transform"
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    # 替换包含 "backbone.backbone.encoders" 的键名为 "encoder.layer"
    if "backbone.backbone.encoders" in orig_key:
        orig_key = orig_key.replace("backbone.backbone.encoders", "encoder.layer")
    # 如果键名中不包含 "cls"，则添加前缀 "mra."
    if "cls" not in orig_key:
        orig_key = "mra." + orig_key

    return orig_key

# 定义函数：帮助转换检查点
def convert_checkpoint_helper(max_position_embeddings, orig_state_dict):
    # 遍历原始状态字典的所有键
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名中包含 "pooler" 或 "sen_class"，则跳过当前键的处理
        if ("pooler" in key) or ("sen_class" in key):
            continue
        else:
            # 否则，使用重命名函数处理键名，并将值放回原始状态字典
            orig_state_dict[rename_key(key)] = val

    # 将 "cls.predictions.decoder.bias" 键名设置为 "cls.predictions.bias"
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    # 设置 "mra.embeddings.position_ids" 键名为一个张量，用于位置编码
    orig_state_dict["mra.embeddings.position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)) + 2
    # 返回函数当前保存的原始状态字典
    return orig_state_dict
# 定义函数，用于将 MRA 模型的检查点文件转换为 PyTorch 格式
def convert_mra_checkpoint(checkpoint_path, mra_config_file, pytorch_dump_path):
    # 使用 torch.load 加载检查点文件，并指定在 CPU 上加载模型状态字典
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    # 从 JSON 文件中加载 MRA 模型配置
    config = MraConfig.from_json_file(mra_config_file)
    # 根据配置创建 MraForMaskedLM 模型对象
    model = MraForMaskedLM(config)

    # 调用辅助函数转换原始状态字典到新状态字典
    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)

    # 使用新状态字典加载模型参数
    print(model.load_state_dict(new_state_dict))
    # 将模型设置为评估模式
    model.eval()
    # 将转换后的模型保存到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印转换成功消息，并显示保存的模型路径
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to Mra pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for Mra model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入解析后的参数
    convert_mra_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```