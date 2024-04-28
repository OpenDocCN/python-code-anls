# `.\transformers\models\yoso\convert_yoso_pytorch_to_pytorch.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证，只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用的法律要求或书面同意，否则以"原样"基础分发的软件，没有任何种类的保证或条件，无论是明示的还是隐含的
# 查看特定语言的限制及权限的许可证

"""从原始代码库转换 YOSO 检查点。URL：https://github.com/mlpen/YOSO"""

# 导入 argparse 模块
import argparse
# 导入 torch 模块
import torch
# 从 transformers 模块导入 YosoConfig 和 YosoForMaskedLM
from transformers import YosoConfig, YosoForMaskedLM

# 定义函数，用于重命名键
def rename_key(orig_key):
    # 如果原始键包含"model"，则替换为""
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    if "ff1" in orig_key:
        orig_key = orig_key.replace("ff1", "intermediate.dense")
    if "ff2" in orig_key:
        orig_key = orig_key.replace("ff2", "output.dense")
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    if "cls" not in orig_key:
        orig_key = "yoso." + orig_key

    return orig_key

# 定义辅助函数，用于转换检查点
def convert_checkpoint_helper(max_position_embeddings, orig_state_dict):
    # 循环遍历原始状态字典的键
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        # 如果键中包含"pooler"或"sen_class"，则跳过
        if ("pooler" in key) or ("sen_class" in key):
            continue
        # 否则，调用重命名键的函数
        else:
            orig_state_dict[rename_key(key)] = val

    # 对原始状态字典进行一些特定的修改和更新
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    orig_state_dict["yoso.embeddings.position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)) + 2

    return orig_state_dict

# 定义函数，用于转换 YOSO 检查点
def convert_yoso_checkpoint(checkpoint_path, yoso_config_file, pytorch_dump_path):
    # 从指定路径加载模型的原始状态字典，使用"cpu"作为设备参数
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    # 从JSON文件中加载配置信息，创建配置对象
    config = YosoConfig.from_json_file(yoso_config_file)
    # 使用配置对象创建YosoForMaskedLM模型
    model = YosoForMaskedLM(config)

    # 转换原始状态字典，使其适用于当前模型的最大位置嵌入
    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)

    # 加载新状态字典到模型中
    print(model.load_state_dict(new_state_dict))
    # 设置模型为评估模式
    model.eval()
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印转换成功的消息，并显示模型保存的路径
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
# 如果该脚本被作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to YOSO pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for YOSO model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 YOSO 检查点文件转换为 PyTorch 模型文件
    convert_yoso_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```