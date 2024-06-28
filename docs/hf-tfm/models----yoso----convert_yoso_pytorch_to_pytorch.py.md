# `.\models\yoso\convert_yoso_pytorch_to_pytorch.py`

```py
# 定义函数，将原始键名转换为适用于转换后模型的新键名
def rename_key(orig_key):
    # 替换以 "model." 开头的键名为空字符串，去除前缀
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    # 将 "norm1" 替换为 "attention.output.LayerNorm"
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    # 将 "norm2" 替换为 "output.LayerNorm"
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    # 将 "norm" 替换为 "LayerNorm"
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    # 将 "transformer" 替换为 "encoder.layer.<layer_num>"
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    # 将 "mha.attn" 替换为 "attention.self"
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    # 将 "mha" 替换为 "attention"
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    # 将 "W_q" 替换为 "self.query"
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    # 将 "W_k" 替换为 "self.key"
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    # 将 "W_v" 替换为 "self.value"
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    # 将 "ff1" 替换为 "intermediate.dense"
    if "ff1" in orig_key:
        orig_key = orig_key.replace("ff1", "intermediate.dense")
    # 将 "ff2" 替换为 "output.dense"
    if "ff2" in orig_key:
        orig_key = orig_key.replace("ff2", "output.dense")
    # 将 "ff" 替换为 "output.dense"
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    # 将 "mlm_class" 替换为 "cls.predictions.decoder"
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    # 将 "mlm" 替换为 "cls.predictions.transform"
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    # 如果键名不包含 "cls"，则添加 "yoso." 前缀
    if "cls" not in orig_key:
        orig_key = "yoso." + orig_key

    return orig_key


# 定义函数，将原始模型的状态字典进行转换，使其适用于 YOSO 模型
def convert_checkpoint_helper(max_position_embeddings, orig_state_dict):
    # 遍历原始状态字典的键
    for key in orig_state_dict.copy().keys():
        # 弹出键名对应的值
        val = orig_state_dict.pop(key)
        
        # 如果键名中包含 "pooler" 或 "sen_class"，则跳过不处理
        if ("pooler" in key) or ("sen_class" in key):
            continue
        else:
            # 使用定义的函数转换键名，并将其与值重新添加到状态字典中
            orig_state_dict[rename_key(key)] = val

    # 将原始状态字典中 "cls.predictions.decoder.bias" 键的值赋给 "cls.predictions.bias"
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    # 生成长度为 max_position_embeddings 的位置 ID，并赋给 "yoso.embeddings.position_ids"
    orig_state_dict["yoso.embeddings.position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)) + 2

    return orig_state_dict
    # 从指定路径加载检查点文件，并使用"cpu"作为目标设备，仅获取其中的"model_state_dict"部分
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]

    # 从 JSON 文件加载 Yoso 模型的配置信息
    config = YosoConfig.from_json_file(yoso_config_file)

    # 基于给定的配置信息创建一个 YosoForMaskedLM 模型实例
    model = YosoForMaskedLM(config)

    # 使用自定义的辅助函数将原始状态字典转换为新的状态字典
    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)

    # 载入新的状态字典到模型中，返回一个包含加载结果的字典
    print(model.load_state_dict(new_state_dict))

    # 将模型设置为评估模式，即禁用梯度计算
    model.eval()

    # 将当前模型的状态保存到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印成功转换检查点并保存模型的消息，显示保存路径
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行以下代码块
    
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters（必需的参数）
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to YOSO pytorch checkpoint."
    )
    # 添加一个命令行参数：pytorch_model_path，用于指定YOSO PyTorch检查点的路径，是必需的参数
    
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for YOSO model config.",
    )
    # 添加一个命令行参数：config_file，用于指定YOSO模型配置的JSON文件路径，是必需的参数
    
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个命令行参数：pytorch_dump_path，用于指定输出PyTorch模型的路径，是必需的参数
    
    # 解析命令行参数并将其存储到args变量中
    args = parser.parse_args()

    # 调用convert_yoso_checkpoint函数，传入解析后的参数
    convert_yoso_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```