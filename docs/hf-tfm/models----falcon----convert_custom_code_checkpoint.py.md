# `.\models\falcon\convert_custom_code_checkpoint.py`

```py
import json  # 导入 json 模块，用于处理 JSON 格式的数据
from argparse import ArgumentParser  # 从 argparse 模块中导入 ArgumentParser 类，用于解析命令行参数
from pathlib import Path  # 从 pathlib 模块中导入 Path 类，用于处理文件路径

"""
This script converts Falcon custom code checkpoints to modern Falcon checkpoints that use code in the Transformers
library. After conversion, performance (especially for generation) should improve and the checkpoint can be loaded
without needing trust_remote_code=True.
"""

if __name__ == "__main__":
    parser = ArgumentParser()  # 创建 ArgumentParser 实例，用于解析命令行参数
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing a custom code checkpoint to convert to a modern Falcon checkpoint.",
    )  # 添加命令行参数 "--checkpoint_dir"，表示包含自定义代码检查点的目录
    args = parser.parse_args()  # 解析命令行参数并存储到 args 变量中

    if not args.checkpoint_dir.is_dir():  # 检查指定的目录是否存在
        raise ValueError("--checkpoint_dir argument should be a directory!")  # 抛出错误提示

    if (
        not (args.checkpoint_dir / "configuration_RW.py").is_file()  # 检查是否存在 "configuration_RW.py" 文件
        or not (args.checkpoint_dir / "modelling_RW.py").is_file()  # 检查是否存在 "modelling_RW.py" 文件
    ):
        raise ValueError(
            "The model directory should contain configuration_RW.py and modelling_RW.py files! Are you sure this is a custom code checkpoint?"
        )  # 如果缺少指定的文件，则抛出错误提示

    (args.checkpoint_dir / "configuration_RW.py").unlink()  # 删除指定文件
    (args.checkpoint_dir / "modelling_RW.py").unlink()  # 删除指定文件

    config = args.checkpoint_dir / "config.json"  # 定义配置文件路径
    text = config.read_text()  # 读取配置文件内容
    text = text.replace("RWForCausalLM", "FalconForCausalLM")  # 替换文本中的内容
    text = text.replace("RefinedWebModel", "falcon")  # 替换文本中的内容
    text = text.replace("RefinedWeb", "falcon")  # 替换文本中的内容
    json_config = json.loads(text)  # 将文本解析为 JSON 格式
    del json_config["auto_map"]  # 删除指定的键值对

    if "n_head" in json_config:  # 检查指定键是否存在于 JSON 配置中
        json_config["num_attention_heads"] = json_config.pop("n_head")  # 重命名键名
    if "n_layer" in json_config:  # 检查指定键是否存在于 JSON 配置中
        json_config["num_hidden_layers"] = json_config.pop("n_layer")  # 重命名键名
    if "n_head_kv" in json_config:  # 检查指定键是否存在于 JSON 配置中
        json_config["num_kv_heads"] = json_config.pop("n_head_kv")  # 重命名键名
        json_config["new_decoder_architecture"] = True  # 添加新的键值对
    else:
        json_config["new_decoder_architecture"] = False  # 添加新的键值对
    bos_token_id = json_config.get("bos_token_id", 1)  # 获取指定键的值，如果键不存在则使用默认值
    eos_token_id = json_config.get("eos_token_id", 2)  # 获取指定键的值，如果键不存在则使用默认值
    config.unlink()  # 删除配置文件
    config.write_text(json.dumps(json_config, indent=2, sort_keys=True))  # 将修改后的 JSON 格式写入配置文件

    tokenizer_config = args.checkpoint_dir / "tokenizer_config.json"  # 定义 tokenizer 配置文件路径
    if tokenizer_config.is_file():  # 检查文件是否存在
        text = tokenizer_config.read_text()  # 读取文件内容
        json_config = json.loads(text)  # 将文本解析为 JSON 格式
        if json_config["tokenizer_class"] == "PreTrainedTokenizerFast":  # 检查指定键的值是否符合预期
            json_config["model_input_names"] = ["input_ids", "attention_mask"]  # 添加新的键值对
            tokenizer_config.unlink()  # 删除文件
            tokenizer_config.write_text(json.dumps(json_config, indent=2, sort_keys=True))  # 将修改后的 JSON 格式写入文件

    generation_config_path = args.checkpoint_dir / "generation_config.json"  # 定义生成配置文件路径
    generation_dict = {
        "_from_model_config": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "transformers_version": "4.33.0.dev0",
    }  # 创建新的字典
    generation_config_path.write_text(json.dumps(generation_dict, indent=2, sort_keys=True))  # 将生成配置写入文件
    # 打印消息，提示操作完成，建议用户确认新的检查点是否符合预期
    print("Done! Please double-check that the new checkpoint works as expected.")
```