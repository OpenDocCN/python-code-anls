# `.\models\falcon\convert_custom_code_checkpoint.py`

```
# 导入所需的模块
import json  # 导入用于处理 JSON 格式的模块
from argparse import ArgumentParser  # 导入用于解析命令行参数的模块中的 ArgumentParser 类
from pathlib import Path  # 导入用于处理文件路径的模块中的 Path 类

"""
This script converts Falcon custom code checkpoints to modern Falcon checkpoints that use code in the Transformers
library. After conversion, performance (especially for generation) should improve and the checkpoint can be loaded
without needing trust_remote_code=True.
"""

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器
    parser = ArgumentParser()
    # 添加命令行参数：--checkpoint_dir，类型为路径，必需参数，用于指定包含自定义代码检查点的目录
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing a custom code checkpoint to convert to a modern Falcon checkpoint.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 检查指定的目录是否存在
    if not args.checkpoint_dir.is_dir():
        # 如果不存在，抛出数值错误异常
        raise ValueError("--checkpoint_dir argument should be a directory!")

    # 检查模型目录是否包含 configuration_RW.py 和 modelling_RW.py 文件
    if (
        not (args.checkpoint_dir / "configuration_RW.py").is_file()
        or not (args.checkpoint_dir / "modelling_RW.py").is_file()
    ):
        # 如果不包含，抛出数值错误异常
        raise ValueError(
            "The model directory should contain configuration_RW.py and modelling_RW.py files! Are you sure this is a custom code checkpoint?"
        )
    
    # 删除模型目录下的 configuration_RW.py 和 modelling_RW.py 文件
    (args.checkpoint_dir / "configuration_RW.py").unlink()
    (args.checkpoint_dir / "modelling_RW.py").unlink()

    # 读取并修改配置文件 config.json
    config = args.checkpoint_dir / "config.json"
    text = config.read_text()
    # 替换 JSON 文本中的特定字符串
    text = text.replace("RWForCausalLM", "FalconForCausalLM")
    text = text.replace("RefinedWebModel", "falcon")
    text = text.replace("RefinedWeb", "falcon")
    # 解析 JSON 文本为 Python 字典
    json_config = json.loads(text)
    # 删除字典中的 auto_map 键值对
    del json_config["auto_map"]

    # 根据键名替换字典中的键值对
    if "n_head" in json_config:
        json_config["num_attention_heads"] = json_config.pop("n_head")
    if "n_layer" in json_config:
        json_config["num_hidden_layers"] = json_config.pop("n_layer")
    if "n_head_kv" in json_config:
        json_config["num_kv_heads"] = json_config.pop("n_head_kv")
        json_config["new_decoder_architecture"] = True
    else:
        json_config["new_decoder_architecture"] = False
    
    # 获取字典中的 bos_token_id 和 eos_token_id，如果不存在默认为 1 和 2
    bos_token_id = json_config.get("bos_token_id", 1)
    eos_token_id = json_config.get("eos_token_id", 2)
    
    # 删除并重新写入修改后的配置文件 config.json
    config.unlink()
    config.write_text(json.dumps(json_config, indent=2, sort_keys=True))

    # 处理 tokenizer_config.json 文件
    tokenizer_config = args.checkpoint_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        text = tokenizer_config.read_text()
        json_config = json.loads(text)
        # 如果 tokenizer_class 是 PreTrainedTokenizerFast，则修改 model_input_names
        if json_config["tokenizer_class"] == "PreTrainedTokenizerFast":
            json_config["model_input_names"] = ["input_ids", "attention_mask"]
            # 删除并重新写入修改后的 tokenizer_config.json
            tokenizer_config.unlink()
            tokenizer_config.write_text(json.dumps(json_config, indent=2, sort_keys=True))

    # 处理 generation_config.json 文件
    generation_config_path = args.checkpoint_dir / "generation_config.json"
    # 创建要写入的字典
    generation_dict = {
        "_from_model_config": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "transformers_version": "4.33.0.dev0",
    }
    # 将生成配置写入 generation_config.json
    generation_config_path.write_text(json.dumps(generation_dict, indent=2, sort_keys=True))
    # 打印消息到标准输出，提示操作完成并建议用户验证新的检查点是否符合预期。
    print("Done! Please double-check that the new checkpoint works as expected.")
```