# `.\simcse_to_huggingface.py`

```
"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
import os  # 导入操作系统相关功能模块
import json  # 导入处理JSON格式的模块


def main():
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
    parser.add_argument("--path", type=str, help="Path of SimCSE checkpoint folder")  # 添加命令行参数--path，用于指定SimCSE模型检查点的路径
    args = parser.parse_args()  # 解析命令行参数

    print("SimCSE checkpoint -> Huggingface checkpoint for {}".format(args.path))  # 打印消息，指示正在将SimCSE模型检查点转换为Huggingface风格的检查点

    state_dict = torch.load(os.path.join(args.path, "pytorch_model.bin"), map_location=torch.device("cpu"))  # 加载PyTorch模型的状态字典，从指定路径中的pytorch_model.bin文件读取，并映射到CPU设备
    new_state_dict = {}  # 创建一个新的状态字典

    for key, param in state_dict.items():
        # 将所有包含"mlp"的键名替换为"pooler"
        if "mlp" in key:
            key = key.replace("mlp", "pooler")

        # 删除键名中的"bert."或"roberta."
        if "bert." in key:
            key = key.replace("bert.", "")
        if "roberta." in key:
            key = key.replace("roberta.", "")

        new_state_dict[key] = param  # 将处理后的键值对存入新的状态字典中

    torch.save(new_state_dict, os.path.join(args.path, "pytorch_model.bin"))  # 将处理后的状态字典保存为pytorch_model.bin文件

    # 修改config.json中的架构名称
    config = json.load(open(os.path.join(args.path, "config.json")))  # 加载config.json文件内容为JSON对象
    for i in range(len(config["architectures"])):
        config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")  # 将每个架构名称中的"ForCL"替换为"Model"

    json.dump(config, open(os.path.join(args.path, "config.json"), "w"), indent=2)  # 将修改后的JSON对象重新写入config.json文件，格式化为缩进两个空格的格式


if __name__ == "__main__":
    main()
```