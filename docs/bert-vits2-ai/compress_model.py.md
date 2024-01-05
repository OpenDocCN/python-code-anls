# `d:/src/tocomm/Bert-VITS2\compress_model.py`

```
# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 从 text.symbols 模块中导入 symbols 列表
from text.symbols import symbols
# 从 torch 模块中导入所有内容
import torch

# 从 tools.log 模块中导入 logger 对象
from tools.log import logger
# 从 utils 模块中导入所有内容
import utils
# 从 models 模块中导入 SynthesizerTrn 类
from models import SynthesizerTrn
# 从 os 模块中导入所有内容
import os

# 定义一个函数，用于复制模型的状态字典
def copyStateDict(state_dict):
    # 如果状态字典的第一个键以 "module" 开头，则设置起始索引为 1，否则为 0
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    # 创建一个新的有序字典
    new_state_dict = OrderedDict()
    # 遍历原状态字典的键值对
    for k, v in state_dict.items():
        # 从键中根据起始索引和点号拆分出名称
        name = ",".join(k.split(".")[start_idx:])
        # 将新的键值对添加到新的状态字典中
        new_state_dict[name] = v
    # 返回新的状态字典
    return new_state_dict
# 定义一个函数，移除优化器
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config)

    # 创建一个合成器训练对象，传入参数为符号的长度、滤波器长度的一半加1、分段大小除以跳跃长度、说话者数量和超参数模型
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    # 使用AdamW优化器，传入参数为合成器训练对象的参数、学习率、beta值和epsilon值
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # 从输入的模型文件中加载状态字典，将其映射到 CPU 上
    state_dict_g = torch.load(input_model, map_location="cpu")
    # 复制状态字典
    new_dict_g = copyStateDict(state_dict_g)
    # 初始化一个空列表用于存储键
    keys = []
    # 遍历新状态字典中的模型项
    for k, v in new_dict_g["model"].items():
        # 如果键名中包含"enc_q"，则跳过当前循环
        if "enc_q" in k:
            continue  # noqa: E701
        # 将键名添加到列表中
        keys.append(k)

    # 根据条件判断是否将状态字典中的值转换为半精度浮点数
    new_dict_g = (
        {k: new_dict_g["model"][k].half() for k in keys}
        if ishalf
        else {k: new_dict_g["model"][k] for k in keys}
    )

    # 保存模型的新状态字典、迭代次数、优化器状态字典和学习率
    torch.save(
        {
            "model": new_dict_g,
            "iteration": 0,
            "optimizer": optim_g.state_dict(),
            "learning_rate": 0.0001,
if __name__ == "__main__":
    # 导入 argparse 模块，用于解析命令行参数
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数 -c/--config，类型为字符串，默认为"configs/config.json"
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")
    # 添加命令行参数 -i/--input，类型为字符串
    parser.add_argument("-i", "--input", type=str)
    # 添加命令行参数 -o/--output，类型为字符串，默认为None
    parser.add_argument("-o", "--output", type=str, default=None)
    # 添加命令行参数 -hf/--half，action 为 store_true，默认为 False，帮助信息为"Save as FP16"
    parser.add_argument(
        "-hf", "--half", action="store_true", default=False, help="Save as FP16"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 将命令行参数中的 output 赋值给 output 变量
    output = args.output
    if output is None:  # 如果输出文件名为空
        import os.path  # 导入 os.path 模块

        filename, ext = os.path.splitext(args.input)  # 获取输入文件的文件名和扩展名
        half = "_half" if args.half else ""  # 如果参数中有半精度参数，则设置 half 为 "_half"，否则为空字符串
        output = filename + "_release" + half + ext  # 根据输入文件名、半精度参数和扩展名生成输出文件名

    removeOptimizer(args.config, args.input, args.half, output)  # 调用 removeOptimizer 函数，传入配置、输入文件、半精度参数和输出文件名
    logger.info(f"压缩模型成功, 输出模型: {os.path.abspath(output)}")  # 记录日志，显示压缩模型成功，并输出模型的绝对路径
```